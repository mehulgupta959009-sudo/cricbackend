"""Cricket run predictor backend. Run: uvicorn main:app --reload"""
import os, json
import pandas as pd
import re
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- CONFIG (secrets come from environment variables, never hardcode) ----
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Set the OPENROUTER_API_KEY environment variable")
MODEL = os.getenv("MODEL", "openrouter/free")  # auto-picks an available free model
SITE_URL = os.getenv("SITE_URL", "http://localhost:8000")
SITE_NAME = "Cricket Predictor"
BAT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batting_features.parquet")
# ---------------------------------------------------------------------------

BAT = pd.read_parquet(BAT_FILE).sort_values("date")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
HEADERS = {"HTTP-Referer": SITE_URL, "X-OpenRouter-Title": SITE_NAME}
cache = {}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def f(x):
    return None if pd.isna(x) else round(float(x), 1)


def has(col, text):
    return col.astype(str).str.contains(text, case=False, na=False, regex=False)


def player_rows(name):
    d = BAT[has(BAT.player_name, name)]
    if d.empty:
        raise HTTPException(404, "player not found")
    return d[d.player_name == d.player_name.mode()[0]]  # most frequent matching name


def split(s):
    return {"n": len(s), "avg": f(s.mean()), "max": int(s.max())} if len(s) else {"n": 0}


def build_context(name, opp, venue):
    d = player_rows(name)
    last = d.iloc[-1]
    r = d.target_runs
    return {
        "player": d.player_name.iloc[0],
        "last10_runs": r.tail(10).tolist(),
        "form": {"avg5": f(last.b_runs_m5), "avg10": f(last.b_runs_m10), "ewm": f(last.b_runs_ewm)},
        "career": {"avg": f(r.mean()), "sr": f(last.b_career_sr_s), "innings": len(d)},
        "vs_opp": split(d[has(d.opposition, opp)].target_runs),
        "at_venue": split(d[has(d.venue_id, venue)].target_runs),
        "pct_30plus": f((r >= 30).mean() * 100),
    }


class Req(BaseModel):
    player: str
    venue: str
    opposition: str
    day_night: str = "day"


@app.get("/players")
def players(q: str = ""):
    names = BAT.player_name.value_counts()
    names = names[names.index.str.contains(q, case=False, regex=False)] if q else names
    return names.head(20).index.tolist()


@app.get("/history/{player}")
def history(player: str, limit: int = 20):
    d = player_rows(player).tail(limit)
    d = d.assign(date=d.date.astype(str))
    return d[["date", "opposition", "venue_id", "target_runs", "balls_faced", "dismissed"]].to_dict("records")


@app.post("/predict")
def predict(req: Req):
    key = req.model_dump_json()
    if key in cache:
        return cache[key]
    ctx = build_context(req.player, req.opposition, req.venue)
    prompt = (
        f"Next match: venue={req.venue}, {req.day_night} game, opposition={req.opposition}.\n"
        f"Stats (small n = unreliable): {json.dumps(ctx)}\n"
        "Predict the batter's runs using ONLY these stats. Reply with JSON only: "
        '{"low":int,"expected":int,"high":int,"confidence":"low|medium|high","factors":[str],"reasoning":str}'
    )
    p = None
    for attempt in range(3):  # free models are flaky: retry, then fall back
        try:
            msg = client.chat.completions.create(
                extra_headers=HEADERS, model=MODEL, max_tokens=2000, temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a cricket analyst. Never invent stats. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = msg.choices[0].message.content or ""
            print(f"[attempt {attempt}] model={msg.model} raw={raw[:300]!r}")
            m = re.search(r"\{.*\}", raw, re.S)
            p = json.loads(m.group(0))
            p["low"], p["expected"], p["high"] = sorted([int(p["low"]), int(p["expected"]), int(p["high"])])
            break
        except Exception as e:
            print(f"[attempt {attempt}] failed: {e!r}")
            p = None
    if p is None:  # stats-only fallback so the API always answers
        exp = ctx["form"]["avg10"] or ctx["career"]["avg"] or 30
        p = {"low": max(0, int(exp * 0.3)), "expected": int(exp), "high": int(exp * 2.2),
             "confidence": "low", "factors": ["LLM unavailable - stats-only estimate"],
             "reasoning": "Fallback based on recent form average."}
    out = {"prediction": p, "context": ctx, "history": history(req.player, 20)}
    cache[key] = out
    return out


@app.get("/meta")
def meta():
    team_of = BAT.groupby("venue_id").team.agg(lambda x: x.mode()[0])  # most frequent team = rough "home" label
    cnt = BAT.groupby("venue_id").match_id.nunique()
    venues = [{"id": v, "label": f"{team_of[v]} home ({v[:8]})", "matches": int(cnt[v])}
              for v in cnt.sort_values(ascending=False).index[:40]]
    top = BAT.player_name.value_counts().head(60)
    return {"teams": sorted(BAT.opposition.dropna().unique().tolist()),
            "players": [{"name": n, "innings": int(c)} for n, c in top.items()],
            "venues": venues}


@app.get("/stats/{player}")
def stats(player: str):
    d = player_rows(player)
    g = lambda col: (d.groupby(col).target_runs.agg(["count", "mean"]).query("count>=3")
                     .sort_values("count", ascending=False).head(10).round(1)
                     .reset_index().rename(columns={col: "key", "count": "n", "mean": "avg"}).to_dict("records"))
    return {"by_venue": g("venue_id"), "by_opposition": g("opposition"),
            "by_season": d.groupby("season").target_runs.agg(["count", "mean"]).round(1)
                          .reset_index().rename(columns={"count": "n", "mean": "avg"}).to_dict("records")}
