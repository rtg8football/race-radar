import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Race Radar Online", layout="wide")
st.title("🏇 Race Radar (Ultra Simple)")

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
LOG_FILE = DATA_DIR / "race_log.csv"

# ---------- Math ----------
def implied_prob(odds):
    return 1 / odds if odds > 1 else np.nan

def expected_value(p, odds):
    return p * odds - 1

def half_kelly(p, odds):
    if odds <= 1:
        return 0
    b = odds - 1
    f = (p * (b + 1) - 1) / b
    return max(0, f * 0.5)

# ---------- Model ----------
def model_probability(trainer_strike, jockey_strike):
    """
    Convert strike rates to win probability.
    Weighted trainer + jockey influence.
    """
    t = np.clip(trainer_strike, 0, 100) / 100
    j = np.clip(jockey_strike, 0, 100) / 100

    # weights — adjust anytime
    p = (0.6 * t + 0.4 * j)

    return np.clip(p, 0.01, 0.99)

# ---------- Radar ----------
def radar_score(ev, odds):
    score = 0

    if ev > 0:
        score += 2

    if 1.8 <= odds <= 8:
        score += 1

    if odds < 25:
        score += 1

    if odds >= 1.4:
        score += 1

    return score

def decision(score, ev):
    if score >= 4 and ev > 0:
        return "BET"
    if score >= 3:
        return "SMALL / CAUTION"
    return "SKIP"

# ---------- Input ----------
st.caption("Enter Horse, Odds, Trainer %, Jockey %. Everything else automatic.")

default = pd.DataFrame([{
    "Horse": "",
    "BackOdds": 0.0,
    "TrainerStrike%": 15.0,
    "JockeyStrike%": 15.0,
}])

df = st.data_editor(
    default,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "BackOdds": st.column_config.NumberColumn("Back Odds", min_value=0.0, step=0.01),
        "TrainerStrike%": st.column_config.NumberColumn("Trainer Strike %", min_value=0.0, max_value=100.0),
        "JockeyStrike%": st.column_config.NumberColumn("Jockey Strike %", min_value=0.0, max_value=100.0),
    }
)

df = pd.DataFrame(df)
df["Horse"] = df["Horse"].astype(str).str.strip()
df = df[df["Horse"] != ""].copy()

if df.empty:
    st.info("Add at least one horse.")
    st.stop()

# ---------- Compute ----------
df["ImpliedProb"] = df["BackOdds"].apply(implied_prob)
df["ModelProb"] = df.apply(
    lambda r: model_probability(r["TrainerStrike%"], r["JockeyStrike%"]),
    axis=1
)

df["EV"] = df.apply(lambda r: expected_value(r["ModelProb"], r["BackOdds"]), axis=1)
df["HalfKelly"] = df.apply(lambda r: half_kelly(r["ModelProb"], r["BackOdds"]), axis=1)
df["RadarScore"] = df.apply(lambda r: radar_score(r["EV"], r["BackOdds"]), axis=1)
df["Decision"] = df.apply(lambda r: decision(r["RadarScore"], r["EV"]), axis=1)

df = df.sort_values(["RadarScore", "EV"], ascending=False).reset_index(drop=True)

# ---------- Results ----------
st.subheader("Results")
st.dataframe(
    df[["Horse","BackOdds","ImpliedProb","ModelProb","EV","HalfKelly","RadarScore","Decision"]],
    use_container_width=True
)

top = df.iloc[0]
st.success(
    f"Top Pick: {top['Horse']} | Odds {top['BackOdds']} | "
    f"Model {top['ModelProb']:.1%} | EV {top['EV']:.3f} | Decision {top['Decision']}"
)

# ---------- Save ----------
st.divider()

if st.button("Save Results"):
    out = df.copy()
    out["SavedAt"] = datetime.utcnow().isoformat()

    if LOG_FILE.exists():
        old = pd.read_csv(LOG_FILE)
        out = pd.concat([old, out], ignore_index=True)

    out.to_csv(LOG_FILE, index=False)
    st.success("Saved.")

