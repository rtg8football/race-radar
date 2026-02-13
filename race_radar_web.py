import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Race Radar Online", layout="wide")
st.title("🏇 Race Radar Online (Simple Input)")

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
LOG_FILE = DATA_DIR / "race_log_simple.csv"

# ---------- Core math ----------
def implied_prob(odds: float) -> float:
    return 1.0 / odds if odds and odds > 1 else np.nan

def expected_value(p: float, odds: float) -> float:
    # EV per 1 unit stake: p*odds - 1
    return p * odds - 1.0

def half_kelly(p: float, odds: float) -> float:
    if odds <= 1:
        return 0.0
    b = odds - 1.0
    f = (p * (b + 1.0) - 1.0) / b
    return max(0.0, 0.5 * f)  # half-kelly safer

# ---------- Model: convert your 3 percentages into a win-prob ----------
def model_probability(form_pct: float, trainer_strike: float, jockey_strike: float) -> float:
    """
    Simple weighted model:
    - Form% weighted heavier
    - Trainer/Jockey strike give support
    Then clamp to [1%, 99%] to avoid extremes.
    """
    form = np.clip(form_pct, 0, 100) / 100.0
    trn  = np.clip(trainer_strike, 0, 100) / 100.0
    jky  = np.clip(jockey_strike, 0, 100) / 100.0

    # weights: tune anytime
    p = 0.55 * form + 0.25 * trn + 0.20 * jky
    return float(np.clip(p, 0.01, 0.99))

def radar_score(ev: float, odds: float) -> int:
    """
    Radar score is now automatic and simple:
    +2 if EV positive
    +1 if odds spread suggests "value range" (not ultra short)
    +1 if model prob is meaningfully above implied (handled via EV)
    +1 if odds are not extreme longshot (reduces variance)
    Total 0–5
    """
    score = 0
    if pd.notna(ev) and ev > 0:
        score += 2
    if odds >= 1.8 and odds <= 8:
        score += 1
    if odds < 25:
        score += 1
    # "discipline point": reward not betting ultra short prices
    if odds >= 1.4:
        score += 1
    return int(np.clip(score, 0, 5))

def decision(score: int, ev: float) -> str:
    if pd.notna(ev) and score >= 4 and ev > 0:
        return "BET"
    if pd.notna(ev) and score >= 3 and ev > -0.03:
        return "SMALL / CAUTION"
    return "SKIP"

# ---------- Input table (only the 5 columns you want) ----------
default = pd.DataFrame([{
    "Horse": "",
    "BackOdds": 0.0,
    "Form%": 50.0,
    "TrainerStrike%": 15.0,
    "JockeyStrike%": 15.0,
}])

st.caption("Fill only these columns. Everything else is calculated automatically.")

df = st.data_editor(
    default,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "BackOdds": st.column_config.NumberColumn("BackOdds", min_value=0.0, step=0.01),
        "Form%": st.column_config.NumberColumn("Form%", min_value=0.0, max_value=100.0, step=1.0),
        "TrainerStrike%": st.column_config.NumberColumn("TrainerStrike%", min_value=0.0, max_value=100.0, step=0.5),
        "JockeyStrike%": st.column_config.NumberColumn("JockeyStrike%", min_value=0.0, max_value=100.0, step=0.5),
    }
)

df = pd.DataFrame(df)
df["Horse"] = df["Horse"].astype(str).str.strip()
df = df[df["Horse"] != ""].copy()

if df.empty:
    st.info("Add at least one runner (Horse name).")
    st.stop()

# ---------- Compute ----------
df["ImpliedProb"] = df["BackOdds"].apply(implied_prob)
df["ModelProb"] = df.apply(lambda r: model_probability(r["Form%"], r["TrainerStrike%"], r["JockeyStrike%"]), axis=1)
df["EV"] = df.apply(lambda r: expected_value(r["ModelProb"], r["BackOdds"]) if r["BackOdds"] > 1 else np.nan, axis=1)
df["HalfKelly"] = df.apply(lambda r: half_kelly(r["ModelProb"], r["BackOdds"]) if r["BackOdds"] > 1 else 0.0, axis=1)
df["RadarScore"] = df.apply(lambda r: radar_score(r["EV"], r["BackOdds"]), axis=1)
df["Decision"] = df.apply(lambda r: decision(r["RadarScore"], r["EV"]), axis=1)

df = df.sort_values(["RadarScore", "EV"], ascending=[False, False]).reset_index(drop=True)

st.subheader("Results (auto)")
st.dataframe(
    df[["Horse","BackOdds","ImpliedProb","ModelProb","EV","HalfKelly","RadarScore","Decision"]],
    use_container_width=True
)

top = df.iloc[0]
st.write(
    f"**Top by Radar:** {top['Horse']} — Odds **{top['BackOdds']}**, "
    f"ModelProb **{top['ModelProb']:.1%}**, EV **{top['EV']:.3f}**, Decision **{top['Decision']}**"
)

st.divider()
st.subheader("Save / Export")

col1, col2 = st.columns(2)
with col1:
    if st.button("Save race snapshot"):
        out = df.copy()
        out["SavedAt"] = datetime.utcnow().isoformat()
        if LOG_FILE.exists():
            old = pd.read_csv(LOG_FILE)
            out = pd.concat([old, out], ignore_index=True)
        out.to_csv(LOG_FILE, index=False)
        st.success("Saved.")

with col2:
    st.download_button(
        "Download results CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"race_radar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
