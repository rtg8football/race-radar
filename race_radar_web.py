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
    j =
