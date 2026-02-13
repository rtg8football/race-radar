race_radar_web.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Race Radar", layout="wide")

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
LOG = DATA_DIR / "log.csv"

def implied_prob(odds): return 1/odds if odds>1 else np.nan
def ev(p, odds): return p*odds-1

def model_prob(form,trn,jky):
    return (0.5*form+0.25*trn+0.25*jky)/100

def radar(r):
    s=0
    if r["EV"]>0:s+=1
    if r["HypeHigh"]==0:s+=1
    if r["CrowdOverbet"]==0:s+=1
    if r["HiddenRisk"]==0:s+=1
    if r["MarketMove"]=="Steady Steam":s+=1
    return s

def decision(r):
    if r["Radar"]>=4 and r["EV"]>0:return "BET"
    if r["Radar"]>=2:return "SMALL"
    return "SKIP"

st.title("🏇 Race Radar Online")

default=pd.DataFrame([{
"Horse":"","BackOdds":2.0,"FormScore":50,
"TrainerScore":50,"JockeyScore":50,
"UseMyProb":0,"MyProb":0.0,
"HypeHigh":0,"CrowdOverbet":0,
"HiddenRisk":0,"MarketMove":"None"}])

df=st.data_editor(default,num_rows="dynamic")

df=df[df["Horse"]!=""]

if len(df)>0:
    df["ImpliedProb"]=df["BackOdds"].apply(implied_prob)
    df["ModelProb"]=df.apply(lambda r:model_prob(r["FormScore"],r["TrainerScore"],r["JockeyScore"]),axis=1)
    df["FinalProb"]=np.where(df["UseMyProb"]==1,df["MyProb"],df["ModelProb"])
    df["EV"]=df.apply(lambda r:ev(r["FinalProb"],r["BackOdds"]),axis=1)
    df["Radar"]=df.apply(radar,axis=1)
    df["Decision"]=df.apply(decision,axis=1)

    st.dataframe(df.sort_values("Radar",ascending=False))

