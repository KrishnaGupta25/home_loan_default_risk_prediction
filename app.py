"""
CreditIQ - Home Loan Default Risk Prediction System
A premium, fintech-style Streamlit dashboard.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""
from src.prediction_pipeline import PredictionPipeline
import hashlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

experiment_tracker = pd.read_csv(
    BASE_DIR / "artifacts/experiment_results/experiment_tracker.csv"
)

metrics = joblib.load(
    BASE_DIR / "artifacts/experiment_results/final_metrics.pkl"
)

# FIX #3: was missing BASE_DIR, so this broke whenever the app was launched
# from a directory other than the project root.
threshold_df = pd.read_csv(
    BASE_DIR / "artifacts/experiment_results/xgboost_optimized_threshold_analysis.csv"
)


pipeline = PredictionPipeline()


def predict_dataframe(df, pipeline):
    """
    FIX #2: previously, 'Prediction' (Default / No Default) came from
    pipeline.predict()'s own internal threshold, while 'Risk Level' /
    'Loan Decision' were recomputed here using metrics["threshold"] and a
    hardcoded 0.45. If those two thresholds ever disagreed, a row could show
    Prediction = "No Default" but Risk Level = "High Risk" (or vice versa) -
    which looked like the model was behaving inconsistently.

    Now everything (Prediction, Risk Level, Loan Decision) is derived from
    the SAME probability + the SAME threshold, so they can never contradict
    each other.
    """
    predictions, probabilities = pipeline.predict(df)

    risk_levels = []
    decisions = []
    prediction_text = []

    for prob in probabilities:
        is_default = prob >= metrics["threshold"]

        prediction_text.append("Default" if is_default else "No Default")

        if prob < metrics["threshold"]:
            risk_levels.append("Low Risk")
            decisions.append("Approve")
        elif prob < 0.45:
            risk_levels.append("Medium Risk")
            decisions.append("Manual Review")
        else:
            risk_levels.append("High Risk")
            decisions.append("Reject")

    customer_ids = (
        df["SK_ID_CURR"]
        if "SK_ID_CURR" in df.columns
        else [f"CUST-{i+1}" for i in range(len(df))]
    )

    result = pd.DataFrame({

        "Customer ID": customer_ids,

        "Prediction": prediction_text,

        "Default Probability": (probabilities * 100).round(2),

        "Risk Level": risk_levels,

        "_level": [x.lower().split()[0] for x in risk_levels],

        "Loan Decision": decisions

    })

    return result
# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Home Loan Default Risk Prediction System",
    page_icon="CI",
    layout="wide",
    initial_sidebar_state="expanded",
)
 

# ============================================================================
# GLOBAL CSS - banking / fintech look (white bg, blue accents, rounded cards)
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root{
  --blue:#2563EB; --blue-dark:#1D4ED8; --blue-light:#EEF3FF; --blue-lighter:#F6F9FF;
  --navy:#102A56; --ink:#101820; --ink-soft:#475467; --ink-mute:#667085;
  --line:#E3E8F0; --line-soft:#F0F2F6; --paper:#FFFFFF; --canvas:#F7F8FB;
  --green:#12805C; --green-bg:#E7F7EF; --orange:#B45309; --orange-bg:#FEF3E2;
  --red:#B42318; --red-bg:#FDE8E7;
  /* Standard brand gradient reused across the hero, sidebar, uploader, and tables */
  --brand-gradient: linear-gradient(120deg, #102A56 0%, #1D4ED8 58%, #2563EB 100%);
}

html, body, [class*="css"], .stMarkdown, p, span, label, div { font-family:'Inter', sans-serif; }
h1,h2,h3,h4,h5 { font-family:'Manrope', sans-serif !important; letter-spacing:0; }



.stApp { background:var(--canvas); }
.block-container { padding-top:1.25rem; padding-bottom:2.75rem; max-width:1240px; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
div[data-testid="stSidebarContent"],
div[data-testid="stSidebarUserContent"]{
  background:var(--paper) !important; border-right:1px solid var(--line) !important;
}
section[data-testid="stSidebar"] > div { padding-top:1.2rem; }
.side-brand{
  display:flex; align-items:center; gap:11px; padding:4px 6px 18px 10px;
  border-bottom:1px solid var(--line); margin-bottom:14px;
}
.side-brand .mark{
  width:38px; height:38px; border-radius:8px; flex-shrink:0;
  background:var(--brand-gradient); border:1px solid rgba(255,255,255,.25);
  display:flex; align-items:center; justify-content:center; font-size:18px;
  box-shadow:none;
}
.side-brand .name{ font-family:'Manrope',sans-serif; font-weight:800; font-size:16px; color:var(--navy); line-height:1.1;}
.side-brand .sub{ font-size:10.5px; font-weight:700; color:var(--ink-mute); text-transform:uppercase; letter-spacing:.05em; margin-top:2px;}
.side-foot{
  display:flex; align-items:center; gap:10px; padding:14px 10px; margin-top:16px;
  border-top:1px solid var(--line);
}
.side-foot .avatar{
  width:30px; height:30px; border-radius:50%; background:var(--blue-light); color:var(--blue-dark);
  font-weight:800; font-size:11.5px; display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
.side-foot .fname{ font-size:12.3px; font-weight:700; color:var(--ink); }
.side-foot .frole{ font-size:10.8px; font-weight:600; color:var(--ink-mute); }

/* ---------- buttons ---------- */
.stButton>button, div[data-testid="stDownloadButton"]>button{
  background:var(--blue); color:#fff; border:none; border-radius:8px;
  font-weight:700; font-size:13.4px; padding:0.55rem 1.1rem; transition:.15s ease;
  box-shadow:none;
}
.stButton>button:hover, div[data-testid="stDownloadButton"]>button:hover{
  background:var(--blue-dark); transform:translateY(-1px); color:#fff;
}
.stButton>button:focus:not(:active){ box-shadow:0 0 0 3px rgba(37,99,235,.25) !important; }
button[kind="secondary"]{ background:#fff !important; color:var(--ink-soft) !important; border:1px solid var(--line) !important; }
button[kind="secondary"]:hover{ border-color:#C7CEDA !important; color:var(--ink) !important; }

/* ---------- file uploader ---------- */
div[data-testid="stFileUploaderDropzone"]{
  background:#F8FAFC !important;
  border:1.5px dashed #AAB7CF !important; border-radius:8px !important;
  min-height:138px;
}
div[data-testid="stFileUploaderDropzone"] *{
  color:var(--ink) !important;
}
div[data-testid="stFileUploaderDropzone"] small{
  color:var(--ink-mute) !important;
}
div[data-testid="stFileUploaderDropzone"] svg{
  fill:var(--blue-dark) !important;
}
div[data-testid="stFileUploaderDropzone"] button{
  background:var(--blue) !important; color:#fff !important; border-radius:8px !important; border:none !important;
}

/* ---------- dataframe ---------- */
div[data-testid="stDataFrame"]{
  border:1px solid var(--line); border-radius:8px; overflow:hidden;
}

/* ---------- generic card ---------- */
.ciq-card{
  background:var(--paper); border:1px solid var(--line); border-radius:8px;
  box-shadow:0 1px 2px rgba(16,24,40,.04), 0 2px 12px rgba(16,24,40,.05);
  padding:20px 20px;
}
.ciq-card{
    color:#101820;
}

.ciq-card h1,
.ciq-card h2,
.ciq-card h3,
.ciq-card h4,
.ciq-card h5,
.ciq-card p,
.ciq-card span,
.ciq-card b,
.ciq-card div{
    color:#101820 !important;
}
.section-head{ display:flex; align-items:baseline; justify-content:space-between; margin:28px 0 14px; flex-wrap:wrap; gap:6px;}
.section-head h2{ font-size:20px; font-weight:800; color:var(--ink); margin:0;}
.section-head p{ font-size:12.8px; color:var(--ink-mute); font-weight:500; margin:3px 0 0;}
.eyebrow{
  font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--blue);
  display:flex; align-items:center; gap:7px; margin-bottom:8px;
}
.eyebrow::before{ content:''; width:14px; height:2px; background:var(--blue); border-radius:2px;}

/* ---------- hero ---------- */
.hero{
  border-radius:8px; padding:40px 38px; position:relative; overflow:hidden; color:#fff;
  background:var(--brand-gradient);
  margin-bottom:6px;
}
.hero:before{
  content:''; position:absolute; inset:0;
  background:linear-gradient(90deg, rgba(255,255,255,.08), rgba(255,255,255,0));
}
.hero-badge{
  display:inline-flex; align-items:center; gap:7px; font-size:11.5px; font-weight:700;
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.22);
  padding:6px 13px 6px 11px; border-radius:100px; margin-bottom:16px; position:relative; z-index:2;
}
.hero-badge .dot{ width:6px; height:6px; border-radius:50%; background:#6EE7B7; box-shadow:0 0 0 3px rgba(110,231,183,.25);}
.hero h1{ font-size:34px; font-weight:800; line-height:1.15; margin:0 0 12px; position:relative; z-index:2; max-width:760px;}
.hero p.sub{ font-size:14.5px; line-height:1.65; color:rgba(255,255,255,.85); font-weight:500; max-width:600px; position:relative; z-index:2;}

/* ---------- KPI cards ---------- */
.kpi-card{
  background:#fff; border:1px solid var(--line); border-radius:8px; padding:17px 17px 15px;
  box-shadow:0 1px 2px rgba(16,24,40,.04);
  min-height:128px;
}
.kpi-icon{
  width:30px; height:30px; border-radius:8px; background:var(--blue-light);
  display:flex; align-items:center; justify-content:center; font-size:11.5px; margin-bottom:12px;
  color:var(--blue-dark); font-weight:800; font-family:'JetBrains Mono',monospace;
}
.kpi-label{ font-size:10.8px; font-weight:700; color:var(--ink-mute); text-transform:uppercase; letter-spacing:.04em; margin-bottom:5px;}
.kpi-value{ font-size:18.5px; font-weight:800; color:var(--ink); font-family:'Manrope',sans-serif;}
.kpi-value.mono{ font-family:'JetBrains Mono',monospace; font-size:19.5px;}
.kpi-foot{ font-size:10.6px; color:var(--ink-mute); font-weight:600; margin-top:5px;}

/* ---------- workflow rail ---------- */
.rail-wrap{ overflow-x:auto; padding:6px 2px 2px;}
.rail{ position:relative; display:flex; justify-content:space-between; min-width:900px; padding:10px 10px 0;}
.rail:before{
  content:''; position:absolute; top:33px; left:46px; right:46px; height:2px;
  background:repeating-linear-gradient(90deg,#D7DEEA 0 8px, transparent 8px 14px); z-index:1;
}
.rail-step{ position:relative; z-index:2; display:flex; flex-direction:column; align-items:center; width:126px; text-align:center;}
.rail-node{
  width:44px; height:44px; border-radius:50%; background:var(--blue-light); border:2px solid var(--blue);
  display:flex; align-items:center; justify-content:center; margin-bottom:10px; font-size:17px;
  box-shadow:0 0 0 5px rgba(37,99,235,.08);
}
.rail-num{ font-size:9.5px; font-weight:800; color:var(--blue-dark); letter-spacing:.05em; margin-bottom:3px;}
.rail-name{ font-size:12px; font-weight:700; color:var(--ink);}

/* ---------- tech stack ---------- */
.stack-chip{
  text-align:center; border-radius:8px; border:1px solid var(--line); background:#fff; padding:14px 6px 12px;
  min-height:86px;
}
.stack-chip .sc-icon{
  width:32px; height:32px; margin:0 auto 7px; border-radius:9px; background:var(--blue-lighter);
  display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:800; color:var(--blue-dark);
  font-family:'JetBrains Mono',monospace;
}
.stack-chip .sc-name{ font-size:11px; font-weight:700; color:var(--ink-soft);}

/* ---------- badges ---------- */
.badge{ display:inline-flex; align-items:center; gap:5px; font-size:11px; font-weight:700; padding:4px 10px; border-radius:100px;}
.badge:before{ content:''; width:6px; height:6px; border-radius:50%;}
.badge.low{ background:var(--green-bg); color:var(--green);} .badge.low:before{ background:var(--green);}
.badge.medium{ background:var(--orange-bg); color:var(--orange);} .badge.medium:before{ background:var(--orange);}
.badge.high{ background:var(--red-bg); color:var(--red);} .badge.high:before{ background:var(--red);}

/* ---------- stat / summary cards ---------- */
.stat-card{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:17px 18px; min-height:95px;}
.stat-card .num{ font-size:23px; font-weight:800; font-family:'Manrope',sans-serif;}
.stat-card .lbl{ font-size:11px; color:var(--ink-mute); font-weight:700; text-transform:uppercase; letter-spacing:.03em; margin-top:4px;}
.stat-blue .num{ color:var(--blue-dark);} .stat-green .num{ color:var(--green);} .stat-red .num{ color:var(--red);}

/* ---------- prediction workspace ---------- */
.prediction-grid{
  display:grid;
  grid-template-columns:1.3fr 1fr;
  gap:16px;
  align-items:start;
}
.pred-panel{
  background:#fff;
  border:1px solid var(--line);
  border-radius:8px;
  padding:20px;
  box-shadow:0 1px 2px rgba(16,24,40,.04), 0 2px 12px rgba(16,24,40,.05);
}
.pred-title{ font-size:15px; font-weight:800; color:var(--ink); margin-bottom:6px; }
.pred-copy{ font-size:12.8px; color:var(--ink-mute); line-height:1.65; font-weight:500; margin-bottom:14px; }
.pred-mini-grid{ display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:10px; }
.pred-mini{
  background:#F8FAFC;
  border:1px solid var(--line-soft);
  border-radius:8px;
  padding:13px 14px;
  min-height:86px;
}
.pred-mini .k{ color:var(--ink-mute); font-size:10.5px; font-weight:800; text-transform:uppercase; letter-spacing:.04em; margin-bottom:6px; }
.pred-mini .v{ color:var(--ink); font-size:16px; font-weight:800; font-family:'Manrope',sans-serif; line-height:1.25; }
.pred-steps{ display:grid; gap:10px; }
.pred-step{
  display:flex;
  gap:12px;
  background:#F8FAFC;
  border:1px solid var(--line-soft);
  border-radius:8px;
  padding:13px 14px;
}
.pred-step .num{
  width:28px;
  height:28px;
  border-radius:8px;
  background:var(--blue-light);
  color:var(--blue-dark);
  display:flex;
  align-items:center;
  justify-content:center;
  font-family:'JetBrains Mono',monospace;
  font-size:11px;
  font-weight:800;
  flex-shrink:0;
}
.pred-step b{ display:block; color:var(--ink); font-size:13px; margin-bottom:3px; }
.pred-step span{ color:var(--ink-mute); font-size:12.4px; line-height:1.5; font-weight:500; }
.empty-scoreboard{
  margin-top:16px;
  display:grid;
  grid-template-columns:repeat(3, minmax(0,1fr));
  gap:12px;
}
.empty-score{
  background:#fff;
  border:1px solid var(--line);
  border-radius:8px;
  padding:17px;
  min-height:116px;
}
.empty-score .label{ color:var(--ink-mute); font-size:10.8px; font-weight:800; text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px; }
.empty-score .value{ color:var(--ink); font-size:20px; font-weight:800; font-family:'Manrope',sans-serif; }
.empty-score .note{ color:var(--ink-mute); font-size:12px; line-height:1.55; margin-top:6px; }
@media (max-width: 980px){
  .prediction-grid{ grid-template-columns:1fr; }
  .empty-scoreboard{ grid-template-columns:1fr; }
}

/* ---------- best model card ---------- */
.best-model-card{
    background:#FFFFFF;
    border:1px solid #E5E7EB;
    border-radius:8px;
    padding:22px;
    color:#101820;   /* force text color */
    box-shadow:0 1px 2px rgba(16,24,40,.04), 0 2px 12px rgba(16,24,40,.05);
}

.best-model-card *{
    color:#101820 !important;
}

.bm-label{
    font-size:12px;
    font-weight:700;
    color:#2563EB !important;
    text-transform:uppercase;
    margin-bottom:10px;
}

.bm-title{
    font-size:22px;
    font-weight:800;
    color:#101820 !important;
    margin-bottom:18px;
}

.bm-row{
    display:flex;
    justify-content:space-between;
    padding:10px 0;
    border-bottom:1px solid #EEF2F7;
}

.bm-row .k{
    color:#667085 !important;
    font-weight:600;
}

.bm-row .v{
    color:#101820 !important;
    font-weight:700;
}

.bm-note{
    margin-top:18px;
    background:#F8FAFC;
    padding:14px;
    border-radius:8px;
    color:#475467 !important;
    line-height:1.6;
}

/* ---------- confusion matrix ---------- */
.cm-cell{ border-radius:8px; padding:16px 8px; text-align:center; font-family:'JetBrains Mono',monospace;}
.cm-cell .cm-val{ font-size:19px; font-weight:800;}
.cm-cell .cm-tag{ font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; margin-top:4px; opacity:.75;}
.cm-tn{ background:var(--green-bg); color:var(--green);} .cm-fp{ background:var(--orange-bg); color:var(--orange);}
.cm-fn{ background:var(--red-bg); color:var(--red);} .cm-tp{ background:var(--blue-light); color:var(--blue-dark);}
.cm-axis{ font-size:9.5px; font-weight:800; color:#000000; text-transform:uppercase; letter-spacing:.04em; text-align:center; padding-top:6px;}

/* ---------- model tags / metric rows ---------- */
.model-tag{ display:inline-block; font-size:11.5px; font-weight:700; padding:7px 13px; border-radius:8px; background:var(--canvas); border:1px solid var(--line); color:var(--ink-soft); margin:0 6px 8px 0;}
.metric-row{ display:flex; align-items:center; justify-content:space-between; padding:9px 0; border-bottom:1px solid var(--line-soft); font-size:12.6px;}
.metric-row:last-child{ border-bottom:none;}
.metric-row .m-name{ font-weight:700; color:var(--ink);}
.metric-row .m-desc{ color:var(--ink-mute); font-weight:600; font-family:'JetBrains Mono',monospace; font-size:12px;}
.check-item{
    display:flex;
    align-items:flex-start;
    gap:14px;
    margin-bottom:18px;
    color:#475467;
    line-height:1.5;
    font-size:13px;
}

.check-item b{
    color:#101820;
}

.check-item span{
    flex-shrink:0;
    margin-top:2px;
}
.check-item .tick{ color:var(--green); font-weight:800; flex-shrink:0;}

.ciq-footer{ margin-top:30px; padding:18px 4px 4px; border-top:1px solid var(--line); display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; font-size:11.8px; color:var(--ink-mute); font-weight:600;}
.ciq-footer b{ color:var(--ink); font-weight:800;}

div[data-testid="stAlert"]{
  border-radius:8px;
  border:1px solid var(--line);
  background:#FFFFFF;
  box-shadow:0 1px 2px rgba(16,24,40,.04);
}

div[data-testid="stAlert"] *{
  color:#101820 !important;
}

@media (max-width: 760px){
  .block-container{ padding-left:1rem; padding-right:1rem; }
  .hero{ padding:30px 24px; }
  .hero h1{ font-size:28px; }
  .hero p.sub{ font-size:14px; }
  .kpi-card, .stat-card, .stack-chip{ min-height:auto; }
}

/* ================= DATAFRAME TOOLBAR ================= */

/* Search box */
[data-testid="stDataFrame"] input{
    color:#101820 !important;
    background:#FFFFFF !important;
    border:1px solid #D0D5DD !important;
}

[data-testid="stDataFrame"] input::placeholder{
    color:#667085 !important;
}

/* Toolbar icons */
[data-testid="stDataFrame"] button{
    color:#101820 !important;
    background:#FFFFFF !important;
}

/* SVG icons */
[data-testid="stDataFrame"] svg{
    fill:#101820 !important;
    color:#101820 !important;
}

/* Dropdown labels */
[data-testid="stDataFrame"] label,
[data-testid="stDataFrame"] span{
    color:#101820 !important;
}

/* Header */
[data-testid="stDataFrame"] thead th{
    color:#1D4ED8 !important;
    background:#EEF3FF !important;
    font-weight:700;
}

/* Table cells */
[data-testid="stDataFrame"] tbody td{
    color:#101820 !important;
}

/* ---------- Markdown inside cards ---------- */

.ciq-card,
.ciq-card *{
    color:#101820 !important;
}

.ciq-card p{
    color:#475467 !important;
}

.ciq-card strong,
.ciq-card b{
    color:#101820 !important;
}

.ciq-card h1,
.ciq-card h2,
.ciq-card h3{
    color:#1D4ED8 !important;
}

.ciq-card li{
    color:#475467 !important;
}

section[data-testid="stFileUploaderDropzone"]{
    background:#F8FAFC !important;
    border:1.5px dashed #AAB7CF !important;
    border-radius:8px !important;
}

section[data-testid="stFileUploaderDropzone"] *,
div[data-testid="stFileUploaderDropzoneInstructions"] *{
    color:#101820 !important;
}

section[data-testid="stFileUploaderDropzone"] svg{
    fill:#1D4ED8 !important;
    color:#1D4ED8 !important;
}

section[data-testid="stFileUploaderDropzone"] button{
    background:#2563EB !important;
    color:#FFFFFF !important;
    border-radius:8px !important;
}

/* ---------- model performance page ---------- */
.perf-hero{
  background:#FFFFFF;
  border:1px solid var(--line);
  border-radius:8px;
  padding:24px 26px;
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap:18px;
  box-shadow:0 1px 2px rgba(16,24,40,.04), 0 8px 22px rgba(16,24,40,.05);
}
.perf-kicker{
  display:inline-flex;
  align-items:center;
  gap:8px;
  color:var(--blue-dark);
  background:var(--blue-light);
  border:1px solid #D8E4FF;
  border-radius:100px;
  padding:6px 11px;
  font-size:10.8px;
  font-weight:800;
  text-transform:uppercase;
  letter-spacing:.05em;
  margin-bottom:12px;
}
.perf-kicker:before{ content:''; width:7px; height:7px; border-radius:50%; background:var(--green); }
.perf-hero h1{ margin:0; color:var(--ink); font-size:28px; line-height:1.18; font-weight:800; }
.perf-hero p{ margin:10px 0 0; color:var(--ink-soft); max-width:760px; font-size:13.5px; line-height:1.65; font-weight:500; }
.perf-status{
  min-width:208px;
  border-radius:8px;
  background:linear-gradient(120deg,#102A56 0%,#1D4ED8 100%);
  color:#fff;
  padding:18px;
}
.perf-status .label{ color:rgba(255,255,255,.72); font-size:10.5px; font-weight:800; text-transform:uppercase; letter-spacing:.05em; }
.perf-status .model{ color:#fff; font-size:18px; line-height:1.25; font-weight:800; margin:7px 0 12px; font-family:'Manrope',sans-serif; }
.perf-status .score{ color:#fff; font-family:'JetBrains Mono',monospace; font-size:21px; font-weight:800; }
.perf-status .caption{ color:rgba(255,255,255,.76); font-size:11.2px; margin-top:4px; font-weight:600; }
.perf-metric{
  background:#fff;
  border:1px solid var(--line);
  border-radius:8px;
  padding:16px 17px;
  min-height:116px;
  box-shadow:0 1px 2px rgba(16,24,40,.035);
}
.perf-metric .top{ display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:12px; }
.perf-metric .label{ color:var(--ink-mute); font-size:10.5px; font-weight:800; text-transform:uppercase; letter-spacing:.04em; }
.perf-metric .pill{ font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:800; color:var(--blue-dark); background:var(--blue-light); padding:4px 7px; border-radius:100px; }
.perf-metric .value{ color:var(--ink); font-size:24px; font-weight:800; line-height:1; font-family:'Manrope',sans-serif; }
.perf-metric .foot{ color:var(--ink-mute); font-size:11.5px; line-height:1.45; font-weight:600; margin-top:9px; }
.perf-card{
  background:#fff;
  border:1px solid var(--line);
  border-radius:8px;
  padding:18px;
  box-shadow:0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.045);
}
.perf-card-head{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:12px; }
.perf-card-title{ color:var(--ink); font-size:14px; font-weight:800; margin-bottom:3px; }
.perf-card-sub{ color:var(--ink-mute); font-size:11.5px; line-height:1.45; font-weight:600; }
.perf-chip{
  white-space:nowrap;
  color:var(--green);
  background:var(--green-bg);
  border-radius:100px;
  padding:5px 9px;
  font-size:10.5px;
  font-weight:800;
}
.perf-note{
  background:#F8FAFC;
  border:1px solid var(--line-soft);
  border-radius:8px;
  padding:14px;
  color:var(--ink-soft);
  font-size:12.4px;
  line-height:1.65;
  font-weight:500;
}
.perf-note b{ color:var(--ink); }
.perf-table-wrap [data-testid="stDataFrame"]{ box-shadow:none; }
.perf-summary-row{
  display:grid;
  grid-template-columns:1fr auto;
  gap:14px;
  align-items:center;
  padding:11px 0;
  border-bottom:1px solid var(--line-soft);
}
.perf-summary-row:last-of-type{ border-bottom:none; }
.perf-summary-row .k{ color:var(--ink-mute); font-size:12px; font-weight:700; }
.perf-summary-row .v{ color:var(--ink); font-size:12.5px; font-weight:800; text-align:right; font-family:'JetBrains Mono',monospace; }
.perf-summary-row .v.model{ font-family:'Manrope',sans-serif; max-width:210px; line-height:1.25; }
.perf-table{
  width:100%;
  border-collapse:separate;
  border-spacing:0;
  overflow:hidden;
  border:1px solid var(--line);
  border-radius:8px;
  font-size:12px;
}
.perf-table th{
  background:#F8FAFC;
  color:var(--ink-mute);
  text-align:left;
  padding:10px 11px;
  font-size:10.5px;
  font-weight:800;
  text-transform:uppercase;
  letter-spacing:.04em;
  border-bottom:1px solid var(--line);
}
.perf-table td{
  background:#fff;
  color:var(--ink);
  padding:11px;
  border-bottom:1px solid var(--line-soft);
  font-weight:600;
}
.perf-table tr:last-child td{ border-bottom:none; }
.perf-table tr.production td{ background:#EEF3FF; color:#102A56; font-weight:800; }
.perf-table .num{ font-family:'JetBrains Mono',monospace; text-align:right; }
.perf-table .status{
  display:inline-flex;
  border-radius:100px;
  padding:4px 8px;
  background:#F2F4F7;
  color:var(--ink-mute);
  font-size:10px;
  font-weight:800;
}
.perf-table tr.production .status{ background:var(--green-bg); color:var(--green); }
.stRadio label, .stSlider label{ color:var(--ink) !important; font-weight:700 !important; }

/* ---------- about page ---------- */
.about-hero{
  background:#fff;
  border:1px solid var(--line);
  border-radius:8px;
  padding:30px 28px;
  display:grid;
  grid-template-columns:1.45fr .9fr;
  gap:24px;
  align-items:center;
  box-shadow:0 1px 2px rgba(16,24,40,.04), 0 8px 22px rgba(16,24,40,.05);
}
.about-hero .kicker{
  display:inline-flex;
  align-items:center;
  gap:8px;
  color:var(--blue-dark);
  background:var(--blue-light);
  border:1px solid #D8E4FF;
  border-radius:100px;
  padding:6px 11px;
  font-size:10.8px;
  font-weight:800;
  text-transform:uppercase;
  letter-spacing:.05em;
  margin-bottom:14px;
}
.about-hero .kicker:before{ content:''; width:7px; height:7px; border-radius:50%; background:var(--green); }
.about-hero h1{ margin:0; color:var(--ink); font-size:30px; line-height:1.18; font-weight:800; }
.about-hero p{ margin:12px 0 0; color:var(--ink-soft); font-size:13.8px; line-height:1.75; font-weight:500; max-width:760px; }
.about-snapshot{
  background:linear-gradient(120deg,#102A56 0%,#1D4ED8 100%);
  border-radius:8px;
  padding:20px;
  color:#fff;
}
.about-snapshot .label{ color:rgba(255,255,255,.72); font-size:10.5px; font-weight:800; text-transform:uppercase; letter-spacing:.05em; }
.about-snapshot .model{ color:#fff; font-size:18px; font-weight:800; line-height:1.28; margin:8px 0 16px; font-family:'Manrope',sans-serif; }
.about-score-grid{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
.about-score{ background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.18); border-radius:8px; padding:12px; }
.about-score .v{ color:#fff; font-family:'JetBrains Mono',monospace; font-size:19px; font-weight:800; }
.about-score .k{ color:rgba(255,255,255,.72); font-size:10px; font-weight:800; text-transform:uppercase; margin-top:4px; }
.about-card{
  background:#fff;
  border:1px solid var(--line);
  border-radius:8px;
  padding:19px;
  box-shadow:0 1px 2px rgba(16,24,40,.035);
}
.about-card-title{ color:var(--ink); font-size:14.5px; font-weight:800; margin-bottom:7px; }
.about-card-copy{ color:var(--ink-soft); font-size:12.7px; line-height:1.65; font-weight:500; margin-bottom:14px; }
.about-icon{
  width:34px;
  height:34px;
  border-radius:8px;
  background:var(--blue-light);
  color:var(--blue-dark);
  display:flex;
  align-items:center;
  justify-content:center;
  font-family:'JetBrains Mono',monospace;
  font-size:12px;
  font-weight:800;
  margin-bottom:13px;
}
.about-list{ display:grid; gap:10px; margin-top:12px; }
.about-list-item{
  display:flex;
  gap:10px;
  align-items:flex-start;
  color:var(--ink-soft);
  font-size:12.6px;
  line-height:1.5;
  font-weight:500;
}
.about-list-item .tick{
  width:20px;
  height:20px;
  border-radius:50%;
  background:var(--green-bg);
  color:var(--green);
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:11px;
  font-weight:900;
  flex-shrink:0;
}
.about-mini-grid{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
.about-mini{
  background:#F8FAFC;
  border:1px solid var(--line-soft);
  border-radius:8px;
  padding:13px;
}
.about-mini .k{ color:var(--ink-mute); font-size:10.5px; font-weight:800; text-transform:uppercase; letter-spacing:.04em; margin-bottom:5px; }
.about-mini .v{ color:var(--ink); font-size:14px; font-weight:800; line-height:1.3; }
.about-chip-wrap{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
.about-chip{
  display:inline-flex;
  align-items:center;
  border:1px solid var(--line);
  background:#F8FAFC;
  color:var(--ink-soft);
  border-radius:100px;
  padding:7px 10px;
  font-size:11.5px;
  font-weight:700;
}
.about-impact{
  background:#fff;
  border:1px solid var(--line);
  border-radius:8px;
  padding:22px;
  display:grid;
  grid-template-columns:.95fr 1.4fr;
  gap:18px;
  align-items:start;
  box-shadow:0 1px 2px rgba(16,24,40,.04), 0 8px 22px rgba(16,24,40,.04);
}
.about-impact h3{ margin:0 0 8px; color:var(--ink); font-size:18px; font-weight:800; }
.about-impact p{ margin:0; color:var(--ink-soft); font-size:13px; line-height:1.65; font-weight:500; }
.impact-grid{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
.impact-cell{ background:#F8FAFC; border:1px solid var(--line-soft); border-radius:8px; padding:14px; }
.impact-cell .v{ color:var(--blue-dark); font-size:17px; font-weight:800; font-family:'Manrope',sans-serif; }
.impact-cell .k{ color:var(--ink-mute); font-size:11px; line-height:1.45; font-weight:700; margin-top:5px; }
@media (max-width: 920px){
  .about-hero, .about-impact{ grid-template-columns:1fr; }
  .impact-grid, .about-mini-grid{ grid-template-columns:1fr; }
}
@media (max-width: 860px){
  .perf-hero{ flex-direction:column; }
  .perf-status{ width:100%; min-width:0; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - brand + navigation
# ============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="mark">CI</div>
            <div>
                <div class="name">CreditIQ</div>
                <div class="sub">Risk Analytics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title=None,
        options=["Home", "Prediction", "Model Performance", "About"],
        icons=["house", "cloud-arrow-up", "bar-chart-line", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color":"#FFFFFF"},
            "icon": {"color": "#2563EB", "font-size": "15px"},
            "nav-link": {
                "font-family": "Inter, sans-serif",
                "font-size": "13.6px",
                "font-weight": "600",
                "text-align": "left",
                "margin": "2px 0",
                "border-radius": "8px",
                "color": "#475467",
                "padding": "10px 12px",
            },
            "nav-link-selected": {
                "background-color": "#1F4BC3",
                "color": "#FFFFFF",
                "font-weight": "700",
            },
        },
    )

    st.markdown(
        """
        <div class="side-foot">
            <div class="avatar">K</div>
            <div>
                <div class="fname">Krishna</div>
                <div class="frole">Data Scientist</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# HELPERS
# ============================================================================
def section_head(title, subtitle):
    st.markdown(
        f'<div class="section-head"><div><h2>{title}</h2>'
        f'<p>{subtitle}</p></div></div>',
        unsafe_allow_html=True,
    )


def kpi_card(icon, label, value, sub, mono=False):
    mono_cls = "mono" if mono else ""
    return (
        f'<div class="kpi-card"><div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {mono_cls}">{value}</div>'
        f'<div class="kpi-foot">{sub}</div></div>'
    )


PLOTLY_CONFIG = {"displayModeBar": False}
CHART_FONT = dict(family="Inter", size=12, color="#000000")
CHART_AXIS_STYLE = dict(
    title_font=dict(color="#000000", size=12),
    tickfont=dict(color="#000000", size=11),
    linecolor="#000000",
    zerolinecolor="#000000",
)


def make_chart_text_black(fig):
    fig.update_layout(
        font=CHART_FONT,
        legend=dict(font=dict(color="#000000", size=11)),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#D0D5DD",
            font=dict(color="#000000", size=12, family="Inter"),
        ),
    )
    fig.update_xaxes(**CHART_AXIS_STYLE)
    fig.update_yaxes(**CHART_AXIS_STYLE)
    return fig
BLUE = "#2563EB"
BLUE_LIGHT = "#C7D9FB"
GREEN = "#12805C"
ORANGE = "#F59E0B"

# ============================================================================
# PAGE: HOME
# ============================================================================
if selected == "Home":
    best_model = metrics["best_model"]
    roc_auc = metrics["roc_auc"]

    st.markdown(
        f'<div class="hero">'
        f'<div class="hero-badge"><span class="dot"></span>'
        f'{best_model} &middot; ROC-AUC {roc_auc:.3f}</div>'
        f'<h1>Home Loan Default Risk Prediction System</h1>'
        f'<p class="sub">A professional credit-risk dashboard for scoring applicant default probability, reviewing model performance, and exporting portfolio-level decisions.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, _ = st.columns([1.3, 1.5, 3])
    with c1:
        if st.button("Run Batch Prediction", use_container_width=True):
            st.session_state["_nav_hint"] = "Prediction"
            st.info("Open **Prediction** from the sidebar to upload a CSV and score applicants.")
    with c2:
        if st.button("View Model Performance", use_container_width=True, type="secondary"):
            st.info("Open **Model Performance** from the sidebar for the full evaluation breakdown.")

    section_head("Key metrics", "Snapshot of the production model and underlying dataset.")
    cols = st.columns(6)
    kpis = [
    (
        "M",
        "Best Model",
        metrics["best_model"],
        "Production Model",
        False
    ),

    (
        "AUC",
        "ROC-AUC",
        f'{metrics["roc_auc"]:.3f}',
        "Validation Set",
        True
    ),

    (
        "F1",
        "F1 Score",
        f'{metrics["f1"]:.3f}',
        "Default Class",
        True
    ),

    (
        "DS",
        "Dataset",
        "Home Credit",
        "Default Risk",
        False
    ),

    (
        "FE",
        "Features",
        str(metrics["features"]),
        "Engineered Features",
        True
    ),

    (
        "ML",
        "Models Compared",
        str(len(experiment_tracker)),
        "Candidate Models",
        True
    ),
]
    for col, (icon, label, value, sub, mono) in zip(cols, kpis):
        with col:
            st.markdown(kpi_card(icon, label, value, sub, mono), unsafe_allow_html=True)

    section_head("Project workflow", "End-to-end pipeline from raw bureau data to a deployed scoring service.")
    steps = [
        ("01", "01", "Data Collection"),
        ("02", "02", "Data Cleaning"),
        ("03", "03", "Feature Engineering"),
        ("04", "04", "Model Training"),
        ("05", "05", "Model Evaluation"),
        ("06", "06", "Prediction"),
        ("07", "07", "Deployment"),
    ]
    steps_html = "".join(
        f'<div class="rail-step"><div class="rail-node">{icon}</div>'
        f'<div class="rail-num">STEP {num}</div><div class="rail-name">{name}</div></div>'
        for num, icon, name in steps
    )
    st.markdown(
        f'<div class="ciq-card rail-wrap"><div class="rail">{steps_html}</div></div>',
        unsafe_allow_html=True,
    )

    section_head("Technology stack", "Libraries and frameworks used across the modeling pipeline.")
    stack = [("Py", "Python"), ("sk", "Scikit-learn"), ("XGB", "XGBoost"), ("LGB", "LightGBM"),
             ("pd", "Pandas"), ("np", "NumPy"), ("plt", "Matplotlib"), ("sns", "Seaborn"), ("st", "Streamlit")]
    cols = st.columns(9)
    for col, (abbr, name) in zip(cols, stack):
        with col:
            st.markdown(
                f'<div class="stack-chip"><div class="sc-icon">{abbr}</div><div class="sc-name">{name}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="ciq-footer"><div>Developed by <b>Krishna</b></div>'
        '<div>Home Credit Default Risk Dataset &middot; Model v2.3</div></div>',
        unsafe_allow_html=True,
    )

# ============================================================================
# PAGE: PREDICTION
# ============================================================================
elif selected == "Prediction":
    section_head("Batch prediction", "Upload a CSV of applicant records to score default risk across your full portfolio.")

    if "pred_df" not in st.session_state:
        st.session_state["pred_df"] = None
    if "pred_results" not in st.session_state:
        st.session_state["pred_results"] = None

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="ciq-card">', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload applicant CSV",
            type=["csv"],
            label_visibility="collapsed",
            help=f"Accepted format: CSV - Applicant feature set ({metrics['features']} columns expected)",
        )
        st.markdown(
            '<div style="font-size:12.5px;color:#8A94A6;font-weight:600;margin-top:8px;">'
            f'Accepted format: CSV &nbsp;&middot;&nbsp; Applicant feature set ({metrics["features"]} columns expected)</div>',
            unsafe_allow_html=True,
        )

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.session_state["pred_df"] = df
                st.session_state["pred_file_name"] = uploaded.name
            except Exception:
                st.error("Could not parse this CSV file. Please check the format and try again.")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="pred-panel" style="margin-top:16px;">
                <div class="pred-title">Input readiness checklist</div>
                <div class="pred-copy">
                    Use the included sample file or any applicant export with the same schema.
                    The model expects complete application, bureau and previous-credit features.
                </div>
                <div class="pred-mini-grid">
                    <div class="pred-mini">
                        <div class="k">Expected features</div>
                        <div class="v">{metrics["features"]}</div>
                    </div>
                    <div class="pred-mini">
                        <div class="k">Accepted type</div>
                        <div class="v">CSV</div>
                    </div>
                    <div class="pred-mini">
                        <div class="k">Sample file</div>
                        <div class="v">sample_input.csv</div>
                    </div>
                    <div class="pred-mini">
                        <div class="k">Required ID</div>
                        <div class="v">SK_ID_CURR</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="pred-panel">
                <div class="pred-title">How scoring works</div>
                <div class="pred-copy">
                    The uploaded portfolio is scored row by row, then summarized into business-friendly risk decisions.
                </div>
                <div class="pred-steps">
                    <div class="pred-step">
                        <div class="num">01</div>
                        <div><b>Validate applicant records</b><span>Read the CSV and preview applicant rows before running the model.</span></div>
                    </div>
                    <div class="pred-step">
                        <div class="num">02</div>
                        <div><b>Estimate default probability</b><span>{metrics["best_model"]} returns a probability score for every applicant.</span></div>
                    </div>
                    <div class="pred-step">
                        <div class="num">03</div>
                        <div><b>Classify lending risk</b><span>Scores above {metrics["threshold"]:.2f} are escalated for risk review.</span></div>
                    </div>
                    <div class="pred-step">
                        <div class="num">04</div>
                        <div><b>Export decisions</b><span>Download predictions with probability, risk level and recommendation.</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    df = st.session_state["pred_df"]

    if df is None:
        st.markdown(
            f"""
            <div class="empty-scoreboard">
                <div class="empty-score">
                    <div class="label">Current batch</div>
                    <div class="value">Waiting for CSV</div>
                    <div class="note">Upload applicant records to unlock preview, scoring and downloadable outputs.</div>
                </div>
                <div class="empty-score">
                    <div class="label">Production model</div>
                    <div class="value">{metrics["best_model"]}</div>
                    <div class="note">Configured with an optimized threshold for credit-risk screening.</div>
                </div>
                <div class="empty-score">
                    <div class="label">Output fields</div>
                    <div class="value">4 columns</div>
                    <div class="note">Prediction, default probability, risk level and loan recommendation.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        fname = st.session_state.get("pred_file_name", "the uploaded file")
        section_head("Dataset preview", f"Showing the first 10 rows of {fname} - {df.shape[0]} rows - {df.shape[1]} columns.")
        st.dataframe(df.head(10), use_container_width=True, height=340)

        b1, b2, _ = st.columns([1, 1, 4])
        with b1:
            predict_clicked = st.button("Predict Loan Risk", use_container_width=True)
        with b2:
            if st.button("Clear", use_container_width=True, type="secondary"):
                st.session_state["pred_df"] = None
                st.session_state["pred_results"] = None
                st.rerun()

        if predict_clicked:
            pipeline = PredictionPipeline()
            st.session_state["pred_results"] = predict_dataframe(df, pipeline)

        results = st.session_state["pred_results"]
        if results is not None:
            section_head("Prediction summary", "Portfolio-level risk breakdown for this batch.")

            total = len(results)
            high = int((results["_level"] == "high").sum())
            low = int((results["_level"] == "low").sum())
            avg_prob = results["Default Probability"].mean()

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(f'<div class="stat-card stat-blue"><div class="num">{total:,}</div><div class="lbl">Total Applicants</div></div>', unsafe_allow_html=True)
            with s2:
                st.markdown(f'<div class="stat-card stat-red"><div class="num">{high:,}</div><div class="lbl">High Risk Customers</div></div>', unsafe_allow_html=True)
            with s3:
                st.markdown(f'<div class="stat-card stat-green"><div class="num">{low:,}</div><div class="lbl">Low Risk Customers</div></div>', unsafe_allow_html=True)
            with s4:
                st.markdown(f'<div class="stat-card stat-blue"><div class="num">{avg_prob:.1f}%</div><div class="lbl">Avg. Default Probability</div></div>', unsafe_allow_html=True)

            section_head("Scored applicants", "Per-customer default probability and recommended loan decision.")

            top100 = results.head(100)

            display_df = top100.drop(columns=["_level"])

            styled = display_df.style.apply(
                lambda s: [
                    {
                        "low": "background-color:#E7F7EF;color:#12805C;font-weight:700;",
                        "medium": "background-color:#FEF3E2;color:#B45309;font-weight:700;",
                        "high": "background-color:#FDE8E7;color:#B42318;font-weight:700;",
                    }[lvl]
                    if s.name == "Risk Level" else ""
                    for lvl in top100["_level"]
                ],
                axis=0,
            ).format({"Default Probability": "{:.1f}%"})

            st.dataframe(
                styled,
                use_container_width=True,
                height=420,
            )

            csv_bytes = (
              results
              .drop(columns=["_level"])
              .to_csv(index=False)
              .encode("utf-8")
)
            st.download_button(
                "Download Predictions CSV",
                data=csv_bytes,
                file_name=f"{metrics['best_model']}_predictions.csv",
                mime="text/csv",
            )

# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================
elif selected == "Model Performance":
    best_model = str(metrics.get("best_model", "N/A"))
    threshold = float(metrics.get("threshold", 0))
    roc_auc = float(metrics.get("roc_auc", 0))
    accuracy = float(metrics.get("accuracy", 0))
    precision = float(metrics.get("precision", 0))
    recall = float(metrics.get("recall", 0))
    f1 = float(metrics.get("f1", 0))
    features = int(metrics.get("features", 0))
    validation_size = int(metrics.get("validation_size", 0))

    model_data = experiment_tracker.copy().rename(columns={"ROC_AUC": "ROC-AUC"})
    metric_options = ["ROC-AUC", "F1", "Recall", "Precision", "Accuracy"]
    palette = ["#1D4ED8", "#12805C", "#F59E0B", "#7C3AED", "#0891B2", "#64748B"]

    st.markdown(
        f"""
        <div class="perf-hero">
            <div>
                <div class="perf-kicker">Validation Performance</div>
                <h1>Model Performance Review</h1>
                <p>
                    Comparative evaluation of candidate credit-risk classifiers, threshold behavior,
                    and production operating metrics for lending decision support.
                </p>
            </div>
            <div class="perf-status">
                <div class="label">Production Model</div>
                <div class="model">{best_model}</div>
                <div class="score">{roc_auc:.3f}</div>
                <div class="caption">ROC-AUC on validation data</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    metric_cards = [
        (m1, "Decision Threshold", f"{threshold:.2f}", "OPT", "Operating point for default classification"),
        (m2, "Recall", f"{recall:.3f}", "DEF", "Coverage of actual default cases"),
        (m3, "Precision", f"{precision:.3f}", "RISK", "Share of flagged applicants that default"),
        (m4, "Validation Size", f"{validation_size:,}", "N", f"{features} engineered features used"),
    ]
    for col, label, value, pill, foot in metric_cards:
        with col:
            st.markdown(
                f"""
                <div class="perf-metric">
                    <div class="top"><div class="label">{label}</div><div class="pill">{pill}</div></div>
                    <div class="value">{value}</div>
                    <div class="foot">{foot}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    if "perf_rank_metric" not in st.session_state:
        st.session_state["perf_rank_metric"] = "ROC-AUC"

    control_1, control_2 = st.columns([2.6, 1])
    with control_1:
        st.markdown(
            '<div style="font-size:12px;font-weight:800;color:#101820;margin-bottom:8px;">Rank models by</div>',
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(len(metric_options))
        for metric_col, option in zip(metric_cols, metric_options):
            with metric_col:
                is_active = st.session_state["perf_rank_metric"] == option
                if st.button(
                    option,
                    key=f"rank_{option}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["perf_rank_metric"] = option
        rank_metric = st.session_state["perf_rank_metric"]
    with control_2:
        top_n = st.slider(
            "Models shown",
            min_value=5,
            max_value=min(12, len(model_data)),
            value=min(8, len(model_data)),
            step=1,
            key="perf_top_n",
        )

    model_data = model_data.sort_values(rank_metric, ascending=False).reset_index(drop=True)
    model_data.insert(0, "#", range(1, len(model_data) + 1))
    model_data["best"] = model_data["Model"].astype(str).str.lower().eq(best_model.lower())
    if not model_data["best"].any():
        model_data.loc[0, "best"] = True

    top_models = model_data.head(top_n).copy()
    top_colors = [
        "#12805C" if is_best else palette[i % len(palette)]
        for i, is_best in enumerate(top_models["best"])
    ]

    table_col, summary_col = st.columns([1.45, 1])
    with table_col:
        st.markdown(
            f"""
            <div class="perf-card perf-table-wrap">
                <div class="perf-card-head">
                    <div>
                        <div class="perf-card-title">Candidate Model Leaderboard</div>
                        <div class="perf-card-sub">Sorted by {rank_metric}; production model highlighted for audit readability.</div>
                    </div>
                    <div class="perf-chip">{len(model_data)} experiments</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def escape_html(value):
            return (
                str(value)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )

        table_rows = []
        for _, row in model_data.head(top_n).iterrows():
            row_class = "production" if row["best"] else ""
            status = "Production" if row["best"] else "Candidate"
            table_rows.append(
                f'<tr class="{row_class}">'
                f'<td class="num">{int(row["#"])}</td>'
                f'<td>{escape_html(row["Model"])}</td>'
                f'<td><span class="status">{status}</span></td>'
                f'<td class="num">{row["ROC-AUC"]:.3f}</td>'
                f'<td class="num">{row["F1"]:.3f}</td>'
                f'<td class="num">{row["Recall"]:.3f}</td>'
                f'<td class="num">{row["Precision"]:.3f}</td>'
                f'</tr>'
            )
        st.markdown(
            '<table class="perf-table">'
            '<thead><tr>'
            '<th>#</th><th>Model</th><th>Status</th><th>ROC-AUC</th><th>F1</th><th>Recall</th><th>Precision</th>'
            '</tr></thead>'
            f'<tbody>{"".join(table_rows)}</tbody></table>',
            unsafe_allow_html=True,
        )

    with summary_col:
        st.markdown(
            f"""
            <div class="perf-card">
                <div class="perf-card-head">
                    <div>
                        <div class="perf-card-title">Production Readiness Summary</div>
                        <div class="perf-card-sub">Current model configuration and business interpretation.</div>
                    </div>
                    <div class="perf-chip">Deployed</div>
                </div>
                <div class="perf-summary-row"><span class="k">Model</span><span class="v model">{best_model}</span></div>
                <div class="perf-summary-row"><span class="k">ROC-AUC</span><span class="v">{roc_auc:.3f}</span></div>
                <div class="perf-summary-row"><span class="k">Accuracy</span><span class="v">{accuracy:.3f}</span></div>
                <div class="perf-summary-row"><span class="k">F1 Score</span><span class="v">{f1:.3f}</span></div>
                <div class="perf-summary-row"><span class="k">Threshold</span><span class="v">{threshold:.2f}</span></div>
                <div class="perf-summary-row"><span class="k">Features</span><span class="v">{features}</span></div>
                <div class="perf-note" style="margin-top:14px;">
                    <b>Business reading:</b> the selected threshold favors stronger default detection
                    while preserving high overall discrimination. Applications above the threshold
                    should move into enhanced review or rejection workflows.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    chart_1, chart_2 = st.columns([1.15, 1])
    with chart_1:
        st.markdown(
            f"""
            <div class="perf-card">
                <div class="perf-card-head">
                    <div>
                        <div class="perf-card-title">{rank_metric} Ranking</div>
                        <div class="perf-card-sub">Top {top_n} models by the selected evaluation metric.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig = go.Figure(
            go.Bar(
                x=top_models[rank_metric],
                y=top_models["Model"],
                orientation="h",
                text=[f"{v:.3f}" for v in top_models[rank_metric]],
                textposition="outside",
                textfont=dict(color="#000000", size=11, family="Inter"),
                marker=dict(color=top_colors, line=dict(color="white", width=1)),
                hovertemplate=f"<b>%{{y}}</b><br>{rank_metric}: %{{x:.3f}}<extra></extra>",
            )
        )
        fig.update_layout(
            height=390,
            margin=dict(l=10, r=38, t=8, b=20),
            xaxis=dict(title=rank_metric, gridcolor="#EEF1F6", range=[0, min(1, max(top_models[rank_metric]) + 0.08)]),
            yaxis=dict(autorange="reversed", title=None),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=CHART_FONT,
            showlegend=False,
        )
        make_chart_text_black(fig)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with chart_2:
        st.markdown(
            """
            <div class="perf-card">
                <div class="perf-card-head">
                    <div>
                        <div class="perf-card-title">Precision and Recall Map</div>
                        <div class="perf-card-sub">Each point represents a candidate model; hover over a marker to view the full algorithm name.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        marker_labels = [f"M{int(rank)}" for rank in top_models["#"]]
        fig2 = go.Figure(
            go.Scatter(
                x=top_models["Precision"],
                y=top_models["Recall"],
                mode="markers+text",
                text=marker_labels,
                textposition="middle center",
                textfont=dict(color="#000000", size=10, family="Inter"),
                marker=dict(
                    size=[34 if b else 26 for b in top_models["best"]],
                    color=top_colors,
                    opacity=0.92,
                    line=dict(color="white", width=2),
                ),
                customdata=np.stack([top_models["Model"], top_models["F1"], marker_labels], axis=-1),
                hovertemplate="<b>%{customdata[2]} - %{customdata[0]}</b><br>Precision: %{x:.3f}<br>Recall: %{y:.3f}<br>F1: %{customdata[1]:.3f}<extra></extra>",
            )
        )
        fig2.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=8, b=20),
            xaxis=dict(title="Precision", range=[0, 1], gridcolor="#EEF1F6"),
            yaxis=dict(title="Recall", range=[0, 1], gridcolor="#EEF1F6"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=CHART_FONT,
            showlegend=False,
        )
        make_chart_text_black(fig2)
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

        legend_items = []
        for label, (_, row) in zip(marker_labels, top_models.iterrows()):
            status = "Production" if row["best"] else "Candidate"
            legend_items.append(
                f'<div class="perf-summary-row">'
                f'<span class="k">{label} &middot; {status}</span>'
                f'<span class="v model">{escape_html(row["Model"])}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div class="perf-note" style="margin-top:10px;">{"".join(legend_items)}</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    lower_1, lower_2 = st.columns([1.15, 1])
    with lower_1:
        st.markdown(
            """
            <div class="perf-card">
                <div class="perf-card-head">
                    <div>
                        <div class="perf-card-title">Threshold Optimization Curve</div>
                        <div class="perf-card-sub">Precision, recall, and F1 behavior across decision thresholds.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=threshold_df["Threshold"], y=threshold_df["Precision"], mode="lines+markers", name="Precision", line=dict(color=ORANGE, width=3)))
        fig3.add_trace(go.Scatter(x=threshold_df["Threshold"], y=threshold_df["Recall"], mode="lines+markers", name="Recall", line=dict(color=BLUE, width=3)))
        fig3.add_trace(go.Scatter(x=threshold_df["Threshold"], y=threshold_df["F1 Score"], mode="lines+markers", name="F1 Score", line=dict(color=GREEN, width=3)))
        fig3.add_vline(x=threshold, line_dash="dash", line_color="#B42318", line_width=2)
        fig3.add_trace(
            go.Scatter(
                x=[threshold],
                y=[f1],
                mode="markers",
                marker=dict(size=13, color="#B42318", line=dict(color="white", width=2)),
                name="Production threshold",
                hovertemplate=f"Threshold: {threshold:.2f}<br>F1 Score: {f1:.3f}<extra></extra>",
            )
        )
        fig3.add_annotation(
            x=threshold,
            y=f1,
            text=f"Production threshold<br>{threshold:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=35,
            ay=-42,
            bgcolor="white",
            bordercolor="#B42318",
            borderwidth=1,
            font=dict(size=10, color="#000000"),
        )
        fig3.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=8, b=20),
            xaxis=dict(title="Decision Threshold", gridcolor="#EEF1F6"),
            yaxis=dict(title="Score", range=[0, 1], gridcolor="#EEF1F6"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=CHART_FONT,
        )
        make_chart_text_black(fig3)
        st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)

    with lower_2:
        st.markdown(
            f"""
            <div class="perf-card">
                <div class="perf-card-head">
                    <div>
                        <div class="perf-card-title">Confusion Matrix</div>
                        <div class="perf-card-sub">{best_model} &middot; Threshold {threshold:.2f}</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        gcol1, gcol2, gcol3 = st.columns([0.85, 1, 1])
        with gcol1:
            st.markdown(
                '<div style="height:45px;"></div>'
                '<div class="cm-axis">Actual<br>No Default</div>'
                '<div style="height:14px;"></div>'
                '<div class="cm-axis">Actual<br>Default</div>',
                unsafe_allow_html=True,
            )
        with gcol2:
            st.markdown(
                f'<div class="cm-axis">Pred. No Default</div>'
                f'<div class="cm-cell cm-tn"><div class="cm-val">{metrics["tn"]:,}</div><div class="cm-tag">True Negative</div></div>'
                f'<div style="height:10px;"></div>'
                f'<div class="cm-cell cm-fn"><div class="cm-val">{metrics["fn"]:,}</div><div class="cm-tag">False Negative</div></div>',
                unsafe_allow_html=True,
            )
        with gcol3:
            st.markdown(
                f'<div class="cm-axis">Pred. Default</div>'
                f'<div class="cm-cell cm-fp"><div class="cm-val">{metrics["fp"]:,}</div><div class="cm-tag">False Positive</div></div>'
                f'<div style="height:10px;"></div>'
                f'<div class="cm-cell cm-tp"><div class="cm-val">{metrics["tp"]:,}</div><div class="cm-tag">True Positive</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""
                <div class="perf-note" style="margin-top:16px;">
                    <b>Risk-control focus:</b> false negatives are the highest-risk error type
                    because they represent applicants predicted as safe who later default.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif selected == "About":

    models_html = "".join(
        f'<span class="about-chip">{m}</span>'
        for m in [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
            "LightGBM",
        ]
    )

    metrics_html = "".join(
        f'<span class="about-chip">{m}</span>'
        for m in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    )

    st.markdown(
        f"""
        <div class="about-hero">
            <div>
                <div class="kicker">Credit Risk Intelligence</div>
                <h1>About the Home Loan Default Risk Prediction System</h1>
                <p>
                    CreditIQ is an end-to-end machine learning application designed to help
                    lenders estimate home-loan default probability before approval. It combines
                    engineered borrower features, model comparison, threshold optimization, and
                    an interactive Streamlit interface for faster and more consistent risk review.
                </p>
            </div>
            <div class="about-snapshot">
                <div class="label">Production Snapshot</div>
                <div class="model">{metrics["best_model"]}</div>
                <div class="about-score-grid">
                    <div class="about-score"><div class="v">{metrics["roc_auc"]:.3f}</div><div class="k">ROC-AUC</div></div>
                    <div class="about-score"><div class="v">{metrics["threshold"]:.2f}</div><div class="k">Threshold</div></div>
                    <div class="about-score"><div class="v">{metrics["features"]}</div><div class="k">Features</div></div>
                    <div class="about-score"><div class="v">{metrics["validation_size"]:,}</div><div class="k">Validation Rows</div></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="about-card">
                <div class="about-icon">01</div>
                <div class="about-card-title">Problem Context</div>
                <div class="about-card-copy">
                    Manual underwriting can be slow, subjective, and limited when applicant
                    histories contain many nonlinear financial signals.
                </div>
                <div class="about-list">
                    <div class="about-list-item"><span class="tick">✓</span><span>Identify high-risk applicants earlier in the lending workflow.</span></div>
                    <div class="about-list-item"><span class="tick">✓</span><span>Reduce inconsistency in credit assessment decisions.</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="about-card">
                <div class="about-icon">02</div>
                <div class="about-card-title">Solution Approach</div>
                <div class="about-card-copy">
                    The system converts applicant information into engineered features and
                    scores each case with a tuned classification model.
                </div>
                <div class="about-list">
                    <div class="about-list-item"><span class="tick">✓</span><span>Estimate default probability before approval.</span></div>
                    <div class="about-list-item"><span class="tick">✓</span><span>Translate probability scores into review-ready decisions.</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="about-card">
                <div class="about-icon">03</div>
                <div class="about-card-title">Business Value</div>
                <div class="about-card-copy">
                    The dashboard supports analysts with consistent evidence, model metrics,
                    and exportable portfolio-level prediction results.
                </div>
                <div class="about-list">
                    <div class="about-list-item"><span class="tick">✓</span><span>Support faster risk triage and underwriting review.</span></div>
                    <div class="about-list-item"><span class="tick">✓</span><span>Improve visibility into model behavior and trade-offs.</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown(
            f"""
            <div class="about-card">
                <div class="about-card-title">Dataset and Modeling Scope</div>
                <div class="about-card-copy">
                    Built around the Home Credit Default Risk dataset, the project frames
                    default prediction as a supervised binary classification problem.
                </div>
                <div class="about-mini-grid">
                    <div class="about-mini"><div class="k">Dataset</div><div class="v">Home Credit Default Risk</div></div>
                    <div class="about-mini"><div class="k">Target</div><div class="v">TARGET</div></div>
                    <div class="about-mini"><div class="k">Task</div><div class="v">Binary Classification</div></div>
                    <div class="about-mini"><div class="k">Features Used</div><div class="v">{metrics["features"]}</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="about-card">
                <div class="about-card-title">Experiment Design</div>
                <div class="about-card-copy">
                    Multiple model families were compared using business-relevant evaluation
                    metrics, then the final classifier was calibrated with an optimized threshold.
                </div>
                <div class="about-card-title" style="font-size:13px;margin-top:14px;">Models Evaluated</div>
                <div class="about-chip-wrap">{models_html}</div>
                <div class="about-card-title" style="font-size:13px;margin-top:18px;">Evaluation Metrics</div>
                <div class="about-chip-wrap">{metrics_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown(
        f"""
        <div class="about-impact">
            <div>
                <h3>Business Impact</h3>
                <p>
                    By identifying risky applications before approval, the system helps lending
                    teams reduce preventable credit exposure, improve portfolio quality, and
                    prioritize manual review where it matters most.
                </p>
            </div>
            <div class="impact-grid">
                <div class="impact-cell"><div class="v">Earlier Risk Signals</div><div class="k">Default probability is surfaced before loan approval.</div></div>
                <div class="impact-cell"><div class="v">Consistent Review</div><div class="k">Threshold-based decisions reduce subjective variation.</div></div>
                <div class="impact-cell"><div class="v">Analyst Ready</div><div class="k">Predictions can be reviewed, exported, and audited.</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    section_head(
        "Machine Learning Pipeline",
        "End-to-end workflow followed during model development."
    )

    steps = [
        ("1", "01", "Data Collection"),
        ("2", "02", "Data Cleaning"),
        ("3", "03", "EDA"),
        ("4", "04", "Feature Engineering"),
        ("5", "05", "Model Training"),
        ("6", "06", "Model Evaluation"),
        ("7", "07", "Threshold Optimization"),
        ("8", "08", "Deployment"),
    ]

    html = "".join(
        f"""
        <div class="rail-step">
            <div class="rail-node">{icon}</div>
            <div class="rail-num">STEP {num}</div>
            <div class="rail-name">{name}</div>
        </div>
        """
        for num, icon, name in steps
    )

    st.markdown(
        f"""
        <div class="ciq-card rail-wrap">
            <div class="rail">
                {html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
        <div class="ciq-footer">
                <div>
                    Developed by <b>Krishna</b>
                </div>
                <div>
                    Home Credit Default Risk Dataset &middot; Streamlit &middot; Scikit-Learn &middot; XGBoost
                </div>
         </div>
        """,
        unsafe_allow_html=True,
    )

