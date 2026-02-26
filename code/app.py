import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Study Abroad Cost Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── INJECT CSS ─────────────────────────────────────────────────────────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700;1,900&family=Special+Elite&family=Oswald:wght@700&family=Permanent+Marker&family=IM+Fell+English:ital@0;1&display=swap" rel="stylesheet">
<style>

/* ── GLOBAL BACKGROUND — light grey with subtle noise ── */
html, body, .stApp, [data-testid="stAppViewContainer"], .main {
    background-color: #e9e9e9 !important;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.06'/%3E%3C/svg%3E") !important;
}
.block-container {
    padding: 0.5rem 2rem 3rem 2rem !important;
    max-width: 1200px !important;
}
#MainMenu, footer, header { visibility: hidden; }
* { box-sizing: border-box; }

/* ── SIDEBAR — crumpled paper look ── */
[data-testid="stSidebar"] {
    background-color: #f0ede6 !important;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)' opacity='0.08'/%3E%3C/svg%3E") !important;
    border-right: 2px solid #ccc !important;
}
[data-testid="stSidebar"] * { color: #1a1a1a !important; }

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: 3px solid #1a1a1a !important;
    gap: 3px;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Special Elite', cursive !important;
    font-size: 0.85rem !important;
    color: #555 !important;
    background: #ddd !important;
    border: 2px solid #aaa !important;
    border-bottom: none !important;
    border-radius: 5px 5px 0 0 !important;
    padding: 0.35rem 1rem !important;
    transform: rotate(-0.8deg);
    letter-spacing: 0.5px;
}
[data-testid="stTabs"] [role="tab"]:nth-child(2) { transform: rotate(0.5deg); }
[data-testid="stTabs"] [role="tab"]:nth-child(3) { transform: rotate(-0.3deg); }
[data-testid="stTabs"] [role="tab"]:nth-child(4) { transform: rotate(0.7deg); }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #fff !important;
    color: #c0392b !important;
    border-color: #1a1a1a !important;
    font-weight: 900 !important;
    transform: rotate(0deg) translateY(2px) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: #c0392b !important;
    color: #fff !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Oswald', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    padding: 0.6rem 2.5rem !important;
    box-shadow: 4px 4px 0 #7b241c !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translate(-2px, -2px) !important;
    box-shadow: 6px 6px 0 #7b241c !important;
}

/* ── SELECTBOX / INPUTS ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stSlider"] {
    font-family: 'Special Elite', cursive !important;
}

/* ── COLLAGE HERO TITLE ── */
.hero-wrap {
    position: relative;
    padding: 2rem 2rem 1.5rem 2rem;
    margin-bottom: 0.5rem;
    background: #fff;
    border: 2px solid #1a1a1a;
    box-shadow: 6px 6px 0 #1a1a1a;
    overflow: visible;
}
.hero-wrap::before {
    content: "";
    position: absolute;
    top: -8px; left: 0; right: 0; height: 8px;
    background: #fff;
    clip-path: polygon(0% 100%,1% 20%,2% 80%,3% 10%,4% 90%,5% 0%,6% 70%,7% 30%,8% 85%,9% 5%,10% 75%,11% 25%,12% 95%,13% 10%,14% 80%,15% 20%,16% 90%,17% 0%,18% 70%,19% 30%,20% 85%,21% 5%,22% 75%,23% 25%,24% 95%,25% 10%,26% 80%,27% 20%,28% 90%,29% 0%,30% 70%,31% 30%,32% 85%,33% 5%,34% 75%,35% 25%,36% 95%,37% 10%,38% 80%,39% 20%,40% 90%,41% 0%,42% 70%,43% 30%,44% 85%,45% 5%,46% 75%,47% 25%,48% 95%,49% 10%,50% 80%,51% 20%,52% 90%,53% 0%,54% 70%,55% 30%,56% 85%,57% 5%,58% 75%,59% 25%,60% 95%,61% 10%,62% 80%,63% 20%,64% 90%,65% 0%,66% 70%,67% 30%,68% 85%,69% 5%,70% 75%,71% 25%,72% 95%,73% 10%,74% 80%,75% 20%,76% 90%,77% 0%,78% 70%,79% 30%,80% 85%,81% 5%,82% 75%,83% 25%,84% 95%,85% 10%,86% 80%,87% 20%,88% 90%,89% 0%,90% 70%,91% 30%,92% 85%,93% 5%,94% 75%,95% 25%,96% 95%,97% 10%,98% 80%,99% 20%,100% 100%);
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 5rem;
    font-weight: 900;
    color: #1a1a1a;
    line-height: 0.95;
    letter-spacing: -2px;
    margin: 0;
}
.hero-title .red-italic {
    color: #c0392b;
    font-style: italic;
}

/* ── NEWSPAPER CLIPPING ── */
.newspaper-clip {
    background: #f5f0e8;
    border: 1px solid #ccc;
    padding: 0.8rem 1rem;
    font-family: 'IM Fell English', serif;
    font-size: 0.82rem;
    line-height: 1.6;
    color: #333;
    transform: rotate(-1.2deg);
    box-shadow: 3px 3px 8px rgba(0,0,0,0.15);
    position: relative;
    display: inline-block;
    max-width: 100%;
}
.newspaper-clip::before {
    content: attr(data-headline);
    display: block;
    font-family: 'Oswald', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #888;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
    margin-bottom: 6px;
}

/* ── STARBURST ── */
.starburst {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 80px; height: 80px;
    background: #c0392b;
    color: white;
    font-family: 'Oswald', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    line-height: 1.2;
    clip-path: polygon(50% 0%,61% 35%,98% 35%,68% 57%,79% 91%,50% 70%,21% 91%,32% 57%,2% 35%,39% 35%);
    transform: rotate(-15deg);
}
.starburst.yellow {
    background: #f1c40f;
    color: #1a1a1a;
    transform: rotate(10deg);
    width: 70px; height: 70px;
}

/* ── CUT-OUT STICKER ── */
.cutout-sticker {
    display: inline-block;
    background: white;
    border-radius: 50px;
    padding: 5px 16px;
    font-family: 'Special Elite', cursive;
    font-size: 0.78rem;
    color: #1a1a1a;
    box-shadow: 0 0 0 3px white, 0 0 0 5px #1a1a1a;
    margin: 4px 6px;
    transform: rotate(-2deg);
    letter-spacing: 0.5px;
}
.cutout-sticker.red-s {
    background: #c0392b; color: white;
    box-shadow: 0 0 0 3px white, 0 0 0 5px #c0392b;
    transform: rotate(1.5deg);
}
.cutout-sticker.blue-s {
    background: #2980b9; color: white;
    box-shadow: 0 0 0 3px white, 0 0 0 5px #2980b9;
    transform: rotate(-3deg);
}
.cutout-sticker.yellow-s {
    background: #f1c40f; color: #1a1a1a;
    box-shadow: 0 0 0 3px white, 0 0 0 5px #f1c40f;
    transform: rotate(2deg);
}
.cutout-sticker.green-s {
    background: #27ae60; color: white;
    box-shadow: 0 0 0 3px white, 0 0 0 5px #27ae60;
    transform: rotate(-1deg);
}

/* ── CRUMPLED PAPER CARD ── */
.paper-card {
    background: #faf8f4;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)' opacity='0.07'/%3E%3C/svg%3E");
    border: 1px solid #ddd;
    padding: 1.4rem 1.6rem;
    margin: 0.4rem 0;
    box-shadow: 3px 3px 10px rgba(0,0,0,0.1), inset 0 0 30px rgba(0,0,0,0.02);
    position: relative;
}
.paper-card::before {
    content: "";
    position: absolute;
    top: -5px; left: 0; right: 0; height: 5px;
    background: #faf8f4;
    clip-path: polygon(0% 100%,1.5% 0%,3% 80%,4.5% 10%,6% 90%,7.5% 5%,9% 75%,10.5% 20%,12% 95%,13.5% 0%,15% 70%,16.5% 30%,18% 85%,19.5% 5%,21% 75%,22.5% 25%,24% 90%,25.5% 10%,27% 80%,28.5% 20%,30% 95%,31.5% 5%,33% 70%,34.5% 30%,36% 85%,37.5% 10%,39% 75%,40.5% 25%,42% 90%,43.5% 0%,45% 80%,46.5% 20%,48% 95%,49.5% 5%,51% 70%,52.5% 30%,54% 85%,55.5% 10%,57% 75%,58.5% 25%,60% 90%,61.5% 0%,63% 80%,64.5% 20%,66% 95%,67.5% 5%,69% 70%,70.5% 30%,72% 85%,73.5% 10%,75% 75%,76.5% 25%,78% 90%,79.5% 0%,81% 80%,82.5% 20%,84% 95%,85.5% 5%,87% 70%,88.5% 30%,90% 85%,91.5% 10%,93% 75%,94.5% 25%,96% 90%,97.5% 0%,99% 80%,100% 100%);
}
.paper-card::after {
    content: "";
    position: absolute;
    bottom: -5px; left: 0; right: 0; height: 5px;
    background: #faf8f4;
    clip-path: polygon(0% 0%,1.5% 100%,3% 20%,4.5% 90%,6% 10%,7.5% 95%,9% 25%,10.5% 80%,12% 5%,13.5% 100%,15% 30%,16.5% 70%,18% 15%,19.5% 95%,21% 25%,22.5% 75%,24% 10%,25.5% 90%,27% 20%,28.5% 80%,30% 5%,31.5% 95%,33% 30%,34.5% 70%,36% 15%,37.5% 90%,39% 25%,40.5% 75%,42% 10%,43.5% 100%,45% 20%,46.5% 80%,48% 5%,49.5% 95%,51% 30%,52.5% 70%,54% 15%,55.5% 90%,57% 25%,58.5% 75%,60% 10%,61.5% 100%,63% 20%,64.5% 80%,66% 5%,67.5% 95%,69% 30%,70.5% 70%,72% 15%,73.5% 90%,75% 25%,76.5% 75%,78% 10%,79.5% 100%,81% 20%,82.5% 80%,84% 5%,85.5% 95%,87% 30%,88.5% 70%,90% 15%,91.5% 90%,93% 25%,94.5% 75%,96% 10%,97.5% 100%,99% 20%,100% 0%);
}

/* ── SECTION HEADER ── */
.sec-label {
    font-family: 'Special Elite', cursive;
    font-size: 0.7rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 2px;
}
.sec-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: #1a1a1a;
    border-bottom: 4px solid #c0392b;
    padding-bottom: 4px;
    display: inline-block;
    margin: 0 0 1rem 0;
}

/* ── METRIC CARD ── */
.metric-card {
    background: white;
    border: 2px solid #1a1a1a;
    padding: 1.2rem 0.8rem;
    text-align: center;
    box-shadow: 4px 4px 0 #1a1a1a;
    position: relative;
    margin: 4px;
}
.metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 5px;
    background: var(--accent, #c0392b);
}
.metric-num {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 900;
    color: #1a1a1a;
    line-height: 1;
}
.metric-lbl {
    font-family: 'Special Elite', cursive;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #888;
    margin-top: 6px;
}

/* ── PREDICTION BOX ── */
.pred-box {
    background: #1a1a1a;
    border: 3px solid #c0392b;
    padding: 2.5rem 2rem;
    text-align: center;
    box-shadow: 8px 8px 0 #c0392b;
    position: relative;
    margin: 1rem 0;
}
.pred-label {
    font-family: 'Special Elite', cursive;
    font-size: 0.8rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #888;
}
.pred-amount {
    font-family: 'Playfair Display', serif;
    font-size: 5rem;
    font-weight: 900;
    color: #f1c40f;
    line-height: 1;
    text-shadow: 3px 3px 0 rgba(0,0,0,0.4);
}
.pred-sub {
    font-family: 'Special Elite', cursive;
    font-size: 0.85rem;
    color: #666;
    margin-top: 0.8rem;
}

/* ── NOTE BOX ── */
.note-box {
    background: #fdf9f0;
    border-left: 5px solid #c0392b;
    padding: 0.8rem 1.2rem;
    font-family: 'Special Elite', cursive;
    font-size: 0.85rem;
    color: #444;
    margin: 0.8rem 0;
    transform: rotate(-0.3deg);
    box-shadow: 2px 2px 6px rgba(0,0,0,0.08);
}

/* ── RAINBOW STRIPE ── */
.rainbow {
    height: 6px;
    background: linear-gradient(90deg,#c0392b,#e67e22,#f1c40f,#27ae60,#2980b9,#8e44ad,#c0392b);
    border-radius: 3px;
    margin: 0.5rem 0 1.2rem 0;
}

/* ── FOOTER ── */
.footer {
    background: #1a1a1a;
    color: #555;
    font-family: 'Special Elite', cursive;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-align: center;
    padding: 14px;
    margin-top: 3rem;
    border-radius: 3px;
}

/* ── TAPE STRIP ── */
.tape {
    display: inline-block;
    background: rgba(241,196,15,0.55);
    padding: 2px 20px;
    transform: rotate(-1.5deg);
    font-family: 'Special Elite', cursive;
    font-size: 0.72rem;
    letter-spacing: 2px;
    color: #555;
    margin: 0 6px;
    border: 1px solid rgba(241,196,15,0.3);
}

/* ── HALFTONE HEADER ACCENT ── */
.halftone-accent {
    display: inline-block;
    width: 60px; height: 60px;
    background-image: radial-gradient(circle, #1a1a1a 1.5px, transparent 1.5px);
    background-size: 8px 8px;
    opacity: 0.15;
    position: absolute;
    top: 10px; right: 10px;
}
</style>
""")


# ── DATA ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("International_Education_Costs.csv")
    df.dropna(inplace=True)
    for c in ["Tuition_USD","Rent_USD","Insurance_USD","Visa_Fee_USD","Living_Cost_Index","Duration_Years"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    df["Total_Cost_USD"] = (
        df["Tuition_USD"] * df["Duration_Years"]
        + df["Rent_USD"] * 12 * df["Duration_Years"]
        + df["Insurance_USD"] * df["Duration_Years"]
        + df["Visa_Fee_USD"]
    )
    return df

@st.cache_resource
def train_model(algo, df):
    features = ["Duration_Years","Tuition_USD","Rent_USD","Insurance_USD","Visa_Fee_USD","Living_Cost_Index"]
    cat_cols  = ["Country","Level"]
    dfc = df.copy()
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        dfc[c+"_enc"] = le.fit_transform(dfc[c].astype(str))
        encoders[c] = le
    feat_cols = features + [c+"_enc" for c in cat_cols]
    X = dfc[feat_cols]; y = dfc["Total_Cost_USD"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = {"Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
           "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
           "Linear Regression": LinearRegression()}[algo]
    mdl.fit(X_tr, y_tr)
    y_pred = mdl.predict(X_te)
    metrics = {"r2": r2_score(y_te, y_pred),
               "mae": mean_absolute_error(y_te, y_pred),
               "rmse": np.sqrt(mean_squared_error(y_te, y_pred))}
    return mdl, encoders, feat_cols, metrics, X_te, y_te, y_pred

df = load_data()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:900;
                color:#1a1a1a;border-bottom:3px solid #c0392b;padding-bottom:6px;margin-bottom:12px;">
        Settings
    </div>""", unsafe_allow_html=True)
    algo = st.selectbox("AI Algorithm", ["Random Forest","Gradient Boosting","Linear Regression"])
    st.markdown('<div class="rainbow"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="newspaper-clip" data-headline="Dataset Info" style="transform:rotate(0.5deg);width:100%;">
        {df['Country'].nunique()} countries &nbsp;·&nbsp; {df['University'].nunique()} universities<br>
        {len(df):,} records &nbsp;·&nbsp; Predicts Total Cost USD
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-family:'Special Elite',cursive;font-size:0.72rem;color:#aaa;line-height:1.9;margin:0;">
    DATA200 · Applied Statistical Analysis<br>Built with Python + Streamlit
    </p>""", unsafe_allow_html=True)

# ── TRAIN ──────────────────────────────────────────────────────────────────────
model, encoders, feat_cols, metrics, X_te, y_te, y_pred = train_model(algo, df)

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="rainbow"></div>', unsafe_allow_html=True)

col_title, col_tags = st.columns([3, 1])
with col_title:
    st.markdown("""
    <div class="hero-wrap" style="position:relative;">
        <div class="halftone-accent"></div>
        <div class="hero-title">
            Study <span class="red-italic">Abroad</span><br>Cost Predictor
        </div>
        <div style="margin-top:1rem;">
            <span class="cutout-sticker red-s">AI Powered</span>
            <span class="cutout-sticker blue-s">73 Countries</span>
            <span class="cutout-sticker yellow-s">Real Data</span>
            <span class="cutout-sticker green-s">Interactive</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_tags:
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:100%;gap:16px;padding-top:1rem;">
        <div class="starburst">AI<br>MODEL</div>
        <div class="starburst yellow">{len(df):,}<br>RECORDS</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="rainbow"></div>', unsafe_allow_html=True)

# ── TAPE LABELS ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:0.5rem;">
    <span class="tape">DATASET</span>
    <span class="tape" style="transform:rotate(1deg);">MACHINE LEARNING</span>
    <span class="tape" style="transform:rotate(-0.8deg);">PREDICTION</span>
</div>
""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📋  Dataset", "📈  Explore", "🤖  Model", "💸  Predict"])

BG   = "#e9e9e9"
RED  = "#c0392b"
BLUE = "#2980b9"
YEL  = "#f1c40f"
GRN  = "#27ae60"
DARK = "#1a1a1a"
FONT = "Special Elite"

def base_layout(h=360, title=""):
    return dict(
        paper_bgcolor=BG, plot_bgcolor="white",
        font_family=FONT, font_color=DARK,
        margin=dict(l=0, r=0, t=36 if title else 10, b=0),
        height=h, title=title,
        title_font=dict(family="Playfair Display", size=15, color=DARK)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-label">overview</div><div class="sec-title">At a Glance</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, lbl, accent in [
        (c1, f"{len(df):,}", "Records",       "#c0392b"),
        (c2, str(df['Country'].nunique()),     "Countries",     "#2980b9"),
        (c3, str(df['University'].nunique()),  "Universities",  "#f1c40f"),
        (c4, str(df['Level'].nunique()),       "Degree Levels", "#27ae60"),
        (c5, f"${df['Total_Cost_USD'].mean():,.0f}", "Avg Total Cost", "#8e44ad"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{accent};">
                <div class="metric-num">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title" style="margin-top:1.5rem;">Raw Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="note-box">Showing the first 10 rows — each row is one university program in one city.</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<div class="sec-title" style="margin-top:1rem;">Stats Summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec-title">Top 15 Countries</div>', unsafe_allow_html=True)
        top_c = df["Country"].value_counts().head(15).reset_index()
        top_c.columns = ["Country","Count"]
        fig = px.bar(top_c, x="Count", y="Country", orientation="h",
                     color="Count", color_continuous_scale=["#f5f5f5", RED],
                     template="simple_white")
        fig.update_layout(**base_layout(400), yaxis=dict(autorange="reversed"))
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-title">Degree Level Split</div>', unsafe_allow_html=True)
        lv = df["Level"].value_counts().reset_index()
        lv.columns = ["Level","Count"]
        fig2 = px.pie(lv, names="Level", values="Count",
                      color_discrete_sequence=[RED, BLUE, YEL, GRN],
                      template="simple_white", hole=0.45)
        fig2.update_layout(**base_layout(400))
        fig2.update_traces(textfont_size=13, marker=dict(line=dict(color="white", width=3)))
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">Explore the Data</div>', unsafe_allow_html=True)
    num_cols = ["Tuition_USD","Rent_USD","Insurance_USD","Visa_Fee_USD","Living_Cost_Index","Duration_Years","Total_Cost_USD"]
    chosen = st.selectbox("Pick a column to explore", num_cols)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=chosen, nbins=40, template="simple_white",
                           color_discrete_sequence=[RED])
        fig.update_layout(**base_layout(320, f"Distribution of {chosen}"))
        fig.update_traces(marker_line_color="white", marker_line_width=0.8)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.box(df, x="Level", y=chosen, template="simple_white",
                      color="Level", color_discrete_sequence=[RED, BLUE, YEL, GRN])
        fig2.update_layout(**base_layout(320, f"{chosen} by Degree Level"), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec-title">Tuition vs Total Cost</div>', unsafe_allow_html=True)
    fig3 = px.scatter(df, x="Tuition_USD", y="Total_Cost_USD", color="Level",
                      hover_data=["Country","University","Program"],
                      template="simple_white",
                      color_discrete_sequence=[RED, BLUE, YEL, GRN])
    fig3.update_layout(**base_layout(380))
    fig3.update_traces(marker=dict(size=7, line=dict(color="white", width=0.8)))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="sec-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    fig_h, ax = plt.subplots(figsize=(9, 5))
    fig_h.patch.set_facecolor(BG)
    ax.set_facecolor("white")
    corr = df[num_cols].corr()
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
                square=True, linewidths=1.5, linecolor=BG, ax=ax,
                annot_kws={"family":"monospace","size":9},
                cbar_kws={"shrink":0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=12, pad=10, color=DARK)
    plt.xticks(rotation=30, ha="right", fontsize=9, color=DARK)
    plt.yticks(fontsize=9, color=DARK)
    plt.tight_layout()
    st.pyplot(fig_h)

    st.markdown('<div class="sec-title">Total Cost by Country (Top 12)</div>', unsafe_allow_html=True)
    top12 = df["Country"].value_counts().head(12).index.tolist()
    df12  = df[df["Country"].isin(top12)]
    fig4  = px.box(df12, x="Country", y="Total_Cost_USD", template="simple_white",
                   color="Country", color_discrete_sequence=px.colors.qualitative.Bold)
    fig4.update_layout(**base_layout(400), showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">Model Performance</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, val, lbl, accent in [
        (c1, f"{metrics['r2']:.4f}",     "R² Score",          "#c0392b"),
        (c2, f"${metrics['mae']:,.0f}",  "MAE",               "#2980b9"),
        (c3, f"${metrics['rmse']:,.0f}", "RMSE",              "#f1c40f"),
        (c4, f"{metrics['r2']*100:.1f}%","Variance Explained","#27ae60"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{accent};">
                <div class="metric-num">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec-title" style="margin-top:1rem;">Actual vs Predicted</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_te, y=y_pred, mode="markers",
                                 marker=dict(color=RED, size=6, opacity=0.55,
                                             line=dict(color="white", width=0.5)),
                                 name="Predictions"))
        mn, mx = float(y_te.min()), float(y_te.max())
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                 line=dict(color=DARK, dash="dash", width=2),
                                 name="Perfect Fit"))
        fig.update_layout(**base_layout(340), xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-title" style="margin-top:1rem;">Residuals</div>', unsafe_allow_html=True)
        resid = np.array(y_te) - np.array(y_pred)
        fig2  = px.histogram(x=resid, nbins=40, template="simple_white",
                             color_discrete_sequence=[BLUE])
        fig2.add_vline(x=0, line_dash="dash", line_color=RED, line_width=2)
        fig2.update_layout(**base_layout(340), xaxis_title="Residual", yaxis_title="Count")
        fig2.update_traces(marker_line_color="white", marker_line_width=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    if hasattr(model, "feature_importances_"):
        st.markdown('<div class="sec-title">Feature Importance</div>', unsafe_allow_html=True)
        fi  = pd.DataFrame({"Feature": feat_cols, "Importance": model.feature_importances_})
        fi  = fi.sort_values("Importance", ascending=True)
        fig3 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                      template="simple_white",
                      color="Importance", color_continuous_scale=["#f5f5f5", RED])
        fig3.update_layout(**base_layout(340), coloraxis_showscale=False)
        fig3.update_traces(marker_line_color="white", marker_line_width=1)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="sec-title">How the Model Works</div>', unsafe_allow_html=True)
    explanations = {
        "Random Forest": "Builds 200 decision trees and averages their predictions. Great at handling complex patterns and non-linear relationships.",
        "Gradient Boosting": "Builds trees one by one, each correcting the errors of the previous. Usually very accurate but slower to train.",
        "Linear Regression": "Fits a straight line through the data. Fast and easy to interpret — each feature has a direct coefficient.",
    }
    st.markdown(f"""
    <div class="newspaper-clip" data-headline="Model Explanation">
        {explanations[algo]}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-title">Predict Your Study Abroad Cost</div>', unsafe_allow_html=True)
    st.markdown('<div class="note-box">Fill in the details below and the AI model will estimate your total study abroad cost.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        country  = st.selectbox("Country", sorted(df["Country"].unique()))
        level    = st.selectbox("Degree Level", sorted(df["Level"].unique()))
        duration = st.slider("Duration (Years)", 1.0, 6.0, 2.0, 0.5)
    with col2:
        tuition   = st.number_input("Tuition per Year (USD)", 1000, 100000, 20000, 500)
        rent      = st.number_input("Monthly Rent (USD)", 100, 5000, 1200, 50)
        insurance = st.number_input("Insurance per Year (USD)", 100, 5000, 800, 50)
    with col3:
        visa_fee = st.number_input("Visa Fee (USD)", 0, 2000, 200, 10)
        lci      = st.slider("Living Cost Index", 20.0, 120.0, 70.0, 0.5)

    if st.button("✈  Predict Total Cost"):
        row = {
            "Duration_Years":    duration,
            "Tuition_USD":       tuition,
            "Rent_USD":          rent,
            "Insurance_USD":     insurance,
            "Visa_Fee_USD":      visa_fee,
            "Living_Cost_Index": lci,
            "Country_enc": encoders["Country"].transform([country])[0]
                           if country in encoders["Country"].classes_ else 0,
            "Level_enc":   encoders["Level"].transform([level])[0]
                           if level in encoders["Level"].classes_ else 0,
        }
        X_in       = pd.DataFrame([row])[feat_cols]
        prediction = model.predict(X_in)[0]

        st.markdown(f"""
        <div class="pred-box">
            <div class="pred-label">Estimated Total Cost</div>
            <div class="pred-amount">${prediction:,.0f}</div>
            <div class="pred-sub">{level} &nbsp;·&nbsp; {country} &nbsp;·&nbsp; {duration} year(s) &nbsp;·&nbsp; {algo}</div>
        </div>""", unsafe_allow_html=True)

        breakdown = {
            "Tuition":   tuition * duration,
            "Rent":      rent * 12 * duration,
            "Insurance": insurance * duration,
            "Visa Fee":  visa_fee,
        }
        fig = px.pie(names=list(breakdown.keys()), values=list(breakdown.values()),
                     color_discrete_sequence=[RED, DARK, YEL, BLUE],
                     template="simple_white", hole=0.45, title="Cost Breakdown")
        fig.update_layout(**base_layout(340))
        fig.update_traces(textfont_size=13, marker=dict(line=dict(color="white", width=2)))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sec-title">Real Universities That Match</div>', unsafe_allow_html=True)
        similar = df[(df["Country"]==country) & (df["Level"]==level)][
            ["University","City","Program","Duration_Years","Tuition_USD","Rent_USD","Total_Cost_USD"]
        ].head(8)
        if len(similar):
            st.dataframe(similar.reset_index(drop=True), use_container_width=True)
        else:
            st.markdown(f'<div class="note-box">No exact matches for {level} in {country} in the dataset.</div>', unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="rainbow"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
  DATA200 &nbsp;·&nbsp; Applied Statistical Analysis &nbsp;·&nbsp; International Education Cost Predictor
</div>
""", unsafe_allow_html=True)