import matplotlib
matplotlib.use("Agg")
import streamlit as st
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="⚡ Options Pricer Pro", layout="wide",
                   page_icon="⚡", initial_sidebar_state="collapsed")

# ── THEME COLORS PER MODEL ──────────────────────────────────
BS = dict(
    primary="#A855F7",   # purple
    light="#D8B4FE",
    dark="#1E0A2E",
    mid="#2D1B4E",
    border="#7C3AED",
    glow="rgba(168,85,247,0.15)",
    label="🔮 BLACK-SCHOLES",
    emoji="🔮"
)
MC = dict(
    primary="#06B6D4",   # cyan
    light="#A5F3FC",
    dark="#021B2E",
    mid="#0C2D3E",
    border="#0891B2",
    glow="rgba(6,182,212,0.15)",
    label="🎲 MONTE CARLO",
    emoji="🎲"
)
BT = dict(
    primary="#10B981",   # emerald
    light="#A7F3D0",
    dark="#022C22",
    mid="#064E3B",
    border="#059669",
    glow="rgba(16,185,129,0.15)",
    label="🌳 BINOMIAL TREE",
    emoji="🌳"
)

st.markdown(f"""
<style>
  html,body,.stApp{{background:#050505!important;color:#FFF;}}
  section[data-testid="stSidebar"]{{display:none;}}
  .block-container{{padding-top:0.4rem!important;padding-bottom:1rem!important;}}

  /* TABS */
  .stTabs [data-baseweb="tab-list"]{{background:#050505;border-bottom:2px solid #333;gap:4px;}}
  .stTabs [data-baseweb="tab"]{{background:#111;color:#777;border:1px solid #333;
    border-radius:6px 6px 0 0;font-family:monospace;font-weight:bold;padding:7px 18px;font-size:13px;}}
  .stTabs [aria-selected="true"]{{background:linear-gradient(135deg,#A855F7,#06B6D4)!important;
    color:#FFF!important;border-color:transparent!important;}}

  /* BUTTONS */
  .stButton>button{{font-family:monospace;font-weight:bold;border:none;border-radius:8px;
    padding:8px 24px;font-size:14px;transition:all 0.2s;}}
  .bs-btn>button{{background:linear-gradient(135deg,#7C3AED,#A855F7);color:#fff;}}
  .bs-btn>button:hover{{background:linear-gradient(135deg,#6D28D9,#9333EA);transform:scale(1.02);}}
  .mc-btn>button{{background:linear-gradient(135deg,#0891B2,#06B6D4);color:#fff;}}
  .mc-btn>button:hover{{background:linear-gradient(135deg,#0E7490,#0891B2);transform:scale(1.02);}}
  .bt-btn>button{{background:linear-gradient(135deg,#059669,#10B981);color:#fff;}}
  .bt-btn>button:hover{{background:linear-gradient(135deg,#047857,#059669);transform:scale(1.02);}}
  .all-btn>button{{background:linear-gradient(135deg,#A855F7,#06B6D4,#10B981);color:#fff;font-size:15px;}}

  /* INPUTS — override per model via parent div */
  .bs-panel input,.bs-panel select{{
    background:{BS['mid']}!important;color:{BS['light']}!important;
    border:1px solid {BS['border']}!important;border-radius:6px!important;font-family:monospace!important;}}
  .mc-panel input,.mc-panel select{{
    background:{MC['mid']}!important;color:{MC['light']}!important;
    border:1px solid {MC['border']}!important;border-radius:6px!important;font-family:monospace!important;}}
  .bt-panel input,.bt-panel select{{
    background:{BT['mid']}!important;color:{BT['light']}!important;
    border:1px solid {BT['border']}!important;border-radius:6px!important;font-family:monospace!important;}}

  /* SLIDER TRACKS */
  .bs-panel [data-testid="stSlider"] div[data-baseweb="slider"] div:first-child div:first-child
    {{background:{BS['primary']}!important;}}
  .mc-panel [data-testid="stSlider"] div[data-baseweb="slider"] div:first-child div:first-child
    {{background:{MC['primary']}!important;}}
  .bt-panel [data-testid="stSlider"] div[data-baseweb="slider"] div:first-child div:first-child
    {{background:{BT['primary']}!important;}}

  label,.stSlider label,.stNumberInput label,.stSelectbox label{{
    font-family:monospace!important;font-weight:bold!important;font-size:12px!important;}}

  h1,h2,h3{{font-family:monospace!important;}}
  .result-card{{border-radius:8px;padding:14px 18px;text-align:center;
    font-family:Courier New,monospace;margin:4px 0;}}
  .section-divider{{border:none;border-top:1px solid #1a1a1a;margin:12px 0;}}
</style>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  MATH CORE
# ════════════════════════════════════════════════════════════
def bs_params(St,K,r,sigma,T,q=0):
    d1=(math.log(St/K)+(r-q+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    return d1, d1-sigma*math.sqrt(T)

def run_bs(St,K,sigma,T,r,q):
    d1,d2=bs_params(St,K,r,sigma,T,q)
    sc=St*math.exp(-q*T)*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2)
    sp=K*math.exp(-r*T)*norm.cdf(-d2)-St*math.exp(-q*T)*norm.cdf(-d1)
    dc=math.exp(-r*T)*norm.cdf(d2)
    dp=math.exp(-r*T)*norm.cdf(-d2)
    delta_c=math.exp(-q*T)*norm.cdf(d1)
    delta_p=-math.exp(-q*T)*norm.cdf(-d1)
    gamma=math.exp(-q*T)*norm.pdf(d1)/(St*sigma*math.sqrt(T))
    vega=St*math.exp(-q*T)*norm.pdf(d1)*math.sqrt(T)/100
    theta_c=(-(St*norm.pdf(d1)*sigma*math.exp(-q*T))/(2*math.sqrt(T))
             -r*K*math.exp(-r*T)*norm.cdf(d2)+q*St*math.exp(-q*T)*norm.cdf(d1))/365
    theta_p=(-(St*norm.pdf(d1)*sigma*math.exp(-q*T))/(2*math.sqrt(T))
             +r*K*math.exp(-r*T)*norm.cdf(-d2)-q*St*math.exp(-q*T)*norm.cdf(-d1))/365
    return dict(d1=d1,d2=d2,dc=dc,sc=sc,dp=dp,sp=sp,
                delta_c=delta_c,delta_p=delta_p,gamma=gamma,
                vega=vega,theta_c=theta_c,theta_p=theta_p)

def run_mc(St,K,sigma,T,r,reps):
    np.random.seed(42)
    Z=np.random.standard_normal(reps)
    ST=St*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*Z)
    d1,d2=bs_params(St,K,r,sigma,T)
    return dict(d1=d1,d2=d2,
                dc=np.exp(-r*T)*np.mean(ST>K),
                sc=np.exp(-r*T)*np.mean(np.maximum(ST-K,0)),
                dp=np.exp(-r*T)*np.mean(ST<K),
                sp=np.exp(-r*T)*np.mean(np.maximum(K-ST,0)),
                ST=ST,reps=reps)

def run_bt(St,K,sigma,T,r,N):
    dt=T/N; u=math.exp(sigma*math.sqrt(dt)); d=1/u
    p=(math.exp(r*dt)-d)/(u-d)
    ST=np.array([St*(u**j)*(d**(N-j)) for j in range(N+1)])
    cp=np.maximum(ST-K,0); pp=np.maximum(K-ST,0)
    dcp=(ST>K).astype(float); dpp=(ST<K).astype(float)
    for _ in range(N,0,-1):
        cp=np.exp(-r*dt)*(p*cp[1:]+(1-p)*cp[:-1])
        pp=np.exp(-r*dt)*(p*pp[1:]+(1-p)*pp[:-1])
        dcp=np.exp(-r*dt)*(p*dcp[1:]+(1-p)*dcp[:-1])
        dpp=np.exp(-r*dt)*(p*dpp[1:]+(1-p)*dpp[:-1])
    d1,d2=bs_params(St,K,r,sigma,T)
    return dict(d1=d1,d2=d2,dc=dcp[0],sc=cp[0],dp=dpp[0],sp=pp[0])

# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#1E0A2E,#021B2E,#022C22);
            font-family:Courier New,monospace;padding:16px 24px;
            border-bottom:1px solid #333;display:flex;justify-content:space-between;
            align-items:center;margin-bottom:16px;border-radius:0 0 8px 8px'>
  <div>
    <span style='font-size:26px;font-weight:bold;background:linear-gradient(135deg,#A855F7,#06B6D4,#10B981);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:2px'>
      ⚡ OPTIONS PRICER PRO
    </span>
    <div style='color:#555;font-size:11px;margin-top:2px'>
      Black-Scholes &nbsp;·&nbsp; Monte Carlo &nbsp;·&nbsp; Binomial Tree
    </div>
  </div>
  <div style='text-align:right'>
    <span style='color:#A855F7;font-size:11px'>🔮 Analytic</span>&nbsp;&nbsp;
    <span style='color:#06B6D4;font-size:11px'>🎲 Simulation</span>&nbsp;&nbsp;
    <span style='color:#10B981;font-size:11px'>🌳 Lattice</span>
  </div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  THREE INPUT PANELS SIDE BY SIDE
# ════════════════════════════════════════════════════════════
col_bs, col_mc, col_bt = st.columns(3)

def panel_header(theme, subtitle):
    return f"""
    <div style='background:linear-gradient(135deg,{theme["dark"]},{theme["mid"]});
                border:1px solid {theme["border"]};border-radius:10px 10px 0 0;
                padding:14px 18px;text-align:center;
                box-shadow:0 0 20px {theme["glow"]}'>
      <div style='font-size:22px'>{theme["emoji"]}</div>
      <div style='color:{theme["primary"]};font-family:monospace;font-weight:bold;
                  font-size:16px;letter-spacing:2px;margin-top:4px'>{theme["label"]}</div>
      <div style='color:{theme["light"]};font-family:monospace;font-size:10px;
                  opacity:0.6;margin-top:2px'>{subtitle}</div>
    </div>"""

def field_row(label, value, color):
    return f"""
    <div style='display:flex;justify-content:space-between;align-items:center;
                padding:8px 12px;border-bottom:1px solid rgba(255,255,255,0.05)'>
      <span style='color:#888;font-family:monospace;font-size:12px'>{label}</span>
      <span style='color:{color};font-family:monospace;font-weight:bold;font-size:14px'>{value}</span>
    </div>"""

# ── BLACK-SCHOLES PANEL ──────────────────────────────────────
with col_bs:
    st.markdown(panel_header(BS, "Closed-form analytical solution"), unsafe_allow_html=True)
    st.markdown(f"<div style='background:{BS['dark']};border:1px solid {BS['border']};border-top:none;"
                f"border-radius:0 0 10px 10px;padding:16px;box-shadow:0 0 20px {BS['glow']}' "
                f"class='bs-panel'>", unsafe_allow_html=True)

    bs_St    = st.number_input("📍 St — Spot Price",    min_value=0.01, value=10.0, step=0.5,  key="bs_St",    format="%.2f")
    bs_K     = st.number_input("🎯 K  — Strike Price",  min_value=0.01, value=10.0, step=0.5,  key="bs_K",     format="%.2f")
    bs_sigma = st.slider(      "📊 σ  — Volatility",    0.01, 2.0,  0.10, 0.01, key="bs_sigma", format="%.2f")
    bs_T     = st.number_input("⏳ T  — Years to Expiry",min_value=0.01,value=2.0, step=0.25, key="bs_T",     format="%.2f")
    bs_r     = st.slider(      "🏦 r  — Risk-Free Rate", 0.0,  0.30, 0.10, 0.005,key="bs_r",   format="%.3f")
    bs_q     = st.slider(      "💰 q  — Dividend Yield", 0.0,  0.20, 0.0,  0.005,key="bs_q",   format="%.3f")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bs-btn'>", unsafe_allow_html=True)
    run_bs_btn = st.button("🔮 Price with Black-Scholes", key="run_bs", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── MONTE CARLO PANEL ────────────────────────────────────────
with col_mc:
    st.markdown(panel_header(MC, "Stochastic path simulation"), unsafe_allow_html=True)
    st.markdown(f"<div style='background:{MC['dark']};border:1px solid {MC['border']};border-top:none;"
                f"border-radius:0 0 10px 10px;padding:16px;box-shadow:0 0 20px {MC['glow']}' "
                f"class='mc-panel'>", unsafe_allow_html=True)

    mc_St    = st.number_input("📍 St — Spot Price",     min_value=0.01, value=10.0, step=0.5,  key="mc_St",   format="%.2f")
    mc_K     = st.number_input("🎯 K  — Strike Price",   min_value=0.01, value=10.0, step=0.5,  key="mc_K",    format="%.2f")
    mc_sigma = st.slider(      "📊 σ  — Volatility",     0.01, 2.0,  0.10, 0.01, key="mc_sigma",format="%.2f")
    mc_T     = st.number_input("⏳ T  — Years to Expiry", min_value=0.01,value=2.0, step=0.25, key="mc_T",    format="%.2f")
    mc_r     = st.slider(      "🏦 r  — Risk-Free Rate",  0.0,  0.30, 0.10, 0.005,key="mc_r",  format="%.3f")
    mc_reps  = st.select_slider("🎰 Repetitions",
                                options=[1000,5000,10000,50000,100000],
                                value=10000, key="mc_reps")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='mc-btn'>", unsafe_allow_html=True)
    run_mc_btn = st.button("🎲 Price with Monte Carlo", key="run_mc", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── BINOMIAL TREE PANEL ──────────────────────────────────────
with col_bt:
    st.markdown(panel_header(BT, "Discrete-time lattice model"), unsafe_allow_html=True)
    st.markdown(f"<div style='background:{BT['dark']};border:1px solid {BT['border']};border-top:none;"
                f"border-radius:0 0 10px 10px;padding:16px;box-shadow:0 0 20px {BT['glow']}' "
                f"class='bt-panel'>", unsafe_allow_html=True)

    bt_St    = st.number_input("📍 St — Spot Price",     min_value=0.01, value=10.0, step=0.5,  key="bt_St",   format="%.2f")
    bt_K     = st.number_input("🎯 K  — Strike Price",   min_value=0.01, value=10.0, step=0.5,  key="bt_K",    format="%.2f")
    bt_sigma = st.slider(      "📊 σ  — Volatility",     0.01, 2.0,  0.10, 0.01, key="bt_sigma",format="%.2f")
    bt_T     = st.number_input("⏳ T  — Years to Expiry", min_value=0.01,value=2.0, step=0.25, key="bt_T",    format="%.2f")
    bt_r     = st.slider(      "🏦 r  — Risk-Free Rate",  0.0,  0.30, 0.10, 0.005,key="bt_r",  format="%.3f")
    bt_N     = st.select_slider("🌿 Tree Steps (N)",
                                options=[50,100,200,500,1000],
                                value=100, key="bt_N")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bt-btn'>", unsafe_allow_html=True)
    run_bt_btn = st.button("🌳 Price with Binomial Tree", key="run_bt", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── RUN ALL BUTTON ───────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, mid_col, _ = st.columns([2,3,2])
with mid_col:
    st.markdown("<div class='all-btn'>", unsafe_allow_html=True)
    run_all = st.button("⚡ COMPARE ALL 3 MODELS", key="run_all", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  RESULTS
# ════════════════════════════════════════════════════════════
def result_card(label, value, theme, hint=""):
    return f"""
    <div class='result-card' style='background:{theme["mid"]};border:1px solid {theme["border"]};
                box-shadow:0 0 12px {theme["glow"]}'>
      <div style='color:{theme["primary"]};font-size:10px;letter-spacing:1px;margin-bottom:6px'>{label}</div>
      <div style='color:{theme["light"]};font-size:20px;font-weight:bold'>{value:.5f}</div>
      {"<div style='color:#444;font-size:9px;margin-top:4px'>"+hint+"</div>" if hint else ""}
    </div>"""

def show_results(res, theme, method_name):
    m = theme["primary"]
    st.markdown(f"""
    <div style='background:{theme["dark"]};border:1px solid {theme["border"]};
                border-radius:10px;padding:16px;box-shadow:0 0 30px {theme["glow"]};
                margin-bottom:8px'>
      <div style='color:{m};font-family:monospace;font-weight:bold;font-size:15px;
                  margin-bottom:12px;border-bottom:1px solid {theme["border"]};padding-bottom:8px'>
        {theme["emoji"]} {method_name} — RESULTS
      </div>
      <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px'>
        {result_card("d1", res["d1"], theme, "N(d1) prob factor")}
        {result_card("d2", res["d2"], theme, "Risk-neutral prob")}
        {result_card("DIGITAL CALL", res["dc"], theme, "Binary call")}
        {result_card("SHARE CALL",   res["sc"], theme, "Vanilla call")}
        {result_card("DIGITAL PUT",  res["dp"], theme, "Binary put")}
        {result_card("SHARE PUT",    res["sp"], theme, "Vanilla put")}
      </div>
    </div>""", unsafe_allow_html=True)

def payoff_chart(res_bs, res_mc, res_bt, St, K, sigma, T, r):
    spots = np.linspace(St*0.5, St*1.8, 300)
    sc=[]; sp=[]; dc=[]; dp=[]
    for s in spots:
        try:
            d1,d2=bs_params(s,K,r,sigma,T)
            sc.append(s*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2))
            sp.append(K*math.exp(-r*T)*norm.cdf(-d2)-s*norm.cdf(-d1))
            dc.append(math.exp(-r*T)*norm.cdf(d2))
            dp.append(math.exp(-r*T)*norm.cdf(-d2))
        except: sc.append(0);sp.append(0);dc.append(0);dp.append(0)

    fig=make_subplots(rows=1,cols=2,
        subplot_titles=["📈 Share Options (Vanilla)","🎯 Digital Options (Binary)"])
    fig.add_trace(go.Scatter(x=spots,y=sc,name="Share Call",
        line=dict(color=BS["primary"],width=2.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=spots,y=sp,name="Share Put",
        line=dict(color=MC["primary"],width=2.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=spots,y=dc,name="Digital Call",
        line=dict(color=BT["primary"],width=2.5)),row=1,col=2)
    fig.add_trace(go.Scatter(x=spots,y=dp,name="Digital Put",
        line=dict(color="#F59E0B",width=2.5)),row=1,col=2)

    # Plot current model results as dots
    for col in [1,2]:
        fig.add_vline(x=K, line=dict(color="#FFD700",dash="dot",width=1.5),row=1,col=col)
        fig.add_vline(x=St,line=dict(color="#555",   dash="dot",width=1.0),row=1,col=col)

    fig.update_layout(
        template="plotly_dark",paper_bgcolor="#050505",plot_bgcolor="#0D0D0D",
        height=420,
        legend=dict(orientation="h",x=0,y=1.1,font=dict(family="Courier New",size=11),
                    bgcolor="rgba(0,0,0,0.5)"),
        font=dict(family="Courier New",color="#777"),
        margin=dict(l=50,r=20,t=60,b=30))
    fig.update_xaxes(gridcolor="#111",title_text="Spot Price (S)")
    fig.update_yaxes(gridcolor="#111",title_text="Option Price")
    return fig

def comparison_chart(bs_r, mc_r, bt_r):
    cats=["Digital Call","Share Call","Digital Put","Share Put"]
    bs_vals=[bs_r["dc"],bs_r["sc"],bs_r["dp"],bs_r["sp"]]
    mc_vals=[mc_r["dc"],mc_r["sc"],mc_r["dp"],mc_r["sp"]]
    bt_vals=[bt_r["dc"],bt_r["sc"],bt_r["dp"],bt_r["sp"]]
    fig=go.Figure()
    fig.add_trace(go.Bar(name="🔮 Black-Scholes",x=cats,y=bs_vals,
        marker_color=BS["primary"],opacity=0.9))
    fig.add_trace(go.Bar(name="🎲 Monte Carlo",  x=cats,y=mc_vals,
        marker_color=MC["primary"],opacity=0.9))
    fig.add_trace(go.Bar(name="🌳 Binomial Tree",x=cats,y=bt_vals,
        marker_color=BT["primary"],opacity=0.9))
    fig.update_layout(
        barmode="group",template="plotly_dark",
        paper_bgcolor="#050505",plot_bgcolor="#0D0D0D",
        height=420,
        title=dict(text="⚡ Model Comparison — Option Prices",
                   font=dict(family="Courier New",size=14,color="#FFF"),x=0),
        legend=dict(orientation="h",x=0,y=1.12,font=dict(family="Courier New",size=11),
                    bgcolor="rgba(0,0,0,0.5)"),
        font=dict(family="Courier New",color="#777"),
        margin=dict(l=50,r=20,t=70,b=30))
    fig.update_xaxes(gridcolor="#111"); fig.update_yaxes(gridcolor="#111")
    return fig

def mc_hist(ST, K, St, theme):
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=ST,nbinsx=100,
        marker=dict(color=theme["primary"],opacity=0.75,
                    line=dict(color=theme["border"],width=0.3)),name="Simulated ST"))
    fig.add_vline(x=K, line=dict(color="#FFD700",dash="dot",width=2),
                  annotation_text=f"K={K}",annotation_font_color="#FFD700",annotation_font_size=11)
    fig.add_vline(x=St,line=dict(color="#aaa",   dash="dot",width=1.5),
                  annotation_text=f"St={St}",annotation_font_color="#aaa",annotation_font_size=11)
    itm=np.mean(ST>K)*100
    fig.add_annotation(text=f"P(ITM Call) = {itm:.1f}%",
        xref="paper",yref="paper",x=0.97,y=0.95,showarrow=False,
        font=dict(color=BT["primary"],size=13,family="Courier New"),align="right")
    fig.update_layout(
        template="plotly_dark",paper_bgcolor="#050505",plot_bgcolor="#0D0D0D",
        height=380,
        title=dict(text="🎲 Monte Carlo — Simulated Terminal Price Distribution",
                   font=dict(family="Courier New",size=13,color=MC["primary"]),x=0),
        font=dict(family="Courier New",color="#777"),
        margin=dict(l=50,r=20,t=55,b=30),
        xaxis=dict(gridcolor="#111",title="Terminal Price ST"),
        yaxis=dict(gridcolor="#111",title="Frequency"))
    return fig

def greeks_chart(St,K,sigma,T,r):
    spots=np.linspace(St*0.5,St*1.5,200)
    dc_=[]; dp_=[]; gc_=[]; vc_=[]
    for s in spots:
        try:
            d1,d2=bs_params(s,K,r,sigma,T)
            dc_.append(math.exp(0)*norm.cdf(d1)); dp_.append(-norm.cdf(-d1))
            gc_.append(norm.pdf(d1)/(s*sigma*math.sqrt(T)))
            vc_.append(s*norm.pdf(d1)*math.sqrt(T)/100)
        except: dc_.append(0);dp_.append(0);gc_.append(0);vc_.append(0)
    fig=make_subplots(rows=1,cols=2,subplot_titles=["Δ Delta","Γ Gamma  &  ν Vega"])
    fig.add_trace(go.Scatter(x=spots,y=dc_,name="Δ Call",
        line=dict(color=BS["primary"],width=2.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=spots,y=dp_,name="Δ Put",
        line=dict(color=MC["primary"],width=2.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=spots,y=gc_,name="Γ Gamma",
        line=dict(color=BT["primary"],width=2.5)),row=1,col=2)
    fig.add_trace(go.Scatter(x=spots,y=vc_,name="ν Vega",
        line=dict(color="#F59E0B",width=2.5,dash="dash")),row=1,col=2)
    for c in [1,2]:
        fig.add_vline(x=K, line=dict(color="#FFD700",dash="dot",width=1.5),row=1,col=c)
        fig.add_vline(x=St,line=dict(color="#555",   dash="dot",width=1.0),row=1,col=c)
    fig.update_layout(
        template="plotly_dark",paper_bgcolor="#050505",plot_bgcolor="#0D0D0D",
        height=400,
        legend=dict(orientation="h",x=0,y=1.12,font=dict(family="Courier New",size=11),
                    bgcolor="rgba(0,0,0,0.5)"),
        font=dict(family="Courier New",color="#777"),
        margin=dict(l=50,r=20,t=60,b=30))
    fig.update_xaxes(gridcolor="#111",title_text="Spot Price")
    fig.update_yaxes(gridcolor="#111")
    return fig

# ── COMPUTE & DISPLAY ────────────────────────────────────────
res_bs_data = res_mc_data = res_bt_data = None

if run_bs_btn or run_all:
    try: res_bs_data = run_bs(bs_St,bs_K,bs_sigma,bs_T,bs_r,bs_q)
    except Exception as e: st.error(f"🔮 Black-Scholes Error: {e}")

if run_mc_btn or run_all:
    try: res_mc_data = run_mc(mc_St,mc_K,mc_sigma,mc_T,mc_r,mc_reps)
    except Exception as e: st.error(f"🎲 Monte Carlo Error: {e}")

if run_bt_btn or run_all:
    try: res_bt_data = run_bt(bt_St,bt_K,bt_sigma,bt_T,bt_r,bt_N)
    except Exception as e: st.error(f"🌳 Binomial Tree Error: {e}")

# Show individual results
any_result = res_bs_data or res_mc_data or res_bt_data
if any_result:
    r1, r2, r3 = st.columns(3)
    with r1:
        if res_bs_data: show_results(res_bs_data, BS, "Black-Scholes")
    with r2:
        if res_mc_data: show_results(res_mc_data, MC, "Monte Carlo")
    with r3:
        if res_bt_data: show_results(res_bt_data, BT, "Binomial Tree")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 COMPARISON","📈 PAYOFF CHARTS","🎲 MC DISTRIBUTION","📐 GREEKS"])

    with tab1:
        if res_bs_data and res_mc_data and res_bt_data:
            st.plotly_chart(comparison_chart(res_bs_data,res_mc_data,res_bt_data),
                            use_container_width=True)
            comp=pd.DataFrame({
                "Method":       ["🔮 Black-Scholes","🎲 Monte Carlo","🌳 Binomial Tree"],
                "Digital Call": [f"{res_bs_data['dc']:.5f}",f"{res_mc_data['dc']:.5f}",f"{res_bt_data['dc']:.5f}"],
                "Share Call":   [f"{res_bs_data['sc']:.5f}",f"{res_mc_data['sc']:.5f}",f"{res_bt_data['sc']:.5f}"],
                "Digital Put":  [f"{res_bs_data['dp']:.5f}",f"{res_mc_data['dp']:.5f}",f"{res_bt_data['dp']:.5f}"],
                "Share Put":    [f"{res_bs_data['sp']:.5f}",f"{res_mc_data['sp']:.5f}",f"{res_bt_data['sp']:.5f}"],
            })
            st.dataframe(comp.set_index("Method"), use_container_width=True)
        else:
            st.info("⚡ Click **COMPARE ALL 3 MODELS** to see the full comparison!")

    with tab2:
        ref = res_bs_data or res_mc_data or res_bt_data
        St_ref = bs_St if res_bs_data else (mc_St if res_mc_data else bt_St)
        K_ref  = bs_K  if res_bs_data else (mc_K  if res_mc_data else bt_K)
        sig_ref= bs_sigma if res_bs_data else (mc_sigma if res_mc_data else bt_sigma)
        T_ref  = bs_T  if res_bs_data else (mc_T  if res_mc_data else bt_T)
        r_ref  = bs_r  if res_bs_data else (mc_r  if res_mc_data else bt_r)
        st.plotly_chart(payoff_chart(res_bs_data,res_mc_data,res_bt_data,
                                     St_ref,K_ref,sig_ref,T_ref,r_ref),
                        use_container_width=True)

    with tab3:
        if res_mc_data and "ST" in res_mc_data:
            st.plotly_chart(mc_hist(res_mc_data["ST"],mc_K,mc_St,MC),
                            use_container_width=True)
            st.markdown(f"""
            <div style='font-family:Courier New,monospace;color:#555;font-size:11px;
                         padding:8px 12px;background:{MC['dark']};border-radius:6px;
                         border-left:3px solid {MC["primary"]}'>
              🎲 Simulations: <b style='color:{MC["light"]}'>{res_mc_data['reps']:,}</b>&nbsp;&nbsp;|&nbsp;&nbsp;
              Seed: 42 (reproducible)&nbsp;&nbsp;|&nbsp;&nbsp;
              P(ITM Call): <b style='color:{BT["primary"]}'>{np.mean(res_mc_data['ST']>mc_K):.2%}</b>&nbsp;&nbsp;|&nbsp;&nbsp;
              P(ITM Put): <b style='color:#F59E0B'>{np.mean(res_mc_data['ST']<mc_K):.2%}</b>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("🎲 Run Monte Carlo to see the simulation distribution!")

    with tab4:
        St_g = bs_St if res_bs_data else (mc_St if res_mc_data else bt_St)
        K_g  = bs_K  if res_bs_data else (mc_K  if res_mc_data else bt_K)
        sig_g= bs_sigma if res_bs_data else (mc_sigma if res_mc_data else bt_sigma)
        T_g  = bs_T  if res_bs_data else (mc_T  if res_mc_data else bt_T)
        r_g  = bs_r  if res_bs_data else (mc_r  if res_mc_data else bt_r)
        st.plotly_chart(greeks_chart(St_g,K_g,sig_g,T_g,r_g), use_container_width=True)
        st.markdown(f"""
        <div style='display:flex;gap:12px;flex-wrap:wrap;font-family:Courier New,monospace;font-size:11px;padding:8px 0'>
          <span style='color:{BS["primary"]};background:{BS["dark"]};padding:5px 10px;border-radius:5px;border:1px solid {BS["border"]}'>Δ Call — spot sensitivity</span>
          <span style='color:{MC["primary"]};background:{MC["dark"]};padding:5px 10px;border-radius:5px;border:1px solid {MC["border"]}'>Δ Put — spot sensitivity</span>
          <span style='color:{BT["primary"]};background:{BT["dark"]};padding:5px 10px;border-radius:5px;border:1px solid {BT["border"]}'>Γ Gamma — delta rate of change</span>
          <span style='color:#F59E0B;background:#1C1400;padding:5px 10px;border-radius:5px;border:1px solid #92400E'>ν Vega — volatility sensitivity (per 1%)</span>
        </div>""", unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div style='text-align:center;padding:50px 20px;font-family:Courier New,monospace;
                background:linear-gradient(135deg,{BS["dark"]},{MC["dark"]},{BT["dark"]});
                border-radius:12px;border:1px solid #333;margin-top:10px'>
      <div style='font-size:48px;margin-bottom:12px'>⚡</div>
      <div style='font-size:18px;font-weight:bold;
                  background:linear-gradient(135deg,{BS["primary"]},{MC["primary"]},{BT["primary"]});
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        Set your parameters and click a pricing button above
      </div>
      <div style='color:#444;font-size:12px;margin-top:8px'>
        🔮 Black-Scholes &nbsp;·&nbsp; 🎲 Monte Carlo &nbsp;·&nbsp; 🌳 Binomial Tree
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;font-family:Courier New,monospace;font-size:10px;
             color:#222;padding:20px 0;margin-top:20px;border-top:1px solid #111'>
  ⚡ Options Pricer Pro · Black-Scholes · Monte Carlo · Binomial Tree · Built with Python + Streamlit
</div>""", unsafe_allow_html=True)
