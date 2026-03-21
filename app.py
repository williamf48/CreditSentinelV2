import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json, os, joblib

st.set_page_config(layout="wide", page_title="CreditSentinel Ensemble")

@st.cache_resource
def load_models():
    xgb_mod = xgb.XGBClassifier()
    xgb_mod.load_model("creditsentinel_xgb.json")
    rf_mod  = joblib.load("creditsentinel_rf.joblib")
    lr_pipe = joblib.load("creditsentinel_lr.joblib")
    return xgb_mod, rf_mod, lr_pipe

xgb_model, rf_model, lr_pipeline = load_models()

def load_data():
    if os.path.exists("stock_cache.json"):
        with open("stock_cache.json") as f:
            return json.load(f)
    return {}

cache = load_data()

# ── SIDEBAR ──────────────────────────────────────────────
st.sidebar.title("CreditSentinel Terminal")
st.sidebar.caption("Ensemble Credit Risk Analytics")
st.sidebar.divider()

ticker_list     = ["Manual Entry"] + sorted(cache.keys())
selected_ticker = st.sidebar.selectbox("Select Issuer / Ticker:", ticker_list)

d_val, i_val, p_val, c_val = 3.5, 2.0, 0.0, 1.2
full_name = "Manual Entry"
mkt_cap = high_52 = low_52 = 0
sec_val   = "General Industry"
hist_data = []
rev_data  = []

if selected_ticker != "Manual Entry":
    data      = cache[selected_ticker]
    d_val     = float(data['debt_ebitda'])
    i_val     = float(data['int_cov'])
    p_val     = float(data['pledged'])
    c_val     = float(data['curr_ratio'])
    full_name = data.get('name', selected_ticker)
    mkt_cap   = data.get('market_cap', 0)
    high_52   = data.get('high_52', 0)
    low_52    = data.get('low_52', 0)
    sec_val   = data.get('sector', "General Industry")
    hist_data = data.get('history', [])
    rev_data  = data.get('revenue', [])
    st.sidebar.success(f"Loaded: {selected_ticker}")

thresholds = {
    "Information Technology": {"safe": 1.5, "risk": 3.0},
    "Utilities":              {"safe": 6.0, "risk": 9.0},
    "Financial":              {"safe":10.0, "risk":12.0},
    "Manufacturing":          {"safe": 3.0, "risk": 5.0},
    "Real Estate":            {"safe": 5.0, "risk": 7.0},
    "Energy":                 {"safe": 4.0, "risk": 6.5},
    "Communication":          {"safe": 4.0, "risk": 7.0},
}
matched_sector = next((k for k in thresholds if sec_val and k in sec_val), "Manufacturing")
sector_limits  = thresholds[matched_sector]

st.sidebar.markdown("### Sector Covenants")
st.sidebar.info(f"**Peer Group:** {matched_sector}")
sc1, sc2 = st.sidebar.columns(2)
sc1.metric("Safe Leverage",  f"<{sector_limits['safe']}x")
sc2.metric("Breach Level",   f">{sector_limits['risk']}x", delta_color="inverse")

st.sidebar.divider()
st.sidebar.markdown("### Sensitivity Analysis")
with st.sidebar.form("analysis"):
    debt_ebitda = st.number_input("Debt / EBITDA (x)",      -100.0, 1e6, d_val)
    int_cov     = st.number_input("Interest Coverage (x)",  -100.0, 1e6, i_val)
    pledged     = st.number_input("Promoter Pledge (%)",       0.0, 100.0, p_val)
    curr_ratio  = st.number_input("Current Ratio (x)",         0.0, 1e6, c_val)
    submitted   = st.form_submit_button("Run Ensemble Model")

# ── MAIN DASHBOARD ────────────────────────────────────────
st.title("CreditSentinel: Ensemble Risk Framework")
st.markdown("---")

def fmt_cap(v):
    return f"Rs. {v/1e7:.2f} Cr" if v >= 1e7 else f"Rs. {v:.2f}"

if submitted or selected_ticker != "Manual Entry":

    inp = pd.DataFrame([[debt_ebitda, int_cov, pledged, curr_ratio]],
                       columns=['Debt_to_EBITDA','Interest_Coverage',
                                'Pledged_Pct','Current_Ratio'])

    p_xgb = xgb_model.predict_proba(inp)[:, 1][0]
    p_rf  = rf_model.predict_proba(inp)[:,  1][0]
    p_lr  = lr_pipeline.predict_proba(inp)[:,1][0]

    base_prob = p_xgb*0.6 + p_rf*0.2 + p_lr*0.2
    risk_adj  = (0.20 if debt_ebitda > sector_limits['risk'] else 0) + \
                (0.10 if pledged > 20 else 0)
    score     = np.clip(base_prob + risk_adj, 0.01, 0.99) * 100

    if   score < 35: color, status = "#00C853", "INVESTMENT GRADE (IG)"
    elif score < 70: color, status = "#FFA000", "SPECULATIVE / HY"
    else:            color, status = "#D50000", "DISTRESSED / DEFAULT"

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown(f"<h2 style='color:{color};margin-bottom:0'>{status}</h2>",
                    unsafe_allow_html=True)
        st.caption(f"Ensemble Default Probability Score: {int(score)}/100")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': "black", 'thickness': 0.3},
                'steps':[{'range':[0,35],'color':"#00C853"},
                          {'range':[35,70],'color':"#FFA000"},
                          {'range':[70,100],'color':"#D50000"}],
                'threshold':{'line':{'color':"black",'width':4},
                             'thickness':0.75,'value':score}
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"### {full_name}")
        st.metric("Market Cap", fmt_cap(mkt_cap))
        ca, cb = st.columns(2)
        ca.metric("52W High", f"Rs. {high_52}")
        cb.metric("52W Low",  f"Rs. {low_52}")
        st.markdown("#### Credit Memo")
        with st.container(border=True):
            if   debt_ebitda <= sector_limits['safe']: st.markdown("- **Leverage:** Conservative vs Peers.")
            elif debt_ebitda >  sector_limits['risk']: st.markdown("- **Leverage:** Breach of Sector Norms.")
            if   int_cov > 5.0: st.markdown("- **Coverage:** Strong Serviceability.")
            elif int_cov < 2.0: st.markdown("- **Coverage:** Cash Flow Stress.")
            if   pledged == 0:  st.markdown("- **Structure:** Clean Equity.")
            else:               st.markdown(f"- **Governance:** {pledged}% Shares Encumbered.")

    st.markdown("#### Algorithmic Audit Trail")
    m1, m2, m3 = st.columns(3)
    m1.info(f"**Champion Model**\n\nBase Score: **{p_xgb*100:.1f}%**\n\n*Weight: 60%*")
    m2.info(f"**Stabilizer Model**\n\nBase Score: **{p_rf*100:.1f}%**\n\n*Weight: 20%*")
    m3.info(f"**Baseline Auditor**\n\nBase Score: **{p_lr*100:.1f}%**\n\n*Weight: 20%*")

    st.markdown("---")
    ch1, ch2 = st.columns(2)

    with ch1:
        if hist_data:
            df_h = pd.DataFrame(hist_data)
            df_h['date'] = pd.to_datetime(df_h['date'])
            df_h = df_h.sort_values('date')
            fig2 = px.line(df_h, x='date', y='debt_ebitda',
                           title="Historical Net Debt / EBITDA", markers=True)
            fig2.add_hline(y=sector_limits['risk'], line_dash="dash",
                           line_color="red", annotation_text="Covenant Breach")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No historical leverage data.")

    with ch2:
        if rev_data:
            df_r = pd.DataFrame(rev_data).sort_values("year")
            fig3 = px.bar(df_r, x='year', y='revenue', title="Annual Revenue",
                          color_discrete_sequence=['#1f77b4'])
            fig3.update_layout(xaxis_type='category')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No revenue data available.")