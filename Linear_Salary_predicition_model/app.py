"""
================================================================
 Model 1 : Linear Regression — Salary Prediction
 File    : app.py  (Frontend — Streamlit UI)
 Project : Job Market & Skill Demand Analysis
================================================================

 HOW TO RUN  (open 2 terminals in the same folder)
 --------------------------------------------------
 Terminal 1 — Backend:
      pip install pandas scikit-learn flask flask-cors
      python model_backend.py

 Terminal 2 — Frontend:
      pip install streamlit requests plotly
      streamlit run app.py

 Then open:  http://localhost:8501
================================================================
"""

import streamlit as st
import requests
import numpy     as np
import pandas    as pd
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Salary Predictor · Model 1",
    page_icon  = "💼",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

BACKEND = "http://localhost:5000"

# ================================================================
#  CUSTOM CSS
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
}

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 14px;
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 1.8rem;
    border-bottom: 3px solid #00d4ff;
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header p {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #8ab4c2;
    margin: 0.4rem 0 0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.accent { color: #00d4ff; }

/* ── Metric tiles ── */
.metric-row { display:flex; gap:0.8rem; margin-bottom:1rem; flex-wrap:wrap; }
.tile {
    flex:1; min-width:90px;
    background:#111827;
    border:1px solid #1f2937;
    border-radius:10px;
    padding:1rem 0.8rem;
    text-align:center;
}
.tile .val {
    font-size:1.55rem; font-weight:800;
    color:#00d4ff; line-height:1;
}
.tile .lbl {
    font-family:'DM Mono',monospace;
    font-size:0.62rem; color:#6b7280;
    letter-spacing:0.1em; text-transform:uppercase;
    margin-top:0.3rem;
}

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, #0d1b2a, #1b3a4b);
    border: 2px solid #00d4ff;
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin: 0.5rem 0 1.2rem;
}
.result-val  { font-size:3.5rem; font-weight:800; color:#00d4ff; line-height:1; }
.result-unit { font-family:'DM Mono',monospace; font-size:0.72rem; color:#6b7280;
               letter-spacing:0.1em; margin-top:0.4rem; }
.result-text { font-size:0.92rem; color:#d1d5db; margin-top:1rem; line-height:1.6; }

/* ── Code box ── */
.eq-box {
    background:#070f17;
    border-left:3px solid #00d4ff;
    border-radius:0 8px 8px 0;
    padding:0.9rem 1rem;
    font-family:'DM Mono',monospace;
    font-size:0.74rem; color:#6b7280;
    line-height:1.9; word-break:break-all;
    margin-top:0.6rem;
}

/* ── Section label ── */
.sec-label {
    font-family:'DM Mono',monospace;
    font-size:0.65rem; color:#00d4ff;
    letter-spacing:0.14em; text-transform:uppercase;
    margin-bottom:0.5rem; margin-top:0.2rem;
}

/* ── Streamlit component overrides ── */
[data-testid="stSidebar"] { background:#080e18 !important; }
.stSlider > div > div > div > div { background:#00d4ff !important; }
.stButton > button {
    background: #00d4ff !important; color: #060d14 !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.65rem 1rem !important;
    width: 100%; letter-spacing: 0.02em;
}
.stButton > button:hover { background: #33dcff !important; }
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"]    label,
div[data-testid="stNumberInput"] label {
    font-family:'DM Mono',monospace !important;
    font-size:0.7rem !important; color:#6b7280 !important;
    letter-spacing:0.08em !important; text-transform:uppercase !important;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
#  HELPERS
# ================================================================
@st.cache_data(ttl=30)
def fetch_model_info():
    try:
        r = requests.get(f"{BACKEND}/model_info", timeout=4)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def call_predict(experience, skills_count, location, job_category):
    try:
        r = requests.post(
            f"{BACKEND}/predict",
            json={"experience": experience, "skills_count": skills_count,
                  "location": location, "job_category": job_category},
            timeout=8,
        )
        return r.json(), r.status_code
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is model_backend.py running on port 5000?"}, 503
    except Exception as e:
        return {"error": str(e)}, 500


# ================================================================
#  BACKEND STATUS CHECK
# ================================================================
try:
    hc = requests.get(f"{BACKEND}/health", timeout=3)
    backend_ok = hc.status_code == 200
except Exception:
    backend_ok = False

# ================================================================
#  SIDEBAR — inputs
# ================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:0.8rem 0 1.4rem;'>
        <div style='font-family:DM Mono,monospace;font-size:0.65rem;
                    color:#00d4ff;letter-spacing:0.14em;text-transform:uppercase;
                    margin-bottom:0.3rem;'>Model 1</div>
        <div style='font-size:1.25rem;font-weight:800;color:#f9fafb;'>Salary Predictor</div>
        <div style='font-family:DM Mono,monospace;font-size:0.7rem;
                    color:#6b7280;margin-top:0.2rem;'>Linear Regression</div>
    </div>
    """, unsafe_allow_html=True)

    # Backend status badge
    if backend_ok:
        st.markdown("<div style='background:#052e16;border:1px solid #166534;border-radius:6px;"
                    "padding:0.4rem 0.8rem;font-family:DM Mono,monospace;font-size:0.7rem;"
                    "color:#4ade80;'>● Backend connected</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:#2d0707;border:1px solid #7f1d1d;border-radius:6px;"
                    "padding:0.4rem 0.8rem;font-family:DM Mono,monospace;font-size:0.7rem;"
                    "color:#f87171;'>● Backend offline</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Input Parameters</div>', unsafe_allow_html=True)

    experience = st.slider("Experience (years)", 0.0, 20.0, 3.0, 0.5)
    skills_count = st.slider("Skills count", 1, 20, 3, 1)

    info = fetch_model_info()
    categories = info.get("valid_job_categories", [
        "Entry-Level Analyst", "Mid-Level Specialist", "Senior Leadership"
    ]) if info else ["Entry-Level Analyst", "Mid-Level Specialist", "Senior Leadership"]

    location     = st.selectbox("Location / Market", ["india", "global"])
    job_category = st.selectbox("Job Category", categories, index=1)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Predict Salary")

    st.markdown("""
    <div style='margin-top:1.5rem;font-family:DM Mono,monospace;
                font-size:0.62rem;color:#374151;line-height:1.9;'>
    ALGORITHM<br>LinearRegression<br><br>
    LIBRARY<br>sklearn.linear_model<br><br>
    DATASET<br>India + Global Jobs<br>77,837 records
    </div>
    """, unsafe_allow_html=True)


# ================================================================
#  MAIN PAGE
# ================================================================
st.markdown("""
<div class="main-header">
    <h1>💼 Salary <span class="accent">Prediction</span></h1>
    <p>Model 1 &nbsp;·&nbsp; Linear Regression &nbsp;·&nbsp; Job Market &amp; Skill Demand Analysis</p>
</div>
""", unsafe_allow_html=True)

if not backend_ok:
    st.error("⚠️  Backend is not running. Open a terminal and run:")
    st.code("python model_backend.py", language="bash")
    st.info("Once the backend starts, refresh this page.")
    st.stop()

# ── Two-column layout ─────────────────────────────────────────
left, right = st.columns([1.05, 0.95], gap="large")

# ================================================================
#  LEFT — Prediction panel
# ================================================================
with left:
    st.markdown('<div class="sec-label">Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn or "result" in st.session_state:

        if predict_btn:
            with st.spinner("Running model..."):
                result, status = call_predict(
                    experience, skills_count, location, job_category
                )
            st.session_state["result"] = result
            st.session_state["inputs"] = dict(
                experience=experience, skills_count=skills_count,
                location=location, job_category=job_category
            )

        result = st.session_state.get("result", {})

        if "predicted_salary" in result:
            sal   = result["predicted_salary"]
            loc   = st.session_state["inputs"]["location"]
            disp  = f"₹{sal:.2f}L" if loc == "india" else f"{sal:.2f}"
            unit  = "₹ LAKHS PER ANNUM  (INDIA)" if loc == "india" else "SKILL-DEMAND SALARY INDEX  (GLOBAL)"

            st.markdown(f"""
            <div class="result-box">
                <div class="result-val">{disp}</div>
                <div class="result-unit">{unit}</div>
                <div class="result-text">{result.get('interpretation','')}</div>
            </div>
            """, unsafe_allow_html=True)

            # Equation
            st.markdown('<div class="sec-label">Model Equation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="eq-box">{result.get("model_equation","")}</div>',
                        unsafe_allow_html=True)

            # Inputs table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sec-label">Encoded Inputs</div>', unsafe_allow_html=True)
            inp = result.get("inputs_used", {})
            st.dataframe(
                pd.DataFrame([{"Feature": k, "Value": v} for k, v in inp.items()]),
                hide_index=True, use_container_width=True,
            )

        elif "error" in result:
            st.error(f"❌  {result['error']}")
        else:
            st.warning("Unexpected response from backend.")

    else:
        # ── Scenario explorer (shown before first predict) ───
        st.markdown("""
        <div style='background:#0d1117;border:1px dashed #1f2937;border-radius:12px;
                    padding:2.5rem 2rem;text-align:center;margin-bottom:1.2rem;'>
            <div style='font-size:2rem;margin-bottom:0.8rem;'>💡</div>
            <div style='font-family:DM Mono,monospace;font-size:0.82rem;color:#6b7280;'>
                Set parameters in the sidebar<br>then click
                <span style='color:#00d4ff;'>⚡ Predict Salary</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-label">Quick Scenario Explorer</div>', unsafe_allow_html=True)
        scenarios = [
            {"label":"Fresher (India)",       "experience":1.0,"skills_count":2,"location":"india", "job_category":"Entry-Level Analyst"},
            {"label":"3yr Dev (India)",        "experience":3.0,"skills_count":5,"location":"india", "job_category":"Mid-Level Specialist"},
            {"label":"5yr Specialist (India)", "experience":5.0,"skills_count":7,"location":"india", "job_category":"Mid-Level Specialist"},
            {"label":"10yr Senior (India)",    "experience":10.0,"skills_count":10,"location":"india","job_category":"Senior Leadership"},
        ]
        rows = []
        for sc in scenarios:
            r, _ = call_predict(sc["experience"], sc["skills_count"],
                                sc["location"], sc["job_category"])
            rows.append({
                "Scenario"  : sc["label"],
                "Exp (yrs)" : sc["experience"],
                "Skills"    : sc["skills_count"],
                "Predicted ₹L" : r.get("predicted_salary", "—"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ================================================================
#  RIGHT — Model stats & charts
# ================================================================
with right:
    info = fetch_model_info()

    if info:
        tm = info.get("test_metrics", {})
        rm = info.get("train_metrics", {})

        # ── Performance tiles ─────────────────────────────
        st.markdown('<div class="sec-label">Model Performance</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="tile">
                <div class="val">{tm.get('r2','—')}</div>
                <div class="lbl">R² (test)</div>
            </div>
            <div class="tile">
                <div class="val">{tm.get('mae','—')}</div>
                <div class="lbl">MAE (test)</div>
            </div>
            <div class="tile">
                <div class="val">{tm.get('rmse','—')}</div>
                <div class="lbl">RMSE (test)</div>
            </div>
            <div class="tile">
                <div class="val">{rm.get('r2','—')}</div>
                <div class="lbl">R² (train)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Coefficients bar chart ────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Feature Coefficients</div>', unsafe_allow_html=True)
        coefs      = info.get("coefficients", {})
        feat_names = [k for k in coefs if k != "intercept"]
        feat_vals  = [coefs[k] for k in feat_names]
        colors     = ["#00d4ff" if v >= 0 else "#f87171" for v in feat_vals]

        fig_bar = go.Figure(go.Bar(
            x=feat_vals, y=feat_names, orientation='h',
            marker_color=colors,
            text=[f"{v:+.4f}" for v in feat_vals],
            textposition='outside',
            textfont=dict(color='#d1d5db', size=11),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#6b7280", family="DM Mono", size=11),
            margin=dict(l=10, r=70, t=10, b=10), height=200,
            xaxis=dict(gridcolor="#111827", zerolinecolor="#374151"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Salary vs Experience line chart ───────────────
        st.markdown('<div class="sec-label">Predicted Salary by Experience</div>', unsafe_allow_html=True)
        exp_range = np.arange(0, 21, 1.0)
        palette   = {
            "Entry-Level Analyst" : "#4ade80",
            "Mid-Level Specialist": "#facc15",
            "Senior Leadership"   : "#00d4ff",
        }
        fig_line = go.Figure()
        for cat in info.get("valid_job_categories", []):
            preds = []
            for e in exp_range:
                r, _ = call_predict(float(e), 5, "india", cat)
                preds.append(r.get("predicted_salary", 0))
            fig_line.add_trace(go.Scatter(
                x=exp_range, y=preds, name=cat, mode="lines",
                line=dict(color=palette.get(cat,"#888"), width=2.5),
            ))
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#6b7280", family="DM Mono", size=10),
            margin=dict(l=10, r=10, t=10, b=10), height=260,
            legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#1f2937",
                        borderwidth=1, font=dict(size=10)),
            xaxis=dict(gridcolor="#111827", zerolinecolor="#374151",
                       title=dict(text="Experience (years)", font=dict(size=11))),
            yaxis=dict(gridcolor="#111827", zerolinecolor="#374151",
                       title=dict(text="Predicted Salary (₹L)", font=dict(size=11))),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # ── Model details card ────────────────────────────
        st.markdown(f"""
        <div style='background:#0d1117;border:1px solid #1f2937;border-radius:10px;
                    padding:1.2rem 1.4rem;font-family:DM Mono,monospace;
                    font-size:0.75rem;color:#6b7280;line-height:2;'>
            <div style='color:#00d4ff;font-size:0.62rem;letter-spacing:0.14em;
                        text-transform:uppercase;margin-bottom:0.5rem;'>Model Details</div>
            Algorithm &nbsp;: <span style='color:#f9fafb'>LinearRegression</span><br>
            Library &nbsp;&nbsp;&nbsp;: <span style='color:#f9fafb'>sklearn.linear_model</span><br>
            Intercept &nbsp;: <span style='color:#00d4ff'>{coefs.get('intercept','—')}</span><br>
            Train R² &nbsp;&nbsp;: <span style='color:#f9fafb'>{rm.get('r2','—')}</span><br>
            Train MAE &nbsp;: <span style='color:#f9fafb'>{rm.get('mae','—')}</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Could not load model info from backend.")

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#111827;margin-top:2rem;'>
<div style='text-align:center;font-family:DM Mono,monospace;font-size:0.62rem;
            color:#374151;padding:0.5rem 0 1rem;line-height:2;'>
    MODEL 1 · LINEAR REGRESSION · SALARY PREDICTION &nbsp;|&nbsp;
    FEATURES: EXPERIENCE · SKILLS COUNT · LOCATION · JOB CATEGORY<br>
    India salaries in ₹ Lakhs p.a. · Global values are skill-demand proxy scores
</div>
""", unsafe_allow_html=True)