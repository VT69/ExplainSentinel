"""
app.py
ExplainSentinel — Streamlit UI
Financial headline sentiment classifier with LIME explainability.
Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
from sentiment_classifier import FinBERTClassifier, LABEL_COLORS
from explainer import lime_explain, build_highlight_html

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ExplainSentinel",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {font-size:2.2rem; font-weight:700; color:#1a1a2e; margin-bottom:0;}
    .subtitle  {font-size:1rem; color:#555; margin-bottom:1.5rem;}
    .card      {background:#f8f9fa; border-radius:10px; padding:1.2rem 1.5rem;
                margin-bottom:1rem; border-left:4px solid #3498db;}
    .positive  {border-left-color:#2ecc71;}
    .negative  {border-left-color:#e74c3c;}
    .neutral   {border-left-color:#95a5a6;}
    .token-box {font-size:1.1rem; line-height:2.2; padding:0.8rem;
                background:#fff; border-radius:8px; border:1px solid #e0e0e0;}
    .legend    {font-size:0.85rem; color:#666;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    num_features = st.slider("LIME: tokens to highlight", 5, 20, 10)
    num_samples = st.slider("LIME: perturbation samples", 100, 500, 300, step=50,
                            help="More samples = more stable explanations, but slower.")
    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown(
        """
        **ExplainSentinel** adds a token-level explainability layer on top of
        [FinBERT](https://huggingface.co/ProsusAI/finbert) using **LIME**.

        - 🟢 Green = pushed prediction **toward** the label
        - 🔴 Red = pushed prediction **against** the label

        Built on top of [FinSentinel](https://github.com/VT69/FinSentinel).
        """
    )
    st.markdown("---")
    st.markdown("**Dataset:** Financial PhraseBank")
    st.markdown("**Model:** ProsusAI/finbert")

# ── Load model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FinBERT model...")
def load_model():
    return FinBERTClassifier()

clf = load_model()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🔍 ExplainSentinel</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Financial headline sentiment classifier · Token-level LIME explainability · Built on FinBERT</p>',
    unsafe_allow_html=True,
)

# ── Example headlines ────────────────────────────────────────────────────────
EXAMPLES = [
    "Apple reports record quarterly earnings, crushing analyst estimates.",
    "The bank faces a major liquidity crisis after bond market collapse.",
    "The Federal Reserve held interest rates steady at its March meeting.",
    "NIFTY 50 surges 2.3% as FIIs return amid easing inflation concerns.",
    "Startup files for bankruptcy after failed Series C funding round.",
    "Oil prices stabilize as OPEC agrees to maintain current output levels.",
]

st.markdown("### 📰 Enter a Financial Headline")
col1, col2 = st.columns([3, 1])

with col1:
    user_text = st.text_area(
        label="headline_input",
        label_visibility="collapsed",
        placeholder="e.g. RBI cuts repo rate by 25 bps amid slowing growth...",
        height=80,
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    example_pick = st.selectbox(
        "Or pick an example",
        ["— select —"] + EXAMPLES,
        label_visibility="collapsed",
    )
    if example_pick != "— select —":
        user_text = example_pick

run_btn = st.button("🔍 Analyse Sentiment", type="primary", use_container_width=False)

# ── Analysis ─────────────────────────────────────────────────────────────────
if run_btn and user_text.strip():
    with st.spinner("Running FinBERT + LIME..."):
        result = lime_explain(clf, user_text.strip(), num_features=num_features, num_samples=num_samples)

    label = result["predicted_label"]
    conf = result["confidence"]
    probs = result["probabilities"]
    token_weights = result["token_weights"]

    st.markdown("---")

    # ── Prediction card ───────────────────────────────────────────────────
    col_pred, col_chart = st.columns([1, 1.6])

    with col_pred:
        css_cls = label.lower()
        emoji = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}[label]
        st.markdown(
            f"""
            <div class="card {css_cls}">
                <div style="font-size:0.85rem;color:#666;margin-bottom:4px;">PREDICTION</div>
                <div style="font-size:2rem;font-weight:700;color:{LABEL_COLORS[label]};">
                    {emoji} {label}
                </div>
                <div style="font-size:1.1rem;margin-top:4px;">
                    Confidence: <strong>{conf:.1%}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_chart:
        fig = go.Figure(
            go.Bar(
                x=list(probs.values()),
                y=list(probs.keys()),
                orientation="h",
                marker_color=[LABEL_COLORS[l] for l in probs.keys()],
                text=[f"{v:.1%}" for v in probs.values()],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Probability Distribution",
            xaxis=dict(range=[0, 1.05], tickformat=".0%"),
            height=200,
            margin=dict(l=0, r=40, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width='stretch')

    # ── Token explanation ─────────────────────────────────────────────────
    st.markdown("### 🧩 Token-Level Explanation (LIME)")
    st.markdown(
        f"Which words **drove** or **opposed** the **{label}** prediction:"
    )

    highlight_html = build_highlight_html(token_weights, threshold=0.0)
    st.markdown(
        f'<div class="token-box">{highlight_html}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="legend">🟢 Green = supports prediction · 🔴 Red = opposes prediction · '
        "Intensity ∝ weight magnitude</p>",
        unsafe_allow_html=True,
    )

    # ── Token weight bar chart ────────────────────────────────────────────
    st.markdown("### 📊 Token Weight Ranking")
    tokens = [t for t, _ in token_weights]
    weights = [w for _, w in token_weights]
    colors = ["#2ecc71" if w > 0 else "#e74c3c" for w in weights]

    fig2 = go.Figure(
        go.Bar(
            x=weights,
            y=tokens,
            orientation="h",
            marker_color=colors,
            text=[f"{w:+.4f}" for w in weights],
            textposition="outside",
        )
    )
    fig2.update_layout(
        title=f"LIME Feature Weights for '{label}' prediction",
        xaxis_title="Weight",
        height=max(300, 30 * len(tokens) + 80),
        margin=dict(l=20, r=60, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,250,1)",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig2, width='stretch')

    # ── Raw LIME HTML export ──────────────────────────────────────────────
    with st.expander("🔬 Full LIME Report (HTML)"):
        lime_html_str = result["lime_exp"].as_html()
        st.components.v1.html(lime_html_str, height=400, scrolling=True)

elif run_btn:
    st.warning("Please enter a headline first.")

# ── Batch mode ───────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Batch Sentiment Analysis (no explainability)"):
    st.markdown("Paste multiple headlines, one per line:")
    batch_input = st.text_area("batch_headlines", height=150, label_visibility="collapsed",
                               placeholder="Apple beats earnings...\nBank collapses after...\n...")
    if st.button("Run Batch", key="batch_btn"):
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        if lines:
            with st.spinner(f"Classifying {len(lines)} headlines..."):
                batch_results = clf.predict(lines)
            import pandas as pd
            df = pd.DataFrame(
                [
                    {
                        "Headline": t,
                        "Sentiment": r["label"],
                        "Confidence": f"{r['confidence']:.1%}",
                        "P(Positive)": f"{r['probabilities']['Positive']:.3f}",
                        "P(Negative)": f"{r['probabilities']['Negative']:.3f}",
                        "P(Neutral)": f"{r['probabilities']['Neutral']:.3f}",
                    }
                    for t, r in zip(lines, batch_results)
                ]
            )
            st.dataframe(df, width='stretch')
        else:
            st.warning("No valid headlines found.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style="margin-top:2rem;">
    <p style="text-align:center;color:#aaa;font-size:0.8rem;">
    ExplainSentinel · Built on <a href="https://github.com/VT69/FinSentinel" target="_blank">FinSentinel</a>
    · Model: ProsusAI/finbert · Explainability: LIME
    </p>
    """,
    unsafe_allow_html=True,
)
