"""
app.py  —  ExplainSentinel  (Main Analyser Page)
Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import math
from sentiment_classifier import FinBERTClassifier, LABEL_COLORS
from explainer import lime_explain, build_highlight_html

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ExplainSentinel · Analyser",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title   { font-size:2.4rem; font-weight:700; color:#0f172a; margin-bottom:0; }
.subtitle     { font-size:1rem; color:#64748b; margin-bottom:1.8rem; }
.step-badge   { display:inline-block; background:#e0f2fe; color:#0369a1;
                font-size:0.72rem; font-weight:700; padding:2px 8px;
                border-radius:20px; margin-bottom:6px; letter-spacing:.04em; }
.step-card    { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
                padding:1rem 1.2rem; margin-bottom:0.8rem; }
.mono         { font-family: 'Courier New', monospace; font-size:0.88rem;
                background:#1e293b; color:#e2e8f0; padding:0.8rem 1rem;
                border-radius:8px; overflow-x:auto; white-space:pre-wrap; }
.token-pill   { display:inline-block; border-radius:5px; padding:2px 6px;
                margin:2px; font-size:1rem; font-weight:500; }
.badge-pos    { background:#dcfce7; color:#166534; border:1px solid #bbf7d0; }
.badge-neg    { background:#fee2e2; color:#991b1b; border:1px solid #fecaca; }
.badge-neu    { background:#f1f5f9; color:#475569; border:1px solid #e2e8f0; }
.logit-box    { background:#1e293b; color:#e2e8f0; border-radius:8px;
                padding:0.9rem 1.1rem; font-family:monospace; font-size:0.87rem; }
.prob-row     { display:flex; align-items:center; gap:10px; margin:5px 0; }
.prob-bar-bg  { flex:1; background:#334155; border-radius:4px; height:14px; }
.prob-bar-fill{ height:14px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    num_features = st.slider("LIME tokens to highlight", 5, 20, 10)
    num_samples  = st.slider("LIME perturbation samples", 100, 500, 300, step=50,
                             help="More = stable but slower")
    show_internals = st.toggle("🔬 Show full internals", value=True)
    st.markdown("---")
    st.markdown("🏠 [🔍 Analyser](/)")
    st.markdown("📖 [How it works](/explanation)")
    st.markdown("---")
    st.caption("Model: ProsusAI/finbert\nDataset: Financial PhraseBank\nXAI: LIME")

# ── Model load ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading FinBERT (first run ~30s)...")
def load_model():
    return FinBERTClassifier()

clf = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🔍 ExplainSentinel</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Financial sentiment · FinBERT internals · LIME explainability</p>',
            unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
EXAMPLES = [
    "Apple reports record quarterly earnings, crushing analyst estimates.",
    "The bank faces a major liquidity crisis after bond market collapse.",
    "The Federal Reserve held interest rates steady at its March meeting.",
    "NIFTY 50 surges 2.3% as FIIs return amid easing inflation concerns.",
    "Startup files for bankruptcy after failed Series C funding round.",
    "Oil prices stabilize as OPEC agrees to maintain current output levels.",
    "RBI cuts repo rate by 25 bps amid slowing GDP growth.",
    "Tech layoffs accelerate as revenue misses trigger restructuring fears.",
]

col1, col2 = st.columns([3, 1])
with col1:
    user_text = st.text_area("headline", label_visibility="collapsed",
                             placeholder="Type a financial headline…", height=80)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    pick = st.selectbox("example", ["— example —"] + EXAMPLES, label_visibility="collapsed")
    if pick != "— example —":
        user_text = pick

run = st.button("🔍 Analyse", type="primary")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
if run and user_text.strip():
    text = user_text.strip()

    # ── Step 1: Tokenisation ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔬 Under the Hood — Every Step")

    with st.expander("**Step 1 · Tokenisation** — text → token IDs", expanded=show_internals):
        st.markdown('<div class="step-badge">BERT WORDPIECE TOKENIZER</div>', unsafe_allow_html=True)

        tokenizer = clf.tokenizer
        encoding  = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        token_ids  = encoding["input_ids"][0].tolist()
        tokens     = tokenizer.convert_ids_to_tokens(token_ids)
        attn_mask  = encoding["attention_mask"][0].tolist()

        st.markdown(f"**Input text:** `{text}`")
        st.markdown(f"**Sequence length:** `{len(token_ids)}` tokens (max 128)")

        # Token table
        tok_df = pd.DataFrame({
            "Position": list(range(len(tokens))),
            "Token":    tokens,
            "Token ID": token_ids,
            "Attn Mask": attn_mask,
            "Special?": ["✅" if t in ["[CLS]","[SEP]","[PAD]"] else "" for t in tokens],
        })
        st.dataframe(tok_df, use_container_width=True, height=220)

        # Visual token display
        token_html = ""
        for tok, tid in zip(tokens, token_ids):
            if tok == "[CLS]":
                color = "#dbeafe"; tc = "#1d4ed8"
            elif tok == "[SEP]":
                color = "#fef9c3"; tc = "#854d0e"
            elif tok.startswith("##"):
                color = "#fce7f3"; tc = "#9d174d"
            else:
                color = "#f0fdf4"; tc = "#166534"
            token_html += (f'<span class="token-pill" style="background:{color};color:{tc};">'
                           f'{tok}<br><small style="opacity:.6">{tid}</small></span> ')

        st.markdown(f'<div style="line-height:3">{token_html}</div>', unsafe_allow_html=True)
        st.caption("🔵 [CLS] = classification token · 🟡 [SEP] = separator · 🩷 ## = subword continuation · 🟢 regular token")

    # ── Step 2: BERT Encoding & Raw Logits ───────────────────────────────────
    with st.expander("**Step 2 · FinBERT Forward Pass** — logits & softmax", expanded=show_internals):
        st.markdown('<div class="step-badge">FINBERT CLASSIFICATION HEAD</div>', unsafe_allow_html=True)

        import torch, torch.nn.functional as F

        inputs = {k: v.to(clf.device) for k, v in encoding.items()}
        # Force attentions at config level (some transformers versions ignore call kwarg)
        clf.model.config.output_attentions = True
        clf.model.config.output_hidden_states = True
        with torch.no_grad():
            outputs = clf.model(**inputs)
        logits  = outputs.logits[0].cpu()
        probs_t = F.softmax(logits, dim=-1).numpy()
        LABEL_ORDER = ["Positive", "Negative", "Neutral"]
        # FinBERT output order: positive=0, negative=1, neutral=2
        probs_map = {LABEL_ORDER[i]: float(probs_t[i]) for i in range(3)}
        logit_map = {LABEL_ORDER[i]: float(logits[i]) for i in range(3)}

        c1, c2, c3 = st.columns(3)
        for col, lbl in zip([c1, c2, c3], LABEL_ORDER):
            col.metric(f"Logit ({lbl})", f"{logit_map[lbl]:.4f}")

        st.markdown("**Softmax conversion** — `P = exp(logit) / Σ exp(logits)`")

        # Logit → prob visual
        fig_logit = go.Figure()
        fig_logit.add_trace(go.Bar(
            name="Raw Logit", x=LABEL_ORDER,
            y=[logit_map[l] for l in LABEL_ORDER],
            marker_color=["#2ecc71","#e74c3c","#95a5a6"],
            text=[f"{logit_map[l]:.4f}" for l in LABEL_ORDER],
            textposition="outside",
        ))
        fig_logit.update_layout(title="Raw Logits (pre-softmax)",
                                height=280, margin=dict(t=40,b=20,l=20,r=20),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(248,250,252,1)")
        st.plotly_chart(fig_logit, use_container_width=True)

        fig_prob = go.Figure()
        fig_prob.add_trace(go.Bar(
            name="Probability", x=LABEL_ORDER,
            y=[probs_map[l] for l in LABEL_ORDER],
            marker_color=["#2ecc71","#e74c3c","#95a5a6"],
            text=[f"{probs_map[l]:.4f}  ({probs_map[l]:.1%})" for l in LABEL_ORDER],
            textposition="outside",
        ))
        fig_prob.update_layout(title="Probabilities after Softmax",
                               yaxis=dict(range=[0,1.1], tickformat=".0%"),
                               height=280, margin=dict(t=40,b=20,l=20,r=20),
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(248,250,252,1)")
        st.plotly_chart(fig_prob, use_container_width=True)

        pred_label = LABEL_ORDER[int(probs_t.argmax())]
        pred_conf  = float(probs_t.max())
        emoji_map  = {"Positive":"🟢","Negative":"🔴","Neutral":"⚪"}
        st.success(f"**Prediction:** {emoji_map[pred_label]} **{pred_label}**  —  "
                   f"confidence `{pred_conf:.4f}` ({pred_conf:.1%})")

        # Exact numbers table
        st.markdown("**Exact numerical output from FinBERT:**")
        num_df = pd.DataFrame({
            "Label":       LABEL_ORDER,
            "Raw Logit":   [f"{logit_map[l]:.6f}" for l in LABEL_ORDER],
            "exp(logit)":  [f"{math.exp(logit_map[l]):.6f}" for l in LABEL_ORDER],
            "Probability": [f"{probs_map[l]:.6f}" for l in LABEL_ORDER],
            "Percentage":  [f"{probs_map[l]*100:.2f}%" for l in LABEL_ORDER],
            "Predicted?":  ["✅" if l == pred_label else "" for l in LABEL_ORDER],
        })
        st.dataframe(num_df, use_container_width=True, hide_index=True)

    with st.expander("**Step 3 · Attention Weights** — what the model attends to", expanded=show_internals):
        st.markdown('<div class="step-badge">BERT SELF-ATTENTION · LAST LAYER · HEAD 0</div>', unsafe_allow_html=True)
        st.markdown("Each cell shows how much token **i** attends to token **j**. "
                    "Brighter = stronger attention.")

        if outputs.attentions is None or len(outputs.attentions) == 0:
            st.warning("⚠️ Attention weights not available — the model did not return them. "
                       "This can happen with certain transformers versions on Streamlit Cloud.")
        else:
            # Last layer, head 0
            attn = outputs.attentions[-1][0][0].cpu().numpy()  # (seq, seq)
            seq_len = min(len(tokens), 20)
            attn_sub = attn[:seq_len, :seq_len]
            tok_sub  = tokens[:seq_len]

            fig_attn = px.imshow(
                attn_sub,
                x=tok_sub, y=tok_sub,
                color_continuous_scale="Blues",
                aspect="auto",
                title="Self-Attention Heatmap (Layer 12, Head 0)",
                labels=dict(x="Key Token", y="Query Token", color="Weight"),
            )
            fig_attn.update_layout(height=420, margin=dict(t=50,b=20,l=20,r=20))
            st.plotly_chart(fig_attn, use_container_width=True)

            # CLS attention (what CLS looks at — drives classification)
            cls_attn = attn[0, :seq_len]
            fig_cls = go.Figure(go.Bar(
                x=tok_sub, y=cls_attn,
                marker_color=px.colors.sequential.Blues[3:],
                text=[f"{v:.3f}" for v in cls_attn], textposition="outside",
            ))
            fig_cls.update_layout(
                title="[CLS] Token Attention — drives the classification decision",
                xaxis_title="Token", yaxis_title="Attention Weight",
                height=280, margin=dict(t=40,b=20,l=20,r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
            )
            st.plotly_chart(fig_cls, use_container_width=True)

    # ── Step 4: LIME Explainability ───────────────────────────────────────────
    with st.expander("**Step 4 · LIME Explainability** — which words drove the prediction", expanded=show_internals):
        st.markdown('<div class="step-badge">LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS</div>',
                    unsafe_allow_html=True)

        st.markdown(f"""
        LIME works by **perturbing** the input `{num_samples}` times — randomly masking words —
        and re-running FinBERT each time. It then fits a **local linear model** to approximate
        FinBERT's decision boundary around *this specific input*.
        Each word gets a weight showing how much it pushed the prediction toward or away from **{pred_label}**.
        """)

        with st.spinner(f"Running LIME ({num_samples} perturbations)..."):
            lime_result = lime_explain(clf, text, num_features=num_features, num_samples=num_samples)

        token_weights = lime_result["token_weights"]

        # Highlighted text
        highlight_html = build_highlight_html(token_weights)
        st.markdown("**Word-level highlights:**")
        st.markdown(
            f'<div style="font-size:1.15rem;line-height:2.4;padding:0.8rem;'
            f'background:#fff;border-radius:8px;border:1px solid #e2e8f0;">'
            f'{highlight_html}</div>',
            unsafe_allow_html=True,
        )
        st.caption("🟢 Green = pushes toward prediction · 🔴 Red = pushes against · intensity ∝ weight")

        # Weight bar chart
        tokens_l  = [t for t, _ in token_weights]
        weights_l = [w for _, w in token_weights]
        colors_l  = ["#2ecc71" if w > 0 else "#e74c3c" for w in weights_l]

        fig_lime = go.Figure(go.Bar(
            x=weights_l, y=tokens_l, orientation="h",
            marker_color=colors_l,
            text=[f"{w:+.4f}" for w in weights_l], textposition="outside",
        ))
        fig_lime.update_layout(
            title=f"LIME Feature Weights → '{pred_label}' prediction",
            xaxis_title="Weight", height=max(300, 30*len(tokens_l)+80),
            margin=dict(l=20,r=60,t=40,b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_lime, use_container_width=True)

        # LIME weights table
        lime_df = pd.DataFrame({
            "Word":       tokens_l,
            "LIME Weight": [f"{w:+.6f}" for w in weights_l],
            "Direction":  ["✅ Supports" if w > 0 else "❌ Opposes" for w in weights_l],
            "Abs Impact": [f"{abs(w):.4f}" for w in weights_l],
        })
        st.dataframe(lime_df, use_container_width=True, hide_index=True)

        with st.expander("🔬 Full LIME HTML Report"):
            st.components.v1.html(lime_result["lime_exp"].as_html(), height=400, scrolling=True)

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## ✅ Final Result")

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Prediction", f"{emoji_map[pred_label]} {pred_label}")
    rc2.metric("Confidence", f"{pred_conf:.4f}", f"{pred_conf:.1%}")
    rc3.metric("LIME tokens analysed", str(len(token_weights)))

    st.markdown(f"""
    | Field | Value |
    |---|---|
    | Input | `{text}` |
    | Tokens | `{len(token_ids)}` (incl. [CLS], [SEP]) |
    | Logit (Positive) | `{logit_map["Positive"]:.6f}` |
    | Logit (Negative) | `{logit_map["Negative"]:.6f}` |
    | Logit (Neutral) | `{logit_map["Neutral"]:.6f}` |
    | P(Positive) | `{probs_map["Positive"]:.6f}` ({probs_map["Positive"]:.1%}) |
    | P(Negative) | `{probs_map["Negative"]:.6f}` ({probs_map["Negative"]:.1%}) |
    | P(Neutral) | `{probs_map["Neutral"]:.6f}` ({probs_map["Neutral"]:.1%}) |
    | **Final label** | **{pred_label}** |
    | LIME top word | `{token_weights[0][0]}` (weight `{token_weights[0][1]:+.4f}`) |
    """)

elif run:
    st.warning("Please enter a headline first.")

# ── Batch ─────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Batch Analysis"):
    batch_text = st.text_area("batch", height=130, label_visibility="collapsed",
                              placeholder="One headline per line…")
    if st.button("Run Batch", key="batch_btn"):
        lines = [l.strip() for l in batch_text.strip().splitlines() if l.strip()]
        if lines:
            with st.spinner("Classifying…"):
                results = clf.predict(lines)
            df = pd.DataFrame([{
                "Headline": t,
                "Prediction": r["label"],
                "Confidence": f"{r['confidence']:.4f}",
                "P(Positive)": f"{r['probabilities']['Positive']:.4f}",
                "P(Negative)": f"{r['probabilities']['Negative']:.4f}",
                "P(Neutral)":  f"{r['probabilities']['Neutral']:.4f}",
            } for t, r in zip(lines, results)])
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No headlines found.")

st.markdown("""<hr style="margin-top:2rem;">
<p style="text-align:center;color:#94a3b8;font-size:0.8rem;">
ExplainSentinel · Built on <a href="https://github.com/VT69/FinSentinel">FinSentinel</a>
· ProsusAI/finbert · LIME XAI
</p>""", unsafe_allow_html=True)
