"""
app.py  —  PrismXAI  (Main Analyser Page)
FinBERT + Bi-LSTM Hybrid with Learned Fusion
Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import math
from sentiment_classifier import HybridSentimentClassifier, FinBERTClassifier, LABEL_COLORS
from explainer import lime_explain, build_highlight_html, lstm_token_attribution, build_lstm_attribution_html

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrismXAI · Analyser",
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
.step-badge-purple { display:inline-block; background:#f3e8ff; color:#7c3aed;
                     font-size:0.72rem; font-weight:700; padding:2px 8px;
                     border-radius:20px; margin-bottom:6px; letter-spacing:.04em; }
.step-badge-amber  { display:inline-block; background:#fef3c7; color:#d97706;
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
.fusion-card  { background:linear-gradient(135deg, #eef2ff 0%, #faf5ff 100%);
                border:1px solid #c7d2fe; border-radius:12px;
                padding:1.2rem 1.5rem; margin:0.5rem 0; }
.alpha-badge  { display:inline-block; background:#4f46e5; color:white;
                font-size:0.85rem; font-weight:700; padding:4px 14px;
                border-radius:20px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    num_features = st.slider("LIME tokens to highlight", 5, 20, 8)
    num_samples  = st.slider("LIME perturbation samples", 50, 300, 50, step=50,
                             help="More = stable but slower. 50 is fast for presentations.")
    show_internals = st.toggle("🔬 Show full internals", value=True)
    st.markdown("---")
    st.markdown("🏠 [🔍 Analyser](/)")
    st.markdown("📖 [How it works](/explanation)")
    st.markdown("---")
    st.caption("Model: ProsusAI/finbert + Bi-LSTM\nFusion: Learned α\nDataset: Financial PhraseBank\nXAI: LIME + LSTM Attribution")

# ── Model load ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading Hybrid Model (first run ~30s)...")
def load_hybrid_model():
    return HybridSentimentClassifier()

@st.cache_resource(show_spinner="⏳ Loading FinBERT...")
def load_finbert_model():
    return FinBERTClassifier()

hybrid_clf = load_hybrid_model()
# Also keep a reference for attention extraction (uses the FinBERT inside hybrid)
# For LIME, we pass hybrid_clf so it explains the fused model

# ── Cached heavy computations ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_hybrid_forward_pass(_clf, text: str):
    """Run hybrid forward pass. Cached per input text."""
    result = _clf.predict_detailed(text)
    # Convert numpy arrays to lists for cache serialisation
    if result.get("lstm_hidden_states") is not None:
        result["lstm_hidden_states"] = result["lstm_hidden_states"].tolist()
    return result

@st.cache_data(show_spinner=False)
def run_forward_with_attentions(_clf, text: str):
    """Run FinBERT with attentions — only called when Step 3 is opened. Cached."""
    import torch
    tokenizer = _clf.tokenizer
    encoding  = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs    = {k: v.to(_clf.device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = _clf.finbert(**inputs, output_attentions=True)
    attentions = outputs.attentions
    if attentions is None or len(attentions) == 0:
        return None
    return attentions[-1][0].cpu().numpy().tolist()

@st.cache_data(show_spinner=False)
def cached_lime_explain(_clf, text: str, num_features: int, num_samples: int):
    """Run LIME explanation against the hybrid classifier. Cached."""
    from explainer import lime_explain
    res = lime_explain(_clf, text, num_features=num_features, num_samples=num_samples)
    return {
        "token_weights": res["token_weights"],
    }

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🔍 PrismXAI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">FinBERT + Bi-LSTM Hybrid · Learned Fusion · LIME + LSTM Attribution</p>',
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

    # ── Run cached hybrid inference ──────────────────────────────────────────
    with st.spinner("⏳ Running Hybrid Model..."):
        fwd = run_hybrid_forward_pass(hybrid_clf, text)

    LABEL_ORDER = ["Positive", "Negative", "Neutral"]
    token_ids   = fwd["token_ids"]
    tokens      = fwd["tokens"]
    attn_mask   = fwd["attention_mask"]

    # FinBERT head values
    fb_logits   = fwd["finbert_logits"]
    fb_probs    = fwd["finbert_probs"]

    # LSTM head values (may be None if no weights)
    lstm_logits = fwd.get("lstm_logits")
    lstm_probs  = fwd.get("lstm_probs")

    # Fused values
    fused_probs = fwd["fused_probs"]
    alpha_val   = fwd.get("alpha", 1.0)
    hybrid_on   = fwd.get("hybrid_active", False)

    pred_label  = fwd["label"]
    pred_conf   = fwd["confidence"]
    emoji_map   = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}

    # ── Model status banner ──────────────────────────────────────────────────
    if hybrid_on:
        st.info(f"🧠 **Hybrid mode active** — FinBERT + Bi-LSTM with learned fusion (α = {alpha_val:.4f})")
    else:
        st.warning("⚠️ **FinBERT-only mode** — `lstm_weights.pt` not found. Run `python train_lstm.py` to train the LSTM head.")

    # ── Section header ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔬 Under the Hood — Every Step")

    # ── Step 1: Tokenisation ─────────────────────────────────────────────────
    with st.expander("**Step 1 · Tokenisation** — text → token IDs", expanded=show_internals):
        st.markdown('<div class="step-badge">BERT WORDPIECE TOKENIZER</div>', unsafe_allow_html=True)
        st.markdown(f"**Input text:** `{text}`")
        st.markdown(f"**Sequence length:** `{len(token_ids)}` tokens (max 128)")

        tok_df = pd.DataFrame({
            "Position":  list(range(len(tokens))),
            "Token":     tokens,
            "Token ID":  token_ids,
            "Attn Mask": attn_mask,
            "Special?":  ["✅" if t in ["[CLS]", "[SEP]", "[PAD]"] else "" for t in tokens],
        })
        st.dataframe(tok_df, use_container_width=True, height=220)

        token_html = ""
        for tok, tid in zip(tokens, token_ids):
            if tok == "[CLS]":
                color, tc = "#dbeafe", "#1d4ed8"
            elif tok == "[SEP]":
                color, tc = "#fef9c3", "#854d0e"
            elif tok.startswith("##"):
                color, tc = "#fce7f3", "#9d174d"
            else:
                color, tc = "#f0fdf4", "#166534"
            token_html += (f'<span class="token-pill" style="background:{color};color:{tc};">'
                           f'{tok}<br><small style="opacity:.6">{tid}</small></span> ')
        st.markdown(f'<div style="line-height:3">{token_html}</div>', unsafe_allow_html=True)
        st.caption("🔵 [CLS] = classification token · 🟡 [SEP] = separator · 🩷 ## = subword continuation · 🟢 regular token")

    # ── Step 2: FinBERT + LSTM Forward Pass ──────────────────────────────────
    with st.expander("**Step 2 · FinBERT + LSTM Forward Pass** — dual head inference", expanded=show_internals):
        st.markdown('<div class="step-badge">FINBERT CLASSIFICATION HEAD</div>', unsafe_allow_html=True)

        # FinBERT logits
        c1, c2, c3 = st.columns(3)
        for col, lbl in zip([c1, c2, c3], LABEL_ORDER):
            col.metric(f"FinBERT Logit ({lbl})", f"{fb_logits[LABEL_ORDER.index(lbl)]:.4f}")

        st.markdown("**Softmax conversion** — `P = exp(logit) / Σ exp(logits)`")

        # FinBERT probabilities bar
        fig_fb_prob = go.Figure(go.Bar(
            name="FinBERT P", x=LABEL_ORDER,
            y=[fb_probs[l] for l in LABEL_ORDER],
            marker_color=["#2ecc71", "#e74c3c", "#95a5a6"],
            text=[f"{fb_probs[l]:.4f}  ({fb_probs[l]:.1%})" for l in LABEL_ORDER],
            textposition="outside",
        ))
        fig_fb_prob.update_layout(title="FinBERT Probabilities (after Softmax)",
                                  yaxis=dict(range=[0, 1.15], tickformat=".0%"),
                                  height=280, margin=dict(t=40, b=20, l=20, r=20),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)")
        st.plotly_chart(fig_fb_prob, use_container_width=True)

        fb_pred_idx = int(np.argmax([fb_probs[l] for l in LABEL_ORDER]))
        fb_pred = LABEL_ORDER[fb_pred_idx]
        st.success(f"**FinBERT Prediction:** {emoji_map[fb_pred]} **{fb_pred}**  —  "
                   f"confidence `{fb_probs[fb_pred]:.4f}` ({fb_probs[fb_pred]:.1%})")

        if hybrid_on and lstm_probs is not None:
            st.markdown("---")
            st.markdown('<div class="step-badge-purple">BI-LSTM CLASSIFICATION HEAD</div>', unsafe_allow_html=True)
            st.markdown("The Bi-LSTM reads the full token embedding sequence `(seq_len × 768)` "
                        "from FinBERT's encoder and produces its own 3-class probability distribution.")

            # LSTM logits
            c1, c2, c3 = st.columns(3)
            for col, lbl, idx in zip([c1, c2, c3], LABEL_ORDER, range(3)):
                col.metric(f"LSTM Logit ({lbl})", f"{lstm_logits[idx]:.4f}")

            fig_lstm_prob = go.Figure(go.Bar(
                name="LSTM P", x=LABEL_ORDER,
                y=[lstm_probs[l] for l in LABEL_ORDER],
                marker_color=["#86efac", "#fca5a5", "#cbd5e1"],
                text=[f"{lstm_probs[l]:.4f}  ({lstm_probs[l]:.1%})" for l in LABEL_ORDER],
                textposition="outside",
            ))
            fig_lstm_prob.update_layout(title="Bi-LSTM Probabilities",
                                        yaxis=dict(range=[0, 1.15], tickformat=".0%"),
                                        height=280, margin=dict(t=40, b=20, l=20, r=20),
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)")
            st.plotly_chart(fig_lstm_prob, use_container_width=True)

            lstm_pred_idx = int(np.argmax([lstm_probs[l] for l in LABEL_ORDER]))
            lstm_pred = LABEL_ORDER[lstm_pred_idx]
            st.info(f"**LSTM Prediction:** {emoji_map[lstm_pred]} **{lstm_pred}**  —  "
                    f"confidence `{lstm_probs[lstm_pred]:.4f}` ({lstm_probs[lstm_pred]:.1%})")

        # Numerical summary
        num_data = {
            "Label": LABEL_ORDER,
            "FinBERT Logit": [f"{fb_logits[LABEL_ORDER.index(l)]:.6f}" for l in LABEL_ORDER],
            "FinBERT P": [f"{fb_probs[l]:.6f}" for l in LABEL_ORDER],
        }
        if hybrid_on and lstm_probs:
            num_data["LSTM Logit"] = [f"{lstm_logits[i]:.6f}" for i in range(3)]
            num_data["LSTM P"] = [f"{lstm_probs[l]:.6f}" for l in LABEL_ORDER]
            num_data["Fused P"] = [f"{fused_probs[l]:.6f}" for l in LABEL_ORDER]
        num_df = pd.DataFrame(num_data)
        st.dataframe(num_df, use_container_width=True, hide_index=True)

    # ── Step 3: Attention Heatmap ────────────────────────────────────────────
    with st.expander("**Step 3 · Attention Weights** — what the model attends to", expanded=show_internals):
        st.markdown('<div class="step-badge">BERT SELF-ATTENTION · LAST LAYER · HEAD 0</div>', unsafe_allow_html=True)
        st.markdown("Each cell shows how much token **i** attends to token **j**. Brighter = stronger attention.")

        with st.spinner("Loading attention weights..."):
            attn_data = run_forward_with_attentions(hybrid_clf, text)

        if attn_data is None:
            st.warning("⚠️ Attention weights not available on this environment.")
        else:
            attn = np.array(attn_data[0])
            seq_len  = min(len(tokens), 20)
            attn_sub = attn[:seq_len, :seq_len]
            tok_sub  = tokens[:seq_len]

            fig_attn = px.imshow(attn_sub, x=tok_sub, y=tok_sub,
                                 color_continuous_scale="Blues", aspect="auto",
                                 title="Self-Attention Heatmap (Layer 12, Head 0)",
                                 labels=dict(x="Key Token", y="Query Token", color="Weight"))
            fig_attn.update_layout(height=420, margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_attn, use_container_width=True)

            cls_attn = attn[0, :seq_len]
            fig_cls = go.Figure(go.Bar(
                x=tok_sub, y=cls_attn,
                marker_color=px.colors.sequential.Blues[3:],
                text=[f"{v:.3f}" for v in cls_attn], textposition="outside",
            ))
            fig_cls.update_layout(
                title="[CLS] Token Attention — drives the classification decision",
                xaxis_title="Token", yaxis_title="Attention Weight",
                height=280, margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
            )
            st.plotly_chart(fig_cls, use_container_width=True)

    # ── Step 4: LIME + LSTM Attribution ──────────────────────────────────────
    with st.expander("**Step 4 · LIME + LSTM Attribution** — which words drove the prediction", expanded=show_internals):
        st.markdown('<div class="step-badge">LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS</div>',
                    unsafe_allow_html=True)

        if hybrid_on:
            st.markdown(f"""
            LIME works by **perturbing** the input `{num_samples}` times — randomly masking words —
            and re-running the **hybrid model** (FinBERT + LSTM fused) each time.
            The LIME weights below reflect the **fused system's** behaviour, not just FinBERT alone.
            """)
        else:
            st.markdown(f"""
            LIME works by **perturbing** the input `{num_samples}` times — randomly masking words —
            and re-running FinBERT each time. It fits a **local linear model** to approximate
            the decision boundary around *this specific input*.
            """)

        with st.spinner(f"Running LIME ({num_samples} perturbations)..."):
            lime_result = cached_lime_explain(hybrid_clf, text, num_features, num_samples)

        token_weights = lime_result["token_weights"]
        highlight_html = build_highlight_html(token_weights)

        # ── LIME sub-panel ──────────────────────────────────────────────────
        st.markdown("#### 🍋 LIME Word-Level Highlights")
        st.markdown(
            f'<div style="font-size:1.15rem;line-height:2.4;padding:0.8rem;'
            f'background:#fff;border-radius:8px;border:1px solid #e2e8f0;">'
            f'{highlight_html}</div>',
            unsafe_allow_html=True,
        )
        st.caption("🟢 Green = pushes toward prediction · 🔴 Red = pushes against · intensity ∝ weight")

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
            xaxis_title="Weight", height=max(300, 30 * len(tokens_l) + 80),
            margin=dict(l=20, r=60, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_lime, use_container_width=True)

        # ── LSTM Attribution sub-panel ──────────────────────────────────────
        if hybrid_on and fwd.get("lstm_hidden_states") is not None:
            st.markdown("---")
            st.markdown("#### 🧠 LSTM Token Attribution (L2 Norm of Hidden States)")
            st.markdown(
                "The Bi-LSTM produces a hidden state **h_t** at each token position. "
                "The L2 norm **‖h_t‖** shows how much the LSTM was 'activated' by each token — "
                "a second, independent signal alongside LIME."
            )

            lstm_hidden = np.array(fwd["lstm_hidden_states"])
            attribs = lstm_token_attribution(lstm_hidden, tokens, attn_mask)
            lstm_html = build_lstm_attribution_html(attribs)

            st.markdown(
                f'<div style="font-size:1.15rem;line-height:2.4;padding:0.8rem;'
                f'background:#faf5ff;border-radius:8px;border:1px solid #e9d5ff;">'
                f'{lstm_html}</div>',
                unsafe_allow_html=True,
            )
            st.caption("🔵 Blue intensity ∝ LSTM activation — brighter = more important to the LSTM")

            # LSTM attribution bar chart
            attrbs_top = attribs[:num_features]
            if attrbs_top:
                fig_lstm_attr = go.Figure(go.Bar(
                    x=[s for _, s in attrbs_top],
                    y=[t for t, _ in attrbs_top],
                    orientation="h",
                    marker_color=["rgba(59,130,246," + f"{0.3 + 0.7*s:.2f})" for _, s in attrbs_top],
                    text=[f"{s:.3f}" for _, s in attrbs_top],
                    textposition="outside",
                ))
                fig_lstm_attr.update_layout(
                    title="LSTM Token Activation (‖h_t‖ normalised)",
                    xaxis_title="Normalised L2 Norm",
                    height=max(300, 30 * len(attrbs_top) + 80),
                    margin=dict(l=20, r=60, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_lstm_attr, use_container_width=True)

            # ── Agreement analysis ──────────────────────────────────────────
            st.markdown("#### 🔍 LIME vs LSTM Agreement")
            lime_top_set = set(t for t, _ in token_weights[:5])
            lstm_top_set = set(t for t, _ in attribs[:5])
            overlap = lime_top_set & lstm_top_set
            agree_pct = len(overlap) / max(len(lime_top_set | lstm_top_set), 1) * 100

            if agree_pct >= 60:
                st.success(f"✅ **High agreement** ({agree_pct:.0f}%) — LIME and LSTM focus on similar tokens: {overlap}")
            elif agree_pct >= 30:
                st.info(f"🔄 **Partial agreement** ({agree_pct:.0f}%) — some overlap: {overlap}")
            else:
                st.warning(f"⚠️ **Low agreement** ({agree_pct:.0f}%) — LIME and LSTM highlight different tokens")

        # LIME table
        lime_df = pd.DataFrame({
            "Word":        tokens_l,
            "LIME Weight": [f"{w:+.6f}" for w in weights_l],
            "Direction":   ["✅ Supports" if w > 0 else "❌ Opposes" for w in weights_l],
            "Abs Impact":  [f"{abs(w):.4f}" for w in weights_l],
        })
        st.dataframe(lime_df, use_container_width=True, hide_index=True)

    # ── Step 5: Fusion Decision ──────────────────────────────────────────────
    if hybrid_on:
        with st.expander("**Step 5 · Fusion Decision** — α · P_finbert + (1-α) · P_lstm", expanded=show_internals):
            st.markdown('<div class="step-badge-amber">LEARNED FUSION · TRAINABLE α</div>', unsafe_allow_html=True)

            st.markdown(f"""
            The fusion layer combines both probability distributions using a **learned scalar α**:

            ```
            P_fused = α · P_finbert + (1 - α) · P_lstm
                    = {alpha_val:.4f} · P_finbert + {1 - alpha_val:.4f} · P_lstm
            ```

            α was initialised at 0.5 and trained via backpropagation on Financial PhraseBank.
            """)

            st.markdown(f'<span class="alpha-badge">α = {alpha_val:.4f} → FinBERT weight: {alpha_val:.1%} · LSTM weight: {1-alpha_val:.1%}</span>',
                        unsafe_allow_html=True)

            # Side-by-side bar chart: P_finbert vs P_lstm vs P_fused
            fig_fusion = go.Figure()
            fig_fusion.add_trace(go.Bar(
                name="P_FinBERT", x=LABEL_ORDER,
                y=[fb_probs[l] for l in LABEL_ORDER],
                marker_color=["#86efac", "#fca5a5", "#cbd5e1"],
                text=[f"{fb_probs[l]:.3f}" for l in LABEL_ORDER],
                textposition="outside",
            ))
            if lstm_probs:
                fig_fusion.add_trace(go.Bar(
                    name="P_LSTM", x=LABEL_ORDER,
                    y=[lstm_probs[l] for l in LABEL_ORDER],
                    marker_color=["#4ade80", "#f87171", "#94a3b8"],
                    text=[f"{lstm_probs[l]:.3f}" for l in LABEL_ORDER],
                    textposition="outside",
                ))
            fig_fusion.add_trace(go.Bar(
                name="P_Fused", x=LABEL_ORDER,
                y=[fused_probs[l] for l in LABEL_ORDER],
                marker_color=["#2ecc71", "#e74c3c", "#64748b"],
                text=[f"{fused_probs[l]:.3f}" for l in LABEL_ORDER],
                textposition="outside",
            ))
            fig_fusion.update_layout(
                title=f"Probability Comparison — FinBERT vs LSTM vs Fused (α={alpha_val:.3f})",
                barmode="group",
                yaxis=dict(range=[0, 1.2], tickformat=".0%"),
                height=350, margin=dict(t=50, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_fusion, use_container_width=True)

            # Fusion arithmetic table
            fusion_df = pd.DataFrame({
                "Label": LABEL_ORDER,
                f"P_FinBERT (×{alpha_val:.3f})": [f"{fb_probs[l]*alpha_val:.6f}" for l in LABEL_ORDER],
                f"P_LSTM (×{1-alpha_val:.3f})": [f"{lstm_probs[l]*(1-alpha_val):.6f}" for l in LABEL_ORDER] if lstm_probs else ["—"]*3,
                "P_Fused": [f"{fused_probs[l]:.6f}" for l in LABEL_ORDER],
                "Predicted?": ["✅" if l == pred_label else "" for l in LABEL_ORDER],
            })
            st.dataframe(fusion_df, use_container_width=True, hide_index=True)

            # Did LSTM change the outcome?
            fb_only_pred = LABEL_ORDER[int(np.argmax([fb_probs[l] for l in LABEL_ORDER]))]
            if fb_only_pred != pred_label:
                st.warning(f"⚡ **LSTM changed the prediction!** FinBERT alone → **{fb_only_pred}**, "
                           f"but the fused model → **{pred_label}**")
            else:
                st.success(f"✅ Both heads agree → **{pred_label}**. The LSTM reinforced FinBERT's decision.")

    # ── Step 6: Summary ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## ✅ Final Result")

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Prediction", f"{emoji_map[pred_label]} {pred_label}")
    rc2.metric("Confidence", f"{pred_conf:.4f}", f"{pred_conf:.1%}")
    rc3.metric("LIME tokens analysed", str(len(token_weights)))
    if hybrid_on:
        rc4.metric("Fusion α", f"{alpha_val:.4f}", f"FinBERT: {alpha_val:.0%}")
    else:
        rc4.metric("Mode", "FinBERT only")

    summary_rows = [
        ("Input", f"`{text}`"),
        ("Tokens", f"`{len(token_ids)}` (incl. [CLS], [SEP])"),
        ("Model", "FinBERT + Bi-LSTM Hybrid" if hybrid_on else "FinBERT only"),
    ]
    for l in LABEL_ORDER:
        summary_rows.append((f"P_FinBERT({l})", f"`{fb_probs[l]:.6f}` ({fb_probs[l]:.1%})"))
    if hybrid_on and lstm_probs:
        for l in LABEL_ORDER:
            summary_rows.append((f"P_LSTM({l})", f"`{lstm_probs[l]:.6f}` ({lstm_probs[l]:.1%})"))
        summary_rows.append(("α (fusion weight)", f"`{alpha_val:.4f}`"))
    for l in LABEL_ORDER:
        summary_rows.append((f"P_Fused({l})", f"`{fused_probs[l]:.6f}` ({fused_probs[l]:.1%})"))
    summary_rows.append(("**Final label**", f"**{pred_label}**"))
    summary_rows.append(("LIME top word", f"`{token_weights[0][0]}` (weight `{token_weights[0][1]:+.4f}`)"))

    md_table = "| Field | Value |\n|---|---|\n"
    for field, val in summary_rows:
        md_table += f"| {field} | {val} |\n"
    st.markdown(md_table)

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
                results = hybrid_clf.predict(lines)
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
PrismXAI · Built on <a href="https://github.com/VT69/FinSentinel">FinSentinel</a>
· ProsusAI/finbert + Bi-LSTM · LIME + LSTM Attribution XAI
</p>""", unsafe_allow_html=True)
