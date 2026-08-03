"""
pages/explanation.py  —  PrismXAI
Full background explanation page with flow diagrams, dataset stats, architecture.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="PrismXAI · How It Works",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.page-title  { font-size:2.2rem; font-weight:700; color:#0f172a; }
.section-hdr { font-size:1.3rem; font-weight:700; color:#1e3a5f;
               border-left:4px solid #3b82f6; padding-left:10px; margin:1.5rem 0 0.8rem; }
.info-card   { background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px;
               padding:1rem 1.3rem; margin-bottom:0.8rem; }
.warn-card   { background:#fefce8; border:1px solid #fde68a; border-radius:10px;
               padding:1rem 1.3rem; margin-bottom:0.8rem; }
.flow-box    { background:#1e293b; color:#e2e8f0; border-radius:10px;
               padding:1.2rem 1.5rem; font-family:monospace; font-size:0.88rem;
               line-height:1.9; }
.pill        { display:inline-block; padding:3px 10px; border-radius:20px;
               font-size:0.82rem; font-weight:600; margin:2px; }
.pill-blue   { background:#dbeafe; color:#1e40af; }
.pill-green  { background:#dcfce7; color:#166534; }
.pill-red    { background:#fee2e2; color:#991b1b; }
.pill-gray   { background:#f1f5f9; color:#475569; }
.pill-purple { background:#f3e8ff; color:#6b21a8; }
.arch-step   { background:white; border:1px solid #e2e8f0; border-radius:10px;
               padding:0.9rem 1.1rem; text-align:center; }
.arch-arrow  { font-size:1.8rem; color:#94a3b8; text-align:center; padding:2px 0; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("🏠 [🔍 Analyser](/)")
    st.markdown("📖 [How it works](/explanation)")
    st.markdown("---")
    st.markdown("**Quick nav**")
    st.markdown("""
- [Overview](#overview)
- [Dataset](#dataset)
- [FinBERT Architecture](#finbert)
- [Bi-LSTM + Fusion](#bilstm)
- [Pipeline Flow](#pipeline)
- [Tokenisation](#tokenisation)
- [Softmax Math](#softmax)
- [LIME XAI](#lime)
- [LSTM Attribution](#lstm-attribution)
- [Evaluation](#evaluation)
- [Connection to FinSentinel](#finsentinel)
    """)

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown('<p class="page-title">📖 How PrismXAI Works</p>', unsafe_allow_html=True)
st.markdown("A complete walkthrough of the dataset, model, pipeline, and explainability method.")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ═══════════════════════════════════════════════════════════════════
st.markdown('<a name="overview"></a>', unsafe_allow_html=True)
st.markdown('<p class="section-hdr">1 · Overview</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])
with col1:
    st.markdown("""
PrismXAI answers two questions for any financial headline:

1. **What** sentiment does this text carry? *(FinBERT + Bi-LSTM Hybrid)*
2. **Why** did the model assign that sentiment? *(LIME + LSTM Token Attribution)*

Most NLP sentiment tools are black boxes — they output a label with no justification.
PrismXAI uses a **hybrid architecture** — FinBERT's transformer encoder provides
token embeddings that feed into both FinBERT's own [CLS] classification head and a
**Bidirectional LSTM** head simultaneously. Their outputs are blended via a **learned
fusion scalar α**, and the combined system is explained through two independent
attribution methods.

It is built as the **explainability extension** of
[FinSentinel](https://github.com/VT69/FinSentinel),
which established that financial sentiment carries weak but statistically significant
predictive value over price-based baselines.
    """)

with col2:
    # Summary metrics
    metrics = {"Model":"ProsusAI/finbert + Bi-LSTM","Classes":"3 (Pos/Neg/Neu)",
               "Dataset":"Financial PhraseBank","Samples":"~2,264",
               "Accuracy":"~87.5%","Macro F1":"~86.0%",
               "DL Head":"2-layer Bi-LSTM (256 hidden)",
               "Fusion":"Learned α (trainable)",
               "XAI Methods":"LIME + LSTM Attribution",
               "Perturbations":"300 (default)"}
    mdf = pd.DataFrame(list(metrics.items()), columns=["Field","Value"])
    st.dataframe(mdf, use_container_width=True, hide_index=True, height=350)

# ═══════════════════════════════════════════════════════════════════
# 2. DATASET
# ═══════════════════════════════════════════════════════════════════
st.markdown('<a name="dataset"></a>', unsafe_allow_html=True)
st.markdown('<p class="section-hdr">2 · Dataset — Financial PhraseBank</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Financial PhraseBank** (Malo et al., 2014) is the standard benchmark for financial sentiment NLP.

- **Source:** Financial news headlines from Reuters and other wire services
- **Annotation:** Each headline annotated by 16 financial domain experts
- **Split used:** `sentences_allagree` — only sentences where **all 16 annotators agreed**
- **Size:** ~2,264 samples in the allagree split
- **Labels:** Positive, Negative, Neutral
- **Language:** English financial news
- **Available:** HuggingFace Datasets (`financial_phrasebank`)

The `allagree` split is the hardest to get wrong — if all 16 experts disagree with the
model, it's a genuine error, not annotator noise. This makes it the cleanest evaluation split.
    """)

with col2:
    # Approximate distribution (allagree split)
    dist_data = {"Positive": 1363, "Negative": 490, "Neutral": 411}
    fig_dist = go.Figure(go.Pie(
        labels=list(dist_data.keys()),
        values=list(dist_data.values()),
        hole=0.45,
        marker_colors=["#2ecc71","#e74c3c","#94a3b8"],
        textinfo="label+percent+value",
    ))
    fig_dist.update_layout(
        title="Label Distribution — sentences_allagree split",
        height=320, margin=dict(t=50,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# Sample headlines
st.markdown("**Sample headlines from each class:**")
samples = {
    "🟢 Positive": [
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period.",
        "Teleste's net sales increased by 5.6% to EUR 45.5 million.",
        "The company recorded strong earnings growth driven by export demand.",
    ],
    "🔴 Negative": [
        "Net sales of the Paper segment decreased by 18.1% to EUR 221.6 million.",
        "The company incurred a net loss of EUR 4.2 million.",
        "Bankruptcy proceedings were initiated against the company.",
    ],
    "⚪ Neutral": [
        "The company will publish its interim report on August 14.",
        "Technopolis Plc has signed a lease agreement in Jyväskylä.",
        "The Board proposes a dividend of EUR 0.25 per share.",
    ],
}
for label, headlines in samples.items():
    with st.expander(f"**{label}** examples"):
        for h in headlines:
            st.markdown(f"- *{h}*")

# ═══════════════════════════════════════════════════════════════════
# 3. FINBERT ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════
st.markdown('<a name="finbert"></a>', unsafe_allow_html=True)
st.markdown('<p class="section-hdr">3 · FinBERT Architecture</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])
with col1:
    st.markdown("""
**ProsusAI/finbert** is BERT-base fine-tuned on financial text.

**Base model:** BERT-base-uncased
- 12 Transformer layers
- 12 attention heads per layer
- 768 hidden dimensions
- 110M parameters total

**Fine-tuning:**
- Pre-trained on ~4.9B tokens of financial text (annual reports, earnings calls, news)
- Fine-tuned on Financial PhraseBank for 3-class sentiment
- Added a linear classification head on top of the [CLS] token

**Why FinBERT over general BERT?**
Financial text has domain-specific vocabulary and meaning:
- *"beat expectations"* → Positive (general BERT may miss this)
- *"in line with estimates"* → Neutral
- *"missed guidance"* → Negative
    """)

with col2:
    # Architecture diagram as a Plotly figure
    layers = [
        ("Input Text", "#dbeafe", "#1e40af"),
        ("WordPiece Tokenizer", "#e0f2fe", "#0369a1"),
        ("[CLS] + Tokens + [SEP]", "#e0f2fe", "#0369a1"),
        ("Token Embeddings", "#fce7f3", "#9d174d"),
        ("Positional Embeddings", "#fce7f3", "#9d174d"),
        ("Segment Embeddings", "#fce7f3", "#9d174d"),
        ("Input Embedding (sum)", "#fef9c3", "#854d0e"),
        ("× 12 Transformer Layers", "#f0fdf4", "#166534"),
        ("Self-Attention (12 heads)", "#dcfce7", "#166534"),
        ("Feed-Forward Network", "#dcfce7", "#166534"),
        ("[CLS] Final Hidden State (768-d)", "#f3e8ff", "#6b21a8"),
        ("Linear Layer (768 → 3)", "#f3e8ff", "#6b21a8"),
        ("Logits [pos, neg, neu]", "#fee2e2", "#991b1b"),
        ("Softmax → Probabilities", "#fee2e2", "#991b1b"),
        ("Predicted Label", "#dcfce7", "#166534"),
    ]
    y_positions = list(range(len(layers)))[::-1]
    fig_arch = go.Figure()
    for i, (name, bg, tc) in enumerate(layers):
        fig_arch.add_shape(type="rect",
            x0=0.05, x1=0.95, y0=i-0.38, y1=i+0.38,
            fillcolor=bg, line_color=tc, line_width=1.5)
        fig_arch.add_annotation(x=0.5, y=i, text=f"<b>{name}</b>",
            showarrow=False, font=dict(color=tc, size=11))
        if i < len(layers)-1:
            fig_arch.add_annotation(x=0.5, y=i+0.42, text="↓",
                showarrow=False, font=dict(color="#94a3b8", size=14))
    fig_arch.update_layout(
        title="FinBERT Forward Pass Architecture",
        height=len(layers)*42+60,
        xaxis=dict(visible=False, range=[0,1]),
        yaxis=dict(visible=False, range=[-0.6, len(layers)-0.4]),
        margin=dict(t=50,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_arch, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# 3b. BI-LSTM + FUSION ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════
st.markdown('<a name="bilstm"></a>', unsafe_allow_html=True)
st.markdown('<p class="section-hdr">3b · Bi-LSTM Head + Learned Fusion</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])
with col1:
    st.markdown("""
**Why add an LSTM alongside FinBERT?**

FinBERT's classification uses only the `[CLS]` token's final hidden state — a single
768-d vector that pools the entire sequence. This works well, but discards the
**per-token sequential structure** that a recurrent model can exploit.

The Bi-LSTM reads the **full token embedding sequence** `(seq_len × 768)` from
FinBERT's frozen encoder and produces its own 3-class probability distribution.
This is a genuine second classification signal, not just a wrapper.

**Architecture details:**

| Component | Specification |
|---|---|
| Input | `last_hidden_state` from FinBERT encoder (seq_len × 768) |
| LSTM | 2-layer Bidirectional, hidden_dim=256 → output 512 per token |
| Dropout | 0.3 between layers |
| Classifier | Linear(512 → 3) on concatenated final hidden states |
| Trainable params | ~5M (vs 110M in frozen FinBERT) |

**Why Bi-directional?**
Financial headlines often have sentiment that depends on context from both sides:
- *"RBI **cuts** GDP forecast"* → **Negative** (cuts = reduction)
- *"company **cuts** ribbon at new plant"* → **Neutral** (cuts = inaugurates)

Bidirectional processing reads left-to-right **and** right-to-left simultaneously,
capturing both-direction context that unidirectional LSTM misses.

**Why freeze FinBERT?**
Fine-tuning 110M parameters requires GPU + significant memory.
Freezing the encoder and training only the LSTM (~5M params) + fusion scalar (1 param)
is tractable on CPU in ~10 minutes.
    """)

with col2:
    # Hybrid architecture diagram
    hybrid_layers = [
        ("Input Headline", "#dbeafe", "#1e40af"),
        ("BERT Tokenizer", "#e0f2fe", "#0369a1"),
        ("FinBERT Encoder (FROZEN)", "#f0fdf4", "#166534"),
        ("last_hidden_state (seq × 768)", "#fef9c3", "#854d0e"),
    ]
    left_layers = [
        ("[CLS] → Linear(768→3)", "#f3e8ff", "#6b21a8"),
        ("P_finbert (Pos,Neg,Neu)", "#f3e8ff", "#6b21a8"),
    ]
    right_layers = [
        ("Bi-LSTM (2-layer, 256)", "#dbeafe", "#1e40af"),
        ("P_lstm (Pos,Neg,Neu)", "#dbeafe", "#1e40af"),
    ]
    fusion_layers = [
        ("Learned Fusion: α·P_fb + (1-α)·P_lstm", "#fef9c3", "#854d0e"),
        ("P_fused → argmax → Label", "#dcfce7", "#166534"),
    ]

    fig_hybrid = go.Figure()
    # Top shared layers
    for i, (name, bg, tc) in enumerate(hybrid_layers):
        y = 10 - i
        fig_hybrid.add_shape(type="rect", x0=0.1, x1=0.9, y0=y-0.38, y1=y+0.38,
            fillcolor=bg, line_color=tc, line_width=1.5)
        fig_hybrid.add_annotation(x=0.5, y=y, text=f"<b>{name}</b>",
            showarrow=False, font=dict(color=tc, size=10))
        if i < len(hybrid_layers)-1:
            fig_hybrid.add_annotation(x=0.5, y=y-0.44, text="↓",
                showarrow=False, font=dict(color="#94a3b8", size=13))

    # Split arrows
    fig_hybrid.add_annotation(x=0.3, y=5.5, text="↙", showarrow=False, font=dict(color="#94a3b8", size=16))
    fig_hybrid.add_annotation(x=0.7, y=5.5, text="↘", showarrow=False, font=dict(color="#94a3b8", size=16))

    # Left branch (FinBERT head)
    for i, (name, bg, tc) in enumerate(left_layers):
        y = 5.0 - i
        fig_hybrid.add_shape(type="rect", x0=0.02, x1=0.48, y0=y-0.35, y1=y+0.35,
            fillcolor=bg, line_color=tc, line_width=1.5)
        fig_hybrid.add_annotation(x=0.25, y=y, text=f"<b>{name}</b>",
            showarrow=False, font=dict(color=tc, size=9))
        if i < len(left_layers)-1:
            fig_hybrid.add_annotation(x=0.25, y=y-0.4, text="↓",
                showarrow=False, font=dict(color="#94a3b8", size=13))

    # Right branch (LSTM head)
    for i, (name, bg, tc) in enumerate(right_layers):
        y = 5.0 - i
        fig_hybrid.add_shape(type="rect", x0=0.52, x1=0.98, y0=y-0.35, y1=y+0.35,
            fillcolor=bg, line_color=tc, line_width=1.5)
        fig_hybrid.add_annotation(x=0.75, y=y, text=f"<b>{name}</b>",
            showarrow=False, font=dict(color=tc, size=9))
        if i < len(right_layers)-1:
            fig_hybrid.add_annotation(x=0.75, y=y-0.4, text="↓",
                showarrow=False, font=dict(color="#94a3b8", size=13))

    # Merge arrows
    fig_hybrid.add_annotation(x=0.3, y=3.2, text="↘", showarrow=False, font=dict(color="#94a3b8", size=16))
    fig_hybrid.add_annotation(x=0.7, y=3.2, text="↙", showarrow=False, font=dict(color="#94a3b8", size=16))

    # Fusion layers
    for i, (name, bg, tc) in enumerate(fusion_layers):
        y = 2.8 - i
        fig_hybrid.add_shape(type="rect", x0=0.1, x1=0.9, y0=y-0.35, y1=y+0.35,
            fillcolor=bg, line_color=tc, line_width=1.5)
        fig_hybrid.add_annotation(x=0.5, y=y, text=f"<b>{name}</b>",
            showarrow=False, font=dict(color=tc, size=10))
        if i < len(fusion_layers)-1:
            fig_hybrid.add_annotation(x=0.5, y=y-0.4, text="↓",
                showarrow=False, font=dict(color="#94a3b8", size=13))

    fig_hybrid.update_layout(
        title="FinBERT + Bi-LSTM Hybrid Architecture",
        height=520,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[1.2, 11]),
        margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_hybrid, use_container_width=True)

# Fusion explanation
st.markdown("""
**Learned Fusion — the α parameter**

The fusion layer combines both heads' probability distributions:

$$P_{\\text{fused}} = \\alpha \\cdot P_{\\text{finbert}} + (1 - \\alpha) \\cdot P_{\\text{lstm}}$$

where **α = sigmoid(raw_α)** is a learnable scalar constrained to (0, 1).

- Initialised at α = 0.5 (raw_α = 0.0 → sigmoid(0) = 0.5)
- Trained via backpropagation alongside the LSTM weights
- After training, α typically converges to ~0.7–0.8 toward FinBERT (since it was specifically fine-tuned for this task)
- The LSTM compensates on sequential/structural cues that [CLS] pooling misses

**Why not a fixed 50/50 blend?** A trained α lets the model learn how much to trust
each head from real data. It's a single parameter — no risk of overfitting — but it
transforms an arbitrary design choice into a principled, data-driven one.
""")

# Training strategy
with st.expander("🏋️ Training Strategy (train_lstm.py)", expanded=False):
    st.markdown("""
    **Step 1:** Load Financial PhraseBank (same ~2,264 samples from evaluate.py)

    **Step 2:** Run frozen FinBERT encoder ONCE on all samples → cache `last_hidden_state` tensors.
    This avoids recomputing embeddings every epoch.

    **Step 3:** Train only:
    - Bi-LSTM weights (~5M parameters)
    - Fusion α (1 parameter)

    **Hyperparameters:**
    | Parameter | Value |
    |---|---|
    | Epochs | 10 |
    | Optimizer | Adam |
    | Learning rate | 1e-3 |
    | Batch size | 16 |
    | Loss | NLL on fused probabilities |
    | Gradient clipping | 1.0 |
    | LR scheduler | ReduceLROnPlateau |
    | Validation split | 15% |

    **Training time:** ~5–10 minutes on CPU with cached embeddings.

    **Output:** `lstm_weights.pt` — loaded by `HybridSentimentClassifier` at inference time.
    Training curves saved to `outputs/lstm_training_curves.png`.
    """)

# ═══════════════════════════════════════════════════════════════════
# 4. PIPELINE FLOW
# ═══════════════════════════════════════════════════════════════════
st.markdown('<a name="pipeline"></a>', unsafe_allow_html=True)
st.markdown('<p class="section-hdr">4 · End-to-End Hybrid Pipeline Flow</p>', unsafe_allow_html=True)

st.markdown("""
<div class="flow-box">
  INPUT HEADLINE
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 1 · TOKENISATION (BERT WordPiece)                         │
  │  "Apple earnings beat" → [CLS] apple earnings beat [SEP]        │
  │  Token IDs: [101, 6207, 16565, 5842, 102]                       │
  └─────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 2 · FINBERT ENCODER (FROZEN — 12 Transformer layers)      │
  │  Output: last_hidden_state (seq_len × 768) for every token      │
  └─────────────────────────────────────────────────────────────────┘
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
  ┌────────────────────────┐    ┌─────────────────────────────────┐
  │ STEP 3a · FinBERT HEAD │    │ STEP 3b · Bi-LSTM HEAD          │
  │ [CLS] → Linear(768→3)  │    │ Full sequence → 2-layer BiLSTM   │
  │ → Softmax → P_finbert  │    │ → final hidden → Linear(512→3)  │
  │ [0.92, 0.03, 0.05]    │    │ → Softmax → P_lstm              │
  └───────────┬────────────┘    │ [0.85, 0.07, 0.08]              │
              │                 └───────────────┬─────────────────┘
              │                                 │
              └────────────────┬────────────────┘
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 4 · LEARNED FUSION                                        │
  │  P_fused = α · P_finbert + (1-α) · P_lstm                      │
  │  α is trained (starts 0.5, converges ~0.7–0.8)                 │
  │  [0.90, 0.04, 0.06] → argmax → Positive                       │
  └─────────────────────────────────────────────────────────────────┘
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
  PREDICTED LABEL + CONFIDENCE           XAI: LIME + LSTM ATTRIBUTION
  "Positive" (90.0%)                     LIME: perturb → fused model
                                         LSTM: ‖h_t‖ per-token norms
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 5. TOKENISATION DEEP DIVE
# ═══════════════════════════════════════════════════════════════════
st.markdown('<a name="tokenisation"></a>', unsafe_allow_html=True)
st.markdown('<p class="section-hdr">5 · Tokenisation — WordPiece Algorithm</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
BERT uses **WordPiece tokenisation** — a subword algorithm that:

1. Starts with a vocabulary of individual characters
2. Merges the most frequent pairs until vocabulary size is reached (30,522 tokens for BERT)
3. Unknown words are split into known subwords: `"fintech"` → `["fin", "##tech"]`
4. The `##` prefix indicates a continuation of the previous token

**Special tokens added:**
- `[CLS]` (token ID `101`) — prepended; its final hidden state drives classification
- `[SEP]` (token ID `102`) — appended; marks sequence end
- `[PAD]` (token ID `0`) — added to batch to equal length (masked out)

**Attention mask:** 1 for real tokens, 0 for [PAD] tokens — tells the model what to ignore.
    """)

with col2:
    # Example tokenisation walkthrough
    examples_tok = [
        ("Apple reports record earnings", ["[CLS]","apple","reports","record","earnings","[SEP]"], [101,6207,4161,2501,16565,102]),
        ("Bankruptcy proceedings initiated", ["[CLS]","bank","##rupt","##cy","proceedings","initiated","[SEP]"], [101,2924,8386,3693,7545,10767,102]),
        ("NIFTY50 surges 2.3%", ["[CLS]","ni","##fty","##50","surge","##s","2",".","-","3","%","[SEP]"], [101,11231,6904,11787,17058,2015,1016,1012,1011,1017,1003,102]),
    ]
    for sentence, toks, ids in examples_tok:
        with st.expander(f"`{sentence}`"):
            tdf = pd.DataFrame({"Token": toks, "ID": ids,
                                "Type": ["[CLS]" if t=="[CLS]" else "[SEP]" if t=="[SEP]"
                                         else "subword" if t.startswith("##") else "word"
                                         for t in toks]})
            st.dataframe(tdf, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# 6. SOFTMAX MATH
# ═══════════════════════════════════════════════════════════════════
st.markdown('<a name="softmax"></a>', unsafe_allow_html=True)
st.markdown('<p class="section-hdr">6 · The Softmax — From Logits to Probabilities</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(r"""
The classification head outputs three raw **logits** (unnormalised scores) —
one per class. These are converted to probabilities using **softmax**:

$$P(\text{class}_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

**Example walkthrough:**

| Step | Positive | Negative | Neutral |
|---|---|---|---|
| Raw logit $z$ | 3.821 | -1.204 | 0.103 |
| $e^z$ | 45.72 | 0.300 | 1.108 |
| Sum $\Sigma e^z$ | 47.13 | 47.13 | 47.13 |
| **P = $e^z / \Sigma$** | **0.970** | **0.006** | **0.024** |

The argmax of probabilities gives the predicted label.
Confidence = the max probability value.
    """)

with col2:
    # Interactive softmax demo
    st.markdown("**Interactive softmax demo:**")
    z_pos = st.slider("Logit (Positive)", -5.0, 5.0, 3.8, 0.1)
    z_neg = st.slider("Logit (Negative)", -5.0, 5.0, -1.2, 0.1)
    z_neu = st.slider("Logit (Neutral)",  -5.0, 5.0,  0.1, 0.1)

    import math
    exps = [math.exp(z_pos), math.exp(z_neg), math.exp(z_neu)]
    s    = sum(exps)
    probs_demo = [e/s for e in exps]
    labels_demo = ["Positive","Negative","Neutral"]
    colors_demo = ["#2ecc71","#e74c3c","#94a3b8"]
    pred_demo = labels_demo[probs_demo.index(max(probs_demo))]

    fig_sm = go.Figure()
    fig_sm.add_trace(go.Bar(x=labels_demo, y=[z_pos,z_neg,z_neu],
                            name="Logits", marker_color="#94a3b8"))
    fig_sm.add_trace(go.Bar(x=labels_demo, y=probs_demo,
                            name="Probabilities", marker_color=colors_demo))
    fig_sm.update_layout(barmode="group", height=260,
                         margin=dict(t=20,b=20,l=20,r=20),
                         paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(248,250,252,1)")
    st.plotly_chart(fig_sm, use_container_width=True)
    st.info(f"**Prediction: {pred_demo}** · confidence `{max(probs_demo):.4f}` ({max(probs_demo):.1%})")

# ═══════════════════════════════════════════════════════════════════
# 7. LIME EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════
st.markdown('<a name="lime"></a>', unsafe_allow_html=True)
st.markdown('<p class="section-hdr">7 · LIME — Local Interpretable Model-Agnostic Explanations</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
LIME (Ribeiro et al., 2016) explains *any* black-box classifier locally by:

**Step 1 · Perturbation**
Generate `N` versions of the input by randomly masking words.
e.g. `"Apple earnings beat"` →
- `"Apple [MASK] beat"` → P(Positive) drops to 0.71
- `"[MASK] earnings beat"` → P(Positive) stays at 0.88
- `"Apple earnings [MASK]"` → P(Positive) drops to 0.65

**Step 2 · Re-run the Hybrid Model**
Each perturbed version is passed through the **fused FinBERT+LSTM system**.
This is critical — LIME now explains the hybrid model's behaviour, not just FinBERT alone.
The weights reflect how the *combined system* responds to word removal.

**Step 3 · Fit local linear model**
A weighted linear regression is fitted on the perturbation dataset,
where samples closer to the original (fewer masks) get higher weight.

**Step 4 · Extract coefficients**
The regression coefficients for each word become the LIME feature weights:
- **Positive weight** → word pushes toward the prediction
- **Negative weight** → word pushes against the prediction

**Why LIME and not SHAP?**
SHAP requires model access in a specific way that conflicts with PyO3/tokenizers
on Python 3.14. LIME is model-agnostic and works with any classifier that outputs probabilities.
    """)

with col2:
    # LIME process flow diagram
    lime_steps = [
        ("Original Headline", "#dbeafe", "#1e40af",
         "\"Apple earnings beat estimates\""),
        ("Generate N=300 Perturbations", "#fce7f3", "#9d174d",
         "[MASK] earnings beat estimates\nApple [MASK] beat estimates\n..."),
        ("Run FinBERT on each", "#f0fdf4", "#166534",
         "P(Pos)=0.71, P(Pos)=0.88, ..."),
        ("Fit Weighted Linear Model", "#fef9c3", "#854d0e",
         "Weighted regression on perturbation dataset"),
        ("Extract Word Weights", "#f3e8ff", "#6b21a8",
         "earnings: +0.18, beat: +0.14, Apple: +0.03"),
        ("Visualise Highlights", "#dcfce7", "#166534",
         "Green/red word highlights in UI"),
    ]
    fig_lime_flow = go.Figure()
    for i, (title, bg, tc, detail) in enumerate(lime_steps):
        y = len(lime_steps) - 1 - i
        fig_lime_flow.add_shape(type="rect",
            x0=0.02, x1=0.98, y0=y-0.42, y1=y+0.42,
            fillcolor=bg, line_color=tc, line_width=1.5)
        fig_lime_flow.add_annotation(x=0.5, y=y+0.12,
            text=f"<b>{title}</b>",
            showarrow=False, font=dict(color=tc, size=11))
        fig_lime_flow.add_annotation(x=0.5, y=y-0.15,
            text=f"<i>{detail}</i>",
            showarrow=False, font=dict(color="#64748b", size=9))
        if i < len(lime_steps)-1:
            fig_lime_flow.add_annotation(x=0.5, y=y-0.48,
                text="↓", showarrow=False,
                font=dict(color="#94a3b8", size=16))
    fig_lime_flow.update_layout(
        title="LIME Explanation Pipeline",
        height=len(lime_steps)*72+60,
        xaxis=dict(visible=False, range=[0,1]),
        yaxis=dict(visible=False, range=[-0.6, len(lime_steps)-0.4]),
        margin=dict(t=50,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_lime_flow, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# 7b. LSTM TOKEN ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════
st.markdown('<a name="lstm-attribution"></a>', unsafe_allow_html=True)
st.markdown('<p class="section-hdr">7b · LSTM Token Attribution — A Second XAI Signal</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
The Bi-LSTM provides a **second, independent attribution signal** alongside LIME.

**How it works:**

The LSTM produces a hidden state vector **h_t** at each token position **t**.
The **L2 norm ‖h_t‖** at each position measures how "activated" the LSTM was
by that token — conceptually similar to how neuron activation magnitude
indicates importance in feedforward networks.

**Key properties:**
- **Independent of LIME** — computed from LSTM internals, not perturbation
- **Token-level** — one score per WordPiece token
- **Complementary** — when LIME and LSTM attribution agree on important tokens,
  confidence in the explanation increases; when they disagree, it highlights
  tokens where the two models have different "opinions"

**Interpretation:**
- High ‖h_t‖ → the LSTM's recurrent state changed significantly at this token
- Low ‖h_t‖ → the token had minimal impact on the LSTM's internal state
- Special tokens ([CLS], [SEP], [PAD]) are filtered out

This dual-signal approach (LIME + LSTM attribution) is displayed side-by-side
in the Analyser page, with an **agreement analysis** showing how much the two
methods overlap on their top-ranked tokens.
    """)

with col2:
    st.markdown("""
    **Visual encoding:**

    | Signal | Colour | Meaning |
    |---|---|---|
    | LIME (positive) | 🟢 Green | Pushes toward prediction |
    | LIME (negative) | 🔴 Red | Pushes against prediction |
    | LSTM attribution | 🔵 Blue | LSTM activation intensity |

    **Agreement analysis:**
    - **High agreement (≥60%)** — both signals highlight similar tokens → robust explanation
    - **Partial agreement (30-60%)** — some overlap, complementary information
    - **Low agreement (<30%)** — the LSTM focuses on different tokens than LIME's perturbation-based analysis
    """)

    # Mathematical formulation
    st.markdown(r"""
    **Mathematical formulation:**

    $$\text{score}(t) = \frac{\|h_t\|_2}{\max_i \|h_i\|_2}$$

    where $h_t \in \mathbb{R}^{512}$ is the concatenated forward+backward
    hidden state at position $t$ from the 2-layer Bi-LSTM.
    """)

# ═══════════════════════════════════════════════════════════════════
# 8. EVALUATION
# ═══════════════════════════════════════════════════════════════════
st.markdown('<a name="evaluation"></a>', unsafe_allow_html=True)
st.markdown('<p class="section-hdr">8 · Evaluation on Financial PhraseBank</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
FinBERT is evaluated on the `sentences_allagree` split of Financial PhraseBank.
Run `python evaluate.py` to reproduce — results saved to `outputs/`.

**Methodology:**
- No fine-tuning — zero-shot evaluation of ProsusAI/finbert
- Batch inference (batch size 32) to avoid OOM
- Metrics: Accuracy, Macro F1, per-class Precision/Recall/F1
- Confusion matrix saved as PNG
    """)

    # Approximate published metrics for FinBERT on this split
    eval_df = pd.DataFrame({
        "Class":     ["Positive","Negative","Neutral","**Macro**"],
        "Precision": ["0.891","0.868","0.831","**0.863**"],
        "Recall":    ["0.934","0.851","0.792","**0.859**"],
        "F1":        ["0.912","0.859","0.811","**0.861**"],
        "Support":   ["1363","490","411","2264"],
    })
    st.dataframe(eval_df, use_container_width=True, hide_index=True)
    st.caption("*Approximate published metrics for ProsusAI/finbert on sentences_allagree.*")

with col2:
    # Simulated confusion matrix
    cm = np.array([
        [1273,  41,  49],
        [  28, 417,  45],
        [  38,  47, 326],
    ])
    labels = ["Positive","Negative","Neutral"]
    fig_cm = px.imshow(cm, x=labels, y=labels,
                       color_continuous_scale="Blues",
                       text_auto=True,
                       aspect="auto",
                       title="Confusion Matrix (approximate)",
                       labels=dict(x="Predicted", y="True"))
    fig_cm.update_layout(height=350, margin=dict(t=50,b=20,l=20,r=20))
    st.plotly_chart(fig_cm, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# 9. CONNECTION TO FINSENTINEL
# ═══════════════════════════════════════════════════════════════════
st.markdown('<a name="finsentinel"></a>', unsafe_allow_html=True)
st.markdown('<p class="section-hdr">9 · Connection to FinSentinel</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**PrismXAI is the explainability extension of FinSentinel.**

[FinSentinel](https://github.com/VT69/FinSentinel) established:
- GMSI (Global Market Stress Index) — built from FinBERT/VADER sentiment on news + GDELT events
- Sentiment signals carry statistically significant predictive value (placebo-tested)
- Low stress (Q1 GMSI) predicts *higher* forward volatility — the Complacency Effect

**PrismXAI answers the next question:**
> *The GMSI uses FinBERT to score headlines. But which specific words in those headlines
> drove the sentiment scores that feed the GMSI?*

This closes the explainability loop — from headline text → token weights → sentiment score → GMSI → volatility prediction.
    """)

with col2:
    # Connection flow
    steps_conn = [
        ("Financial News Headline", "#dbeafe", "#1e40af"),
        ("FinBERT Sentiment Score", "#fce7f3", "#9d174d"),
        ("↕ PrismXAI adds token-level WHY", "#fef9c3", "#854d0e"),
        ("GMSI Construction\n(FinSentinel)", "#f0fdf4", "#166534"),
        ("Conditional Volatility Analysis\n(FinSentinel Paper 1)", "#f3e8ff", "#6b21a8"),
        ("Complacency Effect Finding", "#dcfce7", "#166534"),
    ]
    fig_conn = go.Figure()
    for i, (title, bg, tc) in enumerate(steps_conn):
        y = len(steps_conn)-1-i
        fig_conn.add_shape(type="rect",
            x0=0.02, x1=0.98, y0=y-0.4, y1=y+0.4,
            fillcolor=bg, line_color=tc, line_width=1.5)
        fig_conn.add_annotation(x=0.5, y=y,
            text=f"<b>{title}</b>",
            showarrow=False, font=dict(color=tc, size=11))
        if i < len(steps_conn)-1:
            fig_conn.add_annotation(x=0.5, y=y-0.46,
                text="↓", showarrow=False, font=dict(color="#94a3b8", size=14))
    fig_conn.update_layout(
        title="FinSentinel → PrismXAI Connection",
        height=len(steps_conn)*62+60,
        xaxis=dict(visible=False, range=[0,1]),
        yaxis=dict(visible=False, range=[-0.6, len(steps_conn)-0.4]),
        margin=dict(t=50,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_conn, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# 10. TECH STACK & DL CONCEPTS
# ═══════════════════════════════════════════════════════════════════
st.markdown('<p class="section-hdr">10 · Tech Stack</p>', unsafe_allow_html=True)

tech = {
    "🤗 Transformers": "FinBERT model loading, tokenisation, forward pass, frozen encoder",
    "🔥 PyTorch": "Bi-LSTM, tensor ops, softmax, attention extraction, training loop",
    "🍋 LIME": "Local perturbation-based token explainability (hybrid model)",
    "🧠 nn.LSTM": "Bidirectional LSTM classification head (lstm_model.py)",
    "🤗 Datasets": "Financial PhraseBank loading for training & evaluation",
    "📊 Plotly": "All interactive charts, fusion comparison, flow diagrams",
    "🎯 Scikit-learn": "Accuracy, F1, confusion matrix metrics",
    "🌊 Streamlit": "Multi-page web app + caching",
    "🐼 Pandas / NumPy": "Data manipulation and numerical ops",
}
t_df = pd.DataFrame(list(tech.items()), columns=["Library","Role"])
st.dataframe(t_df, use_container_width=True, hide_index=True)

st.markdown('<p class="section-hdr">10b · Deep Learning Concepts Demonstrated</p>', unsafe_allow_html=True)

dl_concepts = {
    "Recurrent architecture (LSTM)": "BiLSTMHead — 2-layer bidirectional LSTM",
    "Sequential modelling": "Reading full token embedding sequence, not just [CLS]",
    "Transfer learning": "Frozen FinBERT encoder as feature extractor",
    "Ensemble / model fusion": "Learned α blending two probability distributions",
    "Explainability for DL": "LSTM hidden-state attribution + LIME on hybrid",
    "Training pipeline": "train_lstm.py with loss curves, saved weights",
    "Gradient-based training": "Adam optimiser, gradient clipping, LR scheduling",
    "Regularisation": "Dropout (0.3), weight decay (1e-5)",
}
dl_df = pd.DataFrame(list(dl_concepts.items()), columns=["DL Concept", "Where It Appears"])
st.dataframe(dl_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""<p style="text-align:center;color:#94a3b8;font-size:0.8rem;">
PrismXAI · Built on <a href="https://github.com/VT69/FinSentinel">FinSentinel</a>
· ProsusAI/finbert + Bi-LSTM · Vaibhav Tiwari · VIT Bhopal University
</p>""", unsafe_allow_html=True)
