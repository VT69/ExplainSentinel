# 🔍 PrismXAI

**Token-Level Explainability for Financial NLP**  
*An NLP project built on top of [FinSentinel](https://github.com/VT69/FinSentinel)*

[![Python](https://img.shields.io/badge/Python-3.11-3b82f6?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)](https://prismxai.streamlit.app/)
[![Model](https://img.shields.io/badge/Model-ProsusAI%2FFinBERT%20%2B%20BiLSTM-10b981?style=flat-square)](https://huggingface.co/ProsusAI/finbert)
![Explainability](https://img.shields.io/badge/XAI-LIME%20%2B%20LSTM%20Attribution-f59e0b?style=flat-square)
[![Dataset](https://img.shields.io/badge/Dataset-Financial%20PhraseBank-6366f1?style=flat-square)](https://huggingface.co/datasets/financial_phrasebank)

**🚀 Try the live application here:** [https://prismxai.streamlit.app/](https://prismxai.streamlit.app/)

---

## 📖 Overview

FinSentinel showed that sentiment signals carry weak but statistically significant predictive value over price-based baselines. A natural follow-up question is:

> **Which words actually drive these sentiment predictions, and how much do we trust the model?**

PrismXAI answers this by adding a **token-level explainability layer** to a custom **FinBERT + Bi-LSTM Hybrid Architecture**. It uses **LIME** (Local Interpretable Model-agnostic Explanations) alongside **LSTM Token Attribution** to surface the exact words driving predictions, deployed as an interactive, highly-optimized Streamlit web app.

Instead of relying solely on the single `[CLS]` token for classification, our hybrid model feeds FinBERT's raw token embeddings into a **Bidirectional LSTM**, capturing sequential context. The two signals are fused via a **learnable scalar (α)**, providing superior robustness.

---

## ✨ Key Capabilities

The PrismXAI interactive dashboard breaks down every aspect of a prediction:

1. **Step 1: NLP Tokenisation**
   Displays the exact BERT WordPiece tokens, sub-words (`##`), IDs, and internal attention masks generated from the input text.
2. **Step 2: Dual Head Forward Pass**
   Visualizes the raw logits and softmax probabilities from *both* the FinBERT `[CLS]` head and the Bi-LSTM head, evaluating the sequences simultaneously.
3. **Step 3: Attention Weight Heatmaps**
   Extracts layer-12 representations to plot an interactive 2D heatmap of Self-Attention from the FinBERT encoder.
4. **Step 4: LIME Explainability + LSTM Attribution**
   Runs a localized perturbation test to build an exact LIME word-by-word visual breakdown. Independently, calculates the L2 norm of the Bi-LSTM hidden states (`‖h_t‖`) to generate a second, complementary token-attribution signal. Includes an agreement analysis panel.
5. **Step 5: Learned Fusion Decision**
   Charts the final probability blending `P_fused = α·P_fb + (1-α)·P_lstm` where `α` is heavily optimized during training.

---

## 🛠 Project Structure

```text
PrismXAI/
├── app.py                    # Main Streamlit Dashboard (Analyser)
├── pages/
│   └── explanation.py        # Streamlit page detailing how the tech works
├── lstm_model.py             # 🧠 Bi-LSTM classification head + Fusion Module
├── train_lstm.py             # 🏋️ Training script for the hybrid system
├── sentiment_classifier.py   # Wrapper for Hybrid (FinBERT + LSTM) and pure FinBERT
├── explainer.py              # LIME perturbation & LSTM Token Attribution logic
├── evaluate.py               # Evaluation benchmark on Financial PhraseBank
└── requirements.txt          # Python dependencies
```

---

## 🚀 Quickstart

You can run the full PrismXAI dashboard locally. The ProsusAI/FinBERT model weights (~440MB) will be downloaded automatically by HuggingFace on the first run.

```bash
# 1. Clone the repository
git clone https://github.com/VT69/PrismXAI.git
cd PrismXAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional but recommended) Train the LSTM Head
# Downloads FinBERT embeddings and trains the Bi-LSTM in ~5-10 minutes on CPU.
# Generates lstm_weights.pt. If skipped, the app gracefully falls back to FinBERT-only mode.
python train_lstm.py

# 4. Launch the Analyzer Web App
streamlit run app.py
```

*Note: The backend has been explicitly hardened to prevent CPU thread starvation and Out-of-Memory (OOM) errors during heavy LIME generation steps.*

---

## 📊 Evaluation & Benchmarks

Our FinBERT implementation was benchmarked on the `sentences_allagree` split of the Financial PhraseBank dataset (~2,264 strictly annotated samples).

| Metric | Score |
| --- | --- |
| Accuracy | ~87.5% |
| Macro F1 | ~86.0% |

> Run `python evaluate.py` to reproduce the benchmarks locally. Visualizations like the confusion matrix and label distributions are automatically saved to `outputs/`.

---

## 🔗 Connection to FinSentinel

PrismXAI acts as the **explainability extension** of FinSentinel:

| FinSentinel (Original Pipeline) | PrismXAI (This Project) |
| --- | --- |
| *What* sentiment does a headline carry? | *Why* did the model assign that sentiment? |
| Pipeline-level output (GMSI, signal) | Token-level model interpretability |
| End-to-End Trading Research paper scope | NLP transparency & reproducible visualization |

The FinBERT model and data pipelines implemented in FinSentinel feed directly into PrismXAI's architecture.

---

## 🧑‍💻 Author

**Vaibhav Tiwari**  
*B.Tech AI & ML · VIT Bhopal University*  
📧 [vaibhavtiwari159@gmail.com](mailto:vaibhavtiwari159@gmail.com)  
🔗 [linkedin.com/in/vt004](https://linkedin.com/in/vt004) · 💻 [github.com/VT69](https://github.com/VT69)
