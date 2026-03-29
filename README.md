# 🔍 ExplainSentinel

**Token-Level Explainability for Financial NLP**  
*An NLP project built on top of [FinSentinel](https://github.com/VT69/FinSentinel)*

[![Python](https://img.shields.io/badge/Python-3.11-3b82f6?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-ProsusAI%2FFinBERT-10b981?style=flat-square)](https://huggingface.co/ProsusAI/finbert)
![Explainability](https://img.shields.io/badge/XAI-LIME%20%2B%20Attention%20Heatmaps-f59e0b?style=flat-square)
[![Dataset](https://img.shields.io/badge/Dataset-Financial%20PhraseBank-6366f1?style=flat-square)](https://huggingface.co/datasets/financial_phrasebank)

---

## 📖 Overview

FinSentinel showed that sentiment signals carry weak but statistically significant predictive value over price-based baselines. A natural follow-up question is:

> **Which words actually drive these sentiment predictions, and how much do we trust the model?**

ExplainSentinel answers this by adding a **token-level explainability layer** on top of FinBERT using **LIME** (Local Interpretable Model-agnostic Explanations) alongside internal **Attention Heatmaps**, deployed as an interactive, highly-optimized Streamlit web app.

---

## ✨ Key Capabilities

The ExplainSentinel interactive dashboard breaks down every aspect of a prediction:

1. **Step 1: NLP Tokenisation**
   Displays the exact BERT WordPiece tokens, sub-words (`##`), IDs, and internal attention masks generated from the input text.
2. **Step 2: FinBERT Forward Pass**
   Visualizes the raw classification logits extracted directly from the classification head and charts the softmax probability conversion for `Positive`, `Negative`, and `Neutral` labels.
3. **Step 3: Attention Weight Heatmaps**
   Extracts layer-12 representations to plot an interactive 2D heatmap of Self-Attention. Shows precisely which tokens the `[CLS]` classification token is paying attention to in order to make a decision.
4. **Step 4: LIME Explainability**
   Runs a localized perturbation test (~150 inferences) to build an exact word-by-word visual breakdown. Green words support the prediction, while Red words push against it.

---

## 🛠 Project Structure

```text
ExplainSentinel/
├── app.py                    # Main Streamlit Dashboard (Analyser)
├── pages/
│   └── explanation.py        # Streamlit page detailing how the tech works
├── sentiment_classifier.py   # OOM-safe FinBERT inference wrapper
├── explainer.py              # LIME perturbation & plotting logic 
├── evaluate.py               # Evaluation benchmark on Financial PhraseBank
├── requirements.txt          # Python dependencies
└── outputs/                  # Auto-generated charts & metrics (after eval)
```

---

## 🚀 Quickstart

You can run the full ExplainSentinel dashboard locally. The ProsusAI/FinBERT model weights (~440MB) will be downloaded automatically by HuggingFace on the first run.

```bash
# 1. Clone the repository
git clone https://github.com/VT69/ExplainSentinel.git
cd ExplainSentinel

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Analyzer Web App
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

ExplainSentinel acts as the **explainability extension** of FinSentinel:

| FinSentinel (Original Pipeline) | ExplainSentinel (This Project) |
| --- | --- |
| *What* sentiment does a headline carry? | *Why* did the model assign that sentiment? |
| Pipeline-level output (GMSI, signal) | Token-level model interpretability |
| End-to-End Trading Research paper scope | NLP transparency & reproducible visualization |

The FinBERT model and data pipelines implemented in FinSentinel feed directly into ExplainSentinel's architecture.

---

## 🧑‍💻 Author

**Vaibhav Tiwari**  
*B.Tech AI & ML · VIT Bhopal University*  
📧 [vaibhavtiwari159@gmail.com](mailto:vaibhavtiwari159@gmail.com)  
🔗 [linkedin.com/in/vt004](https://linkedin.com/in/vt004) · 💻 [github.com/VT69](https://github.com/VT69)
