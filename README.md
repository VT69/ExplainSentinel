# 🔍 ExplainSentinel

**Token-Level Explainability for Financial NLP**  
*An NLP project built on top of [FinSentinel](https://github.com/VT69/FinSentinel)*

[![Python](https://img.shields.io/badge/Python-3.11-3b82f6?style=flat-square&logo=python)](https://python.org)
[![Model](https://img.shields.io/badge/Model-ProsusAI%2FFinBERT-10b981?style=flat-square)](https://huggingface.co/ProsusAI/finbert)
[![Explainability](https://img.shields.io/badge/XAI-LIME%20%2B%20SHAP-f59e0b?style=flat-square)]()
[![Dataset](https://img.shields.io/badge/Dataset-Financial%20PhraseBank-6366f1?style=flat-square)](https://huggingface.co/datasets/financial_phrasebank)

---

## Overview

FinSentinel showed that sentiment signals carry weak but statistically significant predictive value over price-based baselines. A natural follow-up question is:

> **Which words actually drive these sentiment predictions, and how much do we trust the model?**

ExplainSentinel answers this by adding a **token-level explainability layer** on top of FinBERT using **LIME** (Local Interpretable Model-agnostic Explanations) and **SHAP**, deployed as an interactive Streamlit web app.

---

## Key Features

| Feature | Details |
|---|---|
| **Model** | ProsusAI/FinBERT (fine-tuned BERT on financial corpora) |
| **Task** | 3-class sentiment: Positive / Negative / Neutral |
| **Explainability** | LIME (token-level local explanations) + SHAP (global) |
| **Dataset** | Financial PhraseBank — `sentences_allagree` split (~2264 samples) |
| **App** | Streamlit — single-headline + batch mode |

---

## Results on Financial PhraseBank

| Metric | Score |
|---|---|
| Accuracy | ~0.875 |
| Macro F1 | ~0.860 |

> Run `python evaluate.py` to reproduce. Results saved to `outputs/`.

---

## Project Structure

```
ExplainSentinel/
├── sentiment_classifier.py   # FinBERT inference wrapper
├── explainer.py              # LIME + SHAP explainability
├── evaluate.py               # Evaluation on Financial PhraseBank
├── app.py                    # Streamlit web app
├── requirements.txt
└── outputs/                  # Generated charts & metrics (after eval)
    ├── confusion_matrix.png
    ├── label_distribution.png
    └── metrics.txt
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/VT69/ExplainSentinel.git
cd ExplainSentinel

# 2. Install
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

# 4. (Optional) Reproduce evaluation
python evaluate.py
```

---

## How LIME Works Here

LIME perturbs the input text by randomly masking tokens and observing how FinBERT's output probability changes. It fits a local linear model to approximate FinBERT's decision boundary around that specific input.

```
Input: "Apple reports record earnings, crushing analyst estimates."

→ FinBERT predicts: Positive (94.2%)

→ LIME token weights:
  + record        ██████████  +0.18  (strongly pushes → Positive)
  + crushing      ████████    +0.14
  + earnings      ███████     +0.12
  - analyst       ███         -0.06  (slightly opposes)
```

Green tokens **support** the predicted label. Red tokens **oppose** it. Intensity reflects weight magnitude.

---

## Connection to FinSentinel

ExplainSentinel is the **explainability extension** of FinSentinel:

| FinSentinel | ExplainSentinel |
|---|---|
| *What* sentiment does a headline carry? | *Why* did the model assign that sentiment? |
| Pipeline-level output (GMSI, signal) | Token-level interpretability |
| Research paper scope | NLP course submission + reproducible notebook |

The FinBERT model and GDELT/NewsAPI pipeline from FinSentinel feed directly into ExplainSentinel's classifier.

---

## Reproducibility

All experiments are deterministic given fixed `num_samples` in LIME. No API keys required — Financial PhraseBank loads directly from HuggingFace. FinBERT weights download automatically on first run (~440MB).

---

## Author

**Vaibhav Tiwari**  
B.Tech AI & ML · VIT Bhopal University  
📧 vaibhavtiwari159@gmail.com  
🔗 [linkedin.com/in/vt004](https://linkedin.com/in/vt004) · 💻 [github.com/VT69](https://github.com/VT69)
