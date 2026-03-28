"""
explainer.py
Token-level explainability for FinBERT predictions.
Supports LIME (local) and SHAP (global / attention-based).
"""

import numpy as np
from typing import Optional

# ── LIME ────────────────────────────────────────────────────────────────────

def lime_explain(classifier, text: str, num_features: int = 10, num_samples: int = 300):
    """
    Explain a single prediction using LIME.

    Args:
        classifier : FinBERTClassifier instance
        text       : Input financial headline
        num_features: Number of top tokens to highlight
        num_samples : LIME perturbation samples (higher = more stable)

    Returns:
        dict with keys:
            - predicted_label  : str
            - confidence       : float
            - token_weights    : list of (token, weight) tuples
            - lime_exp         : raw LimeTextExplainer object (for HTML export)
    """
    from lime.lime_text import LimeTextExplainer

    label_names = ["Positive", "Negative", "Neutral"]
    explainer = LimeTextExplainer(class_names=label_names)

    result = classifier.predict([text])[0]
    predicted_label = result["label"]
    predicted_idx = label_names.index(predicted_label)

    exp = explainer.explain_instance(
        text,
        classifier.predict_proba_for_lime,
        num_features=num_features,
        num_samples=num_samples,
        labels=[predicted_idx],
    )

    token_weights = exp.as_list(label=predicted_idx)

    return {
        "predicted_label": predicted_label,
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "token_weights": token_weights,
        "lime_exp": exp,
        "predicted_idx": predicted_idx,
    }


def lime_html(lime_result: dict) -> str:
    """Return LIME's built-in HTML visualization."""
    return lime_result["lime_exp"].as_html()


# ── SHAP ────────────────────────────────────────────────────────────────────

def shap_explain(classifier, texts: list[str], max_evals: int = 200):
    """
    Explain predictions using SHAP's Partition explainer with transformers pipeline.

    Args:
        classifier : FinBERTClassifier instance
        texts      : List of headlines to explain
        max_evals  : SHAP evaluation budget (higher = slower but more accurate)

    Returns:
        shap.Explanation object — use shap.plots.text() or shap.summary_plot()
    """
    import shap
    from transformers import pipeline as hf_pipeline

    # Build a HuggingFace pipeline wrapping the same model
    sentiment_pipe = hf_pipeline(
        "text-classification",
        model=classifier.model,
        tokenizer=classifier.tokenizer,
        device=0 if classifier.device == "cuda" else -1,
        return_all_scores=True,
    )

    # SHAP needs a function: text -> scores
    def pipe_fn(texts):
        outputs = sentiment_pipe(list(texts))
        # outputs is list of list of {label: ..., score: ...}
        label_order = ["Positive", "Negative", "Neutral"]
        result = []
        for item in outputs:
            score_map = {d["label"]: d["score"] for d in item}
            result.append([score_map.get(l, 0.0) for l in label_order])
        return np.array(result)

    masker = shap.maskers.Text(classifier.tokenizer)
    explainer = shap.Explainer(pipe_fn, masker, output_names=["Positive", "Negative", "Neutral"])
    shap_values = explainer(texts, max_evals=max_evals, batch_size=4)

    return shap_values


# ── Token highlight helper (for Streamlit) ──────────────────────────────────

def build_highlight_html(token_weights: list, threshold: float = 0.0) -> str:
    """
    Convert LIME token weights to inline HTML with green/red highlights.

    Args:
        token_weights: list of (token, weight) from LIME
        threshold    : ignore tokens with |weight| below this

    Returns:
        HTML string safe to render with st.markdown(..., unsafe_allow_html=True)
    """
    max_w = max(abs(w) for _, w in token_weights) if token_weights else 1.0

    def token_html(token, weight):
        if abs(weight) < threshold:
            return f"<span>{token}</span>"
        intensity = int(min(abs(weight) / max_w * 200, 200))
        if weight > 0:
            color = f"rgba(46,204,113,{intensity/255:.2f})"   # green
        else:
            color = f"rgba(231,76,60,{intensity/255:.2f})"    # red
        return (
            f'<span style="background-color:{color};'
            f'border-radius:3px;padding:1px 3px;margin:1px;">'
            f"{token}</span>"
        )

    parts = [token_html(t, w) for t, w in token_weights]
    return " ".join(parts)


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sentiment_classifier import FinBERTClassifier

    clf = FinBERTClassifier()
    text = "Apple reports record quarterly earnings, crushing analyst estimates."

    print("\n── LIME Explanation ──")
    result = lime_explain(clf, text, num_features=8, num_samples=200)
    print(f"Predicted: {result['predicted_label']} ({result['confidence']:.1%})")
    print("Token weights:")
    for token, weight in result["token_weights"]:
        bar = "█" * int(abs(weight) * 30)
        sign = "+" if weight > 0 else "-"
        print(f"  {sign} {token:<20} {bar}")
