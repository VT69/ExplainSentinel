"""
explainer.py
Token-level explainability for FinBERT and Hybrid predictions.
Supports:
    - LIME (local perturbation-based)
    - LSTM Token Attribution (L2 norm of Bi-LSTM hidden states)
    - SHAP (global / attention-based — optional)
"""

import numpy as np
from typing import Optional

# ── LIME ────────────────────────────────────────────────────────────────────

def lime_explain(classifier, text: str, num_features: int = 10, num_samples: int = 300):
    """
    Explain a single prediction using LIME.

    Works with both FinBERTClassifier and HybridSentimentClassifier —
    both expose predict_proba_for_lime(). When using the hybrid classifier,
    LIME perturbations run against the fused model, so the explanation
    reflects the hybrid system's behaviour.

    Args:
        classifier : FinBERTClassifier or HybridSentimentClassifier instance
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


# ── LSTM Token Attribution ──────────────────────────────────────────────────

def lstm_token_attribution(lstm_hidden_states: np.ndarray, tokens: list[str],
                           attention_mask: list[int] = None) -> list[tuple[str, float]]:
    """
    Compute per-token attribution scores from Bi-LSTM hidden states.

    The LSTM produces a hidden state h_t at each token position t.
    The L2 norm ||h_t|| measures how much the LSTM was "activated"
    by that token — higher norm = more important to the LSTM.

    This gives a second, independent attribution signal alongside LIME.
    When displayed side-by-side, you can see whether LIME and the LSTM
    agree or disagree on which tokens matter.

    Args:
        lstm_hidden_states: (seq_len, hidden_dim*2) — from BiLSTMHead's output
        tokens:             list of WordPiece tokens (same length as seq_len)
        attention_mask:     optional mask to exclude padding tokens

    Returns:
        list of (token, normalised_score) tuples, sorted by score descending.
        Scores are normalised to [0, 1] range.
    """
    if lstm_hidden_states is None:
        return []

    # Compute L2 norm at each position: ||h_t||
    norms = np.linalg.norm(lstm_hidden_states, axis=1)  # (seq_len,)

    # Mask out padding and special tokens
    if attention_mask is not None:
        mask = np.array(attention_mask[:len(norms)], dtype=bool)
        norms = norms * mask

    # Skip [CLS] and [SEP] tokens for cleaner attribution
    for i, tok in enumerate(tokens[:len(norms)]):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            norms[i] = 0.0

    # Normalise to [0, 1]
    max_norm = norms.max()
    if max_norm > 0:
        normalised = norms / max_norm
    else:
        normalised = norms

    # Build (token, score) pairs, filter out zero-score tokens
    attributions = []
    for i, (tok, score) in enumerate(zip(tokens[:len(normalised)], normalised)):
        if score > 0.0 and tok not in ("[CLS]", "[SEP]", "[PAD]"):
            attributions.append((tok, float(score)))

    # Sort by score descending
    attributions.sort(key=lambda x: x[1], reverse=True)
    return attributions


def build_lstm_attribution_html(attributions: list[tuple[str, float]]) -> str:
    """
    Convert LSTM token attributions to inline HTML with blue-intensity highlights.
    Uses a blue colour scale (distinct from LIME's green/red) to visually
    separate the two attribution signals.

    Args:
        attributions: list of (token, normalised_score) from lstm_token_attribution()

    Returns:
        HTML string safe for st.markdown(..., unsafe_allow_html=True)
    """
    if not attributions:
        return "<span style='color:#94a3b8;'>No LSTM attributions available.</span>"

    max_score = max(s for _, s in attributions) if attributions else 1.0

    def token_html(token, score):
        intensity = int(min(score / max_score * 220, 220))
        color = f"rgba(59,130,246,{intensity/255:.2f})"  # blue
        return (
            f'<span style="background-color:{color};'
            f'border-radius:3px;padding:1px 4px;margin:1px;'
            f'font-size:1rem;">'
            f'{token}</span>'
        )

    parts = [token_html(t, s) for t, s in attributions]
    return " ".join(parts)


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

    # Test LSTM attribution (requires trained hybrid model)
    try:
        from sentiment_classifier import HybridSentimentClassifier
        hybrid = HybridSentimentClassifier()
        if hybrid.lstm_available:
            detailed = hybrid.predict_detailed(text)
            attribs = lstm_token_attribution(
                detailed["lstm_hidden_states"],
                detailed["tokens"],
                detailed["attention_mask"],
            )
            print("\n── LSTM Token Attribution ──")
            for tok, score in attribs[:10]:
                bar = "█" * int(score * 30)
                print(f"  {tok:<20} {bar} ({score:.3f})")
    except Exception as e:
        print(f"\n[LSTM test skipped: {e}]")

