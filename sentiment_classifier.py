"""
sentiment_classifier.py
FinBERT-based financial sentiment classifier.
Model: ProsusAI/finbert (Positive / Negative / Neutral)
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = "ProsusAI/finbert"

# Label mapping from FinBERT output indices
LABEL_MAP = {0: "Positive", 1: "Negative", 2: "Neutral"}
LABEL_COLORS = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}


class FinBERTClassifier:
    """
    Wraps ProsusAI/finbert for inference.
    Supports single headlines and batches.
    """

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FinBERT] Loading model on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        print("[FinBERT] Model loaded.")

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Predict sentiment for a list of texts.

        Returns:
            List of dicts with keys: label, confidence, probabilities
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = F.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for p in probs:
            pred_idx = int(p.argmax())
            results.append(
                {
                    "label": LABEL_MAP[pred_idx],
                    "confidence": float(p[pred_idx]),
                    "probabilities": {
                        LABEL_MAP[i]: float(p[i]) for i in range(len(LABEL_MAP))
                    },
                }
            )
        return results

    def predict_proba_for_lime(self, texts: list[str]):
        """
        Returns numpy array of shape (n_samples, n_classes).
        Required by LIME's TextExplainer.
        Order: [Positive, Negative, Neutral]
        """
        import numpy as np

        results = self.predict(texts)
        return np.array(
            [
                [
                    r["probabilities"]["Positive"],
                    r["probabilities"]["Negative"],
                    r["probabilities"]["Neutral"],
                ]
                for r in results
            ]
        )


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    clf = FinBERTClassifier()
    samples = [
        "Apple reports record quarterly earnings, beating all analyst estimates.",
        "The bank faces a major liquidity crisis after bond market collapse.",
        "The Fed held interest rates steady at its March meeting.",
    ]
    for text, res in zip(samples, clf.predict(samples)):
        print(f"\nHeadline : {text}")
        print(f"Sentiment: {res['label']} ({res['confidence']:.1%})")
        print(f"Probs    : {res['probabilities']}")
