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
        
        # Prevent PyTorch from spawning many CPU threads on tiny Streamlit cloud nodes (causes complete freeze)
        if self.device == "cpu":
            torch.set_num_threads(1)
            
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(self.device)
        model.eval()
        
        # Massively speed up CPU inference (2x-4x) and halve RAM usage via INT8 quantization
        if self.device == "cpu":
            self.model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            self.model = model
            
        print("[FinBERT] Model loaded.")

    def predict(self, texts: list[str], batch_size: int = 8) -> list[dict]:
        """
        Predict sentiment for a list of texts, internally batching to prevent OOM/freeze.
        """
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = F.softmax(logits, dim=-1).cpu().numpy()

            for p in probs:
                pred_idx = int(p.argmax())
                results.append(
                    {
                        "label": LABEL_MAP[pred_idx],
                        "confidence": float(p[pred_idx]),
                        "probabilities": {
                            LABEL_MAP[k]: float(p[k]) for k in range(len(LABEL_MAP))
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
