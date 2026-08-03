"""
sentiment_classifier.py
FinBERT-based financial sentiment classifier + FinBERT+LSTM hybrid.
Model: ProsusAI/finbert (Positive / Negative / Neutral)
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
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


# ═══════════════════════════════════════════════════════════════════════════
# HYBRID CLASSIFIER — FinBERT + Bi-LSTM with Learned Fusion
# ═══════════════════════════════════════════════════════════════════════════

class HybridSentimentClassifier:
    """
    Hybrid FinBERT + Bi-LSTM classifier with learned fusion.

    Architecture:
        Input text → FinBERT tokenizer → FinBERT encoder (frozen)
                                              │
                                    last_hidden_state (seq, 768)
                                         ┌────┴────┐
                                         │         │
                                    FinBERT      BiLSTM
                                  [CLS] head      head
                                         │         │
                                     P_fb(3)    P_lstm(3)
                                         └────┬────┘
                                              │
                                        HybridFusion
                                    α·P_fb + (1-α)·P_lstm
                                              │
                                        P_fused(3)

    Falls back to pure FinBERT if lstm_weights.pt is not found.
    """

    def __init__(self, weights_path: str = "lstm_weights.pt", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Hybrid] Loading on {self.device} ...")

        if self.device == "cpu":
            torch.set_num_threads(1)

        # ── Load FinBERT (frozen encoder + classification head) ─────────────
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.finbert = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            attn_implementation="eager"
        )
        self.finbert.to(self.device)
        self.finbert.eval()

        # Freeze all FinBERT parameters
        for param in self.finbert.parameters():
            param.requires_grad = False

        # ── Load Bi-LSTM head + Fusion ──────────────────────────────────────
        from lstm_model import BiLSTMHead, HybridFusion

        self.lstm_available = False
        self.alpha_value = 0.5

        if os.path.exists(weights_path):
            print(f"[Hybrid] Loading LSTM weights from {weights_path} ...")
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            config = checkpoint.get("config", {})

            self.lstm_head = BiLSTMHead(
                input_dim=768,
                hidden_dim=config.get("hidden_dim", 256),
                num_layers=config.get("num_layers", 2),
                num_classes=3,
                dropout=config.get("dropout", 0.3),
            )
            self.fusion = HybridFusion()

            self.lstm_head.load_state_dict(checkpoint["lstm_head_state_dict"])
            self.fusion.load_state_dict(checkpoint["fusion_state_dict"])

            self.lstm_head.to(self.device)
            self.fusion.to(self.device)
            self.lstm_head.eval()
            self.fusion.eval()

            self.alpha_value = self.fusion.alpha
            self.lstm_available = True
            print(f"[Hybrid] LSTM loaded. α = {self.alpha_value:.4f}")
        else:
            print(f"[Hybrid] ⚠️ {weights_path} not found — falling back to pure FinBERT.")
            self.lstm_head = None
            self.fusion = None

        print("[Hybrid] Ready.")

    def predict_detailed(self, text: str):
        """
        Full hybrid prediction with all intermediate values.

        Returns:
            dict with keys:
                - label, confidence, probabilities         (fused result)
                - finbert_logits, finbert_probs             (FinBERT head)
                - lstm_logits, lstm_probs                   (LSTM head, if available)
                - fused_probs                               (blended)
                - alpha                                     (fusion weight)
                - lstm_hidden_states                        (seq_len × 512, for XAI)
                - tokens, token_ids, attention_mask
        """
        encoding = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.finbert(**inputs, output_hidden_states=True)

        finbert_logits = outputs.logits[0].cpu()
        finbert_probs = F.softmax(finbert_logits, dim=-1).numpy()

        # Token info
        token_ids = encoding["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        attn_mask = encoding["attention_mask"][0].tolist()

        result = {
            "tokens": tokens,
            "token_ids": token_ids,
            "attention_mask": attn_mask,
            "finbert_logits": finbert_logits.tolist(),
            "finbert_probs": {LABEL_MAP[i]: float(finbert_probs[i]) for i in range(3)},
        }

        if self.lstm_available:
            # Get last hidden state for LSTM
            hidden_state = outputs.hidden_states[-1]  # (1, seq, 768)
            mask_tensor = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                lstm_logits, lstm_hidden = self.lstm_head(hidden_state, mask_tensor)
                fused_probs = self.fusion(
                    outputs.logits,  # (1, 3)
                    lstm_logits,     # (1, 3)
                )

            lstm_logits_np = lstm_logits[0].cpu()
            lstm_probs = F.softmax(lstm_logits_np, dim=-1).numpy()
            fused_probs_np = fused_probs[0].cpu().numpy()
            lstm_hidden_np = lstm_hidden[0].cpu().numpy()

            pred_idx = int(fused_probs_np.argmax())
            result.update({
                "lstm_logits": lstm_logits_np.tolist(),
                "lstm_probs": {LABEL_MAP[i]: float(lstm_probs[i]) for i in range(3)},
                "fused_probs": {LABEL_MAP[i]: float(fused_probs_np[i]) for i in range(3)},
                "lstm_hidden_states": lstm_hidden_np,  # (seq, 512) — for XAI
                "alpha": self.alpha_value,
                "label": LABEL_MAP[pred_idx],
                "confidence": float(fused_probs_np[pred_idx]),
                "probabilities": {LABEL_MAP[i]: float(fused_probs_np[i]) for i in range(3)},
                "hybrid_active": True,
            })
        else:
            pred_idx = int(finbert_probs.argmax())
            result.update({
                "lstm_logits": None,
                "lstm_probs": None,
                "fused_probs": {LABEL_MAP[i]: float(finbert_probs[i]) for i in range(3)},
                "lstm_hidden_states": None,
                "alpha": 1.0,
                "label": LABEL_MAP[pred_idx],
                "confidence": float(finbert_probs[pred_idx]),
                "probabilities": {LABEL_MAP[i]: float(finbert_probs[i]) for i in range(3)},
                "hybrid_active": False,
            })

        return result

    def predict(self, texts: list[str], batch_size: int = 8) -> list[dict]:
        """Batch prediction — same interface as FinBERTClassifier.predict()."""
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for text in texts:
            detailed = self.predict_detailed(text)
            results.append({
                "label": detailed["label"],
                "confidence": detailed["confidence"],
                "probabilities": detailed["probabilities"],
            })
        return results

    def predict_proba_for_lime(self, texts: list[str]):
        """
        Returns numpy array of shape (n_samples, 3).
        Uses fused probabilities for LIME — so LIME explains the hybrid system.
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
    print("=" * 60)
    print("Testing FinBERTClassifier (original)")
    print("=" * 60)
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

    print("\n" + "=" * 60)
    print("Testing HybridSentimentClassifier")
    print("=" * 60)
    hybrid = HybridSentimentClassifier()
    for text in samples:
        res = hybrid.predict_detailed(text)
        print(f"\nHeadline: {text}")
        print(f"  FinBERT probs : {res['finbert_probs']}")
        if res.get("hybrid_active"):
            print(f"  LSTM probs    : {res['lstm_probs']}")
            print(f"  Fused probs   : {res['fused_probs']}")
            print(f"  α             : {res['alpha']:.4f}")
        print(f"  Final label   : {res['label']} ({res['confidence']:.1%})")
