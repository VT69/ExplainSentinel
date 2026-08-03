"""
train_lstm.py
Training script for the Bi-LSTM classification head + learned fusion.

Strategy:
    1. Load Financial PhraseBank (sentences_allagree — ~2,264 samples)
    2. Run frozen FinBERT encoder ONCE → cache all embeddings as tensors
    3. Train only: BiLSTMHead weights + HybridFusion α (~5M params)
    4. Save trained weights → lstm_weights.pt

Usage:
    python train_lstm.py
    python train_lstm.py --epochs 15 --lr 5e-4 --batch_size 32
"""

import os
import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from lstm_model import BiLSTMHead, HybridFusion


# ── Constants ───────────────────────────────────────────────────────────────
MODEL_NAME = "ProsusAI/finbert"
PHRASEBANK_LABEL_MAP = {0: 2, 1: 1, 2: 0}
# PhraseBank: 0=negative, 1=neutral, 2=positive
# FinBERT:    0=positive, 1=negative, 2=neutral
# So: PB 0 (neg) → FB 1, PB 1 (neu) → FB 2, PB 2 (pos) → FB 0
PHRASEBANK_TO_FINBERT = {0: 1, 1: 2, 2: 0}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Bi-LSTM head for FinBERT hybrid")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="LSTM hidden dim")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--max_length", type=int, default=128, help="Max token length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_path", type=str, default="lstm_weights.pt",
                        help="Path to save trained weights")
    return parser.parse_args()


def load_phrasebank():
    """Load Financial PhraseBank directly from canonical URL to bypass HuggingFace dataset script errors."""
    print("[Train] Loading Financial PhraseBank (sentences_allagree) directly from source ...")
    import urllib.request
    import zipfile
    import os
    
    url = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
    zip_path = "FinancialPhraseBank-v1.0.zip"
    
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
        
    texts = []
    labels = []
    # PhraseBank to FinBERT: positive=0, negative=1, neutral=2
    label_map = {"positive": 0, "negative": 1, "neutral": 2}
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt") as f:
            for line in f.read().decode('iso-8859-1').splitlines():
                line = line.strip()
                if not line: continue
                parts = line.rsplit('@', 1)
                if len(parts) == 2:
                    texts.append(parts[0])
                    labels.append(label_map[parts[1].lower()])
                    
    print(f"[Train] Loaded {len(texts)} samples.")
    return texts, labels


def cache_finbert_embeddings(texts, labels, max_length=128, batch_size=32):
    """
    Run frozen FinBERT encoder on all texts ONCE.
    Returns cached embeddings, attention masks, FinBERT logits, and labels as tensors.
    """
    print("[Train] Loading FinBERT encoder for embedding extraction ...")
    device = "cpu"
    torch.set_num_threads(2)  # modest parallelism for embedding extraction

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load base BERT model (not the classification model) to get hidden states
    # But we also need FinBERT's classification head logits for fusion training
    from transformers import AutoModelForSequenceClassification
    model_cls = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model_cls.to(device)
    model_cls.eval()

    all_embeddings = []
    all_masks = []
    all_finbert_logits = []

    print(f"[Train] Extracting embeddings from {len(texts)} samples ...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model_cls(**inputs, output_hidden_states=True)

        # last_hidden_state from the encoder (before classification head)
        hidden_states = outputs.hidden_states[-1]  # (batch, seq, 768)
        finbert_logits = outputs.logits              # (batch, 3)

        all_embeddings.append(hidden_states.cpu())
        all_masks.append(encoding["attention_mask"].cpu())
        all_finbert_logits.append(finbert_logits.cpu())

        if (i // batch_size) % 10 == 0:
            print(f"  [{i+len(batch_texts)}/{len(texts)}]")

    embeddings_tensor = torch.cat(all_embeddings, dim=0)        # (N, seq, 768)
    masks_tensor = torch.cat(all_masks, dim=0)                  # (N, seq)
    finbert_logits_tensor = torch.cat(all_finbert_logits, dim=0) # (N, 3)
    labels_tensor = torch.tensor(labels, dtype=torch.long)      # (N,)

    del model_cls  # free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"[Train] Cached embeddings: {embeddings_tensor.shape}")
    print(f"[Train] Cached FinBERT logits: {finbert_logits_tensor.shape}")
    return embeddings_tensor, masks_tensor, finbert_logits_tensor, labels_tensor


def train(args):
    """Main training loop."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load data and cache embeddings ──────────────────────────────────────
    texts, labels = load_phrasebank()
    embeddings, masks, finbert_logits, labels_t = cache_finbert_embeddings(
        texts, labels, max_length=args.max_length
    )

    # ── Train/val split ─────────────────────────────────────────────────────
    dataset = TensorDataset(embeddings, masks, finbert_logits, labels_t)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[Train] Train: {train_size}, Val: {val_size}")

    # ── Model ───────────────────────────────────────────────────────────────
    lstm_head = BiLSTMHead(
        input_dim=768,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=3,
        dropout=args.dropout,
    )
    fusion = HybridFusion()

    total_params = sum(p.numel() for p in lstm_head.parameters())
    total_params += sum(p.numel() for p in fusion.parameters())
    print(f"[Train] Trainable parameters: {total_params:,}")

    # ── Optimiser & Loss ────────────────────────────────────────────────────
    all_params = list(lstm_head.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # ── Training loop ───────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "alpha": [],
    }
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # ── Train ───────────────────────────────────────────────────────────
        lstm_head.train()
        fusion.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for emb, msk, fb_logits, lbl in train_loader:
            optimizer.zero_grad()

            lstm_logits, _ = lstm_head(emb, msk)
            fused_probs = fusion(fb_logits, lstm_logits)

            # Cross-entropy on fused probabilities (use log for numerical stability)
            loss = F.nll_loss(torch.log(fused_probs + 1e-8), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * lbl.size(0)
            train_correct += (fused_probs.argmax(dim=1) == lbl).sum().item()
            train_total += lbl.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validate ────────────────────────────────────────────────────────
        lstm_head.eval()
        fusion.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for emb, msk, fb_logits, lbl in val_loader:
                lstm_logits, _ = lstm_head(emb, msk)
                fused_probs = fusion(fb_logits, lstm_logits)
                loss = F.nll_loss(torch.log(fused_probs + 1e-8), lbl)

                val_loss += loss.item() * lbl.size(0)
                val_correct += (fused_probs.argmax(dim=1) == lbl).sum().item()
                val_total += lbl.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        current_alpha = fusion.alpha

        scheduler.step(val_loss)

        # ── Log ─────────────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["alpha"].append(current_alpha)

        elapsed = time.time() - t_start
        print(
            f"  Epoch {epoch:2d}/{args.epochs} │ "
            f"Loss: {train_loss:.4f} / {val_loss:.4f} │ "
            f"Acc: {train_acc:.3f} / {val_acc:.3f} │ "
            f"α={current_alpha:.4f} │ "
            f"{elapsed:.0f}s"
        )

        # ── Save best ──────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "lstm_head_state_dict": lstm_head.state_dict(),
                "fusion_state_dict": fusion.state_dict(),
                "config": {
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "alpha": current_alpha,
                },
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, args.save_path)
            print(f"  ✅ Best model saved → {args.save_path}")

    total_time = time.time() - t_start
    print(f"\n" + "="*50)
    print(f"[Train] Training Complete!")
    print(f"[Train] Total Time: {total_time:.0f}s")
    print(f"[Train] Best Validation Loss: {best_val_loss:.4f}")
    print(f"[Train] Best Validation Accuracy: {max(history['val_acc']):.4f}")
    print(f"[Train] Final Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"[Train] Final Fusion α: {fusion.alpha:.4f}")
    print("="*50 + "\n")

    # ── Save training history ───────────────────────────────────────────────
    history_path = os.path.join(OUTPUT_DIR, "lstm_training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Train] History saved → {history_path}")

    # ── Plot training curves ────────────────────────────────────────────────
    plot_training_curves(history, args.epochs)

    return history


def plot_training_curves(history, epochs):
    """Save loss, accuracy, and alpha curves as PNGs."""
    epochs_range = range(1, epochs + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Loss
    axes[0].plot(epochs_range, history["train_loss"], "o-", label="Train", color="#3b82f6")
    axes[0].plot(epochs_range, history["val_loss"], "s-", label="Val", color="#ef4444")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs_range, history["train_acc"], "o-", label="Train", color="#3b82f6")
    axes[1].plot(epochs_range, history["val_acc"], "s-", label="Val", color="#ef4444")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    # Alpha
    axes[2].plot(epochs_range, history["alpha"], "D-", color="#8b5cf6", linewidth=2)
    axes[2].axhline(y=0.5, color="#94a3b8", linestyle="--", alpha=0.5, label="Init (0.5)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("α (FinBERT weight)")
    axes[2].set_title("Learned Fusion α")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)

    plt.suptitle("Bi-LSTM + FinBERT Hybrid — Training Curves", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lstm_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Train] Curves saved → {path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
