"""
evaluate.py
Evaluate FinBERT on Financial PhraseBank (sentences_allagree split).
Produces: accuracy, macro F1, per-class report, confusion matrix PNG.

Dataset: https://huggingface.co/datasets/financial_phrasebank
Split used: sentences_allagree (strongest label agreement, ~2264 samples)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sentiment_classifier import FinBERTClassifier

# Financial PhraseBank label mapping: 0=negative, 1=neutral, 2=positive
# FinBERT label mapping: Positive, Negative, Neutral
PHRASEBANK_TO_LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_phrasebank(split: str = "sentences_allagree"):
    """Load Financial PhraseBank from HuggingFace datasets."""
    print(f"[Eval] Loading Financial PhraseBank ({split}) ...")
    ds = load_dataset("takala/financial_phrasebank", split, trust_remote_code=True)
    data = ds["train"]  # only one split available
    texts = data["sentence"]
    labels = [PHRASEBANK_TO_LABEL[l] for l in data["label"]]
    print(f"[Eval] Loaded {len(texts)} samples.")
    return texts, labels


def evaluate(batch_size: int = 32):
    clf = FinBERTClassifier()
    texts, true_labels = load_phrasebank()

    # Run inference in batches
    pred_labels = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        preds = clf.predict(batch)
        pred_labels.extend([p["label"] for p in preds])
        if i % 256 == 0:
            print(f"[Eval] Processed {min(i+batch_size, len(texts))}/{len(texts)} ...")

    # Metrics
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    report = classification_report(true_labels, pred_labels, digits=4)

    print("\n" + "═" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print("═" * 50)
    print(report)

    # Save metrics to text
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Dataset : Financial PhraseBank (sentences_allagree)\n")
        f.write(f"Model   : ProsusAI/finbert\n\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Macro F1 : {macro_f1:.4f}\n\n")
        f.write(report)
    print(f"[Eval] Metrics saved → {metrics_path}")

    # Confusion matrix
    labels_order = ["Positive", "Negative", "Neutral"]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_order)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_order,
        yticklabels=labels_order,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(
        f"FinBERT on Financial PhraseBank\nAccuracy={acc:.1%}  Macro-F1={macro_f1:.3f}",
        fontsize=11,
        pad=12,
    )
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[Eval] Confusion matrix saved → {cm_path}")

    # Class distribution bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, label_list) in zip(
        axes, [("True Labels", true_labels), ("Predicted Labels", pred_labels)]
    ):
        counts = {l: label_list.count(l) for l in labels_order}
        colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
        ax.bar(counts.keys(), counts.values(), color=colors, edgecolor="white")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Count")
        for bar, val in zip(ax.patches, counts.values()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                str(val),
                ha="center",
                va="bottom",
                fontsize=10,
            )
    plt.suptitle("Label Distribution — Financial PhraseBank", fontsize=13, y=1.02)
    plt.tight_layout()
    dist_path = os.path.join(OUTPUT_DIR, "label_distribution.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Distribution chart saved → {dist_path}")

    return {"accuracy": acc, "macro_f1": macro_f1, "report": report}


if __name__ == "__main__":
    evaluate()
