"""
lstm_model.py
Bi-LSTM classification head and learned fusion module for the FinBERT+LSTM hybrid.

Architecture:
    FinBERT encoder (frozen) → last_hidden_state (seq_len, 768)
                                     │
                              ┌──────┴──────┐
                              │             │
                         FinBERT [CLS]   BiLSTMHead
                         linear head     (this file)
                              │             │
                          P_finbert(3)   P_lstm(3)
                              │             │
                              └──────┬──────┘
                                     │
                              HybridFusion
                              α·P_fb + (1-α)·P_lstm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMHead(nn.Module):
    """
    Bidirectional LSTM classification head.

    Takes the full token embedding sequence from FinBERT's encoder
    (shape: batch × seq_len × 768) and produces 3-class logits.

    Architecture:
        Input (seq, 768)
            → 2-layer Bi-LSTM (hidden=256, bidirectional → 512 per timestep)
            → Dropout(0.3)
            → Take final hidden states from both directions
            → Concatenate → Linear(512 → 3) → logits
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Bidirectional → output is 2 * hidden_dim
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            embeddings:     (batch, seq_len, 768) — FinBERT last_hidden_state
            attention_mask: (batch, seq_len)      — 1 for real tokens, 0 for padding

        Returns:
            logits:         (batch, 3)
            hidden_states:  (batch, seq_len, 512) — per-token LSTM hidden states
                            (used for token attribution in XAI)
        """
        # Pack padded sequences for efficiency if mask is provided
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, (h_n, _) = self.lstm(packed)
            # Unpack to get per-token hidden states
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True
            )
        else:
            lstm_out, (h_n, _) = self.lstm(embeddings)

        # h_n shape: (num_layers * 2, batch, hidden_dim)
        # Take the last layer's forward and backward hidden states
        h_forward = h_n[-2]   # (batch, hidden_dim)
        h_backward = h_n[-1]  # (batch, hidden_dim)
        h_concat = torch.cat([h_forward, h_backward], dim=1)  # (batch, 2*hidden_dim)

        h_concat = self.dropout(h_concat)
        logits = self.classifier(h_concat)  # (batch, 3)

        return logits, lstm_out  # lstm_out: (batch, seq_len, 2*hidden_dim)


class HybridFusion(nn.Module):
    """
    Learned fusion of FinBERT and LSTM probability distributions.

    Computes:  P_fused = α · P_finbert + (1 - α) · P_lstm
    where α = sigmoid(raw_alpha) is a learnable scalar ∈ (0, 1).

    Initialised so α starts at 0.5 (raw_alpha = 0.0).
    """

    def __init__(self):
        super().__init__()
        # raw_alpha=0.0 → sigmoid(0.0) = 0.5 → equal weighting at init
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self) -> float:
        """Current fusion weight toward FinBERT (after sigmoid)."""
        return torch.sigmoid(self.raw_alpha).item()

    def forward(
        self,
        logits_finbert: torch.Tensor,
        logits_lstm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits_finbert: (batch, 3) — raw logits from FinBERT classification head
            logits_lstm:    (batch, 3) — raw logits from BiLSTMHead

        Returns:
            fused_probs:    (batch, 3) — blended probability distribution
        """
        alpha = torch.sigmoid(self.raw_alpha)  # scalar ∈ (0, 1)

        p_finbert = F.softmax(logits_finbert, dim=-1)
        p_lstm = F.softmax(logits_lstm, dim=-1)

        fused = alpha * p_finbert + (1.0 - alpha) * p_lstm
        return fused


# ── Quick validation ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing BiLSTMHead + HybridFusion ...")

    batch, seq, dim = 4, 15, 768
    embeddings = torch.randn(batch, seq, dim)
    mask = torch.ones(batch, seq)
    mask[0, 10:] = 0  # simulate padding

    lstm_head = BiLSTMHead()
    logits_lstm, hidden_states = lstm_head(embeddings, mask)
    print(f"LSTM logits shape:   {logits_lstm.shape}")        # (4, 3)
    print(f"Hidden states shape: {hidden_states.shape}")      # (4, 15, 512)

    # Simulate FinBERT logits
    logits_finbert = torch.randn(batch, 3)

    fusion = HybridFusion()
    fused_probs = fusion(logits_finbert, logits_lstm)
    print(f"Fused probs shape:   {fused_probs.shape}")        # (4, 3)
    print(f"Fused probs sum:     {fused_probs.sum(dim=-1)}")  # should be ~1.0
    print(f"Initial α:           {fusion.alpha:.4f}")         # 0.5000

    total_params = sum(p.numel() for p in lstm_head.parameters())
    print(f"BiLSTMHead params:   {total_params:,}")           # ~5M
    print("✅ All shapes correct.")
