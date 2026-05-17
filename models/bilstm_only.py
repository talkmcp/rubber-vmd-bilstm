"""
models/bilstm_only.py

VMD-Augmented BiLSTM (proposed deployed model).
Architecture: 2-layer Bidirectional LSTM with temporal attention.
Input:  (batch, L=30, d=30)  — 24 economic features + 6 VMD modes
Output: (batch, 1)            — Δp̂_{τ+1}

Parameter count: ~573K
Reference: Pinitjitsamut (2026), Section 2.3, Table 5.

NOTE TO USERS: This is a structural skeleton showing the architecture.
The production training entry point is `pipeline/train.py`. The skeleton
below is sufficient to instantiate the model from your own training loop;
to use the provided training loop, see `docs/REPRODUCING_RESULTS.md`.
"""
import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Learnable scalar weights over L look-back steps."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (batch, L, 2H)
        scores = torch.tanh(self.W_a(h))                  # (batch, L, 1)
        alpha = torch.softmax(scores, dim=1)              # (batch, L, 1)
        context = (alpha * h).sum(dim=1)                  # (batch, 2H)
        return context


class VMDBiLSTM(nn.Module):
    """
    VMD-Augmented BiLSTM (proposed model).

    Args:
        input_dim:   d = 30 (24 economic + 6 VMD modes)
        hidden_dim:  H = 128 per direction
        num_layers:  2
        dropout:     0.2
    """
    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim * 2)

        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),                 # 256
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, L, d)  e.g. (batch, 30, 30)
        Returns: (batch, 1) — predicted normalised Δp̂_{τ+1}
        """
        h, _ = self.bilstm(x)                              # (batch, L, 2H=256)
        context = self.attention(h)                        # (batch, 256)
        out = self.output_head(context)                    # (batch, 1)
        return out


if __name__ == "__main__":
    model = VMDBiLSTM()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VMDBiLSTM total trainable parameters: {n_params:,}")
    # Smoke test
    x = torch.randn(4, 30, 30)
    y = model(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
