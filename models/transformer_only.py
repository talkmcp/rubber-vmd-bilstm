"""
models/transformer_only.py

Transformer-only ablation control (no BiLSTM pathway).
Operates on linearly projected input + learnable positional encoding.
Reference: Pinitjitsamut (2026), Section 2.3.2, Table 10.

Single-seed evidence: r = 0.635, StdR = 0.680 (below ideal).
Self-attention alone captures less multi-scale temporal structure
than bidirectional recurrence.
"""
import torch
import torch.nn as nn


class TransformerOnly(nn.Module):
    def __init__(
        self,
        input_dim: int = 30,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 512,
        dropout: float = 0.1,
        L: int = 30,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, L, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_projection(x) + self.pos_embedding
        z = self.transformer(z)
        return self.output_head(z[:, -1, :])
