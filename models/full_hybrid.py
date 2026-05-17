"""
models/full_hybrid.py

Full BiLSTM-Transformer hybrid architecture (ablation control only).
NOT the deployed primary model — implemented to isolate the contribution
of the Transformer pathway. Multi-seed evidence shows this 84%-larger
architecture does not improve any of the three primary metrics over
BiLSTM-only (paper Section 3.3, Table 10).

Parameter count: ~1.05M
Reference: Pinitjitsamut (2026), Section 2.3.2, Figure 4.
"""
import torch
import torch.nn as nn

from .bilstm_only import TemporalAttention


class FullHybrid(nn.Module):
    """BiLSTM pathway + Transformer pathway, fusion head."""

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        d_model: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dim_ff: int = 512,
        dropout: float = 0.2,
        L: int = 30,
    ):
        super().__init__()

        # ─── Pathway 1: BiLSTM + temporal attention ───────────
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim * 2)

        # ─── Pathway 2: Transformer encoder ───────────────────
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, L, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # ─── Fusion head ───────────────────────────────────────
        fused_dim = hidden_dim * 2 + d_model         # 256 + 128 = 384
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, d)

        # Pathway 1
        h_lstm, _ = self.bilstm(x)                    # (batch, L, 256)
        c_lstm = self.attention(h_lstm)               # (batch, 256)

        # Pathway 2
        z = self.input_projection(x) + self.pos_embedding   # (batch, L, 128)
        z = self.transformer(z)                              # (batch, L, 128)
        z_last = z[:, -1, :]                                 # (batch, 128)

        # Fusion
        fused = torch.cat([c_lstm, z_last], dim=-1)          # (batch, 384)
        return self.fusion_head(fused)                       # (batch, 1)


if __name__ == "__main__":
    model = FullHybrid()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FullHybrid total trainable parameters: {n_params:,}")
    x = torch.randn(4, 30, 30)
    print(f"Output shape: {tuple(model(x).shape)}")
