"""
models/vanilla_lstm.py

Vanilla LSTM baseline — unidirectional 2-layer LSTM with hidden=128,
receiving the 24 economic features WITHOUT the VMD modes.
This ablates the VMD-as-features contribution against a standard
deep-learning baseline (paper Table 8, Section 2.5).

The Vanilla LSTM converges to a variance-collapsed forecast across all
5 random seeds (StdR = 0.210 ± 0.007), illustrating that high DA can
be achieved without magnitude fidelity (paper Section 3.5).
"""
import torch
import torch.nn as nn


class VanillaLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 24,        # economic features only, no VMD
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,          # unidirectional
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, 24)
        h, _ = self.lstm(x)
        last = h[:, -1, :]                # (batch, 128)
        return self.output_head(last)     # (batch, 1)


if __name__ == "__main__":
    model = VanillaLSTM()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VanillaLSTM total trainable parameters: {n_params:,}")
    x = torch.randn(4, 30, 24)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(model(x).shape)}")
