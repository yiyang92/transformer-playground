import torch
from torch import nn

from tplayground.params import FFLayerParams


class FeedForward(nn.Module):
    def __init__(self, params: FFLayerParams, model_dim: int) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(model_dim, params.hidden_size)
        self.linear_2 = nn.Linear(params.hidden_size, model_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(params.dropout_drop_prob)

    def forward(self, x: torch.Tensor):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
