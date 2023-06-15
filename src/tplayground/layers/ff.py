import torch
from torch import nn

from tplayground.params import FFLayerParams
from tplayground.layers.activations import ACTIVATIONS_MAP


class FeedForward(nn.Module):
    def __init__(self, params: FFLayerParams, model_dim: int) -> None:
        super().__init__()
        self._inp_proj = nn.Linear(model_dim, params.hidden_size)
        self._out_proj = nn.Linear(params.hidden_size, model_dim)
        self._act = ACTIVATIONS_MAP[params.activation]
        self._dropout = nn.Dropout(params.dropout_drop_prob)

    def forward(self, x: torch.Tensor):
        x = self._inp_proj(x)
        x = self._act(x)
        x = self._out_proj(x)
        x = self._dropout(x)
        return x
