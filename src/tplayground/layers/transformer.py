import torch
from torch import nn

from tplayground.params import TransformerParams
from tplayground.layers import MultiHeadSelfAttention


class TransformerLayer(nn.Module):
    def __init__(self, params: TransformerParams) -> None:
        super.__init__()
        self._params = params

        self._layer_norms = nn.ModuleList(
            [nn.LayerNorm(params.hidden_size), nn.LayerNorm(params.hidden_size)]
        )
        self._attention = MultiHeadSelfAttention(params.attention_params)
