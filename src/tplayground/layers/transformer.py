import torch
from torch import nn

from tplayground.params import TransformerParams
from tplayground.layers.attention import MultiHeadSelfAttention
from tplayground.layers.ff import FeedForward


class TransformerLayer(nn.Module):
    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self._params = params

        self._layer_norms = (
            nn.LayerNorm(params.model_dim),
            nn.LayerNorm(params.model_dim),
        )
        self._attention = MultiHeadSelfAttention(
            params.attention_params, model_dim=params.model_dim
        )
        self._ff = FeedForward(
            params.ff_layer_params, model_dim=params.model_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sublayer_1_out = self._layer_norms[0](x)
        sublayer_1_out = x + self._attention(sublayer_1_out)
        sublayer_2_out = self._layer_norms[1](sublayer_1_out)
        sublayer_2_out = sublayer_1_out + self._ff(sublayer_2_out)
        return sublayer_2_out
