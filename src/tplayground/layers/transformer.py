from torch import Tensor
from torch import nn

from tplayground.utils.constants import NormalizationMode
from tplayground.params import TransformerParams
from tplayground.layers.attention import MultiHeadSelfAttention
from tplayground.layers.ff import FeedForward


class TransformerLayer(nn.Module):
    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self._norm_mode = params.norm_mode

        self._ln_1 = nn.LayerNorm(params.model_dim, eps=params.layer_norm_eps)
        self._ln_2 = nn.LayerNorm(params.model_dim, eps=params.layer_norm_eps)

        self._attention = MultiHeadSelfAttention(params.attention_params)
        self._ff = FeedForward(
            params.ff_layer_params, model_dim=params.model_dim
        )
        # TODO: add cross attention layer for params.encoder_input == True

    def _forward_oninput(self, x: Tensor) -> Tensor:
        """With pre-layer normalization"""
        x = x + self._attention(self._ln_1(x))
        x = x + self._ff(self._ln_2(x))
        return x

    def _forward_onoutput(self, x: Tensor) -> Tensor:
        """With post layer normalization."""
        x = x + self._attention(x)
        x = self._ln_1(x)
        x = x + self._ff(x)
        return self._ln_2(x)

    def forward(self, x: Tensor) -> Tensor:
        if NormalizationMode.on_input:
            return self._forward_oninput(x)
        return self._forward_onoutput(x)
