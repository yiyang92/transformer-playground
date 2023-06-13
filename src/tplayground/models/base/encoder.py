import torch
from torch import nn

from tplayground.params import ModelParams
from tplayground.layers import TransformerLayer


class EncoderTransformer(nn.Module):
    def __init__(self, params: ModelParams) -> None:
        self._layers = nn.ModuleList(
            [TransformerLayer(params.encoder_params) for _ in params.num_layers]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)

        return x
