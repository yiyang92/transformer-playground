from abc import ABC

from torch import nn, Tensor

from tplayground.params import HeadParams


class TransformerHead(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    def generate(self) -> None:
        raise NotImplementedError


class ClassificationHead(TransformerHead):
    def __init__(self, params: HeadParams, model_dim: int) -> None:
        super().__init__()
        assert params.num_classes

        self._class_out = nn.Linear(model_dim, params.num_classes)

    def generate(self) -> None:
        return


class LanguageModelHead(TransformerHead):
    def __init__(
        self, model_dim: int, vocab_size: int, bias: bool = False
    ) -> None:
        super().__init__()
        self._lm_out = nn.Linear(model_dim, vocab_size, bias=bias)

    def tile_weights(self, embed_layer: nn.Embedding) -> None:
        """Tie input and output embedding layer weights."""
        # embed.weight: [num_embed, embed_dim]
        # lm_out.weight: [out_feat, in_feat]
        assert embed_layer.weight.size() == self._lm_out.weight.size()
        self._lm_out.weight = embed_layer.weight

    def generate(self) -> None:
        return super().generate()

    def forward(self, model_out: Tensor) -> Tensor:
        return self._lm_out(model_out)
