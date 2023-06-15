from enum import Enum, auto


class Activations(Enum):
    gelu = auto()
    relu = auto()
    mish = auto()


class Models(Enum):
    bert = "bert"

    def __str__(self):
        return self.value


class NormalizationMode(Enum):
    on_input = "on_input"  # Pre-normalization, GPT, LLaMA
    on_output = "on_output"  # Original variant

    def __str__(self) -> str:
        return self.value


class TransformerType(Enum):
    encoder_based = "encoder_based"
    decoder_based = "decoder_based"
    enc_dec_based = "enc_dec_based"

    def __str__(self) -> str:
        return self.value
