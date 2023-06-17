from pathlib import Path
from typing import Optional
from dataclasses import dataclass, is_dataclass

from tplayground.utils.params import Params, params_decorator, camel_to_snake
from tplayground.utils.constants import (
    TransformerType,
    NormalizationMode,
    Activations,
    TransformerHead,
)


class Registry(Params):
    """
    this class is used to register all available model params

    example of usage:
        @Registry.register
        class BertBase:
            ...

        params = Registry.get_params("bert_base")
    """

    registered_params: dict[str, type["Params"]] = dict()

    @classmethod
    def register(cls, params: type["Params"]):
        assert issubclass(params, Params)
        # without dataclass decorator class attributes will be unavailable
        if not is_dataclass(params):
            params = dataclass(params)
        cls.registered_params[camel_to_snake(params.__name__)] = params
        return params

    @classmethod
    def get_available_params_sets(cls) -> list[str]:
        return list(cls.registered_params.keys())

    @classmethod
    def get_params(cls, params_name: str = "") -> Params:
        return cls.registered_params[params_name]()


@params_decorator
class OptimizerParams:
    beta_one: float = 0.9
    beta_two: float = 0.98


@params_decorator
class TrainingParams:
    optimizer_params: OptimizerParams


@params_decorator
class TextInputParams(Params):
    vocab_size: int
    max_position: int

    # https://paperswithcode.com/method/weight-tying
    # Might generate some torch compile warinings
    out_in_emb_tying: bool = True


@params_decorator
class AttentionParams(Params):
    num_heads: int
    hidden_size: int

    attention_drop_prob: float = 0.0
    residual_drop_prob: float = 0.0

    causal: bool = False
    use_flash: bool = True  # If use torch 2.0 Flash Attention kernel
    linear_out: bool = True
    scale: bool = True
    value_proj: bool = True

    def finalize(self):
        assert self.hidden_size % self.num_heads == 0


@params_decorator
class FFLayerParams(Params):
    # Position-wise FF layer
    dropout_drop_prob: float = 0.0
    activation: Activations = Activations.gelu
    hidden_size: int


@params_decorator
class HeadParams(Params):
    # General paramters collection for different heads
    num_classes: Optional[int]


@params_decorator
class TransformerParams(Params):
    # For encoder and decoder layers
    encoder_input: bool = False
    model_dim: int  # Model input-output dimension
    norm_mode: NormalizationMode = NormalizationMode.on_input
    layer_norm_eps: float = 1e-5

    attention_params: AttentionParams
    ff_layer_params: FFLayerParams

    def finalize(self):
        # Usually like this, may experiment
        assert self.attention_params.hidden_size == self.model_dim
        # According to transformer paper
        if not self.ff_layer_params.hidden_size:
            self.ff_layer_params.hidden_size = 4 * self.model_dim


@params_decorator
class ModelParams(Params):
    model_type: TransformerType
    model_head: TransformerHead = TransformerHead.lm_model
    # Possibly may support ViTs in future
    text_input_params: TextInputParams
    num_layers: int
    embed_dropout: float = 0.0

    encoder_params: Optional[TransformerParams]
    decoder_params: Optional[TransformerParams]

    def finalize(self):
        if self.model_type == TransformerType.decoder_based:
            assert self.decoder_params
            self.decoder_params.causal = True

        if self.model_type == TransformerType.enc_dec_based:
            assert self.decoder_params
            assert self.encoder_params
            self.decoder_params.encoder_input = True

        if self.model_type == TransformerType.encoder_based:
            assert self.encoder_params

    def overwrite_default_attributes(self):
        pass

    def __post_init__(self):
        self.overwrite_default_attributes()
        self.finalize_recursive()


@Registry.register
class BERTSmall(ModelParams):
    def overwrite_default_attributes(self):
        self.model_type = TransformerType.encoder_based


@Registry.register
class Gpt2(ModelParams):
    # 124M parameters
    def overwrite_default_attributes(self):
        self.model_type = TransformerType.decoder_based
        # GPT-2 vocab_size of 50257,
        # padded up to nearest multiple of 64 for efficiency
        self.text_input_params.vocab_size = 50304
        self.text_input_params.max_position = 1024
        self.num_layers = 12
        self.decoder_params.model_dim = 768
        self.decoder_params.attention_params.hidden_size = 768
        self.decoder_params.attention_params.num_heads = 12
        self.decoder_params.causal = True
