from pathlib import Path
from typing import Optional
from dataclasses import dataclass, is_dataclass

from tplayground.utils.params import Params, params_decorator, camel_to_snake
from tplayground.utils.constants import TransformerType


class Registry(Params):
    """
    this class is used to register all available model params

    example of usage:
        @Registry.register
        class BertBase:
            ...

        params = Registry.get_params("bert_base")
    """

    registered_params = dict()

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
class AttentionParams(Params):
    num_heads: int
    input_dim: int
    head_dim: int

    masked: bool = False
    linear_out: bool = True

    def finalize(self):
        assert self.head_dim % self.num_heads == 0


@params_decorator
class TransformerParams(Params):
    # For encoder and decoder layers
    num_layers: int
    hidden_size: int
    attention_params: AttentionParams


@params_decorator
class ModelParams(Params):
    model_type: TransformerType

    encoder_params: Optional[TransformerParams]
    decoder_params: Optional[TransformerParams]

    def overwrite_default_attributes():
        pass

    def __post_init__(self):
        self.overwrite_default_attributes()
        self.finalize_recursive()


@Registry.register
class BERTSmall(ModelParams):
    def overwrite_default_attributes(self):
        self.model_type = TransformerType.encoder_based
