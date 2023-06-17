import math
from typing import Optional

import torch
from torch import Tensor, nn

from tplayground.params import ModelParams
from tplayground.layers import TransformerLayer


class GPTModel(nn.Module):
    # TODO: introduce TransformerModel base class
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        text_params = params.text_input_params
        model_dim = params.decoder_params.model_dim
        self._max_pos = text_params.max_position

        # Embeddings, token and learned positional embeddings
        self._wte = nn.Embedding(text_params.vocab_size, model_dim)
        self._wpe = nn.Embedding(self._max_pos, model_dim)

        self._emb_drop = nn.Dropout(params.embed_dropout)
        self._layers = nn.ModuleList(
            [
                TransformerLayer(params.decoder_params)
                for _ in range(params.num_layers)
            ]
        )
        self._out_norm = nn.LayerNorm(
            model_dim, eps=params.decoder_params.layer_norm_eps
        )

        # TODO: add model parallelism
        # TODO: weigth tying
        # TODO: head and head type and generation/inference for different heads
        # init all weights
        self.apply(self._init_weights)
        # GPT-2 paper initialization for residual projections in MHA
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                scale = 1 / math.sqrt(2 * params.num_layers)
                nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    @property
    def num_parameters(self) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, layer: nn.Module) -> None:
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Embedding):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(
        self, input: Tensor, targets: Optional[Tensor] = None
    ) -> Tensor:
        # Token ids and targets, should be less than max_position
        # input: [b_s, seq_len]
        if input.size()[1] > self._max_pos:
            # Should we raise exception? Cut input for now
            input = input[:, : self._max_pos]

        position_ids = torch.arange(
            0, input.size()[-1], dtype=torch.long, device=input.device
        )
        position_ids = position_ids.unsqueeze(0)

        tok_embed = self._wte(input)
        pos_embed = self._wpe(position_ids)
        x = self._emb_drop(tok_embed + pos_embed)
        # TODO: make more generic
        for layer in self._layers:
            x = layer(x)
        tr_out = self._out_norm(x)
        # Head
