import torch
from torch import Tensor, nn

from tplayground.params import ModelParams
from tplayground.layers import TransformerLayer
from tplayground.utils.constants import TransformerType

_TYPE_VOCAB_SIZE = 2


class BertModel(nn.Module):
    
    def __init__(self, params: ModelParams) -> None:
        """Encoder transformer model - outputs representaion."""
        super().__init__()
        assert params.model_type == TransformerType.encoder_based
        text_params = params.text_input_params
        model_dim = params.encoder_params.model_dim
        self._max_pos = text_params.max_position
        # Word + positonal + token type
        self._wte = nn.Embedding(text_params.vocab_size, model_dim)
        self._wpe = nn.Embedding(self._max_pos, model_dim)
        # self._wtpe = nn.Embedding(_TYPE_VOCAB_SIZE, model_dim)
        
        self._emb_drop = nn.Dropout(params.embed_dropout)
        self._layers = nn.ModuleList(
            [
                TransformerLayer(params.encoder_params)
                for _ in range(params.num_layers)
            ]
        )
        self._out_norm = nn.LayerNorm(
            model_dim, eps=params.encoder_params.layer_norm_eps
        )
        # TODO: add heads to CLS embedding, no heads for now
        self._head = None
    
    @property
    def num_parameters(self) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def embed_id_input(self, input: Tensor) -> Tensor:
        # Input: tokens [b_s, seq_len]
        # Output: embeds [b_s, seq_len, model_dim]
        position_ids = torch.arange(
            0, input.size()[-1], dtype=torch.long, device=input.device
        )
        position_ids = position_ids.unsqueeze(0)
        tok_embed = self._wte(input)
        pos_embed = self._wpe(position_ids)
        return self._emb_drop(tok_embed + pos_embed)
    
    def forward(self, id_input: Tensor = None, embed_input = None) -> Tensor:
        """Input: tokens [b_s, seq_len]."""
        # TODO: make more optimized bert-specific additions
        if id_input is None and embed_input is None:
            raise ValueError("Invalid input.")
        
        if id_input is not None:
            input = self.embed_id_input(id_input)
        
        if embed_input is not None:
            input = embed_input
        
        assert len(input.shape) == 3
        if input.size()[1] > self._max_pos:
            # Should we raise exception? Cut input for now
            input = input[:, :self._max_pos, :]
        
        x = input
        for layer in self._layers:
            x = layer(x)
        tr_out = self._out_norm(x)
        
        if not self._head:
            return tr_out
