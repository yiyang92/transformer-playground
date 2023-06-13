import torch
from torch import nn

from tplayground.params import TextInputParams


class Embeddings(nn.Module):
    def __init__(self, params: TextInputParams, model_dim: int) -> None:
        super().__init__()
        self._token_emb = nn.Embedding(params.vocab_size, model_dim)
        self._position_emb = nn.Embedding(params.max_position, model_dim)

        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        # Create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
