import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from tplayground.params import AttentionParams


def dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    scale = math.sqrt(key.size(-1))
    alpha = query @ key.transpose(1, 2) / scale
    alpha = F.softmax(alpha, dim=-1)
    return torch.bmm(alpha, value)


class AttentionHead(nn.Module):
    def __init__(self, input_dim: int, head_dim: int) -> None:
        super().__init__()
        self._q_layer = nn.Linear(input_dim, head_dim)
        self._k_layer = nn.Linear(input_dim, head_dim)
        self._v_layer = nn.Linear(input_dim, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q is None:
            q = self._q_layer(x)

        if k is None:
            k = self._k_layer(x)

        v = self._v_layer(x)
        out = dot_product_attention(q, k, v)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, params: AttentionParams) -> None:
        super().__init__()
        # TODO: use clever reshape, not just head concat
        self._heads = nn.ModuleList(
            [
                AttentionHead(
                    params.input_dim, params.head_dim // params.num_heads
                )
                for _ in range(params.num_heads)
            ]
        )
        self._out_linear = None
        if params.linear_out:
            self._out_linear = nn.Linear(params.head_dim, params.head_dim)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        outs = []
        for head in self._heads:
            outs.append(head(hidden_state))

        out = torch.concat(outs, dim=-1)
        return out
