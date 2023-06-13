import torch
from torch import Tensor
from torch import nn

from tplayground.params import AttentionParams


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, params: AttentionParams) -> None:
        """Causal and bidirectional Self-Attention layer"""
        # Add masking
        super().__init__()
        # Multi-head attention tensor
        self._params = params
        self._head_dim = params.hidden_size // params.num_heads
        self._c_attn = nn.Linear(params.hidden_size, params.hidden_size * 3)
        # Dropout in attention and output
        self._attn_dropout = nn.Dropout(params.attention_drop_prob)
        self._resid_dropout = nn.Dropout(params.residual_drop_prob)
        # Flash attention is only supported in torch >= 2.0
        if params.use_flash and not hasattr(
            nn.functional, "scaled_dot_product_attention"
        ):
            # We left use_flash as a parameter for debugging/profiling
            # Strongly recommend to use torch >= 2.0 for optimized attention
            raise ValueError(
                "Fused attention is not supported by the "
                "current pytoch version."
            )

        if params.linear_out:
            self._c_proj = nn.Linear(params.hidden_size, params.hidden_size)

    def _qkv_heads(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # [b_s, seq_len, model_dim] -> [b_s, seq_len, head_dim * #heads * 3]
        proj = self._c_attn(input)
        query, key, value = proj.split(proj.size()[-1] // 3, dim=2)
        # Split into heads
        new_shape = query.size()[:-1] + (
            self._params.num_heads,
            self._head_dim,
        )
        # [b_s, seq_len, head_dim * #heads] -> [b_s, #heads, seq_len, head_dim]
        query = query.view(new_shape).permute(0, 2, 1, 3)
        key = key.view(new_shape).permute(0, 2, 1, 3)
        value = value.view(new_shape).permute(0, 2, 1, 3)
        return query, key, value

    def _attn(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # [b_s, #heads, seq_len, hidden]
        if self._params.use_flash:
            dropout_drop_prob = 0.0
            if self.training:
                dropout_drop_prob = self._params.attention_drop_prob

            return nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout_drop_prob,
                is_causal=self._params.causal,
            )
        # Manual implementation
        scale = 1.0
        if self._params.scale:
            scale = 1 / key.size(-1) ** 0.5

        attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale
        # TODO: add causality and attention mask
        attn_weight = nn.functional.softmax(attn_weight, dim=-1)
        if self._params.causal:
            # NOTE: simple not efficient implementation for now
            seq_len = query.size()[-2]
            causal_mask = torch.tril(torch.ones((seq_len, seq_len)))
            causal_mask = causal_mask.view(1, 1, seq_len, seq_len) == 0
            attn_weight = attn_weight.masked_fill(causal_mask, float("-inf"))
        attn_weight = self._attn_dropout(attn_weight)
        out = torch.matmul(attn_weight, value)
        return out

    def _out_proj(self, attn_out: Tensor) -> Tensor:
        # Merge heads dimension into hidden
        # [b_s, #heads, seq_len, head_dim] -> [b_s, seq_len, hidden_size]
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        new_shape = attn_out.size()[:-2] + (self._params.hidden_size,)
        attn_out = attn_out.view(new_shape)

        # Apply linear projection
        if self._params.linear_out:
            return self._c_proj(attn_out)
        return attn_out

    def forward(self, input: Tensor) -> Tensor:
        # TODO: how to use q, k from encoder
        query, key, value = self._qkv_heads(input)
        out = self._attn(query, key, value)
        out = self._resid_dropout(self._out_proj(out))
        return out
