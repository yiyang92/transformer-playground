from typing import Any

import torch

torch.backends.cuda.enable_flash_sdp(True)
from torch import Tensor, nn

from tplayground.params import AttentionParams
from tplayground.utils.distributed import ring_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, params: AttentionParams) -> None:
        """Causal and bidirectional Self-Attention layer"""
        # Add masking
        super().__init__()
        # Multi-head attention tensor
        self._params = params
        self._head_dim = params.hidden_size // params.num_heads
        self._query_proj = nn.Linear(params.hidden_size, params.hidden_size)
        self._key_proj = nn.Linear(params.hidden_size, params.hidden_size)
        if params.value_proj:
            self._value_proj = nn.Linear(params.hidden_size, params.hidden_size)
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
                "Fused attention is not supported by the " "current pytoch version."
            )

        if params.linear_out:
            self._c_proj = nn.Linear(params.hidden_size, params.hidden_size)

    def _qkv_heads(
        self, input: Tensor, encoder_out: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # [b_s, seq_len, model_dim] -> [b_s, seq_len, head_dim * #heads * 3]
        value = input
        if self._params.value_proj:
            value = self._value_proj(input)

        if encoder_out is not None:
            input = encoder_out

        query = self._query_proj(input)
        key = self._value_proj(input)
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
        if self._params.causal:
            # NOTE: simple not efficient implementation for now
            seq_len = query.size()[-2]
            causal_mask = torch.tril(torch.ones((seq_len, seq_len)))
            causal_mask = causal_mask.view(1, 1, seq_len, seq_len) == 0
            attn_weight = attn_weight.masked_fill(causal_mask, float("-inf"))
        attn_weight = nn.functional.softmax(attn_weight, dim=-1)
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

    def forward(self, input: Tensor, encoder_out: Tensor = None) -> Tensor:
        query, key, value = self._qkv_heads(input, encoder_out)
        out = self._attn(query, key, value)
        out = self._resid_dropout(self._out_proj(out))
        return out


class RingAttentionTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, x, rank, world_size):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(q.size(0), q.size(1), self.n_heads, -1).permute(0, 2, 1, 3)
        k = k.view(k.size(0), k.size(1), self.n_heads, -1).permute(0, 2, 1, 3)
        v = v.view(v.size(0), v.size(1), self.n_heads, -1).permute(0, 2, 1, 3)

        attn_output = ring_attention(q, k, v, rank, world_size)

        # Reshape back and project
        attn_output = (
            attn_output.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(1), -1)
        )
        return self.out_proj(attn_output)


class RingAttentionFunction(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs: tuple[Tensor, Tensor, Tensor], output: Any) -> None:
        """Save tensors for backward pass."""
        query_local, keys, values = inputs
        ctx.save_for_backward(query_local, *keys, *values)
        ctx.num_blocks = len(keys)

    @staticmethod
    def forward(query_local, keys, values):
        """Forward pass for the ring attention function, coomputes attention for different blocks and aggregate."""
        outputs = []
        for i in range(len(keys)):
            out = nn.functional.scaled_dot_product_attention(
                query_local, keys[i], values[i], attn_mask=None, dropout_p=0.0
            )
            outputs.append(out)

        # Aggregate results
        return torch.stack(outputs).sum(dim=0)

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Retrieve saved tensors
        saved_tensors = ctx.saved_tensors
        query_local = saved_tensors[0]
        keys = saved_tensors[1 : 1 + ctx.num_blocks]
        values = saved_tensors[1 + ctx.num_blocks :]

        # Initialize gradients
        grad_query = torch.zeros_like(query_local)
        grad_keys = [torch.zeros_like(k) for k in keys]
        grad_values = [torch.zeros_like(v) for v in values]

        # Compute gradients for each block
        for i in range(ctx.num_blocks):
            out = nn.functional.scaled_dot_product_attention(
                query_local, keys[i], values[i], attn_mask=None, dropout_p=0.0
            )

            grads = torch.autograd.grad(
                outputs=out,
                inputs=(query_local, keys[i], values[i]),
                grad_outputs=grad_output,
                retain_graph=True,
                allow_unused=True,
            )
            # Accumulate gradients
            grad_query += grads[0] if grads[0] is not None else 0
            grad_keys[i] += grads[1] if grads[1] is not None else 0
            grad_values[i] += grads[2] if grads[2] is not None else 0

        return grad_query, grad_keys, grad_values
