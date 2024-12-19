import torch
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
    def forward(ctx, query_local, key_locals, value_locals):
        # Convert lists to tuples for compatibility with autograd
        key_locals = tuple(key_locals)
        value_locals = tuple(value_locals)

        # Save tensors for backward
        ctx.save_for_backward(query_local, *key_locals, *value_locals)
        ctx.num_keys = len(key_locals)

        # Forward pass: compute attention for each block
        outputs = []
        for key_local, value_local in zip(key_locals, value_locals):
            out = nn.functional.scaled_dot_product_attention(
                query_local, key_local, value_local, attn_mask=None, dropout_p=0.0
            )
            outputs.append(out)

        # Aggregate results
        final_output = torch.stack(outputs).sum(dim=0)
        return final_output

    @staticmethod
    def backward(
        ctx, grad_output
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        # Retrieve saved tensors
        saved_tensors = ctx.saved_tensors
        query_local = saved_tensors[0]
        num_keys = ctx.num_keys
        key_locals = saved_tensors[1 : num_keys + 1]
        value_locals = saved_tensors[num_keys + 1 :]

        # Initialize gradients for all inputs
        grad_query = torch.zeros_like(query_local)
        grad_keys = [torch.zeros_like(key) for key in key_locals]
        grad_values = [torch.zeros_like(value) for value in value_locals]

        # Compute gradients for each block
        for i, (key_local, value_local) in enumerate(zip(key_locals, value_locals)):
            with torch.enable_grad():
                query_local.requires_grad_()
                key_local.requires_grad_()
                value_local.requires_grad_()

                out = nn.functional.scaled_dot_product_attention(
                    query_local, key_local, value_local, attn_mask=None, dropout_p=0.0
                )
                grad_out = torch.autograd.grad(
                    outputs=out,
                    inputs=(query_local, key_local, value_local),
                    grad_outputs=grad_output,
                    retain_graph=True,
                )
                grad_query += grad_out[0]
                grad_keys[i] += grad_out[1]
                grad_values[i] += grad_out[2]

        # Return gradients for inputs (None for non-tensor inputs)
        return grad_query, tuple(grad_keys), tuple(grad_values)
