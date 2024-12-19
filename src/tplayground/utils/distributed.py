import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def setup_distributed_gpu(rank=None, world_size=None) -> None:
    # Should be called before any other distributed functions
    rank = int(os.environ.get("RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", world_size))

    init_method = os.environ.get("INIT_METHOD", "env://")

    assert rank is not None, "RANK environment variable must be set"
    assert world_size is not None, "WORLD_SIZE environment variable must be set"

    dist.init_process_group(
        backend="gloo|nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def ring_attention(q, k, v, rank, world_size):
    d_model = q.size(2)
    attn_output = torch.zeros_like(q)

    for _ in range(world_size):
        # Local attention
        attn_weights = F.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / (d_model**0.5), dim=-1
        )
        local_output = torch.matmul(attn_weights, v)
        attn_output += local_output

        # Rotate key and value blocks
        dist.send(k, dst=(rank + 1) % world_size)
        dist.recv(k, src=(rank - 1) % world_size)

        dist.send(v, dst=(rank + 1) % world_size)
        dist.recv(v, src=(rank - 1) % world_size)

    return attn_output
