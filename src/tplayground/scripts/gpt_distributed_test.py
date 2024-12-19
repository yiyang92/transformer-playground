import os
from logging import getLogger

import torch

torch.manual_seed(42)
import torch.distributed as dist

from tplayground.layers.attention import RingAttentionFunction
from tplayground.utils.distributed import setup_distributed_gpu

logger = getLogger(__name__)
logger.setLevel("INFO")


def _ring_allreduce_test(local_tensor, rank, world_size):
    """Perform a ring-based all-reduce operation."""
    send_tensor = local_tensor.clone()
    recv_tensor = torch.zeros_like(local_tensor)
    accumulated_result = local_tensor.clone()

    for _ in range(world_size - 1):
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1 + world_size) % world_size

        # Send and receive tensors
        send_req = dist.isend(tensor=send_tensor, dst=send_to)
        dist.recv(tensor=recv_tensor, src=recv_from)
        send_req.wait()

        # Add received tensor to the accumulated result
        accumulated_result += recv_tensor

        # Update send_tensor for the next iteration
        send_tensor = recv_tensor.clone()

    logger.info(f"Process {rank}: accumulated result {accumulated_result}")


def ring_allreduce_test():
    # Environment variables set by torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize the distributed process group
    setup_distributed_gpu(rank, world_size)

    # Each process starts with its rank as the tensor value
    local_tensor = torch.tensor([rank], dtype=torch.float32).cuda(device=rank)
    logger.info(f"Process {rank}: starting with tensor {local_tensor}")

    # Perform the ring all-reduce operation
    _ring_allreduce_test(local_tensor, rank, world_size)


def test_ring_attention_non_distributed():
    device = torch.device("cuda:0")
    d_model = 64
    seq_len = 128
    number_of_blocks = 4

    # Local tensors
    query_local = torch.randn(seq_len, d_model, requires_grad=True).to(device)
    # Local K, V tensors for each block - blocks can be distributed across different GPUs
    key_locals = [
        torch.randn(seq_len, d_model, requires_grad=True).to(device)
        for _ in range(number_of_blocks)
    ]
    value_locals = [
        torch.randn(seq_len, d_model, requires_grad=True).to(device)
        for _ in range(number_of_blocks)
    ]

    # Используем нашу функцию
    output = RingAttentionFunction.apply(query_local, key_locals, value_locals)
    loss = output.sum()
    loss.backward()

    # Print gradients
    logger.info("Gradients for query_local:", query_local.grad)
    for i, (key_grad, value_grad) in enumerate(zip(key_locals, value_locals)):
        logger.info(f"Gradients for key_local_{i}:", key_grad.grad)
        logger.info(f"Gradients for value_local_{i}:", value_grad.grad)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Exiting...")
        exit(1)

    # Test ring attention function
    test_ring_attention_non_distributed()

    # Test communication between distributed processes
    if os.environ.get("RANK") is not None:
        ring_allreduce_test()
