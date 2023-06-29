import time
import argparse
import logging

import torch

from tplayground.params import Registry, ModelParams
from tplayground.models import BertModel

_WARMUP_STEPS = 15
_BATCH_SIZE = 16
_PROFILE_STEPS = 100
_MAX_SEQ_LEN = 512

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        help="Specify device e.g. cpu or cuda:<gpu_num>",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "-p",
        "--params_set",
        choices=Registry.get_available_params_sets(),
        type=str,
        help="parameter set",
        required=True
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Specify device e.g. cpu or cuda:<gpu_num>",
        type=int,
        default=_BATCH_SIZE,
    )
    parser.add_argument(
        "-l",
        "--log_level",
        default="INFO",
        type=str,
        choices=["INFO", "WARNING", "ERROR"],
        help="Logging level for the script (default: %(default)s)",
    )
    return parser.parse_args()


def profile(model_params: ModelParams, device: torch.device, batch_size: int) -> None:
    bert = BertModel(model_params)
    bert.to(device)
    logging.info(f"BERT number of parameters: {bert.num_parameters / 1e6}M")
    bert.eval()
    num_layers = model_params.num_layers
    # Random input
    txt_params = model_params.text_input_params
    max_seqlen = min(_MAX_SEQ_LEN, txt_params.max_position)
    txt_id_input = torch.randint(
        low=0, high=txt_params.vocab_size, size=(batch_size, max_seqlen),
        device=device, requires_grad=False)
    # [b_s, max_seqlen, model_dim]
    embed_input = bert.embed_id_input(txt_id_input)
    logging.info(f"Embedding input shape: {embed_input.size()}")
    # Warmup
    start = time.perf_counter()
    for _ in range(_WARMUP_STEPS):
        _ = bert(embed_input=embed_input)
    warmup_time = (time.perf_counter() - start) * 1000  # to ms
    logging.info(f"Warmup mean time: {warmup_time / _WARMUP_STEPS} ms")
    # Profile
    start = time.perf_counter()
    for _ in range(_PROFILE_STEPS):
        _ = bert(embed_input=embed_input)
    profile_count = (time.perf_counter() - start) / _PROFILE_STEPS * 1000
    logging.info(
        f"batch_size: {batch_size} "
        f"device: {device} seq_len: {max_seqlen} layers: {num_layers} "
        f"FT-PY-time {profile_count} ms ({_PROFILE_STEPS} iterations)"
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.getLevelName(args.log_level))
    model_params = Registry.get_params(args.params_set)
    if "cpu" in args.device:
        device = torch.device("cpu")
    elif "cuda" in args.device:
        device = torch.device(args.device)
    else:
        raise ValueError(f"Invalid device, got {args.device}")
    profile(model_params, device, args.batch_size)
