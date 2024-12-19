import argparse
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a PyTorch distributed script using torchrun."
    )
    parser.add_argument(
        "--script",
        type=str,
        default="train.py",
        help="The name of the Python script to run (default: train.py)",
    )
    parser.add_argument(
        "--gpus", type=int, default=4, help="The number of GPUs to use (default: 4)"
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="127.0.0.1",
        help="The master node address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="12345",
        help="The master node port (default: 12345)",
    )
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    # Build the torchrun command
    torchrun_cmd = [
        "torchrun",
        f"--nproc_per_node={args.gpus}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        args.script,
    ]

    print(f"Running: {' '.join(torchrun_cmd)}")
    subprocess.run(torchrun_cmd)


if __name__ == "__main__":
    main()
