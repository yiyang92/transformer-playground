# Utilities for working with NTS (NIO Training System) cluster.
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T")
R = TypeVar("R")


class classproperty(Generic[T, R]):  # noqa: N801
    def __init__(self, func: Callable[[Type[T]], R]) -> None:
        self.func = func

    def __get__(self, obj: Any, cls: Type[T]) -> R:
        """Get class property."""
        return self.func(cls)


class NtsInstanceType(Enum):
    JOB = "job"
    WORKSHOP = "workshop"

    def __str__(self) -> str:
        """Convert NTSInstanceType to string."""
        return self.value


class GPUType(Enum):
    A100_15C = "a100-15c"

    def __str__(self) -> str:
        """Convert GPUType to string."""
        return self.value


class NtsJobPriorities(Enum):
    NORMAL = "normal"
    LOW = "low"

    def __str__(self) -> str:
        """Convert NTSJobPriorities to string."""
        return self.value


def _get_torchpilot_output_dir() -> Path:
    """Get torchpilot output directory."""
    return Path(
        os.environ.get(
            "TORCHPILOT_output_dir", Path("~/torchpilot_outputs").expanduser()
        )
    )


class Images(Enum):
    IMAGE_TORCHPILOT_BASE = "adas-img.nioint.com/nio-pilot/torchpilot:cu114-v1.0.0"
    IMAGE_FME_UTILS_BASE = (
        "adas-img.nioint.com/aa-pnc/fme_develop:cuda121_debian12_x86_64"
    )


@dataclass
class NtsCtlConstants:
    IMAGE_NAME: str
    GPU_TYPE: GPUType = GPUType.A100_15C

    TORCHPILOT_OUTPUT_DIR: Path = _get_torchpilot_output_dir()
    CKPT_DIR_NAME: str = "ckpt"
    CKPT_SUFFIX: str = ".tar"

    MAX_GPUS_PER_NODE: int = 8
    MAX_CPUS_PER_GPU: int = 15
    MAX_MEMORY_PER_GPU: int = 240

    def set_torchpilot_output_dir(self, output_dir: Path) -> None:
        """Set torchpilot output directory."""
        if not output_dir.is_absolute():
            raise ValueError("Torchpilot output directory should be an absolute path.")
        self.TORCHPILOT_OUTPUT_DIR = output_dir

    def get_num_nodes(self, num_gpus: int = 1) -> int:
        """Get number of nodes."""
        return max(1, num_gpus // self.MAX_GPUS_PER_NODE)

    def get_gpus_per_node(self, num_gpus: int = 1) -> int:
        """Get number of GPUs per node."""
        return num_gpus // self.get_num_nodes(num_gpus)

    def get_max_cpus_per_node(self, num_gpus: int = 1) -> int:
        """Get max CPUs per node."""
        return num_gpus * self.MAX_CPUS_PER_GPU // self.get_num_nodes(num_gpus)

    def get_max_memory_per_node(self, num_gpus: int = 1) -> int:
        """Get max memory per node."""
        return num_gpus * self.MAX_MEMORY_PER_GPU // self.get_num_nodes(num_gpus)


class NtsCtlResourceTemplate(BaseModel):
    CPU: int
    GPUs: Dict[str, int]  # GPU_TYPE: num_gpus
    Memory: int


class NtsCtlJobTemplate(BaseModel):
    NumNodes: int
    Resource: NtsCtlResourceTemplate
    Priority: NtsJobPriorities = NtsJobPriorities.NORMAL
    Type: NtsInstanceType = NtsInstanceType.JOB
    Image: str = NtsCtlConstants.IMAGE_NAME
    WorkingDir: Path
    Command: str
    as_root: bool = False
    Env: Dict[str, str] = {}  # Pass environment variables to the job


class NtsCtlCommands:
    @classproperty
    def get_logs(cls) -> str:  # noqa: N805
        """Get log, -f <instance_name>."""
        return "ntsctl logs -f"

    @classproperty
    def quota(cls) -> str:  # noqa: N805
        """Get GPUs quota/usages."""
        return "ntsctl quota"

    @classproperty
    def run_job_with_template(cls) -> str:  # noqa: N805
        """Run job with job template file."""
        return "ntsctl apply -f"

    @classproperty
    def user_mount_lists(cls) -> str:  # noqa: N805
        """Get user mount lists."""
        return "ntsctl user mount-list"

    @classproperty
    def get_job_template(cls) -> str:  # noqa: N805
        """Get job template."""
        return "ntsctl apply --template"
