from __future__ import annotations

import os
import sys
from subprocess import run as run_subprocess
from typing import Callable
from pathlib import Path

from pysideband.launcher.config import LaunchConfig


def _detect_mpi_environment() -> bool:
    """Detect if the current process is running in an MPI environment."""
    mpi_env_vars = [
        "OMPI_COMM_WORLD_SIZE",  # OpenMPI
        "PMI_SIZE",  # MPICH
        "PMIX_SIZE",  # PMIx
        "MPI_LOCALNRANKS",  # Intel MPI
        "SLURM_NTASKS",  # SLURM
    ]
    return any(var in os.environ for var in mpi_env_vars)


def _build_parallel_launch_command(launch_config: LaunchConfig) -> list[str]:
    if Path(launch_config.external_launcher).name == "srun":
        rank_arg_flag = "--ntasks"
    else:
        rank_arg_flag = "-n"
    return [
        launch_config.external_launcher,
        rank_arg_flag,
        str(launch_config.ranks),
        *[str(arg) for arg in launch_config.external_launcher_args],
        sys.executable,
        "-m",
        "pysideband.__main__",
        *[str(arg) for arg in launch_config.launch_args],
    ]


def launch(fn: Callable[[], int | None], launch_config: LaunchConfig) -> int:
    inside_internal_run = os.environ.get(launch_config.internal_run_env_var) == "1"
    inside_parallel_run = _detect_mpi_environment()

    should_relaunch = (
        launch_config.ranks > 1 and not inside_internal_run and not inside_parallel_run
    )

    if should_relaunch:
        command = _build_parallel_launch_command(launch_config)
        env = os.environ.copy()
        env[launch_config.internal_run_env_var] = "1"
        competed_process = run_subprocess(command, env=env)
        return int(competed_process.returncode)

    def run() -> int:
        result = fn()
        return 0 if result is None else int(result)

    return run()
