from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import platform
import shutil

from pysideband.config import Config

_INTERNAL_RUN_ENV_VAR = "PROJECT_NAME_INTERNAL_RUN"


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _get_default_launcher() -> str:
    if shutil.which("srun"):
        return "srun"

    system = platform.system()
    if system == "Linux":
        for launcher in ["mpirun", "mpiexec"]:
            if shutil.which(launcher):
                return launcher
    elif system == "Darwin":
        for launcher in ["mpirun", "mpiexec"]:
            if shutil.which(launcher):
                return launcher
    elif system == "Windows":
        for launcher in ["mpiexec", "mpiexec.exe"]:
            if shutil.which(launcher):
                return launcher

    return ""


class LauncherError(RuntimeError):
    """Exception raised for errors in the launcher configuration."""


@dataclass
class LaunchConfig:
    ranks: int = 1
    external_launcher: str = "mpirun"
    external_launcher_args: list[str] = field(default_factory=list)
    internal_run_env_var: str = _INTERNAL_RUN_ENV_VAR
    launch_args: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: Config, args: list[str]) -> LaunchConfig:
        parallel = config.parallel

        ranks = int(parallel.get("ranks", 1))
        if ranks < 1:
            raise LauncherError("Number of ranks must be at least 1.")

        default_launcher = _get_default_launcher()
        external_launcher = parallel.get("launcher", default_launcher)
        if ranks > 1 and not external_launcher:
            raise LauncherError(
                "No external launcher specified and no default launcher found."
            )

        external_launcher_args = _as_str_list(parallel.get("launcher_args", []))

        return cls(
            ranks=ranks,
            external_launcher=external_launcher,
            external_launcher_args=external_launcher_args,
            launch_args=args,
        )
