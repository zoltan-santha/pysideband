from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Step:
    name: str
    method: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepInvocation:
    name: str
    input_from: str | None
    step: Step


@dataclass(frozen=True)
class Config:
    input_file: Path
    project: dict[str, Any]
    workflow: list[StepInvocation]
    parallel: dict[str, Any]
    steps: dict[str, Step]
