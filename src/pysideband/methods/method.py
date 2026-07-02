from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
from abc import abstractmethod

from pysideband.mpi import MPIContext


@dataclass
class MethodParameter:
    description: str
    type: type
    default: Any = None


@dataclass
class MethodResult:
    output_files: dict[str, Path] = field(default_factory=dict)


class Method:
    @abstractmethod
    def parameter_errors(self, mpi: MPIContext) -> None: ...

    @abstractmethod
    def parameter_warnings(self, mpi: MPIContext) -> None: ...

    @abstractmethod
    def apply_input(self, inputs: MethodResult | None) -> None: ...

    @abstractmethod
    def run(self, mpi: MPIContext) -> MethodResult: ...
