from __future__ import annotations

from typing import Any


class MPIContext:
    def __init__(self) -> None:
        try:
            from mpi4py import MPI
        except Exception as e:
            self.MPI = None
            self.comm = None
            self.rank = 0
            self.size = 1
            self.available = False
            print(f"MPI not available: {e}")
        else:
            self.MPI = MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.available = True

    @property
    def enabled(self) -> bool:
        return self.available and self.size > 1

    @property
    def is_root(self) -> bool:
        return self.rank == 0
    
    def barrier(self) -> None:
        if self.enabled:
            self.comm.barrier()
    
    def gather(self, data: Any, root: int = 0) -> list[Any]:
        if self.enabled:
            return self.comm.gather(data, root=root)
        else:
            return [data]
    
    def bcast(self, data: Any, root: int = 0) -> Any:
        if self.enabled:
            return self.comm.bcast(data, root=root)
        else:
            return data
