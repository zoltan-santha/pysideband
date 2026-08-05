from __future__ import annotations

from pysideband import __version__
from pysideband.config import Config
from pysideband.mpi import MPIContext
from pysideband.registry import MethodRegistry
from pysideband.methods import MethodResult
from pysideband.workflow.update import check_for_update


def run_workflow(config: Config) -> None:
    mpi = MPIContext()

    if mpi.is_root: print(
        f"pysideband" "\n"
        f"version: {__version__}",
        flush=True
    )
    
    if mpi.size > 1 and mpi.is_root: print(
        f"(parallel mode) running on {mpi.size} processes" "\n",
        flush=True
    )
    elif mpi.is_root: print(
        f"(serial mode) running on a single process" "\n",
        flush=True
    )

    registry = MethodRegistry.default()

    results: dict[str, MethodResult] = {}

    for step_invocation in config.workflow:
        step = step_invocation.step
        
        if mpi.is_root: print(
            f"Step: {step_invocation.name}" "\n"
            f"  method: {step.method}",
            flush=True
        )

        inputs = (
            results.get(step_invocation.input_from)
            if step_invocation.input_from
            else None
        )

        method_class = registry.get(step.method)
        method = method_class(step.parameters)
        method.apply_input(inputs)

        result: MethodResult = method.run(mpi=mpi)
        results[step_invocation.name] = result

        if mpi.is_root: print("", flush=True)
    
    if mpi.is_root: print(
        f"Workflow completed successfully.",
        flush=True
    )
    
    if mpi.is_root:
        update = check_for_update()
        if update is not None:
            latest_version, current_version = update
            print(
                f"Update available: {latest_version} (current: {current_version})" "\n"
                f"Run: 'pip install --upgrade pysideband' or" "\n"
                f"     'python -m pip install --upgrade pysideband'",
                flush=True
            )
