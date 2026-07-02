from __future__ import annotations

from typing import Any, Mapping

from pysideband.config.configs import Step
from pysideband.config.schema import RESERVED_BLOCK_NAMES, RESERVED_STEP_KEYWORDS


def _parse_step(name: str, block: Mapping[str, Any]) -> Step:
    method = str(block["method"])
    parameters = {
        str(key): value
        for key, value in block.items()
        if key not in RESERVED_STEP_KEYWORDS
    }
    return Step(
        name=name,
        method=method,
        parameters=parameters,
    )


def collect_steps(content: Mapping[str, Any]) -> dict[str, Step]:
    steps: dict[str, Step] = {}
    for step_name, step_block in content.items():
        if step_name in RESERVED_BLOCK_NAMES:
            continue
        if isinstance(step_block, Mapping) and "method" in step_block:
            steps[str(step_name)] = _parse_step(str(step_name), step_block)
    return steps
