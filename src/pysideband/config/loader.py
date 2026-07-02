from __future__ import annotations

from re import compile as regex_compile
from json import dumps as json_dumps
from yaml import safe_load as yaml_safe_load
from pathlib import Path
from typing import Any, Mapping

from pysideband.config.configs import Config
from pysideband.config.step import collect_steps
from pysideband.config.workflow import build_workflow

_SEQUENCE_WITH_UNIT_REGEX_PATTERN = regex_compile(
    r"^(?P<prefix>\s*[^#:\n][^:\n]*:\s*)(?P<value>\[[^\]]+\]\s+[A-Za-z][A-Za-z0-9_/-]*)(?P<suffix>\s*(?:#.*)?)$"
)


def _transform_raw_content(raw_content: str) -> str:
    out: list[str] = []
    for line in raw_content.splitlines():
        match = _SEQUENCE_WITH_UNIT_REGEX_PATTERN.match(line)
        if match:
            prefix = match.group("prefix")
            value = match.group("value")
            suffix = match.group("suffix")
            line = prefix + json_dumps(value.strip()) + suffix
        out.append(line)
    return "\n".join(out) + ("\n" if raw_content.endswith("\n") else "")


def _required_block(
    key: str, raw: Mapping[str, Any], content: dict[str, type]
) -> dict[str, Any]:
    block = raw.get(key)
    if not isinstance(block, Mapping):
        raise ValueError(f"The input file is missing the mandatory '{key}' block.")
    for subkey, value in content.items():
        if subkey not in block:
            raise ValueError(
                f"The '{key}' block is missing the mandatory '{subkey}' entry."
            )
        if not isinstance(block[subkey], value):
            raise ValueError(
                f"The '{key}' block entry '{subkey}' must be of type {value.__name__}."
            )
    return dict(block)


def _optional_block(key: str, raw: Mapping[str, Any], block_type: type) -> Any:
    block = raw.get(key, {}) or {}
    if not isinstance(block, block_type):
        raise ValueError(
            f"The optional '{key}' block must be of type {block_type.__name__}."
        )
    return block


def load_config(input_file: Path) -> Config:
    input_file = input_file.resolve()
    raw_content = input_file.read_text(encoding="utf-8")
    transformed_content = _transform_raw_content(raw_content)
    content = yaml_safe_load(transformed_content)

    project = _required_block("project", content, {"workflow": list})

    parallel = _optional_block("parallel", content, Mapping)

    steps = collect_steps(content)
    if not steps:
        raise ValueError("No steps were defined in the input file.")

    workflow = build_workflow(
        project["workflow"],
        steps,
    )

    return Config(
        input_file=input_file,
        project=project,
        workflow=workflow,
        parallel=parallel,
        steps=steps,
    )
