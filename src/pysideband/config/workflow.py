from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pysideband.config.configs import Step, StepInvocation
from pysideband.config.schema import RESERVED_STEP_INVOCATION_KEYWORDS
from pysideband.config.utils import deepcopy_mapping, stable_hash


def _normalize_workflow_item(item: Any) -> tuple[str, dict[str, Any], list[Any]]:
    if isinstance(item, str):
        return item, {}, []
    if not isinstance(item, Mapping):
        raise ValueError(
            f"Invalid workflow item: {item}. Must be a string or a mapping."
        )
    if len(item) != 1:
        raise ValueError(
            f"Invalid workflow item: {item}. Must contain exactly one key."
            "Use '- step_name:' followed by overrides or nested child steps."
        )
    step_name, value = next(iter(item.items()))
    step_name = str(step_name)

    if value is None:
        return step_name, {}, []

    if isinstance(value, list):
        return step_name, {}, list(value)

    if not isinstance(value, Mapping):
        raise ValueError(
            f"Invalid workflow item: {item}. The value must be a mapping of overrides or a list of nested child steps."
        )

    overrides = deepcopy_mapping(value)
    children = overrides.pop("then", [])
    if children is None:
        children = []
    if not isinstance(children, list):
        raise ValueError(
            f"Invalid workflow item: {item}. The 'then' key must be a list of nested child steps."
        )

    return step_name, overrides, children


def _make_step_invocation_name(
    base_name: str,
    parent_name: str | None,
    overrides: dict[str, Any],
    used_step_names: set[str],
) -> str:
    explicit_name = overrides.pop("as", None)
    if explicit_name is not None:
        name = str(explicit_name)
    else:
        clean_overrides = {
            key: value
            for key, value in overrides.items()
            if key not in RESERVED_STEP_INVOCATION_KEYWORDS
        }
        if clean_overrides:
            name = f"{base_name}__{stable_hash(clean_overrides)}"
        else:
            name = base_name

    if name in used_step_names:
        if explicit_name is not None:
            raise ValueError(f"Duplicate workflow invocation name from 'as': {name!r}")
        prefix = f"{parent_name}__" if parent_name else ""
        candidate_name = f"{prefix}{name}"
        if candidate_name in used_step_names:
            suffix = 2
            candidate_name = f"{prefix}{name}__{suffix}"
            while candidate_name in used_step_names:
                suffix += 1
                candidate_name = f"{prefix}{name}__{suffix}"
        name = candidate_name

    return name


@dataclass
class _WorkflowBuilder:
    base_steps: Mapping[str, Step]
    step_invocations: list[StepInvocation] = field(default_factory=list)
    step_invocations_dict: dict[str, StepInvocation] = field(default_factory=dict)

    def append_to_workflow(
        self, item: Any, parent_name: str | None, depth: int
    ) -> None:
        step_name, overrides, children = _normalize_workflow_item(item)

        if step_name not in self.base_steps:
            available_steps = ", ".join(sorted(self.base_steps.keys()))
            raise ValueError(
                f"Step '{step_name}' is not defined in the 'steps' block. "
                f"Available steps: {available_steps}"
            )

        step_invocation_name = _make_step_invocation_name(
            base_name=step_name,
            parent_name=parent_name,
            overrides=overrides,
            used_step_names=set(self.step_invocations_dict.keys()),
        )

        referenced_step = self.base_steps[step_name]

        self._add_step_invocation(
            StepInvocation(
                name=step_invocation_name,
                input_from=parent_name,
                step=Step(
                    name=referenced_step.name,
                    method=referenced_step.method,
                    parameters={**referenced_step.parameters, **overrides},
                ),
            )
        )

        for child in children:
            self.append_to_workflow(
                child, parent_name=step_invocation_name, depth=depth + 1
            )

    def workflow(self) -> list[StepInvocation]:
        return self.step_invocations

    def _add_step_invocation(self, step_invocation: StepInvocation) -> None:
        self.step_invocations.append(step_invocation)
        self.step_invocations_dict[step_invocation.name] = step_invocation


def build_workflow(
    workflow_block: list[Any],
    steps: Mapping[str, Step],
) -> list[StepInvocation]:
    if not workflow_block:
        raise ValueError(
            "The 'workflow' block must contain at least one step invocation."
        )

    builder = _WorkflowBuilder(base_steps=steps)

    for item in workflow_block:
        builder.append_to_workflow(item, parent_name=None, depth=0)

    return builder.workflow()
