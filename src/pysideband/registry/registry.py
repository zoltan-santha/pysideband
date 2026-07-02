from __future__ import annotations

from importlib import import_module
from typing import Any


class MethodRegistry:
    def __init__(self) -> None:
        self._methods: dict[str, type | str] = {}

    def register(self, name: str, method_cls_or_path: type | str) -> None:
        self._methods[name] = method_cls_or_path

    def names(self) -> list[str]:
        return sorted(self._methods)

    def get(self, name: str) -> Any:
        if name not in self._methods:
            available_methods = ", ".join(self.names())
            raise ValueError(
                f"Method '{name}' not found. Available methods: {available_methods}"
            )

        method = self._methods[name]
        if isinstance(method, str):
            module_path, class_name = method.rsplit(".", 1)
            module = import_module(module_path)
            method = getattr(module, class_name)
            self._methods[name] = method

        return method

    @classmethod
    def default(cls) -> MethodRegistry:
        registry = cls()
        registry.register(
            "interpolate", "pysideband.methods.interpolation.Interpolation"
        )
        registry.register(
            "singlephonon", "pysideband.methods.singlephonon.SinglePhonon"
        )
        registry.register("multiphonon", "pysideband.methods.multiphonon.MultiPhonon")

        return registry
