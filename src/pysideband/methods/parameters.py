from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from re import compile as regex_compile
from re import escape as regex_escape
from re import finditer as regex_finditer
from re import Pattern as RegexPattern


class UserInputError(Exception):
    """Exception raised for errors in the user input."""


class InternalError(Exception):
    """Exception raised for internal errors in the code."""


@dataclass
class Parameter:
    attr: str
    type: type
    type_kwargs: dict[str, Any] = field(default_factory=dict)
    length: int | None = None
    default: Any = None


@dataclass
class Field:
    syntax: str
    description: str
    params: dict[str, Parameter]

    def _literal_to_regex(self, literal: str) -> str:
        parts = []
        last_index = 0

        for match in regex_finditer(r"\s+", literal):
            parts.append(regex_escape(literal[last_index:match.start()]))
            parts.append(r"\s+")
            last_index = match.end()
        
        parts.append(regex_escape(literal[last_index:]))
        return "".join(parts)

    def _syntax_to_regex(self) -> RegexPattern:
        names = set()

        def parse(i: int, stop: str | None = None) -> tuple[str, int]:
            parts = []
            literal = []

            def flush_literal():
                if literal:
                    parts.append(self._literal_to_regex("".join(literal)))
                    literal.clear()
            
            while i < len(self.syntax):
                char = self.syntax[i]

                if stop is not None and char == stop:
                    break

                if char == "\\":
                    if i+1 >= len(self.syntax):
                        literal.append("\\")
                        i += 1
                    else:
                        literal.append(self.syntax[i+1])
                        i += 2
                elif char == "[":
                    flush_literal()
                    inner, i = parse(i+1, "]")

                    if i >= len(self.syntax) or self.syntax[i] != "]":
                        raise InternalError(f"Unmatched '[' in syntax: {self.syntax}")

                    parts.append(f"(?:{inner})?")
                    i += 1
                elif char == "]":
                    if stop is None:
                        raise InternalError(f"Unmatched ']' in syntax: {self.syntax}")
                    break
                elif char == "$":
                    flush_literal()

                    match = regex_compile(r"\$([A-Za-z_]\w*)").match(self.syntax, i)
                    if not match:
                        literal.append(char)
                        i += 1
                        continue

                    name = match.group(1)
                    if name in names:
                        raise InternalError(f"Duplicate parameter name '{name}' in syntax: {self.syntax}")
                    
                    names.add(name)
                    parts.append(f"(?P<{name}>.+?)")
                    i = match.end()
                else:
                    literal.append(char)
                    i += 1

            flush_literal()
            return "".join(parts), i
        
        regex_str, i = parse(0)

        if i != len(self.syntax):
            raise InternalError(f"Unexpected closing bracket ']' in syntax: {self.syntax}")
        
        return regex_compile(r"\A" + regex_str + r"\Z")

    def get_params(self, value: str) -> dict[str, str]:
        regex_pattern = self._syntax_to_regex()
        matched = regex_pattern.match(value)
        if not matched:
            raise UserInputError(f"Value '{value}' does not match the syntax '{self.syntax}'")
        return matched.groupdict()


class MethodParameters:
    """Abstract class for method parameters. Each method should implement its own subclass of this class to define its parameters."""


def _flatten_parameters(parameters: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in parameters.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_parameters(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _get_value(parameters: dict[str, Any], key: str, sep: str = ".") -> str:
    key_parts = key.split(sep)
    current_dict = parameters
    for part in key_parts:
        if part in current_dict:
            current_dict = current_dict[part]
        else:
            raise KeyError(f"Key '{key}' not found in parameters.")
    return str(current_dict)


def get_parameters(parameters_class: type[MethodParameters], defined_parameters: dict[str, Any], parameters: dict[str, Any]) -> MethodParameters:
    method_parameters = parameters_class()

    defined_params_flat: dict[str, Field] = _flatten_parameters(defined_parameters)
    for key, field in defined_params_flat.items():
        try:
            value = _get_value(parameters, key)
            params = field.get_params(value)
        except KeyError:
            for param in field.params.values():
                if param.default is not None:
                    setattr(method_parameters, param.attr, param.default)
                else:
                    raise UserInputError(f"Required parameter '{key}' is missing.")
            continue

        for param_key, param_value in params.items():
            param = field.params[param_key]
            
            if not hasattr(method_parameters, param.attr):
                raise InternalError(f"Attribute '{param.attr}' not found in MethodParameters class.")
            
            if param_value is None:
                if param.default is not None:
                    setattr(method_parameters, param.attr, param.default)
                else:
                    raise UserInputError(f"Required parameter '{param_key}' is missing.")
                continue
            
            try:
                typed_value = param.type(param_value, **param.type_kwargs) if param.type is not None else param_value
            except Exception as e:
                raise UserInputError(f"Invalid value defined for parameter '{key}': {param_value}")
            
            if param.length is not None and len(typed_value) != param.length:
                raise UserInputError(f"Expected length {param.length} for parameter '{param.attr}', got {len(typed_value)}")
            
            setattr(method_parameters, param.attr, typed_value)

    return method_parameters
