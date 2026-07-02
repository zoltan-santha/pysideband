from __future__ import annotations

from typing import Any, Mapping
from copy import deepcopy
from json import dumps as json_dumps
from hashlib import sha1


def deepcopy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value or {}))


def stable_hash(value: Any, length: int = 8) -> str:
    encoded_value = json_dumps(
        value, sort_keys=True, default=str, separators=(",", ":")
    ).encode("utf-8")
    return sha1(encoded_value).hexdigest()[:length]
