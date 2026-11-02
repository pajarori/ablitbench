from __future__ import annotations

import json
import math
from typing import Any, BinaryIO, NoReturn, TextIO


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON object key")
        value[key] = item
    return value


def invalid_constant(value: str) -> NoReturn:
    raise ValueError("Non-finite JSON number: " + value)


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds finite float capacity")
    return number


def loads(value: str | bytes | bytearray) -> Any:
    try:
        return json.loads(
            value,
            object_pairs_hook=unique_pairs,
            parse_constant=invalid_constant,
            parse_float=finite_float,
        )
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds interpreter parser capacity") from error


def load(stream: BinaryIO | TextIO) -> Any:
    return loads(stream.read())


def equal(left: Any, right: Any) -> bool:
    pending = [(left, right)]
    while pending:
        first, second = pending.pop()
        if type(first) in {int, float} and type(second) in {int, float}:
            if first != second:
                return False
        elif type(first) is not type(second):
            return False
        elif isinstance(first, dict):
            if first.keys() != second.keys():
                return False
            pending.extend((item, second[key]) for key, item in first.items())
        elif isinstance(first, list):
            if len(first) != len(second):
                return False
            pending.extend(zip(first, second, strict=True))
        elif first != second:
            return False
    return True
