from __future__ import annotations

import math
from collections.abc import Hashable
from typing import Any, BinaryIO, TextIO

import yaml
from yaml.nodes import MappingNode


class UniqueSafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        if not isinstance(node, MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None, "Expected a YAML mapping", node.start_mark
            )
        self.flatten_mapping(node)
        value: dict[Hashable, Any] = {}
        for key_node, item_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise yaml.constructor.ConstructorError(
                    None, None, "Unhashable YAML mapping key", key_node.start_mark
                )
            if key in value:
                raise yaml.constructor.ConstructorError(
                    None, None, "Duplicate YAML mapping key or merge override", key_node.start_mark
                )
            value[key] = self.construct_object(item_node, deep=deep)
        return value


def load(stream: str | bytes | BinaryIO | TextIO) -> Any:
    value = yaml.load(stream, Loader=UniqueSafeLoader)
    pending = [value]
    visited = set()
    while pending:
        item = pending.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Non-finite YAML number")
        if id(item) in visited:
            continue
        visited.add(id(item))
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple, set)):
            pending.extend(item)
    return value
