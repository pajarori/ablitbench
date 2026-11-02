from __future__ import annotations

import ast
import math
import struct
from collections.abc import Iterator
from typing import IO, Any

import numpy as np

from ablitbench.formats.pickle import inspect_pickle


def inspect_numpy(stream: IO[bytes]) -> Iterator[dict[str, Any]]:
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("Invalid NumPy magic")
    version = stream.read(2)
    if version not in {b"\x01\x00", b"\x02\x00", b"\x03\x00"}:
        raise ValueError("Unsupported NumPy version")
    width = 2 if version == b"\x01\x00" else 4
    raw = stream.read(width)
    if len(raw) != width:
        raise ValueError("Truncated NumPy header length")
    size = struct.unpack("<H" if width == 2 else "<I", raw)[0]
    header = stream.read(size)
    if len(header) != size:
        raise ValueError("Truncated NumPy header")
    tree = ast.parse(header.decode("utf-8" if version == b"\x03\x00" else "latin1"), mode="eval")
    if not isinstance(tree.body, ast.Dict):
        raise ValueError("Invalid NumPy header mapping")
    keys = [
        key.value
        for key in tree.body.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]
    if len(keys) != len(tree.body.keys) or len(set(keys)) != len(keys):
        raise ValueError("Invalid or duplicate NumPy header keys")
    metadata = ast.literal_eval(tree)
    if not isinstance(metadata, dict) or set(metadata) != {"descr", "fortran_order", "shape"}:
        raise ValueError("Invalid NumPy header fields")
    shape = metadata["shape"]
    if not isinstance(shape, tuple) or any(type(value) is not int or value < 0 for value in shape):
        raise ValueError("Invalid NumPy shape")
    if not isinstance(metadata["fortran_order"], bool):
        raise ValueError("Invalid NumPy order")
    dtype = np.dtype(metadata["descr"])
    if dtype.hasobject:
        yield {"issue": "NumPy object array uses pickle", "dtype": str(dtype)}
        yield from inspect_pickle(stream)
    else:
        expected = math.prod(shape) * dtype.itemsize
        begin = stream.tell()
        stream.seek(0, 2)
        if stream.tell() - begin != expected:
            raise ValueError("NumPy data length differs from header")
        yield {
            "issue": "format metadata",
            "format": "numpy",
            "scope": "non-object dtype and payload length validated",
            "dtype": str(dtype),
            "shape": shape,
        }
