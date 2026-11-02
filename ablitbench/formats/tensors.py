from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from ablitbench.limits import available_memory

DTYPES = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U64": "<u8",
    "U32": "<u4",
    "U16": "<u2",
    "U8": "u1",
    "BOOL": "?",
    "BF16": "<u2",
}


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key: " + key)
        result[key] = value
    return result


def load_json(stream: BinaryIO) -> Any:
    return json.load(stream, object_pairs_hook=unique_pairs)


def safetensors(path: Path) -> Iterator[dict[str, Any]]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated Safetensors prefix")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > size - 8:
            raise ValueError("Safetensors header exceeds file")
        if header_size > available_memory():
            raise MemoryError("Safetensors header exceeds available memory")
        header = json.loads(stream.read(header_size), object_pairs_hook=unique_pairs)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    ranges = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict):
            raise ValueError("Invalid tensor entry")
        dtype = tensor.get("dtype")
        if dtype not in DTYPES:
            raise NotImplementedError(f"Unsupported Safetensors dtype: {dtype}")
        shape, offsets = tensor["shape"], tensor["data_offsets"]
        if not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape):
            raise ValueError("Invalid tensor shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(v) is not int for v in offsets)
        ):
            raise ValueError("Invalid tensor offsets")
        start, end = offsets
        expected = math.prod(shape) * np.dtype(DTYPES[dtype]).itemsize
        if start < 0 or end < start or end > size - 8 - header_size or end - start != expected:
            raise ValueError("Tensor range/shape exceeds data or has incorrect size")
        ranges.append((start, end, name, dtype, shape))
    position = 0
    for start, end, name, dtype, shape in sorted(ranges):
        if start != position:
            raise ValueError("Tensor data overlaps or contains unindexed bytes")
        position = end
        yield {
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "offset": 8 + header_size + start,
            "count": math.prod(shape),
            "bytes": end - start,
            "quantized": False,
        }
    if position != size - 8 - header_size:
        raise ValueError("Unindexed trailing Safetensors data")


def value_chunks(
    path: Path, offset: int, count: int, dtype: Any, bf16: bool = False
) -> Iterator[Any]:
    if offset < 0 or count < 0:
        raise ValueError("Tensor offset and count must be nonnegative")
    dtype = np.dtype(dtype)
    elements = max(1, min(1024 * 1024, available_memory() // 64) // max(8, dtype.itemsize))
    with path.open("rb") as stream:
        stream.seek(offset)
        while count:
            length = min(count, elements)
            raw = stream.read(length * dtype.itemsize)
            if len(raw) != length * dtype.itemsize:
                raise ValueError("Tensor data truncated during scan")
            values = np.frombuffer(raw, dtype=dtype)
            if bf16:
                values = (values.astype(np.uint32) << 16).view(np.float32)
            yield values.astype(np.float64)
            count -= length


def tensor_chunks(path: Path, tensor: dict[str, Any]) -> Iterator[Any]:
    yield from value_chunks(
        path, tensor["offset"], tensor["count"], DTYPES[tensor["dtype"]], tensor["dtype"] == "BF16"
    )


def statistics(chunks: Iterator[Any]) -> dict[str, Any]:
    count = finite = zeros = 0
    mean = m2 = 0.0
    minimum = maximum = None
    for values in chunks:
        count += values.size
        values = values[np.isfinite(values)]
        n = values.size
        if not n:
            continue
        local_mean = float(values.mean())
        local_m2 = float(np.square(values - local_mean).sum())
        delta = local_mean - mean
        total = finite + n
        m2 += local_m2 + delta * delta * finite * n / total
        mean += delta * n / total
        finite = total
        zeros += int(np.count_nonzero(values == 0))
        low, high = float(values.min()), float(values.max())
        minimum = low if minimum is None else min(minimum, low)
        maximum = high if maximum is None else max(maximum, high)
    return {
        "count": int(count),
        "finite": int(finite),
        "nonfinite": int(count - finite),
        "zeros": zeros,
        "mean": mean if finite and math.isfinite(mean) else None,
        "variance": m2 / finite if finite and math.isfinite(m2 / finite) else None,
        "accumulator_overflow": bool(finite and (not math.isfinite(mean) or not math.isfinite(m2))),
        "min": minimum,
        "max": maximum,
    }


def gguf_descriptor(tensor: Any, endian: Any) -> dict[str, Any]:
    from gguf import GGUFEndian

    plain = tensor.tensor_type.name in {"F32", "F16", "F64", "BF16", "I8", "I16", "I32", "I64"}
    return {
        "name": tensor.name,
        "dtype": tensor.tensor_type.name,
        "shape": tensor.shape.tolist(),
        "bytes": int(tensor.n_bytes),
        "count": int(tensor.n_elements),
        "quantized": not plain,
        "offset": int(tensor.data_offset),
        "storage_dtype": tensor.data.dtype.str,
        "byte_order": "big" if endian == GGUFEndian.BIG else "little",
    }


def gguf_tensors(path: Path, include_statistics: bool = True) -> Iterator[dict[str, Any]]:
    from gguf import GGUFReader

    reader = GGUFReader(path)
    for tensor in reader.tensors:
        record = gguf_descriptor(tensor, reader.endianess)
        record["statistics"] = statistics(gguf_chunks(path, record)) if include_statistics else None
        yield record


def gguf_chunks(path: Path, name: str | dict[str, Any]) -> Iterator[Any]:
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFReader
    from gguf.quants import dequantize

    if isinstance(name, str):
        reader = GGUFReader(path)
        tensor = next(tensor for tensor in reader.tensors if tensor.name == name)
        record = gguf_descriptor(tensor, reader.endianess)
    else:
        record = name
    if not record["quantized"]:
        bf16 = record["dtype"] == "BF16"
        dtype = (
            (">u2" if record["byte_order"] == "big" else "<u2") if bf16 else record["storage_dtype"]
        )
        yield from value_chunks(path, record["offset"], record["count"], dtype, bf16)
        return
    if record["byte_order"] == "big":
        raise NotImplementedError(
            "Big-endian quantized GGUF decoding is unsupported by this adapter"
        )
    qtype = GGMLQuantizationType[record["dtype"]]
    block_elements, block_bytes = GGML_QUANT_SIZES[qtype]
    budget = min(1024 * 1024, available_memory() // 64)
    blocks = max(1, budget // max(block_elements * 8, block_bytes))
    chunk_bytes = blocks * block_bytes
    remaining = record["bytes"]
    if remaining % block_bytes:
        raise ValueError("Quantized tensor data is not block aligned")
    with path.open("rb") as stream:
        stream.seek(record["offset"])
        while remaining:
            length = min(remaining, chunk_bytes)
            raw = stream.read(length)
            if len(raw) != length:
                raise ValueError("Quantized tensor data truncated during scan")
            values = np.frombuffer(raw, dtype=np.uint8).reshape(1, -1)
            yield dequantize(values, qtype).reshape(-1).astype(np.float64)
            remaining -= length


def tensor_diff(left: Iterator[Any], right: Iterator[Any]) -> dict[str, Any]:
    count = 0
    squared = absolute = 0.0
    left_buffer = right_buffer = np.empty(0, dtype=np.float64)
    left_done = right_done = False
    while True:
        if not left_buffer.size and not left_done:
            try:
                left_buffer = next(left).reshape(-1)
            except StopIteration:
                left_done = True
        if not right_buffer.size and not right_done:
            try:
                right_buffer = next(right).reshape(-1)
            except StopIteration:
                right_done = True
        if not left_buffer.size or not right_buffer.size:
            if left_done and right_done:
                break
            if left_done and right_buffer.size or right_done and left_buffer.size:
                raise ValueError("Reference tensor length differs")
            continue
        length = min(left_buffer.size, right_buffer.size)
        delta = left_buffer[:length] - right_buffer[:length]
        squared += float(np.square(delta).sum())
        absolute += float(np.abs(delta).sum())
        count += length
        left_buffer = left_buffer[length:]
        right_buffer = right_buffer[length:]
    return {
        "count": count,
        "l2": float(np.sqrt(squared)) if np.isfinite(squared) else None,
        "mean_absolute": absolute / count if count and np.isfinite(absolute) else None,
    }
