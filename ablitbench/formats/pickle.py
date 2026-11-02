from __future__ import annotations

import pickletools
from collections.abc import Iterator
from typing import IO

EXECUTION_OPS = {
    "GLOBAL",
    "STACK_GLOBAL",
    "REDUCE",
    "BUILD",
    "INST",
    "OBJ",
    "NEWOBJ",
    "NEWOBJ_EX",
    "EXT1",
    "EXT2",
    "EXT4",
    "PERSID",
    "BINPERSID",
}


def inspect_pickle(stream: IO[bytes]) -> Iterator[dict[str, str | int]]:
    while True:
        start = stream.tell()
        if not stream.read(1):
            break
        stream.seek(start)
        for opcode, argument, position in pickletools.genops(stream):
            if opcode.name in EXECUTION_OPS:
                yield {
                    "opcode": opcode.name,
                    "argument": str(argument),
                    "offset": position if position is not None else stream.tell(),
                }
