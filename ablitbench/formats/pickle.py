from __future__ import annotations

import pickletools
from collections.abc import Iterator
from typing import IO, Any

Symbol = tuple[str, Any]
BENIGN_GLOBALS = {
    "builtins.set",
    "builtins.frozenset",
    "builtins.list",
    "builtins.dict",
    "builtins.tuple",
    "builtins.str",
    "builtins.bytes",
    "builtins.bytearray",
    "builtins.complex",
    "builtins.slice",
    "builtins.range",
    "collections.OrderedDict",
    "collections.deque",
}
EXECUTION_GLOBALS = {
    "os.system",
    "posix.system",
    "nt.system",
    "os.popen",
    "subprocess.Popen",
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "builtins.eval",
    "builtins.exec",
    "builtins.compile",
    "builtins.__import__",
    "pickle.loads",
    "pickle.load",
    "dill.loads",
    "dill.load",
    "marshal.loads",
    "importlib.import_module",
}


def global_name(argument: Any) -> str:
    module, separator, name = str(argument).partition(" ")
    if not separator or not module or not name:
        raise ValueError("Invalid pickle GLOBAL reference")
    if module == "__builtin__":
        module = "builtins"
    return module + "." + name


def global_observation(name: str, opcode: str, offset: int) -> dict[str, Any] | None:
    if name in BENIGN_GLOBALS:
        return None
    record: dict[str, Any] = {
        "opcode": opcode,
        "argument": name,
        "global": name,
        "offset": offset,
        "issue": "pickle execution sink reference"
        if name in EXECUTION_GLOBALS
        else "unresolved pickle global reference",
        "interpretation": "Static reconstruction only; deserialization was not executed",
    }
    if name not in EXECUTION_GLOBALS:
        record["coverage_gap"] = "External global behavior not validated: " + name
    return record


def pop(stack: list[Symbol]) -> Symbol:
    if not stack or stack[-1][0] == "mark":
        raise ValueError("Malformed pickle stack")
    return stack.pop()


def marked(stack: list[Symbol]) -> list[Symbol]:
    values = []
    while stack and stack[-1][0] != "mark":
        values.append(stack.pop())
    if not stack:
        raise ValueError("Missing pickle MARK")
    stack.pop()
    return list(reversed(values))


def inspect_pickle(stream: IO[bytes]) -> Iterator[dict[str, Any]]:
    while True:
        start = stream.tell()
        if not stream.read(1):
            break
        stream.seek(start)
        stack: list[Symbol] = []
        memo: dict[int, Symbol] = {}
        frame_end: int | None = None

        for opcode, argument, position in pickletools.genops(stream):
            name = opcode.name
            offset = position if position is not None else stream.tell()
            if frame_end is not None:
                if offset >= frame_end:
                    frame_end = None
                elif name == "FRAME" or stream.tell() > frame_end:
                    raise ValueError("Pickle opcode crosses declared frame boundary")
            if name == "FRAME":
                frame_end = offset + 9 + int(str(argument))
                continue
            if name == "PROTO":
                if not isinstance(argument, int) or argument > max(
                    item.proto for item in pickletools.opcodes
                ):
                    raise ValueError("Unsupported pickle protocol")
                continue
            if name == "MARK":
                stack.append(("mark", None))
            elif name == "STOP":
                if frame_end is not None and stream.tell() != frame_end:
                    raise ValueError("Pickle frame length differs from stream")
                pop(stack)
                if stack:
                    raise ValueError("Unbalanced pickle stack at STOP")
            elif name in {"GLOBAL", "INST", "STACK_GLOBAL"}:
                arguments = marked(stack) if name == "INST" else None
                if name == "STACK_GLOBAL":
                    item, module = pop(stack), pop(stack)
                    if (
                        item[0] != "literal"
                        or module[0] != "literal"
                        or not isinstance(item[1], str)
                        or not isinstance(module[1], str)
                    ):
                        raise ValueError("STACK_GLOBAL requires string module and name")
                    target = global_name(module[1] + " " + item[1])
                else:
                    target = global_name(argument)
                record = global_observation(target, name, offset)
                if record:
                    yield record
                stack.append(("instance" if arguments is not None else "global", target))
            elif name in {"GET", "BINGET", "LONG_BINGET"}:
                index = int(str(argument))
                if index not in memo:
                    raise ValueError("Unknown pickle memo reference")
                stack.append(memo[index])
            elif name in {"PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"}:
                if not stack or stack[-1][0] == "mark":
                    raise ValueError("Cannot memoize empty pickle stack")
                memo[len(memo) if name == "MEMOIZE" else int(str(argument))] = stack[-1]
            elif name in {"REDUCE", "NEWOBJ", "NEWOBJ_EX", "OBJ"}:
                keywords = pop(stack) if name == "NEWOBJ_EX" else None
                if name == "OBJ":
                    items = marked(stack)
                    if not items:
                        raise ValueError("OBJ requires a class")
                    function, args = items[0], ("tuple", items[1:])
                else:
                    args, function = pop(stack), pop(stack)
                    if args[0] != "tuple":
                        raise ValueError("Pickle constructor arguments must be tuple")
                callable_name = function[1] if function[0] == "global" else None
                if callable_name not in BENIGN_GLOBALS or keywords is not None:
                    record = {
                        "opcode": name,
                        "offset": offset,
                        "call": callable_name,
                        "issue": "pickle execution sink invocation"
                        if callable_name in EXECUTION_GLOBALS
                        else "unresolved pickle callable invocation",
                        "arguments": [
                            value
                            if kind == "literal"
                            and isinstance(value, (str, int, float, bool, type(None)))
                            else {"kind": kind}
                            for kind, value in args[1]
                        ],
                        "interpretation": "Reconstructed deserialization operation; not executed",
                    }
                    if callable_name not in EXECUTION_GLOBALS:
                        record["coverage_gap"] = "Callable or constructor semantics not validated"
                    yield record
                stack.append(("instance", callable_name))
            elif name == "BUILD":
                pop(stack)
                instance = pop(stack)
                yield {
                    "opcode": name,
                    "offset": offset,
                    "global": instance[1] if instance[0] == "instance" else None,
                    "issue": "pickle object state restoration",
                    "coverage_gap": "Object state restoration may invoke custom methods",
                }
                stack.append(instance)
            elif name in {"EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID", "NEXT_BUFFER"}:
                if name == "BINPERSID":
                    pop(stack)
                yield {
                    "opcode": name,
                    "offset": offset,
                    "argument": str(argument),
                    "issue": "external pickle resolution",
                    "coverage_gap": "Extension, persistent ID, or buffer requires external state",
                }
                stack.append(("unknown", None))
            elif name == "READONLY_BUFFER":
                value = pop(stack)
                stack.append(value)
            elif name in {"EMPTY_LIST", "EMPTY_DICT", "EMPTY_TUPLE", "EMPTY_SET"}:
                stack.append((name.removeprefix("EMPTY_").lower(), []))
            elif name in {"LIST", "DICT", "TUPLE", "FROZENSET"}:
                values = marked(stack)
                if name == "DICT" and len(values) % 2:
                    raise ValueError("Odd pickle dictionary item count")
                stack.append((name.lower(), values))
            elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
                values = [pop(stack) for _ in range(int(name[-1]))]
                stack.append(("tuple", list(reversed(values))))
            elif name in {"APPEND", "SETITEM"}:
                values = [pop(stack) for _ in range(2 if name == "SETITEM" else 1)]
                if not stack or stack[-1][0] not in {"list", "dict", "instance"}:
                    raise ValueError("Invalid pickle mutation target")
                if stack[-1][0] != "instance":
                    stack[-1][1].extend(reversed(values))
            elif name in {"APPENDS", "SETITEMS", "ADDITEMS"}:
                values = marked(stack)
                if not stack or stack[-1][0] not in {"list", "dict", "set", "instance"}:
                    raise ValueError("Invalid pickle container target")
                if name == "SETITEMS" and len(values) % 2:
                    raise ValueError("Odd pickle dictionary item count")
                if stack[-1][0] != "instance":
                    stack[-1][1].extend(values)
            elif name == "POP":
                pop(stack)
            elif name == "POP_MARK":
                marked(stack)
            elif name == "DUP":
                value = pop(stack)
                stack.extend((value, value))
            elif name == "NONE":
                stack.append(("literal", None))
            elif name in {"NEWTRUE", "NEWFALSE"}:
                stack.append(("literal", name == "NEWTRUE"))
            elif name in {
                "INT",
                "BININT",
                "BININT1",
                "BININT2",
                "LONG",
                "LONG1",
                "LONG4",
                "STRING",
                "BINSTRING",
                "SHORT_BINSTRING",
                "BINBYTES",
                "SHORT_BINBYTES",
                "BINBYTES8",
                "BYTEARRAY8",
                "UNICODE",
                "SHORT_BINUNICODE",
                "BINUNICODE",
                "BINUNICODE8",
                "FLOAT",
                "BINFLOAT",
            }:
                stack.append(("literal", argument))
            else:
                raise ValueError("Unsupported pickle opcode: " + name)
