from __future__ import annotations

import base64
import binascii
import hashlib
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from ablitbench.formats.snapshot import secure_open

PROTOCOLS = {"openai-chat", "openai-responses", "anthropic-messages"}
IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
AUDIO_TYPES = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}
DOCUMENT_TYPES = {
    "application/pdf",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/csv",
    "application/x-iif",
    "application/msword",
    "application/rtf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.ms-excel",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.presentation",
    "application/vnd.oasis.opendocument.spreadsheet",
    "application/vnd.apple.pages",
    "application/vnd.apple.keynote",
    "application/vnd.apple.numbers",
    "application/vnd.apple.iwork",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
}


class UnsupportedContent(ValueError):
    pass


def media_type(value: Any) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+", value) is None
    ):
        raise UnsupportedContent("Attachment requires an explicit MIME type without parameters")
    return value.casefold()


def document_type(value: str) -> bool:
    return value in DOCUMENT_TYPES or value.startswith("text/")


def filename(block: dict[str, Any], mime: str) -> str:
    value = block.get("filename")
    if value is None:
        return "attachment" + (mimetypes.guess_extension(mime) or ".bin")
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or any(char in value for char in "/\\\x00\r\n")
    ):
        raise UnsupportedContent("Attachment filename must be a plain filename")
    return value


def attachment(block: dict[str, Any]) -> tuple[str, str, bytes]:
    if set(block) - {
        "type",
        "media_type",
        "data",
        "filename",
        "sha256",
        "detail",
        "id",
        "transcript",
        "format",
        "sample_rate",
        "channels",
        "complete",
    }:
        raise UnsupportedContent("Unknown normalized attachment fields")
    mime = media_type(block.get("media_type"))
    encoded = block.get("data")
    if not isinstance(encoded, str):
        raise UnsupportedContent("Attachment data must be a base64 string")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise UnsupportedContent("Attachment contains invalid base64 data") from error
    if not data:
        raise UnsupportedContent("Empty attachment is not a supported media input")
    expected = block.get("sha256")
    if expected is not None and expected != hashlib.sha256(data).hexdigest():
        raise UnsupportedContent("Attachment checksum differs from normalized content identity")
    for key in ("id", "transcript", "format"):
        if block.get(key) is not None and not isinstance(block[key], str):
            raise UnsupportedContent("Attachment metadata must be text: " + key)
    for key in ("sample_rate", "channels"):
        if block.get(key) is not None and (type(block[key]) is not int or block[key] <= 0):
            raise UnsupportedContent("Attachment audio metadata must be positive: " + key)
    if block.get("complete") is not None and type(block["complete"]) is not bool:
        raise UnsupportedContent("Attachment completion metadata must be boolean")
    if "filename" in block:
        filename(block, mime)
    detail = block.get("detail")
    if detail is not None and (
        not isinstance(detail, str) or detail not in {"auto", "low", "high"}
    ):
        raise UnsupportedContent("Unsupported explicit media detail")
    return mime, encoded, data


def encode_content(protocol: str, content: Any, role: str = "user") -> Any:
    if protocol == "anthropic":
        protocol = "anthropic-messages"
    if protocol not in PROTOCOLS:
        raise UnsupportedContent("Unknown multimodal protocol")
    if role not in {"user", "assistant", "system", "developer", "tool"}:
        raise UnsupportedContent("Unknown message role")
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise UnsupportedContent("Message content must be text or normalized content blocks")
    encoded = []
    for block in content:
        if not isinstance(block, dict):
            raise UnsupportedContent("Content block must be a mapping")
        kind = block.get("type")
        if kind == "text":
            if set(block) != {"type", "text"} or not isinstance(block.get("text"), str):
                raise UnsupportedContent("Malformed normalized text block")
            text_type = (
                ("output_text" if role == "assistant" else "input_text")
                if protocol == "openai-responses"
                else "text"
            )
            value: dict[str, Any] = {"type": text_type, "text": block["text"]}
            if text_type == "output_text":
                value["annotations"] = []
            encoded.append(value)
            continue
        if not isinstance(kind, str) or kind not in {"image", "audio", "video", "document"}:
            raise UnsupportedContent("Unknown normalized content block type")
        if role != "user":
            raise UnsupportedContent("This protocol adapter accepts media only in user messages")
        if block.get("complete") is False:
            raise UnsupportedContent(
                "Incomplete output media cannot be promoted to a complete input"
            )
        mime, data, decoded = attachment(block)
        if kind == "video":
            raise UnsupportedContent("Direct video input is not supported by the selected protocol")
        if kind == "audio":
            if protocol != "openai-chat" or mime not in AUDIO_TYPES:
                raise UnsupportedContent(
                    "This protocol does not support the requested input audio format"
                )
            if "detail" in block:
                raise UnsupportedContent("Audio detail is not a supported protocol field")
            encoded.append(
                {"type": "input_audio", "input_audio": {"data": data, "format": AUDIO_TYPES[mime]}}
            )
            continue
        if kind == "image":
            if mime not in IMAGE_TYPES:
                raise UnsupportedContent("Image MIME type is unsupported by this protocol")
            if protocol == "anthropic-messages":
                if "detail" in block:
                    raise UnsupportedContent("Anthropic image input has no detail field")
                value = {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": data},
                }
            elif protocol == "openai-chat":
                image = {"url": "data:" + mime + ";base64," + data}
                if "detail" in block:
                    image["detail"] = block["detail"]
                value = {"type": "image_url", "image_url": image}
            else:
                value = {"type": "input_image", "image_url": "data:" + mime + ";base64," + data}
                if "detail" in block:
                    value["detail"] = block["detail"]
            encoded.append(value)
            continue
        name = filename(block, mime)
        if protocol == "openai-chat":
            if mime != "application/pdf" or "detail" in block:
                raise UnsupportedContent(
                    "Chat Completions file input supports PDF without a detail field"
                )
            value = {
                "type": "file",
                "file": {"filename": name, "file_data": "data:" + mime + ";base64," + data},
            }
        elif protocol == "openai-responses":
            if not document_type(mime):
                raise UnsupportedContent("Responses document MIME type is not supported")
            value = {
                "type": "input_file",
                "filename": name,
                "file_data": "data:" + mime + ";base64," + data,
            }
            if "detail" in block:
                if mime != "application/pdf":
                    raise UnsupportedContent("File detail is supported only for PDF inputs")
                value["detail"] = block["detail"]
        elif mime == "application/pdf":
            if "detail" in block:
                raise UnsupportedContent("Anthropic PDF input has no detail field")
            value = {
                "type": "document",
                "source": {"type": "base64", "media_type": mime, "data": data},
            }
        elif mime == "text/plain":
            if "detail" in block:
                raise UnsupportedContent("Anthropic text document has no detail field")
            try:
                text = decoded.decode("utf-8")
            except UnicodeDecodeError as error:
                raise UnsupportedContent("Anthropic text document must be UTF-8") from error
            value = {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": text},
            }
        else:
            raise UnsupportedContent("Anthropic document input requires PDF or UTF-8 plain text")
        encoded.append(value)
    return encoded


def garak_content(message: Any, roots: list[Path]) -> Any:
    text = message.text
    if text is not None and not isinstance(text, str):
        raise UnsupportedContent("Garak text content is not text")
    data = getattr(message, "_data", None)
    path = getattr(message, "data_path", None)
    declared = getattr(message, "data_type", None)
    name = None
    if path is not None:
        candidate = Path(path).absolute()
        permitted = [root.resolve(strict=True) for root in roots]
        resolved = candidate.resolve(strict=True)
        if not any(resolved == root or resolved.is_relative_to(root) for root in permitted):
            raise UnsupportedContent("Attachment file is outside mounted/generated asset roots")
        with secure_open(candidate) as stream:
            before = os.fstat(stream.fileno())
            loaded = stream.read()
            after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            raise UnsupportedContent("Garak attachment changed during reading")
        if data is not None and data != loaded:
            raise UnsupportedContent(
                "Garak attachment cached checksum/content differs from its mounted file"
            )
        data = loaded
        name = candidate.name
        guessed, compressed = mimetypes.guess_type(name)
        if compressed is not None:
            raise UnsupportedContent("Compressed attachments require an explicit decoding adapter")
        if declared is None:
            declared = (guessed, None)
    if data is None:
        return text or ""
    if not isinstance(data, bytes):
        raise UnsupportedContent("Garak attachment is not a byte buffer")
    if not isinstance(declared, (tuple, list)) or len(declared) != 2 or declared[1] is not None:
        raise UnsupportedContent("Garak attachment lacks an uncompressed MIME declaration")
    mime = media_type(declared[0])
    checksum = hashlib.sha256(data).hexdigest()
    expected = getattr(message, "data_checksum", None)
    if expected is not None and expected != checksum:
        raise UnsupportedContent("Garak attachment checksum changed")
    message._data = data
    message.data_type = (mime, None)
    message.data_checksum = checksum
    kind = mime.split("/")[0]
    if kind not in {"image", "audio", "video"}:
        if not document_type(mime):
            raise UnsupportedContent("Garak attachment media type has no normalized adapter")
        kind = "document"
    block: dict[str, Any] = {
        "type": kind,
        "media_type": mime,
        "data": base64.b64encode(data).decode("ascii"),
        "sha256": checksum,
    }
    if name is not None:
        block["filename"] = name
    result = [{"type": "text", "text": text}] if text is not None else []
    result.append(block)
    return result


def content_limitations(protocol: str, content: Any) -> list[str]:
    encode_content(protocol, content)
    if not isinstance(content, list):
        return []
    limitations = []
    for block in content:
        kind = block.get("type")
        mime = block.get("media_type")
        if kind == "image":
            limitations.append("Provider image resizing and native vision resolution apply")
            if mime == "image/gif":
                limitations.append("Provider animation handling may inspect only the first frame")
        if kind == "document":
            limitations.append("Provider document extraction/context/page capacity applies")
            if protocol == "openai-responses" and mime != "application/pdf":
                limitations.append(
                    "Responses non-PDF extraction includes text, not embedded visuals"
                )
    return list(dict.fromkeys(limitations))


def garak_generation(value: dict[str, Any]) -> Any:
    import importlib

    message = importlib.import_module("garak.attempt").Message(text=value.get("text", ""))
    media = value.get("media", [])
    if not isinstance(media, list) or len(media) > 1:
        raise UnsupportedContent("Native Garak Message supports one generated attachment")
    if media:
        block = media[0]
        if block.get("complete") is False:
            raise UnsupportedContent("Native Garak cannot assess an incomplete media output")
        mime, _, data = attachment(block)
        message.data_type = (mime, None)
        message.data = data
        message.notes = {
            "ablitbench_media": {key: item for key, item in block.items() if key != "data"}
        }
    return message
