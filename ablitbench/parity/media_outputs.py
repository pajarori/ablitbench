from __future__ import annotations

import base64
import binascii
import hashlib
import re
import tempfile
from typing import Any
from urllib.parse import urlsplit

from ablitbench.parity.multimodal import UnsupportedContent, media_type

AUDIO_MIMES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "pcm16": "audio/pcm",
}
IMAGE_MIMES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}
VIDEO_MIMES = {"mp4": "video/mp4", "webm": "video/webm"}
MEDIA_KINDS = {
    "image",
    "audio",
    "video",
    "document",
    "image_url",
    "output_image",
    "output_audio",
    "output_video",
    "input_audio",
}


def bytes_media(
    kind: str,
    data: Any,
    mime: Any,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mime = media_type(mime)
    if not isinstance(data, str):
        raise UnsupportedContent("Output media data must be a base64 string")
    try:
        decoded = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as error:
        raise UnsupportedContent("Output media has malformed base64") from error
    if not decoded:
        raise UnsupportedContent("Output media data is empty")
    result = {
        "type": kind,
        "media_type": mime,
        "data": data,
        "sha256": hashlib.sha256(decoded).hexdigest(),
    }
    for key in ("id", "filename", "transcript", "format", "sample_rate", "channels", "complete"):
        if metadata is not None and metadata.get(key) is not None:
            result[key] = metadata[key]
    return result


def data_url(kind: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        raise UnsupportedContent("Output media URL is not a string")
    if value.startswith("data:"):
        match = re.fullmatch(r"data:([^;,]+);base64,(.*)", value, re.DOTALL)
        if match is None:
            raise UnsupportedContent("Output media data URL must explicitly use base64")
        return bytes_media(kind, match.group(2), match.group(1))
    parsed = urlsplit(value)
    if parsed.scheme in {"https", "http"} and parsed.hostname:
        raise UnsupportedContent(
            "External output media locator requires authorized materialization; "
            "bytes were not fetched"
        )
    raise UnsupportedContent("Output media locator is not an inline base64 resource")


def audio_media(value: dict[str, Any], configured_format: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UnsupportedContent("Audio output object is malformed")
    declared_mime = value.get("media_type")
    audio_format = value.get("format") or value.get("output_format") or configured_format
    if declared_mime is not None:
        mime = media_type(declared_mime)
        if not mime.startswith("audio/"):
            raise UnsupportedContent("Declared audio MIME type is not audio")
    elif isinstance(audio_format, str) and audio_format in AUDIO_MIMES:
        mime = AUDIO_MIMES[audio_format]
    else:
        raise UnsupportedContent(
            "Audio format is absent; an explicit request output_audio_format is required"
        )
    metadata = dict(value)
    if audio_format is not None:
        metadata["format"] = audio_format
    return bytes_media("audio", value.get("data"), mime, metadata)


def image_media(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UnsupportedContent("Image output object is malformed")
    if value.get("url") is not None:
        return data_url("image", value["url"])
    data = value.get("data") or value.get("result") or value.get("b64_json")
    mime = value.get("media_type")
    if mime is None:
        image_format = value.get("output_format") or value.get("format")
        mime = IMAGE_MIMES.get(image_format) if isinstance(image_format, str) else None
    if mime is None and isinstance(data, str):
        try:
            prefix = base64.b64decode(data[:32], validate=True)
        except (ValueError, binascii.Error) as error:
            raise UnsupportedContent(
                "Image format absent and encoded image header is invalid"
            ) from error
        if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif prefix.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif prefix.startswith((b"GIF87a", b"GIF89a")):
            mime = "image/gif"
        elif prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP":
            mime = "image/webp"
    if mime is None:
        raise UnsupportedContent("Output image format is undeclared and not identifiable")
    if not media_type(mime).startswith("image/"):
        raise UnsupportedContent("Output image MIME type is not an image")
    return bytes_media("image", data, mime, value)


def output_block(value: dict[str, Any], audio_format: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UnsupportedContent("Output media block is malformed")
    kind = value.get("type")
    if kind in {"image_url", "output_image"}:
        image = value.get("image_url") or value.get("image") or value
        return image_media({"url": image} if isinstance(image, str) else image)
    if kind == "image":
        source = value.get("source") or value
        if not isinstance(source, dict):
            raise UnsupportedContent("Image output source is malformed")
        if source.get("type") == "url":
            return data_url("image", source.get("url"))
        return image_media(source)
    if kind in {"audio", "output_audio", "input_audio"}:
        source = value.get("audio") or value.get("input_audio") or value.get("source") or value
        if isinstance(source, str):
            return audio_media({"data": source}, audio_format)
        if not isinstance(source, dict):
            raise UnsupportedContent("Audio output source is malformed")
        if source.get("url") is not None:
            return data_url("audio", source["url"])
        return audio_media(source, audio_format)
    if kind in {"video", "output_video"}:
        source = value.get("video") or value.get("source") or value
        if isinstance(source, str):
            return data_url("video", source)
        if not isinstance(source, dict):
            raise UnsupportedContent("Video output source is malformed")
        if source.get("url") is not None:
            return data_url("video", source["url"])
        video_format = source.get("format")
        mime = source.get("media_type") or (
            VIDEO_MIMES.get(video_format) if isinstance(video_format, str) else None
        )
        if mime is None or not media_type(mime).startswith("video/"):
            raise UnsupportedContent("Output video requires an explicit video MIME/format")
        return bytes_media("video", source.get("data"), mime, source)
    if kind == "document":
        source = value.get("source") or value
        if not isinstance(source, dict):
            raise UnsupportedContent("Document output source is malformed")
        return bytes_media("document", source.get("data"), source.get("media_type"), source)
    raise UnsupportedContent("Unsupported output media content block")


class AudioStream:
    def __init__(self, configured_format: str | None = None):
        self.data = tempfile.TemporaryFile(mode="w+b")
        self.pending = ""
        self.transcript = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self.size = 0
        self.configured_format = configured_format
        self.metadata: dict[str, Any] = {}

    def append(self, value: dict[str, Any]) -> None:
        for key in ("id", "format", "media_type", "sample_rate", "channels"):
            if value.get(key) is not None:
                if key in self.metadata and value[key] != self.metadata[key]:
                    raise UnsupportedContent("Streamed audio identity/format changed")
                self.metadata[key] = value[key]
        if value.get("data") is not None:
            if not isinstance(value["data"], str):
                raise UnsupportedContent("Streamed audio data fragment is not a string")
            self.pending += value["data"]
            while len(self.pending) >= 4:
                padding = self.pending.find("=")
                length = ((padding + 4) // 4) * 4 if padding >= 0 else len(self.pending) // 4 * 4
                if length > len(self.pending):
                    break
                fragment, self.pending = self.pending[:length], self.pending[length:]
                try:
                    decoded = base64.b64decode(fragment, validate=True)
                except (ValueError, binascii.Error) as error:
                    raise UnsupportedContent("Malformed streamed audio base64") from error
                self.data.write(decoded)
                self.size += len(decoded)
        if value.get("transcript") is not None:
            if not isinstance(value["transcript"], str):
                raise UnsupportedContent("Streamed audio transcript fragment is not text")
            self.transcript.write(value["transcript"])

    def finish(self) -> dict[str, Any] | None:
        if self.pending:
            raise UnsupportedContent("Streamed audio ended inside an encoded byte sequence")
        if not self.size:
            self.transcript.seek(0)
            if self.transcript.read():
                raise UnsupportedContent("Streamed audio transcript has no audio payload")
            return None
        self.data.seek(0)
        self.transcript.seek(0)
        value = dict(self.metadata)
        value["data"] = base64.b64encode(self.data.read()).decode("ascii")
        transcript = self.transcript.read()
        if transcript:
            value["transcript"] = transcript
        return audio_media(value, self.configured_format)

    def close(self) -> None:
        self.data.close()
        self.transcript.close()
