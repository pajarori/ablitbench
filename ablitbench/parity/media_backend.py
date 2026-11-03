from __future__ import annotations

import base64
import importlib
import inspect
import io
import math
from pathlib import Path
from typing import Any

from ablitbench.parity.media_outputs import bytes_media
from ablitbench.parity.multimodal import UnsupportedContent, attachment


def model_class(config: Any) -> Any:
    transformers = importlib.import_module("transformers")
    registry = importlib.import_module("transformers.models.auto.modeling_auto")
    name = getattr(config, "model_type", None)
    multimodal = getattr(registry, "MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES", {})
    image_text = getattr(registry, "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", {})
    if name in multimodal and hasattr(transformers, "AutoModelForMultimodalLM"):
        return transformers.AutoModelForMultimodalLM
    if name in image_text:
        return transformers.AutoModelForImageTextToText
    return None


def load_processor(root: Path, model: Any) -> MediaBackend:
    from ablitbench.worker import isolation

    isolation()
    transformers = importlib.import_module("transformers")
    processor = transformers.AutoProcessor.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    if not callable(getattr(processor, "apply_chat_template", None)):
        raise UnsupportedContent("Safe local processor has no multimodal chat template contract")
    if not callable(processor):
        raise UnsupportedContent("Safe local processor cannot construct multimodal model inputs")
    return MediaBackend(processor, model)


def decode_image(data: bytes) -> tuple[Any, dict[str, Any]]:
    image_module: Any = importlib.import_module("PIL.Image")
    image_ops = importlib.import_module("PIL.ImageOps")
    old_limit = image_module.MAX_IMAGE_PIXELS
    image_module.MAX_IMAGE_PIXELS = None
    try:
        with image_module.open(io.BytesIO(data)) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise UnsupportedContent(
                    "Animated/multipage image requires an explicit temporal/page adapter"
                )
            image.load()
            normalized = image_ops.exif_transpose(image).copy()
            information = {
                "kind": "media-decoding",
                "modality": "image",
                "format": image.format,
                "width": normalized.width,
                "height": normalized.height,
                "source_mode": image.mode,
                "capacity": "Sandbox physical memory and native processor image capacity",
                "processing": (
                    "Complete raster decoded; native model resizing/cropping remains explicit"
                ),
            }
    finally:
        image_module.MAX_IMAGE_PIXELS = old_limit
    return normalized, information


def decode_audio(
    data: bytes, target_rate: Any, metadata: dict[str, Any] | None = None
) -> tuple[Any, dict[str, Any]]:
    if type(target_rate) is not int or target_rate <= 0:
        raise UnsupportedContent("Audio processor does not declare a valid sampling rate")
    soundfile = importlib.import_module("soundfile")
    numpy = importlib.import_module("numpy")
    raw_options = {}
    if metadata is not None and metadata.get("media_type") == "audio/pcm":
        rate, channels = metadata.get("sample_rate"), metadata.get("channels")
        if (
            metadata.get("format") != "pcm16"
            or type(rate) is not int
            or rate <= 0
            or type(channels) is not int
            or channels <= 0
        ):
            raise UnsupportedContent("Raw PCM16 input requires declared sample rate and channels")
        if len(data) % (2 * channels):
            raise UnsupportedContent("Raw PCM16 input ends inside a sample frame")
        raw_options = {
            "format": "RAW",
            "subtype": "PCM_16",
            "endian": "LITTLE",
            "samplerate": rate,
            "channels": channels,
        }
    audio, source_rate = soundfile.read(
        io.BytesIO(data), dtype="float32", always_2d=True, **raw_options
    )
    if audio.shape[0] == 0 or not numpy.isfinite(audio).all():
        raise UnsupportedContent("Audio has no finite samples")
    channels = audio.shape[1]
    if channels != 1:
        raise UnsupportedContent(
            "Multichannel audio requires an explicit channel policy; no silent mono downmix"
        )
    audio = audio[:, 0]
    original_samples = len(audio)
    if source_rate != target_rate:
        signal = importlib.import_module("scipy.signal")
        divisor = math.gcd(source_rate, target_rate)
        audio = signal.resample_poly(audio, target_rate // divisor, source_rate // divisor)
    return audio, {
        "kind": "media-decoding",
        "modality": "audio",
        "source_rate": source_rate,
        "processor_rate": target_rate,
        "source_samples": original_samples,
        "processed_samples": len(audio),
        "channels": channels,
        "resampled": source_rate != target_rate,
        "complete_source_samples": True,
    }


def decode_video(data: bytes) -> tuple[list[Any], Any, dict[str, Any]]:
    av = importlib.import_module("av")
    video_type = importlib.import_module("transformers.video_utils").VideoMetadata
    with av.open(io.BytesIO(data), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise UnsupportedContent(
                "Video requires exactly one explicitly represented visual stream"
            )
        if list(container.streams.audio):
            raise UnsupportedContent(
                "Video contains audio; separate joint audiovisual processor support is required"
            )
        stream = streams[0]
        rate = stream.average_rate or stream.base_rate
        if rate is None or float(rate) <= 0:
            raise UnsupportedContent("Video has no declared frame rate; no synthetic FPS fallback")
        fps = float(rate)
        tolerance = (
            float(stream.time_base) if stream.time_base is not None else math.ulp(1 / fps) * 2
        )
        frames = []
        previous = None
        for frame in container.decode(stream):
            timestamp = float(frame.time) if frame.time is not None else None
            if timestamp is None:
                raise UnsupportedContent("Video frame timestamp is missing")
            if previous is not None and abs((timestamp - previous) - 1 / fps) > tolerance:
                raise UnsupportedContent(
                    "Variable frame timing is not representable by the native uniform-FPS contract"
                )
            previous = timestamp
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise UnsupportedContent("Video contains no decoded frames")
    metadata = video_type(
        total_num_frames=len(frames),
        fps=fps,
        width=frames[0].shape[1],
        height=frames[0].shape[0],
        duration=len(frames) / fps,
        video_backend="pyav-all-frames",
        frames_indices=list(range(len(frames))),
    )
    return (
        frames,
        metadata,
        {
            "kind": "media-decoding",
            "modality": "video",
            "decoded_frames": len(frames),
            "submitted_frames": len(frames),
            "fps": fps,
            "sampling": False,
            "audio_tracks": 0,
            "complete_source_frames": True,
        },
    )


class MediaBackend:
    def __init__(self, processor: Any, model: Any):
        self.processor = processor
        self.model = model
        self.tokenizer = getattr(processor, "tokenizer", None)
        self.observations: list[dict[str, Any]] = []

    @property
    def is_encoder_decoder(self) -> bool:
        return bool(
            getattr(self.model.config, "is_encoder_decoder", False)
            or getattr(getattr(self.model.config, "text_config", None), "is_encoder_decoder", False)
        )

    @staticmethod
    def position_limit(configs: list[Any]) -> int | None:
        limits = [
            value
            for config in configs
            if config is not None
            for key in ("max_position_embeddings", "n_positions")
            if type(value := getattr(config, key, None)) is int and value > 0
        ]
        return min(limits) if limits else None

    @property
    def input_context_limit(self) -> int | None:
        config = self.model.config
        if self.is_encoder_decoder:
            encoder = getattr(config, "encoder", None)
            text = getattr(config, "text_config", None)
            return self.position_limit([encoder] if encoder is not None else [text, config])
        return self.position_limit(
            [
                config,
                getattr(config, "text_config", None),
                getattr(config, "language_config", None),
                getattr(config, "thinker_config", None),
                getattr(getattr(config, "thinker_config", None), "text_config", None),
            ]
        )

    @property
    def output_context_limit(self) -> int | None:
        config = self.model.config
        if self.is_encoder_decoder:
            return self.position_limit(
                [
                    getattr(config, "decoder", None),
                    getattr(config, "text_config", None),
                    config,
                ]
            )
        return self.input_context_limit

    @property
    def context_limit(self) -> int | None:
        return None if self.is_encoder_decoder else self.input_context_limit

    def prepare(self, messages: list[dict[str, Any]], tools: Any = None) -> Any:
        from ablitbench.worker import isolation

        isolation()
        conversation = []
        images, audio, videos, video_metadata = [], [], [], []
        self.observations = []
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        target_rate = getattr(feature_extractor, "sampling_rate", None)
        for message in messages:
            role = message.get("role")
            if role not in {"system", "user", "assistant", "developer", "tool"}:
                raise UnsupportedContent("Unsupported local multimodal conversation role")
            content = message.get("content")
            if content is None:
                content = ""
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
            if not isinstance(blocks, list):
                raise UnsupportedContent("Local media input requires normalized content blocks")
            normalized = []
            for block in blocks:
                if not isinstance(block, dict):
                    raise UnsupportedContent("Local media content block is not a mapping")
                kind = block.get("type")
                if kind == "text":
                    if set(block) != {"type", "text"} or not isinstance(block.get("text"), str):
                        raise UnsupportedContent("Local text block has no text")
                    normalized.append({"type": "text", "text": block["text"]})
                    continue
                if role != "user":
                    raise UnsupportedContent("Media is supported only in local user messages")
                if block.get("complete") is False:
                    raise UnsupportedContent(
                        "Incomplete output media cannot become a local model input"
                    )
                mime, _, data = attachment(block)
                if kind == "image":
                    if getattr(self.processor, "image_processor", None) is None:
                        raise UnsupportedContent("Selected model processor has no image capability")
                    value, information = decode_image(data)
                    images.append(value)
                elif kind == "audio":
                    if feature_extractor is None:
                        raise UnsupportedContent("Selected model processor has no audio capability")
                    value, information = decode_audio(data, target_rate, block)
                    capacity = getattr(feature_extractor, "n_samples", None)
                    if type(capacity) is int and len(value) > capacity:
                        raise UnsupportedContent(
                            "Complete audio exceeds native feature-extractor sample capacity: "
                            + str(capacity)
                        )
                    audio.append(value)
                elif kind == "video":
                    if getattr(self.processor, "video_processor", None) is None:
                        raise UnsupportedContent(
                            "Selected model processor has no complete-video capability"
                        )
                    value, metadata, information = decode_video(data)
                    videos.append(value)
                    video_metadata.append(metadata)
                elif kind == "document" and mime.startswith("text/"):
                    try:
                        text = data.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise UnsupportedContent("Local text document is not UTF-8") from error
                    normalized.append({"type": "text", "text": text})
                    self.observations.append(
                        {
                            "kind": "media-decoding",
                            "modality": "document",
                            "processing": "Complete UTF-8 text document",
                        }
                    )
                    continue
                else:
                    raise UnsupportedContent(
                        "Local model has no safe native adapter for this normalized media type"
                    )
                normalized.append({"type": kind})
                information["declared_media_type"] = mime
                self.observations.append(information)
            normalized_message = dict(message)
            normalized_message["content"] = normalized
            conversation.append(normalized_message)
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            prompt = self.processor.apply_chat_template(conversation, **kwargs)
        except (ValueError, TypeError) as error:
            raise UnsupportedContent(
                "Native multimodal chat template cannot encode this conversation"
            ) from error
        processing: dict[str, Any] = {
            "text": prompt,
            "return_tensors": "pt",
            "text_kwargs": {"truncation": False},
        }
        if images:
            processing["images"] = images
        if audio:
            processing["audio"] = audio
            capacity = getattr(feature_extractor, "n_samples", None)
            audio_kwargs = {
                "sampling_rate": target_rate,
                "truncation": False,
                "padding": "max_length" if type(capacity) is int else "longest",
                "return_attention_mask": True,
            }
            if type(capacity) is int:
                audio_kwargs["max_length"] = capacity
            processing["audio_kwargs"] = audio_kwargs
        if videos:
            processing["videos"] = videos
            processing["videos_kwargs"] = {
                "do_sample_frames": False,
                "video_metadata": video_metadata,
            }
        try:
            encoded = self.processor(**processing)
        except (ValueError, TypeError) as error:
            raise UnsupportedContent(
                "Native multimodal processor cannot encode complete inputs"
            ) from error
        ids = encoded.get("input_ids")
        if (
            ids is not None
            and self.input_context_limit is not None
            and ids.shape[-1] > self.input_context_limit
        ):
            raise UnsupportedContent(
                "Expanded multimodal input exceeds the native text context capacity"
            )
        torch = importlib.import_module("torch")
        for value in encoded.values():
            if (
                isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and not torch.isfinite(value).all()
            ):
                raise UnsupportedContent("Multimodal processor produced nonfinite model inputs")
        self.observations.append(
            {
                "kind": "media-processing",
                "source_images": len(images),
                "source_audio": len(audio),
                "source_videos": len(videos),
                "input_tokens": ids.shape[-1] if ids is not None else None,
                "context_limit": self.context_limit,
                "encoder_decoder": self.is_encoder_decoder,
                "input_context_limit": self.input_context_limit,
                "output_context_limit": self.output_context_limit,
                "text_truncation": False,
                "native_image_preprocessing": "Native processor resize/crop/grid settings apply",
                "video_frame_sampling": False,
            }
        )
        return encoded

    def generation_kwargs(
        self,
        output_modalities: list[str] | None = None,
        output_audio_format: str | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        requested = output_modalities or ["text"]
        audio = "audio" in requested
        family = getattr(self.model.config, "model_type", None)
        if not set(requested) <= {"text", "audio"} or not requested:
            raise UnsupportedContent("Native image/video output has no typed decoder")
        if audio and family not in {"qwen2_5_omni", "qwen3_omni_moe"}:
            raise UnsupportedContent("Selected native model has no verified audio output contract")
        if audio and output_audio_format not in {"wav", "flac", "pcm16"}:
            raise UnsupportedContent("Native audio export requires explicit wav, flac or pcm16")
        parameters = inspect.signature(self.model.generate).parameters
        kwargs: dict[str, Any] = {}
        if "thinker_max_new_tokens" in parameters:
            if type(max_output_tokens) is not int or max_output_tokens <= 0:
                raise UnsupportedContent(
                    "Native split-decoder model requires an explicit per-decoder output budget"
                )
            kwargs["thinker_max_new_tokens"] = max_output_tokens
            if audio:
                talker = getattr(self.model.config, "talker_config", None)
                capacity = self.position_limit([talker, getattr(talker, "text_config", None)])
                if capacity is not None and max_output_tokens > capacity:
                    raise UnsupportedContent("Requested audio tokens exceed native talker context")
                kwargs["talker_max_new_tokens"] = max_output_tokens
        if (
            "generation_mode" in parameters
            or getattr(self.model.config, "model_type", None) == "qwen2_5_omni"
        ):
            kwargs["generation_mode"] = "audio" if audio else "text"
        elif "return_audio" in parameters:
            kwargs["return_audio"] = audio
        if kwargs:
            self.observations.append(
                {
                    "kind": "media-output",
                    "requested": requested,
                    "decoder_options": kwargs,
                    "audio_capacity": "Explicit budget applies separately to thinker and talker",
                }
            )
        return kwargs

    def normalize_generated(
        self,
        output: Any,
        prompt_tokens: int,
        output_modalities: list[str] | None = None,
        output_audio_format: str | None = None,
    ) -> dict[str, Any]:
        requested = output_modalities or ["text"]
        torch = importlib.import_module("torch")
        waveform = None
        if "audio" in requested:
            if not isinstance(output, tuple) or len(output) != 2:
                raise UnsupportedContent(
                    "Native audio output is not the verified text/waveform tuple"
                )
            output, waveform = output
        sequences = output.sequences if hasattr(output, "sequences") else output
        if (
            not isinstance(sequences, torch.Tensor)
            or sequences.ndim != 2
            or sequences.shape[0] != 1
        ):
            raise UnsupportedContent("Native generated output is not a single text sequence")
        media = []
        if waveform is not None:
            media.append(self.encode_audio(waveform, output_audio_format))
        elif set(requested) != {"text"}:
            raise UnsupportedContent("Native generated media has no typed decoder for this model")
        tokens = sequences[0] if self.is_encoder_decoder else sequences[0][prompt_tokens:]
        decoder = self.tokenizer if self.tokenizer is not None else self.processor
        text = self.decode_generated(sequences[0], prompt_tokens)
        if "text" not in requested:
            self.observations.append(
                {
                    "kind": "native-output-text",
                    "text": text,
                    "scope": "Co-generated native text; audio transcript alignment unverified",
                }
            )
        return {
            "text": text if "text" in requested else "",
            "raw_text": str(decoder.decode(tokens, skip_special_tokens=False)),
            "media": media,
            "output_tokens": len(tokens),
        }

    def encode_audio(self, waveform: Any, output_format: str | None) -> dict[str, Any]:
        torch = importlib.import_module("torch")
        numpy = importlib.import_module("numpy")
        if not isinstance(waveform, torch.Tensor):
            raise UnsupportedContent("Native audio output must be a waveform tensor")
        if waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform = waveform[0]
        if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise UnsupportedContent("Native audio output must be one finite mono waveform")
        family = getattr(self.model.config, "model_type", None)
        if family not in {"qwen2_5_omni", "qwen3_omni_moe"}:
            raise UnsupportedContent("Native audio sampling rate has no verified family contract")
        rate = getattr(self.model.config, "output_sampling_rate", 24000)
        if type(rate) is not int or rate <= 0:
            raise UnsupportedContent("Native audio sampling rate is invalid")
        samples = waveform.detach().cpu().float().numpy()
        buffer = io.BytesIO()
        soundfile = importlib.import_module("soundfile")
        if output_format == "wav":
            soundfile.write(buffer, samples, rate, format="WAV", subtype="FLOAT")
            mime = "audio/wav"
            quantization = "float32 waveform preserved"
        elif output_format in {"flac", "pcm16"}:
            if numpy.max(numpy.abs(samples)) > 1:
                raise UnsupportedContent("Integer audio export would clip native waveform")
            if output_format == "flac":
                soundfile.write(buffer, samples, rate, format="FLAC", subtype="PCM_24")
                mime = "audio/flac"
                quantization = "Explicit signed 24-bit quantization"
            else:
                soundfile.write(
                    buffer, samples, rate, format="RAW", subtype="PCM_16", endian="LITTLE"
                )
                mime = "audio/pcm"
                quantization = "Explicit signed 16-bit little-endian quantization"
        else:
            raise UnsupportedContent("Native audio export requires explicit wav, flac or pcm16")
        self.observations.append(
            {
                "kind": "media-output-encoding",
                "sampling_rate": rate,
                "samples": len(samples),
                "channels": 1,
                "encoding": quantization,
                "rate_contract": "Declared output rate or documented native Omni 24000Hz",
            }
        )
        return bytes_media(
            "audio",
            base64.b64encode(buffer.getvalue()).decode(),
            mime,
            {"format": output_format, "sample_rate": rate, "channels": 1, "complete": True},
        )

    def decode_generated(self, sequence: Any, prompt_tokens: int) -> str:
        if self.is_encoder_decoder:
            generated = sequence
        else:
            generated = sequence[prompt_tokens:]
        if not callable(getattr(self.processor, "decode", None)):
            raise UnsupportedContent("Native processor has no text generation decoder")
        return str(self.processor.decode(generated, skip_special_tokens=True))
