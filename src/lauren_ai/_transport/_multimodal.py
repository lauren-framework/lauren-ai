from __future__ import annotations

"""Multimodal content types for vision, audio, and document inputs."""

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class UnsupportedContentError(Exception):
    """Raised when a provider does not support a content type."""


@dataclass
class ImageContent:
    """An image content block for multimodal messages.

    Usage::

        img = ImageContent.from_file("/tmp/chart.png")
        img = ImageContent.from_url("https://example.com/photo.jpg")
        img = ImageContent.from_bytes(b"...", mime_type="image/jpeg")
    """

    _data: bytes | None = field(default=None, repr=False)
    _url: str | None = field(default=None)
    mime_type: str = "image/png"

    @classmethod
    def from_file(cls, path: str | Path) -> ImageContent:
        """Load image bytes from *path* and detect MIME type from extension.

        :param path: Path to the image file.
        :type path: str | Path
        :return: A new :class:`ImageContent` with bytes loaded.
        :rtype: ImageContent
        """
        p = Path(path)
        data = p.read_bytes()
        mime = _guess_mime(p.suffix)
        return cls(_data=data, mime_type=mime)

    @classmethod
    def from_url(cls, url: str, mime_type: str = "image/jpeg") -> ImageContent:
        """Create an image referencing a remote URL.

        :param url: The URL of the image.
        :type url: str
        :param mime_type: MIME type hint.  Defaults to ``"image/jpeg"``.
        :type mime_type: str
        :return: A new :class:`ImageContent` referencing the URL.
        :rtype: ImageContent
        """
        return cls(_url=url, mime_type=mime_type)

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str) -> ImageContent:
        """Create an image from raw bytes.

        :param data: Raw image bytes.
        :type data: bytes
        :param mime_type: MIME type of the image, e.g. ``"image/png"``.
        :type mime_type: str
        :return: A new :class:`ImageContent` wrapping the bytes.
        :rtype: ImageContent
        """
        return cls(_data=data, mime_type=mime_type)

    @classmethod
    def from_base64(cls, b64: str, mime_type: str) -> ImageContent:
        """Create an image from a base64-encoded string.

        :param b64: Base64-encoded image data.
        :type b64: str
        :param mime_type: MIME type of the image.
        :type mime_type: str
        :return: A new :class:`ImageContent` decoded from *b64*.
        :rtype: ImageContent
        """
        return cls(_data=base64.b64decode(b64), mime_type=mime_type)

    @property
    def data(self) -> bytes | None:
        """Raw image bytes, or ``None`` if this is a URL-based image.

        :return: Image bytes or ``None``.
        :rtype: bytes | None
        """
        return self._data

    @property
    def url(self) -> str | None:
        """Remote URL, or ``None`` if this is a bytes-based image.

        :return: Image URL or ``None``.
        :rtype: str | None
        """
        return self._url

    @property
    def base64_data(self) -> str | None:
        """Base64-encoded bytes, or ``None`` if no byte data is stored.

        :return: Base64 string or ``None``.
        :rtype: str | None
        """
        if self._data is not None:
            return base64.b64encode(self._data).decode()
        return None

    def to_anthropic_block(self) -> dict[str, Any]:
        """Serialize to Anthropic API image-block format.

        :return: Dictionary suitable for the Anthropic messages API.
        :rtype: dict[str, Any]
        """
        if self._url:
            return {
                "type": "image",
                "source": {"type": "url", "url": self._url},
            }
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.mime_type,
                "data": self.base64_data,
            },
        }

    def to_openai_block(self) -> dict[str, Any]:
        """Serialize to OpenAI API image-block format.

        :return: Dictionary suitable for the OpenAI messages API.
        :rtype: dict[str, Any]
        """
        if self._url:
            return {"type": "image_url", "image_url": {"url": self._url}}
        data_uri = f"data:{self.mime_type};base64,{self.base64_data}"
        return {"type": "image_url", "image_url": {"url": data_uri}}


@dataclass
class AudioContent:
    """An audio content block.

    Note: Only supported by OpenAI (input_audio).
    Anthropic and Ollama raise :class:`UnsupportedContentError`.
    """

    _data: bytes
    mime_type: str  # "audio/mpeg" | "audio/wav" | "audio/mp4"

    @classmethod
    def from_file(cls, path: str | Path) -> AudioContent:
        """Load audio bytes from *path* and detect MIME type from extension.

        :param path: Path to the audio file.
        :type path: str | Path
        :return: A new :class:`AudioContent` with bytes loaded.
        :rtype: AudioContent
        """
        p = Path(path)
        data = p.read_bytes()
        mime = _guess_mime(p.suffix)
        return cls(_data=data, mime_type=mime)

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str) -> AudioContent:
        """Create audio from raw bytes.

        :param data: Raw audio bytes.
        :type data: bytes
        :param mime_type: MIME type, e.g. ``"audio/mpeg"`` or ``"audio/wav"``.
        :type mime_type: str
        :return: A new :class:`AudioContent` wrapping the bytes.
        :rtype: AudioContent
        """
        return cls(_data=data, mime_type=mime_type)

    @property
    def data(self) -> bytes:
        """Raw audio bytes.

        :return: Audio bytes.
        :rtype: bytes
        """
        return self._data

    @property
    def base64_data(self) -> str:
        """Base64-encoded audio bytes.

        :return: Base64 string.
        :rtype: str
        """
        return base64.b64encode(self._data).decode()

    def to_openai_block(self) -> dict[str, Any]:
        """Serialize to OpenAI API input_audio block format.

        :return: Dictionary suitable for the OpenAI messages API.
        :rtype: dict[str, Any]
        """
        fmt = self.mime_type.split("/")[-1].replace("mpeg", "mp3")
        return {
            "type": "input_audio",
            "input_audio": {"data": self.base64_data, "format": fmt},
        }


@dataclass
class DocumentContent:
    """A document (PDF) content block.

    Anthropic supports native PDF documents.
    Other providers raise :class:`UnsupportedContentError`.
    """

    _data: bytes | None = field(default=None, repr=False)
    _url: str | None = field(default=None)
    mime_type: str = "application/pdf"

    @classmethod
    def from_file(cls, path: str | Path) -> DocumentContent:
        """Load document bytes from *path*.

        :param path: Path to the PDF file.
        :type path: str | Path
        :return: A new :class:`DocumentContent` with bytes loaded.
        :rtype: DocumentContent
        """
        return cls(_data=Path(path).read_bytes())

    @classmethod
    def from_url(cls, url: str) -> DocumentContent:
        """Create a document referencing a remote URL.

        :param url: The URL of the document.
        :type url: str
        :return: A new :class:`DocumentContent` referencing the URL.
        :rtype: DocumentContent
        """
        return cls(_url=url)

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str = "application/pdf") -> DocumentContent:
        """Create a document from raw bytes.

        :param data: Raw document bytes.
        :type data: bytes
        :param mime_type: MIME type.  Defaults to ``"application/pdf"``.
        :type mime_type: str
        :return: A new :class:`DocumentContent` wrapping the bytes.
        :rtype: DocumentContent
        """
        return cls(_data=data, mime_type=mime_type)

    @property
    def data(self) -> bytes | None:
        """Raw document bytes, or ``None`` if this is a URL-based document.

        :return: Document bytes or ``None``.
        :rtype: bytes | None
        """
        return self._data

    @property
    def url(self) -> str | None:
        """Remote URL, or ``None`` if this is a bytes-based document.

        :return: Document URL or ``None``.
        :rtype: str | None
        """
        return self._url

    @property
    def base64_data(self) -> str | None:
        """Base64-encoded bytes, or ``None`` if no byte data is stored.

        :return: Base64 string or ``None``.
        :rtype: str | None
        """
        if self._data is not None:
            return base64.b64encode(self._data).decode()
        return None

    def to_anthropic_block(self) -> dict[str, Any]:
        """Serialize to Anthropic API document-block format.

        :return: Dictionary suitable for the Anthropic messages API.
        :rtype: dict[str, Any]
        """
        if self._url:
            return {
                "type": "document",
                "source": {"type": "url", "url": self._url},
            }
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": self.mime_type,
                "data": self.base64_data,
            },
        }


# Type alias for content parts accepted by multimodal messages.
ContentPart = str | ImageContent | AudioContent | DocumentContent


def _guess_mime(suffix: str) -> str:
    """Map a file extension to a MIME type.

    Returns ``"application/octet-stream"`` for unknown extensions.

    :param suffix: File extension including the leading dot, e.g. ``".png"``.
    :type suffix: str
    :return: MIME type string.
    :rtype: str
    """
    suffix = suffix.lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".mp4": "audio/mp4",
        ".pdf": "application/pdf",
    }
    return mapping.get(suffix, "application/octet-stream")
