"""Unit tests for multimodal content types."""
from __future__ import annotations

import base64
import pytest

from lauren_ai._transport._multimodal import (
    ImageContent,
    AudioContent,
    DocumentContent,
    ContentPart,
    UnsupportedContentError,
)
from lauren_ai._transport import Message


class TestImageContent:
    def test_from_bytes(self):
        data = b"\x89PNG\r\n\x1a\n"
        img = ImageContent.from_bytes(data, mime_type="image/png")
        assert img.data == data
        assert img.mime_type == "image/png"

    def test_from_base64(self):
        data = b"fake image data"
        b64 = base64.b64encode(data).decode()
        img = ImageContent.from_base64(b64, mime_type="image/jpeg")
        assert img.data == data

    def test_from_url(self):
        img = ImageContent.from_url("https://example.com/photo.jpg")
        assert img.url == "https://example.com/photo.jpg"
        assert img.data is None

    def test_base64_data_property(self):
        data = b"hello"
        img = ImageContent.from_bytes(data, "image/png")
        assert img.base64_data == base64.b64encode(data).decode()

    def test_to_anthropic_block_url(self):
        img = ImageContent.from_url("https://example.com/x.png")
        block = img.to_anthropic_block()
        assert block["type"] == "image"
        assert block["source"]["type"] == "url"

    def test_to_anthropic_block_base64(self):
        img = ImageContent.from_bytes(b"data", "image/png")
        block = img.to_anthropic_block()
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"

    def test_to_openai_block_url(self):
        img = ImageContent.from_url("https://example.com/x.png")
        block = img.to_openai_block()
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "https://example.com/x.png"

    def test_to_openai_block_bytes_is_data_uri(self):
        img = ImageContent.from_bytes(b"data", "image/png")
        block = img.to_openai_block()
        assert block["image_url"]["url"].startswith("data:image/png;base64,")


class TestAudioContent:
    def test_from_bytes(self):
        audio = AudioContent.from_bytes(b"audio data", "audio/mpeg")
        assert audio.data == b"audio data"
        assert audio.mime_type == "audio/mpeg"

    def test_base64_data(self):
        audio = AudioContent.from_bytes(b"hello", "audio/wav")
        assert audio.base64_data == base64.b64encode(b"hello").decode()

    def test_to_openai_block(self):
        audio = AudioContent.from_bytes(b"audio", "audio/mpeg")
        block = audio.to_openai_block()
        assert block["type"] == "input_audio"
        assert "data" in block["input_audio"]


class TestDocumentContent:
    def test_from_bytes(self):
        doc = DocumentContent.from_bytes(b"pdf data")
        assert doc.data == b"pdf data"

    def test_from_url(self):
        doc = DocumentContent.from_url("https://example.com/doc.pdf")
        assert doc.url == "https://example.com/doc.pdf"

    def test_to_anthropic_block_url(self):
        doc = DocumentContent.from_url("https://example.com/doc.pdf")
        block = doc.to_anthropic_block()
        assert block["type"] == "document"
        assert block["source"]["url"] == "https://example.com/doc.pdf"

    def test_to_anthropic_block_bytes(self):
        doc = DocumentContent.from_bytes(b"pdf", "application/pdf")
        block = doc.to_anthropic_block()
        assert block["source"]["type"] == "base64"


class TestMessageFromMultimodal:
    def test_creates_message_with_list_content(self):
        img = ImageContent.from_bytes(b"x", "image/png")
        msg = Message.from_multimodal("user", ["Describe:", img])
        assert isinstance(msg.content, list)
        assert msg.content[0] == "Describe:"
        assert isinstance(msg.content[1], ImageContent)

    def test_role_set_correctly(self):
        msg = Message.from_multimodal("assistant", ["response"])
        assert msg.role == "assistant"
