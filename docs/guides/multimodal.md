# Multimodal Inputs

`lauren-ai` provides three typed content classes for sending non-text data to
vision- and audio-capable models:

| Class              | Providers supported      | Data sources            |
|--------------------|--------------------------|-------------------------|
| `ImageContent`     | Anthropic, OpenAI        | bytes, URL, file, base64|
| `AudioContent`     | OpenAI only              | bytes, file             |
| `DocumentContent`  | Anthropic only           | bytes, URL, file        |

Use `Message.from_multimodal()` to assemble a message that contains a mix of
plain text strings and any of the above objects.

## Images

```python
from lauren_ai._transport._multimodal import ImageContent
from lauren_ai._transport import Message

# From a local file
img = ImageContent.from_file("/tmp/chart.png")

# From a public URL
img = ImageContent.from_url("https://example.com/photo.jpg")

# From raw bytes
with open("/tmp/photo.jpg", "rb") as f:
    img = ImageContent.from_bytes(f.read(), mime_type="image/jpeg")

# From a base64 string (e.g. received from another API)
img = ImageContent.from_base64(b64_string, mime_type="image/png")
```

Build a multimodal user message:

```python
msg = Message.from_multimodal("user", [
    "Please describe what you see in this chart:",
    img,
])
result = await llm.complete([msg])
print(result.content)
```

### Serialization formats

`ImageContent` can produce the wire format expected by each provider:

```python
img = ImageContent.from_bytes(b"...", "image/png")

# Anthropic
block = img.to_anthropic_block()
# {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}

# OpenAI
block = img.to_openai_block()
# {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

# URL-based images produce a cleaner wire format
url_img = ImageContent.from_url("https://example.com/photo.jpg")
url_img.to_openai_block()
# {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
```

## Audio

Audio input is supported by OpenAI's `gpt-4o-audio-preview` model.  Attempting
to pass `AudioContent` to an Anthropic or Ollama transport raises
`UnsupportedContentError`.

```python
from lauren_ai._transport._multimodal import AudioContent

audio = AudioContent.from_file("/tmp/recording.mp3")
# or
audio = AudioContent.from_bytes(audio_bytes, mime_type="audio/mpeg")

msg = Message.from_multimodal("user", [
    "Transcribe this recording:",
    audio,
])
```

The `to_openai_block()` method produces the `input_audio` format:

```python
audio.to_openai_block()
# {
#   "type": "input_audio",
#   "input_audio": {"data": "<base64>", "format": "mp3"}
# }
```

Supported MIME types and their format values:

| MIME type       | OpenAI format |
|-----------------|---------------|
| `audio/mpeg`    | `mp3`         |
| `audio/wav`     | `wav`         |
| `audio/mp4`     | `mp4`         |

## Documents (PDFs)

Anthropic's Claude models support native PDF document inputs.

```python
from lauren_ai._transport._multimodal import DocumentContent

# From a local PDF file
doc = DocumentContent.from_file("/tmp/report.pdf")

# From a URL (Anthropic fetches the URL directly)
doc = DocumentContent.from_url("https://example.com/report.pdf")

# From bytes
with open("/tmp/report.pdf", "rb") as f:
    doc = DocumentContent.from_bytes(f.read())

msg = Message.from_multimodal("user", [
    "Summarise the key findings in this report:",
    doc,
])
```

The Anthropic wire format:

```python
doc = DocumentContent.from_bytes(pdf_bytes)
doc.to_anthropic_block()
# {
#   "type": "document",
#   "source": {"type": "base64", "media_type": "application/pdf", "data": "..."}
# }

url_doc = DocumentContent.from_url("https://example.com/paper.pdf")
url_doc.to_anthropic_block()
# {"type": "document", "source": {"type": "url", "url": "https://example.com/paper.pdf"}}
```

## Assembling multimodal messages

`Message.from_multimodal(role, parts)` accepts any list of plain strings and
content objects in any order:

```python
from lauren_ai._transport import Message

msg = Message.from_multimodal("user", [
    "Compare these two charts and explain the difference:",
    ImageContent.from_url("https://example.com/chart_a.png"),
    ImageContent.from_file("/tmp/chart_b.png"),
    "Focus on the trend lines.",
])
```

The resulting `msg.content` is the same list of mixed objects.  Each transport
implementation is responsible for converting `ImageContent`, `AudioContent`,
and `DocumentContent` instances into the provider-specific wire format before
sending.

## Checking MIME types

`_guess_mime(suffix)` is exported from `_multimodal` and maps common file
extensions to MIME types.  It is used internally by the `from_file()` class
methods on all three content types:

```python
from lauren_ai._transport._multimodal import _guess_mime

_guess_mime(".png")   # "image/png"
_guess_mime(".mp3")   # "audio/mpeg"
_guess_mime(".pdf")   # "application/pdf"
_guess_mime(".xyz")   # "application/octet-stream"  (fallback)
```

## Provider support matrix

| Feature              | Anthropic | OpenAI | Ollama | LiteLLM |
|----------------------|-----------|--------|--------|---------|
| `ImageContent`       | Yes       | Yes    | Model-dependent | Via upstream |
| `AudioContent`       | No        | Yes    | No     | Via upstream |
| `DocumentContent`    | Yes       | No     | No     | Via upstream |

Transports that do not support a content type should raise
`UnsupportedContentError` when they encounter it during serialization.

## Error handling

```python
from lauren_ai._transport._multimodal import UnsupportedContentError

try:
    result = await llm.complete([msg_with_audio])
except UnsupportedContentError as exc:
    print(f"Provider does not support this content type: {exc}")
```
