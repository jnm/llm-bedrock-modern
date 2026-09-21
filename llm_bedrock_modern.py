# Imports

import json
import mimetypes
import os
import re
from base64 import b64decode, b64encode
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import boto3
import click
import llm
from PIL import Image
from pydantic import Field, field_validator


@dataclass
class AttachmentData:
    mime_type: str
    content: Union[Path, bytes]
    name: Optional[str] = None

    @property
    def is_file_path(self) -> bool:
        return isinstance(self.content, Path)

    @property
    def is_image(self) -> bool:
        return self.mime_type.startswith("image/")

    @property
    def is_document(self) -> bool:
        return self.mime_type in MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT


# Constants

# See: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
BEDROCK_CONVERSE_IMAGE_FORMATS = ["png", "jpeg", "gif", "webp"]
MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
    "text/markdown": "md",
}

# See: https://docs.anthropic.com/en/docs/build-with-claude/vision
ANTHROPIC_MAX_IMAGE_LONG_SIZE = 1568

# Where the discovered model list is cached, relative to llm.user_dir().
CACHE_FILE = "bedrock-anthropic-profiles.json"

DEFAULT_MAX_TOKENS = 16000

# Used only when AWS is unreachable and nothing has been cached yet.
SEED_PROFILES = ["us.anthropic.claude-opus-5", "us.anthropic.claude-sonnet-5"]

# Models that accept thinking={"type": "adaptive"}. Left with thinking off,
# these tend to write their reasoning into the visible answer instead.
ADAPTIVE_THINKING = re.compile(
    r"claude-(fable|mythos)-\d|claude-opus-(4-[678]|5)|claude-sonnet-(4-6|5)"
)

# Claude 3-era models reject anything above 4096 output tokens.
LEGACY_4K = re.compile(r"claude-(instant|v2|3-(sonnet|haiku|opus))")


# Much of this code is derived from https://github.com/tomviner/llm-claude


def cache_path():
    return llm.user_dir() / CACHE_FILE


def fetch_profiles():
    """Return sorted IDs of every ACTIVE Anthropic inference profile in the account."""
    client = boto3.client("bedrock")
    profiles = []
    kwargs = {"maxResults": 100}
    while True:
        response = client.list_inference_profiles(**kwargs)
        for summary in response.get("inferenceProfileSummaries", []):
            profile_id = summary["inferenceProfileId"]
            if "anthropic" in profile_id and summary.get("status") == "ACTIVE":
                profiles.append(profile_id)
        if not response.get("nextToken"):
            break
        kwargs["nextToken"] = response["nextToken"]
    return sorted(set(profiles))


def read_cache():
    path = cache_path()
    if path.exists():
        try:
            return json.loads(path.read_text())["profiles"]
        except (ValueError, KeyError):
            pass
    return None


def load_profiles(refresh=False):
    """The model list, from cache unless asked to refresh.

    Registration must never depend on a live AWS call: it runs on every single
    llm invocation, and a missing credential or a network hiccup would
    otherwise take down the whole CLI.
    """
    if not refresh:
        cached = read_cache()
        if cached is not None:
            return cached
    try:
        profiles = fetch_profiles()
    except Exception:
        return read_cache() or list(SEED_PROFILES)
    cache_path().write_text(json.dumps({"profiles": profiles}, indent=2))
    return profiles


def aliases_for(profile_id):
    """us.anthropic.claude-opus-5 -> ['bedrock-opus-5', 'bo5']"""
    region, _, rest = profile_id.partition(".")
    name = rest.removeprefix("anthropic.").removeprefix("claude-")
    # Drop Bedrock's version/date suffixes: opus-4-1-20250805-v1:0 -> opus-4-1
    parts = []
    for part in name.split("-"):
        if part.startswith("v") and part[1:].split(":")[0].isdigit():
            break
        if len(part) == 8 and part.isdigit():
            break
        parts.append(part)
    name = "-".join(parts)
    suffix = "" if region == "us" else f"-{region}"
    aliases = [f"bedrock-{name}{suffix}"]
    # Short form: family initial + dotted version, e.g. opus-4-6 -> bo4.6
    family, _, version = name.partition("-")
    if family and version:
        aliases.append(f"b{family[0]}{version.replace('-', '.')}{suffix}")
    return aliases


@llm.hookimpl
def register_models(register):
    seen = set()
    for profile_id in load_profiles():
        aliases = [alias for alias in aliases_for(profile_id) if alias not in seen]
        seen.update(aliases)
        register(BedrockClaude(profile_id, supports_attachments=True), aliases=tuple(aliases))


@llm.hookimpl
def register_commands(cli):
    @cli.command(name="bedrock-refresh")
    def bedrock_refresh():
        "Refresh the cached list of Anthropic models available on Bedrock"
        profiles = load_profiles(refresh=True)
        click.echo(f"Wrote {len(profiles)} profiles to {cache_path()}")
        for profile_id in profiles:
            click.echo(f"  {profile_id}  ({', '.join(aliases_for(profile_id))})")


class BedrockClaude(llm.Model):
    can_stream: bool = True

    class Options(llm.Options):
        max_tokens_to_sample: Optional[int] = Field(
            description="The maximum number of tokens to generate before stopping",
            default=DEFAULT_MAX_TOKENS,  # clamped per-model where Bedrock demands less
        )
        bedrock_model_id: Optional[str] = Field(
            description="Bedrock modelId or ARN of base, custom, or provisioned model",
            default=None,
        )
        thinking: Optional[str] = Field(
            description="Thinking mode: auto (on where supported), adaptive, or off",
            default="auto",
        )
        effort: Optional[str] = Field(
            description="Reasoning effort: low, medium, high, xhigh or max",
            default=None,
        )

        @field_validator("max_tokens_to_sample")
        def validate_length(cls, max_tokens_to_sample):
            if not (0 < max_tokens_to_sample <= 1_000_000):
                raise ValueError("max_tokens_to_sample must be in range 1-1,000,000")
            return max_tokens_to_sample

    def __init__(self, model_id, supports_attachments=False):
        self.model_id = model_id
        self.supports_attachments = supports_attachments
        if supports_attachments:
            image_mime_types = {
                f"image/{fmt}" for fmt in BEDROCK_CONVERSE_IMAGE_FORMATS
            }
            document_mime_types = set(
                MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT.keys()
            )
            self.attachment_types = image_mime_types.union(document_mime_types)

    @staticmethod
    def load_and_preprocess_image(file):
        """
        Load and pre-process the given image for use with Anthropic models and the Bedrock
        Converse API:
        * Resize if needed.
        * Convert into a supported format if needed.
        * Do nothing if the image is already compatible.
        Even if Bedrock can resize images for us, we do this here to avoid unnecessary
        bandwidth and to support additional image file types.

        :param file: An image file path.
        :return: A bytes, image_format tuple containing the resulting image data and format.
                 Use the original data/format if possible, and choose an appropriate format if
                 the image needed to be resized.
        """
        with open(file, "rb") as fp:
            img_bytes = fp.read()

        with Image.open(BytesIO(img_bytes)) as img:
            img_format = img.format
            width, height = img.size
            if (
                width > ANTHROPIC_MAX_IMAGE_LONG_SIZE
                or height > ANTHROPIC_MAX_IMAGE_LONG_SIZE
            ):
                # Resize the image while preserving the aspect ratio
                img.thumbnail(
                    (ANTHROPIC_MAX_IMAGE_LONG_SIZE, ANTHROPIC_MAX_IMAGE_LONG_SIZE)
                )

            # Change format if necessary
            if (
                img_format.lower() in BEDROCK_CONVERSE_IMAGE_FORMATS
                and img.size == (width, height)  # Original size, no resize needed
            ):
                return img_bytes, img_format.lower()

            # Re-export the image with the appropriate format
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                return buffer.getvalue(), "png"

    def image_path_to_content_block(self, path):
        """
        Create a Bedrock Converse content block out of the given image file path.
        :param path: A file path to an image file.
        :return: A Bedrock Converse API content block containing the image.
        """
        source_bytes, file_format = self.load_and_preprocess_image(path)

        return {"image": {"format": file_format, "source": {"bytes": source_bytes}}}

    @staticmethod
    def sanitize_file_name(file_path):
        """
        Generate a file name out of the given file path that conforms to the Bedrock
        Converse API conventions:
        * Alphanumeric characters
        * Whitespace characters (no more than one in a row)
        * Hyphens
        * Parentheses
        * Square brackets
        * Maximum length of 200.
        See also: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
        :param file_path:
        :return:
        """
        head, tail = os.path.split(file_path)
        name, ext = os.path.splitext(tail)

        # Sanitize the name part
        sanitized_name = ""
        for c in name:
            if (
                c
                in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_()[]"
            ):
                sanitized_name += c
            else:
                sanitized_name += "_"

        if not sanitized_name:
            sanitized_name = "file"

        # Ensure total length (including extension) is under 200
        max_name_length = 200 - len(ext)
        sanitized_name = sanitized_name[:max_name_length]

        return sanitized_name + ext

    def document_path_to_content_block(self, file_path, mime_type):
        """
        Create a Bedrock Converse content block out of the given document file path.
        :param file_path: A file path to a document file.
        :param mime_type: The file’s MIME type.
        :return: A Bedrock Converse API content block containing the document.
        """
        with open(file_path, "rb") as fp:
            source_bytes = fp.read()
        return {
            "document": {
                "format": MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT[mime_type],
                "name": self.sanitize_file_name(file_path),
                "source": {"bytes": source_bytes},
            }
        }

    def validate_attachment(self, data: AttachmentData) -> None:
        """Validate attachments"""
        if not (data.is_image or data.is_document):
            raise ValueError(f"Unsupported attachment type: {data.mime_type}")

    def process_attachment(self, data: AttachmentData) -> dict:
        """
        Process attachments and generate content blocks for Bedrock Converse API
        """
        self.validate_attachment(data)

        if data.is_image:
            if data.is_file_path:
                return self.image_path_to_content_block(str(data.content))
            return self.image_bytes_to_content_block(data.content, data.mime_type)

        if data.is_document:
            if data.is_file_path:
                return self.document_path_to_content_block(
                    str(data.content), data.mime_type
                )
            return self.document_bytes_to_content_block(
                data.content, data.mime_type, data.name
            )

    def create_attachment_data(self, attachment) -> AttachmentData:
        """Generate AttachmentData from a modern attachment"""
        if hasattr(attachment, "path") and attachment.path is not None:
            return AttachmentData(
                mime_type=attachment.type, content=Path(attachment.path)
            )

        return AttachmentData(
            mime_type=attachment.type,
            content=attachment.content_bytes(),
            name="attachment",
        )

    def image_bytes_to_content_block(self, image_bytes: bytes, mime_type: str) -> dict:
        """
        Create a Bedrock Converse content block from image bytes.
        :param image_bytes: The raw image bytes
        :param mime_type: The MIME type of the image
        :return: A Bedrock Converse API content block containing the image
        """
        with Image.open(BytesIO(image_bytes)) as img:
            width, height = img.size
            if (
                width > ANTHROPIC_MAX_IMAGE_LONG_SIZE
                or height > ANTHROPIC_MAX_IMAGE_LONG_SIZE
            ):
                # Resize the image while preserving the aspect ratio
                img.thumbnail(
                    (ANTHROPIC_MAX_IMAGE_LONG_SIZE, ANTHROPIC_MAX_IMAGE_LONG_SIZE)
                )

                # Re-export the image
                with BytesIO() as buffer:
                    img.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()
                    file_format = "png"
            else:
                # Use original format from mime_type
                file_format = mime_type.split("/")[-1]
                if file_format not in BEDROCK_CONVERSE_IMAGE_FORMATS:
                    # Convert to PNG if format not supported
                    with BytesIO() as buffer:
                        img.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                        file_format = "png"

        return {"image": {"format": file_format, "source": {"bytes": image_bytes}}}

    def document_bytes_to_content_block(
        self, doc_bytes: bytes, mime_type: str, name: Optional[str] = None
    ) -> dict:
        """
        Create a Bedrock Converse content block from document bytes.
        :param doc_bytes: The raw document bytes
        :param mime_type: The MIME type of the document
        :param name: Optional name for the document
        :return: A Bedrock Converse API content block containing the document
        """
        if name is None:
            name = "document"

        return {
            "document": {
                "format": MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT[mime_type],
                "name": self.sanitize_file_name(name),
                "source": {"bytes": doc_bytes},
            }
        }

    def prompt_to_content(self, prompt):
        """
        Convert a llm.Prompt object to the content format expected by the Bedrock Converse API.

        :param prompt: A llm Prompt objet.
        :return: A content object that conforms to the Bedrock Converse API.
        """
        content = []

        # Attachments with -a or --attachment
        if hasattr(prompt, "attachments"):
            data = [self.create_attachment_data(a) for a in prompt.attachments]
            content_blocks = [self.process_attachment(d) for d in data]
            content.extend(content_blocks)

        # Append the prompt text as a text content block.
        content.append({"text": prompt.prompt})

        return content

    def encode_bytes(self, o):
        """
        Recursively replace any "bytes" dict attribute in the given object with a base64
        encoded value as "bytes_b64". This is done to preserve the data during logging activities.

        :param o: A Python object.
        :return: A copy of the input, but with all "bytes" keys in dicts replaces by base64
                 encoded values names "bytes".
        """
        if isinstance(o, list):
            return [self.encode_bytes(i) for i in o]
        elif isinstance(o, dict):
            result = {}
            for key, value in o.items():
                if key == "bytes":
                    result["bytes_b64"] = b64encode(value).decode("utf-8")
                else:
                    result[key] = self.encode_bytes(value)
            return result
        else:
            return o

    def decode_bytes(self, o):
        """
        Recursively replace any "bytes_b64" dict attribute in the given object with a
        base64 decoded value as "bytes". This is the reverse of the above, so the resulting
        data can be sent to Bedrock in its expected form.

        :param o: A Python object.
        :return: A copy of the input, but with all "bytes_b64" keys in dicts replaced by base64
                 decoded values names "bytes".
        """
        if isinstance(o, list):
            return [self.decode_bytes(i) for i in o]
        elif isinstance(o, dict):
            result = {}
            for key, value in o.items():
                if key == "bytes_b64":
                    result["bytes"] = b64decode(value)
                else:
                    result[key] = self.decode_bytes(value)
            return result
        else:
            return o

    def messages_from_chain(self, prompt) -> Optional[List[dict]]:
        """
        Build Bedrock Converse messages from llm's canonical message chain.

        `prompt.messages` is the authoritative history in llm 0.32+, populated
        from the content-addressed log tables when resuming with `llm -c`.
        build_messages() below instead walks `conversation.responses`, which
        those versions no longer populate - so relying on it silently sends no
        history at all.

        :param prompt: A llm Prompt object.
        :return: Bedrock Converse messages, or None on llm versions that
                 predate the message chain, so the caller can fall back.
        """
        try:
            chain = prompt.messages
        except AttributeError:
            return None
        if not chain:
            return None

        messages = []
        for message in chain:
            if message.role == "system":
                # Carried by the top-level `system` parameter instead.
                continue
            content = []
            for part in message.parts:
                if getattr(part, "type", None) == "reasoning":
                    # Bedrock rejects replayed reasoning without its signature.
                    continue
                text = getattr(part, "text", None)
                if text:
                    content.append({"text": text})
                attachment = getattr(part, "attachment", None)
                if attachment is not None:
                    content.append(
                        self.process_attachment(self.create_attachment_data(attachment))
                    )
            if content:
                messages.append({"role": message.role, "content": content})
        return messages or None

    def build_messages(self, prompt_content, conversation) -> List[dict]:
        """Legacy history reconstruction, for llm versions before 0.32."""
        messages = []
        if conversation:
            for response in conversation.responses:
                if (
                    response.response_json
                    and "bedrock_user_content" in response.response_json
                ):
                    user_content = self.decode_bytes(
                        response.response_json["bedrock_user_content"]
                    )
                else:
                    user_content = [{"text": response.prompt.prompt}]
                assistant_content = [{"text": response.text()}]
                messages.extend(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                )

        messages.append({"role": "user", "content": prompt_content})
        return messages

    def additional_request_fields(self, prompt) -> dict:
        """
        Model-specific Converse fields for thinking and reasoning effort. Left
        with thinking off, current models tend to write their reasoning into
        the visible answer, so it is enabled by default where supported.
        """
        extra = {}
        thinking = (prompt.options.thinking or "auto").lower()
        if thinking == "auto":
            thinking = "adaptive" if ADAPTIVE_THINKING.search(self.model_id) else "off"
        if thinking != "off":
            extra["thinking"] = {"type": thinking}
        if prompt.options.effort:
            extra["output_config"] = {"effort": prompt.options.effort}
        return extra

    def execute(self, prompt, stream, response, conversation):
        prompt_content = self.prompt_to_content(prompt)
        messages = self.messages_from_chain(prompt)
        if messages is None:
            messages = self.build_messages(prompt_content, conversation)

        # Preserve the Bedrock-specific user content dict, so it can be re-used in
        # future conversations.
        response.response_json = {
            "bedrock_user_content": self.encode_bytes(prompt_content)
        }

        max_tokens = prompt.options.max_tokens_to_sample
        if LEGACY_4K.search(self.model_id):
            max_tokens = min(max_tokens, 4096)

        # Put together parameters for the Bedrock Converse API.
        params = {
            "modelId": prompt.options.bedrock_model_id or self.model_id,
            "messages": messages,
            "inferenceConfig": {"maxTokens": max_tokens},
        }

        if prompt.system:
            params["system"] = [{"text": prompt.system}]

        extra = self.additional_request_fields(prompt)
        if extra:
            params["additionalModelRequestFields"] = extra

        client = boto3.client("bedrock-runtime")
        if stream:
            bedrock_response = client.converse_stream(**params)
            response.response_json |= bedrock_response
            events = []
            for event in bedrock_response["stream"]:
                ((event_type, event_content),) = event.items()
                if event_type == "contentBlockDelta":
                    delta = event_content["delta"]
                    # Reasoning deltas carry no "text"; they stay in the log
                    # but are not part of the visible answer.
                    if "text" in delta:
                        yield delta["text"]
                events.append(event)
            response.response_json["stream"] = events
        else:
            bedrock_response = client.converse(**params)
            response.response_json |= bedrock_response
            blocks = bedrock_response["output"]["message"]["content"]
            # Skip reasoningContent blocks; join whatever text blocks remain.
            yield "".join(block["text"] for block in blocks if "text" in block)
        self.set_usage(response)

    def set_usage(self, response: llm.Response):
        if not hasattr(response, "set_usage"):
            # Older versions of llm do not have this method
            return
        res_json = response.response_json

        if "usage" in res_json:
            response.set_usage(
                input=res_json["usage"]["inputTokens"],
                output=res_json["usage"]["outputTokens"],
            )
        elif "stream" in res_json:
            events = res_json["stream"]
            for event in events:
                ((event_type, event_content),) = event.items()
                if event_type == "metadata":
                    input_tokens = event_content["usage"]["inputTokens"]
                    output_tokens = event_content["usage"]["outputTokens"]
                    response.set_usage(input=input_tokens, output=output_tokens)
                    break
