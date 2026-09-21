import json

import pytest

from llm_bedrock_modern import (
    DEFAULT_MAX_TOKENS,
    SEED_PROFILES,
    BedrockClaude,
    aliases_for,
    load_profiles,
    register_models,
)


class FakePart:
    def __init__(self, type="text", text=None, attachment=None):
        self.type = type
        self.text = text
        self.attachment = attachment


class FakeMessage:
    def __init__(self, role, parts):
        self.role = role
        self.parts = parts


class FakePrompt:
    def __init__(self, messages=None, **options):
        if messages is not None:
            self.messages = messages
        self.prompt = "hello"
        self.system = None
        self.attachments = []
        self.options = BedrockClaude.Options(**options)


@pytest.mark.parametrize(
    "profile_id,expected",
    [
        ("us.anthropic.claude-opus-5", ["bedrock-opus-5", "bo5"]),
        (
            "global.anthropic.claude-opus-5",
            ["bedrock-opus-5-global", "bo5-global"],
        ),
        ("us.anthropic.claude-sonnet-4-6", ["bedrock-sonnet-4-6", "bs4.6"]),
        ("us.anthropic.claude-fable-5-1", ["bedrock-fable-5-1", "bf5.1"]),
        # Date and version suffixes are dropped from the alias.
        (
            "us.anthropic.claude-opus-4-1-20250805-v1:0",
            ["bedrock-opus-4-1", "bo4.1"],
        ),
        ("us.anthropic.claude-opus-4-6-v1", ["bedrock-opus-4-6", "bo4.6"]),
    ],
)
def test_aliases_for(profile_id, expected):
    assert aliases_for(profile_id) == expected


def test_load_profiles_prefers_cache(tmp_path, monkeypatch):
    cache = tmp_path / "bedrock-anthropic-profiles.json"
    cache.write_text(json.dumps({"profiles": ["us.anthropic.claude-opus-5"]}))
    monkeypatch.setattr("llm_bedrock_modern.cache_path", lambda: cache)

    def boom():
        raise AssertionError("registration must not call AWS")

    monkeypatch.setattr("llm_bedrock_modern.fetch_profiles", boom)
    assert load_profiles() == ["us.anthropic.claude-opus-5"]


def test_load_profiles_survives_aws_failure(tmp_path, monkeypatch):
    """A missing credential must not take down the whole llm CLI."""
    monkeypatch.setattr(
        "llm_bedrock_modern.cache_path", lambda: tmp_path / "missing.json"
    )
    monkeypatch.setattr(
        "llm_bedrock_modern.fetch_profiles",
        lambda: (_ for _ in ()).throw(RuntimeError("no credentials")),
    )
    assert load_profiles() == SEED_PROFILES


def test_register_models_dedupes_aliases(monkeypatch):
    monkeypatch.setattr(
        "llm_bedrock_modern.load_profiles",
        lambda: ["us.anthropic.claude-opus-5", "us.anthropic.claude-opus-5"],
    )
    registered = []
    register_models(lambda model, aliases=(): registered.append((model, aliases)))
    assert [aliases for _, aliases in registered] == [("bedrock-opus-5", "bo5"), ()]


def test_messages_from_chain_round_trip():
    model = BedrockClaude("us.anthropic.claude-opus-5", supports_attachments=True)
    chain = [
        FakeMessage("system", [FakePart(text="be terse")]),
        FakeMessage("user", [FakePart(text="my number is 41")]),
        FakeMessage(
            "assistant",
            [
                FakePart(type="reasoning", text="they said 41"),
                FakePart(text="noted"),
            ],
        ),
        FakeMessage("user", [FakePart(text="what number?")]),
    ]
    messages = model.messages_from_chain(FakePrompt(messages=chain))
    # System parts go in the top-level `system` field, reasoning is dropped
    # because Bedrock rejects it without the original signature.
    assert messages == [
        {"role": "user", "content": [{"text": "my number is 41"}]},
        {"role": "assistant", "content": [{"text": "noted"}]},
        {"role": "user", "content": [{"text": "what number?"}]},
    ]


def test_messages_from_chain_absent_on_old_llm():
    """Without prompt.messages we must fall back to build_messages()."""
    model = BedrockClaude("us.anthropic.claude-opus-5")
    assert model.messages_from_chain(FakePrompt()) is None


def test_thinking_defaults_on_for_current_models():
    model = BedrockClaude("us.anthropic.claude-opus-5")
    extra = model.additional_request_fields(FakePrompt(messages=[]))
    assert extra == {"thinking": {"type": "adaptive"}}


def test_thinking_defaults_off_for_legacy_models():
    model = BedrockClaude("us.anthropic.claude-3-haiku-20240307-v1:0")
    assert model.additional_request_fields(FakePrompt(messages=[])) == {}


def test_thinking_and_effort_can_be_set():
    model = BedrockClaude("us.anthropic.claude-opus-5")
    extra = model.additional_request_fields(
        FakePrompt(messages=[], thinking="off", effort="low")
    )
    assert extra == {"output_config": {"effort": "low"}}


def test_default_max_tokens_is_generous():
    assert BedrockClaude.Options().max_tokens_to_sample == DEFAULT_MAX_TOKENS


def test_max_tokens_rejects_out_of_range():
    with pytest.raises(ValueError):
        BedrockClaude.Options(max_tokens_to_sample=0)
