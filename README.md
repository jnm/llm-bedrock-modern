# llm-bedrock-modern

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/jnm/llm-bedrock-modern/blob/main/LICENSE)

Plugin for [LLM](https://llm.datasette.io/) adding support for Anthropic's
Claude models on AWS Bedrock.

A fork of [sblakey/llm-bedrock-anthropic](https://github.com/sblakey/llm-bedrock-anthropic),
whose PyPI release predates every current Claude model. The headline
difference: **the model list is
discovered from your own AWS account** instead of being hardcoded in the
source, so new Claude models work the day AWS enables them for you, with no
code change.

## Installation

```bash
llm install 'git+https://github.com/jnm/llm-bedrock-modern.git'
llm bedrock-refresh
```

`llm install` is a `pip install` passthrough, so a tag or branch works too:
`...llm-bedrock-modern.git@v0.5.0`.

If you have the original plugin installed, remove it first — otherwise you get
two sets of Bedrock models, one of them stale:

```bash
llm uninstall -y llm-bedrock-anthropic
```

## Configuration

Credentials and region come from boto3, i.e. the usual environment variables:

```bash
export AWS_DEFAULT_REGION=us-east-1
export AWS_PROFILE=personal
```

`llm bedrock-refresh` calls `bedrock:ListInferenceProfiles`, keeps every ACTIVE
Anthropic profile, and caches the result in
`~/.config/io.datasette.llm/bedrock-anthropic-profiles.json`. Model
registration reads only that cache, so ordinary `llm` invocations make no extra
AWS calls and a missing credential can never take down the CLI. Re-run it when
AWS adds a model.

## Usage

Models are registered under their inference profile ID, with two aliases each:

| Model | Aliases |
| --- | --- |
| `us.anthropic.claude-opus-5` | `bedrock-opus-5`, `bo5` |
| `us.anthropic.claude-sonnet-5` | `bedrock-sonnet-5`, `bs5` |
| `global.anthropic.claude-opus-5` | `bedrock-opus-5-global`, `bo5-global` |
| `us.anthropic.claude-haiku-4-5-20251001-v1:0` | `bedrock-haiku-4-5`, `bh4.5` |

`llm models` lists whatever your account actually has. Discovery only reports
what Bedrock advertises, not what your account is entitled to use — e.g. the
Fable models return `data retention mode 'default' is not available for this
model` unless 30-day retention is enabled for the account.

```bash
llm -m bo5 'hello'
llm -m bo5 -o effort low 'quick question'
llm -m bo5 -a diagram.png 'what does this show?'
llm -c 'and what about the second one?'
llm models default bedrock-opus-5
```

### Options

- `max_tokens_to_sample` — defaults to 16000, clamped to 4096 for Claude 3-era
  models that reject more.
- `thinking` — `auto` (the default: adaptive on models that support it, off
  otherwise), `adaptive`, or `off`.
- `effort` — `low` / `medium` / `high` / `xhigh` / `max`.
- `bedrock_model_id` — override the Bedrock modelId or ARN, e.g. for a
  provisioned or custom model.

Thinking defaults to on because with it off, current Claude models tend to
write their reasoning into the visible answer instead.

## Changes from upstream

- **Model list discovered from the account** rather than hardcoded, with a
  cached list and an `llm bedrock-refresh` command.
- **Fixed `KeyError: 'text'` on streaming.** Upstream reads
  `event["delta"]["text"]` for every `contentBlockDelta`, which crashes as soon
  as a model emits `reasoningContent` deltas — i.e. on every current model with
  thinking enabled. Reasoning deltas are now skipped for display and kept in
  the logged response JSON.
- **Fixed silently-empty history on `llm -c`.** Upstream rebuilds history from
  `conversation.responses`, which llm has not populated since 0.32: writes go
  to the content-addressed `turns`/`messages`/`parts` tables, and
  `load_conversation()` only fills `conversation.responses` from the legacy
  `responses` table. `messages_from_chain()` builds from `prompt.messages`
  instead, which is the canonical chain in llm 0.32+, and falls back to
  upstream's path on older llm versions.
- **Added `thinking` and `effort` options**, and raised the default
  `max_tokens_to_sample` from 4096 to 16000 with a per-model clamp.
- **Removed** the hardcoded model registry, the Claude 2 / Claude Instant
  system-prompt workaround, and the pre-attachments `-o bedrock_attach`
  option (use `-a` / `--attachment`).

## Development

```bash
llm install -e '.[test]'
python -m pytest
```

See [MAINTENANCE.md](MAINTENANCE.md) for the parts that need attention when new
models ship, the invariants the tests protect, and where this plugin is coupled
to `llm`'s internals.

## Credits

Original plugin by Sean Blakey and Will Sorenson, itself derived from
[tomviner/llm-claude](https://github.com/tomviner/llm-claude). Fork work
(account discovery, streaming and conversation-history fixes, thinking/effort
options) by Claude Opus 5. Apache-2.0, as upstream.
