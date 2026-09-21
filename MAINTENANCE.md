# Maintenance notes

Notes for whoever touches this next, written while the context was fresh. The
README is for people using the plugin; this is for people changing it.

## What rots, and what doesn't

The point of the fork was that a hardcoded model list guarantees the plugin
dies the moment AWS ships something new. That problem is solved — models come
from `ListInferenceProfiles`. But **two pieces of hardcoded model knowledge
remain**, and they are the most likely reason a future model behaves oddly:

- `ADAPTIVE_THINKING` — which models accept `thinking={"type": "adaptive"}`.
  A model not matched here gets thinking off, and current Claude models with
  thinking off tend to narrate their reasoning into the visible answer. If a
  new family ships under a new codename (the pattern already carries `fable`
  and `mythos`), **add it to this regex**, or the model will work but sound
  wrong.
- `LEGACY_4K` — models that reject `maxTokens > 4096`. Only Claude 3-era and
  older. Symptom of a miss: `ValidationException: ... exceeds the model limit`.

Neither can be discovered from the Bedrock API as far as I could tell — the
profile listing says nothing about thinking support or token ceilings. If that
changes, deleting these regexes is the right move.

`SEED_PROFILES` is a third one, but it only matters on a machine that has never
successfully run `llm bedrock-refresh`, so it ages harmlessly.

## Design invariant: registration must not touch the network

`register_models()` runs on **every** `llm` invocation, including `llm --help`.
So it reads only the on-disk cache, and `load_profiles()` swallows every
exception and falls back to the cache and then to `SEED_PROFILES`. A broken or
absent AWS credential must never make the whole `llm` CLI unusable — and it
must never add latency to unrelated commands.

`tests/test_discovery.py` enforces this: `test_load_profiles_prefers_cache`
monkeypatches `fetch_profiles` to raise if it is called at all. Keep that test
honest. Live AWS calls belong only in `llm bedrock-refresh`.

## Aliases

`aliases_for()` turns `us.anthropic.claude-opus-5` into `bedrock-opus-5` and
`bo5`, dropping date (`20250805`) and version (`v1:0`) suffixes, and appending
`-global` for non-`us` profiles.

Gotcha: aliases are deduplicated **first-wins** across the profile list. If two
profiles reduce to the same alias — say a bare `claude-opus-4-6` and a dated
`claude-opus-4-6-20260101-v1:0` — the second registers with *no* aliases and is
reachable only by its full profile ID. Nothing errors; the alias just silently
belongs to whichever came first in the cache. If that starts happening, the fix
is to make the suffix-stripping keep enough to disambiguate.

## Coupling to llm internals

This is the fragile part. Two places reach into `llm`'s data model rather than a
stable API, and both are worth re-checking after an `llm` upgrade:

- `messages_from_chain()` reads `prompt.messages` — a list of objects with
  `.role` and `.parts`, where parts have `.type`, `.text`, `.attachment`. This
  is how llm 0.32+ supplies conversation history. It is what upstream got wrong:
  upstream reads `conversation.responses`, which llm stopped populating in
  **0.32** when logging moved to the content-addressed `turns`/`messages`/
  `parts` tables. `-c` then silently produced an amnesiac model rather than an
  error, which is why it went unnoticed.
- `build_messages()` is kept only as the pre-0.32 fallback, reached when
  `prompt.messages` is absent. If you ever drop support for llm < 0.32, that
  method and its `conversation` plumbing can go.

Developed against **llm 0.35**. If history breaks again, compare against
`cli.py` (`conversation.loaded_messages = LogStore(db).thread_messages(...)`)
and `models.py` (where that becomes `prompt.messages`) before assuming it's a
bug here — and note the reverse also applies: the last time this looked like an
llm regression, it was this plugin's fault.

Reasoning parts are deliberately **dropped** when replaying history: Bedrock
rejects a `reasoningContent` block without the original cryptographic
signature, which llm does not store.

## Testing

`python -m pytest` — 21 tests, no AWS, no network, no credentials needed.

The test suite can't cover the Converse wire format, so after any change to
`execute()` run a live smoke pass:

```bash
llm -m bo5 'hi'                                  # streaming + reasoning deltas
llm -c 'what did I just say?'                    # history round-trip
llm -m bo5 -a some.png 'what is this?'           # attachments
llm -m bo5 --no-stream -o effort low 'hi'        # non-streaming + effort
llm -m bedrock-3-haiku 'hi'                      # the 4096 clamp
llm logs -n 1 -u                                 # token usage recorded
```

The streaming bug this fork fixed (`KeyError: 'text'` on `reasoningContent`
deltas) would have been caught by the first line and by nothing else, which is
a fair summary of how much the live pass matters.

## Syncing with upstream

`upstream` is `sblakey/llm-bedrock-anthropic`. The module was renamed with
`git mv`, so `git log --follow llm_bedrock_modern.py` still reaches upstream's
history and `git merge upstream/main` has a chance of working. Upstream was
inactive and had no releases past the hardcoded-model era when this fork was
made, so expect to cherry-pick at most.

Apache-2.0 from upstream: keep `LICENSE` and the attribution in `README.md` and
`pyproject.toml` intact, including the `tomviner/llm-claude` credit that
upstream itself inherited.
