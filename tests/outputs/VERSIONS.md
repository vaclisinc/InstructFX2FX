# Output Versions

This folder holds end-to-end pipeline test outputs (`test_pipeline.py`).
Subdirectories are named `<version>_<case>` (e.g. `v1_A`, `v2_B`).

Each version reflects a snapshot of pipeline behavior — when the pipeline
changes in a way that should produce different results, bump `VERSION` in
`tests/test_pipeline.py` so prior outputs are not overwritten.

## Cases (same across all versions)

- **A** — EQ only (DASP path). `"bright"` → `"warmer"`.
- **B** — Mixed DASP + Pedalboard. `"make it sound like a church"` → `"add some grit"` → `"too harsh, soften it"`.
- **C** — Pedalboard only. `"distorted and crushed"`.

## Versions

### v1 — baseline (initial 3-layer pipeline)

Layer 1 (`FXSelectorAgent`) saw only the current instruction with no session
history, and used `tool_choice="required"`, forcing at least one new FX every
turn. Issue surfaced in `v1_B/turn3` ("too harsh, soften it"): the selector
stacked an EQ on top of the existing reverb+distortion chain instead of
re-tuning the existing distortion, and the BO result was poor.

### v2 — Layer 1 sees session history + may add no FX

Two changes to `src/agents/fx_selector.py`:

1. `select()` now takes `session.history` and renders the prior turns
   (`prompt → chain became [...]`) into the user message, so the selector
   knows what's already in the chain and why.
2. `tool_choice` switched from `"required"` to `"auto"`. The selector may
   return zero tool calls when the existing chain already contains the right
   effects; in that case the orchestrator carries the existing chain into
   parser Case 1, which re-optimizes the existing params for the new
   descriptor.

Goal: in cases like `B/turn3`, let the selector skip adding FX and let BO
re-tune the existing distortion to soften it, instead of stacking EQ.

### v3 — (planned) selector may also REMOVE existing FX

Not implemented in this session. The idea is to give the selector a
`remove_<fx>` tool (or otherwise express subtraction) so it can prune the
chain when a reprompt suggests an earlier FX is no longer wanted. Requires
plumbing changes in `Session` / `Parser` to actually drop FX from
`current_params`, so it is intentionally deferred.
