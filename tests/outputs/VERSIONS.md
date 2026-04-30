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

### v2 — Layer 1 sees session history (interim design, NOT run)

Two minimal changes to `src/agents/fx_selector.py` over v1, still on `gpt-4o`
with a single system prompt:

1. `select()` takes `session.history` and renders prior turns into the user
   message so the selector knows what's already in the chain.
2. `tool_choice` switched from `"required"` to `"auto"` so the selector can
   return zero tool calls when the existing chain already covers the
   request, letting parser Case 1 re-tune the existing FX.

This was the originally-scoped v2. It was rolled forward into v3 before
being run, so there are no `v2_*` output dirs — v2 exists only as a
design step, kept here to make the v1 → v3 trajectory legible.

### v3 — split initial/refinement prompts + Gemini 3 Pro

Builds on v2:

1. `select()` now uses **two distinct system prompts** depending on whether
   it's the first turn:
   - **Initial path** (empty history, turn 1): a clean "pick FX for this
     descriptor" prompt with no mention of refinement / existing chain,
     and `tool_choice="required"` so the model must pick at least one FX.
   - **Refinement path** (non-empty history): the v2-style history-aware
     prompt with `tool_choice="auto"`. Tools for FX already in the chain
     are filtered out so the model can't double-add.
2. Model swapped from `gpt-4o` to `google/gemini-3.1-pro-preview` (via
   OpenRouter). Goal: cleaner one-shot tool-calling decisions, less
   over-adding (e.g. `church → reverb only`, not `eq + reverb`).

Goal in `B/turn3` ("too harsh, soften it"): selector returns zero tool
calls, orchestrator carries the existing chain into parser Case 1, BO
re-tunes the existing distortion instead of stacking an EQ.

### v4 — (planned) selector may also REMOVE existing FX

Not implemented yet. The idea is to give the selector a `remove_<fx>` tool
(or otherwise express subtraction) so it can prune the chain when a
reprompt suggests an earlier FX is no longer wanted. Requires plumbing
changes in `Session` / `Parser` to actually drop FX from `current_params`,
so it is intentionally deferred.
