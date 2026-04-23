# Anchored Refinement Loss

## Motivation

Existing loss functions in CLAP-based audio effect optimization (directional loss, forward loss, guided forward loss) were designed for **from-scratch** generation: given dry audio and a text descriptor, find parameters that make the audio match the text.

However, our problem is **iterative refinement**: given audio that already has effects applied (with parameters from a previous stage or previous user prompt), modify the parameters so that the audio moves in the direction specified by a new instruction — while not unnecessarily disturbing parameters that are unrelated to the instruction.

Current implementation always uses "dry audio" as the anchor (`parser.py:165`, `trainer.py:74`), making the system blind to the existing state during refinement.

## Problem Formulation

Let:
- $x$ — dry input audio
- $\theta_{\text{prev}}$ — existing (normalized) parameters from previous stage
- $\theta$ — parameters being optimized
- $\sigma(\cdot)$ — sigmoid (maps unconstrained → [0,1])
- $I$ — current text instruction (e.g., "make it brighter")
- $I_{\text{prev}}$ — previous text instruction (or "dry audio" if first prompt)
- $\text{CLAP}_a(\cdot)$, $\text{CLAP}_t(\cdot)$ — frozen CLAP audio/text encoders

Define the deltas:

$$\Delta_a = \text{CLAP}_a(\text{FX}(x, \sigma(\theta))) - \text{CLAP}_a(\text{FX}(x, \sigma(\theta_{\text{prev}})))$$

$$\Delta_t = \text{CLAP}_t(I) - \text{CLAP}_t(I_{\text{prev}})$$

$$\Delta_\theta = \sigma(\theta) - \sigma(\theta_{\text{prev}})$$

## Loss Function

$$\mathcal{L}_{\text{refine}}(\theta) = \underbrace{\left(1 - \cos(\Delta_a, \Delta_t)\right)}_{\text{sufficiency}} + \alpha \cdot \underbrace{\|\Delta_\theta\|_2^2 \cdot \left(1 - |\cos(\Delta_a, \Delta_t)|\right)}_{\text{minimality}}$$

### Term 1: Sufficiency (directional alignment)

$$\mathcal{L}_{\text{suf}} = 1 - \frac{\Delta_a \cdot \Delta_t}{\|\Delta_a\| \|\Delta_t\|}$$

Structurally identical to Text2FX's directional loss, but with **anchors set to the previous state** instead of dry audio. Drives the audio embedding to move in the direction that the text instruction specifies, relative to where the audio currently is.

### Term 2: Minimality (parameter-space regularization gated by alignment)

$$\mathcal{L}_{\text{min}} = \|\Delta_\theta\|_2^2 \cdot \left(1 - |\cos(\Delta_a, \Delta_t)|\right)$$

This term is the product of two factors:

1. **Parameter displacement** $\|\Delta_\theta\|_2^2$: how much the parameters changed from the previous state (in normalized [0,1] space).

2. **Alignment complement** $(1 - |\cos(\Delta_a, \Delta_t)|)$: how poorly the audio movement direction matches the text direction.

**Behavior:**
- When audio moves in the **correct direction** (cos ≈ 1): alignment complement → 0, so parameter changes are **unrestricted**. The optimizer is free to make large parameter adjustments as long as they're productive.
- When audio moves in an **unrelated direction** (cos ≈ 0): alignment complement → 1, so parameter changes are **penalized**. The optimizer is discouraged from drifting parameters in ways that don't serve the instruction.

This does not assume the user "approved" the previous parameters. It is purely a mathematical constraint: parameter changes that do not contribute to the instructed direction are wasteful and should be suppressed.

### Hyperparameter α

$\alpha$ controls the strength of the minimality term. Suggested default: `α = 0.1`.
- Higher α → more conservative refinement (less parameter drift)
- Lower α → more aggressive refinement (closer to pure directional loss)

## Comparison with Existing Losses

| Property | forward_loss | directional_loss | guided_forward_loss | **refinement_loss** |
|---|---|---|---|---|
| Audio anchor | — | dry audio | — | FX(x, θ_prev) |
| Text anchor | — | "dry audio" | hardcoded negative | I_prev |
| Absolute vs relative | absolute | relative to dry | absolute | **relative to prev state** |
| Parameter regularization | no | no | no | **yes (gated)** |
| Designed for | from-scratch | from-scratch | from-scratch | **iterative refinement** |

## Degenerate Cases

- **First prompt** (no previous state): set $\theta_{\text{prev}}$ to LLM-initialized params, $I_{\text{prev}}$ to "dry audio". Reduces to approximately directional loss with LLM-init anchor + light regularization.
- **α = 0**: Reduces to directional loss with updated anchors (still an improvement over current implementation).
- **Identical instruction repeated** ($I = I_{\text{prev}}$): $\Delta_t = 0$, loss is dominated by minimality term → parameters stay put. Correct behavior: if the user repeats the same instruction, there's nothing new to refine.

## Implementation Plan

1. Add `refinement_loss()` to `src/training/loss.py`
2. Update `move_in_CLAP()` in `src/training/trainer.py` to accept `prev_params` and `prev_instruction`
3. Update `parser.py` to pass existing params and session history into the optimizer for Case 1 (refinement)
