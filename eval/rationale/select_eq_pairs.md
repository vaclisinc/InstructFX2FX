# EQ Word Pair Selection

## Dataset

Seven EQ adjectives from the SocialFX dataset (seungheondoh/socialfx-gen-eval, `eq` split) passed our selection criteria (sample count ≥ 20 and inter-annotator consistency ≥ 0.45):

| Word | Count | Consistency |
|------|------:|------------:|
| warm | 93 | 0.465 |
| bright | 74 | 0.480 |
| soft | 74 | 0.540 |
| loud | 54 | 0.552 |
| harsh | 39 | 0.492 |
| calm | 26 | 0.583 |
| heavy | 25 | 0.560 |

## Mean EQ Curve Profiles

Mean EQ parameter values (40-band Audealize curves) computed across all samples per word. Values summarized at low (bands 0–4) and high (bands 35–39) ends of the spectrum:

| Word | Low-freq (0–4) | High-freq (35–39) | Dominant character |
|------|:--------------:|:-----------------:|-------------------|
| warm | +0.7 to +1.0 | −0.1 to −0.3 | Low-shelf boost |
| bright | −1.0 to −1.1 | −0.2 to −0.3 | Low-shelf cut |
| heavy | +0.9 to +1.4 | −0.4 to −0.5 | Strong low-shelf boost |
| harsh | −0.7 to −0.9 | −0.4 to −0.5 | Broad cut |
| soft | +0.3 to +0.4 | +0.4 to +0.5 | Gentle full-range lift |
| calm | +0.1 to +0.2 | +0.5 to +0.6 | High-shelf boost |
| loud | −0.1 to +0.4 | −0.9 to −1.1 | High-shelf cut |

## Pair Selection

Each directed pair (A → B) represents an instruction to transform audio processed with word A's EQ toward word B's timbral quality. The sequential ground-truth (GT) audio is rendered by applying params_A then params_B to the dry recording.

A key design concern is **spectral cancellation**: if A and B have near-opposite EQ curves in the same frequency region, the sequential GT collapses toward the unprocessed dry signal, making the evaluation degenerate.

### Control Group

Two bidirectional opposite pairs are intentionally included as a **control condition**. Their sequential GTs are expected to partially cancel, providing a degenerate baseline for comparison:

| Pair | Reason |
|------|--------|
| warm → bright | Low-shelf boost (warm) followed by low-shelf cut (bright): strong low-freq cancellation |
| bright → warm | Symmetric inverse of the above |

### Main Experiment Pairs

Remaining pairs are selected to minimize spectral cancellation by targeting different or complementary frequency regions:

| Pair | A character | B character | Spectral relationship |
|------|-------------|-------------|----------------------|
| heavy → calm | Strong low boost | High-shelf boost | Opposite ends of spectrum; additive across bands |
| heavy → harsh | Strong low boost | Broad cut | B attenuates what A emphasized; tests reversal of a specific region |
| harsh → soft | Broad cut | Gentle full-range lift | B partially repairs A's attenuation |
| harsh → calm | Broad cut | High-shelf boost | B adds highs that A suppressed |
| warm → heavy | Low boost | Stronger low boost | Same direction, more extreme; tests degree of change |
| soft → loud | Gentle full-range lift | High-shelf cut | B cuts the highs that A raised |
| calm → loud | High-shelf boost | High-shelf cut | Same region, opposite direction; intentional within-region contrast |
| loud → heavy | High-shelf cut | Strong low boost | Opposite ends of spectrum; complementary |

## Notes

- "Spacious" and other spatial/reverb-related adjectives are absent from the EQ split of SocialFX. These may be incorporated once the reverb evaluation branch is implemented.
- The control pairs (warm ↔ bright) are useful for diagnosing whether a low MMD score reflects genuine timbral transformation or mere spectral cancellation.
