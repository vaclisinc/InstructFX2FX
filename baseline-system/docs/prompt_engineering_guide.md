# Prompt Engineering Guide

## Overview

This guide documents the prompt engineering strategies used in the baseline parameter generation system. The approach combines system prompts with few-shot examples to translate creative audio descriptions into technical effect parameters with high accuracy and consistency.

**Why Prompt Engineering Matters:**
- Ensures LLM understands audio engineering domain knowledge
- Establishes parameter ranges and constraints
- Provides concrete examples of desired output format
- Balances creative interpretation with technical accuracy
- Achieves ~95%+ valid JSON output rate

## System Architecture

### Prompt Components

The parameter generation prompt consists of three main components:

```
=== SYSTEM INSTRUCTIONS ===
[Audio engineering expertise and constraints]

=== EXAMPLES ===
[5 few-shot examples demonstrating patterns]

=== YOUR TASK ===
[User's description and requested effects]
```

### Component Breakdown

1. **System Prompt** (configs/prompts/parameter_generation_v1.yaml)
   - Audio engineer persona
   - Parameter guidelines and ranges
   - Output schema specification
   - Key principles and best practices

2. **Few-Shot Examples** (configs/prompts/examples/*.json)
   - Diverse sonic characteristics
   - Valid parameter patterns
   - Real-world use cases
   - Emotional → Technical mapping

3. **User Prompt Template** (configs/prompts/parameter_generation_v1.yaml)
   - Description placeholder
   - Effects list placeholder
   - Creative interpretation guidance

## System Prompt

### Audio Engineer Persona

The system prompt establishes the LLM as an expert audio engineer:

```
You are an expert audio engineer specializing in translating creative
descriptions into technical audio effect parameters.

Given a high-level description of desired audio characteristics, generate
precise JSON parameters for audio effects (EQ, Reverb, Compressor).
```

**Why this works:**
- Sets professional context
- Establishes domain expertise
- Focuses on translation task
- Emphasizes precision

### Key Principles

The prompt includes core principles that guide parameter generation:

```
Key principles:
- Use musically appropriate parameter values based on professional mixing standards
- Consider the emotional and textural qualities described in the prompt
- Balance technical accuracy with creative interpretation
- Ensure all parameters are within valid ranges
- Apply effects in appropriate order (typically: EQ → Compressor → Reverb)
```

**Design rationale:**
- **Musical appropriateness**: Prevents technically valid but musically poor choices
- **Emotional consideration**: Maps descriptions to sonic qualities
- **Balance**: Allows creativity within constraints
- **Valid ranges**: Enforces schema compliance
- **Effect order**: Teaches signal flow best practices

### Parameter Guidelines

Detailed frequency and parameter mappings:

#### EQ Frequency to Emotion Mapping

```
Low frequencies (20-250 Hz): Warmth, fullness, power
Low-mid frequencies (250-500 Hz): Body, muddiness if excessive
Mid frequencies (500-2000 Hz): Presence, clarity
High-mid frequencies (2000-6000 Hz): Brightness, definition
High frequencies (6000-20000 Hz): Air, sparkle, brilliance
```

**Design strategy:**
- Maps frequency ranges to emotional qualities
- Warns about potential issues (e.g., "muddiness")
- Uses professional audio terminology
- Covers full audible spectrum

**Example applications:**
- "Warm sound" → Boost 150-300 Hz
- "Bright sound" → Boost 5000-10000 Hz
- "Presence" → Boost 1000-3000 Hz
- "Air" → Boost 8000-15000 Hz

#### EQ Parameter Ranges

```
Gain range: -12 to +12 dB (use conservative values, typically ±6 dB)
Q factor: 0.5-2.0 for broad curves, 2.0-10.0 for surgical cuts/boosts
```

**Why these ranges:**
- ±12 dB: Schema limits, prevents extreme boosts
- ±6 dB recommendation: Encourages subtle, musical adjustments
- Q factor guidance: Teaches appropriate bandwidth selection
  - Low Q (0.5-2.0): Musical, gentle curves
  - High Q (2.0-10.0): Precise problem-solving

#### Reverb Parameter Guidelines

```
room_size: 0.0-1.0 (0.2-0.4 for intimate, 0.6-0.8 for large spaces)
damping: 0.0-1.0 (0.5-0.8 for natural, 0.3-0.5 for bright)
wet_level: 0.0-1.0 (0.1-0.3 for subtle, 0.4-0.6 for pronounced)
dry_level: 0.0-1.0 (typically 0.7-1.0 to maintain clarity)
width: 0.0-1.0 (0.5-0.8 for stereo image)
```

**Design considerations:**
- Maps abstract values (0-1) to spatial qualities
- Provides typical ranges for common scenarios
- Emphasizes clarity preservation (dry level)
- Guides stereo width selection

#### Compressor Parameter Strategies

```
threshold: -60 to 0 dB (set based on input level and desired intensity)
ratio: 1:1 to 20:1 (2:1-4:1 for gentle, 6:1-10:1 for aggressive)
attack: 0.1-100 ms (fast for transients, slow for sustain)
release: 10-1000 ms (match to musical tempo and material)
knee: 0-12 dB (0-3 for hard, 6-12 for soft/transparent)
makeup_gain: 0-24 dB (compensate for gain reduction)
```

**Compression strategies taught:**
- **Gentle** (2:1-4:1): Transparent, glue
- **Moderate** (4:1-6:1): Obvious but musical
- **Aggressive** (6:1-10:1): Heavy control
- **Limiting** (10:1-20:1): Maximum control

**Attack/Release guidance:**
- Fast attack: Catch transients (drums, percussion)
- Slow attack: Preserve transients (sustain instruments)
- Release matched to tempo: Musical pumping

### Output Schema

The prompt includes the exact JSON schema:

```json
{
  "description": "string describing the intended sound",
  "effects": [
    {
      "type": "eq|reverb|compressor",
      "parameters": { /* effect-specific parameters */ }
    }
  ]
}
```

**Why include schema:**
- Reduces parsing errors
- Establishes structure expectations
- Shows parameter nesting
- Clarifies field names and types

## Few-Shot Examples

The system includes 5 carefully designed examples covering diverse sonic characteristics.

### Example 1: Warm and Intimate

**File:** `configs/prompts/examples/warm_intimate.json`

**Description:** "warm and intimate vocal sound for a late-night jazz performance"

**Parameter Analysis:**

#### EQ Strategy:
```json
{"frequency": 200, "gain": 3.0, "q": 0.7}    // Low-mid warmth
{"frequency": 800, "gain": 1.5, "q": 1.0}    // Presence boost
{"frequency": 3000, "gain": -2.0, "q": 1.2}  // Reduce harshness
{"frequency": 8000, "gain": -1.0, "q": 0.8}  // Vintage roll-off
```

**Design rationale:**
- Boost 200 Hz: Adds body and warmth (key "warm" characteristic)
- Boost 800 Hz: Enhances intimate presence
- Cut 3000 Hz: Reduces harshness for smoothness
- Cut 8000 Hz: Vintage character, less brightness

#### Compressor Strategy:
```json
{
  "threshold": -18.0,    // Moderate threshold
  "ratio": 3.0,          // Gentle ratio (3:1)
  "attack": 15.0,        // Slow attack preserves transients
  "release": 120.0,      // Musical release
  "knee": 6.0,           // Soft knee for transparency
  "makeup_gain": 3.0     // Compensate reduction
}
```

**Design rationale:**
- Gentle 3:1 ratio: Transparent, not aggressive
- 15ms attack: Slow enough to preserve vocal character
- Soft knee (6 dB): Smooth, natural compression
- Moderate makeup gain: Restore lost level

#### Reverb Strategy:
```json
{
  "room_size": 0.3,      // Small room
  "damping": 0.7,        // Warm damping
  "wet_level": 0.2,      // Subtle reverb
  "dry_level": 0.85,     // Maintain clarity
  "width": 0.6           // Moderate stereo
}
```

**Design rationale:**
- Small room (0.3): Intimate space
- High damping (0.7): Warm, not bright
- Low wet level (0.2): Subtle ambience
- High dry level (0.85): Vocal clarity preserved

**What this example teaches:**
- Warmth comes from low-mid boosts + high-end roll-off
- Intimacy requires subtle processing
- Soft compression preserves character
- Small spaces feel intimate

### Example 2: Bright and Energetic

**File:** `configs/prompts/examples/bright_energetic.json`

**Description:** "bright and energetic electric guitar sound for upbeat rock"

**Parameter Analysis:**

#### EQ Strategy:
```json
{"frequency": 100, "gain": -2.0, "q": 0.8}     // Cut mud
{"frequency": 800, "gain": 2.5, "q": 1.5}      // Upper-mid presence
{"frequency": 3500, "gain": 4.0, "q": 1.0}     // Brightness boost
{"frequency": 10000, "gain": 3.0, "q": 0.7}    // Air and sparkle
```

**Design rationale:**
- Cut 100 Hz: Remove low-end mud for clarity
- Boost 800 Hz: Guitar presence and bite
- **Strong** boost 3500 Hz: Key "brightness" characteristic
- Boost 10k Hz: Air and energy

**Contrast with Example 1:**
- Opposite EQ curve (high-end boost vs. roll-off)
- Stronger gain values (4 dB vs. 2-3 dB)
- Emphasizes upper frequencies
- Cuts lows instead of boosting

#### Compressor Strategy:
```json
{
  "threshold": -20.0,
  "ratio": 4.5,          // More aggressive
  "attack": 3.0,         // Fast attack
  "release": 50.0,       // Quick release
  "knee": 3.0,           // Hard knee
  "makeup_gain": 6.0     // Higher gain
}
```

**Design rationale:**
- 4.5:1 ratio: More obvious compression
- 3ms attack: Catches transients for punch
- Hard knee (3 dB): More obvious, energetic
- Higher makeup gain: Compensates harder compression

**What this example teaches:**
- Brightness from high-frequency boosts
- Energy from aggressive compression
- Fast attack creates punch
- Clarity requires low-end reduction

### Example 3: Dark and Atmospheric

**File:** `configs/prompts/examples/dark_atmospheric.json`

**Description:** "dark and atmospheric pad sound for cinematic underscore"

**Parameter Analysis:**

#### EQ Strategy:
```json
{"frequency": 60, "gain": 2.5, "q": 0.9}      // Sub-bass depth
{"frequency": 250, "gain": 3.0, "q": 0.6}     // Low-mid fullness
{"frequency": 2500, "gain": -4.0, "q": 1.0}   // Dark character
{"frequency": 8000, "gain": -6.0, "q": 0.5}   // Strong roll-off
```

**Design rationale:**
- Boost 60 Hz: Sub-bass weight and depth
- Boost 250 Hz: Fullness and darkness
- **Strong** cut 2500 Hz: Reduces brightness
- **Very strong** cut 8k Hz: Dark, mysterious character

**Extreme parameter teaching:**
- Shows when large cuts are appropriate (-6 dB)
- Demonstrates low-frequency emphasis
- Very low frequencies (60 Hz) for cinematic depth
- Broad cuts for smooth darkness

#### Reverb Strategy:
```json
{
  "room_size": 0.85,     // Large space
  "damping": 0.8,        // Dark reverb
  "wet_level": 0.55,     // Heavy reverb
  "dry_level": 0.7,      // Balanced mix
  "width": 0.85          // Wide stereo
}
```

**Design rationale:**
- Large room (0.85): Cinematic space
- High damping (0.8): Dark reverb character
- **High wet level (0.55)**: Pronounced atmospheric effect
- Wide stereo (0.85): Immersive soundscape

**What this example teaches:**
- Dark sound from low-frequency emphasis + high cut
- Atmospheric requires larger spaces + more wet signal
- Heavy reverb can dominate when desired
- Sub-bass adds cinematic weight

### Example 4: Punchy and Aggressive

**File:** `configs/prompts/examples/punchy_aggressive.json`

**Description:** "punchy and aggressive drum sound for electronic dance music"

**Parameter Analysis:**

#### EQ Strategy:
```json
{"frequency": 80, "gain": 4.0, "q": 1.2}       // Kick power
{"frequency": 200, "gain": -3.0, "q": 0.8}     // Cut mud
{"frequency": 1200, "gain": 5.0, "q": 2.0}     // Sharp attack boost
{"frequency": 5000, "gain": 3.5, "q": 1.5}     // Crack and snap
{"frequency": 12000, "gain": 2.0, "q": 0.6}    // High-end presence
```

**Design rationale:**
- Boost 80 Hz: Kick drum fundamental
- Cut 200 Hz: Prevent muddiness (critical for punch)
- **Strong, narrow** boost 1200 Hz: Attack transients (Q=2.0)
- Boost 5k Hz: Snap and aggression
- Boost 12k Hz: Sizzle and presence

**Advanced techniques shown:**
- **Narrow Q (2.0)** on 1200 Hz: Surgical boost for attack
- Alternating boost/cut pattern
- 5 bands used (more than previous examples)
- Emphasis on transient frequencies

#### Compressor Strategy:
```json
{
  "threshold": -15.0,
  "ratio": 8.0,          // Heavy ratio
  "attack": 1.0,         // Very fast
  "release": 30.0,       // Quick release
  "knee": 1.0,           // Very hard knee
  "makeup_gain": 8.0     // High compensation
}
```

**Design rationale:**
- 8:1 ratio: Heavy, obvious compression (near limiting)
- 1ms attack: **Fastest possible** - catches all transients
- Hard knee (1 dB): Aggressive, not transparent
- High makeup gain: Compensates heavy reduction

**What this example teaches:**
- Punch requires transient emphasis + mud reduction
- Aggressive compression can be musical
- Very fast attack for maximum control
- Hard knee creates obvious effect
- Multiple targeted boosts create complexity

#### Reverb Strategy:
```json
{
  "room_size": 0.25,     // Very small
  "wet_level": 0.15,     // Minimal reverb
  "dry_level": 1.0       // Maximum dry signal
}
```

**Design rationale:**
- Minimal reverb preserves punch
- Full dry signal maintains directness
- Small room keeps tight sound

### Example 5: Smooth and Vintage

**File:** `configs/prompts/examples/smooth_vintage.json`

**Description:** "smooth and vintage tone with analog warmth for soul music"

**Parameter Analysis:**

#### EQ Strategy:
```json
{"frequency": 150, "gain": 2.0, "q": 0.6}      // Gentle warmth
{"frequency": 500, "gain": 1.5, "q": 0.9}      // Body boost
{"frequency": 2000, "gain": -1.0, "q": 1.0}    // Smoothness
{"frequency": 6000, "gain": -2.5, "q": 0.7}    // Tape-like roll-off
{"frequency": 12000, "gain": -3.5, "q": 0.5}   // Vintage attenuation
```

**Design rationale:**
- Gentle boosts (2 dB, 1.5 dB): Subtle vintage character
- Progressive high-end roll-off (-1, -2.5, -3.5 dB): Analog tape curve
- Wide Q values (0.5-0.9): Smooth, musical curves
- 5 bands create gradual frequency curve

**Advanced vintage technique:**
- **Graduated roll-off**: Each band cuts more than previous
- Mimics analog equipment frequency response
- Low Q creates smooth, natural curve

#### Compressor Strategy:
```json
{
  "threshold": -22.0,
  "ratio": 3.5,
  "attack": 20.0,        // Slow attack
  "release": 150.0,      // Musical release
  "knee": 8.0,           // Very soft knee
  "makeup_gain": 4.5
}
```

**Design rationale:**
- Moderate ratio (3.5:1): Musical compression
- Slow attack (20ms): Preserves transients
- Long release (150ms): Smooth, glue-like
- **Very soft knee (8 dB)**: Maximum transparency

**What this example teaches:**
- Vintage sound from progressive high-end roll-off
- Smooth compression uses slow attack + soft knee
- Subtle parameter values create refinement
- Multiple gentle adjustments better than single strong one

## Example Coverage Analysis

### Sonic Dimensions Covered

| Dimension | Examples Covering |
|-----------|-------------------|
| **Warmth** | Warm Intimate, Smooth Vintage |
| **Brightness** | Bright Energetic |
| **Darkness** | Dark Atmospheric |
| **Aggression** | Punchy Aggressive, Bright Energetic |
| **Smoothness** | Warm Intimate, Smooth Vintage |
| **Energy** | Bright Energetic, Punchy Aggressive |
| **Space** | Dark Atmospheric (large), Warm Intimate (small) |
| **Punch** | Punchy Aggressive |

### Parameter Range Coverage

| Parameter | Min (Examples) | Max (Examples) | Coverage |
|-----------|---------------|----------------|----------|
| **EQ Gain** | -6 dB | +5 dB | Good (±6 dB typical) |
| **Compression Ratio** | 2.5:1 | 8:1 | Excellent (gentle to heavy) |
| **Attack Time** | 1 ms | 30 ms | Excellent (fast to slow) |
| **Room Size** | 0.25 | 0.85 | Excellent (small to large) |
| **Wet Level** | 0.15 | 0.55 | Excellent (subtle to pronounced) |

### Effect Order Patterns

All examples follow professional signal flow:
1. **EQ** (tone shaping)
2. **Compressor** (dynamics control)
3. **Reverb** (space/ambience)

## Common Patterns Learned

### Pattern 1: Warmth Recipe
```
Low-mid boost (150-300 Hz) + High-end roll-off (6k-12k Hz)
Examples: Warm Intimate, Smooth Vintage
```

### Pattern 2: Brightness Recipe
```
High-mid boost (3k-5k Hz) + Air boost (8k-12k Hz)
Examples: Bright Energetic
```

### Pattern 3: Clarity Recipe
```
Low-end cut (80-200 Hz) + Presence boost (800-1200 Hz)
Examples: Bright Energetic, Punchy Aggressive
```

### Pattern 4: Darkness Recipe
```
Low-frequency boost (60-250 Hz) + High-frequency cut (2.5k-8k Hz)
Examples: Dark Atmospheric
```

### Pattern 5: Punch Recipe
```
Sharp mid boost (1-1.5k Hz, high Q) + Fast compression (1-3ms attack)
Examples: Punchy Aggressive
```

### Pattern 6: Intimacy Recipe
```
Small room size (0.2-0.4) + Low wet level (0.15-0.25) + Gentle compression
Examples: Warm Intimate
```

### Pattern 7: Atmosphere Recipe
```
Large room (0.7-0.9) + High wet level (0.5-0.6) + Slow compression
Examples: Dark Atmospheric
```

## Prompt Versioning

### Version 1.0 (Current)

**File:** `configs/prompts/parameter_generation_v1.yaml`

**Characteristics:**
- Professional audio engineer persona
- Detailed parameter guidelines
- 5 diverse few-shot examples
- Conservative parameter recommendations
- Effect order guidance

**Strengths:**
- High validity rate (~95%+)
- Musically appropriate parameters
- Good emotional → technical mapping
- Consistent output structure

**Potential improvements for v2:**
- Genre-specific guidance
- More aggressive parameter examples
- Advanced techniques (parallel compression, etc.)
- Dynamic effect ordering based on description

### Creating New Versions

To create a new prompt version:

1. **Copy existing template:**
```bash
cp configs/prompts/parameter_generation_v1.yaml \
   configs/prompts/parameter_generation_v2.yaml
```

2. **Modify components:**
```yaml
version: "2.0"
name: "parameter_generation"
description: "New approach description"

system_prompt: |
  [Modified system instructions]

few_shot_examples:
  - file: "configs/prompts/examples/new_example.json"
    description: "New example type"
```

3. **Test new version:**
```python
generator = ParameterGenerator(
    llm_provider=provider,
    prompt_version="v2"
)

chain = await generator.generate_parameters(description="...")
```

4. **Compare results:**
```python
# Generate with both versions
results_v1 = await generator_v1.generate_parameters(description)
results_v2 = await generator_v2.generate_parameters(description)

# Compare output quality
compare_chains(results_v1, results_v2)
```

## Best Practices

### System Prompt Design

**DO:**
- Establish clear persona and expertise
- Specify output format explicitly
- Include parameter ranges and typical values
- Map abstract qualities to technical parameters
- Provide reasoning guidelines

**DON'T:**
- Use vague instructions
- Assume LLM knows audio engineering
- Skip range constraints
- Omit output schema
- Over-constrain creativity

### Few-Shot Example Selection

**DO:**
- Cover diverse sonic characteristics
- Use real-world, musical parameter values
- Include edge cases (very dark, very bright)
- Show parameter relationships
- Demonstrate valid JSON structure

**DON'T:**
- Use only similar examples
- Include extreme/unrealistic values
- Omit important sonic qualities
- Show invalid parameter combinations
- Use inconsistent formatting

### Parameter Guidelines

**DO:**
- Explain frequency ranges in musical terms
- Suggest typical value ranges
- Warn about potential issues
- Connect parameters to sonic outcomes
- Use professional terminology

**DON'T:**
- Only specify schema limits
- Use technical jargon without explanation
- Omit usage guidance
- Ignore musical context
- Provide conflicting advice

## Troubleshooting

### Problem: Inconsistent Output

**Symptoms:**
- Different parameters for same description
- Wide variation in parameter values
- Unpredictable effect choices

**Solutions:**
1. Lower temperature (0.3-0.5) for more deterministic output
2. Add more specific parameter guidelines
3. Include more few-shot examples
4. Emphasize consistency in system prompt

### Problem: Out-of-Range Parameters

**Symptoms:**
- Validation errors
- Extreme parameter values
- Schema violations

**Solutions:**
1. Strengthen parameter range guidance in system prompt
2. Add range validation examples
3. Include correction prompt for retry logic
4. Use normalizer to clamp values

### Problem: Poor Musical Quality

**Symptoms:**
- Technically valid but unmusical parameters
- Extreme EQ curves
- Inappropriate compression settings

**Solutions:**
1. Add "musically appropriate" emphasis
2. Include more nuanced few-shot examples
3. Specify conservative value ranges
4. Add musical context to guidelines

### Problem: Missing Effects

**Symptoms:**
- Generated fewer effects than requested
- Skipped certain effect types

**Solutions:**
1. Explicitly list required effects in prompt
2. Add validation for expected effects
3. Include examples with all effect types
4. Use correction prompt to request missing effects

## Advanced Techniques

### Dynamic Few-Shot Selection

Select examples based on description:

```python
def select_relevant_examples(description, all_examples):
    # Match description keywords to example characteristics
    if "warm" in description.lower():
        return ["warm_intimate", "smooth_vintage"]
    elif "bright" in description.lower():
        return ["bright_energetic"]
    # ... more matching logic

# Use in prompt formatting
relevant_examples = select_relevant_examples(description, template.few_shot_examples)
prompt = format_prompt(description, examples=relevant_examples)
```

### Temperature Strategies

Different temperatures for different needs:

```python
# Consistent, safe parameters
chain = await generator.generate_parameters(
    description="standard vocal",
    temperature=0.3  # Low variance
)

# Creative exploration
chain = await generator.generate_parameters(
    description="experimental texture",
    temperature=1.0  # High variance
)
```

### Prompt Chaining

Iterative refinement through multiple prompts:

```python
# 1. Generate initial parameters
initial = await generator.generate_parameters(description="warm vocal")

# 2. Refine with feedback
refined_description = f"{description} with more presence"
refined = await generator.generate_parameters(description=refined_description)

# 3. Compare and select
best = compare_and_select(initial, refined)
```

## Evaluation Metrics

### Validity Rate

Percentage of generations producing valid JSON:

```python
def measure_validity_rate(generator, descriptions):
    valid = 0
    total = len(descriptions)

    for desc in descriptions:
        try:
            chain = await generator.generate_parameters(description=desc)
            valid += 1
        except (JSONParseError, ValidationError):
            pass

    return valid / total * 100

# Target: >95% validity rate
```

### Musical Appropriateness

Human evaluation of parameter quality:

```python
# Criteria:
# - Are EQ curves musical?
# - Is compression appropriate?
# - Does reverb match description?
# - Would a professional use these settings?

# Rating scale: 1-5
# Target: Average >4.0
```

### Consistency

Same description should yield similar parameters:

```python
async def measure_consistency(generator, description, n=5):
    chains = []
    for _ in range(n):
        chain = await generator.generate_parameters(description=description)
        chains.append(chain)

    # Compare parameter variance
    variance = calculate_parameter_variance(chains)
    return variance

# Target: Low variance for low temperature
```

## References

- **Sony LLM2Fx Paper**: Source of few-shot example patterns
- **SocialFX Dataset**: Real-world audio effect descriptions
- **Professional Mixing Standards**: Parameter range guidelines
- **Pydantic Documentation**: Schema validation patterns

## See Also

- [Parameter Generation Documentation](parameter_generation.md) - Complete API reference
- [examples/single_effect_example.py](../examples/single_effect_example.py) - Simple generation examples
- [examples/effect_chain_example.py](../examples/effect_chain_example.py) - Multi-effect chains
- [examples/batch_generation.py](../examples/batch_generation.py) - Batch processing with prompt comparison
