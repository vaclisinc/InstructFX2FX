from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Prompt:
    instruction: str = ""
    sys_prompt: str = ""

    def set_instruction(self, instruction):
        self.instruction = instruction

# ---------------------------------------------------------------------------
# Per-effect text blocks
# ---------------------------------------------------------------------------

_EFFECT_DESCRIPTIONS_DASP = {
    "eq": """\
6-BAND PARAMETRIC EQ (18 parameters)

Each EQ band has THREE parameters:
- frequency (Hz): center frequency of the band
- gain_db (dB): boost or cut applied at the center frequency
- Q (unitless): bandwidth control (higher = narrower)

Parameters:
1. b1_freq   – Band 1 center frequency (Hz)
2. b1_gain   – Band 1 gain (dB)
3. b1_q      – Band 1 Q factor
4. b2_freq   – Band 2 center frequency (Hz)
5. b2_gain   – Band 2 gain (dB)
6. b2_q      – Band 2 Q factor
7. b3_freq   – Band 3 center frequency (Hz)
8. b3_gain   – Band 3 gain (dB)
9. b3_q      – Band 3 Q factor
10. b4_freq  – Band 4 center frequency (Hz)
11. b4_gain  – Band 4 gain (dB)
12. b4_q     – Band 4 Q factor
13. b5_freq  – Band 5 center frequency (Hz)
14. b5_gain  – Band 5 gain (dB)
15. b5_q     – Band 5 Q factor
16. b6_freq  – Band 6 center frequency (Hz)
17. b6_gain  – Band 6 gain (dB)
18. b6_q     – Band 6 Q factor""",

    "compressor": """\
COMPRESSOR (6 parameters)

The compressor is a feed-forward RMS compressor.

Parameters:
1. threshold_db   – Level above which compression starts (dB)
2. ratio          – Compression ratio (e.g., 4 = 4:1)
3. attack         – Attack time (seconds)
4. release        – Release time (seconds)
5. makeup_gain_db – Output gain applied after compression (dB)
6. mix            – Dry/wet blend (0 = dry, 1 = fully compressed)""",

    "reverb": """\
REVERB (25 parameters)

The reverb is an algorithmic reverb with early reflections, late reverb,
modulation, filtering, and output control.

Early reflections:
1.  early_gain        – Level of early reflections
2.  early_delay       – Delay before early reflections (seconds)
3.  early_diffusion   – Diffusion of early reflections
4.  early_width       – Stereo width of early reflections
5.  early_lowcut      – Low-frequency cutoff for early reflections (Hz)
6.  early_highcut     – High-frequency cutoff for early reflections (Hz)
7.  early_mix         – Mix level of early reflections

Late reverb:
8.  late_gain         – Level of late reverb tail
9.  decay_time        – Reverb decay time (seconds)
10. late_diffusion    – Diffusion of late reverb
11. density           – Echo density of the reverb tail
12. mod_rate          – Modulation rate (Hz)
13. mod_depth         – Modulation depth
14. late_lowcut       – Low-frequency cutoff for late reverb (Hz)
15. late_highcut      – High-frequency cutoff for late reverb (Hz)
16. late_width        – Stereo width of late reverb
17. late_mix          – Mix level of late reverb

Global / output:
18. pre_delay         – Delay before reverb onset (seconds)
19. damping           – High-frequency damping
20. lowcut            – Global low-frequency cutoff (Hz)
21. highcut           – Global high-frequency cutoff (Hz)
22. wet               – Wet signal level
23. dry               – Dry signal level
24. width             – Overall stereo width
25. mix               – Global dry/wet mix""",
}

_EFFECT_DESCRIPTIONS_PEDALBOARD = {
    "compressor": """\
COMPRESSOR

{
  "threshold_db": float in [-60.0, 0.0], step 1.0,
  "ratio": float in [1.0, 20.0], step 0.5 (log scale),
  "attack_ms": float in [0.1, 100.0], step 0.1 (log scale),
  "release_ms": float in [10.0, 1000.0], step 1.0 (log scale)
}""",

    "eq": """\
EQ

{
  "mode": one of ["pass-pass", "pass-shelf", "shelf-pass", "shelf-shelf"],
  "low_cut": float in [50, 500], step 10 (log scale),
  "high_cut": float in [8000, 16000], step 100 (log scale),
  "q": float in [0.1, 10.0], step 0.1,
  "gains": {
    "low_shelf": float in [-20.0, 20.0], step 0.2,
    "high_shelf": float in [-20.0, 20.0], step 0.2,
    "peak1": float in [-20.0, 20.0], step 0.2,
    "peak2": float in [-20.0, 20.0], step 0.2,
    "peak3": float in [-20.0, 20.0], step 0.2
  },
  "peak1_freq": float in [100.0, 500.0], step 10 (log scale),
  "peak2_freq": float in [500.0, 4000.0], step 100 (log scale),
  "peak3_freq": float in [4000.0, 12000.0], step 1000 (log scale)
}

If no gain adjustment is implied by the prompt, "gains" MUST be an empty object {}.""",

    "distortion": """\
DISTORTION

{
  "drive_db": float in [0.0, 15.0], step 0.1
}""",

    "reverb": """\
REVERB

{
  "room_size": float in [0.0, 1.0], step 0.05,
  "damping": float in [0.0, 1.0], step 0.05,
  "wet_level": float in [0.0, 1.0], step 0.01
}""",

    "delay": """\
DELAY

{
  "delay": float in [0.0, 0.05], step 0.01
}""",

    "pitchshift": """\
PITCHSHIFT

{
  "semitones": integer in [-12, 12]
}""",

    "bitcrush": """\
BITCRUSH

{
  "bit_depth": integer in [0, 16]
}""",
}

_EFFECT_OUTPUT_BLOCKS_DASP = {
    "eq": '''\
  "EQ": {
    "b1_freq": ..., "b1_gain": ..., "b1_q": ...,
    "b2_freq": ..., "b2_gain": ..., "b2_q": ...,
    "b3_freq": ..., "b3_gain": ..., "b3_q": ...,
    "b4_freq": ..., "b4_gain": ..., "b4_q": ...,
    "b5_freq": ..., "b5_gain": ..., "b5_q": ...,
    "b6_freq": ..., "b6_gain": ..., "b6_q": ...
  }''',

    "compressor": '''\
  "Compressor": {
    "threshold_db": ..., "ratio": ..., "attack": ...,
    "release": ..., "makeup_gain_db": ..., "mix": ...
  }''',

    "reverb": '''\
  "Reverb": {
    "early_gain": ..., "early_delay": ..., "early_diffusion": ...,
    "early_width": ..., "early_lowcut": ..., "early_highcut": ..., "early_mix": ...,
    "late_gain": ..., "decay_time": ..., "late_diffusion": ...,
    "density": ..., "mod_rate": ..., "mod_depth": ...,
    "late_lowcut": ..., "late_highcut": ..., "late_width": ..., "late_mix": ...,
    "pre_delay": ..., "damping": ..., "lowcut": ..., "highcut": ...,
    "wet": ..., "dry": ..., "width": ..., "mix": ...
  }''',
}

_EFFECT_OUTPUT_BLOCKS_PEDALBOARD = {
    "compressor": '''\
  "Compressor": {
    "threshold_db": ...,
    "ratio": ...,
    "attack_ms": ...,
    "release_ms": ...
  }''',

    "eq": '''\
  "EQ": {
    "mode": ...,
    "low_cut": ...,
    "high_cut": ...,
    "q": ...,
    "gains": {
      "low_shelf": ..., "high_shelf": ..., "peak1": ..., "peak2": ..., "peak3": ...
    },
    "peak1_freq": ...,
    "peak2_freq": ...,
    "peak3_freq": ...
  }''',

    "distortion": '''\
  "Distortion": {
    "drive_db": ...
  }''',

    "reverb": '''\
  "Reverb": {
    "room_size": ...,
    "damping": ...,
    "wet_level": ...
  }''',

    "delay": '''\
  "Delay": {
    "delay": ...
  }''',

    "pitchshift": '''\
  "PitchShift": {
    "semitones": ...
  }''',

    "bitcrush": '''\
  "Bitcrush": {
    "bit_depth": ...
  }''',
}

# Dict keys (as used in LLM output JSON) for each effect name
_EFFECT_DICT_KEYS = {
    "eq": "EQ",
    "compressor": "Compressor",
    "reverb": "Reverb",
}

_EFFECT_DICT_KEYS_PEDALBOARD = {
  "compressor": "Compressor",
  "eq": "EQ",
  "distortion": "Distortion",
  "reverb": "Reverb",
  "delay": "Delay",
  "pitchshift": "PitchShift",
  "bitcrush": "Bitcrush",
}


def _build_dasp_sys_prompt(effects: List[str], task_description: str,
                           current_parameters_dict=None) -> str:
    """Build a DASP sys_prompt that covers exactly the effects in `effects`.

    Args:
        effects: ordered list of effect names, e.g. ["eq"] or ["eq", "compressor", "reverb"]
        task_description: one-sentence description of the task (init vs. refine)
        current_parameters_dict: if provided, appended for refinement prompts
    """
    unknown = [e for e in effects if e not in _EFFECT_DESCRIPTIONS_DASP]
    if unknown:
        raise ValueError(f"Unknown effects: {unknown}. Valid: {list(_EFFECT_DESCRIPTIONS_DASP)}")

    n = len(effects)
    n_word = {1: "ONE", 2: "TWO", 3: "THREE"}.get(n, str(n))
    obj_word = "object" if n == 1 else "objects"

    # Chain description
    chain_lines = "\n".join(
        f"{i+1}. {_EFFECT_DICT_KEYS[e]} " +
        f"({_EFFECT_DESCRIPTIONS_DASP[e].split(chr(10))[0]})"
        for i, e in enumerate(effects)
    )
    chain_desc = f"{n_word} effect{'s' if n != 1 else ''} in the following exact order:\n{chain_lines}"

    # Parameter sections
    param_sections = "\n\n".join(
        f"{i+1}) {_EFFECT_DESCRIPTIONS_DASP[e]}"
        for i, e in enumerate(effects)
    )

    # Output format
    key_list = ", ".join(f'"{_EFFECT_DICT_KEYS[e]}"' for e in effects)
    output_blocks = ",\n".join(_EFFECT_OUTPUT_BLOCKS_DASP[e] for e in effects)
    output_format = (
        f"The output MUST be a JSON object with EXACTLY {n} key{'s' if n != 1 else ''} "
        f"({key_list}):\n\nconfig = {{\n{output_blocks}\n}}"
    )

    refine_suffix = (
        "\n\nReturn ONLY the JSON object. No surrounding text. "
        f"The current parameter values are:\n{current_parameters_dict}\n"
        "Adjust these parameters based on the following instruction:"
        if current_parameters_dict is not None
        else "\n\nReturn ONLY the JSON object. No surrounding text."
    )

    return f"""You are an audio effects parameter generator for a differentiable DSP FX chain.

The FX chain consists of {chain_desc}

{task_description}

You MUST output a single JSON object containing EXACTLY {n} {obj_word}, each representing one effect in the FX chain, with their respective numeric parameters.
Do NOT include explanations, comments, prose, or markdown in your output.
Do NOT omit any parameter.
Do NOT add extra parameters.
All values must be real numbers within reasonable audio ranges.

-------------------------
PARAMETER DEFINITIONS
-------------------------

{param_sections}

-------------------------
OUTPUT FORMAT
-------------------------

Return a JSON object where each effect type is a KEY and its parameters are the VALUE (a nested dictionary).

{output_format}
{refine_suffix}
"""


def _build_pedalboard_sys_prompt(effects: List[str], task_description: str) -> str:
  """Build a Pedalboard sys_prompt that covers exactly the effects in `effects`."""
  unknown = [e for e in effects if e not in _EFFECT_DESCRIPTIONS_PEDALBOARD]
  if unknown:
    raise ValueError(
      f"Unknown effects: {unknown}. Valid: {list(_EFFECT_DESCRIPTIONS_PEDALBOARD)}"
    )

  n = len(effects)
  n_word = {1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE", 6: "SIX"}.get(n, str(n))
  obj_word = "object" if n == 1 else "objects"
  key_list = ", ".join(f'"{_EFFECT_DICT_KEYS_PEDALBOARD[e]}"' for e in effects)

  chain_lines = "\n".join(
    f"{index + 1}. {_EFFECT_DICT_KEYS_PEDALBOARD[effect]} "
    f"({_EFFECT_DESCRIPTIONS_PEDALBOARD[effect].split(chr(10))[0]})"
    for index, effect in enumerate(effects)
  )
  chain_desc = f"{n_word} effect{'s' if n != 1 else ''} in the following exact order:\n{chain_lines}"

  param_sections = "\n\n".join(
    f"{index + 1}) {_EFFECT_DESCRIPTIONS_PEDALBOARD[effect]}"
    for index, effect in enumerate(effects)
  )

  output_blocks = ",\n".join(_EFFECT_OUTPUT_BLOCKS_PEDALBOARD[effect] for effect in effects)
  output_format = (
    f"The output MUST be a JSON object with EXACTLY {n} key{'s' if n != 1 else ''} "
    f"({key_list}):\n\nconfig = {{\n{output_blocks}\n}}"
  )

  return f"""You are an expert audio engineer and music producer specializing in sound design and audio signal processing.

The FX chain consists of {chain_desc}

{task_description}

You must output ONLY a valid JSON object containing EXACTLY {n} {obj_word}, one per effect in the FX chain.
Do not add, remove, or rename fields.
Do not include comments, explanations, prose, or markdown.
Do not include a \"type\" field.

-------------------------
PARAMETER DEFINITIONS
-------------------------

{param_sections}

-------------------------
OUTPUT FORMAT
-------------------------

Return a JSON object where each effect type is a KEY and its parameters are the VALUE (a nested dictionary).

{output_format}

-------------------------
INTERPRETATION RULES
-------------------------

- Initialize parameters to musically reasonable values inferred from the prompt.
- Avoid extreme values unless explicitly implied.
- Prefer neutral or reversible settings when the prompt is vague.
- Common mappings:
  - \"warm\", \"dark\" -> lower high_cut, negative high_shelf, higher damping
  - \"bright\", \"airy\" -> higher high_cut, positive high_shelf
  - \"tight\", \"dry\", \"close\" -> low reverb wet_level and room_size
  - \"ambient\", \"distant\", \"huge\" -> larger room_size and wet_level
  - \"gritty\", \"distorted\" -> increased drive_db, reduced bit_depth
  - \"clean\" -> zero distortion, full bit depth, minimal processing

-------------------------
FAILURE CONDITIONS
-------------------------

The response is invalid if:
- The output is not a JSON object
- The keys are not exactly: {key_list}
- Any required field is missing from any effect
- Any \"type\" field is present
- Any extra field is present
- Any value violates range or resolution constraints
- Any text appears outside the JSON object

Return ONLY the JSON object. No surrounding text."""


class PromptFactory:
    @staticmethod
    def create_prompt(instruction: str, sys_prompt: str = "") -> Prompt:
        return Prompt(instruction=instruction, sys_prompt=sys_prompt)

    @staticmethod
    def LLM_PARAMETER_INITIALIZATION_PROMPT_DASP(fx_chain, instruction, effects: List[str] = None):
        if effects is None:
            effects = ["eq", "reverb"] # No compressor at the moment
        sys_prompt = _build_dasp_sys_prompt(
            effects=effects,
            task_description="Your task is to generate a COMPLETE set of parameters for this FX chain.",
        )
        return Prompt(sys_prompt=sys_prompt, instruction=instruction)

    @staticmethod
    def LLM_PARAMETER_REFINEMENT_PROMPT_DASP(fx_chain, instruction, current_parameters_dict,
                                              effects: List[str] = None):
        if effects is None:
            effects = ["eq", "reverb"] # No compressor at the moment
        sys_prompt = _build_dasp_sys_prompt(
            effects=effects,
            task_description="Your task is to tweak an existing set of parameters for the FX chain based on an instruction.",
            current_parameters_dict=current_parameters_dict,
        )
        return Prompt(sys_prompt=sys_prompt, instruction=instruction)


    @staticmethod
    def LLM_PARAMETER_INITIALIZATION_PROMPT_PEDALBOARD(
        fx_chain=None,
        instruction: str = "",
        current_parameters_dict=None,
        effects: List[str] = None,
    ):
        if effects is None:
            effects = ["distortion", "delay", "pitchshift", "bitcrush"]
        sys_prompt = _build_pedalboard_sys_prompt(
            effects=effects,
            task_description="Your task is to generate a COMPLETE set of parameters for this FX chain.",
        )
        return Prompt(sys_prompt=sys_prompt, instruction=instruction)
