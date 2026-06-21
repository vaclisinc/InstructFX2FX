from llms.llmclient import LLMClient

# Human-readable descriptions used as tool descriptions for the LLM agent.
FX_DESCRIPTIONS = {
    "eq":        "Parametric equalizer — shapes frequency content, controls brightness, warmth, presence, harshness",
    "comp":      "Compressor — controls dynamic range, adds punch, sustain, and glue",
    "rev":       "Reverb — adds room ambience, space, depth, and air",
    "dist":      "Distortion — adds harmonic saturation, grit, and overdrive",
    "delay":     "Delay — creates echoes and rhythmic repetition",
    "pitchshift":"Pitch shifter — changes the pitch of the audio",
    "bitcrush":  "Bitcrusher — lo-fi digital degradation, adds noise and aliasing",
    # "panner":    "Panner — controls stereo placement and width",
}

_MODEL = "openai/gpt-5.5"

# Used on the FIRST turn of a session (no history yet). The agent's only job is
# to map a fresh descriptor to an FX chain — no notion of "refinement" or
# "existing chain" should leak in, since nothing exists yet.
_SYSTEM_PROMPT_INITIAL = """You are an expert audio engineer. Given a sound descriptor, select the appropriate audio effects (FX) and decide their order in the signal chain.

Rules:
- Call each tool in the ORDER you want effects applied to the audio signal (first call = first in chain).
- Call each tool AT MOST ONCE.
- Only select effects that are necessary to achieve the described sound — do not add effects "just in case." Prefer the smallest chain that gets the job done."""

# Used on later turns when there is prior history. The agent decides what to
# ADD on top of the existing chain, or returns no tool calls so the existing
# chain gets re-tuned for the new descriptor.
_SYSTEM_PROMPT_REFINEMENT = """You are an expert audio engineer. Given a sound descriptor and the existing effect chain from prior turns in the session, decide which effect(s), if any, to ADD to the chain.

Rules:
- Call each tool in the ORDER you want any new effects applied (first call = first in chain among the new effects).
- Call each tool AT MOST ONCE, and do NOT call a tool for an effect that is already in the existing chain.
- If the existing chain already contains the right effects for the new request, return NO tool calls — the existing effects will be re-tuned (parameters re-optimized) for the new descriptor. For example, if the chain already has distortion and the user says "too harsh, soften it", the right move is usually to re-tune the existing distortion (return no tools), not to stack an EQ on top."""


def _format_history(history: list) -> str:
    """Render prior PromptRecords into a short human-readable trace.

    Each PromptRecord.fx_chain is the full accumulated chain at that turn,
    so we just show "prompt → chain became [...]" per turn.
    """
    if not history:
        return ""
    lines = ["Prior turns in this session:"]
    for i, record in enumerate(history, 1):
        lines.append(f'  Turn {i}: "{record.prompt}" → chain became {record.fx_chain}')
    return "\n".join(lines)


class FXSelectorAgent:
    """Layer 1 agent: selects and orders the FX chain via LLM tool calls.

    Each available FX becomes a tool. The agent calls them in the order
    it wants them applied, so the sequence of tool calls IS the FX chain.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def select(self, instruction: str, available_fx: list, history: list = None, model: str | None = None) -> list:
        """Return ordered list of canonical FX names for the given instruction.

        Args:
            instruction: single word or short phrase (e.g. "warm", "bright")
            available_fx: list of canonical names the user has (e.g. ["eq", "comp", "rev"])
            history: list of PromptRecord from session.history — prior (prompt, fx_chain)
                pairs so the agent can decide whether the existing chain already
                covers the request.

        Returns:
            ordered list of canonical FX names to ADD this turn (may be empty)
        """
        history = history or []
        existing_chain = history[-1].fx_chain if history else []

        tools = [
            {
                "type": "function",
                "function": {
                    "name": f"add_{fx}",
                    "description": FX_DESCRIPTIONS.get(fx, f"Add {fx} effect"),
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for fx in available_fx
            if fx in FX_DESCRIPTIONS and fx not in existing_chain
        ]

        if not tools:
            return []

        is_refinement = bool(history)
        if is_refinement:
            system_prompt = _SYSTEM_PROMPT_REFINEMENT
            user_msg = (
                f"{_format_history(history)}\n\n"
                f"Current chain: {existing_chain}\n"
                f"New request: {instruction}\n\n"
                "Decide what effect(s) to ADD. If the existing chain already has the right "
                "effects for this request, return no tool calls — they will be re-tuned."
            )
            tool_choice = "auto"
        else:
            system_prompt = _SYSTEM_PROMPT_INITIAL
            user_msg = f"Select effects for: {instruction}"
            tool_choice = "required"

        response = self.llm.llm.chat.completions.create(
            model=model or _MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            tools=tools,
            tool_choice=tool_choice,
        )

        message = response.choices[0].message
        if not message.tool_calls:
            return []

        fx_chain = []
        for tc in message.tool_calls:
            fx_name = tc.function.name.removeprefix("add_")
            if fx_name not in fx_chain:
                fx_chain.append(fx_name)

        return fx_chain
