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

_SYSTEM_PROMPT = """You are an expert audio engineer. Given a sound descriptor, select the appropriate audio effects (FX) and decide their order in the signal chain.

Rules:
- Call each tool in the ORDER you want effects applied to the audio signal (first call = first in chain).
- Call each tool AT MOST ONCE.
- Only select effects that are relevant to achieving the described sound.
- You may select zero or more effects."""


class FXSelectorAgent:
    """Layer 1 agent: selects and orders the FX chain via LLM tool calls.

    Each available FX becomes a tool. The agent calls them in the order
    it wants them applied, so the sequence of tool calls IS the FX chain.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def select(self, instruction: str, available_fx: list) -> list:
        """Return ordered list of canonical FX names for the given instruction.

        Args:
            instruction: single word or short phrase (e.g. "warm", "bright")
            available_fx: list of canonical names the user has (e.g. ["eq", "comp", "rev"])

        Returns:
            ordered list of canonical FX names (e.g. ["eq", "rev"])
        """
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
            if fx in FX_DESCRIPTIONS
        ]

        if not tools:
            return []

        response = self.llm.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Select effects for: {instruction}"},
            ],
            tools=tools,
            tool_choice="auto",
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
