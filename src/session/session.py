from dataclasses import dataclass, field


@dataclass
class PromptRecord:
    prompt: str
    fx_chain: list
    params: dict


class Session:
    """Holds in-memory state across multiple user prompts within a session.

    current_params: {canonical_fx_name: params_dict} — the latest optimized
        params for each FX type seen so far. Used by Parser to determine
        which FX already exist (Case 1/2/3 routing).

    history: list of PromptRecord, one per prompt in order.

    available_fx: list of canonical FX names the user has available in their
        DAW (e.g. ["eq", "comp", "rev", "dist"]).
    """

    def __init__(self, available_fx: list):
        self.available_fx = available_fx
        self.current_params: dict = {}
        self.history: list[PromptRecord] = []

    def fx_exists(self, fx: str) -> bool:
        return fx in self.current_params

    def update(self, prompt: str, fx_chain: list, params: dict) -> None:
        self.history.append(PromptRecord(prompt=prompt, fx_chain=list(fx_chain), params=dict(params)))
        self.current_params.update(params)
