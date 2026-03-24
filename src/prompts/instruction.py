from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict

@dataclass
class InstructionSet(ABC):
    """Base class for instruction sets."""
    task: str = ""
    instruction: str = ""
    text_anchor: str = ""
    text_target: str = ""


class InstructionSet1(InstructionSet):
    """Standard instruction set for audio transformation tasks."""
    task: str = ""
    instruction: str = ""
    text_anchor: str = ""
    text_target: str = ""

    def __init__(self, anchor: str = "", target: str = "", context: str = ""):
        self.task = f"Make this sound more {target}."
        self.text_anchor = f"This sound is {anchor}" if anchor != "" else None
        self.text_target = f"This sound is {target}"
        if anchor:
            self.instruction = f"This is {context}, but the sound is {anchor}. {self.task}"
        else:
            self.instruction = f"This is {context}. {self.task}"

    def to_dict(self) -> Dict[str, str]:
        return {
            "task": self.task,
            "instruction": self.instruction,
            "text_anchor": self.text_anchor,
            "text_target": self.text_target,
        }