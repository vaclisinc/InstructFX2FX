from dataclasses import dataclass

@dataclass
class Prompt:
    text: str = ""
    sys_prompt: str = ""
