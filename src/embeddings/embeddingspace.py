from dataclasses import dataclass
from enum import Enum

@dataclass
class EmbeddingSpace(Enum):
    CLAP = 'laion_clap'