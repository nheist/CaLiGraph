from enum import Enum, auto


class Match(Enum):
    EXACT = auto()
    PARTIAL = auto()
    NONE = auto()
    MISSING = auto()
