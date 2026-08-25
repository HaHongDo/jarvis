from enum import Enum, auto


class AssistantState(Enum):
    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()
    SPEAKING = auto()
