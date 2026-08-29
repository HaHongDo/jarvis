from enum import Enum, auto


class AssistantState(Enum):
    LISTENING = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()
    SPEAKING = auto()
