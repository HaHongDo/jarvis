from .base import WakeWordDetector

__all__ = ["WakeWordDetector"]

try:
    from .openwakeword import OpenWakeWordDetector

    __all__.append("OpenWakeWordDetector")
except ImportError:
    pass
