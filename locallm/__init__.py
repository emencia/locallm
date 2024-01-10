"""An api to query local language models using different backends"""
from importlib.metadata import version
from .provider import LmProvider
from .providers.koboldcpp import KoboldcppLm
from .providers.ollama import OllamaLm
from .providers.local import LocalLm
from .schemas import (
    InferenceParams,
    LmParams,
    LmProviderType,
    OnTokenType,
    OnStartEmitType,
)

__pkgname__ = "locallm"
__version__ = version(__pkgname__)

__all__ = [
    "LmProvider",
    "KoboldcppLm",
    "OllamaLm",
    "LocalLm",
    "InferenceParams",
    "LmParams",
    "LmProviderType",
    "OnTokenType",
    "OnStartEmitType",
]
