from abc import ABC, abstractmethod
from typing import Optional
from llama_cpp import Llama
from .schemas import InferenceParams, LmParams, OnStartEmitType, OnTokenType


def defaultOnToken(token: str):
    print(token, end="", flush=True)


class LmProvider(ABC):
    llm: Optional[Llama] = None
    models_dir: str = ""
    loaded_model: str = ""
    api_key: str = ""
    server_url: str = ""
    is_verbose = False
    on_token: OnTokenType
    on_start_emit: OnStartEmitType

    @abstractmethod
    def __init__(
        self,
        params: LmParams,
    ) -> None:
        pass

    @abstractmethod
    def load_model(
        self,
        model_name: str,
        ctx: int,
        gpu_layers: Optional[int],
    ) -> None:
        pass

    @abstractmethod
    def infer(
        self,
        prompt: str,
        params: InferenceParams = InferenceParams(),
    ) -> str:
        pass
