from typing import Optional
from .providers.koboldcpp import Koboldcpp
from .providers.local import LocalLm
from .providers.goinfer import Goinfer
from .schemas import LmParams, InferenceParams
from .provider import LmProvider


class Lm(LmProvider):
    provider: LmProvider

    def __init__(
        self,
        params: LmParams,
    ):
        if params.provider_type == "local":
            self.provider = LocalLm(params)
        elif params.provider_type == "goinfer":
            self.provider = Goinfer(params)
        elif params.provider_type == "koboldcpp":
            self.provider = Koboldcpp(params)

    def load_model(
        self, model_name: str, ctx: int, gpu_layers: Optional[int] = None
    ) -> None:
        self.provider.load_model(model_name, ctx, gpu_layers)

    def infer(
        self,
        prompt: str,
        params: InferenceParams = InferenceParams(),
    ) -> str:
        return self.provider.infer(prompt, params)
