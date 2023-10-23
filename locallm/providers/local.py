from typing import List, Optional
from pathlib import Path
from llama_cpp import Llama
from ..schemas import InferenceParams, LmParams, OnTokenType, OnStartEmitType
from ..provider import LmProvider, defaultOnToken


class LocalLm(LmProvider):
    llm: Llama | None = None
    models_dir = ""
    loaded_model = ""
    is_verbose = False
    on_token: OnTokenType
    on_start_emit: OnStartEmitType

    def __init__(
        self,
        params: LmParams,
    ) -> None:
        if params.models_dir is None:
            raise ValueError("Provide a models_dir parameter")
        # print("Initializing lm", model_path)
        self.models_dir = params.models_dir
        if params.is_verbose is True:
            self.is_verbose = True
        if params.on_token:
            self.on_token = params.on_token
        else:
            self.on_token = defaultOnToken
        if params.on_start_emit:
            self.on_start_emit = params.on_start_emit

    def load_model(self, model_name: str, ctx: int, gpu_layers: Optional[int]):
        if self.is_verbose is True:
            print("Loading model", self.models_dir, model_name)
        p = Path(self.models_dir) / model_name
        if self.loaded_model != model_name:
            self.llm = Llama(
                model_path=str(p),
                n_ctx=ctx,
            )
        self.loaded_model = model_name

    def infer(
        self,
        prompt: str,
        params: InferenceParams = InferenceParams(),
    ) -> str:
        tpl = params.template or "{prompt}"
        final_prompt = tpl.replace("{prompt}", prompt)
        if self.is_verbose is True:
            print("Running inference with prompt:")
            print(final_prompt)

        if self.llm is None:
            raise Exception("No model is loaded: use the load_model method first")
        final_params = params.model_dump(exclude_none=True, exclude_unset=True)
        if "threads" in final_params:
            del final_params["threads"]
        if "template" in final_params:
            del final_params["template"]
        if "tfs" in final_params:
            final_params["tfs_z"] = final_params["tfs"]
            del final_params["tfs"]
        if self.is_verbose is True:
            print("Inference parameters:")
            print(final_params)
        stream = self.llm.create_completion(
            final_prompt,
            **final_params,
        )
        buf: List[str] = []
        i = 0
        for output in stream:
            if i == 0:
                if self.on_start_emit:
                    self.on_start_emit(None)
            txt = output["choices"][0]["text"]  # type: ignore
            if self.on_token:
                self.on_token(txt)
            buf.append(txt)
            i += 1
        text = "".join(buf)
        return text
