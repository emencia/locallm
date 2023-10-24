import json
from typing import Dict, Optional
import requests

from ..schemas import (
    InferenceParams,
    InferenceResult,
    LmParams,
    OnTokenType,
    OnStartEmitType,
    LmProviderType,
)
from ..provider import LmProvider, defaultOnToken


class OllamaLm(LmProvider):
    ptype: LmProviderType
    models_dir = ""
    loaded_model = ""
    headers: Dict[str, str]
    url: str
    is_verbose = False
    on_token: OnTokenType | None = None
    on_start_emit: OnStartEmitType | None = None

    def __init__(
        self,
        params: LmParams,
    ) -> None:
        self.ptype = "ollama"
        if params.server_url is None:
            self.url = "http://127.0.0.1:11434"
            print(
                "No server url provided: using the default local one: "
                "http://127.0.0.1:11434"
            )
        else:
            self.url = params.server_url
        if params.is_verbose is True:
            self.is_verbose = True
        if params.on_token:
            self.on_token = params.on_token
        else:
            self.on_token = defaultOnToken
        if params.on_start_emit:
            self.on_start_emit = params.on_start_emit
        # print("Initializing lm", model_path)
        self.headers = {
            "Accept": "text/event-stream",
        }

    def load_model(self, model_name: str, ctx: int, gpu_layers: Optional[int] = None):
        if self.is_verbose is True:
            print("Setting model context window to", ctx)
        self.ctx = ctx
        self.loaded_model = model_name

    def infer(
        self,
        prompt: str,
        params: InferenceParams = InferenceParams(),
    ) -> InferenceResult:
        tpl = params.template or "{prompt}"
        final_prompt = tpl.replace("{prompt}", prompt)
        if self.is_verbose:
            print("Running inference with prompt:")
            print(final_prompt)
        if self.loaded_model == "":
            raise Exception("No model is loaded: use the load_model method first")
        final_params = params.model_dump(exclude_none=True, exclude_unset=True)
        final_params["model"] = self.loaded_model
        final_params["num_ctx"] = self.ctx
        if "template" in final_params:
            del final_params["template"]
        if "threads" in final_params:
            final_params["num_threads"] = final_params["threads"]
            del final_params["threads"]
        if "stop" in final_params:
            final_params["stop_sequence"] = ",".join(final_params["stop"])
            del final_params["stop"]
        if "tfs" in final_params:
            final_params["tfs_z"] = final_params["tfs"]
            del final_params["tfs"]
        if "max_tokens" in final_params:
            final_params["num_predict"] = final_params["max_tokens"]
            del final_params["max_tokens"]
        if self.is_verbose:
            print("Inference parameters:")
            print(final_params)
        payload = {
            "prompt": final_prompt,
            **final_params,
        }
        url = self.url + "/api/generate"
        response = requests.post(url, stream=True, headers=self.headers, json=payload)
        text = ""
        res = {}
        for line in response.iter_lines():
            body = json.loads(line)
            token = body.get("response", "")
            if self.on_token:
                self.on_token(token)
            text = text + token
            if "error" in body:
                raise Exception(body["error"])  # type: ignore
            if body.get("done", False):
                res = body
                del res["done"]
                del res["context"]
                del res["model"]
                del res["created_at"]
        return {"text": text, "stats": res}
