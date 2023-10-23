import json
from typing import Any, Dict, Optional
import sseclient
import requests

from ..schemas import InferenceParams, LmParams, OnTokenType, OnStartEmitType
from ..provider import LmProvider, defaultOnToken


class Goinfer(LmProvider):
    models_dir = ""
    loaded_model = ""
    headers: Dict[str, str]
    url: str
    is_verbose = False
    on_token: OnTokenType
    on_start_emit: OnStartEmitType

    def __init__(
        self,
        params: LmParams,
    ) -> None:
        if params.server_url is None:
            self.url = "http://localhost:5143"
            print(
                "No server url provided: using the default local one: "
                "http://localhost:5143"
            )
        else:
            self.url = params.server_url
        if params.api_key is None:
            raise ValueError("Provide an api_key parameter")
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
            "Authorization": f"Bearer {params.api_key}",
            "Accept": "text/event-stream",
        }

    def load_model(self, model_name: str, ctx: int, gpu_layers: Optional[int] = None):
        if self.is_verbose:
            print("Loading model", self.models_dir, model_name)
        url = self.url + "/model/load"
        payload = {"name": model_name, "ctx": ctx}
        if gpu_layers:
            payload["gpu_layers"] = gpu_layers
        response = requests.post(url, headers=self.headers, json=payload)
        if response.ok:
            self.loaded_model = model_name
            return
        else:
            raise Exception(
                f"Error loading model: {response.status_code} {response.text}"
            )

    def infer(
        self,
        prompt: str,
        params: InferenceParams = InferenceParams(),
    ) -> str:
        tpl = params.template or "{prompt}"
        final_prompt = tpl.replace("{prompt}", prompt)
        if self.is_verbose:
            print("Running inference with prompt:")
            print(final_prompt)
        if self.loaded_model == "":
            raise Exception("No model is loaded: use the load_model method first")
        final_params = params.model_dump(exclude_none=True, exclude_unset=True)
        del final_params["template"]
        if "tfs" in final_params:
            final_params["tfs_z"] = final_params["tfs"]
            del final_params["tfs"]
        if self.is_verbose:
            print("Inference parameters:")
            print(final_params)
        payload = {
            "prompt": prompt,
            "template": tpl,
            **final_params,
        }
        url = self.url + "/completion"
        response = requests.post(url, stream=True, headers=self.headers, json=payload)
        client = sseclient.SSEClient(response)  # type: ignore
        res: Dict[str, Any] = {}
        for event in client.events():
            data = json.loads(event.data)
            if data["msg_type"] == "token":
                if self.on_token:
                    # print("T", data["content"])
                    self.on_token(data["content"])
            elif data["msg_type"] == "system":
                if data["content"] == "result":
                    res = data
                elif data["content"] == "start_emitting":
                    if self.on_start_emit:
                        self.on_start_emit(data["data"])
                    if self.is_verbose:
                        print("Thinking time:", data["data"]["thinking_time_format"])
                    # print("SYSTEM:", data, "\n")
        return res["data"]["text"]
