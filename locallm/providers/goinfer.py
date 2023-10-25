import json
from typing import Any, Dict, Optional
import sseclient
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


class GoinferLm(LmProvider):
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
        """
        Initialize a new instance of the GoinferLm class.

        Args:
            params (LmParams): The parameters to use when initializing the instance.

        Raises:
            ValueError: If `params.api_key` is not provided.

        Example:
            >>> from locallm import GoinferLm, LmParams
            >>> lm = GoinferLm(LmParams(api_key="5dz78.."))
        """
        self.ptype = "goinfer"
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
        """
        Loads a model on the server

        Args:
            model_name (str): The name of the model to be loaded.
            ctx (int): The context window size for the model.
            gpu_layers (Optional[int], optional): The number of GPU layers to use.
                Defaults to None.

        Example:
            >>> from locallm import GoinferLm, LmParams
            >>> lm = GoinferLm(LmParams(api_key="5dz78.."))
            >>> lm.load_model('my_model.gguf', 2048)
        """
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
    ) -> InferenceResult:
        """
        Run an inference query for a prompt and params

        Args:
            prompt (str): The prompt to use for the inference.
            params (InferenceParams, optional): The inference parameters. Defaults to
                InferenceParams().

        Returns:
            InferenceResult: The result of the inference.

        Raises:
            Exception: If no model is loaded. Use the load_model method first.

        Example:
            >>> from locallm import GoinferLm, LmParams
            >>> lm = GoinferLm(LmParams(api_key="5dz78.."))
            >>> lm.load_model('my_model.gguf', 2048)
            >>> result = lm.infer("What is the capital of France?")
            Paris
            >>> print(result)
            {'text': 'Paris', 'stats': {}}
        """
        tpl = params.template or "{prompt}"
        final_prompt = tpl.replace("{prompt}", prompt)
        if self.is_verbose:
            print("Running inference with prompt:")
            print(final_prompt)
        if self.loaded_model == "":
            raise Exception("No model is loaded: use the load_model method first")
        final_params = params.model_dump(exclude_none=True, exclude_unset=True)
        if "template" in final_params:
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
        return res["data"]
