from typing import Dict, Optional
import json
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


class KoboldcppLm(LmProvider):
    ptype: LmProviderType
    loaded_model = ""
    headers: Dict[str, str]
    url: str
    ctx = 2048
    is_verbose = False
    on_token: OnTokenType | None = None
    on_start_emit: OnStartEmitType | None = None

    def __init__(
        self,
        params: LmParams,
    ) -> None:
        """
        Initialize a new instance of the KoboldcppLm class.

        Args:
            params (LmParams): The parameters to use when initializing the instance.

        Raises:
            ValueError: If `params.models_dir` is not provided.

        Example:
            >>> from locallm import KoboldcppLm, LmParams
            >>> lm = KoboldcppLm(LmParams(is_verbose=True))
        """
        self.ptype = "koboldcpp"
        if params.server_url is None:
            self.url = "http://localhost:5001"
            print(
                "No server url provided: using the default local one: "
                "http://localhost:5001"
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
        self.headers = {
            # "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        self.load_model("", 0)

    def load_model(self, model_name: str, ctx: int, gpu_layers: Optional[int] = None):
        """
        Set the model's param from server

        Args:
            model_name (str): The name of the model to be loaded.
            ctx (int): The context window size for the model.
            gpu_layers (Optional[int], optional): The number of GPU layers to use.
                Defaults to None.

        Example:
            >>> from locallm import KoboldcppLm, LmParams
            >>> lm = KoboldcppLm(LmParams(is_verbose=True))
            >>> lm.load_model('my_model.gguf', 2048)
        """
        url = self.url + "/api/extra/true_max_context_length"
        res = requests.get(url)
        data = res.json()
        v = int(data["value"])
        self.ctx = v
        if self.is_verbose is True:
            print("Setting model context window to", v)
        url = self.url + "/api/v1/model"
        res = requests.get(url)
        data = res.json()
        m = data["result"]
        self.loaded_model = m
        if self.is_verbose is True:
            print("Setting model to", m)

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

        Example:
            >>> from locallm import KoboldcppLm, LmParams
            >>> lm = KoboldcppLm(LmParams(is_verbose=True))
            >>> lm.load_model('my_model.gguf', 2048)
            >>> result = lm.infer("What is the capital of France?")
            Paris
            >>> print(result)
            {'text': 'Paris', 'stats': {}}
        """
        tpl = params.template or "{prompt}"
        final_prompt = tpl.replace("{prompt}", prompt)
        if self.is_verbose is True:
            print("Running inference with prompt:")
            print(final_prompt)
        # convert the params to the Kobold api format
        final_params = params.model_dump(exclude_none=True, exclude_unset=True)
        if "stop" in final_params:
            final_params["stop_sequence"] = final_params["stop"]
            del final_params["stop"]
        if "repeat_penalty" in final_params:
            final_params["rep_pen"] = final_params["repeat_penalty"]
            del final_params["repeat_penalty"]
        if "presence_penalty" in final_params:
            del final_params["presence_penalty"]
        if "frequency_penalty" in final_params:
            del final_params["frequency_penalty"]
        if "threads" in final_params:
            del final_params["threads"]
        if "max_tokens" in final_params:
            final_params["max_length"] = final_params["max_tokens"]
            del final_params["max_tokens"]
        if "stream" in final_params:
            del final_params["stream"]
        if self.is_verbose is True:
            print("Inference parameters:")
            print(final_params)
        # run the query
        payload = {
            "prompt": final_prompt,
            "max_context_length": self.ctx,
            **final_params,
        }
        url = self.url + "/api/extra/generate/stream"
        response = requests.post(url, stream=True, headers=self.headers, json=payload)
        client = sseclient.SSEClient(response)  # type: ignore
        buf = []
        i = 0
        for event in client.events():
            # print(event)
            if i == 0:
                if self.on_start_emit:
                    self.on_start_emit(None)
            # print("|begin|", event, "|end|")
            data = json.loads(event.data)
            # print(data)
            if self.on_token:
                self.on_token(data["token"])
            buf.append(data["token"])
            i += 1
        return {"text": "".join(buf), "stats": {}}
