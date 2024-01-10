import json
from typing import Dict, Iterator, Optional, Any
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
    ctx = 2048
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
        Initialize a new instance of the OllamaLm class.

        Args:
            params (LmParams): The parameters to use when initializing the instance.

        Raises:
            ValueError: If `params.models_dir` is not provided.

        Example:
            >>> from locallm import OllamaLm, LmParams
            >>> lm = OllamaLm(LmParams(is_verbose=True))
        """
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
        """
        Set a model and it's context window size. The model must have
        been registered in Ollama before

        Args:
            model_name (str): The name of the model to be loaded.
            ctx (int): The context window size for the model.
            gpu_layers (Optional[int], optional): The number of GPU layers to use.
                Defaults to None.

        Example:
            >>> from locallm import OllamaLm, LmParams
            >>> lm = OllamaLm(LmParams(is_verbose=True))
            >>> lm.load_model('my_model.gguf', 2048)
        """
        if self.is_verbose is True:
            print("Setting model context window to", ctx)
        self.ctx = ctx
        self.loaded_model = model_name

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
            >>> from locallm import OllamaLm, LmParams
            >>> lm = OllamaLm(LmParams(is_verbose=True))
            >>> lm.load_model('my_model', 2048)
            >>> result = lm.infer("What is the capital of France?")
            >>> print(result)
            {'text': 'Paris', 'stats': {}}
        """
        res: InferenceResult = self._infer(prompt, params)  # type: ignore
        return res

    def generate(
        self,
        prompt: str,
        params: InferenceParams = InferenceParams(),
    ) -> Iterator[Any]:
        """
        Run an inference query for a prompt and params and return a stream

        Args:
            prompt (str): The prompt to use for the inference.
            params (InferenceParams, optional): The inference parameters. Defaults to
                InferenceParams().

        Returns:
            InferenceResult: The result of the inference.

        Raises:
            Exception: If no model is loaded. Use the load_model method first.

        Example:
            >>> from locallm import OllamaLm, LmParams
            >>> lm = OllamaLm(LmParams(is_verbose=True))
            >>> lm.load_model('my_model', 2048)
            >>> stream = lm.infer("What is the capital of France?")
            >>> for (line in stream):
            >>>     # process the line
        """
        params.stream = True
        res: Iterator[Any] = self._infer(prompt, params, True)  # type: ignore
        return res

    def _infer(
        self,
        prompt: str,
        params: InferenceParams = InferenceParams(),
        return_stream=False,
    ) -> InferenceResult | Iterator[Any]:
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
        if return_stream is True:
            s: Iterator[Any] = response.iter_lines()  # type: ignore
            return s
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
