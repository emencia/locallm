from typing import Any, Iterator, List, Optional
from pathlib import Path
from llama_cpp import CompletionChunk, Llama
from ..schemas import (
    InferenceParams,
    InferenceResult,
    LmParams,
    OnTokenType,
    OnStartEmitType,
    LmProviderType,
)
from ..provider import LmProvider, defaultOnToken


class LocalLm(LmProvider):
    ptype: LmProviderType
    llm: Llama | None = None
    models_dir = ""
    loaded_model = ""
    ctx = 2048
    is_verbose = False
    on_token: OnTokenType | None = None
    on_start_emit: OnStartEmitType | None = None

    def __init__(
        self,
        params: LmParams,
    ) -> None:
        """
        Initialize a new instance of the LocalLm class.

        Args:
            params (LmParams): The parameters to use when initializing the instance.

        Raises:
            ValueError: If `params.models_dir` is not provided.

        Example:
            >>> from locallm import LocalLm, LmParams
            >>> lm = LocalLm(LmParams(model_path='/absolute/path/to/models'))
        """
        self.ptype = "local"
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

    def load_model(self, model_name: str, ctx: int, gpu_layers: Optional[int] = None):
        """
        Loads a model in memory

        Args:
            model_name (str): The name of the model to be loaded.
            ctx (int): The context window size for the model.
            gpu_layers (Optional[int], optional): The number of GPU layers to use.
                Defaults to None.

        Example:
            >>> from locallm import LocalLm, LmParams
            >>> lm = LocalLm(LmParams(model_path='/absolute/path/to/models'))
            >>> lm.load_model('my_model.gguf', 2048)
        """
        if self.is_verbose is True:
            print("Loading model", self.models_dir, model_name)
        p = Path(self.models_dir) / model_name
        if self.loaded_model != model_name:
            self.llm = Llama(
                model_path=str(p),
                n_ctx=ctx,
            )
        self.loaded_model = model_name
        self.ctx = ctx

    def generate(
        self,
        prompt: str,
        params: InferenceParams = InferenceParams(),
    ) -> Iterator[CompletionChunk]:
        """
        Run an inference query for a prompt and params and return an iterator

        Args:
            prompt (str): The prompt to use for the inference.
            params (InferenceParams, optional): The inference parameters. Defaults to
                InferenceParams().

        Returns:
            Iterator[CompletionChunk]: The stream iterator

        Raises:
            Exception: If no model is loaded. Use the load_model method first.

        Example:
            >>> from locallm import LocalLm
            >>> lm = LocalLm(model_path='/absolute/path/to/models')
            >>> lm.load_model('my_model.gguf', 2048)
            >>> stream = lm.infer("What is the capital of France?")
            >>> for (line in stream):
            >>>     # process the line
        """
        params.stream = True
        res: Iterator[CompletionChunk] = self._infer(  # type: ignore
            prompt, params, True
        )
        return res

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
            >>> from locallm import LocalLm
            >>> lm = LocalLm(model_path='/absolute/path/to/models')
            >>> lm.load_model('my_model.gguf', 2048)
            >>> result = lm.infer("What is the capital of France?")
            >>> print(result)
            {'text': 'Paris', 'stats': {}}
        """
        res: InferenceResult = self._infer(prompt, params)  # type: ignore
        return res

    def _infer(
        self,
        prompt: str,
        params: InferenceParams = InferenceParams(),
        return_stream=False,
    ) -> InferenceResult | Iterator[Any]:
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
        if return_stream is True:
            s: Iterator[CompletionChunk] = stream  # type: ignore
            return s
        buf: List[str] = []
        i = 0
        if params.stream is True:
            for output in stream:
                if i == 0:
                    if self.on_start_emit:
                        self.on_start_emit(None)
                # print("OUT", output)
                txt = ""
                try:
                    txt = output["choices"][0]["text"]  # type: ignore
                except Exception:
                    pass
                if self.on_token is not None:
                    print("T", txt)
                    self.on_token(txt)
                buf.append(txt)
                i += 1
            text = "".join(buf)
        else:
            text = stream["choices"][0]["text"]  # type: ignore
        return {"text": text, "stats": {}}
