from abc import ABC, abstractmethod
from typing import Optional, Iterator, Any
from llama_cpp import Llama
from .schemas import (
    InferenceParams,
    InferenceResult,
    LmParams,
    OnStartEmitType,
    OnTokenType,
    LmProviderType,
)


def defaultOnToken(token: str):
    print(token, end="", flush=True)


class LmProvider(ABC):
    """
    An abstract base class to describe a language model provider.

    ...

    Attributes
    ----------
    llm : Optional[Llama]
        The language model.
    models_dir : str
        The directory where the models are stored.
    loaded_model : str
        The name of the loaded model.
    api_key : str
        The API key for the language model.
    server_url : str
        The URL of the language model server.
    is_verbose : bool
        Whether to print more information.
    on_token : OnTokenType
        The function to be called when a token is generated. Default:
        outputs the token to the terminal
    on_start_emit : OnStartEmitType
        The function to be called when the model starts emitting tokens.

    Example
    -------
    >>> LmProvider(LmParams(model_name="my_model.gguf", ctx=2048, gpu_layers=32))
    <LmProvider>
    """

    ptype: LmProviderType
    llm: Optional[Llama] = None
    models_dir: str = ""
    loaded_model: str = ""
    api_key: str = ""
    server_url: str = ""
    is_verbose = False
    on_token: OnTokenType | None = None
    on_start_emit: OnStartEmitType | None = None

    @abstractmethod
    def __init__(
        self,
        params: LmParams,
    ) -> None:
        """
        Constructs all the necessary attributes for the LmProvider object.

        Parameters
        ----------
            params : LmParams
                The parameters for the language model.

        Example
        -------
        >>> LmProvider(LmParams(model_name="my_model.gguf", ctx=2048, gpu_layers=32))
        <LmProvider>
        """
        pass

    @abstractmethod
    def load_model(
        self,
        model_name: str,
        ctx: int,
        gpu_layers: Optional[int],
    ) -> None:
        """
        Loads a language model.

        Parameters
        ----------
            model_name : str
                The name of the model to load.
            ctx : int
                The context window size for the model.
            gpu_layers : Optional[int]
                The number of layers to offload to the GPU for the model.

        Example
        -------
        >>> load_model("my_model.gguf", 2048, 32)
        """
        pass

    @abstractmethod
    def infer(
        self,
        prompt: str,
        params: InferenceParams = InferenceParams(),
    ) -> InferenceResult:
        """
        Run an inference query.

        Parameters
        ----------
        prompt : str
            The prompt to generate text from.
        params : InferenceParams
            The parameters for the inference query.

        Returns
        -------
        result : InferenceResult
            The generated text and the stats if any

        Example
        -------
        >>> infer("<s>[INST] List the planets in the solar system [/INST]")
        {
            "text": "The planets ...",
            "stats": {}, // depends on the provider
        }
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        params: InferenceParams = InferenceParams(),
    ) -> Iterator[Any]:
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
            >>> stream = lm.infer("What is the capital of France?")
            >>> for (line in stream):
            >>>     # process the line
        """
        pass
