from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict
from llama_cpp import LlamaGrammar
from pydantic import BaseModel


LmProviderType = Literal["local", "koboldcpp", "ollama"]

OnTokenType = Callable[[str], None]

OnStartEmitType = Callable[[Optional[Any]], None]


class InferenceParams(BaseModel):
    """
    Inference parameters for a language model. These parameters control how the
    model generates text.

    Args:
        stream (Optional[bool], optional): Whether to use streaming inference or batch
            inference. Defaults to `None`.
        template (Optional[str], optional): A template string to be used as input to
            the model. Defaults to `None`.
        threads (Optional[int], optional): The number of threads to use during
            inference. Defaults to `None`.
        max_tokens (Optional[int], optional): The maximum number of tokens to generate.
            Defaults to `None`.
        temperature (Optional[float], optional): The temperature parameter, which
            controls the probability distribution over vocabulary. Defaults to `None`.
        top_p (Optional[float], optional): The probability cutoff for top-p sampling.
            Defaults to `None`.
        top_k (Optional[int], optional): The top k most likely words to sample from.
            Defaults to `None`.
        min_p (Optional[float], optional): the minimum probability for a token to be
            considered, relative to the probability of the most likely token.s.
            Defaults to `None`.
        stop (Optional[List[str]], optional): A list of words to stop the model from
            generating. Defaults to `None`.
        frequency_penalty (Optional[float], optional): The frequency penalty for rare
            words. Defaults to `None`.
        presence_penalty (Optional[float], optional): The presence penalty for rare
            words. Defaults to `None`.
        repeat_penalty (Optional[float], optional): The repeat penalty for repeating
            sequences of words. Defaults to `None`.
        tfs (Optional[float], optional): The temperature factor for top-k sampling.
            Defaults to `None`.
        grammar (Optional[LlamaGrammar]): a gbnf grammar. Defaults to `None`.

    Returns:
        None

    Example:
        >>> params = InferenceParams(stream=True, max_tokens=10)
        >>> params.stream  # True
        >>> params.max_tokens  # 10
    """

    stream: Optional[bool] = None
    template: Optional[str] = None
    threads: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    stop: Optional[List[str]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repeat_penalty: Optional[float] = None
    tfs: Optional[float] = None
    grammar: Optional[LlamaGrammar] = None


class LmParams(BaseModel):
    """
    Parameters for the Language Model.

    Args:
        models_dir (Optional[str], optional): The directory containing the language
            models. Defaults to None.
        api_key (Optional[str], optional): The API key for accessing the language
            models. Defaults to None.
        server_url (Optional[str], optional): The URL of the server hosting the language
            models. Defaults to None.
        is_verbose (Optional[bool], optional): Whether to print verbose output. Defaults
            to False.
        on_token (Optional[OnTokenType], optional): A function to call for when a new
            token is received. Defaults to None. If not specified the default function
            will output the token to the terminal
        on_start_emit (Optional[OnStartEmitType], optional): A function to call for
            when the model starts emitting. Defaults to None.

    Example:
        >>> lm_params = LmParams(models_dir="/path/to/models", api_key="my_api_key")
    """

    models_dir: Optional[str] = None
    api_key: Optional[str] = None
    server_url: Optional[str] = None
    is_verbose: Optional[bool] = None
    on_token: Optional[OnTokenType] = None
    on_start_emit: Optional[OnStartEmitType] = None


class InferenceResult(TypedDict):
    """
    Represents the result of an inference process. Contains both the
    text output by the lm and statistics about the inference if available.

    Args:
        text (str): The input text used for the inference.
        stats (Dict[str, Any]): A dictionary containing any statistics generated
            during the inference. These can include things like time taken or total
            tokens. Note that it depends on the backend used

    Example:
        >>> result = InferenceResult(
            text="The quick brown fox jumps over the lazy dog",
            stats={"total_tokens": 31}
        )
        >>> print(result)
        {
            'text': 'The quick brown fox jumps over the lazy dog',
            'stats': {"total_tokens": 31}
        }
    """

    text: str
    stats: Dict[str, Any]
