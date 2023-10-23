from typing import Any, Callable, List, Literal, Optional
from pydantic import BaseModel


LmProviderType = Literal["local", "goinfer", "koboldcpp"]

OnTokenType = Callable[[str], None]

OnStartEmitType = Callable[[Optional[Any]], None]


class InferenceParams(BaseModel):
    """Parameters for inference.

    Args
    ----
    stream : bool, Optional
        Whether to stream the output.
    template : str, Optional
        The template to use for the inference.
    threads : int, Optional
        The number of threads to use for the inference.
    max_tokens : int, Optional
        The maximum number of tokens to generate.
    temperature : float, Optional
        The temperature for the model.
    top_p : float, Optional
        The probability cutoff for the top k tokens.
    stop : List[str], Optional
        A list of words to stop the model from generating.
    frequency_penalty : float, Optional
        The frequency penalty for the model.
    presence_penalty : float, Optional
        The presence penalty for the model.
    repeat_penalty : float, Optional
        The repeat penalty for the model.
    top_k : int, Optional
        The top k tokens to generate.
    tfs : float, Optional
        The temperature for the model.

    Returns
    -------
    None

    Example
    -------
    >>> InferenceParams(stream=True, template="<s>[INST] {prompt} [/INST]")
    {
        "stream": True,
        "template": "<s>[INST] {prompt} [/INST]"
    }
    """

    stream: Optional[bool] = None
    template: Optional[str] = None
    threads: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repeat_penalty: Optional[float] = None
    top_k: Optional[int] = None
    tfs: Optional[float] = None


class LmParams(BaseModel):
    """Parameters for language model.

    Args
    ----
    provider_type : Literal["local", "goinfer", "koboldcpp"]
        The provider type for the language model.
    models_dir : str, Optional
        The directory containing the language model.
    api_key : str, Optional
        The API key for the language model.
    server_url : str, Optional
        The server URL for the language model.
    is_verbose : bool, Optional
        Whether to enable verbose output.
    on_token : Callable[[str], None], Optional
        A callback function to be called on each token generated.
    on_start_emit : Callable[[Optional[Any]], None], Optional
        A callback function to be called on the start of the emission.

    Returns
    -------
    None

    Example
    -------
    >>> LmParams(provider_type="goinfer", models_dir="/home/me/models", \
    api_key="abc123")
    {
        "provider_type": "goinfer",
        "models_dir": "/home/me/models",
        "api_key": "abc123"
    }
    """

    provider_type: LmProviderType
    models_dir: Optional[str] = None
    api_key: Optional[str] = None
    server_url: Optional[str] = None
    is_verbose: Optional[bool] = None
    on_token: Optional[OnTokenType] = None
    on_start_emit: Optional[OnStartEmitType] = None
