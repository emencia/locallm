# Locallm

An api to query local language models using different backends. Supported backends:

- [Llama.cpp Python](https://github.com/abetlen/llama-cpp-python): the local Python bindings for Llama.cpp
- [Kobold.cpp](https://github.com/LostRuins/koboldcpp): the Koboldcpp api server
- [Goinfer](https://github.com/synw/goinfer): the Goinfer api server

## Quickstart

For now clone the repository, a pip package is to come

### Local

```python
from locallm.core import Lm, LmParams, InferenceParams

lm = Lm(
    LmParams(
        provider_type="local",
        models_dir="/home/me/models/dir",
        is_verbose=True,
    )
)
lm.load_model("mistral-7b-instruct-v0.1.Q4_K_M.gguf", 8192)
template = "<s>[INST] {prompt} [/INST]"
lm.infer(
    "list the planets in the solar system",
    InferenceParams(
        template=template,
        temperature=0.2,
        top_p=0.35,
        stream=True,
        max_tokens=512,
    ),
)
```

### Koboldcpp

```python
from locallm.core import Lm, LmParams, InferenceParams

lm = Lm(
    LmParams(
        provider_type="koboldcpp",
        is_verbose=True,
    )
)
lm.load_model("", 8192) # sets the context window size to 8196 tokens
template = "<s>[INST] {prompt} [/INST]"
lm.infer(
    "list the planets in the solar system",
    InferenceParams(
        template=template,
        temperature=0.2,
        top_p=0.35,
        stream=True,
        max_tokens=512,
    ),
)
```

### Goinfer

```python
from locallm.core import Lm, LmParams, InferenceParams

lm = Lm(
    LmParams(
        provider_type="goinfer",
        api_key="7aea109636aefb984b13f9b6927cd174425a1e05ab5f2e3935ddfeb183099465",
    )
)
lm.load_model("mistral-7b-instruct-v0.1.Q4_K_M.gguf", 8192)
template = "<s>[INST] {prompt} [/INST]"
lm.infer(
    "list the planets in the solar system",
    InferenceParams(
        stream=True,
        template=template,
        temperature=0.2,
        top_p=0.35,
    ),
)
```

## Api

## LmProvider

An abstract base class to describe a language model provider.

### Attributes

- **llm** `Optional[Llama]`: the language model.
- **models_dir** `str`: the directory where the models are stored.
- **loaded_model** `str`: the name of the loaded model.
- **api_key** `str`: the API key for the language model.
- **server_url** `str`: the URL of the language model server.
- **is_verbose** `bool`: whether to print more information.
- **on_token** `OnTokenType`: the function to be called when a token is generated. Default: outputs the token to the terminal.
- **on_start_emit** `OnStartEmitType`: the function to be called when the model starts emitting tokens.

### Example

```python
LmProvider(
    LmParams(
        model_name="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        ctx=8192,
        gpu_layers=32
    )
)
```

Methods:

### `__init__`

Constructs all the necessary attributes for the LmProvider object.

#### Parameters

- **params** `LmParams`: the parameters for the language model.

#### Example

```python
LmProvider(
    LmParams(
        model_name="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        ctx=8192,
        gpu_layers=32
    )
)
```

### `load_model`

Loads a language model.

#### Parameters

- **model\_name** `str`: The name of the model to load.
- **ctx** `int`: The context window size for the model.
- **gpu\_layers** `Optional[int]`: The number of layers to offload to the GPU for the model.

#### Example

```python
load_model("my_model.gguf", 2048, 32)
```

### `infer`

Run an inference query.

#### Parameters

- **prompt** `str`: the prompt to generate text from.
- **params** `InferenceParams`: the parameters for the inference query.

#### Returns

- **text** `str`: the generated text.

#### Example

```python
>>> infer("<s>[INST] List the planets in the solar system [/INST>")
The planets in the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.
```

## Types

## InferenceParams

Parameters for inference.

### Args

- **stream** `bool, Optional`: Whether to stream the output.
- **template** `str, Optional`: The template to use for the inference.
- **threads** `int, Optional`: The number of threads to use for the inference.
- **max\_tokens** `int, Optional`: The maximum number of tokens to generate.
- **temperature** `float, Optional`: The temperature for the model.
- **top\_p** `float, Optional`: The probability cutoff for the top k tokens.
- **stop** `List[str], Optional`: A list of words to stop the model from generating.
- **frequency\_penalty** `float, Optional`: The frequency penalty for the model.
- **presence\_penalty** `float, Optional`: The presence penalty for the model.
- **repeat\_penalty** `float, Optional`: The repeat penalty for the model.
- **top\_k** `int, Optional`: The top k tokens to generate.
- **tfs** `float, Optional`: The temperature for the model.

### Example

```python
InferenceParams(stream=True, template="<s>[INST] {prompt} [/INST>")
{
    "stream": True,
    "template": "<s>[INST] {prompt} [/INST>"
}
```

## LmParams

Parameters for language model.

### Args

- **provider\_type** `Literal["local", "goinfer", "koboldcpp"]`: The provider type for the language model.
- **models\_dir** `str, Optional`: The directory containing the language model.
- **api\_key** `str, Optional`: The API key for the language model.
- **server\_url** `str, Optional`: The server URL for the language model.
- **is\_verbose** `bool, Optional`: Whether to enable verbose output.
- **on\_token** `Callable[[str], None], Optional`: A callback function to be called on each token generated.
- **on\_start\_emit** `Callable[[Optional[Any]], None], Optional`: A callback function to be called on the start of the emission.

### Example

```python
LmParams(
    provider_type="goinfer",
    models_dir="/home/me/models",
    api_key="abc123",
)
```