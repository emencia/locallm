# Locallm

An api to query local language models using different backends. Supported backends:

- [Llama.cpp Python](https://github.com/abetlen/llama-cpp-python): the local Python bindings for Llama.cpp
- [Kobold.cpp](https://github.com/LostRuins/koboldcpp): the Koboldcpp api server
- [Goinfer](https://github.com/synw/goinfer): the Goinfer api server

## Examples

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

## Koboldcpp

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

## Goinfer

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