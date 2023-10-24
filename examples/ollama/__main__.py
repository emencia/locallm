# flake8: noqa: E501
from typing import Any
from locallm import OllamaLm, InferenceParams, LmParams

# 1. get and install the model
# > wget https://huggingface.co/Aryanne/Mamba-gpt-3B-v4-ggml-and-gguf/resolve/main/q5_1-gguf-mamba-gpt-3B_v4.gguf
# > ollama serve
# 2. in another terminal create a Modelfile with this content:
# FROM /home/me/models/dir/q5_1-gguf-mamba-gpt-3B_v4.gguf
# create the model in Ollama:
# > ollama create q5_1-gguf-mamba-gpt-3B_v4.gguf -f Modelfile
# 3. run the example below
# > cd examples
# > python -m ollama


def on_token(t: str):
    print(t)


def on_start_emit(data: Any | None = None):
    print("Start emitting")
    print(data)


def main():
    lm = OllamaLm(
        LmParams(
            on_start_emit=on_start_emit,
            is_verbose=True,
        )
    )
    lm.load_model("q5_1-gguf-mamba-gpt-3B_v4.gguf", 2048)
    system = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request."
    )
    template = system + "\n\n### Instructions:\n{prompt}\n\n### Response:"
    res = lm.infer(
        "list the planets in the solar system",
        InferenceParams(
            stream=True,
            template=template,
            temperature=0.2,
            top_p=0.35,
        ),
    )
    print("\nStats:", res["stats"])


if __name__ == "__main__":
    main()
