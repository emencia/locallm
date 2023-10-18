# flake8: noqa: E501
from typing import Any
from locallm import Lm, InferenceParams, LmParams


# run this script from the root of the repository: run a Goinfer local server then:
# python -m examples.goinfer
# to get the model used in this example:
# wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf


def on_token(t: str):
    print(t)


def on_start_emit(data: Any | None = None):
    print("Start emitting")
    print(data)


def main():
    lm = Lm(
        LmParams(
            provider_type="goinfer",
            api_key="7aea109636aefb984b13f9b6927cd174425a1e05ab5f2e3935ddfeb183099465",
            # on_token=on_token,
            on_start_emit=on_start_emit,
        )
    )
    lm.load_model("mistral-7b-instruct-v0.1.Q4_K_M.gguf", 8192)
    # the template is for Mistral 7B: change it accordingly to you model requirements
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


if __name__ == "__main__":
    main()
