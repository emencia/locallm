# flake8: noqa: E501
from typing import Any, Optional
from locallm import InferenceParams, LmParams, KoboldcppLm

# run a Koboldcpp server with Mistral 7B instruct then:
# > cd examples
# > python -m koboldcpp
# to get the model used in this example:
# wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf


def on_start_emit(data: Optional[Any]):
    print("Start emitting")


def main():
    lm = KoboldcppLm(
        LmParams(
            is_verbose=True,
            # on_start_emit=on_start_emit,
        )
    )
    # the template is for Mistral 7B: change it accordingly to you model requirements
    template = "<s>[INST] {prompt} [/INST]"
    lm.infer(
        "list the planets in the solar system and their distance from the sun. "
        "Answer in json",
        InferenceParams(
            template=template,
            temperature=0.2,
            top_p=0.35,
            stream=True,
            max_tokens=512,
        ),
    )


if __name__ == "__main__":
    main()
