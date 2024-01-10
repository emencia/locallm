# flake8: noqa: E501
from typing import Any, Optional
import sys
from locallm import InferenceParams, LmParams, LocalLm

# > cd examples
# > python -m local /home/me/models
# the argument is the path to the models directory
# to get the model used in this example:
# wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf


def on_start_emit(data: Optional[Any]):
    print("Start emitting")


def main(models_dir):
    lm = LocalLm(
        LmParams(
            models_dir=models_dir,
            is_verbose=True,
            on_start_emit=on_start_emit,
        )
    )
    lm.load_model("mistral-7b-instruct-v0.1.Q4_K_M.gguf", 8192)
    # the template is for Mistral 7B: change it accordingly to you model requirements
    template = "<s>[INST] {prompt} [/INST]"
    lm.infer(
        "list the planets in the solar system and their distance from the sun "
        "in percentage. Answer in json in this format:\n\n"
        '```json\n{"name": distance}\n```',
        InferenceParams(
            template=template,
            temperature=0.2,
            top_p=0.35,
            stream=True,
            max_tokens=512,
        ),
    )


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise Exception("Provide a models directory path")
    main(sys.argv[1])
