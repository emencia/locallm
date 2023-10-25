from prompt_toolkit import prompt
from locallm import InferenceParams, LmParams, KoboldcppLm


def infer(_prompt: str):
    lm = KoboldcppLm(
        LmParams(
            is_verbose=True,
        )
    )
    lm.load_model("", 4096)  # sets the context window size to 4096 tokens
    # the template is for Mistral 7B: change it accordingly to you model requirements
    template = "<s>[INST] {prompt} [/INST]"
    lm.infer(
        _prompt,
        InferenceParams(
            template=template,
            temperature=0.2,
            top_p=0.35,
            stream=True,
            max_tokens=512,
        ),
    )


def run():
    print("Enter your prompt. Use Alt+Enter to submit it.")
    text = prompt("> ", multiline=True)
    infer(text)


if __name__ == "__main__":
    run()
