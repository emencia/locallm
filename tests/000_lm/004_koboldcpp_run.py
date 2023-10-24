from locallm import KoboldcppLm
from locallm.schemas import InferenceParams, LmParams

# https://huggingface.co/Aryanne/Mamba-gpt-3B-v4-ggml-and-gguf/resolve/main/q5_1-gguf-mamba-gpt-3B_v4.gguf
MODEL = "q5_1-gguf-mamba-gpt-3B_v4.gguf"
CTX = 2048


def test_load_model_koboldcpp_lm():
    builder = KoboldcppLm(LmParams())
    builder.load_model(MODEL, CTX)
    assert builder.loaded_model == MODEL


def test_infer_koboldcpp_lm():
    builder = KoboldcppLm(LmParams())
    builder.load_model(MODEL, CTX)
    system = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request."
    )
    resp = builder.infer(
        system + "### Instruction:\nhow are you?\n\n### Response:",
        InferenceParams(stream=True, temperature=1),
    )
    assert len(resp["text"]) > 0
