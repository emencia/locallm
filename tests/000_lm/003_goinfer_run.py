from locallm import GoinferLm
from locallm.schemas import InferenceParams, LmParams

API_KEY = "7aea109636aefb984b13f9b6927cd174425a1e05ab5f2e3935ddfeb183099465"
# https://huggingface.co/Aryanne/Mamba-gpt-3B-v4-ggml-and-gguf/resolve/main/q5_1-gguf-mamba-gpt-3B_v4.gguf
MODEL = "q5_1-gguf-mamba-gpt-3B_v4.gguf"
CTX = 2048


def test_load_model_goinfer_lm():
    builder = GoinferLm(LmParams(api_key=API_KEY))
    builder.load_model(MODEL, CTX)
    assert builder.loaded_model == MODEL


def test_infer_goinfer_lm():
    builder = GoinferLm(LmParams(api_key=API_KEY))
    builder.load_model(MODEL, CTX)
    system = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request."
    )
    resp = builder.infer(
        system + "### Instruction:\nHello\n\n### Response", InferenceParams(stream=True)
    )
    assert len(resp["text"]) > 0
