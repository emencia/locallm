from locallm import GoinferLm
from locallm.schemas import InferenceParams, LmParams
from tests.localconf import API_KEY, MODEL, CTX


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
