from locallm import KoboldcppLm
from locallm.schemas import InferenceParams, LmParams
from tests.localconf import MODEL, CTX


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
