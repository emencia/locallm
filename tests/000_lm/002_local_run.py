from locallm import LocalLm
from locallm.schemas import InferenceParams, LmParams
from tests.localconf import MODELS_DIR, MODEL, CTX


def test_load_model_local_lm():
    builder = LocalLm(LmParams(models_dir=MODELS_DIR))
    builder.load_model(MODEL, CTX)
    assert builder.loaded_model == MODEL


def test_infer_local_lm():
    builder = LocalLm(LmParams(models_dir=MODELS_DIR))
    builder.load_model(MODEL, CTX)
    system = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request."
    )
    resp = builder.infer(
        system + "### Instruction:\nHello\n\n### Response", InferenceParams(stream=True)
    )
    assert len(resp["text"]) > 0
