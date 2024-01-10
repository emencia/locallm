from locallm import LocalLm, KoboldcppLm, OllamaLm
from locallm.schemas import LmParams
from tests.localconf import MODELS_DIR


def test_init_local_lm():
    builder = LocalLm(LmParams(models_dir=MODELS_DIR))
    assert builder.models_dir == MODELS_DIR
    assert builder.ptype == "local"
    assert builder.loaded_model == ""


def test_init_koboldccp():
    builder = KoboldcppLm(LmParams())
    assert builder.ptype == "koboldcpp"
    assert builder.loaded_model == ""


def test_init_ollama():
    builder = OllamaLm(LmParams())
    assert builder.ptype == "ollama"
    assert builder.loaded_model == ""
