from locallm import LocalLm, GoinferLm, KoboldcppLm, OllamaLm
from locallm.schemas import LmParams

MODELS_DIR = "/home/ggg/dev/lm/models"
API_KEY = "7aea109636aefb984b13f9b6927cd174425a1e05ab5f2e3935ddfeb183099465"


def test_init_local_lm():
    builder = LocalLm(LmParams(models_dir=MODELS_DIR))
    assert builder.models_dir == MODELS_DIR
    assert builder.ptype == "local"
    assert builder.loaded_model == ""


def test_init_goinfer():
    builder = GoinferLm(LmParams(api_key=API_KEY))
    assert builder.ptype == "goinfer"
    assert builder.loaded_model == ""


def test_init_koboldccp():
    builder = KoboldcppLm(LmParams())
    assert builder.ptype == "koboldcpp"
    assert builder.loaded_model == ""


def test_init_ollama():
    builder = OllamaLm(LmParams())
    assert builder.ptype == "ollama"
    assert builder.loaded_model == ""
