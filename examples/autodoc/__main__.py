# flake8: noqa: E501
import inspect
import sys
from importlib import import_module
from locallm import InferenceParams, LmParams, KoboldcppLm
from .templates.many import TEMPLATE as TEMPLATE_MANY
from .templates.one import TEMPLATE as TEMPLATE_ONE


# this example generates docstrings for Python code blocks or modules
# first run a Koboldcpp server with Mistral 7B instruct model
#  To generate docstrings for a whole module provide the module path as argument:
# > cd examples
# > python -m autodoc locallm.schemas
#  To generate one docstring for a code node provide as argument the module path and the node name:
# > cd examples
# > python -m autodoc locallm.schema InferenceParams

# to get the model used in this example:
# wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf


def infer(_prompt: str, template: str):
    lm = KoboldcppLm(LmParams())
    lm.load_model("", 8192)  # sets the context window size to 8192 tokens
    lm.infer(
        _prompt,
        InferenceParams(
            template=template,
            temperature=0,
            top_p=0.35,
            stream=True,
            max_tokens=4096,
        ),
    )


def run(rawpath: str, rawnode: str | None = None):
    if rawnode is None:
        nodes = rawpath.split(".")
        node = nodes.pop()
        path = ".".join(nodes)
        template = TEMPLATE_MANY
    else:
        path = rawpath
        node = rawnode
        template = TEMPLATE_ONE
    source = inspect.getsource(getattr(import_module(path), node))
    s = "" if rawnode else "s"
    print(f"----- Creating docstring{s} for source ----")
    print(source)
    print("-----------------------------------------")
    return
    # return
    infer(source, template)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise Exception("Provide a module path")
    print(sys.argv)
    if len(sys.argv) == 3:
        run(sys.argv[1], sys.argv[2])
    else:
        run(sys.argv[1])
