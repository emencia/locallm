# LocalLm autodoc example

This example generates a docstring for a Python code block

## Install

Install the Koboldcpp backend (Linux): 

```bash
git clone https://github.com/LostRuins/koboldcpp
cd koboldcpp
make
```

Get the quantitized language model:

```bash
mkdir models
cd models
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

Run the backend server:

```bash
python koboldcpp.py --contextsize 8192 --model ~/dev/lm/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf --smartcontext
```

## Run

Clone the repository, create a venv and:

```bash
pip install -r requirements
cd examples
python -m autodoc requests get
```

Parameters:

- `module` **str**: the python path to a module: ex: `python -m autodoc mymodule.mysubmodule`
- `node` **str**: the name of a code node: ex: `python -m autodoc mymodule.mysubmodule MyClass`

Example:

```bash
python -m autodoc django.contrib.auth.models AbstractBaseUser
```