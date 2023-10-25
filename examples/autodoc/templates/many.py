# flake8: noqa: E501
TEMPLATE = """<s>[INST] in Python create detailed and helpful Google style docstrings for this code:

```python
def add(a: float, b: float = 2.5) -> float:
    if a < 0:
        raise ValueError("Provide a positive number for a")
    try:
        return a + b
    except Exception as e:
        raise (e)
```

Important: always provide a short example in each docstring. Break lines at 86 characters maximum [/INST]

```python
def add(a: float, b: float = 2.5) -> float:
    \"\"\"
    Sums two float numbers, but ensures the first number is non-negative. If the
    second number is not provided, it defaults to 2.5.

    Args:
        a (float): The first number to be added. Must be a non-negative float.
        b (float, optional): The second number to be added. Defaults to 2.5. Can be
            any float.

    Returns:
        float: The sum of a and b.

    Raises:
        ValueError: If `a` is negative.
        Exception: Any unexpected error that might occur during addition.

    Example:
        >>> add(2.5)
        5.0
        >>> add(2.5, 3.5)
        6.0
        >>> add(-1.0)
        ValueError: Provide a positive number for a
    \"\"\"
    if a < 0:
        raise ValueError("Provide a positive number for a")
    try:
        return a + b
    except Exception as e:
        raise (e)
```</s>
[INST] in Python create detailed and helpful Google style docstrings for this code:

```python
{prompt}
```

Important: always provide a short example in each docstring. Break lines at 86 characters maximum [/INST]"""
