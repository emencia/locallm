"""
Specific application exceptions.
"""


class LocallmBaseException(Exception):
    """
    Exception base.

    You should never use it directly except for test purpose. Instead make or
    use a dedicated exception related to the error context.
    """
    pass


class AppOperationError(LocallmBaseException):
    """
    Sample exception to raise from your code.
    """
    pass
