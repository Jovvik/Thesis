from typing import Any


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(*args: Any) -> Any:
        val = dict.get(*args)
        return dotdict(val) if type(val) is dict else val
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
