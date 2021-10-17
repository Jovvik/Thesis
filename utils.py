from math import log
from typing import Any, Iterable

import numpy as np


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(*args: Any) -> Any:
        val = dict.get(*args)
        return dotdict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def make_range(min: int, max: int, kind: str, step: int = 1) -> Iterable[int]:
    if kind == 'linear':
        return range(min, max + 1, step)
    elif kind == 'exponential':
        return np.logspace(log(min) / log(step),
                           log(max) / log(step),
                           num=int(log(max / min) / log(step)) + 1,
                           base=step).astype(int)
    else:
        raise NotImplementedError(kind)
