from typing import TypeVar, Union

import numpy as np
import pprint

T = TypeVar("T")
RecDict = Union[dict[str, "RecDict[T]"], T]


def format_ndarray_dict(obj: RecDict[np.ndarray]) -> RecDict[str]:
    if type(obj) == dict:
        return {key: format_ndarray_dict(value) for key, value in obj.items()}
    elif type(obj) == np.ndarray:
        return str(obj.shape)
    else:
        raise RuntimeError("Not Reachable.")


def debug_ndarray_dict(obj: RecDict[np.ndarray]):
    pprint.pprint(format_ndarray_dict(obj))
