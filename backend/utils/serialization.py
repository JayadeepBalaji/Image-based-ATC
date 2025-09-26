from typing import Any
import numpy as np


def to_py(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types.

    - np.generic -> Python scalar via .item()
    - np.ndarray -> list
    - dict -> convert values
    - list/tuple/set -> convert elements (returns list for sets/tuples)
    Other objects returned as-is.
    """
    # Numpy scalars (e.g., np.int64, np.float32, np.bool_)
    if isinstance(obj, np.generic):
        return obj.item()

    # Numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Dicts
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}

    # Iterables
    if isinstance(obj, (list, tuple, set)):
        return [to_py(v) for v in obj]

    return obj
