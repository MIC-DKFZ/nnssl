from typing import Any
from dataclasses import dataclass, asdict

def make_serializable(to_serialize: Any):
    """Converts a dataclass to a dictionary, and recursively converts all items"""

    if isinstance(to_serialize, dict):
        return {k: make_serializable(v) for k, v in to_serialize.items()}
    elif isinstance(to_serialize, (list, tuple)):
        return [make_serializable(v) for v in to_serialize]
    elif isinstance(to_serialize, set):
        return {make_serializable(v) for v in to_serialize}
    elif isinstance(to_serialize, dataclass):
        return make_serializable(asdict(to_serialize))
    else:
        return to_serialize

