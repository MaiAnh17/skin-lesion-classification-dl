"""Load existing HAM10000 checkpoints without retraining.

Only load .pt files you created yourself or otherwise explicitly trust.
Older PyTorch lacks torch.serialization.safe_globals and may need pickle.
"""
from collections.abc import Mapping
from pathlib import Path
import warnings

import torch


def _trusted_legacy_load(path):
    warnings.warn(
        "Using legacy torch.load for a trusted local HAM10000 checkpoint. "
        "Untrusted .pt files can execute code when unpickled.",
        RuntimeWarning,
        stacklevel=2,
    )
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_state_dict(path, *, trusted_source=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    try:
        from torch.serialization import safe_globals
    except ImportError:
        safe_globals = None

    if safe_globals is not None:
        try:
            import numpy as np
            try:
                from numpy._core.multiarray import _reconstruct
            except ImportError:
                from numpy.core.multiarray import _reconstruct
            allowed = [
                _reconstruct, np.ndarray, np.dtype,
                type(np.dtype("float32")), type(np.dtype("float64")),
                type(np.dtype("int64")), type(np.dtype("int32")),
            ]
            with safe_globals(allowed):
                obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            if not trusted_source:
                raise RuntimeError(f"Safe checkpoint loading failed: {path}") from exc
            obj = _trusted_legacy_load(path)
    else:
        if not trusted_source:
            raise RuntimeError(
                "This PyTorch version lacks safe_globals. "
                "Upgrade PyTorch to load untrusted checkpoints safely."
            )
        obj = _trusted_legacy_load(path)

    if isinstance(obj, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(obj.get(key), Mapping):
                obj = obj[key]
                break
    if not isinstance(obj, Mapping):
        raise TypeError(f"Expected a state_dict in {path}; got {type(obj).__name__}")

    state = {}
    for key, value in obj.items():
        name = str(key)
        if name.startswith("module."):
            name = name[len("module."):]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Non-tensor value for checkpoint key {name!r}: {type(value).__name__}")
        state[name] = value
    if not state:
        raise ValueError(f"Checkpoint has no model tensors: {path}")
    return state


def load_model(model, path, device):
    state = read_state_dict(path)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Architecture mismatch while loading {path}: {exc}") from exc
    return model.to(device).eval()
