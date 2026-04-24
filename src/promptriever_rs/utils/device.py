from __future__ import annotations


def resolve_device(torch_module, requested_device: str | None) -> str:
    requested = (requested_device or "auto").strip().lower()

    available = {
        "cuda": bool(torch_module.cuda.is_available()),
        "mps": bool(getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available()),
        "cpu": True,
    }

    if requested == "auto":
        if available["cuda"]:
            return "cuda"
        if available["mps"]:
            return "mps"
        return "cpu"

    if requested not in available:
        raise ValueError(
            f"Unsupported device '{requested}'. Expected one of: auto, cuda, mps, cpu."
        )

    if not available[requested]:
        raise ValueError(
            f"Requested device '{requested}' is not available in the current environment."
        )

    return requested
