# MARLIN: de novo structure elucidation from MS/MS without a molecular formula (see README.md).

__all__ = [
    "MarlinDecoder",
    "MassShellSampler",
]


def __getattr__(name):  # lazy imports keep `import marlin` light
    if name == "MarlinDecoder":
        from marlin.model import MarlinDecoder

        return MarlinDecoder
    if name == "MassShellSampler":
        from marlin.sampler import MassShellSampler

        return MassShellSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
