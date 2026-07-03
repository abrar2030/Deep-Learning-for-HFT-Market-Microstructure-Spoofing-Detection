"""
Package initialization for utilities.

Re-exports are LAZY (PEP 562): importing this package, or any specific
submodule such as `utils.data_loading`, must never eagerly import sibling
modules with heavy or fragile dependencies (e.g. the interpretability
module imports plotting libraries). Attribute access still works exactly
as before: `from utils import Trainer` resolves on demand.
"""

_EXPORTS = {
    # name -> submodule
    "SpoofingPatternGenerator": "data_generation",
    "AdversarialBacktestFramework": "data_generation",
    "LOBFeatureExtractor": "feature_engineering",
    "SpoofingLabelGenerator": "feature_engineering",
    "IntegratedGradients": "interpretability",
    "SHAPExplainer": "interpretability",
    "ModelExplainer": "interpretability",
    "AttentionVisualizer": "interpretability",
    "LOBDataset": "training",
    "Trainer": "training",
    "FocalLoss": "training",
    "evaluate_model": "training",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib

        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
