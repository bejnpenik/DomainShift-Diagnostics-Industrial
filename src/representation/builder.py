"""
Processor builder — YAML-driven processor configuration.

Converts a YAML dict or file into a ProcessorConfig that satisfies
the representation.ProcessorConfig protocol.

YAML schema for signal processors:

    # configs/processors/raw_12k.yaml
    type: signal
    name: raw_12k          # optional, auto-generated if missing
    resampling:
      target_sampling_rate: 12000
      max_bandwidth_factor: 0.5      # optional, default 0.5
    segmentation:
      window_duration: 0.05
      window_overlap: 0.2
    view:
      type: raw

    # configs/processors/spec_48k.yaml
    type: signal
    name: spec_48k
    resampling:
      target_sampling_rate: 48000
    segmentation:
      window_duration: 0.05
      window_overlap: 0.2
    view:
      type: stft
      n_fft: 256
      hop_length: 64
      win_length: 256

Entry points:
    build_processor_config(cfg: dict) -> ProcessorConfig
    build_processor_config_from_yaml(path) -> ProcessorConfig
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_processor_config(cfg: dict[str, Any]):
    """Build a ProcessorConfig from a parsed YAML dict.

    Dispatches on cfg["type"]:
        "signal" -> SignalProcessorConfig
        (future: "image", "tabular")

    Returns:
        A ProcessorConfig instance (satisfies the representation protocol).
    """
    proc_type = cfg.get("type")
    if proc_type is None:
        raise ValueError("Processor YAML must have a 'type' field (e.g. 'signal')")

    if proc_type == "signal":
        return _build_signal_config(cfg)
    else:
        raise ValueError(f"Unknown processor type: '{proc_type}'. Expected: signal")


def build_processor_config_from_yaml(path: str | Path):
    """Load a processor YAML file and build a ProcessorConfig."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processor YAML not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    return build_processor_config(cfg)


# ---------------------------------------------------------------------------
# Internal: signal processor config builder
# ---------------------------------------------------------------------------

_VIEW_TYPES = {"raw", "stft"}


def _build_signal_config(cfg: dict[str, Any]):
    """Build a SignalProcessorConfig from a YAML dict.

    Maps YAML structure to the refactored config classes:
        resampling  -> target_sampling_rate, max_bandwidth_factor
        segmentation -> window_duration, window_overlap
        view         -> RawViewConfig | STFTViewConfig
    """
    from .signal.config import (
        SignalProcessorConfig,
        RawViewConfig,
        STFTViewConfig,
    )

    # --- name ---
    resampling = cfg.get("resampling", {})
    view_cfg = cfg.get("view", {})
    view_type = view_cfg.get("type", "raw")

    name = cfg.get("name")
    if name is None:
        sr = resampling.get("target_sampling_rate", "?")
        name = f"{view_type}_{sr}"

    # --- resampling ---
    target_sampling_rate = resampling.get("target_sampling_rate", 12000)
    max_bandwidth_factor = resampling.get("max_bandwidth_factor", 0.5)

    # --- segmentation ---
    seg = cfg.get("segmentation", {})
    window_duration = seg.get("window_duration", 0.05)
    window_overlap = seg.get("window_overlap", 0.5)

    # --- view ---
    if view_type == "raw":
        view = RawViewConfig()
    elif view_type in ("stft"):
        view = STFTViewConfig(
            n_fft=view_cfg.get("n_fft", 256),
            hop_length=view_cfg.get("hop_length", 128),
            win_length=view_cfg.get("win_length", 256),
        )
    else:
        raise ValueError(
            f"Unknown view type: '{view_type}'. Expected one of: {_VIEW_TYPES}"
        )

    return SignalProcessorConfig(
        name=name,
        target_sampling_rate=target_sampling_rate,
        window_duration=window_duration,
        window_overlap=window_overlap,
        view=view,
        max_signal_bandwidth_factor=max_bandwidth_factor,
    )


# ---------------------------------------------------------------------------
# Validation helper (for study loader)
# ---------------------------------------------------------------------------

def validate_processor_yaml(path: str | Path) -> dict[str, Any]:
    """Load and validate a processor YAML, returning the raw dict.

    Useful for the study loader to inspect processor type without
    fully constructing the config object.
    """
    import yaml

    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "type" not in cfg:
        raise ValueError(f"Processor YAML {path} missing 'type' field")
    if cfg["type"] not in ("signal",):
        raise ValueError(f"Processor YAML {path} has unknown type: {cfg['type']}")

    view = cfg.get("view", {})
    view_type = view.get("type", "raw")
    if view_type not in _VIEW_TYPES:
        raise ValueError(
            f"Processor YAML {path} has unknown view type: {view_type}"
        )

    return cfg