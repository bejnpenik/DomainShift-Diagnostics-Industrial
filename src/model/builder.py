"""
YAML-driven model builder.

Constructs encoder → aggregator → head models from declarative configs.
Supports both 1D and 2D architectures through a shared schema.

YAML format:
    name: cnn1d_small
    type: 1d

    encoder:
      - [conv, 1, 2, {k: 7, s: 1}]
      - [pool, {k: 2, s: 2}]
      - [dropout, 0.1]
      - [conv, 2, 4, {k: 5, s: 1}]
      - [pool, {k: 2, s: 2}]

    aggregator:
      type: adaptive        # adaptive | multihead
      levels: 16            # int for adaptive, [fine, balanced, coarse] for multihead

    head:
      depth: 2
      dropout: 0.1
      act: relu
"""

from __future__ import annotations

from pathlib import Path
from math import sqrt
from typing import Any

import yaml
import torch
import torch.nn as nn

from model.modules import (
    Conv1D, Conv2D, Pool1D, Pool2D,
    AdaptivePool1D, AdaptivePool2D,
    MultiHeadPool1D, MultiHeadPool2D,
    MLP,
)

# =====================================================================
# Module maps
# =====================================================================

_MODULE_MAP = {
    "1d": {
        "conv": Conv1D,
        "pool": Pool1D,
        "adaptive": AdaptivePool1D,
        "multihead": MultiHeadPool1D,
        "dropout": nn.Dropout,
    },
    "2d": {
        "conv": Conv2D,
        "pool": Pool2D,
        "adaptive": AdaptivePool2D,
        "multihead": MultiHeadPool2D,
        "dropout": nn.Dropout2d,
    },
}

_ACT_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
    "none": None,
}


# =====================================================================
# Parsing helpers
# =====================================================================

def _parse_act(name: str | None) -> nn.Module | None:
    """Resolve activation name to module instance."""
    if name is None or name == "none":
        return None
    if name not in _ACT_MAP:
        raise ValueError(f"Unknown activation: {name}. Options: {list(_ACT_MAP.keys())}")
    cls = _ACT_MAP[name]
    return cls() if cls is not None else None


def _coerce_tuple(val: Any) -> int | tuple:
    """Convert YAML list to tuple, leave int as-is."""
    if isinstance(val, list):
        return tuple(val)
    return val


# =====================================================================
# Encoder builder
# =====================================================================

def _build_encoder(layers: list[list], dim_type: str) -> tuple[nn.Sequential, int]:
    """Build encoder from layer list.

    Args:
        layers: List of layer specs from YAML.
        dim_type: '1d' or '2d'.

    Returns:
        (nn.Sequential, output_channels) where output_channels is the
        channel count after the last conv layer.
    """
    modules = _MODULE_MAP[dim_type]
    built = []
    output_channels = 0

    for spec in layers:
        layer_type = spec[0]

        if layer_type == "conv":
            in_ch, out_ch = spec[1], spec[2]
            params = spec[3] if len(spec) > 3 else {}
            built.append(modules["conv"](
                input=in_ch,
                output=out_ch,
                k=_coerce_tuple(params.get("k", 3)),
                s=_coerce_tuple(params.get("s", 1)),
                p=_coerce_tuple(params.get("p", 0)),
                d=_coerce_tuple(params.get("d", 1)),
                g=params.get("g", 1),
                act=_parse_act(params.get("act", "relu")) or nn.ReLU(),
                bn=params.get("bn", False),
            ))
            output_channels = out_ch

        elif layer_type == "pool":
            params = spec[1] if len(spec) > 1 else {}
            built.append(modules["pool"](
                k=_coerce_tuple(params.get("k", 2)),
                s=_coerce_tuple(params.get("s", 2)),
                p=_coerce_tuple(params.get("p", 0)),
                d=_coerce_tuple(params.get("d", 1)),
                maxpool=params.get("maxpool", True),
            ))

        elif layer_type == "dropout":
            prob = spec[1] if len(spec) > 1 else 0.1
            built.append(modules["dropout"](prob))

        else:
            raise ValueError(f"Unknown encoder layer type: {layer_type}")

    if output_channels == 0:
        raise ValueError("Encoder must contain at least one conv layer")

    return nn.Sequential(*built), output_channels


# =====================================================================
# Aggregator builder
# =====================================================================

def _build_aggregator(cfg: dict, dim_type: str) -> tuple[nn.Module, int]:
    """Build aggregator from config dict.

    Returns:
        (module, output_features) where output_features is the total
        flattened feature count (before multiplying by encoder channels).
    """
    modules = _MODULE_MAP[dim_type]
    agg_type = cfg.get("type", "adaptive")
    levels = cfg.get("levels", 1)

    if agg_type == "multihead":
        if not isinstance(levels, list) or len(levels) != 3:
            raise ValueError("Multihead aggregator requires levels as [fine, balanced, coarse]")
        fine, balanced, coarse = levels

        if dim_type == "1d":
            module = modules["multihead"](fine, balanced, coarse)
            output_features = fine + balanced + coarse
        else:
            fine_s = int(sqrt(fine))
            balanced_s = int(sqrt(balanced))
            coarse_s = int(sqrt(coarse))
            module = modules["multihead"](
                (fine_s, fine_s), (balanced_s, balanced_s), (coarse_s, coarse_s)
            )
            output_features = fine_s**2 + balanced_s**2 + coarse_s**2

    elif agg_type == "adaptive":
        if not isinstance(levels, int):
            raise ValueError("Adaptive aggregator requires levels as int")

        if dim_type == "1d":
            module = modules["adaptive"](levels)
            output_features = levels
        else:
            levels_s = int(sqrt(levels))
            module = modules["adaptive"]((levels_s, levels_s))
            output_features = levels_s**2

    else:
        raise ValueError(f"Unknown aggregator type: {agg_type}. Options: adaptive, multihead")

    return module, output_features


# =====================================================================
# Head builder
# =====================================================================

def _build_head(cfg: dict, input_features: int, num_classes: int) -> nn.Module:
    """Build classification head from config dict."""
    depth = cfg.get("depth", 2)
    dropout = cfg.get("dropout", 0.1)
    act = _parse_act(cfg.get("act", "relu"))
    return MLP(input_features, num_classes, depth, dropout, act)


# =====================================================================
# Full model
# =====================================================================

class BuiltModel(nn.Module):
    """Model assembled from YAML-defined encoder, aggregator, and head."""

    def __init__(
        self,
        encoder: nn.Module,
        aggregator: nn.Module,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.aggregator(self.encoder(x)))

####################################################################
######                   PUBLIC METHODS                      #######
####################################################################

def build_model(cfg: dict, num_classes: int) -> BuiltModel:
    """Build a model from a parsed YAML config dict.

    Args:
        cfg: Parsed YAML dict with keys: type, encoder, aggregator, head.
        num_classes: Number of output classes.

    Returns:
        BuiltModel instance.
    """
    dim_type = cfg.get("type", "1d")
    if dim_type not in ("1d", "2d"):
        raise ValueError(f"type must be '1d' or '2d', got '{dim_type}'")

    encoder, enc_channels = _build_encoder(cfg["encoder"], dim_type)
    aggregator, agg_features = _build_aggregator(cfg.get("aggregator", {}), dim_type)
    head_input = enc_channels * agg_features
    head = _build_head(cfg.get("head", {}), head_input, num_classes)

    return BuiltModel(encoder, aggregator, head)


def build_model_from_yaml(path: str | Path, num_classes: int) -> BuiltModel:
    """Build a model from a YAML file.

    Args:
        path: Path to YAML file.
        num_classes: Number of output classes.

    Returns:
        BuiltModel instance.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return build_model(cfg, num_classes)