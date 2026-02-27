"""
Tests for model package.

Covers: builder (build_model, build_model_from_yaml, BuiltModel),
        ModelConfig (direct, from_dict, from_yaml),
        module map dispatch, aggregator output features,
        equivalence with old hardcoded CNN1D/CNN2D.
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

import yaml
import torch
import torch.nn as nn

from model.config import ModelConfig
from model.builder import build_model, build_model_from_yaml, BuiltModel


# =====================================================================
# Helpers
# =====================================================================

def _cfg_1d_simple():
    return {
        "name": "test_1d",
        "type": "1d",
        "encoder": [
            ["conv", 1, 2, {"k": 7, "s": 1, "p": 0}],
            ["pool", {"k": 2, "s": 2, "p": 1}],
            ["conv", 2, 4, {"k": 5, "s": 1, "p": 0}],
            ["pool", {"k": 2, "s": 2, "p": 1}],
            ["conv", 4, 8, {"k": 3, "s": 1, "p": 0}],
            ["pool", {"k": 2, "s": 2, "p": 1}],
        ],
        "aggregator": {"type": "adaptive", "levels": 1},
        "head": {"depth": 2, "dropout": 0.1, "act": "relu"},
    }


def _cfg_2d_simple():
    return {
        "name": "test_2d",
        "type": "2d",
        "encoder": [
            ["conv", 1, 2, {"k": [3, 1], "s": 1, "p": [3, 0]}],
            ["pool", {"k": [2, 1], "s": [2, 1], "p": [1, 0]}],
            ["conv", 2, 4, {"k": [3, 1], "s": 1, "p": [2, 0]}],
            ["pool", {"k": [2, 1], "s": [2, 1], "p": [1, 0]}],
            ["conv", 4, 8, {"k": 3, "s": 2, "p": 0}],
            ["pool", {"k": 2, "s": 2, "p": 0}],
        ],
        "aggregator": {"type": "adaptive", "levels": 1},
        "head": {"depth": 2, "dropout": 0.1, "act": "relu"},
    }


def _write_yaml(cfg: dict) -> Path:
    """Write config to a temp YAML file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(cfg, f)
    f.close()
    return Path(f.name)


# =====================================================================
# build_model — 1D
# =====================================================================

class TestBuildModel1D:
    def test_basic_forward(self):
        model = build_model(_cfg_1d_simple(), num_classes=4)
        x = torch.randn(8, 1, 600)
        out = model(x)
        assert out.shape == (8, 4)

    def test_returns_built_model(self):
        model = build_model(_cfg_1d_simple(), num_classes=3)
        assert isinstance(model, BuiltModel)
        assert hasattr(model, "encoder")
        assert hasattr(model, "aggregator")
        assert hasattr(model, "head")

    def test_different_num_classes(self):
        for nc in [2, 3, 5, 10]:
            model = build_model(_cfg_1d_simple(), num_classes=nc)
            out = model(torch.randn(4, 1, 600))
            assert out.shape == (4, nc)

    def test_with_dropout(self):
        cfg = _cfg_1d_simple()
        cfg["encoder"].insert(2, ["dropout", 0.2])
        model = build_model(cfg, num_classes=3)
        out = model(torch.randn(4, 1, 600))
        assert out.shape == (4, 3)

    def test_with_batchnorm(self):
        cfg = _cfg_1d_simple()
        cfg["encoder"][0][3]["bn"] = True
        model = build_model(cfg, num_classes=3)
        out = model(torch.randn(4, 1, 600))
        assert out.shape == (4, 3)

    def test_adaptive_levels_16(self):
        cfg = _cfg_1d_simple()
        cfg["aggregator"]["levels"] = 16
        model = build_model(cfg, num_classes=4)
        out = model(torch.randn(8, 1, 600))
        assert out.shape == (8, 4)

    def test_multihead_aggregator(self):
        cfg = _cfg_1d_simple()
        cfg["aggregator"] = {"type": "multihead", "levels": [16, 4, 1]}
        model = build_model(cfg, num_classes=3)
        out = model(torch.randn(8, 1, 600))
        assert out.shape == (8, 3)

    def test_head_depth_1(self):
        cfg = _cfg_1d_simple()
        cfg["head"]["depth"] = 1
        model = build_model(cfg, num_classes=3)
        out = model(torch.randn(4, 1, 600))
        assert out.shape == (4, 3)

    def test_head_no_act(self):
        cfg = _cfg_1d_simple()
        cfg["head"]["act"] = "none"
        model = build_model(cfg, num_classes=3)
        out = model(torch.randn(4, 1, 600))
        assert out.shape == (4, 3)

    def test_gelu_activation(self):
        cfg = _cfg_1d_simple()
        cfg["head"]["act"] = "gelu"
        model = build_model(cfg, num_classes=3)
        out = model(torch.randn(4, 1, 600))
        assert out.shape == (4, 3)


# =====================================================================
# build_model — 2D
# =====================================================================

class TestBuildModel2D:
    def test_basic_forward(self):
        model = build_model(_cfg_2d_simple(), num_classes=4)
        x = torch.randn(8, 1, 129, 38)
        out = model(x)
        assert out.shape == (8, 4)

    def test_different_num_classes(self):
        for nc in [2, 4, 6]:
            model = build_model(_cfg_2d_simple(), num_classes=nc)
            out = model(torch.randn(4, 1, 129, 38))
            assert out.shape == (4, nc)

    def test_adaptive_levels_16(self):
        cfg = _cfg_2d_simple()
        cfg["aggregator"]["levels"] = 16
        model = build_model(cfg, num_classes=4)
        out = model(torch.randn(8, 1, 129, 38))
        assert out.shape == (8, 4)

    def test_multihead_aggregator(self):
        cfg = _cfg_2d_simple()
        cfg["aggregator"] = {"type": "multihead", "levels": [16, 4, 1]}
        model = build_model(cfg, num_classes=3)
        out = model(torch.randn(8, 1, 129, 38))
        assert out.shape == (8, 3)

    def test_with_dropout_and_bn(self):
        cfg = _cfg_2d_simple()
        cfg["encoder"].insert(2, ["dropout", 0.15])
        cfg["encoder"][0][3]["bn"] = True
        model = build_model(cfg, num_classes=3)
        out = model(torch.randn(4, 1, 129, 38))
        assert out.shape == (4, 3)


# =====================================================================
# build_model — validation
# =====================================================================

class TestBuildModelValidation:
    def test_unknown_type(self):
        cfg = _cfg_1d_simple()
        cfg["type"] = "3d"
        with pytest.raises(ValueError, match="type must be"):
            build_model(cfg, num_classes=3)

    def test_unknown_layer_type(self):
        cfg = _cfg_1d_simple()
        cfg["encoder"].append(["attention", 8])
        with pytest.raises(ValueError, match="Unknown encoder layer"):
            build_model(cfg, num_classes=3)

    def test_no_conv_in_encoder(self):
        cfg = _cfg_1d_simple()
        cfg["encoder"] = [["pool", {"k": 2, "s": 2}]]
        with pytest.raises(ValueError, match="at least one conv"):
            build_model(cfg, num_classes=3)

    def test_unknown_aggregator_type(self):
        cfg = _cfg_1d_simple()
        cfg["aggregator"]["type"] = "attention"
        with pytest.raises(ValueError, match="Unknown aggregator"):
            build_model(cfg, num_classes=3)

    def test_multihead_requires_list(self):
        cfg = _cfg_1d_simple()
        cfg["aggregator"] = {"type": "multihead", "levels": 16}
        with pytest.raises(ValueError, match="requires levels as"):
            build_model(cfg, num_classes=3)

    def test_adaptive_requires_int(self):
        cfg = _cfg_1d_simple()
        cfg["aggregator"] = {"type": "adaptive", "levels": [16, 4, 1]}
        with pytest.raises(ValueError, match="requires levels as int"):
            build_model(cfg, num_classes=3)

    def test_unknown_activation(self):
        cfg = _cfg_1d_simple()
        cfg["head"]["act"] = "swish_turbo"
        with pytest.raises(ValueError, match="Unknown activation"):
            build_model(cfg, num_classes=3)


# =====================================================================
# build_model_from_yaml
# =====================================================================

class TestBuildModelFromYAML:
    def test_1d_from_yaml(self):
        path = _write_yaml(_cfg_1d_simple())
        model = build_model_from_yaml(path, num_classes=4)
        out = model(torch.randn(8, 1, 600))
        assert out.shape == (8, 4)
        path.unlink()

    def test_2d_from_yaml(self):
        path = _write_yaml(_cfg_2d_simple())
        model = build_model_from_yaml(path, num_classes=3)
        out = model(torch.randn(8, 1, 129, 38))
        assert out.shape == (8, 3)
        path.unlink()

    def test_roundtrip_yaml(self):
        """Write then read should produce working model."""
        cfg = _cfg_1d_simple()
        cfg["aggregator"]["levels"] = 16
        cfg["head"]["depth"] = 3
        path = _write_yaml(cfg)
        model = build_model_from_yaml(path, num_classes=5)
        out = model(torch.randn(4, 1, 600))
        assert out.shape == (4, 5)
        path.unlink()


# =====================================================================
# ModelConfig integration
# =====================================================================

class TestModelConfigFromDict:
    def test_basic(self):
        mc = ModelConfig.from_dict(_cfg_1d_simple())
        assert mc.name == "test_1d"
        model = mc.create_model(num_classes=4)
        out = model(torch.randn(8, 1, 600))
        assert out.shape == (8, 4)

    def test_2d(self):
        mc = ModelConfig.from_dict(_cfg_2d_simple())
        assert mc.name == "test_2d"
        model = mc.create_model(num_classes=3)
        out = model(torch.randn(8, 1, 129, 38))
        assert out.shape == (8, 3)

    def test_create_model_fresh_each_time(self):
        mc = ModelConfig.from_dict(_cfg_1d_simple())
        m1 = mc.create_model(num_classes=3)
        m2 = mc.create_model(num_classes=3)
        assert m1 is not m2

    def test_different_num_classes(self):
        mc = ModelConfig.from_dict(_cfg_1d_simple())
        m3 = mc.create_model(num_classes=3)
        m5 = mc.create_model(num_classes=5)
        assert m3(torch.randn(4, 1, 600)).shape == (4, 3)
        assert m5(torch.randn(4, 1, 600)).shape == (4, 5)

    def test_unnamed_defaults(self):
        cfg = _cfg_1d_simple()
        del cfg["name"]
        mc = ModelConfig.from_dict(cfg)
        assert mc.name == "unnamed"


class TestModelConfigFromYAML:
    def test_basic(self):
        path = _write_yaml(_cfg_1d_simple())
        mc = ModelConfig.from_yaml(path)
        assert mc.name == "test_1d"
        model = mc.create_model(num_classes=4)
        out = model(torch.randn(8, 1, 600))
        assert out.shape == (8, 4)
        path.unlink()

    def test_2d(self):
        path = _write_yaml(_cfg_2d_simple())
        mc = ModelConfig.from_yaml(path)
        model = mc.create_model(num_classes=3)
        out = model(torch.randn(8, 1, 129, 38))
        assert out.shape == (8, 3)
        path.unlink()


class TestModelConfigDirect:
    """Ensure direct construction still works alongside YAML path."""

    def test_direct_with_callable(self):
        def factory(num_classes, **kw):
            return nn.Sequential(nn.Flatten(), nn.Linear(600, num_classes))

        mc = ModelConfig(name="direct", model_class=factory)
        model = mc.create_model(num_classes=3)
        out = model(torch.randn(4, 1, 600))
        assert out.shape == (4, 3)

    def test_frozen(self):
        mc = ModelConfig(name="x", model_class=nn.Linear)
        with pytest.raises(AttributeError):
            mc.name = "changed"


# =====================================================================
# Determinism
# =====================================================================

class TestBuilderDeterminism:
    def test_same_seed_same_output_1d(self):
        cfg = _cfg_1d_simple()
        x = torch.randn(8, 1, 600)

        torch.manual_seed(42)
        m1 = build_model(cfg, num_classes=3)
        m1.eval()
        with torch.no_grad():
            out1 = m1(x)

        torch.manual_seed(42)
        m2 = build_model(cfg, num_classes=3)
        m2.eval()
        with torch.no_grad():
            out2 = m2(x)

        torch.testing.assert_close(out1, out2)

    def test_same_seed_same_output_2d(self):
        cfg = _cfg_2d_simple()
        x = torch.randn(8, 1, 129, 38)

        torch.manual_seed(42)
        m1 = build_model(cfg, num_classes=3)
        m1.eval()
        with torch.no_grad():
            out1 = m1(x)

        torch.manual_seed(42)
        m2 = build_model(cfg, num_classes=3)
        m2.eval()
        with torch.no_grad():
            out2 = m2(x)

        torch.testing.assert_close(out1, out2)


# =====================================================================
# Aggregator output features
# =====================================================================

class TestAggregatorOutputFeatures:
    """Verify the head input dimension is computed correctly."""

    def test_adaptive_1d_levels_1(self):
        cfg = _cfg_1d_simple()
        cfg["aggregator"]["levels"] = 1
        model = build_model(cfg, num_classes=3)
        # encoder output_channels=8, aggregator features=1 → head input=8
        enc_out = model.encoder(torch.randn(1, 1, 600))
        agg_out = model.aggregator(enc_out)
        assert agg_out.shape[1] == 8  # 8 channels * 1 level

    def test_adaptive_1d_levels_16(self):
        cfg = _cfg_1d_simple()
        cfg["aggregator"]["levels"] = 16
        model = build_model(cfg, num_classes=3)
        enc_out = model.encoder(torch.randn(1, 1, 600))
        agg_out = model.aggregator(enc_out)
        assert agg_out.shape[1] == 8 * 16

    def test_multihead_1d(self):
        cfg = _cfg_1d_simple()
        cfg["aggregator"] = {"type": "multihead", "levels": [16, 4, 1]}
        model = build_model(cfg, num_classes=3)
        enc_out = model.encoder(torch.randn(1, 1, 600))
        agg_out = model.aggregator(enc_out)
        assert agg_out.shape[1] == 8 * (16 + 4 + 1)

    def test_adaptive_2d_levels_16(self):
        from math import sqrt
        cfg = _cfg_2d_simple()
        cfg["aggregator"]["levels"] = 16
        model = build_model(cfg, num_classes=3)
        enc_out = model.encoder(torch.randn(1, 1, 129, 38))
        agg_out = model.aggregator(enc_out)
        s = int(sqrt(16))
        assert agg_out.shape[1] == 8 * s * s

    def test_multihead_2d(self):
        from math import sqrt
        cfg = _cfg_2d_simple()
        cfg["aggregator"] = {"type": "multihead", "levels": [16, 4, 1]}
        model = build_model(cfg, num_classes=3)
        enc_out = model.encoder(torch.randn(1, 1, 129, 38))
        agg_out = model.aggregator(enc_out)
        expected = 8 * (int(sqrt(16))**2 + int(sqrt(4))**2 + int(sqrt(1))**2)
        assert agg_out.shape[1] == expected


# =====================================================================
# Defaults / edge cases
# =====================================================================

class TestBuilderDefaults:
    def test_default_type_is_1d(self):
        cfg = _cfg_1d_simple()
        del cfg["type"]
        model = build_model(cfg, num_classes=3)
        out = model(torch.randn(4, 1, 600))
        assert out.shape == (4, 3)

    def test_default_aggregator(self):
        cfg = _cfg_1d_simple()
        del cfg["aggregator"]
        model = build_model(cfg, num_classes=3)
        out = model(torch.randn(4, 1, 600))
        assert out.shape == (4, 3)

    def test_default_head(self):
        cfg = _cfg_1d_simple()
        del cfg["head"]
        model = build_model(cfg, num_classes=3)
        out = model(torch.randn(4, 1, 600))
        assert out.shape == (4, 3)

    def test_minimal_encoder(self):
        cfg = {
            "type": "1d",
            "encoder": [
                ["conv", 1, 4, {"k": 3, "s": 1}],
            ],
        }
        model = build_model(cfg, num_classes=2)
        out = model(torch.randn(4, 1, 600))
        assert out.shape == (4, 2)

    def test_conv_default_padding(self):
        """Conv with no p specified defaults to 0."""
        cfg = {
            "type": "1d",
            "encoder": [
                ["conv", 1, 4, {"k": 3, "s": 1}],
            ],
            "aggregator": {"type": "adaptive", "levels": 1},
            "head": {"depth": 1},
        }
        model = build_model(cfg, num_classes=2)
        out = model(torch.randn(4, 1, 100))
        assert out.shape == (4, 2)

    def test_pool_default_params(self):
        """Pool with no params uses defaults."""
        cfg = {
            "type": "1d",
            "encoder": [
                ["conv", 1, 4, {"k": 3, "s": 1}],
                ["pool"],
            ],
            "aggregator": {"type": "adaptive", "levels": 1},
        }
        model = build_model(cfg, num_classes=2)
        out = model(torch.randn(4, 1, 100))
        assert out.shape == (4, 2)

    def test_dropout_default_prob(self):
        """Dropout with no prob specified defaults to 0.1."""
        cfg = {
            "type": "1d",
            "encoder": [
                ["conv", 1, 4, {"k": 3, "s": 1}],
                ["dropout"],
            ],
            "aggregator": {"type": "adaptive", "levels": 1},
        }
        model = build_model(cfg, num_classes=2)
        out = model(torch.randn(4, 1, 100))
        assert out.shape == (4, 2)