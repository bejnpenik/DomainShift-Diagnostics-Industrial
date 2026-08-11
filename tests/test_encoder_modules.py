"""
Tests for new encoder module/builder additions: residual blocks and
channel-attention layers (SE, ECA, CBAM).

Covers: module-level units (ResBlock1D/2D, SE1D/2D, ECA1D/2D, CBAM1D/2D),
        builder wiring (_MODULE_MAP, _build_encoder attention validation),
        end-to-end forward+backward through the new YAML configs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from model.builder import build_model, build_model_from_yaml, BuiltModel
from model.modules import (
    ResBlock1D, ResBlock2D,
    SE1D, SE2D, ECA1D, ECA2D,
)

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "models"


# =====================================================================
# ResBlock1D
# =====================================================================

class TestResBlock1D:
    def test_identity_shortcut_when_same_channels_and_unit_stride(self):
        block = ResBlock1D(4, 4, k=3, s=1)
        assert isinstance(block._blocks[0]._shortcut, nn.Identity)

    def test_projection_shortcut_when_channels_differ(self):
        block = ResBlock1D(4, 8, k=3, s=1)
        assert not isinstance(block._blocks[0]._shortcut, nn.Identity)

    def test_projection_shortcut_when_strided(self):
        block = ResBlock1D(4, 4, k=3, s=2)
        assert not isinstance(block._blocks[0]._shortcut, nn.Identity)

    @pytest.mark.parametrize("k", [3, 5, 7])
    @pytest.mark.parametrize("s", [1, 2])
    def test_output_shape(self, k, s):
        block = ResBlock1D(4, 8, k=k, s=s)
        L = 64
        x = torch.randn(2, 4, L)
        out = block(x)
        expected_L = (L - 1) // s + 1
        assert out.shape == (2, 8, expected_L)

    def test_even_k_raises(self):
        with pytest.raises(ValueError, match="odd"):
            ResBlock1D(4, 8, k=4)

    def test_blocks_stacking(self):
        block = ResBlock1D(4, 8, k=3, s=2, blocks=2)
        assert len(block._blocks) == 2
        x = torch.randn(2, 4, 65)
        out = block(x)
        expected_L = (65 - 1) // 2 + 1
        assert out.shape == (2, 8, expected_L)

    def test_gradient_flow(self):
        block = ResBlock1D(4, 8, k=3, s=1)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.any(x.grad != 0)


# =====================================================================
# ResBlock2D
# =====================================================================

class TestResBlock2D:
    def test_identity_shortcut_when_same_channels_and_unit_stride(self):
        block = ResBlock2D(4, 4, k=3, s=1)
        assert isinstance(block._blocks[0]._shortcut, nn.Identity)

    def test_projection_shortcut_when_channels_differ(self):
        block = ResBlock2D(4, 8, k=3, s=1)
        assert not isinstance(block._blocks[0]._shortcut, nn.Identity)

    def test_projection_shortcut_when_strided(self):
        block = ResBlock2D(4, 4, k=3, s=2)
        assert not isinstance(block._blocks[0]._shortcut, nn.Identity)

    @pytest.mark.parametrize("k", [3, 5, 7])
    @pytest.mark.parametrize("s", [1, 2])
    def test_output_shape(self, k, s):
        block = ResBlock2D(4, 8, k=k, s=s)
        H, W = 32, 32
        x = torch.randn(2, 4, H, W)
        out = block(x)
        expected = (H - 1) // s + 1
        assert out.shape == (2, 8, expected, expected)

    def test_asymmetric_tuple_k_and_s(self):
        block = ResBlock2D(4, 8, k=(3, 5), s=(1, 2))
        H, W = 20, 21
        x = torch.randn(2, 4, H, W)
        out = block(x)
        expected_H = (H - 1) // 1 + 1
        expected_W = (W - 1) // 2 + 1
        assert out.shape == (2, 8, expected_H, expected_W)

    def test_even_k_raises(self):
        with pytest.raises(ValueError, match="odd"):
            ResBlock2D(4, 8, k=4)

    def test_even_element_in_tuple_k_raises(self):
        with pytest.raises(ValueError, match="odd"):
            ResBlock2D(4, 8, k=(3, 4))

    def test_blocks_stacking(self):
        block = ResBlock2D(4, 8, k=3, s=2, blocks=2)
        assert len(block._blocks) == 2
        x = torch.randn(2, 4, 33, 33)
        out = block(x)
        expected = (33 - 1) // 2 + 1
        assert out.shape == (2, 8, expected, expected)

    def test_gradient_flow(self):
        block = ResBlock2D(4, 8, k=3, s=1)
        x = torch.randn(2, 4, 16, 16, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.any(x.grad != 0)


# =====================================================================
# SE1D / SE2D
# =====================================================================

class TestSE1D:
    def test_hidden_clamped_to_one(self):
        se = SE1D(channels=2, r=4)
        assert se._fc1.out_features == 1

    def test_output_shape_matches_input(self):
        se = SE1D(channels=8, r=2)
        x = torch.randn(3, 8, 40)
        out = se(x)
        assert out.shape == x.shape

    def test_gate_only_attenuates(self):
        se = SE1D(channels=8, r=2)
        x = torch.randn(3, 8, 40)
        out = se(x)
        assert torch.all(out.abs() <= x.abs() + 1e-6)

    def test_gradient_flow(self):
        se = SE1D(channels=8, r=2)
        x = torch.randn(3, 8, 40, requires_grad=True)
        out = se(x)
        out.sum().backward()
        assert x.grad is not None


class TestSE2D:
    def test_hidden_clamped_to_one(self):
        se = SE2D(channels=2, r=4)
        assert se._fc1.out_features == 1

    def test_output_shape_matches_input(self):
        se = SE2D(channels=8, r=2)
        x = torch.randn(3, 8, 10, 10)
        out = se(x)
        assert out.shape == x.shape

    def test_gate_only_attenuates(self):
        se = SE2D(channels=8, r=2)
        x = torch.randn(3, 8, 10, 10)
        out = se(x)
        assert torch.all(out.abs() <= x.abs() + 1e-6)

    def test_gradient_flow(self):
        se = SE2D(channels=8, r=2)
        x = torch.randn(3, 8, 10, 10, requires_grad=True)
        out = se(x)
        out.sum().backward()
        assert x.grad is not None


# =====================================================================
# ECA1D / ECA2D
# =====================================================================

class TestECA1D:
    def test_even_k_raises(self):
        with pytest.raises(ValueError, match="odd"):
            ECA1D(channels=8, k=4)

    def test_shape_preserved(self):
        eca = ECA1D(channels=8, k=3)
        x = torch.randn(3, 8, 40)
        out = eca(x)
        assert out.shape == x.shape

    def test_param_count_equals_k(self):
        for k in (3, 5, 7):
            eca = ECA1D(channels=8, k=k)
            assert sum(p.numel() for p in eca.parameters()) == k

    def test_gradient_flow(self):
        eca = ECA1D(channels=8, k=3)
        x = torch.randn(3, 8, 40, requires_grad=True)
        out = eca(x)
        out.sum().backward()
        assert x.grad is not None


class TestECA2D:
    def test_even_k_raises(self):
        with pytest.raises(ValueError, match="odd"):
            ECA2D(channels=8, k=4)

    def test_shape_preserved(self):
        eca = ECA2D(channels=8, k=3)
        x = torch.randn(3, 8, 10, 10)
        out = eca(x)
        assert out.shape == x.shape

    def test_param_count_equals_k(self):
        for k in (3, 5, 7):
            eca = ECA2D(channels=8, k=k)
            assert sum(p.numel() for p in eca.parameters()) == k

    def test_gradient_flow(self):
        eca = ECA2D(channels=8, k=3)
        x = torch.randn(3, 8, 10, 10, requires_grad=True)
        out = eca(x)
        out.sum().backward()
        assert x.grad is not None


# =====================================================================
# Builder wiring — res / se / eca
# =====================================================================

class TestBuilderAttentionLayers:
    def test_se_without_preceding_conv_raises(self):
        cfg = {
            "type": "1d",
            "encoder": [["se", {"r": 2}]],
        }
        with pytest.raises(ValueError, match="must follow a conv or res"):
            build_model(cfg, num_classes=3)

    def test_eca_without_preceding_conv_raises(self):
        cfg = {
            "type": "1d",
            "encoder": [["eca", {"k": 3}]],
        }
        with pytest.raises(ValueError, match="must follow a conv or res"):
            build_model(cfg, num_classes=3)

    def test_se_after_res(self):
        cfg = {
            "type": "1d",
            "encoder": [
                ["res", 1, 4, {"k": 3, "s": 1}],
                ["se", {"r": 2}],
            ],
            "aggregator": {"type": "adaptive", "levels": 1},
        }
        model = build_model(cfg, num_classes=3)
        out = model(torch.randn(2, 1, 64))
        assert out.shape == (2, 3)

    def test_eca_after_conv_pool_dropout(self):
        """Attention validity is 'appeared earlier', not 'immediately precedes'."""
        cfg = {
            "type": "1d",
            "encoder": [
                ["conv", 1, 4, {"k": 3, "s": 1}],
                ["pool", {"k": 2, "s": 2}],
                ["dropout", 0.1],
                ["eca", {"k": 3}],
            ],
            "aggregator": {"type": "adaptive", "levels": 1},
        }
        model = build_model(cfg, num_classes=3)
        out = model(torch.randn(2, 1, 64))
        assert out.shape == (2, 3)

    def test_res_updates_head_input_size(self):
        cfg = {
            "type": "1d",
            "encoder": [
                ["conv", 1, 2, {"k": 3, "s": 1}],
                ["res", 2, 8, {"k": 3, "s": 1}],
            ],
            "aggregator": {"type": "adaptive", "levels": 1},
            "head": {"depth": 1},
        }
        model = build_model(cfg, num_classes=3)
        assert model.head.m[0].in_features == 8

    def test_res_encoder_forward_and_backward(self):
        cfg = {
            "type": "2d",
            "encoder": [
                ["res", 1, 4, {"k": 3, "s": 2}],
            ],
            "aggregator": {"type": "adaptive", "levels": 1},
        }
        model = build_model(cfg, num_classes=3)
        x = torch.randn(2, 1, 16, 16)
        out = model(x)
        assert out.shape == (2, 3)
        out.sum().backward()


# =====================================================================
# New YAML configs — forward + backward
# =====================================================================

_NEW_CONFIGS_1D = [
    "cnn1d_res.yaml",
    "cnn1d_se.yaml",
    "cnn1d_eca.yaml",
    "cnn1d_res_se.yaml",
]

_NEW_CONFIGS_2D = [
    "cnn2d_res.yaml",
    "cnn2d_se.yaml",
    "cnn2d_eca.yaml",
    "cnn2d_res_se.yaml",
]


class TestNewYAMLConfigs1D:
    @pytest.mark.parametrize("filename", _NEW_CONFIGS_1D)
    def test_forward_and_backward(self, filename):
        model = build_model_from_yaml(_CONFIGS_DIR / filename, num_classes=4)
        x = torch.randn(4, 1, 600)
        out = model(x)
        assert out.shape == (4, 4)
        out.sum().backward()
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())


class TestNewYAMLConfigs2D:
    @pytest.mark.parametrize("filename", _NEW_CONFIGS_2D)
    def test_forward_and_backward(self, filename):
        model = build_model_from_yaml(_CONFIGS_DIR / filename, num_classes=4)
        x = torch.randn(4, 1, 129, 38)
        out = model(x)
        assert out.shape == (4, 4)
        out.sum().backward()
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
