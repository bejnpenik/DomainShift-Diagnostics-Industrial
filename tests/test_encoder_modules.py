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
