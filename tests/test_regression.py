"""
Regression tests.

These verify deterministic behavior and known numerical results 
so you can safely refactor without changing outputs.

All synthetic data, no data files needed.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from signal_preprocessor import SignalSegmenter, RawSignalRepresentation, SpectrogramRepresentation
from dataset import Normalisator
from cnn import CNN1D, CNN2D
from nn_utils import inject_noise
from metrics import compute_accuracy, compute_precision_per_class, compute_recall_per_class


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# Segmentation regression
# =====================================================================

class TestSegmentationRegression:
    """WindowSegmenter should produce exact same output as torch.unfold."""

    @pytest.mark.parametrize("sr,dur,ovlp", [
        (12000, 0.05, 0.2),
        (48000, 0.05, 0.5),
        (64000, 0.1, 0.0),
        (10000, 0.01, 0.5),
    ])
    def test_unfold_equivalence(self, sr, dur, ovlp):
        signal = np.random.RandomState(42).randn(sr * 2).astype(np.float32)
        seg = SignalSegmenter(window_duration=dur, overlap=ovlp)
        result = seg(signal, sr)

        # Manual unfold
        x = torch.from_numpy(signal)
        ws = int(dur * sr)
        os = int(ws * ovlp)
        step = ws - os
        expected = x.unfold(0, ws, step)

        torch.testing.assert_close(result, expected)


# =====================================================================
# Representation regression
# =====================================================================

class TestRepresentationRegression:
    def test_raw_is_unsqueeze(self):
        rep = RawSignalRepresentation()
        x = torch.randn(10, 600)
        torch.testing.assert_close(rep(x), x.unsqueeze(1))

    def test_spectrogram_matches_manual(self):
        n_fft, hop, win = 256, 64, 256
        rep = SpectrogramRepresentation(n_fft=n_fft, hop_length=hop, win_length=win)

        torch.manual_seed(42)
        x = torch.randn(10, 600)
        result = rep(x)

        # Manual
        window = torch.hann_window(win)
        manual = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                            window=window, return_complex=True)
        manual = torch.abs(manual)
        manual = torch.log1p(manual)
        manual = manual.unsqueeze(1)

        torch.testing.assert_close(result, manual)


# =====================================================================
# Normalization regression
# =====================================================================

class TestNormalizationRegression:
    def test_sample_mode_formula(self):
        set_seed(42)
        x = torch.randn(10, 1, 600)
        norm = Normalisator(mode="sample")
        result = norm(x)

        # Manual
        std, mean = torch.std_mean(x, dim=tuple(range(1, x.ndim)), keepdim=True)
        expected = (x - mean) / (std + 1e-8)
        torch.testing.assert_close(result, expected)

    def test_dataset_mode_formula(self):
        set_seed(42)
        x = torch.randn(100, 1, 600)
        norm = Normalisator(mode="dataset")
        norm.fit(x)
        result = norm(x)

        # Manual
        std, mean = torch.std_mean(x, dim=0, keepdim=True)
        expected = (x - mean) / (std + 1e-8)
        torch.testing.assert_close(result, expected)


# =====================================================================
# Model determinism
# =====================================================================

class TestModelDeterminism:
    def test_cnn1d_deterministic(self):
        """Same seed → same weights → same output."""
        x = torch.randn(8, 1, 600)

        set_seed(42)
        m1 = CNN1D(num_classes=3, aggregator_levels=16, head_depth=2)
        m1.eval()
        with torch.no_grad():
            out1 = m1(x)

        set_seed(42)
        m2 = CNN1D(num_classes=3, aggregator_levels=16, head_depth=2)
        m2.eval()
        with torch.no_grad():
            out2 = m2(x)

        torch.testing.assert_close(out1, out2)

    def test_cnn2d_deterministic(self):
        x = torch.randn(8, 1, 129, 38)

        set_seed(42)
        m1 = CNN2D(num_classes=3, aggregator_levels=16, head_depth=2)
        m1.eval()
        with torch.no_grad():
            out1 = m1(x)

        set_seed(42)
        m2 = CNN2D(num_classes=3, aggregator_levels=16, head_depth=2)
        m2.eval()
        with torch.no_grad():
            out2 = m2(x)

        torch.testing.assert_close(out1, out2)


# =====================================================================
# Metrics regression (known values)
# =====================================================================

class TestMetricsRegression:
    def test_known_confusion_matrix(self):
        cm = np.array([[45, 3, 2], [2, 48, 0], [1, 1, 48]])

        assert compute_accuracy(cm) == pytest.approx(141 / 150)
        prec = compute_precision_per_class(cm)
        assert prec[0] == pytest.approx(45 / 48, abs=1e-5)  # 45/(45+2+1)
        assert prec[1] == pytest.approx(48 / 52, abs=1e-5)  # 48/(3+48+1)
        assert prec[2] == pytest.approx(48 / 50, abs=1e-5)  # 48/(2+0+48)

        rec = compute_recall_per_class(cm)
        assert rec[0] == pytest.approx(45 / 50, abs=1e-5)  # 45/(45+3+2)
        assert rec[1] == pytest.approx(48 / 50, abs=1e-5)  # 48/(2+48+0)
        assert rec[2] == pytest.approx(48 / 50, abs=1e-5)  # 48/(1+1+48)


# =====================================================================
# Noise injection regression
# =====================================================================

class TestNoiseRegression:
    def test_zero_prob_identity(self):
        x = torch.randn(32, 1, 600)
        result = inject_noise(x, noise_prob=0.0, noise_std=0.1)
        torch.testing.assert_close(result, x)

    def test_deterministic_with_seed(self):
        x = torch.randn(32, 1, 600)
        torch.manual_seed(42)
        r1 = inject_noise(x, noise_prob=0.5, noise_std=0.1)
        torch.manual_seed(42)
        r2 = inject_noise(x, noise_prob=0.5, noise_std=0.1)
        torch.testing.assert_close(r1, r2)
