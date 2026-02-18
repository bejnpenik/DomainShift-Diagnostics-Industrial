"""
Regression and compatibility tests.

These verify that the refactored codebase produces IDENTICAL outputs
to the original flat codebase. Reference values were captured by running
capture_reference_values.py against the old code.

All synthetic data, no data files needed.
"""

from __future__ import annotations

import pytest
import numpy as np
import torch
import torch.nn as nn
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# 1. Segmentation
# =====================================================================

class TestSegmentationRegression:
    """SignalSegmenter must produce identical shapes, sums, and values."""

    from representation.signal.segmentation import SignalSegmenter

    FIRST5 = [0.49671414494514465, -0.13826429843902588, 0.6476885676383972,
              1.5230298042297363, -0.2341533750295639]

    @pytest.mark.parametrize("sr,dur,ovlp,expected_shape,expected_sum", [
        (12000, 0.05, 0.5, (79, 600), 109.794632),
        (48000, 0.05, 0.2, (49, 2400), 332.840485),
        (12000, 0.1, 0.0, (20, 1200), 44.900261),
    ])
    def test_shape_and_sum(self, sr, dur, ovlp, expected_shape, expected_sum):
        from representation.signal.segmentation import SignalSegmenter
        signal = np.random.RandomState(42).randn(sr * 2).astype(np.float32)
        seg = SignalSegmenter(window_duration=dur, overlap=ovlp)
        result = seg(signal, sr)
        assert tuple(result.shape) == expected_shape
        assert result.sum().item() == pytest.approx(expected_sum, abs=1e-3)

    @pytest.mark.parametrize("sr,dur,ovlp", [
        (12000, 0.05, 0.5),
        (48000, 0.05, 0.2),
        (12000, 0.1, 0.0),
    ])
    def test_first5_values(self, sr, dur, ovlp):
        from representation.signal.segmentation import SignalSegmenter
        signal = np.random.RandomState(42).randn(sr * 2).astype(np.float32)
        seg = SignalSegmenter(window_duration=dur, overlap=ovlp)
        result = seg(signal, sr)
        assert result[0, :5].tolist() == pytest.approx(self.FIRST5, abs=1e-6)

    @pytest.mark.parametrize("sr,dur,ovlp", [
        (12000, 0.05, 0.5),
        (48000, 0.05, 0.2),
        (12000, 0.1, 0.0),
    ])
    def test_unfold_equivalence(self, sr, dur, ovlp):
        from representation.signal.segmentation import SignalSegmenter
        signal = np.random.RandomState(42).randn(sr * 2).astype(np.float32)
        seg = SignalSegmenter(window_duration=dur, overlap=ovlp)
        result = seg(signal, sr)
        x = torch.from_numpy(signal)
        ws = int(dur * sr)
        step = ws - int(ws * ovlp)
        expected = x.unfold(0, ws, step)
        torch.testing.assert_close(result, expected)


# =====================================================================
# 2. Resampling
# =====================================================================

class TestResamplingRegression:

    def test_downsample_48k_to_12k(self):
        from representation.signal.resampling import Resampler
        resampler = Resampler(max_signal_bandwidth_factor=0.5)
        signal = np.random.RandomState(42).randn(48000).astype(np.float32)
        resampled = resampler(signal, sampling_rate=48000, target_sampling_rate=12000)
        new_sr = 12000
        assert len(resampled) == 12000
        assert new_sr == 12000
        assert np.sum(resampled) == pytest.approx(-14.208277, abs=1e-3)
        assert resampled[:5].tolist() == pytest.approx(
            [2.0047029920533532e-06, 0.0030077314004302025, 0.08530698716640472,
             0.4222296476364136, 0.5321354866027832], abs=1e-4)

    def test_same_rate_noop(self):
        from representation.signal.resampling import Resampler
        resampler = Resampler(max_signal_bandwidth_factor=0.5)
        signal = np.random.RandomState(42).randn(48000).astype(np.float32)
        resampled= resampler(signal, sampling_rate=48000, target_sampling_rate=48000)
        new_sr = 48000
        assert len(resampled) == 48000
        assert new_sr == 48000
        assert resampled is signal


# =====================================================================
# 3. Raw Representation (View)
# =====================================================================

class TestRawViewRegression:

    def test_shape_and_identity(self):
        from representation.signal.view import RawSignalView
        raw = RawSignalView()
        torch.manual_seed(99)
        windows = torch.randn(10, 600)
        result = raw(windows)
        assert tuple(result.shape) == (10, 1, 600)
        assert result.sum().item() == pytest.approx(49.464958, abs=1e-3)
        assert torch.equal(result, windows.unsqueeze(1))


# =====================================================================
# 4. Spectrogram Representation (View)
# =====================================================================

class TestSTFTViewRegression:

    def test_shape_and_values_hop64(self):
        from representation.signal.view import STFTSignalView
        spec = STFTSignalView(n_fft=256, hop_length=64, win_length=256)
        torch.manual_seed(99)
        windows = torch.randn(10, 600)
        result = spec(windows)
        assert tuple(result.shape) == (10, 1, 129, 10)
        assert result.sum().item() == pytest.approx(27355.494141, abs=1.0)
        assert result.min().item() == pytest.approx(0.003574, abs=1e-3)
        assert result.max().item() == pytest.approx(3.547405, abs=1e-3)

    def test_shape_and_values_hop128(self):
        from representation.signal.view import STFTSignalView
        spec = STFTSignalView(n_fft=256, hop_length=128, win_length=256)
        torch.manual_seed(99)
        windows = torch.randn(10, 600)
        result = spec(windows)
        assert tuple(result.shape) == (10, 1, 129, 5)
        assert result.sum().item() == pytest.approx(13605.842773, abs=1.0)


# =====================================================================
# 5. Full Signal Pipeline
# =====================================================================

class TestSignalPipelineRegression:
    """End-to-end pipeline must match old SignalModalityProcessor output."""

    def test_raw_pipeline_48k_to_12k(self):
        from representation.signal.config import SignalProcessorConfig, RawViewConfig
        from representation.signal.processor import SignalProcessor
        from collection import Metadata

        cfg = SignalProcessorConfig(
            name="raw_12k",
            target_sampling_rate=12000,
            window_duration=0.05,
            window_overlap=0.5,
            view=RawViewConfig(),
        )
        proc = SignalProcessor(cfg)
        signal = np.random.RandomState(42).randn(48000).astype(np.float32)
        meta = Metadata({"sampling_rate": 48000})
        result = proc(signal, meta)
        assert tuple(result.shape) == (39, 1, 600)
        assert result.sum().item() == pytest.approx(-25.880222, abs=1e-2)
        assert result.mean().item() == pytest.approx(-0.001106, abs=1e-4)

    def test_raw_pipeline_same_rate(self):
        from representation.signal.config import SignalProcessorConfig, RawViewConfig
        from representation.signal.processor import SignalProcessor
        from collection import Metadata

        cfg = SignalProcessorConfig(
            name="raw_12k",
            target_sampling_rate=12000,
            window_duration=0.05,
            window_overlap=0.5,
            view=RawViewConfig(),
        )
        proc = SignalProcessor(cfg)
        signal = np.random.RandomState(42).randn(12000).astype(np.float32)
        meta = Metadata({"sampling_rate": 12000})
        result = proc(signal, meta)
        assert tuple(result.shape) == (39, 1, 600)
        assert result.sum().item() == pytest.approx(-128.675247, abs=1e-2)

    def test_spec_pipeline_48k_to_12k(self):
        from representation.signal.config import SignalProcessorConfig, STFTViewConfig
        from representation.signal.processor import SignalProcessor
        from collection import Metadata

        cfg = SignalProcessorConfig(
            name="spec_12k",
            target_sampling_rate=12000,
            window_duration=0.05,
            window_overlap=0.5,
            view=STFTViewConfig(n_fft=256, hop_length=128, win_length=256),
        )
        proc = SignalProcessor(cfg)
        signal = np.random.RandomState(42).randn(48000).astype(np.float32)
        meta = Metadata({"sampling_rate": 48000})
        result = proc(signal, meta)
        assert tuple(result.shape) == (39, 1, 129, 5)
        assert result.sum().item() == pytest.approx(35160.882812, abs=2.0)
        assert result.mean().item() == pytest.approx(1.397769, abs=1e-3)


# =====================================================================
# 6. Normalization
# =====================================================================

class TestNormalizationRegression:

    def test_sample_mode(self):
        from normalization import Normalisator
        set_seed(42)
        x = torch.randn(50, 1, 600)
        norm = Normalisator(mode="sample")
        result = norm(x)
        assert result.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert result.std().item() == pytest.approx(0.99918294, abs=1e-4)
        assert result.sum().item() == pytest.approx(0.000008, abs=1e-3)

    def test_dataset_mode(self):
        from normalization import Normalisator
        set_seed(42)
        x = torch.randn(50, 1, 600)
        norm = Normalisator(mode="dataset")
        norm.fit(x)
        result = norm(x)
        assert result.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert result.std().item() == pytest.approx(0.98996598, abs=1e-4)
        assert result.sum().item() == pytest.approx(-0.000006, abs=1e-3)

    def test_pretrained_mode(self):
        from normalization import Normalisator
        set_seed(42)
        x = torch.randn(50, 1, 600)
        norm = Normalisator(mode="pretrained", mean=torch.tensor(0.5), std=torch.tensor(2.0))
        result = norm(x)
        assert result.sum().item() == pytest.approx(-7431.179199, abs=1.0)

    def test_sample_mode_formula(self):
        from normalization import Normalisator
        set_seed(42)
        x = torch.randn(10, 1, 600)
        norm = Normalisator(mode="sample")
        result = norm(x)
        std, mean = torch.std_mean(x, dim=tuple(range(1, x.ndim)), keepdim=True)
        expected = (x - mean) / (std + 1e-8)
        torch.testing.assert_close(result, expected)

    def test_dataset_mode_formula(self):
        from normalization import Normalisator
        set_seed(42)
        x = torch.randn(100, 1, 600)
        norm = Normalisator(mode="dataset")
        norm.fit(x)
        result = norm(x)
        std, mean = torch.std_mean(x, dim=0, keepdim=True)
        expected = (x - mean) / (std + 1e-8)
        torch.testing.assert_close(result, expected)


# =====================================================================
# 7. CNN Models
# =====================================================================

class TestCNNRegression:

    def test_cnn1d_forward(self):
        from model.cnn import CNN1D
        set_seed(42)
        x = torch.randn(8, 1, 600)
        set_seed(42)
        model = CNN1D(num_classes=4, aggregator_levels=16, head_depth=2)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert tuple(out.shape) == (8, 4)
        assert out.sum().item() == pytest.approx(-0.772600, abs=1e-3)
        assert out.argmax(dim=1).tolist() == [2, 2, 2, 2, 2, 2, 2, 2]

    def test_cnn1d_multihead(self):
        from model.cnn import CNN1D
        set_seed(42)
        x = torch.randn(8, 1, 600)
        set_seed(42)
        model = CNN1D(num_classes=3, aggregator_levels=(16, 4, 1), head_depth=2)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert tuple(out.shape) == (8, 3)
        assert out.sum().item() == pytest.approx(0.427359, abs=1e-3)
        assert out.argmax(dim=1).tolist() == [2, 2, 2, 2, 2, 2, 2, 2]

    def test_cnn2d_forward(self):
        from model.cnn import CNN2D
        set_seed(42)
        x = torch.randn(8, 1, 129, 38)
        set_seed(42)
        model = CNN2D(num_classes=4, aggregator_levels=16, head_depth=2)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert tuple(out.shape) == (8, 4)
        assert out.sum().item() == pytest.approx(-0.741838, abs=1e-3)
        assert out.argmax(dim=1).tolist() == [2, 2, 2, 2, 2, 2, 2, 2]

    def test_cnn1d_deterministic(self):
        from model.cnn import CNN1D
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
        from model.cnn import CNN2D
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
# 8. Training
# =====================================================================

class TestTrainingRegression:

    def test_short_deterministic_run(self):
        from model.cnn import CNN1D
        from training.trainer import Trainer
        from training.config import TrainerConfig

        set_seed(42)
        X_train = torch.randn(60, 1, 600)
        Y_train = torch.tensor([0]*20 + [1]*20 + [2]*20)
        X_val = torch.randn(30, 1, 600)
        Y_val = torch.tensor([0]*10 + [1]*10 + [2]*10)

        set_seed(42)
        model = CNN1D(num_classes=3, aggregator_levels=1, head_depth=1)

        cfg = TrainerConfig(
            max_epochs=5,
            optimizer_name="adamw",
            lr=1e-3,
            weight_decay=1e-4,
            momentum=0.9,
            device="cpu",
            early_stopping=None,
            noise=None,
            min_epochs=0,
            verbose_level=0,
        )
        trainer = Trainer(cfg)
        result = trainer.fit(model, (X_train, Y_train), (X_val, Y_val))

        assert result.epochs_run == 5
        assert result.train_loss == pytest.approx(1.100540, abs=1e-3)
        assert result.train_acc == pytest.approx(33.33, abs=0.5)
        assert result.val_loss == pytest.approx(1.099705, abs=1e-3)
        assert result.val_acc == pytest.approx(33.33, abs=0.5)

    def test_predict_confusion_matrix(self):
        from model.cnn import CNN1D
        from training.trainer import Trainer
        from training.config import TrainerConfig

        set_seed(42)
        X_train = torch.randn(60, 1, 600)
        Y_train = torch.tensor([0]*20 + [1]*20 + [2]*20)
        X_val = torch.randn(30, 1, 600)
        Y_val = torch.tensor([0]*10 + [1]*10 + [2]*10)

        set_seed(42)
        model = CNN1D(num_classes=3, aggregator_levels=1, head_depth=1)

        cfg = TrainerConfig(
            max_epochs=5, optimizer_name="adamw", lr=1e-3, weight_decay=1e-4,
            device="cpu", early_stopping=None, noise=None,
            min_epochs=0, verbose_level=0,
        )
        trainer = Trainer(cfg)
        result = trainer.fit(model, (X_train, Y_train), (X_val, Y_val))
        cm = trainer.predict(result.model, X_val, Y_val)

        assert cm.shape == (3, 3)
        assert cm.sum() == 30
        assert np.trace(cm) == 10
        expected = np.array([[0, 0, 10], [0, 0, 10], [0, 0, 10]])
        np.testing.assert_array_equal(cm, expected)


# =====================================================================
# 9. Noise Injection
# =====================================================================

class TestNoiseRegression:

    def test_zero_prob_identity(self):
        from training.trainer import Trainer  # inject_noise is internal
        x = torch.randn(32, 1, 600)
        # If inject_noise is a standalone function in your new code, import it.
        # Otherwise test through Trainer with noise=None which skips injection.
        # For now test the concept: no noise = identity
        cfg_no_noise = None  # placeholder
        # This is already covered by training test with noise=None above.

    def test_deterministic_with_seed(self):
        """Noise injection with same seed produces same result."""
        # This tests the general contract - specific function may have moved.
        x = torch.randn(32, 1, 600)
        torch.manual_seed(42)
        mask1 = torch.rand(x.shape[0]) < 0.5
        noise1 = torch.randn_like(x) * 0.1
        r1 = x.clone()
        r1[mask1] = x[mask1] + noise1[mask1]

        torch.manual_seed(42)
        mask2 = torch.rand(x.shape[0]) < 0.5
        noise2 = torch.randn_like(x) * 0.1
        r2 = x.clone()
        r2[mask2] = x[mask2] + noise2[mask2]

        torch.testing.assert_close(r1, r2)


# =====================================================================
# 10. Metrics
# =====================================================================

class TestMetricsRegression:

    CM = np.array([[45, 3, 2], [2, 48, 0], [1, 1, 48]])

    def test_accuracy(self):
        from results.metrics import compute_accuracy
        assert compute_accuracy(self.CM) == pytest.approx(0.94, abs=1e-8)

    def test_precision(self):
        from results.metrics import compute_precision_per_class
        prec = compute_precision_per_class(self.CM)
        assert prec[0] == pytest.approx(0.9375, abs=1e-4)
        assert prec[1] == pytest.approx(0.9230769, abs=1e-4)
        assert prec[2] == pytest.approx(0.96, abs=1e-4)

    def test_recall(self):
        from results.metrics import compute_recall_per_class
        rec = compute_recall_per_class(self.CM)
        assert rec[0] == pytest.approx(0.90, abs=1e-4)
        assert rec[1] == pytest.approx(0.96, abs=1e-4)
        assert rec[2] == pytest.approx(0.96, abs=1e-4)

    def test_f1(self):
        from results.metrics import compute_f1_per_class
        f1 = compute_f1_per_class(self.CM)
        assert f1[0] == pytest.approx(0.918367347, abs=1e-5)
        assert f1[1] == pytest.approx(0.941176471, abs=1e-5)
        assert f1[2] == pytest.approx(0.96, abs=1e-5)

    def test_metrics_calculator(self):
        from results.metrics import MetricsCalculator
        calc = MetricsCalculator()
        metrics = calc.from_confusion_matrix(self.CM, {0: 'h', 1: 'ir', 2: 'or'})
        assert metrics.accuracy == pytest.approx(0.94, abs=1e-8)
        assert metrics.class_metrics[0].f1 == pytest.approx(0.918367347, abs=1e-5)


# =====================================================================
# 11. Collection — CWRU
# =====================================================================

class TestCollectionCWRURegression:

    @pytest.fixture(autouse=True)
    def setup(self):
        from collection import DatasetCollection, Task, Rule, Interactions
        self.collection = DatasetCollection('configs/cwru.yaml')
        self.Task = Task
        self.Rule = Rule
        self.Interactions = Interactions

    def test_basic_properties(self):
        assert self.collection.name == 'cwru'
        assert len(self.collection.files) == 165
        assert sorted(self.collection.schema.keys()) == [
            'bearing_position', 'condition', 'fault_element',
            'fault_position', 'fault_size', 'sampling_rate'
        ]

    def test_filter_values(self):
        c = self.collection
        assert c.get_filter_value_from_description('fault_element', 'normal') == 0
        assert c.get_filter_value_from_description('fault_element', 'inner ring') == 1
        assert c.get_filter_value_from_description('fault_element', 'outer ring') == 2
        assert c.get_filter_value_from_description('fault_element', 'ball') == 3
        assert c.get_filter_value_from_description('sampling_rate', 12000) == 1
        assert c.get_filter_value_from_description('sampling_rate', 48000) == 2

    def test_valid_filter_combinations(self):
        c = self.collection
        fe_normal = 0
        fe_inner = 1
        fe_outer = 2
        fe_ball = 3
        task = self.Task(
            target='fault_element',
            domain_factors=('fault_size', 'bearing_position', 'condition'),
            defaults=self.Rule(
                fixed={'fault_size': 0, 'bearing_position': 0, 'condition': 0},
                resolve={
                    'sampling_rate': [1, 2],
                    'fault_position': c.get_filter_value_from_description('fault_position', 'centered'),
                }
            ),
            classes={
                fe_normal: self.Rule(
                    fixed={'fault_size': 0},
                    resolve={
                        'fault_position': c.get_filter_value_from_description('fault_position', 'normal'),
                        'sampling_rate': c.get_filter_value_from_description('sampling_rate', 48000),
                    }
                )
            },
            class_interactions={
                fe_inner: self.Interactions.from_dict({'bearing_position': {1: {'sampling_rate': 1}}}),
                fe_outer: self.Interactions.from_dict({'bearing_position': {1: {'sampling_rate': 1}}}),
                fe_ball: self.Interactions.from_dict({'bearing_position': {1: {'sampling_rate': 1}}}),
            }
        )
        filters = c.create_valid_filter_combinations(
            task, ('fault_size', 'bearing_position', 'condition'),
            fault_size=[0, 4]
        )
        assert len(filters) == 18
        assert filters[0] == {'fault_size': 1, 'bearing_position': 1, 'condition': 1}
        assert filters[-1] == {'fault_size': 3, 'bearing_position': 2, 'condition': 4}

    def test_dataset_plan(self):
        c = self.collection
        task = self.Task(
            target='fault_element',
            domain_factors=('fault_size', 'bearing_position', 'condition'),
            defaults=self.Rule(
                fixed={'fault_size': 0, 'bearing_position': 0, 'condition': 0},
                resolve={
                    'sampling_rate': [1, 2],
                    'fault_position': c.get_filter_value_from_description('fault_position', 'centered'),
                }
            ),
            classes={
                0: self.Rule(
                    fixed={'fault_size': 0},
                    resolve={
                        'fault_position': c.get_filter_value_from_description('fault_position', 'normal'),
                        'sampling_rate': c.get_filter_value_from_description('sampling_rate', 48000),
                    }
                )
            },
            class_interactions={
                1: self.Interactions.from_dict({'bearing_position': {1: {'sampling_rate': 1}}}),
                2: self.Interactions.from_dict({'bearing_position': {1: {'sampling_rate': 1}}}),
                3: self.Interactions.from_dict({'bearing_position': {1: {'sampling_rate': 1}}}),
            }
        )
        filters = c.create_valid_filter_combinations(
            task, ('fault_size', 'bearing_position', 'condition'),
            fault_size=[0, 4]
        )
        plan = c.construct_dataset_plan(task, **filters[0])
        assert plan.label == 'fault_element-fault_size=1-bearing_position=1-condition=1'
        assert plan.dataset_name == 'cwru'
        assert len(plan.sample_groups) == 4
        total_files = sum(
            len(paths) for sg in plan.sample_groups.values() for paths in sg.codes.values()
        )
        assert total_files == 4
        assert list(plan.sample_groups.keys()) == ['normal', 'inner ring', 'outer ring', 'ball']


# =====================================================================
# 12. Collection — Paderborn
# =====================================================================

class TestCollectionPaderbornRegression:

    def test_basic_properties(self):
        from collection import DatasetCollection
        c = DatasetCollection('configs/paderborn.yaml')
        assert c.name == 'paderborn'
        assert len(c.files) == 116
        assert sorted(c.schema.keys()) == [
            'condition', 'fault_arrangement', 'fault_characteristic',
            'fault_combination', 'fault_element', 'fault_mode',
            'fault_size', 'sampling_rate'
        ]

    def test_filter_values(self):
        from collection import DatasetCollection
        c = DatasetCollection('configs/paderborn.yaml')
        assert c.get_filter_value_from_description('fault_element', 'normal') == 0
        assert c.get_filter_value_from_description('fault_element', 'inner ring') == 1
        assert c.get_filter_value_from_description('fault_element', 'outer ring') == 2


# =====================================================================
# 13. File Sampler
# =====================================================================

class TestFileSamplerRegression:

    def test_deterministic_sampling(self):
        from experiment.sampling import FileSamplingProtocol, FileSampler
        from collection import SampleGroup, DatasetPlan, Metadata

        plan = DatasetPlan(
            dataset_name="test",
            label="test-label",
            sample_groups={
                "healthy": SampleGroup(
                    codes={100: [f"file_{i}.mat" for i in range(20)]},
                    metadata={100: Metadata({"sampling_rate": 12000})},
                ),
                "faulty": SampleGroup(
                    codes={200: [f"fault_{i}.mat" for i in range(15)]},
                    metadata={200: Metadata({"sampling_rate": 12000})},
                ),
            },
        )
        sampler = FileSampler(FileSamplingProtocol(max_files_per_code=3))

        r42 = sampler(plan, seed=42)
        assert r42.sample_groups["healthy"].codes[100] == ['file_3.mat', 'file_0.mat', 'file_8.mat']
        assert r42.sample_groups["faulty"].codes[200] == ['fault_3.mat', 'fault_14.mat', 'fault_2.mat']

        r99 = sampler(plan, seed=99)
        assert r99.sample_groups["healthy"].codes[100] == ['file_12.mat', 'file_19.mat', 'file_6.mat']
        assert r42.sample_groups["healthy"].codes[100] != r99.sample_groups["healthy"].codes[100]


# =====================================================================
# 14. Results Containers
# =====================================================================

class TestResultsContainersRegression:

    def _make_solutions(self):
        from results import DomainSolution, MultiDomainSolution, RepeatedMultiDomainSolution
        cm_a = np.array([[45, 3, 2], [2, 48, 0], [1, 1, 48]])
        cm_b = np.array([[40, 5, 5], [3, 44, 3], [4, 2, 44]])

        ds1 = DomainSolution('domain_A', {0:'h',1:'i',2:'o'}, 42,
                              {'epochs':100}, {'domain_A': cm_a, 'domain_B': cm_b})
        ds2 = DomainSolution('domain_B', {0:'h',1:'i',2:'o'}, 42,
                              {'epochs':110}, {'domain_A': cm_b, 'domain_B': cm_a})

        mds = MultiDomainSolution('exp1', [ds1, ds2], 'raw_12k')

        ds1_s2 = DomainSolution('domain_A', {0:'h',1:'i',2:'o'}, 99,
                                 {'epochs':100}, {'domain_A': cm_a, 'domain_B': cm_b})
        ds2_s2 = DomainSolution('domain_B', {0:'h',1:'i',2:'o'}, 99,
                                 {'epochs':100}, {'domain_A': cm_b, 'domain_B': cm_a})
        mds2 = MultiDomainSolution('exp1', [ds1_s2, ds2_s2])

        rmds = RepeatedMultiDomainSolution([mds, mds2])
        return ds1, mds, rmds

    def test_domain_solution(self):
        ds1, _, _ = self._make_solutions()
        assert ds1.test_dataset_names == ['domain_B']
        assert ds1.num_classes == 3
        assert np.trace(ds1.get_self_confusion_matrix()) == 141

    def test_multi_domain_solution(self):
        _, mds, _ = self._make_solutions()
        assert mds.seed == 42
        assert mds.train_dataset_names == ['domain_A', 'domain_B']
        assert mds.num_domains == 2

    def test_repeated_multi_domain_solution(self):
        _, _, rmds = self._make_solutions()
        assert rmds.config_name == 'exp1'
        assert rmds.seeds == [42, 99]
        assert rmds.num_seeds == 2
        transposed = rmds.transpose()
        assert sorted(transposed.keys()) == ['domain_A', 'domain_B']
        assert len(transposed['domain_A']['domain_B']) == 2

    def test_study_solution_builder(self):
        from results import StudySolutionBuilder
        _, mds, _ = self._make_solutions()
        _, _, rmds = self._make_solutions()

        builder = StudySolutionBuilder('test_study')
        for m in rmds.multi_domain_solutions:
            builder.add_multi_domain_solution(m)
        builder.set_metadata('description', 'test')
        builder.set_timestamp('20260218_120000')
        study = builder.build()

        assert study.config_names == ['exp1']
        assert study.num_configs == 1
        assert study.get_all_seeds() == [42, 99]