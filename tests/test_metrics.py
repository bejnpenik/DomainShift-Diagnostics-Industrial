"""
Tests for metrics.py

Covers: compute_accuracy, compute_precision_per_class, compute_recall_per_class,
        compute_f1_per_class, MetricsCalculator, ConfusionMatrixMetrics
"""

import pytest
import numpy as np

from results import (
    compute_accuracy,
    compute_precision_per_class,
    compute_recall_per_class,
    compute_f1_per_class,
    compute_support_per_class,
    compute_metrics,
    MetricsCalculator,
    ClassMetrics,
    ConfusionMatrixMetrics,
    DomainSolution, MultiDomainSolution
)



# =====================================================================
# Core metric functions
# =====================================================================

class TestComputeAccuracy:
    def test_perfect(self):
        cm = np.diag([50, 50, 50])
        assert compute_accuracy(cm) == pytest.approx(1.0)

    def test_zero(self):
        cm = np.array([[0, 50, 0], [50, 0, 0], [0, 0, 0]])
        assert compute_accuracy(cm) == pytest.approx(0.0)

    def test_known_value(self):
        cm = np.array([[45, 3, 2], [2, 48, 0], [1, 1, 48]])
        assert compute_accuracy(cm) == pytest.approx(141 / 150)

    def test_empty(self):
        cm = np.zeros((3, 3), dtype=int)
        assert compute_accuracy(cm) == 0.0

    def test_binary(self):
        cm = np.array([[90, 10], [20, 80]])
        assert compute_accuracy(cm) == pytest.approx(170 / 200)


class TestComputePrecision:
    def test_perfect(self):
        cm = np.diag([50, 50])
        prec = compute_precision_per_class(cm)
        np.testing.assert_allclose(prec, [1.0, 1.0], atol=1e-6)

    def test_known_values(self):
        cm = np.array([[40, 10], [5, 45]])
        prec = compute_precision_per_class(cm)
        # class 0: 40 / (40+5) = 40/45
        # class 1: 45 / (10+45) = 45/55
        assert prec[0] == pytest.approx(40 / 45, abs=1e-5)
        assert prec[1] == pytest.approx(45 / 55, abs=1e-5)

    def test_3class(self):
        cm = np.array([[45, 3, 2], [2, 48, 0], [1, 1, 48]])
        prec = compute_precision_per_class(cm)
        # class 0: 45/(45+2+1) = 45/48
        assert prec[0] == pytest.approx(45 / 48, abs=1e-5)


class TestComputeRecall:
    def test_perfect(self):
        cm = np.diag([50, 50])
        rec = compute_recall_per_class(cm)
        np.testing.assert_allclose(rec, [1.0, 1.0], atol=1e-6)

    def test_known_values(self):
        cm = np.array([[40, 10], [5, 45]])
        rec = compute_recall_per_class(cm)
        # class 0: 40/50
        # class 1: 45/50
        assert rec[0] == pytest.approx(40 / 50, abs=1e-5)
        assert rec[1] == pytest.approx(45 / 50, abs=1e-5)


class TestComputeF1:
    def test_perfect(self):
        cm = np.diag([50, 50])
        f1 = compute_f1_per_class(cm)
        np.testing.assert_allclose(f1, [1.0, 1.0], atol=1e-5)

    def test_matches_formula(self):
        cm = np.array([[40, 10], [5, 45]])
        f1 = compute_f1_per_class(cm)
        prec = compute_precision_per_class(cm)
        rec = compute_recall_per_class(cm)
        expected = 2 * prec * rec / (prec + rec + 1e-10)
        np.testing.assert_allclose(f1, expected, atol=1e-6)


class TestComputeSupport:
    def test_known(self):
        cm = np.array([[40, 10], [5, 45]])
        support = compute_support_per_class(cm)
        assert support[0] == 50
        assert support[1] == 50


# =====================================================================
# MetricsCalculator & containers
# =====================================================================

class TestMetricsCalculator:
    def test_from_confusion_matrix(self):
        cm = np.array([[45, 3, 2], [2, 48, 0], [1, 1, 48]])
        labels = {0: "healthy", 1: "inner", 2: "outer"}
        metrics = compute_metrics(cm, labels)

        assert metrics.accuracy == pytest.approx(141 / 150)
        assert metrics.num_classes == 3
        assert metrics.class_metrics[0].class_label == "healthy"
        assert metrics.class_metrics[1].class_label == "inner"

    def test_auto_labels(self):
        cm = np.diag([10, 20])
        metrics = compute_metrics(cm)
        assert metrics.class_metrics[0].class_label == "class_0"

    def test_to_dict(self):
        cm = np.array([[45, 5], [3, 47]])
        metrics = compute_metrics(cm)
        d = metrics.to_dict()
        assert "accuracy" in d
        assert "precision_0" in d
        assert "recall_1" in d
        assert "f1_0" in d
        assert "support_0" in d

    def test_get_class(self):
        cm = np.array([[45, 5], [3, 47]])
        metrics = compute_metrics(cm, {0: "h", 1: "f"})
        c0 = metrics.get_class(0)
        assert c0.class_label == "h"
        with pytest.raises(KeyError):
            metrics.get_class(5)

    def test_from_domain_solution(self):
        cm = np.array([[45, 5], [3, 47]])
        ds = DomainSolution(
            train_dataset_name="A",
            class_labels={0: "h", 1: "f"},
            seed=42,
            train_metadata={},
            confusion_matrices={"A": cm, "B": cm},
        )
        calc = MetricsCalculator()
        result = calc.from_domain_solution(ds)
        assert "A" in result
        assert "B" in result
        assert result["A"].accuracy == result["B"].accuracy

    def test_from_multi_domain_solution(self):
        cm = np.array([[45, 5], [3, 47]])
        ds1 = DomainSolution("A", {0: "h", 1: "f"}, 42, {}, {"A": cm, "B": cm})
        ds2 = DomainSolution("B", {0: "h", 1: "f"}, 42, {}, {"A": cm, "B": cm})
        mds = MultiDomainSolution("cfg", [ds1, ds2])
        calc = MetricsCalculator()
        result = calc.from_multi_domain_solution(mds)
        assert "A" in result
        assert "B" in result
