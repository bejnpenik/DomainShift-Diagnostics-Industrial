"""
Tests for results.py

Covers: DomainSolution, MultiDomainSolution, 
        RepeatedMultiDomainSolution, StudySolution, StudySolutionBuilder
"""

import pytest
import numpy as np

from results import (
    DomainSolution,
    MultiDomainSolution,
    RepeatedMultiDomainSolution,
    StudySolution,
    StudySolutionBuilder,
)


def _cm(n_classes=3):
    """Helper: random-ish confusion matrix."""
    cm = np.diag([40, 45, 48][:n_classes])
    # Add some off-diagonal noise
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                cm[i, j] = np.random.randint(0, 5)
    return cm


# =====================================================================
# DomainSolution
# =====================================================================

class TestDomainSolution:
    def test_basic(self):
        cm = _cm(3)
        ds = DomainSolution("A", {0: "h", 1: "i", 2: "o"}, 42, {}, {"A": cm, "B": cm})
        assert ds.train_dataset_name == "A"
        assert ds.num_classes == 3
        assert ds.seed == 42

    def test_requires_self_eval(self):
        cm = _cm(3)
        with pytest.raises(ValueError, match="self-evaluation"):
            DomainSolution("A", {0: "h"}, 42, {}, {"B": cm})

    def test_test_dataset_names(self):
        cm = _cm(2)
        ds = DomainSolution("A", {0: "h", 1: "f"}, 42, {}, {"A": cm, "B": cm, "C": cm})
        assert ds.test_dataset_names == ["B", "C"]
        assert set(ds.all_test_dataset_names) == {"A", "B", "C"}

    def test_get_confusion_matrix(self):
        cm = _cm(2)
        ds = DomainSolution("A", {0: "h", 1: "f"}, 42, {}, {"A": cm})
        np.testing.assert_array_equal(ds.get_confusion_matrix("A"), cm)
        with pytest.raises(KeyError):
            ds.get_confusion_matrix("NONEXISTENT")

    def test_get_self_confusion_matrix(self):
        cm = _cm(2)
        ds = DomainSolution("A", {0: "h", 1: "f"}, 42, {}, {"A": cm})
        np.testing.assert_array_equal(ds.get_self_confusion_matrix(), cm)

    def test_train_metadata(self):
        cm = _cm(2)
        meta = {"epochs": 100, "loss": 0.5}
        ds = DomainSolution("A", {0: "h", 1: "f"}, 42, meta, {"A": cm})
        assert ds.train_metadata["epochs"] == 100


# =====================================================================
# MultiDomainSolution
# =====================================================================

class TestMultiDomainSolution:
    def test_basic(self):
        cm = _cm(2)
        ds1 = DomainSolution("A", {0: "h", 1: "f"}, 42, {}, {"A": cm, "B": cm})
        ds2 = DomainSolution("B", {0: "h", 1: "f"}, 42, {}, {"A": cm, "B": cm})
        mds = MultiDomainSolution("cfg1", [ds1, ds2])
        assert mds.seed == 42
        assert mds.num_domains == 2
        assert set(mds.train_dataset_names) == {"A", "B"}

    def test_requires_same_seed(self):
        cm = _cm(2)
        ds1 = DomainSolution("A", {0: "h"}, 42, {}, {"A": cm})
        ds2 = DomainSolution("B", {0: "h"}, 99, {}, {"B": cm})
        with pytest.raises(ValueError, match="same seed"):
            MultiDomainSolution("cfg", [ds1, ds2])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            MultiDomainSolution("cfg", [])

    def test_get_solution(self):
        cm = _cm(2)
        ds1 = DomainSolution("A", {0: "h", 1: "f"}, 42, {}, {"A": cm})
        mds = MultiDomainSolution("cfg", [ds1])
        assert mds.get_solution("A") is ds1
        with pytest.raises(KeyError):
            mds.get_solution("NONEXISTENT")

    def test_get_cross_domain_matrix(self):
        cm = _cm(2)
        ds1 = DomainSolution("A", {0: "h", 1: "f"}, 42, {}, {"A": cm, "B": cm})
        mds = MultiDomainSolution("cfg", [ds1])
        cdm = mds.get_cross_domain_matrix()
        assert "A" in cdm
        assert "B" in cdm["A"]


# =====================================================================
# RepeatedMultiDomainSolution
# =====================================================================

class TestRepeatedMultiDomainSolution:
    def _make_rmds(self, seeds=(42, 99)):
        cm = _cm(2)
        solutions = []
        for seed in seeds:
            ds1 = DomainSolution("A", {0: "h", 1: "f"}, seed, {}, {"A": cm, "B": cm})
            ds2 = DomainSolution("B", {0: "h", 1: "f"}, seed, {}, {"A": cm, "B": cm})
            solutions.append(MultiDomainSolution("cfg", [ds1, ds2]))
        return RepeatedMultiDomainSolution(solutions)

    def test_basic(self):
        rmds = self._make_rmds()
        assert rmds.config_name == "cfg"
        assert rmds.seeds == [42, 99]
        assert rmds.num_seeds == 2
        assert set(rmds.train_dataset_names) == {"A", "B"}

    def test_requires_unique_seeds(self):
        cm = _cm(2)
        ds = DomainSolution("A", {0: "h"}, 42, {}, {"A": cm})
        mds1 = MultiDomainSolution("cfg", [ds])
        mds2 = MultiDomainSolution("cfg", [DomainSolution("A", {0: "h"}, 42, {}, {"A": cm})])
        with pytest.raises(ValueError, match="unique"):
            RepeatedMultiDomainSolution([mds1, mds2])

    def test_requires_same_config(self):
        cm = _cm(2)
        ds1 = DomainSolution("A", {0: "h"}, 42, {}, {"A": cm})
        ds2 = DomainSolution("A", {0: "h"}, 99, {}, {"A": cm})
        mds1 = MultiDomainSolution("cfg1", [ds1])
        mds2 = MultiDomainSolution("cfg2", [ds2])
        with pytest.raises(ValueError):
            RepeatedMultiDomainSolution([mds1, mds2])

    def test_get_by_seed(self):
        rmds = self._make_rmds()
        mds = rmds.get_by_seed(42)
        assert mds.seed == 42
        with pytest.raises(KeyError):
            rmds.get_by_seed(123)

    def test_get_confusion_matrices_for_pair(self):
        rmds = self._make_rmds()
        cms = rmds.get_confusion_matrices_for_pair("A", "B")
        assert len(cms) == 2
        assert all(isinstance(cm, np.ndarray) for cm in cms)

    def test_transpose(self):
        rmds = self._make_rmds()
        t = rmds.transpose()
        assert "A" in t and "B" in t
        assert "A" in t["A"] and "B" in t["A"]
        assert len(t["A"]["B"]) == 2  # 2 seeds


# =====================================================================
# StudySolution & Builder
# =====================================================================

class TestStudySolution:
    def test_basic(self):
        cm = _cm(2)
        ds = DomainSolution("A", {0: "h", 1: "f"}, 42, {}, {"A": cm})
        mds = MultiDomainSolution("cfg1", [ds])
        rmds = RepeatedMultiDomainSolution([mds])
        study = StudySolution("my_study", "20250101", [rmds])
        assert study.study_name == "my_study"
        assert study.config_names == ["cfg1"]
        assert study.num_configs == 1

    def test_requires_unique_configs(self):
        cm = _cm(2)
        ds1 = DomainSolution("A", {0: "h"}, 42, {}, {"A": cm})
        ds2 = DomainSolution("A", {0: "h"}, 99, {}, {"A": cm})
        mds1 = MultiDomainSolution("cfg", [ds1])
        mds2 = MultiDomainSolution("cfg", [ds2])
        rmds1 = RepeatedMultiDomainSolution([mds1])
        rmds2 = RepeatedMultiDomainSolution([mds2])
        with pytest.raises(ValueError, match="unique"):
            StudySolution("s", "t", [rmds1, rmds2])

    def test_get_by_config(self):
        cm = _cm(2)
        ds = DomainSolution("A", {0: "h"}, 42, {}, {"A": cm})
        mds = MultiDomainSolution("cfg1", [ds])
        rmds = RepeatedMultiDomainSolution([mds])
        study = StudySolution("s", "t", [rmds])
        assert study.get_by_config("cfg1") is rmds
        with pytest.raises(KeyError):
            study.get_by_config("NONEXISTENT")


class TestStudySolutionBuilder:
    def test_build(self):
        cm = _cm(2)
        ds1 = DomainSolution("A", {0: "h", 1: "f"}, 42, {}, {"A": cm})
        mds1 = MultiDomainSolution("cfg1", [ds1])
        ds2 = DomainSolution("A", {0: "h", 1: "f"}, 99, {}, {"A": cm})
        mds2 = MultiDomainSolution("cfg1", [ds2])

        builder = StudySolutionBuilder("test_study")
        builder.add_multi_domain_solution(mds1)
        builder.add_multi_domain_solution(mds2)
        study = builder.build()

        assert study.study_name == "test_study"
        assert len(study.repeated_solutions) == 1
        assert study.repeated_solutions[0].num_seeds == 2
