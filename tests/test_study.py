"""
Tests for study.py and study_grid.py

Covers: StudyGridBuilder, ExperimentSpec, StudyDesign, StudyAnalyzer
"""

import pytest
import numpy as np

from collection import Task, Rule
from results import DomainSolution, MultiDomainSolution, RepeatedMultiDomainSolution, StudySolution
from study_grid import StudyGridBuilder
from study import ExperimentSpec, StudyDesign, StudyAnalyzer


# =====================================================================
# StudyGridBuilder
# =====================================================================

class TestStudyGridBuilder:
    def test_single_factor(self):
        b = StudyGridBuilder()
        b.set_factors(normalization=("dataset", "sample"))
        configs = b.build()
        assert len(configs) == 2

    def test_multiple_factors(self):
        b = StudyGridBuilder()
        b.set_factors(
            normalization=("dataset", "sample"),
            model_name=("1x1", "2x2"),
        )
        configs = b.build()
        assert len(configs) == 4

    def test_independent_factors(self):
        b = StudyGridBuilder()
        b.set_factors(x=("a", "b"))
        b.set_independent(lr=1e-3, batch_size=32)
        configs = b.build()
        assert all(c["lr"] == 1e-3 for c in configs)
        assert all(c["batch_size"] == 32 for c in configs)

    def test_dependent_factors(self):
        b = StudyGridBuilder()
        b.set_factors(normalization=("dataset", "sample"))
        b.set_dependent(
            "lr_adjusted",
            depends_on="normalization",
            mapping={"dataset": 0.1, "sample": 0.01},
        )
        configs = b.build()
        dataset_cfgs = [c for c in configs if c["normalization"] == "dataset"]
        sample_cfgs = [c for c in configs if c["normalization"] == "sample"]
        assert all(c["lr_adjusted"] == 0.1 for c in dataset_cfgs)
        assert all(c["lr_adjusted"] == 0.01 for c in sample_cfgs)

    def test_num_combinations(self):
        b = StudyGridBuilder()
        b.set_factors(a=(1, 2, 3), b=("x", "y"))
        assert b.num_combinations == 6

    def test_empty_factors(self):
        b = StudyGridBuilder()
        b.set_independent(lr=0.01)
        configs = b.build()
        assert len(configs) == 1
        assert configs[0]["lr"] == 0.01

    def test_name_generation(self):
        b = StudyGridBuilder()
        b.set_factors(x=("a", "b"))
        configs = b.build()
        # Each config should have a 'name' key
        assert all("name" in c for c in configs)


# =====================================================================
# StudyDesign
# =====================================================================

class TestStudyDesign:
    def _make_design(self):
        """Helper to create a simple StudyDesign."""
        task = Task(target="x", defaults=Rule(fixed={"y": 0, "z": 0}))
        from experiment import ExperimentConfig
        from modality_processor import SignalProcessorConfig
        from cnn import CNN1D

        cfg = SignalProcessorConfig.raw("raw", 12000)
        exp_config = ExperimentConfig(
            name="test_config",
            processor_config=cfg,
            processor_name="raw",
            model_name="CNN1D",
            model_class=CNN1D,
        )
        spec = ExperimentSpec(
            task=task,
            filter_combinations=({"y": 0, "z": 0},),
            config=exp_config,
        )
        return StudyDesign(
            name="test_study",
            experiment_specs=(spec,),
            seeds=(42, 99),
        )

    def test_basic_properties(self):
        design = self._make_design()
        assert design.name == "test_study"
        assert design.num_configs == 1
        assert design.num_seeds == 2

    def test_duplicate_names_raise(self):
        task = Task(target="x", defaults=Rule(fixed={"y": 0}))
        from experiment import ExperimentConfig
        from modality_processor import SignalProcessorConfig
        from cnn import CNN1D

        cfg = SignalProcessorConfig.raw("raw", 12000)
        exp1 = ExperimentConfig(name="same_name", processor_config=cfg,
                                processor_name="r", model_name="m", model_class=CNN1D)
        exp2 = ExperimentConfig(name="same_name", processor_config=cfg,
                                processor_name="r", model_name="m", model_class=CNN1D)
        spec1 = ExperimentSpec(task=task, filter_combinations=({},), config=exp1)
        spec2 = ExperimentSpec(task=task, filter_combinations=({},), config=exp2)
        with pytest.raises(ValueError, match="unique"):
            StudyDesign("s", (spec1, spec2), (42,))


# =====================================================================
# StudyAnalyzer
# =====================================================================

class TestStudyAnalyzer:
    def test_accuracy_summary(self):
        cm = np.diag([45, 48, 47])
        ds_a = DomainSolution("A", {0: "h", 1: "i", 2: "o"}, 42, {}, {"A": cm, "B": cm})
        ds_b = DomainSolution("B", {0: "h", 1: "i", 2: "o"}, 42, {}, {"A": cm, "B": cm})
        mds = MultiDomainSolution("cfg1", [ds_a, ds_b])
        rmds = RepeatedMultiDomainSolution([mds])
        study = StudySolution("test", "t", [rmds])

        analyzer = StudyAnalyzer(study)
        summary = analyzer.get_accuracy_summary()
        assert "cfg1" in summary
        assert "self_eval_mean" in summary["cfg1"]
        assert "cross_domain_mean" in summary["cfg1"]
        assert summary["cfg1"]["self_eval_mean"] > 0

    def test_compare_configs(self):
        cm = np.diag([45, 48])
        ds1 = DomainSolution("A", {0: "h", 1: "f"}, 42, {}, {"A": cm})
        mds1 = MultiDomainSolution("cfg1", [ds1])
        rmds1 = RepeatedMultiDomainSolution([mds1])

        cm2 = np.diag([30, 35])
        ds2 = DomainSolution("A", {0: "h", 1: "f"}, 42, {}, {"A": cm2})
        mds2 = MultiDomainSolution("cfg2", [ds2])
        rmds2 = RepeatedMultiDomainSolution([mds2])

        study = StudySolution("test", "t", [rmds1, rmds2])
        analyzer = StudyAnalyzer(study)
        text = analyzer.compare_configs()
        assert "cfg1" in text
        assert "cfg2" in text
