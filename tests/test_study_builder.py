"""
Tests for study.builder — YAML-driven study configuration loading.

Covers: load_study_config, iter_resolved_grid_points against
cwru_modules_study.yaml's model-architecture comparison grid.
"""

from __future__ import annotations

from pathlib import Path

from study.builder import load_study_config, iter_resolved_grid_points
from model.config import ModelConfig

_STUDY_YAML = Path(__file__).resolve().parent.parent / "configs" / "study" / "cwru_modules_study.yaml"


class TestCwruModulesStudyGrid:
    """Grid-resolution only -- no DatasetCollection/task/data involved."""

    def test_grid_resolves_to_ten_model_configs(self):
        cfg = load_study_config(_STUDY_YAML)
        assert cfg.num_grid_points == 10

        points = iter_resolved_grid_points(cfg)
        assert len(points) == 10
        for point in points:
            assert isinstance(point["model_config"], ModelConfig)

    def test_all_ten_variants_are_distinct_and_expected(self):
        cfg = load_study_config(_STUDY_YAML)
        points = iter_resolved_grid_points(cfg)

        names = {point["model_config"].name for point in points}
        assert names == {
            "cnn1d_1x1", "cnn1d_res", "cnn1d_se", "cnn1d_eca", "cnn1d_res_se",
            "cnn2d_1x1", "cnn2d_res", "cnn2d_se", "cnn2d_eca", "cnn2d_res_se",
        }

    def test_fixed_factors_resolve_consistently(self):
        cfg = load_study_config(_STUDY_YAML)
        points = iter_resolved_grid_points(cfg)

        for point in points:
            assert point["trainer_config"].lr == 0.001
            assert point["trainer_config"].optimizer_name == "adamw"
            assert point["sampling_rate"] == 12000
