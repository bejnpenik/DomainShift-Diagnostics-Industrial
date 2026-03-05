"""
Tests for collection.py

Covers: Metadata, Rule, InteractionRule, Interactions, Task, 
        SampleGroup, DatasetPlan, DatasetCollection (where possible).

No data files needed — all synthetic.
"""

import pytest
import numpy as np
from collection import (
    Metadata, Rule, InteractionRule, Interactions,
    Task, SampleGroup, DatasetPlan,
)


# =====================================================================
# Metadata
# =====================================================================

class TestMetadata:
    def test_attribute_access(self):
        m = Metadata({"bearing_position": "DE", "sampling_rate": 12000})
        assert m.bearing_position == "DE"
        assert m.sampling_rate == 12000

    def test_dict_access(self):
        m = Metadata({"key": "value"})
        assert m["key"] == "value"

    def test_missing_attribute_raises(self):
        m = Metadata({"a": 1})
        with pytest.raises(AttributeError):
            _ = m.nonexistent

    def test_missing_key_raises(self):
        m = Metadata({"a": 1})
        with pytest.raises(KeyError):
            _ = m["nonexistent"]

    def test_len(self):
        m = Metadata({"a": 1, "b": 2, "c": 3})
        assert len(m) == 3

    def test_iter_keys(self):
        m = Metadata({"x": 10, "y": 20})
        assert set(m.keys()) == {"x", "y"}

    def test_items(self):
        m = Metadata({"a": 1})
        assert list(m.items()) == [("a", 1)]

    def test_repr(self):
        m = Metadata({"a": 1})
        assert "Metadata" in repr(m)

    def test_frozen(self):
        m = Metadata({"a": 1})
        with pytest.raises(AttributeError):
            m._data = {"b": 2}

    def test_empty(self):
        m = Metadata({})
        assert len(m) == 0


# =====================================================================
# Rule
# =====================================================================

class TestRule:
    def test_default_empty(self):
        r = Rule()
        assert r.fixed == {}
        assert r.resolve == {}

    def test_with_values(self):
        r = Rule(fixed={"a": 1}, resolve={"b": [1, 2]})
        assert r.fixed["a"] == 1
        assert r.resolve["b"] == [1, 2]

    def test_frozen(self):
        r = Rule(fixed={"a": 1})
        with pytest.raises(AttributeError):
            r.fixed = {}


# =====================================================================
# InteractionRule
# =====================================================================

class TestInteractionRule:
    def test_from_scalar(self):
        rule = InteractionRule.from_constraint("fe", 0, "fs", 0)
        assert rule.allowed_values == frozenset([0])

    def test_from_list(self):
        rule = InteractionRule.from_constraint("fe", 1, "fs", [1, 2, 3])
        assert rule.allowed_values == frozenset([1, 2, 3])

    def test_applies_to(self):
        rule = InteractionRule.from_constraint("fe", 0, "fs", 0)
        assert rule.applies_to({"fe": 0, "fs": 0})
        assert not rule.applies_to({"fe": 1, "fs": 0})

    def test_is_satisfied_by(self):
        rule = InteractionRule.from_constraint("fe", 0, "fs", [0, 1])
        assert rule.is_satisfied_by({"fe": 0, "fs": 0})
        assert rule.is_satisfied_by({"fe": 0, "fs": 1})
        assert not rule.is_satisfied_by({"fe": 0, "fs": 2})


# =====================================================================
# Interactions
# =====================================================================

class TestInteractions:
    def test_empty(self):
        ints = Interactions.from_dict({})
        assert ints.is_satisfied_by({"anything": 42})

    def test_simple_constraint(self):
        ints = Interactions.from_dict({
            "fault_element": {0: {"fault_size": 0}}
        })
        assert ints.is_satisfied_by({"fault_element": 0, "fault_size": 0})
        assert not ints.is_satisfied_by({"fault_element": 0, "fault_size": 1})
        # Rule doesn't apply when trigger doesn't match
        assert ints.is_satisfied_by({"fault_element": 1, "fault_size": 999})

    def test_list_constraint(self):
        ints = Interactions.from_dict({
            "bearing_position": {1: {"sampling_rate": [1, 2]}}
        })
        assert ints.is_satisfied_by({"bearing_position": 1, "sampling_rate": 1})
        assert ints.is_satisfied_by({"bearing_position": 1, "sampling_rate": 2})
        assert not ints.is_satisfied_by({"bearing_position": 1, "sampling_rate": 3})

    def test_multiple_constraints(self):
        ints = Interactions.from_dict({
            "fault_element": {
                0: {"fault_size": 0, "fault_position": 0},
                1: {"fault_size": [1, 2, 3]},
            }
        })
        # fault_element=0 requires both fault_size=0 AND fault_position=0
        assert ints.is_satisfied_by({"fault_element": 0, "fault_size": 0, "fault_position": 0})
        assert not ints.is_satisfied_by({"fault_element": 0, "fault_size": 0, "fault_position": 1})
        assert not ints.is_satisfied_by({"fault_element": 0, "fault_size": 1, "fault_position": 0})
        # fault_element=1 requires fault_size in {1,2,3}
        assert ints.is_satisfied_by({"fault_element": 1, "fault_size": 2, "fault_position": 5})
        assert not ints.is_satisfied_by({"fault_element": 1, "fault_size": 0, "fault_position": 0})


# =====================================================================
# Task
# =====================================================================

class TestTask:
    def test_basic(self):
        t = Task(
            target="fault_element",
            domain_factors = ("fault_size", ),
            defaults=Rule(
                fixed={"fault_size": 0},
                resolve={"sampling_rate": [1, 2]},
            ),
        )
        assert t.target == "fault_element"

    def test_target_in_defaults_fixed_raises(self):
        with pytest.raises(ValueError):
            Task(target="fault_element",
                 domain_factors=("fault_size", ), 
                 defaults=Rule(fixed={"fault_element": 0}))

    def test_target_in_defaults_resolve_raises(self):
        with pytest.raises(ValueError):
            Task(target="fault_element",
                 domain_factors=("fault_size", ),
                 defaults=Rule(resolve={"fault_element": [1]}))

    def test_duplicate_fixed_and_resolve_raises(self):
        with pytest.raises(ValueError, match="both in default fixed and resolves"):
            Task(
                target="x",
                domain_factors=("a", "b"),
                defaults=Rule(fixed={"a": 0}, resolve={"a": [1, 2]}),
            )

    def test_class_duplicate_fixed_resolve_raises(self):
        with pytest.raises(ValueError, match="both in fixed and resolves"):
            Task(
                target="x",
                domain_factors=("a", "b", "c"),
                defaults=Rule(fixed={"b": 0}),
                classes={0: Rule(fixed={"c": 1}, resolve={"c": [2]})},
            )

    def test_label_generation(self):
        t = Task(target="fault_element", domain_factors=("fault_size",))
        label = t.label(fault_size=1, condition=2)
        assert "fault_element" in label
        assert "fault_size=1" in label
        assert "condition=2" in label

    def test_label_prefix(self):
        t = Task(target="target_factor", domain_factors=("fault_size",))
        label = t.label()
        assert label.startswith("target_factor")

    def test_with_interactions(self):
        ints = Interactions.from_dict({"fe": {0: {"fs": 0}}})
        t = Task(
            target="x",
            domain_factors=("a",),
            defaults=Rule(fixed={"fe": 0, "fs": 0}),
            interactions=ints,
        )
        assert t.interactions is not None

    def test_with_class_interactions(self):
        ci = {0: Interactions.from_dict({"bp": {1: {"sr": 1}}})}
        t = Task(
            target="x",
            domain_factors=("a", "b"),
            defaults=Rule(fixed={"bp": 0, "sr": 0}),
            class_interactions=ci,
        )
        assert t.class_interactions is not None


# =====================================================================
# SampleGroup & DatasetPlan
# =====================================================================

class TestSampleGroup:
    def test_basic(self):
        sg = SampleGroup(
            codes={100: ["file1.mat", "file2.mat"]},
            metadata={100: Metadata({"desc": "test"})},
        )
        assert 100 in sg.codes
        assert len(sg.codes[100]) == 2

    def test_empty(self):
        sg = SampleGroup(codes={}, metadata={})
        assert len(sg.codes) == 0


class TestDatasetPlan:
    def test_complete(self):
        plan = DatasetPlan(
            dataset_name="test",
            label="test-label",
            sample_groups={
                "healthy": SampleGroup({100: ["f1"]}, {100: Metadata({})}),
                "faulty": SampleGroup({200: ["f2"]}, {200: Metadata({})}),
            },
        )
        assert plan.is_complete
        assert plan.empty_classes == []

    def test_incomplete(self):
        plan = DatasetPlan(
            dataset_name="test",
            label="test-label",
            sample_groups={
                "healthy": SampleGroup({100: ["f1"]}, {100: Metadata({})}),
                "faulty": SampleGroup({}, {}),
            },
        )
        assert not plan.is_complete
        assert "faulty" in plan.empty_classes

    def test_class_sample_counts(self):
        plan = DatasetPlan(
            dataset_name="test",
            label="test-label",
            sample_groups={
                "healthy": SampleGroup({100: ["f1"], 101: ["f2"]}, {100: Metadata({}), 101: Metadata({})}),
                "faulty": SampleGroup({200: ["f3"]}, {200: Metadata({})}),
            },
        )
        counts = plan.class_sample_counts
        assert counts["healthy"] == 2
        assert counts["faulty"] == 1

"""
Tests for task builder — YAML-driven task configuration.

Covers:
    - build_task from dict (CWRU-style, Paderborn-style)
    - domain_factors at top level
    - Resolution of string descriptions via collection
    - "all" keyword resolution
    - Class key resolution (description -> code)
    - Interactions building
    - Defaults and missing sections
    - Validation errors
    - build_task_and_filters (Task + filter combinations)
    - YAML file roundtrip
    - Equivalence with direct Python construction
"""

import pytest
import tempfile
from pathlib import Path


# =====================================================================
# Mock collection
# =====================================================================

class MockCollection:
    """Mimics DatasetCollection's resolution interface."""

    def __init__(self, schema: dict[str, dict[int, str]]):
        self._schema = schema

    def get_filter_value_from_description(self, filter_name: str, description) -> int:
        if filter_name not in self._schema:
            raise ValueError(f"Unknown filter: {filter_name}")
        desc_str = str(description)
        for code, desc in self._schema[filter_name].items():
            if str(desc) == desc_str:
                return code
        raise ValueError(
            f"Description '{description}' not found for filter '{filter_name}'"
        )

    def get_all_filter_values(self, filter_name: str) -> list[int]:
        if filter_name not in self._schema:
            raise ValueError(f"Unknown filter: {filter_name}")
        return list(self._schema[filter_name].keys())

    def create_valid_filter_combinations(self, task, depends, **excludes):
        return ({"dummy": 1},)


@pytest.fixture
def cwru_mock():
    return MockCollection({
        "fault_element": {0: "normal", 1: "inner ring", 2: "outer ring", 3: "ball"},
        "fault_size": {0: "none", 1: "7mil", 2: "14mil", 3: "21mil"},
        "bearing_position": {0: "none", 1: "drive_end", 2: "fan_end"},
        "condition": {0: "none", 1: "0hp", 2: "1hp", 3: "2hp", 4: "3hp"},
        "sampling_rate": {1: "12000", 2: "48000"},
        "fault_position": {0: "normal", 1: "centered", 2: "orthogonal", 3: "opposite"},
    })


@pytest.fixture
def paderborn_mock():
    return MockCollection({
        "fault_element": {0: "normal", 1: "inner ring", 2: "outer ring"},
        "fault_size": {0: "none", 1: "small", 2: "medium", 3: "large"},
        "fault_arrangement": {0: "none", 1: "single", 2: "multiple"},
        "condition": {0: "none", 1: "N15_M07_F10", 2: "N09_M07_F10", 3: "N15_M01_F10", 4: "N15_M07_F04"},
        "fault_combination": {0: "none", 1: "combined"},
        "fault_mode": {0: "none", 1: "artificial", 2: "real"},
        "fault_characteristic": {0: "none", 1: "single_point", 2: "distributed"},
        "sampling_rate": {1: "64000"},
    })


def _write_yaml(cfg: dict) -> Path:
    import yaml
    path = Path(tempfile.mktemp(suffix=".yaml"))
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


# =====================================================================
# 1. Basic task building
# =====================================================================

class TestBuildTaskBasic:

    def test_minimal_task(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {"target": "fault_element"}
        task = build_task(cfg, cwru_mock)
        assert task.target == "fault_element"
        assert task.domain_factors == ()

    def test_missing_target_raises(self, cwru_mock):
        from collection.task_builder import build_task

        with pytest.raises(ValueError, match="'target'"):
            build_task({}, cwru_mock)

    def test_target_preserved(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "defaults": {"fixed": {"fault_size": 0}},
        }
        task = build_task(cfg, cwru_mock)
        assert task.target == "fault_element"


# =====================================================================
# 2. domain_factors
# =====================================================================

class TestDomainFactors:

    def test_domain_factors_list(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "domain_factors": ["fault_size", "bearing_position", "condition"],
        }
        task = build_task(cfg, cwru_mock)
        assert task.domain_factors == ("fault_size", "bearing_position", "condition")

    def test_domain_factors_single_string(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "domain_factors": "condition",
        }
        task = build_task(cfg, cwru_mock)
        assert task.domain_factors == ("condition",)

    def test_domain_factors_empty(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {"target": "fault_element"}
        task = build_task(cfg, cwru_mock)
        assert task.domain_factors == ()

    def test_domain_factors_paderborn(self, paderborn_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "domain_factors": [
                "fault_size", "fault_arrangement", "condition",
                "fault_combination", "fault_mode", "fault_characteristic",
            ],
        }
        task = build_task(cfg, paderborn_mock)
        assert len(task.domain_factors) == 6
        assert "fault_arrangement" in task.domain_factors


# =====================================================================
# 3. Defaults / Rule resolution
# =====================================================================

class TestDefaultsResolution:

    def test_fixed_values_pass_through(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "defaults": {
                "fixed": {"fault_size": 0, "bearing_position": 0, "condition": 0},
            },
        }
        task = build_task(cfg, cwru_mock)
        assert task.defaults.fixed == {"fault_size": 0, "bearing_position": 0, "condition": 0}

    def test_resolve_string_description(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "defaults": {
                "resolve": {"fault_position": "centered"},
            },
        }
        task = build_task(cfg, cwru_mock)
        assert task.defaults.resolve["fault_position"] == 1

    def test_resolve_int_passes_through(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "defaults": {
                "resolve": {"sampling_rate": [1, 2]},
            },
        }
        task = build_task(cfg, cwru_mock)
        assert task.defaults.resolve["sampling_rate"] == [1, 2]

    def test_resolve_all_keyword(self, paderborn_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "defaults": {
                "resolve": {"sampling_rate": "all"},
            },
        }
        task = build_task(cfg, paderborn_mock)
        assert task.defaults.resolve["sampling_rate"] == [1]

    def test_resolve_all_multiple_values(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "defaults": {
                "resolve": {"sampling_rate": "all"},
            },
        }
        task = build_task(cfg, cwru_mock)
        assert sorted(task.defaults.resolve["sampling_rate"]) == [1, 2]

    def test_empty_defaults(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {"target": "fault_element"}
        task = build_task(cfg, cwru_mock)
        assert task.defaults.fixed == {}
        assert task.defaults.resolve == {}


# =====================================================================
# 4. Class resolution
# =====================================================================

class TestClassResolution:

    def test_class_key_resolved_from_description(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "classes": {
                "normal": {"fixed": {"fault_size": 0}},
            },
        }
        task = build_task(cfg, cwru_mock)
        assert 0 in task.classes
        assert task.classes[0].fixed == {"fault_size": 0}

    def test_class_key_int_passes_through(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "classes": {
                0: {"fixed": {"fault_size": 0}},
            },
        }
        task = build_task(cfg, cwru_mock)
        assert 0 in task.classes

    def test_class_with_resolve(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "classes": {
                "normal": {
                    "fixed": {"fault_size": 0},
                    "resolve": {
                        "fault_position": "normal",
                        "sampling_rate": 48000,
                    },
                },
            },
        }
        task = build_task(cfg, cwru_mock)
        normal_rule = task.classes[0]
        assert normal_rule.resolve["fault_position"] == 0
        assert normal_rule.resolve["sampling_rate"] == 48000

    def test_multiple_classes(self, paderborn_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "classes": {
                "normal": {"fixed": {"fault_size": 0}},
                "inner ring": {"fixed": {"fault_size": 0}},
            },
        }
        task = build_task(cfg, paderborn_mock)
        assert 0 in task.classes
        assert 1 in task.classes

    def test_no_classes_section(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {"target": "fault_element"}
        task = build_task(cfg, cwru_mock)
        assert task.classes == {}


# =====================================================================
# 5. Interactions
# =====================================================================

class TestInteractions:

    def test_class_interactions_resolved(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "class_interactions": {
                "inner ring": {"bearing_position": {1: {"sampling_rate": 1}}},
                "outer ring": {"bearing_position": {1: {"sampling_rate": 1}}},
            },
        }
        task = build_task(cfg, cwru_mock)
        assert task.class_interactions is not None
        assert 1 in task.class_interactions
        assert 2 in task.class_interactions

    def test_interaction_rules_structure(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "class_interactions": {
                "ball": {"bearing_position": {1: {"sampling_rate": 1}}},
            },
        }
        task = build_task(cfg, cwru_mock)
        interactions = task.class_interactions[3]
        assert len(interactions.rules) == 1
        rule = interactions.rules[0]
        assert rule.field == "bearing_position"
        assert rule.trigger_value == 1
        assert rule.constrained_field == "sampling_rate"
        assert 1 in rule.allowed_values

    def test_global_interactions(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "interactions": {
                "bearing_position": {1: {"fault_size": [1, 2]}},
            },
        }
        task = build_task(cfg, cwru_mock)
        assert task.interactions is not None
        rule = task.interactions.rules[0]
        assert rule.allowed_values == frozenset({1, 2})

    def test_no_interactions(self, paderborn_mock):
        from collection.task_builder import build_task

        cfg = {"target": "fault_element"}
        task = build_task(cfg, paderborn_mock)
        assert task.interactions is None
        assert task.class_interactions is None


# =====================================================================
# 6. Full CWRU-style task
# =====================================================================

class TestCWRUFullTask:

    def test_cwru_task_complete(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "domain_factors": ["fault_size", "bearing_position", "condition"],
            "defaults": {
                "fixed": {"fault_size": 0, "bearing_position": 0, "condition": 0},
                "resolve": {
                    "sampling_rate": [1, 2],
                    "fault_position": "centered",
                },
            },
            "classes": {
                "normal": {
                    "fixed": {"fault_size": 0},
                    "resolve": {"fault_position": "normal", "sampling_rate": 48000},
                },
            },
            "class_interactions": {
                "inner ring": {"bearing_position": {1: {"sampling_rate": 1}}},
                "outer ring": {"bearing_position": {1: {"sampling_rate": 1}}},
                "ball": {"bearing_position": {1: {"sampling_rate": 1}}},
            },
        }
        task = build_task(cfg, cwru_mock)

        assert task.target == "fault_element"
        assert task.domain_factors == ("fault_size", "bearing_position", "condition")
        assert task.defaults.fixed == {"fault_size": 0, "bearing_position": 0, "condition": 0}
        assert task.defaults.resolve["fault_position"] == 1  # centered
        assert 0 in task.classes
        assert task.classes[0].resolve["fault_position"] == 0  # normal
        assert set(task.class_interactions.keys()) == {1, 2, 3}


# =====================================================================
# 7. Full Paderborn-style task
# =====================================================================

class TestPaderbornFullTask:

    def test_paderborn_task_complete(self, paderborn_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "domain_factors": [
                "fault_size", "fault_arrangement", "condition",
                "fault_combination", "fault_mode", "fault_characteristic",
            ],
            "defaults": {
                "fixed": {
                    "fault_size": 0, "fault_arrangement": 0, "condition": 0,
                    "fault_combination": 0, "fault_mode": 0, "fault_characteristic": 0,
                },
                "resolve": {"sampling_rate": "all"},
            },
            "classes": {
                "normal": {
                    "fixed": {
                        "fault_size": 0, "fault_arrangement": 0,
                        "fault_combination": 0, "fault_mode": 0, "fault_characteristic": 0,
                    },
                    "resolve": {"sampling_rate": "all"},
                },
            },
        }
        task = build_task(cfg, paderborn_mock)

        assert task.target == "fault_element"
        assert len(task.domain_factors) == 6
        assert task.defaults.resolve["sampling_rate"] == [1]
        assert task.classes[0].resolve["sampling_rate"] == [1]
        assert task.class_interactions is None


# =====================================================================
# 8. build_task_and_filters
# =====================================================================

class TestBuildTaskAndFilters:

    def test_filters_use_domain_factors(self, cwru_mock):
        from collection.task_builder import build_task_and_filters

        cfg = {
            "target": "fault_element",
            "domain_factors": ["fault_size", "bearing_position", "condition"],
            "defaults": {"fixed": {"fault_size": 0}},
            "filters": {"exclude": {"fault_size": [0, 4]}},
        }
        task, filters = build_task_and_filters(cfg, cwru_mock)

        assert task.domain_factors == ("fault_size", "bearing_position", "condition")
        assert isinstance(filters, tuple)
        assert len(filters) > 0

    def test_no_domain_factors_no_filters(self, cwru_mock):
        from collection.task_builder import build_task_and_filters

        cfg = {"target": "fault_element"}
        task, filters = build_task_and_filters(cfg, cwru_mock)
        assert filters == ()

    def test_no_filters_section_still_works(self, cwru_mock):
        from collection.task_builder import build_task_and_filters

        cfg = {
            "target": "fault_element",
            "domain_factors": ["fault_size", "condition"],
        }
        task, filters = build_task_and_filters(cfg, cwru_mock)
        assert isinstance(filters, tuple)


# =====================================================================
# 9. YAML file roundtrip
# =====================================================================

class TestYAMLRoundtrip:

    def test_roundtrip_with_domain_factors(self, cwru_mock):
        from collection.task_builder import build_task_from_yaml

        cfg = {
            "target": "fault_element",
            "domain_factors": ["fault_size", "bearing_position"],
            "defaults": {
                "fixed": {"fault_size": 0},
                "resolve": {"fault_position": "centered"},
            },
        }
        path = _write_yaml(cfg)
        try:
            task = build_task_from_yaml(path, cwru_mock)
            assert task.domain_factors == ("fault_size", "bearing_position")
            assert task.defaults.resolve["fault_position"] == 1
        finally:
            path.unlink(missing_ok=True)

    def test_file_not_found(self, cwru_mock):
        from collection.task_builder import build_task_from_yaml

        with pytest.raises(FileNotFoundError):
            build_task_from_yaml("/nonexistent.yaml", cwru_mock)

    def test_roundtrip_with_filters(self, cwru_mock):
        from collection.task_builder import build_task_and_filters_from_yaml

        cfg = {
            "target": "fault_element",
            "domain_factors": ["fault_size", "condition"],
            "filters": {"exclude": {"fault_size": [0]}},
        }
        path = _write_yaml(cfg)
        try:
            task, filters = build_task_and_filters_from_yaml(path, cwru_mock)
            assert task.domain_factors == ("fault_size", "condition")
            assert isinstance(filters, tuple)
        finally:
            path.unlink(missing_ok=True)


# =====================================================================
# 10. Resolution errors
# =====================================================================

class TestResolutionErrors:

    def test_unknown_class_description(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "classes": {"nonexistent": {"fixed": {"fault_size": 0}}},
        }
        with pytest.raises(ValueError, match="not found"):
            build_task(cfg, cwru_mock)

    def test_unknown_resolve_description(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "defaults": {"resolve": {"fault_position": "nonexistent"}},
        }
        with pytest.raises(ValueError, match="not found"):
            build_task(cfg, cwru_mock)

    def test_resolve_list_with_mixed_types(self, cwru_mock):
        from collection.task_builder import build_task

        cfg = {
            "target": "fault_element",
            "defaults": {"resolve": {"sampling_rate": [1, "48000"]}},
        }
        task = build_task(cfg, cwru_mock)
        assert task.defaults.resolve["sampling_rate"] == [1, 2]