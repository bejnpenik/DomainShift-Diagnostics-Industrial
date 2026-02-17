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
