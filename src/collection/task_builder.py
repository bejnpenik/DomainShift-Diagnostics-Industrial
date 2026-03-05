"""
Task builder — YAML-driven task configuration.

Converts a YAML dict or file into a Task object by resolving human-readable
descriptions (like "normal", "inner ring") against a DatasetCollection.

The task YAML is separate from the collection YAML because one collection
can have multiple tasks. But the task references collection filter names
and descriptions, so the collection is needed at build time for resolution.

YAML schema:

    # configs/tasks/cwru_fault_element.yaml
    collection: cwru              # which collection this task is for
    target: fault_element
    domain_factors:               # what varies across domains (= Task.domain_factors)
      - fault_size
      - bearing_position
      - condition

    defaults:
      fixed:
        fault_size: 0
        bearing_position: 0
        condition: 0
      resolve:
        sampling_rate: [1, 2]     # integer codes from collection header
        fault_position: 1         # code for "centered"

    classes:
      0:                           # integer code for "normal"
        fixed:
          fault_size: 0
        resolve:
          fault_position: 0        # code for "normal position"
          sampling_rate: 2         # code for 48000 Hz

    class_interactions:
      1:                           # integer code for "inner ring"
        bearing_position:
          1:
            sampling_rate: 1

    filters:                       # practical constraints on which combos to run
      exclude:
        fault_size: [0, 4]

Resolution rules:
    - All values must be integer codes (from the collection's header section)
    - The special value "all" expands to all available codes for that field
      via collection.get_all_filter_values()
    - Integer values pass through unchanged
    - Lists of integers pass through unchanged
    - Any other string raises ValueError

Entry points:
    build_task(cfg: dict, collection) -> Task
    build_task_from_yaml(path, collection) -> Task
    build_filters(cfg, task, collection) -> tuple[dict, ...]
    build_task_and_filters(cfg, collection) -> (Task, tuple[dict, ...])
    build_task_and_filters_from_yaml(path, collection) -> (Task, tuple[dict, ...])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .task import Task, Rule, Interactions


def build_task(cfg: dict[str, Any], collection) -> Task:
    """Build a Task from a parsed YAML dict.

    Args:
        cfg: Parsed YAML dict with task configuration.
        collection: DatasetCollection instance for resolving descriptions.

    Returns:
        A Task object ready for use with the collection.
    """

    target = cfg.get("target")
    if target is None:
        raise ValueError("Task YAML must have a 'target' field")

    # --- domain_factors ---
    domain_factors = cfg.get("domain_factors", ())
    if isinstance(domain_factors, str):
        domain_factors = (domain_factors,)
    else:
        domain_factors = tuple(domain_factors)

    # --- defaults ---
    defaults_cfg = cfg.get("defaults", {})
    defaults = _build_rule(defaults_cfg, collection)

    # --- classes ---
    classes_cfg = cfg.get("classes", {})
    classes = {}
    for class_desc, class_rule_cfg in classes_cfg.items():
        class_code = _resolve_class_key(class_desc)
        classes[class_code] = _build_rule(class_rule_cfg, collection)

    # --- interactions (global) ---
    interactions_cfg = cfg.get("interactions")
    interactions = None
    if interactions_cfg is not None:
        interactions = _build_interactions(interactions_cfg)

    # --- class_interactions ---
    class_interactions_cfg = cfg.get("class_interactions", {})
    class_interactions = None
    if class_interactions_cfg:
        class_interactions = {}
        for class_desc, inter_cfg in class_interactions_cfg.items():
            class_code = _resolve_class_key(class_desc)
            class_interactions[class_code] = _build_interactions(inter_cfg)

    return Task(
        target=target,
        domain_factors=domain_factors,
        defaults=defaults,
        classes=classes,
        interactions=interactions,
        class_interactions=class_interactions if class_interactions else None,
    )


def build_task_from_yaml(path: str | Path, collection) -> Task:
    """Load a task YAML file and build a Task."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task YAML not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    return build_task(cfg, collection)


def build_filters(cfg: dict[str, Any], task, collection) -> tuple[dict, ...]:
    """Build filter combinations from a task config.

    Uses task.domain_factors as the 'depends' argument and
    cfg["filters"]["exclude"] for exclusions.

    Args:
        cfg: Parsed YAML dict (full task config).
        task: The Task object (already built).
        collection: DatasetCollection for generating combinations.

    Returns:
        Tuple of valid filter dicts.
    """
    filters_cfg = cfg.get("filters", {})
    exclude = filters_cfg.get("exclude", {})

    if not task.domain_factors:
        return ()

    return collection.create_valid_filter_combinations(
        task=task,
        depends=task.domain_factors,
        **exclude,
    )


def build_task_and_filters(
    cfg: dict[str, Any], collection
) -> tuple[Task, tuple[dict, ...]]:
    """Build both a Task and its filter combinations from a YAML dict.

    Returns:
        (task, filters) tuple.
    """
    task = build_task(cfg, collection)
    filters = build_filters(cfg, task, collection)
    return task, filters


def build_task_and_filters_from_yaml(
    path: str | Path, collection
) -> tuple[Task, tuple[dict, ...]]:
    """Load a task YAML and build both the Task and filter combinations."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task YAML not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    return build_task_and_filters(cfg, collection)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_rule(rule_cfg: dict[str, Any], collection) -> Rule:
    """Build a Rule from a YAML dict, resolving descriptions."""

    if rule_cfg is None:
        return Rule()

    fixed = dict(rule_cfg.get("fixed", {}))
    resolve_cfg = rule_cfg.get("resolve", {})

    resolve = {}
    for field, value in resolve_cfg.items():
        resolve[field] = _resolve_value(field, value, collection)

    return Rule(fixed=fixed, resolve=resolve)


def _resolve_value(field: str, value: Any, collection) -> Any:
    """Resolve a single value from YAML.

    Rules:
        - "all" -> collection.get_all_filter_values(field)
        - int -> pass through
        - list of ints (or "all" items) -> pass through / expand
        - any other string -> ValueError (use integer codes)
    """
    if isinstance(value, str):
        if value == "all":
            return collection.get_all_filter_values(field)
        raise ValueError(
            f"String values in task YAML must be 'all', got {value!r} for field '{field}'. "
            "Use integer codes instead."
        )

    if isinstance(value, int):
        return value

    if isinstance(value, list):
        resolved = []
        for item in value:
            if isinstance(item, str):
                if item == "all":
                    resolved.extend(collection.get_all_filter_values(field))
                else:
                    raise ValueError(
                        f"String values in task YAML must be 'all', got {item!r} for field '{field}'. "
                        "Use integer codes instead."
                    )
            else:
                resolved.append(item)
        return resolved

    return value


def _resolve_class_key(key: Any) -> int:
    """Validate and return a class key from YAML — must be an integer code."""
    if isinstance(key, int):
        return key
    raise ValueError(
        f"Class key must be an integer code, got {type(key).__name__}: {key!r}. "
        "Use the integer code from the collection header."
    )


def _build_interactions(inter_cfg: dict) -> 'Interactions':
    """Build an Interactions object from a YAML dict.

    The YAML format matches Interactions.from_dict() directly:
        field:
          trigger_value:
            constrained_field: allowed_value(s)

    All values are integers (codes), no resolution needed here.
    """

    if inter_cfg is None:
        return Interactions(())

    # Convert string keys to ints if needed (YAML may parse "1" as int already)
    processed = {}
    for field, conditions in inter_cfg.items():
        processed[field] = {}
        for trigger, constraints in conditions.items():
            trigger_int = int(trigger)
            processed[field][trigger_int] = {}
            for constrained_field, allowed in constraints.items():
                processed[field][trigger_int][constrained_field] = allowed

    return Interactions.from_dict(processed)