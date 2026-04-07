"""
Task builder — YAML-driven task configuration.

Converts a YAML dict or file into a Task object by resolving aliases from
the collection header. All filter values, class keys, and interaction trigger
values use aliases (short strings defined in the collection's header section).

The task YAML is separate from the collection YAML because one collection
can have multiple tasks. The task references collection filter names and
aliases, so the collection is needed at build time for resolution.

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
        fault_size: S0            # alias — overridden per class below
        bearing_position: FE      # alias — overridden by domain filters at runtime
        condition: C1             # alias — overridden by domain filters at runtime
      resolve:
        sampling_rate: [12k, 48k] # list of aliases, expands to both codes
        fault_position: CE        # alias for "centered"

    classes:
      NR:                          # alias for the "normal" target class
        fixed:
          fault_size: S0
        resolve:
          fault_position: NR
          sampling_rate: 48k

    class_interactions:
      IR:                          # alias for "inner ring" class
        bearing_position:
          FE:                      # alias trigger — when bearing_position == FE
            sampling_rate: 12k    # alias constraint — sampling_rate must be 12k

    filters:                       # practical constraints on which combos to run
      exclude:
        fault_size: [S0, S28]     # aliases resolved before generating combinations

Resolution rules:
    - "all" expands to all available codes via collection.get_all_filter_values()
    - Alias strings (e.g. "IR", "48k", "FE") resolve via collection.get_filter_value_from_description()
    - Integer codes pass through unchanged (both formats accepted)
    - Lists of aliases/ints are resolved element-wise
    - Class keys in classes: and class_interactions: accept aliases or integer codes
    - Interaction trigger keys accept aliases or integer codes

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
        class_code = _resolve_class_key(class_desc, target, collection)
        classes[class_code] = _build_rule(class_rule_cfg, collection)

    # --- interactions (global) ---
    interactions_cfg = cfg.get("interactions")
    interactions = None
    if interactions_cfg is not None:
        interactions = _build_interactions(interactions_cfg, collection)

    # --- class_interactions ---
    class_interactions_cfg = cfg.get("class_interactions", {})
    class_interactions = None
    if class_interactions_cfg:
        class_interactions = {}
        for class_desc, inter_cfg in class_interactions_cfg.items():
            class_code = _resolve_class_key(class_desc, target, collection)
            class_interactions[class_code] = _build_interactions(inter_cfg, collection)

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
    exclude_raw = filters_cfg.get("exclude", {})
    exclude = {
        field: _resolve_value(field, value, collection)
        for field, value in exclude_raw.items()
    }

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

    fixed = {
        field: _resolve_value(field, value, collection)
        for field, value in rule_cfg.get("fixed", {}).items()
    }
    resolve_cfg = rule_cfg.get("resolve", {})

    resolve = {}
    for field, value in resolve_cfg.items():
        resolve[field] = _resolve_value(field, value, collection)

    return Rule(fixed=fixed, resolve=resolve)


def _resolve_value(field: str, value: Any, collection) -> Any:
    """Resolve a single value from YAML.

    Rules:
        - "all"  -> collection.get_all_filter_values(field)
        - alias string -> collection.get_filter_value_from_description(field, value)
        - int    -> pass through
        - list   -> each item resolved recursively
    """
    if isinstance(value, str):
        if value == "all":
            return collection.get_all_filter_values(field)
        return collection.get_filter_value_from_description(field, value)

    if isinstance(value, int):
        return value

    if isinstance(value, list):
        resolved = []
        for item in value:
            if isinstance(item, str):
                if item == "all":
                    resolved.extend(collection.get_all_filter_values(field))
                else:
                    resolved.append(collection.get_filter_value_from_description(field, item))
            else:
                resolved.append(item)
        return resolved

    return value


def _resolve_class_key(key: Any, target: str, collection) -> int:
    """Resolve a class key from YAML — accepts integer codes or alias strings."""
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        return collection.get_filter_value_from_description(target, key)
    raise ValueError(
        f"Class key must be an integer code or alias string, got {type(key).__name__}: {key!r}."
    )


def _build_interactions(inter_cfg: dict, collection) -> 'Interactions':
    """Build an Interactions object from a YAML dict.

    The YAML format matches Interactions.from_dict() directly:
        field:
          trigger_value:
            constrained_field: allowed_value(s)

    Trigger keys must be integer codes. Constraint values are resolved
    via _resolve_value (aliases accepted).
    """

    if inter_cfg is None:
        return Interactions(())

    processed = {}
    for field, conditions in inter_cfg.items():
        processed[field] = {}
        for trigger, constraints in conditions.items():
            trigger_int = _resolve_value(field, trigger, collection)
            processed[field][trigger_int] = {}
            for constrained_field, allowed in constraints.items():
                processed[field][trigger_int][constrained_field] = (
                    _resolve_value(constrained_field, allowed, collection)
                )

    return Interactions.from_dict(processed)