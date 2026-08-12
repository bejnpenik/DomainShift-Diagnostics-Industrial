"""
Transfer study builder — YAML-driven cross-collection transfer study
configuration.

Dispatched from study/builder.py when a study YAML's top-level `type:` is
`transfer`. Parses the `collections:`/`class_aliases`/`sources`/`targets`
sections, builds real DatasetCollection/Task objects per declared
collection, and reuses study/grid.py's existing grid-resolution machinery
for the `grid:` section (via study.builder's parse_study_config/
_make_grid_builder) rather than duplicating it -- the one addition needed
is resolving `processor` as a *fixed* independent value (transfer studies
require a single shared processor across every collection), since the
existing machinery only auto-resolves YAML-path independents when they're
declared as a grid.dependent mapping.

YAML schema: see configs/study/cwru_pu_transfer_study.yaml.

Entry points:
    build_transfer_study_design_from_yaml(path, check_files=True)
        -> (TransferStudyDesign, {name: (DatasetCollection, BaseFileReader)})
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from collection.collection import DatasetCollection
from collection.task_builder import build_task_and_filters_from_yaml
from experiment.config import ExperimentConfig
from experiment.transfer import TransferSpec
from reader.reader import BaseFileReader

from .builder import parse_study_config, _make_grid_builder


@dataclass(frozen=True)
class TransferStudyDesign:
    """Complete specification for a cross-collection transfer study."""
    name: str
    class_aliases: tuple[str, ...]
    target: str
    source_specs: tuple[TransferSpec, ...]
    target_specs: tuple[TransferSpec, ...]
    experiment_configs: tuple[ExperimentConfig, ...]
    seeds: tuple[int, ...]
    description: str = ""

    @property
    def num_configs(self) -> int:
        return len(self.experiment_configs)


def _resolve_explicit_filters(raw_filters: dict, collection: DatasetCollection) -> dict:
    """Resolve a single-domain filter dict's alias strings to codes (mirrors
    task_builder's alias-resolution convention). Ints pass through unchanged."""
    resolved = {}
    for field, value in raw_filters.items():
        resolved[field] = value if isinstance(value, int) else collection.get_filter_value_from_description(field, value)
    return resolved


def _build_experiment_configs(raw_grid: dict, study_name: str, seeds: tuple[int, ...]) -> list[ExperimentConfig]:
    """Reuse study/grid.py's existing resolution machinery for the grid.

    `processor` must be a fixed independent value for a transfer study (the
    single-shared-processor invariant) -- resolve+rename it to
    'processor_config' before delegating, so build_experiment_configs()
    (which reads params['processor_config']) works completely unmodified.
    """
    from representation.builder import build_processor_config_from_yaml

    grid_factors = raw_grid.get("factors", {})
    grid_dependent = raw_grid.get("dependent", {})
    if "processor" in grid_factors:
        raise ValueError(
            "Transfer studies require a single shared processor -- 'processor' cannot be "
            "a varying grid factor. Move it to grid.independent as a fixed path."
        )
    if "processor" in grid_dependent:
        raise ValueError(
            "Transfer studies require a single shared processor -- 'processor' cannot be "
            "a grid.dependent mapping. Move it to grid.independent as a fixed path."
        )

    cfg = parse_study_config({
        "name": study_name,
        "collection": "__transfer__",  # placeholder: grid resolution never touches this
        "task": "__transfer__",        # placeholder: grid resolution never touches this
        "seeds": list(seeds),
        "grid": raw_grid,
    })
    if "processor" in cfg.independent:
        cfg.independent["processor_config"] = build_processor_config_from_yaml(cfg.independent.pop("processor"))

    configs = _make_grid_builder(cfg).build_experiment_configs()

    # Single-shared-processor is a builder-enforced invariant, not just a
    # YAML convention: every resolved grid point must carry the identical
    # processor_config (guaranteed by construction above, asserted here as
    # a hard backstop in case a future edit reintroduces per-point variation).
    if configs:
        first = configs[0].processor_config
        for c in configs[1:]:
            if c.processor_config != first:
                raise ValueError(
                    "All transfer study grid points must share the same processor_config; "
                    f"got at least two different values ({first!r} vs {c.processor_config!r})."
                )
    return configs


def build_transfer_study_design_from_yaml(
    path: str | Path, check_files: bool = True
) -> tuple[TransferStudyDesign, dict[str, tuple[DatasetCollection, BaseFileReader]]]:
    """Load a transfer study YAML and build the full design + collections.

    check_files: forwarded to each DatasetCollection -- default True for
    real runs (main.py); tests pass False so they don't depend on the
    CWRU/Paderborn data actually being on disk.
    """
    path = Path(path)
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    collections: dict[str, tuple[DatasetCollection, BaseFileReader]] = {}
    tasks = {}
    filter_combos = {}
    target: str | None = None

    for name, entry in raw["collections"].items():
        collection = DatasetCollection(entry["collection"], check_files=check_files)
        reader = collection.reader
        if reader is None:
            raise ValueError(
                f"Collection '{name}' has no reader configured. "
                "Add a 'reader:' key pointing to a reader YAML."
            )
        task, filters = build_task_and_filters_from_yaml(entry["task"], collection)
        if target is None:
            target = task.target
        elif task.target != target:
            raise ValueError(
                f"Collection '{name}' task target '{task.target}' does not match "
                f"'{target}' (from an earlier collection) -- every collection in a "
                "transfer study must use tasks with the same target field."
            )
        collections[name] = (collection, reader)
        tasks[name] = task
        filter_combos[name] = filters

    class_aliases = tuple(raw["class_aliases"])

    def _resolve_spec(entry: dict) -> TransferSpec:
        cname = entry["collection"]
        if entry["filters"] == "pooled":
            filters = filter_combos[cname]
        else:
            filters = _resolve_explicit_filters(entry["filters"], collections[cname][0])
        return TransferSpec(cname, tasks[cname], filters)

    source_specs = tuple(_resolve_spec(e) for e in raw["sources"])

    if raw["targets"] == "all":
        target_specs = tuple(
            TransferSpec(name, tasks[name], filter_combos[name]) for name in collections
        )
    else:
        target_specs = tuple(_resolve_spec(e) for e in raw["targets"])

    seeds = tuple(int(s) for s in raw["seeds"])
    experiment_configs = tuple(_build_experiment_configs(raw["grid"], raw["name"], seeds))

    design = TransferStudyDesign(
        name=raw["name"],
        class_aliases=class_aliases,
        target=target,
        source_specs=source_specs,
        target_specs=target_specs,
        experiment_configs=experiment_configs,
        seeds=seeds,
        description=raw.get("description", ""),
    )
    return design, collections
