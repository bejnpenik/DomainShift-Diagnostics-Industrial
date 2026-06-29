"""
Study builder — YAML-driven study configuration.

Loads a study YAML file and constructs a StudyDesign by resolving:
    - collection + task (via task_builder)
    - grid factors (cartesian product)
    - independent factors (fixed across grid)
    - dependent factors (conditional on grid values)

The study YAML ties together all other builders (model, processor, task)
into a single declarative specification of a full experiment sweep.

YAML schema:

    name: paderborn_study
    collection: paderborn
    task: configs/tasks/paderborn_fault_element.yaml
    seeds: [42, 123, 456]

    grid:
      factors:
        model_type: [1d, 2d]
        model_variant: [1x1, 4x4, multihead]
        sampling_rate: [12000, 48000, 64000]
        normalization: [dataset, sample, pretrained]
        optimizer_name: [adamw, sgd]

      independent:
        file_sampling:
          max_files_per_code: 3
        train_val_split_ratio: 0.33
        max_epochs: 2000
        weight_decay: 0.0001
        momentum: 0.9
        early_stopping: [10, 0.001]
        noise: [0.1, 0.1]
        device: cuda

      dependent:
        model:
          depends_on: [model_type, model_variant]
          mapping:
            1d:
              1x1: configs/models/1d_1x1.yaml
              4x4: configs/models/1d_4x4.yaml
              multihead: configs/models/1d_multihead.yaml
            2d:
              1x1: configs/models/2d_1x1.yaml
              4x4: configs/models/2d_4x4.yaml
              multihead: configs/models/2d_multihead.yaml
        processor:
          depends_on: [model_type, sampling_rate]
          mapping:
            1d:
              12000: configs/processors/raw_12k.yaml
              48000: configs/processors/raw_48k.yaml
              64000: configs/processors/raw_64k.yaml
            2d:
              12000: configs/processors/spec_12k.yaml
              48000: configs/processors/spec_48k.yaml
              64000: configs/processors/spec_64k.yaml
        lr:
          depends_on: optimizer_name
          mapping: {adamw: 0.001, sgd: 0.01}
        normalization_vals:
          depends_on: normalization
          mapping: {pretrained: [0, 1]}
          default: null

Entry points:
    load_study_config(path) -> StudyConfig
    parse_study_config(raw_dict) -> StudyConfig
    build_study_design(cfg, collection) -> (StudyDesign, Task, filters)
    build_study_design_from_yaml(path, collection) -> (StudyDesign, Task, filters)
    resolve_grid_point(grid_point, cfg, base_dir) -> resolved dict
    iter_resolved_grid_points(cfg, base_dir) -> list[resolved dict]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


# =====================================================================
# StudyConfig — parsed study YAML
# =====================================================================

@dataclass
class StudyConfig:
    """Parsed study YAML, ready for resolution against a collection."""

    name: str
    collection: str
    task_path: str
    seeds: tuple[int, ...]

    # Grid factors (cartesian product)
    factors: dict[str, tuple[Any, ...]]

    # Independent factors (fixed values)
    independent: dict[str, Any]

    # Dependent factor specs (raw coerced dicts, keyed by factor name)
    dependent: dict[str, dict]

    @property
    def num_grid_points(self) -> int:
        if not self.factors:
            return 1
        total = 1
        for vals in self.factors.values():
            total *= len(vals)
        return total

    @property
    def total_runs(self) -> int:
        """Grid points × seeds (not counting domains)."""
        return self.num_grid_points * len(self.seeds)


# =====================================================================
# YAML loading
# =====================================================================

def load_study_config(path: str | Path) -> StudyConfig:
    """Load a study YAML file and parse it into a StudyConfig.

    Args:
        path: Path to the study YAML file.

    Returns:
        Parsed StudyConfig (no collection needed at this stage).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required fields are missing.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Study YAML not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return parse_study_config(raw)


def parse_study_config(raw: dict[str, Any]) -> StudyConfig:
    """Parse a raw YAML dict into a StudyConfig.

    Args:
        raw: Parsed YAML dict.

    Returns:
        StudyConfig instance.
    """
    # --- required top-level fields ---
    name = raw.get("name")
    if name is None:
        raise ValueError("Study YAML must have a 'name' field")

    collection = raw.get("collection")
    if collection is None:
        raise ValueError("Study YAML must have a 'collection' field")

    task_path = raw.get("task")
    if task_path is None:
        raise ValueError("Study YAML must have a 'task' field")

    seeds_raw = raw.get("seeds", [42])
    seeds = tuple(int(s) for s in seeds_raw)
    if not seeds:
        raise ValueError("Study YAML must have at least one seed")

    # --- grid section ---
    grid = raw.get("grid", {})

    # factors
    factors_raw = grid.get("factors", {})
    factors = {}
    for factor_name, values in factors_raw.items():
        if not isinstance(values, list):
            raise ValueError(
                f"Grid factor '{factor_name}' must be a list, "
                f"got {type(values).__name__}"
            )
        factors[factor_name] = tuple(values)

    # independent
    independent = dict(grid.get("independent", {}))

    # dependent
    dependent_raw = grid.get("dependent", {})
    dependent = {}
    for dep_name, dep_cfg in dependent_raw.items():
        dependent[dep_name] = _normalise_dependent_spec(dep_name, dep_cfg)

    return StudyConfig(
        name=name,
        collection=collection,
        task_path=task_path,
        seeds=seeds,
        factors=factors,
        independent=independent,
        dependent=dependent,
    )


def _normalise_dependent_spec(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise a single dependent factor spec from YAML.

    Coerces depends_on to a str or list and runs key coercion on the
    mapping.  Returns a plain dict consumed by build_grid_from_dicts.
    """
    depends_on_raw = cfg.get("depends_on")
    if depends_on_raw is None:
        raise ValueError(f"Dependent factor '{name}' must have 'depends_on'")

    if isinstance(depends_on_raw, str):
        depends_on = depends_on_raw
    elif isinstance(depends_on_raw, list):
        depends_on = depends_on_raw[0] if len(depends_on_raw) == 1 else depends_on_raw
    else:
        raise ValueError(
            f"Dependent factor '{name}': depends_on must be string or list"
        )

    mapping_raw = cfg.get("mapping")
    if mapping_raw is None:
        raise ValueError(f"Dependent factor '{name}' must have 'mapping'")

    return {
        "depends_on": depends_on,
        "mapping": _coerce_mapping_keys(mapping_raw),
        "default": cfg.get("default"),
    }


def _coerce_mapping_keys(mapping: dict) -> dict:
    """Recursively coerce mapping keys to match grid factor value types.

    YAML parses '1d' as string, 12000 as int, etc. We coerce all
    string representations of ints to actual ints, and keep strings
    as strings. Nested dicts (for multi-factor depends_on) are
    recursed into.
    """
    result = {}
    for key, value in mapping.items():
        coerced_key = _coerce_key(key)
        if isinstance(value, dict):
            result[coerced_key] = _coerce_mapping_keys(value)
        else:
            result[coerced_key] = value
    return result


def _coerce_key(key: Any) -> Any:
    """Coerce a mapping key: try int first, otherwise keep as-is."""
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        try:
            return int(key)
        except ValueError:
            return key
    return key


def _resolve_yaml_mapping(mapping: dict, loader) -> dict:
    """Recursively resolve .yaml/.yml string leaves in a mapping using loader."""
    result = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            result[key] = _resolve_yaml_mapping(value, loader)
        elif isinstance(value, str) and (value.endswith('.yaml') or value.endswith('.yml')):
            result[key] = loader(value)
        else:
            result[key] = value
    return result


def _make_grid_builder(cfg: StudyConfig):
    """Build a fully-resolved StudyGridBuilder from a StudyConfig.

    Resolves all YAML-specific concerns so grid points contain typed objects
    that build_experiment_configs can consume directly:
      - model/processor: YAML paths → typed ModelConfig/ProcessorConfig (renamed)
      - trainer_config: synthesized from static independent fields + one varying field
      - file_sampling: dict → FileSamplingProtocol
    """
    from .grid import build_grid_from_dicts
    from model.config import ModelConfig
    from representation.builder import build_processor_config_from_yaml
    from training.config import TrainerConfig
    from experiment.sampling import FileSamplingProtocol

    # --- resolve model/processor YAML paths, rename to ExperimentConfig field names ---
    _YAML_LOADERS = {
        'model': (ModelConfig.from_yaml, 'model_config'),
        'processor': (build_processor_config_from_yaml, 'processor_config'),
    }

    resolved_dependent = {}
    for factor_name, dep_spec in cfg.dependent.items():
        entry = _YAML_LOADERS.get(factor_name)
        if entry is not None:
            loader, output_name = entry
            resolved_spec = dict(dep_spec)
            resolved_spec['mapping'] = _resolve_yaml_mapping(dep_spec['mapping'], loader)
            resolved_dependent[output_name] = resolved_spec
        elif factor_name in TrainerConfig.model_fields:
            pass  # collected below as varying trainer field
        else:
            resolved_dependent[factor_name] = dep_spec

    # --- synthesize trainer_config ---
    static_trainer = {k: v for k, v in cfg.independent.items() if k in TrainerConfig.model_fields}
    varying_trainer = {k: cfg.dependent[k] for k in cfg.dependent if k in TrainerConfig.model_fields}

    if not varying_trainer:
        resolved_independent = {**cfg.independent, 'trainer_config': TrainerConfig(**static_trainer)}
    elif len(varying_trainer) == 1:
        varying_field, dep_spec = next(iter(varying_trainer.items()))
        depends_on = dep_spec['depends_on']
        trainer_mapping = {}
        for key, field_val in dep_spec['mapping'].items():
            tc_kwargs = {**static_trainer, varying_field: field_val}
            if isinstance(depends_on, str) and depends_on in TrainerConfig.model_fields:
                tc_kwargs[depends_on] = key
            trainer_mapping[key] = TrainerConfig(**tc_kwargs)
        resolved_dependent['trainer_config'] = {
            'depends_on': depends_on,
            'mapping': trainer_mapping,
            'default': None,
        }
        resolved_independent = cfg.independent
    else:
        raise ValueError(
            f"Multiple varying trainer fields not supported: {list(varying_trainer)}. "
            "Use a single dependent field (e.g. lr)."
        )

    # --- strip flat trainer fields; convert file_sampling + adaptation_config dicts ---
    clean_independent = {k: v for k, v in resolved_independent.items()
                         if k not in TrainerConfig.model_fields}
    if isinstance(clean_independent.get('file_sampling'), dict):
        clean_independent = {**clean_independent,
                             'file_sampling': FileSamplingProtocol(**clean_independent['file_sampling'])}
    # adaptation_config is kept as a raw dict here; grid.py converts it to
    # AdaptationConfig when building ExperimentConfig objects.
    # No conversion needed in the builder — just pass it through.

    return build_grid_from_dicts(
        study_factors=cfg.factors,
        study_independent_factors=clean_independent,
        study_dependent_factors=resolved_dependent,
    )


# =====================================================================
# Study design construction
# =====================================================================

def build_study_design(
    cfg: StudyConfig,
    collection,
) -> tuple:
    """Build a StudyDesign from a parsed StudyConfig and collection.

    Builds the Task + filters from the task YAML, then delegates fully to
    StudyGridBuilder.build_study_design(). All YAML resolution (model, processor,
    trainer_config, file_sampling) is handled by _make_grid_builder.

    Args:
        cfg: Parsed StudyConfig.
        collection: DatasetCollection instance.

    Returns:
        (StudyDesign, Task, filter_combinations) tuple.
    """
    from collection.task_builder import build_task_and_filters_from_yaml

    task, filters = build_task_and_filters_from_yaml(cfg.task_path, collection)

    filters = collection.validate_filters(task, filters)
    if filters is None:
        raise ValueError(
            f"No valid filter combinations produced a complete dataset plan for "
            f"study '{cfg.name}'. Check task path '{cfg.task_path}' and your "
            "collection metadata."
        )

    # TODO: validate pipeline.primary and conditioning channels against
    # collection.channels here, before building the grid, to catch typos
    # at study-build time rather than deep inside the experiment loop.
    # The pipeline is currently a raw dict inside cfg.independent (or
    # cfg.factors for varying pipelines) and only converted to a
    # PipelineConfig in grid.py's build_experiment_configs. Either parse
    # it here or add a _validate_pipeline(cfg, collection) helper.

    design = _make_grid_builder(cfg).build_study_design(
        study_name=cfg.name,
        task=task,
        filter_combinations=filters,
        seeds=cfg.seeds,
    )
    return design, task, filters


def build_study_design_from_yaml(
    path: str | Path,
    collection,
) -> tuple:
    """Load a study YAML and build the full StudyDesign.

    Convenience: combines load_study_config + build_study_design.

    Args:
        path: Path to study YAML file.
        collection: DatasetCollection instance.

    Returns:
        (StudyDesign, Task, filter_combinations) tuple.
    """
    cfg = load_study_config(path)
    return build_study_design(cfg, collection)


# =====================================================================
# Grid point resolution (for the experiment runner)
# =====================================================================

def resolve_grid_point(
    grid_point: dict[str, Any],
    cfg: StudyConfig,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve a single grid point's dependent factors into usable objects.

    Takes the raw grid point dict (from StudyGridBuilder.iter_combinations)
    and resolves YAML paths in dependent factors to actual config dicts.

    This is called at experiment-run time, not at design time, because
    building models/processors requires knowing num_classes etc.

    Args:
        grid_point: Dict with all factor values.
        cfg: StudyConfig for context (which factors are dependent).
        base_dir: Base directory for resolving relative YAML paths.

    Returns:
        New dict with resolved values. Keys that pointed to YAML paths
        are replaced with the loaded config dicts (not yet built into
        objects — that happens in the experiment runner).
    """
    if base_dir is None:
        base_dir = Path(".")

    resolved = dict(grid_point)

    # Resolve YAML path references in dependent factors
    for dep_name in cfg.dependent:
        value = resolved.get(dep_name)
        if value is None:
            continue

        if isinstance(value, str) and (
            value.endswith(".yaml") or value.endswith(".yml")
        ):
            resolved[dep_name] = _load_yaml_ref(base_dir / value)

    return resolved


def _load_yaml_ref(path: Path) -> dict:
    """Load a YAML file reference, returning the parsed dict."""
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"Referenced YAML not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f)


# =====================================================================
# Utility: iterate resolved grid points
# =====================================================================

def iter_resolved_grid_points(
    cfg: StudyConfig,
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Build the full grid and resolve all dependent factors.

    Returns a list of fully resolved grid point dicts, with YAML
    paths loaded into config dicts.

    Args:
        cfg: StudyConfig.
        base_dir: Base directory for resolving relative YAML paths.

    Returns:
        List of resolved grid point dicts.
    """
    return [
        resolve_grid_point(gp, cfg, base_dir)
        for gp in _make_grid_builder(cfg).iter_combinations()
    ]