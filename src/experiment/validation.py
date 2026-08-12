"""
Fail-fast validation for cross-collection transfer setups.

Runs entirely before any training. Checks are ordered cheapest-first and
non-I/O checks are aggregated into a single raised message so a user sees
every config problem in one pass; the one check that reads real files (the
processor probe) only runs once every non-I/O check has already passed.

Companion runtime check (not implemented here): once a plan has gone through
pooling/restriction (Phase 2/3's chokepoint in transfer.py), the resulting
DatasetPlan must satisfy `plan.is_complete`, raising with `plan.empty_classes`
named, before it's used for training/eval. That's a structural property of
the *finished* plan and belongs at the point plans are built, not here --
this module only validates the transfer *setup* (aliases, channels,
processor) ahead of any plan construction.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from collection import DatasetCollection, DatasetPlan
from reader import BaseFileReader
from .config import ExperimentConfig
from .experiment import Experiment
from .sampling import FileSamplingProtocol


def resolve_class_aliases_to_names(
    collection: DatasetCollection, target: str, aliases: tuple[str, ...]
) -> dict[str, str]:
    """Resolve alias strings (e.g. 'IR') to the class label names used as
    DatasetPlan.sample_groups keys (e.g. 'inner ring').

    Only resolves the given aliases -- never the collection's full target
    header. Collections legitimately have different, larger header/alias
    sets (e.g. CWRU has a 'ball' class Paderborn doesn't); that's expected
    and out of scope for transfer validation, which only cares about the
    aliases the study actually declared via class_aliases.
    """
    names = {}
    for alias in aliases:
        code = collection.get_filter_value_from_description(target, alias)
        desc = collection.header[target][code]
        names[alias] = desc["name"] if isinstance(desc, dict) else desc
    return names


def _native_rates(ch_cfg, collection: DatasetCollection) -> tuple[int, ...]:
    """Native sampling rate(s) for a channel, read from config/header only
    (no signal I/O). Static int, or every value in the collection header if
    'dynamic' (e.g. CWRU vibration: 12000/48000 depending on file)."""
    if isinstance(ch_cfg.sampling_rate, int):
        return (ch_cfg.sampling_rate,)
    header = collection.header[ch_cfg.sampling_rate_key]
    rates = {int(d["value"] if isinstance(d, dict) else d) for d in header.values()}
    return tuple(sorted(rates))


@dataclass(frozen=True)
class TransferValidationReport:
    probe_shapes: dict[str, tuple[int, ...]]
    warnings: tuple[str, ...]


def validate_transfer_setup(
    collections: dict[str, DatasetCollection],
    class_aliases: tuple[str, ...],
    target: str,
    config: ExperimentConfig,
    readers: dict[str, BaseFileReader],
    probe_plans: dict[str, DatasetPlan],
    probe_files_per_class: int = 1,
) -> TransferValidationReport:
    """Checks that must pass before any training. Raises on hard failure.

    probe_plans: one representative plan per collection (e.g. the source
    plan already being built by the caller) -- used only by the final,
    I/O-bound check to read a handful of real files and compare output
    tensor shapes. The earlier presence check (b) also reads these plans'
    `sample_groups` keys, but never opens a file.
    """
    names = sorted(collections)
    if len(names) < 2:
        raise ValueError("Transfer requires at least 2 collections")

    errors: list[str] = []

    # (a) class_aliases resolve to the SAME class name in every collection.
    # Scoped strictly to class_aliases -- collections' full target headers
    # are allowed (expected) to differ.
    per_collection_names: dict[str, dict[str, str]] = {}
    for c in names:
        try:
            per_collection_names[c] = resolve_class_aliases_to_names(collections[c], target, class_aliases)
        except ValueError as e:
            errors.append(f"Collection '{c}': {e}")
            per_collection_names[c] = {}

    for alias in class_aliases:
        resolved = {c: per_collection_names[c][alias] for c in names if alias in per_collection_names[c]}
        if len(set(resolved.values())) > 1:
            errors.append(
                f"class_alias '{alias}' resolves to different class names across collections: {resolved}"
            )

    # (b) each class_aliases entry is present as a sample_groups key in the
    # collection's probe plan. Presence only, NOT non-emptiness -- a class
    # being empty in one particular domain cell is legitimate (e.g. CWRU
    # inner-ring has no fault_size=S0 data); that's not a setup error.
    for c in names:
        alias_names = set(per_collection_names[c].values())
        plan_classes = set(probe_plans[c].sample_groups)
        missing_names = alias_names - plan_classes
        if missing_names:
            missing_aliases = [a for a, n in per_collection_names[c].items() if n in missing_names]
            errors.append(
                f"Collection '{c}': class_aliases missing from probe plan: {missing_aliases}"
            )

    # (c) primary channel exists in every collection.
    if config.pipeline is None:
        errors.append("ExperimentConfig.pipeline is required for transfer validation")
    else:
        for c in names:
            if config.pipeline.primary not in collections[c].channels:
                errors.append(
                    f"Pipeline primary '{config.pipeline.primary}' not in collection "
                    f"'{c}' channels: {sorted(collections[c].channels)}"
                )

    if errors:
        detail = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"Transfer setup validation failed:\n{detail}")

    # (d) sampling-rate sanity (warn only, non-I/O): fires when a
    # collection's native rate is BELOW the shared processor's target rate
    # (upsampling), never above.
    warnings: list[str] = []
    target_rate = getattr(config.processor_config, "target_sampling_rate", None)
    if target_rate is not None:
        for c in names:
            ch_cfg = collections[c].channels[config.pipeline.primary]
            native = _native_rates(ch_cfg, collections[c])
            if native and min(native) < target_rate:
                warnings.append(
                    f"Collection '{c}' native rate(s) {native} include values below "
                    f"processor target {target_rate} Hz (upsampling)."
                )

    # (e) shared processor -> identical probe tensor shapes. The only
    # I/O-bound check, gated on every check above passing. Uses its own
    # minimal probe sampling (~1 file per class per collection), independent
    # of the study's configured file_sampling policy -- this must run in
    # seconds; exhaustive sampled counts are the dry-run's job, not this
    # validator's.
    probe_shapes: dict[str, tuple[int, ...]] = {}
    probe_sampling = FileSamplingProtocol(max_files_per_code=probe_files_per_class)
    probe_config = dataclasses.replace(config, file_sampling=probe_sampling)
    for c in names:
        experiment = Experiment(collections[c], readers[c], probe_config)
        X, _, _, _ = experiment.load_plan_arrays(probe_plans[c])
        probe_shapes[c] = tuple(X.shape[1:])
    ref_shape = probe_shapes[names[0]]
    bad = {c: s for c, s in probe_shapes.items() if s != ref_shape}
    if bad:
        raise ValueError(f"Processor produces inconsistent shapes across collections: {probe_shapes}")

    return TransferValidationReport(probe_shapes=probe_shapes, warnings=tuple(warnings))
