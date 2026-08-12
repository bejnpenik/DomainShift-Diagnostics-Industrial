"""
Cross-collection transfer: train a model on one collection, evaluate it on
another (including itself, as an in-collection reference point).

TransferExperiment wraps one ordinary Experiment per collection, all sharing
the same ExperimentConfig (same processor, model config, trainer) -- that
shared-processor invariant is what experiment.validation.validate_transfer_setup
checks before any of this runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch

from collection import Task, DatasetCollection, DatasetPlan
from collection.dataset_plan import SampleGroup
from reader import BaseFileReader
from .config import ExperimentConfig
from .experiment import Experiment
from .validation import resolve_class_aliases_to_names
from results import DomainSolution, MultiDomainSolution


@dataclass(frozen=True)
class TransferSpec:
    """One source or target entry for TransferExperiment.run_transfer.

    filters: a single filter dict (one domain) or a tuple of filter dicts
    (pooled across those domains) -- both source and target specs accept
    either form, resolved through TransferExperiment._get_plan.
    """
    collection: str
    task: Task
    filters: dict | tuple[dict, ...]


def sanitize_label_for_filename(label: str) -> str:
    """Filesystem-safe form of a collection-qualified label (e.g.
    "cwru:fault_element-pooled"). ":" -> "__"; anything outside
    [A-Za-z0-9._=-] -> "-". Only for paths written to disk (model weights,
    artifacts) -- result containers/CSV always keep the colon form.
    """
    sanitized = label.replace(":", "__")
    return re.sub(r"[^A-Za-z0-9._=-]", "-", sanitized)


def _pooled_label(task: Task, filter_combinations: tuple[dict, ...]) -> str:
    """Deterministic label for a pooled plan.

    Built from the union of unique values per domain factor across every
    pooled combination, sorted keys and sorted value-lists, reusing
    Task.label()'s own formatting -- so two _get_plan calls pooling the same
    set of combinations (any input order) always produce the same label,
    and the resulting list-valued Task.label() output can never collide
    with any single-domain (scalar-valued) label. The '-pooled' suffix
    makes that unmistakable to a human reader too.
    """
    factor_values: dict[str, set] = {}
    for filters in filter_combinations:
        for k, v in filters.items():
            factor_values.setdefault(k, set()).add(v)
    merged = {k: sorted(vs) for k, vs in sorted(factor_values.items())}
    return f"{task.label(**merged)}-pooled"


def restrict_to_classes(plan: DatasetPlan, class_names: frozenset[str]) -> DatasetPlan:
    """Drop classes not in class_names, before any file is loaded.

    Operates on the DatasetPlan itself -- this must happen here, at the plan
    level, and never after DomainDataset has already loaded files: if
    exclusion happened post-load, excluded-class signals would still flow
    into the dataset-mode normalizer's .fit() call and contaminate the
    source collection's normalization statistics with data that was never
    supposed to be part of the shared label space (e.g. CWRU's 'ball'
    class, absent from Paderborn).

    Deliberately KEEPS the plan's original label -- self-evaluation label
    matching between a source and its own target depends on the label being
    unaffected by which classes survived restriction.

    Raises if a declared class_names entry is missing from the plan's
    sample_groups -- validate_transfer_setup's presence check is meant to
    catch this before training even starts; this is the runtime backstop.
    """
    missing = class_names - set(plan.sample_groups)
    if missing:
        raise ValueError(f"Plan '{plan.label}' is missing declared classes: {sorted(missing)}")
    return DatasetPlan(
        dataset_name=plan.dataset_name,
        label=plan.label,
        sample_groups={k: v for k, v in plan.sample_groups.items() if k in class_names},
    )


class TransferExperiment:
    """Orchestrates train-on-one-collection, evaluate-on-another.

    Breadcrumb for the Phase 5 study runner (TransferStudy): mirror
    Study._run_spec_multi_seed's `dataclasses.replace(config,
    random_seed=seed)` and rebuild a *fresh* TransferExperiment per seed, so
    every per-collection sub-experiment shares the same seeded config --
    don't reuse one TransferExperiment instance across seeds.
    """

    def __init__(
        self,
        collections: dict[str, tuple[DatasetCollection, BaseFileReader]],
        config: ExperimentConfig,
        class_aliases: tuple[str, ...],
        target: str,
    ):
        if not class_aliases:
            raise ValueError("class_aliases must be non-empty")
        class_aliases = tuple(class_aliases)

        self._collections = {name: c for name, (c, _) in collections.items()}
        self._experiments = {
            name: Experiment(c, r, config) for name, (c, r) in collections.items()
        }
        self._config = config
        self._target = target

        # Resolved once, per collection: the set of class NAMES (not
        # aliases) every plan from that collection gets restricted to.
        # Two aliases resolving to the same name would silently collapse
        # num_classes, so that's a hard construction-time failure too.
        self._class_names_by_collection: dict[str, frozenset[str]] = {}
        for name, collection in self._collections.items():
            resolved_names = resolve_class_aliases_to_names(collection, target, class_aliases)
            names = frozenset(resolved_names.values())
            if len(names) != len(class_aliases):
                raise ValueError(
                    f"Collection '{name}': class_aliases {class_aliases} resolve to only "
                    f"{len(names)} distinct class name(s) ({sorted(names)}) -- two or more "
                    f"aliases collapse to the same class, which would silently change "
                    f"num_classes."
                )
            self._class_names_by_collection[name] = names

    @property
    def processor_name(self) -> str:
        return next(iter(self._experiments.values())).processor_name

    def _get_plan(self, collection_name: str, task: Task, filters: dict | tuple[dict, ...]) -> DatasetPlan:
        """Single chokepoint for ALL plan acquisition in the transfer layer.

        Construct (dict filters) or pool (tuple/list of filter dicts), then
        restrict to this TransferExperiment's class_aliases before
        returning -- every call site already routes through here, so
        restriction can't be forgotten by any caller.
        """
        if task.target != self._target:
            raise ValueError(
                f"Task target '{task.target}' does not match this TransferExperiment's "
                f"configured target '{self._target}' (collection '{collection_name}'). "
                f"class_aliases were resolved against '{self._target}'; a task with a "
                f"different target would silently use the wrong class-name set."
            )

        collection = self._collections[collection_name]
        if isinstance(filters, dict):
            plan = collection.construct_dataset_plan(task, **filters)
        else:
            if not filters:
                raise ValueError(
                    f"Pooling requires at least one filter combination (collection '{collection_name}')"
                )
            plan = self._build_pooled_plan(collection, task, tuple(filters))

        plan = restrict_to_classes(plan, self._class_names_by_collection[collection_name])
        if not plan.is_complete:
            raise ValueError(
                f"Plan '{plan.label}' for collection '{collection_name}' has empty classes "
                f"after restriction: {plan.empty_classes}"
            )
        return plan

    def _build_pooled_plan(
        self, collection: DatasetCollection, task: Task, filter_combinations: tuple[dict, ...]
    ) -> DatasetPlan:
        """Union every valid per-domain plan's sample_groups into one plan.

        Does NOT pass list-valued filters into construct_dataset_plan
        directly -- confirmed that raises (e.g. CWRU inner-ring has no
        fault_size=S0 data). Builds each already-valid per-domain plan and
        unions them: same code mapping to the same files across domains is
        a benign, expected duplicate (e.g. CWRU's 'normal' class reuses the
        same baseline recording across fault-size variants); same code
        mapping to different files is a real conflict and raises.
        """
        plans = tuple(collection.construct_dataset_plan(task, **f) for f in filter_combinations)

        merged_codes: dict[str, dict] = {}
        merged_meta: dict[str, dict] = {}
        for plan in plans:
            for cls, sg in plan.sample_groups.items():
                mc = merged_codes.setdefault(cls, {})
                mm = merged_meta.setdefault(cls, {})
                for code, fnames in sg.codes.items():
                    if code in mc and mc[code] != fnames:
                        raise ValueError(
                            f"Pooling conflict: class '{cls}' code {code} maps to different "
                            f"files across domain plans ({mc[code]} vs {fnames})"
                        )
                    mc[code] = fnames
                    mm[code] = sg.metadata[code]

        sample_groups = {
            cls: SampleGroup(codes=merged_codes[cls], metadata=merged_meta[cls]) for cls in merged_codes
        }
        return DatasetPlan(
            dataset_name=collection.name,
            label=_pooled_label(task, filter_combinations),
            sample_groups=sample_groups,
        )

    def run_transfer(
        self,
        source_specs: tuple[TransferSpec, ...],
        target_specs: tuple[TransferSpec, ...],
        model_save_dir: Path | None = None,
    ) -> MultiDomainSolution:
        """Train on each source, evaluate on every target (including the
        source collection itself, as an in-collection reference point).

        Every plan is resolved through _get_plan and every source's
        self-evaluation target is checked to exist BEFORE any training
        starts -- a mis-specified targets list fails fast here rather than
        wasting a full training run and only failing afterward inside
        DomainSolution.__post_init__.
        """
        resolved_targets = []
        for spec in target_specs:
            plan = self._get_plan(spec.collection, spec.task, spec.filters)
            resolved_targets.append((spec.collection, f"{spec.collection}:{plan.label}", plan))
        target_labels = {label for _, label, _ in resolved_targets}

        resolved_sources = []
        for spec in source_specs:
            plan = self._get_plan(spec.collection, spec.task, spec.filters)
            qualified_label = f"{spec.collection}:{plan.label}"
            if qualified_label not in target_labels:
                raise ValueError(
                    f"Source '{spec.collection}' (plan '{qualified_label}') has no matching "
                    f"self-evaluation target -- targets must include the source's own plan. "
                    f"Target labels available: {sorted(target_labels)}"
                )
            resolved_sources.append((spec.collection, qualified_label, plan))

        domain_solutions = []
        for src_name, qualified_train_label, src_plan in resolved_sources:
            src_experiment = self._experiments[src_name]
            exp_train_result = src_experiment.train_on_plan(src_plan)
            tr = exp_train_result.train_result

            if model_save_dir is not None:
                model_save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    tr.model.state_dict(),
                    model_save_dir / f"{sanitize_label_for_filename(qualified_train_label)}.pt",
                )

            train_metadata = {
                "train_epoch_nbr": tr.epochs_run,
                "train_loss": tr.train_loss,
                "train_acc": tr.train_acc,
                "val_loss": tr.val_loss,
                "val_acc": tr.val_acc,
                "source_collection": src_name,
            }

            confusion_matrices = {}
            for tgt_name, qualified_test_label, tgt_plan in resolved_targets:
                tgt_experiment = self._experiments[tgt_name]
                cm, _ = tgt_experiment.evaluate_on_plan(
                    tr.model,
                    exp_train_result.normalisator,
                    tgt_plan,
                    exp_train_result.cls_labels,
                )
                confusion_matrices[qualified_test_label] = cm

            domain_solutions.append(DomainSolution(
                train_dataset_name=qualified_train_label,
                class_labels=exp_train_result.cls_labels,
                seed=self._config.random_seed,
                train_metadata=train_metadata,
                confusion_matrices=confusion_matrices,
            ))

        return MultiDomainSolution(
            config_name=self._config.name,
            domain_solutions=domain_solutions,
            processor_name=self.processor_name,
        )
