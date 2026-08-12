"""
Transfer study runner — executes a TransferStudyDesign across seeds,
mirroring study.study.Study's seed loop, checkpointing, and failure
isolation (not shared via a common base -- study.py stays untouched, and
the two designs' types genuinely differ).

Also home to run_dry_run: validates a transfer study and prints per-plan
statistics without training (main.py's --dry-run).
"""

from __future__ import annotations

import dataclasses
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import torch

from experiment.experiment import Experiment, set_seed
from experiment.transfer import TransferExperiment, validate_transfer_study_setup
from experiment.sampling import FileSampler
from reader.reader import BaseFileReader
from collection.collection import DatasetCollection
from results.containers import MultiDomainSolution, StudySolution, StudySolutionBuilder

from .transfer_builder import TransferStudyDesign
from .storage import StorageConfig


class TransferStudy:
    """Executes and manages a complete cross-collection transfer study."""

    def __init__(
        self,
        collections: dict[str, tuple[DatasetCollection, BaseFileReader]],
        results_dir: Path = Path('./results'),
        storage_config: StorageConfig | None = None,
    ):
        self._collections = collections
        self._results_dir = results_dir
        self._results_dir.mkdir(exist_ok=True, parents=True)
        self._storage = storage_config or StorageConfig()

    def run(
        self,
        design: TransferStudyDesign,
        verbose: bool = True,
        save_dir: Path | None = None,
    ) -> StudySolution:
        """Execute the complete transfer study.

        First act, before any config/seed loop: validate_transfer_study_setup
        (probe plans through the real chokepoint + validate_transfer_setup).
        This must run before ANY training -- without it, a class-alias
        mismatch across collections would train a full model and only then
        die at evaluation via Experiment._check_train_test_labels, the
        opposite of the fail-before-compute rule the rest of this feature
        enforces. Not caught by the per-config try/except below (that's for
        isolating individual config/seed failures, not a setup-level
        problem that invalidates the whole run) -- it propagates.
        """
        if verbose:
            print("Validating transfer setup...")
        report = validate_transfer_study_setup(
            self._collections, design.class_aliases, design.target,
            design.experiment_configs[0], design.source_specs,
        )
        for w in report.warnings:
            print(f"WARNING: {w}")
        if verbose:
            print("Transfer setup validated.\n")

        builder = StudySolutionBuilder(design.name)
        builder.set_metadata('design_description', design.description)

        total = len(design.experiment_configs)
        failed: List[str] = []

        for idx, config in enumerate(design.experiment_configs):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Transfer experiment {idx + 1}/{total}: {config.name}")
                print(f"Sources: {[s.collection for s in design.source_specs]}")
                print(f"Targets: {[t.collection for t in design.target_specs]}")
                print(f"Seeds: {design.seeds}")
                print(f"{'='*70}")

            try:
                multi_domain_solutions = self._run_config_multi_seed(
                    config=config, design=design, seeds=design.seeds,
                    verbose=verbose, save_dir=save_dir,
                )
            except Exception as exc:
                print(f"\n!!! Config '{config.name}' failed entirely: {exc!r} — skipping.")
                failed.append(config.name)
                continue

            if not multi_domain_solutions:
                print(f"\n!!! Config '{config.name}' produced no results — skipping.")
                failed.append(config.name)
                continue

            for mds in multi_domain_solutions:
                builder.add_multi_domain_solution(mds)

            if save_dir is not None:
                try:
                    partial = builder.build()
                except ValueError:
                    pass
                else:
                    self.save(design.name, partial, design=design, save_dir=save_dir)

        if failed and verbose:
            print(f"\n{len(failed)}/{total} config(s) failed: {failed}")

        return builder.build()

    def _run_config_multi_seed(
        self,
        config,
        design: TransferStudyDesign,
        seeds: tuple[int, ...],
        verbose: bool = True,
        save_dir: Path | None = None,
    ) -> List[MultiDomainSolution]:
        """Run one grid-point config across multiple seeds."""
        multi_domain_solutions = []

        for seed_idx, seed in enumerate(seeds):
            if verbose:
                print(f"\n--- Seed {seed_idx + 1}/{len(seeds)}: {seed} ---")

            try:
                seeded_config = dataclasses.replace(config, random_seed=seed)
                # Breadcrumb from Phase 2, honored: fresh TransferExperiment
                # per seed, so every per-collection sub-experiment shares
                # the same seeded config -- never reuse one instance across
                # seeds.
                experiment = TransferExperiment(
                    self._collections, seeded_config, design.class_aliases, design.target,
                )

                model_save_dir = (
                    save_dir / "models" / config.name / f"seed_{seed}"
                    if save_dir is not None and self._storage.save_model_weights else None
                )

                mds = experiment.run_transfer(design.source_specs, design.target_specs, model_save_dir)

                if self._storage.save_config_snapshot:
                    mds.config_snapshot = seeded_config.to_dict()
            except Exception as exc:
                print(f"\n!!! Config '{config.name}' seed {seed} failed: {exc!r} — skipping this seed.")
                continue

            multi_domain_solutions.append(mds)

        return multi_domain_solutions

    def run_and_save(
        self, design: TransferStudyDesign, verbose: bool = True
    ) -> tuple[StudySolution, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self._results_dir / f"{design.name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        results = self.run(design, verbose, save_dir=save_dir)
        save_path = self.save(design.name, results, design, save_dir=save_dir)
        return results, save_path

    def save(
        self,
        name: str,
        results: StudySolution,
        design: Optional[TransferStudyDesign] = None,
        save_dir: Optional[Path] = None,
    ) -> Path:
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = self._results_dir / f"{name}_{timestamp}"
            save_dir.mkdir(exist_ok=True, parents=True)

        with open(save_dir / "results.pkl", 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        if design and self._storage.save_study_design:
            with open(save_dir / "design.pkl", 'wb') as f:
                pickle.dump(design, f, protocol=pickle.HIGHEST_PROTOCOL)

        summary_lines = [
            f"Transfer study: {results.study_name}",
            f"Timestamp: {results.timestamp}",
            f"Configs: {results.config_names}",
            f"Seeds: {results.get_all_seeds()}",
            "",
            results.summary(),
        ]
        with open(save_dir / "metadata.txt", 'w') as f:
            f.write('\n'.join(summary_lines))

        print(f"\nResults saved to {save_dir}")
        return save_dir

    @staticmethod
    def load(path: Path) -> tuple[StudySolution, Optional[TransferStudyDesign]]:
        path = Path(path)
        with open(path / "results.pkl", 'rb') as f:
            results = pickle.load(f)

        design = None
        design_path = path / "design.pkl"
        if design_path.exists():
            with open(design_path, 'rb') as f:
                design = pickle.load(f)

        return results, design

    def list_saved(self) -> List[Path]:
        return sorted([
            p for p in self._results_dir.iterdir()
            if p.is_dir() and (p / "results.pkl").exists()
        ])


def run_dry_run(
    design: TransferStudyDesign,
    collections: dict[str, tuple[DatasetCollection, BaseFileReader]],
) -> int:
    """Validate a transfer study and print plan statistics -- no training.

    Uses experiment_configs[0] as the representative config (the processor
    is already guaranteed identical across every grid point by the builder;
    file_sampling is asserted identical here too, since dry-run's per-plan
    counts would otherwise be ambiguous about which grid point's policy
    they reflect). Uses seeds[0] explicitly (not each config's default
    random_seed=42) so the printed sampled counts are reproducible and
    honestly labeled with the seed they reflect.
    """
    print(f"Transfer study: {design.name}")
    print(f"Collections: {sorted(collections)}")
    print(f"class_aliases: {design.class_aliases}  target: {design.target}")
    print(f"Grid: {len(design.experiment_configs)} configs x {len(design.seeds)} seeds "
          f"= {len(design.experiment_configs) * len(design.seeds)} total training runs")

    probe_config = design.experiment_configs[0]
    for cfg in design.experiment_configs[1:]:
        if cfg.file_sampling != probe_config.file_sampling:
            raise ValueError(
                "--dry-run requires identical file_sampling across every grid point (the "
                "per-plan counts below would otherwise be ambiguous about which grid "
                f"point's policy they reflect); found {cfg.file_sampling} vs "
                f"{probe_config.file_sampling}."
            )

    seed = design.seeds[0]
    dry_run_config = dataclasses.replace(probe_config, random_seed=seed)
    print(f"(sampled file/segment counts below reflect seed={seed}, the first configured seed)")

    print("\n--- Validating setup ---")
    report = validate_transfer_study_setup(
        collections, design.class_aliases, design.target, dry_run_config, design.source_specs,
    )
    for w in report.warnings:
        print(f"WARNING: {w}")
    print("Setup validation passed.")

    print("\n--- Plan statistics ---")
    cols = {name: c for name, (c, _) in collections.items()}
    readers = {name: r for name, (_, r) in collections.items()}
    te = TransferExperiment(collections, dry_run_config, design.class_aliases, design.target)

    seen: dict[str, tuple[list[str], object, str]] = {}
    for role, specs in (("source", design.source_specs), ("target", design.target_specs)):
        for spec in specs:
            plan = te._get_plan(spec.collection, spec.task, spec.filters)
            qualified = f"{spec.collection}:{plan.label}"
            if qualified not in seen:
                seen[qualified] = ([role], plan, spec.collection)
            elif role not in seen[qualified][0]:
                seen[qualified][0].append(role)

    set_seed(seed)
    shapes: dict[str, tuple[int, ...]] = {}
    for qualified, (roles, plan, cname) in seen.items():
        print(f"\n{qualified}  (used as: {', '.join(roles)})")
        print(f"  unique codes per class: {plan.class_sample_counts}")

        sampler = FileSampler(dry_run_config.file_sampling)
        sampled_plan = sampler(plan, seed)
        file_counts = {cls: sum(len(f) for f in sg.codes.values()) for cls, sg in sampled_plan.sample_groups.items()}
        print(f"  files after file_sampling: {file_counts}")

        print(f"  loading through {cname}'s processor (this may take a moment)...")
        experiment = Experiment(cols[cname], readers[cname], dry_run_config)
        X, Y, cls_labels, _ = experiment.load_plan_arrays(plan)
        idx_to_name = {i: n for n, i in cls_labels.items()}
        counts = torch.bincount(Y)
        segment_counts = {idx_to_name[i]: int(counts[i]) for i in range(len(counts)) if i in idx_to_name}
        print(f"  segments per class: {segment_counts}")
        ratio = max(segment_counts.values()) / max(1, min(segment_counts.values()))
        print(f"  imbalance ratio (max/min): {ratio:.2f}")

        shapes[qualified] = tuple(X.shape[1:])

    ref_label, ref_shape = next(iter(shapes.items()))
    bad = {k: v for k, v in shapes.items() if v != ref_shape}
    if bad:
        raise ValueError(f"Inconsistent output shapes across plans: {shapes}")
    print(f"\nAll plans share output shape {ref_shape}.")

    print("\nDry run complete — no training performed.")
    return 0
