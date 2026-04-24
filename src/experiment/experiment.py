import numpy as np
import torch
import torch.nn as nn
import random
import dataclasses

from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from typing import Literal

from ..collection import Task
from ..collection import DatasetCollection
from ..collection import DatasetPlan
from ..reader import BaseFileReader
from .config import ExperimentConfig
from .sampling import FileSampler
from .dataset import DomainDataset

from ..normalization import Normalisator

from ..training import Trainer, TrainResult

from ..results import DomainSolution, MultiDomainSolution, RepeatedMultiDomainSolution

from ..representation import create_processor

_DA_METHODS = frozenset({"coral", "dann", "mmd"})
_DG_METHODS = frozenset({"mixup", "irm"})

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ExperimentTrainResult:
    """Result from training on a single dataset plan.
    
    Wraps TrainResult with experiment-level context.
    """
    train_result: TrainResult
    normalisator: Normalisator
    cls_labels: dict
    dataset_label: str

    @property
    def model(self) -> nn.Module:
        return self.train_result.model

class Experiment:
    """Orchestrates training and evaluation across single/multiple dataset plans."""
    
    def __init__(
        self,
        collection: DatasetCollection,
        reader: BaseFileReader,
        config: ExperimentConfig
    ):
        if config.pipeline is None:
            raise ValueError(
                "ExperimentConfig must have a pipeline — "
                "set pipeline.primary in study YAML under grid.independent"
            )

        self._collection = collection
        self._reader = reader
        self._config = config

        self._sample_processor = create_processor(config.processor_config)
        
        # Build the dataset pipeline
        self._file_sampler = FileSampler(config.file_sampling)

        self._domain_dataset = DomainDataset(
            collection=collection,
            file_sampler=self._file_sampler,
            reader=reader,
            sample_processor=self._sample_processor,
            pipeline=config.pipeline,
        )
    
    @property
    def processor_name(self) -> str:
        """Get processor name from the sample processor."""
        if hasattr(self._sample_processor, 'name'):
            return self._sample_processor.name
        return ""
    
    def _prepare_data_splits(self, dataset_plan: DatasetPlan):
        """Load and split data for a dataset plan."""
        set_seed(self._config.random_seed)

        X, Y, cls_labels, X_aux = self._domain_dataset(
            dataset_plan, None, self._config.random_seed
        )

        indices = np.arange(len(X))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=self._config.train_val_split_ratio,
            random_state=self._config.random_seed,
        )
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        aux_train = X_aux[train_idx] if X_aux is not None else None
        aux_val   = X_aux[val_idx]   if X_aux is not None else None

        # Handle normalization (primary signal only)
        if self._config.normalization == 'dataset':
            train_norm = Normalisator(mode='dataset')
        elif self._config.normalization == 'sample':
            train_norm = Normalisator(mode='sample')
        elif self._config.normalization == 'pretrained':
            if self._config.normalization_vals is None:
                raise ValueError('Pretrained normalization requires mean and std')
            mean, std = self._config.normalization_vals
            train_norm = Normalisator(mode='pretrained', mean=mean, std=std)
        elif self._config.normalization == 'none':
            train_norm = Normalisator(mode='none')
        else:
            raise ValueError(f'Unknown normalization mode: {self._config.normalization}')

        train_norm.fit(X_train)
        X_train = train_norm(X_train)
        X_val   = train_norm(X_val)

        return (X_train, Y_train, aux_train), (X_val, Y_val, aux_val), cls_labels, train_norm
    
    def _check_train_test_labels(self, cls_labels: dict, test_cls_labels: dict) -> bool:
        """Validate that test label set matches train label set."""
        return set(cls_labels.keys()) == set(test_cls_labels.keys())
    
    def train_on_plan(self, dataset_plan: DatasetPlan) -> TrainResult:
        """Train a model on a single dataset plan."""
        train_data, val_data, cls_labels, train_norm = self._prepare_data_splits(dataset_plan)

        num_classes = len(cls_labels)
        model = self._config.model_config.create_model(num_classes=num_classes)

        trainer = Trainer(self._config.trainer_config)
        train_result = trainer.fit(model, train_data, val_data)

        return ExperimentTrainResult(
            train_result=train_result,
            normalisator=train_norm,
            cls_labels=cls_labels,
            dataset_label=dataset_plan.label
        )
    
    def evaluate_on_plan(
        self,
        model: torch.nn.Module,
        normalisator: Normalisator,
        dataset_plan: DatasetPlan,
        cls_labels: dict
    ):
        """Evaluate a trained model on a dataset plan."""
        X_test, Y_test, test_cls_labels, X_aux_test = self._domain_dataset(
            dataset_plan, normalisator, self._config.random_seed
        )

        if not self._check_train_test_labels(cls_labels, test_cls_labels):
            raise RuntimeError('Train/Test labels mismatch')

        trainer = Trainer(self._config.trainer_config)
        confusion_mat = trainer.predict(model, X_test, Y_test, aux=X_aux_test)

        return confusion_mat, dataset_plan.label
    
    def run_pairwise(
        self,
        task: Task,
        filter_combinations: tuple[dict, ...]
    ) -> MultiDomainSolution:
        """
        Train on each filter combo, test on all combos (including itself).
        
        Returns:
            MultiDomainSolution with results for all train-test pairs
        """
        domain_solutions = []
        
        for train_filters in filter_combinations:
            # Create training dataset plan
            train_plan = self._collection.construct_dataset_plan(task, **train_filters)
            
            print(f"\nTraining on: {train_plan.label}")
            
            # Train
            experiment_train_result = self.train_on_plan(train_plan)
            train_result = experiment_train_result.train_result
            
            # Training metadata
            train_metadata = {
                'train_epoch_nbr': train_result.epochs_run,
                'train_loss': train_result.train_loss,
                'train_acc': train_result.train_acc,
                'val_loss': train_result.val_loss,
                'val_acc': train_result.val_acc,
            }
            
            # Test on all combos
            confusion_matrices = {}
            for test_filters in filter_combinations:
                test_plan = self._collection.construct_dataset_plan(task, **test_filters)
                
                confusion_mat, test_label = self.evaluate_on_plan(
                    train_result.model,
                    experiment_train_result.normalisator,
                    test_plan,
                    experiment_train_result.cls_labels
                )
                
                confusion_matrices[test_label] = confusion_mat
            
            domain_solutions.append(DomainSolution(
                train_dataset_name=experiment_train_result.dataset_label,
                class_labels=experiment_train_result.cls_labels,
                seed=self._config.random_seed,
                train_metadata=train_metadata,
                confusion_matrices=confusion_matrices
            ))
        
        return MultiDomainSolution(
            config_name=self._config.name,
            domain_solutions=domain_solutions,
            processor_name=self.processor_name  # <-- Include processor name
        )
    
    # ------------------------------------------------------------------
    # Domain adaptation helpers
    # ------------------------------------------------------------------

    def _load_target_features(
        self,
        dataset_plan: DatasetPlan,
        normalisator: Normalisator,
    ) -> torch.Tensor:
        """Load target domain features (unlabeled) normalized with source norm."""
        X_tgt, _, _, _ = self._domain_dataset(
            dataset_plan, normalisator, self._config.random_seed
        )
        return X_tgt

    # ------------------------------------------------------------------
    # Domain Adaptation: pairwise with distribution alignment
    # ------------------------------------------------------------------

    def run_pairwise_with_adaptation(
        self,
        task: Task,
        filter_combinations: tuple[dict, ...],
        method: str,
        adaptation_config,
    ) -> MultiDomainSolution:
        """Like run_pairwise but co-trains with unlabeled data from all other domains.

        For N domains this produces N trained models (same structure as run_pairwise).
        Each model is trained on one source domain with all remaining domains merged
        as unlabeled target data.  Evaluation is identical to run_pairwise.

        Args:
            task:               Task definition.
            filter_combinations: All domain filter dicts.
            method:             'coral', 'dann', or 'mmd'.
            adaptation_config:  AdaptationConfig instance.

        Returns:
            MultiDomainSolution (same container as run_pairwise).
        """
        from ..training.da_trainer import DomainAdaptiveTrainer

        domain_solutions = []

        for i, train_filters in enumerate(filter_combinations):
            train_plan = self._collection.construct_dataset_plan(task, **train_filters)
            print(f"\n[DA-{method}] Training on: {train_plan.label}")

            # Prepare source data (labeled)
            train_data, val_data, cls_labels, train_norm = self._prepare_data_splits(train_plan)

            # Collect unlabeled target features from all other domains
            tgt_xs = []
            for j, tgt_filters in enumerate(filter_combinations):
                if j == i:
                    continue
                tgt_plan = self._collection.construct_dataset_plan(task, **tgt_filters)
                tgt_xs.append(self._load_target_features(tgt_plan, train_norm))
            target_x = torch.cat(tgt_xs, dim=0)

            # Build model and train with adaptation
            num_classes = len(cls_labels)
            model = self._config.model_config.create_model(num_classes=num_classes)
            trainer = DomainAdaptiveTrainer(
                self._config.trainer_config, adaptation_config, method
            )
            train_result = trainer.fit(model, train_data, val_data, target_x)

            train_metadata = {
                "train_epoch_nbr": train_result.epochs_run,
                "train_loss": train_result.train_loss,
                "train_acc": train_result.train_acc,
                "val_loss": train_result.val_loss,
                "val_acc": train_result.val_acc,
                "adaptation": method,
            }

            # Evaluate on all domains (same as run_pairwise)
            confusion_matrices = {}
            for test_filters in filter_combinations:
                test_plan = self._collection.construct_dataset_plan(task, **test_filters)
                cm, label = self.evaluate_on_plan(
                    train_result.model, train_norm, test_plan, cls_labels
                )
                confusion_matrices[label] = cm

            domain_solutions.append(DomainSolution(
                train_dataset_name=train_plan.label,
                class_labels=cls_labels,
                seed=self._config.random_seed,
                train_metadata=train_metadata,
                confusion_matrices=confusion_matrices,
            ))

        return MultiDomainSolution(
            config_name=self._config.name,
            domain_solutions=domain_solutions,
            processor_name=self.processor_name,
        )

    # ------------------------------------------------------------------
    # Domain Generalization: leave-one-out multi-source training
    # ------------------------------------------------------------------

    def _prepare_multisource_splits(
        self,
        task: Task,
        source_filter_combinations: tuple[dict, ...],
    ):
        """Load and split data from multiple source domains.

        Returns:
            source_train_datasets: list of (X_train, Y_train) per source domain
            val_data:              (X_val, Y_val) concatenated from all sources
            cls_labels:            from first source domain (all must match)
            train_norm:            normalizer fitted on combined source X_train
        """
        all_X_train, all_Y_train, all_X_val, all_Y_val = [], [], [], []
        cls_labels_ref = None

        for filters in source_filter_combinations:
            plan = self._collection.construct_dataset_plan(task, **filters)
            train_data, val_data, cls_labels, _ = self._prepare_data_splits(plan)
            if cls_labels_ref is None:
                cls_labels_ref = cls_labels
            all_X_train.append(train_data[0])
            all_Y_train.append(train_data[1])
            all_X_val.append(val_data[0])
            all_Y_val.append(val_data[1])

        # Fit a joint normalizer on combined source training data
        X_all_train = torch.cat(all_X_train, dim=0)
        from ..normalization import Normalisator
        train_norm = Normalisator(mode=self._config.normalization)
        if self._config.normalization == "pretrained" and self._config.normalization_vals:
            mean, std = self._config.normalization_vals
            train_norm = Normalisator(mode="pretrained", mean=mean, std=std)
        train_norm.fit(X_all_train)

        # Re-normalize each domain's data with the joint normalizer
        source_train_datasets = [
            (train_norm(X), Y)
            for X, Y in zip(all_X_train, all_Y_train)
        ]
        val_data_combined = (
            train_norm(torch.cat(all_X_val, dim=0)),
            torch.cat(all_Y_val, dim=0),
        )
        return source_train_datasets, val_data_combined, cls_labels_ref, train_norm

    def run_leave_one_out_dg(
        self,
        task: Task,
        filter_combinations: tuple[dict, ...],
        method: str,
        adaptation_config,
    ) -> MultiDomainSolution:
        """Leave-one-out domain generalization.

        For N domains trains N models: each trained on N-1 source domains and
        evaluated on all N domains.  The DomainSolution.train_dataset_name is a
        '+'-joined string of the source domain labels (e.g. "N15+N15L+N15F").
        A self-evaluation confusion matrix for that combined label is included.

        Args:
            task:               Task definition.
            filter_combinations: All domain filter dicts.
            method:             'mixup' or 'irm'.
            adaptation_config:  AdaptationConfig instance.

        Returns:
            MultiDomainSolution.
        """
        from ..training.dg_trainer import DomainGeneralizationTrainer

        all_plans = [
            self._collection.construct_dataset_plan(task, **f)
            for f in filter_combinations
        ]
        domain_solutions = []

        for held_out_idx in range(len(filter_combinations)):
            source_indices = [i for i in range(len(filter_combinations)) if i != held_out_idx]
            source_filters = tuple(filter_combinations[i] for i in source_indices)

            print(
                f"\n[DG-{method}] Hold out: {all_plans[held_out_idx].label} | "
                f"sources: {[all_plans[i].label for i in source_indices]}"
            )

            # Prepare multi-source data
            source_datasets, val_data, cls_labels, train_norm = (
                self._prepare_multisource_splits(task, source_filters)
            )

            # Build model and train
            num_classes = len(cls_labels)
            model = self._config.model_config.create_model(num_classes=num_classes)
            trainer = DomainGeneralizationTrainer(
                self._config.trainer_config, adaptation_config, method
            )
            train_result = trainer.fit(model, source_datasets, val_data)

            # Label for this DomainSolution (combined source)
            src_labels = [all_plans[i].label for i in source_indices]
            combined_label = "+".join(src_labels)

            train_metadata = {
                "train_epoch_nbr": train_result.epochs_run,
                "train_loss": train_result.train_loss,
                "train_acc": train_result.train_acc,
                "val_loss": train_result.val_loss,
                "val_acc": train_result.val_acc,
                "adaptation": method,
                "source_domains": src_labels,
                "target_domain": all_plans[held_out_idx].label,
            }

            # Evaluate on all individual domains
            confusion_matrices = {}
            for plan in all_plans:
                cm, label = self.evaluate_on_plan(
                    train_result.model, train_norm, plan, cls_labels
                )
                confusion_matrices[label] = cm

            # Add self-evaluation entry under the combined source label so that
            # DomainSolution.__post_init__ validation passes.
            combined_X = torch.cat([X for X, _ in source_datasets], dim=0)
            combined_Y = torch.cat([Y for _, Y in source_datasets], dim=0)
            trainer_eval = Trainer(self._config.trainer_config)
            self_cm = trainer_eval.predict(train_result.model, combined_X, combined_Y)
            confusion_matrices[combined_label] = self_cm

            domain_solutions.append(DomainSolution(
                train_dataset_name=combined_label,
                class_labels=cls_labels,
                seed=self._config.random_seed,
                train_metadata=train_metadata,
                confusion_matrices=confusion_matrices,
            ))

        return MultiDomainSolution(
            config_name=self._config.name,
            domain_solutions=domain_solutions,
            processor_name=self.processor_name,
        )

    # ------------------------------------------------------------------
    # Unified dispatch
    # ------------------------------------------------------------------

    def run(
        self,
        task: Task,
        filter_combinations: tuple[dict, ...],
    ) -> MultiDomainSolution:
        """Dispatch to the correct experiment mode based on config.adaptation.

        adaptation='none'            → run_pairwise
        adaptation in coral/dann/mmd → run_pairwise_with_adaptation
        adaptation in mixup/irm      → run_leave_one_out_dg
        """
        method = self._config.adaptation
        adap_cfg = self._config.adaptation_config

        if method == "none" or method is None:
            return self.run_pairwise(task, filter_combinations)

        if method in _DA_METHODS:
            if adap_cfg is None:
                from ..training.da_trainer import AdaptationConfig
                adap_cfg = AdaptationConfig()
            return self.run_pairwise_with_adaptation(
                task, filter_combinations, method, adap_cfg
            )

        if method in _DG_METHODS:
            if adap_cfg is None:
                from ..training.da_trainer import AdaptationConfig
                adap_cfg = AdaptationConfig()
            return self.run_leave_one_out_dg(
                task, filter_combinations, method, adap_cfg
            )

        raise ValueError(
            f"Unknown adaptation method: '{method}'. "
            f"Expected one of: none, coral, dann, mmd, mixup, irm."
        )

    def run_single_train_multiple_test(
        self,
        task: Task,
        train_filters: dict,
        test_filter_combinations: tuple[dict, ...]
    ) -> DomainSolution:
        """Train on one combo, test on multiple."""
        train_plan = self._collection.construct_dataset_plan(task, **train_filters)
        
        print(f"\nTraining on: {train_plan.label}")
        experiment_train_result = self.train_on_plan(train_plan)

        train_result = experiment_train_result.train_result
        
        train_metadata = {
            'train_epoch_nbr': train_result.epochs_run,
            'train_loss': train_result.train_loss,
            'train_acc': train_result.train_acc,
            'val_loss': train_result.val_loss,
            'val_acc': train_result.val_acc,
        }
        
        confusion_matrices = {}
        
        for test_filters in test_filter_combinations:
            test_plan = self._collection.construct_dataset_plan(task, **test_filters)
            
            confusion_mat, test_label = self.evaluate_on_plan(
                train_result.model,
                experiment_train_result.normalisator,
                test_plan,
                experiment_train_result.cls_labels
            )
            
            confusion_matrices[test_label] = confusion_mat
        
        return DomainSolution(
            train_dataset_name=experiment_train_result.dataset_label,
            class_labels=experiment_train_result.cls_labels,
            seed=self._config.random_seed,
            train_metadata=train_metadata,
            confusion_matrices=confusion_matrices
        )


class ExperimentRunner:
    """Handles running experiments with multiple seeds."""
    
    def __init__(
        self,
        collection: DatasetCollection,
        reader: BaseFileReader,
        base_config: ExperimentConfig
    ):
        self._collection = collection
        self._reader = reader
        self._base_config = base_config
    
    def run_multi_seed_pairwise(
        self,
        task: Task,
        filter_combinations: tuple[dict, ...],
        seeds: list[int]
    ) -> RepeatedMultiDomainSolution:
        """
        Run pairwise experiments across multiple seeds.
        
        Args:
            task: Task definition
            filter_combinations: Filter combinations to test
            seeds: List of random seeds
            
        Returns:
            RepeatedMultiDomainSolution containing results for all seeds
        """
        multi_domain_solutions = []
        
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Running with seed: {seed}")
            print(f"{'='*60}")
            
            # Create new config with updated seed
            config = dataclasses.replace(self._base_config, random_seed=seed)
            
            # Create new experiment instance (processor created from config)
            experiment = Experiment(
                collection=self._collection,
                reader=self._reader,
                config=config
            )
            
            multi_domain_solution = experiment.run_pairwise(task, filter_combinations)
            multi_domain_solutions.append(multi_domain_solution)
        
        return RepeatedMultiDomainSolution(
            multi_domain_solutions=multi_domain_solutions
        )

    def run_multi_seed_with_adaptation(
        self,
        task: Task,
        filter_combinations: tuple[dict, ...],
        seeds: list[int],
        method: str,
        adaptation_config,
    ) -> RepeatedMultiDomainSolution:
        """run_pairwise_with_adaptation across multiple seeds."""
        multi_domain_solutions = []
        for seed in seeds:
            print(f"\n{'='*60}\nDA seed: {seed}\n{'='*60}")
            config = dataclasses.replace(self._base_config, random_seed=seed)
            experiment = Experiment(self._collection, self._reader, config)
            mds = experiment.run_pairwise_with_adaptation(
                task, filter_combinations, method, adaptation_config
            )
            multi_domain_solutions.append(mds)
        return RepeatedMultiDomainSolution(multi_domain_solutions=multi_domain_solutions)

    def run_multi_seed_leave_one_out_dg(
        self,
        task: Task,
        filter_combinations: tuple[dict, ...],
        seeds: list[int],
        method: str,
        adaptation_config,
    ) -> RepeatedMultiDomainSolution:
        """run_leave_one_out_dg across multiple seeds."""
        multi_domain_solutions = []
        for seed in seeds:
            print(f"\n{'='*60}\nDG seed: {seed}\n{'='*60}")
            config = dataclasses.replace(self._base_config, random_seed=seed)
            experiment = Experiment(self._collection, self._reader, config)
            mds = experiment.run_leave_one_out_dg(
                task, filter_combinations, method, adaptation_config
            )
            multi_domain_solutions.append(mds)
        return RepeatedMultiDomainSolution(multi_domain_solutions=multi_domain_solutions)

    def run_multi_seed(
        self,
        task: Task,
        filter_combinations: tuple[dict, ...],
        seeds: list[int],
    ) -> RepeatedMultiDomainSolution:
        """Dispatch across seeds based on base_config.adaptation.

        Equivalent to run_multi_seed_pairwise when adaptation='none'.
        """
        method = self._base_config.adaptation
        adap_cfg = self._base_config.adaptation_config

        if method in (None, "none"):
            return self.run_multi_seed_pairwise(task, filter_combinations, seeds)

        if method in _DA_METHODS:
            if adap_cfg is None:
                from ..training.da_trainer import AdaptationConfig
                adap_cfg = AdaptationConfig()
            return self.run_multi_seed_with_adaptation(
                task, filter_combinations, seeds, method, adap_cfg
            )

        if method in _DG_METHODS:
            if adap_cfg is None:
                from ..training.da_trainer import AdaptationConfig
                adap_cfg = AdaptationConfig()
            return self.run_multi_seed_leave_one_out_dg(
                task, filter_combinations, seeds, method, adap_cfg
            )

        raise ValueError(f"Unknown adaptation method: '{method}'")