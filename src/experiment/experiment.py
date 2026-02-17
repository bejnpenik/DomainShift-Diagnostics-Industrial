import numpy as np
import torch
import torch.nn as nn
import random
import dataclasses

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from collection import Task
from collection import DatasetCollection
from collection import DatasetPlan
from reader import BaseFileReader
from experiment import ExperimentConfig
from experiment import FileSampler
from experiment import DomainDataset

from normalization import Normalisator

from training import Trainer, TrainResult

from results import DomainSolution, MultiDomainSolution, RepeatedMultiDomainSolution

from representation import create_processor

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
        self._collection = collection
        self._reader = reader
        self._config = config
        
        # Create processor from config
        #self._sample_processor = config.processor_config.create_processor()
        self._sample_processor = create_processor(config.processor_config)
        
        # Build the dataset pipeline
        self._file_sampler = FileSampler(config.file_sampling)
        
        self._domain_dataset = DomainDataset(
            collection=collection,
            file_sampler=self._file_sampler,
            reader=reader,
            sample_processor=self._sample_processor,
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
        
        self._normalisator = None
        
        X, Y, cls_labels = self._domain_dataset(
            dataset_plan, self._normalisator, self._config.random_seed
        )
        
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y,
            test_size=self._config.train_val_split_ratio,
            random_state=self._config.random_seed
        )
        
        # Handle normalization
        if self._config.normalization == 'dataset':
            train_norm = Normalisator(mode='dataset')
            train_norm.fit(X_train)
            X_train = train_norm(X_train)
            X_val = train_norm(X_val)
            
        elif self._config.normalization == 'sample':
            train_norm = Normalisator(mode='sample')
            train_norm.fit(X_train)
            X_train = train_norm(X_train)
            X_val = train_norm(X_val)
            
        elif self._config.normalization == 'pretrained':
            if self._config.normalization_vals is None:
                raise ValueError('Pretrained normalization requires mean and std')
            mean, std = self._config.normalization_vals
            train_norm = Normalisator(mode='pretrained', mean=mean, std=std)
            train_norm.fit(X_train)
            X_train = train_norm(X_train)
            X_val = train_norm(X_val)
        else:
            raise ValueError(f'Unknown normalization mode: {self._config.normalization}')
        
        return (X_train, Y_train), (X_val, Y_val), cls_labels, train_norm
    
    def _check_train_test_labels(self, cls_labels: dict, test_cls_labels: dict) -> bool:
        """Validate that test labels match train labels."""
        return True
    
    def train_on_plan(self, dataset_plan: DatasetPlan) -> TrainResult:
        """Train a model on a single dataset plan."""
        train_data, val_data, cls_labels, train_norm = self._prepare_data_splits(dataset_plan)
        
        # Create model
        num_classes = len(cls_labels)
        model = self._config.model_class(num_classes=num_classes, **self._config.model_params)
        
        # Train
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
        X_test, Y_test, test_cls_labels = self._domain_dataset(
            dataset_plan, normalisator, self._config.random_seed
        )
        
        if not self._check_train_test_labels(cls_labels, test_cls_labels):
            raise RuntimeError('Train/Test labels mismatch')
        
        trainer = Trainer(self._config.trainer_config)
        
        confusion_mat = trainer.predict(model, X_test, Y_test)
        
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
            train_result = self.train_on_plan(train_plan)
            
            # Training metadata
            train_metadata = {
                'train_epoch_nbr': train_result.train_epoch_nbr,
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
                    train_result.normalisator,
                    test_plan,
                    train_result.cls_labels
                )
                
                confusion_matrices[test_label] = confusion_mat
            
            domain_solutions.append(DomainSolution(
                train_dataset_name=train_result.dataset_label,
                class_labels=train_result.cls_labels,
                seed=self._config.random_seed,
                train_metadata=train_metadata,
                confusion_matrices=confusion_matrices
            ))
        
        return MultiDomainSolution(
            config_name=self._config.name,
            domain_solutions=domain_solutions,
            processor_name=self.processor_name  # <-- Include processor name
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
            'train_epoch_nbr': train_result.train_epoch_nbr,
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
                train_result.normalisator,
                test_plan,
                train_result.cls_labels
            )
            
            confusion_matrices[test_label] = confusion_mat
        
        return DomainSolution(
            train_dataset_name=train_result.dataset_label,
            class_labels=train_result.cls_labels,
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