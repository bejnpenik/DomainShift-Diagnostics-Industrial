"""
Result containers for domain adaptation experiments.

Provides hierarchical result structures:
- DomainSolution: Single training run results
- MultiDomainSolution: All training domains for one config/seed
- RepeatedMultiDomainSolution: Multiple seeds for one config
- StudySolution: Complete study results
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np
import numpy.typing as npt


# =============================================================================
# Level 1: Single Training Domain Result
# =============================================================================

@dataclass
class DomainSolution:
    """
    Results from ONE training run:
    - Single seed
    - Single training dataset
    - Tested on multiple datasets (including itself)
    
    Attributes:
        train_dataset_name: Label/identifier of training dataset
        class_labels: Mapping from class index to class name
        seed: Random seed used for this run
        train_metadata: Training metrics (epochs, loss, accuracy, etc.)
        confusion_matrices: {test_dataset_name: confusion_matrix}
    """
    train_dataset_name: str
    class_labels: Dict[int, str]
    seed: int
    train_metadata: Dict[str, Any]
    confusion_matrices: Dict[str, npt.NDArray]
    
    def __post_init__(self):
        # Ensure training dataset is in confusion matrices (self-evaluation)
        if self.train_dataset_name not in self.confusion_matrices:
            raise ValueError(
                f"confusion_matrices must include self-evaluation on "
                f"'{self.train_dataset_name}'"
            )
    
    @property
    def test_dataset_names(self) -> List[str]:
        """All test datasets (excluding train dataset)."""
        return [
            name for name in self.confusion_matrices.keys()
            if name != self.train_dataset_name
        ]
    
    @property
    def all_test_dataset_names(self) -> List[str]:
        """All test datasets (including train dataset)."""
        return list(self.confusion_matrices.keys())
    
    @property
    def num_classes(self) -> int:
        """Number of classes in the task."""
        return len(self.class_labels)
    
    def get_confusion_matrix(self, test_dataset_name: str) -> npt.NDArray:
        """Get confusion matrix for specific test dataset."""
        if test_dataset_name not in self.confusion_matrices:
            raise KeyError(f"No confusion matrix for '{test_dataset_name}'")
        return self.confusion_matrices[test_dataset_name]
    
    def get_self_confusion_matrix(self) -> npt.NDArray:
        """Get confusion matrix for self-evaluation (train=test)."""
        return self.confusion_matrices[self.train_dataset_name]


# =============================================================================
# Level 2: All Training Domains (Single Config, Single Seed)
# =============================================================================

@dataclass
class MultiDomainSolution:
    """
    Results from one complete experimental configuration:
    - Single seed
    - Single config
    - ALL training datasets
    
    Attributes:
        config_name: Name of the experimental configuration
        domain_solutions: List of DomainSolution for each training domain
        processor_name: Name of the modality processor used
    """
    config_name: str
    domain_solutions: List[DomainSolution]
    processor_name: str = ""  # Optional for backwards compatibility
    
    def __post_init__(self):
        if not self.domain_solutions:
            raise ValueError("domain_solutions cannot be empty")
        
        # Validate all have same seed
        seeds = {ds.seed for ds in self.domain_solutions}
        if len(seeds) > 1:
            raise ValueError(
                f"All domain_solutions must have same seed, got: {seeds}"
            )
    
    @property
    def seed(self) -> int:
        """The random seed used for all domain solutions."""
        return self.domain_solutions[0].seed
    
    @property
    def train_dataset_names(self) -> List[str]:
        """Names of all training datasets."""
        return [ds.train_dataset_name for ds in self.domain_solutions]
    
    @property
    def num_domains(self) -> int:
        """Number of training domains."""
        return len(self.domain_solutions)
    
    def get_solution(self, train_dataset_name: str) -> DomainSolution:
        """Get DomainSolution for specific training dataset."""
        for ds in self.domain_solutions:
            if ds.train_dataset_name == train_dataset_name:
                return ds
        raise KeyError(f"No solution for training dataset '{train_dataset_name}'")
    
    def get_cross_domain_matrix(self) -> Dict[str, Dict[str, npt.NDArray]]:
        """
        Get all confusion matrices organized as nested dict.
        
        Returns:
            {train_name: {test_name: confusion_matrix}}
        """
        return {
            ds.train_dataset_name: ds.confusion_matrices.copy()
            for ds in self.domain_solutions
        }


# =============================================================================
# Level 3: Multiple Seeds (Single Config)
# =============================================================================

@dataclass
class RepeatedMultiDomainSolution:
    """
    Results across multiple random seeds for one configuration.
    
    Enables statistical analysis (mean, std) of results.
    
    Attributes:
        multi_domain_solutions: List of MultiDomainSolution (one per seed)
    """
    multi_domain_solutions: List[MultiDomainSolution]
    
    def __post_init__(self):
        if not self.multi_domain_solutions:
            raise ValueError("multi_domain_solutions cannot be empty")
        
        # Validate all have same config_name
        config_names = {mds.config_name for mds in self.multi_domain_solutions}
        if len(config_names) > 1:
            raise ValueError(
                f"All multi_domain_solutions must have same config_name, got: {config_names}"
            )
        
        # Validate all have same train dataset names
        reference_names = set(self.multi_domain_solutions[0].train_dataset_names)
        for mds in self.multi_domain_solutions[1:]:
            current_names = set(mds.train_dataset_names)
            if reference_names != current_names:
                raise ValueError(
                    f"All multi_domain_solutions must have same train_dataset_names.\n"
                    f"Reference: {reference_names}\n"
                    f"Got: {current_names}"
                )
        
        # Validate all have unique seeds
        seeds = [mds.seed for mds in self.multi_domain_solutions]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Seeds must be unique, got duplicates in: {seeds}")
    
    @property
    def config_name(self) -> str:
        """Configuration name (same across all solutions)."""
        return self.multi_domain_solutions[0].config_name
    
    @property
    def train_dataset_names(self) -> List[str]:
        """Training dataset names (same across all solutions)."""
        return self.multi_domain_solutions[0].train_dataset_names
    
    @property
    def seeds(self) -> List[int]:
        """All random seeds used."""
        return [mds.seed for mds in self.multi_domain_solutions]
    
    @property
    def num_seeds(self) -> int:
        """Number of seeds (repetitions)."""
        return len(self.multi_domain_solutions)
    
    def get_by_seed(self, seed: int) -> MultiDomainSolution:
        """Get MultiDomainSolution for specific seed."""
        for mds in self.multi_domain_solutions:
            if mds.seed == seed:
                return mds
        raise KeyError(f"No solution for seed {seed}")
    
    def get_confusion_matrices_for_pair(
        self, 
        train_dataset_name: str, 
        test_dataset_name: str
    ) -> List[npt.NDArray]:
        """
        Get confusion matrices for a train-test pair across all seeds.
        
        Returns:
            List of confusion matrices (one per seed)
        """
        matrices = []
        for mds in self.multi_domain_solutions:
            ds = mds.get_solution(train_dataset_name)
            matrices.append(ds.get_confusion_matrix(test_dataset_name))
        return matrices
    
    def transpose(self) -> Dict[str, Dict[str, List[npt.NDArray]]]:
        """
        Reorganize results by train/test dataset pair.
        
        Returns:
            {train_name: {test_name: [cm_seed1, cm_seed2, ...]}}
        """
        result = {}
        
        # Get all train/test dataset names from first solution
        first_mds = self.multi_domain_solutions[0]
        
        for train_name in first_mds.train_dataset_names:
            result[train_name] = {}
            ds = first_mds.get_solution(train_name)
            
            for test_name in ds.all_test_dataset_names:
                result[train_name][test_name] = self.get_confusion_matrices_for_pair(
                    train_name, test_name
                )
        
        return result


# =============================================================================
# Level 4: Complete Study (Multiple Configs)
# =============================================================================

@dataclass
class StudySolution:
    """
    Complete study results across multiple configurations.
    
    Attributes:
        study_name: Name of the study
        timestamp: When the study was run
        repeated_solutions: List of RepeatedMultiDomainSolution (one per config)
        study_metadata: Optional metadata about the study setup
    """
    study_name: str
    timestamp: str
    repeated_solutions: List[RepeatedMultiDomainSolution]
    study_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.repeated_solutions:
            raise ValueError("repeated_solutions cannot be empty")
        
        # Validate unique config names
        config_names = [rs.config_name for rs in self.repeated_solutions]
        if len(config_names) != len(set(config_names)):
            raise ValueError(f"Config names must be unique, got: {config_names}")
    
    @property
    def config_names(self) -> List[str]:
        """All configuration names in the study."""
        return [rs.config_name for rs in self.repeated_solutions]
    
    @property
    def num_configs(self) -> int:
        """Number of configurations."""
        return len(self.repeated_solutions)
    
    def get_by_config(self, config_name: str) -> RepeatedMultiDomainSolution:
        """Get RepeatedMultiDomainSolution for specific config."""
        for rs in self.repeated_solutions:
            if rs.config_name == config_name:
                return rs
        raise KeyError(f"No results for config '{config_name}'")
    
    def get_all_seeds(self) -> List[int]:
        """Get all unique seeds used across all configs."""
        all_seeds = set()
        for rs in self.repeated_solutions:
            all_seeds.update(rs.seeds)
        return sorted(all_seeds)
    
    def get_domain_solutions_for_config_and_train_dataset(
        self,
        config_name: str,
        train_dataset_name: str
    ) -> List[DomainSolution]:
        """
        Get all DomainSolutions for specific config and training dataset.
        
        Returns one DomainSolution per seed.
        """
        rs = self.get_by_config(config_name)
        solutions = []
        for mds in rs.multi_domain_solutions:
            try:
                solutions.append(mds.get_solution(train_dataset_name))
            except KeyError:
                pass
        return solutions
    
    def summary(self) -> str:
        """Generate a text summary of the study."""
        lines = [
            f"Study: {self.study_name}",
            f"Timestamp: {self.timestamp}",
            f"Configurations: {self.num_configs}",
            f"Config names: {', '.join(self.config_names)}",
        ]
        
        for rs in self.repeated_solutions:
            lines.append(f"\n  {rs.config_name}:")
            lines.append(f"    Seeds: {rs.seeds}")
            lines.append(f"    Train datasets: {rs.train_dataset_names}")
        
        return '\n'.join(lines)


# =============================================================================
# Builder Classes (Optional - for incremental construction)
# =============================================================================

class StudySolutionBuilder:
    """Builder for constructing StudySolution incrementally."""
    
    def __init__(self, study_name: str):
        self._study_name = study_name
        self._timestamp = None
        self._repeated_solutions: Dict[str, List[MultiDomainSolution]] = {}
        self._metadata: Dict[str, Any] = {}
    
    def add_multi_domain_solution(self, mds: MultiDomainSolution) -> 'StudySolutionBuilder':
        """Add a MultiDomainSolution (will be grouped by config_name)."""
        config_name = mds.config_name
        if config_name not in self._repeated_solutions:
            self._repeated_solutions[config_name] = []
        self._repeated_solutions[config_name].append(mds)
        return self
    
    def set_timestamp(self, timestamp: str) -> 'StudySolutionBuilder':
        """Set the study timestamp."""
        self._timestamp = timestamp
        return self
    
    def set_metadata(self, key: str, value: Any) -> 'StudySolutionBuilder':
        """Add metadata."""
        self._metadata[key] = value
        return self
    
    def build(self) -> StudySolution:
        """Build the final StudySolution."""
        from datetime import datetime
        
        if self._timestamp is None:
            self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert grouped solutions to RepeatedMultiDomainSolution
        repeated_solutions = []
        for config_name, mds_list in self._repeated_solutions.items():
            repeated_solutions.append(RepeatedMultiDomainSolution(mds_list))
        
        return StudySolution(
            study_name=self._study_name,
            timestamp=self._timestamp,
            repeated_solutions=repeated_solutions,
            study_metadata=self._metadata
        )

if __name__ == '__main__':
    # Example usage
    import numpy as np
    
    # Create sample confusion matrices
    cm1 = np.array([[45, 5], [3, 47]])
    cm2 = np.array([[42, 8], [6, 44]])
    
    # Build a DomainSolution
    ds1 = DomainSolution(
        train_dataset_name='domain_A',
        class_labels={0: 'healthy', 1: 'faulty'},
        seed=42,
        train_metadata={'epochs': 100, 'train_acc': 0.95, 'val_acc': 0.92},
        confusion_matrices={
            'domain_A': cm1,  # Self-evaluation
            'domain_B': cm2,  # Cross-domain
        }
    )
    
    ds2 = DomainSolution(
        train_dataset_name='domain_B',
        class_labels={0: 'healthy', 1: 'faulty'},
        seed=42,
        train_metadata={'epochs': 120, 'train_acc': 0.93, 'val_acc': 0.90},
        confusion_matrices={
            'domain_A': cm2,
            'domain_B': cm1,
        }
    )
    
    # Build MultiDomainSolution
    mds = MultiDomainSolution(
        config_name='cnn1d_raw',
        domain_solutions=[ds1, ds2]
    )
    
    print(f"Config: {mds.config_name}")
    print(f"Seed: {mds.seed}")
    print(f"Train datasets: {mds.train_dataset_names}")
    
    # Create another with different seed
    ds1_seed2 = DomainSolution(
        train_dataset_name='domain_A',
        class_labels={0: 'healthy', 1: 'faulty'},
        seed=123,
        train_metadata={'epochs': 95, 'train_acc': 0.94, 'val_acc': 0.91},
        confusion_matrices={
            'domain_A': cm1 + np.random.randint(-2, 3, cm1.shape),
            'domain_B': cm2 + np.random.randint(-2, 3, cm2.shape),
        }
    )
    
    ds2_seed2 = DomainSolution(
        train_dataset_name='domain_B',
        class_labels={0: 'healthy', 1: 'faulty'},
        seed=123,
        train_metadata={'epochs': 115, 'train_acc': 0.92, 'val_acc': 0.89},
        confusion_matrices={
            'domain_A': cm2 + np.random.randint(-2, 3, cm2.shape),
            'domain_B': cm1 + np.random.randint(-2, 3, cm1.shape),
        }
    )
    
    mds_seed2 = MultiDomainSolution(
        config_name='cnn1d_raw',
        domain_solutions=[ds1_seed2, ds2_seed2]
    )
    
    # Build RepeatedMultiDomainSolution
    rmds = RepeatedMultiDomainSolution([mds, mds_seed2])
    
    print(f"\nRepeated solution:")
    print(f"Config: {rmds.config_name}")
    print(f"Seeds: {rmds.seeds}")
    print(f"Train datasets: {rmds.train_dataset_names}")
    
    # Transpose to get matrices by pair
    transposed = rmds.transpose()
    print(f"\nTransposed structure keys: {list(transposed.keys())}")
    print(f"domain_A -> domain_B has {len(transposed['domain_A']['domain_B'])} matrices")
    
    # Build StudySolution using builder
    builder = StudySolutionBuilder('bearing_fault_study')
    builder.add_multi_domain_solution(mds)
    builder.add_multi_domain_solution(mds_seed2)
    builder.set_metadata('description', 'Testing domain adaptation')
    
    study = builder.build()
    print(f"\n{study.summary()}")