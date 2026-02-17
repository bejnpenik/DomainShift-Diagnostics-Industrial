from dataclasses import dataclass, field

from experiment.config import ExperimentConfig

from collection.task import Task



@dataclass(frozen=True)
class ExperimentSpec:
    """
    Complete specification for a single experimental setup.
    
    Attributes:
        task: Task definition (target, defaults, classes)
        filter_combinations: Tuple of filter dicts to test
        config: ExperimentConfig with model, processor, and training settings
    """
    task: Task
    filter_combinations: tuple[dict, ...]
    config: ExperimentConfig  # Contains name, model, processor, training params
    
    @property
    def name(self) -> str:
        """Experiment name from config."""
        return self.config.name
    
    def label(self) -> str:
        """Generate a descriptive label."""
        return f"{self.name}_{self.task.target}_{self.config.model_name}"
    
    @property
    def num_domains(self) -> int:
        """Number of domain/filter combinations."""
        return len(self.filter_combinations)
    
    @property
    def processor_name(self) -> str:
        """Name from processor config."""
        return self.config.processor_name


@dataclass(frozen=True)
class StudyDesign:
    """
    Complete specification for a research study.
    
    Attributes:
        name: Study name/identifier
        experiment_specs: Tuple of ExperimentSpec (one per config)
        seeds: Tuple of random seeds for repetition
        description: Optional description of the study
        metadata: Optional additional metadata
    """
    name: str
    experiment_specs: tuple[ExperimentSpec, ...]
    seeds: tuple[int, ...]
    description: str = ""
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate unique experiment names
        names = [spec.name for spec in self.experiment_specs]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Experiment specs must have unique names. Duplicates: {set(duplicates)}")
        
        # Validate seeds are unique
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Seeds must be unique")
    
    @property
    def num_configs(self) -> int:
        """Number of experimental configurations."""
        return len(self.experiment_specs)
    
    @property
    def num_seeds(self) -> int:
        """Number of random seeds."""
        return len(self.seeds)
    
    @property
    def total_runs(self) -> int:
        """Total number of training runs."""
        return sum(
            spec.num_domains * self.num_seeds
            for spec in self.experiment_specs
        )
    
    def summary(self) -> str:
        """Generate a text summary of the study design."""
        lines = [
            f"Study Design: {self.name}",
            f"Description: {self.description}" if self.description else "",
            f"Configurations: {self.num_configs}",
            f"Seeds: {self.seeds}",
            f"Total training runs: {self.total_runs}",
            "",
            "Experiment Specs:"
        ]
        for spec in self.experiment_specs:
            lines.append(
                f"  - {spec.name}: {spec.num_domains} domains, "
                f"{spec.config.model_class.__name__}, "
                f"processor={spec.processor_name}"
            )
        return '\n'.join(lines)