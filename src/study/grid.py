from __future__ import annotations
from typing import Dict, Any, Tuple, Callable, Iterator, List
from dataclasses import dataclass

from experiment.config import ExperimentConfig
from study.design import ExperimentSpec, StudyDesign
from collection.task import Task

import itertools

@dataclass
class DependentFactor:
    """
    Specification for a factor whose value depends on other factors.
    
    Attributes:
        depends_on: Name of the factor this depends on, or tuple for nested deps
        mapping: Dict mapping factor values to this factor's value
        default: Optional default if no mapping matches
    """
    depends_on: str | Tuple[str, ...]
    mapping: Dict[Any, Any]
    default: Any = None
    
    def resolve(self, factor_values: Dict[str, Any]) -> Any:
        """
        Resolve the value based on current factor values.
        
        Args:
            factor_values: Dict of {factor_name: value} for current combination
            
        Returns:
            Resolved value for this factor
        """
        if isinstance(self.depends_on, str):
            # Simple single dependency
            key = factor_values.get(self.depends_on)
            if key in self.mapping:
                return self.mapping[key]
            return self.default
        else:
            # Nested dependency - traverse the mapping
            current = self.mapping
            for dep in self.depends_on:
                key = factor_values.get(dep)
                if key is None or key not in current:
                    return self.default
                current = current[key]
            return current


class StudyGridBuilder:
    """
    Builds experiment configurations from a parameter grid.
    
    Example:
        builder = StudyGridBuilder()
        
        # Set varying factors (all combinations explored)
        builder.set_factors(
            normalization=('dataset', 'sample', 'pretrained'),
            model_class=(CNN1D, CNN2D),
            optimizer_name=('adamw', 'sgd'),
            target_sampling_rate=(12000, 48000)
        )
        
        # Set independent factors (constant)
        builder.set_independent(
            train_val_split_ratio=0.33,
            max_epochs=2000,
            device='cuda'
        )
        
        # Set dependent factors
        builder.set_dependent(
            'normalization_vals',
            depends_on='normalization',
            mapping={'pretrained': (0, 1)},
            default=None
        )
        
        builder.set_dependent(
            'lr',
            depends_on='optimizer_name',
            mapping={'adamw': 1e-3, 'sgd': 1e-2}
        )
        
        # Nested dependency: processor_config depends on model_class AND target_sampling_rate
        builder.set_dependent(
            'processor_config',
            depends_on=('model_class', 'target_sampling_rate'),
            mapping={
                CNN1D: {
                    12000: SignalProcessorConfig.raw('raw_12k', 12000),
                    48000: SignalProcessorConfig.raw('raw_48k', 48000)
                },
                CNN2D: {
                    12000: SignalProcessorConfig.spectrogram('spec_12k', 12000),
                    48000: SignalProcessorConfig.spectrogram('spec_48k', 48000)
                }
            }
        )
        
        # Build all configs
        configs = builder.build()
    """
    
    def __init__(self):
        self._factors: Dict[str, Tuple[Any, ...]] = {}
        self._independent: Dict[str, Any] = {}
        self._dependent: Dict[str, DependentFactor] = {}
        self._name_template: str | Callable[[Dict], str] | None = None
    
    def set_factors(self, **factors: Tuple[Any, ...]) -> 'StudyGridBuilder':
        """
        Set varying factors. All combinations will be explored.
        
        Args:
            **factors: factor_name=tuple_of_values
            
        Returns:
            self for chaining
        """
        for name, values in factors.items():
            if not isinstance(values, (tuple, list)):
                raise ValueError(f"Factor '{name}' must be tuple or list, got {type(values)}")
            self._factors[name] = tuple(values)
        return self
    
    def set_independent(self, **factors: Any) -> StudyGridBuilder:
        """
        Set independent factors (constant across all configs).
        
        Args:
            **factors: factor_name=value
            
        Returns:
            self for chaining
        """
        self._independent.update(factors)
        return self
    
    def set_dependent(
        self,
        name: str,
        depends_on: str | tuple[str, ...],
        mapping: Dict[Any, Any],
        default: Any = None
    ) -> StudyGridBuilder:
        """
        Set a dependent factor.
        
        Args:
            name: Name of the factor
            depends_on: Factor(s) this depends on
            mapping: Dict mapping dependency values to this factor's value
            default: Default value if no mapping matches
            
        Returns:
            self for chaining
        """
        self._dependent[name] = DependentFactor(
            depends_on=depends_on,
            mapping=mapping,
            default=default
        )
        return self
    
    def set_name_template(
        self, 
        template: str | Callable[[Dict[str, Any]], str]
    ) -> StudyGridBuilder:
        """
        Set template for generating config names.
        
        Args:
            template: Either a format string like "{model_class.__name__}_{normalization}"
                      or a callable that takes factor dict and returns name
                      
        Returns:
            self for chaining
        """
        self._name_template = template
        return self
    
    def _generate_name(self, factors: Dict[str, Any]) -> str:
        """
        Generate config name from varying factors only.
        
        Only uses factors set via set_factors() (study_factors),
        not independent or dependent factors.
        """
        if self._name_template is not None:
            if callable(self._name_template):
                return self._name_template(factors)
            else:
                # Format string
                format_dict = {}
                for name, value in factors.items():
                    format_dict[name] = value
                    if hasattr(value, '__name__'):
                        format_dict[f'{name}.__name__'] = value.__name__
                    if hasattr(value, 'name'):
                        format_dict[f'{name}.name'] = value.name
                return self._name_template.format(**format_dict)
        
        # Default: join only varying factor values
        parts = []
        for name in self._factors.keys():  # Only varying factors
            value = factors.get(name)
            if value is None:
                continue
            if hasattr(value, '__name__'):
                parts.append(value.__name__)
            elif hasattr(value, 'name'):
                parts.append(str(value.name))
            else:
                parts.append(str(value))
        return '_'.join(parts)
    
    def _resolve_dependent(self, factor_values: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve all dependent factors for a given combination."""
        resolved = {}
        for name, dep in self._dependent.items():
            resolved[name] = dep.resolve(factor_values)
        return resolved
    
    def iter_combinations(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over all factor combinations.
        
        Yields:
            Dict with all factors resolved (varying + independent + dependent)
        """
        if not self._factors:
            # No varying factors - just yield independent + dependent defaults
            result = dict(self._independent)
            result.update(self._resolve_dependent(result))
            yield result
            return
        
        # Generate all combinations of varying factors
        factor_names = list(self._factors.keys())
        factor_values = list(self._factors.values())
        
        for combination in itertools.product(*factor_values):
            # Start with varying factors
            result = dict(zip(factor_names, combination))
            
            # Add independent factors
            result.update(self._independent)
            
            # Resolve dependent factors
            result.update(self._resolve_dependent(result))
            
            yield result
    
    def build(self) -> List[Dict[str, Any]]:
        """
        Build all configurations.
        
        Returns:
            List of dicts, each containing all factor values + generated 'name'
        """
        configs = []
        
        for factors in self.iter_combinations():
            # Generate name if not already present
            if 'name' not in factors:
                factors['name'] = self._generate_name(factors)
            configs.append(factors)
        
        return configs
    
    def build_experiment_configs(self) -> List[ExperimentConfig]:
        """
        Build ExperimentConfig objects from the grid.
        
        Returns:
            List of ExperimentConfig instances
        """
        
        configs = []
        for params in self.build():
            # Extract ExperimentConfig fields
            config = ExperimentConfig(
                name=params.get('name', 'unnamed'),
                processor_config=params['processor_config'],
                trainer_config=params['trainer_config'],
                model_config=params['model_config'],
                file_sampling=params.get('file_sampling'),
                normalization=params.get('normalization', 'none'),
                normalization_vals=params.get('normalization_vals'),
                train_val_split_ratio=params.get('train_val_split_ratio', 0.33),
                random_seed=params.get('random_seed', 42)
            )
            configs.append(config)
        
        return configs
    
    def build_study_design(
        self,
        study_name: str,
        task: Task,
        filter_combinations: Tuple[Dict, ...],
        seeds: Tuple[int, ...],
        description: str = ""
    ) -> StudyDesign:
        """
        Build complete StudyDesign from the grid.
        
        Args:
            study_name: Name for the study
            task: Task definition
            filter_combinations: Domain filter combinations
            seeds: Random seeds for repetition
            description: Optional study description
            
        Returns:
            StudyDesign ready to run
        """
        
        specs = []
        for config in self.build_experiment_configs():
            spec = ExperimentSpec(
                task=task,
                filter_combinations=filter_combinations,
                config=config
            )
            specs.append(spec)
        
        return StudyDesign(
            name=study_name,
            experiment_specs=tuple(specs),
            seeds=seeds,
            description=description
        )
    
    @property
    def num_combinations(self) -> int:
        """Total number of configurations that will be generated."""
        if not self._factors:
            return 1
        
        total = 1
        for values in self._factors.values():
            total *= len(values)
        return total
    
    def summary(self) -> str:
        """Generate summary of the grid configuration."""
        lines = [
            "StudyGridBuilder Summary",
            "=" * 40,
            f"Total combinations: {self.num_combinations}",
            "",
            "Varying factors:"
        ]
        for name, values in self._factors.items():
            lines.append(f"  {name}: {len(values)} values")
        
        lines.append("")
        lines.append("Independent factors:")
        for name, value in self._independent.items():
            lines.append(f"  {name}: {value}")
        
        lines.append("")
        lines.append("Dependent factors:")
        for name, dep in self._dependent.items():
            lines.append(f"  {name}: depends on {dep.depends_on}")
        
        return '\n'.join(lines)


# =============================================================================
# Convenience function for dict-based configuration
# =============================================================================

def build_grid_from_dicts(
    study_factors: Dict[str, Tuple],
    study_independent_factors: Dict[str, Any],
    study_dependent_factors: Dict[str, Dict],
    name_template: str | Callable | None = None
) -> StudyGridBuilder:
    """
    Build StudyGridBuilder from dictionary configuration.
    
    Args:
        study_factors: {factor_name: (value1, value2, ...)}
        study_independent_factors: {factor_name: constant_value}
        study_dependent_factors: {
            factor_name: {
                dependency_factor: {dep_value: result_value, ...}
            }
            OR for nested:
            factor_name: {
                outer_factor: {
                    outer_value: {
                        inner_factor: {inner_value: result_value}
                    }
                }
            }
        }
        name_template: Optional name template
        
    Returns:
        Configured StudyGridBuilder
    """
    builder = StudyGridBuilder()
    
    # Set varying factors
    builder.set_factors(**study_factors)
    
    # Set independent factors
    builder.set_independent(**study_independent_factors)
    
    # Parse and set dependent factors
    for factor_name, dep_spec in study_dependent_factors.items():
        # Determine dependency structure
        depends_on, mapping = _parse_dependent_spec(dep_spec)
        builder.set_dependent(factor_name, depends_on, mapping)
    
    if name_template:
        builder.set_name_template(name_template)
    
    return builder


def _parse_dependent_spec(spec: Dict) -> Tuple[str | Tuple[str, ...], Dict]:
    """
    Parse dependent factor specification to extract depends_on and mapping.
    
    Handles both simple and nested dependencies.
    """
    # Check if it's a simple dependency (values are not dicts)
    first_key = next(iter(spec.keys()))
    first_value = spec[first_key]
    
    if not isinstance(first_value, dict):
        # Simple dependency: {dep_value: result}
        # Need to find which factor this depends on - it's the outer key
        return first_key, spec[first_key] if isinstance(spec[first_key], dict) else spec
    
    # Check for nested structure
    # Format: {outer_factor: {outer_val: {inner_factor: {inner_val: result}}}}
    # or: {outer_factor: {outer_val: result}}
    
    outer_factor = first_key
    outer_mapping = first_value
    
    # Check if values are further nested
    first_outer_val = next(iter(outer_mapping.keys()))
    first_inner = outer_mapping[first_outer_val]
    
    if isinstance(first_inner, dict):
        # Check if inner dict has factor names as keys or values as keys
        first_inner_key = next(iter(first_inner.keys()))
        first_inner_val = first_inner[first_inner_key]
        
        if isinstance(first_inner_val, dict):
            # Nested: {outer_factor: {outer_val: {inner_factor: {inner_val: result}}}}
            inner_factor = first_inner_key
            
            # Restructure mapping: {outer_val: {inner_val: result}}
            new_mapping = {}
            for outer_val, inner_dict in outer_mapping.items():
                new_mapping[outer_val] = inner_dict[inner_factor]
            
            return (outer_factor, inner_factor), new_mapping
        else:
            # Simple nested: {outer_factor: {outer_val: result}}
            return outer_factor, outer_mapping
    else:
        # Simple: {dep_factor: {dep_val: result}}
        return outer_factor, outer_mapping