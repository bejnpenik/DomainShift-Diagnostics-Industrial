from __future__ import annotations

from dataclasses import dataclass, field

@dataclass(frozen=True)
class Rule:
    """
    Task rule specification. For default values of resolves and class specific overrides.
    """
    fixed: dict[str, int | str] = field(default_factory=dict)
    resolve: dict[str, str | list | int] = field(default_factory=dict)
@dataclass(frozen=True)
class InteractionRule:
    """Single interaction rule: when field=value, then other_field must be in allowed_values"""
    field: str
    trigger_value: int
    constrained_field: str
    allowed_values: frozenset[int]
    
    @classmethod
    def from_constraint(cls, field: str, trigger_value: int, 
                       constrained_field: str, allowed_value):
        """Create rule from constraint spec"""
        if isinstance(allowed_value, (list, tuple, set)):
            allowed_values = frozenset(allowed_value)
        else:
            allowed_values = frozenset([allowed_value])
        
        return cls(field, trigger_value, constrained_field, allowed_values)
    
    def applies_to(self, combo: dict) -> bool:
        """Check if this rule applies to the combo"""
        return combo.get(self.field) == self.trigger_value
    
    def is_satisfied_by(self, combo: dict) -> bool:
        """Check if combo satisfies this rule"""
        return combo.get(self.constrained_field) in self.allowed_values

@dataclass(frozen=True)
class Interactions:
    """Collection of interaction rules"""
    rules: tuple[InteractionRule, ...]
    
    @classmethod
    def from_dict(cls, interactions_dict: dict):
        """
        Flatten nested dict into list of rules.
        
        {
            'fault_element': {
                0: {'fault_size': 0, 'fault_position': 0},
                1: {'fault_size': [1, 2, 3]}
            }
        }
        becomes:
        [
            InteractionRule('fault_element', 0, 'fault_size', {0}),
            InteractionRule('fault_element', 0, 'fault_position', {0}),
            InteractionRule('fault_element', 1, 'fault_size', {1, 2, 3}),
        ]
        """
        if not interactions_dict:
            return cls(())
        
        rules = []
        for field, conditions in interactions_dict.items():
            for trigger_value, constraints in conditions.items():
                for constrained_field, allowed_value in constraints.items():
                    rule = InteractionRule.from_constraint(
                        field, trigger_value, constrained_field, allowed_value
                    )
                    rules.append(rule)
        
        return cls(tuple(rules))
    
    def is_satisfied_by(self, combo: dict) -> bool:
        """Check if combo satisfies all applicable rules"""
        return all(
            rule.is_satisfied_by(combo)
            for rule in self.rules
            if rule.applies_to(combo)
        )


@dataclass(frozen=True)
class Task:
    """Partitions a collection into labeled datasets.
    
    Maps to T = (F_y, Y, F_d) from Definition 4,
    plus collection-specific construction rules.
    """
    target: str
    domain_factors: tuple[str, ...]
    defaults: Rule = field(default_factory=Rule)
    classes: dict[str, Rule] = field(default_factory=dict)
    interactions: Interactions|None = None
    class_interactions: dict[str, Interactions]|None = None

    def __post_init__(self):
        if self.target in self.defaults.fixed.keys():
            raise ValueError(f'Target {self.target} in defaults.fixed')
        if self.target in self.defaults.resolve.keys():
            raise ValueError(f'Target {self.target} in defaults.resolve')
        if self.defaults.fixed.keys() & self.defaults.resolve.keys():
            duplicates = self.defaults.fixed.keys() & self.defaults.resolve.keys()
            duplicates_str = ', '.join(map(str, duplicates))
            raise ValueError(f'Factors {duplicates_str} both in default fixed and resolves.')
        for cls_label, cls_rule in self.classes.items():
            if self.target in cls_rule.fixed.keys():
                raise ValueError()
            if self.target in cls_rule.resolve.keys():
                raise
            if cls_rule.fixed.keys() & cls_rule.resolve.keys():
                duplicates = cls_rule.fixed.keys() & cls_rule.resolve.keys()
                duplicates_str = ', '.join(map(str, duplicates))
                raise ValueError(f'Factors {duplicates_str} both in fixed and resolves for class {cls_label}.')
    @property
    def label(self)->str:
        return ''
    

    def label(self, **filters)->str:
        label_str = f'{self.target}-'
        for field, value in filters.items():
            label_str += f'{field}={value}-'
        return label_str[:-1]