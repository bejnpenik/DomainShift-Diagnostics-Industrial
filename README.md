# DomainShift-Diagnostics-Industrial [WIP]


A modular framework for condition monitoring experiments with domain shifts.

## Pipeline

```
Collection (YAML) → Task → DatasetPlan → DomainDataset
    → Reader → Processor → Normalization → Model → Trainer
    → DomainSolution → StudySolution
```

## Remaining

- [ ] 

## Requirements

```
torch>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.2
pyyaml>=6.0
pydantic>=2.0
pytest>=7.0
```