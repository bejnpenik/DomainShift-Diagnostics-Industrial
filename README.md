# DomainShift-Diagnostics-Industrial

**Version 0.1.0** — A modular framework for domain shift experiments in industrial condition monitoring.

Trains and evaluates bearing fault classifiers across domains defined by operating conditions,
fault sizes, bearing positions, and measurement setups. Studies are configured entirely via YAML.

---

## Pipeline

```
Collection (YAML) → Task → DatasetPlan → DomainDataset
    → Reader → Processor → Normalization → Model → Trainer
    → DomainSolution → StudySolution
```

---

## Requirements

```
torch>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.2
pyyaml>=6.0
pydantic>=2.0
```

---

## Quickstart

### 1. Download datasets

```bash
python main.py --collection configs/collections/paderborn.yaml --download
python main.py --collection configs/collections/cwru.yaml --download
```

Files are saved to the `dirname` specified in each collection YAML (`data/paderborn/`, `data/cwru/`).
Already-downloaded files are skipped automatically.

> **Note:** CWRU files 101–104 (healthy FE bearing) are not available online and must be created manually.
> The downloader skips them by default (configured in the `skip` list in `cwru.yaml`).

### 2. Run a study

```bash
python main.py \
  --collection configs/collections/paderborn.yaml \
  --study configs/study/paderborn_study.yaml \
  --save-path results/paderborn.csv
```

Results are written as a CSV to `--save-path`.

---

## Configuration

### Collection YAML (`configs/collections/`)

Defines the dataset: file locations, integer code schema, and download sources.

```yaml
download:
  base_url: https://...
  format: individual       # or 'zip'
  filename_template: "{file_id}.mat"
  skip: [101, 102]         # optional: file IDs to skip during download

dirname: data/cwru
filetype: mat
code_schema:
  fault_element: 100
  fault_size: 10000
  ...
files:
  97: 200021
  98: 200022
  ...
```

### Task YAML (`configs/tasks/`)

Defines the classification problem: target variable, domain factors, class rules.
**All values must be integer codes** (from the collection `header` section). Use `all` to include all available codes for a field.

```yaml
collection: cwru
target: fault_element
domain_factors: [fault_size, bearing_position, condition]

defaults:
  fixed:
    fault_size: 0
  resolve:
    sampling_rate: all    # expands to all available codes

classes:
  0:                      # 0 = normal
    fixed:
      fault_size: 0

filters:
  exclude:
    fault_size: [0, 4]
```

### Study YAML (`configs/study/`)

Defines the experiment grid: which models, processors, and training hyperparameters to vary.

```yaml
name: paderborn_study
task_path: configs/tasks/paderborn_fault_element.yaml
seeds: [42, 123, 456]

factors:
  model_type: [1d, 2d]
  normalization: [dataset, sample]

independent:
  max_epochs: 2000
  weight_decay: 0.0001
  device: cuda

dependent:
  model:
    depends_on: model_type
    mapping:
      1d: configs/models/cnn1d.yaml
      2d: configs/models/cnn2d.yaml
  lr:
    depends_on: optimizer_name
    mapping:
      adamw: 0.001
      sgd: 0.01
```

---

## Datasets

| Dataset | Collection YAML | Download format |
|---------|----------------|-----------------|
| [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) | `cwru.yaml` | Individual `.mat` files |
| [Paderborn DataCenter](https://groups.uni-paderborn.de/kat/BearingDataCenter) | `paderborn.yaml` | Rar archives per bearing |
