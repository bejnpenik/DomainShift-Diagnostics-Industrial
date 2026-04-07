# DomainShift-Diagnostics-Industrial

**Version 0.1.1** — A modular framework for domain shift experiments in industrial condition monitoring.

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

Defines the dataset: file locations, code schema, header with aliases, and download sources.
Integer codes are an internal implementation detail — they never appear in task YAMLs or file entries.

```yaml
download:
  base_url: https://...
  format: individual       # 'individual', 'zip', or 'rar'
  filename_template: "{file_id}.mat"
  skip: [101, 102]         # optional: file IDs to skip during download

dirname: data/cwru
filetype: mat
name: cwru

code_schema:               # internal multipliers — not used in config files
  fault_element: 100
  fault_size: 10000
  ...

header:                    # every entry has name, alias, value (plus optional extras)
  fault_element:
    0: {name: normal,     alias: NR, value: normal}
    1: {name: inner ring, alias: IR, value: inner ring}
    2: {name: outer ring, alias: OR, value: outer ring}
  sampling_rate:
    1: {name: 12000 Hz, alias: 12k, value: 12000}
    2: {name: 48000 Hz, alias: 48k, value: 48000}
  condition:
    1: {name: "0HP 1797rpm", alias: C1, value: C1, load: 0, speed: 1797}
    ...

files:                     # each file described by filter aliases
  97:
    bearing_position: DE
    condition: C1
    fault_size: S0
    fault_element: NR
    fault_position: NR
    sampling_rate: 48k
  ...
```

### Task YAML (`configs/tasks/`)

Defines the classification problem: target variable, domain factors, class rules.
All filter values and class keys use **aliases** from the collection header. Use `all` to expand to all available codes for a field.

```yaml
collection: cwru
target: fault_element
domain_factors: [fault_size, bearing_position, condition]

defaults:
  fixed:
    fault_size: S0        # alias — overridden per class
    bearing_position: FE  # alias — overridden by domain filters at runtime
    condition: C1         # alias — overridden by domain filters at runtime
  resolve:
    sampling_rate: all    # expands to all available codes

classes:
  NR:                     # alias for "normal" target class
    fixed:
      fault_size: S0
    resolve:
      fault_position: NR
      sampling_rate: 48k

class_interactions:       # optional: constrain combos per class
  IR:                     # alias for "inner ring"
    bearing_position:
      FE:                 # alias trigger value
        sampling_rate: 12k

filters:
  exclude:
    fault_size: [S0, S28] # aliases resolved before generating combinations
```

### Study YAML (`configs/study/`)

Defines the experiment grid: which models, processors, and training hyperparameters to vary.

```yaml
name: cwru_study
collection: cwru
task: configs/tasks/cwru_fault_element.yaml
seeds: [11, 32, 52]

grid:
  factors:
    model_type: [1d, 2d]
    model_variant: [1x1, multihead]

  independent:             # fixed across all grid points
    max_epochs: 3000
    weight_decay: 0.0001
    device: cuda
    verbose_level: 100    # print progress every N epochs

  dependent:               # resolved from grid factor values
    model:
      depends_on: [model_type, model_variant]
      mapping:
        1d:
          1x1: configs/models/cnn1d_1x1.yaml
          multihead: configs/models/cnn1d_multihead.yaml
        2d:
          1x1: configs/models/cnn2d_1x1.yaml
          multihead: configs/models/cnn2d_multihead.yaml
    processor:
      depends_on: [model_type, sampling_rate]
      mapping:
        1d:
          12000: configs/processors/raw_12k.yaml
        2d:
          12000: configs/processors/spec_12k.yaml
```

---

## Datasets

| Dataset | Collection YAML | Download format |
|---------|----------------|-----------------|
| [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) | `cwru.yaml` | Individual `.mat` files |
| [Paderborn DataCenter](https://groups.uni-paderborn.de/kat/BearingDataCenter) | `paderborn.yaml` | Rar archives per bearing |
