from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class StorageConfig(BaseModel):
    """Declares which artifacts are persisted at the end of a study run.

    Defaults match the most informative setting — turn flags off for quick
    sweeps where disk space or runtime matters.

    Attributes:
        save_model_weights:  Save each trained model's state dict as a .pt
                             file under  models/{spec}/{seed}/{domain}.pt
        save_config_snapshot: Embed a plain-dict snapshot of ExperimentConfig
                              in each MultiDomainSolution (zero cost, always
                              recommended).
        save_study_design:   Persist design.pkl alongside results.pkl so the
                             full StudyDesign is recoverable without extra code.
    """

    model_config = ConfigDict(frozen=True)

    save_model_weights: bool = True
    save_config_snapshot: bool = True
    save_study_design: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> StorageConfig:
        import yaml
        with open(path) as f:
            return cls(**yaml.safe_load(f))
