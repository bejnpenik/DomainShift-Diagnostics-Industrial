from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


class MatChannelConfig(BaseModel):
    """A channel loaded from a .mat file or metadata."""
    source: str = "file"               # "file" or "metadata"
    key_template: str | None = None    # CWRU: "X{file_key}_{bearing_position}_time"
    variable_index: int | None = None  # Paderborn: nested Y[idx]['Data']
    field: str | None = None           # metadata source: dot-path e.g. "condition.speed"
    dtype: str = "float32"


class FileKeyConfig(BaseModel):
    source: str = "filename"           # only "filename" supported
    min_digits: int | None = None      # zero-pad numeric stems
    overrides: dict[str, str] = Field(default_factory=dict)


class ReaderConfig(BaseModel):
    name: str
    format: str = "mat"
    loader: str = "scipy"
    simplify_cells: bool = False
    channels: dict[str, MatChannelConfig]
    file_key: FileKeyConfig = Field(default_factory=FileKeyConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ReaderConfig:
        import yaml
        with open(path) as f:
            return cls(**yaml.safe_load(f))
