from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.io import loadmat

from .config import ReaderConfig
from collection.metadata import Metadata


class BaseFileReader:
    pass


class UniversalFileReader(BaseFileReader):
    """Load one or more named channels from a .mat file per ReaderConfig."""

    def __init__(self, config: ReaderConfig) -> None:
        self._cfg = config

    def __call__(
        self,
        fname: str,
        metadata: Metadata,
        channels: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Load channels from a file.

        Args:
            fname: Path to .mat file.
            metadata: Sample metadata.
            channels: Set of reader channel names to load. None loads all.
        """
        cfg = self._cfg
        file_key = self._resolve_file_key(fname)
        data = None  # lazy — only load file if at least one file channel exists

        result = {}
        for ch_name, ch_cfg in cfg.channels.items():
            if channels is not None and ch_name not in channels:
                continue
            if ch_cfg.source == "metadata":
                # Dot-path traversal: "condition.speed" → metadata['condition']['speed']
                val = metadata
                for part in ch_cfg.field.split('.'):
                    val = val[part]
                result[ch_name] = np.array([val], dtype=ch_cfg.dtype)
                continue

            # File channel — open mat file lazily
            if data is None:
                data = loadmat(fname, appendmat=True, simplify_cells=cfg.simplify_cells)

            if ch_cfg.key_template is not None:
                # CWRU-style: build key from template fields
                bearing = metadata['bearing_position']['value']  # e.g. 'DE'
                key = ch_cfg.key_template.format(
                    file_key=file_key,
                    bearing_position=bearing,
                )
                result[ch_name] = np.asarray(data[key].ravel(), dtype=ch_cfg.dtype)

            elif ch_cfg.variable_index is not None:
                # Paderborn-style: stem is the top-level key, nested Y[idx]['Data']
                dict_key = Path(fname).stem
                result[ch_name] = np.asarray(
                    data[dict_key]['Y'][ch_cfg.variable_index]['Data'],
                    dtype=ch_cfg.dtype,
                )
            else:
                raise ValueError(
                    f"Channel '{ch_name}': must have key_template, variable_index, "
                    "or source='metadata'"
                )

        return result

    def _resolve_file_key(self, fname: str) -> str:
        fk = self._cfg.file_key
        stem = Path(fname).stem
        key = fk.overrides.get(stem, stem)
        if fk.min_digits is not None and key.isdigit():
            key = key.zfill(fk.min_digits)
        return key
