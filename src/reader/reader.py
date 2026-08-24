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
                try:
                    data = loadmat(fname, appendmat=True, simplify_cells=cfg.simplify_cells)
                except Exception as exc:
                    hint = ""
                    if Path(fname).stem == "N15_M01_F10_KA08_2":
                        hint = (
                            " This is the known-corrupt Paderborn realization "
                            "(N15_M01_F10_KA08_2.mat) — it should be excluded via "
                            "exclude_realizations in configs/collections/paderborn.yaml."
                        )
                    raise ValueError(f"Failed to load '{fname}': {exc}.{hint}") from exc

            if ch_cfg.key_template is not None:
                # CWRU-style: build key from template fields
                bearing = metadata['bearing_position']['value']  # e.g. 'DE'
                key = ch_cfg.key_template.format(
                    file_key=file_key,
                    bearing_position=bearing,
                )
                result[ch_name] = np.asarray(data[key].ravel(), dtype=ch_cfg.dtype)

            elif ch_cfg.variable_name is not None:
                # Paderborn-style, name-based (preferred): stem is the top-level
                # key, Y is a struct array of {'Name', 'Data'} entries.
                dict_key = Path(fname).stem
                result[ch_name] = self._lookup_by_name(
                    data[dict_key]['Y'], ch_cfg.variable_name, ch_name, ch_cfg.dtype
                )

            elif ch_cfg.variable_index is not None:
                # Paderborn-style, positional (legacy): nested Y[idx]['Data']
                dict_key = Path(fname).stem
                result[ch_name] = np.asarray(
                    data[dict_key]['Y'][ch_cfg.variable_index]['Data'],
                    dtype=ch_cfg.dtype,
                )
            else:
                raise ValueError(
                    f"Channel '{ch_name}': must have key_template, variable_name, "
                    "variable_index, or source='metadata'"
                )

        return result

    def _resolve_file_key(self, fname: str) -> str:
        fk = self._cfg.file_key
        stem = Path(fname).stem
        key = fk.overrides.get(stem, stem)
        if fk.min_digits is not None and key.isdigit():
            key = key.zfill(fk.min_digits)
        return key

    @staticmethod
    def _entry_name(entry: dict) -> str:
        name = entry['Name']
        if isinstance(name, np.ndarray):
            name = np.squeeze(name)
        return str(name)

    def _lookup_by_name(
        self, y_entries, variable_name: str, ch_name: str, dtype: str
    ) -> np.ndarray:
        for entry in y_entries:
            if self._entry_name(entry) == variable_name:
                return np.asarray(entry['Data'], dtype=dtype)
        available = [self._entry_name(e) for e in y_entries]
        raise ValueError(
            f"Channel '{ch_name}': variable_name '{variable_name}' not found in "
            f"Y struct. Available names: {available}"
        )
