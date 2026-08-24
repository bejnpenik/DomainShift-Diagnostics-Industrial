"""
Tests for reader.reader.UniversalFileReader / reader.config.

No real Paderborn .mat files needed — builds synthetic fixtures with
scipy.io.savemat mimicking the Paderborn layout: top-level key = file stem,
'Y' = list of {'Name': str, 'Data': array} entries.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from reader.config import ReaderConfig, MatChannelConfig
from reader.reader import UniversalFileReader
from collection.metadata import Metadata

# Real Paderborn Y-struct order (per the external review):
# 0=force, 1=phase_current_1, 2=phase_current_2, 3=speed,
# 4=temp_2_bearing_module, 5=torque, 6=vibration_1
_ENTRIES = [
    ("force", np.array([1.0, 2.0, 3.0])),
    ("phase_current_1", np.array([4.0, 5.0, 6.0])),
    ("phase_current_2", np.array([7.0, 8.0, 9.0])),
    ("speed", np.array([1500.0, 1500.0, 1500.0])),
    ("temp_2_bearing_module", np.array([30.0, 31.0, 32.0])),
    ("torque", np.array([0.7, 0.7, 0.7])),
    ("vibration_1", np.array([0.1, -0.1, 0.2, -0.2])),
]


def _write_paderborn_mat(tmp_path, stem: str) -> str:
    y = [{"Name": name, "Data": data} for name, data in _ENTRIES]
    path = tmp_path / f"{stem}.mat"
    savemat(str(path), {stem: {"Y": y}})
    return str(path)


class TestNameBasedLookup:
    def test_returns_correct_arrays_by_name(self, tmp_path):
        path = _write_paderborn_mat(tmp_path, "N09_M07_F10_K001_1")
        cfg = ReaderConfig(
            name="t", simplify_cells=True,
            channels={
                "vibration": MatChannelConfig(variable_name="vibration_1"),
                "rpm": MatChannelConfig(variable_name="speed"),
                "force": MatChannelConfig(variable_name="force"),
            },
        )
        reader = UniversalFileReader(cfg)
        out = reader(path, metadata=Metadata({}), channels=None)
        np.testing.assert_allclose(out["vibration"], [0.1, -0.1, 0.2, -0.2])
        np.testing.assert_allclose(out["rpm"], [1500.0, 1500.0, 1500.0])
        np.testing.assert_allclose(out["force"], [1.0, 2.0, 3.0])

    def test_unknown_name_raises_with_available_list(self, tmp_path):
        path = _write_paderborn_mat(tmp_path, "N09_M07_F10_K001_1")
        cfg = ReaderConfig(
            name="t", simplify_cells=True,
            channels={"vibration": MatChannelConfig(variable_name="not_a_real_channel")},
        )
        reader = UniversalFileReader(cfg)
        with pytest.raises(ValueError, match="Available names"):
            reader(path, metadata=Metadata({}), channels=None)

    def test_positional_lookup_still_works(self, tmp_path):
        """Backward compat: variable_index by position still works."""
        path = _write_paderborn_mat(tmp_path, "N09_M07_F10_K001_1")
        cfg = ReaderConfig(
            name="t", simplify_cells=True,
            channels={
                "vibration": MatChannelConfig(variable_index=6),  # vibration_1
                "rpm": MatChannelConfig(variable_index=3),        # speed
            },
        )
        reader = UniversalFileReader(cfg)
        out = reader(path, metadata=Metadata({}), channels=None)
        np.testing.assert_allclose(out["vibration"], [0.1, -0.1, 0.2, -0.2])
        np.testing.assert_allclose(out["rpm"], [1500.0, 1500.0, 1500.0])

    def test_variable_name_preferred_over_variable_index(self, tmp_path):
        """When both are set, variable_name wins even if variable_index is wrong."""
        path = _write_paderborn_mat(tmp_path, "N09_M07_F10_K001_1")
        cfg = ReaderConfig(
            name="t", simplify_cells=True,
            channels={
                # index 2 is phase_current_2, but variable_name should win
                "force": MatChannelConfig(variable_name="force", variable_index=2),
            },
        )
        reader = UniversalFileReader(cfg)
        out = reader(path, metadata=Metadata({}), channels=None)
        np.testing.assert_allclose(out["force"], [1.0, 2.0, 3.0])

    def test_loadmat_failure_wraps_filename(self, tmp_path, monkeypatch):
        import reader.reader as reader_mod

        def _boom(*a, **kw):
            raise ValueError("bad file")

        monkeypatch.setattr(reader_mod, "loadmat", _boom)
        cfg = ReaderConfig(
            name="t", simplify_cells=True,
            channels={"vibration": MatChannelConfig(variable_name="vibration_1")},
        )
        reader = UniversalFileReader(cfg)
        with pytest.raises(ValueError, match="Failed to load"):
            reader("some/path/N15_M01_F10_KA08_2.mat", metadata=Metadata({}), channels=None)
