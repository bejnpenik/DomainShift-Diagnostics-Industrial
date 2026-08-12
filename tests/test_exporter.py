"""
Tests for exporter.py
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from results import (
    DomainSolution,
    MultiDomainSolution,
    RepeatedMultiDomainSolution,
    StudySolution,
    ResultsExporter,
    CSVExporter
)



class TestResultsExporter:
    def _make_study(self):
        cm = np.array([[45, 5], [3, 47]])
        ds_s1 = DomainSolution("A", {0: "h", 1: "f"}, 42, {}, {"A": cm, "B": cm})
        mds_s1 = MultiDomainSolution("cfg1", [ds_s1], processor_name="raw_12k")

        ds_s2 = DomainSolution("A", {0: "h", 1: "f"}, 99, {}, {"A": cm, "B": cm})
        mds_s2 = MultiDomainSolution("cfg1", [ds_s2], processor_name="raw_12k")

        rmds = RepeatedMultiDomainSolution([mds_s1, mds_s2])
        return StudySolution("test_study", "20250101", [rmds])

    def test_export_creates_file(self):
        study = self._make_study()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            exporter = ResultsExporter()
            exporter.export_study(study, path)
            assert path.exists()

    def test_export_correct_rows(self):
        study = self._make_study()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            exporter = ResultsExporter()
            exporter.export_study(study, path)
            lines = path.read_text().strip().split("\n")
            # header + 4 rows (2 seeds × 2 test datasets: A→A, A→B)
            assert len(lines) == 5

    def test_export_has_required_columns(self):
        study = self._make_study()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            exporter = ResultsExporter()
            exporter.export_study(study, path)
            header = path.read_text().strip().split("\n")[0]
            for col in ["accuracy", "seed", "train_dataset", "test_dataset"]:
                assert col in header, f"Missing column: {col}"


class TestCollectionQualifiedLabels:
    """Collection-qualified labels ("cwru:label") must survive the exporter
    unchanged. RowGenerator/CSVExporter treat train_dataset_name and
    confusion-matrix keys as opaque strings everywhere -- this confirms
    that (no parsing/splitting to break), rather than fixing anything."""

    def test_qualified_labels_survive_csv_round_trip(self):
        import csv

        cm = np.array([[9, 1], [2, 8]])
        train_label = "cwru:fault_element-pooled"
        test_label = "paderborn:fault_element-fault_size=1"
        ds = DomainSolution(
            train_label, {0: "h", 1: "f"}, 42, {},
            {train_label: cm, test_label: cm},
        )
        mds = MultiDomainSolution("cfg1", [ds], processor_name="raw_12k")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            exporter = ResultsExporter()
            exporter.export_multi_domain(mds, path, "transfer_study")

            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))

        assert {r["train_dataset"] for r in rows} == {train_label}
        assert {r["test_dataset"] for r in rows} == {train_label, test_label}
