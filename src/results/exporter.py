"""
Export module for converting study results to tabular formats.

Exports RAW metrics per seed - no aggregation.
Each row = one (seed, train_dataset, test_dataset) combination.

Output columns:
- study_name, config_name, processor_name
- seed, train_dataset, test_dataset
- accuracy
- precision_0, precision_1, ..., precision_n
- recall_0, recall_1, ..., recall_n  
- f1_0, f1_1, ..., f1_n
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, Iterator
from pathlib import Path
import numpy as np

from results.containers import (
    DomainSolution,
    MultiDomainSolution,
    RepeatedMultiDomainSolution,
    StudySolution
)
from results.metrics import MetricsCalculator


# =============================================================================
# Row Container
# =============================================================================

@dataclass
class ExportRow:
    """A single row for export."""
    data: Dict[str, Any]
    
    def keys(self) -> List[str]:
        return list(self.data.keys())
    
    def values(self) -> List[Any]:
        return list(self.data.values())
    
    def __getitem__(self, key: str) -> Any:
        return self.data[key]


# =============================================================================
# Row Generator
# =============================================================================

class RowGenerator:
    """
    Generates one row per (seed, train_dataset, test_dataset) combination.
    
    No aggregation - raw metrics per seed.
    """
    
    def __init__(self, metrics_calculator: Optional[MetricsCalculator] = None):
        self._calc = metrics_calculator or MetricsCalculator()
        self._num_classes: Optional[int] = None
    
    def from_multi_domain_solution(
        self,
        study_name: str,
        mds: MultiDomainSolution,
        processor_name: Optional[str] = None  # Override or use from mds
    ) -> Iterator[ExportRow]:
        """
        Generate rows from a MultiDomainSolution (single seed).
        
        Yields one row per (train_dataset, test_dataset) pair.
        """
        seed = mds.seed
        config_name = mds.config_name
        # Use provided processor_name or get from mds
        proc_name = processor_name if processor_name else mds.processor_name
        
        for ds in mds.domain_solutions:
            train_name = ds.train_dataset_name
            class_labels = ds.class_labels
            self._num_classes = len(class_labels)
            
            for test_name, cm in ds.confusion_matrices.items():
                metrics = self._calc.from_confusion_matrix(cm, class_labels)
                
                row_data = {
                    'study_name': study_name,
                    'config_name': config_name,
                    'processor_name': proc_name,
                    'seed': seed,
                    'train_dataset': train_name,
                    'test_dataset': test_name,
                    'num_classes': len(class_labels),
                    'accuracy': metrics.accuracy,
                }
                
                # Add per-class metrics
                for cm_class in metrics.class_metrics:
                    idx = cm_class.class_index
                    row_data[f'precision_{idx}'] = cm_class.precision
                    row_data[f'recall_{idx}'] = cm_class.recall
                    row_data[f'f1_{idx}'] = cm_class.f1
                
                yield ExportRow(row_data)
    
    def from_repeated_multi_domain_solution(
        self,
        study_name: str,
        rmds: RepeatedMultiDomainSolution,
        processor_name: Optional[str] = None  # Override or use from mds
    ) -> Iterator[ExportRow]:
        """
        Generate rows from a RepeatedMultiDomainSolution (multiple seeds).
        
        Yields one row per (seed, train_dataset, test_dataset) combination.
        NO aggregation over seeds.
        """
        for mds in rmds.multi_domain_solutions:
            yield from self.from_multi_domain_solution(study_name, mds, processor_name)
    
    def from_study_solution(
        self,
        study: StudySolution,
        processor_name: Optional[str] = None  # Override or use from mds
    ) -> Iterator[ExportRow]:
        """
        Generate rows from entire StudySolution.
        
        Yields one row per (config, seed, train_dataset, test_dataset).
        """
        for rmds in study.repeated_solutions:
            yield from self.from_repeated_multi_domain_solution(
                study.study_name, rmds, processor_name
            )
    
    def get_column_order(self) -> List[str]:
        """Return preferred column order."""
        base = [
            'study_name', 'config_name', 'processor_name', 'seed',
            'train_dataset', 'test_dataset', 'num_classes', 'accuracy'
        ]
        
        per_class = []
        if self._num_classes:
            for i in range(self._num_classes):
                per_class.extend([f'precision_{i}', f'recall_{i}', f'f1_{i}'])
        
        return base + per_class


# =============================================================================
# Exporters
# =============================================================================

class CSVExporter:
    """Export to CSV format."""
    
    def __init__(self, delimiter: str = ',', float_format: str = '%.6f'):
        self._delimiter = delimiter
        self._float_format = float_format
    
    def export(
        self, 
        rows: List[ExportRow], 
        path: Path, 
        column_order: Optional[List[str]] = None
    ):
        """Export rows to CSV file."""
        if not rows:
            raise ValueError("Cannot export empty rows")
        
        # Collect all columns
        all_cols = set()
        for row in rows:
            all_cols.update(row.keys())
        
        # Order columns
        if column_order:
            columns = [c for c in column_order if c in all_cols]
            columns += sorted(all_cols - set(columns))
        else:
            columns = sorted(all_cols)
        
        # Write
        with open(path, 'w', newline='') as f:
            f.write(self._delimiter.join(columns) + '\n')
            for row in rows:
                vals = []
                for col in columns:
                    v = row.data.get(col, '')
                    if isinstance(v, float):
                        vals.append(self._float_format % v)
                    else:
                        vals.append(str(v))
                f.write(self._delimiter.join(vals) + '\n')


class DataFrameExporter:
    """Export to pandas DataFrame."""
    
    def export(
        self, 
        rows: List[ExportRow], 
        path: Optional[Path] = None, 
        column_order: Optional[List[str]] = None
    ):
        """
        Convert rows to DataFrame, optionally save to file.
        
        Returns:
            pandas DataFrame
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for DataFrameExporter")
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame([r.data for r in rows])
        
        # Reorder columns
        if column_order:
            existing = [c for c in column_order if c in df.columns]
            remaining = [c for c in df.columns if c not in column_order]
            df = df[existing + remaining]
        
        # Save if path provided
        if path:
            path = Path(path)
            if path.suffix == '.xlsx':
                df.to_excel(path, index=False)
            else:
                df.to_csv(path, index=False)
        
        return df


# =============================================================================
# Main Interface
# =============================================================================

class ResultsExporter:
    """
    Main interface for exporting study results.
    
    Exports RAW per-seed metrics - no aggregation.
    processor_name is taken from MultiDomainSolution.processor_name by default.
    
    Example:
        exporter = ResultsExporter()
        
        # Export - processor_name comes from results
        exporter.export_repeated(rmds, 'results.csv', 'my_study')
        
        # Export complete study
        exporter.export_study(study, 'results.csv')
        
        # Get as DataFrame
        df = exporter.to_dataframe(rmds, 'my_study')
    """
    
    def __init__(self, metrics_calculator: Optional[MetricsCalculator] = None):
        self._calc = metrics_calculator or MetricsCalculator()
        self._generator = RowGenerator(self._calc)
        self._csv = CSVExporter()
        self._df = DataFrameExporter()
    
    def export_multi_domain(
        self, 
        mds: MultiDomainSolution, 
        path: Union[str, Path], 
        study_name: str, 
        processor_name: Optional[str] = None  # Override or use from mds
    ):
        """
        Export single-seed results to file.
        
        Args:
            mds: MultiDomainSolution (single seed)
            path: Output file path (.csv or .xlsx)
            study_name: Name of the study
            processor_name: Override processor name (uses mds.processor_name if None)
        """
        rows = list(self._generator.from_multi_domain_solution(
            study_name, mds, processor_name
        ))
        self._export_rows(rows, path)
    
    def export_repeated(
        self, 
        rmds: RepeatedMultiDomainSolution, 
        path: Union[str, Path], 
        study_name: str, 
        processor_name: Optional[str] = None  # Override or use from mds
    ):
        """
        Export multi-seed results to file (no aggregation).
        
        Args:
            rmds: RepeatedMultiDomainSolution (multiple seeds)
            path: Output file path (.csv or .xlsx)
            study_name: Name of the study
            processor_name: Override processor name (uses mds.processor_name if None)
        """
        rows = list(self._generator.from_repeated_multi_domain_solution(
            study_name, rmds, processor_name
        ))
        self._export_rows(rows, path)
    
    def export_study(
        self, 
        study: StudySolution, 
        path: Union[str, Path], 
        processor_name: Optional[str] = None  # Override or use from mds
    ):
        """
        Export complete study to single file (no aggregation).
        
        Args:
            study: StudySolution
            path: Output file path (.csv or .xlsx)
            processor_name: Override processor name (uses mds.processor_name if None)
        """
        rows = list(self._generator.from_study_solution(study, processor_name))
        self._export_rows(rows, path)
    
    def to_dataframe(
        self,
        data: Union[MultiDomainSolution, RepeatedMultiDomainSolution, StudySolution],
        study_name: str,
        processor_name: Optional[str] = None  # Override or use from mds
    ):
        """
        Convert results to pandas DataFrame (no aggregation).
        
        Args:
            data: Results to convert
            study_name: Name of the study
            processor_name: Override processor name (uses mds.processor_name if None)
            
        Returns:
            pandas DataFrame with one row per (seed, train, test)
        """
        if isinstance(data, MultiDomainSolution):
            rows = list(self._generator.from_multi_domain_solution(
                study_name, data, processor_name
            ))
        elif isinstance(data, RepeatedMultiDomainSolution):
            rows = list(self._generator.from_repeated_multi_domain_solution(
                study_name, data, processor_name
            ))
        elif isinstance(data, StudySolution):
            rows = list(self._generator.from_study_solution(data, processor_name))
        else:
            raise TypeError(f"Unsupported type: {type(data)}")
        
        return self._df.export(rows, column_order=self._generator.get_column_order())
    
    def _export_rows(self, rows: List[ExportRow], path: Union[str, Path]):
        """Internal: export rows to file based on extension."""
        path = Path(path)
        col_order = self._generator.get_column_order()
        
        if path.suffix == '.xlsx':
            self._df.export(rows, path, col_order)
        else:
            self._csv.export(rows, path, col_order)


# =============================================================================
# Convenience Function
# =============================================================================

def export_to_csv(
    data: Union[MultiDomainSolution, RepeatedMultiDomainSolution, StudySolution],
    path: Union[str, Path],
    study_name: str,
    processor_name: Optional[str] = None  # Override or use from mds
):
    """Convenience function to export results to CSV."""
    exp = ResultsExporter()
    
    if isinstance(data, MultiDomainSolution):
        exp.export_multi_domain(data, path, study_name, processor_name)
    elif isinstance(data, RepeatedMultiDomainSolution):
        exp.export_repeated(data, path, study_name, processor_name)
    elif isinstance(data, StudySolution):
        exp.export_study(data, path, processor_name)
    else:
        raise TypeError(f"Unsupported type: {type(data)}")


if __name__ == '__main__':
    from results import DomainSolution, MultiDomainSolution, RepeatedMultiDomainSolution
    
    np.random.seed(42)
    
    def make_cm(acc=0.9):
        n = 100
        c = int(n * acc)
        w = n - c
        return np.array([
            [c//2 + np.random.randint(-3, 4), w//4, w//4],
            [w//4, c//2 + np.random.randint(-3, 4), w//4],
            [w//4, w//4, c//2 + np.random.randint(-3, 4)]
        ])
    
    labels = {0: 'healthy', 1: 'inner', 2: 'outer'}
    
    # Create test data - 2 seeds, 2 domains each, with processor_name
    ds1_s42 = DomainSolution('domain_A', labels, 42, {'epoch': 100}, 
                              {'domain_A': make_cm(0.92), 'domain_B': make_cm(0.85)})
    ds2_s42 = DomainSolution('domain_B', labels, 42, {'epoch': 120}, 
                              {'domain_A': make_cm(0.83), 'domain_B': make_cm(0.91)})
    mds42 = MultiDomainSolution('cnn1d', [ds1_s42, ds2_s42], processor_name='raw_12k')
    
    ds1_s123 = DomainSolution('domain_A', labels, 123, {'epoch': 95}, 
                               {'domain_A': make_cm(0.91), 'domain_B': make_cm(0.84)})
    ds2_s123 = DomainSolution('domain_B', labels, 123, {'epoch': 115}, 
                               {'domain_A': make_cm(0.82), 'domain_B': make_cm(0.90)})
    mds123 = MultiDomainSolution('cnn1d', [ds1_s123, ds2_s123], processor_name='raw_12k')
    
    rmds = RepeatedMultiDomainSolution([mds42, mds123])
    
    # Export - processor_name comes from mds.processor_name
    exp = ResultsExporter()
    exp.export_repeated(rmds, 'tmp/results_raw.csv', 'test_study')
    print("Exported to /tmp/results_raw.csv")
    
    # Show DataFrame
    try:
        df = exp.to_dataframe(rmds, 'test_study')
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nPreview:")
        print(df[['seed', 'train_dataset', 'test_dataset', 'processor_name', 'accuracy', 'f1_0', 'f1_1', 'f1_2']].to_string())
    except ImportError:
        print("pandas not available for DataFrame preview")