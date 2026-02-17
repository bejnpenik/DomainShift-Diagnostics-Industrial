"""
Metrics computation module for classification results.

Computes core classification metrics from confusion matrices:
- Accuracy (overall)
- Precision (per-class)
- Recall (per-class)
- F1-Score (per-class)

No derived/aggregated metrics - aggregation is done in post-analysis.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt

from results.containers import DomainSolution, MultiDomainSolution


# =============================================================================
# Core Metric Functions
# =============================================================================

def compute_accuracy(cm: npt.NDArray) -> float:
    """
    Compute overall accuracy from confusion matrix.
    
    Args:
        cm: Confusion matrix of shape (n_classes, n_classes)
        
    Returns:
        Accuracy as float in [0, 1]
    """
    total = np.sum(cm)
    if total == 0:
        return 0.0
    return float(np.trace(cm) / total)


def compute_precision_per_class(cm: npt.NDArray, eps: float = 1e-10) -> npt.NDArray:
    """
    Compute precision for each class.
    
    Precision_i = TP_i / (TP_i + FP_i) = cm[i,i] / sum(cm[:,i])
    """
    col_sums = cm.sum(axis=0)
    return np.diag(cm) / (col_sums + eps)


def compute_recall_per_class(cm: npt.NDArray, eps: float = 1e-10) -> npt.NDArray:
    """
    Compute recall for each class.
    
    Recall_i = TP_i / (TP_i + FN_i) = cm[i,i] / sum(cm[i,:])
    """
    row_sums = cm.sum(axis=1)
    return np.diag(cm) / (row_sums + eps)


def compute_f1_per_class(cm: npt.NDArray, eps: float = 1e-10) -> npt.NDArray:
    """
    Compute F1-score for each class.
    
    F1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
    """
    precision = compute_precision_per_class(cm, eps)
    recall = compute_recall_per_class(cm, eps)
    return 2 * precision * recall / (precision + recall + eps)


def compute_support_per_class(cm: npt.NDArray) -> npt.NDArray:
    """Compute support (number of true samples) for each class."""
    return cm.sum(axis=1)


# =============================================================================
# Metrics Container
# =============================================================================

@dataclass(frozen=True)
class ClassMetrics:
    """Metrics for a single class."""
    class_index: int
    class_label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ConfusionMatrixMetrics:
    """
    Core metrics from a single confusion matrix.
    
    Attributes:
        accuracy: Overall accuracy
        class_metrics: Per-class metrics
    """
    accuracy: float
    class_metrics: Tuple[ClassMetrics, ...]
    
    @property
    def num_classes(self) -> int:
        return len(self.class_metrics)
    
    def get_class(self, class_index: int) -> ClassMetrics:
        """Get metrics for specific class by index."""
        for cm in self.class_metrics:
            if cm.class_index == class_index:
                return cm
        raise KeyError(f"No metrics for class index {class_index}")
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert to flat dictionary.
        
        Keys: accuracy, precision_0, recall_0, f1_0, support_0, ...
        """
        d = {'accuracy': self.accuracy}
        for cm in self.class_metrics:
            d[f'precision_{cm.class_index}'] = cm.precision
            d[f'recall_{cm.class_index}'] = cm.recall
            d[f'f1_{cm.class_index}'] = cm.f1
            d[f'support_{cm.class_index}'] = cm.support
        return d


# =============================================================================
# MetricsCalculator
# =============================================================================

class MetricsCalculator:
    """Computes metrics from confusion matrices."""
    
    def __init__(self, eps: float = 1e-10):
        self._eps = eps
    
    def from_confusion_matrix(
        self, 
        cm: npt.NDArray,
        class_labels: Optional[Dict[int, str]] = None
    ) -> ConfusionMatrixMetrics:
        """Compute metrics from a single confusion matrix."""
        n_classes = cm.shape[0]
        
        if class_labels is None:
            class_labels = {i: f'class_{i}' for i in range(n_classes)}
        
        accuracy = compute_accuracy(cm)
        precision = compute_precision_per_class(cm, self._eps)
        recall = compute_recall_per_class(cm, self._eps)
        f1 = compute_f1_per_class(cm, self._eps)
        support = compute_support_per_class(cm)
        
        class_metrics = tuple(
            ClassMetrics(
                class_index=i,
                class_label=class_labels.get(i, f'class_{i}'),
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1=float(f1[i]),
                support=int(support[i])
            )
            for i in range(n_classes)
        )
        
        return ConfusionMatrixMetrics(accuracy=accuracy, class_metrics=class_metrics)
    
    def from_domain_solution(self, ds: DomainSolution) -> Dict[str, ConfusionMatrixMetrics]:
        """Compute metrics for all test datasets in a DomainSolution."""
        return {
            test_name: self.from_confusion_matrix(cm, ds.class_labels)
            for test_name, cm in ds.confusion_matrices.items()
        }
    
    def from_multi_domain_solution(self, mds: MultiDomainSolution) -> Dict[str, Dict[str, ConfusionMatrixMetrics]]:
        """Compute metrics for all train-test pairs."""
        return {
            ds.train_dataset_name: self.from_domain_solution(ds)
            for ds in mds.domain_solutions
        }


def compute_metrics(cm: npt.NDArray, class_labels: Optional[Dict[int, str]] = None) -> ConfusionMatrixMetrics:
    """Convenience function."""
    return MetricsCalculator().from_confusion_matrix(cm, class_labels)


if __name__ == '__main__':
    cm = np.array([[45, 3, 2], [2, 48, 0], [1, 1, 48]])
    class_labels = {0: 'healthy', 1: 'inner', 2: 'outer'}
    
    metrics = compute_metrics(cm, class_labels)
    
    print(f"Accuracy: {metrics.accuracy:.4f}")
    for c in metrics.class_metrics:
        print(f"  {c.class_label}: P={c.precision:.3f}, R={c.recall:.3f}, F1={c.f1:.3f}")
    
    print("\nFlat dict:", metrics.to_dict())