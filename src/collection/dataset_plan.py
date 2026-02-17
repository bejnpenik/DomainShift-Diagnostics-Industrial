from dataclasses import dataclass
from typing import Dict

from collection.metadata import Metadata

@dataclass(frozen=True)
class SampleGroup:
    codes: Dict[int, str]
    metadata: Dict[int, Metadata]

@dataclass(frozen=True)
class DatasetPlan:
    dataset_name: str
    label: str
    sample_groups: Dict[str, SampleGroup]

    @property
    def is_complete(self) -> bool:
        """
        Check if all classes have at least one sample.
        
        Returns:
            True if every class has non-empty codes
        """
        return all(bool(sg.codes) for sg in self.sample_groups.values())
    
    @property
    def empty_classes(self) -> list[str]:
        """
        Get list of class labels that have no samples.
        
        Returns:
            List of class labels with empty codes
        """
        return [label for label, sg in self.sample_groups.items() if not sg.codes]
    
    @property
    def class_sample_counts(self) -> Dict[str, int]:
        """
        Get number of unique codes (not files) per class.
        
        Returns:
            {class_label: count} mapping
        """
        return {label: len(sg.codes) for label, sg in self.sample_groups.items()}