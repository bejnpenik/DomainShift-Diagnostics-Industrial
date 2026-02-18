from __future__ import annotations

from dataclasses import dataclass

from collection.metadata import Metadata

@dataclass(frozen=True)
class SampleGroup:
    """
    Class specific files with metadata.

    Args:
        codes: Mapping from file codes to file names
        metadata: Mapping from file codes to code metadata (sampling rate, condition, ...)

    To do: This interface needs revision, because codes are maybe obsolete

    """
    codes: dict[int, str]
    metadata: dict[int, Metadata]

@dataclass(frozen=True)
class DatasetPlan:
    """
    Specific domain files from collection.

    Args:
        dataset_name: Collection name. 
        label: Domain specific name
        sample_groups: Mapping for classes and SampleGroups.

    To do:
        dataset_name is missleading it shoud be collection_name. Also label should be domain_name.
    """
    dataset_name: str
    label: str
    sample_groups: dict[str, SampleGroup]

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
    def class_sample_counts(self) -> dict[str, int]:
        """
        Get number of unique codes (not files) per class.
        
        Returns:
            {class_label: count} mapping
        """
        return {label: len(sg.codes) for label, sg in self.sample_groups.items()}