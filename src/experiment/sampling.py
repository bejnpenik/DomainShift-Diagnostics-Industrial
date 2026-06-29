from dataclasses import dataclass
from random import Random

from collection import DatasetPlan, SampleGroup


@dataclass(frozen=True)
class FileSamplingProtocol:
    max_files_per_code:int | None

    def __post_init__(self):
        if self.max_files_per_code is not None and self.max_files_per_code < 0:
            raise ValueError(
                "max_files_per_code must be >= 0 or None"
            )

class FileSampler:
    def __init__(self, sampling_protocol:FileSamplingProtocol | None = None):
        self._protocol = sampling_protocol
    
    def __call__(self, dataset_plan : DatasetPlan, seed : int) -> DatasetPlan:
        if self._protocol is None:
            return dataset_plan
        rng = Random(seed)

        sample_groups_sampled = {}

        for cls_label, sample_group in dataset_plan.sample_groups.items():

            codes = {}

            for code, fnames in sample_group.codes.items():

                limit = self._protocol.max_files_per_code

                if limit is not None and limit < len(fnames):
                    codes[code] = rng.sample(fnames, limit)
                else:
                    codes[code] = fnames

            sample_groups_sampled[cls_label] = SampleGroup(
                codes = codes, 
                metadata=sample_group.metadata
            )

        return DatasetPlan(
            dataset_name=dataset_plan.dataset_name,
            label=dataset_plan.label,
            sample_groups=sample_groups_sampled
        )