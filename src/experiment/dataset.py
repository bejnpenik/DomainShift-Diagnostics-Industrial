from __future__ import annotations

import torch

from .sampling import FileSampler
from ..reader.reader import BaseFileReader
from ..collection.collection import DatasetCollection
from ..collection.dataset_plan import DatasetPlan
from ..normalization.normalization import Normalisator
from ..representation import Processor


class DomainDataset:
    def __init__(
        self,
        collection: DatasetCollection,
        file_sampler: FileSampler | None,
        reader: BaseFileReader,
        sample_processor: Processor,
        primary_channel: str = "vibration",
        aux_channel: str | None = None,
    ):
        self._collection = collection
        self._fsampler = file_sampler if file_sampler else FileSampler()
        self._reader = reader
        self._processor = sample_processor
        self._primary = primary_channel
        self._aux = aux_channel

    def __call__(
        self,
        dataset_plan: DatasetPlan,
        normalisator: Normalisator | None,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor | None]:
        X, Y, X_aux = [], [], []
        plan = self._fsampler(dataset_plan, seed)
        cls_labels = {}

        for i, (cls_label, sample_group) in enumerate(plan.sample_groups.items()):
            for code, paths in sample_group.codes.items():
                meta = sample_group.metadata[code]

                for path in paths:
                    raw = self._reader(path, metadata=meta)
                    x = self._processor(raw[self._primary], meta)

                    if normalisator:
                        x = normalisator(x)

                    X.append(x)
                    Y.append(i * torch.ones(x.shape[0], dtype=torch.long))
                    cls_labels[cls_label] = i

                    if self._aux and self._aux in raw:
                        aux_seg = self._processor.segment_raw(
                            raw[self._aux],
                            self._processor.config.target_sampling_rate,
                        )
                        X_aux.append(aux_seg)

        X_out = torch.cat(X)
        Y_out = torch.cat(Y)
        X_aux_out = torch.cat(X_aux) if X_aux else None
        return X_out, Y_out, cls_labels, X_aux_out
