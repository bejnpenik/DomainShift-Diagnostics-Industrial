import torch

from experiment.sampling import FileSampler
from reader.reader import BaseFileReader
from collection.collection import DatasetCollection
from collection.dataset_plan import DatasetPlan
from normalization.normalization import Normalisator

from representation import Processor

class DomainDataset:
    def __init__(
            self, 
            collection: DatasetCollection,
            file_sampler: FileSampler|None,
            reader:BaseFileReader, 
            sample_processor: Processor):
        self._collection = collection
        self._fsampler = file_sampler if file_sampler else FileSampler()
        self._reader = reader
        self._processor = sample_processor

    def __call__(self, dataset_plan:DatasetPlan, normalisator:Normalisator|None, seed:int):
        """
        Docstring for __call__
        
        :param self: Description
        :param dataset_plan: Description
        :type dataset_plan: DatasetPlan
        """

        X, Y = [], []

        plan = self._fsampler(dataset_plan, seed)

        cls_labels = {}

        for i, (cls_label, sample_group) in enumerate(plan.sample_groups.items()):
            
            for code, paths in sample_group.codes.items():
                meta = sample_group.metadata[code]

                for path in paths:
                    x = self._reader(path, metadata=meta)

                    x = self._processor(x, meta)

                    if normalisator:
                        x = normalisator(x)

                    X.append(x)
                    Y.append(i * torch.ones(x.shape[0], dtype=torch.long))

                    cls_labels[cls_label] = i

        return torch.cat(X), torch.cat(Y), cls_labels