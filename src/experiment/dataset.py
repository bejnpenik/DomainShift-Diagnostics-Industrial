from __future__ import annotations

import torch

from .sampling import FileSampler
from reader.reader import BaseFileReader
from collection.collection import DatasetCollection
from collection.dataset_plan import DatasetPlan
from collection.channels import SignalChannelConfig, MetadataChannelConfig
from normalization.normalization import Normalisator
from representation import Processor
from study.pipeline import PipelineConfig, ConditioningSource


class DomainDataset:
    def __init__(
        self,
        collection: DatasetCollection,
        file_sampler: FileSampler | None,
        reader: BaseFileReader,
        sample_processor: Processor,
        pipeline: PipelineConfig,
    ):
        self._collection = collection
        self._fsampler = file_sampler if file_sampler else FileSampler()
        self._reader = reader
        self._processor = sample_processor
        self._pipeline = pipeline

        # Resolve and validate primary channel
        if pipeline.primary not in collection.channels:
            raise ValueError(f"Pipeline primary '{pipeline.primary}' not in collection channels")
        self._primary_cfg = collection.channels[pipeline.primary]
        if not isinstance(self._primary_cfg, SignalChannelConfig):
            raise ValueError(
                f"Pipeline primary '{pipeline.primary}' must be a signal channel, not metadata"
            )

        # Resolve conditioning channels
        self._conditioning: list[
            tuple[ConditioningSource, SignalChannelConfig | MetadataChannelConfig]
        ] = []
        for src in pipeline.conditioning:
            if src.channel not in collection.channels:
                raise ValueError(
                    f"Conditioning channel '{src.channel}' not in collection channels"
                )
            self._conditioning.append((src, collection.channels[src.channel]))

        # Compute which reader channels to load
        self._reader_channels: set[str] = {self._primary_cfg.reader_channel}
        for _, ch_cfg in self._conditioning:
            if isinstance(ch_cfg, SignalChannelConfig):
                self._reader_channels.add(ch_cfg.reader_channel)

        # Multi-channel processors (e.g. OrderTrackingProcessor) declare which
        # reader channels they need. Guard: conditioning + multi-channel is not
        # supported in V1 because segment_raw() is not defined on those processors.
        if hasattr(sample_processor, 'required_reader_channels'):
            if self._conditioning:
                raise ValueError(
                    f"Processor '{sample_processor.name}' requires multiple reader "
                    "channels but conditioning channels alongside multi-channel "
                    "processors are not supported in V1. "
                    "Workaround: use a standard signal processor (e.g. raw_12k.yaml) "
                    "with conditioning, and order tracking without conditioning — "
                    "both are valid study configurations for comparison."
                )
            self._reader_channels.update(sample_processor.required_reader_channels)

    def _resolve_sampling_rate(self, ch_cfg: SignalChannelConfig, metadata) -> int:
        sr = ch_cfg.sampling_rate
        if isinstance(sr, int):
            return sr
        if sr == 'dynamic':
            entry = metadata[ch_cfg.sampling_rate_key]
            return int(entry['value'] if isinstance(entry, dict) else entry)
        raise ValueError(f"Unknown sampling_rate spec: {sr}")

    def _resolve_metadata_value(self, path: str, metadata) -> float:
        val = metadata
        for part in path.split('.'):
            val = val[part]
        if isinstance(val, dict):
            if 'value' not in val:
                raise ValueError(
                    f"Metadata path '{path}' resolved to a dict without a 'value' key: {val}"
                )
            val = val['value']
        return float(val)

    def _load_conditioning(
        self, raw: dict, metadata, n_windows: int
    ) -> torch.Tensor | None:
        parts = []
        for src, ch_cfg in self._conditioning:
            if isinstance(ch_cfg, MetadataChannelConfig):
                val = self._resolve_metadata_value(ch_cfg.metadata_path, metadata)
                parts.append(torch.full((n_windows, 1), val))
            elif isinstance(ch_cfg, SignalChannelConfig):
                signal = raw[ch_cfg.reader_channel]
                sr = self._resolve_sampling_rate(ch_cfg, metadata)
                seg = self._processor.segment_raw(signal, sr)
                if seg.shape[0] != n_windows:
                    raise ValueError(
                        f"Conditioning channel '{src.channel}' produced "
                        f"{seg.shape[0]} windows but the primary channel "
                        f"produced {n_windows}. Conditioning and primary "
                        "channels must segment to the same window count — "
                        "check that their sampling rates and recorded "
                        "durations match."
                    )
                if src.reduce == 'mean':
                    parts.append(seg.mean(dim=-1, keepdim=True))
                elif src.reduce == 'none':
                    parts.append(seg)
                else:
                    raise ValueError(f"Unknown reduce: '{src.reduce}'")
        return torch.cat(parts, dim=-1) if parts else None

    def __call__(
        self,
        dataset_plan: DatasetPlan,
        normalisator: Normalisator | None,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor | None]:
        X, Y, X_aux = [], [], []
        plan = self._fsampler(dataset_plan, seed)
        cls_labels = {}

        for i, (cls_label, sample_group) in enumerate(sorted(plan.sample_groups.items())):
            for code, paths in sample_group.codes.items():
                meta = sample_group.metadata[code]
                for path in paths:
                    raw = self._reader(
                        path, metadata=meta, channels=self._reader_channels
                    )
                    if hasattr(self._processor, 'required_reader_channels'):
                        proc_channels = {
                            ch: raw[ch]
                            for ch in self._processor.required_reader_channels
                        }
                        x = self._processor.process(proc_channels)
                    else:
                        primary_sr = self._resolve_sampling_rate(self._primary_cfg, meta)
                        x = self._processor(raw[self._primary_cfg.reader_channel], primary_sr)

                    if normalisator:
                        x = normalisator(x)

                    X.append(x)
                    Y.append(i * torch.ones(x.shape[0], dtype=torch.long))
                    cls_labels[cls_label] = i

                    cond = self._load_conditioning(raw, meta, x.shape[0])
                    if cond is not None:
                        X_aux.append(cond)

        return (
            torch.cat(X),
            torch.cat(Y),
            cls_labels,
            torch.cat(X_aux) if X_aux else None,
        )
