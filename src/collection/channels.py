from __future__ import annotations
from pydantic import BaseModel


class SignalChannelConfig(BaseModel):
    reader_channel: str
    sampling_rate: int | str        # int Hz or "dynamic"
    sampling_rate_key: str | None = None  # metadata field if dynamic
    unit: str | None = None
    description: str | None = None


class MetadataChannelConfig(BaseModel):
    source: str = "metadata"
    metadata_path: str              # dot-path: "condition.speed"
    unit: str | None = None
    description: str | None = None


def parse_channel_config(name: str, raw: dict) -> SignalChannelConfig | MetadataChannelConfig:
    if raw.get('source') == 'metadata':
        return MetadataChannelConfig(**raw)
    cfg = SignalChannelConfig(**raw)
    if cfg.sampling_rate == 'dynamic' and cfg.sampling_rate_key is None:
        raise ValueError(
            f"Channel '{name}': sampling_rate is 'dynamic' but no sampling_rate_key specified"
        )
    return cfg


def parse_all_channels(raw: dict) -> dict[str, SignalChannelConfig | MetadataChannelConfig]:
    return {name: parse_channel_config(name, cfg) for name, cfg in raw.items()}
