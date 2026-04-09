from __future__ import annotations
from pydantic import BaseModel, Field


class ConditioningSource(BaseModel):
    channel: str
    reduce: str = "mean"    # "mean" or "none"


class PipelineConfig(BaseModel):
    primary: str
    conditioning: list[ConditioningSource] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict) -> PipelineConfig:
        if not raw or 'primary' not in raw:
            raise ValueError(
                "Pipeline config must define 'primary' — "
                "which collection channel is the classification signal"
            )
        conditioning = []
        for item in raw.get('conditioning', []):
            if isinstance(item, str):
                conditioning.append(ConditioningSource(channel=item))
            else:
                conditioning.append(ConditioningSource(**item))
        return cls(primary=raw['primary'], conditioning=conditioning)

    @property
    def conditioning_names(self) -> list[str]:
        return [s.channel for s in self.conditioning]
