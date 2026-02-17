from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic import Discriminator
from typing import Annotated
from typing import Literal

from representation.signal.view import RawSignalView, STFTSignalView


# View-specific configs
class RawViewConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["raw"] = "raw"

    def create_view(self) -> RawSignalView:
        return RawSignalView()



class STFTViewConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["stft"] = "stft"
    n_fft: int = Field(default=256, gt=0)
    hop_length: int = Field(default=128, gt=0)
    win_length: int = Field(default=256, gt=0)


    def create_view(self) -> STFTSignalView:
        return STFTSignalView(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )


ViewConfig = Annotated[RawViewConfig | STFTViewConfig, Discriminator("type")]

class SignalProcessorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    name: str
    target_sampling_rate: int = Field(default=12000, gt=0)
    window_duration: float = Field(default=0.05, gt=0)
    window_overlap: float = Field(default=0.5, ge=0, lt=1)
    max_signal_bandwidth_factor: float = Field(default=0.5, gt=0)
    view: ViewConfig
