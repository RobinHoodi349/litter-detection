from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


class DetectionResult(BaseModel):
    timestamp: float
    litter_detected: bool
    pixel_coverage: float
    mask_shape: tuple[int, int]
    frame_height: int
    frame_width: int


class VerifiedDetection(BaseModel):
    litter_confirmed: bool
    confidence: Literal["high", "medium", "low"]
    description: str


@dataclass
class VerifierDeps:
    detection: DetectionResult
