from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Tuple


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    channels: int
    pixels: bytes

    def __post_init__(self) -> None:
        expected = self.width * self.height * self.channels
        if self.width <= 0 or self.height <= 0 or self.channels <= 0:
            raise ValueError("frame dimensions must be positive")
        if len(self.pixels) != expected:
            raise ValueError(
                "pixel byte count does not match dimensions: "
                f"expected {expected}, got {len(self.pixels)}"
            )

    @property
    def digest(self) -> str:
        header = f"{self.width}:{self.height}:{self.channels}:".encode()
        return sha256(header + self.pixels).hexdigest()

    def mean_absolute_difference(self, other: "Frame") -> float:
        if (self.width, self.height, self.channels) != (
            other.width,
            other.height,
            other.channels,
        ):
            return 1.0
        if not self.pixels:
            return 0.0
        difference = sum(abs(a - b) for a, b in zip(self.pixels, other.pixels))
        return difference / (255.0 * len(self.pixels))

    def coarse_signature(self, columns: int = 8, rows: int = 8) -> Tuple[int, ...]:
        """Return spatial intensity bins without assuming tiles or objects."""

        columns = max(1, min(columns, self.width))
        rows = max(1, min(rows, self.height))
        signature = []
        for gy in range(rows):
            y0 = gy * self.height // rows
            y1 = (gy + 1) * self.height // rows
            for gx in range(columns):
                x0 = gx * self.width // columns
                x1 = (gx + 1) * self.width // columns
                total = 0
                samples = 0
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        offset = (y * self.width + x) * self.channels
                        total += sum(self.pixels[offset : offset + self.channels])
                        samples += self.channels
                mean = total // max(1, samples)
                signature.append(mean // 16)
        return tuple(signature)


def signature_key(signature: Iterable[int]) -> str:
    return bytes(signature).hex()

