"""The size ladder (DESIGN.md section 7): one network shape at five costs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetSize:
    """Widths and depths of the stages. Most capacity sits at half and quarter resolution."""

    stem_channels: int  # the light stage at input resolution (unet layout only)
    half_blocks: int
    half_channels: int
    quarter_blocks: int
    quarter_channels: int
    # The flat layout: every layer at input resolution, chosen to cost about
    # what the two-scale layouts cost at the same rung.
    flat_blocks: int
    flat_channels: int


SIZES: dict[str, NetSize] = {
    "s": NetSize(16, 8, 32, 8, 64, 10, 24),
    "m": NetSize(16, 12, 48, 8, 96, 15, 32),
    "l": NetSize(24, 8, 64, 8, 128, 15, 40),
    "xl": NetSize(32, 12, 96, 12, 192, 17, 64),
    "xxl": NetSize(32, 16, 128, 16, 256, 20, 96),
}


def get_size(name: str) -> NetSize:
    """The rung called `name`."""
    try:
        return SIZES[name]
    except KeyError:
        raise KeyError(f"unknown size '{name}' (known: {', '.join(SIZES)})") from None
