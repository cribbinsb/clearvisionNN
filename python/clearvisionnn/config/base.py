"""The base of every configuration model, and the constrained types they share."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# Constrained scalars: a value outside its range is a configuration error
# reported at load time, not a failure hours into a run.
Positive = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
Share = Annotated[float, Field(ge=0, le=1)]


class Config(BaseModel):
    """Immutable; unknown keys are errors, so a typo cannot silently do nothing."""

    model_config = ConfigDict(extra="forbid", frozen=True)
