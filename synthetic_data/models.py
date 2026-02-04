from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, validator


class TitleVariants(BaseModel):
    seed_title: str = Field(..., description="Original clean job title")
    in_the_wild_titles: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="Generated noisy job titles",
    )

    @validator("in_the_wild_titles", each_item=True)
    def strip_titles(cls, value: str) -> str:
        return value.strip()


class BatchTitleVariants(BaseModel):
    titles: List[TitleVariants] = Field(..., min_items=1)
