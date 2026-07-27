"""Pydantic schemas for the recommendation API."""

from floter_ai.config import DEFAULT_CANDIDATE_POOL_SIZE, DEFAULT_TOP_K

from pydantic import BaseModel, Field, field_validator


class RecommendationRequest(BaseModel):
    user_id: str = Field(
        min_length=1,
        max_length=6,
        pattern=r"^\d{1,6}$",
        description="Numeric user ID from the processed dataset. Leading zeros are allowed.",
        examples=["00029"],
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=DEFAULT_CANDIDATE_POOL_SIZE,
        description="Number of recommendations to return.",
        examples=[10],
    )


class SampleRecommendationRequest(BaseModel):
    user_id: str = Field(
        min_length=1,
        max_length=64,
        description="Temporary sample user ID. This profile is not saved to processed_data.csv.",
        examples=["new_user_001"],
    )
    age: int = Field(
        ge=13,
        le=100,
        description="Sample user's age.",
        examples=[28],
    )
    gender: str = Field(
        min_length=1,
        max_length=50,
        description="Sample user's gender. Kept for profile completeness; not used by the current ranker.",
        examples=["Female"],
    )
    city: str = Field(
        min_length=1,
        max_length=100,
        description="Sample user's city.",
        examples=["Mumbai"],
    )
    country: str = Field(
        min_length=1,
        max_length=100,
        description="Sample user's country.",
        examples=["India"],
    )
    interests: list[str] = Field(
        min_length=1,
        max_length=20,
        description="Sample user's interests. Duplicate interests are removed before recommendation.",
        examples=[["Technology", "Travel", "Music"]],
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=DEFAULT_CANDIDATE_POOL_SIZE,
        description="Number of recommendations to return.",
        examples=[10],
    )

    @field_validator("user_id", "gender", "city", "country")
    @classmethod
    def strip_required_string(cls, value: str) -> str:
        cleaned_value = " ".join(value.strip().split())
        if not cleaned_value:
            raise ValueError("value cannot be empty")
        return cleaned_value

    @field_validator("interests")
    @classmethod
    def clean_interests(cls, values: list[str]) -> list[str]:
        cleaned_interests = []
        seen = set()

        for value in values:
            cleaned_value = " ".join(value.strip().split())
            if not cleaned_value:
                raise ValueError("interests cannot contain empty values")

            key = cleaned_value.casefold()
            if key not in seen:
                cleaned_interests.append(cleaned_value)
                seen.add(key)

        return cleaned_interests


class HealthResponse(BaseModel):
    status: str


class RecommendationResponseItem(BaseModel):
    user_id: str
    name: str
    city: str
    country: str
    interests: list[str]
    score: float = Field(description="Predicted neural ranker relevance score.")
    ann_similarity: float
    interest_jaccard_similarity: float
    same_country: bool
    same_city_country_key: bool
    age_difference: float


class RecommendationResponse(BaseModel):
    user_id: str
    top_k: int
    candidate_pool_size: int
    recommendations: list[RecommendationResponseItem]
    note: str
