"""Request bodies. Pydantic rejects out-of-range quiz/feedback values automatically."""
from pydantic import BaseModel, Field


class Quiz(BaseModel):
    sleepiness: int = Field(ge=1, le=5)   # 5 = very sleepy
    stress: int = Field(ge=1, le=5)       # 5 = very stressed
    energy: int = Field(ge=1, le=5)       # 5 = very energetic
    difficulty: int = Field(ge=1, le=5)   # 5 = expect it to be very hard to fall asleep
    last_night: int = Field(ge=1, le=5)   # 5 = slept very well


class RecommendRequest(BaseModel):
    astronaut_id: str = Field(min_length=1, max_length=64)
    quiz: Quiz


class Settings(BaseModel):
    frequency_khz: int
    amplitude_percent: int
    timer_minutes: int
    waveform: str


class Feedback(BaseModel):
    latency_min: int = Field(ge=0, le=180)  # minutes to fall asleep
    quality: int = Field(ge=1, le=5)
    awakenings: int = Field(ge=0, le=20)
    refreshed: int = Field(ge=1, le=5)
    comfort: int = Field(ge=1, le=5)
