from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class SearchStrategy(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, 
                         description="The question to ask")
    max_chunks: int = Field(5, ge=1, le=10, 
                           description="Maximum number of context chunks")
    temperature: float = Field(0.7, ge=0.0, le=1.0, 
                              description="LLM temperature")
    search_strategy: SearchStrategy = Field(SearchStrategy.HYBRID,
                                           description="Search strategy")
    include_sources: bool = Field(True, description="Include source citations")


class Source(BaseModel):
    title: str = Field(..., description="Source title")
    url: str = Field(..., description="Source URL")
    relevance_score: float = Field(..., description="Relevance score")
    excerpt: str = Field(..., description="Text excerpt")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(default=[], description="Source citations")
    confidence_score: float = Field(..., description="Answer confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    tokens_used: int = Field(..., description="LLM tokens used")
    search_results_count: int = Field(..., description="Number of search results")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    dependencies: dict = Field(..., description="Dependency status")


class FeedbackRequest(BaseModel):
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Provided answer")
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5")
    feedback: Optional[str] = Field(None, description="Additional feedback")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    timestamp: datetime = Field(..., description="Error timestamp")
