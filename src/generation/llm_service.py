from typing import Dict, List, Optional, AsyncIterator
import openai
from openai import OpenAI
import asyncio
import os
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    tokens_used: int
    model: str
    finish_reason: str


class LLMService:
    def __init__(self, provider: str = "openai", model: str = "gpt-3.5-turbo"):
        """Initialize LLM service with specified provider and model."""
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                # For testing purposes, allow initialization without API key
                if os.getenv("TESTING", "false").lower() == "true":
                    self.client = None
                else:
                    raise ValueError("OPENAI_API_KEY environment variable required")
            else:
                self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def generate_response(self, prompt: str, 
                              temperature: float = 0.7,
                              max_tokens: int = 1000) -> LLMResponse:
        """Generate response from LLM."""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return LLMResponse(
                    content=response.choices[0].message.content,
                    tokens_used=response.usage.total_tokens,
                    model=self.model,
                    finish_reason=response.choices[0].finish_reason
                )
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token for GPT)."""
        return len(text) // 4
    
    async def stream_response(self, prompt: str, 
                            temperature: float = 0.7) -> AsyncIterator[str]:
        """Stream response from LLM."""
        try:
            if self.provider == "openai":
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error: {str(e)}"
