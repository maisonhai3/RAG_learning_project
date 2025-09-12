from typing import Dict, List, Optional, AsyncIterator
import asyncio
import os
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import AsyncCallbackHandler


@dataclass
class LLMResponse:
    content: str
    tokens_used: int
    model: str
    finish_reason: str


class StreamingCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        self.tokens = []
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)


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
                    self.llm = None
                else:
                    raise ValueError("OPENAI_API_KEY environment variable required")
            else:
                self.llm = ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    temperature=0.7
                )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def generate_response(self, prompt: str, 
                              temperature: float = 0.7,
                              max_tokens: int = 1000) -> LLMResponse:
        """Generate response from LLM."""
        try:
            if self.provider == "openai":
                if self.llm is None:
                    raise ValueError("LLM not initialized (missing API key)")
                
                # Update temperature for this request
                self.llm.temperature = temperature
                self.llm.max_tokens = max_tokens
                
                # Create message and get response
                message = HumanMessage(content=prompt)
                response = await self.llm.ainvoke([message])
                
                # Estimate tokens (LangChain doesn't always provide usage info)
                estimated_tokens = self.estimate_tokens(prompt + response.content)
                
                return LLMResponse(
                    content=response.content,
                    tokens_used=estimated_tokens,
                    model=self.model,
                    finish_reason="stop"  # Default finish reason
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
                if self.llm is None:
                    yield "Error: LLM not initialized (missing API key)"
                    return
                
                # Update temperature for this request
                self.llm.temperature = temperature
                
                # Create message and stream response
                message = HumanMessage(content=prompt)
                async for chunk in self.llm.astream([message]):
                    if chunk.content:
                        yield chunk.content
        except Exception as e:
            yield f"Error: {str(e)}"
