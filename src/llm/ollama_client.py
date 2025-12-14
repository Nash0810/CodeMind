"""
Ollama client for interacting with locally-hosted LLMs.

Provides streaming and non-streaming generation with timeout and error handling.
"""

import requests
import json
from typing import Iterator, Optional, Dict, List, Any
from dataclasses import dataclass
import time


@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""
    model: str = "neural-chat"  # Default lightweight model
    temperature: float = 0.3  # Lower for more deterministic answers
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 2000
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64


class OllamaClient:
    """
    Client for Ollama API.
    
    Ollama runs open-source LLMs locally. This client provides:
    - Streaming and non-streaming generation
    - Automatic retry with exponential backoff
    - Timeout handling
    - Token counting
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: URL of Ollama server (default: localhost:11434)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failures
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_error: Optional[str] = None
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            data = response.json()
            return [model['name'].split(':')[0] for model in data.get('models', [])]
        except Exception as e:
            self._last_error = str(e)
            return []
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            stream: If True, return generator for streaming response
            
        Returns:
            Generated text (or generator if stream=True)
        """
        if config is None:
            config = GenerationConfig()
        
        payload = {
            "model": config.model,
            "prompt": prompt,
            "stream": stream,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "num_predict": config.max_tokens,
            "repeat_penalty": config.repeat_penalty,
            "repeat_last_n": config.repeat_last_n,
        }
        
        if stream:
            return self._stream_generate(payload)
        else:
            return self._blocking_generate(payload)
    
    def _blocking_generate(self, payload: Dict[str, Any]) -> str:
        """Non-streaming generation with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get('response', '')
            
            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    self._last_error = "Request timeout"
                    return ""
                time.sleep(2 ** attempt)  # Exponential backoff
            
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    self._last_error = f"Request failed: {str(e)}"
                    return ""
                time.sleep(2 ** attempt)
        
        return ""
    
    def _stream_generate(self, payload: Dict[str, Any]) -> Iterator[str]:
        """Streaming generation."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        yield chunk.get('response', '')
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            self._last_error = str(e)
            yield f"Error: {str(e)}"
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Estimate token count.
        
        Simple heuristic: ~4 chars per token (accurate for English)
        """
        return len(text) // 4
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        stream: bool = False
    ) -> str:
        """
        Chat-style generation (if model supports it).
        
        Args:
            messages: List of messages with 'role' and 'content'
            config: Generation configuration
            stream: If True, return generator for streaming
            
        Returns:
            Generated response
        """
        if config is None:
            config = GenerationConfig()
        
        payload = {
            "model": config.model,
            "messages": messages,
            "stream": stream,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        
        try:
            if stream:
                return self._stream_chat(payload)
            else:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                return result.get('message', {}).get('content', '')
        
        except Exception as e:
            self._last_error = str(e)
            return f"Error: {str(e)}"
    
    def _stream_chat(self, payload: Dict[str, Any]) -> Iterator[str]:
        """Streaming chat generation."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        content = chunk.get('message', {}).get('content', '')
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            self._last_error = str(e)
            yield f"Error: {str(e)}"
    
    def get_last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error
