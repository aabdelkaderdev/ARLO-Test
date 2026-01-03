"""Ollama service - handles LLM API calls."""
import httpx
import asyncio
import os
from typing import List, Optional

from app.services.llm_interface import LLMServiceInterface


class OllamaService(LLMServiceInterface):
    """Service for communicating with Ollama API."""
    
    MAX_WORD_PER_CALL = 5000  # Token limit consideration
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self.embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL", self.model)
        self._client = httpx.AsyncClient(timeout=300.0)

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def call(
        self, 
        instruction: str, 
        prompt: str,
        max_retries: int = 5,
    ) -> str:
        """
        Make a chat completion call to Ollama.
        
        Args:
            instruction: System instruction for the LLM
            prompt: User prompt/question
            max_retries: Maximum retry attempts
            
        Returns:
            The LLM response text
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        
        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = await self._client.post(url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("message", {}).get("content", "")
                elif response.status_code == 429:
                    # Rate limited, exponential backoff
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    error_text = response.text
                    print(f"Ollama Error: {response.status_code} - {error_text}")
                    raise Exception(f"Error calling Ollama API: {response.status_code}")
                    
            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise Exception("Ollama API timeout after max retries")
            except httpx.RequestError as e:
                print(f"Ollama Connection Error Details: {type(e).__name__}: {e}")
                print(f"Failed URL: {e.request.url}")
                raise Exception(f"Ollama API request error: {e}")
        
        raise Exception("Error calling Ollama API: Too many retries")

    async def get_embeddings(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            
        Returns:
            List of embedding vectors
        """
        url = f"{self.base_url}/api/embeddings"
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for text in batch:
                payload = {
                    "model": self.embed_model,
                    "prompt": text,
                }
                
                try:
                    response = await self._client.post(url, json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        embedding = data.get("embedding", [])
                        embeddings.append(embedding)
                    else:
                        print(f"Embedding error: {response.status_code}")
                        embeddings.append([])
                        
                except Exception as e:
                    print(f"Embedding error: {e}")
                    embeddings.append([])
        
        return embeddings

    async def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
