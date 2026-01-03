"""vLLM service - handles LLM API calls to vLLM server."""
import httpx
import asyncio
from typing import List, Optional

from app.services.llm_interface import LLMServiceInterface
from app.services.embedding_service import EmbeddingService


class VLLMService(LLMServiceInterface):
    """Service for communicating with vLLM API (OpenAI compatible).
    
    Uses the OpenAI-compatible endpoints at http://localhost:4568/v1.
    Falls back to sentence-transformers for embeddings since vLLM
    has limited embedding support.
    """
    
    VLLM_BASE_URL = "http://localhost:4568"
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = base_url or f"{self.VLLM_BASE_URL}/v1"
        self.model = model
        self._client = httpx.AsyncClient(timeout=300.0)
        self._embedding_service: Optional[EmbeddingService] = None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def call(
        self, 
        instruction: str, 
        prompt: str,
        max_retries: int = 5,
    ) -> str:
        """
        Make a chat completion call to vLLM.
        
        Args:
            instruction: System instruction for the LLM
            prompt: User prompt/question
            max_retries: Maximum retry attempts
            
        Returns:
            The LLM response text
        """
        url = f"{self.base_url}/chat/completions"
        
        # vLLM OpenAI-compatible payload
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        
        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = await self._client.post(url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                    return ""
                elif response.status_code == 429:
                    # Rate limited, exponential backoff
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    error_text = response.text
                    print(f"vLLM Error: {response.status_code} - {error_text}")
                    raise Exception(f"Error calling vLLM API: {response.status_code}")
                    
            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise Exception("vLLM API timeout after max retries")
            except httpx.RequestError as e:
                print(f"vLLM Connection Error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise Exception(f"vLLM API request error: {e}")
        
        raise Exception("Error calling vLLM API: Too many retries")

    async def get_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 50
    ) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        vLLM has limited embedding support, so we first try the vLLM
        embeddings endpoint and fall back to sentence-transformers if it fails.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Try vLLM embeddings first
        try:
            return await self._get_vllm_embeddings(texts, batch_size)
        except Exception as e:
            print(f"vLLM embeddings failed ({e}), falling back to local embeddings")
            return await self._get_local_embeddings(texts)
    
    async def _get_vllm_embeddings(
        self, 
        texts: List[str], 
        batch_size: int
    ) -> List[List[float]]:
        """Try to get embeddings from vLLM server."""
        url = f"{self.base_url}/embeddings"
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            payload = {
                "model": self.model,
                "input": batch,
            }
            
            response = await self._client.post(url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                batch_embeddings = [
                    item["embedding"] 
                    for item in data.get("data", [])
                ]
                embeddings.extend(batch_embeddings)
            else:
                raise Exception(f"vLLM embeddings failed: {response.status_code}")
        
        return embeddings
    
    async def _get_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using local sentence-transformers model."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        
        return await self._embedding_service.get_embeddings_async(texts)

    async def health_check(self) -> bool:
        """Check if vLLM server is reachable."""
        try:
            response = await self._client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception:
            return False
