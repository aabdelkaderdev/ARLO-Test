"""Abstract LLM service interface for ARLO."""
from abc import ABC, abstractmethod
from typing import List


class LLMServiceInterface(ABC):
    """Abstract base class defining the common LLM service interface.
    
    Both OllamaService and VLLMService implement this interface,
    allowing them to be used interchangeably.
    """
    
    @abstractmethod
    async def call(
        self, 
        instruction: str, 
        prompt: str,
        max_retries: int = 5,
    ) -> str:
        """
        Make a chat completion call to the LLM.
        
        Args:
            instruction: System instruction for the LLM
            prompt: User prompt/question
            max_retries: Maximum retry attempts
            
        Returns:
            The LLM response text
        """
        pass

    @abstractmethod
    async def get_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 50
    ) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            
        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM server is reachable."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        pass
