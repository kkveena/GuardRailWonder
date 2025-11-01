"""
Embedding providers for generating text embeddings
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional
import logging

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np

from .models import EmbeddingResponse

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""

    @abstractmethod
    def embed(self, text: str) -> EmbeddingResponse:
        """Generate embedding for given text"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResponse]:
        """Generate embeddings for multiple texts"""
        pass


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Google's Gemini API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "models/embedding-001",
        task_type: str = "retrieval_document"
    ):
        """
        Initialize Gemini embedding provider

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Embedding model to use
            task_type: Task type for embeddings (retrieval_document, retrieval_query, etc.)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set in environment")

        self.model = model
        self.task_type = task_type

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        logger.info(f"Initialized GeminiEmbeddingProvider with model: {model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def embed(self, text: str) -> EmbeddingResponse:
        """
        Generate embedding for a single text

        Args:
            text: Input text to embed

        Returns:
            EmbeddingResponse containing the embedding vector
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type=self.task_type
            )

            embedding = result['embedding']

            return EmbeddingResponse(
                embedding=embedding,
                model=self.model,
                dimension=len(embedding)
            )
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResponse]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts to embed

        Returns:
            List of EmbeddingResponse objects
        """
        try:
            # Gemini API supports batch embedding
            results = []
            for text in texts:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type=self.task_type
                )

                embedding = result['embedding']
                results.append(
                    EmbeddingResponse(
                        embedding=embedding,
                        model=self.model,
                        dimension=len(embedding)
                    )
                )

            return results
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    def compute_cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)

        # Compute cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)

        return float(similarity)

    def compute_similarity_matrix(
        self,
        query_embedding: List[float],
        reference_embeddings: List[List[float]]
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and multiple reference embeddings

        This implements your 1×n @ (7×n)^T = 1×7 approach

        Args:
            query_embedding: Query embedding vector (1×n)
            reference_embeddings: List of reference embeddings (7×n)

        Returns:
            Similarity scores array (1×7)
        """
        query_vec = np.array(query_embedding).reshape(1, -1)
        ref_matrix = np.array(reference_embeddings)

        # Normalize vectors
        query_norm = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        ref_norm = ref_matrix / np.linalg.norm(ref_matrix, axis=1, keepdims=True)

        # Matrix multiplication: (1×n) @ (n×7) = (1×7)
        similarity_scores = np.dot(query_norm, ref_norm.T)

        return similarity_scores.flatten()
