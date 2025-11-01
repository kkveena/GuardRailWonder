"""
Tests for embedding providers
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from guardrails.embeddings import GeminiEmbeddingProvider, EmbeddingProvider
from guardrails.models import EmbeddingResponse


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing"""

    def embed(self, text: str) -> EmbeddingResponse:
        # Return a simple mock embedding based on text length
        embedding = [float(i) for i in range(768)]
        return EmbeddingResponse(
            embedding=embedding,
            model="mock-model",
            dimension=768
        )

    def embed_batch(self, texts: list) -> list:
        return [self.embed(text) for text in texts]


class TestGeminiEmbeddingProvider:
    """Tests for GeminiEmbeddingProvider"""

    def test_initialization_with_api_key(self):
        """Test provider initialization with API key"""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            provider = GeminiEmbeddingProvider(api_key='test-key')
            assert provider.api_key == 'test-key'
            assert provider.model == 'models/embedding-001'

    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization fails without API key"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                GeminiEmbeddingProvider()

    @patch('google.generativeai.embed_content')
    def test_embed_single_text(self, mock_embed):
        """Test embedding a single text"""
        mock_embed.return_value = {'embedding': [0.1] * 768}

        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            provider = GeminiEmbeddingProvider(api_key='test-key')
            result = provider.embed("Test text")

            assert isinstance(result, EmbeddingResponse)
            assert len(result.embedding) == 768
            assert result.dimension == 768
            mock_embed.assert_called_once()

    @patch('google.generativeai.embed_content')
    def test_embed_batch(self, mock_embed):
        """Test embedding multiple texts"""
        mock_embed.return_value = {'embedding': [0.1] * 768}

        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            provider = GeminiEmbeddingProvider(api_key='test-key')
            texts = ["Text 1", "Text 2", "Text 3"]
            results = provider.embed_batch(texts)

            assert len(results) == 3
            assert all(isinstance(r, EmbeddingResponse) for r in results)
            assert mock_embed.call_count == 3

    def test_cosine_similarity(self):
        """Test cosine similarity computation"""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            provider = GeminiEmbeddingProvider(api_key='test-key')

            # Identical vectors should have similarity = 1
            vec1 = [1.0, 0.0, 0.0]
            vec2 = [1.0, 0.0, 0.0]
            similarity = provider.compute_cosine_similarity(vec1, vec2)
            assert abs(similarity - 1.0) < 1e-6

            # Orthogonal vectors should have similarity ≈ 0
            vec3 = [1.0, 0.0, 0.0]
            vec4 = [0.0, 1.0, 0.0]
            similarity = provider.compute_cosine_similarity(vec3, vec4)
            assert abs(similarity) < 1e-6

            # Opposite vectors should have similarity = -1
            vec5 = [1.0, 0.0, 0.0]
            vec6 = [-1.0, 0.0, 0.0]
            similarity = provider.compute_cosine_similarity(vec5, vec6)
            assert abs(similarity - (-1.0)) < 1e-6

    def test_similarity_matrix(self):
        """Test similarity matrix computation"""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            provider = GeminiEmbeddingProvider(api_key='test-key')

            query = [1.0, 0.0, 0.0]
            references = [
                [1.0, 0.0, 0.0],  # Same as query
                [0.0, 1.0, 0.0],  # Orthogonal
                [-1.0, 0.0, 0.0],  # Opposite
            ]

            scores = provider.compute_similarity_matrix(query, references)

            assert len(scores) == 3
            assert abs(scores[0] - 1.0) < 1e-6  # Same
            assert abs(scores[1]) < 1e-6  # Orthogonal
            assert abs(scores[2] - (-1.0)) < 1e-6  # Opposite
