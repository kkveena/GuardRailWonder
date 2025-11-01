"""
Tests for core guardrail system
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from guardrails.core import PromptGuardrail
from guardrails.models import (
    GuardrailConfig,
    GuardrailDecision,
    PredefinedPrompt,
    EmbeddingResponse,
)
from guardrails.embeddings import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing"""

    def __init__(self):
        self.embeddings_map = {
            "What is the status of my order?": [1.0, 0.0, 0.0],
            "Tell me about your product features": [0.0, 1.0, 0.0],
            "I want a refund": [0.0, 0.0, 1.0],
            # Test prompts
            "Where is my order?": [0.95, 0.05, 0.0],  # High similarity to order status
            "Can you explain your products?": [0.05, 0.9, 0.05],  # High similarity to product info
            "What's the weather?": [0.1, 0.1, 0.1],  # Low similarity to all
            "I need help with my account": [0.5, 0.4, 0.3],  # Medium similarity
        }

    def embed(self, text: str) -> EmbeddingResponse:
        embedding = self.embeddings_map.get(text, [0.0, 0.0, 0.0])
        return EmbeddingResponse(
            embedding=embedding,
            model="mock-model",
            dimension=len(embedding)
        )

    def embed_batch(self, texts: list) -> list:
        return [self.embed(text) for text in texts]

    def compute_similarity_matrix(self, query, references):
        import numpy as np
        query_vec = np.array(query).reshape(1, -1)
        ref_matrix = np.array(references)

        query_norm = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        ref_norm = ref_matrix / np.linalg.norm(ref_matrix, axis=1, keepdims=True)

        return np.dot(query_norm, ref_norm.T).flatten()


@pytest.fixture
def test_prompts_file():
    """Create a temporary predefined prompts file"""
    prompts_data = {
        "prompts": [
            {
                "id": "order_status",
                "template": "What is the status of my order?",
                "category": "orders",
                "description": "Order status inquiry"
            },
            {
                "id": "product_info",
                "template": "Tell me about your product features",
                "category": "products",
                "description": "Product information"
            },
            {
                "id": "refund",
                "template": "I want a refund",
                "category": "refunds",
                "description": "Refund request"
            }
        ],
        "metadata": {
            "version": "1.0",
            "total_prompts": 3
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(prompts_data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def guardrail_config():
    """Create test guardrail configuration"""
    return GuardrailConfig(
        threshold_high=0.8,
        threshold_medium=0.5,
        log_rejections=False,
        embedding_model="mock-model",
        cache_embeddings=False,
    )


class TestPromptGuardrail:
    """Tests for PromptGuardrail"""

    def test_initialization(self, guardrail_config, test_prompts_file):
        """Test guardrail initialization"""
        mock_provider = MockEmbeddingProvider()
        guardrail = PromptGuardrail(
            config=guardrail_config,
            embedding_provider=mock_provider,
            predefined_prompts_path=test_prompts_file
        )

        assert len(guardrail.predefined_prompts) == 3
        assert guardrail.config.threshold_high == 0.8

    def test_load_predefined_prompts(self, guardrail_config, test_prompts_file):
        """Test loading predefined prompts"""
        mock_provider = MockEmbeddingProvider()
        guardrail = PromptGuardrail(
            config=guardrail_config,
            embedding_provider=mock_provider,
            predefined_prompts_path=test_prompts_file
        )

        prompts = guardrail.predefined_prompts
        assert len(prompts) == 3
        assert all(isinstance(p, PredefinedPrompt) for p in prompts)
        assert prompts[0].id == "order_status"

    def test_initialize_predefined_embeddings(self, guardrail_config, test_prompts_file):
        """Test pre-computing embeddings"""
        mock_provider = MockEmbeddingProvider()
        guardrail = PromptGuardrail(
            config=guardrail_config,
            embedding_provider=mock_provider,
            predefined_prompts_path=test_prompts_file
        )

        guardrail.initialize_predefined_embeddings()

        for prompt in guardrail.predefined_prompts:
            assert prompt.embedding is not None
            assert len(prompt.embedding) > 0

    def test_evaluate_high_similarity_approved(self, guardrail_config, test_prompts_file):
        """Test that high similarity prompts are approved"""
        mock_provider = MockEmbeddingProvider()
        guardrail = PromptGuardrail(
            config=guardrail_config,
            embedding_provider=mock_provider,
            predefined_prompts_path=test_prompts_file
        )
        guardrail.initialize_predefined_embeddings()

        result = guardrail.evaluate("Where is my order?")

        assert result.decision == GuardrailDecision.APPROVED
        assert result.similarity_score >= guardrail_config.threshold_high
        assert result.matched_prompt_id is not None

    def test_evaluate_medium_similarity_warning(self, guardrail_config, test_prompts_file):
        """Test that medium similarity prompts get warning"""
        mock_provider = MockEmbeddingProvider()
        guardrail = PromptGuardrail(
            config=guardrail_config,
            embedding_provider=mock_provider,
            predefined_prompts_path=test_prompts_file
        )
        guardrail.initialize_predefined_embeddings()

        result = guardrail.evaluate("I need help with my account")

        assert result.decision == GuardrailDecision.APPROVED_WITH_WARNING
        assert guardrail_config.threshold_medium <= result.similarity_score < guardrail_config.threshold_high

    def test_evaluate_low_similarity_rejected(self, guardrail_config, test_prompts_file):
        """Test that low similarity prompts are rejected"""
        mock_provider = MockEmbeddingProvider()
        guardrail = PromptGuardrail(
            config=guardrail_config,
            embedding_provider=mock_provider,
            predefined_prompts_path=test_prompts_file
        )
        guardrail.initialize_predefined_embeddings()

        result = guardrail.evaluate("What's the weather?")

        assert result.decision == GuardrailDecision.REJECTED
        assert result.similarity_score < guardrail_config.threshold_medium

    def test_get_supported_categories(self, guardrail_config, test_prompts_file):
        """Test getting supported categories"""
        mock_provider = MockEmbeddingProvider()
        guardrail = PromptGuardrail(
            config=guardrail_config,
            embedding_provider=mock_provider,
            predefined_prompts_path=test_prompts_file
        )

        categories = guardrail.get_supported_categories()

        assert len(categories) == 3
        assert "orders" in categories
        assert "products" in categories
        assert "refunds" in categories

    def test_get_predefined_templates(self, guardrail_config, test_prompts_file):
        """Test getting predefined templates"""
        mock_provider = MockEmbeddingProvider()
        guardrail = PromptGuardrail(
            config=guardrail_config,
            embedding_provider=mock_provider,
            predefined_prompts_path=test_prompts_file
        )

        templates = guardrail.get_predefined_templates()

        assert len(templates) == 3
        assert all('id' in t for t in templates)
        assert all('template' in t for t in templates)
        assert all('category' in t for t in templates)
