"""
Tests for centroid-based guardrail system
"""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from src.guardrails.centroid_guardrail import CentroidGuardrail


# Sample intent configuration for testing
SAMPLE_INTENT_CONFIG = {
    "intent_paraphrases": {
        "check_balance": [
            "What's my account balance?",
            "show balance",
            "current balance"
        ],
        "make_payment": [
            "I want to pay my bill",
            "make a payment",
            "pay now"
        ],
        "contact_agent": [
            "talk to a human",
            "contact support"
        ]
    },
    "intent_thresholds": {
        "check_balance": 0.72,
        "make_payment": 0.72,
        "contact_agent": 0.65
    },
    "intent_metadata": {
        "check_balance": {
            "description": "Check balance",
            "category": "account",
            "route": "balance_service"
        },
        "make_payment": {
            "description": "Make payment",
            "category": "billing",
            "route": "payment_service"
        },
        "contact_agent": {
            "description": "Contact agent",
            "category": "support",
            "route": "agent_service"
        }
    }
}


@pytest.fixture
def temp_config_file():
    """Create temporary configuration file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(SAMPLE_INTENT_CONFIG, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def mock_embedding_response():
    """Mock embedding response from Gemini"""
    def _mock_embed(model, content):
        # Generate deterministic embedding based on text hash
        text_hash = hash(content)
        np.random.seed(abs(text_hash) % (2**32))
        embedding = np.random.randn(768).tolist()
        return {"embedding": embedding}

    return _mock_embed


@pytest.fixture
def mock_llm_response():
    """Mock LLM response from Gemini"""
    def _mock_generate(prompt):
        # Mock response - return "Yes" for verification
        mock_resp = MagicMock()
        mock_resp.text = "Yes"
        return mock_resp

    return _mock_generate


class TestCentroidGuardrail:
    """Test cases for CentroidGuardrail"""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.embed_content')
    def test_initialization(self, mock_embed, mock_configure, temp_config_file, mock_embedding_response):
        """Test guardrail initialization"""
        mock_embed.side_effect = mock_embedding_response

        # Set API key in environment
        os.environ['GOOGLE_API_KEY'] = 'test_key'

        guardrail = CentroidGuardrail(
            intent_paraphrases_path=temp_config_file,
            cache_dir=tempfile.mkdtemp()
        )

        # Check that configuration was loaded
        assert len(guardrail.intent_paraphrases) == 3
        assert "check_balance" in guardrail.intent_paraphrases
        assert "make_payment" in guardrail.intent_paraphrases
        assert "contact_agent" in guardrail.intent_paraphrases

        # Check thresholds
        assert guardrail.intent_thresholds["check_balance"] == 0.72
        assert guardrail.intent_thresholds["make_payment"] == 0.72
        assert guardrail.intent_thresholds["contact_agent"] == 0.65

        # Check centroids were built
        assert len(guardrail.centroids) == 3
        for intent in guardrail.intent_paraphrases.keys():
            assert intent in guardrail.centroids
            assert guardrail.centroids[intent].shape == (768,)

    @patch('google.generativeai.configure')
    @patch('google.generativeai.embed_content')
    def test_embed(self, mock_embed, mock_configure, temp_config_file, mock_embedding_response):
        """Test embedding generation"""
        mock_embed.side_effect = mock_embedding_response
        os.environ['GOOGLE_API_KEY'] = 'test_key'

        guardrail = CentroidGuardrail(
            intent_paraphrases_path=temp_config_file,
            cache_dir=tempfile.mkdtemp()
        )

        # Test normal embedding
        text = "test text"
        embedding = guardrail.embed(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        # Check L2 normalization
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)

        # Test empty text
        empty_embedding = guardrail.embed("")
        assert np.allclose(empty_embedding, np.zeros(768))

    @patch('google.generativeai.configure')
    @patch('google.generativeai.embed_content')
    def test_score_intents(self, mock_embed, mock_configure, temp_config_file, mock_embedding_response):
        """Test intent scoring"""
        mock_embed.side_effect = mock_embedding_response
        os.environ['GOOGLE_API_KEY'] = 'test_key'

        guardrail = CentroidGuardrail(
            intent_paraphrases_path=temp_config_file,
            cache_dir=tempfile.mkdtemp()
        )

        # Score a query similar to check_balance
        user_text = "show my balance"
        best_intent, best_score, margin, all_scores = guardrail.score_intents(user_text)

        assert best_intent in guardrail.intent_paraphrases.keys()
        assert 0.0 <= best_score <= 1.0
        assert margin >= 0.0
        assert len(all_scores) == 3
        assert all(0.0 <= score <= 1.0 for score in all_scores.values())

    @patch('google.generativeai.configure')
    @patch('google.generativeai.embed_content')
    def test_decide_approved(self, mock_embed, mock_configure, temp_config_file, mock_embedding_response):
        """Test decision for high-confidence match"""
        mock_embed.side_effect = mock_embedding_response
        os.environ['GOOGLE_API_KEY'] = 'test_key'

        # Create guardrail with low threshold for testing
        guardrail = CentroidGuardrail(
            intent_paraphrases_path=temp_config_file,
            cache_dir=tempfile.mkdtemp()
        )
        # Lower thresholds for testing
        guardrail.intent_thresholds = {intent: 0.1 for intent in guardrail.intent_paraphrases}

        result = guardrail.decide("What's my balance?")

        assert result["allowed"] is True
        assert "intent" in result
        assert "route" in result
        assert result["reason"] == "pass_threshold"

    @patch('google.generativeai.configure')
    @patch('google.generativeai.embed_content')
    def test_decide_rejected(self, mock_embed, mock_configure, temp_config_file, mock_embedding_response):
        """Test decision for out-of-scope query"""
        mock_embed.side_effect = mock_embedding_response
        os.environ['GOOGLE_API_KEY'] = 'test_key'

        guardrail = CentroidGuardrail(
            intent_paraphrases_path=temp_config_file,
            cache_dir=tempfile.mkdtemp()
        )

        # Set very high thresholds
        guardrail.intent_thresholds = {intent: 0.99 for intent in guardrail.intent_paraphrases}

        result = guardrail.decide("Tell me a joke")

        assert result["allowed"] is False
        assert result["reason"] == "out_of_scope_or_ambiguous"

    @patch('google.generativeai.configure')
    @patch('google.generativeai.embed_content')
    @patch('google.generativeai.GenerativeModel')
    def test_llm_verify(
        self,
        mock_model_class,
        mock_embed,
        mock_configure,
        temp_config_file,
        mock_embedding_response,
        mock_llm_response
    ):
        """Test LLM verification in gray band"""
        mock_embed.side_effect = mock_embedding_response
        os.environ['GOOGLE_API_KEY'] = 'test_key'

        # Mock LLM
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = lambda prompt: mock_llm_response(prompt)
        mock_model_class.return_value = mock_model

        guardrail = CentroidGuardrail(
            intent_paraphrases_path=temp_config_file,
            cache_dir=tempfile.mkdtemp()
        )

        # Test verification
        result = guardrail.llm_verify("show balance", "check_balance")
        assert result is True

        # Test negative verification
        mock_model.generate_content.return_value.text = "No"
        result = guardrail.llm_verify("tell a joke", "check_balance")
        assert result is False

    @patch('google.generativeai.configure')
    @patch('google.generativeai.embed_content')
    def test_add_paraphrase(self, mock_embed, mock_configure, temp_config_file, mock_embedding_response):
        """Test adding new paraphrase"""
        mock_embed.side_effect = mock_embedding_response
        os.environ['GOOGLE_API_KEY'] = 'test_key'

        guardrail = CentroidGuardrail(
            intent_paraphrases_path=temp_config_file,
            cache_dir=tempfile.mkdtemp()
        )

        original_count = len(guardrail.intent_paraphrases["check_balance"])

        # Add new paraphrase
        guardrail.add_paraphrase("check_balance", "show my account balance", rebuild=True)

        assert len(guardrail.intent_paraphrases["check_balance"]) == original_count + 1
        assert "show my account balance" in guardrail.intent_paraphrases["check_balance"]

    @patch('google.generativeai.configure')
    @patch('google.generativeai.embed_content')
    def test_save_config(self, mock_embed, mock_configure, temp_config_file, mock_embedding_response):
        """Test saving configuration"""
        mock_embed.side_effect = mock_embedding_response
        os.environ['GOOGLE_API_KEY'] = 'test_key'

        guardrail = CentroidGuardrail(
            intent_paraphrases_path=temp_config_file,
            cache_dir=tempfile.mkdtemp()
        )

        # Add a paraphrase
        guardrail.add_paraphrase("check_balance", "new phrase", rebuild=False)

        # Save config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        guardrail.save_config(output_path)

        # Load and verify
        with open(output_path, 'r') as f:
            saved_config = json.load(f)

        assert "new phrase" in saved_config["intent_paraphrases"]["check_balance"]

        # Cleanup
        Path(output_path).unlink()

    @patch('google.generativeai.configure')
    @patch('google.generativeai.embed_content')
    def test_get_intent_info(self, mock_embed, mock_configure, temp_config_file, mock_embedding_response):
        """Test getting intent information"""
        mock_embed.side_effect = mock_embedding_response
        os.environ['GOOGLE_API_KEY'] = 'test_key'

        guardrail = CentroidGuardrail(
            intent_paraphrases_path=temp_config_file,
            cache_dir=tempfile.mkdtemp()
        )

        info = guardrail.get_intent_info()

        assert "intents" in info
        assert "total_intents" in info
        assert "paraphrase_counts" in info
        assert "thresholds" in info
        assert info["total_intents"] == 3
        assert len(info["intents"]) == 3

    @patch('google.generativeai.configure')
    @patch('google.generativeai.embed_content')
    def test_empty_input(self, mock_embed, mock_configure, temp_config_file, mock_embedding_response):
        """Test handling of empty input"""
        mock_embed.side_effect = mock_embedding_response
        os.environ['GOOGLE_API_KEY'] = 'test_key'

        guardrail = CentroidGuardrail(
            intent_paraphrases_path=temp_config_file,
            cache_dir=tempfile.mkdtemp()
        )

        result = guardrail.decide("")

        assert result["allowed"] is False
        assert result["reason"] == "empty_input"
