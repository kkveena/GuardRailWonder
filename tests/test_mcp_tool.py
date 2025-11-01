"""
Tests for MCP tool wrapper
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from guardrails.mcp_tool import GuardrailMCPTool
from guardrails.models import GuardrailDecision


@pytest.fixture
def test_config_file():
    """Create a temporary config file"""
    config_data = {
        "threshold_high": 0.8,
        "threshold_medium": 0.5,
        "log_rejections": False,
        "embedding_model": "mock-model",
        "cache_embeddings": False,
    }

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(config_data, f)
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink()


@pytest.fixture
def test_prompts_file():
    """Create a temporary prompts file"""
    prompts_data = {
        "prompts": [
            {
                "id": "test_prompt",
                "template": "Test prompt",
                "category": "test",
                "description": "Test description"
            }
        ],
        "metadata": {"version": "1.0", "total_prompts": 1}
    }

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(prompts_data, f)
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink()


class TestGuardrailMCPTool:
    """Tests for GuardrailMCPTool"""

    @patch('guardrails.mcp_tool.GeminiEmbeddingProvider')
    def test_get_tool_metadata(self, mock_provider, test_config_file, test_prompts_file):
        """Test getting tool metadata"""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            tool = GuardrailMCPTool(
                config_path=test_config_file,
                predefined_prompts_path=test_prompts_file
            )

            metadata = tool.get_tool_metadata()

            assert metadata['name'] == 'guardrail_wonder'
            assert metadata['version'] == '0.1.0'
            assert len(metadata['methods']) == 3
            assert 'config' in metadata

    @patch('guardrails.mcp_tool.GeminiEmbeddingProvider')
    def test_get_supported_categories(self, mock_provider, test_config_file, test_prompts_file):
        """Test getting supported categories"""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            tool = GuardrailMCPTool(
                config_path=test_config_file,
                predefined_prompts_path=test_prompts_file
            )

            result = tool.get_supported_categories()

            assert 'categories' in result
            assert 'templates' in result
            assert 'total' in result
            assert result['total'] > 0
