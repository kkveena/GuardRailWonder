"""
MCP (Model Context Protocol) Tool wrapper for guardrail system

This allows the guardrail to be used as an MCP tool that can be called
by LLMs or other systems.
"""

import json
import logging
import os
from typing import Dict, Any, Optional

from .core import PromptGuardrail
from .models import GuardrailConfig, GuardrailDecision
from .embeddings import GeminiEmbeddingProvider

logger = logging.getLogger(__name__)


class GuardrailMCPTool:
    """
    MCP Tool wrapper for the guardrail system

    This can be exposed as an MCP tool that validates prompts before
    they are sent to the main LLM.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        predefined_prompts_path: Optional[str] = None,
    ):
        """
        Initialize the MCP tool

        Args:
            config_path: Path to configuration file
            predefined_prompts_path: Path to predefined prompts JSON
        """
        self.config = self._load_config(config_path)
        self.guardrail = PromptGuardrail(
            config=self.config,
            embedding_provider=GeminiEmbeddingProvider(
                model=self.config.embedding_model
            ),
            predefined_prompts_path=predefined_prompts_path,
        )

        # Initialize embeddings
        self.guardrail.initialize_predefined_embeddings()

        logger.info("GuardrailMCPTool initialized")

    def _load_config(self, config_path: Optional[str] = None) -> GuardrailConfig:
        """Load configuration from file or environment"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return GuardrailConfig(**config_data)

        # Load from environment variables
        return GuardrailConfig(
            threshold_high=float(os.getenv("GUARDRAIL_THRESHOLD_HIGH", "0.8")),
            threshold_medium=float(os.getenv("GUARDRAIL_THRESHOLD_MEDIUM", "0.5")),
            log_rejections=os.getenv("GUARDRAIL_LOG_REJECTIONS", "true").lower() == "true",
            log_file=os.getenv("GUARDRAIL_LOG_FILE", "logs/guardrail.log"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "models/embedding-001"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
        )

    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        MCP tool method: Validate a prompt

        This is the main method that would be exposed as an MCP tool.

        Args:
            prompt: The user's prompt to validate

        Returns:
            Dict containing validation result
        """
        result = self.guardrail.evaluate(prompt)

        return {
            "approved": result.decision != GuardrailDecision.REJECTED,
            "decision": result.decision,
            "similarity_score": result.similarity_score,
            "matched_category": result.matched_prompt_category,
            "message": result.message,
            "prompt": prompt,
        }

    def get_supported_categories(self) -> Dict[str, Any]:
        """
        MCP tool method: Get supported prompt categories

        Returns:
            Dict containing supported categories
        """
        categories = self.guardrail.get_supported_categories()
        templates = self.guardrail.get_predefined_templates()

        return {
            "categories": categories,
            "templates": templates,
            "total": len(templates),
        }

    def explain_rejection(self, prompt: str) -> Dict[str, Any]:
        """
        MCP tool method: Explain why a prompt was rejected

        Args:
            prompt: The rejected prompt

        Returns:
            Dict with detailed explanation and suggestions
        """
        result = self.guardrail.evaluate(prompt)

        if result.decision == GuardrailDecision.REJECTED:
            # Get top 3 closest matches
            scores = sorted(
                result.all_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            templates = {
                t["id"]: t["template"]
                for t in self.guardrail.get_predefined_templates()
            }

            suggestions = [
                {
                    "prompt_id": pid,
                    "similarity": score,
                    "template": templates.get(pid, ""),
                }
                for pid, score in scores
            ]

            return {
                "rejected": True,
                "reason": result.message,
                "similarity_score": result.similarity_score,
                "threshold_required": self.config.threshold_medium,
                "suggestions": suggestions,
                "help_text": (
                    "Your prompt doesn't match our supported query types. "
                    "Here are some similar examples you can try:"
                ),
            }
        else:
            return {
                "rejected": False,
                "message": "Prompt was approved",
            }

    def get_tool_metadata(self) -> Dict[str, Any]:
        """
        MCP tool method: Get tool metadata

        Returns:
            Dict containing tool metadata and configuration
        """
        return {
            "name": "guardrail_wonder",
            "version": "0.1.0",
            "description": "LLM Prompt Guardrail using embedding-based similarity matching",
            "methods": [
                {
                    "name": "validate_prompt",
                    "description": "Validate if a prompt matches supported categories",
                    "parameters": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to validate",
                            "required": True,
                        }
                    },
                },
                {
                    "name": "get_supported_categories",
                    "description": "Get list of supported prompt categories",
                    "parameters": {},
                },
                {
                    "name": "explain_rejection",
                    "description": "Get detailed explanation for rejected prompt",
                    "parameters": {
                        "prompt": {
                            "type": "string",
                            "description": "The rejected prompt",
                            "required": True,
                        }
                    },
                },
            ],
            "config": {
                "threshold_high": self.config.threshold_high,
                "threshold_medium": self.config.threshold_medium,
                "embedding_model": self.config.embedding_model,
            },
        }


# MCP Tool Interface Functions
# These can be directly exposed as MCP tools

_tool_instance: Optional[GuardrailMCPTool] = None


def initialize_tool(
    config_path: Optional[str] = None,
    predefined_prompts_path: Optional[str] = None,
) -> None:
    """Initialize the global MCP tool instance"""
    global _tool_instance
    _tool_instance = GuardrailMCPTool(config_path, predefined_prompts_path)


def validate_prompt(prompt: str) -> Dict[str, Any]:
    """MCP tool: Validate a prompt"""
    if _tool_instance is None:
        initialize_tool()
    return _tool_instance.validate_prompt(prompt)


def get_supported_categories() -> Dict[str, Any]:
    """MCP tool: Get supported categories"""
    if _tool_instance is None:
        initialize_tool()
    return _tool_instance.get_supported_categories()


def explain_rejection(prompt: str) -> Dict[str, Any]:
    """MCP tool: Explain rejection"""
    if _tool_instance is None:
        initialize_tool()
    return _tool_instance.explain_rejection(prompt)
