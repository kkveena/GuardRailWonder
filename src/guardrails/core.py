"""
Core guardrail system implementation
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

from .models import (
    GuardrailConfig,
    GuardrailResult,
    GuardrailDecision,
    PredefinedPrompt,
)
from .embeddings import EmbeddingProvider, GeminiEmbeddingProvider

logger = logging.getLogger(__name__)


class PromptGuardrail:
    """
    Main guardrail system using embedding-based similarity matching

    This implements your proposed approach:
    1. Embed incoming prompt (1×n)
    2. Compare with predefined prompts (7×n)
    3. Compute cosine similarity (1×7)
    4. Apply threshold-based decision
    """

    def __init__(
        self,
        config: GuardrailConfig,
        embedding_provider: Optional[EmbeddingProvider] = None,
        predefined_prompts_path: Optional[str] = None,
    ):
        """
        Initialize the guardrail system

        Args:
            config: Guardrail configuration
            embedding_provider: Provider for generating embeddings
            predefined_prompts_path: Path to predefined prompts JSON file
        """
        self.config = config
        self.embedding_provider = embedding_provider or GeminiEmbeddingProvider(
            model=config.embedding_model
        )

        # Load predefined prompts
        self.predefined_prompts: List[PredefinedPrompt] = []
        self._load_predefined_prompts(predefined_prompts_path)

        # Cache for embeddings
        self._embeddings_cache: Dict[str, List[float]] = {}

        # Setup logging
        self._setup_logging()

        logger.info("PromptGuardrail initialized successfully")

    def _setup_logging(self) -> None:
        """Setup logging for guardrail system"""
        if self.config.log_rejections and self.config.log_file:
            log_dir = Path(self.config.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def _load_predefined_prompts(self, prompts_path: Optional[str] = None) -> None:
        """
        Load predefined prompts from JSON file

        Args:
            prompts_path: Path to JSON file containing predefined prompts
        """
        if prompts_path is None:
            prompts_path = "config/predefined_prompts.json"

        try:
            with open(prompts_path, 'r') as f:
                data = json.load(f)

            self.predefined_prompts = [
                PredefinedPrompt(**prompt) for prompt in data['prompts']
            ]

            logger.info(f"Loaded {len(self.predefined_prompts)} predefined prompts")

        except FileNotFoundError:
            logger.error(f"Predefined prompts file not found: {prompts_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading predefined prompts: {e}")
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text with caching

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if self.config.cache_embeddings and text in self._embeddings_cache:
            return self._embeddings_cache[text]

        embedding_response = self.embedding_provider.embed(text)
        embedding = embedding_response.embedding

        if self.config.cache_embeddings:
            self._embeddings_cache[text] = embedding

        return embedding

    def initialize_predefined_embeddings(self) -> None:
        """
        Pre-compute embeddings for all predefined prompts

        This should be called once during initialization to speed up
        subsequent guardrail checks
        """
        logger.info("Pre-computing embeddings for predefined prompts...")

        # Check if cached embeddings exist
        cache_file = Path(self.config.cache_dir) / "predefined_embeddings.npy"

        if self.config.cache_embeddings and cache_file.exists():
            logger.info("Loading cached predefined embeddings...")
            try:
                embeddings = np.load(cache_file)
                for i, prompt in enumerate(self.predefined_prompts):
                    prompt.embedding = embeddings[i].tolist()
                logger.info("Loaded cached embeddings successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")

        # Generate embeddings
        texts = [prompt.template for prompt in self.predefined_prompts]
        embeddings_responses = self.embedding_provider.embed_batch(texts)

        for i, prompt in enumerate(self.predefined_prompts):
            prompt.embedding = embeddings_responses[i].embedding

        # Save to cache
        if self.config.cache_embeddings:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                embeddings_array = np.array([p.embedding for p in self.predefined_prompts])
                np.save(cache_file, embeddings_array)
                logger.info(f"Saved embeddings cache to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save embeddings cache: {e}")

        logger.info("Pre-computed embeddings successfully")

    def compute_similarity_scores(
        self, prompt_embedding: List[float]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute similarity scores between prompt and all predefined prompts

        This implements your matrix multiplication approach:
        (1×n) @ (7×n)^T = (1×7)

        Args:
            prompt_embedding: Embedding of incoming prompt

        Returns:
            Tuple of (similarity_scores_array, prompt_id_to_score_dict)
        """
        reference_embeddings = [p.embedding for p in self.predefined_prompts]

        # Use the embedding provider's similarity computation
        if hasattr(self.embedding_provider, 'compute_similarity_matrix'):
            similarity_scores = self.embedding_provider.compute_similarity_matrix(
                prompt_embedding, reference_embeddings
            )
        else:
            # Fallback to manual computation
            query_vec = np.array(prompt_embedding).reshape(1, -1)
            ref_matrix = np.array(reference_embeddings)

            query_norm = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
            ref_norm = ref_matrix / np.linalg.norm(ref_matrix, axis=1, keepdims=True)

            similarity_scores = np.dot(query_norm, ref_norm.T).flatten()

        # Create prompt_id -> score mapping
        scores_dict = {
            self.predefined_prompts[i].id: float(similarity_scores[i])
            for i in range(len(self.predefined_prompts))
        }

        return similarity_scores, scores_dict

    def evaluate(self, prompt: str) -> GuardrailResult:
        """
        Evaluate an incoming prompt against guardrails

        Args:
            prompt: User's input prompt

        Returns:
            GuardrailResult with decision and metadata
        """
        logger.info(f"Evaluating prompt: {prompt[:100]}...")

        # Get embedding for incoming prompt
        prompt_embedding = self._get_embedding(prompt)

        # Compute similarity scores
        similarity_scores, scores_dict = self.compute_similarity_scores(prompt_embedding)

        # Get argmax (most similar prompt)
        max_idx = int(np.argmax(similarity_scores))
        max_score = float(similarity_scores[max_idx])
        matched_prompt = self.predefined_prompts[max_idx]

        logger.info(
            f"Max similarity: {max_score:.3f} with prompt '{matched_prompt.id}'"
        )

        # Apply threshold-based decision
        if max_score >= self.config.threshold_high:
            decision = GuardrailDecision.APPROVED
            message = (
                f"Prompt approved. High confidence match with '{matched_prompt.category}' "
                f"(similarity: {max_score:.3f})"
            )
        elif max_score >= self.config.threshold_medium:
            decision = GuardrailDecision.APPROVED_WITH_WARNING
            message = (
                f"Prompt approved with warning. Medium confidence match with "
                f"'{matched_prompt.category}' (similarity: {max_score:.3f}). "
                f"Please verify intent."
            )
        else:
            decision = GuardrailDecision.REJECTED
            message = (
                f"Prompt rejected. Low similarity score ({max_score:.3f}). "
                f"Please rephrase to match one of the supported query types."
            )

            # Log rejection
            if self.config.log_rejections:
                self._log_rejection(prompt, max_score, matched_prompt.id)

        result = GuardrailResult(
            decision=decision,
            similarity_score=max_score,
            matched_prompt_id=matched_prompt.id,
            matched_prompt_category=matched_prompt.category,
            message=message,
            all_scores=scores_dict,
        )

        return result

    def _log_rejection(
        self, prompt: str, score: float, matched_id: str
    ) -> None:
        """Log rejected prompts for analysis"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "similarity_score": score,
            "matched_prompt_id": matched_id,
            "decision": "rejected",
        }
        logger.warning(f"REJECTION: {json.dumps(log_entry)}")

    def get_supported_categories(self) -> List[str]:
        """Get list of supported prompt categories"""
        return list(set(p.category for p in self.predefined_prompts))

    def get_predefined_templates(self) -> List[Dict[str, str]]:
        """Get list of predefined prompt templates"""
        return [
            {
                "id": p.id,
                "template": p.template,
                "category": p.category,
                "description": p.description,
            }
            for p in self.predefined_prompts
        ]
