"""
Centroid-based guardrail system with per-intent thresholds and LLM verification

This implements the copilot-recommended approach:
1. Closed list of intents (7 routes) with multiple paraphrases each
2. Compute a centroid per intent
3. Per-intent thresholds (not one global threshold)
4. Gray band: if score ∈ [θ−δ, θ), ask Gemini 2.5 Pro for verification
5. Out-of-scope if low score or ambiguous (low margin)
6. Offline learning: collect near-misses and add as paraphrases
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class CentroidGuardrail:
    """
    Centroid-based guardrail with per-intent thresholds and gray band verification

    This uses multiple paraphrases per intent to compute a centroid, allowing for
    better semantic understanding of each intent category.
    """

    def __init__(
        self,
        intent_paraphrases_path: str = "config/intent_paraphrases.json",
        embed_model: str = "text-embedding-004",
        llm_model: str = "gemini-2.5-pro-latest",
        gray_band_delta: float = 0.05,
        min_margin: float = 0.04,
        api_key: Optional[str] = None,
        cache_dir: str = "data/centroids",
        log_file: Optional[str] = "logs/centroid_guardrail.log",
    ):
        """
        Initialize the centroid-based guardrail

        Args:
            intent_paraphrases_path: Path to JSON file with intent paraphrases
            embed_model: Gemini embedding model to use
            llm_model: Gemini LLM model for gray band verification
            gray_band_delta: Width of gray band below threshold
            min_margin: Minimum margin between top-1 and top-2 intents
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            cache_dir: Directory for caching centroid embeddings
            log_file: Path to log file
        """
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.gray_band_delta = gray_band_delta
        self.min_margin = min_margin
        self.cache_dir = Path(cache_dir)
        self.log_file = log_file

        # Configure Gemini API
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be provided or set in environment")
        genai.configure(api_key=api_key)

        # Load intent configuration
        self.intent_paraphrases: Dict[str, List[str]] = {}
        self.intent_thresholds: Dict[str, float] = {}
        self.intent_metadata: Dict[str, Dict] = {}
        self._load_intent_config(intent_paraphrases_path)

        # Centroid cache
        self.centroids: Dict[str, np.ndarray] = {}

        # Setup logging
        self._setup_logging()

        # Build centroids
        self._build_centroids()

        logger.info("CentroidGuardrail initialized successfully")

    def _setup_logging(self) -> None:
        """Setup logging for guardrail system"""
        if self.log_file:
            log_dir = Path(self.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def _load_intent_config(self, config_path: str) -> None:
        """
        Load intent paraphrases and thresholds from JSON file

        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)

            self.intent_paraphrases = data.get('intent_paraphrases', {})
            self.intent_thresholds = data.get('intent_thresholds', {})
            self.intent_metadata = data.get('intent_metadata', {})

            # Set default thresholds if not provided
            for intent in self.intent_paraphrases.keys():
                if intent not in self.intent_thresholds:
                    self.intent_thresholds[intent] = 0.70

            logger.info(f"Loaded {len(self.intent_paraphrases)} intents with paraphrases")

        except FileNotFoundError:
            logger.error(f"Intent paraphrases file not found: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading intent configuration: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def embed(self, text: str) -> np.ndarray:
        """
        Generate L2-normalized embedding for text

        Args:
            text: Input text to embed

        Returns:
            L2-normalized embedding vector as numpy array
        """
        text = (text or "").strip()
        if not text:
            # Return zero vector for empty text
            return np.zeros(768, dtype=np.float32)

        try:
            resp = genai.embed_content(model=self.embed_model, content=text)
            v = np.array(resp["embedding"], dtype=np.float32)

            # L2 normalize for dot product = cosine similarity
            norm = np.linalg.norm(v) + 1e-12
            return v / norm

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def _build_centroids(self) -> None:
        """
        Build centroid embeddings for all intents

        For each intent, embeds all paraphrases and computes the mean (centroid).
        The centroid is then L2-normalized.
        """
        # Check cache
        cache_file = self.cache_dir / "centroids.npz"
        if cache_file.exists():
            try:
                logger.info("Loading cached centroids...")
                data = np.load(cache_file, allow_pickle=True)
                self.centroids = {
                    intent: data[intent] for intent in self.intent_paraphrases.keys()
                }
                logger.info("Loaded cached centroids successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached centroids: {e}")

        logger.info("Building centroids from paraphrases...")

        for intent, examples in self.intent_paraphrases.items():
            logger.info(f"Processing intent '{intent}' with {len(examples)} paraphrases")

            # Embed all paraphrases
            vecs = [self.embed(example) for example in examples]

            # Compute centroid (mean)
            centroid = np.mean(vecs, axis=0)

            # L2 normalize the centroid
            centroid /= (np.linalg.norm(centroid) + 1e-12)

            self.centroids[intent] = centroid

        # Save to cache
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, **self.centroids)
            logger.info(f"Saved centroids cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save centroids cache: {e}")

        logger.info("Built centroids successfully")

    def score_intents(self, user_text: str) -> Tuple[str, float, float, Dict[str, float]]:
        """
        Score user text against all intent centroids

        Args:
            user_text: User's input text

        Returns:
            Tuple of (best_intent, best_score, margin, all_scores)
            - best_intent: Intent with highest similarity
            - best_score: Highest similarity score
            - margin: Difference between top-1 and top-2 scores
            - all_scores: Dictionary of all intent scores
        """
        # Embed user text
        user_vec = self.embed(user_text)

        # Compute dot product with all centroids (equals cosine for normalized vectors)
        scores = {
            intent: float(np.dot(user_vec, centroid))
            for intent, centroid in self.centroids.items()
        }

        # Find best intent
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        # Compute margin (top-1 - top-2)
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)

        return best_intent, best_score, margin, scores

    # LLM yes/no keywords
    YES_KEYWORDS = {"yes", "y", "true", "correct", "affirmative", "indeed", "certainly"}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def llm_verify(self, user_text: str, candidate_intent: str) -> bool:
        """
        Use LLM to verify if user text matches candidate intent

        This is used in the gray band to get a binary yes/no decision.

        Args:
            user_text: User's input text
            candidate_intent: Candidate intent to verify

        Returns:
            True if LLM confirms the intent, False otherwise
        """
        # Build prompt with closed list
        intent_list = list(self.intent_paraphrases.keys())

        prompt = f"""You are a strict intent verifier for a business app.
Intents (closed list): {intent_list}

User query: "{user_text}"
Candidate intent: "{candidate_intent}"

Answer ONLY 'Yes' or 'No':
Is the user query best categorized as the candidate intent, within this closed list?
"""

        try:
            model = genai.GenerativeModel(self.llm_model)
            resp = model.generate_content(prompt)

            # Extract first word and check if it's affirmative
            answer = (resp.text or "").strip().lower().split()[0]
            is_verified = answer in self.YES_KEYWORDS

            logger.info(
                f"LLM verification for intent '{candidate_intent}': "
                f"answer='{answer}', verified={is_verified}"
            )

            return is_verified

        except Exception as e:
            logger.error(f"Error in LLM verification: {e}")
            # On error, default to not verified
            return False

    def decide(self, user_text: str) -> Dict:
        """
        Make guardrail decision for user input

        Decision logic:
        1. If score >= threshold and margin >= min_margin: APPROVE
        2. If score in [threshold - delta, threshold) and margin >= min_margin/2:
           Ask LLM, then APPROVE or REJECT based on response
        3. Otherwise: REJECT as out-of-scope or ambiguous

        Args:
            user_text: User's input text

        Returns:
            Decision dictionary with keys:
            - allowed: bool
            - intent: str (if allowed)
            - route: str (if allowed)
            - score: float
            - scores: Dict[str, float]
            - reason: str
            - verified_by_llm: bool (optional)
        """
        user_text = (user_text or "").strip()

        # Handle empty input
        if not user_text:
            return {
                "allowed": False,
                "reason": "empty_input",
                "score": 0.0,
                "scores": {}
            }

        # Score against all intents
        best_intent, best_score, margin, scores = self.score_intents(user_text)
        theta = self.intent_thresholds.get(best_intent, 0.70)

        logger.info(
            f"User: '{user_text[:100]}' | "
            f"Intent: {best_intent} | "
            f"Score: {best_score:.3f} | "
            f"Threshold: {theta:.3f} | "
            f"Margin: {margin:.3f}"
        )

        # Clear accept path
        if best_score >= theta and margin >= self.min_margin:
            logger.info(f"APPROVED: High confidence match")
            return {
                "allowed": True,
                "intent": best_intent,
                "route": best_intent,
                "score": best_score,
                "scores": scores,
                "reason": "pass_threshold"
            }

        # Gray band → ask LLM verifier
        if (theta - self.gray_band_delta <= best_score < theta and
            margin >= (self.min_margin / 2)):

            logger.info(f"GRAY BAND: Requesting LLM verification")

            if self.llm_verify(user_text, best_intent):
                logger.info(f"APPROVED: LLM verified in gray band")
                return {
                    "allowed": True,
                    "intent": best_intent,
                    "route": best_intent,
                    "score": best_score,
                    "scores": scores,
                    "reason": "grayband_llm_verified",
                    "verified_by_llm": True
                }
            else:
                logger.info(f"REJECTED: LLM did not verify in gray band")
                self._log_near_miss(user_text, best_intent, best_score, "llm_rejected")

        # Otherwise reject as out-of-scope
        logger.warning(f"REJECTED: Out of scope or ambiguous")
        self._log_near_miss(user_text, best_intent, best_score, "out_of_scope")

        return {
            "allowed": False,
            "reason": "out_of_scope_or_ambiguous",
            "score": best_score,
            "scores": scores,
            "best_intent": best_intent,
            "margin": margin
        }

    def _log_near_miss(
        self,
        user_text: str,
        intent: str,
        score: float,
        reason: str
    ) -> None:
        """
        Log near-miss rejections for offline learning

        These can be reviewed and potentially added as new paraphrases.

        Args:
            user_text: User's input text
            intent: Best matching intent
            score: Similarity score
            reason: Reason for rejection
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_text": user_text,
            "intent": intent,
            "score": score,
            "reason": reason,
            "type": "near_miss"
        }
        logger.info(f"NEAR_MISS: {json.dumps(log_entry)}")

        # Also save to separate near-miss file for offline learning
        near_miss_file = Path(self.log_file).parent / "near_misses.jsonl"
        try:
            with open(near_miss_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write near-miss log: {e}")

    def add_paraphrase(self, intent: str, paraphrase: str, rebuild: bool = True) -> None:
        """
        Add a new paraphrase to an intent and optionally rebuild centroids

        This supports offline learning by allowing you to add near-misses
        as new paraphrases.

        Args:
            intent: Intent to add paraphrase to
            paraphrase: New paraphrase text
            rebuild: Whether to rebuild centroids immediately
        """
        if intent not in self.intent_paraphrases:
            raise ValueError(f"Unknown intent: {intent}")

        self.intent_paraphrases[intent].append(paraphrase)
        logger.info(f"Added paraphrase to '{intent}': {paraphrase}")

        if rebuild:
            # Rebuild only this intent's centroid
            examples = self.intent_paraphrases[intent]
            vecs = [self.embed(example) for example in examples]
            centroid = np.mean(vecs, axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-12)
            self.centroids[intent] = centroid
            logger.info(f"Rebuilt centroid for '{intent}'")

    def save_config(self, output_path: str) -> None:
        """
        Save current intent configuration (including new paraphrases) to file

        Args:
            output_path: Path to save configuration
        """
        config = {
            "intent_paraphrases": self.intent_paraphrases,
            "intent_thresholds": self.intent_thresholds,
            "intent_metadata": self.intent_metadata,
            "updated_at": datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved configuration to {output_path}")

    def get_intent_info(self) -> Dict:
        """
        Get information about configured intents

        Returns:
            Dictionary with intent information
        """
        return {
            "intents": list(self.intent_paraphrases.keys()),
            "total_intents": len(self.intent_paraphrases),
            "paraphrase_counts": {
                intent: len(examples)
                for intent, examples in self.intent_paraphrases.items()
            },
            "thresholds": self.intent_thresholds,
            "metadata": self.intent_metadata
        }
