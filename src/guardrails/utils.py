"""
Utility functions for the guardrail system
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from .models import GuardrailConfig

logger = logging.getLogger(__name__)


def load_config_from_env(env_file: Optional[str] = None) -> GuardrailConfig:
    """
    Load configuration from environment variables

    Args:
        env_file: Path to .env file (optional)

    Returns:
        GuardrailConfig instance
    """
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    import os

    config = GuardrailConfig(
        threshold_high=float(os.getenv("GUARDRAIL_THRESHOLD_HIGH", "0.8")),
        threshold_medium=float(os.getenv("GUARDRAIL_THRESHOLD_MEDIUM", "0.5")),
        log_rejections=os.getenv("GUARDRAIL_LOG_REJECTIONS", "true").lower() == "true",
        log_file=os.getenv("GUARDRAIL_LOG_FILE", "logs/guardrail.log"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "models/embedding-001"),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
        cache_embeddings=os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true",
        cache_dir=os.getenv("CACHE_DIR", "data/embeddings"),
    )

    logger.info("Configuration loaded from environment")
    return config


def load_config_from_file(config_path: str) -> GuardrailConfig:
    """
    Load configuration from JSON file

    Args:
        config_path: Path to JSON configuration file

    Returns:
        GuardrailConfig instance
    """
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    config = GuardrailConfig(**config_data)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_config_to_file(config: GuardrailConfig, config_path: str) -> None:
    """
    Save configuration to JSON file

    Args:
        config: GuardrailConfig instance
        config_path: Path to save configuration
    """
    config_dict = config.model_dump()

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Configuration saved to {config_path}")


def validate_predefined_prompts(prompts_path: str) -> bool:
    """
    Validate predefined prompts JSON file

    Args:
        prompts_path: Path to prompts JSON file

    Returns:
        True if valid, raises exception otherwise
    """
    try:
        with open(prompts_path, 'r') as f:
            data = json.load(f)

        # Check required fields
        assert 'prompts' in data, "Missing 'prompts' field"
        assert isinstance(data['prompts'], list), "'prompts' must be a list"

        for i, prompt in enumerate(data['prompts']):
            assert 'id' in prompt, f"Prompt {i} missing 'id'"
            assert 'template' in prompt, f"Prompt {i} missing 'template'"
            assert 'category' in prompt, f"Prompt {i} missing 'category'"
            assert 'description' in prompt, f"Prompt {i} missing 'description'"

        logger.info(f"Validated {len(data['prompts'])} prompts in {prompts_path}")
        return True

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> None:
    """
    Setup logging configuration

    Args:
        level: Logging level
        log_file: Optional file to log to
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


def analyze_rejection_logs(log_file: str) -> Dict[str, Any]:
    """
    Analyze rejection logs to identify patterns

    Args:
        log_file: Path to log file

    Returns:
        Dict with analysis results
    """
    rejections = []

    with open(log_file, 'r') as f:
        for line in f:
            if 'REJECTION' in line:
                try:
                    # Extract JSON from log line
                    json_start = line.index('{')
                    rejection_data = json.loads(line[json_start:])
                    rejections.append(rejection_data)
                except (ValueError, json.JSONDecodeError):
                    continue

    if not rejections:
        return {
            "total_rejections": 0,
            "message": "No rejections found in log file"
        }

    # Analyze patterns
    avg_score = sum(r['similarity_score'] for r in rejections) / len(rejections)

    matched_ids = {}
    for r in rejections:
        matched_id = r.get('matched_prompt_id', 'unknown')
        matched_ids[matched_id] = matched_ids.get(matched_id, 0) + 1

    return {
        "total_rejections": len(rejections),
        "average_similarity_score": avg_score,
        "most_common_matches": sorted(
            matched_ids.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5],
        "recent_rejections": rejections[-10:],  # Last 10
    }
