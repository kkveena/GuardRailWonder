#!/usr/bin/env python3
"""
Basic usage example for GuardRail Wonder

This example shows how to use the guardrail system to validate prompts.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from guardrails import PromptGuardrail, GuardrailConfig, GeminiEmbeddingProvider
from guardrails.utils import setup_logging, load_config_from_env
from dotenv import load_dotenv


def main():
    """Main example function"""

    # Setup logging
    setup_logging()

    # Load environment variables
    load_dotenv()

    print("=" * 60)
    print("GuardRail Wonder - Basic Usage Example")
    print("=" * 60)

    # Initialize configuration
    config = load_config_from_env()
    print(f"\nConfiguration:")
    print(f"  High threshold: {config.threshold_high}")
    print(f"  Medium threshold: {config.threshold_medium}")
    print(f"  Embedding model: {config.embedding_model}")

    # Initialize embedding provider
    embedding_provider = GeminiEmbeddingProvider(
        model=config.embedding_model
    )

    # Initialize guardrail
    guardrail = PromptGuardrail(
        config=config,
        embedding_provider=embedding_provider,
        predefined_prompts_path="config/predefined_prompts.json"
    )

    # Pre-compute embeddings for predefined prompts
    print("\nInitializing predefined prompt embeddings...")
    guardrail.initialize_predefined_embeddings()
    print("✓ Embeddings initialized")

    # Display supported categories
    categories = guardrail.get_supported_categories()
    print(f"\nSupported categories ({len(categories)}):")
    for category in categories:
        print(f"  • {category}")

    # Test prompts
    test_prompts = [
        # Should be approved (high similarity)
        "What's the status of my recent order?",
        "Can you tell me about your products?",

        # Should be approved with warning (medium similarity)
        "I need some help with something",

        # Should be rejected (low similarity)
        "What's the weather like today?",
        "Tell me a joke",
    ]

    print("\n" + "=" * 60)
    print("Testing Prompts")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}] Prompt: \"{prompt}\"")
        print("-" * 60)

        result = guardrail.evaluate(prompt)

        print(f"Decision: {result.decision}")
        print(f"Similarity Score: {result.similarity_score:.3f}")
        print(f"Matched Category: {result.matched_prompt_category}")
        print(f"Message: {result.message}")

        if result.decision == "approved":
            print("✓ APPROVED - Routing to LLM")
        elif result.decision == "approved_with_warning":
            print("⚠ APPROVED WITH WARNING - Routing to LLM with logging")
        else:
            print("✗ REJECTED - Prompt not allowed")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
