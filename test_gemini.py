#!/usr/bin/env python3
"""
Quick test script to verify Gemini credentials and guardrail system

This script will:
1. Verify Gemini API key is valid
2. Test embedding generation
3. Test guardrail evaluation with real prompts
4. Display detailed results
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from guardrails import PromptGuardrail, GuardrailConfig
from guardrails.embeddings import GeminiEmbeddingProvider
from guardrails.utils import setup_logging
from dotenv import load_dotenv


def test_api_key():
    """Test if Gemini API key is valid"""
    print("\n" + "=" * 70)
    print("STEP 1: Testing Gemini API Key")
    print("=" * 70)

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in environment")
        print("\nPlease set your API key:")
        print("1. Copy .env.example to .env")
        print("2. Edit .env and add: GEMINI_API_KEY=your_key_here")
        return False

    print(f"✓ API Key found: {api_key[:10]}...{api_key[-4:]}")
    return True


def test_embedding_generation():
    """Test basic embedding generation"""
    print("\n" + "=" * 70)
    print("STEP 2: Testing Embedding Generation")
    print("=" * 70)

    try:
        provider = GeminiEmbeddingProvider(model="models/embedding-001")

        test_text = "What is the status of my order?"
        print(f"\nGenerating embedding for: '{test_text}'")

        result = provider.embed(test_text)

        print(f"✓ Embedding generated successfully!")
        print(f"  - Model: {result.model}")
        print(f"  - Dimension: {result.dimension}")
        print(f"  - First 5 values: {result.embedding[:5]}")

        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to generate embedding")
        print(f"  - Error: {str(e)}")
        print("\nPossible issues:")
        print("1. Invalid API key")
        print("2. No internet connection")
        print("3. Gemini API service down")
        return False


def test_guardrail_initialization():
    """Test guardrail system initialization"""
    print("\n" + "=" * 70)
    print("STEP 3: Initializing Guardrail System")
    print("=" * 70)

    try:
        config = GuardrailConfig(
            threshold_high=0.9,
            threshold_medium=0.85,
            log_rejections=True,
            embedding_model="models/embedding-001"
        )

        provider = GeminiEmbeddingProvider(model=config.embedding_model)

        guardrail = PromptGuardrail(
            config=config,
            embedding_provider=provider,
            predefined_prompts_path="config/predefined_prompts.json"
        )

        print(f"✓ Guardrail initialized")
        print(f"  - Predefined prompts loaded: {len(guardrail.predefined_prompts)}")
        print(f"  - High threshold: {config.threshold_high}")
        print(f"  - Medium threshold: {config.threshold_medium}")

        # Pre-compute embeddings
        print("\nPre-computing embeddings for predefined prompts...")
        print("(This may take 10-20 seconds for 7 prompts)")

        guardrail.initialize_predefined_embeddings()

        print("✓ Embeddings pre-computed and cached")

        return guardrail

    except Exception as e:
        print(f"❌ ERROR: Failed to initialize guardrail")
        print(f"  - Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_prompt_evaluation(guardrail):
    """Test prompt evaluation with various examples"""
    print("\n" + "=" * 70)
    print("STEP 4: Testing Prompt Evaluation")
    print("=" * 70)

    test_cases = [
        {
            "prompt": "What is the status of my order?",
            "expected": "approved",
            "description": "Direct match with predefined prompt"
        },
        {
            "prompt": "Where is my package? I want to track it.",
            "expected": "approved",
            "description": "Similar to order status query"
        },
        {
            "prompt": "Can you tell me about your product features and pricing?",
            "expected": "approved",
            "description": "Product information query"
        },
        {
            "prompt": "I need help with my account settings",
            "expected": "approved_with_warning",
            "description": "Somewhat related to account management"
        },
        {
            "prompt": "What's the weather like today?",
            "expected": "rejected",
            "description": "Completely off-topic"
        },
        {
            "prompt": "Tell me a funny joke",
            "expected": "rejected",
            "description": "Off-topic entertainment request"
        },
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Prompt: \"{test['prompt']}\"")
        print(f"Description: {test['description']}")
        print(f"Expected: {test['expected']}")

        try:
            result = guardrail.evaluate(test['prompt'])

            print(f"\nResult:")
            print(f"  Decision: {result.decision}")
            print(f"  Similarity Score: {result.similarity_score:.4f}")
            print(f"  Matched Category: {result.matched_prompt_category}")
            print(f"  Message: {result.message}")

            # Check if matches expectation
            match = "✓" if result.decision == test['expected'] else "⚠"
            print(f"\n{match} Expected: {test['expected']}, Got: {result.decision}")

            results.append({
                "test": test,
                "result": result,
                "matches": result.decision == test['expected']
            })

        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({
                "test": test,
                "error": str(e),
                "matches": False
            })

    return results


def print_summary(results):
    """Print test summary"""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r.get('matches', False))

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    print("\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status = "✓ PASS" if result.get('matches', False) else "✗ FAIL"
        prompt = result['test']['prompt'][:50]
        print(f"  {status} - Test {i}: {prompt}...")

    if passed == total:
        print("\n🎉 All tests passed! Guardrail system is working correctly.")
    else:
        print("\n⚠ Some tests failed. This might be expected due to threshold tuning.")
        print("Consider adjusting thresholds in config if needed.")


def main():
    """Main test function"""
    print("\n" + "=" * 70)
    print("GuardRail Wonder - Gemini API Test")
    print("=" * 70)

    setup_logging()

    # Step 1: Test API key
    if not test_api_key():
        return

    # Step 2: Test embedding generation
    if not test_embedding_generation():
        return

    # Step 3: Initialize guardrail
    guardrail = test_guardrail_initialization()
    if not guardrail:
        return

    # Step 4: Test prompt evaluation
    results = test_prompt_evaluation(guardrail)

    # Print summary
    print_summary(results)

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Check logs/guardrail.log for rejection details")
    print("2. Run: python examples/analyze_logs.py")
    print("3. Customize config/predefined_prompts.json for your use case")
    print("4. Tune thresholds based on results")


if __name__ == "__main__":
    main()
