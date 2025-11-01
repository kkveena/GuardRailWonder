"""
Example usage of centroid-based guardrail with per-intent thresholds

This demonstrates:
1. Basic guardrail usage
2. Gray band verification
3. Adding new paraphrases
4. Offline learning from near-misses
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.guardrails.centroid_guardrail import CentroidGuardrail


def main():
    """Main example function"""

    # Make sure API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Please set it with: export GOOGLE_API_KEY=your_api_key")
        return

    print("="*70)
    print("CENTROID-BASED GUARDRAIL EXAMPLE")
    print("="*70)

    # Initialize guardrail
    print("\n1. Initializing guardrail with intent paraphrases...")
    guardrail = CentroidGuardrail(
        intent_paraphrases_path="config/intent_paraphrases.json",
        gray_band_delta=0.05,
        min_margin=0.04
    )

    # Show intent information
    print("\n2. Intent Information:")
    info = guardrail.get_intent_info()
    print(f"   Total intents: {info['total_intents']}")
    for intent, count in info['paraphrase_counts'].items():
        threshold = info['thresholds'][intent]
        print(f"   - {intent}: {count} paraphrases (threshold: {threshold:.2f})")

    # Test various queries
    print("\n3. Testing Queries:")
    print("-"*70)

    test_queries = [
        # Should match check_balance
        "What's my account balance?",
        "show me my balance",
        "how much money do I have",

        # Should match make_payment
        "I want to pay my bill",
        "process a payment",

        # Should match refund_status
        "where is my refund",
        "track my refund",

        # Should match contact_agent
        "talk to a human",
        "I need help from support",

        # Edge cases - might hit gray band
        "check my funds",  # Similar to check_balance
        "pay something",    # Vague payment intent

        # Out of scope
        "tell me a joke",
        "what's the weather today",
        "write me a poem"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        result = guardrail.decide(query)

        if result['allowed']:
            print(f"  ✓ APPROVED")
            print(f"    Intent: {result['intent']}")
            print(f"    Score: {result['score']:.3f}")
            print(f"    Reason: {result['reason']}")
            if result.get('verified_by_llm'):
                print(f"    Note: Verified by LLM in gray band")
        else:
            print(f"  ✗ REJECTED")
            print(f"    Score: {result['score']:.3f}")
            print(f"    Reason: {result['reason']}")
            if 'best_intent' in result:
                print(f"    Closest intent: {result['best_intent']}")

    # Demonstrate adding paraphrases
    print("\n" + "="*70)
    print("4. Adding New Paraphrases (Offline Learning)")
    print("="*70)

    print("\nAdding 'check my funds' as a paraphrase for 'check_balance'...")
    guardrail.add_paraphrase("check_balance", "check my funds", rebuild=True)

    print("Re-testing 'check my funds'...")
    result = guardrail.decide("check my funds")
    if result['allowed']:
        print(f"  ✓ Now APPROVED with score: {result['score']:.3f}")
    else:
        print(f"  Still rejected with score: {result['score']:.3f}")

    # Show intent scores for a query
    print("\n" + "="*70)
    print("5. Detailed Intent Scoring")
    print("="*70)

    query = "I want to check my account"
    print(f"\nQuery: '{query}'")

    best_intent, best_score, margin, all_scores = guardrail.score_intents(query)

    print(f"\nIntent Scores:")
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    for intent, score in sorted_scores:
        threshold = guardrail.intent_thresholds[intent]
        indicator = "✓" if score >= threshold else "✗"
        print(f"  {indicator} {intent:20s}: {score:.3f} (threshold: {threshold:.2f})")

    print(f"\nBest Intent: {best_intent}")
    print(f"Best Score: {best_score:.3f}")
    print(f"Margin (top1 - top2): {margin:.3f}")

    # Save updated configuration
    print("\n" + "="*70)
    print("6. Saving Updated Configuration")
    print("="*70)

    output_path = "config/intent_paraphrases_updated.json"
    guardrail.save_config(output_path)
    print(f"\nSaved updated configuration to: {output_path}")

    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)

    print("\nNext steps:")
    print("1. Review logs at: logs/centroid_guardrail.log")
    print("2. Check near-misses at: logs/near_misses.jsonl")
    print("3. Tune thresholds with: python tune.py --dev-file data/dev_data.jsonl")
    print("4. Run tests with: pytest tests/test_centroid_guardrail.py -v")
    print("5. Start API server with: python app.py")


if __name__ == "__main__":
    main()
