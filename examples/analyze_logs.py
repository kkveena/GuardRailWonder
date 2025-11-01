#!/usr/bin/env python3
"""
Log analysis example for GuardRail Wonder

This example shows how to analyze rejection logs to identify patterns
and potential gaps in predefined prompts.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from guardrails.utils import analyze_rejection_logs, setup_logging
import json


def main():
    """Main example function"""

    setup_logging()

    print("=" * 60)
    print("GuardRail Wonder - Log Analysis Example")
    print("=" * 60)

    log_file = "logs/guardrail.log"

    if not Path(log_file).exists():
        print(f"\nLog file not found: {log_file}")
        print("Run the basic_usage.py example first to generate logs.")
        return

    print(f"\nAnalyzing rejection logs from: {log_file}")
    print("-" * 60)

    analysis = analyze_rejection_logs(log_file)

    print(f"\nTotal rejections: {analysis['total_rejections']}")

    if analysis['total_rejections'] > 0:
        print(f"Average similarity score: {analysis['average_similarity_score']:.3f}")

        print("\nMost common matches (prompts closest to rejections):")
        for prompt_id, count in analysis['most_common_matches']:
            print(f"  • {prompt_id}: {count} rejections")

        print("\nRecent rejections:")
        for i, rejection in enumerate(analysis['recent_rejections'][-5:], 1):
            print(f"\n  [{i}] {rejection['prompt']}")
            print(f"      Score: {rejection['similarity_score']:.3f}")
            print(f"      Closest: {rejection['matched_prompt_id']}")

        print("\n" + "=" * 60)
        print("Recommendations:")
        print("=" * 60)
        print("""
1. Review rejected prompts to identify common patterns
2. Consider adding new predefined prompts for frequently rejected queries
3. Adjust thresholds if needed based on similarity scores
4. Monitor which predefined prompts are being matched most often
        """)

    print("\n" + "=" * 60)
    print("Analysis completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
