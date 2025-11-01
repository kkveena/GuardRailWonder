#!/usr/bin/env python3
"""
MCP Tool usage example for GuardRail Wonder

This example shows how to use the MCP tool interface.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from guardrails.mcp_tool import GuardrailMCPTool
from guardrails.utils import setup_logging
from dotenv import load_dotenv
import json


def main():
    """Main example function"""

    # Setup
    setup_logging()
    load_dotenv()

    print("=" * 60)
    print("GuardRail Wonder - MCP Tool Example")
    print("=" * 60)

    # Initialize MCP tool
    print("\nInitializing MCP Tool...")
    tool = GuardrailMCPTool(
        predefined_prompts_path="config/predefined_prompts.json"
    )
    print("✓ MCP Tool initialized")

    # Get tool metadata
    print("\n" + "=" * 60)
    print("Tool Metadata")
    print("=" * 60)
    metadata = tool.get_tool_metadata()
    print(json.dumps(metadata, indent=2))

    # Get supported categories
    print("\n" + "=" * 60)
    print("Supported Categories")
    print("=" * 60)
    categories = tool.get_supported_categories()
    print(f"\nTotal templates: {categories['total']}")
    print(f"Categories: {', '.join(categories['categories'])}")

    # Validate prompts
    print("\n" + "=" * 60)
    print("Validating Prompts")
    print("=" * 60)

    test_prompts = [
        "What is the status of my order?",  # Should be approved
        "Tell me a joke",  # Should be rejected
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: \"{prompt}\"")
        print("-" * 60)

        result = tool.validate_prompt(prompt)
        print(json.dumps(result, indent=2))

        if not result['approved']:
            print("\nGetting explanation for rejection...")
            explanation = tool.explain_rejection(prompt)
            print("\nRejection explanation:")
            print(json.dumps(explanation, indent=2))

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
