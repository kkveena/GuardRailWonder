#!/usr/bin/env python3
"""
MCP Server for GuardRail Wonder

This script starts an MCP server that exposes the guardrail system
as callable tools.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from guardrails.mcp_tool import (
    initialize_tool,
    validate_prompt,
    get_supported_categories,
    explain_rejection,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for MCP server"""
    try:
        # Initialize the guardrail tool
        logger.info("Initializing GuardRail Wonder MCP Server...")

        initialize_tool(
            config_path=None,  # Will use environment variables
            predefined_prompts_path="config/predefined_prompts.json"
        )

        logger.info("GuardRail Wonder MCP Server initialized successfully")
        logger.info("Available tools:")
        logger.info("  - validate_prompt(prompt: str)")
        logger.info("  - get_supported_categories()")
        logger.info("  - explain_rejection(prompt: str)")

        # In a real MCP implementation, you would start the MCP server here
        # For now, this serves as a demonstration

        print("\n=== GuardRail Wonder MCP Server Ready ===\n")
        print("Example usage:")
        print("  result = validate_prompt('What is the status of my order?')")
        print("  categories = get_supported_categories()")
        print("  explanation = explain_rejection('Tell me a joke')")
        print("\nPress Ctrl+C to exit\n")

        # Keep server running
        import time
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down MCP server...")
    except Exception as e:
        logger.error(f"Error starting MCP server: {e}")
        raise


if __name__ == "__main__":
    main()
