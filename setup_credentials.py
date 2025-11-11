#!/usr/bin/env python3
"""
Interactive script to set up Gemini API credentials
"""

import os
from pathlib import Path


def setup_credentials():
    """Interactive credential setup"""
    print("=" * 70)
    print("GuardRail Wonder - Credential Setup")
    print("=" * 70)

    print("\nThis script will help you set up your Gemini API credentials.")
    print("\nTo get a Gemini API key:")
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Click 'Create API Key'")
    print("3. Copy the generated key")

    print("\n" + "-" * 70)
    api_key = input("\nEnter your Gemini API key: ").strip()

    if not api_key:
        print("❌ No API key provided. Exiting.")
        return

    # Create .env file
    env_file = Path(".env")

    if env_file.exists():
        print(f"\n⚠ Warning: {env_file} already exists.")
        overwrite = input("Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Aborted.")
            return

    # Write .env file
    with open(env_file, 'w') as f:
        f.write(f"# Gemini API Configuration\n")
        f.write(f"GEMINI_API_KEY={api_key}\n\n")
        f.write(f"# Guardrail Configuration\n")
        f.write(f"GUARDRAIL_THRESHOLD_HIGH=0.8\n")
        f.write(f"GUARDRAIL_THRESHOLD_MEDIUM=0.5\n")
        f.write(f"GUARDRAIL_LOG_REJECTIONS=true\n")
        f.write(f"GUARDRAIL_LOG_FILE=logs/guardrail.log\n\n")
        f.write(f"# Embedding Model\n")
        f.write(f"EMBEDDING_MODEL=models/embedding-001\n")
        f.write(f"EMBEDDING_DIMENSION=768\n")

    print(f"\n✓ Created {env_file}")
    print(f"✓ API key configured: {api_key[:10]}...{api_key[-4:]}")

    print("\n" + "=" * 70)
    print("Setup complete!")
    print("=" * 70)
    print("\nNext step: Run the test script")
    print("  python test_gemini.py")


if __name__ == "__main__":
    setup_credentials()
