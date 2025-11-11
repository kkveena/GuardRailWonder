#!/usr/bin/env python3
"""
Test API with explicit no-proxy configuration
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("=" * 70)
print("Testing Gemini API (NO PROXY)")
print("=" * 70)

# Explicitly set no proxy for this request
session = requests.Session()
session.proxies = {
    'http': None,
    'https': None,
}

print(f"\nAPI Key: {api_key[:15]}...{api_key[-4:]}")

# Try embeddings with no proxy
print("\nTesting embeddings API...")
url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={api_key}"

data = {
    "model": "models/embedding-001",
    "content": {
        "parts": [{"text": "What is the status of my order?"}]
    }
}

try:
    response = session.post(url, json=data, timeout=30)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        embedding = result.get('embedding', {}).get('values', [])
        print(f"\n✓ SUCCESS! Embedding generated")
        print(f"  - Dimension: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
        print("\n✓ Your API key is working perfectly!")
    else:
        print(f"✗ ERROR: {response.status_code}")
        print(f"Response: {response.text[:500]}")

except Exception as e:
    print(f"✗ ERROR: {e}")

print("=" * 70)
