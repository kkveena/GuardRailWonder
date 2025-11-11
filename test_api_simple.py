#!/usr/bin/env python3
"""
Simple test to verify API key works with basic REST API
(bypasses gRPC/SSL issues)
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("=" * 70)
print("Testing Gemini API with REST (bypassing gRPC)")
print("=" * 70)

print(f"\nAPI Key: {api_key[:15]}...{api_key[-4:]}")

# Use REST API instead of gRPC
url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={api_key}"

data = {
    "model": "models/embedding-001",
    "content": {
        "parts": [{
            "text": "What is the status of my order?"
        }]
    }
}

print("\nSending request to Gemini API...")
try:
    response = requests.post(url, json=data, timeout=30)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        embedding = result.get('embedding', {}).get('values', [])
        print(f"✓ SUCCESS! Embedding generated")
        print(f"  - Dimension: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
        print("\n✓ Your API key is valid and working!")
    else:
        print(f"✗ ERROR: {response.status_code}")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"✗ ERROR: {e}")

print("=" * 70)
