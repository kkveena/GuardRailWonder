#!/usr/bin/env python3
"""
Test if API key works at all (try text generation first)
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("=" * 70)
print("Testing Gemini API Key Validity")
print("=" * 70)

# Try listing models first (basic API test)
print("\n1. Testing: List available models...")
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

try:
    response = requests.get(url, timeout=10)
    print(f"   Status: {response.status_code}")

    if response.status_code == 200:
        models = response.json().get('models', [])
        print(f"   ✓ API key is valid!")
        print(f"   Available models: {len(models)}")
        for model in models[:5]:
            print(f"     - {model['name']}")
    else:
        print(f"   ✗ Error: {response.text[:200]}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Try text generation
print("\n2. Testing: Text generation...")
#url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

data = {
    "contents": [{
        "parts": [{"text": "Say hello in one word"}]
    }]
}

try:
    response = requests.post(url, json=data, timeout=10)
    print(f"   Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        text = result['candidates'][0]['content']['parts'][0]['text']
        print(f"   ✓ Text generation works!")
        print(f"   Response: {text}")
    else:
        print(f"   ✗ Error: {response.text[:200]}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Try embeddings
print("\n3. Testing: Embeddings...")
url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={api_key}"

data = {
    "model": "models/embedding-001",
    "content": {
        "parts": [{"text": "Hello world"}]
    }
}

try:
    response = requests.post(url, json=data, timeout=10)
    print(f"   Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        embedding = result.get('embedding', {}).get('values', [])
        print(f"   ✓ Embeddings work!")
        print(f"   Dimension: {len(embedding)}")
    else:
        print(f"   ✗ Error: {response.text[:200]}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 70)
