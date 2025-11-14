import os, sys, pathlib
from dotenv import load_dotenv
import google.generativeai as genai

# Absolute load of .env
env_path = pathlib.Path(__file__).resolve().parents[1] / ".env"
print("Loading .env from:", env_path)
load_dotenv(env_path)

# Read keys
key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
print("Detected key:", key[:4] + "..." if key else None)

if not key:
    raise ValueError("API key missing, .env not loaded")

# MUST configure BEFORE making model
genai.configure(api_key=key)

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Key loaded correctly? →", bool(key))

# Test model
model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content("Say hello.")
print("Response:", response.text)
