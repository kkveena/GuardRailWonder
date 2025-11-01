# Deployment Guide

## Prerequisites

- Python 3.9+
- Gemini API key
- pip or poetry for package management

## Installation

### Option 1: pip (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/GuardRailWonder.git
cd GuardRailWonder

# Install dependencies
pip install -r requirements.txt

# Or with optional dependencies
pip install -e ".[mcp,dev]"
```

### Option 2: Docker (Coming Soon)

```bash
docker build -t guardrail-wonder .
docker run -e GEMINI_API_KEY=your_key guardrail-wonder
```

## Configuration

### 1. Environment Variables

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (with defaults)
GUARDRAIL_THRESHOLD_HIGH=0.8
GUARDRAIL_THRESHOLD_MEDIUM=0.5
GUARDRAIL_LOG_REJECTIONS=true
GUARDRAIL_LOG_FILE=logs/guardrail.log
EMBEDDING_MODEL=models/embedding-001
EMBEDDING_DIMENSION=768
CACHE_EMBEDDINGS=true
CACHE_DIR=data/embeddings
```

### 2. Predefined Prompts

Edit `config/predefined_prompts.json` with your business-specific prompts:

```json
{
  "prompts": [
    {
      "id": "your_prompt_1",
      "template": "Example prompt text",
      "category": "your_category",
      "description": "Description of what this prompt represents"
    }
  ],
  "metadata": {
    "version": "1.0",
    "last_updated": "2025-11-01",
    "total_prompts": 7
  }
}
```

**Guidelines for predefined prompts**:
- Use 7-20 prompts for optimal performance
- Cover all major business use cases
- Use clear, representative examples
- Group by category for better organization
- Update regularly based on rejection logs

### 3. Validate Configuration

```bash
python -c "from guardrails.utils import validate_predefined_prompts; validate_predefined_prompts('config/predefined_prompts.json')"
```

## Initialization

### Pre-compute Embeddings

Before using in production, pre-compute embeddings:

```python
from guardrails import PromptGuardrail, GuardrailConfig
from guardrails.embeddings import GeminiEmbeddingProvider
from guardrails.utils import load_config_from_env

config = load_config_from_env()
provider = GeminiEmbeddingProvider(model=config.embedding_model)
guardrail = PromptGuardrail(
    config=config,
    embedding_provider=provider,
    predefined_prompts_path="config/predefined_prompts.json"
)

# This will cache embeddings for future use
guardrail.initialize_predefined_embeddings()
```

## Production Deployment

### Option 1: As a Library

Integrate directly into your application:

```python
from guardrails import PromptGuardrail, GuardrailConfig
from guardrails.embeddings import GeminiEmbeddingProvider
import os

# Initialize once at startup
config = GuardrailConfig(
    threshold_high=float(os.getenv("GUARDRAIL_THRESHOLD_HIGH", "0.8")),
    threshold_medium=float(os.getenv("GUARDRAIL_THRESHOLD_MEDIUM", "0.5")),
)
guardrail = PromptGuardrail(
    config=config,
    embedding_provider=GeminiEmbeddingProvider(),
    predefined_prompts_path="config/predefined_prompts.json"
)
guardrail.initialize_predefined_embeddings()

# Use for each request
def handle_user_prompt(prompt: str):
    result = guardrail.evaluate(prompt)

    if result.decision == "rejected":
        return {"error": result.message}

    # Route to LLM
    llm_response = your_llm.generate(prompt)
    return {"response": llm_response}
```

### Option 2: As an MCP Tool

```python
from guardrails.mcp_tool import initialize_tool, validate_prompt

# Initialize at startup
initialize_tool(
    config_path=None,  # Uses environment variables
    predefined_prompts_path="config/predefined_prompts.json"
)

# Use as a tool
result = validate_prompt("User's prompt here")
if result['approved']:
    # Proceed with LLM
    pass
else:
    # Handle rejection
    pass
```

### Option 3: As a Microservice (REST API)

Create `app.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from guardrails import PromptGuardrail, GuardrailConfig
from guardrails.embeddings import GeminiEmbeddingProvider
from guardrails.utils import load_config_from_env

app = FastAPI()

# Initialize guardrail
config = load_config_from_env()
guardrail = PromptGuardrail(
    config=config,
    embedding_provider=GeminiEmbeddingProvider(),
    predefined_prompts_path="config/predefined_prompts.json"
)
guardrail.initialize_predefined_embeddings()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/validate")
async def validate(request: PromptRequest):
    result = guardrail.evaluate(request.prompt)
    return {
        "approved": result.decision != "rejected",
        "decision": result.decision,
        "similarity_score": result.similarity_score,
        "matched_category": result.matched_prompt_category,
        "message": result.message,
    }

@app.get("/categories")
async def get_categories():
    return {
        "categories": guardrail.get_supported_categories(),
        "templates": guardrail.get_predefined_templates(),
    }
```

Run with:
```bash
pip install fastapi uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Monitoring

### 1. Log Monitoring

Rejection logs are in `logs/guardrail.log`:

```bash
# View recent rejections
tail -f logs/guardrail.log | grep REJECTION

# Analyze rejection patterns
python examples/analyze_logs.py
```

### 2. Metrics

Track these metrics:
- **Approval rate**: % of prompts approved
- **Rejection rate**: % of prompts rejected
- **Average similarity score**: For approved prompts
- **Latency**: Time to evaluate

Example monitoring script:

```python
from guardrails.utils import analyze_rejection_logs

analysis = analyze_rejection_logs("logs/guardrail.log")
print(f"Total rejections: {analysis['total_rejections']}")
print(f"Avg score: {analysis['average_similarity_score']:.3f}")
```

### 3. Alerts

Set up alerts for:
- High rejection rate (>20%)
- Low similarity scores (<0.6) for approved prompts
- API errors from Gemini

## Maintenance

### Regular Tasks

**Weekly**:
1. Review rejection logs
2. Identify common rejection patterns
3. Consider adding new predefined prompts

**Monthly**:
1. Analyze approval/rejection metrics
2. Tune thresholds if needed
3. Update predefined prompts
4. Re-compute embeddings cache

**Quarterly**:
1. Review overall guardrail effectiveness
2. Consider architecture changes for scale
3. Update to latest Gemini models if available

### Updating Predefined Prompts

When updating prompts:

```bash
# 1. Edit config/predefined_prompts.json
vim config/predefined_prompts.json

# 2. Validate
python -c "from guardrails.utils import validate_predefined_prompts; validate_predefined_prompts('config/predefined_prompts.json')"

# 3. Clear embedding cache
rm -rf data/embeddings/predefined_embeddings.npy

# 4. Re-initialize (will recompute embeddings)
python -c "from guardrails import PromptGuardrail, GuardrailConfig; from guardrails.embeddings import GeminiEmbeddingProvider; from guardrails.utils import load_config_from_env; config = load_config_from_env(); g = PromptGuardrail(config, GeminiEmbeddingProvider(), 'config/predefined_prompts.json'); g.initialize_predefined_embeddings()"

# 5. Restart application
```

### Threshold Tuning

To find optimal thresholds:

```python
from guardrails import PromptGuardrail, GuardrailConfig
from guardrails.embeddings import GeminiEmbeddingProvider

test_prompts = [
    ("What is my order status?", True),  # Should be approved
    ("Tell me a joke", False),  # Should be rejected
    # ... add more test cases
]

for high_threshold in [0.7, 0.75, 0.8, 0.85]:
    for med_threshold in [0.4, 0.5, 0.6]:
        config = GuardrailConfig(
            threshold_high=high_threshold,
            threshold_medium=med_threshold
        )
        guardrail = PromptGuardrail(config, GeminiEmbeddingProvider(), "config/predefined_prompts.json")
        guardrail.initialize_predefined_embeddings()

        correct = 0
        for prompt, should_approve in test_prompts:
            result = guardrail.evaluate(prompt)
            approved = result.decision != "rejected"
            if approved == should_approve:
                correct += 1

        accuracy = correct / len(test_prompts)
        print(f"High={high_threshold}, Med={med_threshold}: {accuracy:.2%}")
```

## Troubleshooting

### Common Issues

**1. "GEMINI_API_KEY not found"**
- Ensure `.env` file exists
- Check API key is set correctly
- Verify environment variables are loaded

**2. "Too many rejections"**
- Lower thresholds (try 0.7 and 0.4)
- Add more predefined prompts
- Review rejection logs for patterns

**3. "Slow response times"**
- Enable embedding caching
- Pre-compute embeddings at startup
- Consider using local embedding models

**4. "High memory usage"**
- Limit embedding cache size
- Use disk-based cache instead of memory
- Consider vector database for large prompt sets

## Security

### Best Practices

1. **API Key Security**
   - Never commit API keys to version control
   - Use environment variables or secret managers
   - Rotate keys periodically

2. **Input Validation**
   - Limit prompt length (e.g., 1000 chars)
   - Sanitize inputs before embedding
   - Rate limit requests

3. **Monitoring**
   - Log all rejections
   - Alert on unusual patterns
   - Regular security audits

### Example Security Middleware

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)

# Rate limiting
@app.post("/validate")
@limiter.limit("100/minute")
async def validate(request: Request, prompt_request: PromptRequest):
    # Input validation
    if len(prompt_request.prompt) > 1000:
        raise HTTPException(400, "Prompt too long")

    # Guardrail check
    result = guardrail.evaluate(prompt_request.prompt)
    return result
```

## Scaling

### For High Traffic (>1000 req/s)

1. **Use caching**: Redis/Memcached for recent prompts
2. **Load balancing**: Multiple guardrail instances
3. **Vector database**: For large prompt sets (>100)
4. **Async processing**: Use asyncio for concurrent requests

Example with caching:

```python
import redis
from guardrails import PromptGuardrail

cache = redis.Redis(host='localhost', port=6379)
guardrail = PromptGuardrail(...)

def evaluate_with_cache(prompt: str):
    # Check cache
    cached = cache.get(f"guardrail:{prompt}")
    if cached:
        return json.loads(cached)

    # Evaluate
    result = guardrail.evaluate(prompt)

    # Cache result (TTL: 1 hour)
    cache.setex(
        f"guardrail:{prompt}",
        3600,
        json.dumps(result.dict())
    )

    return result
```

## Support

For deployment issues:
- GitHub Issues: https://github.com/yourusername/GuardRailWonder/issues
- Documentation: https://github.com/yourusername/GuardRailWonder/blob/main/README.md
