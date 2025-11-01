# GuardRail Wonder

LLM Prompt Guardrails using embedding-based similarity matching for business applications.

## Overview

GuardRail Wonder is a lightweight, efficient guardrail system that validates LLM prompts against predefined business use cases. It uses embedding similarity to ensure users only ask relevant questions, preventing off-topic queries and potential security issues.

## How It Works

```
User Prompt → Embed (1×n) → Compare with Predefined (7×n) → Cosine Similarity → Decision
```

1. **Incoming prompt** is embedded into a vector (1×n)
2. **Compared** with predefined prompt embeddings (7×n matrix)
3. **Cosine similarity** computed via matrix multiplication (1×7 result)
4. **Decision** made based on max similarity:
   - ≥ 0.8: ✓ Approved
   - 0.5-0.8: ⚠ Approved with warning
   - < 0.5: ✗ Rejected

## Features

- **Fast**: Pre-computed embeddings, O(n) inference time
- **Simple**: Matrix multiplication-based similarity
- **Flexible**: Tiered threshold system
- **Observable**: Comprehensive logging and monitoring
- **MCP-Ready**: Can be exposed as Model Context Protocol tool
- **Type-Safe**: Full Pydantic models with validation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GuardRailWonder.git
cd GuardRailWonder

# Install dependencies
pip install -r requirements.txt

# Or install with optional dependencies
pip install -e ".[mcp,dev]"
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Gemini API key
GEMINI_API_KEY=your_key_here
```

### Basic Usage

```python
from guardrails import PromptGuardrail, GuardrailConfig
from guardrails.embeddings import GeminiEmbeddingProvider

# Initialize
config = GuardrailConfig(threshold_high=0.8, threshold_medium=0.5)
provider = GeminiEmbeddingProvider()
guardrail = PromptGuardrail(
    config=config,
    embedding_provider=provider,
    predefined_prompts_path="config/predefined_prompts.json"
)

# Pre-compute embeddings
guardrail.initialize_predefined_embeddings()

# Validate a prompt
result = guardrail.evaluate("What is the status of my order?")

if result.decision == "approved":
    # Route to LLM
    response = your_llm.generate(prompt)
else:
    # Reject or ask for clarification
    print(result.message)
```

### MCP Tool Usage

```python
from guardrails.mcp_tool import GuardrailMCPTool

# Initialize MCP tool
tool = GuardrailMCPTool(
    predefined_prompts_path="config/predefined_prompts.json"
)

# Use as a validation tool
result = tool.validate_prompt("Where is my order?")
print(result['approved'])  # True or False

# Get supported categories
categories = tool.get_supported_categories()

# Explain rejections
explanation = tool.explain_rejection("Tell me a joke")
```

## Project Structure

```
GuardRailWonder/
├── src/guardrails/          # Main package
│   ├── core.py              # Core guardrail system
│   ├── embeddings.py        # Embedding providers
│   ├── models.py            # Pydantic models
│   ├── mcp_tool.py          # MCP tool wrapper
│   └── utils.py             # Utility functions
├── config/                  # Configuration files
│   ├── predefined_prompts.json
│   └── config.json
├── tests/                   # Comprehensive tests
├── examples/                # Usage examples
├── logs/                    # Application logs
└── data/embeddings/         # Cached embeddings
```

## Examples

### Run Basic Example
```bash
python examples/basic_usage.py
```

### Run MCP Example
```bash
python examples/mcp_usage.py
```

### Analyze Rejection Logs
```bash
python examples/analyze_logs.py
```

## Configuration

### Environment Variables

```bash
GEMINI_API_KEY=your_api_key
GUARDRAIL_THRESHOLD_HIGH=0.8
GUARDRAIL_THRESHOLD_MEDIUM=0.5
GUARDRAIL_LOG_REJECTIONS=true
GUARDRAIL_LOG_FILE=logs/guardrail.log
EMBEDDING_MODEL=models/embedding-001
EMBEDDING_DIMENSION=768
```

### Predefined Prompts

Edit `config/predefined_prompts.json`:

```json
{
  "prompts": [
    {
      "id": "order_status",
      "template": "What is the status of my order?",
      "category": "order_management",
      "description": "Customer order status inquiries"
    }
  ]
}
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_core.py -v
```

## Design Decisions

### Why Embedding Similarity?

- **Semantic understanding**: Catches variations of same intent
- **Fast inference**: Pre-computed embeddings mean quick comparisons
- **Interpretable**: Clear similarity scores aid debugging
- **Scalable**: Works well for 7-100 predefined prompts

### Why Gemini Embeddings?

- High quality 768-dimensional embeddings
- Same ecosystem as Gemini 2.5 LLM
- Good semantic understanding
- Reasonable pricing

### Why Tiered Thresholds?

- Reduces false rejections
- Provides flexibility
- Enables monitoring and continuous improvement
- Better user experience

See [DESIGN.md](DESIGN.md) for detailed design documentation.

## Performance

- **Embedding generation**: ~50-100ms (Gemini API)
- **Similarity computation**: <1ms (matrix multiplication)
- **Total latency**: ~50-100ms per request
- **Throughput**: 1000+ requests/second (with caching)

## Limitations

1. **Fixed template set**: Requires predefined prompts
2. **Novel queries**: Legitimate but unusual queries might be rejected
3. **Threshold tuning**: Requires experimentation for your domain
4. **API dependency**: Relies on Gemini API for embeddings

## Roadmap

- [ ] Support for local embedding models (no API dependency)
- [ ] Adaptive threshold learning from usage patterns
- [ ] Multi-language support
- [ ] Vector database integration for large prompt sets
- [ ] LangChain and LlamaIndex integrations
- [ ] Prometheus metrics export
- [ ] Web UI for configuration and monitoring

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use GuardRail Wonder in your research or project, please cite:

```bibtex
@software{guardrail_wonder,
  title = {GuardRail Wonder: LLM Prompt Guardrails},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/GuardRailWonder}
}
```

## Support

- Issues: https://github.com/yourusername/GuardRailWonder/issues
- Discussions: https://github.com/yourusername/GuardRailWonder/discussions

## Acknowledgments

- Built with Google Gemini embeddings
- Inspired by enterprise LLM security best practices
- Uses Pydantic for type safety
