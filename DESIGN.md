# GuardRail Wonder - Design Document

## Overview

GuardRail Wonder is an LLM prompt validation system that uses embedding-based similarity matching to ensure incoming prompts align with predefined business use cases.

## Problem Statement

In business applications, LLMs should only respond to queries relevant to specific use cases. Allowing arbitrary prompts can lead to:
- Off-topic responses
- Potential security risks (prompt injection)
- Poor user experience
- Wasted API costs

## Solution Approach

### Core Algorithm

The system implements a simple but effective approach:

1. **Embed incoming prompt** → 1×n vector
2. **Compare with predefined prompts** → 7×n matrix
3. **Compute cosine similarity** → 1×7 vector via matrix multiplication
4. **Apply threshold-based decision** → Approve, warn, or reject

### Mathematical Foundation

```
Given:
- Query embedding: Q ∈ ℝ^(1×n)
- Reference embeddings: R ∈ ℝ^(7×n)

Compute:
- Normalized query: Q̂ = Q / ||Q||
- Normalized references: R̂ = R / ||R||
- Similarity scores: S = Q̂ × R̂^T ∈ ℝ^(1×7)

Decision:
- max_score = argmax(S)
- if max_score >= threshold_high (0.8): APPROVE
- elif max_score >= threshold_medium (0.5): APPROVE_WITH_WARNING
- else: REJECT
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   MCP Tool Layer                         │
│  (GuardrailMCPTool - exposes as callable tool)          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 Core Guardrail System                    │
│                  (PromptGuardrail)                       │
│  • Load predefined prompts                               │
│  • Compute similarities                                  │
│  • Apply decision logic                                  │
│  • Log rejections                                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Embedding Provider Layer                    │
│         (GeminiEmbeddingProvider)                        │
│  • Generate embeddings via Gemini API                    │
│  • Compute cosine similarity                             │
│  • Cache embeddings                                      │
└─────────────────────────────────────────────────────────┘
```

### Data Models

**GuardrailConfig**
- `threshold_high`: High confidence threshold (default: 0.8)
- `threshold_medium`: Medium confidence threshold (default: 0.5)
- `log_rejections`: Whether to log rejected prompts
- `embedding_model`: Gemini embedding model to use
- `cache_embeddings`: Enable embedding cache

**PredefinedPrompt**
- `id`: Unique identifier
- `template`: Example prompt text
- `category`: Business category
- `description`: Human-readable description
- `embedding`: Pre-computed embedding vector

**GuardrailResult**
- `decision`: APPROVED | APPROVED_WITH_WARNING | REJECTED
- `similarity_score`: Max similarity score (0-1)
- `matched_prompt_id`: ID of closest match
- `message`: Human-readable explanation
- `all_scores`: All similarity scores for analysis

## Design Decisions

### 1. Tiered Thresholds

Instead of binary approve/reject, we use three tiers:

- **High (≥0.8)**: Auto-approve with confidence
- **Medium (0.5-0.8)**: Approve but log for review
- **Low (<0.5)**: Reject and suggest alternatives

**Rationale**: Provides flexibility and reduces false rejections while maintaining security.

### 2. Pre-computed Embeddings

Predefined prompt embeddings are computed once at initialization and cached.

**Rationale**:
- Reduces API calls
- Faster response times
- Predictable costs

### 3. Gemini Embeddings

Using Google's Gemini embedding-001 model.

**Rationale**:
- High quality embeddings (768 dimensions)
- Same ecosystem as Gemini 2.5 LLM
- Good semantic understanding
- Reasonable pricing

### 4. Matrix Multiplication Approach

Direct matrix multiplication for similarity computation.

**Rationale**:
- Mathematically elegant
- Efficient for small numbers of prompts (7-20)
- Easy to understand and debug
- Leverages NumPy optimizations

### 5. Logging & Monitoring

All rejections are logged with:
- Timestamp
- Original prompt
- Similarity score
- Closest match

**Rationale**: Enables continuous improvement by identifying:
- Common rejection patterns
- Missing prompt templates
- Threshold tuning needs

## Performance Considerations

### Time Complexity

For n-dimensional embeddings and k predefined prompts:
- Embedding generation: O(n) - single API call
- Similarity computation: O(k×n) - matrix multiplication
- Decision logic: O(k) - find argmax

**Total**: O(k×n) ≈ O(n) for small k

### Space Complexity

- Predefined embeddings: O(k×n)
- Cache: O(m×n) for m cached queries
- Configuration: O(1)

**Total**: O((k+m)×n)

### Scalability

**Current approach scales well for**:
- k ≤ 100 predefined prompts
- Real-time inference (<100ms)
- 1000s of requests/second

**For larger scale (k > 100)**:
- Consider approximate nearest neighbor (ANN) search
- Use vector databases (Pinecone, Weaviate, etc.)
- Implement hierarchical classification

## Security Considerations

### Prompt Injection Mitigation

The guardrail provides defense against:
1. **Off-topic queries**: Rejected by low similarity
2. **Jailbreaking attempts**: Won't match business prompts
3. **Malicious instructions**: Filtered before reaching LLM

**Limitations**:
- Sophisticated attacks that mimic business prompts may pass
- Should be combined with other security measures

### Data Privacy

- Prompts are sent to Gemini API for embedding
- Consider privacy implications
- Can use local embedding models if needed

## Limitations & Trade-offs

### Current Limitations

1. **Fixed Template Set**: Requires predefined prompts
2. **Semantic Gaps**: Novel but valid queries might be rejected
3. **Language Support**: Depends on embedding model's capabilities
4. **Threshold Sensitivity**: Requires tuning for specific domains

### Trade-offs

| Aspect | Strict (high threshold) | Permissive (low threshold) |
|--------|------------------------|---------------------------|
| False Rejections | More | Fewer |
| Security | Higher | Lower |
| User Experience | Restrictive | Flexible |
| Maintenance | More template updates | Less frequent |

## Future Enhancements

### Potential Improvements

1. **Adaptive Thresholds**: Learn optimal thresholds from usage data
2. **Prompt Clustering**: Automatically group similar queries
3. **Multi-language Support**: Detect and handle multiple languages
4. **Hierarchical Classification**: Category-level then template-level matching
5. **Federated Learning**: Learn from rejections without central data collection
6. **Fallback LLM**: Use small LLM for edge cases
7. **A/B Testing**: Compare different threshold configurations

### Integration Opportunities

- **LangChain**: As a middleware component
- **LlamaIndex**: As a query filter
- **API Gateway**: As a pre-processing step
- **Observability**: Integration with Datadog, Prometheus, etc.

## Deployment Recommendations

### Development
```bash
python examples/basic_usage.py
```

### Testing
```bash
pytest tests/ -v --cov
```

### Production

1. **Set environment variables**:
   ```bash
   export GEMINI_API_KEY=your_key
   export GUARDRAIL_THRESHOLD_HIGH=0.8
   export GUARDRAIL_THRESHOLD_MEDIUM=0.5
   ```

2. **Pre-compute embeddings**:
   ```python
   guardrail.initialize_predefined_embeddings()
   ```

3. **Monitor rejections**:
   ```bash
   python examples/analyze_logs.py
   ```

4. **Update templates monthly** based on rejection patterns

## Conclusion

The embedding-based guardrail approach provides a good balance between:
- Simplicity and effectiveness
- Performance and accuracy
- Security and usability

It's particularly well-suited for business applications with well-defined use cases and is easily extensible for future requirements.
