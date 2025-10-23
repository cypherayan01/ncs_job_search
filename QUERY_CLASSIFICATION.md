# Intelligent Query Classification Layer

## Overview

This job search application now includes an **Azure OpenAI-powered query classification layer** that intelligently understands natural language queries and classifies them into three main categories:

1. **Skill-based** - User searching for jobs based on skills/technologies
2. **Location-based** - User searching for jobs in a specific location
3. **Skill + Location** - User searching for jobs with both criteria

## Problem Solved

Previously, the bot couldn't understand natural language queries like:
- "Hey, I am a Java Developer. Can you find any job openings for me?"
- "I'm a Python programmer looking for opportunities"
- "I know React and Node.js, any jobs available?"

The bot relied solely on regex pattern matching which required specific formats like:
- "Java jobs in Mumbai"
- "Show me Python positions in Bangalore"

## Solution Architecture

### Two-Layer Approach

```
User Query
    ↓
┌─────────────────────────────────────┐
│  LAYER 1: Azure OpenAI Classifier   │
│  (Intelligent NLP-based)            │
│  - Confidence threshold: 0.7        │
│  - Extracts skills & locations      │
│  - Handles natural language         │
└─────────────────────────────────────┘
    ↓
    ├─── High Confidence (≥0.7) ──→ Use AI Classification
    │
    └─── Low Confidence (<0.7) ──→ Fallback to Layer 2
                                     ↓
                        ┌─────────────────────────────┐
                        │  LAYER 2: Regex Patterns    │
                        │  (Traditional matching)     │
                        │  - Pattern-based detection  │
                        │  - Keyword matching         │
                        └─────────────────────────────┘
```

## Query Classification Types

### 1. Skill-Only Queries

**Examples:**
- "Hey, I am a Java Developer. Can you find any job openings for me?"
- "I'm a Python programmer looking for opportunities"
- "I know React and Node.js, any jobs available?"
- "Data Analyst positions available?"
- "Looking for Machine Learning engineer roles"

**Response Format:**
```json
{
  "query_type": "skill_only",
  "skills": ["Java"],
  "location": null,
  "confidence": 0.95
}
```

### 2. Location-Only Queries

**Examples:**
- "Show me jobs in Mumbai"
- "Any openings in Bangalore?"
- "I want to work in Delhi"
- "Jobs available in Maharashtra?"

**Response Format:**
```json
{
  "query_type": "location_only",
  "skills": [],
  "location": "Mumbai",
  "confidence": 0.98
}
```

### 3. Skill + Location Queries

**Examples:**
- "I need Python developer jobs in Bangalore"
- "Java jobs in Mumbai"
- "Looking for Data Entry positions in Delhi"
- "React developer openings in Pune"
- "Machine Learning jobs in Hyderabad"

**Response Format:**
```json
{
  "query_type": "skill_location",
  "skills": ["Python"],
  "location": "Bangalore",
  "confidence": 0.97
}
```

### 4. General Queries

**Examples:**
- "Hello"
- "How are you?"
- "What can you do?"
- "Thanks for your help"

**Response Format:**
```json
{
  "query_type": "general",
  "skills": [],
  "location": null,
  "confidence": 0.99
}
```

## Implementation Details

### QueryClassificationService Class

Location: `app.py` (lines 3250-3391)

**Key Method:**
```python
async def classify_and_extract(user_query: str) -> Dict[str, Any]
```

**Features:**
- Uses Azure OpenAI GPT model for classification
- Temperature: 0.1 (low temperature for consistent results)
- Max tokens: 500
- Returns structured JSON with query type, skills, location, and confidence

### Integration Point

The classification layer is integrated into the `_parse_query_intent()` method in the `EnhancedChatService` class (app.py:714).

**Workflow:**
1. User query comes in
2. Call `query_classifier.classify_and_extract(message)`
3. If confidence ≥ 0.7 → Use AI classification
4. If confidence < 0.7 → Fallback to regex patterns
5. Return parsed intent with skills, location, and query type

## Configuration

### Required Environment Variables

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_GPT_DEPLOYMENT=gpt-4
```

### Setup Steps

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Update the Azure OpenAI credentials in `.env`:
   - Get your Azure OpenAI endpoint from Azure Portal
   - Get your API key from Azure Portal
   - Specify your GPT deployment name (e.g., "gpt-4", "gpt-35-turbo")

3. Verify configuration:
   ```bash
   python test_query_classification.py
   ```

## Testing

### Running Classification Tests

```bash
python test_query_classification.py
```

This test script:
- Tests 20+ different query types
- Shows classification results for each query
- Provides a summary of accuracy
- Verifies Azure OpenAI connectivity

### Expected Output

```
================================================================================
AZURE OPENAI QUERY CLASSIFICATION TEST RESULTS
================================================================================

Query: Hey, I am a Java Developer. Can you find any job openings for me?
  Type: skill_only
  Skills: ['Java']
  Location: None
  Confidence: 0.95

Query: Show me jobs in Mumbai
  Type: location_only
  Skills: []
  Location: Mumbai
  Confidence: 0.98

Query: I need Python developer jobs in Bangalore
  Type: skill_location
  Skills: ['Python']
  Location: Bangalore
  Confidence: 0.97

...

================================================================================
SUMMARY
================================================================================
Skill-only queries: 5
Location-only queries: 4
Skill+Location queries: 5
General queries: 4
Total queries tested: 18
================================================================================
```

## Benefits

### 1. Natural Language Understanding
- Users can speak naturally instead of following rigid formats
- Handles conversational queries like "Hey, I am a Java Developer..."

### 2. Improved User Experience
- No need to train users on specific query formats
- More intuitive and user-friendly

### 3. Accurate Intent Detection
- Azure OpenAI GPT understands context and intent
- Extracts multiple skills from a single query
- Handles variations in phrasing

### 4. Robust Fallback
- If AI classification fails, falls back to regex patterns
- Ensures system always works even with API issues

### 5. Confidence-Based Routing
- Only uses AI classification when confidence is high (≥0.7)
- Reduces false positives

## Code Locations

| Component | File | Line |
|-----------|------|------|
| QueryClassificationService | app.py | 3250-3391 |
| Integration in _parse_query_intent | app.py | 714-870 |
| Service Initialization | app.py | 3399 |
| Test Script | test_query_classification.py | 1-180 |
| Environment Config | .env.example | 1-9 |

## Performance Considerations

### Response Time
- Azure OpenAI API call: ~1-3 seconds
- Regex fallback: <100ms
- Total query processing: 1-3 seconds (with AI) or <100ms (regex only)

### Cost
- Azure OpenAI charges per token
- Each classification uses ~200-500 tokens
- Estimated cost: $0.001-0.003 per query (GPT-4)
- Consider using GPT-3.5-Turbo for lower costs

### Optimization Tips
1. Use GPT-3.5-Turbo instead of GPT-4 for cost savings
2. Cache common queries to avoid repeated API calls
3. Adjust confidence threshold (0.7) based on your accuracy requirements
4. Monitor API usage in Azure Portal

## Error Handling

The system handles errors gracefully:

1. **API Connection Errors**: Falls back to regex patterns
2. **Invalid JSON Response**: Logs error and returns general query type
3. **Missing Configuration**: System warns but continues with regex-only mode
4. **Rate Limiting**: Catches exception and uses fallback

## Future Enhancements

Potential improvements:

1. **Query Caching**: Cache common query classifications
2. **Multi-language Support**: Extend to support Hindi, regional languages
3. **Feedback Loop**: Learn from user corrections
4. **Batch Classification**: Process multiple queries in parallel
5. **Specialized Classifiers**: Add classifiers for salary, experience level, etc.

## Troubleshooting

### Issue: Classification always falls back to regex

**Solution:**
- Check Azure OpenAI credentials in `.env`
- Verify endpoint URL is correct
- Check API key has proper permissions
- Run `python test_query_classification.py` to verify connectivity

### Issue: Low confidence scores

**Solution:**
- Queries may be too vague or ambiguous
- Try more specific queries
- Adjust confidence threshold in `_parse_query_intent()` (line 731)

### Issue: Incorrect skill/location extraction

**Solution:**
- Update the prompt in `QueryClassificationService` (lines 3274-3314)
- Provide more examples in the prompt
- Consider using a fine-tuned model

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Run the test script to verify setup
3. Review the code comments in `app.py`
4. Check Azure OpenAI service health in Azure Portal

---

**Last Updated**: 2025-10-23
**Version**: 1.0
**Author**: NCS Job Search Team
