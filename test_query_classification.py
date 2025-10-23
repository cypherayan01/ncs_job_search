"""
Test script for Azure OpenAI Query Classification Layer

This script tests the intelligent query classification system that categorizes user queries into:
1. skill_only - User searching for jobs based on skills/technologies
2. location_only - User searching for jobs in a specific location
3. skill_location - User searching for jobs with both skills AND location
4. general - General conversation or unclear intent
"""

import asyncio
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import re
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Azure OpenAI client
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
gpt_deployment = os.getenv("AZURE_GPT_DEPLOYMENT", "gpt-4")


async def classify_query(user_query: str) -> dict:
    """Classify user query using Azure OpenAI"""

    prompt = f"""
You are an intelligent job search query analyzer. Analyze the user's query and extract job search intent.

User Query: "{user_query}"

Your task:
1. Determine the query type:
   - "skill_only": User is searching for jobs based on skills/technologies only
   - "location_only": User is searching for jobs in a specific location only
   - "skill_location": User is searching for jobs with both skills AND location
   - "general": General conversation, greetings, or unclear intent

2. Extract skills/technologies mentioned (programming languages, frameworks, tools, job roles, etc.)
   Examples: Java, Python, React, Data Analyst, Machine Learning, etc.

3. Extract location if mentioned (city, state, region)
   Examples: Mumbai, Bangalore, Maharashtra, etc.

4. Provide confidence score (0.0 to 1.0) for your classification

Return ONLY valid JSON in this exact format:
{{
  "query_type": "skill_only" | "location_only" | "skill_location" | "general",
  "skills": ["skill1", "skill2"],
  "location": "location_name" or null,
  "confidence": 0.95
}}

Examples:
- "Hey, I am a Java Developer. Can you find any job openings for me?"
  → {{"query_type": "skill_only", "skills": ["Java"], "location": null, "confidence": 0.95}}

- "Show me jobs in Mumbai"
  → {{"query_type": "location_only", "skills": [], "location": "Mumbai", "confidence": 0.98}}

- "I need Python developer jobs in Bangalore"
  → {{"query_type": "skill_location", "skills": ["Python"], "location": "Bangalore", "confidence": 0.97}}

- "Hello, how are you?"
  → {{"query_type": "general", "skills": [], "location": null, "confidence": 0.99}}
"""

    try:
        logger.info(f"Classifying: {user_query}")

        response = azure_client.chat.completions.create(
            model=gpt_deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a job search query analyzer. Return ONLY valid JSON. No explanation text. No markdown."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()

        # Clean response
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'\s*```', '', content)
        content = content.strip()

        result = json.loads(content)
        return result

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return {
            'query_type': 'general',
            'skills': [],
            'location': None,
            'confidence': 0.0,
            'error': str(e)
        }


async def run_tests():
    """Run comprehensive tests for query classification"""

    test_queries = [
        # Skill-only queries
        "Hey, I am a Java Developer. Can you find any job openings for me?",
        "I'm a Python programmer looking for opportunities",
        "I know React and Node.js, any jobs available?",
        "Data Analyst positions available?",
        "Looking for Machine Learning engineer roles",

        # Location-only queries
        "Show me jobs in Mumbai",
        "Any openings in Bangalore?",
        "I want to work in Delhi",
        "Jobs available in Maharashtra?",

        # Skill + Location queries
        "I need Python developer jobs in Bangalore",
        "Java jobs in Mumbai",
        "Looking for Data Entry positions in Delhi",
        "React developer openings in Pune",
        "Machine Learning jobs in Hyderabad",

        # General queries
        "Hello",
        "How are you?",
        "What can you do?",
        "Thanks for your help",
    ]

    print("\n" + "="*80)
    print("AZURE OPENAI QUERY CLASSIFICATION TEST RESULTS")
    print("="*80 + "\n")

    results = {
        'skill_only': [],
        'location_only': [],
        'skill_location': [],
        'general': []
    }

    for query in test_queries:
        result = await classify_query(query)
        query_type = result.get('query_type', 'general')
        results[query_type].append({
            'query': query,
            'result': result
        })

        # Print result
        print(f"Query: {query}")
        print(f"  Type: {result.get('query_type', 'N/A')}")
        print(f"  Skills: {result.get('skills', [])}")
        print(f"  Location: {result.get('location', 'None')}")
        print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        print()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Skill-only queries: {len(results['skill_only'])}")
    print(f"Location-only queries: {len(results['location_only'])}")
    print(f"Skill+Location queries: {len(results['skill_location'])}")
    print(f"General queries: {len(results['general'])}")
    print(f"Total queries tested: {len(test_queries)}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Check if Azure OpenAI is configured
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("\n❌ ERROR: AZURE_OPENAI_API_KEY not found in environment variables")
        print("Please set up your .env file with Azure OpenAI credentials\n")
        exit(1)

    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("\n❌ ERROR: AZURE_OPENAI_ENDPOINT not found in environment variables")
        print("Please set up your .env file with Azure OpenAI endpoint\n")
        exit(1)

    print(f"\n✓ Azure OpenAI configured")
    print(f"  Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"  Deployment: {gpt_deployment}\n")

    asyncio.run(run_tests())
