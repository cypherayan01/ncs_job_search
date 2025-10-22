#!/usr/bin/env python3
"""
Test script to validate query parsing for job search chatbot
"""

import re
from typing import Dict, Any, List

# Test the _parse_query_intent logic
def parse_query_intent(message: str) -> Dict[str, Any]:
    """Parse user query to extract location, skills, and intent"""
    intent = {
        'has_location': False,
        'has_skills': False,
        'location': None,
        'skills': [],
        'job_type': None,
        'query_type': 'general'
    }

    # Known cities and states
    known_locations = [
        'mumbai', 'delhi', 'bangalore', 'bengaluru', 'chennai', 'hyderabad',
        'pune', 'kolkata', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur',
        'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri', 'patna',
        'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut',
        'rajkot', 'kalyan', 'vasai', 'varanasi', 'srinagar', 'aurangabad', 'dhanbad',
        'amritsar', 'navi mumbai', 'allahabad', 'ranchi', 'howrah', 'coimbatore',
        'maharashtra', 'karnataka', 'tamil nadu', 'delhi', 'telangana', 'andhra pradesh',
        'west bengal', 'gujarat', 'rajasthan', 'uttar pradesh', 'madhya pradesh'
    ]

    # Enhanced patterns
    combined_patterns = [
        # Pattern: "jobs in [location] on/for [skill]"
        r'(?:jobs?|openings?|positions?|vacancies?)\s+in\s+([a-zA-Z\s]+)\s+(?:on|for|in|with|of)\s+([a-zA-Z\s]+)',
        # Pattern: "[skill] jobs in [location]"
        r'([a-zA-Z\s]+)\s+(?:jobs?|openings?|positions?|vacancies?)\s+in\s+([a-zA-Z\s]+)',
        # Pattern: "show me [skill] in [location]"
        r'(?:show|find|get|search)\s+(?:me\s+)?([a-zA-Z\s]+)\s+(?:jobs?|positions?)?\s+in\s+([a-zA-Z\s]+)',
    ]

    # Check combined patterns
    for pattern_idx, pattern in enumerate(combined_patterns):
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            part1 = match.group(1).strip()
            part2 = match.group(2).strip()

            # Clean up common words
            part1_clean = re.sub(r'\b(the|all|any|some)\b', '', part1, flags=re.IGNORECASE).strip()
            part2_clean = re.sub(r'\b(the|all|any|some)\b', '', part2, flags=re.IGNORECASE).strip()

            # Check which part is location
            part2_is_location = part2_clean.lower() in known_locations
            part1_is_location = part1_clean.lower() in known_locations

            if part2_is_location:
                intent['location'] = part2_clean
                intent['skills'] = [part1_clean]  # Simplified
            elif part1_is_location:
                intent['location'] = part1_clean
                intent['skills'] = [part2_clean]  # Simplified
            else:
                # Pattern-specific logic
                if pattern_idx == 0:
                    intent['location'] = part1_clean
                    intent['skills'] = [part2_clean]
                else:
                    intent['skills'] = [part1_clean]
                    intent['location'] = part2_clean

            if intent['location'] or intent['skills']:
                intent['has_location'] = bool(intent['location'])
                intent['has_skills'] = bool(intent['skills'])
                intent['query_type'] = 'combined'
                return intent

    return intent


# Test cases
test_cases = [
    "Data Entry jobs in Mumbai",
    "Show me Python jobs in Bangalore",
    "jobs in Delhi on Customer Service",
    "React developer positions in Chennai",
    "Find me nursing jobs in Pune",
    "Customer Service positions in Hyderabad",
    "jobs in Mumbai on Data Entry",
    "Sales jobs in Gujarat",
    "Python developer jobs in Karnataka",
    "Show me Data Entry in Mumbai",
]

print("=" * 80)
print("QUERY PARSING TEST")
print("=" * 80)

for query in test_cases:
    result = parse_query_intent(query)
    print(f"\nQuery: '{query}'")
    print(f"  Location: {result['location']}")
    print(f"  Skills: {result['skills']}")
    print(f"  Type: {result['query_type']}")
    print(f"  ✓ SUCCESS" if result['has_location'] and result['has_skills'] else f"  ✗ INCOMPLETE")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
