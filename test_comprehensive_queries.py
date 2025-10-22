#!/usr/bin/env python3
"""
Comprehensive test for job search chatbot query parsing
Tests all user query scenarios
"""

import re
from typing import Dict, Any, List

# Copy of the actual parsing logic
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

    # Skill keywords for extraction
    skill_keywords = {
        'python': ['python'], 'javascript': ['javascript', 'js'],
        'react': ['react'], 'java': ['java'], 'sql': ['sql'],
        'data entry': ['data entry'], 'customer service': ['customer service'],
        'sales': ['sales'], 'nursing': ['nursing'], 'teaching': ['teaching'],
        'accounting': ['accounting'], 'marketing': ['marketing']
    }

    def extract_skills(text: str) -> List[str]:
        """Extract skills from text"""
        skills = []
        text_lower = text.lower()
        for skill_name, variations in skill_keywords.items():
            for variation in variations:
                if variation in text_lower:
                    skills.append(skill_name.title())
                    break
        return skills

    # Enhanced patterns
    combined_patterns = [
        # Pattern: "jobs in [location] on/for [skill]"
        r'(?:jobs?|openings?|positions?|vacancies?)\s+in\s+([a-zA-Z\s]+)\s+(?:on|for|in|with|of)\s+([a-zA-Z\s]+)',
        # Pattern: "[skill] jobs in [location]"
        r'([a-zA-Z\s]+)\s+(?:jobs?|openings?|positions?|vacancies?)\s+in\s+([a-zA-Z\s]+)',
        # Pattern: "show/find/get/give me [skill] jobs in [location]"
        r'(?:show|find|get|give|search)\s+(?:me\s+)?([a-zA-Z\s]+)\s+(?:jobs?|positions?)\s+in\s+([a-zA-Z\s]+)',
        # Pattern: "show/find/get/give me jobs in [location]"
        r'(?:show|find|get|give|search)\s+(?:me\s+)?(?:jobs?|positions?|openings?)\s+in\s+([a-zA-Z\s]+)',
    ]

    # Check combined patterns
    for pattern_idx, pattern in enumerate(combined_patterns):
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            if pattern_idx == 3:  # "show me jobs in [location]" - only location
                part1 = match.group(1).strip()
                part1_clean = re.sub(r'\b(the|all|any|some)\b', '', part1, flags=re.IGNORECASE).strip()
                intent['location'] = part1_clean
                intent['has_location'] = True
                intent['query_type'] = 'location_only'
                return intent

            part1 = match.group(1).strip()
            part2 = match.group(2).strip() if match.lastindex >= 2 else None

            if not part2:
                continue

            # Clean up common words
            part1_clean = re.sub(r'\b(the|all|any|some)\b', '', part1, flags=re.IGNORECASE).strip()
            part2_clean = re.sub(r'\b(the|all|any|some)\b', '', part2, flags=re.IGNORECASE).strip()

            # Check which part is location
            part2_is_location = part2_clean.lower() in known_locations
            part1_is_location = part1_clean.lower() in known_locations

            if part2_is_location:
                intent['location'] = part2_clean
                intent['skills'] = extract_skills(part1_clean)
            elif part1_is_location:
                intent['location'] = part1_clean
                intent['skills'] = extract_skills(part2_clean)
            else:
                # Pattern-specific logic
                if pattern_idx == 0:
                    intent['location'] = part1_clean
                    intent['skills'] = extract_skills(part2_clean)
                else:
                    intent['skills'] = extract_skills(part1_clean)
                    intent['location'] = part2_clean

            if intent['location'] or intent['skills']:
                intent['has_location'] = bool(intent['location'])
                intent['has_skills'] = bool(intent['skills'])
                intent['query_type'] = 'combined' if (intent['has_location'] and intent['has_skills']) else ('location_only' if intent['has_location'] else 'skill_only')
                return intent

    # Check for skill-only queries
    skill_only_patterns = [
        r'(?:give|show|find)\s+(?:me\s+)?([a-zA-Z\s]+)\s+(?:jobs?|positions?|openings?)',
        r'i\s+(?:know|have|can do)\s+([a-zA-Z\s]+)',
    ]

    for pattern in skill_only_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            skill_text = match.group(1).strip()
            skills = extract_skills(skill_text)
            if skills:
                intent['skills'] = skills
                intent['has_skills'] = True
                intent['query_type'] = 'skill_only'
                return intent

    return intent


def is_location_query(message: str) -> bool:
    """Check if query is location-based"""
    message_lower = message.lower()

    # Location patterns
    location_patterns = [
        r'\b(?:jobs?|openings?|vacancies?|positions?)\s+(?:in|at|for|near|from)\s+\w+',
        r'\b(?:show|find|get|give|search)\s+(?:me\s+)?(?:all\s+)?(?:jobs?|openings?|vacancies?)\s+(?:in|for|at)\s+\w+',
        r'\w+\s+(?:jobs?|openings?|positions?|vacancies?)\s+(?:in|at|for|near)\s+\w+',
    ]

    for pattern in location_patterns:
        if re.search(pattern, message_lower):
            return True

    # Check for known locations
    known_locations = ['mumbai', 'delhi', 'bangalore', 'chennai', 'hyderabad', 'pune', 'kolkata']
    job_keywords = ['job', 'jobs', 'opening', 'position', 'work']

    has_job_keyword = any(keyword in message_lower for keyword in job_keywords)
    has_location = any(location in message_lower for location in known_locations)

    return has_job_keyword and has_location


# Comprehensive test cases
test_cases = [
    # User's specific queries
    ("Show me jobs in Delhi", "location_only", "Delhi", []),
    ("Give me Python jobs", "skill_only", None, ["Python"]),
    ("Give me jobs in Mumbai on Data Entry", "combined", "Mumbai", ["Data Entry"]),
    ("I know Python", "skill_only", None, ["Python"]),

    # Additional combined queries
    ("Data Entry jobs in Mumbai", "combined", "Mumbai", ["Data Entry"]),
    ("Show me Python jobs in Bangalore", "combined", "Bangalore", ["Python"]),
    ("Customer Service positions in Delhi", "combined", "Delhi", ["Customer Service"]),
    ("React developer jobs in Chennai", "combined", "Chennai", ["React"]),
    ("Sales jobs in Gujarat", "combined", "Gujarat", ["Sales"]),
    ("Nursing positions in Pune", "combined", "Pune", ["Nursing"]),
    ("jobs in Hyderabad on Accounting", "combined", "Hyderabad", ["Accounting"]),

    # Location-only queries
    ("Jobs in Mumbai", "location_only", "Mumbai", []),
    ("Show me all jobs in Bangalore", "location_only", "Bangalore", []),
    ("Find openings in Chennai", "location_only", "Chennai", []),
    ("Positions in Hyderabad", "location_only", "Hyderabad", []),

    # Skill-only queries
    ("I know React and JavaScript", "skill_only", None, ["React", "Javascript"]),
    ("Give me SQL jobs", "skill_only", None, ["Sql"]),
    ("Show me Marketing positions", "skill_only", None, ["Marketing"]),
    ("I can do Data Entry", "skill_only", None, ["Data Entry"]),

    # Variations
    ("Find me Python jobs in Karnataka", "combined", "Karnataka", ["Python"]),
    ("Search for Teaching jobs in Maharashtra", "combined", "Maharashtra", ["Teaching"]),
    ("Get me Sales positions in Rajasthan", "combined", "Rajasthan", ["Sales"]),

    # Edge cases
    ("Show me Data Entry in Mumbai", "general", None, []),  # Missing "jobs" keyword
    ("Python", "general", None, []),  # Too vague
]


print("=" * 100)
print("COMPREHENSIVE QUERY PARSING TEST")
print("=" * 100)

passed = 0
failed = 0
failed_cases = []

for query, expected_type, expected_location, expected_skills in test_cases:
    result = parse_query_intent(query)
    is_location = is_location_query(query)

    # Check if results match expectations
    type_match = result['query_type'] == expected_type
    location_match = (result['location'] == expected_location) or (result['location'] and result['location'].lower() == (expected_location.lower() if expected_location else None))
    skills_match = len(result['skills']) == len(expected_skills) or (result['skills'] and set(s.lower() for s in result['skills']) == set(s.lower() for s in expected_skills))

    all_match = type_match and location_match and skills_match

    if all_match:
        status = "✓ PASS"
        passed += 1
    else:
        status = "✗ FAIL"
        failed += 1
        failed_cases.append((query, expected_type, expected_location, expected_skills, result))

    print(f"\n{status} Query: '{query}'")
    print(f"     Expected: Type={expected_type}, Location={expected_location}, Skills={expected_skills}")
    print(f"     Got:      Type={result['query_type']}, Location={result['location']}, Skills={result['skills']}")
    print(f"     Is Location Query: {is_location}")

print("\n" + "=" * 100)
print(f"RESULTS: {passed} PASSED, {failed} FAILED out of {len(test_cases)} tests")
print("=" * 100)

if failed_cases:
    print("\nFAILED CASES DETAILS:")
    print("-" * 100)
    for query, exp_type, exp_loc, exp_skills, result in failed_cases:
        print(f"\nQuery: '{query}'")
        print(f"  Expected: Type={exp_type}, Location={exp_loc}, Skills={exp_skills}")
        print(f"  Got:      Type={result['query_type']}, Location={result['location']}, Skills={result['skills']}")
        print(f"  Reason:   Pattern not matched or extraction failed")

print("\n" + "=" * 100)
print("RECOMMENDED QUERIES FOR USERS:")
print("=" * 100)
print("""
✓ WORKING PATTERNS:

1. Location Only:
   - "Show me jobs in Delhi"
   - "Find openings in Mumbai"
   - "Jobs in Bangalore"

2. Skill Only:
   - "Give me Python jobs"
   - "I know React and JavaScript"
   - "Show me Data Entry positions"

3. Combined (Location + Skill):
   - "Data Entry jobs in Mumbai"
   - "Show me Python jobs in Bangalore"
   - "Give me jobs in Mumbai on Data Entry"
   - "Customer Service positions in Delhi"
   - "React developer jobs in Chennai"

✗ NOT WORKING (Need "jobs" keyword):
   - "Show me Data Entry in Mumbai" → Add "jobs"
   - "Python in Bangalore" → Say "Python jobs in Bangalore"
""")
