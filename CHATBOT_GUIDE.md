# Job Search Chatbot - Comprehensive Guide

## Overview

This is a **robust, production-ready job search chatbot** that helps users find jobs through multiple interaction modes:

1. **CV Upload** - Upload resume for automatic skill extraction and job matching
2. **Skill-Based Search** - Tell the chatbot your skills ("I know Python and React")
3. **Location-Based Search** - Search jobs by location ("Jobs in Mumbai")
4. **Combined Search** - Search with both skills and location ("Data Entry jobs in Mumbai")

---

## How It Works

### Architecture

```
User Query → Query Parsing → Intent Detection → Job Search → Results
```

#### 1. **Query Parsing** (`_parse_query_intent`)
- Analyzes user message to extract:
  - Location (Mumbai, Delhi, Bangalore, etc.)
  - Skills (Python, Data Entry, Customer Service, etc.)
  - Job type (IT, Sales, Healthcare, etc.)
  - Query type (combined, location_only, skill_only)

#### 2. **Intent Detection** (`_is_location_query`)
- Determines if query requires location-based search
- Detects patterns like:
  - "jobs in Mumbai"
  - "Data Entry jobs in Delhi"
  - "Show me Python positions in Bangalore"

#### 3. **Job Search** (`_handle_location_job_query`)
- Searches database based on extracted parameters
- Filters by skills if provided
- Sorts by relevance, salary, or experience
- Returns top matching jobs

#### 4. **Smart Filtering** (`_filter_by_skills`)
- Matches jobs against user skills
- Uses skill variations (e.g., "Data Entry" matches "Data Processing")
- Calculates match percentage
- Returns best matches

---

## Supported Query Types

### 1. Location-Only Queries
```
User: "Jobs in Mumbai"
Bot: "Found 1,247 job openings in Mumbai!"
```

### 2. Skill-Only Queries
```
User: "I know Python and React"
Bot: "Great! I found 5 jobs matching your skills..."
```

### 3. Combined Queries (Location + Skills)
```
User: "Data Entry jobs in Mumbai"
Bot: "Found 50 Data Entry job openings in Mumbai!"

User: "Show me Python jobs in Bangalore"
Bot: "Found 150 Python job openings in Bangalore!"

User: "Customer Service positions in Delhi"
Bot: "Found 80 Customer Service job openings in Delhi!"
```

### 4. CV Upload
```
User: [Uploads CV]
Bot: "Successfully processed your CV! Found your skills: Python, React, SQL.
      Here are 10 matching jobs..."
```

---

## Key Features

### 1. **Intelligent Query Parsing**
- Detects 35+ Indian cities (Mumbai, Delhi, Bangalore, Chennai, etc.)
- Recognizes 10+ states (Maharashtra, Karnataka, Tamil Nadu, etc.)
- Handles natural language variations:
  - "Data Entry jobs in Mumbai"
  - "jobs in Mumbai on Data Entry"
  - "Show me Data Entry positions in Mumbai"
  - "Find Data Entry work in Mumbai"

### 2. **Comprehensive Skill Recognition**
Over 100+ skills across domains:
- **Tech**: Python, JavaScript, React, Java, SQL, AWS, Docker
- **BPO**: Data Entry, Voice Process, Customer Service, Chat Support
- **Finance**: Accounting, Tally, Excel, SAP, QuickBooks
- **Sales**: Sales, Digital Marketing, SEO, SEM, Social Media
- **Healthcare**: Nursing, Medical, Pharmacy
- **Others**: Teaching, Manufacturing, Logistics, HR, Legal

### 3. **Fallback Loop Prevention**
- Tracks conversation state per user
- Prevents getting stuck in repetitive responses
- After 2 failed attempts, provides specific examples
- Resets on valid interactions

### 4. **Smart Skill Matching**
- **Exact matches**: "Data Entry" in job description
- **Variations**: "Data Entry" matches "Data Processing", "Typing", "Data Operator"
- **Partial matches**: Weighted scoring for partial word matches
- **Lenient filtering**: Returns broader results if strict filtering finds too few jobs

### 5. **Advanced Filtering**
- Experience range: "Fresher", "2-5 years", "5+ years"
- Salary range: "Rs. 20k-40k", "Above 5 lakh"
- Job type: IT, Sales, Finance, Healthcare, etc.

---

## Usage Examples

### Example 1: Job Seeker with No CV
```
User: "Hello"
Bot: "I can help you find jobs in multiple ways:
      1. Upload your CV
      2. Chat with me about your skills
      3. Search by location
      Which would you prefer?"

User: "I know Data Entry"
Bot: "Great! I found 5 jobs matching your skills: Data Entry..."

User: "Show me jobs in Mumbai"
Bot: "Found 50 Data Entry job openings in Mumbai!"
```

### Example 2: Direct Combined Query
```
User: "Data Entry jobs in Mumbai"
Bot: "Found 50 Data Entry job openings in Mumbai!

     Location: Mumbai, Maharashtra
     Skills: Data Entry
     Results: Showing top 10 opportunities

     [Job listings...]"
```

### Example 3: CV Upload Flow
```
User: [Uploads CV with Python, React, SQL skills]
Bot: "Successfully processed your CV!
      Extracted 10 skills: Python, React, SQL, JavaScript...
      Found 10 matching jobs"

User: "Show me jobs in Bangalore"
Bot: "Found 150 Python, React job openings in Bangalore!"
     [Uses CV skills + Bangalore location]
```

### Example 4: Skill Addition
```
User: [After CV upload]
Bot: "Found 10 jobs based on your CV"

User: "I also know AWS and Docker"
Bot: "Great! Added AWS, Docker to your profile.
      Here are updated job matches..."
```

---

## Error Handling

### Scenario 1: Unclear Query
```
User: "okay"
Bot: "Let me help you find jobs. What skills do you have?"

User: "yes"
Bot: [After 2 fallbacks]
     "Let me help you get started! Here are specific examples:
      1. 'Show me Python jobs in Mumbai'
      2. 'Data Entry positions in Delhi'
      3. Upload your CV"
```

### Scenario 2: No Results Found
```
User: "Blockchain jobs in small town"
Bot: "No jobs found for [location].

     Try these alternatives:
     • Search in nearby cities
     • Remove specific skill requirements
     • Try broader job categories
     • Check remote work opportunities"
```

### Scenario 3: Location Not Recognized
```
User: "Jobs in unknown place"
Bot: "I'd be happy to help you find jobs by location!
      Please specify a city or state:
      • 'Jobs in Mumbai'
      • 'Show openings in Karnataka'
      • 'IT positions in Delhi'"
```

---

## Technical Implementation

### Core Functions

#### 1. `handle_chat_message(request: ChatRequest)`
Main entry point for all chat interactions. Routes to appropriate handler.

#### 2. `_parse_query_intent(message: str)`
Parses query to extract location, skills, and intent using regex patterns.

#### 3. `_is_location_query(message: str)`
Determines if query is location-based using patterns and keyword matching.

#### 4. `_handle_location_job_query(request, query_intent)`
Handles location-based searches with optional skill filtering.

#### 5. `_filter_by_skills(jobs: List[Dict], skills: List[str])`
Filters and scores jobs based on skill matching with variations.

#### 6. `_extract_skills_from_text(message: str)`
Extracts skills from text using comprehensive keyword dictionary.

#### 7. `_extract_location_from_message(message: str)`
Extracts location using regex patterns and keyword matching.

### Database Schema
```sql
Table: vacancies_summary
- ncspjobid: Unique job ID
- title: Job title
- keywords: Comma-separated skill keywords
- description: Job description
- organization_name: Company name
- statename: State (Maharashtra, Karnataka, etc.)
- districtname: City (Mumbai, Bangalore, etc.)
- industryname: Industry sector
- sectorname: Job sector
- functionalareaname: Functional area
- functionalrolename: Role name
- aveexp: Average experience required (years)
- avewage: Average salary
- numberofopenings: Number of positions
- highestqualification: Education requirement
- gendercode: Gender preference
- date: Posting date
```

---

## API Endpoints

### 1. POST `/chat`
Main chat endpoint for text-based interactions.

**Request:**
```json
{
  "message": "Data Entry jobs in Mumbai",
  "chat_phase": "profile_building",
  "user_profile": {},
  "conversation_history": []
}
```

**Response:**
```json
{
  "response": "Found 50 Data Entry job openings in Mumbai!",
  "message_type": "job_results",
  "chat_phase": "job_results",
  "jobs": [...],
  "location_searched": "Mumbai",
  "total_found": 50,
  "suggestions": [...]
}
```

### 2. POST `/upload_cv`
Upload CV for analysis and job matching.

**Request:** Multipart form-data with CV file (PDF, DOC, DOCX)

**Response:**
```json
{
  "success": true,
  "message": "Successfully processed your CV!",
  "profile": {
    "skills": ["Python", "React", "SQL"],
    "experience": [...],
    "education": [...]
  },
  "jobs": [...],
  "total_jobs_found": 10
}
```

### 3. POST `/chat_with_cv`
Continue chat after CV upload with context.

---

## Configuration

### Environment Variables
```bash
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint
AZURE_GPT_DEPLOYMENT=gpt-4
DATABASE_URL=postgresql://user:pass@host/db
```

### Skill Keywords Dictionary
Located in `EnhancedChatService.__init__()`. Add new skills as needed:
```python
self.skill_keywords = {
    'your_skill': ['skill_name', 'variation1', 'variation2'],
    ...
}
```

---

## Best Practices

### For Users:
1. Be specific: "Python jobs in Mumbai" vs "jobs"
2. Mention both skill and location: "Data Entry in Delhi"
3. Upload CV for best results
4. Use natural language - system handles variations

### For Developers:
1. Monitor logs for query patterns
2. Add new skills to keyword dictionary regularly
3. Update known_locations list for new cities
4. Test combined queries after changes
5. Keep fallback messages helpful and actionable

---

## Testing

Run the test script to validate query parsing:
```bash
python3 test_query_parsing.py
```

Expected output shows correct location and skill extraction for various queries.

---

## Troubleshooting

### Issue: "Data Entry jobs in Mumbai" not working
**Solution**: Check logs to see if pattern is matched. Verify "mumbai" is in known_locations list.

### Issue: Skills not extracted
**Solution**: Add skill variations to skill_keywords dictionary. Check _extract_skills_from_text() logic.

### Issue: Too many/few results
**Solution**: Adjust filtering threshold in _filter_by_skills(). Modify limit parameter in LocationJobRequest.

### Issue: Fallback loop
**Solution**: Check fallback_count tracking. Ensure reset on valid interactions. Review exception handling.

---

## Performance Optimization

1. **Database Indexing**: Create indexes on statename, districtname, keywords
2. **Caching**: Cache frequent location queries
3. **Batch Processing**: Use batch embedding generation for multiple skills
4. **Connection Pooling**: Use asyncpg connection pool
5. **Rate Limiting**: Implement rate limiting for API endpoints

---

## Future Enhancements

1. **Multilingual Support**: Hindi, Tamil, Telugu translations
2. **Voice Interface**: Speech-to-text for voice queries
3. **Job Alerts**: Email/SMS notifications for new matching jobs
4. **Salary Prediction**: ML model for salary estimation
5. **Application Tracking**: Track application status
6. **Company Reviews**: Integrate company review data
7. **Interview Prep**: AI-powered interview preparation tips

---

## Support

For issues or questions:
- Check logs: `tail -f logs/app.log`
- Review code: `app.py:542-1700` (EnhancedChatService class)
- Test queries: `python3 test_query_parsing.py`

---

## Version History

**v2.0** (Current)
- Fixed critical query parsing bug
- Added comprehensive location detection
- Enhanced skill matching with variations
- Implemented fallback loop prevention
- Added detailed logging

**v1.0**
- Initial release with basic chat functionality
- CV upload and analysis
- Simple location-based search

---

## License

MIT License - Feel free to use and modify for your needs.

---

**Built with Claude Code** 🤖
