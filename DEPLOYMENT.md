# Job Search Chatbot - Deployment Guide

## 🚀 Quick Start Deployment

### Prerequisites
- Python 3.8+
- PostgreSQL database with job data
- Azure OpenAI API access
- 2GB+ RAM (for embedding models)

---

## 📋 Step-by-Step Deployment

### 1. Clone Repository (If not already done)
```bash
cd /home/user/ncs_job_search
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Spacy Model (Required)
```bash
python -m spacy download en_core_web_sm
```

### 5. Configure Environment Variables

Create a `.env` file from the template:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
nano .env
```

Required variables:
```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-actual-api-key
AZURE_GPT_DEPLOYMENT=gpt-4

# Database Configuration
DATABASE_URL=postgresql://username:password@host:5432/database_name
```

### 6. Verify Database Connection

Make sure your PostgreSQL database is running and accessible:
```bash
psql -h localhost -U username -d ncs_job_search -c "SELECT COUNT(*) FROM vacancies_summary;"
```

Expected table schema:
- `vacancies_summary` - Main job table with columns:
  - ncspjobid, title, keywords, description
  - organization_name, statename, districtname
  - industryname, sectorname, functionalareaname
  - aveexp, avewage, numberofopenings
  - date, gendercode, highestqualification

### 7. Initialize FAISS Index (First Time Only)

The app will automatically build the FAISS index on first startup. This may take 5-10 minutes depending on the number of jobs in your database.

### 8. Start the Application

**Option A: Development Mode (with auto-reload)**
```bash
python app.py
```

**Option B: Production Mode (using uvicorn directly)**
```bash
uvicorn app:app --host 0.0.0.0 --port 8888 --workers 4
```

**Option C: Background Process**
```bash
nohup python app.py > app.log 2>&1 &
```

### 9. Verify Deployment

Check if the app is running:
```bash
curl http://localhost:8888/
```

Expected response:
```json
{
  "message": "NCS Job Search API with Enhanced Chatbot",
  "status": "running",
  "version": "2.0"
}
```

---

## 🐳 Docker Deployment (Recommended for Production)

### Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy application files
COPY . .

# Expose port
EXPOSE 8888

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8888"]
```

### Create docker-compose.yml
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8888:8888"
    environment:
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - AZURE_GPT_DEPLOYMENT=${AZURE_GPT_DEPLOYMENT}
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - ./faiss_job_index.bin:/app/faiss_job_index.bin
      - ./job_metadata.pkl:/app/job_metadata.pkl
    restart: unless-stopped
    depends_on:
      - db

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=ncs_job_search
      - POSTGRES_USER=ncsuser
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### Deploy with Docker
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop
docker-compose down
```

---

## 🔧 Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| AZURE_OPENAI_ENDPOINT | Yes | - | Azure OpenAI endpoint URL |
| AZURE_OPENAI_API_KEY | Yes | - | Azure OpenAI API key |
| AZURE_GPT_DEPLOYMENT | Yes | - | GPT deployment name |
| DATABASE_URL | Yes | - | PostgreSQL connection string |
| EMBEDDING_MODEL | No | all-MiniLM-L6-v2 | Sentence transformer model |
| APP_HOST | No | 0.0.0.0 | Application host |
| APP_PORT | No | 8888 | Application port |
| LOG_LEVEL | No | info | Logging level |

### Port Configuration

Default port: `8888`

To change the port, edit `app.py` line 3825:
```python
port=8888,  # Change to your desired port
```

---

## 📊 Health Checks & Monitoring

### Health Check Endpoint
```bash
curl http://localhost:8888/health
```

### Monitor Application Logs
```bash
# If running with nohup
tail -f app.log

# If running with docker-compose
docker-compose logs -f app

# If running directly
# Logs will appear in console
```

### Check Database Connection
```bash
curl http://localhost:8888/db-status
```

---

## 🔒 Security Configuration

### 1. CORS Configuration (Already configured in app.py)
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, restrict origins:
```python
allow_origins=["https://yourdomain.com", "https://app.yourdomain.com"],
```

### 2. Environment Variables
Never commit `.env` file. Add to `.gitignore`:
```bash
echo ".env" >> .gitignore
```

### 3. API Key Protection
Consider adding API key authentication for production.

---

## 🚦 API Endpoints

### Main Endpoints

#### 1. Health Check
```bash
GET /
Response: {"message": "NCS Job Search API...", "status": "running"}
```

#### 2. Chat Interface
```bash
POST /chat
Content-Type: application/json

{
  "message": "Data Entry jobs in Mumbai",
  "chat_phase": "intro",
  "user_profile": {},
  "conversation_history": []
}
```

#### 3. CV Upload
```bash
POST /upload_cv
Content-Type: multipart/form-data

Form Data:
- file: [CV file - PDF/DOC/DOCX]
```

#### 4. Job Search (Direct)
```bash
POST /search_jobs
Content-Type: application/json

{
  "skills": ["Python", "React"],
  "limit": 20
}
```

#### 5. Location-Based Search
```bash
POST /search_jobs_by_location
Content-Type: application/json

{
  "location": "Mumbai",
  "skills": ["Data Entry"],
  "limit": 50
}
```

---

## 📈 Performance Optimization

### 1. Database Indexing
Ensure indexes on frequently queried columns:
```sql
CREATE INDEX idx_vacancies_state ON vacancies_summary(statename);
CREATE INDEX idx_vacancies_district ON vacancies_summary(districtname);
CREATE INDEX idx_vacancies_keywords ON vacancies_summary USING gin(to_tsvector('english', keywords));
CREATE INDEX idx_vacancies_title ON vacancies_summary USING gin(to_tsvector('english', title));
```

### 2. FAISS Index Caching
The app saves FAISS index to disk after building:
- `faiss_job_index.bin` - Vector index
- `job_metadata.pkl` - Job metadata

These files are loaded on subsequent startups for faster initialization.

### 3. Connection Pooling
The app uses asyncpg with connection pooling for efficient database access.

### 4. Concurrent Workers
For production, run with multiple workers:
```bash
uvicorn app:app --host 0.0.0.0 --port 8888 --workers 4
```

---

## 🐛 Troubleshooting

### Issue: "Failed to initialize Azure OpenAI client"
**Solution**: Check your `.env` file:
- Verify AZURE_OPENAI_ENDPOINT is correct
- Verify AZURE_OPENAI_API_KEY is valid
- Ensure endpoint ends with `/`

### Issue: "Connection refused" to database
**Solution**:
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Start PostgreSQL
sudo systemctl start postgresql

# Verify connection string in .env
```

### Issue: "No module named 'cv'"
**Solution**: Ensure `cv.py` is in the same directory as `app.py`

### Issue: FAISS index build takes too long
**Solution**:
- This is normal for first startup with large datasets
- Index is cached after first build
- Consider running index build separately during off-hours

### Issue: High memory usage
**Solution**:
- Reduce embedding model size
- Use CPU version of FAISS (`faiss-cpu` in requirements.txt)
- Limit concurrent workers

### Issue: Port 8888 already in use
**Solution**:
```bash
# Find process using port 8888
lsof -i :8888

# Kill the process
kill -9 <PID>

# Or change port in app.py
```

---

## 📦 Production Deployment Checklist

- [ ] Environment variables configured in `.env`
- [ ] Database connection tested and working
- [ ] FAISS index built and cached
- [ ] Spacy model downloaded (`en_core_web_sm`)
- [ ] CORS origins restricted to production domains
- [ ] Logs configured for production monitoring
- [ ] SSL/TLS certificate configured (if using HTTPS)
- [ ] Firewall rules configured for port 8888
- [ ] Health check endpoint tested
- [ ] Load testing completed
- [ ] Backup strategy for FAISS index files
- [ ] Database backup configured
- [ ] Monitoring and alerting setup
- [ ] API documentation provided to frontend team

---

## 🔄 Updates & Maintenance

### Update Application Code
```bash
# Pull latest changes
git pull origin main

# Restart application
# If running with nohup
pkill -f "python app.py"
nohup python app.py > app.log 2>&1 &

# If running with docker-compose
docker-compose down
docker-compose build
docker-compose up -d
```

### Rebuild FAISS Index
If job data changes significantly:
```bash
# Delete existing index files
rm faiss_job_index.bin job_metadata.pkl

# Restart app - it will rebuild automatically
python app.py
```

### Database Maintenance
```bash
# Vacuum and analyze
psql -h localhost -U username -d ncs_job_search -c "VACUUM ANALYZE vacancies_summary;"

# Reindex
psql -h localhost -U username -d ncs_job_search -c "REINDEX TABLE vacancies_summary;"
```

---

## 📞 Support & Documentation

- **CHATBOT_GUIDE.md** - Complete chatbot usage guide
- **QUERY_EXAMPLES.md** - All supported query patterns
- **Test Scripts**:
  - `test_query_parsing.py` - Basic tests
  - `test_comprehensive_queries.py` - Full test suite

---

## 🎯 Quick Commands Reference

```bash
# Start app
python app.py

# Start with production settings
uvicorn app:app --host 0.0.0.0 --port 8888 --workers 4

# Run in background
nohup python app.py > app.log 2>&1 &

# View logs
tail -f app.log

# Stop background process
pkill -f "python app.py"

# Test health
curl http://localhost:8888/

# Test chat
curl -X POST http://localhost:8888/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Data Entry jobs in Mumbai", "chat_phase": "intro"}'
```

---

**Deployment Complete!** 🚀

Your job search chatbot is now running on `http://localhost:8888`
