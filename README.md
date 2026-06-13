# 🏠 Real Estate AI Assistant

An intelligent, Vietnamese-language real estate assistant featuring a **data crawling engine**, a **PostgreSQL + Qdrant RAG ingestion pipeline**, and a **ReAct (Reasoning & Action) Agent** served via **FastAPI** to an interactive **React + Leaflet Map** dashboard.

![Chatbot Interface](imgs/chatbot.png)

---

## 🗺️ System Architecture

The assistant implements a multi-layered **Agentic RAG pipeline** driven by a **ReAct** (Reasoning and Action) loop. The core agent orchestrates user requests by parsing intents, checking structured property filters, performing hybrid searches (PostgreSQL metadata + Qdrant dense vector embeddings), and calling custom tools (e.g., geospatial amenities search or mortgage calculation).

![RAG Architecture](imgs/rag.png)

### 🛠️ Technology Stack
- **AI Agent Framework:** ReAct loop implemented in Python, integrated with local LLMs (via Ollama Qwen2.5) or Google Gemini API.
- **Data & Embedding Layer:** PostgreSQL (structured metadata & geospatial cache) and Qdrant (dense vectors with `paraphrase-multilingual-MiniLM-L12-v2`).
- **Web App & UI:** FastAPI (backend with SSE streaming) and React + Leaflet Maps (frontend interactive dashboard).

---

## 🚀 Installation & Running Guide

### 1. Prerequisites
Ensure you have **Python 3.10+**, **Node.js (v18+)**, and **Docker** installed.

### 2. Environment Setup
Create a `.env` file in the project root:
```ini
# LLM Provider: "gemini" or "ollama"
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434

# If using Gemini:
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your_gemini_key
# GEMINI_MODEL=gemini-2.0-flash-lite

# Database Ports & Credentials
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=real_estate
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Database & Dependencies Setup
Start PostgreSQL and Qdrant via Docker Compose:
```bash
docker-compose up -d
```

Install Python dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

Build the React frontend assets:
```bash
cd frontend
npm install
npm run build
cd ..
```

---

## 📥 Data Ingestion Workflow

Before running the chatbot, populate the databases with crawled real estate data:

```bash
# 1. Crawl active property listings & project profiles
python crawlers/run.py --type ban --pages 5
python crawlers/run.py --type cho-thue --pages 5
python crawlers/run.py --type du-an --pages 3

# 2. Scrape news articles and guidelines
python crawlers/run.py --type tin-tuc --pages 3
python crawlers/run.py --type wiki --pages 3

# 3. Geocode addresses and pull nearby OSM POIs
python crawlers/run.py --type geocode --geocode-limit 0
python crawlers/run.py --type osm-poi

# 4. Crawl social media reviews (YouTube, Voz, TikTok)
python crawlers/run.py --type youtube --pages 3
python crawlers/run.py --type voz --pages 3
python crawlers/run.py --type tiktok --pages 3

# 5. Index datasets into PostgreSQL and Qdrant DB
python -m db.ingest --source all
```

---

## 🖥️ Running the Application

Launch the FastAPI web server:
```bash
PYTHONPATH=. .venv/bin/python -u web/server.py
```
Open your browser and navigate to `http://localhost:8000` to interact with the assistant:
*   **Chat Panel:** Watch the Agent stream its reasoning thoughts and tool execution tracks dynamically alongside final markdown answers.
*   **Interactive Map:** Explore property pins dynamically plotted on the Leaflet map client as the chatbot suggests them.
