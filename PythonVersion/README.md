# ARLO Python Microservice

A dockerized Python microservice that converts software requirements into architectural decisions using LLM (Ollama) and ILP optimization.

## Quick Start

### Run Locally (without Docker)
```bash
cd PythonVersion
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### Run with Docker
```bash
cp .env.example .env
# Edit .env with your defaults (optional - can override per-request)
docker-compose up --build
```

Access the application at `http://localhost:11433`

## Web Interface

The application includes a web-based user interface for easy interaction:

1. **Open** `http://localhost:11433` in your browser
2. **Upload** a `.txt` file with requirements (one requirement per line)
3. **Configure** optimization settings (ILP/Greedy, quality weights mode)
4. **Click** "Analyze Requirements"
5. **View** results in structured tables
6. **Download** a PDF report

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check with Ollama connectivity |
| GET | `/api/config` | View current environment config |
| GET | `/api/matrix` | View quality-pattern matrix |
| POST | `/api/analyze` | Analyze requirements and get decisions |
| GET | `/api/docs` | Interactive API documentation |

## Example Request

**Basic (uses env config):**
```bash
curl -X POST http://localhost:11433/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "requirements": [
      "The system shall support 10000 concurrent users",
      "All data must be encrypted at rest and in transit"
    ]
  }'
```

**With Custom Ollama Config (per-request):**
```bash
curl -X POST http://localhost:11433/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "requirements": [
      "The system shall support 10000 concurrent users"
    ],
    "settings": {
      "optimization_strategy": "ILP",
      "ollama": {
        "base_url": "http://192.168.1.100:11434",
        "model": "llama3.1",
        "embed_model": "nomic-embed-text"
      }
    }
  }'
```

## Configuration

### Environment Variables (Defaults)
| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Chat model name | `llama3.1` |
| `OLLAMA_EMBED_MODEL` | Embedding model | Same as chat model |

### Per-Request Override
You can override Ollama settings in each API request by including the `ollama` object in `settings`:
```json
{
  "settings": {
    "ollama": {
      "base_url": "http://your-server:11434",
      "model": "mistral",
      "embed_model": "nomic-embed-text"
    }
  }
}
```

## Interactive Container Access
To modify environment variables in a running container:
```bash
docker exec -it <container_id> bash
export OLLAMA_MODEL=mistral
# Changes apply to new requests
```
