"""FastAPI application entry point for ARLO microservice."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from app.api.routes import router as api_router


# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("=" * 50)
    print("ARLO Microservice Starting...")
    print(f"Ollama URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    print(f"Ollama Model: {os.getenv('OLLAMA_MODEL', 'llama3.1')}")
    print("=" * 50)
    yield
    # Shutdown
    print("ARLO Microservice Shutting Down...")


app = FastAPI(
    title="ARLO - Architectural Requirements to Logical Optimization",
    description=(
        "A microservice that converts software requirements into architectural "
        "decisions using LLM-based parsing and ILP optimization."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ARLO Microservice",
        "version": "1.0.0",
        "description": "Architectural Requirements to Logical Optimization",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "11433"))
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
    )
