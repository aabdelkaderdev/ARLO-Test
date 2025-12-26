"""FastAPI application entry point for ARLO microservice."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import gradio as gr

from app.api.routes import router as api_router
from app.web.gradio_app import gradio_app


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
    print("Web UI: http://localhost:11433/")
    print("API Docs: http://localhost:11433/api/docs")
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
    docs_url="/api/docs",
    redoc_url="/api/redoc",
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

# Mount Gradio app at root
app = gr.mount_gradio_app(app, gradio_app, path="/")


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
