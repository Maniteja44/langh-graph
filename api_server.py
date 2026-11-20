#!/usr/bin/env python3
"""FastAPI web service for the Research & Planning Agent."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load env
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.State import ResearchState
from src.main import run_research
from src.graph import create_research_graph

app = FastAPI(
    title="Research & Planning Agent API",
    description="Domain-independent AI research agent using LangGraph",
    version="1.0.0"
)

# Check for required keys
if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY not set in .env")


class ResearchRequest(BaseModel):
    """Request model for research queries."""
    question: str
    output_citations: bool = True


class ResearchResponse(BaseModel):
    """Response model for research results."""
    question: str
    completed: bool
    steps_count: int
    steps_completed: int
    report: dict = None
    duration_seconds: float = None


@app.post("/research", response_model=ResearchResponse)
async def research_endpoint(request: ResearchRequest):
    """Run a research query and return results."""
    import time
    start = time.time()

    try:
        result = run_research(request.question)
        result["duration_seconds"] = round(time.time() - start, 2)
        return ResearchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Research & Planning Agent",
        "version": "1.0.0",
        "endpoints": {
            "/research": "POST - Submit research queries",
            "/docs": "OpenAPI documentation"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "tavily_key_available": bool(os.getenv("TAVILY_API_KEY"))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)