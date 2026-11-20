"""Tools for the Research & Planning Agent."""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from tavily import TavilyClient


class SearchResult(BaseModel):
    """A single search result."""
    title: str
    url: str
    snippet: str
    score: Optional[float] = None


class WebSearchResult(BaseModel):
    """Results from a web search."""
    query: str
    results: List[SearchResult] = Field(default_factory=list)
    raw_answer: str = ""


def get_tavily_client() -> TavilyClient:
    """Get a Tavily API client."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    return TavilyClient(api_key=api_key)


def search_web(query: str, max_results: int = 5, include_answer: bool = True) -> WebSearchResult:
    """Search the web using Tavily API."""
    client = get_tavily_client()

    response = client.search(
        query=query,
        max_results=max_results,
        include_answer=include_answer,
        search_depth="basic"
    )

    results = []
    for r in response.get("results", []):
        results.append(SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("snippet", ""),
            score=r.get("score")
        ))

    output = WebSearchResult(query=query, results=results)
    if "answer" in response:
        output.raw_answer = response["answer"]

    return output


def search_tool_factory() -> callable:
    """Create a LangChain-compatible search tool."""

    def search_web_lc(query: str, max_results: int = 5) -> str:
        """
        Search the web and return formatted results.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            Formatted string with search results and citations
        """
        result = search_web(query, max_results=max_results)

        formatted = f"Search Query: {result.query}\n\n"
        formatted += f"Answer: {result.raw_answer}\n\n"
        formatted += "Results:\n"

        for i, r in enumerate(result.results, 1):
            formatted += f"\n{i}. {r.title}\n"
            formatted += f"   URL: {r.url}\n"
            formatted += f"   Snippet: {r.snippet[:500]}...\n"
            if r.score:
                formatted += f"   Score: {r.score:.2f}\n"

        return formatted

    return search_web_lc


# Create a tool for LangGraph
def create_search_tool() -> callable:
    """Create search tool instance for use in the agent graph."""
    return search_web_lc