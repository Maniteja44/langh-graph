#!/usr/bin/env python3
"""Test script for the Research & Planning Agent."""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.State import ResearchState, ResearchStep, ResearchReport, Section
from src.tools import search_web


def test_tavily_search():
    """Test Tavily search integration."""
    print("Testing Tavily search...")
    try:
        result = search_web("What is AI?", max_results=3)
        print(f"  [OK] Search works: {len(result.results)} results")
        print(f"  [OK] Query: {result.query}")
        return True
    except Exception as e:
        print(f"  [FAIL] Search failed: {e}")
        return False


def test_state_models():
    """Test state model creation."""
    print("\nTesting state models...")
    try:
        state = ResearchState(
            question="Test question",
            steps=[
                ResearchStep(step_id=1, topic="Test", question="Test Q"),
                ResearchStep(step_id=2, topic="Test 2", question="Test Q2")
            ]
        )
        print(f"  [OK] State created: {len(state.steps)} steps")

        report = ResearchReport(
            title="Test Report",
            summary="Test summary",
            sections=[Section(title="Intro", content="Test content", citations=[])],
            recommendations=["Do this"],
            citations=[{"title": "Source", "url": "https://example.com"}]
        )
        print(f"  [OK] Report created: {len(report.sections)} sections")
        print(f"  [OK] to_dict(): {bool(report.to_dict())}")
        return True
    except Exception as e:
        print(f"  [FAIL] State test failed: {e}")
        return False


def test_graph():
    """Test graph creation."""
    print("\nTesting graph creation...")
    try:
        from src.graph import create_research_graph
        graph = create_research_graph()
        print(f"  [OK] Graph created with nodes: {list(graph.nodes.keys())}")
        return True
    except Exception as e:
        print(f"  [FAIL] Graph creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm():
    """Test LLM connection (requires valid API key)."""
    print("\nTesting LLM connection...")
    try:
        from src.chains import get_llm

        provider = os.getenv("LLM_PROVIDER", "anthropic")
        llm = get_llm()
        print(f"  [OK] LLM created: {type(llm).__name__}")

        # Quick test
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content="Hello")])
        print(f"  [OK] LLM responds: {resp.content[:50]}...")
        return True
    except Exception as e:
        print(f"  [WARN] LLM test: {e}")
        print("  (This is expected if API key is not configured)")
        return False


def run_full_test():
    """Run a full end-to-end test."""
    print("=" * 60)
    print("FULL TEST RUN")
    print("=" * 60)

    results = {
        "tavily": test_tavily_search(),
        "state": test_state_models(),
        "graph": test_graph(),
        "llm": test_llm()
    }

    print("\n" + "=" * 60)
    print("RESULTS:")
    for name, passed in results.items():
        if passed is True:
            status = "[PASS]"
        elif passed is False:
            status = "[FAIL]"
        else:
            status = "[SKIP]"
        print(f"  {name}: {status}")

    all_passed = all(r is True for r in results.values())
    return all_passed


if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)