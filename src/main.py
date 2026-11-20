#!/usr/bin/env python3
"""CLI entry point for the Research & Planning Agent."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langgraph.graph import Graph

from .State import ResearchState, ResearchReport
from .graph import create_research_graph


def get_graph() -> Graph:
    """Get or create the research graph."""
    return create_research_graph()


def run_research(question: str, output_file: str = None, parallel: bool = False) -> dict:
    """Run a research query and return results."""
    graph = get_graph()

    # Initial state
    initial_state = ResearchState(question=question)

    # Thread ID for checkpointing
    config = {"configurable": {"thread_id": f"research-{datetime.now().isoformat()}"}}

    # Run the graph
    result = graph.invoke(initial_state.model_dump(), config)

    # Extract the report if available
    final_state = ResearchState(**result)

    output = {
        "question": question,
        "completed": final_state.completed,
        "steps_count": len(final_state.steps),
        "steps_completed": sum(1 for s in final_state.steps if s.completed),
        "error": final_state.error
    }

    if final_state.report:
        output["report"] = final_state.report.to_dict() if final_state.report else None

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        output["output_file"] = output_file

    return output


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Research & Planning Agent - Decompose questions and conduct research"
    )
    parser.add_argument(
        "question",
        help="The research question or planning goal to investigate"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path for the JSON report"
    )
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Use parallel research mode"
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output as JSON only (no other text)"
    )

    args = parser.parse_args()

    # Check API key
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not set. Set it in your environment or .env file.")
        sys.exit(1)

    try:
        result = run_research(args.question, args.output, args.parallel)

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\n{'='*60}")
            print(f"Research Results")
            print(f"{'='*60}")
            print(f"\nQuestion: {result['question']}")
            print(f"Completed: {result['completed']}")
            print(f"Steps: {result['steps_completed']}/{result['steps_count']} completed")

            if result.get('error'):
                print(f"\nError: {result['error']}")

            if result.get('report'):
                report = result['report']
                print(f"\n{'-'*60}")
                print(f"REPORT: {report['title']}")
                print(f"{'-'*60}")
                print(f"\nSummary:\n{report['summary']}\n")

                print("Sections:")
                for section in report.get('sections', []):
                    print(f"\n  {section.get('title', 'Untitled')}")
                    print(f"  {section.get('content', '')[:300]}...")

                if report.get('recommendations'):
                    print("\nRecommendations:")
                    for i, rec in enumerate(report['recommendations'], 1):
                        print(f"  {i}. {rec}")

                if report.get('citations'):
                    print("\nCitations:")
                    for cite in report['citations']:
                        print(f"  - {cite.get('title', '')}: {cite.get('url', '')}")

            if args.output:
                print(f"\nReport saved to: {args.output}")

    except Exception as e:
        print(f"Error during research: {e}")
        if not args.json:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()