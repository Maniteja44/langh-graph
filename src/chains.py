"""LLM chains for the Research & Planning Agent."""

import json
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from .State import PlannerOutput, SynthesisOutput, CritiqueOutput, ResearchReport, ResearchStep, Section


def get_llm(model_name: str = None, provider: str = None):
    """Get the LLM instance based on provider.

    Args:
        model_name: Model to use (overrides default per provider)
        provider: LLM provider - 'anthropic', 'openai', 'google', or 'groq'

    If provider is not specified, checks LLM_PROVIDER env var.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "anthropic")

    if provider == "anthropic":
        model = model_name or "claude-sonnet-4-20250514"
        return ChatAnthropic(model=model, temperature=0.1, max_tokens=4000)

    elif provider == "openai":
        model = model_name or "gpt-4.1-mini"
        return ChatOpenAI(model=model, temperature=0.1, max_tokens=4000)

    elif provider == "google":
        # Gemini models need "models/" prefix
        model_name = model_name or "gemini-3.5-flash"
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.1, max_tokens=4000)

    elif provider == "groq":
        from langchain_groq import ChatGroq
        model = model_name or "qwen-qia-32b-instruct"
        return ChatGroq(model=model, temperature=0.1, max_tokens=4000)

    else:
        raise ValueError(f"Unknown provider: {provider}")


# Planner prompt
PLANNER_PROMPT = """You are a research planning agent. Your job is to decompose a research question into a set of independent, actionable steps.

Question: {question}

Guidelines:
1. Break the question into 3-7 clear, independent research steps
2. Each step should be researchable via web search
3. Steps should be ordered by logical dependency (earlier steps inform later ones)
4. Each step should have a specific, answerable question

Output a JSON object with:
- "steps": array of step objects, each with "topic" and "question" fields
- "rationale": explanation of why these steps are needed

Example output format:
{{
  "steps": [
    {{"topic": "Market size", "question": "What is the current market size for X?"}},
    {{"topic": "Key players", "question": "Who are the main competitors in X?"}}
  ],
  "rationale": "These steps establish baseline facts before deeper analysis."
}}
"""


# Synthesizer prompt
SYNTHESIZER_PROMPT = """You are a research report synthesizer. Combine search results into a coherent, structured report with citations.

Question: {question}

Research Steps and Results: {research_data}

Create a comprehensive report that answers the question. Include:
1. An executive summary
2. Detailed sections answering each research topic
3. Actionable recommendations
4. Proper citations to all sources used

For each claim in your report, include the source URL. If you don't have sources for a claim, mark it as needing verification.

Output a JSON object with:
- "title": a compelling title for the report
- "summary": 3-5 sentence executive summary
- "sections": array of objects with "title", "content", and "citations" fields
- "recommendations": array of actionable recommendations
- "citations": array of objects with "title" and "url" fields
"""


# Critic prompt
CRITIC_PROMPT = """You are a critical reviewer of research reports. Check for completeness and accuracy.

Original Question: {question}

Report to Review: {report_json}

Your job:
1. Check if all sections address the original question
2. Identify any gaps in coverage
3. Check if all claims have proper citations
4. Identify sections that need more evidence
5. Provide suggestions for improvement

If the report is complete and well-cited, output is_complete: true.
If there are gaps, output is_complete: false with suggestions.

Output a JSON object with:
- "is_complete": boolean
- "missing_sections": array of section names that should be added
- "missing_citations": array of claim descriptions lacking citations
- "suggestions": array of specific improvement suggestions
- "revised_report": if significant changes needed, provide revised ResearchReport object, else null
"""


class ResearchDecomposer:
    """Decomposes research questions into steps."""

    def __init__(self, llm=None):
        if llm is None:
            llm = get_llm()
        self.llm = llm
        self._template = PLANNER_PROMPT

    def decompose(self, question: str) -> List[ResearchStep]:
        """Decompose a question into research steps."""
        messages = [
            HumanMessage(content=self._template.format(question=question))
        ]
        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)

        # Handle different response formats
        if isinstance(content, list):
            # Gemini returns a list of content parts
            content = "".join(part.get('text', '') if isinstance(part, dict) else str(part) for part in content)
        elif not isinstance(content, str):
            content = str(content)

        # Strip markdown code blocks
        content = content.strip()
        if content.startswith("```"):
            lines = content.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = '\n'.join(lines)

        try:
            data = json.loads(content)
            steps = []
            for i, step_data in enumerate(data.get("steps", []), 1):
                topic = step_data.get("topic", f"Step {i}")
                q = step_data.get("question", "")
                steps.append(ResearchStep(
                    step_id=i,
                    topic=topic,
                    question=q,
                    completed=False
                ))
            return steps
        except json.JSONDecodeError:
            return [ResearchStep(
                step_id=1,
                topic="Research",
                question=question,
                completed=False
            )]


class ReportSynthesizer:
    """Synthesizes research results into a report."""

    def __init__(self, llm=None):
        if llm is None:
            llm = get_llm()
        self.llm = llm
        self._template = SYNTHESIZER_PROMPT

    def synthesize(self, question: str, steps_results: dict) -> ResearchReport:
        """Create a report from research results."""
        research_data = []
        for step_id, step in steps_results.items():
            research_data.append({
                "step_id": step_id,
                "topic": step.get("topic", ""),
                "question": step.get("question", ""),
                "findings": step.get("result", ""),
                "sources": step.get("sources", [])
            })

        formatted_data = json.dumps(research_data, indent=2)
        content = self._template.format(
            question=question,
            research_data=formatted_data
        )

        messages = [HumanMessage(content=content)]
        result = self.llm.invoke(messages)
        output = result.content if hasattr(result, 'content') else str(result)

        # Handle different response formats
        if isinstance(output, list):
            output = "".join(part.get('text', '') if isinstance(part, dict) else str(part) for part in output)
        elif not isinstance(output, str):
            output = str(output)

        # Strip markdown code blocks
        output = output.strip()
        if output.startswith("```"):
            # Remove ```json or ``` at start
            lines = output.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            output = '\n'.join(lines)

        try:
            data = json.loads(output)
            sections = []
            for s in data.get("sections", []):
                sections.append(Section(
                    title=s.get("title", ""),
                    content=s.get("content", ""),
                    citations=s.get("citations", [])
                ))
            return ResearchReport(
                title=data.get("title", "Research Report"),
                summary=data.get("summary", ""),
                sections=sections,
                recommendations=data.get("recommendations", []),
                citations=data.get("citations", [])
            )
        except json.JSONDecodeError:
            return ResearchReport(
                title="Research Report",
                summary="Unable to generate summary.",
                sections=[],
                recommendations=[],
                citations=[]
            )


class ReportCritic:
    """Critiques and improves research reports."""

    def __init__(self, llm=None):
        if llm is None:
            llm = get_llm()
        self.llm = llm
        self._template = CRITIC_PROMPT

    def critique(self, question: str, report: ResearchReport, iteration: int = 0) -> CritiqueOutput:
        """Critique a research report."""
        content = self._template.format(
            question=question,
            report_json=report.model_dump_json(indent=2)
        )

        messages = [HumanMessage(content=content)]
        result = self.llm.invoke(messages)
        output = result.content if hasattr(result, 'content') else str(result)

        # Handle different response formats
        if isinstance(output, list):
            output = "".join(part.get('text', '') if isinstance(part, dict) else str(part) for part in output)
        elif not isinstance(output, str):
            output = str(output)

        # Strip markdown code blocks
        output = output.strip()
        if output.startswith("```"):
            lines = output.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            output = '\n'.join(lines)

        try:
            data = json.loads(output)
            revised = None
            if data.get("revised_report"):
                revised = ResearchReport(**data["revised_report"])

            return CritiqueOutput(
                is_complete=data.get("is_complete", False),
                missing_sections=data.get("missing_sections", []),
                missing_citations=data.get("missing_citations", []),
                suggestions=data.get("suggestions", []),
                revised_report=revised
            )
        except json.JSONDecodeError:
            return CritiqueOutput(
                is_complete=False,
                missing_sections=[],
                missing_citations=[],
                suggestions=["Unable to parse critique results"],
                revised_report=None
            )