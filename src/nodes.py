"""Node implementations for the Research & Planning Agent."""

import json
import os
from typing import Dict, Any, Optional
from langgraph.types import interrupt
from langchain_core.messages import HumanMessage, AIMessage

from .State import (
    ResearchState, ResearchStep, ResearchReport, Section,
    PlannerOutput, SynthesisOutput, CritiqueOutput
)
from .tools import search_web, WebSearchResult
from .chains import ResearchDecomposer, ReportSynthesizer, ReportCritic, get_llm


class PlannerNode:
    """Decomposes the research question into steps."""

    def __init__(self, decomposer: ResearchDecomposer = None):
        if decomposer is None:
            decomposer = ResearchDecomposer()
        self.decomposer = decomposer

    def __call__(self, state: ResearchState) -> Dict[str, Any]:
        """Decompose the question into research steps."""
        question = state.question

        if state.waiting_for_approval:
            return {"current_step": 0}

        steps = self.decomposer.decompose(question)

        return {
            "steps": steps,
            "current_step": 0,
            "iteration": state.iteration + 1
        }


class ResearcherNode:
    """Researches a single step. Can be called in parallel."""

    def __init__(self):
        pass

    def __call__(self, state: ResearchState, step_id_override: Optional[int] = None) -> Dict[str, Any]:
        """Research a specific step (or current step if no override)."""
        if step_id_override is not None:
            step_idx = step_id_override
        else:
            step_idx = getattr(state, 'current_step', 0)

        steps = state.steps

        if step_idx >= len(steps):
            return {}

        step = steps[step_idx]

        query = f"{step.topic}: {step.question}"

        try:
            result = search_web(query, max_results=5)

            findings = result.raw_answer if result.raw_answer else ""
            for r in result.results:
                findings += f"\n\n{r.title}: {r.snippet}"

            sources = [
                {"title": r.title, "url": r.url}
                for r in result.results
            ]

            updated_step = ResearchStep(
                step_id=step.step_id,
                topic=step.topic,
                question=step.question,
                findings=findings,
                sources=sources,
                completed=True
            )

            new_steps = list(steps)
            new_steps[step_idx] = updated_step

            new_current = state.current_step
            if state.current_step == step_idx:
                new_current = step_idx + 1

            return {
                "steps": new_steps,
                "current_step": new_current
            }

        except Exception as e:
            return {
                "steps": steps,
                "error": f"Research error for step {step_idx + 1}: {str(e)}"
            }


class SynthesizerNode:
    """Synthesizes research results into a report."""

    def __init__(self, synthesizer: ReportSynthesizer = None):
        if synthesizer is None:
            synthesizer = ReportSynthesizer()
        self.synthesizer = synthesizer

    def __call__(self, state: ResearchState) -> Dict[str, Any]:
        """Synthesize research results into a comprehensive report."""
        question = state.question
        steps = state.steps

        research_results = {}
        for step in steps:
            if step.completed:
                research_results[step.step_id] = {
                    "topic": step.topic,
                    "question": step.question,
                    "result": step.findings or "",
                    "sources": step.sources or []
                }

        report = self.synthesizer.synthesize(question, research_results)

        return {
            "report": report,
            "waiting_for_approval": True,
            "plan_approved": False
        }


class CriticNode:
    """Critiques and potentially improves the report."""

    def __init__(self, critic: ReportCritic = None):
        if critic is None:
            critic = ReportCritic()
        self.critic = critic

    def __call__(self, state: ResearchState) -> Dict[str, Any]:
        """Critique the report and decide if iteration is needed."""
        if state.report is None:
            return {
                "critique": "No report to critique",
                "completed": True
            }

        critique_result = self.critic.critique(
            state.question,
            state.report,
            state.iteration
        )

        need_iteration = (
            not critique_result.is_complete and
            state.iteration < state.max_iterations
        )

        if need_iteration:
            report = critique_result.revised_report or state.report
            return {
                "report": report,
                "critique": json.dumps(critique_result.model_dump(exclude=['revised_report']), default=str),
                "iteration": state.iteration + 1,
                "waiting_for_approval": True
            }
        elif critique_result.is_complete:
            return {
                "report": state.report,
                "critique": json.dumps(critique_result.model_dump(exclude=['revised_report']), default=str),
                "completed": True,
                "waiting_for_approval": False
            }
        else:
            return {
                "report": state.report,
                "critique": json.dumps({
                    "is_complete": False,
                    "missing_sections": critique_result.missing_sections,
                    "missing_citations": critique_result.missing_citations,
                    "suggestions": critique_result.suggestions
                }, default=str),
                "completed": True,
                "waiting_for_approval": False
            }


class ApprovalHandler:
    """Handles human approval of the research plan."""

    def __call__(self, state: ResearchState) -> Dict[str, Any]:
        """Auto-approve in non-interactive mode (web service)."""
        # Skip interrupt in non-interactive mode
        if os.getenv("HUMAN_APPROVAL", "true").lower() == "false":
            return {
                "plan_approved": True,
                "waiting_for_approval": False
            }

        # Only interrupt if explicitly enabled and in interactive mode
        if state.waiting_for_approval and not state.plan_approved:
            try:
                user_input = interrupt({
                    "question": "Does this synthesized report look good?",
                    "report_preview": f"Title: {state.report.title}\nSummary: {state.report.summary}" if state.report else "",
                    "options": ["Approve and finalize", "Request revision"]
                })
                action = user_input.get("action", "Approve and finalize")
                return {
                    "plan_approved": action == "Approve and finalize",
                    "waiting_for_approval": False
                }
            except Exception:
                # Fallback: auto-approve if interrupt fails
                return {
                    "plan_approved": True,
                    "waiting_for_approval": False
                }

        return {"waiting_for_approval": False}