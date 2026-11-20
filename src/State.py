"""State definitions for the Research & Planning Agent."""

from typing import Annotated, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class ResearchStep(BaseModel):
    """A single research step with its findings."""
    step_id: int
    topic: str
    question: str
    findings: Optional[str] = None
    sources: List[Dict[str, str]] = Field(default_factory=list)
    completed: bool = False


class Section(BaseModel):
    """A section in the research report."""
    title: str
    content: str
    citations: List[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """The final synthesized report."""
    title: str
    summary: str
    sections: List[Section] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    citations: List[Dict[str, str]] = Field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary with sections as dicts."""
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [
                {"title": s.title, "content": s.content, "citations": s.citations}
                for s in self.sections
            ],
            "recommendations": self.recommendations,
            "citations": self.citations
        }


class ResearchState(BaseModel):
    """State for the research & planning agent."""
    # Conversation history
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Original user question
    question: str = ""

    # Decomposed research steps
    steps: List[ResearchStep] = Field(default_factory=list)
    current_step: int = 0

    # Research results
    research_results: Dict[int, Dict[str, Any]] = Field(default_factory=dict)

    # Synthesis
    report: Optional[ResearchReport] = None

    # Reflection/critique
    critique: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 3

    # Human approval
    plan_approved: bool = False
    waiting_for_approval: bool = False

    # Status
    completed: bool = False
    error: Optional[str] = None


class PlannerOutput(BaseModel):
    """Output from the planner node."""
    steps: List[str] = Field(default_factory=list)
    rationale: str = ""


class SynthesisOutput(BaseModel):
    """Output from the synthesizer node."""
    report: ResearchReport
    gaps_identified: List[str] = Field(default_factory=list)


class CritiqueOutput(BaseModel):
    """Output from the critic node."""
    is_complete: bool
    missing_sections: List[str] = Field(default_factory=list)
    missing_citations: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    revised_report: Optional[ResearchReport] = None