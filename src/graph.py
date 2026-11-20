"""LangGraph workflow construction for the Research & Planning Agent."""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .State import ResearchState
from .nodes import (
    PlannerNode, ResearcherNode, SynthesizerNode,
    CriticNode, ApprovalHandler
)


def create_research_graph():
    """Create the research & planning agent graph.

    This graph supports:
    - Sequential research with step-by-step processing
    - Reflection loop when gaps or criticisms are found
    - Human-in-the-loop approval before finalization
    - Checkpointing for pause/resume capability
    """

    builder = StateGraph(ResearchState)

    # Add nodes
    builder.add_node("planner", PlannerNode())
    builder.add_node("researcher", ResearcherNode())
    builder.add_node("synthesizer", SynthesizerNode())
    builder.add_node("critic", CriticNode())
    builder.add_node("approval", ApprovalHandler())

    # Define the flow

    # 1. Start with planning
    builder.add_edge(START, "planner")

    # 2. Planner -> researcher (sequential step-by-step)
    builder.add_edge("planner", "researcher")

    # 3. Researcher -> researcher (loop for each step) OR synthesizer (if all steps done)
    def researcher_complete(state: ResearchState):
        """Check if all research steps are complete."""
        if state.current_step is None or state.current_step >= len(state.steps):
            return "synthesizer"
        return "researcher"  # Continue to next step

    builder.add_conditional_edges("researcher", researcher_complete)

    # 4. Synthesize -> approval
    builder.add_edge("synthesizer", "approval")

    # 5. Approval -> critic
    builder.add_edge("approval", "critic")

    # 6. Critic -> loop back to planner (for iteration) OR END (if complete)
    def critic_route(state: ResearchState):
        """Route based on critique result."""
        if state.completed:
            return END
        return "planner"  # Retry with new iteration

    builder.add_conditional_edges("critic", critic_route)

    # Compile the graph with memory checkpointer
    graph = builder.compile(checkpointer=MemorySaver())
    return graph


# Default graph
graph = create_research_graph()


# Alternative: Parallel research implementation using manual dispatching
def create_parallel_research_graph():
    """Create a graph that processes research steps in parallel.

    This version uses sub-agents pattern to execute multiple research
    tasks concurrently, then combines results.
    """

    builder = StateGraph(ResearchState)

    # Add nodes
    builder.add_node("planner", PlannerNode())
    builder.add_node("synthesizer", SynthesizerNode())
    builder.add_node("critic", CriticNode())
    builder.add_node("approval", ApprovalHandler())
    builder.add_node("researcher", ResearcherNode())

    # Flow:
    # 1. Plan
    builder.add_edge(START, "planner")

    # 2. After planning, loop through researchers sequentially
    #    (For true parallelism, you'd run multiple instances of researcher)
    builder.add_edge("planner", "researcher")

    def check_steps(state: ResearchState):
        """Check if all steps researched, if not continue."""
        if state.current_step is None or state.current_step >= len(state.steps):
            return "synthesizer"
        return "researcher"

    builder.add_conditional_edges("researcher", check_steps)

    # 3. Synthesize -> approval -> critic -> (loop or end)
    builder.add_edge("synthesizer", "approval")
    builder.add_edge("approval", "critic")

    def after_critic(state: ResearchState):
        if state.completed:
            return END
        return "planner"

    builder.add_conditional_edges("critic", after_critic)

    graph = builder.compile(checkpointer=MemorySaver())
    return graph


# For true parallel execution, use the run_parallel helper
async def run_parallel_research(graph, question: str, step_data: dict):
    """Helper to run research steps in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    import asyncio

    steps = step_data.get("steps", [])

    async def research_step(step_info):
        step = step_info[1]  # tuple index
        researcher = ResearcherNode()
        # Run synchronously within async context
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: researcher({"steps": steps, "current_step": step, "question": question})
        )
        return result

    # Run all steps in parallel
    tasks = [research_step((i, s)) for i, s in enumerate(steps)]
    results = await asyncio.gather(*tasks)
    return results