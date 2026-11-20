# Research & Planning Agent with LangGraph

A domain-independent, multi-step AI agent that accepts any research or planning request and produces a structured, source-backed report with an actionable plan.

## Features

- **Decompose** any question into research steps
- **Parallel research** via web search (Tavily)
- **Synthesize** into structured reports with citations
- **Self-critique** to identify gaps and iterate
- **Human-in-the-loop** approval checkpoints
- **Checkpointing** for pause/resume and persistence

## Architecture

```
Question → Planner → Research Steps → Parallel Research → Synthesize → Report → Critic → (loop if gaps) → Final Report
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install langgraph langchain langchain-anthropic langchain-openai langchain-google-genai tavily-python python-dotenv
   ```

2. **Set API keys** in `.env`:
   ```bash
   TAVILY_API_KEY=your_key_from_tavily_dashboard
   ANTHROPIC_API_KEY=your_key_from_console_anthropic
   ```

3. **Run:**
   ```bash
   python -m src.main "What is the market size for electric vehicles in India?"
   ```

## CLI Usage

```bash
python -m src.main "Your research question"
python -m src.main "Research question" -o report.json
python -m src.main "Research question" -j  # JSON output only
```

## Components

| File | Purpose |
|------|---------|
| `src/State.py` | State models (ResearchState, ResearchStep, ResearchReport) |
| `src/tools.py` | Tavily search integration |
| `src/chains.py` | LLM chains (planner, synthesizer, critic) |
| `src/nodes.py` | LangGraph node implementations |
| `src/graph.py` | StateGraph construction |
| `src/main.py` | CLI entry point |

## Success Criteria

- Works across domains without code changes
- Parallel research reduces runtime
- Reflection loop terminates within limit
- Every claim traces to a source URL