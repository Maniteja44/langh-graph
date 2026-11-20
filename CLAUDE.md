# Research & Planning Agent with LangGraph

A domain-independent, multi-step AI agent that accepts any research or planning request and produces a structured, source-backed report with an actionable plan.

## Architecture

- **Orchestration**: LangGraph `StateGraph` with conditional edges, `Send` for parallelism, `interrupt` for human-in-the-loop
- **LLM**: Provider-agnostic via `get_llm()` - supports anthropic, openai, google, groq
- **Search**: Tavily web search
- **State**: Pydantic models for structured data
- **Persistence**: `MemorySaver` with SQLite for production
- **Language**: Python 3.14

## Key Components

```
Question → Planner → Research Steps → Parallel Research → Synthesize → Report → Critic → (loop back if gaps) → Final Report
```

## Files

- `src/State.py` - ResearchState, ResearchStep, ResearchReport, Section models
- `src/chains.py` - LLM chains (planner, synthesizer, critic) - `get_llm()` factory
- `src/tools.py` - Tavily search wrapper with `search_web()`
- `src/nodes.py` - LangGraph node implementations
- `src/graph.py` - StateGraph construction with sequential or parallel research
- `src/main.py` - CLI entry point
- `.env` - API keys configuration

## Configuration

Set in `.env`:
- `TAVILY_API_KEY` - Your Tavily API key (from dashboard.tavily.com)
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY` - LLM provider key
- `LLM_PROVIDER` - openai, anthropic, google, or groq
- `LLM_MODEL` - Model name (default: gpt-4.1-mini for openai)

## Usage

```bash
python -m src.main "What is the market size for electric vehicles in India?"
python -m src.main "Question" -o output.json
python -m src.main "Question" -j  # JSON only
```

## Success Criteria

- Works across domains without code changes
- Parallel research reduces runtime
- Reflection loop terminates within limit
- Every claim traces to source URL