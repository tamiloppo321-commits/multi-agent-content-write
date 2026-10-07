# Multi-Agent Content Writer — LangGraph

A GitHub-ready implementation of the **Multi-Agent Content Writer** assignment.

## Architecture

```text
                         ┌── SEO Blog Writer ── Research/Search Tools
                         │
User ──> Router ─────────┼── X/Twitter Writer ── Search Tool
                         │
                         └── General Handler ──> END
```

## Features

- LangGraph state graph
- Three-way LLM/router classification
- SEO Blog Writer
- X/Twitter Writer
- General Handler
- Research and internet-search tools
- Tool binding
- Conversation messages
- `MemorySaver` persistence
- Thread-based conversation state
- Automated routing tests
- Mock tools for development without search API credentials

## Assignment mapping

The repository follows the original notebook structure:

1. Setup
2. Define State
3. Router Node
4. Tools
5. SEO Blog Writer
6. X/Twitter Writer
7. General Handler
8. Build Graph
9. Persistence
10. Test

The original notebook is preserved at:

`notebooks/Multi-Agent Content Writer - Assignment.ipynb`

## Setup

```bash
git clone https://github.com/<YOUR_USERNAME>/multi-agent-content-writer.git
cd multi-agent-content-writer

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Optional OpenAI setup:

```bash
copy .env.example .env
```

or on macOS/Linux:

```bash
cp .env.example .env
```

Add your `OPENAI_API_KEY`.

## Run

```bash
python src/multi_agent_content_writer.py
```

The project has a deterministic fallback mode, so basic routing and generation work even without an OpenAI key.

## Test

```bash
pytest -q
```

## Persistence

The graph uses LangGraph `MemorySaver`.

Reuse a `thread_id`:

```python
run_agent("What was my previous request?", thread_id="user-001")
```

to keep the same conversation thread.

## Tool strategy

The assignment recommends separating expensive deep research from lighter internet search.

### `research_tool`

Designed for:
- Articles
- Papers
- References
- Deep background research

### `internet_search_tool`

Designed for:
- SEO keywords
- Trending topics
- Hashtags
- Quick lookups

Both are currently deterministic mock implementations. They can be replaced with Tavily, Perplexity, SerpAPI, Google Custom Search, or another provider.

## Resume / LinkedIn description

**Multi-Agent Content Writer — LangGraph**

> Built a multi-agent content generation system using LangGraph with an intelligent router, specialized SEO and X/Twitter agents, tool-calling workflows, shared state, and MemorySaver persistence. Implemented extensible research/search tools and automated routing tests for production-oriented agentic AI workflows.

## Production enhancements

- Live Tavily/Perplexity/SerpAPI integration
- LangSmith tracing
- Persistent database checkpointer
- Explicit tool-loop nodes
- Agent evaluation dataset
- Cost and latency tracking
- Guardrails and content moderation
- Rate limiting
- Structured router outputs
