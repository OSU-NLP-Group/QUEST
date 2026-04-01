# Mind2Web-2: Evaluation Framework for Agentic Search

## Project Overview

Mind2Web-2 is a benchmark and evaluation toolkit for agentic search systems (e.g., OpenAI Deep Research). It evaluates how well AI agents can autonomously browse the web, gather information, and synthesize long-form answers with proper source attribution.

The core innovation is **Agent-as-a-Judge**: task-specific LLM judge agents that use tree-structured rubrics to automatically assess answer correctness and source attribution.

**Paper**: "Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge" (NeurIPS'25 D&B)

## Architecture Overview

```
run_eval.py                  # CLI entrypoint for evaluation
batch_answer_cache.py        # Pre-caches webpages referenced in answers
cache_all_answers.sh         # Shell wrapper to batch-cache all answers for an agent

mind2web2/                   # Core Python package
├── evaluator.py             # High-level Evaluator class (extract + verify orchestration)
├── eval_toolkit.py          # Extractor & Verifier classes (LLM-powered extraction/verification)
├── verification_tree.py     # VerificationNode tree with aggregation strategies
├── eval_runner.py           # Async task/answer evaluation orchestrator
├── api_tools/               # External API integrations (arXiv, Google Maps, PDF)
├── llm_client/              # LLM provider abstraction (OpenAI, Azure, Bedrock)
├── utils/                   # Shared utilities (caching, logging, browser, paths)
└── prompts/                 # Prompt templates for LLM extraction

answers/                     # Agent answer files (markdown), organized by agent/task
eval_scripts/                # Per-task evaluation scripts (downloaded from HuggingFace)
cache/                       # Cached webpage content (text + screenshots)
eval_results/                # Evaluation output (JSON results + logs)
cache_manager/               # [DEPRECATED] PySide6 GUI for managing webpage cache
```

## Key Concepts

### Evaluation Pipeline
1. **Agent produces answers** as markdown files with URL citations in `answers/<agent>/<task>/answer_*.md`
2. **Webpages are cached** (text + screenshots) via `batch_answer_cache.py` to ensure reproducibility
3. **Evaluation scripts** (one per task) define `async def evaluate_answer(...)` that builds a verification tree
4. **The judge agent** extracts claims from answers, then verifies each claim against source URLs using LLM calls

### Verification Tree
- Tree-structured rubric where each node is a `VerificationNode`
- Leaf nodes get binary pass/fail scores via LLM verification
- Two aggregation strategies: `PARALLEL` (weighted average) and `SEQUENTIAL` (short-circuit on failure)
- **Critical nodes**: if any critical child fails, the parent scores 0.0 (acts as a gate)

### Evaluator API (used in eval scripts)
```python
evaluator = Evaluator()
root = evaluator.initialize(task_id=..., agent_name=..., answer_name=...)

# Extract structured info from answer
info = await evaluator.extract(prompt, TemplateClass)

# Verify claims (with optional URL sources)
await evaluator.verify(claim="...", node=leaf_node, sources="https://...")

# Get final result
return evaluator.get_summary()
```

### Concurrency Model
- Async throughout (`asyncio`)
- Semaphores control concurrency at multiple levels: tasks, answers, webpage retrieval, LLM requests
- `DualSemaphore` wraps both webpage and LLM semaphores

## Environment Setup

```bash
uv sync                         # Install dependencies
patchright install               # Install browsers
export OPENAI_API_KEY="..."     # Required for evaluation
```

## Running Evaluation

```bash
# Evaluate all tasks for an agent
uv run run_eval.py --agent_name example

# Evaluate a specific task
uv run run_eval.py --agent_name example --task_id yu_lineage
```

## Key Dependencies
- **patchright**: Stealth browser automation for webpage capture (undetected Playwright fork)
- **openai**: LLM API calls (OpenAI / Azure OpenAI)
- **pydantic**: Structured output parsing from LLM responses
- **PyMuPDF (fitz)**: PDF parsing and rendering
- **html2text**: HTML to markdown conversion

## Notes for Developers

- `cache_manager/` is deprecated and scheduled for removal. Do not invest effort there.
- Eval scripts are per-task Python files downloaded from HuggingFace; they are not part of this repo's source.
- The default judge model is `o4-mini` (OpenAI reasoning model).
- URL matching is complex due to variants (utm params, www prefix, encoding, trailing slashes); see `CacheFileSys._find_url()` and `url_tools.py`.
- All LLM calls use exponential backoff retry via the `backoff` library.
