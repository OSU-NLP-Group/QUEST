# prompts — LLM Prompt Templates

## Modules

### cache_prompts.py
Contains `llm_extraction_prompts`: A system prompt used in `batch_answer_cache.py` to extract URLs from answer text via LLM. The LLM is instructed to find all unique website URLs in the text, including hard-to-find URLs like bare domains, URLs in quotes, and URLs split across lines.

Used with the `URLs` Pydantic model from `utils/url_tools.py` for structured output.

## Other Prompts (in eval_toolkit.py)
The main evaluation prompts are defined as class attributes in `eval_toolkit.py`:

- **`Extractor.GENERAL_PROMPT`**: Extract structured info from the answer text
- **`Extractor.URL_PROMPT`**: Extract structured info from a fetched webpage
- **`Verifier.SIMPLE_PROMPT`**: Verify a factual/logical claim without external evidence
- **`Verifier.URL_PROMPT`**: Verify a claim against webpage content (text + screenshot)

All prompts include placeholders for: `{task_description}`, `{answer}`, `{claim}`, `{additional_instruction}`, `{web_text}`, `{url}`.

Key prompt design principles:
- Judge should NOT use its own knowledge (except when explicitly asked)
- Minor name variants / number rounding should be accepted
- Source URLs must be explicitly present in the answer (not inferred)
- Screenshots are included as visual context alongside extracted text
