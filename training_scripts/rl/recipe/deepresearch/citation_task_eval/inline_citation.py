import asyncio
import json
import re
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

INLINE_CITATION_MAX_SCORE_DEFAULT = 0.25
_INLINE_CITATION_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
_INLINE_CITATION_MAX_WORKERS = 64
_INLINE_CITATION_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")
_INLINE_CITATION_BRACKET_NUM_RE = re.compile(r"\[\s*\d+[^\]]*\]")

EvalLLMAddresses = Optional[Union[str, List[str]]]
EvalLLMChatFn = Callable[..., Awaitable[Optional[str]]]
VisitURLFn = Callable[[str, int], Awaitable[Dict[str, Any]]]
BackendCheckFn = Callable[[EvalLLMAddresses, str, str], bool]


def _coerce_float(value: Any, default: float, min_value: Optional[float] = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if min_value is not None:
        parsed = max(min_value, parsed)
    return parsed


def _strip_json_fence(text: str) -> str:
    if not text:
        return ""
    m = _INLINE_CITATION_JSON_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.replace("```json", "").replace("```", "").strip()


def _clean_escape(input_text: str) -> str:
    if not input_text:
        return ""
    input_text = input_text.replace("\\>", ">")
    input_text = input_text.replace("\\<", "<")
    input_text = input_text.replace("\\+", "+")
    input_text = input_text.replace("\\~", "~")
    return input_text


def _has_inline_citation_markers(text: str) -> bool:
    if not text:
        return False
    if _INLINE_CITATION_MARKDOWN_LINK_RE.search(text):
        return True
    if _INLINE_CITATION_BRACKET_NUM_RE.search(text):
        return True
    return False


def _build_extract_prompt(report_text: str) -> str:
    return f"""You will be provided with a research report. The body of the report will contain some citations to references.

Citations in the main text may appear in the following forms:
1. A segment of text + space + number, for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels 15"
2. A segment of text + [number], for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels[15]"
3. A segment of text + [number†(some line numbers, etc.)], for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels[15†L10][5L23][7†summary]"
4. [Citation Source](Citation Link), for example: "According to [ChinaFile: A Guide to Social Class in Modern China](https://www.chinafile.com/reporting-opinion/media/guide-social-class-modern-china)'s classification, Chinese society can be divided into nine strata"

Please identify **all** instances where references are cited in the main text, and extract (fact, url) pairs. When extracting, pay attention to the following:
1. Since these facts will need to be verified later, you may need to look for some context before and after the citation to ensure that the fact is complete and understandable, rather than just a simple phrase or short expression.
2. If a fact cites multiple references, then it should correspond to two pairs: (fact, url_1) and (fact, url_2).
4. If the main text does not specify the exact location of the citation (for example, only the reference list is listed at the end of the article, without specifying the citation point in the text), please return an empty list.

You should return a JSON list format, where each item in the list is a pair, for example:
[
    {{
        "fact": "Text segment from the original document. And add a single backslash before the English quotation mark to make it a readable for python json module.",
        "url": "The URL of the cited reference for this text segment (extracted from the reference list at the end of the research report or from the parentheses at the citation point)."
    }}
]

Here is the main text of the research report:
{report_text}

Please begin the extraction now. Output only the JSON list directly, without any chitchat or explanations."""


def _build_dedup_prompt(statements: str) -> str:
    return f"""You will be given a list of statements. You need to de-duplicate them and return a list of indices of the unique statements. Note: Two statements are considered duplicates only if they express *exactly the same thing*. If there are no duplicate statements in the list, return the complete list of indices.

You should return a List(int), where each item in the list is the index of a unique, non-duplicated statement that has been retained. For example:
[1, 3, 5]

Below is the list of statements you need to de-duplicate:
{statements}

Please begin the extraction now. Output only the integer list, without any conversational text or explanations."""


def _build_validate_prompt(reference: str, statements: str) -> str:
    return f"""You will be provided with a reference and some statements. Please determine whether each statement is 'supported', 'unsupported', or 'unknown' with respect to the reference. Please note:
First, assess whether the reference contains any valid content. If the reference contains no valid information, such as a 'page not found' message, then all statements should be considered 'unknown'.
If the reference is valid, for a given statement: if the facts or data it contains can be found entirely or partially within the reference, it is considered 'supported' (data accepts rounding); if all facts and data in the statement cannot be found in the reference, it is considered 'unsupported'.

You should return the result in a JSON list format, where each item in the list contains the statement's index and the judgment result, for example:
[
    {{
        "idx": 1,
        "result": "supported"
    }},
    {{
        "idx": 2,
        "result": "unsupported"
    }}
]

Below are the reference and statements:
<reference>
{reference}
</reference>

<statements>
{statements}
</statements>

Begin the assessment now. Output only the JSON list, without any conversational text or explanations."""


async def _call_eval_llm_with_retries(
    eval_llm_addresses: EvalLLMAddresses,
    eval_llm_model: str,
    user_prompt: str,
    llm_chat_fn: EvalLLMChatFn,
    profile: str,
    max_retries: int = 3,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Optional[str]:
    messages = [
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": user_prompt},
    ]
    for _ in range(max_retries):
        resp = await llm_chat_fn(
            eval_llm_addresses=eval_llm_addresses,
            messages=messages,
            model=eval_llm_model,
            profile=profile,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if resp:
            return resp
    return None


def _parse_extracted_citations(raw: str) -> List[Dict[str, str]]:
    if not raw:
        return []
    try:
        parsed = json.loads(_clean_escape(_strip_json_fence(raw)))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []

    citations: List[Dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact", "")).strip()
        url = str(item.get("url", "")).strip()
        if not fact or not url:
            continue
        citations.append({"fact": fact, "url": url})
    return citations


async def _extract_single_citations(
    report_text: str,
    eval_llm_addresses: EvalLLMAddresses,
    eval_llm_model: str,
    llm_chat_fn: EvalLLMChatFn,
    profile: str,
) -> List[Dict[str, str]]:
    resp = await _call_eval_llm_with_retries(
        eval_llm_addresses=eval_llm_addresses,
        eval_llm_model=eval_llm_model,
        user_prompt=_build_extract_prompt(report_text),
        llm_chat_fn=llm_chat_fn,
        profile=profile,
        max_retries=3,
        temperature=0.0,
        max_tokens=8192,
    )
    return _parse_extracted_citations(resp or "")


def _parse_dedup_indices(raw: str, max_idx: int) -> List[int]:
    if not raw:
        return []
    try:
        parsed = json.loads(_clean_escape(_strip_json_fence(raw)))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    kept: List[int] = []
    for x in parsed:
        try:
            kept.append(int(x))
        except Exception:
            continue
    if (not kept) or (0 in kept) or (len(kept) > max_idx):
        return []
    return kept


async def _deduplicate_citations(
    citations: List[Dict[str, str]],
    eval_llm_addresses: EvalLLMAddresses,
    eval_llm_model: str,
    llm_chat_fn: EvalLLMChatFn,
    profile: str,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for c in citations:
        grouped[c["url"]].append(c["fact"])

    deduped: Dict[str, Dict[str, Any]] = {}
    tasks = []
    urls = []
    for url, facts in grouped.items():
        if len(facts) <= 1:
            deduped[url] = {"facts": facts[:], "url_content": None}
            continue
        statements = "\n".join(f"{i + 1}. {fact}" for i, fact in enumerate(facts))
        tasks.append(
            _call_eval_llm_with_retries(
                eval_llm_addresses=eval_llm_addresses,
                eval_llm_model=eval_llm_model,
                user_prompt=_build_dedup_prompt(statements),
                llm_chat_fn=llm_chat_fn,
                profile=profile,
                max_retries=3,
                temperature=0.0,
                max_tokens=512,
            )
        )
        urls.append(url)

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for url, raw in zip(urls, results):
            facts = grouped[url]
            if isinstance(raw, Exception):
                kept_idx = []
            else:
                kept_idx = _parse_dedup_indices(raw or "", len(facts))
            if not kept_idx:
                kept_idx = list(range(1, len(facts) + 1))
            deduped[url] = {
                "facts": [facts[i - 1] for i in kept_idx],
                "url_content": None,
            }

    return deduped


async def _scrape_url_with_retry_async(
    url: str,
    visit_url_fn: VisitURLFn,
    max_retries: int = 3,
) -> Dict[str, Any]:
    return await visit_url_fn(url, max_retries)


async def _scrape_citations(
    citations_deduped: Dict[str, Dict[str, Any]],
    visit_url_fn: VisitURLFn,
) -> Dict[str, Dict[str, Any]]:
    urls = [url for url, item in citations_deduped.items() if not item.get("url_content")]
    if not urls:
        return citations_deduped

    sem = asyncio.Semaphore(_INLINE_CITATION_MAX_WORKERS)

    async def _bounded_scrape(u: str):
        async with sem:
            return await _scrape_url_with_retry_async(u, visit_url_fn=visit_url_fn)

    results = await asyncio.gather(*[_bounded_scrape(url) for url in urls], return_exceptions=True)
    for url, res in zip(urls, results):
        if isinstance(res, Exception):
            citations_deduped[url]["url_content"] = f"scrape failed: {res}"
        else:
            citations_deduped[url]["url_content"] = res.get("url_content", "")
    return citations_deduped


def _parse_validate_results(raw: str, n_facts: int) -> Optional[List[Dict[str, Any]]]:
    if not raw:
        return None
    try:
        parsed = json.loads(_clean_escape(_strip_json_fence(raw)))
    except Exception:
        return None
    if not isinstance(parsed, list):
        return None

    for item in parsed:
        if not isinstance(item, dict):
            return None
        try:
            item["idx"] = int(item.get("idx", 0)) - 1
        except Exception:
            return None
    if len(parsed) != n_facts:
        return None
    return parsed


async def _validate_single_url_citations(
    url: str,
    citation_data: Dict[str, Any],
    eval_llm_addresses: EvalLLMAddresses,
    eval_llm_model: str,
    llm_chat_fn: EvalLLMChatFn,
    profile: str,
) -> Dict[str, Any]:
    ref = citation_data.get("url_content")
    facts = citation_data.get("facts", [])
    if ref is None:
        return {"url": url, "validate_res": [], "validate_error": "no reference"}

    facts_text = "\n".join(f"{i + 1}. {fact}" for i, fact in enumerate(facts))
    prompt = _build_validate_prompt(str(ref)[:100 * 1024], facts_text)

    error = None
    for _ in range(3):
        resp = await _call_eval_llm_with_retries(
            eval_llm_addresses=eval_llm_addresses,
            eval_llm_model=eval_llm_model,
            user_prompt=prompt,
            llm_chat_fn=llm_chat_fn,
            profile=profile,
            max_retries=1,
            temperature=0.0,
            max_tokens=2048,
        )
        parsed = _parse_validate_results(resp or "", len(facts))
        if parsed is not None:
            return {"url": url, "validate_res": parsed, "validate_error": None}
        error = "invalid_validate_response"
        await asyncio.sleep(1)

    return {"url": url, "validate_res": [], "validate_error": error}


async def _validate_citations(
    scrape_res: Dict[str, Dict[str, Any]],
    eval_llm_addresses: EvalLLMAddresses,
    eval_llm_model: str,
    llm_chat_fn: EvalLLMChatFn,
    profile: str,
) -> Dict[str, Dict[str, Any]]:
    urls = list(scrape_res.keys())
    if not urls:
        return scrape_res

    sem = asyncio.Semaphore(_INLINE_CITATION_MAX_WORKERS)

    async def _bounded_validate(u: str):
        async with sem:
            return await _validate_single_url_citations(
                url=u,
                citation_data=scrape_res[u],
                eval_llm_addresses=eval_llm_addresses,
                eval_llm_model=eval_llm_model,
                llm_chat_fn=llm_chat_fn,
                profile=profile,
            )

    results = await asyncio.gather(*[_bounded_validate(url) for url in urls], return_exceptions=True)
    for url, res in zip(urls, results):
        if isinstance(res, Exception):
            scrape_res[url]["validate_res"] = []
            scrape_res[url]["validate_error"] = str(res)
        else:
            scrape_res[url]["validate_res"] = res.get("validate_res", [])
            scrape_res[url]["validate_error"] = res.get("validate_error")
    return scrape_res


def _calculate_citation_statistics(res_map: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    total_citations = 0
    total_valid_citations = 0

    for item in res_map.values():
        if item.get("validate_error") is not None:
            continue
        for v in item.get("validate_res", []):
            state = str(v.get("result", "unknown")).lower()
            if state != "unknown":
                total_citations += 1
                if state == "supported":
                    total_valid_citations += 1

    valid_rate = total_valid_citations / total_citations if total_citations > 0 else 0.0
    return {
        "avg_citations": float(total_citations),
        "avg_valid_citations": float(total_valid_citations),
        "valid_rate": float(valid_rate),
    }


async def compute_inline_citation_score(
    solution_str: str,
    extracted_answer: str,
    eval_llm_addresses: EvalLLMAddresses,
    eval_llm_model: str,
    llm_chat_fn: EvalLLMChatFn,
    visit_url_fn: VisitURLFn,
    has_backend_fn: Optional[BackendCheckFn] = None,
    profile: str = "citation",
    max_score: float = INLINE_CITATION_MAX_SCORE_DEFAULT,
    min_required_citations: int = 2,
    max_urls: int = 3,
    max_facts_per_url: int = 3,
) -> Dict[str, Any]:
    del solution_str
    del min_required_citations, max_urls, max_facts_per_url

    max_score = _coerce_float(max_score, INLINE_CITATION_MAX_SCORE_DEFAULT, min_value=0.0)
    if not _has_inline_citation_markers(extracted_answer):
        return {
            "citation_score": 0.0,
            "citation_score_max": max_score,
            "citation_score_applied": True,
            "citation_eval_status": "no_inline_penalty",
            "citation_stats": {"avg_citations": 0.0, "avg_valid_citations": 0.0, "valid_rate": 0.0},
        }

    has_backend = has_backend_fn(eval_llm_addresses, eval_llm_model, profile) if has_backend_fn else bool(eval_llm_addresses)
    if not has_backend:
        return {
            "citation_score": 0.0,
            "citation_score_max": max_score,
            "citation_score_applied": False,
            "citation_eval_status": "skipped_no_eval_llm",
        }

    citations = await _extract_single_citations(
        report_text=extracted_answer,
        eval_llm_addresses=eval_llm_addresses,
        eval_llm_model=eval_llm_model,
        llm_chat_fn=llm_chat_fn,
        profile=profile,
    )
    if not citations:
        return {
            "citation_score": 0.0,
            "citation_score_max": max_score,
            "citation_score_applied": True,
            "citation_eval_status": "no_citations",
            "citation_stats": {"avg_citations": 0.0, "avg_valid_citations": 0.0, "valid_rate": 0.0},
        }

    deduped = await _deduplicate_citations(
        citations=citations,
        eval_llm_addresses=eval_llm_addresses,
        eval_llm_model=eval_llm_model,
        llm_chat_fn=llm_chat_fn,
        profile=profile,
    )
    if not deduped:
        return {
            "citation_score": 0.0,
            "citation_score_max": max_score,
            "citation_score_applied": True,
            "citation_eval_status": "no_citations_after_dedup",
            "citation_stats": {"avg_citations": 0.0, "avg_valid_citations": 0.0, "valid_rate": 0.0},
        }

    scraped = await _scrape_citations(deduped, visit_url_fn=visit_url_fn)
    validated = await _validate_citations(
        scrape_res=scraped,
        eval_llm_addresses=eval_llm_addresses,
        eval_llm_model=eval_llm_model,
        llm_chat_fn=llm_chat_fn,
        profile=profile,
    )
    stats = _calculate_citation_statistics(validated)
    valid_rate = float(stats.get("valid_rate", 0.0))
    citation_score = round(max_score * valid_rate, 6)

    return {
        "citation_score": citation_score,
        "citation_score_max": max_score,
        "citation_score_applied": True,
        "citation_eval_status": "scored",
        "citation_stats": stats,
    }
