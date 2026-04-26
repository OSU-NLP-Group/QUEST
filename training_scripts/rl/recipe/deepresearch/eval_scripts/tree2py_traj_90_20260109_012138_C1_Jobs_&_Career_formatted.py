import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_pe_categories"
TASK_DESCRIPTION = (
    "According to the California Board for Professional Engineers, Land Surveyors, and Geologists (BPELSG), "
    "what are the three categories of Professional Engineer licensure available in California? For each category, "
    "provide a brief explanation of what the category designation means in terms of practice or title restrictions."
)

EXPECTED_CATEGORIES = ["practice act", "title act", "title authority"]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CategoryInfo(BaseModel):
    name: Optional[str] = None
    explanation: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LicensureCategoriesExtraction(BaseModel):
    practice_act: Optional[CategoryInfo] = None
    title_act: Optional[CategoryInfo] = None
    title_authority: Optional[CategoryInfo] = None
    # Any additional URLs the answer cites globally (e.g., a sources section)
    all_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_categories() -> str:
    return """
    Extract the three California Professional Engineer licensure categories as described in the answer, focusing on the BPELSG framework:
    1) Practice Act
    2) Title Act
    3) Title Authority

    For each category, extract:
    - name: the category name as used in the answer (e.g., "Practice Act", "Title Act", "Title Authority").
    - explanation: the brief explanation given in the answer about what the designation means (practice or title restrictions).
    - sources: all URLs the answer cites specifically in relation to that category (these may be inline links or listed references).

    Also extract:
    - all_sources: any other URLs cited in the answer (e.g., a general sources section), excluding duplicates from per-category sources.

    Notes and rules:
    - Only extract URLs explicitly present in the answer text (including markdown links). Do not invent or infer URLs.
    - Include full URLs. If a URL is missing a protocol (http/https), prepend http://.
    - If a category is not mentioned, set that category object to null.
    - If explanation is missing, set it to null.
    - If no sources are provided for a category, return an empty list for that category's sources.
    - The answer may use synonyms or slightly different wording; map them appropriately:
      • "Practice Act" may be phrased as practice‑act branch(es).
      • "Title Act" may be phrased as title‑act branch(es).
      • "Title Authority" may reference specialty authorities such as Structural Engineer or Geotechnical Engineer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _lc(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _contains_phrase(text: Optional[str], phrase: str) -> bool:
    return phrase in _lc(text)


def category_is_mentioned(cat: Optional[CategoryInfo], key_phrase: str) -> bool:
    if cat is None:
        return False
    # Consider mention if the name or explanation contains the key phrase
    return (_contains_phrase(cat.name, key_phrase) or _contains_phrase(cat.explanation, key_phrase)) and bool(cat.explanation and cat.explanation.strip())


def dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        u = (u or "").strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def is_official_url(url: str) -> bool:
    """
    Treat any *.ca.gov domain as official California government documentation.
    This includes bpelsg.ca.gov, dca.ca.gov, leginfo.legislature.ca.gov, etc.
    """
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host.endswith(".ca.gov")


def filter_official_urls(urls: List[str]) -> List[str]:
    return [u for u in urls if is_official_url(u)]


def collect_all_sources(extracted: LicensureCategoriesExtraction) -> List[str]:
    urls: List[str] = []
    for cat in [extracted.practice_act, extracted.title_act, extracted.title_authority]:
        if cat:
            urls.extend(cat.sources or [])
    urls.extend(extracted.all_sources or [])
    return dedup_urls(urls)


def has_official_for_category(cat: Optional[CategoryInfo], fallback_all: List[str]) -> bool:
    """
    For a present category (with explanation), require at least one official URL either in its own sources
    or, if its own sources are empty, in the global all_sources list.
    """
    if not (cat and cat.explanation and cat.explanation.strip()):
        # If the category isn't present (or explanation not provided), we don't require an official source for it here.
        return True
    cat_sources = dedup_urls(cat.sources or [])
    official = filter_official_urls(cat_sources)
    if official:
        return True
    # fallback to all_sources if category-specific sources are missing
    return len(filter_official_urls(fallback_all)) > 0


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_category(
    evaluator: Evaluator,
    parent_node,
    extracted: LicensureCategoriesExtraction,
    category_key: str,
    parent_id: str,
    parent_desc: str,
    mention_phrase: str,
    claim_text: str,
    add_ins_suffix: str,
) -> None:
    """
    Build a critical parallel node for a single category with:
    - Existence/mention check (custom node, critical)
    - Explanation supported by official/regulatory sources (leaf verify, critical)
    """
    cat_parent = evaluator.add_parallel(
        id=parent_id,
        desc=parent_desc,
        parent=parent_node,
        critical=True,
    )

    cat_info: Optional[CategoryInfo] = getattr(extracted, category_key)

    # 1) Category mentioned and explanation provided (critical gate)
    existence_ok = category_is_mentioned(cat_info, mention_phrase)
    evaluator.add_custom_node(
        result=existence_ok,
        id=f"{parent_id}_mentioned",
        desc=f"Answer mentions '{mention_phrase}' category and provides an explanation.",
        parent=cat_parent,
        critical=True
    )

    # 2) Explanation is supported by sources (critical)
    exp_node = evaluator.add_leaf(
        id=f"{parent_id}_explanation_supported",
        desc=f"{mention_phrase.title()} explanation aligns with official BPELSG/regulatory sources.",
        parent=cat_parent,
        critical=True
    )

    # Prepare sources: prefer official URLs if any; otherwise use whatever is provided; if none, falls back to simple verify
    per_cat_sources = dedup_urls((cat_info.sources if cat_info else []) or [])
    all_sources = collect_all_sources(extracted)
    preferred_sources = filter_official_urls(per_cat_sources) or filter_official_urls(all_sources) or (per_cat_sources or all_sources)

    extracted_explanation = (cat_info.explanation if cat_info and cat_info.explanation else "").strip()
    additional_instruction = (
        "Verify this definition specifically according to California BPELSG or official CA regulatory documentation. "
        "Accept reasonable paraphrases. Do not rely on non-official summaries if official documentation is available. "
        f"The answer's explanation to compare (for context) is:\n\"{extracted_explanation}\"\n"
        f"{add_ins_suffix}"
    )

    await evaluator.verify(
        claim=claim_text,
        node=exp_node,
        sources=preferred_sources if preferred_sources else None,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the California PE licensure categories task (BPELSG).
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured category info
    extracted = await evaluator.extract(
        prompt=prompt_extract_categories(),
        template_class=LicensureCategoriesExtraction,
        extraction_name="category_extraction"
    )

    # Add ground truth context (informational; not used for verification)
    evaluator.add_ground_truth({
        "expected_categories": EXPECTED_CATEGORIES,
        "requirements": "Each category should be named and briefly explained; use official BPELSG or official CA regulatory documentation."
    }, gt_type="rubric_requirements")

    # Build top-level critical node
    top = evaluator.add_parallel(
        id="California_PE_Licensure_Categories",
        desc="Answer identifies all three BPELSG-defined California Professional Engineer licensure categories and briefly explains what each designation means (practice/title restrictions), using official BPELSG or official regulatory documentation.",
        parent=root,
        critical=True
    )

    # Verify Practice Act category
    await verify_category(
        evaluator=evaluator,
        parent_node=top,
        extracted=extracted,
        category_key="practice_act",
        parent_id="Practice_Act_Category",
        parent_desc="Correctly identifies 'practice act' as a licensure category and explains that only appropriately licensed individuals may practice or offer to practice in the covered branch(es) (practice restriction).",
        mention_phrase="practice act",
        claim_text=(
            "In California, 'practice act' PE branches are those where only appropriately licensed individuals in the "
            "covered branch (e.g., Civil, Electrical, Mechanical) may practice or offer to practice engineering in that branch."
        ),
        add_ins_suffix="Focus on the notion that practice (and offers to practice) in these branches is restricted to licensees."
    )

    # Verify Title Act category
    await verify_category(
        evaluator=evaluator,
        parent_node=top,
        extracted=extracted,
        category_key="title_act",
        parent_id="Title_Act_Category",
        parent_desc="Correctly identifies 'title act' as a licensure category and explains that only licensed individuals may use the professional title for that branch (title restriction).",
        mention_phrase="title act",
        claim_text=(
            "In California, 'title act' PE branches restrict the use of the specific professional title (e.g., "
            "'Chemical Engineer') to licensed individuals, while practice in the field is not exclusively restricted "
            "to licensees."
        ),
        add_ins_suffix="Emphasize that the restriction is on the title, not necessarily on all practice activities."
    )

    # Verify Title Authority category
    await verify_category(
        evaluator=evaluator,
        parent_node=top,
        extracted=extracted,
        category_key="title_authority",
        parent_id="Title_Authority_Category",
        parent_desc="Correctly identifies 'title authority' as a licensure category and explains that it indicates an advanced/special authority beyond standard licensure requirements (as defined by BPELSG).",
        mention_phrase="title authority",
        claim_text=(
            "In California, 'title authority' denotes a special authority/title (such as Structural Engineer or "
            "Geotechnical Engineer) that goes beyond the standard Professional Engineer license and requires "
            "additional qualifications or examinations; it governs the use of the specialty title."
        ),
        add_ins_suffix="Look for BPELSG or regulatory descriptions indicating additional qualifications and authority to use a specialty title."
    )

    # Official source verification (critical)
    # Require that each present category (with an explanation) is supported by at least one official *.ca.gov source,
    # either in per-category sources or via the global all_sources.
    all_sources_union = collect_all_sources(extracted)
    official_ok = (
        has_official_for_category(extracted.practice_act, all_sources_union) and
        has_official_for_category(extracted.title_act, all_sources_union) and
        has_official_for_category(extracted.title_authority, all_sources_union)
    )

    evaluator.add_custom_node(
        result=official_ok,
        id="Official_Source_Verification",
        desc="Citations/claims are sourced from the official California BPELSG website or official regulatory documentation.",
        parent=top,
        critical=True
    )

    # Return final summary
    return evaluator.get_summary()