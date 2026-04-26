import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pa_7th_grade_cdc_2026_alignment"
TASK_DESCRIPTION = (
    "In January 2026, the U.S. CDC updated its childhood immunization schedule, "
    "reducing universal recommendations from 17 to 11. Some vaccines moved to high-risk "
    "or shared clinical decision-making categories. Pennsylvania law still mandates certain "
    "immunizations for school entry regardless of CDC universal guidance. Identify which "
    "vaccine(s) Pennsylvania requires for students entering 7th grade that are no longer "
    "universally recommended by CDC after the January 2026 update, and for each: (1) cite the "
    "relevant Pennsylvania legal code confirming the 7th grade requirement, (2) provide an "
    "official Pennsylvania Department of Health or PA legal-code URL for that 7th grade "
    "requirement, (3) verify that CDC no longer universally recommends the vaccine as of Jan 2026, "
    "(4) specify the new CDC category (high-risk or shared clinical decision-making), and "
    "(5) provide an official CDC/HHS URL referencing the January 2026 schedule update."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VaccineItem(BaseModel):
    vaccine_name: Optional[str] = None
    pa_legal_code_citation: Optional[str] = None
    pa_official_urls: List[str] = Field(default_factory=list)
    cdc_non_universal_stated: Optional[bool] = None
    cdc_category: Optional[str] = None
    cdc_update_urls: List[str] = Field(default_factory=list)


class VaccineExtraction(BaseModel):
    vaccines: List[VaccineItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_vaccines() -> str:
    return """
Extract the vaccine(s) that the answer claims meet BOTH conditions:
A) Pennsylvania requires the vaccine for 7th grade entry (mandate under PA law/regulation),
B) The vaccine is no longer universally recommended by the CDC as of the January 2026 childhood schedule update.

For each such vaccine in the answer, return an object with:
- vaccine_name: The vaccine name exactly as stated (e.g., "Tdap", "MenACWY", etc.).
- pa_legal_code_citation: The explicit Pennsylvania legal code citation string the answer provides for the 7th grade requirement (e.g., something like "28 Pa. Code § 23.85" or similar). If none, set to null.
- pa_official_urls: A list of URLs (only official Pennsylvania sources) that the answer provides to confirm the 7th grade requirement. Examples of acceptable official sources include:
  • Pennsylvania Department of Health (e.g., health.pa.gov),
  • Official Pennsylvania Code/Bulletin (e.g., pacodeandbulletin.gov),
  • Other *.pa.gov official domains.
  Only include URLs actually present in the answer. If none, return an empty list.
- cdc_non_universal_stated: A boolean; set true iff the answer explicitly states that this vaccine is no longer universally recommended by CDC as of January 2026. Otherwise false.
- cdc_category: The CDC category stated in the answer that this vaccine moved to, such as "high-risk groups/populations" or "shared clinical decision-making" (SCDM). If not stated, set to null.
- cdc_update_urls: A list of official CDC or HHS announcement/update URLs the answer provides that reference the January 2026 schedule change and this vaccine’s new status/category (e.g., cdc.gov or hhs.gov). Only include URLs actually present in the answer. If none, return an empty list.

Return a JSON object with field "vaccines" as a list of these objects. Do not invent anything not explicitly present in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _boolish_non_universal(val: Optional[bool]) -> bool:
    return bool(val is True)


def _looks_like_cdc_category(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    # Accept common phrasings
    return any(
        key in t
        for key in [
            "shared clinical decision",
            "scdm",
            "shared decision",
            "high-risk",
            "high risk",
            "risk group",
            "risk-based",
        ]
    )


def _flatten_unique(url_groups: List[List[str]]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for group in url_groups:
        for u in group:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                unique.append(u)
    return unique


def _vname(v: VaccineItem, idx: int) -> str:
    return v.vaccine_name if (v and v.vaccine_name) else f"Vaccine #{idx + 1}"


# --------------------------------------------------------------------------- #
# Node builders and verifications                                             #
# --------------------------------------------------------------------------- #
async def build_and_verify_vaccine_set_correct(
    evaluator: Evaluator,
    parent,
    extracted: VaccineExtraction,
):
    """
    Build the 'vaccine_set_correct' node as a sequential critical node with:
      1) at_least_one_vaccine_identified (custom existence)
      2) no_extras_among_listed (leaf verified with union of all provided URLs)
      3) no_missing_qualifying_vaccines (leaf verified with union of all provided URLs)
    """
    node = evaluator.add_sequential(
        id="vaccine_set_correct",
        desc="The answer identifies all and only the vaccine(s) that satisfy BOTH PA 7th-grade requirement and CDC non-universal status as of Jan 2026.",
        parent=parent,
        critical=True,
    )

    # 1) At least one vaccine identified
    has_any = len(extracted.vaccines) > 0
    evaluator.add_custom_node(
        result=has_any,
        id="at_least_one_vaccine_identified",
        desc="At least one qualifying vaccine is identified in the answer.",
        parent=node,
        critical=True,
    )

    # Prepare union URLs for set-level checks
    union_urls = _flatten_unique(
        [v.pa_official_urls + v.cdc_update_urls for v in extracted.vaccines]
    )

    # 2) No extras among listed
    leaf_no_extras = evaluator.add_leaf(
        id="no_extras_among_listed",
        desc="Every vaccine listed in the answer truly meets BOTH conditions (no extra/non-qualifying vaccines included).",
        parent=node,
        critical=True,
    )
    no_extras_claim = (
        "Every vaccine listed in the answer is BOTH: "
        "(i) required by Pennsylvania for 7th-grade entry under state law/regulation AND "
        "(ii) no longer universally recommended by CDC as of the January 2026 schedule update. "
        "If any listed vaccine fails either condition, mark Incorrect."
    )
    await evaluator.verify(
        claim=no_extras_claim,
        node=leaf_no_extras,
        sources=union_urls if union_urls else None,
        additional_instruction=(
            "Use only the provided official Pennsylvania and CDC/HHS URLs. "
            "Confirm PA pages explicitly reference 7th-grade requirements. "
            "Confirm CDC/HHS pages explicitly reference Jan 2026 and the removal from universal recommendations. "
            "If evidence is insufficient for any listed vaccine, mark Incorrect."
        ),
    )

    # 3) No missing qualifying vaccines
    leaf_no_missing = evaluator.add_leaf(
        id="no_missing_qualifying_vaccines",
        desc="No qualifying vaccines are missing (the set is complete).",
        parent=node,
        critical=True,
    )
    no_missing_claim = (
        "Based on the provided official Pennsylvania (PA) and CDC/HHS sources, "
        "there are no additional vaccines (beyond those listed in the answer) that "
        "simultaneously: (a) are required by Pennsylvania for 7th-grade entry and "
        "(b) were removed from CDC universal recommendations in January 2026. "
        "If the provided sources do not allow you to confirm completeness, mark Incorrect."
    )
    await evaluator.verify(
        claim=no_missing_claim,
        node=leaf_no_missing,
        sources=union_urls if union_urls else None,
        additional_instruction=(
            "Judge completeness strictly using the provided official sources only "
            "(health.pa.gov, pacodeandbulletin.gov, *.pa.gov, cdc.gov, hhs.gov). "
            "If you cannot conclusively establish completeness from these sources, mark Incorrect."
        ),
    )


async def build_and_verify_pa_legal_code_presence(
    evaluator: Evaluator,
    parent,
    extracted: VaccineExtraction,
):
    """
    For each vaccine, ensure the answer cites the relevant Pennsylvania legal code for 7th-grade entry.
    Presence check only (citation string exists in the answer for that vaccine).
    """
    node = evaluator.add_parallel(
        id="pa_legal_code_cited_for_each_vaccine",
        desc="For each identified vaccine, the answer cites the relevant Pennsylvania legal code for the 7th grade requirement.",
        parent=parent,
        critical=True,
    )

    for i, v in enumerate(extracted.vaccines):
        has_citation = bool(v.pa_legal_code_citation and v.pa_legal_code_citation.strip())
        evaluator.add_custom_node(
            result=has_citation,
            id=f"pa_code_cited_{i}",
            desc=f"{_vname(v, i)}: Pennsylvania legal code citation is present in the answer.",
            parent=node,
            critical=True,
        )


async def build_and_verify_pa_official_requirement_by_url(
    evaluator: Evaluator,
    parent,
    extracted: VaccineExtraction,
):
    """
    For each vaccine, verify via official PA source URL(s) that the vaccine is required for 7th-grade entry.
    """
    node = evaluator.add_parallel(
        id="pa_official_url_provided_for_each_vaccine",
        desc="For each vaccine, an official PA URL confirms the 7th grade entry requirement.",
        parent=parent,
        critical=True,
    )

    batch: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    for i, v in enumerate(extracted.vaccines):
        leaf = evaluator.add_leaf(
            id=f"pa_7th_grade_requirement_{i}",
            desc=f"{_vname(v, i)}: Official PA source confirms 7th grade entry requirement.",
            parent=node,
            critical=True,
        )

        claim = (
            f"Pennsylvania requires the {_vname(v, i)} vaccine for entry into 7th grade "
            "under state school immunization rules."
        )
        add_ins = (
            "Only pass if the page is an official Pennsylvania source (e.g., health.pa.gov, pacodeandbulletin.gov, *.pa.gov) "
            "and it explicitly confirms a 7th-grade entry requirement (e.g., 'upon entry into 7th grade'). "
            "If the page is non-official or does not mention 7th grade specifically, mark Incorrect."
        )

        sources = v.pa_official_urls if v.pa_official_urls else []
        batch.append((claim, sources, leaf, add_ins))

    if batch:
        await evaluator.batch_verify(batch)


async def build_and_verify_cdc_non_universal_statement_presence(
    evaluator: Evaluator,
    parent,
    extracted: VaccineExtraction,
):
    """
    For each vaccine, ensure the answer explicitly states it is no longer universally recommended by CDC as of Jan 2026.
    Presence check only.
    """
    node = evaluator.add_parallel(
        id="cdc_non_universal_statement_present_for_each_vaccine",
        desc="For each vaccine, the answer explicitly states CDC no longer universally recommends it as of Jan 2026.",
        parent=parent,
        critical=True,
    )

    for i, v in enumerate(extracted.vaccines):
        stated = _boolish_non_universal(v.cdc_non_universal_stated)
        evaluator.add_custom_node(
            result=stated,
            id=f"cdc_non_universal_stmt_{i}",
            desc=f"{_vname(v, i)}: The answer explicitly states CDC no longer universally recommends it as of Jan 2026.",
            parent=node,
            critical=True,
        )


async def build_and_verify_cdc_category_specified(
    evaluator: Evaluator,
    parent,
    extracted: VaccineExtraction,
):
    """
    For each vaccine, ensure the answer specifies which CDC category it moved to (high-risk or SCDM).
    Presence/format check only.
    """
    node = evaluator.add_parallel(
        id="cdc_category_specified_for_each_vaccine",
        desc="For each vaccine, the answer specifies the CDC category (high-risk or shared clinical decision-making).",
        parent=parent,
        critical=True,
    )

    for i, v in enumerate(extracted.vaccines):
        ok = _looks_like_cdc_category(v.cdc_category)
        evaluator.add_custom_node(
            result=ok,
            id=f"cdc_category_present_{i}",
            desc=f"{_vname(v, i)}: CDC category (high-risk or SCDM) is specified in the answer.",
            parent=node,
            critical=True,
        )


async def build_and_verify_cdc_hhs_update_by_url(
    evaluator: Evaluator,
    parent,
    extracted: VaccineExtraction,
):
    """
    For each vaccine, verify via CDC/HHS URL(s) that as of Jan 2026 it is no longer universally recommended
    and was moved to the specified category.
    """
    node = evaluator.add_parallel(
        id="cdc_hhs_official_url_provided_for_each_vaccine",
        desc="For each vaccine, an official CDC/HHS Jan 2026 update URL supports non-universal status and the specified category.",
        parent=parent,
        critical=True,
    )

    batch: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    for i, v in enumerate(extracted.vaccines):
        leaf = evaluator.add_leaf(
            id=f"cdc_2026_update_confirms_{i}",
            desc=f"{_vname(v, i)}: CDC/HHS Jan 2026 update confirms non-universal status and specified category.",
            parent=node,
            critical=True,
        )

        category_text = v.cdc_category if v.cdc_category else "the new category"
        claim = (
            f"As of January 2026, the CDC's childhood immunization schedule update indicates that the "
            f"{_vname(v, i)} vaccine is no longer universally recommended and was moved to '{category_text}'."
        )
        add_ins = (
            "Only pass if the URL is official (cdc.gov or hhs.gov) AND it clearly references the January 2026 schedule update "
            "with the vaccine's non-universal status and the stated category (high-risk or SCDM). "
            "If the page is not official or does not explicitly support both points, mark Incorrect."
        )

        sources = v.cdc_update_urls if v.cdc_update_urls else []
        batch.append((claim, sources, leaf, add_ins))

    if batch:
        await evaluator.batch_verify(batch)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point for evaluating an answer to the Pennsylvania 7th grade vs CDC Jan 2026 schedule task.
    """
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
        default_model=model,
    )

    # Critical task root (because Evaluator root is always non-critical)
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Identify PA-required (7th grade) vaccines that CDC no longer universally recommends as of Jan 2026, with all required PA/CDC evidence.",
        parent=root,
        critical=True,
    )

    # 1) Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_vaccines(),
        template_class=VaccineExtraction,
        extraction_name="extracted_vaccines",
    )

    # 2) Build and run verifications according to rubric
    # 2.1 The vaccine set correctness (as a structured sequential check)
    await build_and_verify_vaccine_set_correct(evaluator, task_root, extracted)

    # 2.2 For each vaccine: PA legal code citation presence (answer-level presence check)
    await build_and_verify_pa_legal_code_presence(evaluator, task_root, extracted)

    # 2.3 For each vaccine: PA official URL confirming 7th grade requirement
    await build_and_verify_pa_official_requirement_by_url(evaluator, task_root, extracted)

    # 2.4 For each vaccine: CDC non-universal statement present in the answer (presence check)
    await build_and_verify_cdc_non_universal_statement_presence(evaluator, task_root, extracted)

    # 2.5 For each vaccine: CDC category specified (presence/format check)
    await build_and_verify_cdc_category_specified(evaluator, task_root, extracted)

    # 2.6 For each vaccine: CDC/HHS official Jan 2026 URL supports non-universal status and category
    await build_and_verify_cdc_hhs_update_by_url(evaluator, task_root, extracted)

    # 3) Return evaluation summary
    return evaluator.get_summary()