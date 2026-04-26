import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs_phd_ai_ml_top10_stipend_gre_2026"
TASK_DESCRIPTION = """I am planning to apply for Computer Science PhD programs in Fall 2026 and want to focus on universities with strong Artificial Intelligence and Machine Learning research. Please identify three universities in the United States that meet ALL of the following criteria:

1. The university must be ranked in the top 10 for Computer Science PhD programs according to either U.S. News Best Graduate Schools rankings or CSRankings
2. The university must have an active, named research group or lab specifically focused on AI or Machine Learning, with a dedicated webpage describing the group
3. The university must offer a minimum annual PhD stipend of at least $30,000 for full-time doctoral students (for 12-month appointments)
4. The university must have made GRE scores optional or not required for Fall 2026 Computer Science PhD applications

For each of the three universities you identify, please provide:
- The university name
- A reference URL to the university's Computer Science PhD program webpage or admissions information
- A reference URL to the AI/ML research group webpage
- The stated minimum PhD stipend amount
- Confirmation of the GRE policy for Fall 2026 applications
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    name: Optional[str] = None
    cs_phd_url: Optional[str] = None
    ai_ml_group_url: Optional[str] = None
    stipend_amount: Optional[str] = None  # keep as string for robustness
    gre_policy: Optional[str] = None      # text as stated in the answer
    ranking_source: Optional[str] = None  # e.g., "U.S. News" or "CSRankings"
    ranking_url: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)  # any other referenced URLs


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract ALL universities mentioned in the answer that are proposed as meeting the requested criteria. For each university, extract the following fields exactly as stated:

- name: The university name
- cs_phd_url: A URL to the university's Computer Science PhD program or admissions information (if multiple provided, choose the most directly relevant one)
- ai_ml_group_url: A URL to a named AI/ML research group or lab page at the university
- stipend_amount: The stated minimum annual PhD stipend amount (as text, e.g., "$34,000" or "at least $30k")
- gre_policy: The statement regarding GRE policy for Fall 2026 applications as written in the answer
- ranking_source: Which ranking source the answer claims (must be either "U.S. News" or "CSRankings" if specified; otherwise null)
- ranking_url: A URL to a ranking page, if provided in the answer (from either U.S. News or CSRankings). If none provided, null.
- extra_urls: Any additional URLs cited that relate to funding/financial support, GRE policy, or other CS admissions details (exclude cs_phd_url, ai_ml_group_url, and ranking_url if already captured)

IMPORTANT:
- Do not invent any URLs. Extract only those explicitly present in the answer (including markdown links).
- Include all universities mentioned by the answer (not just three); we will filter later.
- If a field is missing for a university, set it to null (or empty array for extra_urls).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return " ".join(name.strip().lower().split())


def _has_digits(s: Optional[str]) -> bool:
    if not s:
        return False
    return any(ch.isdigit() for ch in s)


def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _collect_all_sources(uni: UniversityItem) -> List[str]:
    return _dedup_urls([uni.cs_phd_url, uni.ai_ml_group_url, uni.ranking_url] + (uni.extra_urls or []))


def _collect_ranking_sources(uni: UniversityItem) -> List[str]:
    candidates = []
    if uni.ranking_url:
        candidates.append(uni.ranking_url)
    if uni.extra_urls:
        for u in uni.extra_urls:
            u_low = u.lower()
            if ("usnews" in u_low) or ("csrankings" in u_low):
                candidates.append(u)
    return _dedup_urls(candidates)


def _infer_ranking_source(uni: UniversityItem) -> Optional[str]:
    if uni.ranking_source and uni.ranking_source.strip():
        return uni.ranking_source.strip()
    if uni.ranking_url:
        r = uni.ranking_url.lower()
        if "csrankings" in r:
            return "CSRankings"
        if "usnews" in r:
            return "U.S. News"
    if uni.extra_urls:
        for u in uni.extra_urls:
            r = u.lower()
            if "csrankings" in r:
                return "CSRankings"
            if "usnews" in r:
                return "U.S. News"
    return None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx: int
) -> None:
    """Build and verify the sub-tree for one university."""
    uni_node = evaluator.add_parallel(
        id=f"University_{idx+1}",
        desc=f"Evaluate the {idx+1}st university against all required constraints and required output fields." if idx == 0 else
             (f"Evaluate the {idx+1}nd university against all required constraints and required output fields." if idx == 1 else
              f"Evaluate the {idx+1}rd university against all required constraints and required output fields."),
        parent=parent_node,
        critical=False  # allow partial across universities
    )

    name = uni.name or ""
    all_sources = _collect_all_sources(uni)
    ranking_sources = _collect_ranking_sources(uni)
    cs_url = uni.cs_phd_url
    ai_url = uni.ai_ml_group_url

    # U_Name_Provided (critical)
    evaluator.add_custom_node(
        result=bool(name.strip()),
        id=f"U{idx+1}_Name_Provided",
        desc=f"University {idx+1} name is provided.",
        parent=uni_node,
        critical=True
    )

    # U_US_Located (critical) – use available official URLs as evidence
    us_loc_node = evaluator.add_leaf(
        id=f"U{idx+1}_US_Located",
        desc=f"University {idx+1} is located in the United States.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is located in the United States.",
        node=us_loc_node,
        sources=[u for u in [cs_url, ai_url] if u] or all_sources,
        additional_instruction="Rely on the provided webpage(s). Accept if indications such as '.edu' domain, U.S. addresses, or explicit location confirm it's a U.S. university."
    )

    # U_Top10_CS_PhD_Ranking_With_Source (critical)
    top10_node = evaluator.add_leaf(
        id=f"U{idx+1}_Top10_CS_PhD_Ranking_With_Source",
        desc=f"University {idx+1} is ranked top 10 for CS PhD programs by U.S. News or CSRankings, and the response specifies which source is used.",
        parent=uni_node,
        critical=True
    )
    inferred_source = _infer_ranking_source(uni)
    source_text = inferred_source if inferred_source else "(source unspecified in answer)"
    top10_claim = f"The answer specifies the ranking source as {source_text}, and according to that source, {name} is in the top 10 for Computer Science PhD programs."
    await evaluator.verify(
        claim=top10_claim,
        node=top10_node,
        sources=ranking_sources if ranking_sources else None,
        additional_instruction="Only pass if BOTH conditions are satisfied: (1) the answer names either 'U.S. News' or 'CSRankings' as the source; and (2) the provided ranking page(s) support that this university is in the top 10 for Computer Science (overall CS). If no valid ranking webpage is provided or it does not show top-10 status, mark as not supported."
    )

    # U_CS_PhD_Program_URL_Provided (critical)
    evaluator.add_custom_node(
        result=bool(cs_url and cs_url.strip()),
        id=f"U{idx+1}_CS_PhD_Program_URL_Provided",
        desc=f"A URL to University {idx+1}'s CS PhD program/admissions information is provided.",
        parent=uni_node,
        critical=True
    )

    # U_CS_PhD_Program_URL_Accessible (critical)
    cs_url_access = evaluator.add_leaf(
        id=f"U{idx+1}_CS_PhD_Program_URL_Accessible",
        desc=f"The provided CS PhD program/admissions URL for University {idx+1} is publicly accessible (valid and reachable).",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage is publicly accessible and provides Computer Science PhD program/admissions information for {name}.",
        node=cs_url_access,
        sources=cs_url if cs_url else None,
        additional_instruction="If the page content loads and clearly relates to the CS PhD program or admissions, consider it accessible and relevant."
    )

    # U_AI_ML_Group_Meets_Criteria (critical)
    ai_group_meets = evaluator.add_leaf(
        id=f"U{idx+1}_AI_ML_Group_Meets_Criteria",
        desc=f"University {idx+1} has an active, named AI/ML research group/lab with a dedicated webpage describing the group.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This page describes a named AI or Machine Learning research group/lab at {name}, and indicates active status (e.g., members, projects, publications, events, or recent updates).",
        node=ai_group_meets,
        sources=ai_url if ai_url else None,
        additional_instruction="Pass only if the page is clearly a dedicated group/lab page focusing on AI/ML and appears active."
    )

    # U_AI_ML_Group_URL_Provided (critical)
    evaluator.add_custom_node(
        result=bool(ai_url and ai_url.strip()),
        id=f"U{idx+1}_AI_ML_Group_URL_Provided",
        desc=f"A URL to University {idx+1}'s AI/ML research group/lab webpage is provided.",
        parent=uni_node,
        critical=True
    )

    # U_AI_ML_Group_URL_Accessible (critical)
    ai_url_access = evaluator.add_leaf(
        id=f"U{idx+1}_AI_ML_Group_URL_Accessible",
        desc=f"The provided AI/ML group/lab URL for University {idx+1} is publicly accessible (valid and reachable).",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is publicly accessible and reachable.",
        node=ai_url_access,
        sources=ai_url if ai_url else None,
        additional_instruction="If the page content loads, consider it accessible."
    )

    # U_Stipend_Minimum_AtLeast_30000_12mo (critical)
    stipend_min_node = evaluator.add_leaf(
        id=f"U{idx+1}_Stipend_Minimum_AtLeast_30000_12mo",
        desc=f"University {idx+1} offers a minimum annual PhD stipend of at least $30,000 for 12-month appointments.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum annual PhD stipend for 12-month appointments at {name} is at least $30,000.",
        node=stipend_min_node,
        sources=[u for u in [cs_url] if u] + (uni.extra_urls or []),
        additional_instruction="Pass only if the provided webpage(s) explicitly indicate a 12-month stipend at or above $30,000 (or state 'at least $30,000+' or similar). If only a 9-month amount is given without a stated 12-month minimum, do NOT assume conversion; mark as not supported."
    )

    # U_Stipend_Amount_Stated (critical)
    evaluator.add_custom_node(
        result=bool(uni.stipend_amount and _has_digits(uni.stipend_amount)),
        id=f"U{idx+1}_Stipend_Amount_Stated",
        desc=f"The response states the minimum PhD stipend amount for University {idx+1}.",
        parent=uni_node,
        critical=True
    )

    # U_GRE_OptionalOrNotRequired_Fall2026 (critical)
    gre_node = evaluator.add_leaf(
        id=f"U{idx+1}_GRE_OptionalOrNotRequired_Fall2026",
        desc=f"University {idx+1} GRE scores are optional or not required for Fall 2026 CS PhD applications (as stated in the response).",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For Fall 2026 CS PhD applications at {name}, GRE scores are optional or not required.",
        node=gre_node,
        sources=[u for u in [cs_url] if u] + (uni.extra_urls or []),
        additional_instruction="Pass only if the page(s) indicate GRE is optional, not required, or waived for Fall 2026. If the page references a different cycle or lacks a clear 2026 policy, mark as not supported."
    )

    # U_Info_Current_Applicable_To_Fall2026 (critical)
    current_app_node = evaluator.add_leaf(
        id=f"U{idx+1}_Info_Current_Applicable_To_Fall2026",
        desc=f"The provided information is applicable to the Fall 2026 admissions cycle (not an unrelated year/cycle).",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cited stipend and GRE policy information for {name} are applicable to the Fall 2026 CS PhD admissions cycle.",
        node=current_app_node,
        sources=[u for u in [cs_url] if u] + (uni.extra_urls or []),
        additional_instruction="Look for explicit 'Fall 2026', '2026-2027', or clear statements covering the 2026 cycle. General policy pages that explicitly state applicability to the 2026 intake are acceptable."
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
    Evaluate an answer for the task: identify three U.S. universities meeting top-10 CS PhD ranking,
    AI/ML group, stipend >= $30k (12-month), and GRE policy optional/not required for Fall 2026.
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
        default_model=model
    )

    # Extract all universities as presented in the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Critical check: exactly three distinct universities provided in the answer text
    # Use raw extracted count (do not trim here)
    raw_names = [_norm_name(u.name) for u in extraction.universities if _norm_name(u.name)]
    distinct_names = set(raw_names)
    exactly_three = (len(raw_names) == 3) and (len(distinct_names) == 3)

    evaluator.add_custom_node(
        result=exactly_three,
        id="Exactly_Three_Universities_Provided",
        desc="Solution provides exactly three (no more, no less) distinct universities.",
        parent=root,
        critical=True
    )

    # For subsequent detailed verification, follow the standard policy to consider only the first 3
    selected: List[UniversityItem] = list(extraction.universities[:3])

    # Pad to ensure three items for consistent evaluation tree shape
    while len(selected) < 3:
        selected.append(UniversityItem())

    # Build sub-trees for each university
    for i in range(3):
        await verify_university(evaluator, root, selected[i], i)

    return evaluator.get_summary()