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
TASK_ID = "nasa_lunar_universities"
TASK_DESCRIPTION = (
    "Identify 3-4 universities in the United States that are qualified to serve as lead academic partners in a new "
    "NASA-funded research initiative focused on lunar resource utilization and regolith processing. Each university "
    "must meet all of the following requirements:\n\n"
    "1. Have an existing formal partnership with NASA through either the Space Grant Consortium network or the Jet "
    "Propulsion Laboratory's (JPL) Joint University and Corporation Initiative (JUCI) program\n"
    "2. Demonstrate active research in lunar science, lunar regolith analysis, in-situ resource utilization (ISRU), or "
    "related lunar exploration technologies, with documented evidence from the period 2023-2026\n"
    "3. Offer graduate degree programs (Master's and/or PhD) in relevant disciplines such as aerospace engineering, "
    "planetary science, materials science, or related STEM fields\n"
    "4. Maintain institutional subscriptions to major academic research databases that support graduate research, "
    "including access to comprehensive dissertation and thesis repositories\n"
    "5. Provide graduate student funding opportunities through fellowships, research assistantships, or Space Grant "
    "programs specifically supporting aerospace and space science research\n\n"
    "For each university you identify, provide the university name and documentation showing how it satisfies each of "
    "the five requirements above, including specific reference URLs that verify the information."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class UniversityItemExtraction(BaseModel):
    name: Optional[str] = None
    # Requirement 1: NASA partnership (Space Grant or JPL JUCI)
    partnership_urls: List[str] = Field(default_factory=list)
    # Requirement 2: Active lunar research (2023-2026)
    research_urls: List[str] = Field(default_factory=list)
    # Requirement 3: Graduate programs (Master's/PhD) in relevant fields
    program_urls: List[str] = Field(default_factory=list)
    # Requirement 4: Institutional research database subscriptions including theses/dissertations
    infrastructure_urls: List[str] = Field(default_factory=list)
    # Requirement 5: Graduate funding for aerospace/space research
    funding_urls: List[str] = Field(default_factory=list)


class UniversityCollectionExtraction(BaseModel):
    universities: List[UniversityItemExtraction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return (
        "Extract up to four U.S. universities mentioned in the answer that the solution proposes for the NASA lunar "
        "resource utilization/regolith processing initiative. For each identified university, extract the following "
        "fields as they appear in the answer:\n"
        "- name: The university name\n"
        "- partnership_urls: A list of URLs that specifically verify a formal partnership with NASA via the Space Grant "
        "Consortium (national or state network lists) or JPL's JUCI program\n"
        "- research_urls: A list of URLs that document active lunar-related research (lunar science, lunar regolith, "
        "ISRU, lunar exploration technologies). Prefer project pages, publications, lab pages, or news releases. "
        "Include only items dated within 2023-2026 when possible\n"
        "- program_urls: A list of URLs for graduate program pages (Master's and/or PhD) in relevant disciplines such "
        "as aerospace engineering, planetary science, materials science, or closely related STEM fields\n"
        "- infrastructure_urls: A list of URLs that demonstrate institutional subscriptions/access to major academic "
        "research databases that support graduate research; include pages listing comprehensive thesis/dissertation "
        "repositories (e.g., ProQuest Dissertations & Theses) or major index databases (e.g., Web of Science, Scopus)\n"
        "- funding_urls: A list of URLs that show graduate student funding opportunities—fellowships, research "
        "assistantships, or Space Grant funding—specifically supporting aerospace and space science research\n\n"
        "Rules:\n"
        "1) Extract only URLs that are explicitly present in the answer; do not invent URLs.\n"
        "2) If any category for a university lacks URLs in the answer, return an empty list for that field.\n"
        "3) If the answer lists more than four universities, include only the first four.\n"
        "4) If the answer references a source without a URL (e.g., 'according to NASA'), return an empty list for that "
        "field.\n"
        "5) Return an object with a 'universities' array of university objects as specified."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _count_named_universities(unis: List[UniversityItemExtraction]) -> int:
    return sum(1 for u in unis if u.name and u.name.strip() != "")


def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst or []


# --------------------------------------------------------------------------- #
# Verification for a single university                                        #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItemExtraction,
    index: int,
) -> None:
    """
    Build verification sub-tree and run checks for one university item.
    The index is 1-based for ID naming consistency with rubric.
    """
    uni_name = uni.name or ""

    # University container node (non-critical; allows partial credit across universities)
    uni_node = evaluator.add_parallel(
        id=f"university_{index}",
        desc=f"{['First','Second','Third','Fourth'][index-1]} identified university meets all partnership and capability requirements",
        parent=parent_node,
        critical=False,
    )

    # ---------------- Requirement 1: NASA partnership ----------------
    nasa_node = evaluator.add_parallel(
        id=f"u{index}_nasa_partnership",
        desc="University has formal NASA partnership through Space Grant or JPL JUCI",
        parent=uni_node,
        critical=True,
    )

    partnership_status = evaluator.add_parallel(
        id=f"u{index}_partnership_status",
        desc="University is documented as Space Grant Consortium member OR JPL JUCI partner",
        parent=nasa_node,
        critical=True,
    )

    # Existence of verification URL(s)
    partnership_url_exists = evaluator.add_custom_node(
        result=len(_safe_list(uni.partnership_urls)) > 0,
        id=f"u{index}_partnership_verification_url",
        desc="Verification URL provided from official NASA Space Grant or JPL JUCI source",
        parent=partnership_status,
        critical=True,
    )

    partnership_doc_leaf = evaluator.add_leaf(
        id=f"u{index}_partnership_documentation",
        desc="Official documentation or listing confirms partnership status",
        parent=partnership_status,
        critical=True,
    )
    claim_partnership = (
        f"The university {uni_name} is listed as either a NASA Space Grant Consortium member (national or state-level) "
        f"or a JPL JUCI partner on the provided page(s)."
    )
    await evaluator.verify(
        claim=claim_partnership,
        node=partnership_doc_leaf,
        sources=_safe_list(uni.partnership_urls),
        additional_instruction=(
            "Confirm that the page explicitly lists the university as a Space Grant Consortium member (national site "
            "or official state Space Grant site) or as a JPL JUCI partner. Look for member lists, partner directories, "
            "or official program pages."
        ),
        extra_prerequisites=[partnership_url_exists],
    )

    # ---------------- Requirement 2: Active lunar research (2023-2026) ----------------
    lunar_node = evaluator.add_parallel(
        id=f"u{index}_lunar_research",
        desc="University demonstrates active lunar research activity (2023-2026)",
        parent=uni_node,
        critical=True,
    )

    relevance_node = evaluator.add_parallel(
        id=f"u{index}_research_relevance",
        desc="Research involves lunar science, regolith, ISRU, or lunar exploration technologies",
        parent=lunar_node,
        critical=True,
    )

    # Topic match
    research_topic_leaf = evaluator.add_leaf(
        id=f"u{index}_research_topic_match",
        desc="Research topic directly relates to lunar resource utilization or regolith processing",
        parent=relevance_node,
        critical=True,
    )
    claim_topic = (
        "The provided page(s) demonstrate research directly related to lunar resource utilization (ISRU), lunar regolith "
        "analysis/processing, or lunar exploration technologies."
    )
    research_urls_safe = _safe_list(uni.research_urls)
    research_url_exists = evaluator.add_custom_node(
        result=len(research_urls_safe) > 0,
        id=f"u{index}_research_verification_url",
        desc="Verification URL provided showing specific lunar research projects or publications",
        parent=lunar_node,  # Placed under lunar_node; will be a critical sibling via auto preconditions
        critical=True,
    )
    await evaluator.verify(
        claim=claim_topic,
        node=research_topic_leaf,
        sources=research_urls_safe,
        additional_instruction=(
            "Look for explicit mentions of 'lunar', 'regolith', 'ISRU', 'in-situ resource utilization', "
            "'lunar exploration', 'Moon', or equivalent terms on project pages, publications, or lab pages."
        ),
        extra_prerequisites=[research_url_exists],
    )

    # Documentation (projects/publications/collaborations)
    research_doc_leaf = evaluator.add_leaf(
        id=f"u{index}_research_documentation",
        desc="Specific research projects, publications, or collaborations are documented",
        parent=relevance_node,
        critical=True,
    )
    claim_research_doc = (
        "The provided page(s) document concrete lunar-related research activities, such as specific projects, "
        "peer-reviewed publications, or formal collaborations."
    )
    await evaluator.verify(
        claim=claim_research_doc,
        node=research_doc_leaf,
        sources=research_urls_safe,
        additional_instruction=(
            "Prefer evidence such as named projects, publication citations, conference papers, or formal research "
            "collaboration descriptions rather than generic statements."
        ),
        extra_prerequisites=[research_url_exists],
    )

    # Timeframe (2023-2026)
    timeframe_node = evaluator.add_parallel(
        id=f"u{index}_research_timeframe",
        desc="Documented research activity falls within 2023-2026 period",
        parent=lunar_node,
        critical=True,
    )

    timeframe_leaf = evaluator.add_leaf(
        id=f"u{index}_timeframe_verification",
        desc="Evidence shows research activity dated within 2023-2026",
        parent=timeframe_node,
        critical=True,
    )
    claim_timeframe = (
        "The provided page(s) show dates indicating lunar-related research activity in the years 2023, 2024, 2025, or 2026."
    )
    await evaluator.verify(
        claim=claim_timeframe,
        node=timeframe_leaf,
        sources=research_urls_safe,
        additional_instruction=(
            "Check for explicit publication dates, project timelines, news release dates, or update timestamps within 2023-2026. "
            "Minor date formatting variations are acceptable."
        ),
        extra_prerequisites=[research_url_exists],
    )

    # ---------------- Requirement 3: Graduate programs (Master's/PhD) in relevant fields ----------------
    grad_node = evaluator.add_parallel(
        id=f"u{index}_graduate_programs",
        desc="University offers relevant graduate degree programs",
        parent=uni_node,
        critical=True,
    )

    prog_level_node = evaluator.add_parallel(
        id=f"u{index}_program_level",
        desc="Programs offered at Master's and/or PhD level in relevant STEM fields",
        parent=grad_node,
        critical=True,
    )

    program_urls_safe = _safe_list(uni.program_urls)
    program_url_exists = evaluator.add_custom_node(
        result=len(program_urls_safe) > 0,
        id=f"u{index}_program_verification_url",
        desc="Verification URL provided from university graduate program website",
        parent=prog_level_node,
        critical=True,
    )

    degree_doc_leaf = evaluator.add_leaf(
        id=f"u{index}_degree_level_documentation",
        desc="Graduate program documentation confirms Master's or PhD level offerings",
        parent=prog_level_node,
        critical=True,
    )
    claim_degree = (
        "The provided page(s) confirm that the university offers graduate degree programs at the Master's (MS) and/or PhD level."
    )
    await evaluator.verify(
        claim=claim_degree,
        node=degree_doc_leaf,
        sources=program_urls_safe,
        additional_instruction=(
            "Look for explicit degree listings such as 'MS', 'M.S.', 'PhD', 'Doctor of Philosophy', or similar on official program pages."
        ),
        extra_prerequisites=[program_url_exists],
    )

    prog_rel_node = evaluator.add_parallel(
        id=f"u{index}_program_relevance",
        desc="Programs in aerospace engineering, planetary science, materials science, or related fields",
        parent=grad_node,
        critical=True,
    )

    field_align_leaf = evaluator.add_leaf(
        id=f"u{index}_field_alignment",
        desc="Program fields align with aerospace, planetary science, materials science, or closely related STEM disciplines",
        parent=prog_rel_node,
        critical=True,
    )
    claim_fields = (
        "The provided page(s) indicate that the graduate program fields include aerospace engineering, planetary science, "
        "materials science, or closely related STEM disciplines."
    )
    await evaluator.verify(
        claim=claim_fields,
        node=field_align_leaf,
        sources=program_urls_safe,
        additional_instruction=(
            "Accept closely related fields such as mechanical engineering with aerospace tracks, space systems, geosciences "
            "with planetary focus, or materials engineering relevant to space applications if stated."
        ),
        extra_prerequisites=[program_url_exists],
    )

    # ---------------- Requirement 4: Research database subscriptions ----------------
    infra_node = evaluator.add_parallel(
        id=f"u{index}_research_infrastructure",
        desc="University maintains institutional research database subscriptions",
        parent=uni_node,
        critical=True,
    )

    db_access_node = evaluator.add_parallel(
        id=f"u{index}_database_access",
        desc="Institutional access to major academic research databases including comprehensive dissertation/thesis repositories",
        parent=infra_node,
        critical=True,
    )

    infra_urls_safe = _safe_list(uni.infrastructure_urls)
    infra_url_exists = evaluator.add_custom_node(
        result=len(infra_urls_safe) > 0,
        id=f"u{index}_infrastructure_verification_url",
        desc="Verification URL or evidence of institutional library research database subscriptions",
        parent=db_access_node,
        critical=True,
    )

    db_sub_evidence_leaf = evaluator.add_leaf(
        id=f"u{index}_database_subscription_evidence",
        desc="Evidence of institutional subscriptions to research databases supporting graduate research",
        parent=db_access_node,
        critical=True,
    )
    claim_db = (
        "The provided page(s) demonstrate the institution's subscriptions or access to major academic research databases "
        "supporting graduate research, including comprehensive dissertation/thesis repositories."
    )
    await evaluator.verify(
        claim=claim_db,
        node=db_sub_evidence_leaf,
        sources=infra_urls_safe,
        additional_instruction=(
            "Look for library resource pages listing ProQuest Dissertations & Theses (Global), Web of Science, Scopus, "
            "EBSCO databases, IEEE Xplore, and similar. Pages can be A-Z database lists or access instructions."
        ),
        extra_prerequisites=[infra_url_exists],
    )

    # ---------------- Requirement 5: Graduate funding for space/aerospace ----------------
    funding_node = evaluator.add_parallel(
        id=f"u{index}_graduate_funding",
        desc="University provides graduate student funding for space research",
        parent=uni_node,
        critical=True,
    )

    funding_avail_node = evaluator.add_parallel(
        id=f"u{index}_funding_availability",
        desc="Fellowships, assistantships, or Space Grant funding available for aerospace/space research",
        parent=funding_node,
        critical=True,
    )

    funding_urls_safe = _safe_list(uni.funding_urls)
    funding_url_exists = evaluator.add_custom_node(
        result=len(funding_urls_safe) > 0,
        id=f"u{index}_funding_verification_url",
        desc="Verification URL showing graduate funding programs in space/aerospace research",
        parent=funding_avail_node,
        critical=True,
    )

    funding_doc_leaf = evaluator.add_leaf(
        id=f"u{index}_funding_program_documentation",
        desc="Specific funding programs for aerospace/space research are documented",
        parent=funding_avail_node,
        critical=True,
    )
    claim_funding = (
        "The provided page(s) document graduate funding opportunities—such as fellowships, research assistantships, or "
        "Space Grant-supported awards—specifically supporting aerospace or space science research."
    )
    await evaluator.verify(
        claim=claim_funding,
        node=funding_doc_leaf,
        sources=funding_urls_safe,
        additional_instruction=(
            "Accept departmental fellowships, RA positions in aerospace/space labs, Space Grant scholarships/fellowships, "
            "and explicit references to space or aerospace research support."
        ),
        extra_prerequisites=[funding_url_exists],
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the NASA lunar universities task and return a structured result summary.
    """
    # Initialize evaluator with a parallel root (non-critical root to allow partial credit aggregation)
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

    # Extract university items
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversityCollectionExtraction,
        extraction_name="universities_extraction",
    )

    # Normalize to at most 4 universities (as required)
    universities = extracted.universities[:4] if extracted and extracted.universities else []
    named_count = _count_named_universities(universities)

    # ---------------- Quantity check (critical) ----------------
    qty_node = evaluator.add_parallel(
        id="quantity_check",
        desc="Solution provides between 3 and 4 universities (not fewer than 3, not more than 4)",
        parent=root,
        critical=True,
    )

    min_count_node = evaluator.add_custom_node(
        result=named_count >= 3,
        id="minimum_count",
        desc="At least 3 universities are identified in the solution",
        parent=qty_node,
        critical=True,
    )

    max_count_node = evaluator.add_custom_node(
        result=named_count <= 4,
        id="maximum_count",
        desc="No more than 4 universities are identified in the solution",
        parent=qty_node,
        critical=True,
    )

    # ---------------- University items collection (non-critical) ----------------
    uni_items_node = evaluator.add_parallel(
        id="university_items",
        desc="Collection of identified universities, each meeting all five requirements",
        parent=root,
        critical=False,
    )

    # Prepare exactly 4 slots; pad with empty items if fewer provided
    while len(universities) < 4:
        universities.append(UniversityItemExtraction())

    # Verify each of up to four universities
    for idx in range(1, 5):  # 1..4
        try:
            await verify_university(
                evaluator=evaluator,
                parent_node=uni_items_node,
                uni=universities[idx - 1],
                index=idx,
            )
        except Exception as e:
            logger.error(f"Error verifying university #{idx}: {e}")

    # Return structured summary
    return evaluator.get_summary()