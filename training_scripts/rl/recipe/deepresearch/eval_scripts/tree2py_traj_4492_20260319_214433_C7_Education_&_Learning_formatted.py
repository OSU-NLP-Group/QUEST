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
TASK_ID = "public_r1_universities_4_states"
TASK_DESCRIPTION = """
Find four public universities in the United States from four different states that meet all of the following criteria:

1. Each university must be a public institution (state-funded, not private)
2. Each university must hold R1 classification (Research 1 - Doctoral Universities with Very High Research Activity) according to the Carnegie Classification
3. Each university must be regionally accredited by a recognized U.S. regional accrediting agency
4. Each university must have a total enrollment between 30,000 and 50,000 students (based on most recent 2025 or fall 2024 data)
5. Each university's in-state annual tuition and fees for the 2025-26 academic year must be between $8,000 and $12,000
6. Each university must offer at least 100 different undergraduate degree programs or majors

For each of the four universities you identify, provide:
- The official university name
- The state where it is located
- Current total enrollment (with the year of the data)
- The 2025-26 in-state annual tuition and fees amount
- The number of undergraduate degree programs/majors offered
- The regional accrediting body
- A link to the university's official website homepage
- A link to a page confirming its R1 classification or research university status
- A link to the university's tuition and fees information page for 2025-26
- A link to the university's page listing undergraduate programs or majors

All four universities must be from different U.S. states.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None

    enrollment_total: Optional[str] = None           # e.g., "38,200"
    enrollment_year: Optional[str] = None            # e.g., "2025" or "Fall 2024"
    enrollment_url: Optional[str] = None             # optional, if provided in the answer

    tuition_instate_2025_26: Optional[str] = None    # e.g., "$10,200"
    tuition_url_2025_26: Optional[str] = None

    programs_count_undergrad: Optional[str] = None   # e.g., "120" or "100+"
    programs_url: Optional[str] = None

    accrediting_body: Optional[str] = None           # e.g., "HLC", "SACSCOC"
    accreditation_url: Optional[str] = None          # optional, if provided in the answer

    homepage_url: Optional[str] = None
    r1_status_url: Optional[str] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to six universities mentioned in the answer (we will consider the first four).
    For each university, extract ONLY what is explicitly present in the answer. Do not invent or infer missing information.

    For each university, return the following fields:
    - name: Official university name (string)
    - state: The U.S. state (string; can be full name or USPS abbreviation as written)
    - enrollment_total: The total enrollment number as written (string; keep formatting, e.g., "38,200")
    - enrollment_year: The stated data year, e.g., "2025" or "Fall 2024" (string)
    - enrollment_url: A URL explicitly provided in the answer that supports enrollment (if any); else null
    - tuition_instate_2025_26: The in-state annual tuition and fees for the 2025-26 academic year as written (string)
    - tuition_url_2025_26: A URL to the 2025-26 tuition and fees page
    - programs_count_undergrad: The number of undergraduate degree programs/majors (string; keep as written, e.g., "100+", "over 120")
    - programs_url: URL to the page listing undergraduate programs or majors
    - accrediting_body: The regional accrediting body name or acronym as written (e.g., "HLC", "SACSCOC", "MSCHE", "NECHE", "WSCUC", "NWCCU", "ACCJC")
    - accreditation_url: A URL explicitly provided that confirms institutional accreditation (if any); else null
    - homepage_url: URL to the official university homepage
    - r1_status_url: URL to a page confirming R1 classification or research university status

    STRICT RULES:
    - Only extract URLs that actually appear in the answer (plain URL or markdown link).
    - If any field is not present in the answer, set it to null.
    - Do not normalize numbers; keep formatting and units exactly as written.

    Return a JSON object with a single key "universities" which is an array of these university objects.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def non_empty_urls(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip()]


def fallback_sources_for_uni(uni: UniversityItem) -> List[str]:
    # General fallback list covering most official pages that might contain facts
    return non_empty_urls(
        uni.r1_status_url,
        uni.accreditation_url,
        uni.enrollment_url,
        uni.tuition_url_2025_26,
        uni.programs_url,
        uni.homepage_url,
    )


# --------------------------------------------------------------------------- #
# Verification per university                                                 #
# --------------------------------------------------------------------------- #
async def verify_university(evaluator: Evaluator, parent_node, uni: UniversityItem, index: int) -> None:
    """
    Build verification nodes for one university and run checks.

    Structure (parallel at university level):
    - Institution Type & R1 (critical, parallel)
        - Public institution (leaf)
        - R1 classification (leaf)
    - Enrollment in range 30k–50k (critical, leaf)
    - Tuition (2025-26 in-state, 8k–12k) (critical, leaf)
    - Programs >= 100 undergrad (critical, leaf)
    - Regional accreditation (critical, leaf)
    """
    uidx = index + 1
    uni_label = uni.name or f"University #{uidx}"

    uni_node = evaluator.add_parallel(
        id=f"university_{uidx}",
        desc=f"{uni_label} verification against task constraints",
        parent=parent_node,
        critical=False  # overall item allows partial credit at root level
    )

    # 1) Institution Type & R1 (split into two atomic leaves under a critical parallel node)
    inst_r1_main = evaluator.add_parallel(
        id=f"u{uidx}_institution_type_r1_main",
        desc=f"U{uidx}: University is public and holds R1 classification",
        parent=uni_node,
        critical=True
    )

    # 1.a) Public institution
    public_leaf = evaluator.add_leaf(
        id=f"u{uidx}_public_institution",
        desc=f"U{uidx}: {uni_label} is a public (state-funded) university",
        parent=inst_r1_main,
        critical=True
    )
    public_claim = f"{uni_label} is a public (state-funded) university (not private)."
    await evaluator.verify(
        claim=public_claim,
        node=public_leaf,
        sources=non_empty_urls(uni.homepage_url, uni.r1_status_url) or fallback_sources_for_uni(uni),
        additional_instruction="Accept clear phrases like 'public research university', 'public university', 'state university', or similar wording on the official site."
    )

    # 1.b) R1 classification
    r1_leaf = evaluator.add_leaf(
        id=f"u{uidx}_r1_classification",
        desc=f"U{uidx}: {uni_label} holds R1 (Very High Research Activity) classification",
        parent=inst_r1_main,
        critical=True
    )
    r1_claim = f"{uni_label} holds R1 classification (Doctoral Universities – Very High Research Activity) per Carnegie Classification."
    await evaluator.verify(
        claim=r1_claim,
        node=r1_leaf,
        sources=non_empty_urls(uni.r1_status_url) or fallback_sources_for_uni(uni),
        additional_instruction="Look for explicit 'R1' or 'Very High Research Activity' mention. A page from Carnegie Classifications or the institution confirming R1 status is acceptable."
    )

    # 2) Enrollment in range [30k, 50k]
    enroll_leaf = evaluator.add_leaf(
        id=f"u{uidx}_enrollment_range",
        desc=f"U{uidx}: Total enrollment is between 30,000 and 50,000 students",
        parent=uni_node,
        critical=True
    )
    if uni.enrollment_year and uni.enrollment_total:
        enroll_claim = f"As of {uni.enrollment_year}, {uni_label}'s total enrollment is between 30,000 and 50,000 students."
    else:
        enroll_claim = f"{uni_label}'s total enrollment is between 30,000 and 50,000 students (most recent 2025 or Fall 2024 data)."
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_leaf,
        sources=non_empty_urls(uni.enrollment_url) or fallback_sources_for_uni(uni),
        additional_instruction="Verify 'total enrollment' (overall headcount including undergrad+grad). Prefer 2025 or Fall 2024. If the official page shows a number within 30k–50k, conclude 'supported'."
    )

    # 3) Tuition 2025-26 in-state annual tuition+fees in [8k, 12k]
    tuition_leaf = evaluator.add_leaf(
        id=f"u{uidx}_tuition_range_2025_26",
        desc=f"U{uidx}: In-state annual tuition and fees for 2025-26 are between $8,000 and $12,000",
        parent=uni_node,
        critical=True
    )
    tuition_claim = f"For the 2025-26 academic year, {uni_label}'s in-state undergraduate annual tuition and mandatory fees total between $8,000 and $12,000."
    await evaluator.verify(
        claim=tuition_claim,
        node=tuition_leaf,
        sources=non_empty_urls(uni.tuition_url_2025_26),
        additional_instruction=(
            "Use the official tuition/fees page for 2025-26. If amounts are shown per semester or per credit, "
            "convert to the typical annual load (two main semesters, full-time undergrad) to judge whether the total "
            "tuition+mandatory fees fall within $8k–$12k. Ignore out-of-state and graduate rates."
        )
    )

    # 4) Programs: at least 100 undergraduate majors/programs
    programs_leaf = evaluator.add_leaf(
        id=f"u{uidx}_programs_100plus",
        desc=f"U{uidx}: Offers at least 100 undergraduate degree programs or majors",
        parent=uni_node,
        critical=True
    )
    programs_claim = f"{uni_label} offers at least 100 distinct undergraduate majors or bachelor's degree programs."
    await evaluator.verify(
        claim=programs_claim,
        node=programs_leaf,
        sources=non_empty_urls(uni.programs_url) or fallback_sources_for_uni(uni),
        additional_instruction="Look for explicit counts like '100+ majors', 'over 120 majors', or a listing implying at least 100 distinct undergraduate programs."
    )

    # 5) Regional accreditation by recognized agency
    accred_leaf = evaluator.add_leaf(
        id=f"u{uidx}_regional_accreditation",
        desc=f"U{uidx}: Institution is regionally accredited by a recognized U.S. regional accrediting agency",
        parent=uni_node,
        critical=True
    )
    # Include accrediting body name if provided to tighten the check
    if uni.accrediting_body:
        accred_claim = (
            f"{uni_label} is institutionally accredited by {uni.accrediting_body}, which is a recognized U.S. regional accrediting agency."
        )
    else:
        accred_claim = (
            f"{uni_label} is institutionally accredited by a recognized U.S. regional accrediting agency (e.g., HLC, MSCHE, NECHE, SACSCOC, WSCUC, NWCCU, ACCJC)."
        )
    await evaluator.verify(
        claim=accred_claim,
        node=accred_leaf,
        sources=non_empty_urls(uni.accreditation_url) or fallback_sources_for_uni(uni),
        additional_instruction=(
            "Confirm the institution's regional accreditation on an official page. The accrediting body should be one of: "
            "HLC, MSCHE, NECHE, SACSCOC, WSCUC, NWCCU, ACCJC. If the page confirms accreditation by one of these, pass."
        )
    )


# --------------------------------------------------------------------------- #
# Geographic diversity check                                                  #
# --------------------------------------------------------------------------- #
async def verify_geographic_diversity(evaluator: Evaluator, parent_node, universities: List[UniversityItem]) -> None:
    """
    Verify that the first four universities are from four different U.S. states.
    """
    geo_leaf = evaluator.add_leaf(
        id="geographic_diversity",
        desc="The four universities are from four different U.S. states",
        parent=parent_node,
        critical=True
    )
    # Build a clear claim that the LLM can reason about (handles 'CA' vs 'California')
    pairs_str = "; ".join(
        [f"{(u.name or f'University #{i+1}')} — {(u.state or 'UNKNOWN')}" for i, u in enumerate(universities[:4])]
    )
    claim = (
        f"Consider these university-state pairs: {pairs_str}. Confirm that the four universities are located in four different U.S. states. "
        "Treat state abbreviations and full names as the same state (e.g., 'CA' equals 'California')."
    )
    await evaluator.verify(
        claim=claim,
        node=geo_leaf,
        sources=None,  # pure logical check; no web evidence needed
        additional_instruction="If any two universities are in the same state (regardless of abbreviation vs full name), mark incorrect."
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
    Evaluate an answer for the 'public R1 universities from 4 different states' task.
    """
    # Initialize evaluator with root as PARALLEL aggregation
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

    # Extract structured university info
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Keep first four items; pad with empty placeholders if fewer
    unis: List[UniversityItem] = list(extracted.universities[:4])
    while len(unis) < 4:
        unis.append(UniversityItem())

    # Add a brief custom info note about how many were found
    evaluator.add_custom_info(
        {"reported_universities_in_answer": len(extracted.universities)},
        info_type="extraction_stats",
        info_name="count_summary"
    )

    # Geographic diversity (critical at root)
    await verify_geographic_diversity(evaluator, root, unis)

    # University checks (each parallel, non-critical at root to allow partial credit across items)
    for idx, uni in enumerate(unis):
        await verify_university(evaluator, root, uni, idx)

    # Return evaluation summary
    return evaluator.get_summary()