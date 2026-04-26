import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "quantum_company_breakthrough_2024_2025"
TASK_DESCRIPTION = """Identify a quantum computing company that meets all of the following criteria:

1. The company announced a major breakthrough between July 2024 and December 2025 where it achieved EITHER:
   - Two-qubit gate fidelity of at least 99.5%, OR
   - Qubit coherence time or bit-flip stability time of at least 15 minutes

2. The company uses superconducting quantum computing technology (including cat qubit implementations, which are a type of superconducting approach)

3. The company has its primary headquarters or a major headquarters location in Europe

4. The company was founded or formed in 2018 or later

5. The technical achievement must be documented in an official company press release, blog post, or announcement on the company's website, with specific performance metrics clearly stated

6. The company develops full-stack quantum computers (complete quantum computing systems including both hardware and software), not just individual components

7. The announced achievement represented a record-breaking or industry-leading performance metric in its specific category (either gate fidelity or coherence time) at the time of the announcement

Provide the company name and a reference URL to the official announcement documenting the qualifying technical achievement.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CompanyAnswerExtraction(BaseModel):
    """
    Structured extraction of the user's proposed company and evidence.
    """
    company_name: Optional[str] = None
    announcement_url: Optional[str] = None
    announcement_title: Optional[str] = None
    announcement_date: Optional[str] = None  # Keep as free-form string from answer
    # Metrics (free-form strings; do not coerce to numbers)
    metric_type: Optional[str] = None  # e.g., "two-qubit gate fidelity", "coherence time", "bit-flip stability time"
    fidelity_percentage: Optional[str] = None  # e.g., "99.7%"
    coherence_time: Optional[str] = None       # e.g., "20 minutes", "1200 seconds"
    bit_flip_stability_time: Optional[str] = None  # e.g., "16 minutes"
    # Platform/technology and corporate info
    platform: Optional[str] = None  # e.g., "superconducting", "cat qubits", "transmons", etc.
    headquarters_location: Optional[str] = None  # e.g., "Paris, France"
    founding_year: Optional[str] = None
    full_stack_evidence: Optional[str] = None  # free-form description if mentioned
    record_significance_phrase: Optional[str] = None  # e.g., "record", "world-leading", "best to date"
    # Additional official company URLs explicitly listed in the answer (not inferred)
    supporting_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_company() -> str:
    return """
    Extract the key information about the proposed quantum computing company and its referenced official announcement from the answer. Return JSON with the following fields:

    - company_name: The company name proposed in the answer.
    - announcement_url: The single reference URL that points to the official announcement documenting the technical achievement. Must be an official company website URL and must be explicitly present in the answer. If multiple URLs are provided, choose the one that best matches the announcement of the performance metrics; otherwise return the first official announcement URL mentioned. If none provided, return null.
    - announcement_title: The title of the announcement if mentioned in the answer; otherwise null.
    - announcement_date: The publication date for the announcement if mentioned in the answer; otherwise null. Keep the original format (e.g., "July 22, 2025").
    - metric_type: The specific category of the metric described in the answer (e.g., "two-qubit gate fidelity", "coherence time", "bit-flip stability time"), if mentioned; otherwise null.
    - fidelity_percentage: The fidelity value (e.g., "99.6%") if the answer mentions fidelity; otherwise null.
    - coherence_time: The coherence time (e.g., "15 minutes", "1200 seconds", "0.25 hours") if mentioned; otherwise null.
    - bit_flip_stability_time: The bit-flip stability time if mentioned; otherwise null.
    - platform: The technology/platform the company uses if mentioned (e.g., "superconducting", "cat qubits", "transmon"); otherwise null.
    - headquarters_location: The HQ or major HQ location if mentioned (e.g., "Paris, France"); otherwise null.
    - founding_year: The founding/formation year if mentioned (e.g., "2019"); otherwise null.
    - full_stack_evidence: If the answer states the company develops full-stack quantum computers (hardware+software), copy the phrasing; otherwise null.
    - record_significance_phrase: If the answer states the performance was record-breaking or industry-leading, copy the exact phrase; otherwise null.
    - supporting_urls: An array of any additional official company website URLs explicitly listed in the answer that may help verify platform, HQ, founding year, full-stack status, etc. Only include URLs explicitly present in the answer. If none, return an empty array.

    IMPORTANT:
    - Only extract URLs explicitly present in the answer text. Do not infer or add URLs.
    - Prefer official company domains (e.g., company.com) for announcement_url and supporting_urls.
    - If a URL lacks a protocol, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(primary: Optional[str], extras: Optional[List[str]]) -> List[str]:
    urls: List[str] = []
    if primary and isinstance(primary, str) and primary.strip():
        urls.append(primary.strip())
    if extras:
        for u in extras:
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_company_tree(evaluator: Evaluator, parent_node, extracted: CompanyAnswerExtraction) -> None:
    """
    Build the critical verification tree according to the rubric and run verifications.
    """
    # Top-level critical node
    company_node = evaluator.add_parallel(
        id="Company_Identification",
        desc="Identifies a quantum computing company meeting all specified criteria and provides the required reference URL.",
        parent=parent_node,
        critical=True,
    )

    # Aggregate sources (the announcement URL + any supporting official URLs)
    all_sources = _combine_sources(extracted.announcement_url, extracted.supporting_urls)

    # --------------------------- Solution Deliverables ----------------------- #
    deliverables_node = evaluator.add_parallel(
        id="Solution_Deliverables",
        desc="Provides the company name and a reference URL that points to the official company announcement documenting the qualifying technical achievement.",
        parent=company_node,
        critical=True,
    )

    # Company name provided
    evaluator.add_custom_node(
        result=bool(extracted.company_name and extracted.company_name.strip()),
        id="Company_Name_Provided",
        desc="Company name is provided in the answer.",
        parent=deliverables_node,
        critical=True
    )

    # Reference URL provided
    evaluator.add_custom_node(
        result=bool(extracted.announcement_url and extracted.announcement_url.strip().startswith(("http://", "https://"))),
        id="Reference_URL_Provided",
        desc="Reference URL to the official announcement is provided in the answer.",
        parent=deliverables_node,
        critical=True
    )

    # --------------------------- Official Documentation ---------------------- #
    official_doc_node = evaluator.add_sequential(
        id="Official_Documentation",
        desc="The technical achievement is documented in an official company press release, blog post, or announcement with specific metrics stated.",
        parent=company_node,
        critical=True
    )

    # Ensure the URL exists before attempting URL verifications
    evaluator.add_custom_node(
        result=bool(extracted.announcement_url and extracted.announcement_url.strip()),
        id="Official_URL_Exists",
        desc="Announcement URL exists to enable official documentation checks.",
        parent=official_doc_node,
        critical=True
    )

    # Check that the URL is an official company site page (press release/blog/announcement)
    url_official_node = evaluator.add_leaf(
        id="URL_Is_Official_Company_Site",
        desc="The provided URL is an official company website page (press release/blog/announcement) documenting the achievement.",
        parent=official_doc_node,
        critical=True
    )
    url_official_claim = (
        f"This URL is an official page on the company's own website (press release/blog/announcement) "
        f"that documents the technical achievement."
    )
    await evaluator.verify(
        claim=url_official_claim,
        node=url_official_node,
        sources=extracted.announcement_url,
        additional_instruction=(
            "Verify the URL domain and page content branding. "
            f"If available, confirm it belongs to the company named '{extracted.company_name or 'the company'}'."
        )
    )

    # Check that specific numerical metrics are clearly stated on the page
    metrics_stated_node = evaluator.add_leaf(
        id="Metrics_Clearly_Stated",
        desc="The announcement explicitly states specific performance metrics (numbers) for fidelity and/or coherence/stability time.",
        parent=official_doc_node,
        critical=True
    )
    metrics_stated_claim = (
        "The announcement explicitly states quantitative performance metrics (numbers) for either two-qubit gate fidelity "
        "and/or qubit coherence/bit-flip stability time."
    )
    await evaluator.verify(
        claim=metrics_stated_claim,
        node=metrics_stated_node,
        sources=extracted.announcement_url,
        additional_instruction=(
            "Look for explicit numbers like '99.5%' or durations like '15 minutes'/'900 seconds' on the announcement page."
        )
    )

    # --------------------------- Technical Milestone ------------------------- #
    tech_milestone_node = evaluator.add_sequential(
        id="Technical_Milestone",
        desc="The company announced (between July 2024 and December 2025) a major breakthrough achieving either (a) two-qubit gate fidelity ≥ 99.5% OR (b) coherence/stability time ≥ 15 minutes.",
        parent=company_node,
        critical=True
    )

    # Date in range (July 1, 2024 to December 31, 2025 inclusive)
    milestone_date_node = evaluator.add_leaf(
        id="Milestone_Date_Range",
        desc="The announcement was published between July 1, 2024 and December 31, 2025 (inclusive).",
        parent=tech_milestone_node,
        critical=True
    )
    date_claim = (
        "The announcement page shows a publication date between July 1, 2024 and December 31, 2025 inclusive. "
        "If there are multiple dates (e.g., updated), use the original publication date."
    )
    await evaluator.verify(
        claim=date_claim,
        node=milestone_date_node,
        sources=extracted.announcement_url,
        additional_instruction="Confirm the initial publication date displayed on the page falls within the specified window."
    )

    # Metric condition (either fidelity >= 99.5% OR coherence/bit-flip stability time >= 15 minutes)
    metric_condition_node = evaluator.add_leaf(
        id="Milestone_Metric_Condition",
        desc="The announcement satisfies: fidelity ≥ 99.5% OR coherence/bit-flip stability time ≥ 15 minutes.",
        parent=tech_milestone_node,
        critical=True
    )
    metric_condition_claim = (
        "Based on the announcement content, the achievement satisfies at least one of the following: "
        "(i) two-qubit gate fidelity is at least 99.5%, OR "
        "(ii) qubit coherence time or bit-flip stability time is at least 15 minutes (>= 900 seconds)."
    )
    await evaluator.verify(
        claim=metric_condition_claim,
        node=metric_condition_node,
        sources=extracted.announcement_url,
        additional_instruction=(
            f"Use the page's explicit numbers. Accept equivalent units (e.g., seconds to minutes). "
            f"From the answer extraction: fidelity='{extracted.fidelity_percentage}', "
            f"coherence='{extracted.coherence_time}', stability='{extracted.bit_flip_stability_time}'. "
            f"Treat 'cat qubit lifetime' as coherence/stability time."
        )
    )

    # --------------------------- Technology Platform ------------------------- #
    tech_platform_node = evaluator.add_leaf(
        id="Technology_Platform",
        desc="The company uses superconducting quantum computing technology (including cat qubits).",
        parent=company_node,
        critical=True
    )
    platform_claim = (
        "According to official sources, the company's quantum computing technology platform is superconducting. "
        "Cat qubit implementations count as superconducting."
    )
    await evaluator.verify(
        claim=platform_claim,
        node=tech_platform_node,
        sources=all_sources,
        additional_instruction=(
            "Look for terms like 'superconducting qubits', 'transmon', 'cat qubits', or similar on official pages. "
            "It is acceptable if the announcement page or other official pages mention it."
        )
    )

    # --------------------------- European Headquarters ----------------------- #
    european_hq_node = evaluator.add_leaf(
        id="European_Headquarters",
        desc="The company has its primary headquarters or a major headquarters location in Europe.",
        parent=company_node,
        critical=True
    )
    hq_claim = (
        "According to official pages, the company has its primary headquarters OR a major headquarters location in Europe."
    )
    await evaluator.verify(
        claim=hq_claim,
        node=european_hq_node,
        sources=all_sources,
        additional_instruction=(
            f"If the announcement includes a dateline such as a European city (e.g., '{extracted.headquarters_location or ''}'), "
            "that can serve as evidence of a major European HQ location if the page indicates it. "
            "Prefer explicit statements like 'headquartered in [City, Country]' on official pages."
        )
    )

    # --------------------------- Company Founding ---------------------------- #
    founding_node = evaluator.add_leaf(
        id="Company_Founding",
        desc="The company was founded or formed in 2018 or later.",
        parent=company_node,
        critical=True
    )
    founding_claim = (
        "According to official pages, the company was founded or formed in 2018 or later (year >= 2018). "
        "For mergers or spin-outs, use the formal formation year."
    )
    await evaluator.verify(
        claim=founding_claim,
        node=founding_node,
        sources=all_sources,
        additional_instruction=(
            f"If a founding year is explicitly stated (e.g., '{extracted.founding_year or ''}'), "
            "confirm it is 2018 or later."
        )
    )

    # --------------------------- Full-stack Development ---------------------- #
    fullstack_node = evaluator.add_leaf(
        id="Full_Stack_Development",
        desc="The company develops full-stack quantum computers (complete systems including both hardware and software).",
        parent=company_node,
        critical=True
    )
    fullstack_claim = (
        "According to official pages, the company develops full-stack quantum computers (i.e., complete systems that include "
        "quantum hardware and the relevant system software), not just individual components."
    )
    await evaluator.verify(
        claim=fullstack_claim,
        node=fullstack_node,
        sources=all_sources,
        additional_instruction=(
            "Look for language like 'full-stack', 'end-to-end system', "
            "'integrated hardware and software', or descriptions of a complete system offering."
        )
    )

    # --------------------------- Achievement Significance -------------------- #
    significance_node = evaluator.add_leaf(
        id="Achievement_Significance",
        desc="At the time of the announcement, the achievement was record-breaking or industry-leading.",
        parent=company_node,
        critical=True
    )
    significance_claim = (
        "The official announcement states that, at the time of publication, the performance metric was record-breaking or "
        "industry-leading (e.g., 'record', 'world-leading', 'best to date', 'industry-leading')."
    )
    await evaluator.verify(
        claim=significance_claim,
        node=significance_node,
        sources=extracted.announcement_url,
        additional_instruction=(
            f"Check for explicit phrases on the announcement page such as '{extracted.record_significance_phrase or ''}' "
            "or equivalent wording indicating industry-leading or record performance."
        )
    )


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
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'quantum_company_breakthrough_2024_2025' task.
    """
    # Initialize evaluator (root is non-critical)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_company(),
        template_class=CompanyAnswerExtraction,
        extraction_name="company_extraction"
    )

    # Optional: record rubric details into custom info for transparency
    evaluator.add_custom_info(
        info={
            "criteria": [
                "Announcement between Jul 2024 and Dec 2025 with either fidelity ≥ 99.5% or coherence/stability ≥ 15 min",
                "Superconducting technology (including cat qubits)",
                "HQ or major HQ in Europe",
                "Founded/formed in 2018 or later",
                "Official announcement on company website with explicit metrics",
                "Full-stack quantum computers (hardware + software)",
                "Record-breaking/industry-leading at the time",
                "Answer provides company name and the announcement URL"
            ]
        },
        info_type="rubric_summary"
    )

    # Build verification tree and run checks
    await build_and_verify_company_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()