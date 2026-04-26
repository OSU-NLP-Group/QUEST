import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "non_us_superconducting_100q_2025_single_system"
TASK_DESCRIPTION = (
    "In 2025, multiple superconducting quantum computing systems with 100 or more qubits were announced by organizations outside the United States. "
    "Identify one such system, specifying the developing organization and its location, the number of qubits, the month of public announcement, and the key technical achievement or milestone associated with it. "
    "Provide at least one verifiable reference URL from an official source."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SystemInfo(BaseModel):
    system_name: Optional[str] = None
    organization: Optional[str] = None
    location_text: Optional[str] = None
    location_city: Optional[str] = None
    location_country: Optional[str] = None
    architecture: Optional[str] = None
    qubit_count: Optional[str] = None
    announcement_month: Optional[str] = None  # e.g., "January", "Feb", "Aug"
    announcement_year: Optional[str] = None   # e.g., "2025"
    milestone: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_system_info() -> str:
    return """
    Extract exactly one qualifying superconducting quantum computing system from the answer (choose the first clearly described candidate if multiple are present). 
    Return the following fields, using values exactly as they appear in the answer text:

    - system_name: The specific quantum computing system name/identifier (e.g., chip/system name).
    - organization: The developing organization or institution responsible for the system.
    - location_text: Free-form location string if provided (e.g., "Beijing, China" or "Tsukuba, Japan"). If absent, return null.
    - location_city: The city (or region) if explicitly stated. If absent, return null.
    - location_country: The country if explicitly stated (prefer full country names like "China", "Japan", "Germany", "Switzerland"). If absent, return null.
    - architecture: The architecture or qubit modality as stated (e.g., "superconducting", "transmon", "flux qubits"). If absent, return null.
    - qubit_count: The stated physical qubit count for this system as it appears (e.g., "127", "100+", "about 120"). If absent, return null.
    - announcement_month: The month of the public announcement/reveal as stated (e.g., "January", "Feb", "September"). If absent, return null.
    - announcement_year: The year of announcement/reveal as stated (e.g., "2025"). If absent, return null.
    - milestone: A concrete technical achievement or milestone associated with the announcement (e.g., scale-up, new benchmark, error-correction progress). If absent, return null.
    - source_urls: An array of all URLs cited for this system (press releases, official announcements, or peer-reviewed publications). 
        Extract actual URLs only (including markdown links). If no URLs are present, return an empty array.

    Important:
    - Do not invent any information. If a field is missing in the answer, return null (or [] for source_urls).
    - For URL extraction, only return valid complete URLs. Ignore malformed links.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
US_SYNONYMS = {
    "united states", "united states of america", "usa", "u.s.", "u.s.a.", "us", "u.s.a", "u s a", "u s"
}

MONTH_ALIASES = {
    "january": "January", "jan": "January",
    "february": "February", "feb": "February",
    "march": "March", "mar": "March",
    "april": "April", "apr": "April",
    "may": "May",
    "june": "June", "jun": "June",
    "july": "July", "jul": "July",
    "august": "August", "aug": "August",
    "september": "September", "sep": "September", "sept": "September",
    "october": "October", "oct": "October",
    "november": "November", "nov": "November",
    "december": "December", "dec": "December",
}
VALID_MONTHS = set(MONTH_ALIASES.values())


def normalize_country(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.strip().lower()
    t = re.sub(r"[^\w\s.-]", "", t)  # remove punctuation
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_non_us_country(country: Optional[str], location_text: Optional[str]) -> bool:
    c = normalize_country(country)
    if c:
        return c not in US_SYNONYMS
    # Try to infer from location_text like "Beijing, China"
    if location_text:
        lt = normalize_country(location_text)
        # If any US synonym appears as a whole word, treat as US
        if lt:
            for us in US_SYNONYMS:
                # simple contains check; robust enough for common cases
                if us in lt:
                    return False
    # If we cannot determine, return False (conservative)
    return False


def parse_qubit_count_to_int(qubit_str: Optional[str]) -> Optional[int]:
    if not qubit_str:
        return None
    s = qubit_str.lower().strip()
    # Common forms like "127", "100+", ">= 100", "over 100", "around 120"
    nums = re.findall(r"\d+", s)
    if not nums:
        return None
    # use the largest number present as "specific count" proxy
    candidates = [int(n) for n in nums]
    if not candidates:
        return None
    return max(candidates)


def is_valid_announcement_month_2025(month_str: Optional[str], year_str: Optional[str]) -> bool:
    if not month_str or not month_str.strip():
        return False
    month_norm = MONTH_ALIASES.get(month_str.strip().lower())
    if month_norm not in VALID_MONTHS:
        return False
    if not year_str:
        return False
    year_clean = re.sub(r"[^\d]", "", year_str)
    return year_clean == "2025"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, info: SystemInfo) -> None:
    # Top-level "Complete_Task" as sequential critical node
    complete_node = evaluator.add_sequential(
        id="Complete_Task",
        desc="Identify one qualifying non-US 100+ qubit superconducting quantum computing system announced in 2025; "
             "provide organization + location, qubit count, announcement month, key milestone; include at least one official verifiable URL.",
        parent=evaluator.root,
        critical=True
    )

    # 1) System & Announcement Details (parallel, critical)
    details_node = evaluator.add_parallel(
        id="System_And_Announcement_Details",
        desc="The answer identifies a qualifying system and provides the required announcement and specification details.",
        parent=complete_node,
        critical=True
    )

    # 1.a System_Name_Provided (existence check)
    evaluator.add_custom_node(
        result=bool(info.system_name and info.system_name.strip()),
        id="System_Name_Provided",
        desc="The answer clearly names/identifies the specific quantum computing system.",
        parent=details_node,
        critical=True
    )

    # 1.b Organization_Provided (existence check)
    evaluator.add_custom_node(
        result=bool(info.organization and info.organization.strip()),
        id="Organization_Provided",
        desc="The answer states the developing organization or institution.",
        parent=details_node,
        critical=True
    )

    # 1.c Organization_Location_Provided_And_NonUS (existence + non-US check)
    loc_provided = bool(
        (info.location_country and info.location_country.strip()) or
        (info.location_text and info.location_text.strip()) or
        (info.location_city and info.location_city.strip())
    )
    non_us = is_non_us_country(info.location_country, info.location_text)
    evaluator.add_custom_node(
        result=loc_provided and non_us,
        id="Organization_Location_Provided_And_NonUS",
        desc="The answer states the organization’s location, and it is outside the United States.",
        parent=details_node,
        critical=True
    )

    # 1.d Architecture_Is_Superconducting (verify via sources if possible)
    arch_node = evaluator.add_leaf(
        id="Architecture_Is_Superconducting",
        desc="The identified system is a superconducting quantum computer (not another architecture).",
        parent=details_node,
        critical=True
    )
    arch_claim = (
        f"The system {repr(info.system_name) if info.system_name else 'described system'} uses a superconducting "
        f"qubit architecture (e.g., superconducting circuits, transmons, flux qubits), not trapped ions, neutral atoms, "
        f"photonic, or semiconductor spin qubits."
    )
    await evaluator.verify(
        claim=arch_claim,
        node=arch_node,
        sources=info.source_urls if info.source_urls else None,
        additional_instruction="Judge based on the webpage(s) whether the device explicitly uses superconducting qubits. "
                               "Look for words like 'superconducting', 'transmon', 'flux qubits', 'Josephson junctions'. "
                               "If sources are absent, use the answer text alone to judge; if unclear, mark as incorrect."
    )

    # 1.e Qubit_Count_Provided_And_AtLeast_100 (existence + >=100 check from the answer-extracted count)
    qubits = parse_qubit_count_to_int(info.qubit_count)
    evaluator.add_custom_node(
        result=(qubits is not None and qubits >= 100),
        id="Qubit_Count_Provided_And_AtLeast_100",
        desc="The answer provides a specific physical qubit count for the system, and it is at least 100.",
        parent=details_node,
        critical=True
    )

    # 1.f Announcement_Month_In_2025_Provided (existence + month validity in 2025)
    evaluator.add_custom_node(
        result=is_valid_announcement_month_2025(info.announcement_month, info.announcement_year),
        id="Announcement_Month_In_2025_Provided",
        desc="The answer provides the month of public announcement/reveal, and it is within calendar year 2025.",
        parent=details_node,
        critical=True
    )

    # 2) Technical_Achievement (parallel, critical)
    tech_node = evaluator.add_parallel(
        id="Technical_Achievement",
        desc="The answer describes the key technical achievement or milestone associated with the announcement.",
        parent=complete_node,
        critical=True
    )

    milestone_node = evaluator.add_leaf(
        id="Milestone_Is_Specific_And_Relevant",
        desc="The answer states a specific, concrete technical achievement/milestone tied to the system/announcement (e.g., qubit-count scale-up, error-correction result, benchmark, architectural innovation).",
        parent=tech_node,
        critical=True
    )
    milestone_text = info.milestone or ""
    milestone_claim = (
        f"The milestone/technical achievement stated for the system {repr(info.system_name) if info.system_name else ''} "
        f"is specific and relevant to its 2025 announcement: {repr(milestone_text)}"
    )
    await evaluator.verify(
        claim=milestone_claim,
        node=milestone_node,
        sources=info.source_urls if info.source_urls else None,
        additional_instruction="Verify that the milestone is concrete and connected to the system/announcement, not a generic description. "
                               "Examples include achieving a particular qubit count, a new benchmark result, or an error-correction milestone. "
                               "If the sources include a press release or publication, confirm that the milestone is mentioned there."
    )

    # 3) Source_Verification (parallel, critical)
    src_node = evaluator.add_parallel(
        id="Source_Verification",
        desc="The answer provides official/verifiable sourcing for the claims.",
        parent=complete_node,
        critical=True
    )

    if not info.source_urls:
        # Explicitly fail when no URLs are provided
        evaluator.add_leaf(
            id="Official_Reference_URL_Provided",
            desc="Provides at least one verifiable reference URL from an official press release, research institution announcement, or peer-reviewed publication documenting the system/announcement.",
            parent=src_node,
            critical=True,
            score=0.0,
            status="failed"
        )
    else:
        official_url_node = evaluator.add_leaf(
            id="Official_Reference_URL_Provided",
            desc="Provides at least one verifiable reference URL from an official press release, research institution announcement, or peer-reviewed publication documenting the system/announcement.",
            parent=src_node,
            critical=True
        )
        url_claim = (
            f"At least one of these URLs is an official source (e.g., organization/company/university/lab official site "
            f"or a peer-reviewed journal page) that documents or announces the 2025 {repr(info.system_name) if info.system_name else 'system'} "
            f"superconducting 100+ qubit system and/or its stated milestone."
        )
        await evaluator.verify(
            claim=url_claim,
            node=official_url_node,
            sources=info.source_urls,
            additional_instruction=(
                "Accept official organization/company/university/lab domains (including press/news subdomains), or peer-reviewed journals "
                "(e.g., nature.com, science.org, aps.org, iopscience.iop.org, ieee.org). "
                "Do not accept general news aggregators, blogs, or third-party media as 'official'. "
                "A single qualifying URL is sufficient to pass."
            )
        )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the non-US superconducting 100+ qubit system announced in 2025 (single system) task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Overall: enforce order Complete_Task -> subgroups
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

    # Extract structured info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_system_info(),
        template_class=SystemInfo,
        extraction_name="system_info"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted_info)

    # Return the summary with verification tree and scores
    return evaluator.get_summary()