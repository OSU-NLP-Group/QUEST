import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_theater_seats_per_capita"
TASK_DESCRIPTION = """
A touring Broadway production company is conducting market research to identify cities with the strongest theater infrastructure for potential tour stops. They specifically want to know about the theater district in the United States that has the highest concentration of theater seats relative to population size.

Identify the theater district in the United States that ranks #1 for the highest number of theater seats per capita, and provide the following information:

1. The official name of the theater district
2. The city and state where this district is located
3. The total number of theater seats across all venues within this district
4. Confirmation that this district indeed holds the #1 ranking in the United States for theater seats per capita
5. The approximate number of annual performances hosted by this district
6. The minimum seating capacity required for a theater to be classified as a "Broadway theater"
7. Based on available information, an assessment of approximately how many individual theaters in this district would meet the Broadway theater classification threshold
8. URL references from search results that verify the per-capita ranking and other key claims
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DistrictExtraction(BaseModel):
    # Core fields requested by the task
    district_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    total_seats: Optional[str] = None
    annual_performances: Optional[str] = None
    broadway_min_capacity: Optional[str] = None  # e.g., "500", "500 seats", "≥500"
    qualifying_theaters_estimate: Optional[str] = None  # e.g., "6-8", "~7", "about 7"

    # Source URLs – field-specific when possible
    per_capita_sources: List[str] = Field(default_factory=list)              # to verify #1 per-capita ranking
    total_seats_sources: List[str] = Field(default_factory=list)             # to verify total seats
    performances_sources: List[str] = Field(default_factory=list)            # to verify annual performances
    broadway_definition_sources: List[str] = Field(default_factory=list)     # to verify Broadway min capacity requirement
    qualifying_estimate_sources: List[str] = Field(default_factory=list)     # to justify the qualifying theaters estimate
    other_sources: List[str] = Field(default_factory=list)                   # any other URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_district() -> str:
    return """
    Extract the following information exactly as presented in the answer. If a field is missing, return null (for single values) or an empty list (for URL arrays). Preserve approximate wording for quantities (e.g., "about 12,000", "roughly 500+", "6–8") as strings.

    Fields to extract:
    - district_name: The official or commonly used name of the theater district identified as #1 for seats per capita in the U.S.
    - city: The city where this district is located.
    - state: The U.S. state where this district is located.
    - total_seats: The total number of theater seats across all venues in the district (keep formatting as in the answer, e.g., "about 15,000", "15k", "15,000+").
    - annual_performances: The approximate number of annual performances hosted by this district (string, preserve approximations like "~1,200", "1,000+", etc.).
    - broadway_min_capacity: The minimum seating capacity required for a theater to be classified as a "Broadway theater" (string, e.g., "500", "500 seats", "≥500").
    - qualifying_theaters_estimate: The approximate number of individual theaters in this district that meet the Broadway threshold (string, e.g., "6-8", "~7").

    URL sources (extract only actual URLs explicitly present in the answer text):
    - per_capita_sources: URLs that directly support the claim that this district is ranked #1 in the U.S. for theater seats per capita.
    - total_seats_sources: URLs supporting the total seat count across all venues in the district.
    - performances_sources: URLs supporting the approximate annual performances figure.
    - broadway_definition_sources: URLs supporting the minimum seat requirement for Broadway classification.
    - qualifying_estimate_sources: URLs used to justify the estimated number of qualifying theaters (e.g., venue capacity pages, official listings).
    - other_sources: Any other URLs cited in the answer that could support the above facts.

    Important:
    - Do not invent URLs. Only extract URLs that are explicitly present in the answer (plain or markdown links).
    - If a URL is missing the protocol, prepend http:// as instructed by the system rules.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def collect_all_sources(ex: DistrictExtraction) -> List[str]:
    all_urls = []
    all_urls.extend(ex.per_capita_sources or [])
    all_urls.extend(ex.total_seats_sources or [])
    all_urls.extend(ex.performances_sources or [])
    all_urls.extend(ex.broadway_definition_sources or [])
    all_urls.extend(ex.qualifying_estimate_sources or [])
    all_urls.extend(ex.other_sources or [])
    return _dedup_urls(all_urls)


def pick_sources(preferred: List[str], fallback: List[str]) -> List[str]:
    if preferred and len(preferred) > 0:
        return _dedup_urls(preferred)
    return _dedup_urls(fallback)


def nz(s: Optional[str], default: str = "UNKNOWN") -> str:
    return s.strip() if s else default


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_basic_identification_checks(evaluator: Evaluator, parent, ex: DistrictExtraction) -> None:
    node = evaluator.add_parallel(
        id="Basic_Identification",
        desc="Identify the basic information about the theater district",
        parent=parent,
        critical=True
    )

    district_name = nz(ex.district_name)
    city = nz(ex.city)
    state = nz(ex.state)
    all_src = collect_all_sources(ex)

    # District_Name
    dn_leaf = evaluator.add_leaf(
        id="District_Name",
        desc="Provide the official name of the theater district",
        parent=node,
        critical=True
    )
    dn_claim = f"The theater district's recognized/official name is '{district_name}'."
    await evaluator.verify(
        claim=dn_claim,
        node=dn_leaf,
        sources=all_src,
        additional_instruction=(
            "Confirm that reputable sources refer to the district by this name (minor capitalization or 'theatre/theater' variations are acceptable). "
            "If multiple names exist, accept if this is a commonly used or official designation for the district."
        )
    )

    # City_Location
    city_leaf = evaluator.add_leaf(
        id="City_Location",
        desc="Specify the city where the district is located",
        parent=node,
        critical=True
    )
    city_claim = f"The {district_name} theater district is located in the city of {city}."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=all_src,
        additional_instruction="Verify the district's city as stated by reliable sources (city government pages, reputable articles, or Wikipedia)."
    )

    # State_Location
    state_leaf = evaluator.add_leaf(
        id="State_Location",
        desc="Specify the state where the district is located",
        parent=node,
        critical=True
    )
    state_claim = f"The {district_name} theater district is located in the U.S. state of {state}."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=all_src,
        additional_instruction="Verify the district's state as stated by reliable sources."
    )


async def build_capacity_spec_checks(evaluator: Evaluator, parent, ex: DistrictExtraction) -> None:
    node = evaluator.add_parallel(
        id="Capacity_Specifications",
        desc="Provide detailed capacity information about the district",
        parent=parent,
        critical=True
    )

    district_name = nz(ex.district_name)
    all_src = collect_all_sources(ex)

    # Total_Seats
    ts_leaf = evaluator.add_leaf(
        id="Total_Seats",
        desc="State the total number of theater seats across all venues in the district",
        parent=node,
        critical=True
    )
    ts_claim = f"The total number of theater seats across all venues in the {district_name} district is approximately {nz(ex.total_seats)}."
    await evaluator.verify(
        claim=ts_claim,
        node=ts_leaf,
        sources=pick_sources(ex.total_seats_sources, all_src),
        additional_instruction=(
            "Be lenient to approximations and phrasing like 'about', 'roughly', 'over', or '+'. "
            "If sources provide a close number or explicit total consistent with the claim, consider it supported."
        )
    )

    # Per_Capita_Ranking
    rank_leaf = evaluator.add_leaf(
        id="Per_Capita_Ranking",
        desc="Verify that this district ranks #1 in the United States for highest number of theater seats per capita",
        parent=node,
        critical=True
    )
    rank_claim = f"The {district_name} theater district ranks #1 in the United States for the highest number of theater seats per capita."
    await evaluator.verify(
        claim=rank_claim,
        node=rank_leaf,
        sources=pick_sources(ex.per_capita_sources, all_src),
        additional_instruction=(
            "Focus strictly on a per-capita ranking within the United States and at the district level (not entire cities). "
            "Allow reasonable synonyms like 'most seats per capita' or 'highest concentration of theater seats per capita'. "
            "If a source is entirely irrelevant or does not support #1 at the district level in the U.S., mark as not supported."
        )
    )


async def build_performance_metrics_checks(evaluator: Evaluator, parent, ex: DistrictExtraction) -> None:
    node = evaluator.add_parallel(
        id="Performance_Metrics",
        desc="Provide operational performance data for the district",
        parent=parent,
        critical=False
    )

    district_name = nz(ex.district_name)
    all_src = collect_all_sources(ex)

    # Annual_Performances (non-critical)
    ap_leaf = evaluator.add_leaf(
        id="Annual_Performances",
        desc="State the approximate number of annual performances hosted by the district",
        parent=node,
        critical=False
    )
    ap_claim = f"The {district_name} district hosts approximately {nz(ex.annual_performances)} performances per year."
    await evaluator.verify(
        claim=ap_claim,
        node=ap_leaf,
        sources=pick_sources(ex.performances_sources, all_src),
        additional_instruction="Accept approximations (e.g., ~, about, +). Verify the number is supported by the cited source(s)."
    )


async def build_broadway_classification_checks(evaluator: Evaluator, parent, ex: DistrictExtraction) -> None:
    # Set this group as non-critical to allow partial credit while still gating on the definition leaf
    node = evaluator.add_parallel(
        id="Broadway_Classification_Analysis",
        desc="Analyze theaters based on Broadway classification standards",
        parent=parent,
        critical=False
    )

    district_name = nz(ex.district_name)
    min_cap_text = nz(ex.broadway_min_capacity)
    all_src = collect_all_sources(ex)

    # Broadway_Definition (critical within this group)
    bd_leaf = evaluator.add_leaf(
        id="Broadway_Definition",
        desc="State the minimum seating capacity requirement for Broadway theater classification",
        parent=node,
        critical=True
    )
    # Frame as exact threshold requirement
    bd_claim = f"The minimum seating capacity required for a Broadway theater is {min_cap_text}."
    await evaluator.verify(
        claim=bd_claim,
        node=bd_leaf,
        sources=pick_sources(ex.broadway_definition_sources, all_src),
        additional_instruction=(
            "The widely recognized threshold for 'Broadway theaters' in NYC is 500 seats (i.e., '500 or more'). "
            "Accept equivalent phrasings like '500 seats or greater'. Verify the provided value matches reputable sources."
        )
    )

    # Qualifying_Theaters_Estimate (non-critical)
    qte_leaf = evaluator.add_leaf(
        id="Qualifying_Theaters_Estimate",
        desc="Provide an estimate or analysis of how many theaters in the district meet Broadway classification requirements",
        parent=node,
        critical=False
    )
    qte_claim = (
        f"Approximately {nz(ex.qualifying_theaters_estimate)} theaters in the {district_name} district meet or exceed the "
        f"Broadway classification seating threshold of {min_cap_text}."
    )
    await evaluator.verify(
        claim=qte_claim,
        node=qte_leaf,
        sources=pick_sources(ex.qualifying_estimate_sources, all_src),
        additional_instruction=(
            "Check whether sources reasonably support the stated estimate (e.g., via venue capacity pages or official listings). "
            "Allow approximations or ranges. If no support is found in the provided links, mark as not supported."
        )
    )


async def build_documentation_check(evaluator: Evaluator, parent, ex: DistrictExtraction) -> None:
    # Documentation existence check as a binary custom node (critical)
    has_core_docs = len(ex.per_capita_sources or []) > 0
    doc_node = evaluator.add_custom_node(
        result=has_core_docs,
        id="Documentation",
        desc="Provide URL references that support the per-capita ranking claim and other key facts",
        parent=parent,
        critical=True
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'highest theater seats per capita district' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Keep sequential as specified by the rubric
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

    # Root node of this task (make root non-critical to allow mixed criticality below)
    main = evaluator.add_sequential(
        id="Performing_Arts_District_Analysis",
        desc="Complete analysis of the theater district in the United States that ranks #1 for highest number of theater seats per capita",
        parent=root,
        critical=False
    )

    # 1) Extraction
    extraction: DistrictExtraction = await evaluator.extract(
        prompt=prompt_extract_district(),
        template_class=DistrictExtraction,
        extraction_name="district_extraction"
    )

    # 2) Build verification subtrees (respecting the rubric structure and criticalities)
    await build_basic_identification_checks(evaluator, main, extraction)
    await build_capacity_spec_checks(evaluator, main, extraction)
    await build_performance_metrics_checks(evaluator, main, extraction)
    await build_broadway_classification_checks(evaluator, main, extraction)
    await build_documentation_check(evaluator, main, extraction)

    # 3) Return summary
    return evaluator.get_summary()