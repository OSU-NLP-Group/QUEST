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
TASK_ID = "news_agencies_early_2026"
TASK_DESCRIPTION = """
Identify four major international news wire services or news agencies that provided coverage of significant global events in early 2026. Your selection must include exactly two agencies with headquarters located in North America and exactly two agencies with headquarters located in Europe.

For each of the four agencies, provide the following information:

1. Official Name: The complete, official name of the news wire service or agency
2. Headquarters Address: The complete headquarters address, including street address, city, and country
3. Coverage Evidence: A reference URL from the agency's own website or from a reputable news source that demonstrates the agency provided coverage of at least one major international news event that occurred between January 1, 2026 and March 20, 2026
"""

DATE_RANGE_TEXT = "between January 1, 2026 and March 20, 2026 (inclusive)"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HQAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    full_address: Optional[str] = None


class AgencyItem(BaseModel):
    official_name: Optional[str] = None
    headquarters: Optional[HQAddress] = None
    coverage_urls: List[str] = Field(default_factory=list)


class AgenciesExtraction(BaseModel):
    agencies: List[AgencyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_agencies() -> str:
    return """
    Extract up to four (4) news wire services or news agencies listed in the answer, in the same order they appear.
    For each agency, extract the following fields:
    - official_name: The complete official name of the news wire service or news agency.
    - headquarters: The headquarters address, broken into:
        - street: The street address line, including number if provided (e.g., "200 Liberty Street").
        - city: The city name (e.g., "New York").
        - country: The country name (e.g., "United States" or "USA").
        - full_address: The full headquarters address as a single line if the answer provided it.
    - coverage_urls: An array of one or more URLs that the answer cites as evidence that this agency provided coverage
      of at least one major international news event during early 2026. Only include URLs explicitly present in the answer.
      These may be links to the agency's own website or to reputable news outlets clearly showing the agency's wire/byline.
    
    Rules:
    - Return at most four agencies (if more are mentioned, include only the first four).
    - If fewer than four agencies are provided, include whatever is present.
    - If any field is missing, set it to null (for strings) or an empty array (for URLs).
    - For coverage_urls, include only valid HTTP/HTTPS URLs explicitly present in the answer; do not infer or create URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def sanitize_urls(urls: List[str]) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for u in urls or []:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            continue
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


EUROPE_TERMS = {
    # Countries (lowercase)
    "united kingdom", "uk", "england", "scotland", "wales", "northern ireland", "great britain", "britain",
    "ireland", "republic of ireland",
    "france", "germany", "spain", "italy", "portugal", "netherlands", "belgium", "luxembourg",
    "switzerland", "austria", "denmark", "norway", "sweden", "finland", "iceland",
    "poland", "czech", "slovakia", "hungary", "romania", "bulgaria", "slovenia", "croatia",
    "serbia", "bosnia", "montenegro", "kosovo", "albania", "north macedonia",
    "greece", "cyprus", "malta", "estonia", "latvia", "lithuania", "ukraine", "moldova",
    # Major cities/regions
    "london", "paris", "berlin", "madrid", "rome", "milan", "lisbon", "amsterdam", "brussels",
    "luxembourg city", "zurich", "geneva", "vienna", "copenhagen", "oslo", "stockholm", "helsinki",
    "reykjavik", "warsaw", "prague", "bratislava", "budapest", "bucharest", "sofia",
    "ljubljana", "zagreb", "belgrade", "sarajevo", "podgorica", "tirana", "skopje",
    "athens", "nicosia", "valletta", "tallinn", "riga", "vilnius", "kyiv", "chisinau"
}

NORTH_AMERICA_TERMS = {
    # Countries
    "united states", "usa", "u.s.", "u.s.a.", "us",
    "canada", "mexico",
    # Major cities/regions
    "new york", "washington", "washington dc", "washington, dc", "los angeles", "chicago", "atlanta",
    "toronto", "ottawa", "montreal", "vancouver", "calgary",
    "mexico city", "guadalajara", "monterrey", "tijuana",
    # States/provinces abbreviations (commonly appear in addresses)
    "ny", "ca", "dc", "il", "ga", "tx", "ma", "wa", "pa", "fl", "on", "qc", "bc", "ab"
}


def _text_contains_any(text: str, terms: set) -> bool:
    t = (text or "").lower()
    return any(term in t for term in terms)


def classify_region_from_hq(hq: Optional[HQAddress]) -> str:
    """
    Heuristic classification of HQ into 'North America', 'Europe', or 'Other'.
    """
    if hq is None:
        return "Other"
    fields = " ".join(filter(None, [hq.country, hq.city, hq.street, hq.full_address])).lower()
    if not fields:
        return "Other"
    if _text_contains_any(fields, NORTH_AMERICA_TERMS):
        return "North America"
    if _text_contains_any(fields, EUROPE_TERMS):
        return "Europe"
    return "Other"


def count_regions(agencies: List[AgencyItem]) -> Dict[str, int]:
    counts = {"North America": 0, "Europe": 0, "Other": 0}
    for a in agencies:
        region = classify_region_from_hq(a.headquarters)
        counts[region] = counts.get(region, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Verification subroutine per agency                                          #
# --------------------------------------------------------------------------- #
async def verify_agency(
    evaluator: Evaluator,
    parent_node,
    agency: AgencyItem,
    label: str,  # "first", "second", "third", "fourth"
) -> None:
    """
    Build verification nodes for one agency.
    """
    # Parent node for this agency (parallel aggregation within the agency)
    agency_node = evaluator.add_parallel(
        id=f"{label}_agency",
        desc=f"Evaluate the {label} news agency provided in the solution",
        parent=parent_node,
        critical=False
    )

    # --- 1) Official name provided (existence) ---
    name_exists = bool(agency and agency.official_name and agency.official_name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id=f"{label}_agency_name",
        desc="The official name of a major international news wire service or agency is provided",
        parent=agency_node,
        critical=True
    )

    # --- 2) Complete HQ address provided (street, city, country all present) ---
    hq = agency.headquarters or HQAddress()
    hq_complete = bool(hq.street and hq.street.strip() and hq.city and hq.city.strip() and hq.country and hq.country.strip())
    evaluator.add_custom_node(
        result=hq_complete,
        id=f"{label}_agency_headquarters",
        desc="The complete headquarters address is provided, including street address, city, and country",
        parent=agency_node,
        critical=True
    )

    # --- 3) HQ region is either North America or Europe (LLM simple verify) ---
    # Construct a human-readable address string for the claim
    full_address = hq.full_address or ", ".join([p for p in [hq.street, hq.city, hq.country] if p])
    region_leaf = evaluator.add_leaf(
        id=f"{label}_agency_region",
        desc="The agency's headquarters is located in either North America or Europe",
        parent=agency_node,
        critical=True
    )
    region_claim = f"The headquarters address '{full_address}' is located in either North America or Europe."
    await evaluator.verify(
        claim=region_claim,
        node=region_leaf,
        additional_instruction=(
            "Judge using general geographic knowledge. "
            "North America includes the United States (USA/U.S.), Canada, and Mexico. "
            "Europe includes the United Kingdom, France, Germany, Italy, Spain, Portugal, Netherlands, Belgium, "
            "Luxembourg, Switzerland, Austria, Denmark, Norway, Sweden, Finland, Iceland, Ireland, Poland, "
            "Czech Republic, Slovakia, Hungary, Romania, Bulgaria, Slovenia, Croatia, Serbia, Bosnia, Montenegro, "
            "Kosovo, Albania, North Macedonia, Greece, Cyprus, Malta, Estonia, Latvia, Lithuania, Ukraine, Moldova, etc. "
            "If the address is ambiguous or clearly outside these regions, mark Incorrect."
        )
    )

    # --- 4) Coverage evidence: verify URL(s) show coverage of a major INTL event in early 2026 ---
    urls = sanitize_urls(agency.coverage_urls or [])
    coverage_desc = (
        "A reference URL is provided that demonstrates the agency covered at least one major international news event "
        f"{DATE_RANGE_TEXT}"
    )

    if not urls:
        # No URLs: fail this critical check directly
        evaluator.add_custom_node(
            result=False,
            id=f"{label}_agency_coverage",
            desc=coverage_desc,
            parent=agency_node,
            critical=True
        )
    else:
        coverage_leaf = evaluator.add_leaf(
            id=f"{label}_agency_coverage",
            desc=coverage_desc,
            parent=agency_node,
            critical=True
        )
        name_for_claim = agency.official_name or "the agency"
        coverage_claim = (
            f"This webpage shows that {name_for_claim} provided news coverage of a major international event "
            f"{DATE_RANGE_TEXT}."
        )
        await evaluator.verify(
            claim=coverage_claim,
            node=coverage_leaf,
            sources=urls,
            additional_instruction=(
                "Requirements to PASS:\n"
                "1) The page either (a) is on the agency’s own site and is a news article or report by the agency; OR\n"
                "   (b) is on a reputable news outlet that clearly credits/names the agency (e.g., Reuters/AP/AFP) as the wire or byline,\n"
                "      thereby demonstrating that the agency provided coverage.\n"
                f"2) The coverage/publication date on the page must fall {DATE_RANGE_TEXT}. If the page shows a date outside this range, FAIL.\n"
                "3) The event should be of significant international importance (e.g., conflicts, major diplomatic or political events, "
                "global economic or security developments, large natural disasters with international impact, etc.).\n"
                "4) If the URL is invalid, irrelevant, or does not clearly show the agency's coverage and date, mark as Incorrect."
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
    Evaluate an answer for the 'news_agencies_early_2026' task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent checks across agencies + overall distribution
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

    # Extract up to four agencies
    extracted = await evaluator.extract(
        prompt=prompt_extract_agencies(),
        template_class=AgenciesExtraction,
        extraction_name="agencies_extraction"
    )

    agencies: List[AgencyItem] = list(extracted.agencies or [])
    # Keep only the first 4; pad if fewer than 4
    agencies = agencies[:4]
    while len(agencies) < 4:
        agencies.append(AgencyItem())

    # Verify each agency
    labels = ["first", "second", "third", "fourth"]
    for idx, label in enumerate(labels):
        await verify_agency(evaluator, root, agencies[idx], label)

    # Regional distribution check (critical): exactly two NA and exactly two Europe
    counts = count_regions(agencies)
    dist_ok = (counts.get("North America", 0) == 2) and (counts.get("Europe", 0) == 2)
    evaluator.add_custom_node(
        result=dist_ok,
        id="regional_distribution",
        desc="Verify that exactly two agencies are headquartered in North America and exactly two in Europe",
        parent=root,
        critical=True
    )

    # Record helpful custom info for debugging
    evaluator.add_custom_info(
        info={
            "region_counts": counts,
            "agencies_used": [
                {
                    "official_name": a.official_name,
                    "hq_street": (a.headquarters.street if a.headquarters else None),
                    "hq_city": (a.headquarters.city if a.headquarters else None),
                    "hq_country": (a.headquarters.country if a.headquarters else None),
                    "computed_region": classify_region_from_hq(a.headquarters),
                    "coverage_urls_count": len(sanitize_urls(a.coverage_urls or [])),
                } for a in agencies
            ]
        },
        info_type="diagnostics",
        info_name="region_and_agency_diagnostics"
    )

    return evaluator.get_summary()