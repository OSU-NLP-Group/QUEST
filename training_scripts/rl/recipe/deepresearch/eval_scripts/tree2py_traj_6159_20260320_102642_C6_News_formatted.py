import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "ca_2025_special_election_ballot_measure"
TASK_DESCRIPTION = (
    "In 2025, California held a special election on a ballot measure to temporarily redraw the state's congressional "
    "districts in response to redistricting actions by another state. Provide comprehensive information with citations: "
    "(1) Number and name of the constitutional amendment that placed this measure on the ballot, including the full name "
    "of its primary author, the author's current leadership position in the California State Legislature, and the Assembly "
    "or Senate district the author represents. "
    "(2) The number of the Assembly Bill that contains the district boundary definitions and at least one primary author. "
    "(3) The exact date of the special election, the outcome, and the official certified vote counts with Yes/No and their percentages. "
    "(4) All five specific congressional district numbers that were redrawn to shift toward Democrats under the new maps based on 2024 presidential results. "
    "(5) The name and location (city, state) of the firm that drafted the adopted map. "
    "(6) The time period (start year through end year) the new maps will be used, and which state's redistricting prompted California's action."
)


# ------------------------- Data Models ------------------------- #
class MeasureIdentification(BaseModel):
    proposition_number: Optional[str] = None
    special_election_date: Optional[str] = None
    citations: List[str] = Field(default_factory=list)


class AmendmentInfo(BaseModel):
    amendment_number: Optional[str] = None
    primary_author_full_name: Optional[str] = None
    author_leadership_position: Optional[str] = None
    author_district: Optional[str] = None
    citations: List[str] = Field(default_factory=list)


class ImplementingLegislation(BaseModel):
    bill_number: Optional[str] = None
    primary_authors: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)


class ElectionResults(BaseModel):
    outcome: Optional[str] = None  # e.g., "passed" or "failed"
    yes_votes: Optional[str] = None
    yes_percent: Optional[str] = None
    no_votes: Optional[str] = None
    no_percent: Optional[str] = None
    citations: List[str] = Field(default_factory=list)


class DistrictsInfo(BaseModel):
    redrawn_districts: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)


class FirmInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    citations: List[str] = Field(default_factory=list)


class UsageContext(BaseModel):
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    prompting_state: Optional[str] = None
    citations: List[str] = Field(default_factory=list)


class BallotMeasureExtraction(BaseModel):
    measure: MeasureIdentification = Field(default_factory=MeasureIdentification)
    amendment: AmendmentInfo = Field(default_factory=AmendmentInfo)
    implementing_legislation: ImplementingLegislation = Field(default_factory=ImplementingLegislation)
    results: ElectionResults = Field(default_factory=ElectionResults)
    districts: DistrictsInfo = Field(default_factory=DistrictsInfo)
    firm: FirmInfo = Field(default_factory=FirmInfo)
    usage: UsageContext = Field(default_factory=UsageContext)


# ------------------------- Extraction Prompt ------------------------- #
def prompt_extract_ballot_measure() -> str:
    return """
Extract the following fields strictly from the provided answer text without adding or inferring missing information. Use exact strings as written in the answer. If a field is missing, set it to null or an empty list accordingly. For all citation fields, only include URLs explicitly present in the answer (plain, markdown, or other recognizable formats).

Return a JSON object with this structure:
{
  "measure": {
    "proposition_number": string or null,               // e.g., "Proposition 50" or "Prop 50"
    "special_election_date": string or null,            // e.g., "November 4, 2025"
    "citations": [urls...]                              
  },
  "amendment": {
    "amendment_number": string or null,                 // e.g., "ACA 8"
    "primary_author_full_name": string or null,         // e.g., "Robert Rivas"
    "author_leadership_position": string or null,       // e.g., "Speaker of the California State Assembly"
    "author_district": string or null,                  // e.g., "Assembly District 29" or "AD 29"
    "citations": [urls...]
  },
  "implementing_legislation": {
    "bill_number": string or null,                      // e.g., "AB 604"
    "primary_authors": [string...],                     // include at least one if present; authors as written
    "citations": [urls...]
  },
  "results": {
    "outcome": string or null,                          // e.g., "passed" or "failed"
    "yes_votes": string or null,                        // keep formatting as written, e.g., "7,453,339"
    "yes_percent": string or null,                      // e.g., "64.42%"
    "no_votes": string or null,                         // e.g., "4,116,998"
    "no_percent": string or null,                       // e.g., "35.58%"
    "citations": [urls...]
  },
  "districts": {
    "redrawn_districts": [string...],                   // list exactly the districts as written, e.g., ["District 1", "CA-3", "22"]
    "citations": [urls...]
  },
  "firm": {
    "name": string or null,                             // e.g., "Redistricting Partners, LLC"
    "city": string or null,                             // e.g., "Sacramento"
    "state": string or null,                            // e.g., "California" or "CA"
    "citations": [urls...]
  },
  "usage": {
    "start_year": string or null,                       // e.g., "2026"
    "end_year": string or null,                         // e.g., "2030"
    "prompting_state": string or null,                  // e.g., "Texas"
    "citations": [urls...]
  }
}

Special rules for URL extraction:
- Extract only actual URLs present in the answer (including markdown links).
- If a URL is missing a protocol, prepend "http://".
- Do not fabricate any URLs.
"""


# ------------------------- Helpers ------------------------- #
def _normalize_url(u: str) -> str:
    if not u:
        return u
    u = u.strip()
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+\-.]*://', u):
        u = "http://" + u
    return u


def _all_citation_urls(data: BallotMeasureExtraction) -> List[str]:
    urls: List[str] = []
    urls.extend([_normalize_url(x) for x in (data.measure.citations or [])])
    urls.extend([_normalize_url(x) for x in (data.amendment.citations or [])])
    urls.extend([_normalize_url(x) for x in (data.implementing_legislation.citations or [])])
    urls.extend([_normalize_url(x) for x in (data.results.citations or [])])
    urls.extend([_normalize_url(x) for x in (data.districts.citations or [])])
    urls.extend([_normalize_url(x) for x in (data.firm.citations or [])])
    urls.extend([_normalize_url(x) for x in (data.usage.citations or [])])
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _host(url: str) -> str:
    try:
        p = urlparse(url)
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _is_ca_gov(host: str) -> bool:
    return host.endswith(".ca.gov") or host == "ca.gov"


def _is_reliable_news(host: str) -> bool:
    # Broad, non-exhaustive list of widely recognized outlets (national + California-focused)
    allowed = {
        "latimes.com", "sfchronicle.com", "sacbee.com", "mercurynews.com", "calmatters.org",
        "kqed.org", "capradio.org", "laist.com", "voiceofsandiego.org", "nbcnews.com",
        "cbsnews.com", "abcnews.go.com", "cnn.com", "foxnews.com", "apnews.com",
        "reuters.com", "bloomberg.com", "nytimes.com", "washingtonpost.com", "politico.com",
        "pbs.org", "npr.org", "ktla.com", "kron4.com", "nbcbayarea.com", "abc7.com",
        "axios.com", "time.com", "wsj.com", "theatlantic.com", "theguardian.com", "propublica.org",
    }
    # Accept subdomains of allowed bases
    return any(host == base or host.endswith("." + base) for base in allowed)


def _citations_all_valid(urls: List[str]) -> bool:
    if not urls:
        return False  # The task requires citations across the answer; empty overall set should fail this validity check
    for u in urls:
        h = _host(_normalize_url(u))
        if not h:
            return False
        if _is_ca_gov(h):
            continue
        if _is_reliable_news(h):
            continue
        return False
    return True


def _extract_digits(s: str) -> Optional[int]:
    if not s:
        return None
    m = re.search(r'\d+', s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _districts_set_ok(extracted: List[str]) -> bool:
    expected = {1, 3, 22, 41, 48}
    nums = []
    for item in extracted or []:
        n = _extract_digits(item)
        if n is not None:
            nums.append(n)
    s = set(nums)
    return s == expected and len(s) == 5


# ------------------------- Verification Subtrees ------------------------- #
async def verify_citations_source_validity(evaluator: Evaluator, parent, data: BallotMeasureExtraction) -> None:
    urls = _all_citation_urls(data)
    evaluator.add_custom_node(
        result=_citations_all_valid(urls),
        id="Citations_Source_Validity",
        desc="All provided reference URLs are from official CA government sources or reliable news organizations.",
        parent=parent,
        critical=True
    )


async def verify_measure_identification(evaluator: Evaluator, parent, data: BallotMeasureExtraction) -> None:
    node = evaluator.add_parallel(
        id="Measure_Ballot_Identification",
        desc="Correctly identifies the ballot measure with supporting citations.",
        parent=parent,
        critical=True
    )

    # Citations presence check
    measure_cites_exist = bool(data.measure.citations)
    evaluator.add_custom_node(
        result=measure_cites_exist,
        id="Measure_Identification_Citations",
        desc="Includes citation URL(s) supporting the proposition number and special election date.",
        parent=node,
        critical=True
    )

    # Proposition number is 50
    prop_leaf = evaluator.add_leaf(
        id="Proposition_Number_Is_50",
        desc="States that the ballot measure is California Proposition 50.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The ballot measure is California Proposition 50.",
        node=prop_leaf,
        sources=data.measure.citations,
        additional_instruction="Verify the proposition number for the 2025 California special election measure. Treat 'Prop 50' as 'Proposition 50'."
    )

    # Special election date
    date_leaf = evaluator.add_leaf(
        id="Special_Election_Date_Is_2025_11_04",
        desc="States that the special election date was November 4, 2025.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The special election date for California Proposition 50 was November 4, 2025.",
        node=date_leaf,
        sources=data.measure.citations,
        additional_instruction="Confirm the official special election date; accept minor formatting like 'Nov. 4, 2025'. Prefer California government sites when available."
    )


async def verify_constitutional_amendment(evaluator: Evaluator, parent, data: BallotMeasureExtraction) -> None:
    node = evaluator.add_parallel(
        id="Constitutional_Amendment_Placing_Measure_On_Ballot",
        desc="Provides the constitutional amendment and author details with citations.",
        parent=parent,
        critical=True
    )

    # Citations presence
    aca_cites_exist = bool(data.amendment.citations)
    evaluator.add_custom_node(
        result=aca_cites_exist,
        id="ACA_8_Citations",
        desc="Includes citation URL(s) supporting ACA 8 identification and Robert Rivas’s authorship/position/district.",
        parent=node,
        critical=True
    )

    # ACA 8 identification
    aca_leaf = evaluator.add_leaf(
        id="Amendment_Is_ACA_8",
        desc="States that the constitutional amendment is Assembly Constitutional Amendment 8 (ACA 8).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The constitutional amendment that placed the measure on the ballot is Assembly Constitutional Amendment 8 (ACA 8).",
        node=aca_leaf,
        sources=data.amendment.citations,
        additional_instruction="Verify that ACA 8 placed the measure on the ballot."
    )

    # Primary author: Robert Rivas
    author_leaf = evaluator.add_leaf(
        id="Primary_Author_Is_Robert_Rivas",
        desc="States that the primary author is Robert Rivas.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The primary author of ACA 8 is Robert Rivas.",
        node=author_leaf,
        sources=data.amendment.citations,
        additional_instruction="Confirm authorship details on official legislature or CA government pages."
    )

    # Leadership position: Assembly Speaker
    leader_leaf = evaluator.add_leaf(
        id="Author_Leadership_Position_Is_Assembly_Speaker",
        desc="States that Robert Rivas's current leadership position is Speaker of the California State Assembly.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Robert Rivas is the Speaker of the California State Assembly.",
        node=leader_leaf,
        sources=data.amendment.citations,
        additional_instruction="Leadership titles can be shown on legislative pages or official bios."
    )

    # District: AD 29
    district_leaf = evaluator.add_leaf(
        id="Author_District_Is_Assembly_District_29",
        desc="States that Robert Rivas represents California Assembly District 29.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Robert Rivas represents California Assembly District 29.",
        node=district_leaf,
        sources=data.amendment.citations,
        additional_instruction="Allow minor formatting like 'AD 29' or 'Assembly District 29'."
    )


async def verify_implementing_legislation(evaluator: Evaluator, parent, data: BallotMeasureExtraction) -> None:
    node = evaluator.add_parallel(
        id="Implementing_Legislation",
        desc="Provides the implementing legislation and primary author with citations.",
        parent=parent,
        critical=True
    )

    # Citations presence
    ab_cites_exist = bool(data.implementing_legislation.citations)
    evaluator.add_custom_node(
        result=ab_cites_exist,
        id="AB_604_Citations",
        desc="Includes citation URL(s) supporting AB 604 identification and Marc Berman’s primary authorship.",
        parent=node,
        critical=True
    )

    # Implementing bill: AB 604
    ab_leaf = evaluator.add_leaf(
        id="Implementing_Bill_Is_AB_604",
        desc="States that the implementing legislation is Assembly Bill 604 (AB 604).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The implementing legislation containing the congressional district boundary definitions is Assembly Bill 604 (AB 604).",
        node=ab_leaf,
        sources=data.implementing_legislation.citations,
        additional_instruction="Verify that AB 604 contains the district boundary definitions."
    )

    # Primary author: Marc Berman
    berman_leaf = evaluator.add_leaf(
        id="AB_604_Includes_Primary_Author_Marc_Berman",
        desc="Identifies Marc Berman as a primary author of AB 604.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="AB 604 lists Assemblymember Marc Berman as a primary author (author or principal coauthor).",
        node=berman_leaf,
        sources=data.implementing_legislation.citations,
        additional_instruction="Accept 'author' or 'principal coauthor' designations as primary authorship."
    )


async def verify_special_election_results(evaluator: Evaluator, parent, data: BallotMeasureExtraction) -> None:
    node = evaluator.add_parallel(
        id="Special_Election_Results",
        desc="Provides certified results with citations.",
        parent=parent,
        critical=True
    )

    # Citations presence
    res_cites_exist = bool(data.results.citations)
    evaluator.add_custom_node(
        result=res_cites_exist,
        id="Election_Results_Citations",
        desc="Includes citation URL(s) to official certified results or reliable reporting supporting the outcome and stated certified vote totals/percentages.",
        parent=node,
        critical=True
    )

    # Outcome passed
    outcome_leaf = evaluator.add_leaf(
        id="Outcome_Is_Passed",
        desc="States that the measure passed.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="California Proposition 50 passed.",
        node=outcome_leaf,
        sources=data.results.citations,
        additional_instruction="Prefer official California Secretary of State or state certification sources."
    )

    # Certified Yes votes and percentage
    yes_leaf = evaluator.add_leaf(
        id="Certified_Yes_Votes_And_Percentage_Match",
        desc="States certified Yes votes as 7,453,339 and Yes percentage as 64.42%.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The official certified statewide results for California Proposition 50 report 7,453,339 Yes votes, representing 64.42%.",
        node=yes_leaf,
        sources=data.results.citations,
        additional_instruction="Check the certified results table; allow minor formatting differences such as commas or percent symbols."
    )

    # Certified No votes and percentage
    no_leaf = evaluator.add_leaf(
        id="Certified_No_Votes_And_Percentage_Match",
        desc="States certified No votes as 4,116,998 and No percentage as 35.58%.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The official certified statewide results for California Proposition 50 report 4,116,998 No votes, representing 35.58%.",
        node=no_leaf,
        sources=data.results.citations,
        additional_instruction="Check the certified results table; allow minor formatting differences such as commas or percent symbols."
    )


async def verify_five_redrawn_districts(evaluator: Evaluator, parent, data: BallotMeasureExtraction) -> None:
    node = evaluator.add_parallel(
        id="Five_Redrawn_Districts",
        desc="Identifies the affected congressional districts with citations.",
        parent=parent,
        critical=True
    )

    # Exact district set match (code-level check against rubric)
    evaluator.add_custom_node(
        result=_districts_set_ok(data.districts.redrawn_districts),
        id="District_Set_Matches_Exactly_Five_Specified",
        desc="Lists exactly these five U.S. congressional districts and no others: Districts 1, 3, 22, 41, and 48.",
        parent=node,
        critical=True
    )

    # Citations presence
    evaluator.add_custom_node(
        result=bool(data.districts.citations),
        id="Districts_Citations",
        desc="Includes citation URL(s) supporting that these are the five districts affected as described.",
        parent=node,
        critical=True
    )


async def verify_map_drafting_firm(evaluator: Evaluator, parent, data: BallotMeasureExtraction) -> None:
    node = evaluator.add_parallel(
        id="Map_Drafting_Firm",
        desc="Identifies the map-drafting firm and its location with citations.",
        parent=parent,
        critical=True
    )

    # Citations presence
    evaluator.add_custom_node(
        result=bool(data.firm.citations),
        id="Firm_Citations",
        desc="Includes citation URL(s) supporting the firm identity and location.",
        parent=node,
        critical=True
    )

    # Firm name
    firm_leaf = evaluator.add_leaf(
        id="Firm_Is_Redistricting_Partners_LLC",
        desc="States the firm name as Redistricting Partners, LLC.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The firm that drafted the adopted congressional district map was Redistricting Partners, LLC.",
        node=firm_leaf,
        sources=data.firm.citations,
        additional_instruction="Confirm the firm credited with drafting the adopted California congressional map."
    )

    # Firm location
    loc_leaf = evaluator.add_leaf(
        id="Firm_Location_Is_Sacramento_CA",
        desc="States the firm location as Sacramento, California.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Redistricting Partners, LLC is based in Sacramento, California.",
        node=loc_leaf,
        sources=data.firm.citations,
        additional_instruction="Confirm the firm's location (city and state)."
    )


async def verify_maps_usage_and_prompting_state(evaluator: Evaluator, parent, data: BallotMeasureExtraction) -> None:
    node = evaluator.add_parallel(
        id="Maps_Usage_Period_And_Prompting_State_Action",
        desc="States the maps’ usage period and the prompting state action with citations.",
        parent=parent,
        critical=True
    )

    # Citations presence
    evaluator.add_custom_node(
        result=bool(data.usage.citations),
        id="Usage_And_Context_Citations",
        desc="Includes citation URL(s) supporting the 2026–2030 usage period and that Texas is the prompting state.",
        parent=node,
        critical=True
    )

    # Usage period
    usage_leaf = evaluator.add_leaf(
        id="Maps_Used_2026_Through_2030",
        desc="States the new maps will be used for congressional elections from 2026 through 2030.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The new California congressional maps will be used for congressional elections from 2026 through 2030.",
        node=usage_leaf,
        sources=data.usage.citations,
        additional_instruction="Confirm the calendar years of use for the adopted maps."
    )

    # Prompting state action
    prompt_leaf = evaluator.add_leaf(
        id="Prompting_State_Is_Texas_2025",
        desc="Identifies Texas congressional redistricting in 2025 as the prompting action for California’s response.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Texas's 2025 congressional redistricting prompted California’s response and special-election measure.",
        node=prompt_leaf,
        sources=data.usage.citations,
        additional_instruction="Confirm that Texas’s 2025 congressional redistricting action was the explicit trigger."
    )


# ------------------------- Main Evaluation ------------------------- #
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_ballot_measure(),
        template_class=BallotMeasureExtraction,
        extraction_name="ballot_measure_extraction"
    )

    evaluator.add_ground_truth({
        "expected": {
            "proposition_number": "Proposition 50",
            "special_election_date": "November 4, 2025",
            "amendment_number": "ACA 8",
            "amendment_primary_author": "Robert Rivas",
            "leadership_position": "Speaker of the California State Assembly",
            "author_district": "Assembly District 29",
            "implementing_bill": "AB 604",
            "implementing_primary_author_example": "Marc Berman",
            "outcome": "passed",
            "yes_votes": "7,453,339",
            "yes_percent": "64.42%",
            "no_votes": "4,116,998",
            "no_percent": "35.58%",
            "redrawn_districts": [1, 3, 22, 41, 48],
            "firm_name": "Redistricting Partners, LLC",
            "firm_city": "Sacramento",
            "firm_state": "California",
            "usage_period": {"start": "2026", "end": "2030"},
            "prompting_state": "Texas (2025)"
        }
    })

    # Top-level critical node mirroring rubric root
    main = evaluator.add_parallel(
        id="Ballot_Measure_Comprehensive_Info_With_Citations",
        desc="Provides comprehensive information about the 2025 California special-election ballot measure with valid citations.",
        parent=root,
        critical=True
    )

    await verify_citations_source_validity(evaluator, main, extracted)
    await verify_measure_identification(evaluator, main, extracted)
    await verify_constitutional_amendment(evaluator, main, extracted)
    await verify_implementing_legislation(evaluator, main, extracted)
    await verify_special_election_results(evaluator, main, extracted)
    await verify_five_redrawn_districts(evaluator, main, extracted)
    await verify_map_drafting_firm(evaluator, main, extracted)
    await verify_maps_usage_and_prompting_state(evaluator, main, extracted)

    return evaluator.get_summary()