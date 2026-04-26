import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "performing_arts_venues_us"
TASK_DESCRIPTION = """
Identify four major performing arts venues in the United States that meet the following specific criteria:

1. Venue 1: The Broadway theater in New York City that has the largest seating capacity of any Broadway theater. Provide its name, exact seating capacity, and confirm it meets the Broadway classification requirement of at least 500 seats.

2. Venue 2: The Lincoln Center theater where American Ballet Theatre (ABT) performed its Spring 2026 season from March 6-21, 2026. Provide the theater's name, seating capacity, and confirm the specific dates and company that performed there during this season.

3. Venue 3: The Broadway-classified theater located at Lincoln Center that has exactly 1,100 seats. Provide its name, address, and confirm it meets the Broadway theater minimum capacity requirement.

4. Venue 4: The theater at the Kennedy Center for the Performing Arts in Washington, D.C. that was specifically designed for ballet, opera, and musical theater, has a seating capacity between 2,300 and 2,400 seats, and is the second-largest theater in the Kennedy Center. Provide its name, exact capacity, and its design purpose.

For each venue, provide its name, seating capacity, location details, and reference URL(s) that support your answer.
"""

# Ground-truth expectations encoded from rubric for direct matching checks in some leaves
GT = {
    "venue_1": {
        "name": "Gershwin Theatre",
        "capacity_exact": "1,933",
        "location_req": "New York City and a Broadway theater",
    },
    "venue_2": {
        "name": "David H. Koch Theater",
        "capacity_exact": "2,544",
        "location_req": "at Lincoln Center in New York City",
        "company": "American Ballet Theatre",
        "dates_text": "March 6–21, 2026",
    },
    "venue_3": {
        "name": "Vivian Beaumont Theater",
        "capacity_exact": "1,100",
        "address": "150 West 65th Street",
        "location_req": "at Lincoln Center at 150 West 65th Street",
    },
    "venue_4": {
        "name": "Opera House",  # Kennedy Center Opera House
        "capacity_exact": "2,364",
        "location_req": "at the Kennedy Center for the Performing Arts in Washington, D.C.",
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Venue1Info(BaseModel):
    name: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to be flexible (e.g., "1,933")
    location: Optional[str] = None  # Free-form description from the answer
    classification: Optional[str] = None  # e.g., "Broadway theater"
    urls: List[str] = Field(default_factory=list)  # References used by the answer


class Venue2Info(BaseModel):
    name: Optional[str] = None
    capacity: Optional[str] = None
    location: Optional[str] = None
    performance_company: Optional[str] = None  # e.g., "American Ballet Theatre"
    performance_start_date: Optional[str] = None  # e.g., "March 6, 2026"
    performance_end_date: Optional[str] = None  # e.g., "March 21, 2026"
    performance_dates_text: Optional[str] = None  # Free text for the date span
    performance_urls: List[str] = Field(default_factory=list)  # URLs specifically about the ABT Spring 2026 season at this theater
    urls: List[str] = Field(default_factory=list)  # General theater info URLs (capacity/location)


class Venue3Info(BaseModel):
    name: Optional[str] = None
    capacity: Optional[str] = None
    location: Optional[str] = None  # Should include Lincoln Center and address context
    address: Optional[str] = None  # Prefer extracting "150 West 65th Street"
    classification: Optional[str] = None  # e.g., "Broadway theater"
    urls: List[str] = Field(default_factory=list)


class Venue4Info(BaseModel):
    name: Optional[str] = None
    capacity: Optional[str] = None
    location: Optional[str] = None
    design_purpose: Optional[str] = None  # e.g., "designed for ballet, opera, and musical theater"
    size_rank: Optional[str] = None  # e.g., "second-largest theater at the Kennedy Center"
    urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venue1: Optional[Venue1Info] = None
    venue2: Optional[Venue2Info] = None
    venue3: Optional[Venue3Info] = None
    venue4: Optional[Venue4Info] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract structured information for exactly four venues as specified. Only extract what appears explicitly in the answer. If a field is missing, use null or an empty list accordingly.

For each venue, extract the following:

Venue 1 (Largest Broadway theater by capacity in NYC):
- name: the theater's name as written in the answer
- capacity: the seating capacity as written (keep punctuation like commas)
- location: the location description as written (e.g., "in New York City", "on Broadway")
- classification: how the answer describes its classification (e.g., "Broadway theater")
- urls: an array of all URLs the answer cites for this venue's facts (capacity, location, 'largest Broadway' status, etc.)

Venue 2 (Lincoln Center theater where ABT performed Spring 2026 from March 6–21, 2026):
- name
- capacity
- location: the location description as written (should reference Lincoln Center in NYC)
- performance_company: the performing company name as written (e.g., "American Ballet Theatre")
- performance_start_date: the start date as written (e.g., "March 6, 2026")
- performance_end_date: the end date as written (e.g., "March 21, 2026")
- performance_dates_text: the date span as a single string if present (e.g., "March 6–21, 2026")
- performance_urls: all URLs the answer cites that specifically support the ABT Spring 2026 season at this theater with dates
- urls: all other URLs the answer cites for this theater (capacity and location)

Venue 3 (Broadway-classified theater at Lincoln Center with exactly 1,100 seats):
- name
- capacity
- location: description as written (should include Lincoln Center context)
- address: address string as written (preferably "150 West 65th Street" or close variant)
- classification: how the answer describes classification (e.g., "Broadway theater")
- urls: all URLs the answer cites supporting capacity, address, and Broadway classification

Venue 4 (Kennedy Center theater designed for ballet, opera, and musical theater; capacity between 2,300 and 2,400; second-largest in the Kennedy Center):
- name
- capacity
- location: description as written (should reference the Kennedy Center in Washington, D.C.)
- design_purpose: the stated design purpose as written, ideally including "ballet, opera, and musical theater"
- size_rank: the statement of being "second-largest" at the Kennedy Center, if provided
- urls: all URLs the answer cites supporting capacity, design purpose, location, and size ranking

Special rules:
- Extract only URLs that actually appear in the answer text (including in markdown links).
- Do not infer or generate URLs.
- Keep dates and numbers as strings exactly as written.
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _to_int_or_none(x: Optional[str]) -> Optional[int]:
    if not x:
        return None
    digits = re.sub(r"[^\d]", "", x)
    return int(digits) if digits.isdigit() else None


def _has_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")):
            return True
    return False


def _prefer_urls(primary: List[str], fallback: List[str]) -> List[str]:
    return primary if _has_urls(primary) else fallback


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_venue_1(evaluator: Evaluator, parent_node, v1: Optional[Venue1Info]) -> None:
    node = evaluator.add_parallel(
        id="venue_1",
        desc="Identify the Broadway theater with the largest seating capacity in New York City",
        parent=parent_node,
        critical=False,
    )

    # Leaf: Name must be Gershwin Theatre (match answer content to this expected name)
    name_leaf = evaluator.add_leaf(
        id="venue_1_name",
        desc="The theater is named the Gershwin Theatre",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the Broadway theater with the largest seating capacity is identified as 'Gershwin Theatre' (accept minor variants like 'The Gershwin Theatre' or 'Gershwin Theater').",
        node=name_leaf,
        additional_instruction="Judge by comparing to the provided answer text; allow minor capitalization and 'Theatre/Theater' spelling variants.",
    )

    # Leaf: Capacity equals 1,933 (as stated in the answer)
    capacity_leaf = evaluator.add_leaf(
        id="venue_1_capacity",
        desc="The theater has a seating capacity of 1,933 seats",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the Gershwin Theatre's seating capacity is stated as 1,933 (commas optional).",
        node=capacity_leaf,
        additional_instruction="Check the answer for an explicit capacity equal to 1933 (treat '1,933' and '1933' as equivalent).",
    )

    # Leaf: Location — NYC and Broadway (answer content)
    location_leaf = evaluator.add_leaf(
        id="venue_1_location",
        desc="The theater is located in New York City on Broadway",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the Gershwin Theatre is described as being in New York City and a Broadway theater.",
        node=location_leaf,
        additional_instruction="Verify the answer text mentions New York City and Broadway classification/context.",
    )

    # Leaf: Broadway classification requirement (>= 500 seats) — numeric check from extracted capacity
    # Implemented as a custom leaf using parsed capacity if available; falls back to stated 1933 if missing
    cap_int = _to_int_or_none(v1.capacity if v1 else None) or _to_int_or_none(GT["venue_1"]["capacity_exact"])
    evaluator.add_custom_node(
        result=(cap_int is not None and cap_int >= 500),
        id="venue_1_classification",
        desc="The theater meets the Broadway classification requirement of having at least 500 seats",
        parent=node,
        critical=True,
    )

    # Leaf: Reference — URLs confirm both capacity and "largest Broadway theater" status
    ref_leaf = evaluator.add_leaf(
        id="venue_1_reference",
        desc="Provides a valid reference URL confirming the theater's capacity and status as Broadway's largest",
        parent=node,
        critical=True,
    )
    v1_urls = (v1.urls if v1 else []) or []
    if not _has_urls(v1_urls):
        # No URLs provided – fail this critical leaf
        ref_leaf.score = 0.0
        ref_leaf.status = "failed"
    else:
        await evaluator.verify(
            claim="The sources confirm that the Gershwin Theatre has a seating capacity of 1,933 and that it is the largest Broadway theater by seating capacity.",
            node=ref_leaf,
            sources=v1_urls,
            additional_instruction="Look for explicit statements about capacity (1,933) and superlatives like 'largest Broadway theater' or 'greatest seating capacity among Broadway houses'.",
        )


async def verify_venue_2(evaluator: Evaluator, parent_node, v2: Optional[Venue2Info]) -> None:
    node = evaluator.add_parallel(
        id="venue_2",
        desc="Identify the Lincoln Center theater where American Ballet Theatre performed its Spring 2026 season from March 6-21, 2026",
        parent=parent_node,
        critical=False,
    )

    # Leaf: Name must be David H. Koch Theater (answer content)
    name_leaf = evaluator.add_leaf(
        id="venue_2_name",
        desc="The theater is named the David H. Koch Theater",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the Lincoln Center theater where ABT performed its Spring 2026 season is identified as the 'David H. Koch Theater' (allow 'Theatre' spelling variants).",
        node=name_leaf,
        additional_instruction="Judge by answer content; accept minor capitalization and 'Theater/Theatre' variants.",
    )

    # Leaf: Capacity equals 2,544 (answer content)
    capacity_leaf = evaluator.add_leaf(
        id="venue_2_capacity",
        desc="The theater has a seating capacity of 2,544 seats",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the David H. Koch Theater's seating capacity is stated as 2,544 (commas optional).",
        node=capacity_leaf,
        additional_instruction="Verify the answer text explicitly lists 2544 (treat '2,544' and '2544' as equivalent).",
    )

    # Leaf: Location at Lincoln Center in NYC (answer content)
    location_leaf = evaluator.add_leaf(
        id="venue_2_location",
        desc="The theater is located at Lincoln Center in New York City",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the David H. Koch Theater is described as being at Lincoln Center in New York City.",
        node=location_leaf,
        additional_instruction="Look for 'Lincoln Center' and 'New York City' in the answer.",
    )

    # Sub-node: Performance details (critical parallel group)
    perf_node = evaluator.add_parallel(
        id="venue_2_performance_details",
        desc="American Ballet Theatre performed at this theater during its Spring 2026 season from March 6-21, 2026",
        parent=node,
        critical=True,
    )

    # Performance URLs preferred; if none, fall back to general URLs
    perf_urls = _prefer_urls((v2.performance_urls if v2 else []), (v2.urls if v2 else []))

    # Company leaf (critical)
    company_leaf = evaluator.add_leaf(
        id="venue_2_performance_company",
        desc="The performing company was American Ballet Theatre (ABT)",
        parent=perf_node,
        critical=True,
    )
    if not _has_urls(perf_urls):
        company_leaf.score = 0.0
        company_leaf.status = "failed"
    else:
        await evaluator.verify(
            claim="American Ballet Theatre (ABT) performed at the David H. Koch Theater during its Spring 2026 season.",
            node=company_leaf,
            sources=perf_urls,
            additional_instruction="The source should explicitly mention 'American Ballet Theatre' and the venue 'David H. Koch Theater' together for Spring 2026.",
        )

    # Dates leaf (critical)
    dates_leaf = evaluator.add_leaf(
        id="venue_2_performance_dates",
        desc="The performances occurred from March 6 to March 21, 2026",
        parent=perf_node,
        critical=True,
    )
    if not _has_urls(perf_urls):
        dates_leaf.score = 0.0
        dates_leaf.status = "failed"
    else:
        await evaluator.verify(
            claim="The ABT Spring 2026 season at the David H. Koch Theater ran from March 6 to March 21, 2026.",
            node=dates_leaf,
            sources=perf_urls,
            additional_instruction="Accept minor date formatting variants (e.g., 'Mar' vs 'March', en-dash vs hyphen). The range must cover March 6 through March 21, 2026.",
        )

    # Performance reference leaf (critical)
    perf_ref_leaf = evaluator.add_leaf(
        id="venue_2_performance_reference",
        desc="Provides a valid reference URL confirming ABT's Spring 2026 season at this venue",
        parent=perf_node,
        critical=True,
    )
    if not _has_urls(perf_urls):
        perf_ref_leaf.score = 0.0
        perf_ref_leaf.status = "failed"
    else:
        await evaluator.verify(
            claim="The sources explicitly confirm ABT's Spring 2026 season at the David H. Koch Theater with dates March 6–21, 2026.",
            node=perf_ref_leaf,
            sources=perf_urls,
            additional_instruction="Look for an official or credible page (e.g., ABT, Lincoln Center, or venue site) stating ABT performed at David H. Koch Theater from March 6 to March 21, 2026.",
        )

    # General reference for capacity and location (critical)
    ref_leaf = evaluator.add_leaf(
        id="venue_2_reference",
        desc="Provides a valid reference URL confirming the theater's capacity and location",
        parent=node,
        critical=True,
    )
    gen_urls = (v2.urls if v2 else []) or []
    if not _has_urls(gen_urls):
        ref_leaf.score = 0.0
        ref_leaf.status = "failed"
    else:
        await evaluator.verify(
            claim="The sources confirm that the David H. Koch Theater is at Lincoln Center in New York City and has a seating capacity of 2,544.",
            node=ref_leaf,
            sources=gen_urls,
            additional_instruction="Capacity must be 2,544 (commas optional). The page must clearly place the theater at Lincoln Center in NYC.",
        )


async def verify_venue_3(evaluator: Evaluator, parent_node, v3: Optional[Venue3Info]) -> None:
    node = evaluator.add_parallel(
        id="venue_3",
        desc="Identify the theater at Lincoln Center that is classified as a Broadway theater with the descriptor 'Broadway theater' and has 1,100 seats",
        parent=parent_node,
        critical=False,
    )

    # Name must be Vivian Beaumont Theater (answer content)
    name_leaf = evaluator.add_leaf(
        id="venue_3_name",
        desc="The theater is named the Vivian Beaumont Theater",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the Lincoln Center Broadway-classified theater with 1,100 seats is identified as the 'Vivian Beaumont Theater' (allow minor capitalization and 'Theatre/Theater' variant).",
        node=name_leaf,
        additional_instruction="Judge by the answer content. Accept 'The Vivian Beaumont Theater/Theatre'.",
    )

    # Capacity equals 1,100 (answer content)
    capacity_leaf = evaluator.add_leaf(
        id="venue_3_capacity",
        desc="The theater has a seating capacity of 1,100 seats",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the Vivian Beaumont Theater's seating capacity is stated as exactly 1,100 (commas optional).",
        node=capacity_leaf,
        additional_instruction="The capacity must be exactly 1100; treat '1,100' and '1100' as equivalent.",
    )

    # Location with address (answer content)
    location_leaf = evaluator.add_leaf(
        id="venue_3_location",
        desc="The theater is located at Lincoln Center at 150 West 65th Street",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the Vivian Beaumont Theater is described as being at Lincoln Center at 150 West 65th Street.",
        node=location_leaf,
        additional_instruction="Allow minor formatting variants like '150 W 65th St'.",
    )

    # Broadway classification minimum (>= 500 seats) — custom numeric check
    cap_int = _to_int_or_none(v3.capacity if v3 else None) or _to_int_or_none(GT["venue_3"]["capacity_exact"])
    evaluator.add_custom_node(
        result=(cap_int is not None and cap_int >= 500),
        id="venue_3_classification",
        desc="The theater is classified as a Broadway theater, meeting the minimum 500-seat requirement",
        parent=node,
        critical=True,
    )

    # References confirm capacity, location, and Broadway classification
    ref_leaf = evaluator.add_leaf(
        id="venue_3_reference",
        desc="Provides a valid reference URL confirming the theater's capacity, location, and Broadway classification",
        parent=node,
        critical=True,
    )
    v3_urls = (v3.urls if v3 else []) or []
    if not _has_urls(v3_urls):
        ref_leaf.score = 0.0
        ref_leaf.status = "failed"
    else:
        await evaluator.verify(
            claim="The sources confirm that the Vivian Beaumont Theater has 1,100 seats, is located at (or listed as) 150 West 65th Street at Lincoln Center, and is classified as a Broadway theater.",
            node=ref_leaf,
            sources=v3_urls,
            additional_instruction="Look for explicit seat count 1,100, location/address confirming Lincoln Center and 150 West 65th Street, and a Broadway designation.",
        )


async def verify_venue_4(evaluator: Evaluator, parent_node, v4: Optional[Venue4Info]) -> None:
    node = evaluator.add_parallel(
        id="venue_4",
        desc="Identify the Kennedy Center theater designed for ballet, opera, and musical theater with capacity between 2,300 and 2,400 seats",
        parent=parent_node,
        critical=False,
    )

    # Name must be Opera House (Kennedy Center Opera House) — answer content
    name_leaf = evaluator.add_leaf(
        id="venue_4_name",
        desc="The theater is named the Opera House (Kennedy Center Opera House)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the identified Kennedy Center theater is the Opera House (also called the Kennedy Center Opera House).",
        node=name_leaf,
        additional_instruction="Accept 'Opera House' or 'Kennedy Center Opera House'.",
    )

    # Capacity equals 2,364 (answer content)
    capacity_leaf = evaluator.add_leaf(
        id="venue_4_capacity",
        desc="The theater has a seating capacity of 2,364 seats",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the Opera House seating capacity is stated as 2,364 (commas optional).",
        node=capacity_leaf,
        additional_instruction="The capacity must be exactly 2364; treat '2,364' and '2364' as equivalent.",
    )

    # Location — at the Kennedy Center in Washington, D.C. (answer content)
    location_leaf = evaluator.add_leaf(
        id="venue_4_location",
        desc="The theater is located at the Kennedy Center for the Performing Arts in Washington, D.C.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the Opera House is described as being at the Kennedy Center for the Performing Arts in Washington, D.C.",
        node=location_leaf,
        additional_instruction="Look for 'Kennedy Center' and 'Washington, D.C.' in the answer.",
    )

    # Purpose — designed for ballet, opera, and musical theater (URL verification)
    purpose_leaf = evaluator.add_leaf(
        id="venue_4_purpose",
        desc="The theater was designed for ballet, opera, and musical theater performances",
        parent=node,
        critical=True,
    )

    # Size rank — second-largest at the Kennedy Center (URL verification)
    size_rank_leaf = evaluator.add_leaf(
        id="venue_4_size_rank",
        desc="The Opera House is described as the second-largest theater in the Kennedy Center",
        parent=node,
        critical=True,
    )

    # References confirm capacity, location, and design purpose
    ref_leaf = evaluator.add_leaf(
        id="venue_4_reference",
        desc="Provides a valid reference URL confirming the theater's capacity, location, and design purpose",
        parent=node,
        critical=True,
    )

    v4_urls = (v4.urls if v4 else []) or []
    if not _has_urls(v4_urls):
        # Fail all URL-grounded leaves due to missing sources
        for url_leaf in (purpose_leaf, size_rank_leaf, ref_leaf):
            url_leaf.score = 0.0
            url_leaf.status = "failed"
    else:
        await evaluator.verify(
            claim="The sources state that the Kennedy Center Opera House was designed for ballet, opera, and musical theater.",
            node=purpose_leaf,
            sources=v4_urls,
            additional_instruction="Look for design-purpose language explicitly mentioning ballet, opera, and musical theater.",
        )
        await evaluator.verify(
            claim="The sources state that the Opera House is the second-largest theater at the Kennedy Center.",
            node=size_rank_leaf,
            sources=v4_urls,
            additional_instruction="Look for explicit 'second-largest' claims relative to other Kennedy Center theaters (e.g., Concert Hall vs. Opera House).",
        )
        await evaluator.verify(
            claim="The sources confirm that the Kennedy Center Opera House has 2,364 seats, is located at the Kennedy Center in Washington, D.C., and is designed for ballet, opera, and musical theater.",
            node=ref_leaf,
            sources=v4_urls,
            additional_instruction="All three aspects (capacity 2,364, Kennedy Center DC location, and design purpose) must be supported.",
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
    # Initialize evaluator with parallel root strategy
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

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Add GT info for transparency
    evaluator.add_ground_truth({
        "expected_venues": {
            "venue_1": {
                "name": GT["venue_1"]["name"],
                "capacity": GT["venue_1"]["capacity_exact"],
                "note": "Largest Broadway theater by seating capacity in NYC",
            },
            "venue_2": {
                "name": GT["venue_2"]["name"],
                "capacity": GT["venue_2"]["capacity_exact"],
                "company": GT["venue_2"]["company"],
                "dates": GT["venue_2"]["dates_text"],
            },
            "venue_3": {
                "name": GT["venue_3"]["name"],
                "capacity": GT["venue_3"]["capacity_exact"],
                "address": GT["venue_3"]["address"],
            },
            "venue_4": {
                "name": GT["venue_4"]["name"],
                "capacity": GT["venue_4"]["capacity_exact"],
                "note": "Second-largest theater in the Kennedy Center",
            },
        }
    })

    # Verify each venue subtree
    await verify_venue_1(evaluator, root, extraction.venue1)
    await verify_venue_2(evaluator, root, extraction.venue2)
    await verify_venue_3(evaluator, root, extraction.venue3)
    await verify_venue_4(evaluator, root, extraction.venue4)

    # Return structured summary
    return evaluator.get_summary()