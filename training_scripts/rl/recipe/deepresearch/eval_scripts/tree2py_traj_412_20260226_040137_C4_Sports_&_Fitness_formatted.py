import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_cactus_league_stadium_2026"
TASK_DESCRIPTION = (
    "Identify the spring training stadium with the largest seating capacity among all Cactus League facilities in Arizona for the 2026 season. "
    "Provide the following verified information about this stadium:\n\n"
    "1. The official stadium name\n"
    "2. The complete street address (including street number, street name, city, state, and ZIP code)\n"
    "3. The seating capacity\n"
    "4. The MLB team(s) that use this facility as their spring training home\n"
    "5. At least one reference URL from an official source (MLB, Cactus League, or team website) that supports your answer"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AddressInfo(BaseModel):
    full_address: Optional[str] = None
    street_number: Optional[str] = None
    street_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None


class StadiumExtraction(BaseModel):
    stadium_name: Optional[str] = None
    address: AddressInfo = Field(default_factory=AddressInfo)
    seating_capacity: Optional[str] = None
    home_teams: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)
    largest_support_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadium_info() -> str:
    return """
    From the provided answer text, extract the single stadium that the answer claims is the largest (by seating capacity) among all Cactus League spring training facilities in Arizona for the 2026 season.

    Return the following fields:
    - stadium_name: The official name of the stadium as stated in the answer.
    - address: 
        - full_address: The complete address string as presented in the answer (street number and name, city, state, ZIP).
        - street_number, street_name, city, state, zip: Fill these if explicitly present; otherwise leave null.
    - seating_capacity: The seating capacity number/string as written in the answer (do not normalize; keep commas or qualifiers if present).
    - home_teams: The MLB team(s) that use this stadium for spring training as listed in the answer. Return an array of team names.
    - reference_urls: All URLs the answer cites as sources for this stadium's information. Include only URLs that are explicitly present in the answer text (including markdown links). Do not invent any URL.
    - largest_support_urls: The subset of URLs in the answer that directly support the 'largest seating capacity in the Cactus League' claim (e.g., league-wide lists or explicit statements). If none are clearly marked or implied, return an empty array (do not guess).

    IMPORTANT:
    - Extract only what is explicitly present in the answer text. Do not add or infer any information not stated.
    - For URLs, include only valid URLs explicitly included in the answer.
    - If any field is not present in the answer, set it to null (or empty list where appropriate).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_complete_address_like(addr: Optional[str]) -> bool:
    if not addr:
        return False
    # Heuristic: must contain digits (street number), a comma, and a 5-digit ZIP;
    # and reference AZ/Arizona (since Cactus League is in AZ).
    has_number = bool(re.search(r"\b\d{1,6}\b", addr))
    has_zip = bool(re.search(r"\b\d{5}(?:-\d{4})?\b", addr))
    mentions_az = (" AZ" in addr) or ("Arizona" in addr) or (",AZ" in addr) or (", AZ" in addr)
    has_comma = "," in addr
    return has_number and has_zip and mentions_az and has_comma


def join_teams(teams: List[str]) -> str:
    if not teams:
        return ""
    if len(teams) == 1:
        return teams[0]
    return ", ".join(teams[:-1]) + " and " + teams[-1]


def union_sources(primary: List[str], secondary: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for url in (primary or []) + (secondary or []):
        if url and url not in seen:
            seen.add(url)
            result.append(url)
    return result


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_stadium_identification(
    evaluator: Evaluator,
    parent_node,
    info: StadiumExtraction
) -> None:
    # Critical node: Stadium Identification (use SEQUENTIAL to gate steps)
    ident_node = evaluator.add_sequential(
        id="Stadium_Identification",
        desc="The stadium identified must verifiably have the highest seating capacity among all Cactus League stadiums in Arizona for the 2026 season",
        parent=parent_node,
        critical=True,
    )

    all_sources = union_sources(info.largest_support_urls, info.reference_urls)

    # Step 0: Existence of comparison/identification sources
    evaluator.add_custom_node(
        result=len(all_sources) > 0,
        id="stadium_id_has_sources",
        desc="At least one supporting reference URL is provided to validate identification and comparison claims",
        parent=ident_node,
        critical=True
    )

    # Step 1: Verify it's a Cactus League facility in Arizona (2026 season context)
    is_cl_az_leaf = evaluator.add_leaf(
        id="stadium_is_cactus_league_az",
        desc="Stadium is a Cactus League spring training facility in Arizona (for the 2026 season context)",
        parent=ident_node,
        critical=True
    )
    stadium_name = info.stadium_name or "the stadium"
    cl_claim = (
        f"{stadium_name} is a Cactus League spring training facility located in Arizona for the 2026 season."
    )
    await evaluator.verify(
        claim=cl_claim,
        node=is_cl_az_leaf,
        sources=all_sources,
        additional_instruction=(
            "Confirm that the stadium is a Cactus League (Arizona) spring training ballpark used by an MLB team. "
            "If the page does not explicitly mention '2026', but clearly identifies it as an active Cactus League venue "
            "for MLB spring training, consider it valid unless there is explicit contradictory information."
        ),
    )

    # Step 2: Verify largest seating capacity among all Cactus League stadiums in Arizona
    largest_leaf = evaluator.add_leaf(
        id="stadium_has_largest_capacity",
        desc="Stadium has the largest seating capacity among all Cactus League stadiums in Arizona (2026)",
        parent=ident_node,
        critical=True
    )
    largest_claim = (
        f"Among all Cactus League spring training facilities in Arizona for the 2026 season, {stadium_name} has the largest seating capacity."
    )
    largest_sources = info.largest_support_urls if info.largest_support_urls else all_sources
    await evaluator.verify(
        claim=largest_claim,
        node=largest_leaf,
        sources=largest_sources,
        additional_instruction=(
            "Look for explicit statements or league-wide lists/tables that compare capacities across Cactus League stadiums. "
            "The claim should be supported by the source(s); if the sources do not allow you to determine 'largest', mark as not supported. "
            "Focus on 'seating capacity' rather than record attendance or standing-room capacity."
        ),
    )


async def verify_stadium_name(
    evaluator: Evaluator,
    parent_node,
    info: StadiumExtraction
) -> None:
    name_node = evaluator.add_sequential(
        id="Stadium_Name",
        desc="The official name of the stadium must be provided",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.stadium_name and info.stadium_name.strip()),
        id="stadium_name_exists",
        desc="The stadium name is provided",
        parent=name_node,
        critical=True
    )

    name_leaf = evaluator.add_leaf(
        id="stadium_name_supported",
        desc="Official stadium name is supported by cited sources",
        parent=name_node,
        critical=True
    )
    all_sources = union_sources(info.reference_urls, info.largest_support_urls)
    claim = f"The official name of the stadium is '{info.stadium_name or ''}'."
    await evaluator.verify(
        claim=claim,
        node=name_leaf,
        sources=all_sources,
        additional_instruction=(
            "Verify the current official stadium name as shown on an official MLB/team site or the official Cactus League site. "
            "Allow minor stylistic variations (e.g., punctuation, 'Ballpark' vs 'Ball Park') if clearly the same named facility."
        ),
    )


async def verify_address(
    evaluator: Evaluator,
    parent_node,
    info: StadiumExtraction
) -> None:
    addr_node = evaluator.add_sequential(
        id="Complete_Address",
        desc="The complete street address including street number, street name, city, state, and ZIP code must be provided",
        parent=parent_node,
        critical=True
    )

    full_addr = info.address.full_address or ""
    evaluator.add_custom_node(
        result=is_complete_address_like(full_addr),
        id="complete_address_exists",
        desc="A complete street address (street number/name, city, state AZ, and ZIP) is provided",
        parent=addr_node,
        critical=True
    )

    addr_leaf = evaluator.add_leaf(
        id="complete_address_supported",
        desc="The complete street address is supported by cited sources",
        parent=addr_node,
        critical=True
    )
    all_sources = union_sources(info.reference_urls, info.largest_support_urls)
    claim = f"The complete street address of '{info.stadium_name or 'the stadium'}' is '{full_addr}'."
    await evaluator.verify(
        claim=claim,
        node=addr_leaf,
        sources=all_sources,
        additional_instruction=(
            "Confirm the full postal address including street number, street name, city, state (AZ), and ZIP code. "
            "Formatting differences (commas, abbreviations) are acceptable as long as the address is equivalent."
        ),
    )


async def verify_capacity(
    evaluator: Evaluator,
    parent_node,
    info: StadiumExtraction
) -> None:
    cap_node = evaluator.add_sequential(
        id="Seating_Capacity",
        desc="The official seating capacity of the stadium must be stated",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.seating_capacity and info.seating_capacity.strip()),
        id="seating_capacity_exists",
        desc="Seating capacity is provided",
        parent=cap_node,
        critical=True
    )

    cap_leaf = evaluator.add_leaf(
        id="seating_capacity_supported",
        desc="Seating capacity is supported by cited sources",
        parent=cap_node,
        critical=True
    )
    all_sources = union_sources(info.reference_urls, info.largest_support_urls)
    cap_str = info.seating_capacity or ""
    claim = f"The seating capacity of '{info.stadium_name or 'the stadium'}' is {cap_str}."
    await evaluator.verify(
        claim=claim,
        node=cap_leaf,
        sources=all_sources,
        additional_instruction=(
            "Verify the 'seating capacity' figure (fixed seats). Allow minor rounding or formatting differences (e.g., commas). "
            "Do not confuse with total attendance record, standing-room capacity, or berm capacity unless the page clearly labels it as 'seating capacity'."
        ),
    )


async def verify_home_team(
    evaluator: Evaluator,
    parent_node,
    info: StadiumExtraction
) -> None:
    team_node = evaluator.add_sequential(
        id="Home_Team",
        desc="The MLB team(s) that use this stadium for spring training must be identified",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.home_teams and len(info.home_teams) > 0),
        id="home_team_exists",
        desc="At least one MLB spring training home team is provided",
        parent=team_node,
        critical=True
    )

    team_leaf = evaluator.add_leaf(
        id="home_team_supported",
        desc="MLB spring training home team(s) are supported by cited sources",
        parent=team_node,
        critical=True
    )
    all_sources = union_sources(info.reference_urls, info.largest_support_urls)
    teams_str = join_teams(info.home_teams)
    claim = f"The MLB spring training home team(s) at '{info.stadium_name or 'the stadium'}' are: {teams_str}."
    await evaluator.verify(
        claim=claim,
        node=team_leaf,
        sources=all_sources,
        additional_instruction=(
            "Confirm the MLB team(s) that use this facility as their spring training home. "
            "Some venues host two MLB clubs; ensure both are included if applicable."
        ),
    )


async def verify_supporting_reference(
    evaluator: Evaluator,
    parent_node,
    info: StadiumExtraction
) -> None:
    ref_node = evaluator.add_sequential(
        id="Supporting_Reference",
        desc="At least one valid reference URL from an official MLB, Cactus League, or reliable source must be provided",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.reference_urls and len(info.reference_urls) > 0),
        id="supporting_reference_exists",
        desc="At least one reference URL is provided",
        parent=ref_node,
        critical=True
    )

    # Official source check (at least one)
    official_leaf = evaluator.add_leaf(
        id="supporting_reference_official",
        desc="At least one provided reference URL is an official MLB, MLB team, or official Cactus League website",
        parent=ref_node,
        critical=True
    )

    # Use multi-URL verification to pass if any URL qualifies as official
    await evaluator.verify(
        claim=(
            "This page is an official source: either the official MLB website (mlb.com or its subpaths), "
            "an official MLB team website/domain, or the official Cactus League website (cactusleague.com)."
        ),
        node=official_leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Judge 'official' primarily by domain and on-page branding. Accept: mlb.com (including team subpaths like mlb.com/angels), "
            "team official domains that clearly indicate official MLB team ownership, and cactusleague.com. "
            "Do NOT consider Wikipedia, news outlets, travel blogs, or ticket resellers as official."
        ),
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
) -> Dict[str, Any]:
    # Initialize evaluator and root
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

    # Extraction
    extracted: StadiumExtraction = await evaluator.extract(
        prompt=prompt_extract_stadium_info(),
        template_class=StadiumExtraction,
        extraction_name="stadium_extraction",
    )

    # Build critical main node per rubric
    main_node = evaluator.add_parallel(
        id="Largest_Cactus_League_Stadium_Information",
        desc="Verify that comprehensive and accurate information is provided about the largest spring training stadium in Arizona's Cactus League for the 2026 season, based on seating capacity",
        parent=root,
        critical=True
    )

    # Sub-verifications (all critical per rubric)
    await verify_stadium_identification(evaluator, main_node, extracted)
    await verify_stadium_name(evaluator, main_node, extracted)
    await verify_address(evaluator, main_node, extracted)
    await verify_capacity(evaluator, main_node, extracted)
    await verify_home_team(evaluator, main_node, extracted)
    await verify_supporting_reference(evaluator, main_node, extracted)

    # Optional: record extracted summary for convenience
    evaluator.add_custom_info(
        info={
            "extracted_stadium_name": extracted.stadium_name,
            "extracted_address": extracted.address.dict(),
            "extracted_seating_capacity": extracted.seating_capacity,
            "extracted_home_teams": extracted.home_teams,
            "reference_urls": extracted.reference_urls,
            "largest_support_urls": extracted.largest_support_urls,
        },
        info_type="extraction_summary"
    )

    return evaluator.get_summary()