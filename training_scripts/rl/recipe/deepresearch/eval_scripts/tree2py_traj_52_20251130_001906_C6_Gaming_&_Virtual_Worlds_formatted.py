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
TASK_ID = "nc_game_city_task"
TASK_DESCRIPTION = (
    "Identify the city in North Carolina that serves as the headquarters location for a major game development company founded in 1991, which is now located at 620 Crossroads Boulevard. This same city must also host another game development studio owned by Ubisoft, located at 3001 Weston Parkway. Additionally, provide the following information: (1) The name of the company headquartered at 620 Crossroads Boulevard, its founding year, and its approximate employee count as of 2024. (2) The name of the Ubisoft-owned studio at 3001 Weston Parkway. (3) The name, location (city and specific venue), and typical month of the world's largest annual professional game industry conference. (4) The approximate number of game development companies located in this North Carolina city. (5) The regional designation of this city within North Carolina (if applicable). For each piece of information, provide supporting reference URLs from your research."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CityTaskExtraction(BaseModel):
    # City and state identification
    city_name: Optional[str] = None
    state_name: Optional[str] = None
    city_sources: List[str] = Field(default_factory=list)

    # Epic Games HQ details
    epic_company_name: Optional[str] = None
    epic_address_line: Optional[str] = None  # Expect something like "620 Crossroads Boulevard"
    epic_address_city: Optional[str] = None
    epic_address_state: Optional[str] = None
    epic_founding_year: Optional[str] = None
    epic_employee_count_2024: Optional[str] = None
    epic_sources: List[str] = Field(default_factory=list)

    # Red Storm Entertainment details
    red_storm_name: Optional[str] = None
    red_storm_address_line: Optional[str] = None  # Expect something like "3001 Weston Parkway"
    red_storm_address_city: Optional[str] = None
    red_storm_address_state: Optional[str] = None
    red_storm_ownership: Optional[str] = None  # e.g., "Ubisoft"
    red_storm_sources: List[str] = Field(default_factory=list)

    # GDC (Game Developers Conference) details
    gdc_name: Optional[str] = None
    gdc_is_annual: Optional[str] = None  # e.g., "annual", "yes"
    gdc_typical_month: Optional[str] = None  # e.g., "March"
    gdc_venue: Optional[str] = None  # e.g., "Moscone Center"
    gdc_city: Optional[str] = None  # e.g., "San Francisco"
    gdc_state: Optional[str] = None  # e.g., "California"
    gdc_attendance_approx: Optional[str] = None  # e.g., "30,000", "30k", "approximately 30,000"
    gdc_sources: List[str] = Field(default_factory=list)

    # Number of game development companies in the city
    company_count_in_city: Optional[str] = None
    company_count_sources: List[str] = Field(default_factory=list)

    # Regional designation (non-critical)
    regional_designation: Optional[str] = None  # e.g., "Research Triangle"
    regional_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_city_task() -> str:
    return """
    Extract the following fields exactly as they appear in the provided answer. If a field is not present in the answer, return null (or an empty list for URL lists). For all 'sources' fields, only include URLs explicitly present in the answer text (plain URLs or markdown links). Do not invent URLs.

    1) City and State:
       - city_name: The city identified by the answer as the location for both specified addresses.
       - state_name: The state for that city (e.g., "North Carolina" or "NC").
       - city_sources: All URLs the answer provides to support the city/state identification.

    2) Epic Games headquarters (expected address line contains "620 Crossroads Boulevard"):
       - epic_company_name
       - epic_address_line
       - epic_address_city
       - epic_address_state
       - epic_founding_year
       - epic_employee_count_2024
       - epic_sources: URLs supporting Epic Games details (name, address, founding year, 2024 employee count).

    3) Red Storm Entertainment details (expected address line contains "3001 Weston Parkway"):
       - red_storm_name
       - red_storm_address_line
       - red_storm_address_city
       - red_storm_address_state
       - red_storm_ownership  (e.g., "Ubisoft", "owned by Ubisoft")
       - red_storm_sources: URLs supporting Red Storm details (name, address, ownership).

    4) Game Developers Conference (GDC) details:
       - gdc_name
       - gdc_is_annual
       - gdc_typical_month
       - gdc_venue
       - gdc_city
       - gdc_state
       - gdc_attendance_approx
       - gdc_sources: URLs supporting GDC facts (name, annual, month, venue/location, attendance).

    5) Company count in the city:
       - company_count_in_city
       - company_count_sources: URLs supporting the stated number of game development companies in the city.

    6) Regional designation (if mentioned, e.g., "Research Triangle"):
       - regional_designation
       - regional_sources: URLs supporting the regional designation.

    Notes:
    - For numeric-like fields (years, counts), keep them as strings exactly as they appear in the answer (e.g., "1991", "4,000+", "approximately 30,000").
    - For addresses, extract them as written (e.g., "620 Crossroads Boulevard", "3001 Weston Parkway").
    - For sources lists, include every URL shown in the answer for that subtopic.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*lists: Optional[List[str]]) -> List[str]:
    """Combine multiple URL lists into a de-duplicated list while preserving order."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if u not in seen:
                combined.append(u)
                seen.add(u)
    return combined


async def add_verification_group_with_source_gating(
    evaluator: Evaluator,
    *,
    parent,
    base_id: str,
    desc: str,
    claim: str,
    sources: List[str],
    critical: bool = True,
    additional_instruction: Optional[str] = None,
) -> None:
    """
    Create a sequential sub-node to (1) ensure sources are provided, then (2) verify the claim against those sources.
    """
    group_node = evaluator.add_sequential(
        id=f"{base_id}_group",
        desc=f"{desc} (source-gated)",
        parent=parent,
        critical=critical,
    )

    # Step 1: Source presence check (critical)
    evaluator.add_custom_node(
        result=bool(sources),
        id=f"{base_id}_sources_provided",
        desc=f"Supporting reference URL(s) provided for: {desc}",
        parent=group_node,
        critical=True,
    )

    # Step 2: Actual verification against the provided URLs (critical)
    leaf_node = evaluator.add_leaf(
        id=base_id,
        desc=desc,
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf_node,
        sources=sources,
        additional_instruction=additional_instruction or "None",
    )


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_geographic_identification(evaluator: Evaluator, parent, ex: CityTaskExtraction):
    node = evaluator.add_parallel(
        id="Geographic_Identification",
        desc="Verify the target city and state are correctly identified and cited.",
        parent=parent,
        critical=True,
    )

    city_sources = combine_sources(ex.city_sources, ex.epic_sources, ex.red_storm_sources)

    # City name correct and cited
    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="City_Name_Correct_And_Cited",
        desc="Provides the correct city name (consistent with the required addresses) with a supporting reference URL.",
        claim=f"The correct city identified for the addresses is '{ex.city_name}'.",
        sources=city_sources,
        critical=True,
        additional_instruction=(
            "Confirm that the referenced page(s) indicate the relevant address(es) "
            "are located in the named city. Allow minor formatting differences."
        ),
    )

    # State is North Carolina and cited
    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="State_Is_North_Carolina_And_Cited",
        desc="Identifies the state as North Carolina with a supporting reference URL.",
        claim="The state is North Carolina.",
        sources=city_sources,
        critical=True,
        additional_instruction=(
            "Verify the source(s) explicitly indicate the city is in North Carolina (or NC)."
        ),
    )


async def build_epic_hq(evaluator: Evaluator, parent, ex: CityTaskExtraction):
    node = evaluator.add_parallel(
        id="Epic_Games_Headquarters",
        desc="Verify Epic Games HQ details at 620 Crossroads Boulevard (founded 1991; employees >=4,000 as of 2024), with citations.",
        parent=parent,
        critical=True,
    )

    epic_sources = ex.epic_sources or []

    # Epic company name and cited
    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Epic_Games_Name_Stated_And_Cited",
        desc="States the company name as Epic Games with a supporting reference URL.",
        claim="The company is named 'Epic Games'.",
        sources=epic_sources,
        critical=True,
        additional_instruction="Confirm that the referenced page identifies the company as Epic Games.",
    )

    # HQ address and city match and cited
    city_display = ex.city_name or ex.epic_address_city or ""
    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Epic_HQ_Address_620_Crossroads_In_City_And_Cited",
        desc="States the HQ address as 620 Crossroads Boulevard and the address city matches the identified city, with a supporting reference URL.",
        claim=f"Epic Games' headquarters address is 620 Crossroads Boulevard in {city_display}, North Carolina.",
        sources=epic_sources,
        critical=True,
        additional_instruction=(
            "Allow Blvd vs. Boulevard variants and minor formatting differences. "
            "The address must clearly be 620 Crossroads Boulevard in the stated city."
        ),
    )

    # Founding year 1991 and cited
    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Epic_Founding_Year_1991_And_Cited",
        desc="States Epic Games was founded in 1991 with a supporting reference URL.",
        claim="Epic Games was founded in 1991.",
        sources=epic_sources,
        critical=True,
        additional_instruction="Verify the founding year stated on the source page is 1991.",
    )

    # Employee count >= 4000 (2024) and cited
    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Epic_Employee_Count_2024_AtLeast_4000_And_Cited",
        desc="Provides an approximate 2024 employee count that is >= 4,000, with a supporting reference URL.",
        claim="As of 2024, Epic Games has at least 4,000 employees.",
        sources=epic_sources,
        critical=True,
        additional_instruction=(
            "Treat approximate phrasings (e.g., 4,000+, ~5,000) as satisfying 'at least 4,000'. "
            "Confirm the timeframe is reasonably aligned with 2024."
        ),
    )


async def build_red_storm(evaluator: Evaluator, parent, ex: CityTaskExtraction):
    node = evaluator.add_parallel(
        id="Red_Storm_Entertainment",
        desc="Verify Red Storm Entertainment (Ubisoft-owned) at 3001 Weston Parkway in the same city, with citations.",
        parent=parent,
        critical=True,
    )

    rs_sources = ex.red_storm_sources or []

    # Red Storm name and cited
    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Red_Storm_Name_Stated_And_Cited",
        desc="States the studio name as Red Storm Entertainment with a supporting reference URL.",
        claim="The studio is named 'Red Storm Entertainment'.",
        sources=rs_sources,
        critical=True,
        additional_instruction="Confirm the referenced page identifies the studio as Red Storm Entertainment.",
    )

    # Red Storm address in city and cited
    city_display = ex.city_name or ex.red_storm_address_city or ""
    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Red_Storm_Address_3001_Weston_In_City_And_Cited",
        desc="States the studio address as 3001 Weston Parkway and the address city matches the identified city, with a supporting reference URL.",
        claim=f"Red Storm Entertainment's address is 3001 Weston Parkway in {city_display}, North Carolina.",
        sources=rs_sources,
        critical=True,
        additional_instruction="Allow Pkwy vs. Parkway variants and minor formatting differences.",
    )

    # Ubisoft-owned and cited
    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Red_Storm_Ubisoft_Owned_And_Cited",
        desc="Confirms Red Storm Entertainment is owned by Ubisoft with a supporting reference URL.",
        claim="Red Storm Entertainment is owned by Ubisoft.",
        sources=rs_sources,
        critical=True,
        additional_instruction="Confirm that ownership or parent company is Ubisoft.",
    )


async def build_gdc_info(evaluator: Evaluator, parent, ex: CityTaskExtraction):
    node = evaluator.add_parallel(
        id="GDC_Convention_Information",
        desc="Verify the world's largest annual professional game industry conference details (per constraints), with citations.",
        parent=parent,
        critical=True,
    )

    gdc_sources = ex.gdc_sources or []

    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Conference_Is_GDC_And_Cited",
        desc="Identifies the conference as GDC (Game Developers Conference) with a supporting reference URL.",
        claim="The world's largest annual professional game industry conference is the Game Developers Conference (GDC).",
        sources=gdc_sources,
        critical=True,
        additional_instruction="Confirm the referenced page identifies the conference as GDC and describes it as a leading/major/large professional game industry event.",
    )

    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Conference_Is_Annual_And_Cited",
        desc="States that the conference is annual with a supporting reference URL.",
        claim="The Game Developers Conference (GDC) is an annual conference.",
        sources=gdc_sources,
        critical=True,
        additional_instruction="Verify that GDC is held annually.",
    )

    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Conference_Typical_Month_March_And_Cited",
        desc="States the typical month is March with a supporting reference URL.",
        claim="GDC is typically held in March.",
        sources=gdc_sources,
        critical=True,
        additional_instruction="Confirm that GDC commonly takes place in March (allow phrasing like 'usually in March').",
    )

    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Conference_Location_Moscone_San_Francisco_And_Cited",
        desc="States the location as Moscone Center in San Francisco, California, with a supporting reference URL.",
        claim="GDC takes place at the Moscone Center in San Francisco, California.",
        sources=gdc_sources,
        critical=True,
        additional_instruction="Verify both the venue (Moscone Center) and the city (San Francisco, CA).",
    )

    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Conference_Attendance_Approx_30000_And_Cited",
        desc="Provides attendance of approximately 30,000 with a supporting reference URL.",
        claim="GDC attendance is approximately 30,000.",
        sources=gdc_sources,
        critical=True,
        additional_instruction="Allow approximate wording and nearby figures (e.g., ~28k–35k) as 'approximately 30,000'.",
    )


async def build_city_company_count(evaluator: Evaluator, parent, ex: CityTaskExtraction):
    node = evaluator.add_parallel(
        id="City_Game_Company_Count",
        desc="Verify the stated number of game development companies in the city (per constraints) with citation.",
        parent=parent,
        critical=True,
    )

    count_sources = ex.company_count_sources or []
    city_display = ex.city_name or "the city"

    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Company_Count_Equals_5_And_Cited",
        desc="States the city hosts 5 game development companies, with a supporting reference URL.",
        claim=f"There are 5 game development companies located in {city_display}, North Carolina.",
        sources=count_sources,
        critical=True,
        additional_instruction="Only pass if the source explicitly indicates five (5) companies or lists exactly five entities.",
    )


async def build_regional_designation(evaluator: Evaluator, parent, ex: CityTaskExtraction):
    node = evaluator.add_parallel(
        id="Regional_Designation",
        desc="Check regional designation within North Carolina (if applicable).",
        parent=parent,
        critical=False,
    )

    regional_sources = ex.regional_sources or []
    city_display = ex.city_name or "the city"

    # Non-critical: Research Triangle mention and cited
    # We still gate by source presence.
    await add_verification_group_with_source_gating(
        evaluator,
        parent=node,
        base_id="Research_Triangle_Mentioned_And_Cited",
        desc="Identifies the city as part of the Research Triangle area with a supporting reference URL.",
        claim=f"{city_display} is part of North Carolina's Research Triangle region.",
        sources=regional_sources,
        critical=False,
        additional_instruction="Allow synonyms like 'the Triangle' or 'Raleigh–Durham–Chapel Hill Triangle'.",
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
    Evaluate an answer for the NC city and related gaming industry facts task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level: independent major sections
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

    # NOTE: Set root as non-critical to allow inclusion of non-critical sub-criteria (e.g., Regional_Designation).
    root.critical = False

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_city_task(),
        template_class=CityTaskExtraction,
        extraction_name="city_task_extraction",
    )

    # Build all rubric branches
    await build_geographic_identification(evaluator, root, extraction)
    await build_epic_hq(evaluator, root, extraction)
    await build_red_storm(evaluator, root, extraction)
    await build_gdc_info(evaluator, root, extraction)
    await build_city_company_count(evaluator, root, extraction)
    await build_regional_designation(evaluator, root, extraction)

    return evaluator.get_summary()