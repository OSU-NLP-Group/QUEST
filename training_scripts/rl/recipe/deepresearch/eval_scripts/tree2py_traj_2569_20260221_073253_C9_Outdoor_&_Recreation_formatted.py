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
TASK_ID = "tahoe_family_mlk_2026"
TASK_DESCRIPTION = (
    "You are planning a family ski vacation to the Lake Tahoe region during Martin Luther King Jr. Day weekend "
    "in January 2026 (which includes Monday, January 19, 2026). Your family will be flying into Reno-Tahoe International "
    "Airport (RNO) and includes beginner skiers and a 3-year-old child who will need childcare while the adults ski.\n\n"
    "Identify 4 different ski resorts in the California/Nevada Lake Tahoe area that meet ALL of the following requirements:\n\n"
    "1. Located within 90 minutes drive from Reno-Tahoe International Airport\n"
    "2. Have a vertical drop of at least 2,000 feet\n"
    "3. Have at least 2,000 skiable acres\n"
    "4. Have at least 15% of their terrain designated as beginner (green circle) runs\n"
    "5. Offer on-mountain childcare facilities that accept children ages 3-4 years\n"
    "6. Have on-mountain dining facilities (restaurants or cafeterias)\n"
    "7. Have ski patrol and medical/first aid facilities\n"
    "8. Offer on-site equipment rental services\n"
    "9. Provide ski lesson programs for children\n"
    "10. Have parking facilities or shuttle service access\n"
    "11. Are confirmed to be fully operational during mid-January 2026\n"
    "12. Have all terrain statistics (vertical drop, acreage, beginner terrain percentage) verifiable from official resort "
    "sources or reliable ski industry websites\n\n"
    "For each of the 4 resorts, provide:\n"
    "- Resort name\n"
    "- Vertical drop (in feet)\n"
    "- Skiable acreage\n"
    "- Beginner terrain percentage\n"
    "- Approximate drive time from RNO airport\n"
    "- Childcare age range accepted\n"
    "- URL reference for vertical drop verification\n"
    "- URL reference for skiable acreage verification\n"
    "- URL reference for beginner terrain verification\n"
    "- URL reference for airport access/location information\n"
    "- URL reference for childcare services information"
)

MLK_DAY_2026 = "Monday, January 19, 2026"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResortStats(BaseModel):
    vertical_drop_ft: Optional[str] = None
    vertical_drop_url: Optional[str] = None
    skiable_acreage: Optional[str] = None
    skiable_acreage_url: Optional[str] = None
    beginner_terrain_percent: Optional[str] = None
    beginner_terrain_url: Optional[str] = None


class ResortAccess(BaseModel):
    drive_time_minutes_from_rno: Optional[str] = None
    airport_access_url: Optional[str] = None
    parking_url: Optional[str] = None
    shuttle_url: Optional[str] = None


class ResortChildcare(BaseModel):
    childcare_age_range: Optional[str] = None  # e.g., "3-4", "6 months to 5 years"
    childcare_url: Optional[str] = None


class ResortAmenities(BaseModel):
    dining_url: Optional[str] = None
    ski_patrol_url: Optional[str] = None
    rentals_url: Optional[str] = None
    lessons_url: Optional[str] = None
    operations_url: Optional[str] = None  # calendar/schedule confirming January operations


class ResortItem(BaseModel):
    name: Optional[str] = None
    stats: ResortStats = Field(default_factory=ResortStats)
    access: ResortAccess = Field(default_factory=ResortAccess)
    childcare: ResortChildcare = Field(default_factory=ResortChildcare)
    amenities: ResortAmenities = Field(default_factory=ResortAmenities)


class ResortsExtraction(BaseModel):
    resorts: List[ResortItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt builder                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_resorts() -> str:
    return """
    Extract information about up to four ski resorts in the California/Nevada Lake Tahoe region mentioned in the answer.
    For each resort, extract ONLY what is explicitly provided in the answer text. Do not invent values or URLs.

    For each resort, return an object with the following fields:
    - name: The resort name exactly as written in the answer.
    - stats:
        - vertical_drop_ft: The stated vertical drop (in feet) as a string (e.g., "2850 ft", "2,850 feet").
        - vertical_drop_url: The URL provided to verify the vertical drop value.
        - skiable_acreage: The stated skiable acreage as a string (e.g., "6000 acres", "6,000 ac").
        - skiable_acreage_url: The URL provided to verify the acreage.
        - beginner_terrain_percent: The stated percentage of beginner terrain (e.g., "25%", "15 percent").
        - beginner_terrain_url: The URL provided to verify beginner terrain percentage.
    - access:
        - drive_time_minutes_from_rno: The approximate drive time from Reno-Tahoe International Airport (RNO) as a string, preferably in minutes (e.g., "45 min", "1 hr 20 min").
        - airport_access_url: The URL that supports travel time or directions/access from RNO or general location relative to Reno/Tahoe.
        - parking_url: The URL that mentions parking facilities (if provided in the answer).
        - shuttle_url: The URL that mentions shuttle service access (if provided).
    - childcare:
        - childcare_age_range: The age range accepted by the on-mountain childcare (e.g., "3-4", "6 months–5 years").
        - childcare_url: The URL that describes childcare services and accepted ages.
    - amenities:
        - dining_url: URL that indicates on-mountain dining facilities (restaurants/cafeterias).
        - ski_patrol_url: URL that mentions ski patrol or medical/first aid facilities.
        - rentals_url: URL that indicates on-site equipment rental services.
        - lessons_url: URL that describes children's ski lesson programs.
        - operations_url: URL that confirms operational status or calendar around mid-January 2026.

    Rules:
    - Extract only what appears in the answer. If any field is missing, set it to null.
    - For URLs: return actual URLs shown in the answer, including full protocol.
    - Keep numeric values as strings to preserve the original formatting.
    - Return a JSON object with a single key "resorts" that is an array of resort objects.
    - If more than four resorts are mentioned, return the first four. If fewer, return however many are present.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def non_empty_urls(urls: List[Optional[str]]) -> List[str]:
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def resort_all_urls(resort: ResortItem) -> List[str]:
    return non_empty_urls([
        resort.stats.vertical_drop_url,
        resort.stats.skiable_acreage_url,
        resort.stats.beginner_terrain_url,
        resort.access.airport_access_url,
        resort.access.parking_url,
        resort.access.shuttle_url,
        resort.childcare.childcare_url,
        resort.amenities.dining_url,
        resort.amenities.ski_patrol_url,
        resort.amenities.rentals_url,
        resort.amenities.lessons_url,
        resort.amenities.operations_url,
    ])


def first_non_null(*values: Optional[str]) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# --------------------------------------------------------------------------- #
# Verification for a single resort                                            #
# --------------------------------------------------------------------------- #
async def verify_single_resort(
    evaluator: Evaluator,
    parent_node,
    resort: ResortItem,
    ordinal: int
) -> None:
    """
    Build verification subtree for one resort with all criteria.
    """
    # Create resort node (parallel aggregation, non-critical to allow partial credit across resorts)
    resort_node = evaluator.add_parallel(
        id=f"resort_{ordinal}",
        desc=f"{['First','Second','Third','Fourth'][ordinal-1]} qualifying ski resort meeting all criteria",
        parent=parent_node,
        critical=False
    )

    # 1) Name & Lake Tahoe location
    name_loc_node = evaluator.add_leaf(
        id=f"resort_{ordinal}_name_location",
        desc="Resort name and confirmed location in CA/NV Lake Tahoe region",
        parent=resort_node,
        critical=True
    )
    name_for_claim = resort.name or "the resort"
    await evaluator.verify(
        claim=f"The ski resort named '{name_for_claim}' is located within the Lake Tahoe region in either California or Nevada.",
        node=name_loc_node,
        sources=resort_all_urls(resort),
        additional_instruction="Check any provided resort pages to confirm the resort is in the Lake Tahoe region (North/West/South Lake Tahoe) and lies within CA or NV."
    )

    # 2) Vertical drop (>=2000 ft) with official source
    vert_group = evaluator.add_parallel(
        id=f"resort_{ordinal}_vertical_drop",
        desc="Vertical drop of at least 2,000 feet with official source",
        parent=resort_node,
        critical=True
    )

    vert_value_leaf = evaluator.add_leaf(
        id=f"resort_{ordinal}_vertical_value",
        desc="Specific vertical drop measurement in feet meets or exceeds 2,000 feet",
        parent=vert_group,
        critical=True
    )
    vert_value = resort.stats.vertical_drop_ft or ""
    await evaluator.verify(
        claim=f"The vertical drop of '{name_for_claim}' is {vert_value} and is at least 2,000 feet.",
        node=vert_value_leaf,
        sources=resort.stats.vertical_drop_url,
        additional_instruction="Verify the numeric vertical drop value on the provided page and confirm it is >= 2,000 ft. Allow minor formatting (commas, 'feet' vs 'ft')."
    )

    evaluator.add_custom_node(
        result=bool(resort.stats.vertical_drop_url and resort.stats.vertical_drop_url.strip()),
        id=f"resort_{ordinal}_vertical_url",
        desc="URL reference for vertical drop verification",
        parent=vert_group,
        critical=True
    )

    # 3) Skiable acreage (>=2000 acres) with official source
    acres_group = evaluator.add_parallel(
        id=f"resort_{ordinal}_skiable_acreage",
        desc="At least 2,000 skiable acres with official source",
        parent=resort_node,
        critical=True
    )
    acres_value_leaf = evaluator.add_leaf(
        id=f"resort_{ordinal}_acreage_value",
        desc="Specific skiable acreage number meets or exceeds 2,000 acres",
        parent=acres_group,
        critical=True
    )
    acres_value = resort.stats.skiable_acreage or ""
    await evaluator.verify(
        claim=f"'{name_for_claim}' has {acres_value} of skiable terrain and this is at least 2,000 acres.",
        node=acres_value_leaf,
        sources=resort.stats.skiable_acreage_url,
        additional_instruction="Verify the acreage number on the page and confirm it is >= 2,000 acres. Allow number formatting variations."
    )

    evaluator.add_custom_node(
        result=bool(resort.stats.skiable_acreage_url and resort.stats.skiable_acreage_url.strip()),
        id=f"resort_{ordinal}_acreage_url",
        desc="URL reference for acreage verification",
        parent=acres_group,
        critical=True
    )

    # 4) Beginner terrain (>=15%) with official source
    beg_group = evaluator.add_parallel(
        id=f"resort_{ordinal}_beginner_terrain",
        desc="At least 15% beginner (green) terrain with official source",
        parent=resort_node,
        critical=True
    )
    beg_pct_leaf = evaluator.add_leaf(
        id=f"resort_{ordinal}_beginner_percentage",
        desc="Specific percentage of beginner terrain meets or exceeds 15%",
        parent=beg_group,
        critical=True
    )
    beg_pct = resort.stats.beginner_terrain_percent or ""
    await evaluator.verify(
        claim=f"The percentage of beginner (green circle) terrain at '{name_for_claim}' is {beg_pct} and is at least 15%.",
        node=beg_pct_leaf,
        sources=resort.stats.beginner_terrain_url,
        additional_instruction="Verify the beginner terrain percentage on the page and confirm it is >= 15%. Allow rounding (e.g., 14.9 ≈ 15%)."
    )

    evaluator.add_custom_node(
        result=bool(resort.stats.beginner_terrain_url and resort.stats.beginner_terrain_url.strip()),
        id=f"resort_{ordinal}_beginner_url",
        desc="URL reference for beginner terrain verification",
        parent=beg_group,
        critical=True
    )

    # 5) Airport access (<=90 minutes from RNO)
    access_group = evaluator.add_parallel(
        id=f"resort_{ordinal}_airport_access",
        desc="Within 90 minutes drive from Reno-Tahoe International Airport",
        parent=resort_node,
        critical=True
    )
    drive_leaf = evaluator.add_leaf(
        id=f"resort_{ordinal}_drive_time",
        desc="Approximate drive time in minutes from RNO airport is 90 minutes or less",
        parent=access_group,
        critical=True
    )
    drive_str = resort.access.drive_time_minutes_from_rno or ""
    await evaluator.verify(
        claim=f"The approximate drive time from RNO to '{name_for_claim}' is 90 minutes or less. Reported time: {drive_str}.",
        node=drive_leaf,
        sources=resort.access.airport_access_url,
        additional_instruction="Use the provided travel/access page to confirm the typical drive time from RNO. If stated in hours, convert approximately (e.g., 1.5 hr ≈ 90 min)."
    )

    evaluator.add_custom_node(
        result=bool(resort.access.airport_access_url and resort.access.airport_access_url.strip()),
        id=f"resort_{ordinal}_access_url",
        desc="URL reference for airport access information",
        parent=access_group,
        critical=True
    )

    # 6) Childcare: accepts ages 3–4
    childcare_group = evaluator.add_parallel(
        id=f"resort_{ordinal}_childcare",
        desc="On-mountain childcare accepting ages 3-4 years",
        parent=resort_node,
        critical=True
    )
    childcare_leaf = evaluator.add_leaf(
        id=f"resort_{ordinal}_childcare_ages",
        desc="Age range accepted by childcare facility includes 3-4 year olds",
        parent=childcare_group,
        critical=True
    )
    age_range = resort.childcare.childcare_age_range or ""
    await evaluator.verify(
        claim=f"The resort's on-mountain childcare accepts children ages 3–4 years. Reported accepted ages: {age_range}.",
        node=childcare_leaf,
        sources=resort.childcare.childcare_url,
        additional_instruction="Verify on the childcare page that the accepted ages explicitly include 3 and 4 years (or a range that covers them)."
    )

    evaluator.add_custom_node(
        result=bool(resort.childcare.childcare_url and resort.childcare.childcare_url.strip()),
        id=f"resort_{ordinal}_childcare_url",
        desc="URL reference for childcare services",
        parent=childcare_group,
        critical=True
    )

    # 7) Dining facilities available
    dining_leaf = evaluator.add_leaf(
        id=f"resort_{ordinal}_dining",
        desc="On-mountain dining facilities available",
        parent=resort_node,
        critical=True
    )
    dining_sources = non_empty_urls([resort.amenities.dining_url]) or resort_all_urls(resort)
    await evaluator.verify(
        claim=f"'{name_for_claim}' provides on-mountain dining facilities (restaurants or cafeterias).",
        node=dining_leaf,
        sources=dining_sources,
        additional_instruction="Look for mentions of restaurants, cafeterias, or dining options on resort pages."
    )

    # 8) Ski patrol / medical / first aid present
    patrol_leaf = evaluator.add_leaf(
        id=f"resort_{ordinal}_ski_patrol",
        desc="Ski patrol and medical/first aid facilities present",
        parent=resort_node,
        critical=True
    )
    patrol_sources = non_empty_urls([resort.amenities.ski_patrol_url]) or resort_all_urls(resort)
    await evaluator.verify(
        claim=f"'{name_for_claim}' has ski patrol services and medical/first aid facilities available.",
        node=patrol_leaf,
        sources=patrol_sources,
        additional_instruction="Confirm presence of Ski Patrol and first aid/medical facilities on official resort pages."
    )

    # 9) On-site equipment rentals
    rentals_leaf = evaluator.add_leaf(
        id=f"resort_{ordinal}_rentals",
        desc="On-site equipment rental services available",
        parent=resort_node,
        critical=True
    )
    rentals_sources = non_empty_urls([resort.amenities.rentals_url]) or resort_all_urls(resort)
    await evaluator.verify(
        claim=f"'{name_for_claim}' offers on-site equipment rental services.",
        node=rentals_leaf,
        sources=rentals_sources,
        additional_instruction="Verify rental services (ski/snowboard, boots) available at on-mountain/base locations."
    )

    # 10) Children's ski lesson programs
    lessons_leaf = evaluator.add_leaf(
        id=f"resort_{ordinal}_lessons",
        desc="Children's ski lesson programs offered",
        parent=resort_node,
        critical=True
    )
    lessons_sources = non_empty_urls([resort.amenities.lessons_url]) or resort_all_urls(resort)
    await evaluator.verify(
        claim=f"'{name_for_claim}' provides ski lesson programs for children.",
        node=lessons_leaf,
        sources=lessons_sources,
        additional_instruction="Confirm kids' lessons (age-specific programs) offered at the resort."
    )

    # 11) Parking or shuttle access
    parking_leaf = evaluator.add_leaf(
        id=f"resort_{ordinal}_parking",
        desc="Parking facilities or shuttle service access provided",
        parent=resort_node,
        critical=True
    )
    parking_sources = non_empty_urls([resort.access.parking_url, resort.access.shuttle_url, resort.access.airport_access_url]) or resort_all_urls(resort)
    await evaluator.verify(
        claim=f"'{name_for_claim}' provides parking facilities or has shuttle service access.",
        node=parking_leaf,
        sources=parking_sources,
        additional_instruction="Look for parking information (lots/garages) or shuttle access details on resort pages."
    )

    # 12) Confirm operational mid-January 2026
    ops_leaf = evaluator.add_leaf(
        id=f"resort_{ordinal}_january_operations",
        desc="Confirmed operational during mid-January 2026",
        parent=resort_node,
        critical=True
    )
    ops_sources = non_empty_urls([resort.amenities.operations_url]) or resort_all_urls(resort)
    await evaluator.verify(
        claim=f"'{name_for_claim}' is operational during mid-January 2026, including the MLK weekend around {MLK_DAY_2026}.",
        node=ops_leaf,
        sources=ops_sources,
        additional_instruction="Check winter operating calendar/notices for January 2026 (e.g., Jan 17–19) to confirm lifts/terrain are scheduled to open (weather permitting)."
    )

    # 13) Stats verifiable from official or reliable sources
    # Interpreted here as presence of specific verification URLs for all three stats.
    evaluator.add_custom_node(
        result=all([
            bool(resort.stats.vertical_drop_url and resort.stats.vertical_drop_url.strip()),
            bool(resort.stats.skiable_acreage_url and resort.stats.skiable_acreage_url.strip()),
            bool(resort.stats.beginner_terrain_url and resort.stats.beginner_terrain_url.strip()),
        ]),
        id=f"resort_{ordinal}_official_stats",
        desc="All terrain statistics verifiable from official resort sources",
        parent=resort_node,
        critical=True
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
) -> Dict:
    """
    Evaluate an answer for the Lake Tahoe family MLK 2026 ski resort task.
    """
    # Initialize evaluator (root kept non-critical to allow partial scoring across resorts
    # and to satisfy critical-node child consistency constraints)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify 4 ski resorts in the California/Nevada Lake Tahoe region that meet all specified criteria for a family ski vacation during MLK weekend 2026",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Record MLK date context
    evaluator.add_custom_info({"mlk_day_2026": MLK_DAY_2026}, info_type="context", info_name="holiday_info")

    # Extract resorts data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_resorts(),
        template_class=ResortsExtraction,
        extraction_name="resorts_extraction",
    )

    # Normalize to exactly 4 resorts (pad with empty entries if fewer)
    resorts: List[ResortItem] = list(extraction.resorts[:4])
    while len(resorts) < 4:
        resorts.append(ResortItem())

    # Build verification subtrees for each of the four resorts
    for i in range(4):
        await verify_single_resort(evaluator, root, resorts[i], i + 1)

    # Return final summary
    return evaluator.get_summary()