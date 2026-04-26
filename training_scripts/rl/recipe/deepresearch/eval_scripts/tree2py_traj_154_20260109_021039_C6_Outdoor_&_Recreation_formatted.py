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
TASK_ID = "usfs_campground_solution"
TASK_DESCRIPTION = """
Identify a developed campground managed by the U.S. Forest Service in a National Forest in a western U.S. state that meets all of the following requirements:
- Has at least one group campsite designed to accommodate 20 or more people;
- Has accessible camping units that meet ADA standards, with at least 20% of camping units (or a minimum of 2 units, whichever is greater) being accessible;
- Can accommodate RVs that are 35 feet or longer in length;
- Provides potable water that meets federal clean water standards;
- Includes standard campsite amenities: picnic tables, fire rings (or fire grills), and toilet facilities;
- Is bookable through Recreation.gov with the standard 6-month advance reservation window;
- Is open and available for camping during the summer season (June through August).
Provide the name of the campground, the state it's located in, the specific National Forest, and supporting URLs that verify each category of requirements.
"""

# Western U.S. states (full names and abbreviations)
WESTERN_STATES = {
    "washington": "WA", "oregon": "OR", "california": "CA", "nevada": "NV", "arizona": "AZ",
    "new mexico": "NM", "colorado": "CO", "utah": "UT", "wyoming": "WY", "montana": "MT",
    "idaho": "ID", "alaska": "AK", "hawaii": "HI"
}
WESTERN_ABBREVIATIONS = set(WESTERN_STATES.values())


def _normalize_str(s: Optional[str]) -> str:
    return (s or "").strip()


def is_western_state(state_text: Optional[str]) -> bool:
    if not state_text:
        return False
    s = state_text.strip().lower()
    # Remove punctuation and parentheses content often like "California (CA)"
    s = s.replace(",", " ").replace("(", " ").replace(")", " ").strip()
    tokens = [t for t in s.split() if t]
    # If two-letter abbreviation present
    for tok in tokens:
        tok_upper = tok.upper()
        if tok_upper in WESTERN_ABBREVIATIONS:
            return True
    # Check full names
    joined = " ".join(tokens)
    return joined in WESTERN_STATES.keys()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GroupSiteInfo(BaseModel):
    has_group_site: Optional[bool] = None
    min_capacity_number: Optional[str] = None  # numeric as string if present
    capacity_text: Optional[str] = None        # free text like "up to 50"


class AccessibilityInfo(BaseModel):
    has_accessible_units: Optional[bool] = None
    accessible_units_count: Optional[str] = None  # numeric as string
    total_units_count: Optional[str] = None       # numeric as string
    accessible_units_pct: Optional[str] = None    # percent text if present
    rule_claim_text: Optional[str] = None         # any explicit claim text regarding the rule


class RVInfo(BaseModel):
    accommodates_35ft_or_more: Optional[bool] = None
    max_length_claim: Optional[str] = None  # e.g., "45 ft max RV length"


class WaterInfo(BaseModel):
    potable_water_claim: Optional[str] = None  # e.g., "drinking water available"
    meets_federal_standards_claim: Optional[bool] = None  # if explicitly stated


class AmenitiesInfo(BaseModel):
    picnic_tables: Optional[bool] = None
    fire_rings_or_grills: Optional[bool] = None
    toilets: Optional[bool] = None


class ReservationInfo(BaseModel):
    bookable_on_recgov: Optional[bool] = None
    uses_standard_6_month_window: Optional[bool] = None
    recgov_url: Optional[str] = None


class SeasonInfo(BaseModel):
    open_in_june_july_august: Optional[bool] = None
    season_text: Optional[str] = None


class CampgroundExtraction(BaseModel):
    # Identification and location
    name: Optional[str] = None
    state: Optional[str] = None
    national_forest: Optional[str] = None
    is_usfs_managed: Optional[bool] = None
    is_developed: Optional[bool] = None

    # Evidence URLs by category
    id_loc_mgmt_urls: List[str] = Field(default_factory=list)
    group_urls: List[str] = Field(default_factory=list)
    accessibility_urls: List[str] = Field(default_factory=list)
    rv_length_urls: List[str] = Field(default_factory=list)
    water_urls: List[str] = Field(default_factory=list)
    amenities_urls: List[str] = Field(default_factory=list)
    reservation_urls: List[str] = Field(default_factory=list)
    summer_urls: List[str] = Field(default_factory=list)

    # Constraint-specific info
    group: Optional[GroupSiteInfo] = None
    accessibility: Optional[AccessibilityInfo] = None
    rv: Optional[RVInfo] = None
    water: Optional[WaterInfo] = None
    amenities: Optional[AmenitiesInfo] = None
    reservation: Optional[ReservationInfo] = None
    summer: Optional[SeasonInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campground() -> str:
    return """
    Extract a single campground solution from the answer that attempts to satisfy all requirements.
    Return the following fields exactly as they appear in the answer (do not invent anything):

    Identification and location:
    - name: The campground name (string).
    - state: The U.S. state where the campground is located (string; may be full name or abbreviation).
    - national_forest: The specific National Forest (string).
    - is_usfs_managed: true/false based on the answer's claim (do not infer beyond the text).
    - is_developed: true/false if the answer claims it is a developed campground (standard amenities, established sites).

    Evidence URLs (must be actual URLs explicitly present in the answer; include markdown link targets if used):
    - id_loc_mgmt_urls: URLs supporting identification/location/USFS management facts.
    - group_urls: URLs supporting group campsite capacity (≥20).
    - accessibility_urls: URLs supporting accessible units and the minimum rule (≥20% or ≥2).
    - rv_length_urls: URLs supporting RV length accommodation (≥35 ft).
    - water_urls: URLs supporting potable water provision/standards.
    - amenities_urls: URLs supporting picnic tables, fire rings/grills, and toilets.
    - reservation_urls: URLs supporting Recreation.gov booking and the 6-month window.
    - summer_urls: URLs supporting summer availability (June–August).

    Constraint-specific info (strings are allowed; use null if not given):
    - group:
        - has_group_site: true/false if the answer claims there is at least one group campsite.
        - min_capacity_number: numeric string if the answer mentions a capacity number (e.g., "25", "50"), else null.
        - capacity_text: any free text about capacity (e.g., "up to 50 people"), else null.
    - accessibility:
        - has_accessible_units: true/false if the answer claims accessible camping units exist.
        - accessible_units_count: numeric string if provided (e.g., "5"), else null.
        - total_units_count: numeric string if provided (e.g., "30"), else null.
        - accessible_units_pct: percent string if provided (e.g., "20%"), else null.
        - rule_claim_text: any explicit text claiming the rule is satisfied, else null.
    - rv:
        - accommodates_35ft_or_more: true/false if the answer claims RVs of 35 ft or longer are accommodated.
        - max_length_claim: any text about RV max length, else null.
    - water:
        - potable_water_claim: any text claiming potable/drinking water availability, else null.
        - meets_federal_standards_claim: true/false if the answer explicitly claims federal clean water standards, else null.
    - amenities:
        - picnic_tables: true/false if the answer claims picnic tables present.
        - fire_rings_or_grills: true/false if the answer claims fire rings or grills present.
        - toilets: true/false if the answer claims toilets present.
    - reservation:
        - bookable_on_recgov: true/false if the answer claims booking on Recreation.gov.
        - uses_standard_6_month_window: true/false if the answer claims the standard 6-month (≈180-day) window.
        - recgov_url: the Recreation.gov URL if present (also include in reservation_urls), else null.
    - summer:
        - open_in_june_july_august: true/false if the answer claims June–August availability.
        - season_text: free text describing season dates, else null.

    IMPORTANT:
    - Only include URLs that are explicitly present in the answer.
    - If any field is not mentioned, return null (or empty list for URL lists).
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_identification_and_location(evaluator: Evaluator, parent_node, ext: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Campground_identification_and_location",
        desc="Campground is properly identified and located per the constraints, with URL support.",
        parent=parent_node,
        critical=True,
    )

    # Existence fields
    name_ok = bool(_normalize_str(ext.name))
    state_ok = bool(_normalize_str(ext.state))
    nf_ok = bool(_normalize_str(ext.national_forest))

    evaluator.add_custom_node(
        result=name_ok,
        id="Campground_name_provided",
        desc="Provides the campground name.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=state_ok,
        id="State_provided",
        desc="Provides the U.S. state where the campground is located.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_western_state(ext.state),
        id="State_is_western_US",
        desc="The stated location is in a western U.S. state.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=nf_ok,
        id="National_Forest_provided",
        desc="Provides the specific National Forest name.",
        parent=node,
        critical=True
    )

    # Within a National Forest
    within_nf_leaf = evaluator.add_leaf(
        id="Within_a_National_Forest",
        desc="Confirms the campground is within a National Forest.",
        parent=node,
        critical=True
    )
    claim_nf = f"The campground '{_normalize_str(ext.name)}' is located within the '{_normalize_str(ext.national_forest)}' National Forest."
    await evaluator.verify(
        claim=claim_nf,
        node=within_nf_leaf,
        sources=ext.id_loc_mgmt_urls,
        additional_instruction="Verify the campground is within the specified National Forest. If no URLs are provided, mark this as not supported."
    )

    # USFS managed
    usfs_leaf = evaluator.add_leaf(
        id="USFS_managed",
        desc="Confirms the campground is managed by the U.S. Forest Service.",
        parent=node,
        critical=True
    )
    claim_usfs = f"The campground '{_normalize_str(ext.name)}' is managed by the U.S. Forest Service (USFS)."
    await evaluator.verify(
        claim=claim_usfs,
        node=usfs_leaf,
        sources=ext.id_loc_mgmt_urls,
        additional_instruction="Confirm that the managing agency is USFS. Accept official USFS domains or Recreation.gov pages indicating USFS management. If no URLs are provided, mark as not supported."
    )

    # Developed campground
    developed_leaf = evaluator.add_leaf(
        id="Developed_campground",
        desc="Confirms the campground is a developed campground (not dispersed/backcountry-only).",
        parent=node,
        critical=True
    )
    claim_dev = f"The campground '{_normalize_str(ext.name)}' is a developed campground with established sites and standard amenities."
    dev_sources = list({*ext.id_loc_mgmt_urls, *ext.amenities_urls})
    await evaluator.verify(
        claim=claim_dev,
        node=developed_leaf,
        sources=dev_sources,
        additional_instruction="Look for indications of established campsites and standard amenities (tables, rings/grills, toilets). If sources only describe dispersed/backcountry camping, mark incorrect. If no URLs are provided, mark not supported."
    )

    # URL supports identification/location/management
    id_support_leaf = evaluator.add_leaf(
        id="URL_supports_identification_location_management",
        desc="Provides at least one supporting URL that verifies identification/location/management facts.",
        parent=node,
        critical=True
    )
    claim_id_support = f"At least one of the provided URL(s) confirms the campground's identification, location in {_normalize_str(ext.state)}, and/or USFS management."
    await evaluator.verify(
        claim=claim_id_support,
        node=id_support_leaf,
        sources=ext.id_loc_mgmt_urls,
        additional_instruction="Pass only if at least one URL explicitly supports identification/location or USFS management. If no URLs are provided, mark as not supported."
    )


async def verify_group_requirement(evaluator: Evaluator, parent_node, ext: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Group_campsite_requirement",
        desc="Group campsite capacity requirement is satisfied and verified by URL(s).",
        parent=parent_node,
        critical=True
    )

    # Has group site 20+
    group_leaf = evaluator.add_leaf(
        id="Has_group_site_20plus",
        desc="Has at least one group campsite designed to accommodate 20 or more people.",
        parent=node,
        critical=True
    )
    capacity_phrase = ext.group.capacity_text if ext.group else None
    capacity_num = ext.group.min_capacity_number if ext.group else None
    base_claim = "The campground offers at least one group campsite that can accommodate 20 or more people."
    if capacity_num:
        base_claim = f"The campground offers at least one group campsite with capacity of {capacity_num} people (≥20)."
    elif capacity_phrase:
        base_claim = f"The campground offers at least one group campsite with capacity '{capacity_phrase}', satisfying ≥20."
    await evaluator.verify(
        claim=base_claim,
        node=group_leaf,
        sources=ext.group_urls,
        additional_instruction="Confirm a group site exists with capacity ≥20 people. If the page states a capacity number, compare it against 20. If no URLs are provided, mark as not supported."
    )

    # URL supports group capacity
    group_support_leaf = evaluator.add_leaf(
        id="URL_supports_group_capacity",
        desc="Provides supporting URL(s) verifying the group campsite capacity (≥20).",
        parent=node,
        critical=True
    )
    claim_group_support = "The provided URL(s) explicitly confirm the group campsite capacity requirement (≥20 people)."
    await evaluator.verify(
        claim=claim_group_support,
        node=group_support_leaf,
        sources=ext.group_urls,
        additional_instruction="Pass only if at least one URL explicitly states or clearly implies group capacity ≥20. If no URLs are provided, mark as not supported."
    )


async def verify_accessibility_requirement(evaluator: Evaluator, parent_node, ext: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Accessibility_requirements",
        desc="ADA accessibility requirements are satisfied and verified by URL(s).",
        parent=parent_node,
        critical=True
    )

    # Has ADA accessible units
    ada_leaf = evaluator.add_leaf(
        id="Has_ADA_accessible_units",
        desc="Has accessible camping units that meet ADA/ABA standards.",
        parent=node,
        critical=True
    )
    claim_ada = "The campground provides accessible camping units that meet ADA/ABA standards."
    await evaluator.verify(
        claim=claim_ada,
        node=ada_leaf,
        sources=ext.accessibility_urls,
        additional_instruction="Accept explicit mentions of ADA/ABA-compliant or accessible campsites. If no URLs are provided, mark as not supported."
    )

    # Meets accessible unit minimum rule
    rule_leaf = evaluator.add_leaf(
        id="Meets_accessible_unit_minimum_rule",
        desc="Meets the rule: if more than 2 total units, at least 20% (and not less than 2) are accessible.",
        parent=node,
        critical=True
    )
    acc = ext.accessibility or AccessibilityInfo()
    acc_count = _normalize_str(acc.accessible_units_count)
    total_count = _normalize_str(acc.total_units_count)
    acc_pct = _normalize_str(acc.accessible_units_pct)
    if acc_count and total_count:
        claim_rule = f"There are {acc_count} accessible units out of {total_count} total units, which satisfies the rule (≥20% accessible and at least 2 accessible units)."
    elif acc_pct:
        claim_rule = f"The campground indicates accessible units at {acc_pct}, which satisfies the rule (≥20% accessible and at least 2 accessible units)."
    else:
        claim_rule = "The campground meets the accessible-unit rule: at least 20% of units (or a minimum of 2 units) are accessible."
    await evaluator.verify(
        claim=claim_rule,
        node=rule_leaf,
        sources=ext.accessibility_urls,
        additional_instruction="If numbers are present, compute whether accessible units ≥ 20% of total and ≥ 2. If only a clear assertion of meeting ADA minimums is present, accept. If no URLs provided, mark as not supported."
    )

    # URL supports accessibility
    acc_support_leaf = evaluator.add_leaf(
        id="URL_supports_accessibility",
        desc="Provides supporting URL(s) verifying accessible-unit presence and the minimum accessible-unit rule.",
        parent=node,
        critical=True
    )
    claim_acc_support = "The provided URL(s) confirm accessible campsites and support the minimum accessible-unit rule."
    await evaluator.verify(
        claim=claim_acc_support,
        node=acc_support_leaf,
        sources=ext.accessibility_urls,
        additional_instruction="Pass only if at least one URL supports accessible units and the minimum rule. If no URLs provided, mark as not supported."
    )


async def verify_rv_length_requirement(evaluator: Evaluator, parent_node, ext: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="RV_length_requirement",
        desc="RV length accommodation requirement is satisfied and verified by URL(s).",
        parent=parent_node,
        critical=True
    )

    rv_leaf = evaluator.add_leaf(
        id="Accommodates_RVs_35ft_or_longer",
        desc="Can accommodate RVs that are 35 feet or longer in length (at least some sites).",
        parent=node,
        critical=True
    )
    max_len_text = ext.rv.max_length_claim if ext.rv else None
    claim_rv = "The campground can accommodate RVs of 35 feet or longer at some sites."
    if max_len_text:
        claim_rv = f"The campground accommodates RVs with maximum length '{max_len_text}', satisfying ≥35 ft."
    await evaluator.verify(
        claim=claim_rv,
        node=rv_leaf,
        sources=ext.rv_length_urls,
        additional_instruction="Check RV site length info. Accept if any site allows ≥35 ft RVs. If no URLs are provided, mark as not supported."
    )

    rv_support_leaf = evaluator.add_leaf(
        id="URL_supports_RV_length",
        desc="Provides supporting URL(s) verifying RV length accommodation (≥35 ft).",
        parent=node,
        critical=True
    )
    claim_rv_support = "The provided URL(s) confirm RV length accommodation of at least 35 feet."
    await evaluator.verify(
        claim=claim_rv_support,
        node=rv_support_leaf,
        sources=ext.rv_length_urls,
        additional_instruction="Pass only if a URL explicitly shows site length or RV length allowance ≥35 ft. If no URLs, mark as not supported."
    )


async def verify_potable_water_requirement(evaluator: Evaluator, parent_node, ext: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Potable_water_requirement",
        desc="Potable water requirement is satisfied and verified by URL(s).",
        parent=parent_node,
        critical=True
    )

    water_leaf = evaluator.add_leaf(
        id="Provides_potable_water_meeting_federal_standards",
        desc="Provides potable water that meets federal clean water standards (as claimed/attested by an authoritative source).",
        parent=node,
        critical=True
    )
    claim_water = "The campground provides potable (drinking) water that meets federal clean water standards."
    await evaluator.verify(
        claim=claim_water,
        node=water_leaf,
        sources=ext.water_urls,
        additional_instruction="Accept official USFS or Recreation.gov pages stating 'drinking water' or 'potable water' as meeting federal standards. If sources explicitly deny standards, mark incorrect. If no URLs provided, mark not supported."
    )

    water_support_leaf = evaluator.add_leaf(
        id="URL_supports_potable_water",
        desc="Provides supporting URL(s) verifying potable water provision/standard claim.",
        parent=node,
        critical=True
    )
    claim_water_support = "At least one provided URL confirms potable/drinking water is available and meets standard expectations."
    await evaluator.verify(
        claim=claim_water_support,
        node=water_support_leaf,
        sources=ext.water_urls,
        additional_instruction="Pass only if at least one URL explicitly indicates potable/drinking water availability. If no URLs provided, mark not supported."
    )


async def verify_amenities_requirement(evaluator: Evaluator, parent_node, ext: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Standard_amenities_requirements",
        desc="Required standard amenities are present and verified by URL(s).",
        parent=parent_node,
        critical=True
    )

    # Picnic tables
    picnic_leaf = evaluator.add_leaf(
        id="Has_picnic_tables",
        desc="Campsites include picnic tables.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Campsites include picnic tables.",
        node=picnic_leaf,
        sources=ext.amenities_urls,
        additional_instruction="Look for amenities listing or description confirming picnic tables. If no URLs provided, mark not supported."
    )

    # Fire rings or grills
    fire_leaf = evaluator.add_leaf(
        id="Has_fire_rings_or_grills",
        desc="Campsites include fire rings or fire grills.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Campsites include fire rings or fire grills.",
        node=fire_leaf,
        sources=ext.amenities_urls,
        additional_instruction="Look for amenities listing confirming fire rings or grills. If no URLs provided, mark not supported."
    )

    # Toilet facilities
    toilet_leaf = evaluator.add_leaf(
        id="Has_toilet_facilities",
        desc="Has toilet facilities (vault toilets or flush toilets).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The campground provides toilet facilities (vault or flush toilets).",
        node=toilet_leaf,
        sources=ext.amenities_urls,
        additional_instruction="Look for amenities listing confirming toilets (vault or flush). If no URLs provided, mark not supported."
    )

    # URL supports amenities (all)
    amenities_support_leaf = evaluator.add_leaf(
        id="URL_supports_amenities",
        desc="Provides supporting URL(s) verifying picnic tables, fire rings/grills, and toilets.",
        parent=node,
        critical=True
    )
    claim_amenities_support = "The provided URL(s) confirm the presence of picnic tables, fire rings/grills, and toilet facilities."
    await evaluator.verify(
        claim=claim_amenities_support,
        node=amenities_support_leaf,
        sources=ext.amenities_urls,
        additional_instruction="Prefer a single authoritative listing (e.g., Recreation.gov) showing all amenities. If no URLs are provided, mark not supported."
    )


async def verify_reservation_requirement(evaluator: Evaluator, parent_node, ext: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Reservation_requirements",
        desc="Reservation platform and advance window requirements are satisfied and verified by URL(s).",
        parent=parent_node,
        critical=True
    )

    # Bookable on Rec.gov
    book_leaf = evaluator.add_leaf(
        id="Bookable_on_Recreation_gov",
        desc="Is bookable through Recreation.gov.",
        parent=node,
        critical=True
    )
    claim_book = "The campground can be booked via Recreation.gov."
    await evaluator.verify(
        claim=claim_book,
        node=book_leaf,
        sources=ext.reservation_urls,
        additional_instruction="Confirm that the official booking platform is Recreation.gov. The presence of a Recreation.gov listing suffices. If no URLs provided, mark not supported."
    )

    # Uses standard 6-month window
    window_leaf = evaluator.add_leaf(
        id="Uses_standard_6_month_window",
        desc="Uses the standard 6-month (180-day) advance reservation window.",
        parent=node,
        critical=True
    )
    claim_window = "Recreation.gov uses a standard 6-month (≈180-day) rolling advance reservation window for this campground."
    await evaluator.verify(
        claim=claim_window,
        node=window_leaf,
        sources=ext.reservation_urls,
        additional_instruction="On the Recreation.gov page, look for booking window info (e.g., 'Reservations can be made 6 months in advance' or 'Release window 6 months'). Accept equivalent phrasings indicating 6 months. If no URLs provided, mark not supported."
    )

    # URL supports reservations
    res_support_leaf = evaluator.add_leaf(
        id="URL_supports_reservations",
        desc="Provides supporting URL(s) verifying Recreation.gov booking and the 6-month window.",
        parent=node,
        critical=True
    )
    claim_res_support = "At least one provided URL confirms Recreation.gov booking and the 6-month advance window."
    await evaluator.verify(
        claim=claim_res_support,
        node=res_support_leaf,
        sources=ext.reservation_urls,
        additional_instruction="Prefer the Recreation.gov listing showing both booking platform and window. If no URLs provided, mark not supported."
    )


async def verify_summer_availability_requirement(evaluator: Evaluator, parent_node, ext: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Summer_availability_requirement",
        desc="Summer availability requirement is satisfied and verified by URL(s).",
        parent=parent_node,
        critical=True
    )

    open_leaf = evaluator.add_leaf(
        id="Open_June_through_August",
        desc="Is open and available for camping during June through August.",
        parent=node,
        critical=True
    )
    season_text = ext.summer.season_text if ext.summer else None
    claim_open = "The campground is open and available for camping during June, July, and August."
    if season_text:
        claim_open = f"The campground's operating season '{season_text}' includes June through August."
    await evaluator.verify(
        claim=claim_open,
        node=open_leaf,
        sources=ext.summer_urls,
        additional_instruction="Check operating season/peak season dates. Accept if the page indicates availability in June, July, and August. If no URLs provided, mark not supported."
    )

    summer_support_leaf = evaluator.add_leaf(
        id="URL_supports_summer_availability",
        desc="Provides supporting URL(s) verifying summer-season availability (June–August).",
        parent=node,
        critical=True
    )
    claim_summer_support = "At least one provided URL confirms the campground is open during June–August."
    await evaluator.verify(
        claim=claim_summer_support,
        node=summer_support_leaf,
        sources=ext.summer_urls,
        additional_instruction="Pass only if at least one URL shows operating dates including June–August. If no URLs provided, mark not supported."
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
    Evaluate an answer for the USFS campground constraint task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # categories evaluated independently
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

    # Create the top-level critical solution node
    solution_node = evaluator.add_parallel(
        id="Valid_campground_solution",
        desc="Solution identifies exactly one developed USFS-managed campground in a National Forest in a western U.S. state, and verifies every stated constraint with supporting URLs.",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    ext: CampgroundExtraction = await evaluator.extract(
        prompt=prompt_extract_campground(),
        template_class=CampgroundExtraction,
        extraction_name="campground_extraction",
    )

    # Build verification subtrees for each rubric category
    await verify_identification_and_location(evaluator, solution_node, ext)
    await verify_group_requirement(evaluator, solution_node, ext)
    await verify_accessibility_requirement(evaluator, solution_node, ext)
    await verify_rv_length_requirement(evaluator, solution_node, ext)
    await verify_potable_water_requirement(evaluator, solution_node, ext)
    await verify_amenities_requirement(evaluator, solution_node, ext)
    await verify_reservation_requirement(evaluator, solution_node, ext)
    await verify_summer_availability_requirement(evaluator, solution_node, ext)

    # Return evaluation summary
    return evaluator.get_summary()