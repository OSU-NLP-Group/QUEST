import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "weekend_montauk_getaway_public_transit_budget"
TASK_DESCRIPTION = """
Plan a budget-conscious weekend getaway from New York City to Montauk, Long Island, for one person using public transportation only (no car). Your plan should include: (1) Round-trip transportation via the Long Island Rail Road (LIRR) from Penn Station to Montauk, including approximate travel time; (2) One night of beachfront accommodation at a hotel or resort in Montauk that offers direct beach access (for Saturday night); (3) A visit to Montauk Point Lighthouse, including the standard adult admission fee; (4) At least one additional beach or outdoor location to visit in Montauk (other than the lighthouse); (5) A budget breakdown that estimates costs for: LIRR round-trip, accommodation, lighthouse admission, and NYC subway/bus usage to/from Penn Station (mention the current OMNY fare structure). For each component, provide specific details (property names, locations, fees) and include reference URLs from your research to support your recommendations.
"""

# Reference values as of Jan 2026 per rubric
NYC_BASE_FARE_TEXT = "$3.00"
OMNY_WEEKLY_CAP_TEXT = "$35 for 12 rides within 7 days; rides after the 12th are free."
LIGHTHOUSE_ADULT_FEE_TEXT = "$15"
EXPECTED_LIRR_ONE_WAY_RANGE_TEXT = "about 2.5–3 hours"
EXPECTED_MONTAUK_ZONE = "Zone 14"
TYPICAL_WEEKEND_NIGHT_RANGE_TEXT = "$229–$463+"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LIRRInfo(BaseModel):
    mentions_roundtrip: Optional[bool] = None
    one_way_travel_time_text: Optional[str] = None
    mentions_zone14: Optional[bool] = None
    reference_urls: List[str] = Field(default_factory=list)


class AccommodationInfo(BaseModel):
    name: Optional[str] = None
    saturday_night: Optional[bool] = None
    direct_beach_access: Optional[bool] = None
    location_details: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class LighthouseInfo(BaseModel):
    included: Optional[bool] = None
    location_details: Optional[str] = None
    adult_admission_fee_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class AdditionalLocationInfo(BaseModel):
    name: Optional[str] = None
    location_details: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class NYCTransitInfo(BaseModel):
    omny_mentioned: Optional[bool] = None
    base_fare_text: Optional[str] = None
    weekly_cap_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class BudgetInfo(BaseModel):
    lirr_roundtrip_cost_text: Optional[str] = None
    accommodation_cost_text: Optional[str] = None
    lighthouse_cost_included: Optional[bool] = None
    nyc_transit_cost_text: Optional[str] = None
    cost_saving_tactics: List[str] = Field(default_factory=list)


class PlanExtraction(BaseModel):
    public_transport_only: Optional[bool] = None
    lirr: Optional[LIRRInfo] = None
    accommodation: Optional[AccommodationInfo] = None
    lighthouse: Optional[LighthouseInfo] = None
    additional_location: Optional[AdditionalLocationInfo] = None
    nyc_transit: Optional[NYCTransitInfo] = None
    budget: Optional[BudgetInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
Extract the following structured information from the answer. Do not invent data.

1) public_transport_only: true/false — whether the plan explicitly commits to public transportation only (no car).
2) lirr:
   - mentions_roundtrip: true/false — explicitly states round-trip via LIRR Penn Station ↔ Montauk.
   - one_way_travel_time_text: the stated approximate one-way LIRR travel time (string; e.g., "about 2 hours 45 minutes", "≈3 hours").
   - mentions_zone14: true/false — explicitly notes Montauk is in LIRR Zone 14 (or says it's the farthest zone consistent with Zone 14).
   - reference_urls: list of LIRR-related URLs cited (routes/schedules/fares/zones). Only include actual URLs mentioned in the answer.
3) accommodation:
   - name: property name (hotel/resort) in Montauk.
   - saturday_night: true/false — explicitly states the stay is for Saturday night.
   - direct_beach_access: true/false — explicitly states direct beach access.
   - location_details: address or clear area/neighborhood details (string).
   - reference_urls: list of URLs referenced for the accommodation (official site, booking, etc.).
4) lighthouse:
   - included: true/false — visit to Montauk Point Lighthouse is included.
   - location_details: address/siting at Montauk Point State Park (string).
   - adult_admission_fee_text: the stated standard adult admission fee (string, e.g., "$15").
   - reference_urls: list of URLs referenced for the lighthouse.
5) additional_location:
   - name: the additional beach/outdoor location (not the lighthouse).
   - location_details: address/area/park name (string).
   - reference_urls: list of URLs referenced for this location.
6) nyc_transit:
   - omny_mentioned: true/false — mentions OMNY payment/fare system.
   - base_fare_text: the stated NYC subway/bus base fare (string, e.g., "$3.00").
   - weekly_cap_text: the stated OMNY weekly cap text (string, e.g., "$35 cap for 12 rides in 7 days; rides after the 12th are free").
   - reference_urls: list of URLs for OMNY/MTA fare info.
7) budget:
   - lirr_roundtrip_cost_text: the plan’s LIRR round-trip cost estimate (string).
   - accommodation_cost_text: the plan’s one-night (Saturday) accommodation cost estimate (string).
   - lighthouse_cost_included: true/false — includes lighthouse admission in budget.
   - nyc_transit_cost_text: subway/bus estimate to/from Penn Station (string; may mention base fare or weekly cap).
   - cost_saving_tactics: list of concrete cost-saving tactics mentioned (e.g., off-peak fares, early booking).

Important:
- Extract only information explicitly present in the answer.
- For all "reference_urls" fields, include only actual URLs present in the answer (plain URLs or markdown links).
- If something is missing, return null (for single value) or [] (for lists).
"""


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if isinstance(u, str):
            s = u.strip()
            if s and ("http://" in s or "https://" in s):
                out.append(s)
    return out


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return len(_valid_urls(urls)) > 0


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_public_transportation_only(evaluator: Evaluator, root) -> None:
    node = evaluator.add_leaf(
        id="Public_Transportation_Only",
        desc="Plan adheres to the constraint of public transportation only (no car travel required/suggested as part of the plan).",
        parent=root,
        critical=True,
    )
    claim = "The plan strictly uses public transportation only (e.g., LIRR and NYC subways/buses) and does not require or suggest any car travel."
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Search the answer for any mention of driving, rental car, rideshare required, or parking. If any car usage is required or suggested as part of the plan, mark this incorrect."
    )


async def build_lirr_transportation(evaluator: Evaluator, root, extracted: PlanExtraction) -> None:
    parent = evaluator.add_parallel(
        id="LIRR_Transportation",
        desc="Round-trip transportation via LIRR from Penn Station to Montauk with approximate travel time and references.",
        parent=root,
        critical=True
    )
    # Route round-trip mention
    leaf_route = evaluator.add_leaf(
        id="Route_RoundTrip_Penn_To_Montauk",
        desc="States round-trip travel via LIRR between New York Penn Station and Montauk.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan explicitly states round-trip travel via the Long Island Rail Road (LIRR) between New York Penn Station and Montauk.",
        node=leaf_route,
        additional_instruction="Look for 'round-trip' or equivalent phrasing and both endpoints 'Penn Station' and 'Montauk'."
    )

    # Approx one-way time
    leaf_time = evaluator.add_leaf(
        id="Approximate_One_Way_Travel_Time",
        desc="Provides approximate one-way travel time consistent with ~2.5–3 hours.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan provides an approximate one-way LIRR travel time between about 2.5 and 3 hours.",
        node=leaf_time,
        additional_instruction="Accept variants like 'about 2 hr 45 min', '≈3 hours', or 'roughly 2.5–3 hrs'."
    )

    # Zone 14 mention
    leaf_zone = evaluator.add_leaf(
        id="Montauk_LIRR_Zone",
        desc="Notes that Montauk is in LIRR fare Zone 14 (or otherwise clearly indicates the farthest/furthest zone consistent with Zone 14).",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan notes that Montauk is in LIRR fare Zone 14 (or clearly indicates the farthest zone consistent with Zone 14).",
        node=leaf_zone,
        additional_instruction="Look for explicit 'Zone 14'. If not found, accept clear wording that it is the farthest/furthest zone consistent with Zone 14."
    )

    # LIRR reference URL(s) existence
    lirr_urls = _valid_urls(getattr(extracted.lirr, "reference_urls", []) if extracted and extracted.lirr else [])
    evaluator.add_custom_node(
        result=_has_any_url(lirr_urls),
        id="LIRR_Reference_URL",
        desc="Provides at least one valid reference URL supporting the LIRR route/schedule (and/or zone/fare context) information.",
        parent=parent,
        critical=True
    )


async def build_accommodation(evaluator: Evaluator, root, extracted: PlanExtraction) -> None:
    parent = evaluator.add_parallel(
        id="Beachfront_Accommodation",
        desc="One night (Saturday) beachfront accommodation in Montauk with direct beach access and references.",
        parent=root,
        critical=True
    )

    # Beachfront Saturday night stay explicitly with direct beach access
    acc_name = getattr(extracted.accommodation, "name", None) if extracted and extracted.accommodation else None
    acc_claim_name = f" '{acc_name}'" if acc_name else ""
    leaf_stay = evaluator.add_leaf(
        id="Beachfront_Saturday_Night_Stay",
        desc="Identifies a specific Montauk hotel/resort for Saturday night and explicitly indicates it offers direct beach access.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The plan identifies a specific Montauk hotel or resort{acc_claim_name} for Saturday night and explicitly indicates it offers direct beach access.",
        node=leaf_stay,
        additional_instruction="Accept synonyms like 'on the beach', 'beachfront with private access', 'direct access to the beach'. It must clearly be for Saturday night."
    )

    # Accommodation location details
    leaf_loc = evaluator.add_leaf(
        id="Accommodation_Location_Details",
        desc="Provides location details for the accommodation (e.g., address or clear neighborhood/area in Montauk).",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan provides location details for the chosen accommodation (address, street, or clear neighborhood/area in Montauk).",
        node=leaf_loc,
        additional_instruction="Look for a street address or an identifiable area/neighborhood in Montauk."
    )

    # Accommodation reference URL(s) existence
    acc_urls = _valid_urls(getattr(extracted.accommodation, "reference_urls", []) if extracted and extracted.accommodation else [])
    evaluator.add_custom_node(
        result=_has_any_url(acc_urls),
        id="Accommodation_Reference_URL",
        desc="Provides a valid reference URL for the chosen accommodation property.",
        parent=parent,
        critical=True
    )


async def build_lighthouse(evaluator: Evaluator, root, extracted: PlanExtraction) -> None:
    parent = evaluator.add_parallel(
        id="Montauk_Point_Lighthouse",
        desc="Lighthouse visit included with standard adult admission fee and references.",
        parent=root,
        critical=True
    )

    # Visit included
    leaf_included = evaluator.add_leaf(
        id="Lighthouse_Visit_Included",
        desc="Includes a visit to Montauk Point Lighthouse in the itinerary/plan.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes a visit to Montauk Point Lighthouse.",
        node=leaf_included
    )

    # Location details
    leaf_loc = evaluator.add_leaf(
        id="Lighthouse_Location_Details",
        desc="Provides location details for Montauk Point Lighthouse (e.g., address or clear siting at Montauk Point).",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan provides location details for Montauk Point Lighthouse, such as its address or that it is located at Montauk Point/within Montauk Point State Park.",
        node=leaf_loc,
        additional_instruction="Accept any clear siting like 'at the tip of Long Island', 'Montauk Point', or 'Montauk Point State Park'."
    )

    # Standard adult admission fee stated as $15
    leaf_fee = evaluator.add_leaf(
        id="Standard_Adult_Admission_Fee",
        desc="States the standard adult admission fee as $15.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states the standard adult admission fee for Montauk Point Lighthouse is $15.",
        node=leaf_fee,
        additional_instruction="Look for an explicit $15 figure tied to the adult admission for the lighthouse."
    )

    # Lighthouse reference URL(s) existence
    lh_urls = _valid_urls(getattr(extracted.lighthouse, "reference_urls", []) if extracted and extracted.lighthouse else [])
    evaluator.add_custom_node(
        result=_has_any_url(lh_urls),
        id="Lighthouse_Reference_URL",
        desc="Provides a valid reference URL supporting the lighthouse visit/admission information.",
        parent=parent,
        critical=True
    )


async def build_additional_location(evaluator: Evaluator, root, extracted: PlanExtraction) -> None:
    parent = evaluator.add_parallel(
        id="Additional_Outdoor_Location",
        desc="At least one additional beach/outdoor location in Montauk (not the lighthouse) with references.",
        parent=root,
        critical=True
    )

    # Qualifies
    addl_name = getattr(extracted.additional_location, "name", None) if extracted and extracted.additional_location else None
    name_snippet = f" '{addl_name}'" if addl_name else ""
    leaf_qual = evaluator.add_leaf(
        id="Additional_Location_Qualifies",
        desc="Names at least one additional beach or outdoor location in Montauk other than Montauk Point Lighthouse.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The plan names at least one additional beach or outdoor location{name_snippet} in Montauk other than Montauk Point Lighthouse.",
        node=leaf_qual,
        additional_instruction="Examples include state parks, beaches, trails, or outdoor overlooks in Montauk (excluding the lighthouse)."
    )

    # Location details
    leaf_loc = evaluator.add_leaf(
        id="Additional_Location_Location_Details",
        desc="Provides location details for the additional location (e.g., address/area/park name sufficient to find it).",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan provides location details for the additional outdoor location (address, area, or park name) sufficient to find it.",
        node=leaf_loc
    )

    # Reference URL(s) existence
    addl_urls = _valid_urls(getattr(extracted.additional_location, "reference_urls", []) if extracted and extracted.additional_location else [])
    evaluator.add_custom_node(
        result=_has_any_url(addl_urls),
        id="Additional_Location_Reference_URL",
        desc="Provides a valid reference URL for the additional location.",
        parent=parent,
        critical=True
    )


async def build_nyc_transit_omny(evaluator: Evaluator, root, extracted: PlanExtraction) -> None:
    parent = evaluator.add_parallel(
        id="NYC_Transit_OMNY_Context",
        desc="NYC subway/bus context to/from Penn Station including OMNY fare structure and references.",
        parent=root,
        critical=True
    )

    # OMNY mentioned
    leaf_omny = evaluator.add_leaf(
        id="OMNY_Mentioned",
        desc="Mentions OMNY as the NYC subway/bus payment/fare system.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan mentions OMNY as the NYC subway/bus payment/fare system.",
        node=leaf_omny,
        additional_instruction="Accept if OMNY is referenced as tap-and-go/contactless fare payment used on subways/buses."
    )

    # NYC Base fare $3.00
    leaf_base = evaluator.add_leaf(
        id="NYC_Base_Fare",
        desc="States NYC subway/bus base fare as $3.00 per ride (as of January 2026).",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states the NYC subway/bus base fare is $3.00 per ride.",
        node=leaf_base,
        additional_instruction="Look for an explicit $3.00 per ride figure."
    )

    # OMNY weekly cap statement
    leaf_cap = evaluator.add_leaf(
        id="OMNY_Weekly_Cap",
        desc="States OMNY weekly cap is $35 for 12 rides within a 7-day period; rides after the 12th are free.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states the OMNY weekly cap is $35 for 12 rides within a 7-day period and that rides after the 12th are free.",
        node=leaf_cap,
        additional_instruction="Accept minor word variants as long as the amounts and logic match."
    )

    # OMNY Reference URL(s) existence
    omny_urls = _valid_urls(getattr(extracted.nyc_transit, "reference_urls", []) if extracted and extracted.nyc_transit else [])
    evaluator.add_custom_node(
        result=_has_any_url(omny_urls),
        id="OMNY_Reference_URL",
        desc="Provides a valid reference URL supporting OMNY fare structure information.",
        parent=parent,
        critical=True
    )


async def build_budget_breakdown(evaluator: Evaluator, root, extracted: PlanExtraction) -> None:
    parent = evaluator.add_parallel(
        id="Budget_Breakdown",
        desc="Required budget breakdown estimating costs for the specified components.",
        parent=root,
        critical=True
    )

    # LIRR round-trip cost estimate
    leaf_lirr_cost = evaluator.add_leaf(
        id="Budget_LIRR_RoundTrip_Cost",
        desc="Estimates the LIRR round-trip cost (Penn Station ↔ Montauk).",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan provides an estimated cost for the LIRR round-trip between New York Penn Station and Montauk.",
        node=leaf_lirr_cost,
        additional_instruction="Any clear dollar estimate or range for the full round-trip qualifies."
    )

    # Accommodation cost estimate consistent with typical weekend-night range
    leaf_acc_cost = evaluator.add_leaf(
        id="Budget_Accommodation_Cost",
        desc="Estimates the one-night (Saturday) accommodation cost and is consistent with/mentions the stated typical weekend-night range of ~$229–$463+.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan estimates the one-night (Saturday) accommodation cost and either explicitly mentions a typical weekend-night range around $229–$463+ or gives a figure that reasonably falls within that range.",
        node=leaf_acc_cost,
        additional_instruction="If a single value is provided, judge whether it reasonably falls within or near the stated range."
    )

    # Lighthouse admission in budget
    leaf_lh_budget = evaluator.add_leaf(
        id="Budget_Lighthouse_Admission_Cost",
        desc="Includes the lighthouse adult admission cost in the budget breakdown.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan's budget includes the Montauk Point Lighthouse adult admission cost.",
        node=leaf_lh_budget,
        additional_instruction="Accept if the budget clearly lists this line item."
    )

    # NYC transit to/from Penn Station cost and relation to OMNY usage
    leaf_nyc_budget = evaluator.add_leaf(
        id="Budget_NYC_Transit_ToFrom_Penn",
        desc="Estimates NYC subway/bus costs to/from Penn Station (and relates it to OMNY usage).",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The plan estimates NYC subway/bus costs to/from Penn Station and relates them to OMNY usage (per-ride fare or weekly cap).",
        node=leaf_nyc_budget,
        additional_instruction="Look for either a per-ride multiplication or mention of the OMNY weekly cap."
    )


async def build_budget_consciousness(evaluator: Evaluator, root) -> None:
    node = evaluator.add_leaf(
        id="Budget_Consciousness",
        desc="Explicitly includes at least one concrete cost-saving tactic consistent with a budget-conscious trip (e.g., off-peak fares, booking timing, avoiding optional add-ons).",
        parent=root,
        critical=False
    )
    await evaluator.verify(
        claim="The plan includes at least one concrete cost-saving tactic (e.g., taking off-peak LIRR trains, booking early, traveling shoulder season, packing snacks, avoiding add-ons).",
        node=node
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
    # Initialize evaluator (root is non-critical to allow a non-critical child while still gating via critical children)
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

    # Extract structured plan info
    extracted: PlanExtraction = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction",
    )

    # Add contextual ground-truth info (for transparency only)
    evaluator.add_ground_truth({
        "expected_constants": {
            "nyc_base_fare_text": NYC_BASE_FARE_TEXT,
            "omny_weekly_cap_text": OMNY_WEEKLY_CAP_TEXT,
            "lighthouse_adult_fee_text": LIGHTHOUSE_ADULT_FEE_TEXT,
            "expected_lirr_one_way_range": EXPECTED_LIRR_ONE_WAY_RANGE_TEXT,
            "expected_montauk_zone": EXPECTED_MONTAUK_ZONE,
            "typical_weekend_night_range": TYPICAL_WEEKEND_NIGHT_RANGE_TEXT
        }
    })

    # Build verification tree according to rubric
    await build_public_transportation_only(evaluator, root)
    await build_lirr_transportation(evaluator, root, extracted)
    await build_accommodation(evaluator, root, extracted)
    await build_lighthouse(evaluator, root, extracted)
    await build_additional_location(evaluator, root, extracted)
    await build_nyc_transit_omny(evaluator, root, extracted)
    await build_budget_breakdown(evaluator, root, extracted)
    await build_budget_consciousness(evaluator, root)

    # Return evaluation summary
    return evaluator.get_summary()