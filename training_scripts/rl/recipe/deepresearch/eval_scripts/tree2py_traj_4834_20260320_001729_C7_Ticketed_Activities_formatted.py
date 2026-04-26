import asyncio
import logging
from typing import Any, List, Optional, Dict
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "oceans_calling_2026_info"
TASK_DESCRIPTION = (
    "Provide comprehensive information about the Oceans Calling 2026 music festival. Your response must include: "
    "(1) the specific dates the festival will take place, "
    "(2) the exact venue location including city and state, "
    "(3) the total number of days the festival runs, and "
    "(4) a description of the venue type and setting. Additionally, include as much relevant information as available about: "
    "ticket purchasing options, expected attendance capacity, age restriction policies, parking and transportation arrangements, "
    "ADA accessibility accommodations, stage configuration details, food vendor availability, merchandise sales areas, "
    "the official festival website URL, ticket refund/exchange policies, and security screening procedures. Provide reference URLs for all information."
)

# Expected ground-truth target phrasing for required items
EXPECTED_DATES_TEXT = "September 25–27, 2026"
EXPECTED_LOCATION_TEXT = "Ocean City Boardwalk, Ocean City, Maryland"
EXPECTED_DURATION_TEXT = "3 days (Friday through Sunday)"
EXPECTED_VENUE_DESC = "outdoor festival with multiple stages along the boardwalk and beach"
EXPECTED_OFFICIAL_SITE = "https://www.oceanscallingfestival.com/"


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class AdditionalCategoryItem(BaseModel):
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    not_available: Optional[bool] = None


class RequiredCoreExtraction(BaseModel):
    dates_text: Optional[str] = None
    dates_urls: List[str] = Field(default_factory=list)

    location_text: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    duration_text: Optional[str] = None
    duration_urls: List[str] = Field(default_factory=list)

    venue_type_desc: Optional[str] = None
    venue_type_urls: List[str] = Field(default_factory=list)

    official_website_url: Optional[str] = None


class OceansCallingExtraction(BaseModel):
    required: Optional[RequiredCoreExtraction] = None

    ticket_purchasing: Optional[AdditionalCategoryItem] = None
    expected_capacity: Optional[AdditionalCategoryItem] = None
    age_policy: Optional[AdditionalCategoryItem] = None
    parking_transport: Optional[AdditionalCategoryItem] = None
    ada_accommodations: Optional[AdditionalCategoryItem] = None
    food_vendors: Optional[AdditionalCategoryItem] = None
    merchandise_areas: Optional[AdditionalCategoryItem] = None
    refund_exchange: Optional[AdditionalCategoryItem] = None
    security_screening: Optional[AdditionalCategoryItem] = None

    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_oc2026() -> str:
    return """
Extract structured information from the answer about Oceans Calling 2026. Follow these rules strictly:
- Extract ONLY what is explicitly present in the answer.
- If a required field is not present in the answer, set it to null (or empty list for URL arrays).
- For each category, collect the exact URLs explicitly cited for that category (in plain text or markdown). Do not invent URLs.

Return a JSON object with this schema:

{
  "required": {
    "dates_text": string | null,                 // e.g., "September 25–27, 2026"
    "dates_urls": string[],

    "location_text": string | null,              // e.g., "Ocean City Boardwalk, Ocean City, Maryland"
    "location_urls": string[],

    "duration_text": string | null,              // e.g., "3 days (Friday through Sunday)"
    "duration_urls": string[],

    "venue_type_desc": string | null,            // e.g., "outdoor festival with multiple stages along the boardwalk and beach"
    "venue_type_urls": string[],

    "official_website_url": string | null        // the official website URL as shown in the answer (exactly)
  },

  "ticket_purchasing": {
    "description": string | null,
    "urls": string[],
    "not_available": boolean | null              // true only if the answer explicitly says info is not available/not found
  },
  "expected_capacity": {
    "description": string | null,
    "urls": string[],
    "not_available": boolean | null
  },
  "age_policy": {
    "description": string | null,
    "urls": string[],
    "not_available": boolean | null
  },
  "parking_transport": {
    "description": string | null,
    "urls": string[],
    "not_available": boolean | null
  },
  "ada_accommodations": {
    "description": string | null,
    "urls": string[],
    "not_available": boolean | null
  },
  "food_vendors": {
    "description": string | null,
    "urls": string[],
    "not_available": boolean | null
  },
  "merchandise_areas": {
    "description": string | null,
    "urls": string[],
    "not_available": boolean | null
  },
  "refund_exchange": {
    "description": string | null,
    "urls": string[],
    "not_available": boolean | null
  },
  "security_screening": {
    "description": string | null,
    "urls": string[],
    "not_available": boolean | null
  },

  "all_urls": string[]                           // all URLs cited anywhere in the answer, de-duplicated
}

Special URL rules:
- Only extract URLs that are explicitly present in the answer (including markdown links).
- Ensure URLs are valid. If a URL lacks http/https, prepend http://.
- Do not infer or fabricate URLs.

For "not_available": set to true only if the answer explicitly uses phrasing like "not available", "TBD", "not found in reliable sources", "no official info yet", or similar.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def coalesce_urls(*url_lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        if lst:
            for u in lst:
                if u and isinstance(u, str):
                    if u not in merged:
                        merged.append(u)
    return merged


def is_nonempty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_core_verifications(
    evaluator: Evaluator,
    parent_node,
    required: Optional[RequiredCoreExtraction],
) -> None:
    """
    Build the 'Required_Core_Details' subtree (critical).
    """
    core_node = evaluator.add_parallel(
        id="Required_Core_Details",
        desc="All required core festival details are present and match the constraints.",
        parent=parent_node,
        critical=True
    )

    # Defensive defaults
    if not required:
        required = RequiredCoreExtraction()

    # 1) Event_Dates_Exact
    node_dates = evaluator.add_leaf(
        id="Event_Dates_Exact",
        desc="Festival dates are stated exactly as September 25–27, 2026.",
        parent=core_node,
        critical=True
    )
    claim_dates = (
        "The Oceans Calling 2026 festival takes place on September 25–27, 2026 "
        "(any dash style between 25 and 27 is acceptable; treat 'Sept' vs 'September' as equivalent)."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=node_dates,
        sources=required.dates_urls,
        additional_instruction="Verify that at least one cited source explicitly supports that the 2026 event occurs from September 25 through September 27, 2026."
    )

    # 2) Event_Location_Exact
    node_loc = evaluator.add_leaf(
        id="Event_Location_Exact",
        desc="Venue location is stated exactly as Ocean City Boardwalk, Ocean City, Maryland (city and state included).",
        parent=core_node,
        critical=True
    )
    claim_loc = (
        "The venue location is the Ocean City Boardwalk in Ocean City, Maryland (explicitly including the city and state)."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=node_loc,
        sources=required.location_urls,
        additional_instruction="Confirm that the cited source(s) explicitly tie the event to the Ocean City Boardwalk and include 'Ocean City, Maryland'."
    )

    # 3) Event_Duration
    node_dur = evaluator.add_leaf(
        id="Event_Duration",
        desc="Total festival duration is stated as 3 days (Friday through Sunday).",
        parent=core_node,
        critical=True
    )
    duration_sources = required.duration_urls if required.duration_urls else required.dates_urls
    claim_dur = "The festival runs for 3 days (Friday through Sunday)."
    await evaluator.verify(
        claim=claim_dur,
        node=node_dur,
        sources=duration_sources,
        additional_instruction="If the source shows the dates spanning Friday to Sunday, that implies 3 days; minor phrasing differences are acceptable."
    )

    # 4) Venue_Type_And_Setting
    node_venue = evaluator.add_leaf(
        id="Venue_Type_And_Setting",
        desc="Venue type/setting is described as an outdoor festival with multiple stages along the boardwalk and beach.",
        parent=core_node,
        critical=True
    )
    claim_venue = (
        "Oceans Calling is an outdoor festival with multiple stages situated along the Ocean City boardwalk and beach area."
    )
    await evaluator.verify(
        claim=claim_venue,
        node=node_venue,
        sources=required.venue_type_urls,
        additional_instruction="Support can include explicit mention of multiple stages and outdoor beachfront/boardwalk setting."
    )

    # 5) Official_Website_URL_Exact (answer-format check; simple verification)
    node_site = evaluator.add_leaf(
        id="Official_Website_URL_Exact",
        desc="Official festival website URL is provided exactly as https://www.oceanscallingfestival.com/.",
        parent=core_node,
        critical=True
    )
    claim_site = (
        "The answer provides the official festival website URL exactly as 'https://www.oceanscallingfestival.com/'."
    )
    await evaluator.verify(
        claim=claim_site,
        node=node_site,
        additional_instruction=(
            "Check the answer text (not external pages). It must include the URL exactly with protocol and trailing slash. "
            "Case sensitivity for the scheme/host doesn't matter; path must be '/'."
        )
    )


async def add_citations_guard_leaf(evaluator: Evaluator, parent_node) -> None:
    """
    Critical leaf to ensure the answer includes citations for its factual claims.
    This is a meta check over the answer, so we use simple verification.
    """
    node = evaluator.add_leaf(
        id="Citations_For_All_Factual_Claims",
        desc="All factual claims included in the response are accompanied by at least one verifiable reference URL (no uncited factual assertions).",
        parent=parent_node,
        critical=True
    )
    claim = (
        "Every factual claim in the provided answer is accompanied by at least one reference URL; "
        "there are no uncited factual assertions. Statements explicitly marked 'not available' or 'not found in reliable sources' "
        "are not considered factual claims and are exempt."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction=(
            "Examine the answer text: ensure that each concrete factual assertion (dates, location, duration, venue type, and any additional details) "
            "has an accompanying URL somewhere in the answer or an explicit sources section. "
            "General connective text does not require citation."
        )
    )


async def handle_additional_category(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    item: Optional[AdditionalCategoryItem],
    label_for_claim: str,
    require_official: bool = False
) -> None:
    """
    Build a single leaf for an additional (non-critical) category with the following logic:
      - If item.not_available is True (explicitly stated), verify via simple check over the answer text.
      - Else if item.description present AND there is at least one URL:
          - If require_official is True, verify that at least one URL is an official channel and that sources support the described info.
            Use multi-URL verification and instruct that official domains like oceanscallingfestival.com or frontgatetickets.com qualify.
          - Else, verify that sources support the described info.
      - Else (no URLs and not explicitly unavailable): mark as failed via a custom node.
    """
    # Normalize
    if item is None:
        item = AdditionalCategoryItem()

    # Case 1: explicitly not available
    if item.not_available:
        node = evaluator.add_leaf(
            id=node_id,
            desc=f"{label_for_claim} explicitly stated as not available/not found in reliable sources.",
            parent=parent_node,
            critical=False
        )
        claim = f"The answer explicitly states that {label_for_claim} information is not available or not found in reliable sources."
        await evaluator.verify(
            claim=claim,
            node=node,
            additional_instruction="Look for explicit phrasing like 'not available', 'TBD', 'not found in reliable sources', or similar in the answer text."
        )
        return

    # Case 2: have description and at least one URL
    if is_nonempty(item.description) and item.urls:
        node = evaluator.add_leaf(
            id=node_id,
            desc=f"{label_for_claim} addressed with citations.",
            parent=parent_node,
            critical=False
        )
        if require_official:
            claim = (
                f"At least one of the cited sources is an official festival channel (official site or official ticketing partner) "
                f"and the sources support the described {label_for_claim} for Oceans Calling 2026."
            )
            add_ins = (
                "Treat these as official channels if applicable: oceanscallingfestival.com (official), "
                "frontgatetickets.com (official ticketing partner for many festivals), axs.com, ticketmaster.com. "
                "Minor paraphrasing is acceptable; confirm the essence of the described options is supported."
            )
        else:
            claim = f"The cited sources support the described {label_for_claim} for Oceans Calling 2026."
            add_ins = (
                "Focus on whether the key points in the answer's description are supported by the provided sources. "
                "Minor wording differences are acceptable."
            )

        await evaluator.verify(
            claim=claim,
            node=node,
            sources=item.urls,
            additional_instruction=add_ins
        )
        return

    # Case 3: otherwise fail (no URLs and not explicitly unavailable)
    evaluator.add_custom_node(
        result=False,
        id=node_id,
        desc=f"{label_for_claim} missing required citation(s) or explicit 'not available' note.",
        parent=parent_node,
        critical=False
    )


async def build_additional_categories(
    evaluator: Evaluator,
    parent_node,
    data: OceansCallingExtraction
) -> None:
    """
    Build the 'Requested_Additional_Categories_Addressed' subtree (non-critical, parallel).
    """
    add_node = evaluator.add_parallel(
        id="Requested_Additional_Categories_Addressed",
        desc="Each requested additional category is either addressed with at least one reliable reference URL or explicitly stated as not available/not found in reliable sources.",
        parent=parent_node,
        critical=False
    )

    await handle_additional_category(
        evaluator, add_node,
        node_id="Ticket_Purchasing_Options_With_Official_Source",
        item=data.ticket_purchasing,
        label_for_claim="ticket purchasing options",
        require_official=True
    )
    await handle_additional_category(
        evaluator, add_node,
        node_id="Expected_Attendance_Capacity",
        item=data.expected_capacity,
        label_for_claim="attendance capacity/crowd size"
    )
    await handle_additional_category(
        evaluator, add_node,
        node_id="Age_Restriction_Policies",
        item=data.age_policy,
        label_for_claim="age restriction/all-ages policies"
    )
    await handle_additional_category(
        evaluator, add_node,
        node_id="Parking_And_Transportation",
        item=data.parking_transport,
        label_for_claim="parking and transportation arrangements"
    )
    await handle_additional_category(
        evaluator, add_node,
        node_id="ADA_Accessibility_Accommodations",
        item=data.ada_accommodations,
        label_for_claim="ADA/accessibility accommodations"
    )
    await handle_additional_category(
        evaluator, add_node,
        node_id="Food_Vendor_Availability",
        item=data.food_vendors,
        label_for_claim="food vendor/concessions availability"
    )
    await handle_additional_category(
        evaluator, add_node,
        node_id="Merchandise_Sales_Areas",
        item=data.merchandise_areas,
        label_for_claim="merchandise/artist merch availability or locations"
    )
    await handle_additional_category(
        evaluator, add_node,
        node_id="Ticket_Refund_Exchange_Policies",
        item=data.refund_exchange,
        label_for_claim="ticket refund/exchange policies"
    )
    await handle_additional_category(
        evaluator, add_node,
        node_id="Security_Screening_Procedures",
        item=data.security_screening,
        label_for_claim="security screening/safety procedures"
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
    Evaluate an answer for the Oceans Calling 2026 festival information task using Mind2Web2 framework.
    """
    # Initialize evaluator (root: non-critical parallel to allow partial credit on optional sections,
    # while critical child nodes will gate overall pass/fail as needed)
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

    # Extract structured information from the answer
    extracted: OceansCallingExtraction = await evaluator.extract(
        prompt=prompt_extract_oc2026(),
        template_class=OceansCallingExtraction,
        extraction_name="oceans_calling_2026_extraction"
    )

    # Add ground truth expectations for traceability
    evaluator.add_ground_truth({
        "expected_dates": EXPECTED_DATES_TEXT,
        "expected_location": EXPECTED_LOCATION_TEXT,
        "expected_duration": EXPECTED_DURATION_TEXT,
        "expected_venue_desc": EXPECTED_VENUE_DESC,
        "expected_official_site": EXPECTED_OFFICIAL_SITE
    })

    # Build the top-level rubric tree according to the JSON
    top_node = evaluator.add_parallel(
        id="Oceans_Calling_2026_Event_Information",
        desc="Comprehensive information about Oceans Calling 2026 covering required attributes plus requested additional categories, with citations.",
        parent=root,
        critical=False  # Keep parent non-critical to allow non-critical subtree;
                        # critical gating is applied via its critical children.
    )

    # Critical: Required core details (with URL verification where applicable)
    await build_core_verifications(evaluator, top_node, extracted.required)

    # Critical: All factual claims must have citations (answer-level meta check)
    await add_citations_guard_leaf(evaluator, top_node)

    # Non-critical: Additional requested categories
    await build_additional_categories(evaluator, top_node, extracted)

    # Optionally record simple custom info for debug
    evaluator.add_custom_info(
        info={
            "extracted_total_urls": len(extracted.all_urls) if extracted and extracted.all_urls else 0,
            "node_count": evaluator.get_node_count()
        },
        info_type="debug_stats",
        info_name="debug_statistics"
    )

    # Return evaluation summary
    return evaluator.get_summary()