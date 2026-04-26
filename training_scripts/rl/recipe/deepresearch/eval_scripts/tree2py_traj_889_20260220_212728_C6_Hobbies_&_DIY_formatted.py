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
TASK_ID = "holiday_ornaments_workshops_2025_2026"
TASK_DESCRIPTION = (
    "A family of four (including two children aged 7 and 9) is planning to attend beginner-friendly holiday ornament-"
    "making workshops during the 2025-2026 winter season. They want to compare options from two different major craft "
    "store chains to decide which workshops best fit their schedule and preferences. For each of the two craft store "
    "chains you identify: (1) Find a specific beginner-friendly workshop focused on creating holiday ornaments or "
    "decorations that is scheduled between December 2025 and January 2026; (2) Verify that the workshop is suitable "
    "for children aged 6 and older; (3) Confirm that the workshop duration does not exceed 2 hours; (4) Ensure that all "
    "materials and supplies are included in the registration (no separate materials purchase required); (5) Verify that "
    "the workshop allows online registration/booking; (6) Confirm that the workshop takes place at a physical store "
    "location (not online-only); (7) Verify that the workshop can accommodate their group of 4 participants; (8) "
    "Identify the specific store location (city and state) where the workshop is held; (9) Confirm that this store "
    "location opens before 9:00 AM on Black Friday 2025 (November 29, 2025). Additionally, if available, note whether "
    "each workshop has a cancellation policy that offers refunds or credits with at least 24 hours advance notice. "
    "Provide the workshop name, store chain, location, and reference URLs for all verified information for both workshops."
)

BLACK_FRIDAY_2025_DATE = "November 29, 2025"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WorkshopItem(BaseModel):
    # Core identification
    store_chain: Optional[str] = None
    workshop_name: Optional[str] = None

    # Basic characteristics
    project_type_text: Optional[str] = None
    skill_level_text: Optional[str] = None

    # Scheduling & duration
    scheduled_date_text: Optional[str] = None  # Any date text (e.g., "Dec 14, 2025")
    duration_text: Optional[str] = None        # e.g., "1.5 hours", "90 minutes"

    # Participant requirements
    age_minimum_text: Optional[str] = None     # e.g., "Ages 6+", "Ages 7 and up"
    group_size_text: Optional[str] = None      # e.g., "Capacity 8", "Max 10", "Select up to 4 seats"

    # Logistics
    materials_included_text: Optional[str] = None
    registration_text: Optional[str] = None     # e.g., "Register online", "Sign up", "Add to cart"
    location_format_text: Optional[str] = None  # e.g., "In-store", "In person", "Online"

    # Store
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    black_friday_opening_text: Optional[str] = None  # e.g., "Black Friday opens 7:00 AM"

    # Cancellation policy (optional)
    cancellation_policy_text: Optional[str] = None

    # URLs (source references)
    info_url: Optional[str] = None
    requirements_url: Optional[str] = None
    schedule_url: Optional[str] = None
    logistics_url: Optional[str] = None
    store_url: Optional[str] = None
    policy_url: Optional[str] = None


class WorkshopsExtraction(BaseModel):
    workshops: List[WorkshopItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_workshops() -> str:
    return """
    Extract details for up to two beginner-friendly holiday ornament/decoration workshops from the answer. 
    These must come from two different major national craft store chains (e.g., Michaels, JOANN, Hobby Lobby).

    For each workshop, extract the following fields exactly from the answer (use null for anything not explicitly provided):
    - store_chain: The craft store chain name (e.g., "Michaels", "JOANN", "Hobby Lobby")
    - workshop_name: The specific name/title of the workshop
    - project_type_text: Any text indicating it involves holiday ornaments or holiday decorations
    - skill_level_text: Any mention that it's beginner-friendly or suitable for beginners / no prior experience
    - scheduled_date_text: The date or month/year (should be in Dec 2025 or Jan 2026 if provided)
    - duration_text: The workshop duration (e.g., "2 hours", "90 minutes")
    - age_minimum_text: The minimum age requirement (e.g., "Ages 6+", "Age 6 and up")
    - group_size_text: Any capacity or booking quantity info (e.g., "Capacity 8", "Select up to 4 participants")
    - materials_included_text: Text indicating all materials/supplies are included in registration (no separate purchase)
    - registration_text: Text indicating online registration/booking is available (e.g., "Register", "Sign Up", "Add to cart")
    - location_format_text: Text indicating the format (e.g., "In-store", "In person", "Online")
    - location_city: City of the store location
    - location_state: State of the store location (use the abbreviated or full name exactly as shown)
    - black_friday_opening_text: Any text about Black Friday 2025 opening time for that store
    - cancellation_policy_text: Any mention of a cancellation/refund/credit policy with at least 24 hours notice

    Also extract the reference URLs cited for each aspect (these should be real URLs present in the answer):
    - info_url: The official class/workshop listing URL on the store's website
    - requirements_url: URL mentioning age and participant requirements (may be the same as info_url)
    - schedule_url: URL confirming the schedule and duration (may be the same as info_url)
    - logistics_url: URL confirming materials inclusion, registration method, and location format (may be the same as info_url)
    - store_url: URL for the specific store location/hours page (used for Black Friday hours and location)
    - policy_url: URL mentioning cancellation policy

    Return a JSON object with a 'workshops' array containing up to two objects with the fields above. 
    Only include URLs that are explicitly present in the answer text. If the answer provides more than two workshops, only extract the first two.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def non_empty_urls(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip()]


def pick_sources_for(item: WorkshopItem, preferred: List[str], fallbacks: List[str]) -> List[str]:
    """Return a list of URLs for verification, preferring specific ones but falling back when needed."""
    return non_empty_urls(*preferred) or non_empty_urls(*fallbacks)


# --------------------------------------------------------------------------- #
# Verification for one workshop                                               #
# --------------------------------------------------------------------------- #
async def verify_workshop(
    evaluator: Evaluator,
    parent_node,
    item: WorkshopItem,
    index: int,
    other_chain_name: Optional[str] = None
) -> None:
    """
    Build the verification sub-tree for a single workshop based on the rubric.
    index: 0 for Workshop_1, 1 for Workshop_2
    other_chain_name: for Workshop_2 store chain difference check against Workshop_1
    """
    # Create the main sequential node for this workshop
    ws_node = evaluator.add_sequential(
        id=f"Workshop_{index+1}",
        desc=f"{'First' if index == 0 else 'Second'} workshop identification and verification",
        parent=parent_node,
        critical=False
    )

    # ---------------- Phase 1: Identification ----------------
    ident_node = evaluator.add_parallel(
        id=f"Workshop_{index+1}_Identification",
        desc="Phase 1: Identify the workshop and verify basic characteristics",
        parent=ws_node,
        critical=True
    )

    basic_info_node = evaluator.add_parallel(
        id=f"Workshop_{index+1}_Basic_Info",
        desc="Core workshop characteristics",
        parent=ident_node,
        critical=True
    )

    # 1.A Store Chain (critical)
    store_chain_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Store_Chain",
        desc=(
            "The workshop is offered by a major national craft store chain"
            + ("" if index == 0 else " and is different from the chain selected for Workshop 1")
        ),
        parent=basic_info_node,
        critical=True
    )
    chain_name = item.store_chain or ""
    # Build claim depending on index
    if index == 0:
        store_chain_claim = (
            f"The workshop page indicates it is offered by '{chain_name}', and '{chain_name}' is a major national craft "
            "store chain (acceptable examples include Michaels, JOANN, or Hobby Lobby)."
        )
        await evaluator.verify(
            claim=store_chain_claim,
            node=store_chain_leaf,
            sources=item.info_url,
            additional_instruction=(
                "Verify that the page belongs to or clearly indicates the brand. Consider 'Michaels', 'JOANN', "
                "or 'Hobby Lobby' as major national craft store chains. If the domain or brand matches one of these, "
                "treat this condition as satisfied."
            ),
        )
    else:
        other_chain = other_chain_name or ""
        store_chain_claim = (
            f"The workshop is offered by '{chain_name}', which is a major national craft store chain, "
            f"and '{chain_name}' is different from '{other_chain}'."
        )
        # This involves a logical difference check (non-web factual) plus major chain check.
        # We'll use simple verification so the judge can consider both parts using the answer context.
        await evaluator.verify(
            claim=store_chain_claim,
            node=store_chain_leaf,
            sources=item.info_url,  # still provide the page for brand support
            additional_instruction=(
                "Judge two aspects: (1) '{chain_name}' is one of the major national craft chains (Michaels, JOANN, Hobby Lobby); "
                f"(2) '{chain_name}' is different from '{other_chain}'. "
                "Minor name variants (e.g., 'JOANN Fabrics') count as the same chain."
            ),
        )

    # 1.B Project Type (critical)
    project_type_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Project_Type",
        desc="The workshop involves creating holiday ornaments or holiday decorations",
        parent=basic_info_node,
        critical=True
    )
    await evaluator.verify(
        claim="This workshop focuses on making holiday ornaments or holiday decorations.",
        node=project_type_leaf,
        sources=item.info_url,
        additional_instruction=(
            "Accept synonyms like 'holiday ornament', 'Christmas ornaments', 'seasonal decorations', 'holiday decor'. "
            "The page should explicitly suggest the project involves ornaments or decorations."
        ),
    )

    # 1.C Skill Level (critical)
    skill_level_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Skill_Level",
        desc="The workshop is explicitly labeled or described as beginner-friendly",
        parent=basic_info_node,
        critical=True
    )
    await evaluator.verify(
        claim="This workshop is beginner-friendly or suitable for beginners with no prior experience.",
        node=skill_level_leaf,
        sources=item.info_url,
        additional_instruction=(
            "Accept phrases like 'Beginner', 'All levels', 'No experience required', or 'Beginner-friendly'."
        ),
    )

    # 1.D Info URL existence (critical custom)
    evaluator.add_custom_node(
        result=bool(item.info_url and item.info_url.strip()),
        id=f"Workshop_{index+1}_Info_URL",
        desc="A reference URL from the craft store's official website or class listing that confirms the workshop details",
        parent=ident_node,
        critical=True
    )

    # ---------------- Phase 2: Requirements Verification ----------------
    req_node = evaluator.add_parallel(
        id=f"Workshop_{index+1}_Requirements_Verification",
        desc="Phase 2: Verify participant and scheduling requirements",
        parent=ws_node,
        critical=True
    )

    participant_node = evaluator.add_parallel(
        id=f"Workshop_{index+1}_Participant_Requirements",
        desc="Participant eligibility requirements",
        parent=req_node,
        critical=True
    )

    # 2.A Age Minimum (critical)
    age_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Age_Minimum",
        desc="The workshop allows participants aged 6 years or older",
        parent=participant_node,
        critical=True
    )
    age_sources = pick_sources_for(item, [item.requirements_url], [item.info_url])
    await evaluator.verify(
        claim="The workshop permits participants aged 6+ (minimum age requirement is 6 or younger).",
        node=age_leaf,
        sources=age_sources,
        additional_instruction=(
            "Accept if the page states 'Ages 6+', '6 and up', 'ages 5+' (which implies 6+ are allowed), or similar. "
            "The key is that 6-year-olds are allowed to attend."
        ),
    )

    # 2.B Minimum participants / capacity for 4 (critical)
    min_group_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Min_Participants",
        desc="The workshop accommodates a group of at least 4 participants",
        parent=participant_node,
        critical=True
    )
    capacity_sources = pick_sources_for(item, [item.requirements_url], [item.info_url])
    await evaluator.verify(
        claim=(
            "The workshop can accommodate a group of 4 participants (e.g., class capacity is at least 4, "
            "or the online booking allows selecting quantity 4 or more)."
        ),
        node=min_group_leaf,
        sources=capacity_sources,
        additional_instruction=(
            "Look for capacity info (>= 4) or UI that allows booking 4 seats in one registration. "
            "If capacity is clearly listed as >= 4, or booking interface allows 4, consider this satisfied."
        ),
    )

    # 2.C Requirements URL existence (critical custom)
    evaluator.add_custom_node(
        result=bool((item.requirements_url and item.requirements_url.strip()) or (item.info_url and item.info_url.strip())),
        id=f"Workshop_{index+1}_Requirements_URL",
        desc="A reference URL that confirms age and participant requirements",
        parent=req_node,
        critical=True
    )

    # Scheduling node (critical)
    scheduling_node = evaluator.add_parallel(
        id=f"Workshop_{index+1}_Scheduling",
        desc="Workshop timing and duration",
        parent=req_node,
        critical=True
    )

    # 2.D Time period (Dec 2025 or Jan 2026) (critical)
    time_period_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Time_Period",
        desc="The workshop is scheduled to take place during December 2025 or January 2026",
        parent=scheduling_node,
        critical=True
    )
    schedule_sources = pick_sources_for(item, [item.schedule_url], [item.info_url])
    await evaluator.verify(
        claim=(
            "This workshop has at least one session scheduled between December 1, 2025 and January 31, 2026."
        ),
        node=time_period_leaf,
        sources=schedule_sources,
        additional_instruction=(
            "Accept if the listing shows a date in December 2025 or January 2026. If multiple dates, at least one must "
            "fall in that range."
        ),
    )

    # 2.E Duration <= 2 hours (critical)
    duration_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Duration",
        desc="The workshop duration does not exceed 2 hours",
        parent=scheduling_node,
        critical=True
    )
    await evaluator.verify(
        claim="The workshop duration is 120 minutes or less (does not exceed 2 hours).",
        node=duration_leaf,
        sources=schedule_sources,
        additional_instruction=(
            "Accept durations like '2 hours', '90 minutes', '1.5 hours', etc. If duration appears as 2 hours exactly, "
            "that is acceptable."
        ),
    )

    # 2.F Schedule URL existence (critical custom)
    evaluator.add_custom_node(
        result=bool((item.schedule_url and item.schedule_url.strip()) or (item.info_url and item.info_url.strip())),
        id=f"Workshop_{index+1}_Schedule_URL",
        desc="A reference URL that confirms the workshop schedule and duration",
        parent=req_node,
        critical=True
    )

    # ---------------- Phase 3: Logistics Verification ----------------
    logistics_node = evaluator.add_parallel(
        id=f"Workshop_{index+1}_Logistics_Verification",
        desc="Phase 3: Verify registration and location logistics",
        parent=ws_node,
        critical=True
    )

    logistics_details_node = evaluator.add_parallel(
        id=f"Workshop_{index+1}_Logistics",
        desc="Registration and location details",
        parent=logistics_node,
        critical=True
    )

    logistics_sources = pick_sources_for(item, [item.logistics_url], [item.info_url])

    # 3.A Materials included (critical)
    materials_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Materials",
        desc="All necessary materials and supplies are included in the workshop registration fee",
        parent=logistics_details_node,
        critical=True
    )
    await evaluator.verify(
        claim="All necessary materials/supplies are included in registration; no separate materials purchase required.",
        node=materials_leaf,
        sources=logistics_sources,
        additional_instruction=(
            "Look for wording like 'materials included', 'supplies provided', or 'no additional purchase required'."
        ),
    )

    # 3.B Online registration (critical)
    registration_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Registration",
        desc="The workshop allows online registration or booking",
        parent=logistics_details_node,
        critical=True
    )
    await evaluator.verify(
        claim="Online registration/booking is available for the workshop.",
        node=registration_leaf,
        sources=logistics_sources,
        additional_instruction=(
            "Look for a 'Register', 'Sign up', 'Book', or 'Add to cart' button/link on the class page."
        ),
    )

    # 3.C Physical location (critical)
    physical_location_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Physical_Location",
        desc="The workshop is held at a physical store location (not an online-only virtual workshop)",
        parent=logistics_details_node,
        critical=True
    )
    await evaluator.verify(
        claim="The workshop is in-person at a physical store location (not online-only).",
        node=physical_location_leaf,
        sources=logistics_sources,
        additional_instruction=(
            "Accept phrases like 'In-store', 'In person', 'At [Store Name] [City, State]'. "
            "Reject if clearly marked 'Online' only."
        ),
    )

    # 3.D Logistics URL existence (critical custom)
    evaluator.add_custom_node(
        result=bool((item.logistics_url and item.logistics_url.strip()) or (item.info_url and item.info_url.strip())),
        id=f"Workshop_{index+1}_Logistics_URL",
        desc="A reference URL that confirms materials inclusion, registration method, and location format",
        parent=logistics_node,
        critical=True
    )

    # ---------------- Phase 4: Store Verification ----------------
    store_node = evaluator.add_parallel(
        id=f"Workshop_{index+1}_Store_Verification",
        desc="Phase 4: Verify store location and operational requirements",
        parent=ws_node,
        critical=True
    )

    store_details_node = evaluator.add_parallel(
        id=f"Workshop_{index+1}_Store_Details",
        desc="Store location and Black Friday hours",
        parent=store_node,
        critical=True
    )

    store_sources = pick_sources_for(item, [item.store_url], [item.info_url])

    # 4.A Store location (critical)
    store_location_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Store_Location",
        desc="The specific store location (city and state) where the workshop is held",
        parent=store_details_node,
        critical=True
    )
    city = item.location_city or ""
    state = item.location_state or ""
    await evaluator.verify(
        claim=f"The workshop takes place at a store in {city}, {state}.",
        node=store_location_leaf,
        sources=store_sources,
        additional_instruction=(
            "Verify that the page indicates the store location city and state. "
            "Minor formatting differences (e.g., 'CA' vs 'California') are acceptable."
        ),
    )

    # 4.B Black Friday hours (critical)
    bf_hours_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Black_Friday_Hours",
        desc=f"The store opens before 9:00 AM on Black Friday 2025 ({BLACK_FRIDAY_2025_DATE})",
        parent=store_details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The store's opening time on {BLACK_FRIDAY_2025_DATE} is earlier than 9:00 AM.",
        node=bf_hours_leaf,
        sources=store_sources,
        additional_instruction=(
            "Look for special hours for Black Friday 2025. Accept any opening time strictly before 9:00 AM "
            "(e.g., 6:00 AM, 7:00 AM, 8:00 AM). If the page lists holiday hours including Black Friday for that specific store, use that."
        ),
    )

    # 4.C Store URL existence (critical custom)
    evaluator.add_custom_node(
        result=bool((item.store_url and item.store_url.strip()) or (item.info_url and item.info_url.strip())),
        id=f"Workshop_{index+1}_Store_URL",
        desc="A reference URL that confirms the store location and Black Friday 2025 hours",
        parent=store_node,
        critical=True
    )

    # ---------------- Phase 5: Optional Policy Information ----------------
    policy_node = evaluator.add_parallel(
        id=f"Workshop_{index+1}_Policy_Information",
        desc="Phase 5: Optional policy information",
        parent=ws_node,
        critical=False
    )

    cancellation_node = evaluator.add_parallel(
        id=f"Workshop_{index+1}_Cancellation",
        desc="Cancellation and refund policy",
        parent=policy_node,
        critical=False
    )

    # 5.A Cancellation policy (non-critical)
    cancel_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Cancellation_Policy",
        desc="The workshop has a cancellation policy that offers refunds or credits with at least 24 hours advance notice",
        parent=cancellation_node,
        critical=False
    )
    cancel_sources = pick_sources_for(item, [item.policy_url], [item.info_url])
    await evaluator.verify(
        claim="A cancellation policy offers refunds or credits with at least 24 hours advance notice.",
        node=cancel_leaf,
        sources=cancel_sources,
        additional_instruction=(
            "Look for language like '24-hour notice', 'refunds/credits', 'cancellation policy'. "
            "If no such policy is mentioned, this should fail (non-critical)."
        ),
    )

    # 5.B Policy URL existence (non-critical custom)
    evaluator.add_custom_node(
        result=bool((item.policy_url and item.policy_url.strip()) or (item.info_url and item.info_url.strip())),
        id=f"Workshop_{index+1}_Policy_URL",
        desc="A reference URL that confirms the cancellation policy",
        parent=cancellation_node,
        critical=False
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
    Evaluate an answer for the holiday ornament-making workshops task.
    """
    # Initialize evaluator
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

    # Extract workshop info
    extraction = await evaluator.extract(
        prompt=prompt_extract_workshops(),
        template_class=WorkshopsExtraction,
        extraction_name="workshops_extraction",
    )

    # Normalize to exactly two workshops (pad if fewer)
    workshops: List[WorkshopItem] = list(extraction.workshops[:2])
    while len(workshops) < 2:
        workshops.append(WorkshopItem())

    # Build verification trees for the two workshops sequentially
    # Workshop 1
    await verify_workshop(
        evaluator=evaluator,
        parent_node=root,
        item=workshops[0],
        index=0,
        other_chain_name=None
    )

    # Workshop 2 (requires chain difference from workshop 1)
    chain1 = workshops[0].store_chain or ""
    await verify_workshop(
        evaluator=evaluator,
        parent_node=root,
        item=workshops[1],
        index=1,
        other_chain_name=chain1
    )

    # Optional: record custom info for debugging/context
    evaluator.add_custom_info(
        info={
            "months_required": ["December 2025", "January 2026"],
            "black_friday_required_date": BLACK_FRIDAY_2025_DATE,
            "accepted_major_chains": ["Michaels", "JOANN", "Hobby Lobby"],
        },
        info_type="constraints_context",
    )

    return evaluator.get_summary()