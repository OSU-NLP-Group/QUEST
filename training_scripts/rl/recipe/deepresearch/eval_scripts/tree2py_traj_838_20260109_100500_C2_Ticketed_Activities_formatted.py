import asyncio
import logging
import math
import re
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wi_theater_venue_2026"
TASK_DESCRIPTION = (
    "A regional theater company is planning to bring their touring musical production to Wisconsin in 2026. "
    "They need to identify a suitable performing arts venue that meets their specific requirements.\n\n"
    "Identify one theater venue in Wisconsin that satisfies ALL of the following criteria:\n\n"
    "1. The venue must have a seating capacity between 1,800 and 2,500 seats.\n\n"
    "2. The venue must have an orchestra pit or convertible stage area capable of accommodating live musicians "
    "for the production's orchestra.\n\n"
    "3. The venue must provide wheelchair accessible seating that meets ADA compliance requirements "
    "(for venues with 501-5,000 seats, this means at least 1% of the total seating capacity, rounded up to the nearest whole number, "
    "must be designated wheelchair accessible spaces).\n\n"
    "For your identified venue, provide:\n"
    "- The venue name and the city where it is located\n"
    "- The exact seating capacity\n"
    "- Confirmation that an orchestra pit is available\n"
    "- Confirmation of ADA-compliant wheelchair accessible seating\n"
    "- A valid URL to the venue's official website or official information page that documents the seating capacity and accessibility features\n"
    "- Contact information (phone number or email address) for booking inquiries"
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # e.g., "Wisconsin" or "WI"
    seating_capacity: Optional[str] = None  # Prefer a single integer as text (e.g., "2100")
    orchestra_pit_confirmation: Optional[str] = None  # e.g., "yes", "no", or descriptive text
    ada_accessible_seating_confirmation: Optional[str] = None  # e.g., "yes", "ADA-compliant", or descriptive text
    wheelchair_accessible_spaces_count: Optional[str] = None  # digits only if answer provides count; else null
    official_url: Optional[str] = None  # A single official URL that documents capacity & accessibility
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_fields() -> str:
    return """
Extract exactly one Wisconsin performing-arts theater venue referenced in the answer and return the following fields:

- venue_name: The theater/performing arts venue name as written.
- city: The city where the venue is located.
- state: The U.S. state (use "Wisconsin" or "WI" if present).
- seating_capacity: The exact seating capacity as a single integer (digits only, no commas or words). If the answer shows a number like "2,100", return "2100". If the answer only gives a range (e.g., "1,800–2,500") or a non-exact quantity (e.g., "about 2,000"), return null.
- orchestra_pit_confirmation: Return "yes" if the answer explicitly confirms an orchestra pit or convertible stage area for live musicians; otherwise return "no" or null if not stated.
- ada_accessible_seating_confirmation: Return "yes" if the answer explicitly confirms wheelchair accessible seating or ADA compliance; otherwise return "no" or null if not stated.
- wheelchair_accessible_spaces_count: If the answer provides a specific count of wheelchair accessible seating spaces, return that count as digits only (e.g., "25"). Otherwise return null.
- official_url: A single valid URL to the venue’s official website or an official information page (e.g., venue site, city/county-owned venue official page) that the answer uses to document seating capacity and accessibility features. If multiple are mentioned, prefer the one that includes both capacity and accessibility details. If none is provided in the answer, return null.
- contact_phone: A phone number for booking or venue inquiries if provided (as it appears in the answer). If none, return null.
- contact_email: An email address for booking or venue inquiries if provided (as it appears in the answer). If none, return null.

Important:
- Only extract values explicitly present in the answer text.
- Do not fabricate or infer any values that are not clearly stated.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _parse_int_from_text(num_text: Optional[str]) -> Optional[int]:
    if not num_text:
        return None
    # Expect digits-only when possible; however, be robust to commas
    s = num_text.strip()
    s = s.replace(",", " ")
    # Extract a single integer token
    nums = re.findall(r"\d+", s)
    if not nums:
        return None
    # If multiple numbers found in the "capacity" field, take the longest (likely the actual capacity)
    nums.sort(key=lambda x: len(x), reverse=True)
    try:
        return int(nums[0])
    except Exception:
        return None


def _is_valid_phone(phone: Optional[str]) -> bool:
    if not phone:
        return False
    # Simple North American format detector (fairly permissive)
    return bool(re.search(r"(\+1[\s\-\.]?)?(\(?\d{3}\)?[\s\-\.]?)\d{3}[\s\-\.]?\d{4}", phone))


def _is_valid_email(email: Optional[str]) -> bool:
    if not email:
        return False
    return bool(re.search(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", email))


def _normalize_state_text(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().lower()
    if s in {"wi", "wisconsin"}:
        return "Wisconsin"
    return state.strip()


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, root_node, vx: VenueExtraction) -> None:
    """
    Build the rubric tree as specified and run verifications for each leaf.
    """
    # Create the top critical node aggregating all checks in parallel
    top = evaluator.add_parallel(
        id="One_Wisconsin_Theater_Venue_Meeting_All_Criteria",
        desc="Identify exactly one Wisconsin performing-arts theater venue and provide all required details, meeting all stated constraints.",
        parent=root_node,
        critical=True
    )

    # Extract parsed values
    capacity_int = _parse_int_from_text(vx.seating_capacity)
    state_norm = _normalize_state_text(vx.state)
    min_req_spaces = math.ceil(0.01 * capacity_int) if capacity_int else None
    spaces_cnt_int = _parse_int_from_text(vx.wheelchair_accessible_spaces_count) if vx.wheelchair_accessible_spaces_count else None

    # 1) Venue_Name_And_City_Provided (existence check)
    name_city_ok = bool(vx.venue_name and vx.venue_name.strip()) and bool(vx.city and vx.city.strip())
    evaluator.add_custom_node(
        result=name_city_ok,
        id="Venue_Name_And_City_Provided",
        desc="Answer provides the venue name and the city where it is located.",
        parent=top,
        critical=True
    )

    # 2) Venue_Is_Wisconsin_Theater_Or_Performing_Arts_Center (verify against official URL if available)
    wi_leaf = evaluator.add_leaf(
        id="Venue_Is_Wisconsin_Theater_Or_Performing_Arts_Center",
        desc="The identified venue is a theater/performance arts center located in Wisconsin.",
        parent=top,
        critical=True
    )
    venue_for_claim = vx.venue_name or "the venue"
    city_for_claim = vx.city or "a city"
    claim_wi = (
        f"{venue_for_claim} is a theater or performing arts center located in {city_for_claim}, Wisconsin."
    )
    await evaluator.verify(
        claim=claim_wi,
        node=wi_leaf,
        sources=vx.official_url,  # may be None; verifier will fallback to simple verify if missing
        additional_instruction=(
            "Confirm from the page that the place is a theater/performing arts venue and is in Wisconsin (WI). "
            "Allow city naming variations and abbreviations (e.g., WI for Wisconsin)."
        ),
    )

    # 3) Seating_Capacity_Stated_And_In_Range (existence + numeric range check)
    capacity_ok = (capacity_int is not None) and (1800 <= capacity_int <= 2500)
    evaluator.add_custom_node(
        result=capacity_ok,
        id="Seating_Capacity_Stated_And_In_Range",
        desc="Answer states the exact seating capacity (a specific number) and it is between 1,800 and 2,500 seats (inclusive).",
        parent=top,
        critical=True
    )

    # 4) Orchestra_Pit_Or_Convertible_Stage_Available (verify against official URL if available)
    orchestra_leaf = evaluator.add_leaf(
        id="Orchestra_Pit_Or_Convertible_Stage_Available",
        desc="Venue has an orchestra pit or convertible stage area capable of accommodating live musicians (and the answer explicitly confirms this).",
        parent=top,
        critical=True
    )
    claim_pit = (
        f"{venue_for_claim} has an orchestra pit or a convertible stage area suitable for accommodating live musicians."
    )
    await evaluator.verify(
        claim=claim_pit,
        node=orchestra_leaf,
        sources=vx.official_url,
        additional_instruction=(
            "Look for terms like 'orchestra pit', 'pit lift', 'pit filler', 'convertible pit', 'stage extension', "
            "or similar language that clearly indicates the presence of a musician-accommodating pit/area."
        ),
    )

    # 5) ADA_Wheelchair_Accessible_Seating_Compliance (verify)
    ada_leaf = evaluator.add_leaf(
        id="ADA_Wheelchair_Accessible_Seating_Compliance",
        desc="Answer confirms wheelchair accessible seating meeting ADA minimums for 501–5,000 seats: at least 1% of total capacity, rounded up to a whole number.",
        parent=top,
        critical=True
    )
    if capacity_int and min_req_spaces:
        if spaces_cnt_int is not None:
            claim_ada = (
                f"The venue provides at least {spaces_cnt_int} wheelchair accessible seating spaces, and with a total seating capacity of "
                f"{capacity_int}, the ADA requirement for 501–5,000 seats is at least {min_req_spaces} spaces (1% rounded up). "
                f"Therefore, the accessible seating meets or exceeds this ADA minimum."
            )
        else:
            claim_ada = (
                f"The venue provides wheelchair accessible seating and indicates ADA accessibility/compliance. With a total capacity of "
                f"{capacity_int} seats, the ADA minimum for 501–5,000 seats is {min_req_spaces} wheelchair spaces (1% rounded up). "
                f"The official page confirms ADA/accessible seating is provided, indicating compliance with this minimum."
            )
    else:
        # If capacity is missing, we cannot compute the ADA threshold; still verify general ADA-accessible seating statement
        claim_ada = (
            "The venue provides ADA-compliant wheelchair accessible seating."
        )
    await evaluator.verify(
        claim=claim_ada,
        node=ada_leaf,
        sources=vx.official_url,
        additional_instruction=(
            "Verify that the page explicitly mentions wheelchair accessible seating and/or ADA accessibility/compliance. "
            "If a specific wheelchair-seating count is provided, compare it to the ADA minimum (1% of total seats for capacities between 501 and 5,000). "
            "If no count is provided, but ADA accessible seating is clearly documented, treat the compliance claim as supported."
        ),
    )

    # 6) Official_URL_Documents_Capacity_And_Accessibility (verify against the official URL; if missing URL, fail)
    if not vx.official_url or not vx.official_url.strip():
        evaluator.add_leaf(
            id="Official_URL_Documents_Capacity_And_Accessibility",
            desc="Provides a valid URL to the venue’s official website or official information page that documents seating capacity and accessibility features.",
            parent=top,
            critical=True,
            score=0.0,
            status="failed"
        )
    else:
        official_url_leaf = evaluator.add_leaf(
            id="Official_URL_Documents_Capacity_And_Accessibility",
            desc="Provides a valid URL to the venue’s official website or official information page that documents seating capacity and accessibility features.",
            parent=top,
            critical=True
        )
        if capacity_int:
            claim_official = (
                f"This official page includes the venue's seating capacity of {capacity_int} seats and explicitly documents "
                f"wheelchair accessible seating and/or ADA accessibility features."
            )
        else:
            claim_official = (
                f"This official page includes the venue's seating capacity number and explicitly documents wheelchair accessible "
                f"seating and/or ADA accessibility features."
            )
        await evaluator.verify(
            claim=claim_official,
            node=official_url_leaf,
            sources=vx.official_url,
            additional_instruction=(
                "Confirm BOTH: (1) the seating capacity number is present on the page and (2) the page mentions wheelchair accessible seating "
                "or ADA accessibility features. If either element is missing from the page, mark as not supported."
            ),
        )

    # 7) Booking_Contact_Info_Provided (existence check for phone or email)
    contact_ok = _is_valid_phone(vx.contact_phone) or _is_valid_email(vx.contact_email)
    evaluator.add_custom_node(
        result=contact_ok,
        id="Booking_Contact_Info_Provided",
        desc="Provides booking inquiry contact information consisting of at least one of: a phone number or an email address.",
        parent=top,
        critical=True
    )

    # Record some computed info for debugging/traceability
    evaluator.add_custom_info(
        {
            "parsed_capacity": capacity_int,
            "min_required_wheelchair_spaces": min_req_spaces,
            "wheelchair_spaces_from_answer": spaces_cnt_int,
            "normalized_state": state_norm,
            "official_url": vx.official_url
        },
        info_type="computed_fields",
        info_name="computed_normalizations"
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
    Evaluate an answer for the Wisconsin theater venue selection task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # One venue with parallel critical checks
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

    # 1) Extract the structured venue info from the answer
    vx: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue_fields(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # 2) Build verification tree and run checks
    await build_and_verify(evaluator, root, vx)

    # 3) Return evaluation summary
    return evaluator.get_summary()