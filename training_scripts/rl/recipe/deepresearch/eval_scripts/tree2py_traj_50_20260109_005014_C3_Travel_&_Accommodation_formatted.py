import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "portland_ada_hotel_rollin"
TASK_DESCRIPTION = (
    "I am planning a trip to Portland, Oregon and need to find an accessible hotel accommodation that meets specific "
    "ADA requirements for mobility accessibility. Find one hotel located in downtown Portland, Oregon that offers "
    "accessible guest rooms with roll-in showers, and verify that it meets ALL of the following specifications:\n\n"
    "Room Accessibility Requirements:\n"
    "- Doorways must have at least 32 inches of clear width\n"
    "- The room must have visual notification devices including both a fire alarm with strobe light and a doorbell with "
    "strobe light for guests with hearing impairments\n"
    "- The bed must have at least 30 inches of maneuvering clearance on each side\n\n"
    "Bathroom Requirements:\n"
    "- The toilet must have a seat height between 17 and 19 inches\n"
    "- The toilet area must have grab bars installed\n\n"
    "Roll-in Shower Requirements:\n"
    "- The shower must be at least 30 inches wide by 60 inches long (minimum dimensions per ADA standards)\n"
    "- The shower must have either a fold-down or fixed shower seat\n"
    "- The shower must have grab bars installed\n"
    "- The shower must have a detachable hand-held shower head\n\n"
    "Provide the hotel name and reference URLs that confirm the hotel's location in downtown Portland and that verify "
    "all of the above accessibility features are present in the accessible room with roll-in shower."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FeatureEvidence(BaseModel):
    detail: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class RoomFeatures(BaseModel):
    doorway_clear_width: Optional[FeatureEvidence] = None
    visual_notification_devices: Optional[FeatureEvidence] = None
    bed_maneuvering_clearance: Optional[FeatureEvidence] = None


class ToiletFeatures(BaseModel):
    seat_height: Optional[FeatureEvidence] = None
    grab_bars: Optional[FeatureEvidence] = None


class ShowerFeatures(BaseModel):
    minimum_dimensions: Optional[FeatureEvidence] = None
    shower_seat: Optional[FeatureEvidence] = None
    grab_bars: Optional[FeatureEvidence] = None
    handheld_showerhead: Optional[FeatureEvidence] = None


class HotelExtraction(BaseModel):
    hotel_name: Optional[str] = None
    # URL(s) confirming "downtown Portland" location (e.g., hotel landing page, neighborhood page, map detail, etc.)
    location_urls: List[str] = Field(default_factory=list)
    # URL(s) confirming there are accessible guest rooms with roll-in showers
    roll_in_shower_urls: List[str] = Field(default_factory=list)
    # Per-feature evidence (exact phrasing from answer) plus any feature-specific URLs if provided
    room: Optional[RoomFeatures] = None
    toilet: Optional[ToiletFeatures] = None
    shower: Optional[ShowerFeatures] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel() -> str:
    return """
    Extract the single (primary) hotel and its evidence from the answer. The user needs one hotel in downtown Portland, Oregon
    that offers an accessible guest room with a roll-in shower meeting all listed ADA requirements. Return exactly the fields below.

    You must strictly extract only what is explicitly present in the answer. Do not invent URLs or details.
    If a detail is not present, return null for 'detail' and an empty list for 'urls'.

    JSON schema to return:
    {
      "hotel_name": string | null,
      "location_urls": string[] ,           // URL(s) explicitly provided that indicate the hotel is in downtown Portland
      "roll_in_shower_urls": string[] ,     // URL(s) explicitly provided that indicate the hotel has accessible guest rooms with roll-in showers
      "room": {
        "doorway_clear_width": { "detail": string | null, "urls": string[] },
        "visual_notification_devices": { "detail": string | null, "urls": string[] }, // ideally mentions both strobe fire alarm and strobe doorbell
        "bed_maneuvering_clearance": { "detail": string | null, "urls": string[] }
      } | null,
      "toilet": {
        "seat_height": { "detail": string | null, "urls": string[] },
        "grab_bars": { "detail": string | null, "urls": string[] }
      } | null,
      "shower": {
        "minimum_dimensions": { "detail": string | null, "urls": string[] },   // target ≥ 30 in by 60 in
        "shower_seat": { "detail": string | null, "urls": string[] },
        "grab_bars": { "detail": string | null, "urls": string[] },
        "handheld_showerhead": { "detail": string | null, "urls": string[] }
      } | null
    }

    Notes:
    - location_urls should reflect pages that explicitly indicate the hotel is in "downtown Portland" or equivalent phrasing
      (e.g., "Downtown Portland", "Portland Downtown", "city center").
    - roll_in_shower_urls should reflect pages stating accessible guest rooms with roll-in shower exist.
    - For each 'FeatureEvidence', set 'detail' to the exact phrase/measurement cited in the answer (if present).
    - For 'urls' lists in any feature evidence, include only URLs explicitly provided in the answer that are meant to support that feature.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if u and isinstance(u, str) and u not in merged:
                merged.append(u)
    return merged


def _evidence_urls(ev: Optional[FeatureEvidence]) -> List[str]:
    if ev and ev.urls:
        return [u for u in ev.urls if isinstance(u, str) and u.strip() != ""]
    return []


def _string_or_fallback(name: Optional[str], fallback: str) -> str:
    name = (name or "").strip()
    return name if name else fallback


def _no_url_fail_instruction() -> str:
    return (
        "Important: Treat the claim as NOT SUPPORTED unless at least one provided URL explicitly and clearly supports it. "
        "If the provided URL(s) are missing, irrelevant, or do not clearly confirm the claim, mark it as not supported."
    )


def _general_match_instructions() -> str:
    return (
        "Allow reasonable synonyms and formatting variants for dimensions (e.g., 30x60, 30 by 60 inches, 30'' x 60''). "
        "For 'downtown Portland', accept equivalent phrases such as 'Downtown Portland', 'Portland Downtown', or 'city center'. "
        "For handheld showerheads, accept terms such as 'hand-held', 'hand shower', or 'detachable shower head'. "
        "For shower seats, accept 'fold-down seat', 'fixed seat', or 'bench' if clearly part of the shower for accessibility."
    )


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_and_verify_portland_hotel(evaluator: Evaluator, extracted: HotelExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    We follow the original rubric structure, with a top-level sequential critical node and
    critical parallel subgroups. We split the 'Visual_Notification_Devices' into two atomic checks
    (fire alarm with strobe; doorbell with strobe) to adhere to single-step leaf checks.
    """
    # Top-level critical sequential node
    main_node = evaluator.add_sequential(
        id="Portland_Accessible_Hotel_Roll_In_Shower_Verification",
        desc="Verify a downtown Portland hotel offers accessible rooms with ADA-compliant roll-in showers meeting all required specifications",
        parent=evaluator.root,
        critical=True
    )

    # ------------------------------------------------------------------ #
    # Step 1: Hotel location & accessibility (parallel, critical)        #
    # ------------------------------------------------------------------ #
    step1 = evaluator.add_parallel(
        id="Hotel_Location_And_Accessibility",
        desc="Identify and verify hotel in downtown Portland, Oregon offering accessible rooms with roll-in showers",
        parent=main_node,
        critical=True
    )

    hotel_name = _string_or_fallback(extracted.hotel_name, "the hotel referenced in the answer")

    # Leaf: Hotel in downtown Portland
    leaf_loc = evaluator.add_leaf(
        id="Hotel_Downtown_Portland",
        desc="Provide hotel name and reference URL confirming location in downtown Portland, Oregon",
        parent=step1,
        critical=True
    )
    loc_claim = (
        f"This page confirms that the hotel '{hotel_name}' is located in downtown Portland, Oregon "
        f"(accept synonyms like 'Downtown Portland', 'Portland Downtown', or 'city center')."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=leaf_loc,
        sources=extracted.location_urls if extracted.location_urls else None,
        additional_instruction=_general_match_instructions() + " " + _no_url_fail_instruction()
    )

    # Leaf: Accessible room with roll-in shower exists
    leaf_rollin_exists = evaluator.add_leaf(
        id="Accessible_Room_With_Roll_In_Shower",
        desc="Confirm hotel offers accessible guest rooms with roll-in shower facilities (provide reference URL)",
        parent=step1,
        critical=True
    )
    # Prefer explicit roll-in URLs; otherwise, fall back to shower feature URLs
    shower_urls_union = _merge_sources(
        extracted.roll_in_shower_urls,
        _evidence_urls(extracted.shower.minimum_dimensions if extracted.shower else None),
        _evidence_urls(extracted.shower.shower_seat if extracted.shower else None),
        _evidence_urls(extracted.shower.grab_bars if extracted.shower else None),
        _evidence_urls(extracted.shower.handheld_showerhead if extracted.shower else None)
    )
    rollin_claim = f"This page confirms that '{hotel_name}' offers accessible guest rooms with roll-in shower(s)."
    await evaluator.verify(
        claim=rollin_claim,
        node=leaf_rollin_exists,
        sources=shower_urls_union if shower_urls_union else None,
        additional_instruction="The page should clearly indicate the existence of roll-in shower(s) in accessible guest rooms. "
                               + _no_url_fail_instruction()
    )

    # ------------------------------------------------------------------ #
    # Step 2: Room Accessibility Features (parallel, critical)           #
    # ------------------------------------------------------------------ #
    step2 = evaluator.add_parallel(
        id="Room_Accessibility_Features",
        desc="Verify accessible room has required ADA-compliant features",
        parent=main_node,
        critical=True
    )

    # Doorway width ≥ 32 inches
    leaf_door = evaluator.add_leaf(
        id="Doorway_Width_Clearance",
        desc="Doorways have at least 32 inches of clear width",
        parent=step2,
        critical=True
    )
    door_urls = _merge_sources(
        _evidence_urls(extracted.room.doorway_clear_width) if (extracted.room and extracted.room.doorway_clear_width) else [],
        shower_urls_union  # fallback to general accessible/roll-in evidence if specific URL isn't provided
    )
    door_claim = "The accessible guest room doorways have at least 32 inches of clear width."
    await evaluator.verify(
        claim=door_claim,
        node=leaf_door,
        sources=door_urls if door_urls else None,
        additional_instruction="Confirm explicit wording indicating 32-inch (or greater) clear width at the door. "
                               + _general_match_instructions() + " " + _no_url_fail_instruction()
    )

    # Visual notification devices (split into two atomic leaves under a sub-node)
    visual_node = evaluator.add_parallel(
        id="Visual_Notification_Devices",
        desc="Room has visual notification devices (fire alarm with strobe light and doorbell with strobe light)",
        parent=step2,
        critical=True
    )
    # Fire alarm with strobe
    leaf_visual_fire = evaluator.add_leaf(
        id="Visual_Alarm_Strobe",
        desc="Room has a fire alarm with strobe light",
        parent=visual_node,
        critical=True
    )
    visual_urls = _merge_sources(
        _evidence_urls(extracted.room.visual_notification_devices) if (extracted.room and extracted.room.visual_notification_devices) else [],
        shower_urls_union
    )
    await evaluator.verify(
        claim="The accessible guest room includes a visual fire alarm with a strobe light.",
        node=leaf_visual_fire,
        sources=visual_urls if visual_urls else None,
        additional_instruction="Look for explicit 'visual alarm with strobe' or equivalent phrasing specifically for the room. "
                               + _no_url_fail_instruction()
    )
    # Doorbell with strobe
    leaf_visual_doorbell = evaluator.add_leaf(
        id="Doorbell_Strobe",
        desc="Room has a doorbell with strobe light",
        parent=visual_node,
        critical=True
    )
    await evaluator.verify(
        claim="The accessible guest room includes a doorbell with a strobe light (visual doorbell).",
        node=leaf_visual_doorbell,
        sources=visual_urls if visual_urls else None,
        additional_instruction="Look for explicit 'doorbell with strobe' or 'visual doorbell' for the room. "
                               + _no_url_fail_instruction()
    )

    # Bed maneuvering clearance ≥ 30 inches per side
    leaf_bed = evaluator.add_leaf(
        id="Bed_Maneuvering_Clearance",
        desc="Bed has at least 30 inches of maneuvering clearance on each side",
        parent=step2,
        critical=True
    )
    bed_urls = _merge_sources(
        _evidence_urls(extracted.room.bed_maneuvering_clearance) if (extracted.room and extracted.room.bed_maneuvering_clearance) else [],
        shower_urls_union
    )
    bed_claim = "The accessible guest room provides at least 30 inches of maneuvering clearance on each side of the bed."
    await evaluator.verify(
        claim=bed_claim,
        node=leaf_bed,
        sources=bed_urls if bed_urls else None,
        additional_instruction="Look for explicit bed-side clearance dimensions meeting or exceeding 30 inches on BOTH sides. "
                               + _no_url_fail_instruction()
    )

    # ------------------------------------------------------------------ #
    # Step 3: Bathroom ADA Compliance (parallel, critical)               #
    # ------------------------------------------------------------------ #
    step3 = evaluator.add_parallel(
        id="Bathroom_ADA_Compliance",
        desc="Verify bathroom toilet and roll-in shower meet all ADA specifications",
        parent=main_node,
        critical=True
    )

    # Toilet seat height 17–19 inches
    leaf_seat_height = evaluator.add_leaf(
        id="Toilet_Seat_Height",
        desc="Toilet seat height is between 17 and 19 inches",
        parent=step3,
        critical=True
    )
    toilet_seat_urls = _merge_sources(
        _evidence_urls(extracted.toilet.seat_height) if (extracted.toilet and extracted.toilet.seat_height) else [],
        shower_urls_union
    )
    await evaluator.verify(
        claim="The accessible bathroom toilet seat height is between 17 and 19 inches (inclusive).",
        node=leaf_seat_height,
        sources=toilet_seat_urls if toilet_seat_urls else None,
        additional_instruction="Confirm explicit toilet seat height within the 17–19 inch range. "
                               + _general_match_instructions() + " " + _no_url_fail_instruction()
    )

    # Toilet grab bars
    leaf_toilet_grab = evaluator.add_leaf(
        id="Toilet_Grab_Bars",
        desc="Toilet area has grab bars installed",
        parent=step3,
        critical=True
    )
    toilet_grab_urls = _merge_sources(
        _evidence_urls(extracted.toilet.grab_bars) if (extracted.toilet and extracted.toilet.grab_bars) else [],
        shower_urls_union
    )
    await evaluator.verify(
        claim="The accessible bathroom has grab bars installed at the toilet area.",
        node=leaf_toilet_grab,
        sources=toilet_grab_urls if toilet_grab_urls else None,
        additional_instruction="The page should clearly state toilet grab bars or equivalent wording. " + _no_url_fail_instruction()
    )

    # Roll-in shower complete specifications (parallel, critical)
    shower_node = evaluator.add_parallel(
        id="Roll_In_Shower_Complete_Specifications",
        desc="Verify roll-in shower meets all dimensional and equipment requirements",
        parent=step3,
        critical=True
    )

    # 30x60 minimum dimensions
    leaf_shower_dims = evaluator.add_leaf(
        id="Shower_Minimum_Dimensions",
        desc="Roll-in shower is at least 30 inches wide by 60 inches long",
        parent=shower_node,
        critical=True
    )
    shower_dim_urls = _merge_sources(
        _evidence_urls(extracted.shower.minimum_dimensions) if (extracted.shower and extracted.shower.minimum_dimensions) else [],
        shower_urls_union
    )
    await evaluator.verify(
        claim="The roll-in shower has minimum dimensions of at least 30 inches in width and 60 inches in length.",
        node=leaf_shower_dims,
        sources=shower_dim_urls if shower_dim_urls else None,
        additional_instruction=_general_match_instructions() + " " + _no_url_fail_instruction()
    )

    # Shower seat presence
    leaf_shower_seat = evaluator.add_leaf(
        id="Shower_Seat_Presence",
        desc="Shower has either a fold-down or fixed shower seat",
        parent=shower_node,
        critical=True
    )
    shower_seat_urls = _merge_sources(
        _evidence_urls(extracted.shower.shower_seat) if (extracted.shower and extracted.shower.shower_seat) else [],
        shower_urls_union
    )
    await evaluator.verify(
        claim="The roll-in shower includes a shower seat (fold-down or fixed).",
        node=leaf_shower_seat,
        sources=shower_seat_urls if shower_seat_urls else None,
        additional_instruction=_general_match_instructions() + " " + _no_url_fail_instruction()
    )

    # Shower grab bars
    leaf_shower_grab = evaluator.add_leaf(
        id="Shower_Grab_Bars",
        desc="Shower has grab bars installed",
        parent=shower_node,
        critical=True
    )
    shower_grab_urls = _merge_sources(
        _evidence_urls(extracted.shower.grab_bars) if (extracted.shower and extracted.shower.grab_bars) else [],
        shower_urls_union
    )
    await evaluator.verify(
        claim="The roll-in shower has grab bars installed.",
        node=leaf_shower_grab,
        sources=shower_grab_urls if shower_grab_urls else None,
        additional_instruction="The page should clearly mention grab bars in the shower. " + _no_url_fail_instruction()
    )

    # Handheld shower head
    leaf_shower_handheld = evaluator.add_leaf(
        id="Shower_Handheld_Head",
        desc="Shower has a detachable hand-held shower head",
        parent=shower_node,
        critical=True
    )
    shower_handheld_urls = _merge_sources(
        _evidence_urls(extracted.shower.handheld_showerhead) if (extracted.shower and extracted.shower.handheld_showerhead) else [],
        shower_urls_union
    )
    await evaluator.verify(
        claim="The roll-in shower includes a detachable hand-held shower head (hand shower).",
        node=leaf_shower_handheld,
        sources=shower_handheld_urls if shower_handheld_urls else None,
        additional_instruction=_general_match_instructions() + " " + _no_url_fail_instruction()
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
    Evaluate an answer for the Portland accessible hotel with roll-in shower task.
    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # root strategy; the critical main node will live under root
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotel(),
        template_class=HotelExtraction,
        extraction_name="hotel_accessibility_extraction",
    )

    # Optionally record chosen hotel and provided URLs in summary
    evaluator.add_custom_info(
        info={
            "hotel_name": extracted.hotel_name,
            "location_urls": extracted.location_urls,
            "roll_in_shower_urls": extracted.roll_in_shower_urls,
        },
        info_type="extraction_summary",
        info_name="chosen_hotel_and_urls"
    )

    # Build verification tree and run checks
    await build_and_verify_portland_hotel(evaluator, extracted)

    # Return the final evaluation summary
    return evaluator.get_summary()