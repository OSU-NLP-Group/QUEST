import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.verification_tree import VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sustainable_beachfront_hotel_ca"
TASK_DESCRIPTION = (
    "A corporation is planning a sustainable corporate retreat in California and requires a beachfront hotel that "
    "demonstrates exceptional environmental commitment. The hotel must hold LEED Gold certification (earning 60-79 points "
    "on the LEED scorecard), have fewer than 250 rooms, and be currently operational. The property must implement at least "
    "one major energy-efficient system (such as LED lighting throughout, solar panels, or HVAC systems providing at least "
    "40% energy savings) and at least one water conservation system achieving a minimum of 20% water usage reduction "
    "(such as rainwater harvesting, low-flow fixtures, or greywater recycling). Additionally, the hotel must have conference "
    "facilities and offer green meeting or sustainable event planning services. Identify a California beachfront hotel that "
    "meets all these requirements and provide verification of its LEED Gold certification status, specific energy efficiency "
    "features, water conservation systems with documented reduction achievements, and corporate event capabilities with green "
    "meeting options."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EnergyFeature(BaseModel):
    kind: Optional[str] = None  # e.g., "LED", "Solar", "HVAC", "Other"
    description: Optional[str] = None
    savings_percent: Optional[str] = None  # e.g., "40%", "45 percent", ">= 40%"
    urls: List[str] = Field(default_factory=list)


class WaterFeature(BaseModel):
    kind: Optional[str] = None  # e.g., "Low-flow fixtures", "Rainwater harvesting", "Greywater", "Other"
    description: Optional[str] = None
    reduction_percent: Optional[str] = None  # e.g., "20%", "25 percent"
    urls: List[str] = Field(default_factory=list)


class HotelExtraction(BaseModel):
    # Identification
    name: Optional[str] = None
    official_url: Optional[str] = None

    # Location and beachfront
    city: Optional[str] = None
    state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    beachfront_urls: List[str] = Field(default_factory=list)

    # Rooms and operational status
    rooms_count: Optional[str] = None
    rooms_urls: List[str] = Field(default_factory=list)
    operational_urls: List[str] = Field(default_factory=list)

    # LEED
    leed_level: Optional[str] = None  # Expecting "Gold" if applicable
    leed_points: Optional[str] = None  # Free-form (e.g., "68 points"), can be null
    leed_urls: List[str] = Field(default_factory=list)

    # Energy and water features
    energy_features: List[EnergyFeature] = Field(default_factory=list)
    water_features: List[WaterFeature] = Field(default_factory=list)

    # Corporate events
    conference_urls: List[str] = Field(default_factory=list)
    green_meeting_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_candidate() -> str:
    return """
    You must extract a single California beachfront hotel proposed by the answer as the final recommended property
    for a sustainable corporate retreat. If multiple hotels are mentioned, choose the one presented as the final or primary recommendation;
    otherwise choose the first property described with the most details. Extract the following fields strictly from the answer text and its cited URLs:

    Identification:
    - name: Hotel's exact name (single property).
    - official_url: A URL explicitly included in the answer that corresponds to the hotel's official website (if provided).

    Location & beachfront:
    - city: City of the hotel (if mentioned).
    - state: State (should be "CA" or "California" if the hotel is in California; else null).
    - location_urls: All URLs in the answer that support the hotel's address/location (include hotel contact/location pages or credible listings).
    - beachfront_urls: All URLs in the answer that support that the property is beachfront or oceanfront (include the hotel's site pages or credible sources).

    Rooms & operational:
    - rooms_count: The number or description of room count as provided (e.g., "210 rooms", "123 keys"). Keep as a string.
    - rooms_urls: All URLs in the answer that support the room count.
    - operational_urls: All URLs that demonstrate the hotel is open/accepting reservations (e.g., booking or "book now" pages). If none provided, return an empty list.

    LEED:
    - leed_level: Extract the LEED certification level explicitly claimed (e.g., "Gold", "Silver"). If absent, null.
    - leed_points: Extract any points mentioned for LEED (e.g., "68 points") if provided; otherwise null.
    - leed_urls: All URLs that substantiate LEED certification (prefer official USGBC/LEED listings or hotel sustainability pages).

    Energy feature(s):
    Extract any major energy-efficient system(s) claimed. For each feature, create an object:
      - kind: One of ["LED", "Solar", "HVAC", "Other"] based on the description.
      - description: Short text summarizing what the answer claims (e.g., "LED lighting throughout", "rooftop solar panels", "HVAC system with 45% energy savings").
      - savings_percent: The percent savings (if present in the answer) as text, e.g., "40%", "45 percent", ">= 40%"; otherwise null.
      - urls: All URLs in the answer that support this energy feature.
    Put all such features into energy_features array.

    Water feature(s):
    Extract any water conservation system(s) claimed. For each feature, create an object:
      - kind: e.g., "Low-flow fixtures", "Rainwater harvesting", "Greywater", "Other".
      - description: Short text summarizing the claim.
      - reduction_percent: The percent water-use reduction if provided (e.g., "20%", "30 percent"); otherwise null.
      - urls: All URLs in the answer that support this water feature.
    Put all such features into water_features array.

    Corporate events:
    - conference_urls: All URLs that show the hotel has conference/meeting facilities (e.g., meetings page).
    - green_meeting_urls: All URLs that show the hotel offers green meeting or sustainable event planning services.

    IMPORTANT:
    - Only include URLs that are explicitly present in the answer. Do not invent or infer any URLs.
    - If the answer does not include a particular type of URL, return an empty list for that field.
    - Keep all number-like values (rooms_count, leed_points, savings/reduction percents) as strings; do NOT coerce to numeric.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(*url_lists: List[str]) -> List[str]:
    """Flatten, deduplicate, and clean URL lists."""
    seen = set()
    out: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not u:
                continue
            u = u.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def parse_first_percent(text: Optional[str]) -> Optional[float]:
    """Extract first percentage number from a string like '>= 40%' or '45 percent'."""
    if not text:
        return None
    # Look for numbers possibly followed by % or the word percent
    m = re.search(r"(?P<num>\d+(?:\.\d+)?)\s*(?:%|percent)", text.lower())
    if m:
        try:
            return float(m.group("num"))
        except Exception:
            return None
    # As a fallback if only a plain number is present
    m2 = re.search(r"(?P<num>\d+(?:\.\d+)?)", text.lower())
    if m2:
        try:
            return float(m2.group("num"))
        except Exception:
            return None
    return None


def pick_energy_feature(features: List[EnergyFeature]) -> Optional[EnergyFeature]:
    """
    Choose the best energy feature to verify:
    - Prefer HVAC with >= 40% savings (strict requirement if HVAC is used).
    - Else prefer Solar.
    - Else prefer LED.
    - Else any available.
    """
    if not features:
        return None

    hvac_candidates = []
    solar_candidates = []
    led_candidates = []
    other_candidates = []

    for f in features:
        kind = (f.kind or "").strip().lower()
        if kind == "hvac":
            pct = parse_first_percent(f.savings_percent)
            if pct is not None and pct >= 40.0:
                hvac_candidates.append(f)
        elif kind == "solar":
            solar_candidates.append(f)
        elif kind == "led":
            led_candidates.append(f)
        else:
            other_candidates.append(f)

    if hvac_candidates:
        return hvac_candidates[0]
    if solar_candidates:
        return solar_candidates[0]
    if led_candidates:
        return led_candidates[0]
    # If only HVAC below threshold or other: still return first other, caller will decide how to claim/verify
    # but will likely fail strict threshold; better to return first feature at least.
    return other_candidates[0] if other_candidates else features[0]


def pick_water_feature(features: List[WaterFeature]) -> Optional[WaterFeature]:
    """
    Choose a water feature achieving >= 20% reduction if available; otherwise None.
    """
    if not features:
        return None
    valid = []
    for f in features:
        pct = parse_first_percent(f.reduction_percent)
        if pct is not None and pct >= 20.0:
            valid.append(f)
    if valid:
        return valid[0]
    # If none meet 20%, return None to enforce requirement fail
    return None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_hotel_tree(evaluator: Evaluator, root: VerificationNode, extracted: HotelExtraction) -> None:
    """
    Build the rubric tree and run all verifications according to the rubric.
    """
    # Create the critical task node under evaluator root
    task_node = evaluator.add_parallel(
        id="Hotel_Selection_Task",
        desc="Identify a currently operational California beachfront hotel (<250 rooms) with LEED Gold certification "
             "and required sustainability + corporate event capabilities, and provide verification for the required claims.",
        parent=root,
        critical=True
    )

    # 1) Hotel_Name_Provided (custom existence check)
    name_exists = bool(extracted and extracted.name and extracted.name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id="Hotel_Name_Provided",
        desc="Provides a specific hotel name (single property) as the proposed answer.",
        parent=task_node,
        critical=True
    )

    hotel_name = extracted.name or "the hotel"

    # Prepare general URLs to help with generic checks where applicable
    general_urls = _safe_urls(
        [extracted.official_url] if extracted and extracted.official_url else [],
        extracted.location_urls,
        extracted.beachfront_urls,
        extracted.rooms_urls,
        extracted.operational_urls,
    )

    # 2) California_Location (verify with URLs if available; otherwise simple)
    location_node = evaluator.add_leaf(
        id="California_Location",
        desc="The hotel is located in California.",
        parent=task_node,
        critical=True
    )
    loc_claim = f"The hotel '{hotel_name}' is located in California (CA)."
    await evaluator.verify(
        claim=loc_claim,
        node=location_node,
        sources=_safe_urls(extracted.location_urls) or None,
        additional_instruction="Verify the address shows the state as California (CA). Accept 'CA' or 'California'. Prefer the hotel's official contact/location page or a credible directory cited in the answer."
    )

    # 3) Beachfront_Position (verify)
    beachfront_node = evaluator.add_leaf(
        id="Beachfront_Position",
        desc="The hotel is directly on the beach or oceanfront.",
        parent=task_node,
        critical=True
    )
    beachfront_claim = f"The hotel '{hotel_name}' is directly on the beach or oceanfront (i.e., beachfront or oceanfront)."
    await evaluator.verify(
        claim=beachfront_claim,
        node=beachfront_node,
        sources=_safe_urls(extracted.beachfront_urls) or None,
        additional_instruction="Accept synonyms like 'beachfront', 'oceanfront', or 'on the beach'. Do not accept 'near the beach' if it implies not directly beachfront."
    )

    # 4) Room_Count_Under_250 (verify)
    rooms_node = evaluator.add_leaf(
        id="Room_Count_Under_250",
        desc="The hotel has fewer than 250 rooms.",
        parent=task_node,
        critical=True
    )
    rooms_claim = f"The hotel '{hotel_name}' has fewer than 250 rooms."
    await evaluator.verify(
        claim=rooms_claim,
        node=rooms_node,
        sources=_safe_urls(extracted.rooms_urls) or None,
        additional_instruction="Verify the page states a room/keys count that is < 250. Accept terms like 'rooms', 'guest rooms', 'keys', or 'rooms & suites'."
    )

    # 5) Operational_Status (verify)
    operational_node = evaluator.add_leaf(
        id="Operational_Status",
        desc="The hotel is currently operational and accepting reservations.",
        parent=task_node,
        critical=True
    )
    operational_claim = f"The hotel '{hotel_name}' is currently operational and accepting reservations."
    await evaluator.verify(
        claim=operational_claim,
        node=operational_node,
        sources=_safe_urls(extracted.operational_urls) or None,
        additional_instruction="Look for an official booking/reservations/availability or 'Book Now' page that is functional. Third-party booking engines are acceptable if explicitly cited in the answer."
    )

    # 6) LEED_Gold_Certification_Verified (verify via URLs; fail immediately if no LEED URLs)
    leed_node = evaluator.add_leaf(
        id="LEED_Gold_Certification_Verified",
        desc="Provides verification that the hotel holds LEED Gold certification (60–79 points), using official certification documentation or a credible third-party source (with a citation/link).",
        parent=task_node,
        critical=True
    )
    leed_urls = _safe_urls(extracted.leed_urls)
    if not leed_urls:
        # No cited source in the answer; fail this critical verification directly
        leed_node.score = 0.0
        leed_node.status = "failed"
    else:
        # Build claim focused on LEED Gold (points range added as context)
        leed_level_text = (extracted.leed_level or "").strip().lower()
        # We explicitly require Gold, so craft the claim accordingly
        leed_claim = f"The hotel '{hotel_name}' holds LEED Gold certification (Gold is awarded for earning between 60 and 79 points on the LEED scorecard)."
        await evaluator.verify(
            claim=leed_claim,
            node=leed_node,
            sources=leed_urls,
            additional_instruction="Prefer USGBC (LEED) official listing or the hotel's official sustainability/certification page. Accept 'LEED Gold' phrasing. The exact points may not be shown; focus on verifying the Gold level."
        )

    # 7) Energy_Efficiency_Requirement_Met (verify at least one major energy feature)
    energy_node = evaluator.add_leaf(
        id="Energy_Efficiency_Requirement_Met",
        desc="Describes and verifies (with a credible source citation/link) at least one major energy-efficient system meeting the allowed examples: LED lighting throughout, OR solar panels/solar generation, OR HVAC providing at least 40% energy savings (must explicitly meet the ≥40% threshold if HVAC is used).",
        parent=task_node,
        critical=True
    )
    chosen_energy = pick_energy_feature(extracted.energy_features if extracted else [])
    if not chosen_energy or not _safe_urls(chosen_energy.urls):
        # No qualifying feature or no URLs cited — fail
        energy_node.score = 0.0
        energy_node.status = "failed"
    else:
        kind = (chosen_energy.kind or "").strip().lower()
        desc = chosen_energy.description or ""
        energy_urls = _safe_urls(chosen_energy.urls)

        if kind == "hvac":
            pct = parse_first_percent(chosen_energy.savings_percent)
            if pct is None or pct < 40.0:
                # Does not meet the strict requirement for HVAC
                energy_node.score = 0.0
                energy_node.status = "failed"
            else:
                claim = f"The hotel '{hotel_name}' implements a major energy-efficient HVAC system achieving at least 40% energy savings. Details: {desc}"
                await evaluator.verify(
                    claim=claim,
                    node=energy_node,
                    sources=energy_urls,
                    additional_instruction="Verify the cited page states an HVAC upgrade/system with ≥40% energy savings."
                )
        elif kind == "solar":
            claim = f"The hotel '{hotel_name}' implements a major energy-efficient system: solar panels/solar energy generation. Details: {desc}"
            await evaluator.verify(
                claim=claim,
                node=energy_node,
                sources=energy_urls,
                additional_instruction="Verify the presence of onsite solar panels or solar energy generation described on the cited page."
            )
        elif kind == "led":
            claim = f"The hotel '{hotel_name}' implements a major energy-efficient system: LED lighting (ideally throughout the property). Details: {desc}"
            await evaluator.verify(
                claim=claim,
                node=energy_node,
                sources=energy_urls,
                additional_instruction="Verify the cited page states LED lighting initiatives. If 'throughout' is not explicit, allow property-wide or pervasive LED upgrades when clearly implied by the source."
            )
        else:
            # Other energy feature: we cannot guarantee it qualifies; still attempt, but may fail
            claim = f"The hotel '{hotel_name}' implements a major energy-efficient system. Details: {desc}"
            await evaluator.verify(
                claim=claim,
                node=energy_node,
                sources=energy_urls,
                additional_instruction="Verify that the described system meaningfully reduces energy consumption comparable to the allowed examples (LED, solar, or ≥40% HVAC savings). If the feature is minor or unclear, mark as unsupported."
            )

    # 8) Water_Conservation_Requirement_Met (verify >=20% reduction and system present)
    water_node = evaluator.add_leaf(
        id="Water_Conservation_Requirement_Met",
        desc="Describes and verifies (with a credible source citation/link) at least one water conservation system (e.g., rainwater harvesting, low-flow fixtures, greywater recycling) and includes documented achievement (or documented capability) of at least 20% water usage reduction.",
        parent=task_node,
        critical=True
    )
    chosen_water = pick_water_feature(extracted.water_features if extracted else [])
    if not chosen_water or not _safe_urls(chosen_water.urls):
        water_node.score = 0.0
        water_node.status = "failed"
    else:
        water_urls = _safe_urls(chosen_water.urls)
        desc = chosen_water.description or ""
        pct = parse_first_percent(chosen_water.reduction_percent)
        if pct is None or pct < 20.0:
            # Even if selected by picker, guard again
            water_node.score = 0.0
            water_node.status = "failed"
        else:
            claim = f"The hotel '{hotel_name}' implements a water conservation system achieving at least 20% water-use reduction. Details: {desc}"
            await evaluator.verify(
                claim=claim,
                node=water_node,
                sources=water_urls,
                additional_instruction="Verify the cited page explicitly states a ≥20% reduction (or a number ≥20%). Accept features like low‑flow fixtures, rainwater harvesting, or greywater recycling if the reduction is clearly stated."
            )

    # 9) Corporate_Event_Capabilities (parallel critical parent)
    corp_node = evaluator.add_parallel(
        id="Corporate_Event_Capabilities",
        desc="Verifies the hotel can host corporate events and offers sustainability-focused meeting options (with citations/links).",
        parent=task_node,
        critical=True
    )

    # 9.a) Conference_Facilities_Available
    conf_node = evaluator.add_leaf(
        id="Conference_Facilities_Available",
        desc="The hotel has conference/meeting facilities (verified with a citation/link).",
        parent=corp_node,
        critical=True
    )
    conf_urls = _safe_urls(extracted.conference_urls if extracted else [])
    if not conf_urls:
        conf_node.score = 0.0
        conf_node.status = "failed"
    else:
        conf_claim = f"The hotel '{hotel_name}' has conference/meeting facilities suitable for corporate events."
        await evaluator.verify(
            claim=conf_claim,
            node=conf_node,
            sources=conf_urls,
            additional_instruction="Look for official 'Meetings', 'Events', or 'Conference' pages describing meeting rooms, ballrooms, or event spaces."
        )

    # 9.b) Green_Meeting_Services_Available
    green_node = evaluator.add_leaf(
        id="Green_Meeting_Services_Available",
        desc="The hotel offers green meeting or sustainable event planning services (verified with a citation/link).",
        parent=corp_node,
        critical=True
    )
    green_urls = _safe_urls(extracted.green_meeting_urls if extracted else [])
    if not green_urls:
        green_node.score = 0.0
        green_node.status = "failed"
    else:
        green_claim = f"The hotel '{hotel_name}' offers green meeting or sustainable event planning services."
        await evaluator.verify(
            claim=green_claim,
            node=green_node,
            sources=green_urls,
            additional_instruction="Verify language such as 'green meetings', 'sustainable events', 'eco-friendly meetings', or sustainability certifications/policies specifically for meetings/events."
        )

    # Record some auxiliary info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "selected_hotel_name": extracted.name,
            "official_url": extracted.official_url,
            "chosen_energy_feature": (chosen_energy.dict() if chosen_energy else None),
            "chosen_water_feature": (chosen_water.dict() if chosen_water else None),
        },
        info_type="debug",
        info_name="selection_and_features"
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
    Evaluate an answer for the sustainable beachfront hotel in California task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator; task node will be critical parallel
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

    # Extract the single hotel candidate and related evidence URLs from the answer
    extracted: HotelExtraction = await evaluator.extract(
        prompt=prompt_extract_hotel_candidate(),
        template_class=HotelExtraction,
        extraction_name="hotel_candidate_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_hotel_tree(evaluator, root, extracted)

    # Return summary with verification tree and metadata
    return evaluator.get_summary()