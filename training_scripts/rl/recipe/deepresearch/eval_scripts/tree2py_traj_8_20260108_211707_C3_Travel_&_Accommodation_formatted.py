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
TASK_ID = "chicago_leed_gold_hotel"
TASK_DESCRIPTION = (
    "I am planning a corporate business conference in Chicago and need to select a hotel that demonstrates strong "
    "environmental sustainability credentials. Identify one hotel located in downtown Chicago that holds LEED Gold "
    "certification from the U.S. Green Building Council. The hotel must provide all of the following amenities to "
    "accommodate our conference needs: on-site conference facilities or meeting rooms, a fitness center or gym for "
    "attendees, and at least one on-site restaurant for dining convenience. Please provide the following information "
    "for the hotel: (1) Hotel name, (2) Complete physical street address, (3) A direct link to the hotel's official "
    "website, and (4) A link to verify the hotel's LEED Gold certification from the USGBC project directory or the "
    "hotel's official documentation."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelExtraction(BaseModel):
    """
    Structured information for a single hotel as presented in the answer.
    If the answer lists multiple hotels, extract the first/primary one only.
    """
    hotel_name: Optional[str] = None
    address: Optional[str] = None
    official_website: Optional[str] = None

    # One or more URLs that substantiate LEED Gold status.
    # Accept USGBC project directory links or official hotel documentation pages (press releases, sustainability pages).
    leed_verification_urls: List[str] = Field(default_factory=list)

    # Optional amenity-specific URLs if provided by the answer (e.g., meeting/events page, fitness page, dining page)
    conference_urls: List[str] = Field(default_factory=list)
    fitness_urls: List[str] = Field(default_factory=list)
    restaurant_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_info() -> str:
    return """
    Extract details for the single hotel proposed in the answer (if multiple hotels are mentioned, extract the first/primary one only).
    Return a JSON object with the following fields:
    - hotel_name: The name of the hotel.
    - address: The complete physical street address of the hotel as stated in the answer (e.g., street number, street name, city, state, and postal code if available).
    - official_website: A direct URL to the hotel's official website (home page or official subpage).
    - leed_verification_urls: An array of URLs that verify the hotel's LEED Gold certification, preferably the USGBC project directory link; alternatively, the hotel's official LEED/certification page, press release, or sustainability page that explicitly states "LEED Gold".
    - conference_urls: An array of URLs for on-site meeting rooms or conference/event facilities (if included in the answer).
    - fitness_urls: An array of URLs for the on-site fitness center or gym (if included in the answer).
    - restaurant_urls: An array of URLs for on-site restaurants or dining options (if included in the answer).

    Rules:
    - Extract only URLs explicitly present in the answer; do not invent or infer URLs.
    - Include full URLs; if a URL lacks protocol, prepend http://.
    - If a field is not present in the answer, set it to null (for strings) or an empty list (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _combine_sources(*url_lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for urls in url_lists:
        for u in urls:
            if _nonempty_str(u):
                u_norm = u.strip()
                if u_norm not in seen:
                    seen.add(u_norm)
                    combined.append(u_norm)
    return combined


def _maybe_list(url: Optional[str]) -> List[str]:
    return [url] if _nonempty_str(url) else []


# --------------------------------------------------------------------------- #
# Tree construction + verification                                            #
# --------------------------------------------------------------------------- #
async def _build_and_verify_hotel_tree(evaluator: Evaluator, root, extracted: HotelExtraction) -> None:
    """
    Build verification tree according to the rubric and execute verifications.
    All intermediate existence checks are implemented as critical custom nodes.
    Evidence-backed checks use LLM verification (with URLs whenever possible).
    """

    # Root of rubric
    hotel_info_node = evaluator.add_parallel(
        id="Hotel_Information",
        desc="Answer identifies one qualifying downtown Chicago hotel and provides all required info, certification proof, location, and amenities.",
        parent=root,
        critical=True,
    )

    # ----------------------- Hotel Identity -----------------------
    identity_node = evaluator.add_parallel(
        id="Hotel_Identity",
        desc="Basic hotel identification information is provided.",
        parent=hotel_info_node,
        critical=True,
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=_nonempty_str(extracted.hotel_name),
        id="Hotel_Name",
        desc="Hotel name is provided.",
        parent=identity_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty_str(extracted.address),
        id="Physical_Address",
        desc="Complete physical street address is provided.",
        parent=identity_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty_str(extracted.official_website),
        id="Official_Website",
        desc="A direct link to the hotel's official website is provided.",
        parent=identity_node,
        critical=True,
    )

    # ----------------------- Certification ------------------------
    cert_node = evaluator.add_parallel(
        id="Certification_Verification",
        desc="Hotel's LEED Gold certification is satisfied and verifiable.",
        parent=hotel_info_node,
        critical=True,
    )

    # Presence of verification URL(s)
    evaluator.add_custom_node(
        result=len(extracted.leed_verification_urls) > 0,
        id="LEED_Verification_URL",
        desc="A link verifying LEED Gold certification is provided (USGBC project directory or official hotel documentation).",
        parent=cert_node,
        critical=True,
    )

    # LEED Gold status check (verified via provided URLs)
    leed_gold_leaf = evaluator.add_leaf(
        id="LEED_Gold_Status",
        desc="Hotel holds LEED Gold certification (60–79 points) from the U.S. Green Building Council.",
        parent=cert_node,
        critical=True,
    )

    # ----------------------- Location -----------------------------
    location_node = evaluator.add_parallel(
        id="Location_Verification",
        desc="Hotel location meets geographic requirements.",
        parent=hotel_info_node,
        critical=True,
    )

    chicago_loc_leaf = evaluator.add_leaf(
        id="Chicago_Location",
        desc="Hotel is located in Chicago.",
        parent=location_node,
        critical=True,
    )

    downtown_loc_leaf = evaluator.add_leaf(
        id="Downtown_Location",
        desc="Hotel is located in downtown Chicago or the central business district.",
        parent=location_node,
        critical=True,
    )

    # ----------------------- Amenities ----------------------------
    amenities_node = evaluator.add_parallel(
        id="Required_Amenities",
        desc="Hotel provides all required amenities for the conference needs.",
        parent=hotel_info_node,
        critical=True,
    )

    conference_leaf = evaluator.add_leaf(
        id="Conference_Facilities",
        desc="Hotel has on-site conference facilities or meeting rooms.",
        parent=amenities_node,
        critical=True,
    )

    fitness_leaf = evaluator.add_leaf(
        id="Fitness_Center",
        desc="Hotel has an on-site fitness center or gym.",
        parent=amenities_node,
        critical=True,
    )

    restaurant_leaf = evaluator.add_leaf(
        id="Onsite_Restaurant",
        desc="Hotel has at least one on-site restaurant or dining facility.",
        parent=amenities_node,
        critical=True,
    )

    # ----------------------- Prepare claims & sources --------------
    name = extracted.hotel_name or ""
    off_site = _maybe_list(extracted.official_website)
    leed_urls = extracted.leed_verification_urls
    conference_urls = extracted.conference_urls
    fitness_urls = extracted.fitness_urls
    restaurant_urls = extracted.restaurant_urls

    # Location sources: official website + LEED verification pages (often include city/state)
    loc_sources = _combine_sources(off_site, leed_urls)

    # Amenities sources: official website + amenity-specific URLs
    conference_sources = _combine_sources(off_site, conference_urls)
    fitness_sources = _combine_sources(off_site, fitness_urls)
    restaurant_sources = _combine_sources(off_site, restaurant_urls)

    # ----------------------- Execute verifications in a robust order ------
    # 1) Location: Chicago first, then Downtown (so Downtown can be skipped if Chicago fails)
    chicago_claim = f"The hotel named '{name}' is located in Chicago, Illinois."
    await evaluator.verify(
        claim=chicago_claim,
        node=chicago_loc_leaf,
        sources=loc_sources,
        additional_instruction=(
            "Verify that the page explicitly indicates the hotel's city as 'Chicago, IL' or 'Chicago, Illinois' "
            "(e.g., in the address, contact, or overview)."
        ),
    )

    downtown_claim = f"The hotel named '{name}' is located in downtown Chicago (the central business district)."
    await evaluator.verify(
        claim=downtown_claim,
        node=downtown_loc_leaf,
        sources=loc_sources,
        additional_instruction=(
            "Confirm that the webpage clearly states the hotel is 'downtown' or in Chicago's central business district. "
            "Treat neighborhoods commonly recognized as downtown Chicago as supportive evidence (e.g., The Loop, "
            "River North, Streeterville, Magnificent Mile, Near North Side, West Loop, South Loop)."
        ),
    )

    # 2) Amenities: conference facilities, fitness center, on-site restaurant
    conference_claim = f"The hotel named '{name}' has on-site conference facilities, event spaces, or meeting rooms."
    await evaluator.verify(
        claim=conference_claim,
        node=conference_leaf,
        sources=conference_sources,
        additional_instruction=(
            "Look for pages like 'Meetings', 'Events', 'Conference Center', 'Floor Plans', or 'Meeting Rooms' "
            "that indicate on-site facilities (not off-site partner venues)."
        ),
    )

    fitness_claim = f"The hotel named '{name}' has an on-site fitness center or gym."
    await evaluator.verify(
        claim=fitness_claim,
        node=fitness_leaf,
        sources=fitness_sources,
        additional_instruction=(
            "Look for mentions of 'Fitness Center', 'Gym', 'Health Club', or similar facilities clearly located on-site."
        ),
    )

    restaurant_claim = f"The hotel named '{name}' has at least one on-site restaurant or dining facility."
    await evaluator.verify(
        claim=restaurant_claim,
        node=restaurant_leaf,
        sources=restaurant_sources,
        additional_instruction=(
            "Confirm there is at least one on-site restaurant or dining venue (e.g., 'restaurant', 'grill', 'kitchen', "
            "'cafe') that serves meals. A bar alone without meals does not suffice."
        ),
    )

    # 3) Certification: LEED Gold status using verification URLs; this will be skipped automatically if
    #    the 'LEED_Verification_URL' existence check above failed (critical sibling precondition).
    leed_claim = f"The hotel named '{name}' holds LEED Gold certification from the U.S. Green Building Council."
    await evaluator.verify(
        claim=leed_claim,
        node=leed_gold_leaf,
        sources=leed_urls,
        additional_instruction=(
            "Rely on the provided LEED verification URLs only (USGBC project directory or the hotel's official "
            "documentation pages). The page must explicitly indicate 'LEED Gold' (not Silver, Certified, or Platinum). "
            "Ensure the certified project corresponds to the same hotel/property in Chicago."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Chicago LEED Gold hotel selection task.
    """
    # Initialize the evaluator
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

    # Extract structured hotel info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction",
    )

    # Build tree and run verifications
    await _build_and_verify_hotel_tree(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()