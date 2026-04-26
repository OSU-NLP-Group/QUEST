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
TASK_ID = "fs_koh_samui_retreat_verification"
TASK_DESCRIPTION = """
A corporate event planning team is organizing a 5-day executive retreat in Thailand for June 2026 and needs to verify comprehensive details about the Four Seasons Resort Koh Samui, which was featured as the primary filming location for HBO's "The White Lotus" Season 3. The retreat will host 6 executives requiring luxury accommodations with privacy and personalized service.

Provide the following verified information about Four Seasons Resort Koh Samui:

1. Property Identification & Location
   - The resort's complete official name
   - The full street address including the Moo number, sub-district, district, and province
   - The complete Thailand postal code

2. Direct Booking Contact
   - The resort's direct reservation telephone number (including country code)
   - The reservations email address

3. Multi-Bedroom Accommodation
   - Confirmation that the resort offers Two-Bedroom Residence Villas
   - The guest capacity range for Two-Bedroom Residence Villas (minimum and maximum guests)
   - Confirmation that Two-Bedroom Residence Villas include a private infinity pool
   - Confirmation that Two-Bedroom Residence Villas include personalized residential assistant service

4. Signature Experiences
   - The name of the on-site spa facility
   - Confirmation that the resort offers Angthong National Marine Park boat excursions
   - The approximate duration (in hours) of the Angthong Marine Park excursion

5. Travel Logistics to Thailand
   - Identify a Mexican city served by Volaris that launched new nonstop service to Orlando (MCO) in June 2026, which corporate travelers could use as a connection point
   - The three-letter airport code for this Mexican city
   - The days of the week this Volaris flight operates

All information must be directly verifiable from official resort websites, press releases, or authoritative hospitality sources.
"""

# Ground truth targets (used for claim construction)
GROUND_TRUTH = {
    "official_name": "Four Seasons Resort Koh Samui",
    "full_address": "219 Moo 5, Angthong, Koh Samui, Surat Thani",
    "postal_code": "84140",
    "phone_number": "+66 77 243-000",
    "email_address": "reservations.thailand@fourseasons.com",
    "villa_type": "Two-Bedroom Residence Villas",
    "guest_capacity_range_text": "4-6 guests",
    "spa_name": "Secret Garden Spa",
    "marine_park": "Angthong National Marine Park",
    "excursion_duration_hours": "7 hours",
    "mexican_city": "Querétaro",
    "airport_code": "QRO",
    "operating_days_text": "Mondays and Fridays",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResortProperty(BaseModel):
    official_name: Optional[str] = None
    full_address: Optional[str] = None
    postal_code: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class BookingContact(BaseModel):
    phone_number: Optional[str] = None
    email_address: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class AccommodationDetails(BaseModel):
    villa_type_mentioned: Optional[str] = None
    two_bedroom_residence_villas_confirmed: Optional[bool] = None
    guest_capacity_range_text: Optional[str] = None
    private_infinity_pool_confirmed: Optional[bool] = None
    assistant_service_confirmed: Optional[bool] = None
    reference_urls: List[str] = Field(default_factory=list)


class SignatureExperiences(BaseModel):
    spa_name: Optional[str] = None
    marine_park_excursion_confirmed: Optional[bool] = None
    excursion_duration_hours: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class TravelLogistics(BaseModel):
    mexican_city: Optional[str] = None
    airport_code: Optional[str] = None
    operating_days: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class AllExtraction(BaseModel):
    property: Optional[ResortProperty] = None
    booking: Optional[BookingContact] = None
    accommodation: Optional[AccommodationDetails] = None
    experiences: Optional[SignatureExperiences] = None
    travel: Optional[TravelLogistics] = None
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the specific structured information about Four Seasons Resort Koh Samui and related travel logistics exactly as stated in the answer.

    Return a JSON object with the following nested fields:

    property:
      - official_name: the resort's complete official name (e.g., "Four Seasons Resort Koh Samui")
      - full_address: the full street address including Moo number, sub-district, district, and province (exclude the postal code here)
      - postal_code: the Thailand postal code for the resort
      - reference_urls: a list of URLs in the answer that directly support property identification or address (prefer fourseasons.com domain)

    booking:
      - phone_number: direct reservation phone number with country code (e.g., "+66 77 243-000")
      - email_address: reservations email (e.g., "reservations.thailand@fourseasons.com")
      - reference_urls: URLs that explicitly show reservation phone/email for the Koh Samui resort

    accommodation:
      - villa_type_mentioned: the exact villa type term mentioned (e.g., "Two-Bedroom Residence Villas")
      - two_bedroom_residence_villas_confirmed: true/false if the answer explicitly confirms availability
      - guest_capacity_range_text: the stated capacity range text for Two-Bedroom Residence Villas (e.g., "4-6")
      - private_infinity_pool_confirmed: true/false if explicitly stated
      - assistant_service_confirmed: true/false if explicitly stated
      - reference_urls: URLs that describe the Two-Bedroom Residence Villas features/specs

    experiences:
      - spa_name: the on-site spa facility name (e.g., "Secret Garden Spa")
      - marine_park_excursion_confirmed: true/false if Angthong National Marine Park boat excursions are offered
      - excursion_duration_hours: the stated duration (e.g., "7" or "7 hours")
      - reference_urls: URLs that describe spa and boat excursions (prefer fourseasons.com domain)

    travel:
      - mexican_city: the Mexican city served by Volaris launching new non-stop service to Orlando (MCO) in June 2026
      - airport_code: the city's three-letter airport code (e.g., "QRO")
      - operating_days: list of operating days (e.g., ["Monday", "Friday"])
      - reference_urls: URLs from Volaris or authoritative aviation sources confirming route details

    all_urls:
      - include every URL present in the answer text (including markdown links). If none, return an empty list.

    Rules:
    - Extract only what is explicitly present in the answer; do not invent.
    - For URL fields, include full valid URLs; accept both plain and markdown links.
    - If any field is missing, set it to null (or empty list for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        url = u.strip()
        if url and url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _filter_by_domains(urls: List[str], domains: List[str]) -> List[str]:
    urls = _dedup(urls)
    return [u for u in urls if any(d in u for d in domains)]


def _section_urls(extracted: AllExtraction) -> Dict[str, List[str]]:
    property_urls = extracted.property.reference_urls if extracted.property else []
    booking_urls = extracted.booking.reference_urls if extracted.booking else []
    accommodation_urls = extracted.accommodation.reference_urls if extracted.accommodation else []
    experiences_urls = extracted.experiences.reference_urls if extracted.experiences else []
    travel_urls = extracted.travel.reference_urls if extracted.travel else []
    all_urls = extracted.all_urls if extracted.all_urls else []
    return {
        "property": _dedup(property_urls),
        "booking": _dedup(booking_urls),
        "accommodation": _dedup(accommodation_urls),
        "experiences": _dedup(experiences_urls),
        "travel": _dedup(travel_urls),
        "all": _dedup(all_urls),
    }


def _fs_urls(extracted: AllExtraction) -> List[str]:
    urls = []
    sections = _section_urls(extracted)
    urls.extend(sections["property"])
    urls.extend(sections["booking"])
    urls.extend(sections["accommodation"])
    urls.extend(sections["experiences"])
    urls.extend(sections["all"])
    return _filter_by_domains(urls, ["fourseasons.com"])


def _accommodation_urls(extracted: AllExtraction) -> List[str]:
    sections = _section_urls(extracted)
    urls = sections["accommodation"] or _fs_urls(extracted)
    return _filter_by_domains(urls, ["fourseasons.com"])


def _contact_urls(extracted: AllExtraction) -> List[str]:
    sections = _section_urls(extracted)
    urls = sections["booking"] or _fs_urls(extracted)
    return _filter_by_domains(urls, ["fourseasons.com"])


def _property_urls(extracted: AllExtraction) -> List[str]:
    sections = _section_urls(extracted)
    urls = sections["property"] or _fs_urls(extracted)
    return _filter_by_domains(urls, ["fourseasons.com"])


def _experience_urls(extracted: AllExtraction) -> List[str]:
    sections = _section_urls(extracted)
    urls = sections["experiences"] or _fs_urls(extracted)
    return _filter_by_domains(urls, ["fourseasons.com"])


def _flight_urls(extracted: AllExtraction) -> List[str]:
    sections = _section_urls(extracted)
    urls = sections["travel"]
    if not urls:
        urls = sections["all"]
    return _filter_by_domains(urls, ["volaris.com", "orlandoairports.net", "routesonline.com", "aeroroutes.com", "simpleflying.com"])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_property_identification(evaluator: Evaluator, parent_node, extracted: AllExtraction) -> None:
    node = evaluator.add_parallel(
        id="property_identification",
        desc="Accurate identification and location details of the resort",
        parent=parent_node,
        critical=True,
    )

    sources_fs = _property_urls(extracted) or _fs_urls(extracted)

    # Official name
    official_name_leaf = evaluator.add_leaf(
        id="official_name",
        desc="Provides the complete official resort name: 'Four Seasons Resort Koh Samui'",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The resort's official name is '{GROUND_TRUTH['official_name']}'.",
        node=official_name_leaf,
        sources=sources_fs,
        additional_instruction="Verify that the page clearly shows the property's official name; allow minor variations such as inclusion of 'Thailand' suffix.",
    )

    # Full address (without postal code)
    full_address_leaf = evaluator.add_leaf(
        id="full_address",
        desc="Provides the complete street address: 219 Moo 5, Angthong, Koh Samui, Surat Thani",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The resort's full street address is '{GROUND_TRUTH['full_address']}'.",
        node=full_address_leaf,
        sources=sources_fs,
        additional_instruction="Confirm the address components (Moo number, sub-district Angthong, district Koh Samui, province Surat Thani). Ignore postal code here.",
    )

    # Postal code
    postal_code_leaf = evaluator.add_leaf(
        id="postal_code",
        desc="Provides the correct Thailand postal code: 84140",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Thailand postal code for the resort is {GROUND_TRUTH['postal_code']}.",
        node=postal_code_leaf,
        sources=sources_fs,
        additional_instruction="Verify the postal code adjacent to the address on official pages.",
    )

    # Reference URL from FS site confirming property details
    ref_prop_leaf = evaluator.add_leaf(
        id="reference_url_property",
        desc="Provides a valid reference URL from the Four Seasons official website confirming the property details",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This Four Seasons webpage confirms the property's official name and address for Four Seasons Resort Koh Samui.",
        node=ref_prop_leaf,
        sources=sources_fs,
        additional_instruction="Ensure the URL belongs to fourseasons.com and the page explicitly shows the resort name and address.",
    )


async def verify_booking_contact(evaluator: Evaluator, parent_node, extracted: AllExtraction) -> None:
    node = evaluator.add_parallel(
        id="booking_contact",
        desc="Correct direct booking contact information",
        parent=parent_node,
        critical=True,
    )

    sources_contact = _contact_urls(extracted) or _fs_urls(extracted)

    # Phone
    phone_leaf = evaluator.add_leaf(
        id="phone_number",
        desc="Provides the correct direct reservation phone number with country code: +66 77 243-000",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The direct reservation phone number for Four Seasons Resort Koh Samui is '{GROUND_TRUTH['phone_number']}' (allow spaces or hyphen formatting).",
        node=phone_leaf,
        sources=sources_contact,
        additional_instruction="Treat formatting variations equivalently, e.g., '+66 77 243 000' vs '+66 77 243-000'. Confirm it's for the Koh Samui resort.",
    )

    # Email
    email_leaf = evaluator.add_leaf(
        id="email_address",
        desc="Provides the correct reservations email: reservations.thailand@fourseasons.com",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The reservations email address is '{GROUND_TRUTH['email_address']}'.",
        node=email_leaf,
        sources=sources_contact,
        additional_instruction="Confirm the email appears on official Four Seasons pages for Thailand reservations applicable to Koh Samui.",
    )

    # Reference URL
    ref_contact_leaf = evaluator.add_leaf(
        id="reference_url_contact",
        desc="Provides a valid reference URL from the Four Seasons official website confirming the contact information",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This Four Seasons webpage provides the reservations contact information (phone and/or email) for the Koh Samui resort.",
        node=ref_contact_leaf,
        sources=sources_contact,
        additional_instruction="Ensure the URL is on fourseasons.com and includes phone/email relevant to Koh Samui.",
    )


async def verify_accommodation_details(evaluator: Evaluator, parent_node, extracted: AllExtraction) -> None:
    node = evaluator.add_parallel(
        id="accommodation_details",
        desc="Accurate details about Two-Bedroom Residence Villas",
        parent=parent_node,
        critical=True,
    )

    sources_accom = _accommodation_urls(extracted)

    # Villa type exists
    villa_exists_leaf = evaluator.add_leaf(
        id="villa_type_exists",
        desc="Confirms that Two-Bedroom Residence Villas are available at the property",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The resort offers {GROUND_TRUTH['villa_type']}.",
        node=villa_exists_leaf,
        sources=sources_accom,
        additional_instruction="Confirm that the Two-Bedroom Residence Villas are explicitly listed/available at the Koh Samui property.",
    )

    # Guest capacity
    capacity_leaf = evaluator.add_leaf(
        id="guest_capacity",
        desc="Confirms the villa accommodates 4-6 guests",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Two-Bedroom Residence Villas accommodate between 4 and 6 guests.",
        node=capacity_leaf,
        sources=sources_accom,
        additional_instruction="Verify occupancy or maximum guests information; 'sleeps up to 6' or details like '4 adults + 2 children' are acceptable evidence.",
    )

    # Infinity pool
    infinity_pool_leaf = evaluator.add_leaf(
        id="infinity_pool",
        desc="Confirms that Two-Bedroom Residence Villas include a private infinity pool",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Two-Bedroom Residence Villas include a private infinity pool.",
        node=infinity_pool_leaf,
        sources=sources_accom,
        additional_instruction="Look for villa features listing 'private infinity pool' or equivalent wording.",
    )

    # Assistant service
    assistant_leaf = evaluator.add_leaf(
        id="assistant_service",
        desc="Confirms that Two-Bedroom Residence Villas include personalized residential assistant service",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Two-Bedroom Residence Villas include personalized residential assistant service.",
        node=assistant_leaf,
        sources=sources_accom,
        additional_instruction="The service may be phrased as 'Residential Assistant,' 'butler,' or similar. Confirm that personalized assistance is included for Residence Villas.",
    )

    # Reference URL
    ref_accom_leaf = evaluator.add_leaf(
        id="reference_url_accommodation",
        desc="Provides a valid reference URL from the Four Seasons official website confirming the villa specifications",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This official Four Seasons page details the Two-Bedroom Residence Villa features such as occupancy, private infinity pool, and assistant service.",
        node=ref_accom_leaf,
        sources=sources_accom,
        additional_instruction="Ensure the URL belongs to fourseasons.com and specifically describes Two-Bedroom Residence Villas at Koh Samui.",
    )


async def verify_signature_experiences(evaluator: Evaluator, parent_node, extracted: AllExtraction) -> None:
    node = evaluator.add_parallel(
        id="signature_experiences",
        desc="Accurate information about resort experiences and amenities",
        parent=parent_node,
        critical=False,
    )

    sources_exp = _experience_urls(extracted) or _fs_urls(extracted)

    # Spa name
    spa_leaf = evaluator.add_leaf(
        id="spa_name",
        desc="Identifies the on-site spa as 'Secret Garden Spa'",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The on-site spa facility is called '{GROUND_TRUTH['spa_name']}'.",
        node=spa_leaf,
        sources=sources_exp,
        additional_instruction="Allow minor variants like 'The Secret Garden Spa'; verify that the spa name corresponds to the Koh Samui property.",
    )

    # Marine park excursion
    marine_leaf = evaluator.add_leaf(
        id="marine_park_excursion",
        desc="Confirms that Angthong National Marine Park boat excursions are offered",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The resort offers boat excursions to {GROUND_TRUTH['marine_park']}.",
        node=marine_leaf,
        sources=sources_exp,
        additional_instruction="Look for experience pages mentioning private boat excursions to Angthong National Marine Park.",
    )

    # Excursion duration
    duration_leaf = evaluator.add_leaf(
        id="excursion_duration",
        desc="Provides the correct excursion duration: 7 hours",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim="The Angthong Marine Park boat excursion duration is approximately 7 hours.",
        node=duration_leaf,
        sources=sources_exp,
        additional_instruction="Duration may be phrased as 'around 7 hours'; minor wording differences acceptable.",
    )

    # Reference URL (critical under experiences group)
    ref_exp_leaf = evaluator.add_leaf(
        id="reference_url_experiences",
        desc="Provides a valid reference URL from the Four Seasons official website confirming the experiences offered",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This official Four Seasons page confirms resort experiences (spa and Angthong Marine Park excursions), including the approximate excursion duration.",
        node=ref_exp_leaf,
        sources=sources_exp,
        additional_instruction="Ensure the URL belongs to fourseasons.com and references the Koh Samui property.",
    )


async def verify_travel_logistics(evaluator: Evaluator, parent_node, extracted: AllExtraction) -> None:
    node = evaluator.add_parallel(
        id="travel_logistics",
        desc="Accurate information about Volaris Mexico-Orlando connection option",
        parent=parent_node,
        critical=False,
    )

    sources_flight = _flight_urls(extracted)

    # Mexican city
    city_leaf = evaluator.add_leaf(
        id="mexican_city",
        desc="Identifies Querétaro as the Mexican city with new Volaris service to Orlando starting June 2026",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim="In June 2026, Volaris launched a new nonstop route to Orlando (MCO) from Querétaro.",
        node=city_leaf,
        sources=sources_flight,
        additional_instruction="Confirm the origin city is Querétaro for the Volaris MCO route starting June 2026.",
    )

    # Airport code
    code_leaf = evaluator.add_leaf(
        id="airport_code",
        desc="Provides the correct airport code: QRO",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim="The IATA airport code for Querétaro is QRO.",
        node=code_leaf,
        sources=sources_flight,
        additional_instruction="Accept authoritative aviation/airport sources or official announcements.",
    )

    # Operating days
    days_leaf = evaluator.add_leaf(
        id="operating_days",
        desc="Correctly identifies the operating days as Mondays and Fridays",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim="The Volaris Querétaro–Orlando (MCO) flight operates on Mondays and Fridays.",
        node=days_leaf,
        sources=sources_flight,
        additional_instruction="Verify the specific operating days stated in official or credible route announcements.",
    )

    # Reference URL (critical within travel group)
    ref_flight_leaf = evaluator.add_leaf(
        id="reference_url_flight",
        desc="Provides a valid reference URL from Volaris or official aviation source confirming the route details",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This official or authoritative aviation source confirms Volaris' Querétaro–Orlando (MCO) route starting June 2026 and its operating days.",
        node=ref_flight_leaf,
        sources=sources_flight,
        additional_instruction="Prefer volaris.com or official airport/route announcement domains; the page must mention QRO–MCO with start date and/or schedule details.",
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
    Evaluate the answer for Four Seasons Resort Koh Samui verification and travel logistics.
    """
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

    # Extract all structured info
    extracted: AllExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllExtraction,
        extraction_name="extracted_info_all",
    )

    # Add ground truth for transparency
    evaluator.add_ground_truth({
        "expected": GROUND_TRUTH,
        "notes": "Values represent the canonical items to verify against official sources."
    })

    # Build and run verifications for each rubric section
    await verify_property_identification(evaluator, root, extracted)
    await verify_booking_contact(evaluator, root, extracted)
    await verify_accommodation_details(evaluator, root, extracted)
    await verify_signature_experiences(evaluator, root, extracted)
    await verify_travel_logistics(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()