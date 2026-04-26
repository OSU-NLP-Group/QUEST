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
TASK_ID = "weekend_pet_trip_larimer"
TASK_DESCRIPTION = """
I am planning a weekend pet adoption trip in Larimer County, Colorado, and need to prepare a comprehensive resource guide. Please identify three types of facilities in Larimer County or nearby areas (within Fort Collins, Loveland, or surrounding communities):

1. An Animal Shelter that is open on Saturdays for pet adoptions. Provide:
   - The shelter's name and complete physical address
   - Saturday operating hours
   - Adoption fee information and what services are included
   - Minimum age requirement for adopters
   - Required documentation for adoption
   - Official website URL

2. A Dog Park or Off-Leash Dog Area in the same region. Provide:
   - The park's name and complete physical address
   - Description of park features (fenced areas, size, amenities)
   - Leash requirements and any entry restrictions
   - Whether proof of rabies vaccination is required
   - Operating hours or access information
   - An official reference URL

3. An Emergency Veterinary Clinic with weekend availability. Provide:
   - The clinic's name and complete physical address
   - Weekend operating hours (Saturday and Sunday)
   - Types of emergency services offered
   - Emergency contact phone number
   - Whether the clinic operates 24/7 or has limited hours
   - Official website URL

For each facility, ensure all information is current and verifiable through official sources or the facility's official website.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class AnimalShelter(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    saturday_hours: Optional[str] = None
    adoption_fee_info: Optional[str] = None  # include fee ranges + included services as a single text if needed
    age_requirement: Optional[str] = None
    required_documentation: Optional[str] = None
    website_url: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class DogPark(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    features: Optional[str] = None
    leash_requirements: Optional[str] = None
    rabies_requirement: Optional[str] = None
    hours_access: Optional[str] = None
    reference_url: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class EmergencyClinic(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    saturday_hours: Optional[str] = None
    sunday_hours: Optional[str] = None
    services_offered: Optional[str] = None
    phone: Optional[str] = None
    is_24_7: Optional[str] = None  # e.g., "24/7", "Yes", "No - limited hours", "Open nights/weekends"
    website_url: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class TripPlanExtraction(BaseModel):
    animal_shelter: Optional[AnimalShelter] = None
    dog_park: Optional[DogPark] = None
    emergency_clinic: Optional[EmergencyClinic] = None


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_plan() -> str:
    return """
    Extract the three requested facilities exactly as presented in the answer text. Return a JSON object with these top-level fields:
    - animal_shelter
    - dog_park
    - emergency_clinic

    Each field should itself be an object with the following fields (use null when not explicitly present):

    For animal_shelter:
      - name: Shelter name (string)
      - address: Complete physical address (string)
      - saturday_hours: Saturday operating or adoption hours (string, e.g., "Sat 10am–5pm", "By appointment", or "Closed")
      - adoption_fee_info: Exact fee info and what’s included (spay/neuter, microchip, vaccines) as a single text snippet (string)
      - age_requirement: Minimum age to adopt (string, e.g., "18+")
      - required_documentation: Required documents (ID, proof of residence, etc.) as a single text snippet (string)
      - website_url: The official shelter website URL as explicitly written in the answer (string or null)
      - source_urls: An array of any additional URLs cited for this shelter in the answer (exclude duplicates and invalid URLs)

    For dog_park:
      - name: Park name (string)
      - address: Complete physical address (string)
      - features: Key features (fenced, small/large areas, amenities) as a text snippet (string)
      - leash_requirements: Leash rules (string)
      - rabies_requirement: Whether proof of rabies vaccine is required; capture the exact phrasing if given (string)
      - hours_access: Operating hours or access information (string)
      - reference_url: The official or authoritative URL for the park as in the answer (string or null)
      - source_urls: An array of any additional URLs cited for this park in the answer

    For emergency_clinic:
      - name: Clinic name (string)
      - address: Complete physical address (string)
      - saturday_hours: Saturday hours (string)
      - sunday_hours: Sunday hours (string)
      - services_offered: Types of emergency services (string)
      - phone: Emergency contact phone number as presented (string)
      - is_24_7: A short descriptor indicating 24/7 vs limited hours (e.g., "24/7", "Yes", "No, open 8am–10pm") (string)
      - website_url: The official clinic website URL (string or null)
      - source_urls: An array of any additional URLs cited for this clinic in the answer

    Rules:
    - Do not invent any information. Only extract what is explicitly in the answer.
    - For URL fields, include only valid URLs that appear in the answer. If absent, set to null.
    - Prefer official websites when multiple URLs are present, but still list other URLs in source_urls.
    - Keep dates, times, fees, and policy language as text strings exactly or nearly as written.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def _merge_urls(primary: Optional[str], extras: Optional[List[str]]) -> List[str]:
    urls: List[str] = []
    if primary and primary.strip():
        urls.append(primary.strip())
    if extras:
        for e in extras:
            if e and e.strip() and e.strip() not in urls:
                urls.append(e.strip())
    return urls


def _blank_guard(val: Optional[str]) -> str:
    return (val or "").strip()


def _is_24_7_text(s: Optional[str]) -> Optional[bool]:
    """Interpret a free-text 24/7 indicator into a boolean if possible, else None."""
    if not s:
        return None
    low = s.lower()
    if "24/7" in low or "24x7" in low or "24-7" in low or "24 • 7" in low or "24 hours" in low or "24hrs" in low:
        return True
    if any(x in low for x in ["not 24", "limited", "8am", "9am", "10pm", "closed", "open weekends only"]):
        return False
    if low in ["yes", "y", "true"]:
        return True
    if low in ["no", "n", "false"]:
        return False
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_animal_shelter(evaluator: Evaluator, parent) -> None:
    """
    Build verification nodes and checks for the Animal Shelter facility.
    """
    node = evaluator.add_parallel(
        id="Animal_Shelter_Facility",
        desc="Identify an animal shelter that is open on Saturdays and suitable for pet adoption",
        parent=parent,
        critical=False
    )

    # Extracted data
    # Note: We rely on the single extraction result recorded earlier.
    # The last extraction added will be accessible via our stored result, but we keep a local reference by passing it in main.
    # Here we fetch from the evaluator's recorded extractions for clarity (the last TripPlanExtraction).
    # However, to avoid relying on internal structures, we will pass the object through custom_info and retrieve it there.
    # Simpler: We'll just attach the extraction to this function via closure. For code clarity, we fetch from custom_info.
    # To avoid complexity, we'll require caller to pass the object (set via evaluator.add_custom_info). Fetch here:
    # But simpler in this script: we assume the caller set a custom_info entry named "extracted_trip_plan".
    extracted_all = None
    for info in evaluator.get_summary()["eval_breakdown"][0]["info"]:
        if "extracted_trip_plan" in info:
            extracted_all = info["extracted_trip_plan"]
    # Fallback: do nothing if not found; but we will instead pass data directly as function arg.
    # Since above approach depends on summary, we will not use it. We'll redesign to pass data directly.
    # We keep this comment for reference.

    # We'll instead pull the latest extraction from evaluator._extraction_results directly (internal).
    # To stay within public APIs, the main function will pass the object into this verifier via parameter.
    # Therefore, we modify signature to accept 'data' below.
    pass  # Placeholder to satisfy function definition (will be overridden below)


async def verify_animal_shelter_with_data(evaluator: Evaluator, parent, data: Optional[AnimalShelter]) -> None:
    node = evaluator.add_parallel(
        id="Animal_Shelter_Facility",
        desc="Identify an animal shelter that is open on Saturdays and suitable for pet adoption",
        parent=parent,
        critical=False
    )

    # Critical: Website URL present
    evaluator.add_custom_node(
        result=_valid_url(data.website_url) if data else False,
        id="Shelter_Website_URL",
        desc="Provide the official website URL for the shelter",
        parent=node,
        critical=True
    )

    urls = _merge_urls(data.website_url if data else None, data.source_urls if data else [])

    # Shelter name and address (critical)
    leaf = evaluator.add_leaf(
        id="Shelter_Location_and_Name",
        desc="Provide the shelter's name and complete physical address",
        parent=node,
        critical=True
    )
    claim = (
        f"The official shelter page shows the shelter named '{_blank_guard(data.name) if data else ''}' "
        f"with the physical address '{_blank_guard(data.address) if data else ''}'."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "If either the name or address in the claim is missing/blank, judge Incorrect. "
            "Accept minor formatting differences (e.g., Rd vs Road). "
            "The address should be in Larimer County or nearby communities such as Fort Collins or Loveland."
        )
    )

    # Saturday Operating Hours (critical)
    leaf = evaluator.add_leaf(
        id="Saturday_Operating_Hours",
        desc="Confirm the shelter is open on Saturdays and provide specific operating hours",
        parent=node,
        critical=True
    )
    claim = (
        f"The shelter is open on Saturdays for pet adoptions; Saturday hours: '{_blank_guard(data.saturday_hours) if data else ''}'."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Verify Saturday availability specifically for adoptions if distinguished from general shelter hours. "
            "If the hours string is blank, or the page indicates closed on Saturday for adoptions, judge Incorrect. "
            "If the page states 'by appointment' for Saturday adoptions, consider that as open."
        )
    )

    # Adoption Fee Information (critical)
    leaf = evaluator.add_leaf(
        id="Adoption_Fee_Information",
        desc="Provide adoption fee ranges and what services are included (spay/neuter, microchip, vaccines)",
        parent=node,
        critical=True
    )
    claim = f"The official page lists adoption fee information and included services as: '{_blank_guard(data.adoption_fee_info) if data else ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Match fee amounts and included services (e.g., spay/neuter, microchip, vaccinations). "
            "If the provided fee info is blank, judge Incorrect. Allow reasonable paraphrase."
        )
    )

    # Adoption Age Requirement (non-critical)
    leaf = evaluator.add_leaf(
        id="Adoption_Age_Requirement",
        desc="Confirm the minimum age requirement for adopters (typically 18 years old)",
        parent=node,
        critical=False
    )
    claim = f"The minimum age requirement for adopters is '{_blank_guard(data.age_requirement) if data else ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="If the provided age value is blank, judge Incorrect."
    )

    # Required Documentation (non-critical)
    leaf = evaluator.add_leaf(
        id="Required_Documentation",
        desc="List required documentation for adoption (ID, proof of residence, etc.)",
        parent=node,
        critical=False
    )
    claim = f"The required documentation for adoption includes: '{_blank_guard(data.required_documentation) if data else ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="Accept reasonable paraphrase. If blank, judge Incorrect."
    )


async def verify_dog_park_with_data(evaluator: Evaluator, parent, data: Optional[DogPark]) -> None:
    node = evaluator.add_parallel(
        id="Dog_Park_Facility",
        desc="Identify a dog park or off-leash area in the same county/region",
        parent=parent,
        critical=False
    )

    # Critical: Park reference URL present
    evaluator.add_custom_node(
        result=_valid_url(data.reference_url) if data else False,
        id="Park_Reference_URL",
        desc="Provide an official reference URL for the dog park",
        parent=node,
        critical=True
    )

    urls = _merge_urls(data.reference_url if data else None, data.source_urls if data else [])

    # Park Name and Address (critical)
    leaf = evaluator.add_leaf(
        id="Park_Name_and_Address",
        desc="Provide the park's name and complete physical address",
        parent=node,
        critical=True
    )
    claim = (
        f"The official/reference page shows the dog park named '{_blank_guard(data.name) if data else ''}' "
        f"with the physical address '{_blank_guard(data.address) if data else ''}'."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "If name or address is blank, judge Incorrect. "
            "Accept minor formatting differences. Prefer parks in Larimer County or nearby (Fort Collins, Loveland)."
        )
    )

    # Park Features (non-critical)
    leaf = evaluator.add_leaf(
        id="Park_Features",
        desc="Describe key features (fenced areas, separate small/large dog sections, amenities)",
        parent=node,
        critical=False
    )
    claim = f"Key park features include: '{_blank_guard(data.features) if data else ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="Accept paraphrase and feature synonyms. If blank, judge Incorrect."
    )

    # Leash Requirements (critical)
    leaf = evaluator.add_leaf(
        id="Leash_Requirements",
        desc="Specify leash requirements (in parking areas, designated zones, etc.)",
        parent=node,
        critical=True
    )
    claim = f"Leash requirements/policies are: '{_blank_guard(data.leash_requirements) if data else ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Specifically verify leash rules (e.g., must be leashed outside designated off‑leash enclosure). "
            "If blank or not supported, judge Incorrect."
        )
    )

    # Required Vaccinations (non-critical)
    leaf = evaluator.add_leaf(
        id="Required_Vaccinations",
        desc="Confirm if proof of rabies vaccination is required",
        parent=node,
        critical=False
    )
    claim = f"The page states the rabies vaccination/proof policy as: '{_blank_guard(data.rabies_requirement) if data else ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="If the claim is blank, judge Incorrect. Accept 'not specified' only if the page clearly says it is not required."
    )

    # Operating Hours / Access (non-critical)
    leaf = evaluator.add_leaf(
        id="Operating_Hours_Access",
        desc="Provide information about park hours or access times",
        parent=node,
        critical=False
    )
    claim = f"Park hours/access information: '{_blank_guard(data.hours_access) if data else ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="If blank, judge Incorrect. Accept reasonable paraphrase."
    )


async def verify_emergency_clinic_with_data(evaluator: Evaluator, parent, data: Optional[EmergencyClinic]) -> None:
    node = evaluator.add_parallel(
        id="Emergency_Veterinary_Clinic",
        desc="Identify an emergency veterinary clinic with weekend availability",
        parent=parent,
        critical=False
    )

    # Critical: Clinic website URL present
    evaluator.add_custom_node(
        result=_valid_url(data.website_url) if data else False,
        id="Clinic_Website_URL",
        desc="Provide the official website URL for the emergency veterinary clinic",
        parent=node,
        critical=True
    )

    urls = _merge_urls(data.website_url if data else None, data.source_urls if data else [])

    # Clinic Name and Location (critical)
    leaf = evaluator.add_leaf(
        id="Clinic_Name_and_Location",
        desc="Provide the clinic's name and complete physical address",
        parent=node,
        critical=True
    )
    claim = (
        f"The official clinic page shows the clinic named '{_blank_guard(data.name) if data else ''}' "
        f"with the physical address '{_blank_guard(data.address) if data else ''}'."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "If name or address is blank, judge Incorrect. "
            "Address should be in/near Larimer County (e.g., Fort Collins, Loveland)."
        )
    )

    # Weekend Hours Availability (critical)
    leaf = evaluator.add_leaf(
        id="Weekend_Hours_Availability",
        desc="Confirm weekend operating hours (Saturday and Sunday availability)",
        parent=node,
        critical=True
    )
    sat = _blank_guard(data.saturday_hours) if data else ""
    sun = _blank_guard(data.sunday_hours) if data else ""
    claim = f"The clinic is open on weekends: Saturday hours '{sat}', Sunday hours '{sun}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Verify explicit weekend availability. If either day is blank or clearly closed, judge Incorrect. "
            "If the page says 24/7, that implies both Saturday and Sunday are covered."
        )
    )

    # Emergency Services Offered (non-critical)
    leaf = evaluator.add_leaf(
        id="Emergency_Services_Offered",
        desc="Describe the types of emergency services available",
        parent=node,
        critical=False
    )
    claim = f"Emergency services offered include: '{_blank_guard(data.services_offered) if data else ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="If blank, judge Incorrect. Accept summarized service lists."
    )

    # Contact Information (non-critical)
    leaf = evaluator.add_leaf(
        id="Contact_Information",
        desc="Provide phone number for emergency contact",
        parent=node,
        critical=False
    )
    claim = f"The emergency contact phone number is '{_blank_guard(data.phone) if data else ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="If the phone number is blank or not found on the page, judge Incorrect."
    )

    # 24/7 Status (non-critical)
    leaf = evaluator.add_leaf(
        id="Twenty_Four_Hour_Status",
        desc="Indicate whether the clinic operates 24/7 or has limited hours",
        parent=node,
        critical=False
    )
    interpreted_247 = _is_24_7_text(data.is_24_7 if data else None)
    if interpreted_247 is True:
        claim = "The clinic operates 24/7."
    elif interpreted_247 is False:
        claim = "The clinic does not operate 24/7 and has limited hours."
    else:
        claim = f"The clinic's 24/7 status is described as: '{_blank_guard(data.is_24_7) if data else ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Prefer explicit '24/7' or '24 hours' statements. If limited hours are specified, the 'does not operate 24/7' claim should be supported. "
            "If the provided status text is blank, judge Incorrect."
        )
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
    Evaluate an answer for the Weekend Pet Adoption Trip Planning task in Larimer County, CO.
    """
    # Initialize evaluator (root must be non-critical to avoid constraint that critical parent requires all critical children)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Facilities are independent
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

    # Extract structured data for all three facilities
    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction"
    )

    # Record extraction in custom info to help debugging
    evaluator.add_custom_info(
        info=extracted.dict(),
        info_type="extraction",
        info_name="extracted_trip_plan"
    )

    # Build verification subtree according to rubric (three parallel facility categories)
    # Animal Shelter
    await verify_animal_shelter_with_data(evaluator, root, extracted.animal_shelter if extracted else None)

    # Dog Park
    await verify_dog_park_with_data(evaluator, root, extracted.dog_park if extracted else None)

    # Emergency Veterinary Clinic
    await verify_emergency_clinic_with_data(evaluator, root, extracted.emergency_clinic if extracted else None)

    # Return structured evaluation summary
    return evaluator.get_summary()