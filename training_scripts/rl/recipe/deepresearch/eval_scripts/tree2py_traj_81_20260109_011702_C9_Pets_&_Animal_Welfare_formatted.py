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
TASK_ID = "la_animal_shelters_weekend_volunteer"
TASK_DESCRIPTION = """I'm relocating to Louisiana and want to get involved with local animal welfare organizations. I'm specifically looking for animal shelters that are accessible on weekends and offer comprehensive volunteer opportunities. Please identify three animal shelters located in Louisiana that meet all of the following requirements:

1. The shelter must be open to the public on Saturdays, with specific operating hours clearly stated. (Sunday hours should also be indicated if available.)

2. The shelter must have an active volunteer program that clearly states:
   - The minimum age requirement for volunteers
   - The time commitment required from volunteers (weekly or monthly hours)

3. The shelter must offer adoption services that include medical care (such as spay/neuter, vaccinations, and/or microchipping) as part of the adoption package.

4. The shelter must have an active foster program for dogs or cats.

For each shelter, please provide:
- Official name
- Physical address in Louisiana
- Contact information (phone number and/or email)
- Official website URL
- Specific operating hours for Saturday (and Sunday if applicable)
- Volunteer program details (minimum age and time commitment)
- Confirmation of medical services included in adoptions
- Information about the foster program
- Information about any additional community services they offer, such as low-cost spay/neuter services or education programs (if available)
- URL references to verify all the information provided
"""

# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class CommunityServices(BaseModel):
    low_cost_spay_neuter_info: Optional[str] = None
    education_outreach_info: Optional[str] = None


class ShelterItem(BaseModel):
    # Basic info
    name: Optional[str] = None
    address: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    website: Optional[str] = None

    # Hours
    saturday_hours: Optional[str] = None
    sunday_hours: Optional[str] = None

    # Volunteer
    volunteer_min_age: Optional[str] = None
    volunteer_time_commitment: Optional[str] = None

    # Adoption
    adoption_available: Optional[str] = None
    adoption_medical_included: Optional[str] = None  # e.g., "spay/neuter, vaccines, microchip"

    # Foster
    foster_program_info: Optional[str] = None  # general description
    foster_species: Optional[str] = None  # e.g., "dogs", "cats", "dogs and cats"

    # Additional services (optional)
    community_services: CommunityServices = Field(default_factory=CommunityServices)

    # URLs cited in the answer to verify
    verification_urls: List[str] = Field(default_factory=list)


class SheltersExtraction(BaseModel):
    shelters: List[ShelterItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shelters() -> str:
    return """
    Extract up to three (3) animal shelters located in Louisiana as presented in the answer. For each shelter, extract the following fields exactly as stated in the answer. Do not invent or infer information not explicitly in the answer.

    Required fields per shelter (use null if not provided):
    - name: Official shelter name.
    - address: Physical mailing/street address (should indicate it's in Louisiana).
    - contact_phone: A phone number if provided.
    - contact_email: An email address if provided.
    - website: Official website URL (absolute URL; if missing protocol, prepend http://).
    - saturday_hours: The exact Saturday public hours string (e.g., "Sat 10am–4pm" or "Saturday: 12:00–17:00"). If the answer explicitly states "closed" for Saturday, extract that literal text.
    - sunday_hours: The exact Sunday hours/status string if provided (e.g., "Sun Closed", "By appointment", or a time range). If not mentioned in the answer, set to null.
    - volunteer_min_age: The stated minimum age requirement for volunteers (e.g., "16+", "18 and older", "14 with guardian").
    - volunteer_time_commitment: The stated volunteer time commitment (e.g., "2 hours/week", "8 hours/month", "one shift per week").
    - adoption_available: A short phrase indicating adoption is available if explicitly stated (e.g., "adoptions available"). If not explicitly stated, set to null.
    - adoption_medical_included: The medical services included with adoptions exactly as stated (e.g., "spay/neuter, vaccines, and microchip"; "spay/neuter included"). If not explicitly stated, set to null.
    - foster_program_info: A short description/summary of the foster program if present (e.g., what it is or requirements). If not mentioned, set to null.
    - foster_species: Which animals are fostered if stated (e.g., "dogs", "cats", "dogs and cats"). If not specified, set to null.
    - community_services.low_cost_spay_neuter_info: Mention of low‑cost spay/neuter if available; else null.
    - community_services.education_outreach_info: Mention of education/outreach if available; else null.
    - verification_urls: An array of all URLs cited in the answer for this shelter (including the official website if present). Only include valid URLs explicitly appearing in the answer.

    Return a JSON object with a single field:
    {
      "shelters": [ ... up to 3 shelter objects as specified ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
def _all_sources_for_shelter(s: ShelterItem) -> List[str]:
    urls = []
    if s.website and isinstance(s.website, str) and s.website.strip():
        urls.append(s.website.strip())
    for u in s.verification_urls:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _display_name(s: ShelterItem, idx: int) -> str:
    return s.name.strip() if s.name else f"Shelter #{idx+1}"


# --------------------------------------------------------------------------- #
# Shelter Verification Logic                                                  #
# --------------------------------------------------------------------------- #
async def verify_single_shelter(
    evaluator: Evaluator,
    parent_node,
    shelter: ShelterItem,
    idx: int,
) -> None:
    """
    Build the verification subtree for a single shelter entry, following (and slightly adapting)
    the rubric. We adjust criticality for the Weekend_Hours grouping to allow Sunday to be optional,
    while keeping Saturday as a mandatory (critical) requirement at the shelter level.
    """
    # Top-level shelter node (parallel, non-critical; overall root tallies partial credit across shelters)
    shelter_node = evaluator.add_parallel(
        id=f"shelter_{idx+1}",
        desc=f"Shelter #{idx+1} (one qualifying Louisiana animal shelter)",
        parent=parent_node,
        critical=False
    )

    all_sources = _all_sources_for_shelter(shelter)
    shelter_name_for_claim = _display_name(shelter, idx)

    # ------------------------ Basic_Info (critical group) ------------------------
    basic_node = evaluator.add_parallel(
        id=f"s{idx+1}_basic_info",
        desc="Provide required basic identifying/contact info",
        parent=shelter_node,
        critical=True
    )

    # Name (existence)
    evaluator.add_custom_node(
        result=bool(shelter.name and shelter.name.strip()),
        id=f"s{idx+1}_name",
        desc="Official name is provided",
        parent=basic_node,
        critical=True
    )

    # Address in Louisiana (verify by URLs)
    addr_node = evaluator.add_leaf(
        id=f"s{idx+1}_address_LA",
        desc="Physical address is provided and is in Louisiana",
        parent=basic_node,
        critical=True
    )
    addr_text = shelter.address.strip() if shelter.address else ""
    addr_claim = f"The physical address for {shelter_name_for_claim} is '{addr_text}', and this address is located in the state of Louisiana."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_node,
        sources=all_sources,
        additional_instruction="Confirm the page shows the shelter's postal/street address in Louisiana (LA or 'Louisiana'). Fuzzy match OK."
    )

    # Contact info provided (existence of phone or email)
    contact_exists = bool((shelter.contact_phone and shelter.contact_phone.strip()) or
                          (shelter.contact_email and shelter.contact_email.strip()))
    evaluator.add_custom_node(
        result=contact_exists,
        id=f"s{idx+1}_contact",
        desc="Contact information is provided (phone number and/or email)",
        parent=basic_node,
        critical=True
    )

    # Website provided (existence)
    evaluator.add_custom_node(
        result=bool(shelter.website and shelter.website.strip()),
        id=f"s{idx+1}_website",
        desc="Official website URL is provided",
        parent=basic_node,
        critical=True
    )

    # ------------------ Saturday_Public_Hours (critical leaf at shelter level) ------------------
    # We place Saturday as its own critical leaf to ensure it is enforced as mandatory,
    # while Sunday remains optional within a non-critical Weekend_Hours group below.
    sat_node = evaluator.add_leaf(
        id=f"s{idx+1}_saturday_hours",
        desc="Shelter is open to the public on Saturdays and specific Saturday operating hours are stated",
        parent=shelter_node,
        critical=True
    )
    sat_text = shelter.saturday_hours.strip() if shelter.saturday_hours else ""
    sat_claim = f"For {shelter_name_for_claim}, they are open to the public on Saturdays with the following hours: {sat_text}."
    await evaluator.verify(
        claim=sat_claim,
        node=sat_node,
        sources=all_sources,
        additional_instruction="Verify that Saturday hours are explicitly listed and indicate the shelter is open to the public (not merely 'by appointment' unless explicitly stated as public access)."
    )

    # ------------------------ Weekend_Hours (non-critical group; Sunday optional) ------------------------
    weekend_node = evaluator.add_parallel(
        id=f"s{idx+1}_weekend_hours",
        desc="Weekend public accessibility information",
        parent=shelter_node,
        critical=False  # Set non-critical to allow Sunday to be optional
    )

    # Sunday status (non-critical)
    sun_node = evaluator.add_leaf(
        id=f"s{idx+1}_sunday_status",
        desc="Sunday hours/status are stated if available (e.g., open hours, closed, or by appointment)",
        parent=weekend_node,
        critical=False
    )
    if shelter.sunday_hours and shelter.sunday_hours.strip():
        sun_text = shelter.sunday_hours.strip()
        sun_claim = f"For {shelter_name_for_claim}, the Sunday hours/status is: {sun_text}."
        await evaluator.verify(
            claim=sun_claim,
            node=sun_node,
            sources=all_sources,
            additional_instruction="Confirm the page shows Sunday hours or an explicit status (such as Closed or By Appointment). Allow small formatting variations."
        )
    else:
        # If no Sunday info in the answer, mark as skipped to avoid penalization on an optional field.
        sun_node.score = 0.0
        sun_node.status = "skipped"

    # ------------------------ Volunteer_Program (critical group) ------------------------
    volunteer_node = evaluator.add_parallel(
        id=f"s{idx+1}_volunteer",
        desc="Volunteer program meets stated requirements",
        parent=shelter_node,
        critical=True
    )

    # Volunteer Exists
    vol_exists_node = evaluator.add_leaf(
        id=f"s{idx+1}_volunteer_exists",
        desc="Active volunteer program exists",
        parent=volunteer_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{shelter_name_for_claim} has an active volunteer program.",
        node=vol_exists_node,
        sources=all_sources,
        additional_instruction="Look for a page or section about volunteering that suggests active participation is possible (applications, onboarding, schedules, etc.)."
    )

    # Volunteer Min Age
    vol_age_node = evaluator.add_leaf(
        id=f"s{idx+1}_volunteer_min_age",
        desc="Minimum age requirement for volunteers is stated",
        parent=volunteer_node,
        critical=True
    )
    vol_age_text = shelter.volunteer_min_age.strip() if shelter.volunteer_min_age else ""
    await evaluator.verify(
        claim=f"The minimum age for volunteers at {shelter_name_for_claim} is: {vol_age_text}.",
        node=vol_age_node,
        sources=all_sources,
        additional_instruction="Confirm the stated age requirement (e.g., 16+, 18+, or with guardian). Allow variants like 'must be 16 years old'."
    )

    # Volunteer Time Commitment
    vol_time_node = evaluator.add_leaf(
        id=f"s{idx+1}_volunteer_time_commitment",
        desc="Volunteer time commitment is stated (weekly or monthly hours)",
        parent=volunteer_node,
        critical=True
    )
    vol_time_text = shelter.volunteer_time_commitment.strip() if shelter.volunteer_time_commitment else ""
    await evaluator.verify(
        claim=f"The volunteer program at {shelter_name_for_claim} requires the following time commitment: {vol_time_text}.",
        node=vol_time_node,
        sources=all_sources,
        additional_instruction="Verify explicit time requirements (e.g., hours per week/month, shifts per week). Allow reasonable paraphrase."
    )

    # ------------------------ Adoption_With_Medical (critical group) ------------------------
    adoption_node = evaluator.add_parallel(
        id=f"s{idx+1}_adoption",
        desc="Adoptions include medical care elements",
        parent=shelter_node,
        critical=True
    )

    # Adoption Available
    adopt_exists_node = evaluator.add_leaf(
        id=f"s{idx+1}_adoption_available",
        desc="Adoption services are available",
        parent=adoption_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{shelter_name_for_claim} offers adoption services.",
        node=adopt_exists_node,
        sources=all_sources,
        additional_instruction="Confirm the shelter adopts out animals (adoptable animals page, how to adopt, etc.)."
    )

    # Medical Included
    adopt_med_node = evaluator.add_leaf(
        id=f"s{idx+1}_adoption_medical_included",
        desc="Adoption package includes medical care such as spay/neuter, vaccinations, and/or microchipping",
        parent=adoption_node,
        critical=True
    )
    med_detail = shelter.adoption_medical_included.strip() if shelter.adoption_medical_included else "medical care (e.g., spay/neuter or vaccinations or microchipping)"
    await evaluator.verify(
        claim=f"Adoption packages at {shelter_name_for_claim} include {med_detail}.",
        node=adopt_med_node,
        sources=all_sources,
        additional_instruction="Verify that at least one medical element (spay/neuter, vaccinations, or microchip) is included with adoptions. Accept clear implication that fees cover these services."
    )

    # ------------------------ Foster_Program (critical group) ------------------------
    foster_node = evaluator.add_parallel(
        id=f"s{idx+1}_foster",
        desc="Foster program availability",
        parent=shelter_node,
        critical=True
    )

    # Foster Exists (dogs or cats)
    foster_exists_node = evaluator.add_leaf(
        id=f"s{idx+1}_foster_exists",
        desc="Active foster program exists for dogs or cats",
        parent=foster_node,
        critical=True
    )
    species_text = shelter.foster_species.strip() if shelter.foster_species else "dogs or cats"
    await evaluator.verify(
        claim=f"{shelter_name_for_claim} has an active foster program for {species_text}.",
        node=foster_exists_node,
        sources=all_sources,
        additional_instruction="Look for a foster page or call-to-action indicating that people can foster animals (dogs/cats/kittens/puppies)."
    )

    # Foster Info Provided (verify some description)
    foster_info_node = evaluator.add_leaf(
        id=f"s{idx+1}_foster_info",
        desc="Some description/information about the foster program is provided",
        parent=foster_node,
        critical=True
    )
    foster_info_text = shelter.foster_program_info.strip() if shelter.foster_program_info else ""
    await evaluator.verify(
        claim=f"Foster program details at {shelter_name_for_claim}: {foster_info_text}",
        node=foster_info_node,
        sources=all_sources,
        additional_instruction="Verify that the source mentions some descriptive details about the foster program (requirements, responsibilities, how to apply, etc.)."
    )

    # ------------------------ Additional_Community_Services (non-critical) ------------------------
    additional_node = evaluator.add_parallel(
        id=f"s{idx+1}_additional_services",
        desc="Additional community services are mentioned if available (optional)",
        parent=shelter_node,
        critical=False
    )

    # Low-cost spay/neuter (non-critical, only if stated in the answer)
    if shelter.community_services.low_cost_spay_neuter_info and shelter.community_services.low_cost_spay_neuter_info.strip():
        low_node = evaluator.add_leaf(
            id=f"s{idx+1}_low_cost_spayneuter",
            desc="Mentions low-cost spay/neuter services if offered",
            parent=additional_node,
            critical=False
        )
        low_text = shelter.community_services.low_cost_spay_neuter_info.strip()
        await evaluator.verify(
            claim=f"{shelter_name_for_claim} offers low-cost spay/neuter: {low_text}",
            node=low_node,
            sources=all_sources,
            additional_instruction="Confirm an affordable/low-cost spay-neuter program is offered or promoted."
        )

    # Education/outreach (non-critical, only if stated in the answer)
    if shelter.community_services.education_outreach_info and shelter.community_services.education_outreach_info.strip():
        edu_node = evaluator.add_leaf(
            id=f"s{idx+1}_education_outreach",
            desc="Mentions education/outreach programs if offered",
            parent=additional_node,
            critical=False
        )
        edu_text = shelter.community_services.education_outreach_info.strip()
        await evaluator.verify(
            claim=f"{shelter_name_for_claim} provides community education/outreach programs: {edu_text}",
            node=edu_node,
            sources=all_sources,
            additional_instruction="Verify references to education, outreach, humane education, community programs, etc."
        )

    # ------------------------ Verification_URLs (critical existence) ------------------------
    evaluator.add_custom_node(
        result=len(all_sources) > 0,
        id=f"s{idx+1}_verification_urls_present",
        desc="Provide URL references that verify the stated information (hours, volunteer details, adoption medical services, foster program, and basic info)",
        parent=shelter_node,
        critical=True
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
    Evaluate an answer for the Louisiana animal shelters task.
    """
    evaluator = Evaluator()

    # IMPORTANT: Make root non-critical to allow partial scoring across shelters
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

    # Extract structured shelters info
    extracted: SheltersExtraction = await evaluator.extract(
        prompt=prompt_extract_shelters(),
        template_class=SheltersExtraction,
        extraction_name="shelters_extraction",
    )

    # Keep only the first 3 shelters; pad with empty if fewer
    shelters: List[ShelterItem] = list(extracted.shelters[:3])
    while len(shelters) < 3:
        shelters.append(ShelterItem())

    # Build verification subtrees for three shelters
    for i in range(3):
        await verify_single_shelter(evaluator, root, shelters[i], i)

    # Return evaluation summary
    return evaluator.get_summary()