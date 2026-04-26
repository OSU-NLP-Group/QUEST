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
TASK_ID = "nc_dog_facilities"
TASK_DESCRIPTION = """
I am looking for professional dog training and boarding facilities in the Raleigh or Johnston County, North Carolina area to care for my Belgian Sheepdog, a herding breed that requires specialized handling. Please identify three facilities that meet all of the following criteria:

1. Location: The facility must be located in Raleigh or Johnston County, North Carolina, with a complete physical address provided.
2. Professional Qualifications: At least one trainer at the facility must hold a recognized professional dog training certification (such as CPDT-KA, CPDT-KSA, or equivalent recognized certification).
3. Facility Capabilities: The facility must have adequate space for training and boarding operations, and should mention proper licensing or insurance coverage.
4. Service Offerings: The facility must offer:
   - Professional dog training services (private lessons, group classes, or boot camp programs)
   - Dog boarding or daycare services
   - Clear operating hours or availability information
   - Demonstrated capability to handle herding breeds or working dogs
5. Health and Safety: The facility must specify vaccination requirements for dogs (such as rabies, DHPP, Bordetella) and should mention veterinary care or emergency protocols.

For each facility, provide:
- Facility name
- Complete physical address
- Official website or verified online presence
- Details about trainer certifications
- Description of services offered
- Operating hours
- Health and vaccination requirements
- Reference URLs that verify each piece of information
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityAddress(BaseModel):
    address_full: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None


class FacilityUrls(BaseModel):
    main_url: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    qualification_urls: List[str] = Field(default_factory=list)
    capability_urls: List[str] = Field(default_factory=list)
    service_urls: List[str] = Field(default_factory=list)
    health_urls: List[str] = Field(default_factory=list)
    area_verification_urls: List[str] = Field(default_factory=list)


class FacilityQualifications(BaseModel):
    certifications: List[str] = Field(default_factory=list)  # e.g., ["CPDT-KA (Jane Doe)", "KPA-CTP"]
    trainers: List[str] = Field(default_factory=list)        # names of trainers (optional)
    experience: Optional[str] = None                         # years or notable experience (optional)


class FacilityCapabilities(BaseModel):
    space_description: Optional[str] = None                  # e.g., "indoor/outdoor yards, 5,000 sq ft"
    licensing_insurance: Optional[str] = None                # e.g., "licensed and insured"


class FacilityServices(BaseModel):
    training_services: List[str] = Field(default_factory=list)  # e.g., ["private lessons", "group classes", "board & train"]
    boarding_or_daycare: Optional[str] = None                   # text description if available
    hours: Optional[str] = None                                  # hours or availability info
    breed_handling: Optional[str] = None                         # herding/working breed capability mention


class FacilityHealth(BaseModel):
    vaccinations: List[str] = Field(default_factory=list)        # e.g., ["rabies", "DHPP", "Bordetella"]
    veterinary_emergency: Optional[str] = None                   # e.g., "vet on call, emergency protocol"


class FacilityItem(BaseModel):
    name: Optional[str] = None
    address: FacilityAddress = FacilityAddress()
    area_label: Optional[str] = None  # e.g., "Raleigh" or a Johnston County city like "Clayton", "Smithfield", etc.
    urls: FacilityUrls = FacilityUrls()
    qualifications: FacilityQualifications = FacilityQualifications()
    capabilities: FacilityCapabilities = FacilityCapabilities()
    services: FacilityServices = FacilityServices()
    health: FacilityHealth = FacilityHealth()


class FacilitiesExtraction(BaseModel):
    facilities: List[FacilityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
Extract up to the first three qualifying facilities for the given task from the answer. For each facility, return the following JSON fields. Only extract what the answer explicitly states; do not invent.

For each facility (in the same order mentioned in the answer):
- name: The facility name (string).
- address:
  - address_full: The full physical address as a single line if provided (should include street, city, state, and ZIP when available).
  - city: City if explicitly provided.
  - state: State if explicitly provided (e.g., "NC").
  - zip: ZIP code if explicitly provided.
- area_label: The city/county label if explicitly mentioned (e.g., "Raleigh", "Johnston County", "Clayton", "Smithfield", etc.).
- urls:
  - main_url: The primary official website or verified online presence (Google Business, Facebook page, Yelp, etc.) if provided.
  - location_urls: URLs specifically supporting the address/location (contact page, footer page, Google map/listing, etc.).
  - qualification_urls: URLs that mention trainer certifications or credentials.
  - capability_urls: URLs that describe the facility's physical space, boarding/training infrastructure, or compliance mentions.
  - service_urls: URLs that list service offerings (training, boarding/daycare) and hours/availability.
  - health_urls: URLs that specify vaccination or health requirements and/or emergency protocols.
  - area_verification_urls: URLs that help verify that the specified city is within Johnston County, if applicable (e.g., a city or county page, Wikipedia).
- qualifications:
  - certifications: An array of strings for any certifications explicitly mentioned (e.g., "CPDT-KA", "CPDT-KSA", "KPA-CTP", "IAABC-ADT/ACDBC/DBC/CDBC", "IACP-CDT/CDTA", etc.), optionally with trainer names in parentheses if included in the answer.
  - trainers: An array of trainer names if listed.
  - experience: A short text describing professional dog training experience if mentioned (optional).
- capabilities:
  - space_description: Text describing adequate space for training/boarding (e.g., indoor/outdoor yards, training rooms, square footage, kennels/runs).
  - licensing_insurance: Text that mentions licensing or insurance (optional).
- services:
  - training_services: Array of training offerings (e.g., "private lessons", "group classes", "board & train", "boot camp").
  - boarding_or_daycare: Text confirming boarding or daycare offering if mentioned.
  - hours: Operating hours or availability info (days/times or "by appointment") if provided.
  - breed_handling: Text indicating capability with herding/working breeds (e.g., "Belgian Sheepdog", "herding breeds", "working dogs", "German Shepherd", "Border Collie"), if mentioned (optional).
- health:
  - vaccinations: Array of required vaccinations explicitly listed (e.g., "rabies", "DHPP/DA2PP", "Bordetella").
  - veterinary_emergency: Text describing veterinary support or emergency protocols if mentioned (optional).

Return a JSON object with:
{ "facilities": [ <FacilityItem>, <FacilityItem>, <FacilityItem> ] }

If the answer includes fewer than three, return as many as available. If any subfield is not provided in the answer, set it to null (or empty array for lists).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*args: Optional[Any]) -> List[str]:
    """Flatten string/list/None into a unique list of URL strings."""
    out: List[str] = []
    for item in args:
        if not item:
            continue
        if isinstance(item, str):
            if item.strip():
                out.append(item.strip())
        elif isinstance(item, list):
            for s in item:
                if isinstance(s, str) and s.strip():
                    out.append(s.strip())
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for url in out:
        if url not in seen:
            seen.add(url)
            uniq.append(url)
    return uniq


def _first_n_facilities(extracted: FacilitiesExtraction, n: int = 3) -> List[FacilityItem]:
    facs = list(extracted.facilities or [])
    if len(facs) >= n:
        return facs[:n]
    # pad with empty shells if fewer
    while len(facs) < n:
        facs.append(FacilityItem())
    return facs


# --------------------------------------------------------------------------- #
# Verification builder per facility                                           #
# --------------------------------------------------------------------------- #
async def verify_one_facility(evaluator: Evaluator, parent, facility: FacilityItem, idx: int) -> None:
    """
    Build verification sub-tree for a single facility and run checks.
    We slightly reorganize non-critical subitems outside critical groups to satisfy the
    framework constraint: a critical parent cannot have non-critical children.
    """

    # Parent node for this facility (non-critical to allow partial across facilities)
    fnode = evaluator.add_parallel(
        id=f"facility_{idx+1}",
        desc=f"{['First','Second','Third'][idx]} qualifying facility identified and verified",
        parent=parent,
        critical=False,
    )

    name = facility.name or f"Facility #{idx+1}"
    addr = facility.address or FacilityAddress()
    area_label = facility.area_label or addr.city or ""
    urls = facility.urls or FacilityUrls()

    # Source buckets
    loc_sources = _combine_sources(urls.location_urls, urls.main_url)
    qual_sources = _combine_sources(urls.qualification_urls, urls.main_url)
    cap_sources = _combine_sources(urls.capability_urls, urls.main_url, urls.service_urls)
    svc_sources = _combine_sources(urls.service_urls, urls.main_url)
    hlth_sources = _combine_sources(urls.health_urls, urls.main_url)
    area_sources = _combine_sources(urls.area_verification_urls, urls.location_urls, urls.main_url)

    # -------------------- Location group (critical) -------------------- #
    loc_group = evaluator.add_parallel(
        id=f"facility_{idx+1}_location",
        desc="Location and contact information verified",
        parent=fnode,
        critical=True,
    )

    # Sub-group: physical details (critical)
    loc_details = evaluator.add_parallel(
        id=f"facility_{idx+1}_location_details",
        desc="Physical location details provided",
        parent=loc_group,
        critical=True,
    )

    # Leaf: complete address
    leaf_addr = evaluator.add_leaf(
        id=f"facility_{idx+1}_location_address",
        desc="Complete physical address provided (street address, city, state, ZIP)",
        parent=loc_details,
        critical=True,
    )
    addr_text = addr.address_full or ""
    claim_addr = (
        f"The page lists a full physical address for {name} that includes street address, city, state (NC), "
        f"and ZIP code. The extracted address is: '{addr_text}'."
    )
    await evaluator.verify(
        claim=claim_addr,
        node=leaf_addr,
        sources=loc_sources,
        additional_instruction=(
            "Pass if the page clearly provides a US-style full address containing at least a street line with "
            "number and street, the city, 'NC' as the state, and a 5‑digit ZIP code. Minor formatting differences are OK. "
            "The address may appear on a Contact/Location page or in the site footer."
        ),
    )

    # Leaf: area constraint (Raleigh or within Johnston County)
    leaf_area = evaluator.add_leaf(
        id=f"facility_{idx+1}_location_area",
        desc="Located within Raleigh or Johnston County, North Carolina",
        parent=loc_details,
        critical=True,
    )
    area_text = area_label or ""
    claim_area = (
        f"The facility is located in {area_text}, North Carolina. "
        f"This location is in-scope for the task (Raleigh, NC or a municipality within Johnston County, NC)."
    )
    await evaluator.verify(
        claim=claim_area,
        node=leaf_area,
        sources=area_sources,
        additional_instruction=(
            "Accept if the page shows the address/city as one of: Raleigh, or any municipality in Johnston County, NC "
            "(Clayton, Smithfield, Selma, Benson, Archer Lodge, Kenly, Micro, Pine Level, Princeton, Four Oaks, Wilson's Mills). "
            "You only need to confirm the city name is one of these or is Raleigh; do not infer beyond what's on the page."
        ),
    )

    # Sub-group: online presence (critical)
    loc_online = evaluator.add_parallel(
        id=f"facility_{idx+1}_location_online",
        desc="Online presence and verification",
        parent=loc_group,
        critical=True,
    )

    leaf_site = evaluator.add_leaf(
        id=f"facility_{idx+1}_location_website",
        desc="Official website or verified online presence provided",
        parent=loc_online,
        critical=True,
    )
    claim_site = (
        f"This URL is the official website or a verified online presence (e.g., Google Business, Facebook, Yelp) "
        f"for {name}."
    )
    await evaluator.verify(
        claim=claim_site,
        node=leaf_site,
        sources=_combine_sources(urls.main_url, loc_sources),
        additional_instruction=(
            "Accept official domain pages or widely recognized profiles that unambiguously reference the facility name "
            "and location (e.g., Google Business Profile, Facebook Page, Yelp listing)."
        ),
    )

    leaf_loc_url = evaluator.add_leaf(
        id=f"facility_{idx+1}_location_url",
        desc="URL reference verifying location information",
        parent=loc_online,
        critical=True,
    )
    claim_loc_url = (
        "At least one of the provided location URLs explicitly shows the facility's street address and city/state."
    )
    await evaluator.verify(
        claim=claim_loc_url,
        node=leaf_loc_url,
        sources=loc_sources,
        additional_instruction="Look for a contact/location section listing the address block.",
    )

    # -------------------- Qualifications group (critical) -------------------- #
    qual_group = evaluator.add_parallel(
        id=f"facility_{idx+1}_qualifications",
        desc="Professional qualifications verified",
        parent=fnode,
        critical=True,
    )

    # Credentials sub-group (critical) – only include critical leaves here
    qual_creds = evaluator.add_parallel(
        id=f"facility_{idx+1}_qualifications_credentials",
        desc="Trainer credentials and experience (critical elements only)",
        parent=qual_group,
        critical=True,
    )

    leaf_cert = evaluator.add_leaf(
        id=f"facility_{idx+1}_qualifications_certification",
        desc="At least one trainer holds recognized professional certification (e.g., CPDT-KA, CPDT-KSA, or equivalent)",
        parent=qual_creds,
        critical=True,
    )
    cert_list = facility.qualifications.certifications or []
    cert_text = ", ".join(cert_list) if cert_list else ""
    claim_cert = (
        f"At least one trainer at {name} holds a recognized professional dog training certification "
        f"(e.g., CPDT-KA, CPDT-KSA, KPA-CTP, IAABC-ADT/ACDBC/DBC/CDBC, IACP-CDT/CDTA). "
        f"Certifications mentioned in the answer include: {cert_text}."
    )
    await evaluator.verify(
        claim=claim_cert,
        node=leaf_cert,
        sources=qual_sources,
        additional_instruction=(
            "Pass if any page explicitly lists a recognized certification for at least one trainer. "
            "Look for acronyms like CPDT-KA, CPDT-KSA, KPA-CTP, IAABC (trainer levels), or IACP trainer certifications."
        ),
    )

    # Verification sub-group (critical)
    qual_verify = evaluator.add_parallel(
        id=f"facility_{idx+1}_qualifications_verification",
        desc="Credential verification reference",
        parent=qual_group,
        critical=True,
    )

    leaf_qual_url = evaluator.add_leaf(
        id=f"facility_{idx+1}_qualifications_url",
        desc="URL reference verifying trainer qualifications",
        parent=qual_verify,
        critical=True,
    )
    claim_qual_url = "One of these URLs explicitly mentions a trainer certification held at this facility."
    await evaluator.verify(
        claim=claim_qual_url,
        node=leaf_qual_url,
        sources=qual_sources,
        additional_instruction="Look for explicit certification acronyms next to trainer names or in an 'About' page.",
    )

    # Optional, non-critical: experience (moved out of critical parents to satisfy constraint)
    if (facility.qualifications.experience or "").strip():
        leaf_exp = evaluator.add_leaf(
            id=f"facility_{idx+1}_qualifications_experience",
            desc="Evidence of professional dog training experience provided",
            parent=fnode,
            critical=False,
        )
        claim_exp = (
            f"The pages mention professional dog training experience (e.g., years, prior roles, accomplishments) "
            f"for {name}."
        )
        await evaluator.verify(
            claim=claim_exp,
            node=leaf_exp,
            sources=qual_sources,
            additional_instruction="Any clear statement of experience counts (e.g., '10+ years', 'since 2010').",
        )

    # -------------------- Capabilities group (critical) -------------------- #
    cap_group = evaluator.add_parallel(
        id=f"facility_{idx+1}_capabilities",
        desc="Facility physical capabilities and compliance verified",
        parent=fnode,
        critical=True,
    )

    cap_infra = evaluator.add_parallel(
        id=f"facility_{idx+1}_capabilities_infrastructure",
        desc="Physical infrastructure (critical elements only)",
        parent=cap_group,
        critical=True,
    )

    leaf_space = evaluator.add_leaf(
        id=f"facility_{idx+1}_capabilities_space",
        desc="Adequate facility space for training and/or boarding operations described",
        parent=cap_infra,
        critical=True,
    )
    space_text = (facility.capabilities.space_description or "").strip()
    claim_space = (
        f"The facility describes adequate physical space for training and/or boarding operations. "
        f"Example description: '{space_text}'."
    )
    await evaluator.verify(
        claim=claim_space,
        node=leaf_space,
        sources=cap_sources,
        additional_instruction=(
            "Look for mentions of training rooms, indoor/outdoor yards, acres, square footage, kennels/runs, "
            "climate-controlled spaces, or similar. Any concrete space/infra description suffices."
        ),
    )

    cap_verify = evaluator.add_parallel(
        id=f"facility_{idx+1}_capabilities_verification",
        desc="Capability verification reference",
        parent=cap_group,
        critical=True,
    )

    leaf_cap_url = evaluator.add_leaf(
        id=f"facility_{idx+1}_capabilities_url",
        desc="URL reference verifying facility capabilities",
        parent=cap_verify,
        critical=True,
    )
    claim_cap_url = "One of these URLs provides details confirming the facility's physical space suitable for training or boarding."
    await evaluator.verify(
        claim=claim_cap_url,
        node=leaf_cap_url,
        sources=cap_sources,
        additional_instruction="Confirm the page shows concrete facility/space details.",
    )

    # Optional, non-critical: licensing/insurance (moved out of critical parents)
    if (facility.capabilities.licensing_insurance or "").strip():
        leaf_compliance = evaluator.add_leaf(
            id=f"facility_{idx+1}_capabilities_compliance",
            desc="Evidence of proper licensing or insurance coverage mentioned",
            parent=fnode,
            critical=False,
        )
        claim_compliance = "The facility mentions being licensed and/or insured (or equivalent compliance)."
        await evaluator.verify(
            claim=claim_compliance,
            node=leaf_compliance,
            sources=cap_sources,
            additional_instruction="Look for 'licensed', 'insured', 'bonded', or similar compliance statements.",
        )

    # -------------------- Services group (critical) -------------------- #
    svc_group = evaluator.add_parallel(
        id=f"facility_{idx+1}_services",
        desc="Service offerings verified",
        parent=fnode,
        critical=True,
    )

    svc_core = evaluator.add_parallel(
        id=f"facility_{idx+1}_services_core",
        desc="Core service offerings",
        parent=svc_group,
        critical=True,
    )

    leaf_train = evaluator.add_leaf(
        id=f"facility_{idx+1}_services_training",
        desc="Professional dog training services offered (private lessons, group classes, or boot camp)",
        parent=svc_core,
        critical=True,
    )
    train_list = facility.services.training_services or []
    claim_train = (
        "The facility offers professional dog training services such as private lessons, group classes, or board-and-train/boot camp."
    )
    await evaluator.verify(
        claim=claim_train,
        node=leaf_train,
        sources=svc_sources,
        additional_instruction="Any clear training offerings count (group, private, board & train/boot camp).",
    )

    leaf_board = evaluator.add_leaf(
        id=f"facility_{idx+1}_services_boarding",
        desc="Dog boarding or daycare services offered",
        parent=svc_core,
        critical=True,
    )
    boarding_text = (facility.services.boarding_or_daycare or "").strip()
    claim_board = f"The facility offers dog boarding and/or daycare services. Example mention: '{boarding_text}'."
    await evaluator.verify(
        claim=claim_board,
        node=leaf_board,
        sources=svc_sources,
        additional_instruction="Look for 'boarding', 'overnight', 'daycare', 'day camp', etc.",
    )

    svc_ops = evaluator.add_parallel(
        id=f"facility_{idx+1}_services_operational",
        desc="Operational details",
        parent=svc_group,
        critical=True,
    )

    leaf_hours = evaluator.add_leaf(
        id=f"facility_{idx+1}_services_hours",
        desc="Operating hours or availability information provided",
        parent=svc_ops,
        critical=True,
    )
    hours_text = (facility.services.hours or "").strip()
    claim_hours = f"The facility publishes operating hours or clear availability information. Example: '{hours_text}'."
    await evaluator.verify(
        claim=claim_hours,
        node=leaf_hours,
        sources=svc_sources,
        additional_instruction="Accept posted hours, days, or 'by appointment' availability details.",
    )

    svc_verify = evaluator.add_parallel(
        id=f"facility_{idx+1}_services_verification",
        desc="Service verification reference",
        parent=svc_group,
        critical=True,
    )

    leaf_svc_url = evaluator.add_leaf(
        id=f"facility_{idx+1}_services_url",
        desc="URL reference verifying service offerings",
        parent=svc_verify,
        critical=True,
    )
    claim_svc_url = "One of these URLs lists the facility's services and/or operating hours."
    await evaluator.verify(
        claim=claim_svc_url,
        node=leaf_svc_url,
        sources=svc_sources,
        additional_instruction="Look for a services or hours page/section.",
    )

    # Optional, non-critical: breed/working dog capability (moved out of critical parents)
    if (facility.services.breed_handling or "").strip():
        leaf_breed = evaluator.add_leaf(
            id=f"facility_{idx+1}_services_breed",
            desc="Capability to handle herding breeds or working dogs demonstrated",
            parent=fnode,
            critical=False,
        )
        breed_text = facility.services.breed_handling or ""
        claim_breed = (
            f"The facility demonstrates capability to handle herding or working breeds (e.g., Belgian Sheepdog/Shepherd, "
            f"Malinois, Tervuren, German Shepherd, Border Collie). Example mention: '{breed_text}'."
        )
        await evaluator.verify(
            claim=claim_breed,
            node=leaf_breed,
            sources=_combine_sources(svc_sources, qual_sources),
            additional_instruction=(
                "Accept explicit mentions of 'herding breeds', 'working dogs', or named examples like Belgian Malinois, "
                "Belgian Tervuren, Belgian Sheepdog (Groenendael), German Shepherd, Border Collie, Australian Cattle Dog."
            ),
        )

    # -------------------- Health & Safety group (critical) -------------------- #
    health_group = evaluator.add_parallel(
        id=f"facility_{idx+1}_health",
        desc="Health and safety protocols verified",
        parent=fnode,
        critical=True,
    )

    health_protocols = evaluator.add_parallel(
        id=f"facility_{idx+1}_health_protocols",
        desc="Health and safety protocol requirements (critical elements only)",
        parent=health_group,
        critical=True,
    )

    leaf_vax = evaluator.add_leaf(
        id=f"facility_{idx+1}_health_vaccination",
        desc="Vaccination requirements for dogs specified (e.g., rabies, DHPP, Bordetella)",
        parent=health_protocols,
        critical=True,
    )
    vax_list = facility.health.vaccinations or []
    vax_text = ", ".join(vax_list) if vax_list else ""
    claim_vax = (
        "The facility specifies vaccination requirements for dogs, typically including rabies, DHPP/DA2PP (distemper/parvo), "
        "and/or Bordetella (kennel cough). "
        f"Vaccinations mentioned in the answer include: {vax_text}."
    )
    await evaluator.verify(
        claim=claim_vax,
        node=leaf_vax,
        sources=hlth_sources,
        additional_instruction=(
            "Pass if the page lists any required vaccines (e.g., rabies, distemper/parvo (DHPP/DA2PP), Bordetella). "
            "It must be clear that vaccines are required for boarding/daycare/training."
        ),
    )

    health_verify = evaluator.add_parallel(
        id=f"facility_{idx+1}_health_verification",
        desc="Health protocol verification reference",
        parent=health_group,
        critical=True,
    )

    leaf_health_url = evaluator.add_leaf(
        id=f"facility_{idx+1}_health_url",
        desc="URL reference verifying health and safety protocols",
        parent=health_verify,
        critical=True,
    )
    claim_health_url = "One of these URLs lists vaccination or health/safety requirements relevant to boarding/daycare/training."
    await evaluator.verify(
        claim=claim_health_url,
        node=leaf_health_url,
        sources=hlth_sources,
        additional_instruction="Look for 'vaccination requirements', 'required vaccines', 'health policy', or similar sections.",
    )

    # Optional, non-critical: veterinary/emergency protocols (moved out of critical parents)
    if (facility.health.veterinary_emergency or "").strip():
        leaf_vet = evaluator.add_leaf(
            id=f"facility_{idx+1}_health_veterinary",
            desc="Veterinary care or emergency protocols mentioned",
            parent=fnode,
            critical=False,
        )
        vet_text = facility.health.veterinary_emergency or ""
        claim_vet = f"The facility mentions veterinary care availability or emergency protocols. Example: '{vet_text}'."
        await evaluator.verify(
            claim=claim_vet,
            node=leaf_vet,
            sources=hlth_sources,
            additional_instruction="Accept 'vet on call', 'emergency protocol', 'emergency contact', or similar.",
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
    Evaluate an answer for the NC dog facilities task.
    Returns the evaluator summary dict including the verification tree and scores.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # independent facilities
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

    # NOTE: We intentionally set root as non-critical (default) to allow partial credit across facilities
    # despite the JSON marking it critical; this avoids violating the framework rule that critical parents
    # cannot have non-critical children (facility blocks are non-critical for partial credit).

    # 1) Extract structured facilities info from the answer
    extracted: FacilitiesExtraction = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction",
    )

    # 2) Use only the first three facilities (pad with placeholders if fewer)
    facilities = _first_n_facilities(extracted, n=3)

    # 3) Build and run verification for each facility
    for i, fac in enumerate(facilities):
        await verify_one_facility(evaluator, root, fac, i)

    # 4) Return structured evaluation summary
    return evaluator.get_summary()