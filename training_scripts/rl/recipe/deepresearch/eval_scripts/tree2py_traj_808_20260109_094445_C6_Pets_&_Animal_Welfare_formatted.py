import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_no_kill_shelters_adult_dogs"
TASK_DESCRIPTION = """Find three no-kill animal shelters in California that meet the following requirements:

Shelter Requirements:
1. The shelter must have a documented save rate of 90% or higher
2. The shelter must be physically located in California
3. The shelter must provide verifiable contact information including a website URL, phone number, and physical address
4. The shelter must have public adoption hours on at least weekdays
5. The shelter must specify the following adoption documentation requirements:
   - Valid photo identification
   - Minimum age requirement (18 or 21 years old)
   - Proof of home ownership or landlord approval for renters
6. The shelter must publicly disclose its adoption fee structure
7. The shelter should have a stated return policy

Dog Requirements:
For each shelter, identify one adult dog (1 year of age or older) currently available for adoption that meets these criteria:
1. The dog must be 1 year of age or older
2. The dog must be spayed or neutered (or will be before adoption)
3. The dog must have received required vaccinations including rabies
4. The specific adoption fee for the dog must be provided
5. A direct URL link to the dog's adoption profile page must be provided

For each shelter, provide:
- Shelter name
- Physical address
- Website URL
- Phone number
- Operating/adoption hours
- Documentation of no-kill status (save rate percentage and source)
- Minimum adoption age requirement
- Adoption fee structure
- Return policy details
- For the identified dog: name, age, breed, spay/neuter status, vaccination status, adoption fee, and profile URL
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ShelterContact(BaseModel):
    website_url: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    hours: Optional[str] = None
    hours_url: Optional[str] = None
    contact_url: Optional[str] = None


class NoKillInfo(BaseModel):
    save_rate: Optional[str] = None
    save_rate_source_urls: List[str] = Field(default_factory=list)


class AdoptionRequirements(BaseModel):
    min_age: Optional[str] = None
    photo_id_required_text: Optional[str] = None
    housing_requirement_text: Optional[str] = None
    requirements_url: Optional[str] = None


class FeeStructure(BaseModel):
    fee_structure_text: Optional[str] = None
    fee_structure_url: Optional[str] = None


class ReturnPolicy(BaseModel):
    return_policy_text: Optional[str] = None
    return_policy_url: Optional[str] = None


class DogInfo(BaseModel):
    name: Optional[str] = None
    age_text: Optional[str] = None
    breed: Optional[str] = None
    spay_neuter_status: Optional[str] = None
    vaccination_status: Optional[str] = None
    adoption_fee: Optional[str] = None
    profile_url: Optional[str] = None
    availability_status: Optional[str] = None


class ShelterRecord(BaseModel):
    shelter_name: Optional[str] = None
    contact: ShelterContact = Field(default_factory=ShelterContact)
    nokill: NoKillInfo = Field(default_factory=NoKillInfo)
    requirements: AdoptionRequirements = Field(default_factory=AdoptionRequirements)
    fee: FeeStructure = Field(default_factory=FeeStructure)
    return_policy: ReturnPolicy = Field(default_factory=ReturnPolicy)
    dog: DogInfo = Field(default_factory=DogInfo)


class SheltersExtraction(BaseModel):
    shelters: List[ShelterRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shelters() -> str:
    return """
    Extract from the provided answer up to THREE (3) California no-kill shelters and, for each shelter, one dog that is currently adoptable. Return the result as a JSON object with a single field "shelters" which is an array of up to 3 objects. Each object must have the following nested fields (return null for any field not explicitly present in the answer):

    For each shelters[i]:
      shelter_name: string or null

      contact: object
        - website_url: string (full URL) or null
        - phone: string or null
        - address: string or null
        - hours: string or null
        - hours_url: string (URL for hours page if provided) or null
        - contact_url: string (URL for contact/location page if provided) or null

      nokill: object
        - save_rate: string (e.g., "92%") or null
        - save_rate_source_urls: array of strings (each a URL explicitly present in the answer); can be empty

      requirements: object
        - min_age: string (e.g., "18+" or "21") or null
        - photo_id_required_text: string (quote or paraphrase from the answer indicating valid photo ID) or null
        - housing_requirement_text: string (quote or paraphrase indicating proof of home ownership OR landlord approval) or null
        - requirements_url: string (URL for adoption requirements/policies page if provided) or null

      fee: object
        - fee_structure_text: string (the public adoption fee structure text from the answer) or null
        - fee_structure_url: string (URL where the fee structure is shown, if provided) or null

      return_policy: object
        - return_policy_text: string (details of return policy if stated) or null
        - return_policy_url: string (URL to return policy page if provided) or null

      dog: object
        - name: string or null
        - age_text: string (as written in the answer, e.g., "2 years", "Adult") or null
        - breed: string or null
        - spay_neuter_status: string or null
        - vaccination_status: string (include text about rabies if present) or null
        - adoption_fee: string (specific fee, e.g., "$150") or null
        - profile_url: string (direct link to the dog's profile, if provided) or null
        - availability_status: string (e.g., "Available", "Adoptable") or null

    Rules:
    - Extract only information explicitly present in the answer text.
    - Extract only valid and complete URLs that appear in the answer; do not invent URLs.
    - If more than 3 shelters/dogs are present, keep only the first 3 in order of appearance.
    - Preserve the exact wording where reasonable (especially for fee, requirements, and policy texts).
    - If a field is not present in the answer, set it to null (or [] for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _ensure_list(*vals: Optional[str], extra: Optional[List[str]] = None) -> List[str]:
    out: List[str] = []
    for v in vals:
        if v and isinstance(v, str) and v.strip():
            out.append(v.strip())
    if extra:
        for v in extra:
            if v and isinstance(v, str) and v.strip():
                out.append(v.strip())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in out:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _nz(text: Optional[str], default: str = "N/A") -> str:
    return text if (text is not None and str(text).strip()) else default


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_shelter_pair(
    evaluator: Evaluator,
    parent_node,
    pair_index_1based: int,
    shelter: ShelterRecord,
) -> None:
    """
    Build verification nodes for a single shelter/dog pair and run checks.
    """
    sid = pair_index_1based  # 1,2,3
    pair_node = evaluator.add_parallel(
        id=f"shelter_{sid}",
        desc=f"Shelter/Dog pair #{sid} (independently evaluable)",
        parent=parent_node,
        critical=False
    )

    # Convenience variables
    name = _nz(shelter.shelter_name, "Unknown Shelter")
    website_url = shelter.contact.website_url
    contact_url = shelter.contact.contact_url or website_url
    hours_url = shelter.contact.hours_url or website_url
    req_url = shelter.requirements.requirements_url or website_url
    fee_url = shelter.fee.fee_structure_url or website_url
    return_url = shelter.return_policy.return_policy_url or website_url

    # 1) Documented no-kill save rate >= 90% with verifiable source(s)
    node_save = evaluator.add_leaf(
        id=f"shelter_{sid}_no_kill_documentation",
        desc="Documented save rate >= 90% is provided AND a verifiable source/citation for that save rate is provided",
        parent=pair_node,
        critical=True
    )
    save_rate_txt = _nz(shelter.nokill.save_rate, "unknown")
    save_sources = shelter.nokill.save_rate_source_urls or ([] if not website_url else [website_url])
    claim_save = (
        f"The shelter '{name}' has a published save (live release) rate of {save_rate_txt}, "
        f"which is at least 90%, and at least one provided source page explicitly states this percentage for this shelter."
    )
    await evaluator.verify(
        claim=claim_save,
        node=node_save,
        sources=save_sources,
        additional_instruction=(
            "Only mark Correct if the source page explicitly mentions a save rate (also called live release rate) "
            "for this shelter and the value is >= 90%. If no working source URL is provided or the page doesn't show "
            "the number, mark Incorrect."
        ),
    )

    # 2) Shelter physically located in California
    node_loc = evaluator.add_leaf(
        id=f"shelter_{sid}_location_ca",
        desc="Shelter is physically located in California",
        parent=pair_node,
        critical=True
    )
    claim_loc = (
        f"The shelter '{name}' is physically located in California. "
        f"The address in the answer is '{_nz(shelter.contact.address)}', and the website shows a matching CA address."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=node_loc,
        sources=contact_url,
        additional_instruction=(
            "Verify that the website shows a physical address in California (e.g., contains 'CA' or 'California' "
            "with a valid city). If the page does not show a California address, mark Incorrect."
        ),
    )

    # 3) Verifiable contact info: website URL, phone number, and physical address
    node_contact = evaluator.add_leaf(
        id=f"shelter_{sid}_contact_information_complete",
        desc="Verifiable shelter contact information is provided: website URL, phone number, and physical address",
        parent=pair_node,
        critical=True
    )
    claim_contact = (
        f"The shelter '{name}' website provides both a phone number and a physical address that match the answer "
        f"(allowing formatting differences). Phone: '{_nz(shelter.contact.phone)}'; Address: '{_nz(shelter.contact.address)}'. "
        f"A website URL is also provided."
    )
    await evaluator.verify(
        claim=claim_contact,
        node=node_contact,
        sources=contact_url,
        additional_instruction=(
            "Confirm that the page shows a phone number and a street/physical address for the shelter. "
            "If the phone OR the address cannot be found on the site or no site URL is available, mark Incorrect."
        ),
    )

    # 4) Public adoption hours include at least weekdays
    node_hours = evaluator.add_leaf(
        id=f"shelter_{sid}_adoption_hours_weekdays",
        desc="Public adoption hours are provided and include at least weekdays",
        parent=pair_node,
        critical=True
    )
    claim_hours = (
        f"The shelter '{name}' publicly lists adoption/operating hours that include at least some weekdays "
        f"(Mon–Fri). Hours text from the answer: '{_nz(shelter.contact.hours)}'."
    )
    await evaluator.verify(
        claim=claim_hours,
        node=node_hours,
        sources=hours_url,
        additional_instruction=(
            "Pass only if the page shows hours including at least one weekday (Mon–Fri). "
            "If only weekends are listed or no hours are shown, mark Incorrect."
        ),
    )

    # 5) Adoption documentation requirements: photo ID + min age + housing/landlord approval
    node_require = evaluator.add_leaf(
        id=f"shelter_{sid}_adoption_documentation_requirements_complete",
        desc="Shelter specifies required adoption documentation including: valid photo ID, a minimum age requirement (18 or 21), and proof of home ownership OR landlord approval for renters",
        parent=pair_node,
        critical=True
    )
    claim_require = (
        f"The shelter '{name}' specifies all of the following in its adoption requirements: "
        f"(1) valid photo ID, (2) a minimum adopter age (18+ or 21+), and (3) either proof of home ownership or "
        f"written landlord approval for renters. Text from the answer includes: "
        f"photo ID: '{_nz(shelter.requirements.photo_id_required_text)}'; "
        f"minimum age: '{_nz(shelter.requirements.min_age)}'; "
        f"housing requirement: '{_nz(shelter.requirements.housing_requirement_text)}'."
    )
    await evaluator.verify(
        claim=claim_require,
        node=node_require,
        sources=req_url,
        additional_instruction=(
            "All three elements must be present on the cited page(s): a photo ID requirement, an explicit minimum age "
            "(18 or 21), AND either proof of home ownership or landlord approval. Accept clear synonyms. "
            "If any element is missing or ambiguous, mark Incorrect."
        ),
    )

    # 6) Adoption fee structure publicly disclosed
    node_fee_struct = evaluator.add_leaf(
        id=f"shelter_{sid}_adoption_fee_structure",
        desc="Shelter publicly discloses its adoption fee structure",
        parent=pair_node,
        critical=True
    )
    claim_fee_struct = (
        f"The shelter '{name}' publicly discloses an adoption fee structure (e.g., specific fees or fee ranges) on its website. "
        f"Answer text for fee structure: '{_nz(shelter.fee.fee_structure_text)}'."
    )
    await evaluator.verify(
        claim=claim_fee_struct,
        node=node_fee_struct,
        sources=fee_url,
        additional_instruction=(
            "The page must show explicit adoption fees or a fee schedule/structure. "
            "Statements like 'contact for fees' without numbers should fail."
        ),
    )

    # 7) Return policy (non-critical)
    node_return = evaluator.add_leaf(
        id=f"shelter_{sid}_return_policy",
        desc="Shelter states return policy details",
        parent=pair_node,
        critical=False
    )
    claim_return = (
        f"The shelter '{name}' states a return policy for adopted animals (e.g., instructions or allowed returns). "
        f"Return policy text from the answer: '{_nz(shelter.return_policy.return_policy_text)}'."
    )
    await evaluator.verify(
        claim=claim_return,
        node=node_return,
        sources=return_url,
        additional_instruction=(
            "Mark Correct only if the page includes an explicit return policy or guidance on returning adopted animals. "
            "If silent, mark Incorrect."
        ),
    )

    # Dog-related checks
    dog = shelter.dog
    dog_profile_url = dog.profile_url

    # 8) Direct URL to dog's profile
    node_dog_url = evaluator.add_leaf(
        id=f"shelter_{sid}_dog_profile_url",
        desc="Direct URL to the dog’s adoption profile page is provided",
        parent=pair_node,
        critical=True
    )
    claim_dog_url = (
        f"The provided URL is a direct link to an individual dog adoption profile page for '{_nz(dog.name)}' "
        f"at the shelter '{name}'. URL provided: '{_nz(dog_profile_url, 'None')}'."
    )
    await evaluator.verify(
        claim=claim_dog_url,
        node=node_dog_url,
        sources=dog_profile_url,
        additional_instruction=(
            "Pass only if the URL loads a specific dog's profile page (not a general listing page). "
            "If no URL was provided, or the page is a general listing, mark Incorrect."
        ),
    )

    # 9) Dog currently available for adoption
    node_dog_avail = evaluator.add_leaf(
        id=f"shelter_{sid}_dog_currently_available",
        desc="Dog is indicated as currently available for adoption (per the linked profile/listing)",
        parent=pair_node,
        critical=True
    )
    claim_dog_avail = (
        f"The profile indicates the dog is currently available for adoption (not adopted, not pending). "
        f"Availability text in the answer: '{_nz(dog.availability_status)}'."
    )
    await evaluator.verify(
        claim=claim_dog_avail,
        node=node_dog_avail,
        sources=dog_profile_url,
        additional_instruction=(
            "Look for 'Available', 'Adoptable', or similar. If 'Adopted', 'Hold', 'Pending', or no availability signal, mark Incorrect."
        ),
    )

    # 10) Dog name provided (and matches profile)
    node_dog_name = evaluator.add_leaf(
        id=f"shelter_{sid}_dog_name",
        desc="Dog name is provided",
        parent=pair_node,
        critical=True
    )
    claim_dog_name = f"The dog's name on the profile page is '{_nz(dog.name)}'."
    await evaluator.verify(
        claim=claim_dog_name,
        node=node_dog_name,
        sources=dog_profile_url,
        additional_instruction="Allow case-insensitive matching and minor punctuation differences.",
    )

    # 11) Dog breed provided (and matches profile)
    node_dog_breed = evaluator.add_leaf(
        id=f"shelter_{sid}_dog_breed",
        desc="Dog breed is provided",
        parent=pair_node,
        critical=True
    )
    claim_dog_breed = f"The dog's breed on the profile page is '{_nz(dog.breed)}' or an equivalent description (e.g., 'mix')."
    await evaluator.verify(
        claim=claim_dog_breed,
        node=node_dog_breed,
        sources=dog_profile_url,
        additional_instruction="Accept equivalent or mixed breed descriptions if clearly referring to the same dog.",
    )

    # 12) Dog age adult (>= 1 year)
    node_dog_age = evaluator.add_leaf(
        id=f"shelter_{sid}_dog_age_adult",
        desc="Dog age is stated and is >= 1 year",
        parent=pair_node,
        critical=True
    )
    claim_dog_age = (
        f"The profile indicates the dog is at least 1 year old (adult). Age shown: '{_nz(dog.age_text)}'. "
        f"Consider labels like 'Adult' or numeric ages >= 1 year as satisfying this."
    )
    await evaluator.verify(
        claim=claim_dog_age,
        node=node_dog_age,
        sources=dog_profile_url,
        additional_instruction=(
            "If the age is given in months, treat 12 months as 1 year and anything above as adult. "
            "If 'Adult' is stated, that qualifies. If unclear or <1 year, mark Incorrect."
        ),
    )

    # 13) Spay/neuter status
    node_dog_sn = evaluator.add_leaf(
        id=f"shelter_{sid}_dog_spay_neuter",
        desc="Dog is spayed/neutered or will be before adoption",
        parent=pair_node,
        critical=True
    )
    claim_dog_sn = (
        f"The profile indicates the dog is spayed/neutered or will be before adoption. "
        f"Status text: '{_nz(dog.spay_neuter_status)}'."
    )
    await evaluator.verify(
        claim=claim_dog_sn,
        node=node_dog_sn,
        sources=dog_profile_url,
        additional_instruction="Look for 'spayed', 'neutered', 'altered', or a clear statement that this will be done before adoption.",
    )

    # 14) Vaccinations including rabies
    node_dog_vax = evaluator.add_leaf(
        id=f"shelter_{sid}_dog_vaccinations_rabies",
        desc="Dog has received required vaccinations including rabies (or will by adoption, if stated as such)",
        parent=pair_node,
        critical=True
    )
    claim_dog_vax = (
        f"The profile (or shelter policy pages) indicate the dog has required vaccinations including rabies, "
        f"or will receive rabies by adoption. Status/policy text: '{_nz(dog.vaccination_status)}'."
    )
    # Include multiple sources: dog profile and requirements/policy/website where vaccination policy may be stated
    src_vax = _ensure_list(dog_profile_url, req_url, website_url)
    await evaluator.verify(
        claim=claim_dog_vax,
        node=node_dog_vax,
        sources=src_vax,
        additional_instruction=(
            "Rabies must be explicitly mentioned or clearly included in the vaccination list/policy. "
            "If only 'vaccinated' is stated without clear inclusion of rabies, mark Incorrect."
        ),
    )

    # 15) Specific adoption fee for this dog
    node_dog_fee = evaluator.add_leaf(
        id=f"shelter_{sid}_dog_specific_fee",
        desc="Specific adoption fee for this dog is provided",
        parent=pair_node,
        critical=True
    )
    claim_dog_fee = f"The dog's specific adoption fee is listed on the profile as '{_nz(dog.adoption_fee)}'."
    # Sometimes a dog's fee might be on the profile; if not, sometimes the system lists fees per dog.
    await evaluator.verify(
        claim=claim_dog_fee,
        node=node_dog_fee,
        sources=dog_profile_url,
        additional_instruction=(
            "Pass only if the profile shows a specific numeric amount for the adoption fee (e.g., with a dollar sign). "
            "If it says 'varies', 'see shelter', or no explicit amount is shown, mark Incorrect."
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
    Evaluate an answer for the California no-kill shelters & adult dog adoption task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # per rubric
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

    # Extract structured data
    extraction = await evaluator.extract(
        prompt=prompt_extract_shelters(),
        template_class=SheltersExtraction,
        extraction_name="shelters_dogs_extraction",
    )

    # Keep only first 3 shelters; pad if fewer
    shelters = list(extraction.shelters[:3])
    while len(shelters) < 3:
        shelters.append(ShelterRecord())

    # Build verification subtrees for each shelter/dog pair
    for idx in range(3):
        await verify_shelter_pair(
            evaluator=evaluator,
            parent_node=root,
            pair_index_1based=idx + 1,
            shelter=shelters[idx],
        )

    return evaluator.get_summary()