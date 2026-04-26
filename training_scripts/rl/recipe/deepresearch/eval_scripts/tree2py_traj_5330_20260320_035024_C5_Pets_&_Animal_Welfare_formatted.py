import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "buffalo_pet_adoption_research"
TASK_DESCRIPTION = (
    "I am planning to adopt a dog or cat in Buffalo, New York, and I would like to compare my options. "
    "Research three different animal shelters or rescue organizations in the Buffalo, NY area that offer dog or cat adoptions. "
    "For each shelter, provide the following information:\n\n"
    "1. Identification: The shelter's official name, physical street address, and official website URL.\n"
    "2. Adoption Requirements: The minimum age required to adopt; whether government-issued photo ID is required; "
    "whether veterinary references are required for applicants who have or previously had pets; whether landlord verification is required for renters.\n"
    "3. Fees and Contact: The typical adoption fee range for adult dogs or cats (not puppies or kittens); a phone number or email address for inquiries.\n\n"
    "Ensure that all three shelters you select are distinct organizations operating in the Buffalo, New York area, and that all information is current "
    "and publicly available on their official websites or verified sources."
)


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class ShelterItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    website: Optional[str] = None

    # Adoption requirements
    min_adoption_age: Optional[str] = None
    photo_id_required: Optional[str] = None            # expected values: "required", "not required", or "not specified"
    vet_references_required: Optional[str] = None      # expected values: "required", "not required", or "not specified"
    landlord_verification_required: Optional[str] = None  # expected values: "required", "not required", or "not specified"

    # Fees and contact
    adult_dog_fee_range: Optional[str] = None
    adult_cat_fee_range: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None

    # Source URLs explicitly cited in the answer for this shelter (e.g., homepage, adoption policy, fees page, contact page)
    sources: List[str] = Field(default_factory=list)


class SheltersExtraction(BaseModel):
    shelters: List[ShelterItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shelters() -> str:
    return """
Extract up to three distinct animal shelters or rescue organizations in the Buffalo, NY area that the answer presents as options for adopting a dog or cat.

For each shelter, return an object with the following fields:
- name: The organization's official name as written in the answer.
- address: The physical street address in the Buffalo, NY area exactly as written in the answer (include city/state/ZIP if provided).
- website: The organization's official website URL as written in the answer (full URL, including http/https).
- min_adoption_age: The minimum age required to adopt (e.g., "18", "21", "18 years or older"). If not clear, set "not specified".
- photo_id_required: Whether government-issued photo ID is required; use one of: "required", "not required", "not specified".
- vet_references_required: Whether veterinary references are required for applicants who have or previously had pets; use one of: "required", "not required", "not specified".
- landlord_verification_required: Whether landlord verification/permission is required for renters; use one of: "required", "not required", "not specified".
- adult_dog_fee_range: The typical adoption fee (or range) for ADULT dogs (not puppies). If not present, set null.
- adult_cat_fee_range: The typical adoption fee (or range) for ADULT cats (not kittens). If not present, set null.
- contact_phone: A phone number for inquiries if provided. If none, set null.
- contact_email: An email address for inquiries if provided. If none, set null.
- sources: A list of ALL URLs explicitly cited in the answer that pertain to this shelter (e.g., homepage, adoption policy, fee page, contact page). Do not invent URLs. Include only URLs actually present in the answer text (plain links or markdown links). If none, return an empty array.

GENERAL RULES:
- Do not infer or fabricate any information not in the answer text.
- Only extract URLs explicitly mentioned in the answer.
- Normalize yes/no requirement fields to the requested values.
- If the answer lists more than 3 organizations, only return the first 3 in the same order.
- If fewer than 3 organizations are presented, return as many as present.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(n: int) -> str:
    return ["first", "second", "third", "fourth", "fifth"][n] if 0 <= n < 5 else f"#{n+1}"


def _normalize_req_value(v: Optional[str]) -> str:
    if not v:
        return "not specified"
    s = v.strip().lower()
    truthy = {"required", "yes", "y", "true", "must", "mandatory", "gov id required", "id required"}
    falsy = {"not required", "no", "n", "false", "not mandatory", "optional"}
    if s in truthy:
        return "required"
    if s in falsy:
        return "not required"
    # try to detect patterns
    if "require" in s or "id" in s or "license" in s:
        # defer to model judgment by keeping original; but prefer "required" if clearly affirmative
        if "not" in s or "no " in s:
            return "not required"
        return "required"
    return "not specified"


def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _gather_sources(shelter: ShelterItem) -> List[str]:
    # Prefer all cited sources; ensure the official website is included if present
    return _dedup_urls(([shelter.website] if shelter.website else []) + (shelter.sources or []))


def _names_websites_distinct(shelters: List[ShelterItem]) -> bool:
    def norm_name(x: Optional[str]) -> Optional[str]:
        return re.sub(r"\W+", "", x.lower()) if x else None

    def norm_site(x: Optional[str]) -> Optional[str]:
        return x.lower().rstrip("/") if x else None

    normalized_names = [norm_name(s.name) for s in shelters if s]
    normalized_sites = [norm_site(s.website) for s in shelters if s]

    # Consider only non-empty entries
    filtered_names = [n for n in normalized_names if n]
    filtered_sites = [u for u in normalized_sites if u]

    # If fewer than 2 names present, cannot confirm distinctness reliably; return False (fail)
    if len(filtered_names) < 2 and len(filtered_sites) < 2:
        return False

    # Check duplicates among names or among sites
    if len(set(filtered_names)) != len(filtered_names):
        return False
    if len(set(filtered_sites)) != len(filtered_sites):
        return False
    return True


# --------------------------------------------------------------------------- #
# Verification for a single shelter                                           #
# --------------------------------------------------------------------------- #
async def verify_single_shelter(
    evaluator: Evaluator,
    parent_node,
    shelter: ShelterItem,
    index: int,
) -> None:
    human_idx = index + 1
    ord_str = _ordinal(index)

    # Container node for this shelter (parallel, non-critical as per rubric)
    shelter_node = evaluator.add_parallel(
        id=f"shelter_{human_idx}",
        desc=f"Complete information about the {ord_str} animal shelter or rescue organization",
        parent=parent_node,
        critical=False
    )

    sources = _gather_sources(shelter)

    # 1) Identification (critical)
    ident_node = evaluator.add_parallel(
        id=f"shelter_{human_idx}_identification",
        desc=f"Basic identification information for the {ord_str} shelter",
        parent=shelter_node,
        critical=True
    )

    # 1.a) Official Name
    name_leaf = evaluator.add_leaf(
        id=f"shelter_{human_idx}_name",
        desc="Provide the official name of the shelter or rescue organization",
        parent=ident_node,
        critical=True
    )
    name_claim = f"The official name of the organization is '{shelter.name or ''}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=sources,
        additional_instruction="Confirm that the cited page(s) clearly present this exact organization name as the official name. "
                               "Allow minor variations (e.g., Inc., Rescue, SPCA), case differences, or punctuation."
    )

    # 1.b) Physical Address in Buffalo area
    address_leaf = evaluator.add_leaf(
        id=f"shelter_{human_idx}_address",
        desc="Provide the physical street address in Buffalo, NY area",
        parent=ident_node,
        critical=True
    )
    address_claim = (
        f"The physical street address of the organization is '{shelter.address or ''}', and this address is in the Buffalo, NY area."
    )
    await evaluator.verify(
        claim=address_claim,
        node=address_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the street address text and confirm it is located in the Buffalo, New York area. "
            "Accept the City of Buffalo or nearby Erie County suburbs (e.g., Amherst, Cheektowaga, Tonawanda, West Seneca, Orchard Park, "
            "Hamburg, Williamsville, Lackawanna, Kenmore, Depew, Lancaster). If multiple locations are listed, focus on the Buffalo-area location."
        )
    )

    # 1.c) Official Website URL
    website_leaf = evaluator.add_leaf(
        id=f"shelter_{human_idx}_website",
        desc="Provide the official website URL",
        parent=ident_node,
        critical=True
    )
    website_claim = f"This URL is the official website of the organization named '{shelter.name or ''}': {shelter.website or ''}"
    await evaluator.verify(
        claim=website_claim,
        node=website_leaf,
        sources=shelter.website if shelter.website else sources,
        additional_instruction="Check branding (name/logo), header/footer, or About page to confirm this is the organization's official site."
    )

    # 2) Adoption Requirements (critical)
    req_node = evaluator.add_parallel(
        id=f"shelter_{human_idx}_adoption_requirements",
        desc=f"Documented adoption requirements for the {ord_str} shelter",
        parent=shelter_node,
        critical=True
    )

    # Normalize requirement flags for more robust claims
    photo_id_norm = _normalize_req_value(shelter.photo_id_required)
    vet_ref_norm = _normalize_req_value(shelter.vet_references_required)
    landlord_norm = _normalize_req_value(shelter.landlord_verification_required)

    # 2.a) Minimum Age
    age_leaf = evaluator.add_leaf(
        id=f"shelter_{human_idx}_age_requirement",
        desc="State the minimum age required to adopt (typically 18 or 21)",
        parent=req_node,
        critical=True
    )
    age_claim = f"The minimum age required to adopt is '{shelter.min_adoption_age or ''}'."
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=sources,
        additional_instruction="Confirm the minimum adopter age stated on adoption policy, FAQs, or application pages. "
                               "If multiple programs differ, use the general adoption program's minimum age."
    )

    # 2.b) Government-issued photo ID required?
    id_req_leaf = evaluator.add_leaf(
        id=f"shelter_{human_idx}_id_requirement",
        desc="Specify whether government-issued photo ID is required",
        parent=req_node,
        critical=True
    )
    if photo_id_norm == "required":
        id_req_claim = "Government-issued photo ID is required for adopters."
    elif photo_id_norm == "not required":
        id_req_claim = "Government-issued photo ID is not required for adopters."
    else:
        id_req_claim = "The provided source pages do not specify whether a government-issued photo ID is required for adopters."
    await evaluator.verify(
        claim=id_req_claim,
        node=id_req_leaf,
        sources=sources,
        additional_instruction="Check adoption requirements or application instructions for explicit 'ID required' language. "
                               "If the pages make no mention of photo ID, consider the claim 'not specified'."
    )

    # 2.c) Veterinary references required?
    vet_ref_leaf = evaluator.add_leaf(
        id=f"shelter_{human_idx}_vet_reference",
        desc="Indicate whether veterinary references are required for applicants with current or previous pets",
        parent=req_node,
        critical=True
    )
    if vet_ref_norm == "required":
        vet_req_claim = "Veterinary references are required for applicants who have or previously had pets."
    elif vet_ref_norm == "not required":
        vet_req_claim = "Veterinary references are not required for applicants who have or previously had pets."
    else:
        vet_req_claim = "The provided source pages do not specify whether veterinary references are required for applicants."
    await evaluator.verify(
        claim=vet_req_claim,
        node=vet_ref_leaf,
        sources=sources,
        additional_instruction="Look for application or policy text referencing 'veterinary reference', 'current vet', or similar."
    )

    # 2.d) Landlord verification required for renters?
    landlord_leaf = evaluator.add_leaf(
        id=f"shelter_{human_idx}_landlord_verification",
        desc="Indicate whether landlord verification is required for renters",
        parent=req_node,
        critical=True
    )
    if landlord_norm == "required":
        landlord_claim = "Landlord permission/verification is required for renters to adopt."
    elif landlord_norm == "not required":
        landlord_claim = "Landlord permission/verification is not required for renters to adopt."
    else:
        landlord_claim = "The provided source pages do not specify whether landlord permission/verification is required for renters."
    await evaluator.verify(
        claim=landlord_claim,
        node=landlord_leaf,
        sources=sources,
        additional_instruction="Check for policy text requiring rental lease, landlord permission letter, or landlord contact verification."
    )

    # 3) Fees & Contact (critical)
    fees_node = evaluator.add_parallel(
        id=f"shelter_{human_idx}_fees_contact",
        desc=f"Fee information and contact details for the {ord_str} shelter",
        parent=shelter_node,
        critical=True
    )

    # 3.a) Adoption fees (adult dogs or cats)
    fees_leaf = evaluator.add_leaf(
        id=f"shelter_{human_idx}_adoption_fees",
        desc="Provide typical adoption fee ranges for adult dogs or cats",
        parent=fees_node,
        critical=True
    )
    dog_fee = shelter.adult_dog_fee_range or ""
    cat_fee = shelter.adult_cat_fee_range or ""
    if dog_fee and cat_fee:
        fees_claim = f"The typical adoption fees are: adult dogs: {dog_fee}; adult cats: {cat_fee} (not puppies/kittens)."
    elif dog_fee:
        fees_claim = f"The typical adoption fee for adult dogs (not puppies) is {dog_fee}."
    elif cat_fee:
        fees_claim = f"The typical adoption fee for adult cats (not kittens) is {cat_fee}."
    else:
        fees_claim = "The provided source pages do not specify typical adoption fees for adult dogs or adult cats."
    await evaluator.verify(
        claim=fees_claim,
        node=fees_leaf,
        sources=sources,
        additional_instruction="Confirm fees for adult animals only; ignore puppies/kittens specials. "
                               "Accept reasonable rounding or ranges if consistent with the cited pages."
    )

    # 3.b) Contact: phone or email
    contact_leaf = evaluator.add_leaf(
        id=f"shelter_{human_idx}_contact",
        desc="Provide a phone number or email address for inquiries",
        parent=fees_node,
        critical=True
    )
    if shelter.contact_phone and shelter.contact_email:
        contact_claim = f"The organization lists the following contacts for inquiries: phone {shelter.contact_phone}, email {shelter.contact_email}."
    elif shelter.contact_phone:
        contact_claim = f"The organization lists the following contact for inquiries: phone {shelter.contact_phone}."
    elif shelter.contact_email:
        contact_claim = f"The organization lists the following contact for inquiries: email {shelter.contact_email}."
    else:
        contact_claim = "The provided source pages do not list a phone number or email address for adoption inquiries."
    await evaluator.verify(
        claim=contact_claim,
        node=contact_leaf,
        sources=sources,
        additional_instruction="Confirm at least one working contact method (phone or email) from the organization's official site. "
                               "General shelter contact is acceptable if used for adoption inquiries."
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extract shelter information from the answer
    extracted: SheltersExtraction = await evaluator.extract(
        prompt=prompt_extract_shelters(),
        template_class=SheltersExtraction,
        extraction_name="shelters_extraction",
    )

    # Normalize to exactly 3 slots
    shelters: List[ShelterItem] = list(extracted.shelters or [])
    shelters = shelters[:3]
    while len(shelters) < 3:
        shelters.append(ShelterItem())

    # Add a distinct organizations check (critical to the overall task)
    distinct_node = evaluator.add_custom_node(
        result=_names_websites_distinct(shelters),
        id="distinct_organizations",
        desc="All three shelters are distinct organizations (no duplicate names or websites)",
        parent=root,
        critical=True,
    )

    # Build per-shelter verification subtrees
    tasks = []
    for i in range(3):
        tasks.append(verify_single_shelter(evaluator, root, shelters[i], i))

    await asyncio.gather(*tasks)

    # Return the evaluator's standard summary
    return evaluator.get_summary()