import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pa_no_kill_shelters_4"
TASK_DESCRIPTION = """
Find 4 no-kill animal shelters located in Pennsylvania that meet the following requirements:

1. No-Kill Status: The shelter must explicitly identify as a no-kill shelter OR state that they achieve a 90% or higher save rate (also called live release rate) OR provide adoption statistics demonstrating 90% or better positive outcomes for animals in their care.

2. Pennsylvania Location: The shelter must be physically located in Pennsylvania, with a verifiable address, city, or service area within the state.

3. Adoption Process Information: The shelter must publicly provide:
   - The minimum age requirement for adopters (typically 18 or 21 years old)
   - A description of their adoption application process or procedure

4. Pre-Adoption Veterinary Services: The shelter must provide the following veterinary services to animals before adoption:
   - Spay/neuter surgery
   - Vaccinations, including:
     * Rabies vaccination (for dogs 4 months and older)
     * Core vaccinations (such as distemper/parvo for dogs or FVRCP for cats)
   - Microchipping

5. Contact Information: The shelter should provide publicly accessible contact information and adoption hours (preferred but not required).

For each shelter, provide:
- Shelter name
- Specific evidence of no-kill status (statement, statistics, or policy)
- Pennsylvania location details (city and/or address)
- Adoption age requirement
- Description of adoption process
- Confirmation of each required pre-adoption veterinary service
- Contact information and hours (if available)
- URL references supporting each piece of information

All information must be verifiable through the shelter's official website or publicly available sources.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ShelterItem(BaseModel):
    # Identification
    name: Optional[str] = None
    website_url: Optional[str] = None

    # No-kill evidence
    no_kill_evidence: Optional[str] = None
    no_kill_urls: List[str] = Field(default_factory=list)

    # Location
    location_details: Optional[str] = None  # e.g., "123 Main St, Pittsburgh, PA 15222" or "Erie, PA"
    location_urls: List[str] = Field(default_factory=list)

    # Adoption process
    age_requirement: Optional[str] = None  # e.g., "18+", "21 years or older", "must be 18"
    adoption_process: Optional[str] = None  # brief description from the answer
    adoption_urls: List[str] = Field(default_factory=list)

    # Veterinary services prior to adoption
    spay_neuter: Optional[str] = None
    rabies_vaccination: Optional[str] = None
    core_vaccinations: Optional[str] = None
    microchipping: Optional[str] = None
    vet_urls: List[str] = Field(default_factory=list)           # general vet/services/adoption package URL(s)
    vaccination_urls: List[str] = Field(default_factory=list)   # vaccination-specific page(s) if any

    # Contact and hours (non-critical)
    contact_info: Optional[str] = None          # phone/email/contact method summary from the answer
    adoption_hours: Optional[str] = None        # hours or scheduling info
    contact_urls: List[str] = Field(default_factory=list)


class SheltersExtraction(BaseModel):
    shelters: List[ShelterItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shelters() -> str:
    return """
    Extract up to 6 candidate shelters mentioned in the answer (we will take the first 4).
    For each shelter, extract the following fields exactly as they appear in the answer; do not invent:
    - name: Shelter name
    - website_url: The primary official website URL if present
    - no_kill_evidence: The statement, save rate, or statistics supporting no-kill status as quoted/summarized in the answer
    - no_kill_urls: URL(s) cited for the no-kill evidence
    - location_details: City/address/region indicating a location in Pennsylvania (PA)
    - location_urls: URL(s) cited to support the Pennsylvania location
    - age_requirement: The minimum adopter age value/statement (e.g., "18+", "21 years or older")
    - adoption_process: Brief description of the adoption process/application as given
    - adoption_urls: URL(s) cited for adoption policy/process/requirements
    - spay_neuter: Any phrase indicating spay/neuter is done before adoption (if provided)
    - rabies_vaccination: Any phrase indicating rabies vaccination is provided before adoption (if provided)
    - core_vaccinations: Any phrase indicating core vaccinations (e.g., DHPP/DA2PP for dogs, FVRCP for cats) are provided before adoption (if provided)
    - microchipping: Any phrase indicating microchipping is provided before adoption (if provided)
    - vet_urls: URL(s) cited for pre-adoption veterinary services/packages (can include adoption package pages)
    - vaccination_urls: URL(s) specifically about vaccinations if distinct (optional)
    - contact_info: Any phone/email/contact method the answer lists (optional)
    - adoption_hours: Any adoption hours or scheduling info the answer lists (optional)
    - contact_urls: URL(s) cited for contact info or hours (optional)

    RULES:
    - Only include URLs that are explicitly present in the answer text (plain or markdown links). Do not invent URLs.
    - If a URL is missing a protocol, prepend http://
    - If a field is not present, set it to null or an empty list as appropriate.
    - Return as JSON with a top-level 'shelters' array of objects using the exact field names above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_seq(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _as_list(x: Optional[Any]) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    return [str(x).strip()]


def merge_sources(*args: Any) -> List[str]:
    all_urls: List[str] = []
    for a in args:
        all_urls.extend(_as_list(a))
    return _unique_seq(all_urls)


def has_any_url(urls: List[str]) -> bool:
    return bool(urls and len(urls) > 0)


def ordinal(idx: int) -> str:
    mapping = {0: "First", 1: "Second", 2: "Third", 3: "Fourth", 4: "Fifth", 5: "Sixth"}
    return mapping.get(idx, f"#{idx + 1}")


def _quoted_fragment(prefix: str, fragment: Optional[str]) -> str:
    if fragment and str(fragment).strip():
        return f"{prefix} Specifically cited in the answer: \"{fragment.strip()}\"."
    return prefix


# --------------------------------------------------------------------------- #
# Verification for one shelter                                                #
# --------------------------------------------------------------------------- #
async def verify_shelter(
    evaluator: Evaluator,
    parent_node,
    shelter: ShelterItem,
    index: int,
) -> None:
    name = shelter.name or f"Shelter #{index + 1}"

    # Top-level node for this shelter (non-critical to allow partial credit across shelters)
    shelter_node = evaluator.add_parallel(
        id=f"shelter_{index + 1}",
        desc=f"{ordinal(index)} qualifying no-kill shelter in Pennsylvania",
        parent=parent_node,
        critical=False,
    )

    # 1) No-Kill Status (critical)
    nk_node = evaluator.add_parallel(
        id=f"shelter_{index + 1}_no_kill_status",
        desc="Verification of no-kill shelter status",
        parent=shelter_node,
        critical=True,
    )
    nk_sources = merge_sources(shelter.no_kill_urls, shelter.adoption_urls, shelter.website_url)

    nk_url_ref = evaluator.add_custom_node(
        result=has_any_url(nk_sources),
        id=f"shelter_{index + 1}_no_kill_url_ref",
        desc="URL reference supporting no-kill status verification",
        parent=nk_node,
        critical=True,
    )

    nk_leaf = evaluator.add_leaf(
        id=f"shelter_{index + 1}_save_rate_statement",
        desc="Shelter explicitly claims 'no-kill' or ≥90% save/live release rate, or provides ≥90% outcome statistics",
        parent=nk_node,
        critical=True,
    )
    nk_claim_base = (
        f"At the provided source(s), the shelter '{name}' explicitly identifies as a no-kill shelter "
        f"OR reports a 90% or higher save/live release rate OR provides adoption/outcome statistics "
        f"showing 90% or higher positive outcomes."
    )
    nk_claim = _quoted_fragment(nk_claim_base, shelter.no_kill_evidence)
    await evaluator.verify(
        claim=nk_claim,
        node=nk_leaf,
        sources=nk_sources,
        additional_instruction="Require explicit textual support on the cited page(s). Accept synonyms like 'no kill', 'no‑kill'. For statistics, confirm ≥90% live release/save rate or outcomes.",
        extra_prerequisites=[nk_url_ref],
    )

    # 2) Pennsylvania Location (critical)
    loc_node = evaluator.add_parallel(
        id=f"shelter_{index + 1}_pa_location",
        desc="Verification of Pennsylvania location",
        parent=shelter_node,
        critical=True,
    )
    loc_sources = merge_sources(shelter.location_urls, shelter.website_url, shelter.contact_urls, shelter.adoption_urls)

    loc_url_ref = evaluator.add_custom_node(
        result=has_any_url(loc_sources),
        id=f"shelter_{index + 1}_location_url_ref",
        desc="URL reference supporting Pennsylvania location",
        parent=loc_node,
        critical=True,
    )

    pa_state_leaf = evaluator.add_leaf(
        id=f"shelter_{index + 1}_pa_state_confirm",
        desc="Shelter location confirmed to be in Pennsylvania",
        parent=loc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The shelter '{name}' is located in Pennsylvania (PA).",
        node=pa_state_leaf,
        sources=loc_sources,
        additional_instruction="Look for 'PA' or 'Pennsylvania' in address, city/state lines, or service area statements.",
        extra_prerequisites=[loc_url_ref],
    )

    loc_detail_leaf = evaluator.add_leaf(
        id=f"shelter_{index + 1}_physical_location_details",
        desc="City, address, or service area in Pennsylvania provided",
        parent=loc_node,
        critical=True,
    )
    loc_detail_claim = _quoted_fragment(
        "The shelter publicly lists a city, address, or service area within Pennsylvania.",
        shelter.location_details,
    )
    await evaluator.verify(
        claim=loc_detail_claim,
        node=loc_detail_leaf,
        sources=loc_sources,
        additional_instruction="The page should show a concrete city/address/area in PA (e.g., 'Pittsburgh, PA', zip codes, or street addresses).",
        extra_prerequisites=[loc_url_ref],
    )

    # 3) Adoption Process Information (critical)
    adopt_node = evaluator.add_parallel(
        id=f"shelter_{index + 1}_adoption_process",
        desc="Required adoption process details",
        parent=shelter_node,
        critical=True,
    )
    adopt_sources = merge_sources(shelter.adoption_urls, shelter.website_url)

    adopt_url_ref = evaluator.add_custom_node(
        result=has_any_url(adopt_sources),
        id=f"shelter_{index + 1}_adoption_process_url",
        desc="URL reference for adoption process information",
        parent=adopt_node,
        critical=True,
    )

    age_leaf = evaluator.add_leaf(
        id=f"shelter_{index + 1}_age_requirement",
        desc="Minimum age requirement for adopters stated (typically 18 or 21)",
        parent=adopt_node,
        critical=True,
    )
    age_claim = (
        f"The shelter states a minimum adopter age requirement"
        + (f" of {shelter.age_requirement.strip()}." if shelter.age_requirement else ".")
    )
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=adopt_sources,
        additional_instruction="Pass only if the page explicitly shows a minimum adopter age (e.g., 'must be 18', '21+'). Minor phrasing variants acceptable.",
        extra_prerequisites=[adopt_url_ref],
    )

    appl_leaf = evaluator.add_leaf(
        id=f"shelter_{index + 1}_application_process",
        desc="Description of adoption application or process provided",
        parent=adopt_node,
        critical=True,
    )
    appl_claim = _quoted_fragment(
        "The website provides a description of the adoption application/process (e.g., steps, forms, review, home check, fees).",
        shelter.adoption_process,
    )
    await evaluator.verify(
        claim=appl_claim,
        node=appl_leaf,
        sources=adopt_sources,
        additional_instruction="A clear process description on the page is required; summaries/FAQ/policy pages acceptable.",
        extra_prerequisites=[adopt_url_ref],
    )

    # 4) Pre-Adoption Veterinary Services (critical)
    vet_node = evaluator.add_parallel(
        id=f"shelter_{index + 1}_vet_services",
        desc="Veterinary services provided before adoption",
        parent=shelter_node,
        critical=True,
    )
    vet_sources = merge_sources(shelter.vet_urls, shelter.vaccination_urls, shelter.adoption_urls, shelter.website_url)

    vet_url_ref = evaluator.add_custom_node(
        result=has_any_url(vet_sources),
        id=f"shelter_{index + 1}_vet_services_url",
        desc="URL reference for veterinary services information",
        parent=vet_node,
        critical=True,
    )

    spay_leaf = evaluator.add_leaf(
        id=f"shelter_{index + 1}_spay_neuter",
        desc="Shelter provides spay/neuter before adoption",
        parent=vet_node,
        critical=True,
    )
    spay_claim = _quoted_fragment(
        "Animals are spayed/neutered prior to adoption (or included as part of the adoption package/fee).",
        shelter.spay_neuter,
    )
    await evaluator.verify(
        claim=spay_claim,
        node=spay_leaf,
        sources=vet_sources,
        additional_instruction="Require explicit statement of spay/neuter being done pre-adoption or included with adoption.",
        extra_prerequisites=[vet_url_ref],
    )

    vacc_node = evaluator.add_parallel(
        id=f"shelter_{index + 1}_vaccination_services",
        desc="Vaccinations provided before adoption",
        parent=vet_node,
        critical=True,
    )
    vacc_sources = merge_sources(shelter.vaccination_urls, shelter.vet_urls, shelter.adoption_urls, shelter.website_url)

    vacc_url_ref = evaluator.add_custom_node(
        result=has_any_url(vacc_sources),
        id=f"shelter_{index + 1}_vaccination_url_ref",
        desc="URL reference for vaccination information",
        parent=vacc_node,
        critical=True,
    )

    rabies_leaf = evaluator.add_leaf(
        id=f"shelter_{index + 1}_rabies_vaccination",
        desc="Rabies vaccination provided (for dogs 4+ months)",
        parent=vacc_node,
        critical=True,
    )
    rabies_claim = _quoted_fragment(
        "Rabies vaccination is provided prior to adoption (for dogs ≥4 months, as applicable).",
        shelter.rabies_vaccination,
    )
    await evaluator.verify(
        claim=rabies_claim,
        node=rabies_leaf,
        sources=vacc_sources,
        additional_instruction="Accept statements like 'rabies vaccine included' or policy clarifying timing for puppies <4 months.",
        extra_prerequisites=[vacc_url_ref, vet_url_ref],
    )

    core_leaf = evaluator.add_leaf(
        id=f"shelter_{index + 1}_core_vaccinations",
        desc="Core vaccinations provided (e.g., distemper/parvo for dogs, FVRCP for cats)",
        parent=vacc_node,
        critical=True,
    )
    core_claim = _quoted_fragment(
        "Core vaccinations (e.g., DA2PP/DHPP for dogs and/or FVRCP for cats) are provided prior to adoption.",
        shelter.core_vaccinations,
    )
    await evaluator.verify(
        claim=core_claim,
        node=core_leaf,
        sources=vacc_sources,
        additional_instruction="Allow common synonyms: DA2PP, DHPP, 'distemper/parvo' for dogs; FVRCP for cats.",
        extra_prerequisites=[vacc_url_ref, vet_url_ref],
    )

    micro_leaf = evaluator.add_leaf(
        id=f"shelter_{index + 1}_microchipping",
        desc="Microchipping service provided before adoption",
        parent=vet_node,
        critical=True,
    )
    micro_claim = _quoted_fragment(
        "Microchipping is provided prior to adoption.",
        shelter.microchipping,
    )
    await evaluator.verify(
        claim=micro_claim,
        node=micro_leaf,
        sources=vet_sources,
        additional_instruction="Accept 'microchip included' or equivalent wording indicating it's done before or included with adoption.",
        extra_prerequisites=[vet_url_ref],
    )

    # 5) Contact and Accessibility (non-critical)
    contact_node = evaluator.add_parallel(
        id=f"shelter_{index + 1}_contact_access",
        desc="Public accessibility and contact information",
        parent=shelter_node,
        critical=False,
    )
    contact_sources = merge_sources(shelter.contact_urls, shelter.website_url, shelter.adoption_urls)

    contact_url_ref = evaluator.add_custom_node(
        result=has_any_url(contact_sources),
        id=f"shelter_{index + 1}_contact_url_ref",
        desc="URL reference for contact information",
        parent=contact_node,
        critical=False,
    )

    hours_leaf = evaluator.add_leaf(
        id=f"shelter_{index + 1}_adoption_hours",
        desc="Adoption hours or scheduling information provided",
        parent=contact_node,
        critical=False,
    )
    hours_claim = _quoted_fragment(
        "Adoption hours or scheduling/appointment information is publicly provided.",
        shelter.adoption_hours,
    )
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=contact_sources,
        additional_instruction="Look for open hours, appointment scheduling notes, or adoption event times.",
        extra_prerequisites=[contact_url_ref],
    )

    contact_leaf = evaluator.add_leaf(
        id=f"shelter_{index + 1}_contact_information",
        desc="Phone, email, or contact method provided",
        parent=contact_node,
        critical=False,
    )
    contact_claim = _quoted_fragment(
        "Public contact information (phone, email, contact form, or address) is provided.",
        shelter.contact_info,
    )
    await evaluator.verify(
        claim=contact_claim,
        node=contact_leaf,
        sources=contact_sources,
        additional_instruction="Any obvious contact method on the page suffices (phone, email, contact form, or listed address).",
        extra_prerequisites=[contact_url_ref],
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
    # Initialize evaluator (root is non-critical by design to allow partial credit across shelters)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # shelters are independent
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

    # Extract structured shelter info
    extracted = await evaluator.extract(
        prompt=prompt_extract_shelters(),
        template_class=SheltersExtraction,
        extraction_name="shelters_extraction",
    )

    # Ensure exactly 4 items (pad with empty placeholders if needed)
    shelters: List[ShelterItem] = list(extracted.shelters or [])
    if len(shelters) > 4:
        shelters = shelters[:4]
    while len(shelters) < 4:
        shelters.append(ShelterItem())

    # Add a compact requirements note
    evaluator.add_custom_info(
        info={
            "required_shelters": 4,
            "must_have": [
                "No‑kill OR ≥90% save/live release rate (with source)",
                "Location in Pennsylvania (with source)",
                "Adoption age + process (with source)",
                "Pre‑adoption vet services: spay/neuter, vaccinations (rabies + core), microchip (with source)"
            ],
            "preferred": ["Contact info and adoption hours (with source)"]
        },
        info_type="requirements",
        info_name="requirements_summary"
    )

    # Build verification tree per shelter
    for idx in range(4):
        await verify_shelter(evaluator, root, shelters[idx], idx)

    # Return standardized summary
    return evaluator.get_summary()