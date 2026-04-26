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
TASK_ID = "ri_legal_aid_resources"
TASK_DESCRIPTION = (
    "I am seeking legal assistance in Rhode Island and need to identify appropriate resources for two different types "
    "of legal issues. Please identify two distinct nonprofit legal aid organizations or pro bono referral services "
    "located in Rhode Island. One organization must provide family law services (such as divorce, child custody, or "
    "domestic relations matters), and the other must provide housing law services (such as eviction defense, tenant "
    "rights, or landlord-tenant disputes). These must be two different organizations or services. For each of the two "
    "organizations, provide the following information: (1) The official name of the organization, (2) Confirmation that "
    "it is a nonprofit legal aid organization or pro bono referral service (not a private for-profit law firm), "
    "(3) The complete physical address, including street address, city, state, and ZIP code, (4) The phone number for "
    "contacting the organization, (5) Verification that the organization serves the specified practice area (family law "
    "for the first, housing law for the second), (6) The eligibility requirements for their services, such as income "
    "limits (expressed as a percentage of Federal Poverty Guidelines if applicable) or whether free consultation is available"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OrgExtraction(BaseModel):
    """Information extracted for a single organization from the answer text."""
    name: Optional[str] = None
    org_kind: Optional[str] = None  # e.g., "nonprofit legal aid", "pro bono referral service"
    is_nonprofit_or_referral: Optional[bool] = None  # True if nonprofit legal aid or pro bono referral service
    address_street: Optional[str] = None
    address_city: Optional[str] = None
    address_state: Optional[str] = None  # Expect "RI" or "Rhode Island"
    address_zip: Optional[str] = None
    phone: Optional[str] = None
    practice_area_verified: Optional[str] = None  # e.g., "family law", "housing law"
    eligibility: Optional[str] = None  # free text describing eligibility or "free consultation"
    source_urls: List[str] = Field(default_factory=list)  # URLs cited in the answer for this org


class LegalAidExtraction(BaseModel):
    """Top-level extraction holding two distinct orgs: one for family and one for housing."""
    family: Optional[OrgExtraction] = None
    housing: Optional[OrgExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_legal_aid_resources() -> str:
    return """
Extract details for exactly two Rhode Island legal assistance organizations referenced in the answer:
- One must be for family law services (e.g., divorce, child custody, domestic relations) → put in the `family` slot.
- One must be for housing law services (e.g., eviction defense, tenant rights, landlord-tenant disputes) → put in the `housing` slot.
They must be different organizations.

For each of the two organizations, extract the following fields exactly as presented in the answer:
- name: Official organization name
- org_kind: Short description of the type (e.g., "nonprofit legal aid", "legal services nonprofit", "pro bono referral service")
- is_nonprofit_or_referral: true if the answer makes clear it is a nonprofit legal aid org or a pro bono referral service; false if clearly a private for‑profit law firm; null if unclear
- address_street
- address_city
- address_state (use "RI" or "Rhode Island" if provided; otherwise null)
- address_zip
- phone
- practice_area_verified: put "family law" for the family org; "housing law" for the housing org; if the answer indicates otherwise, write what is indicated; if unspecified, null
- eligibility: free-text summary of eligibility requirements (e.g., income limits as % of FPL, residency, case type) OR "free consultation" if that is the relevant policy; null if not provided
- source_urls: array of all URLs explicitly mentioned in the answer for that organization (extract actual URLs; include homepage or specific service pages; allow markdown links, plain URLs, or "Sources" section)

Rules:
- Only extract data explicitly present in the answer text. Do not invent or infer missing fields.
- If a field is missing, set it to null (or empty array for URLs).
- If the answer mentions more than two organizations, select the first suitable one for each required practice area.
- If the answer provides fewer than two suitable organizations, fill what is available and set the missing one to null.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def build_address_str(org: Optional[OrgExtraction]) -> str:
    if not org:
        return ""
    parts = []
    if org.address_street:
        parts.append(org.address_street.strip())
    city_state_zip = ""
    if org.address_city:
        city_state_zip += org.address_city.strip()
    if org.address_state:
        if city_state_zip:
            city_state_zip += ", "
        # Normalize state to "RI" if the answer used "Rhode Island"
        st = org.address_state.strip()
        if st.lower() in ["rhode island", "ri"]:
            st = "RI"
        city_state_zip += st
    if org.address_zip:
        if city_state_zip:
            city_state_zip += f" {org.address_zip.strip()}"
        else:
            city_state_zip = org.address_zip.strip()
    if city_state_zip:
        parts.append(city_state_zip)
    return ", ".join(parts)


def make_source_instruction(base_instruction: str, sources: Optional[List[str]]) -> str:
    """Append guidance for missing sources to enforce source-grounding policy."""
    if sources and len(sources) > 0:
        return base_instruction
    # Strongly guide the verifier to mark as unsupported if no URLs were provided in answer
    return (
        base_instruction
        + " IMPORTANT: The answer did not provide any URL sources for this claim. Treat the claim as Not Supported "
          "due to missing evidence and return Incorrect."
    )


def _sources_or_none(org: Optional[OrgExtraction]) -> Optional[List[str]]:
    if org and org.source_urls:
        cleaned = [u for u in org.source_urls if isinstance(u, str) and u.strip() != ""]
        return cleaned if cleaned else None
    return None


# --------------------------------------------------------------------------- #
# Verification for one organization                                           #
# --------------------------------------------------------------------------- #
async def verify_org(
    evaluator: Evaluator,
    parent_node,
    org: Optional[OrgExtraction],
    org_slot_label: str,  # "Family_Law_Resource" or "Housing_Law_Resource"
    required_practice_area: str  # "family law" or "housing law"
) -> None:
    """
    Build the verification subtree for a single organization.
    This follows the rubric: sequential → Basic_Information (parallel, critical) & Service_Details (parallel, critical).
    """
    # Create the resource node (sequential, non-critical under root)
    res_node = evaluator.add_sequential(
        id=org_slot_label,
        desc=(
            "Identify one Rhode Island nonprofit legal aid organization or pro bono referral service that provides "
            f"{required_practice_area} services"
        ),
        parent=parent_node,
        critical=False
    )

    # ---------------------- Basic Information (critical, parallel) ---------------------- #
    basic_node = evaluator.add_parallel(
        id=f"{org_slot_label}_Basic_Information",
        desc="Provide the organization's name, type (nonprofit legal aid or pro bono referral), location, and complete contact information",
        parent=res_node,
        critical=True
    )

    # Organization Identity
    org_identity_leaf = evaluator.add_leaf(
        id=f"{org_slot_label}_Organization_Identity",
        desc="Provide the official name of the organization and confirm it is a nonprofit legal aid organization or pro bono referral service (not a private for-profit law firm)",
        parent=basic_node,
        critical=True
    )
    name = org.name if org and org.name else ""
    identity_claim = (
        f"'{name}' is a nonprofit legal aid organization or a pro bono referral service, "
        f"and is not a private for-profit law firm."
    )
    identity_instruction = make_source_instruction(
        "Confirm the entity is a nonprofit legal aid or pro bono referral service. "
        "Accept minor naming variations (e.g., including/excluding 'Inc.' or 'of Rhode Island'). "
        "If the page indicates it is a private for-profit law firm, mark Incorrect.",
        _sources_or_none(org)
    )
    await evaluator.verify(
        claim=identity_claim,
        node=org_identity_leaf,
        sources=_sources_or_none(org),
        additional_instruction=identity_instruction
    )

    # Rhode Island Location
    ri_loc_leaf = evaluator.add_leaf(
        id=f"{org_slot_label}_Rhode_Island_Location",
        desc="Verify and confirm that the organization is located in Rhode Island",
        parent=basic_node,
        critical=True
    )
    ri_loc_claim = f"'{name}' is located in Rhode Island (RI)."
    ri_loc_instruction = make_source_instruction(
        "Confirm that the organization has a physical location in Rhode Island. "
        "Accept 'RI' or 'Rhode Island'. Multi-office orgs are fine as long as an RI office/location is shown.",
        _sources_or_none(org)
    )
    await evaluator.verify(
        claim=ri_loc_claim,
        node=ri_loc_leaf,
        sources=_sources_or_none(org),
        additional_instruction=ri_loc_instruction
    )

    # Contact Information
    contact_leaf = evaluator.add_leaf(
        id=f"{org_slot_label}_Contact_Information",
        desc="Provide complete contact information including physical address (street address, city, state, ZIP code) and phone number",
        parent=basic_node,
        critical=True
    )
    address_str = build_address_str(org)
    phone = org.phone if org and org.phone else ""
    contact_claim = (
        f"'{name}' lists its physical address as '{address_str}', and its phone number as '{phone}'."
    )
    contact_instruction = make_source_instruction(
        "Verify that the page lists both the full physical address (street, city, state, ZIP) and a phone number as stated. "
        "Accept minor formatting differences (punctuation, suite abbreviations, phone formatting). "
        "If any of these elements cannot be confirmed, mark Incorrect.",
        _sources_or_none(org)
    )
    await evaluator.verify(
        claim=contact_claim,
        node=contact_leaf,
        sources=_sources_or_none(org),
        additional_instruction=contact_instruction
    )

    # ---------------------- Service Details (critical, parallel) ---------------------- #
    service_node = evaluator.add_parallel(
        id=f"{org_slot_label}_Service_Details",
        desc="Verify the organization's practice area coverage and eligibility requirements",
        parent=res_node,
        critical=True
    )

    # Practice Area Coverage
    practice_leaf = evaluator.add_leaf(
        id=f"{org_slot_label}_Practice_Area_Coverage",
        desc=(
            f"Verify and confirm that the organization provides legal services in {required_practice_area} "
            "(such as relevant example matters)"
        ),
        parent=service_node,
        critical=True
    )
    if required_practice_area.lower().strip() == "family law":
        practice_claim = (
            f"'{name}' provides family law services, such as divorce, child custody, or domestic relations matters."
        )
        practice_instruction_base = (
            "Confirm that the organization provides family law services (e.g., divorce, child custody, domestic relations, "
            "protection from abuse). Accept reasonable synonyms. If this is not supported, mark Incorrect."
        )
    else:
        practice_claim = (
            f"'{name}' provides housing law services, such as eviction defense, tenant rights, or landlord-tenant disputes."
        )
        practice_instruction_base = (
            "Confirm that the organization provides housing law services (e.g., eviction defense, tenant rights, "
            "landlord-tenant disputes, housing issues). Accept reasonable synonyms. If this is not supported, mark Incorrect."
        )
    practice_instruction = make_source_instruction(practice_instruction_base, _sources_or_none(org))
    await evaluator.verify(
        claim=practice_claim,
        node=practice_leaf,
        sources=_sources_or_none(org),
        additional_instruction=practice_instruction
    )

    # Eligibility Information
    eligibility_leaf = evaluator.add_leaf(
        id=f"{org_slot_label}_Eligibility_Information",
        desc=(
            "Provide the organization's eligibility requirements (such as income limits expressed as percentage of Federal "
            "Poverty Guidelines) or specify if free consultation is available"
        ),
        parent=service_node,
        critical=True
    )
    eligibility_text = org.eligibility if org and org.eligibility else ""
    eligibility_claim = (
        f"Eligibility for services at '{name}' is described as: '{eligibility_text}'."
    )
    eligibility_instruction = make_source_instruction(
        "Verify that the page describes eligibility requirements consistent with the provided text (e.g., income limits as "
        "percent of Federal Poverty Guidelines, residence, case type) or that free consultation is available as described. "
        "Paraphrasing is acceptable, but the substance must match. If no eligibility information is provided on the page(s), mark Incorrect.",
        _sources_or_none(org)
    )
    await evaluator.verify(
        claim=eligibility_claim,
        node=eligibility_leaf,
        sources=_sources_or_none(org),
        additional_instruction=eligibility_instruction
    )


# --------------------------------------------------------------------------- #
# Distinctness verification                                                   #
# --------------------------------------------------------------------------- #
async def add_distinctness_check(
    evaluator: Evaluator,
    parent_node,
    family_org: Optional[OrgExtraction],
    housing_org: Optional[OrgExtraction]
) -> None:
    """
    Add a critical check to ensure the two organizations are different.
    This is a simple logical verification without URLs.
    """
    distinct_leaf = evaluator.add_leaf(
        id="Distinctness_Verification",
        desc=(
            "Verify that the family law resource and the housing law resource are two different organizations "
            "(not the same organization or different branches/offices of the same organization)"
        ),
        parent=parent_node,
        critical=True
    )

    fam_name = (family_org.name if family_org and family_org.name else "").strip()
    hou_name = (housing_org.name if housing_org and housing_org.name else "").strip()

    claim = (
        f"The family law organization and the housing law organization are different entities: "
        f"'{fam_name}' and '{hou_name}'. They are not the same organization and not simply different offices/branches."
    )
    add_ins = (
        "Focus on whether these are distinct organizations, not just different addresses or offices of the same entity. "
        "Use common-sense reasoning; you do not need to consult external websites for this check. "
        "If either name is missing or clearly the same, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim,
        node=distinct_leaf,
        sources=None,  # Logical check based on the answer itself
        additional_instruction=add_ins
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
    Evaluate an answer for Rhode Island legal aid resources (family + housing) with evidence-backed verification.
    """
    # Initialize evaluator with a parallel root (allow partial credit between the two resources)
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
        default_model=model
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_legal_aid_resources(),
        template_class=LegalAidExtraction,
        extraction_name="ri_legal_aid_resources"
    )

    # Build main node that mirrors rubric description (optional grouping under root)
    main_node = evaluator.add_parallel(
        id="Rhode_Island_Legal_Aid_Resources",
        desc=(
            "Identify two distinct nonprofit legal aid organizations or pro bono referral services in Rhode Island, "
            "each serving a different practice area (one must serve family law and one must serve housing law). "
            "The two organizations must be different from each other."
        ),
        parent=root,
        critical=False
    )

    # Family law resource subtree
    await verify_org(
        evaluator=evaluator,
        parent_node=main_node,
        org=extracted.family,
        org_slot_label="Family_Law_Resource",
        required_practice_area="family law"
    )

    # Housing law resource subtree
    await verify_org(
        evaluator=evaluator,
        parent_node=main_node,
        org=extracted.housing,
        org_slot_label="Housing_Law_Resource",
        required_practice_area="housing law"
    )

    # Distinctness verification (critical)
    await add_distinctness_check(
        evaluator=evaluator,
        parent_node=main_node,
        family_org=extracted.family,
        housing_org=extracted.housing
    )

    # Return final structured summary
    return evaluator.get_summary()