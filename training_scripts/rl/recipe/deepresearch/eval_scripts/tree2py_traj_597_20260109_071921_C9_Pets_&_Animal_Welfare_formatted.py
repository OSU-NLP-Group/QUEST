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
TASK_ID = "Texas_Animal_Welfare_Organizations_Comprehensive_Evaluation"
TASK_DESCRIPTION = """
Identify 4 animal welfare organizations in Texas that meet the following comprehensive standards:

Legal and Regulatory Compliance (Critical Requirements):
1. Implement Texas-mandated sterilization requirements (animals must be sterilized before or within 30 days of adoption per Texas Health and Safety Code Chapter 828)
2. Ensure rabies vaccination compliance for all dogs and cats per Texas state law
3. Maintain proper separation of healthy animals from sick, injured, or diseased animals in accordance with Texas Health and Safety Code §823.003

Adoption Process Standards (Critical Requirements):
4. Require adopters to be at least 18 years of age
5. Require valid photo identification from all adopters
6. Require signed adoption contracts for all adoptions

Medical and Health Services (Additional Requirements):
7. Provide microchip implantation services for adopted animals
8. Provide core vaccination services (such as DA2PP for dogs or FVRCP for cats)

Program Offerings (Additional Requirements):
9. Operate an active foster care program
10. Maintain a volunteer program with opportunities for community involvement

For each organization, provide the organization's name, evidence from the organization's website or official sources confirming compliance with each requirement, and reference URLs supporting each criterion.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OrganizationInfo(BaseModel):
    name: Optional[str] = None
    homepage_url: Optional[str] = None

    # Basic eligibility evidence URLs
    location_urls: List[str] = Field(default_factory=list)
    mission_urls: List[str] = Field(default_factory=list)

    # Critical requirements evidence URLs
    sterilization_urls: List[str] = Field(default_factory=list)
    rabies_urls: List[str] = Field(default_factory=list)
    separation_urls: List[str] = Field(default_factory=list)
    min_age_urls: List[str] = Field(default_factory=list)
    photo_id_urls: List[str] = Field(default_factory=list)
    contract_urls: List[str] = Field(default_factory=list)

    # Additional requirements evidence URLs
    microchip_urls: List[str] = Field(default_factory=list)
    core_vaccines_urls: List[str] = Field(default_factory=list)
    foster_urls: List[str] = Field(default_factory=list)
    volunteer_urls: List[str] = Field(default_factory=list)


class OrganizationsExtraction(BaseModel):
    organizations: List[OrganizationInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_organizations() -> str:
    return """
    Extract up to 4 Texas animal welfare organizations mentioned in the answer. For each organization, return:
    - name: The organization's name (string).
    - homepage_url: The main official website URL (string) if provided.
    - location_urls: URLs explicitly showing the organization is in Texas (addresses, location or "About/Contact" pages).
    - mission_urls: URLs showing the organization is an animal welfare org (shelter/rescue/adoption services).
    - sterilization_urls: URLs showing sterilization/spay-neuter policy timing (before or within 30 days of adoption) consistent with Texas law.
    - rabies_urls: URLs showing rabies vaccination compliance for dogs and cats.
    - separation_urls: URLs showing policy/standards for separating healthy animals from sick/injured/diseased animals (or isolation/quarantine protocols).
    - min_age_urls: URLs showing adopters must be at least 18 years old.
    - photo_id_urls: URLs showing valid photo ID is required from adopters.
    - contract_urls: URLs showing signed adoption contract is required.
    - microchip_urls: URLs showing microchip implantation services are provided for adopted animals.
    - core_vaccines_urls: URLs showing core vaccination services (e.g., DA2PP for dogs or FVRCP for cats).
    - foster_urls: URLs showing an active foster program.
    - volunteer_urls: URLs showing a volunteer program.

    Rules:
    - Extract only URLs explicitly present in the answer. If a field is not mentioned, return an empty list for that URLs field.
    - Include up to 4 organizations in the array 'organizations'. If more are present, only keep the first 4.
    - If the answer does not provide a value for 'name' or 'homepage_url', set it to null.
    - Ensure URLs are valid and complete (if protocol is missing, prepend http://).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(org: OrganizationInfo, primary_lists: List[List[str]]) -> List[str]:
    """Combine multiple URL lists and include homepage_url as fallback when lists are empty."""
    urls: List[str] = []
    for lst in primary_lists:
        urls.extend([u for u in lst if isinstance(u, str) and u.strip() != ""])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    # Fallback to homepage if nothing else
    if not deduped and org.homepage_url and org.homepage_url.strip():
        deduped = [org.homepage_url.strip()]
    return deduped


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_organization(
    evaluator: Evaluator,
    parent_node,
    org: OrganizationInfo,
    idx: int,
) -> None:
    """
    Build verification tree for a single organization with checks aligned to the rubric.
    """
    org_num = idx + 1

    # Top-level node for this organization (non-critical to allow partial credit across organizations)
    org_node = evaluator.add_parallel(
        id=f"Organization_{org_num}",
        desc=f"{org_num}st organization" if org_num == 1 else (f"{org_num}nd organization" if org_num == 2 else (f"{org_num}rd organization" if org_num == 3 else f"{org_num}th organization")),
        parent=parent_node,
        critical=False,
    )

    # ---------------------- Basic Eligibility (Critical) ---------------------- #
    basic_node = evaluator.add_parallel(
        id=f"Org{org_num}_Basic_Eligibility",
        desc="Organization is an animal welfare organization in Texas and is clearly identified",
        parent=org_node,
        critical=True,
    )

    # Name provided (existence check)
    name_provided_node = evaluator.add_custom_node(
        result=bool(org.name and org.name.strip()),
        id=f"Org{org_num}_Name_Provided",
        desc="Organization name is provided",
        parent=basic_node,
        critical=True,
    )

    # Location evidence + URL
    loc_leaf = evaluator.add_leaf(
        id=f"Org{org_num}_Texas_Location_Evidence_URL",
        desc="Evidence + URL shows the organization is in Texas",
        parent=basic_node,
        critical=True,
    )
    loc_sources = combine_sources(org, [org.location_urls, [org.homepage_url] if org.homepage_url else []])
    loc_claim = f"This webpage shows that {org.name or 'the organization'} is located in Texas or operates in Texas."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=loc_sources,
        additional_instruction="Look for Texas addresses (city/state), service area statements, or explicit mention of Texas operations on the provided pages. If none of the URLs support Texas location, mark as not supported.",
    )

    # Mission evidence + URL
    mission_leaf = evaluator.add_leaf(
        id=f"Org{org_num}_Animal_Welfare_Mission_Evidence_URL",
        desc="Evidence + URL shows the organization is an animal welfare organization (e.g., shelter/rescue/adoption services)",
        parent=basic_node,
        critical=True,
    )
    mission_sources = combine_sources(org, [org.mission_urls, [org.homepage_url] if org.homepage_url else []])
    mission_claim = f"This webpage shows that {org.name or 'the organization'} is an animal welfare organization, such as a shelter, rescue, or adoption service provider."
    await evaluator.verify(
        claim=mission_claim,
        node=mission_leaf,
        sources=mission_sources,
        additional_instruction="Check for statements like 'animal shelter', 'rescue', 'adoption services', or similar mission descriptions on the provided URLs.",
    )

    # ------------------- Critical Requirements (Critical) --------------------- #
    critical_node = evaluator.add_parallel(
        id=f"Org{org_num}_Critical_Requirements",
        desc="All critical legal/regulatory and adoption-process requirements are met with evidence and URLs",
        parent=org_node,
        critical=True,
    )

    # Sterilization timing per Texas law (before or within 30 days)
    steril_leaf = evaluator.add_leaf(
        id=f"Org{org_num}_Sterilization_Evidence_URL",
        desc="Evidence + URL confirms sterilization policy consistent with Texas-mandated timing (before or within 30 days of adoption)",
        parent=critical_node,
        critical=True,
    )
    steril_sources = combine_sources(org, [org.sterilization_urls])
    steril_claim = f"This webpage states that {org.name or 'the organization'} requires animals to be sterilized before adoption or within 30 days of adoption, consistent with Texas Health and Safety Code Chapter 828."
    await evaluator.verify(
        claim=steril_claim,
        node=steril_leaf,
        sources=steril_sources,
        additional_instruction="Look for spay/neuter policy language specifying compliance with Texas timing: before adoption or within 30 days after adoption.",
    )

    # Rabies vaccination compliance
    rabies_leaf = evaluator.add_leaf(
        id=f"Org{org_num}_Rabies_Compliance_Evidence_URL",
        desc="Evidence + URL confirms rabies vaccination compliance for dogs and cats per Texas law",
        parent=critical_node,
        critical=True,
    )
    rabies_sources = combine_sources(org, [org.rabies_urls])
    rabies_claim = f"This webpage states that {org.name or 'the organization'} requires rabies vaccination compliance for dogs and cats per Texas law."
    await evaluator.verify(
        claim=rabies_claim,
        node=rabies_leaf,
        sources=rabies_sources,
        additional_instruction="Look for rabies vaccination requirements/policies applicable to cats and dogs, referencing Texas law or compliance.",
    )

    # Healthy vs sick/injured/diseased separation (Texas HSC §823.003)
    sep_leaf = evaluator.add_leaf(
        id=f"Org{org_num}_Healthy_Sick_Separation_Evidence_URL",
        desc="Evidence + URL confirms separation of healthy animals from sick/injured/diseased animals consistent with Texas Health and Safety Code §823.003",
        parent=critical_node,
        critical=True,
    )
    sep_sources = combine_sources(org, [org.separation_urls])
    sep_claim = f"This webpage shows that {org.name or 'the organization'} maintains separation of healthy animals from sick, injured, or diseased animals, consistent with Texas Health and Safety Code §823.003."
    await evaluator.verify(
        claim=sep_claim,
        node=sep_leaf,
        sources=sep_sources,
        additional_instruction="Look for facility standards, isolation/quarantine protocols, or similar statements indicating separation of sick/injured/diseased animals from healthy animals.",
    )

    # Adopter minimum age 18
    age_leaf = evaluator.add_leaf(
        id=f"Org{org_num}_Adopter_Min_Age_Evidence_URL",
        desc="Evidence + URL confirms adopters must be at least 18 years of age",
        parent=critical_node,
        critical=True,
    )
    age_sources = combine_sources(org, [org.min_age_urls])
    age_claim = f"This webpage states that {org.name or 'the organization'} requires adopters to be at least 18 years old."
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=age_sources,
        additional_instruction="Look for adoption eligibility policies requiring that adopters are 18+ years of age.",
    )

    # Photo ID required
    id_leaf = evaluator.add_leaf(
        id=f"Org{org_num}_Photo_ID_Evidence_URL",
        desc="Evidence + URL confirms valid photo identification is required from adopters",
        parent=critical_node,
        critical=True,
    )
    id_sources = combine_sources(org, [org.photo_id_urls])
    id_claim = f"This webpage states that {org.name or 'the organization'} requires adopters to present valid photo identification."
    await evaluator.verify(
        claim=id_claim,
        node=id_leaf,
        sources=id_sources,
        additional_instruction="Look for adoption process requirements that explicitly mention photo ID or government-issued ID being required.",
    )

    # Signed adoption contract required
    contract_leaf = evaluator.add_leaf(
        id=f"Org{org_num}_Signed_Adoption_Contract_Evidence_URL",
        desc="Evidence + URL confirms signed adoption contract is required",
        parent=critical_node,
        critical=True,
    )
    contract_sources = combine_sources(org, [org.contract_urls])
    contract_claim = f"This webpage states that {org.name or 'the organization'} requires a signed adoption contract for all adoptions."
    await evaluator.verify(
        claim=contract_claim,
        node=contract_leaf,
        sources=contract_sources,
        additional_instruction="Look for mention of an adoption contract/agreement that adopters must sign.",
    )

    # ---------------- Additional Requirements (Non-Critical) ------------------ #
    additional_node = evaluator.add_parallel(
        id=f"Org{org_num}_Additional_Requirements",
        desc="Additional (non-critical / partial-credit) standards are met with evidence and URLs",
        parent=org_node,
        critical=False,
    )

    # Microchip implantation services
    chip_leaf = evaluator.add_leaf(
        id=f"Org{org_num}_Microchip_Evidence_URL",
        desc="Evidence + URL confirms microchip implantation services for adopted animals",
        parent=additional_node,
        critical=False,
    )
    chip_sources = combine_sources(org, [org.microchip_urls])
    chip_claim = f"This webpage shows that {org.name or 'the organization'} provides microchip implantation services for adopted animals."
    await evaluator.verify(
        claim=chip_claim,
        node=chip_leaf,
        sources=chip_sources,
        additional_instruction="Look for policies or service descriptions mentioning microchipping or microchip implantation for adopted animals.",
    )

    # Core vaccinations (DA2PP for dogs, FVRCP for cats)
    core_vax_leaf = evaluator.add_leaf(
        id=f"Org{org_num}_Core_Vaccines_Evidence_URL",
        desc="Evidence + URL confirms provision of core vaccinations (e.g., DA2PP for dogs or FVRCP for cats)",
        parent=additional_node,
        critical=False,
    )
    core_vax_sources = combine_sources(org, [org.core_vaccines_urls])
    core_vax_claim = f"This webpage shows that {org.name or 'the organization'} provides core vaccination services (e.g., DA2PP for dogs or FVRCP for cats)."
    await evaluator.verify(
        claim=core_vax_claim,
        node=core_vax_leaf,
        sources=core_vax_sources,
        additional_instruction="Look for vaccination services lists or medical intake policies specifying core vaccines like DA2PP (dogs) or FVRCP (cats).",
    )

    # Foster program
    foster_leaf = evaluator.add_leaf(
        id=f"Org{org_num}_Foster_Program_Evidence_URL",
        desc="Evidence + URL confirms an active foster care program",
        parent=additional_node,
        critical=False,
    )
    foster_sources = combine_sources(org, [org.foster_urls])
    foster_claim = f"This webpage shows that {org.name or 'the organization'} operates an active foster care program."
    await evaluator.verify(
        claim=foster_claim,
        node=foster_leaf,
        sources=foster_sources,
        additional_instruction="Look for pages recruiting fosters, describing foster programs, or providing foster application details.",
    )

    # Volunteer program
    volunteer_leaf = evaluator.add_leaf(
        id=f"Org{org_num}_Volunteer_Program_Evidence_URL",
        desc="Evidence + URL confirms a volunteer program with opportunities for community involvement",
        parent=additional_node,
        critical=False,
    )
    volunteer_sources = combine_sources(org, [org.volunteer_urls])
    volunteer_claim = f"This webpage shows that {org.name or 'the organization'} maintains a volunteer program with opportunities for community involvement."
    await evaluator.verify(
        claim=volunteer_claim,
        node=volunteer_leaf,
        sources=volunteer_sources,
        additional_instruction="Look for volunteer program pages, sign-up links, or descriptions of volunteer opportunities.",
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
) -> Dict[str, Any]:
    """
    Evaluate the answer for Texas animal welfare organizations comprehensive standards.
    """
    # Initialize evaluator (root is parallel per rubric)
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

    # Extract organizations and evidence URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_organizations(),
        template_class=OrganizationsExtraction,
        extraction_name="organizations_extraction",
    )

    # Normalize to exactly 4 organizations (pad with empty if fewer)
    orgs: List[OrganizationInfo] = list(extracted.organizations[:4])
    while len(orgs) < 4:
        orgs.append(OrganizationInfo())

    # Add a summary of requested criteria as custom info for transparency
    evaluator.add_custom_info(
        {
            "critical_requirements": [
                "Sterilization before or within 30 days (Texas HSC Chapter 828)",
                "Rabies vaccination compliance for dogs and cats",
                "Separation of healthy from sick/injured/diseased animals (Texas HSC §823.003)",
                "Adopter minimum age 18",
                "Photo ID required",
                "Signed adoption contract required",
            ],
            "additional_requirements": [
                "Microchip implantation services",
                "Core vaccination services (DA2PP for dogs / FVRCP for cats)",
                "Active foster program",
                "Volunteer program",
            ],
        },
        info_type="requirements_overview",
        info_name="requirements_overview",
    )

    # Build tree and verify each organization
    for idx, org in enumerate(orgs):
        await verify_organization(evaluator, root, org, idx)

    # Return structured evaluation summary
    return evaluator.get_summary()