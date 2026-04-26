import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "teacher_app_requirements_3districts"
TASK_DESCRIPTION = (
    "I am researching certified teaching positions in three school districts across different states. "
    "Please compile detailed application requirement information for certified teaching positions at the following three school districts: "
    "1. Albuquerque Public Schools (New Mexico), 2. Greenville County Schools (South Carolina), 3. Mansfield Independent School District (Texas). "
    "For each school district, provide the following information with official source URLs: Online Application System (name or type of the online application platform used), "
    "Minimum Educational Requirement (the minimum degree required for certified teaching positions), State Certification Requirement (which state's teaching certification/licensure is required), "
    "Required Application Documents (list at least two documents required as part of the application such as resume, transcripts, letters of recommendation, etc.), "
    "Background Check Requirements (whether background checks and/or fingerprinting are required), and HR Contact Information (phone number and/or email address for the Human Resources or Employment office). "
    "Each piece of information must be supported by a reference URL to the official district website or documentation."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ValueWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ListWithSources(BaseModel):
    items: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class DistrictRequirements(BaseModel):
    online_application_system: Optional[ValueWithSources] = None
    minimum_education_requirement: Optional[ValueWithSources] = None
    state_certification_requirement: Optional[ValueWithSources] = None
    required_application_documents: Optional[ListWithSources] = None
    background_check_requirement: Optional[ValueWithSources] = None
    hr_contact_information: Optional[ValueWithSources] = None


class ApplicationRequirementsExtraction(BaseModel):
    albuquerque_public_schools_nm: Optional[DistrictRequirements] = None
    greenville_county_schools_sc: Optional[DistrictRequirements] = None
    mansfield_isd_tx: Optional[DistrictRequirements] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_application_requirements() -> str:
    return """
You will extract, from the provided answer text only, district-specific application requirement information for certified teaching positions for the following three districts:
1) Albuquerque Public Schools (New Mexico)
2) Greenville County Schools (South Carolina)
3) Mansfield Independent School District (Texas)

For EACH district, extract the following fields and their cited source URLs exactly as present in the answer:

- online_application_system: 
  • value: The platform or system name/type the district uses for online applications (e.g., "PowerSchool Applicant Tracking", "TalentEd/PowerSchool Unified Talent", "Frontline", etc.).
  • sources: An array of URL strings to the official district website pages that support this information. Only include URLs explicitly present in the answer text.

- minimum_education_requirement:
  • value: The minimum degree/education required for certified teaching positions (e.g., "Bachelor’s degree", "Bachelor’s degree from an accredited institution", etc.).
  • sources: Array of official district URLs cited in the answer supporting this.

- state_certification_requirement:
  • value: The state teaching certification/licensure required (e.g., "New Mexico teaching license", "South Carolina educator certification", "Texas certification").
  • sources: Array of official district URLs cited in the answer supporting this.

- required_application_documents:
  • items: Array listing the document names cited (e.g., "Resume", "Unofficial transcripts", "Letters of recommendation"). Include as many as the answer provides.
  • sources: Array of official district URLs cited in the answer supporting these required documents.

- background_check_requirement:
  • value: A concise statement capturing whether background checks and/or fingerprinting are required, as stated in the answer.
  • sources: Array of official district URLs cited in the answer supporting this.

- hr_contact_information:
  • value: The HR/Employment contact information (phone number and/or email) as presented in the answer.
  • sources: Array of official district URLs cited in the answer supporting this.

IMPORTANT:
- Only extract URLs that are explicitly present in the answer text (plain or as markdown links). Do not fabricate URLs.
- If the answer does not provide a value for a field, set that field to null (for value fields) or an empty array (for items/sources).
- If the answer provides a value but no supporting URLs for that field, return an empty 'sources' array for that field.
- Prefer extracting strings as-is (do not normalize or alter formatting).
- Do not pull any information from outside the answer text.

Return a single JSON object matching this exact schema:

{
  "albuquerque_public_schools_nm": {
    "online_application_system": {"value": string|null, "sources": [string, ...]},
    "minimum_education_requirement": {"value": string|null, "sources": [string, ...]},
    "state_certification_requirement": {"value": string|null, "sources": [string, ...]},
    "required_application_documents": {"items": [string, ...], "sources": [string, ...]},
    "background_check_requirement": {"value": string|null, "sources": [string, ...]},
    "hr_contact_information": {"value": string|null, "sources": [string, ...]}
  },
  "greenville_county_schools_sc": {
    "online_application_system": {"value": string|null, "sources": [string, ...]},
    "minimum_education_requirement": {"value": string|null, "sources": [string, ...]},
    "state_certification_requirement": {"value": string|null, "sources": [string, ...]},
    "required_application_documents": {"items": [string, ...], "sources": [string, ...]},
    "background_check_requirement": {"value": string|null, "sources": [string, ...]},
    "hr_contact_information": {"value": string|null, "sources": [string, ...]}
  },
  "mansfield_isd_tx": {
    "online_application_system": {"value": string|null, "sources": [string, ...]},
    "minimum_education_requirement": {"value": string|null, "sources": [string, ...]},
    "state_certification_requirement": {"value": string|null, "sources": [string, ...]},
    "required_application_documents": {"items": [string, ...], "sources": [string, ...]},
    "background_check_requirement": {"value": string|null, "sources": [string, ...]},
    "hr_contact_information": {"value": string|null, "sources": [string, ...]}
  }
}
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(sources: Optional[List[str]]) -> List[str]:
    if not sources:
        return []
    return [s for s in sources if isinstance(s, str) and s.strip()]


def _join_items_english(items: List[str]) -> str:
    items = [i for i in items if isinstance(i, str) and i.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_value_field(
    evaluator: Evaluator,
    parent_node,
    *,
    base_id: str,
    district_name: str,
    label: str,
    field: Optional[ValueWithSources],
    domain_hints: List[str],
    claim_template: str,
    addl_note: Optional[str] = None
) -> None:
    """
    Build sequential verification for a single value field:
    - Existence w/ sources (custom critical)
    - Supported by cited official URLs (critical leaf via verify_by_urls)
    """
    seq_node = evaluator.add_sequential(
        id=f"{base_id}_{label.lower().replace(' ', '_')}",
        desc=f"{label} for {district_name}: identified and supported by official sources",
        parent=parent_node,
        critical=True
    )

    # Existence check: value present and at least one source URL
    has_value = field is not None and isinstance(field.value, str) and field.value.strip() != ""
    has_sources = field is not None and len(_safe_sources(field.sources)) > 0
    evaluator.add_custom_node(
        result=bool(has_value and has_sources),
        id=f"{base_id}_{label.lower().replace(' ', '_')}_exists",
        desc=f"{label} value and at least one official URL are provided in the answer",
        parent=seq_node,
        critical=True
    )

    # Support check
    support_leaf = evaluator.add_leaf(
        id=f"{base_id}_{label.lower().replace(' ', '_')}_supported",
        desc=f"{label} is supported by the cited official {district_name} page(s)",
        parent=seq_node,
        critical=True
    )

    # Build claim string
    value_text = field.value if field and field.value else ""
    claim = claim_template.format(district=district_name, value=value_text)

    # Instruction emphasizing official domains and tolerance to naming variants
    domain_hint_text = ", ".join(domain_hints) if domain_hints else "the district’s official domain"
    extra_ins = (
        f"Only accept this claim if it is explicitly supported by content on an official {district_name} webpage "
        f"(for example, domains like {domain_hint_text} or clear official subdomains). "
        f"Reject third-party job boards or non-official sites. "
        f"Allow reasonable naming variations and synonyms. "
    )
    if addl_note:
        extra_ins += f"Additional note: {addl_note}"

    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=_safe_sources(field.sources) if field else [],
        additional_instruction=extra_ins
    )


async def _verify_list_field(
    evaluator: Evaluator,
    parent_node,
    *,
    base_id: str,
    district_name: str,
    label: str,
    field: Optional[ListWithSources],
    domain_hints: List[str],
    min_items: int = 2
) -> None:
    """
    Build sequential verification for a list field:
    - Existence w/ at least min_items items and at least one source URL (custom critical)
    - Supported by cited official URLs (critical leaf via verify_by_urls)
    """
    seq_node = evaluator.add_sequential(
        id=f"{base_id}_{label.lower().replace(' ', '_')}",
        desc=f"{label} for {district_name}: listed and supported by official sources",
        parent=parent_node,
        critical=True
    )

    items = field.items if field and field.items else []
    sources = _safe_sources(field.sources) if field else []
    exists_ok = (len([i for i in items if isinstance(i, str) and i.strip()]) >= min_items) and (len(sources) > 0)

    evaluator.add_custom_node(
        result=bool(exists_ok),
        id=f"{base_id}_{label.lower().replace(' ', '_')}_exists",
        desc=f"{label}: at least {min_items} item(s) listed and at least one official URL provided in the answer",
        parent=seq_node,
        critical=True
    )

    support_leaf = evaluator.add_leaf(
        id=f"{base_id}_{label.lower().replace(' ', '_')}_supported",
        desc=f"{label} are supported by the cited official {district_name} page(s)",
        parent=seq_node,
        critical=True
    )

    items_str = _join_items_english(items)
    claim = (
        f"The required application documents for certified teaching positions at {district_name} include: {items_str}."
    )

    domain_hint_text = ", ".join(domain_hints) if domain_hints else "the district’s official domain"
    extra_ins = (
        f"Support this claim only if at least two of the listed documents are explicitly stated as required on an official {district_name} webpage "
        f"(for example, domains like {domain_hint_text} or clear official subdomains). "
        f"The list does not need to be exhaustive; it is acceptable that the page lists more than what is shown. "
        f"Reject third-party job boards or aggregator sites."
    )

    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=sources,
        additional_instruction=extra_ins
    )


async def _verify_district(
    evaluator: Evaluator,
    parent_node,
    *,
    district_id: str,
    district_name: str,
    district_data: Optional[DistrictRequirements],
    domain_hints: List[str]
) -> None:
    """
    Build the verification subtree for a single district.
    Each field group is a critical sequential node under a critical parallel district node.
    """
    district_node = evaluator.add_parallel(
        id=district_id,
        desc=f"Provide required application requirement fields for {district_name}.",
        parent=parent_node,
        critical=True
    )

    # Online Application System
    await _verify_value_field(
        evaluator,
        district_node,
        base_id=f"{district_id}_online_app_system",
        district_name=district_name,
        label="Online Application System",
        field=district_data.online_application_system if district_data else None,
        domain_hints=domain_hints,
        claim_template="The online application system/platform used by {district} for certified teaching positions is '{value}'.",
        addl_note="Accept platform brand names or descriptors (e.g., 'PowerSchool Applicant Tracking', 'TalentEd/PowerSchool', 'Frontline'), allowing reasonable naming variants."
    )

    # Minimum Educational Requirement
    await _verify_value_field(
        evaluator,
        district_node,
        base_id=f"{district_id}_min_edu",
        district_name=district_name,
        label="Minimum Educational Requirement",
        field=district_data.minimum_education_requirement if district_data else None,
        domain_hints=domain_hints,
        claim_template="The minimum educational requirement for certified teaching positions at {district} is '{value}'.",
        addl_note="Focus on the minimum degree/education phrasing (e.g., 'Bachelor’s degree' or similar)."
    )

    # State Certification Requirement
    await _verify_value_field(
        evaluator,
        district_node,
        base_id=f"{district_id}_state_cert",
        district_name=district_name,
        label="State Certification Requirement",
        field=district_data.state_certification_requirement if district_data else None,
        domain_hints=domain_hints,
        claim_template="The required state teaching certification/licensure for certified teaching positions at {district} is '{value}'.",
        addl_note="Confirm the statement clearly relates to the state's teacher certification for this district (e.g., NM, SC, TX)."
    )

    # Required Application Documents (list; at least two)
    await _verify_list_field(
        evaluator,
        district_node,
        base_id=f"{district_id}_required_docs",
        district_name=district_name,
        label="Required Application Documents",
        field=district_data.required_application_documents if district_data else None,
        domain_hints=domain_hints,
        min_items=2
    )

    # Background Check / Fingerprinting
    await _verify_value_field(
        evaluator,
        district_node,
        base_id=f"{district_id}_bg_check",
        district_name=district_name,
        label="Background Check or Fingerprinting",
        field=district_data.background_check_requirement if district_data else None,
        domain_hints=domain_hints,
        claim_template="The background check and/or fingerprinting requirement for certified teaching applicants at {district} is: '{value}'.",
        addl_note="Confirm whether the page states background checks and/or fingerprinting are required for certified teaching applicants."
    )

    # HR Contact Information
    await _verify_value_field(
        evaluator,
        district_node,
        base_id=f"{district_id}_hr_contact",
        district_name=district_name,
        label="HR Contact Information",
        field=district_data.hr_contact_information if district_data else None,
        domain_hints=domain_hints,
        claim_template="The HR contact information for {district} is: '{value}'.",
        addl_note="Accept phone numbers and/or emails associated with Human Resources, Talent, or Employment offices."
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
    Evaluate an answer for the three-district certified teaching application requirements task.
    """
    # Initialize evaluator (root is non-critical by framework design; we enforce criticality on children)
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

    # Add top-level critical node to reflect rubric's "Application_Requirements_Research"
    top_node = evaluator.add_parallel(
        id="application_requirements_research",
        desc="Compile district-specific application requirement information (with official district source URLs) for certified teaching positions in the three specified districts.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_application_requirements(),
        template_class=ApplicationRequirementsExtraction,
        extraction_name="application_requirements_extraction"
    )

    # District configurations: (id, display_name, data, official_domain_hints)
    districts: List[Tuple[str, str, Optional[DistrictRequirements], List[str]]] = [
        (
            "aps_nm",
            "Albuquerque Public Schools (New Mexico)",
            extraction.albuquerque_public_schools_nm if extraction else None,
            ["aps.edu"]
        ),
        (
            "gcs_sc",
            "Greenville County Schools (South Carolina)",
            extraction.greenville_county_schools_sc if extraction else None,
            ["greenville.k12.sc.us"]
        ),
        (
            "misd_tx",
            "Mansfield Independent School District (Texas)",
            extraction.mansfield_isd_tx if extraction else None,
            ["mansfieldisd.org"]
        )
    ]

    # Build verification subtrees for each district in parallel at the root level
    for dist_id, dist_name, dist_data, domain_hints in districts:
        await _verify_district(
            evaluator,
            top_node,
            district_id=dist_id,
            district_name=dist_name,
            district_data=dist_data,
            domain_hints=domain_hints
        )

    # Return the evaluation summary
    return evaluator.get_summary()