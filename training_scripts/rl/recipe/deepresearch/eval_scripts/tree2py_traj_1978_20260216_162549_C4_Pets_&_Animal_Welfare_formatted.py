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
TASK_ID = "ca_aza_zoo"
TASK_DESCRIPTION = """
Identify an AZA-accredited zoo in California that actively participates in endangered species breeding programs, maintains formal written enrichment protocols meeting AZA standards, has documented veterinary care programs including necropsy and zoonotic disease training protocols, provides public educational access, and has no direct Animal Welfare Act violations in its most recent USDA inspection report.
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ZooExtraction(BaseModel):
    # Core facility identity
    facility_name: Optional[str] = None
    facility_website: Optional[str] = None

    # Per-criterion support URLs
    aza_accreditation_sources: List[str] = Field(default_factory=list)
    endangered_program_sources: List[str] = Field(default_factory=list)
    enrichment_sources: List[str] = Field(default_factory=list)
    veterinary_sources: List[str] = Field(default_factory=list)
    location_sources: List[str] = Field(default_factory=list)
    public_education_sources: List[str] = Field(default_factory=list)

    # USDA / AWA
    usda_most_recent_report_url: Optional[str] = None
    usda_other_urls: List[str] = Field(default_factory=list)

    # Optional descriptive claims from the answer (free text or lists)
    claimed_programs: List[str] = Field(default_factory=list)
    claimed_enrichment_summary: Optional[str] = None
    claimed_vetcare_summary: Optional[str] = None
    claimed_public_access_summary: Optional[str] = None
    location_state_claim: Optional[str] = None
    claimed_no_direct_violations: Optional[str] = None  # "yes" | "no" | "unknown"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_zoo() -> str:
    return """
    Extract structured information for a single zoo/facility that the answer claims satisfies the task requirements.
    If multiple facilities are mentioned, select the main/first one that is used to satisfy the requirements.
    Return only the facility explicitly discussed in the answer.

    Required fields:
    1) facility_name: The name of the zoo/facility (string).
    2) facility_website: A URL to the facility's official website (if provided in the answer, else null).

    Per-criterion source URLs (extract only URLs explicitly present in the answer text; do not invent):
    3) aza_accreditation_sources: Array of URLs that support AZA accreditation (e.g., AZA official accredited list page, or the facility’s official page stating AZA accreditation).
    4) endangered_program_sources: Array of URLs supporting participation in endangered species breeding or conservation programs (e.g., SSP, EEP, SAFE).
    5) enrichment_sources: Array of URLs supporting the existence of formal written enrichment protocols meeting AZA standards (e.g., an enrichment program page, animal welfare policy, etc.).
    6) veterinary_sources: Array of URLs supporting documented veterinary care programs that include necropsy procedures and zoonotic disease training protocols (e.g., veterinary services page, animal health policy, AZA accreditation documentation citing the facility).
    7) location_sources: Array of URLs supporting that the facility is physically located in California, United States (e.g., contact/visit page on the official website, Wikipedia page for the facility, etc.).
    8) public_education_sources: Array of URLs supporting that the facility is open to the public and provides educational programming or exhibits (e.g., tickets/visit page, education programs page).

    USDA AWA / inspection report:
    9) usda_most_recent_report_url: A single URL to the most recent USDA inspection report for the facility, if provided in the answer. Prefer a direct inspection report PDF or the USDA inspection report page that contains the latest report. If multiple are present, pick the most recent one. If none are present, set to null.
    10) usda_other_urls: Array of any other USDA AWA/inspection-related URLs mentioned in the answer, excluding the one selected as most recent.

    Optional textual claims for context (as presented in the answer; can be null/empty if not stated):
    11) claimed_programs: Array of program names or acronyms the answer claims (e.g., "SSP", "SAFE", species-specific SSPs).
    12) claimed_enrichment_summary: Short quote/summary about enrichment protocols from the answer (string or null).
    13) claimed_vetcare_summary: Short quote/summary about veterinary care (including necropsy and zoonoses training) from the answer (string or null).
    14) claimed_public_access_summary: Short quote/summary about public access and education from the answer (string or null).
    15) location_state_claim: The U.S. state claimed in the answer for the facility location (e.g., "California") if stated (string or null).
    16) claimed_no_direct_violations: One of "yes", "no", or "unknown" based on the answer’s explicit claim about having no direct AWA violations in the most recent USDA report.

    Notes:
    - Do not infer URLs. Only extract those explicitly provided in the answer.
    - URLs may appear as plain links or markdown links; extract the actual URL targets.
    - If a category has no URLs, return an empty array for that category.
    - If a field is not present in the answer, return null (for strings) or empty array (for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(*url_lists: List[str], website: Optional[str] = None) -> List[str]:
    """Merge multiple URL lists and an optional website into a unique, ordered list."""
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if isinstance(u, str):
                u = u.strip()
            if not u or not isinstance(u, str):
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    if website and isinstance(website, str):
        w = website.strip()
        if w and w not in seen:
            merged.append(w)
    return merged


def pick_usda_report_url(extracted: ZooExtraction) -> Optional[str]:
    """Choose the most appropriate USDA report URL to verify against."""
    if extracted.usda_most_recent_report_url and isinstance(extracted.usda_most_recent_report_url, str):
        return extracted.usda_most_recent_report_url.strip()
    if extracted.usda_other_urls:
        # Fallback to the first provided other USDA URL
        for u in extracted.usda_other_urls:
            if isinstance(u, str) and u.strip():
                return u.strip()
    return None


# --------------------------------------------------------------------------- #
# Verification node builders                                                  #
# --------------------------------------------------------------------------- #
async def verify_aza_accreditation(evaluator: Evaluator, parent, data: ZooExtraction) -> None:
    node = evaluator.add_sequential(
        id="AZA_accreditation",
        desc="The facility holds current AZA accreditation status as listed on the official AZA accredited institutions list",
        parent=parent,
        critical=True,
    )

    sources = merge_sources(data.aza_accreditation_sources, website=data.facility_website)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="AZA_accreditation_sources_provided",
        desc="Sources provided to support AZA accreditation claim",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="AZA_accreditation_supported",
        desc="AZA accreditation claim is supported by the provided sources",
        parent=node,
        critical=True
    )

    facility = data.facility_name or "the facility"
    claim = f"{facility} is currently accredited by the Association of Zoos and Aquariums (AZA)."
    add_ins = (
        "Verify using the provided URLs that the facility is AZA-accredited. Prefer the official AZA accredited "
        "institutions list. It is also acceptable if the facility’s official website states 'AZA accredited'. "
        "Allow minor name variations. If none of the provided URLs clearly show current AZA accreditation, judge as not supported."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=sources, additional_instruction=add_ins)


async def verify_endangered_programs(evaluator: Evaluator, parent, data: ZooExtraction) -> None:
    node = evaluator.add_sequential(
        id="endangered_species_programs",
        desc="The facility actively participates in at least one endangered species breeding or conservation program such as SSP, EEP, SAFE, or similar recognized programs",
        parent=parent,
        critical=True
    )

    sources = merge_sources(data.endangered_program_sources, website=data.facility_website)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="endangered_species_programs_sources_provided",
        desc="Sources provided to support endangered species program participation",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="endangered_species_programs_supported",
        desc="Endangered species breeding/conservation program participation is supported",
        parent=node,
        critical=True
    )

    programs_list = ", ".join(data.claimed_programs) if data.claimed_programs else "at least one recognized program"
    facility = data.facility_name or "the facility"
    claim = (
        f"{facility} actively participates in endangered species breeding or conservation program(s), such as {programs_list}."
    )
    add_ins = (
        "Look for explicit mentions of programs like 'Species Survival Plan' (SSP), 'AZA SAFE', 'EEP', or similar "
        "recognized conservation breeding programs. Participation can be species-specific (e.g., 'Snow Leopard SSP'). "
        "If the provided URLs do not clearly show participation in a recognized program, judge as not supported."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=sources, additional_instruction=add_ins)


async def verify_enrichment_protocols(evaluator: Evaluator, parent, data: ZooExtraction) -> None:
    node = evaluator.add_sequential(
        id="enrichment_protocols",
        desc="The facility maintains formal written enrichment protocols that promote species-appropriate behavioral opportunities, meeting AZA Standard 1.5.7 requirements",
        parent=parent,
        critical=True
    )

    sources = merge_sources(data.enrichment_sources, website=data.facility_website)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="enrichment_protocols_sources_provided",
        desc="Sources provided to support enrichment protocols claim",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="enrichment_protocols_supported",
        desc="Formal written enrichment protocols meeting AZA standards are supported",
        parent=node,
        critical=True
    )

    facility = data.facility_name or "the facility"
    enrichment_snippet = f" ({data.claimed_enrichment_summary})" if (data.claimed_enrichment_summary and data.claimed_enrichment_summary.strip()) else ""
    claim = (
        f"{facility} maintains formal written enrichment protocols that provide species-appropriate behavioral "
        f"opportunities and comply with AZA standards.{enrichment_snippet}"
    )
    add_ins = (
        "Verify that the provided sources indicate the existence of formal/written enrichment protocols or plans. "
        "Accept equivalent phrasing such as 'formal enrichment plan', 'documented enrichment program', or similar. "
        "Mentions that the program aligns with AZA standards are a strong indicator. If the sources only mention general "
        "enrichment without indicating formal/written protocols, judge as not supported."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=sources, additional_instruction=add_ins)


async def verify_veterinary_care(evaluator: Evaluator, parent, data: ZooExtraction) -> None:
    node = evaluator.add_sequential(
        id="veterinary_care_documentation",
        desc="The facility has documented veterinary care programs including necropsy procedures and zoonotic disease training protocols as required by AZA standards",
        parent=parent,
        critical=True
    )

    sources = merge_sources(data.veterinary_sources, website=data.facility_website)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="veterinary_care_documentation_sources_provided",
        desc="Sources provided to support documented veterinary care (necropsy + zoonotic disease training) claim",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="veterinary_care_documentation_supported",
        desc="Documented veterinary care (includes necropsy and zoonotic disease training) is supported",
        parent=node,
        critical=True
    )

    facility = data.facility_name or "the facility"
    vet_snippet = f" ({data.claimed_vetcare_summary})" if (data.claimed_vetcare_summary and data.claimed_vetcare_summary.strip()) else ""
    claim = (
        f"{facility} has documented veterinary care programs that include necropsy procedures and zoonotic disease "
        f"training protocols required by AZA standards.{vet_snippet}"
    )
    add_ins = (
        "Check that the provided URLs explicitly indicate both: (1) necropsy/post-mortem procedures (may be described as "
        "'necropsy' or 'post-mortem examination' or 'pathology'), and (2) training/protocols for zoonotic diseases (may be described as "
        "'zoonoses training', 'zoonotic disease training', 'biosecurity training', or equivalent). Both aspects must be present. "
        "If either necropsy or zoonoses training is not clearly indicated, judge as not supported."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=sources, additional_instruction=add_ins)


async def verify_california_location(evaluator: Evaluator, parent, data: ZooExtraction) -> None:
    node = evaluator.add_sequential(
        id="california_location",
        desc="The facility is physically located within the state of California, United States",
        parent=parent,
        critical=True
    )

    sources = merge_sources(data.location_sources, website=data.facility_website)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="california_location_sources_provided",
        desc="Sources provided to support California location",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="california_location_supported",
        desc="California location is supported by provided sources",
        parent=node,
        critical=True
    )

    facility = data.facility_name or "the facility"
    claim = f"{facility} is physically located in the state of California, United States."
    add_ins = (
        "Use the provided URLs (e.g., contact/visit page, address listing, Wikipedia entry) to confirm the facility is in "
        "California (CA), USA. Accept city names with 'CA' or explicit 'California'. If the location is outside California "
        "or unclear, judge as not supported."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=sources, additional_instruction=add_ins)


async def verify_public_education(evaluator: Evaluator, parent, data: ZooExtraction) -> None:
    node = evaluator.add_sequential(
        id="public_educational_access",
        desc="The facility is open to the public and provides educational programming or exhibits",
        parent=parent,
        critical=True
    )

    sources = merge_sources(data.public_education_sources, website=data.facility_website)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="public_educational_access_sources_provided",
        desc="Sources provided to support public access and education",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="public_educational_access_supported",
        desc="Public access and educational programming are supported",
        parent=node,
        critical=True
    )

    facility = data.facility_name or "the facility"
    educ_snippet = f" ({data.claimed_public_access_summary})" if (data.claimed_public_access_summary and data.claimed_public_access_summary.strip()) else ""
    claim = f"{facility} is open to the public and provides educational programming or exhibits.{educ_snippet}"
    add_ins = (
        "Confirm from the provided URLs that the facility is open to the public (e.g., hours, tickets, visit page) and "
        "that it provides educational programming (e.g., school programs, camps, classes, interpretive exhibits). "
        "If either public access or education is not clearly shown, judge as not supported."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=sources, additional_instruction=add_ins)


async def verify_awa_compliance(evaluator: Evaluator, parent, data: ZooExtraction) -> None:
    node = evaluator.add_sequential(
        id="AWA_compliance",
        desc="The facility has no direct Animal Welfare Act violations cited in its most recent USDA inspection report",
        parent=parent,
        critical=False  # Non-critical per rubric
    )

    # For AWA, require a USDA report URL; do not fallback to general website to satisfy source-grounding
    main_usda_url = pick_usda_report_url(data)
    evaluator.add_custom_node(
        result=bool(main_usda_url),
        id="AWA_compliance_usda_report_provided",
        desc="Most recent USDA inspection report URL is provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="AWA_compliance_supported",
        desc="No direct AWA violations in the most recent USDA inspection report",
        parent=node,
        critical=False  # Inside non-critical subtree; this leaf itself can be non-critical
    )

    facility = data.facility_name or "the facility"
    claim = f"The most recent USDA inspection report for {facility} shows no direct Animal Welfare Act violations."
    add_ins = (
        "Check ONLY the most recent USDA inspection report (the specific report URL provided). Determine whether any "
        "noncompliant items are classified as 'Direct' violations. If the report states 'No noncompliant items identified', "
        "that satisfies the claim. If the page is just a search index or does not clearly show the inspection details, judge as not supported. "
        "Ignore 'Indirect' or 'Repeat' unless explicitly marked 'Direct'."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=main_usda_url, additional_instruction=add_ins)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the California AZA-accredited zoo task.
    """

    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Root parallel: criteria evaluated independently
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

    # Extract structured info
    extracted: ZooExtraction = await evaluator.extract(
        prompt=prompt_extract_zoo(),
        template_class=ZooExtraction,
        extraction_name="zoo_extraction"
    )

    # Add quick custom info for debugging
    evaluator.add_custom_info(
        info={
            "facility_name": extracted.facility_name,
            "facility_website": extracted.facility_website,
            "counts": {
                "aza_sources": len(extracted.aza_accreditation_sources),
                "endangered_program_sources": len(extracted.endangered_program_sources),
                "enrichment_sources": len(extracted.enrichment_sources),
                "veterinary_sources": len(extracted.veterinary_sources),
                "location_sources": len(extracted.location_sources),
                "public_education_sources": len(extracted.public_education_sources),
                "usda_other_urls": len(extracted.usda_other_urls),
                "usda_most_recent_report_present": bool(extracted.usda_most_recent_report_url),
            }
        },
        info_type="extraction_overview"
    )

    # Build and verify each criterion subtree
    await verify_aza_accreditation(evaluator, root, extracted)
    await verify_endangered_programs(evaluator, root, extracted)
    await verify_enrichment_protocols(evaluator, root, extracted)
    await verify_veterinary_care(evaluator, root, extracted)
    await verify_california_location(evaluator, root, extracted)
    await verify_public_education(evaluator, root, extracted)
    await verify_awa_compliance(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()