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
TASK_ID = "space_health_collab"
TASK_DESCRIPTION = (
    "Identify a research collaboration between a commercial space or technology company and at least one U.S. academic institution "
    "that meets ALL of the following criteria: (1) The research focuses on human health or physiology in space environments, "
    "(2) The collaboration has produced a peer-reviewed publication in a scientific journal OR is part of a registered clinical trial, "
    "(3) The publication was published OR the clinical trial was initiated between January 1, 2023 and December 31, 2025, "
    "(4) The research involves data collection from actual space missions or clinical trials (not purely theoretical or simulation-based), "
    "and (5) The corresponding author or principal investigator is affiliated with a university medical school, school of medicine, "
    "or department of medicine. Provide the name of the research collaboration or project, and include a reference URL that verifies this information."
)

DATE_RANGE_START = "2023-01-01"
DATE_RANGE_END = "2025-12-31"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NamedURL(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None


class PublicationInfo(BaseModel):
    title: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    corresponding_author_affiliation: Optional[str] = None
    corresponding_author_department: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ClinicalTrialInfo(BaseModel):
    registry: Optional[str] = None  # e.g., ClinicalTrials.gov
    trial_id: Optional[str] = None
    start_date: Optional[str] = None
    principal_investigator_affiliation: Optional[str] = None
    principal_investigator_department: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CollaborationExtraction(BaseModel):
    project_name: Optional[str] = None
    company_entities: List[NamedURL] = Field(default_factory=list)
    academic_entities: List[NamedURL] = Field(default_factory=list)
    publication: Optional[PublicationInfo] = None
    clinical_trial: Optional[ClinicalTrialInfo] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_collaboration() -> str:
    return (
        "Extract the research collaboration or project details mentioned in the answer. Return a JSON object with the following fields:\n"
        "- project_name: The name of the research collaboration or project (if provided).\n"
        "- company_entities: A list of objects, each with 'name' and 'url' (if a URL is provided) for commercial space or technology companies involved.\n"
        "- academic_entities: A list of objects, each with 'name' and 'url' (if a URL is provided) for academic institutions involved.\n"
        "- publication: If a peer-reviewed publication is provided, include an object with:\n"
        "    * title: Article title\n"
        "    * journal: Journal name\n"
        "    * publication_date: Publication date as presented\n"
        "    * corresponding_author_affiliation: Affiliation of the corresponding author (full string as in the answer)\n"
        "    * corresponding_author_department: Department/School (e.g., 'School of Medicine', 'Department of Medicine') if explicitly mentioned\n"
        "    * urls: All URLs referencing the publication (journal page, DOI, PubMed, etc.)\n"
        "- clinical_trial: If a registered clinical trial is provided, include an object with:\n"
        "    * registry: Name of the registry (e.g., ClinicalTrials.gov)\n"
        "    * trial_id: Identifier (e.g., NCT number)\n"
        "    * start_date: Trial start/initiation date as presented\n"
        "    * principal_investigator_affiliation: PI's affiliation (full string as in the answer)\n"
        "    * principal_investigator_department: Department/School (e.g., 'School of Medicine', 'Department of Medicine') if explicitly mentioned\n"
        "    * urls: All URLs referencing the trial (registry page, official trial page)\n"
        "- reference_urls: All other URLs provided in the answer that support the collaboration.\n"
        "Rules:\n"
        "1) Extract only what is explicitly present in the answer. Do not invent names or URLs.\n"
        "2) URLs can be plain or markdown; return the actual URL strings. If protocol is missing, prepend http://.\n"
        "3) If something is missing, set the field to null or empty list accordingly.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list_str(items: List[Optional[str]]) -> List[str]:
    return [x.strip() for x in items if isinstance(x, str) and x.strip()]

def _entity_names(entities: List[NamedURL]) -> List[str]:
    return _safe_list_str([e.name for e in entities])

def _entity_urls(entities: List[NamedURL]) -> List[str]:
    return _safe_list_str([e.url for e in entities])

def _merge_sources(extracted: CollaborationExtraction) -> List[str]:
    merged: List[str] = []
    merged.extend(_entity_urls(extracted.company_entities))
    merged.extend(_entity_urls(extracted.academic_entities))
    if extracted.publication:
        merged.extend(_safe_list_str(extracted.publication.urls))
    if extracted.clinical_trial:
        merged.extend(_safe_list_str(extracted.clinical_trial.urls))
    merged.extend(_safe_list_str(extracted.reference_urls))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in merged:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique

def _pub_sources(extracted: CollaborationExtraction) -> List[str]:
    return _safe_list_str(extracted.publication.urls) if extracted.publication else []

def _trial_sources(extracted: CollaborationExtraction) -> List[str]:
    return _safe_list_str(extracted.clinical_trial.urls) if extracted.clinical_trial else []

def _company_sources(extracted: CollaborationExtraction) -> List[str]:
    urls = _entity_urls(extracted.company_entities)
    return urls if urls else _merge_sources(extracted)

def _academic_sources(extracted: CollaborationExtraction) -> List[str]:
    urls = _entity_urls(extracted.academic_entities)
    return urls if urls else _merge_sources(extracted)

def _evidence_sources_for_output(extracted: CollaborationExtraction) -> List[str]:
    # Prioritize publication/trial URLs, fallback to reference URLs
    urls = []
    urls.extend(_pub_sources(extracted))
    urls.extend(_trial_sources(extracted))
    if not urls:
        urls.extend(_safe_list_str(extracted.reference_urls))
    return urls

def _has_any_evidence_url(extracted: CollaborationExtraction) -> bool:
    return len(_merge_sources(extracted)) > 0


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, extracted: CollaborationExtraction) -> None:
    # Main critical node mirroring rubric root
    collab_node = evaluator.add_parallel(
        id="Research_Collaboration_Identification",
        desc="Identifies a research collaboration between a commercial space/technology company and academic institution(s) focused on human health in space, with verifiable publication or clinical trial output",
        parent=root_node,
        critical=True
    )

    # Gate: ensure at least one evidence URL is present to make verification meaningful
    gate_node = evaluator.add_custom_node(
        result=_has_any_evidence_url(extracted),
        id="Evidence_URLs_Present",
        desc="At least one reference or evidence URL is provided (publication, clinical trial, or other reference link)",
        parent=collab_node,
        critical=True
    )

    # 1) Commercial company involvement
    companies = _entity_names(extracted.company_entities)
    company_claim_names = ", ".join(companies) if companies else "at least one commercial company"
    company_node = evaluator.add_leaf(
        id="Commercial_Company_Involvement",
        desc="The research involves a commercial space or technology company (not purely government agencies)",
        parent=collab_node,
        critical=True
    )
    company_claim = (
        f"The identified collaboration involves {company_claim_names} as a commercial space or technology company partner."
        " Confirm that a private/commercial entity is involved (not solely government agencies)."
    )
    await evaluator.verify(
        claim=company_claim,
        node=company_node,
        sources=_company_sources(extracted),
        additional_instruction="Use the provided URLs to confirm that a private/commercial company is involved in the collaboration (e.g., SpaceX, Blue Origin, Axiom Space, Boeing, Lockheed Martin, Microsoft, etc.). If a government agency appears, it does not negate the requirement as long as a private company is also a partner."
    )

    # 2) Academic institution involvement
    institutions = _entity_names(extracted.academic_entities)
    institution_claim_names = ", ".join(institutions) if institutions else "at least one academic institution"
    academic_node = evaluator.add_leaf(
        id="Academic_Institution_Involvement",
        desc="The research involves at least one academic institution or university",
        parent=collab_node,
        critical=True
    )
    academic_claim = (
        f"The collaboration involves {institution_claim_names} as an academic institution or university partner."
    )
    await evaluator.verify(
        claim=academic_claim,
        node=academic_node,
        sources=_academic_sources(extracted),
        additional_instruction="Confirm that at least one university or academic institution is a formal collaborator on this research project."
    )

    # 3) Human health/physiology focus
    hh_node = evaluator.add_leaf(
        id="Human_Health_Physiology_Focus",
        desc="The research specifically focuses on human health, physiology, or medical aspects of space environments",
        parent=collab_node,
        critical=True
    )
    hh_claim = "The research focuses on human health, physiology, or medical aspects in space environments."
    await evaluator.verify(
        claim=hh_claim,
        node=hh_node,
        sources=_evidence_sources_for_output(extracted),
        additional_instruction="Look for language indicating human physiology, biomedical outcomes, medical risk mitigation, clinical endpoints, or human health in microgravity/spaceflight."
    )

    # 4) Peer-reviewed publication OR registered clinical trial
    output_node = evaluator.add_leaf(
        id="Peer_Reviewed_Publication_Or_Clinical_Trial",
        desc="The collaboration has produced at least one peer-reviewed publication in a scientific journal OR is part of a registered clinical trial",
        parent=collab_node,
        critical=True
    )
    output_claim = (
        "This collaboration has produced at least one peer-reviewed scientific journal publication OR is part of a registered clinical trial."
    )
    await evaluator.verify(
        claim=output_claim,
        node=output_node,
        sources=_evidence_sources_for_output(extracted),
        additional_instruction="Accept either a peer-reviewed journal article (journal page, DOI, PubMed) or a registered clinical trial (e.g., ClinicalTrials.gov NCT record)."
    )

    # 5) Publication or trial date range between 2023-01-01 and 2025-12-31
    date_node = evaluator.add_leaf(
        id="Publication_Trial_Date_Range",
        desc="The publication date or clinical trial start date falls between January 1, 2023 and December 31, 2025",
        parent=collab_node,
        critical=True
    )
    pub_date = extracted.publication.publication_date if extracted.publication else None
    trial_start = extracted.clinical_trial.start_date if extracted.clinical_trial else None
    date_detail = (
        f"Publication date: {pub_date}" if pub_date else (f"Clinical trial start date: {trial_start}" if trial_start else "Date not explicitly provided")
    )
    date_claim = (
        f"The publication date (if present) or clinical trial start/initiation date (if present) falls between {DATE_RANGE_START} and {DATE_RANGE_END}. "
        f"{date_detail}."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=_evidence_sources_for_output(extracted),
        additional_instruction=f"Check the explicit date shown on the journal page, DOI page, PubMed, or trial registry page. It must lie in [{DATE_RANGE_START}, {DATE_RANGE_END}] inclusive. If both are present, it is sufficient that at least one fits the range."
    )

    # 6) U.S. academic institution involved
    us_node = evaluator.add_leaf(
        id="US_Academic_Institution",
        desc="At least one of the academic institutions involved is located in the United States",
        parent=collab_node,
        critical=True
    )
    us_claim = (
        f"At least one academic institution involved in this collaboration is a U.S. institution. Institutions listed: {institution_claim_names}."
    )
    await evaluator.verify(
        claim=us_claim,
        node=us_node,
        sources=_academic_sources(extracted),
        additional_instruction="Verify that at least one institution is U.S.-based (e.g., the campus location/address or affiliation indicating United States). If institution URLs are missing, use the publication or trial pages where the affiliation location appears."
    )

    # 7) Actual mission or clinical trial data (not purely theoretical)
    data_node = evaluator.add_leaf(
        id="Actual_Mission_Trial_Data",
        desc="The research involves data collection from actual space missions or clinical trials, not purely theoretical or simulation-based research",
        parent=collab_node,
        critical=True
    )
    data_claim = (
        "The research uses empirical data collected from actual space missions (e.g., ISS, spaceflight) or clinical trial participants, rather than purely theoretical or simulation-based studies."
    )
    await evaluator.verify(
        claim=data_claim,
        node=data_node,
        sources=_evidence_sources_for_output(extracted),
        additional_instruction="Look for explicit mentions of human subjects, clinical trial enrollment, astronaut/cosmonaut data, space mission experiments, ISS study cohorts, or flight samples."
    )

    # 8) Medical school affiliation for corresponding author or PI
    med_node = evaluator.add_leaf(
        id="Medical_School_Affiliation",
        desc="The corresponding author or principal investigator is affiliated with a university medical school, school of medicine, or department of medicine",
        parent=collab_node,
        critical=True
    )
    affil_text = None
    affil_sources = []
    if extracted.publication and _pub_sources(extracted):
        affil_text = extracted.publication.corresponding_author_affiliation or extracted.publication.corresponding_author_department
        affil_sources = _pub_sources(extracted)
    elif extracted.clinical_trial and _trial_sources(extracted):
        affil_text = extracted.clinical_trial.principal_investigator_affiliation or extracted.clinical_trial.principal_investigator_department
        affil_sources = _trial_sources(extracted)
    else:
        affil_sources = _evidence_sources_for_output(extracted)

    med_claim_detail = f"Example affiliation: {affil_text}." if affil_text else "Affiliation text not explicitly provided in the answer."
    med_claim = (
        f"The corresponding author (if a publication) or principal investigator (if a clinical trial) is affiliated with a university medical school, school of medicine, or department of medicine. {med_claim_detail}"
    )
    await evaluator.verify(
        claim=med_claim,
        node=med_node,
        sources=affil_sources,
        additional_instruction="Look for affiliation strings such as 'School of Medicine', 'Medical School', or 'Department of Medicine' tied to a university. Minor variations or abbreviations are acceptable."
    )

    # 9) Reference URL provided that supports the collaboration
    ref_node = evaluator.add_leaf(
        id="Reference_URL",
        desc="A valid reference URL is provided that supports the identified research collaboration",
        parent=collab_node,
        critical=True
    )
    ref_claim = "The provided reference URL(s) support and describe the identified collaboration/project and its key parties or outputs."
    await evaluator.verify(
        claim=ref_claim,
        node=ref_node,
        sources=_safe_list_str(extracted.reference_urls),
        additional_instruction="The URL(s) should substantively describe the collaboration, project, and relevant details; promotional pages are acceptable if they explicitly confirm the collaboration."
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_collaboration(),
        template_class=CollaborationExtraction,
        extraction_name="collaboration_extraction"
    )

    evaluator.add_custom_info(
        info={
            "requirements": [
                "Commercial company involvement",
                "Academic institution involvement (at least one U.S.)",
                "Human health/physiology focus in space",
                "Peer‑reviewed publication OR registered clinical trial",
                f"Publication or trial date in [{DATE_RANGE_START}, {DATE_RANGE_END}]",
                "Actual mission/clinical trial data (not purely theoretical)",
                "Corresponding author or PI affiliated with medical school/school of medicine/department of medicine",
                "Reference URL provided"
            ]
        },
        info_type="constraints"
    )

    await build_verification_tree(evaluator, root, extracted)

    return evaluator.get_summary()