import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple, Set
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nsf_career_prep_plan"
TASK_DESCRIPTION = (
    "You are a postdoctoral researcher in Computer Science preparing to apply for an NSF CAREER grant. "
    "Create a detailed research preparation document that addresses four domains (Publication Strategy, "
    "Grant Application Components, Research Compliance Requirements, Collaboration Framework), providing "
    "specific, verifiable information with URL references from authoritative sources. Information must be "
    "current and applicable to U.S.-based academic research as of March 2026, and each major area should "
    "include at least one supporting URL reference."
)

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ItemSection(BaseModel):
    text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CAREERPrepExtraction(BaseModel):
    # 1. Publication Strategy
    quartile_classification: Optional[ItemSection] = None
    review_timeline: Optional[ItemSection] = None
    peer_review_type: Optional[ItemSection] = None

    core_ranking_system: Optional[ItemSection] = None
    acceptance_rate_ranges: Optional[ItemSection] = None
    program_committee_structure: Optional[ItemSection] = None

    open_access_license_types: Optional[ItemSection] = None
    repository_selection_criteria: Optional[ItemSection] = None

    # 2. Grant Application Components
    direct_cost_categories: Optional[ItemSection] = None
    fa_indirect_costs: Optional[ItemSection] = None

    dmp_page_limit: Optional[ItemSection] = None
    dmp_required_elements: Optional[ItemSection] = None

    career_stage_requirements: Optional[ItemSection] = None
    doctoral_degree_field_requirements: Optional[ItemSection] = None

    prior_support_required_content: Optional[ItemSection] = None

    # 3. Research Compliance Requirements
    expedited_irb_criteria: Optional[ItemSection] = None
    minimal_risk_definition: Optional[ItemSection] = None

    minimum_retention_period: Optional[ItemSection] = None
    governing_regulations: Optional[ItemSection] = None

    informed_consent_components: Optional[ItemSection] = None
    ethics_committee_review_steps: Optional[ItemSection] = None

    # 4. Collaboration Framework
    mou_essential_elements: Optional[ItemSection] = None
    publication_rights_clauses: Optional[ItemSection] = None
    data_sharing_practices: Optional[ItemSection] = None

    # Global statements
    us_based_applicability_statement: Optional[str] = None
    asof_march_2026_statement: Optional[str] = None

    # Collected sources
    global_sources: List[str] = Field(default_factory=list)


# List of all substantive section attribute names requiring URL support
SUBSTANTIVE_SECTION_ATTRS = [
    # Publication Strategy
    "quartile_classification", "review_timeline", "peer_review_type",
    "core_ranking_system", "acceptance_rate_ranges", "program_committee_structure",
    "open_access_license_types", "repository_selection_criteria",
    # Grant Application Components
    "direct_cost_categories", "fa_indirect_costs",
    "dmp_page_limit", "dmp_required_elements",
    "career_stage_requirements", "doctoral_degree_field_requirements",
    "prior_support_required_content",
    # Research Compliance Requirements
    "expedited_irb_criteria", "minimal_risk_definition",
    "minimum_retention_period", "governing_regulations",
    "informed_consent_components", "ethics_committee_review_steps",
    # Collaboration Framework
    "mou_essential_elements", "publication_rights_clauses", "data_sharing_practices",
]


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract structured information from the answer for the NSF CAREER preparation plan. For each requested item, return:
- text: The exact statement(s) the answer makes for that specific item (paraphrase minimally; keep specific details such as numbers, page limits, definitions, roles).
- sources: A list of all URLs explicitly cited in the answer that directly support that item. Include only valid URLs. If none cited for that item, return an empty list.

Return a JSON object matching this schema:

{
  "quartile_classification": {"text": str|null, "sources": [str, ...]},
  "review_timeline": {"text": str|null, "sources": [str, ...]},
  "peer_review_type": {"text": str|null, "sources": [str, ...]},

  "core_ranking_system": {"text": str|null, "sources": [str, ...]},
  "acceptance_rate_ranges": {"text": str|null, "sources": [str, ...]},
  "program_committee_structure": {"text": str|null, "sources": [str, ...]},

  "open_access_license_types": {"text": str|null, "sources": [str, ...]},
  "repository_selection_criteria": {"text": str|null, "sources": [str, ...]},

  "direct_cost_categories": {"text": str|null, "sources": [str, ...]},
  "fa_indirect_costs": {"text": str|null, "sources": [str, ...]},

  "dmp_page_limit": {"text": str|null, "sources": [str, ...]},
  "dmp_required_elements": {"text": str|null, "sources": [str, ...]},

  "career_stage_requirements": {"text": str|null, "sources": [str, ...]},
  "doctoral_degree_field_requirements": {"text": str|null, "sources": [str, ...]},

  "prior_support_required_content": {"text": str|null, "sources": [str, ...]},

  "expedited_irb_criteria": {"text": str|null, "sources": [str, ...]},
  "minimal_risk_definition": {"text": str|null, "sources": [str, ...]},

  "minimum_retention_period": {"text": str|null, "sources": [str, ...]},
  "governing_regulations": {"text": str|null, "sources": [str, ...]},

  "informed_consent_components": {"text": str|null, "sources": [str, ...]},
  "ethics_committee_review_steps": {"text": str|null, "sources": [str, ...]},

  "mou_essential_elements": {"text": str|null, "sources": [str, ...]},
  "publication_rights_clauses": {"text": str|null, "sources": [str, ...]},
  "data_sharing_practices": {"text": str|null, "sources": [str, ...]},

  "us_based_applicability_statement": str|null,
  "asof_march_2026_statement": str|null,

  "global_sources": [str, ...]
}

Instructions:
- Extract only what appears in the answer. Do not invent content.
- For each 'sources' list, include only URLs explicitly present in that part of the answer. If the answer mentions a source without a URL, do not include it.
- global_sources must be a deduplicated list of all URLs cited anywhere in the answer (regardless of item). Ensure full absolute URLs with protocol.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_snippet(text: Optional[str], max_len: int = 800) -> str:
    if not text:
        return ""
    t = text.strip()
    return (t[: max_len - 3] + "...") if len(t) > max_len else t


def collect_all_sources(data: CAREERPrepExtraction) -> List[str]:
    urls: Set[str] = set()

    # Gather from each ItemSection
    for attr in SUBSTANTIVE_SECTION_ATTRS:
        sec: Optional[ItemSection] = getattr(data, attr, None)
        if sec and sec.sources:
            for u in sec.sources:
                if isinstance(u, str) and u.strip():
                    urls.add(u.strip())

    # Add global
    for u in data.global_sources or []:
        if isinstance(u, str) and u.strip():
            urls.add(u.strip())

    return sorted(urls)


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def is_authoritative_url(url: str) -> bool:
    d = _domain_from_url(url)
    if not d:
        return False

    # TLD patterns commonly used for authoritative sources
    if d.endswith(".gov") or ".gov." in d:
        return True
    if d.endswith(".mil") or ".mil." in d:
        return True
    # Education across ccTLDs
    if d.endswith(".edu") or ".edu." in d or d.endswith(".ac.uk") or ".ac." in d:
        return True

    # Established academic/professional resources and organizations (heuristic list)
    WHITELIST = [
        "nsf.gov", "nih.gov", "hhs.gov", "ohrp.hhs.gov", "grants.gov",
        "core.edu.au", "scimagojr.com", "clarivate.com",
        "acm.org", "ieee.org", "computer.org", "aaai.org",
        "springer.com", "springernature.com", "nature.com", "wiley.com", "elsevier.com", "sagepub.com",
        "creativecommons.org", "datacite.org", "orcid.org", "zenodo.org", "figshare.com", "osf.io",
        "datadryad.org", "dryad.org", "icpsr.umich.edu", "pnas.org", "aps.org", "asme.org",
        "publicationethics.org", "icmje.org", "apa.org",
        "nist.gov", "ed.gov", "loc.gov", "archives.gov",
    ]
    if any(d == w or d.endswith("." + w) for w in WHITELIST):
        return True

    return False


def all_authoritative(urls: List[str]) -> bool:
    if not urls:
        return False
    return all(is_authoritative_url(u) for u in urls)


def section_has_sources(data: CAREERPrepExtraction, attr: str) -> bool:
    sec: Optional[ItemSection] = getattr(data, attr, None)
    return bool(sec and sec.sources and len(sec.sources) > 0)


# --------------------------------------------------------------------------- #
# Verification helper constructors                                            #
# --------------------------------------------------------------------------- #
async def add_section_verification(
    evaluator: Evaluator,
    parent,
    section_id: str,
    section_desc: str,
    section_item: Optional[ItemSection],
    add_ins: Optional[str] = None,
) -> None:
    """
    Build a small sequential gate for a single requirement item:
      1) existence (text present + at least one URL)
      2) source-backed accuracy verification of the provided text
    All nodes here are critical (parent in rubric is critical).
    """
    group = evaluator.add_sequential(
        id=section_id,
        desc=section_desc,
        parent=parent,
        critical=True,
    )

    provided = evaluator.add_custom_node(
        result=bool(section_item and section_item.text and section_item.text.strip()) and bool(section_item and section_item.sources),
        id=f"{section_id}_provided",
        desc=f"{section_desc} — content provided with at least one URL citation",
        parent=group,
        critical=True,
    )

    # Verification leaf
    verify_leaf = evaluator.add_leaf(
        id=f"{section_id}_supported",
        desc=f"{section_desc} — statement is accurate and supported by cited sources",
        parent=group,
        critical=True,
    )

    statement = safe_snippet(section_item.text if section_item else "")
    sources = (section_item.sources if section_item else []) or []

    # Even if sources are empty, the sequential strategy + auto preconditions will skip this verification after 'provided' fails.
    instruction = add_ins or "Verify that the statement aligns with the content of the cited authoritative sources; allow minor paraphrasing but not changes in meaning."
    await evaluator.verify(
        claim=f"The following statement in the answer is correct according to the cited sources: \"{statement}\"",
        node=verify_leaf,
        sources=sources,
        additional_instruction=instruction,
    )


async def add_scope_statement_verification(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    statement_text: Optional[str],
    check_kind: str,  # "US_scope" or "As_of_date"
) -> None:
    """
    Build a small sequential gate for scope/time statements that are properties of the answer itself.
      1) existence (explicit statement is provided)
      2) simple verification on the answer text (no external URLs needed)
    """
    group = evaluator.add_sequential(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=True,
    )

    provided = evaluator.add_custom_node(
        result=bool(statement_text and statement_text.strip()),
        id=f"{node_id}_provided",
        desc=f"{desc} — explicit statement is present in the answer",
        parent=group,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id=f"{node_id}_verified",
        desc=f"{desc} — explicit statement correctly reflects the intended scope/timing",
        parent=group,
        critical=True,
    )

    if check_kind == "US_scope":
        claim = "The answer explicitly indicates that its guidance is intended for U.S.-based academic research."
        add_ins = "Accept phrasing such as 'U.S.-based', 'United States', or explicit reliance on U.S. agencies as scope qualifiers."
    else:
        claim = "The answer explicitly indicates that the guidance is current as of March 2026."
        add_ins = "Accept phrasing such as 'as of March 2026', 'current to March 2026', or an explicit 'updated March 2026' note."

    # Simple verification (no external URLs, since this checks the presence of a statement within the answer itself)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction=f"{add_ins}\nHere is the extracted statement from the answer to help you locate it:\n\"{safe_snippet(statement_text)}\"",
    )


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_publication_strategy_nodes(evaluator: Evaluator, parent, data: CAREERPrepExtraction) -> None:
    pub_node = evaluator.add_parallel(
        id="Publication_Strategy_Documentation",
        desc="Publication strategy documentation requirements (journals, conferences, open access)",
        parent=parent,
        critical=True,
    )

    # Q1 journal requirements
    q1_node = evaluator.add_parallel(
        id="Q1_Journal_Requirements",
        desc="Q1 journal requirements documentation",
        parent=pub_node,
        critical=True,
    )

    await add_section_verification(
        evaluator, q1_node,
        "Quartile_Classification",
        "Explain how journals are classified into quartiles (Q1–Q4)",
        data.quartile_classification,
        add_ins="Confirm that quartiles are within subject categories and based on ranking/percentiles per the referenced system (e.g., SJR, JCR), as the answer states."
    )

    await add_section_verification(
        evaluator, q1_node,
        "Review_Timeline",
        "Provide a typical peer review timeline range for academic journals",
        data.review_timeline,
        add_ins="Check that the timeline range is realistic and backed by sources (e.g., weeks to months); allow ranges or distributions as long as sources support them."
    )

    await add_section_verification(
        evaluator, q1_node,
        "Peer_Review_Type",
        "Explain the difference between single-blind and double-blind peer review",
        data.peer_review_type,
        add_ins="Verify the definitions and differences of single-blind vs double-blind peer review as per the cited sources."
    )

    # Conference ranking standards
    conf_node = evaluator.add_parallel(
        id="Conference_Ranking_Standards",
        desc="Conference ranking standards documentation",
        parent=pub_node,
        critical=True,
    )

    await add_section_verification(
        evaluator, conf_node,
        "CORE_Ranking_System",
        "Explain the CORE conference ranking tiers (A*, A, B, C) used in computer science",
        data.core_ranking_system,
        add_ins="Confirm that the description of CORE tiers matches the authoritative CORE source(s)."
    )

    await add_section_verification(
        evaluator, conf_node,
        "Acceptance_Rate_Ranges",
        "Provide typical acceptance rate ranges for top-tier CS conferences",
        data.acceptance_rate_ranges,
        add_ins="Check that the acceptance rates and ranges are supported by cited historical stats from authoritative sources (conference sites, societies)."
    )

    await add_section_verification(
        evaluator, conf_node,
        "Program_Committee_Structure",
        "Describe the standard structure of a conference program committee (key roles)",
        data.program_committee_structure,
        add_ins="Verify program committee roles and structure (e.g., PC chairs, area chairs, reviewers) per the cited sources."
    )

    # Open access publication
    oa_node = evaluator.add_parallel(
        id="Open_Access_Publication",
        desc="Open access publication documentation",
        parent=pub_node,
        critical=True,
    )

    await add_section_verification(
        evaluator, oa_node,
        "Open_Access_License_Types",
        "Identify standard open access license types (e.g., CC-0, CC-BY) and explain their meanings",
        data.open_access_license_types,
        add_ins="Verify that the license types and meanings (e.g., CC BY, CC0) match authoritative sources such as Creative Commons."
    )

    await add_section_verification(
        evaluator, oa_node,
        "Repository_Selection_Criteria",
        "List key criteria for selecting research data repositories, including persistent identifiers, typical size limits, and access policies",
        data.repository_selection_criteria,
        add_ins="Ensure the criteria (e.g., PIDs/DOIs, storage/size, access policies) are supported by authoritative guidance."
    )


async def build_grant_components_nodes(evaluator: Evaluator, parent, data: CAREERPrepExtraction) -> None:
    grant_node = evaluator.add_parallel(
        id="Grant_Application_Components",
        desc="NSF CAREER grant application components documentation",
        parent=parent,
        critical=True,
    )

    # Budget Structure
    budget_node = evaluator.add_parallel(
        id="Budget_Structure",
        desc="Budget structure requirements documentation",
        parent=grant_node,
        critical=True,
    )

    await add_section_verification(
        evaluator, budget_node,
        "Direct_Cost_Categories",
        "List all required direct cost categories that must be included in a grant budget",
        data.direct_cost_categories,
        add_ins="Confirm that listed direct cost categories align with U.S. federal/NSF or institutional guidance."
    )

    await add_section_verification(
        evaluator, budget_node,
        "FA_Indirect_Costs",
        "Explain how F&A indirect costs are calculated, including typical rate ranges",
        data.fa_indirect_costs,
        add_ins="Verify how F&A/indirects are applied (base, rate) and that example rate ranges are realistic for U.S. universities."
    )

    # Data Management Plan
    dmp_node = evaluator.add_parallel(
        id="Data_Management_Plan",
        desc="NSF data management and sharing plan requirements documentation",
        parent=grant_node,
        critical=True,
    )

    await add_section_verification(
        evaluator, dmp_node,
        "DMP_Page_Limit",
        "Specify the NSF-required page limit for data management and sharing plans",
        data.dmp_page_limit,
        add_ins="Verify the page limit language matches current NSF requirements."
    )

    await add_section_verification(
        evaluator, dmp_node,
        "DMP_Required_Elements",
        "List the required elements that must be included in NSF data management and sharing plans",
        data.dmp_required_elements,
        add_ins="Check that listed elements (types, standards, access/sharing, preservation, responsibilities) align with NSF guidance."
    )

    # Eligibility
    elig_node = evaluator.add_parallel(
        id="Eligibility_Criteria",
        desc="NSF CAREER eligibility requirements documentation",
        parent=grant_node,
        critical=True,
    )

    await add_section_verification(
        evaluator, elig_node,
        "Career_Stage_Requirements",
        "Specify the career stage requirements for NSF CAREER eligibility",
        data.career_stage_requirements,
        add_ins="Confirm timelines and faculty status criteria as per NSF CAREER eligibility."
    )

    await add_section_verification(
        evaluator, elig_node,
        "Doctoral_Degree_and_Field_Requirements",
        "Identify the doctoral degree and research field requirements for NSF CAREER eligibility",
        data.doctoral_degree_field_requirements,
        add_ins="Verify doctoral degree and disciplinary/field expectations as per NSF CAREER guidance."
    )

    # Prior support
    prior_node = evaluator.add_parallel(
        id="Prior_Support_Documentation",
        desc="Results from Prior Support requirements documentation",
        parent=grant_node,
        critical=True,
    )

    await add_section_verification(
        evaluator, prior_node,
        "Prior_Support_Required_Content",
        "Explain what information must be included in the 'Results from Prior Support' section if the applicant has previous NSF funding",
        data.prior_support_required_content,
        add_ins="Check that the elements required by NSF (e.g., award details, results, products, data sharing) are described accurately."
    )


async def build_compliance_nodes(evaluator: Evaluator, parent, data: CAREERPrepExtraction) -> None:
    comp_node = evaluator.add_parallel(
        id="Research_Compliance_Requirements",
        desc="Research ethics and compliance procedures documentation",
        parent=parent,
        critical=True,
    )

    # IRB Review
    irb_node = evaluator.add_parallel(
        id="IRB_Review",
        desc="IRB review requirements documentation",
        parent=comp_node,
        critical=True,
    )

    await add_section_verification(
        evaluator, irb_node,
        "Expedited_IRB_Criteria",
        "Explain criteria determining whether research is eligible for expedited IRB review",
        data.expedited_irb_criteria,
        add_ins="Verify that the criteria for expedited review match U.S. regulations/policies (e.g., HHS/OHRP), as cited."
    )

    await add_section_verification(
        evaluator, irb_node,
        "Minimal_Risk_Definition",
        "Define what constitutes 'minimal risk' in the context of IRB review",
        data.minimal_risk_definition,
        add_ins="Confirm the definition of minimal risk is accurate per U.S. federal IRB regulations/guidance."
    )

    # Data Retention
    dr_node = evaluator.add_parallel(
        id="Data_Retention",
        desc="Research data retention requirements documentation",
        parent=comp_node,
        critical=True,
    )

    await add_section_verification(
        evaluator, dr_node,
        "Minimum_Retention_Period",
        "Specify the minimum period that research data must be retained after project completion",
        data.minimum_retention_period,
        add_ins="Verify the retention period is supported by U.S. policy or funder requirements cited."
    )

    await add_section_verification(
        evaluator, dr_node,
        "Governing_Regulations",
        "Identify key regulations governing research data retention (e.g., OMB Circular A-110 or funder-specific requirements)",
        data.governing_regulations,
        add_ins="Check that the listed regulations/policies are relevant and described correctly."
    )

    # Research Ethics
    re_node = evaluator.add_parallel(
        id="Research_Ethics",
        desc="Research ethics documentation (informed consent + ethics review process)",
        parent=comp_node,
        critical=True,
    )

    await add_section_verification(
        evaluator, re_node,
        "Informed_Consent_Components",
        "List required components that must be included in informed consent documents for research participants",
        data.informed_consent_components,
        add_ins="Confirm that consent components align with U.S. human subjects regulations/guidance."
    )

    await add_section_verification(
        evaluator, re_node,
        "Ethics_Committee_Review_Steps",
        "Describe the typical steps in a research ethics committee review process",
        data.ethics_committee_review_steps,
        add_ins="Verify the described steps match standard U.S. IRB/ethics committee processes per cited sources."
    )


async def build_collaboration_nodes(evaluator: Evaluator, parent, data: CAREERPrepExtraction) -> None:
    collab_node = evaluator.add_parallel(
        id="Collaboration_Framework",
        desc="Collaboration agreement requirements documentation",
        parent=parent,
        critical=True,
    )

    mou_node = evaluator.add_parallel(
        id="MOU_Requirements",
        desc="Memorandum of Understanding (MOU) requirements documentation",
        parent=collab_node,
        critical=True,
    )
    await add_section_verification(
        evaluator, mou_node,
        "MOU_Essential_Elements",
        "List the essential elements that must be included in research MOUs (scope of work, roles/responsibilities, IP terms, timeline)",
        data.mou_essential_elements,
        add_ins="Check that essential elements (scope, roles, IP, schedule/term) are correctly described and sourced."
    )

    pub_rights_node = evaluator.add_parallel(
        id="Publication_Rights",
        desc="Publication rights clauses documentation",
        parent=collab_node,
        critical=True,
    )
    await add_section_verification(
        evaluator, pub_rights_node,
        "Standard_Publication_Rights_Clauses",
        "Identify standard publication rights clauses typically included in research collaboration agreements",
        data.publication_rights_clauses,
        add_ins="Verify that the listed publication clauses (e.g., review periods, authorship, embargo/confidentiality carve-outs) align with authoritative guidance."
    )

    ds_node = evaluator.add_parallel(
        id="Data_Sharing",
        desc="Collaborative data sharing practices documentation",
        parent=collab_node,
        critical=True,
    )
    await add_section_verification(
        evaluator, ds_node,
        "Standard_Data_Sharing_Practices",
        "Document standard data sharing practices and requirements in collaborative research arrangements",
        data.data_sharing_practices,
        add_ins="Check for accurate description of sharing agreements, access controls, repositories, and compliance with funder/institutional policies."
    )


async def build_global_requirements_nodes(evaluator: Evaluator, parent, data: CAREERPrepExtraction) -> None:
    global_node = evaluator.add_parallel(
        id="Global_Source_and_Scope_Requirements",
        desc="Global requirements about citations, authority, and scope/time applicability",
        parent=parent,
        critical=True,
    )

    # 1) All substantive factual claims have at least one URL reference
    all_have_urls = all(section_has_sources(data, attr) for attr in SUBSTANTIVE_SECTION_ATTRS)
    evaluator.add_custom_node(
        result=all_have_urls,
        id="URL_Citations_For_All_Substantive_Claims",
        desc="All substantive factual claims are supported by at least one URL reference",
        parent=global_node,
        critical=True,
    )

    # 2) Authoritative sources only (heuristic domain/TLD/whitelist check)
    all_urls = collect_all_sources(data)
    evaluator.add_custom_node(
        result=all_authoritative(all_urls),
        id="Authoritative_Sources_Only",
        desc="All cited sources are authoritative (government, academic, professional organizations, established academic resources)",
        parent=global_node,
        critical=True,
    )

    # 3) U.S.-based applicability explicitly addressed (answer-level statement)
    await add_scope_statement_verification(
        evaluator,
        global_node,
        "US_Based_Applicability_Addressed",
        "The document explicitly indicates applicability to U.S.-based academic research",
        data.us_based_applicability_statement,
        check_kind="US_scope",
    )

    # 4) Current as of March 2026 explicitly addressed (answer-level statement)
    await add_scope_statement_verification(
        evaluator,
        global_node,
        "AsOf_March_2026_Addressed",
        "The document explicitly indicates currency as of March 2026",
        data.asof_march_2026_statement,
        check_kind="As_of_date",
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Entry point: evaluate an answer for the NSF CAREER Preparation Plan task.
    """
    # Initialize evaluator (root is a container; we will add the true critical root as a child)
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

    # True task root (critical as per rubric)
    plan_root = evaluator.add_parallel(
        id="NSF_CAREER_Grant_Preparation_Plan",
        desc="Complete documentation of NSF CAREER grant preparation requirements across publication strategy, grant components, compliance, and collaboration",
        parent=root,
        critical=True,
    )

    # Extract structured info from the answer
    extracted: CAREERPrepExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=CAREERPrepExtraction,
        extraction_name="career_prep_extraction",
    )

    # Build verification subtrees according to rubric
    await build_publication_strategy_nodes(evaluator, plan_root, extracted)
    await build_grant_components_nodes(evaluator, plan_root, extracted)
    await build_compliance_nodes(evaluator, plan_root, extracted)
    await build_collaboration_nodes(evaluator, plan_root, extracted)
    await build_global_requirements_nodes(evaluator, plan_root, extracted)

    # Add custom evaluation info (diagnostics)
    all_urls = collect_all_sources(extracted)
    unique_domains = sorted({_domain_from_url(u) for u in all_urls if _domain_from_url(u)})
    auth_flags = {d: any(is_authoritative_url(u) and _domain_from_url(u) == d for u in all_urls) for d in unique_domains}

    evaluator.add_custom_info(
        info={
            "total_urls_collected": len(all_urls),
            "unique_domains_count": len(unique_domains),
            "unique_domains": unique_domains,
            "authoritative_domain_flags": auth_flags,
            "sections_with_urls": {attr: section_has_sources(extracted, attr) for attr in SUBSTANTIVE_SECTION_ATTRS},
        },
        info_type="diagnostics",
        info_name="url_and_section_diagnostics",
    )

    # Return standard summary
    return evaluator.get_summary()