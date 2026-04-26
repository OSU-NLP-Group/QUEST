import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "quantum_centers_2020_2021"
TASK_DESCRIPTION = (
    "Identify 3 quantum research centers or institutes in the United States that were established or launched in 2020 or 2021 and are funded through either the National Science Foundation's Quantum Leap Challenge Institutes (QLCI) program or the Department of Energy's National Quantum Information Science Research Centers (NQISRC) program. "
    "For each of the 3 centers, provide: "
    "1) Basic Identification (official center name, lead institution, year established/launched, and URL references); "
    "2) Funding Details (funding agency/program NSF QLCI or DOE NQISRC, award identifier, funding amount if documented, and URL references); "
    "3) Partnership Information (partner institutions list, collaborative structure, and URL references); "
    "4) Research Focus (primary QIS research areas, scientific goals/objectives, and URL references); "
    "5) Educational Component (workforce development/student training/educational components, and URL references if available). "
    "All information must be verifiable through official NSF, DOE, or institutional websites with appropriate URL references for each piece of information."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class CenterBasic(BaseModel):
    name: Optional[str] = None
    lead_institution: Optional[str] = None
    launch_year: Optional[str] = None  # Keep string for flexibility (e.g., "2020", "2021")
    identification_urls: List[str] = Field(default_factory=list)


class CenterFunding(BaseModel):
    program: Optional[str] = None  # Expected values like "NSF QLCI" or "DOE NQISRC"
    award_identifier: Optional[str] = None  # e.g., NSF award number or DOE center designation
    funding_amount: Optional[str] = None  # e.g., "$25M over 5 years" if claimed
    director_name: Optional[str] = None  # required when DOE NQISRC (if applicable)
    funding_urls: List[str] = Field(default_factory=list)


class CenterPartnership(BaseModel):
    partner_institutions: List[str] = Field(default_factory=list)
    partnership_structure: Optional[str] = None
    partnership_urls: List[str] = Field(default_factory=list)


class CenterResearch(BaseModel):
    primary_research_areas: List[str] = Field(default_factory=list)
    scientific_objectives: Optional[str] = None
    research_urls: List[str] = Field(default_factory=list)


class CenterEducation(BaseModel):
    workforce_programs: Optional[str] = None
    educational_urls: List[str] = Field(default_factory=list)


class CenterItem(BaseModel):
    basic: Optional[CenterBasic] = None
    funding: Optional[CenterFunding] = None
    partnership: Optional[CenterPartnership] = None
    research: Optional[CenterResearch] = None
    education: Optional[CenterEducation] = None


class CentersExtraction(BaseModel):
    centers: List[CenterItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_centers() -> str:
    return """
Extract up to 5 quantum research centers or institutes mentioned in the answer. For each center, extract a complete structured record with the following fields. Only extract what is explicitly present in the answer; do not invent.

Return a JSON object:
{
  "centers": [
    {
      "basic": {
        "name": string or null,
        "lead_institution": string or null,
        "launch_year": string or null,  // Use a 4-digit year if mentioned, like "2020" or "2021"
        "identification_urls": [array of URL strings explicitly present in the answer for center name/lead/year]
      },
      "funding": {
        "program": string or null,  // Use exactly "NSF QLCI" when the NSF Quantum Leap Challenge Institutes program is referenced; use exactly "DOE NQISRC" when the DOE National Quantum Information Science Research Centers program is referenced. If unclear, return null.
        "award_identifier": string or null,  // NSF award number (e.g., "NSF Award 2016245") or DOE center designation if given
        "funding_amount": string or null,  // Include only if explicitly claimed in the answer (e.g., "$25M over 5 years")
        "director_name": string or null,   // If DOE NQISRC center and director is named in the answer; otherwise null
        "funding_urls": [array of URL strings explicitly present in the answer for program/award/funding amount]
      },
      "partnership": {
        "partner_institutions": [array of institution names parsed from the answer; leave empty if not mentioned],
        "partnership_structure": string or null,  // brief description of collaborative structure if present
        "partnership_urls": [array of URL strings explicitly present in the answer for partners/structure]
      },
      "research": {
        "primary_research_areas": [array of areas such as 'quantum computing', 'quantum sensing', 'quantum simulation', 'quantum networking', 'quantum communication', 'quantum materials', 'quantum algorithms'; keep textual variants from the answer],
        "scientific_objectives": string or null,
        "research_urls": [array of URL strings explicitly present in the answer for research areas/objectives]
      },
      "education": {
        "workforce_programs": string or null,  // description if present; otherwise null
        "educational_urls": [array of URL strings explicitly present for education/workforce components]
      }
    },
    ...
  ]
}

Rules:
- Extract only URLs that are explicitly present in the answer; include full http/https protocol.
- Do not deduplicate or merge centers; keep them distinct as they appear.
- If a field is missing in the answer, set it to null or an empty list accordingly.
- Keep strings as they appear; do not normalize names or amounts.
- For the "program" field, use only "NSF QLCI" or "DOE NQISRC" when those are explicitly referenced; otherwise null.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def safe_list(xs: Optional[List[str]]) -> List[str]:
    return [x for x in (xs or []) if is_nonempty(x)]


def dedup_preserve_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def collect_all_urls_from_center(center: CenterItem) -> List[str]:
    urls: List[str] = []
    if center.basic:
        urls += safe_list(center.basic.identification_urls)
    if center.funding:
        urls += safe_list(center.funding.funding_urls)
    if center.partnership:
        urls += safe_list(center.partnership.partnership_urls)
    if center.research:
        urls += safe_list(center.research.research_urls)
    if center.education:
        urls += safe_list(center.education.educational_urls)
    return dedup_preserve_order(urls)


def collect_all_urls(centers: List[CenterItem]) -> List[str]:
    all_urls: List[str] = []
    for c in centers:
        all_urls += collect_all_urls_from_center(c)
    return dedup_preserve_order(all_urls)


def normalize_program_label(program: Optional[str]) -> Optional[str]:
    if not program:
        return None
    p = program.strip().lower()
    if "qlci" in p or "quantum leap challenge" in p:
        return "NSF QLCI"
    if "nqisrc" in p or "national quantum information science research center" in p:
        return "DOE NQISRC"
    # If user wrote NSF QLCI or DOE NQISRC already
    if p == "nsf qlci":
        return "NSF QLCI"
    if p == "doe nqisrc":
        return "DOE NQISRC"
    return program  # leave as-is if something else


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_overall_requirements(evaluator: Evaluator, root_node, centers: List[CenterItem]) -> None:
    overall_node = evaluator.add_parallel(
        id="overall_response_requirements",
        desc="Overall response-level requirements",
        parent=root_node,
        critical=True,
    )

    # Exactly three distinct centers (no duplicates)
    names = [c.basic.name.strip() for c in centers if c.basic and is_nonempty(c.basic.name)]
    exactly_three = len(centers) == 3 and len(names) == 3 and len({n.lower() for n in names}) == 3
    evaluator.add_custom_node(
        result=exactly_three,
        id="exactly_three_distinct_centers",
        desc="Response provides exactly 3 distinct centers (no duplicates; no more/no fewer).",
        parent=overall_node,
        critical=True,
    )

    # All URLs official sources node (parent), with child leaves per URL
    urls_node = evaluator.add_parallel(
        id="all_urls_official_sources",
        desc="All provided references are verifiable via official NSF, DOE, or institutional websites (not unofficial/secondary sources).",
        parent=overall_node,
        critical=True,
    )

    urls = collect_all_urls(centers)
    # Add a minimal existence check to avoid empty passing
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="official_urls_exist_any",
        desc="At least one reference URL is provided across the response.",
        parent=urls_node,
        critical=True,
    )

    # For each URL, verify it is an official source (.nsf.gov, .energy.gov or DOE lab .gov domains, or .edu institutional domains)
    claims_and_sources = []
    for idx, url in enumerate(urls):
        leaf = evaluator.add_leaf(
            id=f"official_url_{idx+1}",
            desc=f"URL is an official source: {url}",
            parent=urls_node,
            critical=True,
        )
        claim = (
            "This webpage is an official source from either: "
            "1) the National Science Foundation (domain ends with .nsf.gov), or "
            "2) the U.S. Department of Energy (domain includes energy.gov/science.osti.gov or a DOE national laboratory .gov domain such as anl.gov, ornl.gov, pnnl.gov, bnl.gov, llnl.gov, lanl.gov, fnal.gov, slac.stanford.edu, lbl.gov), or "
            "3) a U.S. university or research institute website with a .edu domain. "
            "Do not consider news aggregators, Wikipedia, press reprint sites, or generic .com pages as official."
        )
        claims_and_sources.append((
            claim,
            url,
            leaf,
            "Judge primarily from the URL domain and the page's self-identity header/footer. Pass only if the URL qualifies as an official page as described."
        ))
    if claims_and_sources:
        await evaluator.batch_verify(claims_and_sources)


async def verify_center(
    evaluator: Evaluator,
    parent_node,
    center: CenterItem,
    index: int,
) -> None:
    """
    Build the verification subtree for one center.
    """
    idx = index + 1
    center_node = evaluator.add_parallel(
        id=f"center_{idx}",
        desc=f"Center #{idx}: quantum research center meeting all specified criteria",
        parent=parent_node,
        critical=False,  # allow partial credit per center
    )

    basic = center.basic or CenterBasic()
    funding = center.funding or CenterFunding()
    partnership = center.partnership or CenterPartnership()
    research = center.research or CenterResearch()
    education = center.education or CenterEducation()

    # ------------------ Basic Identification ------------------ #
    basic_node = evaluator.add_parallel(
        id=f"center_{idx}_basic_identification",
        desc=f"Basic identification information for center #{idx}",
        parent=center_node,
        critical=True,
    )

    # Name provided (existence)
    evaluator.add_custom_node(
        result=is_nonempty(basic.name),
        id=f"center_{idx}_name",
        desc="Official name of the quantum research center or institute is provided.",
        parent=basic_node,
        critical=True,
    )

    # Lead institution provided (existence)
    evaluator.add_custom_node(
        result=is_nonempty(basic.lead_institution),
        id=f"center_{idx}_lead_institution",
        desc="Lead institution directing the center is provided.",
        parent=basic_node,
        critical=True,
    )

    # US-based check (verify via identification URLs)
    us_based_leaf = evaluator.add_leaf(
        id=f"center_{idx}_us_based",
        desc="Center is based in the United States (e.g., lead institution/host is US-based).",
        parent=basic_node,
        critical=True,
    )
    us_claim = (
        f"The center is US-based; the lead institution '{basic.lead_institution or ''}' is a United States institution."
    )
    await evaluator.verify(
        claim=us_claim,
        node=us_based_leaf,
        sources=basic.identification_urls,
        additional_instruction=(
            "Use the identification URL(s) to confirm the US affiliation. "
            "Typical signals: .edu domain in the US, .gov DOE lab pages, or explicit mentions of locations in the United States. "
            "If the evidence is not explicit or not from an official page, mark as not supported."
        ),
    )

    # Launch year provided and is either 2020 or 2021 (and verify via URL)
    year_leaf = evaluator.add_leaf(
        id=f"center_{idx}_launch_year_2020_or_2021",
        desc="Year established/launched is provided and is either 2020 or 2021.",
        parent=basic_node,
        critical=True,
    )
    year_val = (basic.launch_year or "").strip()
    year_claim = (
        f"The center was established or launched in {year_val}, and this year is either 2020 or 2021."
    )
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=basic.identification_urls,
        additional_instruction=(
            "Pass only if the evidence page(s) clearly support the specific year and the value is 2020 or 2021."
        ),
    )

    # Identification URLs provided (existence)
    evaluator.add_custom_node(
        result=len(safe_list(basic.identification_urls)) > 0,
        id=f"center_{idx}_identification_urls",
        desc="URL reference(s) provided that verify the center name, lead institution, and launch year.",
        parent=basic_node,
        critical=True,
    )

    # ------------------ Funding Details ------------------ #
    funding_node = evaluator.add_parallel(
        id=f"center_{idx}_funding_details",
        desc=f"Funding information for center #{idx}",
        parent=center_node,
        critical=True,
    )

    program_norm = normalize_program_label(funding.program)
    program_leaf = evaluator.add_leaf(
        id=f"center_{idx}_funding_agency_program",
        desc="Funding is through either NSF QLCI or DOE NQISRC.",
        parent=funding_node,
        critical=True,
    )
    program_claim = (
        f"The center is funded through the program '{program_norm or (funding.program or '')}', which must be either NSF QLCI or DOE NQISRC."
    )
    await evaluator.verify(
        claim=program_claim,
        node=program_leaf,
        sources=funding.funding_urls or basic.identification_urls,
        additional_instruction=(
            "Pass only if the evidence explicitly states NSF Quantum Leap Challenge Institutes (QLCI) or "
            "DOE National Quantum Information Science Research Centers (NQISRC) funding."
        ),
    )

    # Lead type matches program
    lead_type_leaf = evaluator.add_leaf(
        id=f"center_{idx}_lead_type_matches_program",
        desc="Lead institution type matches the funding program: NSF QLCI → led by a research university; DOE NQISRC → led by a DOE national laboratory.",
        parent=funding_node,
        critical=True,
    )
    if (program_norm or "").upper() == "DOE NQISRC":
        lead_type_claim = (
            f"The lead institution '{basic.lead_institution or ''}' is a U.S. Department of Energy national laboratory."
        )
        lead_type_add_ins = (
            "Check that the lead institution is a DOE national laboratory (e.g., Argonne National Laboratory, "
            "Oak Ridge National Laboratory, Brookhaven National Laboratory, Pacific Northwest National Laboratory, "
            "Fermi National Accelerator Laboratory, Lawrence Berkeley National Laboratory, Los Alamos National Laboratory, "
            "Lawrence Livermore National Laboratory, SLAC National Accelerator Laboratory, etc.). "
            "Use the provided official sources to confirm."
        )
    else:
        lead_type_claim = (
            f"The lead institution '{basic.lead_institution or ''}' is a research university (not a DOE national laboratory)."
        )
        lead_type_add_ins = (
            "Check that the lead is a university (typically .edu domain and described as a university) "
            "if the program is NSF QLCI. Use the provided official sources to confirm."
        )
    await evaluator.verify(
        claim=lead_type_claim,
        node=lead_type_leaf,
        sources=funding.funding_urls or basic.identification_urls,
        additional_instruction=lead_type_add_ins,
    )

    # DOE director named if applicable
    if (program_norm or "").upper() == "DOE NQISRC":
        # If DOE, director must be provided and verifiable
        if is_nonempty(funding.director_name):
            director_leaf = evaluator.add_leaf(
                id=f"center_{idx}_doe_director_named_if_applicable",
                desc="If DOE NQISRC-funded, a named director is provided (N/A for NSF QLCI centers).",
                parent=funding_node,
                critical=True,
            )
            director_claim = f"The DOE NQISRC center lists the director as '{funding.director_name}'."
            await evaluator.verify(
                claim=director_claim,
                node=director_leaf,
                sources=funding.funding_urls or basic.identification_urls,
                additional_instruction="Verify the director's name on the official DOE or center site.",
            )
        else:
            evaluator.add_custom_node(
                result=False,
                id=f"center_{idx}_doe_director_named_if_applicable",
                desc="If DOE NQISRC-funded, a named director is provided (N/A for NSF QLCI centers).",
                parent=funding_node,
                critical=True,
            )
    else:
        # NSF centers: Not applicable; mark as passed
        evaluator.add_custom_node(
            result=True,
            id=f"center_{idx}_doe_director_named_if_applicable",
            desc="If DOE NQISRC-funded, a named director is provided (N/A for NSF QLCI centers).",
            parent=funding_node,
            critical=True,
        )

    # Award identifier (verify)
    award_leaf = evaluator.add_leaf(
        id=f"center_{idx}_award_identifier",
        desc="Award identifier is provided: NSF award number for QLCI OR official center designation for DOE centers.",
        parent=funding_node,
        critical=True,
    )
    award_claim = (
        f"The center's award identifier is '{funding.award_identifier or ''}', which must be an NSF award number (for QLCI) "
        f"or an official DOE center designation (for NQISRC)."
    )
    await evaluator.verify(
        claim=award_claim,
        node=award_leaf,
        sources=funding.funding_urls or basic.identification_urls,
        additional_instruction="Pass only if the identifier is explicitly present in the official source.",
    )

    # Funding amount (non-critical) – if provided, verify; if not claimed, treat as N/A (pass)
    if is_nonempty(funding.funding_amount):
        amt_leaf = evaluator.add_leaf(
            id=f"center_{idx}_funding_amount",
            desc="Funding amount is provided when documented.",
            parent=funding_node,
            critical=False,
        )
        amt_claim = f"The funding amount associated with this center is '{funding.funding_amount}'."
        await evaluator.verify(
            claim=amt_claim,
            node=amt_leaf,
            sources=funding.funding_urls or basic.identification_urls,
            additional_instruction="Verify the exact funding amount or lifecycle total as stated on the official source.",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"center_{idx}_funding_amount",
            desc="Funding amount is provided when documented (N/A if not claimed).",
            parent=funding_node,
            critical=False,
        )

    # Funding URLs provided (existence)
    evaluator.add_custom_node(
        result=len(safe_list(funding.funding_urls)) > 0,
        id=f"center_{idx}_funding_urls",
        desc="URL reference(s) provided that verify funding agency/program and award identifier (and funding amount if claimed).",
        parent=funding_node,
        critical=True,
    )

    # ------------------ Partnership Information ------------------ #
    partner_node = evaluator.add_parallel(
        id=f"center_{idx}_partnership_information",
        desc=f"Partnership and collaboration details for center #{idx}",
        parent=center_node,
        critical=True,
    )

    partners_leaf = evaluator.add_leaf(
        id=f"center_{idx}_partner_institutions_multiple",
        desc="Provides a list of partner institutions demonstrating multi-institution collaboration (at least 2 partners, or otherwise explicitly multi-institution).",
        parent=partner_node,
        critical=True,
    )
    partners_claim = (
        f"The center has multi-institution collaboration with at least two partner institutions: {partnership.partner_institutions}."
    )
    await evaluator.verify(
        claim=partners_claim,
        node=partners_leaf,
        sources=partnership.partnership_urls or basic.identification_urls,
        additional_instruction=(
            "Pass only if the official source indicates at least two distinct partner institutions or explicitly states a multi-institution collaboration."
        ),
    )

    structure_leaf = evaluator.add_leaf(
        id=f"center_{idx}_partnership_structure",
        desc="Describes the multi-institution collaborative structure (how the partnership is organized).",
        parent=partner_node,
        critical=True,
    )
    structure_claim = (
        f"The following describes the multi-institution collaborative structure: {partnership.partnership_structure or ''}."
    )
    await evaluator.verify(
        claim=structure_claim,
        node=structure_leaf,
        sources=partnership.partnership_urls or basic.identification_urls,
        additional_instruction="Verify that the described collaboration structure is supported by the source page(s).",
    )

    evaluator.add_custom_node(
        result=len(safe_list(partnership.partnership_urls)) > 0,
        id=f"center_{idx}_partnership_urls",
        desc="URL reference(s) provided that verify partners and/or collaboration structure.",
        parent=partner_node,
        critical=True,
    )

    # ------------------ Research Focus ------------------ #
    research_node = evaluator.add_parallel(
        id=f"center_{idx}_research_focus",
        desc=f"Research areas and scientific objectives of center #{idx}",
        parent=center_node,
        critical=True,
    )

    areas_leaf = evaluator.add_leaf(
        id=f"center_{idx}_primary_research_areas_qis",
        desc="Primary research areas are stated and are quantum information science-related.",
        parent=research_node,
        critical=True,
    )
    areas_claim = (
        f"The center's primary research areas include {research.primary_research_areas}, and these are quantum information science-related."
    )
    await evaluator.verify(
        claim=areas_claim,
        node=areas_leaf,
        sources=research.research_urls or basic.identification_urls,
        additional_instruction=(
            "Accept QIS-related areas such as quantum computing, quantum sensing, quantum simulation, quantum networking, "
            "quantum communication, quantum materials, quantum algorithms, etc. Verify the areas are stated on the source page(s)."
        ),
    )

    objectives_leaf = evaluator.add_leaf(
        id=f"center_{idx}_scientific_objectives",
        desc="Specific scientific goals or technological objectives are described.",
        parent=research_node,
        critical=True,
    )
    objectives_claim = f"The center's scientific goals/objectives include: {research.scientific_objectives or ''}."
    await evaluator.verify(
        claim=objectives_claim,
        node=objectives_leaf,
        sources=research.research_urls or basic.identification_urls,
        additional_instruction="Verify the stated objectives on the official source page(s).",
    )

    evaluator.add_custom_node(
        result=len(safe_list(research.research_urls)) > 0,
        id=f"center_{idx}_research_urls",
        desc="URL reference(s) provided that verify the research focus and objectives.",
        parent=research_node,
        critical=True,
    )

    # ------------------ Educational Component ------------------ #
    edu_node = evaluator.add_parallel(
        id=f"center_{idx}_educational_component",
        desc=f"Educational/workforce development information for center #{idx}",
        parent=center_node,
        critical=False,
    )

    if is_nonempty(education.workforce_programs):
        wf_leaf = evaluator.add_leaf(
            id=f"center_{idx}_workforce_programs",
            desc="Describes workforce development, student training, or educational components (if available).",
            parent=edu_node,
            critical=False,
        )
        wf_claim = f"The center has workforce development/education components, such as: {education.workforce_programs}."
        await evaluator.verify(
            claim=wf_claim,
            node=wf_leaf,
            sources=education.educational_urls or research.research_urls or basic.identification_urls,
            additional_instruction="Verify existence of described workforce/education programs on the official source page(s).",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"center_{idx}_workforce_programs",
            desc="Describes workforce development, student training, or educational components (if available) – N/A if not claimed.",
            parent=edu_node,
            critical=False,
        )

    # If education was claimed, ensure URLs provided; else treat as N/A
    if is_nonempty(education.workforce_programs):
        evaluator.add_custom_node(
            result=len(safe_list(education.educational_urls)) > 0,
            id=f"center_{idx}_educational_urls",
            desc="URL reference(s) provided verifying educational/workforce components (if available/claimed).",
            parent=edu_node,
            critical=False,
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"center_{idx}_educational_urls",
            desc="URL reference(s) verifying educational/workforce components – N/A if not claimed.",
            parent=edu_node,
            critical=False,
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
    Evaluate a single answer for the quantum centers (2020/2021; NSF QLCI or DOE NQISRC) task.
    """
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

    # Extract centers
    extracted: CentersExtraction = await evaluator.extract(
        prompt=prompt_extract_centers(),
        template_class=CentersExtraction,
        extraction_name="centers_extraction",
    )

    centers_all = extracted.centers or []
    # Per general guidance, keep the first 3 items (filter/pad)
    centers_used: List[CenterItem] = centers_all[:3]
    while len(centers_used) < 3:
        centers_used.append(CenterItem())

    # Overall requirements (critical)
    await verify_overall_requirements(evaluator, root, centers_used)

    # Verify each of the 3 centers (non-critical per-center, allowing partial scoring)
    for i in range(3):
        await verify_center(evaluator, root, centers_used[i], i)

    return evaluator.get_summary()