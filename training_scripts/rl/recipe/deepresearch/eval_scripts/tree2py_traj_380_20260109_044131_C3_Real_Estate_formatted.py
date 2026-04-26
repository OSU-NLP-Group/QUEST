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
TASK_ID = "architect_2018_sustainability_brazil_partner_march_year"
TASK_DESCRIPTION = (
    "An architectural firm was ranked #1 for sustainability by Architect Magazine in 2018. "
    "This firm was founded in 1977 by two partners who both served in the Peace Corps after graduation "
    "and met as classmates at Washington State University, where they both earned Bachelor of Architecture degrees in 1968. "
    "One of these founding partners served in the Peace Corps in Brazil, while the other served in Afghanistan. "
    "After completing their Peace Corps service, the partner who served in Brazil went on to pursue graduate education and earned "
    "a Master of Architecture degree from a university described as having one of the oldest and largest architecture schools in the United States. "
    "In what year did this partner earn their Master of Architecture degree?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FounderExtraction(BaseModel):
    name: Optional[str] = None
    peace_corps_country: Optional[str] = None  # e.g., "Brazil" or "Afghanistan"
    undergrad_degree: Optional[str] = None     # e.g., "Bachelor of Architecture"
    undergrad_school: Optional[str] = None     # e.g., "Washington State University"
    undergrad_year: Optional[str] = None       # e.g., "1968"
    masters_degree: Optional[str] = None       # e.g., "Master of Architecture" or "M.Arch"
    masters_school: Optional[str] = None       # e.g., "University of Illinois at Urbana-Champaign"
    masters_year: Optional[str] = None         # e.g., "1978"
    founder_sources: List[str] = Field(default_factory=list)                 # general bio/founder page(s)
    peace_corps_sources: List[str] = Field(default_factory=list)             # URLs supporting Peace Corps details
    masters_sources: List[str] = Field(default_factory=list)                 # URLs supporting M.Arch details
    masters_school_profile_sources: List[str] = Field(default_factory=list)  # URLs describing the school as "one of the oldest and largest..."


class FirmExtraction(BaseModel):
    firm_name: Optional[str] = None
    founded_year: Optional[str] = None
    founders: List[FounderExtraction] = Field(default_factory=list)

    # Evidence URLs for firm-level constraints
    ranking_2018_sustainability_sources: List[str] = Field(default_factory=list)  # Architect Magazine (ARCHITECT 50) sustainability #1 in 2018
    founding_sources: List[str] = Field(default_factory=list)                     # founding year, founders info
    wsu_classmates_sources: List[str] = Field(default_factory=list)               # met as classmates at WSU
    barch_wsu_1968_sources: List[str] = Field(default_factory=list)               # both earned B.Arch at WSU in 1968


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_firm_and_founders() -> str:
    return (
        "From the provided answer, extract the architectural firm and founder details required to determine the M.Arch year for the "
        "Brazil-serving founding partner. Only extract URLs that are explicitly present in the answer text. If any field is missing, return null "
        "or an empty list as appropriate.\n\n"
        "Required JSON fields and meanings:\n"
        "- firm_name: The name of the architectural firm that is claimed to match all constraints.\n"
        "- founded_year: The founding year of the firm (e.g., '1977').\n"
        "- founders: An array of up to two founder objects (if more mentioned, include at least the two primary founders). For each founder:\n"
        "  - name: Full name of the founder, if provided.\n"
        "  - peace_corps_country: The country where this founder served in the Peace Corps (e.g., 'Brazil' or 'Afghanistan').\n"
        "  - undergrad_degree: The undergraduate architecture degree (e.g., 'Bachelor of Architecture' or 'B.Arch').\n"
        "  - undergrad_school: The undergraduate school (e.g., 'Washington State University').\n"
        "  - undergrad_year: The year of the undergraduate degree (e.g., '1968').\n"
        "  - masters_degree: The graduate architecture degree (e.g., 'Master of Architecture' or 'M.Arch').\n"
        "  - masters_school: The university where the founder earned the M.Arch.\n"
        "  - masters_year: The year the M.Arch was earned.\n"
        "  - founder_sources: URLs cited for this founder’s general bio/identity and founding role.\n"
        "  - peace_corps_sources: URLs cited that support the Peace Corps service country for this founder.\n"
        "  - masters_sources: URLs cited that support this founder’s M.Arch details (school and year).\n"
        "  - masters_school_profile_sources: URLs (e.g., official school pages) that describe the school as having one of the oldest and largest architecture schools in the United States.\n"
        "- ranking_2018_sustainability_sources: URLs cited that support the claim the firm was ranked #1 for sustainability by Architect Magazine in 2018 (e.g., ARCHITECT 50 Sustainability category).\n"
        "- founding_sources: URLs cited that support the founding year and who the founders are.\n"
        "- wsu_classmates_sources: URLs cited that support that the two founders met as classmates at Washington State University.\n"
        "- barch_wsu_1968_sources: URLs cited that support that both founders earned Bachelor of Architecture degrees at Washington State University in 1968.\n\n"
        "Special URL rules:\n"
        "- Extract only URLs explicitly present in the answer (including markdown links). Do not invent or infer URLs.\n"
        "- If a URL is missing a protocol, prepend 'http://'.\n"
        "- If a field is unknown or not provided in the answer, return null or an empty list for that field."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: List[str]) -> List[str]:
    """Combine multiple lists of URLs, remove empties and duplicates, preserve order."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in lst or []:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                combined.append(url)
    return combined


def _find_founder_by_country(founders: List[FounderExtraction], country_substr: str) -> Optional[FounderExtraction]:
    """Find the first founder whose peace_corps_country contains the given substring (case-insensitive)."""
    target = (country_substr or "").lower().strip()
    for f in founders or []:
        if f.peace_corps_country and target in f.peace_corps_country.lower():
            return f
    return None


def _safe(val: Optional[str], fallback: str) -> str:
    """Return val if non-empty, else fallback."""
    if val and str(val).strip():
        return str(val).strip()
    return fallback


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_identify_firm_nodes(evaluator: Evaluator, parent_node, info: FirmExtraction) -> None:
    """
    Build and verify the 'identify_firm' parallel node:
    - ranked_1_sustainability_2018
    - founded_1977
    - two_partners_peace_corps
    - met_as_classmates_wsu
    - barch_wsu_1968
    All leaves are critical, as per rubric.
    """
    firm_node = evaluator.add_parallel(
        id="identify_firm",
        desc="Identify the architectural firm matching all given constraints (Architect Magazine sustainability ranking, founding year, and founder background).",
        parent=parent_node,
        critical=True
    )

    firm_name = _safe(info.firm_name, "the firm")

    # 1) Ranked #1 in Sustainability by Architect Magazine in 2018
    n_rank = evaluator.add_leaf(
        id="ranked_1_sustainability_2018",
        desc="Firm was ranked #1 for sustainability by Architect Magazine in 2018.",
        parent=firm_node,
        critical=True
    )
    claim_rank = (
        f"According to the cited source(s), {firm_name} was ranked #1 for sustainability by Architect Magazine in 2018 "
        f"(e.g., ARCHITECT 50 Sustainability category)."
    )
    await evaluator.verify(
        claim=claim_rank,
        node=n_rank,
        sources=info.ranking_2018_sustainability_sources,
        additional_instruction="Confirm that the source explicitly shows a 2018 Architect Magazine sustainability ranking with the firm at #1. Accept 'ARCHITECT 50' Sustainability category phrasing."
    )

    # 2) Founded in 1977
    n_found = evaluator.add_leaf(
        id="founded_1977",
        desc="Firm was founded in 1977.",
        parent=firm_node,
        critical=True
    )
    claim_found = f"The cited source(s) indicate that {firm_name} was founded in 1977."
    await evaluator.verify(
        claim=claim_found,
        node=n_found,
        sources=info.founding_sources,
        additional_instruction="Prefer authoritative firm, archival, or reputable sources confirming the founding year 1977."
    )

    # 3) Two partners who both served in the Peace Corps after graduation
    n_peace = evaluator.add_leaf(
        id="two_partners_peace_corps",
        desc="Firm was founded by two partners who both served in the Peace Corps.",
        parent=firm_node,
        critical=True
    )
    all_peace_sources = _combine_sources(
        info.founding_sources,
        *(f.peace_corps_sources for f in info.founders or [])
    )
    claim_peace = (
        "The cited source(s) indicate the firm was founded by two partners and both founders served in the Peace Corps after graduation."
    )
    await evaluator.verify(
        claim=claim_peace,
        node=n_peace,
        sources=all_peace_sources,
        additional_instruction="Look for explicit mentions that both (two) founding partners served in the Peace Corps; timeline 'after graduation' can be inferred from dates if clearly indicated."
    )

    # 4) Met as classmates at Washington State University
    n_wsu_class = evaluator.add_leaf(
        id="met_as_classmates_wsu",
        desc="The two founding partners met as classmates at Washington State University.",
        parent=firm_node,
        critical=True
    )
    claim_wsu_class = (
        "The cited source(s) indicate the two founding partners met as classmates at Washington State University (WSU)."
    )
    await evaluator.verify(
        claim=claim_wsu_class,
        node=n_wsu_class,
        sources=info.wsu_classmates_sources,
        additional_instruction="Accept reasonable wording variants such as 'met while classmates' or 'studied together at WSU'."
    )

    # 5) Both B.Arch from WSU in 1968
    n_barch = evaluator.add_leaf(
        id="barch_wsu_1968",
        desc="Both founding partners earned Bachelor of Architecture degrees from Washington State University in 1968.",
        parent=firm_node,
        critical=True
    )
    # Combine any founder bios and explicit B.Arch/WSU evidence
    all_barch_sources = _combine_sources(
        info.barch_wsu_1968_sources,
        *(f.founder_sources for f in info.founders or [])
    )
    claim_barch = (
        "The cited source(s) indicate both founding partners earned Bachelor of Architecture (B.Arch) degrees from Washington State University in 1968."
    )
    await evaluator.verify(
        claim=claim_barch,
        node=n_barch,
        sources=all_barch_sources,
        additional_instruction="Verify both individuals share B.Arch degrees from WSU with the class year 1968."
    )


async def build_identify_brazil_partner_nodes(evaluator: Evaluator, parent_node, info: FirmExtraction) -> Dict[str, Optional[str]]:
    """
    Build and verify the 'identify_brazil_partner' parallel node:
    - brazil_vs_afghanistan_service
    Returns a dict with resolved names for 'brazil_name' and 'afghanistan_name' (may be None).
    """
    id_node = evaluator.add_parallel(
        id="identify_brazil_partner",
        desc="Among the two founding partners, identify the partner who served in the Peace Corps in Brazil (as opposed to the partner who served in Afghanistan).",
        parent=parent_node,
        critical=True
    )

    brazil_f = _find_founder_by_country(info.founders, "brazil")
    afg_f = _find_founder_by_country(info.founders, "afghanistan")

    brazil_name = brazil_f.name if brazil_f else None
    afg_name = afg_f.name if afg_f else None

    n_brazil_afg = evaluator.add_leaf(
        id="brazil_vs_afghanistan_service",
        desc="Correctly distinguishes the Brazil-serving founder from the Afghanistan-serving founder.",
        parent=id_node,
        critical=True
    )

    # Build sources for Peace Corps service mapping
    mapping_sources = _combine_sources(
        *(f.peace_corps_sources for f in info.founders or []),
        *(f.founder_sources for f in info.founders or [])
    )

    firm_name = _safe(info.firm_name, "the firm")
    if brazil_name and afg_name:
        claim_mapping = (
            f"Among the founding partners of {firm_name}, {brazil_name} served in the Peace Corps in Brazil and {afg_name} served in Afghanistan."
        )
    else:
        # Fallback generic claim if names were not extracted, but countries are still implied
        claim_mapping = (
            "Among the two founding partners, one served in the Peace Corps in Brazil and the other served in Afghanistan; "
            "the cited source(s) clearly distinguish which founder served where."
        )

    await evaluator.verify(
        claim=claim_mapping,
        node=n_brazil_afg,
        sources=mapping_sources,
        additional_instruction="Confirm the correct mapping of founder names to Peace Corps countries (Brazil vs. Afghanistan). Allow minor name variants or initials."
    )

    return {
        "brazil_name": brazil_name,
        "afghanistan_name": afg_name
    }


async def build_masters_details_nodes(
    evaluator: Evaluator,
    parent_node,
    info: FirmExtraction,
    brazil_name: Optional[str]
) -> None:
    """
    Build and verify the 'masters_architecture_details' parallel node:
    - pursued_grad_education_after_peace_corps
    - march_from_oldest_largest_arch_school_university
    - provide_march_year
    All are critical.
    """
    m_node = evaluator.add_parallel(
        id="masters_architecture_details",
        desc="Extract the Master of Architecture degree details for the Brazil-serving founder per the constraints.",
        parent=parent_node,
        critical=True
    )

    brazil_f = _find_founder_by_country(info.founders, "brazil")
    partner_label = _safe(brazil_name, "the Brazil-serving founding partner")

    # Safely pull M.Arch info if available
    masters_school = _safe(brazil_f.masters_school if brazil_f else None, "the stated university")
    masters_year = _safe(brazil_f.masters_year if brazil_f else None, "the stated year")
    masters_sources = (brazil_f.masters_sources if brazil_f else []) or []
    school_profile_sources = (brazil_f.masters_school_profile_sources if brazil_f else []) or []
    peace_sources = (brazil_f.peace_corps_sources if brazil_f else []) or []

    # 1) Pursued graduate education after completing Peace Corps
    n_after_pc = evaluator.add_leaf(
        id="pursued_grad_education_after_peace_corps",
        desc="States that the Brazil-serving partner pursued graduate education after completing Peace Corps service.",
        parent=m_node,
        critical=True
    )
    claim_after = (
        f"After completing Peace Corps service, {partner_label} pursued graduate education (i.e., proceeded to graduate school)."
    )
    await evaluator.verify(
        claim=claim_after,
        node=n_after_pc,
        sources=_combine_sources(peace_sources, masters_sources),
        additional_instruction="Check that the timeline indicates Peace Corps service occurred before the graduate study; explicit 'after' phrasing or clearly ordered dates are acceptable."
    )

    # 2) Earned M.Arch from a university described as having one of the oldest and largest architecture schools in the U.S.
    n_oldest_largest = evaluator.add_leaf(
        id="march_from_oldest_largest_arch_school_university",
        desc="Identifies that this partner earned a Master of Architecture degree from a university described as having one of the oldest and largest architecture schools in the United States.",
        parent=m_node,
        critical=True
    )
    claim_oldest = (
        f"{partner_label} earned a Master of Architecture degree from {masters_school}, and that university (or its School of Architecture) "
        f"is described as having one of the oldest and largest architecture schools in the United States."
    )
    await evaluator.verify(
        claim=claim_oldest,
        node=n_oldest_largest,
        sources=_combine_sources(masters_sources, school_profile_sources),
        additional_instruction="Verify both parts: (1) the M.Arch was earned from the named university; (2) an authoritative source describes that university's architecture school as 'one of the oldest and largest' in the United States."
    )

    # 3) Provide the M.Arch year for the Brazil-serving partner
    n_year = evaluator.add_leaf(
        id="provide_march_year",
        desc="Provides the year in which this partner earned the Master of Architecture degree.",
        parent=m_node,
        critical=True
    )
    claim_year = f"The year {partner_label} earned the Master of Architecture degree is {masters_year}."
    await evaluator.verify(
        claim=claim_year,
        node=n_year,
        sources=masters_sources,
        additional_instruction="Look for an explicit year on the cited source(s) indicating when the M.Arch was awarded."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the task:
    Determine the year the Brazil-serving founding partner earned their Master of Architecture degree,
    under the given firm/founder identification constraints.
    """
    # 1) Initialize evaluator and root (sequential, critical)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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
    # Make root critical as per rubric
    root.critical = True

    # 2) Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_firm_and_founders(),
        template_class=FirmExtraction,
        extraction_name="firm_and_founders_extraction"
    )

    # 3) Build verification tree according to rubric (all critical, sequential)
    #    3.1 Identify firm (parallel critical group)
    await build_identify_firm_nodes(evaluator, root, extracted)

    #    3.2 Identify the Brazil-serving partner
    names = await build_identify_brazil_partner_nodes(evaluator, root, extracted)
    brazil_name = names.get("brazil_name")

    #    3.3 M.Arch details for the Brazil-serving partner
    await build_masters_details_nodes(evaluator, root, extracted, brazil_name)

    # Optionally, add a small custom info to help debugging/trace
    try:
        brazil_f = _find_founder_by_country(extracted.founders, "brazil")
        evaluator.add_custom_info(
            info={
                "resolved_firm_name": extracted.firm_name,
                "brazil_founder_name": brazil_f.name if brazil_f else None,
                "brazil_founder_masters_school": brazil_f.masters_school if brazil_f else None,
                "brazil_founder_masters_year": brazil_f.masters_year if brazil_f else None
            },
            info_type="debug",
            info_name="resolved_entities"
        )
    except Exception:
        pass

    # 4) Return evaluation summary
    return evaluator.get_summary()