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
TASK_ID = "3I_ATLAS_first_paper_mnras_letters_2025"
TASK_DESCRIPTION = (
    "Identify the first peer-reviewed scientific paper published about the interstellar comet 3I/ATLAS, "
    "which was discovered on July 1, 2025, by the Asteroid Terrestrial-impact Last Alert System (ATLAS). "
    "The paper must meet the following requirements: (1) Published in 2025 in Monthly Notices of the "
    "Royal Astronomical Society Letters (MNRAS Letters), (2) Report original observational data obtained "
    "from three specific ground-based telescopes: the Kottamia Astronomical Observatory 1.88-m telescope, "
    "the Palomar 200-inch telescope, and the Astrophysical Research Consortium (ARC) 3.5-m telescope, "
    "(3) Confirm that 3I/ATLAS is an interstellar comet with a hyperbolic orbit. Provide the following "
    "information: the paper's full title, the name of the lead (first) author, and a reference URL "
    "(either to the journal article or arXiv preprint)."
)

# Canonical telescope names for instructions
TELESCOPE_REQUIREMENTS = [
    "Kottamia Astronomical Observatory 1.88-m",
    "Palomar 200-inch (Hale Telescope)",
    "ARC 3.5-m (Apache Point Observatory)"
]


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class PaperEntry(BaseModel):
    title: Optional[str] = None
    lead_author: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class PaperCandidates(BaseModel):
    papers: List[PaperEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_candidates() -> str:
    return """
    Extract the paper(s) identified in the answer related to 3I/ATLAS. Preserve the order they appear.
    For each paper, extract:
    1) title: the paper's full title as written in the answer (string or null if missing)
    2) lead_author: the first (lead) author's name as written in the answer (string or null if missing)
    3) reference_urls: an array of all URLs provided in the answer that point to the journal article and/or the arXiv preprint (if any).
       - Only include valid URLs explicitly present in the answer.
       - Include both the journal and arXiv URLs if both are provided.
    Return a JSON object with a single field "papers", which is an array of {title, lead_author, reference_urls}.
    If the answer mentions multiple candidate papers, extract them all in the order they are presented.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def select_primary_paper(extracted: PaperCandidates) -> PaperEntry:
    """
    Select the first candidate paper (as the primary one for verification).
    If no candidates, return an empty PaperEntry placeholder.
    """
    if extracted.papers:
        return extracted.papers[0]
    return PaperEntry()


def is_valid_http_url(u: str) -> bool:
    if not isinstance(u, str):
        return False
    u = u.strip()
    return u.startswith("http://") or u.startswith("https://")


def dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    parent_node,
    paper: PaperEntry,
    logger: logging.Logger
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Create the top-level task node (critical, sequential)
    task_node = evaluator.add_sequential(
        id="Research_Paper_Identification",
        desc="Identify the first peer-reviewed scientific paper about 3I/ATLAS meeting all stated constraints, and provide required output fields.",
        parent=parent_node,
        critical=True
    )

    # Prepare sources (deduplicate; keep only valid http(s) URLs)
    sources = dedup_urls([u for u in paper.reference_urls if is_valid_http_url(u)])

    # Child 1: Paper_Eligibility (critical, parallel)
    eligibility_node = evaluator.add_parallel(
        id="Paper_Eligibility",
        desc="Paper satisfies all eligibility constraints (topic, venue, year, first peer-reviewed, telescope data, interstellar/hyperbolic confirmation, discovery details, and collaboration constraint).",
        parent=task_node,
        critical=True
    )

    # Child 2: Required_Output_Fields (critical, parallel)
    required_fields_node = evaluator.add_parallel(
        id="Required_Output_Fields",
        desc="Answer provides all required fields requested by the question for the identified paper.",
        parent=task_node,
        critical=True
    )

    # Required fields (existence checks)
    title_provided_node = evaluator.add_custom_node(
        result=bool(paper.title and paper.title.strip()),
        id="Paper_Full_Title_Provided",
        desc="Provides the paper's full title.",
        parent=required_fields_node,
        critical=True
    )
    lead_author_provided_node = evaluator.add_custom_node(
        result=bool(paper.lead_author and paper.lead_author.strip()),
        id="Lead_Author_Provided",
        desc="Provides the name of the lead (first) author.",
        parent=required_fields_node,
        critical=True
    )
    reference_url_provided_node = evaluator.add_custom_node(
        result=bool(sources and all(is_valid_http_url(u) for u in sources)),
        id="Reference_URL_Provided",
        desc="Provides a valid reference URL to the journal article or an arXiv preprint.",
        parent=required_fields_node,
        critical=True
    )

    # Now, create all eligibility leaf nodes (all critical under eligibility_node)
    # Subject_Matter
    subject_node = evaluator.add_leaf(
        id="Subject_Matter",
        desc="The paper's primary subject is the interstellar comet 3I/ATLAS.",
        parent=eligibility_node,
        critical=True
    )
    # First_Peer_Reviewed_Publication
    first_peer_node = evaluator.add_leaf(
        id="First_Peer_Reviewed_Publication",
        desc="This is the first peer-reviewed scientific paper published about 3I/ATLAS.",
        parent=eligibility_node,
        critical=True
    )
    # Venue_And_Year
    venue_year_node = evaluator.add_leaf(
        id="Venue_And_Year",
        desc="Published in 2025 in Monthly Notices of the Royal Astronomical Society Letters (MNRAS Letters).",
        parent=eligibility_node,
        critical=True
    )
    # Original_Observational_Data_From_Specified_Telescopes
    telescopes_node = evaluator.add_leaf(
        id="Original_Observational_Data_From_Specified_Telescopes",
        desc="Reports original observational data obtained using all three specified ground-based telescopes: Kottamia Astronomical Observatory 1.88-m, Palomar 200-inch, and ARC 3.5-m.",
        parent=eligibility_node,
        critical=True
    )
    # Confirms_Interstellar_Comet_With_Hyperbolic_Orbit
    hyperbolic_node = evaluator.add_leaf(
        id="Confirms_Interstellar_Comet_With_Hyperbolic_Orbit",
        desc="Confirms that 3I/ATLAS is an interstellar comet with a hyperbolic orbit indicating interstellar origin.",
        parent=eligibility_node,
        critical=True
    )
    # Discovery_Date_July_1_2025
    discovery_date_node = evaluator.add_leaf(
        id="Discovery_Date_July_1_2025",
        desc="States/uses that 3I/ATLAS was discovered on July 1, 2025 (consistent with the prompt constraint).",
        parent=eligibility_node,
        critical=True
    )
    # Discovery_By_ATLAS
    discovery_by_atlas_node = evaluator.add_leaf(
        id="Discovery_By_ATLAS",
        desc="States/uses that 3I/ATLAS was discovered by the Asteroid Terrestrial-impact Last Alert System (ATLAS).",
        parent=eligibility_node,
        critical=True
    )
    # Multi_Institutional_Collaboration
    multi_institution_node = evaluator.add_leaf(
        id="Multi_Institutional_Collaboration",
        desc="The paper represents a multi-institutional collaboration (e.g., authors have affiliations spanning multiple institutions).",
        parent=eligibility_node,
        critical=True
    )

    # Build claims and run batch verification for eligibility (depend on Reference_URL_Provided)
    claims_and_sources: List[tuple] = []

    # Subject_Matter
    claim_subject = (
        "This paper is primarily about the interstellar comet '3I/ATLAS' (allowing minor naming variants like '3I (ATLAS)', "
        "'3I ATLAS', or similar forms)."
    )
    ins_subject = (
        "Verify the main topic focuses on 3I/ATLAS. Allow minor formatting variants (e.g., '3I (ATLAS)', '3I ATLAS', "
        "or alternative notations referring to the same object)."
    )
    claims_and_sources.append((claim_subject, sources, subject_node, ins_subject))

    # First_Peer_Reviewed_Publication
    claim_first_peer = (
        "This is the first peer-reviewed scientific paper published about the interstellar comet 3I/ATLAS."
    )
    ins_first_peer = (
        "Look for explicit statements like 'first results', 'first observations', 'first peer-reviewed report', "
        "or other wording that clearly claims primacy. If the webpage does not explicitly support this claim, "
        "judge it as not supported."
    )
    claims_and_sources.append((claim_first_peer, sources, first_peer_node, ins_first_peer))

    # Venue_And_Year
    claim_venue_year = (
        "This paper was published in 2025 in 'Monthly Notices of the Royal Astronomical Society: Letters' "
        "(also known as 'MNRAS Letters' or 'MNRAS Lett.')."
    )
    ins_venue_year = (
        "Verify that the journal venue is the Letters section of MNRAS (e.g., 'MNRAS Letters', 'MNRAS: Letters', "
        "'Monthly Notices of the Royal Astronomical Society Letters') and that the publication year is 2025."
    )
    claims_and_sources.append((claim_venue_year, sources, venue_year_node, ins_venue_year))

    # Original Observational Data from Specific Telescopes
    claim_telescopes = (
        "This paper reports original observational data obtained using all three specified telescopes: "
        "the Kottamia Astronomical Observatory 1.88-m telescope, the Palomar 200-inch (Hale) telescope, "
        "and the ARC 3.5-m telescope at Apache Point Observatory."
    )
    ins_telescopes = (
        "Check the Observations/Data sections for explicit mentions that new data were obtained with each of the three: "
        "1) 'Kottamia 1.88-m' (Kottamia Astronomical Observatory, Egypt), "
        "2) 'Palomar 200-inch' (Hale Telescope), "
        "3) 'ARC 3.5-m' (Apache Point Observatory). "
        "Minor variations in naming are acceptable if they clearly refer to these instruments. "
        "The page must make it clear that these data are original observations reported by the authors (not merely cited)."
    )
    claims_and_sources.append((claim_telescopes, sources, telescopes_node, ins_telescopes))

    # Confirms interstellar hyperbolic orbit
    claim_hyperbolic = (
        "This paper confirms that 3I/ATLAS is an interstellar comet on a hyperbolic orbit (e.g., eccentricity e > 1), "
        "indicating an interstellar origin."
    )
    ins_hyperbolic = (
        "Look for language indicating a hyperbolic orbit or interstellar origin (e.g., 'hyperbolic', 'e>1', 'interstellar object'). "
        "The confirmation should be explicit or clearly supported by the content."
    )
    claims_and_sources.append((claim_hyperbolic, sources, hyperbolic_node, ins_hyperbolic))

    # Discovery date
    claim_disc_date = "This paper states that 3I/ATLAS was discovered on July 1, 2025 (UTC)."
    ins_disc_date = (
        "Look for explicit mention of the discovery date written in any reasonable format, e.g., '2025-07-01', "
        "'1 July 2025', or 'July 1, 2025'."
    )
    claims_and_sources.append((claim_disc_date, sources, discovery_date_node, ins_disc_date))

    # Discovery by ATLAS
    claim_disc_by = (
        "This paper states that 3I/ATLAS was discovered by the Asteroid Terrestrial-impact Last Alert System (ATLAS)."
    )
    ins_disc_by = (
        "Check for a clear attribution to ATLAS (the Asteroid Terrestrial-impact Last Alert System) as the discoverer."
    )
    claims_and_sources.append((claim_disc_by, sources, discovery_by_atlas_node, ins_disc_by))

    # Multi-institutional collaboration
    claim_multi_inst = (
        "This paper involves a multi-institutional collaboration (authors have affiliations from more than one institution)."
    )
    ins_multi_inst = (
        "Check the author list and affiliations to confirm at least two distinct institutions are represented."
    )
    claims_and_sources.append((claim_multi_inst, sources, multi_institution_node, ins_multi_inst))

    # Run batch verification with dependency on 'Reference_URL_Provided'
    await evaluator.batch_verify(
        claims_and_sources=claims_and_sources,
        extra_prerequisites=[reference_url_provided_node]
    )

    # Add some helpful debugging info
    evaluator.add_custom_info(
        {
            "selected_title_from_answer": paper.title,
            "selected_lead_author_from_answer": paper.lead_author,
            "reference_urls_from_answer": sources,
            "telescope_requirements": TELESCOPE_REQUIREMENTS
        },
        info_type="debug_info"
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
    Evaluate an answer for the 3I/ATLAS first peer-reviewed paper identification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Top-level orchestration; we keep a container root
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

    # 1) Extract candidate paper(s) from the answer
    candidates = await evaluator.extract(
        prompt=prompt_extract_paper_candidates(),
        template_class=PaperCandidates,
        extraction_name="paper_candidates"
    )

    # 2) Select primary paper (first in order)
    primary_paper = select_primary_paper(candidates)

    # 3) Build verification tree and run checks
    await build_and_verify(evaluator, root, primary_paper, logger)

    # 4) Return structured result
    return evaluator.get_summary()