import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "qc_oct2025_breakthrough"
TASK_DESCRIPTION = (
    "In October 2025, a quantum computing breakthrough was announced that demonstrated verifiable quantum advantage "
    "through a collaboration between Google Quantum AI and a University of California campus. The breakthrough involved "
    "running an algorithm on a quantum processor that achieved a speedup of at least 10,000 times compared to the fastest "
    "classical supercomputers. The research included proof-of-principle experiments that predicted the molecular structure "
    "of real molecules, with results validated against traditional methods.\n\n"
    "Identify this quantum computing breakthrough and provide the following information:\n\n"
    "1. The name of the algorithm or method announced\n"
    "2. The name of the quantum chip used in the demonstration\n"
    "3. The exact announcement date in October 2025\n"
    "4. The speedup factor achieved compared to classical supercomputers\n"
    "5. The name and role/title of at least one lead scientist from Google Quantum AI who authored the announcement\n"
    "6. The specific University of California campus that collaborated on this research\n"
    "7. The name, academic rank, and department of the UC faculty member who collaborated on this project\n"
    "8. The number of molecules studied in the proof-of-principle experiments and their sizes (atom counts)\n"
    "9. The validation method used to verify the quantum computing results\n"
    "10. The publication venue where this research was announced or published\n\n"
    "Provide valid URL references for each major claim."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class QCLeadScientist(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class UCFaculty(BaseModel):
    name: Optional[str] = None
    rank: Optional[str] = None
    department: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class QCABreakthroughExtraction(BaseModel):
    # Identity of the breakthrough
    breakthrough_title: Optional[str] = None
    breakthrough_urls: List[str] = Field(default_factory=list)

    # Algorithm / method
    algorithm_name: Optional[str] = None
    algorithm_sources: List[str] = Field(default_factory=list)

    # Quantum chip / processor
    quantum_chip_name: Optional[str] = None
    quantum_chip_sources: List[str] = Field(default_factory=list)

    # Announcement date
    announcement_date: Optional[str] = None
    announcement_date_sources: List[str] = Field(default_factory=list)

    # Speedup factor
    speedup_factor: Optional[str] = None
    speedup_sources: List[str] = Field(default_factory=list)

    # Verifiable quantum advantage claim
    vqa_sources: List[str] = Field(default_factory=list)

    # Google lead scientist (author)
    lead_scientist: Optional[QCLeadScientist] = None

    # UC campus collaborator
    uc_campus: Optional[str] = None
    uc_campus_sources: List[str] = Field(default_factory=list)

    # UC faculty collaborator details
    uc_faculty: Optional[UCFaculty] = None

    # Molecular structure experiments
    molecules_count: Optional[str] = None
    molecules_sizes: List[str] = Field(default_factory=list)
    molecules_sources: List[str] = Field(default_factory=list)

    # Validation method
    validation_method: Optional[str] = None
    validation_sources: List[str] = Field(default_factory=list)

    # Publication / announcement venue
    publication_venue: Optional[str] = None
    publication_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_breakthrough() -> str:
    return """
You will extract structured information exactly as stated in the provided answer text about the October 2025 Google Quantum AI–UC collaboration breakthrough.

Return a JSON object with the following fields. If an item is missing in the answer, set it to null (for strings) or an empty array (for lists).

1) breakthrough_title: The announcement/paper title or named breakthrough (string).
2) breakthrough_urls: Array of HTTP/HTTPS URLs that point to the announcement/paper describing the breakthrough.

3) algorithm_name: The algorithm or method name used in the demonstration (string).
4) algorithm_sources: Array of HTTP/HTTPS URLs that support the stated algorithm/method.

5) quantum_chip_name: The name of the quantum processor/chip used (string).
6) quantum_chip_sources: Array of HTTP/HTTPS URLs that support the stated chip.

7) announcement_date: The exact announcement date as written in the answer (string, e.g., "October 14, 2025").
8) announcement_date_sources: Array of HTTP/HTTPS URLs that support the announcement date.

9) speedup_factor: The speedup factor text as written in the answer (string, e.g., "10,000×" or "at least 10^4").
10) speedup_sources: Array of HTTP/HTTPS URLs that support the speedup.

11) vqa_sources: Array of HTTP/HTTPS URLs that explicitly support that this work demonstrates "verifiable quantum advantage".

12) lead_scientist: Object with:
    - name: Name of at least one lead scientist from Google Quantum AI who authored the announcement (string).
    - title: Their role/title (string).
    - sources: Array of HTTP/HTTPS URLs supporting authorship and title.

13) uc_campus: The specific University of California campus collaborator (string, e.g., "UC Santa Barbara") as written in the answer.
14) uc_campus_sources: Array of HTTP/HTTPS URLs supporting the named campus collaboration.

15) uc_faculty: Object with:
    - name: UC faculty collaborator (string).
    - rank: Academic rank (e.g., "Professor", "Associate Professor", "Assistant Professor") as written in the answer.
    - department: Department name as written in the answer.
    - sources: Array of HTTP/HTTPS URLs supporting the faculty identity/rank/department and their collaboration role.

16) molecules_count: Number of real molecules studied in the proof-of-principle experiments (string, keep formatting in the answer).
17) molecules_sizes: Array of the sizes (atom counts) per molecule as provided in the answer (strings, e.g., "8 atoms", "12", "H2O: 3 atoms").
18) molecules_sources: Array of HTTP/HTTPS URLs that support the number of molecules and their sizes.

19) validation_method: The traditional validation method used to verify the quantum results (string), e.g., "coupled-cluster", "DFT", "classical simulation cross-check", etc.
20) validation_sources: Array of HTTP/HTTPS URLs supporting the validation method.

21) publication_venue: The publication/announcement venue (string), e.g., "Nature", "Science", "arXiv", or "Google AI Blog".
22) publication_sources: Array of HTTP/HTTPS URLs supporting the venue.

STRICT URL RULES:
- Include only URLs explicitly present in the answer text.
- Only valid HTTP or HTTPS URLs. If a URL is missing a protocol, prepend http://
- If the answer contains no URL for any field, return an empty array for that field's sources.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _ai_with_url_requirement(base: str, urls: List[str]) -> str:
    suffix = ""
    if not urls:
        suffix = "\nImportant: The answer provided no source URL for this item. You must judge the claim as not supported/Incorrect."
    return (base or "").strip() + suffix


def _join_sizes(sizes: List[str]) -> str:
    cleaned = [s.strip() for s in (sizes or []) if isinstance(s, str)]
    return ", ".join(cleaned) if cleaned else ""


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
    # Initialize evaluator
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
    # Make root critical to match rubric intent (all children must be critical)
    root.critical = True

    # Extract structured information from the answer
    extraction: QCABreakthroughExtraction = await evaluator.extract(
        prompt=prompt_extract_breakthrough(),
        template_class=QCABreakthroughExtraction,
        extraction_name="breakthrough_extraction",
    )

    # Build verification nodes (all critical under critical root)
    claims_and_sources: List[tuple[str, List[str], Any, Optional[str]]] = []

    # 1) Breakthrough identity
    node_identity = evaluator.add_leaf(
        id="breakthrough_identity",
        desc="Breakthrough is clearly identified (e.g., announcement/paper title or named breakthrough) with supporting URL.",
        parent=root,
        critical=True,
    )
    identity_sources = _safe_list(extraction.breakthrough_urls)
    identity_title = extraction.breakthrough_title or ""
    identity_claim = (
        f"The announcement/paper describing the October 2025 Google Quantum AI–UC breakthrough is titled '{identity_title}' "
        f"(or an equivalent title), and the provided page(s) corresponds to that breakthrough."
    )
    identity_ai = _ai_with_url_requirement(
        "Verify that at least one provided page is the primary announcement or paper for this breakthrough. "
        "Allow minor title variations, punctuation changes, or subtitle differences.",
        identity_sources,
    )
    claims_and_sources.append((identity_claim, identity_sources, node_identity, identity_ai))

    # 2) Algorithm or method
    node_algo = evaluator.add_leaf(
        id="algorithm_or_method",
        desc="Provides the algorithm/method name used in the demonstration and at least one supporting HTTP(S) URL.",
        parent=root,
        critical=True,
    )
    algo_sources = _safe_list(extraction.algorithm_sources)
    algo_name = extraction.algorithm_name or ""
    algo_claim = f"The algorithm/method used in the demonstration is named '{algo_name}'."
    algo_ai = _ai_with_url_requirement(
        "Confirm the page explicitly names the algorithm/method used in this exact October 2025 breakthrough. "
        "Accept common synonyms or standard abbreviations.",
        algo_sources,
    )
    claims_and_sources.append((algo_claim, algo_sources, node_algo, algo_ai))

    # 3) Quantum chip / processor
    node_chip = evaluator.add_leaf(
        id="quantum_chip",
        desc="Provides the publicly announced quantum processor/chip name used and at least one supporting HTTP(S) URL.",
        parent=root,
        critical=True,
    )
    chip_sources = _safe_list(extraction.quantum_chip_sources)
    chip_name = extraction.quantum_chip_name or ""
    chip_claim = f"The quantum processor/chip used in the demonstration is named '{chip_name}'."
    chip_ai = _ai_with_url_requirement(
        "Verify that the page names the chip/processor used for this demonstration. "
        "Accept code names or platform names if they are standard for Google's quantum processors.",
        chip_sources,
    )
    claims_and_sources.append((chip_claim, chip_sources, node_chip, chip_ai))

    # 4) Announcement date (October 2025)
    node_date = evaluator.add_leaf(
        id="announcement_date",
        desc="Provides the exact announcement date and it is in October 2025, with at least one supporting HTTP(S) URL.",
        parent=root,
        critical=True,
    )
    date_sources = _safe_list(extraction.announcement_date_sources)
    date_text = extraction.announcement_date or ""
    date_claim = f"The announcement date was '{date_text}', and it falls in October 2025."
    date_ai = _ai_with_url_requirement(
        "Verify the page's dateline or publication date shows the announcement in October 2025. "
        "Minor timezone or regional date format differences are acceptable.",
        date_sources,
    )
    claims_and_sources.append((date_claim, date_sources, node_date, date_ai))

    # 5) Speedup factor (≥ 10,000×)
    node_speedup = evaluator.add_leaf(
        id="speedup_factor",
        desc="Provides the speedup factor and it is ≥ 10,000× versus classical supercomputers, with at least one supporting HTTP(S) URL.",
        parent=root,
        critical=True,
    )
    speed_sources = _safe_list(extraction.speedup_sources)
    speed_text = extraction.speedup_factor or ""
    speed_claim = (
        f"The speedup factor reported for the breakthrough is '{speed_text}', and it is at least 10,000× "
        f"compared to the fastest classical supercomputers."
    )
    speed_ai = _ai_with_url_requirement(
        "Verify that the page explicitly supports a ≥10,000× speedup versus the fastest classical supercomputers. "
        "Accept forms like '10^4', 'ten thousand times', or equivalent statements.",
        speed_sources,
    )
    claims_and_sources.append((speed_claim, speed_sources, node_speedup, speed_ai))

    # 6) Verifiable quantum advantage
    node_vqa = evaluator.add_leaf(
        id="verifiable_quantum_advantage",
        desc="Explicitly states that the breakthrough demonstrated verifiable quantum advantage (not just a general claim) and provides at least one supporting HTTP(S) URL.",
        parent=root,
        critical=True,
    )
    vqa_sources = _safe_list(extraction.vqa_sources)
    vqa_claim = "The announcement explicitly states that the work demonstrates verifiable quantum advantage."
    vqa_ai = _ai_with_url_requirement(
        "The page must explicitly support 'verifiable quantum advantage' (not just 'quantum advantage'). "
        "Accept equivalent phrasing like 'verified quantum advantage' if clearly referring to formal verification.",
        vqa_sources,
    )
    claims_and_sources.append((vqa_claim, vqa_sources, node_vqa, vqa_ai))

    # 7) Google lead scientist (author + title)
    node_lead = evaluator.add_leaf(
        id="google_lead_scientist",
        desc="Gives at least one lead Google Quantum AI scientist who authored the announcement, including both name and role/title, with at least one supporting HTTP(S) URL.",
        parent=root,
        critical=True,
    )
    lead_name = extraction.lead_scientist.name if extraction.lead_scientist else ""
    lead_title = extraction.lead_scientist.title if extraction.lead_scientist else ""
    lead_sources = _safe_list(extraction.lead_scientist.sources if extraction.lead_scientist else [])
    lead_claim = (
        f"{lead_name} is a lead scientist with the title '{lead_title}' at Google Quantum AI and authored the announcement "
        f"(or is listed as an author)."
    )
    lead_ai = _ai_with_url_requirement(
        "Verify the person is credited on the announcement or in an official Google/Google Quantum AI page, and that their role/title is stated.",
        lead_sources,
    )
    claims_and_sources.append((lead_claim, lead_sources, node_lead, lead_ai))

    # 8) UC campus collaborator
    node_campus = evaluator.add_leaf(
        id="uc_campus",
        desc="Names the specific University of California campus collaborator and provides at least one supporting HTTP(S) URL.",
        parent=root,
        critical=True,
    )
    campus = extraction.uc_campus or ""
    campus_sources = _safe_list(extraction.uc_campus_sources)
    campus_claim = f"The collaborating University of California campus on this project is '{campus}'."
    campus_ai = _ai_with_url_requirement(
        "Verify the page explicitly names the UC campus collaborating with Google Quantum AI for this breakthrough.",
        campus_sources,
    )
    claims_and_sources.append((campus_claim, campus_sources, node_campus, campus_ai))

    # 9) UC faculty collaborator (name, rank, department)
    node_faculty = evaluator.add_leaf(
        id="uc_faculty_collaborator",
        desc="Names the UC faculty collaborator and provides their academic rank and department, with at least one supporting HTTP(S) URL.",
        parent=root,
        critical=True,
    )
    fac_name = extraction.uc_faculty.name if extraction.uc_faculty else ""
    fac_rank = extraction.uc_faculty.rank if extraction.uc_faculty else ""
    fac_dept = extraction.uc_faculty.department if extraction.uc_faculty else ""
    fac_sources = _safe_list(extraction.uc_faculty.sources if extraction.uc_faculty else [])
    faculty_claim = (
        f"The UC faculty collaborator is '{fac_name}', with academic rank '{fac_rank}' in the Department of '{fac_dept}', "
        f"and collaborated on this project."
    )
    faculty_ai = _ai_with_url_requirement(
        "Verify the faculty member's name, academic rank (e.g., Professor/Associate/Assistant), department, and collaboration role are supported.",
        fac_sources,
    )
    claims_and_sources.append((faculty_claim, fac_sources, node_faculty, faculty_ai))

    # 10) Molecular structure proof-of-principle experiments
    node_molecules = evaluator.add_leaf(
        id="molecular_structure_experiments",
        desc="States that proof-of-principle experiments predicted molecular structure of real molecules, and provides the number and sizes (atom counts), with at least one supporting HTTP(S) URL.",
        parent=root,
        critical=True,
    )
    mol_count = extraction.molecules_count or ""
    mol_sizes = _join_sizes(extraction.molecules_sizes)
    mol_sources = _safe_list(extraction.molecules_sources)
    mol_claim = (
        f"The proof-of-principle experiments predicted molecular structure of real molecules. "
        f"The number of molecules studied was '{mol_count}', and their atom counts were: {mol_sizes}."
    )
    mol_ai = _ai_with_url_requirement(
        "Verify the page reports both the number of real molecules studied and their sizes (atom counts). "
        "Allow minor formatting variations or per-molecule annotations.",
        mol_sources,
    )
    claims_and_sources.append((mol_claim, mol_sources, node_molecules, mol_ai))

    # 11) Validation method
    node_validation = evaluator.add_leaf(
        id="validation_method",
        desc="Specifies the traditional validation method used to verify the quantum results and provides at least one supporting HTTP(S) URL.",
        parent=root,
        critical=True,
    )
    val_method = extraction.validation_method or ""
    val_sources = _safe_list(extraction.validation_sources)
    val_claim = f"The quantum results were validated using '{val_method}'."
    val_ai = _ai_with_url_requirement(
        "Verify the page names the specific traditional/classical validation method (e.g., coupled-cluster, FCI, DFT, classical simulation cross-check).",
        val_sources,
    )
    claims_and_sources.append((val_claim, val_sources, node_validation, val_ai))

    # 12) Publication / announcement venue
    node_venue = evaluator.add_leaf(
        id="publication_venue",
        desc="Identifies the publication/announcement venue and provides at least one supporting HTTP(S) URL.",
        parent=root,
        critical=True,
    )
    venue = extraction.publication_venue or ""
    venue_sources = _safe_list(extraction.publication_sources)
    venue_claim = f"The research was announced or published at '{venue}'."
    venue_ai = _ai_with_url_requirement(
        "Verify that the page indicates the venue (e.g., Nature, Science, arXiv, Google AI Blog) for this breakthrough.",
        venue_sources,
    )
    claims_and_sources.append((venue_claim, venue_sources, node_venue, venue_ai))

    # Run all verifications in parallel
    await evaluator.batch_verify(claims_and_sources)

    # Return summary
    return evaluator.get_summary()