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
TASK_ID = "la_alt_cert_stem_6_12"
TASK_DESCRIPTION = (
    "Identify four alternative teacher certification programs in Louisiana that accept career-changers "
    "with bachelor's degrees and lead to certification for teaching STEM subjects (Science and/or Mathematics) "
    "in grades 6-12. For each program, provide: (1) the type of Louisiana teaching certificate it leads to "
    "(Level 1 Professional Certificate or Practitioner License), (2) the program format (online, hybrid, "
    "or in-person), (3) the program duration or completion timeline, and (4) contact information or "
    "institutional affiliation with a reference URL."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    """Structured info for one program item extracted from the answer."""
    name: Optional[str] = None
    certificate_type: Optional[str] = None  # e.g., "Level 1 Professional Certificate" or "Practitioner License"
    format: Optional[str] = None            # e.g., "online", "hybrid", "in-person"
    duration: Optional[str] = None          # e.g., "1 year", "12-18 months", "varies"
    contact_or_affiliation: Optional[str] = None  # e.g., email/phone or institution/department name
    reference_urls: List[str] = Field(default_factory=list)

    # Optional contextual fields (strings preferred to maximize compatibility)
    subject_areas: List[str] = Field(default_factory=list)   # e.g., ["Mathematics", "Science", "Physics"]
    grade_levels: Optional[str] = None                      # e.g., "Grades 6-12", "secondary (6-12)"

    # Admission / pathway descriptors (strings; used for contextual verification)
    louisiana_affiliation: Optional[str] = None             # free-text mention that it's LA-based
    alternative_pathway_desc: Optional[str] = None          # free-text mention of alternative certification route
    bachelors_requirement_desc: Optional[str] = None        # free-text mention accepting bachelor's/career-changers


class ProgramsExtraction(BaseModel):
    """Top-level extracted list of programs."""
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return (
        "Extract up to four alternative teacher certification programs described in the answer that are associated "
        "with Louisiana and relevant to teaching STEM (Science and/or Mathematics) in grades 6-12. For each program, "
        "return a JSON object with the following fields:\n"
        "1. name: The program name or title as stated.\n"
        "2. certificate_type: The Louisiana certificate type the program leads to (e.g., 'Level 1 Professional Certificate' "
        "or 'Practitioner License'). If the answer mentions variants (e.g., Practitioner Teacher License), return exactly "
        "what is written.\n"
        "3. format: The program format (online, hybrid, or in-person). If multiple formats are mentioned, return the most "
        "prominent or list the main one as a single string.\n"
        "4. duration: The program duration or completion timeline (e.g., '1 year', '12-18 months'). Return exactly as written.\n"
        "5. contact_or_affiliation: A contact method (email/phone/portal) or the institutional affiliation (e.g., a department, "
        "university, or agency) as stated in the answer.\n"
        "6. reference_urls: All reference URLs explicitly mentioned that support this program.\n"
        "7. subject_areas: Any STEM subject areas explicitly mentioned (e.g., 'Mathematics', 'Science', 'Physics', 'Chemistry').\n"
        "8. grade_levels: Any grade levels explicitly mentioned (e.g., 'Grades 6-12', 'secondary', '6-12').\n"
        "9. louisiana_affiliation: Text in the answer indicating this is a Louisiana program or affiliated with a Louisiana institution/agency.\n"
        "10. alternative_pathway_desc: Text in the answer indicating this is an alternative certification pathway (post-baccalaureate).\n"
        "11. bachelors_requirement_desc: Text in the answer indicating the program accepts candidates with a bachelor's degree "
        "in non-education fields (career-changers).\n\n"
        "Rules:\n"
        "- Only extract information explicitly present in the answer; do not invent.\n"
        "- Include only full valid URLs in 'reference_urls'. If no URLs are present, return an empty list.\n"
        "- If any field is not mentioned for a program, set it to null (for strings) or an empty list (for arrays).\n"
        "- If more than four programs are present, return only the first four mentioned.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(program: ProgramItem, idx: int) -> str:
    return program.name.strip() if program.name else f"Program #{idx + 1}"


def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _clean_urls(urls: List[str]) -> List[str]:
    """Return only non-empty strings; leave as-is otherwise."""
    return [u for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification per program                                                    #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramItem,
    index: int,
) -> None:
    """
    Build verification subtree for one program and execute verifications.
    The program node is parallel; leaf nodes are critical checks aligned to rubric.
    """
    prog_label = _safe_name(program, index)
    prog_node = evaluator.add_parallel(
        id=f"Program_{index + 1}",
        desc=f"{prog_label} (one of the four required programs)",
        parent=parent_node,
        critical=False
    )

    urls = _clean_urls(program.reference_urls)

    # Create leaf nodes for evidence-based checks
    leaf_la_based = evaluator.add_leaf(
        id=f"P{index + 1}_Louisiana_Based",
        desc="Program is based in Louisiana (operated by a Louisiana institution/agency or explicitly offered as a Louisiana program)",
        parent=prog_node,
        critical=True,
    )
    leaf_alt_cert = evaluator.add_leaf(
        id=f"P{index + 1}_Alternative_Certification",
        desc="Program is an alternative certification pathway (not a traditional undergraduate teacher-prep pathway)",
        parent=prog_node,
        critical=True,
    )
    leaf_bachelors_career = evaluator.add_leaf(
        id=f"P{index + 1}_Career_Changer_Bachelors_NonEducation",
        desc="Program explicitly accepts/targets career-changers with bachelor’s degrees in non-education fields (post-baccalaureate entry)",
        parent=prog_node,
        critical=True,
    )
    leaf_stem = evaluator.add_leaf(
        id=f"P{index + 1}_STEM_Subject_Area",
        desc="Program leads to certification in Science and/or Mathematics",
        parent=prog_node,
        critical=True,
    )
    leaf_grades = evaluator.add_leaf(
        id=f"P{index + 1}_Grade_Level_6_12",
        desc="Program leads to certification for grades 6–12 (secondary level)",
        parent=prog_node,
        critical=True,
    )
    leaf_contact_url = evaluator.add_leaf(
        id=f"P{index + 1}_ContactOrAffiliation_With_URL",
        desc="Provides contact information and/or institutional affiliation AND includes a reference URL that supports the provided program information",
        parent=prog_node,
        critical=True,
    )

    # Build claims and run them in parallel
    claims_and_sources: List[tuple[str, List[str] | None, Any, Optional[str]]] = []

    claims_and_sources.append((
        f"The program '{prog_label}' is based in Louisiana or is explicitly a Louisiana alternative teacher certification program.",
        urls,
        leaf_la_based,
        "Supported if the page belongs to a Louisiana institution (e.g., LDOE or a Louisiana university) or explicitly states the program is for Louisiana."
    ))
    claims_and_sources.append((
        f"The program '{prog_label}' is an alternative certification pathway (post-baccalaureate) rather than a traditional undergraduate teacher-prep program.",
        urls,
        leaf_alt_cert,
        "Look for mentions of 'alternative certification', 'post-baccalaureate', 'alternative route', or similar phrasing indicating a non-undergraduate pathway."
    ))
    claims_and_sources.append((
        f"The program '{prog_label}' accepts candidates who already hold a bachelor's degree in a non-education field and targets career-changers.",
        urls,
        leaf_bachelors_career,
        "Check admission requirements for acceptance of bachelor's degree holders (non-education) and language indicating suitability for career-changers."
    ))
    claims_and_sources.append((
        f"The program '{prog_label}' leads to certification eligibility in secondary Mathematics and/or Science subjects.",
        urls,
        leaf_stem,
        "Evidence may include endorsements or certification areas such as Mathematics, Science, Biology, Chemistry, Physics, or General Science."
    ))
    claims_and_sources.append((
        f"The program '{prog_label}' leads to certification for grades 6–12 (secondary).",
        urls,
        leaf_grades,
        "Look for 'grades 6-12', 'secondary', 'middle/secondary', or statements covering both middle and high school levels."
    ))
    claims_and_sources.append((
        f"The provided URL(s) for '{prog_label}' is an official program or institutional page that includes contact information or clearly shows institutional affiliation for the program.",
        urls,
        leaf_contact_url,
        "It suffices if the page includes contact email/phone or clearly indicates the offering institution/department."
    ))

    await evaluator.batch_verify(claims_and_sources)

    # Existence checks for specified attributes: certificate type, format, duration
    evaluator.add_custom_node(
        result=_non_empty_str(program.certificate_type),
        id=f"P{index + 1}_Certificate_Type_Specified",
        desc="Program specifies whether it leads to a Level 1 Professional Certificate or a Practitioner License",
        parent=prog_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty_str(program.format),
        id=f"P{index + 1}_Format_Specified",
        desc="Program format is stated (online, hybrid, or in-person)",
        parent=prog_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty_str(program.duration),
        id=f"P{index + 1}_Duration_Specified",
        desc="Program duration or completion timeline is stated",
        parent=prog_node,
        critical=True
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
    Entry point to evaluate an agent's answer for Louisiana alternative certification programs (STEM, grades 6–12).
    Builds a verification tree aligned with the rubric and returns a structured summary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root evaluates programs independently
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

    # NOTE: The rubric's root was marked critical in JSON, but to allow partial credit across programs
    # and to satisfy the framework's critical-child consistency, we set root as non-critical here.

    # 1) Extract program list from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    # Normalize to exactly 4 programs (pad with empty items or truncate)
    programs: List[ProgramItem] = list(extracted.programs[:4])
    while len(programs) < 4:
        programs.append(ProgramItem())

    # 2) Build verification nodes per program
    for idx, prog in enumerate(programs):
        await verify_program(evaluator, root, prog, idx)

    # 3) Return summary
    return evaluator.get_summary()