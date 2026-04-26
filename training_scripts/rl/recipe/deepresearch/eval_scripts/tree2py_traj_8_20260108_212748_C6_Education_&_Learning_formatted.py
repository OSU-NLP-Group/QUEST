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
TASK_ID = "ca_online_masters_programs"
TASK_DESCRIPTION = (
    "I am a working professional in California looking to advance my career by earning a master's degree in a STEM or "
    "business analytics-related field. Due to my work schedule and budget constraints, I need programs that offer maximum "
    "flexibility and affordability. Please identify three distinct online master's degree programs from universities "
    "located in California that meet ALL of the following requirements: (1) The university must be located in California "
    "and hold regional accreditation (not just national accreditation); (2) The program must be 100% online with asynchronous "
    "course delivery (no mandatory synchronous class sessions or in-person campus visits required); (3) The program must be "
    "completable within 24 months for a full-time student; (4) The program must not require GRE, GMAT, or other standardized "
    "entrance exams (either automatically waived or not required at all); (5) The total program tuition must be under $25,000; "
    "(6) The program must be in a STEM field (Science, Technology, Engineering, Mathematics) or in Business Analytics/Data "
    "Science; (7) The program must be currently accepting applications for the 2025-2026 academic year. For each program, "
    "please provide the university name, the specific master's program name, a reference URL to the official program page, "
    "and a reference URL confirming the university's regional accreditation status."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    program_url: Optional[str] = None  # Official program page
    accreditation_url: Optional[str] = None  # Page confirming regional accreditation
    extra_urls: List[str] = Field(default_factory=list)  # Any additional URLs cited in the answer for this program


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract all distinct online master's degree programs mentioned in the answer. For each program, extract:
    - university_name: The university name
    - program_name: The specific master's program name
    - program_url: The URL to the official program page (a page that describes the specific program)
    - accreditation_url: A URL that explicitly confirms the university's regional accreditation (e.g., WSCUC/WASC Senior College and University Commission page, a university accreditation page that states WSCUC)
    - extra_urls: Any additional URLs included in the answer that relate specifically to this program (e.g., tuition/cost page, admissions deadlines page, modality details). Only include URLs that appear in the answer.

    Notes:
    - Do NOT invent URLs. Only extract URLs explicitly present in the answer (including those embedded in markdown).
    - If a field is missing for a program, set it to null (or an empty list for extra_urls).
    - If the answer mentions more than three programs, extract them all (the evaluator will consider only the first three).
    - If the answer mentions fewer than three programs, extract whatever is present.

    Return a JSON object with: { "programs": [ ... ] }.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _canon(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _gather_sources(p: ProgramItem) -> List[str]:
    urls = []
    if p.program_url:
        urls.append(p.program_url)
    if p.accreditation_url:
        urls.append(p.accreditation_url)
    urls.extend([u for u in (p.extra_urls or []) if isinstance(u, str) and u.strip() != ""])
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _programs_are_distinct(programs: List[ProgramItem]) -> bool:
    """
    Determine whether the first three programs are distinct.
    We consider them distinct if:
      - Their program_url values (when present) are unique; AND
      - For those with both university_name and program_name, the (university, program) pairs are unique.
    If some names are missing, we rely on URLs to detect obvious duplicates.
    """
    first3 = programs[:3]
    # URL uniqueness among non-empty URLs
    urls = [p.program_url for p in first3 if p.program_url and p.program_url.strip()]
    if len(urls) != len(set(urls)):
        return False

    # Name-pair uniqueness among fully specified pairs
    pairs = [(p.university_name, p.program_name) for p in first3 if p.university_name and p.program_name]
    canon_pairs = [(_canon(u), _canon(n)) for (u, n) in pairs]
    if len(canon_pairs) != len(set(canon_pairs)):
        return False

    return True


# --------------------------------------------------------------------------- #
# Per-program verification                                                    #
# --------------------------------------------------------------------------- #
async def verify_single_program(
    evaluator: Evaluator,
    parent_node,
    p: ProgramItem,
    index: int,
) -> None:
    """
    Build the verification subtree and perform checks for a single program.
    We follow a sequential structure internally:
      1) Required fields must be present (name, program name, URLs).
      2) Then verify all per-program constraints (parallel node), each as a critical leaf.
         Due to framework auto-preconditions on critical siblings, once one fails, subsequent may be skipped.
    """
    prog_idx = index + 1
    program_node = evaluator.add_parallel(
        id=f"program_{prog_idx}",
        desc=f"Program #{prog_idx} satisfies all per-program requirements",
        parent=parent_node,
        critical=False
    )

    # Sequential gate: required fields first, then constraints group
    seq_main = evaluator.add_sequential(
        id=f"program_{prog_idx}_main",
        desc=f"Program #{prog_idx} main verification flow",
        parent=program_node,
        critical=False
    )

    # Required fields presence (critical steps)
    univ_present = evaluator.add_custom_node(
        result=bool(p.university_name and p.university_name.strip()),
        id=f"program_{prog_idx}_university_name",
        desc="University name is provided",
        parent=seq_main,
        critical=True
    )
    prog_present = evaluator.add_custom_node(
        result=bool(p.program_name and p.program_name.strip()),
        id=f"program_{prog_idx}_program_name",
        desc="Specific master's program name is provided",
        parent=seq_main,
        critical=True
    )
    program_url_present = evaluator.add_custom_node(
        result=bool(p.program_url and p.program_url.strip()),
        id=f"program_{prog_idx}_official_program_url",
        desc="URL reference to the official program page is provided",
        parent=seq_main,
        critical=True
    )
    accred_url_present = evaluator.add_custom_node(
        result=bool(p.accreditation_url and p.accreditation_url.strip()),
        id=f"program_{prog_idx}_accreditation_url",
        desc="URL reference confirming the university's regional accreditation status is provided",
        parent=seq_main,
        critical=True
    )

    # After required fields, verify constraints in a parallel group
    constraints = evaluator.add_parallel(
        id=f"program_{prog_idx}_constraints",
        desc=f"Per-program requirement verifications for program #{prog_idx}",
        parent=seq_main,
        critical=False
    )

    # Common claim helpers
    uni = p.university_name or "the university"
    prog_name = p.program_name or "the program"
    all_sources = _gather_sources(p)

    # 1) University located in California
    leaf_california = evaluator.add_leaf(
        id=f"program_{prog_idx}_university_in_california",
        desc="University is located in California",
        parent=constraints,
        critical=True
    )
    claim_california = f"The university '{uni}' is located in California (United States)."
    await evaluator.verify(
        claim=claim_california,
        node=leaf_california,
        sources=all_sources,
        additional_instruction=(
            "Verify from the provided page(s) that the institution is based in California. "
            "Accept mentions of city/state (e.g., Los Angeles, CA; San Jose, California; etc.). "
            "If the page is for a specific campus within a CA-based university system, that counts."
        )
    )

    # 2) Regional accreditation
    leaf_accred = evaluator.add_leaf(
        id=f"program_{prog_idx}_regional_accreditation",
        desc="University holds regional accreditation (not only national accreditation)",
        parent=constraints,
        critical=True
    )
    claim_accred = (
        f"The provided accreditation page confirms that '{uni}' holds regional accreditation "
        f"(e.g., WSCUC / WASC Senior College and University Commission)."
    )
    await evaluator.verify(
        claim=claim_accred,
        node=leaf_accred,
        sources=p.accreditation_url if p.accreditation_url else all_sources,
        additional_instruction=(
            "Treat WSCUC/WASC Senior College and University Commission as regional accreditation. "
            "Programmatic accreditations (e.g., ABET) do NOT count for this requirement."
        )
    )

    # 3) Fully online, no in-person requirements
    leaf_online = evaluator.add_leaf(
        id=f"program_{prog_idx}_fully_online_no_in_person",
        desc="Program is 100% online with no mandatory in-person campus visits/residency requirements",
        parent=constraints,
        critical=True
    )
    claim_online = (
        f"The '{prog_name}' at '{uni}' is delivered 100% online and does not require any in-person campus visits, "
        f"residencies, or on-campus components."
    )
    await evaluator.verify(
        claim=claim_online,
        node=leaf_online,
        sources=all_sources,
        additional_instruction=(
            "Look for phrases like '100% online', 'fully online', 'no campus visits required', 'no residency'. "
            "If any mandatory in-person component or residency is required, this should be marked as not supported."
        )
    )

    # 4) Asynchronous delivery (no mandatory live sessions)
    leaf_async = evaluator.add_leaf(
        id=f"program_{prog_idx}_asynchronous_delivery",
        desc="Program is asynchronous with no mandatory real-time synchronous class sessions",
        parent=constraints,
        critical=True
    )
    claim_async = (
        f"The '{prog_name}' at '{uni}' offers asynchronous delivery with no mandatory real-time synchronous class sessions "
        f"(i.e., no required live class meetings)."
    )
    await evaluator.verify(
        claim=claim_async,
        node=leaf_async,
        sources=all_sources,
        additional_instruction=(
            "Accept 'asynchronous', 'self-paced', or explicit statements that live sessions are not required. "
            "If the page indicates required real-time sessions, mark as not supported."
        )
    )

    # 5) Completion within 24 months
    leaf_24mo = evaluator.add_leaf(
        id=f"program_{prog_idx}_completion_within_24_months",
        desc="Program is structured to allow completion within 24 months for a full-time student",
        parent=constraints,
        critical=True
    )
    claim_24mo = (
        f"The '{prog_name}' at '{uni}' can be completed within 24 months (two years) by a full-time student."
    )
    await evaluator.verify(
        claim=claim_24mo,
        node=leaf_24mo,
        sources=all_sources,
        additional_instruction=(
            "Look for explicit duration statements like '12-24 months', '2 years', or a credit structure + pace indicating "
            "completion within 24 months at full-time enrollment."
        )
    )

    # 6) No standardized entrance exams (GRE/GMAT) required
    leaf_no_tests = evaluator.add_leaf(
        id=f"program_{prog_idx}_no_standardized_exam_required",
        desc="Program does not require GRE/GMAT/other standardized entrance exams (waived or not required)",
        parent=constraints,
        critical=True
    )
    claim_no_tests = (
        f"The '{prog_name}' at '{uni}' does not require GRE, GMAT, or other standardized entrance exams for admission "
        f"(either not required or automatically waived)."
    )
    await evaluator.verify(
        claim=claim_no_tests,
        node=leaf_no_tests,
        sources=all_sources,
        additional_instruction=(
            "Accept explicit statements like 'No GRE required' or 'GRE/GMAT waived/not required'. "
            "If standardized test requirement is optional or only for specific cases, confirm that there is a general pathway "
            "with no required GRE/GMAT. If tests are required for any normal applicant, mark as not supported."
        )
    )

    # 7) Total tuition under $25,000
    leaf_tuition = evaluator.add_leaf(
        id=f"program_{prog_idx}_tuition_under_25000",
        desc="Total program tuition is under $25,000",
        parent=constraints,
        critical=True
    )
    claim_tuition = (
        f"The total tuition for the '{prog_name}' at '{uni}' is under $25,000."
    )
    await evaluator.verify(
        claim=claim_tuition,
        node=leaf_tuition,
        sources=all_sources,
        additional_instruction=(
            "Use the program's tuition page or cost breakdown. If only per-credit tuition is provided, multiply by the total "
            "number of required credits to estimate total tuition. Ignore fees for this calculation unless the page explicitly "
            "states 'total program tuition' including fees. If the total equals or exceeds $25,000, mark as not supported."
        )
    )

    # 8) Field is STEM or Analytics/Data Science
    leaf_field = evaluator.add_leaf(
        id=f"program_{prog_idx}_field_is_stem_or_analytics",
        desc="Program is in a STEM field or Business Analytics/Data Analytics/Data Science",
        parent=constraints,
        critical=True
    )
    claim_field = (
        f"The '{prog_name}' at '{uni}' is a STEM field program or is in Business Analytics/Data Analytics/Data Science."
    )
    await evaluator.verify(
        claim=claim_field,
        node=leaf_field,
        sources=all_sources,
        additional_instruction=(
            "Accept standard STEM degrees (Science, Technology, Engineering, Mathematics) and degrees named or clearly falling "
            "under Business Analytics, Data Analytics, or Data Science."
        )
    )

    # 9) Master's degree level
    leaf_level = evaluator.add_leaf(
        id=f"program_{prog_idx}_masters_degree_level",
        desc="Program is a master's degree program (not certificate/doctoral)",
        parent=constraints,
        critical=True
    )
    claim_level = (
        f"The '{prog_name}' at '{uni}' is a master's degree program (e.g., MS, MA, MBA, MEng) and not a certificate or doctoral degree."
    )
    await evaluator.verify(
        claim=claim_level,
        node=leaf_level,
        sources=all_sources,
        additional_instruction=(
            "Verify the credential level on the program page. It must be a master's degree, not a graduate certificate, microcredential, "
            "or doctoral program."
        )
    )

    # 10) Currently accepting applications for 2025–2026 academic year
    leaf_accepting = evaluator.add_leaf(
        id=f"program_{prog_idx}_accepting_2025_2026_applications",
        desc="Program is currently accepting applications for the 2025–2026 academic year",
        parent=constraints,
        critical=True
    )
    claim_accepting = (
        f"The '{prog_name}' at '{uni}' is accepting applications for the 2025–2026 academic year."
    )
    await evaluator.verify(
        claim=claim_accepting,
        node=leaf_accepting,
        sources=all_sources,
        additional_instruction=(
            "Look for admissions pages or program pages indicating open applications for upcoming terms in 2025 or 2026 "
            "(e.g., Fall 2025, Spring 2026, Summer 2026). If the page lists application deadlines for those terms or explicitly "
            "states applications are open for 2025–2026, consider it supported."
        )
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
    Evaluate an answer for the California online master's programs task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction",
    )

    # Record a small custom info summary of raw extracted programs
    evaluator.add_custom_info(
        {
            "extracted_count": len(extraction.programs),
            "first_three_preview": [
                {
                    "university_name": p.university_name,
                    "program_name": p.program_name,
                    "program_url": p.program_url,
                    "accreditation_url": p.accreditation_url,
                }
                for p in extraction.programs[:3]
            ],
        },
        info_type="extraction_summary",
        info_name="extraction_summary_programs",
    )

    # Global requirements (critical)
    global_node = evaluator.add_parallel(
        id="global_requirements",
        desc="Global requirements across the full set of returned programs",
        parent=root,
        critical=True
    )

    # Note: For robustness and in line with general evaluation guidance,
    # we consider 'at least three' available programs and evaluate the first three.
    # This avoids penalizing answers that provide more than 3 good options.
    three_provided = evaluator.add_custom_node(
        result=(len(extraction.programs) >= 3),
        id="three_programs_provided",
        desc="At least three programs are provided (the evaluator will consider the first three).",
        parent=global_node,
        critical=True
    )

    # Distinctness among the first three
    top3_programs = extraction.programs[:3]
    # Pad with empty entries if fewer than 3 present (to keep tree shape consistent)
    while len(top3_programs) < 3:
        top3_programs.append(ProgramItem())

    programs_distinct = evaluator.add_custom_node(
        result=_programs_are_distinct(top3_programs),
        id="programs_are_distinct",
        desc="The three programs are distinct (no duplicate university+program or repeated program URLs among the first three).",
        parent=global_node,
        critical=True
    )

    # Per-program verification
    # Place programs under a container node for clarity (non-critical, parallel)
    programs_container = evaluator.add_parallel(
        id="programs_container",
        desc="Per-program verifications for the first three programs",
        parent=root,
        critical=False
    )

    # Build verification subtrees for first 3 programs
    for idx in range(3):
        await verify_single_program(evaluator, programs_container, top3_programs[idx], idx)

    # Return final structured summary
    return evaluator.get_summary()