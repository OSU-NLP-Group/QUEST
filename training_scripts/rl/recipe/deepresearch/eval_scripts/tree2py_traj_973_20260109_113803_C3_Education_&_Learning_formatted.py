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
TASK_ID = "phd_program_requirements"
TASK_DESCRIPTION = """Identify a university with a Computer Science PhD program ranked in the top 10 according to U.S. News & World Report, QS World University Rankings, or another recognized ranking system. For that program, document the following sequential degree milestones:

1. The qualifying examination or comprehensive examination requirement, including its format (written, oral, or both)
2. The timeline or deadline by which students must advance to PhD candidacy
3. The dissertation defense requirement, including confirmation that an oral examination or defense is required
4. The minimum research or dissertation credit hours required beyond coursework

For your answer, provide:
- The university name and a URL confirming its top-10 ranking
- A description of the qualifying/comprehensive exam format and a URL to the official program page describing this requirement
- The specific candidacy advancement timeline and a URL to the page specifying this deadline
- Confirmation of the dissertation defense requirement with oral examination and a URL to the page describing these requirements
- The minimum research credit hours required (if specified) and a URL to the complete program requirements page
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramInfoExtraction(BaseModel):
    # Program identification
    university_name: Optional[str] = None
    ranking_system: Optional[str] = None  # e.g., "U.S. News", "QS"
    ranking_source_url: Optional[str] = None

    # Milestone 1: Qualifying or Comprehensive Exam
    exam_requirement_text: Optional[str] = None
    exam_format: Optional[str] = None  # expected values like "written", "oral", "both", "unspecified"
    exam_url: Optional[str] = None

    # Milestone 2: Candidacy Advancement Timeline
    candidacy_timeline_text: Optional[str] = None  # e.g., "by the end of the second year"
    candidacy_url: Optional[str] = None

    # Milestone 3: Dissertation and Defense
    dissertation_required_text: Optional[str] = None  # e.g., "A dissertation is required"
    oral_defense_required_text: Optional[str] = None  # e.g., "An oral defense is required"
    defense_url: Optional[str] = None

    # Milestone 4: Research/Dissertation Credits
    research_credits_required_text: Optional[str] = None  # e.g., "Students must complete research/dissertation credits beyond coursework"
    min_research_credits: Optional[str] = None  # e.g., "24 credits" or "not specified" (as literal string) or None
    program_requirements_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_program_info() -> str:
    return """
    Extract the requested information from the answer about a top-10 Computer Science PhD program and its milestones.
    Return exactly the following JSON fields:

    Program identification:
    - university_name: The specific university named in the answer (string).
    - ranking_system: The name of the ranking system cited (e.g., "U.S. News", "QS", "THE", "CSRankings"), if explicitly mentioned (string or null).
    - ranking_source_url: A direct URL in the answer that evidences the top-10 Computer Science ranking (string URL or null).

    Milestone 1 (Qualifying/Comprehensive Exam):
    - exam_requirement_text: The text in the answer that states the qualifying/comprehensive (or preliminary/qualifier) exam requirement (string or null).
    - exam_format: The exam format as stated in the answer: use 'written', 'oral', 'both', or 'unspecified' (string or null).
    - exam_url: A direct official URL in the answer that describes the exam requirement (string URL or null).

    Milestone 2 (Candidacy Advancement Timeline):
    - candidacy_timeline_text: The specific timeline/deadline as stated in the answer by which students must advance to PhD candidacy (string or null).
    - candidacy_url: A direct official URL in the answer that specifies this candidacy timeline (string URL or null).

    Milestone 3 (Dissertation and Defense):
    - dissertation_required_text: The text in the answer that confirms an original research dissertation is required (string or null).
    - oral_defense_required_text: The text in the answer that confirms an oral examination/defense is required (string or null).
    - defense_url: A direct official URL in the answer that describes dissertation and/or defense requirements (string URL or null).

    Milestone 4 (Research/Dissertation Credits):
    - research_credits_required_text: The text in the answer that confirms the program requires research/dissertation credits beyond coursework (string or null).
    - min_research_credits: If the answer specifies a minimum number of research/dissertation credits (e.g., '24 credits'), return that text; if the answer explicitly states no minimum is specified, return the literal string 'not specified'; otherwise return null.
    - program_requirements_url: A direct official URL in the answer that lists the complete PhD program requirements (string URL or null).

    IMPORTANT INSTRUCTIONS:
    - Extract only what is explicitly present in the answer.
    - For URL fields, extract actual URLs that appear in the answer (including URLs in markdown links). If no such URL is present, return null.
    - Do not invent URLs or values. If a field is not clearly stated, return null.
    - For exam_format, prefer a single token among: written, oral, both, unspecified. If the answer clearly indicates two components (e.g., both written and oral), use 'both'. If it mentions just 'exam' without format details, use 'unspecified'.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _collect_official_program_urls(info: ProgramInfoExtraction) -> List[str]:
    candidates = [
        info.exam_url,
        info.candidacy_url,
        info.defense_url,
        info.program_requirements_url
    ]
    return [u for u in candidates if _non_empty(u)]


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_program_identification(
    evaluator: Evaluator,
    parent_node,
    info: ProgramInfoExtraction
) -> None:
    """
    Build and verify 'Program_Identification' subtree:
    - University name provided (existence)
    - CS PhD program confirmed (via official URLs if available)
    - Top-10 ranking verified (via ranking URL)
    - Ranking source URL provided (existence)
    """
    node = evaluator.add_parallel(
        id="Program_Identification",
        desc="Identify the university/program and verify top-10 ranking with a source URL.",
        parent=parent_node,
        critical=True
    )

    # University name provided (existence)
    evaluator.add_custom_node(
        result=_non_empty(info.university_name),
        id="University_Name_Provided",
        desc="A specific university is named.",
        parent=node,
        critical=True
    )

    # Ranking source URL provided (existence)
    evaluator.add_custom_node(
        result=_non_empty(info.ranking_source_url),
        id="Ranking_Source_URL_Provided",
        desc="A direct URL to the ranking page evidencing the top-10 status is provided.",
        parent=node,
        critical=True
    )

    # Top-10 ranking verified (by ranking page)
    top10_leaf = evaluator.add_leaf(
        id="Top_10_Ranking_Verified",
        desc="The program/university is shown to be top-10 for Computer Science per U.S. News, QS, or another recognized ranking system.",
        parent=node,
        critical=True
    )
    rank_sys = info.ranking_system or "a recognized ranking system"
    uni_name = info.university_name or "the university"
    top10_claim = (
        f"The Computer Science subject ranking on this page shows that {uni_name} is ranked within the top 10 "
        f"according to {rank_sys}. Only accept if the ranking is specifically for Computer Science or a very close "
        f"CS subject category (e.g., Computer Science & Engineering, Computer Science & Information Systems). "
        f"Do not accept overall/university rankings that are not specific to Computer Science."
    )
    await evaluator.verify(
        claim=top10_claim,
        node=top10_leaf,
        sources=info.ranking_source_url,
        additional_instruction="If the ranking list shows numbered positions, ensure the institution's position is 10 or better. "
                               "Accept national or global CS subject top-10. Reject if the page is not a ranking page or not CS-specific."
    )

    # CS PhD program confirmed (by any official program URL if available; otherwise fallback to answer with simple verify)
    csphd_leaf = evaluator.add_leaf(
        id="CS_PhD_Program_Confirmed",
        desc="The identified institution offers a PhD program in Computer Science (or equivalent CS doctoral program).",
        parent=node,
        critical=True
    )
    program_urls = _collect_official_program_urls(info)
    csphd_claim = (
        f"This page describes that {uni_name} offers a PhD program in Computer Science (or a closely named CS doctoral program). "
        f"Accept synonyms such as 'Computer Science and Engineering' or 'Computer Science PhD'."
    )
    await evaluator.verify(
        claim=csphd_claim,
        node=csphd_leaf,
        sources=program_urls if program_urls else None,
        additional_instruction="Check the page text for phrases like 'PhD in Computer Science', 'Doctoral program in Computer Science', "
                               "'Computer Science (PhD)', or similar. Prefer official department/graduate-school pages."
    )


async def verify_exam_milestone(
    evaluator: Evaluator,
    parent_node,
    info: ProgramInfoExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Qualifying_or_Comprehensive_Exam",
        desc="Document the qualifying/comprehensive exam requirement and format with an official URL.",
        parent=parent_node,
        critical=True
    )

    # Exam requirement stated (answer-level)
    exam_req_leaf = evaluator.add_leaf(
        id="Exam_Requirement_Stated",
        desc="States the qualifying/comprehensive exam (or equivalent) requirement for the program.",
        parent=node,
        critical=True
    )
    exam_req_claim = (
        "The answer explicitly states that the Computer Science PhD program requires a qualifying, comprehensive, "
        "or preliminary examination (or equivalent)."
    )
    await evaluator.verify(
        claim=exam_req_claim,
        node=exam_req_leaf,
        additional_instruction="Look for terms like 'qualifying exam', 'comprehensive exam', 'prelim', 'preliminary exam', or 'qualifier'."
    )

    # Exam format stated (answer-level)
    exam_format_leaf = evaluator.add_leaf(
        id="Exam_Format_Stated",
        desc="Specifies the exam format (written, oral, or both).",
        parent=node,
        critical=True
    )
    exam_format_claim = (
        "The answer specifies the format of the qualifying/comprehensive exam as written, oral, or both "
        "(even if phrased in natural sentences)."
    )
    await evaluator.verify(
        claim=exam_format_claim,
        node=exam_format_leaf,
        additional_instruction="Accept equivalent phrasing indicating written and/or oral components."
    )

    # Exam official URL provided (page-level)
    exam_url_leaf = evaluator.add_leaf(
        id="Exam_Official_URL_Provided",
        desc="Provides a direct URL to an official university/department/graduate-school page describing the exam requirement.",
        parent=node,
        critical=True
    )
    exam_url_claim = (
        "This page is an official university/department/graduate-school page that describes the PhD qualifying/comprehensive "
        "exam requirement for the Computer Science program."
    )
    await evaluator.verify(
        claim=exam_url_claim,
        node=exam_url_leaf,
        sources=info.exam_url,
        additional_instruction="Prefer .edu domains or clearly official university subdomains. The page should mention the exam requirement or structure. "
                               "Reject third-party/non-official sites."
    )


async def verify_candidacy_milestone(
    evaluator: Evaluator,
    parent_node,
    info: ProgramInfoExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Candidacy_Advancement_Timeline",
        desc="Document the timeline/deadline to advance to candidacy with an official URL.",
        parent=parent_node,
        critical=True
    )

    # Candidacy timeline stated (answer-level)
    timeline_leaf = evaluator.add_leaf(
        id="Candidacy_Timeline_or_Deadline_Stated",
        desc="Provides the specific timeline/deadline by which students must advance to candidacy.",
        parent=node,
        critical=True
    )
    timeline_claim = (
        "The answer provides a specific timeline or deadline by which students must advance to PhD candidacy "
        "(for example, 'by the end of the second year' or 'within X semesters')."
    )
    await evaluator.verify(
        claim=timeline_claim,
        node=timeline_leaf,
        additional_instruction="General references like 'early in the program' are insufficient; it should indicate a concrete time window or deadline."
    )

    # Candidacy timeline official URL provided (page-level)
    timeline_url_leaf = evaluator.add_leaf(
        id="Candidacy_Timeline_Official_URL_Provided",
        desc="Provides a direct URL to an official university/department/graduate-school page specifying the candidacy timeline/deadline.",
        parent=node,
        critical=True
    )
    timeline_url_claim = (
        "This page specifies the timeline or deadline to advance to PhD candidacy for the Computer Science program."
    )
    await evaluator.verify(
        claim=timeline_url_claim,
        node=timeline_url_leaf,
        sources=info.candidacy_url,
        additional_instruction="Prefer official department/graduate-school pages. The page should clearly state the candidacy timing requirement."
    )


async def verify_dissertation_defense_milestone(
    evaluator: Evaluator,
    parent_node,
    info: ProgramInfoExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Dissertation_and_Defense",
        desc="Confirm dissertation requirement and oral defense with an official URL.",
        parent=parent_node,
        critical=True
    )

    # Dissertation required (answer-level)
    diss_leaf = evaluator.add_leaf(
        id="Dissertation_Required",
        desc="Confirms an original research dissertation is required for the PhD.",
        parent=node,
        critical=True
    )
    diss_claim = "The answer confirms that an original research dissertation is required for the PhD."
    await evaluator.verify(
        claim=diss_claim,
        node=diss_leaf,
        additional_instruction="Accept equivalent statements like 'doctoral dissertation is required' or 'thesis (PhD) required'."
    )

    # Oral defense required (answer-level)
    oral_leaf = evaluator.add_leaf(
        id="Oral_Defense_or_Oral_Examination_Required",
        desc="Confirms an oral examination/defense component is required.",
        parent=node,
        critical=True
    )
    oral_claim = "The answer confirms that an oral examination or dissertation defense is required."
    await evaluator.verify(
        claim=oral_claim,
        node=oral_leaf,
        additional_instruction="Accept synonyms like 'oral defense', 'final oral examination', or 'thesis defense'."
    )

    # Defense official URL provided (page-level)
    defense_url_leaf = evaluator.add_leaf(
        id="Defense_Official_URL_Provided",
        desc="Provides a direct URL to an official university/department/graduate-school page describing dissertation/defense requirements.",
        parent=node,
        critical=True
    )
    defense_url_claim = (
        "This page describes dissertation requirements and/or the oral defense requirement for the Computer Science PhD program."
    )
    await evaluator.verify(
        claim=defense_url_claim,
        node=defense_url_leaf,
        sources=info.defense_url,
        additional_instruction="Prefer official department/graduate-school pages. The page should mention dissertation and/or oral defense."
    )


async def verify_research_credits_milestone(
    evaluator: Evaluator,
    parent_node,
    info: ProgramInfoExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Research_or_Dissertation_Credits",
        desc="Document research/dissertation credits beyond coursework and provide the complete requirements URL.",
        parent=parent_node,
        critical=True
    )

    # Research credits beyond coursework confirmed (answer-level)
    rc_leaf = evaluator.add_leaf(
        id="Research_Credits_Beyond_Coursework_Confirmed",
        desc="Confirms the program requires research/dissertation credit hours (or equivalent registration units) beyond coursework.",
        parent=node,
        critical=True
    )
    rc_claim = (
        "The answer confirms that students must complete research/dissertation credits (or equivalent registration units) "
        "beyond regular coursework requirements."
    )
    await evaluator.verify(
        claim=rc_claim,
        node=rc_leaf,
        additional_instruction="Look for phrases like 'thesis/dissertation credits', 'research credits', 'doctoral research registration', etc."
    )

    # Minimum credit hours handled (answer-level)
    min_leaf = evaluator.add_leaf(
        id="Minimum_Credit_Hours_Handled",
        desc="Provides the minimum required research/dissertation credit hours if specified; otherwise explicitly states that a minimum is not specified on the cited official page(s).",
        parent=node,
        critical=True
    )
    min_claim = (
        "The answer either (a) provides a specific minimum number of research/dissertation credits required, "
        "or (b) explicitly states that a minimum is not specified."
    )
    await evaluator.verify(
        claim=min_claim,
        node=min_leaf,
        additional_instruction="Accept explicit numeric minimums (e.g., '24 credits'), or explicit text that no minimum is specified. "
                               "Reject if neither is present."
    )

    # Complete program requirements URL provided (page-level)
    comp_url_leaf = evaluator.add_leaf(
        id="Complete_Program_Requirements_URL_Provided",
        desc="Provides a direct URL to an official page that lists the complete PhD program requirements.",
        parent=node,
        critical=True
    )
    comp_url_claim = (
        "This page lists the complete set of PhD program requirements for the Computer Science program (coursework, exams, candidacy, dissertation/defense, etc.)."
    )
    await evaluator.verify(
        claim=comp_url_claim,
        node=comp_url_leaf,
        sources=info.program_requirements_url,
        additional_instruction="Prefer official department/graduate-school pages. The page should present an overview of requirements, not an unrelated page."
    )


async def verify_milestones_sequence(
    evaluator: Evaluator,
    parent_node,
) -> None:
    """
    Check that the answer presents the milestones in the requested order:
    exam → candidacy timeline → dissertation/oral defense → research/dissertation credits
    This is an answer-level simple verification.
    Note: Due to critical-parent constraints in the framework, we mark this as critical=True.
    """
    seq_leaf = evaluator.add_leaf(
        id="Milestones_Presented_as_Sequence",
        desc="Presents the milestones in the requested order (exam → candidacy timeline → dissertation/oral defense → research/dissertation credits).",
        parent=parent_node,
        critical=True  # Must be critical to satisfy framework constraints under a critical parent
    )
    seq_claim = (
        "The answer presents the milestones in the order: "
        "1) qualifying/comprehensive exam, "
        "2) candidacy advancement timeline, "
        "3) dissertation/defense with oral examination, "
        "4) research/dissertation credits."
    )
    await evaluator.verify(
        claim=seq_claim,
        node=seq_leaf,
        additional_instruction="Assess the ordering of the sections/content in the answer only; ignore minor formatting or headings."
    )


async def verify_program_milestones(
    evaluator: Evaluator,
    parent_node,
    info: ProgramInfoExtraction
) -> None:
    """
    Build and verify 'Program_Milestones_Documentation' subtree containing:
    - Qualifying/Comprehensive Exam
    - Candidacy Advancement Timeline
    - Dissertation and Defense
    - Research/Dissertation Credits
    - Milestones order
    """
    node = evaluator.add_parallel(
        id="Program_Milestones_Documentation",
        desc="Provide the four requested milestones (exam, candidacy timeline, defense/oral exam, research/dissertation credits) each with descriptions and official URLs.",
        parent=parent_node,
        critical=True
    )

    await verify_exam_milestone(evaluator, node, info)
    await verify_candidacy_milestone(evaluator, node, info)
    await verify_dissertation_defense_milestone(evaluator, node, info)
    await verify_research_credits_milestone(evaluator, node, info)
    await verify_milestones_sequence(evaluator, node)


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
    Entry point to evaluate an answer for the PhD program requirements task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator (non-critical by design in framework)
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

    # Extract structured information from the answer
    info: ProgramInfoExtraction = await evaluator.extract(
        prompt=prompt_extract_program_info(),
        template_class=ProgramInfoExtraction,
        extraction_name="program_info_extraction"
    )

    # Create task node as critical sequential aggregator to match rubric
    task_node = evaluator.add_sequential(
        id="PhD_Program_Requirements_Task",
        desc="Identify a top-10-ranked Computer Science PhD program and document required degree milestones with verifiable URLs.",
        parent=root,
        critical=True
    )

    # Phase 1: Program Identification
    await verify_program_identification(evaluator, task_node, info)

    # Phase 2: Program Milestones Documentation
    await verify_program_milestones(evaluator, task_node, info)

    # Return the full evaluation summary (includes verification tree and extraction info)
    return evaluator.get_summary()