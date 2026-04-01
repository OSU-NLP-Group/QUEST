import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs_master_toefl"
TASK_DESCRIPTION = """
Find out the top 10 best grad schools for computer science in the US according to the most recent US News ranking. For each one. please list its ranking, a direct link to to the official page that mentions the TOEFL requirements for admission to its Computer Science master's program (not third-party sources).
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Program(BaseModel):
    """Model for an individual CS master's program."""
    school_name: Optional[str] = None
    program_name: Optional[str] = None
    ranking: Optional[str] = None
    toefl_requirement: Optional[str] = None
    toefl_page_urls: List[str] = Field(default_factory=list)


class ProgramsList(BaseModel):
    """Model for the list of CS master's programs."""
    programs: List[Program] = Field(default_factory=list)


class RankingLinks(BaseModel):
    """Model for extracting potential ranking source URLs."""
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract the list of computer science master's programs mentioned in the answer. 

    For each program, extract:
    1. The school name (university name)
    2. The program name (e.g., "Computer Science", "CS", etc.)
    3. The ranking position (as a string, e.g. "1", "2", "3-way tie for 4th", etc.)
    4. The TOEFL requirement(s) mentioned (as a string)
    5. Any URLs to the official TOEFL requirements pages (as a list of strings)

    It's okay if either the school name or program name is missing, but at least one should be present.

    Return all programs mentioned in the answer.
    """


def prompt_extract_ranking_links(program_info: str) -> str:
    return f"""
    For the following computer science program information:

    {program_info}

    Extract any URLs mentioned in the answer that might contain or support the US News ranking information.
    Return these as a list of URLs. If no such URLs are found, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_school(
        evaluator: Evaluator,
        parent_node,
        program: Program,
        index: int,
) -> None:
    """
    Verify a single school's information, including ranking and TOEFL requirements.
    Uses a sequential structure where ranking verification must pass before TOEFL verification.
    """
    # Get program display name for verification
    program_display = f"{program.school_name} - {program.program_name}" if program.school_name and program.program_name else None
    if program_display is None:
        program_display = program.school_name or program.program_name or f"Program #{index + 1}"
    
    # Create a sequential node for this school
    school_node = evaluator.add_sequential(
        id=f"school_{index + 1}",
        desc=f"School #{index + 1}: {program_display} - Verification of ranking and TOEFL requirements",
        parent=parent_node
    )

    info_exists = evaluator.add_custom_node(
        result=bool(program.school_name or program.program_name),
        id=f"school_{index + 1}_info_exists",
        desc=f"School #{index + 1}: Program info are provided",
        parent=school_node,
        critical=True
    )

    # For this program, extract any potential ranking links
    program_info = f"School: {program.school_name}, Program: {program.program_name}, Ranking: {program.ranking}"
    ranking_links = await evaluator.extract(
        prompt=prompt_extract_ranking_links(program_info),
        template_class=RankingLinks,
        extraction_name=f"ranking_links_school_{index + 1}",
        additional_instruction="the program information is extracted from the answer. Plz further extract the related URLs to the info, that may potentially support the ranking information."
    )

    # Step 1: Verify ranking information
    ranking_node = evaluator.add_parallel(
        id=f"school_{index + 1}_ranking",
        desc=f"School #{index + 1}: {program_display} - Ranking verification",
        parent=school_node,
        critical=False
    )

    # 1.1: Check if basic info exists
    ranking_info_exists = evaluator.add_custom_node(
        result=bool(program.ranking),
        id=f"school_{index + 1}_ranking_info_exists",
        desc=f"School #{index + 1}: Program info and ranking are provided",
        parent=ranking_node,
        critical=True
    )

    # 1.2: Simple verify - Is ranking in top 10?
    ranking_top10_node = evaluator.add_leaf(
        id=f"school_{index + 1}_ranking_top10",
        desc=f"School #{index + 1}: {program_display} has a ranking in the top 10",
        parent=ranking_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The ranking '{program.ranking}' indicates it is among the top 10, and the program information {program_display} indicates it's a program related to computer science.",
        node=ranking_top10_node,
        additional_instruction="Accept any ranking indication that clearly places the program in the top 10, whether expressed as a number (1-10) or as a description ('tied for 4th', etc.)"
    )

    # 1.3: URL verify - Ranking is substantiated by URL
    ranking_supported_node = evaluator.add_leaf(
        id=f"school_{index + 1}_ranking_supported",
        desc=f"School #{index + 1}: {program_display}'s ranking ({program.ranking}) is supported by a URL source",
        parent=ranking_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The computer science program at {program_display} is ranked {program.ranking} according to US News",
        node=ranking_supported_node,
        sources=ranking_links.urls,
        additional_instruction="The ranking might be expressed in different formats (e.g., '#1', 'number 1', 'ranked 1st', etc.). Consider these equivalent."
    )

    # Step 2: Verify TOEFL requirements
    toefl_node = evaluator.add_parallel(
        id=f"school_{index + 1}_toefl",
        desc=f"School #{index + 1}: {program_display} - TOEFL requirements verification",
        parent=school_node,
        critical=False
    )

    # 2.1: Check if TOEFL URLs exist
    toefl_urls_exist = evaluator.add_custom_node(
        result=bool(program.toefl_page_urls),
        id=f"school_{index + 1}_toefl_urls_exist",
        desc=f"School #{index + 1}: TOEFL requirement URLs are provided",
        parent=toefl_node,
        critical=True
    )

    # 2.2: URL verify - TOEFL requirement is substantiated by URL
    toefl_supported_node = evaluator.add_leaf(
        id=f"school_{index + 1}_toefl_supported",
        desc=f"School #{index + 1}: {program_display}'s TOEFL requirement is supported by a URL source",
        parent=toefl_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The page has information related to TOEFL or English Proficiency requirement for {program_display}'s computer science program, or the general toefl/english proficiency requirement from this univeristy/college/school",
        node=toefl_supported_node,
        sources=program.toefl_page_urls,
        additional_instruction="We loosen the creteria for this because often the time, the information will be folded. So, as long as their are any thing related to TOEFL/english proficiency, for example, even a folded section titled with 'Toefl'/'TOEFL', or just toefl code, they should be considered as a pass" + """\n, or, another way to check this  is: 
            - The page must be an official page from the specific university.
            - The page clearly mentions TOEFL or English Proficiency (exact score details are not required).
            - The page should not clearly indicate that it's intended exclusively for another unrelated program (i.e., not Computer Science) (e.g., Medicine, Biology, Law).
If these three criteria are met, the provided link is considered acceptable. Do not impose any stricter evaluation than what's explicitly stated here."""
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                   #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer to the CS master's TOEFL requirements task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        task_description=TASK_DESCRIPTION,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
        extract_model=model,
        verify_model=model
    )

    # Extract the list of programs
    programs_info = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsList,
        extraction_name="programs_list"
    )

    # Get the list of programs (up to 10)
    extracted_programs = programs_info.programs

    # Ensure we have exactly 10 programs to evaluate (pad with empty ones if needed)
    programs = []
    for i in range(10):
        if i < len(extracted_programs):
            programs.append(extracted_programs[i])
        else:
            # Add empty Program instance for missing programs
            programs.append(Program(school_name=None, program_name=None, ranking=None, 
                                  toefl_requirement=None, toefl_page_urls=[]))

    # Verify each school individually
    for i, program in enumerate(programs):
        await verify_school(
            evaluator=evaluator,
            parent_node=evaluator.root,
            program=program,
            index=i,
        )

    # Return the standard summary
    return evaluator.get_summary()