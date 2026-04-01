import asyncio
import logging
from typing import Optional, List, Dict

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nlp_faculty"
TASK_DESCRIPTION = """
I am interested in applying to PhD programs in Computer Science and want to connect with faculty members whose primary research area is natural language processing. Please help me gather information on five professors, each from a different university ranked between 20 and 30 in the Computer Science Open Rankings (with all fields selected). For each professor, include their affiliated institution and a link to either their personal webpage or their official faculty profile hosted by the university.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Professor(BaseModel):
    name: Optional[str] = None
    university: Optional[str] = None
    personal_website: Optional[str] = None # Main personal webpage or faculty profile
    all_urls: List[str] = Field(default_factory=list)  # All related URLs


class ExtractedProfessors(BaseModel):
    professors: List[Professor] = Field(default_factory=list)


class RankedUniversity(BaseModel):
    name: Optional[str] = None
    rank: Optional[int] = None


class RankedUniversities(BaseModel):
    universities: List[RankedUniversity] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_professors() -> str:
    return """
    Extract information about professors mentioned in the answer. For each professor, extract:
    1. Their name
    2. Their affiliated university/institution
    3. Their main personal webpage or faculty profile URL (the primary link provided for them)
    4. ALL other URLs mentioned in relation to this professor (such as publication pages, lab pages, research group pages, etc.)

    If any information is missing, set the corresponding field to null or empty list for URLs.

    Return the information in the specified JSON format with a list of professors.
    """


def prompt_extract_universities_ranking() -> str:
    return """
    Extract universities ranked between 20 and 30 (inclusive) from the Computer Science Open Rankings. 
    Include universities that are tied at rank 20 or 30. For each university, extract:
    1. The full name of the university
    2. Its numerical rank

    Return the information in the specified JSON format with a list of ranked universities.
    """


# --------------------------------------------------------------------------- #
# Professor verification functions                                            #
# --------------------------------------------------------------------------- #
async def verify_professor(
        evaluator: Evaluator,
        parent_node,
        professor: Professor,
        ranked_university_names: List[str],
        index: int
):
    """
    Verify all aspects of a professor:
    1. Existence of professor data
    2. Valid personal website
    3. University affiliation confirmation
    4. NLP research focus
    5. University ranking (20-30)
    """
    # Create a parallel parent node for this professor
    prof_node = evaluator.add_parallel(
        id=f"professor_{index}",
        desc=f"Verify all requirements for professor {index + 1}: {professor.name or 'N/A'} from {professor.university or 'N/A'}",
        parent=parent_node,
        critical=False
    )

    # Check if professor data exists
    professor_exists = bool(professor.name) and bool(professor.university) and bool(professor.personal_website)
    
    existence_node = evaluator.add_custom_node(
        result=professor_exists,
        id=f"professor_{index}_exists",
        desc=f"Professor {index + 1} data exists (name, university, and website provided)",
        parent=prof_node,
        critical=True
    )


    # Step 1: Verify personal website is valid and belongs to the professor
    website_node = evaluator.add_leaf(
        id=f"professor_{index}_website",
        desc=f"Verify that the provided URL is a valid personal webpage or faculty profile for {professor.name or 'the professor'}",
        parent=prof_node,
        critical=True,
    )

    website_claim = f"The website at {professor.personal_website} is a personal webpage or faculty profile of a computer science professor named {professor.name}."
    await evaluator.verify(
        claim=website_claim,
        node=website_node,
        sources=professor.personal_website,
        additional_instruction="Check if this is a personal webpage or faculty profile of a computer science professor. Look for academic titles, research descriptions, publications, or other indicators that this belongs to a CS faculty member with the specified name."
    )

    # Step 2: Verify university affiliation using all available URLs
    affiliation_node = evaluator.add_leaf(
        id=f"professor_{index}_affiliation",
        desc=f"Verify that {professor.name or 'the professor'} is affiliated with {professor.university or 'the claimed university'}",
        parent=prof_node,
        critical=True,
    )

    # Collect all URLs for verification
    all_prof_urls = []
    if professor.personal_website:
        all_prof_urls.append(professor.personal_website)
    all_prof_urls.extend(professor.all_urls)
    # Filter out None values and duplicates
    all_prof_urls = list(set([url for url in all_prof_urls if url]))

    affiliation_claim = f"Professor {professor.name} is affiliated with {professor.university}"
    await evaluator.verify(
        claim=affiliation_claim,
        node=affiliation_node,
        sources=all_prof_urls,
        additional_instruction="Check if the professor's affiliation matches the claimed university. Look for the university name, logo, or institutional affiliation in the professor's profile, contact information, or about sections."
    )

    # Step 3: Verify professor's NLP research focus using all available URLs
    nlp_node = evaluator.add_leaf(
        id=f"professor_{index}_nlp_focus",
        desc=f"Verify that {professor.name or 'the professor'}'s primary research area includes natural language processing",
        parent=prof_node,
        critical=True,
    )

    nlp_claim = f"Professor {professor.name} has natural language processing (NLP) as one of their primary research areas"
    await evaluator.verify(
        claim=nlp_claim,
        node=nlp_node,
        sources=all_prof_urls,
        additional_instruction="Look for mentions of natural language processing, NLP, computational linguistics, language models, text processing, machine translation, or related topics in the broad NLP in the professor's research interests."
    )

    # Step 4: Verify university ranking (20-30) using simple verification
    ranking_node = evaluator.add_leaf(
        id=f"professor_{index}_university_ranked",
        desc=f"Verify that {professor.university or 'the university'} is ranked between 20-30 in CS Open Rankings",
        parent=prof_node,
        critical=True,
    )

    ranking_claim = f"The university '{professor.university}' is in this list of universities: {', '.join(ranked_university_names)}."
    await evaluator.verify(
        claim=ranking_claim,
        node=ranking_node,
        sources=None,  # Simple verification without URL
        additional_instruction=f"Minor variations in how university names or abbreviations are written are acceptable (e.g., 'UC Berkeley' vs 'University of California, Berkeley' vs 'UCB')"
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer to the 'find_nlp_faculty' task.

    This evaluates whether the answer provides five professors from different universities
    ranked 20-30 in CS Open Rankings, who specialize in NLP, with valid website links.
    """
    # -------- 1. Initialize evaluator ------------------------------------ #
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
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

    # -------- 2. Extract professors from the answer ---------------------- #
    professors_info = await evaluator.extract(
        prompt=prompt_extract_professors(),
        template_class=ExtractedProfessors,
        extraction_name="professors_extraction"
    )

    # -------- 3. Extract universities ranked 20-30 from CS Open Rankings -- #
    ranked_universities_info = await evaluator.extract(
        prompt=prompt_extract_universities_ranking(),
        template_class=RankedUniversities,
        extraction_name="ranked_universities",
        source="https://drafty.cs.brown.edu/csopenrankings/"
    )

    # Extract university names for easier verification
    ranked_university_names = [
        univ.name for univ in ranked_universities_info.universities
        if univ.name and univ.rank and 20 <= univ.rank <= 30
    ]

    # -------- 4. Deduplicate professors by university -------------------- #
    deduplicated_professors = []
    seen_universities = set()

    for prof in professors_info.professors:
        if prof.university:
            univ_key = prof.university.lower().strip()
            if univ_key not in seen_universities:
                seen_universities.add(univ_key)
                deduplicated_professors.append(prof)
        else:
            # Include professors with missing university info for verification
            # (they will likely fail the university verification step)
            deduplicated_professors.append(prof)

    # Take only the first 5 professors (if more than 5 provided)
    required_professors = 5
    professors_to_verify = deduplicated_professors[:required_professors]

    # Add placeholder professors if fewer than 5 were provided
    while len(professors_to_verify) < required_professors:
        professors_to_verify.append(Professor(
            name=None,
            university=None,
            personal_website=None,
            all_urls=[]
        ))

    # -------- 5. Add custom info to the summary ------------------------- #
    evaluator.add_custom_info(
        {
            "deduplicated_professors": [p.dict() for p in deduplicated_professors],
            "original_count": len(professors_info.professors),
            "deduplicated_count": len(deduplicated_professors),
        },
        "deduplication_info"
    )
    
    evaluator.add_custom_info(
        {"ranked_university_names": ranked_university_names},
        "ranking_info"
    )

    # -------- 6. Build verification tree --------------------------------- #
    # Verify each professor
    for i, professor in enumerate(professors_to_verify):
        await verify_professor(
            evaluator,
            evaluator.root,
            professor,
            ranked_university_names,
            i
        )

    # -------- 7. Return the summary result ------------------------------- #
    return evaluator.get_summary()