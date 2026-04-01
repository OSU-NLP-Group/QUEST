import asyncio
import logging
from typing import Dict, List, Optional

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "han_alumni_academia"
TASK_DESCRIPTION = """
I am interested in data mining and one of the well-known professors in this field is Prof. Jiawei Han. I hope to apply to universities in California. Please find three of his former PhD students who is now a faculty in California. For each, please include their names, current affiliation, a link to their homepages and Google Scholar citation count. 
"""

ALUMNI_SOURCE_URL = "http://dm1.cs.uiuc.edu/alumni.html"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                     #
# --------------------------------------------------------------------------- #
class AlumniInfo(BaseModel):
    """Model for each individual alumni's information"""
    name: Optional[str] = None
    affiliation: Optional[str] = None
    homepage_url: Optional[str] = None
    citation_count: Optional[str] = None


class AlumniList(BaseModel):
    """Model for the list of alumni"""
    alumni: List[AlumniInfo] = Field(default_factory=list)


class URLList(BaseModel):
    """Simple model for extracting a list of URLs"""
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_alumni_list() -> str:
    return """
    From the answer, extract a list of Professor Jiawei Han's former PhD students who are now faculty members in California.
    For each alumnus, extract:
    1. Their full name
    2. Their current affiliation (university or institution in California)
    3. URL to their homepage
    4. Their Google Scholar citation count
    
    The extraction should be structured, with each alumnus as a separate entry.
    If the answer mentions more than 3 alumni, extract all of them.
    If any field is not provided for an alumnus, return null for that field.
    """


def prompt_extract_faculty_urls(name: str, affiliation: str) -> str:
    return f"""
    From the answer, extract ALL URLs that might help verify that {name} is:
    1. A former PhD student of Prof. Jiawei Han
    2. Currently a faculty member at {affiliation} in California
    
    This includes homepage URLs, Google Scholar profiles, department pages, or any other relevant links mentioned in relation to {name}.
    
    Only extract URLs that are explicitly present in the answer text.
    """


def prompt_extract_citation_urls(name: str) -> str:
    return f"""
    From the answer, extract ALL URLs that might help verify the Google Scholar citation count for {name}.
    This includes Google Scholar profiles, academic profile pages, or any other URLs that might contain citation information.
    
    Only extract URLs that are explicitly present in the answer text.
    """


# --------------------------------------------------------------------------- #
# Alumnus verification function                                               #
# --------------------------------------------------------------------------- #
async def verify_alumnus(
    evaluator: Evaluator,
    parent_node,
    alumni_info: AlumniInfo,
    alumni_index: int,
) -> None:
    """
    Verify all aspects of an individual alumnus entry with a flat structure.
    """
    alumnus_node = evaluator.add_parallel(
        id=f"alumnus_{alumni_index}",
        desc=f"Verification for alumnus {alumni_index + 1}: {alumni_info.name if alumni_info.name else 'Unknown'}",
        parent=parent_node,
        critical=False,
    )
    
    # Extract URLs for faculty verification
    faculty_url_list = await evaluator.extract(
        prompt=prompt_extract_faculty_urls(alumni_info.name or "Unknown", alumni_info.affiliation or "Unknown"),
        template_class=URLList,
        extraction_name=f"alumni_{alumni_index}_faculty_urls",
    )
    
    # Extract URLs for citation verification
    citation_url_list = await evaluator.extract(
        prompt=prompt_extract_citation_urls(alumni_info.name or "Unknown"),
        template_class=URLList,
        extraction_name=f"alumni_{alumni_index}_citation_urls",
    )
    
    # Add homepage URL to faculty URLs if not already included
    faculty_urls = faculty_url_list.urls.copy()
    if alumni_info.homepage_url and alumni_info.homepage_url not in faculty_urls:
        faculty_urls.append(alumni_info.homepage_url)
    
    # 1. Comprehensive completeness check (gates all other verifications)
    completeness_check = evaluator.add_custom_node(
        result=bool(
            alumni_info.name and 
            alumni_info.affiliation and 
            alumni_info.homepage_url and 
            alumni_info.citation_count and
            faculty_urls and
            citation_url_list.urls
        ),
        id=f"alumni_{alumni_index}_completeness_check",
        desc=f"Check if all required information is provided for alumnus {alumni_index + 1}",
        parent=alumnus_node,
        critical=True
    )
    
    # 2. Verify former Han PhD student
    han_student_node = evaluator.add_leaf(
        id=f"alumni_{alumni_index}_former_han_student",
        desc=f"Verify {alumni_info.name} is a former PhD student of Prof. Jiawei Han",
        parent=alumnus_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"{alumni_info.name} is a former PhD student of Prof. Jiawei Han",
        node=han_student_node,
        sources=ALUMNI_SOURCE_URL,
    )
    
    # 3. Verify California faculty status
    california_faculty_node = evaluator.add_leaf(
        id=f"alumni_{alumni_index}_california_faculty",
        desc=f"Verify {alumni_info.name} is faculty at {alumni_info.affiliation} in California",
        parent=alumnus_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"{alumni_info.name} is currently a faculty member at {alumni_info.affiliation} in California",
        node=california_faculty_node,
        sources=faculty_urls,
        additional_instruction="Verify both that the institution is in California and that the person holds a faculty position there.",
    )
    
    # 4. Verify homepage URL
    homepage_node = evaluator.add_leaf(
        id=f"alumni_{alumni_index}_homepage_url",
        desc=f"Verify homepage URL belongs to {alumni_info.name}",
        parent=alumnus_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"This webpage is the homepage or faculty profile of {alumni_info.name}",
        node=homepage_node,
        sources=alumni_info.homepage_url,
    )
    
    # 5. Verify citation count
    citation_node = evaluator.add_leaf(
        id=f"alumni_{alumni_index}_citation_count",
        desc=f"Verify citation count for {alumni_info.name}",
        parent=alumnus_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"{alumni_info.name} has approximately {alumni_info.citation_count} citations on Google Scholar",
        node=citation_node,
        sources=citation_url_list.urls,
        additional_instruction="The citation count doesn't need to be exact, as numbers may change over time. Verify that it's in the general range mentioned.",
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
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator with parallel strategy for root
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
        default_model=model
    )

    # -------- 2. Extract structured alumni information ------------------ #
    alumni_list = await evaluator.extract(
        prompt=prompt_extract_alumni_list(),
        template_class=AlumniList,
        extraction_name="alumni_list",
    )
    
    # Limit to first 3 alumni for evaluation (as per task requirements)
    alumni_for_evaluation = alumni_list.alumni[:3] if alumni_list.alumni else []
    
    # If fewer than 3 alumni were provided, pad the list with empty entries
    while len(alumni_for_evaluation) < 3:
        alumni_for_evaluation.append(AlumniInfo())

    # -------- 3. Build verification tree -------------------------------- #
    # Verify each alumnus (all directly under root)
    for i, alumni_info in enumerate(alumni_for_evaluation):
        await verify_alumnus(evaluator, root, alumni_info, i)
    
    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()