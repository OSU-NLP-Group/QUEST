import asyncio
import logging
from typing import Optional, List, Dict, Any

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pritzker_prize_winner"
TASK_DESCRIPTION = """
Who is the most recent recipient of the Pritzker Prize? What is their educational background? Specifically, what is the highest degree they have earned, and from which institution? Please list two past architectural projects by this person. For each project, include its location, the year it was completed, and a link to an article or project page that contains photos and a description.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Project(BaseModel):
    """Model for an architectural project."""
    name: Optional[str] = None
    location: Optional[str] = None
    year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ProjectsList(BaseModel):
    """Model for a list of projects."""
    projects: List[Project] = Field(default_factory=list)


class LaureateBasicInfo(BaseModel):
    """Model for basic information about a Pritzker Prize laureate."""
    name: Optional[str] = None
    highest_degree: Optional[str] = None
    institution: Optional[str] = None
    links: List[str] = Field(default_factory=list)


class StandardLaureateInfo(BaseModel):
    """Model for extracting the standard answer from the Pritzker website."""
    name: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_standard_info() -> str:
    return """
    Extract the name of the most recent Pritzker Prize laureate from the official Pritzker Prize website.

    Focus only on the most recent laureate (the one at the top of the list or most prominently featured).
    If the name is not available on the webpage, set the corresponding field to null.
    """


def prompt_extract_laureate_basic_info() -> str:
    return """
    Extract the following information about the Pritzker Prize laureate from the answer:
    1. The name of the most recent Pritzker Prize recipient.
    2. Their highest academic degree (e.g., Bachelor's, Master's, PhD, etc.).
    3. The educational institution they received this degree from.
    4. Any source URLs or links mentioned in the answer that might support any of the above information.

    In the JSON response:
    - For the name, extract the full name as provided in the answer.
    - For highest_degree, specify the exact degree title/type mentioned.
    - For institution, provide the complete name of the educational institution.
    - For links, include all URLs that might validate any of the above information.

    If any information is missing, set the corresponding field to null.
    """


def prompt_extract_projects() -> str:
    return """
    Extract information about the architectural projects mentioned in the answer.

    For each project, extract:
    1. The name of the project.
    2. The location of the project.
    3. The year the project was completed.
    4. Any URLs or links provided for the project.

    In the JSON response:
    - For urls, include all URLs mentioned that might relate to the project, even if multiple URLs are mentioned for a single project.

    The task asks for two projects, but extract all projects mentioned in the answer.

    If any information about a project is missing, set the corresponding field to null.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_laureate_identity(
        evaluator: Evaluator,
        parent_node,
        laureate_info: LaureateBasicInfo,
        standard_info: StandardLaureateInfo,
) -> None:
    """
    Verify the identity of the Pritzker Prize laureate against the standard information.
    """
    identity_parent = evaluator.add_parallel(
        id="laureate_identity",
        desc="Laureate Identity Verification",
        parent=parent_node,
        critical=True,
    )

    # Existence check
    existence_check = evaluator.add_custom_node(
        result=(laureate_info.name is not None and laureate_info.name.strip() != "") and 
               (standard_info.name is not None and standard_info.name.strip() != ""),
        id="laureate_names_exist",
        desc="Check if both laureate name and standard name are provided",
        parent=identity_parent,
        critical=True
    )

    # Identity verification
    identity_node = evaluator.add_leaf(
        id="laureate_name_match",
        desc=f"The answer correctly identifies the most recent Pritzker Prize recipient",
        parent=identity_parent,
        critical=True,
    )

    claim = f"The name '{laureate_info.name}' is the same as or equivalent to '{standard_info.name}'."
    await evaluator.verify(
        claim=claim,
        node=identity_node,
        additional_instruction="Just check the name match. Allow minor variations in capitalization, punctuation, or formatting."
    )


async def verify_educational_info(
        evaluator: Evaluator,
        parent_node,
        laureate_info: LaureateBasicInfo,
) -> None:
    """
    Verify the educational information (highest degree and institution).
    """
    # Verify highest degree
    degree_parent = evaluator.add_parallel(
        id="degree_verification",
        desc="Highest Degree Verification",
        parent=parent_node,
        critical=False,
    )

    degree_exists = evaluator.add_custom_node(
        result=(laureate_info.highest_degree is not None and laureate_info.highest_degree.strip() != "") and
               (laureate_info.name is not None) and
               bool(laureate_info.links),
        id="degree_info_exists",
        desc="Check if degree information and supporting URLs are provided",
        parent=degree_parent,
        critical=True
    )

    degree_node = evaluator.add_leaf(
        id="degree_verification",
        desc=f"The highest degree is correctly stated",
        parent=degree_parent,
        critical=True,
    )

    degree_claim = f"{laureate_info.name}'s highest academic degree is {laureate_info.highest_degree}."
    await evaluator.verify(
        claim=degree_claim,
        node=degree_node,
        sources=laureate_info.links,
    )

    # Verify institution
    institution_parent = evaluator.add_parallel(
        id="institution_verification",
        desc="Institution Verification",
        parent=parent_node,
        critical=False,
    )

    institution_exists = evaluator.add_custom_node(
        result=(laureate_info.institution is not None and laureate_info.institution.strip() != "") and
               (laureate_info.name is not None) and
               bool(laureate_info.links),
        id="institution_info_exists",
        desc="Check if institution information and supporting URLs are provided",
        parent=institution_parent,
        critical=True
    )

    institution_node = evaluator.add_leaf(
        id="institution_verification",
        desc=f"The institution is correctly identified",
        parent=institution_parent,
        critical=True,
    )

    institution_claim = f"{laureate_info.name} earned their {laureate_info.highest_degree if laureate_info.highest_degree else 'highest degree'} from {laureate_info.institution}."
    await evaluator.verify(
        claim=institution_claim,
        node=institution_node,
        sources=laureate_info.links,
    )


async def verify_project(
        evaluator: Evaluator,
        parent_node,
        project: Project,
        project_index: int,
        laureate_name: Optional[str],
) -> None:
    """
    Verify details of an architectural project.
    """
    # Create a node for this specific project
    project_node = evaluator.add_sequential(
        id=f"project_{project_index}",
        desc=f"Project {project_index + 1}: {project.name if project.name else 'Unnamed project'}",
        parent=parent_node,
        critical=False,
    )

    # Project existence check (name and URLs must exist)
    project_exists = evaluator.add_custom_node(
        result=(project.name is not None and project.name.strip() != "") and
               bool(project.urls) and
               any(url.startswith(('http://', 'https://')) for url in project.urls),
        id=f"project_{project_index}_exists",
        desc=f"Check if project {project_index + 1} has name and valid URLs",
        parent=project_node,
        critical=True
    )

    # Verify project attribution
    attribution_node = evaluator.add_leaf(
        id=f"project_{project_index}_attribution",
        desc=f"Verify that the project is by the laureate",
        parent=project_node,
        critical=True,
    )

    attribution_claim = f"{project.name} is an architectural project by {laureate_name}."
    await evaluator.verify(
        claim=attribution_claim,
        node=attribution_node,
        sources=project.urls,
        additional_instruction="Try your best to find any possible indicators from the page. For example, many of the time, the full name may not be present, for example, if the page says 'jiakun architects' or 'liu's', it's enough to support the ownership 'liu jiakun'. I.e., don't be too strict on the full naming."
    )

    # Verify location
    location_parent = evaluator.add_parallel(
        id=f"project_{project_index}_location_wrapper",
        desc=f"Location verification for project {project_index + 1}",
        parent=project_node,
        critical=False,
    )

    location_exists = evaluator.add_custom_node(
        result=(project.location is not None and project.location.strip() != "") and bool(project.urls),
        id=f"project_{project_index}_location_exists",
        desc="Check if location and URLs are provided",
        parent=location_parent,
        critical=True
    )

    location_node = evaluator.add_leaf(
        id=f"project_{project_index}_location",
        desc=f"The location is correctly stated",
        parent=location_parent,
        critical=True,
    )

    location_claim = f"{project.name} is located in {project.location}."
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=project.urls,
    )

    # Verify year
    year_parent = evaluator.add_parallel(
        id=f"project_{project_index}_year_wrapper",
        desc=f"Year verification for project {project_index + 1}",
        parent=project_node,
        critical=False,
    )

    year_exists = evaluator.add_custom_node(
        result=(project.year is not None and project.year.strip() != "") and bool(project.urls),
        id=f"project_{project_index}_year_exists",
        desc="Check if year and URLs are provided",
        parent=year_parent,
        critical=True
    )

    year_node = evaluator.add_leaf(
        id=f"project_{project_index}_year",
        desc=f"The completion year is correctly stated",
        parent=year_parent,
        critical=True,
    )

    year_claim = f"{project.name} was completed in {project.year}."
    await evaluator.verify(
        claim=year_claim,
        node=year_node,
        sources=project.urls,
    )

    # Verify URL contains photos and description
    url_node = evaluator.add_leaf(
        id=f"project_{project_index}_url_content",
        desc=f"At least one URL contains photos and description",
        parent=project_node,
        critical=False,
    )

    url_claim = f"At least one of the provided URLs contains photos (at least one) and descriptions for the {project.name} project."
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=project.urls,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                   #
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
) -> Dict[str, Any]:
    """
    Evaluate a single answer and return a structured result dictionary.

    This evaluation script validates an answer to the task of identifying the most recent
    Pritzker Prize winner, their educational background, and two of their architectural projects.
    """
    # Set up evaluator
    evaluator = Evaluator()
    
    # Initialize evaluator with sequential strategy for root
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
        default_model=model if model else JUDGE_MODEL
    )

    # First, extract the standard information (name only) from the Pritzker Prize website
    standard_info = await evaluator.extract(
        prompt=prompt_extract_standard_info(),
        template_class=StandardLaureateInfo,
        extraction_name="standard_info",
        source="https://www.pritzkerprize.com/laureates",
    )

    # Extract basic laureate information from the answer
    laureate_info = await evaluator.extract(
        prompt=prompt_extract_laureate_basic_info(),
        template_class=LaureateBasicInfo,
        extraction_name="laureate_info",
    )

    # Extract projects
    projects_result = await evaluator.extract(
        prompt=prompt_extract_projects(),
        template_class=ProjectsList,
        extraction_name="projects",
    )

    # First: Verify laureate identity
    await verify_laureate_identity(
        evaluator,
        root,
        laureate_info,
        standard_info
    )

    # Second section: Verify educational information
    education_section = evaluator.add_parallel(
        id="education_section",
        desc="Educational Information Section",
        critical=False,
    )

    await verify_educational_info(
        evaluator,
        education_section,
        laureate_info
    )

    # Third section: Verify projects
    projects_section = evaluator.add_parallel(
        id="projects_section",
        desc="Architectural Projects Section",
        critical=False,
    )

    # Ensure we have exactly 2 projects for evaluation (pad with empty if needed)
    projects = projects_result.projects[:2]  # Take at most 2
    while len(projects) < 2:
        projects.append(Project())  # Add empty project

    # Verify each project
    for i, project in enumerate(projects):
        await verify_project(
            evaluator,
            projects_section,
            project,
            i,
            laureate_info.name,
        )

    # Return structured result
    return evaluator.get_summary()