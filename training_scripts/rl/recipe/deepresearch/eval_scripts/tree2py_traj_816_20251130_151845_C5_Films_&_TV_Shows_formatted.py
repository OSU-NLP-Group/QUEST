import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "actress_identification_task"
TASK_DESCRIPTION = (
    "Identify the actress who meets all of the following criteria:\n\n"
    "1. She was born in 1997\n"
    "2. She graduated from Arizona State University (ASU) in 2023 with honors\n"
    "3. Her degree was in Film and Media Studies\n"
    "4. She appears in the TV series \"1923\" playing the character Elizabeth Strafford\n"
    "5. She appears in the TV series \"Landman\" playing the character Ainsley Norris\n"
    "6. The series \"1923\" was filmed in Montana\n"
    "7. The series \"Landman\" was filmed in the Fort Worth, Texas area\n\n"
    "Provide the actress's full name along with reference URLs that verify each aspect of the information "
    "(birth year, education, TV series participation and characters, and filming locations)."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ActressExtraction(BaseModel):
    """Structured extraction of the actress info and all supporting URLs, strictly from the answer text."""
    full_name: Optional[str] = None

    # Birth year sources
    birth_year_sources: List[str] = Field(default_factory=list)

    # Education details and sources
    education_school: Optional[str] = None
    education_year: Optional[str] = None
    education_degree: Optional[str] = None
    education_honors: Optional[str] = None
    education_sources: List[str] = Field(default_factory=list)

    # Roles and sources
    role_1923_character: Optional[str] = None
    role_1923_sources: List[str] = Field(default_factory=list)

    role_landman_character: Optional[str] = None
    role_landman_sources: List[str] = Field(default_factory=list)

    # Filming locations sources
    filming_1923_sources: List[str] = Field(default_factory=list)
    filming_landman_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_actress_info() -> str:
    return (
        "You must extract the actress's identity and all aspect-specific reference URLs as explicitly provided in the answer.\n\n"
        "Return a JSON object with the following fields:\n"
        "- full_name: The actress's full name as stated in the answer (string or null).\n"
        "- birth_year_sources: Array of URLs that directly support the birth-year claim (extract only URLs explicitly present in the answer).\n"
        "- education_school: The school name as provided (string or null), e.g., 'Arizona State University' or 'ASU'.\n"
        "- education_year: The graduation year as provided (string or null), e.g., '2023'.\n"
        "- education_degree: The degree name as provided (string or null), e.g., 'Film and Media Studies'.\n"
        "- education_honors: The honors description as provided (string or null), e.g., 'with honors' or 'cum laude'.\n"
        "- education_sources: Array of URLs that support the education claims (ASU graduation, 2023, Film and Media Studies, honors).\n"
        "- role_1923_character: The character name in '1923' as provided (string or null), e.g., 'Elizabeth Strafford'.\n"
        "- role_1923_sources: Array of URLs that support the role in '1923'.\n"
        "- role_landman_character: The character name in 'Landman' as provided (string or null), e.g., 'Ainsley Norris'.\n"
        "- role_landman_sources: Array of URLs that support the role in 'Landman'.\n"
        "- filming_1923_sources: Array of URLs that support that '1923' was filmed in Montana.\n"
        "- filming_landman_sources: Array of URLs that support that 'Landman' was filmed in the Fort Worth, Texas area.\n\n"
        "Rules:\n"
        "1) Extract only what is explicitly present in the answer. Do not invent or infer missing info.\n"
        "2) For all URL arrays, include only valid URLs (plain links or markdown links). If none are present, return an empty array.\n"
        "3) If a string field is not present, return null for that field.\n"
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
def add_actress_name_node(evaluator: Evaluator, parent_node, extracted: ActressExtraction) -> None:
    # Actress_Full_Name (critical): existence check of full name
    evaluator.add_custom_node(
        result=bool(extracted.full_name and extracted.full_name.strip()),
        id="Actress_Full_Name",
        desc="Provide the actress's full name.",
        parent=parent_node,
        critical=True
    )


async def add_birth_year_nodes(evaluator: Evaluator, parent_node, extracted: ActressExtraction) -> None:
    # Birth_Year group (critical parallel)
    birth_group = evaluator.add_parallel(
        id="Birth_Year",
        desc="Birth year requirement with supporting reference(s).",
        parent=parent_node,
        critical=True
    )

    # Born_1997 (critical verification by URLs)
    born_1997_leaf = evaluator.add_leaf(
        id="Born_1997",
        desc="The actress was born in 1997.",
        parent=birth_group,
        critical=True
    )
    claim_born_1997 = f"{extracted.full_name or 'The actress'} was born in 1997."
    await evaluator.verify(
        claim=claim_born_1997,
        node=born_1997_leaf,
        sources=extracted.birth_year_sources,
        additional_instruction="Verify that the cited page(s) explicitly state the actress's birth year as 1997. "
                               "Allow simple phrasing variants like 'born in 1997' or a date that clearly falls in 1997."
    )

    # Birth_Year_Reference_URL (critical existence of at least one URL)
    evaluator.add_custom_node(
        result=len(extracted.birth_year_sources) > 0,
        id="Birth_Year_Reference_URL",
        desc="Provide at least one reference URL supporting the 1997 birth-year claim.",
        parent=birth_group,
        critical=True
    )


async def add_education_nodes(evaluator: Evaluator, parent_node, extracted: ActressExtraction) -> None:
    # Education group (critical parallel)
    edu_group = evaluator.add_parallel(
        id="Education",
        desc="Education requirements (ASU, 2023, honors, Film and Media Studies) with supporting reference(s).",
        parent=parent_node,
        critical=True
    )

    # Graduated_From_ASU
    grad_asu_leaf = evaluator.add_leaf(
        id="Graduated_From_ASU",
        desc="The actress graduated from Arizona State University (ASU).",
        parent=edu_group,
        critical=True
    )
    claim_asu = f"{extracted.full_name or 'The actress'} graduated from Arizona State University."
    await evaluator.verify(
        claim=claim_asu,
        node=grad_asu_leaf,
        sources=extracted.education_sources,
        additional_instruction="Verify that the sources explicitly state graduation from ASU. "
                               "Allow synonyms like 'Arizona State University' or 'ASU'."
    )

    # Graduated_In_2023
    grad_2023_leaf = evaluator.add_leaf(
        id="Graduated_In_2023",
        desc="The actress graduated in 2023.",
        parent=edu_group,
        critical=True
    )
    claim_2023 = f"{extracted.full_name or 'The actress'} graduated in 2023."
    await evaluator.verify(
        claim=claim_2023,
        node=grad_2023_leaf,
        sources=extracted.education_sources,
        additional_instruction="Check the graduation year stated in the sources. Accept reasonable variants such as "
                               "commencement in 2023 or 'Class of 2023'."
    )

    # Degree_Film_And_Media_Studies
    degree_fms_leaf = evaluator.add_leaf(
        id="Degree_Film_And_Media_Studies",
        desc="The actress’s degree was in Film and Media Studies.",
        parent=edu_group,
        critical=True
    )
    claim_degree = f"{extracted.full_name or 'The actress'} earned a degree in Film and Media Studies."
    await evaluator.verify(
        claim=claim_degree,
        node=degree_fms_leaf,
        sources=extracted.education_sources,
        additional_instruction="Verify the major/degree as Film and Media Studies. "
                               "Minor formatting variants are acceptable (e.g., 'Film & Media Studies')."
    )

    # Graduated_With_Honors
    honors_leaf = evaluator.add_leaf(
        id="Graduated_With_Honors",
        desc="The actress graduated with honors.",
        parent=edu_group,
        critical=True
    )
    claim_honors = f"{extracted.full_name or 'The actress'} graduated with honors."
    await evaluator.verify(
        claim=claim_honors,
        node=honors_leaf,
        sources=extracted.education_sources,
        additional_instruction="Verify that the sources explicitly mention graduating with honors. "
                               "Accept common honors phrasing (e.g., 'with honors', 'cum laude', 'magna cum laude', etc.)."
    )

    # Education_Reference_URLs existence
    evaluator.add_custom_node(
        result=len(extracted.education_sources) > 0,
        id="Education_Reference_URLs",
        desc="Provide reference URL(s) that support the education claims (ASU graduation, 2023 year, Film and Media Studies degree, and honors).",
        parent=edu_group,
        critical=True
    )


async def add_role_1923_nodes(evaluator: Evaluator, parent_node, extracted: ActressExtraction) -> None:
    # Role_In_1923 group (critical parallel)
    role1923_group = evaluator.add_parallel(
        id="Role_In_1923",
        desc="Role requirement for '1923' with supporting reference(s).",
        parent=parent_node,
        critical=True
    )

    # Appears_In_1923_As_Elizabeth_Strafford
    role_leaf = evaluator.add_leaf(
        id="Appears_In_1923_As_Elizabeth_Strafford",
        desc="The actress appears in the TV series '1923' playing the character Elizabeth Strafford.",
        parent=role1923_group,
        critical=True
    )
    claim_role1923 = f"{extracted.full_name or 'The actress'} appears in the TV series '1923' as Elizabeth Strafford."
    await evaluator.verify(
        claim=claim_role1923,
        node=role_leaf,
        sources=extracted.role_1923_sources,
        additional_instruction="Verify that the sources explicitly link the actress to the series '1923' and the character "
                               "'Elizabeth Strafford'. Allow minor spelling or formatting variations."
    )

    # 1923_Role_Reference_URL existence
    evaluator.add_custom_node(
        result=len(extracted.role_1923_sources) > 0,
        id="1923_Role_Reference_URL",
        desc="Provide at least one reference URL supporting the actress’s role in '1923' as Elizabeth Strafford.",
        parent=role1923_group,
        critical=True
    )


async def add_role_landman_nodes(evaluator: Evaluator, parent_node, extracted: ActressExtraction) -> None:
    # Role_In_Landman group (critical parallel)
    role_landman_group = evaluator.add_parallel(
        id="Role_In_Landman",
        desc="Role requirement for 'Landman' with supporting reference(s).",
        parent=parent_node,
        critical=True
    )

    # Appears_In_Landman_As_Ainsley_Norris
    role_leaf = evaluator.add_leaf(
        id="Appears_In_Landman_As_Ainsley_Norris",
        desc="The actress appears in the TV series 'Landman' playing the character Ainsley Norris.",
        parent=role_landman_group,
        critical=True
    )
    claim_role_landman = f"{extracted.full_name or 'The actress'} appears in the TV series 'Landman' as Ainsley Norris."
    await evaluator.verify(
        claim=claim_role_landman,
        node=role_leaf,
        sources=extracted.role_landman_sources,
        additional_instruction="Verify that the sources explicitly link the actress to the series 'Landman' and the character "
                               "'Ainsley Norris'. Allow minor spelling or formatting variations."
    )

    # Landman_Role_Reference_URL existence
    evaluator.add_custom_node(
        result=len(extracted.role_landman_sources) > 0,
        id="Landman_Role_Reference_URL",
        desc="Provide at least one reference URL supporting the actress’s role in 'Landman' as Ainsley Norris.",
        parent=role_landman_group,
        critical=True
    )


async def add_filming_1923_nodes(evaluator: Evaluator, parent_node, extracted: ActressExtraction) -> None:
    # Filming_Location_1923 group (critical parallel)
    filming1923_group = evaluator.add_parallel(
        id="Filming_Location_1923",
        desc="Filming location requirement for '1923' with supporting reference(s).",
        parent=parent_node,
        critical=True
    )

    # 1923_Filmed_In_Montana
    filmed_montana_leaf = evaluator.add_leaf(
        id="1923_Filmed_In_Montana",
        desc="The series '1923' was filmed in Montana.",
        parent=filming1923_group,
        critical=True
    )
    claim_filmed_montana = "The TV series '1923' was filmed in Montana."
    await evaluator.verify(
        claim=claim_filmed_montana,
        node=filmed_montana_leaf,
        sources=extracted.filming_1923_sources,
        additional_instruction="Verify that the sources explicitly mention filming in Montana for '1923'. "
                               "Accept phrasing like 'filmed in Montana', 'shot in Montana', or similar."
    )

    # 1923_Filming_Montana_Reference_URL existence
    evaluator.add_custom_node(
        result=len(extracted.filming_1923_sources) > 0,
        id="1923_Filming_Montana_Reference_URL",
        desc="Provide at least one reference URL supporting that '1923' was filmed in Montana.",
        parent=filming1923_group,
        critical=True
    )


async def add_filming_landman_nodes(evaluator: Evaluator, parent_node, extracted: ActressExtraction) -> None:
    # Filming_Location_Landman group (critical parallel)
    filming_landman_group = evaluator.add_parallel(
        id="Filming_Location_Landman",
        desc="Filming location requirement for 'Landman' with supporting reference(s).",
        parent=parent_node,
        critical=True
    )

    # Landman_Filmed_In_Fort_Worth_Area
    filmed_fw_leaf = evaluator.add_leaf(
        id="Landman_Filmed_In_Fort_Worth_Area",
        desc="The series 'Landman' was filmed in the Fort Worth, Texas area.",
        parent=filming_landman_group,
        critical=True
    )
    claim_filmed_fw = "The TV series 'Landman' was filmed in the Fort Worth, Texas area."
    await evaluator.verify(
        claim=claim_filmed_fw,
        node=filmed_fw_leaf,
        sources=extracted.filming_landman_sources,
        additional_instruction="Verify that the sources explicitly mention filming in the Fort Worth area for 'Landman'. "
                               "Accept variants such as 'Fort Worth, Texas', 'Dallas–Fort Worth (DFW) region', "
                               "or 'Fort Worth and surrounding areas'."
    )

    # Landman_Filming_Fort_Worth_Reference_URL existence
    evaluator.add_custom_node(
        result=len(extracted.filming_landman_sources) > 0,
        id="Landman_Filming_Fort_Worth_Reference_URL",
        desc="Provide at least one reference URL supporting that 'Landman' was filmed in the Fort Worth, Texas area.",
        parent=filming_landman_group,
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
    Evaluate the actress identification task answer:
    - Extract actress info and aspect-specific URLs from the answer.
    - Build a critical parallel verification tree per rubric.
    - Verify each claim against the cited sources.
    """
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

    # Extraction
    extracted: ActressExtraction = await evaluator.extract(
        prompt=prompt_extract_actress_info(),
        template_class=ActressExtraction,
        extraction_name="actress_extraction",
    )

    # Build top-level critical node (as rubric root)
    rubric_root = evaluator.add_parallel(
        id="Actress_Identification_Task",
        desc="Identify the actress who meets all specified criteria and provide reference URLs verifying each required aspect.",
        parent=root,
        critical=True
    )

    # Actress name existence
    add_actress_name_node(evaluator, rubric_root, extracted)

    # Birth year verification
    await add_birth_year_nodes(evaluator, rubric_root, extracted)

    # Education verification
    await add_education_nodes(evaluator, rubric_root, extracted)

    # Role in '1923'
    await add_role_1923_nodes(evaluator, rubric_root, extracted)

    # Role in 'Landman'
    await add_role_landman_nodes(evaluator, rubric_root, extracted)

    # Filming location '1923'
    await add_filming_1923_nodes(evaluator, rubric_root, extracted)

    # Filming location 'Landman'
    await add_filming_landman_nodes(evaluator, rubric_root, extracted)

    return evaluator.get_summary()