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
TASK_ID = "educator_identification_gcps_2026"
TASK_DESCRIPTION = (
    "Who is the educator with a Doctorate of Education from Sage College of Albany "
    "(earned between 2015-2017 with honors), who served as superintendent of Community "
    "School District 4 in East Harlem, New York, for seven years, currently serves as "
    "Superintendent of Norwalk Public Schools in Connecticut since July 1, 2020, and was "
    "named the sole finalist for the Superintendent position at Gwinnett County Public Schools in March 2026?"
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class EducatorExtraction(BaseModel):
    """
    Extracted information from the answer:
    - The identified educator's name (string)
    - Source URLs that the answer cites for each of the four required criteria
    """
    name: Optional[str] = None
    credentials_sources: List[str] = Field(default_factory=list)
    east_harlem_sources: List[str] = Field(default_factory=list)
    norwalk_sources: List[str] = Field(default_factory=list)
    gcps_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_educator() -> str:
    return """
    Extract the educator's identity and the specific source URLs cited in the answer that support each criterion below.

    You must extract:
    1) name: The full name of the educator identified in the answer text.
    2) credentials_sources: All URLs in the answer that support the claim that the educator earned a Doctorate of Education (Ed.D.) from Sage College of Albany (or The Sage Colleges / Sage Graduate Schools), earned between 2015 and 2017 (inclusive), and with honors (e.g., “with honors”, “cum laude”, “magna cum laude”, “summa cum laude”, “with distinction”).
    3) east_harlem_sources: All URLs in the answer that support the claim that the educator served as superintendent of Community School District 4 (East Harlem, NYC) for seven years (accept “CSD 4”, “District 4”, “NYC DOE District 4”, “East Harlem”).
    4) norwalk_sources: All URLs in the answer that support the claim that the educator currently serves as Superintendent of Norwalk Public Schools (Connecticut) since July 1, 2020.
    5) gcps_sources: All URLs in the answer that support the claim that in March 2026 the educator was named the sole finalist for the Superintendent position at Gwinnett County Public Schools (GCPS) in Georgia.

    Rules:
    - Extract ONLY URLs explicitly present in the answer (plain links, markdown links, or otherwise embedded). Do not invent or infer any URL.
    - If a category has no URLs in the answer, return an empty list for that field.
    - Return a single JSON object with fields: name, credentials_sources, east_harlem_sources, norwalk_sources, gcps_sources.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _person_phrase(name: Optional[str]) -> str:
    return name if (name and name.strip()) else "the educator identified in the answer"


def _safe_list(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_educational_credentials(evaluator: Evaluator, parent_node, data: EducatorExtraction) -> None:
    """
    Node: Educational_Credentials (critical)
    Children (sequential, both critical):
      - sources_present (custom, binary)
      - claim_supported (verify by URL(s))
    """
    node = evaluator.add_sequential(
        id="Educational_Credentials",
        desc="The identified educator holds a Doctorate of Education from Sage College of Albany, earned between 2015-2017 with honors",
        parent=parent_node,
        critical=True
    )

    sources = _safe_list(data.credentials_sources)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Educational_Credentials_sources_present",
        desc="Sources provided for credential verification",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Educational_Credentials_supported",
        desc="Doctorate of Education from Sage College of Albany (2015–2017, with honors) is supported by cited sources",
        parent=node,
        critical=True
    )

    claim = (
        f"{_person_phrase(data.name)} earned a Doctorate of Education (Ed.D.) from Sage College of Albany "
        f"(also known as The Sage Colleges/Sage Graduate Schools) between 2015 and 2017 (inclusive) and graduated with honors."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the page(s) explicitly indicate: "
            "1) the institution is Sage College of Albany or The Sage Colleges/Sage Graduate Schools; "
            "2) the degree is a Doctorate of Education (Ed.D.); "
            "3) the timeframe falls within 2015–2017 (inclusive); "
            "4) there is an honors notation (e.g., 'with honors', 'cum laude', 'magna cum laude', 'summa cum laude', 'with distinction'). "
            "Minor name variants and institutional branding differences are acceptable if they clearly refer to the same entity."
        )
    )


async def verify_east_harlem_superintendent(evaluator: Evaluator, parent_node, data: EducatorExtraction) -> None:
    """
    Node: East_Harlem_Superintendent (critical)
    Children (sequential, both critical):
      - sources_present (custom)
      - claim_supported (verify by URL(s))
    """
    node = evaluator.add_sequential(
        id="East_Harlem_Superintendent",
        desc="The identified educator served as superintendent of Community School District 4 in East Harlem, New York, for seven years",
        parent=parent_node,
        critical=True
    )

    sources = _safe_list(data.east_harlem_sources)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="East_Harlem_Superintendent_sources_present",
        desc="Sources provided for East Harlem District 4 superintendent verification",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="East_Harlem_Superintendent_supported",
        desc="CSD 4 (East Harlem) superintendent role for seven years is supported by cited sources",
        parent=node,
        critical=True
    )

    claim = (
        f"{_person_phrase(data.name)} served as superintendent of Community School District 4 (East Harlem, NYC) for seven years."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            "Accept phrasing such as 'Community School District 4', 'District 4', 'CSD 4', "
            "'NYC DOE District 4', or explicit mention of East Harlem. "
            "The evidence should support a tenure length of seven years (allowing reasonable paraphrases like 'for 7 years' "
            "or a date range whose difference is approximately 7 years)."
        )
    )


async def verify_norwalk_superintendent(evaluator: Evaluator, parent_node, data: EducatorExtraction) -> None:
    """
    Node: Norwalk_Superintendent (critical)
    Children (sequential, both critical):
      - sources_present (custom)
      - claim_supported (verify by URL(s))
    """
    node = evaluator.add_sequential(
        id="Norwalk_Superintendent",
        desc="The identified educator currently serves as Superintendent of Norwalk Public Schools in Connecticut since July 1, 2020",
        parent=parent_node,
        critical=True
    )

    sources = _safe_list(data.norwalk_sources)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Norwalk_Superintendent_sources_present",
        desc="Sources provided for Norwalk Superintendent (since July 1, 2020) verification",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Norwalk_Superintendent_supported",
        desc="Norwalk Public Schools Superintendent since July 1, 2020 is supported by cited sources",
        parent=node,
        critical=True
    )

    claim = (
        f"{_person_phrase(data.name)} has served as Superintendent of Norwalk Public Schools (Connecticut) since July 1, 2020."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the page(s) indicate the person is/was Superintendent of Norwalk Public Schools in CT, "
            "and that the start date is July 1, 2020 (or equivalent phrasing). "
            "If 'currently' is not explicitly stated, it is sufficient that the start date is July 1, 2020 and that "
            "the role is Superintendent of Norwalk Public Schools."
        )
    )


async def verify_gcps_sole_finalist(evaluator: Evaluator, parent_node, data: EducatorExtraction) -> None:
    """
    Node: GCPS_Sole_Finalist (critical)
    Children (sequential, both critical):
      - sources_present (custom)
      - claim_supported (verify by URL(s))
    """
    node = evaluator.add_sequential(
        id="GCPS_Sole_Finalist",
        desc="The identified educator was named the sole finalist for the Superintendent position at Gwinnett County Public Schools in March 2026",
        parent=parent_node,
        critical=True
    )

    sources = _safe_list(data.gcps_sources)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="GCPS_Sole_Finalist_sources_present",
        desc="Sources provided for GCPS sole finalist (March 2026) verification",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="GCPS_Sole_Finalist_supported",
        desc="GCPS sole finalist in March 2026 is supported by cited sources",
        parent=node,
        critical=True
    )

    claim = (
        f"In March 2026, {_person_phrase(data.name)} was named the sole finalist for the Superintendent position at Gwinnett County Public Schools (GCPS)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            "Confirm explicit mention that the person was named the 'sole finalist' for the Superintendent position at "
            "Gwinnett County Public Schools (GCPS) in March 2026. Accept minor phrasing variations that clearly indicate the same fact."
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
    Build the evaluation tree and run checks for the educator identification task.
    Returns a standardized summary dictionary from the evaluator.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level: check all four criteria independently
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

    # Extract structured information from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_educator(),
        template_class=EducatorExtraction,
        extraction_name="educator_extraction",
    )

    # Create the critical parent node mirroring the rubric root
    rubric_root = evaluator.add_parallel(
        id="Educator_Identification",
        desc="Correctly identify the educator who matches all specified criteria",
        parent=root,
        critical=True
    )

    # Build and verify each critical criterion subtree
    await verify_educational_credentials(evaluator, rubric_root, extracted)
    await verify_east_harlem_superintendent(evaluator, rubric_root, extracted)
    await verify_norwalk_superintendent(evaluator, rubric_root, extracted)
    await verify_gcps_sole_finalist(evaluator, rubric_root, extracted)

    # Return standardized summary
    return evaluator.get_summary()