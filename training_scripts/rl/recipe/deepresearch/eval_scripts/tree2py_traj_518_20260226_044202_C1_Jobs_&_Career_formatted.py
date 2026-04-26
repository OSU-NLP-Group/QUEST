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
TASK_ID = "scott_beardsley_doctoral_degree"
TASK_DESCRIPTION = (
    "Scott C. Beardsley was appointed as the 10th President of the University of Virginia in December 2025. "
    "Identify the doctoral degree he holds that qualifies him for this leadership position. Specifically, provide: "
    "(1) the type of doctoral degree, (2) the field or specialization of the degree, and (3) the university that "
    "granted this degree."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DoctoralDegreeExtraction(BaseModel):
    degree_type: Optional[str] = None
    field_of_study: Optional[str] = None
    granting_institution: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_doctoral_degree() -> str:
    return """
    Extract the doctoral degree information for Scott C. Beardsley as presented in the answer.

    Required fields:
    1) degree_type: The type of doctoral degree, e.g., "Ph.D.", "PhD", "Doctor of Philosophy", "Ed.D.", "Doctor of Education", etc.
    2) field_of_study: The specific field or specialization of the doctoral degree, e.g., "Higher Education", "Education Leadership", "Business Administration", etc.
    3) granting_institution: The full name of the university that granted the doctoral degree, e.g., "University of Pennsylvania".
    4) sources: An array of all URLs explicitly mentioned in the answer that directly support this doctoral degree information (any part of it). 
       Only include URLs that are cited in the answer.

    Rules:
    - Return null for any field not explicitly provided in the answer.
    - The 'sources' array must only include URLs explicitly present in the answer (plain URLs or markdown links).
    - Do not infer URLs. If no URLs are provided in the answer, return an empty array for 'sources'.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def add_degree_type_verification(evaluator: Evaluator, parent_node, extracted: DoctoralDegreeExtraction) -> None:
    seq_node = evaluator.add_sequential(
        id="degree_type",
        desc="Identify the type of doctoral degree (e.g., Ph.D., Ed.D., etc.)",
        parent=parent_node,
        critical=True
    )

    # Existence check: value provided
    evaluator.add_custom_node(
        result=bool(extracted.degree_type and extracted.degree_type.strip()),
        id="degree_type_value_provided",
        desc="Degree type value is provided in the answer",
        parent=seq_node,
        critical=True
    )

    # Existence check: sources provided
    evaluator.add_custom_node(
        result=bool(extracted.sources),
        id="degree_type_sources_provided",
        desc="At least one supporting source URL is provided for the doctoral degree information",
        parent=seq_node,
        critical=True
    )

    # Evidence-backed verification
    leaf = evaluator.add_leaf(
        id="degree_type_supported",
        desc="The identified degree type is supported by the cited sources",
        parent=seq_node,
        critical=True
    )

    degree_type_val = extracted.degree_type or ""
    claim = f"Scott C. Beardsley holds a doctoral degree of type '{degree_type_val}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Verify that the provided webpages explicitly support the doctoral degree type for Scott C. Beardsley. "
            "Treat equivalent forms as the same (e.g., 'PhD', 'Ph.D.', 'Doctor of Philosophy'; "
            "'Ed.D.' and 'Doctor of Education'). If the pages list multiple degrees, ensure you identify the doctoral "
            "degree type."
        ),
    )


async def add_field_verification(evaluator: Evaluator, parent_node, extracted: DoctoralDegreeExtraction) -> None:
    seq_node = evaluator.add_sequential(
        id="field_of_study",
        desc="Identify the specific field or specialization of the doctoral degree",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.field_of_study and extracted.field_of_study.strip()),
        id="field_value_provided",
        desc="Field/specialization value is provided in the answer",
        parent=seq_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.sources),
        id="field_sources_provided",
        desc="At least one supporting source URL is provided for the doctoral degree information",
        parent=seq_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="field_supported",
        desc="The identified field/specialization is supported by the cited sources",
        parent=seq_node,
        critical=True
    )

    field_val = extracted.field_of_study or ""
    claim = f"The field or specialization of Scott C. Beardsley's doctoral degree is '{field_val}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Verify that the webpages explicitly indicate the field/specialization of Scott C. Beardsley's doctoral degree. "
            "Allow reasonable synonyms and phrasing variants (e.g., 'Higher Education', 'Higher Education Management', "
            "'Education Leadership and Policy' may be overlapping areas). Confirm that the field refers to the doctoral "
            "degree, not another degree."
        ),
    )


async def add_institution_verification(evaluator: Evaluator, parent_node, extracted: DoctoralDegreeExtraction) -> None:
    seq_node = evaluator.add_sequential(
        id="granting_institution",
        desc="Identify the university that granted the doctoral degree",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.granting_institution and extracted.granting_institution.strip()),
        id="institution_value_provided",
        desc="Granting institution value is provided in the answer",
        parent=seq_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.sources),
        id="institution_sources_provided",
        desc="At least one supporting source URL is provided for the doctoral degree information",
        parent=seq_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="institution_supported",
        desc="The identified granting institution is supported by the cited sources",
        parent=seq_node,
        critical=True
    )

    institution_val = extracted.granting_institution or ""
    claim = f"The doctoral degree was granted by '{institution_val}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Verify that the webpages explicitly state the university that awarded Scott C. Beardsley's doctoral degree. "
            "Allow common name variants (e.g., 'University of Pennsylvania' vs 'Penn', 'UPenn'), but ensure it refers to "
            "the awarding institution for the doctoral degree."
        ),
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

    # Extract structured doctoral degree info
    extracted = await evaluator.extract(
        prompt=prompt_extract_doctoral_degree(),
        template_class=DoctoralDegreeExtraction,
        extraction_name="doctoral_degree_extraction"
    )

    # Build tree: add a critical parent aggregator under root to mimic critical root behavior
    critical_parent = evaluator.add_parallel(
        id="scott_beardsley_doctoral_degree",
        desc="Verify the doctoral degree held by Scott C. Beardsley that qualifies him for the UVA presidency",
        parent=root,
        critical=True
    )

    # Add verifications for each critical criterion
    await add_degree_type_verification(evaluator, critical_parent, extracted)
    await add_field_verification(evaluator, critical_parent, extracted)
    await add_institution_verification(evaluator, critical_parent, extracted)

    # Return evaluation summary
    return evaluator.get_summary()