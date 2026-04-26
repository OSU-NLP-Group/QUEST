import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "ky_largest_district"
TASK_DESCRIPTION = (
    "Identify the largest public school district in Kentucky by student enrollment. "
    "Provide the following information: (1) the official name of the district, "
    "(2) the primary city where it is located, (3) its national ranking among U.S. school districts by size, "
    "(4) the approximate number of students currently enrolled, (5) the total number of schools it operates, "
    "and (6) a reference URL to verify this information."
)

GROUND_TRUTH = {
    "official_name": "Jefferson County Public Schools",
    "primary_city": "Louisville, Kentucky",
    "national_ranking": "30th",
    "approx_enrollment": "≈94,793",
    "schools_range": "169–171 (inclusive)",
}


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class DistrictExtraction(BaseModel):
    district_name: Optional[str] = None
    primary_city: Optional[str] = None
    national_ranking: Optional[str] = None
    student_enrollment: Optional[str] = None
    number_of_schools: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_district_info() -> str:
    return """
    From the provided answer, extract the following fields about the single largest public school district in Kentucky by student enrollment:
    - district_name: The district's official name exactly as stated in the answer (e.g., "Jefferson County Public Schools", allow abbreviation "JCPS" if that's what the answer states).
    - primary_city: The primary city (and state if present) where the district is located as stated in the answer (e.g., "Louisville, Kentucky" or "Louisville, KY").
    - national_ranking: The stated national ranking by size among U.S. school districts (e.g., "30th", "No. 30", "ranked 30th").
    - student_enrollment: The approximate current enrollment exactly as stated (keep formatting like ~, ≈, commas, or words like "about" if present).
    - number_of_schools: The stated total number of schools (can be a single number or an approximate/range like "169-171").
    - reference_urls: An array of all explicit URLs the answer cites to support the above claims. Extract only actual URLs (including those inside markdown links). If none are present, return an empty list.

    Rules:
    - Do not invent information. Only extract what appears in the answer.
    - If any field is missing from the answer, set it to null (or empty list for reference_urls).
    - The reference_urls should be all URLs the answer uses to support the facts about the district; include multiple if listed.
    """


# -----------------------------------------------------------------------------
# Verification helpers
# -----------------------------------------------------------------------------
async def build_references_subtree(
    evaluator: Evaluator,
    parent,
    extracted: DistrictExtraction,
):
    """
    Build 'Provide_Verifying_References' subtree (critical, parallel).
    Returns the 'has_reference_url' leaf node for use as an extra prerequisite in other verifications.
    """
    refs_node = evaluator.add_parallel(
        id="Provide_Verifying_References",
        desc="Provides reference URL(s) that verify the identification and constrained attributes.",
        parent=parent,
        critical=True,
    )

    # Leaf: Has at least one reference URL
    has_ref_leaf = evaluator.add_custom_node(
        result=bool(extracted.reference_urls),
        id="Has_Reference_URL",
        desc="Provides at least one reference URL.",
        parent=refs_node,
        critical=True,
    )

    district_label = extracted.district_name or "the identified district"
    urls = extracted.reference_urls

    # Leaf: Reference supports 'largest in Kentucky'
    ref_supports_largest = evaluator.add_leaf(
        id="Reference_Supports_Largest_In_Kentucky",
        desc="At least one provided reference supports the claim that the district is the largest in Kentucky by enrollment.",
        parent=refs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{district_label} is the largest public school district in Kentucky by student enrollment.",
        node=ref_supports_largest,
        sources=urls,
        additional_instruction="Decide 'supported' only if at least one provided URL explicitly states or clearly confirms this largest-in-Kentucky-by-enrollment claim.",
    )

    # Leaf: Reference supports '30th largest in the U.S.'
    ref_supports_ranking = evaluator.add_leaf(
        id="Reference_Supports_National_Ranking",
        desc="At least one provided reference supports the '30th largest in the U.S.' ranking claim.",
        parent=refs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{district_label} is the 30th largest school district in the United States by student enrollment.",
        node=ref_supports_ranking,
        sources=urls,
        additional_instruction="Accept reasonable variants like 'ranked 30th', 'No. 30', or similar, but it must explicitly indicate 30th in the U.S.",
    )

    # Leaf: Reference supports enrollment ~94,793
    ref_supports_enrollment = evaluator.add_leaf(
        id="Reference_Supports_Enrollment",
        desc="At least one provided reference supports the ~94,793 enrollment claim.",
        parent=refs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{district_label} has approximately 94,793 students currently enrolled (allowing small rounding variation).",
        node=ref_supports_enrollment,
        sources=urls,
        additional_instruction="Treat 'approximately 94,793' as satisfied if a source states 94,793 or a very close value with 'approx/about/~' wording.",
    )

    # Leaf: Reference supports number of schools ~169–171
    ref_supports_schools = evaluator.add_leaf(
        id="Reference_Supports_Number_Of_Schools",
        desc="At least one provided reference supports the ~169–171 schools claim.",
        parent=refs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{district_label} operates approximately between 169 and 171 schools (inclusive).",
        node=ref_supports_schools,
        sources=urls,
        additional_instruction="Supported if the source states a number within 169–171 or an explicit range that lies within 169–171.",
    )

    return has_ref_leaf


async def build_identify_district_subtree(
    evaluator: Evaluator,
    parent,
    extracted: DistrictExtraction,
    has_ref_leaf,
):
    """
    Build 'Identify_Correct_District' subtree (critical, parallel).
    """
    id_node = evaluator.add_parallel(
        id="Identify_Correct_District",
        desc="Correctly identifies the district entity required by the constraints.",
        parent=parent,
        critical=True,
    )

    district_label = extracted.district_name or "the identified district"
    urls = extracted.reference_urls

    # The identified entity is a public school district
    is_public_leaf = evaluator.add_leaf(
        id="District_Is_Public_School_District",
        desc="The identified entity is a public school district.",
        parent=id_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{district_label} is a public school district in Kentucky.",
        node=is_public_leaf,
        sources=urls,
        additional_instruction="Mark 'supported' only if at least one provided reference clearly indicates it is a public school district.",
        extra_prerequisites=[has_ref_leaf],
    )

    # Official name must be JCPS
    official_name_leaf = evaluator.add_leaf(
        id="Official_Name_Is_Jefferson_County_Public_Schools",
        desc="The identified district’s official name is Jefferson County Public Schools (JCPS).",
        parent=id_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The extracted district name '{extracted.district_name}' refers to the same entity as 'Jefferson County Public Schools (JCPS)'.",
        node=official_name_leaf,
        additional_instruction="Allow common abbreviation 'JCPS'. Case-insensitive and minor formatting differences are acceptable. If the extracted name is missing or unrelated, mark incorrect.",
    )

    # Largest in Kentucky by enrollment (supported by citation)
    largest_ky_leaf = evaluator.add_leaf(
        id="Largest_In_Kentucky_By_Enrollment",
        desc="The identified district is the largest public school district in Kentucky by student enrollment (supported by citation).",
        parent=id_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{district_label} is the largest public school district in Kentucky by student enrollment.",
        node=largest_ky_leaf,
        sources=urls,
        additional_instruction="Decide 'supported' only if a provided source explicitly confirms this.",
        extra_prerequisites=[has_ref_leaf],
    )


async def build_constrained_attributes_subtree(
    evaluator: Evaluator,
    parent,
    extracted: DistrictExtraction,
    has_ref_leaf,
):
    """
    Build 'Provide_Constrained_Attributes' subtree (critical, parallel).
    """
    attr_node = evaluator.add_parallel(
        id="Provide_Constrained_Attributes",
        desc="Provides the remaining required attributes matching the explicit constraints.",
        parent=parent,
        critical=True,
    )

    # Primary city: Louisville, Kentucky
    city_leaf = evaluator.add_leaf(
        id="Primary_City_Is_Louisville_Kentucky",
        desc="States the district’s primary city as Louisville, Kentucky.",
        parent=attr_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The stated primary city value '{extracted.primary_city}' refers to Louisville, Kentucky (allow variants like 'Louisville, KY' or just 'Louisville').",
        node=city_leaf,
        additional_instruction="If the provided value is missing, empty, or does not indicate Louisville, Kentucky, mark incorrect.",
        extra_prerequisites=[has_ref_leaf],
    )

    # National ranking: 30th
    ranking_leaf = evaluator.add_leaf(
        id="National_Ranking_Is_30th",
        desc="States the district’s national ranking by size as 30th largest among U.S. school districts.",
        parent=attr_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The stated national ranking '{extracted.national_ranking}' indicates the district is the 30th largest in the U.S. (accept '30th', 'No. 30', 'ranked 30th').",
        node=ranking_leaf,
        additional_instruction="If the provided value is missing/blank or does not clearly indicate '30th', mark incorrect.",
        extra_prerequisites=[has_ref_leaf],
    )

    # Enrollment: approximately 94,793
    enrollment_leaf = evaluator.add_leaf(
        id="Enrollment_Is_Approximately_94793",
        desc="Reports approximate student enrollment as approximately 94,793 (or explicitly indicates ~94,793).",
        parent=attr_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The stated enrollment '{extracted.student_enrollment}' indicates approximately 94,793 students (allow ±1% and symbols like ~, ≈, 'about').",
        node=enrollment_leaf,
        additional_instruction="If the provided value is missing/blank or clearly not around 94,793, mark incorrect.",
        extra_prerequisites=[has_ref_leaf],
    )

    # Number of schools: approximately 169–171 (inclusive)
    schools_leaf = evaluator.add_leaf(
        id="Operates_Approximately_169_to_171_Schools",
        desc="Reports the total number of schools operated as approximately 169–171 (inclusive) or explicitly gives a value/range within 169–171.",
        parent=attr_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The stated number of schools '{extracted.number_of_schools}' indicates approximately 169–171 schools inclusive (or a number within that range).",
        node=schools_leaf,
        additional_instruction="If missing/blank or outside 169–171, mark incorrect. Accept wording like 'about ~170' or a range overlapping 169–171.",
        extra_prerequisites=[has_ref_leaf],
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Kentucky largest school district task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
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
        default_model=model,
    )

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_district_info(),
        template_class=DistrictExtraction,
        extraction_name="district_extraction",
    )

    # Add GT for reference
    evaluator.add_ground_truth(
        {
            "expected_official_name": GROUND_TRUTH["official_name"],
            "expected_primary_city": GROUND_TRUTH["primary_city"],
            "expected_national_ranking": GROUND_TRUTH["national_ranking"],
            "expected_enrollment": GROUND_TRUTH["approx_enrollment"],
            "expected_schools": GROUND_TRUTH["schools_range"],
        },
        gt_type="ground_truth",
    )

    # Build the task node and subtrees (critical, sequential root-level node)
    task_node = evaluator.add_sequential(
        id="Kentucky_Largest_District_Information",
        desc="Identify the largest public school district in Kentucky and provide required constrained attributes with verifying reference URL(s).",
        parent=root,
        critical=True,
    )

    # According to source-grounding best practice, verify references first (critical)
    has_ref_leaf = await build_references_subtree(evaluator, task_node, extracted)

    # Then, identify district (critical)
    await build_identify_district_subtree(evaluator, task_node, extracted, has_ref_leaf)

    # Then, verify constrained attributes (critical)
    await build_constrained_attributes_subtree(evaluator, task_node, extracted, has_ref_leaf)

    # Return final summary
    return evaluator.get_summary()