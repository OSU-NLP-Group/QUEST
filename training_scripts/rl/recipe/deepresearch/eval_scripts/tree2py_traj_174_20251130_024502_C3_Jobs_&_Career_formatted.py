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
TASK_ID = "edu_sec_university_verification"
TASK_DESCRIPTION = (
    "A person was confirmed as the 13th U.S. Secretary of Education and was sworn into this position on March 3, 2025, "
    "following a Senate confirmation vote of 51-45. This person previously served as the Administrator of the U.S. Small "
    "Business Administration from 2017 to 2019. They graduated in 1969 from a public university located in Greenville, "
    "North Carolina, earning a bachelor's degree in French. This person also served on the Connecticut State Board of "
    "Education, having been appointed in 2009, and served as a trustee at a private Catholic university in Connecticut "
    "from 2004 to 2017. What is the name of the private Catholic university where this person served as a trustee, what "
    "city is it located in, and what year was that university founded?"
)

# Expected ground truth values for the private Catholic university details
EXPECTED_UNIVERSITY_NAME = "Sacred Heart University"
EXPECTED_CITY = "Fairfield"
EXPECTED_FOUNDING_YEAR = "1963"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityFacts(BaseModel):
    """Structured facts about the requested university from the answer."""
    university_name: Optional[str] = None
    university_city: Optional[str] = None
    founding_year: Optional[str] = None
    # Optional: any URLs the answer cited for these facts (not required by rubric)
    cited_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_facts() -> str:
    return """
    From the answer text, extract the specific facts requested about the private Catholic university where the person served as a trustee (2004–2017):
    - university_name: The exact name of that private Catholic university (e.g., "Sacred Heart University").
    - university_city: The city where that university is located (prefer just the city name like "Fairfield"; if the answer writes "Fairfield, Connecticut" you may return that exact string).
    - founding_year: The 4-digit founding year of that university as a string (e.g., "1963").
    - cited_urls: An array of any URLs explicitly cited in the answer that support these facts (if none, return an empty array).

    Rules:
    1) Extract only what is explicitly present in the answer.
    2) Do not infer or invent any values; if a field is not present, return null (or an empty array for cited_urls).
    3) If multiple universities are mentioned, select the one tied to the trustee role (2004–2017).
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_run_verification(evaluator: Evaluator, extracted: UniversityFacts) -> None:
    """
    Build the rubric tree exactly as specified and run the three critical leaf checks.
    The checks focus on whether the answer provides the correct university name, city, and founding year.
    """
    # Create the rubric's top-level node under root: critical + parallel aggregation
    answer_node = evaluator.add_parallel(
        id="Answer_Verification",
        desc="Verify the answer provides the three requested facts about the private Catholic university (name, city, founding year).",
        parent=evaluator.root,
        critical=True
    )

    # Leaf 1: University name accuracy (critical)
    name_leaf = evaluator.add_leaf(
        id="University_Name_Accuracy",
        desc="Provides the correct private Catholic university name: Sacred Heart University.",
        parent=answer_node,
        critical=True
    )
    # We verify the answer text itself states the correct university name.
    name_claim = (
        "The answer identifies the private Catholic university where the person served as a trustee as 'Sacred Heart University'. "
        "Minor abbreviations like 'Sacred Heart Univ.' are acceptable if unambiguous."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        additional_instruction=(
            "Focus only on whether the answer explicitly names the university as Sacred Heart University (allowing small variations like 'Sacred Heart Univ.'). "
            "Ignore other details."
        )
    )

    # Leaf 2: University city accuracy (critical)
    city_leaf = evaluator.add_leaf(
        id="University_City_Accuracy",
        desc="Provides the correct city where the university is located: Fairfield.",
        parent=answer_node,
        critical=True
    )
    city_claim = (
        "The answer states that Sacred Heart University is located in the city of 'Fairfield'. "
        "Variants such as 'Fairfield, CT' or 'Fairfield, Connecticut' should be treated as equivalent for the city value."
    )
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        additional_instruction=(
            "Accept 'Fairfield' as correct even if the answer writes 'Fairfield, CT' or 'Fairfield, Connecticut'. "
            "Reject if the answer indicates a different city."
        )
    )

    # Leaf 3: University founding year accuracy (critical)
    year_leaf = evaluator.add_leaf(
        id="University_Founding_Year_Accuracy",
        desc="Provides the correct founding year of the university: 1963.",
        parent=answer_node,
        critical=True
    )
    year_claim = (
        "The answer states that Sacred Heart University was founded in 1963. "
        "Accept phrasings like 'founded in 1963' or 'established in 1963'."
    )
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        additional_instruction=(
            "Judge only whether the answer clearly indicates the founding year as 1963 (allowing phrasing variants). "
            "Do not rely on external knowledge; rely solely on the answer text."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the requested university facts task.
    """
    # Initialize evaluator with a parallel root
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
        default_model=model
    )

    # Extract the structured facts from the answer (recorded in summary for transparency)
    extracted_facts = await evaluator.extract(
        prompt=prompt_extract_university_facts(),
        template_class=UniversityFacts,
        extraction_name="university_facts"
    )

    # Add ground truth snapshot to the summary
    evaluator.add_ground_truth({
        "expected_university_name": EXPECTED_UNIVERSITY_NAME,
        "expected_city": EXPECTED_CITY,
        "expected_founding_year": EXPECTED_FOUNDING_YEAR
    }, gt_type="ground_truth_university_facts")

    # Build verification tree and run checks
    await build_and_run_verification(evaluator, extracted_facts)

    # Return the structured evaluation summary
    return evaluator.get_summary()