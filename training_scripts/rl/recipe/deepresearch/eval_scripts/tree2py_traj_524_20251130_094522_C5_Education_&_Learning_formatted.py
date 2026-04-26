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
TASK_ID = "linda_mcmahon_universities"
TASK_DESCRIPTION = (
    "Linda McMahon was confirmed as the U.S. Secretary of Education in March 2025. "
    "Identify two universities with which she has significant connections: "
    "(1) the university where she earned her undergraduate degree, which she completed in 1969, and "
    "(2) a university where she served on the Board of Trustees for multiple years, including service from 2004 to 2017. "
    "For each of the two universities, provide the following information as of the 2024-2025 academic year: "
    "the state where the university's main campus is located, the year the university was founded, "
    "whether the institution is a public or private university, the primary athletic conference for varsity sports, "
    "and the approximate total student enrollment (undergraduate and graduate combined)."
)

# Ground truth / constraints for verification
EXPECTED_UNIVERSITIES = {
    "undergraduate": {
        "name": "East Carolina University",
        "state": "North Carolina",
        "founded": "1907",
        "type": "public",
        "conference": "American Athletic Conference",
        "enrollment": "27000"  # approximately 27,000
    },
    "trustee": {
        "name": "Sacred Heart University",
        "state": "Connecticut",
        "founded": "1963",
        "type": "private",
        "conference": "Metro Atlantic Athletic Conference",
        "enrollment": "11000"  # approximately 11,000
    }
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityInfo(BaseModel):
    """Information for a single university as stated in the answer."""
    name: Optional[str] = None
    state: Optional[str] = None
    founding_year: Optional[str] = None
    institution_type: Optional[str] = None
    athletic_conference: Optional[str] = None
    enrollment_total: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    """Two universities associated with Linda McMahon."""
    undergraduate: Optional[UniversityInfo] = None
    trustee: Optional[UniversityInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract the two universities connected to Linda McMahon that the answer mentions:

    1) undergraduate: The university where she earned her undergraduate degree (completed in 1969).
    2) trustee: A university where she served on the Board of Trustees for multiple years, including service from 2004 to 2017.

    For each of these two universities, extract the following fields exactly as presented in the answer (do not infer):
    - name: The full university name.
    - state: The U.S. state where the university's main campus is located (e.g., 'North Carolina', 'Connecticut'). Accept common abbreviations if used in the answer (e.g., 'NC', 'Conn.').
    - founding_year: The year the university was founded (string format; do not convert to number).
    - institution_type: 'public' or 'private' (string; use the exact wording from the answer if present).
    - athletic_conference: The primary varsity athletic conference for the 2024–2025 academic year (string; use the exact wording from the answer if present).
    - enrollment_total: The approximate total student enrollment (undergraduate + graduate combined) as stated (string; can include words like 'about', '~', 'approximately', etc.).
    - sources: All URL(s) the answer cites for this university. Extract only valid URLs explicitly present in the answer (plain URLs or markdown links). If none are cited, return an empty array.

    Return a JSON object with two objects: 'undergraduate' and 'trustee', each following the schema.

    If the answer does not provide a particular field, return null for that field. If the answer does not mention any sources, return an empty array for 'sources'.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe(x: Optional[str]) -> str:
    return x or ""


async def verify_university_block(
    evaluator: Evaluator,
    parent_node,
    uni_info: Optional[UniversityInfo],
    block_id: str,
    block_desc: str,
    expected: Dict[str, str]
) -> None:
    """
    Build and verify the university sub-tree:
    - Sequential block (identification → characteristics).
    - Identification is critical (must match expected name).
    - Characteristics are grouped in a critical parallel block with 5 critical leaves.
    """
    # Create sequential block for this university
    seq_node = evaluator.add_sequential(
        id=block_id,
        desc=block_desc,
        parent=parent_node,
        critical=False  # Non-critical at this level; allows partial success across the two universities
    )

    # ---------------- Identification leaf (critical) ----------------
    ident_node = evaluator.add_leaf(
        id=f"{block_id}_university_identification",
        desc=f"Correct identification of the university ('{expected['name']}')",
        parent=seq_node,
        critical=True
    )

    extracted_name = _safe(uni_info.name if uni_info else None)
    ident_claim = f"The name '{extracted_name}' and '{expected['name']}' refer to the same university."
    await evaluator.verify(
        claim=ident_claim,
        node=ident_node,
        additional_instruction=(
            "Treat minor variations (e.g., abbreviations, letter casing, or inclusion/exclusion of 'University') "
            "as equivalent if they clearly refer to the same institution."
        )
    )

    # ---------------- Characteristics group (critical parallel) ----------------
    char_node = evaluator.add_parallel(
        id=f"{block_id}_university_characteristics",
        desc="Provide required characteristics for the identified university (as of 2024–25 academic year)",
        parent=seq_node,
        critical=True
    )

    sources = (uni_info.sources if uni_info and uni_info.sources else [])

    # 1) State Location
    state_leaf = evaluator.add_leaf(
        id=f"{block_id}_state_location",
        desc=f"State where the main campus is located ('{expected['state']}')",
        parent=char_node,
        critical=True
    )
    extracted_state = _safe(uni_info.state if uni_info else None)
    state_claim = (
        f"For {expected['name']}, the state reported in the answer is '{extracted_state}', "
        f"and it should be '{expected['state']}'. These refer to the same state."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=sources,
        additional_instruction=(
            "Allow common abbreviations and variants (e.g., 'NC' vs 'North Carolina', 'Conn.' vs 'Connecticut'). "
            "Confirm the university's main campus state from the provided sources."
        )
    )

    # 2) Founding Year
    founding_leaf = evaluator.add_leaf(
        id=f"{block_id}_founding_year",
        desc=f"Year the university was founded ('{expected['founded']}')",
        parent=char_node,
        critical=True
    )
    extracted_year = _safe(uni_info.founding_year if uni_info else None)
    founding_claim = (
        f"For {expected['name']}, the founding year reported in the answer is '{extracted_year}', "
        f"which should be '{expected['founded']}'. They match."
    )
    await evaluator.verify(
        claim=founding_claim,
        node=founding_leaf,
        sources=sources,
        additional_instruction=(
            "Check the official history or profile pages. Accept 'established'/'founded' wording equivalently. "
            "If multiple historical dates exist (e.g., charter vs opening), prefer the commonly cited founding year."
        )
    )

    # 3) Institution Type
    type_leaf = evaluator.add_leaf(
        id=f"{block_id}_institution_type",
        desc=f"Whether the institution is public or private ('{expected['type']}')",
        parent=char_node,
        critical=True
    )
    extracted_type = _safe(uni_info.institution_type if uni_info else None)
    type_claim = (
        f"For {expected['name']}, the institution type reported in the answer is '{extracted_type}', "
        f"which should be '{expected['type']}'. They are consistent."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm whether the university is 'public' (state-supported) or 'private' from the provided sources. "
            "Treat 'private not-for-profit' as 'private' for this verification."
        )
    )

    # 4) Athletic Conference (as of 2024–25)
    conf_leaf = evaluator.add_leaf(
        id=f"{block_id}_athletic_conference",
        desc=f"Primary athletic conference ('{expected['conference']}') as of 2024–25",
        parent=char_node,
        critical=True
    )
    extracted_conf = _safe(uni_info.athletic_conference if uni_info else None)
    conf_claim = (
        f"For the 2024–25 academic year, {expected['name']}'s primary varsity athletic conference reported in the answer "
        f"is '{extracted_conf}', which should be '{expected['conference']}'. They match."
    )
    await evaluator.verify(
        claim=conf_claim,
        node=conf_leaf,
        sources=sources,
        additional_instruction=(
            "Use official athletics or conference pages. Consider the main all-sports conference "
            "(e.g., the one governing men's and women's basketball). Allow abbreviations like 'AAC' for "
            "'American Athletic Conference' and 'MAAC' for 'Metro Atlantic Athletic Conference'. "
            "Ensure the timeframe is 2024–25."
        )
    )

    # 5) Enrollment Size (approximate)
    enroll_leaf = evaluator.add_leaf(
        id=f"{block_id}_enrollment_size",
        desc=f"Approximate total student enrollment (combined) (~{expected['enrollment']})",
        parent=char_node,
        critical=True
    )
    extracted_enroll = _safe(uni_info.enrollment_total if uni_info else None)
    enroll_claim = (
        f"For {expected['name']}, the total combined enrollment reported in the answer is '{extracted_enroll}', "
        f"which should be approximately {expected['enrollment']} students. This is approximately consistent."
    )
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_leaf,
        sources=sources,
        additional_instruction=(
            "Treat approximations, rounding, and phrasing like 'about', '~', or 'approximately' as acceptable. "
            "Allow a reasonable tolerance (e.g., within ±20%) when comparing enrollment figures."
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
    Evaluate an answer for Linda McMahon's university connections and characteristics.
    """
    # Initialize evaluator with a parallel root (two independent universities)
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

    # Add a parallel aggregation node for the overall universities analysis
    top_node = evaluator.add_parallel(
        id="universities_analysis",
        desc="Identification and detailed characteristics of two universities connected to Linda McMahon",
        parent=root,
        critical=False
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Record ground-truth constraints for transparency
    evaluator.add_ground_truth({
        "expected_undergraduate": EXPECTED_UNIVERSITIES["undergraduate"],
        "expected_trustee": EXPECTED_UNIVERSITIES["trustee"]
    }, gt_type="expected_constraints")

    # Verify the undergraduate university block
    await verify_university_block(
        evaluator=evaluator,
        parent_node=top_node,
        uni_info=extracted.undergraduate,
        block_id="undergraduate_university",
        block_desc="University where Linda McMahon earned her undergraduate degree (completed 1969)",
        expected=EXPECTED_UNIVERSITIES["undergraduate"]
    )

    # Verify the trustee university block
    await verify_university_block(
        evaluator=evaluator,
        parent_node=top_node,
        uni_info=extracted.trustee,
        block_id="trustee_university",
        block_desc="University where Linda McMahon served on Board of Trustees from 2004–2017",
        expected=EXPECTED_UNIVERSITIES["trustee"]
    )

    # Return structured summary
    return evaluator.get_summary()