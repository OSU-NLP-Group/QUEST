import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_superintendents_2025_26"
TASK_DESCRIPTION = """
For the 2025-26 school year in Texas, identify the current superintendent and approximate student enrollment for both McKinney Independent School District (ISD) and Fort Bend Independent School District (ISD). Additionally, state the two primary educational credential requirements that Texas mandates for superintendent certification according to the Texas Education Agency.
"""

# Ground truth expectations used for matching checks
GROUND_TRUTH = {
    "mckinney": {
        "district_name": "McKinney ISD",
        "expected_superintendent": "Shawn Pratt",
        "enrollment_range_low": 23000,
        "enrollment_range_high": 24000,
    },
    "fort_bend": {
        "district_name": "Fort Bend ISD",
        "expected_superintendent": "Marc Smith",
        "enrollment_range_low": 78000,
        "enrollment_range_high": 80000,
    },
    "certification": {
        "masters_requirement": "Master's degree from an accredited university",
        "principal_or_experience": "Valid principal certificate OR managerial experience",
    }
}


# --------------------------------------------------------------------------- #
# Data Models for Extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictInfo(BaseModel):
    superintendent_name: Optional[str] = None
    superintendent_sources: List[str] = Field(default_factory=list)
    enrollment: Optional[str] = None
    enrollment_sources: List[str] = Field(default_factory=list)


class CertificationInfo(BaseModel):
    masters_requirement_text: Optional[str] = None
    masters_sources: List[str] = Field(default_factory=list)
    principal_or_experience_requirement_text: Optional[str] = None
    principal_or_experience_sources: List[str] = Field(default_factory=list)


class CombinedExtraction(BaseModel):
    mckinney: Optional[DistrictInfo] = None
    fort_bend: Optional[DistrictInfo] = None
    certification: Optional[CertificationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract structured information explicitly stated in the answer for two Texas school districts (McKinney ISD and Fort Bend ISD) and the Texas superintendent certification requirements. Return a single JSON object with three top-level keys: "mckinney", "fort_bend", and "certification".
    
    For each district (mckinney and fort_bend), extract:
    - superintendent_name: The name of the current superintendent as stated in the answer (string or null).
    - superintendent_sources: All URLs cited in the answer that support the superintendent information (array of URLs; if none, return an empty array).
    - enrollment: The reported student enrollment figure stated in the answer (string or null). The answer may use approximate language (e.g., "around 23,500", "about eighty thousand"); extract the text verbatim.
    - enrollment_sources: All URLs cited in the answer that support the enrollment figure (array of URLs; if none, return an empty array).
    
    For certification (Texas superintendent certification requirements per TEA), extract:
    - masters_requirement_text: The statement the answer uses to describe that a master's degree from an accredited university is required (string or null).
    - masters_sources: All URLs cited in the answer that support the master's degree requirement (array of URLs).
    - principal_or_experience_requirement_text: The statement the answer uses to describe that either a valid principal certificate OR managerial experience is required (string or null).
    - principal_or_experience_sources: All URLs cited in the answer that support the principal certificate OR managerial experience requirement (array of URLs).
    
    RULES:
    - Only extract information explicitly present in the answer.
    - For URL fields, include only valid URLs that appear in the answer (plain or markdown formats are acceptable).
    - If an item is missing in the answer, set its text field to null and its corresponding sources list to an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


# --------------------------------------------------------------------------- #
# Verification Subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_superintendent(
    evaluator: Evaluator,
    parent_node,
    district_key: str,
    district_name: str,
    expected_name: str,
    info: Optional[DistrictInfo],
) -> None:
    """
    Build verification nodes for superintendent identification with:
    - Existence of sources
    - Name match to expected
    - Source support for the claim on cited URLs
    """
    group_node = evaluator.add_parallel(
        id=f"{district_key}_Superintendent_Info",
        desc=f"{district_name} superintendent correctly identified as {expected_name} with valid reference URL",
        parent=parent_node,
        critical=False,
    )

    extracted_name = (info.superintendent_name if info else None) or ""
    sources = _safe_list(info.superintendent_sources if info else [])

    # Existence check: name provided AND at least one source URL
    evaluator.add_custom_node(
        result=(bool(extracted_name.strip()) and len(sources) > 0),
        id=f"{district_key}_superintendent_exists",
        desc=f"{district_name} superintendent info is provided with at least one source URL",
        parent=group_node,
        critical=True,
    )

    # Name match check
    match_node = evaluator.add_leaf(
        id=f"{district_key}_superintendent_match_expected",
        desc=f"The identified superintendent name matches the expected '{expected_name}'",
        parent=group_node,
        critical=True,
    )
    match_claim = (
        f"The extracted superintendent name '{extracted_name}' refers to the same person as '{expected_name}', "
        f"allowing minor variants such as honorifics (e.g., 'Dr.'), middle initials, or casing differences."
    )
    await evaluator.verify(
        claim=match_claim,
        node=match_node,
        additional_instruction="Judge equivalence flexibly (e.g., 'Marc Smith' vs. 'Dr. Marc A. Smith')."
    )

    # Source support check
    support_node = evaluator.add_leaf(
        id=f"{district_key}_superintendent_supported_by_sources",
        desc=f"'{expected_name}' is the current superintendent of {district_name} supported by cited sources",
        parent=group_node,
        critical=True,
    )
    support_claim = f"As of the 2025-26 school year, {expected_name} is the superintendent of {district_name}."
    await evaluator.verify(
        claim=support_claim,
        node=support_node,
        sources=sources,
        additional_instruction=(
            "Verify on the provided URLs that the person is identified as the district's superintendent. "
            "Allow reasonable recency (e.g., current bio or district leadership pages)."
        ),
    )


async def verify_enrollment(
    evaluator: Evaluator,
    parent_node,
    district_key: str,
    district_name: str,
    low: int,
    high: int,
    info: Optional[DistrictInfo],
) -> None:
    """
    Build verification nodes for enrollment with:
    - Existence of reported enrollment and source URL(s)
    - Value within expected range (simple verify on extracted text)
    - Source support for range claim using cited URLs
    """
    group_node = evaluator.add_parallel(
        id=f"{district_key}_Enrollment_Info",
        desc=f"{district_name} enrollment reported as approximately {low:,}-{high:,} students with valid reference URL",
        parent=parent_node,
        critical=False,
    )

    enrollment_text = (info.enrollment if info else None) or ""
    sources = _safe_list(info.enrollment_sources if info else [])

    # Existence check: enrollment text provided AND at least one source URL
    evaluator.add_custom_node(
        result=(bool(enrollment_text.strip()) and len(sources) > 0),
        id=f"{district_key}_enrollment_exists",
        desc=f"{district_name} enrollment value and source URL(s) are provided",
        parent=group_node,
        critical=True,
    )

    # Match to expected range based on extracted text
    match_node = evaluator.add_leaf(
        id=f"{district_key}_enrollment_within_range",
        desc=f"The reported {district_name} enrollment is approximately between {low:,} and {high:,} students",
        parent=group_node,
        critical=True,
    )
    match_claim = (
        f"The reported enrollment '{enrollment_text}' indicates a student count approximately between {low} and {high}."
    )
    await evaluator.verify(
        claim=match_claim,
        node=match_node,
        additional_instruction=(
            "Interpret the text's number(s) flexibly: words (e.g., 'about eighty thousand'), commas, rounding, or approximate phrasing. "
            "If the stated figure clearly falls within the range, consider it a match."
        ),
    )

    # Source support for range claim
    support_node = evaluator.add_leaf(
        id=f"{district_key}_enrollment_supported_by_sources",
        desc=f"{district_name} enrollment approx {low:,}-{high:,} is supported by cited sources",
        parent=group_node,
        critical=True,
    )
    support_claim = f"{district_name} student enrollment is approximately between {low} and {high}."
    await evaluator.verify(
        claim=support_claim,
        node=support_node,
        sources=sources,
        additional_instruction=(
            "From the provided URLs, confirm that the published enrollment figure reasonably falls within this range. "
            "Allow rounding; exact numbers like 78,187 should satisfy a 78,000–80,000 range."
        ),
    )


async def verify_certification_requirements(
    evaluator: Evaluator,
    parent_node,
    cert: Optional[CertificationInfo],
) -> None:
    """
    Build verification nodes for TEA superintendent certification requirements:
    - Master's degree requirement (existence, match, source support)
    - Principal certificate OR managerial experience requirement (existence, match, source support)
    """
    # Masters degree requirement
    masters_group = evaluator.add_parallel(
        id="Masters_Degree_Requirement",
        desc="Master's degree requirement from accredited university documented with valid reference URL",
        parent=parent_node,
        critical=False,
    )
    masters_text = (cert.masters_requirement_text if cert else None) or ""
    masters_sources = _safe_list(cert.masters_sources if cert else [])

    evaluator.add_custom_node(
        result=(bool(masters_text.strip()) and len(masters_sources) > 0),
        id="masters_requirement_exists",
        desc="Master's degree requirement statement and source URL(s) are provided in the answer",
        parent=masters_group,
        critical=True,
    )

    masters_match_node = evaluator.add_leaf(
        id="masters_requirement_text_mentions",
        desc="Extracted statement indicates a master's degree from an accredited university is required",
        parent=masters_group,
        critical=True,
    )
    masters_match_claim = (
        f"The extracted statement '{masters_text}' communicates that a master's degree from an accredited university is required for "
        f"Texas superintendent certification."
    )
    await evaluator.verify(
        claim=masters_match_claim,
        node=masters_match_node,
        additional_instruction="Accept synonymous phrasing like 'graduate degree (Master's)' and references to accredited institutions."
    )

    masters_support_node = evaluator.add_leaf(
        id="masters_requirement_supported_by_sources",
        desc="TEA master's degree requirement is supported by the cited sources",
        parent=masters_group,
        critical=True,
    )
    masters_support_claim = (
        "The Texas Education Agency requires a master's degree from an accredited university for the superintendent certificate."
    )
    await evaluator.verify(
        claim=masters_support_claim,
        node=masters_support_node,
        sources=masters_sources,
        additional_instruction=(
            "Prefer TEA official documentation pages. The source must explicitly state the master's degree requirement."
        ),
    )

    # Principal certificate OR managerial experience requirement
    po_group = evaluator.add_parallel(
        id="Principal_or_Experience_Requirement",
        desc="Principal certificate OR managerial experience requirement documented with valid reference URL",
        parent=parent_node,
        critical=False,
    )
    po_text = (cert.principal_or_experience_requirement_text if cert else None) or ""
    po_sources = _safe_list(cert.principal_or_experience_sources if cert else [])

    evaluator.add_custom_node(
        result=(bool(po_text.strip()) and len(po_sources) > 0),
        id="principal_or_experience_requirement_exists",
        desc="Principal certificate OR managerial experience statement and source URL(s) are provided in the answer",
        parent=po_group,
        critical=True,
    )

    po_match_node = evaluator.add_leaf(
        id="principal_or_experience_requirement_text_mentions",
        desc="Extracted statement indicates either a valid principal certificate OR managerial experience is required",
        parent=po_group,
        critical=True,
    )
    po_match_claim = (
        f"The extracted statement '{po_text}' clearly communicates that the requirement can be met by either a valid principal certificate "
        f"or managerial experience."
    )
    await evaluator.verify(
        claim=po_match_claim,
        node=po_match_node,
        additional_instruction="Accept equivalent phrasing like 'principal certification or qualifying managerial experience.'"
    )

    po_support_node = evaluator.add_leaf(
        id="principal_or_experience_requirement_supported_by_sources",
        desc="TEA principal certificate OR managerial experience requirement is supported by the cited sources",
        parent=po_group,
        critical=True,
    )
    po_support_claim = (
        "The Texas Education Agency allows meeting the superintendent certification prerequisite with either a valid principal certificate "
        "or managerial experience."
    )
    await evaluator.verify(
        claim=po_support_claim,
        node=po_support_node,
        sources=po_sources,
        additional_instruction=(
            "Prefer TEA official documentation pages. The source must explicitly present the 'either principal certificate OR managerial experience' option."
        ),
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Texas 2025-26 superintendent and certification requirements task.
    """
    # Initialize evaluator with parallel root (maps to "Task_Completion")
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

    # Record expected ground truth for transparency
    evaluator.add_ground_truth({
        "expected_superintendents": {
            "McKinney ISD": GROUND_TRUTH["mckinney"]["expected_superintendent"],
            "Fort Bend ISD": GROUND_TRUTH["fort_bend"]["expected_superintendent"],
        },
        "expected_enrollment_ranges": {
            "McKinney ISD": [GROUND_TRUTH["mckinney"]["enrollment_range_low"], GROUND_TRUTH["mckinney"]["enrollment_range_high"]],
            "Fort Bend ISD": [GROUND_TRUTH["fort_bend"]["enrollment_range_low"], GROUND_TRUTH["fort_bend"]["enrollment_range_high"]],
        },
        "expected_certification_requirements": {
            "masters_requirement": GROUND_TRUTH["certification"]["masters_requirement"],
            "principal_or_experience": GROUND_TRUTH["certification"]["principal_or_experience"],
        }
    }, gt_type="ground_truth")

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=CombinedExtraction,
        extraction_name="extracted_districts_and_certification",
    )

    # Build subtrees under root corresponding to rubric sections
    mck_node = evaluator.add_parallel(
        id="McKinney_ISD_Information",
        desc="Information about McKinney ISD's superintendent and enrollment",
        parent=root,
        critical=False,
    )
    fb_node = evaluator.add_parallel(
        id="Fort_Bend_ISD_Information",
        desc="Information about Fort Bend ISD's superintendent and enrollment",
        parent=root,
        critical=False,
    )
    cert_node = evaluator.add_parallel(
        id="Texas_Superintendent_Certification",
        desc="Texas state requirements for superintendent certification",
        parent=root,
        critical=False,
    )

    # McKinney ISD verifications
    await verify_superintendent(
        evaluator=evaluator,
        parent_node=mck_node,
        district_key="McKinney",
        district_name=GROUND_TRUTH["mckinney"]["district_name"],
        expected_name=GROUND_TRUTH["mckinney"]["expected_superintendent"],
        info=extraction.mckinney,
    )
    await verify_enrollment(
        evaluator=evaluator,
        parent_node=mck_node,
        district_key="McKinney",
        district_name=GROUND_TRUTH["mckinney"]["district_name"],
        low=GROUND_TRUTH["mckinney"]["enrollment_range_low"],
        high=GROUND_TRUTH["mckinney"]["enrollment_range_high"],
        info=extraction.mckinney,
    )

    # Fort Bend ISD verifications
    await verify_superintendent(
        evaluator=evaluator,
        parent_node=fb_node,
        district_key="Fort_Bend",
        district_name=GROUND_TRUTH["fort_bend"]["district_name"],
        expected_name=GROUND_TRUTH["fort_bend"]["expected_superintendent"],
        info=extraction.fort_bend,
    )
    await verify_enrollment(
        evaluator=evaluator,
        parent_node=fb_node,
        district_key="Fort_Bend",
        district_name=GROUND_TRUTH["fort_bend"]["district_name"],
        low=GROUND_TRUTH["fort_bend"]["enrollment_range_low"],
        high=GROUND_TRUTH["fort_bend"]["enrollment_range_high"],
        info=extraction.fort_bend,
    )

    # TEA certification requirement verifications
    await verify_certification_requirements(
        evaluator=evaluator,
        parent_node=cert_node,
        cert=extraction.certification,
    )

    # Return evaluation summary
    return evaluator.get_summary()