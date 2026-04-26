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
TASK_ID = "ahsaa_6a_highest_enrollment_2026_2028"
TASK_DESCRIPTION = """
In the Alabama High School Athletic Association (AHSAA) 2026-2028 reclassification period, identify the public high school with the highest enrollment in Class 6A. Provide the school's exact enrollment number as listed in the official AHSAA classification document, identify which Alabama county the school is located in, determine how many public high schools from that same county are classified in Class 6A for the 2026-2028 period, and cite the official AHSAA enrollment document as your source.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AHSAAAnswerExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer for AHSAA 6A task.
    """
    school_name: Optional[str] = None
    claimed_class: Optional[str] = None  # e.g., "6A"
    enrollment_value: Optional[str] = None  # exact as written (e.g., "1,234")
    county_name: Optional[str] = None  # e.g., "Jefferson County"
    county_6a_school_count: Optional[str] = None  # number as string, e.g., "3"
    official_source_url: Optional[str] = None  # AHSAA 2026-2028 enrollment/classification document
    other_source_urls: List[str] = Field(default_factory=list)  # any additional URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_core_info() -> str:
    return """
    Extract the key information the answer provides about the AHSAA 2026–2028 Class 6A highest-enrollment public high school.

    Required fields to return:
    - school_name: The specific Alabama public high school the answer claims has the highest enrollment in Class 6A for 2026–2028.
    - claimed_class: The classification explicitly stated in the answer for the named school (e.g., "6A"). If not stated, return null.
    - enrollment_value: The exact enrollment (Average Daily Enrollment / ADE) number stated for the identified school, matching the format in the answer (keep commas or punctuation as-is). If not stated, return null.
    - county_name: The Alabama county where the identified school is located, as stated in the answer. If not stated, return null.
    - county_6a_school_count: The number of public high schools from that county classified in AHSAA Class 6A for 2026–2028, as stated in the answer. If not stated, return null.
    - official_source_url: The single URL that directly points to the official AHSAA 2026–2028 classification/enrollment document or its official AHSAA page that contains or links to that document (prefer a URL on the ahsaa.com domain or an official PDF). If multiple possible official AHSAA links are present, choose the most direct one. If no such URL is present, return null.
    - other_source_urls: An array of all other URLs mentioned in the answer (excluding the official_source_url). Include URLs regardless of format (plain, markdown). If none, return an empty array.

    Rules:
    - Return only what is explicitly stated in the answer; do not infer.
    - Maintain exact strings for numeric fields (e.g., preserve commas).
    - For URLs, extract full URLs. If a URL is missing a protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(official: Optional[str], others: List[str]) -> List[str]:
    """
    Combine official and other sources into a unique list (official first).
    """
    combined = []
    if official and official.strip():
        combined.append(official.strip())
    seen = set(combined)
    for u in others:
        if not u:
            continue
        u2 = u.strip()
        if not u2:
            continue
        if u2 not in seen:
            combined.append(u2)
            seen.add(u2)
    return combined


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, info: AHSAAAnswerExtraction) -> None:
    """
    Build the verification tree and run verifications according to the rubric.
    """

    # Root node: We set non-critical to allow partial credit for non-critical children.
    root = evaluator.root

    # 1) School Identification and Verification (Critical, Sequential)
    siv_node = evaluator.add_sequential(
        id="School_Identification_and_Verification",
        desc="The answer identifies a specific high school and verifies it is the school with the highest enrollment in AHSAA Class 6A for 2026-2028",
        parent=root,
        critical=True
    )

    # 1.1) School Named (Critical existence check)
    school_named_exists = bool(info.school_name and info.school_name.strip())
    school_named_node = evaluator.add_custom_node(
        result=school_named_exists,
        id="School_Named",
        desc="The answer provides the name of a specific Alabama public high school",
        parent=siv_node,
        critical=True
    )

    # Prepare official source and all sources list
    official_url = info.official_source_url or None
    all_sources = _combine_sources(official_url, info.other_source_urls)

    # 1.2) Classification Verified (Critical leaf)
    class_leaf = evaluator.add_leaf(
        id="Classification_Verified",
        desc="The named school is verified to be classified in AHSAA Class 6A for the 2026-2028 period according to official AHSAA documents",
        parent=siv_node,
        critical=True
    )
    classification_claim = f"The school '{info.school_name or ''}' is classified in AHSAA Class 6A for the 2026–2028 reclassification period."
    await evaluator.verify(
        claim=classification_claim,
        node=class_leaf,
        sources=official_url,  # Prefer the official document for classification verification
        additional_instruction="Use the official AHSAA 2026–2028 classification/enrollment document. Allow minor name variations (case, punctuation)."
    )

    # 1.3) Highest Enrollment Verified (Critical leaf)
    highest_leaf = evaluator.add_leaf(
        id="Highest_Enrollment_Verified",
        desc="The named school is verified to have the highest Average Daily Enrollment among all Class 6A public schools according to the official AHSAA 2026-2028 enrollment document",
        parent=siv_node,
        critical=True
    )
    highest_claim = (
        f"Among AHSAA Class 6A public high schools for the 2026–2028 period, '{info.school_name or ''}' "
        f"has the highest Average Daily Enrollment (ADE). If there is a tie for the highest ADE, this statement is still correct."
    )
    await evaluator.verify(
        claim=highest_claim,
        node=highest_leaf,
        sources=official_url,
        additional_instruction="Check the AHSAA 2026–2028 enrollment document table for Class 6A public schools and confirm the named school has the top ADE (ties acceptable)."
    )

    # 2) Enrollment Data (Non-Critical leaf)
    enrollment_leaf = evaluator.add_leaf(
        id="Enrollment_Data",
        desc="The answer provides the exact enrollment number and it matches the value listed in the official AHSAA 2026-2028 enrollment document for the identified school",
        parent=root,
        critical=False
    )
    enrollment_claim = (
        f"The official AHSAA 2026–2028 document lists the enrollment (ADE) for '{info.school_name or ''}' "
        f"as exactly '{info.enrollment_value or ''}'."
    )
    await evaluator.verify(
        claim=enrollment_claim,
        node=enrollment_leaf,
        sources=official_url,
        additional_instruction="Require an exact match to the value shown in the official document (respecting thousands separators or formatting as printed).",
        extra_prerequisites=[school_named_node, class_leaf]  # Gate on school name and classification success
    )

    # 3) Location and Count Information (Non-Critical, Sequential)
    loc_node = evaluator.add_sequential(
        id="Location_and_Count_Information",
        desc="The answer provides county location information and an accurate count of Class 6A schools from that county",
        parent=root,
        critical=False
    )

    # 3.1) County Identified (Non-Critical leaf)
    county_leaf = evaluator.add_leaf(
        id="County_Identified",
        desc="The answer identifies which Alabama county the school is located in, and this identification is verifiable through official school records",
        parent=loc_node,
        critical=False
    )
    county_claim = f"The school '{info.school_name or ''}' is located in {info.county_name or ''} County, Alabama."
    await evaluator.verify(
        claim=county_claim,
        node=county_leaf,
        sources=all_sources if all_sources else None,  # Try all available sources: official AHSAA + others
        additional_instruction="Prefer official school or district pages when available; otherwise accept reliable sources. Allow minor naming variations (e.g., 'Jefferson County Schools').",
        extra_prerequisites=[school_named_node]
    )

    # 3.2) County School Count Accurate (Non-Critical leaf)
    count_leaf = evaluator.add_leaf(
        id="County_School_Count_Accurate",
        desc="The answer provides a count of public high schools from the identified county that are classified in Class 6A for 2026-2028, and this count matches the actual number found in official AHSAA documents",
        parent=loc_node,
        critical=False
    )
    count_claim = (
        f"There are {info.county_6a_school_count or ''} public high schools from {info.county_name or ''} County "
        f"classified in AHSAA Class 6A for the 2026–2028 period."
    )
    await evaluator.verify(
        claim=count_claim,
        node=count_leaf,
        sources=official_url if official_url else all_sources,
        additional_instruction="Use the official AHSAA 2026–2028 document to count Class 6A public high schools from the specified county. If county info is not explicit in the document, corroborate using other provided reliable sources.",
        extra_prerequisites=[school_named_node, county_leaf]
    )

    # 4) Source Citation (Non-Critical leaf)
    source_leaf = evaluator.add_leaf(
        id="Source_Citation",
        desc="The answer cites the official AHSAA 2026-2028 enrollment document with a complete and accessible URL",
        parent=root,
        critical=False
    )
    if official_url:
        source_claim = "This URL is the official AHSAA 2026–2028 classification/enrollment document (or its official AHSAA page that contains/links to it)."
        await evaluator.verify(
            claim=source_claim,
            node=source_leaf,
            sources=official_url,
            additional_instruction="Confirm that the linked page/PDF is the official AHSAA document covering 2026–2028 classifications and enrollment numbers."
        )
    else:
        # Fallback: simple verification against answer content if no URL was extracted
        source_claim = "The answer includes at least one complete and accessible URL to the official AHSAA 2026–2028 classification/enrollment document."
        await evaluator.verify(
            claim=source_claim,
            node=source_leaf,
            sources=None,
            additional_instruction="Check the provided answer text for a proper URL to the official AHSAA document or official AHSAA page linking to it."
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
    Evaluate an answer for the AHSAA Class 6A highest enrollment (2026–2028) task.
    """
    # Initialize evaluator with a parallel root to allow independent non-critical checks
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract core info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_core_info(),
        template_class=AHSAAAnswerExtraction,
        extraction_name="ahsaa_core_info",
    )

    # Optionally record custom info to aid debugging
    evaluator.add_custom_info(
        info={
            "school_name": extracted_info.school_name,
            "claimed_class": extracted_info.claimed_class,
            "enrollment_value": extracted_info.enrollment_value,
            "county_name": extracted_info.county_name,
            "county_6a_school_count": extracted_info.county_6a_school_count,
            "official_source_url": extracted_info.official_source_url,
            "other_source_urls_count": len(extracted_info.other_source_urls),
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, extracted_info)

    # Return structured evaluation summary
    return evaluator.get_summary()