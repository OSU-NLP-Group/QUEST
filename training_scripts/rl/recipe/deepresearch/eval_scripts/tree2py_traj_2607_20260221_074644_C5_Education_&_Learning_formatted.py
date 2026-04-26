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
TASK_ID = "nj_private_uni_identification"
TASK_DESCRIPTION = (
    "I'm researching universities in New Jersey with strong professional school programs. "
    "Can you identify the private university in New Jersey that meets all of the following criteria: "
    "(1) Founded in the 1850s, (2) Currently the largest private university in the state by total enrollment, "
    "(3) Has an AACSB-accredited business school that holds dual accreditation in both business and accounting, "
    "(4) Its business school was the first private business school in New Jersey to earn AACSB accreditation, "
    "(5) Has a law school. Please provide the name of the university and include reference URLs to support each criterion."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityExtraction(BaseModel):
    """
    Extracted information about the identified university and criterion-specific sources.
    All URLs must be explicitly present in the answer.
    """
    university_name: Optional[str] = None
    founding_year: Optional[str] = None
    business_school_name: Optional[str] = None
    law_school_name: Optional[str] = None

    # Sources for each criterion
    sources_location: List[str] = Field(default_factory=list)
    sources_private: List[str] = Field(default_factory=list)
    sources_founded: List[str] = Field(default_factory=list)
    sources_largest: List[str] = Field(default_factory=list)
    sources_aacsb_business: List[str] = Field(default_factory=list)
    sources_aacsb_accounting: List[str] = Field(default_factory=list)
    sources_first_private_aacsb: List[str] = Field(default_factory=list)
    sources_law_school: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_info() -> str:
    return """
    Extract the single private university in New Jersey that the answer identifies and collect criterion-specific reference URLs.

    Return a JSON object with the following fields:
    - university_name: The full name of the university (e.g., "Seton Hall University"). Return null if not provided.
    - founding_year: The specific founding year if the answer mentions one (e.g., "1856"). Return null if not provided.
    - business_school_name: The proper name of the university's business school if provided (e.g., "Stillman School of Business"). Return null if not provided.
    - law_school_name: The proper name of the university's law school if provided (e.g., "Seton Hall University School of Law"). Return null if not provided.

    For each criterion below, extract the explicit URLs cited in the answer that support it. Include ONLY URLs explicitly present in the answer. Do not invent or infer any URLs.
    - sources_location: URLs that support that the university is located in New Jersey.
    - sources_private: URLs that support that the institution is private (private university).
    - sources_founded: URLs that support the founding year and/or that the university was founded in the 1850s.
    - sources_largest: URLs that support that the university is currently the largest private university in New Jersey by total enrollment.
    - sources_aacsb_business: URLs that support that the business school is accredited by AACSB (business accreditation).
    - sources_aacsb_accounting: URLs that support AACSB accounting accreditation (dual accreditation).
    - sources_first_private_aacsb: URLs that support that it was the first private business school in New Jersey to earn AACSB accreditation.
    - sources_law_school: URLs that support that the university has a law school.

    SPECIAL RULES FOR URL SOURCES EXTRACTION:
    - Extract only valid full URLs explicitly present in the answer. Accept plain URLs, markdown links ([text](url)), or embedded links. If a URL is missing a protocol, prepend http://.
    - If the answer references a source in text without a URL (e.g., "according to AACSB"), return an empty list for that criterion.
    - Do not deduplicate across different criteria; keep the URLs exactly as provided for the specific criterion.

    If a criterion has no cited URLs in the answer, return an empty array for that criterion.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def nonempty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any((u or "").strip() for u in urls or [])


def safe_university_name(extracted: UniversityExtraction) -> str:
    return (extracted.university_name or "the university").strip()


def safe_business_school_name(extracted: UniversityExtraction) -> str:
    return (extracted.business_school_name or "the university's business school").strip()


def safe_law_school_name(extracted: UniversityExtraction) -> str:
    return (extracted.law_school_name or "the university's law school").strip()


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_institutional_profile(
    evaluator: Evaluator,
    parent_node,
    extracted: UniversityExtraction
) -> None:
    """
    Build and verify the 'Institutional_Profile' subtree:
    - New Jersey location
    - Private institution
    - Founded in the 1850s
    """
    inst_node = evaluator.add_parallel(
        id="Institutional_Profile",
        desc="Verify the basic institutional characteristics including location, type, and founding period",
        parent=parent_node,
        critical=True
    )

    # New Jersey location: Sources existence (critical) + claim verification
    evaluator.add_custom_node(
        result=nonempty_urls(extracted.sources_location),
        id="New_Jersey_Location_Sources_Provided",
        desc="Sources are provided to support New Jersey location",
        parent=inst_node,
        critical=True
    )
    nj_loc_leaf = evaluator.add_leaf(
        id="New_Jersey_Location",
        desc="The university is located in New Jersey",
        parent=inst_node,
        critical=True
    )
    nj_claim = f"{safe_university_name(extracted)} is located in New Jersey (NJ)."
    await evaluator.verify(
        claim=nj_claim,
        node=nj_loc_leaf,
        sources=extracted.sources_location,
        additional_instruction="Confirm that the referenced page(s) explicitly indicate the university is in New Jersey (NJ). Accept locality such as city in NJ."
    )

    # Private institution: Sources existence (critical) + claim verification
    evaluator.add_custom_node(
        result=nonempty_urls(extracted.sources_private),
        id="Private_Institution_Sources_Provided",
        desc="Sources are provided to support that the institution is private",
        parent=inst_node,
        critical=True
    )
    private_leaf = evaluator.add_leaf(
        id="Private_Institution",
        desc="The university is a private institution",
        parent=inst_node,
        critical=True
    )
    private_claim = f"{safe_university_name(extracted)} is a private university (non-public)."
    await evaluator.verify(
        claim=private_claim,
        node=private_leaf,
        sources=extracted.sources_private,
        additional_instruction="Verify that the source states the institution is private (not public/state). Wording like 'private research university' is acceptable."
    )

    # Founded in the 1850s: Sources existence (critical) + claim verification
    evaluator.add_custom_node(
        result=nonempty_urls(extracted.sources_founded),
        id="Founded_1850s_Sources_Provided",
        desc="Sources are provided to support founding in the 1850s",
        parent=inst_node,
        critical=True
    )
    founded_leaf = evaluator.add_leaf(
        id="Founded_1850s",
        desc="The university was founded in the 1850s",
        parent=inst_node,
        critical=True
    )
    if extracted.founding_year and extracted.founding_year.strip():
        founded_claim = (
            f"{safe_university_name(extracted)} was founded in {extracted.founding_year.strip()}, "
            "which is within the 1850s decade (1850–1859)."
        )
    else:
        founded_claim = f"{safe_university_name(extracted)} was founded in the 1850s."
    await evaluator.verify(
        claim=founded_claim,
        node=founded_leaf,
        sources=extracted.sources_founded,
        additional_instruction="If the founding year is between 1850 and 1859 inclusive, consider this criterion satisfied."
    )


async def verify_enrollment_status(
    evaluator: Evaluator,
    parent_node,
    extracted: UniversityExtraction
) -> None:
    """
    Build and verify the 'Enrollment_Status' subtree:
    - Largest private university in New Jersey by total enrollment
    """
    enr_node = evaluator.add_parallel(
        id="Enrollment_Status",
        desc="Verify the enrollment ranking among New Jersey private universities",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=nonempty_urls(extracted.sources_largest),
        id="Largest_Private_University_NJ_Sources_Provided",
        desc="Sources are provided to support largest private university status",
        parent=enr_node,
        critical=True
    )
    largest_leaf = evaluator.add_leaf(
        id="Largest_Private_University_NJ",
        desc="The university is the largest private university in New Jersey by total enrollment",
        parent=enr_node,
        critical=True
    )
    largest_claim = (
        f"Currently, {safe_university_name(extracted)} is the largest private university in New Jersey by total enrollment."
    )
    await evaluator.verify(
        claim=largest_claim,
        node=largest_leaf,
        sources=extracted.sources_largest,
        additional_instruction="Confirm the source explicitly asserts 'largest private university in New Jersey' by total enrollment. Prefer statements explicitly mentioning 'largest private' and 'total enrollment'."
    )


async def verify_business_school_accreditation(
    evaluator: Evaluator,
    parent_node,
    extracted: UniversityExtraction
) -> None:
    """
    Build and verify the 'Business_School_Accreditation' subtree:
    - AACSB business accreditation
    - AACSB accounting accreditation (dual)
    - First private business school in NJ to earn AACSB accreditation
    """
    biz_node = evaluator.add_parallel(
        id="Business_School_Accreditation",
        desc="Verify the business school's AACSB accreditation status and historical significance",
        parent=parent_node,
        critical=True
    )

    school_ref = safe_business_school_name(extracted)
    uni_ref = safe_university_name(extracted)

    # AACSB Business accreditation
    evaluator.add_custom_node(
        result=nonempty_urls(extracted.sources_aacsb_business),
        id="AACSB_Business_Accreditation_Sources_Provided",
        desc="Sources provided for AACSB business accreditation",
        parent=biz_node,
        critical=True
    )
    aacsb_business_leaf = evaluator.add_leaf(
        id="AACSB_Business_Accreditation",
        desc="The university has an AACSB-accredited business school",
        parent=biz_node,
        critical=True
    )
    aacsb_business_claim = f"The {school_ref} at {uni_ref} is accredited by AACSB (business accreditation)."
    await evaluator.verify(
        claim=aacsb_business_claim,
        node=aacsb_business_leaf,
        sources=extracted.sources_aacsb_business,
        additional_instruction="Confirm AACSB business accreditation is explicitly stated for the business school."
    )

    # AACSB Accounting accreditation (dual)
    evaluator.add_custom_node(
        result=nonempty_urls(extracted.sources_aacsb_accounting),
        id="AACSB_Accounting_Accreditation_Sources_Provided",
        desc="Sources provided for AACSB accounting accreditation (dual accreditation)",
        parent=biz_node,
        critical=True
    )
    aacsb_accounting_leaf = evaluator.add_leaf(
        id="AACSB_Accounting_Accreditation",
        desc="The business school holds dual AACSB accreditation in both business and accounting",
        parent=biz_node,
        critical=True
    )
    aacsb_accounting_claim = (
        f"The {school_ref} at {uni_ref} holds AACSB accreditation in Accounting in addition to Business (i.e., dual AACSB accreditation)."
    )
    await evaluator.verify(
        claim=aacsb_accounting_claim,
        node=aacsb_accounting_leaf,
        sources=extracted.sources_aacsb_accounting,
        additional_instruction="Confirm that AACSB accreditation for Accounting is explicitly stated (separate from Business), indicating dual AACSB accreditation."
    )

    # First private NJ business school to earn AACSB accreditation
    evaluator.add_custom_node(
        result=nonempty_urls(extracted.sources_first_private_aacsb),
        id="First_Private_NJ_AACSB_Sources_Provided",
        desc="Sources provided for 'first private NJ business school to earn AACSB accreditation'",
        parent=biz_node,
        critical=True
    )
    first_private_leaf = evaluator.add_leaf(
        id="First_Private_NJ_AACSB",
        desc="The business school was the first private business school in New Jersey to earn AACSB accreditation",
        parent=biz_node,
        critical=True
    )
    first_private_claim = (
        f"The {school_ref} at {uni_ref} was the first private business school in New Jersey to earn AACSB accreditation."
    )
    await evaluator.verify(
        claim=first_private_claim,
        node=first_private_leaf,
        sources=extracted.sources_first_private_aacsb,
        additional_instruction="Confirm the source explicitly asserts 'first private business school in New Jersey' to earn AACSB accreditation."
    )


async def verify_law_school_presence(
    evaluator: Evaluator,
    parent_node,
    extracted: UniversityExtraction
) -> None:
    """
    Build and verify the 'Law_School_Presence' subtree:
    - University has a law school
    """
    law_node = evaluator.add_parallel(
        id="Law_School_Presence",
        desc="Verify that the university has a law school",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=nonempty_urls(extracted.sources_law_school),
        id="Has_Law_School_Sources_Provided",
        desc="Sources provided to support that the university has a law school",
        parent=law_node,
        critical=True
    )
    law_leaf = evaluator.add_leaf(
        id="Has_Law_School",
        desc="The university has a law school",
        parent=law_node,
        critical=True
    )
    law_name = safe_law_school_name(extracted)
    law_claim = f"{safe_university_name(extracted)} has a law school (e.g., {law_name})."
    await evaluator.verify(
        claim=law_claim,
        node=law_leaf,
        sources=extracted.sources_law_school,
        additional_instruction="Confirm that the source explicitly indicates the university operates a law school. Naming the law school is acceptable but not required."
    )


# --------------------------------------------------------------------------- #
# Main verification orchestration                                             #
# --------------------------------------------------------------------------- #
async def build_tree_and_verify(evaluator: Evaluator, extracted: UniversityExtraction) -> None:
    """
    Build the verification tree following the rubric and perform all checks.
    """
    # Top-level critical node
    uni_id_node = evaluator.add_parallel(
        id="University_Identification",
        desc="Identify a university in New Jersey that meets all specified institutional, historical, enrollment, and academic program criteria",
        parent=evaluator.root,
        critical=True
    )

    # Global prerequisite: University name must be provided
    evaluator.add_custom_node(
        result=bool(extracted.university_name and extracted.university_name.strip()),
        id="University_Name_Provided",
        desc="The university name is provided in the answer",
        parent=uni_id_node,
        critical=True
    )

    # Institutional profile checks
    await verify_institutional_profile(evaluator, uni_id_node, extracted)

    # Enrollment status check
    await verify_enrollment_status(evaluator, uni_id_node, extracted)

    # Business school accreditation checks
    await verify_business_school_accreditation(evaluator, uni_id_node, extracted)

    # Law school presence check
    await verify_law_school_presence(evaluator, uni_id_node, extracted)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
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
    Evaluate an answer for the 'NJ Private University Identification' task.
    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator (root node is non-critical parallel aggregator)
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_university_info(),
        template_class=UniversityExtraction,
        extraction_name="university_info"
    )

    # Build verification tree and run checks
    await build_tree_and_verify(evaluator, extracted)

    # Return final summary
    return evaluator.get_summary()