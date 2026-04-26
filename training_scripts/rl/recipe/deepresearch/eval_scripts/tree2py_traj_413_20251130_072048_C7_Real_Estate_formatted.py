import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fl_broker_license_2025"
TASK_DESCRIPTION = (
    "What are the complete requirements to obtain a Florida real estate broker license in 2025? "
    "Provide a comprehensive list that includes all age requirements, educational prerequisites, "
    "experience requirements, pre-licensing education hours, examination requirements including passing scores, "
    "identification requirements, background check requirements, prior license status requirements, "
    "post-licensing education requirements for the first renewal period, and continuing education requirements for subsequent renewals. "
    "For each requirement, specify exact numerical values (such as hours, percentages, or time periods) where applicable."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BrokerRequirementsExtraction(BaseModel):
    # Core requirement statements as explicitly presented in the answer
    age_requirement: Optional[str] = None
    education_prerequisite: Optional[str] = None
    experience_requirement: Optional[str] = None
    prelicensing_hours: Optional[str] = None
    examination_requirement: Optional[str] = None
    passing_score: Optional[str] = None
    ssn_requirement: Optional[str] = None
    background_check: Optional[str] = None
    license_good_standing: Optional[str] = None
    postlicensing_first_renewal: Optional[str] = None
    continuing_ed_hours: Optional[str] = None
    continuing_ed_frequency: Optional[str] = None
    ce_core_law_hours: Optional[str] = None
    ce_ethics_hours: Optional[str] = None
    ce_specialty_hours: Optional[str] = None

    # URLs explicitly cited in the answer, categorized by requirement
    urls_all: List[str] = Field(default_factory=list)
    age_urls: List[str] = Field(default_factory=list)
    education_urls: List[str] = Field(default_factory=list)
    experience_urls: List[str] = Field(default_factory=list)
    prelicensing_urls: List[str] = Field(default_factory=list)
    exam_urls: List[str] = Field(default_factory=list)
    passing_score_urls: List[str] = Field(default_factory=list)
    ssn_urls: List[str] = Field(default_factory=list)
    background_check_urls: List[str] = Field(default_factory=list)
    good_standing_urls: List[str] = Field(default_factory=list)
    post_licensing_urls: List[str] = Field(default_factory=list)
    continuing_ed_urls: List[str] = Field(default_factory=list)
    ce_core_law_urls: List[str] = Field(default_factory=list)
    ce_ethics_urls: List[str] = Field(default_factory=list)
    ce_specialty_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_broker_requirements() -> str:
    return (
        "Extract the Florida real estate broker license requirements exactly as they appear in the provided answer. "
        "Return a JSON object with the following fields:\n"
        "- age_requirement: Text stating the age requirement (e.g., 'at least 18 years old').\n"
        "- education_prerequisite: Text stating the high school diploma or GED requirement.\n"
        "- experience_requirement: Text stating experience (e.g., 'active sales associate license for at least 24 months during the preceding 5 years').\n"
        "- prelicensing_hours: Text with broker pre-licensing education hours (e.g., '72 hours').\n"
        "- examination_requirement: Text stating the exam requirement (e.g., 'must pass the Florida broker exam').\n"
        "- passing_score: Text with the passing score (e.g., '75%').\n"
        "- ssn_requirement: Text stating social security number requirement.\n"
        "- background_check: Text stating electronic fingerprint/background check requirement.\n"
        "- license_good_standing: Text indicating prior sales associate license must be in good standing.\n"
        "- postlicensing_first_renewal: Text stating broker post-licensure hours and timing (e.g., '60 hours within the first renewal period').\n"
        "- continuing_ed_hours: Text stating total CE hours after first renewal (e.g., '14 hours').\n"
        "- continuing_ed_frequency: Text stating CE frequency (e.g., 'every 2 years').\n"
        "- ce_core_law_hours: Text stating Core Law hours (e.g., '3 hours').\n"
        "- ce_ethics_hours: Text stating Ethics & Business Practices hours (e.g., '3 hours').\n"
        "- ce_specialty_hours: Text stating specialty hours (e.g., '8 hours').\n"
        "Also extract the URLs mentioned in the answer and categorize them into arrays for each requirement:\n"
        "- urls_all: All URLs cited anywhere in the answer.\n"
        "- age_urls, education_urls, experience_urls, prelicensing_urls, exam_urls, passing_score_urls,\n"
        "  ssn_urls, background_check_urls, good_standing_urls, post_licensing_urls, continuing_ed_urls,\n"
        "  ce_core_law_urls, ce_ethics_urls, ce_specialty_urls: Each is an array of URLs specifically associated with that requirement.\n"
        "If a particular requirement is not mentioned, use null for the text field and an empty array for the corresponding URLs.\n"
        "Extract only what is explicitly present in the answer. Do not invent information."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def get_sources_for(extracted: BrokerRequirementsExtraction, keys: List[str]) -> List[str]:
    """
    Collect URLs from categorized fields with optional fallback to urls_all.
    keys must be suffixes without '_urls' (e.g., 'age', 'prelicensing').
    """
    urls: List[str] = []
    for k in keys:
        field_name = f"{k}_urls"
        if hasattr(extracted, field_name):
            urls.extend(getattr(extracted, field_name) or [])
    # Fallback: if category URLs are empty, include any overall sources
    if not urls:
        urls.extend(extracted.urls_all or [])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _nonempty(text: Optional[str]) -> bool:
    return bool(text and str(text).strip())


# --------------------------------------------------------------------------- #
# Verification functions (each requirement => its own critical sequential node) #
# --------------------------------------------------------------------------- #
async def verify_age(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="Age_Requirement",
        desc="Identifies that applicant must be at least 18 years old",
        parent=parent_node,
        critical=True,
    )
    # Presence in answer (critical)
    evaluator.add_custom_node(
        result=_nonempty(ext.age_requirement),
        id="age_presence",
        desc="Age requirement is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    # Numeric value included in the answer (critical)
    leaf_val = evaluator.add_leaf(
        id="age_value_in_answer",
        desc="Answer includes the exact age value: at least 18 years old",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly includes the age requirement as at least 18 years old (minor phrasing variations acceptable).",
        node=leaf_val,
        additional_instruction="Check only the answer content for '18 years old' or equivalent phrasing like 'at least 18'. Allow reasonable synonyms or minor wording variations.",
    )
    # Supported by sources (critical)
    leaf_sup = evaluator.add_leaf(
        id="age_supported_by_sources",
        desc="Age requirement (at least 18 years old) is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Florida real estate broker license applicants must be at least 18 years old.",
        node=leaf_sup,
        sources=get_sources_for(ext, ["age"]),
        additional_instruction="Verify using the provided URLs that the minimum age requirement is 18 years. Accept equivalent statements.",
    )


async def verify_education(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="Education_Prerequisite",
        desc="Identifies that applicant must have a high school diploma or GED",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.education_prerequisite),
        id="education_presence",
        desc="Education prerequisite is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    leaf_sup = evaluator.add_leaf(
        id="education_supported_by_sources",
        desc="Education prerequisite (high school diploma or GED) is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Florida real estate broker license applicants must have a high school diploma or GED.",
        node=leaf_sup,
        sources=get_sources_for(ext, ["education"]),
        additional_instruction="Verify using the provided URLs that the education requirement is a high school diploma or GED.",
    )


async def verify_experience(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="Experience_Requirement",
        desc="Identifies that applicant must have held an active real estate sales associate license for at least 24 months during the preceding 5 years",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.experience_requirement),
        id="experience_presence",
        desc="Experience requirement is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    leaf_val = evaluator.add_leaf(
        id="experience_value_in_answer",
        desc="Answer includes the exact experience numeric details: at least 24 months within preceding 5 years",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes the experience requirement details: at least 24 months (two years) during the preceding 5 years.",
        node=leaf_val,
        additional_instruction="Check answer content only. Accept equivalent phrasing like 'two years' for 24 months; ensure the 'preceding 5 years' window is present.",
    )
    leaf_sup = evaluator.add_leaf(
        id="experience_supported_by_sources",
        desc="Experience requirement (24 months in the preceding 5 years) is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Florida requires broker applicants to have held an active real estate sales associate license for at least 24 months during the preceding five years.",
        node=leaf_sup,
        sources=get_sources_for(ext, ["experience"]),
        additional_instruction="Confirm the duration (24 months/two years) and the 5-year look-back window via the provided URLs.",
    )


async def verify_prelicensing(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="PreLicensing_Education_Hours",
        desc="Identifies that applicant must complete 72 hours of FREC-approved broker pre-licensing education",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.prelicensing_hours),
        id="prelicensing_presence",
        desc="Pre-licensing education requirement is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    leaf_val = evaluator.add_leaf(
        id="prelicensing_value_in_answer",
        desc="Answer includes the numeric value: 72 hours of broker pre-licensing education",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes broker pre-licensing hours as 72 hours.",
        node=leaf_val,
        additional_instruction="Check answer content only; allow minor phrasing variations.",
    )
    leaf_sup = evaluator.add_leaf(
        id="prelicensing_supported_by_sources",
        desc="72-hour broker pre-licensing requirement is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Florida broker pre-licensing education requires 72 hours of FREC-approved coursework.",
        node=leaf_sup,
        sources=get_sources_for(ext, ["prelicensing"]),
        additional_instruction="Verify via URLs that broker pre-licensing education is 72 hours and FREC-approved.",
    )


async def verify_exam(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="Examination_Requirement",
        desc="Identifies that applicant must pass the Florida broker license examination",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.examination_requirement),
        id="exam_presence",
        desc="Examination requirement is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    leaf_sup = evaluator.add_leaf(
        id="exam_supported_by_sources",
        desc="Exam requirement (must pass Florida broker exam) is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Florida broker license applicants must pass the Florida broker examination.",
        node=leaf_sup,
        sources=get_sources_for(ext, ["exam"]),
        additional_instruction="Use provided URLs to confirm that passing the broker exam is required.",
    )


async def verify_passing_score(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="Passing_Score",
        desc="Identifies that applicant must achieve at least 75% on the broker examination",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.passing_score),
        id="passing_score_presence",
        desc="Passing score requirement is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    leaf_val = evaluator.add_leaf(
        id="passing_score_value_in_answer",
        desc="Answer includes the numeric passing score: at least 75%",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes the passing score for the broker exam as at least 75%.",
        node=leaf_val,
        additional_instruction="Check answer content only; allow variants like 'score of 75' or '75 percent'.",
    )
    leaf_sup = evaluator.add_leaf(
        id="passing_score_supported_by_sources",
        desc="Passing score (≥75%) is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The passing score for the Florida broker examination is at least 75%.",
        node=leaf_sup,
        sources=get_sources_for(ext, ["passing_score"]),
        additional_instruction="Confirm the passing score threshold using URLs.",
    )


async def verify_ssn(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="SSN_Requirement",
        desc="Identifies that applicant must have a valid Social Security Number",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.ssn_requirement),
        id="ssn_presence",
        desc="SSN requirement is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    leaf_sup = evaluator.add_leaf(
        id="ssn_supported_by_sources",
        desc="SSN requirement is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Florida broker applicants must have a valid Social Security Number.",
        node=leaf_sup,
        sources=get_sources_for(ext, ["ssn"]),
        additional_instruction="Confirm SSN requirement via URLs; accept statements referencing identification/SSN specifically.",
    )


async def verify_background_check(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="Background_Check",
        desc="Identifies that applicant must submit electronic fingerprints for background check",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.background_check),
        id="background_presence",
        desc="Fingerprint/background check requirement is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    leaf_sup = evaluator.add_leaf(
        id="background_supported_by_sources",
        desc="Electronic fingerprint/background check requirement is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Florida broker license applicants must submit electronic fingerprints for a background check.",
        node=leaf_sup,
        sources=get_sources_for(ext, ["background_check"]),
        additional_instruction="Confirm fingerprint/background check requirement via URLs.",
    )


async def verify_good_standing(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="License_Good_Standing",
        desc="Identifies that applicant's prior sales associate license must have been held in good standing",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.license_good_standing),
        id="good_standing_presence",
        desc="Good-standing requirement is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    leaf_sup = evaluator.add_leaf(
        id="good_standing_supported_by_sources",
        desc="Good-standing requirement is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Florida requires that the prior sales associate license has been held in good standing.",
        node=leaf_sup,
        sources=get_sources_for(ext, ["good_standing"]),
        additional_instruction="Confirm good-standing condition via URLs; allow equivalent language such as 'no disciplinary issues' or 'in good standing'.",
    )


async def verify_postlicensing(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="PostLicense_First_Renewal",
        desc="Identifies that broker must complete 60 hours of approved broker post-licensure courses within the first renewal period",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.postlicensing_first_renewal),
        id="postlicensing_presence",
        desc="Post-licensing requirement for first renewal is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    leaf_val = evaluator.add_leaf(
        id="postlicensing_value_in_answer",
        desc="Answer includes the numeric value: 60 hours within the first renewal period",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes broker post-licensing as 60 hours within the first renewal period.",
        node=leaf_val,
        additional_instruction="Check answer content only; allow equivalent phrasing.",
    )
    leaf_sup = evaluator.add_leaf(
        id="postlicensing_supported_by_sources",
        desc="60-hour post-licensing requirement (first renewal period) is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Florida brokers must complete 60 hours of approved post-licensure education within the first renewal period.",
        node=leaf_sup,
        sources=get_sources_for(ext, ["post_licensing"]),
        additional_instruction="Confirm via URLs the 60-hour broker post-licensing requirement within the first renewal timeframe.",
    )


async def verify_continuing_ed(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="Continuing_Education_Hours",
        desc="Identifies that broker must complete 14 hours of continuing education every 2 years after first renewal",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.continuing_ed_hours) or _nonempty(ext.continuing_ed_frequency),
        id="ce_presence",
        desc="Continuing education hours/frequency is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    leaf_hours_val = evaluator.add_leaf(
        id="ce_hours_value_in_answer",
        desc="Answer includes the numeric CE hours: 14 hours",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes continuing education hours as 14 hours.",
        node=leaf_hours_val,
        additional_instruction="Check answer content only; allow minor phrasing variations.",
    )
    leaf_freq_val = evaluator.add_leaf(
        id="ce_frequency_in_answer",
        desc="Answer includes CE frequency: every 2 years (biennial)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes continuing education frequency as every 2 years (biennial).",
        node=leaf_freq_val,
        additional_instruction="Check answer content only; allow equivalent phrasing like 'biennial' or 'every other year'.",
    )
    leaf_hours_sup = evaluator.add_leaf(
        id="ce_hours_supported_by_sources",
        desc="14-hour CE requirement is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="After the first renewal, Florida brokers must complete 14 hours of continuing education.",
        node=leaf_hours_sup,
        sources=get_sources_for(ext, ["continuing_ed"]),
        additional_instruction="Confirm the 14-hour CE requirement using URLs.",
    )
    leaf_freq_sup = evaluator.add_leaf(
        id="ce_frequency_supported_by_sources",
        desc="Biennial CE frequency (every 2 years) is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Florida broker continuing education is required every 2 years (biennially) after the first renewal.",
        node=leaf_freq_sup,
        sources=get_sources_for(ext, ["continuing_ed"]),
        additional_instruction="Confirm CE frequency via URLs; accept equivalent phrasing.",
    )


async def verify_ce_core_law(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="CE_Core_Law",
        desc="Identifies that continuing education must include 3 hours of Core Law",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.ce_core_law_hours),
        id="ce_core_law_presence",
        desc="Core Law CE component is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    leaf_val = evaluator.add_leaf(
        id="ce_core_law_value_in_answer",
        desc="Answer includes the numeric Core Law hours: 3 hours",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes that continuing education must include 3 hours of Core Law.",
        node=leaf_val,
        additional_instruction="Check answer content only; accept equivalent phrasing.",
    )
    leaf_sup = evaluator.add_leaf(
        id="ce_core_law_supported_by_sources",
        desc="Core Law 3-hour requirement is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Florida broker continuing education must include 3 hours of Core Law.",
        node=leaf_sup,
        sources=get_sources_for(ext, ["ce_core_law"]),
        additional_instruction="Confirm Core Law hours via URLs.",
    )


async def verify_ce_ethics(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="CE_Ethics",
        desc="Identifies that continuing education must include 3 hours of Ethics and Business Practices",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.ce_ethics_hours),
        id="ce_ethics_presence",
        desc="Ethics/Business Practices CE component is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    leaf_val = evaluator.add_leaf(
        id="ce_ethics_value_in_answer",
        desc="Answer includes the numeric Ethics & Business Practices hours: 3 hours",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes that continuing education must include 3 hours of Ethics and Business Practices.",
        node=leaf_val,
        additional_instruction="Check answer content only; accept equivalent phrasing.",
    )
    leaf_sup = evaluator.add_leaf(
        id="ce_ethics_supported_by_sources",
        desc="Ethics & Business Practices 3-hour requirement is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Florida broker continuing education must include 3 hours of Ethics and Business Practices.",
        node=leaf_sup,
        sources=get_sources_for(ext, ["ce_ethics"]),
        additional_instruction="Confirm Ethics/Business Practices hours via URLs.",
    )


async def verify_ce_specialty(evaluator: Evaluator, parent_node, ext: BrokerRequirementsExtraction) -> None:
    node = evaluator.add_sequential(
        id="CE_Specialty",
        desc="Identifies that continuing education must include 8 hours of specialty education",
        parent=parent_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.ce_specialty_hours),
        id="ce_specialty_presence",
        desc="Specialty CE component is explicitly included in the answer",
        parent=node,
        critical=True,
    )
    leaf_val = evaluator.add_leaf(
        id="ce_specialty_value_in_answer",
        desc="Answer includes the numeric specialty hours: 8 hours",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes that continuing education must include 8 hours of specialty education.",
        node=leaf_val,
        additional_instruction="Check answer content only; accept equivalent phrasing.",
    )
    leaf_sup = evaluator.add_leaf(
        id="ce_specialty_supported_by_sources",
        desc="Specialty 8-hour requirement is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Florida broker continuing education must include 8 hours of specialty education.",
        node=leaf_sup,
        sources=get_sources_for(ext, ["ce_specialty"]),
        additional_instruction="Confirm specialty hours via URLs.",
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
    Evaluate an answer for the Florida broker license requirements (2025) task.
    Builds a critical parallel node that aggregates critical sequential checks for each requirement.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extract structured requirements and cited URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_broker_requirements(),
        template_class=BrokerRequirementsExtraction,
        extraction_name="broker_requirements_extraction",
    )

    # Record expected numeric values as ground-truth references for transparency (informational only)
    evaluator.add_ground_truth({
        "expected_values": {
            "age": "at least 18 years old",
            "experience": "at least 24 months within the preceding 5 years",
            "prelicensing_hours": "72 hours (FREC-approved)",
            "exam": "must pass Florida broker exam",
            "passing_score": "at least 75%",
            "ssn": "valid Social Security Number",
            "background_check": "electronic fingerprints for background check",
            "good_standing": "prior sales associate license held in good standing",
            "postlicensing_first_renewal": "60 hours within first renewal period",
            "continuing_education": "14 hours every 2 years after first renewal",
            "ce_core_law": "3 hours",
            "ce_ethics": "3 hours",
            "ce_specialty": "8 hours",
        }
    }, gt_type="expected_requirements")

    # Create a critical parent to reflect rubric "Root" critical requirement aggregation
    requirements_main = evaluator.add_parallel(
        id="Requirements_Root",
        desc=(
            "Complete identification of all Florida real estate broker license requirements including education, "
            "experience, examination, identification, background check, post-licensing, and continuing education requirements"
        ),
        parent=root,
        critical=True,
    )

    # Build verification subtrees per requirement (all critical under requirements_main)
    await verify_age(evaluator, requirements_main, extracted)
    await verify_education(evaluator, requirements_main, extracted)
    await verify_experience(evaluator, requirements_main, extracted)
    await verify_prelicensing(evaluator, requirements_main, extracted)
    await verify_exam(evaluator, requirements_main, extracted)
    await verify_passing_score(evaluator, requirements_main, extracted)
    await verify_ssn(evaluator, requirements_main, extracted)
    await verify_background_check(evaluator, requirements_main, extracted)
    await verify_good_standing(evaluator, requirements_main, extracted)
    await verify_postlicensing(evaluator, requirements_main, extracted)
    await verify_continuing_ed(evaluator, requirements_main, extracted)
    await verify_ce_core_law(evaluator, requirements_main, extracted)
    await verify_ce_ethics(evaluator, requirements_main, extracted)
    await verify_ce_specialty(evaluator, requirements_main, extracted)

    # Return standardized summary
    return evaluator.get_summary()