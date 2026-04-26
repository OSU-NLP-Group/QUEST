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
TASK_ID = "texas_superintendent_requirements"
TASK_DESCRIPTION = (
    "An educator in Texas is considering pursuing a superintendent position and wants to understand the eligibility "
    "requirements. What are the minimum educational, certification, and experience requirements that must be met to be "
    "eligible for a Texas Superintendent certificate? Additionally, what was the median salary for a Texas school "
    "superintendent for the 2025-26 school year according to the TASB survey?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementItem(BaseModel):
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SuperintendentExtraction(BaseModel):
    masters_degree: Optional[RequirementItem] = None
    principal_certification: Optional[RequirementItem] = None
    principal_experience: Optional[RequirementItem] = None
    median_salary_2025_26: Optional[str] = None
    median_salary_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt(s)                                                        #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent_info() -> str:
    return """
    Extract from the answer the specific items related to Texas Superintendent certification eligibility and the TASB salary figure.

    Return a JSON object with the following fields:
    - masters_degree: { statement, sources }
        • statement: The exact text in the answer that describes the minimum education requirement for a Texas Superintendent certificate.
          This should mention a master's degree if the answer states it.
        • sources: A list of all URLs cited in the answer that the answer uses to support this requirement.
    - principal_certification: { statement, sources }
        • statement: The exact text in the answer that describes any requirement to hold a Texas Principal certificate (or equivalent).
        • sources: A list of all URLs cited in the answer that the answer uses to support this requirement.
    - principal_experience: { statement, sources }
        • statement: The exact text in the answer that describes the minimum experience as a school principal (e.g., “two years”).
        • sources: A list of all URLs cited in the answer that the answer uses to support this requirement.
    - median_salary_2025_26: The median salary figure for Texas school superintendents for the 2025–26 school year as stated in the answer (include any currency symbols and commas as shown).
    - median_salary_sources: A list of all URLs cited in the answer that support the TASB median salary figure for 2025–26.

    Rules:
    1) Only extract URLs explicitly present in the answer (plain URLs or markdown links). Do not invent any URLs.
    2) If an item is not mentioned in the answer, set its statement to null (and an empty list for sources).
    3) For median_salary_2025_26, only extract the number/amount that the answer labels as the “median” for the “2025–26” school year;
       if ambiguous or not present, set it to null and return an empty list for its sources.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_statement_and_sources(item: Optional[RequirementItem]) -> bool:
    if not item:
        return False
    has_stmt = bool(item.statement and item.statement.strip())
    has_src = bool(item.sources and len(item.sources) > 0)
    return has_stmt and has_src


def _nonempty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_master_degree_checks(
    evaluator: Evaluator,
    parent,
    extracted: SuperintendentExtraction,
) -> None:
    """
    Build and execute checks for the master's degree requirement.
    """
    node = evaluator.add_sequential(
        id="Master_Degree_Verification",
        desc="The answer correctly identifies that a master's degree from an accredited university is required",
        parent=parent,
        critical=True,
    )

    # Existence + sources check (critical)
    exists_node = evaluator.add_custom_node(
        result=_has_statement_and_sources(extracted.masters_degree),
        id="master_degree_present_with_sources",
        desc="Master's degree requirement is stated in the answer and includes at least one supporting URL",
        parent=node,
        critical=True,
    )

    # Evidence-backed correctness check (critical)
    masters_leaf = evaluator.add_leaf(
        id="master_degree_supported",
        desc="Texas requires a master's degree from an accredited university for Superintendent certification eligibility (supported by cited sources)",
        parent=node,
        critical=True,
    )
    masters_claim = (
        "Texas requires a master's degree from an accredited university (or institution) to be eligible for a "
        "Superintendent certificate."
    )
    await evaluator.verify(
        claim=masters_claim,
        node=masters_leaf,
        sources=_nonempty_urls(extracted.masters_degree.sources if extracted.masters_degree else []),
        additional_instruction="Verify on authoritative sources (e.g., TEA or Texas Administrative Code). "
                               "Allow minor wording variations like 'accredited institution' vs. 'accredited university'.",
    )


async def build_principal_credentials_checks(
    evaluator: Evaluator,
    parent,
    extracted: SuperintendentExtraction,
) -> None:
    """
    Build and execute checks for principal certificate and principal experience.
    """
    group = evaluator.add_parallel(
        id="Principal_Credentials",
        desc="The answer correctly identifies both the principal certification and experience requirements",
        parent=parent,
        critical=True,
    )

    # Principal certification requirement
    cert_seq = evaluator.add_sequential(
        id="Principal_Certification_Status",
        desc="The answer correctly identifies that a Texas Principal certificate (or equivalent) is required",
        parent=group,
        critical=True,
    )
    cert_exists = evaluator.add_custom_node(
        result=_has_statement_and_sources(extracted.principal_certification),
        id="principal_cert_present_with_sources",
        desc="Principal certificate requirement is stated in the answer and includes at least one supporting URL",
        parent=cert_seq,
        critical=True,
    )
    cert_leaf = evaluator.add_leaf(
        id="principal_cert_supported",
        desc="Holding a Texas Principal certificate (or recognized equivalent) is required for Superintendent certification (supported by cited sources)",
        parent=cert_seq,
        critical=True,
    )
    cert_claim = (
        "Eligibility for a Texas Superintendent certificate requires holding a valid Texas Principal certificate, "
        "or an equivalent/out-of-state principal credential recognized by Texas."
    )
    await evaluator.verify(
        claim=cert_claim,
        node=cert_leaf,
        sources=_nonempty_urls(extracted.principal_certification.sources if extracted.principal_certification else []),
        additional_instruction="Focus on official certification requirements (e.g., TEA, TAC). Recognize reasonable "
                               "equivalency phrasing for out-of-state credentials if explicitly allowed.",
    )

    # Principal experience requirement (two years as a school principal)
    exp_seq = evaluator.add_sequential(
        id="Principal_Experience_Requirement",
        desc="The answer correctly identifies that two years of experience as a school principal is required",
        parent=group,
        critical=True,
    )
    exp_exists = evaluator.add_custom_node(
        result=_has_statement_and_sources(extracted.principal_experience),
        id="principal_exp_present_with_sources",
        desc="Principal experience requirement is stated in the answer and includes at least one supporting URL",
        parent=exp_seq,
        critical=True,
    )
    exp_leaf = evaluator.add_leaf(
        id="principal_exp_supported",
        desc="Two years of experience as a school principal is required for Superintendent certification eligibility (supported by cited sources)",
        parent=exp_seq,
        critical=True,
    )
    exp_claim = (
        "Eligibility for a Texas Superintendent certificate includes a requirement of two years of experience as a "
        "school principal."
    )
    await evaluator.verify(
        claim=exp_claim,
        node=exp_leaf,
        sources=_nonempty_urls(extracted.principal_experience.sources if extracted.principal_experience else []),
        additional_instruction="Confirm that the official requirement specifies two years as a school principal. "
                               "Allow small textual variations like 'at least two years'.",
    )


async def build_salary_checks(
    evaluator: Evaluator,
    parent,
    extracted: SuperintendentExtraction,
) -> None:
    """
    Build and execute checks for the TASB 2025–26 median salary figure.
    """
    salary_seq = evaluator.add_sequential(
        id="Expected_Median_Salary",
        desc="The answer provides the median salary for Texas superintendents for the 2025-26 school year according to TASB",
        parent=parent,
        critical=False,
    )

    # Existence + at least one source (critical under non-critical parent)
    has_salary_and_src = evaluator.add_custom_node(
        result=bool(extracted.median_salary_2025_26 and extracted.median_salary_2025_26.strip())
               and bool(_nonempty_urls(extracted.median_salary_sources)),
        id="salary_present_with_sources",
        desc="Median salary (2025–26) is stated in the answer and includes at least one supporting URL",
        parent=salary_seq,
        critical=True,
    )

    # Evidence-backed correctness leaf (critical under non-critical parent)
    salary_leaf = evaluator.add_leaf(
        id="tasb_median_salary_supported",
        desc="The TASB 2025–26 survey median salary figure for Texas superintendents is correctly reported (supported by cited sources)",
        parent=salary_seq,
        critical=True,
    )

    salary_value = extracted.median_salary_2025_26 or ""
    salary_claim = (
        f"According to the Texas Association of School Boards (TASB) 2025–26 Superintendent Salary Survey, "
        f"the median salary for Texas school superintendents for the 2025–26 school year is {salary_value}."
    )
    await evaluator.verify(
        claim=salary_claim,
        node=salary_leaf,
        sources=_nonempty_urls(extracted.median_salary_sources),
        additional_instruction="Verify both that the figure is the median and that it is for the 2025–26 school year. "
                               "Prefer TASB sources. Allow minor currency formatting differences.",
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
    Evaluate an answer for the Texas Superintendent certification requirements and TASB median salary task.
    """
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

    # Extraction
    extracted: SuperintendentExtraction = await evaluator.extract(
        prompt=prompt_extract_superintendent_info(),
        template_class=SuperintendentExtraction,
        extraction_name="superintendent_requirements_and_salary",
    )

    # Build tree
    # 1) Critical certification requirements group (maps to Texas_Superintendent_Qualification_Assessment core)
    cert_group = evaluator.add_parallel(
        id="Certification_Requirements",
        desc="Evaluation of whether the answer correctly identifies the minimum education, principal certification, and principal experience requirements for Texas Superintendent certification",
        parent=root,
        critical=True,
    )
    await build_master_degree_checks(evaluator, cert_group, extracted)
    await build_principal_credentials_checks(evaluator, cert_group, extracted)

    # 2) Non-critical salary group (TASB 2025–26 median)
    await build_salary_checks(evaluator, root, extracted)

    # 3) Scoring anchor to allow partial credit when non-critical items fail (without negating correct critical items)
    evaluator.add_custom_node(
        result=True,
        id="scoring_anchor",
        desc="Anchor for partial credit when non-critical items are missing or incorrect",
        parent=root,
        critical=False,
    )

    # Return structured summary
    return evaluator.get_summary()