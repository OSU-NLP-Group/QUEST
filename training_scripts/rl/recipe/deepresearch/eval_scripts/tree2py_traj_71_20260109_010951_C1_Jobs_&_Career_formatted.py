import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "dod8570_entry_level_cert"
TASK_DESCRIPTION = (
    "I'm interested in pursuing a career in cybersecurity and want to start with an entry-level certification that is DoD 8570 approved. "
    "Can you identify one such certification and provide the following information: (1) confirmation that it is classified as entry-level, "
    "(2) the current exam cost, and (3) salary information for professionals who hold this certification? Please include reference URLs for verification."
)


class CertificationExtraction(BaseModel):
    cert_name: Optional[str] = None
    entry_level_label: Optional[str] = None
    entry_level_urls: List[str] = Field(default_factory=list)
    dod_8570_urls: List[str] = Field(default_factory=list)
    exam_cost: Optional[str] = None
    exam_cost_urls: List[str] = Field(default_factory=list)
    salary_info: Optional[str] = None
    salary_urls: List[str] = Field(default_factory=list)


def prompt_extract_certification_data() -> str:
    return """
    Extract exactly one specific cybersecurity certification mentioned in the answer that the user is advised to pursue.
    If multiple certifications are mentioned, pick the first one that is relevant to entry-level and/or DoD 8570 approval.
    Return a JSON object with the following fields:

    - cert_name: The exact certification name/designation as stated in the answer (e.g., "CompTIA Security+" or "ISC2 Certified in Cybersecurity (CC)").
    - entry_level_label: The statement or phrasing from the answer that indicates this certification is "entry-level" (or synonyms like "foundational", "introductory", "associate-level"), if present; otherwise null.
    - entry_level_urls: An array of URLs cited in the answer that support the entry-level classification of the certification. Extract only actual URLs from the answer; do not invent. If none, return [].
    - dod_8570_urls: An array of URLs cited in the answer that support DoD 8570 (or DoD 8140 baseline equivalency) approval of the certification. Extract only actual URLs. If none, return [].
    - exam_cost: The current exam cost as stated in the answer (string). If the answer provides a range or different regional prices, include the phrase exactly as given. If missing, null.
    - exam_cost_urls: An array of URLs cited in the answer that support the stated exam cost. Extract only actual URLs. If none, return [].
    - salary_info: Salary information (e.g., "average $95,000", "median $80,000", or a range) for professionals who hold the certification, exactly as stated in the answer. If missing, null.
    - salary_urls: An array of URLs cited in the answer that support the salary information. Extract only actual URLs. If none, return [].

    SPECIAL RULES FOR URL SOURCES EXTRACTION:
    - Extract only URLs explicitly present in the answer text (plain URLs or markdown links). If the answer mentions a source without a concrete URL, do not fabricate; return [] for that field.
    - Ensure each URL is complete; if missing a protocol, prepend http://.
    - Do not include the same URL in multiple fields unless the answer explicitly uses it for that specific claim.

    IMPORTANT:
    - Do not add or infer information. Only extract exactly what is present in the answer.
    - If any field is not present in the answer, return null (for scalar fields) or [] (for URL arrays).
    """


def _has_nonempty_string(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_urls(urls: List[str]) -> bool:
    return bool(urls and len(urls) > 0)


async def _verify_entry_level(
    evaluator: Evaluator,
    parent,
    data: CertificationExtraction,
) -> None:
    node = evaluator.add_sequential(
        id="EntryLevelClassificationWithURL",
        desc="Confirms the certification is classified as entry-level for cybersecurity professionals AND provides a supporting reference URL.",
        parent=parent,
        critical=True,
    )

    url_present = evaluator.add_custom_node(
        result=_has_urls(data.entry_level_urls),
        id="EntryLevelClassification_URL_Provided",
        desc="At least one supporting URL is provided for entry-level classification.",
        parent=node,
        critical=True,
    )

    supported_leaf = evaluator.add_leaf(
        id="EntryLevelClassification_Supported",
        desc="Entry-level classification is supported by the provided URL(s).",
        parent=node,
        critical=True,
    )
    claim = f"The certification '{data.cert_name or ''}' is classified as entry-level for cybersecurity professionals."
    await evaluator.verify(
        claim=claim,
        node=supported_leaf,
        sources=data.entry_level_urls,
        additional_instruction=(
            "Verify that the provided page(s) explicitly indicate the certification is entry-level, foundational, introductory, "
            "associate-level, or otherwise suitable for beginners. Allow reasonable synonyms. The support must be explicit on the page."
        ),
    )


async def _verify_dod_8570(
    evaluator: Evaluator,
    parent,
    data: CertificationExtraction,
) -> None:
    node = evaluator.add_sequential(
        id="DoD8570ApprovalWithURL",
        desc="Confirms the certification is DoD 8570 approved AND provides a supporting reference URL.",
        parent=parent,
        critical=True,
    )

    url_present = evaluator.add_custom_node(
        result=_has_urls(data.dod_8570_urls),
        id="DoD8570Approval_URL_Provided",
        desc="At least one supporting URL is provided for DoD 8570 approval.",
        parent=node,
        critical=True,
    )

    supported_leaf = evaluator.add_leaf(
        id="DoD8570Approval_Supported",
        desc="DoD 8570 approval is supported by the provided URL(s).",
        parent=node,
        critical=True,
    )
    claim = f"The certification '{data.cert_name or ''}' is approved/listed under DoD 8570 (or DoD 8140 baseline equivalency)."
    await evaluator.verify(
        claim=claim,
        node=supported_leaf,
        sources=data.dod_8570_urls,
        additional_instruction=(
            "Confirm that the page(s) list the certification within DoD 8570 baseline categories (e.g., IAT, IAM, CSSP) or reference DoD 8140 "
            "as the updated framework encompassing 8570. If the page uses DoD 8140 to list the certification in equivalent baseline roles, "
            "consider it valid support for 8570 approval."
        ),
    )


async def _verify_exam_cost(
    evaluator: Evaluator,
    parent,
    data: CertificationExtraction,
) -> None:
    node = evaluator.add_sequential(
        id="CurrentExamCostWithURL",
        desc="Provides the current exam cost AND provides a supporting reference URL.",
        parent=parent,
        critical=True,
    )

    url_present = evaluator.add_custom_node(
        result=_has_urls(data.exam_cost_urls),
        id="CurrentExamCost_URL_Provided",
        desc="At least one supporting URL is provided for the exam cost.",
        parent=node,
        critical=True,
    )

    supported_leaf = evaluator.add_leaf(
        id="CurrentExamCost_Supported",
        desc="The stated current exam cost is supported by the provided URL(s).",
        parent=node,
        critical=True,
    )
    cost_string = data.exam_cost or ""
    claim = f"The current exam cost for '{data.cert_name or ''}' is stated as '{cost_string}'."
    await evaluator.verify(
        claim=claim,
        node=supported_leaf,
        sources=data.exam_cost_urls,
        additional_instruction=(
            "Verify that the page(s) explicitly list the exam price or voucher cost matching or reasonably aligning with the stated value. "
            "Allow minor regional or currency variations and ranges if the page presents them. The key is that the page supports the claimed cost."
        ),
    )


async def _verify_salary_info(
    evaluator: Evaluator,
    parent,
    data: CertificationExtraction,
) -> None:
    node = evaluator.add_sequential(
        id="SalaryInfoWithURL",
        desc="Provides salary information (floor, median, or average) for professionals holding the certification AND provides a supporting reference URL.",
        parent=parent,
        critical=True,
    )

    url_present = evaluator.add_custom_node(
        result=_has_urls(data.salary_urls),
        id="SalaryInfo_URL_Provided",
        desc="At least one supporting URL is provided for the salary information.",
        parent=node,
        critical=True,
    )

    supported_leaf = evaluator.add_leaf(
        id="SalaryInfo_Supported",
        desc="The stated salary information is supported by the provided URL(s).",
        parent=node,
        critical=True,
    )
    salary_str = data.salary_info or ""
    claim = f"Salary information for professionals who hold '{data.cert_name or ''}' is reported as '{salary_str}'."
    await evaluator.verify(
        claim=claim,
        node=supported_leaf,
        sources=data.salary_urls,
        additional_instruction=(
            "Confirm that the page(s) present salary data (e.g., median, average, or a range) consistent with the stated information. "
            "Allow reasonable differences in formatting, rounding, currency, and timeframes if the substance is supported."
        ),
    )


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_certification_data(),
        template_class=CertificationExtraction,
        extraction_name="certification_extraction",
    )

    task_main = evaluator.add_sequential(
        id="EntryLevelCybersecurityCertification",
        desc="Identify one DoD 8570 approved entry-level cybersecurity certification and provide entry-level confirmation, current exam cost, and salary information, each verifiable via reference URLs.",
        parent=root,
        critical=True,
    )

    cert_identified = evaluator.add_custom_node(
        result=_has_nonempty_string(extracted.cert_name),
        id="CertificationIdentified",
        desc="The response clearly identifies one specific cybersecurity certification (name/designation).",
        parent=task_main,
        critical=True,
    )

    required_attrs = evaluator.add_parallel(
        id="RequiredAttributesAndEvidence",
        desc="Provide required attributes and evidence (URLs) for the identified certification.",
        parent=task_main,
        critical=True,
    )

    await _verify_entry_level(evaluator, required_attrs, extracted)
    await _verify_dod_8570(evaluator, required_attrs, extracted)
    await _verify_exam_cost(evaluator, required_attrs, extracted)
    await _verify_salary_info(evaluator, required_attrs, extracted)

    return evaluator.get_summary()