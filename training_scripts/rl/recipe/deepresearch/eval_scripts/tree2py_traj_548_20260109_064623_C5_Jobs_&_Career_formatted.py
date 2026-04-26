import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "cert_remote_no_degree_under_1000"
TASK_DESCRIPTION = (
    "You are helping a professional who wants to transition into a remote career but does not have a bachelor's degree. "
    "They are looking for a professional certification that meets the following requirements:\n\n"
    "1. Does not require a bachelor's degree as a prerequisite for eligibility\n"
    "2. Does not require more than 2 years of professional work experience as a prerequisite\n"
    "3. Has a total initial cost (including both exam fee and any required training) of less than $1,000 USD\n"
    "4. Can have its preparation and training completed within 6 months of part-time study\n"
    "5. Any required instructor-led training component can be completed in 3 days or less\n"
    "6. Is explicitly recognized in published sources from 2025-2026 as suitable for remote work opportunities\n"
    "7. Requires renewal every 2 or 3 years (not annually or longer than 3 years)\n"
    "8. Provides PDUs (Professional Development Units) or equivalent continuing education credits that can be applied toward other professional certifications\n\n"
    "Identify one professional certification that satisfies all eight requirements. Provide the official name of the certification, and for each of the eight requirements, provide a reference URL that verifies the certification meets that specific criterion."
)


class CertificationExtraction(BaseModel):
    certification_name: Optional[str] = None

    no_degree_urls: List[str] = Field(default_factory=list)
    experience_urls: List[str] = Field(default_factory=list)
    cost_urls: List[str] = Field(default_factory=list)
    completion_urls: List[str] = Field(default_factory=list)
    training_duration_urls: List[str] = Field(default_factory=list)
    remote_work_urls: List[str] = Field(default_factory=list)
    renewal_urls: List[str] = Field(default_factory=list)
    pdu_urls: List[str] = Field(default_factory=list)


def prompt_extract_certification_evidence() -> str:
    return (
        "Extract the certification and the specific reference URLs the answer uses to support each of the eight criteria.\n"
        "Return a JSON object with the following fields:\n"
        "- certification_name: official name of the proposed certification\n"
        "- no_degree_urls: array of URLs that verify no bachelor's degree is required\n"
        "- experience_urls: array of URLs that verify no more than 2 years of professional experience is required\n"
        "- cost_urls: array of URLs that verify the total initial cost (exam + any required training) is under $1,000 USD\n"
        "- completion_urls: array of URLs that verify preparation/training can be completed within 6 months with part-time study\n"
        "- training_duration_urls: array of URLs that verify any required instructor-led training is 3 days or less (or that none is required)\n"
        "- remote_work_urls: array of URLs (published in 2025 or 2026) that explicitly recognize the certification as suitable for remote work opportunities\n"
        "- renewal_urls: array of URLs that verify renewal every 2 or 3 years\n"
        "- pdu_urls: array of URLs that verify the certification (or its training) provides PDUs or equivalent CE credits that can be applied toward other certifications\n\n"
        "Rules:\n"
        "1) Extract only URLs explicitly present in the answer (plain or markdown links). If a URL is missing a protocol, prepend http://.\n"
        "2) Map each URL to the criterion it supports based on the answer text. If the answer lists sources without explicit mapping, "
        "assign each URL to the most relevant criterion it appears to support.\n"
        "3) If no URL is provided for a criterion, return an empty list for that criterion.\n"
        "4) Do not invent or infer URLs.\n"
    )


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0 and any((u or "").strip() for u in urls)


async def verify_certification_requirements(
    evaluator: Evaluator,
    root_node,
    extracted: CertificationExtraction,
) -> None:
    cert_name = (extracted.certification_name or "").strip()
    cert_label = cert_name if cert_name else "the certification"

    # Main certification identification node (critical)
    cert_node = evaluator.add_parallel(
        id="Certification_Identification",
        desc="Identify a professional certification that meets all specified criteria for career changers seeking remote work opportunities",
        parent=root_node,
        critical=True,
    )

    # Certification name provided (critical)
    evaluator.add_custom_node(
        result=bool(cert_name),
        id="Certification_Name_Provided",
        desc="The official name of the certification is provided",
        parent=cert_node,
        critical=True,
    )

    # Eligibility requirements (critical group)
    eligibility_node = evaluator.add_parallel(
        id="Eligibility_Requirements",
        desc="Certification eligibility and prerequisite requirements are met with supporting evidence",
        parent=cert_node,
        critical=True,
    )

    # No Degree Requirement - URL existence check
    evaluator.add_custom_node(
        result=_has_urls(extracted.no_degree_urls),
        id="No_Degree_Requirement_URL_Provided",
        desc="Reference URL is provided for the no-degree requirement criterion",
        parent=eligibility_node,
        critical=True,
    )
    # No Degree Requirement - verification leaf
    no_degree_leaf = evaluator.add_leaf(
        id="No_Degree_Requirement",
        desc="Certification does not require a bachelor's degree as a prerequisite, and a reference URL is provided verifying this criterion",
        parent=eligibility_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The certification '{cert_label}' does not require a bachelor's degree as a prerequisite for eligibility.",
        node=no_degree_leaf,
        sources=extracted.no_degree_urls,
        additional_instruction=(
            "Check eligibility/prerequisites pages. If multiple pathways exist, it is sufficient that at least one official pathway "
            "allows eligibility without a bachelor's degree."
        ),
    )

    # Experience Requirement - URL existence check
    evaluator.add_custom_node(
        result=_has_urls(extracted.experience_urls),
        id="Experience_Requirement_URL_Provided",
        desc="Reference URL is provided for the work experience requirement criterion",
        parent=eligibility_node,
        critical=True,
    )
    # Experience Requirement - verification leaf
    exp_leaf = evaluator.add_leaf(
        id="Experience_Requirement",
        desc="Certification requires no more than 2 years of professional work experience as a prerequisite, and a reference URL is provided verifying this criterion",
        parent=eligibility_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The certification '{cert_label}' requires no more than 2 years of professional work experience for eligibility "
            "(i.e., at most 24 months; zero or no experience is acceptable)."
        ),
        node=exp_leaf,
        sources=extracted.experience_urls,
        additional_instruction=(
            "Check official eligibility. If multiple options exist, it suffices that at least one option meets the <= 2 years threshold."
        ),
    )

    # Cost Requirements (critical group)
    cost_node = evaluator.add_parallel(
        id="Cost_Requirements",
        desc="Total initial certification costs including exam and required training are acceptable with supporting evidence",
        parent=cert_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_urls(extracted.cost_urls),
        id="Total_Cost_Under_1000_URL_Provided",
        desc="Reference URL(s) are provided for the total initial cost criterion",
        parent=cost_node,
        critical=True,
    )
    cost_leaf = evaluator.add_leaf(
        id="Total_Cost_Under_1000",
        desc="Total initial cost (exam fee plus any required training) is less than $1,000 USD, and a reference URL is provided verifying this criterion",
        parent=cost_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The total initial cost for '{cert_label}'—including the exam fee plus any required training—is under $1,000 USD."
        ),
        node=cost_leaf,
        sources=extracted.cost_urls,
        additional_instruction=(
            "Use official pricing pages and any required training pages. If training is optional (not required for eligibility), "
            "do not include optional costs in the total. If multiple fees apply, sum them and ensure the total remains below $1,000 USD. "
            "If currency is non-USD, a reasonable contemporary conversion is acceptable."
        ),
    )

    # Time Requirements (critical group)
    time_node = evaluator.add_parallel(
        id="Time_Requirements",
        desc="Time required to complete certification training and preparation is acceptable with supporting evidence",
        parent=cert_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_urls(extracted.completion_urls),
        id="Completion_Within_6_Months_URL_Provided",
        desc="Reference URL(s) are provided for preparation within 6 months criterion",
        parent=time_node,
        critical=True,
    )
    completion_leaf = evaluator.add_leaf(
        id="Completion_Within_6_Months",
        desc="Certification preparation and training can be completed within 6 months of part-time study, and a reference URL is provided verifying this criterion",
        parent=time_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The preparation and training for '{cert_label}' can be completed within 6 months of part-time study."
        ),
        node=completion_leaf,
        sources=extracted.completion_urls,
        additional_instruction=(
            "Consider typical vendor guidance or reputable training provider estimates. Treat part-time as approximately 8–12 hours/week; "
            "self-paced programs that are commonly completed within 6 months also satisfy the criterion."
        ),
    )

    evaluator.add_custom_node(
        result=_has_urls(extracted.training_duration_urls),
        id="Training_Duration_URL_Provided",
        desc="Reference URL(s) are provided for instructor-led training duration criterion",
        parent=time_node,
        critical=True,
    )
    training_leaf = evaluator.add_leaf(
        id="Training_Duration",
        desc="Any required instructor-led training component can be completed in 3 days or less, and a reference URL is provided verifying this criterion",
        parent=time_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"For '{cert_label}', any required instructor-led training component is either not required or can be completed in 3 days or less."
        ),
        node=training_leaf,
        sources=extracted.training_duration_urls,
        additional_instruction=(
            "Check whether instructor-led training is required for eligibility. If it is required, confirm the duration is 3 days (or fewer). "
            "If no instructor-led training is required, the criterion is satisfied."
        ),
    )

    # Career Applicability (critical group)
    career_node = evaluator.add_parallel(
        id="Career_Applicability",
        desc="Certification's relevance to remote work opportunities with supporting evidence",
        parent=cert_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_urls(extracted.remote_work_urls),
        id="Remote_Work_Suitable_URL_Provided",
        desc="Reference URL(s) are provided for remote work suitability (2025-2026) criterion",
        parent=career_node,
        critical=True,
    )
    remote_leaf = evaluator.add_leaf(
        id="Remote_Work_Suitable",
        desc="Certification is explicitly recognized in published sources from 2025-2026 as suitable for remote work opportunities, and a reference URL is provided verifying this criterion",
        parent=career_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"Published sources from 2025 or 2026 explicitly recognize '{cert_label}' as suitable for remote work opportunities."
        ),
        node=remote_leaf,
        sources=extracted.remote_work_urls,
        additional_instruction=(
            "Verify that the page shows a publication or update date in 2025 or 2026 and that it explicitly indicates suitability for remote roles, "
            "remote job prospects, or remote work opportunities associated with this certification."
        ),
    )

    # Renewal Requirements (critical group)
    renewal_node = evaluator.add_parallel(
        id="Renewal_Requirements",
        desc="Certification renewal cycle and maintenance requirements with supporting evidence",
        parent=cert_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_urls(extracted.renewal_urls),
        id="Renewal_Cycle_URL_Provided",
        desc="Reference URL(s) are provided for renewal cycle criterion",
        parent=renewal_node,
        critical=True,
    )
    renewal_leaf = evaluator.add_leaf(
        id="Renewal_Cycle",
        desc="Certification requires renewal every 2 or 3 years (not annually or longer than 3 years), and a reference URL is provided verifying this criterion",
        parent=renewal_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The certification '{cert_label}' requires renewal every 2 or 3 years (i.e., not annually and not longer than 3 years)."
        ),
        node=renewal_leaf,
        sources=extracted.renewal_urls,
        additional_instruction=(
            "Check the official maintenance/renewal policy. Accept a renewal interval of exactly 2 years or exactly 3 years; other intervals do not satisfy."
        ),
    )

    # Credit Recognition (critical group)
    credit_node = evaluator.add_parallel(
        id="Credit_Recognition",
        desc="Continuing education credit recognition from the certification with supporting evidence",
        parent=cert_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_urls(extracted.pdu_urls),
        id="Provides_PDUs_or_Equivalent_URL_Provided",
        desc="Reference URL(s) are provided for PDUs/CE credits criterion",
        parent=credit_node,
        critical=True,
    )
    pdu_leaf = evaluator.add_leaf(
        id="Provides_PDUs_or_Equivalent",
        desc="Certification or its training provides PDUs (Professional Development Units) or equivalent continuing education credits applicable to other professional certifications, and a reference URL is provided verifying this criterion",
        parent=credit_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"'{cert_label}' provides PDUs or equivalent continuing education credits that can be applied toward other professional certifications."
        ),
        node=pdu_leaf,
        sources=extracted.pdu_urls,
        additional_instruction=(
            "Look for mentions of PDUs, CEUs, CPE, CPD, or similar recognized credits. The source must indicate applicability to other programs "
            "(e.g., PMI PDUs, ISACA CPE, CompTIA CE, etc.)."
        ),
    )


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
        prompt=prompt_extract_certification_evidence(),
        template_class=CertificationExtraction,
        extraction_name="certification_evidence",
    )

    await verify_certification_requirements(evaluator, root, extracted)

    return evaluator.get_summary()