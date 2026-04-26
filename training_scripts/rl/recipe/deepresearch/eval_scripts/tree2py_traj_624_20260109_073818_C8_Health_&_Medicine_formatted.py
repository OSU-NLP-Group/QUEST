import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "pa_hospital_designations"
TASK_DESCRIPTION = """
Identify a hospital in Pennsylvania that currently holds all of the following designations simultaneously:
(1) Level I Trauma Center verification for adult patients,
(2) Comprehensive Stroke Center certification,
(3) Level IV Neonatal Intensive Care Unit (NICU) designation,
(4) American Nurses Credentialing Center (ANCC) Magnet Recognition, and
(5) teaching hospital status with documented affiliation to an LCME-accredited medical school.

Provide the hospital's full name, city location, and reference URLs verifying each of the five designations.
"""


class HospitalDesignationExtraction(BaseModel):
    hospital_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Normalize PA -> Pennsylvania when possible in extraction
    trauma_urls: List[str] = Field(default_factory=list)
    stroke_urls: List[str] = Field(default_factory=list)
    nicu_urls: List[str] = Field(default_factory=list)
    magnet_urls: List[str] = Field(default_factory=list)
    teaching_urls: List[str] = Field(default_factory=list)


def prompt_extract_hospital_designations() -> str:
    return """
    Extract the required hospital identification information and the verification URLs for each designation exactly as presented in the answer.

    Return a JSON object with the following fields:
    - hospital_name: The hospital's full official name.
    - city: The city where the hospital is located.
    - state: The U.S. state for the hospital location. If an abbreviation like "PA" is provided, return "PA". Do not infer anything not present in the answer; only extract what is explicitly written.
    - trauma_urls: An array of URL strings that verify the hospital's Level I Trauma Center status for adult patients.
    - stroke_urls: An array of URL strings that verify the Comprehensive Stroke Center certification/designation.
    - nicu_urls: An array of URL strings that verify the hospital operates a Level IV NICU.
    - magnet_urls: An array of URL strings that verify current ANCC Magnet Recognition.
    - teaching_urls: An array of URL strings that verify teaching hospital status with documented affiliation to an LCME-accredited medical school. Prefer sources that explicitly mention the affiliation and LCME accreditation.

    RULES:
    - Extract only URLs explicitly present in the answer (including markdown links).
    - If a field is missing, set it to null (for strings) or [] (for arrays).
    - Do not invent any information.
    """


def _normalize_state_str(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().lower()
    if s in {"pa", "pennsylvania"}:
        return "Pennsylvania"
    return state.strip()


async def _add_identity_subtree(evaluator: Evaluator, parent, data: HospitalDesignationExtraction) -> None:
    node = evaluator.add_parallel(
        id="hospital_identity_and_location",
        desc="Hospital identification details are provided and place the hospital in Pennsylvania",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.hospital_name and data.hospital_name.strip()),
        id="hospital_full_name_provided",
        desc="Hospital full name is provided",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.city and data.city.strip()),
        id="hospital_city_provided",
        desc="Hospital city location is provided",
        parent=node,
        critical=True,
    )

    normalized_state = _normalize_state_str(data.state)
    evaluator.add_custom_node(
        result=(normalized_state == "Pennsylvania"),
        id="hospital_in_pennsylvania",
        desc="Hospital is located in Pennsylvania",
        parent=node,
        critical=True,
    )


async def _add_trauma_subtree(evaluator: Evaluator, parent, data: HospitalDesignationExtraction) -> None:
    node = evaluator.add_parallel(
        id="level_i_adult_trauma_center",
        desc="Adult Level I Trauma Center requirement is met and verified",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.trauma_urls),
        id="trauma_reference_url_provided",
        desc="A valid reference URL is provided that documents the Level I adult trauma status",
        parent=node,
        critical=True,
    )

    status_leaf = evaluator.add_leaf(
        id="current_level_i_adult_trauma_status",
        desc="Hospital currently holds Level I Trauma Center verification/designation for adult patients",
        parent=node,
        critical=True,
    )
    trauma_claim = f"The hospital '{data.hospital_name or ''}' currently holds Level I Trauma Center verification/designation for adult patients."
    await evaluator.verify(
        claim=trauma_claim,
        node=status_leaf,
        sources=data.trauma_urls,
        additional_instruction="Verify that the provided page(s) explicitly indicate the hospital is a Level I adult trauma center (e.g., ACS Verified Trauma Centers list, state trauma system list, or official hospital documentation).",
    )

    verifier_leaf = evaluator.add_leaf(
        id="trauma_verified_by_acs_or_state",
        desc="The Level I adult trauma status is verified by ACS or a state authority",
        parent=node,
        critical=True,
    )
    verifier_claim = "The Level I adult trauma status is verified by either the American College of Surgeons (ACS) or a state authority (e.g., Pennsylvania Department of Health)."
    await evaluator.verify(
        claim=verifier_claim,
        node=verifier_leaf,
        sources=data.trauma_urls,
        additional_instruction="Check whether the verification source is ACS (e.g., facs.org) or an official state authority page listing trauma levels. If the URL is from the hospital, it must clearly state ACS verification or state designation with level.",
    )


async def _add_stroke_subtree(evaluator: Evaluator, parent, data: HospitalDesignationExtraction) -> None:
    node = evaluator.add_parallel(
        id="comprehensive_stroke_center",
        desc="Comprehensive Stroke Center requirement is met and verified",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.stroke_urls),
        id="stroke_reference_url_provided",
        desc="A valid reference URL is provided that documents the Comprehensive Stroke Center certification/designation",
        parent=node,
        critical=True,
    )

    status_leaf = evaluator.add_leaf(
        id="current_comprehensive_stroke_certification",
        desc="Hospital currently holds Comprehensive Stroke Center certification/designation",
        parent=node,
        critical=True,
    )
    stroke_claim = f"The hospital '{data.hospital_name or ''}' currently holds Comprehensive Stroke Center certification/designation."
    await evaluator.verify(
        claim=stroke_claim,
        node=status_leaf,
        sources=data.stroke_urls,
        additional_instruction="Verify that the page(s) explicitly show Comprehensive Stroke Center status, typically from The Joint Commission, DNV Healthcare, or an official state designation.",
    )

    recognized_leaf = evaluator.add_leaf(
        id="stroke_certifier_recognized",
        desc="Certification/designation is from a recognized body (Joint Commission, DNV, or state designation)",
        parent=node,
        critical=True,
    )
    recognized_claim = "The Comprehensive Stroke Center certification/designation is issued by a recognized body such as The Joint Commission, DNV Healthcare, or an official state designation."
    await evaluator.verify(
        claim=recognized_claim,
        node=recognized_leaf,
        sources=data.stroke_urls,
        additional_instruction="Check the source domain/content: accept jointcommission.org, dnv.com, or official state pages. Hospital pages are acceptable if they explicitly reference a recognized certifier.",
    )


async def _add_nicu_subtree(evaluator: Evaluator, parent, data: HospitalDesignationExtraction) -> None:
    node = evaluator.add_parallel(
        id="level_iv_nicu",
        desc="Level IV NICU requirement is met and verified",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.nicu_urls),
        id="nicu_reference_url_provided",
        desc="A valid reference URL is provided that documents the Level IV NICU designation/operation",
        parent=node,
        critical=True,
    )

    nicu_leaf = evaluator.add_leaf(
        id="level_iv_nicu_operated",
        desc="Hospital operates a Level IV NICU (as designated/identified by official or hospital documentation)",
        parent=node,
        critical=True,
    )
    nicu_claim = f"The hospital '{data.hospital_name or ''}' operates a Level IV NICU."
    await evaluator.verify(
        claim=nicu_claim,
        node=nicu_leaf,
        sources=data.nicu_urls,
        additional_instruction="Confirm that the provided page(s) explicitly state Level IV NICU (e.g., hospital pages, state designation lists, or authoritative neonatal care documentation).",
    )


async def _add_magnet_subtree(evaluator: Evaluator, parent, data: HospitalDesignationExtraction) -> None:
    node = evaluator.add_parallel(
        id="ancc_magnet_recognition",
        desc="ANCC Magnet Recognition requirement is met and verified",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.magnet_urls),
        id="magnet_reference_url_provided",
        desc="A valid reference URL is provided that documents current ANCC Magnet Recognition",
        parent=node,
        critical=True,
    )

    magnet_leaf = evaluator.add_leaf(
        id="current_magnet_recognition",
        desc="Hospital currently holds ANCC Magnet Recognition",
        parent=node,
        critical=True,
    )
    magnet_claim = f"The hospital '{data.hospital_name or ''}' currently holds ANCC Magnet Recognition."
    await evaluator.verify(
        claim=magnet_claim,
        node=magnet_leaf,
        sources=data.magnet_urls,
        additional_instruction="Confirm the page(s) explicitly indicate current Magnet Recognition (e.g., official ANCC Magnet designated list, hospital announcement with clear Magnet status).",
    )


async def _add_teaching_subtree(evaluator: Evaluator, parent, data: HospitalDesignationExtraction) -> None:
    node = evaluator.add_parallel(
        id="teaching_hospital_lcme_affiliation",
        desc="Teaching hospital status with LCME-accredited medical school affiliation is met and verified",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.teaching_urls),
        id="teaching_affiliation_reference_url_provided",
        desc="A valid reference URL is provided that documents the teaching hospital status and LCME-accredited medical school affiliation",
        parent=node,
        critical=True,
    )

    teach_leaf = evaluator.add_leaf(
        id="teaching_hospital_with_lcme_affiliation",
        desc="Hospital is a teaching hospital with documented affiliation to an LCME-accredited medical school",
        parent=node,
        critical=True,
    )
    teach_claim = f"The hospital '{data.hospital_name or ''}' is a teaching hospital with documented affiliation to an LCME-accredited medical school."
    await evaluator.verify(
        claim=teach_claim,
        node=teach_leaf,
        sources=data.teaching_urls,
        additional_instruction="Verify that the provided page(s) document both (a) the hospital's teaching/clinical affiliation and (b) that the affiliated medical school is LCME-accredited (e.g., LCME or school page confirming accreditation). Multiple sources may collectively satisfy these.",
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
        task_description="Evaluate whether the response identifies a single Pennsylvania hospital that currently holds all five required designations and provides the required identifying info and verifying URLs",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    extracted = await evaluator.extract(
        prompt=prompt_extract_hospital_designations(),
        template_class=HospitalDesignationExtraction,
        extraction_name="hospital_designations",
    )

    # Create a critical top-level task node under the (non-critical) framework root
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Evaluate whether the response identifies a single Pennsylvania hospital that currently holds all five required designations and provides the required identifying info and verifying URLs",
        parent=root,
        critical=True,
    )

    await _add_identity_subtree(evaluator, task_root, extracted)
    await _add_trauma_subtree(evaluator, task_root, extracted)
    await _add_stroke_subtree(evaluator, task_root, extracted)
    await _add_nicu_subtree(evaluator, task_root, extracted)
    await _add_magnet_subtree(evaluator, task_root, extracted)
    await _add_teaching_subtree(evaluator, task_root, extracted)

    return evaluator.get_summary()