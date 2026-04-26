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
TASK_ID = "rsv_vaccination_assessment"
TASK_DESCRIPTION = """
A 68-year-old patient with asthma and type 2 diabetes (currently using insulin) is considering RSV vaccination. It is currently September, and the patient lives in the United States.

Based on current CDC RSV vaccination guidelines and pharmacy services, please provide:

1. An eligibility determination: Is this patient eligible to receive the RSV vaccine according to CDC recommendations?
2. Risk factor identification: Which of the patient's health conditions qualify as CDC-recognized risk factors that make adults aged 50-74 eligible for RSV vaccination?
3. Pharmacy service options: Identify at least one major U.S. pharmacy chain where this patient can receive RSV vaccination, and confirm the pharmacy's age requirements accommodate this patient.
4. Timing assessment: Is September an appropriate time for RSV vaccination based on CDC timing recommendations?
5. Co-administration guidance: Can the RSV vaccine be given at the same time as other vaccines such as flu or COVID-19?

Support your answer with references to official CDC guidelines and pharmacy service documentation.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RSVAssessmentExtraction(BaseModel):
    """Structured extraction from the agent's answer."""
    eligibility_conclusion: Optional[str] = None
    risk_factors: List[str] = Field(default_factory=list)
    pharmacy_name: Optional[str] = None
    pharmacy_url: Optional[str] = None
    cdc_refs: List[str] = Field(default_factory=list)
    timing_statement: Optional[str] = None
    coadmin_statement: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_rsv_assessment() -> str:
    return """
    Extract the structured information about RSV vaccination from the answer. Return a JSON object with the following fields:

    - eligibility_conclusion: The explicit conclusion stated in the answer about the patient's eligibility for RSV vaccination (e.g., "eligible", "not eligible", or the exact phrase the answer uses). If the answer does not state a clear conclusion, return null.
    - risk_factors: A list of the risk factors explicitly identified in the answer as relevant for RSV vaccination eligibility (e.g., "asthma", "diabetes", "insulin-treated diabetes"). Include only the factors mentioned in the answer.
    - pharmacy_name: The name of a major U.S. pharmacy chain identified in the answer where RSV vaccination can be received. If multiple are mentioned, extract only the first one. If none are mentioned, return null.
    - pharmacy_url: The URL to the official pharmacy service page referenced in the answer that discusses RSV vaccination or immunization services for the selected chain. Extract only if the URL is explicitly present in the answer; otherwise return null.
    - cdc_refs: An array of URLs to official CDC pages cited in the answer that support eligibility, risk factors, timing, or co-administration guidance. Extract only URLs explicitly present in the answer. If none, return an empty array.
    - timing_statement: The statement in the answer about when RSV vaccination should be given (e.g., "late summer to early fall", "September is appropriate"), extracted verbatim. If not mentioned, return null.
    - coadmin_statement: The statement in the answer about whether RSV vaccine can be administered with other vaccines (e.g., flu or COVID-19), extracted verbatim. If not mentioned, return null.

    IMPORTANT:
    - Extract only information explicitly present in the answer; do not infer or invent.
    - For URLs, include the full URL as presented (plain or markdown). If a URL lacks protocol, prepend http://.
    - If a field is not present in the answer, set it to null (for string fields) or [] (for lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
MAJOR_PHARMACY_DOMAINS = [
    "walgreens.com",
    "cvs.com",
    "riteaid.com",
    "walmart.com",
    "kroger.com",
    "publix.com",
    "costco.com",
    "safeway.com",
    "meijer.com",
    "hy-vee.com",
    "heb.com",
    "albertsons.com",
    "giantfood.com",
    "wegmans.com",
    "shoprite.com",
    "harristeeter.com",
    "samsclub.com",
    "target.com",  # often via CVS in Target
]

def is_major_pharmacy_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    lower = url.lower()
    return any(d in lower for d in MAJOR_PHARMACY_DOMAINS)

def normalize_eligibility(conclusion: Optional[str]) -> Optional[str]:
    if not conclusion:
        return None
    c = conclusion.strip().lower()
    if "not eligible" in c or "ineligible" in c or "not recommended" in c:
        return "not eligible"
    if "eligible" in c or "recommended" in c:
        # guard against "not eligible" already handled above
        return "eligible"
    return None


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_rsv_assessment(
    evaluator: Evaluator,
    extracted: RSVAssessmentExtraction
) -> None:
    """
    Build the verification tree according to the rubric and perform verifications.
    """
    # Top-level critical assessment node under the framework root
    assessment_node = evaluator.add_parallel(
        id="RSV_Vaccination_Assessment",
        desc="Assess RSV vaccination eligibility, risk factors, pharmacy service options, timing, co-administration, and required documentation for the given patient scenario.",
        parent=evaluator.root,
        critical=True
    )

    # ----------------------- Required References ------------------------- #
    refs_node = evaluator.add_parallel(
        id="Required_References",
        desc="Provides supporting references as requested (official CDC guidance and pharmacy service documentation).",
        parent=assessment_node,
        critical=True
    )

    cdc_urls = extracted.cdc_refs or []
    cdc_has_official = any(isinstance(u, str) and "cdc.gov" in u.lower() for u in cdc_urls)

    evaluator.add_custom_node(
        result=cdc_has_official,
        id="CDC_Reference_Provided",
        desc="Includes at least one official CDC reference supporting the eligibility/risk factor/timing/co-administration claims.",
        parent=refs_node,
        critical=True
    )

    pharmacy_doc_present = is_major_pharmacy_url(extracted.pharmacy_url)
    evaluator.add_custom_node(
        result=pharmacy_doc_present,
        id="Pharmacy_Documentation_Reference_Provided",
        desc="Includes at least one official pharmacy service documentation reference supporting pharmacy availability and/or age requirement claims for the chosen pharmacy chain.",
        parent=refs_node,
        critical=True
    )

    # Record references info in summary
    evaluator.add_custom_info(
        {"cdc_refs_count": len(cdc_urls), "cdc_refs": cdc_urls},
        info_type="reference_stats",
        info_name="cdc_reference_overview"
    )
    evaluator.add_custom_info(
        {"pharmacy_name": extracted.pharmacy_name, "pharmacy_url": extracted.pharmacy_url},
        info_type="reference_stats",
        info_name="pharmacy_reference_overview"
    )

    # ----------------------- Eligibility Determination ------------------- #
    eligibility_node = evaluator.add_leaf(
        id="Eligibility_Determination",
        desc="Determines whether the 68-year-old patient is eligible for RSV vaccination under CDC guidance for adults with qualifying risk factors, and states the eligibility conclusion.",
        parent=assessment_node,
        critical=True
    )

    normalized_elig = normalize_eligibility(extracted.eligibility_conclusion)
    # Build claim aligned with the answer's stated conclusion (if present)
    if normalized_elig == "eligible":
        elig_claim = (
            "Under CDC adult RSV vaccine recommendations, a 68-year-old with qualifying risk factors "
            "(e.g., chronic lung disease such as asthma, and diabetes mellitus) is eligible to receive the RSV vaccine."
        )
    elif normalized_elig == "not eligible":
        elig_claim = (
            "Under CDC adult RSV vaccine recommendations, a 68-year-old with asthma and insulin-treated diabetes "
            "is not eligible to receive the RSV vaccine."
        )
    else:
        # If unclear, default to the guideline-based positive eligibility claim for older adult with risk factors
        elig_claim = (
            "Under CDC adult RSV vaccine recommendations, a 68-year-old with qualifying risk factors "
            "(e.g., chronic lung disease such as asthma, and diabetes mellitus) is eligible to receive the RSV vaccine."
        )

    await evaluator.verify(
        claim=elig_claim,
        node=eligibility_node,
        sources=cdc_urls,
        additional_instruction=(
            "Use the provided CDC RSV adult vaccination guidance. Confirm whether CDC recommends RSV vaccination "
            "for older adults (e.g., ≥60 years) when risk factors such as chronic lung disease (including asthma) "
            "and diabetes mellitus are present, or recommends routinely for ≥75 years. Judge the stated conclusion "
            "for a 68-year-old with asthma and insulin-treated diabetes accordingly."
        ),
    )

    # ----------------------- Risk Factor Identification ------------------ #
    risk_node = evaluator.add_parallel(
        id="Risk_Factor_Identification",
        desc="Identifies which of the patient's listed conditions are CDC-recognized risk factors for RSV vaccination eligibility in adults aged 50–74/60–74.",
        parent=assessment_node,
        critical=True
    )

    asthma_leaf = evaluator.add_leaf(
        id="Asthma_As_Risk_Factor",
        desc="Correctly identifies asthma/chronic respiratory disease as a qualifying CDC risk factor for older adults.",
        parent=risk_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "CDC lists chronic lung disease—including asthma—as a risk factor for severe RSV disease in older adults, "
            "which is used to identify those at increased risk for whom RSV vaccination is recommended or may be offered."
        ),
        node=asthma_leaf,
        sources=cdc_urls,
        additional_instruction=(
            "Check CDC adult RSV guidance for risk conditions. Accept phrasing such as 'chronic lung disease (e.g., COPD, asthma)'."
        ),
    )

    diabetes_leaf = evaluator.add_leaf(
        id="Insulin_Treated_Diabetes_As_Risk_Factor",
        desc="Correctly identifies diabetes mellitus requiring insulin treatment as a qualifying CDC risk factor for older adults.",
        parent=risk_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "CDC recognizes diabetes mellitus as a risk factor for severe RSV disease in older adults; "
            "insulin-treated diabetes is within the scope of diabetes risk conditions considered by CDC."
        ),
        node=diabetes_leaf,
        sources=cdc_urls,
        additional_instruction=(
            "Look for CDC listing of diabetes mellitus among conditions that increase risk for severe RSV in older adults. "
            "References to diabetes without specifying insulin treatment still count as recognizing the condition."
        ),
    )

    # ----------------------- Pharmacy Service Options -------------------- #
    pharmacy_node = evaluator.add_parallel(
        id="Pharmacy_Service_Options",
        desc="Provides at least one U.S. pharmacy service option and confirms the patient meets that pharmacy's age requirements.",
        parent=assessment_node,
        critical=True
    )

    chain_identified = bool((extracted.pharmacy_name or "").strip()) and bool((extracted.pharmacy_url or "").strip())
    evaluator.add_custom_node(
        result=chain_identified,
        id="Pharmacy_Chain_Identified",
        desc="Identifies at least one national/major U.S. pharmacy chain offering RSV vaccination services.",
        parent=pharmacy_node,
        critical=True
    )

    pharmacy_age_leaf = evaluator.add_leaf(
        id="Pharmacy_Age_Requirement_Accommodates_68",
        desc="Confirms the selected pharmacy's published age requirements allow vaccination of a 68-year-old.",
        parent=pharmacy_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The selected pharmacy offers RSV vaccination and its published age requirements include adults whose age "
            "covers a 68-year-old (e.g., age 60+ or similar), therefore a 68-year-old patient can receive RSV vaccination there."
        ),
        node=pharmacy_age_leaf,
        sources=extracted.pharmacy_url,
        additional_instruction=(
            "Review the pharmacy's RSV or immunizations page. If it states RSV vaccine is available for 'adults 60 years and older' "
            "or comparable wording, conclude age 68 qualifies. If the page indicates routine RSV for ≥75 years and offers for "
            "60–74 by clinical judgment, still conclude a 68-year-old can be vaccinated at the pharmacy."
        ),
    )

    # ----------------------- Timing Assessment --------------------------- #
    timing_leaf = evaluator.add_leaf(
        id="Timing_Assessment",
        desc="Assesses whether September is an appropriate/optimal time for RSV vaccination based on CDC timing recommendations (late summer/early fall).",
        parent=assessment_node,
        critical=True
    )
    await evaluator.verify(
        claim="According to CDC recommendations, September is an appropriate time for RSV vaccination (late summer to early fall).",
        node=timing_leaf,
        sources=cdc_urls,
        additional_instruction=(
            "Check CDC's timing guidance for RSV vaccination in older adults. 'Late summer to early fall' should encompass September."
        ),
    )

    # ----------------------- Co-Administration Guidance ------------------ #
    coadmin_leaf = evaluator.add_leaf(
        id="Co_Administration_Guidance",
        desc="States whether RSV vaccine can be co-administered with other vaccines (e.g., flu and/or COVID-19) consistent with CDC guidance.",
        parent=assessment_node,
        critical=True
    )
    await evaluator.verify(
        claim="CDC allows RSV vaccine to be administered at the same visit as other vaccines such as influenza (flu) and COVID-19 vaccines.",
        node=coadmin_leaf,
        sources=cdc_urls,
        additional_instruction=(
            "Look for CDC statements that RSV vaccines may be co-administered with influenza and/or COVID-19 vaccines."
        ),
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
    Evaluate an answer for the RSV vaccination assessment task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # overall assessment items evaluated independently
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_rsv_assessment(),
        template_class=RSVAssessmentExtraction,
        extraction_name="rsv_assessment_extraction",
    )

    # Add some scenario context as custom info
    evaluator.add_custom_info(
        {
            "patient_age": 68,
            "month": "September",
            "conditions": ["asthma", "type 2 diabetes (insulin)"],
            "country": "United States"
        },
        info_type="scenario",
        info_name="patient_context"
    )

    # Build and verify according to rubric
    await build_and_verify_rsv_assessment(evaluator, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()