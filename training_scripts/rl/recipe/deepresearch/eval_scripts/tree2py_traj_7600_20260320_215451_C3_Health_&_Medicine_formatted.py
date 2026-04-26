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
TASK_ID = "cdc_aap_2026_schedule_comparison"
TASK_DESCRIPTION = """
In early 2026, both the CDC and the American Academy of Pediatrics (AAP) released updated childhood immunization schedules that differed in significant ways. Identify the specific date when the CDC released its updated 2026 childhood immunization schedule and provide the URL of the official CDC press release or announcement about this update. Then, identify one vaccine that the CDC moved from universal routine recommendations to either the high-risk groups category or the shared clinical decision-making category, but that the AAP continues to recommend as routine for all children in its 2026 schedule. For the vaccine you identify, specify which category (high-risk groups or shared clinical decision-making) the CDC assigned it to, and confirm that the AAP maintains it as a routine recommendation for all children in their 2026 schedule by providing a reference URL from an official AAP source or a reliable news source discussing the AAP's 2026 schedule.
"""

EXPECTED_CDC_RELEASE_DATE = "January 5, 2026"
ALLOWED_MOVED_VACCINES = [
    "hepatitis A",
    "hepatitis B",
    "rotavirus",
    "COVID-19",
    "influenza",
    "meningococcal disease",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CDCInfo(BaseModel):
    release_date: Optional[str] = None
    announcement_url: Optional[str] = None


class VaccineComparison(BaseModel):
    vaccine_name: Optional[str] = None
    cdc_category: Optional[str] = None  # Expect "high-risk groups" or "shared clinical decision-making" (allow synonyms)
    vaccine_analysis_urls: List[str] = Field(default_factory=list)  # Any reliable page documenting CDC moved vs AAP routine
    cdc_categorization_urls: List[str] = Field(default_factory=list)  # Official CDC/HHS page(s) showing category assignment
    aap_position_urls: List[str] = Field(default_factory=list)  # AAP official or reliable news documenting AAP routine


class ScheduleComparisonExtraction(BaseModel):
    cdc: Optional[CDCInfo] = None
    vaccine: Optional[VaccineComparison] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_schedule_comparison() -> str:
    return """
Extract the following structured information from the answer. Return null for any field not explicitly present in the answer.

Section 1 — CDC 2026 Schedule Release:
- cdc.release_date: The specific date the CDC released its updated 2026 childhood immunization schedule (extract verbatim from the answer, allow any date formatting).
- cdc.announcement_url: The URL to the official CDC press release or announcement (prefer cdc.gov; if HHS.gov announcement is given instead, extract that URL). If multiple links are given, choose the single most relevant official page.

Section 2 — Vaccine moved by CDC but kept routine by AAP:
Extract details for exactly one vaccine the answer highlights as moved by CDC out of universal routine but that AAP still keeps as routine in its 2026 schedule.
- vaccine.vaccine_name: The name of the vaccine (e.g., "hepatitis A", "hepatitis B", "rotavirus", "COVID-19", "influenza", "meningococcal disease"). Keep the text as written in the answer.
- vaccine.cdc_category: The category the CDC placed this vaccine into for 2026 (text string as written in the answer, typically "shared clinical decision-making" or "high-risk groups", but may include synonyms like "SCDM", "persons at increased risk", "risk-based").
- vaccine.vaccine_analysis_urls: All cited URLs (up to 3) that discuss which vaccines CDC moved from routine and how AAP differs (these may be reliable news, policy analyses, or summaries).
- vaccine.cdc_categorization_urls: All cited official CDC/HHS URLs (up to 3) that explicitly state the 2026 CDC category assignment for this vaccine.
- vaccine.aap_position_urls: All cited official AAP URLs (aap.org or publications.aap.org) or reliable news coverage URLs (up to 3) that explicitly state that AAP keeps this vaccine as routine for all children (or standard age groups) in its 2026 schedule.

Important extraction rules:
- Extract only what is explicitly present in the answer.
- For URL fields, extract only valid complete URLs. If none provided, return an empty list (or null for the single CDC announcement URL).
- Do not infer or invent any data.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_cdc_info(
    evaluator: Evaluator,
    parent_node,
    cdc: Optional[CDCInfo],
) -> None:
    """
    Build and verify the 'CDC_Schedule_Information' subtree.
    """
    cdc_node = evaluator.add_parallel(
        id="CDC_Schedule_Information",
        desc="Accurate identification of CDC's 2026 schedule release date and official announcement source.",
        parent=parent_node,
        critical=True,
    )

    # CDC Release Date leaf (critical)
    release_date_leaf = evaluator.add_leaf(
        id="CDC_Release_Date",
        desc="The CDC release date is correctly identified as January 5, 2026.",
        parent=cdc_node,
        critical=True,
    )
    extracted_date = (cdc.release_date or "").strip() if cdc else ""
    date_claim = (
        f"The CDC release date identified in the answer equals {EXPECTED_CDC_RELEASE_DATE}. "
        f"Extracted date text from the answer: '{extracted_date}'."
    )
    await evaluator.verify(
        claim=date_claim,
        node=release_date_leaf,
        additional_instruction=(
            "Judge strictly based on equality of meaning. Accept different date formats "
            "(e.g., 'Jan 5, 2026', '2026-01-05') as equivalent to January 5, 2026. "
            "If the extracted date is missing, ambiguous, or not equal to Jan 5, 2026, return Incorrect."
        ),
    )

    # CDC Official Source leaf (critical)
    official_src_leaf = evaluator.add_leaf(
        id="CDC_Official_Source",
        desc="A valid official CDC or HHS URL is provided that documents the schedule release.",
        parent=cdc_node,
        critical=True,
    )
    cdc_url = (cdc.announcement_url or "").strip() if cdc else ""
    if not cdc_url:
        # Explicitly fail if no URL provided
        official_src_leaf.score = 0.0
        official_src_leaf.status = "failed"
    else:
        src_claim = (
            "This page is an official CDC (cdc.gov) or HHS (hhs.gov) announcement or press release "
            "that documents the release of the CDC's 2026 childhood immunization schedule "
            "and clearly indicates it is about the 2026 schedule update."
        )
        await evaluator.verify(
            claim=src_claim,
            node=official_src_leaf,
            sources=cdc_url,
            additional_instruction=(
                "Treat as correct only if the page is hosted on cdc.gov or hhs.gov and clearly announces "
                "or documents the release of the updated 2026 childhood immunization schedule."
            ),
        )


async def verify_vaccine_comparison(
    evaluator: Evaluator,
    parent_node,
    vaccine: Optional[VaccineComparison],
) -> None:
    """
    Build and verify the 'Vaccine_Comparison_Analysis' subtree.
    """
    comp_node = evaluator.add_sequential(
        id="Vaccine_Comparison_Analysis",
        desc="Identification and analysis of a vaccine that CDC moved from routine recommendations but AAP maintains as routine.",
        parent=parent_node,
        critical=True,
    )

    # 1) Vaccine_Analysis_URL_Reference (critical leaf)
    analysis_ref_leaf = evaluator.add_leaf(
        id="Vaccine_Analysis_URL_Reference",
        desc="Valid URL reference documenting which vaccines CDC moved from routine recommendations and which vaccines AAP maintains as routine.",
        parent=comp_node,
        critical=True,
    )
    analysis_urls = vaccine.vaccine_analysis_urls if vaccine else []
    if not analysis_urls:
        analysis_ref_leaf.score = 0.0
        analysis_ref_leaf.status = "failed"
    else:
        analysis_claim = (
            "This page discusses the CDC's 2026 childhood immunization schedule changes, including which vaccines "
            "were moved out of universal routine recommendations, and also addresses how the AAP's 2026 schedule "
            "differs or maintains certain vaccines as routine."
        )
        await evaluator.verify(
            claim=analysis_claim,
            node=analysis_ref_leaf,
            sources=analysis_urls,
            additional_instruction=(
                "Consider the claim supported if the page explicitly mentions CDC's 2026 schedule changes and addresses AAP's stance. "
                "General vaccine pages without specific 2026 context should be considered not supported."
            ),
        )

    # 2) Moved_Vaccine_Identification (critical leaf)
    moved_vax_leaf = evaluator.add_leaf(
        id="Moved_Vaccine_Identification",
        desc="The identified vaccine is one that CDC moved from universal routine recommendations (must be one of: hepatitis A, hepatitis B, rotavirus, COVID-19, influenza, or meningococcal disease).",
        parent=comp_node,
        critical=True,
    )
    vaccine_name = (vaccine.vaccine_name or "").strip() if vaccine else ""
    moved_vax_claim = (
        f'The identified vaccine "{vaccine_name}" corresponds to one of the allowed set: '
        f'{", ".join(ALLOWED_MOVED_VACCINES)}.'
    )
    await evaluator.verify(
        claim=moved_vax_claim,
        node=moved_vax_leaf,
        additional_instruction=(
            "Judge based on semantic equivalence. Allow common synonyms/variants such as "
            "‘flu’ for influenza; ‘COVID’/‘COVID vaccine’ for COVID-19; ‘MenACWY’/‘MenB’ or just ‘meningococcal’ for meningococcal disease; "
            "‘Hep A’ and ‘Hep B’ for hepatitis A/B. If it does not clearly match any allowed item, mark Incorrect."
        ),
    )

    # 3) CDC_Categorization_Analysis (critical sequential subtree)
    cat_node = evaluator.add_sequential(
        id="CDC_Categorization_Analysis",
        desc="Correct specification of which category CDC assigned the identified vaccine to and verification of AAP's position.",
        parent=comp_node,
        critical=True,
    )

    # 3.1) Categorization_URL_Reference (critical leaf)
    cat_ref_leaf = evaluator.add_leaf(
        id="Categorization_URL_Reference",
        desc="Valid URL reference from CDC or HHS source specifying which category (high-risk groups or shared clinical decision-making) the identified vaccine was assigned to.",
        parent=cat_node,
        critical=True,
    )
    cat_urls = vaccine.cdc_categorization_urls if vaccine else []
    if not cat_urls:
        cat_ref_leaf.score = 0.0
        cat_ref_leaf.status = "failed"
    else:
        cat_ref_claim = (
            "This page is an official CDC (cdc.gov) or HHS (hhs.gov) source for the 2026 CDC childhood immunization schedule "
            "that specifies the category assignment (e.g., shared clinical decision-making or high-risk groups) for the identified vaccine."
        )
        await evaluator.verify(
            claim=cat_ref_claim,
            node=cat_ref_leaf,
            sources=cat_urls,
            additional_instruction=(
                "Only accept if the page is clearly CDC/HHS and explicitly addresses the 2026 schedule categorization of the vaccine."
            ),
        )

    # 3.2) CDC_Category_Assignment (critical leaf)
    cat_assign_leaf = evaluator.add_leaf(
        id="CDC_Category_Assignment",
        desc="The category (high-risk groups or shared clinical decision-making) is correctly specified for the identified vaccine based on CDC's 2026 schedule.",
        parent=cat_node,
        critical=True,
    )
    cdc_category_txt = (vaccine.cdc_category or "").strip() if vaccine else ""
    cat_assign_claim = (
        f"In the 2026 CDC childhood immunization schedule, the vaccine '{vaccine_name}' is assigned to the category "
        f"'{cdc_category_txt}' (interpreting synonyms like 'SCDM' for shared clinical decision-making, "
        f"or 'persons at increased risk'/'risk-based' for high-risk groups), and not kept as universal routine."
    )
    if not cat_urls:
        # If no categorization URLs available, we cannot verify the assignment against CDC/HHS evidence → fail
        cat_assign_leaf.score = 0.0
        cat_assign_leaf.status = "failed"
    else:
        await evaluator.verify(
            claim=cat_assign_claim,
            node=cat_assign_leaf,
            sources=cat_urls,
            additional_instruction=(
                "Verify that the CDC 2026 schedule (or associated CDC/HHS documentation) places the named vaccine into the "
                "stated category (shared clinical decision-making or high-risk groups). Treat equivalent phrasings as matches. "
                "If the page contradicts the claim or lacks explicit categorization for 2026, mark Incorrect."
            ),
        )

    # 3.3) AAP_Position_Verification (critical parallel subtree)
    aap_node = evaluator.add_parallel(
        id="AAP_Position_Verification",
        desc="Verification that AAP maintains the identified vaccine as a routine recommendation for all children (or specified age groups) in its 2026 schedule.",
        parent=cat_node,
        critical=True,
    )

    # 3.3.a) AAP_Position_URL_Reference (critical leaf)
    aap_ref_leaf = evaluator.add_leaf(
        id="AAP_Position_URL_Reference",
        desc="Valid URL reference from official AAP source or reliable news source documenting AAP's position on the identified vaccine in their 2026 schedule.",
        parent=aap_node,
        critical=True,
    )
    aap_urls = vaccine.aap_position_urls if vaccine else []
    if not aap_urls:
        aap_ref_leaf.score = 0.0
        aap_ref_leaf.status = "failed"
    else:
        aap_ref_claim = (
            "This page is either an official AAP source (aap.org or publications.aap.org) or reliable news coverage "
            "that documents AAP's 2026 immunization schedule position for the identified vaccine."
        )
        await evaluator.verify(
            claim=aap_ref_claim,
            node=aap_ref_leaf,
            sources=aap_urls,
            additional_instruction=(
                "Prefer official AAP pages. Reliable coverage must explicitly reference AAP's 2026 schedule and the vaccine. "
                "General vaccine info without 2026 schedule context should be considered not supported."
            ),
        )

    # 3.3.b) AAP_Routine_Recommendation_Status (critical leaf)
    aap_status_leaf = evaluator.add_leaf(
        id="AAP_Routine_Recommendation_Status",
        desc="AAP's position is correctly stated as maintaining the vaccine as routine for all children (or specified age groups) in their 2026 schedule.",
        parent=aap_node,
        critical=True,
    )
    aap_status_claim = (
        f"In AAP's 2026 childhood immunization schedule, the vaccine '{vaccine_name}' is maintained as a routine "
        f"recommendation for all children (or standard age groups), i.e., not restricted solely to high-risk/SCDM."
    )
    if not aap_urls:
        aap_status_leaf.score = 0.0
        aap_status_leaf.status = "failed"
    else:
        await evaluator.verify(
            claim=aap_status_claim,
            node=aap_status_leaf,
            sources=aap_urls,
            additional_instruction=(
                "Confirm that AAP's 2026 schedule (or reliable coverage explicitly referencing it) shows the vaccine as part "
                "of routine recommendations for the general pediatric population or standard age-based groups. "
                "If the material indicates only risk-based/SCDM or lacks clear routine language, mark Incorrect."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the CDC vs AAP 2026 childhood immunization schedule comparison task.
    """
    # Initialize evaluator (root: parallel)
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

    # Record ground truth/expectations
    evaluator.add_ground_truth(
        {
            "expected_cdc_release_date": EXPECTED_CDC_RELEASE_DATE,
            "allowed_moved_vaccines": ALLOWED_MOVED_VACCINES,
            "notes": "CDC category should be either 'shared clinical decision-making' or 'high-risk groups' (allowing synonyms).",
        },
        gt_type="ground_truth",
    )

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_schedule_comparison(),
        template_class=ScheduleComparisonExtraction,
        extraction_name="schedule_comparison_extraction",
    )

    # Build and verify CDC info subtree
    await verify_cdc_info(evaluator, root, extraction.cdc if extraction else None)

    # Build and verify vaccine comparison subtree
    await verify_vaccine_comparison(evaluator, root, extraction.vaccine if extraction else None)

    # Return summary
    return evaluator.get_summary()