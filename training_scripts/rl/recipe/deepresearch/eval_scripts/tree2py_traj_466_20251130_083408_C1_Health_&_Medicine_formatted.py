import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "morocco_travel_vaccines_cdc"
TASK_DESCRIPTION = """
For someone planning to travel to Morocco, is the typhoid vaccine recommended by the CDC? Additionally, what is the CDC's recommended timeframe for scheduling a travel vaccine consultation before departure?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TyphoidClaim(BaseModel):
    """
    Information the answer states about CDC's typhoid vaccine recommendation for Morocco.
    """
    stance: Optional[str] = None  # expected values: "recommended", "not_recommended", "unclear"
    mentions_high_risk: Optional[bool] = None  # true if the answer explicitly mentions higher-risk travelers (e.g., staying with friends/relatives; smaller cities/rural areas)
    sources: List[str] = Field(default_factory=list)  # URLs supporting the typhoid claim (as cited in the answer)


class ConsultationClaim(BaseModel):
    """
    Information the answer states about CDC's timeframe for pre-travel consultation.
    """
    lead_time_text: Optional[str] = None  # e.g., "at least 1 month", "4–6 weeks", "ideally 4-6 weeks"
    sources: List[str] = Field(default_factory=list)  # URLs supporting the lead time statement (as cited in the answer)


class TravelVaccineExtraction(BaseModel):
    """
    Combined extraction for this task.
    """
    typhoid: Optional[TyphoidClaim] = None
    consultation: Optional[ConsultationClaim] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_vaccine_info() -> str:
    return """
    Extract the following information strictly from the provided answer text.

    1) Typhoid vaccine recommendation for Morocco (CDC):
       - stance: One of ["recommended", "not_recommended", "unclear"].
         • "recommended" only if the answer explicitly states the CDC recommends typhoid vaccination for travelers to Morocco (or "most travelers") OR clearly implies CDC recommendation.
         • "not_recommended" only if the answer explicitly states CDC does NOT recommend it.
         • "unclear" if the answer does not clearly state either.
       - mentions_high_risk: true/false. Return true only if the answer explicitly includes CDC’s emphasis that typhoid vaccination is especially important for higher-risk travelers (for example, travelers staying with friends or relatives, visiting smaller cities or rural areas).
       - sources: Array of all URLs that the answer cites as sources for the typhoid recommendation. Include only URLs explicitly present in the answer.

    2) Travel consultation lead time (CDC):
       - lead_time_text: The phrase or timespan the answer claims for the CDC-recommended timeframe to schedule a travel vaccine/health consultation before departure (e.g., "at least 1 month", "4–6 weeks", "ideally 4-6 weeks"). Extract the phrase as written.
       - sources: Array of all URLs that the answer cites as sources for this lead time. Include only URLs explicitly present in the answer.

    If any requested field is missing in the answer, set it to null or an empty list accordingly.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def filter_cdc_urls(urls: List[str]) -> List[str]:
    """Return only CDC URLs from a list of URLs."""
    out: List[str] = []
    for u in urls or []:
        if isinstance(u, str) and "cdc.gov" in u.lower():
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_typhoid_verification(
    evaluator: Evaluator,
    parent_node,
    extracted: TravelVaccineExtraction
) -> None:
    """
    Build and verify the 'CDC_Typhoid_Recommendation' sequential subtree.
    """
    ty_node = evaluator.add_sequential(
        id="CDC_Typhoid_Recommendation",
        desc="Answer addresses whether CDC recommends typhoid vaccination for travelers to Morocco (including the CDC emphasis for higher-risk travelers such as those staying with friends/relatives or visiting smaller cities/rural areas)",
        parent=parent_node,
        critical=True,
    )

    # Leaf 1: Typhoid_Recommendation_Stated (critical)
    leaf_stated = evaluator.add_leaf(
        id="Typhoid_Recommendation_Stated",
        desc="States that CDC recommends typhoid vaccination for Morocco travelers (with the specified higher-risk emphasis as applicable)",
        parent=ty_node,
        critical=True,
    )

    # We verify against the answer text itself, requiring both the recommendation and the "higher-risk emphasis"
    claim_stated = (
        "In the answer text, it is explicitly stated that the CDC recommends typhoid vaccination for travelers to Morocco, "
        "and the answer includes the CDC's emphasis that this recommendation is especially important for higher-risk travelers "
        "(for example, those staying with friends or relatives or visiting smaller cities or rural areas)."
    )
    await evaluator.verify(
        claim=claim_stated,
        node=leaf_stated,
        additional_instruction=(
            "Judge only based on the answer text. Accept equivalent wording such as 'most travelers' for Morocco and "
            "'especially if staying with friends or relatives or visiting smaller cities/rural areas'. "
            "If the answer omits the higher-risk emphasis entirely, mark this as incorrect."
        ),
    )

    # Leaf 2: Typhoid_Recommendation_CDC_Source (critical)
    leaf_cdc_source = evaluator.add_leaf(
        id="Typhoid_Recommendation_CDC_Source",
        desc="Provides an official CDC travel health source (e.g., CDC Travelers’ Health destination page for Morocco) that supports the typhoid recommendation claim",
        parent=ty_node,
        critical=True,
    )

    typhoid_sources = extracted.typhoid.sources if (extracted and extracted.typhoid and extracted.typhoid.sources) else []
    cdc_typhoid_sources = filter_cdc_urls(typhoid_sources)

    claim_cdc_supports = (
        "The CDC states that typhoid vaccination is recommended for travelers to Morocco, "
        "especially for higher-risk travelers (e.g., staying with friends or relatives or visiting smaller cities/rural areas)."
    )
    await evaluator.verify(
        claim=claim_cdc_supports,
        node=leaf_cdc_source,
        sources=cdc_typhoid_sources,  # must be CDC; if empty or invalid, verification should fail
        additional_instruction=(
            "Only consider this supported if the URL is an official CDC page (preferably CDC Travelers’ Health). "
            "The page should clearly indicate that typhoid vaccination is recommended for Morocco travelers and highlight higher-risk scenarios. "
            "If the URL is not CDC or does not explicitly support the claim, mark as not supported."
        ),
    )


async def build_lead_time_verification(
    evaluator: Evaluator,
    parent_node,
    extracted: TravelVaccineExtraction
) -> None:
    """
    Build and verify the 'Consultation_Lead_Time' sequential subtree.
    """
    lt_node = evaluator.add_sequential(
        id="Consultation_Lead_Time",
        desc="Answer provides the CDC-recommended timeframe for scheduling a travel vaccine/health consultation before departure",
        parent=parent_node,
        critical=True,
    )

    # Leaf 1: Lead_Time_Stated (critical)
    leaf_lt_stated = evaluator.add_leaf(
        id="Lead_Time_Stated",
        desc="Gives a CDC-consistent lead time of at least 1 month (≈4 weeks; commonly expressed as 4–6 weeks) before departure",
        parent=lt_node,
        critical=True,
    )

    claim_lt_stated = (
        "In the answer text, it is stated that travelers should schedule a travel vaccine/health consultation "
        "at least 1 month (about 4 weeks), commonly expressed as 4–6 weeks, before departure."
    )
    await evaluator.verify(
        claim=claim_lt_stated,
        node=leaf_lt_stated,
        additional_instruction=(
            "Judge strictly by the answer text. Accept phrasing like 'at least 1 month before travel', '4–6 weeks before travel', "
            "'ideally 4–6 weeks', or equivalent language indicating ≥1 month lead time."
        ),
    )

    # Leaf 2: Lead_Time_CDC_Source (critical)
    leaf_lt_cdc = evaluator.add_leaf(
        id="Lead_Time_CDC_Source",
        desc="Provides an official CDC travel health source supporting the stated consultation lead time",
        parent=lt_node,
        critical=True,
    )

    lt_sources = extracted.consultation.sources if (extracted and extracted.consultation and extracted.consultation.sources) else []
    cdc_lt_sources = filter_cdc_urls(lt_sources)

    claim_lt_cdc = (
        "The CDC recommends that travelers seek pre-travel health consultation ideally 4–6 weeks (i.e., at least 1 month) before departure."
    )
    await evaluator.verify(
        claim=claim_lt_cdc,
        node=leaf_lt_cdc,
        sources=cdc_lt_sources,  # must be CDC; if empty, verification should fail
        additional_instruction=(
            "Only consider this supported if the URL is an official CDC page (e.g., CDC Travelers’ Health) that explicitly states "
            "a recommended lead time around 4–6 weeks or at least 1 month before travel."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
    Evaluate an answer for CDC Morocco travel vaccine and lead-time guidance.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
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

    # 1) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_travel_vaccine_info(),
        template_class=TravelVaccineExtraction,
        extraction_name="extracted_travel_vaccine_info",
    )

    # 2) Build the rubric tree structure: top-level critical node with two critical sequential children
    top_node = evaluator.add_parallel(
        id="Morocco_Travel_Vaccine_Information",
        desc="Verify Morocco travel vaccine guidance per CDC and ensure claims are supported by official CDC travel health sources",
        parent=root,
        critical=True,
    )

    # Build and verify each critical sequential subtree
    await build_typhoid_verification(evaluator, top_node, extracted)
    await build_lead_time_verification(evaluator, top_node, extracted)

    # Optional: record expected policy notes for debugging (not used for scoring)
    evaluator.add_custom_info(
        info={
            "expected_policy_notes": {
                "typhoid": "CDC recommends typhoid vaccine for most travelers to Morocco; especially important if staying with friends/relatives or visiting smaller cities/rural areas.",
                "lead_time": "CDC advises scheduling pre-travel consultation ideally 4–6 weeks (at least 1 month) before departure."
            }
        },
        info_type="notes",
        info_name="policy_expectations"
    )

    return evaluator.get_summary()