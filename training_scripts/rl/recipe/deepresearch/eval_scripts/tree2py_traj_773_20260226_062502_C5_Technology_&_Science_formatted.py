import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "verizon_outage_2026_followup"
TASK_DESCRIPTION = (
    "Provide comprehensive information about the January 2026 Verizon outage aftermath, including FCC complaint process, "
    "emergency communication alternatives, and carrier reliability data"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FCCInfo(BaseModel):
    fcc_deadline: Optional[str] = None
    fcc_email_address: Optional[str] = None
    fcc_efs_name: Optional[str] = None
    fcc_sources: List[str] = Field(default_factory=list)


class IPhoneSOSInfo(BaseModel):
    iphone_supported_models: Optional[str] = None
    iphone_min_ios_version_us_canada: Optional[str] = None
    iphone_cost_terms: Optional[str] = None
    iphone_physical_requirements: Optional[str] = None
    iphone_sources: List[str] = Field(default_factory=list)


class VerizonCompInfo(BaseModel):
    verizon_credit_amount: Optional[str] = None
    verizon_credit_redemption: Optional[str] = None
    verizon_sources: List[str] = Field(default_factory=list)


class ReliabilityInfo(BaseModel):
    att_rootmetrics_states_won_h1_2025: Optional[str] = None
    opensignal_tied_carriers: List[str] = Field(default_factory=list)
    opensignal_reliability_score: Optional[str] = None
    rootmetrics_sources: List[str] = Field(default_factory=list)
    opensignal_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_fcc() -> str:
    return """
    Extract information about the FCC public comment period for the January 14, 2026 Verizon outage investigation.
    Return the following fields:
    - fcc_deadline: The deadline date (and time if provided) for submitting public comments about the Verizon outage to the FCC. Extract exactly as written in the answer.
    - fcc_email_address: The specific email address (if any) designated by the FCC to receive comments about this Verizon outage. If none is provided in the answer, return null.
    - fcc_efs_name: The name or description of the FCC's electronic filing system used for public comment submissions (e.g., 'ECFS – Electronic Comment Filing System'), as stated in the answer.
    - fcc_sources: A list of all URL sources cited in the answer that support the above FCC information (docket pages, FCC notices, etc.). Only include URLs mentioned in the answer.
    """


def prompt_extract_iphone() -> str:
    return """
    Extract the requirements and specifications for Apple's Emergency SOS via satellite feature as presented in the answer.
    Return the following fields:
    - iphone_supported_models: The iPhone model generations that support Emergency SOS via satellite (e.g., 'iPhone 14 and later', or a list of supported models) exactly as stated.
    - iphone_min_ios_version_us_canada: The minimum iOS version required to use Emergency SOS via satellite in the US and Canada, exactly as stated.
    - iphone_cost_terms: The cost structure and free service duration (e.g., 'free for X years/months, then $Y per month'), exactly as stated.
    - iphone_physical_requirements: The physical/environmental conditions required (e.g., 'must be outside with a clear view of the sky; obstructions like trees/buildings may block signal'), exactly as stated.
    - iphone_sources: A list of all URL sources cited in the answer that support these statements (e.g., Apple Support pages, Apple newsroom posts). Only include URLs that were explicitly mentioned.
    """


def prompt_extract_verizon_comp() -> str:
    return """
    Extract the details of Verizon's compensation to customers affected by the January 2026 outage as presented in the answer.
    Return the following fields:
    - verizon_credit_amount: The dollar amount of the account credit offered (e.g., '$5 bill credit'), exactly as stated in the answer.
    - verizon_credit_redemption: How affected customers can claim, receive, or redeem the credit (e.g., 'automatically applied', 'via account/billing page', 'through customer support'), exactly as stated.
    - verizon_sources: A list of URL sources cited that directly support this compensation information (e.g., Verizon announcements, press reports). Only include URLs mentioned in the answer.
    """


def prompt_extract_reliability() -> str:
    return """
    Extract recent carrier network reliability results as presented in the answer.
    Return the following fields:
    - att_rootmetrics_states_won_h1_2025: The number of U.S. states in which AT&T won the Reliability RootScore Awards during the first half of 2025 (H1 2025), exactly as stated in the answer.
    - opensignal_tied_carriers: An array of the two carrier names that tied for Opensignal's Reliability Experience award in January 2025 (e.g., ["AT&T", "Verizon"]), exactly as stated.
    - opensignal_reliability_score: The score they received for Reliability Experience (as shown by Opensignal; often a number on a 100–1000 style scale), exactly as stated in the answer.
    - rootmetrics_sources: A list of URLs cited for the RootMetrics data (H1 2025). Only include URLs mentioned in the answer.
    - opensignal_sources: A list of URLs cited for the Opensignal January 2025 Reliability Experience award. Only include URLs mentioned in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper for verification instructions                                        #
# --------------------------------------------------------------------------- #
def require_sources_instruction(topic: str) -> str:
    return (
        f"Source-grounded verification for: {topic}. "
        "Use only the provided URL evidence to decide whether the claim is supported. "
        "If no valid URL sources are provided, or if the webpages do not explicitly support the claim, "
        "mark the claim as Incorrect/Not supported. Allow minor wording or formatting variations, but the fact itself must be explicitly supported."
    )


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_fcc_process_subtree(evaluator: Evaluator, parent) -> None:
    fcc_info = await evaluator.extract(
        prompt=prompt_extract_fcc(),
        template_class=FCCInfo,
        extraction_name="fcc_comment_info",
    )

    node = evaluator.add_parallel(
        id="FCC_Comment_Process",
        desc="Information about the FCC public comment period for the Verizon outage investigation",
        parent=parent,
        critical=False,
    )

    # Leaf: Comment_Submission_Deadline (critical)
    deadline_leaf = evaluator.add_leaf(
        id="Comment_Submission_Deadline",
        desc="The deadline date for submitting public comments to the FCC about the Verizon outage",
        parent=node,
        critical=True,
    )
    claim_deadline = (
        f"The FCC's public comment submission deadline for the Verizon outage investigation is: {fcc_info.fcc_deadline}."
    )
    await evaluator.verify(
        claim=claim_deadline,
        node=deadline_leaf,
        sources=fcc_info.fcc_sources if fcc_info.fcc_sources else None,
        additional_instruction=require_sources_instruction("FCC public comment deadline"),
    )

    # Parallel critical: Valid_Submission_Methods (container)
    methods_node = evaluator.add_parallel(
        id="Valid_Submission_Methods",
        desc="The officially recognized methods for submitting comments to the FCC about this outage",
        parent=node,
        critical=True,
    )

    # Leaf: Email_Submission (critical)
    email_leaf = evaluator.add_leaf(
        id="Email_Submission",
        desc="The specific email address designated by the FCC for Verizon outage comments",
        parent=methods_node,
        critical=True,
    )
    claim_email = (
        f"The FCC designated the following email address to receive public comments regarding the Verizon outage: {fcc_info.fcc_email_address}."
    )
    await evaluator.verify(
        claim=claim_email,
        node=email_leaf,
        sources=fcc_info.fcc_sources if fcc_info.fcc_sources else None,
        additional_instruction=require_sources_instruction("FCC email submission method"),
    )

    # Leaf: Electronic_Filing_System (critical)
    efs_leaf = evaluator.add_leaf(
        id="Electronic_Filing_System",
        desc="The name or description of the FCC's electronic filing system that can be used for submissions",
        parent=methods_node,
        critical=True,
    )
    claim_efs = (
        f"Public comments can be submitted via the FCC's electronic filing system: {fcc_info.fcc_efs_name}."
    )
    await evaluator.verify(
        claim=claim_efs,
        node=efs_leaf,
        sources=fcc_info.fcc_sources if fcc_info.fcc_sources else None,
        additional_instruction=require_sources_instruction("FCC electronic filing system (e.g., ECFS)"),
    )


async def build_iphone_subtree(evaluator: Evaluator, parent) -> None:
    iphone_info = await evaluator.extract(
        prompt=prompt_extract_iphone(),
        template_class=IPhoneSOSInfo,
        extraction_name="iphone_sos_info",
    )

    node = evaluator.add_parallel(
        id="iPhone_Satellite_Emergency",
        desc="Requirements and specifications for using iPhone's Emergency SOS via satellite as a backup communication method",
        parent=parent,
        critical=False,
    )

    # Compatible_Device_Models (critical)
    models_leaf = evaluator.add_leaf(
        id="Compatible_Device_Models",
        desc="Which iPhone model generations support the Emergency SOS via satellite feature",
        parent=node,
        critical=True,
    )
    claim_models = f"Emergency SOS via satellite is supported on: {iphone_info.iphone_supported_models}."
    await evaluator.verify(
        claim=claim_models,
        node=models_leaf,
        sources=iphone_info.iphone_sources if iphone_info.iphone_sources else None,
        additional_instruction=require_sources_instruction("iPhone Emergency SOS via satellite - compatible models"),
    )

    # Required_iOS_Version (critical)
    ios_leaf = evaluator.add_leaf(
        id="Required_iOS_Version",
        desc="The minimum iOS version required for using Emergency SOS via satellite in the US and Canada",
        parent=node,
        critical=True,
    )
    claim_ios = (
        f"The minimum iOS version required in the US and Canada for Emergency SOS via satellite is: "
        f"{iphone_info.iphone_min_ios_version_us_canada}."
    )
    await evaluator.verify(
        claim=claim_ios,
        node=ios_leaf,
        sources=iphone_info.iphone_sources if iphone_info.iphone_sources else None,
        additional_instruction=require_sources_instruction("iPhone Emergency SOS via satellite - minimum iOS version"),
    )

    # Service_Cost_Terms (critical)
    cost_leaf = evaluator.add_leaf(
        id="Service_Cost_Terms",
        desc="The cost structure and free service duration for the satellite emergency feature",
        parent=node,
        critical=True,
    )
    claim_cost = (
        f"The cost and service duration terms for Emergency SOS via satellite are: {iphone_info.iphone_cost_terms}."
    )
    await evaluator.verify(
        claim=claim_cost,
        node=cost_leaf,
        sources=iphone_info.iphone_sources if iphone_info.iphone_sources else None,
        additional_instruction=require_sources_instruction("iPhone Emergency SOS via satellite - cost and free duration"),
    )

    # Physical_Location_Requirement (critical)
    environment_leaf = evaluator.add_leaf(
        id="Physical_Location_Requirement",
        desc="The environmental conditions required for the satellite feature to function",
        parent=node,
        critical=True,
    )
    claim_env = (
        f"To use Emergency SOS via satellite, the following physical/environmental conditions are required: "
        f"{iphone_info.iphone_physical_requirements}."
    )
    await evaluator.verify(
        claim=claim_env,
        node=environment_leaf,
        sources=iphone_info.iphone_sources if iphone_info.iphone_sources else None,
        additional_instruction=require_sources_instruction("iPhone Emergency SOS via satellite - physical conditions"),
    )


async def build_verizon_comp_subtree(evaluator: Evaluator, parent) -> None:
    vz_info = await evaluator.extract(
        prompt=prompt_extract_verizon_comp(),
        template_class=VerizonCompInfo,
        extraction_name="verizon_compensation_info",
    )

    node = evaluator.add_parallel(
        id="Customer_Compensation",
        desc="Information about Verizon's compensation offered to customers affected by the January 2026 outage",
        parent=parent,
        critical=False,
    )

    # Account_Credit_Amount (critical)
    credit_amount_leaf = evaluator.add_leaf(
        id="Account_Credit_Amount",
        desc="The dollar amount of the account credit offered to affected customers",
        parent=node,
        critical=True,
    )
    claim_amount = f"Verizon offered an account credit amount of {vz_info.verizon_credit_amount} to affected customers."
    await evaluator.verify(
        claim=claim_amount,
        node=credit_amount_leaf,
        sources=vz_info.verizon_sources if vz_info.verizon_sources else None,
        additional_instruction=require_sources_instruction("Verizon outage compensation - credit amount"),
    )

    # Credit_Redemption_Process (critical)
    redemption_leaf = evaluator.add_leaf(
        id="Credit_Redemption_Process",
        desc="How affected customers can claim or access their account credit",
        parent=node,
        critical=True,
    )
    claim_redemption = f"Customers can redeem or receive this credit as follows: {vz_info.verizon_credit_redemption}."
    await evaluator.verify(
        claim=claim_redemption,
        node=redemption_leaf,
        sources=vz_info.verizon_sources if vz_info.verizon_sources else None,
        additional_instruction=require_sources_instruction("Verizon outage compensation - redemption process"),
    )


async def build_reliability_subtree(evaluator: Evaluator, parent) -> None:
    rel_info = await evaluator.extract(
        prompt=prompt_extract_reliability(),
        template_class=ReliabilityInfo,
        extraction_name="carrier_reliability_info",
    )

    node = evaluator.add_parallel(
        id="Carrier_Reliability_Rankings",
        desc="Recent carrier network reliability rankings from independent testing organizations",
        parent=parent,
        critical=False,
    )

    # AT&T RootMetrics H1 2025 state wins (critical)
    att_leaf = evaluator.add_leaf(
        id="AT&T_Reliability_Performance",
        desc="AT&T's network reliability award count from RootMetrics testing in the first half of 2025",
        parent=node,
        critical=True,
    )
    claim_att = (
        f"In the first half of 2025, AT&T won Reliability RootScore Awards in "
        f"{rel_info.att_rootmetrics_states_won_h1_2025} states."
    )
    await evaluator.verify(
        claim=claim_att,
        node=att_leaf,
        sources=rel_info.rootmetrics_sources if rel_info.rootmetrics_sources else None,
        additional_instruction=require_sources_instruction("RootMetrics H1 2025 - AT&T state reliability wins"),
    )

    # Opensignal Reliability Experience tie (critical container)
    open_node = evaluator.add_parallel(
        id="Opensignal_Reliability_Tie",
        desc="Information about carriers that tied in Opensignal's January 2025 Reliability Experience award",
        parent=node,
        critical=True,
    )

    carriers_str = ", ".join(rel_info.opensignal_tied_carriers) if rel_info.opensignal_tied_carriers else "None"

    tied_leaf = evaluator.add_leaf(
        id="Tied_Carrier_Names",
        desc="The names of the two carriers that shared the Opensignal Reliability Experience award",
        parent=open_node,
        critical=True,
    )
    claim_tied = (
        f"In Opensignal's January 2025 USA report, the Reliability Experience award was a tie between: {carriers_str}."
    )
    await evaluator.verify(
        claim=claim_tied,
        node=tied_leaf,
        sources=rel_info.opensignal_sources if rel_info.opensignal_sources else None,
        additional_instruction=require_sources_instruction("Opensignal Jan 2025 - tied carriers for Reliability Experience"),
    )

    score_leaf = evaluator.add_leaf(
        id="Reliability_Score",
        desc="The numerical score (on 100-1000 scale) that the tied carriers received",
        parent=open_node,
        critical=True,
    )
    claim_score = f"The tied carriers both received a Reliability Experience score of {rel_info.opensignal_reliability_score}."
    await evaluator.verify(
        claim=claim_score,
        node=score_leaf,
        sources=rel_info.opensignal_sources if rel_info.opensignal_sources else None,
        additional_instruction=require_sources_instruction("Opensignal Jan 2025 - Reliability Experience score"),
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
    Evaluate an answer for the Verizon outage aftermath task using obj_task_eval evaluator.
    """
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

    # Build verification subtrees (can run concurrently at the group level)
    await asyncio.gather(
        build_fcc_process_subtree(evaluator, root),
        build_iphone_subtree(evaluator, root),
        build_verizon_comp_subtree(evaluator, root),
        build_reliability_subtree(evaluator, root),
    )

    return evaluator.get_summary()