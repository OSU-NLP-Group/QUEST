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
TASK_ID = "dhs_shutdown_feb2026"
TASK_DESCRIPTION = (
    "Provide a comprehensive analysis of the February 2026 Department of Homeland Security shutdown. "
    "Your analysis must include: (1) the specific Senate vote details that triggered the shutdown, including the date of the vote, "
    "the bill number, the exact vote count, the number of votes required for passage, and which party blocked the legislation; "
    "(2) information about when the DHS shutdown began and what mechanism caused it to occur; "
    "(3) a comparison with the previous government shutdown in FY2026, including its start date, end date, duration in days, and its significance in US history; "
    "(4) context about the bipartisan spending deal that was reached around the same time, including how many agencies were funded in that deal, "
    "whether DHS was included, and when the deal was reached. Additionally, if available, provide explanations of relevant legislative procedures such as the Senate cloture rule "
    "and continuing resolutions, as well as information about the typical impacts of government shutdowns on federal workers and government services."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SenateVoteDetails(BaseModel):
    vote_date: Optional[str] = None
    vote_date_sources: List[str] = Field(default_factory=list)

    bill_number: Optional[str] = None
    bill_number_sources: List[str] = Field(default_factory=list)

    vote_count: Optional[str] = None  # e.g., "52-47"
    vote_count_sources: List[str] = Field(default_factory=list)

    votes_required: Optional[str] = None  # e.g., "60"
    threshold_sources: List[str] = Field(default_factory=list)

    blocking_party: Optional[str] = None  # e.g., "Senate Democrats"
    party_sources: List[str] = Field(default_factory=list)


class ShutdownTimelineInfo(BaseModel):
    start_desc: Optional[str] = None  # e.g., "midnight after February 12, 2026"
    start_sources: List[str] = Field(default_factory=list)

    trigger_desc: Optional[str] = None  # e.g., "failed Senate vote on DHS funding caused funding lapse"
    trigger_sources: List[str] = Field(default_factory=list)


class PreviousShutdownInfo(BaseModel):
    prev_start_date: Optional[str] = None
    prev_end_date: Optional[str] = None
    prev_dates_sources: List[str] = Field(default_factory=list)

    prev_duration_days: Optional[str] = None  # e.g., "43"
    prev_duration_sources: List[str] = Field(default_factory=list)

    prev_record_desc: Optional[str] = None  # e.g., "longest shutdown in modern US history"
    prev_record_sources: List[str] = Field(default_factory=list)


class BipartisanDealInfo(BaseModel):
    agencies_funded_count: Optional[str] = None  # e.g., "five"
    dhs_included: Optional[str] = None  # e.g., "excluded" or "included"
    deal_sources: List[str] = Field(default_factory=list)

    deal_timing: Optional[str] = None  # e.g., "late January 2026"
    timing_sources: List[str] = Field(default_factory=list)


class LegislativeProceduresInfo(BaseModel):
    cloture_rule_desc: Optional[str] = None  # e.g., "Three-fifths (60 votes) needed to invoke cloture and end debate"
    cloture_sources: List[str] = Field(default_factory=list)

    cr_desc: Optional[str] = None  # e.g., "A CR is a temporary funding measure to avoid shutdowns"
    cr_sources: List[str] = Field(default_factory=list)


class ShutdownImpactsInfo(BaseModel):
    worker_impact_desc: Optional[str] = None  # e.g., "Federal workers do not receive paychecks during shutdowns"
    worker_sources: List[str] = Field(default_factory=list)

    service_impact_desc: Optional[str] = None  # e.g., "Certain government services go dark during shutdowns"
    service_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_senate_vote_details() -> str:
    return (
        "Extract details about the Senate vote that triggered the DHS shutdown as stated in the answer. "
        "Return a JSON object with the following fields, using exactly what the answer provides:\n"
        "1. vote_date: the date of the Senate vote (string)\n"
        "2. vote_date_sources: an array of URLs explicitly cited for the vote date\n"
        "3. bill_number: the official bill designation for the DHS funding legislation (e.g., H.R. 7147)\n"
        "4. bill_number_sources: URLs cited for the bill number\n"
        "5. vote_count: the exact vote tally (e.g., '52-47')\n"
        "6. vote_count_sources: URLs cited for the vote count\n"
        "7. votes_required: the number of votes required for passage/cloture (e.g., '60')\n"
        "8. threshold_sources: URLs cited for the cloture requirement\n"
        "9. blocking_party: which party blocked the bill (string)\n"
        "10. party_sources: URLs cited for the blocking party info\n\n"
        "Rules for URLs: extract only valid URLs explicitly present in the answer. If a field or sources are missing, set them to null or empty array."
    )


def prompt_extract_shutdown_timeline() -> str:
    return (
        "Extract timeline information for the DHS shutdown from the answer. "
        "Return: \n"
        "1. start_desc: description of when the shutdown began (e.g., 'midnight after February 12, 2026')\n"
        "2. start_sources: URLs cited for the shutdown start time\n"
        "3. trigger_desc: description of what caused the shutdown (e.g., failed Senate vote causing funding to lapse)\n"
        "4. trigger_sources: URLs cited for the shutdown trigger\n"
        "If any field is missing, set it to null or an empty array accordingly."
    )


def prompt_extract_previous_shutdown_comparison() -> str:
    return (
        "Extract details about the previous FY2026 government shutdown as referenced in the answer. Return:\n"
        "1. prev_start_date: the start date (string)\n"
        "2. prev_end_date: the end date (string)\n"
        "3. prev_dates_sources: URLs cited for the dates\n"
        "4. prev_duration_days: duration in days (string)\n"
        "5. prev_duration_sources: URLs cited for duration\n"
        "6. prev_record_desc: description of its historical significance (e.g., longest shutdown)\n"
        "7. prev_record_sources: URLs cited for historical comparison\n"
        "Use only URLs explicitly present in the answer. Missing items should be null or empty arrays."
    )


def prompt_extract_bipartisan_context() -> str:
    return (
        "Extract information about the bipartisan spending deal around the same time. Return:\n"
        "1. agencies_funded_count: how many agencies were funded in the deal (string or number)\n"
        "2. dhs_included: whether DHS was included or excluded (string)\n"
        "3. deal_sources: URLs cited for the agencies funded and DHS inclusion/exclusion\n"
        "4. deal_timing: when the deal was reached (e.g., 'late January 2026')\n"
        "5. timing_sources: URLs cited for the timing\n"
        "Extract only what is explicitly present in the answer; set missing values to null or empty arrays."
    )


def prompt_extract_legislative_procedures() -> str:
    return (
        "Extract explanations of legislative procedures from the answer if provided. Return:\n"
        "1. cloture_rule_desc: explanation of the Senate cloture rule (string)\n"
        "2. cloture_sources: URLs cited for cloture rule explanation\n"
        "3. cr_desc: explanation of continuing resolutions (string)\n"
        "4. cr_sources: URLs cited for CR explanation\n"
        "If any explanations or sources are missing, set them to null or empty arrays."
    )


def prompt_extract_shutdown_impacts() -> str:
    return (
        "Extract typical impacts of government shutdowns from the answer if provided. Return:\n"
        "1. worker_impact_desc: impact on federal workers (string)\n"
        "2. worker_sources: URLs cited for worker impact\n"
        "3. service_impact_desc: impact on government services (string)\n"
        "4. service_sources: URLs cited for service impact\n"
        "Use only URLs explicitly present; set missing values to null or empty arrays."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_senate_vote_details(
    evaluator: Evaluator,
    parent_node,
    info: SenateVoteDetails,
) -> None:
    sen_node = evaluator.add_parallel(
        id="senate_vote_details",
        desc="Accurate details about the Senate vote that triggered the DHS shutdown",
        parent=parent_node,
        critical=True,
    )

    # Vote date
    vote_date_node = evaluator.add_parallel(
        id="vote_date",
        desc="The date of the Senate vote on the DHS funding bill",
        parent=sen_node,
        critical=True,
    )
    date_details_node = evaluator.add_parallel(
        id="date_details",
        desc="Date verification and source documentation",
        parent=vote_date_node,
        critical=True,
    )
    date_source_exists = evaluator.add_custom_node(
        result=bool(info.vote_date_sources),
        id="date_source",
        desc="URL reference for vote date",
        parent=date_details_node,
        critical=True,
    )
    date_verif_leaf = evaluator.add_leaf(
        id="date_verification",
        desc="Vote occurred on February 12, 2026",
        parent=date_details_node,
        critical=True,
    )
    claim_date = (
        f"The Senate vote on the DHS funding bill occurred on {info.vote_date}."
        if info.vote_date else "The Senate vote date is specified."
    )
    await evaluator.verify(
        claim=claim_date,
        node=date_verif_leaf,
        sources=info.vote_date_sources,
        additional_instruction="Verify the exact date of the relevant Senate vote; allow minor format differences (e.g., abbreviations).",
    )

    # Bill designation
    bill_node = evaluator.add_parallel(
        id="bill_designation",
        desc="The official bill number for the DHS funding legislation",
        parent=sen_node,
        critical=True,
    )
    bill_details_node = evaluator.add_parallel(
        id="bill_details",
        desc="Bill number verification and source documentation",
        parent=bill_node,
        critical=True,
    )
    bill_source_exists = evaluator.add_custom_node(
        result=bool(info.bill_number_sources),
        id="bill_source",
        desc="URL reference for bill number",
        parent=bill_details_node,
        critical=True,
    )
    bill_verif_leaf = evaluator.add_leaf(
        id="bill_number_verification",
        desc="Bill designated as H.R. 7147",
        parent=bill_details_node,
        critical=True,
    )
    claim_bill = (
        f"The DHS funding bill was designated {info.bill_number}."
        if info.bill_number else "The DHS funding bill designation is specified."
    )
    await evaluator.verify(
        claim=claim_bill,
        node=bill_verif_leaf,
        sources=info.bill_number_sources,
        additional_instruction="Confirm the bill designation (e.g., H.R. 7147) on the cited source.",
    )

    # Vote count
    count_node = evaluator.add_parallel(
        id="vote_count",
        desc="The exact vote tally on the DHS funding bill",
        parent=sen_node,
        critical=True,
    )
    tally_details_node = evaluator.add_parallel(
        id="tally_details",
        desc="Vote tally verification and source documentation",
        parent=count_node,
        critical=True,
    )
    tally_source_exists = evaluator.add_custom_node(
        result=bool(info.vote_count_sources),
        id="tally_source",
        desc="URL reference for vote count",
        parent=tally_details_node,
        critical=True,
    )
    tally_verif_leaf = evaluator.add_leaf(
        id="tally_verification",
        desc="Vote count was 52-47",
        parent=tally_details_node,
        critical=True,
    )
    claim_tally = (
        f"The vote tally was {info.vote_count}."
        if info.vote_count else "The vote tally is specified."
    )
    await evaluator.verify(
        claim=claim_tally,
        node=tally_verif_leaf,
        sources=info.vote_count_sources,
        additional_instruction="Verify the exact vote tally on the cited page; allow formatting variations.",
    )

    # Cloture threshold
    cloture_node = evaluator.add_parallel(
        id="cloture_threshold",
        desc="The number of votes required for passage",
        parent=sen_node,
        critical=True,
    )
    threshold_details_node = evaluator.add_parallel(
        id="threshold_details",
        desc="Cloture threshold verification and source documentation",
        parent=cloture_node,
        critical=True,
    )
    threshold_source_exists = evaluator.add_custom_node(
        result=bool(info.threshold_sources),
        id="threshold_source",
        desc="URL reference for cloture requirement",
        parent=threshold_details_node,
        critical=True,
    )
    threshold_verif_leaf = evaluator.add_leaf(
        id="threshold_verification",
        desc="60 votes required for cloture",
        parent=threshold_details_node,
        critical=True,
    )
    claim_threshold = (
        f"{info.votes_required} votes are required to invoke cloture in the U.S. Senate."
        if info.votes_required else "60 votes are required to invoke cloture in the U.S. Senate."
    )
    await evaluator.verify(
        claim=claim_threshold,
        node=threshold_verif_leaf,
        sources=info.threshold_sources,
        additional_instruction="Verify the Senate cloture threshold (three-fifths, typically 60 votes) on authoritative sources (e.g., Senate.gov).",
    )

    # Blocking party
    party_node = evaluator.add_parallel(
        id="blocking_party",
        desc="Which party blocked the funding bill",
        parent=sen_node,
        critical=True,
    )
    party_details_node = evaluator.add_parallel(
        id="party_details",
        desc="Blocking party verification and source documentation",
        parent=party_node,
        critical=True,
    )
    party_source_exists = evaluator.add_custom_node(
        result=bool(info.party_sources),
        id="party_source",
        desc="URL reference for party blocking information",
        parent=party_details_node,
        critical=True,
    )
    party_verif_leaf = evaluator.add_leaf(
        id="party_verification",
        desc="Senate Democrats blocked the bill",
        parent=party_details_node,
        critical=True,
    )
    claim_party = (
        f"{info.blocking_party} blocked the DHS funding bill."
        if info.blocking_party else "A specific party blocked the DHS funding bill."
    )
    await evaluator.verify(
        claim=claim_party,
        node=party_verif_leaf,
        sources=info.party_sources,
        additional_instruction="Verify which party blocked advancement of the DHS funding bill on the cited sources.",
    )


async def build_shutdown_timeline(
    evaluator: Evaluator,
    parent_node,
    info: ShutdownTimelineInfo,
) -> None:
    timeline_node = evaluator.add_parallel(
        id="shutdown_timeline",
        desc="Accurate timeline information for the DHS shutdown",
        parent=parent_node,
        critical=True,
    )

    # Start time
    start_node = evaluator.add_parallel(
        id="shutdown_start",
        desc="When the DHS shutdown began",
        parent=timeline_node,
        critical=True,
    )
    start_details_node = evaluator.add_parallel(
        id="start_details",
        desc="Shutdown start verification and source documentation",
        parent=start_node,
        critical=True,
    )
    start_source_exists = evaluator.add_custom_node(
        result=bool(info.start_sources),
        id="start_source",
        desc="URL reference for shutdown start time",
        parent=start_details_node,
        critical=True,
    )
    start_verif_leaf = evaluator.add_leaf(
        id="start_verification",
        desc="Shutdown began at midnight after February 12, 2026",
        parent=start_details_node,
        critical=True,
    )
    claim_start = (
        f"The DHS shutdown began at {info.start_desc}."
        if info.start_desc else "The DHS shutdown began at midnight after February 12, 2026."
    )
    await evaluator.verify(
        claim=claim_start,
        node=start_verif_leaf,
        sources=info.start_sources,
        additional_instruction="Confirm the precise start timing of the DHS shutdown; allow minor phrasing variations.",
    )

    # Trigger mechanism
    mech_node = evaluator.add_parallel(
        id="trigger_mechanism",
        desc="What caused the shutdown to occur",
        parent=timeline_node,
        critical=True,
    )
    mech_details_node = evaluator.add_parallel(
        id="mechanism_details",
        desc="Trigger mechanism verification and source documentation",
        parent=mech_node,
        critical=True,
    )
    mech_source_exists = evaluator.add_custom_node(
        result=bool(info.trigger_sources),
        id="mechanism_source",
        desc="URL reference for shutdown trigger",
        parent=mech_details_node,
        critical=True,
    )
    mech_verif_leaf = evaluator.add_leaf(
        id="mechanism_verification",
        desc="Failed Senate vote on funding bill led to shutdown",
        parent=mech_details_node,
        critical=True,
    )
    claim_mech = (
        f"The DHS shutdown occurred because {info.trigger_desc}."
        if info.trigger_desc else "The DHS shutdown occurred because the Senate failed to pass the DHS funding bill, causing funding to lapse."
    )
    await evaluator.verify(
        claim=claim_mech,
        node=mech_verif_leaf,
        sources=info.trigger_sources,
        additional_instruction="Verify the causal connection between the failed Senate vote and the DHS funding lapse resulting in shutdown.",
    )


async def build_previous_shutdown_comparison(
    evaluator: Evaluator,
    parent_node,
    info: PreviousShutdownInfo,
) -> None:
    prev_node = evaluator.add_parallel(
        id="previous_shutdown_comparison",
        desc="Accurate comparison with the previous FY2026 shutdown",
        parent=parent_node,
        critical=True,
    )

    # Dates
    dates_node = evaluator.add_parallel(
        id="previous_shutdown_dates",
        desc="Start and end dates of the previous shutdown",
        parent=prev_node,
        critical=True,
    )
    dates_details_node = evaluator.add_parallel(
        id="dates_details",
        desc="Previous shutdown dates verification and source documentation",
        parent=dates_node,
        critical=True,
    )
    prev_dates_source_exists = evaluator.add_custom_node(
        result=bool(info.prev_dates_sources),
        id="previous_dates_source",
        desc="URL reference for previous shutdown dates",
        parent=dates_details_node,
        critical=True,
    )
    prev_start_leaf = evaluator.add_leaf(
        id="previous_start_verification",
        desc="Previous shutdown started October 1, 2025",
        parent=dates_details_node,
        critical=True,
    )
    prev_end_leaf = evaluator.add_leaf(
        id="previous_end_verification",
        desc="Previous shutdown ended November 12, 2025",
        parent=dates_details_node,
        critical=True,
    )
    claim_prev_start = (
        f"The previous shutdown started on {info.prev_start_date}."
        if info.prev_start_date else "The previous shutdown start date is specified."
    )
    claim_prev_end = (
        f"The previous shutdown ended on {info.prev_end_date}."
        if info.prev_end_date else "The previous shutdown end date is specified."
    )
    await evaluator.verify(
        claim=claim_prev_start,
        node=prev_start_leaf,
        sources=info.prev_dates_sources,
        additional_instruction="Verify the previous shutdown start date listed on the cited source.",
    )
    await evaluator.verify(
        claim=claim_prev_end,
        node=prev_end_leaf,
        sources=info.prev_dates_sources,
        additional_instruction="Verify the previous shutdown end date listed on the cited source.",
    )

    # Duration
    dur_node = evaluator.add_parallel(
        id="previous_shutdown_duration",
        desc="Length of the previous shutdown",
        parent=prev_node,
        critical=True,
    )
    dur_details_node = evaluator.add_parallel(
        id="duration_details",
        desc="Duration verification and source documentation",
        parent=dur_node,
        critical=True,
    )
    dur_source_exists = evaluator.add_custom_node(
        result=bool(info.prev_duration_sources),
        id="duration_source",
        desc="URL reference for shutdown duration",
        parent=dur_details_node,
        critical=True,
    )
    dur_verif_leaf = evaluator.add_leaf(
        id="duration_verification",
        desc="Previous shutdown lasted 43 days",
        parent=dur_details_node,
        critical=True,
    )
    claim_duration = (
        f"The previous shutdown lasted {info.prev_duration_days} days."
        if info.prev_duration_days else "The previous shutdown duration is specified."
    )
    await evaluator.verify(
        claim=claim_duration,
        node=dur_verif_leaf,
        sources=info.prev_duration_sources,
        additional_instruction="Verify the total duration (in days) of the previous shutdown on the cited source.",
    )

    # Historical record
    record_node = evaluator.add_parallel(
        id="historical_record",
        desc="How the previous shutdown compared historically",
        parent=prev_node,
        critical=True,
    )
    record_details_node = evaluator.add_parallel(
        id="record_details",
        desc="Historical record verification and source documentation",
        parent=record_node,
        critical=True,
    )
    record_source_exists = evaluator.add_custom_node(
        result=bool(info.prev_record_sources),
        id="record_source",
        desc="URL reference for historical comparison",
        parent=record_details_node,
        critical=True,
    )
    record_verif_leaf = evaluator.add_leaf(
        id="record_verification",
        desc="Was the longest government shutdown in modern US history",
        parent=record_details_node,
        critical=True,
    )
    claim_record = (
        f"The previous shutdown was {info.prev_record_desc}."
        if info.prev_record_desc else "The previous shutdown was the longest government shutdown in modern U.S. history."
    )
    await evaluator.verify(
        claim=claim_record,
        node=record_verif_leaf,
        sources=info.prev_record_sources,
        additional_instruction="Verify the comparative historical significance (e.g., longest shutdown) on reliable sources.",
    )


async def build_bipartisan_context(
    evaluator: Evaluator,
    parent_node,
    info: BipartisanDealInfo,
) -> None:
    context_node = evaluator.add_parallel(
        id="bipartisan_context",
        desc="Information about the bipartisan spending deal context",
        parent=parent_node,
        critical=True,
    )

    # Five-agency deal details
    five_node = evaluator.add_parallel(
        id="five_agency_deal",
        desc="Details about the bipartisan agreement reached for other agencies",
        parent=context_node,
        critical=True,
    )
    deal_details_node = evaluator.add_parallel(
        id="deal_details",
        desc="Agency deal verification and source documentation",
        parent=five_node,
        critical=True,
    )
    deal_source_exists = evaluator.add_custom_node(
        result=bool(info.deal_sources),
        id="deal_source",
        desc="URL reference for bipartisan deal details",
        parent=deal_details_node,
        critical=True,
    )
    agencies_verif_leaf = evaluator.add_leaf(
        id="agencies_funded_verification",
        desc="Five agencies were funded in bipartisan deal",
        parent=deal_details_node,
        critical=True,
    )
    dhs_excl_leaf = evaluator.add_leaf(
        id="dhs_exclusion_verification",
        desc="DHS was excluded from the bipartisan package",
        parent=deal_details_node,
        critical=True,
    )
    claim_agencies = (
        f"{info.agencies_funded_count} agencies were funded in the bipartisan deal."
        if info.agencies_funded_count else "Five agencies were funded in the bipartisan deal."
    )
    claim_dhs = (
        f"DHS was {info.dhs_included} from the bipartisan package."
        if info.dhs_included else "DHS was excluded from the bipartisan package."
    )
    await evaluator.verify(
        claim=claim_agencies,
        node=agencies_verif_leaf,
        sources=info.deal_sources,
        additional_instruction="Verify the number of agencies funded in the bipartisan deal on the cited sources.",
    )
    await evaluator.verify(
        claim=claim_dhs,
        node=dhs_excl_leaf,
        sources=info.deal_sources,
        additional_instruction="Verify whether DHS was included or excluded in the bipartisan spending package.",
    )

    # Deal timing
    timing_node = evaluator.add_parallel(
        id="deal_timing",
        desc="When the bipartisan deal was reached",
        parent=context_node,
        critical=True,
    )
    timing_details_node = evaluator.add_parallel(
        id="timing_details",
        desc="Deal timing verification and source documentation",
        parent=timing_node,
        critical=True,
    )
    timing_source_exists = evaluator.add_custom_node(
        result=bool(info.timing_sources),
        id="timing_source",
        desc="URL reference for deal timing",
        parent=timing_details_node,
        critical=True,
    )
    timing_verif_leaf = evaluator.add_leaf(
        id="timing_verification",
        desc="Deal reached in late January 2026",
        parent=timing_details_node,
        critical=True,
    )
    claim_timing = (
        f"The bipartisan deal was reached in {info.deal_timing}."
        if info.deal_timing else "The bipartisan deal was reached in late January 2026."
    )
    await evaluator.verify(
        claim=claim_timing,
        node=timing_verif_leaf,
        sources=info.timing_sources,
        additional_instruction="Verify the timing of the deal; consider 'late January 2026' to include end-of-month dates.",
    )


async def build_legislative_procedures(
    evaluator: Evaluator,
    parent_node,
    info: LegislativeProceduresInfo,
) -> None:
    proc_node = evaluator.add_parallel(
        id="legislative_procedures",
        desc="Understanding of relevant legislative procedures",
        parent=parent_node,
        critical=False,
    )

    # Cloture rule explanation
    cloture_node = evaluator.add_parallel(
        id="cloture_rule",
        desc="Explanation of the Senate cloture rule",
        parent=proc_node,
        critical=False,
    )
    cloture_details_node = evaluator.add_parallel(
        id="cloture_details",
        desc="Cloture rule explanation and source documentation",
        parent=cloture_node,
        critical=False,
    )
    cloture_source_exists = evaluator.add_custom_node(
        result=bool(info.cloture_sources),
        id="cloture_source",
        desc="URL reference for cloture rule explanation",
        parent=cloture_details_node,
        critical=False,
    )
    cloture_def_leaf = evaluator.add_leaf(
        id="cloture_definition",
        desc="Three-fifths (60 votes) needed to invoke cloture and end debate",
        parent=cloture_details_node,
        critical=False,
    )
    claim_cloture = (
        f"{info.cloture_rule_desc}" if info.cloture_rule_desc
        else "Three-fifths (typically 60 votes) are needed to invoke cloture and end debate in the U.S. Senate."
    )
    await evaluator.verify(
        claim=claim_cloture,
        node=cloture_def_leaf,
        sources=info.cloture_sources,
        additional_instruction="Confirm the cloture threshold and purpose from authoritative sources (e.g., Senate.gov).",
    )

    # Continuing resolution explanation
    cr_node = evaluator.add_parallel(
        id="continuing_resolution",
        desc="Explanation of continuing resolution mechanism",
        parent=proc_node,
        critical=False,
    )
    cr_details_node = evaluator.add_parallel(
        id="cr_details",
        desc="CR explanation and source documentation",
        parent=cr_node,
        critical=False,
    )
    cr_source_exists = evaluator.add_custom_node(
        result=bool(info.cr_sources),
        id="cr_source",
        desc="URL reference for CR explanation",
        parent=cr_details_node,
        critical=False,
    )
    cr_def_leaf = evaluator.add_leaf(
        id="cr_definition",
        desc="CR is temporary funding measure to avoid shutdowns",
        parent=cr_details_node,
        critical=False,
    )
    claim_cr = (
        f"{info.cr_desc}" if info.cr_desc
        else "A continuing resolution (CR) is a temporary funding measure used to avoid shutdowns."
    )
    await evaluator.verify(
        claim=claim_cr,
        node=cr_def_leaf,
        sources=info.cr_sources,
        additional_instruction="Verify the role of continuing resolutions as temporary funding to prevent shutdowns.",
    )


async def build_shutdown_impacts(
    evaluator: Evaluator,
    parent_node,
    info: ShutdownImpactsInfo,
) -> None:
    impacts_node = evaluator.add_parallel(
        id="shutdown_impacts",
        desc="Information about the impacts of government shutdowns",
        parent=parent_node,
        critical=False,
    )

    # Federal workers
    workers_node = evaluator.add_parallel(
        id="federal_workers",
        desc="Impact on federal employees",
        parent=impacts_node,
        critical=False,
    )
    worker_details_node = evaluator.add_parallel(
        id="worker_details",
        desc="Federal worker impact and source documentation",
        parent=workers_node,
        critical=False,
    )
    worker_source_exists = evaluator.add_custom_node(
        result=bool(info.worker_sources),
        id="worker_source",
        desc="URL reference for worker impact information",
        parent=worker_details_node,
        critical=False,
    )
    worker_pay_leaf = evaluator.add_leaf(
        id="paycheck_impact",
        desc="Federal workers do not receive paychecks during shutdowns",
        parent=worker_details_node,
        critical=False,
    )
    claim_workers = (
        f"{info.worker_impact_desc}" if info.worker_impact_desc
        else "Federal workers do not receive paychecks during shutdowns (until back pay is later authorized)."
    )
    await evaluator.verify(
        claim=claim_workers,
        node=worker_pay_leaf,
        sources=info.worker_sources,
        additional_instruction="Verify typical payroll impacts for federal workers during shutdowns; allow references to furloughs and back pay.",
    )

    # Government services
    services_node = evaluator.add_parallel(
        id="services",
        desc="Impact on government services",
        parent=impacts_node,
        critical=False,
    )
    service_details_node = evaluator.add_parallel(
        id="service_details",
        desc="Government service impact and source documentation",
        parent=services_node,
        critical=False,
    )
    service_source_exists = evaluator.add_custom_node(
        result=bool(info.service_sources),
        id="service_source",
        desc="URL reference for service impact information",
        parent=service_details_node,
        critical=False,
    )
    service_impact_leaf = evaluator.add_leaf(
        id="service_impact",
        desc="Certain government services go dark during shutdowns",
        parent=service_details_node,
        critical=False,
    )
    claim_services = (
        f"{info.service_impact_desc}" if info.service_impact_desc
        else "Certain government services are curtailed or go dark during shutdowns."
    )
    await evaluator.verify(
        claim=claim_services,
        node=service_impact_leaf,
        sources=info.service_sources,
        additional_instruction="Verify typical impacts on government services during shutdowns (e.g., closures, delays).",
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the February 2026 DHS shutdown analysis task.
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

    # Extract data in parallel
    senate_task = evaluator.extract(
        prompt=prompt_extract_senate_vote_details(),
        template_class=SenateVoteDetails,
        extraction_name="senate_vote_details",
    )
    timeline_task = evaluator.extract(
        prompt=prompt_extract_shutdown_timeline(),
        template_class=ShutdownTimelineInfo,
        extraction_name="shutdown_timeline",
    )
    previous_task = evaluator.extract(
        prompt=prompt_extract_previous_shutdown_comparison(),
        template_class=PreviousShutdownInfo,
        extraction_name="previous_shutdown_info",
    )
    deal_task = evaluator.extract(
        prompt=prompt_extract_bipartisan_context(),
        template_class=BipartisanDealInfo,
        extraction_name="bipartisan_deal_info",
    )
    proc_task = evaluator.extract(
        prompt=prompt_extract_legislative_procedures(),
        template_class=LegislativeProceduresInfo,
        extraction_name="legislative_procedures_info",
    )
    impacts_task = evaluator.extract(
        prompt=prompt_extract_shutdown_impacts(),
        template_class=ShutdownImpactsInfo,
        extraction_name="shutdown_impacts_info",
    )

    (
        senate_info,
        timeline_info,
        previous_info,
        deal_info,
        procedures_info,
        impacts_info,
    ) = await asyncio.gather(
        senate_task, timeline_task, previous_task, deal_task, proc_task, impacts_task
    )

    # Build verification tree according to rubric
    await build_senate_vote_details(evaluator, root, senate_info)
    await build_shutdown_timeline(evaluator, root, timeline_info)
    await build_previous_shutdown_comparison(evaluator, root, previous_info)
    await build_bipartisan_context(evaluator, root, deal_info)
    await build_legislative_procedures(evaluator, root, procedures_info)
    await build_shutdown_impacts(evaluator, root, impacts_info)

    # Return structured summary
    return evaluator.get_summary()