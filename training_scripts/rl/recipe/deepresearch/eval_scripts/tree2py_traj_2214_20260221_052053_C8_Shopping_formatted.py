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
TASK_ID = "holiday_shopping_2025_2026"
TASK_DESCRIPTION = (
    "Create a comprehensive holiday shopping reference guide for the 2025-2026 Christmas and New Year period. "
    "For the following three dates, provide specific store hours, availability status, and service deadlines for major retailers:\n\n"
    "1. Christmas Eve (December 24, 2025):\n"
    "- What time does Walmart close?\n"
    "- What are Target's operating hours?\n"
    "- What are CVS's operating hours?\n"
    "- Is Walgreens open, and do pharmacy hours vary by location?\n"
    "- What are Kohl's operating hours?\n"
    "- What is the cutoff time for Walmart Express Delivery orders?\n"
    "- What is the cutoff time for Walmart same-day pickup orders?\n"
    "- What is the cutoff time for Target Drive-Up and Pickup orders?\n\n"
    "2. Christmas Day (December 25, 2025):\n"
    "- Is CVS open, and if so, what are the typical operating hours?\n"
    "- Is Walgreens open, and do pharmacy hours vary by location?\n"
    "- Is 7-Eleven open, and does it maintain 24/7 operations?\n"
    "- Are Walmart and Target closed?\n\n"
    "3. New Year's Day (January 1, 2026):\n"
    "- Is Walmart open, and does it operate regular hours?\n"
    "- Is Target open, and does it operate regular hours?\n"
    "- What time does Wegmans open?\n\n"
    "For each piece of information, provide the specific hours/status and include supporting reference URLs from official retailer sources or reliable news outlets."
)

# Ground-truth expectations used for value-match verification
EXPECTED = {
    "christmas_eve": {
        "walmart_close": "6 PM local time",
        "target_hours": "7 AM to 8 PM",
        "cvs_hours_typical": "10 AM to 8 PM",
        "walgreens_open": "open; pharmacy hours vary by location",
        "kohls_hours": "7 AM to 7 PM",
        "walmart_express_cutoff": "5 PM local time",
        "walmart_pickup_cutoff": "12 PM (noon) local time",
        "target_pickup_cutoff": "6 PM local time",
    },
    "christmas_day": {
        "cvs_hours_typical": "10 AM to 8 PM",
        "walgreens_open": "open; pharmacy hours vary by location",
        "seven_eleven_24_7": "24/7",
        "walmart_target_closed": "closed",
    },
    "new_years_day": {
        "walmart_regular_hours": "open with regular hours",
        "target_regular_hours": "open with regular hours",
        "wegmans_open": "6 AM",
    }
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ChristmasEveExtraction(BaseModel):
    walmart_close_time: Optional[str] = None
    walmart_close_sources: List[str] = Field(default_factory=list)

    target_hours: Optional[str] = None
    target_hours_sources: List[str] = Field(default_factory=list)

    cvs_hours: Optional[str] = None
    cvs_hours_sources: List[str] = Field(default_factory=list)

    walgreens_open_status: Optional[str] = None  # e.g., "open"
    walgreens_pharmacy_varies_text: Optional[str] = None  # e.g., "pharmacy hours vary by location"
    walgreens_sources: List[str] = Field(default_factory=list)

    kohls_hours: Optional[str] = None
    kohls_hours_sources: List[str] = Field(default_factory=list)

    walmart_express_delivery_cutoff: Optional[str] = None
    walmart_express_delivery_sources: List[str] = Field(default_factory=list)

    walmart_pickup_cutoff: Optional[str] = None
    walmart_pickup_sources: List[str] = Field(default_factory=list)

    target_pickup_cutoff: Optional[str] = None
    target_pickup_sources: List[str] = Field(default_factory=list)


class ChristmasDayExtraction(BaseModel):
    cvs_open_hours: Optional[str] = None
    cvs_day_sources: List[str] = Field(default_factory=list)

    walgreens_open_status: Optional[str] = None  # "open" or similar
    walgreens_pharmacy_varies_text: Optional[str] = None
    walgreens_day_sources: List[str] = Field(default_factory=list)

    seven_eleven_24_7: Optional[str] = None  # "24/7" or "open 24 hours"
    seven_eleven_sources: List[str] = Field(default_factory=list)

    walmart_closed: Optional[str] = None  # "closed"
    target_closed: Optional[str] = None  # "closed"
    major_retailer_closures_sources: List[str] = Field(default_factory=list)


class NewYearsDayExtraction(BaseModel):
    walmart_open_regular_hours: Optional[str] = None  # "open with regular hours"
    walmart_new_year_sources: List[str] = Field(default_factory=list)

    target_open_regular_hours: Optional[str] = None  # "open with regular hours"
    target_new_year_sources: List[str] = Field(default_factory=list)

    wegmans_open_time: Optional[str] = None  # "6 AM"
    wegmans_new_year_sources: List[str] = Field(default_factory=list)


class HolidayPlanExtraction(BaseModel):
    christmas_eve: Optional[ChristmasEveExtraction] = None
    christmas_day: Optional[ChristmasDayExtraction] = None
    new_years_day: Optional[NewYearsDayExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_holiday_plan() -> str:
    return """
    Extract the structured holiday shopping information EXACTLY as presented in the answer. 
    For each item below, extract the stated hours/status/cutoff and the cited source URLs associated with that specific item.

    IMPORTANT:
    - Only extract URLs explicitly present in the answer. Do not fabricate or infer.
    - If the answer does not provide a specific value or URL, return null for the value and an empty list for sources.
    - For boolean-like assertions (e.g., "open", "closed", "24/7", "regular hours"), store them as short text strings.

    Structure to extract:

    christmas_eve:
      walmart_close_time: string or null
      walmart_close_sources: array of URLs
      target_hours: string or null
      target_hours_sources: array of URLs
      cvs_hours: string or null
      cvs_hours_sources: array of URLs
      walgreens_open_status: string or null
      walgreens_pharmacy_varies_text: string or null
      walgreens_sources: array of URLs
      kohls_hours: string or null
      kohls_hours_sources: array of URLs
      walmart_express_delivery_cutoff: string or null
      walmart_express_delivery_sources: array of URLs
      walmart_pickup_cutoff: string or null
      walmart_pickup_sources: array of URLs
      target_pickup_cutoff: string or null
      target_pickup_sources: array of URLs

    christmas_day:
      cvs_open_hours: string or null
      cvs_day_sources: array of URLs
      walgreens_open_status: string or null
      walgreens_pharmacy_varies_text: string or null
      walgreens_day_sources: array of URLs
      seven_eleven_24_7: string or null
      seven_eleven_sources: array of URLs
      walmart_closed: string or null
      target_closed: string or null
      major_retailer_closures_sources: array of URLs

    new_years_day:
      walmart_open_regular_hours: string or null
      walmart_new_year_sources: array of URLs
      target_open_regular_hours: string or null
      target_new_year_sources: array of URLs
      wegmans_open_time: string or null
      wegmans_new_year_sources: array of URLs

    Notes:
    - Normalize minor phrasing variants (e.g., "noon" -> "12 PM (noon)"; "6 p.m." -> "6 PM") only if the answer explicitly uses a variant; otherwise, keep the exact text.
    - Sources can be official retailer pages or credible news articles, as cited in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_value_and_sources(value: Optional[str], sources: List[str]) -> bool:
    return (value is not None and str(value).strip() != "") and (sources is not None and len(sources) > 0)


def _has_all_values_and_sources(values_and_sources: List[tuple[Optional[str], List[str]]]) -> bool:
    return all(_has_value_and_sources(v, s) for v, s in values_and_sources)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_christmas_eve(evaluator: Evaluator, parent_node, data: Optional[ChristmasEveExtraction]) -> None:
    ce_node = evaluator.add_parallel(
        id="Christmas_Eve_Dec24",
        desc="Store hours and service deadlines for Christmas Eve December 24, 2025",
        parent=parent_node,
        critical=True
    )

    # Walmart Hours
    walmart_hours_node = evaluator.add_parallel(
        id="Walmart_Hours",
        desc="Walmart closing time on Christmas Eve is 6 PM local time",
        parent=ce_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(data.walmart_close_time if data else None,
                                      data.walmart_close_sources if data else []),
        id="Walmart_Hours_exists",
        desc="Walmart Christmas Eve closing time is provided with sources",
        parent=walmart_hours_node,
        critical=True
    )
    walmart_match_leaf = evaluator.add_leaf(
        id="Walmart_Hours_value_match",
        desc="Answer states Walmart closes at 6 PM local time on Christmas Eve",
        parent=walmart_hours_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), the answer states Walmart closes at 6 PM local time.",
        node=walmart_match_leaf,
        additional_instruction="Allow minor phrasing variants like '6 p.m.' or '6pm'."
    )
    walmart_source_leaf = evaluator.add_leaf(
        id="Walmart_Hours_source_support",
        desc="Sources support Walmart closing at 6 PM local time on Christmas Eve",
        parent=walmart_hours_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), Walmart stores close at 6 PM local time.",
        node=walmart_source_leaf,
        sources=(data.walmart_close_sources if data else []),
        additional_instruction="Verify using official Walmart communications or reliable news coverage. Local variations are acceptable; focus on typical/announced national guidance."
    )

    # Target Hours
    target_hours_node = evaluator.add_parallel(
        id="Target_Hours",
        desc="Target operates from 7 AM to 8 PM on Christmas Eve",
        parent=ce_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(data.target_hours if data else None,
                                      data.target_hours_sources if data else []),
        id="Target_Hours_exists",
        desc="Target Christmas Eve operating hours are provided with sources",
        parent=target_hours_node,
        critical=True
    )
    target_match_leaf = evaluator.add_leaf(
        id="Target_Hours_value_match",
        desc="Answer states Target operates 7 AM to 8 PM on Christmas Eve",
        parent=target_hours_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), the answer states Target operates from 7 AM to 8 PM.",
        node=target_match_leaf,
        additional_instruction="Allow minor phrasing variants like '7am-8pm'."
    )
    target_source_leaf = evaluator.add_leaf(
        id="Target_Hours_source_support",
        desc="Sources support Target operating 7 AM to 8 PM on Christmas Eve",
        parent=target_hours_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), Target stores operate from 7 AM to 8 PM.",
        node=target_source_leaf,
        sources=(data.target_hours_sources if data else []),
        additional_instruction="Check official Target announcements or credible news sources."
    )

    # CVS Hours
    cvs_hours_node = evaluator.add_parallel(
        id="CVS_Hours",
        desc="CVS operates with modified hours, typically 10 AM to 8 PM on Christmas Eve",
        parent=ce_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(data.cvs_hours if data else None,
                                      data.cvs_hours_sources if data else []),
        id="CVS_Hours_exists",
        desc="CVS Christmas Eve operating hours are provided with sources",
        parent=cvs_hours_node,
        critical=True
    )
    cvs_match_leaf = evaluator.add_leaf(
        id="CVS_Hours_value_match",
        desc="Answer states CVS typically operates 10 AM to 8 PM on Christmas Eve",
        parent=cvs_hours_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), the answer states CVS has modified hours, typically around 10 AM to 8 PM.",
        node=cvs_match_leaf,
        additional_instruction="Allow phrasing variants (e.g., '10 a.m. to 8 p.m.'), and recognize 'modified hours' caveats."
    )
    cvs_source_leaf = evaluator.add_leaf(
        id="CVS_Hours_source_support",
        desc="Sources support CVS typical 10 AM to 8 PM on Christmas Eve",
        parent=cvs_hours_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), CVS operates with modified hours, typically around 10 AM to 8 PM.",
        node=cvs_source_leaf,
        sources=(data.cvs_hours_sources if data else []),
        additional_instruction="Use official CVS pages or reliable news outlets; acknowledge local variations."
    )

    # Walgreens Availability
    walgreens_avail_node = evaluator.add_parallel(
        id="Walgreens_Availability",
        desc="Walgreens is open on Christmas Eve with pharmacy hours varying by location",
        parent=ce_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(
            # Treat open_status presence + vary text presence as one combined "value"
            (f"{(data.walgreens_open_status or '').strip()} { (data.walgreens_pharmacy_varies_text or '').strip() }" if data else None),
            data.walgreens_sources if data else []
        ),
        id="Walgreens_Availability_exists",
        desc="Walgreens Christmas Eve open status and pharmacy variation note are provided with sources",
        parent=walgreens_avail_node,
        critical=True
    )
    walgreens_match_leaf = evaluator.add_leaf(
        id="Walgreens_Availability_value_match",
        desc="Answer states Walgreens is open and pharmacy hours vary by location on Christmas Eve",
        parent=walgreens_avail_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), the answer states Walgreens stores are open and pharmacy hours vary by location.",
        node=walgreens_match_leaf,
        additional_instruction="Minor wording differences are acceptable as long as both 'open' and 'pharmacy hours vary by location' are conveyed."
    )
    walgreens_source_leaf = evaluator.add_leaf(
        id="Walgreens_Availability_source_support",
        desc="Sources support Walgreens open and pharmacy hours vary by location on Christmas Eve",
        parent=walgreens_avail_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), Walgreens stores are open and pharmacy hours vary by location.",
        node=walgreens_source_leaf,
        sources=(data.walgreens_sources if data else []),
        additional_instruction="Official Walgreens communications or credible news coverage should support both points."
    )

    # Kohl's Hours
    kohls_hours_node = evaluator.add_parallel(
        id="Kohls_Hours",
        desc="Kohl's operates from 7 AM to 7 PM on Christmas Eve",
        parent=ce_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(data.kohls_hours if data else None,
                                      data.kohls_hours_sources if data else []),
        id="Kohls_Hours_exists",
        desc="Kohl's Christmas Eve operating hours are provided with sources",
        parent=kohls_hours_node,
        critical=True
    )
    kohls_match_leaf = evaluator.add_leaf(
        id="Kohls_Hours_value_match",
        desc="Answer states Kohl's operates 7 AM to 7 PM on Christmas Eve",
        parent=kohls_hours_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), the answer states Kohl's operates from 7 AM to 7 PM.",
        node=kohls_match_leaf,
        additional_instruction="Allow minor formatting differences (e.g., '7am–7pm')."
    )
    kohls_source_leaf = evaluator.add_leaf(
        id="Kohls_Hours_source_support",
        desc="Sources support Kohl's operating 7 AM to 7 PM on Christmas Eve",
        parent=kohls_hours_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), Kohl's operates from 7 AM to 7 PM.",
        node=kohls_source_leaf,
        sources=(data.kohls_hours_sources if data else []),
        additional_instruction="Check Kohl's announcements or credible news coverage."
    )

    # Pickup Service Deadlines
    pickup_node = evaluator.add_parallel(
        id="Pickup_Service_Deadlines",
        desc="Verification of pickup and delivery service cutoff times for Christmas Eve",
        parent=ce_node,
        critical=True
    )

    # Walmart Express Delivery cutoff
    walmart_express_node = evaluator.add_parallel(
        id="Walmart_Express_Delivery",
        desc="Walmart Express Delivery accepts orders until 5 PM local time on December 24",
        parent=pickup_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(data.walmart_express_delivery_cutoff if data else None,
                                      data.walmart_express_delivery_sources if data else []),
        id="Walmart_Express_Delivery_exists",
        desc="Walmart Express Delivery cutoff is provided with sources",
        parent=walmart_express_node,
        critical=True
    )
    walmart_express_match_leaf = evaluator.add_leaf(
        id="Walmart_Express_Delivery_value_match",
        desc="Answer states Walmart Express Delivery cutoff is 5 PM local time on Dec 24",
        parent=walmart_express_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), the answer states Walmart Express Delivery accepts orders until 5 PM local time.",
        node=walmart_express_match_leaf,
        additional_instruction="Accept 'order by 5 PM' phrasing."
    )
    walmart_express_source_leaf = evaluator.add_leaf(
        id="Walmart_Express_Delivery_source_support",
        desc="Sources support Express Delivery cutoff at 5 PM local time on Dec 24",
        parent=walmart_express_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), Walmart Express Delivery accepts orders until 5 PM local time.",
        node=walmart_express_source_leaf,
        sources=(data.walmart_express_delivery_sources if data else []),
        additional_instruction="Use official Walmart references or credible news outlining holiday delivery cutoffs."
    )

    # Walmart same-day pickup cutoff
    walmart_pickup_node = evaluator.add_parallel(
        id="Walmart_Pickup_Deadline",
        desc="Walmart same-day pickup accepts orders until noon local time on December 24",
        parent=pickup_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(data.walmart_pickup_cutoff if data else None,
                                      data.walmart_pickup_sources if data else []),
        id="Walmart_Pickup_Deadline_exists",
        desc="Walmart same-day pickup cutoff is provided with sources",
        parent=walmart_pickup_node,
        critical=True
    )
    walmart_pickup_match_leaf = evaluator.add_leaf(
        id="Walmart_Pickup_Deadline_value_match",
        desc="Answer states Walmart same-day pickup cutoff is 12 PM (noon) local time on Dec 24",
        parent=walmart_pickup_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), the answer states Walmart same-day pickup orders are accepted until 12 PM (noon) local time.",
        node=walmart_pickup_match_leaf,
        additional_instruction="Accept 'noon' phrasing."
    )
    walmart_pickup_source_leaf = evaluator.add_leaf(
        id="Walmart_Pickup_Deadline_source_support",
        desc="Sources support Walmart same-day pickup cutoff at noon local time on Dec 24",
        parent=walmart_pickup_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), Walmart same-day pickup accepts orders until 12 PM (noon) local time.",
        node=walmart_pickup_source_leaf,
        sources=(data.walmart_pickup_sources if data else []),
        additional_instruction="Use official Walmart references or credible news outlining holiday pickup cutoffs."
    )

    # Target Drive-Up and Pickup cutoff
    target_pickup_node = evaluator.add_parallel(
        id="Target_Pickup_Deadline",
        desc="Target Drive-Up and Pickup accepts orders until 6 PM local time on December 24",
        parent=pickup_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(data.target_pickup_cutoff if data else None,
                                      data.target_pickup_sources if data else []),
        id="Target_Pickup_Deadline_exists",
        desc="Target Drive-Up/Pickup cutoff is provided with sources",
        parent=target_pickup_node,
        critical=True
    )
    target_pickup_match_leaf = evaluator.add_leaf(
        id="Target_Pickup_Deadline_value_match",
        desc="Answer states Target Drive-Up/Pickup cutoff is 6 PM local time on Dec 24",
        parent=target_pickup_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), the answer states Target Drive-Up and Pickup accept orders until 6 PM local time.",
        node=target_pickup_match_leaf,
        additional_instruction="Allow phrasing variants conveying 'order by 6 PM'."
    )
    target_pickup_source_leaf = evaluator.add_leaf(
        id="Target_Pickup_Deadline_source_support",
        desc="Sources support Target Drive-Up/Pickup cutoff at 6 PM local time on Dec 24",
        parent=target_pickup_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 24, 2025 (Christmas Eve), Target Drive-Up and Pickup accept orders until 6 PM local time.",
        node=target_pickup_source_leaf,
        sources=(data.target_pickup_sources if data else []),
        additional_instruction="Use official Target references or credible news outlining holiday pickup cutoffs."
    )


async def verify_christmas_day(evaluator: Evaluator, parent_node, data: Optional[ChristmasDayExtraction]) -> None:
    cd_node = evaluator.add_parallel(
        id="Christmas_Day_Dec25",
        desc="Store availability and hours for Christmas Day December 25, 2025",
        parent=parent_node,
        critical=True
    )

    # CVS Christmas Day
    cvs_day_node = evaluator.add_parallel(
        id="CVS_Christmas_Day",
        desc="CVS is open on Christmas Day, typically 10 AM to 8 PM at most locations",
        parent=cd_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(data.cvs_open_hours if data else None,
                                      data.cvs_day_sources if data else []),
        id="CVS_Christmas_Day_exists",
        desc="CVS Christmas Day hours are provided with sources",
        parent=cvs_day_node,
        critical=True
    )
    cvs_day_match_leaf = evaluator.add_leaf(
        id="CVS_Christmas_Day_value_match",
        desc="Answer states CVS is open with typical hours around 10 AM to 8 PM on Christmas Day",
        parent=cvs_day_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 25, 2025 (Christmas Day), the answer states CVS is open, typically around 10 AM to 8 PM.",
        node=cvs_day_match_leaf,
        additional_instruction="Allow phrasing indicating 'typical' or 'varies by location'; core range should reflect ~10 AM to 8 PM."
    )
    cvs_day_source_leaf = evaluator.add_leaf(
        id="CVS_Christmas_Day_source_support",
        desc="Sources support CVS open and typical 10 AM to 8 PM on Christmas Day",
        parent=cvs_day_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 25, 2025 (Christmas Day), CVS is open, typically around 10 AM to 8 PM.",
        node=cvs_day_source_leaf,
        sources=(data.cvs_day_sources if data else []),
        additional_instruction="Use official CVS pages or credible news; acknowledge local variations."
    )

    # Walgreens Christmas Day
    walgreens_day_node = evaluator.add_parallel(
        id="Walgreens_Christmas_Day",
        desc="Walgreens is open on Christmas Day with pharmacy hours varying by location",
        parent=cd_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(
            (f"{(data.walgreens_open_status or '').strip()} { (data.walgreens_pharmacy_varies_text or '').strip() }" if data else None),
            data.walgreens_day_sources if data else []
        ),
        id="Walgreens_Christmas_Day_exists",
        desc="Walgreens Christmas Day open status and pharmacy variation note are provided with sources",
        parent=walgreens_day_node,
        critical=True
    )
    walgreens_day_match_leaf = evaluator.add_leaf(
        id="Walgreens_Christmas_Day_value_match",
        desc="Answer states Walgreens is open and pharmacy hours vary by location on Christmas Day",
        parent=walgreens_day_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 25, 2025 (Christmas Day), the answer states Walgreens stores are open and pharmacy hours vary by location.",
        node=walgreens_day_match_leaf,
        additional_instruction="Minor wording differences are acceptable as long as both 'open' and 'pharmacy varies by location' are conveyed."
    )
    walgreens_day_source_leaf = evaluator.add_leaf(
        id="Walgreens_Christmas_Day_source_support",
        desc="Sources support Walgreens open and pharmacy variation on Christmas Day",
        parent=walgreens_day_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 25, 2025 (Christmas Day), Walgreens stores are open and pharmacy hours vary by location.",
        node=walgreens_day_source_leaf,
        sources=(data.walgreens_day_sources if data else []),
        additional_instruction="Use official Walgreens references or credible news."
    )

    # 7-Eleven Availability
    seven_eleven_node = evaluator.add_parallel(
        id="Seven_Eleven_Availability",
        desc="Most 7-Eleven stores operate 24/7 including Christmas Day",
        parent=cd_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(data.seven_eleven_24_7 if data else None,
                                      data.seven_eleven_sources if data else []),
        id="Seven_Eleven_Availability_exists",
        desc="7-Eleven Christmas Day availability (24/7) is provided with sources",
        parent=seven_eleven_node,
        critical=True
    )
    seven_eleven_match_leaf = evaluator.add_leaf(
        id="Seven_Eleven_Availability_value_match",
        desc="Answer states most 7-Eleven stores operate 24/7 on Christmas Day",
        parent=seven_eleven_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 25, 2025 (Christmas Day), the answer states most 7-Eleven stores operate 24/7.",
        node=seven_eleven_match_leaf,
        additional_instruction="Accept 'open 24 hours' phrasing; recognize franchised location exceptions."
    )
    seven_eleven_source_leaf = evaluator.add_leaf(
        id="Seven_Eleven_Availability_source_support",
        desc="Sources support most 7-Eleven stores operating 24/7 on Christmas Day",
        parent=seven_eleven_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 25, 2025 (Christmas Day), most 7-Eleven stores operate 24/7.",
        node=seven_eleven_source_leaf,
        sources=(data.seven_eleven_sources if data else []),
        additional_instruction="Use official 7-Eleven references or credible news."
    )

    # Major Retailer Closures (Walmart and Target)
    closures_node = evaluator.add_parallel(
        id="Major_Retailer_Closures",
        desc="Walmart and Target are closed on Christmas Day",
        parent=cd_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(
            # Combine both closure statements presence as the "value" together
            (f"walmart:{(data.walmart_closed or '').strip()} target:{(data.target_closed or '').strip()}" if data else None),
            data.major_retailer_closures_sources if data else []
        ),
        id="Major_Retailer_Closures_exists",
        desc="Walmart and Target closure info on Christmas Day provided with sources",
        parent=closures_node,
        critical=True
    )
    closures_match_leaf = evaluator.add_leaf(
        id="Major_Retailer_Closures_value_match",
        desc="Answer states Walmart and Target are closed on Christmas Day",
        parent=closures_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 25, 2025 (Christmas Day), the answer states both Walmart and Target are closed.",
        node=closures_match_leaf,
        additional_instruction="The statement must indicate closure for both retailers."
    )
    closures_source_leaf = evaluator.add_leaf(
        id="Major_Retailer_Closures_source_support",
        desc="Sources support Walmart and Target closures on Christmas Day",
        parent=closures_node,
        critical=True
    )
    await evaluator.verify(
        claim="On December 25, 2025 (Christmas Day), Walmart and Target are closed.",
        node=closures_source_leaf,
        sources=(data.major_retailer_closures_sources if data else []),
        additional_instruction="Use official retailer announcements or credible news reporting closures."
    )


async def verify_new_years_day(evaluator: Evaluator, parent_node, data: Optional[NewYearsDayExtraction]) -> None:
    ny_node = evaluator.add_parallel(
        id="New_Years_Day_Jan1",
        desc="Store availability and hours for New Year's Day January 1, 2026",
        parent=parent_node,
        critical=True
    )

    # Walmart New Year's
    walmart_ny_node = evaluator.add_parallel(
        id="Walmart_New_Years",
        desc="Walmart is open with regular hours on New Year's Day",
        parent=ny_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(data.walmart_open_regular_hours if data else None,
                                      data.walmart_new_year_sources if data else []),
        id="Walmart_New_Years_exists",
        desc="Walmart New Year's Day regular hours info provided with sources",
        parent=walmart_ny_node,
        critical=True
    )
    walmart_ny_match_leaf = evaluator.add_leaf(
        id="Walmart_New_Years_value_match",
        desc="Answer states Walmart is open with regular hours on New Year's Day",
        parent=walmart_ny_node,
        critical=True
    )
    await evaluator.verify(
        claim="On January 1, 2026 (New Year's Day), the answer states Walmart is open and runs regular hours.",
        node=walmart_ny_match_leaf,
        additional_instruction="Accept phrases like 'normal hours' or 'regular schedule'."
    )
    walmart_ny_source_leaf = evaluator.add_leaf(
        id="Walmart_New_Years_source_support",
        desc="Sources support Walmart open with regular hours on New Year's Day",
        parent=walmart_ny_node,
        critical=True
    )
    await evaluator.verify(
        claim="On January 1, 2026 (New Year's Day), Walmart is open with regular hours.",
        node=walmart_ny_source_leaf,
        sources=(data.walmart_new_year_sources if data else []),
        additional_instruction="Use official Walmart references or credible news."
    )

    # Target New Year's
    target_ny_node = evaluator.add_parallel(
        id="Target_New_Years",
        desc="Target is open with regular hours on New Year's Day",
        parent=ny_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(data.target_open_regular_hours if data else None,
                                      data.target_new_year_sources if data else []),
        id="Target_New_Years_exists",
        desc="Target New Year's Day regular hours info provided with sources",
        parent=target_ny_node,
        critical=True
    )
    target_ny_match_leaf = evaluator.add_leaf(
        id="Target_New_Years_value_match",
        desc="Answer states Target is open with regular hours on New Year's Day",
        parent=target_ny_node,
        critical=True
    )
    await evaluator.verify(
        claim="On January 1, 2026 (New Year's Day), the answer states Target is open and runs regular hours.",
        node=target_ny_match_leaf,
        additional_instruction="Accept phrases like 'normal hours' or 'regular schedule'."
    )
    target_ny_source_leaf = evaluator.add_leaf(
        id="Target_New_Years_source_support",
        desc="Sources support Target open with regular hours on New Year's Day",
        parent=target_ny_node,
        critical=True
    )
    await evaluator.verify(
        claim="On January 1, 2026 (New Year's Day), Target is open with regular hours.",
        node=target_ny_source_leaf,
        sources=(data.target_new_year_sources if data else []),
        additional_instruction="Use official Target references or credible news."
    )

    # Wegmans Opening Time
    wegmans_node = evaluator.add_parallel(
        id="Wegmans_Opening",
        desc="Wegmans opens at 6 AM on New Year's Day",
        parent=ny_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_value_and_sources(data.wegmans_open_time if data else None,
                                      data.wegmans_new_year_sources if data else []),
        id="Wegmans_Opening_exists",
        desc="Wegmans New Year's Day opening time provided with sources",
        parent=wegmans_node,
        critical=True
    )
    wegmans_match_leaf = evaluator.add_leaf(
        id="Wegmans_Opening_value_match",
        desc="Answer states Wegmans opens at 6 AM on New Year's Day",
        parent=wegmans_node,
        critical=True
    )
    await evaluator.verify(
        claim="On January 1, 2026 (New Year's Day), the answer states Wegmans opens at 6 AM.",
        node=wegmans_match_leaf,
        additional_instruction="Allow minor variants like '6 a.m.'."
    )
    wegmans_source_leaf = evaluator.add_leaf(
        id="Wegmans_Opening_source_support",
        desc="Sources support Wegmans opening at 6 AM on New Year's Day",
        parent=wegmans_node,
        critical=True
    )
    await evaluator.verify(
        claim="On January 1, 2026 (New Year's Day), Wegmans opens at 6 AM.",
        node=wegmans_source_leaf,
        sources=(data.wegmans_new_year_sources if data else []),
        additional_instruction="Use official Wegmans references or credible news."
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
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator (root created as non-critical by framework)
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
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_holiday_plan(),
        template_class=HolidayPlanExtraction,
        extraction_name="holiday_plan_extraction"
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_claims": EXPECTED,
        "notes": "Expected values used for value-match verification; source support is required from citations provided in the answer."
    })

    # Build the top-level critical node to match rubric
    holiday_root = evaluator.add_parallel(
        id="Holiday_Shopping_Plan",
        desc="Comprehensive validation of holiday shopping plan across three specific dates with store hours, availability, and service deadlines",
        parent=root,
        critical=True
    )

    # Subtrees for each date
    await verify_christmas_eve(evaluator, holiday_root, extracted.christmas_eve)
    await verify_christmas_day(evaluator, holiday_root, extracted.christmas_day)
    await verify_new_years_day(evaluator, holiday_root, extracted.new_years_day)

    # Return the structured evaluation summary
    return evaluator.get_summary()