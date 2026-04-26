import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "travel_planning_carnival_cape_verde_2026"
TASK_DESCRIPTION = """
A US citizen living in Nashville, Tennessee is planning a vacation that begins with a 7-day Carnival cruise departing from New Orleans. After the cruise, they will fly directly from New Orleans to Cape Verde for an additional week-long stay. They need to plan the logistics for this trip.

For the cruise portion:
- They will drive their personal vehicle from Nashville to the Port of New Orleans
- They need to park their vehicle at the cruise terminal for the duration of the 7-day cruise
- They plan to make a parking reservation in advance

For the Cape Verde portion:
- They are aware that Cape Verde recently changed entry requirements in January 2026
- They need to understand the mandatory pre-travel registration and fee requirements

Provide the following information:

1. What is the approximate driving distance in miles from Nashville, Tennessee to New Orleans, Louisiana?

2. What is the total parking cost for a 7-day stay at Port NOLA (Port of New Orleans) for Carnival cruise passengers when making an advance reservation? (Include both the base daily rate and any reservation convenience fees in your calculation)

3. What is the name of the electronic pre-registration system that US citizens must complete before traveling to Cape Verde as of 2026, even though they can enter visa-free for stays up to 30 days?

4. What is the Cape Verde airport security tax amount (in either CVE or USD) that travelers must pay?

For each answer, provide at least one supporting URL reference.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TravelPlanExtraction(BaseModel):
    # Ground transportation
    driving_distance_miles: Optional[str] = None
    driving_sources: List[str] = Field(default_factory=list)

    # Parking (Port NOLA) – advance reservation
    parking_daily_rate: Optional[str] = None
    parking_reservation_fee: Optional[str] = None
    parking_total_7days: Optional[str] = None
    parking_sources: List[str] = Field(default_factory=list)

    # Cape Verde entry (EASE)
    cape_verde_prereg_name: Optional[str] = None
    cape_verde_ease_sources: List[str] = Field(default_factory=list)

    # Cape Verde Airport Security Tax
    cape_verde_security_tax_amount: Optional[str] = None  # e.g., "3400 CVE" or "$33"
    cape_verde_security_tax_currency: Optional[str] = None  # e.g., "CVE" or "USD"
    cape_verde_security_tax_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_plan() -> str:
    return """
Extract the following items exactly as presented in the answer. Do not invent values. If the answer omits any item, return null for that field. Also extract all explicit URLs the answer used as sources for each item.

1) Ground transportation (Nashville → New Orleans):
- driving_distance_miles: The approximate driving distance in miles as stated in the answer (keep units/wording if included, e.g., "530 miles", "about 530 miles").
- driving_sources: List all URLs in the answer that support the driving distance.

2) Port NOLA cruise parking (advance reservation for a 7-day cruise):
- parking_daily_rate: The base daily parking rate stated in the answer for cruise parking at Port NOLA (Erato or Julia Street Cruise Terminal) when making an advance reservation (keep currency symbol and wording).
- parking_reservation_fee: Any reservation convenience/processing fee stated (e.g., "$5", "$0", "none"). If the answer asserts there is no such fee, return the exact wording used (e.g., "no fee", "0", etc.).
- parking_total_7days: The total parking cost for 7 days as calculated/presented in the answer (including any reservation fee).
- parking_sources: List all URLs that support the parking information and/or rates (prefer official portnola.com links if present).

3) Cape Verde entry requirements (2026):
- cape_verde_prereg_name: The name of the electronic pre-registration system that US citizens must complete before travel (e.g., "EASE" or "E.A.S.E.").
- cape_verde_ease_sources: List all URLs that support the EASE/pre-registration requirement.

4) Cape Verde Airport Security Tax:
- cape_verde_security_tax_amount: The tax amount travelers must pay (e.g., "3400 CVE", "$33"). Keep whatever the answer uses.
- cape_verde_security_tax_currency: The currency used in the answer for the amount (e.g., "CVE" or "USD"). If mixed or unclear, return what best matches the answer context.
- cape_verde_security_tax_sources: List all URLs that support the tax amount.

Rules for sources:
- Extract only real URLs explicitly present in the answer (plain or in markdown).
- Do not infer or create URLs.
- Include at least one URL for each category if the answer provided one; otherwise, leave the list empty.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_any_url(urls: List[str]) -> bool:
    return bool(urls and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0)


def _filter_domain(urls: List[str], domain_keyword: str) -> List[str]:
    return [u for u in urls if isinstance(u, str) and domain_keyword.lower() in u.lower()]


def _uniq(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_ground_transportation_checks(
    evaluator: Evaluator,
    parent_node,
    data: TravelPlanExtraction,
) -> None:
    # Ground Transportation (critical under Travel_Planning due to parent critical rule)
    gt_node = evaluator.add_parallel(
        id="Ground_Transportation",
        desc="Verify ground transportation planning from Nashville to New Orleans cruise port",
        parent=parent_node,
        critical=True,
    )

    # Driving Distance group
    dd_group = evaluator.add_parallel(
        id="Driving_Distance",
        desc="Provide the approximate driving distance in miles from Nashville to New Orleans",
        parent=gt_node,
        critical=True,
    )

    dd_provided = evaluator.add_custom_node(
        result=(data.driving_distance_miles is not None and str(data.driving_distance_miles).strip() != "" and _has_any_url(data.driving_sources)),
        id="Driving_Distance_Provided",
        desc="Driving distance and supporting source(s) provided",
        parent=dd_group,
        critical=True,
    )

    dd_leaf = evaluator.add_leaf(
        id="Driving_Distance_Supported",
        desc="Approximate driving distance claim is supported by sources",
        parent=dd_group,
        critical=True,
    )

    dd_value = data.driving_distance_miles or ""
    await evaluator.verify(
        claim=f"The driving distance from Nashville, Tennessee to New Orleans, Louisiana is approximately {dd_value}.",
        node=dd_leaf,
        sources=data.driving_sources,
        additional_instruction="Use the provided webpage(s) to confirm a typical road driving distance. Allow reasonable approximations and rounding (±10% tolerance). If a page shows kilometers, consider that 1 mile ≈ 1.609 km.",
    )

    # Parking Information group
    park_group = evaluator.add_parallel(
        id="Parking_Information",
        desc="Specify the parking cost for Carnival cruise passengers at Port NOLA for a 7-day cruise with advance reservation",
        parent=gt_node,
        critical=True,
    )

    park_info_provided = evaluator.add_custom_node(
        result=(
            (data.parking_daily_rate is not None and str(data.parking_daily_rate).strip() != "")
            and (data.parking_total_7days is not None and str(data.parking_total_7days).strip() != "")
            and _has_any_url(data.parking_sources)
        ),
        id="Parking_Info_Provided",
        desc="Parking daily rate, 7-day total, and supporting source(s) provided",
        parent=park_group,
        critical=True,
    )

    # Daily rate supported
    rate_leaf = evaluator.add_leaf(
        id="Parking_Daily_Rate_Supported",
        desc="Base daily parking rate (advance reservation) is supported by sources",
        parent=park_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Port NOLA cruise parking base daily rate for an advance reservation is {data.parking_daily_rate or ''}.",
        node=rate_leaf,
        sources=data.parking_sources,
        additional_instruction="Verify the standard passenger vehicle daily rate for cruise parking at the Erato or Julia Street Cruise Terminal (advance reservation / pre-paid if specified). Allow minor wording variations (e.g., 'per day', 'per 24 hours').",
    )

    # Reservation convenience fee provided (required by task)
    res_fee_provided = evaluator.add_custom_node(
        result=(data.parking_reservation_fee is not None and str(data.parking_reservation_fee).strip() != ""),
        id="Reservation_Fee_Provided",
        desc="Reservation convenience/processing fee value is provided (can be '0' or 'none' if asserted)",
        parent=park_group,
        critical=True,
    )

    # Reservation convenience fee supported
    res_fee_leaf = evaluator.add_leaf(
        id="Reservation_Fee_Supported",
        desc="Reservation convenience/processing fee is supported by sources",
        parent=park_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"There is a reservation convenience or processing fee of {data.parking_reservation_fee or ''} when making an advance parking reservation for Port NOLA cruise parking.",
        node=res_fee_leaf,
        sources=data.parking_sources,
        additional_instruction="Confirm whether a convenience/processing fee applies for advance/prepaid cruise parking bookings (if the claim asserts 'no fee' or '0', verify that the source supports that).",
    )

    # Total cost calculation check (simple arithmetic verification)
    total_calc_leaf = evaluator.add_leaf(
        id="Parking_Total_7days_Calculation_Correct",
        desc="7-day total parking cost calculation is arithmetically correct given the stated daily rate and reservation fee",
        parent=park_group,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"Given a 7-day cruise, a base daily parking rate of '{data.parking_daily_rate or ''}' per day "
            f"and a reservation convenience fee of '{data.parking_reservation_fee or ''}', "
            f"the total parking cost for 7 days is '{data.parking_total_7days or ''}'. "
            f"This arithmetic should be correct: total = 7 * daily_rate + reservation_fee (if applicable)."
        ),
        node=total_calc_leaf,
        sources=None,  # Pure arithmetic/logical check
        additional_instruction="Evaluate the arithmetic only, based on the textual amounts as given. Ignore taxes/surcharges not mentioned. Treat currency symbols consistently.",
    )

    # Reference URL (Port NOLA) group
    ref_group = evaluator.add_parallel(
        id="Reference_URL_Parking",
        desc="Provide a valid URL from portnola.com that documents the parking rates for cruise passengers",
        parent=gt_node,
        critical=True,
    )

    portnola_urls = _filter_domain(data.parking_sources, "portnola.com")
    has_portnola_url = evaluator.add_custom_node(
        result=_has_any_url(portnola_urls),
        id="PortNOLA_URL_Provided",
        desc="At least one portnola.com URL was provided for parking rates",
        parent=ref_group,
        critical=True,
    )

    portnola_rates_leaf = evaluator.add_leaf(
        id="PortNOLA_Rates_Documented",
        desc="The portnola.com page documents cruise parking rates",
        parent=ref_group,
        critical=True,
    )

    await evaluator.verify(
        claim="This page provides the official Port NOLA cruise parking rates (for Erato and/or Julia Street Cruise Terminals) and/or advance reservation information.",
        node=portnola_rates_leaf,
        sources=portnola_urls if portnola_urls else data.parking_sources,
        additional_instruction="Confirm that the page is on portnola.com and that it explicitly shows cruise parking rates and related booking details.",
    )


async def build_international_requirements_checks(
    evaluator: Evaluator,
    parent_node,
    data: TravelPlanExtraction,
) -> None:
    intl_node = evaluator.add_parallel(
        id="International_Travel_Requirements",
        desc="Verify Cape Verde entry requirements for US citizens in 2026",
        parent=parent_node,
        critical=True,
    )

    # EASE (pre-registration) group
    ease_group = evaluator.add_parallel(
        id="EASE_Registration",
        desc="Confirm that Cape Verde requires US citizens to complete EASE pre-registration despite visa-free entry",
        parent=intl_node,
        critical=True,
    )

    ease_provided = evaluator.add_custom_node(
        result=(data.cape_verde_prereg_name is not None and str(data.cape_verde_prereg_name).strip() != "" and _has_any_url(data.cape_verde_ease_sources)),
        id="EASE_Info_Provided",
        desc="EASE pre-registration name and supporting source(s) provided",
        parent=ease_group,
        critical=True,
    )

    ease_name_leaf = evaluator.add_leaf(
        id="EASE_Name_Correct",
        desc="The extracted pre-registration system name corresponds to 'EASE'/'E.A.S.E.'",
        parent=ease_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The electronic pre-registration system for Cabo Verde travel is called 'EASE' (also written 'E.A.S.E.'). The extracted name '{data.cape_verde_prereg_name or ''}' refers to the same system.",
        node=ease_name_leaf,
        sources=None,
        additional_instruction="Allow variations like 'EASE', 'E.A.S.E.', or an expanded phrase that clearly maps to EASE. This is a simple name-equivalence check.",
    )

    ease_required_leaf = evaluator.add_leaf(
        id="EASE_Requirement_Supported",
        desc="EASE pre-registration requirement is supported by sources",
        parent=ease_group,
        critical=True,
    )
    await evaluator.verify(
        claim="US citizens must complete the EASE (online) pre-registration before traveling to Cabo Verde, even though stays up to 30 days are visa-free.",
        node=ease_required_leaf,
        sources=data.cape_verde_ease_sources,
        additional_instruction="Confirm the requirement and that it applies to US citizens for short (≤30 days) visa-free stays as of 2026.",
    )

    # Airport Security Tax group
    tax_group = evaluator.add_parallel(
        id="Airport_Security_Fee",
        desc="Provide the Cape Verde airport security tax amount in CVE or USD that must be paid",
        parent=intl_node,
        critical=True,
    )

    tax_provided = evaluator.add_custom_node(
        result=(data.cape_verde_security_tax_amount is not None and str(data.cape_verde_security_tax_amount).strip() != "" and _has_any_url(data.cape_verde_security_tax_sources)),
        id="Airport_Tax_Info_Provided",
        desc="Airport security tax amount and supporting source(s) provided",
        parent=tax_group,
        critical=True,
    )

    tax_leaf = evaluator.add_leaf(
        id="Airport_Tax_Amount_Supported",
        desc="Airport security tax amount is supported by sources",
        parent=tax_group,
        critical=True,
    )

    tax_amount_text = data.cape_verde_security_tax_amount or ""
    tax_currency_text = f" {data.cape_verde_security_tax_currency}" if (data.cape_verde_security_tax_currency and data.cape_verde_security_tax_currency not in tax_amount_text) else ""
    await evaluator.verify(
        claim=f"The Cabo Verde airport security tax amount that travelers must pay is {tax_amount_text}{tax_currency_text}.",
        node=tax_leaf,
        sources=data.cape_verde_security_tax_sources,
        additional_instruction="Verify the specific amount for international passengers (the main TSA/EASE tax). If the source shows the same amount or a clearly equivalent value (e.g., currency formatting), accept it.",
    )

    # Official reference URL group
    ref_group = evaluator.add_parallel(
        id="Reference_URL_Cape_Verde",
        desc="Provide a valid official URL documenting Cape Verde's EASE registration requirement or airport security fee",
        parent=intl_node,
        critical=True,
    )

    combined_sources = _uniq((data.cape_verde_ease_sources or []) + (data.cape_verde_security_tax_sources or []))
    official_ref_provided = evaluator.add_custom_node(
        result=_has_any_url(combined_sources),
        id="Cape_Verde_Official_URL_Provided",
        desc="At least one reference URL about EASE or the airport security tax was provided",
        parent=ref_group,
        critical=True,
    )

    official_ref_leaf = evaluator.add_leaf(
        id="Cape_Verde_Official_Reference_Supported",
        desc="An official government or embassy page documents EASE or the airport security tax",
        parent=ref_group,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This webpage is an official government or embassy resource (e.g., ends with .gov.cv, ease.gov.cv, state.gov, or usembassy.gov) "
            "and it explicitly documents either the Cabo Verde EASE pre-registration requirement or the Airport Security Tax (TSA) amount."
        ),
        node=official_ref_leaf,
        sources=combined_sources,
        additional_instruction="Prefer primary sources such as ease.gov.cv or other .gov.cv domains. US embassy/state.gov domains are also considered official for guidance.",
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
    # Initialize evaluator (framework root is non-critical by design)
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_travel_plan(),
        template_class=TravelPlanExtraction,
        extraction_name="travel_plan_extraction",
    )

    # Add a critical top-level node reflecting the rubric root
    travel_planning_node = evaluator.add_parallel(
        id="Travel_Planning",
        desc="Evaluate the complete travel planning solution including pre-cruise transportation, parking logistics, and international travel requirements",
        parent=root,
        critical=True,
    )

    # Build subtrees
    await build_ground_transportation_checks(evaluator, travel_planning_node, extracted)
    await build_international_requirements_checks(evaluator, travel_planning_node, extracted)

    # Optional: record task goals into summary as ground truth context
    evaluator.add_ground_truth({
        "requirements": [
            "Approximate driving distance (Nashville → New Orleans) with URL",
            "Port NOLA cruise parking 7-day total including base daily rate and any reservation convenience fee, with URL(s) (prefer portnola.com)",
            "Name of Cabo Verde electronic pre-registration system (EASE), with URL",
            "Cabo Verde airport security tax amount (CVE or USD), with URL",
        ],
        "notes": "All factual claims should be supported by the cited URLs. Arithmetic checks are verified logically."
    })

    # Return evaluation summary
    return evaluator.get_summary()