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
TASK_ID = "cre_q1_2026_briefing"
TASK_DESCRIPTION = (
    "As a commercial real estate analyst preparing a Q1 2026 market intelligence briefing, "
    "compile a comprehensive report covering five key areas: "
    "(1) the national retail vacancy rate and market conditions for Q4 2024, "
    "(2) the details of the Saks Off 5th and Last Call store closure portfolio including total stores, "
    "average size, total square footage, bankruptcy filing timing, and which state has the most affected locations, "
    "(3) the federal housing initiative announced by HUD and the Interior Department including the announcement date, "
    "amount of federal land involved, and the current HUD Secretary's name and confirmation date, "
    "(4) CAVA restaurant's recent Chicago-area expansion including the Des Plaines location address and square footage "
    "plus typical store size specifications, and "
    "(5) the Opportunity Zone affordable housing project visited by Senator Tim Scott and Secretary Turner including "
    "the project's location, cost, number of affordable units, and total investment driven by the Opportunity Zones initiative. "
    "Each section must include at least one authoritative reference URL supporting the provided information."
)

# Ground truth hints (for debugging/summary context only; verification relies on cited URLs)
GROUND_TRUTH_HINTS = {
    "market": {
        "national_vacancy_rate_q4_2024": "4.1%",
        "key_conditions": "limited availability of quality space; sustained post‑pandemic demand"
    },
    "saks": {
        "total_stores_closing": "59",
        "average_store_size_sqft": "28,000 sq ft",
        "total_square_footage": "1.7 million sq ft",
        "chapter_11_filing_month": "January 2026",
        "state_with_most_locations": "California (9)"
    },
    "federal": {
        "task_force_announcement_date": "March 17, 2025",
        "federal_land_acreage": "over 500 million acres",
        "hud_secretary_name": "Scott Turner",
        "secretary_confirmation_date": "February 5, 2025"
    },
    "cava": {
        "des_plaines_location_address": "2761 Mannheim Road",
        "des_plaines_square_footage": "2,500 sq ft",
        "typical_store_size": "around 2,800 sq ft (or 2,500–3,000 sq ft range)"
    },
    "oz": {
        "project_location": "Charleston, South Carolina",
        "project_cost": "$44 million",
        "number_of_affordable_units": "70",
        "total_oz_investment": "$84.7 billion"
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MarketFundamentals(BaseModel):
    reference_urls: List[str] = Field(default_factory=list)
    national_vacancy_rate_q4_2024: Optional[str] = None
    market_conditions: Optional[str] = None


class SaksPortfolio(BaseModel):
    reference_urls: List[str] = Field(default_factory=list)
    total_stores_closing: Optional[str] = None
    average_store_size_sqft: Optional[str] = None
    total_square_footage: Optional[str] = None
    chapter_11_filing_month: Optional[str] = None
    state_with_most_locations: Optional[str] = None


class FederalInitiative(BaseModel):
    reference_urls: List[str] = Field(default_factory=list)
    task_force_announcement_date: Optional[str] = None
    federal_land_acreage: Optional[str] = None
    hud_secretary_name: Optional[str] = None
    secretary_confirmation_date: Optional[str] = None


class CAVAExpansion(BaseModel):
    reference_urls: List[str] = Field(default_factory=list)
    des_plaines_location_address: Optional[str] = None
    des_plaines_square_footage: Optional[str] = None
    typical_store_size: Optional[str] = None


class OpportunityZoneProject(BaseModel):
    reference_urls: List[str] = Field(default_factory=list)
    project_location: Optional[str] = None
    project_cost: Optional[str] = None
    number_of_affordable_units: Optional[str] = None
    total_oz_investment: Optional[str] = None


class ReportExtraction(BaseModel):
    market: Optional[MarketFundamentals] = None
    saks: Optional[SaksPortfolio] = None
    federal: Optional[FederalInitiative] = None
    cava: Optional[CAVAExpansion] = None
    oz: Optional[OpportunityZoneProject] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_report() -> str:
    return """
    Extract the key facts, by section, exactly as stated in the answer. Return null for any missing field.
    Include the authoritative reference URLs explicitly cited in the answer for each section (extract actual URLs only).

    Sections and fields to extract (strings preferred, keep units like "%" or "sq ft" when present):

    market:
      - reference_urls: list of URLs supporting the market statistics/conditions for Q4 2024
      - national_vacancy_rate_q4_2024: string (e.g., "4.1%")
      - market_conditions: brief text summary provided by the answer (e.g., mentions of limited quality space, post‑pandemic demand)

    saks:
      - reference_urls: list of URLs about Saks Off 5th / Last Call closures or related bankruptcy
      - total_stores_closing: string (e.g., "59")
      - average_store_size_sqft: string (e.g., "28,000 sq ft")
      - total_square_footage: string (e.g., "1.7 million sq ft")
      - chapter_11_filing_month: string (e.g., "January 2026")
      - state_with_most_locations: string (e.g., "California (9)")

    federal:
      - reference_urls: list of URLs for the HUD–Interior Joint Task Force on Federal Land for Housing
      - task_force_announcement_date: string date (e.g., "March 17, 2025")
      - federal_land_acreage: string (e.g., "over 500 million acres")
      - hud_secretary_name: string (e.g., "Scott Turner")
      - secretary_confirmation_date: string date (e.g., "February 5, 2025")

    cava:
      - reference_urls: list of URLs for CAVA's Chicago-area expansion
      - des_plaines_location_address: string (e.g., "2761 Mannheim Road")
      - des_plaines_square_footage: string (e.g., "2,500 sq ft")
      - typical_store_size: string (e.g., "around 2,800 sq ft" or "2,500–3,000 sq ft")

    oz:
      - reference_urls: list of URLs for the One80 Place Opportunity Zone project visit
      - project_location: string (e.g., "Charleston, South Carolina")
      - project_cost: string (e.g., "$44 million")
      - number_of_affordable_units: string (e.g., "70")
      - total_oz_investment: string (e.g., "$84.7 billion")

    Rules:
    - Extract only what appears in the answer; do not invent data.
    - For any required URL list, include only valid URLs that appear in the answer text or markdown links.
    - If a field is not present in the answer, return null (or an empty list for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_non_empty(val: Optional[str]) -> bool:
    return bool(val and str(val).strip())


async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    node,
    claim: str,
    urls: Optional[List[str]],
    additional_instruction: Optional[str] = None,
) -> None:
    if urls and len(urls) > 0:
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=additional_instruction or "None",
        )
    else:
        node.score = 0.0
        node.status = "failed"


def _fail_if_missing(node, value_present: bool) -> bool:
    if not value_present:
        node.score = 0.0
        node.status = "failed"
        return True
    return False


# --------------------------------------------------------------------------- #
# Section verifiers                                                           #
# --------------------------------------------------------------------------- #
async def verify_market_fundamentals(evaluator: Evaluator, parent_node, data: Optional[MarketFundamentals]) -> None:
    section = evaluator.add_parallel(
        id="Market_Fundamentals",
        desc="National retail market statistics and conditions for Q4 2024",
        parent=parent_node,
        critical=False,
    )

    urls = data.reference_urls if data else []

    # Market_Reference_URL (critical)
    n_ref = evaluator.add_leaf(
        id="Market_Reference_URL",
        desc="Provide authoritative URL source for national retail market statistics (e.g., Colliers, CoStar, or similar commercial real estate research firm)",
        parent=section,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator,
        n_ref,
        "This webpage is an authoritative and relevant source for Q4 2024 U.S. retail market statistics (e.g., national retail vacancy).",
        urls,
        additional_instruction="Accept credible CRE research sources: Colliers, CoStar, CBRE, JLL, Cushman & Wakefield, Marcus & Millichap, etc. The page should mention national retail vacancy for Q4 2024 or late 2024.",
    )

    # National_Vacancy_Rate (critical)
    n_vac = evaluator.add_leaf(
        id="National_Vacancy_Rate",
        desc="Report the national retail vacancy rate for Q4 2024 (should be 4.1%)",
        parent=section,
        critical=True
    )
    value_present = data is not None and _is_non_empty(data.national_vacancy_rate_q4_2024)
    if _fail_if_missing(n_vac, value_present):
        pass
    else:
        claim = f"The national retail vacancy rate for Q4 2024 is {data.national_vacancy_rate_q4_2024}."
        await _verify_with_urls_or_fail(
            evaluator,
            n_vac,
            claim,
            urls,
            additional_instruction="Target is 4.1%. Allow minor rounding if the page clearly supports ~4.1% for Q4 2024 national retail.",
        )

    # Market_Conditions (non-critical)
    n_cond = evaluator.add_leaf(
        id="Market_Conditions",
        desc="Describe the key market conditions (limited quality space, post-pandemic demand characteristics)",
        parent=section,
        critical=False
    )
    cond_present = data is not None and _is_non_empty(data.market_conditions)
    if _fail_if_missing(n_cond, cond_present):
        pass
    else:
        claim = f"The cited source supports these Q4 2024 U.S. retail market conditions: {data.market_conditions}"
        await _verify_with_urls_or_fail(
            evaluator,
            n_cond,
            claim,
            urls,
            additional_instruction="Specifically check for notions like limited availability of high-quality space and continued post‑pandemic demand or consumer resilience.",
        )


async def verify_saks_portfolio(evaluator: Evaluator, parent_node, data: Optional[SaksPortfolio]) -> None:
    section = evaluator.add_parallel(
        id="Saks_Off_5th_Portfolio",
        desc="Details about Saks Off 5th and Last Call store closures and portfolio restructuring",
        parent=parent_node,
        critical=False,
    )

    urls = data.reference_urls if data else []

    # Saks_Reference_URL (critical)
    s_ref = evaluator.add_leaf(
        id="Saks_Reference_URL",
        desc="Provide authoritative URL source for Saks Off 5th closure information (e.g., CoStar, commercial real estate news, or company announcement)",
        parent=section,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator,
        s_ref,
        "This webpage is an authoritative and relevant source on Saks Off 5th/Last Call 2026 closures and/or the bankruptcy filing.",
        urls,
        additional_instruction="Accept credible business/CRE sources or official company statements that clearly detail closures or the Chapter 11 filing.",
    )

    # Total_Stores_Closing (critical)
    s_total = evaluator.add_leaf(
        id="Total_Stores_Closing",
        desc="Report the total number of Saks Off 5th and Last Call stores being closed (should be 59 stores)",
        parent=section,
        critical=True
    )
    present = data is not None and _is_non_empty(data.total_stores_closing)
    if _fail_if_missing(s_total, present):
        pass
    else:
        claim = f"A combined total of {data.total_stores_closing} Saks Off 5th and Last Call stores are being closed."
        await _verify_with_urls_or_fail(
            evaluator,
            s_total,
            claim,
            urls,
            additional_instruction="Target is 59 stores. Minor wording tolerance allowed (e.g., '59 closures').",
        )

    # Average_Store_Size (critical)
    s_avg = evaluator.add_leaf(
        id="Average_Store_Size",
        desc="Report the average size of the stores being closed in square feet (should be 28,000 sq ft)",
        parent=section,
        critical=True
    )
    present = data is not None and _is_non_empty(data.average_store_size_sqft)
    if _fail_if_missing(s_avg, present):
        pass
    else:
        claim = f"The average size of the closing stores is about {data.average_store_size_sqft}."
        await _verify_with_urls_or_fail(
            evaluator,
            s_avg,
            claim,
            urls,
            additional_instruction="Target is ~28,000 sq ft; accept variants like '28k square feet'.",
        )

    # Total_Square_Footage (critical)
    s_tot_sf = evaluator.add_leaf(
        id="Total_Square_Footage",
        desc="Report the total square footage being made available (should be 1.7 million sq ft)",
        parent=section,
        critical=True
    )
    present = data is not None and _is_non_empty(data.total_square_footage)
    if _fail_if_missing(s_tot_sf, present):
        pass
    else:
        claim = f"The total square footage being made available from these closures is about {data.total_square_footage}."
        await _verify_with_urls_or_fail(
            evaluator,
            s_tot_sf,
            claim,
            urls,
            additional_instruction="Target is ~1.7 million sq ft; accept '1.7M square feet' style equivalents.",
        )

    # Chapter_11_Filing_Month (critical)
    s_ch11 = evaluator.add_leaf(
        id="Chapter_11_Filing_Month",
        desc="Report the month and year Saks Global filed for Chapter 11 (should be January 2026)",
        parent=section,
        critical=True
    )
    present = data is not None and _is_non_empty(data.chapter_11_filing_month)
    if _fail_if_missing(s_ch11, present):
        pass
    else:
        claim = f"Saks Global filed for Chapter 11 in {data.chapter_11_filing_month}."
        await _verify_with_urls_or_fail(
            evaluator,
            s_ch11,
            claim,
            urls,
            additional_instruction="Target is January 2026.",
        )

    # State_With_Most_Locations (non-critical)
    s_state = evaluator.add_leaf(
        id="State_With_Most_Locations",
        desc="Identify which state has the most Saks Off 5th locations being closed (should be California with 9 stores)",
        parent=section,
        critical=False
    )
    present = data is not None and _is_non_empty(data.state_with_most_locations)
    if _fail_if_missing(s_state, present):
        pass
    else        :
        claim = f"The state with the most affected Saks Off 5th locations is {data.state_with_most_locations}."
        await _verify_with_urls_or_fail(
            evaluator,
            s_state,
            claim,
            urls,
            additional_instruction="Expected: California has the most closures (9). Minor formatting differences allowed (e.g., 'California (9)').",
        )


async def verify_federal_initiative(evaluator: Evaluator, parent_node, data: Optional[FederalInitiative]) -> None:
    section = evaluator.add_parallel(
        id="Federal_Housing_Initiative",
        desc="HUD and Interior Department joint task force on federal land for housing",
        parent=parent_node,
        critical=False,
    )

    urls = data.reference_urls if data else []

    # Federal_Initiative_Reference_URL (critical)
    f_ref = evaluator.add_leaf(
        id="Federal_Initiative_Reference_URL",
        desc="Provide authoritative URL source for the HUD-Interior task force announcement (e.g., HUD.gov or DOI.gov official press release)",
        parent=section,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator,
        f_ref,
        "This webpage is an official or authoritative announcement about the HUD–Interior Joint Task Force on Federal Land for Housing.",
        urls,
        additional_instruction="Prefer HUD.gov or DOI.gov press releases; otherwise an authoritative government or reputable outlet summarizing the announcement.",
    )

    # Task_Force_Announcement_Date (critical)
    f_date = evaluator.add_leaf(
        id="Task_Force_Announcement_Date",
        desc="Report the date the Joint Task Force on Federal Land for Housing was announced (should be March 17, 2025)",
        parent=section,
        critical=True
    )
    present = data is not None and _is_non_empty(data.task_force_announcement_date)
    if _fail_if_missing(f_date, present):
        pass
    else:
        claim = f"The Joint Task Force on Federal Land for Housing was announced on {data.task_force_announcement_date}."
        await _verify_with_urls_or_fail(
            evaluator,
            f_date,
            claim,
            urls,
            additional_instruction="Target date is March 17, 2025.",
        )

    # Federal_Land_Acreage (critical)
    f_acre = evaluator.add_leaf(
        id="Federal_Land_Acreage",
        desc="Report the amount of federal land overseen by the Interior Department (should be over 500 million acres)",
        parent=section,
        critical=True
    )
    present = data is not None and _is_non_empty(data.federal_land_acreage)
    if _fail_if_missing(f_acre, present):
        pass
    else:
        claim = f"The Interior Department oversees {data.federal_land_acreage} of federal land."
        await _verify_with_urls_or_fail(
            evaluator,
            f_acre,
            claim,
            urls,
            additional_instruction="Target phrasing: 'over 500 million acres' (allow close variants).",
        )

    # HUD_Secretary_Name (critical)
    f_sec = evaluator.add_leaf(
        id="HUD_Secretary_Name",
        desc="Identify the current HUD Secretary (should be Scott Turner)",
        parent=section,
        critical=True
    )
    present = data is not None and _is_non_empty(data.hud_secretary_name)
    if _fail_if_missing(f_sec, present):
        pass
    else:
        claim = f"The current HUD Secretary is {data.hud_secretary_name}."
        await _verify_with_urls_or_fail(
            evaluator,
            f_sec,
            claim,
            urls,
            additional_instruction="Expected name: Scott Turner.",
        )

    # Secretary_Confirmation_Date (non-critical)
    f_conf = evaluator.add_leaf(
        id="Secretary_Confirmation_Date",
        desc="Report the date Scott Turner was confirmed as HUD Secretary (should be February 5, 2025)",
        parent=section,
        critical=False
    )
    present = data is not None and _is_non_empty(data.secretary_confirmation_date)
    if _fail_if_missing(f_conf, present):
        pass
    else:
        claim = f"Scott Turner was confirmed as HUD Secretary on {data.secretary_confirmation_date}."
        await _verify_with_urls_or_fail(
            evaluator,
            f_conf,
            claim,
            urls,
            additional_instruction="Target date is February 5, 2025.",
        )


async def verify_cava_expansion(evaluator: Evaluator, parent_node, data: Optional[CAVAExpansion]) -> None:
    section = evaluator.add_parallel(
        id="CAVA_Expansion",
        desc="CAVA restaurant expansion details in the Chicago area",
        parent=parent_node,
        critical=False,
    )

    urls = data.reference_urls if data else []

    # CAVA_Reference_URL (critical)
    c_ref = evaluator.add_leaf(
        id="CAVA_Reference_URL",
        desc="Provide authoritative URL source for CAVA expansion information (e.g., commercial real estate news or company announcement)",
        parent=section,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator,
        c_ref,
        "This webpage is an authoritative and relevant source on CAVA's Chicago-area expansion.",
        urls,
        additional_instruction="Accept credible CRE news outlets, local business press, or official company announcements.",
    )

    # Des_Plaines_Location_Address (critical)
    c_addr = evaluator.add_leaf(
        id="Des_Plaines_Location_Address",
        desc="Report the address of the CAVA Des Plaines location (should be 2761 Mannheim Road)",
        parent=section,
        critical=True
    )
    present = data is not None and _is_non_empty(data.des_plaines_location_address)
    if _fail_if_missing(c_addr, present):
        pass
    else:
        claim = f"The address of the CAVA Des Plaines location is {data.des_plaines_location_address}."
        await _verify_with_urls_or_fail(
            evaluator,
            c_addr,
            claim,
            urls,
            additional_instruction="Expected: 2761 Mannheim Road. Allow minor formatting variations (e.g., 'Rd.').",
        )

    # Des_Plaines_Square_Footage (critical)
    c_sf = evaluator.add_leaf(
        id="Des_Plaines_Square_Footage",
        desc="Report the square footage of the Des Plaines location (should be 2,500 sq ft)",
        parent=section,
        critical=True
    )
    present = data is not None and _is_non_empty(data.des_plaines_square_footage)
    if _fail_if_missing(c_sf, present):
        pass
    else:
        claim = f"The Des Plaines CAVA location is approximately {data.des_plaines_square_footage}."
        await _verify_with_urls_or_fail(
            evaluator,
            c_sf,
            claim,
            urls,
            additional_instruction="Expected about 2,500 sq ft; allow small rounding/approximation.",
        )

    # Typical_Store_Size (non-critical)
    c_typ = evaluator.add_leaf(
        id="Typical_Store_Size",
        desc="Report the typical size for CAVA's standard new locations (should be 2,800 sq ft or in the range of 2,500-3,000 sq ft)",
        parent=section,
        critical=False
    )
    present = data is not None and _is_non_empty(data.typical_store_size)
    if _fail_if_missing(c_typ, present):
        pass
    else:
        claim = f"CAVA's typical standard new restaurant footprint is {data.typical_store_size}."
        await _verify_with_urls_or_fail(
            evaluator,
            c_typ,
            claim,
            urls,
            additional_instruction="Expected around 2,800 sq ft or a range roughly 2,500–3,000 sq ft.",
        )


async def verify_oz_project(evaluator: Evaluator, parent_node, data: Optional[OpportunityZoneProject]) -> None:
    section = evaluator.add_parallel(
        id="Opportunity_Zone_Project",
        desc="Senator Scott and Secretary Turner's visit to One80 Place affordable housing project",
        parent=parent_node,
        critical=False,
    )

    urls = data.reference_urls if data else []

    # OZ_Project_Reference_URL (critical)
    o_ref = evaluator.add_leaf(
        id="OZ_Project_Reference_URL",
        desc="Provide authoritative URL source for the One80 Place project visit (e.g., Senate Banking Committee or HUD press release)",
        parent=section,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator,
        o_ref,
        "This webpage is an authoritative and relevant source describing Senator Tim Scott and Secretary Turner's visit to the One80 Place Opportunity Zone project.",
        urls,
        additional_instruction="Prefer official Senate/HUD communications or reputable outlets covering the visit and project details.",
    )

    # Project_Location (critical)
    o_loc = evaluator.add_leaf(
        id="Project_Location",
        desc="Report the location of the One80 Place project (should be Charleston, South Carolina)",
        parent=section,
        critical=True
    )
    present = data is not None and _is_non_empty(data.project_location)
    if _fail_if_missing(o_loc, present):
        pass
    else:
        claim = f"The One80 Place Opportunity Zone project is located in {data.project_location}."
        await _verify_with_urls_or_fail(
            evaluator,
            o_loc,
            claim,
            urls,
            additional_instruction="Expected: Charleston, South Carolina.",
        )

    # Project_Cost (critical)
    o_cost = evaluator.add_leaf(
        id="Project_Cost",
        desc="Report the cost of the One80 Place housing project (should be $44 million)",
        parent=section,
        critical=True
    )
    present = data is not None and _is_non_empty(data.project_cost)
    if _fail_if_missing(o_cost, present):
        pass
    else:
        claim = f"The One80 Place housing project cost is {data.project_cost}."
        await _verify_with_urls_or_fail(
            evaluator,
            o_cost,
            claim,
            urls,
            additional_instruction="Expected approximately $44 million; accept '$44M' style.",
        )

    # Number_of_Affordable_Units (critical)
    o_units = evaluator.add_leaf(
        id="Number_of_Affordable_Units",
        desc="Report the number of affordable housing units in the project (should be 70 units)",
        parent=section,
        critical=True
    )
    present = data is not None and _is_non_empty(data.number_of_affordable_units)
    if _fail_if_missing(o_units, present):
        pass
    else:
        claim = f"The project includes {data.number_of_affordable_units} affordable housing units."
        await _verify_with_urls_or_fail(
            evaluator,
            o_units,
            claim,
            urls,
            additional_instruction="Expected: 70 units.",
        )

    # Total_OZ_Investment (non-critical)
    o_total = evaluator.add_leaf(
        id="Total_OZ_Investment",
        desc="Report the total investment driven by Senator Scott's Opportunity Zones initiative (should be $84.7 billion)",
        parent=section,
        critical=False
    )
    present = data is not None and _is_non_empty(data.total_oz_investment)
    if _fail_if_missing(o_total, present):
        pass
    else:
        claim = f"Senator Scott's Opportunity Zones initiative has driven a total investment of {data.total_oz_investment}."
        await _verify_with_urls_or_fail(
            evaluator,
            o_total,
            claim,
            urls,
            additional_instruction="Expected: $84.7 billion; accept '$84.7B' style.",
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

    # Extract structured information once
    extracted = await evaluator.extract(
        prompt=prompt_extract_report(),
        template_class=ReportExtraction,
        extraction_name="report_extraction",
    )

    # Record ground truth hints for transparency (not used for scoring directly)
    evaluator.add_ground_truth({"expected": GROUND_TRUTH_HINTS}, gt_type="ground_truth_hints")

    # Build verification tree per rubric (all sections are parallel under root)
    await verify_market_fundamentals(evaluator, root, extracted.market)
    await verify_saks_portfolio(evaluator, root, extracted.saks)
    await verify_federal_initiative(evaluator, root, extracted.federal)
    await verify_cava_expansion(evaluator, root, extracted.cava)
    await verify_oz_project(evaluator, root, extracted.oz)

    return evaluator.get_summary()