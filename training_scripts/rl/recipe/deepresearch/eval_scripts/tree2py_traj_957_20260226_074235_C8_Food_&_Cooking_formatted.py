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
TASK_ID = "holiday_policies_restaurants"
TASK_DESCRIPTION = (
    "Research and verify the holiday operating policies for the following four national restaurant chains, "
    "providing specific details and supporting references for each:\n\n"
    "1. Chick-fil-A:\n"
    "- Identify which days of the week Chick-fil-A restaurants are always closed\n"
    "- Verify whether Chick-fil-A is open or closed on Thanksgiving Day\n"
    "- Verify whether Chick-fil-A is open or closed on Christmas Day\n"
    "- Provide an official Chick-fil-A source (company website or official customer support page) that confirms these closure policies\n\n"
    "2. Cracker Barrel:\n"
    "- Verify whether Cracker Barrel restaurants are open on Thanksgiving Day\n"
    "- If open, provide the specific operating hours (opening and closing times) for Thanksgiving Day\n"
    "- Clarify whether all Cracker Barrel locations are open on Thanksgiving, or only some locations\n"
    "- Provide a credible source (news article, company announcement, or official website) that confirms this Thanksgiving operating policy\n\n"
    "3. Golden Corral:\n"
    "- Verify whether Golden Corral restaurants are open on Thanksgiving Day\n"
    "- Provide the typical opening time for Golden Corral on Thanksgiving Day\n"
    "- Provide the typical closing time range for Golden Corral on Thanksgiving Day\n"
    "- Provide a credible source (restaurant guide website, news article, or official announcement) that confirms these Thanksgiving hours\n\n"
    "4. McDonald's:\n"
    "- Research and state the approximate percentage of McDonald's restaurants that are independently owned and operated as franchises (rather than company-owned)\n"
    "- Explain how this franchise ownership structure affects holiday operating hours across different McDonald's locations\n"
    "- Identify what method McDonald's recommends for customers to verify their local restaurant's holiday hours\n"
    "- Provide an official McDonald's corporate source that confirms the franchise ownership percentage\n\n"
    "For each restaurant chain, include the reference URL that supports your findings."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Chain1ChickFilA(BaseModel):
    sunday_policy: Optional[str] = None  # e.g., "closed", "closed every Sunday", "open"
    thanksgiving_policy: Optional[str] = None  # e.g., "closed", "open"
    christmas_policy: Optional[str] = None  # e.g., "closed", "open"
    source_urls: List[str] = Field(default_factory=list)


class Chain2CrackerBarrel(BaseModel):
    thanksgiving_open: Optional[str] = None  # "open" / "closed" / phrase
    thanksgiving_opening_time: Optional[str] = None  # "6 a.m."
    thanksgiving_closing_time: Optional[str] = None  # "10 p.m."
    all_locations_open: Optional[str] = None  # "all locations", "some locations", "varies", "unknown"
    source_urls: List[str] = Field(default_factory=list)


class Chain3GoldenCorral(BaseModel):
    thanksgiving_open: Optional[str] = None  # "open" / "closed" / phrase
    opening_time: Optional[str] = None  # e.g., "11 a.m."
    closing_time_range: Optional[str] = None  # e.g., "4 p.m. to 7 p.m."
    source_urls: List[str] = Field(default_factory=list)


class Chain4McDonalds(BaseModel):
    franchise_percentage: Optional[str] = None  # e.g., "95%", "about 95%", "roughly 93%"
    hours_vary: Optional[str] = None  # text indicating hours vary by location
    verification_method: Optional[str] = None  # "store locator", "website", "mobile app"
    source_urls: List[str] = Field(default_factory=list)


class HolidayPoliciesExtraction(BaseModel):
    chick_fil_a: Optional[Chain1ChickFilA] = None
    cracker_barrel: Optional[Chain2CrackerBarrel] = None
    golden_corral: Optional[Chain3GoldenCorral] = None
    mcdonalds: Optional[Chain4McDonalds] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_policies() -> str:
    return """
Extract the restaurant holiday policy details exactly as stated in the answer, organized by chain. Return null for any field the answer does not specify.

For each chain, extract the following fields:

1) chick_fil_a:
- sunday_policy: The answer's statement about whether Chick-fil-A is closed on Sundays (e.g., "closed every Sunday", "closed", "open", or a brief phrase). Keep it concise.
- thanksgiving_policy: The answer's statement for Thanksgiving Day ("open", "closed", or a concise phrase like "most locations closed").
- christmas_policy: The answer's statement for Christmas Day ("open", "closed", or a concise phrase).
- source_urls: An array of all URLs cited in the answer that support Chick-fil-A policies. Include official links (e.g., chick-fil-a.com) if present. If no URLs are provided, return an empty array.

2) cracker_barrel:
- thanksgiving_open: The answer's statement for Thanksgiving Day ("open", "closed", or a concise phrase).
- thanksgiving_opening_time: The opening time for Thanksgiving Day as stated (e.g., "6 a.m.", "6:00 AM"), or null if not specified.
- thanksgiving_closing_time: The closing time for Thanksgiving Day (e.g., "10 p.m.", "10:00 PM"), or null if not specified.
- all_locations_open: The answer's statement whether all locations are open on Thanksgiving ("all locations", "some locations", "varies", or similar).
- source_urls: An array of all URLs cited for Cracker Barrel's Thanksgiving policy/hours.

3) golden_corral:
- thanksgiving_open: The answer's statement for Thanksgiving Day ("open", "closed", or a concise phrase).
- opening_time: The typical opening time on Thanksgiving Day (e.g., "11 a.m."), or null.
- closing_time_range: The typical closing time range on Thanksgiving (e.g., "4 p.m. to 7 p.m."), or null.
- source_urls: An array of URLs cited for Golden Corral's Thanksgiving policy/hours.

4) mcdonalds:
- franchise_percentage: The approximate franchise percentage stated in the answer (e.g., "95%", "about 95%").
- hours_vary: The answer's statement explaining that holiday hours vary by location due to franchise ownership (short sentence or phrase).
- verification_method: The method McDonald's recommends to check local holiday hours ("store locator", "website", "mobile app", or a brief phrase).
- source_urls: An array of URLs cited for McDonald's claims (prioritize official corporate sources).

IMPORTANT:
- Extract only what the answer explicitly states. Do not infer or add information.
- For URL fields, extract only explicit URLs present in the answer (including in markdown links).
- If any item is not provided in the answer, set it to null (or empty array for URLs).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_open_closed(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()
    # Look for explicit keywords
    if "closed" in t or "not open" in t or "never open" in t:
        return "closed"
    if "open" in t and "not" not in t:
        return "open"
    return None


def _additional_instruction_with_source_policy(
    sources: List[str],
    base_instruction: str,
    require_value_note: Optional[str] = None
) -> str:
    """
    Compose an additional instruction enforcing source-grounding and optionally penalizing missing values.
    If no sources are provided, instruct the verifier to mark the claim as NOT SUPPORTED.
    """
    parts = [base_instruction.strip()] if base_instruction else []
    if require_value_note:
        parts.append(require_value_note.strip())
    if not sources:
        parts.append(
            "No URL sources were provided in the answer for this claim. "
            "Per policy, you must treat the claim as NOT SUPPORTED and return Incorrect."
        )
    else:
        parts.append(
            "Use only the provided URL(s). If they do not explicitly support the claim, return Incorrect."
        )
    return " ".join(parts)


def _has_domain(urls: List[str], domain_keyword: str) -> bool:
    for u in urls:
        if domain_keyword.lower() in (u or "").lower():
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification subroutines per chain                                          #
# --------------------------------------------------------------------------- #
async def verify_chick_fil_a(evaluator: Evaluator, parent_node, info: Optional[Chain1ChickFilA]) -> None:
    node = evaluator.add_parallel(
        id="chain_1_chick_fil_a",
        desc="Research and verify holiday operating policies for Chick-fil-A",
        parent=parent_node,
        critical=False
    )

    sources = info.source_urls if info and info.source_urls else []

    # Leaf: Sunday policy (critical)
    sunday_node = evaluator.add_leaf(
        id="chain_1_sunday_policy",
        desc="Chick-fil-A is closed every Sunday",
        parent=node,
        critical=True
    )
    sunday_norm = _normalize_open_closed(info.sunday_policy if info else None)
    # Build claim from the answer's stance if present; if absent, still test canonical claim but instruct to fail due to missing info.
    if sunday_norm == "closed":
        claim_sunday = "Chick-fil-A restaurants are closed every Sunday."
        missing_note = None
    elif sunday_norm == "open":
        claim_sunday = "Chick-fil-A restaurants are open on Sundays."
        missing_note = None
    else:
        # Missing/unclear in answer
        claim_sunday = "Chick-fil-A restaurants are closed every Sunday."
        missing_note = "The answer did not state the Sunday policy; treat this as not supported."
    await evaluator.verify(
        claim=claim_sunday,
        node=sunday_node,
        sources=sources,
        additional_instruction=_additional_instruction_with_source_policy(
            sources,
            "Verify Chick-fil-A's weekly closure policy regarding Sundays. Allow minor wording variations (e.g., 'Closed on Sundays'). Prefer official Chick-fil-A sources.",
            require_value_note=missing_note
        )
    )

    # Leaf: Thanksgiving policy (critical)
    thanks_node = evaluator.add_leaf(
        id="chain_1_thanksgiving_policy",
        desc="Chick-fil-A is closed on Thanksgiving Day",
        parent=node,
        critical=True
    )
    thanks_norm = _normalize_open_closed(info.thanksgiving_policy if info else None)
    if thanks_norm == "closed":
        claim_thanks = "Chick-fil-A restaurants are closed on Thanksgiving Day."
        missing_note = None
    elif thanks_norm == "open":
        claim_thanks = "Chick-fil-A restaurants are open on Thanksgiving Day."
        missing_note = None
    else:
        claim_thanks = "Chick-fil-A restaurants are closed on Thanksgiving Day."
        missing_note = "The answer did not specify Chick-fil-A's Thanksgiving policy; treat this as not supported."
    await evaluator.verify(
        claim=claim_thanks,
        node=thanks_node,
        sources=sources,
        additional_instruction=_additional_instruction_with_source_policy(
            sources,
            "Verify the Thanksgiving Day open/closed policy for Chick-fil-A. Prefer official Chick-fil-A pages.",
            require_value_note=missing_note
        )
    )

    # Leaf: Christmas policy (critical)
    xmas_node = evaluator.add_leaf(
        id="chain_1_christmas_policy",
        desc="Chick-fil-A is closed on Christmas Day",
        parent=node,
        critical=True
    )
    xmas_norm = _normalize_open_closed(info.christmas_policy if info else None)
    if xmas_norm == "closed":
        claim_xmas = "Chick-fil-A restaurants are closed on Christmas Day."
        missing_note = None
    elif xmas_norm == "open":
        claim_xmas = "Chick-fil-A restaurants are open on Christmas Day."
        missing_note = None
    else:
        claim_xmas = "Chick-fil-A restaurants are closed on Christmas Day."
        missing_note = "The answer did not specify Chick-fil-A's Christmas Day policy; treat this as not supported."
    await evaluator.verify(
        claim=claim_xmas,
        node=xmas_node,
        sources=sources,
        additional_instruction=_additional_instruction_with_source_policy(
            sources,
            "Verify the Christmas Day open/closed policy for Chick-fil-A. Prefer official Chick-fil-A pages.",
            require_value_note=missing_note
        )
    )

    # Leaf (custom): Official reference presence (critical)
    official_ok = bool(sources) and (_has_domain(sources, "chick-fil-a.com") or _has_domain(sources, "chickfila.com"))
    evaluator.add_custom_node(
        result=official_ok,
        id="chain_1_reference_url",
        desc="Provide official Chick-fil-A source (company website or official statement) confirming these closure policies",
        parent=node,
        critical=True
    )


async def verify_cracker_barrel(evaluator: Evaluator, parent_node, info: Optional[Chain2CrackerBarrel]) -> None:
    node = evaluator.add_parallel(
        id="chain_2_cracker_barrel",
        desc="Research and verify holiday operating policies for Cracker Barrel",
        parent=parent_node,
        critical=False
    )

    sources = info.source_urls if info and info.source_urls else []

    # Leaf: Thanksgiving open/closed (critical)
    open_node = evaluator.add_leaf(
        id="chain_2_thanksgiving_open",
        desc="Cracker Barrel is open on Thanksgiving Day",
        parent=node,
        critical=True
    )
    open_norm = _normalize_open_closed(info.thanksgiving_open if info else None)
    if open_norm == "open":
        claim_open = "Cracker Barrel restaurants are open on Thanksgiving Day."
        missing_note = None
    elif open_norm == "closed":
        claim_open = "Cracker Barrel restaurants are closed on Thanksgiving Day."
        missing_note = None
    else:
        claim_open = "Cracker Barrel restaurants are open on Thanksgiving Day."
        missing_note = "The answer did not clearly state if Cracker Barrel is open on Thanksgiving; treat as not supported."
    await evaluator.verify(
        claim=claim_open,
        node=open_node,
        sources=sources,
        additional_instruction=_additional_instruction_with_source_policy(
            sources,
            "Verify whether Cracker Barrel is open on Thanksgiving Day. Use the provided source(s) only.",
            require_value_note=missing_note
        )
    )

    # Leaf: Thanksgiving hours (critical)
    hours_node = evaluator.add_leaf(
        id="chain_2_thanksgiving_hours",
        desc="Cracker Barrel operates during regular hours (6 a.m. to 10 p.m.) on Thanksgiving Day",
        parent=node,
        critical=True
    )
    opening = (info.thanksgiving_opening_time or "").strip() if info else ""
    closing = (info.thanksgiving_closing_time or "").strip() if info else ""
    if opening and closing:
        claim_hours = f"On Thanksgiving Day, Cracker Barrel operates from {opening} to {closing}."
        missing_note = None
    else:
        # Fall back to generic hours claim but instruct to fail for missing specifics
        claim_hours = "On Thanksgiving Day, Cracker Barrel operates during specific stated hours."
        missing_note = "The answer did not provide both explicit opening and closing times; treat this claim as not supported."
    await evaluator.verify(
        claim=claim_hours,
        node=hours_node,
        sources=sources,
        additional_instruction=_additional_instruction_with_source_policy(
            sources,
            "Verify the Thanksgiving Day operating hours for Cracker Barrel (both an opening and a closing time must be supported).",
            require_value_note=missing_note
        )
    )

    # Leaf: All locations open? (critical)
    all_loc_node = evaluator.add_leaf(
        id="chain_2_all_locations",
        desc="All Cracker Barrel locations (not just some) are open on Thanksgiving",
        parent=node,
        critical=True
    )
    all_loc_text = (info.all_locations_open or "").strip().lower() if info and info.all_locations_open else ""
    if "all" in all_loc_text:
        claim_all = "All Cracker Barrel locations are open on Thanksgiving Day."
        missing_note = None
    elif "some" in all_loc_text or "varies" in all_loc_text or "select" in all_loc_text:
        claim_all = "Only some Cracker Barrel locations are open on Thanksgiving Day."
        missing_note = None
    else:
        claim_all = "All Cracker Barrel locations are open on Thanksgiving Day."
        missing_note = "The answer did not clarify whether all or only some locations are open; treat this as not supported."
    await evaluator.verify(
        claim=claim_all,
        node=all_loc_node,
        sources=sources,
        additional_instruction=_additional_instruction_with_source_policy(
            sources,
            "Verify the scope of Thanksgiving openings across locations (all vs. some).",
            require_value_note=missing_note
        )
    )

    # Leaf (custom): Reference presence (critical)
    evaluator.add_custom_node(
        result=bool(sources),
        id="chain_2_reference_url",
        desc="Provide credible source (news article, company announcement, or official website) confirming Thanksgiving operating policy",
        parent=node,
        critical=True
    )


async def verify_golden_corral(evaluator: Evaluator, parent_node, info: Optional[Chain3GoldenCorral]) -> None:
    node = evaluator.add_parallel(
        id="chain_3_golden_corral",
        desc="Research and verify holiday operating policies for Golden Corral",
        parent=parent_node,
        critical=False
    )

    sources = info.source_urls if info and info.source_urls else []

    # Leaf: Thanksgiving open/closed (critical)
    open_node = evaluator.add_leaf(
        id="chain_3_thanksgiving_open",
        desc="Golden Corral is open on Thanksgiving Day",
        parent=node,
        critical=True
    )
    open_norm = _normalize_open_closed(info.thanksgiving_open if info else None)
    if open_norm == "open":
        claim_open = "Golden Corral restaurants are open on Thanksgiving Day."
        missing_note = None
    elif open_norm == "closed":
        claim_open = "Golden Corral restaurants are closed on Thanksgiving Day."
        missing_note = None
    else:
        claim_open = "Golden Corral restaurants are open on Thanksgiving Day."
        missing_note = "The answer did not clearly state if Golden Corral is open on Thanksgiving; treat as not supported."
    await evaluator.verify(
        claim=claim_open,
        node=open_node,
        sources=sources,
        additional_instruction=_additional_instruction_with_source_policy(
            sources,
            "Verify whether Golden Corral is open on Thanksgiving Day. Use the provided sources only.",
            require_value_note=missing_note
        )
    )

    # Leaf: Opening time (critical)
    opening_node = evaluator.add_leaf(
        id="chain_3_opening_time",
        desc="Golden Corral typically opens at 11 a.m. on Thanksgiving Day",
        parent=node,
        critical=True
    )
    opening_time = (info.opening_time or "").strip() if info else ""
    if opening_time:
        claim_opening = f"On Thanksgiving Day, Golden Corral typically opens at {opening_time}."
        missing_note = None
    else:
        claim_opening = "On Thanksgiving Day, Golden Corral has a typical stated opening time."
        missing_note = "The answer did not provide a specific opening time; treat this as not supported."
    await evaluator.verify(
        claim=claim_opening,
        node=opening_node,
        sources=sources,
        additional_instruction=_additional_instruction_with_source_policy(
            sources,
            "Verify the typical Thanksgiving Day opening time for Golden Corral.",
            require_value_note=missing_note
        )
    )

    # Leaf: Closing time range (critical)
    closing_node = evaluator.add_leaf(
        id="chain_3_closing_time",
        desc="Golden Corral typically closes between 4 p.m. and 7 p.m. on Thanksgiving Day (some locations may offer extended hours)",
        parent=node,
        critical=True
    )
    closing_range = (info.closing_time_range or "").strip() if info else ""
    if closing_range:
        claim_closing = f"On Thanksgiving Day, Golden Corral typically closes between {closing_range}."
        missing_note = None
    else:
        claim_closing = "On Thanksgiving Day, Golden Corral has a typical closing time range."
        missing_note = "The answer did not provide a specific closing time range; treat this as not supported."
    await evaluator.verify(
        claim=claim_closing,
        node=closing_node,
        sources=sources,
        additional_instruction=_additional_instruction_with_source_policy(
            sources,
            "Verify the typical Thanksgiving Day closing time range for Golden Corral.",
            require_value_note=missing_note
        )
    )

    # Leaf (custom): Reference presence (critical)
    evaluator.add_custom_node(
        result=bool(sources),
        id="chain_3_reference_url",
        desc="Provide credible source (restaurant guide website, news article, or official announcement) confirming Thanksgiving hours",
        parent=node,
        critical=True
    )


async def verify_mcdonalds(evaluator: Evaluator, parent_node, info: Optional[Chain4McDonalds]) -> None:
    node = evaluator.add_parallel(
        id="chain_4_mcdonalds",
        desc="Research and verify franchise structure and holiday hour policies for McDonald's",
        parent=parent_node,
        critical=False
    )

    sources = info.source_urls if info and info.source_urls else []

    # Leaf: Franchise percentage (critical)
    percent_node = evaluator.add_leaf(
        id="chain_4_franchise_percentage",
        desc="Approximately 95% of McDonald's restaurants are independently owned and operated franchises",
        parent=node,
        critical=True
    )
    percentage_text = (info.franchise_percentage or "").strip() if info else ""
    if percentage_text:
        claim_percentage = f"Approximately {percentage_text} of McDonald's restaurants are independently owned and operated by franchisees."
        missing_note = None
    else:
        claim_percentage = "Approximately 95% of McDonald's restaurants are independently owned and operated by franchisees."
        missing_note = "The answer did not provide a franchise percentage; treat this as not supported."
    await evaluator.verify(
        claim=claim_percentage,
        node=percent_node,
        sources=sources,
        additional_instruction=_additional_instruction_with_source_policy(
            sources,
            "Verify the approximate franchise percentage. Allow minor variations in percent and phrasing (e.g., 'around', 'approximately'). Prefer official corporate McDonald's pages.",
            require_value_note=missing_note
        )
    )

    # Leaf: Hours vary by location (critical)
    vary_node = evaluator.add_leaf(
        id="chain_4_hours_vary",
        desc="Holiday operating hours vary by location due to franchise ownership",
        parent=node,
        critical=True
    )
    vary_text = (info.hours_vary or "").strip().lower() if info and info.hours_vary else ""
    if vary_text:
        claim_vary = "Holiday operating hours vary by location due to franchise ownership."
        missing_note = None
    else:
        claim_vary = "Holiday operating hours vary by location due to franchise ownership."
        missing_note = "The answer did not provide an explanation that hours vary by location; treat this as not supported."
    await evaluator.verify(
        claim=claim_vary,
        node=vary_node,
        sources=sources,
        additional_instruction=_additional_instruction_with_source_policy(
            sources,
            "Verify that McDonald's communicates that hours vary by location due to franchise ownership.",
            require_value_note=missing_note
        )
    )

    # Leaf: Verification method for local hours (critical)
    method_node = evaluator.add_leaf(
        id="chain_4_verification_method",
        desc="McDonald's recommends customers check local hours using their official store locator, website, or mobile app",
        parent=node,
        critical=True
    )
    method_text = (info.verification_method or "").strip().lower() if info and info.verification_method else ""
    if method_text:
        claim_method = "McDonald's recommends customers check local hours using their official store locator, website, or mobile app."
        missing_note = None
    else:
        claim_method = "McDonald's recommends customers check local hours using their official store locator, website, or mobile app."
        missing_note = "The answer did not specify the recommended method to verify local hours; treat this as not supported."
    await evaluator.verify(
        claim=claim_method,
        node=method_node,
        sources=sources,
        additional_instruction=_additional_instruction_with_source_policy(
            sources,
            "Look for explicit guidance such as 'use the mobile app', 'check the store locator', or 'visit our website' to verify local hours.",
            require_value_note=missing_note
        )
    )

    # Leaf (custom): Official McDonald's corporate reference presence (critical)
    official_ok = bool(sources) and _has_domain(sources, "mcdonalds.com")
    evaluator.add_custom_node(
        result=official_ok,
        id="chain_4_reference_url",
        desc="Provide official McDonald's corporate source confirming franchise ownership percentage and hour variability",
        parent=node,
        critical=True
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
    Evaluate an answer for research and verification of holiday operating policies for 4 restaurant chains.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Chains are independent -> parallel aggregation
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_policies(),
        template_class=HolidayPoliciesExtraction,
        extraction_name="holiday_policies_structured"
    )

    # Build and verify per chain
    await verify_chick_fil_a(evaluator, root, extracted.chick_fil_a)
    await verify_cracker_barrel(evaluator, root, extracted.cracker_barrel)
    await verify_golden_corral(evaluator, root, extracted.golden_corral)
    await verify_mcdonalds(evaluator, root, extracted.mcdonalds)

    # Return structured result
    return evaluator.get_summary()