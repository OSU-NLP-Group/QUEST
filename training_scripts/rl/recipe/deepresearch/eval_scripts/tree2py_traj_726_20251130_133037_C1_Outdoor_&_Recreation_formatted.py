import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chatfield_pass_2025"
TASK_DESCRIPTION = (
    "A Colorado resident plans to visit Chatfield State Park 4 times during 2025. "
    "Should they purchase daily vehicle passes each time or buy a Keep Colorado Wild Pass? "
    "What is the total cost of your recommended option?"
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PassRecommendationExtraction(BaseModel):
    """
    Structured extraction of the agent's recommendation and cited prices/sources.
    """
    # Core recommendation
    recommended_option: Optional[str] = None  # Expected canonical labels: 'keep_colorado_wild_pass' or 'daily_vehicle_passes'
    recommended_option_text: Optional[str] = None  # Raw wording from the answer
    recommended_total_cost: Optional[str] = None  # e.g., "$29", "USD 29", "29 dollars", "around $40"

    # Prices explicitly stated in the answer
    daily_vehicle_pass_price: Optional[str] = None  # e.g., "$10", "$11"
    kcw_pass_price: Optional[str] = None  # e.g., "$29"

    # Visits mentioned (should be 4 per the task)
    number_of_visits: Optional[str] = None  # e.g., "4", "four"

    # Source URLs (extracted exactly as URLs from the answer)
    daily_price_sources: List[str] = Field(default_factory=list)
    kcw_price_sources: List[str] = Field(default_factory=list)
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pass_recommendation() -> str:
    return """
    Extract the recommendation and pricing details stated in the answer for a Colorado resident visiting Chatfield State Park 4 times in 2025.

    Required fields:
    - recommended_option: Use exact canonical labels:
        • "keep_colorado_wild_pass" if the answer recommends buying the Keep Colorado Wild Pass.
        • "daily_vehicle_passes" if the answer recommends buying daily vehicle passes for each visit.
        If no clear recommendation is given, return null.
    - recommended_option_text: Copy the exact phrasing the answer used to recommend the option (if any), else null.
    - recommended_total_cost: The total cost quoted by the answer for the recommended option (include the currency if present). If missing, null.
    - daily_vehicle_pass_price: The per-day vehicle pass rate as stated in the answer (e.g., "$10"). If missing, null.
    - kcw_pass_price: The Keep Colorado Wild Pass price for Colorado residents as stated in the answer (e.g., "$29"). If missing, null.
    - number_of_visits: The number of visits the answer considered/calculated with. Extract exactly as in the answer; if not explicitly stated, set to null.
    - daily_price_sources: Array of URLs that the answer cites for the Chatfield or Colorado State Parks daily vehicle pass price. Only URLs explicitly present in the answer.
    - kcw_price_sources: Array of URLs that the answer cites for the Keep Colorado Wild Pass price. Only URLs explicitly present in the answer.
    - general_sources: Array of any other URLs the answer cites that are relevant to pricing or the recommendation.

    Rules:
    - Extract only what is explicitly present in the answer. Do not invent prices or URLs.
    - For URLs, include only valid URLs (prepend http:// if protocol missing).
    - If multiple prices are shown, prefer the standard private vehicle daily park pass rate when extracting daily_vehicle_pass_price.
    - If multiple URLs are present, include all that are relevant for the specific category.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def canonicalize_option(option: Optional[str], fallback_text: Optional[str] = None) -> Optional[str]:
    """
    Normalize the recommended option into one of:
      - "keep_colorado_wild_pass"
      - "daily_vehicle_passes"
    Return None if cannot determine.
    """
    if not option and fallback_text:
        norm = fallback_text.strip().lower()
    else:
        norm = (option or "").strip().lower()

    if any(k in norm for k in ["keep colorado wild", "kcw", "annual pass", "keep‑colorado wild", "keep coloradowild"]):
        return "keep_colorado_wild_pass"
    if any(k in norm for k in ["daily vehicle", "daily pass", "day pass", "per day", "buy daily"]):
        return "daily_vehicle_passes"

    if option in ("keep_colorado_wild_pass", "daily_vehicle_passes"):
        return option
    return None


def option_human_readable(option: Optional[str]) -> str:
    if option == "keep_colorado_wild_pass":
        return "Keep Colorado Wild Pass"
    if option == "daily_vehicle_passes":
        return "daily vehicle passes"
    return "unknown option"


def parse_first_amount(value: Optional[str]) -> Optional[float]:
    """
    Parse the first monetary-like number from a string, return float.
    Accepts formats like "$10", "10", "11.00", "USD 29".
    Returns None if not found.
    """
    if not value:
        return None
    s = value.strip().lower()
    if "free" in s:
        return 0.0
    m = re.search(r"(\d[\d,]*(?:\.\d{1,2})?)", s)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return float(num)
    except Exception:
        return None


def money_str(amount: Optional[float]) -> Optional[str]:
    if amount is None:
        return None
    if abs(amount - round(amount)) < 1e-6:
        return f"${int(round(amount))}"
    return f"${amount:.2f}"


def dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        u_norm = u.strip()
        if not u_norm:
            continue
        if not (u_norm.startswith("http://") or u_norm.startswith("https://")):
            u_norm = "http://" + u_norm
        if u_norm not in seen:
            seen.add(u_norm)
            deduped.append(u_norm)
    return deduped


def visits_from_text(txt: Optional[str], default_visits: int = 4) -> int:
    if not txt:
        return default_visits
    # Try number
    m = re.search(r"\d+", txt)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            pass
    # Try words
    words = txt.strip().lower()
    mapping = {
        "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10
    }
    return mapping.get(words, default_visits)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_nodes(
    evaluator: Evaluator,
    parent_node,
    extracted: PassRecommendationExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run verification.
    """
    # Canonicalize option
    option = canonicalize_option(extracted.recommended_option, extracted.recommended_option_text)
    option_hr = option_human_readable(option)

    # Visits
    visits = visits_from_text(extracted.number_of_visits, default_visits=4)

    # Prices
    daily_price_txt = extracted.daily_vehicle_pass_price or ""
    kcw_price_txt = extracted.kcw_pass_price or ""

    daily_price_val = parse_first_amount(daily_price_txt)
    kcw_price_val = parse_first_amount(kcw_price_txt)
    daily_total_val = (daily_price_val * visits) if (daily_price_val is not None) else None

    # Sources for verification
    daily_sources = dedup_urls(extracted.daily_price_sources)
    kcw_sources = dedup_urls(extracted.kcw_price_sources)
    general_sources = dedup_urls(extracted.general_sources)
    both_sources = dedup_urls(daily_sources + kcw_sources + general_sources)

    # Add the two rubric leaf nodes
    # 1) Cost_Effective_Pass_Identified (critical)
    node_cost_effective = evaluator.add_leaf(
        id="Cost_Effective_Pass_Identified",
        desc="Must identify the pass option that results in the lower total annual cost when comparing daily vehicle passes (4 visits at Chatfield's current daily rate) versus the Keep Colorado Wild Pass (annual rate for Colorado residents)",
        parent=parent_node,
        critical=True,
    )

    # Construct claim for cost-effectiveness identification
    # We will assert that the ANSWER's recommended option is the cheaper choice for 4 visits, based on 2025 pricing.
    # The verifier will consult the URLs to read actual prices and perform the simple arithmetic comparison.
    rec_text = option_hr if option_hr != "unknown option" else (extracted.recommended_option_text or "unknown")
    claim_cost_effective = (
        f"The answer recommends '{rec_text}'. For a Colorado resident making {visits} visits to Chatfield State Park in 2025, "
        f"comparing the standard daily vehicle pass price at Chatfield (multiplied by {visits}) against the "
        f"Keep Colorado Wild Pass annual price for residents, the recommended option is indeed the lower total cost choice."
    )

    add_ins_cost_effective = (
        "Use the cited webpages to extract 2025 pricing: (1) the standard daily vehicle pass rate for Chatfield State Park or the Colorado "
        "State Parks standard daily vehicle pass rate; (2) the Keep Colorado Wild (KCW) Pass price for Colorado residents. "
        f"Compute total daily cost as {visits} × daily_rate. Then compare it to the KCW pass price. "
        "Decide which option is cheaper. Treat minor rounding differences as acceptable. "
        "If multiple daily rates are shown, use the standard private vehicle day-pass price (not special endorsements). "
        "If both totals are equal, consider either recommendation acceptable as 'cost-effective.' "
        "Finally, judge whether the answer’s recommendation matches the cheaper option."
    )

    await evaluator.verify(
        claim=claim_cost_effective,
        node=node_cost_effective,
        sources=both_sources if both_sources else None,
        additional_instruction=add_ins_cost_effective,
    )

    # 2) Correct_Total_Cost_Provided (critical)
    node_total_cost = evaluator.add_leaf(
        id="Correct_Total_Cost_Provided",
        desc="Must state the accurate total cost of the most cost-effective pass option (the option with lower total annual cost for 4 visits to Chatfield State Park) based on 2025 pricing",
        parent=parent_node,
        critical=True,
    )

    rec_total_txt = extracted.recommended_total_cost or ""
    daily_total_txt = money_str(daily_total_val) if daily_total_val is not None else None
    kcw_price_norm = money_str(kcw_price_val) if kcw_price_val is not None else None

    # Build claim for total cost correctness based on which option was recommended
    if option == "keep_colorado_wild_pass":
        # For KCW recommendation, the total cost equals the KCW pass price.
        claim_total_cost = (
            f"The answer states the total cost of the recommended option (Keep Colorado Wild Pass) as '{rec_total_txt}'. "
            f"Based on the cited sources, the 2025 Keep Colorado Wild Pass price for Colorado residents is {kcw_price_txt or 'as shown on the source'}. "
            "Therefore, the total cost for four visits using the KCW pass should equal the KCW price itself; "
            "the provided total must match that price (allowing minor formatting/rounding)."
        )
        add_ins_total_cost = (
            "Check the KCW price for 2025 on the provided KCW source(s). "
            "Confirm that the answer's stated total cost for the recommended option equals that KCW price "
            "(allowing minor formatting/rounding)."
        )
        sources_for_total = kcw_sources if kcw_sources else both_sources or None
    elif option == "daily_vehicle_passes":
        # For daily passes recommendation, the total cost equals daily price multiplied by number of visits (4).
        claim_total_cost = (
            f"The answer states the total cost of the recommended option (daily vehicle passes) as '{rec_total_txt}'. "
            f"Based on the cited sources, the 2025 standard daily vehicle pass rate at Chatfield State Park "
            f"(or the statewide standard daily park pass rate) is {daily_price_txt or 'as shown on the source'}. "
            f"Therefore, the total cost for {visits} visits should equal {visits} × daily_rate; "
            "the provided total must match that product (allowing minor formatting/rounding)."
        )
        add_ins_total_cost = (
            f"Find the standard daily vehicle pass price for 2025 from the daily pass source(s), multiply it by {visits}, "
            "and verify that the answer's stated total cost equals this result (allowing minor rounding). "
            "Use the standard private vehicle day-pass price (not special endorsements)."
        )
        sources_for_total = daily_sources if daily_sources else both_sources or None
    else:
        # Unknown/unclear recommendation; still attempt verification against whatever is provided.
        claim_total_cost = (
            f"The answer provides a total cost '{rec_total_txt}' for its recommendation, "
            f"which should be accurate for the most cost-effective option for {visits} visits in 2025. "
            "Using the cited sources, determine the daily pass price and KCW price, identify the cheaper option, "
            "and confirm that the answer's total matches the correct total for that cheaper option."
        )
        add_ins_total_cost = (
            f"Use the sources to determine both the 2025 daily vehicle pass price (and multiply by {visits}) and the KCW price for residents. "
            "Choose the cheaper option and check whether the answer's stated total equals the correct total for that cheaper option "
            "(allow small rounding)."
        )
        sources_for_total = both_sources if both_sources else None

    await evaluator.verify(
        claim=claim_total_cost,
        node=node_total_cost,
        sources=sources_for_total,
        additional_instruction=add_ins_total_cost,
    )

    # Record some helpful computed info to the summary (not part of scoring)
    evaluator.add_custom_info(
        info={
            "canonical_recommended_option": option or "unknown",
            "recommended_option_text": extracted.recommended_option_text,
            "stated_recommended_total_cost": extracted.recommended_total_cost,
            "daily_vehicle_pass_price_raw": extracted.daily_vehicle_pass_price,
            "kcw_pass_price_raw": extracted.kcw_pass_price,
            "parsed_daily_price": daily_price_val,
            "parsed_kcw_price": kcw_price_val,
            "visits_used": visits,
            "computed_daily_total": daily_total_val,
            "daily_price_sources": daily_sources,
            "kcw_price_sources": kcw_sources,
            "other_sources": general_sources,
        },
        info_type="debug",
        info_name="extracted_and_computed_values"
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
    Evaluate an answer for the Chatfield State Park pass recommendation task (2025, 4 visits).
    """
    # Initialize evaluator with a descriptive root
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

    # Create rubric root node according to provided JSON (as a child under evaluator root)
    rubric_root = evaluator.add_parallel(
        id="Pass_Recommendation",
        desc="Evaluates whether the answer correctly identifies the most cost-effective pass option and provides the accurate total cost for a Colorado resident planning 4 visits to Chatfield State Park in 2025",
        parent=root,
        critical=False,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_pass_recommendation(),
        template_class=PassRecommendationExtraction,
        extraction_name="pass_recommendation_extraction"
    )

    # Build verification nodes and verify
    await build_and_verify_nodes(evaluator, rubric_root, extracted)

    # Return the final structured summary
    return evaluator.get_summary()