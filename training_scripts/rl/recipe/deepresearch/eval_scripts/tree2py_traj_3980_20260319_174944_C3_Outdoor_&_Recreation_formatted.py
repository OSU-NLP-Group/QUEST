import asyncio
import logging
import re
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "federal_paria_katahdin_2026"
TASK_DESCRIPTION = (
    "A 62-year-old U.S. citizen is planning two wilderness trips in 2026 and does not currently own any federal "
    "recreation passes:\n\n"
    "1. Paria Canyon Trip: A solo backpacking trip to Paria Canyon (administered by the Bureau of Land Management) "
    "from May 15-17, 2026 (3 days, 2 nights overnight)\n\n"
    "2. Mount Katahdin Trip: A day hike of Mount Katahdin at Baxter State Park in Maine on July 20, 2026 (no overnight camping)\n\n"
    "Calculate the minimum total upfront cost for:\n"
    "- The most cost-effective federal recreation pass that provides value for these trips\n"
    "- The required overnight permit for Paria Canyon\n"
    "- The required day-use hiking pass for Mount Katahdin\n\n"
    "Provide the total cost and itemize each component with its cost and a reference URL supporting the pricing."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LineItem(BaseModel):
    label: Optional[str] = None  # Expected: "federal_pass", "paria_canyon_overnight", "katahdin_day_use"
    name: Optional[str] = None
    cost: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CostExtraction(BaseModel):
    # Federal pass (or None if not useful)
    federal_pass_name: Optional[str] = None
    federal_pass_cost: Optional[str] = None
    federal_pass_urls: List[str] = Field(default_factory=list)

    # Paria Canyon overnight permit
    paria_rate_per_person_per_day: Optional[str] = None  # per-day rate as text if provided in the answer
    paria_total_cost: Optional[str] = None
    paria_urls: List[str] = Field(default_factory=list)

    # Mount Katahdin day-use hiking/parking (Baxter State Park)
    katahdin_pass_name: Optional[str] = None
    katahdin_pass_cost: Optional[str] = None
    katahdin_urls: List[str] = Field(default_factory=list)

    # Final total in the answer
    total_cost: Optional[str] = None

    # Optional line-itemization parsed from the answer
    line_items: List[LineItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_costs() -> str:
    return """
    Parse the answer and extract the required pricing breakdown and sources exactly as stated.

    Required fields:
    - federal_pass_name: The selected federal recreation pass name (e.g., "Senior Annual Pass", "None/No pass", etc.). If explicitly no pass is chosen, set this to "None" or "No pass" as stated.
    - federal_pass_cost: The stated upfront cost for the chosen federal pass. If the answer explicitly selects no pass, set this to "$0" (or "0") only if the answer explicitly states a zero cost.
    - federal_pass_urls: All URLs the answer cites to support the federal pass choice and/or pricing (array). If no URL is provided, return an empty list.

    - paria_rate_per_person_per_day: The stated per-person per-day rate for Paria Canyon overnight use, if the answer mentions it; otherwise null.
    - paria_total_cost: The stated total cost for the Paria Canyon overnight permit (for the described solo 3-day trip).
    - paria_urls: All URLs the answer cites to support the Paria Canyon permit requirement/pricing (array). If no URL is provided, return an empty list.

    - katahdin_pass_name: The stated required day-use hiking/parking pass/reservation for Mount Katahdin (e.g., DUPR).
    - katahdin_pass_cost: The stated cost for that day-use hiking/parking pass/reservation.
    - katahdin_urls: All URLs the answer cites to support the Katahdin requirement/pricing (array). If no URL is provided, return an empty list.

    - total_cost: The final total cost explicitly stated in the answer.

    Also extract a structured itemization if present:
    - line_items: an array of up to 3 items with:
        - label: one of "federal_pass", "paria_canyon_overnight", "katahdin_day_use" (if clearly inferable); otherwise null
        - name: the component name as written
        - cost: the cost string as written
        - urls: all URLs for that line item (array)

    General rules:
    - Only extract what is explicitly in the answer. Do not invent or calculate values.
    - Extract full, valid URLs exactly as written in the answer (support markdown links and plain URLs).
    - Preserve currency symbols as written (e.g., "$20").
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _text_nonempty(x: Optional[str]) -> bool:
    return bool(x and str(x).strip())


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0 and any(u and isinstance(u, str) for u in urls))


def _looks_like_no_pass(name: Optional[str], cost_text: Optional[str]) -> bool:
    name_s = (name or "").strip().lower()
    if any(k in name_s for k in ["none", "no pass", "no federal pass", "not needed", "n/a"]):
        return True
    # If cost clearly zero, also accept it as "no pass" indicator
    if cost_text:
        s = cost_text.strip().lower()
        if s in {"0", "$0", "0.00", "$0.00", "free"}:
            return True
        # Parse numeric
        nums = re.findall(r"[-+]?\d*\.?\d+", s.replace(",", ""))
        if nums:
            try:
                return float(nums[0]) == 0.0
            except Exception:
                pass
    return False


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_federal_pass(evaluator: Evaluator, parent_node, data: CostExtraction) -> None:
    """
    Build and verify the 'Federal_Recreation_Pass' subtree.
    """
    fed_node = evaluator.add_parallel(
        id="Federal_Recreation_Pass",
        desc="Select and price the most cost-effective eligible federal recreation pass (if any) that provides value for the described trips under the given constraints.",
        parent=parent_node,
        critical=True
    )

    pass_name = data.federal_pass_name or ""
    pass_cost = data.federal_pass_cost or ""
    pass_urls = data.federal_pass_urls or []

    none_selected = _looks_like_no_pass(pass_name, pass_cost)

    # 1) Eligibility applied
    elig_node = evaluator.add_leaf(
        id="Federal_Pass_Eligibility_Applied",
        desc="Eligibility is correctly applied given the traveler is a 62-year-old U.S. citizen.",
        parent=fed_node,
        critical=True
    )
    if none_selected:
        claim_elig = "A 62-year-old U.S. citizen is eligible for the America the Beautiful Senior Pass."
    else:
        claim_elig = f"A 62-year-old U.S. citizen is eligible for the {pass_name}."

    await evaluator.verify(
        claim=claim_elig,
        node=elig_node,
        sources=pass_urls,
        additional_instruction="Verify eligibility conditions for the referenced pass (if a pass is selected). If no pass is selected, verify that a 62-year-old U.S. citizen is eligible for a Senior Pass."
    )

    # 2) Most cost-effective (logical reasoning; may not require URL)
    mce_node = evaluator.add_leaf(
        id="Federal_Pass_Most_Cost_Effective",
        desc="The chosen option is the minimum-upfront-cost choice among applicable federal pass options under the provided constraints while still providing value for the trips (or correctly concludes no federal pass provides value).",
        parent=fed_node,
        critical=True
    )

    if none_selected:
        claim_mce = (
            "Given the trips (a BLM Paria Canyon overnight permit and a Baxter State Park day hike), and that "
            "federal passes generally do not cover state parks or special wilderness permits, selecting no federal pass "
            "provides the minimum upfront cost while still providing appropriate value (i.e., no benefit from a federal pass)."
        )
    else:
        claim_mce = (
            f"Given the described trips and common federal pass coverage rules, selecting '{pass_name}' at upfront cost '{pass_cost}' "
            f"is the minimum-upfront-cost pass option that provides value for these trips."
        )

    await evaluator.verify(
        claim=claim_mce,
        node=mce_node,
        additional_instruction=(
            "Consider that federal passes cover entrance/standard amenity fees at federal sites, typically not state parks (Baxter is a state park) "
            "and not special recreation permits (like many BLM wilderness overnight permits). Reason about whether a federal pass offers any monetary "
            "benefit for these specific trips and if the chosen option minimizes upfront cost."
        )
    )

    # 3) Federal pass cost stated
    cost_node = evaluator.add_leaf(
        id="Federal_Pass_Cost_Stated",
        desc="State the upfront cost of the selected federal recreation pass (or $0 if no pass is selected because none provides value).",
        parent=fed_node,
        critical=True
    )
    if none_selected:
        claim_cost = "The upfront cost for the selected federal pass option is $0 (no pass selected)."
        await evaluator.verify(
            claim=claim_cost,
            node=cost_node,
            additional_instruction="Accept this only if the answer explicitly states a zero-dollar cost or 'no pass' chosen."
        )
    else:
        claim_cost = f"The upfront cost of the selected federal pass '{pass_name}' is {pass_cost}."
        await evaluator.verify(
            claim=claim_cost,
            node=cost_node,
            sources=pass_urls,
            additional_instruction="Verify the stated price against the cited federal pass information page."
        )

    # 4) Pricing reference URL supports pricing/decision
    ref_node = evaluator.add_leaf(
        id="Federal_Pass_Pricing_Reference_URL",
        desc="Provide a reference URL supporting the selected federal pass pricing (and/or the basis for selecting no pass if applicable).",
        parent=fed_node,
        critical=True
    )
    if none_selected:
        claim_ref = (
            "The cited webpage(s) support that federal recreation passes do not cover state parks and/or do not cover special recreation permits "
            "such as BLM Paria Canyon overnight permits; therefore, no federal pass provides value for these trips."
        )
    else:
        claim_ref = (
            f"The cited webpage(s) support the '{pass_name}' pricing of {pass_cost} (or clearly state the listed price used)."
        )

    await evaluator.verify(
        claim=claim_ref,
        node=ref_node,
        sources=pass_urls,
        additional_instruction=(
            "Judge as not supported if no valid URL is provided or if the page content does not align with the stated pricing/decision."
        )
    )


async def verify_paria_permit(evaluator: Evaluator, parent_node, data: CostExtraction) -> None:
    """
    Build and verify the 'Paria_Canyon_Overnight_Permit' subtree.
    """
    paria_node = evaluator.add_parallel(
        id="Paria_Canyon_Overnight_Permit",
        desc="Determine and price the required Paria Canyon overnight permit for the specified solo 3-day trip.",
        parent=parent_node,
        critical=True
    )

    total_cost = data.paria_total_cost or ""
    paria_urls = data.paria_urls or []
    per_day_rate_text = data.paria_rate_per_person_per_day or ""

    # 1) Total cost computed correctly (3 days x 1 person)
    total_node = evaluator.add_leaf(
        id="Paria_Permit_Total_Cost_Computed_Correctly",
        desc="Compute the Paria Canyon overnight permit total cost correctly using the provided rate and trip parameters (per-person per-day rate × 1 person × 3 days).",
        parent=paria_node,
        critical=True
    )
    claim_total = (
        f"For one person over three days (May 15–17, 2026), the Paria Canyon overnight permit total upfront cost is {total_cost}."
    )
    await evaluator.verify(
        claim=claim_total,
        node=total_node,
        sources=paria_urls,
        additional_instruction=(
            "Use the webpage to confirm the per-person per-day rate and verify that 3 × (per-day rate) equals the stated total. "
            "If the answer explicitly included a per-day rate ('{per_day_rate_text}'), check consistency with the page. "
            "Ignore separate lottery/processing fees unless the answer explicitly included them in the total."
        )
    )

    # 2) Pricing reference URL
    ref_node = evaluator.add_leaf(
        id="Paria_Permit_Pricing_Reference_URL",
        desc="Provide a reference URL supporting the Paria Canyon overnight permit requirement and pricing.",
        parent=paria_node,
        critical=True
    )
    claim_paria_ref = (
        "The cited webpage(s) confirm that a Paria Canyon overnight permit is required and provide the relevant per-person per-day fee used for pricing."
    )
    await evaluator.verify(
        claim=claim_paria_ref,
        node=ref_node,
        sources=paria_urls,
        additional_instruction="Judge as unsupported if the URL(s) are missing or the page content does not show the permit requirement and fee."
    )


async def verify_katahdin_pass(evaluator: Evaluator, parent_node, data: CostExtraction) -> None:
    """
    Build and verify the 'Mount_Katahdin_Day_Use_Pass' subtree.
    """
    katahdin_node = evaluator.add_parallel(
        id="Mount_Katahdin_Day_Use_Pass",
        desc="Determine and price the required day-use hiking pass for Mount Katahdin at Baxter State Park for a non-camper day hike.",
        parent=parent_node,
        critical=True
    )

    ktp_name = data.katahdin_pass_name or ""
    ktp_cost = data.katahdin_pass_cost or ""
    ktp_urls = data.katahdin_urls or []

    # 1) Cost stated
    cost_node = evaluator.add_leaf(
        id="KTP_Cost_Stated",
        desc="State the required day-use hiking pass cost for Mount Katahdin under the constraints.",
        parent=katahdin_node,
        critical=True
    )
    claim_cost = (
        f"The required day-use hiking/parking reservation for Mount Katahdin (for a non-camper day hike) costs {ktp_cost}."
    )
    await evaluator.verify(
        claim=claim_cost,
        node=cost_node,
        sources=ktp_urls,
        additional_instruction=(
            "Accept synonyms such as 'Day Use Parking Reservation (DUPR)' as the required day-use pass. Verify the price on the cited Baxter State Park page."
        )
    )

    # 2) Pricing reference URL
    ref_node = evaluator.add_leaf(
        id="KTP_Pricing_Reference_URL",
        desc="Provide a reference URL supporting the day-use hiking pass requirement and pricing.",
        parent=katahdin_node,
        critical=True
    )
    claim_ref = (
        "The cited webpage(s) confirm the requirement for a day-use hiking/parking reservation for Mount Katahdin for non-campers and show the applicable price."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_node,
        sources=ktp_urls,
        additional_instruction="Judge as unsupported if the URL(s) are missing or do not show both the requirement and the price."
    )


async def verify_components_and_validation(evaluator: Evaluator, parent_node, data: CostExtraction) -> None:
    """
    Build and verify the 'Components_Provided_And_Validated' subtree.
    """
    comp_node = evaluator.add_parallel(
        id="Components_Provided_And_Validated",
        desc="Provide and validate the three required cost components (including citations).",
        parent=parent_node,
        critical=True
    )

    # A. Check three itemized components are present (federal pass, Paria overnight, Katahdin day-use)
    # We'll validate presence using the dedicated extracted fields; if also a structured 'line_items' is present, accept that too.
    federal_present = _text_nonempty(data.federal_pass_name) and _text_nonempty(data.federal_pass_cost)
    paria_present = _text_nonempty(data.paria_total_cost)
    katahdin_present = _text_nonempty(data.katahdin_pass_cost)

    itemized_bool = (federal_present and paria_present and katahdin_present) or (len(data.line_items) >= 3)
    evaluator.add_custom_node(
        result=itemized_bool,
        id="Components_Are_Itemized_As_Three_Line_Items",
        desc="The answer presents three separate itemized components: (a) federal recreation pass (or explicitly none if applicable), (b) Paria Canyon overnight permit, and (c) Mount Katahdin day-use hiking pass.",
        parent=comp_node,
        critical=True
    )

    # B. Federal recreation pass subtree
    await verify_federal_pass(evaluator, comp_node, data)

    # C. Paria Canyon overnight permit subtree
    await verify_paria_permit(evaluator, comp_node, data)

    # D. Mount Katahdin day-use pass subtree
    await verify_katahdin_pass(evaluator, comp_node, data)


async def verify_final_total(evaluator: Evaluator, parent_node, data: CostExtraction) -> None:
    """
    Build and verify the 'Final_Total_Sum' subtree.
    """
    total_node = evaluator.add_parallel(
        id="Final_Total_Sum",
        desc="Provide the final total upfront cost and ensure it equals the sum of the itemized component costs (federal pass + Paria permit + Katahdin day-use pass).",
        parent=parent_node,
        critical=True
    )

    # 1) Total provided (existence)
    evaluator.add_custom_node(
        result=_text_nonempty(data.total_cost),
        id="Total_Cost_Provided",
        desc="A single final total upfront cost is explicitly stated.",
        parent=total_node,
        critical=True
    )

    # 2) Total equals sum (use LLM arithmetic verification)
    equals_node = evaluator.add_leaf(
        id="Total_Equals_Sum_Of_Components",
        desc="The stated total equals the arithmetic sum of the three component costs given in the breakdown.",
        parent=total_node,
        critical=True
    )

    fed_cost = data.federal_pass_cost or ""
    paria_cost = data.paria_total_cost or ""
    katahdin_cost = data.katahdin_pass_cost or ""
    total_cost = data.total_cost or ""

    claim_sum = (
        f"The final total {total_cost} equals the sum of the three components: "
        f"{fed_cost} (federal pass) + {paria_cost} (Paria permit) + {katahdin_cost} (Katahdin day-use)."
    )
    await evaluator.verify(
        claim=claim_sum,
        node=equals_node,
        additional_instruction=(
            "Compute numerically from the given currency amounts (treat missing currency symbols reasonably). "
            "Minor formatting differences (e.g., $ signs, commas) should be ignored. Accept if equal within normal rounding to nearest cent."
        )
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
    Evaluate an answer for the 2026 federal pass + Paria Canyon + Mount Katahdin pricing task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root container
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

    # Add the top-level critical sequential node as specified by the rubric
    top = evaluator.add_sequential(
        id="Total_Cost_Calculation",
        desc="Compute the minimum total upfront cost for (1) the most cost-effective federal recreation pass (if any) that provides value for these trips, (2) the Paria Canyon overnight permit, and (3) the Mount Katahdin day-use hiking pass; provide the total and itemize components with prices supported by reference URLs.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_costs(),
        template_class=CostExtraction,
        extraction_name="cost_breakdown_extraction",
    )

    # Child 1: Components provided and validated (parallel, critical)
    await verify_components_and_validation(evaluator, top, extracted)

    # Child 2: Final total sum (parallel, critical)
    await verify_final_total(evaluator, top, extracted)

    # Return summary
    return evaluator.get_summary()