import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "america_the_beautiful_pass_trip_planning"
TASK_DESCRIPTION = (
    "I am a 63-year-old US citizen planning a summer 2025 national parks road trip with my spouse (age 61) and two grandchildren (ages 10 and 18). "
    "We'll travel in one rental car and camp at some parks while staying at hotels near others. I'm interested in purchasing an America the Beautiful pass. "
    "Please provide comprehensive information about my pass options, including: which pass types I'm eligible for and their costs; how many people in my group would be covered at per-vehicle versus per-person fee sites; "
    "whether my 18-year-old granddaughter qualifies for free admission; what fees the pass covers (entrance fees, camping, parking) and what discounts I might receive as a senior; where and how I can purchase the pass (including digital options); "
    "what documentation is required; how long the pass is valid; and whether it can be replaced if lost during my trip."
)

# Official candidate URLs for verification (fallback if the answer provides no sources)
OFFICIAL_URLS = [
    # USGS (Interagency Pass Store) – canonical authority
    "https://store.usgs.gov/pass",
    "https://store.usgs.gov/senior-pass",
    # National Park Service overview for passes
    "https://www.nps.gov/planyourvisit/passes.htm",
    # Recreation.gov – digital Annual Pass
    "https://www.recreation.gov/passes",
    "https://www.recreation.gov/passes/annual-pass",
]

# --------------------------------------------------------------------------- #
# Data model for extracting structured info from the answer                   #
# --------------------------------------------------------------------------- #
class PassAnswerExtraction(BaseModel):
    # Source URLs mentioned in the answer (any URLs)
    urls: List[str] = Field(default_factory=list)

    # Pass types mentioned
    mentions_annual_pass: Optional[bool] = None
    mentions_senior_annual_pass: Optional[bool] = None
    mentions_senior_lifetime_pass: Optional[bool] = None

    # Costs explicitly stated in the answer (as written)
    cost_annual: Optional[str] = None  # e.g., "$80"
    cost_senior_annual: Optional[str] = None  # e.g., "$20"
    cost_senior_lifetime: Optional[str] = None  # e.g., "$80"
    mentions_online_processing_fee: Optional[bool] = None
    online_processing_fee_amount: Optional[str] = None  # e.g., "~$5" or "$5"

    # Senior eligibility content in the answer
    mentions_senior_eligibility_rule: Optional[bool] = None  # Explicitly states: citizens/permanent residents age 62+
    applies_63yo_eligible: Optional[bool] = None  # Explicitly applies that a 63-year-old US citizen is eligible

    # Coverage rules and application
    mentions_per_vehicle_rule: Optional[bool] = None  # One pass covers owner + passengers in a non-commercial vehicle
    applies_per_vehicle_to_group: Optional[bool] = None  # Explicitly applies to this group in one rental car
    mentions_per_person_rule: Optional[bool] = None  # One pass covers owner + up to 3 additional adults (max 4 adults)
    applies_per_person_to_group: Optional[bool] = None  # Explicitly applies correct adult count handling in this group

    # Under 16 policy
    mentions_under_16_free: Optional[bool] = None  # Children under 16 are free at pass-accepting sites
    applies_under_16_and_18: Optional[bool] = None  # Explicitly states 10-year-old is free and 18-year-old is not free under that policy

    # Fees covered and exclusions
    mentions_covers_entrance_and_standard_amenity: Optional[bool] = None
    mentions_not_cover_expanded_amenities: Optional[bool] = None  # camping, boat launch, parking, tours, ferries, special permits
    mentions_not_cover_concessionaire: Optional[bool] = None

    # Senior pass discounts and limits
    mentions_senior_discount_50: Optional[bool] = None  # 50% on some amenity fees (camping, swimming, boat launch, interpretive)
    mentions_senior_discount_limits: Optional[bool] = None  # No discount on special recreation permits or concessionaires

    # Purchase options and documentation
    mentions_digital_recgov_immediate: Optional[bool] = None
    mentions_in_person_sites_1000: Optional[bool] = None
    mentions_usgs_delivery_timeline: Optional[bool] = None  # up to ~3 weeks
    mentions_senior_documentation: Optional[bool] = None  # age + citizenship/residency
    mentions_photo_id_required: Optional[bool] = None

    # Validity and replacement/administrative policies
    mentions_valid_12_months_from_month: Optional[bool] = None
    mentions_nonrefundable: Optional[bool] = None
    mentions_nontransferable: Optional[bool] = None
    mentions_cannot_replace_lost: Optional[bool] = None

    # Acceptance and scope
    mentions_accepted_agencies: Optional[bool] = None  # NPS, USFWS, USFS, BLM, Bureau of Reclamation, USACE
    mentions_more_than_2000_sites: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pass_answer() -> str:
    return """
Extract from the answer the exact information requested below. Return booleans as True only if the answer explicitly states the fact. Do not infer anything not written in the answer.

Also extract all URLs explicitly present in the answer text (any http/https links). Do not invent URLs.

Return fields:
- urls: array of all URLs in the answer.

Pass types mentioned (booleans):
- mentions_annual_pass
- mentions_senior_annual_pass
- mentions_senior_lifetime_pass

Costs (as written in the answer; strings):
- cost_annual
- cost_senior_annual
- cost_senior_lifetime
Fee note (booleans/strings):
- mentions_online_processing_fee
- online_processing_fee_amount  // e.g., "~$5", "$5", "about $5", etc.

Senior eligibility (booleans):
- mentions_senior_eligibility_rule  // explicitly states: available to US citizens or permanent residents age 62+
- applies_63yo_eligible  // explicitly applies that a 63-year-old US citizen is eligible

Coverage rules and application (booleans):
- mentions_per_vehicle_rule       // one pass covers pass owner + passengers in a non-commercial vehicle
- applies_per_vehicle_to_group    // explicitly applies this to the group in one rental car
- mentions_per_person_rule        // one pass covers pass owner + up to 3 additional adults (max 4 adults total)
- applies_per_person_to_group     // explicitly applies adult count correctly to the described group

Under 16 policy (booleans):
- mentions_under_16_free          // under 16 are free at pass-accepting sites
- applies_under_16_and_18         // 10-year-old free; 18-year-old not free under under-16 policy

Fees coverage (booleans):
- mentions_covers_entrance_and_standard_amenity      // entrance + standard amenity (day-use) fees covered
- mentions_not_cover_expanded_amenities              // not covered: camping, boat launch, parking, tours, ferries, special permits
- mentions_not_cover_concessionaire                  // not covered: concessionaire fees

Senior Pass discounts and limits (booleans):
- mentions_senior_discount_50                        // 50% on some amenities: camping, swimming, boat launch, specialized interpretive
- mentions_senior_discount_limits                    // NOT reduced: special recreation permits or concessionaires

Purchase options and documentation (booleans):
- mentions_digital_recgov_immediate                  // digital Annual Pass available immediately via Recreation.gov
- mentions_in_person_sites_1000                      // physical passes purchasable at 1,000+ federal recreation sites
- mentions_usgs_delivery_timeline                    // physical Annual Pass ordered via USGS may take up to ~3 weeks
- mentions_senior_documentation                      // senior pass requires proof of age + US citizenship or permanent residency
- mentions_photo_id_required                         // pass holders must show valid photo ID when using the pass

Validity and administrative (booleans):
- mentions_valid_12_months_from_month                // validity: 12 months from the month of purchase
- mentions_nonrefundable                             // non-refundable
- mentions_nontransferable                           // non-transferable
- mentions_cannot_replace_lost                       // cannot be replaced if lost or stolen

Acceptance and scope (booleans):
- mentions_accepted_agencies                         // accepted at: NPS, USFWS, USFS, BLM, Bureau of Reclamation, USACE
- mentions_more_than_2000_sites                      // more than 2,000 federal recreation sites

Rules:
- Set a boolean to True only if the answer explicitly states it.
- For any missing field, return null for strings or False for booleans if not explicitly stated.
- For costs, copy exactly what the answer wrote (e.g., "$80", "80 dollars").
- For URLs, collect any http/https links, including markdown links.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def b(val: Optional[bool]) -> bool:
    return bool(val)

def merge_sources(answer_urls: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for url in (answer_urls or []) + OFFICIAL_URLS:
        if not url:
            continue
        if url not in seen:
            seen.add(url)
            merged.append(url)
    return merged


async def add_presence_and_support(
    evaluator: Evaluator,
    parent_node,
    base_id: str,
    node_desc: str,
    presence_bool: bool,
    presence_desc: str,
    support_claim: str,
    urls: List[str],
    support_instruction: str = "Verify the claim against the official interagency pass information. Allow minor wording variations.",
):
    """
    Build a sequential critical node with two critical leaves:
    - presence check (custom node)
    - support by sources (verify_by_urls)
    """
    seq_node = evaluator.add_sequential(
        id=base_id,
        desc=node_desc,
        parent=parent_node,
        critical=True
    )

    # Presence
    evaluator.add_custom_node(
        result=presence_bool,
        id=f"{base_id}_stated_in_answer",
        desc=presence_desc,
        parent=seq_node,
        critical=True
    )

    # Support by sources
    support_leaf = evaluator.add_leaf(
        id=f"{base_id}_supported_by_sources",
        desc=f"{node_desc} — supported by official sources",
        parent=seq_node,
        critical=True
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_leaf,
        sources=urls,
        additional_instruction=support_instruction
    )

    return seq_node


async def add_presence_support_and_application(
    evaluator: Evaluator,
    parent_node,
    base_id: str,
    node_desc: str,
    presence_bool: bool,
    presence_desc: str,
    support_claim: str,
    urls: List[str],
    application_desc: str,
    application_claim: str,
    support_instruction: str = "Verify the claim against the official interagency pass information.",
    application_instruction: str = "Judge the logical application using the provided group composition."
):
    """
    Build a sequential critical node with three critical leaves:
    - presence check (custom node)
    - support by sources (verify_by_urls)
    - application to the described group (simple verification)
    """
    seq_node = evaluator.add_sequential(
        id=base_id,
        desc=node_desc,
        parent=parent_node,
        critical=True
    )

    # Presence
    evaluator.add_custom_node(
        result=presence_bool,
        id=f"{base_id}_stated_in_answer",
        desc=presence_desc,
        parent=seq_node,
        critical=True
    )

    # Support by sources
    support_leaf = evaluator.add_leaf(
        id=f"{base_id}_supported_by_sources",
        desc=f"{node_desc} — supported by official sources",
        parent=seq_node,
        critical=True
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_leaf,
        sources=urls,
        additional_instruction=support_instruction
    )

    # Application
    application_leaf = evaluator.add_leaf(
        id=f"{base_id}_applied_to_group",
        desc=application_desc,
        parent=seq_node,
        critical=True
    )
    await evaluator.verify(
        claim=application_claim,
        node=application_leaf,
        sources=None,
        additional_instruction=application_instruction
    )

    return seq_node


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_eligibility_and_pass_types(
    evaluator: Evaluator,
    parent,
    ex: PassAnswerExtraction,
    urls: List[str]
):
    elig_node = evaluator.add_parallel(
        id="Eligibility_and_Pass_Types",
        desc="Identifies pass types relevant to the traveler’s situation per constraints.",
        parent=parent,
        critical=True
    )

    # Senior pass eligibility (rule + application)
    await add_presence_support_and_application(
        evaluator=evaluator,
        parent_node=elig_node,
        base_id="Senior_Pass_Eligibility",
        node_desc="Senior passes are for US citizens/permanent residents age 62+, applied to the 63-year-old US citizen",
        presence_bool=b(ex.mentions_senior_eligibility_rule) and b(ex.applies_63yo_eligible),
        presence_desc="The answer states the senior eligibility rule (US citizen/permanent resident age 62+) and explicitly applies it to the 63-year-old US citizen.",
        support_claim="The Interagency Senior Pass is available to U.S. citizens or permanent residents aged 62 or older.",
        urls=urls,
        application_desc="Correctly applies that a 63-year-old U.S. citizen is eligible for a Senior Pass.",
        application_claim="A 63-year-old U.S. citizen is eligible for a Senior Pass."
    )

    # Pass types mentioned (Annual, Senior Annual, Senior Lifetime)
    presence_all_types = b(ex.mentions_annual_pass) and b(ex.mentions_senior_annual_pass) and b(ex.mentions_senior_lifetime_pass)
    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=elig_node,
        base_id="Pass_Types_Mentioned",
        node_desc="Mentions relevant pass types: Annual Pass, Senior Annual Pass, Senior Lifetime Pass",
        presence_bool=presence_all_types,
        presence_desc="The answer mentions the Annual Pass, the Senior Annual Pass, and the Senior Lifetime Pass.",
        support_claim="The Interagency pass program includes the Annual Pass and the Senior Pass (available as an Annual Senior Pass and a Lifetime Senior Pass).",
        urls=urls,
        support_instruction="Verify that these pass types exist in the Interagency pass program. Allow minor naming variations like 'Interagency Annual Pass'."
    )


async def build_pass_costs(
    evaluator: Evaluator,
    parent,
    ex: PassAnswerExtraction,
    urls: List[str]
):
    # Convert costs directly from the answer (strings). Presence = field is non-empty.
    costs_root = evaluator.add_parallel(
        id="Pass_Costs",
        desc="States costs from constraints and the online processing fee note.",
        parent=parent,
        critical=True
    )

    # Annual Pass cost
    annual_cost_node = evaluator.add_sequential(
        id="Annual_Pass_Cost",
        desc="Annual Pass cost is correctly stated and supported",
        parent=costs_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ex.cost_annual),
        id="Annual_Pass_Cost_stated_in_answer",
        desc="The answer states the Annual Pass price.",
        parent=annual_cost_node,
        critical=True
    )
    annual_cost_leaf = evaluator.add_leaf(
        id="Annual_Pass_Cost_supported_by_sources",
        desc="Annual Pass cost as stated in the answer is supported by official sources",
        parent=annual_cost_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Interagency Annual Pass costs {ex.cost_annual}.",
        node=annual_cost_leaf,
        sources=urls,
        additional_instruction="Verify that the price matches official sources. Allow '$' or 'dollars' equivalents."
    )

    # Senior Annual Pass cost
    s_annual_cost_node = evaluator.add_sequential(
        id="Senior_Annual_Pass_Cost",
        desc="Senior Annual Pass cost is correctly stated and supported",
        parent=costs_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ex.cost_senior_annual),
        id="Senior_Annual_Pass_Cost_stated_in_answer",
        desc="The answer states the Senior Annual Pass price.",
        parent=s_annual_cost_node,
        critical=True
    )
    s_annual_cost_leaf = evaluator.add_leaf(
        id="Senior_Annual_Pass_Cost_supported_by_sources",
        desc="Senior Annual Pass cost as stated in the answer is supported by official sources",
        parent=s_annual_cost_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Senior Annual Pass costs {ex.cost_senior_annual}.",
        node=s_annual_cost_leaf,
        sources=urls,
        additional_instruction="Verify the stated Senior Annual Pass price against official sources."
    )

    # Senior Lifetime Pass cost
    s_life_cost_node = evaluator.add_sequential(
        id="Senior_Lifetime_Pass_Cost",
        desc="Senior Lifetime Pass cost is correctly stated and supported",
        parent=costs_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ex.cost_senior_lifetime),
        id="Senior_Lifetime_Pass_Cost_stated_in_answer",
        desc="The answer states the Senior Lifetime Pass price.",
        parent=s_life_cost_node,
        critical=True
    )
    s_life_cost_leaf = evaluator.add_leaf(
        id="Senior_Lifetime_Pass_Cost_supported_by_sources",
        desc="Senior Lifetime Pass cost as stated in the answer is supported by official sources",
        parent=s_life_cost_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Senior Lifetime Pass costs {ex.cost_senior_lifetime}.",
        node=s_life_cost_leaf,
        sources=urls,
        additional_instruction="Verify the stated Senior Lifetime Pass price against official sources."
    )

    # Online processing fee (~$5) when ordering Annual Pass online
    fee_node = evaluator.add_sequential(
        id="Annual_Pass_Online_Processing_Fee",
        desc="Online Annual Pass order includes an additional processing fee (~$5)",
        parent=costs_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=b(ex.mentions_online_processing_fee),
        id="Annual_Pass_Online_Processing_Fee_stated_in_answer",
        desc="The answer notes an approximate online processing fee when ordering the Annual Pass online.",
        parent=fee_node,
        critical=True
    )
    fee_leaf = evaluator.add_leaf(
        id="Annual_Pass_Online_Processing_Fee_supported_by_sources",
        desc="Online Annual Pass processing fee note is supported by official sources",
        parent=fee_node,
        critical=True
    )
    approx_fee_text = ex.online_processing_fee_amount or "about $5"
    await evaluator.verify(
        claim=f"Ordering the Interagency Annual Pass online includes an additional processing fee (approximately {approx_fee_text}).",
        node=fee_leaf,
        sources=urls,
        additional_instruction="Check that online orders include a processing/handling fee. Allow approximate amounts around $5."
    )


async def build_coverage_rules(
    evaluator: Evaluator,
    parent,
    ex: PassAnswerExtraction,
    urls: List[str]
):
    cov_root = evaluator.add_parallel(
        id="Coverage_Rules_For_Group",
        desc="Correctly explains how many people are covered at per-vehicle vs per-person sites, applied to the described group.",
        parent=parent,
        critical=True
    )

    # Per-vehicle coverage
    await add_presence_support_and_application(
        evaluator=evaluator,
        parent_node=cov_root,
        base_id="Per_Vehicle_Site_Coverage_For_Group",
        node_desc="Per-vehicle fee sites: one pass covers the pass holder and all passengers in a non-commercial vehicle",
        presence_bool=b(ex.mentions_per_vehicle_rule) and b(ex.applies_per_vehicle_to_group),
        presence_desc="The answer states the per-vehicle rule and explicitly applies it to the group traveling together in one rental car.",
        support_claim="At per-vehicle fee sites, a single Interagency pass covers the pass holder and passengers in one non-commercial vehicle.",
        urls=urls,
        application_desc="Applies the per-vehicle rule to this group traveling in one car (all covered in the same non-commercial vehicle).",
        application_claim="Because they will travel in one non-commercial rental car, one pass covers the pass holder and all passengers at per-vehicle fee sites."
    )

    # Per-person coverage
    await add_presence_support_and_application(
        evaluator=evaluator,
        parent_node=cov_root,
        base_id="Per_Person_Site_Coverage_For_Group",
        node_desc="Per-person fee sites: one pass covers the pass holder plus up to three additional adults (max 4 adults total)",
        presence_bool=b(ex.mentions_per_person_rule) and b(ex.applies_per_person_to_group),
        presence_desc="The answer states the per-person rule and correctly applies it to the group's adult count (spouse and 18-year-old counted as adults).",
        support_claim="At per-person fee sites, one pass covers the pass holder plus up to 3 additional adults (maximum of 4 adults total), with children under 16 admitted free.",
        urls=urls,
        application_desc="Applies the per-person rule correctly to the group's adults (pass holder, spouse, and 18-year-old are within the 4-adult limit).",
        application_claim="The group has the pass holder (age 63), spouse (61), and an 18-year-old (adult). They are within the 4-adult limit covered by one pass at per-person fee sites."
    )

    # Under 16 policy and application
    await add_presence_support_and_application(
        evaluator=evaluator,
        parent_node=cov_root,
        base_id="Under_16_Free_Admission_And_Application",
        node_desc="Children under 16 are free at pass-accepting sites; applied to this group's 10- and 18-year-olds",
        presence_bool=b(ex.mentions_under_16_free) and b(ex.applies_under_16_and_18),
        presence_desc="The answer states that children under 16 are free and applies it (10-year-old free; 18-year-old not free under this rule).",
        support_claim="Children under age 16 are admitted free at sites that accept the Interagency pass.",
        urls=urls,
        application_desc="Correctly applies under-16 policy to the group's children.",
        application_claim="The 10-year-old is admitted free under the under-16 policy; the 18-year-old is not free under that policy and counts as an adult."
    )


async def build_fees_coverage_and_exclusions(
    evaluator: Evaluator,
    parent,
    ex: PassAnswerExtraction,
    urls: List[str]
):
    fees_root = evaluator.add_parallel(
        id="Fees_Coverage_And_Exclusions",
        desc="Correctly states what the Annual Pass covers and does not cover.",
        parent=parent,
        critical=True
    )

    # Covered fees (entrance + standard amenity/day-use)
    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=fees_root,
        base_id="Fees_Covered_By_Annual_Pass",
        node_desc="Annual Pass covers entrance and standard amenity (day-use) fees",
        presence_bool=b(ex.mentions_covers_entrance_and_standard_amenity),
        presence_desc="The answer states that the pass covers entrance fees and standard amenity (day-use) fees.",
        support_claim="The Interagency Annual Pass covers entrance fees and standard amenity (day-use) fees at federal recreation sites.",
        urls=urls
    )

    # Not covered: expanded amenity fees (camping, boat launch, parking, tours, ferries, special permits)
    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=fees_root,
        base_id="Expanded_Amenity_Fees_Not_Covered",
        node_desc="Annual Pass does not cover expanded amenity fees (e.g., camping, boat launch, parking, tours, special permits, ferries)",
        presence_bool=b(ex.mentions_not_cover_expanded_amenities),
        presence_desc="The answer states that expanded amenity fees are not covered (e.g., camping, boat launch, parking, special tours, special permits, ferries).",
        support_claim="The Interagency Annual Pass does not cover expanded amenity fees such as camping, boat launching, parking, special tours, special permits, or ferries.",
        urls=urls
    )

    # Not covered: concessionaire fees
    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=fees_root,
        base_id="Concessionaire_Fees_Not_Covered",
        node_desc="Annual Pass does not cover concessionaire fees",
        presence_bool=b(ex.mentions_not_cover_concessionaire),
        presence_desc="The answer states that concessionaire fees are not covered.",
        support_claim="The Interagency Annual Pass does not cover concessionaire fees.",
        urls=urls
    )


async def build_senior_discounts_and_limits(
    evaluator: Evaluator,
    parent,
    ex: PassAnswerExtraction,
    urls: List[str]
):
    disc_root = evaluator.add_parallel(
        id="Senior_Pass_Discounts_And_Limits",
        desc="Correctly states Senior Pass discount availability and limitations.",
        parent=parent,
        critical=True
    )

    # 50% discount on some amenities
    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=disc_root,
        base_id="Senior_Pass_50Percent_Discount_On_Some_Amenities",
        node_desc="Senior Pass may provide a 50% discount on some amenity fees (camping, swimming, boat launch, specialized interpretive services)",
        presence_bool=b(ex.mentions_senior_discount_50),
        presence_desc="The answer states that the Senior Pass may provide a 50% discount on some amenity fees (e.g., camping, swimming, boat launch, specialized interpretive services).",
        support_claim="The Senior Pass may provide a 50% discount on some amenity fees such as camping, swimming, boat launch, and specialized interpretive services.",
        urls=urls
    )

    # No discount on special recreation permits or concessionaires
    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=disc_root,
        base_id="Senior_Pass_No_Discount_On_Special_Recreation_Permits_Or_Concessionaires",
        node_desc="Senior Pass does not reduce special recreation permit fees or concessionaire fees",
        presence_bool=b(ex.mentions_senior_discount_limits),
        presence_desc="The answer states that the Senior Pass does not reduce special recreation permit fees or concessionaire fees.",
        support_claim="The Senior Pass does not reduce special recreation permit fees or concessionaire fees.",
        urls=urls
    )


async def build_purchase_options_and_documentation(
    evaluator: Evaluator,
    parent,
    ex: PassAnswerExtraction,
    urls: List[str]
):
    purchase_root = evaluator.add_parallel(
        id="Purchase_Options_And_Documentation",
        desc="Correctly states purchase methods and required documentation.",
        parent=parent,
        critical=True
    )

    # Digital Annual Pass via Recreation.gov (immediate)
    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=purchase_root,
        base_id="Purchase_Options_Digital",
        node_desc="Digital Annual Passes are available immediately through Recreation.gov",
        presence_bool=b(ex.mentions_digital_recgov_immediate),
        presence_desc="The answer states that digital Annual Passes are available immediately through Recreation.gov.",
        support_claim="Digital Interagency Annual Passes are available immediately through Recreation.gov.",
        urls=urls,
        support_instruction="Verify that Recreation.gov offers a digital Annual Pass that is available immediately upon purchase."
    )

    # In-person at 1,000+ sites
    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=purchase_root,
        base_id="Purchase_Options_In_Person",
        node_desc="Physical passes can be purchased in person at over 1,000 federal recreation sites",
        presence_bool=b(ex.mentions_in_person_sites_1000),
        presence_desc="The answer states that physical passes can be purchased at over 1,000 federal recreation sites.",
        support_claim="Physical Interagency passes can be purchased in person at more than 1,000 federal recreation sites.",
        urls=urls
    )

    # USGS Store delivery time up to ~3 weeks
    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=purchase_root,
        base_id="Purchase_Options_USGS_Online_Delivery_Time",
        node_desc="Physical Annual Passes ordered through the USGS Store may take up to 3 weeks to be processed and delivered",
        presence_bool=b(ex.mentions_usgs_delivery_timeline),
        presence_desc="The answer states that physical Annual Passes ordered via the USGS Store may take up to ~3 weeks to process and deliver.",
        support_claim="Physical Annual Passes ordered through the USGS Store may take up to about three weeks for processing and delivery.",
        urls=urls,
        support_instruction="Check the USGS Store page for processing and shipping timelines; allow 'up to ~3 weeks' phrasing."
    )

    # Documentation for Senior Pass purchase (age + citizenship/residency)
    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=purchase_root,
        base_id="Documentation_For_Senior_Pass_Purchase",
        node_desc="Senior Pass applicants must provide documentation of age and residency or citizenship",
        presence_bool=b(ex.mentions_senior_documentation),
        presence_desc="The answer states that Senior Pass applicants must provide documentation of age and U.S. citizenship or permanent residency.",
        support_claim="Senior Pass applicants must provide proof of age and U.S. citizenship or permanent residency.",
        urls=urls
    )

    # Photo ID requirement
    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=purchase_root,
        base_id="Photo_ID_Required_For_Pass_Use",
        node_desc="Pass holders must show valid photo identification to verify pass ownership",
        presence_bool=b(ex.mentions_photo_id_required),
        presence_desc="The answer states that pass holders must show valid photo identification to verify pass ownership.",
        support_claim="Pass holders must show a valid photo ID to verify pass ownership; passes are signed and non-transferable.",
        urls=urls
    )


async def build_validity_and_replacement_policies(
    evaluator: Evaluator,
    parent,
    ex: PassAnswerExtraction,
    urls: List[str]
):
    pol_root = evaluator.add_parallel(
        id="Validity_And_Replacement_Policies",
        desc="Correctly states validity and administrative policies.",
        parent=parent,
        critical=True
    )

    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=pol_root,
        base_id="Annual_Pass_Validity",
        node_desc="Annual Pass validity: 12 months from the month of purchase",
        presence_bool=b(ex.mentions_valid_12_months_from_month),
        presence_desc="The answer states that the Annual Pass is valid for 12 months from the month of purchase.",
        support_claim="An Interagency Annual Pass is valid for 12 months from the month of purchase.",
        urls=urls
    )

    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=pol_root,
        base_id="Nonrefundable_Policy",
        node_desc="Annual Passes are non-refundable",
        presence_bool=b(ex.mentions_nonrefundable),
        presence_desc="The answer states that passes are non-refundable.",
        support_claim="Interagency passes are non-refundable.",
        urls=urls
    )

    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=pol_root,
        base_id="Nontransferable_Policy",
        node_desc="Annual Passes are non-transferable",
        presence_bool=b(ex.mentions_nontransferable),
        presence_desc="The answer states that passes are non-transferable.",
        support_claim="Interagency passes are non-transferable (signed and must be shown with photo ID).",
        urls=urls
    )

    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=pol_root,
        base_id="Lost_Or_Stolen_Replacement_Policy",
        node_desc="Annual Passes cannot be replaced if lost or stolen",
        presence_bool=b(ex.mentions_cannot_replace_lost),
        presence_desc="The answer states that passes cannot be replaced if lost or stolen.",
        support_claim="Interagency passes cannot be replaced if lost or stolen.",
        urls=urls
    )


async def build_acceptance_and_scope(
    evaluator: Evaluator,
    parent,
    ex: PassAnswerExtraction,
    urls: List[str]
):
    scope_root = evaluator.add_parallel(
        id="Acceptance_And_Scope",
        desc="Correctly states where the Annual Pass is accepted and its general scope.",
        parent=parent,
        critical=True
    )

    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=scope_root,
        base_id="Accepted_Agencies",
        node_desc="Pass accepted at six federal agencies (NPS, USFWS, USFS, BLM, Bureau of Reclamation, USACE)",
        presence_bool=b(ex.mentions_accepted_agencies),
        presence_desc="The answer states that the pass is accepted at NPS, USFWS, USFS, BLM, Bureau of Reclamation, and USACE sites.",
        support_claim="The Interagency pass is accepted at sites managed by the National Park Service (NPS), U.S. Fish and Wildlife Service (USFWS), U.S. Forest Service (USFS), Bureau of Land Management (BLM), Bureau of Reclamation, and the U.S. Army Corps of Engineers (USACE).",
        urls=urls
    )

    await add_presence_and_support(
        evaluator=evaluator,
        parent_node=scope_root,
        base_id="Number_Of_Sites",
        node_desc="Pass provides access to more than 2,000 federal recreation sites",
        presence_bool=b(ex.mentions_more_than_2000_sites),
        presence_desc="The answer states that the pass provides access to more than 2,000 federal recreation sites.",
        support_claim="The Interagency pass provides access to more than 2,000 federal recreation sites.",
        urls=urls
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
    Evaluate an answer for the 'America the Beautiful' pass planning scenario.
    """
    # Initialize evaluator and root
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

    # Create main critical node (as per rubric)
    main = evaluator.add_parallel(
        id="Trip_Planning_Pass_Information",
        desc="Accurate answers about America the Beautiful pass options for the described travel group and scenario, evaluated strictly against the provided constraints.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    ex: PassAnswerExtraction = await evaluator.extract(
        prompt=prompt_extract_pass_answer(),
        template_class=PassAnswerExtraction,
        extraction_name="pass_answer_extraction"
    )

    # Merge candidate sources: answer-provided + official fallbacks
    candidate_urls = merge_sources(ex.urls)

    # Record ground-truth context for transparency (not used for scoring)
    evaluator.add_ground_truth(
        gt_info={
            "group": {
                "traveler_age": 63,
                "traveler_citizenship": "US citizen",
                "spouse_age": 61,
                "grandchildren_ages": [10, 18],
                "vehicle": "one non-commercial rental car"
            },
            "official_candidate_urls": candidate_urls
        },
        gt_type="context"
    )

    # Build all rubric subtrees
    await build_eligibility_and_pass_types(evaluator, main, ex, candidate_urls)
    await build_pass_costs(evaluator, main, ex, candidate_urls)
    await build_coverage_rules(evaluator, main, ex, candidate_urls)
    await build_fees_coverage_and_exclusions(evaluator, main, ex, candidate_urls)
    await build_senior_discounts_and_limits(evaluator, main, ex, candidate_urls)
    await build_purchase_options_and_documentation(evaluator, main, ex, candidate_urls)
    await build_validity_and_replacement_policies(evaluator, main, ex, candidate_urls)
    await build_acceptance_and_scope(evaluator, main, ex, candidate_urls)

    # Return summary with verification tree and final score
    return evaluator.get_summary()