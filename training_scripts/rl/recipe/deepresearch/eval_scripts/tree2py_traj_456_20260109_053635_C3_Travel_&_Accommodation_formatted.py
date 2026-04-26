import asyncio
import logging
import re
from datetime import date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task constants and itinerary details
# -----------------------------------------------------------------------------
TASK_ID = "sea_passport_validity_2026_trip"
TASK_DESCRIPTION = (
    "I am a U.S. citizen planning a Southeast Asia trip with the following itinerary: I will depart from the United "
    "States on February 15, 2026, arrive in Bangkok, Thailand on February 16, 2026, stay in Thailand for 45 days, then "
    "travel to Bali, Indonesia on April 1, 2026, and stay in Indonesia for 14 days before returning to the United "
    "States on April 15, 2026. My U.S. passport expires on August 20, 2026. Based on the current passport validity "
    "requirements for U.S. citizens entering Thailand and Indonesia, determine whether my passport will be valid for "
    "this entire trip. Specifically, verify: (1) the minimum passport validity requirement for entering Thailand, (2) "
    "the maximum visa-exempt stay duration allowed in Thailand for U.S. citizens as of 2024, (3) the minimum passport "
    "validity requirement for entering Indonesia, and (4) whether my passport expiration date of August 20, 2026 "
    "satisfies all requirements for this itinerary. Provide reference URLs from official government sources (such as "
    "travel.state.gov or official embassy websites) that confirm these requirements."
)

# Itinerary dates (given in the task)
THAILAND_ARRIVAL = date(2026, 2, 16)
THAILAND_STAY_DAYS = 45
INDONESIA_ARRIVAL = date(2026, 4, 1)
INDONESIA_STAY_DAYS = 14
US_RETURN = date(2026, 4, 15)
PASSPORT_EXPIRY = date(2026, 8, 20)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def last_day_of_month(y: int, m: int) -> int:
    """Return last day for year y and month m."""
    if m in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if m in (4, 6, 9, 11):
        return 30
    # February (no leap-year complexity needed beyond this simple check)
    return 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28


def add_months(d: date, months: int) -> date:
    """Add months to a date, clamping day to the month's end if needed."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, last_day_of_month(y, m))
    return date(y, m, day)


def parse_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(\d{1,3})", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class ThailandRulesExtraction(BaseModel):
    # Passport validity rule info & sources
    th_passport_validity_rule_text: Optional[str] = None
    th_passport_validity_min_months: Optional[str] = None
    th_passport_validity_urls: List[str] = Field(default_factory=list)

    # Visa-exempt rule info & sources
    th_visa_exemption_duration_days: Optional[str] = None
    th_visa_extension_policy_text: Optional[str] = None
    th_visa_exemption_urls: List[str] = Field(default_factory=list)


class IndonesiaRulesExtraction(BaseModel):
    id_passport_validity_rule_text: Optional[str] = None
    id_passport_validity_min_months: Optional[str] = None
    id_passport_validity_urls: List[str] = Field(default_factory=list)
    id_enforcement_text: Optional[str] = None


class ConclusionExtraction(BaseModel):
    overall_conclusion_text: Optional[str] = None
    overall_conclusion_valid_for_entire_trip: Optional[bool] = None


# -----------------------------------------------------------------------------
# Extraction prompts
# -----------------------------------------------------------------------------
def prompt_extract_thailand_rules() -> str:
    return """
    Extract the Thailand entry and stay rules as explicitly stated in the answer and the official URLs the answer cited.

    Required fields:
    1) th_passport_validity_rule_text: The statement in the answer about the minimum passport validity required for entry to Thailand.
    2) th_passport_validity_min_months: The minimum months remaining validity mentioned (e.g., "6"). If unspecified, return null.
    3) th_passport_validity_urls: List all URLs the answer cites that confirm Thailand passport validity requirement. Include only URLs explicitly present in the answer.

    4) th_visa_exemption_duration_days: The maximum visa-exempt stay duration for U.S. citizens the answer claims (e.g., "60").
    5) th_visa_extension_policy_text: The text in the answer about whether/how a visa-exempt stay can be extended (e.g., "extendable by 30 days").
    6) th_visa_exemption_urls: List all URLs the answer cites that confirm the visa-exempt duration and/or extension policy. Include only URLs explicitly present in the answer.

    IMPORTANT:
    - Extract only what is actually present in the answer. Do not invent any URLs or numbers.
    - If the answer does not include a URL for a given item, return an empty list for that URL field.
    - Keep URLs exactly as they appear (full absolute URLs).
    """


def prompt_extract_indonesia_rules() -> str:
    return """
    Extract the Indonesia entry rule as explicitly stated in the answer and the official URLs the answer cited.

    Required fields:
    1) id_passport_validity_rule_text: The statement in the answer about the minimum passport validity required for entry to Indonesia.
    2) id_passport_validity_min_months: The minimum months remaining validity mentioned (e.g., "6"). If unspecified, return null.
    3) id_passport_validity_urls: List all URLs the answer cites that confirm Indonesia's passport validity requirement.
    4) id_enforcement_text: Any mention in the answer that this rule is strictly enforced (e.g., refusal of entry or airline boarding denial). If absent, return null.

    IMPORTANT:
    - Extract only what is explicitly present in the answer text.
    - If the answer does not include URLs, return an empty list for id_passport_validity_urls.
    """


def prompt_extract_overall_conclusion() -> str:
    return """
    Determine whether the answer provides a clear final determination about whether the passport will be valid for the entire trip.

    Required fields:
    1) overall_conclusion_text: The sentence(s) in the answer that give the final determination. If absent, return null.
    2) overall_conclusion_valid_for_entire_trip: A boolean:
       - true if the answer clearly states the passport IS valid for the entire trip,
       - false if the answer clearly states the passport is NOT valid for the entire trip,
       - null if the answer does not clearly state a final determination.

    Return null for fields that are not present.
    """


# -----------------------------------------------------------------------------
# Verification helpers
# -----------------------------------------------------------------------------
def official_site_claim() -> str:
    return (
        "This webpage is an official government or embassy/consulate website (e.g., travel.state.gov, state.gov, "
        "usembassy.gov, mfa.go.th, thailand.go.th, thaiembassy.org, go.th domains for Thailand; kemlu.go.id, "
        "imigrasi.go.id, indonesia.go.id, or a U.S. embassy/consulate domain for Indonesia)."
    )


def th_passport_rule_claim() -> str:
    return (
        "This webpage confirms that U.S. citizens must have a passport with at least six months validity remaining on "
        "arrival to Thailand (i.e., minimum 6 months at the time of entry)."
    )


def th_visa_rule_claim() -> str:
    return (
        "This webpage confirms that, as of 2024, U.S. citizens may enter Thailand visa-exempt for up to 60 days, and "
        "that the stay can be extended by about 30 days at an immigration office."
    )


def id_passport_rule_claim() -> str:
    return (
        "This webpage confirms that travelers (including U.S. citizens) must have a passport valid for at least six "
        "months from the date of arrival into Indonesia, and that this rule is strictly enforced (e.g., refusal of "
        "entry or airline boarding if less than six months)."
    )


# -----------------------------------------------------------------------------
# Verification subtrees
# -----------------------------------------------------------------------------
async def verify_thailand_requirements(
    evaluator: Evaluator,
    parent_node,
    th_rules: ThailandRulesExtraction,
):
    # Parent node for Thailand requirements
    th_node = evaluator.add_parallel(
        id="Thailand_Requirements",
        desc="Verify Thailand entry passport validity and visa-exempt stay rules for U.S. citizens (with official references).",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Thailand passport validity rule supported by official source(s)
    th_pass_rule_node = evaluator.add_leaf(
        id="Thailand_Passport_Validity_Rule",
        desc="States Thailand’s minimum passport validity requirement for entry (per constraints: >= 6 months beyond date of arrival).",
        parent=th_node,
        critical=True,
    )
    await evaluator.verify(
        claim=th_passport_rule_claim(),
        node=th_pass_rule_node,
        sources=th_rules.th_passport_validity_urls,
        additional_instruction="Accept equivalent wording such as '6 months validity on arrival'."
    )

    # Leaf: Official government/embassy URL for Thailand passport validity
    th_pass_official_node = evaluator.add_leaf(
        id="Thailand_Passport_Validity_Official_Source_URL",
        desc="Provides an official government/embassy URL confirming Thailand’s passport validity requirement.",
        parent=th_node,
        critical=True,
    )
    await evaluator.verify(
        claim=official_site_claim(),
        node=th_pass_official_node,
        sources=th_rules.th_passport_validity_urls,
        additional_instruction="Judge official status by domain and page identity (logos, headers). If the list is empty or non-official, mark as not supported."
    )

    # Leaf: Thailand visa-exempt duration and extension policy (as of 2024)
    th_visa_rule_node = evaluator.add_leaf(
        id="Thailand_Visa_Exemption_Duration_And_Extension_Rule",
        desc="States Thailand’s visa-exempt stay duration rule for U.S. citizens as of 2024 (per constraints: up to 60 days) and notes the extension policy (per constraints: extendable by an additional 30 days).",
        parent=th_node,
        critical=True,
    )
    await evaluator.verify(
        claim=th_visa_rule_claim(),
        node=th_visa_rule_node,
        sources=th_rules.th_visa_exemption_urls,
        additional_instruction="Do not require the exact phrase 'as of 2024' on-page; confirm that pages reflect 60 days visa exemption and 30 days extension."
    )

    # Leaf: Official government/embassy URL for Thailand visa exemption/extension policy
    th_visa_official_node = evaluator.add_leaf(
        id="Thailand_Visa_Exemption_Official_Source_URL",
        desc="Provides an official government/embassy URL confirming Thailand’s visa-exempt stay duration and/or extension policy.",
        parent=th_node,
        critical=True,
    )
    await evaluator.verify(
        claim=official_site_claim(),
        node=th_visa_official_node,
        sources=th_rules.th_visa_exemption_urls,
        additional_instruction="Official sites include travel.state.gov, mfa.go.th, or relevant embassies/consulates."
    )


async def verify_indonesia_requirements(
    evaluator: Evaluator,
    parent_node,
    id_rules: IndonesiaRulesExtraction,
):
    id_node = evaluator.add_parallel(
        id="Indonesia_Requirements",
        desc="Verify Indonesia entry passport validity rule for U.S. citizens (with official references).",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Indonesia passport validity rule supported by official source(s)
    id_pass_rule_node = evaluator.add_leaf(
        id="Indonesia_Passport_Validity_Rule",
        desc="States Indonesia’s minimum passport validity requirement for entry (per constraints: >= 6 months beyond date of arrival) and that it is strictly enforced.",
        parent=id_node,
        critical=True,
    )
    await evaluator.verify(
        claim=id_passport_rule_claim(),
        node=id_pass_rule_node,
        sources=id_rules.id_passport_validity_urls,
        additional_instruction="Accept equivalent wording such as 'passport valid at least six months on entry'; enforcement may be described implicitly (e.g., refusal/denied boarding)."
    )

    # Leaf: Official government/embassy URL for Indonesia passport validity
    id_pass_official_node = evaluator.add_leaf(
        id="Indonesia_Passport_Validity_Official_Source_URL",
        desc="Provides an official government/embassy URL confirming Indonesia’s passport validity requirement (and strict enforcement if stated).",
        parent=id_node,
        critical=True,
    )
    await evaluator.verify(
        claim=official_site_claim(),
        node=id_pass_official_node,
        sources=id_rules.id_passport_validity_urls,
        additional_instruction="Official Indonesian sites include kemlu.go.id, imigrasi.go.id; U.S. official sites include travel.state.gov or U.S. embassy domains."
    )


async def verify_itinerary_compliance(
    evaluator: Evaluator,
    parent_node,
    th_rules: ThailandRulesExtraction,
    id_rules: IndonesiaRulesExtraction,
    conc: ConclusionExtraction
):
    comp_node = evaluator.add_parallel(
        id="Itinerary_Compliance_With_Stated_Rules",
        desc="Determines whether the specific itinerary dates and planned stays comply with the stated Thailand/Indonesia rules (and other listed constraints).",
        parent=parent_node,
        critical=True,
    )

    # Thailand arrival passport compliance: expiry >= arrival + 6 months
    th_arrival_threshold = add_months(THAILAND_ARRIVAL, 6)
    th_arrival_ok = PASSPORT_EXPIRY >= th_arrival_threshold
    evaluator.add_custom_node(
        result=th_arrival_ok,
        id="Thailand_Arrival_Passport_Compliance",
        desc=f"Checks whether the passport expiration date is at least 6 months beyond Thailand arrival (arrival: {THAILAND_ARRIVAL.isoformat()}, +6 months: {th_arrival_threshold.isoformat()}, expiry: {PASSPORT_EXPIRY.isoformat()}).",
        parent=comp_node,
        critical=True,
    )

    # Thailand stay duration compliance using stated (or default assumed) allowance
    # Prefer values extracted from the answer; if absent, assume 60 days base and 30 days extension as per constraints.
    base_days = parse_first_int(th_rules.th_visa_exemption_duration_days) or 60
    ext_days = parse_first_int(th_rules.th_visa_extension_policy_text) or 30
    th_stay_ok = (THAILAND_STAY_DAYS <= base_days) or (THAILAND_STAY_DAYS <= (base_days + ext_days))
    extension_needed = THAILAND_STAY_DAYS > base_days and THAILAND_STAY_DAYS <= (base_days + ext_days)
    evaluator.add_custom_node(
        result=th_stay_ok,
        id="Thailand_Stay_Duration_Compliance",
        desc=(
            f"Checks whether the planned Thailand stay length ({THAILAND_STAY_DAYS} days) fits within the visa-exempt "
            f"allowance as stated/assumed (base {base_days} days, extension {ext_days} days). "
            f"Extension needed: {'Yes' if extension_needed else 'No'}."
        ),
        parent=comp_node,
        critical=True,
    )

    # Indonesia arrival passport compliance: expiry >= arrival + 6 months
    id_arrival_threshold = add_months(INDONESIA_ARRIVAL, 6)
    id_arrival_ok = PASSPORT_EXPIRY >= id_arrival_threshold
    evaluator.add_custom_node(
        result=id_arrival_ok,
        id="Indonesia_Arrival_Passport_Compliance",
        desc=f"Checks whether the passport expiration date is at least 6 months beyond Indonesia arrival (arrival: {INDONESIA_ARRIVAL.isoformat()}, +6 months: {id_arrival_threshold.isoformat()}, expiry: {PASSPORT_EXPIRY.isoformat()}).",
        parent=comp_node,
        critical=True,
    )

    # U.S. return passport compliance per listed constraint (6 months beyond period of stay in the U.S.)
    us_return_threshold = add_months(US_RETURN, 6)
    us_return_ok = PASSPORT_EXPIRY >= us_return_threshold
    evaluator.add_custom_node(
        result=us_return_ok,
        id="US_Return_Passport_Compliance_Per_Constraints",
        desc=f"Addresses the listed constraint: passport valid 6 months beyond U.S. return (return: {US_RETURN.isoformat()}, +6 months: {us_return_threshold.isoformat()}, expiry: {PASSPORT_EXPIRY.isoformat()}).",
        parent=comp_node,
        critical=True,
    )

    # Overall conclusion for the entire trip must be consistent with the above checks
    expected_valid_for_entire_trip = th_arrival_ok and th_stay_ok and id_arrival_ok and us_return_ok
    ans_conclusion: Optional[bool] = conc.overall_conclusion_valid_for_entire_trip

    evaluator.add_custom_info(
        info={
            "computed_checks": {
                "thailand_arrival_ok": th_arrival_ok,
                "thailand_stay_ok": th_stay_ok,
                "thailand_extension_needed": extension_needed,
                "indonesia_arrival_ok": id_arrival_ok,
                "us_return_ok": us_return_ok,
            },
            "expected_valid_for_entire_trip": expected_valid_for_entire_trip,
            "answer_conclusion_bool": ans_conclusion,
        },
        info_type="computation_summary",
        info_name="itinerary_computation"
    )

    evaluator.add_custom_node(
        result=(ans_conclusion is not None and ans_conclusion == expected_valid_for_entire_trip),
        id="Overall_Conclusion_For_Entire_Trip",
        desc=(
            "Provides a final determination (valid/not valid for the entire trip) consistent with all prior compliance "
            f"checks. Expected: {'valid' if expected_valid_for_entire_trip else 'not valid'}. "
            f"Answer states: {'valid' if ans_conclusion else ('not valid' if ans_conclusion is False else 'unspecified')}."
        ),
        parent=comp_node,
        critical=True,
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Southeast Asia passport validity and itinerary compliance task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # top-level aggregation
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

    # Create the top-level critical parent node per rubric
    top_node = evaluator.add_parallel(
        id="Travel_Feasibility_Assessment",
        desc="Evaluate Thailand and Indonesia entry/stay requirements (with official references) and determine whether the given passport expiration satisfies the full itinerary.",
        parent=root,
        critical=True,
    )

    # Extract information from the answer (in parallel)
    th_rules_task = evaluator.extract(
        prompt=prompt_extract_thailand_rules(),
        template_class=ThailandRulesExtraction,
        extraction_name="thailand_rules",
    )
    id_rules_task = evaluator.extract(
        prompt=prompt_extract_indonesia_rules(),
        template_class=IndonesiaRulesExtraction,
        extraction_name="indonesia_rules",
    )
    conc_task = evaluator.extract(
        prompt=prompt_extract_overall_conclusion(),
        template_class=ConclusionExtraction,
        extraction_name="overall_conclusion",
    )

    th_rules, id_rules, conc = await asyncio.gather(th_rules_task, id_rules_task, conc_task)

    # Build and verify subtrees
    await verify_thailand_requirements(evaluator, top_node, th_rules)
    await verify_indonesia_requirements(evaluator, top_node, id_rules)
    await verify_itinerary_compliance(evaluator, top_node, th_rules, id_rules, conc)

    # Add ground truth info (computed expectations for transparency)
    evaluator.add_ground_truth({
        "itinerary": {
            "thailand_arrival": THAILAND_ARRIVAL.isoformat(),
            "thailand_stay_days": THAILAND_STAY_DAYS,
            "indonesia_arrival": INDONESIA_ARRIVAL.isoformat(),
            "indonesia_stay_days": INDONESIA_STAY_DAYS,
            "us_return": US_RETURN.isoformat(),
            "passport_expiry": PASSPORT_EXPIRY.isoformat(),
            "six_months_after_thailand_arrival": add_months(THAILAND_ARRIVAL, 6).isoformat(),
            "six_months_after_indonesia_arrival": add_months(INDONESIA_ARRIVAL, 6).isoformat(),
            "six_months_after_us_return": add_months(US_RETURN, 6).isoformat(),
        },
        "assumptions_for_compliance": {
            "thailand_visa_exempt_base_days_default_if_missing": 60,
            "thailand_extension_days_default_if_missing": 30
        }
    })

    return evaluator.get_summary()