import asyncio
import logging
from datetime import date, timedelta
import calendar
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "angola_entry_requirements_2026"
TASK_DESCRIPTION = """
A US citizen is planning to travel to Angola for tourism on April 15, 2026. Their current US passport expires on February 20, 2027.

Based on Angola's entry requirements for US citizens:
1. Verify whether the traveler's passport meets Angola's validity requirement for entry on the planned date.
2. Confirm whether Angola requires proof of yellow fever vaccination for US citizens arriving for tourism.
3. Calculate the latest date by which the traveler must receive the yellow fever vaccine to ensure the vaccination certificate is valid for their April 15, 2026 travel date.

For each point, provide:
- The specific requirement or rule
- The calculation or verification performed
- The conclusion
- Reference URLs from official sources that confirm the requirements
"""

# Ground truth parameters for this evaluation
ENTRY_DATE = date(2026, 4, 15)
PASSPORT_EXPIRATION = date(2027, 2, 20)
MONTHS_REQUIRED = 9
YF_VALIDITY_DELAY_DAYS = 10


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def add_months(d: date, months: int) -> date:
    """Add months to a date, clamping the day to the last day of the target month if needed."""
    month_index = d.month - 1 + months
    new_year = d.year + month_index // 12
    new_month = (month_index % 12) + 1
    new_day = min(d.day, _last_day_of_month(new_year, new_month))
    return date(new_year, new_month, new_day)


def ymd(dt: date) -> str:
    return dt.strftime("%Y-%m-%d")


EXPECTED_MIN_VALID_UNTIL = add_months(ENTRY_DATE, MONTHS_REQUIRED)  # 2027-01-15
EXPECTED_MIN_VALID_UNTIL_STR = ymd(EXPECTED_MIN_VALID_UNTIL)
EXPECTED_LATEST_YF_VAX_DATE = ENTRY_DATE - timedelta(days=YF_VALIDITY_DELAY_DAYS)  # 2026-04-05
EXPECTED_LATEST_YF_VAX_DATE_STR = ymd(EXPECTED_LATEST_YF_VAX_DATE)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PassportValidityInfo(BaseModel):
    rule_text: Optional[str] = None
    rule_urls: List[str] = Field(default_factory=list)
    calculation_text: Optional[str] = None
    min_valid_until_date: Optional[str] = None  # Prefer ISO format 'YYYY-MM-DD' if provided
    conclusion_text: Optional[str] = None       # e.g., "meets"/"does not meet" or equivalent


class YellowFeverRequirementInfo(BaseModel):
    rule_text: Optional[str] = None
    rule_urls: List[str] = Field(default_factory=list)
    conclusion_text: Optional[str] = None       # e.g., "yes, required"/"not required"


class YellowFeverTimingInfo(BaseModel):
    timing_rule_text: Optional[str] = None
    rule_urls: List[str] = Field(default_factory=list)
    latest_vaccination_date: Optional[str] = None  # Prefer ISO 'YYYY-MM-DD' if provided
    conclusion_text: Optional[str] = None          # e.g., "vaccinate by 2026-04-05 ensures validity"


class AngolaTravelExtraction(BaseModel):
    passport_validity: Optional[PassportValidityInfo] = None
    yellow_fever_requirement: Optional[YellowFeverRequirementInfo] = None
    yellow_fever_timing: Optional[YellowFeverTimingInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_angola_travel() -> str:
    return f"""
    Extract the specific content the answer provided for each of the three checks. Return a JSON strictly following this schema:

    passport_validity:
      - rule_text: The exact rule text or paraphrase the answer states for Angola passport validity (e.g., "valid for at least 9 months from entry"). If absent, null.
      - rule_urls: All URLs the answer cites to support the passport validity rule (array of URLs). If none, [].
      - calculation_text: The text showing how the answer verified/calculated the rule against the provided dates. If absent, null.
      - min_valid_until_date: If the answer computed a specific minimum acceptable validity end date (entry date + required months), extract it as a date string (prefer ISO 'YYYY-MM-DD' if the answer gives a recognizable date). If not explicitly given, null.
      - conclusion_text: The answer's explicit conclusion on whether the passport meets/does not meet the requirement. If absent, null.

    yellow_fever_requirement:
      - rule_text: The exact rule text or paraphrase the answer states for Angola's yellow fever entry requirement (e.g., "required for all travelers aged 9 months or older"). If absent, null.
      - rule_urls: All URLs the answer cites to support this rule (array). If none, [].
      - conclusion_text: The answer's explicit yes/no conclusion on whether the US tourist must present proof of yellow fever vaccination. If absent, null.

    yellow_fever_timing:
      - timing_rule_text: The timing rule text (e.g., "certificate becomes valid 10 days after vaccination"). If absent, null.
      - rule_urls: All URLs the answer cites to support the 10-day timing rule (array). If none, [].
      - latest_vaccination_date: The latest vaccination date the answer gives to ensure validity on 2026-04-15 (prefer 'YYYY-MM-DD' if recognizable). If not explicitly provided, null.
      - conclusion_text: The answer's explicit conclusion that vaccinating by that latest date ensures the certificate is valid on arrival. If absent, null.

    IMPORTANT:
    - Extract only what is explicitly present in the answer. Do not invent or add anything.
    - Parse URLs in any format (plain or markdown). Return only valid absolute URLs including protocol.
    - If any field is missing in the answer, set it to null (or [] for the URL arrays).
    """


# --------------------------------------------------------------------------- #
# Verification helpers (subtrees)                                             #
# --------------------------------------------------------------------------- #
async def verify_passport_validity_section(
    evaluator: Evaluator,
    parent_node,
    extracted: AngolaTravelExtraction
) -> None:
    # Create main sequential node (critical as per rubric)
    node = evaluator.add_sequential(
        id="1_Passport_Validity_On_Entry_Date",
        desc="Check Angola passport validity requirement and whether the given passport expiration satisfies it for entry on 2026-04-15.",
        parent=parent_node,
        critical=True
    )

    info = extracted.passport_validity or PassportValidityInfo()

    # 1A - States 9-month rule
    leaf_1a = evaluator.add_leaf(
        id="1A_State_Passport_Validity_Rule_9_Months",
        desc="States the passport validity rule as: passport must be valid for at least 9 months from the date of entry.",
        parent=node,
        critical=True
    )
    stated_rule = info.rule_text or ""
    claim_1a = (
        "The answer explicitly states that Angola requires a passport to be valid for at least nine months from "
        "the date of entry (≥ 9 months from arrival), or an equivalent clear paraphrase."
    )
    await evaluator.verify(
        claim=claim_1a + (f" The answer's stated rule text is: '{stated_rule}'." if stated_rule else ""),
        node=leaf_1a,
        additional_instruction="Judge only based on the provided answer. Accept minor paraphrases that clearly mean ≥9 months from entry. If the timeframe differs (e.g., 6 months), mark incorrect."
    )

    # 1B - Shows calculation and comparison
    leaf_1b = evaluator.add_leaf(
        id="1B_Show_Verification_Calculation_And_Comparison",
        desc="Shows calculation: entry date + 9 months = 2027-01-15 and compares to passport expiration 2027-02-20.",
        parent=node,
        critical=True
    )
    calc_text = info.calculation_text or ""
    min_until = info.min_valid_until_date or ""
    claim_1b = (
        f"The answer explicitly shows the verification by computing the minimum acceptable validity end date as "
        f"2026-04-15 + 9 months = {EXPECTED_MIN_VALID_UNTIL_STR} and comparing it to the passport expiration "
        f"{ymd(PASSPORT_EXPIRATION)}."
    )
    await evaluator.verify(
        claim=claim_1b + (f" The answer's calculation text is: '{calc_text}'. The answer's computed date (if any) is '{min_until}'." if calc_text or min_until else ""),
        node=leaf_1b,
        additional_instruction="Accept equivalent reasoning (e.g., showing that the expiration is ≥9 months beyond entry). Small date-format variations are fine; the substance must match 2027-01-15 as the threshold and that 2027-02-20 is later."
    )

    # 1C - Conclusion (meets)
    leaf_1c = evaluator.add_leaf(
        id="1C_Passport_Validity_Conclusion_For_This_Passport",
        desc="Provides a clear conclusion whether a passport expiring 2027-02-20 satisfies the ≥9-months-from-2026-04-15 requirement.",
        parent=node,
        critical=True
    )
    conclusion_text = info.conclusion_text or ""
    expected_conclusion = (
        PASSPORT_EXPIRATION >= EXPECTED_MIN_VALID_UNTIL
    )  # True for this scenario
    claim_1c = (
        "The answer clearly concludes that this passport expiring on 2027-02-20 meets the requirement of being valid "
        "for at least nine months from the 2026-04-15 entry date."
    )
    await evaluator.verify(
        claim=claim_1c + (f" The answer's conclusion text is: '{conclusion_text}'." if conclusion_text else ""),
        node=leaf_1c,
        additional_instruction="Accept synonyms like 'meets', 'satisfies', 'sufficient'. If it says it does not meet, mark incorrect."
    )

    # 1D - Official source URLs existence
    urls = list(info.rule_urls or [])
    urls_present = len(urls) > 0
    leaf_1d_present = evaluator.add_custom_node(
        result=urls_present,
        id="1D_Official_Source_URLs_For_Passport_Rule_present",
        desc="Includes at least one reference URL to support the passport validity rule.",
        parent=node,
        critical=True
    )

    # 1D - Official source URLs support (verify by URLs)
    leaf_1d_support = evaluator.add_leaf(
        id="1D_Official_Source_URLs_For_Passport_Rule",
        desc="Official source URL(s) explicitly support the '≥9 months from entry' passport validity rule.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Angola requires that passports be valid for at least nine months from the date of entry.",
        node=leaf_1d_support,
        sources=urls,  # This will be skipped if the 'present' node failed due to precondition logic
        additional_instruction=(
            "Only mark as supported if at least one of the provided URLs is an official source and explicitly states "
            "the nine-month-from-entry rule. Treat as official: government/embassy/consulate (.gov, gov.* domains, "
            "official .ao government sites), or U.S. State Department (travel.state.gov). If sources are non-official "
            "blogs or do not explicitly mention 9 months, mark as not supported."
        )
    )


async def verify_yf_requirement_section(
    evaluator: Evaluator,
    parent_node,
    extracted: AngolaTravelExtraction
) -> None:
    node = evaluator.add_sequential(
        id="2_Yellow_Fever_Vaccination_Requirement",
        desc="Check whether Angola requires proof of yellow fever vaccination for US citizens arriving for tourism.",
        parent=parent_node,
        critical=True
    )

    info = extracted.yellow_fever_requirement or YellowFeverRequirementInfo()

    # 2A - States YF rule (≥ 9 months old must present certificate)
    leaf_2a = evaluator.add_leaf(
        id="2A_State_Yellow_Fever_Entry_Rule_Age_Threshold",
        desc="States: valid yellow fever vaccination certificate required for all arriving travelers aged 9 months or older.",
        parent=node,
        critical=True
    )
    rule_text = info.rule_text or ""
    claim_2a = (
        "The answer explicitly states that Angola requires a valid yellow fever vaccination certificate for all "
        "arriving travelers aged 9 months or older (or an equivalent clear paraphrase)."
    )
    await evaluator.verify(
        claim=claim_2a + (f" The answer's stated rule text is: '{rule_text}'." if rule_text else ""),
        node=leaf_2a,
        additional_instruction="Judge only from the answer. Accept clear paraphrases. If the rule is narrower or absent, mark incorrect."
    )

    # 2B - Conclude applicability to US tourist (yes)
    leaf_2b = evaluator.add_leaf(
        id="2B_Conclude_Applicability_To_US_Tourist",
        desc="Provides a clear yes/no conclusion on whether a US tourist must present proof (consistent with 'all travelers ≥9 months' rule).",
        parent=node,
        critical=True
    )
    concl_2 = info.conclusion_text or ""
    claim_2b = "The answer clearly concludes that a US citizen arriving for tourism must present proof of yellow fever vaccination."
    await evaluator.verify(
        claim=claim_2b + (f" The answer's conclusion text is: '{concl_2}'." if concl_2 else ""),
        node=leaf_2b,
        additional_instruction="A clear 'yes' (or equivalent) is expected for this scenario."
    )

    # 2C - Official source URLs existence
    urls = list(info.rule_urls or [])
    leaf_2c_present = evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="2C_Official_Source_URLs_For_Yellow_Fever_Rule_present",
        desc="Includes at least one reference URL to support the yellow fever entry requirement rule.",
        parent=node,
        critical=True
    )

    # 2C - Official source URLs support
    leaf_2c_support = evaluator.add_leaf(
        id="2C_Official_Source_URLs_For_Yellow_Fever_Rule",
        desc="Official source URL(s) explicitly support that Angola requires proof of yellow fever vaccination for travelers aged ≥9 months.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Angola requires that travelers aged 9 months or older present a valid yellow fever vaccination certificate upon entry.",
        node=leaf_2c_support,
        sources=urls,
        additional_instruction=(
            "Only accept support from official sources: e.g., CDC (cdc.gov), WHO (who.int), Angola government/embassy/consulate, "
            "or U.S. State Department (travel.state.gov). The page must explicitly state the requirement; otherwise mark as not supported."
        )
    )


async def verify_yf_timing_section(
    evaluator: Evaluator,
    parent_node,
    extracted: AngolaTravelExtraction
) -> None:
    node = evaluator.add_sequential(
        id="3_Latest_Yellow_Fever_Vaccination_Date",
        desc="Compute the latest date to receive yellow fever vaccine so the certificate is valid for travel on 2026-04-15.",
        parent=parent_node,
        critical=True
    )

    info = extracted.yellow_fever_timing or YellowFeverTimingInfo()

    # 3A - States timing rule: valid after 10 days
    leaf_3a = evaluator.add_leaf(
        id="3A_State_Certificate_Becomes_Valid_After_10_Days",
        desc="States: the yellow fever vaccination certificate becomes valid 10 days after the vaccination date.",
        parent=node,
        critical=True
    )
    timing_text = info.timing_rule_text or ""
    claim_3a = "The answer explicitly states that the yellow fever vaccination certificate becomes valid 10 days after vaccination."
    await evaluator.verify(
        claim=claim_3a + (f" The answer's stated timing rule is: '{timing_text}'." if timing_text else ""),
        node=leaf_3a,
        additional_instruction="Judge only from the answer. Accept equivalent phrasing such as 'effective 10 days after vaccination'."
    )

    # 3B - Calculates latest vaccination date (2026-04-05)
    leaf_3b = evaluator.add_leaf(
        id="3B_Calculate_Latest_Vaccination_Date_For_2026_04_15",
        desc="Correctly calculates latest vaccination date as travel date minus 10 days: 2026-04-05.",
        parent=node,
        critical=True
    )
    latest_in_ans = info.latest_vaccination_date or ""
    claim_3b = (
        f"The answer correctly calculates the latest vaccination date for a 2026-04-15 trip as "
        f"{EXPECTED_LATEST_YF_VAX_DATE_STR} (10 days before travel)."
    )
    await evaluator.verify(
        claim=claim_3b + (f" The answer's latest date is: '{latest_in_ans}'." if latest_in_ans else ""),
        node=leaf_3b,
        additional_instruction="Accept if the answer states 'on or before 2026-04-05' or an equivalent correct computation. Minor format differences are fine."
    )

    # 3C - Clear conclusion about validity on arrival
    leaf_3c = evaluator.add_leaf(
        id="3C_Latest_Vaccination_Date_Conclusion",
        desc="States the latest vaccination date and concludes that vaccinating by that date ensures the certificate is valid on arrival.",
        parent=node,
        critical=True
    )
    concl_3 = info.conclusion_text or ""
    claim_3c = (
        f"The answer clearly concludes that vaccinating by {EXPECTED_LATEST_YF_VAX_DATE_STR} ensures the certificate is valid on arrival on 2026-04-15."
    )
    await evaluator.verify(
        claim=claim_3c + (f" The answer's conclusion text is: '{concl_3}'." if concl_3 else ""),
        node=leaf_3c,
        additional_instruction="Look for an explicit statement tying the latest date to validity on arrival."
    )

    # 3D - Official source URLs existence
    urls = list(info.rule_urls or [])
    leaf_3d_present = evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="3D_Official_Source_URLs_For_Certificate_Timing_present",
        desc="Includes at least one reference URL to support the 'valid after 10 days' timing rule.",
        parent=node,
        critical=True
    )

    # 3D - Official source URLs support (WHO/CDC/IHR)
    leaf_3d_support = evaluator.add_leaf(
        id="3D_Official_Source_URLs_For_Certificate_Timing",
        desc="Official source URL(s) explicitly support that the yellow fever certificate becomes valid 10 days after vaccination.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="A yellow fever vaccination certificate becomes valid 10 days after vaccination.",
        node=leaf_3d_support,
        sources=urls,
        additional_instruction=(
            "Only accept support from official public health or governmental sources (e.g., WHO who.int, CDC cdc.gov, "
            "official government/embassy/consulate sites, or the IHR). The page must explicitly state the 10-day rule."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for Angola entry requirements (passport validity, yellow fever requirement, latest vaccination date).
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # As specified by rubric
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
        prompt=prompt_extract_angola_travel(),
        template_class=AngolaTravelExtraction,
        extraction_name="angola_travel_requirements_extraction"
    )

    # Add ground-truth computation details for transparency
    evaluator.add_ground_truth({
        "entry_date": ymd(ENTRY_DATE),
        "passport_expiration": ymd(PASSPORT_EXPIRATION),
        "months_required_from_entry": MONTHS_REQUIRED,
        "expected_min_valid_until_date": EXPECTED_MIN_VALID_UNTIL_STR,
        "yellow_fever_validity_delay_days": YF_VALIDITY_DELAY_DAYS,
        "expected_latest_yf_vaccination_date": EXPECTED_LATEST_YF_VAX_DATE_STR
    }, gt_type="computed_expectations")

    # Build main critical node per rubric
    main_node = evaluator.add_parallel(
        id="Angola_Travel_Requirements_Verification",
        desc="Evaluate the three requested Angola entry requirement checks with rules, calculations, conclusions, and official sources.",
        parent=root,
        critical=True
    )

    # Verify each section (all critical under the main critical node)
    await verify_passport_validity_section(evaluator, main_node, extracted)
    await verify_yf_requirement_section(evaluator, main_node, extracted)
    await verify_yf_timing_section(evaluator, main_node, extracted)

    # Return the summary including the verification tree and scores
    return evaluator.get_summary()