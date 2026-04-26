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
TASK_ID = "irs_2026_filing_season"
TASK_DESCRIPTION = """The IRS has announced details for the 2026 tax filing season. Provide comprehensive information covering the following aspects, with each answer supported by an official IRS or government source URL:

1. Key Filing Dates: What is the official opening date when the IRS will begin accepting 2025 tax returns? What is the filing deadline for 2025 returns? When does the IRS Free File program begin accepting returns? What is the extended deadline for taxpayers who request an extension by the April deadline?

2. Free File Program: What is the adjusted gross income (AGI) eligibility limit for the IRS Free File program? How many days before the general filing season does the Free File program start accepting returns?

3. New Changes: What new Schedule form has been introduced for claiming new deductions? What change has been announced regarding paper refund checks? Approximately how many individual income tax returns does the IRS expect to receive?

4. Standard Deductions: What is the standard deduction amount for single filers in 2026? What is the standard deduction amount for married couples filing jointly in 2026?
"""

# Ground-truth targets from rubric (used for claim phrasing)
OPENING_DATE_STR = "January 26, 2026"
OPENING_DATE_WITH_WEEKDAY = "Monday, January 26, 2026"
FILING_DEADLINE_STR = "April 15, 2026"
FILING_DEADLINE_WITH_WEEKDAY = "Wednesday, April 15, 2026"
FREE_FILE_START_STR = "January 9, 2026"
FREE_FILE_START_WITH_WEEKDAY = "Friday, January 9, 2026"
EXTENSION_DEADLINE_STR = "October 15, 2026"
FREE_FILE_AGI_LIMIT = "$84,000 or less"
FREE_FILE_LEAD_TIME_DAYS = 17
NEW_SCHEDULE_FORM = "Schedule 1-A"
EXPECTED_RETURN_VOLUME_APPROX = "approximately 164 million"
STD_DED_SINGLE = "$16,100"
STD_DED_MFJ = "$32,200"


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class KeyDates(BaseModel):
    season_opening_date: Optional[str] = None
    filing_deadline: Optional[str] = None
    free_file_start_date: Optional[str] = None
    extension_deadline: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class FreeFileDetails(BaseModel):
    agi_eligibility_limit: Optional[str] = None
    lead_time_days: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class NewChanges(BaseModel):
    new_schedule_form: Optional[str] = None
    refund_policy_change: Optional[str] = None
    expected_return_volume: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class StandardDeductions(BaseModel):
    single_filer: Optional[str] = None
    married_joint: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class IRS2026Extraction(BaseModel):
    key_dates: Optional[KeyDates] = None
    free_file: Optional[FreeFileDetails] = None
    new_changes: Optional[NewChanges] = None
    standard_deductions: Optional[StandardDeductions] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_irs_2026() -> str:
    return """
Extract structured information exactly as stated in the provided answer text, without adding or inferring new facts.

Return a JSON with the following top-level fields: key_dates, free_file, new_changes, standard_deductions.

1) key_dates:
   - season_opening_date: The date the IRS begins accepting 2025 returns, exactly as written in the answer (include weekday if present; otherwise as-is).
   - filing_deadline: The filing deadline for 2025 returns, exactly as written in the answer.
   - free_file_start_date: The date the IRS Free File program begins accepting returns, exactly as written in the answer.
   - extension_deadline: The extended deadline for taxpayers who request an extension by the April deadline, exactly as written in the answer.
   - urls: Array of all source URLs cited in the answer specifically supporting key filing dates. Include every URL format you see (plain or markdown). Do not invent URLs.

2) free_file:
   - agi_eligibility_limit: The AGI eligibility limit for IRS Free File, exactly as written (e.g., "$84,000 or less").
   - lead_time_days: The number of days before the general filing season that Free File starts, as explicitly stated in the answer (if provided). Do NOT compute this; only extract if the number is explicitly stated (e.g., "17 days").
   - urls: Array of all source URLs cited in the answer specifically supporting Free File info.

3) new_changes:
   - new_schedule_form: The new Schedule form introduced for claiming new deductions (as written, e.g., "Schedule 1-A").
   - refund_policy_change: The described change regarding paper refund checks (use the answer's own phrasing).
   - expected_return_volume: The expected number of individual income tax returns (e.g., "approximately 164 million").
   - urls: Array of all source URLs cited in the answer supporting these changes.

4) standard_deductions:
   - single_filer: The 2026 standard deduction amount for single filers as stated (e.g., "$16,100").
   - married_joint: The 2026 standard deduction for married filing jointly as stated (e.g., "$32,200").
   - urls: Array of all source URLs cited in the answer supporting the standard deduction amounts.

General rules:
- If a field is not present in the answer, set it to null (or [] for urls).
- Do not normalize or compute values. Use the exact text from the answer (including currency symbols or commas).
- Extract only URLs explicitly present in the answer.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _urls_or_empty(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_key_filing_dates_subtree(evaluator: Evaluator, parent_node, extracted: IRS2026Extraction) -> None:
    node = evaluator.add_parallel(
        id="Key_Filing_Dates",
        desc="Provide the required key filing dates for the 2026 filing season.",
        parent=parent_node,
        critical=True
    )
    urls = _urls_or_empty(extracted.key_dates.urls if extracted.key_dates else [])

    # Season Opening Date
    leaf_open = evaluator.add_leaf(
        id="Season_Opening_Date",
        desc="States the IRS season opening date (when IRS begins accepting 2025 returns) as Monday, January 26, 2026.",
        parent=node,
        critical=True
    )
    claim_open = (
        f"The answer states that the IRS will begin accepting 2025 returns on {OPENING_DATE_WITH_WEEKDAY} "
        f"(accept also phrasing that simply says {OPENING_DATE_STR} without the weekday)."
    )
    await evaluator.verify(
        claim=claim_open,
        node=leaf_open,
        additional_instruction="Focus only on whether the answer text contains that date (weekday optional). Minor phrasing variants are acceptable."
    )

    # Filing Deadline
    leaf_deadline = evaluator.add_leaf(
        id="Filing_Deadline",
        desc="States the filing deadline for 2025 returns as Wednesday, April 15, 2026.",
        parent=node,
        critical=True
    )
    claim_deadline = (
        f"The answer states that the filing deadline for 2025 tax returns is {FILING_DEADLINE_WITH_WEEKDAY} "
        f"(accept {FILING_DEADLINE_STR} even if weekday is omitted)."
    )
    await evaluator.verify(
        claim=claim_deadline,
        node=leaf_deadline,
        additional_instruction="Allow the weekday to be omitted; ensure the date April 15, 2026 is clearly stated as the general filing deadline."
    )

    # Free File Start Date
    leaf_freefile_start = evaluator.add_leaf(
        id="Free_File_Start_Date",
        desc="States the IRS Free File program start date as Friday, January 9, 2026.",
        parent=node,
        critical=True
    )
    claim_ff_start = (
        f"The answer states that the IRS Free File program begins accepting returns on {FREE_FILE_START_WITH_WEEKDAY} "
        f"(accept {FREE_FILE_START_STR} without the weekday)."
    )
    await evaluator.verify(
        claim=claim_ff_start,
        node=leaf_freefile_start,
        additional_instruction="Check the answer text for that date; small wording variations are acceptable."
    )

    # Extension Deadline
    leaf_extension = evaluator.add_leaf(
        id="Extension_Deadline",
        desc="States the extended deadline (if extension requested by April 15) as October 15, 2026.",
        parent=node,
        critical=True
    )
    claim_extension = f"The answer states that the extended deadline for timely filed extensions is {EXTENSION_DEADLINE_STR}."
    await evaluator.verify(
        claim=claim_extension,
        node=leaf_extension,
        additional_instruction="The answer should clearly indicate October 15, 2026 as the extension deadline; minor phrasing variants acceptable."
    )

    # Source URL support (official + supportive)
    leaf_key_sources = evaluator.add_leaf(
        id="Key_Dates_Source_URL",
        desc="Provides at least one official IRS or other U.S. government URL supporting the key filing dates stated.",
        parent=node,
        critical=True
    )
    claim_key_sources = (
        "At least one of these URLs is an official IRS or other U.S. government webpage (.gov) and it provides "
        "key filing dates for the 2026 season, such as the opening date (January 26, 2026), the general filing "
        "deadline (April 15, 2026), the Free File start date (around January 9, 2026), or the extension deadline "
        "(October 15, 2026)."
    )
    await evaluator.verify(
        claim=claim_key_sources,
        node=leaf_key_sources,
        sources=urls,
        additional_instruction="Confirm the domain is .gov (e.g., irs.gov) and that the page contains any of the listed key dates accurately. If any one URL meets this, pass."
    )


async def build_free_file_program_subtree(evaluator: Evaluator, parent_node, extracted: IRS2026Extraction) -> None:
    node = evaluator.add_parallel(
        id="Free_File_Program_Details",
        desc="Provide required IRS Free File eligibility and timing details.",
        parent=parent_node,
        critical=True
    )
    urls = _urls_or_empty(extracted.free_file.urls if extracted.free_file else [])

    # AGI eligibility limit
    leaf_agi = evaluator.add_leaf(
        id="AGI_Eligibility_Limit",
        desc="States the Free File AGI eligibility limit as $84,000 or less.",
        parent=node,
        critical=True
    )
    claim_agi = f"The answer states that the IRS Free File AGI eligibility limit is {FREE_FILE_AGI_LIMIT}."
    await evaluator.verify(
        claim=claim_agi,
        node=leaf_agi,
        additional_instruction="Accept equivalent phrasings such as 'AGI up to $84,000' or 'AGI ≤ $84,000'."
    )

    # Lead time days
    leaf_lead = evaluator.add_leaf(
        id="Free_File_Lead_Time_Days",
        desc="States how many days before the general filing season Free File starts, and the value matches the difference between the stated Free File start date (Jan 9, 2026) and season opening date (Jan 26, 2026).",
        parent=node,
        critical=True
    )
    claim_lead = (
        f"The answer states that IRS Free File starts {FREE_FILE_LEAD_TIME_DAYS} days before the general filing season opening."
    )
    await evaluator.verify(
        claim=claim_lead,
        node=leaf_lead,
        additional_instruction="Match the explicitly stated number in the answer, not a computed value. Accept '17 days earlier' or equivalent phrasing."
    )

    # Source URL support
    leaf_ff_sources = evaluator.add_leaf(
        id="Free_File_Source_URL",
        desc="Provides at least one official IRS or other U.S. government URL supporting the Free File details stated.",
        parent=node,
        critical=True
    )
    claim_ff_sources = (
        "At least one of these URLs is an official IRS or other U.S. government webpage (.gov) and it states the Free File "
        "AGI eligibility limit (e.g., $84,000 or less) and/or when Free File opens relative to the season (e.g., early/mid-January 2026, around Jan 9)."
    )
    await evaluator.verify(
        claim=claim_ff_sources,
        node=leaf_ff_sources,
        sources=urls,
        additional_instruction="Confirm .gov domain and that the page supports at least one of the Free File details mentioned."
    )


async def build_new_changes_subtree(evaluator: Evaluator, parent_node, extracted: IRS2026Extraction) -> None:
    node = evaluator.add_parallel(
        id="New_Changes_For_2026",
        desc="Provide the required new forms/policies/estimates announced for the 2026 filing season.",
        parent=parent_node,
        critical=True
    )
    urls = _urls_or_empty(extracted.new_changes.urls if extracted.new_changes else [])

    # New Schedule form
    leaf_sched = evaluator.add_leaf(
        id="New_Schedule_Form",
        desc="Identifies the new Schedule form introduced for claiming new deductions as Schedule 1-A.",
        parent=node,
        critical=True
    )
    claim_sched = f"The answer identifies the new schedule form for claiming new deductions as '{NEW_SCHEDULE_FORM}'."
    await evaluator.verify(
        claim=claim_sched,
        node=leaf_sched,
        additional_instruction="Allow minor variations like including 'Form' or 'Schedule 1-A (New)'; the core identifier must be 'Schedule 1-A'."
    )

    # Refund policy change regarding paper checks
    leaf_refund = evaluator.add_leaf
    leaf_refund = evaluator.add_leaf(
        id="Refund_Policy_Change",
        desc="Describes the announced change regarding paper refund checks (paper checks being phased out and/or direct deposit encouraged).",
        parent=node,
        critical=True
    )
    claim_refund = (
        "The answer states that the IRS announced a change regarding refund delivery, such as phasing out or reducing paper refund checks "
        "and encouraging or prioritizing direct deposit."
    )
    await evaluator.verify(
        claim=claim_refund,
        node=leaf_refund,
        additional_instruction="Accept paraphrases that clearly communicate fewer paper checks and more direct deposit; minor wording differences acceptable."
    )

    # Expected return volume
    leaf_volume = evaluator.add_leaf(
        id="Expected_Return_Volume",
        desc="States the IRS expected number of individual income tax returns as approximately 164 million.",
        parent=node,
        critical=True
    )
    claim_volume = "The answer states that the IRS expects approximately 164 million individual income tax returns."
    await evaluator.verify(
        claim=claim_volume,
        node=leaf_volume,
        additional_instruction="Accept reasonable qualifiers such as 'about', 'roughly', 'around 164 million'."
    )

    # Source URL support
    leaf_changes_sources = evaluator.add_leaf(
        id="Changes_Source_URL",
        desc="Provides at least one official IRS or other U.S. government URL supporting the stated new changes.",
        parent=node,
        critical=True
    )
    claim_changes_sources = (
        "At least one of these URLs is an official IRS or other U.S. government webpage (.gov) and it supports one or more of the following: "
        "the introduction of Schedule 1-A, a change regarding paper refund checks/direct deposit, or the expected volume (~164 million) of returns."
    )
    await evaluator.verify(
        claim=claim_changes_sources,
        node=leaf_changes_sources,
        sources=urls,
        additional_instruction="Confirm .gov domain and that at least one of the described changes is explicitly supported."
    )


async def build_standard_deductions_subtree(evaluator: Evaluator, parent_node, extracted: IRS2026Extraction) -> None:
    node = evaluator.add_parallel(
        id="Standard_Deduction_Amounts",
        desc="Provide required 2026 standard deduction amounts for specified filing statuses.",
        parent=parent_node,
        critical=True
    )
    urls = _urls_or_empty(extracted.standard_deductions.urls if extracted.standard_deductions else [])

    # Single filer
    leaf_single = evaluator.add_leaf(
        id="Single_Filer_Deduction",
        desc="States the 2026 standard deduction for single filers as $16,100.",
        parent=node,
        critical=True
    )
    claim_single = f"The answer states the 2026 standard deduction amount for single filers is {STD_DED_SINGLE}."
    await evaluator.verify(
        claim=claim_single,
        node=leaf_single,
        additional_instruction="Accept formatting variants like 16100, $16,100, or 16,100 dollars."
    )

    # Married filing jointly
    leaf_mfj = evaluator.add_leaf(
        id="Married_Joint_Deduction",
        desc="States the 2026 standard deduction for married filing jointly as $32,200.",
        parent=node,
        critical=True
    )
    claim_mfj = f"The answer states the 2026 standard deduction amount for married filing jointly is {STD_DED_MFJ}."
    await evaluator.verify(
        claim=claim_mfj,
        node=leaf_mfj,
        additional_instruction="Accept formatting variants like 32200, $32,200, or 32,200 dollars."
    )

    # Source URL support
    leaf_ded_sources = evaluator.add_leaf(
        id="Deduction_Source_URL",
        desc="Provides at least one official IRS or other U.S. government URL supporting the standard deduction amounts stated.",
        parent=node,
        critical=True
    )
    claim_ded_sources = (
        "At least one of these URLs is an official IRS or other U.S. government webpage (.gov) and it states the 2026 standard deduction amounts "
        f"for single ({STD_DED_SINGLE}) and/or married filing jointly ({STD_DED_MFJ})."
    )
    await evaluator.verify(
        claim=claim_ded_sources,
        node=leaf_ded_sources,
        sources=urls,
        additional_instruction="Confirm .gov domain and that the page provides the 2026 standard deduction amounts (either one or both statuses is acceptable)."
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
    Evaluate an answer for the IRS 2026 filing season task and return a structured evaluation summary.
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
        default_model=model
    )

    # Add a critical top-level node representing the task (since root is always non-critical by framework design)
    top = evaluator.add_parallel(
        id="IRS_2026_Filing_Season_Information",
        desc="Comprehensive information about the IRS 2026 filing season, with official IRS/government source URL support.",
        parent=root,
        critical=True
    )

    # Extract structured data from the answer
    extracted: IRS2026Extraction = await evaluator.extract(
        prompt=prompt_extract_irs_2026(),
        template_class=IRS2026Extraction,
        extraction_name="irs_2026_structured"
    )

    # Optional: add ground-truth targets for transparency
    evaluator.add_ground_truth({
        "opening_date": OPENING_DATE_WITH_WEEKDAY,
        "filing_deadline": FILING_DEADLINE_WITH_WEEKDAY,
        "free_file_start": FREE_FILE_START_WITH_WEEKDAY,
        "extension_deadline": EXTENSION_DEADLINE_STR,
        "free_file_agi_limit": FREE_FILE_AGI_LIMIT,
        "free_file_lead_time_days": FREE_FILE_LEAD_TIME_DAYS,
        "new_schedule_form": NEW_SCHEDULE_FORM,
        "expected_return_volume": EXPECTED_RETURN_VOLUME_APPROX,
        "std_ded_single_2026": STD_DED_SINGLE,
        "std_ded_mfj_2026": STD_DED_MFJ
    }, gt_type="expected_targets")

    # Build subtrees
    await build_key_filing_dates_subtree(evaluator, top, extracted)
    await build_free_file_program_subtree(evaluator, top, extracted)
    await build_new_changes_subtree(evaluator, top, extracted)
    await build_standard_deductions_subtree(evaluator, top, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()