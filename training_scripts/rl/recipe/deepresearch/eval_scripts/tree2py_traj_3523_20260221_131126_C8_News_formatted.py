import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "news_briefing_early_2026"
TASK_DESCRIPTION = """I need to prepare a comprehensive news briefing document covering four major US news developments from late 2025 through early 2026. For each topic below, provide the specific factual details requested:

1. The 2026 US Federal Government Shutdowns:
- The start date of the first shutdown
- The end date of the first shutdown
- The duration (in days) of the first shutdown
- The start date of the second shutdown
- The underlying cause of both shutdowns (including the specific triggering event and date)
- Which department is affected by the second shutdown

2. The US Military Intervention in Venezuela (Operation Absolute Resolve):
- The date the operation occurred
- The approximate start time of the operation (local Venezuela time)
- The names of the two primary individuals captured
- The name and swearing-in date of the person who became acting president
- The number of Venezuelan security personnel casualties reported by Venezuelan officials
- The number of Cuban military/intelligence casualties reported by the Cuban government
- The date Maduro and Flores were arraigned in US court
- The number of political prisoners released as of February 12, 2026

3. The Warrior Dividend Announcement:
- The dollar amount of the Warrior Dividend per service member
- The date the Warrior Dividend was announced
- The tax status of the payment
- The pay grade eligibility range for active duty service members
- The pay grade eligibility range for National Guard and Reserve members

4. The US Secretary of the Treasury:
- The name of the current Secretary of the Treasury
- Which number Secretary of the Treasury this person is (e.g., 79th)
- The date this person was sworn into office

For each factual detail, please provide a reference URL from a reliable source (government website, major news outlet, or established reference source) that verifies the information.
"""

# Expected facts (ground-truth targets used for content checks)
EXPECTED = {
    "shutdowns": {
        "first_start_date": "January 31, 2026",
        "first_end_date": "February 3, 2026",
        "first_duration": "4 days",
        "second_start_date": "February 14, 2026",
        "cause_phrase": "disputes over immigration enforcement reforms following the killing of Alex Pretti by CBP agents on January 24, 2026",
        "second_department": "Department of Homeland Security"
    },
    "venezuela": {
        "operation_date": "January 3, 2026",
        "operation_start_time": "around 2:00 a.m. VET",
        "captured_individuals": ["Nicolás Maduro", "Cilia Flores"],
        "acting_president_name": "Delcy Rodríguez",
        "acting_president_sworn_date": "January 5, 2026",
        "ven_casualties": "at least 23",
        "cuban_casualties": "32",
        "arraignment_date": "January 5, 2026",
        "political_prisoners_count": "431",
        "political_prisoners_as_of": "February 12, 2026",
    },
    "warrior_dividend": {
        "amount": "$1,776",
        "announcement_date": "December 17, 2025",
        "tax_status": "tax-free",
        "eligibility_active": "E-1 through O-6",
        "eligibility_guard_reserve": "E-1 through O-6",
    },
    "treasury": {
        "name": "Scott Bessent",
        "number": "79th",
        "sworn_date": "January 28, 2025",
    }
}

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ShutdownsExtraction(BaseModel):
    first_start_date: Optional[str] = None
    first_start_urls: List[str] = Field(default_factory=list)

    first_end_date: Optional[str] = None
    first_end_urls: List[str] = Field(default_factory=list)

    first_duration: Optional[str] = None
    first_duration_urls: List[str] = Field(default_factory=list)

    second_start_date: Optional[str] = None
    second_start_urls: List[str] = Field(default_factory=list)

    cause_summary: Optional[str] = None
    cause_trigger_event_date: Optional[str] = None
    cause_urls: List[str] = Field(default_factory=list)

    second_shutdown_department: Optional[str] = None
    second_dept_urls: List[str] = Field(default_factory=list)


class VenezuelaExtraction(BaseModel):
    operation_date: Optional[str] = None
    operation_date_urls: List[str] = Field(default_factory=list)

    operation_start_time_local: Optional[str] = None
    operation_start_time_urls: List[str] = Field(default_factory=list)

    captured_individuals: List[str] = Field(default_factory=list)
    captured_individuals_urls: List[str] = Field(default_factory=list)

    acting_president_name: Optional[str] = None
    acting_president_sworn_date: Optional[str] = None
    acting_president_urls: List[str] = Field(default_factory=list)

    venezuelan_casualties: Optional[str] = None
    venezuelan_casualties_urls: List[str] = Field(default_factory=list)

    cuban_casualties: Optional[str] = None
    cuban_casualties_urls: List[str] = Field(default_factory=list)

    arraignment_date: Optional[str] = None
    arraignment_date_urls: List[str] = Field(default_factory=list)

    political_prisoners_released: Optional[str] = None
    political_prisoners_as_of_date: Optional[str] = None
    political_prisoners_urls: List[str] = Field(default_factory=list)


class WarriorDividendExtraction(BaseModel):
    amount: Optional[str] = None
    amount_urls: List[str] = Field(default_factory=list)

    announcement_date: Optional[str] = None
    announcement_urls: List[str] = Field(default_factory=list)

    tax_status: Optional[str] = None
    tax_urls: List[str] = Field(default_factory=list)

    eligibility_active_pay_grades: Optional[str] = None
    eligibility_active_urls: List[str] = Field(default_factory=list)

    eligibility_guard_reserve_pay_grades: Optional[str] = None
    eligibility_guard_reserve_urls: List[str] = Field(default_factory=list)


class TreasuryExtraction(BaseModel):
    secretary_name: Optional[str] = None
    secretary_name_urls: List[str] = Field(default_factory=list)

    secretary_number: Optional[str] = None
    secretary_number_urls: List[str] = Field(default_factory=list)

    swearing_in_date: Optional[str] = None
    swearing_in_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_shutdowns() -> str:
    return """
    Extract the following details about the 2026 US federal government shutdowns from the answer. For each detail, also extract all explicitly cited reference URLs that verify the detail.

    Fields:
    - first_start_date: the stated start date of the first shutdown (e.g., "January 31, 2026")
    - first_start_urls: list of URLs that verify the first shutdown start date
    - first_end_date: the stated end date of the first shutdown (e.g., "February 3, 2026")
    - first_end_urls: list of URLs that verify the first shutdown end date
    - first_duration: the stated total duration of the first shutdown (e.g., "4 days")
    - first_duration_urls: list of URLs that verify the duration (or provide start/end dates usable to compute duration)
    - second_start_date: the stated start date of the second shutdown (e.g., "February 14, 2026")
    - second_start_urls: list of URLs that verify the second shutdown start date
    - cause_summary: a concise sentence summarizing the underlying cause (e.g., disputes over immigration enforcement reforms following the killing of Alex Pretti)
    - cause_trigger_event_date: the date of the triggering event (e.g., "January 24, 2026")
    - cause_urls: list of URLs that verify the cause and triggering event/date
    - second_shutdown_department: the department affected by the second shutdown (e.g., "Department of Homeland Security")
    - second_dept_urls: list of URLs that verify which department is affected

    URL extraction rules:
    - Extract only valid URLs explicitly present in the answer text.
    - Include URLs from government sites, major news outlets, or established reference sources where available.
    - If a required URL is not provided, return an empty list for that field.
    """


def prompt_extract_venezuela() -> str:
    return """
    Extract the following details about the US military intervention in Venezuela (Operation Absolute Resolve) from the answer. For each detail, also extract all explicitly cited reference URLs that verify the detail.

    Fields:
    - operation_date: the stated date the operation occurred (e.g., "January 3, 2026")
    - operation_date_urls: list of URLs that verify the operation date
    - operation_start_time_local: the stated approximate local start time in Venezuela (e.g., "around 2:00 a.m. VET")
    - operation_start_time_urls: list of URLs that verify the start time
    - captured_individuals: list of names of the two primary captured individuals (e.g., ["Nicolás Maduro", "Cilia Flores"])
    - captured_individuals_urls: list of URLs that verify the captured individuals
    - acting_president_name: the name of the acting president (e.g., "Delcy Rodríguez")
    - acting_president_sworn_date: the swearing-in date of the acting president (e.g., "January 5, 2026")
    - acting_president_urls: list of URLs that verify the acting president and swearing-in date
    - venezuelan_casualties: the reported number/phrase of Venezuelan security personnel casualties (e.g., "at least 23")
    - venezuelan_casualties_urls: list of URLs that verify Venezuelan casualties
    - cuban_casualties: the reported number of Cuban military/intelligence casualties (e.g., "32")
    - cuban_casualties_urls: list of URLs that verify Cuban casualties
    - arraignment_date: the date Maduro and Flores were arraigned in US court (e.g., "January 5, 2026")
    - arraignment_date_urls: list of URLs that verify the arraignment date
    - political_prisoners_released: the number of political prisoners released (e.g., "431")
    - political_prisoners_as_of_date: the "as of" date of that count (e.g., "February 12, 2026")
    - political_prisoners_urls: list of URLs that verify the number released and the date context

    URL extraction rules:
    - Extract only valid URLs explicitly present in the answer text.
    - Include URLs from government sites, major news outlets, or established reference sources where available.
    - If a required URL is not provided, return an empty list for that field.
    """


def prompt_extract_warrior_dividend() -> str:
    return """
    Extract the following details about the Warrior Dividend from the answer. For each detail, also extract all explicitly cited reference URLs that verify the detail.

    Fields:
    - amount: the stated dollar amount per service member (e.g., "$1,776")
    - amount_urls: list of URLs that verify the amount
    - announcement_date: the date it was announced (e.g., "December 17, 2025")
    - announcement_urls: list of URLs that verify the announcement date
    - tax_status: the stated tax status (e.g., "tax-free")
    - tax_urls: list of URLs that verify the tax status
    - eligibility_active_pay_grades: the pay grade eligibility range for active duty (e.g., "E-1 through O-6")
    - eligibility_active_urls: list of URLs that verify active duty eligibility
    - eligibility_guard_reserve_pay_grades: the pay grade eligibility range for National Guard and Reserve (e.g., "E-1 through O-6")
    - eligibility_guard_reserve_urls: list of URLs that verify Guard/Reserve eligibility

    URL extraction rules:
    - Extract only valid URLs explicitly present in the answer text.
    - Include URLs from government sites, major news outlets, or established reference sources where available.
    - If a required URL is not provided, return an empty list for that field.
    """


def prompt_extract_treasury() -> str:
    return """
    Extract the following details about the current US Secretary of the Treasury from the answer. For each detail, also extract all explicitly cited reference URLs that verify the detail.

    Fields:
    - secretary_name: the name of the current Secretary of the Treasury (e.g., "Scott Bessent")
    - secretary_name_urls: list of URLs that verify the Secretary's name
    - secretary_number: which number Secretary this person is (e.g., "79th")
    - secretary_number_urls: list of URLs that verify the ordinal number
    - swearing_in_date: the date sworn into office (e.g., "January 28, 2025")
    - swearing_in_urls: list of URLs that verify the swearing-in date

    URL extraction rules:
    - Extract only valid URLs explicitly present in the answer text.
    - Include URLs from government sites, major news outlets, or established reference sources where available.
    - If a required URL is not provided, return an empty list for that field.
    """

# --------------------------------------------------------------------------- #
# Helper for URL-supported verification                                       #
# --------------------------------------------------------------------------- #
async def verify_with_urls_or_fail_if_missing(
    evaluator: Evaluator,
    claim: str,
    node,
    urls: List[str],
    guidance: str
) -> None:
    """
    Verify a claim with provided URLs. If URLs are missing, explicitly fail per policy.
    """
    if urls and len(urls) > 0:
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=guidance
        )
    else:
        await evaluator.verify(
            claim=f"NO-URL-PROVIDED: {claim}",
            node=node,
            sources=None,
            additional_instruction="No reference URLs were provided in the answer for this detail. Per evaluation policy, mark the claim as not supported and Incorrect."
        )

# --------------------------------------------------------------------------- #
# Verification builders per section                                           #
# --------------------------------------------------------------------------- #
async def build_shutdowns_section(evaluator: Evaluator, root, data: ShutdownsExtraction) -> None:
    sec = evaluator.add_parallel(
        id="Government_Shutdowns_Section",
        desc="Information about the 2026 US federal government shutdowns",
        parent=root,
        critical=False
    )

    # First Shutdown Start Date
    s1 = evaluator.add_parallel(
        id="First_Shutdown_Start_Date_Detail",
        desc="Information about the first shutdown start date",
        parent=sec,
        critical=False
    )
    leaf_content = evaluator.add_leaf(
        id="First_Shutdown_Start_Date_Content",
        desc="States that the first shutdown began on January 31, 2026",
        parent=s1,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that the first 2026 US federal government shutdown began on {EXPECTED['shutdowns']['first_start_date']}.",
        node=leaf_content,
        additional_instruction="Judge solely based on whether the answer text explicitly provides this exact start date; if the date is missing or different, mark Incorrect."
    )
    leaf_url = evaluator.add_leaf(
        id="First_Shutdown_Start_Date_URL",
        desc="Provides a reference URL verifying the first shutdown start date",
        parent=s1,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"The first 2026 US federal government shutdown began on {EXPECTED['shutdowns']['first_start_date']}.",
        node=leaf_url,
        urls=data.first_start_urls,
        guidance="Use the cited source(s) to confirm the exact start date. Accept minor formatting variations (e.g., 'Jan. 31, 2026')."
    )

    # First Shutdown End Date
    s2 = evaluator.add_parallel(
        id="First_Shutdown_End_Date_Detail",
        desc="Information about the first shutdown end date",
        parent=sec,
        critical=False
    )
    leaf_content = evaluator.add_leaf(
        id="First_Shutdown_End_Date_Content",
        desc="States that the first shutdown ended on February 3, 2026",
        parent=s2,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that the first 2026 US federal government shutdown ended on {EXPECTED['shutdowns']['first_end_date']}.",
        node=leaf_content,
        additional_instruction="Judge solely based on whether the answer text explicitly provides this exact end date; if the date is missing or different, mark Incorrect."
    )
    leaf_url = evaluator.add_leaf(
        id="First_Shutdown_End_Date_URL",
        desc="Provides a reference URL verifying the first shutdown end date",
        parent=s2,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"The first 2026 US federal government shutdown ended on {EXPECTED['shutdowns']['first_end_date']}.",
        node=leaf_url,
        urls=data.first_end_urls,
        guidance="Use the cited source(s) to confirm the exact end date. Accept minor formatting variations."
    )

    # First Shutdown Duration
    s3 = evaluator.add_parallel(
        id="First_Shutdown_Duration_Detail",
        desc="Information about the first shutdown duration",
        parent=sec,
        critical=False
    )
    leaf_content = evaluator.add_leaf(
        id="First_Shutdown_Duration_Content",
        desc="States that the first shutdown lasted 4 days",
        parent=s3,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that the first shutdown lasted {EXPECTED['shutdowns']['first_duration']}.",
        node=leaf_content,
        additional_instruction="Judge solely based on whether the answer text explicitly provides this exact duration; if missing or different phrasing implying a different duration, mark Incorrect."
    )
    leaf_url = evaluator.add_leaf(
        id="First_Shutdown_Duration_URL",
        desc="Provides a reference URL verifying the first shutdown duration",
        parent=s3,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"The first shutdown lasted {EXPECTED['shutdowns']['first_duration']}.",
        node=leaf_url,
        urls=data.first_duration_urls,
        guidance="If duration is not directly stated, compute from start/end dates present on the page. Accept reasonable equivalence ('four days' vs '4 days')."
    )

    # Second Shutdown Start Date
    s4 = evaluator.add_parallel(
        id="Second_Shutdown_Start_Date_Detail",
        desc="Information about the second shutdown start date",
        parent=sec,
        critical=False
    )
    leaf_content = evaluator.add_leaf(
        id="Second_Shutdown_Start_Date_Content",
        desc="States that the second shutdown began on February 14, 2026",
        parent=s4,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that the second 2026 US federal government shutdown began on {EXPECTED['shutdowns']['second_start_date']}.",
        node=leaf_content,
        additional_instruction="Judge solely based on whether the answer text explicitly provides this exact start date; if missing or different, mark Incorrect."
    )
    leaf_url = evaluator.add_leaf(
        id="Second_Shutdown_Start_Date_URL",
        desc="Provides a reference URL verifying the second shutdown start date",
        parent=s4,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"The second 2026 US federal government shutdown began on {EXPECTED['shutdowns']['second_start_date']}.",
        node=leaf_url,
        urls=data.second_start_urls,
        guidance="Use the cited source(s) to confirm the exact start date."
    )

    # Shutdown Cause (triggering event and date)
    s5 = evaluator.add_parallel(
        id="Shutdown_Cause_Detail",
        desc="Information about the underlying cause of both shutdowns",
        parent=sec,
        critical=False
    )
    leaf_content = evaluator.add_leaf(
        id="Shutdown_Cause_Content",
        desc="Identifies the cause as disputes over immigration enforcement reforms following the killing of Alex Pretti by CBP agents on January 24, 2026",
        parent=s5,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly identifies the cause as disputes over immigration enforcement reforms following the killing of Alex Pretti by CBP agents on January 24, 2026.",
        node=leaf_content,
        additional_instruction="Judge solely from the answer text whether it contains these elements: (1) disputes over immigration enforcement reforms, and (2) the killing of Alex Pretti by CBP agents on January 24, 2026. If either is missing or different, mark Incorrect."
    )
    leaf_url = evaluator.add_leaf(
        id="Shutdown_Cause_URL",
        desc="Provides a reference URL verifying the shutdown cause and Alex Pretti death date",
        parent=s5,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim="The shutdowns were driven by disputes over immigration enforcement reforms following the killing of Alex Pretti by CBP agents on January 24, 2026.",
        node=leaf_url,
        urls=data.cause_urls,
        guidance="Confirm both the reform dispute characterization and Alex Pretti's death details/date from the cited sources."
    )

    # Second Shutdown Department
    s6 = evaluator.add_parallel(
        id="Second_Shutdown_Department_Detail",
        desc="Information about which department is affected by the second shutdown",
        parent=sec,
        critical=False
    )
    leaf_content = evaluator.add_leaf(
        id="Second_Shutdown_Department_Content",
        desc="Identifies that the second shutdown affects only the Department of Homeland Security",
        parent=s6,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the second shutdown affects only the Department of Homeland Security.",
        node=leaf_content,
        additional_instruction="Judge solely based on the answer text; if the department is not DHS or 'only DHS' is not stated, mark Incorrect."
    )
    leaf_url = evaluator.add_leaf(
        id="Second_Shutdown_Department_URL",
        desc="Provides a reference URL verifying which department is affected by the second shutdown",
        parent=s6,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim="The second 2026 US federal shutdown affected only the Department of Homeland Security.",
        node=leaf_url,
        urls=data.second_dept_urls,
        guidance="Confirm from the referenced source(s) that the second shutdown scope was limited to DHS."
    )


async def build_venezuela_section(evaluator: Evaluator, root, data: VenezuelaExtraction) -> None:
    sec = evaluator.add_parallel(
        id="Venezuela_Intervention_Section",
        desc="Information about the US military intervention in Venezuela",
        parent=root,
        critical=False
    )

    # Operation Date
    d1 = evaluator.add_parallel(
        id="Operation_Date_Detail",
        desc="Information about when the operation occurred",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Operation_Date_Content",
        desc="States that Operation Absolute Resolve occurred on January 3, 2026",
        parent=d1,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that Operation Absolute Resolve occurred on {EXPECTED['venezuela']['operation_date']}.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; if the date is missing or different, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Operation_Date_URL",
        desc="Provides a reference URL verifying the operation date",
        parent=d1,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"Operation Absolute Resolve occurred on {EXPECTED['venezuela']['operation_date']}.",
        node=leaf_u,
        urls=data.operation_date_urls,
        guidance="Verify the operation date via cited source(s); accept minor date format variants."
    )

    # Operation Start Time
    d2 = evaluator.add_parallel(
        id="Operation_Start_Time_Detail",
        desc="Information about the operation start time",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Operation_Start_Time_Content",
        desc="States that the operation began around 2:00 a.m. local time VET",
        parent=d2,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that the operation began {EXPECTED['venezuela']['operation_start_time']} local time in Venezuela.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; allow phrasing like 'about 2 a.m. VET'; if missing or clearly different, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Operation_Start_Time_URL",
        desc="Provides a reference URL verifying the operation start time",
        parent=d2,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"The operation began {EXPECTED['venezuela']['operation_start_time']} (local VET).",
        node=leaf_u,
        urls=data.operation_start_time_urls,
        guidance="Confirm the approximate start time from the cited source(s); allow 'about/around 2 a.m.' variants."
    )

    # Captured Individuals
    d3 = evaluator.add_parallel(
        id="Captured_Individuals_Detail",
        desc="Information about the individuals captured",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Captured_Individuals_Content",
        desc="Identifies that President Nicolás Maduro and his wife Cilia Flores were captured",
        parent=d3,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that President Nicolás Maduro and his wife Cilia Flores were captured.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; if either name is missing or different, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Captured_Individuals_URL",
        desc="Provides a reference URL verifying the captured individuals",
        parent=d3,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim="President Nicolás Maduro and Cilia Flores were captured during the operation.",
        node=leaf_u,
        urls=data.captured_individuals_urls,
        guidance="Verify from the cited source(s) that Maduro and Cilia Flores were captured."
    )

    # Acting President and swearing-in date
    d4 = evaluator.add_parallel(
        id="Acting_President_Detail",
        desc="Information about the acting president",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Acting_President_Content",
        desc="Identifies Delcy Rodríguez as the acting president sworn in on January 5, 2026",
        parent=d4,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that Delcy Rodríguez became acting president and was sworn in on {EXPECTED['venezuela']['acting_president_sworn_date']}.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; both name and date must be present; otherwise mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Acting_President_URL",
        desc="Provides a reference URL verifying the acting president and swearing-in date",
        parent=d4,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"Delcy Rodríguez became acting president and was sworn in on {EXPECTED['venezuela']['acting_president_sworn_date']}.",
        node=leaf_u,
        urls=data.acting_president_urls,
        guidance="Verify both identity and swearing-in date via cited source(s)."
    )

    # Venezuelan casualties
    d5 = evaluator.add_parallel(
        id="Venezuelan_Casualties_Detail",
        desc="Information about Venezuelan casualties",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Venezuelan_Casualties_Content",
        desc="Reports at least 23 Venezuelan security officers killed",
        parent=d5,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly reports {EXPECTED['venezuela']['ven_casualties']} Venezuelan security officers killed.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; allow wording like 'no fewer than 23'; if missing or clearly different amount, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Venezuelan_Casualties_URL",
        desc="Provides a reference URL verifying Venezuelan casualties",
        parent=d5,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"Venezuelan officials reported {EXPECTED['venezuela']['ven_casualties']} security officers killed.",
        node=leaf_u,
        urls=data.venezuelan_casualties_urls,
        guidance="Confirm the reported minimum count via cited source(s); allow phrasing like 'at least' or 'no fewer than'."
    )

    # Cuban casualties
    d6 = evaluator.add_parallel(
        id="Cuban_Casualties_Detail",
        desc="Information about Cuban casualties",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Cuban_Casualties_Content",
        desc="Reports 32 Cuban military and intelligence personnel killed",
        parent=d6,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly reports {EXPECTED['venezuela']['cuban_casualties']} Cuban military and intelligence personnel killed.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; if missing or different amount, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Cuban_Casualties_URL",
        desc="Provides a reference URL verifying Cuban casualties",
        parent=d6,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"The Cuban government reported {EXPECTED['venezuela']['cuban_casualties']} military/intelligence personnel killed.",
        node=leaf_u,
        urls=data.cuban_casualties_urls,
        guidance="Confirm the count via cited source(s)."
    )

    # Arraignment date
    d7 = evaluator.add_parallel(
        id="Arraignment_Date_Detail",
        desc="Information about the arraignment date",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Arraignment_Date_Content",
        desc="States that Maduro and Flores were arraigned on January 5, 2026",
        parent=d7,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that Maduro and Flores were arraigned on {EXPECTED['venezuela']['arraignment_date']}.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; if the date is missing or different, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Arraignment_Date_URL",
        desc="Provides a reference URL verifying the arraignment date",
        parent=d7,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"Maduro and Flores were arraigned in US court on {EXPECTED['venezuela']['arraignment_date']}.",
        node=leaf_u,
        urls=data.arraignment_date_urls,
        guidance="Verify the arraignment date via cited source(s)."
    )

    # Political prisoners released
    d8 = evaluator.add_parallel(
        id="Political_Prisoners_Detail",
        desc="Information about political prisoners released",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Political_Prisoners_Content",
        desc="Reports that 431 political prisoners were released as of February 12, 2026",
        parent=d8,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly reports {EXPECTED['venezuela']['political_prisoners_count']} political prisoners released as of {EXPECTED['venezuela']['political_prisoners_as_of']}.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; both the number and 'as of' date must be present; otherwise mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Political_Prisoners_URL",
        desc="Provides a reference URL verifying the number of political prisoners released",
        parent=d8,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"As of {EXPECTED['venezuela']['political_prisoners_as_of']}, {EXPECTED['venezuela']['political_prisoners_count']} political prisoners were released.",
        node=leaf_u,
        urls=data.political_prisoners_urls,
        guidance="Verify both the count and the 'as of' date via cited source(s)."
    )


async def build_warrior_dividend_section(evaluator: Evaluator, root, data: WarriorDividendExtraction) -> None:
    sec = evaluator.add_parallel(
        id="Warrior_Dividend_Section",
        desc="Information about the Warrior Dividend payment to service members",
        parent=root,
        critical=False
    )

    # Payment Amount
    d1 = evaluator.add_parallel(
        id="Payment_Amount_Detail",
        desc="Information about the payment amount",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Payment_Amount_Content",
        desc="States that the Warrior Dividend is $1,776 per service member",
        parent=d1,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that the Warrior Dividend is {EXPECTED['warrior_dividend']['amount']} per service member.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; if missing or different amount, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Payment_Amount_URL",
        desc="Provides a reference URL verifying the payment amount",
        parent=d1,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"The Warrior Dividend is {EXPECTED['warrior_dividend']['amount']} per service member.",
        node=leaf_u,
        urls=data.amount_urls,
        guidance="Verify the amount via cited source(s)."
    )

    # Announcement Date
    d2 = evaluator.add_parallel(
        id="Announcement_Date_Detail",
        desc="Information about the announcement date",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Announcement_Date_Content",
        desc="States that the announcement was made on December 17, 2025",
        parent=d2,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that the Warrior Dividend was announced on {EXPECTED['warrior_dividend']['announcement_date']}.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; if missing or different date, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Announcement_Date_URL",
        desc="Provides a reference URL verifying the announcement date",
        parent=d2,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"The Warrior Dividend was announced on {EXPECTED['warrior_dividend']['announcement_date']}.",
        node=leaf_u,
        urls=data.announcement_urls,
        guidance="Verify the announcement date via cited source(s)."
    )

    # Tax Status
    d3 = evaluator.add_parallel(
        id="Tax_Status_Detail",
        desc="Information about the tax status",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Tax_Status_Content",
        desc="States that the Warrior Dividend is tax-free",
        parent=d3,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that the Warrior Dividend is {EXPECTED['warrior_dividend']['tax_status']}.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; if the tax status is missing or different, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Tax_Status_URL",
        desc="Provides a reference URL verifying the tax status",
        parent=d3,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim="The Warrior Dividend is tax-free.",
        node=leaf_u,
        urls=data.tax_urls,
        guidance="Verify the tax treatment via cited source(s); accept equivalent phrasing like 'not taxable' or 'tax exempt'."
    )

    # Eligibility Active Duty
    d4 = evaluator.add_parallel(
        id="Eligibility_Active_Duty_Detail",
        desc="Information about active duty eligibility",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Eligibility_Active_Duty_Content",
        desc="Identifies that active duty service members in pay grades E-1 through O-6 are eligible",
        parent=d4,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that active duty service members in pay grades {EXPECTED['warrior_dividend']['eligibility_active']} are eligible.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; if range missing or different, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Eligibility_Active_Duty_URL",
        desc="Provides a reference URL verifying active duty eligibility",
        parent=d4,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"Active duty service members in pay grades {EXPECTED['warrior_dividend']['eligibility_active']} are eligible for the Warrior Dividend.",
        node=leaf_u,
        urls=data.eligibility_active_urls,
        guidance="Verify eligibility range via cited source(s); accept variants like 'E1–O6'."
    )

    # Eligibility Guard/Reserve
    d5 = evaluator.add_parallel(
        id="Eligibility_Guard_Reserve_Detail",
        desc="Information about Guard and Reserve eligibility",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Eligibility_Guard_Reserve_Content",
        desc="Identifies that National Guard and Reserve members in pay grades E-1 through O-6 are eligible",
        parent=d5,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that National Guard and Reserve members in pay grades {EXPECTED['warrior_dividend']['eligibility_guard_reserve']} are eligible.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; if range missing or different, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Eligibility_Guard_Reserve_URL",
        desc="Provides a reference URL verifying Guard and Reserve eligibility",
        parent=d5,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"National Guard and Reserve members in pay grades {EXPECTED['warrior_dividend']['eligibility_guard_reserve']} are eligible for the Warrior Dividend.",
        node=leaf_u,
        urls=data.eligibility_guard_reserve_urls,
        guidance="Verify eligibility range via cited source(s); accept variants like 'E1–O6'."
    )


async def build_treasury_section(evaluator: Evaluator, root, data: TreasuryExtraction) -> None:
    sec = evaluator.add_parallel(
        id="Treasury_Secretary_Section",
        desc="Information about the US Secretary of the Treasury",
        parent=root,
        critical=False
    )

    # Secretary Name
    d1 = evaluator.add_parallel(
        id="Secretary_Name_Detail",
        desc="Information about the Secretary's name",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Secretary_Name_Content",
        desc="Identifies Scott Bessent as the Secretary of the Treasury",
        parent=d1,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly identifies {EXPECTED['treasury']['name']} as the Secretary of the Treasury.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; if the name is different or missing, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Secretary_Name_URL",
        desc="Provides a reference URL verifying the Secretary's name",
        parent=d1,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"{EXPECTED['treasury']['name']} is the current US Secretary of the Treasury.",
        node=leaf_u,
        urls=data.secretary_name_urls,
        guidance="Verify the identity via cited source(s); prefer official government sources when available."
    )

    # Secretary Number
    d2 = evaluator.add_parallel(
        id="Secretary_Number_Detail",
        desc="Information about which number Secretary this is",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Secretary_Number_Content",
        desc="States that Scott Bessent is the 79th Secretary of the Treasury",
        parent=d2,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that {EXPECTED['treasury']['name']} is the {EXPECTED['treasury']['number']} Secretary of the Treasury.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; if ordinal number is missing or different, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Secretary_Number_URL",
        desc="Provides a reference URL verifying the Secretary number",
        parent=d2,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"{EXPECTED['treasury']['name']} is the {EXPECTED['treasury']['number']} US Secretary of the Treasury.",
        node=leaf_u,
        urls=data.secretary_number_urls,
        guidance="Verify the ordinal number via cited source(s)."
    )

    # Swearing-in date
    d3 = evaluator.add_parallel(
        id="Swearing_In_Date_Detail",
        desc="Information about the swearing-in date",
        parent=sec,
        critical=False
    )
    leaf_c = evaluator.add_leaf(
        id="Swearing_In_Date_Content",
        desc="States that Scott Bessent was sworn in on January 28, 2025",
        parent=d3,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that {EXPECTED['treasury']['name']} was sworn in on {EXPECTED['treasury']['sworn_date']}.",
        node=leaf_c,
        additional_instruction="Judge solely from the answer text; if date is missing or different, mark Incorrect."
    )
    leaf_u = evaluator.add_leaf(
        id="Swearing_In_Date_URL",
        desc="Provides a reference URL verifying the swearing-in date",
        parent=d3,
        critical=True
    )
    await verify_with_urls_or_fail_if_missing(
        evaluator,
        claim=f"{EXPECTED['treasury']['name']} was sworn in as Secretary of the Treasury on {EXPECTED['treasury']['sworn_date']}.",
        node=leaf_u,
        urls=data.swearing_in_urls,
        guidance="Verify swearing-in date via cited source(s); prefer official government sources when available."
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
    """
    Build the verification tree for the comprehensive news briefing and run checks.
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

    # Record expected facts as ground truth info for transparency
    evaluator.add_ground_truth({"expected_facts": EXPECTED}, gt_type="expected_facts")

    # Extract structured info from the answer
    shutdowns_data, venezuela_data, warrior_data, treasury_data = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_shutdowns(),
            template_class=ShutdownsExtraction,
            extraction_name="shutdowns_section"
        ),
        evaluator.extract(
            prompt=prompt_extract_venezuela(),
            template_class=VenezuelaExtraction,
            extraction_name="venezuela_section"
        ),
        evaluator.extract(
            prompt=prompt_extract_warrior_dividend(),
            template_class=WarriorDividendExtraction,
            extraction_name="warrior_dividend_section"
        ),
        evaluator.extract(
            prompt=prompt_extract_treasury(),
            template_class=TreasuryExtraction,
            extraction_name="treasury_section"
        ),
    )

    # Build sections according to rubric and run verifications
    await build_shutdowns_section(evaluator, root, shutdowns_data)
    await build_venezuela_section(evaluator, root, venezuela_data)
    await build_warrior_dividend_section(evaluator, root, warrior_data)
    await build_treasury_section(evaluator, root, treasury_data)

    # Return the evaluation summary
    return evaluator.get_summary()