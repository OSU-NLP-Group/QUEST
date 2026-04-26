import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gov_shutdown_2025_info"
TASK_DESCRIPTION = (
    "Provide comprehensive information about the 2025 U.S. federal government shutdown, including: "
    "(1) Duration and Timeline: The exact number of days the shutdown lasted, the start date, the end date, and the date when it became the longest shutdown in U.S. history. "
    "(2) Historical Context: Confirmation that it was the longest government shutdown in U.S. history, and information about the previous record (duration and the administration during which it occurred). "
    "(3) Workforce Impact: The approximate number of federal employees who were furloughed, the approximate number who worked without pay, and the total number of federal workers affected. "
    "(4) Economic Impact: The Congressional Budget Office's (CBO) estimated impact on GDP growth, and estimates of the total economic loss. "
    "(5) Resolution: The type of legislation Congress passed to end the shutdown, confirmation that President Trump signed it, and the duration of funding provided by the resolution. "
    "For each piece of information, provide a reference URL from a credible source that confirms the details."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FactWithSources(BaseModel):
    statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ShutdownInfoExtraction(BaseModel):
    # Duration and Timeline
    duration_days: Optional[FactWithSources] = None
    start_date: Optional[FactWithSources] = None
    start_of_fy2026: Optional[FactWithSources] = None
    end_date: Optional[FactWithSources] = None
    longest_date: Optional[FactWithSources] = None

    # Historical Context
    confirmed_longest: Optional[FactWithSources] = None
    previous_record_duration: Optional[FactWithSources] = None
    previous_record_timeframe: Optional[FactWithSources] = None
    previous_record_administration: Optional[FactWithSources] = None

    # Workforce Impact
    furloughed_employees: Optional[FactWithSources] = None
    worked_without_pay: Optional[FactWithSources] = None
    total_affected: Optional[FactWithSources] = None

    # Economic Impact
    cbo_gdp_impact_q4: Optional[FactWithSources] = None
    total_economic_loss: Optional[FactWithSources] = None

    # Resolution
    resolution_type: Optional[FactWithSources] = None
    resolution_signed_by_president: Optional[FactWithSources] = None
    resolution_signing_date: Optional[FactWithSources] = None
    resolution_funding_duration: Optional[FactWithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shutdown_info() -> str:
    return """
Extract the following facts about the 2025 U.S. federal government shutdown as explicitly stated in the answer. For each fact, return:
- statement: the exact claim as stated in the answer (use the answer’s wording; concise and unambiguous)
- urls: an array of the specific reference URLs in the answer that directly support this particular statement

If an item is not present in the answer, set that item to null. If present but without any supporting URLs, set urls to an empty array.

Required JSON schema with all fields (keep field names exact):

{
  "duration_days": {"statement": string|null, "urls": string[]},
  "start_date": {"statement": string|null, "urls": string[]},
  "start_of_fy2026": {"statement": string|null, "urls": string[]},
  "end_date": {"statement": string|null, "urls": string[]},
  "longest_date": {"statement": string|null, "urls": string[]},

  "confirmed_longest": {"statement": string|null, "urls": string[]},
  "previous_record_duration": {"statement": string|null, "urls": string[]},
  "previous_record_timeframe": {"statement": string|null, "urls": string[]},
  "previous_record_administration": {"statement": string|null, "urls": string[]},

  "furloughed_employees": {"statement": string|null, "urls": string[]},
  "worked_without_pay": {"statement": string|null, "urls": string[]},
  "total_affected": {"statement": string|null, "urls": string[]},

  "cbo_gdp_impact_q4": {"statement": string|null, "urls": string[]},
  "total_economic_loss": {"statement": string|null, "urls": string[]},

  "resolution_type": {"statement": string|null, "urls": string[]},
  "resolution_signed_by_president": {"statement": string|null, "urls": string[]},
  "resolution_signing_date": {"statement": string|null, "urls": string[]},
  "resolution_funding_duration": {"statement": string|null, "urls": string[]}
}

Instructions:
- The statements must relate specifically to the 2025 U.S. federal government shutdown, not to other years.
- For each urls list, include only the URLs explicitly present in the answer that support that specific statement (plain URLs or markdown links).
- Do not invent or infer any values or URLs. If the answer gives a range or approximate wording, keep that wording in 'statement'.
- Dates should keep the original formatting from the answer (e.g., 'October 1, 2025', 'Nov. 12, 2025').
- If the answer references a credible institution by name without a URL, do NOT create a URL; leave urls empty.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _collect_all_urls(data: ShutdownInfoExtraction) -> List[str]:
    seen = set()
    ordered_urls: List[str] = []

    fields: List[Optional[FactWithSources]] = [
        data.duration_days,
        data.start_date,
        data.start_of_fy2026,
        data.end_date,
        data.longest_date,
        data.confirmed_longest,
        data.previous_record_duration,
        data.previous_record_timeframe,
        data.previous_record_administration,
        data.furloughed_employees,
        data.worked_without_pay,
        data.total_affected,
        data.cbo_gdp_impact_q4,
        data.total_economic_loss,
        data.resolution_type,
        data.resolution_signed_by_president,
        data.resolution_signing_date,
        data.resolution_funding_duration,
    ]
    for f in fields:
        if f and f.urls:
            for u in f.urls:
                u2 = (u or "").strip()
                if u2 and u2 not in seen:
                    seen.add(u2)
                    ordered_urls.append(u2)
    return ordered_urls


async def _add_fact_with_support(
    evaluator: Evaluator,
    parent,
    field: Optional[FactWithSources],
    leaf_id: str,
    leaf_desc: str,
    exists_id_suffix: str = "provided",
    additional_instruction: str = "",
) -> None:
    exists = field is not None and _non_empty(field.statement) and bool(field.urls)
    evaluator.add_custom_node(
        result=exists,
        id=f"{leaf_id}_{exists_id_suffix}",
        desc=f"{leaf_desc} — statement and at least one supporting URL are provided in the answer",
        parent=parent,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id=leaf_id,
        desc=leaf_desc,
        parent=parent,
        critical=True,
    )

    claim = field.statement if field and field.statement else ""
    sources = field.urls if field and field.urls else None

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction or "None",
    )


# --------------------------------------------------------------------------- #
# Verification builders per rubric section                                    #
# --------------------------------------------------------------------------- #
async def build_source_credibility_section(
    evaluator: Evaluator,
    parent,
    all_urls: List[str],
) -> None:
    src_node = evaluator.add_parallel(
        id="Source_Credibility",
        desc="All provided reference URLs are from credible sources (i.e., official government/public-institution sites, established news organizations, or established nonpartisan research institutions).",
        parent=parent,
        critical=True,
    )

    # Optional existence check for any URLs present at all
    evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id="Any_URLs_Present",
        desc="At least one reference URL is provided in the answer",
        parent=src_node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="All_URLs_Are_Credible",
        desc="Every reference URL used for any claim is from a credible source as defined in this rubric.",
        parent=src_node,
        critical=True,
    )

    # Build a single simple verification claim that lists all URLs and asks the judge to decide credibility.
    url_list_text = "\n".join(f"- {u}" for u in all_urls) if all_urls else "(none)"
    claim = (
        "Evaluate the credibility of the following URLs used as references for the 2025 shutdown facts. "
        "Credible sources include: official government/public-institution sites (.gov, .mil, .edu; Congress, White House, OMB, CBO, GAO, CRS), "
        "established nonpartisan research institutions (e.g., Pew, RAND, Brookings), and established news organizations (e.g., AP, Reuters, WSJ, FT, NYT, WaPo, NPR, BBC). "
        "Only pass if ALL of the listed URLs are from credible sources by domain/organization identity.\n"
        f"URLs to assess:\n{url_list_text}"
    )

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=None,  # This is a meta-credibility check; judge based on domains, not page content.
        additional_instruction="Judge solely based on the domains/organizations. If any URL is not clearly credible per the rubric, mark as Incorrect.",
    )


async def build_duration_timeline_section(
    evaluator: Evaluator,
    parent,
    data: ShutdownInfoExtraction,
) -> None:
    dt_node = evaluator.add_parallel(
        id="Duration_and_Timeline",
        desc="Shutdown duration and key dates; each fact must include a URL that confirms it.",
        parent=parent,
        critical=True,
    )

    # Total duration (expects exactly 43 days per rubric name; we verify the user's stated duration against their sources)
    await _add_fact_with_support(
        evaluator,
        dt_node,
        data.duration_days,
        "Total_Duration_43_Days_With_URL",
        "States the shutdown lasted exactly 43 full days and provides a URL that confirms this duration.",
        additional_instruction="Verify that the page(s) explicitly state the shutdown lasted 43 days (accept 'forty-three'). If the page provides start and end dates, it should be consistent with a 43-day span.",
    )

    # Start date Oct 1, 2025
    await _add_fact_with_support(
        evaluator,
        dt_node,
        data.start_date,
        "Start_Date_Oct_1_2025_With_URL",
        "States the shutdown began on October 1, 2025 and provides a URL that confirms the start date.",
        additional_instruction="Allow minor date formatting variants (e.g., 'Oct. 1, 2025'); confirm the start date refers to the 2025 federal shutdown.",
    )

    # Start at beginning of FY 2026
    await _add_fact_with_support(
        evaluator,
        dt_node,
        data.start_of_fy2026,
        "Start_At_Beginning_Of_FY2026_With_URL",
        "States the shutdown began at the start of fiscal year 2026 and provides a URL that confirms this fiscal-year context.",
        additional_instruction="Confirm that FY 2026 starts on 10/1/2025 and that the shutdown's start aligns with the first day of FY 2026.",
    )

    # End date Nov 12, 2025
    await _add_fact_with_support(
        evaluator,
        dt_node,
        data.end_date,
        "End_Date_Nov_12_2025_With_URL",
        "States the shutdown ended on November 12, 2025 and provides a URL that confirms the end date.",
        additional_instruction="Allow minor date formatting variants (e.g., 'Nov. 12, 2025'); confirm the end date refers to the 2025 shutdown.",
    )

    # Longest date Nov 5, 2025 (day 36)
    await _add_fact_with_support(
        evaluator,
        dt_node,
        data.longest_date,
        "Longest_Shutdown_Date_Nov_5_2025_Day_36_With_URL",
        "States that on November 5, 2025 (day 36), the shutdown officially became the longest in U.S. history by surpassing the prior record, and provides a URL that confirms this.",
        additional_instruction="Confirm the source explicitly notes that on Nov 5, 2025 (day 36) it surpassed the previous record to become the longest shutdown.",
    )


async def build_historical_context_section(
    evaluator: Evaluator,
    parent,
    data: ShutdownInfoExtraction,
) -> None:
    hist_node = evaluator.add_parallel(
        id="Historical_Context",
        desc="Longest-shutdown claim and prior record details; each fact must include a URL that confirms it.",
        parent=parent,
        critical=True,
    )

    await _add_fact_with_support(
        evaluator,
        hist_node,
        data.confirmed_longest,
        "Confirmed_Longest_In_US_History_With_URL",
        "Confirms the 2025 shutdown was the longest U.S. government shutdown in history and provides a URL that confirms this claim.",
        additional_instruction="Confirm the page explicitly says the 2025 shutdown is the longest in U.S. history.",
    )

    await _add_fact_with_support(
        evaluator,
        hist_node,
        data.previous_record_duration,
        "Previous_Record_Duration_35_Days_With_URL",
        "States the previous record lasted 35 days and provides a URL that confirms this duration.",
        additional_instruction="The page should clearly indicate that the previous record shutdown lasted 35 days.",
    )

    await _add_fact_with_support(
        evaluator,
        hist_node,
        data.previous_record_timeframe,
        "Previous_Record_Timeframe_Dec_2018_to_Jan_2019_With_URL",
        "States the previous record occurred from December 2018 to January 2019 and provides a URL that confirms this timeframe.",
        additional_instruction="Accept reasonable phrasing variants (e.g., 'Dec. 2018–Jan. 2019').",
    )

    await _add_fact_with_support(
        evaluator,
        hist_node,
        data.previous_record_administration,
        "Previous_Record_Administration_With_URL",
        "Identifies the presidential administration during which the previous record shutdown occurred and provides a URL that confirms the stated administration (do not hard-code the administration name in the rubric).",
        additional_instruction="Ensure the page names the presidential administration for the 2018–2019 shutdown (e.g., 'Trump administration').",
    )


async def build_workforce_impact_section(
    evaluator: Evaluator,
    parent,
    data: ShutdownInfoExtraction,
) -> None:
    wf_node = evaluator.add_parallel(
        id="Workforce_Impact",
        desc="Federal workforce impacts; each fact must include a URL that confirms it.",
        parent=parent,
        critical=True,
    )

    await _add_fact_with_support(
        evaluator,
        wf_node,
        data.furloughed_employees,
        "Furloughed_670k_to_700k_With_URL",
        "States approximately 670,000–700,000 federal employees were furloughed and provides a URL that confirms this estimate.",
        additional_instruction="Approximate ranges and rounding are acceptable if consistent with the source (e.g., 670k–700k).",
    )

    await _add_fact_with_support(
        evaluator,
        wf_node,
        data.worked_without_pay,
        "Worked_Without_Pay_Approx_730k_With_URL",
        "States approximately 730,000 federal employees worked without pay and provides a URL that confirms this estimate.",
        additional_instruction="Numbers may be approximate; accept minor rounding differences if the sense matches the source (e.g., 'about 730,000').",
    )

    await _add_fact_with_support(
        evaluator,
        wf_node,
        data.total_affected,
        "Total_Affected_Approx_1_4_Million_With_URL",
        "States approximately 1.4 million total federal workers were affected and provides a URL that confirms this estimate.",
        additional_instruction="Accept 'about/approximately' phrasing and minor rounding; verify the number refers to total workers affected in 2025 shutdown context.",
    )


async def build_economic_impact_section(
    evaluator: Evaluator,
    parent,
    data: ShutdownInfoExtraction,
) -> None:
    econ_node = evaluator.add_parallel(
        id="Economic_Impact",
        desc="Economic impact details; each fact must include a URL that confirms it.",
        parent=parent,
        critical=True,
    )

    await _add_fact_with_support(
        evaluator,
        econ_node,
        data.cbo_gdp_impact_q4,
        "CBO_GDP_Impact_Q4_1_to_2_pp_With_URL",
        "States the CBO estimated a 1.0–2.0 percentage point reduction in Q4 GDP growth and provides a URL that confirms the CBO estimate.",
        additional_instruction="Confirm that the cited page attributes the 1.0–2.0 percentage point Q4 GDP growth reduction estimate specifically to CBO and to the 2025 shutdown.",
    )

    await _add_fact_with_support(
        evaluator,
        econ_node,
        data.total_economic_loss,
        "Total_Economic_Loss_Quantified_Estimate_With_URL",
        "Provides at least one quantified estimate of total economic loss attributable to the shutdown and provides a URL that confirms that estimate.",
        additional_instruction="Any credible quantified estimate is acceptable (e.g., from CBO or reputable institutions); confirm it refers to the 2025 shutdown’s total economic loss.",
    )


async def build_resolution_section(
    evaluator: Evaluator,
    parent,
    data: ShutdownInfoExtraction,
) -> None:
    res_node = evaluator.add_parallel(
        id="Resolution",
        desc="How the shutdown ended, including legislation type, presidential signing, and funding duration; each fact must include a URL that confirms it.",
        parent=parent,
        critical=True,
    )

    await _add_fact_with_support(
        evaluator,
        res_node,
        data.resolution_type,
        "Legislation_Type_Continuing_Resolution_With_URL",
        "States Congress passed a continuing resolution to reopen the government and provides a URL that confirms the legislation type.",
        additional_instruction="Verify that the page explicitly identifies the legislation as a 'continuing resolution' ending the 2025 shutdown.",
    )

    await _add_fact_with_support(
        evaluator,
        res_node,
        data.resolution_signed_by_president,
        "President_Trump_Signed_Ending_Legislation_With_URL",
        "Confirms President Trump signed the legislation ending the shutdown and provides a URL that confirms this fact.",
        additional_instruction="Confirm that President Donald Trump signed the ending legislation; ensure it refers to the 2025 shutdown resolution.",
    )

    await _add_fact_with_support(
        evaluator,
        res_node,
        data.resolution_signing_date,
        "Signing_Date_Nov_12_2025_With_URL",
        "States President Trump signed the ending legislation on November 12, 2025 and provides a URL that confirms the signing date.",
        additional_instruction="Allow minor date formatting variants (e.g., 'Nov. 12, 2025'); confirm the date is the signing date of the ending legislation.",
    )

    await _add_fact_with_support(
        evaluator,
        res_node,
        data.resolution_funding_duration,
        "Funding_Duration_Provided_By_Resolution_With_URL",
        "States the duration of funding provided by the resolution (duration or 'through' date) and provides a URL that confirms the stated funding duration/through date.",
        additional_instruction="Confirm the resolution’s funding duration or 'through' date as stated on the page; ensure it is clearly tied to the bill that ended the shutdown.",
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
        strategy=AggregationStrategy.PARALLEL,  # We will build a critical top-level aggregator under root
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

    # 1) Extract structured info from the answer
    extraction: ShutdownInfoExtraction = await evaluator.extract(
        prompt=prompt_extract_shutdown_info(),
        template_class=ShutdownInfoExtraction,
        extraction_name="shutdown_2025_extraction",
    )

    # 2) Prepare top-level critical aggregator (mirrors rubric root)
    main = evaluator.add_parallel(
        id="2025_Government_Shutdown_Information",
        desc="Complete and accurate information about the 2025 U.S. federal government shutdown. Each requested fact is stated and is accompanied by at least one reference URL from a credible source that confirms that specific fact.",
        parent=root,
        critical=True,
    )

    # Collect all URLs across facts for the credibility check
    all_urls = _collect_all_urls(extraction)
    evaluator.add_custom_info(
        info={"total_urls": len(all_urls), "urls": all_urls},
        info_type="url_collection",
        info_name="all_reference_urls",
    )

    # 3) Build and verify sections (run credibility first to avoid precondition skips)
    await build_source_credibility_section(evaluator, main, all_urls)
    await build_duration_timeline_section(evaluator, main, extraction)
    await build_historical_context_section(evaluator, main, extraction)
    await build_workforce_impact_section(evaluator, main, extraction)
    await build_economic_impact_section(evaluator, main, extraction)
    await build_resolution_section(evaluator, main, extraction)

    # 4) Return evaluation summary
    return evaluator.get_summary()