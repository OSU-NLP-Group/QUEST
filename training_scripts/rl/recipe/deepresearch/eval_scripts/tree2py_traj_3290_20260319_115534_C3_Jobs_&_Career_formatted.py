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
TASK_ID = "beardsley_transition_timeline"
TASK_DESCRIPTION = (
    "Scott C. Beardsley transitioned from a long consulting career to become a university president. "
    "Calculate the total number of years between when he first joined McKinsey & Company and when he officially assumed the role of President of the University of Virginia. "
    "In your answer, provide: 1. The year Beardsley joined McKinsey & Company and which office location he initially joined, "
    "2. The total number of years he worked at McKinsey & Company, "
    "3. The year and degree title of the doctoral qualification he earned from the University of Pennsylvania during his transition to higher education, "
    "4. The month and year when he became Dean of the UVA Darden School of Business, "
    "5. The exact date (month, day, and year) when he officially assumed the presidency of the University of Virginia, "
    "6. The total duration in years from his McKinsey start to his UVA presidency assumption. "
    "For each factual claim, provide at least one supporting URL reference."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class McKinseyJoin(BaseModel):
    year: Optional[str] = None
    initial_office: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class McKinseyTenure(BaseModel):
    years: Optional[str] = None
    end_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class YearClaim(BaseModel):
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class YearTitleClaim(BaseModel):
    year: Optional[str] = None
    title: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MonthYearClaim(BaseModel):
    month: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DateClaim(BaseModel):
    date: Optional[str] = None  # e.g., "January 1, 2026" or "01/01/2026"
    sources: List[str] = Field(default_factory=list)


class OrdinalClaim(BaseModel):
    ordinal: Optional[str] = None  # e.g., "10th" or "tenth"
    sources: List[str] = Field(default_factory=list)


class LocationClaim(BaseModel):
    location: Optional[str] = None  # e.g., "Charlottesville, Virginia"
    sources: List[str] = Field(default_factory=list)


class ValueClaim(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ComputationClaim(BaseModel):
    duration_years: Optional[str] = None  # numeric as string; e.g., "37"
    shows_calculation: Optional[bool] = None  # True if the answer explicitly shows the subtraction or equivalent


class BeardsleyTimelineExtraction(BaseModel):
    mckinsey_join: Optional[McKinseyJoin] = None
    mckinsey_tenure: Optional[McKinseyTenure] = None

    mckinsey_transfer_brussels: Optional[YearClaim] = None
    mckinsey_partner: Optional[YearClaim] = None
    mckinsey_senior_partner: Optional[YearClaim] = None

    doctoral_upenn: Optional[YearTitleClaim] = None

    darden_dean_start: Optional[MonthYearClaim] = None
    darden_established: Optional[YearClaim] = None

    uva_presidency: Optional[DateClaim] = None
    uva_presidency_ordinal: Optional[OrdinalClaim] = None
    uva_location: Optional[LocationClaim] = None

    stat_deloitte_2025_genz_leadership_goal_6pct: Optional[ValueClaim] = None
    stat_randstad_2025_genz_tenure_1_1_years: Optional[ValueClaim] = None

    computed_duration: Optional[ComputationClaim] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_timeline() -> str:
    return """
Extract the following fields exactly as stated in the answer. For every factual claim that references external information, also extract the supporting URL(s) explicitly cited in the answer text. If a field is not present, set it to null or an empty list as appropriate.

1) mckinsey_join:
   - year: the year Scott C. Beardsley joined McKinsey & Company (e.g., "1989")
   - initial_office: the office location he initially joined (e.g., "San Francisco")
   - sources: list of URLs supporting this join year and initial office

2) mckinsey_tenure:
   - years: total number of years he worked at McKinsey (e.g., "26 years" or "26")
   - end_year: the year his McKinsey tenure ended (e.g., "2015")
   - sources: list of URLs supporting tenure length and end year

3) mckinsey_transfer_brussels:
   - year: the year he transferred to McKinsey's Brussels office (e.g., "1991")
   - sources: list of URLs

4) mckinsey_partner:
   - year: the year he became a partner at McKinsey (e.g., "1995")
   - sources: list of URLs

5) mckinsey_senior_partner:
   - year: the year he became a senior partner (director) at McKinsey (e.g., "2000")
   - sources: list of URLs

6) doctoral_upenn:
   - year: year of his doctoral qualification from the University of Pennsylvania (e.g., "2015")
   - title: exact degree title (e.g., "EdD in Higher Education Management")
   - sources: list of URLs

7) darden_dean_start:
   - month: month he became Dean of UVA Darden (e.g., "August")
   - year: year he became Dean (e.g., "2015")
   - sources: list of URLs

8) darden_established:
   - year: the year the UVA Darden School of Business was established (e.g., "1955")
   - sources: list of URLs

9) uva_presidency:
   - date: the exact date he officially assumed the UVA presidency (e.g., "January 1, 2026" or "01/01/2026")
   - sources: list of URLs

10) uva_presidency_ordinal:
    - ordinal: the ordinal number of his UVA presidency (e.g., "10th" or "tenth")
    - sources: list of URLs

11) uva_location:
    - location: the location of the University of Virginia (e.g., "Charlottesville, Virginia")
    - sources: list of URLs

12) stat_deloitte_2025_genz_leadership_goal_6pct:
    - value: the statistic value as stated (e.g., "6%")
    - sources: list of URLs that directly support this statistic

13) stat_randstad_2025_genz_tenure_1_1_years:
    - value: the statistic value as stated (e.g., "1.1 years")
    - sources: list of URLs that directly support this statistic

14) computed_duration:
    - duration_years: the total number of years computed from joining McKinsey to assuming UVA presidency (numeric as string)
    - shows_calculation: true if the answer explicitly shows the calculation (e.g., "2026 - 1989 = 37") or otherwise clearly demonstrates the arithmetic; false otherwise

SPECIAL URL EXTRACTION RULES:
- Only extract URLs that are explicitly present in the answer text. If no URL is provided for a claim, return an empty list for its 'sources'.
- Accept plain URLs and markdown links; extract the actual link target.
- Do not invent or infer URLs.

Return a single JSON object conforming to the BeardsleyTimelineExtraction schema.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_valid_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str) and u.strip() and ("http://" in u or "https://" in u):
            return True
    return False


def parse_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    try:
        # Extract the first integer-like token
        import re
        m = re.search(r"(-?\d{1,4})", s.replace(",", ""))
        if m:
            return int(m.group(1))
        return None
    except Exception:
        return None


def parse_year_from_date_str(s: Optional[str]) -> Optional[int]:
    # Attempt to find a 4-digit year in a date string like "January 1, 2026" or "01/01/2026"
    if s is None:
        return None
    import re
    m = re.search(r"\b(19|20)\d{2}\b", s)
    if m:
        return int(m.group(0))
    return None


def month_year_str(month: Optional[str], year: Optional[str]) -> Optional[str]:
    if month and year:
        return f"{month} {year}"
    return None


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_claim_with_sources(
    evaluator: Evaluator,
    parent_node,
    base_id: str,
    base_desc: str,
    value_present: bool,
    sources: Optional[List[str]],
    claim_text: str,
    additional_instruction: str = "",
) -> None:
    """
    For a required fact that must have >=1 supporting URL:
    - Create a sequential container under parent_node (critical).
    - First add a critical existence/sources check as a custom node.
    - Then verify the claim against the provided URLs.
    """
    container = evaluator.add_sequential(
        id=base_id,
        desc=base_desc,
        parent=parent_node,
        critical=True,
    )

    exists_and_has_source = value_present and has_valid_urls(sources)
    evaluator.add_custom_node(
        result=exists_and_has_source,
        id=f"{base_id}_sources_present",
        desc=f"{base_desc} — at least one supporting URL is provided and required value(s) are present",
        parent=container,
        critical=True,
    )

    supported_node = evaluator.add_leaf(
        id=f"{base_id}_supported",
        desc=f"{base_desc} — supported by cited source(s)",
        parent=container,
        critical=True,
    )

    await evaluator.verify(
        claim=claim_text,
        node=supported_node,
        sources=sources or [],
        additional_instruction=additional_instruction or "None",
    )


# --------------------------------------------------------------------------- #
# Main verification builder                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: BeardsleyTimelineExtraction) -> None:
    # Root under evaluator.root to mirror rubric structure
    timeline_node = evaluator.add_parallel(
        id="Career_Transition_Timeline",
        desc="Verify the required timeline facts and computed duration. Each required factual claim should be accompanied by at least one supporting URL reference.",
        parent=evaluator.root,
        critical=True,
    )

    # 1) McKinsey join year and initial office
    mj = extraction.mckinsey_join or McKinseyJoin()
    join_present = bool(mj.year and mj.initial_office)
    await verify_claim_with_sources(
        evaluator=evaluator,
        parent_node=timeline_node,
        base_id="McKinsey_Join_Year_And_Initial_Office",
        base_desc="Answer states the year Beardsley joined McKinsey & Company and the initial office location (as required), with ≥1 supporting URL.",
        value_present=join_present,
        sources=mj.sources,
        claim_text=f"Scott C. Beardsley joined McKinsey & Company in {mj.year} at the {mj.initial_office} office.",
        additional_instruction="Confirm both the year and the initial office location appear on the cited source(s); allow reasonable variants like 'San Francisco, CA' or 'San Francisco office'.",
    )

    # 2) McKinsey tenure and end year
    mt = extraction.mckinsey_tenure or McKinseyTenure()
    tenure_present = bool(mt.years and mt.end_year)
    await verify_claim_with_sources(
        evaluator=evaluator,
        parent_node=timeline_node,
        base_id="McKinsey_Tenure_And_End_Year",
        base_desc="Answer states Beardsley worked at McKinsey for exactly 26 years and that his McKinsey tenure ended in 2015 (sufficient to substantiate the 1989–2015 span without re-checking the start year here), with ≥1 supporting URL.",
        value_present=tenure_present,
        sources=mt.sources,
        claim_text=f"Scott C. Beardsley worked at McKinsey for {mt.years} and his tenure ended in {mt.end_year}.",
        additional_instruction="Check that the page(s) explicitly indicate both the total years at McKinsey and the end year (e.g., retired or left the firm in that year).",
    )

    # 3) McKinsey milestones (parallel group)
    milestones_node = evaluator.add_parallel(
        id="McKinsey_Milestones_From_Constraints",
        desc="Verify additional McKinsey career milestones explicitly listed in constraints; each must include ≥1 supporting URL.",
        parent=timeline_node,
        critical=True,
    )

    # 3.a) Transfer to Brussels in 1991
    tb = extraction.mckinsey_transfer_brussels or YearClaim()
    tb_present = bool(tb.year)
    await verify_claim_with_sources(
        evaluator=evaluator,
        parent_node=milestones_node,
        base_id="Transfer_To_Brussels_1991",
        base_desc="Answer states he transferred to McKinsey's Brussels office in 1991, with ≥1 supporting URL.",
        value_present=tb_present,
        sources=tb.sources,
        claim_text=f"Scott C. Beardsley transferred to McKinsey's Brussels office in {tb.year}.",
        additional_instruction="The page must clearly mention 'Brussels' and the transfer year.",
    )

    # 3.b) Became partner in 1995
    bp = extraction.mckinsey_partner or YearClaim()
    bp_present = bool(bp.year)
    await verify_claim_with_sources(
        evaluator=evaluator,
        parent_node=milestones_node,
        base_id="Became_Partner_1995",
        base_desc="Answer states he became a McKinsey partner in 1995, with ≥1 supporting URL.",
        value_present=bp_present,
        sources=bp.sources,
        claim_text=f"Scott C. Beardsley became a partner at McKinsey in {bp.year}.",
        additional_instruction="The page should indicate promotion to partner in the specified year.",
    )

    # 3.c) Became senior partner (director) in 2000
    sp = extraction.mckinsey_senior_partner or YearClaim()
    sp_present = bool(sp.year)
    await verify_claim_with_sources(
        evaluator=evaluator,
        parent_node=milestones_node,
        base_id="Became_Senior_Partner_Director_2000",
        base_desc="Answer states he became a McKinsey senior partner (director) in 2000, with ≥1 supporting URL.",
        value_present=sp_present,
        sources=sp.sources,
        claim_text=f"Scott C. Beardsley became a senior partner (director) at McKinsey in {sp.year}.",
        additional_instruction="Allow synonyms like 'director' for senior partner.",
    )

    # 4) Doctoral qualification at UPenn (year and title)
    dq = extraction.doctoral_upenn or YearTitleClaim()
    dq_present = bool(dq.year and dq.title)
    await verify_claim_with_sources(
        evaluator=evaluator,
        parent_node=timeline_node,
        base_id="Doctoral_Qualification_Upenn_Year_And_Title",
        base_desc="Answer states the year and degree title of the doctoral qualification earned from the University of Pennsylvania (2015; EdD in Higher Education Management), with ≥1 supporting URL.",
        value_present=dq_present,
        sources=dq.sources,
        claim_text=f"In {dq.year}, Scott C. Beardsley earned an {dq.title} from the University of Pennsylvania.",
        additional_instruction="Confirm both the year and the exact degree title (e.g., EdD in Higher Education Management). Minor formatting variations are acceptable.",
    )

    # 5) Darden Dean start (month and year)
    dd = extraction.darden_dean_start or MonthYearClaim()
    dd_text = month_year_str(dd.month, dd.year)
    dd_present = bool(dd_text)
    await verify_claim_with_sources(
        evaluator=evaluator,
        parent_node=timeline_node,
        base_id="Darden_Dean_Start_Month_And_Year",
        base_desc="Answer states the month and year he became Dean of the UVA Darden School of Business (August 2015), with ≥1 supporting URL.",
        value_present=dd_present,
        sources=dd.sources,
        claim_text=f"Scott C. Beardsley became Dean of the University of Virginia's Darden School of Business in {dd_text}.",
        additional_instruction="The page must reference both the month and the year of the dean appointment.",
    )

    # 6) Darden established in 1955
    de = extraction.darden_established or YearClaim()
    de_present = bool(de.year)
    await verify_claim_with_sources(
        evaluator=evaluator,
        parent_node=timeline_node,
        base_id="Darden_Established_1955",
        base_desc="Answer states the UVA Darden School of Business was established in 1955, with ≥1 supporting URL.",
        value_present=de_present,
        sources=de.sources,
        claim_text=f"The UVA Darden School of Business was established in {de.year}.",
        additional_instruction="The page must clearly indicate the founding/established year.",
    )

    # 7) UVA presidency assumption exact date
    up = extraction.uva_presidency or DateClaim()
    up_present = bool(up.date)
    await verify_claim_with_sources(
        evaluator=evaluator,
        parent_node=timeline_node,
        base_id="UVA_Presidency_Assumption_Exact_Date",
        base_desc="Answer states the exact date (month/day/year) when he officially assumed the UVA presidency (January 1, 2026), with ≥1 supporting URL.",
        value_present=up_present,
        sources=up.sources,
        claim_text=f"Scott C. Beardsley officially assumed the presidency of the University of Virginia on {up.date}.",
        additional_instruction="Confirm the exact date of assuming office (month, day, year).",
    )

    # 8) UVA presidency ordinal (10th)
    uo = extraction.uva_presidency_ordinal or OrdinalClaim()
    uo_present = bool(uo.ordinal)
    await verify_claim_with_sources(
        evaluator=evaluator,
        parent_node=timeline_node,
        base_id="UVA_Presidency_Ordinal_10th",
        base_desc="Answer states that he became the 10th President of the University of Virginia, with ≥1 supporting URL.",
        value_present=uo_present,
        sources=uo.sources,
        claim_text=f"Scott C. Beardsley is the {uo.ordinal} President of the University of Virginia.",
        additional_instruction="Allow variants like 'tenth' vs '10th'.",
    )

    # 9) UVA location (Charlottesville, VA)
    ul = extraction.uva_location or LocationClaim()
    ul_present = bool(ul.location)
    await verify_claim_with_sources(
        evaluator=evaluator,
        parent_node=timeline_node,
        base_id="UVA_Location_Charlottesville_VA",
        base_desc="Answer states the University of Virginia is located in Charlottesville, Virginia, with ≥1 supporting URL.",
        value_present=ul_present,
        sources=ul.sources,
        claim_text=f"The University of Virginia is located in {ul.location}.",
        additional_instruction="The page must clearly state the location of the university.",
    )

    # 10) Additional constraint statistics (parallel)
    stats_node = evaluator.add_parallel(
        id="Additional_Constraint_Statistics",
        desc="Verify extra explicit constraint statistics; each must include ≥1 supporting URL.",
        parent=timeline_node,
        critical=True,
    )

    # 10.a) Deloitte 2025 6% Gen Z leadership goal
    sd = extraction.stat_deloitte_2025_genz_leadership_goal_6pct or ValueClaim()
    sd_present = bool(sd.value)
    await verify_claim_with_sources(
        evaluator=evaluator,
        parent_node=stats_node,
        base_id="Deloitte_2025_GenZ_Leadership_Goal_6pct",
        base_desc="Answer states the Deloitte 2025 statistic that only 6% of Gen Z workers say their primary career goal is to reach a leadership position, with ≥1 supporting URL.",
        value_present=sd_present,
        sources=sd.sources,
        claim_text=f"According to a 2025 Deloitte report or survey, only {sd.value} of Gen Z workers say their primary career goal is to reach a leadership position.",
        additional_instruction="The page should be a Deloitte 2025 source or credible citation explicitly mentioning the 6% figure for Gen Z primary leadership goal.",
    )

    # 10.b) Randstad 2025 Gen Z tenure 1.1 years
    sr = extraction.stat_randstad_2025_genz_tenure_1_1_years or ValueClaim()
    sr_present = bool(sr.value)
    await verify_claim_with_sources(
        evaluator=evaluator,
        parent_node=stats_node,
        base_id="Randstad_2025_GenZ_Tenure_1_1_years",
        base_desc="Answer states the Randstad 2025 statistic that Gen Z's average job tenure during their first five years of career is 1.1 years, with ≥1 supporting URL.",
        value_present=sr_present,
        sources=sr.sources,
        claim_text=f"According to a 2025 Randstad report or study, Gen Z's average job tenure during their first five years of career is {sr.value}.",
        additional_instruction="The page should be a Randstad 2025 source or credible citation explicitly mentioning the '1.1 years' tenure metric.",
    )

    # 11) Computed duration: from McKinsey start year to UVA presidency assumption year
    comp_node = evaluator.add_sequential(
        id="Computed_Duration_McKinseyStart_To_UVAPresidency",
        desc="Answer correctly computes the total duration in years from the McKinsey start year to the UVA presidency assumption year using the stated endpoints, and shows the calculation.",
        parent=timeline_node,
        critical=True,
    )

    # Derive integers for start and presidency years
    start_year_int = parse_int(mj.year)
    presidency_year_int = parse_year_from_date_str(up.date)
    extracted_duration_int = parse_int((extraction.computed_duration.duration_years if extraction.computed_duration else None))

    # 11.a) Math check (custom critical leaf)
    math_ok = (
        start_year_int is not None
        and presidency_year_int is not None
        and extracted_duration_int is not None
        and (presidency_year_int - start_year_int == extracted_duration_int)
    )
    evaluator.add_custom_node(
        result=bool(math_ok),
        id="Computed_Duration_McKinseyStart_To_UVAPresidency_math",
        desc=f"Computed duration equals presidency_year - start_year ({extracted_duration_int} = {presidency_year_int} - {start_year_int})",
        parent=comp_node,
        critical=True,
    )

    # 11.b) Shows calculation explicitly in the answer (simple verify on answer text)
    shows_calc_leaf = evaluator.add_leaf(
        id="Computed_Duration_McKinseyStart_To_UVAPresidency_shows_calculation",
        desc="Answer explicitly shows the calculation (e.g., '2026 - 1989 = 37') or an equivalent clear arithmetic explanation.",
        parent=comp_node,
        critical=True,
    )
    # Build a flexible claim referencing the concrete numbers if available
    if start_year_int is not None and presidency_year_int is not None and extracted_duration_int is not None:
        calc_claim = (
            f"The answer explicitly shows or clearly describes the arithmetic from {presidency_year_int} minus {start_year_int} equals {extracted_duration_int}, "
            f"for example as '{presidency_year_int} - {start_year_int} = {extracted_duration_int}', or an equivalent unambiguous calculation."
        )
    else:
        # Fall back to a generic calculation visibility check
        calc_claim = (
            "The answer explicitly shows or clearly describes the subtraction or arithmetic used to compute the duration in years between the McKinsey start year and the UVA presidency assumption year."
        )

    await evaluator.verify(
        claim=calc_claim,
        node=shows_calc_leaf,
        sources=None,
        additional_instruction="Look for explicit numeric subtraction or a plainly described calculation. If no clear calculation is shown, mark as Incorrect.",
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # top-level aggregation; detailed structure added under this
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_timeline(),
        template_class=BeardsleyTimelineExtraction,
        extraction_name="timeline_extraction",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return structured summary
    return evaluator.get_summary()