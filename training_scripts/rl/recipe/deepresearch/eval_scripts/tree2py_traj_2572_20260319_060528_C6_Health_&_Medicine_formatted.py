import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "measles_vax_study_2026"
TASK_DESCRIPTION = (
    "A public health researcher is conducting a comparative analysis of measles vaccination programs and outbreak "
    "preparedness across different countries. The analysis includes:\n"
    "Part A: Identify 3 countries that (1) have 2024 MCV1/MMR1 coverage ≥93%, (2) administer the first MMR/MCV dose at "
    "15 months of age or earlier, and (3) DO NOT include a universal HepB birth dose in the routine schedule. Provide "
    "coverage %, dose age, and HepB birth dose policy with URLs.\n"
    "Part B: For a case with measles rash onset on March 1, 2026: compute contagious period start/end, its duration, "
    "and the secondary case rash-onset window accounting for incubation.\n"
    "Part C: For Denmark and Costa Rica: provide 2024 MMR coverage, compare to 95% herd immunity threshold, and assess "
    "the risk of sustained transmission. Provide URLs for all factual data."
)


# -----------------------------------------------------------------------------
# Extraction data models
# -----------------------------------------------------------------------------
class CountryCriteriaItem(BaseModel):
    country: Optional[str] = None
    coverage_2024_text: Optional[str] = None
    coverage_year_text: Optional[str] = None
    mmr1_age_text: Optional[str] = None
    mmr1_age_months_earliest: Optional[int] = None  # If a range is given (e.g., 12–15 months), use the earliest
    hepB_birth_dose_policy_text: Optional[str] = None
    hepB_birth_dose_universal: Optional[bool] = None  # True only if universal birth dose within 24 hours for all infants

    coverage_source_urls: List[str] = Field(default_factory=list)
    schedule_source_urls: List[str] = Field(default_factory=list)
    hepB_policy_source_urls: List[str] = Field(default_factory=list)


class CountriesExtraction(BaseModel):
    countries: List[CountryCriteriaItem] = Field(default_factory=list)


class TransmissionExtraction(BaseModel):
    # From answer
    rash_onset_date_text: Optional[str] = None  # e.g., "March 1, 2026"
    contagious_start_date_text: Optional[str] = None  # e.g., "February 26, 2026"
    contagious_end_date_text: Optional[str] = None  # e.g., "March 5, 2026"
    contagious_duration_days: Optional[str] = None  # e.g., "9 days" or "9"

    incubation_period_text: Optional[str] = None  # e.g., "7-14 days after exposure"
    secondary_window_start_date_text: Optional[str] = None  # e.g., "March 5, 2026"
    secondary_window_end_date_text: Optional[str] = None  # e.g., "March 19, 2026"

    transmission_reference_urls: List[str] = Field(default_factory=list)  # contagious period references
    incubation_reference_urls: List[str] = Field(default_factory=list)  # incubation/rash timing references


class CountryRiskInfo(BaseModel):
    coverage_2024_text: Optional[str] = None
    coverage_source_urls: List[str] = Field(default_factory=list)

    threshold_comparison_made: Optional[bool] = None  # True if the answer explicitly compares coverage to the 95% threshold
    threshold_reference_urls: List[str] = Field(default_factory=list)

    risk_assessment_text: Optional[str] = None  # e.g., "elevated risk", "not elevated", "below herd immunity threshold", etc.


class RiskAssessmentExtraction(BaseModel):
    denmark: Optional[CountryRiskInfo] = None
    costa_rica: Optional[CountryRiskInfo] = None


# -----------------------------------------------------------------------------
# Extraction prompts
# -----------------------------------------------------------------------------
def prompt_extract_countries() -> str:
    return """
Extract up to THREE countries presented in the answer that are claimed to meet ALL of these criteria:
1) 2024 measles-containing vaccine (MCV1) or MMR first-dose coverage is ≥93%
2) The first MMR/MCV dose is given at 15 months of age or earlier in the routine childhood schedule
3) The routine schedule does NOT include a universal hepatitis B birth dose (i.e., not given to all infants within 24 hours of birth). Targeted birth dose for at-risk infants DOES NOT count as universal.

For each country in the answer, extract the following fields:
- country: country name as stated
- coverage_2024_text: the stated 2024 MCV1/MMR1 coverage (verbatim, include the % if present)
- coverage_year_text: the year(s) associated with the stated coverage (verbatim; ensure '2024' if present)
- mmr1_age_text: the stated timing for the first MMR/MCV dose (verbatim, e.g., '12 months', '12–15 months')
- mmr1_age_months_earliest: the earliest month in integer if a range is given (e.g., '12–15 months' -> 12); if a single month is given, that month as an integer; otherwise null
- hepB_birth_dose_policy_text: verbatim statement of the HepB birth dose policy
- hepB_birth_dose_universal: true only if a universal HepB birth dose within 24 hours for ALL infants is included in the schedule; false if NOT universal; null if unclear

- coverage_source_urls: list all URLs cited for the coverage figure (2024 MCV1/MMR1). Return an empty list if none are provided.
- schedule_source_urls: list all URLs cited for the vaccination schedule/age. Return an empty list if none are provided.
- hepB_policy_source_urls: list all URLs cited for HepB birth dose policy. Return an empty list if none are provided.

Important:
- Do NOT invent URLs. Only include URLs explicitly present in the answer.
- Preserve percentages and years exactly as stated in the answer text.
- If the answer provides more than 3 countries, extract the first three that appear.
"""


def prompt_extract_transmission() -> str:
    return """
From the answer, extract the measles transmission calculations for the case with rash onset on March 1, 2026. Provide:

- rash_onset_date_text: the rash onset date as stated (verbatim)
- contagious_start_date_text: the stated contagious period START date (verbatim)
- contagious_end_date_text: the stated contagious period END date (verbatim)
- contagious_duration_days: the stated total contagious period duration (e.g., '9 days' or '9')

- incubation_period_text: the stated incubation period to rash onset (verbatim, e.g., '7–14 days after exposure')
- secondary_window_start_date_text: the stated earliest secondary rash-onset date (verbatim)
- secondary_window_end_date_text: the stated latest secondary rash-onset date (verbatim)

- transmission_reference_urls: list all URLs supporting the measles contagious period rule (e.g., 4 days before through 4 days after rash onset). Empty list if none.
- incubation_reference_urls: list all URLs supporting the incubation/rash onset timing window. Empty list if none.

Do NOT invent any URLs; include only those shown in the answer.
"""


def prompt_extract_risk_assessment() -> str:
    return """
Extract the outbreak risk assessment details explicitly provided in the answer for:
- Denmark
- Costa Rica

For EACH country, extract:
- coverage_2024_text: the stated 2024 MMR/MCV1 coverage (verbatim, include ‘%’ if provided)
- coverage_source_urls: list of URLs cited for that coverage (empty list if none)
- threshold_comparison_made: true if the answer explicitly compares the country’s coverage to a 95% herd immunity threshold for measles; false otherwise
- threshold_reference_urls: list of URLs cited for the herd immunity threshold (empty list if none)
- risk_assessment_text: the stated qualitative risk assessment (verbatim), e.g., 'elevated risk of sustained transmission', 'risk is low', etc.

Do NOT invent URLs. Only include those explicitly presented in the answer.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def parse_percentage(text: Optional[str]) -> Optional[float]:
    """
    Extract the first percentage value from a string.
    Examples:
        "95%" -> 95.0
        "about 94 percent" -> 94.0
        "93.5 %" -> 93.5
    """
    if not text:
        return None
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%|\b(\d{1,3}(?:\.\d+)?)\s*(?:percent|per cent)\b", text, flags=re.I)
    if not m:
        return None
    g = m.group(1) or m.group(2)
    try:
        val = float(g)
        if 0.0 <= val <= 100.0:
            return val
        return None
    except Exception:
        return None


def text_contains_2024(text: Optional[str]) -> bool:
    if not text:
        return False
    return "2024" in text


def non_empty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def safe_list(urls: Optional[List[str]]) -> List[str]:
    return urls or []


# -----------------------------------------------------------------------------
# Verification builders
# -----------------------------------------------------------------------------
async def verify_country_criteria(
    evaluator: Evaluator,
    parent_node,
    country: CountryCriteriaItem,
    idx: int,
) -> None:
    """
    Build and run verification for a single country against the Part A rubric.
    """
    label = f"Country{idx + 1}"
    country_name = country.country or f"Country {idx + 1}"

    # Country node (non-critical overall to allow partial credit across countries)
    country_node = evaluator.add_parallel(
        id=f"partA_{label}",
        desc=f"{label}: {country_name} verification (meets all specified criteria)",
        parent=parent_node,
        critical=False,
    )

    # 1) Coverage criterion (critical group)
    cov_group = evaluator.add_parallel(
        id=f"{label}_MMR_Coverage_Criterion",
        desc="Country has MMR/MCV1 first dose vaccination coverage of 93% or higher (2024 data)",
        parent=country_node,
        critical=True,
    )
    # Presence of source URL for coverage (critical)
    cov_src_node = evaluator.add_custom_node(
        result=non_empty_urls(country.coverage_source_urls),
        id=f"{label}_Coverage_Source",
        desc="Reference URL provided for coverage data",
        parent=cov_group,
        critical=True,
    )
    # Coverage value meets threshold (verify with sources; gated by source presence)
    cov_val_leaf = evaluator.add_leaf(
        id=f"{label}_Coverage_Value",
        desc="Coverage percentage is 93% or higher (2024)",
        parent=cov_group,
        critical=True,
    )
    cov_claim = (
        f"For {country_name}, the 2024 first-dose measles-containing vaccine (MCV1/MMR1) coverage is at least 93%."
    )
    await evaluator.verify(
        claim=cov_claim,
        node=cov_val_leaf,
        sources=safe_list(country.coverage_source_urls),
        additional_instruction=(
            "Verify the 2024 national-level MCV1 (or first-dose MMR) coverage is ≥93% on the cited source(s). "
            "Allow synonyms (MCV1/MMR1). Minor rounding differences are acceptable, but the 2024 figure must be ≥93%."
        ),
        extra_prerequisites=[cov_src_node],
    )

    # 2) MMR timing criterion (critical group)
    time_group = evaluator.add_parallel(
        id=f"{label}_MMR_Timing_Criterion",
        desc="Country administers the first MMR/MCV dose at 15 months of age or earlier",
        parent=country_node,
        critical=True,
    )
    # Presence of schedule URL (critical)
    time_src_node = evaluator.add_custom_node(
        result=non_empty_urls(country.schedule_source_urls),
        id=f"{label}_Timing_Source",
        desc="Reference URL provided for vaccination schedule",
        parent=time_group,
        critical=True,
    )
    # Timing value check (verify against schedule URLs; gated by source presence)
    mmr_age_text = country.mmr1_age_text or "an unspecified age"
    time_val_leaf = evaluator.add_leaf(
        id=f"{label}_Timing_Value",
        desc="MMR first dose timing is at 15 months or earlier",
        parent=time_group,
        critical=True,
    )
    time_claim = (
        f"According to the cited schedule source(s), the first dose of MMR (or MCV) in {country_name} is administered at "
        f"{mmr_age_text}, which is 15 months of age or earlier."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_val_leaf,
        sources=safe_list(country.schedule_source_urls),
        additional_instruction=(
            "Check the age for the FIRST dose of MMR or measles-containing vaccine. "
            "If a range is given (e.g., 12–15 months), use the earliest month in that range. "
            "This statement should be considered correct only if the earliest permissible age is ≤ 15 months."
        ),
        extra_prerequisites=[time_src_node],
    )

    # 3) HepB birth dose criterion (critical group)
    hep_group = evaluator.add_parallel(
        id=f"{label}_HepB_Birth_Dose_Criterion",
        desc="Country does NOT include a universal HepB birth dose in routine infant immunization",
        parent=country_node,
        critical=True,
    )
    # Presence of HepB policy URL (critical)
    hep_src_node = evaluator.add_custom_node(
        result=non_empty_urls(country.hepB_policy_source_urls),
        id=f"{label}_HepB_Source",
        desc="Reference URL provided for hepatitis B policy",
        parent=hep_group,
        critical=True,
    )
    # HepB policy verification (verify with URLs; gated by source presence)
    hep_policy_leaf = evaluator.add_leaf(
        id=f"{label}_HepB_Policy",
        desc="Confirmation that universal hepatitis B birth dose is not included",
        parent=hep_group,
        critical=True,
    )
    hep_claim = (
        f"In {country_name}, the routine infant immunization schedule does not include a UNIVERSAL hepatitis B birth dose "
        f"administered within 24 hours of birth (i.e., not given to all infants; targeted birth dose for at-risk infants "
        f"does NOT count as universal)."
    )
    await evaluator.verify(
        claim=hep_claim,
        node=hep_policy_leaf,
        sources=safe_list(country.hepB_policy_source_urls),
        additional_instruction=(
            "Confirm that a universal HepB birth dose (administered within 24 hours of birth to ALL infants) is NOT part "
            "of the routine infant schedule. Policies that offer birth dose only to infants of HBsAg-positive mothers or "
            "other at‑risk groups should be considered NOT universal."
        ),
        extra_prerequisites=[hep_src_node],
    )


async def verify_transmission_timeline(
    evaluator: Evaluator,
    parent_node,
    tx: TransmissionExtraction,
) -> None:
    """
    Build and run verification for Part B (transmission timeline and incubation window).
    """
    part_b = evaluator.add_sequential(
        id="PartB_TransmissionTimeline",
        desc="Calculation of measles contagious period and secondary case symptom window for index case (rash onset March 1, 2026)",
        parent=parent_node,
        critical=False,
    )

    # 1) Rash onset date (critical)
    rash_leaf = evaluator.add_leaf(
        id="Rash_Onset_Date",
        desc="Initial case rash onset date is correctly identified as March 1, 2026",
        parent=part_b,
        critical=True,
    )
    rash_claim = "The index case’s rash onset date is March 1, 2026."
    await evaluator.verify(
        claim=rash_claim,
        node=rash_leaf,
        additional_instruction="Accept minor formatting or equivalent date expressions (e.g., '1 March 2026').",
    )

    # 2) Contagious period (critical group)
    cont_group = evaluator.add_parallel(
        id="Contagious_Period",
        desc="Complete contagious period calculated correctly",
        parent=part_b,
        critical=True,
    )

    cont_start_leaf = evaluator.add_leaf(
        id="Contagious_Start",
        desc="Contagious period start date calculated as 4 days before rash onset (February 26, 2026)",
        parent=cont_group,
        critical=True,
    )
    start_claim = (
        "The contagious period for measles begins 4 days before rash onset; therefore it starts on February 26, 2026 "
        "for a case with rash onset on March 1, 2026."
    )
    await evaluator.verify(
        claim=start_claim,
        node=cont_start_leaf,
        additional_instruction="Focus on the date arithmetic and the conventional rule '4 days before rash onset'.",
    )

    cont_end_leaf = evaluator.add_leaf(
        id="Contagious_End",
        desc="Contagious period end date calculated as 4 days after rash onset (March 5, 2026)",
        parent=cont_group,
        critical=True,
    )
    end_claim = (
        "The contagious period for measles ends 4 days after rash onset; therefore it ends on March 5, 2026 "
        "for a case with rash onset on March 1, 2026."
    )
    await evaluator.verify(
        claim=end_claim,
        node=cont_end_leaf,
        additional_instruction="Focus on the date arithmetic and the conventional rule '4 days after rash onset'.",
    )

    cont_dur_leaf = evaluator.add_leaf(
        id="Contagious_Duration",
        desc="Total contagious period stated as 9 days",
        parent=cont_group,
        critical=True,
    )
    dur_claim = (
        "Counting inclusively from 4 days before through 4 days after rash onset results in a 9-day contagious period."
    )
    await evaluator.verify(
        claim=dur_claim,
        node=cont_dur_leaf,
        additional_instruction="Inclusive counting of the 9-day window (-4, -3, -2, -1, 0, +1, +2, +3, +4).",
    )

    # Transmission references presence (critical)
    trans_ref_leaf = evaluator.add_custom_node(
        result=non_empty_urls(tx.transmission_reference_urls),
        id="Transmission_Reference",
        desc="Reference URL provided for measles contagious period information",
        parent=cont_group,
        critical=True,
    )

    # 3) Secondary case window (critical group)
    sec_group = evaluator.add_parallel(
        id="Secondary_Case_Window",
        desc="Window for secondary case symptom development calculated correctly",
        parent=part_b,
        critical=True,
    )

    incub_leaf = evaluator.add_leaf(
        id="Incubation_Period",
        desc="Measles incubation period correctly stated as 7-14 days after exposure",
        parent=sec_group,
        critical=True,
    )
    incub_claim = "The measles incubation period to rash onset is approximately 7 to 14 days after exposure."
    await evaluator.verify(
        claim=incub_claim,
        node=incub_leaf,
        additional_instruction=(
            "Interpret 'incubation period' as time from exposure to ONSET OF RASH (exanthem), not prodromal symptoms."
        ),
    )

    sec_start_leaf = evaluator.add_leaf(
        id="Secondary_Window_Start",
        desc="Earliest possible secondary case rash onset calculated based on exposure during contagious period (March 5, 2026: 7 days after Feb 26)",
        parent=sec_group,
        critical=True,
    )
    sec_start_claim = (
        "If exposure occurred on the first contagious day (February 26, 2026), the earliest rash onset in a secondary "
        "case is 7 days later: March 5, 2026."
    )
    await evaluator.verify(
        claim=sec_start_claim,
        node=sec_start_leaf,
        additional_instruction="Focus on date arithmetic using the 7-day minimum incubation.",
    )

    sec_end_leaf = evaluator.add_leaf(
        id="Secondary_Window_End",
        desc="Latest possible secondary case rash onset calculated based on exposure during contagious period (March 19, 2026: 14 days after March 5)",
        parent=sec_group,
        critical=True,
    )
    sec_end_claim = (
        "If exposure occurred on the last contagious day (March 5, 2026), the latest rash onset in a secondary case is "
        "14 days later: March 19, 2026."
    )
    await evaluator.verify(
        claim=sec_end_claim,
        node=sec_end_leaf,
        additional_instruction="Focus on date arithmetic using the 14-day maximum incubation.",
    )

    incub_ref_leaf = evaluator.add_custom_node(
        result=non_empty_urls(tx.incubation_reference_urls),
        id="Incubation_Reference",
        desc="Reference URL provided for measles incubation period information",
        parent=sec_group,
        critical=True,
    )


async def verify_outbreak_risk_country(
    evaluator: Evaluator,
    parent_node,
    country_label: str,
    country_name: str,
    info: Optional[CountryRiskInfo],
) -> None:
    """
    Build and run verification for Part C for a specific country (Denmark or Costa Rica).
    """
    node = evaluator.add_parallel(
        id=f"{country_label}_Assessment",
        desc=f"Outbreak risk assessment for {country_name}",
        parent=parent_node,
        critical=False,
    )

    info = info or CountryRiskInfo()

    # Coverage data group (critical)
    cov_group = evaluator.add_parallel(
        id=f"{country_label}_Coverage",
        desc=f"{country_name}'s 2024 MMR vaccination coverage data provided",
        parent=node,
        critical=True,
    )
    # Coverage provided (critical) - must state data and year 2024
    cov_provided = evaluator.add_custom_node(
        result=bool(info.coverage_2024_text) and text_contains_2024(info.coverage_2024_text),
        id=f"{country_label}_Coverage_Provided",
        desc=f"{country_name}'s MMR vaccination coverage data is stated (includes 2024)",
        parent=cov_group,
        critical=True,
    )
    # Coverage source presence (critical)
    cov_src = evaluator.add_custom_node(
        result=non_empty_urls(info.coverage_source_urls),
        id=f"{country_label}_Coverage_Source",
        desc=f"Reference URL provided for {country_name}'s vaccination coverage data",
        parent=cov_group,
        critical=True,
    )

    # Threshold comparison group (critical)
    thr_group = evaluator.add_parallel(
        id=f"{country_label}_Threshold_Comparison",
        desc=f"Comparison of {country_name}'s coverage to 95% herd immunity threshold",
        parent=node,
        critical=True,
    )
    thr_stmt = evaluator.add_custom_node(
        result=bool(info.threshold_comparison_made),
        id=f"{country_label}_Threshold_Statement",
        desc="Comparison to 95% threshold is performed",
        parent=thr_group,
        critical=True,
    )
    thr_src = evaluator.add_custom_node(
        result=non_empty_urls(info.threshold_reference_urls),
        id=f"{country_label}_Threshold_Reference",
        desc="Reference URL provided for herd immunity threshold information",
        parent=thr_group,
        critical=True,
    )

    # Risk assessment (critical)
    risk_leaf = evaluator.add_leaf(
        id=f"{country_label}_Risk_Assessment",
        desc=f"{country_name}'s outbreak risk assessment is provided and logically consistent with the stated coverage data",
        parent=node,
        critical=True,
    )

    cov_pct = parse_percentage(info.coverage_2024_text)
    expected_logic = None
    if cov_pct is not None:
        expected_logic = "not elevated" if cov_pct >= 95.0 else "elevated"

    risk_text = info.risk_assessment_text or ""
    risk_claim = (
        f"The answer's risk assessment for {country_name} ('{risk_text}') is logically consistent with comparing the "
        f"stated 2024 coverage to a 95% herd immunity threshold."
    )
    add_ins = (
        "Use the rule: if MMR/MCV1 coverage ≥95%, sustained transmission risk is NOT elevated; if <95%, risk is elevated. "
        f"Based on the parsed coverage value {cov_pct if cov_pct is not None else 'unknown'}, "
        f"the expected qualitative judgment is '{expected_logic}' if known. "
        "Accept reasonable synonyms ('low risk' ~ not elevated; 'higher risk' ~ elevated)."
    )
    await evaluator.verify(
        claim=risk_claim,
        node=risk_leaf,
        additional_instruction=add_ins,
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
    Entrypoint for evaluating the measles vaccination study analysis.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Run extractions in parallel
    countries_task = evaluator.extract(
        prompt=prompt_extract_countries(),
        template_class=CountriesExtraction,
        extraction_name="partA_countries",
    )
    transmission_task = evaluator.extract(
        prompt=prompt_extract_transmission(),
        template_class=TransmissionExtraction,
        extraction_name="partB_transmission",
    )
    risk_task = evaluator.extract(
        prompt=prompt_extract_risk_assessment(),
        template_class=RiskAssessmentExtraction,
        extraction_name="partC_risk_assessment",
    )

    extracted_countries, transmission_info, risk_info = await asyncio.gather(
        countries_task, transmission_task, risk_task
    )

    # --------------------------
    # Part A: Country criteria
    # --------------------------
    part_a = evaluator.add_parallel(
        id="PartA_CountryIdentification",
        desc="Identification of three countries meeting specified vaccination schedule and coverage criteria",
        parent=root,
        critical=False,
    )

    countries_list = list(extracted_countries.countries or [])
    # Pad/truncate to exactly 3
    if len(countries_list) < 3:
        countries_list.extend([CountryCriteriaItem() for _ in range(3 - len(countries_list))])
    else:
        countries_list = countries_list[:3]

    for i, ctry in enumerate(countries_list):
        await verify_country_criteria(evaluator, part_a, ctry, i)

    # --------------------------
    # Part B: Transmission timeline
    # --------------------------
    await verify_transmission_timeline(evaluator, root, transmission_info)

    # --------------------------
    # Part C: Outbreak risk
    # --------------------------
    part_c = evaluator.add_parallel(
        id="PartC_OutbreakRisk",
        desc="Assessment of measles outbreak risk for Denmark and Costa Rica based on vaccination coverage",
        parent=root,
        critical=False,
    )

    await verify_outbreak_risk_country(
        evaluator,
        part_c,
        country_label="Denmark",
        country_name="Denmark",
        info=risk_info.denmark if risk_info else None,
    )
    await verify_outbreak_risk_country(
        evaluator,
        part_c,
        country_label="CostaRica",
        country_name="Costa Rica",
        info=risk_info.costa_rica if risk_info else None,
    )

    # Finish and return summary
    return evaluator.get_summary()