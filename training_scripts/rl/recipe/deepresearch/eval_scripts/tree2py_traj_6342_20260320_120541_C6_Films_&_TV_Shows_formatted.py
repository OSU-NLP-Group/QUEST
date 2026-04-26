import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "emmys76_tax_incentives"
TASK_DESCRIPTION = """
Identify three television series that won Emmy awards at the 76th Primetime Emmy Awards (held on September 15, 2024) in the Outstanding Drama Series, Outstanding Comedy Series, or Outstanding Limited or Anthology Series categories. Each series must have been filmed primarily in a different jurisdiction, and each jurisdiction must offer a film or television tax incentive program with a base tax credit rate of at least 25% for qualified production expenditures.

For each of the three series, provide:
1. The series title
2. The specific Emmy award category won (Outstanding Drama Series, Outstanding Comedy Series, or Outstanding Limited or Anthology Series)
3. The primary filming location with geographic specificity (city, county, or region - not just country)
4. The filming jurisdiction's base tax credit rate for film/television production
5. The type of tax incentive (refundable tax credit or transferable tax credit)
6. Confirmation that principal photography was completed in a timeframe consistent with the Emmy eligibility period (June 1, 2023 - May 31, 2024)
7. Reference URLs supporting each piece of information

The three series must each be filmed in different jurisdictions (different states, provinces, or countries with distinct tax incentive programs).
"""

ALLOWED_CATEGORIES = [
    "Outstanding Drama Series",
    "Outstanding Comedy Series",
    "Outstanding Limited or Anthology Series",
]
CEREMONY_DATE_TEXT = "September 15, 2024"
ELIGIBILITY_START = "June 1, 2023"
ELIGIBILITY_END = "May 31, 2024"


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class SeriesItemExtraction(BaseModel):
    title: Optional[str] = None
    emmy_category: Optional[str] = None
    emmy_refs: List[str] = Field(default_factory=list)

    primary_location: Optional[str] = None  # city/county/region
    location_refs: List[str] = Field(default_factory=list)

    jurisdiction: Optional[str] = None  # state/province/country administering the incentive
    tax_base_rate: Optional[str] = None  # e.g., "25%", "30 percent"
    incentive_type: Optional[str] = None  # e.g., "refundable", "transferable"
    tax_refs: List[str] = Field(default_factory=list)

    principal_photography_window: Optional[str] = None  # e.g., "Aug–Nov 2023"
    timeline_refs: List[str] = Field(default_factory=list)


class AllSeriesExtraction(BaseModel):
    series: List[SeriesItemExtraction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series() -> str:
    categories_list = "; ".join(ALLOWED_CATEGORIES)
    return f"""
Extract up to 5 television series mentioned in the answer that purportedly meet the task. For each series, extract ONLY what the answer explicitly states. Do not infer or add information.

For each series, return the following fields:
- title: The series title (string)
- emmy_category: The exact Emmy category the series is claimed to have WON at the 76th Primetime Emmy Awards; must be one of: {categories_list}. If not explicitly one of these or unclear, set null.
- emmy_refs: Array of URLs cited in the answer that support the Emmy win claim at the 76th Primetime Emmy Awards (ceremony on {CEREMONY_DATE_TEXT})

- primary_location: The primary filming location with geographic specificity (city, county, or region; not just a country). If only a country is given, return it anyway.
- location_refs: Array of URLs cited that support the primary filming location

- jurisdiction: The filming jurisdiction administering the incentive program (e.g., U.S. state, Canadian province, or a country if relevant). If the answer indicates both city and state/province, use the state/province as the jurisdiction.
- tax_base_rate: The base tax credit rate for film/TV production in the jurisdiction as written in the answer (e.g., "25%", "30 percent"). If missing or only bonus/stacked rates are given without a base, set null.
- incentive_type: The incentive type as written in the answer ("refundable" or "transferable"). If unclear or not provided, set null.
- tax_refs: Array of URLs cited that support the jurisdiction’s incentive base rate and incentive type

- principal_photography_window: The filming/production timeframe text in the answer for the Emmy-eligible season (e.g., "filmed fall 2023").
- timeline_refs: Array of URLs cited that support the production timeline

Return a JSON object with field:
- series: an array of 0–5 objects, each containing the above fields.

If any field is missing in the answer for a given series, set it to null (or [] for arrays).
Do not invent URLs; include only explicit URLs mentioned in the answer text.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return " ".join(s.strip().lower().split())


def sources_absent(urls: List[str]) -> bool:
    return not urls or len([u for u in urls if isinstance(u, str) and u.strip()]) == 0


# --------------------------------------------------------------------------- #
# Verification for a single series                                            #
# --------------------------------------------------------------------------- #
async def verify_one_series(
    evaluator: Evaluator,
    parent_node,
    series: SeriesItemExtraction,
    index: int,
    prev_jurisdictions: List[Optional[str]],
) -> None:
    """
    Build verification sub-tree and run checks for one series.
    """
    idx1 = index + 1  # human-readable index

    # Series sequential container
    series_seq = evaluator.add_sequential(
        id=f"series_{idx1}",
        desc=(
            "First qualifying Emmy-winning series with all required details"
            if idx1 == 1 else
            ("Second qualifying Emmy-winning series with all required details, filmed in a different jurisdiction than Series 1"
             if idx1 == 2 else
             "Third qualifying Emmy-winning series with all required details, filmed in a different jurisdiction than Series 1 and Series 2")
        ),
        parent=parent_node,
        critical=False,
    )

    # Requirements (critical group)
    req = evaluator.add_parallel(
        id=f"series_{idx1}_requirements",
        desc=f"All verification requirements for series #{idx1}",
        parent=series_seq,
        critical=True,
    )

    # 1) Emmy Award Verification (critical group)
    emmy_grp = evaluator.add_parallel(
        id=f"s{idx1}_emmy_award_verification",
        desc="Verification that the series won at least one Emmy at the 76th Primetime Emmy Awards",
        parent=req,
        critical=True,
    )

    # 1.a) Award_Won_At_76th_Emmys (leaf)
    won_leaf = evaluator.add_leaf(
        id=f"s{idx1}_award_won_76th",
        desc="Series won at least one Emmy award at the 76th Primetime Emmy Awards ceremony held on September 15, 2024",
        parent=emmy_grp,
        critical=True,
    )
    claim_won = (
        f"'{series.title}' won at least one Emmy award at the 76th Primetime Emmy Awards "
        f"held on {CEREMONY_DATE_TEXT}."
    )
    add_ins_won = (
        "Judge strictly for the 76th Primetime Emmy Awards (ceremony on {date}). "
        "Nominations or Creative Arts only are insufficient unless they are clearly part of the 76th Primetime Emmy WINS. "
        "If the provided URLs are irrelevant or do not explicitly show a WIN for the 76th ceremony, mark Incorrect."
    ).format(date=CEREMONY_DATE_TEXT)
    await evaluator.verify(
        claim=claim_won,
        node=won_leaf,
        sources=series.emmy_refs,
        additional_instruction=add_ins_won,
    )

    # 1.b) Award_Category (leaf)
    cat_leaf = evaluator.add_leaf(
        id=f"s{idx1}_award_category",
        desc="The Emmy award(s) won include at least one in the allowed categories",
        parent=emmy_grp,
        critical=True,
    )
    category_text = series.emmy_category or "UNKNOWN"
    allowed_text = "; ".join(ALLOWED_CATEGORIES)
    claim_cat = (
        f"At the 76th Primetime Emmy Awards, '{series.title}' won the category '{category_text}', "
        f"which must be exactly one of: {allowed_text}."
    )
    add_ins_cat = (
        "Only pass if the referenced page explicitly indicates a WIN at the 76th ceremony "
        "in the exact claimed category. If the category is not exactly one of the allowed list, mark Incorrect. "
        "Do not accept nominations alone."
    )
    await evaluator.verify(
        claim=claim_cat,
        node=cat_leaf,
        sources=series.emmy_refs,
        additional_instruction=add_ins_cat,
    )

    # 1.c) Emmy_Verification_Reference (leaf)
    emmy_ref_leaf = evaluator.add_leaf(
        id=f"s{idx1}_emmy_refs_confirm",
        desc="Provided reference URL confirms the Emmy win(s) at the 76th Primetime Emmy Awards",
        parent=emmy_grp,
        critical=True,
    )
    claim_emmy_refs = (
        f"The provided sources confirm that '{series.title}' won at the 76th Primetime Emmy Awards."
    )
    add_ins_emmy_refs = (
        "The claim must be supported by the provided URLs. "
        "If no URLs are provided or they are invalid/unrelated, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_emmy_refs,
        node=emmy_ref_leaf,
        sources=series.emmy_refs,
        additional_instruction=add_ins_emmy_refs,
    )

    # 2) Filming Location (critical group)
    loc_grp = evaluator.add_parallel(
        id=f"s{idx1}_filming_location",
        desc="Primary filming location identification and verification",
        parent=req,
        critical=True,
    )

    # 2.a) Primary_Location_Specified (existence as custom, critical)
    has_location = bool(series.primary_location and series.primary_location.strip())
    evaluator.add_custom_node(
        result=has_location,
        id=f"s{idx1}_primary_location_specified",
        desc="Primary filming location is identified with specific geographic detail (city, county, or region)",
        parent=loc_grp,
        critical=True,
    )

    # 2.b) Location_Geographic_Specificity (leaf)
    spec_leaf = evaluator.add_leaf(
        id=f"s{idx1}_location_specificity",
        desc="Location is specified beyond country-level (e.g., specific city, county, or region within a country/state)",
        parent=loc_grp,
        critical=True,
    )
    loc_text = series.primary_location or ""
    claim_specificity = (
        f"The text '{loc_text}' specifies a concrete locality (city, county, or region), "
        f"not just a country-level specification."
    )
    add_ins_specificity = (
        "Consider 'city, county, or region' as specific; examples: 'Vancouver, British Columbia', "
        "'Atlanta, Georgia', 'Northern Ireland'. Non-specific examples: 'United States', 'UK', 'Canada' "
        "without a city/region. If the string is clearly only a country, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_specificity,
        node=spec_leaf,
        sources=None,
        additional_instruction=add_ins_specificity,
    )

    # 2.c) Different_Jurisdiction for series 2 and 3 (critical custom)
    if idx1 == 2:
        jur1 = normalize_text(prev_jurisdictions[0] if len(prev_jurisdictions) >= 1 else None)
        jur2 = normalize_text(series.jurisdiction)
        evaluator.add_custom_node(
            result=bool(jur1 and jur2 and jur1 != jur2),
            id=f"s{idx1}_different_jurisdiction",
            desc="The filming jurisdiction is different from Series 1's jurisdiction",
            parent=loc_grp,
            critical=True,
        )
    elif idx1 == 3:
        jur1 = normalize_text(prev_jurisdictions[0] if len(prev_jurisdictions) >= 1 else None)
        jur2 = normalize_text(prev_jurisdictions[1] if len(prev_jurisdictions) >= 2 else None)
        jur3 = normalize_text(series.jurisdiction)
        all_diff = bool(jur1 and jur2 and jur3 and jur3 != jur1 and jur3 != jur2)
        evaluator.add_custom_node(
            result=all_diff,
            id=f"s{idx1}_different_jurisdiction",
            desc="The filming jurisdiction is different from both Series 1 and Series 2's jurisdictions",
            parent=loc_grp,
            critical=True,
        )

    # 2.d) Location_Verification_Reference (leaf)
    loc_ref_leaf = evaluator.add_leaf(
        id=f"s{idx1}_location_refs_confirm",
        desc="Provided reference URL confirms the primary filming location",
        parent=loc_grp,
        critical=True,
    )
    claim_loc_refs = (
        f"The provided sources confirm that the primary filming location for '{series.title}' is {loc_text}."
    )
    add_ins_loc_refs = (
        "The page(s) should indicate that this was the primary or majority filming location for the relevant season. "
        "If multiple locations are listed, it should clearly indicate the named location as primary/major. "
        "If no URLs are provided or they don't support the claim, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_loc_refs,
        node=loc_ref_leaf,
        sources=series.location_refs,
        additional_instruction=add_ins_loc_refs,
    )

    # 3) Tax Incentive Program (critical group)
    tax_grp = evaluator.add_parallel(
        id=f"s{idx1}_tax_incentive",
        desc="Tax incentive program details for the filming jurisdiction",
        parent=req,
        critical=True,
    )

    # 3.a) Base_Rate_Requirement (leaf)
    base_leaf = evaluator.add_leaf(
        id=f"s{idx1}_base_rate_requirement",
        desc="The jurisdiction's base film/television tax credit rate is at least 25% for qualified production expenditures",
        parent=tax_grp,
        critical=True,
    )
    jur_text = series.jurisdiction or "the jurisdiction"
    claim_base = (
        f"In {jur_text}, the base film/TV production tax credit rate is at least 25%."
    )
    add_ins_base = (
        "Verify the BASE or minimum standard rate (exclude stacked bonuses/add-ons unless explicitly part of the base). "
        "If only bonus or combined rates reach >=25% but the base is lower, mark Incorrect. "
        "If there are no URLs, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_base,
        node=base_leaf,
        sources=series.tax_refs,
        additional_instruction=add_ins_base,
    )

    # 3.b) Incentive_Type (leaf)
    type_leaf = evaluator.add_leaf(
        id=f"s{idx1}_incentive_type",
        desc="The tax incentive is identified as either a refundable tax credit or a transferable tax credit",
        parent=tax_grp,
        critical=True,
    )
    type_text = (series.incentive_type or "UNKNOWN").lower()
    claim_type = (
        f"The film/TV incentive in {jur_text} is identified in the answer as a '{type_text}' tax credit, "
        f"and the provided sources confirm that it is either refundable or transferable."
    )
    add_ins_type = (
        "Only pass if the answer specifies the incentive type (refundable or transferable) and the URLs confirm it. "
        "Accept common synonyms (e.g., 'assignable'/'saleable' for transferable, 'rebate paid out as refund' for refundable). "
        "If the answer does not state the type or URLs are missing/unrelated, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_type,
        node=type_leaf,
        sources=series.tax_refs,
        additional_instruction=add_ins_type,
    )

    # 3.c) Tax_Incentive_Reference (leaf)
    tax_ref_leaf = evaluator.add_leaf(
        id=f"s{idx1}_tax_refs_confirm",
        desc="Provided reference URL confirms the tax incentive program details including rate and type",
        parent=tax_grp,
        critical=True,
    )
    claim_tax_refs = (
        f"The provided sources confirm a base rate of at least 25% and identify whether the credit in {jur_text} is "
        f"refundable or transferable."
    )
    add_ins_tax_refs = (
        "If sources are missing or do not clearly state the base rate (excluding bonuses) and the incentive type, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_tax_refs,
        node=tax_ref_leaf,
        sources=series.tax_refs,
        additional_instruction=add_ins_tax_refs,
    )

    # 4) Production Timeline (critical group)
    time_grp = evaluator.add_parallel(
        id=f"s{idx1}_production_timeline",
        desc="Production timeline verification for Emmy eligibility",
        parent=req,
        critical=True,
    )

    # 4.a) Filming_Period_Eligibility (leaf)
    timeline_leaf = evaluator.add_leaf(
        id=f"s{idx1}_filming_period_eligibility",
        desc="Principal photography for the Emmy-eligible season fits within the June 1, 2023 - May 31, 2024 eligibility period",
        parent=time_grp,
        critical=True,
    )
    claim_timeline = (
        f"Principal photography for the Emmy-eligible season of '{series.title}' occurred between "
        f"{ELIGIBILITY_START} and {ELIGIBILITY_END}."
    )
    add_ins_timeline = (
        "Accept if the sources show filming dates substantially within the window (e.g., mid/late 2023 or early 2024) "
        "for the season that won at the 76th Emmys. Release/premiere dates alone are insufficient."
    )
    await evaluator.verify(
        claim=claim_timeline,
        node=timeline_leaf,
        sources=series.timeline_refs,
        additional_instruction=add_ins_timeline,
    )

    # 4.b) Timeline_Verification_Reference (leaf)
    time_ref_leaf = evaluator.add_leaf(
        id=f"s{idx1}_timeline_refs_confirm",
        desc="Provided reference URL confirms production timeline or filming dates",
        parent=time_grp,
        critical=True,
    )
    claim_time_refs = (
        f"The provided sources include filming or production dates indicating compatibility with the eligibility window "
        f"({ELIGIBILITY_START} - {ELIGIBILITY_END}) for '{series.title}'."
    )
    add_ins_time_refs = (
        "If there are no URLs or they only mention non-filming milestones (e.g., release date) without filming dates, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_time_refs,
        node=time_ref_leaf,
        sources=series.timeline_refs,
        additional_instruction=add_ins_time_refs,
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
    """
    Evaluate an answer for the 76th Primetime Emmys + Tax Incentives task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root should be non-critical to allow partial credit across series
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

    # Extract series data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_series(),
        template_class=AllSeriesExtraction,
        extraction_name="series_extraction",
    )

    # Keep only first 3 series, pad if fewer
    series_list = list(extracted.series[:3])
    while len(series_list) < 3:
        series_list.append(SeriesItemExtraction())

    # Add some contextual info
    evaluator.add_custom_info(
        {
            "allowed_categories": ALLOWED_CATEGORIES,
            "ceremony_date": CEREMONY_DATE_TEXT,
            "eligibility_window": {"start": ELIGIBILITY_START, "end": ELIGIBILITY_END},
        },
        info_type="context",
        info_name="evaluation_context",
    )

    # Build/verify per-series subtrees
    prev_jurisdictions: List[Optional[str]] = []
    for i in range(3):
        await verify_one_series(
            evaluator=evaluator,
            parent_node=root,
            series=series_list[i],
            index=i,
            prev_jurisdictions=prev_jurisdictions,
        )
        prev_jurisdictions.append(series_list[i].jurisdiction)

    return evaluator.get_summary()