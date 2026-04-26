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
TASK_ID = "cdc_cauris_2023_press_release_eval"
TASK_DESCRIPTION = (
    "In March 2023, the Centers for Disease Control and Prevention (CDC) issued a press release announcing that a "
    "specific antimicrobial-resistant pathogen, classified as an urgent threat, had spread at an alarming rate in U.S. "
    "healthcare facilities during 2020-2021. Based on CDC data published in the Annals of Internal Medicine: "
    "(1) Identify the specific pathogen (provide genus and species name). "
    "(2) Report the exact number of clinical cases documented for this pathogen in 2019 and in 2021 according to CDC data. "
    "(3) Describe the most concerning change in resistance patterns that occurred specifically in 2021. "
    "(4) Provide the valid CDC press release URL from March 2023 that announced this alarming spread. "
    "Your answer must be verifiable from official CDC sources and the cited publication."
)

# Ground truth anchors used for verification phrasing and consistency checks
EXPECTED_PATHOGEN = "Candida auris"
EXPECTED_CASES_2019 = "476"
EXPECTED_CASES_2021 = "1471"  # Accept 1,471 or 1471 equivalently
PRESS_RELEASE_MONTH = "March"
PRESS_RELEASE_YEAR = "2023"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PathogenReportExtraction(BaseModel):
    pathogen_name: Optional[str] = None
    cases_2019: Optional[str] = None
    cases_2021: Optional[str] = None
    resistance_change_2021: Optional[str] = None
    spread_mechanism_text: Optional[str] = None

    press_release_url: Optional[str] = None
    annals_url: Optional[str] = None
    cdc_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_report_fields() -> str:
    return """
Extract the following items exactly as they appear in the provided answer. If something is not explicitly present in the answer text, return null (or an empty list for URL collections).

Required fields:
1) pathogen_name: The pathogen's genus and species as written (e.g., "Candida auris" or "C. auris").
2) cases_2019: The number of clinical cases for 2019 reported in the answer (extract digits and any punctuation like commas as written).
3) cases_2021: The number of clinical cases for 2021 reported in the answer (extract digits and any punctuation like commas as written).
4) resistance_change_2021: The description (free text) in the answer of the most concerning resistance pattern change that occurred in 2021.
5) spread_mechanism_text: The description (free text) in the answer of how the pathogen spreads in healthcare settings (e.g., via contact with contaminated surfaces/equipment or person-to-person).
6) press_release_url: The CDC press release URL from March 2023 referenced in the answer (must be an actual URL on cdc.gov). If multiple are present, choose the one that explicitly corresponds to the March 2023 'alarming rate' announcement.
7) annals_url: The URL to the Annals of Internal Medicine article (or ACP Journals/Annals page) cited by CDC with the 2019–2021 U.S. data (e.g., acpjournals.org or annals.org).
8) cdc_urls: A list of any additional CDC URLs present in the answer (cdc.gov links) besides the press release URL. If none, return an empty list.

Notes:
- Do not fabricate any URLs or values.
- Extract URLs in full (include protocol). If a URL is missing protocol, prepend http://.
- Return only what is explicitly present in the answer for each field.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _is_cdc_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return "cdc.gov" in url.lower()


def _as_list(url: Optional[str]) -> List[str]:
    return [url] if url and url.strip() else []


def _merge_sources(*url_lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        combined.extend([u for u in lst if u])
    return _dedup(combined)


def _filter_cdc(urls: List[str]) -> List[str]:
    return [u for u in urls if _is_cdc_url(u)]


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_pathogen_identity(evaluator: Evaluator, parent, ex: PathogenReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="Pathogen_Identity",
        desc="Correctly identify the pathogen and its CDC urgent-threat status.",
        parent=parent,
        critical=True,
    )

    # Leaf: Pathogen name provided equals Candida auris (allow abbreviation)
    leaf_name = evaluator.add_leaf(
        id="Pathogen_Name_Genus_Species",
        desc="Provides the pathogen genus and species name.",
        parent=node,
        critical=True,
    )
    provided_name = ex.pathogen_name or ""
    claim_name = (
        f"The pathogen named in the answer is '{provided_name}', and this is equivalent to '{EXPECTED_PATHOGEN}' "
        f"(allow 'C. auris' as an acceptable abbreviation)."
    )
    await evaluator.verify(
        claim=claim_name,
        node=leaf_name,
        additional_instruction=(
            "Judge this only by comparing the name(s) mentioned in the answer. Consider 'C. auris' equivalent to "
            "'Candida auris'. Case-insensitive; minor punctuation/casing differences are acceptable."
        ),
    )

    # Leaf: CDC urgent threat classification
    leaf_urgent = evaluator.add_leaf(
        id="CDC_Urgent_Threat_Classification",
        desc="Confirms the pathogen is classified by CDC as an urgent antimicrobial resistance threat as of 2023.",
        parent=node,
        critical=True,
    )
    cdc_evidence = _merge_sources(_as_list(ex.press_release_url), ex.cdc_urls)
    # Ensure at least one CDC URL; fallback to press release if any.
    if not cdc_evidence and _is_cdc_url(ex.press_release_url):
        cdc_evidence = _as_list(ex.press_release_url)

    urgent_claim = (
        "CDC classifies Candida auris as an urgent antimicrobial resistance threat in the United States (as of 2023)."
    )
    await evaluator.verify(
        claim=urgent_claim,
        node=leaf_urgent,
        sources=cdc_evidence if cdc_evidence else _as_list(ex.press_release_url),
        additional_instruction="Verify this classification from an official CDC webpage; prioritise the March 2023 press release or CDC Candida auris threat pages.",
    )


async def verify_case_data(evaluator: Evaluator, parent, ex: PathogenReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="Clinical_Case_Data_2019_and_2021",
        desc="Reports the exact CDC clinical case counts for 2019 and 2021 and the rapid rise during 2020–2021.",
        parent=parent,
        critical=True,
    )

    # Leaf: Case counts exactly as CDC documented (anchor to answer's numbers; validate against sources)
    leaf_counts = evaluator.add_leaf(
        id="Case_Counts_2019_and_2021",
        desc="Reports CDC clinical case counts for both 2019 and 2021 and matches the CDC-documented values (476 in 2019; 1,471 in 2021).",
        parent=node,
        critical=True,
    )
    cases_2019 = (ex.cases_2019 or "").strip()
    cases_2021 = (ex.cases_2021 or "").strip()
    sources_counts = _merge_sources(_as_list(ex.annals_url), _as_list(ex.press_release_url))

    claim_counts = (
        f"According to CDC/Annals, the U.S. clinical case counts of Candida auris were {cases_2019} in 2019 and "
        f"{cases_2021} in 2021. These should exactly match {EXPECTED_CASES_2019} (2019) and {EXPECTED_CASES_2021} (2021), "
        f"treating digit group separators (e.g., 1471 vs 1,471) as equivalent."
    )
    await evaluator.verify(
        claim=claim_counts,
        node=leaf_counts,
        sources=sources_counts,
        additional_instruction=(
            "Confirm on the provided pages that the 2019 and 2021 clinical case totals match the claim. "
            "Accept both '1471' and '1,471' as equivalent. If either number differs from what the pages report, mark Incorrect."
        ),
    )

    # Leaf: Rapid rise 2020–2021 explicitly documented
    leaf_rise = evaluator.add_leaf(
        id="Rapid_Rise_2020_2021_Documented",
        desc="States (with CDC/Annals support) that cases significantly increased between 2020 and 2021, with the most rapid rise during this period.",
        parent=node,
        critical=True,
    )
    sources_rise = _merge_sources(_as_list(ex.press_release_url), _as_list(ex.annals_url))
    claim_rise = (
        "CDC/Annals report that Candida auris clinical cases increased substantially during 2020–2021 in the U.S., "
        "with the most rapid rise across that period."
    )
    await evaluator.verify(
        claim=claim_rise,
        node=leaf_rise,
        sources=sources_rise,
        additional_instruction="Look for explicit language about a sharp or alarming increase during 2020–2021.",
    )


async def verify_resistance_patterns(evaluator: Evaluator, parent, ex: PathogenReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="Resistance_Pattern_Change_2021",
        desc="Describes the most concerning resistance change occurring specifically in 2021 and confirms multidrug resistance per constraints.",
        parent=parent,
        critical=True,
    )

    # Leaf: Multidrug resistance (multiple classes)
    leaf_mdr = evaluator.add_leaf(
        id="Multidrug_Resistance_Multiple_Classes",
        desc="Confirms resistance to multiple classes of antimicrobial drugs (not just one drug).",
        parent=node,
        critical=True,
    )
    mdr_sources = _merge_sources(_filter_cdc(_merge_sources(ex.cdc_urls, _as_list(ex.press_release_url))))
    claim_mdr = (
        "Candida auris displays multidrug resistance, i.e., resistance to multiple classes of antifungal drugs "
        "(e.g., azoles, echinocandins, and polyenes)."
    )
    await evaluator.verify(
        claim=claim_mdr,
        node=leaf_mdr,
        sources=mdr_sources if mdr_sources else _as_list(ex.press_release_url),
        additional_instruction="Confirm via CDC sources; explicit mention of resistance to multiple classes qualifies.",
    )

    # Leaf: Echinocandin resistance tripled in 2021
    leaf_ech = evaluator.add_leaf(
        id="Echinocandin_Resistance_Tripled_2021",
        desc="Describes that echinocandin resistance tripled in 2021 according to CDC data.",
        parent=node,
        critical=True,
    )
    ech_sources = _merge_sources(_as_list(ex.press_release_url), _as_list(ex.annals_url))
    claim_ech = "CDC/Annals report that echinocandin-resistant Candida auris cases tripled in 2021."
    await evaluator.verify(
        claim=claim_ech,
        node=leaf_ech,
        sources=ech_sources,
        additional_instruction="Look specifically for the assertion that echinocandin resistance tripled in 2021.",
    )


async def verify_spread_characteristics(evaluator: Evaluator, parent, ex: PathogenReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="Spread_Characteristics",
        desc="Checks the required spread characteristics from constraints.",
        parent=parent,
        critical=True,
    )

    leaf_spread = evaluator.add_leaf(
        id="Healthcare_Spread_and_Mechanism",
        desc="Confirms the pathogen spreads easily in healthcare facilities through contact with contaminated surfaces/equipment or person-to-person contact.",
        parent=node,
        critical=True,
    )
    spread_sources = _filter_cdc(_merge_sources(ex.cdc_urls, _as_list(ex.press_release_url)))
    claim_spread = (
        "CDC states that Candida auris spreads easily in healthcare settings via person-to-person contact and/or "
        "contact with contaminated surfaces or equipment."
    )
    await evaluator.verify(
        claim=claim_spread,
        node=leaf_spread,
        sources=spread_sources if spread_sources else _as_list(ex.press_release_url),
        additional_instruction="Confirm explicit statements about transmission in healthcare via contact with people or contaminated surfaces/equipment.",
    )


async def verify_press_release(evaluator: Evaluator, parent, ex: PathogenReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="CDC_March_2023_Press_Release",
        desc="Provides and validates the March 2023 CDC press release referenced in the question.",
        parent=parent,
        critical=True,
    )

    # Leaf: Press release URL provided (valid cdc.gov and likely March 2023)
    leaf_url = evaluator.add_custom_node(
        result=bool(ex.press_release_url) and _is_cdc_url(ex.press_release_url) and ("2023" in (ex.press_release_url or "")),
        id="Press_Release_URL_Provided",
        desc="Provides a valid CDC press release URL (on cdc.gov) for the March 2023 announcement.",
        parent=node,
        critical=True,
    )

    # Leaf: Press release contains 'alarming rate' language for 2020–2021 and is March 2023
    leaf_claim = evaluator.add_leaf(
        id="Press_Release_Contains_Alarming_Rate_Claim",
        desc="Confirms the March 2023 CDC announcement states the pathogen spread at an 'alarming rate' in U.S. healthcare facilities during 2020–2021.",
        parent=node,
        critical=True,
    )
    claim_press = (
        f"This CDC press release (published in {PRESS_RELEASE_MONTH} {PRESS_RELEASE_YEAR}) explicitly states that Candida auris "
        "spread at an 'alarming rate' in U.S. healthcare facilities during 2020–2021."
    )
    await evaluator.verify(
        claim=claim_press,
        node=leaf_claim,
        sources=_as_list(ex.press_release_url),
        additional_instruction="Verify both the 'alarming rate' phrasing (or a direct equivalent) and the publication date being March 2023.",
    )


async def verify_sources_and_verifiability(evaluator: Evaluator, parent, ex: PathogenReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="Source_and_Verifiability",
        desc="Meets the question’s source/verifiability requirements (official CDC sources + cited Annals publication) including date constraints.",
        parent=parent,
        critical=True,
    )

    # Additional existence check to ensure Annals URL is present (to avoid ungrounded verification)
    annals_present = evaluator.add_custom_node(
        result=bool(ex.annals_url),
        id="Annals_URL_Provided",
        desc="Annals of Internal Medicine URL is provided in the answer.",
        parent=node,
        critical=True,
    )

    # Leaf: Annals/CDC data source cited and dated 2022–2023
    leaf_annals = evaluator.add_leaf(
        id="Annals_CDC_Data_Source_Cited_and_Dated_2022_2023",
        desc="Cites the relevant CDC-data publication in Annals of Internal Medicine (or equivalent peer-reviewed/CDC report) and confirms it is from 2022–2023 as required by constraints.",
        parent=node,
        critical=True,
    )
    claim_annals = (
        "This webpage is an Annals of Internal Medicine (ACP Journals) article presenting CDC data on Candida auris "
        "in the United States (2019–2021), published in 2022 or 2023 (the CDC-cited article is in 2023)."
    )
    await evaluator.verify(
        claim=claim_annals,
        node=leaf_annals,
        sources=_as_list(ex.annals_url),
        additional_instruction="Check the journal venue, topic scope (U.S. 2019–2021), and that the publication year is 2022 or 2023.",
    )

    # Leaf: Official CDC verifiability (press release and CDC pages suffice)
    leaf_official = evaluator.add_leaf(
        id="Official_CDC_Source_Verifiability",
        desc="All requested claims are verifiable from official CDC sources (including the cited press release/CDC pages).",
        parent=node,
        critical=True,
    )
    cdc_sources_all = _merge_sources(_as_list(ex.press_release_url), _filter_cdc(ex.cdc_urls))
    claim_official = (
        "The provided CDC URLs (on cdc.gov), including the press release, are official sources that can verify the "
        "urgent threat classification and the 'alarming rate during 2020–2021' announcement for Candida auris."
    )
    await evaluator.verify(
        claim=claim_official,
        node=leaf_official,
        sources=cdc_sources_all if cdc_sources_all else _as_list(ex.press_release_url),
        additional_instruction="Assess whether at least one of these URLs is an official CDC page explicitly supporting the stated claims.",
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

    # Extract structured fields from the answer
    extracted: PathogenReportExtraction = await evaluator.extract(
        prompt=prompt_extract_report_fields(),
        template_class=PathogenReportExtraction,
        extraction_name="extracted_report_fields",
    )

    # Record ground truth anchors for transparency
    evaluator.add_ground_truth(
        {
            "expected_pathogen": EXPECTED_PATHOGEN,
            "expected_cases": {"2019": EXPECTED_CASES_2019, "2021": EXPECTED_CASES_2021},
            "press_release_expected_month_year": f"{PRESS_RELEASE_MONTH} {PRESS_RELEASE_YEAR}",
        }
    )

    # Build top-level critical node mirroring the rubric root
    top = evaluator.add_parallel(
        id="Pathogen_Identification_and_CDC_Announcement_Verification",
        desc="Identify the CDC-referenced urgent-threat pathogen and provide Annals/CDC-verifiable details (cases, resistance change, and March 2023 press release URL).",
        parent=root,
        critical=True,
    )

    # Subtrees according to rubric
    await verify_pathogen_identity(evaluator, top, extracted)
    await verify_case_data(evaluator, top, extracted)
    await verify_resistance_patterns(evaluator, top, extracted)
    await verify_spread_characteristics(evaluator, top, extracted)
    await verify_press_release(evaluator, top, extracted)
    await verify_sources_and_verifiability(evaluator, top, extracted)

    # Optional: add a compact summary of source URLs captured
    evaluator.add_custom_info(
        {
            "press_release_url": extracted.press_release_url,
            "annals_url": extracted.annals_url,
            "cdc_urls": extracted.cdc_urls,
        },
        info_type="source_urls",
        info_name="extracted_source_urls",
    )

    return evaluator.get_summary()