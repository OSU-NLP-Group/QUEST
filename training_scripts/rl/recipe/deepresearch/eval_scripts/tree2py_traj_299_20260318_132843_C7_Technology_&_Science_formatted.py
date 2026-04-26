import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "tmobile_5g_profile_mar2026"
TASK_DESCRIPTION = (
    "As of March 2026, compile a comprehensive profile of T-Mobile's 5G network deployment in the United States. "
    "Your profile must include the following specific information: (1) Total population coverage in number of people, "
    "(2) Coverage as a percentage of US population, (3) T-Mobile's market position relative to other major US carriers "
    "in terms of 5G availability, (4) Low-band 5G spectrum deployment status specifically for band n71 at 600 MHz, "
    "(5) Mid-band 5G spectrum deployment status specifically for band n41 at 2.5 GHz, (6) C-band spectrum deployment "
    "status specifically for band n77, (7) Millimeter wave (mmWave) 5G availability status in the network, "
    "(8) 5G Standalone (SA) network architecture deployment status, (9) Voice over New Radio (VoNR) deployment status "
    "including coverage extent, (10) Specific details about spectrum holdings including bandwidth amounts, "
    "(11) Median 5G download speed performance metrics, (12) Rural coverage expansion status and progress, "
    "(13) Support for 5G-Advanced features, and (14) Future network expansion plans including any announced "
    "infrastructure investments. Provide specific factual information with supporting evidence for each aspect."
)

# For reference/ground-truth expectations (used for contextual info in the summary)
EXPECTED_FACTS = {
    "coverage_total_population_people": "323 million",
    "coverage_percentage_population": "98%",
    "market_position": "T-Mobile leads US carriers in 5G availability",
    "lowband_n71": "Deployed (600 MHz, band n71)",
    "midband_n41": "Deployed (2.5 GHz, band n41)",
    "cband_n77": "Deployed (3.7 GHz, band n77)",
    "mmwave": "Available in select areas",
    "sa": "5G Standalone deployed",
    "vonr_coverage": "VoNR deployed to over 100 million people",
    "spectrum_2p5_ghz_holdings": "Approximately 194 MHz in many markets",
    "median_5g_speed_h1_2025": "Approximately 299 Mbps (H1 2025)",
    "rural_expansion": "Expanded rural 5G coverage",
    "fiveg_advanced_features": "Supports multi-band carrier aggregation (part of 5G-Advanced)",
    "future_investments": "Plans to invest $8B and deploy 6,000 towers for rural expansion",
}


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class Aspect(BaseModel):
    # statement: exact text quoted or paraphrased from the answer describing this aspect
    statement: Optional[str] = None
    # value: the specific figure/value claimed in the answer (include units exactly as in the answer, if any)
    value: Optional[str] = None
    # urls: all URLs explicitly cited in the answer for this aspect
    urls: List[str] = Field(default_factory=list)


class TMobile5GProfileExtraction(BaseModel):
    timeframe: Optional[Aspect] = None

    coverage_total_population: Optional[Aspect] = None
    coverage_percentage: Optional[Aspect] = None
    market_position: Optional[Aspect] = None

    lowband_n71: Optional[Aspect] = None
    midband_n41: Optional[Aspect] = None
    cband_n77: Optional[Aspect] = None
    mmwave: Optional[Aspect] = None
    sa: Optional[Aspect] = None
    vonr: Optional[Aspect] = None

    spectrum_holdings_2p5ghz: Optional[Aspect] = None
    median_5g_speed_h1_2025: Optional[Aspect] = None
    rural_expansion: Optional[Aspect] = None
    fiveg_advanced_features: Optional[Aspect] = None
    future_plans_investment: Optional[Aspect] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_profile() -> str:
    return """
    You must extract the specific aspects of the T-Mobile 5G network profile as they appear in the answer text, along with the exact URLs the answer cites for each aspect. For each aspect below, extract:
    - statement: the exact statement (or a faithful paraphrase) the answer uses for that aspect.
    - value: the key figure/number or succinct value if applicable (include units exactly as shown in the answer; leave null if not numeric/specific).
    - urls: an array of the explicit URLs cited in the answer to support that aspect. Only include URLs that the answer actually shows.

    Extract the following fields (all are optional if missing in the answer; use null/empty accordingly):
    - timeframe: statement, value (e.g., "as of March 2026"), urls
    - coverage_total_population: statement, value (e.g., "323 million"), urls
    - coverage_percentage: statement, value (e.g., "98%"), urls
    - market_position: statement (e.g., "leads US carriers in 5G availability"), value (if any), urls
    - lowband_n71: statement (mention "n71", "600 MHz"), value (if any), urls
    - midband_n41: statement (mention "n41", "2.5 GHz"), value (if any), urls
    - cband_n77: statement (mention "n77", "3.7 GHz"), value (if any), urls
    - mmwave: statement (mention "mmWave" / "millimeter wave" and "select areas" or equivalent), value (if any), urls
    - sa: statement (mention "Standalone" or "SA"), value (if any), urls
    - vonr: statement (mention "VoNR" and any coverage quantity like "over 100 million"), value, urls
    - spectrum_holdings_2p5ghz: statement (mention "2.5 GHz" holdings), value (e.g., "~194 MHz"), urls
    - median_5g_speed_h1_2025: statement (mention H1 2025 and median 5G download), value (e.g., "~299 Mbps"), urls
    - rural_expansion: statement (mention rural 5G expansion), value (if any), urls
    - fiveg_advanced_features: statement (mention 5G-Advanced and "multi-band carrier aggregation"), value (if any), urls
    - future_plans_investment: statement (mention $8B and 6,000 towers, if present), value (e.g., "$8B; 6,000 towers"), urls

    IMPORTANT:
    - Only extract URLs explicitly present in the answer text (including markdown links). Do not invent any URLs.
    - If a field is not mentioned in the answer, set its statement and value to null and urls to an empty list.
    - For numeric values, do not normalize; keep the exact string and units as in the answer (e.g., "323 million", "98%", "299 Mbps").
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(aspect: Optional[Aspect]) -> List[str]:
    return [u for u in (aspect.urls if (aspect and aspect.urls) else []) if isinstance(u, str) and u.strip()]


def _all_urls(profile: TMobile5GProfileExtraction) -> List[str]:
    urls: List[str] = []
    for field in profile.__fields__:
        aspect: Optional[Aspect] = getattr(profile, field)
        urls.extend(_safe_urls(aspect))
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_timeframe(evaluator: Evaluator, parent, profile: TMobile5GProfileExtraction) -> None:
    """
    Build verification for:
    - Timeframe_AsOf_March_2026 (grouped into: answer states timeframe + recency supported by at least one source)
    """
    node = evaluator.add_parallel(
        id="Timeframe_AsOf_March_2026",
        desc="The profile explicitly frames the information as current 'as of March 2026' and supports recency with sourcing context.",
        parent=parent,
        critical=True
    )

    # Leaf 1: Answer explicitly states "as of March 2026"
    leaf_answer_timeframe = evaluator.add_leaf(
        id="Timeframe_AsOf_March_2026_answer_mentions",
        desc="Answer explicitly frames the information as 'as of March 2026' (or clear equivalent).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the information is current as of March 2026 (acceptable variants include 'as of Mar 2026', 'as of 03/2026').",
        node=leaf_answer_timeframe,
        additional_instruction="Look only at the answer text. Accept minor format variants like 'as of Mar. 2026' or 'as of 03/2026'."
    )

    # Leaf 2a: Ensure we have at least one URL overall to support recency context
    all_urls = _all_urls(profile)
    evaluator.add_custom_node(
        result=(len(all_urls) > 0),
        id="Timeframe_AsOf_March_2026_sources_provided",
        desc="At least one cited source URL is provided to support recency/context.",
        parent=node,
        critical=True
    )

    # Leaf 2b: Check at least one source is reasonably recent (2025 or 2026)
    leaf_source_recent = evaluator.add_leaf(
        id="Timeframe_AsOf_March_2026_source_recency",
        desc="At least one cited source is recent (published/updated around 2025 or 2026).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage was published or substantively updated in 2025 or 2026, or it clearly presents T‑Mobile 5G information current for 2025/2026.",
        node=leaf_source_recent,
        sources=all_urls,
        additional_instruction="Check the page date line, update/press release date, or explicit context indicating 2025/2026. If any one page clearly matches, mark as supported."
    )


async def _verify_aspect_with_value_and_evidence(
    evaluator: Evaluator,
    parent,
    base_id: str,
    group_desc: str,
    answer_presence_claim: str,
    evidence_claim: str,
    urls: List[str],
    answer_presence_instruction: Optional[str] = None,
    evidence_instruction: Optional[str] = None,
) -> None:
    """
    Common builder for aspects where we:
      1) check the answer explicitly states the required info/value
      2) require at least one source URL
      3) verify the claim is supported by the cited source(s)
    """
    group = evaluator.add_parallel(
        id=base_id,
        desc=group_desc,
        parent=parent,
        critical=True
    )

    # Enforce that the answer explicitly includes the requested fact/value
    leaf_answer_has = evaluator.add_leaf(
        id=f"{base_id}_answer_mentions",
        desc="Answer explicitly includes the required statement/value for this aspect.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=answer_presence_claim,
        node=leaf_answer_has,
        additional_instruction=answer_presence_instruction or "Look only at the answer text."
    )

    # Require URLs (source grounding)
    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id=f"{base_id}_sources_provided",
        desc="Cited source URL(s) provided for this aspect.",
        parent=group,
        critical=True
    )

    # Verify the claim against the cited sources
    leaf_supported = evaluator.add_leaf(
        id=f"{base_id}_sources_supported",
        desc="Claim is supported by the cited source(s).",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=evidence_claim,
        node=leaf_supported,
        sources=urls,
        additional_instruction=evidence_instruction or "Focus on whether the page explicitly supports the claim."
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the T-Mobile 5G network profile (as of March 2026).
    """
    # Initialize evaluator with a parallel aggregation at the true root
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

    # A logical "task root" (critical) mirroring the rubric's named root
    rubric_root = evaluator.add_parallel(
        id="TMobile_5G_Network_Profile_AsOf_Mar2026",
        desc="Verify the profile covers all required aspects of T-Mobile's US 5G deployment as of March 2026 and that each aspect includes supporting evidence.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted: TMobile5GProfileExtraction = await evaluator.extract(
        prompt=prompt_extract_profile(),
        template_class=TMobile5GProfileExtraction,
        extraction_name="tmobile_5g_profile_structured"
    )

    # Add ground-truth expectations info (for transparency in summary only)
    evaluator.add_ground_truth(
        {
            "expected": EXPECTED_FACTS,
            "timeframe": "As of March 2026"
        },
        gt_type="expected_facts"
    )

    # Timeframe verification
    await _verify_timeframe(evaluator, rubric_root, extracted)

    # Build each aspect (presence + evidence)
    # 1) Total population coverage: 323M
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="Coverage_Total_Population_323M_With_Evidence",
        group_desc="States that T-Mobile's 5G covers ~323 million people in the US, with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile’s 5G (Extended Range) covers approximately 323 million people in the United States (minor rounding variations acceptable).",
        evidence_claim="T‑Mobile’s 5G (Extended Range) network covers around 323 million people in the United States.",
        urls=_safe_urls(extracted.coverage_total_population),
        answer_presence_instruction="Search the answer for '323 million' (allowing minor rounding/slight variants like 'about 323M').",
        evidence_instruction="Verify that the page explicitly mentions ~323 million people covered by T‑Mobile 5G (Extended Range). Allow small rounding."
    )

    # 2) Coverage percentage: 98%
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="Coverage_Percentage_98pct_With_Evidence",
        group_desc="States that T-Mobile's 5G coverage is ~98% of the US population, with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile’s 5G covers about 98% of the U.S. population (allowing minor rounding).",
        evidence_claim="T‑Mobile’s 5G network covers about 98% of the U.S. population.",
        urls=_safe_urls(extracted.coverage_percentage),
        evidence_instruction="Verify that the page explicitly mentions ~98% population coverage for T‑Mobile 5G. Small rounding OK."
    )

    # 3) Market position: leads in 5G availability
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="Market_Position_Leads_5G_Availability_With_Evidence",
        group_desc="States that T-Mobile leads other major US carriers in 5G availability, with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile leads other major U.S. carriers in 5G availability (per independent testing reports).",
        evidence_claim="T‑Mobile leads other major U.S. carriers in 5G availability.",
        urls=_safe_urls(extracted.market_position),
        evidence_instruction="Look for Opensignal/Ookla/Umlaut/etc. reports showing T‑Mobile leading in '5G Availability' or similar metric."
    )

    # 4) Low-band n71 600 MHz deployed
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="LowBand_n71_600MHz_Deployed_With_Evidence",
        group_desc="States that T-Mobile has deployed low-band 5G on n71 (600 MHz), with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile has deployed low‑band 5G on band n71 (600 MHz).",
        evidence_claim="T‑Mobile has deployed low‑band 5G using band n71 (600 MHz).",
        urls=_safe_urls(extracted.lowband_n71),
        evidence_instruction="Verify that the page clearly mentions band 'n71' at 600 MHz as 5G for T‑Mobile."
    )

    # 5) Mid-band n41 2.5 GHz deployed
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="MidBand_n41_2p5GHz_Deployed_With_Evidence",
        group_desc="States that T-Mobile has deployed mid-band 5G on n41 (2.5 GHz), with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile has deployed mid‑band 5G on band n41 (2.5 GHz).",
        evidence_claim="T‑Mobile has deployed mid‑band 5G using band n41 (2.5 GHz).",
        urls=_safe_urls(extracted.midband_n41),
        evidence_instruction="Verify that the page clearly mentions band 'n41' at 2.5 GHz as 5G for T‑Mobile."
    )

    # 6) C-band n77 3.7 GHz deployed
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="CBand_n77_3p7GHz_Deployed_With_Evidence",
        group_desc="States that T-Mobile has deployed C-band 5G on n77 (3.7 GHz), with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile has deployed C‑band 5G on band n77 (around 3.7 GHz).",
        evidence_claim="T‑Mobile has deployed C‑band 5G using band n77 (~3.7 GHz).",
        urls=_safe_urls(extracted.cband_n77),
        evidence_instruction="Verify that the page clearly mentions C‑band and band 'n77' (3.7 GHz) for T‑Mobile 5G."
    )

    # 7) mmWave in select areas
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="mmWave_Select_Areas_With_Evidence",
        group_desc="States that T-Mobile offers mmWave 5G in select areas, with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile offers millimeter wave (mmWave) 5G in select or limited areas (e.g., venues/urban hotspots).",
        evidence_claim="T‑Mobile offers millimeter wave (mmWave) 5G in select/limited areas.",
        urls=_safe_urls(extracted.mmwave),
        evidence_instruction="Verify that the page mentions 'mmWave'/'millimeter wave' 5G for T‑Mobile, limited/select area availability."
    )

    # 8) 5G Standalone (SA) deployed
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="5G_SA_Deployed_With_Evidence",
        group_desc="States that T-Mobile has deployed 5G Standalone (SA), with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile has deployed 5G Standalone (SA) network architecture (e.g., nationwide SA).",
        evidence_claim="T‑Mobile has deployed 5G Standalone (SA) network architecture.",
        urls=_safe_urls(extracted.sa),
        evidence_instruction="Verify that the page explicitly mentions 5G SA deployment for T‑Mobile."
    )

    # 9) VoNR deployed and over 100M coverage
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="VoNR_Deployed_Over_100M_With_Evidence",
        group_desc="States that T-Mobile has deployed VoNR and that it covers over 100M people, with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile has deployed VoNR (Voice over New Radio) and that VoNR coverage reaches over 100 million people.",
        evidence_claim="T‑Mobile has deployed VoNR and coverage reaches more than 100 million people.",
        urls=_safe_urls(extracted.vonr),
        evidence_instruction="Verify both that T‑Mobile has VoNR and that its VoNR coverage exceeds 100 million people."
    )

    # 10) Spectrum holdings ~194 MHz at 2.5 GHz
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="Spectrum_Holdings_Approx_194MHz_2p5GHz_With_Evidence",
        group_desc="States that T-Mobile holds approximately 194 MHz of 2.5 GHz spectrum (in many markets), with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile holds approximately 194 MHz of 2.5 GHz spectrum (e.g., in many markets).",
        evidence_claim="T‑Mobile holds approximately 194 MHz of 2.5 GHz spectrum (in many markets).",
        urls=_safe_urls(extracted.spectrum_holdings_2p5ghz),
        evidence_instruction="Verify that the page mentions ~194 MHz of 2.5 GHz spectrum holdings for T‑Mobile. Some markets phrasing acceptable."
    )

    # 11) Median 5G download speed ~299 Mbps (H1 2025)
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="Median_5G_Download_Speed_Approx_299Mbps_H1_2025_With_Evidence",
        group_desc="States that median 5G download speed was ~299 Mbps in H1 2025, with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile’s median 5G download speed was about 299 Mbps in H1 2025.",
        evidence_claim="T‑Mobile’s median 5G download speed was approximately 299 Mbps in H1 2025.",
        urls=_safe_urls(extracted.median_5g_speed_h1_2025),
        evidence_instruction="Prefer Ookla/Speedtest Intelligence or reputable measurement sources. Allow small rounding (e.g., 297–301 Mbps)."
    )

    # 12) Rural coverage expansion
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="Rural_Coverage_Expansion_With_Evidence",
        group_desc="States that T-Mobile has expanded 5G coverage into rural areas, with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile has expanded 5G coverage into rural areas (recent progress/status).",
        evidence_claim="T‑Mobile has expanded 5G coverage into rural areas.",
        urls=_safe_urls(extracted.rural_expansion),
        evidence_instruction="Look for official statements, coverage updates, or third‑party analyses on rural 5G expansion."
    )

    # 13) 5G-Advanced features including multi-band carrier aggregation
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="5G_Advanced_MultiBand_Carrier_Aggregation_With_Evidence",
        group_desc="States that T-Mobile supports 5G-Advanced features including multi-band carrier aggregation, with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile supports 5G‑Advanced features, including multi‑band carrier aggregation (e.g., combining n71, n41, n77).",
        evidence_claim="T‑Mobile supports 5G‑Advanced features including multi‑band 5G carrier aggregation.",
        urls=_safe_urls(extracted.fiveg_advanced_features),
        evidence_instruction="Verify that the page mentions 5G‑Advanced or multi‑band 5G carrier aggregation on T‑Mobile (e.g., 3xCA across low/mid/c‑band)."
    )

    # 14) Future plans: invest $8B, 6,000 new towers (rural expansion)
    await _verify_aspect_with_value_and_evidence(
        evaluator=evaluator,
        parent=rubric_root,
        base_id="Future_Plans_Invest_8B_Deploy_6000_Towers_Rural_With_Evidence",
        group_desc="States future plans to invest $8B and deploy 6,000 new towers for rural expansion, with evidence.",
        answer_presence_claim="The answer explicitly states that T‑Mobile plans to invest about $8 billion and deploy roughly 6,000 new towers for rural expansion.",
        evidence_claim="T‑Mobile plans to invest about $8 billion and deploy around 6,000 new towers to expand coverage in rural areas.",
        urls=_safe_urls(extracted.future_plans_investment),
        evidence_instruction="Verify that the page explicitly cites both figures: ~$8B investment and ~6,000 new towers for rural build‑out."
    )

    # Return final structured summary
    return evaluator.get_summary()