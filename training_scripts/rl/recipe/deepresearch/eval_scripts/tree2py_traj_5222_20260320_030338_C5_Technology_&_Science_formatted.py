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
TASK_ID = "us_5g_compare_march_2026"
TASK_DESCRIPTION = (
    "Compare the 5G network coverage of the three major U.S. wireless carriers (AT&T, T-Mobile, and Verizon) as of March 2026. "
    "For each carrier, provide: 1) Their 5G network coverage percentage in the United States, 2) Their 4G LTE coverage metric "
    "(percentage or population coverage), 3) Their network reliability score from Opensignal or a similar network measurement "
    "service (if available), and 4) Reference URL(s) supporting the metrics. All data must be from the 2025-2026 timeframe and "
    "come from reputable sources such as carrier official coverage maps, Opensignal reports, or other recognized network measurement services."
)

RECOGNIZED_MEASUREMENT_PROVIDERS = [
    "Opensignal",
    "Ookla",
    "RootMetrics",
    "umlaut",
    "PCMag",
    "FCC",
]

RECOGNIZED_REPUTABLE_DOMAINS = [
    # Official carriers
    "att.com",
    "t-mobile.com",
    "verizon.com",
    # Measurement services / reputable outlets
    "opensignal.com",
    "speedtest.net",      # Ookla
    "rootmetrics.com",
    "umlaut.com",
    "pcmag.com",
    "fcc.gov",
    "gsma.com",
    "lightreading.com",
    "fiercewireless.com",
]


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class CarrierMetrics(BaseModel):
    carrier_name: Optional[str] = None  # Expect one of: "AT&T", "T-Mobile", "Verizon"
    five_g_coverage_percent: Optional[str] = None  # numeric percentage string as written, e.g., "92%" or "92.0%"
    five_g_urls: List[str] = Field(default_factory=list)

    lte_coverage_metric: Optional[str] = None  # string, can be percentage or population coverage string
    lte_metric_label: Optional[str] = None     # e.g., "percentage", "population", "geographic"
    lte_urls: List[str] = Field(default_factory=list)

    reliability_score: Optional[str] = None    # optional
    reliability_provider: Optional[str] = None # e.g., "Opensignal"
    reliability_urls: List[str] = Field(default_factory=list)

    timeframe_notes: Optional[str] = None      # any temporal qualifier text extracted from the answer


class CarriersExtraction(BaseModel):
    carriers: List[CarrierMetrics] = Field(default_factory=list)
    carriers_in_answer: List[str] = Field(default_factory=list)  # any carrier brand names that appear in the answer
    has_comparison: Optional[bool] = None  # whether the answer offers an explicit comparison (table/side-by-side or text)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_carriers() -> str:
    return """
    Your task is to extract the reported metrics for exactly these three U.S. carriers: AT&T, T-Mobile, and Verizon.
    Return a JSON object following this schema:

    {
      "carriers": [
        {
          "carrier_name": "AT&T" | "T-Mobile" | "Verizon",
          "five_g_coverage_percent": string or null,  // a numeric percentage as it appears in the answer, e.g., "92%" (do not invent)
          "five_g_urls": string[]                     // all explicit URLs in the answer that support this 5G figure
          "lte_coverage_metric": string or null,      // a 4G LTE coverage metric as written (e.g., "99% population coverage" or "2.68M sq mi")
          "lte_metric_label": string or null,         // "percentage" | "population" | "geographic" (pick the closest label based on the answer text)
          "lte_urls": string[],                       // all explicit URLs that support the 4G LTE metric
          "reliability_score": string or null,        // if a reliability score is mentioned; otherwise null
          "reliability_provider": string or null,     // e.g., "Opensignal", "RootMetrics", "Ookla", etc. If not mentioned, null
          "reliability_urls": string[],               // all explicit URLs that support the reliability score/provider
          "timeframe_notes": string or null           // any phrasing like "as of March 2026", "2025 report", etc. If absent, null
        },
        ... // include one object for each of: AT&T, T-Mobile, and Verizon. If the answer does not mention a metric, put null and empty array(s) as needed.
      ],
      "carriers_in_answer": string[],                 // list all carrier brand names that appear anywhere in the answer (e.g., "AT&T", "T-Mobile", "Verizon", "Dish", "UScellular", "Spectrum Mobile", etc.)
      "has_comparison": boolean or null               // true if the answer provides a clear cross-carrier comparison (side-by-side table or explicit comparative statements); false if not; null if unclear
    }

    STRICT RULES:
    - Only include these three carriers in the 'carriers' array: AT&T, T-Mobile, Verizon. If the answer mentions additional carriers, do NOT add them to 'carriers', but DO list them under 'carriers_in_answer'.
    - Extract only what is explicitly present in the answer. Do not infer or fabricate values.
    - For all URL fields, extract actual URLs verbatim from the answer text (including markdown links).
    - If a metric is missing, set the corresponding field to null and the URL list to [].
    - Preserve numeric values as strings exactly as written (e.g., keep the '%' sign if present).
    - The lte_metric_label must be one of: "percentage", "population", or "geographic" based on the phrasing in the answer (e.g., "population coverage" => "population", "covers X% of the U.S." => "percentage").
    - If the answer indicates a timeframe (e.g., "as of March 2026", "2025 report"), copy that phrase into 'timeframe_notes' for that carrier.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _find_carrier(extracted: CarriersExtraction, name: str) -> CarrierMetrics:
    for c in extracted.carriers:
        if (c.carrier_name or "").strip().lower().replace("‑", "-") == name.strip().lower().replace("‑", "-"):
            return c
    # Return empty placeholder for missing carrier
    return CarrierMetrics(carrier_name=name)


def _all_urls(extracted: CarriersExtraction) -> List[str]:
    urls: List[str] = []
    for c in extracted.carriers:
        urls.extend(c.five_g_urls or [])
        urls.extend(c.lte_urls or [])
        urls.extend(c.reliability_urls or [])
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def _coverage_urls_only(extracted: CarriersExtraction) -> List[str]:
    urls: List[str] = []
    for c in extracted.carriers:
        urls.extend(c.five_g_urls or [])
        urls.extend(c.lte_urls or [])
    # Deduplicate
    seen = set()
    uniq: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification per carrier                                                    #
# --------------------------------------------------------------------------- #
async def verify_carrier_block(
    evaluator: Evaluator,
    parent_node,
    carrier_metrics: CarrierMetrics,
    block_id: str,
    display_name: str,
) -> None:
    """
    Build verification leaves for one carrier under a critical parallel parent block.
    The three required leaves are:
      - {prefix}_5g_metric_and_url
      - {prefix}_4g_metric_and_url
      - {prefix}_reliability_conditional
    """
    # 5G coverage percentage + URL(s)
    fiveg_node = evaluator.add_leaf(
        id=f"{block_id}_5g_metric_and_url",
        desc=f"Provides {display_name} 5G coverage percentage (numeric %) AND at least one supporting reference URL",
        parent=parent_node,
        critical=True,
    )
    fiveg_value = carrier_metrics.five_g_coverage_percent or ""
    fiveg_urls = carrier_metrics.five_g_urls or []

    fiveg_claim = (
        f"The answer includes a numeric percentage for {display_name} 5G coverage in the United States "
        f"('{fiveg_value}') and at least one of the provided reference URLs explicitly supports this figure "
        f"(allowing minor rounding)."
    )
    fiveg_instruction = (
        "This leaf should be marked Correct only if BOTH conditions are satisfied:\n"
        "1) The answer actually reports a numeric percentage for 5G coverage for the specified carrier.\n"
        "2) At least one of the cited URLs explicitly supports that figure (minor rounding differences are OK).\n"
        "- Treat synonyms like 'US', 'United States', or 'nationwide (US)' as US scope.\n"
        "- If the answer provides no numeric percent or no URL, return Incorrect.\n"
        "- Prefer 2025–2026 sources; if the figure is clearly from 2025 or 2026, it's acceptable."
    )
    await evaluator.verify(
        claim=fiveg_claim,
        node=fiveg_node,
        sources=fiveg_urls if len(fiveg_urls) > 0 else None,
        additional_instruction=fiveg_instruction,
    )

    # 4G LTE coverage metric + URL(s)
    lte_node = evaluator.add_leaf(
        id=f"{block_id}_4g_metric_and_url",
        desc=f"Provides a {display_name} 4G LTE coverage metric (percentage or population coverage, clearly labeled) AND at least one supporting reference URL",
        parent=parent_node,
        critical=True,
    )
    lte_value = carrier_metrics.lte_coverage_metric or ""
    lte_label = (carrier_metrics.lte_metric_label or "").lower()
    lte_urls = carrier_metrics.lte_urls or []

    lte_claim = (
        f"The answer provides a 4G LTE coverage metric for {display_name} in the United States "
        f"('{lte_value}' labeled as '{lte_label if lte_label else 'unknown'}') and at least one of the provided URLs supports it."
    )
    lte_instruction = (
        "Mark Correct only if BOTH conditions are satisfied:\n"
        "1) The answer includes a clear 4G LTE coverage metric for the carrier (accepted forms: percentage or population coverage or geographic coverage).\n"
        "2) At least one cited URL explicitly supports that metric for the US.\n"
        "- If the answer provides no metric string or no URL, return Incorrect.\n"
        "- Allow minor formatting differences; ensure the metric scope is US (geographic % or population coverage within the US)."
    )
    await evaluator.verify(
        claim=lte_claim,
        node=lte_node,
        sources=lte_urls if len(lte_urls) > 0 else None,
        additional_instruction=lte_instruction,
    )

    # Reliability conditional (optional but constrained)
    rel_node = evaluator.add_leaf(
        id=f"{block_id}_reliability_conditional",
        desc=f"If a reliability score is reported for {display_name}, it is from Opensignal or a similar recognized measurement service AND includes at least one supporting reference URL (otherwise, omission is acceptable)",
        parent=parent_node,
        critical=True,
    )
    rel_score = carrier_metrics.reliability_score or ""
    rel_provider = carrier_metrics.reliability_provider or ""
    rel_urls = carrier_metrics.reliability_urls or []

    rel_claim = (
        f"Either the answer omits any reliability score for {display_name} (which is acceptable), "
        f"OR, if it reports one (reported score: '{rel_score}' from provider: '{rel_provider}'), "
        f"then the provider is a recognized measurement service (e.g., {', '.join(RECOGNIZED_MEASUREMENT_PROVIDERS)}) "
        f"and at least one cited URL supports that score/provider."
    )
    rel_instruction = (
        "Decision rules:\n"
        "- If NO reliability score is present in the answer for this carrier (no score text and no provider), mark Correct.\n"
        "- If a reliability score IS present, it must:\n"
        "   (a) Clearly come from a recognized provider such as Opensignal, RootMetrics, Ookla, umlaut, PCMag, or FCC, and\n"
        "   (b) Be supported by at least one of the provided URLs.\n"
        "- If a score is present but provider is unknown/unrecognized OR no supporting URL exists, mark Incorrect."
    )
    await evaluator.verify(
        claim=rel_claim,
        node=rel_node,
        sources=rel_urls if len(rel_urls) > 0 else None,
        additional_instruction=rel_instruction,
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
    """
    Evaluate an answer for the U.S. 5G coverage comparison task (March 2026 timeframe).
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root-level: parallel aggregation of major checks
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

    # Extraction
    extracted: CarriersExtraction = await evaluator.extract(
        prompt=prompt_extract_carriers(),
        template_class=CarriersExtraction,
        extraction_name="carrier_metrics_extraction",
    )

    # Prepare per-carrier metrics (ensure all three exist, even if empty)
    att = _find_carrier(extracted, "AT&T")
    tmo = _find_carrier(extracted, "T-Mobile")
    vz = _find_carrier(extracted, "Verizon")

    # ---------------- Global/Root-level requirements -------------------- #
    # 1) Carrier set requirement
    carrier_set_node = evaluator.add_leaf(
        id="carrier_set_requirement",
        desc="Covers exactly these three carriers (AT&T, T-Mobile, Verizon) and no additional carriers",
        parent=root,
        critical=True,
    )
    carriers_in_answer = extracted.carriers_in_answer or []
    carrier_set_claim = (
        "The answer includes metrics for exactly these three carriers and no others: AT&T, T-Mobile, and Verizon. "
        "It must not include additional carriers such as Dish, UScellular, Spectrum Mobile, Google Fi, etc."
    )
    carrier_set_instruction = (
        f"Carriers detected by extractor: {carriers_in_answer}. "
        "Judge by reading the answer text: if any additional carrier beyond AT&T, T-Mobile, and Verizon is included with metrics, return Incorrect. "
        "Allow generic mentions without metrics, but if the answer presents metrics for an extra carrier, that violates this requirement."
    )
    await evaluator.verify(
        claim=carrier_set_claim,
        node=carrier_set_node,
        additional_instruction=carrier_set_instruction,
    )

    # 2) Global data requirements (parallel, all critical)
    global_req = evaluator.add_parallel(
        id="global_data_requirements",
        desc="Global data quality and scope requirements for all reported metrics",
        parent=root,
        critical=True,
    )

    # 2.1 timeframe requirement
    timeframe_leaf = evaluator.add_leaf(
        id="timeframe_requirement",
        desc="All reported metrics reflect the 2025–2026 timeframe (around March 2026)",
        parent=global_req,
        critical=True,
    )
    # Using simple verification because 'verify_by_urls' for multiple URLs returns true if any single URL supports the claim.
    # Here we require that the answer’s reported metrics are from 2025–2026. The judge should read the answer itself.
    all_urls_list = _all_urls(extracted)
    timeframe_claim = (
        "All reported metrics in the answer are explicitly from the 2025–2026 timeframe (around March 2026). "
        "This can be shown via statements like 'as of March 2026' or by referencing 2025/2026 reports."
    )
    timeframe_instruction = (
        "Decide based on the answer text. The requirement is satisfied if the answer clearly indicates that the metrics come "
        "from 2025 or 2026 (e.g., 'as of March 2026', '2025 report'). "
        "If the answer fails to establish 2025–2026 timeframe for its reported metrics, return Incorrect. "
        f"Cited URLs (for your reference, do not fetch): {all_urls_list}"
    )
    await evaluator.verify(
        claim=timeframe_claim,
        node=timeframe_leaf,
        additional_instruction=timeframe_instruction,
    )

    # 2.2 U.S. scope requirement
    us_scope_leaf = evaluator.add_leaf(
        id="us_scope_requirement",
        desc="All coverage metrics are explicitly U.S.-scoped (U.S. geographic %, U.S. population coverage, or clearly stated U.S. scope)",
        parent=global_req,
        critical=True,
    )
    coverage_urls = _coverage_urls_only(extracted)
    us_scope_claim = (
        "All coverage metrics presented in the answer are explicitly scoped to the United States "
        "(geographic % of US land area, % of US population, or clearly stated US scope)."
    )
    us_scope_instruction = (
        "Judge from the answer text: references to 'US', 'United States', '% of US population', or '% of US geographic area' satisfy this. "
        "If any reported coverage metric lacks explicit US scope, return Incorrect. "
        f"Cited coverage URLs (for your reference, do not fetch): {coverage_urls}"
    )
    await evaluator.verify(
        claim=us_scope_claim,
        node=us_scope_leaf,
        additional_instruction=us_scope_instruction,
    )

    # 2.3 Source reputation requirement
    reputation_leaf = evaluator.add_leaf(
        id="source_reputation_requirement",
        desc="All cited sources are reputable (e.g., carrier official coverage maps, Opensignal, or other recognized network measurement services)",
        parent=global_req,
        critical=True,
    )
    reputation_claim = (
        "All sources cited in the answer come from reputable domains such as official carrier sites "
        "(att.com, t-mobile.com, verizon.com) and/or recognized measurement services "
        "(Opensignal, Ookla/speedtest.net, RootMetrics, umlaut, PCMag, FCC), or similarly reputable industry publications."
    )
    reputation_instruction = (
        "Decide from the domain names shown in the answer. Treat the following domains as reputable examples: "
        f"{RECOGNIZED_REPUTABLE_DOMAINS}. "
        f"List of extracted URLs to consider (judge by domain only; do not fetch): {all_urls_list}. "
        "If any primary metric relies on a clearly non-reputable or obscure source, return Incorrect."
    )
    await evaluator.verify(
        claim=reputation_claim,
        node=reputation_leaf,
        additional_instruction=reputation_instruction,
    )

    # 3) Comparison requirement
    comparison_leaf = evaluator.add_leaf(
        id="comparison_requirement",
        desc="Provides a clear comparison across the three carriers (e.g., side-by-side table and/or explicit comparative statements based on the reported metrics)",
        parent=root,
        critical=True,
    )
    has_comparison_text = f"Extractor flag has_comparison={extracted.has_comparison}"
    comparison_claim = (
        "The answer provides a clear comparison across AT&T, T-Mobile, and Verizon based on the reported metrics, "
        "either via a side-by-side list/table or explicit comparative statements."
    )
    comparison_instruction = (
        f"{has_comparison_text}. Treat concise bullet-by-bullet or tabular comparisons as sufficient. "
        "If the answer only lists each carrier individually without any clear comparison or cross-referencing, return Incorrect."
    )
    await evaluator.verify(
        claim=comparison_claim,
        node=comparison_leaf,
        additional_instruction=comparison_instruction,
    )

    # ---------------- Per-carrier information blocks -------------------- #
    # AT&T
    att_block = evaluator.add_parallel(
        id="ATT_information",
        desc="AT&T metrics and citations",
        parent=root,
        critical=True,
    )
    await verify_carrier_block(
        evaluator=evaluator,
        parent_node=att_block,
        carrier_metrics=att,
        block_id="att",
        display_name="AT&T",
    )

    # T-Mobile
    tmo_block = evaluator.add_parallel(
        id="TMobile_information",
        desc="T-Mobile metrics and citations",
        parent=root,
        critical=True,
    )
    await verify_carrier_block(
        evaluator=evaluator,
        parent_node=tmo_block,
        carrier_metrics=tmo,
        block_id="tmobile",
        display_name="T-Mobile",
    )

    # Verizon
    vz_block = evaluator.add_parallel(
        id="Verizon_information",
        desc="Verizon metrics and citations",
        parent=root,
        critical=True,
    )
    await verify_carrier_block(
        evaluator=evaluator,
        parent_node=vz_block,
        carrier_metrics=vz,
        block_id="verizon",
        display_name="Verizon",
    )

    # Optionally record some custom info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "recognized_measurement_providers": RECOGNIZED_MEASUREMENT_PROVIDERS,
            "recognized_reputable_domains": RECOGNIZED_REPUTABLE_DOMAINS,
            "all_extracted_urls": _all_urls(extracted),
        },
        info_type="context",
        info_name="recognition_policy",
    )

    # Return standardized evaluation summary
    return evaluator.get_summary()