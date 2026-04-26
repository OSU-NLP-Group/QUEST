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
TASK_ID = "fcc_5g_carrier_criteria_2025"
TASK_DESCRIPTION = """
Based on the most recent Federal Communications Commission (FCC) 5G coverage data and independent network testing reports from 2024-2025, identify which major US wireless carrier (among T-Mobile, AT&T, and Verizon) satisfies ALL of the following five criteria:

1. Has the highest percentage of US land area covered by 5G service at the 7 Mbps download / 1 Mbps upload speed tier according to FCC data published in December 2025
2. Has the highest percentage of US land area covered by 5G service at the 35 Mbps download / 3 Mbps upload speed tier according to FCC data published in December 2025
3. Leads among the three major carriers in 5G availability metrics (measuring how frequently users can access 5G signals) according to independent testing organizations' 2024-2025 reports
4. Wins the 5G Coverage Experience award in Opensignal's USA Mobile Network Experience Report for January 2025
5. Officially claims to provide 5G coverage to at least 98% of the American population according to their published coverage information

Provide the name of the carrier and cite specific data points supporting each criterion.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CarrierEvidence(BaseModel):
    # A short free-text data point cited in the answer (e.g., "T-Mobile 57% land area @7/1 in Dec 2025 FCC")
    # Optional; used only for logging/context.
    data_point: Optional[str] = None
    # All URLs explicitly provided in the answer text that support this criterion
    sources: List[str] = Field(default_factory=list)


class CarrierExtraction(BaseModel):
    # The carrier the answer claims meets all five criteria; normalize to one of:
    # "T-Mobile", "AT&T", "Verizon" if possible; else leave as given by the answer
    carrier: Optional[str] = None

    # Evidence per criterion
    fcc_7_1: Optional[CarrierEvidence] = None
    fcc_35_3: Optional[CarrierEvidence] = None
    availability: Optional[CarrierEvidence] = None
    opensignal_2025: Optional[CarrierEvidence] = None
    population_98: Optional[CarrierEvidence] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_carrier_and_evidence() -> str:
    return """
    Your task is to extract, from the provided answer text, the single major US wireless carrier (among "T-Mobile", "AT&T", and "Verizon") that the answer claims satisfies ALL five specified criteria, along with the cited sources for each criterion.

    Extract the following fields:
    - carrier: the carrier the answer concludes as meeting all five criteria. If the answer names multiple carriers, choose the one explicitly stated as satisfying all the listed criteria. Normalize values:
        * If the answer uses variants like "T Mobile", "T-Mobile US", or "TMUS", return "T-Mobile".
        * If it uses "AT&T Mobility" or "ATT", return "AT&T".
        * If it uses "Verizon Wireless" or "VZ", return "Verizon".
        If uncertain, return null.

    For each criterion below, extract:
      - data_point: a short snippet or number the answer cites for this criterion (optional; null if not provided).
      - sources: an array of all explicit URLs in the answer that support this specific criterion. Only include URLs actually present in the answer (including markdown links). If none are present, return an empty array.

    Criteria objects:
    - fcc_7_1: Evidence that the carrier leads at the 7 Mbps down / 1 Mbps up tier in FCC 5G land-area coverage, specifically according to FCC data published in December 2025.
    - fcc_35_3: Evidence that the carrier leads at the 35 Mbps down / 3 Mbps up tier in FCC 5G land-area coverage, specifically according to FCC data published in December 2025.
    - availability: Evidence from independent testing organizations (e.g., Opensignal, Ookla, RootMetrics, umlaut) in 2024 or 2025 that the carrier leads in 5G availability (how often users are connected to 5G), versus AT&T and Verizon.
    - opensignal_2025: Evidence that in Opensignal's USA Mobile Network Experience Report (January 2025), the carrier won the "5G Coverage Experience" award.
    - population_98: Evidence that the carrier’s official site or press materials claim 5G coverage for at least 98% of Americans (population coverage claim).

    Important rules:
    - Return only URLs explicitly present in the answer text; do not invent any.
    - If the answer cites non-URL references (e.g., “FCC report Dec 2025” without a link), treat 'sources' for that criterion as an empty array.
    - If any item is missing, set it to null (for data_point) or [] (for sources).

    Return a JSON object with this exact structure:
    {
      "carrier": string or null,
      "fcc_7_1": { "data_point": string or null, "sources": string[] },
      "fcc_35_3": { "data_point": string or null, "sources": string[] },
      "availability": { "data_point": string or null, "sources": string[] },
      "opensignal_2025": { "data_point": string or null, "sources": string[] },
      "population_98": { "data_point": string or null, "sources": string[] }
    }
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _norm_carrier(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    if any(x in s for x in ["t-mobile", "t mobile", "tmus", "t‑mobile"]):
        return "T-Mobile"
    if any(x in s for x in ["at&t", "att", "at & t"]):
        return "AT&T"
    if any(x in s for x in ["verizon", "vz", "verizon wireless"]):
        return "Verizon"
    # If it already matches exactly one of the expected forms (case-insensitive)
    if s in ["t-mobile", "at&t", "verizon"]:
        return name.strip().title() if s != "at&t" else "AT&T"
    return name.strip()


def _get_sources(ev: Optional[CarrierEvidence]) -> List[str]:
    if not ev or not ev.sources:
        return []
    # Keep only non-empty strings
    return [u for u in ev.sources if isinstance(u, str) and u.strip()]


async def _verify_with_sources_or_fail(
    evaluator: Evaluator,
    node_id: str,
    desc: str,
    parent_node,
    claim: str,
    sources: List[str],
    additional_instruction: str,
    critical: bool = True,
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )
    # Enforce source-grounding: if no sources are provided for a web-grounded claim, fail immediately.
    if not sources:
        leaf.score = 0.0
        leaf.status = "failed"
        return

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Build verification tree and checks                                          #
# --------------------------------------------------------------------------- #
async def verify_carrier_identification(evaluator: Evaluator, root, extraction: CarrierExtraction) -> None:
    carrier = _norm_carrier(extraction.carrier)
    ev_7_1 = _get_sources(extraction.fcc_7_1)
    ev_35_3 = _get_sources(extraction.fcc_35_3)
    ev_avail = _get_sources(extraction.availability)
    ev_os_2025 = _get_sources(extraction.opensignal_2025)
    ev_pop98 = _get_sources(extraction.population_98)

    # Add a critical parent node to mirror the rubric
    cid_node = evaluator.add_parallel(
        id="Carrier_Identification",
        desc="Identify the US wireless carrier that meets all specified 5G coverage and performance criteria based on 2025 data",
        parent=root,
        critical=True,
    )

    # FCC 7/1 leader (critical)
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="FCC_7_1_Coverage_Leader",
        desc="The carrier has the highest percentage of US land area covered by 5G at the 7 Mbps download / 1 Mbps upload tier according to FCC data from December 2025",
        parent_node=cid_node,
        claim=f"According to FCC 5G coverage data published in December 2025, among T-Mobile, AT&T, and Verizon, {carrier} has the highest percentage of U.S. land area covered by 5G at the 7 Mbps download / 1 Mbps upload speed tier.",
        sources=ev_7_1,
        additional_instruction=(
            "Accept only if the page(s) explicitly reference FCC's December 2025 5G coverage release "
            "or clearly cite December 2025 FCC land-area coverage at the 7/1 Mbps tier, and show that the named carrier "
            "ranks highest among T-Mobile, AT&T, and Verizon. Minor wording variants like '7/1', '7 down/1 up', or " 
            "'≥7/1 Mbps' are acceptable."
        ),
        critical=True,
    )

    # FCC 35/3 leader (critical)
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="FCC_35_3_Coverage_Leader",
        desc="The carrier has the highest percentage of US land area covered by 5G at the 35 Mbps download / 3 Mbps upload tier according to FCC data from December 2025",
        parent_node=cid_node,
        claim=f"According to FCC 5G coverage data published in December 2025, among T-Mobile, AT&T, and Verizon, {carrier} has the highest percentage of U.S. land area covered by 5G at the 35 Mbps download / 3 Mbps upload speed tier.",
        sources=ev_35_3,
        additional_instruction=(
            "Accept only if the page(s) explicitly reference FCC's December 2025 5G coverage release "
            "or clearly cite December 2025 FCC land-area coverage at the 35/3 Mbps tier, and show that the named carrier "
            "ranks highest among T-Mobile, AT&T, and Verizon. Minor wording variants like '35/3', '35 down/3 up', or " 
            "'≥35/3 Mbps' are acceptable."
        ),
        critical=True,
    )

    # 5G availability winner in 2024-2025 independent tests (critical)
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="5G_Availability_Winner",
        desc="The carrier leads among major US carriers in 5G availability metrics (frequency of 5G signal access) according to 2024-2025 independent testing reports",
        parent_node=cid_node,
        claim=f"Independent U.S. network testing reports from 2024 or 2025 (such as Opensignal, Ookla, RootMetrics, or umlaut) show that {carrier} leads AT&T and Verizon in 5G availability (how often users are connected to 5G).",
        sources=ev_avail,
        additional_instruction=(
            "Verify from credible U.S. testing organizations in 2024 or 2025 (e.g., Opensignal, Ookla, RootMetrics, umlaut) "
            "that the named carrier leads in a '5G availability' metric against AT&T and Verizon. Synonyms include 'time on 5G', "
            "'share of time on 5G', or similarly defined availability measures. Prefer national U.S. results; if only regional "
            "studies are cited, they should still make clear an overall leadership vs both competitors."
        ),
        critical=True,
    )

    # Opensignal January 2025 5G Coverage Experience award (critical)
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="Opensignal_Coverage_Experience_2025",
        desc="The carrier wins the 5G Coverage Experience award in Opensignal's USA January 2025 Mobile Network Experience Report",
        parent_node=cid_node,
        claim=f"In Opensignal's USA Mobile Network Experience Report (January 2025), {carrier} won the '5G Coverage Experience' award.",
        sources=ev_os_2025,
        additional_instruction=(
            "Confirm using Opensignal's USA Mobile Network Experience Report dated January 2025 that the named carrier won "
            "the '5G Coverage Experience' award specifically. Exact category name '5G Coverage Experience' must match, but "
            "minor casing differences are acceptable."
        ),
        critical=True,
    )

    # Population coverage claim ≥ 98% (critical)
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="Population_Coverage_98_Percent",
        desc="The carrier officially claims to provide 5G coverage to at least 98% of Americans according to their published coverage information",
        parent_node=cid_node,
        claim=f"According to its own official website or press materials, {carrier} claims that its 5G network covers at least 98% of Americans (U.S. population).",
        sources=ev_pop98,
        additional_instruction=(
            "Accept only official claims from the carrier's own site or press materials (not third-party summaries). "
            "Phrasings like 'covers 98% of Americans', 'reaches at least 98% of the U.S. population', or '≥98% population coverage' are acceptable. "
            "If the percentage is below 98% or the page is not an official carrier source, the claim is not supported."
        ),
        critical=True,
    )

    # Record a small custom info block for transparency
    evaluator.add_custom_info(
        info={
            "extracted_carrier_raw": extraction.carrier,
            "normalized_carrier": carrier,
            "source_counts": {
                "fcc_7_1": len(ev_7_1),
                "fcc_35_3": len(ev_35_3),
                "availability": len(ev_avail),
                "opensignal_2025": len(ev_os_2025),
                "population_98": len(ev_pop98),
            }
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
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
    # Initialize evaluator (root is non-critical by design)
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

    # Extraction
    extraction: CarrierExtraction = await evaluator.extract(
        prompt=prompt_extract_carrier_and_evidence(),
        template_class=CarrierExtraction,
        extraction_name="carrier_and_criteria_evidence",
    )

    # Build verification sub-tree and run checks
    await verify_carrier_identification(evaluator, root, extraction)

    # Return standard summary
    return evaluator.get_summary()