import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "telecom_outage_compare_2026_2024"
TASK_DESCRIPTION = (
    "Conduct a comprehensive comparative analysis of two major telecommunications network outages in the United States: "
    "the Verizon wireless outage that occurred in January 2026 and the AT&T Mobility outage that occurred in February 2024. "
    "For each outage, provide the following specific information with supporting reference URLs: "
    "(1) The exact date and time when the outage began, "
    "(2) The technical root cause of the outage as officially reported, "
    "(3) The total duration of the outage, "
    "(4) The number of devices or users affected, "
    "(5) Whether the outage impacted 911 emergency service access and the extent of impact, "
    "(6) The timeline by which the carrier was required to submit an initial outage report to the FCC's Network Outage Reporting System (NORS), "
    "(7) The compensation or credits offered to affected customers, and "
    "(8) The time when the carrier officially announced the outage was resolved. "
    "Additionally, identify which outage had a longer duration and which affected more devices or users."
)

# Expected key facts (used to form claims for verification against cited sources)
EXPECTED = {
    "verizon": {
        "start_date": "January 14, 2026",
        "root_cause": "software issue",
        "duration_resolution": "several hours and was resolved by 10:15 PM ET on January 14, 2026",
        "devices": "approximately 180,000 users (sometimes described as tens of thousands)",
        "impact_911": "impacted access to 911 (multiple emergency services warned calls might not connect)",
        "fcc_timeline": "initial NORS report within 72 hours (3 calendar days) of discovering the outage",
        "compensation": "$20 account credits",
        "resolution_time": "10:15 PM ET on January 14, 2026",
    },
    "att": {
        "start_datetime": "February 22, 2024 at 2:42 AM",
        "root_cause": "equipment configuration error during a network change",
        "duration": "at least 12 hours",
        "devices": "over 125 million devices",
        "impact_911": "blocked more than 92 million voice calls total, including more than 25,000 calls to 911",
        "fcc_timeline": "initial NORS report within 72 hours (3 calendar days) of discovering the outage",
        "compensation": "offered/planned credits to affected consumers and has an AT&T Guarantee program",
        "resolution_note": "the outage lasted at least 12 hours from a 2:42 AM start on Feb 22, 2024",
    },
    "comparative": {
        "longer_duration": "AT&T February 2024 outage had a longer duration (≥12 hours) than Verizon’s January 2026 outage (several hours, resolved same day)",
        "more_devices": "AT&T February 2024 outage affected more devices (>125M) than Verizon’s January 2026 outage (~180k users)",
    }
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class FieldValue(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OutageDetails(BaseModel):
    # (1) Outage start date/time
    start_datetime: Optional[FieldValue] = None
    # (2) Technical root cause
    root_cause: Optional[FieldValue] = None
    # (3) Total duration
    duration: Optional[FieldValue] = None
    # (4) Number of devices/users affected
    affected: Optional[FieldValue] = None
    # (5) 911 impact and extent
    impact_911: Optional[FieldValue] = None
    # (6) FCC NORS initial reporting timeline
    fcc_nors_timeline: Optional[FieldValue] = None
    # (7) Compensation/credits offered
    compensation: Optional[FieldValue] = None
    # (8) Official resolution time
    resolution_time: Optional[FieldValue] = None
    # General supporting URLs for the outage (besides per-field URLs)
    supporting_urls: List[str] = Field(default_factory=list)


class ComparativeInfo(BaseModel):
    longer_duration: Optional[FieldValue] = None
    more_devices: Optional[FieldValue] = None


class OutagesExtraction(BaseModel):
    verizon: Optional[OutageDetails] = None
    att: Optional[OutageDetails] = None
    comparison: Optional[ComparativeInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outages() -> str:
    return """
    Extract structured information about two outages: (A) Verizon wireless outage (January 2026) and (B) AT&T Mobility outage (February 2024).
    Extract ONLY what is explicitly present in the answer. Do not invent details.

    For each outage ("verizon" and "att"), extract the following fields. For each field, also extract the URLs explicitly cited in the answer that support that field. If multiple URLs are cited, include them all.

    Fields per outage (each is an object {value, sources[]}):
    - start_datetime: The exact date/time the outage began, as stated in the answer.
    - root_cause: The official/technical root cause as reported.
    - duration: The total duration as stated (e.g., 'several hours', 'at least 12 hours').
    - affected: The number of devices/users affected (e.g., 'approximately 180,000 users', 'over 125 million devices').
    - impact_911: Whether/how 911 was impacted, with any stated quantitative details (e.g., 'blocked 25,000+ 911 calls').
    - fcc_nors_timeline: The timeline by which the carrier was required to submit the initial NORS report (e.g., 'within 72 hours').
    - compensation: Any compensation/credits offered to affected customers.
    - resolution_time: The time when the carrier officially announced resolution.
    Also extract "supporting_urls": an array of any general reference URLs cited for that outage (do not duplicate per-field URLs).

    Additionally, extract a "comparison" object with two fields, each as {value, sources[]}:
    - longer_duration: Which outage had a longer duration (e.g., 'AT&T' or 'AT&T 2024'), as claimed by the answer; include any URLs that support this comparison.
    - more_devices: Which outage affected more devices/users, as claimed by the answer; include any URLs that support this comparison.

    Return a JSON object with this schema:
    {
      "verizon": {
        "start_datetime": {"value": str|null, "sources": [urls...]},
        "root_cause": {"value": str|null, "sources": [urls...]},
        "duration": {"value": str|null, "sources": [urls...]},
        "affected": {"value": str|null, "sources": [urls...]},
        "impact_911": {"value": str|null, "sources": [urls...]},
        "fcc_nors_timeline": {"value": str|null, "sources": [urls...]},
        "compensation": {"value": str|null, "sources": [urls...]},
        "resolution_time": {"value": str|null, "sources": [urls...]},
        "supporting_urls": [urls...]
      },
      "att": {
        "start_datetime": {"value": str|null, "sources": [urls...]},
        "root_cause": {"value": str|null, "sources": [urls...]},
        "duration": {"value": str|null, "sources": [urls...]},
        "affected": {"value": str|null, "sources": [urls...]},
        "impact_911": {"value": str|null, "sources": [urls...]},
        "fcc_nors_timeline": {"value": str|null, "sources": [urls...]},
        "compensation": {"value": str|null, "sources": [urls...]},
        "resolution_time": {"value": str|null, "sources": [urls...]},
        "supporting_urls": [urls...]
      },
      "comparison": {
        "longer_duration": {"value": str|null, "sources": [urls...]},
        "more_devices": {"value": str|null, "sources": [urls...]}
      }
    }

    URL extraction rules:
    - Extract only URLs explicitly present in the answer. If none are present for a field, set its sources to an empty list.
    - Accept plain URLs or markdown links; output the actual URL strings.
    - Ensure all URLs include http:// or https://; if missing, prepend http://
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _val(fv: Optional[FieldValue]) -> str:
    return fv.value if (fv and fv.value) else ""


def _srcs(fv: Optional[FieldValue]) -> List[str]:
    return fv.sources if (fv and fv.sources) else []


def _merge_sources(*args: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in args:
        if not lst:
            continue
        for url in lst:
            if isinstance(url, str):
                u = url.strip()
                if u and u not in seen:
                    seen.add(u)
                    merged.append(u)
    return merged


def _collect_outage_all_sources(od: Optional[OutageDetails]) -> List[str]:
    if not od:
        return []
    return _merge_sources(
        _srcs(od.start_datetime),
        _srcs(od.root_cause),
        _srcs(od.duration),
        _srcs(od.affected),
        _srcs(od.impact_911),
        _srcs(od.fcc_nors_timeline),
        _srcs(od.compensation),
        _srcs(od.resolution_time),
        od.supporting_urls,
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_verizon_subtree(
    evaluator: Evaluator,
    parent_node,
    verizon: Optional[OutageDetails],
) -> Dict[str, Any]:
    """
    Build the Verizon January 2026 outage subtree with leaf verifications aligned to the rubric.
    Returns references to some key leaf nodes for potential dependency in comparative checks.
    """
    node_verizon = evaluator.add_parallel(
        id="verizon_january_2026_outage",
        desc="Complete analysis of the Verizon wireless network outage that occurred in January 2026",
        parent=parent_node,
        critical=False,
    )

    ver_all_urls = _collect_outage_all_sources(verizon)

    # 1) Outage date/time
    n_date = evaluator.add_leaf(
        id="verizon_outage_date_time",
        desc="Correctly identifies that the Verizon outage occurred on January 14, 2026",
        parent=node_verizon,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Verizon wireless outage occurred on {EXPECTED['verizon']['start_date']}.",
        node=n_date,
        sources=_merge_sources(_srcs(verizon.start_datetime), verizon.supporting_urls),
        additional_instruction="Confirm that the page explicitly mentions January 14, 2026 as the outage date; accept reasonable timezone context.",
    )

    # 2) Root cause
    n_root = evaluator.add_leaf(
        id="verizon_root_cause",
        desc="Correctly identifies that the outage was caused by a software issue",
        parent=node_verizon,
        critical=True,
    )
    await evaluator.verify(
        claim="The official reported root cause of the Verizon January 2026 outage was a software issue.",
        node=n_root,
        sources=_merge_sources(_srcs(verizon.root_cause), verizon.supporting_urls),
        additional_instruction="Look for explicit statements attributing the outage to a software issue.",
    )

    # 3) Duration
    n_duration = evaluator.add_leaf(
        id="verizon_duration",
        desc="Correctly states that the outage lasted several hours and was resolved by 10:15 PM ET on January 14, 2026",
        parent=node_verizon,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The outage lasted several hours and was resolved by {EXPECTED['verizon']['resolution_time']}.",
        node=n_duration,
        sources=_merge_sources(_srcs(verizon.duration), _srcs(verizon.resolution_time), verizon.supporting_urls),
        additional_instruction="Confirm both aspects: (1) 'several hours' duration characterization and (2) resolution time 10:15 PM ET on Jan 14, 2026.",
    )

    # 4) Devices/users affected
    n_devices = evaluator.add_leaf(
        id="verizon_devices_affected",
        desc="Provides the number of users/devices affected (approximately 180,000 users or tens of thousands)",
        parent=node_verizon,
        critical=True,
    )
    await evaluator.verify(
        claim="Approximately 180,000 users were affected by the Verizon January 2026 outage (some reports describe this as 'tens of thousands').",
        node=n_devices,
        sources=_merge_sources(_srcs(verizon.affected), verizon.supporting_urls),
        additional_instruction="Allow reasonable approximations and phrasings such as 'tens of thousands' if accompanied by ~180k figure in reliable reporting.",
    )

    # 5) 911 impact
    n_911 = evaluator.add_leaf(
        id="verizon_911_impact",
        desc="Correctly identifies that the outage impacted 911 emergency service access, with multiple emergency services warning calls might not connect",
        parent=node_verizon,
        critical=True,
    )
    await evaluator.verify(
        claim="The Verizon outage impacted access to 911; multiple emergency services warned Verizon customers that calls might not connect.",
        node=n_911,
        sources=_merge_sources(_srcs(verizon.impact_911), verizon.supporting_urls),
        additional_instruction="Look for explicit public safety or emergency management notices about 911 calling issues for Verizon customers.",
    )

    # 6) FCC NORS timeline
    n_fcc = evaluator.add_leaf(
        id="verizon_fcc_reporting_timeline",
        desc="Correctly states that Verizon was required to submit an initial NORS report within 72 hours (3 calendar days) of discovering the outage per FCC regulations",
        parent=node_verizon,
        critical=True,
    )
    await evaluator.verify(
        claim="Per FCC regulations, Verizon was required to submit an initial Network Outage Reporting System (NORS) report within 72 hours (3 calendar days) of discovering the outage.",
        node=n_fcc,
        sources=_merge_sources(_srcs(verizon.fcc_nors_timeline), verizon.supporting_urls),
        additional_instruction="Accept authoritative references that describe FCC Part 4/NORS initial reporting deadlines for significant outages.",
    )

    # 7) Compensation/credits
    n_comp = evaluator.add_leaf(
        id="verizon_compensation",
        desc="Correctly identifies that Verizon offered $20 account credits to affected customers",
        parent=node_verizon,
        critical=True,
    )
    await evaluator.verify(
        claim="Verizon offered $20 account credits to affected customers for the January 2026 outage.",
        node=n_comp,
        sources=_merge_sources(_srcs(verizon.compensation), verizon.supporting_urls),
        additional_instruction="Look for official Verizon statements or reliable reporting confirming $20 credits.",
    )

    # 8) Resolution time
    n_res = evaluator.add_leaf(
        id="verizon_resolution_time",
        desc="Correctly identifies that Verizon announced resolution at 10:15 PM ET on January 14, 2026",
        parent=node_verizon,
        critical=True,
    )
    await evaluator.verify(
        claim="Verizon announced the outage was resolved at 10:15 PM ET on January 14, 2026.",
        node=n_res,
        sources=_merge_sources(_srcs(verizon.resolution_time), verizon.supporting_urls),
        additional_instruction="Confirm the official resolution announcement time.",
    )

    # 9) Supporting URLs
    n_urls = evaluator.add_leaf(
        id="verizon_supporting_urls",
        desc="Provides valid reference URLs from official sources (Verizon, NPR, FCC, AARP, etc.) that support the information provided",
        parent=node_verizon,
        critical=True,
    )
    # If no URLs at all, fail immediately to enforce source-grounding
    if not ver_all_urls:
        n_urls.score = 0.0
        n_urls.status = "failed"
    else:
        await evaluator.verify(
            claim="This page is an official or reputable source (e.g., Verizon, NPR, FCC, AARP, Reuters) that reports details of the Verizon January 2026 outage.",
            node=n_urls,
            sources=ver_all_urls,
            additional_instruction="Pass if the page clearly discusses the Verizon January 2026 outage; prefer official or major reputable outlets.",
        )

    return {
        "duration_node": n_duration,
        "devices_node": n_devices,
    }


async def build_att_subtree(
    evaluator: Evaluator,
    parent_node,
    att: Optional[OutageDetails],
) -> Dict[str, Any]:
    """
    Build the AT&T February 2024 outage subtree with leaf verifications aligned to the rubric.
    Returns references to key leaf nodes for comparative dependencies.
    """
    node_att = evaluator.add_parallel(
        id="att_february_2024_outage",
        desc="Complete analysis of the AT&T Mobility network outage that occurred in February 2024",
        parent=parent_node,
        critical=False,
    )

    att_all_urls = _collect_outage_all_sources(att)

    # 1) Outage start date/time
    n_date = evaluator.add_leaf(
        id="att_outage_date_time",
        desc="Correctly identifies that the AT&T outage began on February 22, 2024 at 2:42 AM",
        parent=node_att,
        critical=True,
    )
    await evaluator.verify(
        claim="The AT&T Mobility outage began on February 22, 2024 at 2:42 AM.",
        node=n_date,
        sources=_merge_sources(_srcs(att.start_datetime), att.supporting_urls),
        additional_instruction="Confirm an explicit start timestamp (2:42 AM) on Feb 22, 2024; accept reasonable timezone context.",
    )

    # 2) Root cause
    n_root = evaluator.add_leaf(
        id="att_root_cause",
        desc="Correctly identifies that the outage was caused by an equipment configuration error during a network change",
        parent=node_att,
        critical=True,
    )
    await evaluator.verify(
        claim="The official reported root cause was an equipment configuration error during a network change.",
        node=n_root,
        sources=_merge_sources(_srcs(att.root_cause), att.supporting_urls),
        additional_instruction="Look for official AT&T or FCC statements describing a configuration error during planned changes.",
    )

    # 3) Duration
    n_duration = evaluator.add_leaf(
        id="att_duration",
        desc="Correctly states that the outage lasted at least 12 hours",
        parent=node_att,
        critical=True,
    )
    await evaluator.verify(
        claim="The outage lasted at least 12 hours.",
        node=n_duration,
        sources=_merge_sources(_srcs(att.duration), att.supporting_urls),
        additional_instruction="Confirm the duration is ≥12 hours from the reported start time.",
    )

    # 4) Devices/users affected
    n_devices = evaluator.add_leaf(
        id="att_devices_affected",
        desc="Correctly identifies that the outage affected over 125 million devices",
        parent=node_att,
        critical=True,
    )
    await evaluator.verify(
        claim="The outage affected over 125 million devices.",
        node=n_devices,
        sources=_merge_sources(_srcs(att.affected), att.supporting_urls),
        additional_instruction="Look for counts of devices impacted (e.g., >125 million) cited by AT&T or regulators.",
    )

    # 5) 911 impact
    n_911 = evaluator.add_leaf(
        id="att_911_impact",
        desc="Correctly identifies that the outage blocked more than 92 million voice calls total, including more than 25,000 calls to 911",
        parent=node_att,
        critical=True,
    )
    await evaluator.verify(
        claim="The outage blocked more than 92 million voice calls total, including more than 25,000 calls to 911.",
        node=n_911,
        sources=_merge_sources(_srcs(att.impact_911), att.supporting_urls),
        additional_instruction="Confirm both figures (>92 million total calls blocked, and >25,000 calls to 911).",
    )

    # 6) FCC NORS timeline
    n_fcc = evaluator.add_leaf(
        id="att_fcc_reporting_timeline",
        desc="Correctly states that AT&T was required to submit an initial NORS report within 72 hours (3 calendar days) of discovering the outage per FCC regulations",
        parent=node_att,
        critical=True,
    )
    await evaluator.verify(
        claim="Per FCC regulations, AT&T was required to submit an initial NORS report within 72 hours (3 calendar days) of discovering the outage.",
        node=n_fcc,
        sources=_merge_sources(_srcs(att.fcc_nors_timeline), att.supporting_urls),
        additional_instruction="Accept authoritative references that describe FCC Part 4/NORS initial reporting deadlines for significant outages.",
    )

    # 7) Compensation/credits
    n_comp = evaluator.add_leaf(
        id="att_compensation",
        desc="Identifies that AT&T planned to offer credits to affected consumers and has an AT&T Guarantee program for outages",
        parent=node_att,
        critical=True,
    )
    await evaluator.verify(
        claim="AT&T offered or planned to offer credits to affected consumers and maintains an AT&T Guarantee program for outages.",
        node=n_comp,
        sources=_merge_sources(_srcs(att.compensation), att.supporting_urls),
        additional_instruction="Look for AT&T statements about credits and their service guarantee program.",
    )

    # 8) Resolution timing info
    n_res = evaluator.add_leaf(
        id="att_resolution_time",
        desc="Provides information about when AT&T resolved the outage (noting the outage lasted at least 12 hours from 2:42 AM start)",
        parent=node_att,
        critical=True,
    )
    await evaluator.verify(
        claim="AT&T fully restored service after at least 12 hours from the 2:42 AM start on February 22, 2024.",
        node=n_res,
        sources=_merge_sources(_srcs(att.resolution_time), _srcs(att.duration), att.supporting_urls),
        additional_instruction="Confirm that resolution timing implies ≥12 hours elapsed from 2:42 AM start; accept precise or approximate restoration times.",
    )

    # 9) Supporting URLs
    n_urls = evaluator.add_leaf(
        id="att_supporting_urls",
        desc="Provides valid reference URLs from official sources (FCC reports, AT&T, Reuters, etc.) that support the information provided",
        parent=node_att,
        critical=True,
    )
    if not att_all_urls:
        n_urls.score = 0.0
        n_urls.status = "failed"
    else:
        await evaluator.verify(
            claim="This page is an official or reputable source (e.g., FCC, AT&T, Reuters) that reports details of the AT&T February 2024 outage.",
            node=n_urls,
            sources=att_all_urls,
            additional_instruction="Pass if the page clearly discusses the AT&T February 2024 outage; prefer official or major reputable outlets.",
    )

    return {
        "duration_node": n_duration,
        "devices_node": n_devices,
    }


async def build_comparative_subtree(
    evaluator: Evaluator,
    parent_node,
    verizon_info: Optional[OutageDetails],
    att_info: Optional[OutageDetails],
    verizon_nodes: Dict[str, Any],
    att_nodes: Dict[str, Any],
) -> None:
    """
    Build the comparative analysis subtree with two leaf verifications:
    - longer_duration_identification
    - more_devices_identification
    """
    node_cmp = evaluator.add_parallel(
        id="comparative_analysis",
        desc="Direct comparison identifying which outage had longer duration and affected more devices",
        parent=parent_node,
        critical=False,
    )

    # Longer duration
    n_longer = evaluator.add_leaf(
        id="longer_duration_identification",
        desc="Correctly identifies that the AT&T February 2024 outage had a longer duration (at least 12 hours) compared to Verizon's outage (several hours, resolved same day)",
        parent=node_cmp,
        critical=True,
    )
    cmp_sources_duration = _merge_sources(
        _srcs(att_info.duration) if att_info else [],
        _srcs(verizon_info.duration) if verizon_info else [],
        _srcs(verizon_info.resolution_time) if verizon_info else [],
        _collect_outage_all_sources(att_info),
        _collect_outage_all_sources(verizon_info),
    )
    await evaluator.verify(
        claim=(
            "The AT&T February 2024 outage had a longer duration (at least 12 hours) than the Verizon January 2026 outage "
            "(which lasted several hours and was resolved the same day)."
        ),
        node=n_longer,
        sources=cmp_sources_duration,
        additional_instruction="Base your judgment on the cited sources' durations; accept if AT&T ≥12h and Verizon 'several hours' same-day resolution.",
        extra_prerequisites=[verizon_nodes.get("duration_node"), att_nodes.get("duration_node")],
    )

    # More devices affected
    n_more = evaluator.add_leaf(
        id="more_devices_identification",
        desc="Correctly identifies that the AT&T February 2024 outage affected more devices (over 125 million) compared to Verizon's outage (approximately 180,000 users)",
        parent=node_cmp,
        critical=True,
    )
    cmp_sources_devices = _merge_sources(
        _srcs(att_info.affected) if att_info else [],
        _srcs(verizon_info.affected) if verizon_info else [],
        _collect_outage_all_sources(att_info),
        _collect_outage_all_sources(verizon_info),
    )
    await evaluator.verify(
        claim=(
            "The AT&T February 2024 outage affected more devices (over 125 million) than the Verizon January 2026 outage "
            "(approximately 180,000 users)."
        ),
        node=n_more,
        sources=cmp_sources_devices,
        additional_instruction="Compare magnitudes from cited sources: >125M (AT&T) vs ~180k (Verizon).",
        extra_prerequisites=[verizon_nodes.get("devices_node"), att_nodes.get("devices_node")],
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an agent's answer for the telecom outages comparative analysis task.
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
        default_model=model,
    )

    # Extraction
    extraction: OutagesExtraction = await evaluator.extract(
        prompt=prompt_extract_outages(),
        template_class=OutagesExtraction,
        extraction_name="outages_extraction",
    )

    # Build Verizon and AT&T subtrees
    verizon_nodes = await build_verizon_subtree(evaluator, root, extraction.verizon or OutageDetails())
    att_nodes = await build_att_subtree(evaluator, root, extraction.att or OutageDetails())

    # Build comparative subtree
    await build_comparative_subtree(
        evaluator=evaluator,
        parent_node=root,
        verizon_info=extraction.verizon or OutageDetails(),
        att_info=extraction.att or OutageDetails(),
        verizon_nodes=verizon_nodes,
        att_nodes=att_nodes,
    )

    # Optionally record some ground-truth-like expectations to assist debugging
    evaluator.add_ground_truth({
        "expected_verizon": EXPECTED["verizon"],
        "expected_att": EXPECTED["att"],
        "expected_comparative": EXPECTED["comparative"],
    }, gt_type="expected_facts_for_reference")

    return evaluator.get_summary()