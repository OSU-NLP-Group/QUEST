import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "q4_2025_major_tech_outages"
TASK_DESCRIPTION = (
    "Identify major technology service outages that occurred during the fourth quarter of 2025 (October 1 through December 31, 2025). "
    "You should find at least three qualifying outages, and may identify up to five. For each outage, verify that it meets ALL of the following criteria:\n\n"
    "1. The outage affected a major cloud service provider (such as AWS, Microsoft Azure, Google Cloud, or Cloudflare), a major streaming platform (such as YouTube or Netflix), or a major telecommunications service provider\n"
    "2. The outage lasted for at least 2 continuous hours\n"
    "3. The outage met at least ONE of these impact thresholds:\n"
    "   - Impacted users across at least two distinct geographic regions (North America, Europe, Asia Pacific, Latin America, or Middle East/Africa), OR\n"
    "   - Generated at least 500,000 user reports on outage monitoring platforms such as Downdetector\n"
    "4. The outage has a publicly documented root cause or official technical explanation from the service provider or credible technology news sources\n\n"
    "For each qualifying outage, provide:\n"
    "- The service name and the exact date (including year, month, and day) the outage occurred\n"
    "- The duration of the outage stated in hours\n"
    "- The geographic regions that were affected\n"
    "- Quantitative impact metrics, such as the number of user reports on Downdetector or estimated number of affected users\n"
    "- A brief description of the documented root cause or technical explanation for what caused the outage\n"
    "- At least one reference URL linking to an official incident report, service provider status page update, or credible technology news article that documents the incident"
)

CANONICAL_REGIONS = [
    "North America", "Europe", "Asia Pacific", "Latin America", "Middle East/Africa"
]

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class OutageItem(BaseModel):
    service_name: Optional[str] = None
    date: Optional[str] = None  # Keep as free text; verification checks will reason over it
    duration_hours: Optional[str] = None  # Free text, e.g., "2 hours", "2.5h", "about 3 hours"
    regions: List[str] = Field(default_factory=list)  # Canonicalized region names if possible
    impact_metric_text: Optional[str] = None  # e.g., "over 1 million reports"
    downdetector_reports: Optional[str] = None  # keep as string to allow "500,000+" etc.
    affected_users_estimate: Optional[str] = None  # keep as string for flexibility
    root_cause: Optional[str] = None  # brief description
    urls: List[str] = Field(default_factory=list)  # reference URLs


class OutageExtraction(BaseModel):
    outages: List[OutageItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outages() -> str:
    return """
    Extract up to five outages from the answer in the order they appear. For each outage entry, extract the following fields exactly as stated in the answer:

    - service_name: The affected service's name (e.g., "AWS", "Azure", "YouTube", "Cloudflare", "AT&T")
    - date: The exact outage date as provided in the answer (include year, month, and day if present; keep the original format). If not provided, return null.
    - duration_hours: The reported outage duration in hours as text (e.g., "2 hours", "2.5h", "about 3 hours"). If not provided, return null.
    - regions: A list of the geographic regions that the answer claims were affected, canonicalized to these buckets when possible:
      ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East/Africa"].
      If the text mentions similar descriptors (e.g., "US" -> "North America", "EMEA" -> ["Europe","Middle East/Africa"]), map accordingly.
      If unclear or not specified, return an empty list.
    - impact_metric_text: A textual quantitative impact statement if present (e.g., "over 1 million reports", "tens of millions affected"). If not present, return null.
    - downdetector_reports: The Downdetector report count if present (e.g., "500,000", "1,200,000+"). Extract as a string exactly as stated if present, else null.
    - affected_users_estimate: Estimated number of affected users if present (as text). Else null.
    - root_cause: The documented root cause or official technical explanation (summarized) if present; else null.
    - urls: An array of URLs cited for this outage (official status/incident pages or credible news/analysis). Extract actual URLs if present; return empty list if none.

    Return a JSON object with a single key "outages" that is an array of objects with the above fields.
    Do not add or infer content that is not explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_number(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r"\d", text))


def is_duration_in_hours(text: Optional[str]) -> bool:
    if not text:
        return False
    # Look for patterns like "2 h", "2h", "2 hr", "2 hrs", "2 hour(s)", "2.5 hours", etc.
    patterns = [
        r"\b\d+(\.\d+)?\s*(h|hr|hrs|hour|hours)\b",
        r"\babout\s+\d+(\.\d+)?\s*(h|hr|hrs|hour|hours)\b",
        r"\baround\s+\d+(\.\d+)?\s*(h|hr|hrs|hour|hours)\b",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def parse_int_like(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Extract digits, ignoring commas and plus signs
    m = re.findall(r"\d+", text.replace(",", ""))
    if not m:
        return None
    try:
        return int("".join(m))
    except Exception:
        return None


def has_complete_date_ymd(text: Optional[str]) -> bool:
    """
    Heuristic check that a date string seems to include year, month, and day for 2025.
    Accepts formats like:
      - 2025-10-03
      - Oct 3, 2025
      - October 03, 2025
      - 3 Oct 2025
    """
    if not text:
        return False
    text = text.strip()
    # Require year 2025
    if "2025" not in text:
        return False
    # Look for:
    # - YYYY-MM-DD
    iso = re.search(r"\b2025[-/]\d{1,2}[-/]\d{1,2}\b", text)
    if iso:
        return True
    # - Month name + day + , + year
    named = re.search(r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|"
                      r"Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},\s*2025\b", text, re.IGNORECASE)
    if named:
        return True
    # - Day + Month name + Year
    rev_named = re.search(r"\b\d{1,2}\s+(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|"
                          r"Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+2025\b", text, re.IGNORECASE)
    if rev_named:
        return True
    return False


def count_distinct_regions(regions: List[str]) -> int:
    norm = []
    for r in regions:
        if not r:
            continue
        r_clean = r.strip()
        # Attempt to map common synonyms
        if re.search(r"\bUS(A)?\b|\bUnited States\b|\bCanada\b|\bNorth America\b", r_clean, re.IGNORECASE):
            norm.append("North America")
        elif re.search(r"\bEurope\b|\bUK\b|\bUnited Kingdom\b|\bEU\b", r_clean, re.IGNORECASE):
            norm.append("Europe")
        elif re.search(r"\bAPAC\b|\bAsia\b|\bAsia Pacific\b|\bAustralia\b|\bIndia\b|\bJapan\b|\bKorea\b|\bSEA\b", r_clean, re.IGNORECASE):
            norm.append("Asia Pacific")
        elif re.search(r"\bLATAM\b|\bLatin America\b|\bBrazil\b|\bMexico\b|\bArgentina\b|\bChile\b", r_clean, re.IGNORECASE):
            norm.append("Latin America")
        elif re.search(r"\bMiddle East\b|\bAfrica\b|\bMEA\b", r_clean, re.IGNORECASE):
            norm.append("Middle East/Africa")
        else:
            # If provided as canonical already
            if r_clean in CANONICAL_REGIONS:
                norm.append(r_clean)
    return len(set(norm))


def outage_is_provided(ot: Optional[OutageItem]) -> bool:
    """
    Determine whether an outage entry is considered 'provided' by the answer.
    We require at least: service_name present, exact date present (ymd-looking), and at least one reference URL.
    """
    if not ot:
        return False
    return bool(ot.service_name and has_complete_date_ymd(ot.date) and ot.urls)


# --------------------------------------------------------------------------- #
# Per-outage verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_single_outage(
    evaluator: Evaluator,
    parent: VerificationNode,
    idx: int,
    outage: Optional[OutageItem],
) -> Tuple[VerificationNode, bool]:
    """
    Build and verify the tree for a single outage entry.

    Returns:
        (outage_parent_node, provided_flag)
    """
    n = idx + 1
    outage_node = evaluator.add_parallel(
        id=f"Outage_{n}",
        desc=f"Outage entry #{n} (evaluate only if provided).",
        parent=parent,
        critical=False
    )

    # Data shortcuts
    service = outage.service_name if outage else None
    date_text = outage.date if outage else None
    duration_text = outage.duration_hours if outage else None
    regions = outage.regions if outage else []
    urls = outage.urls if outage else []
    impact_text = outage.impact_metric_text if outage else None
    dd_text = outage.downdetector_reports if outage else None
    users_text = outage.affected_users_estimate if outage else None
    root_cause = outage.root_cause if outage else None

    provided_flag = outage_is_provided(outage)

    # 1) Service_Name_Provided (critical)
    evaluator.add_custom_node(
        result=bool(service and service.strip()),
        id=f"Outage_{n}_Service_Name_Provided",
        desc="Provides the service name for the outage.",
        parent=outage_node,
        critical=True
    )

    # 2) Exact_Date_Provided (critical): require year-month-day presence (heuristic)
    evaluator.add_custom_node(
        result=has_complete_date_ymd(date_text),
        id=f"Outage_{n}_Exact_Date_Provided",
        desc="Provides the exact outage date including year, month, and day.",
        parent=outage_node,
        critical=True
    )

    # 3) Date_In_Q4_2025 (critical) - pure logical check; can be LLM simple verify
    date_check_node = evaluator.add_leaf(
        id=f"Outage_{n}_Date_In_Q4_2025",
        desc="Outage date is between October 1, 2025 and December 31, 2025 (inclusive).",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The outage date '{date_text}' falls within 2025-10-01 to 2025-12-31 inclusive.",
        node=date_check_node,
        sources=None,
        additional_instruction=(
            "Judge based solely on the provided date string. Accept common date formats. "
            "If the date unambiguously refers to a day within Q4 2025, return Correct."
        )
    )

    # 4) Service_Category_Eligible (critical) - verify with sources if available
    category_node = evaluator.add_leaf(
        id=f"Outage_{n}_Service_Category_Eligible",
        desc="Affected service is a major cloud provider, major streaming platform, or major telecommunications provider (per question constraint).",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The affected service '{service}' is a major cloud service provider (e.g., AWS, Microsoft Azure, Google Cloud, Cloudflare), "
            f"a major streaming platform (e.g., YouTube or Netflix), or a major telecommunications service provider."
        ),
        node=category_node,
        sources=urls if urls else None,
        additional_instruction=(
            "Use the provided evidence to confirm the service's category. Consider well-known global-scale providers. "
            "Minor/local services should not be considered 'major'. If evidence clearly indicates major status or category, return Supported."
        )
    )

    # 5) Duration_Provided_In_Hours (critical) - existence with hours semantics
    evaluator.add_custom_node(
        result=is_duration_in_hours(duration_text),
        id=f"Outage_{n}_Duration_Provided_In_Hours",
        desc="States the outage duration in hours.",
        parent=outage_node,
        critical=True
    )

    # 6) Duration_At_Least_2_Continuous_Hours (critical) - verify with sources if possible
    duration_node = evaluator.add_leaf(
        id=f"Outage_{n}_Duration_At_Least_2_Continuous_Hours",
        desc="Outage lasted at least 2 continuous hours.",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The outage lasted at least 2 continuous hours (reported duration: '{duration_text}').",
        node=duration_node,
        sources=urls if urls else None,
        additional_instruction=(
            "Check the timeline in the evidence. Pass if the evidence supports an outage duration of 2 hours or longer, "
            "even if approximate (e.g., 'several hours') clearly implies >= 2 hours."
        )
    )

    # 7) Geographic_Regions_Provided (critical) - existence
    evaluator.add_custom_node(
        result=bool(regions and len(regions) > 0),
        id=f"Outage_{n}_Geographic_Regions_Provided",
        desc="Lists the geographic regions affected.",
        parent=outage_node,
        critical=True
    )

    # 8) Quantitative_Impact_Metrics_Provided (critical) - existence of at least one quantitative metric
    any_quant = has_number(impact_text) or has_number(dd_text) or has_number(users_text)
    evaluator.add_custom_node(
        result=bool(any_quant),
        id=f"Outage_{n}_Quantitative_Impact_Metrics_Provided",
        desc="Provides at least one quantitative impact metric (e.g., number of outage reports, estimated affected users).",
        parent=outage_node,
        critical=True
    )

    # 9) Impact_Threshold_Met (critical) - verify either 2+ distinct regions OR >= 500,000 Downdetector reports
    dd_count = parse_int_like(dd_text)
    region_count = count_distinct_regions(regions)
    impact_claim_text = (
        f"The outage impacted users across at least two distinct geographic regions (regions provided: {regions}; distinct count={region_count}) "
        f"OR it generated at least 500,000 user reports on an outage monitoring platform (reported count: {dd_text}). "
        f"Accept as Supported if either condition is clearly met."
    )
    impact_node = evaluator.add_leaf(
        id=f"Outage_{n}_Impact_Threshold_Met",
        desc="Meets at least one impact threshold: (a) impacted users across at least two distinct geographic regions OR (b) generated at least 500,000 user reports on an outage monitoring platform (e.g., Downdetector).",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=impact_claim_text,
        node=impact_node,
        sources=urls if urls else None,
        additional_instruction=(
            "Use the evidence to determine if EITHER (1) at least two distinct global regions (among North America, Europe, Asia Pacific, Latin America, Middle East/Africa) were impacted, "
            "OR (2) Downdetector (or similar) reports reached >= 500,000. "
            "If region names imply the same canonical region, count them once. "
            "If at least one condition is met per the evidence, return Supported."
        )
    )

    # 10) Root_Cause_Or_Technical_Explanation_Provided (critical) - verify documented root cause from evidence
    root_node = evaluator.add_leaf(
        id=f"Outage_{n}_Root_Cause_Or_Technical_Explanation_Provided",
        desc="Provides a publicly documented root cause or official technical explanation from the provider or credible technology news sources.",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"There is a publicly documented root cause or official technical explanation for this outage. "
            f"Example/summary provided: '{root_cause}'."
        ),
        node=root_node,
        sources=urls if urls else None,
        additional_instruction=(
            "Check the evidence for an explicit reason, root cause, or technical explanation (e.g., configuration error, network issue, DDoS, data center problem). "
            "Wording does not need to match exactly, but there must be a clear explanation from an official status/incident report or credible technology news/analysis."
        )
    )

    # 11) Reference_URL_Provided (critical) - existence
    evaluator.add_custom_node(
        result=bool(urls and len(urls) > 0),
        id=f"Outage_{n}_Reference_URL_Provided",
        desc="Includes at least one publicly accessible reference URL documenting the incident (official status/incident report or credible technology news/analysis).",
        parent=outage_node,
        critical=True
    )

    return outage_node, provided_flag


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
    Evaluate an answer for the Q4 2025 major technology outages task.
    """
    # Initialize evaluator (root node is non-critical to avoid strict critical-child constraint at root)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates all outages + global checks in parallel
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

    # Extract outages from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_outages(),
        template_class=OutageExtraction,
        extraction_name="extracted_outages"
    )

    # Normalize to at most 5 outages; keep original order
    outages: List[OutageItem] = (extraction.outages or [])[:5]

    # Build per-outage verification subtrees
    outage_nodes: List[VerificationNode] = []
    provided_flags: List[bool] = []

    for idx in range(len(outages)):
        node, provided = await verify_single_outage(evaluator, root, idx, outages[idx])
        outage_nodes.append(node)
        provided_flags.append(provided)

    # If fewer than 3 outages were provided in the answer, do not fabricate extras; the global count check will fail.
    # However, we still allow up to 5 entries overall by design (we already trimmed above).
    # For completeness, if the answer includes fewer than 5 outages, we won't add placeholder nodes.

    # Global constraints node (critical)
    global_node = evaluator.add_parallel(
        id="Global_Item_Count_And_Validity_Constraints",
        desc="Global constraints on number of outages and that every included outage is qualifying.",
        parent=root,
        critical=True
    )

    # Compute number of outages that are truly "provided"
    provided_count = sum(1 for f in provided_flags if f)

    # Leaf: Outage_Count_Between_Three_And_Five (critical)
    evaluator.add_custom_node(
        result=(3 <= provided_count <= 5),
        id="Outage_Count_Between_Three_And_Five",
        desc="The response includes at least three outages and no more than five outages.",
        parent=global_node,
        critical=True
    )

    # Determine whether all provided outages satisfy all per-outage critical requirements
    # For each provided outage, check that all its critical checks passed (i.e., parent outage node aggregated score == 1.0)
    all_provided_qualify = True
    for idx, node in enumerate(outage_nodes):
        if provided_flags[idx]:
            # Force computation without mutating child statuses beyond what already happened
            score = node.compute_score(mutate=False)
            if score < 1.0:
                all_provided_qualify = False
                break

    evaluator.add_custom_node(
        result=all_provided_qualify,
        id="All_Provided_Outages_Are_Qualifying",
        desc="Every outage entry included in the response satisfies all per-outage critical requirements (no non-qualifying outages are listed).",
        parent=global_node,
        critical=True
    )

    # Return structured evaluation summary
    return evaluator.get_summary()