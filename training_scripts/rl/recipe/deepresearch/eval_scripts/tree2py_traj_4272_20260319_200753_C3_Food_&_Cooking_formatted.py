import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "fda_cheese_recall_2025_2026"
TASK_DESCRIPTION = """
In early 2026, the FDA upgraded a major cheese recall to Class I status—its highest risk classification—due to potential Listeria contamination. This recall originally began in late November 2025 and involved grated Pecorino Romano cheese products sold under multiple brand names across 20 U.S. states.

Research this food safety incident and provide the following information:

1. Timeline: What was the exact date of the original voluntary recall announcement, and what was the exact date when the FDA upgraded this recall to Class I status?

2. Production Facility: In which city and state was the facility located that produced these contaminated cheese products?

3. FDA Classification: According to official FDA standards, what does a "Class I" recall classification specifically mean in terms of health risk?

4. Distribution Scope: How many U.S. states were affected by this recall? (Provide the exact number of states where the products were distributed.)

Your answer must be based on verifiable information from the FDA and reliable news sources, with each piece of information supported by reference URLs.
"""

# Expected ground truth hints (used only for evaluation context/prompts)
EXPECTED = {
    "company": "The Ambriola Company",
    "product": "grated Pecorino Romano cheese",
    "reason": "Listeria monocytogenes contamination",
    "detection": "routine testing",
    "original_recall_date": "November 25, 2025",
    "upgrade_date": "January 6, 2026",
    "facility_city": "West Caldwell",
    "facility_state": "New Jersey",
    "num_states": "20",
    "distribution_start_date": "November 3, 2025",
    "distribution_end_date": "November 20, 2025",
    "class_i_definition": "reasonable probability that use of or exposure to the product will cause serious adverse health consequences or death",
}


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class RecallExtraction(BaseModel):
    # Key facts (as stated in the answer)
    company: Optional[str] = None
    product: Optional[str] = None
    reason: Optional[str] = None
    detection: Optional[str] = None

    original_recall_date: Optional[str] = None
    class_upgrade_date: Optional[str] = None

    facility_city: Optional[str] = None
    facility_state: Optional[str] = None

    class_I_definition: Optional[str] = None

    num_states: Optional[str] = None
    distribution_start_date: Optional[str] = None
    distribution_end_date: Optional[str] = None

    # URL sources explicitly cited in the answer for each item
    sources_incident: List[str] = Field(default_factory=list)             # company/product/reason/detection (general incident)
    sources_orig_date: List[str] = Field(default_factory=list)
    sources_upgrade_date: List[str] = Field(default_factory=list)
    sources_facility: List[str] = Field(default_factory=list)
    sources_class_definition: List[str] = Field(default_factory=list)
    sources_num_states: List[str] = Field(default_factory=list)
    sources_date_range: List[str] = Field(default_factory=list)

    # All URLs cited anywhere in the answer (deduplicated if possible)
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_recall_info() -> str:
    return """
    Extract, exactly as stated in the answer, the following fields about the FDA cheese recall incident (late 2025 -> early 2026). If an item is not present in the answer, set it to null. Also extract the specific URLs that the answer cites for each requested item (only URLs explicitly present in the answer).

    Required fields (strings; keep the exact phrasing used in the answer; dates can be in any readable format, e.g., "November 25, 2025"):
    - company: The manufacturer/recaller name stated in the answer.
    - product: The recalled product description (e.g., "grated Pecorino Romano cheese").
    - reason: The recall reason (e.g., "Listeria monocytogenes contamination").
    - detection: The detection method if mentioned (e.g., "routine testing").
    - original_recall_date: The original voluntary recall announcement date.
    - class_upgrade_date: The date FDA upgraded the recall to Class I (highest risk).
    - facility_city: The city of the producing facility.
    - facility_state: The U.S. state of the producing facility.
    - class_I_definition: The FDA Class I recall definition text, as quoted or paraphrased in the answer.
    - num_states: The exact number of U.S. states affected (prefer digits if given).
    - distribution_start_date: The beginning of the distribution window.
    - distribution_end_date: The end of the distribution window.

    For each category below, also extract the list of URLs that the answer explicitly cites for that category (only valid URLs):
    - sources_incident: URLs supporting company/product/reason/detection (general incident context).
    - sources_orig_date: URLs supporting the original recall announcement date.
    - sources_upgrade_date: URLs supporting the FDA upgrade date.
    - sources_facility: URLs supporting the facility city/state.
    - sources_class_definition: URLs supporting the Class I definition (prefer FDA).
    - sources_num_states: URLs supporting the number of affected states.
    - sources_date_range: URLs supporting the distribution date range.

    Finally, extract: 
    - all_urls: every URL present in the answer text (deduplicate if possible).

    Notes:
    - Only include URLs actually present in the answer. Do not invent any.
    - If a URL is missing a protocol, prepend http://.
    - If some items are present without a supporting URL, still extract the item but the corresponding sources_* list may be empty.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _coalesce_urls(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for u in lst or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return url.lower()


def _has_fda_url(urls: List[str]) -> bool:
    return any("fda.gov" in _domain(u) for u in urls)


SOCIAL_DOMAINS = {
    "instagram.com", "facebook.com", "fb.com", "x.com", "twitter.com",
    "tiktok.com", "youtube.com", "youtu.be", "reddit.com", "linkedin.com", "threads.net"
}


def _has_third_party_news_url(urls: List[str]) -> bool:
    for u in urls:
        d = _domain(u)
        if "fda.gov" in d:
            continue
        if any(s in d for s in SOCIAL_DOMAINS):
            continue
        # treat any other domain as acceptable third-party/news site
        if "." in d:
            return True
    return False


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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

    # Ground truth hints (not used for direct matching, but recorded for transparency)
    evaluator.add_ground_truth({
        "expected_company": EXPECTED["company"],
        "expected_product": EXPECTED["product"],
        "expected_reason": EXPECTED["reason"],
        "expected_detection": EXPECTED["detection"],
        "expected_original_recall_date": EXPECTED["original_recall_date"],
        "expected_upgrade_date": EXPECTED["upgrade_date"],
        "expected_facility_city": EXPECTED["facility_city"],
        "expected_facility_state": EXPECTED["facility_state"],
        "expected_num_states": EXPECTED["num_states"],
        "expected_distribution_start_date": EXPECTED["distribution_start_date"],
        "expected_distribution_end_date": EXPECTED["distribution_end_date"],
        "expected_class_i_definition": EXPECTED["class_i_definition"],
    })

    # 1) Extract structured info from the answer
    extracted: RecallExtraction = await evaluator.extract(
        prompt=prompt_extract_recall_info(),
        template_class=RecallExtraction,
        extraction_name="recall_extraction",
    )

    # Prepared URL collections
    all_urls = _coalesce_urls(
        extracted.all_urls,
        extracted.sources_incident,
        extracted.sources_orig_date,
        extracted.sources_upgrade_date,
        extracted.sources_facility,
        extracted.sources_class_definition,
        extracted.sources_num_states,
        extracted.sources_date_range,
    )
    # Build top-level critical node
    recall_node = evaluator.add_parallel(
        id="Recall_Investigation",
        desc="Evaluate whether the answer correctly addresses the specified FDA cheese recall incident and provides the requested details with appropriate sourcing.",
        parent=root,
        critical=True,
    )

    # ---------------------- Incident Match --------------------------------- #
    incident_node = evaluator.add_parallel(
        id="Incident_Match",
        desc="Answer matches the constrained recall incident (company/product/reason).",
        parent=recall_node,
        critical=True,
    )

    # Company
    company_leaf = evaluator.add_leaf(
        id="Company",
        desc="Recall involves The Ambriola Company as the manufacturer/recaller.",
        parent=incident_node,
        critical=True,
    )
    company_claim = f"The recall involves the company '{extracted.company}' as the recalling firm/manufacturer."
    await evaluator.verify(
        claim=company_claim,
        node=company_leaf,
        sources=_coalesce_urls(extracted.sources_incident, extracted.sources_orig_date, extracted.sources_upgrade_date) or all_urls,
        additional_instruction=(
            "Check the cited pages to confirm the recalling firm/manufacturer for this recall. "
            f"Expected correct entity: {EXPECTED['company']}. "
            "Allow minor name variants (e.g., missing 'The', Inc.). The claim should be supported by the provided sources."
        ),
    )

    # Product
    product_leaf = evaluator.add_leaf(
        id="Product",
        desc="Recalled products are grated Pecorino Romano cheese.",
        parent=incident_node,
        critical=True,
    )
    product_claim = f"The recalled product(s) described in the answer are: '{extracted.product}'."
    await evaluator.verify(
        claim=product_claim,
        node=product_leaf,
        sources=extracted.sources_incident or all_urls,
        additional_instruction=(
            "Verify that the recalled products include grated Pecorino Romano cheese (or equivalent phrasing). "
            "The claim should match what the page states for the incident's recalled products."
        ),
    )

    # Reason and detection
    reason_leaf = evaluator.add_leaf(
        id="Reason_and_Detection",
        desc="Recall reason is Listeria monocytogenes contamination detected through routine testing.",
        parent=incident_node,
        critical=True,
    )
    reason_text = extracted.reason or ""
    detection_text = extracted.detection or ""
    reason_claim = (
        f"The recall reason stated in the answer is '{reason_text}', and the contamination was detected through '{detection_text}'."
    )
    await evaluator.verify(
        claim=reason_claim,
        node=reason_leaf,
        sources=_coalesce_urls(extracted.sources_incident, extracted.sources_orig_date) or all_urls,
        additional_instruction=(
            f"Confirm the recall reason is {EXPECTED['reason']} and that detection was via {EXPECTED['detection']} or equivalent (e.g., 'routine testing by the firm'). "
            "If the answer's phrasing differs materially from the page, mark as not supported."
        ),
    )

    # ------------------------- Timeline ------------------------------------ #
    timeline_node = evaluator.add_parallel(
        id="Timeline",
        desc="Answer provides the original voluntary recall announcement date and the FDA Class I upgrade date.",
        parent=recall_node,
        critical=True,
    )

    # Original recall date
    orig_date_leaf = evaluator.add_leaf(
        id="Original_Recall_Announcement_Date",
        desc="Original voluntary recall announcement date is November 25, 2025.",
        parent=timeline_node,
        critical=True,
    )
    orig_date_val = extracted.original_recall_date or ""
    orig_date_claim = f"The original voluntary recall announcement date was '{orig_date_val}'."
    await evaluator.verify(
        claim=orig_date_claim,
        node=orig_date_leaf,
        sources=extracted.sources_orig_date or all_urls,
        additional_instruction=(
            f"Confirm the first voluntary recall announcement date (by the firm/FDA posting) is {EXPECTED['original_recall_date']} (allow minor format variants). "
            "Do not confuse this with classification update or enforcement reports dates."
        ),
    )

    # FDA upgrade date
    upgrade_date_leaf = evaluator.add_leaf(
        id="FDA_Class_I_Upgrade_Date",
        desc="FDA upgraded the recall to Class I on January 6, 2026.",
        parent=timeline_node,
        critical=True,
    )
    upgrade_date_val = extracted.class_upgrade_date or ""
    upgrade_date_claim = f"The FDA upgraded the recall classification to Class I on '{upgrade_date_val}'."
    await evaluator.verify(
        claim=upgrade_date_claim,
        node=upgrade_date_leaf,
        sources=extracted.sources_upgrade_date or all_urls,
        additional_instruction=(
            f"Verify the date the FDA upgraded this recall to Class I (highest risk) is {EXPECTED['upgrade_date']} (minor format variants allowed). "
            "The supporting page should explicitly indicate this classification/date."
        ),
    )

    # ---------------- Production Facility Location ------------------------- #
    facility_node = evaluator.add_parallel(
        id="Production_Facility_Location",
        desc="Answer identifies the producing facility's city and state.",
        parent=recall_node,
        critical=True,
    )

    facility_leaf = evaluator.add_leaf(
        id="Facility_City_and_State",
        desc="Contaminated products were produced at the West Caldwell, New Jersey facility.",
        parent=facility_node,
        critical=True,
    )
    city = extracted.facility_city or ""
    state = extracted.facility_state or ""
    facility_claim = f"The contaminated cheese products were produced at the facility in {city}, {state}."
    await evaluator.verify(
        claim=facility_claim,
        node=facility_leaf,
        sources=extracted.sources_facility or all_urls,
        additional_instruction=(
            f"Confirm the facility location is {EXPECTED['facility_city']}, {EXPECTED['facility_state']} (accept 'NJ' for New Jersey)."
        ),
    )

    # -------------------- FDA Class I Definition --------------------------- #
    classdef_node = evaluator.add_parallel(
        id="FDA_Class_I_Definition",
        desc="Answer defines FDA Class I recall per FDA standards.",
        parent=recall_node,
        critical=True,
    )

    classdef_leaf = evaluator.add_leaf(
        id="Definition_Text",
        desc="Class I means a 'reasonable probability that use of or exposure to the product will cause serious adverse health consequences or death.'",
        parent=classdef_node,
        critical=True,
    )
    # Prefer a constant, authoritative definition checked against FDA sources
    classdef_claim = (
        "According to the FDA, a Class I recall means there is a 'reasonable probability that use of or exposure to the product will cause serious adverse health consequences or death.'"
    )
    await evaluator.verify(
        claim=classdef_claim,
        node=classdef_leaf,
        sources=extracted.sources_class_definition or all_urls,
        additional_instruction="Treat only FDA official pages (fda.gov) or pages that clearly quote FDA as authoritative for this definition. Prefer fda.gov.",
    )

    # ---------------------- Distribution Scope ----------------------------- #
    dist_node = evaluator.add_parallel(
        id="Distribution_Scope",
        desc="Answer provides distribution scope per constraints (states count and distribution window).",
        parent=recall_node,
        critical=True,
    )

    # Number of affected states
    states_leaf = evaluator.add_leaf(
        id="Number_of_Affected_States",
        desc="Products were distributed to exactly 20 U.S. states.",
        parent=dist_node,
        critical=True,
    )
    num_states_val = extracted.num_states or ""
    states_claim = f"The recalled products were distributed to exactly {num_states_val} U.S. states."
    await evaluator.verify(
        claim=states_claim,
        node=states_leaf,
        sources=extracted.sources_num_states or all_urls,
        additional_instruction=(
            f"Confirm that the number of affected U.S. states is exactly {EXPECTED['num_states']} (accept 'twenty')."
        ),
    )

    # Distribution date range
    range_leaf = evaluator.add_leaf(
        id="Distribution_Date_Range",
        desc="Products were distributed between November 3, 2025, and November 20, 2025.",
        parent=dist_node,
        critical=True,
    )
    start = extracted.distribution_start_date or ""
    end = extracted.distribution_end_date or ""
    range_claim = f"The products were distributed between {start} and {end}."
    await evaluator.verify(
        claim=range_claim,
        node=range_leaf,
        sources=extracted.sources_date_range or all_urls,
        additional_instruction=(
            f"Confirm the distribution window is from {EXPECTED['distribution_start_date']} through {EXPECTED['distribution_end_date']} "
            "(allow minor date format variants)."
        ),
    )

    # ---------------------- Sources and Citations -------------------------- #
    sources_node = evaluator.add_parallel(
        id="Sources_and_Citations",
        desc="Answer is supported by FDA and third-party reporting sources with URLs.",
        parent=recall_node,
        critical=True,
    )

    # FDA source present
    fda_present = _has_fda_url(all_urls)
    evaluator.add_custom_node(
        result=fda_present,
        id="FDA_Source_URL_Present",
        desc="Provides at least one FDA URL (domain fda.gov) supporting the recall facts.",
        parent=sources_node,
        critical=True,
    )

    # Third-party news present (not fda.gov, not social)
    third_party_present = _has_third_party_news_url(all_urls)
    evaluator.add_custom_node(
        result=third_party_present,
        id="Third_Party_News_URL_Present",
        desc="Provides at least one third-party reporting/news URL that is NOT social media and is not fda.gov.",
        parent=sources_node,
        critical=True,
    )

    # Each requested info item supported by URLs AND passed earlier verification
    required_sources_non_empty = all([
        len(extracted.sources_orig_date) > 0,
        len(extracted.sources_upgrade_date) > 0,
        len(extracted.sources_facility) > 0,
        len(extracted.sources_class_definition) > 0,
        len(extracted.sources_num_states) > 0,
        len(extracted.sources_date_range) > 0,
    ])
    required_nodes_passed = all([
        orig_date_leaf.status == "passed",
        upgrade_date_leaf.status == "passed",
        facility_leaf.status == "passed",
        classdef_leaf.status == "passed",
        states_leaf.status == "passed",
        range_leaf.status == "passed",
    ])
    evaluator.add_custom_node(
        result=(required_sources_non_empty and required_nodes_passed),
        id="Citations_Support_Requested_Information",
        desc="Each requested piece of information (timeline dates, facility location, Class I meaning, number of states) is supported by reference URL(s).",
        parent=sources_node,
        critical=True,
    )

    # Return summary
    return evaluator.get_summary()