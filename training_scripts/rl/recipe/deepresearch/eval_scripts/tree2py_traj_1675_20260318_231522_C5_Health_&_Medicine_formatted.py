import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "ohio_medium_hospitals"
TASK_DESCRIPTION = """
I am conducting research on infection prevention programs and hospital quality metrics in Ohio. Please identify three medium-sized hospitals in Ohio that meet all of the following criteria:

1. The hospital must be located in the state of Ohio.
2. The hospital must hold current accreditation from The Joint Commission.
3. The hospital must be classified as a medium-sized facility, with a total staffed bed capacity between 100 and 499 beds.
4. The hospital must publicly report healthcare-associated infection (HAI) data through the CMS Care Compare system.
5. The hospital must have achieved an overall CMS Hospital Star Rating of 3 stars or higher.

For each hospital, please provide:
- The hospital's full official name
- The city where it is located
- The current staffed bed count
- The CMS Hospital Star Rating
- Reference URLs for verification of each criterion
"""


# ----------------------------- Data Models ----------------------------- #
class HospitalItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    staffed_beds: Optional[str] = None  # keep string to allow ranges or approximations
    cms_star_rating: Optional[str] = None

    # Criterion-specific reference URLs (extract only URLs explicitly present in the answer)
    location_source_urls: List[str] = Field(default_factory=list)
    accreditation_source_urls: List[str] = Field(default_factory=list)
    bed_count_source_urls: List[str] = Field(default_factory=list)
    hai_reporting_source_urls: List[str] = Field(default_factory=list)
    star_rating_source_urls: List[str] = Field(default_factory=list)


class HospitalsExtraction(BaseModel):
    hospitals: List[HospitalItem] = Field(default_factory=list)


# --------------------------- Extraction Prompt ------------------------- #
def prompt_extract_hospitals() -> str:
    return """
    Extract up to three hospitals listed in the answer that are claimed to meet ALL of the following criteria:
    – Located in the state of Ohio (OH)
    – Currently accredited by The Joint Commission (TJC)
    – Medium-sized: total staffed bed capacity between 100 and 499
    – Publicly reports HAI data through CMS Care Compare (Medicare)
    – Overall CMS Hospital Star Rating of 3 or higher

    For each hospital you find in the answer (in the same order they appear), extract these fields:
    1) name: Full official hospital name as written in the answer
    2) city: City as written in the answer (do not infer)
    3) state: Two-letter state (e.g., OH) or the state name as written, if present
    4) staffed_beds: The current staffed bed count as written (string; keep ranges/approx if used)
    5) cms_star_rating: The overall CMS Hospital Star Rating as written (string; e.g., "3", "3 of 5", "3 stars")

    Also extract criterion-specific reference URLs EXACTLY as shown in the answer text:
    6) location_source_urls: URLs used to support the hospital's location (address/city/state)
    7) accreditation_source_urls: URLs used to support current Joint Commission accreditation (prefer jointcommission.org Quality Check pages)
    8) bed_count_source_urls: URLs used to support the staffed bed count or bed size information
    9) hai_reporting_source_urls: URLs used to support that the hospital publicly reports HAI measures on CMS Care Compare (prefer medicare.gov/care-compare hospital profile/measure pages)
    10) star_rating_source_urls: URLs used to support the overall CMS Hospital Star Rating

    IMPORTANT RULES:
    - Only extract hospitals and URLs that are explicitly present in the provided answer text.
    - Do NOT invent or infer any URL. Omit anything not present.
    - For URLs embedded in markdown links, extract the actual URL targets.
    - If a field is missing for a hospital, set it to null (for strings) or [] for URL lists.
    - Return at most three hospitals, in order of appearance in the answer.
    """


# ----------------------------- Helpers -------------------------------- #
def non_empty_text(s: Optional[str]) -> bool:
    if s is None:
        return False
    val = s.strip().lower()
    if val in {"", "n/a", "na", "none", "unknown", "not available"}:
        return False
    return True


def unique_preserve_order(url_lists: List[List[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            u2 = (u or "").strip()
            if not u2:
                continue
            if u2 not in seen:
                seen.add(u2)
                result.append(u2)
    return result


def pick_sources(primary: List[str], fallback_groups: List[List[str]]) -> List[str]:
    if primary and len([u for u in primary if u and u.strip()]) > 0:
        return [u for u in primary if u and u.strip()]
    return unique_preserve_order(fallback_groups)


# --------------------------- Verification Logic ------------------------ #
async def verify_one_hospital(
    evaluator: Evaluator,
    parent_node,
    hospital: HospitalItem,
    idx: int,
) -> None:
    hosp_id = idx + 1
    hosp_node = evaluator.add_parallel(
        id=f"hospital_{hosp_id}",
        desc=f"Hospital #{hosp_id}: qualifying hospital meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Build a union of all provided URLs to use as fallback when a criterion-specific list is empty.
    all_urls = unique_preserve_order([
        hospital.location_source_urls,
        hospital.accreditation_source_urls,
        hospital.bed_count_source_urls,
        hospital.hai_reporting_source_urls,
        hospital.star_rating_source_urls,
    ])

    # 1) Eligibility (critical) with 5 criterion leaves
    elig_node = evaluator.add_parallel(
        id=f"h{hosp_id}_eligibility",
        desc="Hospital meets all eligibility criteria",
        parent=hosp_node,
        critical=True
    )

    # 1.a) Location in Ohio
    n_loc = evaluator.add_leaf(
        id=f"h{hosp_id}_location",
        desc="Hospital is located in Ohio",
        parent=elig_node,
        critical=True
    )
    city_part = f" in {hospital.city}" if non_empty_text(hospital.city) else ""
    location_claim = f"The hospital named '{hospital.name}' is located in the state of Ohio{city_part}."
    await evaluator.verify(
        claim=location_claim,
        node=n_loc,
        sources=pick_sources(hospital.location_source_urls, [all_urls]),
        additional_instruction="Accept if the page explicitly shows the state as Ohio (OH) for this hospital. City matching is helpful but not strictly required as long as the page clearly indicates the hospital is in Ohio."
    )

    # 1.b) Joint Commission accreditation (current)
    n_acc = evaluator.add_leaf(
        id=f"h{hosp_id}_accreditation",
        desc="Hospital holds current Joint Commission accreditation",
        parent=elig_node,
        critical=True
    )
    acc_claim = f"The hospital named '{hospital.name}' is currently accredited by The Joint Commission."
    await evaluator.verify(
        claim=acc_claim,
        node=n_acc,
        sources=pick_sources(hospital.accreditation_source_urls, [all_urls]),
        additional_instruction="Prefer evidence from jointcommission.org Quality Check (QualityReport) pages. Consider the accreditation 'current' if the page indicates 'Accredited' or an active accreditation status without an expiration in the past."
    )

    # 1.c) Medium size 100–499 staffed beds
    n_size = evaluator.add_leaf(
        id=f"h{hosp_id}_size",
        desc="Hospital is classified as medium-sized with 100-499 staffed beds",
        parent=elig_node,
        critical=True
    )
    size_text = f" (around {hospital.staffed_beds} beds)" if non_empty_text(hospital.staffed_beds) else ""
    size_claim = f"The hospital named '{hospital.name}' has a staffed (or licensed if staffed is not available) bed capacity between 100 and 499{size_text}."
    await evaluator.verify(
        claim=size_claim,
        node=n_size,
        sources=pick_sources(hospital.bed_count_source_urls, [all_urls]),
        additional_instruction="Verify that the source indicates a bed count within 100–499. If only 'licensed beds' or a bed size category implying this range is provided, accept it. If multiple values are shown, prefer the most recent or official-looking figure."
    )

    # 1.d) HAI reporting via CMS Care Compare
    n_hai = evaluator.add_leaf(
        id=f"h{hosp_id}_hai_reporting",
        desc="Hospital publicly reports HAI data through CMS Care Compare",
        parent=elig_node,
        critical=True
    )
    hai_claim = f"The hospital named '{hospital.name}' publicly reports HAI measures on the CMS Care Compare (Medicare) website (e.g., CLABSI, CAUTI, SSI, MRSA, CDI)."
    await evaluator.verify(
        claim=hai_claim,
        node=n_hai,
        sources=pick_sources(hospital.hai_reporting_source_urls, [all_urls]),
        additional_instruction="Prefer evidence from medicare.gov/care-compare hospital pages. Accept if the hospital profile presents HAI measures. If the URL is not on medicare.gov, it must still clearly show CMS Care Compare HAI data for this hospital."
    )

    # 1.e) CMS star rating >= 3
    n_star = evaluator.add_leaf(
        id=f"h{hosp_id}_star_rating",
        desc="Hospital has achieved a CMS Hospital Star Rating of 3 or higher",
        parent=elig_node,
        critical=True
    )
    rating_num_txt = f" with an overall rating noted as '{hospital.cms_star_rating}'" if non_empty_text(hospital.cms_star_rating) else ""
    star_claim = f"The overall CMS Hospital Star Rating for the hospital named '{hospital.name}' is 3 stars or higher{rating_num_txt}."
    await evaluator.verify(
        claim=star_claim,
        node=n_star,
        sources=pick_sources(hospital.star_rating_source_urls, [all_urls]),
        additional_instruction="Check the 'Overall rating' on CMS/Medicare hospital pages. Do not confuse with HCAHPS or other sub-ratings. Accept if the overall rating is 3, 4, or 5."
    )

    # 2) Output info (non-critical) – presence checks
    out_node = evaluator.add_parallel(
        id=f"h{hosp_id}_output_info",
        desc="Answer provides all requested information about the hospital",
        parent=hosp_node,
        critical=False
    )

    # 2.a) Name provided
    evaluator.add_custom_node(
        result=non_empty_text(hospital.name),
        id=f"h{hosp_id}_name_provided",
        desc="Answer provides the hospital's full official name",
        parent=out_node,
        critical=False
    )

    # 2.b) City provided
    evaluator.add_custom_node(
        result=non_empty_text(hospital.city),
        id=f"h{hosp_id}_city_provided",
        desc="Answer provides the city where the hospital is located",
        parent=out_node,
        critical=False
    )

    # 2.c) Bed count provided
    evaluator.add_custom_node(
        result=non_empty_text(hospital.staffed_beds),
        id=f"h{hosp_id}_bed_count_provided",
        desc="Answer provides the current staffed bed count",
        parent=out_node,
        critical=False
    )

    # 2.d) Star rating provided
    evaluator.add_custom_node(
        result=non_empty_text(hospital.cms_star_rating),
        id=f"h{hosp_id}_rating_provided",
        desc="Answer provides the CMS Hospital Star Rating",
        parent=out_node,
        critical=False
    )

    # 2.e) Reference URLs provided for each criterion (all five lists non-empty)
    all_criterion_urls_present = all([
        len(hospital.location_source_urls) > 0,
        len(hospital.accreditation_source_urls) > 0,
        len(hospital.bed_count_source_urls) > 0,
        len(hospital.hai_reporting_source_urls) > 0,
        len(hospital.star_rating_source_urls) > 0,
    ])
    evaluator.add_custom_node(
        result=all_criterion_urls_present,
        id=f"h{hosp_id}_urls_provided",
        desc="Answer provides reference URLs for verification",
        parent=out_node,
        critical=False
    )


# ----------------------------- Main Entry ------------------------------ #
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

    # Extract structured hospital info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hospitals(),
        template_class=HospitalsExtraction,
        extraction_name="hospitals_extraction",
    )

    # Record the task requirements as "ground truth context"
    evaluator.add_ground_truth({
        "jurisdiction": "Ohio (OH)",
        "required_accreditation": "The Joint Commission (current)",
        "size_range_staffed_beds": "100–499",
        "hai_reporting_platform": "CMS Care Compare (Medicare)",
        "min_cms_star_rating": "3",
    })

    # Choose up to 3 hospitals; pad with empty ones if fewer provided
    hospitals: List[HospitalItem] = list(extracted.hospitals[:3])
    while len(hospitals) < 3:
        hospitals.append(HospitalItem())

    # Build verification tree per hospital
    for i in range(3):
        await verify_one_hospital(evaluator, root, hospitals[i], i)

    return evaluator.get_summary()