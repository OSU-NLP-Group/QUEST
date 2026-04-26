import asyncio
import logging
import re
from typing import Any, List, Optional, Tuple, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_admin_pathways_districts"
TASK_DESCRIPTION = (
    "An experienced teacher in Texas is planning to transition into district-level educational administration within "
    "the next 2-3 years. To make an informed career decision, they need to identify school districts that offer strong "
    "professional growth opportunities, competitive compensation, and administrative career pathways.\n\n"
    "Identify 3 school districts in Texas that meet ALL of the following criteria:\n"
    "- At least 2 districts must be from Education Service Center (ESC) Region 4 (Houston area)\n"
    "- At least 1 district must be from Education Service Center (ESC) Region 13 (Austin area)\n\n"
    "Each district must satisfy ALL 12 of the following professional criteria:\n\n"
    "1. The district received an A or B rating in the 2024-25 Texas Education Agency (TEA) accountability system\n"
    "2. The district is classified by TEA as \"major suburban\" or \"other central city suburban\"\n"
    "3. The district has a total enrollment of at least 30,000 students\n"
    "4. The district maintains an accessible employment/careers webpage with current job opportunities\n"
    "5. The district's teacher starting salary is at least $58,000 for the 2024-25 or 2025-26 school year\n"
    "6. The district provides or supports pathways for principal or administrative certification (through partnerships, "
    "programs, or documented support)\n"
    "7. The district's current superintendent holds proper TEA certification credentials\n"
    "8. The district has posted at least one district-level administrative position opening during 2025\n"
    "9. The district offers a comprehensive benefits package that includes health insurance, dental insurance, and "
    "retirement benefits\n"
    "10. The district provides professional development programs for current educators\n"
    "11. The district operates high school campuses (indicating the presence of various administrative roles)\n"
    "12. The district demonstrates documented career advancement opportunities for educators seeking to move into "
    "administrative positions\n\n"
    "For each of the 3 districts you identify, provide:\n"
    "- The district name\n"
    "- Evidence and URL references verifying each of the 12 criteria"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictItem(BaseModel):
    """A single district and all evidence URLs provided in the answer."""
    name: Optional[str] = None

    esc_region: Optional[str] = None
    region_evidence_urls: List[str] = Field(default_factory=list)

    rating_value: Optional[str] = None
    rating_urls: List[str] = Field(default_factory=list)

    district_type_value: Optional[str] = None
    district_type_urls: List[str] = Field(default_factory=list)

    enrollment_value: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)

    employment_page_url: Optional[str] = None  # A single careers URL
    employment_urls: List[str] = Field(default_factory=list)  # Optional additional supporting URLs

    salary_value: Optional[str] = None
    salary_urls: List[str] = Field(default_factory=list)

    admin_cert_urls: List[str] = Field(default_factory=list)

    superintendent_name: Optional[str] = None
    superintendent_cert_urls: List[str] = Field(default_factory=list)

    admin_openings_urls: List[str] = Field(default_factory=list)

    benefits_urls: List[str] = Field(default_factory=list)

    pd_urls: List[str] = Field(default_factory=list)

    hs_list_urls: List[str] = Field(default_factory=list)

    career_advancement_urls: List[str] = Field(default_factory=list)


class DistrictsExtraction(BaseModel):
    """All districts mentioned in the answer."""
    districts: List[DistrictItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return (
        "Extract all Texas school districts mentioned in the answer and the explicit URL evidence the answer cites for "
        "each required criterion. For each district, return a JSON object with the following fields:\n"
        "- name: district name\n"
        "- esc_region: the Education Service Center region text as stated (e.g., 'Region 4', 'ESC 13', or similar)\n"
        "- region_evidence_urls: URLs that support the district's ESC region membership (TEA or official sources)\n"
        "- rating_value: the TEA accountability rating stated (e.g., 'A' or 'B') if provided\n"
        "- rating_urls: URLs that support the TEA accountability rating for the most recent relevant cycle\n"
        "- district_type_value: the TEA district type classification (e.g., 'Major Suburban' or 'Other central city suburban') if stated\n"
        "- district_type_urls: URLs that support the TEA district type classification\n"
        "- enrollment_value: reported total enrollment number or phrase (as text) if provided\n"
        "- enrollment_urls: URLs that support enrollment being at least 30,000 students\n"
        "- employment_page_url: the main careers/employment page URL, if provided\n"
        "- employment_urls: additional URLs (if any) showing current job opportunities\n"
        "- salary_value: the stated teacher starting salary text (include $ and year, e.g., '$60,000 for 2025-26') if provided\n"
        "- salary_urls: URLs that support the starting salary being at least $58,000 for 2024-25 or 2025-26\n"
        "- admin_cert_urls: URLs that support principal/administrative certification pathways or partnerships\n"
        "- superintendent_name: the superintendent's name, if provided\n"
        "- superintendent_cert_urls: URLs that support the superintendent holding proper TEA certification\n"
        "- admin_openings_urls: URLs that show at least one district-level administrative opening during 2025 (open or closed is fine)\n"
        "- benefits_urls: URLs that show a comprehensive benefits package including health, dental, and retirement\n"
        "- pd_urls: URLs that show professional development programs for educators\n"
        "- hs_list_urls: URLs that show the district operates high school campuses (e.g., schools list or campus directory)\n"
        "- career_advancement_urls: URLs that show documented career advancement opportunities for educators into admin roles\n\n"
        "Rules:\n"
        "1) Include only URLs explicitly present in the answer; do not invent URLs.\n"
        "2) If a field is missing in the answer, set it to null (for single value fields) or [] (for URL lists).\n"
        "3) Return all districts the answer mentions (not limited to 3)."
    )


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str] | str]) -> List[str]:
    if urls is None:
        return []
    if isinstance(urls, list):
        return [u for u in urls if isinstance(u, str) and len(u.strip()) > 0]
    if isinstance(urls, str) and len(urls.strip()) > 0:
        return [urls.strip()]
    return []


def infer_region_number(region_str: Optional[str]) -> Optional[int]:
    """Infer ESC region number from a free-form string."""
    if not region_str:
        return None
    s = region_str.lower()
    # Direct number match like 'region 4', 'esc 13', etc.
    m = re.search(r'(?:region|esc)\s*(\d{1,2})', s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Area hints
    if "houston" in s:
        return 4
    if "austin" in s:
        return 13
    # Any standalone number appearance
    m2 = re.search(r'\b(1[0-9]|[1-9])\b', s)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    return None


def select_three_districts(extracted: DistrictsExtraction) -> Tuple[Optional[DistrictItem], Optional[DistrictItem], Optional[DistrictItem]]:
    """Select 3 districts prioritizing 2 from Region 4 and 1 from Region 13."""
    all_items = list(extracted.districts or [])
    if not all_items:
        return None, None, None

    region4 = [d for d in all_items if infer_region_number(d.esc_region) == 4]
    region13 = [d for d in all_items if infer_region_number(d.esc_region) == 13]

    chosen1 = region4[0] if len(region4) >= 1 else (all_items[0] if len(all_items) >= 1 else None)
    # Avoid duplicates
    remaining = [d for d in all_items if d is not chosen1]

    chosen2 = region4[1] if len(region4) >= 2 else (remaining[0] if len(remaining) >= 1 else None)
    remaining2 = [d for d in remaining if d is not chosen2]

    chosen3 = region13[0] if len(region13) >= 1 and region13[0] not in (chosen1, chosen2) else (
        remaining2[0] if len(remaining2) >= 1 else None
    )

    return chosen1, chosen2, chosen3


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def _add_region_check(
    evaluator: Evaluator,
    parent_node,
    district: DistrictItem,
    district_id_prefix: str,
    expected_region_num: int,
) -> None:
    """Add and verify ESC region membership for the district."""
    # Region check aggregator
    region_main = evaluator.add_parallel(
        id=f"{district_id_prefix}_ESC_Region",
        desc=f"ESC region membership verification for {district.name or 'District'}",
        parent=parent_node,
        critical=True,
    )

    # URLs existence check
    evaluator.add_custom_node(
        result=len(_safe_urls(district.region_evidence_urls)) > 0,
        id=f"{district_id_prefix}_ESC_Region_URL",
        desc="URL reference to ESC region membership",
        parent=region_main,
        critical=True,
    )

    # Claim verification (will auto-depend on critical sibling existence)
    region_claim_leaf = evaluator.add_leaf(
        id=f"{district_id_prefix}_ESC_Region_Supported",
        desc=f"District belongs to ESC Region {expected_region_num}",
        parent=region_main,
        critical=True,
    )
    claim = f"This district belongs to ESC Region {expected_region_num}."
    await evaluator.verify(
        claim=claim,
        node=region_claim_leaf,
        sources=_safe_urls(district.region_evidence_urls),
        additional_instruction=(
            "Verify ESC region membership using TEA or official region/district sources. "
            "Minor naming variations are acceptable."
        ),
    )


async def _add_criterion_with_urls(
    evaluator: Evaluator,
    parent_node,
    main_id: str,
    main_desc: str,
    url_id: str,
    url_desc: str,
    claim_id: str,
    claim_desc: str,
    claim_text: str,
    urls: List[str],
    additional_instruction: str,
) -> None:
    """
    Build a critical parallel criterion node with:
      - a critical custom leaf checking that URLs exist
      - a critical claim leaf verified against the URLs
    """
    # Criterion aggregator
    criterion_node = evaluator.add_parallel(
        id=main_id,
        desc=main_desc,
        parent=parent_node,
        critical=True,
    )

    # URLs existence leaf
    evaluator.add_custom_node(
        result=len(_safe_urls(urls)) > 0,
        id=url_id,
        desc=url_desc,
        parent=criterion_node,
        critical=True,
    )

    # Claim verification leaf (auto-depends on the critical sibling above)
    claim_leaf = evaluator.add_leaf(
        id=claim_id,
        desc=claim_desc,
        parent=criterion_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim_text,
        node=claim_leaf,
        sources=_safe_urls(urls),
        additional_instruction=additional_instruction,
    )


async def verify_single_district(
    evaluator: Evaluator,
    root_parent,
    district: Optional[DistrictItem],
    district_slot_index: int,
    expected_region_num: int,
) -> None:
    """
    Construct the verification subtree for a single district slot:
      - District_1_Region_4 (slot_index=0, expected_region_num=4)
      - District_2_Region_4 (slot_index=1, expected_region_num=4)
      - District_3_Region_13 (slot_index=2, expected_region_num=13)
    """
    # Create the district node
    if district_slot_index == 0:
        node_id = "District_1_Region_4"
        node_desc = "First qualifying district from ESC Region 4"
    elif district_slot_index == 1:
        node_id = "District_2_Region_4"
        node_desc = "Second qualifying district from ESC Region 4"
    else:
        node_id = "District_3_Region_13"
        node_desc = "Qualifying district from ESC Region 13"

    district_node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=root_parent,
        critical=False,
    )

    # District existence check (name present)
    name_ok = bool(district and district.name and district.name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id=f"{node_id}_Name_Provided",
        desc="District name is provided",
        parent=district_node,
        critical=True,
    )

    # If the district is missing, we still build placeholder children; claim leaves will be skipped/fail appropriately
    d = district or DistrictItem()

    # ESC Region verification
    await _add_region_check(
        evaluator=evaluator,
        parent_node=district_node,
        district=d,
        district_id_prefix=node_id.replace("District_", f"D{district_slot_index + 1}_"),
        expected_region_num=expected_region_num,
    )

    # 1) TEA Rating: A or B in the 2025 cycle (accept most recent)
    await _add_criterion_with_urls(
        evaluator=evaluator,
        parent_node=district_node,
        main_id=f"D{district_slot_index + 1}_TEA_Rating",
        main_desc="District received A or B rating in TEA accountability system",
        url_id=f"D{district_slot_index + 1}_Rating_URL",
        url_desc="URL reference to TEA rating or district accountability report",
        claim_id=f"D{district_slot_index + 1}_TEA_Rating_Supported",
        claim_desc="TEA rating (A or B) is supported by cited sources",
        claim_text=(
            f"The district received an A or B rating in the TEA accountability system "
            f"(most recent relevant cycle, typically 2024-25 or 2025)."
        ),
        urls=_safe_urls(d.rating_urls),
        additional_instruction=(
            "Check TEA accountability or district report pages for explicit rating ('A' or 'B'). "
            "Allow the most recent available cycle if clearly stated."
        ),
    )

    # 2) TEA District Type Classification
    await _add_criterion_with_urls(
        evaluator=evaluator,
        parent_node=district_node,
        main_id=f"D{district_slot_index + 1}_District_Type",
        main_desc="District classified as 'major suburban' or 'other central city suburban'",
        url_id=f"D{district_slot_index + 1}_Type_URL",
        url_desc="URL reference to TEA district type classification",
        claim_id=f"D{district_slot_index + 1}_District_Type_Supported",
        claim_desc="TEA district type classification is supported by sources",
        claim_text=(
            "The district is classified by the TEA as 'Major Suburban' or 'Other central city suburban'."
        ),
        urls=_safe_urls(d.district_type_urls),
        additional_instruction=(
            "Use TEA classification references or official documentation. Minor wording variations acceptable."
        ),
    )

    # 3) Enrollment >= 30,000
    enrollment_txt = d.enrollment_value or ""
    await _add_criterion_with_urls(
        evaluator=evaluator,
        parent_node=district_node,
        main_id=f"D{district_slot_index + 1}_Enrollment",
        main_desc="District enrollment is at least 30,000 students",
        url_id=f"D{district_slot_index + 1}_Enrollment_URL",
        url_desc="URL reference to enrollment data",
        claim_id=f"D{district_slot_index + 1}_Enrollment_Supported",
        claim_desc="Enrollment threshold (>=30,000) is supported",
        claim_text=(
            f"The district's total enrollment is at least 30,000 students. "
            f"Reported enrollment: {enrollment_txt}."
        ),
        urls=_safe_urls(d.enrollment_urls),
        additional_instruction=(
            "Verify from TEA or district facts pages; thresholds can be inferred if the number clearly exceeds 30,000."
        ),
    )

    # 4) Employment/Careers page with current openings
    employment_sources = _safe_urls(d.employment_urls) + _safe_urls(d.employment_page_url)
    await _add_criterion_with_urls(
        evaluator=evaluator,
        parent_node=district_node,
        main_id=f"D{district_slot_index + 1}_Employment_Page",
        main_desc="District website has employment/careers page with current job opportunities",
        url_id=f"D{district_slot_index + 1}_Employment_URL",
        url_desc="URL to district employment/careers page",
        claim_id=f"D{district_slot_index + 1}_Employment_Page_Supported",
        claim_desc="Employment page (with openings) is supported",
        claim_text=(
            "The district maintains an accessible employment/careers webpage that shows current job opportunities."
        ),
        urls=_safe_urls(employment_sources),
        additional_instruction=(
            "Confirm that the employment/careers page (or portal) lists current postings or links to active job listings."
        ),
    )

    # 5) Starting salary >= $58,000 for 2024-25 or 2025-26
    salary_txt = d.salary_value or ""
    await _add_criterion_with_urls(
        evaluator=evaluator,
        parent_node=district_node,
        main_id=f"D{district_slot_index + 1}_Starting_Salary",
        main_desc="Teacher starting salary is at least $58,000",
        url_id=f"D{district_slot_index + 1}_Salary_URL",
        url_desc="URL reference to salary schedule",
        claim_id=f"D{district_slot_index + 1}_Starting_Salary_Supported",
        claim_desc="Starting salary threshold (>= $58,000) is supported",
        claim_text=(
            f"The district's teacher starting salary is at least $58,000 for the 2024-25 or 2025-26 school year. "
            f"Stated/indicative value: {salary_txt}."
        ),
        urls=_safe_urls(d.salary_urls),
        additional_instruction=(
            "Check salary schedules or HR compensation pages. Minor rounding is acceptable; ensure year context (2024-25 or 2025-26)."
        ),
    )

    # 6) Admin certification pathway/partnerships
    await _add_criterion_with_urls(
        evaluator=evaluator,
        parent_node=district_node,
        main_id=f"D{district_slot_index + 1}_Admin_Certification",
        main_desc="Principal/administrative certification pathway or partnership",
        url_id=f"D{district_slot_index + 1}_Cert_URL",
        url_desc="URL reference to certification pathway information",
        claim_id=f"D{district_slot_index + 1}_Admin_Certification_Supported",
        claim_desc="Admin certification pathway support is documented",
        claim_text=(
            "The district provides or supports pathways for principal or administrative certification (e.g., partnerships or programs)."
        ),
        urls=_safe_urls(d.admin_cert_urls),
        additional_instruction=(
            "Accept district leadership programs, university partnerships, or formal documentation of administrative certification support."
        ),
    )

    # 7) Superintendent holds proper TEA certification
    sup_name = d.superintendent_name or "the superintendent"
    await _add_criterion_with_urls(
        evaluator=evaluator,
        parent_node=district_node,
        main_id=f"D{district_slot_index + 1}_Superintendent_Cert",
        main_desc="Superintendent has proper TEA certification",
        url_id=f"D{district_slot_index + 1}_Super_URL",
        url_desc="URL reference to superintendent information",
        claim_id=f"D{district_slot_index + 1}_Superintendent_Cert_Supported",
        claim_desc="Superintendent TEA certification is supported",
        claim_text=(
            f"The district's current superintendent ({sup_name}) holds proper TEA certification credentials."
        ),
        urls=_safe_urls(d.superintendent_cert_urls),
        additional_instruction=(
            "Verify via TEA educator certification lookup, district bio, or official sources that the superintendent holds the appropriate certification."
        ),
    )

    # 8) Administrative openings in 2025
    await _add_criterion_with_urls(
        evaluator=evaluator,
        parent_node=district_node,
        main_id=f"D{district_slot_index + 1}_Admin_Openings",
        main_desc="Evidence of district-level administrative openings in 2025",
        url_id=f"D{district_slot_index + 1}_Openings_URL",
        url_desc="URL reference to administrative job postings",
        claim_id=f"D{district_slot_index + 1}_Admin_Openings_Supported",
        claim_desc="2025 administrative openings are supported",
        claim_text=(
            "The district posted at least one district-level administrative position opening during 2025 (open or closed)."
        ),
        urls=_safe_urls(d.admin_openings_urls),
        additional_instruction=(
            "Accept job postings showing a 2025 date, even if the posting is now closed."
        ),
    )

    # 9) Comprehensive benefits (health, dental, retirement)
    await _add_criterion_with_urls(
        evaluator=evaluator,
        parent_node=district_node,
        main_id=f"D{district_slot_index + 1}_Benefits",
        main_desc="Comprehensive benefits include health, dental, and retirement",
        url_id=f"D{district_slot_index + 1}_Benefits_URL",
        url_desc="URL reference to benefits information",
        claim_id=f"D{district_slot_index + 1}_Benefits_Supported",
        claim_desc="Benefits coverage is supported",
        claim_text=(
            "The district offers a comprehensive benefits package that includes health insurance, dental insurance, and retirement benefits."
        ),
        urls=_safe_urls(d.benefits_urls),
        additional_instruction=(
            "Verify benefits pages or HR documents showing the presence of health, dental, and retirement benefits."
        ),
    )

    # 10) Professional development (PD) programs
    await _add_criterion_with_urls(
        evaluator=evaluator,
        parent_node=district_node,
        main_id=f"D{district_slot_index + 1}_Professional_Dev",
        main_desc="District provides professional development programs",
        url_id=f"D{district_slot_index + 1}_PD_URL",
        url_desc="URL reference to professional development programs",
        claim_id=f"D{district_slot_index + 1}_PD_Supported",
        claim_desc="PD programs are supported",
        claim_text=(
            "The district provides professional development programs for current educators."
        ),
        urls=_safe_urls(d.pd_urls),
        additional_instruction=(
            "Accept PD department pages, training calendars, or documents describing educator PD offerings."
        ),
    )

    # 11) Operates high school campuses
    await _add_criterion_with_urls(
        evaluator=evaluator,
        parent_node=district_node,
        main_id=f"D{district_slot_index + 1}_High_Schools",
        main_desc="District has high school campuses",
        url_id=f"D{district_slot_index + 1}_HS_URL",
        url_desc="URL reference to district schools list",
        claim_id=f"D{district_slot_index + 1}_High_Schools_Supported",
        claim_desc="Presence of high school campuses is supported",
        claim_text=(
            "The district operates high school campuses."
        ),
        urls=_safe_urls(d.hs_list_urls),
        additional_instruction=(
            "Verify via district schools list or campus directory pages that clearly include High School(s)."
        ),
    )

    # 12) Career advancement opportunities into admin roles
    await _add_criterion_with_urls(
        evaluator=evaluator,
        parent_node=district_node,
        main_id=f"D{district_slot_index + 1}_Career_Advancement",
        main_desc="Documented career advancement pathways for educators into admin roles",
        url_id=f"D{district_slot_index + 1}_Career_URL",
        url_desc="URL reference to career advancement information",
        claim_id=f"D{district_slot_index + 1}_Career_Advancement_Supported",
        claim_desc="Career advancement evidence is supported",
        claim_text=(
            "The district demonstrates documented career advancement opportunities for educators seeking administrative positions."
        ),
        urls=_safe_urls(d.career_advancement_urls),
        additional_instruction=(
            "Accept leadership pathways, mentorship programs, aspiring leaders initiatives, or documented internal advancement routes."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate the agent's answer for the Texas district administrative pathways task.
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

    # Extract all district candidates with evidence
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction",
    )

    # Record ground truth constraints (for transparency in summary)
    evaluator.add_ground_truth({
        "region_requirements": {
            "min_region_4": 2,
            "min_region_13": 1
        },
        "criteria_list": [
            "TEA rating A/B",
            "TEA district type: Major Suburban or Other Central City Suburban",
            "Enrollment >= 30,000",
            "Employment page with current openings",
            "Starting salary >= $58,000 (2024-25 or 2025-26)",
            "Admin certification pathway/partnership",
            "Superintendent has proper TEA certification",
            "At least one district-level admin opening in 2025",
            "Benefits include health, dental, retirement",
            "Professional development programs",
            "Operates high school campuses",
            "Documented career advancement opportunities into admin roles",
        ]
    })

    # Select the three districts to evaluate (2 from Region 4, 1 from Region 13 if possible)
    d1, d2, d3 = select_three_districts(extracted)

    # Build verification subtrees for each selected district slot
    await verify_single_district(evaluator, root, d1, district_slot_index=0, expected_region_num=4)
    await verify_single_district(evaluator, root, d2, district_slot_index=1, expected_region_num=4)
    await verify_single_district(evaluator, root, d3, district_slot_index=2, expected_region_num=13)

    # Optional distribution check (custom node, not strictly necessary due to per-district region checks)
    # Compute region counts from extracted selection
    selected = [d for d in (d1, d2, d3) if d is not None]
    region_counts = {
        4: sum(1 for d in selected if infer_region_number(d.esc_region) == 4),
        13: sum(1 for d in selected if infer_region_number(d.esc_region) == 13),
    }
    evaluator.add_custom_info(
        info={"selected_region_counts": region_counts},
        info_type="selection_stats"
    )
    evaluator.add_custom_node(
        result=(region_counts.get(4, 0) >= 2 and region_counts.get(13, 0) >= 1),
        id="Region_Distribution_Check",
        desc="At least 2 districts from ESC Region 4 and at least 1 from ESC Region 13 in the selection",
        parent=root,
        critical=False,  # Non-critical to allow partial scoring even if distribution is not met
    )

    # Return summary
    return evaluator.get_summary()