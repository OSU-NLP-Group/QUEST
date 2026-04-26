import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "planetary_science_programs_sw_us"
TASK_DESCRIPTION = (
    "Identify four universities in the Southwestern United States that offer PhD programs in planetary science or closely related fields. "
    "Each university must be located in Arizona, California, or Texas, and the four universities together must include at least one from each "
    "of these three states. All universities must be listed on the American Astronomical Society's Division for Planetary Sciences (DPS) "
    "graduate schools directory (https://dps.aas.org/education/graduate-schools/). For each of the four universities, provide: "
    "(1) The complete name of the university, (2) The specific department or program name that offers the planetary science graduate program, "
    "(3) The city and state where the university is located, (4) A direct URL to the department's or program's official website, "
    "(5) A description of at least one specific research area or focus within planetary science that the program emphasizes, and "
    "(6) A reference URL from the DPS directory page confirming that the university is listed."
)

ALLOWED_STATE_ABBREV = {"AZ", "CA", "TX"}
STATE_NAME_TO_ABBREV = {
    "arizona": "AZ",
    "california": "CA",
    "texas": "TX",
    "az": "AZ",
    "ca": "CA",
    "tx": "TX",
}

DPS_DIRECTORY_BASE = "https://dps.aas.org/education/graduate-schools/"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    university_name: Optional[str] = None
    department_or_program_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    official_program_website_url: Optional[str] = None
    research_focus_area_description: Optional[str] = None
    dps_reference_url: Optional[str] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract all universities mentioned in the answer that correspond to planetary science (or closely related) PhD programs.
    For each university listed in the answer, extract the following fields exactly as stated:
    - university_name: The complete name of the university.
    - department_or_program_name: The specific department or program name offering the planetary science graduate program.
    - city: The city where the university is located (as presented in the answer).
    - state: The state where the university is located (as presented in the answer). Accept full state names or two-letter abbreviations.
    - official_program_website_url: A direct URL to the department’s or program’s official website. Must be an actual URL present in the answer.
    - research_focus_area_description: At least one specific planetary science research area or focus emphasized by the program (as described in the answer).
    - dps_reference_url: A URL from the AAS DPS graduate schools directory confirming the university’s listing. Must be a URL present in the answer, typically within the dps.aas.org domain.

    Rules:
    - Extract every university the answer provides, in the order it appears. Do not invent or infer fields that are not explicitly present.
    - If any field is not provided for a university, set it to null.
    - For URL fields, extract only valid URLs explicitly contained in the answer (plain URLs or markdown links). If the answer references a site without a URL, set the URL to null.
    - Do not normalize state names here; extract them as presented.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_state(abbrev_or_name: Optional[str]) -> Optional[str]:
    if not abbrev_or_name:
        return None
    s = abbrev_or_name.strip().lower()
    return STATE_NAME_TO_ABBREV.get(s)


def is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def is_dps_url(url: Optional[str]) -> bool:
    if not is_valid_url(url):
        return False
    return "dps.aas.org" in url and "/education/graduate-schools" in url


def canonical_university_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return " ".join(name.split()).strip().lower()


def compute_state_distribution(unis: List[UniversityItem]) -> Dict[str, int]:
    counts = {"AZ": 0, "CA": 0, "TX": 0}
    for u in unis:
        abbr = normalize_state(u.state)
        if abbr in counts:
            counts[abbr] += 1
    return counts


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    index: int,
) -> None:
    """
    Build and verify the subtree for a single university (University_{index+1}).
    """
    # Parent node for this university (parallel, non-critical)
    univ_node = evaluator.add_parallel(
        id=f"university_{index+1}",
        desc=f"Evaluation of the {['first','second','third','fourth'][index]} identified university.",
        parent=parent_node,
        critical=False,
    )

    # University_Complete_Name (critical presence)
    name_exists = bool(uni.university_name and uni.university_name.strip())
    univ_name_node = evaluator.add_custom_node(
        result=name_exists,
        id=f"u{index+1}_complete_name",
        desc="Provide the complete name of the university.",
        parent=univ_node,
        critical=True
    )

    # Geographic_Location_Constraint (critical membership in AZ/CA/TX)
    state_abbr = normalize_state(uni.state)
    geo_ok = state_abbr in ALLOWED_STATE_ABBREV
    geo_node = evaluator.add_custom_node(
        result=geo_ok,
        id=f"u{index+1}_geo_location_constraint",
        desc="University is located in Arizona, California, or Texas.",
        parent=univ_node,
        critical=True
    )

    # Required_Information_For_University (parallel, critical): presence checks
    req_info_node = evaluator.add_parallel(
        id=f"u{index+1}_required_info",
        desc="All required per-university output fields are provided.",
        parent=univ_node,
        critical=True
    )

    # Department_or_Program_Name (presence)
    dept_exists = bool(uni.department_or_program_name and uni.department_or_program_name.strip())
    dept_node = evaluator.add_custom_node(
        result=dept_exists,
        id=f"u{index+1}_dept_program_name",
        desc="Provide the specific department or program name offering the planetary science graduate program.",
        parent=req_info_node,
        critical=True
    )

    # City_and_State_Provided (presence)
    city_state_exists = bool(uni.city and uni.city.strip()) and bool(uni.state and uni.state.strip())
    city_state_node = evaluator.add_custom_node(
        result=city_state_exists,
        id=f"u{index+1}_city_state_provided",
        desc="Provide the city and state where the university is located.",
        parent=req_info_node,
        critical=True
    )

    # Official_Program_Website_URL (presence & validity)
    official_ok = is_valid_url(uni.official_program_website_url)
    official_url_node = evaluator.add_custom_node(
        result=official_ok,
        id=f"u{index+1}_official_program_url",
        desc="Provide a direct URL to the department’s or program’s official website.",
        parent=req_info_node,
        critical=True
    )

    # Research_Focus_Area_Described (verification against official URL)
    research_focus_leaf = evaluator.add_leaf(
        id=f"u{index+1}_research_focus_described",
        desc="Describe at least one specific planetary-science research area or focus emphasized by the program.",
        parent=req_info_node,
        critical=True
    )

    focus_text = uni.research_focus_area_description or ""
    await evaluator.verify(
        claim=f"The program emphasizes this planetary science research area: '{focus_text}'.",
        node=research_focus_leaf,
        sources=uni.official_program_website_url if official_ok else None,
        additional_instruction="Verify on the official program/department page whether the stated research area (or a close equivalent) is emphasized. Allow synonyms and closely related phrasing (e.g., 'planetary geology', 'planetary atmospheres', 'small bodies', 'exoplanets', 'planet formation').",
        extra_prerequisites=[official_url_node]  # Ensure official URL presence gates this verification
    )

    # DPS_Reference_URL_Provided (presence & validity)
    dps_ok = is_dps_url(uni.dps_reference_url)
    dps_ref_leaf = evaluator.add_custom_node(
        result=dps_ok,
        id=f"u{index+1}_dps_reference_url_provided",
        desc="Provide a reference URL from the DPS directory confirming the university’s listing.",
        parent=req_info_node,
        critical=True
    )

    # Program_Qualifications (sequential, critical): PhD exists -> DPS listing
    program_qual_node = evaluator.add_sequential(
        id=f"u{index+1}_program_qualifications",
        desc="University meets program-level eligibility requirements.",
        parent=univ_node,
        critical=True
    )

    # PhD_Program_Existence (verification against official program page)
    phd_exists_leaf = evaluator.add_leaf(
        id=f"u{index+1}_phd_program_existence",
        desc="University offers a PhD program in planetary science or a closely related field with a planetary science focus.",
        parent=program_qual_node,
        critical=True
    )

    # Build claim: use provided department/program and university name
    uni_name_for_claim = uni.university_name or "the university"
    dept_for_claim = uni.department_or_program_name or "the program/department"
    await evaluator.verify(
        claim=(
            f"{dept_for_claim} at {uni_name_for_claim} offers a doctoral (PhD) program that is focused on planetary science or a closely related field "
            f"(e.g., Earth and Planetary Sciences, Geosciences with planetary emphasis, Astronomy with planetary track)."
        ),
        node=phd_exists_leaf,
        sources=uni.official_program_website_url if official_ok else None,
        additional_instruction=(
            "Check the official program page for clear evidence of a doctoral/Ph.D. program and planetary-science focus. "
            "Allow variants like 'Ph.D.', 'doctoral', or program pages listing degrees including PhD with tracks or emphases in planetary science."
        ),
        extra_prerequisites=[official_url_node]  # Depend on official program URL presence
    )

    # DPS_Directory_Listing (verification against DPS URL)
    dps_listing_leaf = evaluator.add_leaf(
        id=f"u{index+1}_dps_directory_listing",
        desc="University is listed on the AAS DPS graduate schools directory.",
        parent=program_qual_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"{uni_name_for_claim} (or the relevant planetary program) is listed in the DPS graduate schools directory.",
        node=dps_listing_leaf,
        sources=uni.dps_reference_url if dps_ok else None,
        additional_instruction="Confirm that the DPS directory page lists this university/program. Allow minor naming variations (e.g., full vs abbreviated university names).",
        extra_prerequisites=[dps_ref_leaf]  # Depend on DPS reference URL presence
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate the agent's answer for the Southwestern US planetary science PhD programs task.
    """
    # Initialize evaluator with a parallel root (global constraints + per-university checks evaluated independently)
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

    # Extract structured universities data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Select the first four universities for detailed verification (padding if fewer)
    selected_unis: List[UniversityItem] = list(extracted.universities[:4])
    while len(selected_unis) < 4:
        selected_unis.append(UniversityItem())

    # Record ground-truth constraints info for reference
    evaluator.add_ground_truth({
        "allowed_states": sorted(list(ALLOWED_STATE_ABBREV)),
        "dps_directory_base": DPS_DIRECTORY_BASE,
        "require_exactly_four": True,
        "require_distinct_universities": True,
        "require_coverage_each_state": True
    }, gt_type="constraints")

    # ---------------- Global checks (critical) ---------------- #
    # Exactly four universities (as provided in the answer)
    exactly_four_node = evaluator.add_custom_node(
        result=(len(extracted.universities) == 4),
        id="global_exactly_four_universities",
        desc="Response identifies exactly 4 universities (no fewer, no more).",
        parent=root,
        critical=True
    )

    # Distinct universities among the selected four
    names_canonical = [canonical_university_name(u.university_name) for u in selected_unis if canonical_university_name(u.university_name)]
    distinct_node = evaluator.add_custom_node(
        result=(len(names_canonical) == len(set(names_canonical)) and len(names_canonical) == 4),
        id="global_universities_are_distinct",
        desc="All 4 identified universities are distinct (no duplicates).",
        parent=root,
        critical=True
    )

    # State distribution: at least one from each AZ, CA, TX among the selected four
    state_counts = compute_state_distribution(selected_unis)
    coverage_ok = all(state_counts.get(s, 0) >= 1 for s in ALLOWED_STATE_ABBREV)
    state_distribution_node = evaluator.add_custom_node(
        result=coverage_ok,
        id="global_state_distribution",
        desc="Across the 4 universities, there is at least one from each of Arizona, California, and Texas.",
        parent=root,
        critical=True
    )

    # Add custom info for diagnostics
    evaluator.add_custom_info(
        info={
            "selected_universities": [u.dict() for u in selected_unis],
            "state_counts": state_counts,
            "extracted_total": len(extracted.universities)
        },
        info_type="diagnostics",
        info_name="selection_summary"
    )

    # ---------------- Per-university verification ---------------- #
    for idx, uni in enumerate(selected_unis):
        await verify_university(evaluator, root, uni, idx)

    # Return final summary with verification tree and aggregated score
    return evaluator.get_summary()