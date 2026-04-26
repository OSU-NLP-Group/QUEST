import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "identify_research_facility_or"
TASK_DESCRIPTION = """
Identify the name of a university research facility in the United States that satisfies ALL of the following criteria:

1. The facility must be located at a public research university in Oregon where a technology company CEO graduated with a Bachelor of Science in Electrical Engineering between 1980-1989 (inclusive).

2. This same CEO must have co-founded a graphics chip or GPU manufacturing company in 1993.

3. The CEO must also hold a Master of Science in Electrical Engineering from Stanford University, obtained between 1990-1995 (inclusive).

4. The facility must have received a philanthropic donation of exactly $50 million from this CEO and their spouse to the university foundation in 2022.

5. The facility must be scheduled to open or become operational in 2026.

6. The facility's total area must be between 140,000 and 155,000 square feet (inclusive).

7. The total project cost must fall between $195 million and $220 million (inclusive).

8. The facility must be dedicated to semiconductor technology and/or artificial intelligence research and education.

9. The facility must be officially named after the donor CEO and their spouse.

10. The facility must be designated as a "collaborative innovation complex" or similar interdisciplinary research center.

11. The university must have had semiconductor research facilities or partnerships with semiconductor companies operational before 2022.

Provide the full official name of this facility.
"""


# Optional ground truth reference to aid downstream analysis (not used for scoring directly)
GROUND_TRUTH_REFERENCE = {
    "suspected_facility_name": "Jen-Hsun and Lori Huang Collaborative Innovation Complex",
    "suspected_university": "Oregon State University",
    "notes": "This GT reference is provided for context only; scoring relies on verification against cited sources in the answer."
}


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class FacilitySources(BaseModel):
    facility_main_urls: List[str] = Field(default_factory=list)
    donation_urls: List[str] = Field(default_factory=list)
    ceo_education_urls: List[str] = Field(default_factory=list)
    company_foundation_urls: List[str] = Field(default_factory=list)
    pre_2022_semiconductor_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class CEOEducation(BaseModel):
    ceo_name: Optional[str] = None
    bachelor_university: Optional[str] = None
    bachelor_degree: Optional[str] = None
    bachelor_field: Optional[str] = None
    bachelor_year: Optional[str] = None
    ms_university: Optional[str] = None
    ms_degree: Optional[str] = None
    ms_field: Optional[str] = None
    ms_year: Optional[str] = None
    spouse_name: Optional[str] = None


class CompanyInfo(BaseModel):
    company_name: Optional[str] = None
    company_type: Optional[str] = None
    founded_year: Optional[str] = None


class DonationInfo(BaseModel):
    amount_text: Optional[str] = None
    year: Optional[str] = None
    foundation_name: Optional[str] = None
    donors_text: Optional[str] = None  # e.g., "Jensen Huang and Lori Huang"


class FacilitySpecs(BaseModel):
    facility_name: Optional[str] = None
    university_name: Optional[str] = None
    state: Optional[str] = None
    scheduled_open_year: Optional[str] = None
    area_sqft_text: Optional[str] = None
    total_cost_text: Optional[str] = None
    focus_text: Optional[str] = None
    designation_text: Optional[str] = None
    named_after_text: Optional[str] = None
    pre_2022_semiconductor_note: Optional[str] = None


class FacilityExtraction(BaseModel):
    facility: Optional[FacilitySpecs] = None
    ceo: Optional[CEOEducation] = None
    company: Optional[CompanyInfo] = None
    donation: Optional[DonationInfo] = None
    sources: FacilitySources = Field(default_factory=FacilitySources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_info() -> str:
    return """
    Extract the structured information about the identified facility and related constraints from the answer.

    Return a JSON object with the following sections and fields (return null for any missing fields):
    - facility:
        • facility_name: The full official name of the facility as given in the answer (not an acronym).
        • university_name: The hosting institution name.
        • state: The U.S. state where the university is located.
        • scheduled_open_year: The year it is scheduled to open/become operational.
        • area_sqft_text: The total area as stated (e.g., "150,000 square feet").
        • total_cost_text: The total project cost as stated (e.g., "$200 million").
        • focus_text: Text indicating dedication to semiconductors and/or AI research/education.
        • designation_text: Text showing it's designated a "collaborative innovation complex" OR explicitly described as an interdisciplinary research center/complex.
        • named_after_text: Text showing the facility is named after the CEO and spouse (e.g., "Jen-Hsun and Lori Huang").
        • pre_2022_semiconductor_note: Any text indicating pre-2022 semiconductor facilities/partnerships.

    - ceo:
        • ceo_name: The CEO's name.
        • bachelor_university: University for the B.S./BSEE.
        • bachelor_degree: Degree name (e.g., "Bachelor of Science in Electrical Engineering").
        • bachelor_field: Field of study (e.g., "Electrical Engineering").
        • bachelor_year: Graduation year (1980–1989 inclusive).
        • ms_university: "Stanford University" if present.
        • ms_degree: Degree name (e.g., "Master of Science in Electrical Engineering").
        • ms_field: Field of study (e.g., "Electrical Engineering").
        • ms_year: Graduation year (1990–1995 inclusive).
        • spouse_name: Name of the spouse if provided.

    - company:
        • company_name: The GPU/graphics chip company the CEO co-founded.
        • company_type: e.g., "GPU" or "graphics chip".
        • founded_year: The founding year (expect "1993").

    - donation:
        • amount_text: Donation amount text (must be exactly "$50 million" if present).
        • year: Donation year (expect "2022").
        • foundation_name: The university foundation name receiving the gift.
        • donors_text: Donors text showing CEO and spouse (e.g., "Jensen Huang and Lori").

    - sources:
        • facility_main_urls: URLs explicitly cited in the answer about the facility or official university pages for the facility.
        • donation_urls: URLs explicitly cited in the answer about the $50M donation in 2022 to the university foundation.
        • ceo_education_urls: URLs explicitly cited in the answer confirming the CEO's BSEE and Stanford MSEE details.
        • company_foundation_urls: URLs explicitly cited in the answer confirming the CEO co-founded a GPU/graphics chip company in 1993.
        • pre_2022_semiconductor_urls: URLs explicitly cited in the answer about pre-2022 semiconductor facilities/partnerships at the university.
        • other_urls: Any other URLs cited in the answer.

    IMPORTANT:
    - Extract only what appears explicitly in the answer text.
    - For URL fields, include full, valid URLs exactly as shown (markdown links are acceptable if they contain URLs).
    - If a field is not present in the answer, set it to null, and an empty list for URL arrays.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_str(x: Optional[str]) -> str:
    return x or ""


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def combine_sources(src: FacilitySources) -> List[str]:
    all_urls = (
        src.facility_main_urls
        + src.donation_urls
        + src.ceo_education_urls
        + src.company_foundation_urls
        + src.pre_2022_semiconductor_urls
        + src.other_urls
    )
    return _dedup_urls(all_urls)


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def verify_facility_constraints(
    evaluator: Evaluator,
    parent_node,
    data: FacilityExtraction
) -> None:
    """
    Build and evaluate the verification tree for the facility constraints.
    """
    # Create the top-level critical node mirroring the rubric root
    root_node = evaluator.add_parallel(
        id="Identify_Research_Facility",
        desc="Return the full official name of a US university research facility that satisfies all listed constraints.",
        parent=parent_node,
        critical=True
    )

    facility = data.facility or FacilitySpecs()
    ceo = data.ceo or CEOEducation()
    company = data.company or CompanyInfo()
    donation = data.donation or DonationInfo()
    sources = data.sources or FacilitySources()

    # Prepare commonly used values
    facility_name = safe_str(facility.facility_name)
    university_name = safe_str(facility.university_name)
    state = safe_str(facility.state)
    ceo_name = safe_str(ceo.ceo_name)
    spouse_name = safe_str(ceo.spouse_name)
    bachelor_year = safe_str(ceo.bachelor_year)
    ms_year = safe_str(ceo.ms_year)
    company_name = safe_str(company.company_name)
    company_year = safe_str(company.founded_year)
    donation_amount = safe_str(donation.amount_text)
    donation_year = safe_str(donation.year)
    foundation_name = safe_str(donation.foundation_name)

    # Build source pools
    facility_sources = _dedup_urls(sources.facility_main_urls + sources.donation_urls)
    education_sources = _dedup_urls(sources.ceo_education_urls)
    company_sources = _dedup_urls(sources.company_foundation_urls)
    donation_sources = _dedup_urls(sources.donation_urls)
    pre2022_sources = _dedup_urls(sources.pre_2022_semiconductor_urls)
    all_sources = combine_sources(sources)

    # 1) Output_Full_Official_Facility_Name: existence check (critical)
    name_is_full = bool(facility_name and (
        (" " in facility_name.strip()) or
        any(t in facility_name.lower() for t in ["complex", "center", "collaborative", "innovation"])
        or (len(facility_name.strip()) >= 10)
    ))
    evaluator.add_custom_node(
        result=name_is_full,
        id="Output_Full_Official_Facility_Name",
        desc="Response provides the facility’s full official name (not merely an acronym or partial name).",
        parent=root_node,
        critical=True
    )

    # Prepare leaf nodes and claims
    leaf_nodes_and_tasks: List[tuple[str, List[str], Any, str]] = []

    # 2) University_Is_Public_Research_University
    node_public = evaluator.add_leaf(
        id="University_Is_Public_Research_University",
        desc="The hosting institution is a public research university.",
        parent=root_node,
        critical=True
    )
    claim_public = f"{university_name or 'The hosting institution'} is a public research university."
    leaf_nodes_and_tasks.append((
        claim_public,
        facility_sources if facility_sources else all_sources,
        node_public,
        "Verify from official university or authoritative pages that the institution is publicly funded and designated as a research university."
    ))

    # 3) University_Located_In_Oregon
    node_oregon = evaluator.add_leaf(
        id="University_Located_In_Oregon",
        desc="The hosting institution is located in Oregon.",
        parent=root_node,
        critical=True
    )
    claim_oregon = f"{university_name or 'The hosting institution'} is located in Oregon."
    leaf_nodes_and_tasks.append((
        claim_oregon,
        facility_sources if facility_sources else all_sources,
        node_oregon,
        "Confirm the university is in the U.S. state of Oregon."
    ))

    # 4) CEO_BSEE_At_University_1980_1989
    node_bsee = evaluator.add_leaf(
        id="CEO_BSEE_At_University_1980_1989",
        desc="A technology company CEO graduated from the hosting university with a Bachelor of Science in Electrical Engineering between 1980–1989 (inclusive).",
        parent=root_node,
        critical=True
    )
    if bachelor_year:
        claim_bsee = f"The CEO {ceo_name} earned a Bachelor of Science in Electrical Engineering from {university_name} in {bachelor_year}, which is between 1980 and 1989 inclusive."
    else:
        claim_bsee = f"The CEO {ceo_name or 'the CEO'} earned a Bachelor of Science in Electrical Engineering from {university_name or 'the hosting university'} between 1980 and 1989 inclusive."
    leaf_nodes_and_tasks.append((
        claim_bsee,
        education_sources if education_sources else all_sources,
        node_bsee,
        "Check credible biographies or university/alumni pages; allow variations like 'B.S.'/'BSEE'. The graduation year must fall within 1980–1989 inclusive."
    ))

    # 5) CEO_Cofounded_GPU_Company_1993
    node_gpu = evaluator.add_leaf(
        id="CEO_Cofounded_GPU_Company_1993",
        desc="That same CEO co-founded a graphics chip/GPU manufacturing company in 1993.",
        parent=root_node,
        critical=True
    )
    if company_name and company_year:
        claim_gpu = f"The CEO {ceo_name} co-founded the GPU/graphics chip company {company_name} in 1993."
    else:
        claim_gpu = f"The CEO {ceo_name or 'the CEO'} co-founded a GPU/graphics chip company in 1993."
    leaf_nodes_and_tasks.append((
        claim_gpu,
        company_sources if company_sources else all_sources,
        node_gpu,
        "Verify that the CEO is a co-founder and the company's founding year is 1993; allow wording variants like 'founded' vs 'co-founded'."
    ))

    # 6) CEO_Stanford_MSEE_1990_1995
    node_msee = evaluator.add_leaf(
        id="CEO_Stanford_MSEE_1990_1995",
        desc="That same CEO earned a Master of Science in Electrical Engineering from Stanford University between 1990–1995 (inclusive).",
        parent=root_node,
        critical=True
    )
    if ms_year:
        claim_msee = f"The CEO {ceo_name} earned a Master of Science in Electrical Engineering from Stanford University in {ms_year}, which is between 1990 and 1995 inclusive."
    else:
        claim_msee = f"The CEO {ceo_name or 'the CEO'} earned a Master of Science in Electrical Engineering from Stanford University between 1990 and 1995 inclusive."
    leaf_nodes_and_tasks.append((
        claim_msee,
        education_sources if education_sources else all_sources,
        node_msee,
        "Verify master's degree in Electrical Engineering at Stanford with a graduation year between 1990–1995 inclusive. Accept 'M.S.' variations."
    ))

    # 7) Donation_50M_2022_To_Foundation_By_CEO_And_Spouse
    node_donation = evaluator.add_leaf(
        id="Donation_50M_2022_To_Foundation_By_CEO_And_Spouse",
        desc="The facility received a philanthropic donation of exactly $50 million from the CEO and their spouse to the university foundation in 2022.",
        parent=root_node,
        critical=True
    )
    claim_donation = f"In 2022, {ceo_name or 'the CEO'} and spouse {spouse_name or 'the spouse'} donated exactly $50 million to the {foundation_name or (university_name + ' foundation' if university_name else 'university foundation')} for the {facility_name or 'facility'}."
    leaf_nodes_and_tasks.append((
        claim_donation,
        donation_sources if donation_sources else all_sources,
        node_donation,
        "Confirm an exact $50 million philanthropic gift in 2022 from the CEO and spouse to the university foundation connected to the facility."
    ))

    # 8) Scheduled_Open_2026
    node_open = evaluator.add_leaf(
        id="Scheduled_Open_2026",
        desc="The facility is scheduled to open or become operational in 2026.",
        parent=root_node,
        critical=True
    )
    claim_open = f"The facility {facility_name or 'the facility'} is scheduled to open or become operational in 2026."
    leaf_nodes_and_tasks.append((
        claim_open,
        facility_sources if facility_sources else all_sources,
        node_open,
        "Verify scheduling/expected operational date in 2026 on official facility or university pages; wording like 'expected to open' counts."
    ))

    # 9) Area_140k_155k_sqft
    node_area = evaluator.add_leaf(
        id="Area_140k_155k_sqft",
        desc="The facility’s total area is between 140,000 and 155,000 square feet (inclusive).",
        parent=root_node,
        critical=True
    )
    claim_area = "The facility’s total area is between 140,000 and 155,000 square feet (inclusive)."
    leaf_nodes_and_tasks.append((
        claim_area,
        facility_sources if facility_sources else all_sources,
        node_area,
        "Confirm the stated facility area falls within [140,000, 155,000] sq ft inclusive; accept reasonable rounding."
    ))

    # 10) Project_Cost_195M_220M
    node_cost = evaluator.add_leaf(
        id="Project_Cost_195M_220M",
        desc="The total project cost is between $195 million and $220 million (inclusive).",
        parent=root_node,
        critical=True
    )
    claim_cost = "The total project cost is between $195 million and $220 million (inclusive)."
    leaf_nodes_and_tasks.append((
        claim_cost,
        facility_sources if facility_sources else all_sources,
        node_cost,
        "Confirm official statements show total project cost within [$195M, $220M] inclusive; accept rounding variants."
    ))

    # 11) Dedicated_To_Semiconductor_Or_AI_Research_Education
    node_focus = evaluator.add_leaf(
        id="Dedicated_To_Semiconductor_Or_AI_Research_Education",
        desc="The facility is dedicated to semiconductor technology and/or artificial intelligence research and education.",
        parent=root_node,
        critical=True
    )
    claim_focus = "The facility is dedicated to semiconductor technology and/or artificial intelligence research and education."
    leaf_nodes_and_tasks.append((
        claim_focus,
        facility_sources if facility_sources else all_sources,
        node_focus,
        "Verify official descriptions indicate dedication to semiconductors and/or AI; mentions of HPC, chips, AI labs are acceptable evidence."
    ))

    # 12) Officially_Named_After_CEO_And_Spouse
    node_named = evaluator.add_leaf(
        id="Officially_Named_After_CEO_And_Spouse",
        desc="The facility is officially named after the donor CEO and their spouse.",
        parent=root_node,
        critical=True
    )
    claim_named = f"The official facility name includes the CEO and spouse: {facility_name or 'the official name'} is named after {ceo_name or 'the CEO'} and {spouse_name or 'the spouse'}."
    leaf_nodes_and_tasks.append((
        claim_named,
        facility_sources if facility_sources else all_sources,
        node_named,
        "Verify that both the CEO’s name and spouse’s name appear in the official facility name."
    ))

    # 13) Designated_Collaborative_Innovation_Complex_Or_Explicit_Interdisciplinary_Center
    node_designation = evaluator.add_leaf(
        id="Designated_Collaborative_Innovation_Complex_Or_Explicit_Interdisciplinary_Center",
        desc="The facility is explicitly designated as a “collaborative innovation complex” OR is explicitly described by the university/foundation as an interdisciplinary research center/complex (wording must be explicitly stated, not inferred).",
        parent=root_node,
        critical=True
    )
    claim_designation = "The facility is explicitly designated as a 'collaborative innovation complex' or explicitly described as an interdisciplinary research center/complex."
    leaf_nodes_and_tasks.append((
        claim_designation,
        facility_sources if facility_sources else all_sources,
        node_designation,
        "Look for exact phrases like 'Collaborative Innovation Complex' or explicit 'interdisciplinary research center/complex' descriptors."
    ))

    # 14) Pre_2022_Semiconductor_Facilities_Or_Partnerships
    node_pre2022 = evaluator.add_leaf(
        id="Pre_2022_Semiconductor_Facilities_Or_Partnerships",
        desc="The university had semiconductor research facilities or partnerships with semiconductor companies operational before 2022.",
        parent=root_node,
        critical=True
    )
    claim_pre2022 = f"Before 2022, {university_name or 'the university'} had semiconductor research facilities or partnerships with semiconductor companies."
    leaf_nodes_and_tasks.append((
        claim_pre2022,
        pre2022_sources if pre2022_sources else all_sources,
        node_pre2022,
        "Verify historical pages or announcements indicating pre-2022 semiconductor facilities or industry partnerships (e.g., labs, institutes, or company collaborations)."
    ))

    # Execute all verifications (parallelized where possible)
    await evaluator.batch_verify(leaf_nodes_and_tasks)


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
) -> Dict:
    """
    Evaluate an answer for the Oregon public research university facility identification task.
    """
    # Initialize evaluator with a parallel root
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_facility_info(),
        template_class=FacilityExtraction,
        extraction_name="facility_extraction"
    )

    # Optional GT info (context only)
    evaluator.add_ground_truth(GROUND_TRUTH_REFERENCE, gt_type="reference_context")

    # Build and run verification tree
    await verify_facility_constraints(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()