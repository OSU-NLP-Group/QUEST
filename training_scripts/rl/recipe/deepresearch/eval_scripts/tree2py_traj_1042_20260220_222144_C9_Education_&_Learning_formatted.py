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
TASK_ID = "phd_eng_universities"
TASK_DESCRIPTION = """I am planning to apply for PhD programs in engineering and need to identify 4 public universities in the United States that meet comprehensive criteria for research excellence, program offerings, and student support. The 4 universities must collectively represent all four of these distinct geographic regions (one university per region): (1) Southeast states (Florida, Georgia, or North Carolina), (2) Texas, (3) Midwest states (Illinois, Michigan, or Wisconsin), and (4) California.

Each university must satisfy ALL of the following requirements:

Institutional Characteristics:
- Must be a public university with R1 Carnegie Classification (Very High Research Activity)
- Must have total student enrollment of at least 30,000 students
- Must have a first-year student retention rate of at least 90%
- Must be regionally accredited

Engineering Programs:
- Must have a dedicated college or school of engineering
- Must offer graduate programs (MS and PhD) in at least 3 of these specific disciplines: Biomedical Engineering, Chemical Engineering, Civil Engineering, Electrical Engineering or Computer Engineering, or Mechanical Engineering
- The engineering college must have at least 5 distinct academic departments or schools
- Must offer online or distance graduate engineering degree programs

Research and Resources:
- Total research expenditures must be at least $500 million in the most recent fiscal year
- Must have at least 10 research centers or institutes affiliated with the engineering college or university
- University library system must hold at least 5 million volumes
- Student-to-faculty ratio must be 20:1 or better (lower number is better)

Student Support:
- Must have an honors program or honors college for undergraduates with stated minimum GPA requirements
- Must have at least 100 officially recognized student organizations or clubs
- Must offer study abroad programs available to engineering students
- Must have cooperative education (co-op) or internship programs specifically for engineering majors

For each of the 4 universities, provide:
- University name
- Specific geographic region it represents (from the 4 regions listed)
- Official university website URL
- Official engineering college website URL
"""

ALLOWED_REGIONS = ["Southeast", "Texas", "Midwest", "California"]
REGION_STATES = {
    "Southeast": ["FL", "GA", "NC"],
    "Texas": ["TX"],
    "Midwest": ["IL", "MI", "WI"],
    "California": ["CA"],
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    name: Optional[str] = None
    region: Optional[str] = None  # One of "Southeast", "Texas", "Midwest", "California"
    state: Optional[str] = None   # 2-letter USPS code, e.g., "NC", "TX"
    university_url: Optional[str] = None
    engineering_url: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to the first 4 universities explicitly listed in the answer that are proposed as meeting the criteria. For each university, return an object with:
    - name: The official university name as stated in the answer.
    - region: One of exactly these strings: "Southeast", "Texas", "Midwest", or "California". If the answer gives an equivalent description (e.g., "Southeastern states"), normalize to the canonical label. If uncertain, infer a best guess based on the state mentioned in the answer, but still return one of the four canonical labels.
    - state: The 2-letter state code for the university’s main campus location (e.g., "FL", "GA", "NC", "TX", "IL", "MI", "WI", "CA"). If the answer provides city/state, use that; otherwise infer a likely state from the university name if obvious; if not available, return null.
    - university_url: The official university website URL provided in the answer. If missing, return null.
    - engineering_url: The official engineering college or school website URL provided in the answer. If missing, return null.
    - supporting_urls: Any additional URLs cited in the answer that support the claims for this university (e.g., accreditation, R1 classification, research centers, library, enrollment, retention, study abroad, co-op programs). If none are cited, return an empty array.
    
    Rules:
    - Do not invent URLs; only include URLs actually present in the answer text. Normalize them to full URLs, add http:// if protocol is missing.
    - If more than 4 universities are listed in the answer, include only the first 4 in the returned list, in the same order as in the answer.
    - If fewer than 4 are listed, include as many as available and leave missing fields as null where information is not provided.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_region_label(region: Optional[str]) -> Optional[str]:
    if not region:
        return None
    r = region.strip().lower()
    if "southeast" in r or "south east" in r or "fl" in r or "ga" in r or "nc" in r:
        return "Southeast"
    if "texas" in r or "tx" in r:
        return "Texas"
    if "midwest" in r or "il" in r or "mi" in r or "wi" in r:
        return "Midwest"
    if "california" in r or "ca" in r:
        return "California"
    # Fallback to provided string if it exactly matches allowed
    for ar in ALLOWED_REGIONS:
        if r == ar.lower():
            return ar
    return None


def all_sources_for_uni(item: UniversityItem) -> List[str]:
    urls = []
    if item.university_url:
        urls.append(item.university_url)
    if item.engineering_url:
        urls.append(item.engineering_url)
    urls.extend([u for u in item.supporting_urls if isinstance(u, str) and len(u) > 0])
    return urls


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx: int,
    prior_regions: List[str]
) -> None:
    """
    Build verification sub-tree for one university with index idx (0-based).
    """
    uni_idx = idx + 1
    uni_node = evaluator.add_parallel(
        id=f"university_{uni_idx}",
        desc=f"{['First','Second','Third','Fourth'][idx]} university (representing one of the four required regions)",
        parent=parent_node,
        critical=False
    )

    # Hard precheck: URLs presence (gate later nodes if missing)
    urls_provided = bool(uni.university_url) and bool(uni.engineering_url)
    evaluator.add_custom_node(
        result=urls_provided,
        id=f"university_{uni_idx}_required_urls",
        desc=f"University #{uni_idx} has both official university and engineering college URLs provided",
        parent=uni_node,
        critical=True
    )

    sources = all_sources_for_uni(uni)
    uni_name = uni.name or f"University #{uni_idx}"
    region_lbl = normalize_region_label(uni.region)
    state_code = (uni.state or "").upper().strip() if uni.state else None

    # 1) Institutional classification & location
    inst_node = evaluator.add_parallel(
        id=f"institutional_classification_location_{uni_idx}",
        desc="Verify institutional classification and geographic location",
        parent=uni_node,
        critical=True
    )

    # 1.1 Public status
    leaf_public = evaluator.add_leaf(
        id=f"public_status_{uni_idx}",
        desc="University is a public institution",
        parent=inst_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} is a public university.",
        node=leaf_public,
        sources=sources,
        additional_instruction="Look for phrases like 'public research university', 'state university', or mission/about pages indicating public status."
    )

    # 1.2 R1 classification
    leaf_r1 = evaluator.add_leaf(
        id=f"r1_classification_{uni_idx}",
        desc="University has R1 Carnegie Classification (Very High Research Activity)",
        parent=inst_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} has the Carnegie Classification R1: Very High Research Activity.",
        node=leaf_r1,
        sources=sources,
        additional_instruction="Confirm explicit mention of 'R1' classification on official pages or linked official data sources."
    )

    # 1.3 Regional accreditation
    leaf_accred = evaluator.add_leaf(
        id=f"regional_accreditation_{uni_idx}",
        desc="University is regionally accredited",
        parent=inst_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} is regionally accredited.",
        node=leaf_accred,
        sources=sources,
        additional_instruction="Check for regional accreditor mentions (e.g., SACSCOC, HLC, WSCUC) on accreditation or institutional facts pages."
    )

    # 1.4 Geographic region mapping (location-state to region)
    leaf_geo = evaluator.add_leaf(
        id=f"geographic_region_{uni_idx}",
        desc="University is located in one of the four specified regions: Southeast (FL, GA, NC), Texas, Midwest (IL, MI, WI), or California",
        parent=inst_node,
        critical=True
    )
    geo_claim = f"{uni_name} is located in the state '{state_code}' which corresponds to the '{region_lbl}' region defined in the task."
    await evaluator.verify(
        claim=geo_claim,
        node=leaf_geo,
        sources=sources,
        additional_instruction="Confirm the campus location (city/state) and verify the state belongs to the assigned region set: Southeast={FL,GA,NC}; Texas={TX}; Midwest={IL,MI,WI}; California={CA}."
    )

    # 1.5 Official university website URL validity
    leaf_uni_url = evaluator.add_leaf(
        id=f"institutional_url_reference_{uni_idx}",
        desc="Valid official university website URL is provided as reference",
        parent=inst_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This URL is the official website of {uni_name}.",
        node=leaf_uni_url,
        sources=uni.university_url,
        additional_instruction="The page should clearly represent the official university site (branding, domain, 'About' info). If irrelevant or invalid, mark as not supported."
    )

    # Region uniqueness vs prior selections (additional single-step check)
    if prior_regions:
        leaf_region_unique = evaluator.add_leaf(
            id=f"geographic_region_unique_{uni_idx}",
            desc=f"University #{uni_idx} region differs from earlier selections",
            parent=inst_node,
            critical=True
        )
        prev_list = ", ".join([pr for pr in prior_regions if pr])
        uniq_claim = f"The assigned region for {uni_name} ('{region_lbl}') is different from previously chosen regions [{prev_list}]."
        await evaluator.verify(
            claim=uniq_claim,
            node=leaf_region_unique,
            sources=None,
            additional_instruction="Simple logical check only; verify that the new region label does not appear in the previous labels list."
        )

    # 2) Engineering programs & departments
    eng_node = evaluator.add_parallel(
        id=f"engineering_programs_departments_{uni_idx}",
        desc="Verify engineering college structure and program offerings",
        parent=uni_node,
        critical=True
    )

    leaf_dedicated = evaluator.add_leaf(
        id=f"dedicated_engineering_college_{uni_idx}",
        desc="Has a dedicated college or school of engineering",
        parent=eng_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} has a dedicated college or school of engineering.",
        node=leaf_dedicated,
        sources=sources,
        additional_instruction="Check for 'College of Engineering' or 'School of Engineering' official unit."
    )

    leaf_dept_count = evaluator.add_leaf(
        id=f"engineering_departments_count_{uni_idx}",
        desc="Engineering college has at least 5 distinct academic departments or schools",
        parent=eng_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The engineering college at {uni_name} has at least 5 distinct academic departments.",
        node=leaf_dept_count,
        sources=sources,
        additional_instruction="Look for department listings; counts like 'more than five departments' suffice."
    )

    leaf_grad_disciplines = evaluator.add_leaf(
        id=f"graduate_programs_disciplines_{uni_idx}",
        desc="Offers graduate programs (MS and PhD) in at least 3 of the specified disciplines: Biomedical Engineering, Chemical Engineering, Civil Engineering, Electrical/Computer Engineering, or Mechanical Engineering",
        parent=eng_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name}'s engineering college offers graduate (MS and PhD) programs in at least three of: Biomedical, Chemical, Civil, Electrical/Computer, Mechanical Engineering.",
        node=leaf_grad_disciplines,
        sources=sources,
        additional_instruction="Check graduate program pages for the listed disciplines and degree levels. Minor variations or naming (e.g., 'Bioengineering') can count."
    )

    leaf_online = evaluator.add_leaf(
        id=f"online_graduate_programs_{uni_idx}",
        desc="Offers online or distance graduate engineering degree programs",
        parent=eng_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name}'s engineering programs include online or distance graduate degree options.",
        node=leaf_online,
        sources=sources,
        additional_instruction="Confirm via engineering college pages or graduate program descriptions indicating online/distance options."
    )

    leaf_eng_url = evaluator.add_leaf(
        id=f"engineering_url_reference_{uni_idx}",
        desc="Valid official engineering college website URL is provided as reference",
        parent=eng_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This URL is the official engineering college website of {uni_name}.",
        node=leaf_eng_url,
        sources=uni.engineering_url,
        additional_instruction="Ensure the site represents the official engineering college/school (branding, domain, unit pages)."
    )

    # 3) Research & academic resources
    res_node = evaluator.add_parallel(
        id=f"research_academic_resources_{uni_idx}",
        desc="Verify research expenditures and academic resources",
        parent=uni_node,
        critical=True
    )

    leaf_research_exp = evaluator.add_leaf(
        id=f"research_expenditures_{uni_idx}",
        desc="Total research expenditures are at least $500 million in the most recent fiscal year",
        parent=res_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name}'s total research expenditures are at least $500 million in the most recent fiscal year.",
        node=leaf_research_exp,
        sources=sources,
        additional_instruction="Check official research reports or institutional facts pages for annual research expenditure amounts."
    )

    leaf_centers = evaluator.add_leaf(
        id=f"research_centers_count_{uni_idx}",
        desc="Has at least 10 research centers or institutes affiliated with engineering or university",
        parent=res_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} has at least 10 research centers or institutes affiliated with the engineering college or the university.",
        node=leaf_centers,
        sources=sources,
        additional_instruction="Check research centers/institutes listings. Phrases like 'over 10 centers' suffice."
    )

    leaf_library = evaluator.add_leaf(
        id=f"library_volumes_{uni_idx}",
        desc="University library system holds at least 5 million volumes",
        parent=res_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name}'s library system holds at least 5 million volumes.",
        node=leaf_library,
        sources=sources,
        additional_instruction="Look for library statistics, facts pages, or annual reports indicating total volumes/holdings."
    )

    leaf_ratio = evaluator.add_leaf(
        id=f"faculty_ratio_{uni_idx}",
        desc="Student-to-faculty ratio is 20:1 or better (lower)",
        parent=res_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The student-to-faculty ratio at {uni_name} is 20:1 or better (i.e., 20 or lower).",
        node=leaf_ratio,
        sources=sources,
        additional_instruction="Check institutional stats, Common Data Set summaries, or facts pages reporting student-faculty ratio."
    )

    # 4) Student support & campus experience
    support_node = evaluator.add_parallel(
        id=f"student_support_campus_{uni_idx}",
        desc="Verify enrollment, retention, and student support programs",
        parent=uni_node,
        critical=True
    )

    leaf_enrollment = evaluator.add_leaf(
        id=f"enrollment_size_{uni_idx}",
        desc="Total student enrollment is at least 30,000 students",
        parent=support_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total student enrollment at {uni_name} is at least 30,000.",
        node=leaf_enrollment,
        sources=sources,
        additional_instruction="Check institutional facts or enrollment summary pages; 'more than 30,000' suffices."
    )

    leaf_retention = evaluator.add_leaf(
        id=f"retention_rate_{uni_idx}",
        desc="First-year student retention rate is at least 90%",
        parent=support_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The first-year student retention rate at {uni_name} is at least 90%.",
        node=leaf_retention,
        sources=sources,
        additional_instruction="Look for retention metrics on institutional research or facts pages."
    )

    leaf_honors = evaluator.add_leaf(
        id=f"honors_program_{uni_idx}",
        desc="Has an honors program or honors college with stated minimum GPA requirements",
        parent=support_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} has an honors program or honors college with stated minimum GPA requirements.",
        node=leaf_honors,
        sources=sources,
        additional_instruction="Check for Honors College/program pages indicating minimum GPA or eligibility criteria."
    )

    leaf_orgs = evaluator.add_leaf(
        id=f"student_organizations_{uni_idx}",
        desc="Has at least 100 officially recognized student organizations or clubs",
        parent=support_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} has at least 100 officially recognized student organizations or clubs.",
        node=leaf_orgs,
        sources=sources,
        additional_instruction="Check student affairs or involvement pages listing total number of organizations."
    )

    leaf_abroad = evaluator.add_leaf(
        id=f"study_abroad_{uni_idx}",
        desc="Offers study abroad programs available to engineering students",
        parent=support_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} offers study abroad programs available to engineering students.",
        node=leaf_abroad,
        sources=sources,
        additional_instruction="Check engineering study abroad pages or university study abroad pages indicating availability for engineering majors."
    )

    leaf_coop = evaluator.add_leaf(
        id=f"coop_internship_{uni_idx}",
        desc="Has cooperative education or internship programs for engineering majors",
        parent=support_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} has cooperative education (co-op) or internship programs for engineering majors.",
        node=leaf_coop,
        sources=sources,
        additional_instruction="Check engineering career programs pages for co-op or internship pathways specifically for engineering."
    )


async def verify_region_coverage(evaluator: Evaluator, parent_node, regions: List[Optional[str]]) -> None:
    """
    Add a critical verification node ensuring the four universities collectively cover all four required regions with no duplicates.
    """
    coverage_node = evaluator.add_parallel(
        id="region_coverage",
        desc="Collective region coverage across 4 universities",
        parent=parent_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="region_coverage_unique_complete",
        desc="Four universities cover exactly Southeast, Texas, Midwest, and California (no duplicates)",
        parent=coverage_node,
        critical=True
    )

    assigned = [normalize_region_label(r) or "None" for r in regions]
    claim = f"The assigned regions for the four universities are {assigned}, which include each of the four: {ALLOWED_REGIONS}, with no duplicates."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=None,
        additional_instruction="Perform a pure logical check: the set of assigned labels must equal {'Southeast','Texas','Midwest','California'} exactly and each appears once."
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
) -> Dict:
    """
    Evaluate an answer for the PhD engineering universities selection task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Universities evaluated independently
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

    # IMPORTANT: Set root to non-critical to allow partial credit across universities
    # (JSON root was 'critical': true but that conflicts with child non-critical nodes)
    root.critical = False

    # Extract proposed universities
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_selection"
    )

    # Prepare exactly 4 items (pad if fewer)
    items = extracted.universities[:4]
    while len(items) < 4:
        items.append(UniversityItem())

    # Verify each university block
    prior_regions: List[str] = []
    assigned_regions: List[Optional[str]] = []

    for idx, uni in enumerate(items):
        await verify_university(
            evaluator=evaluator,
            parent_node=root,
            uni=uni,
            idx=idx,
            prior_regions=prior_regions
        )
        norm_r = normalize_region_label(uni.region)
        assigned_regions.append(norm_r)
        if norm_r:
            prior_regions.append(norm_r)

    # Collective region coverage check (critical at root)
    await verify_region_coverage(evaluator, root, assigned_regions)

    return evaluator.get_summary()