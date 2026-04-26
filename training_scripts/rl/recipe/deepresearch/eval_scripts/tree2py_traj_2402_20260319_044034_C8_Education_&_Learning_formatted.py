import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ma_high_school_selection"
TASK_DESCRIPTION = """
A family is relocating to Massachusetts and searching for public high schools suitable for their child, who is a student-athlete aspiring to compete in NCAA Division I basketball and track & field while pursuing rigorous academic preparation for college.

Identify four (4) Massachusetts public high schools that meet ALL of the following requirements:

1. Located in Massachusetts
2. Member school of the Massachusetts Interscholastic Athletic Association (MIAA)
3. Offers varsity basketball programs
4. Offers varsity track and field programs (indoor and/or outdoor)
5. Has a total enrollment of at least 800 students
6. Offers at least 15 Advanced Placement (AP) courses
7. Provides students access to the Commonwealth Dual Enrollment Partnership (CDEP) or similar dual enrollment programs allowing high school students to earn college credits
8. Maintains a student-to-counselor ratio better than (lower than) 390:1, which is the Massachusetts state average
9. Requires a minimum 2.0 GPA for athletic eligibility
10. Meets or exceeds Massachusetts' 22-credit graduation requirement
11. Provides at least 1,000 annual instructional hours
12. Offers the core courses required for NCAA Division I eligibility, including four years of English, three years of mathematics (Algebra 1 or higher), and two years of science

For each school identified, provide:
- The school's full name
- Brief description of how it meets the key criteria
- At least one reference URL (school website, official profiles, or MIAA listing) that supports your answer
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class SchoolItem(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class SchoolsExtraction(BaseModel):
    schools: List[SchoolItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_schools() -> str:
    return """
    From the answer, extract up to four Massachusetts public high schools that the answer claims meet the listed criteria.
    For each school, extract:
    - name: the school's full name
    - reference_urls: ALL URLs explicitly cited in the answer for that school (school website pages, MIAA listings, district pages, school profile PDFs, program of studies, or Massachusetts DESE pages). Return only valid absolute URLs. Do not invent URLs.

    Return as:
    {
      "schools": [
        {"name": "...", "reference_urls": ["...", "..."]},
        ...
      ]
    }

    Rules:
    - Only include schools explicitly mentioned in the answer.
    - Only include URLs explicitly present in the answer text. If none are present, return an empty list for that school.
    - If more than four schools are present, include only the first four.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _filter_valid_urls(urls: List[str]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if isinstance(u, str):
            s = u.strip()
            if s.startswith("http://") or s.startswith("https://"):
                out.append(s)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _safe_school_name(school: SchoolItem, index: int) -> str:
    return school.name.strip() if (school and school.name) else f"the school identified as School #{index}"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_one_school(
    evaluator: Evaluator,
    parent_node,
    school: SchoolItem,
    idx_1_based: int,
) -> None:
    """
    Build verification sub-tree for a single school and launch criterion checks.
    All per-criterion leaves are critical, while the school node is non-critical to allow partial credit across schools.
    A critical "reference presence" node gates all other leaves via automatic precondition handling.
    """
    school_node = evaluator.add_parallel(
        id=f"School_{idx_1_based}",
        desc=f"{['First','Second','Third','Fourth'][idx_1_based-1]} identified high school meets all requirements",
        parent=parent_node,
        critical=False,
    )

    # Gate: reference URL presence (critical). If this fails, all other leaves should be skipped.
    urls = _filter_valid_urls(school.reference_urls if school else [])
    ref_exists = evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"School_{idx_1_based}_Reference",
        desc=f"Valid reference URL provided for School {idx_1_based}",
        parent=school_node,
        critical=True
    )

    # Prepare claims (each with its own leaf) – all critical
    school_name = _safe_school_name(school, idx_1_based)

    claims_and_nodes = []

    # 1) Location
    node_loc = evaluator.add_leaf(
        id=f"School_{idx_1_based}_Location",
        desc="School is located in Massachusetts",
        parent=school_node,
        critical=True,
    )
    claim_loc = f"{school_name} is located in Massachusetts."
    add_ins_loc = "Verify the page shows the school's city/town in Massachusetts or a clear statement that it is in Massachusetts."
    claims_and_nodes.append((claim_loc, urls, node_loc, add_ins_loc))

    # 2) MIAA membership
    node_miaa = evaluator.add_leaf(
        id=f"School_{idx_1_based}_MIAA_Membership",
        desc="School is a member of the Massachusetts Interscholastic Athletic Association (MIAA)",
        parent=school_node,
        critical=True,
    )
    claim_miaa = f"{school_name} is a member school of the Massachusetts Interscholastic Athletic Association (MIAA)."
    add_ins_miaa = "Prefer evidence from the MIAA member directory or the school's official athletics pages explicitly stating MIAA membership."
    claims_and_nodes.append((claim_miaa, urls, node_miaa, add_ins_miaa))

    # 3) Varsity basketball
    node_bball = evaluator.add_leaf(
        id=f"School_{idx_1_based}_Basketball",
        desc="School offers varsity basketball programs",
        parent=school_node,
        critical=True,
    )
    claim_bball = f"{school_name} offers varsity basketball."
    add_ins_bball = "Accept either boys or girls varsity basketball listings on the athletics page, team pages, schedules, or program descriptions."
    claims_and_nodes.append((claim_bball, urls, node_bball, add_ins_bball))

    # 4) Varsity track & field
    node_track = evaluator.add_leaf(
        id=f"School_{idx_1_based}_Track_Field",
        desc="School offers varsity track and field programs",
        parent=school_node,
        critical=True,
    )
    claim_track = f"{school_name} offers varsity track and field (indoor and/or outdoor)."
    add_ins_track = "Look for 'track and field', 'indoor track', or 'outdoor track' team listings or schedules on official pages."
    claims_and_nodes.append((claim_track, urls, node_track, add_ins_track))

    # 5) Enrollment >= 800
    node_enroll = evaluator.add_leaf(
        id=f"School_{idx_1_based}_Enrollment",
        desc="School has total enrollment of at least 800 students",
        parent=school_node,
        critical=True,
    )
    claim_enroll = f"{school_name} has a total enrollment of at least 800 students."
    add_ins_enroll = "Use school profiles, DESE dashboards, or official 'About' pages. Numeric statements or tables are acceptable."
    claims_and_nodes.append((claim_enroll, urls, node_enroll, add_ins_enroll))

    # 6) >= 15 AP courses
    node_ap = evaluator.add_leaf(
        id=f"School_{idx_1_based}_AP_Courses",
        desc="School offers at least 15 Advanced Placement (AP) courses",
        parent=school_node,
        critical=True,
    )
    claim_ap = f"{school_name} offers at least 15 Advanced Placement (AP) courses."
    add_ins_ap = "Check program of studies, curriculum guides, or AP pages. A list that clearly has 15 or more distinct AP courses suffices."
    claims_and_nodes.append((claim_ap, urls, node_ap, add_ins_ap))

    # 7) Dual enrollment (CDEP or similar)
    node_dual = evaluator.add_leaf(
        id=f"School_{idx_1_based}_Dual_Enrollment",
        desc="School provides access to Commonwealth Dual Enrollment Partnership (CDEP) or similar dual enrollment programs",
        parent=school_node,
        critical=True,
    )
    claim_dual = f"{school_name} provides access to dual enrollment such as the Commonwealth Dual Enrollment Partnership (CDEP) or a similar program allowing students to earn college credits."
    add_ins_dual = "Look for 'dual enrollment', 'early college', or partnerships with local colleges where HS students earn college credit."
    claims_and_nodes.append((claim_dual, urls, node_dual, add_ins_dual))

    # 8) Counselor ratio better than 390:1
    node_counsel = evaluator.add_leaf(
        id=f"School_{idx_1_based}_Counselor_Ratio",
        desc="School has a student-to-counselor ratio better than 390:1",
        parent=school_node,
        critical=True,
    )
    claim_counsel = f"{school_name} has a student-to-counselor ratio lower than 390 to 1."
    add_ins_counsel = "Accept explicit ratio on school/district profiles, program of studies, or data pages. If both student count and counselor count are given implying <390:1, that also suffices."
    claims_and_nodes.append((claim_counsel, urls, node_counsel, add_ins_counsel))

    # 9) Minimum 2.0 GPA for athletic eligibility
    node_gpa = evaluator.add_leaf(
        id=f"School_{idx_1_based}_Athletic_GPA",
        desc="School maintains minimum 2.0 GPA requirement for athletic eligibility",
        parent=school_node,
        critical=True,
    )
    claim_gpa = f"{school_name} requires a minimum 2.0 GPA for athletic eligibility."
    add_ins_gpa = "Look for athletic handbooks, eligibility rules, student-athlete guides, or MIAA/School policy pages stating the 2.0 requirement."
    claims_and_nodes.append((claim_gpa, urls, node_gpa, add_ins_gpa))

    # 10) Meets/exceeds 22-credit graduation requirement
    node_credits = evaluator.add_leaf(
        id=f"School_{idx_1_based}_Graduation_Credits",
        desc="School meets Massachusetts' 22-credit graduation requirement",
        parent=school_node,
        critical=True,
    )
    claim_credits = f"{school_name} requires at least 22 total credits to graduate (meets or exceeds a 22-credit requirement)."
    add_ins_credits = "Use graduation requirements in the program of studies or handbook; phrasing like 'minimum of 22 credits' or more should be accepted."
    claims_and_nodes.append((claim_credits, urls, node_credits, add_ins_credits))

    # 11) >= 1,000 annual instructional hours
    node_hours = evaluator.add_leaf(
        id=f"School_{idx_1_based}_Instructional_Hours",
        desc="School provides at least 1,000 annual instructional hours",
        parent=school_node,
        critical=True,
    )
    claim_hours = f"{school_name} provides at least 1,000 annual instructional hours."
    add_ins_hours = "Accept explicit hour totals in handbooks/calendars, or language clearly stating compliance at or above 1,000 hours."
    claims_and_nodes.append((claim_hours, urls, node_hours, add_ins_hours))

    # 12) NCAA DI core courses available
    node_ncaa = evaluator.add_leaf(
        id=f"School_{idx_1_based}_NCAA_Core",
        desc="School offers NCAA Division I required core courses (4 years English, 3 years math including Algebra 1+, 2 years science)",
        parent=school_node,
        critical=True,
    )
    claim_ncaa = f"{school_name} offers core courses meeting NCAA Division I eligibility: four years of English, three years of mathematics at Algebra I or higher, and two years of science."
    add_ins_ncaa = "Use program of studies/curriculum/graduation requirements pages indicating sequences that meet or exceed NCAA DI core expectations."
    claims_and_nodes.append((claim_ncaa, urls, node_ncaa, add_ins_ncaa))

    # Run all verifications in parallel (the critical reference presence node is already decided)
    await evaluator.batch_verify(claims_and_nodes)


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
    Evaluate an answer for the Massachusetts high school selection task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Schools evaluated independently with partial credit
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

    # Extract up to 4 schools and their reference URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_schools(),
        template_class=SchoolsExtraction,
        extraction_name="schools_extraction",
    )

    # Normalize to exactly 4 entries (pad with empty if fewer, truncate if more)
    schools: List[SchoolItem] = list(extracted.schools[:4])
    while len(schools) < 4:
        schools.append(SchoolItem())

    # Build and verify each school's subtree
    tasks = []
    for i, school in enumerate(schools, start=1):
        tasks.append(verify_one_school(evaluator, root, school, i))
    await asyncio.gather(*tasks)

    # Return aggregated summary
    return evaluator.get_summary()