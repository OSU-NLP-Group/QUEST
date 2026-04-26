import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ahsaa_6a_region4_huntsville_2026_2028"
TASK_DESCRIPTION = """
A family is relocating to northern Alabama and wants to identify public high schools that compete at the highest classification level in the state. They are specifically interested in schools located in or near the Huntsville area that have moderate enrollment sizes and compete in the same football region for scheduling purposes.

Identify 3 AHSAA (Alabama High School Athletic Association) Class 6A public high schools that meet ALL of the following criteria:

1. Classification: The school must be classified as AHSAA Class 6A for the 2026-2028 reclassification period (Class 6A consists of the 32 largest public high schools in Alabama).

2. Location: The school must be located in one of the following northern Alabama counties: Madison County, Limestone County, Morgan County, or Lauderdale County.

3. Enrollment Size: The school must have an Average Daily Enrollment between 1,200 and 1,600 students for the 2026-2028 classification period.

4. Football Region: The school must be assigned to Region 4 for Class 6A football competition.

For each identified school, provide:
- The school's official name
- The county where it is located
- The school's Average Daily Enrollment number for 2026-2028
- Its Region 4 assignment confirmation
- A reference URL from official AHSAA classification documents
"""

ALLOWED_COUNTIES = {"Madison", "Limestone", "Morgan", "Lauderdale"}
RECLASS_SPAN = "2026–2028"
RECLASS_SPAN_ALT = "2026-2028"
REQUIRED_SCHOOLS = 3


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SchoolItem(BaseModel):
    name: Optional[str] = None
    county: Optional[str] = None  # The county name only (e.g., "Madison", not "Madison County")
    ade_2026_2028: Optional[str] = None  # Keep as string to be robust to formats like "1,452"
    classification: Optional[str] = None  # e.g., "Class 6A" or "6A"
    football_region: Optional[str] = None  # e.g., "Region 4"
    ahsaa_urls: List[str] = Field(default_factory=list)  # Official AHSAA classification/reclassification docs (as cited)
    other_urls: List[str] = Field(default_factory=list)  # Any additional URLs cited for this school


class SchoolsExtraction(BaseModel):
    schools: List[SchoolItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_schools() -> str:
    return f"""
Extract all schools mentioned in the answer that the author claims meet the specified AHSAA criteria. For each school mentioned, extract the following fields:

- name: The school's official name exactly as written in the answer.
- county: The Alabama county name associated with the school as provided in the answer (just the county name word, e.g., "Madison", not "Madison County"). If the answer gives "Madison County", extract "Madison".
- ade_2026_2028: The Average Daily Enrollment (ADE) or Average Daily Membership (ADM) number specifically for the {RECLASS_SPAN_ALT} classification period, exactly as shown in the answer (keep any commas or formatting).
- classification: The AHSAA classification label provided in the answer for this period (e.g., "Class 6A" or "6A").
- football_region: The region label for football provided in the answer (e.g., "Region 4").
- ahsaa_urls: All URLs cited in the answer that appear to be official AHSAA classification/reclassification documents (e.g., domains like ahsaa.com, cdn.ahsaa.com, ahsaa.arbitersports.com) relevant to {RECLASS_SPAN_ALT}. Include only these AHSAA official links here.
- other_urls: Any other URLs cited in the answer that are associated with this school (school website, district page, Wikipedia, news, etc.) that might support location or other facts.

Rules:
- Extract only URLs explicitly present in the answer; do not invent any.
- Preserve the original order of schools from the answer.
- If any field is not present in the answer for a given school, set it to null (for strings) or an empty list (for URL lists).
- Do not merge multiple schools into one item even if they share the same sources.

Return a JSON object with a single field:
- schools: an array of SchoolItem objects (as defined).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_nonempty(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out


def _combine_sources(item: SchoolItem) -> List[str]:
    return _unique_nonempty((item.ahsaa_urls or []) + (item.other_urls or []))


def _county_full_name(county_short: Optional[str]) -> Optional[str]:
    if not county_short:
        return None
    c = county_short.strip()
    if not c:
        return None
    # Normalize capitalization (e.g., "madison" -> "Madison")
    c_norm = c[0].upper() + c[1:].lower() if len(c) > 1 else c.upper()
    return f"{c_norm} County"


# --------------------------------------------------------------------------- #
# Verification for one school                                                 #
# --------------------------------------------------------------------------- #
async def verify_one_school(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    item: SchoolItem,
) -> None:
    """
    Build verification sub-tree and run checks for a single school.
    """
    school_label = f"School #{idx + 1}"
    node_school = evaluator.add_parallel(
        id=f"school_{idx + 1}",
        desc=f"{school_label} qualifying high school meeting all criteria",
        parent=parent_node,
        critical=False,  # allow partial credit per school
    )

    # Prepare sources and some normalized strings
    ahsaa_sources = _unique_nonempty(item.ahsaa_urls or [])
    all_sources = _combine_sources(item)
    name = (item.name or "").strip()
    county_full = _county_full_name(item.county)
    county_short = (item.county or "").strip()
    ade_str = (item.ade_2026_2028 or "").strip()

    # -------------------------- Leaf: Class 6A --------------------------- #
    leaf_class = evaluator.add_leaf(
        id=f"school_{idx + 1}_class_6a",
        desc="School is classified as AHSAA Class 6A for the 2026–2028 reclassification period",
        parent=node_school,
        critical=True,
    )
    if name and all_sources:
        claim_class = (
            f"In the official AHSAA {RECLASS_SPAN} reclassification listings, the school '{name}' is classified in Class 6A."
        )
        await evaluator.verify(
            claim=claim_class,
            node=leaf_class,
            sources=ahsaa_sources if ahsaa_sources else all_sources,
            additional_instruction=(
                "Verify that the provided page(s) are official AHSAA classification/reclassification resources "
                f"for {RECLASS_SPAN_ALT} and that the school is explicitly listed as 'Class 6A' (allow minor variants like '6A'). "
                "Do not rely on your own memory; use the page text or screenshot to confirm."
            ),
        )
    else:
        leaf_class.score = 0.0
        leaf_class.status = "failed"

    # -------------------------- Leaf: Location --------------------------- #
    leaf_location = evaluator.add_leaf(
        id=f"school_{idx + 1}_location",
        desc="School is located in Madison, Limestone, Morgan, or Lauderdale County in Alabama",
        parent=node_school,
        critical=True,
    )
    if name and county_full and all_sources:
        allowed_list = ", ".join(sorted(ALLOWED_COUNTIES))
        claim_loc = (
            f"The school '{name}' is located in {county_full}, Alabama, and this county is one of the allowed set: "
            f"{allowed_list}."
        )
        await evaluator.verify(
            claim=claim_loc,
            node=leaf_location,
            sources=all_sources,
            additional_instruction=(
                "Confirm the school's county location from any of the provided sources (school/district site, "
                "AHSAA materials that mention location, Wikipedia, etc.). "
                f"Allowed counties are exactly: {', '.join(sorted(ALLOWED_COUNTIES))}. "
                "If the county cannot be confirmed from the provided sources, mark as not supported."
            ),
        )
    else:
        leaf_location.score = 0.0
        leaf_location.status = "failed"

    # -------------------------- Leaf: Enrollment ------------------------- #
    leaf_enroll = evaluator.add_leaf(
        id=f"school_{idx + 1}_enrollment",
        desc="School has Average Daily Enrollment between 1,200 and 1,600 students for 2026–2028",
        parent=node_school,
        critical=True,
    )
    if name and ade_str and all_sources:
        claim_enroll = (
            f"For the AHSAA {RECLASS_SPAN} reclassification, the school's Average Daily Enrollment (also called ADM/ADE) "
            f"for '{name}' is '{ade_str}', and that value lies between 1,200 and 1,600 inclusive."
        )
        await evaluator.verify(
            claim=claim_enroll,
            node=leaf_enroll,
            sources=ahsaa_sources if ahsaa_sources else all_sources,
            additional_instruction=(
                "Locate the ADE/ADM value for this school in the AHSAA classification document for 2026–2028. "
                "Treat thousands separators (commas) flexibly. Determine whether the numeric value is within [1200, 1600]. "
                "If the page only shows a number outside this range or no number is shown, mark as not supported."
            ),
        )
    else:
        leaf_enroll.score = 0.0
        leaf_enroll.status = "failed"

    # -------------------------- Leaf: Region 4 --------------------------- #
    leaf_region = evaluator.add_leaf(
        id=f"school_{idx + 1}_region",
        desc="School is assigned to Region 4 for Class 6A football",
        parent=node_school,
        critical=True,
    )
    if name and all_sources:
        claim_region = (
            f"In Class 6A football for the {RECLASS_SPAN} period, the school '{name}' is assigned to Region 4."
        )
        await evaluator.verify(
            claim=claim_region,
            node=leaf_region,
            sources=ahsaa_sources if ahsaa_sources else all_sources,
            additional_instruction=(
                "Use the AHSAA football region alignment for 2026–2028 to confirm the school's assignment. "
                "Allow minor variants like 'R4' or listings/tables denoting 'Region 4'. "
                "Ensure the context is Class 6A football, not another sport."
            ),
        )
    else:
        leaf_region.score = 0.0
        leaf_region.status = "failed"

    # -------------------------- Leaf: Reference URL ---------------------- #
    leaf_ref = evaluator.add_leaf(
        id=f"school_{idx + 1}_reference",
        desc="Provide URL reference from AHSAA official classification documents",
        parent=node_school,
        critical=False,  # non-critical per rubric
    )
    if ahsaa_sources:
        claim_ref = (
            f"The provided URL(s) are official AHSAA classification/reclassification resources for {RECLASS_SPAN_ALT} "
            "that list school classifications, regions and/or ADE/ADM values."
        )
        await evaluator.verify(
            claim=claim_ref,
            node=leaf_ref,
            sources=ahsaa_sources,
            additional_instruction=(
                "Confirm the URL belongs to an official AHSAA domain (e.g., ahsaa.com, cdn.ahsaa.com, ahsaa.arbitersports.com) "
                f"and is a classification/reclassification document relevant to {RECLASS_SPAN_ALT}."
            ),
        )
    else:
        # No AHSAA reference URL provided in the answer; treat as failed for this non-critical leaf.
        leaf_ref.score = 0.0
        leaf_ref.status = "failed"


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for identifying 3 AHSAA Class 6A Region 4 schools in or near the Huntsville area
    (northern Alabama counties) with ADE between 1,200 and 1,600 for 2026–2028.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # schools evaluated independently
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

    # 1) Extract structured school data from the answer
    extracted: SchoolsExtraction = await evaluator.extract(
        prompt=prompt_extract_schools(),
        template_class=SchoolsExtraction,
        extraction_name="schools_extraction",
    )

    # 2) Keep only the first REQUIRED_SCHOOLS items; pad if fewer
    items = list(extracted.schools[:REQUIRED_SCHOOLS])
    while len(items) < REQUIRED_SCHOOLS:
        items.append(SchoolItem())

    # 3) Add context info (criteria)
    evaluator.add_custom_info(
        {
            "required_schools": REQUIRED_SCHOOLS,
            "allowed_counties": sorted(list(ALLOWED_COUNTIES)),
            "reclassification_span": RECLASS_SPAN_ALT,
            "required_region": "Region 4",
            "ade_range": [1200, 1600],
        },
        info_type="criteria",
        info_name="evaluation_criteria",
    )

    # 4) Build verification subtree for each school
    verify_tasks = []
    for i in range(REQUIRED_SCHOOLS):
        verify_tasks.append(verify_one_school(evaluator, root, i, items[i]))

    await asyncio.gather(*verify_tasks)

    # 5) Return evaluation summary
    return evaluator.get_summary()