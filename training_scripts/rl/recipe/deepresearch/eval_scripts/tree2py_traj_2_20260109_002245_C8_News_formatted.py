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
TASK_ID = "california_acejmc_programs"
TASK_DESCRIPTION = """
Identify four ACEJMC-accredited journalism programs in California that meet the following criteria:

1. One program from a California State University (CSU) campus in Northern California
2. One program from a California State University (CSU) campus in Central California
3. One program from a California State University (CSU) campus in Southern California
4. One program from a private university in California

For each program, provide:
- The full name of the university
- The program's ACEJMC accreditation status
- The institutional type (CSU system member or private university)
- The geographic region in California where the university is located
- A reference URL from the official ACEJMC accredited programs list or the university's official website confirming the accreditation

Note: Northern California includes the San Francisco Bay Area and regions north; Central California includes the Central Coast and Central Valley; Southern California includes the Los Angeles area, San Diego, and regions south.
"""

REGION_DEFINITIONS = {
    "Northern California": "Includes the San Francisco Bay Area and regions north (e.g., San Francisco, San José, Sacramento, Chico, Humboldt, etc.).",
    "Central California": "Includes the Central Coast and Central Valley (e.g., San Luis Obispo, Fresno, Bakersfield, Monterey, Santa Barbara, etc.).",
    "Southern California": "Includes the Los Angeles area, San Diego, and regions south (e.g., Los Angeles, Northridge, Fullerton, Long Beach, San Diego, etc.)."
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramEntry(BaseModel):
    university_name: Optional[str] = None
    acejmc_status: Optional[str] = None
    institutional_type: Optional[str] = None  # Expected values like "CSU", "California State University", "Private"
    geographic_region: Optional[str] = None   # Expected values: "Northern California", "Central California", "Southern California"
    reference_urls: List[str] = Field(default_factory=list)  # One or more URLs from ACEJMC list or official university site


class CaliforniaProgramsExtraction(BaseModel):
    northern_csu: Optional[ProgramEntry] = None
    central_csu: Optional[ProgramEntry] = None
    southern_csu: Optional[ProgramEntry] = None
    private_university: Optional[ProgramEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract four ACEJMC-accredited journalism programs in California from the answer, categorized into:
    1) northern_csu: a CSU campus in Northern California
    2) central_csu: a CSU campus in Central California
    3) southern_csu: a CSU campus in Southern California
    4) private_university: a private (non-public) university in California

    For each category, extract the following fields from the answer exactly as stated:
    - university_name: Full university name
    - acejmc_status: The accreditation status string as stated (e.g., "ACEJMC-accredited", "accredited", "provisionally accredited")
    - institutional_type: The institutional type as stated (e.g., "CSU", "California State University", "Private")
    - geographic_region: The California region string as stated (e.g., "Northern California", "Central California", "Southern California")
    - reference_urls: An array of one or more URLs that support the accreditation; Prefer URLs from ACEJMC’s official accredited programs pages or an official university webpage explicitly stating ACEJMC accreditation. Extract the actual URLs (including protocol). If no URL is provided in the answer, return an empty array.

    Selection rules:
    - If the answer lists multiple possible programs for a category, choose the first one that clearly fits the category.
    - If the answer omits a category or required fields for that category, return null for that category.
    - If the answer provides more than four programs, only map one program to each category as above.

    Region guidance you should use for mapping (do not invent, just use the answer text):
    - Northern California: includes the San Francisco Bay Area and regions north.
    - Central California: includes the Central Coast and Central Valley.
    - Southern California: includes the Los Angeles area, San Diego, and regions south.

    Return a single JSON object with keys: northern_csu, central_csu, southern_csu, private_university, each containing the specified fields or null.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
CATEGORY_CONFIG = {
    "Northern_California_CSU_Program": {
        "slot": "northern_csu",
        "desc": "One ACEJMC-accredited journalism program from a CSU campus in Northern California, with all required fields provided.",
        "expected_region": "Northern California",
        "expected_type": "CSU",
        "leaf_desc": {
            "University_Full_Name_Provided_Northern": "Response provides the full name of the university for the Northern California CSU program.",
            "ACEJMC_Status_Stated_And_Correct_Northern": "Response states the program is ACEJMC-accredited and this is correct.",
            "Institutional_Type_Stated_And_Correct_Northern": "Response states the institutional type as a CSU system campus and this is correct.",
            "Geographic_Region_Stated_And_Correct_Northern": "Response states the geographic region as Northern California (per the provided definition) and this classification is correct.",
            "Reference_URL_Authoritative_Northern": "Response provides a reference URL from ACEJMC’s accredited programs list or an official university source that corroborates the accreditation."
        }
    },
    "Central_California_CSU_Program": {
        "slot": "central_csu",
        "desc": "One ACEJMC-accredited journalism program from a CSU campus in Central California, with all required fields provided.",
        "expected_region": "Central California",
        "expected_type": "CSU",
        "leaf_desc": {
            "University_Full_Name_Provided_Central": "Response provides the full name of the university for the Central California CSU program.",
            "ACEJMC_Status_Stated_And_Correct_Central": "Response states the program is ACEJMC-accredited and this is correct.",
            "Institutional_Type_Stated_And_Correct_Central": "Response states the institutional type as a CSU system campus and this is correct.",
            "Geographic_Region_Stated_And_Correct_Central": "Response states the geographic region as Central California (per the provided definition) and this classification is correct.",
            "Reference_URL_Authoritative_Central": "Response provides a reference URL from ACEJMC’s accredited programs list or an official university source that corroborates the accreditation."
        }
    },
    "Southern_California_CSU_Program": {
        "slot": "southern_csu",
        "desc": "One ACEJMC-accredited journalism program from a CSU campus in Southern California, with all required fields provided.",
        "expected_region": "Southern California",
        "expected_type": "CSU",
        "leaf_desc": {
            "University_Full_Name_Provided_Southern": "Response provides the full name of the university for the Southern California CSU program.",
            "ACEJMC_Status_Stated_And_Correct_Southern": "Response states the program is ACEJMC-accredited and this is correct.",
            "Institutional_Type_Stated_And_Correct_Southern": "Response states the institutional type as a CSU system campus and this is correct.",
            "Geographic_Region_Stated_And_Correct_Southern": "Response states the geographic region as Southern California (per the provided definition) and this classification is correct.",
            "Reference_URL_Authoritative_Southern": "Response provides a reference URL from ACEJMC’s accredited programs list or an official university source that corroborates the accreditation."
        }
    },
    "California_Private_University_Program": {
        "slot": "private_university",
        "desc": "One ACEJMC-accredited journalism program from a private university in California, with all required fields provided.",
        "expected_region": None,  # Region can be any of the three, but must be correctly classified
        "expected_type": "Private",
        "leaf_desc": {
            "University_Full_Name_Provided_Private": "Response provides the full name of the university for the private-university program.",
            "ACEJMC_Status_Stated_And_Correct_Private": "Response states the program is ACEJMC-accredited and this is correct.",
            "Institutional_Type_Stated_And_Correct_Private": "Response states the institutional type as a private (non-public) university and this is correct.",
            "Geographic_Region_Stated_And_Correct_Private": "Response states the geographic region in California where the university is located and this classification is correct under the provided definitions.",
            "Reference_URL_Authoritative_Private": "Response provides a reference URL from ACEJMC’s accredited programs list or an official university source that corroborates the accreditation."
        }
    }
}


def _is_csu_type_text(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return ("csu" in t) or ("california state university" in t) or ("state university" in t) or ("cal poly" in t)


def _is_private_type_text(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return "private" in t or "non-public" in t or "independent" in t


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_category(
    evaluator: Evaluator,
    parent_node,
    category_id: str,
    cfg: Dict[str, Any],
    extracted: CaliforniaProgramsExtraction,
) -> None:
    """
    Build and verify the subtree for one category (Northern CSU, Central CSU, Southern CSU, Private University).
    """
    # Create category node (non-critical, parallel aggregation)
    cat_node = evaluator.add_parallel(
        id=category_id,
        desc=cfg["desc"],
        parent=parent_node,
        critical=False
    )

    # Fetch the program entry
    slot_name = cfg["slot"]
    program: Optional[ProgramEntry] = getattr(extracted, slot_name)
    uni_name = (program.university_name if program else None) or ""
    urls = (program.reference_urls if program and program.reference_urls else [])

    # University full name provided (critical)
    evaluator.add_custom_node(
        result=bool(uni_name.strip()),
        id=list(cfg["leaf_desc"].keys())[0],  # the first key corresponds to "University_Full_Name_Provided_*"
        desc=cfg["leaf_desc"][list(cfg["leaf_desc"].keys())[0]],
        parent=cat_node,
        critical=True
    )

    # ACEJMC Status stated and correct (critical)
    acejmc_leaf_id = [k for k in cfg["leaf_desc"].keys() if "ACEJMC_Status_Stated_And_Correct" in k][0]
    acejmc_leaf = evaluator.add_leaf(
        id=acejmc_leaf_id,
        desc=cfg["leaf_desc"][acejmc_leaf_id],
        parent=cat_node,
        critical=True
    )
    ace_claim = f"The journalism or mass communication program at {uni_name} is accredited by ACEJMC."
    await evaluator.verify(
        claim=ace_claim,
        node=acejmc_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm the accreditation strictly from the provided page(s). "
            "Acceptable sources include: (a) ACEJMC's official accredited programs list pages, "
            "or (b) the university's official website explicitly stating ACEJMC accreditation. "
            "Look for phrases like 'ACEJMC-accredited', 'accredited by ACEJMC', or ACEJMC logo/mention. "
            "Do not rely on third-party directories or news unless hosted on the official university domain."
        )
    )

    # Institutional type stated and correct (critical)
    inst_leaf_id = [k for k in cfg["leaf_desc"].keys() if "Institutional_Type_Stated_And_Correct" in k][0]
    inst_leaf = evaluator.add_leaf(
        id=inst_leaf_id,
        desc=cfg["leaf_desc"][inst_leaf_id],
        parent=cat_node,
        critical=True
    )

    expected_type = cfg["expected_type"]
    if expected_type == "CSU":
        type_claim = (
            f"{uni_name} is a campus of the California State University (CSU) system (i.e., a public state university)."
        )
        type_add_ins = (
            "Use the provided URL(s) to confirm affiliation with the CSU system. "
            "CSU campuses often contain 'California State University' or 'Cal State' or 'Cal Poly' in their official naming. "
            "Do not confuse UC (University of California) campuses with CSU. "
            "If the page clearly shows CSU branding or statements indicating the institution is a CSU campus, consider it correct."
        )
    else:
        type_claim = f"{uni_name} is a private (non-public) university."
        type_add_ins = (
            "Use the provided page(s) to confirm the institution is private (non-public). "
            "Private universities include institutions like USC, Stanford, Pepperdine, Chapman, etc. "
            "If the page indicates independent/private status or is clearly a private institution, consider it correct."
        )
    await evaluator.verify(
        claim=type_claim,
        node=inst_leaf,
        sources=urls,
        additional_instruction=type_add_ins
    )

    # Geographic region stated and correct (critical)
    geo_leaf_id = [k for k in cfg["leaf_desc"].keys() if "Geographic_Region_Stated_And_Correct" in k][0]
    geo_leaf = evaluator.add_leaf(
        id=geo_leaf_id,
        desc=cfg["leaf_desc"][geo_leaf_id],
        parent=cat_node,
        critical=True
    )

    expected_region = cfg["expected_region"]  # May be None for private category (must still be a correct region classification)
    if expected_region:
        geo_claim = (
            f"{uni_name} is located in {expected_region} according to the given definitions."
        )
    else:
        # Private category: verify whatever region was stated is correct under the definitions
        stated_region = (program.geographic_region if program else None) or ""
        geo_claim = (
            f"{uni_name} is located in {stated_region} according to the provided region definitions."
        )

    region_defs_text = (
        f"Region definitions: Northern California -> {REGION_DEFINITIONS['Northern California']}; "
        f"Central California -> {REGION_DEFINITIONS['Central California']}; "
        f"Southern California -> {REGION_DEFINITIONS['Southern California']}."
    )
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=urls,
        additional_instruction=(
            "Infer the region by the city/campus location shown on the provided page(s) and match it to the definitions. "
            "Do not rely on external knowledge beyond the page content and the definitions. "
            "If the page indicates a city or region that clearly falls under the defined region, consider the classification correct. "
            + region_defs_text
        )
    )

    # Reference URL authoritative (critical)
    ref_leaf_id = [k for k in cfg["leaf_desc"].keys() if "Reference_URL_Authoritative" in k][0]
    ref_leaf = evaluator.add_leaf(
        id=ref_leaf_id,
        desc=cfg["leaf_desc"][ref_leaf_id],
        parent=cat_node,
        critical=True
    )
    ref_claim = (
        f"At least one provided URL is authoritative (ACEJMC list or official university site) and corroborates ACEJMC accreditation for {uni_name}."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=urls,
        additional_instruction=(
            "To be authoritative, the URL must be either: "
            "(1) on acejmc.org within the accredited programs section; or "
            "(2) an official university website page explicitly confirming ACEJMC accreditation. "
            "Check the domain and page content. Third-party directories/articles are not authoritative."
        )
    )


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
    Evaluate an answer for the California ACEJMC-accredited journalism programs task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent categories
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

    # Extract structured programs data
    extracted_programs = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=CaliforniaProgramsExtraction,
        extraction_name="california_acejmc_programs",
    )

    # Record ground truth info (definitions for transparency)
    evaluator.add_ground_truth({
        "region_definitions": REGION_DEFINITIONS,
        "institutional_types_guidance": {
            "CSU": "California State University system campuses (public). Includes campuses branded as California State University (e.g., CSU Northridge, San José State University, Cal Poly).",
            "Private": "Private (non-public) universities in California (e.g., USC, Stanford, Pepperdine, Chapman)."
        }
    })

    # Build and verify each category subtree
    for category_id, cfg in CATEGORY_CONFIG.items():
        await verify_category(evaluator, root, category_id, cfg, extracted_programs)

    # Return structured summary
    return evaluator.get_summary()