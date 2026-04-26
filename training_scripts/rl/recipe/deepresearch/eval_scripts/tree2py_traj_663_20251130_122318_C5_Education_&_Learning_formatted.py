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
TASK_ID = "nj_university_bigten_sports_admin"
TASK_DESCRIPTION = """
A high school student-athlete from New Jersey is planning to pursue a career in athletic administration and wants to stay in-state for both undergraduate and graduate education while competing at the highest level of college athletics. Identify the university in New Jersey that meets ALL of the following criteria: (1) Is a member of the Big Ten Conference, (2) Competes at the NCAA Division I level, (3) Offers an undergraduate degree program in sports management, kinesiology, physical education, exercise science, or a related sports administration field, and (4) Offers a master's degree program in sports administration, sports management, kinesiology, or a related field. Provide the name of the university, the specific city in New Jersey where its main athletic campus is located, and reference URLs confirming: (a) its location in New Jersey, (b) its Big Ten Conference membership, (c) its undergraduate program in a sports-related field, and (d) its graduate program in sports administration or related field.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityExtraction(BaseModel):
    university_name: Optional[str] = None
    athletic_campus_city_nj: Optional[str] = None

    # Required reference URLs per rubric
    location_urls: List[str] = Field(default_factory=list)
    big_ten_urls: List[str] = Field(default_factory=list)
    undergraduate_program_name: Optional[str] = None
    undergraduate_program_urls: List[str] = Field(default_factory=list)
    graduate_program_name: Optional[str] = None
    graduate_program_urls: List[str] = Field(default_factory=list)

    # Optional: extra URLs that may help verify NCAA Division I status (e.g., official athletics site, NCAA page)
    athletics_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_info() -> str:
    return """
    Extract from the answer the single university proposed to meet the stated criteria and all required outputs and reference URLs.

    You must return a JSON object with the following fields:
    1) university_name: The full, official name of the university identified in the answer.
    2) athletic_campus_city_nj: The specific city in New Jersey where the university's main athletic campus (primary athletics facilities) is located. If multiple NJ cities are mentioned, choose the one explicitly described as the main athletic campus or the city most consistently associated with athletics in the answer. If not provided, return null.
    3) location_urls: An array of all URLs in the answer intended to confirm the university's location in New Jersey.
    4) big_ten_urls: An array of all URLs in the answer intended to confirm the university's Big Ten Conference membership.
    5) undergraduate_program_name: The name of an eligible undergraduate degree program (e.g., Sport Management, Kinesiology, Physical Education, Exercise Science, or a closely related sports administration field). If multiple are mentioned, pick the best-fitting one; if none is named, return null.
    6) undergraduate_program_urls: An array of URLs in the answer intended to confirm the eligible undergraduate program.
    7) graduate_program_name: The name of an eligible master's degree program (e.g., Sports Administration/Management, Kinesiology, or a closely related field). If multiple are mentioned, pick the best-fitting one; if none is named, return null.
    8) graduate_program_urls: An array of URLs in the answer intended to confirm the eligible master's program.
    9) athletics_urls: An array of URLs that can help verify NCAA Division I participation (e.g., official athletics website, NCAA pages). Include only if such URLs appear in the answer; otherwise return an empty array.

    IMPORTANT URL RULES:
    - Only include URLs explicitly present in the answer text (plain URLs or markdown links).
    - Extract full, valid URLs. If a URL lacks protocol, prepend http://.
    - Do not invent or infer URLs not present in the answer.

    If any required piece of information is missing from the answer, return null for that field (or an empty array for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for u in urls:
        if not _non_empty_str(u):
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_university_tree(
    evaluator: Evaluator,
    root_node,
    info: UniversityExtraction
) -> None:
    """
    Build the verification tree as specified by the rubric and run verifications.
    All nodes under 'University_Identification' are critical, so failing any will fail the overall task.
    """

    # Normalize URL lists
    location_urls = _dedup_urls(info.location_urls)
    big_ten_urls = _dedup_urls(info.big_ten_urls)
    undergraduate_urls = _dedup_urls(info.undergraduate_program_urls)
    graduate_urls = _dedup_urls(info.graduate_program_urls)
    athletics_urls = _dedup_urls(info.athletics_urls)

    uni_name = info.university_name if _non_empty_str(info.university_name) else "the identified university"
    city_name = info.athletic_campus_city_nj if _non_empty_str(info.athletic_campus_city_nj) else None

    # Top-level critical node
    uni_node = evaluator.add_parallel(
        id="University_Identification",
        desc="Identify the NJ university meeting all athletic and academic criteria and provide required outputs and references",
        parent=root_node,
        critical=True
    )

    # 1) University name provided (critical presence check)
    evaluator.add_custom_node(
        result=_non_empty_str(info.university_name),
        id="University_Name_Provided",
        desc="Answer provides the name of the university",
        parent=uni_node,
        critical=True
    )

    # 2) Geographic requirements (parallel, all children critical)
    geo_node = evaluator.add_parallel(
        id="Geographic_Requirements",
        desc="Verify the university's geographic location and required location-related output",
        parent=uni_node,
        critical=True
    )

    # 2.a) New Jersey location (verification by URL)
    nj_location_leaf = evaluator.add_leaf(
        id="New_Jersey_Location",
        desc="The university is located in the state of New Jersey",
        parent=geo_node,
        critical=True
    )
    nj_claim = f"The university '{uni_name}' is located in New Jersey."
    await evaluator.verify(
        claim=nj_claim,
        node=nj_location_leaf,
        sources=location_urls,
        additional_instruction="Confirm the page(s) explicitly indicate the university is in New Jersey (e.g., city/state lines like 'New Brunswick, NJ' or 'Piscataway, New Jersey')."
    )

    # 2.b) Main athletic campus city in NJ (critical presence check per rubric)
    evaluator.add_custom_node(
        result=_non_empty_str(city_name),
        id="Main_Athletic_Campus_City_In_NJ",
        desc="Answer includes the specific city in New Jersey where the university's main athletic campus is located",
        parent=geo_node,
        critical=True
    )

    # 2.c) Location reference URL provided (critical presence check)
    evaluator.add_custom_node(
        result=len(location_urls) > 0,
        id="Location_Reference_URL",
        desc="Provide a reference URL confirming the university's location in New Jersey",
        parent=geo_node,
        critical=True
    )

    # 3) Athletic requirements (parallel, all children critical)
    ath_node = evaluator.add_parallel(
        id="Athletic_Requirements",
        desc="Verify the university meets the specified athletics/conference constraints",
        parent=uni_node,
        critical=True
    )

    # 3.a) NCAA Division I (verification by URLs - Big Ten membership implies Division I; athletics URLs can help)
    ncaa_leaf = evaluator.add_leaf(
        id="NCAA_Division_I",
        desc="The university competes at the NCAA Division I level",
        parent=ath_node,
        critical=True
    )
    ncaa_claim = f"The university '{uni_name}' competes at the NCAA Division I level."
    ncaa_sources = big_ten_urls + athletics_urls
    await evaluator.verify(
        claim=ncaa_claim,
        node=ncaa_leaf,
        sources=ncaa_sources,
        additional_instruction="Big Ten is an NCAA Division I conference. Confirm from the page(s) that the university participates in NCAA Division I athletics."
    )

    # 3.b) Big Ten membership (verification by URL)
    bigten_leaf = evaluator.add_leaf(
        id="Big_Ten_Membership",
        desc="The university is a member of the Big Ten Conference",
        parent=ath_node,
        critical=True
    )
    bigten_claim = f"The university '{uni_name}' is a member of the Big Ten Conference."
    await evaluator.verify(
        claim=bigten_claim,
        node=bigten_leaf,
        sources=big_ten_urls,
        additional_instruction="Verify current Big Ten membership using the provided conference membership page(s) or official sources."
    )

    # 3.c) Big Ten reference URL provided (critical presence check)
    evaluator.add_custom_node(
        result=len(big_ten_urls) > 0,
        id="Big_Ten_Reference_URL",
        desc="Provide a reference URL confirming the university's Big Ten Conference membership",
        parent=ath_node,
        critical=True
    )

    # 4) Undergraduate program requirements (parallel, all children critical)
    ug_node = evaluator.add_parallel(
        id="Undergraduate_Program_Requirements",
        desc="Verify the university offers an eligible undergraduate sports-related program and provide required reference",
        parent=uni_node,
        critical=True
    )

    # 4.a) Eligible undergraduate program (verification by URL)
    ug_leaf = evaluator.add_leaf(
        id="Eligible_Undergraduate_Program",
        desc="The university offers an undergraduate degree program in sports management, kinesiology, physical education, exercise science, or a related sports administration field",
        parent=ug_node,
        critical=True
    )
    if _non_empty_str(info.undergraduate_program_name):
        ug_claim = f"The university '{uni_name}' offers an undergraduate degree program in {info.undergraduate_program_name}, which is a sports-related field (e.g., sports management, kinesiology, physical education, exercise science, or closely related)."
    else:
        ug_claim = f"The provided page(s) show that the university '{uni_name}' offers an undergraduate degree program that falls within sports management, kinesiology, physical education, exercise science, or a closely related sports administration field."
    await evaluator.verify(
        claim=ug_claim,
        node=ug_leaf,
        sources=undergraduate_urls,
        additional_instruction="Confirm that the program is undergraduate-level (e.g., BA/BS/BEd) and clearly falls into the specified sports-related categories."
    )

    # 4.b) Undergraduate program reference URL provided (critical presence check)
    evaluator.add_custom_node(
        result=len(undergraduate_urls) > 0,
        id="Undergraduate_Program_Reference_URL",
        desc="Provide a reference URL confirming the eligible undergraduate program",
        parent=ug_node,
        critical=True
    )

    # 5) Graduate program requirements (parallel, all children critical)
    grad_node = evaluator.add_parallel(
        id="Graduate_Program_Requirements",
        desc="Verify the university offers an eligible master's program and provide required reference",
        parent=uni_node,
        critical=True
    )

    # 5.a) Eligible master's program (verification by URL)
    grad_leaf = evaluator.add_leaf(
        id="Eligible_Masters_Program",
        desc="The university offers a master's degree program in sports administration, sports management, kinesiology, or a related field",
        parent=grad_node,
        critical=True
    )
    if _non_empty_str(info.graduate_program_name):
        grad_claim = f"The university '{uni_name}' offers a master's degree program in {info.graduate_program_name}, which is sports administration/management, kinesiology, or a closely related field."
    else:
        grad_claim = f"The provided page(s) show that the university '{uni_name}' offers a master's degree program in sports administration/management, kinesiology, or a closely related field."
    await evaluator.verify(
        claim=grad_claim,
        node=grad_leaf,
        sources=graduate_urls,
        additional_instruction="Confirm that the program is master's-level (e.g., MS/MA/MEd) and clearly in sports administration/management, kinesiology, or a closely related field."
    )

    # 5.b) Graduate program reference URL provided (critical presence check)
    evaluator.add_custom_node(
        result=len(graduate_urls) > 0,
        id="Graduate_Program_Reference_URL",
        desc="Provide a reference URL confirming the eligible master's program",
        parent=grad_node,
        critical=True
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
    Evaluate an answer for the New Jersey university identification task with Big Ten, NCAA Division I,
    and both undergraduate and master's sports-related programs, returning a structured summary.
    """
    # Initialize evaluator
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

    # Extract structured info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_university_info(),
        template_class=UniversityExtraction,
        extraction_name="university_info",
    )

    # Build verification tree and run verifications
    await build_and_verify_university_tree(evaluator, root, extracted_info)

    # Return the structured summary containing the verification tree and final score
    return evaluator.get_summary()