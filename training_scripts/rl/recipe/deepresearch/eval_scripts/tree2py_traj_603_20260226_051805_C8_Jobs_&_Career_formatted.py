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
TASK_ID = "coops_2026_task"
TASK_DESCRIPTION = (
    "Identify 4 universities from the U.S. News & World Report's 2026 'Schools with Great Internships/Co-ops' ranking "
    "that offer cooperative education (co-op) programs and meet the following criteria: "
    "(1) each university must be a 4-year degree-granting institution, "
    "(2) each must have publicly accessible co-op program information on its official website or career center website, "
    "(3) each must have publicly accessible employment outcome data on its official website or career center website, "
    "(4) the 4 universities must collectively represent at least 3 different U.S. states, and "
    "(5) provide verifiable reference URLs from official university sources for each university's co-op program details and employment outcomes data."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    """Information for one university."""
    name: Optional[str] = None
    state: Optional[str] = None
    ranking_url: Optional[str] = None
    coop_urls: List[str] = Field(default_factory=list)
    employment_urls: List[str] = Field(default_factory=list)
    official_homepage_url: Optional[str] = None


class UniversitiesExtraction(BaseModel):
    """Container for up to 4 universities extracted from the answer."""
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to 4 universities that the answer claims come from the U.S. News & World Report 2026 "Schools with Great Internships/Co-ops" ranking.
    For each selected university, return a JSON object containing the following fields:
    - name: The university name as stated in the answer (string).
    - state: The U.S. state where the university is located, as stated in the answer (string; abbreviations like "MA" or "CA" are acceptable).
    - ranking_url: A URL to a U.S. News page that verifies inclusion in the 2026 "Schools with Great Internships/Co-ops" ranking. 
      If the answer provides one ranking page that covers multiple universities, repeat that URL for each university.
    - coop_urls: A list of official university or career center URLs that provide publicly accessible cooperative education (co-op) program information.
      These should be university-owned domains (e.g., *.edu, career.*.edu, or clearly official subdomains). Extract them exactly as provided in the answer.
    - employment_urls: A list of official university or career center URLs that provide publicly accessible employment outcomes data 
      (e.g., first destination outcomes, career outcomes dashboards, undergraduate outcomes reports). Use only URLs explicitly present in the answer.
    - official_homepage_url: The official university homepage URL if explicitly present in the answer; otherwise null.
    
    RULES:
    - Only extract URLs explicitly present in the answer. Do not invent or infer URLs.
    - If a field is not present in the answer, set it to null (or an empty list for coop_urls/employment_urls).
    - If more than 4 universities are mentioned, return only the first 4 in the order they appear.
    - If fewer than 4 are mentioned, return only those that appear.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"#{n}")


def _dedupe_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        u_str = u.strip()
        if not u_str:
            continue
        if u_str not in seen:
            seen.add(u_str)
            out.append(u_str)
    return out


# --------------------------------------------------------------------------- #
# University verification subroutine                                          #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    item: UniversityItem,
    index_1_based: int,
) -> Tuple[Any, str]:
    """
    Build verification subtree for one university and run all leaf checks.
    Returns (university_group_node, extracted_state_or_empty).
    """
    # Group node per university (non-critical to allow partial credit across universities)
    uni_node = evaluator.add_parallel(
        id=f"University_{index_1_based}",
        desc=f"{ordinal(index_1_based)} university identified meets all criteria",
        parent=parent_node,
        critical=False
    )

    # Existence checks (critical siblings to gate subsequent leaves automatically)
    name_exists = evaluator.add_custom_node(
        result=bool(item.name and item.name.strip()),
        id=f"University_{index_1_based}_Name_Provided",
        desc="University name is clearly stated",
        parent=uni_node,
        critical=True
    )

    state_exists = evaluator.add_custom_node(
        result=bool(item.state and item.state.strip()),
        id=f"University_{index_1_based}_State_Location",
        desc="U.S. state location is specified",
        parent=uni_node,
        critical=True
    )

    ranking_url_exists = evaluator.add_custom_node(
        result=bool(item.ranking_url and item.ranking_url.strip()),
        id=f"University_{index_1_based}_Reference_URL_Ranking",
        desc="Valid reference URL verifying the university's inclusion in the US News ranking",
        parent=uni_node,
        critical=True
    )

    coop_urls_exist = evaluator.add_custom_node(
        result=bool(item.coop_urls and len(item.coop_urls) > 0),
        id=f"University_{index_1_based}_Reference_URL_Coop",
        desc="Valid reference URL from official university source documenting co-op program details",
        parent=uni_node,
        critical=True
    )

    emp_urls_exist = evaluator.add_custom_node(
        result=bool(item.employment_urls and len(item.employment_urls) > 0),
        id=f"University_{index_1_based}_Reference_URL_Employment",
        desc="Valid reference URL from official university source providing employment outcome data",
        parent=uni_node,
        critical=True
    )

    # Verification leaves
    # 1) Ranking inclusion
    ranking_leaf = evaluator.add_leaf(
        id=f"University_{index_1_based}_Ranking_Verification",
        desc="University appears in US News 2026 Schools with Great Internships/Co-ops ranking",
        parent=uni_node,
        critical=True
    )
    rank_claim = (
        f"The university '{item.name or ''}' appears on the U.S. News & World Report 2026 "
        f"'Schools with Great Internships/Co-ops' ranking page."
    )
    await evaluator.verify(
        claim=rank_claim,
        node=ranking_leaf,
        sources=item.ranking_url,
        additional_instruction=(
            "Use the provided U.S. News ranking page to confirm the school name is listed for the 2026 'Schools with Great Internships/Co-ops'. "
            "Minor name variations are acceptable (case, punctuation). The URL should be on usnews.com or education.usnews.com."
        ),
    )

    # 2) Four-year institution check
    four_year_leaf = evaluator.add_leaf(
        id=f"University_{index_1_based}_Four_Year_Institution",
        desc="Confirmed as a 4-year degree-granting institution",
        parent=uni_node,
        critical=True
    )
    # Combine all available official sources to support the claim
    fy_sources = _dedupe_urls(
        [item.ranking_url, item.official_homepage_url] + item.coop_urls + item.employment_urls
    )
    fy_claim = (
        f"'{item.name or ''}' is a 4-year degree-granting institution that offers bachelor's degree programs."
    )
    await evaluator.verify(
        claim=fy_claim,
        node=four_year_leaf,
        sources=fy_sources if fy_sources else item.ranking_url,
        additional_instruction=(
            "Confirm via official pages (homepage, academics, facts pages) or the ranking page that the school grants 4-year bachelor's degrees. "
            "If multiple URLs are provided, use any official university-owned domain (e.g., *.edu, career.*.edu) or corresponding official subdomains."
        ),
    )

    # 3) Co-op exists
    coop_exists_leaf = evaluator.add_leaf(
        id=f"University_{index_1_based}_Coop_Program_Exists",
        desc="University offers cooperative education (co-op) programs with structured work experiences",
        parent=uni_node,
        critical=True
    )
    coop_claim = (
        f"'{item.name or ''}' offers cooperative education (co-op) programs that provide structured work experiences."
    )
    await evaluator.verify(
        claim=coop_claim,
        node=coop_exists_leaf,
        sources=item.coop_urls,
        additional_instruction=(
            "Verify the presence of a cooperative education program. Accept synonyms like 'co-op', 'cooperative education', "
            "'co-operative', and confirm program structure/experiential learning aligns with co-op definition."
        ),
    )

    # 4) Co-op info publicly accessible on official site
    coop_info_leaf = evaluator.add_leaf(
        id=f"University_{index_1_based}_Coop_Info_Accessible",
        desc="Co-op program information is publicly accessible on official university or career center website",
        parent=uni_node,
        critical=True
    )
    coop_info_claim = (
        "The provided URLs are official university or career center pages containing publicly accessible co-op program details."
    )
    await evaluator.verify(
        claim=coop_info_claim,
        node=coop_info_leaf,
        sources=item.coop_urls,
        additional_instruction=(
            f"Confirm that these pages belong to the official domain of '{item.name or ''}' (e.g., *.edu or recognized university subdomains) "
            "and that the pages present co-op program information accessible without login."
        ),
    )

    # 5) Employment outcomes publicly accessible on official site
    emp_info_leaf = evaluator.add_leaf(
        id=f"University_{index_1_based}_Employment_Data_Accessible",
        desc="Employment outcome data is publicly accessible on official university or career center website",
        parent=uni_node,
        critical=True
    )
    emp_info_claim = (
        "The provided URLs are official university or career center pages that publish publicly accessible employment outcomes data."
    )
    await evaluator.verify(
        claim=emp_info_claim,
        node=emp_info_leaf,
        sources=item.employment_urls,
        additional_instruction=(
            f"Confirm that these pages belong to the official domain of '{item.name or ''}' and include employment outcomes content "
            "(e.g., first-destination outcomes, career outcomes dashboards, undergraduate outcomes reports) accessible without login."
        ),
    )

    return uni_node, (item.state or "").strip()


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
    Evaluate an answer for the US News 2026 Co-ops task.
    """
    # Initialize evaluator (root is non-critical by design; we will add our own critical gate nodes)
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

    # Extract universities presented in the answer
    extracted: UniversitiesExtraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Keep only the first 4 universities; pad if fewer to maintain structure
    universities = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Add a container node for university groups
    universities_main = evaluator.add_parallel(
        id="Universities_Main",
        desc="Four universities verification bundle",
        parent=root,
        critical=False
    )

    # Verify each university block
    university_nodes: List[Any] = []
    states_collected: List[str] = []
    for idx, uni in enumerate(universities, start=1):
        node, st = await verify_university(
            evaluator=evaluator,
            parent_node=universities_main,
            item=uni,
            index_1_based=idx,
        )
        university_nodes.append(node)
        if st:
            states_collected.append(st)

    # Geographic diversity check (critical): at least 3 distinct states among the 4
    distinct_states = set(s.strip().lower() for s in states_collected if s.strip())
    geo_diversity_ok = (len(distinct_states) >= 3)
    evaluator.add_custom_info(
        {"states": states_collected, "distinct_state_count": len(distinct_states)},
        info_type="state_distribution",
        info_name="geographic_diversity_input"
    )

    geo_node = evaluator.add_custom_node(
        result=geo_diversity_ok,
        id="Geographic_Diversity",
        desc="The 4 identified universities are located in at least 3 different U.S. states",
        parent=root,
        critical=True
    )

    # Task Completion gate (critical): all 4 universities fully pass AND geographic diversity holds
    # Compute each university group's aggregated score
    all_universities_valid = all(u_node.aggregated_score == 1.0 for u_node in university_nodes)
    task_completion_result = bool(all_universities_valid and geo_diversity_ok)

    evaluator.add_custom_info(
        {
            "universities_pass_all": all_universities_valid,
            "geo_diversity_ok": geo_diversity_ok,
        },
        info_type="task_gate_components",
        info_name="task_completion_components"
    )

    evaluator.add_custom_node(
        result=task_completion_result,
        id="Task_Completion",
        desc="Successfully identify 4 universities from the US News 2026 'Schools with Great Internships/Co-ops' ranking that meet all specified criteria",
        parent=root,
        critical=True
    )

    # Return structured summary
    return evaluator.get_summary()