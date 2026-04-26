import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_cs_top15_career_center_chi"
TASK_DESCRIPTION = """
Identify 4 universities in the United States where each university satisfies ALL of the following criteria:

1. Graduate Program Ranking: The university is ranked in the top 15 for Computer Science graduate programs according to U.S. News & World Report Best Graduate Schools rankings (2025 or 2026 edition).

2. NSF CAREER Award: The university has at least one faculty member in the computer science or closely related department (such as EECS) who received an NSF CAREER award between January 1, 2023, and December 31, 2025 (the award must have been officially announced by the National Science Foundation).

3. Research Center: The university officially hosts at least one recognized research center, institute, or laboratory focused on artificial intelligence, machine learning, human-computer interaction, or related computational areas (must be verifiable on official university websites as of February 2026).

4. CHI Conference Publication: The university authored at least one research paper that was accepted and appears in the proceedings of the ACM CHI Conference on Human Factors in Computing Systems in either 2024 or 2025, where the first author was affiliated with that university.

For each of the 4 universities, provide:
- The complete official name of the university
- The U.S. News Computer Science graduate program ranking (specify edition year and numerical rank)
- The name of at least one qualifying faculty member and their NSF CAREER award year
- The official name of at least one qualifying research center, institute, or laboratory
- The complete title of at least one qualifying CHI conference paper and the conference year (2024 or 2025)
- Reference URLs verifying each of the above requirements
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    # Core identification
    name: Optional[str] = None

    # Ranking info
    ranking_edition_year: Optional[str] = None  # Expect "2025" or "2026"
    ranking_number: Optional[str] = None        # Keep as string (e.g., "7", "tie 7")
    ranking_urls: List[str] = Field(default_factory=list)

    # NSF CAREER info
    career_faculty_name: Optional[str] = None
    career_year: Optional[str] = None           # Expect "2023", "2024", or "2025"
    career_urls: List[str] = Field(default_factory=list)

    # Research center info
    center_name: Optional[str] = None
    center_urls: List[str] = Field(default_factory=list)

    # CHI paper info
    chi_paper_title: Optional[str] = None
    chi_year: Optional[str] = None              # Expect "2024" or "2025"
    chi_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to 4 universities meeting the task criteria as presented in the answer. For each university, create a JSON object with the following fields. Only extract information explicitly present in the answer text. If an item is missing, set it to null (or an empty list for URLs).

    For each university object, extract:
    - name: Complete official university name as written in the answer.
    - ranking_edition_year: The U.S. News Best Graduate Schools edition year for Computer Science (must be "2025" or "2026" if provided).
    - ranking_number: The numerical rank for the Computer Science graduate program (e.g., "7", "tie 7", "5 (tie)"), exactly as written in the answer.
    - ranking_urls: Array of URL(s) given in the answer that verify the ranking (prefer U.S. News links; include any provided relevant ranking URL(s)).
    - career_faculty_name: Name of at least one faculty member (CS or closely related like EECS) who received an NSF CAREER award.
    - career_year: Award year for the NSF CAREER (e.g., "2023", "2024", or "2025") as stated in the answer.
    - career_urls: Array of URL(s) that verify the NSF CAREER award; prefer official NSF (nsf.gov) award or announcement pages. Include any provided links.
    - center_name: Official name of at least one qualifying university research center, institute, or laboratory in AI/ML/HCI or related computational areas.
    - center_urls: Array of URL(s) to official university web pages that verify the center exists as of February 2026.
    - chi_paper_title: Complete title of at least one CHI paper authored with first author affiliated with the university.
    - chi_year: Conference year ("2024" or "2025") for the CHI paper.
    - chi_urls: Array of URL(s) that verify the CHI paper (e.g., ACM Digital Library or conference proceedings page).

    Return a JSON object of the form:
    {
      "universities": [
        { ... up to 4 items ... }
      ]
    }

    Special rules:
    - Extract only URLs explicitly present in the answer text (including markdown links); do not invent URLs.
    - Keep all fields as strings (do not convert to numbers). Preserve formatting (like "tie 7") for ranking_number exactly as written.
    - If more than 4 universities are listed, include only the first 4.
    - If fewer than 4 are listed, include all available and leave missing fields as null or empty lists accordingly.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_lists: List[str]) -> List[str]:
    """Merge multiple URL lists, deduplicate while preserving order."""
    seen = set()
    merged = []
    for urls in url_lists:
        for u in urls:
            if u and isinstance(u, str):
                if u not in seen:
                    seen.add(u)
                    merged.append(u)
    return merged


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
    Build verification sub-tree for a single university with all required checks.
    All nodes under the university are marked critical to satisfy the task's critical gating.
    """
    uid = f"U{index + 1}"

    # University parent (critical, parallel)
    uni_node = evaluator.add_parallel(
        id=f"{uid}",
        desc=f"{['First','Second','Third','Fourth'][index]} qualifying university with all required criteria and information satisfied",
        parent=parent_node,
        critical=True
    )

    # 1) Name + US Location (Leaf, critical)
    name_loc_node = evaluator.add_leaf(
        id=f"{uid}_Name_Location",
        desc="Complete official university name provided and university is located in the United States",
        parent=uni_node,
        critical=True
    )
    # Use ranking and center URLs as evidence for location
    name_for_claim = uni.name or ""
    name_loc_sources = _merge_urls(uni.ranking_urls, uni.center_urls)
    name_loc_claim = f"The university '{name_for_claim}' is located in the United States of America."
    await evaluator.verify(
        claim=name_loc_claim,
        node=name_loc_node,
        sources=name_loc_sources if name_loc_sources else None,
        additional_instruction="Use the provided webpage(s) to confirm the institution is a U.S. university. Evidence may include campus location (city/state), address, or domain conventions. If no source is provided, do not rely on external knowledge."
    )

    # 2) Ranking (Aggregator: parallel, critical)
    ranking_node = evaluator.add_parallel(
        id=f"{uid}_Ranking",
        desc="University ranking criterion satisfied and complete information provided",
        parent=uni_node,
        critical=True
    )
    # 2.a) Ranking Top15 (Leaf, critical)
    rank_top15_node = evaluator.add_leaf(
        id=f"{uid}_Ranking_Top15",
        desc="University is ranked in top 15 for CS graduate programs in U.S. News 2025 or 2026 edition",
        parent=ranking_node,
        critical=True
    )
    rank_year = uni.ranking_edition_year or ""
    rank_top15_claim = f"In the {rank_year} U.S. News 'Best Graduate Schools' Computer Science ranking, {name_for_claim} is in the top 15."
    await evaluator.verify(
        claim=rank_top15_claim,
        node=rank_top15_node,
        sources=uni.ranking_urls if uni.ranking_urls else None,
        additional_instruction="Confirm the Computer Science graduate program ranking. Allow ties. The edition year must be 2025 or 2026. If the page indicates the rank number ≤ 15, count as top 15."
    )
    # 2.b) Ranking Details (Leaf, critical)
    rank_details_node = evaluator.add_leaf(
        id=f"{uid}_Ranking_Details",
        desc="Specific rank number and edition year (2025 or 2026) are provided",
        parent=ranking_node,
        critical=True
    )
    rank_number = uni.ranking_number or ""
    rank_details_claim = f"The {rank_year} edition lists {name_for_claim}'s Computer Science graduate program rank as '{rank_number}'."
    await evaluator.verify(
        claim=rank_details_claim,
        node=rank_details_node,
        sources=uni.ranking_urls if uni.ranking_urls else None,
        additional_instruction="Verify that the edition year is one of 2025 or 2026 and that the rank string (including 'tie' or formatting) matches the evidence page."
    )
    # 2.c) Ranking URL provided (Custom, critical)
    evaluator.add_custom_node(
        result=bool(uni.ranking_urls),
        id=f"{uid}_Ranking_URL",
        desc="Valid reference URL provided for ranking verification",
        parent=ranking_node,
        critical=True
    )

    # 3) NSF CAREER (Aggregator: parallel, critical)
    career_node = evaluator.add_parallel(
        id=f"{uid}_NSF_CAREER",
        desc="NSF CAREER award criterion satisfied and complete information provided",
        parent=uni_node,
        critical=True
    )
    # 3.a) Award exists (Leaf, critical)
    career_award_node = evaluator.add_leaf(
        id=f"{uid}_CAREER_Award",
        desc="At least one faculty member received NSF CAREER award between Jan 1, 2023 and Dec 31, 2025 in CS or closely related field",
        parent=career_node,
        critical=True
    )
    career_name = uni.career_faculty_name or ""
    career_year = uni.career_year or ""
    career_award_claim = f"The provided source(s) officially announce an NSF CAREER award to {career_name} at {name_for_claim} in {career_year}."
    await evaluator.verify(
        claim=career_award_claim,
        node=career_award_node,
        sources=uni.career_urls if uni.career_urls else None,
        additional_instruction="Prefer nsf.gov award or announcement pages. Confirm the award is CAREER and the institution matches the university. The year must be 2023, 2024, or 2025."
    )
    # 3.b) Details provided (Leaf, critical; logical check)
    career_details_node = evaluator.add_leaf(
        id=f"{uid}_CAREER_Details",
        desc="Faculty member name and their specific award year (2023, 2024, or 2025) are provided",
        parent=career_node,
        critical=True
    )
    career_details_claim = f"The award year '{career_year}' is one of 2023, 2024, or 2025, and the faculty name '{career_name}' is provided."
    await evaluator.verify(
        claim=career_details_claim,
        node=career_details_node,
        additional_instruction="This is a simple logical check. Pass only if the year is exactly 2023, 2024, or 2025 and the faculty name string is non-empty."
    )
    # 3.c) CAREER URL provided (Custom, critical)
    evaluator.add_custom_node(
        result=bool(uni.career_urls),
        id=f"{uid}_CAREER_URL",
        desc="Valid reference URL provided for NSF CAREER award verification",
        parent=career_node,
        critical=True
    )

    # 4) Research Center (Aggregator: parallel, critical)
    center_node = evaluator.add_parallel(
        id=f"{uid}_Research_Center",
        desc="Research center criterion satisfied and complete information provided",
        parent=uni_node,
        critical=True
    )
    # 4.a) Center exists (Leaf, critical)
    center_exists_node = evaluator.add_leaf(
        id=f"{uid}_Center_Exists",
        desc="University officially hosts at least one qualifying research center/institute/laboratory focused on AI, ML, HCI, or related computational areas",
        parent=center_node,
        critical=True
    )
    center_name = uni.center_name or ""
    center_exists_claim = f"The university {name_for_claim} officially hosts the research entity '{center_name}' focusing on AI, ML, HCI, or related computational areas."
    await evaluator.verify(
        claim=center_exists_claim,
        node=center_exists_node,
        sources=uni.center_urls if uni.center_urls else None,
        additional_instruction="Confirm via official university website that the center/institute/lab exists and the focus is AI, ML, HCI, or similar computational areas. The page should clearly indicate affiliation to the university."
    )
    # 4.b) Center name provided (Custom, critical)
    evaluator.add_custom_node(
        result=bool(center_name.strip()),
        id=f"{uid}_Center_Name",
        desc="Official name of at least one qualifying research center/institute/laboratory is provided",
        parent=center_node,
        critical=True
    )
    # 4.c) Center URL provided (Custom, critical)
    evaluator.add_custom_node(
        result=bool(uni.center_urls),
        id=f"{uid}_Center_URL",
        desc="Valid reference URL from official university website for research center verification (as of February 2026)",
        parent=center_node,
        critical=True
    )

    # 5) CHI Paper (Aggregator: parallel, critical)
    chi_node = evaluator.add_parallel(
        id=f"{uid}_CHI_Paper",
        desc="CHI conference paper criterion satisfied and complete information provided",
        parent=uni_node,
        critical=True
    )
    # 5.a) Paper exists with first author affiliation (Leaf, critical)
    chi_exists_node = evaluator.add_leaf(
        id=f"{uid}_CHI_Paper_Exists",
        desc="At least one paper with first author from this university appears in CHI 2024 or 2025 proceedings",
        parent=chi_node,
        critical=True
    )
    chi_title = uni.chi_paper_title or ""
    chi_year = uni.chi_year or ""
    chi_exists_claim = f"The paper '{chi_title}' appears in the CHI {chi_year} proceedings and the first author is affiliated with {name_for_claim}."
    await evaluator.verify(
        claim=chi_exists_claim,
        node=chi_exists_node,
        sources=uni.chi_urls if uni.chi_urls else None,
        additional_instruction="Use ACM Digital Library or official proceedings pages. Confirm the conference year is 2024 or 2025 and the first author lists {name_for_claim} as the affiliation."
    )
    # 5.b) Paper details provided (Leaf, critical)
    chi_details_node = evaluator.add_leaf(
        id=f"{uid}_CHI_Details",
        desc="Complete paper title and conference year (2024 or 2025) are provided",
        parent=chi_node,
        critical=True
    )
    chi_details_claim = f"The CHI paper title is '{chi_title}' and the conference year is {chi_year} (either 2024 or 2025)."
    await evaluator.verify(
        claim=chi_details_claim,
        node=chi_details_node,
        sources=uni.chi_urls if uni.chi_urls else None,
        additional_instruction="Confirm both the title string and the CHI year (2024 or 2025) match the evidence page."
    )
    # 5.c) CHI URL provided (Custom, critical)
    evaluator.add_custom_node(
        result=bool(uni.chi_urls),
        id=f"{uid}_CHI_URL",
        desc="Valid reference URL provided for CHI paper verification",
        parent=chi_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the task requiring 4 qualifying U.S. universities with specified criteria.
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
        default_model=model
    )

    # Extract structured universities information
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Add a critical Task Completion node to gate overall result
    task_node = evaluator.add_parallel(
        id="task_completion",
        desc="Successfully identify 4 qualifying universities in the United States with complete information for each",
        parent=root,
        critical=True
    )

    # Normalize to exactly 4 universities (pad placeholders if fewer)
    universities = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Build verification subtrees for each university
    for idx in range(4):
        await verify_university(
            evaluator=evaluator,
            parent_node=task_node,
            uni=universities[idx],
            index=idx
        )

    # Optionally, record custom info for transparency
    evaluator.add_custom_info(
        info={
            "required_universities": 4,
            "ranking_allowed_years": ["2025", "2026"],
            "career_year_range": ["2023", "2024", "2025"],
            "chi_years": ["2024", "2025"],
            "evaluation_date_context": "February 2026"
        },
        info_type="constraints",
        info_name="task_constraints"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()