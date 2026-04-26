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
TASK_ID = "tx_school_district_criteria_2024_25"
TASK_DESCRIPTION = (
    "Identify the Texas school district that meets ALL of the following criteria for the 2024-25 school year:\n\n"
    "- Student enrollment between 65,000 and 75,000\n"
    "- Ranks among the 10 largest school districts in Texas by enrollment\n"
    "- Operates at least 6 high schools, all of which compete in UIL Class 6A for football\n"
    "- Received a B accountability rating from the Texas Education Agency (TEA) for the 2024-25 school year\n"
    "- Is officially designated as a fast growth district\n"
    "- Is located in a Texas county other than Harris County or Dallas County\n\n"
    "Provide the following information:\n"
    "1. The district's name\n"
    "2. The county where it is located\n"
    "3. Its exact student enrollment for 2024-25\n"
    "4. The number of high schools it operates\n"
    "5. Its TEA accountability score for 2024-25"
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class DistrictExtraction(BaseModel):
    # Identity and location
    district_name: Optional[str] = None
    state: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)

    county: Optional[str] = None
    county_urls: List[str] = Field(default_factory=list)

    # Enrollment and ranking
    enrollment_2024_25: Optional[str] = None
    enrollment_year: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)

    ranking_position_text: Optional[str] = None
    ranking_urls: List[str] = Field(default_factory=list)

    # High schools and UIL
    high_school_count: Optional[str] = None
    high_school_urls: List[str] = Field(default_factory=list)

    uil_statement: Optional[str] = None  # e.g., "All HS compete in UIL 6A for football"
    uil_urls: List[str] = Field(default_factory=list)

    # TEA accountability
    tea_rating_letter: Optional[str] = None  # e.g., "B"
    tea_rating_score: Optional[str] = None   # e.g., "86"
    tea_rating_year: Optional[str] = None    # e.g., "2024-25"
    tea_urls: List[str] = Field(default_factory=list)

    # Fast growth designation
    fast_growth_statement: Optional[str] = None  # e.g., "District is fast-growth"
    growth_trends_statement: Optional[str] = None  # e.g., "enrollment has increased ..."
    growth_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_district() -> str:
    return """
    Extract structured information about the identified Texas school district exactly as it appears in the answer text. Do not infer any data not present in the answer. If a field is missing, return null (for strings) or an empty array (for URL arrays).

    Required fields to extract:

    Identity and location
    - district_name: The official name of the school district as stated in the answer.
    - state: The U.S. state explicitly mentioned for the district (should be "Texas" if present).
    - identity_urls: All URLs cited in the answer that provide basic district information (official district site, Wikipedia, TEA profile, etc.).
    - county: The county where the district is located as stated in the answer.
    - county_urls: All URLs cited that support the county location.

    Enrollment and ranking
    - enrollment_2024_25: The exact student enrollment figure for the 2024-25 school year as stated in the answer (string; keep formatting, e.g., "70,123").
    - enrollment_year: The school year associated with the enrollment figure (e.g., "2024-25", "2024/25", "2024–25").
    - enrollment_urls: All URLs cited that support the enrollment figure.
    - ranking_position_text: The district's ranking position among Texas districts as stated in the answer, if any (e.g., "7th largest").
    - ranking_urls: All URLs cited that support any ranking/top-10 claims.

    High schools and UIL
    - high_school_count: The number of high schools the district operates (string).
    - high_school_urls: All URLs cited that support the high school count.
    - uil_statement: The statement in the answer about UIL classification, preferably mentioning football and "Class 6A".
    - uil_urls: All URLs cited that support the UIL classification information.

    TEA accountability (2024-25)
    - tea_rating_letter: The TEA accountability letter grade (e.g., "B").
    - tea_rating_score: The TEA numerical score (string; keep formatting, e.g., "86").
    - tea_rating_year: The school year for which the TEA rating applies (should be 2024-25 if present).
    - tea_urls: All URLs cited that support TEA accountability information.

    Fast growth designation
    - fast_growth_statement: The statement that the district is designated as a fast growth district (as written in the answer).
    - growth_trends_statement: Any mention of enrollment increases or growth trends (optional).
    - growth_urls: All URLs cited that support fast growth designation or trends.

    Rules for URLs:
    - Extract only actual URLs present in the answer. Include full URLs (with protocol).
    - Accept both plain URLs and markdown links; always return the actual URL.
    - If no URL is provided for a category, return an empty list for that URLs field.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def non_empty_str(value: Optional[str]) -> bool:
    return isinstance(value, str) and value.strip() != ""


def any_urls(*url_lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        if lst:
            combined.extend([u for u in lst if non_empty_str(u)])
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in combined:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_identity_and_location(evaluator: Evaluator, parent, data: DistrictExtraction) -> None:
    node = evaluator.add_parallel(
        id="district_identity_and_location",
        desc="Verification of basic district identity and geographic location",
        parent=parent,
        critical=False
    )

    # Basic Identity
    basic = evaluator.add_parallel(
        id="basic_identity",
        desc="Core identifying information about the district",
        parent=node,
        critical=False
    )

    # District name provided (existence)
    evaluator.add_custom_node(
        result=non_empty_str(data.district_name),
        id="district_name_provided",
        desc="The name of the school district is provided",
        parent=basic,
        critical=True
    )

    # Identity reference URL exists
    evaluator.add_custom_node(
        result=len(data.identity_urls) > 0,
        id="identity_reference",
        desc="Reference URL provided for basic district information",
        parent=basic,
        critical=True
    )

    # Texas state location (verify via URLs)
    state_leaf = evaluator.add_leaf(
        id="texas_state_location",
        desc="The district is confirmed to be located in Texas",
        parent=basic,
        critical=True
    )
    state_sources = any_urls(data.identity_urls, data.county_urls)
    claim = f"The school district named '{data.district_name or 'the district'}' is located in Texas."
    await evaluator.verify(
        claim=claim,
        node=state_leaf,
        sources=state_sources,
        additional_instruction="Confirm that the district is in the U.S. state of Texas; evidence can be that the page states Texas or lists a Texas county for the district."
    )

    # County Location
    county = evaluator.add_parallel(
        id="county_location",
        desc="Geographic county location verification",
        parent=node,
        critical=False
    )

    # County name specified (existence)
    evaluator.add_custom_node(
        result=non_empty_str(data.county),
        id="county_name_specified",
        desc="The county location of the district is specified",
        parent=county,
        critical=True
    )

    # County reference URL exists
    evaluator.add_custom_node(
        result=len(data.county_urls) > 0,
        id="county_reference",
        desc="Reference URL provided for county location verification",
        parent=county,
        critical=True
    )

    # Not Harris County
    not_harris = evaluator.add_leaf(
        id="not_harris_county",
        desc="The district is not located in Harris County",
        parent=county,
        critical=True
    )
    claim_harris = f"The district is not located in Harris County, Texas. It is located in {data.county or '[county not specified]'} County."
    await evaluator.verify(
        claim=claim_harris,
        node=not_harris,
        sources=data.county_urls,
        additional_instruction="Verify the page shows the district's county is something other than Harris County. If the page states a specific non-Harris county, that satisfies the claim."
    )

    # Not Dallas County
    not_dallas = evaluator.add_leaf(
        id="not_dallas_county",
        desc="The district is not located in Dallas County",
        parent=county,
        critical=True
    )
    claim_dallas = f"The district is not located in Dallas County, Texas. It is located in {data.county or '[county not specified]'} County."
    await evaluator.verify(
        claim=claim_dallas,
        node=not_dallas,
        sources=data.county_urls,
        additional_instruction="Verify the page shows the district's county is something other than Dallas County. If the page states a specific non-Dallas county, that satisfies the claim."
    )


async def build_enrollment_and_ranking(evaluator: Evaluator, parent, data: DistrictExtraction) -> None:
    node = evaluator.add_parallel(
        id="enrollment_size_and_ranking",
        desc="Verification of district enrollment size and state ranking",
        parent=parent,
        critical=False
    )

    # Enrollment range verification (sequential)
    enr_seq = evaluator.add_sequential(
        id="enrollment_range_verification",
        desc="Verification that enrollment falls within the specified range",
        parent=node,
        critical=False
    )

    # Enrollment data collection (parallel)
    enr_collect = evaluator.add_parallel(
        id="enrollment_data_collection",
        desc="Collection of enrollment data for verification",
        parent=enr_seq,
        critical=False
    )

    evaluator.add_custom_node(
        result=non_empty_str(data.enrollment_2024_25),
        id="specific_enrollment_number",
        desc="Specific enrollment number is provided",
        parent=enr_collect,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(data.enrollment_urls) > 0,
        id="enrollment_data_reference",
        desc="Reference URL provided for enrollment data",
        parent=enr_collect,
        critical=True
    )

    enr_year_leaf = evaluator.add_leaf(
        id="school_year_2024_25_confirmed",
        desc="The enrollment figure is for the 2024-25 school year",
        parent=enr_collect,
        critical=True
    )
    claim_year = "The student enrollment figure is for the 2024–25 school year."
    await evaluator.verify(
        claim=claim_year,
        node=enr_year_leaf,
        sources=data.enrollment_urls,
        additional_instruction="Look for explicit mention of '2024-25', '2024/25', or '2024–25' associated with the reported enrollment figure."
    )

    # Range validation (parallel)
    range_valid = evaluator.add_parallel(
        id="range_validation",
        desc="Validation that enrollment meets the specified range",
        parent=enr_seq,
        critical=False
    )

    in_range_leaf = evaluator.add_leaf(
        id="within_65000_to_75000",
        desc="District enrollment is between 65,000 and 75,000 students",
        parent=range_valid,
        critical=True
    )
    claim_range = "The district's student enrollment for the 2024–25 school year is between 65,000 and 75,000 students (inclusive)."
    await evaluator.verify(
        claim=claim_range,
        node=in_range_leaf,
        sources=data.enrollment_urls,
        additional_instruction="Use the enrollment figure shown on the page to check if it falls within 65,000–75,000 inclusive."
    )

    # State ranking verification (sequential)
    ranking_seq = evaluator.add_sequential(
        id="state_ranking_verification",
        desc="Verification of ranking among Texas school districts",
        parent=node,
        critical=False
    )

    rank_collect = evaluator.add_parallel(
        id="ranking_data_collection",
        desc="Collection of ranking data for verification",
        parent=ranking_seq,
        critical=False
    )

    evaluator.add_custom_node(
        result=non_empty_str(data.ranking_position_text),
        id="ranking_position_stated",
        desc="The district's ranking position among Texas districts is stated",
        parent=rank_collect,
        critical=False
    )

    evaluator.add_custom_node(
        result=len(data.ranking_urls) > 0,
        id="ranking_data_reference",
        desc="Reference URL provided for ranking information",
        parent=rank_collect,
        critical=True
    )

    top10 = evaluator.add_parallel(
        id="top_10_validation",
        desc="Validation that the district ranks in top 10",
        parent=ranking_seq,
        critical=False
    )

    top10_leaf = evaluator.add_leaf(
        id="ranks_top_10_largest",
        desc="District ranks among the top 10 largest in Texas by enrollment",
        parent=top10,
        critical=True
    )
    claim_top10 = "The district ranks among the top 10 largest school districts in Texas by student enrollment."
    await evaluator.verify(
        claim=claim_top10,
        node=top10_leaf,
        sources=data.ranking_urls,
        additional_instruction="Confirm that the source explicitly places the district within the top 10 by enrollment among Texas school districts."
    )


async def build_high_school_operations(evaluator: Evaluator, parent, data: DistrictExtraction) -> None:
    node = evaluator.add_parallel(
        id="high_school_operations",
        desc="Verification of high school count and athletic classification",
        parent=parent,
        critical=False
    )

    # High school count verification (sequential)
    count_seq = evaluator.add_sequential(
        id="high_school_count_verification",
        desc="Verification of the number of high schools operated",
        parent=node,
        critical=False
    )

    count_collect = evaluator.add_parallel(
        id="count_data_collection",
        desc="Collection of high school count data",
        parent=count_seq,
        critical=False
    )

    evaluator.add_custom_node(
        result=non_empty_str(data.high_school_count),
        id="number_of_high_schools_stated",
        desc="The actual number of high schools is stated",
        parent=count_collect,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(data.high_school_urls) > 0,
        id="high_school_data_reference",
        desc="Reference URL provided for high school count",
        parent=count_collect,
        critical=True
    )

    min_count = evaluator.add_parallel(
        id="minimum_count_validation",
        desc="Validation that minimum count requirement is met",
        parent=count_seq,
        critical=False
    )

    at_least_six = evaluator.add_leaf(
        id="at_least_six_high_schools",
        desc="The district operates at least 6 high schools",
        parent=min_count,
        critical=True
    )
    claim_six = "The district operates at least six (6) high schools."
    await evaluator.verify(
        claim=claim_six,
        node=at_least_six,
        sources=data.high_school_urls,
        additional_instruction="Verify from the source(s) that the district operates six or more high schools (grades 9–12 campuses)."
    )

    # UIL classification verification (sequential)
    uil_seq = evaluator.add_sequential(
        id="uil_classification_verification",
        desc="Verification of UIL athletic classification for all high schools",
        parent=node,
        critical=False
    )

    uil_collect = evaluator.add_parallel(
        id="classification_data_collection",
        desc="Collection of UIL classification data",
        parent=uil_seq,
        critical=False
    )

    evaluator.add_custom_node(
        result=non_empty_str(data.uil_statement),
        id="uil_classification_documented",
        desc="UIL classification for the district's high schools is documented",
        parent=uil_collect,
        critical=False
    )

    evaluator.add_custom_node(
        result=len(data.uil_urls) > 0,
        id="uil_data_reference",
        desc="Reference URL provided for UIL classification information",
        parent=uil_collect,
        critical=True
    )

    class6a_parallel = evaluator.add_parallel(
        id="class_6a_validation",
        desc="Validation that all high schools compete in Class 6A for football",
        parent=uil_seq,
        critical=False
    )

    all_6a = evaluator.add_leaf(
        id="all_high_schools_class_6a_football",
        desc="All high schools in the district compete in UIL Class 6A for football",
        parent=class6a_parallel,
        critical=True
    )
    claim_6a = "All of the district's high schools compete in UIL Class 6A for football."
    await evaluator.verify(
        claim=claim_6a,
        node=all_6a,
        sources=data.uil_urls,
        additional_instruction="Confirm for each high school in the district that their UIL classification for football is 6A. Realignment pages or official UIL/district/school athletics pages are acceptable."
    )


async def build_tea_accountability(evaluator: Evaluator, parent, data: DistrictExtraction) -> None:
    node = evaluator.add_sequential(
        id="tea_accountability_performance",
        desc="Verification of Texas Education Agency accountability rating",
        parent=parent,
        critical=False
    )

    rating_collect = evaluator.add_parallel(
        id="rating_data_collection",
        desc="Collection of TEA accountability rating data",
        parent=node,
        critical=False
    )

    evaluator.add_custom_node(
        result=non_empty_str(data.tea_rating_letter),
        id="letter_grade_stated",
        desc="The district's letter grade rating is stated",
        parent=rating_collect,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_str(data.tea_rating_score),
        id="numerical_score_stated",
        desc="The district's numerical accountability score is stated",
        parent=rating_collect,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(data.tea_urls) > 0,
        id="tea_rating_reference",
        desc="Reference URL provided for TEA accountability rating",
        parent=rating_collect,
        critical=True
    )

    rating_year_leaf = evaluator.add_leaf(
        id="rating_year_2024_25",
        desc="The rating is for the 2024-25 school year",
        parent=rating_collect,
        critical=True
    )
    claim_year = "The TEA accountability rating shown applies to the 2024–25 school year."
    await evaluator.verify(
        claim=claim_year,
        node=rating_year_leaf,
        sources=data.tea_urls,
        additional_instruction="Look for explicit mention that the rating/score is for 2024-25 (or 2024–25, 2024/25)."
    )

    b_validation = evaluator.add_parallel(
        id="b_rating_validation",
        desc="Validation that the district received a B rating",
        parent=node,
        critical=False
    )

    letter_b_leaf = evaluator.add_leaf(
        id="letter_grade_is_b",
        desc="The district received a B accountability rating",
        parent=b_validation,
        critical=True
    )
    claim_b = "The district received a 'B' accountability rating from TEA for 2024–25."
    await evaluator.verify(
        claim=claim_b,
        node=letter_b_leaf,
        sources=data.tea_urls,
        additional_instruction="Confirm that the TEA page or an official summary lists the district's overall letter grade as 'B' for 2024–25."
    )

    score_between_leaf = evaluator.add_leaf(
        id="score_between_80_and_89",
        desc="The district's accountability score is between 80 and 89",
        parent=b_validation,
        critical=True
    )
    claim_score_range = "The district's TEA accountability score for 2024–25 is between 80 and 89 (inclusive)."
    await evaluator.verify(
        claim=claim_score_range,
        node=score_between_leaf,
        sources=data.tea_urls,
        additional_instruction="Confirm the numerical score is 80–89 inclusive. If the exact number is shown (e.g., 86), it satisfies the claim."
    )


async def build_growth_status(evaluator: Evaluator, parent, data: DistrictExtraction) -> None:
    node = evaluator.add_sequential(
        id="growth_status_verification",
        desc="Verification of the district's fast growth designation",
        parent=parent,
        critical=False
    )

    growth_collect = evaluator.add_parallel(
        id="growth_data_collection",
        desc="Collection of growth status and enrollment trend data",
        parent=node,
        critical=False
    )

    growth_doc_leaf = evaluator.add_leaf(
        id="fast_growth_status_documented",
        desc="The district's fast growth status is documented",
        parent=growth_collect,
        critical=True
    )
    claim_growth_doc = "The sources describe the district as 'fast growth' or otherwise indicate a fast-growth status."
    await evaluator.verify(
        claim=claim_growth_doc,
        node=growth_doc_leaf,
        sources=data.growth_urls,
        additional_instruction="Look for phrases like 'fast growth', 'fast-growing', or affiliation with the Fast Growth School Coalition."
    )

    trends_leaf = evaluator.add_leaf(
        id="enrollment_trends_documented",
        desc="Evidence of enrollment increase trends is documented",
        parent=growth_collect,
        critical=False
    )
    claim_trends = "The sources indicate that the district has experienced enrollment growth or increasing trends."
    await evaluator.verify(
        claim=claim_trends,
        node=trends_leaf,
        sources=data.growth_urls,
        additional_instruction="Mentions of increasing enrollment, rapid growth, or similar trends satisfy this non-critical check."
    )

    evaluator.add_custom_node(
        result=len(data.growth_urls) > 0,
        id="growth_data_reference",
        desc="Reference URL provided for fast growth designation",
        parent=growth_collect,
        critical=True
    )

    fast_growth_valid = evaluator.add_parallel(
        id="fast_growth_validation",
        desc="Validation that the district is officially designated as fast growth",
        parent=node,
        critical=False
    )

    official_fast_growth_leaf = evaluator.add_leaf(
        id="official_fast_growth_designation",
        desc="The district is officially designated as a fast growth district",
        parent=fast_growth_valid,
        critical=True
    )
    claim_official = "The district is officially designated as a fast growth district."
    await evaluator.verify(
        claim=claim_official,
        node=official_fast_growth_leaf,
        sources=data.growth_urls,
        additional_instruction="Prefer official designations, e.g., district board documents, TEA references, or the Fast Growth School Coalition membership listings."
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
    model: str = "o4-mini"
) -> Dict:
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

    # Create a top-level domain node to mirror the rubric root
    rubric_root = evaluator.add_parallel(
        id="texas_school_district_identification",
        desc="Comprehensive verification that the identified Texas school district meets all specified criteria",
        parent=root,
        critical=False
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_district(),
        template_class=DistrictExtraction,
        extraction_name="district_extraction"
    )

    # Add a compact extracted summary for convenience
    evaluator.add_custom_info(
        {
            "district_name": extracted.district_name,
            "county": extracted.county,
            "enrollment_2024_25": extracted.enrollment_2024_25,
            "high_school_count": extracted.high_school_count,
            "tea_score_2024_25": extracted.tea_rating_score
        },
        info_type="extracted_summary"
    )

    # Build verification subtrees
    await build_identity_and_location(evaluator, rubric_root, extracted)
    await build_enrollment_and_ranking(evaluator, rubric_root, extracted)
    await build_high_school_operations(evaluator, rubric_root, extracted)
    await build_tea_accountability(evaluator, rubric_root, extracted)
    await build_growth_status(evaluator, rubric_root, extracted)

    # Return final summary
    return evaluator.get_summary()