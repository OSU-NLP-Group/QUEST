import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "districts_ivies_2024_2026"
TASK_DESCRIPTION = (
    "An education research organization is preparing a comparative analysis report on large school districts' performance and university admissions policy changes for the 2024-2025 academic year. They need you to gather the following specific information:\n\n"
    "Part 1: Identify the largest school district in Texas (by total student enrollment) that earned an overall \"A\" rating from the Texas Education Agency (TEA) for the 2024-25 school year. For this district, provide the total number of high schools it operates.\n\n"
    "Part 2: Identify the largest public school district in the United States by student enrollment. Provide the state where this district is located, and for the 2025-26 school year, provide the total number of schools it operates as well as the breakdown by school type (number of elementary schools, number of middle schools, and number of high schools).\n\n"
    "Part 3: Name two Ivy League universities that reinstated standardized testing requirements (SAT/ACT or equivalent) for applicants to the Class of 2029, after having test-optional policies during the pandemic period.\n\n"
    "For each part, provide reference URLs from your research that verify the information."
)

IVY_LEAGUE_SET = {
    "Brown University",
    "Columbia University",
    "Cornell University",
    "Dartmouth College",
    "Harvard University",
    "University of Pennsylvania",
    "Princeton University",
    "Yale University",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Part1TexasDistrict(BaseModel):
    district_name: Optional[str] = None
    tea_rating: Optional[str] = None
    rating_year: Optional[str] = None
    total_enrollment: Optional[str] = None
    high_school_count: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Part2USDistrict(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None
    enrollment: Optional[str] = None
    total_school_count_2025_26: Optional[str] = None
    elementary_schools_2025_26: Optional[str] = None
    middle_schools_2025_26: Optional[str] = None
    high_schools_2025_26: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class IvyPolicyEntry(BaseModel):
    university: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Part3IvyPolicies(BaseModel):
    universities: List[IvyPolicyEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_part1() -> str:
    return (
        "Extract the information about the largest Texas school district (by total student enrollment) that earned an overall 'A' rating from the Texas Education Agency (TEA) for the 2024-25 school year. Return a JSON object with the following fields:\n"
        "- district_name: The name of the Texas school district identified.\n"
        "- tea_rating: The overall rating letter mentioned (e.g., 'A').\n"
        "- rating_year: The school year string as presented (e.g., '2024-25', '2024–25', or '2024-2025').\n"
        "- total_enrollment: The enrollment number or description cited in the answer.\n"
        "- high_school_count: The total number of high schools the district operates as cited.\n"
        "- sources: An array of reference URLs explicitly included in the answer that verify the TEA rating, the district's enrollment size, and the high school count. Only include actual URLs present in the answer."
    )


def prompt_extract_part2() -> str:
    return (
        "Extract the information about the largest public school district in the United States by student enrollment. Return a JSON object with the following fields:\n"
        "- district_name: The name of the district.\n"
        "- state: The state where this district is located.\n"
        "- enrollment: The enrollment figure or description cited.\n"
        "- total_school_count_2025_26: The total number of schools the district operates for the 2025-26 school year.\n"
        "- elementary_schools_2025_26: The number of elementary schools for 2025-26.\n"
        "- middle_schools_2025_26: The number of middle schools for 2025-26.\n"
        "- high_schools_2025_26: The number of high schools for 2025-26.\n"
        "- sources: An array of reference URLs explicitly included in the answer that verify the district's size and the 2025-26 school counts. Only include actual URLs present in the answer."
    )


def prompt_extract_part3() -> str:
    return (
        "Extract exactly two Ivy League universities named in the answer that reinstated standardized testing requirements for applicants to the Class of 2029 after having test-optional policies during the pandemic. If more than two are listed, return only the first two.\n"
        "Return a JSON object with:\n"
        "- universities: An array of exactly two objects, each with:\n"
        "  • university: The university's full name as written.\n"
        "  • sources: An array of reference URLs explicitly included in the answer that verify the testing policy change for this university. Only include actual URLs present in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _merge_sources(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        for u in lst:
            if u not in merged:
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_part1(
    evaluator: Evaluator,
    parent_node,
    tx_info: Part1TexasDistrict,
) -> None:
    part_node = evaluator.add_parallel(
        id="Part1_TexasDistrictAnalysis",
        desc="Analysis of the largest Texas school district with an A rating from TEA in 2024-25",
        parent=parent_node,
        critical=False,
    )

    district = (tx_info.district_name or "").strip()
    tea_rating = (tx_info.tea_rating or "").strip()
    rating_year = (tx_info.rating_year or "").strip()
    hs_count = (tx_info.high_school_count or "").strip()
    sources = _clean_urls(tx_info.sources)

    # Reference existence (gate other factual checks)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="TexasReferenceProvided",
        desc="Valid reference URL(s) are provided that verify the district's A rating, enrollment size, and high school count",
        parent=part_node,
        critical=True,
    )

    # Identification present
    evaluator.add_custom_node(
        result=bool(district),
        id="TexasDistrictIdentification",
        desc="A specific school district in Texas is identified",
        parent=part_node,
        critical=True,
    )

    # Has A rating in 2024-25
    node_rating = evaluator.add_leaf(
        id="TexasDistrictHasARating",
        desc="The identified district earned an overall A rating from TEA for 2024-25",
        parent=part_node,
        critical=True,
    )
    claim_rating = (
        f"The Texas school district '{district}' earned an overall 'A' rating from the Texas Education Agency "
        f"for the {rating_year or '2024-25'} school year."
    )
    await evaluator.verify(
        claim=claim_rating,
        node=node_rating,
        sources=sources,
        additional_instruction=(
            "Confirm the TEA accountability rating. Allow minor year formatting variants such as 2024–25 or 2024-2025. "
            "The claim is only correct if the rating is the overall district rating and explicitly 'A'."
        ),
    )

    # Is largest among A-rated Texas districts
    node_largest = evaluator.add_leaf(
        id="TexasDistrictIsLargestWithARating",
        desc="Among all Texas districts with an A rating in 2024-25, the identified district has the largest total student enrollment",
        parent=part_node,
        critical=True,
    )
    claim_largest = (
        f"Among all Texas school districts that received an overall 'A' TEA rating in {rating_year or '2024-25'}, "
        f"'{district}' has the largest total student enrollment."
    )
    await evaluator.verify(
        claim=claim_largest,
        node=node_largest,
        sources=sources,
        additional_instruction=(
            "Use the provided sources to confirm that this district has the largest enrollment among A-rated Texas districts. "
            "Valid evidence includes TEA reports, district comparisons, or credible summaries quantifying enrollment."
        ),
    )

    # High school count provided
    evaluator.add_custom_node(
        result=bool(hs_count),
        id="TexasHighSchoolCountProvided",
        desc="The total number of high schools operated by the identified district is provided",
        parent=part_node,
        critical=True,
    )

    # High school count accurate
    node_hs_count = evaluator.add_leaf(
        id="TexasHighSchoolCountAccurate",
        desc="The provided high school count matches the actual number for the identified district as of 2024-25 or 2025-26",
        parent=part_node,
        critical=True,
    )
    claim_hs = f"The district '{district}' operates {hs_count} high schools."
    await evaluator.verify(
        claim=claim_hs,
        node=node_hs_count,
        sources=sources,
        additional_instruction=(
            "Verify the total number of high schools using official district directories, TEA data, or credible lists. "
            "Accept reasonable naming variants (e.g., 'Senior High', '9–12', or 'magnet high school')."
        ),
    )


async def verify_part2(
    evaluator: Evaluator,
    parent_node,
    us_info: Part2USDistrict,
) -> None:
    part_node = evaluator.add_parallel(
        id="Part2_LargestUSDistrictAnalysis",
        desc="Analysis of the largest public school district in the United States",
        parent=parent_node,
        critical=False,
    )

    district = (us_info.district_name or "").strip()
    state = (us_info.state or "").strip()
    total_count = (us_info.total_school_count_2025_26 or "").strip()
    elem_count = (us_info.elementary_schools_2025_26 or "").strip()
    mid_count = (us_info.middle_schools_2025_26 or "").strip()
    high_count = (us_info.high_schools_2025_26 or "").strip()
    sources = _clean_urls(us_info.sources)

    # Reference existence gate
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="USDistrictReferenceProvided",
        desc="Valid reference URL(s) are provided that verify the district's size and 2025-26 school counts",
        parent=part_node,
        critical=True,
    )

    # Identification present
    evaluator.add_custom_node(
        result=bool(district),
        id="USDistrictIdentification",
        desc="A specific school district is identified",
        parent=part_node,
        critical=True,
    )

    # Largest by enrollment
    node_largest = evaluator.add_leaf(
        id="USDistrictIsLargest",
        desc="The identified district is the largest public school district in the United States by student enrollment",
        parent=part_node,
        critical=True,
    )
    claim_largest = (
        f"The public school district '{district}' is the largest in the United States by student enrollment."
    )
    await evaluator.verify(
        claim=claim_largest,
        node=node_largest,
        sources=sources,
        additional_instruction=(
            "Confirm the district ranking by enrollment (not by number of schools). "
            "Accept authoritative sources such as official district statistics, state education reports, or credible national summaries."
        ),
    )

    # State provided
    evaluator.add_custom_node(
        result=bool(state),
        id="USDistrictStateProvided",
        desc="The state where the district is located is provided",
        parent=part_node,
        critical=True,
    )

    # State accurate
    node_state = evaluator.add_leaf(
        id="USDistrictStateAccurate",
        desc="The provided state correctly matches where the identified district is located",
        parent=part_node,
        critical=True,
    )
    claim_state = f"The district '{district}' is located in the state of {state}."
    await evaluator.verify(
        claim=claim_state,
        node=node_state,
        sources=sources,
        additional_instruction="Use the provided sources to confirm the district's state.",
    )

    # School counts (critical group)
    counts_node = evaluator.add_parallel(
        id="USDistrictSchoolCounts",
        desc="School counts for the 2025-26 school year are provided",
        parent=part_node,
        critical=True,
    )

    # Total provided
    evaluator.add_custom_node(
        result=bool(total_count),
        id="TotalSchoolCountProvided",
        desc="The total number of schools for 2025-26 is provided",
        parent=counts_node,
        critical=True,
    )

    # Total accurate
    node_total_acc = evaluator.add_leaf(
        id="TotalSchoolCountAccurate",
        desc="The provided total school count matches the actual count for the identified district for 2025-26",
        parent=counts_node,
        critical=True,
    )
    claim_total = f"For the 2025-26 school year, the district '{district}' operates {total_count} schools in total."
    await evaluator.verify(
        claim=claim_total,
        node=node_total_acc,
        sources=sources,
        additional_instruction="Confirm the total number of schools for 2025-26 in official statistics or credible sources.",
    )

    # Elementary provided
    evaluator.add_custom_node(
        result=bool(elem_count),
        id="ElementaryCountProvided",
        desc="The number of elementary schools is provided",
        parent=counts_node,
        critical=True,
    )

    # Elementary accurate
    node_elem_acc = evaluator.add_leaf(
        id="ElementaryCountAccurate",
        desc="The provided elementary school count matches the actual count for the identified district for 2025-26",
        parent=counts_node,
        critical=True,
    )
    claim_elem = f"The district '{district}' has {elem_count} elementary schools for 2025-26."
    await evaluator.verify(
        claim=claim_elem,
        node=node_elem_acc,
        sources=sources,
        additional_instruction="Verify the elementary school count for 2025-26 using official or credible sources.",
    )

    # Middle provided
    evaluator.add_custom_node(
        result=bool(mid_count),
        id="MiddleCountProvided",
        desc="The number of middle schools is provided",
        parent=counts_node,
        critical=True,
    )

    # Middle accurate
    node_mid_acc = evaluator.add_leaf(
        id="MiddleCountAccurate",
        desc="The provided middle school count matches the actual count for the identified district for 2025-26",
        parent=counts_node,
        critical=True,
    )
    claim_mid = f"The district '{district}' has {mid_count} middle schools for 2025-26."
    await evaluator.verify(
        claim=claim_mid,
        node=node_mid_acc,
        sources=sources,
        additional_instruction="Verify the middle school count for 2025-26 using official or credible sources.",
    )

    # High provided
    evaluator.add_custom_node(
        result=bool(high_count),
        id="HighCountProvided",
        desc="The number of high schools is provided",
        parent=counts_node,
        critical=True,
    )

    # High accurate
    node_high_acc = evaluator.add_leaf(
        id="HighCountAccurate",
        desc="The provided high school count matches the actual count for the identified district for 2025-26",
        parent=counts_node,
        critical=True,
    )
    claim_high = f"The district '{district}' has {high_count} high schools for 2025-26."
    await evaluator.verify(
        claim=claim_high,
        node=node_high_acc,
        sources=sources,
        additional_instruction="Verify the high school count for 2025-26 using official or credible sources.",
    )


async def verify_part3(
    evaluator: Evaluator,
    parent_node,
    ivy_info: Part3IvyPolicies,
) -> None:
    part_node = evaluator.add_parallel(
        id="Part3_IvyLeagueTestingChanges",
        desc="Identification of two Ivy League universities that reinstated testing for Class of 2029",
        parent=parent_node,
        critical=False,
    )

    # Take the first two named universities with non-empty names
    named = [u for u in ivy_info.universities if (u.university or "").strip()]
    # Ensure exactly two entries for evaluation
    if len(named) > 2:
        named = named[:2]

    uni1 = (named[0].university if len(named) >= 1 else "") or ""
    uni2 = (named[1].university if len(named) >= 2 else "") or ""
    sources1 = _clean_urls(named[0].sources if len(named) >= 1 else [])
    sources2 = _clean_urls(named[1].sources if len(named) >= 2 else [])
    all_sources = _merge_sources(sources1, sources2)

    # Reference existence gate (both must have sources)
    evaluator.add_custom_node(
        result=(len(sources1) > 0 and len(sources2) > 0),
        id="IvyLeagueReferenceProvided",
        desc="Valid reference URL(s) are provided that verify the testing policy changes for both universities",
        parent=part_node,
        critical=True,
    )

    # Exactly two universities named
    evaluator.add_custom_node(
        result=(len(named) == 2),
        id="TwoUniversitiesNamed",
        desc="Exactly two universities are named",
        parent=part_node,
        critical=True,
    )

    # Both are Ivy League members
    node_ivy_membership = evaluator.add_leaf(
        id="BothAreIvyLeague",
        desc="Both named universities are members of the Ivy League (Brown, Columbia, Cornell, Dartmouth, Harvard, Penn, Princeton, or Yale)",
        parent=part_node,
        critical=True,
    )
    claim_ivy = f"Both '{uni1}' and '{uni2}' are members of the Ivy League."
    await evaluator.verify(
        claim=claim_ivy,
        node=node_ivy_membership,
        sources=all_sources,
        additional_instruction=(
            "Confirm that each institution is one of: Brown, Columbia, Cornell, Dartmouth, Harvard, University of Pennsylvania, Princeton, or Yale."
        ),
    )

    # Both reinstated standardized testing requirements
    node_reinstated = evaluator.add_leaf(
        id="BothReinstatedTesting",
        desc="Both universities reinstated standardized testing requirements (SAT/ACT or equivalent) for applicants",
        parent=part_node,
        critical=True,
    )
    claim_reinstated = (
        f"Both '{uni1}' and '{uni2}' reinstated standardized testing requirements for applicants (SAT/ACT or acceptable equivalents)."
    )
    await evaluator.verify(
        claim=claim_reinstated,
        node=node_reinstated,
        sources=all_sources,
        additional_instruction=(
            "Look for official announcements or policy pages indicating that standardized tests are again required or test-flexible with mandatory scores."
        ),
    )

    # Applies to Class of 2029 admissions
    node_class_2029 = evaluator.add_leaf(
        id="BothReinstatedForClass2029",
        desc="Both universities' testing reinstatement applies specifically to Class of 2029 admissions",
        parent=part_node,
        critical=True,
    )
    claim_class_2029 = (
        f"The reinstated testing requirements at '{uni1}' and '{uni2}' apply to applicants to the Class of 2029 "
        f"(i.e., entering Fall 2025 and graduating in 2029)."
    )
    await evaluator.verify(
        claim=claim_class_2029,
        node=node_class_2029,
        sources=all_sources,
        additional_instruction=(
            "Confirm that the reinstatement timeline corresponds to the Class of 2029 admissions cycle."
        ),
    )

    # Both had test-optional policies during pandemic
    node_prior_optional = evaluator.add_leaf(
        id="BothHadTestOptional",
        desc="Both universities had test-optional policies during the pandemic period before reinstating requirements",
        parent=part_node,
        critical=True,
    )
    claim_prior_optional = (
        f"Before reinstating testing requirements, both '{uni1}' and '{uni2}' had test-optional policies during the pandemic period."
    )
    await evaluator.verify(
        claim=claim_prior_optional,
        node=node_prior_optional,
        sources=all_sources,
        additional_instruction=(
            "Verify that each institution had test-optional policies during the pandemic (e.g., 2020–2023) before returning to required testing."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Complete analysis of large school districts and Ivy League testing policy changes",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured info from the answer
    part1_info = await evaluator.extract(
        prompt=prompt_extract_part1(),
        template_class=Part1TexasDistrict,
        extraction_name="part1_texas_district",
    )

    part2_info = await evaluator.extract(
        prompt=prompt_extract_part2(),
        template_class=Part2USDistrict,
        extraction_name="part2_us_district",
    )

    part3_info = await evaluator.extract(
        prompt=prompt_extract_part3(),
        template_class=Part3IvyPolicies,
        extraction_name="part3_ivies",
    )

    # Build verification tree for each part
    await verify_part1(evaluator, root, part1_info)
    await verify_part2(evaluator, root, part2_info)
    await verify_part3(evaluator, root, part3_info)

    # Return the summary
    return evaluator.get_summary()