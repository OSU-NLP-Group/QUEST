import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "superintendent_dual_task"
TASK_DESCRIPTION = (
    "A current school district CEO (as of 2023) has the following educational and career background:\n\n"
    "- Graduated in 2006 with a bachelor's degree in psychology\n"
    "- Served as student body president at their undergraduate institution\n"
    "- Earned a master's degree in educational administration in 2009 from a university in Missouri\n"
    "- Earned a doctoral degree in urban education leadership in 2016\n"
    "- Served as a White House Fellow during the 2016-2017 term\n"
    "- Previously worked in a network leadership role for a major school district from July 2014 through August 2016\n\n"
    "Based on this information, provide:\n"
    "1. The name of the school district where this superintendent worked from July 2014 through August 2016\n"
    "2. The founding year of the university whose president assumed office on June 1, 2006"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Part1Info(BaseModel):
    superintendent_name: Optional[str] = None
    district_name: Optional[str] = None
    network_role_title: Optional[str] = None
    network_role_dates: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Part2Info(BaseModel):
    university_name: Optional[str] = None
    president_name: Optional[str] = None
    president_assumed_office_date: Optional[str] = None
    founding_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MainExtraction(BaseModel):
    part1: Optional[Part1Info] = None
    part2: Optional[Part2Info] = None


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return (
        "Extract the concrete answers and supporting details provided in the answer text for two parts.\n\n"
        "Part 1 (School District for July 2014–Aug 2016 role):\n"
        "- superintendent_name: the person's full name if stated\n"
        "- district_name: the school district name associated with the network leadership role from July 2014 through August 2016\n"
        "- network_role_title: the title of that role as stated (e.g., Network Chief, Network Leader)\n"
        "- network_role_dates: the dates or date range stated for that role\n"
        "- sources: all URLs explicitly cited that substantiate this person’s role at that district and/or the dates. Include only actual URLs present in the answer (plain or markdown links). If none, return an empty array.\n\n"
        "Part 2 (University founding year for the university whose president assumed office on June 1, 2006):\n"
        "- university_name: the university’s name\n"
        "- president_name: the president’s name if stated\n"
        "- president_assumed_office_date: the date the president assumed office, as stated in the answer\n"
        "- founding_year: the founding year value stated in the answer (as a string). If not stated, return null.\n"
        "- sources: all URLs explicitly cited that substantiate the founding year (and/or the presidency date/university identity). Include only actual URLs present in the answer (plain or markdown links). If none, return an empty array.\n\n"
        "Return a JSON object with two top-level fields: 'part1' and 'part2'. For any missing field, use null. Do not invent information not present in the answer text."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_part1(evaluator: Evaluator, parent) -> None:
    """
    Build and evaluate Part 1 subtree:
    - District_Name_Provided (critical existence)
    - District_Matches_Described_Superintendent (critical, URL-grounded)
    """
    # Create Part 1 sequential node
    part1_node = evaluator.add_sequential(
        id="Part_1_School_District",
        desc="Identify the school district where the described superintendent worked from July 2014 through August 2016.",
        parent=parent,
        critical=False
    )

    # Retrieve extracted info
    # The extraction has already been recorded as the first (and only) extraction in this script
    # Find it directly from evaluator summary would be heavy; instead, we assume we have kept a reference.
    # To keep self-contained, re-extract the last recorded extraction from internal memory is not exposed.
    # Therefore, we require verify_part1 to receive the parsed object. We'll fetch from a closure variable via evaluator.add_custom_info.
    # However, to keep clean, we will store it on evaluator as custom info and retrieve here.
    # For simplicity in this script, we will attach the extracted object to evaluator via a custom attribute.
    extracted: MainExtraction = getattr(evaluator, "_extracted_main", MainExtraction())

    p1 = extracted.part1 or Part1Info()

    # Leaf 1: Existence check for district name
    district_exists = bool(p1.district_name and p1.district_name.strip())
    evaluator.add_custom_node(
        result=district_exists,
        id="District_Name_Provided",
        desc="Answer provides a school district name for the July 2014–Aug 2016 network leadership role.",
        parent=part1_node,
        critical=True
    )

    # Leaf 2: Verify correctness via sources
    district_leaf = evaluator.add_leaf(
        id="District_Matches_Described_Superintendent",
        desc="The provided district is correct for the superintendent described by the given education/career timeline and the July 2014–Aug 2016 network leadership role.",
        parent=part1_node,
        critical=True
    )

    # Build claim
    name_clause = f"{p1.superintendent_name} " if p1.superintendent_name else "the described superintendent "
    role_clause = ""
    if p1.network_role_title and p1.network_role_title.strip():
        role_clause = f"as {p1.network_role_title.strip()} "
    dates_text = p1.network_role_dates.strip() if p1.network_role_dates else "July 2014 through August 2016"

    claim = (
        f"Between {dates_text}, {name_clause}worked {role_clause}at {p1.district_name}."
        if p1.district_name else
        "The district stated is correct for the described superintendent's July 2014–Aug 2016 network leadership role."
    )

    add_ins = (
        "Use the cited webpages to confirm that the same person described in the task (2006 BA in psychology and student "
        "body president; 2009 master's in educational administration from a Missouri university; 2016 doctorate in urban "
        "education leadership; White House Fellow 2016–2017) held a network leadership role at the named district in the "
        "stated timeframe. Allow title variants such as Network Chief, Network Leader, or similar leadership of networks. "
        "Minor wording differences for dates (e.g., from/to) are acceptable as long as they clearly cover July 2014 through August 2016."
    )

    sources = p1.sources if p1.sources else None
    await evaluator.verify(
        claim=claim,
        node=district_leaf,
        sources=sources,
        additional_instruction=add_ins
    )


async def verify_part2(evaluator: Evaluator, parent) -> None:
    """
    Build and evaluate Part 2 subtree:
    - Founding_Year_Provided (critical existence/format)
    - Founding_Year_Correct (critical, URL-grounded)
    """
    # Create Part 2 sequential node
    part2_node = evaluator.add_sequential(
        id="Part_2_University_Founding_Year",
        desc="Provide the founding year of the university whose president assumed office on June 1, 2006.",
        parent=parent,
        critical=False
    )

    # Retrieve extracted info
    extracted: MainExtraction = getattr(evaluator, "_extracted_main", MainExtraction())
    p2 = extracted.part2 or Part2Info()

    # Leaf 1: Existence/format check for founding year (must resemble a 4-digit year)
    year_text = (p2.founding_year or "").strip()
    year_is_4digit = bool(re.fullmatch(r"\d{4}", year_text))
    evaluator.add_custom_node(
        result=bool(year_text) and year_is_4digit,
        id="Founding_Year_Provided",
        desc="Answer provides a founding year (a year value) for the referenced university.",
        parent=part2_node,
        critical=True
    )

    # Leaf 2: Verify correctness via sources
    founding_leaf = evaluator.add_leaf(
        id="Founding_Year_Correct",
        desc="Answer provides the correct founding year of the university whose president assumed office on June 1, 2006.",
        parent=part2_node,
        critical=True
    )

    uni_name = (p2.university_name or "the referenced university").strip()
    claim = f"The founding year of {uni_name} is {year_text}." if year_text else "The provided founding year is correct."

    add_ins = (
        "Verify the founding year using the cited sources. The university in question is the one whose president assumed "
        "office on June 1, 2006. If a source also mentions the presidency date, use that to ensure you are looking at the "
        "correct university; otherwise, it is sufficient to confirm the founding year for the named university. If multiple "
        "dates are present (e.g., chartered vs. opened), treat the commonly cited founding year as correct."
    )

    sources = p2.sources if p2.sources else None
    await evaluator.verify(
        claim=claim,
        node=founding_leaf,
        sources=sources,
        additional_instruction=add_ins
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the superintendent dual-task:
    1) Identify the school district for the July 2014–Aug 2016 network leadership role.
    2) Provide the founding year of the university whose president assumed office on June 1, 2006.
    """
    # Initialize evaluator with a parallel root (two independent parts)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Provide the two requested outputs: "
            "(1) the school district for the superintendent’s July 2014–Aug 2016 network leadership role, "
            "and (2) the founding year of the university whose president assumed office on June 1, 2006."
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Add a root node description (kept in initialize); if needed, we can store task details
    evaluator.add_custom_info(
        info={"task_description_full": TASK_DESCRIPTION},
        info_type="task_context",
        info_name="task_context_full"
    )

    # Extract structured information from the answer
    extracted_main = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=MainExtraction,
        extraction_name="extracted_information"
    )

    # Keep a reference for downstream verification helpers
    setattr(evaluator, "_extracted_main", extracted_main)

    # Build the tree according to rubric and run verifications
    # Root node is already created in initialize; use it as parent for both parts
    root_node = evaluator.add_parallel(
        id="Root",
        desc="Provide the two requested outputs: (1) the school district for the superintendent’s July 2014–Aug 2016 network leadership role, and (2) the founding year of the university whose president assumed office on June 1, 2006.",
        parent=None,
        critical=False
    )

    # Execute Part 1 and Part 2 verifications
    await verify_part1(evaluator, root_node)
    await verify_part2(evaluator, root_node)

    # Return the structured summary
    return evaluator.get_summary()