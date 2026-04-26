import asyncio
import logging
from typing import Any, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "dos_sip_eligibility"
TASK_DESCRIPTION = "Correctly identify all eligibility requirements for the U.S. Department of State Student Internship Program"


class SourceExtraction(BaseModel):
    source_urls: List[str] = Field(default_factory=list)


def prompt_extract_sources() -> str:
    return """
    Extract all URLs explicitly cited in the answer that relate to the U.S. Department of State Student Internship Program,
    including eligibility requirements, program details, security clearance guidance, compensation, or documentation.
    Return a JSON object with:
    - source_urls: an array of all URLs found in the answer (include full URLs; accept plain URLs, markdown links, or embedded links).
    If no URLs are provided, return an empty array.
    Only include URLs that appear in the answer text.
    """


async def build_requirement_group(
    evaluator: Evaluator,
    parent_node,
    group_id: str,
    group_desc: str,
    mention_claim: str,
    support_claim: str,
    sources: List[str],
    group_critical: bool,
    mention_additional_instruction: str,
    support_additional_instruction: str,
) -> None:
    """
    Build a sequential group for a single requirement:
      1) Check the answer explicitly mentions the requirement (simple verification).
      2) Check the requirement is supported by cited sources (URL verification).
    """
    req_node = evaluator.add_sequential(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=group_critical,
    )

    mention_node = evaluator.add_leaf(
        id=f"{group_id}_mentioned",
        desc=f"Answer mentions: {group_desc}",
        parent=req_node,
        critical=True,
    )
    await evaluator.verify(
        claim=mention_claim,
        node=mention_node,
        additional_instruction=mention_additional_instruction,
    )

    support_node = evaluator.add_leaf(
        id=f"{group_id}_supported_by_sources",
        desc=f"Sources support: {group_desc}",
        parent=req_node,
        critical=True,
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_node,
        sources=sources if sources else None,
        additional_instruction=support_additional_instruction,
    )


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
    Evaluate an answer for the U.S. Department of State Student Internship Program eligibility requirements.
    """
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

    # Extract sources from the answer
    extracted_sources = await evaluator.extract(
        prompt=prompt_extract_sources(),
        template_class=SourceExtraction,
        extraction_name="answer_sources",
    )
    sources_list = extracted_sources.source_urls

    # Record expected requirements (ground truth rubric descriptors)
    evaluator.add_ground_truth({
        "expected_requirements": [
            "Applicants must be U.S. citizens",
            "Applicants must have a minimum cumulative GPA of 3.2",
            "Applicants must be undergraduate juniors/seniors or graduate students enrolled in a degree-seeking program at an accredited college or university",
            "Applicants must be able to receive either a Public Trust, Secret, or Top Secret security clearance",
            "Students must be returning to school immediately following the internship",
            "Applicants must provide an official or unofficial transcript as proof of student status",
        ],
        "optional_information": [
            "Internship is paid at GS-04/Step 1 base pay level"
        ]
    })

    # Build and verify each requirement group

    # 1. U.S. Citizenship (critical)
    await build_requirement_group(
        evaluator=evaluator,
        parent_node=root,
        group_id="US_Citizenship_Requirement",
        group_desc="State that applicants must be U.S. citizens",
        mention_claim="The answer explicitly states that applicants must be U.S. citizens to be eligible for the U.S. Department of State Student Internship Program.",
        support_claim="Applicants must be U.S. citizens to be eligible for the U.S. Department of State Student Internship Program.",
        sources=sources_list,
        group_critical=True,
        mention_additional_instruction="Check only the provided answer text and determine if it clearly states the U.S. citizenship requirement. Minor rephrasing is acceptable.",
        support_additional_instruction="Verify on the cited webpage(s) that U.S. citizenship is explicitly listed as an eligibility requirement for the U.S. Department of State Student Internship Program.",
    )

    # 2. Minimum GPA 3.2 (critical)
    await build_requirement_group(
        evaluator=evaluator,
        parent_node=root,
        group_id="Minimum_GPA_Requirement",
        group_desc="Specify that applicants must have a minimum cumulative GPA of 3.2",
        mention_claim="The answer explicitly states that applicants must have a minimum cumulative GPA of 3.2.",
        support_claim="Applicants must have a minimum cumulative GPA of 3.2 to be eligible for the U.S. Department of State Student Internship Program.",
        sources=sources_list,
        group_critical=True,
        mention_additional_instruction="Check the answer for a clear reference to a minimum GPA of 3.2. Accept equivalent phrasing such as '3.2 or higher'.",
        support_additional_instruction="Confirm that the cited webpage(s) explicitly state the minimum GPA requirement as 3.2 (or '3.2 or higher').",
    )

    # 3. Student status: juniors/seniors or graduate; degree-seeking at accredited institution (critical)
    await build_requirement_group(
        evaluator=evaluator,
        parent_node=root,
        group_id="Student_Status_Requirement",
        group_desc="Identify that applicants must be undergraduate juniors/seniors or graduate students enrolled in a degree-seeking program at an accredited college or university",
        mention_claim="The answer explicitly states that applicants must be undergraduate juniors or seniors OR graduate students, enrolled in a degree-seeking program at an accredited college or university.",
        support_claim="Applicants must be undergraduate juniors or seniors OR graduate students enrolled in a degree-seeking program at an accredited college or university to be eligible for the U.S. Department of State Student Internship Program.",
        sources=sources_list,
        group_critical=True,
        mention_additional_instruction="Check the answer for both elements: class standing (junior/senior or graduate) and being enrolled in a degree-seeking program at an accredited institution. Minor rephrasing is acceptable.",
        support_additional_instruction="Verify that the cited webpage(s) explicitly state the student status requirement, including class standing and enrollment in a degree-seeking program at an accredited institution.",
    )

    # 4. Security clearance capability (critical)
    await build_requirement_group(
        evaluator=evaluator,
        parent_node=root,
        group_id="Security_Clearance_Requirement",
        group_desc="State that applicants must be able to receive either a Public Trust, Secret, or Top Secret security clearance",
        mention_claim="The answer explicitly states that applicants must be able to receive a security clearance such as Public Trust, Secret, or Top Secret.",
        support_claim="Applicants must be able to receive a Public Trust, Secret, or Top Secret security clearance to be eligible for the U.S. Department of State Student Internship Program.",
        sources=sources_list,
        group_critical=True,
        mention_additional_instruction="Check the answer for a requirement to be eligible for or capable of obtaining a security clearance (Public Trust, Secret, or Top Secret). Equivalent wording is acceptable.",
        support_additional_instruction="Confirm on the cited webpage(s) that eligibility requires being able to receive a security clearance (Public Trust, Secret, or Top Secret).",
    )

    # 5. Return to school immediately after internship (critical)
    await build_requirement_group(
        evaluator=evaluator,
        parent_node=root,
        group_id="Return_to_School_Requirement",
        group_desc="Specify that students must be returning to school immediately following the internship",
        mention_claim="The answer explicitly states that students must be returning to school immediately following the internship.",
        support_claim="Students must be returning to school immediately following the internship to be eligible for the U.S. Department of State Student Internship Program.",
        sources=sources_list,
        group_critical=True,
        mention_additional_instruction="Check the answer for a clear statement that students must continue their enrollment or return to school right after the internship.",
        support_additional_instruction="Verify that the cited webpage(s) explicitly state the requirement to return to school immediately after the internship (continue enrollment in the term following the internship).",
    )

    # 6. Required documentation: transcript (critical)
    await build_requirement_group(
        evaluator=evaluator,
        parent_node=root,
        group_id="Required_Documentation",
        group_desc="Identify that applicants must provide an official or unofficial transcript as proof of student status",
        mention_claim="The answer explicitly states that applicants must provide an official or unofficial transcript as proof of student status.",
        support_claim="Applicants must provide an official or unofficial transcript as proof of student status to apply for the U.S. Department of State Student Internship Program.",
        sources=sources_list,
        group_critical=True,
        mention_additional_instruction="Check the answer for 'official or unofficial transcript' as a required document to demonstrate student status.",
        support_additional_instruction="Confirm on the cited webpage(s) that an official or unofficial transcript is required to prove student status.",
    )

    # 7. Compensation information: GS-04 Step 1 (non-critical informational)
    await build_requirement_group(
        evaluator=evaluator,
        parent_node=root,
        group_id="Compensation_Information",
        group_desc="State that the internship is paid at GS-04/Step 1 base pay level",
        mention_claim="The answer explicitly states that the internship is paid at the GS-04, Step 1 base pay level.",
        support_claim="The U.S. Department of State Student Internship Program is paid at the GS-04, Step 1 base pay level.",
        sources=sources_list,
        group_critical=False,
        mention_additional_instruction="Check the answer for a statement that compensation is at GS-04 (GS-4) Step 1 base pay. Minor variants (GS-4) are acceptable.",
        support_additional_instruction="Verify on the cited webpage(s) that the internship compensation is described as GS-04 (GS-4), Step 1 base pay level. Accept equivalent phrasing.",
    )

    return evaluator.get_summary()