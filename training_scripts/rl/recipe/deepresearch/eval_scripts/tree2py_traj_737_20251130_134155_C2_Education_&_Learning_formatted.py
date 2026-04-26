import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "csun_seton_spring_2026_credentials"
TASK_DESCRIPTION = """
I am a prospective teacher planning to pursue a teaching credential program starting in Spring 2026. I am comparing California State University, Northridge (CSUN) and Seton Hall University as potential options. For my application planning, I need to know: (1) What is the application deadline for CSUN's credential programs for Spring 2026 admission? (2) When do CSUN's Spring 2026 credential program applications become available? (3) What is the application deadline for Seton Hall University's graduate education programs for spring semester admission? (4) What is the minimum GPA requirement for admission to Seton Hall's elementary/special education and secondary education programs? Please provide official source URLs to support each piece of information.
"""

# Ground truth expectations encoded in the rubric
EXPECTED = {
    "CSUN": {
        "deadline": "December 1, 2025",
        "availability": "September 22, 2025",
        "official_domains": ["csun.edu", "calstate.edu"],
    },
    "SHU": {
        "deadline": "December 1",
        "min_gpa": "3.0",
        "official_domains": ["shu.edu"],
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CSUNFields(BaseModel):
    deadline_text: Optional[str] = None
    deadline_urls: List[str] = Field(default_factory=list)
    availability_text: Optional[str] = None
    availability_urls: List[str] = Field(default_factory=list)


class SHUFields(BaseModel):
    deadline_text: Optional[str] = None
    deadline_urls: List[str] = Field(default_factory=list)
    min_gpa_text: Optional[str] = None
    gpa_urls: List[str] = Field(default_factory=list)


class ApplicationExtraction(BaseModel):
    csun: Optional[CSUNFields] = None
    shu: Optional[SHUFields] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_application_info() -> str:
    return """
Extract from the answer the specific requested details for CSUN and Seton Hall, and the explicit official source URLs provided for each piece of information.

Return a JSON object with the following structure:
{
  "csun": {
    "deadline_text": string or null,              // The exact phrasing used in the answer for the CSUN Spring 2026 credential application deadline
    "deadline_urls": [urls...],                   // All official URLs cited that support the CSUN Spring 2026 credential deadline; only include URLs explicitly present in the answer
    "availability_text": string or null,          // The exact phrasing used in the answer for when CSUN Spring 2026 credential applications become available
    "availability_urls": [urls...]                // All official URLs cited that support the availability/open date
  },
  "shu": {
    "deadline_text": string or null,              // The exact phrasing used in the answer for Seton Hall's spring semester graduate education application deadline
    "deadline_urls": [urls...],                   // All official URLs cited that support Seton Hall's spring deadline
    "min_gpa_text": string or null,               // The exact phrasing used in the answer for the minimum GPA requirement for elementary/special education and secondary education programs
    "gpa_urls": [urls...]                         // All official URLs cited that support the minimum GPA requirement
  }
}

Important instructions:
- Extract ONLY what the answer explicitly states. Do not infer or invent values.
- For URLs: include only valid, explicit URLs present in the answer. Accept markdown links by extracting the URL target. If a URL is missing protocol, prepend http://.
- "Official" sources should correspond to university domains (CSUN: csun.edu or calstate.edu; Seton Hall: shu.edu). However, still extract any URLs the answer cites; do not filter them out here.
- If any field is missing in the answer, return null for the text field and an empty array for the corresponding URL list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _list_or_empty(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


def _domain_policy_text(university: str, domains: List[str]) -> str:
    dlist = ", ".join([f"'{d}'" for d in domains])
    return (
        f"Treat a URL as official for {university} ONLY if its domain ends with one of: {dlist}. "
        f"If none of the provided URLs are from these official domains, judge the claim as NOT supported."
    )


def _empty_sources_fail_instruction() -> str:
    return "If no URL(s) are provided for this check, you MUST judge the claim as NOT supported."


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_csun_verification(evaluator: Evaluator, parent_node, extracted: ApplicationExtraction) -> None:
    csun_node = evaluator.add_parallel(
        id="CSUN_Requirements",
        desc="Provide CSUN credential program Spring 2026 deadline, availability date, and official URLs supporting each",
        parent=parent_node,
        critical=True,
    )

    csun_info = extracted.csun or CSUNFields()

    # CSUN Deadline Item (Sequential)
    csun_deadline_item = evaluator.add_sequential(
        id="CSUN_Deadline_Item",
        desc="CSUN Spring 2026 credential program application deadline and its official source URL",
        parent=csun_node,
        critical=True,
    )

    # Value leaf: must state exact expected date
    csun_deadline_value = evaluator.add_leaf(
        id="CSUN_Deadline_Value",
        desc="States the CSUN credential program application deadline for Spring 2026 admission as December 1, 2025",
        parent=csun_deadline_item,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the application deadline for CSUN's credential programs for Spring 2026 admission is stated as 'December 1, 2025'.",
        node=csun_deadline_value,
        additional_instruction="Allow minor formatting variations such as 'Dec 1, 2025' or 'Dec. 1, 2025'. "
                               "However, the answer must explicitly associate this date with CSUN credential programs and Spring 2026 admission."
    )

    # Source leaf: verify by URL(s)
    csun_deadline_source = evaluator.add_leaf(
        id="CSUN_Deadline_Source",
        desc="Provides an official CSUN URL that supports the stated Spring 2026 credential application deadline",
        parent=csun_deadline_item,
        critical=True,
    )
    csun_deadline_urls = _list_or_empty(csun_info.deadline_urls)
    await evaluator.verify(
        claim="The webpage confirms that the application deadline for CSUN's credential program for Spring 2026 admission is December 1, 2025.",
        node=csun_deadline_source,
        sources=csun_deadline_urls,
        additional_instruction=(
            f"{_empty_sources_fail_instruction()} "
            f"{_domain_policy_text('CSUN', EXPECTED['CSUN']['official_domains'])} "
            "Only pass if the page explicitly indicates (or strongly implies in an official timeline table) that the Spring 2026 CSUN credential program application deadline is December 1, 2025. "
            "If multiple deadlines are shown, ensure the one relevant to CSUN credential (teacher credential) Spring 2026 is December 1, 2025."
        ),
    )

    # CSUN Availability Item (Sequential)
    csun_avail_item = evaluator.add_sequential(
        id="CSUN_Availability_Item",
        desc="CSUN Spring 2026 credential applications availability date and its official source URL",
        parent=csun_node,
        critical=True,
    )

    # Value leaf
    csun_avail_value = evaluator.add_leaf(
        id="CSUN_Availability_Value",
        desc="States that CSUN Spring 2026 credential program applications become available on September 22, 2025",
        parent=csun_avail_item,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, CSUN Spring 2026 credential program applications become available on 'September 22, 2025'.",
        node=csun_avail_value,
        additional_instruction="Allow minor formatting variations such as 'Sept. 22, 2025'. "
                               "The answer must explicitly link this date to when CSUN Spring 2026 credential applications open/become available."
    )

    # Source leaf
    csun_avail_source = evaluator.add_leaf(
        id="CSUN_Availability_Source",
        desc="Provides an official CSUN URL that supports the stated application availability date",
        parent=csun_avail_item,
        critical=True,
    )
    csun_availability_urls = _list_or_empty(csun_info.availability_urls)
    await evaluator.verify(
        claim="The webpage states that CSUN Spring 2026 credential program applications open or become available on September 22, 2025.",
        node=csun_avail_source,
        sources=csun_availability_urls,
        additional_instruction=(
            f"{_empty_sources_fail_instruction()} "
            f"{_domain_policy_text('CSUN', EXPECTED['CSUN']['official_domains'])} "
            "Pass only if the official page explicitly indicates the opening/availability date as September 22, 2025 for CSUN Spring 2026 credential applications. "
            "Accept synonyms like 'open', 'become available', or 'applications available'."
        ),
    )


async def build_shu_verification(evaluator: Evaluator, parent_node, extracted: ApplicationExtraction) -> None:
    shu_node = evaluator.add_parallel(
        id="Seton_Hall_Requirements",
        desc="Provide Seton Hall spring semester graduate education deadline, minimum GPA requirement, and official URLs supporting each",
        parent=parent_node,
        critical=True,
    )

    shu_info = extracted.shu or SHUFields()

    # SHU Deadline Item (Sequential)
    shu_deadline_item = evaluator.add_sequential(
        id="SHU_Deadline_Item",
        desc="Seton Hall spring semester graduate education application deadline and its official source URL",
        parent=shu_node,
        critical=True,
    )

    # Value leaf
    shu_deadline_value = evaluator.add_leaf(
        id="SHU_Deadline_Value",
        desc="States the Seton Hall University spring semester application deadline for graduate education programs as December 1",
        parent=shu_deadline_item,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the spring semester application deadline for Seton Hall University's graduate education programs is stated as 'December 1'.",
        node=shu_deadline_value,
        additional_instruction="Allow minor formatting variations like 'Dec 1' or 'Dec. 1'. "
                               "The answer must clearly tie 'December 1' to the spring semester and Seton Hall graduate education programs."
    )

    # Source leaf
    shu_deadline_source = evaluator.add_leaf(
        id="SHU_Deadline_Source",
        desc="Provides an official Seton Hall University URL that supports the stated spring semester graduate education application deadline",
        parent=shu_deadline_item,
        critical=True,
    )
    shu_deadline_urls = _list_or_empty(shu_info.deadline_urls)
    await evaluator.verify(
        claim="The webpage confirms that the spring semester application deadline for Seton Hall University's graduate education programs is December 1.",
        node=shu_deadline_source,
        sources=shu_deadline_urls,
        additional_instruction=(
            f"{_empty_sources_fail_instruction()} "
            f"{_domain_policy_text('Seton Hall University', EXPECTED['SHU']['official_domains'])} "
            "The page should clearly indicate the spring application deadline as December 1 for graduate programs in education. "
            "If the page lists multiple deadlines (e.g., priority vs final), accept 'December 1' only if it's clearly applicable as the spring deadline students must meet."
        ),
    )

    # SHU GPA Item (Sequential)
    shu_gpa_item = evaluator.add_sequential(
        id="SHU_GPA_Item",
        desc="Seton Hall minimum GPA requirement for elementary/special education and secondary education programs and its official source URL",
        parent=shu_node,
        critical=True,
    )

    # Value leaf
    shu_gpa_value = evaluator.add_leaf(
        id="SHU_GPA_Value",
        desc="States the minimum GPA requirement as 3.0 for admission to Seton Hall's elementary/special education and secondary education programs",
        parent=shu_gpa_item,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the minimum GPA requirement is stated as 3.0 for Seton Hall's elementary/special education and secondary education programs.",
        node=shu_gpa_value,
        additional_instruction="Allow phrasing such as 'minimum 3.0 GPA', '3.0 on a 4.0 scale', or 'at least 3.0'. "
                               "The answer must explicitly indicate that 3.0 applies to both elementary/special education and secondary education programs."
    )

    # Source leaf
    shu_gpa_source = evaluator.add_leaf(
        id="SHU_GPA_Source",
        desc="Provides an official Seton Hall University URL that supports the stated minimum GPA requirement",
        parent=shu_gpa_item,
        critical=True,
    )
    shu_gpa_urls = _list_or_empty(shu_info.gpa_urls)
    await evaluator.verify(
        claim="The webpage confirms that the minimum GPA requirement is 3.0 for admission to Seton Hall's elementary/special education and secondary education programs.",
        node=shu_gpa_source,
        sources=shu_gpa_urls,
        additional_instruction=(
            f"{_empty_sources_fail_instruction()} "
            f"{_domain_policy_text('Seton Hall University', EXPECTED['SHU']['official_domains'])} "
            "Accept a single official page covering both programs or multiple official program pages collectively indicating 3.0 as the minimum GPA. "
            "If a page describes general College of Education graduate admissions with a 3.0 minimum applicable to these programs, it is acceptable."
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
    # Initialize evaluator (root is non-critical by design)
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_application_info(),
        template_class=ApplicationExtraction,
        extraction_name="extracted_application_info",
    )

    # Add a critical top-level node to match rubric's "Root"
    rubric_root = evaluator.add_parallel(
        id="Root",
        desc="Provide CSUN and Seton Hall Spring 2026 / spring semester application details and official supporting URLs for each required piece of information",
        parent=root,
        critical=True,
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "expected": EXPECTED,
        "notes": "All items are critical. Official URLs: CSUN -> csun.edu/calstate.edu; Seton Hall -> shu.edu"
    })

    # Build subtrees for CSUN and Seton Hall
    await build_csun_verification(evaluator, rubric_root, extracted)
    await build_shu_verification(evaluator, rubric_root, extracted)

    # Return final summary
    return evaluator.get_summary()