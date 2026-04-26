import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "nace_outcomes_2024"
TASK_DESCRIPTION = """
I am conducting research on university career services effectiveness and need to compare institutions that maintain high-quality career outcomes reporting. Find four U.S. universities that publicly report their career outcomes data following NACE (National Association of Colleges and Employers) First-Destination Survey standards, with a knowledge rate of at least 84% for their Class of 2024 graduates.

For each of the four universities, provide the following information:
1. The name of the university
2. The URL to their official career outcomes reporting page where the knowledge rate and outcomes data are displayed
3. Their reported knowledge rate percentage for the Class of 2024
4. Their reported positive career outcomes rate (the percentage of graduates who are employed, continuing education, in military service, or in volunteer service)
5. Confirmation that their measurement timeframe aligns with NACE standards (outcomes measured within 6-9 months of graduation)

Note: The "knowledge rate" is defined as the percentage of graduates for which the institution has reasonable and verifiable information about their post-graduation career outcomes, collected from NACE-defined reputable sources such as surveys, LinkedIn, employers, and the National Student Clearinghouse.
"""


class UniversityEntry(BaseModel):
    name: Optional[str] = None
    page_url: Optional[str] = None
    class_year: Optional[str] = None
    knowledge_rate: Optional[str] = None
    positive_outcomes_rate: Optional[str] = None
    timeframe_text: Optional[str] = None


class UniversityList(BaseModel):
    universities: List[UniversityEntry] = Field(default_factory=list)


def prompt_extract_universities() -> str:
    return """
    Extract up to four U.S. universities described in the answer that publicly report their career outcomes for the Class of 2024.

    For each university, extract the following fields:
    - name: The university's official name exactly as given in the answer.
    - page_url: The URL to the official career outcomes reporting page where knowledge rate and outcomes data are displayed. This should be a valid URL explicitly present in the answer.
    - class_year: The class year that the reported data correspond to (e.g., "Class of 2024", "2024").
    - knowledge_rate: The knowledge rate percentage for the Class of 2024 as stated in the answer (e.g., "86%", "86.5%"). If not provided, set to null.
    - positive_outcomes_rate: The positive career outcomes rate percentage for the Class of 2024 as stated in the answer (e.g., "92%", "92.1%"). If not provided, set to null.
    - timeframe_text: Any text in the answer describing the measurement timeframe (e.g., "6 months after graduation", "within 6-9 months", "by December"). If not provided, set to null.

    Rules:
    1. Only include universities for which the answer provides an explicit URL to the outcomes page.
    2. Extract only what is explicitly stated in the answer; do not infer values.
    3. Prefer entries where the class year is 2024; if multiple years are present, select Class of 2024 entries first.
    4. Return at most four entries in the 'universities' array. If more than four are present, include only the first four.
    5. Ensure URLs are valid and complete; if missing protocol, prepend http://

    Return a JSON object with a single key 'universities' that contains an array of objects with the specified fields.
    """


def ordinal_name(index: int) -> str:
    mapping = ["First", "Second", "Third", "Fourth"]
    return mapping[index] if 0 <= index < len(mapping) else f"#{index + 1}"


async def verify_university(
    evaluator: Evaluator,
    parent_node,
    entry: UniversityEntry,
    idx: int,
) -> None:
    uni_node = evaluator.add_parallel(
        id=f"University_{idx + 1}",
        desc=f"{ordinal_name(idx)} university with NACE-compliant career outcomes reporting",
        parent=parent_node,
        critical=False
    )

    # Name provided (existence check)
    evaluator.add_custom_node(
        result=(entry.name is not None and entry.name.strip() != ""),
        id=f"U{idx + 1}_Name",
        desc="University name is provided",
        parent=uni_node,
        critical=True
    )

    # URL provided (existence check)
    evaluator.add_custom_node(
        result=(entry.page_url is not None and entry.page_url.strip() != ""),
        id=f"U{idx + 1}_URL_provided",
        desc="URL to official career outcomes page is provided",
        parent=uni_node,
        critical=True
    )

    # URL accessible and relevant
    url_leaf = evaluator.add_leaf(
        id=f"U{idx + 1}_URL",
        desc="URL to official career outcomes page is provided and accessible",
        parent=uni_node,
        critical=True
    )
    url_claim = (
        f"This webpage is the institution's official career outcomes or first-destination report page"
        f" that displays outcomes metrics (e.g., knowledge rate, career outcomes rate) for the Class of 2024."
    )
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=entry.page_url,
        additional_instruction=(
            "Confirm the page is hosted on the institution's official domain (often .edu or an official subdomain) "
            "and that it shows career outcomes data. Look for terms like 'First Destination', 'Career Outcomes', "
            "'Knowledge Rate', or 'NACE'. The page must be accessible; if content retrieval fails, mark as not supported."
        )
    )

    # Knowledge rate >= 84% for Class of 2024
    kr_leaf = evaluator.add_leaf(
        id=f"U{idx + 1}_Knowledge_Rate",
        desc="Knowledge rate of at least 84% is reported for Class of 2024",
        parent=uni_node,
        critical=True
    )
    kr_value_text = entry.knowledge_rate or "a knowledge rate"
    kr_claim = (
        f"The page reports {kr_value_text} for the Class of 2024 and the knowledge rate is at least 84%."
    )
    await evaluator.verify(
        claim=kr_claim,
        node=kr_leaf,
        sources=entry.page_url,
        additional_instruction=(
            "Verify that the page explicitly states a 'knowledge rate' for Class of 2024 and that the number is >= 84%. "
            "Allow minor rounding differences. Do not confuse 'response rate' with 'knowledge rate'."
        )
    )

    # Positive career outcomes rate reported for Class of 2024
    po_leaf = evaluator.add_leaf(
        id=f"U{idx + 1}_Positive_Outcomes",
        desc="Positive career outcomes rate is reported for Class of 2024",
        parent=uni_node,
        critical=True
    )
    po_value_text = entry.positive_outcomes_rate or "a positive career outcomes rate"
    po_claim = (
        f"The page reports {po_value_text} for the Class of 2024, defined as the percentage of graduates who are "
        f"employed, continuing education, in military service, or in volunteer service."
    )
    await evaluator.verify(
        claim=po_claim,
        node=po_leaf,
        sources=entry.page_url,
        additional_instruction=(
            "Look for 'Career Outcomes Rate', 'Positive Outcomes', or similar metric that aggregates employed, continuing "
            "education, military service, or volunteer service for the Class of 2024."
        )
    )

    # Timeframe alignment with NACE (6–9 months after graduation)
    tf_leaf = evaluator.add_leaf(
        id=f"U{idx + 1}_Timeframe",
        desc="Measurement timeframe is within 6-9 months of graduation per NACE standards",
        parent=uni_node,
        critical=True
    )
    tf_text = entry.timeframe_text or "a NACE-aligned timeframe"
    tf_claim = (
        f"The outcomes measurement timeframe used in this report ({tf_text}) is within 6 to 9 months after graduation, "
        f"consistent with NACE First-Destination Survey standards, for the Class of 2024."
    )
    await evaluator.verify(
        claim=tf_claim,
        node=tf_leaf,
        sources=entry.page_url,
        additional_instruction=(
            "Check for language such as 'within 6 months', '6-9 months after graduation', 'by December after May graduation', "
            "'NACE standards', or 'First Destination Survey methodology'. The timeframe must be within 6 to 9 months of graduation."
        )
    )


async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversityList,
        extraction_name="universities_extraction"
    )

    # Ensure exactly 4 entries by truncating or padding
    universities = (extracted.universities or [])[:4]
    while len(universities) < 4:
        universities.append(UniversityEntry())

    # Build verification tree per university
    for i in range(4):
        await verify_university(evaluator, root, universities[i], i)

    # Add custom info summary
    evaluator.add_custom_info(
        info={
            "requested_universities": 4,
            "extracted_count": len(extracted.universities) if extracted.universities else 0,
            "evaluation_focus": [
                "Name presence",
                "Official outcomes page accessibility",
                "Knowledge rate >= 84% (Class of 2024)",
                "Positive career outcomes rate reported (Class of 2024)",
                "NACE timeframe alignment (6–9 months)"
            ]
        },
        info_type="evaluation_meta",
        info_name="task_meta"
    )

    return evaluator.get_summary()