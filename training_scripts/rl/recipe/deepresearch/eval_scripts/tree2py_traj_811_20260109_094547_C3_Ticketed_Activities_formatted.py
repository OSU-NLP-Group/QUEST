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
TASK_ID = "broadway_largest_theater"
TASK_DESCRIPTION = (
    "Identify Broadway's largest theater by total seating capacity. For this theater, provide the following information: "
    "(1) The name of the theater, (2) Its total seating capacity that establishes it as the largest Broadway theater, "
    "(3) A reference URL that verifies the theater's capacity information, "
    "(4) The seating capacity of the Orchestra section, and "
    "(5) The minimum age requirement for admission to this theater."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TheaterExtraction(BaseModel):
    theater_name: Optional[str] = None
    total_capacity: Optional[str] = None
    reference_url: Optional[str] = None
    orchestra_capacity: Optional[str] = None
    minimum_admission_age: Optional[str] = None
    other_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theater_info() -> str:
    return """
    Extract the information the answer provides about Broadway's largest theater by total seating capacity.

    Return a JSON object with the following fields:
    - theater_name: The name of the theater identified as the largest on Broadway by total seating capacity.
    - total_capacity: The total seating capacity value stated for that theater (as shown in the answer; keep formatting like commas).
    - reference_url: A single URL explicitly cited in the answer that verifies the theater's total seating capacity (prefer the most relevant or authoritative one). If multiple are present, pick the one most clearly tied to verifying capacity. If none, set to null.
    - orchestra_capacity: The seating capacity of the Orchestra section (as stated in the answer). If not provided, set to null.
    - minimum_admission_age: The minimum age requirement for admission to this theater (as stated in the answer). If the answer uses a policy phrasing like "no children under 4" or "ages 5+ only", extract that exact phrasing (e.g., "4+" or "no children under 4"). If not provided, set to null.
    - other_urls: An array of any additional URLs mentioned in the answer aside from the main reference_url. If none, return an empty array.

    Rules:
    - Extract only what is explicitly present in the answer.
    - For URLs, extract valid, complete URLs (include http/https). If a URL is missing a protocol, prepend http://.
    - Do not invent or infer information.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def collect_all_sources(info: TheaterExtraction) -> List[str]:
    urls: List[str] = []
    if info.reference_url:
        urls.append(info.reference_url)
    if info.other_urls:
        urls.extend([u for u in info.other_urls if isinstance(u, str) and u.strip() != ""])
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: TheaterExtraction) -> None:
    # Node group: Identify the theater as the largest by total capacity
    ident_node = evaluator.add_parallel(
        id="theater_identification_and_verification",
        desc="Identify the theater that is Broadway's largest by total seating capacity",
        parent=evaluator.root,
        critical=True
    )

    # Leaf: theater_name -> verify that this named theater is indeed the largest Broadway theater by total capacity
    theater_name_leaf = evaluator.add_leaf(
        id="theater_name",
        desc="Provide the name of Broadway's largest theater by total seating capacity",
        parent=ident_node,
        critical=True
    )
    theater_name = extracted.theater_name or ""
    name_claim_sources = collect_all_sources(extracted)
    await evaluator.verify(
        claim=f"{theater_name} is the largest Broadway theater by total seating capacity.",
        node=theater_name_leaf,
        sources=name_claim_sources if name_claim_sources else None,
        additional_instruction=(
            "Verify that the provided sources explicitly state or strongly imply that this theater has the largest "
            "seat count on Broadway (by total seating capacity). Accept minor wording variations such as "
            "'largest Broadway theater' or 'largest seating capacity on Broadway'."
        ),
    )

    # Leaf: total_capacity -> ensure a capacity value was provided (existence/format presence check)
    capacity_str = (extracted.total_capacity or "").strip()
    capacity_exists = bool(capacity_str)
    evaluator.add_custom_node(
        result=capacity_exists,
        id="total_capacity",
        desc="Provide the theater's total seating capacity (the value used to establish it as the largest)",
        parent=ident_node,
        critical=True
    )

    # Node group: Provide additional details and verify with sources
    details_node = evaluator.add_parallel(
        id="theater_details",
        desc="Provide the required additional information about the identified theater",
        parent=evaluator.root,
        critical=True
    )

    # Leaf: reference_url -> verify that the source supports the total capacity claim
    ref_leaf = evaluator.add_leaf(
        id="reference_url",
        desc="Provide a reference URL that verifies the theater's total seating capacity",
        parent=details_node,
        critical=True
    )
    theater_for_text = theater_name if theater_name else "the theater"
    await evaluator.verify(
        claim=f"The total seating capacity of {theater_for_text} is {capacity_str}.",
        node=ref_leaf,
        sources=name_claim_sources if name_claim_sources else None,
        additional_instruction=(
            "Verify that the page explicitly states the theater's total seating capacity equals the provided value. "
            "Allow minor formatting differences (e.g., commas in numbers) and phrasing like 'seats' or 'seat capacity'. "
            "If the page indicates an 'approximately' or 'about' value that reasonably matches, treat as correct."
        ),
    )

    # Leaf: orchestra_section_capacity -> verify orchestra capacity using available sources
    orch_leaf = evaluator.add_leaf(
        id="orchestra_section_capacity",
        desc="State the seating capacity of the Orchestra section",
        parent=details_node,
        critical=True
    )
    orchestra_str = (extracted.orchestra_capacity or "").strip()
    await evaluator.verify(
        claim=f"The seating capacity of the Orchestra section at {theater_for_text} is {orchestra_str}.",
        node=orch_leaf,
        sources=name_claim_sources if name_claim_sources else None,
        additional_instruction=(
            "Confirm the Orchestra section's seat count from the provided sources. "
            "The evidence could be a capacity number for the Orchestra specifically, or a clear breakdown/chart "
            "that states the Orchestra seat count. If the page uses approximate phrasing or ranges but clearly "
            "corresponds to the stated value, treat as supported. If the value is missing or unverifiable, mark as unsupported."
        ),
    )

    # Leaf: minimum_admission_age -> verify age policy using available sources
    age_leaf = evaluator.add_leaf(
        id="minimum_admission_age",
        desc="State the minimum age requirement for admission to this theater",
        parent=details_node,
        critical=True
    )
    age_str = (extracted.minimum_admission_age or "").strip()
    await evaluator.verify(
        claim=f"The minimum age requirement for admission to {theater_for_text} is {age_str}.",
        node=age_leaf,
        sources=name_claim_sources if name_claim_sources else None,
        additional_instruction=(
            "Verify the admission age policy as stated on the referenced page(s). "
            "Interpret statements like 'no children under 4' as a minimum age of 4. "
            "Some pages describe production-specific policies; if the page clearly states the age policy for "
            "this theater or its resident production, treat it as valid. If the policy is not present on the page(s), mark unsupported."
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
    """
    Evaluate an answer for the Broadway largest theater task.
    """
    # Initialize evaluator with a sequential root to reflect the two-step nature:
    # (1) Identify largest theater with capacity info, then (2) verify details by sources.
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify Broadway's largest theater by total seating capacity and provide required capacity/section/age details with a supporting source",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured fields from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_theater_info(),
        template_class=TheaterExtraction,
        extraction_name="theater_info",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()