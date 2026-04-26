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
TASK_ID = "andrew_ng_lineage_mgp_5gen"
TASK_DESCRIPTION = (
    "Starting with Professor Andrew Ng, who is currently an adjunct professor in the Computer Science Department "
    "at Stanford University, trace his academic lineage by identifying his PhD advisor and continuing backward "
    "through four additional generations of advisors (5 generations total). For each of the five advisors in this "
    "lineage chain, provide the following information: (1) The advisor's full name as it appears in the Mathematics "
    "Genealogy Project, (2) The university where the advisor obtained their PhD degree, (3) The year the advisor "
    "graduated with their PhD, and (4) A direct URL link to the advisor's page on the Mathematics Genealogy Project. "
    "Present your findings as a sequential chain, clearly indicating which generation each advisor represents "
    "(1st generation = Andrew Ng's direct PhD advisor, 2nd generation = that advisor's PhD advisor, and so on through the 5th generation)."
)

START_PERSON_NAME = "Andrew Ng"


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class GenAdvisor(BaseModel):
    # A single advisor record as provided in the answer (in sequence order)
    generation_label: Optional[str] = None  # e.g., "1st generation", "Gen 1", "Generation 1"
    advisor_full_name: Optional[str] = None  # exact as MGP
    phd_university: Optional[str] = None
    phd_year: Optional[str] = None  # keep as string to be robust to formats
    advisor_mgp_url: Optional[str] = None  # direct URL to advisor's MGP page
    student_name: Optional[str] = None  # the student for whom this person is advisor (optional in answer)
    student_mgp_url: Optional[str] = None  # direct URL to the student's MGP page (optional in answer)


class LineageExtraction(BaseModel):
    # All advisors listed in the answer, in the exact order they appear as generations
    advisors: List[GenAdvisor] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_lineage() -> str:
    return """
    Extract the advisor lineage chain described in the answer as an ordered list named 'advisors'.
    The list should follow the exact order presented in the answer (e.g., 1st generation first, then 2nd, ...).
    For each advisor in the chain, extract the following fields:
    - generation_label: The explicit generation label used in the answer text for this advisor (e.g., "1st generation", "Generation 1", "Gen 1"). If not explicitly labeled, set to null.
    - advisor_full_name: The advisor’s full name as written in the answer (intended to match the name as it appears on the Mathematics Genealogy Project (MGP) page).
    - phd_university: The university where the advisor received their PhD (as stated in the answer).
    - phd_year: The year the advisor received their PhD (as stated in the answer). Keep the value as a string exactly as it appears.
    - advisor_mgp_url: A direct URL link to this advisor’s MGP page. If not provided, set to null.
    - student_name: The name of the student whom this person advised (if explicitly provided in the answer). If not explicitly provided, set to null.
    - student_mgp_url: A direct URL link to that student's MGP page (if provided in the answer). If not provided, set to null.

    SPECIAL RULES:
    - Extract only what is explicitly present in the answer. Do not invent or infer fields.
    - For URLs, extract only valid URLs that appear in the answer. If a URL is missing a protocol, prepend http://
    - If the answer lists more than 5 advisors (more than the requested 5 generations), extract them all; we will only use the first 5 for evaluation.
    - If the answer lists fewer than 5 advisors, still extract what is present.

    Return a JSON object strictly matching this schema:
    {
      "advisors": [
        {
          "generation_label": string | null,
          "advisor_full_name": string | null,
          "phd_university": string | null,
          "phd_year": string | null,
          "advisor_mgp_url": string | null,
          "student_name": string | null,
          "student_mgp_url": string | null
        },
        ...
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    # Return ordinal string: 1 -> 1st, 2 -> 2nd, etc.
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def safe_sources(*urls: Optional[str]) -> List[str]:
    # Filter out Nones and empty strings
    return [u for u in urls if isinstance(u, str) and u.strip()]


def compute_student_context(
    extracted: LineageExtraction,
    gen_index: int
) -> Dict[str, Optional[str]]:
    """
    Compute the expected student context for a given generation index (1-based index).
    Returns dict with 'student_name' and 'student_mgp_url' to be used in claims.
    Fallbacks:
      - Gen 1: student is Andrew Ng (START_PERSON_NAME), use extracted[0].student_mgp_url if provided, else None.
      - Gen >1: student is previous generation's advisor_full_name; student's MGP url is previous generation's advisor_mgp_url.
    """
    idx0 = gen_index - 1
    if gen_index == 1:
        student_name = START_PERSON_NAME
        student_mgp_url = None
        if idx0 < len(extracted.advisors):
            student_mgp_url = extracted.advisors[idx0].student_mgp_url or None
        return {"student_name": student_name, "student_mgp_url": student_mgp_url}

    # For gen > 1, use previous generation's advisor as the "student" for this generation
    prev_idx0 = idx0 - 1
    if 0 <= prev_idx0 < len(extracted.advisors):
        student_name = extracted.advisors[prev_idx0].advisor_full_name
        student_mgp_url = extracted.advisors[prev_idx0].advisor_mgp_url
        return {"student_name": student_name, "student_mgp_url": student_mgp_url}
    return {"student_name": None, "student_mgp_url": None}


def get_generation_item(extracted: LineageExtraction, gen_index: int) -> GenAdvisor:
    idx0 = gen_index - 1
    if 0 <= idx0 < len(extracted.advisors):
        return extracted.advisors[idx0]
    return GenAdvisor()


# --------------------------------------------------------------------------- #
# Verification logic for each generation                                      #
# --------------------------------------------------------------------------- #
async def verify_generation(
    evaluator: Evaluator,
    parent_node,
    gen_index: int,
    extracted: LineageExtraction
) -> None:
    """
    Build and verify all leaves for a specific generation according to the rubric.
    gen_index is 1-based (1..5).
    """
    ordinal_label = ordinal(gen_index)
    gen_item = get_generation_item(extracted, gen_index)

    # Generation container node (parallel checks within this generation)
    gen_node = evaluator.add_parallel(
        id=f"Generation_{gen_index}_Advisor",
        desc=f"{gen_index}{'st' if gen_index == 1 else ('nd' if gen_index == 2 else ('rd' if gen_index == 3 else 'th'))} generation advisor verification",
        parent=parent_node,
        critical=False
    )

    # 1) Generation label present (critical leaf)
    label_present = bool(gen_item.generation_label and gen_item.generation_label.strip())
    evaluator.add_custom_node(
        result=label_present,
        id=f"Gen{gen_index}_Generation_Label_Present",
        desc=f"Clearly label this advisor as {ordinal_label} generation.",
        parent=gen_node,
        critical=True
    )

    # Compute student context (name + student's MGP page)
    student_ctx = compute_student_context(extracted, gen_index)
    student_name = student_ctx.get("student_name")
    student_mgp_url = student_ctx.get("student_mgp_url")

    # 2) Advisor linkage per MGP (critical leaf) - verify the advisor-student relationship
    linkage_node = evaluator.add_leaf(
        id=f"Gen{gen_index}_Advisor_Linkage_Per_MGP",
        desc="Advisor is correctly identified as the direct PhD advisor on the student's MGP page (primary if multiple).",
        parent=gen_node,
        critical=True
    )
    advisor_name_for_claim = gen_item.advisor_full_name or "the advisor"
    student_name_for_claim = student_name or "the student"
    claim_linkage = (
        f"According to the Mathematics Genealogy Project, {advisor_name_for_claim} is the direct PhD advisor of "
        f"{student_name_for_claim}. When multiple advisors are listed, treat the primary advisor/committee chair as the main advisor."
    )
    sources_linkage = safe_sources(student_mgp_url, gen_item.advisor_mgp_url)
    await evaluator.verify(
        claim=claim_linkage,
        node=linkage_node,
        sources=sources_linkage,  # ideally includes the student's page; advisor page acceptable if it lists the student
        additional_instruction=(
            "Prefer confirming on the student's MGP page under 'Advisor(s)'. If the student's page is unavailable, "
            "the advisor's MGP page listing the student under 'Students' also supports the claim. Allow minor variations in name formatting."
        ),
    )

    # 3) Advisor page reachable from student page (critical leaf)
    reachable_node = evaluator.add_leaf(
        id=f"Gen{gen_index}_Advisor_Page_Reachable_From_Student_Page",
        desc="The advisor has an accessible MGP entry reachable via the advisor link from the student's MGP page.",
        parent=gen_node,
        critical=True
    )
    claim_reachable = (
        f"From the student's Mathematics Genealogy Project page for {student_name_for_claim}, "
        f"there is an advisor link pointing to the MGP page of {advisor_name_for_claim}"
        + (f" at {gen_item.advisor_mgp_url}." if gen_item.advisor_mgp_url else ".")
    )
    sources_reachable = safe_sources(student_mgp_url, gen_item.advisor_mgp_url)
    await evaluator.verify(
        claim=claim_reachable,
        node=reachable_node,
        sources=sources_reachable,
        additional_instruction=(
            "Check that on the student's MGP page the advisor's name appears as a clickable link that leads to the advisor's MGP page. "
            "Use the screenshot if needed to confirm links. If both pages are provided, ensure they correspond to the same person."
        ),
    )

    # 4) Advisor full name exactly as MGP (critical leaf)
    name_exact_node = evaluator.add_leaf(
        id=f"Gen{gen_index}_Advisor_Full_Name_As_MGP",
        desc="Provide the advisor’s full name exactly as it appears on the advisor’s MGP page.",
        parent=gen_node,
        critical=True
    )
    claim_name_exact = (
        f"On the advisor’s MGP page, the displayed name matches exactly (allowing minor punctuation/diacritics/casing) "
        f"the extracted name: '{gen_item.advisor_full_name}'."
    )
    await evaluator.verify(
        claim=claim_name_exact,
        node=name_exact_node,
        sources=gen_item.advisor_mgp_url,
        additional_instruction=(
            "Compare the extracted advisor name with the header/name shown on the MGP page. Allow minor punctuation, "
            "diacritics, abbreviations (e.g., middle initials), and case-insensitive matches."
        ),
    )

    # 5) PhD University (critical leaf)
    uni_node = evaluator.add_leaf(
        id=f"Gen{gen_index}_PhD_University",
        desc="Provide the university where this advisor obtained their PhD (as shown on MGP).",
        parent=gen_node,
        critical=True
    )
    claim_uni = (
        f"On the advisor’s MGP page, the Ph.D. awarding institution (university) matches the extracted value: "
        f"'{gen_item.phd_university}'."
    )
    await evaluator.verify(
        claim=claim_uni,
        node=uni_node,
        sources=gen_item.advisor_mgp_url,
        additional_instruction=(
            "Look for the thesis/degree information on the MGP page and confirm that the university matches the extracted value "
            "or an equivalent official naming variant."
        ),
    )

    # 6) PhD Year (critical leaf)
    year_node = evaluator.add_leaf(
        id=f"Gen{gen_index}_PhD_Year",
        desc="Provide the year this advisor received their PhD (as shown on MGP).",
        parent=gen_node,
        critical=True
    )
    claim_year = (
        f"On the advisor’s MGP page, the Ph.D. year matches the extracted value: '{gen_item.phd_year}'. "
        "Allow minor variations like year ranges only if the page indicates uncertainty for the year."
    )
    await evaluator.verify(
        claim=claim_year,
        node=year_node,
        sources=gen_item.advisor_mgp_url,
        additional_instruction="Verify the degree year shown on the MGP page matches the extracted year.",
    )

    # 7) Direct MGP URL provided and correct (critical leaf)
    url_node = evaluator.add_leaf(
        id=f"Gen{gen_index}_MGP_Direct_URL",
        desc="Provide a direct URL to this advisor’s MGP page.",
        parent=gen_node,
        critical=True
    )
    if gen_item.advisor_mgp_url:
        claim_url = (
            f"The provided URL {gen_item.advisor_mgp_url} is a direct Mathematics Genealogy Project page for "
            f"{advisor_name_for_claim} (i.e., it is the person's profile page on MGP)."
        )
        await evaluator.verify(
            claim=claim_url,
            node=url_node,
            sources=gen_item.advisor_mgp_url,
            additional_instruction=(
                "Confirm that the URL leads to the Mathematics Genealogy Project and corresponds to the advisor's profile page."
            ),
        )
    else:
        # If no URL provided, verify (and fail) via answer-only claim
        claim_url_missing = (
            "A direct URL to the advisor’s MGP page has been provided in the answer for this generation."
        )
        await evaluator.verify(
            claim=claim_url_missing,
            node=url_node,
            sources=None,
            additional_instruction=(
                "Check the answer text itself: if there is no URL provided for this advisor's MGP page, this claim should be judged incorrect."
            ),
        )

    # 8) For Generation 5 only: No additional generations beyond 5 (critical leaf)
    if gen_index == 5:
        no_extra = len(extracted.advisors) <= 5
        evaluator.add_custom_node(
            result=no_extra,
            id="No_Additional_Generations_Beyond_5",
            desc="Do not include any additional advisor generations beyond the 5th generation (task requires exactly 5 generations).",
            parent=gen_node,
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
    Evaluate an answer for tracing Andrew Ng's academic lineage via MGP across 5 generations.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Create the main task node (sequential chain as per rubric)
    lineage_root = evaluator.add_sequential(
        id="Complete_Academic_Lineage",
        desc="Trace 5 generations of PhD advisors from Andrew Ng using MGP entries, verifying all required fields.",
        parent=root,
        critical=False  # Set non-critical to allow leaf-level critical enforcement; avoids critical-parent constraint.
    )

    # Extract lineage data from answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_lineage(),
        template_class=LineageExtraction,
        extraction_name="lineage_extraction"
    )

    # Record a bit of custom info (counts)
    evaluator.add_custom_info(
        info={"total_advisors_listed_in_answer": len(extraction.advisors)},
        info_type="stats",
        info_name="extracted_counts"
    )

    # Verify each of the first 5 generations in order (pad gracefully if fewer)
    for gen_idx in range(1, 6):
        await verify_generation(evaluator, lineage_root, gen_idx, extraction)

    # Return structured result
    return evaluator.get_summary()