import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sec_large_stadiums"
TASK_DESCRIPTION = (
    "A high school football recruit is interested in playing for NCAA Division I FBS programs in the "
    "Southeastern Conference (SEC) that have large stadium facilities to showcase his talents. Identify four SEC "
    "football programs whose home stadiums have a seating capacity of at least 85,000. For each program, provide "
    "the university name, the stadium name, the exact seating capacity, and a reference URL that confirms this information."
)

MIN_CAPACITY = 85_000
ORDINALS = ["First", "Second", "Third", "Fourth"]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    university_name: Optional[str] = None
    stadium_name: Optional[str] = None
    seating_capacity: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract up to four NCAA Division I FBS football programs in the Southeastern Conference (SEC) that the answer mentions,
    and for each include:
    - university_name: The university or school name for the football program.
    - stadium_name: The home football stadium name.
    - seating_capacity: The exact seating capacity value as stated in the answer (keep the original formatting, e.g., with commas).
    - reference_urls: A list of URL(s) explicitly provided in the answer that can be used to confirm the program, conference, and/or stadium capacity.
    
    Rules:
    - Preserve the order in which the programs appear in the answer; extract at most four.
    - Return null for missing text fields; return an empty array for reference_urls if no URLs are provided.
    - Only include URLs that actually appear in the answer (plain URLs or markdown links). Do not invent any URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_and_filter_urls(urls: List[str]) -> List[str]:
    """Normalize and filter URL strings; keep http/https and add http:// if missing scheme but looks like a URL."""
    cleaned: List[str] = []
    for u in urls:
        if not u:
            continue
        url = u.strip()
        if not url:
            continue

        # If missing scheme but looks like a domain, prepend http://
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
            if url.startswith("www."):
                url = "http://" + url
            elif re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/.*)?$", url):
                url = "http://" + url

        # Accept http(s) only
        if re.match(r"^https?://", url, flags=re.IGNORECASE):
            cleaned.append(url)
    # Deduplicate preserving order
    deduped: List[str] = []
    seen = set()
    for c in cleaned:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def parse_capacity_to_int(cap_str: Optional[str]) -> Optional[int]:
    """
    Parse a capacity string into an integer. 
    Heuristics:
      - handle commas and plain integers (e.g., "101,821" -> 101821)
      - handle 'k' suffix (e.g., "85k" or "85.5k")
      - if multiple numbers present, choose the largest 5+ digit number; otherwise choose the largest number found
    """
    if not cap_str:
        return None

    s = cap_str.strip().lower()
    if not s:
        return None

    candidates: List[int] = []

    # 1) Handle k/K suffix patterns like "85k", "85.5k"
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*[kK]\b", s):
        try:
            val = float(m.group(1)) * 1000
            candidates.append(int(round(val)))
        except Exception:
            pass

    # 2) Handle plain integers with or without commas
    for m in re.finditer(r"\d[\d,]*", s):
        try:
            num = int(m.group(0).replace(",", ""))
            candidates.append(num)
        except Exception:
            pass

    if not candidates:
        return None

    # Prefer 5+ digit numbers typical for stadiums; fallback to max of all candidates
    five_plus = [n for n in candidates if n >= 10000]
    return max(five_plus) if five_plus else max(candidates)


def ordinal_for_index(i: int) -> str:
    return ORDINALS[i] if 0 <= i < len(ORDINALS) else f"#{i + 1}"


# --------------------------------------------------------------------------- #
# Verification logic for each program                                         #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramItem,
    index: int,
) -> None:
    """
    Build and verify the sub-tree for a single program.
    This follows the rubric leaves exactly and uses URL-grounded checks for factual claims.
    """
    ord_label = ordinal_for_index(index)
    prog_node = evaluator.add_parallel(
        id=f"program_{index + 1}",
        desc=f"{ord_label} identified program meets all requirements",
        parent=parent_node,
        critical=False,
    )

    # Normalize URLs upfront
    refs = normalize_and_filter_urls(program.reference_urls or [])

    # 1) University name is provided (existence)
    evaluator.add_custom_node(
        result=bool(program.university_name and program.university_name.strip()),
        id=f"program_{index + 1}_university_name",
        desc="University name is provided",
        parent=prog_node,
        critical=True,
    )

    # 2) Program is NCAA Division I FBS (URL-verified; allow SEC mention to suffice)
    division_leaf = evaluator.add_leaf(
        id=f"program_{index + 1}_division",
        desc="Program is NCAA Division I FBS",
        parent=prog_node,
        critical=True,
    )
    div_claim_univ = program.university_name or "the university"
    div_claim = (
        f"The football program at {div_claim_univ} competes in NCAA Division I FBS (Football Bowl Subdivision)."
    )
    await evaluator.verify(
        claim=div_claim,
        node=division_leaf,
        sources=refs,
        additional_instruction=(
            "Treat explicit mentions of SEC (Southeastern Conference) on the page as sufficient evidence that the "
            "program competes in NCAA Division I FBS, since the SEC is an NCAA Division I FBS conference."
        ),
    )

    # 3) Program is a member of the SEC (URL-verified)
    conference_leaf = evaluator.add_leaf(
        id=f"program_{index + 1}_conference",
        desc="Program is a member of the Southeastern Conference (SEC)",
        parent=prog_node,
        critical=True,
    )
    conf_claim_univ = program.university_name or "the university"
    conf_claim = f"The football program at {conf_claim_univ} is a member of the Southeastern Conference (SEC)."
    await evaluator.verify(
        claim=conf_claim,
        node=conference_leaf,
        sources=refs,
        additional_instruction="Accept mentions such as 'SEC' or 'Southeastern Conference' as evidence of membership.",
    )

    # 4) Stadium name is provided (existence)
    evaluator.add_custom_node(
        result=bool(program.stadium_name and program.stadium_name.strip()),
        id=f"program_{index + 1}_stadium_name",
        desc="Stadium name is provided",
        parent=prog_node,
        critical=True,
    )

    # 5) Exact seating capacity value is provided (existence)
    evaluator.add_custom_node(
        result=bool(program.seating_capacity and program.seating_capacity.strip()),
        id=f"program_{index + 1}_capacity_value",
        desc="Exact seating capacity value is provided",
        parent=prog_node,
        critical=True,
    )

    # 6) Stadium seating capacity meets or exceeds 85,000 (computed check)
    cap_int = parse_capacity_to_int(program.seating_capacity)
    evaluator.add_custom_node(
        result=(cap_int is not None and cap_int >= MIN_CAPACITY),
        id=f"program_{index + 1}_capacity_threshold",
        desc=f"Stadium seating capacity meets or exceeds {MIN_CAPACITY}",
        parent=prog_node,
        critical=True,
    )

    # 7) Verifiable reference URL is provided (existence of at least one normalized http(s) URL)
    evaluator.add_custom_node(
        result=bool(refs),
        id=f"program_{index + 1}_reference",
        desc="Verifiable reference URL is provided",
        parent=prog_node,
        critical=True,
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
    Evaluate an answer for the SEC large-stadium programs task and return a structured summary.
    """
    evaluator = Evaluator()
    # NOTE: Although the JSON marks the root as critical, we relax it to non-critical to allow partial credit
    # across the four programs (framework requires all children of a critical parent to be critical).
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

    # Extract programs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction",
    )

    # Keep only the first 4 items; pad with empty if fewer
    programs: List[ProgramItem] = list(extracted.programs[:4])
    while len(programs) < 4:
        programs.append(ProgramItem())

    # Build and verify each program subtree
    verify_tasks = []
    for i in range(4):
        verify_tasks.append(verify_program(evaluator, root, programs[i], i))
    # Run verifications sequentially to preserve clearer logs; could be gathered if desired.
    for t in verify_tasks:
        await t

    # Optional: record minimal GT constraints for transparency
    evaluator.add_ground_truth(
        {
            "required_conference": "SEC",
            "required_division": "NCAA Division I FBS",
            "min_stadium_capacity": MIN_CAPACITY,
            "num_programs_required": 4,
        },
        gt_type="constraints",
    )

    return evaluator.get_summary()