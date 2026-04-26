import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wapo_west_africa_bureau_chief_identification"
TASK_DESCRIPTION = (
    "What is the full name of the journalist who works for The Washington Post as West Africa bureau chief "
    "(a position to which they were appointed in 2022), graduated from Duke University in 2017 with a BA in Political Science, "
    "and previously worked at The Washington Post's Local desk covering politics and government in Prince George's County, Maryland?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class JournalistExtraction(BaseModel):
    full_name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_journalist_info() -> str:
    return """
    Extract from the provided answer:
    1) full_name: The full name (first and last name at minimum) of the journalist that the answer claims satisfies all constraints.
       - If multiple names are mentioned, return the one the answer explicitly identifies as the journalist who meets the constraints.
       - If the name is missing or unclear, return null.
    2) source_urls: A list of all URLs explicitly mentioned in the answer as sources or references that support the identification and constraints.
       - Include URLs that appear in plain text or within markdown links.
       - Extract only valid URLs (prepend http:// if the protocol is missing).
       - Do not fabricate URLs. If no URLs are present, return an empty list.
    Return a JSON object with fields: full_name (string or null), source_urls (array of strings).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x is None:
            continue
        x = x.strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _name_looks_full(name: Optional[str]) -> bool:
    if not name:
        return False
    # Heuristic: at least two tokens and contains alphabetic characters
    tokens = [t for t in name.strip().split() if t]
    return len(tokens) >= 2 and any(ch.isalpha() for ch in name)


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_journalist_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: JournalistExtraction,
) -> None:
    """
    Build the rubric tree and perform verifications according to the given JSON rubric.
    """
    # Create the rubric's root (critical sequential node) as a child of the global root
    journalist_node = evaluator.add_sequential(
        id="Journalist_Identification",
        desc="Provide the full name of the journalist who satisfies all stated employment, appointment, education, and prior-role constraints.",
        parent=parent_node,
        critical=True
    )

    # --------------------- Name Provided (Critical) ---------------------- #
    name_ok = _name_looks_full(extracted.full_name)
    evaluator.add_custom_node(
        result=name_ok,
        id="Name_Provided",
        desc="Answer provides the journalist's full name.",
        parent=journalist_node,
        critical=True
    )

    # --------------------- Constraints Verification (Critical, Parallel) ---------------------- #
    constraints_node = evaluator.add_parallel(
        id="Constraint_Verification",
        desc="The identified journalist satisfies all stated constraints.",
        parent=journalist_node,
        critical=True
    )

    # Normalize and deduplicate sources extracted from the answer
    sources = _dedup_preserve_order(extracted.source_urls) if extracted and extracted.source_urls else None

    # Prepare common name string
    name = extracted.full_name or ""

    # Create all leaf nodes first
    works_now_node = evaluator.add_leaf(
        id="Works_For_WaPo_As_Of_Nov_2025",
        desc="Journalist works for The Washington Post as of November 2025.",
        parent=constraints_node,
        critical=True
    )

    west_africa_node = evaluator.add_leaf(
        id="Is_West_Africa_Bureau_Chief",
        desc="Journalist serves as West Africa bureau chief for The Washington Post.",
        parent=constraints_node,
        critical=True
    )

    appointed_2022_node = evaluator.add_leaf(
        id="Appointed_Bureau_Chief_In_2022",
        desc="Journalist was appointed to the West Africa bureau chief position in 2022.",
        parent=constraints_node,
        critical=True
    )

    education_node = evaluator.add_leaf(
        id="Education_Duke_2017_BA_Political_Science",
        desc="Journalist graduated from Duke University in 2017 with a BA in Political Science.",
        parent=constraints_node,
        critical=True
    )

    prior_local_node = evaluator.add_leaf(
        id="Prior_WaPo_Local_Desk_PG_County_Politics_Before_Foreign_Correspondent",
        desc="Before becoming a foreign correspondent, journalist previously worked at The Washington Post's Local desk covering politics and government in Prince George's County, Maryland.",
        parent=constraints_node,
        critical=True
    )

    # Batch verify the five constraint leaves in parallel
    claims_and_sources = [
        (
            f"As of November 2025, {name} works for The Washington Post.",
            sources,
            works_now_node,
            "Accept evidence such as an active staff profile, current bylines, or biography that clearly indicate the person works at The Washington Post. "
            "The source does not need to explicitly mention 'November 2025' but should reasonably indicate current employment (e.g., present-tense descriptions, recent bylines). "
            "Reject if the source suggests former employment only."
        ),
        (
            f"{name} serves as the West Africa bureau chief for The Washington Post.",
            sources,
            west_africa_node,
            "Look for explicit mention of the title 'West Africa bureau chief' in association with The Washington Post. "
            "Minor phrasing variations like 'West Africa bureau chief for The Washington Post' or 'The Washington Post's West Africa bureau chief' should be accepted."
        ),
        (
            f"{name} was appointed West Africa bureau chief in 2022.",
            sources,
            appointed_2022_node,
            "Check for explicit statements such as 'appointed in 2022', 'since 2022', or equivalent phrasing connecting the West Africa bureau chief role with the year 2022."
        ),
        (
            f"{name} graduated from Duke University in 2017 with a BA in Political Science.",
            sources,
            education_node,
            "Confirm all three aspects: (1) Duke University, (2) year 2017 (allow formats like 'Class of 2017' or '2017 graduate'), and (3) BA/B.A. in Political Science (allow 'Bachelor's degree in Political Science')."
        ),
        (
            f"Before becoming a foreign correspondent, {name} worked at The Washington Post's Local desk covering politics and government in Prince George's County, Maryland.",
            sources,
            prior_local_node,
            "Look for explicit mention that the journalist previously worked on the Local desk covering politics and government in Prince George's County (Maryland). "
            "Evidence may be from a staff bio, profile, or credible article indicating that prior beat. "
            "The timing 'before becoming a foreign correspondent' can be inferred if the bio describes the Local desk role as prior or earlier."
        ),
    ]
    await evaluator.batch_verify(claims_and_sources)


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
    Entry point for evaluating an answer for the journalist identification task.
    """
    # Initialize evaluator (framework root is non-critical by design)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_journalist_info(),
        template_class=JournalistExtraction,
        extraction_name="journalist_extraction",
    )

    # Build and verify rubric tree
    await build_and_verify_journalist_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()