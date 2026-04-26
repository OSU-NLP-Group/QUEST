import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "macys_parade_2027_requirements"
TASK_DESCRIPTION = (
    "A community arts organization in New Jersey is planning to participate in the 2027 Macy's Thanksgiving Day Parade "
    "with a custom float and a youth dance performance group. To prepare their proposal and ensure they meet all requirements, "
    "they need specific information about parade participation. Provide the following details: "
    "(1) What are the maximum collapsed dimensions (height and width) that parade floats must meet to travel through the Lincoln Tunnel "
    "from New Jersey to the parade route in Manhattan? "
    "(2) When does the application process for performance groups to participate in the 2027 Macy's Thanksgiving Day Parade open, and by what date "
    "will accepted groups be notified of their selection? "
    "(3) What are the age requirements (minimum and maximum age) for dancers participating in the parade's dance performance groups organized "
    "by Spirit of America Productions?"
)

# Ground truth snapshot (for info only; actual verification relies on cited sources)
GROUND_TRUTH_EXPECTED = {
    "float_collapsed_height": "12.5 feet",
    "float_collapsed_width": "8 feet",
    "application_opening_date": "November 29, 2025",
    "notification_deadline": "June 30, 2026",
    "dance_min_age": "14",
    "dance_max_age": "18",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SingleFact(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ParadeFactsExtraction(BaseModel):
    float_collapsed_height: Optional[SingleFact] = None
    float_collapsed_width: Optional[SingleFact] = None
    application_opening_date: Optional[SingleFact] = None
    notification_deadline: Optional[SingleFact] = None
    dance_min_age: Optional[SingleFact] = None
    dance_max_age: Optional[SingleFact] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parade_facts() -> str:
    return (
        "Extract the following specific facts as presented in the answer, along with all URLs that the answer cites as sources for each fact. "
        "Return each fact with two fields: 'value' (a string, exactly as written in the answer) and 'sources' (an array of URLs explicitly mentioned in the answer that support this fact). "
        "If a value is missing, use null. If no URLs are provided, return an empty array.\n"
        "Facts to extract:\n"
        "1) float_collapsed_height: maximum collapsed float height required to travel through the Lincoln Tunnel from New Jersey to Manhattan.\n"
        "2) float_collapsed_width: maximum collapsed float width required to travel through the Lincoln Tunnel from New Jersey to Manhattan.\n"
        "3) application_opening_date: when the application process for performance groups to participate in the 2027 Macy's Thanksgiving Day Parade opens.\n"
        "4) notification_deadline: the latest date by which accepted performance groups will be notified of their selection.\n"
        "5) dance_min_age: minimum age requirement for Spirit of America Productions dance performers.\n"
        "6) dance_max_age: maximum age requirement for Spirit of America Productions dance performers.\n"
        "Important:\n"
        "- Extract values as strings exactly as they appear in the answer (do not normalize or convert units). Examples of acceptable formats include \"12.5 feet\", \"12'6\"\", \"8 ft\", \"November 29, 2025\", \"no later than June 30, 2026\", \"14\", \"18\".\n"
        "- For 'sources', include only actual URLs explicitly present in the answer, in any reasonable format (plain or markdown links). If an item has multiple URLs, include all of them.\n"
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _sanitize_sources(sources: Optional[List[str]]) -> Optional[List[str]]:
    if not sources:
        return None
    # Filter out obvious non-URLs and trim whitespace
    cleaned = []
    for s in sources:
        if not isinstance(s, str):
            continue
        t = s.strip()
        if t:
            cleaned.append(t)
    return cleaned or None


async def _verify_or_fail(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    value: Optional[str],
    sources: Optional[List[str]],
    claim_template: str,
    add_ins: str,
    critical: bool = True,
) -> None:
    """
    Create a leaf node and verify the claim. If the value is missing, mark the node as failed immediately.
    """
    if value is None or str(value).strip() == "":
        # If the answer did not provide the value, mark this mandatory item as failed.
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=node_desc,
            parent=parent_node,
            critical=critical,
        )
        return

    node = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=critical,
    )

    claim = claim_template.format(value=value.strip())
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=_sanitize_sources(sources),
        additional_instruction=add_ins,
    )


# --------------------------------------------------------------------------- #
# Verification entry                                                          #
# --------------------------------------------------------------------------- #
async def verify_parade_requirements(
    evaluator: Evaluator,
    root_node,
    facts: ParadeFactsExtraction,
) -> None:
    # Float collapsed height
    await _verify_or_fail(
        evaluator=evaluator,
        parent_node=root_node,
        node_id="Float_Collapsed_Height",
        node_desc="Provides the maximum collapsed float height required to travel through the Lincoln Tunnel (12.5 feet).",
        value=(facts.float_collapsed_height.value if facts.float_collapsed_height else None),
        sources=(facts.float_collapsed_height.sources if facts.float_collapsed_height else None),
        claim_template="The maximum collapsed float height required to travel through the Lincoln Tunnel from New Jersey to the Macy's Thanksgiving Day Parade route in Manhattan is {value}.",
        add_ins=(
            "Verify this is specifically the transport/collapsed height constraint for parade floats using the Lincoln Tunnel. "
            "Accept reasonable equivalences in units and notation (e.g., 12'6\" is equivalent to 12.5 feet). "
            "The page must clearly state such a maximum collapsed height requirement."
        ),
        critical=True,
    )

    # Float collapsed width
    await _verify_or_fail(
        evaluator=evaluator,
        parent_node=root_node,
        node_id="Float_Collapsed_Width",
        node_desc="Provides the maximum collapsed float width required to travel through the Lincoln Tunnel (8 feet).",
        value=(facts.float_collapsed_width.value if facts.float_collapsed_width else None),
        sources=(facts.float_collapsed_width.sources if facts.float_collapsed_width else None),
        claim_template="The maximum collapsed float width required to travel through the Lincoln Tunnel from New Jersey to the Macy's Thanksgiving Day Parade route in Manhattan is {value}.",
        add_ins=(
            "Verify this is specifically the transport/collapsed width constraint for parade floats using the Lincoln Tunnel. "
            "Accept minor formatting variations (e.g., '8 ft' or '8 feet'). "
            "The page must clearly state such a maximum collapsed width requirement."
        ),
        critical=True,
    )

    # Application opening date
    await _verify_or_fail(
        evaluator=evaluator,
        parent_node=root_node,
        node_id="Application_Opening_Date",
        node_desc="Provides when the application process for 2027 performance groups opens (November 29, 2025).",
        value=(facts.application_opening_date.value if facts.application_opening_date else None),
        sources=(facts.application_opening_date.sources if facts.application_opening_date else None),
        claim_template="The application process for performance groups to participate in the 2027 Macy's Thanksgiving Day Parade opens on {value}.",
        add_ins=(
            "Verify that the date refers to the opening of applications for performance groups for the 2027 Macy's Thanksgiving Day Parade. "
            "Ensure the source page pertains to the 2027 event cycle and to performance groups (e.g., school/community dance/cheer/band ensembles)."
        ),
        critical=True,
    )

    # Notification deadline
    await _verify_or_fail(
        evaluator=evaluator,
        parent_node=root_node,
        node_id="Notification_Deadline",
        node_desc="Provides the latest date by which accepted groups will be notified (no later than June 30, 2026).",
        value=(facts.notification_deadline.value if facts.notification_deadline else None),
        sources=(facts.notification_deadline.sources if facts.notification_deadline else None),
        claim_template="Accepted performance groups will be notified of their selection by {value}.",
        add_ins=(
            "Verify that the source explicitly states the notification timeline or deadline for accepted performance groups for the 2027 parade cycle. "
            "Accept phrasing such as 'no later than June 30, 2026' as equivalent to 'by June 30, 2026'."
        ),
        critical=True,
    )

    # Dancer minimum age
    await _verify_or_fail(
        evaluator=evaluator,
        parent_node=root_node,
        node_id="Dance_Performer_Minimum_Age",
        node_desc="Provides the minimum age requirement for Spirit of America Productions dance performers (at least 14 years old).",
        value=(facts.dance_min_age.value if facts.dance_min_age else None),
        sources=(facts.dance_min_age.sources if facts.dance_min_age else None),
        claim_template="Spirit of America Productions requires dancers in parade performance groups to be at least {value} years old.",
        add_ins=(
            "Verify the age requirement specifically for dancers in Spirit of America Productions parade performance groups. "
            "Accept equivalent wording indicating a minimum age (e.g., 'must be 14 or older', 'ages 14–18')."
        ),
        critical=True,
    )

    # Dancer maximum age
    await _verify_or_fail(
        evaluator=evaluator,
        parent_node=root_node,
        node_id="Dance_Performer_Maximum_Age",
        node_desc="Provides the maximum age requirement for Spirit of America Productions dance performers (no older than 18 years old).",
        value=(facts.dance_max_age.value if facts.dance_max_age else None),
        sources=(facts.dance_max_age.sources if facts.dance_max_age else None),
        claim_template="Spirit of America Productions requires dancers in parade performance groups to be no older than {value} years old.",
        add_ins=(
            "Verify the age requirement specifically for dancers in Spirit of America Productions parade performance groups. "
            "Accept equivalent wording indicating a maximum age (e.g., 'up to age 18', 'ages 14–18')."
        ),
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
    Evaluate the answer for the Macy's Thanksgiving Day Parade 2027 participation requirements.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Verify all mandatory required details for participating in the 2027 Macy's Thanksgiving Day Parade with a float and dance performance group.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Root is critical; all children must also be critical
    root.critical = True

    # Record ground truth expectations (for reference only)
    evaluator.add_ground_truth(
        {
            "expected": GROUND_TRUTH_EXPECTED,
            "notes": "Verification ultimately depends on whether the cited sources support the claims from the answer."
        },
        gt_type="ground_truth"
    )

    # Extract facts and source URLs from the answer
    extracted_facts = await evaluator.extract(
        prompt=prompt_extract_parade_facts(),
        template_class=ParadeFactsExtraction,
        extraction_name="parade_requirements"
    )

    # Build the verification leaves according to rubric
    await verify_parade_requirements(evaluator, root, extracted_facts)

    # Return structured evaluation summary
    return evaluator.get_summary()