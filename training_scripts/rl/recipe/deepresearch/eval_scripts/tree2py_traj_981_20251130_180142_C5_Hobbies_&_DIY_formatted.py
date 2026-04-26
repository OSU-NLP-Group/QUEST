import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "kids_diy_workshops_dec_2025"
TASK_DESCRIPTION = (
    "Identify two FREE kids' DIY workshops offered by major home improvement or craft store retailers that are "
    "scheduled for December 6-7, 2025, and are suitable for children ages 5 and 7. The two workshops must be from different retailers. "
    "For each workshop, provide: (1) The store name and retailer type, (2) The workshop date and time, (3) The age requirements, "
    "(4) The registration requirements (including whether registration is required or optional, and how to register), "
    "(5) A description of the project that children will create, and (6) A reference URL that supports the workshop information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Workshop(BaseModel):
    store_name: Optional[str] = None
    retailer_type: Optional[str] = None  # e.g., "home improvement", "craft store"
    date: Optional[str] = None           # keep as text to be flexible (e.g., "Dec 7, 2025", "12/07/2025")
    time: Optional[str] = None           # e.g., "9:00 AM - 12:00 PM"
    age_requirements: Optional[str] = None  # e.g., "ages 5-12", "recommended for ages 5 and up"
    registration_requirements: Optional[str] = None  # free text from answer
    registration_required: Optional[str] = None      # "required" | "optional" | "unspecified"
    registration_how: Optional[str] = None           # e.g., "online form", "in-store sign-up", "Eventbrite"
    project_description: Optional[str] = None
    # URLs referenced in the answer that support the workshop info; can include multiple
    reference_urls: List[str] = Field(default_factory=list)
    # Whether the answer claims the workshop is free (extracted flag; verification uses URLs)
    is_free_claimed: Optional[bool] = None


class WorkshopsExtraction(BaseModel):
    workshops: List[Workshop] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_workshops() -> str:
    return """
    Extract up to two kids' DIY workshops mentioned in the answer. Return them in a JSON object under 'workshops' as an array.

    For each workshop, extract the following fields exactly as presented in the answer (use null if missing):
    - store_name: The retailer/store name (e.g., "The Home Depot", "Lowe's", "Michaels", "Joann", "Ace Hardware").
    - retailer_type: Classify as either "home improvement" or "craft store" based on the store. If unclear, use the best inference from the answer.
    - date: The workshop date text (e.g., "Dec 6, 2025", "December 7, 2025", or "12/07/2025").
    - time: The specific workshop time text (start/end times or specific time).
    - age_requirements: The age policy text (e.g., "ages 5-12", "recommended for ages 5 and up").
    - registration_requirements: The registration policy text (e.g., "registration required via online form", "optional registration").
    - registration_required: One of "required", "optional", or "unspecified", reflecting the answer's claim.
    - registration_how: How to register (e.g., "online form", "Eventbrite", "in-store"; include link text if present).
    - project_description: Description of the project the children will create.
    - is_free_claimed: true if the answer explicitly claims the workshop is free of charge, false otherwise; null if not stated.
    - reference_urls: All URLs cited in the answer for this workshop (as full URLs, accept plain or markdown link targets). If none, return an empty list.

    Notes:
    - Do not invent information; only extract what is explicitly present in the answer text.
    - Keep original phrasing of date/time and descriptions.
    - If URLs are missing a protocol, prepend "http://".
    - Return at most two workshops (if more are listed, include only the first two). If fewer, include what is available.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _valid_urls(urls: List[str]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if not (s.startswith("http://") or s.startswith("https://")):
            s = "http://" + s
        cleaned.append(s)
    return cleaned


def _has_valid_url(urls: List[str]) -> bool:
    return any(u.startswith("http://") or u.startswith("https://") for u in urls)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_workshop(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    workshop: Workshop,
    index: int,
) -> Dict[str, VerificationNode]:
    """
    Build verification sub-tree and run verifications for a single workshop.
    Returns a dict with some key nodes for external prerequisites (e.g., store info).
    """
    is_first = index == 0
    node_desc = "First workshop meeting the specified criteria" if is_first else "Second workshop meeting the specified criteria"

    ws_node = evaluator.add_parallel(
        id=f"workshop_{index + 1}",
        desc=node_desc,
        parent=parent_node,
        critical=False  # allow partial credit per workshop
    )

    urls = _valid_urls(workshop.reference_urls)

    # 1) Store name and retailer type provided (existence check - critical)
    store_provided = bool(workshop.store_name and workshop.store_name.strip()) and bool(workshop.retailer_type and workshop.retailer_type.strip())
    store_node = evaluator.add_custom_node(
        result=store_provided,
        id=f"workshop_{index + 1}_store_name_and_type_provided",
        desc="Store name and retailer type are provided",
        parent=ws_node,
        critical=True
    )

    # 2) Reference URL provided (existence/validity - critical)
    ref_url_node = evaluator.add_custom_node(
        result=(len(urls) > 0 and _has_valid_url(urls)),
        id=f"workshop_{index + 1}_reference_url",
        desc="A valid reference URL is provided that supports the workshop information",
        parent=ws_node,
        critical=True
    )

    # 3) Workshop cost is FREE (critical; verify via URL(s))
    cost_node = evaluator.add_leaf(
        id=f"workshop_{index + 1}_workshop_cost",
        desc="Workshop is verified to be FREE of charge",
        parent=ws_node,
        critical=True
    )
    await evaluator.verify(
        claim="The workshop is free of charge for participants (no fee required).",
        node=cost_node,
        sources=urls,
        additional_instruction="Look for language such as 'Free', 'No cost', 'Complimentary', or similar on the referenced page.",
        extra_prerequisites=[ref_url_node]
    )

    # 4) Retailer qualifies as a major home improvement or craft store (critical; simple verify)
    retailer_qual_node = evaluator.add_leaf(
        id=f"workshop_{index + 1}_retailer_qualifies",
        desc="Retailer qualifies as a major home improvement or craft store retailer",
        parent=ws_node,
        critical=True
    )
    store_for_claim = workshop.store_name or ""
    await evaluator.verify(
        claim=f"The retailer '{store_for_claim}' qualifies as a major home improvement or craft store retailer (widely recognized national chain in hardware/home improvement or arts & crafts).",
        node=retailer_qual_node,
        additional_instruction="Use general knowledge and reasonable judgment of U.S. retail chains; consider variants like 'Home Depot' vs 'The Home Depot' equivalent.",
        extra_prerequisites=[store_node]
    )

    # 5) Date is within December 6-7, 2025 (critical; verify via URL(s))
    date_node = evaluator.add_leaf(
        id=f"workshop_{index + 1}_date_within_dec_6_7_2025",
        desc="Workshop date falls within December 6-7, 2025",
        parent=ws_node,
        critical=True
    )
    await evaluator.verify(
        claim="The workshop date stated on the referenced page is either December 6, 2025 or December 7, 2025.",
        node=date_node,
        sources=urls,
        additional_instruction="Interpret common date formats (e.g., 'Dec 6, 2025', '12/06/2025', 'Saturday, December 6, 2025').",
        extra_prerequisites=[ref_url_node]
    )

    # 6) Time provided (critical; existence check)
    time_provided = bool(workshop.time and workshop.time.strip())
    time_node = evaluator.add_custom_node(
        result=time_provided,
        id=f"workshop_{index + 1}_time_provided",
        desc="A specific workshop time is provided (start/end time or a specific time)",
        parent=ws_node,
        critical=True
    )

    # 7a) Age requirements provided (critical helper existence)
    age_text_provided = bool(workshop.age_requirements and workshop.age_requirements.strip())
    age_text_node = evaluator.add_custom_node(
        result=age_text_provided,
        id=f"workshop_{index + 1}_age_requirements_provided",
        desc="Age requirements are specified in the answer",
        parent=ws_node,
        critical=True
    )
    # 7b) Age requirements confirm 5 and 7 eligible (critical; verify via URL(s))
    age_ok_node = evaluator.add_leaf(
        id=f"workshop_{index + 1}_age_requirements",
        desc="Age requirements are specified and confirm children ages 5 and 7 are eligible",
        parent=ws_node,
        critical=True
    )
    await evaluator.verify(
        claim="Children aged 5 and children aged 7 are eligible to attend this workshop (the allowed age range includes both 5 and 7).",
        node=age_ok_node,
        sources=urls,
        additional_instruction="Accept ranges like 'ages 5-12', '5+' as including both ages; if the page shows stricter limits excluding 5 or 7, mark incorrect.",
        extra_prerequisites=[ref_url_node, age_text_node]
    )

    # 8a) Registration information provided (critical helper existence)
    reg_info_provided = bool(workshop.registration_required and workshop.registration_required.strip()) and bool(workshop.registration_how and workshop.registration_how.strip())
    reg_info_node = evaluator.add_custom_node(
        result=reg_info_provided,
        id=f"workshop_{index + 1}_registration_requirements_provided",
        desc="Registration details (required/optional and how to register) are provided in the answer",
        parent=ws_node,
        critical=True
    )
    # 8b) Registration requirements verified (critical; verify via URL(s))
    reg_req_node = evaluator.add_leaf(
        id=f"workshop_{index + 1}_registration_requirements",
        desc="Registration requirements are specified, including whether registration is required or optional and how to register",
        parent=ws_node,
        critical=True
    )
    reg_required_text = (workshop.registration_required or "").strip()
    reg_how_text = (workshop.registration_how or "").strip()
    claim_reg = (
        f"The page indicates registration is {reg_required_text if reg_required_text else 'clearly specified'} "
        f"and participants register by {reg_how_text if reg_how_text else 'a clearly described method'}."
    )
    await evaluator.verify(
        claim=claim_reg,
        node=reg_req_node,
        sources=urls,
        additional_instruction="Look for explicit statements like 'Registration required', 'Optional registration', and instructions such as 'Register online', 'Sign up in store', or a linked registration page.",
        extra_prerequisites=[ref_url_node, reg_info_node]
    )

    # 9) Project description provided (critical; existence)
    project_provided = bool(workshop.project_description and workshop.project_description.strip())
    project_node = evaluator.add_custom_node(
        result=project_provided,
        id=f"workshop_{index + 1}_project_description",
        desc="Description of the project/activity children will create is provided",
        parent=ws_node,
        critical=True
    )

    # Return references for external checks
    return {
        "store_node": store_node,
        "ref_url_node": ref_url_node,
        "ws_node": ws_node
    }


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
    Evaluate an answer for the kids' DIY workshops (Dec 6-7, 2025) task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # workshops evaluated independently + different retailers check
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

    # 1) Extract workshop information
    extracted: WorkshopsExtraction = await evaluator.extract(
        prompt=prompt_extract_workshops(),
        template_class=WorkshopsExtraction,
        extraction_name="workshops_extraction"
    )

    # Normalize and take up to two workshops; pad with empty if fewer
    workshops: List[Workshop] = list(extracted.workshops[:2])
    while len(workshops) < 2:
        workshops.append(Workshop())

    # 2) Build and verify each workshop subtree
    ws_nodes_info: List[Dict[str, VerificationNode]] = []
    for idx in range(2):
        ws_info = await verify_workshop(
            evaluator=evaluator,
            parent_node=root,
            workshop=workshops[idx],
            index=idx
        )
        ws_nodes_info.append(ws_info)

    # 3) Cross-workshop constraint: different retailers (critical)
    diff_retailers_leaf = evaluator.add_leaf(
        id="different_retailers",
        desc="Workshop 1 and Workshop 2 are from different retailers (not the same retailer/brand)",
        parent=root,
        critical=True
    )
    store1 = (workshops[0].store_name or "").strip()
    store2 = (workshops[1].store_name or "").strip()

    # Ensure prerequisite: both store nodes provided
    prereqs = [ws_nodes_info[0]["store_node"], ws_nodes_info[1]["store_node"]]

    await evaluator.verify(
        claim=f"The two retailers '{store1}' and '{store2}' are different brands; they are not the same retailer/brand.",
        node=diff_retailers_leaf,
        additional_instruction="Treat minor naming variants (e.g., 'Home Depot' vs 'The Home Depot') as the same brand; the two must be distinct chains.",
        extra_prerequisites=prereqs
    )

    # 4) Return summary
    return evaluator.get_summary()