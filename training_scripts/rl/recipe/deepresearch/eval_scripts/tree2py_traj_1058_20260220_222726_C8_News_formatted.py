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
TASK_ID = "trump_term2_cabinet_confirmations"
TASK_DESCRIPTION = (
    "During President Donald Trump's second term (2025-2026), the United States Senate confirmed his cabinet "
    "secretaries through constitutional processes that resulted in varying levels of support. Your task is to identify "
    "four different cabinet secretaries from Trump's second term who were confirmed with distinctly different vote "
    "margins, and provide detailed information about each confirmation.\n\n"
    "Specifically, identify:\n\n"
    "1. One secretary confirmed with near-unanimous or unanimous support (95 or more yes votes)\n"
    "2. One secretary confirmed with a narrow margin (51-52 yes votes)\n"
    "3. One secretary confirmed with a close vote (53-60 yes votes)\n"
    "4. One secretary confirmed with strong bipartisan support (70 or more yes votes)\n\n"
    "For each of the four secretaries, provide:\n"
    "- The secretary's full name\n"
    "- The cabinet department they lead\n"
    "- The exact Senate confirmation vote count (yes-no breakdown)\n"
    "- The date of Senate confirmation\n"
    "- The Senate committee that held jurisdiction over this nomination\n"
    "- A reference URL documenting the confirmation details\n\n"
    "All four secretaries must be different individuals, and all information must be accurate and verifiable through "
    "reliable sources such as Senate records, official government websites, or established news organizations."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class SecretaryInfo(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    # Keep votes as strings for flexibility; agent may provide "51" or "51 votes"
    yes_votes: Optional[str] = None
    no_votes: Optional[str] = None
    # Optional combined text if provided, e.g., "51-49"
    vote_breakdown: Optional[str] = None
    confirmation_date: Optional[str] = None
    committee: Optional[str] = None
    reference_url: Optional[str] = None


class ConfirmationsExtraction(BaseModel):
    unanimous: Optional[SecretaryInfo] = None     # 95+ yes votes
    narrow: Optional[SecretaryInfo] = None        # 51-52 yes votes
    close: Optional[SecretaryInfo] = None         # 53-60 yes votes
    bipartisan: Optional[SecretaryInfo] = None    # 70+ yes votes


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_confirmations() -> str:
    return """
Extract the four cabinet secretary confirmations described in the answer for President Donald Trump's second term (2025-2026), grouped by vote margin category. For each category, extract exactly one secretary (choose the one explicitly labeled for that category or the first one that clearly fits the category in the answer). If a category is not present, return null for that category.

For each secretary, extract these fields exactly as they appear in the answer:
- name: Full name of the cabinet secretary.
- department: The cabinet department (e.g., 'Department of State', 'Homeland Security', etc.). Keep the department wording as close as possible to the answer.
- yes_votes: The yes vote count (digits only if possible; otherwise the exact string from the answer).
- no_votes: The no vote count (digits only if possible; otherwise the exact string from the answer).
- vote_breakdown: If the answer presents the vote as a single text like '51-49' or 'confirmed 51-49', extract that string as-is (otherwise set null).
- confirmation_date: The confirmation date string as shown (do not transform the date format).
- committee: The Senate committee with jurisdiction, as presented in the answer (e.g., 'Senate Committee on Finance').
- reference_url: A single URL that documents the confirmation details; prefer an official Senate page, official government site, or a reputable news organization link mentioned with that secretary in the answer.

Return a JSON object with the following top-level keys and one object per category:
- unanimous: {...}       // 95 or more yes votes (near-unanimous or unanimous)
- narrow: {...}          // 51-52 yes votes
- close: {...}           // 53-60 yes votes
- bipartisan: {...}      // 70 or more yes votes

If any field is missing in the answer for a secretary, set it to null. Do not invent or guess any information not present in the answer.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s if isinstance(s, str) else ""

def _urls_list(primary: Optional[str]) -> Optional[str] | List[str]:
    """Return either None or a single URL string. The verification API accepts str or list[str] or None."""
    u = _safe(primary).strip()
    return u if u else None


# --------------------------------------------------------------------------- #
# Category verification logic                                                 #
# --------------------------------------------------------------------------- #
async def verify_category(
    evaluator: Evaluator,
    parent_node,
    category_id: str,
    category_desc: str,
    item: Optional[SecretaryInfo],
    vote_threshold_type: str,  # one of: "unanimous_95plus", "narrow_51to52", "close_53to60", "bipartisan_70plus"
) -> None:
    """
    Build verification sub-tree for a single category and execute all leaf verifications.
    All leaves are critical under the category to reflect the rubric's requirements.
    """

    # Create a parallel node for this category (non-critical so root can get partial credit if others pass)
    cat_node = evaluator.add_parallel(
        id=category_id,
        desc=category_desc,
        parent=parent_node,
        critical=False
    )

    # Prepare fields robustly
    name = _safe(item.name if item else None)
    department = _safe(item.department if item else None)
    yes_votes = _safe(item.yes_votes if item else None)
    no_votes = _safe(item.no_votes if item else None)
    vote_breakdown = _safe(item.vote_breakdown if item else None)
    date_str = _safe(item.confirmation_date if item else None)
    committee = _safe(item.committee if item else None)
    ref_url = _safe(item.reference_url if item else None)

    # Common guidance for minor formatting differences
    common_additional_ins = (
        "Allow reasonable variations in department phrasing (e.g., 'Secretary of Veterans Affairs' vs "
        "'Veterans Affairs Secretary'), and in date formatting (e.g., 'Jan 3, 2025' vs 'January 3, 2025'). "
        "Focus on whether the provided webpage explicitly documents the Senate confirmation and the specified detail."
    )

    # 1) Identity check
    identity_node = evaluator.add_leaf(
        id=f"{category_id.replace(' ', '_')}_identity",
        desc="The provided name is a valid Trump second-term cabinet secretary",
        parent=cat_node,
        critical=True
    )
    identity_claim = (
        f"This page documents that {name} was confirmed by the U.S. Senate to serve as the Secretary of {department}."
        if name and department else
        f"This page documents the Senate confirmation of the named cabinet secretary."
    )
    await evaluator.verify(
        claim=identity_claim,
        node=identity_node,
        sources=_urls_list(ref_url),
        additional_instruction=(
            common_additional_ins + " If the confirmation occurred in 2025 or 2026, treat it as part of Trump's second term."
        )
    )

    # 2) Department check
    department_node = evaluator.add_leaf(
        id=f"{category_id.replace(' ', '_')}_department",
        desc="The department specified correctly matches the secretary's actual department",
        parent=cat_node,
        critical=True
    )
    department_claim = (
        f"The page shows that the position is Secretary of {department} (or equivalent phrasing)."
        if department else
        "The page shows and matches the stated department for this nomination."
    )
    await evaluator.verify(
        claim=department_claim,
        node=department_node,
        sources=_urls_list(ref_url),
        additional_instruction=common_additional_ins
    )

    # 3) Vote threshold category check
    vote_threshold_node = evaluator.add_leaf(
        id=f"{category_id.replace(' ', '_')}_vote_threshold",
        desc="The confirmation vote count shows the required yes-vote threshold for this category",
        parent=cat_node,
        critical=True
    )
    if vote_threshold_type == "unanimous_95plus":
        threshold_claim = (
            f"The Senate confirmation for {name} shows 95 or more yes votes."
            if name else "The Senate confirmation shows 95 or more yes votes."
        )
    elif vote_threshold_type == "narrow_51to52":
        threshold_claim = (
            f"The Senate confirmation for {name} shows either 51 or 52 yes votes."
            if name else "The Senate confirmation shows either 51 or 52 yes votes."
        )
    elif vote_threshold_type == "close_53to60":
        threshold_claim = (
            f"The Senate confirmation for {name} shows between 53 and 60 yes votes (inclusive)."
            if name else "The Senate confirmation shows between 53 and 60 yes votes (inclusive)."
        )
    elif vote_threshold_type == "bipartisan_70plus":
        threshold_claim = (
            f"The Senate confirmation for {name} shows 70 or more yes votes."
            if name else "The Senate confirmation shows 70 or more yes votes."
        )
    else:
        threshold_claim = "The page shows the correct yes-vote threshold for this category."

    await evaluator.verify(
        claim=threshold_claim,
        node=vote_threshold_node,
        sources=_urls_list(ref_url),
        additional_instruction=(
            common_additional_ins +
            " Verify the yes vote count on the page falls into the specified range for this category."
        )
    )

    # 4) Exact vote accuracy check
    vote_accuracy_node = evaluator.add_leaf(
        id=f"{category_id.replace(' ', '_')}_vote_accuracy",
        desc="The exact vote count (yes-no breakdown) is accurately reported",
        parent=cat_node,
        critical=True
    )
    # Try to state exact breakdown if available; otherwise generic phrasing will likely fail (as intended)
    if yes_votes and no_votes:
        vote_accuracy_claim = f"The Senate confirmation vote was {yes_votes.strip()}-{no_votes.strip()}."
    elif vote_breakdown:
        vote_accuracy_claim = f"The Senate confirmation vote was {vote_breakdown.strip()}."
    else:
        vote_accuracy_claim = "The page reflects the exact yes-no vote breakdown as stated in the answer."
    await evaluator.verify(
        claim=vote_accuracy_claim,
        node=vote_accuracy_node,
        sources=_urls_list(ref_url),
        additional_instruction=(
            common_additional_ins + " Confirm the yes-no counts match exactly, allowing for minor formatting differences."
        )
    )

    # 5) Date accuracy check
    date_node = evaluator.add_leaf(
        id=f"{category_id.replace(' ', '_')}_date",
        desc="The confirmation date is accurately reported",
        parent=cat_node,
        critical=True
    )
    if date_str:
        date_claim = f"The Senate confirmed {name} on {date_str}."
    else:
        date_claim = "The page shows the exact confirmation date as stated in the answer."
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=_urls_list(ref_url),
        additional_instruction=(
            common_additional_ins +
            " Focus on matching the confirmation date; equivalently formatted dates should be accepted."
        )
    )

    # 6) Committee accuracy check
    committee_node = evaluator.add_leaf(
        id=f"{category_id.replace(' ', '_')}_committee",
        desc="The Senate committee that handled this nomination is correctly identified",
        parent=cat_node,
        critical=True
    )
    if committee:
        committee_claim = f"The nomination was handled by the {committee}."
    else:
        committee_claim = "The page identifies the correct Senate committee that handled this nomination."
    await evaluator.verify(
        claim=committee_claim,
        node=committee_node,
        sources=_urls_list(ref_url),
        additional_instruction=(
            common_additional_ins +
            " Confirm that the committee named matches the one shown or explicitly associated with the nomination."
        )
    )

    # 7) Reference validity and support check
    reference_node = evaluator.add_leaf(
        id=f"{category_id.replace(' ', '_')}_reference",
        desc="A valid reference URL is provided that confirms the secretary's confirmation details",
        parent=cat_node,
        critical=True
    )
    reference_claim = (
        f"This page is a reliable source documenting the Senate confirmation of {name} as {department} secretary, "
        f"including the vote and date."
        if name and department else
        "This page is a reliable source documenting the Senate confirmation details for the secretary."
    )
    await evaluator.verify(
        claim=reference_claim,
        node=reference_node,
        sources=_urls_list(ref_url),
        additional_instruction=(
            "Consider U.S. Senate official records, official government sites (e.g., .gov), and established news "
            "organizations as reliable sources. The page should explicitly support the confirmation details mentioned."
        )
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
    Evaluate the answer for the four Trump second-term cabinet secretary confirmations with distinct vote margins.
    """

    # Initialize evaluator (root is non-critical by design to allow partial credit; parallel as categories are independent)
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

    # Extract structured data
    extracted = await evaluator.extract(
        prompt=prompt_extract_confirmations(),
        template_class=ConfirmationsExtraction,
        extraction_name="confirmations_extraction"
    )

    # Build verification trees for each category
    await verify_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="unanimous_secretary",
        category_desc="Identify one cabinet secretary who was confirmed with a unanimous or near-unanimous vote (95+ yes votes)",
        item=extracted.unanimous,
        vote_threshold_type="unanimous_95plus",
    )

    await verify_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="narrow_margin_secretary",
        category_desc="Identify one cabinet secretary who was confirmed with 51-52 yes votes (narrow margin)",
        item=extracted.narrow,
        vote_threshold_type="narrow_51to52",
    )

    await verify_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="close_vote_secretary",
        category_desc="Identify one cabinet secretary who was confirmed with 53-60 yes votes",
        item=extracted.close,
        vote_threshold_type="close_53to60",
    )

    await verify_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="bipartisan_secretary",
        category_desc="Identify one cabinet secretary who was confirmed with 70+ yes votes (strong bipartisan support)",
        item=extracted.bipartisan,
        vote_threshold_type="bipartisan_70plus",
    )

    # Add the "no_duplicates" critical leaf at root
    no_duplicates_leaf = evaluator.add_leaf(
        id="no_duplicates",
        desc="All four secretaries identified are different individuals",
        parent=root,
        critical=True
    )
    names = [
        _safe(extracted.unanimous.name if extracted and extracted.unanimous else None),
        _safe(extracted.narrow.name if extracted and extracted.narrow else None),
        _safe(extracted.close.name if extracted and extracted.close else None),
        _safe(extracted.bipartisan.name if extracted and extracted.bipartisan else None),
    ]
    # Formulate a simple logical check claim
    uniq_claim = (
        f"The following four names are all distinct individuals: {', '.join([n for n in names if n])}."
        if any(names) else
        "The four provided names are all distinct."
    )
    await evaluator.verify(
        claim=uniq_claim,
        node=no_duplicates_leaf,
        additional_instruction=(
            "Treat blank or missing names as not satisfying the distinctness requirement. If any two non-empty names are "
            "the same person (allowing for minor formatting differences like middle initials), this should be considered not distinct."
        )
    )

    # Return full evaluation summary
    return evaluator.get_summary()