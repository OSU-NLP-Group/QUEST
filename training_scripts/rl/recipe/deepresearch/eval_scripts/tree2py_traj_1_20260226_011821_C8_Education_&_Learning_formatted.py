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
TASK_ID = "weather_delay_policies_4_institutions"
TASK_DESCRIPTION = (
    "Identify four school districts or universities in the United States that have publicly documented inclement "
    "weather delay policies. For each institution, provide the following information: (1) The institution's name, "
    "(2) The specific time by which day-of weather closure/delay decisions are communicated to families and staff, "
    "(3) Whether the institution uses defined operational status codes (such as Code Red, Code Orange, etc.) to "
    "communicate different types of closures or delays, (4) The communication channels used for emergency notifications "
    "(list all channels mentioned, such as text, email, phone calls, website, mobile app, etc.), and (5) Whether the "
    "institution provides a mechanism for parents/guardians and students to update their contact information for "
    "emergency notifications. Include a reference URL for each institution that supports the provided information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class InstitutionItem(BaseModel):
    """One institution's weather delay policy information extracted from the answer."""
    name: Optional[str] = None
    decision_time: Optional[str] = None  # e.g., "by 5:30 a.m.", "by 6:00 AM", "no later than 5:45 a.m."
    uses_codes: Optional[str] = None     # Prefer "yes" | "no" | "unknown"
    codes_list: List[str] = Field(default_factory=list)  # Example code names if any (e.g., ["Code Red", "Code Orange"])
    channels: List[str] = Field(default_factory=list)    # e.g., ["text", "email", "phone", "website", "mobile app"]
    contact_update: Optional[str] = None # Prefer "yes" | "no" | "unknown" or short description like "via parent portal"
    reference_url: Optional[str] = None  # A single reference URL supporting the above info


class InstitutionsExtraction(BaseModel):
    institutions: List[InstitutionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_institutions() -> str:
    return """
    Extract up to four U.S. educational institutions (school districts or universities) from the answer that have
    publicly documented inclement weather closure/delay policies. For each institution, extract these fields:

    - name: The institution's name (district or university)
    - decision_time: The specific time by which day-of weather closure/delay decisions are communicated (e.g., "by 5:30 a.m.")
    - uses_codes: Whether the institution uses defined operational status codes to communicate closures/delays.
                  Use lowercase 'yes', 'no', or 'unknown'. If 'yes', fill 'codes_list' with example names.
    - codes_list: An array of code names if 'uses_codes' is 'yes' (e.g., ["Code Red", "Code Orange"]). Otherwise empty.
    - channels: A list of communication channels (e.g., ["text", "email", "phone", "website", "mobile app", "social media"]).
                Include all channels mentioned for that institution.
    - contact_update: Whether the institution provides a way for parents/guardians and students to update their contact
                      info for emergency notifications. Use 'yes', 'no', or 'unknown'. If a short description such as
                      "via parent portal" or "contact the school office" is provided in the answer, include that phrase.
    - reference_url: A single URL that is cited to support the provided information for this institution.

    Rules and notes:
    1) Only extract information explicitly present in the provided answer content. Do not invent.
    2) For uses_codes and contact_update, prefer normalized values 'yes' | 'no' | 'unknown' based on the answer text.
    3) If the answer lists more than four institutions, include only the first four as they appear.
    4) If any field is missing for an institution, return null for that field (or an empty array where applicable).
    5) For channels, extract each channel as a separate string and keep them simple (e.g., 'text', 'email', 'phone', 'website', 'mobile app').
    6) reference_url must be a URL explicitly present in the answer for that same institution.

    Return a JSON with a top-level field:
    {
      "institutions": [
        {
          "name": ...,
          "decision_time": ...,
          "uses_codes": ...,
          "codes_list": [...],
          "channels": [...],
          "contact_update": ...,
          "reference_url": ...
        },
        ...
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_boolish(value: Optional[str]) -> str:
    """Normalize yes/no/unknown strings; default to 'unknown' if value missing."""
    if not value:
        return "unknown"
    v = value.strip().lower()
    if v in {"yes", "y", "true"}:
        return "yes"
    if v in {"no", "n", "false"}:
        return "no"
    return "unknown"


def _join_channels(channels: List[str]) -> str:
    return ", ".join([c.strip() for c in channels if c and c.strip()]) if channels else ""


# --------------------------------------------------------------------------- #
# Verification for one institution                                            #
# --------------------------------------------------------------------------- #
async def verify_one_institution(
    evaluator: Evaluator,
    parent_node,
    inst: InstitutionItem,
    idx: int,
) -> None:
    """
    Build and verify the subtree for a single institution.
    The subtree follows the rubric:
      - Institution_{k} (parallel, non-critical)
        - Institution_{k}_Name (leaf, critical)
        - Institution_{k}_Timeline (leaf, critical)
        - Institution_{k}_Codes (leaf, critical)
        - Institution_{k}_Channels (leaf, critical)
        - Institution_{k}_Contact_Updates (leaf, critical)
        - Institution_{k}_Reference (leaf, critical)
    """
    k = idx + 1
    inst_node = evaluator.add_parallel(
        id=f"Institution_{k}",
        desc=(
            "First institution with complete weather delay policy information" if k == 1 else
            "Second institution with complete weather delay policy information" if k == 2 else
            "Third institution with complete weather delay policy information" if k == 3 else
            "Fourth institution with complete weather delay policy information"
        ),
        parent=parent_node,
        critical=False,  # Parent is non-critical to allow partial credit across institutions
    )

    # Create all leaf nodes first (so we can set prerequisites cleanly)
    name_leaf = evaluator.add_leaf(
        id=f"Institution_{k}_Name",
        desc="A valid U.S. school district or university name is provided" if k == 1 else (
            "A valid U.S. school district or university name is provided (different from Institution 1)" if k == 2 else (
                "A valid U.S. school district or university name is provided (different from Institutions 1 and 2)"
                if k == 3 else
                "A valid U.S. school district or university name is provided (different from Institutions 1, 2, and 3)"
            )
        ),
        parent=inst_node,
        critical=True
    )
    timeline_leaf = evaluator.add_leaf(
        id=f"Institution_{k}_Timeline",
        desc="The decision timeline for day-of weather closures is provided with a specific time",
        parent=inst_node,
        critical=True
    )
    codes_leaf = evaluator.add_leaf(
        id=f"Institution_{k}_Codes",
        desc="Information about whether the institution uses defined operational status codes is provided",
        parent=inst_node,
        critical=True
    )
    channels_leaf = evaluator.add_leaf(
        id=f"Institution_{k}_Channels",
        desc="The number or list of communication channels used for emergency notifications is provided",
        parent=inst_node,
        critical=True
    )
    contact_leaf = evaluator.add_leaf(
        id=f"Institution_{k}_Contact_Updates",
        desc="Information about whether the institution has a mechanism for parents/students to update contact information is provided",
        parent=inst_node,
        critical=True
    )
    ref_leaf = evaluator.add_leaf(
        id=f"Institution_{k}_Reference",
        desc="A valid URL reference supporting the information about this institution is provided",
        parent=inst_node,
        critical=True
    )

    # Evaluate the Reference leaf first; others will depend on it.
    ref_url = (inst.reference_url or "").strip() if inst and inst.reference_url else ""
    if not ref_url:
        # Fail the reference node immediately if no URL is provided
        ref_leaf.score = 0.0
        ref_leaf.status = "failed"
    else:
        ref_claim = (
            "This webpage provides publicly documented information about inclement weather closures and/or delays, or "
            "emergency notification procedures for the institution described on the page."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=ref_leaf,
            sources=ref_url,
            additional_instruction=(
                "Treat 'inclement weather', 'school closings', 'delays', 'operational status', 'emergency notifications', "
                "or similar phrases as relevant. The page should be clearly relevant to closures/delays or emergency "
                "communications. Accept district or university pages (including policy pages, frequently asked questions, "
                "or emergency alert pages)."
            )
        )

    # Extra prerequisites: gate all subsequent leaves on the reference success/failure
    prereq = [ref_leaf]

    # 1) Name leaf
    name_val = (inst.name or "").strip()
    if not name_val:
        name_leaf.score = 0.0
        name_leaf.status = "failed"
    else:
        name_claim = (
            f"This webpage is about the educational institution named '{name_val}', which is a U.S. school district "
            f"or university (or a sub-entity clearly part of that institution)."
        )
        await evaluator.verify(
            claim=name_claim,
            node=name_leaf,
            sources=ref_url if ref_url else None,
            extra_prerequisites=prereq,
            additional_instruction=(
                "Allow minor naming variants (e.g., 'Public Schools' vs 'School District', abbreviations like 'ISD', "
                "or inclusion/exclusion of city/state). Focus on whether the page is clearly about the named district "
                "or university."
            )
        )

    # 2) Timeline leaf
    time_val = (inst.decision_time or "").strip()
    if not time_val:
        timeline_leaf.score = 0.0
        timeline_leaf.status = "failed"
    else:
        timeline_claim = (
            f"Day-of weather closure or delay decisions are communicated by '{time_val}' (local time) or earlier."
        )
        await evaluator.verify(
            claim=timeline_claim,
            node=timeline_leaf,
            sources=ref_url if ref_url else None,
            extra_prerequisites=prereq,
            additional_instruction=(
                "Look for explicit timing such as 'by 5:30 a.m.' or 'no later than 6:00 AM'. If the page shows a "
                "specific time or a time-bound commitment for notifying families/staff on the day of a closure/delay, "
                "consider it a match (allow minor formatting differences like 'am' vs 'a.m.')."
            )
        )

    # 3) Codes leaf
    uses_codes_norm = _norm_boolish(inst.uses_codes)
    codes_for_text = ", ".join(inst.codes_list) if inst.codes_list else ""
    if uses_codes_norm == "unknown":
        codes_leaf.score = 0.0
        codes_leaf.status = "failed"
    elif uses_codes_norm == "yes":
        codes_claim = (
            "This institution uses defined operational status codes (e.g., named codes or levels) to communicate "
            "different types of closures or delays."
        )
        if codes_for_text:
            codes_claim = (
                f"This institution uses defined operational status codes to communicate closures or delays, such as: "
                f"{codes_for_text}."
            )
        await evaluator.verify(
            claim=codes_claim,
            node=codes_leaf,
            sources=ref_url if ref_url else None,
            extra_prerequisites=prereq,
            additional_instruction=(
                "Look for explicit named statuses (e.g., 'Code Red', 'Code Orange', 'Level 1/2/3', 'Operating Status X'). "
                "If examples are provided, ensure they appear or are clearly implied on the page."
            )
        )
    else:  # uses_codes_norm == "no"
        no_codes_claim = (
            "This institution does not use defined operational status codes to communicate closures or delays; "
            "instead, it communicates closures/delays without code labels."
        )
        await evaluator.verify(
            claim=no_codes_claim,
            node=codes_leaf,
            sources=ref_url if ref_url else None,
            extra_prerequisites=prereq,
            additional_instruction=(
                "To support this claim, the page should explicitly indicate that codes are not used, or clearly describe "
                "a communication approach without any named code scheme. Absence of evidence alone is insufficient—prefer "
                "explicit statements."
            )
        )

    # 4) Channels leaf
    channels_text = _join_channels(inst.channels)
    if not channels_text:
        channels_leaf.score = 0.0
        channels_leaf.status = "failed"
    else:
        channels_claim = (
            f"The institution uses the following communication channels for emergency notifications: {channels_text}."
        )
        await evaluator.verify(
            claim=channels_claim,
            node=channels_leaf,
            sources=ref_url if ref_url else None,
            extra_prerequisites=prereq,
            additional_instruction=(
                "Match each listed channel with the page content. Minor synonymous phrasing is acceptable "
                "(e.g., 'text' vs 'SMS', 'phone' vs 'robocall/phone call', 'mobile app' vs a named district app). "
                "It's acceptable if the page includes additional channels beyond those listed, as long as the listed "
                "ones are indeed on the page."
            )
        )

    # 5) Contact updates leaf
    contact_norm = _norm_boolish(inst.contact_update)
    if contact_norm == "unknown":
        contact_leaf.score = 0.0
        contact_leaf.status = "failed"
    elif contact_norm == "yes":
        # If a short description exists (e.g., via parent portal), incorporate it to improve matching
        detail = ""
        if inst.contact_update and inst.contact_update.strip().lower() not in {"yes", "no", "unknown"}:
            detail = f" ({inst.contact_update.strip()})"
        contact_claim = (
            "The institution provides a mechanism for parents/guardians and students to update their contact information "
            f"for emergency notifications{detail}."
        )
        await evaluator.verify(
            claim=contact_claim,
            node=contact_leaf,
            sources=ref_url if ref_url else None,
            extra_prerequisites=prereq,
            additional_instruction=(
                "Look for instructions such as updating info via a parent portal, contacting the school office, or a "
                "form to update contact details. If such a mechanism or instructions exist on the page (or linked from it), "
                "consider it supported."
            )
        )
    else:  # contact_norm == "no"
        contact_no_claim = (
            "The institution does not provide a mechanism for parents/guardians and students to update their contact "
            "information for emergency notifications."
        )
        await evaluator.verify(
            claim=contact_no_claim,
            node=contact_leaf,
            sources=ref_url if ref_url else None,
            extra_prerequisites=prereq,
            additional_instruction=(
                "Support for this negative claim requires explicit text indicating that contact information cannot be "
                "updated or that no such mechanism is available. Absence of mention is not sufficient."
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
    Evaluate an answer for the weather delay policy extraction/verification task.
    """
    # Initialize evaluator (root should be non-critical to allow partial credit across institutions)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Institutions are evaluated independently
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

    # Extract up to 4 institutions from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=InstitutionsExtraction,
        extraction_name="institutions_extraction"
    )

    # Normalize to exactly 4 entries (pad with empty items if fewer)
    institutions = list(extracted.institutions[:4])
    while len(institutions) < 4:
        institutions.append(InstitutionItem())

    # Add a small summary of extraction as custom info (not part of scoring)
    evaluator.add_custom_info(
        info={
            "extracted_count": len(extracted.institutions),
            "used_count": 4,
            "names_used": [i.name for i in institutions]
        },
        info_type="extraction_meta",
        info_name="extraction_overview"
    )

    # Build verification subtree for each institution (parallel at root)
    for idx in range(4):
        await verify_one_institution(evaluator, root, institutions[idx], idx)

    # Return structured summary including verification tree and final score
    return evaluator.get_summary()