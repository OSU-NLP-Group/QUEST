import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "education_job_fairs_2026_spring"
TASK_DESCRIPTION = (
    "You are a K-12 certified teacher currently exploring relocation opportunities and planning to attend education job fairs in spring 2026. "
    "Identify four in-person education or teacher recruitment job fairs that meet all of the following criteria:\n\n"
    "1. Time Requirement: The job fair must take place between February 1, 2026, and May 31, 2026 (spring 2026)\n"
    "2. Event Type: The event must be specifically designed for K-12 education professionals (teachers, administrators, or education support staff) and must be an in-person job fair (not virtual)\n"
    "3. Employer Participation: The job fair must feature multiple participating school districts or educational institutions (not a single-employer hiring event)\n"
    "4. Geographic Requirement: The job fair must be located in a state that participates in the NASDTEC Interstate Agreement\n\n"
    "For each of the four job fairs you identify, provide the following information:\n"
    "- Event Name\n- Date (or date range)\n- Venue\n- Address (including city and state)\n"
    "- Employer Participation confirmation (multiple districts or educational employers)\n"
    "- State reciprocity info (brief description)\n- Reference URL"
)
SPRING_START = datetime(2026, 2, 1)
SPRING_END = datetime(2026, 5, 31)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FairItem(BaseModel):
    event_name: Optional[str] = None
    date_text: Optional[str] = None
    start_date_iso: Optional[str] = None  # YYYY-MM-DD if available in the answer
    end_date_iso: Optional[str] = None    # YYYY-MM-DD if available in the answer
    venue_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Full state name or 2-letter abbreviation, as in answer
    in_person: Optional[bool] = None
    k12_specific: Optional[bool] = None
    employer_participation_text: Optional[str] = None
    employers_count: Optional[str] = None
    employers_examples: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)
    reciprocity_text: Optional[str] = None
    reciprocity_urls: List[str] = Field(default_factory=list)


class FairsExtraction(BaseModel):
    fairs: List[FairItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_fairs() -> str:
    return """
Extract up to four in-person education (K-12) job fairs described in the answer. For each fair, extract the fields below exactly as they appear in the answer. Do not invent information. If a field is missing, return null (or [] for arrays).

Return a JSON object with a single field 'fairs' which is an array of up to four objects. Each object must include:
- event_name: string | null
- date_text: string | null      # The date string as presented in the answer (e.g., "March 12, 2026" or "April 3–4, 2026")
- start_date_iso: string | null # If the answer clearly indicates an exact start date, normalize to YYYY-MM-DD; else null
- end_date_iso: string | null   # If the answer clearly indicates an exact end date, normalize to YYYY-MM-DD; else null (equal to start date for one-day events if explicitly clear)
- venue_name: string | null
- address: string | null        # Full street address text if present
- city: string | null
- state: string | null          # As written in the answer (full name or 2-letter code)
- in_person: boolean | null     # True if the answer explicitly indicates in-person or provides a physical venue/address; False if explicitly virtual; null if unclear
- k12_specific: boolean | null  # True if the answer explicitly states it targets K-12 educators/schools (including "PreK-12" or "PK-12"); False if explicitly higher-ed only; null if unclear
- employer_participation_text: string | null  # The phrase(s) describing multiple participating employers/districts
- employers_count: string | null              # e.g., "30+ districts", "over 50 employers"; keep as text
- employers_examples: string[]                # Up to 5 example employer names mentioned
- reference_urls: string[]                    # URL(s) cited for the event page(s)
- reciprocity_text: string | null             # Brief description of the host state's reciprocity policy as stated in the answer
- reciprocity_urls: string[]                  # URL(s) cited for state reciprocity or NASDTEC membership (must be actual URLs in the answer)

Important:
- Only extract URLs that are explicitly present in the answer text.
- Do not infer dates or URLs. If multiple events are in the answer, keep their order and take at most the first 4.
- Do not perform any extra normalization other than the optional ISO dates if explicitly clear.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _date_phrase(f: FairItem) -> str:
    if f.start_date_iso and f.end_date_iso:
        if f.start_date_iso == f.end_date_iso:
            return f.start_date_iso
        return f"{f.start_date_iso} to {f.end_date_iso}"
    return f.date_text or ""


# --------------------------------------------------------------------------- #
# Verification per fair                                                       #
# --------------------------------------------------------------------------- #
async def verify_fair(evaluator: Evaluator, parent_node, fair: FairItem, idx: int) -> None:
    """
    Build verification sub-tree for a single fair.
    """
    fair_node = evaluator.add_parallel(
        id=f"fair_{idx+1}",
        desc=f"Education job fair #{idx + 1} verification",
        parent=parent_node,
        critical=False  # each fair contributes partial credit independently
    )

    # --------------------- Reference URL block (create first for gating) ---------------------
    ref_block = evaluator.add_parallel(
        id=f"fair_{idx+1}_reference_url",
        desc="Provides a valid reference URL that supports the event information",
        parent=fair_node,
        critical=True
    )

    has_ref_urls = evaluator.add_custom_node(
        result=(len(fair.reference_urls) > 0),
        id=f"fair_{idx+1}_ref_urls_present",
        desc="At least one reference URL is provided for this fair",
        parent=ref_block,
        critical=True
    )

    # Leaf: Reference URL supports the event (name presence)
    ref_supports_event = evaluator.add_leaf(
        id=f"fair_{idx+1}_ref_supports_event",
        desc="Reference URL describes the job fair/event (name or equivalent) credibly",
        parent=ref_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page describes the education job fair named or equivalent to '{(fair.event_name or '').strip()}' with event details.",
        node=ref_supports_event,
        sources=fair.reference_urls,
        additional_instruction="Accept close naming variants (e.g., 'Education Career Fair', 'Teacher Job Fair'). The page should clearly be about the same event.",
    )

    # --------------------- Event Details block ---------------------
    details_block = evaluator.add_parallel(
        id=f"fair_{idx+1}_event_details",
        desc="Event details: event name, date within spring 2026, complete venue, and full physical address incl. city/state",
        parent=fair_node,
        critical=True
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(fair.event_name and fair.event_name.strip()),
        id=f"fair_{idx+1}_event_name_present",
        desc="Event name is provided",
        parent=details_block,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fair.date_text and fair.date_text.strip()) or bool(fair.start_date_iso),
        id=f"fair_{idx+1}_date_present",
        desc="Event date/date range is provided",
        parent=details_block,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fair.venue_name and fair.venue_name.strip()),
        id=f"fair_{idx+1}_venue_present",
        desc="Venue name is provided",
        parent=details_block,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fair.address and fair.address.strip()),
        id=f"fair_{idx+1}_address_present",
        desc="Full venue address is provided",
        parent=details_block,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fair.state and (fair.state or "").strip()),
        id=f"fair_{idx+1}_state_present",
        desc="Host state is provided (from address or event details)",
        parent=details_block,
        critical=True
    )

    # Date within Spring 2026 (LLM simple check against the answer text)
    date_in_range_leaf = evaluator.add_leaf(
        id=f"fair_{idx+1}_date_in_range",
        desc="Event date occurs between Feb 1, 2026 and May 31, 2026 (inclusive)",
        parent=details_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event date described as '{_date_phrase(fair)}' occurs between February 1, 2026 and May 31, 2026 (inclusive).",
        node=date_in_range_leaf,
        additional_instruction="Interpret common date formats and ranges (e.g., 'Mar 12–13, 2026'). If only a range is given, ensure both endpoints fall within the window.",
    )

    # Cross-check details on the event page (URL-based leaves)
    name_on_page = evaluator.add_leaf(
        id=f"fair_{idx+1}_name_on_page",
        desc="Event name matches or is equivalent on the reference page",
        parent=details_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event page indicates the job fair name is the same as or equivalent to '{(fair.event_name or '').strip()}'.",
        node=name_on_page,
        sources=fair.reference_urls,
        additional_instruction="Allow close variants and abbreviations. Focus on equivalence rather than exact punctuation/casing.",
        extra_prerequisites=[has_ref_urls],
    )

    date_on_page = evaluator.add_leaf(
        id=f"fair_{idx+1}_date_on_page",
        desc="Event date matches on the reference page",
        parent=details_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event page shows the event occurs on '{_date_phrase(fair)}' (or an equivalent date expression).",
        node=date_on_page,
        sources=fair.reference_urls,
        additional_instruction="Allow standard formatting differences (e.g., weekday names, en-dashes).",
        extra_prerequisites=[has_ref_urls],
    )

    venue_on_page = evaluator.add_leaf(
        id=f"fair_{idx+1}_venue_on_page",
        desc="Venue name matches on the reference page",
        parent=details_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event page indicates the venue is '{(fair.venue_name or '').strip()}' or an equivalent venue name.",
        node=venue_on_page,
        sources=fair.reference_urls,
        additional_instruction="Allow official venue naming variants (e.g., building complex vs. hall name).",
        extra_prerequisites=[has_ref_urls],
    )

    address_on_page = evaluator.add_leaf(
        id=f"fair_{idx+1}_address_on_page",
        desc="Physical address matches on the reference page",
        parent=details_block,
        critical=True
    )
    address_claim = f"The event page shows the venue address is '{(fair.address or '').strip()}', or at minimum the city '{(fair.city or '').strip()}' and state '{(fair.state or '').strip()}' match."
    await evaluator.verify(
        claim=address_claim,
        node=address_on_page,
        sources=fair.reference_urls,
        additional_instruction="City and state agreement is acceptable when a full street address variant appears on the page.",
        extra_prerequisites=[has_ref_urls],
    )

    in_person_leaf = evaluator.add_leaf(
        id=f"fair_{idx+1}_in_person_on_page",
        desc="Event is explicitly in-person (not virtual) on the reference page",
        parent=details_block,
        critical=True
    )
    await evaluator.verify(
        claim="This event is an in-person job fair held at a physical location, not a virtual event.",
        node=in_person_leaf,
        sources=fair.reference_urls,
        additional_instruction="Evidence can include a street address, on‑site logistics, or the phrase 'in-person'. If the page clearly indicates a physical venue, accept as in-person.",
        extra_prerequisites=[has_ref_urls],
    )

    k12_leaf = evaluator.add_leaf(
        id=f"fair_{idx+1}_k12_specific_on_page",
        desc="Event is specifically for K-12 education professionals",
        parent=details_block,
        critical=True
    )
    await evaluator.verify(
        claim="This job fair is specifically intended for K-12 educators (teachers, administrators, or K-12 school staff).",
        node=k12_leaf,
        sources=fair.reference_urls,
        additional_instruction="Accept synonyms like 'PreK-12', 'PK-12', or 'K–12'. If the page targets all educators but explicitly includes K-12 teachers, accept.",
        extra_prerequisites=[has_ref_urls],
    )

    # --------------------- Employer Participation block ---------------------
    employers_block = evaluator.add_parallel(
        id=f"fair_{idx+1}_employer_participation",
        desc="Multiple school districts or educational institutions participate (not a single-employer event)",
        parent=fair_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool((fair.employer_participation_text and fair.employer_participation_text.strip())
                    or (fair.employers_count and fair.employers_count.strip())
                    or (len(fair.employers_examples) > 0)),
        id=f"fair_{idx+1}_employer_info_present",
        desc="Employer participation info provided (text/count/examples)",
        parent=employers_block,
        critical=True
    )

    multi_emp_leaf = evaluator.add_leaf(
        id=f"fair_{idx+1}_multi_employers_on_page",
        desc="Reference page confirms multiple participating employers",
        parent=employers_block,
        critical=True
    )
    await evaluator.verify(
        claim="Multiple school districts or educational employers (more than one) are participating in this job fair.",
        node=multi_emp_leaf,
        sources=fair.reference_urls,
        additional_instruction="Look for phrases like 'districts', 'employers', 'exhibitors', lists/logos of multiple organizations, or explicit counts (e.g., '30+ districts'). Must be clearly more than one.",
        extra_prerequisites=[has_ref_urls],
    )

    # --------------------- State Reciprocity block ---------------------
    reciprocity_block = evaluator.add_parallel(
        id=f"fair_{idx+1}_state_reciprocity",
        desc="Host state is a NASDTEC member and reciprocity policy is provided",
        parent=fair_node,
        critical=True
    )

    # Existence of reciprocity URLs (for source-grounding)
    has_recip_urls = evaluator.add_custom_node(
        result=(len(fair.reciprocity_urls) > 0),
        id=f"fair_{idx+1}_reciprocity_urls_present",
        desc="At least one reciprocity/NASDTEC membership URL is provided",
        parent=reciprocity_block,
        critical=True
    )

    # Confirm host state is NASDTEC member
    host_state = (fair.state or "").strip()
    state_member_leaf = evaluator.add_leaf(
        id=f"fair_{idx+1}_state_is_nasdtec_member",
        desc="Host state is a NASDTEC Interstate Agreement member",
        parent=reciprocity_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The state of {host_state} is a participant (member/signatory) of the NASDTEC Interstate Agreement for teacher licensure reciprocity.",
        node=state_member_leaf,
        sources=fair.reciprocity_urls,
        additional_instruction="Look for explicit mention of 'NASDTEC' and the state's participation/membership.",
        extra_prerequisites=[has_recip_urls],
    )

    # Verify reciprocity policy text (high-level description)
    reciprocity_text_supported = evaluator.add_leaf(
        id=f"fair_{idx+1}_reciprocity_text_supported",
        desc="Reciprocity policy description is supported by the cited reciprocity URL(s)",
        parent=reciprocity_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The following is an accurate high-level description of {host_state}'s out-of-state teacher licensure reciprocity policy: '{(fair.reciprocity_text or '').strip()}'.",
        node=reciprocity_text_supported,
        sources=fair.reciprocity_urls,
        additional_instruction="Allow faithful paraphrases (e.g., 'accepts out-of-state licenses with 2 years experience', 'NASDTEC member; additional testing may be required'). Focus on correctness, not wording.",
        extra_prerequisites=[has_recip_urls],
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Spring 2026 K-12 in-person education job fairs task.
    """
    # Initialize evaluator (root as non-critical to allow partial credit across fairs)
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

    # Record ground-truth constraints for transparency
    evaluator.add_ground_truth({
        "time_window": {
            "start_inclusive": SPRING_START.strftime("%Y-%m-%d"),
            "end_inclusive": SPRING_END.strftime("%Y-%m-%d"),
        },
        "requirements": [
            "In-person (not virtual)",
            "K-12 specific",
            "Multiple participating employers",
            "Located in a NASDTEC member state",
            "Provide event name, date, venue, full address, reciprocity info, and reference URL(s)"
        ]
    }, gt_type="evaluation_constraints")

    # Extract structured info (up to four fairs)
    extracted: FairsExtraction = await evaluator.extract(
        prompt=prompt_extract_fairs(),
        template_class=FairsExtraction,
        extraction_name="fairs_extraction",
    )

    # Keep first four fairs; pad with placeholders if fewer than four
    fairs: List[FairItem] = list(extracted.fairs[:4])
    while len(fairs) < 4:
        fairs.append(FairItem())

    # Build verification subtrees for each fair
    verify_tasks = []
    for i, fair in enumerate(fairs):
        verify_tasks.append(verify_fair(evaluator, root, fair, i))

    # Execute verifications sequentially (can be changed to gather if desired)
    for t in verify_tasks:
        await t

    return evaluator.get_summary()