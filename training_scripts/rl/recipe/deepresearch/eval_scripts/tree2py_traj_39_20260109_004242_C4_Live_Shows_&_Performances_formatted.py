import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nashville_broadway_venue_2026"
TASK_DESCRIPTION = (
    "A Broadway touring production company is planning to bring a large-scale musical to Nashville, Tennessee in 2026. "
    "The production requires a venue that meets specific technical specifications. Identify a performance venue in Nashville "
    "that meets ALL of the following requirements: (1) Seating capacity of at least 2,000 seats, (2) Must be a proscenium theater, "
    "(3) Proscenium opening at least 50 feet wide, (4) Stage depth of at least 40 feet from plaster line to back wall, "
    "(5) Must have an orchestra pit, (6) Must provide wheelchair-accessible seating. Provide the venue name, exact seating capacity, "
    "proscenium width in feet, stage depth in feet, and reference URL(s) that verify these specifications."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueSpecsExtraction(BaseModel):
    """
    Structured information to be extracted from the agent's answer about the selected Nashville venue.
    All fields should be directly extracted from the answer text.
    Prefer strings for numeric fields to maximize compatibility with varied formats.
    """
    venue_name: Optional[str] = None
    # Optional location fields; the verification focuses on Nashville, Tennessee
    location_city: Optional[str] = None
    location_state: Optional[str] = None

    seating_capacity: Optional[str] = None  # exact capacity provided in the answer
    proscenium_type: Optional[str] = None   # e.g., "proscenium", "end-stage", etc., as stated
    proscenium_width_ft: Optional[str] = None  # exact width in feet as stated
    stage_depth_ft: Optional[str] = None       # exact depth in feet from plaster line to back wall
    orchestra_pit: Optional[str] = None        # textual confirmation (e.g., "yes", description)
    wheelchair_accessible_seating: Optional[str] = None  # textual confirmation of accessible seating

    reference_urls: List[str] = Field(default_factory=list)  # URLs cited in the answer that verify specs


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_specs() -> str:
    return (
        "From the answer, extract details for a single identified performance venue in Nashville, Tennessee that is claimed "
        "to meet the Broadway technical requirements. If multiple venues are mentioned, choose the first one that is presented "
        "as meeting the requirements. Extract the following fields exactly as stated in the answer:\n"
        "1) venue_name: The official venue name.\n"
        "2) location_city: The city name (e.g., Nashville).\n"
        "3) location_state: The state name or abbreviation (e.g., Tennessee or TN).\n"
        "4) seating_capacity: The exact seating capacity number stated.\n"
        "5) proscenium_type: The stated stage/theatre type (e.g., 'proscenium').\n"
        "6) proscenium_width_ft: The exact proscenium opening width in feet as stated (prefer digits, but extract verbatim if needed).\n"
        "7) stage_depth_ft: The exact stage depth in feet from plaster line to back wall as stated.\n"
        "8) orchestra_pit: The statement confirming an orchestra pit (e.g., 'has an orchestra pit', 'orchestra pit available').\n"
        "9) wheelchair_accessible_seating: The statement confirming wheelchair-accessible seating.\n"
        "10) reference_urls: All URLs cited that support the venue specifications; extract only actual URLs (plain or markdown links).\n"
        "If any field is missing in the answer, return null for that field. For reference_urls, return an empty list if none are provided."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty_str(s: Optional[str]) -> bool:
    return bool(s) and bool(str(s).strip())


def _valid_urls(urls: List[str]) -> List[str]:
    """Filter URLs to likely-valid forms to avoid passing malformed strings to the verifier."""
    out = []
    for u in urls:
        if isinstance(u, str) and u.strip() and (u.strip().startswith("http://") or u.strip().startswith("https://")):
            out.append(u.strip())
    return out


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_venue_tree(
    evaluator: Evaluator,
    extracted: VenueSpecsExtraction,
) -> None:
    """
    Build the verification tree for the Nashville venue suitability and run all checks.
    The root is a critical parallel node requiring all critical children to pass.
    For criteria that combine 'value specified' and 'meets threshold / supported by sources',
    use a critical sequential sub-node with distinct leaf checks to keep each verification atomic.
    """
    # Top-level critical parallel node
    suitable_node = evaluator.add_parallel(
        id="Suitable_Venue_Nashville",
        desc="Identified venue in Nashville meets all specified requirements for hosting a Broadway touring production and all specifications are properly documented",
        parent=evaluator.root,
        critical=True
    )

    # Shared sources (reference URLs)
    sources = _valid_urls(extracted.reference_urls)

    # 1) Venue identified (existence check)
    evaluator.add_custom_node(
        result=_is_nonempty_str(extracted.venue_name),
        id="Venue_Identified",
        desc="A specific venue name is provided",
        parent=suitable_node,
        critical=True
    )

    # 2) Reference URLs provided (existence check)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Reference_URLs_Provided",
        desc="Reference URL(s) are provided that verify the venue specifications including location, seating capacity, proscenium dimensions, stage depth, orchestra pit, and accessibility features",
        parent=suitable_node,
        critical=True
    )

    # 3) Nashville location (URL-supported)
    loc_node = evaluator.add_leaf(
        id="Nashville_Location",
        desc="Venue is confirmed to be located in Nashville, Tennessee",
        parent=suitable_node,
        critical=True
    )
    venue_name = extracted.venue_name or "the venue"
    loc_claim = f"The venue '{venue_name}' is located in Nashville, Tennessee (Nashville, TN)."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=sources,
        additional_instruction="Confirm the venue is in Nashville, Tennessee. Allow 'Nashville, TN' or equivalent phrasing."
    )

    # 4) Seating capacity: sequential checks (specified + threshold + exact supported)
    cap_main = evaluator.add_sequential(
        id="Minimum_Seating_Capacity",
        desc="Venue has seating capacity of at least 2,000 seats and the exact capacity is specified",
        parent=suitable_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty_str(extracted.seating_capacity),
        id="Seating_Capacity_Specified",
        desc="Exact seating capacity is specified in the answer",
        parent=cap_main,
        critical=True
    )

    cap_threshold_node = evaluator.add_leaf(
        id="Seating_Capacity_AtLeast_2000_Supported",
        desc="Seating capacity is at least 2,000 seats (supported by cited sources)",
        parent=cap_main,
        critical=True
    )
    cap_threshold_claim = "The venue has at least 2,000 seats in seating capacity."
    await evaluator.verify(
        claim=cap_threshold_claim,
        node=cap_threshold_node,
        sources=sources,
        additional_instruction="Check that the page indicates a capacity ≥ 2000 seats. Exact number can be used to infer ≥ 2000."
    )

    cap_exact_node = evaluator.add_leaf(
        id="Seating_Capacity_Exact_Supported",
        desc="Exact seating capacity value is supported by cited sources",
        parent=cap_main,
        critical=True
    )
    cap_exact_val = extracted.seating_capacity or ""
    cap_exact_claim = f"The venue's seating capacity is {cap_exact_val} seats."
    await evaluator.verify(
        claim=cap_exact_claim,
        node=cap_exact_node,
        sources=sources,
        additional_instruction="Verify the exact capacity number stated in the answer matches the source(s). Minor formatting variations are acceptable."
    )

    # 5) Proscenium theater type: sequential checks (specified + supported)
    pros_type_main = evaluator.add_sequential(
        id="Proscenium_Theater_Type",
        desc="Venue is a proscenium theater suitable for Broadway productions",
        parent=suitable_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty_str(extracted.proscenium_type),
        id="Proscenium_Type_Specified",
        desc="Proscenium theater type is specified in the answer",
        parent=pros_type_main,
        critical=True
    )

    pros_type_node = evaluator.add_leaf(
        id="Proscenium_Type_Supported",
        desc="Venue is a proscenium theater (supported by cited sources)",
        parent=pros_type_main,
        critical=True
    )
    pros_type_claim = f"The venue '{venue_name}' is a proscenium theater."
    await evaluator.verify(
        claim=pros_type_claim,
        node=pros_type_node,
        sources=sources,
        additional_instruction="Confirm the venue has a proscenium stage/stage house (e.g., 'proscenium', 'proscenium arch')."
    )

    # 6) Proscenium width: sequential checks (specified + threshold + exact supported)
    pros_width_main = evaluator.add_sequential(
        id="Proscenium_Width",
        desc="Proscenium opening is at least 50 feet wide and the exact width in feet is specified",
        parent=suitable_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty_str(extracted.proscenium_width_ft),
        id="Proscenium_Width_Specified",
        desc="Exact proscenium opening width (feet) is specified in the answer",
        parent=pros_width_main,
        critical=True
    )

    pros_width_threshold_node = evaluator.add_leaf(
        id="Proscenium_Width_AtLeast_50_Supported",
        desc="Proscenium opening is at least 50 feet wide (supported by cited sources)",
        parent=pros_width_main,
        critical=True
    )
    pros_width_threshold_claim = "The proscenium opening is at least 50 feet wide."
    await evaluator.verify(
        claim=pros_width_threshold_claim,
        node=pros_width_threshold_node,
        sources=sources,
        additional_instruction="Confirm the proscenium opening width is ≥ 50 ft. Use the stated number on the page."
    )

    pros_width_exact_node = evaluator.add_leaf(
        id="Proscenium_Width_Exact_Supported",
        desc="Exact proscenium opening width value is supported by cited sources",
        parent=pros_width_main,
        critical=True
    )
    pros_width_exact_val = extracted.proscenium_width_ft or ""
    pros_width_exact_claim = f"The proscenium opening width is {pros_width_exact_val} feet."
    await evaluator.verify(
        claim=pros_width_exact_claim,
        node=pros_width_exact_node,
        sources=sources,
        additional_instruction="Verify the exact width (in feet) stated in the answer matches the source(s)."
    )

    # 7) Stage depth: sequential checks (specified + threshold + exact supported)
    stage_depth_main = evaluator.add_sequential(
        id="Stage_Depth",
        desc="Stage depth from plaster line is at least 40 feet and the exact depth in feet is specified",
        parent=suitable_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty_str(extracted.stage_depth_ft),
        id="Stage_Depth_Specified",
        desc="Exact stage depth from plaster line to back wall (feet) is specified in the answer",
        parent=stage_depth_main,
        critical=True
    )

    stage_depth_threshold_node = evaluator.add_leaf(
        id="Stage_Depth_AtLeast_40_Supported",
        desc="Stage depth from plaster line to back wall is at least 40 feet (supported by cited sources)",
        parent=stage_depth_main,
        critical=True
    )
    stage_depth_threshold_claim = "The stage depth from the plaster line to the back wall is at least 40 feet."
    await evaluator.verify(
        claim=stage_depth_threshold_claim,
        node=stage_depth_threshold_node,
        sources=sources,
        additional_instruction="Confirm the stage depth (plaster line to back wall) is ≥ 40 ft."
    )

    stage_depth_exact_node = evaluator.add_leaf(
        id="Stage_Depth_Exact_Supported",
        desc="Exact stage depth from plaster line value is supported by cited sources",
        parent=stage_depth_main,
        critical=True
    )
    stage_depth_exact_val = extracted.stage_depth_ft or ""
    stage_depth_exact_claim = f"The stage depth from the plaster line to the back wall is {stage_depth_exact_val} feet."
    await evaluator.verify(
        claim=stage_depth_exact_claim,
        node=stage_depth_exact_node,
        sources=sources,
        additional_instruction="Verify the exact stage depth value stated in the answer matches the source(s)."
    )

    # 8) Orchestra pit availability: sequential checks (specified + supported)
    pit_main = evaluator.add_sequential(
        id="Orchestra_Pit_Available",
        desc="Venue has an orchestra pit that can accommodate musicians",
        parent=suitable_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty_str(extracted.orchestra_pit),
        id="Orchestra_Pit_Specified",
        desc="Orchestra pit availability is specified in the answer",
        parent=pit_main,
        critical=True
    )

    pit_supported_node = evaluator.add_leaf(
        id="Orchestra_Pit_Supported",
        desc="Orchestra pit availability is supported by cited sources",
        parent=pit_main,
        critical=True
    )
    pit_claim = f"The venue '{venue_name}' has an orchestra pit suitable for accommodating musicians."
    await evaluator.verify(
        claim=pit_claim,
        node=pit_supported_node,
        sources=sources,
        additional_instruction="Confirm the venue has an orchestra pit. It may be fixed or removable; verify availability for musicians."
    )

    # 9) Wheelchair accessibility: sequential checks (specified + supported)
    accessibility_main = evaluator.add_sequential(
        id="Wheelchair_Accessibility",
        desc="Venue provides wheelchair-accessible seating",
        parent=suitable_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty_str(extracted.wheelchair_accessible_seating),
        id="Wheelchair_Accessibility_Specified",
        desc="Wheelchair-accessible seating is specified in the answer",
        parent=accessibility_main,
        critical=True
    )

    accessibility_supported_node = evaluator.add_leaf(
        id="Wheelchair_Accessibility_Supported",
        desc="Wheelchair-accessible seating availability is supported by cited sources",
        parent=accessibility_main,
        critical=True
    )
    accessibility_claim = f"The venue '{venue_name}' provides wheelchair-accessible seating."
    await evaluator.verify(
        claim=accessibility_claim,
        node=accessibility_supported_node,
        sources=sources,
        additional_instruction="Confirm that the venue offers wheelchair-accessible seating or ADA-compliant seating areas."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
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
    """
    Evaluate the agent's answer for the Nashville Broadway venue suitability task.
    Returns the standard evaluation summary dict produced by the Evaluator.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation; the critical child will gate results
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

    # Extract structured venue specs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_specs(),
        template_class=VenueSpecsExtraction,
        extraction_name="venue_specs"
    )

    # Optional: Record criteria summary for transparency
    evaluator.add_custom_info(
        info={
            "requirements": {
                "seating_capacity_min": 2000,
                "proscenium_required": True,
                "proscenium_width_min_ft": 50,
                "stage_depth_min_ft": 40,
                "orchestra_pit_required": True,
                "wheelchair_accessible_seating_required": True
            }
        },
        info_type="criteria_summary",
        info_name="broadway_venue_requirements"
    )

    # Build verification tree and run checks
    await build_and_verify_venue_tree(evaluator, extracted)

    # Return unified summary
    return evaluator.get_summary()