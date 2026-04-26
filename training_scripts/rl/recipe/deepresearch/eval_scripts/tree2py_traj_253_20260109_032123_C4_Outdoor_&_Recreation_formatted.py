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
TASK_ID = "accessible_wilderness_camping_2026"
TASK_DESCRIPTION = (
    "Identify a U.S. national park that offers wilderness camping suitable for a group of 9 people "
    "(including 2 wheelchair users) planning a 5-night trip in July 2026. The park must have "
    "wheelchair-accessible wilderness camping facilities, accommodate the full group size, allow "
    "5 consecutive nights of camping, and provide an advance reservation system for permits. Provide the park name, "
    "calculate the total permit cost per person for this 5-night trip, explain the campfire policy at the accessible "
    "camping location, and include at least one reference URL from the park's official website (nps.gov domain) "
    "to support your answer."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkCampingPlan(BaseModel):
    """Structured extraction for the agent's proposed wilderness camping plan."""
    park_name: Optional[str] = None
    campsite_or_area_name: Optional[str] = None  # Zone/site/area relevant to wilderness camping
    wheelchair_accessibility: Optional[str] = None  # Summary statement of ADA/wheelchair accessibility
    group_size_supported: Optional[str] = None  # e.g., "max 9", "up to 12", or a sentence confirming 9 is allowed
    consecutive_night_limit: Optional[str] = None  # e.g., "max 5 nights", "up to 7 consecutive nights"
    july_eligibility: Optional[str] = None  # e.g., "open in July", "season runs May–October"
    reservation_system: Optional[str] = None  # e.g., "advance reservation via Recreation.gov"
    cost_per_person_total_for_5_nights: Optional[str] = None  # e.g., "$47.50"
    cost_breakdown: Optional[str] = None  # textual breakdown from the answer if any
    campfire_policy: Optional[str] = None  # short statement of campfire rules at the accessible site/area
    sources: List[str] = Field(default_factory=list)  # URLs from the answer (extract exactly)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract the proposed wilderness camping plan details exactly as stated in the answer. Return a JSON object with the following fields:
    - park_name: The specific U.S. national park named in the answer (e.g., "Yosemite National Park").
    - campsite_or_area_name: The specific wilderness/backcountry campsite/zone/area referenced for the accessible location, if mentioned; otherwise null.
    - wheelchair_accessibility: A short statement describing wheelchair accessibility or ADA-accessible features for the wilderness camping option; otherwise null.
    - group_size_supported: The text that indicates the group size capacity policy relevant to the wilderness camping (e.g., "max group size 9", "up to 12", or a sentence confirming that 9 people are allowed); otherwise null.
    - consecutive_night_limit: The text indicating the maximum consecutive nights allowed for wilderness camping (e.g., "max 5 nights"); otherwise null.
    - july_eligibility: Text indicating that camping is allowed in July (e.g., "open year-round", "season May–Oct; July is allowed"); otherwise null.
    - reservation_system: The text describing how permits are obtained and whether advance reservations are available (e.g., "advance reservations via Recreation.gov", "lottery opens months in advance"); otherwise null.
    - cost_per_person_total_for_5_nights: The total permit cost per person for the 5-night trip as explicitly stated or calculated by the answer. Include the currency symbol if present (e.g., "$52.00"). If the answer did not provide this, return null.
    - cost_breakdown: A brief textual explanation of how the answer computed the total per-person cost for 5 nights (e.g., "permit $x per night x 5 / 9 people + processing fee"); otherwise null.
    - campfire_policy: A short statement of campfire restrictions/allowances at the accessible wilderness campsite/area (e.g., "campfires prohibited above 9,600 ft", "allowed only in designated fire rings"); otherwise null.
    - sources: A list of all URLs that the answer cites as references. Extract the actual URLs exactly as they appear (including markdown links), and include only valid URLs. If no URLs are provided, return an empty list.

    If any field is not present in the answer, return null for that field. Follow the special URL rules in your system prompt.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _filter_nps_urls(urls: List[str]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and "nps.gov" in u.lower()]


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not isinstance(u, str):
            continue
        v = u.strip()
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _build_additional_instruction(base: str, urls: List[str], require_nps: bool = False) -> str:
    """
    Build an instruction for the LLM judge. If URLs are missing, force incorrect.
    Optionally require at least one NPS domain for stronger support.
    """
    urls = urls or []
    if len(urls) == 0:
        return base + "\nIMPORTANT: No URL sources are provided for this verification. You must return 'Incorrect'."
    if require_nps and not any("nps.gov" in (u.lower()) for u in urls):
        return base + "\nIMPORTANT: No official NPS (nps.gov) URL is provided. If the claim requires official NPS confirmation, return 'Incorrect'. Otherwise, only pass if a provided official page (e.g., recreation.gov) explicitly supports the claim."
    return base


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root_node, data: ParkCampingPlan) -> None:
    """
    Build verification leaves under the root and run verifications according to the rubric.
    All children are critical; failing any will fail the whole task.
    """

    # Normalize sources
    all_sources = _dedup_urls(data.sources)
    nps_sources = _filter_nps_urls(all_sources)

    # 1) Reference documentation (evaluate first to gate others if missing)
    ref_node = evaluator.add_custom_node(
        result=(len(nps_sources) >= 1),
        id="reference_documentation",
        desc="At least one official reference URL from the national park's website (nps.gov domain) is provided to support the information",
        parent=root_node,
        critical=True
    )

    # 2) Park identification
    park_node = evaluator.add_leaf(
        id="park_identification",
        desc="The specific U.S. national park name is clearly identified",
        parent=root_node,
        critical=True
    )
    park_claim = f"The identified destination is '{data.park_name}', and it is a U.S. National Park managed by the National Park Service."
    await evaluator.verify(
        claim=park_claim,
        node=park_node,
        sources=nps_sources if len(nps_sources) > 0 else all_sources,
        additional_instruction=_build_additional_instruction(
            base="Verify that the named destination is an official U.S. National Park (minor naming variations such as 'National Park & Preserve' are acceptable). Prefer NPS pages for confirmation.",
            urls=(nps_sources if len(nps_sources) > 0 else all_sources),
            require_nps=True
        )
    )

    # 3) Group capacity
    capacity_node = evaluator.add_leaf(
        id="group_capacity",
        desc="The park's wilderness camping accommodates groups of 9 people",
        parent=root_node,
        critical=True
    )
    capacity_claim = (
        f"The wilderness/backcountry camping policy for {data.park_name or 'the selected park'} "
        f"permits a party size of 9 people for the referenced location "
        f"{('('+data.campsite_or_area_name+')') if data.campsite_or_area_name else ''}."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=all_sources,
        additional_instruction=_build_additional_instruction(
            base="Check official wilderness/backcountry party size rules. The claim is that 9 people are allowed. If the max group size is below 9, or if the cited page does not clearly allow 9, return 'Incorrect'.",
            urls=all_sources
        )
    )

    # 4) Accessibility accommodation
    accessibility_node = evaluator.add_leaf(
        id="accessibility_accommodation",
        desc="The park offers wheelchair-accessible wilderness camping facilities that can accommodate at least 2 wheelchair users within the group",
        parent=root_node,
        critical=True
    )
    accessibility_claim = (
        f"The described wilderness/backcountry camping location at {data.park_name or 'the selected park'} "
        f"is wheelchair-accessible and can accommodate at least two wheelchair users."
    )
    await evaluator.verify(
        claim=accessibility_claim,
        node=accessibility_node,
        sources=all_sources,
        additional_instruction=_build_additional_instruction(
            base=(
                "Confirm that the wilderness/backcountry camping option itself is wheelchair-accessible (ADA or equivalent) "
                "and suitable for at least two wheelchair users. Look for explicit accessibility statements for the backcountry "
                "or designated accessible wilderness campsite/area; general campground accessibility evidence is insufficient."
            ),
            urls=all_sources
        )
    )

    # 5) Stay duration
    stay_node = evaluator.add_leaf(
        id="stay_duration",
        desc="The park permits wilderness camping for 5 consecutive nights during July",
        parent=root_node,
        critical=True
    )
    stay_claim = (
        f"The wilderness/backcountry camping policy at {data.park_name or 'the selected park'} permits 5 consecutive nights "
        f"of camping during July at the referenced location "
        f"{('('+data.campsite_or_area_name+')') if data.campsite_or_area_name else ''}."
    )
    await evaluator.verify(
        claim=stay_claim,
        node=stay_node,
        sources=all_sources,
        additional_instruction=_build_additional_instruction(
            base=(
                "Verify the maximum consecutive-night limit and seasonal restrictions. The claim is that 5 consecutive nights "
                "are permitted in July. If the policy caps consecutive nights below 5 or prohibits camping in July, return 'Incorrect'. "
                "Minor rounding or phrasing differences are acceptable, but the substance must match the official policy."
            ),
            urls=all_sources
        )
    )

    # 6) Advance reservation
    reserv_node = evaluator.add_leaf(
        id="advance_reservation",
        desc="The park provides an advance reservation system for wilderness camping permits (not walk-up only)",
        parent=root_node,
        critical=True
    )
    reservation_claim = (
        "Wilderness permits for this trip can be reserved in advance (not exclusively walk-up), via an official online "
        "reservation system or documented advance lottery process."
    )
    await evaluator.verify(
        claim=reservation_claim,
        node=reserv_node,
        sources=all_sources,
        additional_instruction=_build_additional_instruction(
            base=(
                "Confirm that advance reservations are available for wilderness/backcountry permits (e.g., via Recreation.gov or an NPS-run system). "
                "If the policy is strictly walk-up only with no advance reservations, return 'Incorrect'."
            ),
            urls=all_sources
        )
    )

    # 7) Cost information
    cost_node = evaluator.add_leaf(
        id="cost_information",
        desc="The total permit cost per person for the 5-night trip is calculated and provided",
        parent=root_node,
        critical=True
    )
    cost_value = (data.cost_per_person_total_for_5_nights or "").strip()
    cost_claim = (
        f"The total permit cost per person for a 5-night wilderness trip for a 9-person group is {cost_value}."
        if cost_value else
        "The answer provides a specific total permit cost per person for a 5-night wilderness trip for a 9-person group."
    )
    await evaluator.verify(
        claim=cost_claim,
        node=cost_node,
        sources=all_sources,
        additional_instruction=_build_additional_instruction(
            base=(
                "Check official permit fee pages to validate the per-person total for 5 nights. If fees are per group, convert to per person for 9; "
                "if per night, multiply appropriately for 5 nights. Allow reasonable rounding (e.g., 66.7 → 67). "
                "If the provided sources do not allow confirming the stated per-person total for 5 nights, return 'Incorrect'."
            ),
            urls=all_sources
        )
    )

    # 8) Campfire policy
    fire_node = evaluator.add_leaf(
        id="campfire_policy",
        desc="Information about campfire restrictions or allowances at the accessible wilderness campsite is provided",
        parent=root_node,
        critical=True
    )
    policy_text = (data.campfire_policy or "").strip()
    fire_claim = (
        f"The campfire policy at the referenced accessible wilderness campsite/area is: {policy_text}."
        if policy_text else
        "The answer provides the official campfire policy for the referenced accessible wilderness campsite/area."
    )
    await evaluator.verify(
        claim=fire_claim,
        node=fire_node,
        sources=all_sources,
        additional_instruction=_build_additional_instruction(
            base=(
                "Verify the wilderness/backcountry campfire rules for the stated location/area. Accept variations such as "
                "seasonal bans, elevation-based restrictions, or 'only in designated rings.' If the cited pages do not clearly "
                "support the stated policy, return 'Incorrect'."
            ),
            urls=all_sources
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
) -> Dict[str, Any]:
    """
    Evaluate the agent's answer for the accessible wilderness camping task.
    """
    # Initialize evaluator (root is parallel; children will be critical)
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
        default_model=model
    )

    # Extract structured plan from the answer
    extracted_plan = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=ParkCampingPlan,
        extraction_name="wilderness_camping_plan"
    )

    # Build tree and run verifications per rubric
    await build_and_verify_tree(evaluator, root, extracted_plan)

    # Return standard summary
    return evaluator.get_summary()