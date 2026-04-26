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
TASK_ID = "q2_2026_gaming_cons"
TASK_DESCRIPTION = (
    "Find three gaming conventions in the United States that take place during Q2 2026 (April 1 - June 30, 2026) and meet all of the following requirements:\n\n"
    "1. Time and Location: The convention must occur between April 1, 2026 and June 30, 2026, and be held in the United States.\n\n"
    "2. Duration: The convention must run for at least 3 consecutive days.\n\n"
    "3. Venue Requirements:\n"
    "   - The venue must have at least 400,000 square feet of exhibit space\n"
    "   - The venue must be wheelchair accessible and ADA-compliant\n"
    "   - The venue must be located in or within 50 miles of a major US metropolitan area (defined as a city with a population over 500,000)\n\n"
    "4. Gaming Content Requirements:\n"
    "   - Must feature organized esports tournaments or competitive gaming events\n"
    "   - Must include a dedicated indie game showcase or exhibition area for independent developers\n"
    "   - Must feature gaming content across at least 3 different gaming platforms (such as PC gaming, console gaming, tabletop/board gaming, VR gaming, or mobile gaming)\n"
    "   - Must offer a BYOC (Bring Your Own Computer) LAN party area or similar competitive PC gaming setup\n\n"
    "5. Professional Features: The convention must include at least one of the following: industry panels or talks, game developer presentations, or professional networking events for gaming industry professionals.\n\n"
    "6. Accessibility: General admission tickets must be publicly available for purchase (not invite-only or restricted events).\n\n"
    "For each of the three conventions, provide:\n"
    "- The official convention name\n"
    "- The exact dates of the convention\n"
    "- The venue name and its city/state location\n"
    "- A URL to the convention's official website or announcement page\n"
    "- A URL documenting the venue's exhibit space specifications\n"
    "- A URL documenting the esports tournaments or competitive gaming schedule\n"
    "- Confirmation that all requirements are met, with references to where this information can be verified"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConventionItem(BaseModel):
    name: Optional[str] = None
    start_date: Optional[str] = None    # Accept free-form strings (e.g., "June 12–15, 2026")
    end_date: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    venue_name: Optional[str] = None

    official_url: Optional[str] = None
    venue_spec_url: Optional[str] = None
    esports_schedule_url: Optional[str] = None

    indie_showcase_url: Optional[str] = None
    byoc_url: Optional[str] = None
    pro_feature_url: Optional[str] = None
    tickets_url: Optional[str] = None
    ada_url: Optional[str] = None

    platforms: List[str] = Field(default_factory=list)  # e.g., ["PC", "Console", "VR", "Tabletop"]

    # For metro verification (population > 500k and distance)
    metro_city: Optional[str] = None
    metro_verification_urls: List[str] = Field(default_factory=list)

    # Extra evidence links if the answer provided them
    extra_urls: List[str] = Field(default_factory=list)

    # Text snippet for checklist (optional; used to help simple verification)
    citations_checklist_text: Optional[str] = None


class ConventionsExtraction(BaseModel):
    conventions: List[ConventionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_conventions() -> str:
    return """
    Extract up to the first three gaming conventions mentioned in the answer that the agent claims meet the task requirements.
    For each convention, extract the following fields exactly as presented in the answer:
    - name: Official convention name
    - start_date: Convention start date (string)
    - end_date: Convention end date (string)
    - city: City where the venue is located
    - state: US state of the venue
    - venue_name: Venue name

    URLs (extract only valid, explicit URLs that appear in the answer; if missing, set null):
    - official_url: Convention's official website or announcement page
    - venue_spec_url: URL documenting exhibit space specs (e.g., venue spec sheet, official venue page with square footage)
    - esports_schedule_url: URL documenting esports tournaments or competitive gaming schedule/events
    - indie_showcase_url: URL evidencing a dedicated indie showcase/exhibition area (or null if not provided)
    - byoc_url: URL evidencing BYOC/LAN area (or null if not provided)
    - pro_feature_url: URL evidencing at least one professional feature (industry panels, developer presentations, or networking) (or null if not provided)
    - tickets_url: URL showing publicly available general admission tickets (or null if not provided)
    - ada_url: URL evidencing wheelchair accessibility / ADA compliance (or null if not provided)

    Other supporting info:
    - platforms: List of platform categories the answer claims are covered (e.g., ["PC", "Console", "VR", "Tabletop", "Mobile"])
    - metro_city: The major US metropolitan city used by the answer to support the "within 50 miles of a major metro >500k population" requirement (or null if not provided)
    - metro_verification_urls: URLs the answer cites to support metro population/major metro or the venue's proximity (list; may include Wikipedia/Census or maps pages)
    - extra_urls: Any additional URLs mentioned for this convention (list)
    - citations_checklist_text: If the answer includes a checklist mapping each requirement to its verification URLs for this convention, copy that checklist text snippet here; otherwise null.

    Rules:
    - If more than three conventions are present, include only the first three.
    - If fewer than three are present, return only those available.
    - Do not invent URLs or data. If a field is not provided in the answer, set it to null (or [] for list fields).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_url_list(*urls: Optional[str], extras: Optional[List[str]] = None) -> List[str]:
    seen = set()
    result: List[str] = []
    for u in urls:
        if u and isinstance(u, str):
            v = u.strip()
            if v and v not in seen:
                result.append(v)
                seen.add(v)
    if extras:
        for u in extras:
            if u and isinstance(u, str):
                v = u.strip()
                if v and v not in seen:
                    result.append(v)
                    seen.add(v)
    return result


# --------------------------------------------------------------------------- #
# Convention verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_convention(
    evaluator: Evaluator,
    parent_node,
    conv: ConventionItem,
    idx: int,
) -> None:
    """
    Build and execute verification for one convention. Each requirement is a separate critical leaf.
    If a prerequisite (e.g., a required URL) is missing and fails as a critical sibling, dependent leaves will auto-skip.
    """
    # Create a parallel node for this convention (non-critical to allow partial credit across conventions)
    conv_node = evaluator.add_parallel(
        id=f"Convention_{idx+1}",
        desc=f"Convention #{idx+1} verification",
        parent=parent_node,
        critical=False
    )

    # 1. Official Convention Name (existence)
    evaluator.add_custom_node(
        result=bool(conv.name and conv.name.strip()),
        id=f"Convention_{idx+1}_Official_Convention_Name",
        desc="Provide the official convention name.",
        parent=conv_node,
        critical=True
    )

    # 2. Exact Dates Provided (existence)
    dates_provided = bool(conv.start_date and conv.start_date.strip() and conv.end_date and conv.end_date.strip())
    evaluator.add_custom_node(
        result=dates_provided,
        id=f"Convention_{idx+1}_Exact_Dates_Provided",
        desc="Provide the exact dates of the convention.",
        parent=conv_node,
        critical=True
    )

    # 3. Occurs In US During Q2 2026 (URL-backed)
    occurs_leaf = evaluator.add_leaf(
        id=f"Convention_{idx+1}_Occurs_In_US_During_Q2_2026",
        desc="The convention occurs in the United States between April 1, 2026 and June 30, 2026.",
        parent=conv_node,
        critical=True
    )
    occurs_claim = (
        "This convention occurs in the United States between April 1, 2026 and June 30, 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=occurs_claim,
        node=occurs_leaf,
        sources=_safe_url_list(conv.official_url),
        additional_instruction=(
            f"Use the official page to confirm the venue is in the US (city/state provided: {conv.city}, {conv.state}) "
            f"and the dates fall within Q2 2026. Accept reasonable date formatting variations."
        )
    )

    # 4. Duration At Least 3 Consecutive Days (URL-backed)
    duration_leaf = evaluator.add_leaf(
        id=f"Convention_{idx+1}_Duration_At_Least_3_Consecutive_Days",
        desc="The convention runs for at least 3 consecutive days.",
        parent=conv_node,
        critical=True
    )
    await evaluator.verify(
        claim="This convention runs for at least three consecutive days.",
        node=duration_leaf,
        sources=_safe_url_list(conv.official_url),
        additional_instruction=(
            "From the official page, confirm the listed schedule spans at least three consecutive dates (e.g., Fri–Sun). "
            f"If start/end dates are provided in the answer, they are: start='{conv.start_date}', end='{conv.end_date}'. "
            "Use the page's stated schedule or date range as the primary evidence."
        )
    )

    # 5. Venue Name and City/State Provided (existence)
    venue_info_provided = bool(conv.venue_name and conv.venue_name.strip() and conv.city and conv.city.strip() and conv.state and conv.state.strip())
    evaluator.add_custom_node(
        result=venue_info_provided,
        id=f"Convention_{idx+1}_Venue_Name_and_City_State_Provided",
        desc="Provide the venue name and its city/state location.",
        parent=conv_node,
        critical=True
    )

    # 6. Venue Exhibit Space ≥ 400k sq ft (URL-backed)
    exhibit_space_leaf = evaluator.add_leaf(
        id=f"Convention_{idx+1}_Venue_Exhibit_Space_At_Least_400k",
        desc="The venue has at least 400,000 square feet of exhibit space.",
        parent=conv_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue offers at least 400,000 square feet of exhibit/exhibition space.",
        node=exhibit_space_leaf,
        sources=_safe_url_list(conv.venue_spec_url),
        additional_instruction="Check the venue specification sheet or official venue page for the total exhibit/exhibition square footage."
    )

    # 7. Venue Exhibit Space Spec URL (existence)
    evaluator.add_custom_node(
        result=bool(conv.venue_spec_url and conv.venue_spec_url.strip()),
        id=f"Convention_{idx+1}_Venue_Exhibit_Space_Spec_URL",
        desc="Provide a URL documenting the venue's exhibit space specifications.",
        parent=conv_node,
        critical=True
    )

    # 8. Venue Wheelchair Accessible and ADA-Compliant (URL-backed)
    ada_leaf = evaluator.add_leaf(
        id=f"Convention_{idx+1}_Venue_Wheelchair_Accessible_and_ADA_Compliant",
        desc="The venue is wheelchair accessible and ADA-compliant.",
        parent=conv_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue is wheelchair accessible and ADA-compliant.",
        node=ada_leaf,
        sources=_safe_url_list(conv.ada_url, conv.venue_spec_url, conv.official_url, extras=conv.extra_urls),
        additional_instruction="Use venue/official pages or accessibility statements to confirm ADA compliance and wheelchair accessibility."
    )

    # 9. Venue Within 50 Miles of Major Metro > 500k (URL-backed)
    metro_leaf = evaluator.add_leaf(
        id=f"Convention_{idx+1}_Venue_Within_50_Miles_of_Major_Metro",
        desc="The venue is located in or within 50 miles of a major US metropolitan area (population over 500,000).",
        parent=conv_node,
        critical=True
    )
    metro_claim = (
        f"The venue is located in or within 50 miles of {conv.metro_city}, which is a major US metropolitan area "
        "with population over 500,000."
    )
    await evaluator.verify(
        claim=metro_claim,
        node=metro_leaf,
        sources=_safe_url_list(conv.official_url, extras=(conv.metro_verification_urls or [])),
        additional_instruction=(
            "Use the venue location and the cited metro verification URLs (e.g., city Wikipedia/Census pages, maps) "
            "to confirm that the venue is either in the metro city or within ~50 miles, and that the metro city's population exceeds 500,000."
        )
    )

    # 10. Esports or Competitive Events (URL-backed)
    esports_leaf = evaluator.add_leaf(
        id=f"Convention_{idx+1}_Esports_or_Competitive_Events",
        desc="The convention features organized esports tournaments or competitive gaming events.",
        parent=conv_node,
        critical=True
    )
    await evaluator.verify(
        claim="This convention features organized esports tournaments or competitive gaming events.",
        node=esports_leaf,
        sources=_safe_url_list(conv.esports_schedule_url, conv.official_url),
        additional_instruction="Confirm from the esports schedule/events page or official site that competitive gaming/esports are organized at the convention."
    )

    # 11. Esports or Competitive Schedule URL (existence)
    evaluator.add_custom_node(
        result=bool(conv.esports_schedule_url and conv.esports_schedule_url.strip()),
        id=f"Convention_{idx+1}_Esports_or_Competitive_Schedule_URL",
        desc="Provide a URL documenting the esports tournaments or competitive gaming schedule/events.",
        parent=conv_node,
        critical=True
    )

    # 12. Dedicated Indie Showcase (URL-backed)
    indie_leaf = evaluator.add_leaf(
        id=f"Convention_{idx+1}_Dedicated_Indie_Showcase",
        desc="The convention includes a dedicated indie game showcase or exhibition area for independent developers.",
        parent=conv_node,
        critical=True
    )
    await evaluator.verify(
        claim="This convention includes a dedicated indie game showcase or exhibition area for independent developers.",
        node=indie_leaf,
        sources=_safe_url_list(conv.indie_showcase_url, conv.official_url, extras=conv.extra_urls),
        additional_instruction="Look for explicit mention of an indie showcase, indie pavilion, or similar dedicated area."
    )

    # 13. At Least 3 Gaming Platforms (URL-backed)
    platforms_leaf = evaluator.add_leaf(
        id=f"Convention_{idx+1}_At_Least_3_Gaming_Platforms",
        desc="The convention features gaming content across at least 3 different gaming platforms (e.g., PC, console, tabletop, VR, mobile).",
        parent=conv_node,
        critical=True
    )
    platforms_str = ", ".join(conv.platforms) if conv.platforms else "unspecified"
    await evaluator.verify(
        claim=f"The convention features gaming content across at least three distinct platforms (claimed: {platforms_str}).",
        node=platforms_leaf,
        sources=_safe_url_list(conv.official_url, extras=conv.extra_urls),
        additional_instruction=(
            "Confirm that at least three distinct platform categories are represented (e.g., PC, console, VR/AR, mobile, tabletop/board). "
            "Minor wording variations are acceptable."
        )
    )

    # 14. BYOC or Equivalent LAN Area (URL-backed)
    byoc_leaf = evaluator.add_leaf(
        id=f"Convention_{idx+1}_BYOC_or_Equivalent_LAN_Area",
        desc="The convention offers a BYOC (Bring Your Own Computer) LAN party area or similar competitive PC gaming setup.",
        parent=conv_node,
        critical=True
    )
    await evaluator.verify(
        claim="The convention offers a BYOC/LAN area or similar competitive PC gaming setup.",
        node=byoc_leaf,
        sources=_safe_url_list(conv.byoc_url, conv.official_url, extras=conv.extra_urls),
        additional_instruction="Verify that a BYOC/LAN party area or comparable PC competitive setup is offered at the event."
    )

    # 15. At Least One Professional Feature (URL-backed)
    pro_leaf = evaluator.add_leaf(
        id=f"Convention_{idx+1}_At_Least_One_Professional_Feature",
        desc="The convention includes at least one of: industry panels/talks, developer presentations, or professional networking events for gaming industry professionals.",
        parent=conv_node,
        critical=True
    )
    await evaluator.verify(
        claim="The convention includes at least one professional feature: industry panels/talks, developer presentations, or professional networking events.",
        node=pro_leaf,
        sources=_safe_url_list(conv.pro_feature_url, conv.official_url, extras=conv.extra_urls),
        additional_instruction="Confirm presence of at least one of the listed professional features from official schedules or program pages."
    )

    # 16. Public General Admission Tickets (URL-backed)
    tickets_leaf = evaluator.add_leaf(
        id=f"Convention_{idx+1}_Public_General_Admission_Tickets",
        desc="General admission tickets are publicly available for purchase (not invite-only or restricted events).",
        parent=conv_node,
        critical=True
    )
    await evaluator.verify(
        claim="General admission tickets are publicly available for purchase by the general public (not invite-only).",
        node=tickets_leaf,
        sources=_safe_url_list(conv.tickets_url, conv.official_url),
        additional_instruction="Confirm the presence of a public ticket purchase page or general admission availability."
    )

    # 17. Official Website or Announcement URL (existence)
    evaluator.add_custom_node(
        result=bool(conv.official_url and conv.official_url.strip()),
        id=f"Convention_{idx+1}_Official_Website_or_Announcement_URL",
        desc="Provide a URL to the convention's official website or announcement page.",
        parent=conv_node,
        critical=True
    )

    # 18. Citations Checklist Present (simple verification against the answer text)
    checklist_leaf = evaluator.add_leaf(
        id=f"Convention_{idx+1}_Citations_Checklist_Present",
        desc="Provide an explicit checklist/statement mapping each requirement to where it can be verified (citations/URLs), without re-stating the requirements as a separate pass/fail gate.",
        parent=conv_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The answer includes an explicit checklist for this convention that maps each requirement to one or more citations/URLs."
        ),
        node=checklist_leaf,
        sources=None,
        additional_instruction=(
            f"Look for a checklist-like section for '{conv.name or f'Convention #{idx+1}'}' in the answer that "
            "explicitly maps each requirement to citation URLs."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an agent's answer for the Q2 2026 gaming conventions task.
    """
    # Initialize evaluator (root is non-critical parallel to allow partial credit across conventions)
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

    # Extract conventions data
    extracted = await evaluator.extract(
        prompt=prompt_extract_conventions(),
        template_class=ConventionsExtraction,
        extraction_name="conventions_extraction",
    )

    # Post-process: keep only the first 3; pad if fewer (for evaluation consistency)
    conventions: List[ConventionItem] = list(extracted.conventions[:3])
    while len(conventions) < 3:
        conventions.append(ConventionItem())

    # Top-level critical check: exactly three distinct conventions provided
    names = [c.name.strip() for c in conventions if c.name and c.name.strip()]
    unique_names = set(names)
    three_distinct = (len(conventions) == 3) and (len(names) == 3) and (len(unique_names) == 3)

    evaluator.add_custom_node(
        result=three_distinct,
        id="Three_Distinct_Conventions_Provided",
        desc="Provide exactly three distinct gaming conventions (not the same event repeated).",
        parent=root,
        critical=True
    )

    # Add custom info (for transparency)
    evaluator.add_custom_info(
        info={"convention_names": names},
        info_type="extraction_summary",
        info_name="convention_names"
    )

    # Verify each convention independently (non-critical children under root for partial credit)
    for i, conv in enumerate(conventions):
        await verify_convention(evaluator, root, conv, i)

    # Return summarized evaluation
    return evaluator.get_summary()