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
TASK_ID = "gaming_conventions_and_vr_2026"
TASK_DESCRIPTION = """
Identify three gaming conventions taking place in the United States during 2026. For each convention, provide the following information:

1. Convention Identification:
   - Official convention name
   - Convention type or theme (e.g., gaming, anime, comics, general pop culture)
   - Reference URL

2. Date Information:
   - Start date (in format: Month Day, Year)
   - End date (in format: Month Day, Year)
   - Reference URL verifying the dates

3. Venue Information:
   - Official venue name
   - City where the venue is located
   - US state where the venue is located
   - Reference URL verifying the venue location
   - Reference URL for the venue name

Additionally, identify two VR (Virtual Reality) headsets currently available or announced for 2026. For each VR headset, provide:

1. Headset Identification:
   - Manufacturer name
   - Official model name
   - Reference URL

2. Display Specifications:
   - Horizontal resolution per eye (in pixels)
   - Vertical resolution per eye (in pixels)
   - Maximum refresh rate (in Hz)
   - Display technology type (e.g., LCD, OLED)
   - Reference URL verifying resolution specifications
   - Reference URL for refresh rate and display technology

3. Field of View: (optional, if available)

All conventions must be held within the United States during the calendar year 2026, and all information must be supported by reference URLs from official or reliable sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConventionItem(BaseModel):
    # Identification
    name: Optional[str] = None
    type: Optional[str] = None
    name_ref_url: Optional[str] = None  # URL reference for the convention name and type
    # Dates
    start_date: Optional[str] = None  # Month Day, Year
    end_date: Optional[str] = None    # Month Day, Year
    date_ref_url: Optional[str] = None
    # Venue
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_state: Optional[str] = None  # Allow full name or 2-letter code
    location_ref_url: Optional[str] = None  # URL verifying the venue location (city/state)
    venue_ref_url: Optional[str] = None     # URL reference for the venue name


class ConventionsExtraction(BaseModel):
    conventions: List[ConventionItem] = Field(default_factory=list)


class VRHeadsetItem(BaseModel):
    # Identification
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    identification_url: Optional[str] = None
    # Display specs
    horiz_res_per_eye: Optional[str] = None
    vert_res_per_eye: Optional[str] = None
    resolution_ref_url: Optional[str] = None
    refresh_rate_hz: Optional[str] = None
    display_tech: Optional[str] = None  # e.g., LCD, OLED, micro-OLED, mini-LED LCD, etc.
    display_specs_ref_url: Optional[str] = None
    # Optional
    field_of_view: Optional[str] = None


class VRExtraction(BaseModel):
    headsets: List[VRHeadsetItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_conventions() -> str:
    return """
    Extract up to three gaming conventions in the United States during 2026 that are mentioned in the answer.
    For each convention, extract the following fields exactly as they appear in the answer:
    - name: The official convention name (string)
    - type: The convention type or theme (e.g., "gaming", "anime", "comics", "pop culture") (string)
    - name_ref_url: The URL cited for the convention name and/or type (string URL; if none provided, set to null)

    - start_date: The convention's start date in the format "Month Day, Year" if available in the answer (string)
    - end_date: The convention's end date in the format "Month Day, Year" if available in the answer (string)
    - date_ref_url: The URL cited for the date range verification (string URL; if none provided, set to null)

    - venue_name: The official venue/facility name (string)
    - venue_city: The city where the venue is located (string)
    - venue_state: The US state where the venue is located (can be full name like "California" or two-letter code like "CA") (string)
    - location_ref_url: The URL cited for verifying the venue location (city/state) (string URL; if none, set to null)
    - venue_ref_url: The URL cited for the venue name (string URL; if none, set to null)

    IMPORTANT:
    - Only extract what is explicitly present in the answer. Do not invent or infer missing information.
    - For any missing field, return null.
    - For URLs, extract actual URLs as they appear (including those inside markdown links).
    - Return a JSON object with a top-level "conventions" array of objects with the above fields.
    """


def prompt_extract_vr_headsets() -> str:
    return """
    Extract up to two VR headsets that are currently available or announced for 2026 as mentioned in the answer.
    For each headset, extract the following fields exactly as they appear in the answer:

    IDENTIFICATION:
    - manufacturer: The company that makes the headset (string)
    - model: The official model name (string)
    - identification_url: URL cited for the manufacturer and/or model identification (string URL; if none, set to null)

    DISPLAY SPECIFICATIONS:
    - horiz_res_per_eye: Horizontal pixels per eye (string; keep units or descriptors if present, e.g., "2160" or "2K")
    - vert_res_per_eye: Vertical pixels per eye (string)
    - resolution_ref_url: URL cited for resolution details (string URL; if none, set to null)
    - refresh_rate_hz: Maximum refresh rate in Hz (string; e.g., "90 Hz", "up to 120 Hz")
    - display_tech: Display technology (e.g., "LCD", "OLED", "micro-OLED") (string)
    - display_specs_ref_url: URL cited for refresh rate and display technology (string URL; if none, set to null)

    OPTIONAL:
    - field_of_view: Field of view value if provided (string; any format, e.g., "~110°")

    IMPORTANT:
    - Only extract what is explicitly present in the answer. Do not invent or infer missing information.
    - For any missing field, return null.
    - For URLs, extract actual URLs as they appear (including those inside markdown links).
    - Return a JSON object with a top-level "headsets" array of objects with the above fields.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n+1}"


def _non_empty_url(url: Optional[str]) -> bool:
    return bool(url and isinstance(url, str) and url.strip().lower().startswith(("http://", "https://")))


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_convention(
    evaluator: Evaluator,
    parent_node,
    item: ConventionItem,
    idx: int,
) -> None:
    """
    Build verification sub-tree and perform checks for a single convention.
    Tree structure follows the rubric: Identification, Date Information, Venue Information.
    """
    conv_node = evaluator.add_parallel(
        id=f"gaming_convention_{idx+1}",
        desc=f"{ordinal(idx)} gaming convention in the United States during 2026",
        parent=parent_node,
        critical=False
    )

    # -------------------- 1) Convention Identification (Critical group) --------------------
    ident_node = evaluator.add_parallel(
        id=f"con_{idx+1}_identification",
        desc="Basic identification information for the convention",
        parent=conv_node,
        critical=True
    )

    # Name/Type reference URL provided (critical gating)
    evaluator.add_custom_node(
        result=_non_empty_url(item.name_ref_url),
        id=f"con_{idx+1}_name_ref_url",
        desc="URL reference for the convention name and type is provided",
        parent=ident_node,
        critical=True
    )

    # Convention Name (critical leaf) - grounded to name_ref_url
    name_leaf = evaluator.add_leaf(
        id=f"con_{idx+1}_name",
        desc="Official name of the gaming convention",
        parent=ident_node,
        critical=True
    )
    name_claim = f"The official name of the convention is '{(item.name or '').strip()}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=item.name_ref_url,
        additional_instruction=(
            "Verify that the page clearly names the event. Allow minor variations such as inclusion of the year "
            "or edition (e.g., '2026' or 'Expo 2026'). Focus on confirming the core brand/event name."
        ),
    )

    # Convention Type/Theme (critical leaf) - grounded to name_ref_url
    type_leaf = evaluator.add_leaf(
        id=f"con_{idx+1}_type",
        desc="Type or theme of the convention (e.g., gaming, anime, comics, general pop culture)",
        parent=ident_node,
        critical=True
    )
    type_claim = f"The convention's type/theme is '{(item.type or '').strip()}', and it is presented as such on the reference page."
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=item.name_ref_url,
        additional_instruction=(
            "Confirm the stated theme/type on the page. Accept close synonyms or broader categories "
            "when reasonable (e.g., 'video game' ≈ 'gaming', 'pop culture' that explicitly includes gaming)."
        ),
    )

    # -------------------- 2) Date Information (Critical group) --------------------
    date_node = evaluator.add_parallel(
        id=f"con_{idx+1}_dates",
        desc="Complete date range for the convention",
        parent=conv_node,
        critical=True
    )

    # Date reference URL provided (critical gating)
    evaluator.add_custom_node(
        result=_non_empty_url(item.date_ref_url),
        id=f"con_{idx+1}_date_ref_url",
        desc="URL reference verifying the convention dates is provided",
        parent=date_node,
        critical=True
    )

    # Start date (critical leaf) - grounded to date_ref_url
    start_leaf = evaluator.add_leaf(
        id=f"con_{idx+1}_start_date",
        desc="Convention start date in format: Month Day, Year",
        parent=date_node,
        critical=True
    )
    start_claim = (
        f"The start date of the 2026 edition of {(item.name or 'the convention').strip()} is "
        f"'{(item.start_date or '').strip()}'."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=item.date_ref_url,
        additional_instruction=(
            "Verify the start date specifically for the year 2026 if multiple years are shown on the page. "
            "The format may vary slightly; allow reasonable formatting variants matching the same calendar date."
        ),
    )

    # End date (critical leaf) - grounded to date_ref_url
    end_leaf = evaluator.add_leaf(
        id=f"con_{idx+1}_end_date",
        desc="Convention end date in format: Month Day, Year",
        parent=date_node,
        critical=True
    )
    end_claim = (
        f"The end date of the 2026 edition of {(item.name or 'the convention').strip()} is "
        f"'{(item.end_date or '').strip()}'."
    )
    await evaluator.verify(
        claim=end_claim,
        node=end_leaf,
        sources=item.date_ref_url,
        additional_instruction=(
            "Verify the end date specifically for the year 2026 if multiple years are shown on the page. "
            "The format may vary slightly; allow reasonable formatting variants matching the same calendar date."
        ),
    )

    # -------------------- 3) Venue Information (Critical group) --------------------
    venue_node = evaluator.add_parallel(
        id=f"con_{idx+1}_venue",
        desc="Complete venue and location details",
        parent=conv_node,
        critical=True
    )

    # Venue reference URL provided (critical gating for venue name)
    evaluator.add_custom_node(
        result=_non_empty_url(item.venue_ref_url),
        id=f"con_{idx+1}_venue_ref_url",
        desc="URL reference for the venue name is provided",
        parent=venue_node,
        critical=True
    )

    # Venue name (critical leaf) - grounded to venue_ref_url
    venue_leaf = evaluator.add_leaf(
        id=f"con_{idx+1}_venue_name",
        desc="Official name of the venue/facility hosting the convention",
        parent=venue_node,
        critical=True
    )
    venue_claim = (
        f"The official venue for the 2026 edition of {(item.name or 'the convention').strip()} "
        f"is '{(item.venue_name or '').strip()}'."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        sources=item.venue_ref_url,
        additional_instruction=(
            "Confirm the venue/facility name explicitly on the provided source. "
            "Allow minor naming variants (e.g., inclusion/exclusion of 'Convention Center', abbreviations)."
        ),
    )

    # Complete location (Critical sub-group)
    loc_node = evaluator.add_parallel(
        id=f"con_{idx+1}_complete_location",
        desc="Geographic location of the venue",
        parent=venue_node,
        critical=True
    )

    # Location reference URL provided (critical gating for city/state)
    evaluator.add_custom_node(
        result=_non_empty_url(item.location_ref_url),
        id=f"con_{idx+1}_location_ref_url",
        desc="URL reference verifying the venue location is provided",
        parent=loc_node,
        critical=True
    )

    # City (critical leaf) - grounded to location_ref_url
    city_leaf = evaluator.add_leaf(
        id=f"con_{idx+1}_city",
        desc="City where the venue is located",
        parent=loc_node,
        critical=True
    )
    city_claim = (
        f"The venue for {(item.name or 'the convention').strip()} is located in the city of "
        f"'{(item.venue_city or '').strip()}', United States."
    )
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=item.location_ref_url,
        additional_instruction=(
            "Verify the city in the venue or event address. Accept formats like 'City, ST' or full address lines."
        ),
    )

    # State (critical leaf) - grounded to location_ref_url
    state_leaf = evaluator.add_leaf(
        id=f"con_{idx+1}_state",
        desc="US state where the venue is located",
        parent=loc_node,
        critical=True
    )
    state_claim = (
        f"The venue for {(item.name or 'the convention').strip()} is located in the U.S. state of "
        f"'{(item.venue_state or '').strip()}'."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=item.location_ref_url,
        additional_instruction=(
            "Verify the U.S. state (either full name or two-letter abbreviation) "
            "in the venue's official address or event location details."
        ),
    )


async def verify_vr_headset(
    evaluator: Evaluator,
    parent_node,
    item: VRHeadsetItem,
    idx: int,
) -> None:
    """
    Build verification sub-tree and perform checks for a single VR headset.
    Tree structure follows the rubric: Identification, Display Specifications, Field of View (optional).
    """
    vr_node = evaluator.add_parallel(
        id=f"vr_headset_{idx+1}",
        desc=f"{ordinal(idx)} VR headset with complete technical specifications",
        parent=parent_node,
        critical=False
    )

    # -------------------- 1) Headset Identification (Critical group) --------------------
    ident_node = evaluator.add_parallel(
        id=f"vr_{idx+1}_identification",
        desc="Manufacturer and model identification",
        parent=vr_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_url(item.identification_url),
        id=f"vr_{idx+1}_ident_ref_url",
        desc="URL reference for the headset manufacturer and model is provided",
        parent=ident_node,
        critical=True
    )

    # Manufacturer (critical leaf) - grounded to identification_url
    mfg_leaf = evaluator.add_leaf(
        id=f"vr_{idx+1}_manufacturer",
        desc="Name of the company that manufactures the VR headset",
        parent=ident_node,
        critical=True
    )
    mfg_claim = (
        f"The manufacturer of the headset model '{(item.model or '').strip()}' is "
        f"'{(item.manufacturer or '').strip()}'."
    )
    await evaluator.verify(
        claim=mfg_claim,
        node=mfg_leaf,
        sources=item.identification_url,
        additional_instruction=(
            "Confirm the company/brand responsible for the headset on the provided source."
        ),
    )

    # Model name (critical leaf) - grounded to identification_url
    model_leaf = evaluator.add_leaf(
        id=f"vr_{idx+1}_model",
        desc="Official model name or designation of the VR headset",
        parent=ident_node,
        critical=True
    )
    model_claim = f"The official model name of the VR headset is '{(item.model or '').strip()}'."
    await evaluator.verify(
        claim=model_claim,
        node=model_leaf,
        sources=item.identification_url,
        additional_instruction=(
            "Confirm the official product/model name for this headset. "
            "Allow minor punctuation or spacing differences."
        ),
    )

    # -------------------- 2) Display Specifications (Critical group) --------------------
    disp_node = evaluator.add_parallel(
        id=f"vr_{idx+1}_display_specs",
        desc="Complete display technical specifications",
        parent=vr_node,
        critical=True
    )

    # 2.1 Resolution details (Critical sub-group)
    res_node = evaluator.add_parallel(
        id=f"vr_{idx+1}_resolution_details",
        desc="Per-eye display resolution in pixels",
        parent=disp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_url(item.resolution_ref_url),
        id=f"vr_{idx+1}_resolution_ref_url",
        desc="URL reference verifying the resolution specifications is provided",
        parent=res_node,
        critical=True
    )

    # Horizontal per-eye resolution (critical leaf) - grounded to resolution_ref_url
    horiz_leaf = evaluator.add_leaf(
        id=f"vr_{idx+1}_horiz_res",
        desc="Horizontal resolution per eye in pixels",
        parent=res_node,
        critical=True
    )
    horiz_claim = (
        f"The per-eye horizontal display resolution is '{(item.horiz_res_per_eye or '').strip()}' pixels."
    )
    await evaluator.verify(
        claim=horiz_claim,
        node=horiz_leaf,
        sources=item.resolution_ref_url,
        additional_instruction=(
            "Verify the per-eye horizontal resolution. Accept formats like '2160', "
            "'2160 px', or part of '2160 x 2160 per eye'."
        ),
    )

    # Vertical per-eye resolution (critical leaf) - grounded to resolution_ref_url
    vert_leaf = evaluator.add_leaf(
        id=f"vr_{idx+1}_vert_res",
        desc="Vertical resolution per eye in pixels",
        parent=res_node,
        critical=True
    )
    vert_claim = (
        f"The per-eye vertical display resolution is '{(item.vert_res_per_eye or '').strip()}' pixels."
    )
    await evaluator.verify(
        claim=vert_claim,
        node=vert_leaf,
        sources=item.resolution_ref_url,
        additional_instruction=(
            "Verify the per-eye vertical resolution. Accept formats like '2160', "
            "'2160 px', or part of '2160 x 2160 per eye'."
        ),
    )

    # 2.2 Refresh rate (critical leaf) - grounded to display_specs_ref_url
    evaluator.add_custom_node(
        result=_non_empty_url(item.display_specs_ref_url),
        id=f"vr_{idx+1}_display_specs_ref_url",
        desc="URL reference for refresh rate and display technology is provided",
        parent=disp_node,
        critical=True
    )

    refresh_leaf = evaluator.add_leaf(
        id=f"vr_{idx+1}_refresh_rate",
        desc="Maximum refresh rate in Hz",
        parent=disp_node,
        critical=True
    )
    refresh_claim = (
        f"The headset's maximum refresh rate is '{(item.refresh_rate_hz or '').strip()}'."
    )
    await evaluator.verify(
        claim=refresh_claim,
        node=refresh_leaf,
        sources=item.display_specs_ref_url,
        additional_instruction=(
            "Verify the maximum refresh rate. Accept phrasing such as 'up to 120 Hz', 'max 120Hz', or similar."
        ),
    )

    # 2.3 Display technology (critical leaf) - grounded to display_specs_ref_url
    tech_leaf = evaluator.add_leaf(
        id=f"vr_{idx+1}_display_tech",
        desc="Type of display technology (e.g., LCD, OLED)",
        parent=disp_node,
        critical=True
    )
    tech_claim = f"The display technology used is '{(item.display_tech or '').strip()}'."
    await evaluator.verify(
        claim=tech_claim,
        node=tech_leaf,
        sources=item.display_specs_ref_url,
        additional_instruction=(
            "Verify the display type (e.g., LCD, OLED, micro-OLED, mini-LED LCD). "
            "Allow close variants or additional qualifiers."
        ),
    )

    # -------------------- 3) Field of View (Optional, non-critical) --------------------
    # Add the optional node only if FOV is provided and at least one plausible source exists, to avoid penalizing missing optional data.
    if item.field_of_view and any(
        _non_empty_url(u) for u in [item.display_specs_ref_url, item.resolution_ref_url, item.identification_url]
    ):
        # Prefer display specs URL, then resolution URL, then identification URL
        fov_source = item.display_specs_ref_url or item.resolution_ref_url or item.identification_url
        fov_leaf = evaluator.add_leaf(
            id=f"vr_{idx+1}_field_of_view",
            desc="Field of view specification if available",
            parent=vr_node,
            critical=False
        )
        fov_claim = f"The headset offers a field of view of '{item.field_of_view.strip()}' (approximate values acceptable)."
        await evaluator.verify(
            claim=fov_claim,
            node=fov_leaf,
            sources=fov_source,
            additional_instruction=(
                "Verify the field of view as stated on the page. Accept approximate or range values (e.g., '~110°', "
                "'up to 100°', 'horizontal FOV 100°')."
            ),
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
    Evaluate an answer for: three US gaming conventions in 2026 and two VR headsets for 2026.
    Returns a structured summary with verification tree and final score.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation as per rubric
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

    # Perform extractions (in parallel)
    conventions_extraction_task = evaluator.extract(
        prompt=prompt_extract_conventions(),
        template_class=ConventionsExtraction,
        extraction_name="conventions_extraction",
    )
    vr_extraction_task = evaluator.extract(
        prompt=prompt_extract_vr_headsets(),
        template_class=VRExtraction,
        extraction_name="vr_headsets_extraction",
    )
    conventions_extraction, vr_extraction = await asyncio.gather(
        conventions_extraction_task, vr_extraction_task
    )

    # Normalize number of items: exactly 3 conventions and 2 headsets (padding with empty if needed)
    conventions: List[ConventionItem] = list(conventions_extraction.conventions[:3])
    while len(conventions) < 3:
        conventions.append(ConventionItem())

    headsets: List[VRHeadsetItem] = list(vr_extraction.headsets[:2])
    while len(headsets) < 2:
        headsets.append(VRHeadsetItem())

    # Build verification for each convention
    for i in range(3):
        await verify_convention(evaluator, root, conventions[i], i)

    # Build verification for each VR headset
    for i in range(2):
        await verify_vr_headset(evaluator, root, headsets[i], i)

    # Return the evaluation summary
    return evaluator.get_summary()