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
TASK_ID = "flagship_gaming_smartphones_2025_2026"
TASK_DESCRIPTION = """Identify four flagship gaming smartphones that were announced or released between January 2025 and March 2026 and are available for purchase in the United States market. Each smartphone must meet ALL of the following technical specifications and certification requirements:

Processor Requirements:
- Must use either Qualcomm Snapdragon 8 Elite Gen 5 or MediaTek Dimensity 9500 processor
- The processor must be manufactured using 3nm process node technology

Display Requirements:
- Minimum 120Hz refresh rate
- Minimum 480Hz touch sampling rate for gaming responsiveness
- Must use AMOLED or OLED display technology
- Must support HDR (any HDR certification acceptable)

5G Connectivity Requirements:
- Must support C-band 5G spectrum (3.7-3.98 GHz)
- Must support mmWave 5G frequencies (above 24 GHz)
- Must have FCC certification for wireless device operation in the United States

Battery Requirements:
- Minimum 5000mAh battery capacity
- Must support fast charging at minimum 65W
- Must have battery safety certification (UL or IEC 62133 compliant)

Durability Requirements:
- Minimum IP68 rating for dust and water resistance
- Must have Gorilla Glass Victus or newer generation screen protection
- MIL-STD-810H military durability certification (optional but preferred)

Gaming Features Requirements:
- Must have a dedicated cooling system (vapor chamber, liquid cooling, or active cooling fan)
- Must have touch-sensitive shoulder triggers or capacitive gaming buttons

For each of the four smartphones, provide:
1. The complete model name
2. The specific processor model used
3. Display specifications (refresh rate and touch sampling rate)
4. 5G band support details
5. Battery capacity and fast charging wattage
6. IP rating and screen protection type
7. Gaming cooling system type and trigger/button configuration
8. At least one URL reference for each major specification category (processor, display, connectivity, battery, durability, gaming features)

All specifications must be verifiable through official manufacturer specifications, reliable tech review sites, or regulatory certification databases.
"""

WINDOW_START = "2025-01-01"
WINDOW_END = "2026-03-31"
ALLOWED_PROCESSORS = ["Qualcomm Snapdragon 8 Elite Gen 5", "MediaTek Dimensity 9500"]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SmartphoneSpec(BaseModel):
    # Core identification and availability
    model_name: Optional[str] = None
    announce_or_release_date: Optional[str] = None
    us_availability: Optional[str] = None  # description indicating US availability (e.g., carriers/retailers)
    general_urls: List[str] = Field(default_factory=list)

    # Processor
    processor_model: Optional[str] = None
    process_node: Optional[str] = None  # e.g., "3nm", "TSMC N3E"
    processor_urls: List[str] = Field(default_factory=list)

    # Display
    display_refresh_rate: Optional[str] = None  # e.g., "120Hz", "144Hz"
    touch_sampling_rate: Optional[str] = None   # e.g., "480Hz", "720Hz", "1000Hz"
    display_tech: Optional[str] = None          # e.g., "AMOLED", "OLED"
    hdr_support: Optional[str] = None           # e.g., "HDR10", "HDR10+", "Dolby Vision"
    display_urls: List[str] = Field(default_factory=list)

    # Connectivity
    connectivity_cband: Optional[str] = None    # description or bands implying C-band (e.g., "n77 support")
    connectivity_mmwave: Optional[str] = None   # description or bands implying mmWave (e.g., "n258/n260/n261")
    fcc_cert: Optional[str] = None              # FCC ID string or certification link text
    connectivity_urls: List[str] = Field(default_factory=list)

    # Battery
    battery_capacity: Optional[str] = None      # e.g., "5000mAh", "5500 mAh"
    fast_charging: Optional[str] = None         # e.g., "65W", "120W"
    safety_cert: Optional[str] = None           # e.g., "UL", "IEC 62133"
    battery_urls: List[str] = Field(default_factory=list)

    # Durability
    ip_rating: Optional[str] = None             # e.g., "IP68", "IP69"
    screen_protection: Optional[str] = None     # e.g., "Gorilla Glass Victus 2"
    mil_std: Optional[str] = None               # e.g., "MIL-STD-810H" (optional)
    durability_urls: List[str] = Field(default_factory=list)

    # Gaming features
    cooling_system: Optional[str] = None        # e.g., "vapor chamber", "active cooling fan"
    gaming_triggers: Optional[str] = None       # e.g., "shoulder triggers", "capacitive gaming buttons"
    gaming_urls: List[str] = Field(default_factory=list)


class SmartphonesExtraction(BaseModel):
    phones: List[SmartphoneSpec] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_smartphones() -> str:
    return """
    Extract up to the first four flagship gaming smartphones described in the answer. For each, fill the following fields strictly from what is explicitly stated in the answer (do not infer beyond the answer). For all URL fields, return an array of actual URLs that appear in the answer text (plain or markdown). If none are given, return an empty array.

    For each smartphone, extract:
    - model_name
    - announce_or_release_date  (text as stated, e.g., "Announced Feb 2026" or "Released Mar 2025")
    - us_availability           (text indicating US availability: retailers, carriers, or official US store)
    - general_urls              (additional supporting URLs like product page, press release, retailer listing)

    Processor:
    - processor_model           (e.g., "Qualcomm Snapdragon 8 Elite Gen 5", "MediaTek Dimensity 9500")
    - process_node              (e.g., "3nm", "TSMC N3E", "3 nm")
    - processor_urls            (URLs that document the processor spec for this phone)

    Display:
    - display_refresh_rate      (e.g., "120Hz", "144Hz")
    - touch_sampling_rate       (e.g., "480Hz", "720Hz", "1000Hz"; may be called "touch response rate" or "touch sampling")
    - display_tech              (e.g., "AMOLED", "OLED")
    - hdr_support               (e.g., "HDR", "HDR10", "HDR10+", "Dolby Vision")
    - display_urls              (URLs that document display specs)

    Connectivity:
    - connectivity_cband        (text mentioning C-band support or band n77/n78 ranges that cover 3.7–3.98 GHz)
    - connectivity_mmwave       (text mentioning mmWave or bands n258/n260/n261)
    - fcc_cert                  (FCC certification ID or link text if mentioned)
    - connectivity_urls         (URLs that document 5G bands and/or FCC listing)

    Battery:
    - battery_capacity          (e.g., "5000mAh", "5500 mAh")
    - fast_charging             (e.g., "65W", "90W", "120W")
    - safety_cert               (e.g., "UL", "IEC 62133", "UL 2054")
    - battery_urls              (URLs that document battery specs and certifications)

    Durability:
    - ip_rating                 (e.g., "IP68", "IP69K")
    - screen_protection         (e.g., "Gorilla Glass Victus", "Gorilla Glass Victus 2", "Gorilla Glass Armor")
    - mil_std                   (e.g., "MIL-STD-810H", if mentioned)
    - durability_urls           (URLs that document durability and protection info)

    Gaming features:
    - cooling_system            (e.g., "vapor chamber", "liquid cooling", "active cooling fan")
    - gaming_triggers           (e.g., "shoulder triggers", "capacitive gaming buttons")
    - gaming_urls               (URLs that document gaming features)

    Return JSON with a single field:
    { "phones": [ { ...smartphone fields as specified... } ] }

    Notes:
    - If a field is not explicitly present in the answer, set it to null (or empty array for URLs).
    - Do not invent or infer missing URLs.
    - Keep all values as strings as they appear (do not normalize units).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def unique_nonempty_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for urls in url_lists:
        if not urls:
            continue
        for u in urls:
            if isinstance(u, str):
                u2 = u.strip()
                if u2 and u2 not in seen:
                    seen.add(u2)
                    out.append(u2)
    return out


def phone_desc(phone: SmartphoneSpec, idx: int) -> str:
    name = phone.model_name or f"Phone #{idx + 1}"
    return name


# --------------------------------------------------------------------------- #
# Verification logic per smartphone                                           #
# --------------------------------------------------------------------------- #
async def verify_smartphone(
    evaluator: Evaluator,
    parent_node,
    phone: SmartphoneSpec,
    idx: int,
) -> None:
    # Smartphone node (non-critical to allow partial credit across devices)
    sp_node = evaluator.add_parallel(
        id=f"smartphone_{idx + 1}",
        desc=[
            "First", "Second", "Third", "Fourth"
        ][idx] + " qualifying flagship gaming smartphone with complete specifications",
        parent=parent_node,
        critical=False
    )

    # Aggregate sources for general checks (release date & US availability)
    combined_sources = unique_nonempty_urls(
        phone.general_urls,
        phone.processor_urls, phone.display_urls, phone.connectivity_urls,
        phone.battery_urls, phone.durability_urls, phone.gaming_urls
    )

    # 1) Release window check (critical)
    rel_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_release_date",
        desc="Smartphone was announced or released between January 2025 and March 2026",
        parent=sp_node,
        critical=True
    )
    claim_release = (
        f"The smartphone model '{phone_desc(phone, idx)}' was announced or released between "
        f"January 1, 2025 and March 31, 2026 (answer states: {phone.announce_or_release_date})."
    )
    await evaluator.verify(
        claim=claim_release,
        node=rel_node,
        sources=combined_sources,
        additional_instruction=(
            "Accept 'announced' or 'released' within the window Jan 1, 2025 to Mar 31, 2026. "
            "Reject rumors/leaks without official confirmation."
        )
    )

    # 2) US availability check (critical)
    usa_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_us_availability",
        desc="Smartphone is available for purchase in the United States market",
        parent=sp_node,
        critical=True
    )
    claim_us = (
        f"The smartphone model '{phone_desc(phone, idx)}' is available for purchase in the United States market "
        f"(answer states: {phone.us_availability})."
    )
    await evaluator.verify(
        claim=claim_us,
        node=usa_node,
        sources=combined_sources,
        additional_instruction=(
            "Consider availability valid if sold via official US channels (manufacturer US store), major US retailers "
            "(e.g., Best Buy, Amazon US listing from manufacturer), or US carriers. Do not count grey-market imports."
        )
    )

    # 3) Processor (critical group)
    proc_node = evaluator.add_parallel(
        id=f"smartphone_{idx + 1}_processor",
        desc="Processor specifications meet requirements",
        parent=sp_node,
        critical=True
    )

    # 3.a) Processor source presence (critical)
    evaluator.add_custom_node(
        result=len(phone.processor_urls) > 0,
        id=f"smartphone_{idx + 1}_processor_source",
        desc="URL reference documenting processor specifications",
        parent=proc_node,
        critical=True
    )

    # 3.b) Processor model in allowed list (critical) + 3nm node (critical)
    proc_model_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_processor_model",
        desc="Uses either Qualcomm Snapdragon 8 Elite Gen 5 or MediaTek Dimensity 9500",
        parent=proc_node,
        critical=True
    )
    claim_proc_model = (
        f"The smartphone '{phone_desc(phone, idx)}' uses the processor '{phone.processor_model}', "
        f"which is either Qualcomm Snapdragon 8 Elite Gen 5 or MediaTek Dimensity 9500."
    )
    proc_model_ins = (
        "Verify the specific processor model for this phone. The claim is satisfied only if it is exactly one of: "
        f"{ALLOWED_PROCESSORS}. If the phone uses any other chip, mark as not supported. "
        "Use the provided processor URLs as evidence (manufacturer product page, spec sheets, or trusted reviews)."
    )
    await evaluator.verify(
        claim=claim_proc_model,
        node=proc_model_node,
        sources=phone.processor_urls,
        additional_instruction=proc_model_ins
    )

    proc_node_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_process_node",
        desc="Manufactured using 3nm process technology",
        parent=proc_node,
        critical=True
    )
    claim_node = (
        f"The processor '{phone.processor_model}' is manufactured on a 3nm process (e.g., TSMC N3/N3E). "
        f"Answer states: {phone.process_node}."
    )
    await evaluator.verify(
        claim=claim_node,
        node=proc_node_node,
        sources=phone.processor_urls,
        additional_instruction="Check chip vendor/manufacturer sources or authoritative reviews to confirm 3nm."
    )

    # 4) Display (critical group)
    disp_node = evaluator.add_parallel(
        id=f"smartphone_{idx + 1}_display",
        desc="Display specifications meet gaming requirements",
        parent=sp_node,
        critical=True
    )

    # Display source presence
    evaluator.add_custom_node(
        result=len(phone.display_urls) > 0,
        id=f"smartphone_{idx + 1}_display_source",
        desc="URL reference documenting display specifications",
        parent=disp_node,
        critical=True
    )

    # Display leaves
    rr_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_refresh_rate",
        desc="Minimum 120Hz refresh rate",
        parent=disp_node,
        critical=True
    )
    ts_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_touch_sampling",
        desc="Minimum 480Hz touch sampling rate for gaming responsiveness",
        parent=disp_node,
        critical=True
    )
    tech_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_display_tech",
        desc="AMOLED or OLED display technology",
        parent=disp_node,
        critical=True
    )
    hdr_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_hdr_support",
        desc="HDR display support with certification",
        parent=disp_node,
        critical=True
    )

    disp_verifications = [
        (
            f"The display refresh rate of '{phone_desc(phone, idx)}' is at least 120Hz. "
            f"Answer states: {phone.display_refresh_rate}.",
            phone.display_urls,
            rr_node,
            "Accept 120Hz or higher (e.g., 120/144/165/240Hz)."
        ),
        (
            f"The touch sampling rate of '{phone_desc(phone, idx)}' is at least 480Hz. "
            f"Answer states: {phone.touch_sampling_rate}.",
            phone.display_urls,
            ts_node,
            "Touch sampling may be labeled as 'touch response rate' or 'instant touch sampling'. Accept 480Hz or higher."
        ),
        (
            f"The display technology of '{phone_desc(phone, idx)}' is AMOLED or OLED. "
            f"Answer states: {phone.display_tech}.",
            phone.display_urls,
            tech_node,
            "Accept AMOLED or OLED (including LTPO AMOLED/OLED). LCD/Mini-LED is not acceptable."
        ),
        (
            f"The display of '{phone_desc(phone, idx)}' supports HDR with a recognized certification. "
            f"Answer states: {phone.hdr_support}.",
            phone.display_urls,
            hdr_node,
            "Accept HDR, HDR10, HDR10+, Dolby Vision, or similar recognized HDR certifications."
        )
    ]
    await evaluator.batch_verify(disp_verifications)

    # 5) Connectivity (critical group)
    conn_node = evaluator.add_parallel(
        id=f"smartphone_{idx + 1}_connectivity",
        desc="5G network compatibility meets US requirements",
        parent=sp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(phone.connectivity_urls) > 0,
        id=f"smartphone_{idx + 1}_connectivity_source",
        desc="URL reference documenting 5G capabilities and FCC certification",
        parent=conn_node,
        critical=True
    )

    cband_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_cband_support",
        desc="Supports C-band 5G spectrum (3.7-3.98 GHz)",
        parent=conn_node,
        critical=True
    )
    mmw_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_mmwave_support",
        desc="Supports mmWave 5G frequencies",
        parent=conn_node,
        critical=True
    )
    fcc_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_fcc_certification",
        desc="FCC certified for US wireless operation",
        parent=conn_node,
        critical=True
    )

    conn_verifications = [
        (
            f"The smartphone '{phone_desc(phone, idx)}' supports US C-band 5G (3.7–3.98 GHz), typically via band n77. "
            f"Answer notes: {phone.connectivity_cband}.",
            phone.connectivity_urls,
            cband_node,
            "C-band in the US primarily refers to 3.7–3.98 GHz (band n77). Accept explicit mention of C-band or n77 covering that range."
        ),
        (
            f"The smartphone '{phone_desc(phone, idx)}' supports mmWave 5G (above 24 GHz), e.g., bands n258/n260/n261. "
            f"Answer notes: {phone.connectivity_mmwave}.",
            phone.connectivity_urls,
            mmw_node,
            "Accept explicit 'mmWave' support or presence of bands n258, n260, or n261."
        ),
        (
            f"The smartphone '{phone_desc(phone, idx)}' has an FCC certification for US operation. "
            f"Answer notes FCC: {phone.fcc_cert}.",
            phone.connectivity_urls,
            fcc_node,
            "Accept an FCC ID listing or official FCC database page for this model/US variant."
        )
    ]
    await evaluator.batch_verify(conn_verifications)

    # 6) Battery (critical group)
    batt_node = evaluator.add_parallel(
        id=f"smartphone_{idx + 1}_battery",
        desc="Battery specifications meet capacity and charging requirements",
        parent=sp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(phone.battery_urls) > 0,
        id=f"smartphone_{idx + 1}_battery_source",
        desc="URL reference documenting battery specifications",
        parent=batt_node,
        critical=True
    )

    cap_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_capacity",
        desc="Minimum 5000mAh battery capacity",
        parent=batt_node,
        critical=True
    )
    fast_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_fast_charging",
        desc="Minimum 65W fast charging support",
        parent=batt_node,
        critical=True
    )
    safety_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_safety_cert",
        desc="Battery safety certification (UL or IEC 62133 compliant)",
        parent=batt_node,
        critical=True
    )

    batt_verifications = [
        (
            f"The smartphone '{phone_desc(phone, idx)}' has a battery capacity of at least 5000mAh. "
            f"Answer states: {phone.battery_capacity}.",
            phone.battery_urls,
            cap_node,
            "Accept 5000mAh or higher. Small formatting differences (e.g., spaces) are okay."
        ),
        (
            f"The smartphone '{phone_desc(phone, idx)}' supports at least 65W fast charging. "
            f"Answer states: {phone.fast_charging}.",
            phone.battery_urls,
            fast_node,
            "Accept 65W or higher (e.g., 67W, 80W, 120W)."
        ),
        (
            f"The smartphone '{phone_desc(phone, idx)}' has a battery safety certification compliant with UL or IEC 62133. "
            f"Answer states: {phone.safety_cert}.",
            phone.battery_urls,
            safety_node,
            "Accept explicit statements of UL certification (e.g., UL 2054/62368 relevant) or IEC 62133 compliance for the battery."
        )
    ]
    await evaluator.batch_verify(batt_verifications)

    # 7) Durability (critical group for mandatory protections)
    dur_node = evaluator.add_parallel(
        id=f"smartphone_{idx + 1}_durability",
        desc="Durability certifications meet protection requirements",
        parent=sp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(phone.durability_urls) > 0,
        id=f"smartphone_{idx + 1}_durability_source",
        desc="URL reference documenting durability certifications",
        parent=dur_node,
        critical=True
    )

    ip_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_ip_rating",
        desc="Minimum IP68 rating for dust and water resistance",
        parent=dur_node,
        critical=True
    )
    glass_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_screen_protection",
        desc="Gorilla Glass Victus or newer generation protection",
        parent=dur_node,
        critical=True
    )

    dur_verifications = [
        (
            f"The smartphone '{phone_desc(phone, idx)}' has an IP rating of at least IP68. "
            f"Answer states: {phone.ip_rating}.",
            phone.durability_urls,
            ip_node,
            "Accept IP68 or higher (e.g., IP68K/IP69/IP69K)."
        ),
        (
            f"The smartphone '{phone_desc(phone, idx)}' uses Corning Gorilla Glass Victus or a newer generation for screen protection. "
            f"Answer states: {phone.screen_protection}.",
            phone.durability_urls,
            glass_node,
            "Accept Gorilla Glass Victus, Victus 2, or newer branding such as 'Gorilla Glass Armor' if recognized as newer than Victus."
        )
    ]
    await evaluator.batch_verify(dur_verifications)

    # Optional MIL-STD-810H (preferred) as a separate non-critical leaf (cannot be child of a critical parent due to framework constraint)
    mil_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_mil_std",
        desc="MIL-STD-810H military durability certification (optional)",
        parent=sp_node,
        critical=False
    )
    await evaluator.verify(
        claim=(
            f"The smartphone '{phone_desc(phone, idx)}' has MIL-STD-810H (or equivalent MIL-STD-810) certification. "
            f"Answer states: {phone.mil_std}."
        ),
        node=mil_node,
        sources=phone.durability_urls,
        additional_instruction="If not mentioned or not certified, this can fail without impacting mandatory durability checks."
    )

    # 8) Gaming features (critical group)
    game_node = evaluator.add_parallel(
        id=f"smartphone_{idx + 1}_gaming_features",
        desc="Gaming-specific features meet performance requirements",
        parent=sp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(phone.gaming_urls) > 0,
        id=f"smartphone_{idx + 1}_gaming_source",
        desc="URL reference documenting gaming features",
        parent=game_node,
        critical=True
    )

    cool_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_cooling_system",
        desc="Dedicated cooling system (vapor chamber or active cooling)",
        parent=game_node,
        critical=True
    )
    trig_node = evaluator.add_leaf(
        id=f"smartphone_{idx + 1}_gaming_triggers",
        desc="Touch-sensitive shoulder triggers or capacitive gaming buttons",
        parent=game_node,
        critical=True
    )

    game_verifications = [
        (
            f"The smartphone '{phone_desc(phone, idx)}' has a dedicated gaming cooling system such as a vapor chamber, "
            f"liquid cooling, or active cooling fan. Answer states: {phone.cooling_system}.",
            phone.gaming_urls,
            cool_node,
            "Accept built-in vapor chamber/liquid cooling or an official active cooling accessory/fan that attaches for gaming."
        ),
        (
            f"The smartphone '{phone_desc(phone, idx)}' has touch-sensitive shoulder triggers or capacitive gaming buttons. "
            f"Answer states: {phone.gaming_triggers}.",
            phone.gaming_urls,
            trig_node,
            "Accept integrated shoulder triggers, capacitive gaming buttons, or equivalent touch-sensitive gaming shoulder inputs."
        )
    ]
    await evaluator.batch_verify(game_verifications)


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
    # Initialize evaluator (root as non-critical parallel to allow partial credit across devices)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify four flagship gaming smartphones released between January 2025 and March 2026 that meet all specified technical requirements for processor, display, 5G connectivity, battery, durability, and gaming features",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured smartphone data
    extracted = await evaluator.extract(
        prompt=prompt_extract_smartphones(),
        template_class=SmartphonesExtraction,
        extraction_name="smartphones_extraction"
    )

    # Ground-truth constraints (for context)
    evaluator.add_ground_truth({
        "time_window": {"start": WINDOW_START, "end": WINDOW_END},
        "allowed_processors": ALLOWED_PROCESSORS,
        "display_min_refresh_hz": 120,
        "display_min_touch_sampling_hz": 480,
        "battery_min_mAh": 5000,
        "fast_charge_min_W": 65,
        "durability_min_ip": "IP68",
        "connectivity_requirements": {
            "cband": "3.7–3.98 GHz (n77 US)",
            "mmwave": "e.g., n258/n260/n261",
            "fcc": "FCC certification present"
        }
    }, gt_type="constraints")

    # Normalize to exactly 4 smartphones (pad with empty if fewer)
    phones = list(extracted.phones or [])
    while len(phones) < 4:
        phones.append(SmartphoneSpec())
    phones = phones[:4]

    # Build verification for each smartphone
    for i in range(4):
        await verify_smartphone(evaluator, root, phones[i], i)

    # Return structured result
    return evaluator.get_summary()