import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tn_state_parks_campgrounds"
TASK_DESCRIPTION = """
Identify Tennessee state park campground(s) that meet ALL of the following requirements:

1. Offers full hookup campsites with sewer, water, and electrical connections
2. Provides electrical hookups with multiple amperage options (20-amp, 30-amp, and 50-amp)
3. Has bathhouse facilities with hot showers
4. Provides a dump station for RV waste disposal
5. Offers laundry facilities (washer and dryer access)
6. Has a camp store that sells firewood, ice, and camping supplies
7. Provides Wi-Fi internet access at the campground
8. Operates year-round for camping
9. Allows pets (dogs) on leash
10. Accepts reservations through the Tennessee State Parks online reservation system
11. Offers ADA-accessible campsites
12. Is located on or provides direct access to a lake
13. Has at least 3 different designated campground sections or areas
14. Has at least 100 total campsites available

For each qualifying campground you identify, provide:
- The name of the state park
- The specific lake it is located on (if applicable)
- The number of campground sections
- The total number of campsites
- Reference URL(s) that verify the amenities and features
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FeatureFlags(BaseModel):
    full_hookup: Optional[bool] = None
    amp_20: Optional[bool] = None
    amp_30: Optional[bool] = None
    amp_50: Optional[bool] = None
    bathhouse_hot_showers: Optional[bool] = None
    dump_station: Optional[bool] = None
    laundry: Optional[bool] = None
    camp_store_sells_firewood_ice_supplies: Optional[bool] = None
    wifi: Optional[bool] = None
    year_round: Optional[bool] = None
    pet_friendly: Optional[bool] = None
    online_reservation_tn_system: Optional[bool] = None
    ada_accessible_sites: Optional[bool] = None
    lake_access: Optional[bool] = None


class CampgroundItem(BaseModel):
    state_park_name: Optional[str] = None
    lake_name: Optional[str] = None
    section_count: Optional[str] = None
    total_campsites: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    features: FeatureFlags = Field(default_factory=FeatureFlags)


class CampgroundsExtraction(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract the Tennessee State Park campground(s) described in the answer. Return up to the first 3 items if more are provided.
    For each campground, extract the following fields:

    - state_park_name: The name of the Tennessee State Park.
    - lake_name: The specific lake that the campground is on or directly accesses (if applicable; otherwise null).
    - section_count: The number of designated campground sections/loops/areas. Keep as a short text if a number is not explicitly stated.
    - total_campsites: The total number of campsites available at the campground. Keep as a short text if a number is not explicitly stated.
    - reference_urls: A list of all URLs explicitly mentioned in the answer that support amenities, policies, site counts, sections, or reservations for this campground. Include only valid full URLs.
    - features: An object of boolean flags (true/false/null) based ONLY on the answer text:
        * full_hookup: offers full hookup campsites (sewer + water + electric)
        * amp_20: provides 20-amp electrical option
        * amp_30: provides 30-amp electrical option
        * amp_50: provides 50-amp electrical option
        * bathhouse_hot_showers: has bathhouse with hot showers
        * dump_station: provides an RV dump station
        * laundry: offers laundry facilities (washer and dryer)
        * camp_store_sells_firewood_ice_supplies: camp store that sells firewood, ice, and camping supplies
        * wifi: provides Wi-Fi internet access at the campground
        * year_round: operates year-round for camping
        * pet_friendly: allows pets (dogs) on leash
        * online_reservation_tn_system: accepts reservations through the Tennessee State Parks online reservation system
        * ada_accessible_sites: offers ADA-accessible campsites
        * lake_access: located on or provides direct access to a lake

    IMPORTANT:
    - Do NOT infer or invent information. Set a field to null if not clearly stated in the answer.
    - For reference_urls, include only URLs explicitly present in the answer (plain or markdown links).
    - Keep numbers as strings if the answer presents them in ranges or approximate forms (e.g., "100+", "about 120", or "over 150").
    - The 'features' booleans must reflect the answer content; if uncertain or unstated, set to null.

    Return JSON with a single key 'campgrounds' as an array of CampgroundItem objects.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_int_maybe(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    # Extract first integer-like token
    m = re.search(r"\d+", text.replace(",", ""))
    try:
        return int(m.group()) if m else None
    except Exception:
        return None


def safe_park_name(cg: CampgroundItem, idx: int) -> str:
    return cg.state_ark_name if hasattr(cg, "state_ark_name") else (cg.state_park_name or f"Tennessee State Park campground #{idx + 1}")


def safe_lake_name(cg: CampgroundItem) -> str:
    return cg.lake_name or "a lake"


# --------------------------------------------------------------------------- #
# Verification for a single campground                                        #
# --------------------------------------------------------------------------- #
async def verify_campground(evaluator: Evaluator, parent_node, cg: CampgroundItem, idx: int) -> None:
    # Container for one campground
    cg_label = cg.state_park_name or f"Unnamed Tennessee State Park (item #{idx + 1})"
    cg_node = evaluator.add_parallel(
        id=f"campground_{idx}",
        desc=f"Campground #{idx + 1}: {cg_label} - overall verification",
        parent=parent_node,
        critical=False
    )

    # Non-critical presence checks (information provided)
    provided_info_node = evaluator.add_parallel(
        id=f"cg{idx}_provided_info",
        desc="Provided key information presence checks",
        parent=cg_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(cg.state_park_name and cg.state_park_name.strip()),
        id=f"cg{idx}_Park_Name_Provided",
        desc="The solution provides the name of the state park",
        parent=provided_info_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(cg.lake_name and cg.lake_name.strip()),
        id=f"cg{idx}_Lake_Name_Provided",
        desc="The solution provides the specific lake the campground is located on (if applicable)",
        parent=provided_info_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(cg.section_count and str(cg.section_count).strip()),
        id=f"cg{idx}_Section_Count_Provided",
        desc="The solution provides the number of campground sections",
        parent=provided_info_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(cg.total_campsites and str(cg.total_campsites).strip()),
        id=f"cg{idx}_Total_Campsite_Count_Provided",
        desc="The solution provides the total number of campsites",
        parent=provided_info_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(cg.reference_urls),
        id=f"cg{idx}_Reference_URLs_Provided",
        desc="The solution provides reference URL(s) that verify the amenities and features",
        parent=provided_info_node,
        critical=False
    )

    # Critical requirements group
    req_node = evaluator.add_parallel(
        id=f"cg{idx}_Tennessee_State_Park_Campground_Requirements",
        desc="Evaluate whether the identified Tennessee state park campground(s) meet all specified requirements for amenities, policies, and features, and whether all required information is provided",
        parent=cg_node,
        critical=True
    )

    urls = cg.reference_urls or []

    # Prepare leaves and corresponding verification claims
    leaves_and_claims: List[tuple] = []

    # 1. Full hookups
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_Full_Hookup_Sites_Available",
        desc="The campground offers full hookup sites with sewer, water, and electrical connections",
        parent=req_node,
        critical=True
    )
    claim = f"{cg_label} campground offers full hookup campsites with sewer, water, and electrical connections."
    add_ins = "Confirm that the campground has full hookup (sewer + water + electricity) sites. Phrases like 'full hookups' or 'sewer, water, and electric' should count."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # 2. Electrical amp options
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_Electrical_Amp_Options",
        desc="The campground provides multiple electrical amperage options (20-amp, 30-amp, and 50-amp)",
        parent=req_node,
        critical=True
    )
    claim = f"{cg_label} campground provides 20-amp, 30-amp, and 50-amp electrical service options at campsites."
    add_ins = "The page should explicitly indicate availability of all three amperage options (20, 30, and 50 amp). Allow minor formatting differences (e.g., 20/30/50-amp)."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # 3. Bathhouse with hot showers
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_Bathhouse_Hot_Showers",
        desc="The campground has bathhouse facilities with hot showers",
        parent=req_node,
        critical=True
    )
    claim = f"{cg_label} campground has bathhouse facilities with hot showers."
    add_ins = "Look for mentions of 'bathhouse', 'bath houses', 'restrooms with hot showers', or similar wording."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # 4. Dump station
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_Dump_Station_Available",
        desc="The campground provides a dump station for RV waste disposal",
        parent=req_node,
        critical=True
    )
    claim = f"{cg_label} campground provides a dump station for RV waste disposal."
    add_ins = "Confirm an RV dump station is available either within the campground or adjacent to it."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # 5. Laundry
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_Laundry_Facilities",
        desc="The campground offers laundry facilities (washer and dryer)",
        parent=req_node,
        critical=True
    )
    claim = f"{cg_label} campground offers laundry facilities, including washer and dryer access."
    add_ins = "Verify terms like 'laundry', 'washers', 'dryers', or 'laundry facility' at or for the campground."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # 6. Camp store with firewood, ice, and supplies
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_Camp_Store_Access",
        desc="The campground has a camp store that sells firewood, ice, and camping supplies",
        parent=req_node,
        critical=True
    )
    claim = f"{cg_label} campground has a camp store that sells firewood, ice, and camping supplies."
    add_ins = "The evidence should indicate a camp store (or similar on-site shop) selling at least firewood and ice, plus common camping supplies or essentials."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # 7. Wi-Fi
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_WiFi_Available",
        desc="The campground provides Wi-Fi internet access",
        parent=req_node,
        critical=True
    )
    claim = f"Wi-Fi internet access is provided at or for {cg_label} campground."
    add_ins = "Ensure Wi-Fi is available to campers at the campground or clearly accessible within the campground area (camp store/office counts if usable by campers)."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # 8. Year-round operation
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_Year_Round_Operation",
        desc="The campground is open year-round for camping",
        parent=req_node,
        critical=True
    )
    claim = f"{cg_label} campground is open year-round for camping."
    add_ins = "Look for phrases like 'open year-round' or equivalent schedule indicating availability in all seasons."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # 9. Pet-friendly on leash
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_Pet_Friendly_Policy",
        desc="The campground allows pets (dogs) with appropriate leash requirements",
        parent=req_node,
        critical=True
    )
    claim = f"Pets (dogs) are allowed at {cg_label} campground with leash requirements."
    add_ins = "Confirm pets or dogs are allowed in the campground; standard leash rules are acceptable."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # 10. Online reservation via TN State Parks system
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_Online_Reservation_System",
        desc="The campground accepts reservations through the Tennessee State Parks online reservation system",
        parent=req_node,
        critical=True
    )
    claim = f"Reservations for {cg_label} campground are accepted through the Tennessee State Parks online reservation system."
    add_ins = "Support should indicate booking via the official Tennessee State Parks reservation platform (e.g., reserve.tnstateparks.com or equivalent official TN system)."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # 11. ADA-accessible campsites
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_ADA_Accessible_Sites",
        desc="The campground offers ADA-accessible campsites for wheelchair users",
        parent=req_node,
        critical=True
    )
    claim = f"{cg_label} campground offers ADA-accessible campsites."
    add_ins = "Prefer explicit mentions of 'ADA accessible campsites' or 'accessible sites'; general park accessibility is insufficient unless it clearly applies to campsites."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # 12. Located on or direct access to a lake
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_Waterfront_Lake_Access",
        desc="The campground is located on or provides direct access to a lake",
        parent=req_node,
        critical=True
    )
    lake_phrase = safe_lake_name(cg)
    claim = f"{cg_label} campground is located on or provides direct access to {lake_phrase}."
    add_ins = "A reservoir counts as a lake. The evidence should clearly associate the campground with direct access or on-the-lake location."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # 13. At least 3 designated sections
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_Multiple_Campground_Sections",
        desc="The campground has multiple designated camping areas or sections (at least 3 different sections)",
        parent=req_node,
        critical=True
    )
    claim = f"{cg_label} campground has at least 3 designated campground sections, areas, or loops."
    add_ins = "Look for mentions of multiple loops/sections (e.g., Loops A, B, C) or a clear section count of 3 or more."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # 14. At least 100 total campsites
    leaf = evaluator.add_leaf(
        id=f"cg{idx}_Minimum_Site_Count",
        desc="The campground has at least 100 total campsites available",
        parent=req_node,
        critical=True
    )
    claim = f"{cg_label} campground has at least 100 total campsites."
    add_ins = "Verify total campsite count is 100 or more; if the page lists a larger total or multiple loops summing to >=100, that satisfies the requirement."
    leaves_and_claims.append((claim, urls, leaf, add_ins))

    # Execute all requirement verifications in parallel
    await evaluator.batch_verify(leaves_and_claims)


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
    Evaluate an answer for Tennessee State Park campground requirements.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent evaluation per campground
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

    # Extract structured campground info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds_extraction",
    )

    # Record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "campgrounds_extracted": len(extraction.campgrounds),
            "first_park_names": [cg.state_park_name for cg in extraction.campgrounds[:3]],
        },
        info_type="meta",
        info_name="extraction_overview"
    )

    # Evaluate up to the first 3 campgrounds
    targets = extraction.campgrounds[:3] if extraction.campgrounds else []
    for idx, cg in enumerate(targets):
        await verify_campground(evaluator, root, cg, idx)

    return evaluator.get_summary()