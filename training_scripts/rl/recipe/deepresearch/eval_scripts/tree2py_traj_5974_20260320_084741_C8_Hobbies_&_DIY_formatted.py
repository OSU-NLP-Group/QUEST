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
TASK_ID = "craft_resource_us_2026"
TASK_DESCRIPTION = """
A beginner DIY enthusiast is creating a comprehensive resource guide for craft supplies, workshops, and maker spaces in the United States as of 2026. They need to compile detailed information about major national craft store chains, their operational policies, free workshop programs, and typical costs for maker spaces and pottery classes. Please provide the following information:

Part 1 - Major Craft Store Presence:
Which of these major craft store chains have locations in Colorado: Michaels, Hobby Lobby, and Blick Art Materials?

Part 2 - Craft Store Operational Details:
For the major craft store chains identified, provide:
- What is the rewards percentage for Michaels Platinum tier members, and what annual spending is required to reach this tier?
- What is Hobby Lobby's policy regarding Sunday operations?
- Is Hobby Lobby open or closed on Thanksgiving Day?
- Is Michaels open or closed on Christmas Day?
- How many U.S. states does Michaels operate in as of 2026?
- How many U.S. states does Hobby Lobby operate in?
- Approximately how many Blick Art Materials stores operate in the United States?
- Approximately how many Michaels stores are located in California, and what percentage of all Michaels stores does this represent?
- Which U.S. states do not have Hobby Lobby locations?
- Approximately how many total Michaels stores exist in the United States as of 2026?

Part 3 - Free DIY Workshop Programs:
Provide information about free workshop programs:
- Does Home Depot offer free kids workshops? If so, what age range do they serve, and which day of the month do they typically occur?
- Does Lowe's offer free kids workshops?
- Does Michaels offer free online classes?

Part 4 - Maker Space Access:
What are the typical requirements and costs for maker spaces:
- What is the typical minimum age range for unsupervised equipment use?
- Is safety training mandatory before using woodworking equipment?
- What is the typical cost range for day passes?
- What is the typical cost range for monthly memberships?

Part 5 - Pottery Classes:
For beginner pottery wheel throwing classes:
- What is the typical session length in hours?
- Is equipment typically provided for beginners?
- What is the typical cost range per pound for glaze firing fees?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ChainPresence(BaseModel):
    has_locations_in_state: Optional[str] = None  # "yes" or "no" (string, not boolean)
    sources: List[str] = Field(default_factory=list)


class PresenceSection(BaseModel):
    state: Optional[str] = None  # Expect "Colorado"
    michaels: Optional[ChainPresence] = None
    hobby_lobby: Optional[ChainPresence] = None
    blick: Optional[ChainPresence] = None


class CraftStoreCharacteristics(BaseModel):
    michaels_platinum_rewards_percent: Optional[str] = None
    michaels_platinum_spending_requirement: Optional[str] = None
    hobby_lobby_sunday_policy: Optional[str] = None
    hobby_lobby_thanksgiving_policy: Optional[str] = None
    michaels_christmas_policy: Optional[str] = None
    michaels_state_count_2026: Optional[str] = None
    hobby_lobby_state_count: Optional[str] = None
    blick_us_store_count: Optional[str] = None
    michaels_california_store_count: Optional[str] = None
    michaels_california_percentage: Optional[str] = None
    hobby_lobby_excluded_states: List[str] = Field(default_factory=list)
    michaels_total_store_count_2026: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FreeWorkshops(BaseModel):
    home_depot_kids_workshop_available: Optional[str] = None  # "yes"/"no" (string)
    home_depot_kids_age_range: Optional[str] = None
    home_depot_kids_day_of_month: Optional[str] = None
    lowes_kids_workshop_available: Optional[str] = None  # "yes"/"no"
    michaels_free_online_classes: Optional[str] = None  # "yes"/"no"
    sources: List[str] = Field(default_factory=list)


class MakerSpaceInfo(BaseModel):
    unsupervised_min_age: Optional[str] = None
    safety_training_mandatory: Optional[str] = None  # "yes"/"no"
    day_pass_cost_range: Optional[str] = None
    monthly_membership_cost_range: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PotteryInfo(BaseModel):
    session_length_hours: Optional[str] = None
    equipment_provided_for_beginners: Optional[str] = None  # "yes"/"no"
    glaze_firing_cost_per_lb_range: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CraftResourceExtraction(BaseModel):
    presence: Optional[PresenceSection] = None
    characteristics: Optional[CraftStoreCharacteristics] = None
    workshops: Optional[FreeWorkshops] = None
    maker_space: Optional[MakerSpaceInfo] = None
    pottery: Optional[PotteryInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_craft_resource() -> str:
    return """
Extract structured information from the answer for five parts. Use the exact wording and numbers the answer provides. For any boolean-like item, extract a short "yes" or "no" (lowercase string). For ranges or approximate values, keep the original text (e.g., "about 450", "2–3 hours", "$4–$10/lb"). For URLs, extract all valid URLs explicitly present in the answer.

Return a JSON object with the following structure:

{
  "presence": {
    "state": "Colorado",
    "michaels": {
      "has_locations_in_state": "yes|no|null",
      "sources": ["..."]
    },
    "hobby_lobby": {
      "has_locations_in_state": "yes|no|null",
      "sources": ["..."]
    },
    "blick": {
      "has_locations_in_state": "yes|no|null",
      "sources": ["..."]
    }
  },
  "characteristics": {
    "michaels_platinum_rewards_percent": "e.g., '6% back' or '6%'",
    "michaels_platinum_spending_requirement": "e.g., '$500 per year'",
    "hobby_lobby_sunday_policy": "short policy text, e.g., 'Closed on Sundays'",
    "hobby_lobby_thanksgiving_policy": "short text, e.g., 'Closed on Thanksgiving Day'",
    "michaels_christmas_policy": "short text, e.g., 'Closed on Christmas Day'",
    "michaels_state_count_2026": "number or text, e.g., '49'",
    "hobby_lobby_state_count": "number or text, e.g., '48'",
    "blick_us_store_count": "approx count, e.g., '~60'",
    "michaels_california_store_count": "approx count, e.g., '200+'",
    "michaels_california_percentage": "percentage text, e.g., '12%'",
    "hobby_lobby_excluded_states": ["State1", "State2"],
    "michaels_total_store_count_2026": "approx total, e.g., '~1250'",
    "sources": ["URLs that support any of the above facts"]
  },
  "workshops": {
    "home_depot_kids_workshop_available": "yes|no|null",
    "home_depot_kids_age_range": "e.g., 'ages 5–12'",
    "home_depot_kids_day_of_month": "e.g., 'first Saturday of each month'",
    "lowes_kids_workshop_available": "yes|no|null",
    "michaels_free_online_classes": "yes|no|null",
    "sources": ["URLs that support the workshop info"]
  },
  "maker_space": {
    "unsupervised_min_age": "e.g., '16+' or '18+'",
    "safety_training_mandatory": "yes|no|null",
    "day_pass_cost_range": "e.g., '$20–$40'",
    "monthly_membership_cost_range": "e.g., '$60–$200'",
    "sources": ["URLs that support typical maker space policies and pricing"]
  },
  "pottery": {
    "session_length_hours": "e.g., '2–3 hours'",
    "equipment_provided_for_beginners": "yes|no|null",
    "glaze_firing_cost_per_lb_range": "e.g., '$4–$10/lb'",
    "sources": ["URLs that support pottery class norms and fees"]
  }
}

Rules:
- If any field is not explicitly stated in the answer, set it to null (or [] for lists).
- Extract only URLs that are actually present in the answer text (including in markdown links).
- Do not invent or infer values.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def interpret_yes_no(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    txt = s.strip().lower()
    if txt in {"yes", "y", "true", "has", "have", "open", "present"}:
        return True
    if txt in {"no", "n", "false", "not", "none", "closed", "absent"}:
        return False
    return None


def list_to_comma_separated(items: List[str]) -> str:
    return ", ".join(items)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_presence_for_brand(
    evaluator: Evaluator,
    parent_node,
    brand_label: str,
    presence_item: Optional[ChainPresence],
    state_label: str,
    id_prefix: str
) -> None:
    """
    Build a small sub-tree for one brand's presence in a given state.
    Leaf IDs follow rubric names:
      - {Brand}_Presence (verification leaf)
      - {Brand}_Presence_URL (custom existence leaf for sources)
    """
    group = evaluator.add_parallel(
        id=f"{id_prefix}_Group",
        desc=f"{brand_label} presence verification in {state_label}",
        parent=parent_node,
        critical=False
    )

    sources = presence_item.sources if presence_item else []
    # URL existence check (critical for this brand only)
    evaluator.add_custom_node(
        result=bool(sources),
        id=f"{id_prefix}_URL",
        desc=f"Provide URL reference confirming {brand_label} presence in the state",
        parent=group,
        critical=True
    )

    # Presence verification leaf (critical)
    presence_leaf = evaluator.add_leaf(
        id=f"{id_prefix}",
        desc=f"Confirm {brand_label} has store locations in the state" if interpret_yes_no(
            presence_item.has_locations_in_state if presence_item else None) is not False else
        f"Confirm {brand_label} has no store locations in the state",
        parent=group,
        critical=True
    )

    yn = interpret_yes_no(presence_item.has_locations_in_state if presence_item else None)
    if yn is True:
        claim = f"{brand_label} has at least one store location in {state_label}."
    elif yn is False:
        claim = f"{brand_label} does not have any store locations in {state_label}."
    else:
        # Fallback to a neutral statement reflecting the extracted text
        val = (presence_item.has_locations_in_state if presence_item else None) or "unspecified"
        claim = f"The statement about {brand_label} presence in {state_label} is: '{val}', and it is supported by the cited sources."

    await evaluator.verify(
        claim=claim,
        node=presence_leaf,
        sources=sources,
        additional_instruction=(
            f"Check the provided store-locator or official pages to confirm whether {brand_label} has stores in {state_label} "
            f"(Colorado may also be abbreviated as 'CO'). Pages listing Colorado store addresses or a state filter indicating "
            f"{'availability' if yn is True else 'no results'} are acceptable evidence."
        ),
    )


async def verify_major_chain_presence(
    evaluator: Evaluator,
    root_parent,
    presence: Optional[PresenceSection],
    default_state: str = "Colorado"
) -> None:
    node = evaluator.add_parallel(
        id="Major_Craft_Chain_Presence",
        desc="Identify which major national craft store chains have locations in the specified state",
        parent=root_parent,
        critical=False
    )

    state = (presence.state if presence and presence.state else default_state) or default_state

    await verify_presence_for_brand(
        evaluator, node, "Michaels",
        presence.michaels if presence else None, state, "Michaels_Presence"
    )
    await verify_presence_for_brand(
        evaluator, node, "Hobby Lobby",
        presence.hobby_lobby if presence else None, state, "Hobby_Lobby_Presence"
    )
    await verify_presence_for_brand(
        evaluator, node, "Blick Art Materials",
        presence.blick if presence else None, state, "Blick_Presence"
    )


async def verify_craft_store_characteristics(
    evaluator: Evaluator,
    root_parent,
    chars: Optional[CraftStoreCharacteristics],
) -> None:
    node = evaluator.add_parallel(
        id="Craft_Store_Characteristics",
        desc="Document specific operational characteristics of identified craft stores",
        parent=root_parent,
        critical=False
    )

    sources = chars.sources if chars else []
    # Global source presence for this section (critical)
    evaluator.add_custom_node(
        result=bool(sources),
        id="Craft_Store_Characteristics_URL",
        desc="Provide URL reference for craft store operational characteristics",
        parent=node,
        critical=True
    )

    # Helper to add a leaf and verify against section sources
    async def _leaf(id_: str, desc_: str, claim_text: str, add_ins: str, critical: bool = True):
        leaf = evaluator.add_leaf(
            id=id_,
            desc=desc_,
            parent=node,
            critical=critical
        )
        await evaluator.verify(
            claim=claim_text,
            node=leaf,
            sources=sources,
            additional_instruction=add_ins
        )

    # Michaels Platinum rewards percent
    await _leaf(
        "Michaels_Rewards_Program",
        "Specify the rewards percentage for Michaels Platinum tier members",
        f"Michaels Platinum tier rewards percentage is '{(chars.michaels_platinum_rewards_percent or 'unspecified')}'.",
        "Verify the exact percent or 'X% back' phrasing on an official Michaels Rewards or help page. Minor formatting variations are acceptable."
    )

    # Michaels Platinum spending requirement
    await _leaf(
        "Michaels_Platinum_Spending",
        "Specify the annual spending requirement to reach Michaels Platinum tier",
        f"The annual spending requirement to reach Michaels Platinum tier is '{(chars.michaels_platinum_spending_requirement or 'unspecified')}'.",
        "Confirm on Michaels official Rewards tier details. Accept equivalent wording (e.g., 'in a calendar year')."
    )

    # Hobby Lobby Sunday policy
    await _leaf(
        "Hobby_Lobby_Sunday_Policy",
        "Confirm Hobby Lobby's Sunday closure policy",
        f"Hobby Lobby's policy regarding Sunday operations is: '{(chars.hobby_lobby_sunday_policy or 'unspecified')}'.",
        "Look for official policy pages, hours pages, or credible news coverage indicating Sunday closure."
    )

    # Hobby Lobby Thanksgiving policy
    await _leaf(
        "Hobby_Lobby_Thanksgiving_Policy",
        "Confirm whether Hobby Lobby is open or closed on Thanksgiving Day",
        f"Hobby Lobby's Thanksgiving Day policy is: '{(chars.hobby_lobby_thanksgiving_policy or 'unspecified')}'.",
        "Holiday hours pages or company announcements are acceptable. Prioritize official sources."
    )

    # Michaels Christmas policy
    await _leaf(
        "Michaels_Christmas_Policy",
        "Confirm whether Michaels is open or closed on Christmas Day",
        f"Michaels' Christmas Day policy is: '{(chars.michaels_christmas_policy or 'unspecified')}'.",
        "Check holiday hours on Michaels sites or trustworthy retail-hours references."
    )

    # Michaels state count (as of 2026)
    await _leaf(
        "Michaels_State_Count",
        "Specify how many states Michaels operates in as of 2026",
        f"As of 2026, Michaels operates in '{(chars.michaels_state_count_2026 or 'unspecified')}' U.S. states.",
        "Confirm from Michaels official site or credible sources summarizing state coverage. Allow minor phrasing variations."
    )

    # Hobby Lobby state count
    await _leaf(
        "Hobby_Lobby_State_Count",
        "Specify how many states Hobby Lobby operates in",
        f"Hobby Lobby operates in '{(chars.hobby_lobby_state_count or 'unspecified')}' U.S. states.",
        "Confirm via Hobby Lobby official store locator coverage or credible summaries."
    )

    # Blick US store count (approx)
    await _leaf(
        "Blick_Store_Count",
        "Specify approximately how many Blick Art Materials stores operate in the United States",
        f"Approximately '{(chars.blick_us_store_count or 'unspecified')}' Blick Art Materials stores operate in the U.S.",
        "Approximate counts are acceptable if stated by Blick or credible industry sources. Allow rounding."
    )

    # Michaels CA store count (approx)
    await _leaf(
        "California_Michaels_Count",
        "Specify approximately how many Michaels stores are located in California",
        f"Approximately '{(chars.michaels_california_store_count or 'unspecified')}' Michaels stores are in California.",
        "Evidence can include a store list or an explicit count from Michaels or credible sources."
    )

    # Michaels CA percentage of all stores (treat as non-critical to allow partial credit)
    await _leaf(
        "California_Michaels_Percentage",
        "Specify what percentage of all Michaels stores are in California",
        f"About '{(chars.michaels_california_percentage or 'unspecified')}' of all Michaels stores are in California.",
        "This can be directly stated on a source page. If a source shows both counts (CA and total), computed percentages are acceptable.",
        critical=False
    )

    # States without Hobby Lobby
    excluded_states_str = list_to_comma_separated(chars.hobby_lobby_excluded_states) if chars and chars.hobby_lobby_excluded_states else "unspecified"
    await _leaf(
        "Hobby_Lobby_Excluded_States",
        "Identify which U.S. states do not have Hobby Lobby locations",
        f"The U.S. states without Hobby Lobby locations are: {excluded_states_str}.",
        "Check credible maps or official sources that list state coverage and identify states with zero locations. Minor naming variants (e.g., 'D.C.') acceptable."
    )

    # Total Michaels stores (as of 2026)
    await _leaf(
        "Michaels_Total_Store_Count",
        "Specify approximately how many total Michaels stores exist in the United States as of 2026",
        f"As of 2026, there are approximately '{(chars.michaels_total_store_count_2026 or 'unspecified')}' total Michaels stores in the U.S.",
        "Use Michaels official statements, investor materials, or credible industry sources stating approximate totals."
    )


async def verify_free_workshop_programs(
    evaluator: Evaluator,
    root_parent,
    wk: Optional[FreeWorkshops],
) -> None:
    node = evaluator.add_parallel(
        id="Free_Workshop_Programs",
        desc="Document free DIY workshop programs offered by major retailers",
        parent=root_parent,
        critical=False
    )

    sources = wk.sources if wk else []
    evaluator.add_custom_node(
        result=bool(sources),
        id="Free_Workshop_URL",
        desc="Provide URL reference for free workshop program information",
        parent=node,
        critical=True
    )

    async def _leaf(id_: str, desc_: str, claim_text: str, add_ins: str, critical: bool = True):
        leaf = evaluator.add_leaf(
            id=id_,
            desc=desc_,
            parent=node,
            critical=critical
        )
        await evaluator.verify(
            claim=claim_text,
            node=leaf,
            sources=sources,
            additional_instruction=add_ins
        )

    # Home Depot availability
    yn_hd = interpret_yes_no(wk.home_depot_kids_workshop_available if wk else None)
    await _leaf(
        "Home_Depot_Workshop_Availability",
        "Confirm Home Depot offers free kids workshops",
        ("Home Depot offers free kids workshops." if yn_hd is True else
         "Home Depot does not offer free kids workshops." if yn_hd is False else
         "The availability of Home Depot free kids workshops is as stated in the answer."),
        "Use Home Depot official Kids Workshops page or news/PR pages describing free monthly workshops."
    )

    # Home Depot ages
    await _leaf(
        "Home_Depot_Workshop_Ages",
        "Specify the age range for Home Depot kids workshops",
        f"The age range for Home Depot kids workshops is '{(wk.home_depot_kids_age_range if wk else None) or 'unspecified'}'.",
        "Confirm typical age range from Home Depot official pages or registration details. Minor formatting variations acceptable."
    )

    # Home Depot schedule (day of month)
    await _leaf(
        "Home_Depot_Workshop_Schedule",
        "Specify which day of the month Home Depot kids workshops typically occur",
        f"Home Depot kids workshops typically occur on '{(wk.home_depot_kids_day_of_month if wk else None) or 'unspecified'}'.",
        "Many sources state 'first Saturday of each month'; verify exact phrasing from official pages or consistent announcements."
    )

    # Lowe's availability
    yn_lowes = interpret_yes_no(wk.lowes_kids_workshop_available if wk else None)
    await _leaf(
        "Lowes_Workshop_Availability",
        "Confirm Lowe's offers free kids workshops",
        ("Lowe's offers free kids workshops." if yn_lowes is True else
         "Lowe's does not offer free kids workshops." if yn_lowes is False else
         "The availability of Lowe's free kids workshops is as stated in the answer."),
        "Use Lowe's official Kids Workshops pages or registration portals to verify."
    )

    # Michaels free online classes
    yn_michaels_classes = interpret_yes_no(wk.michaels_free_online_classes if wk else None)
    await _leaf(
        "Michaels_Online_Classes",
        "Confirm Michaels offers free online classes",
        ("Michaels offers free online classes." if yn_michaels_classes is True else
         "Michaels does not offer free online classes." if yn_michaels_classes is False else
         "The status of Michaels free online classes is as stated in the answer."),
        "Confirm via Michaels Classes/Community Classroom pages or official announcements."
    )


async def verify_maker_space_requirements(
    evaluator: Evaluator,
    root_parent,
    mk: Optional[MakerSpaceInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Maker_Space_Requirements",
        desc="Document typical requirements and costs for maker space access",
        parent=root_parent,
        critical=False
    )

    sources = mk.sources if mk else []
    evaluator.add_custom_node(
        result=bool(sources),
        id="Maker_Space_URL",
        desc="Provide URL reference for maker space requirements and costs",
        parent=node,
        critical=True
    )

    async def _leaf(id_: str, desc_: str, claim_text: str, add_ins: str, critical: bool = True):
        leaf = evaluator.add_leaf(
            id=id_,
            desc=desc_,
            parent=node,
            critical=critical
        )
        await evaluator.verify(
            claim=claim_text,
            node=leaf,
            sources=sources,
            additional_instruction=add_ins
        )

    await _leaf(
        "Maker_Space_Age_Minimum",
        "Specify the typical minimum age range for unsupervised maker space equipment use",
        f"The typical minimum age for unsupervised maker space equipment use is '{(mk.unsupervised_min_age if mk else None) or 'unspecified'}'.",
        "Verify from multiple maker space policy pages; typical values are often 16+ or 18+. Accept consistent ranges across sources."
    )

    yn_training = interpret_yes_no(mk.safety_training_mandatory if mk else None)
    await _leaf(
        "Safety_Training_Requirement",
        "Confirm whether safety training is mandatory before using maker space woodworking equipment",
        ("Safety training is mandatory before using woodworking equipment at maker spaces." if yn_training is True else
         "Safety training is not mandatory before using woodworking equipment at maker spaces." if yn_training is False else
         "Safety training requirements for maker spaces are as stated in the answer."),
        "Most reputable spaces require safety or certification courses before tool access; verify via policy pages."
    )

    await _leaf(
        "Day_Pass_Cost_Range",
        "Specify the typical cost range for maker space day passes",
        f"The typical cost range for maker space day passes is '{(mk.day_pass_cost_range if mk else None) or 'unspecified'}'.",
        "Confirm from several maker space pricing pages; allow reasonable ranges and rounding."
    )

    await _leaf(
        "Monthly_Membership_Cost_Range",
        "Specify the typical cost range for maker space monthly memberships",
        f"The typical cost range for maker space monthly memberships is '{(mk.monthly_membership_cost_range if mk else None) or 'unspecified'}'.",
        "Confirm from several maker space pricing pages; allow typical ranges."
    )


async def verify_pottery_class_information(
    evaluator: Evaluator,
    root_parent,
    pot: Optional[PotteryInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Pottery_Class_Information",
        desc="Document typical characteristics of beginner pottery wheel throwing classes",
        parent=root_parent,
        critical=False
    )

    sources = pot.sources if pot else []
    evaluator.add_custom_node(
        result=bool(sources),
        id="Pottery_Class_URL",
        desc="Provide URL reference for pottery class information",
        parent=node,
        critical=True
    )

    async def _leaf(id_: str, desc_: str, claim_text: str, add_ins: str, critical: bool = True):
        leaf = evaluator.add_leaf(
            id=id_,
            desc=desc_,
            parent=node,
            critical=critical
        )
        await evaluator.verify(
            claim=claim_text,
            node=leaf,
            sources=sources,
            additional_instruction=add_ins
        )

    await _leaf(
        "Class_Session_Length",
        "Specify the typical duration in hours for beginner pottery wheel throwing class sessions",
        f"The typical session length for beginner pottery wheel throwing classes is '{(pot.session_length_hours if pot else None) or 'unspecified'}'.",
        "Verify using several pottery studio class descriptions; common ranges like 2–3 hours are acceptable."
    )

    yn_equipment = interpret_yes_no(pot.equipment_provided_for_beginners if pot else None)
    await _leaf(
        "Equipment_Provided",
        "Confirm whether pottery classes typically provide equipment for beginners",
        ("Beginner pottery classes typically provide equipment for students." if yn_equipment is True else
         "Beginner pottery classes typically do not provide equipment for students." if yn_equipment is False else
         "Whether equipment is provided for beginners is as stated in the answer."),
        "Check multiple studio pages; 'equipment provided' may include wheel, tools, and clay for class."
    )

    await _leaf(
        "Glaze_Firing_Cost_Range",
        "Specify the typical cost range per pound for pottery studio glaze firing fees",
        f"The typical glaze firing fee range per pound is '{(pot.glaze_firing_cost_per_lb_range if pot else None) or 'unspecified'}'.",
        "Verify from pottery studio pricing pages; accept common ranges like $3–$10/lb; allow rounding."
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
    Evaluate an answer for the 2026 U.S. craft resource guide task.
    """
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

    # Extract all relevant structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_craft_resource(),
        template_class=CraftResourceExtraction,
        extraction_name="craft_resource_extraction"
    )

    # Build verification subtrees per rubric sections
    await verify_major_chain_presence(
        evaluator=evaluator,
        root_parent=root,
        presence=extracted.presence,
        default_state="Colorado"
    )

    await verify_craft_store_characteristics(
        evaluator=evaluator,
        root_parent=root,
        chars=extracted.characteristics
    )

    await verify_free_workshop_programs(
        evaluator=evaluator,
        root_parent=root,
        wk=extracted.workshops
    )

    await verify_maker_space_requirements(
        evaluator=evaluator,
        root_parent=root,
        mk=extracted.maker_space
    )

    await verify_pottery_class_information(
        evaluator=evaluator,
        root_parent=root,
        pot=extracted.pottery
    )

    # Return evaluator summary
    return evaluator.get_summary()