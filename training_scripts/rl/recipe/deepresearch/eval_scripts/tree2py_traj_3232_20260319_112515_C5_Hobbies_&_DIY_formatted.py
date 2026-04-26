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
TASK_ID = "weekend_diy_2026_03_21_22"
TASK_DESCRIPTION = """
I'm planning a family DIY weekend for March 21-22, 2026, and need help with the following:

On Saturday, March 21, 2026, I want to take my 6-year-old child to a free kids woodworking workshop at either Home Depot or Lowe's. Which of these two retailers has a workshop scheduled for that specific date? What are the registration requirements (including when registration opens), and does my child meet the age requirements?

After the workshop, we plan to purchase materials to build a simple side table together as a beginner woodworking project. What are the essential materials I need, including: the primary lumber size commonly used for legs, the secondary lumber size used for trim/edges, the recommended plywood thickness for structural components, and the three basic power tools required for beginner woodworking?

Finally, on Sunday, March 22, I need to visit a major craft store (either Hobby Lobby or Michaels) to buy decorating supplies. Which of these two stores is open on Sundays based on their regular store hours policy?

Please provide your answer with supporting reference URLs for verification.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class KidsWorkshopInfo(BaseModel):
    provider: Optional[str] = None  # "Home Depot" or "Lowe's"
    date_text: Optional[str] = None  # any explicit date text mentioned for the workshop
    urls: List[str] = Field(default_factory=list)  # event/workshop/policy pages
    registration_requirements: Optional[str] = None  # free text (e.g., “registration required online”)
    registration_opens_when: Optional[str] = None  # e.g., “opens 4 weeks prior”
    age_requirement_text: Optional[str] = None  # e.g., “ages 5–12”
    is_free_text: Optional[str] = None  # e.g., “free”, “no cost”


class SideTableInfo(BaseModel):
    leg_lumber_size: Optional[str] = None  # e.g., "2x2"
    trim_edge_lumber_size: Optional[str] = None  # e.g., "1x2"
    plywood_thickness: Optional[str] = None  # e.g., "3/4 inch"
    three_basic_power_tools: List[str] = Field(default_factory=list)  # exactly three (extractor should pick first 3)
    urls: List[str] = Field(default_factory=list)  # references supporting the recommendations


class CraftStoreInfo(BaseModel):
    sunday_open_store: Optional[str] = None  # "Hobby Lobby" or "Michaels"
    urls: List[str] = Field(default_factory=list)  # hours/policy pages for Sunday operations


class WeekendPlanExtraction(BaseModel):
    kids_workshop: Optional[KidsWorkshopInfo] = None
    side_table: Optional[SideTableInfo] = None
    craft_store: Optional[CraftStoreInfo] = None
    all_reference_urls: List[str] = Field(default_factory=list)  # a flat list of all URLs found anywhere in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_weekend_plan() -> str:
    return """
    Extract the structured information explicitly stated in the answer. Only extract values that the answer actually claims and only URLs that are explicitly present in the answer text (including markdown links).
    
    1) kids_workshop:
       - provider: Which retailer is claimed to host the kids woodworking workshop on March 21, 2026? Use exactly "Home Depot" or "Lowe's" if possible.
       - date_text: Any explicit date phrasing the answer used for the workshop (e.g., "Saturday, March 21, 2026").
       - urls: All URLs cited in the answer that are relevant to the kids workshop (event listing pages, FAQ, registration, policy pages).
       - registration_requirements: The registration/participation requirements text as stated in the answer (e.g., "registration required online").
       - registration_opens_when: The timing for when registration opens as stated in the answer (e.g., "opens four weeks before").
       - age_requirement_text: The allowed age range or rule text as stated (e.g., "ages 5–12", "ages 6+").
       - is_free_text: The answer’s statement indicating free/no cost if provided (e.g., "free", "no cost"). If not explicitly stated, return null.

    2) side_table:
       - leg_lumber_size: The primary lumber size commonly used for side-table legs as stated (e.g., "2x2").
       - trim_edge_lumber_size: The secondary lumber size commonly used for trim/edges (e.g., "1x2").
       - plywood_thickness: The recommended plywood thickness for structural components (e.g., "3/4 in").
       - three_basic_power_tools: Exactly three tools listed in the answer as "basic power tools" for beginners. If the answer lists more than three, choose the first three mentioned. If fewer than three are mentioned, return only those mentioned.
       - urls: All URLs cited in the answer that support these materials/tool recommendations.

    3) craft_store:
       - sunday_open_store: Which of "Hobby Lobby" or "Michaels" the answer claims is open on Sundays based on regular store-hours policy.
       - urls: All URLs cited in the answer relevant to Sunday hours for these stores (store policy/hours pages, official sites).

    4) all_reference_urls:
       - A flat list of every URL present anywhere in the answer. If no URLs are present, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def first_n(items: List[str], n: int) -> List[str]:
    return [s for s in items[:n] if isinstance(s, str) and s.strip()]


def list_to_english(items: List[str]) -> str:
    items = [i.strip() for i in items if i and i.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def collect_all_urls(extracted: WeekendPlanExtraction) -> List[str]:
    urls: List[str] = []
    if extracted.kids_workshop and extracted.kids_workshop.urls:
        urls.extend(extracted.kids_workshop.urls)
    if extracted.side_table and extracted.side_table.urls:
        urls.extend(extracted.side_table.urls)
    if extracted.craft_store and extracted.craft_store.urls:
        urls.extend(extracted.craft_store.urls)
    if extracted.all_reference_urls:
        urls.extend(extracted.all_reference_urls)
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if isinstance(u, str) and u.strip() and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_kids_workshop(evaluator: Evaluator, parent_node, extracted: WeekendPlanExtraction) -> None:
    """
    kids_workshop_march_21 (sequential, critical)
      - workshop_provider_on_date (leaf, critical)
      - registration_requirements_including_open (leaf, critical)
      - age_eligibility_for_child (leaf, critical)
      - workshop_is_free (leaf, critical)
    """
    kw = extracted.kids_workshop or KidsWorkshopInfo()

    kids_node = evaluator.add_sequential(
        id="kids_workshop_march_21",
        desc="Determine which retailer has a kids woodworking workshop on Saturday, March 21, 2026 and provide participation/registration/eligibility details.",
        parent=parent_node,
        critical=True
    )

    # 1) Provider on date
    provider_leaf = evaluator.add_leaf(
        id="workshop_provider_on_date",
        desc="Identify whether Home Depot or Lowe’s has a kids workshop scheduled specifically on March 21, 2026.",
        parent=kids_node,
        critical=True
    )
    provider = (kw.provider or "").strip()
    provider_claim = (
        f"The retailer that has a kids woodworking workshop scheduled on Saturday, March 21, 2026 is {provider}."
        if provider else
        "The answer identifies which retailer has a kids woodworking workshop scheduled on Saturday, March 21, 2026."
    )
    await evaluator.verify(
        claim=provider_claim,
        node=provider_leaf,
        sources=kw.urls,
        additional_instruction="Pass only if the cited source(s) explicitly show a kids workshop occurring on Saturday, March 21, 2026 for the named retailer (Home Depot or Lowe's). "
                               "If the provider name is blank or unsupported by the sources for that specific date, mark as Incorrect. "
                               "Accept reasonable synonyms like 'Kids Workshop', 'Kids Clinic', or 'Build and Grow'."
    )

    # 2) Registration requirements including when it opens
    reg_leaf = evaluator.add_leaf(
        id="registration_requirements_including_open",
        desc="State the registration requirements for the identified workshop, including when registration opens.",
        parent=kids_node,
        critical=True
    )
    reg_req = (kw.registration_requirements or "").strip()
    reg_open = (kw.registration_opens_when or "").strip()
    reg_claim = (
        f"For the identified retailer's kids workshop on March 21, 2026, registration is required and registration opens: {reg_open}. "
        f"Additional stated requirement details: {reg_req}"
    )
    await evaluator.verify(
        claim=reg_claim,
        node=reg_leaf,
        sources=kw.urls,
        additional_instruction="Pass only if the sources support that registration (or RSVP) is required AND the stated 'registration opens' timing matches. "
                               "If the answer omits the 'opens when' timing or it does not match the sources, mark as Incorrect."
    )

    # 3) Age eligibility for a 6-year-old
    age_leaf = evaluator.add_leaf(
        id="age_eligibility_for_child",
        desc="Determine whether a 6-year-old meets the workshop age requirements, including stating the allowed age range.",
        parent=kids_node,
        critical=True
    )
    age_text = (kw.age_requirement_text or "").strip()
    age_claim = (
        f"A 6-year-old meets the workshop age requirement. The allowed age range per the sources is: '{age_text}'."
        if age_text else
        "A 6-year-old meets the workshop age requirement for the identified program."
    )
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=kw.urls,
        additional_instruction="Pass only if the sources explicitly indicate that age 6 is within the allowed range (e.g., 'ages 5–12', '6+', etc.). "
                               "If age range text is missing in the answer or not supported by sources, mark as Incorrect. "
                               "Assume inclusive ranges unless clearly stated otherwise."
    )

    # 4) Workshop is free
    free_leaf = evaluator.add_leaf(
        id="workshop_is_free",
        desc="Confirm the identified workshop is free.",
        parent=kids_node,
        critical=True
    )
    free_claim = "The identified kids workshop is free to attend (no cost)."
    await evaluator.verify(
        claim=free_claim,
        node=free_leaf,
        sources=kw.urls,
        additional_instruction="Pass only if the sources clearly indicate the event is free/no-cost. If pricing is unclear or not free, mark as Incorrect."
    )


async def verify_side_table(evaluator: Evaluator, parent_node, extracted: WeekendPlanExtraction) -> None:
    """
    side_table_materials_and_tools (parallel, critical)
      - leg_lumber_size (leaf, critical)
      - trim_edge_lumber_size (leaf, critical)
      - plywood_thickness (leaf, critical)
      - three_basic_power_tools (leaf, critical)
    """
    st = extracted.side_table or SideTableInfo()

    side_node = evaluator.add_parallel(
        id="side_table_materials_and_tools",
        desc="Provide the essential beginner side-table build materials and tools requested.",
        parent=parent_node,
        critical=True
    )

    # Leg lumber size
    leg_leaf = evaluator.add_leaf(
        id="leg_lumber_size",
        desc="Provide the primary lumber size commonly used for side-table legs.",
        parent=side_node,
        critical=True
    )
    leg_size = (st.leg_lumber_size or "").strip()
    leg_claim = (
        f"A commonly used primary lumber size for small side-table legs is '{leg_size}'."
        if leg_size else
        "The answer identifies a commonly used primary lumber size for small side-table legs."
    )
    await evaluator.verify(
        claim=leg_claim,
        node=leg_leaf,
        sources=st.urls,
        additional_instruction="Pass only if at least one cited source supports or exemplifies this size as common/typical for side-table legs in beginner projects. "
                               "If the answer provides no size or the sources do not support it, mark as Incorrect."
    )

    # Trim/edge lumber size
    trim_leaf = evaluator.add_leaf(
        id="trim_edge_lumber_size",
        desc="Provide the secondary lumber size commonly used for trim/edges.",
        parent=side_node,
        critical=True
    )
    trim_size = (st.trim_edge_lumber_size or "").strip()
    trim_claim = (
        f"A commonly used secondary lumber size for trim/edges in beginner side-table projects is '{trim_size}'."
        if trim_size else
        "The answer identifies a commonly used secondary lumber size for trim/edges in beginner side-table projects."
    )
    await evaluator.verify(
        claim=trim_claim,
        node=trim_leaf,
        sources=st.urls,
        additional_instruction="Pass only if at least one cited source supports or exemplifies this size as common/typical for trim/edges. "
                               "If missing or unsupported, mark as Incorrect."
    )

    # Plywood thickness
    ply_leaf = evaluator.add_leaf(
        id="plywood_thickness",
        desc="Provide the recommended plywood thickness for structural components.",
        parent=side_node,
        critical=True
    )
    ply = (st.plywood_thickness or "").strip()
    ply_claim = (
        f"A recommended plywood thickness for structural components of a small side table is '{ply}'."
        if ply else
        "The answer provides a recommended plywood thickness for structural components of a small side table."
    )
    await evaluator.verify(
        claim=ply_claim,
        node=ply_leaf,
        sources=st.urls,
        additional_instruction="Pass only if at least one cited source supports this thickness as recommended/common for structural components in a small/beginner side table. "
                               "If missing or unsupported, mark as Incorrect."
    )

    # Three basic power tools
    tools_leaf = evaluator.add_leaf(
        id="three_basic_power_tools",
        desc="List the three basic power tools required for beginner woodworking.",
        parent=side_node,
        critical=True
    )
    tools = first_n(st.three_basic_power_tools, 3)
    tools_claim = (
        f"The answer lists exactly three basic power tools for beginner woodworking: {list_to_english(tools)}."
        if tools else
        "The answer lists exactly three basic power tools for beginner woodworking."
    )
    await evaluator.verify(
        claim=tools_claim,
        node=tools_leaf,
        sources=st.urls,
        additional_instruction="Pass only if: (1) exactly three tools are listed in the answer for this item, "
                               "and (2) the cited sources support that these are basic beginner-friendly power tools (e.g., drill/driver, circular saw, jigsaw, sander). "
                               "If fewer or more than three are listed or sources do not support the choices, mark as Incorrect."
    )


async def verify_craft_store(evaluator: Evaluator, parent_node, extracted: WeekendPlanExtraction) -> None:
    """
    sunday_craft_store_open (leaf, critical under root)
    """
    cs = extracted.craft_store or CraftStoreInfo()

    craft_leaf = evaluator.add_leaf(
        id="sunday_craft_store_open",
        desc="Identify which of Hobby Lobby or Michaels is open on Sundays based on regular store-hours policy.",
        parent=parent_node,
        critical=True
    )
    store = (cs.sunday_open_store or "").strip()
    craft_claim = (
        f"Based on regular store-hours policy, the store that is open on Sundays is {store}."
        if store else
        "Based on regular store-hours policy, the answer identifies which store is open on Sundays."
    )
    await evaluator.verify(
        claim=craft_claim,
        node=craft_leaf,
        sources=cs.urls,
        additional_instruction="Pass only if the cited official hours/policy pages clearly indicate Sunday opening for the named store (among Hobby Lobby or Michaels). "
                               "If the answer is blank or contradicts the sources, mark as Incorrect."
    )


def add_supporting_urls_presence_node(evaluator: Evaluator, parent_node, extracted: WeekendPlanExtraction) -> None:
    """
    supporting_reference_urls (leaf/custom, critical under root)
    This node checks that the answer includes at least one supporting URL.
    """
    all_urls = collect_all_urls(extracted)
    evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id="supporting_reference_urls",
        desc="Include supporting reference URL(s) for verification of the claims made in the answer.",
        parent=parent_node,
        critical=True
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
    Evaluate an answer for the weekend DIY planning task (March 21–22, 2026).
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root as parallel (all critical children must pass)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_weekend_plan(),
        template_class=WeekendPlanExtraction,
        extraction_name="weekend_plan_extraction",
    )

    # Build verification tree
    await verify_kids_workshop(evaluator, root, extracted)
    await verify_side_table(evaluator, root, extracted)
    await verify_craft_store(evaluator, root, extracted)
    add_supporting_urls_presence_node(evaluator, root, extracted)

    # Return structured result
    return evaluator.get_summary()