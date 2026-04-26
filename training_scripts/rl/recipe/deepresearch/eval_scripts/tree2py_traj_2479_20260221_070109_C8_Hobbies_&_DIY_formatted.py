import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "craft_store_holiday_projects"
TASK_DESCRIPTION = (
    "Find craft store locations in four different U.S. states, verify their holiday "
    "operating hours, and identify suitable beginner DIY projects with complete material "
    "lists for each location. Stores must be from Michaels, Hobby Lobby, or Joann."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StoreHours(BaseModel):
    thanksgiving: Optional[str] = None
    black_friday_open: Optional[str] = None
    christmas_eve: Optional[str] = None
    hours_sources: List[str] = Field(default_factory=list)


class ProjectInfo(BaseModel):
    project_type: Optional[str] = None
    materials: List[str] = Field(default_factory=list)
    tutorial_url: Optional[str] = None


class StoreInfo(BaseModel):
    chain: Optional[str] = None
    address: Optional[str] = None
    state: Optional[str] = None
    store_url: Optional[str] = None
    hours: Optional[StoreHours] = None
    project: Optional[ProjectInfo] = None


class StoresExtraction(BaseModel):
    stores: List[StoreInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stores() -> str:
    return """
Extract up to four craft store entries from the answer. Only include stores from these chains: Michaels, Hobby Lobby, or Joann (also written as JOANN / Jo-Ann / JOANN Fabric and Craft). Each store must be in a different U.S. state.

For each store, extract the following fields into an array 'stores':
- chain: The store chain name exactly as given (e.g., "Michaels", "Hobby Lobby", "Joann"). If given as "JOANN", "Jo-Ann", or "JOANN Fabric and Craft", keep as presented in the answer.
- address: The complete physical address string for the specific store location (street, city, state, ZIP) as presented in the answer.
- state: The U.S. state where the store is located, as presented in the answer (full name or USPS abbreviation is acceptable).
- store_url: A reference URL to the store’s official locator page or specific store information page (must be a valid URL explicitly shown in the answer).
- hours: An object with:
  - thanksgiving: The store's operating hours for Thanksgiving Day as presented in the answer (e.g., "Closed", "10 AM – 6 PM"). If not provided, set null.
  - black_friday_open: The opening time for Black Friday (day after Thanksgiving) as presented in the answer (e.g., "6 AM", "8:00 AM"). If not provided, set null.
  - christmas_eve: The store's operating hours for Christmas Eve as presented in the answer (e.g., "9 AM – 5 PM", "Closed early at 6 PM"). If not provided, set null.
  - hours_sources: All URLs cited in the answer that support these holiday hours (brand holiday hours pages, store-specific announcements, etc.). If none are given, return an empty array.
- project: An object with:
  - project_type: The specific beginner-level DIY holiday project type (e.g., "Thanksgiving centerpiece", "Christmas ornament"). If no project is given for this store, set null.
  - materials: A complete list of materials required for the project as presented in the answer. If unspecified, return an empty array.
  - tutorial_url: A URL to a tutorial or instructional guide for the project. If not provided, set null.

Rules:
- Extract only what appears in the answer; do not infer or add missing information.
- Ensure each store's 'store_url' and any 'hours_sources' are the exact URLs provided in the answer text (or shown in markdown links).
- Include at most the first four stores found in the order presented in the answer. If fewer than four stores are provided, return fewer entries.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(v: Optional[List[str]]) -> List[str]:
    return v if isinstance(v, list) else []


def _normalize_chain_instructions(chain: Optional[str]) -> str:
    base = (
        "Treat brand name variants as equivalent where reasonable:\n"
        "- Joann may appear as JOANN, Jo-Ann, or JOANN Fabric and Craft.\n"
        "- Michaels may include 'Michaels Stores' or 'Michaels Arts & Crafts'.\n"
        "- Hobby Lobby may include 'HobbyLobby' or 'Hobby Lobby Stores'.\n"
        "Minor differences in capitalization, punctuation, or added descriptors should be tolerated.\n"
    )
    if chain:
        return base + f"Focus on verifying this page belongs to the '{chain}' brand."
    return base


# --------------------------------------------------------------------------- #
# Verification for a single store                                             #
# --------------------------------------------------------------------------- #
async def verify_one_store(
    evaluator: Evaluator,
    parent_node,
    store: StoreInfo,
    index: int,
    prior_states: List[str]
) -> None:
    store_num = index + 1
    store_node = evaluator.add_parallel(
        id=f"store_location_{store_num}",
        desc=f"{['First','Second','Third','Fourth'][index]} craft store location meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Prepare handy accessors / defaults
    chain = (store.chain or "").strip()
    address = (store.address or "").strip()
    state = (store.state or "").strip()
    store_url = (store.store_url or "").strip()
    hours = store.hours or StoreHours()
    thanksgiving = (hours.thanksgiving or "").strip()
    black_friday_open = (hours.black_friday_open or "").strip()
    christmas_eve = (hours.christmas_eve or "").strip()
    hours_sources = _safe_list(hours.hours_sources)

    project = store.project or ProjectInfo()
    project_type = (project.project_type or "").strip()
    materials = project.materials or []
    tutorial_url = (project.tutorial_url or "").strip()

    # ---------------- Existence / sanity checks (custom nodes) ---------------- #
    evaluator.add_custom_node(
        result=bool(chain),
        id=f"store_{store_num}_chain_exists",
        desc="Chain name is provided",
        parent=store_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(address),
        id=f"store_{store_num}_address_exists",
        desc="Complete physical address is provided",
        parent=store_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(state),
        id=f"store_{store_num}_state_exists",
        desc="State is provided",
        parent=store_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(store_url),
        id=f"store_{store_num}_reference_url_exists",
        desc="Reference URL is provided",
        parent=store_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(thanksgiving),
        id=f"store_{store_num}_thanksgiving_hours_exists",
        desc="Thanksgiving Day hours are provided",
        parent=store_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(black_friday_open),
        id=f"store_{store_num}_black_friday_open_exists",
        desc="Black Friday opening time is provided",
        parent=store_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(christmas_eve),
        id=f"store_{store_num}_christmas_eve_hours_exists",
        desc="Christmas Eve hours are provided",
        parent=store_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(project_type),
        id=f"store_{store_num}_project_type_exists",
        desc="Beginner-level project type is provided",
        parent=store_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(materials) > 0,
        id=f"store_{store_num}_materials_list_exists",
        desc="Materials list is non-empty",
        parent=store_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(tutorial_url),
        id=f"store_{store_num}_project_source_exists",
        desc="Tutorial URL is provided",
        parent=store_node,
        critical=True
    )

    # ---------------- Chain verification (brand + official page) ------------- #
    chain_node = evaluator.add_leaf(
        id=f"store_{store_num}_chain_verification",
        desc="The store must be one of the three major U.S. craft store chains: Michaels, Hobby Lobby, or Joann",
        parent=store_node,
        critical=True
    )
    chain_claim = (
        f"This webpage is an official page (store information or locator) belonging to the '{chain}' brand "
        f"(one of Michaels, Hobby Lobby, or Joann)."
    )
    await evaluator.verify(
        claim=chain_claim,
        node=chain_node,
        sources=store_url,
        additional_instruction=_normalize_chain_instructions(chain),
    )

    # ---------------- Address verification ----------------------------------- #
    addr_node = evaluator.add_leaf(
        id=f"store_{store_num}_address",
        desc="Provide the complete physical address of the store location",
        parent=store_node,
        critical=True
    )
    addr_claim = f"The complete store address on this page is exactly: '{address}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_node,
        sources=store_url,
        additional_instruction="Allow minor punctuation or formatting variations (e.g., commas, line breaks). Focus on substantive match."
    )

    # ---------------- State verification (supported by page) ------------------ #
    state_supported_node = evaluator.add_leaf(
        id=f"store_{store_num}_state",
        desc=("The store must be located in a U.S. state"
              + ("" if index == 0 else f" different from the previous {'store' if index==1 else f'{index} stores'}' states")),
        parent=store_node,
        critical=True
    )
    state_supported_claim = f"The store shown on this page is located in the state: {state}."
    await evaluator.verify(
        claim=state_supported_claim,
        node=state_supported_node,
        sources=store_url,
        additional_instruction="Accept reasonable variants like full state name vs. USPS abbreviation. The location on the page should clearly correspond to this state."
    )

    # ---------------- State uniqueness (logic check) ------------------------- #
    if prior_states:
        state_unique_node = evaluator.add_leaf(
            id=f"store_{store_num}_state_unique",
            desc="The store's state is different from previously selected store states",
            parent=store_node,
            critical=True
        )
        unique_claim = f"The state '{state}' is not among the previously used states: {prior_states}."
        await evaluator.verify(
            claim=unique_claim,
            node=state_unique_node,
            sources=None,
            additional_instruction="Pure logical check; no URL needed. Treat state names and USPS abbreviations as equivalent when comparing."
        )

    # ---------------- Reference URL validity (official store/locator) -------- #
    ref_node = evaluator.add_leaf(
        id=f"store_{store_num}_reference_url",
        desc="Provide a reference URL from the official store locator or store information page",
        parent=store_node,
        critical=True
    )
    ref_claim = (
        f"This URL is an official {chain} store locator or store information page for a specific store location."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_node,
        sources=store_url,
        additional_instruction="Verify that the page belongs to the brand site and provides store-specific details (address, hours, etc.)."
    )

    # ---------------- Holiday hours verifications ----------------------------- #
    # Build sources preference: hours_sources (if any) + store_url as fallback
    holiday_sources: List[str] = []
    if hours_sources:
        holiday_sources.extend(hours_sources)
    if store_url:
        holiday_sources.append(store_url)

    tg_node = evaluator.add_leaf(
        id=f"store_{store_num}_thanksgiving_hours",
        desc="Verify and provide the store's operating hours specifically for Thanksgiving Day",
        parent=store_node,
        critical=True
    )
    tg_claim = f"On Thanksgiving Day, this store's hours are: {thanksgiving}."
    await evaluator.verify(
        claim=tg_claim,
        node=tg_node,
        sources=holiday_sources if holiday_sources else None,
        additional_instruction="Confirm the Thanksgiving Day hours (including 'Closed' if applicable). Allow simple phrasing differences but ensure the meaning matches."
    )

    bf_node = evaluator.add_leaf(
        id=f"store_{store_num}_black_friday_hours",
        desc="Verify and provide the store's opening time for Black Friday (day after Thanksgiving)",
        parent=store_node,
        critical=True
    )
    bf_claim = f"On Black Friday (the day after Thanksgiving), this store opens at {black_friday_open}."
    await evaluator.verify(
        claim=bf_claim,
        node=bf_node,
        sources=holiday_sources if holiday_sources else None,
        additional_instruction="Verify specifically the opening time on Black Friday; allow time format variations (e.g., '6 AM' vs '6:00 AM')."
    )

    ce_node = evaluator.add_leaf(
        id=f"store_{store_num}_christmas_eve_hours",
        desc="Verify and provide the store's operating hours specifically for Christmas Eve",
        parent=store_node,
        critical=True
    )
    ce_claim = f"On Christmas Eve, this store's hours are: {christmas_eve}."
    await evaluator.verify(
        claim=ce_claim,
        node=ce_node,
        sources=holiday_sources if holiday_sources else None,
        additional_instruction="Confirm the Christmas Eve hours (e.g., special closing times). Allow minor formatting differences."
    )

    # ---------------- Project verification ----------------------------------- #
    proj_type_node = evaluator.add_leaf(
        id=f"store_{store_num}_project_type",
        desc=("Identify one specific beginner-level DIY craft project type "
              "(e.g., Thanksgiving centerpiece, Christmas ornament, woodworking decoration) suitable for the upcoming holiday season"),
        parent=store_node,
        critical=True
    )
    proj_type_claim = (
        f"This tutorial page describes a holiday project of type '{project_type}' that is suitable for beginners."
    )
    await evaluator.verify(
        claim=proj_type_claim,
        node=proj_type_node,
        sources=tutorial_url if tutorial_url else None,
        additional_instruction="Treat labels like 'easy', 'beginner-friendly', or 'simple' as beginner-level. The page should clearly be a tutorial for this project type."
    )

    materials_node = evaluator.add_leaf(
        id=f"store_{store_num}_materials_list",
        desc="Provide a complete list of materials needed for the identified DIY project",
        parent=store_node,
        critical=True
    )
    materials_claim = (
        f"The tutorial lists the following materials for the project: {materials}."
    )
    await evaluator.verify(
        claim=materials_claim,
        node=materials_node,
        sources=tutorial_url if tutorial_url else None,
        additional_instruction="Allow minor naming variations or brand substitutions; verify that the listed items (or clear equivalents) are included in the tutorial’s materials list."
    )

    proj_src_node = evaluator.add_leaf(
        id=f"store_{store_num}_project_source",
        desc="Provide a URL to a tutorial or guide for the identified DIY project",
        parent=store_node,
        critical=True
    )
    proj_src_claim = "This URL is a tutorial or instructional guide page for the described project."
    await evaluator.verify(
        claim=proj_src_claim,
        node=proj_src_node,
        sources=tutorial_url if tutorial_url else None,
        additional_instruction="Verify that the page presents step-by-step instructions, materials, or clear guidance to complete the project."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the craft store holiday projects task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root should allow parallel evaluation of each store
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
        prompt=prompt_extract_stores(),
        template_class=StoresExtraction,
        extraction_name="stores_extraction"
    )

    # Keep only the first four stores; pad with empties if fewer
    stores: List[StoreInfo] = list(extracted.stores[:4])
    while len(stores) < 4:
        stores.append(StoreInfo())

    # Build verification for each store
    prior_states: List[str] = []
    for idx in range(4):
        await verify_one_store(
            evaluator=evaluator,
            parent_node=root,
            store=stores[idx],
            index=idx,
            prior_states=prior_states.copy()
        )
        # Track state for uniqueness checks of subsequent stores
        st = (stores[idx].state or "").strip()
        if st:
            # Normalize to upper for simple comparison; keep as-is in verification
            prior_states.append(st)

    return evaluator.get_summary()