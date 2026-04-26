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
TASK_ID = "fastfood_chains_four_criteria"
TASK_DESCRIPTION = """Identify four fast food restaurant chains operating in the United States that meet all of the following criteria:

1. FDA Menu Labeling Compliance: The chain must operate 20 or more locations under the same name nationwide (making it subject to FDA menu labeling requirements), and must display calorie information on its menus and menu boards.

2. Multi-State Geographic Presence: The chain must have operational locations in at least three of the following major states: California, Texas, Florida, and New York.

3. Third-Party Delivery Integration: The chain must be partnered with at least two of the following major delivery platforms: DoorDash, Uber Eats, or Grubhub.

For each of the four chains you identify, provide:
- The official name of the restaurant chain
- The chain's official website URL
- The approximate number of US locations (with a source URL)
- Evidence that calorie information is displayed on menus (with a source URL showing menu with calorie information)
- Documentation of which specific states (among CA, TX, FL, NY) the chain operates in (with a source URL)
- Documentation of which delivery platforms (among DoorDash, Uber Eats, Grubhub) the chain partners with (with a source URL)

All four chains must be distinct from one another and must fully satisfy all the specified criteria.
"""

MAJOR_STATES = {"CA": "California", "TX": "Texas", "FL": "Florida", "NY": "New York"}
PLATFORM_CANONICAL = {"doordash": "DoorDash", "door dash": "DoorDash",
                      "uber eats": "Uber Eats", "ubereats": "Uber Eats", "uber-eats": "Uber Eats",
                      "grubhub": "Grubhub", "grub hub": "Grubhub"}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ChainItem(BaseModel):
    chain_name: Optional[str] = None
    chain_website: Optional[str] = None
    location_count_text: Optional[str] = None
    location_count_source_url: Optional[str] = None
    calorie_menu_source_url: Optional[str] = None
    states: List[str] = Field(default_factory=list)
    states_source_url: Optional[str] = None
    delivery_platforms: List[str] = Field(default_factory=list)
    delivery_source_url: Optional[str] = None


class FastFoodChainsExtraction(BaseModel):
    chains: List[ChainItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chains() -> str:
    return """
    Extract up to four fast food restaurant chains from the answer that claim to meet all of the specified criteria. For each chain, return an object with the following fields:

    - chain_name: The official name of the restaurant chain (string).
    - chain_website: The official website URL for the chain (string URL). If multiple are present, pick the main brand homepage.
    - location_count_text: The approximate number of US locations as described in the answer (string, keep as-is; examples: "120+", "around 150", "over 20", "300").
    - location_count_source_url: A URL cited in the answer that provides the store/location count or relevant evidence about number of locations (string URL).
    - calorie_menu_source_url: A URL cited in the answer that shows a menu or menu board with calorie information (string URL). This can be a menu webpage or a PDF menu.
    - states: A list of states among the following four: CA, TX, FL, NY. Only include these abbreviations. Normalize state names to their two-letter abbreviations exactly: "CA", "TX", "FL", "NY". Ignore other states.
    - states_source_url: A URL cited in the answer that confirms state-level presence (e.g., store locator page, locations page) (string URL).
    - delivery_platforms: A list of delivery platforms among exactly these canonical names: "DoorDash", "Uber Eats", "Grubhub". Normalize names if necessary and include only these if mentioned.
    - delivery_source_url: A URL cited in the answer that confirms delivery partnerships (string URL). This can be the chain's own delivery page, or partner platform pages.

    Rules:
    - Extract only information explicitly present in the provided answer. Do not invent any data.
    - If any field is missing, set it to null (for strings) or an empty list (for lists).
    - Always include full URLs with protocol (http:// or https://).
    - For states, only include items from {CA, TX, FL, NY} and use abbreviations exactly.
    - For delivery_platforms, only include canonical names from {"DoorDash", "Uber Eats", "Grubhub"}.

    Return a JSON object with a single field 'chains' that is an array of up to four chain objects following this schema.
    """


# --------------------------------------------------------------------------- #
# Helper normalization functions                                              #
# --------------------------------------------------------------------------- #
def normalize_states(raw_states: List[str]) -> List[str]:
    if not raw_states:
        return []
    norm = []
    for s in raw_states:
        if not s:
            continue
        t = s.strip().upper()
        if t in MAJOR_STATES:
            norm.append(t)
        else:
            # Try mapping full names to abbreviations
            low = s.strip().lower()
            if low == "california":
                norm.append("CA")
            elif low == "texas":
                norm.append("TX")
            elif low == "florida":
                norm.append("FL")
            elif low == "new york":
                norm.append("NY")
    # Deduplicate while preserving order
    seen = set()
    out = []
    for x in norm:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_platforms(raw_platforms: List[str]) -> List[str]:
    if not raw_platforms:
        return []
    norm = []
    for p in raw_platforms:
        if not p:
            continue
        key = p.strip().lower()
        canonical = PLATFORM_CANONICAL.get(key)
        if canonical:
            norm.append(canonical)
        else:
            # Try simple contains
            if "doordash" in key:
                norm.append("DoorDash")
            elif "uber" in key and "eat" in key:
                norm.append("Uber Eats")
            elif "grubhub" in key or ("grub" in key and "hub" in key):
                norm.append("Grubhub")
    # Deduplicate while preserving order
    seen = set()
    out = []
    for x in norm:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_chain(
        evaluator: Evaluator,
        parent_node,
        chain: ChainItem,
        chain_index: int
) -> None:
    """
    Build verification subtree for one chain, implementing all rubric checks.
    The chain node is critical: failing any essential criterion should fail the chain.
    """
    idx = chain_index + 1
    chain_node = evaluator.add_parallel(
        id=f"chain_{idx}",
        desc=f"{['First','Second','Third','Fourth'][chain_index]} qualified fast food chain meeting all requirements",
        parent=parent_node,
        critical=True
    )

    # --------------------------- Chain identification --------------------------- #
    ident_node = evaluator.add_parallel(
        id=f"chain_{idx}_chain_identification",
        desc="Chain name and basic information provided",
        parent=chain_node,
        critical=True
    )

    # chain_name existence (critical)
    evaluator.add_custom_node(
        result=bool(chain.chain_name and chain.chain_name.strip()),
        id=f"chain_{idx}_chain_name",
        desc="Official name of the restaurant chain",
        parent=ident_node,
        critical=True
    )

    # website provided existence (critical, to gate verification)
    website_exists_node = evaluator.add_custom_node(
        result=bool(chain.chain_website and chain.chain_website.strip()),
        id=f"chain_{idx}_chain_website_provided",
        desc="Official website URL is provided",
        parent=ident_node,
        critical=True
    )

    # website verification (critical)
    website_verify_leaf = evaluator.add_leaf(
        id=f"chain_{idx}_chain_website",
        desc="URL of chain's official website",
        parent=ident_node,
        critical=True
    )
    website_claim = f"This webpage is the official website of the restaurant chain named '{chain.chain_name or ''}'."
    await evaluator.verify(
        claim=website_claim,
        node=website_verify_leaf,
        sources=chain.chain_website,
        additional_instruction="Verify whether the page clearly represents the brand's official site (branding, trademark notices, official messaging). Allow reasonable variants but require clear branding."
    )

    # ------------------ FDA menu labeling compliance ------------------ #
    fda_node = evaluator.add_parallel(
        id=f"chain_{idx}_fda_menu_labeling_compliance",
        desc="Verification that chain is subject to and complies with FDA menu labeling requirements",
        parent=chain_node,
        critical=True
    )

    # Minimum location threshold
    min_loc_node = evaluator.add_parallel(
        id=f"chain_{idx}_minimum_location_threshold",
        desc="Chain operates 20 or more locations under the same name nationwide",
        parent=fda_node,
        critical=True
    )

    # location_count_source existence (critical)
    loc_src_exists = evaluator.add_custom_node(
        result=bool(chain.location_count_source_url and chain.location_count_source_url.strip()),
        id=f"chain_{idx}_location_count_source",
        desc="URL reference providing location count data",
        parent=min_loc_node,
        critical=True
    )

    # location_count_verification (critical)
    loc_verify_leaf = evaluator.add_leaf(
        id=f"chain_{idx}_location_count_verification",
        desc="Specific number of US locations documented and verified",
        parent=min_loc_node,
        critical=True
    )
    loc_claim = f"The chain '{chain.chain_name or ''}' operates 20 or more locations under the same name nationwide in the United States."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_verify_leaf,
        sources=chain.location_count_source_url,
        additional_instruction="Use the provided source to confirm that the brand operates at least 20 locations nationwide (US). Accept wording such as '20+', 'over 20', 'more than 20', or any larger number."
    )

    # Calorie labeling implementation
    calorie_node = evaluator.add_parallel(
        id=f"chain_{idx}_calorie_labeling_implementation",
        desc="Chain displays calorie information on menus and menu boards as required",
        parent=fda_node,
        critical=True
    )

    # calorie_display_source existence (critical)
    cal_src_exists = evaluator.add_custom_node(
        result=bool(chain.calorie_menu_source_url and chain.calorie_menu_source_url.strip()),
        id=f"chain_{idx}_calorie_display_source",
        desc="URL reference showing menu with calorie information",
        parent=calorie_node,
        critical=True
    )

    # calorie_display_confirmed (critical)
    calorie_leaf = evaluator.add_leaf(
        id=f"chain_{idx}_calorie_display_confirmed",
        desc="Evidence of calorie information display on menu materials",
        parent=calorie_node,
        critical=True
    )
    calorie_claim = "This provided menu or menu board clearly displays calorie information for menu items (e.g., per item or per portion)."
    await evaluator.verify(
        claim=calorie_claim,
        node=calorie_leaf,
        sources=chain.calorie_menu_source_url,
        additional_instruction="Confirm the presence of calorie values (e.g., numbers labeled 'cal', 'calories', or a dedicated calorie column) on the menu page/PDF."
    )

    # ---------------- Multi-state geographic presence ----------------- #
    multi_node = evaluator.add_sequential(
        id=f"chain_{idx}_multi_state_geographic_presence",
        desc="Chain has operational presence across multiple major US states",
        parent=chain_node,
        critical=True
    )

    major_node = evaluator.add_parallel(
        id=f"chain_{idx}_major_state_coverage",
        desc="Chain has locations in at least three of these states: California, Texas, Florida, New York",
        parent=multi_node,
        critical=True
    )

    norm_states = normalize_states(chain.states)
    # state_locations_identified (critical)
    evaluator.add_custom_node(
        result=len(norm_states) >= 3,
        id=f"chain_{idx}_state_locations_identified",
        desc="Specific states where chain operates are documented",
        parent=major_node,
        critical=True
    )

    # state_presence_source (critical)
    states_leaf = evaluator.add_leaf(
        id=f"chain_{idx}_state_presence_source",
        desc="URL reference confirming state-level presence",
        parent=major_node,
        critical=True
    )
    states_human = ", ".join(norm_states) if norm_states else "none"
    states_claim = f"The provided source confirms that the chain operates in these states among the target set: {states_human}. The chain is present in at least three of CA, TX, FL, NY."
    await evaluator.verify(
        claim=states_claim,
        node=states_leaf,
        sources=chain.states_source_url,
        additional_instruction="Look for a store locator or locations page that shows or allows selection of CA, TX, FL, NY. Confirm presence in at least three of them."
    )

    # ------------- Third-party delivery integration ------------------- #
    delivery_node = evaluator.add_sequential(
        id=f"chain_{idx}_third_party_delivery_integration",
        desc="Chain partners with major third-party delivery platforms",
        parent=chain_node,
        critical=True
    )

    platform_node = evaluator.add_parallel(
        id=f"chain_{idx}_platform_partnership_count",
        desc="Chain is partnered with at least two of: DoorDash, Uber Eats, Grubhub",
        parent=delivery_node,
        critical=True
    )

    norm_platforms = normalize_platforms(chain.delivery_platforms)
    # delivery_platforms_identified (critical)
    evaluator.add_custom_node(
        result=len(norm_platforms) >= 2,
        id=f"chain_{idx}_delivery_platforms_identified",
        desc="Specific delivery platforms partnerships are documented",
        parent=platform_node,
        critical=True
    )

    # delivery_partnership_source (critical)
    delivery_leaf = evaluator.add_leaf(
        id=f"chain_{idx}_delivery_partnership_source",
        desc="URL reference confirming delivery platform partnerships",
        parent=platform_node,
        critical=True
    )
    platforms_text = ", ".join(norm_platforms) if norm_platforms else "none"
    delivery_claim = f"The provided source confirms that the chain partners with at least two of DoorDash, Uber Eats, Grubhub. Specifically mentioned: {platforms_text}."
    await evaluator.verify(
        claim=delivery_claim,
        node=delivery_leaf,
        sources=chain.delivery_source_url,
        additional_instruction="Confirm via the chain's order/delivery page, a partners page, or partner platform pages showing the brand. Evidence should explicitly show at least two of the specified platforms."
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
    Evaluate an answer for the fast food chains criteria task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent verification per chain
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

    # IMPORTANT: Root must be non-critical if any child is non-critical; however the task requires all four chains to be valid.
    # To enforce that, we set all child chain nodes critical and keep root non-critical aggregation. Then add a critical distinctness node and a critical "all_chains_present" existence gate.
    # Alternatively, we can mark root critical, but then all children must be critical. We'll mark root critical by adding a gate node and making chain nodes critical as we did.
    # Since root was initialized non-critical, we add a critical wrapper node under root to enforce global constraints.
    global_gate = evaluator.add_parallel(
        id="global_requirements",
        desc="Global requirements: All four distinct chains must be provided and each must meet all criteria",
        parent=root,
        critical=True
    )

    # Extract chains from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_chains(),
        template_class=FastFoodChainsExtraction,
        extraction_name="fastfood_chains"
    )

    # Prepare exactly four chains
    chains = (extraction.chains or [])[:4]
    while len(chains) < 4:
        chains.append(ChainItem())

    # Add distinctness check
    names = [c.chain_name.strip() for c in chains if c.chain_name]
    unique_names = set(n.lower() for n in names if n)
    all_four_named = len([c for c in chains if c.chain_name and c.chain_name.strip()]) == 4
    distinct_all = all_four_named and (len(unique_names) == 4)

    evaluator.add_custom_node(
        result=distinct_all,
        id="distinct_chains",
        desc="All four chains are distinct from one another (no duplicates and all names provided)",
        parent=global_gate,
        critical=True
    )

    # Add existence checkpoint: ensure four chains are present (names provided)
    evaluator.add_custom_node(
        result=all_four_named,
        id="four_chains_present",
        desc="Four chain names are provided",
        parent=global_gate,
        critical=True
    )

    # Verify each chain under the global gate
    for i in range(4):
        await verify_chain(evaluator, global_gate, chains[i], i)

    return evaluator.get_summary()