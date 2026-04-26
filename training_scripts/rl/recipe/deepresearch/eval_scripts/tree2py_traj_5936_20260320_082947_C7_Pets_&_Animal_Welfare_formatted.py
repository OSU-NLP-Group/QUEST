import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "national_dog_show_2025_bis"
TASK_DESCRIPTION = """
I need comprehensive, verified information about the dog that won Best in Show at the 2025 National Dog Show (the dog show that airs on NBC on Thanksgiving Day). Please provide the following details: the dog's breed, registered show name, the handler's full name and hometown, the broadcast network and exact date the show aired, all co-owners' names and their hometowns, how many Best in Show titles this win represents for the dog, the organization that hosts this show, and which group the dog won before taking Best in Show.
"""

# Expected facts (ground truth requirements from rubric)
EXPECTED_BREED = "Belgian Sheepdog"
EXPECTED_HANDLER_NAME = "Daniel Martin"
EXPECTED_HANDLER_HOMETOWN = "Princeton, North Carolina"
EXPECTED_BROADCAST_NETWORK = "NBC"
EXPECTED_AIR_DATE = "November 27, 2025"  # Thanksgiving Day 2025
EXPECTED_COOWNERS = [
    {"name": "Connie Jasinski", "hometown": "Mount Pleasant, South Carolina"},  # Accept 'Mt. Pleasant, SC'
    {"name": "Pat Snow", "hometown": "Sapulpa, Oklahoma"},
    {"name": "Nancy Maye", "hometown": "Towanda, Kansas"},
]
EXPECTED_BIS_TITLE_COUNT_WORD = "seventh"  # also accept "7th" or "7"
EXPECTED_HOST_ORG = "Kennel Club of Philadelphia"
EXPECTED_GROUP = "Herding Group"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OwnerInfo(BaseModel):
    name: Optional[str] = None
    hometown: Optional[str] = None  # Prefer "City, State" (full or postal abbreviation)


class SourcesPerField(BaseModel):
    winner: List[str] = Field(default_factory=list)
    breed: List[str] = Field(default_factory=list)
    show_name: List[str] = Field(default_factory=list)
    handler: List[str] = Field(default_factory=list)
    broadcast: List[str] = Field(default_factory=list)
    coowners: List[str] = Field(default_factory=list)
    title_count: List[str] = Field(default_factory=list)
    host_org: List[str] = Field(default_factory=list)
    group: List[str] = Field(default_factory=list)
    other: List[str] = Field(default_factory=list)


class DogShowExtraction(BaseModel):
    breed: Optional[str] = None
    registered_show_name: Optional[str] = None

    handler_name: Optional[str] = None
    handler_hometown: Optional[str] = None

    broadcast_network: Optional[str] = None
    broadcast_date: Optional[str] = None  # Keep as free-form string

    coowners: List[OwnerInfo] = Field(default_factory=list)

    best_in_show_title_count: Optional[str] = None  # e.g., "7", "7th", "seventh"
    host_organization: Optional[str] = None
    pre_bis_group: Optional[str] = None

    sources: SourcesPerField = Field(default_factory=SourcesPerField)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_dogshow_info() -> str:
    return """
    Extract the specific details about the 2025 National Dog Show Best in Show winner mentioned in the answer.

    Return a JSON object with these fields:
    - breed: string or null
    - registered_show_name: string or null
    - handler_name: string or null
    - handler_hometown: string or null  (use "City, State" form if available; state may be full name or postal abbreviation)
    - broadcast_network: string or null
    - broadcast_date: string or null (e.g., "November 27, 2025" or "11/27/2025")
    - coowners: array of objects, each { "name": string or null, "hometown": string or null }
    - best_in_show_title_count: string or null (e.g., "7", "7th", or "seventh")
    - host_organization: string or null
    - pre_bis_group: string or null
    - sources: object with per-field URL arrays:
        {
          "winner": [...],
          "breed": [...],
          "show_name": [...],
          "handler": [...],
          "broadcast": [...],
          "coowners": [...],
          "title_count": [...],
          "host_org": [...],
          "group": [...],
          "other": [...]
        }

    IMPORTANT:
    - Extract only what is explicitly present in the answer text.
    - The 'sources' fields must include only URLs actually shown in the answer (plain URLs or markdown links).
    - If the answer gives no URL for a particular field, return an empty array for that field.
    - Keep strings exactly as written in the answer (do not normalize except trimming obvious surrounding whitespace).
    - For hometowns, short forms like "Mt. Pleasant, SC" are acceptable; do not invent expansions.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls or []:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def all_sources_union(extracted: DogShowExtraction) -> List[str]:
    all_lists = [
        extracted.sources.winner,
        extracted.sources.breed,
        extracted.sources.show_name,
        extracted.sources.handler,
        extracted.sources.broadcast,
        extracted.sources.coowners,
        extracted.sources.title_count,
        extracted.sources.host_org,
        extracted.sources.group,
        extracted.sources.other,
    ]
    flat: List[str] = []
    for lst in all_lists:
        if lst:
            flat.extend(lst)
    return dedup_urls(flat)


def field_sources(extracted: DogShowExtraction, key: str) -> List[str]:
    srcs = getattr(extracted.sources, key, []) or []
    if srcs:
        return dedup_urls(srcs)
    # Fallback to union (still better than no sources at all)
    union = all_sources_union(extracted)
    return union


# --------------------------------------------------------------------------- #
# Build verification subtree (URL-grounded checks)                            #
# --------------------------------------------------------------------------- #
async def build_verification_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: DogShowExtraction,
    reg_provided_node
) -> Dict[str, Any]:
    """
    Create the 'Verification' parallel subtree and perform URL-grounded checks for each major fact.
    If no URLs are available for a check, the corresponding node is set to failed immediately.
    """
    verification_node = evaluator.add_parallel(
        id="verification",
        desc="Provides verifiable support (e.g., citations/URLs) for each major requested fact.",
        parent=parent_node,
        critical=True
    )

    created = {}

    # Helper to add a verify leaf with sources, failing if no sources provided
    async def _add_verified_leaf(node_id: str, desc: str, claim: str, src_key: str, extra_prereq_nodes=None, add_ins: str = "None"):
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=verification_node,
            critical=True
        )
        created[node_id] = node
        urls = field_sources(extracted, src_key)
        # If truly no URLs (even after fallback), fail immediately to enforce source-grounding
        if len(dedup_urls(getattr(extracted.sources, src_key, []))) == 0:
            node.score = 0.0
            node.status = "failed"
            return
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=add_ins,
            extra_prerequisites=[reg_provided_node] if (extra_prereq_nodes is None and node_id == "verify_registered_show_name") else extra_prereq_nodes
        )

    # Verify winner identity (needs registered show name)
    rn = extracted.registered_show_name or ""
    await _add_verified_leaf(
        node_id="verify_winner_identity",
        desc="Includes a verifiable citation/URL supporting the Best in Show winner identification.",
        claim=f"The registered show name '{rn}' won Best in Show at the 2025 National Dog Show (the NBC Thanksgiving broadcast).",
        src_key="winner",
        extra_prereq_nodes=[reg_provided_node],
        add_ins="Treat minor punctuation or spacing variations in the show name as equivalent. Ensure the page explicitly ties this dog to 'Best in Show' at the 2025 National Dog Show."
    )

    # Verify breed
    await _add_verified_leaf(
        node_id="verify_breed",
        desc="Includes a verifiable citation/URL supporting the dog's breed.",
        claim=f"The 2025 National Dog Show Best in Show winner is a {EXPECTED_BREED}.",
        src_key="breed",
        add_ins="Allow synonyms like 'Belgian Shepherd (Groenendael)' to match 'Belgian Sheepdog'."
    )

    # Verify registered show name
    await _add_verified_leaf(
        node_id="verify_registered_show_name",
        desc="Includes a verifiable citation/URL supporting the dog's registered show name.",
        claim=f"The registered show name of the 2025 National Dog Show Best in Show winner is '{rn}'.",
        src_key="show_name",
        extra_prereq_nodes=[reg_provided_node],
        add_ins="Confirm that the page lists the dog's official registered show name. Minor formatting differences are acceptable."
    )

    # Verify handler
    await _add_verified_leaf(
        node_id="verify_handler_name_and_hometown",
        desc="Includes a verifiable citation/URL supporting the handler’s full name and hometown.",
        claim=f"The handler of the 2025 National Dog Show Best in Show winner is {EXPECTED_HANDLER_NAME} from {EXPECTED_HANDLER_HOMETOWN}.",
        src_key="handler",
        add_ins="Allow state abbreviations (e.g., 'NC' for North Carolina)."
    )

    # Verify broadcast network and date
    await _add_verified_leaf(
        node_id="verify_broadcast_network_and_date",
        desc="Includes a verifiable citation/URL supporting the broadcast network and exact air date.",
        claim=f"The 2025 National Dog Show aired on {EXPECTED_BROADCAST_NETWORK} on {EXPECTED_AIR_DATE} (Thanksgiving Day).",
        src_key="broadcast",
        add_ins="Accept equivalent date formats like 'Nov. 27, 2025' or '11/27/2025'."
    )

    # Verify co-owners (full set and count)
    names_list = ", ".join([f"{c['name']} ({c['hometown']})" for c in EXPECTED_COOWNERS])
    await _add_verified_leaf(
        node_id="verify_coowners",
        desc="Includes a verifiable citation/URL supporting the full co-owner set (names and hometowns) and the stated count.",
        claim=f"The dog's co-owners are exactly: {names_list}. There are three co-owners in total.",
        src_key="coowners",
        add_ins="Treat 'Mt. Pleasant' and 'Mount Pleasant' as equivalent; allow state abbreviations (e.g., 'SC' for South Carolina)."
    )

    # Verify BIS title count
    await _add_verified_leaf(
        node_id="verify_bis_title_count",
        desc="Includes a verifiable citation/URL supporting how many Best in Show titles this win represents for the dog.",
        claim="This 2025 National Dog Show Best in Show victory represents the seventh Best in Show title for the dog.",
        src_key="title_count",
        add_ins="Allow '7', '7th', or 'seventh' to be treated as equivalent."
    )

    # Verify host organization
    await _add_verified_leaf(
        node_id="verify_host_organization",
        desc="Includes a verifiable citation/URL supporting the hosting organization.",
        claim=f"The National Dog Show is hosted by the {EXPECTED_HOST_ORG}.",
        src_key="host_org",
        add_ins="Abbreviation 'KCP' refers to the Kennel Club of Philadelphia."
    )

    # Verify group win
    await _add_verified_leaf(
        node_id="verify_group_win",
        desc="Includes a verifiable citation/URL supporting which group the dog won before Best in Show.",
        claim=f"Before winning Best in Show, the dog won the {EXPECTED_GROUP}.",
        src_key="group",
        add_ins="Small variations like 'Herding' vs 'Herding Group' are acceptable."
    )

    return created


# --------------------------------------------------------------------------- #
# Main evaluation logic                                                       #
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
    Evaluate an answer for the 2025 National Dog Show Best in Show information task.
    """
    # Initialize evaluator (root stays non-critical; we add a critical main node under it)
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

    # Add main critical node mirroring the rubric's top node
    main_node = evaluator.add_parallel(
        id="2025_National_Dog_Show_Best_in_Show_Winner_Information",
        desc="Provide comprehensive, verified information about the dog that won Best in Show at the 2025 National Dog Show (NBC Thanksgiving broadcast), satisfying all stated constraints.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted: DogShowExtraction = await evaluator.extract(
        prompt=prompt_extract_dogshow_info(),
        template_class=DogShowExtraction,
        extraction_name="extracted_dogshow_2025_bis"
    )

    # Record expected facts as ground truth context
    evaluator.add_ground_truth({
        "expected_breed": EXPECTED_BREED,
        "expected_handler_name": EXPECTED_HANDLER_NAME,
        "expected_handler_hometown": EXPECTED_HANDLER_HOMETOWN,
        "expected_broadcast_network": EXPECTED_BROADCAST_NETWORK,
        "expected_air_date": EXPECTED_AIR_DATE,
        "expected_coowners": EXPECTED_COOWNERS,
        "expected_bis_title_count": EXPECTED_BIS_TITLE_COUNT_WORD,
        "expected_host_org": EXPECTED_HOST_ORG,
        "expected_group": EXPECTED_GROUP
    }, gt_type="expected_facts")

    # 1) Registered show name provided (existence)
    reg_name_exists = bool(extracted.registered_show_name and extracted.registered_show_name.strip())
    reg_provided_node = evaluator.add_custom_node(
        result=reg_name_exists,
        id="Registered_Show_Name_Provided",
        desc="Provides the dog's registered show name.",
        parent=main_node,
        critical=True
    )

    # 2) Build Verification (URL-grounded) subtree first (will be used conceptually to enforce citations)
    await build_verification_subtree(evaluator, main_node, extracted, reg_provided_node)

    # 3) Winner identification (answer states and identifies the winner by name)
    winner_ident_leaf = evaluator.add_leaf(
        id="Winner_Identification",
        desc="Correctly identifies the dog as the official 2025 National Dog Show Best in Show winner.",
        parent=main_node,
        critical=True
    )
    rn = extracted.registered_show_name or ""
    winner_ident_claim = (
        f"In the answer, the dog '{rn}' is explicitly identified as the 2025 National Dog Show Best in Show winner."
        if rn else
        "In the answer, the official 2025 National Dog Show Best in Show winner is explicitly identified by name."
    )
    await evaluator.verify(
        claim=winner_ident_claim,
        node=winner_ident_leaf,
        additional_instruction="Focus on whether the answer text clearly identifies the official 2025 National Dog Show Best in Show winner by registered show name.",
        extra_prerequisites=[reg_provided_node] if rn else None
    )

    # 4) Breed is Belgian Sheepdog (answer states)
    breed_leaf = evaluator.add_leaf(
        id="Dog_Breed_Is_Belgian_Sheepdog",
        desc="States the dog's breed, and it is Belgian Sheepdog.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the dog's breed is stated as Belgian Sheepdog.",
        node=breed_leaf,
        additional_instruction="Treat 'Belgian Shepherd (Groenendael)' as equivalent to Belgian Sheepdog."
    )

    # 5) Handler is Daniel Martin from Princeton, NC (answer states)
    handler_leaf = evaluator.add_leaf(
        id="Handler_Is_Daniel_Martin_From_Princeton_NC",
        desc="Provides the handler’s full name and hometown, and they are Daniel Martin from Princeton, North Carolina.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the answer, the handler is {EXPECTED_HANDLER_NAME} from {EXPECTED_HANDLER_HOMETOWN}.",
        node=handler_leaf,
        additional_instruction="Allow 'NC' abbreviation for North Carolina and minor punctuation variations."
    )

    # 6) Broadcast is NBC on Thanksgiving 2025-11-27 (answer states)
    broadcast_leaf = evaluator.add_leaf(
        id="Broadcast_Is_NBC_On_Thanksgiving_2025_11_27",
        desc="Provides the broadcast network and exact air date, and they are NBC on Thanksgiving Day, November 27, 2025.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the answer, the broadcast network is {EXPECTED_BROADCAST_NETWORK} and the exact air date is {EXPECTED_AIR_DATE} (Thanksgiving Day).",
        node=broadcast_leaf,
        additional_instruction="Accept equivalent date formats like 'Nov. 27, 2025' or '11/27/2025'."
    )

    # 7) Co-owners parallel group (answer states)
    coowners_node = evaluator.add_parallel(
        id="CoOwners",
        desc="Provides all co-owners’ names and hometowns, matching the stated set and count.",
        parent=main_node,
        critical=True
    )

    # 7.1) Count is three (answer states)
    co_count_ok = len(extracted.coowners) == 3
    evaluator.add_custom_node(
        result=co_count_ok,
        id="CoOwner_Count_Is_Three",
        desc="States there are three co-owners.",
        parent=coowners_node,
        critical=True
    )

    # 7.2) Individual co-owners (answer states) - simple checks for presence/match in the answer
    # Helper to add a simple verification for each expected co-owner
    async def _add_coowner_state_leaf(node_id: str, name: str, hometown_variants: List[str]):
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=f"Includes co-owner {name} from {hometown_variants[0]}.",
            parent=coowners_node,
            critical=True
        )
        # Build a tolerant claim referencing the accepted variants
        pretty_hometowns = " | ".join(hometown_variants)
        claim = f"In the answer, one of the co-owners is {name} from {hometown_variants[0]}."
        add_ins = f"Consider hometown variants equivalent: {pretty_hometowns}. Allow state abbreviations as well."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            additional_instruction=add_ins
        )

    # Connie Jasinski (Mt./Mount Pleasant, South Carolina / SC)
    await _add_coowner_state_leaf(
        node_id="CoOwner_Connie_Jasinski_Mt_Pleasant_SC",
        name="Connie Jasinski",
        hometown_variants=[
            "Mount Pleasant, South Carolina",
            "Mt. Pleasant, South Carolina",
            "Mt Pleasant, South Carolina",
            "Mount Pleasant, SC",
            "Mt. Pleasant, SC",
            "Mt Pleasant, SC",
        ],
    )

    # Pat Snow (Sapulpa, Oklahoma / OK)
    await _add_coowner_state_leaf(
        node_id="CoOwner_Pat_Snow_Sapulpa_OK",
        name="Pat Snow",
        hometown_variants=[
            "Sapulpa, Oklahoma",
            "Sapulpa, OK",
        ],
    )

    # Nancy Maye (Towanda, Kansas / KS)
    await _add_coowner_state_leaf(
        node_id="CoOwner_Nancy_Maye_Towanda_KS",
        name="Nancy Maye",
        hometown_variants=[
            "Towanda, Kansas",
            "Towanda, KS",
        ],
    )

    # 8) BIS title count is seventh (answer states)
    bis_count_leaf = evaluator.add_leaf(
        id="Best_in_Show_Title_Count_Is_Seventh",
        desc="States how many Best in Show titles this win represents for the dog, and it is the seventh.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, this National Dog Show Best in Show win is the seventh Best in Show title for the dog.",
        node=bis_count_leaf,
        additional_instruction="Treat '7', '7th', and 'seventh' as equivalent ways of stating the count."
    )

    # 9) Host organization is Kennel Club of Philadelphia (answer states)
    host_leaf = evaluator.add_leaf(
        id="Host_Organization_Is_Kennel_Club_of_Philadelphia",
        desc="Identifies the organization that hosts this show, and it is the Kennel Club of Philadelphia.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the answer, the National Dog Show is hosted by the {EXPECTED_HOST_ORG}.",
        node=host_leaf,
        additional_instruction="Allow 'KCP' to refer to the Kennel Club of Philadelphia."
    )

    # 10) Pre-BIS group is Herding Group (answer states)
    group_leaf = evaluator.add_leaf(
        id="Pre_BIS_Group_Is_Herding_Group",
        desc="Identifies which group the dog won before taking Best in Show, and it is the Herding Group.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the answer, before Best in Show the dog won the {EXPECTED_GROUP}.",
        node=group_leaf,
        additional_instruction="Variant 'Herding' or 'Herding Group' are equivalent."
    )

    # Provide some custom info summary about extracted URLs (optional diagnostics)
    evaluator.add_custom_info(
        info={
            "total_urls_found": len(all_sources_union(extracted)),
            "per_field_counts": {
                "winner": len(extracted.sources.winner),
                "breed": len(extracted.sources.breed),
                "show_name": len(extracted.sources.show_name),
                "handler": len(extracted.sources.handler),
                "broadcast": len(extracted.sources.broadcast),
                "coowners": len(extracted.sources.coowners),
                "title_count": len(extracted.sources.title_count),
                "host_org": len(extracted.sources.host_org),
                "group": len(extracted.sources.group),
                "other": len(extracted.sources.other),
            }
        },
        info_type="url_statistics",
        info_name="url_stats"
    )

    # Return evaluation summary
    return evaluator.get_summary()