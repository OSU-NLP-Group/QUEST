import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task metadata                                                               #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_platforms_2026_requirements"
TASK_DESCRIPTION = (
    "A gaming content creator is building their 2026 gaming setup and needs to identify current-generation "
    "gaming platforms that can support both their content creation workflow and upcoming major game releases. "
    "Find at least two gaming platforms (consoles, handheld gaming devices, or prebuilt gaming systems) that "
    "meet ALL of the following requirements: (1) RAM Requirement: The platform must have 16GB of RAM or more, "
    "(2) Storage Requirement: The platform must have NVMe SSD storage with a capacity of at least 512GB, "
    "(3) Game Compatibility: The platform must officially support Grand Theft Auto VI (GTA VI) when it releases "
    "on November 19, 2026, (4) Game Library: The platform must either have exclusive games available OR provide "
    "access to a classic game library containing at least 150 games. For each platform you identify, provide the "
    "platform name, verification that it meets each of the four requirements above, and reference URLs from official "
    "sources or reliable tech publications that confirm each specification."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PlatformEntry(BaseModel):
    name: Optional[str] = None

    ram_text: Optional[str] = None
    ram_sources: List[str] = Field(default_factory=list)

    storage_type_text: Optional[str] = None
    storage_type_sources: List[str] = Field(default_factory=list)

    storage_capacity_text: Optional[str] = None
    storage_capacity_sources: List[str] = Field(default_factory=list)

    gta_support_text: Optional[str] = None
    gta_sources: List[str] = Field(default_factory=list)

    library_text: Optional[str] = None
    library_sources: List[str] = Field(default_factory=list)


class PlatformsExtraction(BaseModel):
    platforms: List[PlatformEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_platforms() -> str:
    return """
    Identify up to four (4) distinct gaming platforms listed in the answer. A "platform" can be a home console, a handheld gaming device, or a prebuilt desktop PC system identified by make/model.
    For each platform you find in the answer, extract the following fields strictly from the answer text:

    - name: The platform name (e.g., "PlayStation 5", "Xbox Series X", "ROG Ally X (prebuilt config)", etc.)
    - ram_text: The statement/figure given in the answer about RAM or memory capacity (e.g., "16GB GDDR6 unified memory", "32GB RAM", etc.)
    - ram_sources: The list of URLs cited in the answer that support the RAM/memory capacity. Extract only actual URLs mentioned (plain or markdown). If none, return an empty list.
    - storage_type_text: The statement in the answer about storage type (e.g., "NVMe SSD", "PCIe 4.0 NVMe M.2", etc.)
    - storage_type_sources: URLs cited that support the storage type. Only extract URLs present in the answer. If none, return an empty list.
    - storage_capacity_text: The statement in the answer about internal storage capacity (e.g., "1TB", "825GB", "512GB", etc.)
    - storage_capacity_sources: URLs cited that support the storage capacity. Only extract URLs present in the answer. If none, return an empty list.
    - gta_support_text: The statement in the answer about GTA VI platform support/compatibility for this platform.
    - gta_sources: URLs cited that support GTA VI platform compatibility. Prefer official sources (Rockstar or platform holder) or reliable tech publications. Extract only URLs actually present. If none, return an empty list.
    - library_text: The statement about EXCLUSIVE games availability OR access to a CLASSIC/retro game library/catalog with at least 150 titles.
    - library_sources: URLs cited that support the exclusives or the classic game library claim. Extract only URLs actually present. If none, return an empty list.

    IMPORTANT EXTRACTION RULES:
    - Do not invent or infer any data. If the answer does not provide a particular field or URL(s), output null for text fields and [] for URL arrays.
    - Extract URLs exactly as they appear (plain links or inside markdown). Include the full URL with protocol.
    - Keep the original phrasing of *_text fields from the answer; do not paraphrase.
    - If the answer lists more than four platforms, keep only the first four in order of appearance. If fewer, extract all available.

    Return a JSON object with a single key "platforms" that is an array of up to 4 PlatformEntry objects as defined.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _norm_name(name: Optional[str], fallback: str) -> str:
    return (name or "").strip() or fallback


async def _verify_platform(
    evaluator: Evaluator,
    parent_node,
    platform: PlatformEntry,
    index_zero_based: int,
    platform_is_critical: bool = False
) -> None:
    """
    Build verification subtree for a single platform and run verifications.
    """
    idx = index_zero_based + 1
    plat_name = _norm_name(platform.name, f"Platform #{idx}")

    # Platform container (Parallel). Keep non-critical at this level to avoid cross-platform gating side effects.
    platform_node = evaluator.add_parallel(
        id=f"platform_{idx}",
        desc=f"{plat_name}: meets all specified requirements",
        parent=parent_node,
        critical=False
    )

    # ------------------ RAM requirement group ------------------ #
    ram_group = evaluator.add_parallel(
        id=f"platform_{idx}_ram",
        desc="RAM: at least 16 GB",
        parent=platform_node,
        critical=True
    )

    # Reference existence (critical)
    evaluator.add_custom_node(
        result=bool(platform.ram_sources),
        id=f"platform_{idx}_ram_reference",
        desc="URL reference provided for RAM specification",
        parent=ram_group,
        critical=True
    )

    # Actual RAM verification (critical)
    ram_met_node = evaluator.add_leaf(
        id=f"platform_{idx}_ram_met",
        desc="Platform has 16GB RAM or more",
        parent=ram_group,
        critical=True
    )
    ram_claim = (
        f"{plat_name} has at least 16 GB of system memory (RAM). "
        f"Unified memory (e.g., GDDR6/LPDDR) of 16 GB or more qualifies."
    )

    # ------------------ Storage type group ------------------ #
    stype_group = evaluator.add_parallel(
        id=f"platform_{idx}_storage_type",
        desc="Storage type: NVMe SSD",
        parent=platform_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(platform.storage_type_sources),
        id=f"platform_{idx}_storage_type_reference",
        desc="URL reference provided for storage type specification",
        parent=stype_group,
        critical=True
    )

    stype_met_node = evaluator.add_leaf(
        id=f"platform_{idx}_storage_type_met",
        desc="Platform has NVMe SSD storage",
        parent=stype_group,
        critical=True
    )
    stype_claim = (
        f"{plat_name} uses NVMe SSD storage (e.g., M.2 NVMe over PCIe). "
        f"Equivalent wording that clearly indicates NVMe SSD is acceptable."
    )

    # ------------------ Storage capacity group ------------------ #
    scapa_group = evaluator.add_parallel(
        id=f"platform_{idx}_storage_capacity",
        desc="Storage capacity: at least 512 GB",
        parent=platform_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(platform.storage_capacity_sources),
        id=f"platform_{idx}_storage_capacity_reference",
        desc="URL reference provided for storage capacity specification",
        parent=scapa_group,
        critical=True
    )

    scapa_met_node = evaluator.add_leaf(
        id=f"platform_{idx}_storage_capacity_met",
        desc="Platform has at least 512GB storage capacity",
        parent=scapa_group,
        critical=True
    )
    scapa_claim = (
        f"{plat_name} has internal storage capacity of at least 512 GB "
        f"(values such as 512 GB, 825 GB, 1 TB or larger all qualify)."
    )

    # ------------------ GTA VI support group ------------------ #
    gta_group = evaluator.add_parallel(
        id=f"platform_{idx}_gta",
        desc="GTA VI official platform support",
        parent=platform_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(platform.gta_sources),
        id=f"platform_{idx}_gta_reference",
        desc="URL reference provided for GTA VI platform compatibility",
        parent=gta_group,
        critical=True
    )

    gta_met_node = evaluator.add_leaf(
        id=f"platform_{idx}_gta_met",
        desc="Platform supports GTA VI (officially confirmed for platform)",
        parent=gta_group,
        critical=True
    )
    gta_claim = (
        f"Grand Theft Auto VI (GTA VI) is officially confirmed for {plat_name} by Rockstar Games "
        f"or the platform holder (e.g., PlayStation/Xbox). Rumors/speculation do not count."
    )

    # ------------------ Game library group ------------------ #
    library_group = evaluator.add_parallel(
        id=f"platform_{idx}_library",
        desc="Exclusive titles OR classic catalog (>=150 games)",
        parent=platform_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(platform.library_sources),
        id=f"platform_{idx}_library_reference",
        desc="URL reference provided for game library or exclusives",
        parent=library_group,
        critical=True
    )

    library_met_node = evaluator.add_leaf(
        id=f"platform_{idx}_library_met",
        desc="Platform has exclusives OR access to a 150+ classic game library",
        parent=library_group,
        critical=True
    )
    library_claim = (
        f"{plat_name} satisfies the game library requirement: EITHER it offers first‑party/console‑exclusive titles, "
        f"OR it provides access to a classic/retro catalog with at least 150 games. Meeting either condition is sufficient."
    )

    # ------------------ Run verifications (URL‑grounded) ------------------ #
    claims_and_sources = [
        (ram_claim, platform.ram_sources, ram_met_node,
         "This must be explicitly supported by the provided URL(s). Accept synonyms such as 'memory', 'system memory', "
         "'unified memory', 'GDDR/LPDDR'. The capacity must be 16 GB or more."),

        (stype_claim, platform.storage_type_sources, stype_met_node,
         "Confirm the storage technology is NVMe SSD (e.g., M.2 NVMe over PCIe). Equivalent wording acceptable."),

        (scapa_claim, platform.storage_capacity_sources, scapa_met_node,
         "Confirm the internal storage capacity is >= 512 GB (e.g., 512 GB, 825 GB, 1 TB)."),

        (gta_claim, platform.gta_sources, gta_met_node,
         "Only accept official confirmation from Rockstar Games or the platform holder (or their official store/listing). "
         "Mere rumors or unverified speculation should be rejected."),

        (library_claim, platform.library_sources, library_met_node,
         "Pass if EITHER: (a) exclusive/first‑party titles exist for this platform, OR (b) access to a classic/retro "
         "catalog with >=150 games is provided. Either is sufficient."),
    ]

    # Execute URL verifications in parallel for this platform
    await evaluator.batch_verify(claims_and_sources)


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
    Entry point: evaluate an answer against the 2026 gaming platform requirements rubric.
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

    # Extract proposed platforms and their cited sources
    extracted = await evaluator.extract(
        prompt=prompt_extract_platforms(),
        template_class=PlatformsExtraction,
        extraction_name="platforms_extraction",
    )

    # Record rubric/requirements as "ground truth context" (informational)
    evaluator.add_ground_truth({
        "requirements": {
            "ram": ">= 16 GB",
            "storage_type": "NVMe SSD",
            "storage_capacity": ">= 512 GB",
            "gta_vi_support": "Officially confirmed for the platform",
            "game_library": "Exclusive titles OR classic library >= 150 games"
        },
        "note": "Every factual check should be supported by cited URLs from official sources or reliable tech publications."
    }, gt_type="task_requirements")

    # Build a container node for evaluation (matches rubric theme)
    selection_node = evaluator.add_parallel(
        id="gaming_platform_selection",
        desc="Evaluate whether the provided gaming platforms meet all specified requirements for 2026 gaming content creation",
        parent=root,
        critical=False
    )

    # Use up to 4 platforms per rubric; pad if fewer for consistent tree structure
    platforms: List[PlatformEntry] = list(extracted.platforms[:4])
    while len(platforms) < 4:
        platforms.append(PlatformEntry())

    # Verify each platform subtree
    # According to the rubric: Platform_1 and Platform_2 are the primary targets; 3 and 4 are optional.
    # We keep all platform containers as non-critical here to avoid cross-platform gating side effects.
    for i, plat in enumerate(platforms, start=1):
        await _verify_platform(
            evaluator=evaluator,
            parent_node=selection_node,
            platform=plat,
            index_zero_based=i - 1,
            platform_is_critical=(i in (1, 2))
        )

    # Return standard evaluation summary
    return evaluator.get_summary()