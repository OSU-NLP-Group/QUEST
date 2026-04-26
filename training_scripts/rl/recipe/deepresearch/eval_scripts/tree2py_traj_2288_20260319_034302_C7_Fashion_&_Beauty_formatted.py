import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "jcrew_spring_2026_western"
TASK_DESCRIPTION = """
J.Crew released their Spring 2026 catalogue featuring a themed collection that represents a departure from their traditional East Coast preppy style. Identify the official theme name of this Spring 2026 collection and describe at least three key Western-inspired design elements or item categories that characterize the collection. Include specific examples of items or design details that exemplify the Western theme. Provide a reference URL from a fashion publication that discusses this collection.
"""

EXPECTED_THEME = "Way Out West"
REQUIRED_ELEMENTS = [
    "eyelet-detailed blouses",
    "bold colored denim skirts",
    "Western-detailed/stitched wool cardigans",
]
EXPECTED_PALETTE = [
    "navy-and-white pieces",
    "moss green",
    "cool blue",
    "colorful floral accents",
]
OPTIONAL_NAMED_ITEMS = [
    "Jules Classic-Fit Eyelet Tie-Neck Shirt",
    "Grommet Belt",
    "Western Buckle Belt",
    "Engraved Heart Necklace",
    "Utility-Pocket Pencil Skirt",
    "Delphine Shoulder Bag",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CollectionExtraction(BaseModel):
    theme_name: Optional[str] = None

    # Mentions in the answer (booleans judged from the answer text)
    mentions_eyelet_blouses: Optional[bool] = None
    mentions_bold_colored_denim_skirts: Optional[bool] = None
    mentions_western_stitched_wool_cardigans: Optional[bool] = None

    mentions_blend_preppy_western: Optional[bool] = None

    # Color palette mentions in the answer
    palette_navy_white: Optional[bool] = None
    palette_moss_green: Optional[bool] = None
    palette_cool_blue: Optional[bool] = None
    palette_colorful_floral_accents: Optional[bool] = None

    # Release timing and availability mentions in the answer
    mentions_release_march_2026: Optional[bool] = None
    mentions_available_online_and_instore: Optional[bool] = None

    # Examples that the answer provides to illustrate the Western theme
    example_items_or_details: List[str] = Field(default_factory=list)

    # All reference URLs the answer cites (as written in the answer text)
    reference_urls: List[str] = Field(default_factory=list)

    # Optional: which of the specified named items are mentioned in the answer
    named_examples_mentioned: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_collection_info() -> str:
    return """
    You will extract structured information from the answer about J.Crew's Spring 2026 catalogue collection.

    Return a JSON object with the following fields:

    1) theme_name: The official theme name as explicitly written in the answer (string or null).
    2) reference_urls: An array of all URLs that the answer provides as references or sources. Extract actual URLs only (plain or markdown). If none, return [].
    3) example_items_or_details: Up to 5 example item names or design details the answer cites to illustrate the Western theme. If none, [].

    The following are boolean flags indicating whether the answer text explicitly mentions these items/concepts (allow reasonable synonyms/variants):
    4) mentions_eyelet_blouses: true/false — counts if wording like "eyelet blouse", "eyelet-detailed blouse", or "eyelet-adorned top" appears.
    5) mentions_bold_colored_denim_skirts: true/false — counts for phrases like "bold colored denim skirt(s)", "bright denim skirt(s)", or equivalent strong/saturated color descriptors.
    6) mentions_western_stitched_wool_cardigans: true/false — counts for "Western-stitched cardigans", "Western-detailed wool cardigans", or close synonyms.
    7) mentions_blend_preppy_western: true/false — counts if the answer notes the blend of Western elements with J.Crew's East Coast preppy aesthetic.

    Color palette booleans — mark true only if the answer mentions each one (allow close synonyms like "navy and white", "sage/olive for moss", "icy/sky for cool blue"):
    8) palette_navy_white: true/false
    9) palette_moss_green: true/false
    10) palette_cool_blue: true/false
    11) palette_colorful_floral_accents: true/false

    Release timing and availability booleans:
    12) mentions_release_march_2026: true/false — the answer explicitly states the catalogue released in March 2026.
    13) mentions_available_online_and_instore: true/false — the answer explicitly states availability both online and in-store.

    Optional helper list:
    14) named_examples_mentioned: an array listing any of the following that the answer explicitly mentions (match case-insensitively, include the original casing from the answer):
        - Jules Classic-Fit Eyelet Tie-Neck Shirt
        - Grommet Belt
        - Western Buckle Belt
        - Engraved Heart Necklace
        - Utility-Pocket Pencil Skirt
        - Delphine Shoulder Bag

    IMPORTANT:
    - Do not invent content; extract only what appears in the answer.
    - For URLs, extract only actual URLs present in the answer text (plain or markdown).
    - For booleans, judge strictly based on the answer text (with reasonable synonym tolerance).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_named_item_in_answer(extraction: CollectionExtraction) -> bool:
    items_in_answer = [s.lower() for s in (extraction.example_items_or_details or [])]
    for opt in OPTIONAL_NAMED_ITEMS:
        needle = opt.lower()
        if any(needle in s for s in items_in_answer):
            return True
    # Also check named_examples_mentioned if extractor provided them
    for opt in (extraction.named_examples_mentioned or []):
        if opt:
            return True
    return False


def _choose_example_for_verification(extraction: CollectionExtraction) -> Optional[str]:
    if extraction.example_items_or_details:
        return extraction.example_items_or_details[0]
    return None


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extraction: CollectionExtraction) -> None:
    root = evaluator.root

    # ---------------- Mandatory pipeline (sequential; all children critical) ---------------- #
    mandatory = evaluator.add_sequential(
        id="mandatory",
        desc="All mandatory constraints verified using a fashion-publication source",
        parent=root,
        critical=True,
    )

    # Step 1: Check that at least one reference URL is provided in the answer
    refs_present = evaluator.add_custom_node(
        result=bool(extraction.reference_urls),
        id="reference_urls_provided",
        desc="At least one reference URL is provided in the answer",
        parent=mandatory,
        critical=True,
    )

    # Step 2: Verify the provided URL(s) are fashion-publication coverage discussing J.Crew Spring 2026 collection
    ref_is_fashion_pub = evaluator.add_leaf(
        id="reference_url_is_fashion_publication",
        desc="Provided reference URL is from a fashion/media publication that discusses J.Crew's Spring 2026 collection",
        parent=mandatory,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This webpage is an editorial article or report from a fashion or style publication (e.g., WWD, Vogue, Elle, Harper's Bazaar, GQ, InStyle, Nylon, etc.) "
            "that discusses J.Crew's Spring 2026 catalogue/collection."
        ),
        node=ref_is_fashion_pub,
        sources=extraction.reference_urls,
        additional_instruction=(
            "Pass only if the page clearly reads like an editorial fashion/media article, not a shopping product page or unrelated page, "
            "and it specifically discusses J.Crew's Spring 2026 catalogue/collection."
        ),
    )

    # Step 3: Core content claims grounded by the same source(s)
    claims = evaluator.add_parallel(
        id="content_claims",
        desc="Core content claims grounded by the fashion-publication source(s)",
        parent=mandatory,
        critical=True,
    )

    # 3.1 Theme name identified in the answer (existence) + supported by source(s)
    theme_exist = evaluator.add_custom_node(
        result=bool(extraction.theme_name and extraction.theme_name.strip()),
        id="theme_name_provided",
        desc="Answer explicitly identifies a collection theme name",
        parent=claims,
        critical=True,
    )
    theme_supported = evaluator.add_leaf(
        id="theme_name_supported",
        desc=f"Collection's official theme name is '{EXPECTED_THEME}' (supported by sources)",
        parent=claims,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official theme name of J.Crew's Spring 2026 collection is '{EXPECTED_THEME}'.",
        node=theme_supported,
        sources=extraction.reference_urls,
        additional_instruction="Allow small punctuation/casing variations; the theme name itself must be clearly indicated.",
    )

    # 3.2 Eyelet-detailed blouses
    eyelet_exist = evaluator.add_custom_node(
        result=bool(extraction.mentions_eyelet_blouses),
        id="eyelet_blouses_mentioned_in_answer",
        desc="Answer mentions eyelet-detailed blouses",
        parent=claims,
        critical=True,
    )
    eyelet_supported = evaluator.add_leaf(
        id="eyelet_blouses_supported_by_source",
        desc="Source(s) confirm the collection features eyelet-detailed blouses",
        parent=claims,
        critical=True,
    )
    await evaluator.verify(
        claim="The J.Crew Spring 2026 collection features eyelet-detailed blouses (or eyelet-adorned tops).",
        node=eyelet_supported,
        sources=extraction.reference_urls,
        additional_instruction="Accept synonyms like 'eyelet blouse', 'eyelet-embroidered top', or similar phrasing indicating eyelet detail.",
    )

    # 3.3 Bold colored denim skirts
    denim_exist = evaluator.add_custom_node(
        result=bool(extraction.mentions_bold_colored_denim_skirts),
        id="bold_colored_denim_skirts_mentioned_in_answer",
        desc="Answer mentions bold colored denim skirts",
        parent=claims,
        critical=True,
    )
    denim_supported = evaluator.add_leaf(
        id="bold_colored_denim_skirts_supported_by_source",
        desc="Source(s) confirm the collection includes bold colored denim skirts",
        parent=claims,
        critical=True,
    )
    await evaluator.verify(
        claim="The J.Crew Spring 2026 collection includes bold colored denim skirts.",
        node=denim_supported,
        sources=extraction.reference_urls,
        additional_instruction=(
            "Allow reasonable variants such as 'brightly colored denim skirts', 'denim skirts in saturated hues', or similar language."
        ),
    )

    # 3.4 Western-detailed/stitched wool cardigans
    cardigans_exist = evaluator.add_custom_node(
        result=bool(extraction.mentions_western_stitched_wool_cardigans),
        id="western_stitched_wool_cardigans_mentioned_in_answer",
        desc="Answer mentions Western-detailed/stitched wool cardigans",
        parent=claims,
        critical=True,
    )
    cardigans_supported = evaluator.add_leaf(
        id="western_stitched_wool_cardigans_supported_by_source",
        desc="Source(s) confirm Western-detailed/stitched wool cardigans are featured",
        parent=claims,
        critical=True,
    )
    await evaluator.verify(
        claim="The J.Crew Spring 2026 collection features Western-detailed or Western-stitched wool cardigans.",
        node=cardigans_supported,
        sources=extraction.reference_urls,
        additional_instruction="Allow descriptive variants like 'Western-inspired stitching' on wool cardigans.",
    )

    # 3.5 Concrete example(s) illustrating the Western theme
    example_present = evaluator.add_custom_node(
        result=bool(extraction.example_items_or_details),
        id="western_example_present_in_answer",
        desc="Answer provides at least one concrete example item or design detail illustrating the Western theme",
        parent=claims,
        critical=True,
    )
    example_for_check = _choose_example_for_verification(extraction) or ""
    example_supported = evaluator.add_leaf(
        id="western_example_supported_by_source",
        desc="At least one provided example is verifiable in source(s)",
        parent=claims,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The page mentions '{example_for_check}' as part of J.Crew's Spring 2026 collection and it exemplifies a Western-inspired item or design detail."
        ),
        node=example_supported,
        sources=extraction.reference_urls,
        additional_instruction=(
            "If the example is a design detail, confirm it is explicitly associated with the collection and aligns with the Western theme."
        ),
    )

    # 3.6 Color palette is described
    # Existence: ensure all 4 palette components are mentioned in the answer
    palette_all_in_answer = bool(extraction.palette_navy_white and extraction.palette_moss_green and
                                 extraction.palette_cool_blue and extraction.palette_colorful_floral_accents)
    palette_exist = evaluator.add_custom_node(
        result=palette_all_in_answer,
        id="palette_described_in_answer",
        desc="Answer describes the palette including navy-and-white, moss green, cool blue, and colorful floral accents",
        parent=claims,
        critical=True,
    )
    palette_supported = evaluator.add_leaf(
        id="palette_supported_by_source",
        desc="Source(s) confirm the palette includes navy-and-white, moss green, cool blue, and colorful floral accents",
        parent=claims,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The J.Crew Spring 2026 collection color palette includes navy-and-white pieces, moss green, cool blue, and colorful floral accents."
        ),
        node=palette_supported,
        sources=extraction.reference_urls,
        additional_instruction=(
            "Allow near-synonyms (e.g., 'navy and white'; 'moss/olive/sage green' for moss green; 'cool blue' including 'icy' or 'sky' blue; "
            "'colorful floral accents' may be phrased as colorful floral prints/details)."
        ),
    )

    # 3.7 Blend of Western flair with East Coast preppy aesthetic
    blend_exist = evaluator.add_custom_node(
        result=bool(extraction.mentions_blend_preppy_western),
        id="blend_preppy_western_mentioned_in_answer",
        desc="Answer notes the blend of Western flair with J.Crew's East Coast preppy aesthetic",
        parent=claims,
        critical=True,
    )
    blend_supported = evaluator.add_leaf(
        id="blend_preppy_western_supported_by_source",
        desc="Source(s) confirm the collection blends Western flair with J.Crew's East Coast preppy aesthetic",
        parent=claims,
        critical=True,
    )
    await evaluator.verify(
        claim="The collection blends Western flair with J.Crew's East Coast preppy aesthetic.",
        node=blend_supported,
        sources=extraction.reference_urls,
        additional_instruction="Accept equivalent phrasings indicating a fusion of Western elements with preppy/East Coast styling.",
    )

    # 3.8 Catalogue release timing and availability (split into two critical checks)
    release_group = evaluator.add_parallel(
        id="release_and_availability",
        desc="Catalogue release month and distribution availability",
        parent=claims,
        critical=True,
    )

    released_exist = evaluator.add_custom_node(
        result=bool(extraction.mentions_release_march_2026),
        id="release_march_2026_mentioned_in_answer",
        desc="Answer states the catalogue released in March 2026",
        parent=release_group,
        critical=True,
    )
    released_supported = evaluator.add_leaf(
        id="release_march_2026_supported_by_source",
        desc="Source(s) confirm the catalogue was released in March 2026",
        parent=release_group,
        critical=True,
    )
    await evaluator.verify(
        claim="J.Crew's Spring 2026 catalogue was released in March 2026.",
        node=released_supported,
        sources=extraction.reference_urls,
        additional_instruction="Confirm timing; accept phrasing like 'released/launched/dropped in March 2026'.",
    )

    availability_exist = evaluator.add_custom_node(
        result=bool(extraction.mentions_available_online_and_instore),
        id="availability_online_instore_mentioned_in_answer",
        desc="Answer states availability both online and in-store",
        parent=release_group,
        critical=True,
    )
    availability_supported = evaluator.add_leaf(
        id="availability_online_instore_supported_by_source",
        desc="Source(s) confirm the collection is available both online and in-store",
        parent=release_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The collection is available both online and in-store.",
        node=availability_supported,
        sources=extraction.reference_urls,
        additional_instruction=(
            "Accept equivalent wording like 'online and in stores' or 'in-store and online'."
        ),
    )

    # ---------------- Optional bonus (non-critical) ---------------- #
    bonus = evaluator.add_parallel(
        id="optional_named_item_bonus",
        desc="Optional: mentions at least one specific named item (bonus)",
        parent=root,
        critical=False,
    )

    # Optional mention in the answer (existence)
    optional_item_mentioned = evaluator.add_custom_node(
        result=_has_named_item_in_answer(extraction),
        id="optional_named_item_mentioned_in_answer",
        desc="Answer mentions at least one of the specified example named items",
        parent=bonus,
        critical=True,  # Critical within bonus group; group itself is non-critical
    )

    # Optional: verify at least one of those items appears in the fashion-publication source(s)
    optional_item_supported = evaluator.add_leaf(
        id="optional_named_item_supported_by_source",
        desc="Source(s) mention at least one of the specified example named items for this collection",
        parent=bonus,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This page mentions at least one of the following items as part of J.Crew's Spring 2026 collection: "
            "Jules Classic-Fit Eyelet Tie-Neck Shirt; Grommet Belt; Western Buckle Belt; Engraved Heart Necklace; "
            "Utility-Pocket Pencil Skirt; Delphine Shoulder Bag."
        ),
        node=optional_item_supported,
        sources=extraction.reference_urls,
        additional_instruction=(
            "Pass if the page clearly mentions any one of these named items in the context of J.Crew Spring 2026."
        ),
        extra_prerequisites=[refs_present, ref_is_fashion_pub],  # Ensure source gating for the optional check
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator (root as non-critical parallel to allow optional partial credit)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_collection_info(),
        template_class=CollectionExtraction,
        extraction_name="collection_extraction",
    )

    # Add ground truth / expectation info
    evaluator.add_ground_truth(
        {
            "expected_theme": EXPECTED_THEME,
            "required_elements": REQUIRED_ELEMENTS,
            "expected_palette": EXPECTED_PALETTE,
            "release_month": "March 2026",
            "availability": "online and in-store",
            "optional_named_items": OPTIONAL_NAMED_ITEMS,
        },
        gt_type="expectations",
    )

    # Build the verification tree and run checks
    await build_and_verify_tree(evaluator, extraction)

    # Return structured summary
    return evaluator.get_summary()