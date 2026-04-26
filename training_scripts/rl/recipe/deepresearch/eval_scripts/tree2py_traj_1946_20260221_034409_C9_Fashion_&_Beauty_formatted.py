import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_fashion_collab_2023_2026"
TASK_DESCRIPTION = (
    "Identify 4 different celebrity fashion collaborations that were launched between January 2023 and February 2026 "
    "(inclusive), where each collaboration must meet ALL of the following criteria:\n\n"
    "1. The collaboration involves a named celebrity serving as a creative director, ambassador, or official collaboration partner "
    "(not merely an endorsement) with an established fashion or accessory brand\n"
    "2. The partnership has verifiable launch details including the specific month and year of launch\n"
    "3. Manufacturing details are publicly documented, including the country or region where products are manufactured and specific craftsmanship information "
    "(such as handmade, artisan-crafted, or material specifications)\n"
    "4. The initial collection specifications are available, including the exact number of styles or pieces launched\n"
    "5. The collaboration demonstrates either a documented sustainability commitment OR inclusivity/accessibility features "
    "(such as adaptive clothing, ethical production, or serving an underrepresented demographic)\n\n"
    "For each of the 4 collaborations, provide:\n"
    "- Celebrity name and their specific role\n"
    "- Brand name\n"
    "- Launch date (month and year minimum)\n"
    "- Product category\n"
    "- Manufacturing location and craftsmanship details\n"
    "- Number of pieces/styles in initial collection\n"
    "- Type and specific description of sustainability or inclusivity features\n"
    "- Reference URLs supporting each major claim"
)

TIMEFRAME_START_TEXT = "January 2023"
TIMEFRAME_END_TEXT = "February 2026"


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _nonempty(text: Optional[str]) -> bool:
    return isinstance(text, str) and text.strip() != ""


def _normalize(text: Optional[str]) -> str:
    return (text or "").strip()


def _lower(text: Optional[str]) -> str:
    return _normalize(text).lower()


def _prep_urls(urls: Optional[List[str]]) -> List[str]:
    return [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip() != ""]


def _merge_urls(*lists: List[str]) -> List[str]:
    seen = set()
    res = []
    for lst in lists:
        for u in lst:
            if u not in seen:
                res.append(u)
                seen.add(u)
    return res


def _is_apparel(category: Optional[str]) -> bool:
    cat = _lower(category)
    if not cat:
        return False
    apparel_keywords = [
        "apparel", "clothing", "ready-to-wear", "rtw", "garment", "garments",
        "dress", "dresses", "shirt", "shirts", "pant", "pants", "jeans", "denim",
        "outerwear", "jacket", "jackets", "coat", "coats", "skirt", "skirts",
        "sweater", "sweaters", "hoodie", "hoodies", "t-shirt", "tshirts", "tee", "tees",
        "knitwear", "tops", "blouse", "blouses", "suit", "suits"
    ]
    return any(k in cat for k in apparel_keywords)


def _has_sustainability(text: Optional[str]) -> bool:
    t = _lower(text)
    if not t:
        return False
    sus_kw = [
        "sustain", "sustainable", "recycled", "recyclable", "organic", "eco",
        "environmental", "ethical", "fair trade", "fair-trade", "carbon", "circular",
        "responsibly", "traceable", "upcycled", "vegan", "low-impact"
    ]
    return any(k in t for k in sus_kw)


def _has_inclusivity(text: Optional[str]) -> bool:
    t = _lower(text)
    if not t:
        return False
    inc_kw = [
        "inclusive", "inclusivity", "diverse", "accessibility", "accessible",
        "adaptive", "plus-size", "petite", "modest", "gender-neutral",
        "wheelchair", "disabled", "hearing", "visual", "braille", "asl", "for all"
    ]
    return any(k in t for k in inc_kw)


def _feature_type_ok(feature_type: Optional[str], feature_specifics: Optional[str]) -> bool:
    # Accept either explicit feature_type indicating sustainability/inclusivity,
    # or infer from the feature_specifics string if feature_type is missing/ambiguous.
    return _has_sustainability(feature_type) or _has_inclusivity(feature_type) or \
        _has_sustainability(feature_specifics) or _has_inclusivity(feature_specifics)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CollaborationItem(BaseModel):
    celebrity_name: Optional[str] = None
    celebrity_role: Optional[str] = None
    brand_name: Optional[str] = None

    launch_month_year: Optional[str] = None
    launch_event_location: Optional[str] = None

    product_category: Optional[str] = None
    manufacturing_location: Optional[str] = None
    craftsmanship_details: Optional[str] = None
    material_specifications: Optional[str] = None
    design_lead: Optional[str] = None

    collection_size_exact: Optional[str] = None

    feature_type: Optional[str] = None  # sustainability, inclusivity, or both (free text allowed)
    feature_specifics: Optional[str] = None

    # URL groups supporting each atomic claim
    partnership_and_role_urls: List[str] = Field(default_factory=list)
    launch_month_year_urls: List[str] = Field(default_factory=list)
    launch_location_urls: List[str] = Field(default_factory=list)
    manufacturing_location_urls: List[str] = Field(default_factory=list)
    craftsmanship_or_material_urls: List[str] = Field(default_factory=list)
    collection_size_urls: List[str] = Field(default_factory=list)
    feature_urls: List[str] = Field(default_factory=list)


class CollaborationExtraction(BaseModel):
    collaborations: List[CollaborationItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_collaborations() -> str:
    return """
    Extract up to all celebrity–brand fashion collaborations mentioned in the answer.
    Each collaboration must include the following fields, exactly as they appear in the answer:

    Required fields per collaboration:
    - celebrity_name: the celebrity’s full name
    - celebrity_role: their specific role (e.g., creative director, brand ambassador, official collaboration partner, capsule collaborator, co-designer). Avoid vague terms like “endorsement” unless the answer directly states they are officially an ambassador or equivalent.
    - brand_name: the established fashion/accessory brand (not a celebrity-owned brand)
    - launch_month_year: the launch timing at minimum month + year (e.g., “March 2024”)
    - launch_event_location: location where launch happened or was revealed (city/country/event as provided)
    - product_category: the product type (e.g., apparel, footwear, handbags, jewelry, eyewear, accessories)
    - manufacturing_location: country/region where products are made (as stated)
    - craftsmanship_details: publicly documented craftsmanship/manufacturing method details (e.g., handmade, artisan-crafted, specific build or techniques)
    - material_specifications: if apparel, list material specs (e.g., 100% cotton, leather upper, recycled polyester blend); otherwise null
    - design_lead: if apparel, the design lead or designer name; otherwise null
    - collection_size_exact: the exact number of styles/pieces in the initial collection (e.g., “12 pieces” or “12”)
    - feature_type: whether it demonstrates sustainability and/or inclusivity/accessibility features (free text allowed, but should indicate at least one)
    - feature_specifics: a specific description of the sustainability/inclusivity features (e.g., “recycled polyester and fair-trade factory”; or “adaptive closures designed for wheelchair users”)

    Source URL groups per collaboration (extract as arrays of URLs, if present):
    - partnership_and_role_urls: URLs supporting the celebrity–brand partnership and the celebrity’s official role
    - launch_month_year_urls: URLs supporting the launch timing (month/year)
    - launch_location_urls: URLs supporting the launch event/location
    - manufacturing_location_urls: URLs supporting the country/region of manufacture
    - craftsmanship_or_material_urls: URLs supporting craftsmanship/manufacturing methods and/or material specifications
    - collection_size_urls: URLs supporting the initial collection size (exact number of pieces/styles)
    - feature_urls: URLs supporting the sustainability/inclusivity/accessibility claims

    Rules:
    - Only extract information explicitly present in the answer.
    - For URLs, extract the actual URL strings as they appear (including from markdown links).
    - If a field is not provided in the answer, set it to null (for strings) or an empty list (for URL arrays).
    - Do NOT invent or infer URLs or details not present in the answer.
    - Preserve the author’s wording for names/titles when possible.
    - For collection_size_exact, return the exact string (e.g., “12 pieces”, “12”, or “a 12-style drop”).
    - For launch_month_year, keep a concise month-year string (e.g., “Jan 2023”, “January 2023”, “Mar 2025”).

    Return a JSON object with:
    {
      "collaborations": [ { ...fields above... }, ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helpers for building verification nodes                                     #
# --------------------------------------------------------------------------- #
async def _add_support_check(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    critical: bool = True,
    add_ins: Optional[str] = None,
) -> None:
    """
    Add a support check leaf node. If no URLs are provided, add a failing custom node.
    Otherwise, add a leaf and verify against the provided URLs (multi-URL verification).
    """
    clean_urls = _prep_urls(urls)
    if not clean_urls:
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=f"{desc} (FAILED: no supporting URLs provided in the answer)",
            parent=parent_node,
            critical=critical
        )
        return

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=clean_urls,
        additional_instruction=add_ins or "None"
    )


def _add_presence_check(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    present: bool,
    critical: bool = True
) -> None:
    evaluator.add_custom_node(
        result=present,
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical
    )


# --------------------------------------------------------------------------- #
# Per-collaboration verification                                              #
# --------------------------------------------------------------------------- #
async def verify_collaboration(
    evaluator: Evaluator,
    parent_node,
    collab: CollaborationItem,
    index: int
) -> None:
    cidx = index + 1
    collab_node = evaluator.add_parallel(
        id=f"collaboration_{cidx}",
        desc=f"Collaboration {cidx} meets all constraints and required fields are provided",
        parent=parent_node,
        critical=False  # Allow partial credit across different collaborations
    )

    # ---------------- Identity & Relationship (critical) ------------------ #
    ident = evaluator.add_parallel(
        id=f"collab_{cidx}_identity_and_relationship",
        desc="Celebrity–brand collaboration identity and relationship validity",
        parent=collab_node,
        critical=True
    )

    _add_presence_check(
        evaluator, ident, f"collab_{cidx}_celebrity_name",
        "Celebrity name is provided",
        present=_nonempty(collab.celebrity_name),
        critical=True
    )
    _add_presence_check(
        evaluator, ident, f"collab_{cidx}_brand_name",
        "Brand name is provided",
        present=_nonempty(collab.brand_name),
        critical=True
    )

    # Celebrity role validity (verify with URLs; must be official collaboration role)
    role_claim = (
        f"{_normalize(collab.celebrity_name)} serves in an official collaboration role "
        f"('{_normalize(collab.celebrity_role)}') with brand '{_normalize(collab.brand_name)}' "
        f"(e.g., creative director, ambassador, official partner, collaborator; not merely an endorsement)."
    )
    await _add_support_check(
        evaluator, ident, f"collab_{cidx}_celebrity_role_valid",
        "Celebrity role is specified and is an official collaboration role (not merely an endorsement)",
        role_claim,
        collab.partnership_and_role_urls,
        critical=True,
        add_ins="Confirm that the role wording indicates an official collaboration (creative director/ambassador/official partner/"
                "capsule collaborator/co-designer). Generic 'endorsement' or 'face of' without an official role is insufficient."
    )

    # Brand is established and not celebrity-owned (verify via partnership/brand sources)
    brand_claim = (
        f"The brand '{_normalize(collab.brand_name)}' is an established fashion/accessory brand and is not owned or founded by "
        f"'{_normalize(collab.celebrity_name)}'."
    )
    await _add_support_check(
        evaluator, ident, f"collab_{cidx}_brand_is_established_and_not_celebrity_owned",
        "Brand is an established fashion/accessory brand and is not a celebrity-owned brand",
        brand_claim,
        collab.partnership_and_role_urls,
        critical=True,
        add_ins="Use the provided brand/press sources to determine whether this is a recognized standalone brand and not the celebrity's own line."
    )

    # ---------------- Launch details (critical) --------------------------- #
    launch = evaluator.add_parallel(
        id=f"collab_{cidx}_launch_details",
        desc="Launch details are provided and are within the required timeframe",
        parent=collab_node,
        critical=True
    )

    _add_presence_check(
        evaluator, launch, f"collab_{cidx}_launch_month_year",
        "Launch month and year (at minimum) are provided",
        present=_nonempty(collab.launch_month_year),
        critical=True
    )

    # Verify within timeframe using launch timing sources
    timeframe_claim = (
        f"The collaboration launch occurred in '{_normalize(collab.launch_month_year)}', which falls between {TIMEFRAME_START_TEXT} "
        f"and {TIMEFRAME_END_TEXT} (inclusive)."
    )
    await _add_support_check(
        evaluator, launch, f"collab_{cidx}_launch_within_timeframe",
        "Launch date is between January 2023 and February 2026 (inclusive)",
        timeframe_claim,
        collab.launch_month_year_urls,
        critical=True,
        add_ins="Check the date/month on the provided sources and confirm it lies within the inclusive window Jan 2023 to Feb 2026."
    )

    _add_presence_check(
        evaluator, launch, f"collab_{cidx}_launch_event_location",
        "Launch event/launch location is provided (per constraints)",
        present=_nonempty(collab.launch_event_location),
        critical=True
    )

    # ---------------- Product & Manufacturing (critical) ------------------ #
    pm = evaluator.add_parallel(
        id=f"collab_{cidx}_product_and_manufacturing",
        desc="Product category and publicly documented manufacturing/craftsmanship details",
        parent=collab_node,
        critical=True
    )

    _add_presence_check(
        evaluator, pm, f"collab_{cidx}_product_category",
        "Product category is specified",
        present=_nonempty(collab.product_category),
        critical=True
    )
    _add_presence_check(
        evaluator, pm, f"collab_{cidx}_manufacturing_location",
        "Country/region of manufacture is specified",
        present=_nonempty(collab.manufacturing_location),
        critical=True
    )
    _add_presence_check(
        evaluator, pm, f"collab_{cidx}_craftsmanship_details",
        "Craftsmanship/manufacturing method details are provided (e.g., handmade/artisan-crafted/material build details)",
        present=_nonempty(collab.craftsmanship_details),
        critical=True
    )

    # Conditional requirements for apparel
    is_apparel = _is_apparel(collab.product_category)

    _add_presence_check(
        evaluator, pm, f"collab_{cidx}_material_specifications_if_apparel",
        "If the product category is apparel: material specifications are provided (per constraints)",
        present=(not is_apparel) or _nonempty(collab.material_specifications),
        critical=True
    )

    _add_presence_check(
        evaluator, pm, f"collab_{cidx}_design_lead_if_apparel",
        "If the product category is apparel: the design lead (designer name) is identified (per constraints)",
        present=(not is_apparel) or _nonempty(collab.design_lead),
        critical=True
    )

    # ---------------- Initial collection specs (critical) ----------------- #
    init_spec = evaluator.add_parallel(
        id=f"collab_{cidx}_initial_collection_specifications",
        desc="Initial collection specs include exact collection size",
        parent=collab_node,
        critical=True
    )

    _add_presence_check(
        evaluator, init_spec, f"collab_{cidx}_collection_size_exact",
        "Exact number of styles/pieces in the initial collection is provided",
        present=_nonempty(collab.collection_size_exact),
        critical=True
    )

    # ---------------- Sustainability or Inclusivity (critical) ------------ #
    features = evaluator.add_parallel(
        id=f"collab_{cidx}_sustainability_or_inclusivity",
        desc="Sustainability commitment and/or inclusivity/accessibility features are documented",
        parent=collab_node,
        critical=True
    )

    feature_type_ok = _feature_type_ok(collab.feature_type, collab.feature_specifics)
    _add_presence_check(
        evaluator, features, f"collab_{cidx}_feature_type",
        "States whether the collaboration demonstrates sustainability and/or inclusivity/accessibility (at least one)",
        present=feature_type_ok,
        critical=True
    )
    _add_presence_check(
        evaluator, features, f"collab_{cidx}_feature_specifics",
        "Provides a specific description of the sustainability commitment and/or inclusivity/accessibility features",
        present=_nonempty(collab.feature_specifics),
        critical=True
    )

    # Conditional documentation checks using feature URLs
    sus_claim = (
        f"If sustainability is claimed, the sources document a concrete environmental and/or ethical commitment for this collaboration: "
        f"'{_normalize(collab.feature_specifics)}'. If sustainability is not claimed, consider this requirement satisfied."
    )
    await _add_support_check(
        evaluator, features, f"collab_{cidx}_sustainability_commitment_documented_if_sustainability_claimed",
        "If sustainability is claimed: includes a documented environmental and/or ethical commitment (per constraints)",
        sus_claim,
        collab.feature_urls,
        critical=True,
        add_ins="Pass if either (a) sustainability is not claimed, or (b) when claimed, the sources clearly document the sustainability commitment."
    )

    inc_claim = (
        f"If inclusivity/accessibility is claimed, the sources specify the target demographic and/or accessibility/adaptive features "
        f"for this collaboration: '{_normalize(collab.feature_specifics)}'. If inclusivity is not claimed, consider this requirement satisfied."
    )
    await _add_support_check(
        evaluator, features, f"collab_{cidx}_inclusivity_target_or_access_features_if_inclusivity_claimed",
        "If inclusivity/accessibility is claimed: specifies target demographic and/or accessibility/adaptive features (per constraints)",
        inc_claim,
        collab.feature_urls,
        critical=True,
        add_ins="Pass if either (a) inclusivity is not claimed, or (b) when claimed, the sources clearly describe target group and/or adaptive features."
    )

    # ---------------- Sources (critical; atomic support checks) ----------- #
    srcs = evaluator.add_parallel(
        id=f"collab_{cidx}_sources",
        desc="Reference URLs are provided to support each major claim (atomic support checks)",
        parent=collab_node,
        critical=True
    )

    # Partnership & role
    pr_claim = (
        f"{_normalize(collab.celebrity_name)} has an official collaboration with '{_normalize(collab.brand_name)}' in the role of "
        f"'{_normalize(collab.celebrity_role)}' (e.g., creative director/ambassador/official partner/collaborator)."
    )
    await _add_support_check(
        evaluator, srcs, f"collab_{cidx}_reference_partnership_and_role",
        "At least one URL supports the celebrity–brand partnership and the celebrity’s role",
        pr_claim,
        collab.partnership_and_role_urls,
        critical=True,
        add_ins="Confirm both the collaboration relationship and the official nature of the role."
    )

    # Launch month/year
    lmy_claim = (
        f"The collaboration launched in '{_normalize(collab.launch_month_year)}'."
    )
    await _add_support_check(
        evaluator, srcs, f"collab_{cidx}_reference_launch_month_year",
        "At least one URL supports the launch timing (month/year)",
        lmy_claim,
        collab.launch_month_year_urls,
        critical=True,
        add_ins="Verify the source states the launch month and year."
    )

    # Launch location
    lloc_claim = (
        f"The launch event or reveal took place at/in '{_normalize(collab.launch_event_location)}'."
    )
    await _add_support_check(
        evaluator, srcs, f"collab_{cidx}_reference_launch_location",
        "At least one URL supports the launch event/launch location",
        lloc_claim,
        collab.launch_location_urls,
        critical=True,
        add_ins="Confirm the location or event venue/city/country as provided."
    )

    # Manufacturing location
    mf_claim = (
        f"The products for this collaboration are manufactured in '{_normalize(collab.manufacturing_location)}'."
    )
    await _add_support_check(
        evaluator, srcs, f"collab_{cidx}_reference_manufacturing_location",
        "At least one URL supports the manufacturing country/region",
        mf_claim,
        collab.manufacturing_location_urls,
        critical=True,
        add_ins="Look for explicit mentions of country/region of manufacture."
    )

    # Craftsmanship/materials
    cm_parts = []
    if _nonempty(collab.craftsmanship_details):
        cm_parts.append(f"craftsmanship/manufacturing details: '{_normalize(collab.craftsmanship_details)}'")
    if _nonempty(collab.material_specifications):
        cm_parts.append(f"material specifications: '{_normalize(collab.material_specifications)}'")
    cm_text = "; ".join(cm_parts) if cm_parts else "craftsmanship/materials details"
    cm_claim = f"The sources document {cm_text} for this collaboration."
    await _add_support_check(
        evaluator, srcs, f"collab_{cidx}_reference_craftsmanship_or_materials",
        "At least one URL supports craftsmanship/manufacturing method details and/or material specifications (as applicable)",
        cm_claim,
        collab.craftsmanship_or_material_urls,
        critical=True,
        add_ins="Accept either explicit craft/method descriptions (e.g., handmade, artisan) and/or material specs (e.g., leather, recycled polyester)."
    )

    # Collection size
    cs_claim = (
        f"The initial collection includes exactly '{_normalize(collab.collection_size_exact)}' pieces/styles."
    )
    await _add_support_check(
        evaluator, srcs, f"collab_{cidx}_reference_collection_size",
        "At least one URL supports the initial collection size (number of styles/pieces)",
        cs_claim,
        collab.collection_size_urls,
        critical=True,
        add_ins="Verify the exact number of styles/pieces for the first drop/initial collection."
    )

    # Sustainability/inclusivity feature support
    feat_claim = (
        f"The collaboration demonstrates sustainability and/or inclusivity/accessibility features as described: "
        f"'{_normalize(collab.feature_specifics)}'."
    )
    await _add_support_check(
        evaluator, srcs, f"collab_{cidx}_reference_feature",
        "At least one URL supports the sustainability/inclusivity/accessibility claims",
        feat_claim,
        collab.feature_urls,
        critical=True,
        add_ins="Look for explicit mentions of sustainable materials/processes or inclusive/adaptive design features and target users."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
) -> Dict[str, Any]:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel aggregator
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

    # ----------------------- Extraction ---------------------------------- #
    extraction: CollaborationExtraction = await evaluator.extract(
        prompt=prompt_extract_collaborations(),
        template_class=CollaborationExtraction,
        extraction_name="collaborations_extraction"
    )

    all_items = extraction.collaborations or []
    # Filter only the first 4 collaborations, per final reminder
    selected: List[CollaborationItem] = list(all_items[:4])

    # If fewer than 4 provided, pad with empty placeholders to allow per-item checks to fail appropriately
    while len(selected) < 4:
        selected.append(CollaborationItem())

    # ------------------ Set-level requirements (critical) ---------------- #
    set_level = evaluator.add_parallel(
        id="set_level_requirements",
        desc="Set-level requirements about the overall list of collaborations",
        parent=root,
        critical=True
    )

    # Exactly 4 collaborations provided: consider "provided" as items with both celebrity and brand present
    real_count = sum(1 for it in selected if _nonempty(it.celebrity_name) and _nonempty(it.brand_name))
    evaluator.add_custom_node(
        result=(real_count == 4),
        id="exactly_four_collaborations_provided",
        desc="Exactly 4 collaborations are provided",
        parent=set_level,
        critical=True
    )

    # Distinct collaborations (no duplicate celebrity–brand pairs among provided items)
    seen_pairs = set()
    duplicates_found = False
    for it in selected:
        if _nonempty(it.celebrity_name) and _nonempty(it.brand_name):
            key = (_lower(it.celebrity_name), _lower(it.brand_name))
            if key in seen_pairs:
                duplicates_found = True
                break
            seen_pairs.add(key)
    evaluator.add_custom_node(
        result=not duplicates_found,
        id="collaborations_are_distinct",
        desc="All 4 collaborations are different (no duplicate celebrity–brand collaboration repeated)",
        parent=set_level,
        critical=True
    )

    # ------------------ Per-collaboration verification ------------------- #
    for idx in range(4):
        await verify_collaboration(evaluator, root, selected[idx], idx)

    return evaluator.get_summary()