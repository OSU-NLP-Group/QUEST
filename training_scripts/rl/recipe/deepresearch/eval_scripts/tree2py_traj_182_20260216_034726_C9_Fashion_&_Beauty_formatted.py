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
TASK_ID = "celeb_fashion_beauty_2025_2026"
TASK_DESCRIPTION = (
    "Identify 4 celebrity fashion and beauty professionals based on their work in 2025-2026, "
    "ensuring all specified criteria and details are provided with supporting URLs."
)


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class Professional1Info(BaseModel):
    # Beauty Brand Ambassador
    name: Optional[str] = None
    brand: Optional[str] = None  # e.g., Dior Beauty, YSL Beauté, etc. (luxury fashion house beauty division)
    announcement_date: Optional[str] = None  # specific date in August 2025
    fragrance_name: Optional[str] = None
    campaign_director: Optional[str] = None

    ambassador_urls: List[str] = Field(default_factory=list)           # for role & announcement
    fragrance_campaign_urls: List[str] = Field(default_factory=list)   # fronting men's fragrance
    fragrance_name_urls: List[str] = Field(default_factory=list)       # fragrance name
    director_urls: List[str] = Field(default_factory=list)             # campaign director


class Professional2Info(BaseModel):
    # Fashion Photographer
    photographer_name: Optional[str] = None
    cover_subject_name: Optional[str] = None
    tv_series_name: Optional[str] = None  # the “major” TV series
    stylist_name: Optional[str] = None

    photographer_issue_urls: List[str] = Field(default_factory=list)  # photographer shot British Vogue April 2025 cover
    subject_urls: List[str] = Field(default_factory=list)             # subject is the cover subject of April 2025 British Vogue
    tv_series_urls: List[str] = Field(default_factory=list)           # subject appeared in the named TV series
    tv_series_major_urls: List[str] = Field(default_factory=list)     # evidence that series is “major”
    stylist_urls: List[str] = Field(default_factory=list)             # stylist credited for the cover shoot


class Professional3Info(BaseModel):
    # Celebrity Stylist
    stylist_name: Optional[str] = None
    client_name: Optional[str] = None
    agency_name: Optional[str] = None

    styling_event_urls: List[str] = Field(default_factory=list)       # styled client for Golden Globes 2026 red carpet (Jan 11, 2026)
    client_actress_urls: List[str] = Field(default_factory=list)      # client is an actress
    gown_collection_urls: List[str] = Field(default_factory=list)     # Armani Privé Fall 2021 couture
    gown_description_urls: List[str] = Field(default_factory=list)    # powder pink pleated crinoline with embroidered crystal drops
    agency_urls: List[str] = Field(default_factory=list)              # agency representation for stylist


class Professional4Info(BaseModel):
    # Beauty Creative Director
    professional_name: Optional[str] = None
    beauty_line_name: Optional[str] = None

    appointment_date: Optional[str] = None      # March 5, 2025
    preorders_date: Optional[str] = None        # August 25, 2025
    launch_date: Optional[str] = None           # August 29, 2025

    lipstick_shades: Optional[str] = None       # "55"
    tinted_balms: Optional[str] = None          # "10"
    eyeshadow_palettes: Optional[str] = None    # "8"

    appointment_urls: List[str] = Field(default_factory=list)
    beauty_line_name_urls: List[str] = Field(default_factory=list)
    preorders_urls: List[str] = Field(default_factory=list)
    launch_urls: List[str] = Field(default_factory=list)
    product_counts_urls: List[str] = Field(default_factory=list)


class ProfessionalsExtraction(BaseModel):
    professional1: Optional[Professional1Info] = None
    professional2: Optional[Professional2Info] = None
    professional3: Optional[Professional3Info] = None
    professional4: Optional[Professional4Info] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_professionals() -> str:
    return """
Extract structured information from the answer for FOUR distinct professionals that match the categories and constraints below. You must only extract facts and URLs explicitly present in the answer.

RULES:
- Do not infer or hallucinate any information or URLs that are not explicitly mentioned.
- If a field is not present in the answer, set it to null (for strings) or an empty array (for URL lists).
- For every factual detail that requires evidence, extract the supporting URLs explicitly cited in the answer.
- URLs may appear as plain URLs or in markdown links. Extract the final URLs.
- Do NOT merge the four professionals; they should be distinct and mapped to the correct categories.

CATEGORIES & REQUIRED FIELDS:

1) Professional 1 — Beauty Brand Ambassador
- Fields:
  - name (string)
  - brand (string; the beauty division of a luxury fashion house, if specified)
  - announcement_date (string; specific date in August 2025)
  - fragrance_name (string)
  - campaign_director (string)
  - ambassador_urls (array of URLs) — supports the ambassador announcement & brand beauty division context
  - fragrance_campaign_urls (array of URLs) — supports that they are fronting the SAME brand’s men’s fragrance campaign
  - fragrance_name_urls (array of URLs) — supports the fragrance name
  - director_urls (array of URLs) — supports the campaign director

2) Professional 2 — Fashion Photographer
- Fields:
  - photographer_name (string)
  - cover_subject_name (string)
  - tv_series_name (string) — a major TV series the cover subject has appeared in
  - stylist_name (string) — stylist credited for the cover shoot
  - photographer_issue_urls (array of URLs) — supports that the photographer photographed a British Vogue cover for the April 2025 issue
  - subject_urls (array of URLs) — supports that the named subject is on that British Vogue April 2025 cover
  - tv_series_urls (array of URLs) — supports that the subject appeared in the named TV series
  - tv_series_major_urls (array of URLs) — supports the claim that the named TV series is “major” (e.g., reputable sources noting it as major/hit/flagship, awards, mainstream coverage)
  - stylist_urls (array of URLs) — supports the stylist credit for the cover shoot

3) Professional 3 — Celebrity Stylist
- Fields:
  - stylist_name (string)
  - client_name (string)
  - agency_name (string)
  - styling_event_urls (array of URLs) — supports that the stylist styled a client for the Golden Globes 2026 red carpet (January 11, 2026)
  - client_actress_urls (array of URLs) — supports that the client is an actress
  - gown_collection_urls (array of URLs) — supports that the gown is from Armani Privé Fall 2021 couture
  - gown_description_urls (array of URLs) — supports that the gown is a powder pink pleated crinoline gown with embroidered crystal drops
  - agency_urls (array of URLs) — supports the agency representing the stylist

4) Professional 4 — Beauty Creative Director
- Fields:
  - professional_name (string)
  - beauty_line_name (string)
  - appointment_date (string; March 5, 2025)
  - preorders_date (string; August 25, 2025)
  - launch_date (string; August 29, 2025)
  - lipstick_shades (string; exact number such as "55")
  - tinted_balms (string; exact number such as "10")
  - eyeshadow_palettes (string; exact number such as "8")
  - appointment_urls (array of URLs) — supports appointment & role
  - beauty_line_name_urls (array of URLs) — supports the beauty line name
  - preorders_urls (array of URLs) — supports the pre-orders date
  - launch_urls (array of URLs) — supports the launch date
  - product_counts_urls (array of URLs) — supports the exact product counts

Return one JSON object with keys professional1, professional2, professional3, professional4 corresponding to the above structures. If any professional is missing in the answer, return that professional object with all fields null or empty arrays accordingly.
"""


# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Filter out empty/invalid-looking entries; keep simple heuristic
    cleaned = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


async def _add_requirement_with_sources(
    evaluator: Evaluator,
    parent,
    req_id: str,
    req_desc: str,
    required_present: bool,
    urls: List[str],
    claim: str,
    add_ins: str,
) -> None:
    """
    Build a critical sequential requirement node with:
    - existence check (sources + required fields present)
    - URL-grounded verification of the claim
    """
    req_node = evaluator.add_sequential(
        id=req_id,
        desc=req_desc,
        parent=parent,
        critical=True,
    )

    existence_node = evaluator.add_custom_node(
        result=(required_present and len(urls) > 0),
        id=f"{req_id}_sources_provided",
        desc=f"Sources and required values provided for: {req_desc}",
        parent=req_node,
        critical=True,
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{req_id}_supported",
        desc=f"Claim supported by cited sources for: {req_desc}",
        parent=req_node,
        critical=True,
    )

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=urls,
        additional_instruction=add_ins,
    )


# --------------------------------------------------------------------------- #
# Verification Subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_professional_1(evaluator: Evaluator, root, p1: Optional[Professional1Info]) -> None:
    node = evaluator.add_parallel(
        id="professional_1",
        desc="Professional 1 — Beauty Brand Ambassador (August 2025 ambassador + men's fragrance campaign).",
        parent=root,
        critical=False,
    )

    name = (p1.name if p1 else None) or ""
    brand = (p1.brand if p1 else None) or ""
    ann_date = (p1.announcement_date if p1 else None) or ""
    fragrance = (p1.fragrance_name if p1 else None) or ""
    director = (p1.campaign_director if p1 else None) or ""

    ambassador_urls = _normalize_urls(p1.ambassador_urls if p1 else [])
    frag_campaign_urls = _normalize_urls(p1.fragrance_campaign_urls if p1 else [])
    frag_name_urls = _normalize_urls(p1.fragrance_name_urls if p1 else [])
    director_urls = _normalize_urls(p1.director_urls if p1 else [])

    # p1_ambassador_role_and_brand_context
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p1_ambassador_role_and_brand_context",
        "Provide the professional’s name and verify they were announced as a global beauty brand ambassador for a luxury fashion house’s beauty division (supporting URL).",
        required_present=bool(name),
        urls=ambassador_urls,
        claim=(
            f"{name} was announced as a global beauty brand ambassador for "
            + (f"{brand} (a luxury fashion house's beauty division)." if brand else "a luxury fashion house’s beauty division.")
        ),
        add_ins=(
            "Verify the announcement explicitly names a luxury fashion house's beauty division (e.g., Dior Beauty/YSL Beauté, etc.). "
            "Confirm that the role is 'global beauty brand ambassador' or an equivalent phrasing."
        ),
    )

    # p1_announcement_specific_date (must be in August 2025)
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p1_announcement_specific_date",
        "Provide the specific announcement date (must be in August 2025) with a supporting URL.",
        required_present=bool(ann_date),
        urls=ambassador_urls,
        claim=(
            f"The announcement date for {name}'s appointment as a global beauty brand ambassador was {ann_date}, "
            "and this date falls in August 2025."
        ),
        add_ins="Confirm that the stated date is a specific calendar date in August 2025 (e.g., August 10, 2025), not merely 'August 2025'.",
    )

    # p1_mens_fragrance_campaign
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p1_mens_fragrance_campaign",
        "Verify the professional is fronting the same brand’s men’s fragrance campaign (supporting URL).",
        required_present=bool(name),
        urls=frag_campaign_urls,
        claim=(
            f"{name} fronts a men's fragrance campaign for the same brand referenced in the ambassador announcement."
        ),
        add_ins="Ensure that the brand for the men's fragrance campaign matches the beauty division brand context of the ambassador role.",
    )

    # p1_fragrance_name
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p1_fragrance_name",
        "Provide the name of the men’s fragrance being campaigned (supporting URL).",
        required_present=bool(fragrance),
        urls=frag_name_urls or frag_campaign_urls,
        claim=(
            f"The men's fragrance campaign fronted by {name} is for the fragrance named '{fragrance}'."
        ),
        add_ins="Confirm the exact fragrance product name as stated on the cited brand or reputable media sources.",
    )

    # p1_campaign_director
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p1_campaign_director",
        "Provide the name of the director who directed the fragrance campaign (supporting URL).",
        required_present=bool(director),
        urls=director_urls or frag_campaign_urls,
        claim=(
            f"The fragrance campaign featuring {name} was directed by {director}."
        ),
        add_ins="Verify that the cited sources explicitly credit the named individual as the campaign film director.",
    )


async def verify_professional_2(evaluator: Evaluator, root, p2: Optional[Professional2Info]) -> None:
    node = evaluator.add_parallel(
        id="professional_2",
        desc="Professional 2 — Fashion Photographer (British Vogue April 2025 cover).",
        parent=root,
        critical=False,
    )

    photographer = (p2.photographer_name if p2 else None) or ""
    subject = (p2.cover_subject_name if p2 else None) or ""
    tv_series = (p2.tv_series_name if p2 else None) or ""
    stylist = (p2.stylist_name if p2 else None) or ""

    photographer_issue_urls = _normalize_urls(p2.photographer_issue_urls if p2 else [])
    subject_urls = _normalize_urls(p2.subject_urls if p2 else [])
    tv_series_urls = _normalize_urls(p2.tv_series_urls if p2 else [])
    tv_series_major_urls = _normalize_urls(p2.tv_series_major_urls if p2 else [])
    stylist_urls = _normalize_urls(p2.stylist_urls if p2 else [])

    # p2_photographer_and_issue
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p2_photographer_and_issue",
        "Provide the photographer’s name and verify they photographed a British Vogue cover for the April 2025 issue (supporting URL).",
        required_present=bool(photographer),
        urls=photographer_issue_urls,
        claim=f"{photographer} photographed a British Vogue cover for the April 2025 issue.",
        add_ins="Ensure the source explicitly says 'British Vogue' (UK edition) and references the April 2025 cover(s).",
    )

    # p2_cover_subject
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p2_cover_subject",
        "Provide the cover subject’s name and verify they are the subject of that British Vogue April 2025 cover (supporting URL).",
        required_present=bool(subject),
        urls=subject_urls or photographer_issue_urls,
        claim=f"{subject} is a cover subject of the British Vogue April 2025 issue.",
        add_ins="Confirm the subject is explicitly credited as a cover star for British Vogue April 2025.",
    )

    # p2_major_tv_series_name
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p2_major_tv_series_name",
        "Provide the name of a major TV series the cover subject appeared in (as claimed by the answer) (supporting URL(s)).",
        required_present=bool(tv_series),
        urls=tv_series_urls,
        claim=f"The named TV series associated with the cover subject {subject} is '{tv_series}'.",
        add_ins="Validate that the cited sources clearly name the series and associate it with the cover subject.",
    )

    # p2_tv_series_appearance_evidence
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p2_tv_series_appearance_evidence",
        "Provide supporting URL(s) that verify the cover subject appeared in the named TV series.",
        required_present=bool(tv_series) and bool(subject),
        urls=tv_series_urls,
        claim=f"{subject} appeared in the TV series '{tv_series}'.",
        add_ins="Accept reputable sources such as official network pages, IMDb, or major press confirming the subject's role/appearance.",
    )

    # p2_tv_series_major_evidence
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p2_tv_series_major_evidence",
        "Provide supporting URL(s) that reasonably support the claim that the named TV series is 'major'.",
        required_present=bool(tv_series),
        urls=tv_series_major_urls,
        claim=f"The TV series '{tv_series}' is a major series with mainstream prominence.",
        add_ins="Look for language like 'hit', 'flagship', 'acclaimed', notable awards, or broad mainstream coverage from reputable sources.",
    )

    # p2_stylist_credit
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p2_stylist_credit",
        "Provide the stylist’s name credited for styling the cover shoot (supporting URL).",
        required_present=bool(stylist),
        urls=stylist_urls or photographer_issue_urls,
        claim=f"The stylist credited for the British Vogue April 2025 cover shoot is {stylist}.",
        add_ins="Ensure the source clearly credits the stylist for the specific April 2025 British Vogue cover shoot.",
    )


async def verify_professional_3(evaluator: Evaluator, root, p3: Optional[Professional3Info]) -> None:
    node = evaluator.add_parallel(
        id="professional_3",
        desc="Professional 3 — Celebrity Stylist (Golden Globes 2026 + Armani Privé Fall 2021 gown specifics + agency).",
        parent=root,
        critical=False,
    )

    stylist_name = (p3.stylist_name if p3 else None) or ""
    client_name = (p3.client_name if p3 else None) or ""
    agency_name = (p3.agency_name if p3 else None) or ""

    styling_event_urls = _normalize_urls(p3.styling_event_urls if p3 else [])
    client_actress_urls = _normalize_urls(p3.client_actress_urls if p3 else [])
    gown_collection_urls = _normalize_urls(p3.gown_collection_urls if p3 else [])
    gown_description_urls = _normalize_urls(p3.gown_description_urls if p3 else [])
    agency_urls = _normalize_urls(p3.agency_urls if p3 else [])

    # p3_event_and_styling_relationship
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p3_event_and_styling_relationship",
        "Provide the stylist’s name and verify they styled a client for the Golden Globes 2026 red carpet (Jan 11, 2026) (supporting URL).",
        required_present=bool(stylist_name),
        urls=styling_event_urls,
        claim=(
            f"{stylist_name} styled a client for the Golden Globes 2026 red carpet held on January 11, 2026."
        ),
        add_ins="Confirm both the event (Golden Globes 2026) and the red carpet context on January 11, 2026.",
    )

    # p3_client_name
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p3_client_name",
        "Provide the client’s name (supporting URL).",
        required_present=bool(client_name),
        urls=styling_event_urls,
        claim=f"The client styled by {stylist_name} for the Golden Globes 2026 red carpet was {client_name}.",
        add_ins="Ensure the cited sources explicitly pair the stylist and the named client for the Golden Globes 2026 red carpet.",
    )

    # p3_client_actress_status
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p3_client_actress_status",
        "Verify the client is an actress (supporting URL).",
        required_present=bool(client_name),
        urls=client_actress_urls,
        claim=f"{client_name} is an actress.",
        add_ins="Accept reliable sources that identify the client's profession/occupation as an actress.",
    )

    # p3_gown_collection
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p3_gown_collection",
        "Verify the client wore a gown from the Armani Privé Fall 2021 couture collection (supporting URL).",
        required_present=bool(client_name),
        urls=gown_collection_urls or styling_event_urls,
        claim=f"{client_name} wore a gown from the Armani Privé Fall 2021 couture collection at the Golden Globes 2026.",
        add_ins="Source should explicitly mention Armani Privé Fall 2021 couture in connection with the client's look.",
    )

    # p3_gown_specific_description
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p3_gown_specific_description",
        "Verify the gown is specifically a powder pink pleated crinoline gown with embroidered crystal drops (supporting URL).",
        required_present=True,  # description might appear even if not in a dedicated field
        urls=gown_description_urls or gown_collection_urls or styling_event_urls,
        claim=(
            f"The gown worn by {client_name} is described as a powder pink pleated crinoline gown with embroidered crystal drops."
        ),
        add_ins="The cited source must include an explicit or very close description matching the stated details.",
    )

    # p3_stylist_agency_representation
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p3_stylist_agency_representation",
        "Provide the name of the agency that represents the stylist (supporting URL).",
        required_present=bool(agency_name) and bool(stylist_name),
        urls=agency_urls,
        claim=f"The stylist {stylist_name} is represented by the agency '{agency_name}'.",
        add_ins="The source should clearly indicate the agency's representation of the stylist.",
    )


async def verify_professional_4(evaluator: Evaluator, root, p4: Optional[Professional4Info]) -> None:
    node = evaluator.add_parallel(
        id="professional_4",
        desc="Professional 4 — Beauty Creative Director (appointment + launch timeline + exact product counts).",
        parent=root,
        critical=False,
    )

    prof_name = (p4.professional_name if p4 else None) or ""
    line_name = (p4.beauty_line_name if p4 else None) or ""
    appoint_date = (p4.appointment_date if p4 else None) or ""
    pre_date = (p4.preorders_date if p4 else None) or ""
    launch_date = (p4.launch_date if p4 else None) or ""
    lip = (p4.lipstick_shades if p4 else None) or ""
    balm = (p4.tinted_balms if p4 else None) or ""
    palettes = (p4.eyeshadow_palettes if p4 else None) or ""

    appointment_urls = _normalize_urls(p4.appointment_urls if p4 else [])
    line_urls = _normalize_urls(p4.beauty_line_name_urls if p4 else []) or appointment_urls
    pre_urls = _normalize_urls(p4.preorders_urls if p4 else [])
    launch_urls = _normalize_urls(p4.launch_urls if p4 else [])
    product_urls = _normalize_urls(p4.product_counts_urls if p4 else []) or launch_urls

    # p4_appointment_role
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p4_appointment_role",
        "Provide the professional’s name and verify they were appointed as Creative Director for a luxury fashion house’s first beauty line (supporting URL).",
        required_present=bool(prof_name),
        urls=appointment_urls,
        claim=(
            f"{prof_name} was appointed as Creative Director for a luxury fashion house’s first beauty line"
            + (f", named {line_name}." if line_name else ".")
        ),
        add_ins="Verify that the appointment is specifically for the FIRST beauty line of a luxury fashion house (the brand's first beauty venture).",
    )

    # p4_appointment_announcement_date
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p4_appointment_announcement_date",
        "Verify the appointment announcement date is March 5, 2025 (supporting URL).",
        required_present=bool(appoint_date),
        urls=appointment_urls,
        claim=f"The appointment was announced on March 5, 2025; the stated date is '{appoint_date}'.",
        add_ins="Confirm the announcement date is exactly March 5, 2025.",
    )

    # p4_beauty_line_name
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p4_beauty_line_name",
        "Provide the name of the luxury fashion house’s beauty line (supporting URL).",
        required_present=bool(line_name),
        urls=line_urls,
        claim=f"The beauty line is named '{line_name}'.",
        add_ins="The cited source should state the official beauty line name.",
    )

    # p4_preorder_date
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p4_preorder_date",
        "Verify pre-orders began on August 25, 2025 (supporting URL).",
        required_present=bool(pre_date),
        urls=pre_urls,
        claim=f"Pre-orders for the beauty line began on August 25, 2025; the stated date is '{pre_date}'.",
        add_ins="Confirm the pre-orders start date is exactly August 25, 2025.",
    )

    # p4_launch_date
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p4_launch_date",
        "Verify the official launch date is August 29, 2025 (supporting URL).",
        required_present=bool(launch_date),
        urls=launch_urls,
        claim=f"The official launch date for the beauty line is August 29, 2025; the stated date is '{launch_date}'.",
        add_ins="Confirm the launch date is exactly August 29, 2025.",
    )

    # p4_product_counts
    await _add_requirement_with_sources(
        evaluator,
        node,
        "p4_product_counts",
        "Verify the product line includes exactly 55 lipstick shades, 10 tinted balms, and 8 eyeshadow palettes (supporting URL).",
        required_present=bool(lip) and bool(balm) and bool(palettes),
        urls=product_urls,
        claim=(
            "The product line includes exactly 55 lipstick shades, 10 tinted balms, and 8 eyeshadow palettes."
            f" The stated counts are: lipsticks='{lip}', tinted_balms='{balm}', eyeshadow_palettes='{palettes}'."
        ),
        add_ins="Confirm the exact counts (55, 10, 8) from official brand communications or reputable press coverage.",
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
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
    """
    Evaluate an answer for the 2025–2026 celebrity fashion/beauty professionals task.
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
        default_model=model,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_professionals(),
        template_class=ProfessionalsExtraction,
        extraction_name="professionals_extraction",
    )

    # Build verification subtrees
    await verify_professional_1(evaluator, root, extracted.professional1)
    await verify_professional_2(evaluator, root, extracted.professional2)
    await verify_professional_3(evaluator, root, extracted.professional3)
    await verify_professional_4(evaluator, root, extracted.professional4)

    # Return final summary
    return evaluator.get_summary()