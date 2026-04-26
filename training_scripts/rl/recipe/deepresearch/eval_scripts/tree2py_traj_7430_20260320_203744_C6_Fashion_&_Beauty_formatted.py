import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_brand_campaign_redcarpet_2024_2026"
TASK_DESCRIPTION = """
Identify a celebrity who satisfies all of the following criteria:

1. Brand Ambassador Partnership (2024-2025): The celebrity was officially identified as a brand ambassador for a luxury fashion house, beauty brand, or luxury accessories brand between January 1, 2024 and December 31, 2025. Provide the public identification date, the brand name, and the brand category (fashion/beauty/accessories).

2. Major Campaign Participation (2025): The same celebrity fronted a major advertising campaign for the same brand during Spring/Summer 2025 or Fall/Winter 2025. Provide the campaign season, the official campaign name or theme, and at least one official creative credit (photographer or director).

3. Red Carpet Appearance (2026): The same celebrity wore a custom (not ready-to-wear) creation by a luxury fashion house to a major internationally recognized red carpet event between January 1, 2026 and March 20, 2026 (e.g., BAFTAs, Oscars, Golden Globes, SAG Awards, or a major fashion week). Provide the event name, the specific event date, the fashion house, and—if documented—the creative director or designer.

For each of the three components, include reference URLs from official brand sources, established fashion publications (e.g., WWD, Vogue, Business of Fashion, Harper's Bazaar), or verified news outlets. All information must be verifiable and consistent across sources.
"""

ALLOWED_BRAND_CATEGORIES = {"fashion", "beauty", "accessories"}
ALLOWED_EVENT_WINDOW_TEXT = "between January 1, 2026 and March 20, 2026"
ALLOWED_AMB_WINDOW_TEXT = "between January 1, 2024 and December 31, 2025"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AmbassadorDetails(BaseModel):
    announcement_date: Optional[str] = None
    brand_name: Optional[str] = None
    brand_category: Optional[str] = None  # expect one of fashion/beauty/accessories (lowercase)
    sources: List[str] = Field(default_factory=list)


class CampaignDetails(BaseModel):
    brand_name: Optional[str] = None
    season: Optional[str] = None  # e.g., "Spring/Summer 2025", "SS25", "Fall/Winter 2025", "FW25"
    campaign_name_or_theme: Optional[str] = None
    creative_credit_name: Optional[str] = None  # photographer or director name
    creative_credit_role: Optional[str] = None  # e.g., photographer, director
    sources: List[str] = Field(default_factory=list)


class RedCarpetDetails(BaseModel):
    event_name: Optional[str] = None
    event_date: Optional[str] = None  # specific date string provided in the answer
    fashion_house: Optional[str] = None
    outfit_type: Optional[str] = None  # expected strings like "custom", "couture", "bespoke", or "ready-to-wear/rtw"
    designer_or_cd: Optional[str] = None  # if documented
    sources: List[str] = Field(default_factory=list)


class CelebrityExtraction(BaseModel):
    celebrity_name: Optional[str] = None
    ambassador: Optional[AmbassadorDetails] = None
    campaign: Optional[CampaignDetails] = None
    red_carpet: Optional[RedCarpetDetails] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_celebrity_details() -> str:
    return """
    Extract exactly one celebrity and the details for the three required components from the answer text.

    Fields to extract (use null for any field not explicitly provided in the answer):

    celebrity_name: The celebrity's full name.

    ambassador:
      announcement_date: The specific public date when the celebrity was identified as an ambassador (as written).
      brand_name: The luxury brand's name.
      brand_category: One of: "fashion", "beauty", or "accessories". Use lowercase if present; return null if not provided.
      sources: All URLs cited for the ambassadorship/date claim.

    campaign:
      brand_name: The brand name for the campaign (should match ambassadorship brand).
      season: The campaign season as written (e.g., "Spring/Summer 2025", "SS25", "Fall/Winter 2025", "FW25").
      campaign_name_or_theme: The official campaign name or theme (as written).
      creative_credit_name: The named photographer or director (as written).
      creative_credit_role: "photographer" or "director" if stated; otherwise null.
      sources: All URLs cited for campaign season/name/theme/creative credits.

    red_carpet:
      event_name: The red carpet event name (e.g., Oscars, BAFTAs, Golden Globes, SAG Awards, or specific Fashion Week show/event).
      event_date: The specific event date (as written).
      fashion_house: The luxury fashion house that created the custom look.
      outfit_type: The phrasing used regarding the look type (e.g., "custom", "couture", "bespoke", or "ready-to-wear"/"RTW").
      designer_or_cd: The designer or creative director name if documented; otherwise null.
      sources: All URLs cited that document the red carpet appearance details.

    Important:
    - Extract only what appears in the answer text. Do not infer or invent.
    - For any URLs, return the actual URL strings present in the answer.
    - For brand_category, return a single lowercase word: "fashion", "beauty", or "accessories" if stated; else null.
    - If multiple URLs are given for a component, include them all in the corresponding sources list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def category_is_allowed(cat: Optional[str]) -> bool:
    if not is_non_empty(cat):
        return False
    return cat.strip().lower() in ALLOWED_BRAND_CATEGORIES


def season_is_allowed(season: Optional[str]) -> bool:
    """
    Accept any reasonable variant for SS25 / FW25:
    - "Spring/Summer 2025", "SS25", "S/S 2025"
    - "Fall/Winter 2025", "FW25", "F/W 2025", "Autumn/Winter 2025", "AW25", "A/W 2025"
    """
    if not is_non_empty(season):
        return False
    s = season.strip().lower()
    if "2025" in s:
        if ("spring" in s or "ss" in s or "s/s" in s) and ("summer" in s or "ss" in s or "s/s" in s):
            return True
        if any(k in s for k in ["fall", "autumn", "fw", "f/w", "aw", "a/w"]) and any(
            k in s for k in ["winter", "fw", "f/w", "aw", "a/w"]
        ):
            return True
    if s in {"ss25", "fw25", "aw25"}:
        return True
    return False


def join_urls(urls: Optional[List[str]]) -> List[str]:
    return list(urls or [])


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_ambassador_subtree(
    evaluator: Evaluator,
    parent: VerificationNode,
    celeb_name: Optional[str],
    amb: Optional[AmbassadorDetails],
) -> Dict[str, VerificationNode]:
    node = evaluator.add_parallel(
        id="Brand_Ambassador_Partnership_2024_2025",
        desc="Ambassador identified in 2024-2025 with brand/category and allowed-source citation",
        parent=parent,
        critical=True,  # all children critical inside this group
    )

    amb_sources = join_urls(amb.sources if amb else [])
    brand_name = amb.brand_name if amb else None
    brand_cat = amb.brand_category if amb else None
    ann_date = amb.announcement_date if amb else None
    celeb = celeb_name or ""

    # Presence checks (critical)
    srcs_provided = evaluator.add_custom_node(
        result=len(amb_sources) > 0,
        id="ambassador_sources_provided",
        desc="Ambassador: at least one source URL is provided",
        parent=node,
        critical=True,
    )
    date_provided = evaluator.add_custom_node(
        result=is_non_empty(ann_date),
        id="ambassador_date_provided",
        desc="Ambassador: announcement date is provided",
        parent=node,
        critical=True,
    )
    brand_name_provided = evaluator.add_custom_node(
        result=is_non_empty(brand_name),
        id="ambassador_brand_name_provided",
        desc="Ambassador: brand name is provided",
        parent=node,
        critical=True,
    )
    brand_cat_allowed = evaluator.add_custom_node(
        result=category_is_allowed(brand_cat),
        id="ambassador_brand_category_allowed",
        desc="Ambassador: brand category provided and is one of fashion/beauty/accessories",
        parent=node,
        critical=True,
    )

    # Date in range + supported by sources
    date_in_range_supported = evaluator.add_leaf(
        id="ambassador_date_in_range_supported",
        desc=f"Ambassador: the announcement on '{ann_date}' is within 2024-01-01 to 2025-12-31 and is supported by sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{celeb} was publicly identified as a brand ambassador for {brand_name} on {ann_date}. "
              f"This public identification date falls {ALLOWED_AMB_WINDOW_TEXT}.",
        node=date_in_range_supported,
        sources=amb_sources,
        additional_instruction=(
            "Confirm the page(s) explicitly identify the person as an 'ambassador' (allow 'brand ambassador', "
            "'global ambassador', 'house ambassador' etc.) and that the public identification date matches or is "
            "clearly stated; also confirm the date falls between 2024-01-01 and 2025-12-31."
        ),
    )

    # Luxury brand supported
    brand_is_luxury = evaluator.add_leaf(
        id="ambassador_brand_is_luxury",
        desc="Ambassador: cited source(s) support that the brand is a luxury brand in the stated category",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{brand_name} is a luxury {brand_cat} brand.",
        node=brand_is_luxury,
        sources=amb_sources,
        additional_instruction=(
            "Judge from the page whether the brand is recognized as a luxury house/brand in the stated category. "
            "Mentions of luxury positioning, couture/haute couture, and coverage by top-tier fashion press count."
        ),
    )

    # Ambassador status explicit
    amb_status_explicit = evaluator.add_leaf(
        id="ambassador_status_explicit",
        desc="Ambassador: sources explicitly identify the celebrity as an ambassador (not just appearing in content)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page explicitly identifies {celeb} as an ambassador of {brand_name}.",
        node=amb_status_explicit,
        sources=amb_sources,
        additional_instruction=(
            "Look for explicit terms like 'ambassador', 'brand ambassador', 'global ambassador', 'house ambassador'. "
            "Merely starring in a campaign without ambassador wording is insufficient."
        ),
    )

    # Allowed source type
    amb_src_allowed = evaluator.add_leaf(
        id="ambassador_source_url_allowed_type",
        desc="Ambassador: at least one cited URL is from an allowed source type",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one of the provided ambassador URLs is an allowed source type: an official brand source "
              "(brand website or official press release) or an established fashion publication (e.g., Vogue, WWD, "
              "Business of Fashion, Harper's Bazaar, Elle, GQ, Vanity Fair) or a verified mainstream news outlet.",
        node=amb_src_allowed,
        sources=amb_sources,
        additional_instruction=(
            "Assess source type by domain and page context. Social media posts do NOT count unless they are official "
            "brand newsroom/press pages on the brand's own site. At least one URL must be allowed and document the "
            "ambassador identification and date."
        ),
    )

    return {
        "node": node,
        "brand_name_provided": brand_name_provided,
    }


async def build_campaign_subtree(
    evaluator: Evaluator,
    parent: VerificationNode,
    celeb_name: Optional[str],
    camp: Optional[CampaignDetails],
    amb_brand_name_provided_node: VerificationNode,
    amb_brand_name: Optional[str],
) -> Dict[str, VerificationNode]:
    node = evaluator.add_parallel(
        id="Major_Campaign_Participation_2025",
        desc="Major campaign in SS25 or FW25 for the same brand; includes creative credits; allowed-source citation",
        parent=parent,
        critical=True,
    )

    camp_sources = join_urls(camp.sources if camp else [])
    c_brand = camp.brand_name if camp else None
    c_season = camp.season if camp else None
    c_name_theme = camp.campaign_name_or_theme if camp else None
    c_credit_name = camp.creative_credit_name if camp else None
    c_credit_role = camp.creative_credit_role if camp else None
    celeb = celeb_name or ""
    amb_brand = amb_brand_name or ""

    # Presence checks (critical)
    srcs_provided = evaluator.add_custom_node(
        result=len(camp_sources) > 0,
        id="campaign_sources_provided",
        desc="Campaign: at least one source URL is provided",
        parent=node,
        critical=True,
    )
    camp_brand_provided = evaluator.add_custom_node(
        result=is_non_empty(c_brand),
        id="campaign_brand_name_provided",
        desc="Campaign: brand name is provided",
        parent=node,
        critical=True,
    )
    camp_season_provided = evaluator.add_custom_node(
        result=is_non_empty(c_season),
        id="campaign_season_provided",
        desc="Campaign: season is provided",
        parent=node,
        critical=True,
    )
    camp_season_allowed = evaluator.add_custom_node(
        result=season_is_allowed(c_season),
        id="campaign_season_allowed",
        desc="Campaign: season value is allowed (SS25 or FW25 variants)",
        parent=node,
        critical=True,
    )
    camp_name_theme_provided = evaluator.add_custom_node(
        result=is_non_empty(c_name_theme),
        id="campaign_name_theme_provided",
        desc="Campaign: official name or theme is provided",
        parent=node,
        critical=True,
    )
    camp_credit_provided = evaluator.add_custom_node(
        result=is_non_empty(c_credit_name),
        id="campaign_creative_credit_provided",
        desc="Campaign: at least one creative credit (photographer or director) is provided",
        parent=node,
        critical=True,
    )

    # Same brand as ambassadorship (logical check)
    same_brand = evaluator.add_leaf(
        id="campaign_same_brand_as_ambassador",
        desc="Campaign: campaign brand matches ambassadorship brand",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The campaign brand '{c_brand}' and the ambassadorship brand '{amb_brand}' are the same brand.",
        node=same_brand,
        additional_instruction="Treat minor casing/punctuation differences as the same brand.",
        extra_prerequisites=[amb_brand_name_provided_node, camp_brand_provided],
    )

    # Season supported by sources
    season_supported = evaluator.add_leaf(
        id="campaign_season_supported",
        desc=f"Campaign: sources document the {c_season} campaign",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page documents {c_brand}'s {c_season} advertising campaign.",
        node=season_supported,
        sources=camp_sources,
        additional_instruction="Confirm that the page refers to the brand's advertising campaign for the specified season (SS25/FW25).",
    )

    # Fronted by celebrity
    fronted_confirmed = evaluator.add_leaf(
        id="campaign_fronted_by_celebrity",
        desc="Campaign: sources confirm the celebrity fronted (was face/lead) the campaign",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page confirms that {celeb} fronted (was the lead/face of) {c_brand}'s {c_season} campaign.",
        node=fronted_confirmed,
        sources=camp_sources,
        additional_instruction="Accept synonyms: 'fronts', 'stars in', 'leads', 'faces', 'the face of', etc.",
    )

    # Campaign name/theme supported
    name_theme_supported = evaluator.add_leaf(
        id="campaign_name_theme_supported",
        desc="Campaign: sources document the official campaign name/theme",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page documents the official campaign name or theme as '{c_name_theme}'.",
        node=name_theme_supported,
        sources=camp_sources,
        additional_instruction="Look for explicit naming or clear thematic designation used by the brand or top-tier fashion press.",
    )

    # Creative credit supported
    credit_supported = evaluator.add_leaf(
        id="campaign_creative_credit_supported",
        desc="Campaign: sources document at least one official creative credit (photographer/director)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page credits {c_credit_name} as the {c_credit_role or 'photographer/director'} of the campaign.",
        node=credit_supported,
        sources=camp_sources,
        additional_instruction="Accept 'shot by', 'photographed by', 'directed by', 'filmed by', or explicit credit lists.",
    )

    # Allowed source type
    camp_src_allowed = evaluator.add_leaf(
        id="campaign_source_url_allowed_type",
        desc="Campaign: at least one cited URL is from an allowed source type",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one of the provided campaign URLs is an allowed source type (official brand source or "
              "established fashion publication or verified news outlet) and documents the season, name/theme, and "
              "creative credit(s).",
        node=camp_src_allowed,
        sources=camp_sources,
        additional_instruction=(
            "Assess the domain and content. Social media posts don't count. The allowed page(s) must substantively "
            "cover campaign season, name/theme, and creative credit(s)."
        ),
    )

    return {"node": node}


async def build_red_carpet_subtree(
    evaluator: Evaluator,
    parent: VerificationNode,
    celeb_name: Optional[str],
    rc: Optional[RedCarpetDetails],
) -> Dict[str, VerificationNode]:
    node = evaluator.add_parallel(
        id="Red_Carpet_Appearance_Early_2026",
        desc="Custom (not RTW) red carpet appearance at a major event (2026-01-01 to 2026-03-20) with allowed sources",
        parent=parent,
        critical=True,
    )

    rc_sources = join_urls(rc.sources if rc else [])
    event_name = rc.event_name if rc else None
    event_date = rc.event_date if rc else None
    house = rc.fashion_house if rc else None
    outfit_type = rc.outfit_type if rc else None
    celeb = celeb_name or ""

    # Presence checks (critical)
    srcs_provided = evaluator.add_custom_node(
        result=len(rc_sources) > 0,
        id="redcarpet_sources_provided",
        desc="Red Carpet: at least one source URL is provided",
        parent=node,
        critical=True,
    )
    event_name_provided = evaluator.add_custom_node(
        result=is_non_empty(event_name),
        id="redcarpet_event_name_provided",
        desc="Red Carpet: event name is provided",
        parent=node,
        critical=True,
    )
    event_date_provided = evaluator.add_custom_node(
        result=is_non_empty(event_date),
        id="redcarpet_event_date_provided",
        desc="Red Carpet: event date is provided",
        parent=node,
        critical=True,
    )
    house_provided = evaluator.add_custom_node(
        result=is_non_empty(house),
        id="redcarpet_fashion_house_provided",
        desc="Red Carpet: fashion house is provided",
        parent=node,
        critical=True,
    )

    # Event qualifies as major (verify via sources)
    event_is_major = evaluator.add_leaf(
        id="redcarpet_event_is_major",
        desc="Red Carpet: event qualifies as a major internationally recognized red carpet event",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page indicates that '{event_name}' is a major internationally recognized red carpet event "
              f"(e.g., Oscars, BAFTAs, Golden Globes, SAG Awards, or a major Fashion Week event).",
        node=event_is_major,
        sources=rc_sources,
        additional_instruction="If the event matches those examples or is comparable in global significance, consider it major.",
    )

    # Event date in allowed window supported
    date_in_window_supported = evaluator.add_leaf(
        id="redcarpet_event_date_in_range_supported",
        desc=f"Red Carpet: the event date '{event_date}' is within 2026-01-01 to 2026-03-20 and is supported by sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{celeb}'s red carpet appearance at {event_name} occurred on {event_date}, which falls {ALLOWED_EVENT_WINDOW_TEXT}.",
        node=date_in_window_supported,
        sources=rc_sources,
        additional_instruction="Confirm the event date stated on the page and ensure it falls within the specified window.",
    )

    # Fashion house is luxury
    house_is_luxury = evaluator.add_leaf(
        id="redcarpet_house_is_luxury",
        desc="Red Carpet: sources support that the fashion house is a luxury house",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{house} is a luxury fashion house.",
        node=house_is_luxury,
        sources=rc_sources,
        additional_instruction="Use page context to judge luxury status; couture/haute couture or top-tier fashion press coverage indicates luxury.",
    )

    # Custom (not RTW) confirmed
    custom_confirmed = evaluator.add_leaf(
        id="redcarpet_custom_not_rtw_confirmed",
        desc="Red Carpet: sources explicitly document the look is custom (not ready-to-wear)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page documents that {celeb} wore a custom (not ready-to-wear) creation by {house}.",
        node=custom_confirmed,
        sources=rc_sources,
        additional_instruction="Accept explicit phrases like 'custom', 'bespoke', 'made-to-measure', 'haute couture'; deny if only 'ready-to-wear/RTW' is indicated.",
    )

    # Allowed source type
    rc_src_allowed = evaluator.add_leaf(
        id="redcarpet_source_url_allowed_type",
        desc="Red Carpet: at least one cited URL is from an allowed source type",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one of the provided red carpet URLs is an allowed source type (established fashion publication, "
              "official event documentation, official brand source, or verified mainstream news outlet) that documents "
              "the event, date, and custom look details.",
        node=rc_src_allowed,
        sources=rc_sources,
        additional_instruction="Assess the domain and content; social-only posts don't count. The allowed page must clearly include event/date/custom details.",
    )

    return {"node": node}


async def build_optional_designer_subtree(
    evaluator: Evaluator,
    parent: VerificationNode,
    rc: Optional[RedCarpetDetails],
):
    """
    Optional: If documented, provide and support designer/creative director.
    Non-critical and sequential: if name not provided, support check will be skipped.
    """
    node = evaluator.add_sequential(
        id="Red_Carpet_Designer_If_Documented",
        desc="Optional: designer/creative director for the custom look, if documented",
        parent=parent,
        critical=False,
    )

    rc_sources = join_urls(rc.sources if rc else [])
    designer = rc.designer_or_cd if rc else None
    house = rc.fashion_house if rc else None

    provided = evaluator.add_custom_node(
        result=is_non_empty(designer),
        id="redcarpet_designer_provided",
        desc="Red Carpet (optional): designer/creative director name is provided if documented",
        parent=node,
        critical=False,
    )

    supported = evaluator.add_leaf(
        id="redcarpet_designer_supported",
        desc="Red Carpet (optional): sources document the named designer/creative director",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The page documents that the custom look by {house} was by designer/creative director '{designer}'.",
        node=supported,
        sources=rc_sources,
        additional_instruction="If the designer/creative director is not provided in the answer, skip.",
        # Ensure skip when 'provided' failed in this sequential group
        extra_prerequisites=[provided],
    )


async def build_cross_consistency_subtree(
    evaluator: Evaluator,
    parent: VerificationNode,
    extraction: CelebrityExtraction,
):
    node = evaluator.add_parallel(
        id="Cross_Source_Consistency",
        desc="Cross-source consistency checks",
        parent=parent,
        critical=True,
    )

    celeb = extraction.celebrity_name or ""
    amb_brand = (extraction.ambassador.brand_name if extraction.ambassador else None) or ""
    camp_brand = (extraction.campaign.brand_name if extraction.campaign else None) or ""

    # Same celebrity across all three criteria (logical check from the answer)
    same_celeb = evaluator.add_leaf(
        id="same_celebrity_all_three",
        desc="All three criteria attributed to the same celebrity in the answer",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The ambassadorship, campaign, and red carpet sections all refer to the same celebrity: {celeb}.",
        node=same_celeb,
        additional_instruction="Focus on consistency within the answer; allow minor name formatting differences.",
    )

    # No internal contradictions (meta-check over the answer)
    no_contradictions = evaluator.add_leaf(
        id="no_internal_source_contradictions",
        desc="No contradictions across key facts (dates, brand, season/name/theme, credits, custom status)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="There are no contradictions across the provided details in the answer regarding dates, brand/fashion house, "
              "campaign season/name/theme, creative credits, and custom status.",
        node=no_contradictions,
        additional_instruction="Use only the answer context; check obvious internal inconsistencies.",
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator
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

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_celebrity_details(),
        template_class=CelebrityExtraction,
        extraction_name="celebrity_extraction",
    )

    # Optionally record allowed constraints for transparency
    evaluator.add_custom_info(
        info={
            "ambassador_window": ALLOWED_AMB_WINDOW_TEXT,
            "campaign_season_allowed": ["Spring/Summer 2025 (SS25)", "Fall/Winter 2025 (FW25)"],
            "red_carpet_window": ALLOWED_EVENT_WINDOW_TEXT,
            "allowed_brand_categories": sorted(list(ALLOWED_BRAND_CATEGORIES)),
        },
        info_type="constraints",
        info_name="allowed_constraints",
    )

    # Build a top-level task node (non-critical to allow a mix of critical/non-critical children beneath)
    task_node = evaluator.add_parallel(
        id="Celebrity_Identification",
        desc="Identify a single celebrity meeting ambassadorship (2024-2025), major campaign (2025), and custom red carpet (early 2026) with verifiable sources",
        parent=root,
        critical=False,
    )

    # Celebrity name presence (critical basic requirement)
    celeb_name_provided = evaluator.add_custom_node(
        result=is_non_empty(extraction.celebrity_name),
        id="celebrity_name_provided",
        desc="Provide the celebrity's name",
        parent=task_node,
        critical=True,
    )

    # Ambassador subtree
    amb_result = await build_ambassador_subtree(
        evaluator=evaluator,
        parent=task_node,
        celeb_name=extraction.celebrity_name,
        amb=extraction.ambassador,
    )
    amb_brand_name_node = amb_result.get("brand_name_provided")

    # Campaign subtree (depends on ambassador brand being present)
    await build_campaign_subtree(
        evaluator=evaluator,
        parent=task_node,
        celeb_name=extraction.celebrity_name,
        camp=extraction.campaign,
        amb_brand_name_provided_node=amb_brand_name_node,
        amb_brand_name=(extraction.ambassador.brand_name if extraction.ambassador else None),
    )

    # Red carpet subtree
    await build_red_carpet_subtree(
        evaluator=evaluator,
        parent=task_node,
        celeb_name=extraction.celebrity_name,
        rc=extraction.red_carpet,
    )

    # Optional designer/creative director (non-critical)
    await build_optional_designer_subtree(
        evaluator=evaluator,
        parent=task_node,
        rc=extraction.red_carpet,
    )

    # Cross-source consistency checks
    await build_cross_consistency_subtree(
        evaluator=evaluator,
        parent=task_node,
        extraction=extraction,
    )

    # Return evaluation summary
    return evaluator.get_summary()