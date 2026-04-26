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
TASK_ID = "celebrity_spring_2026_campaign"
TASK_DESCRIPTION = (
    "Identify a celebrity who meets all of the following criteria: (1) The celebrity is an actor, musician, or "
    "entertainment industry professional; (2) The celebrity appeared as featured talent in a major fashion brand's "
    "Spring 2026 campaign; (3) The campaign was publicly announced or released between January and March 2026; "
    "(4) The celebrity has a professional celebrity stylist for public appearances; (5) The celebrity attended at least "
    "one major 2026 awards show (Golden Globes, Oscars, Grammys, Critics Choice Awards, SAG Awards, BAFTA, Met Gala, or "
    "Emmy Awards). Provide the celebrity's full name, the fashion brand name, the month and year when the campaign was "
    "announced, and at least one major awards show they attended in 2026. Additionally, if documented in media sources, "
    "provide the name of their celebrity stylist and at least one specific designer brand they wore at a 2026 awards show."
)

ALLOWED_MAJOR_EVENTS_2026 = [
    "Golden Globes",
    "Academy Awards",
    "Oscars",
    "Grammy Awards",
    "Grammys",
    "Critics Choice Awards",
    "SAG Awards",
    "Screen Actors Guild Awards",
    "BAFTA",
    "BAFTA Film Awards",
    "Met Gala",
    "Emmy Awards",
    "Emmys",
]
JFM_2026_MONTHS = ["January", "February", "March"]
JFM_2026_RANGE_DESC = "between January 1 and March 31, 2026 (inclusive)"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CelebrityCaseExtraction(BaseModel):
    # Core identification
    celebrity_name: Optional[str] = None
    profession: Optional[str] = None
    bio_sources: List[str] = Field(default_factory=list)

    # Campaign details
    brand_name: Optional[str] = None
    campaign_season: Optional[str] = None  # e.g., "Spring 2026", "Spring/Summer 2026", "SS26"
    campaign_role: Optional[str] = None  # e.g., "face", "featured talent", "ambassador"
    campaign_announcement_month: Optional[str] = None  # Expect January/February/March
    campaign_announcement_year: Optional[str] = None  # Expect "2026"
    campaign_announcement_source: Optional[str] = None  # Single best URL to announcement if present
    campaign_sources: List[str] = Field(default_factory=list)  # All URLs about the campaign
    brand_info_sources: List[str] = Field(default_factory=list)  # Brand profile/official/press or Wikipedia

    # Stylist details
    stylist_name: Optional[str] = None
    stylist_sources: List[str] = Field(default_factory=list)
    stylist_other_clients: List[str] = Field(default_factory=list)
    stylist_more_sources: List[str] = Field(default_factory=list)

    # Awards 2026 attendance
    awards_attended_2026: List[str] = Field(default_factory=list)
    primary_award_2026: Optional[str] = None  # One selected major event from the allowed list
    primary_award_source: Optional[str] = None  # A URL directly supporting attendance
    awards_sources: List[str] = Field(default_factory=list)  # Any additional awards coverage URLs

    # Designer worn at a 2026 awards show
    designer_worn_2026: Optional[str] = None
    designer_event_2026: Optional[str] = None  # The specific 2026 event where the designer was worn, if provided
    designer_sources: List[str] = Field(default_factory=list)

    # Red carpet media coverage (with styling credits)
    redcarpet_media_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_celebrity_case() -> str:
    return """
    From the provided answer, extract a single celebrity case that the answer claims satisfies the task. Return a JSON object with these fields:

    1) celebrity_name: Full name of the chosen celebrity.
    2) profession: The celebrity's primary profession as described (e.g., actor, musician, entertainer). If multiple are listed, keep the primary one.
    3) bio_sources: Array of URLs explicitly present in the answer that substantiate the celebrity's profession (e.g., Wikipedia, IMDb, official bio, reputable media).

    4) brand_name: Name of the fashion brand whose Spring 2026 campaign features this celebrity as talent.
    5) campaign_season: As written (e.g., "Spring 2026", "Spring/Summer 2026", "SS26") if present.
    6) campaign_role: The described role such as "face", "featured talent", "ambassador", or similar (if mentioned).
    7) campaign_announcement_month: Month when the Spring 2026 campaign was publicly announced or released, if given (January, February, or March).
    8) campaign_announcement_year: Year of announcement/release (expect "2026" if provided).
    9) campaign_announcement_source: A single best URL (from the answer) that most directly announces/releases the Spring 2026 campaign.
    10) campaign_sources: Array of all URLs from the answer that cover or announce the Spring 2026 campaign.
    11) brand_info_sources: Array of URLs (from the answer) that help establish the brand as a globally recognized designer brand or major fashion house (e.g., official site, Wikipedia, Vogue/WWD coverage).

    12) stylist_name: The professional celebrity stylist's name for this celebrity's public/awards appearances (if provided).
    13) stylist_sources: Array of URLs from the answer that credit/confirm the stylist-client relationship for this celebrity.
    14) stylist_other_clients: Array of other high-profile celebrity clients for this stylist, as listed in the answer (if any).
    15) stylist_more_sources: Array of additional URLs about the stylist's broader work/clients or industry recognition (from the answer).

    16) awards_attended_2026: Array of 2026 major events the celebrity attended as listed in the answer. Use event names as written (e.g., "Oscars", "Golden Globes", "SAG Awards", "BAFTA", "Met Gala", "Grammys", "Emmys", "Critics Choice Awards").
    17) primary_award_2026: Choose one major 2026 event from awards_attended_2026 that is best supported by a specific URL in the answer.
    18) primary_award_source: A single URL (from the answer) that directly supports attendance at the selected primary_award_2026.
    19) awards_sources: Array of any other URLs in the answer that document 2026 awards attendance.

    20) designer_worn_2026: The name of a specific designer brand the celebrity wore at a 2026 major awards show, if provided (e.g., "Prada", "Louis Vuitton").
    21) designer_event_2026: The specific 2026 event where the above designer was worn (if mentioned).
    22) designer_sources: Array of URLs from the answer showing outfit credits confirming the designer for that 2026 event.

    23) redcarpet_media_sources: Array of URLs from the answer showing fashion/red-carpet media coverage with styling credits for the celebrity's 2026 awards appearance.

    Important rules:
    - Only extract URLs that are explicitly present in the answer. Do not fabricate or infer URLs.
    - If a field is not present in the answer, set it to null (for strings) or [] (for arrays).
    - Keep strings exactly as they appear in the answer when reasonable.
    - Do not include more than one celebrity; pick the single main one if multiple are listed.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _to_list(x: Optional[Any]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(u).strip() for u in x if isinstance(u, str) and str(u).strip()]
    if isinstance(x, str):
        u = x.strip()
        return [u] if u else []
    return []


def combine_sources(*items: Any) -> List[str]:
    """Flatten and deduplicate URL lists/strings, preserving order."""
    seen = set()
    out: List[str] = []
    for it in items:
        for url in _to_list(it):
            if url not in seen:
                seen.add(url)
                out.append(url)
    return out


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def verify_celebrity_case(
    evaluator: Evaluator,
    root_node,
    data: CelebrityCaseExtraction,
) -> None:
    # 0) Organize the top-level structure: non-critical root with two sections
    core = evaluator.add_parallel(
        id="core_requirements",
        desc="All mandatory criteria must be satisfied",
        parent=root_node,
        critical=True,  # Gate by mandatory checks only
    )

    supplemental = evaluator.add_parallel(
        id="supplemental_evidence",
        desc="Optional supplemental evidence for stronger support",
        parent=root_node,
        critical=False,  # Partial credit allowed here
    )

    # ------------------------ Core 1: Entertainment Professional ------------------------
    seq_ent = evaluator.add_sequential(
        id="entertainment_professional_main",
        desc="Celebrity is an actor, musician, or entertainment industry professional",
        parent=core,
        critical=True,
    )
    ent_exists = bool(data.celebrity_name and data.profession and len(data.bio_sources) > 0)
    evaluator.add_custom_node(
        result=ent_exists,
        id="entertainment_professional_exists",
        desc="Required info and at least one bio/profession source are provided",
        parent=seq_ent,
        critical=True,
    )
    ent_leaf = evaluator.add_leaf(
        id="entertainment_professional_verify",
        desc="The identified celebrity is an actor, musician, or entertainment industry professional",
        parent=seq_ent,
        critical=True,
    )
    ent_claim = (
        f"{data.celebrity_name} is primarily an actor, musician, or entertainment industry professional "
        f"(not primarily a model, athlete, or social media influencer)."
    )
    await evaluator.verify(
        claim=ent_claim,
        node=ent_leaf,
        sources=data.bio_sources,
        additional_instruction=(
            "Use the provided source(s) to determine the celebrity's primary public profession. "
            "If they have multiple roles, accept if they are widely recognized as an actor/musician/entertainment professional."
        ),
    )

    # ------------------------ Core 2: Campaign Participation ------------------------
    seq_camp = evaluator.add_sequential(
        id="campaign_participation_main",
        desc="Celebrity appeared as featured talent/face in a Spring 2026 fashion campaign",
        parent=core,
        critical=True,
    )
    camp_exists = bool(
        data.celebrity_name and data.brand_name and len(data.campaign_sources) > 0
    )
    evaluator.add_custom_node(
        result=camp_exists,
        id="campaign_participation_exists",
        desc="Campaign info and at least one campaign source are provided",
        parent=seq_camp,
        critical=True,
    )
    camp_leaf = evaluator.add_leaf(
        id="campaign_participation_verify",
        desc="The celebrity appeared as featured talent/face in the brand's Spring 2026 campaign",
        parent=seq_camp,
        critical=True,
    )
    camp_claim = (
        f"{data.celebrity_name} appeared as featured talent (e.g., face/star/ambassador) in "
        f"{data.brand_name}'s Spring 2026 campaign."
    )
    await evaluator.verify(
        claim=camp_claim,
        node=camp_leaf,
        sources=data.campaign_sources,
        additional_instruction=(
            "Verify that the sources explicitly state the celebrity appears in the Spring 2026 (or Spring/Summer 2026, SS26) campaign "
            "as featured talent/face/star/ambassador."
        ),
    )

    # ------------------------ Core 3: Brand Is Major ------------------------
    seq_brand = evaluator.add_sequential(
        id="brand_is_major_main",
        desc="Campaign brand is a globally recognized, established designer brand or major fashion house",
        parent=core,
        critical=True,
    )
    brand_sources = combine_sources(data.brand_info_sources, data.campaign_sources)
    brand_exists = bool(data.brand_name and len(brand_sources) > 0)
    evaluator.add_custom_node(
        result=brand_exists,
        id="brand_is_major_exists",
        desc="Brand name and at least one brand/campaign source are provided",
        parent=seq_brand,
        critical=True,
    )
    brand_leaf = evaluator.add_leaf(
        id="brand_is_major_verify",
        desc="The campaign brand is a major, globally recognized designer brand/fashion house",
        parent=seq_brand,
        critical=True,
    )
    brand_claim = f"{data.brand_name} is a globally recognized, established designer brand or major fashion house."
    await evaluator.verify(
        claim=brand_claim,
        node=brand_leaf,
        sources=brand_sources,
        additional_instruction=(
            "Judge based on the provided sources whether the brand is globally recognized (e.g., shows at global fashion weeks, "
            "covered by Vogue/WWD, part of groups like LVMH/Kering, or otherwise clearly established)."
        ),
    )

    # ------------------------ Core 4: Announcement Timing (Jan–Mar 2026) ------------------------
    seq_time = evaluator.add_sequential(
        id="campaign_timing_main",
        desc=f"Campaign announced or released {JFM_2026_RANGE_DESC}",
        parent=core,
        critical=True,
    )
    timing_sources = combine_sources(data.campaign_announcement_source, data.campaign_sources)
    timing_exists = bool(len(timing_sources) > 0)
    evaluator.add_custom_node(
        result=timing_exists,
        id="campaign_timing_exists",
        desc="At least one source for campaign announcement/release date is provided",
        parent=seq_time,
        critical=True,
    )
    time_leaf = evaluator.add_leaf(
        id="campaign_timing_verify",
        desc=f"The Spring 2026 campaign announcement/release is dated {JFM_2026_RANGE_DESC}",
        parent=seq_time,
        critical=True,
    )
    time_claim = (
        f"The Spring 2026 campaign featuring {data.celebrity_name} for {data.brand_name} "
        f"was publicly announced or released between January 1 and March 31, 2026."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        sources=timing_sources,
        additional_instruction=(
            "Use the article/press-release publish date or clearly stated release date on the page. "
            "Accept if the date is within Jan 1–Mar 31, 2026 (inclusive). If month/year are stated, confirm they fall in that window."
        ),
    )

    # ------------------------ Core 5: Professional Celebrity Stylist ------------------------
    seq_sty = evaluator.add_sequential(
        id="stylist_professional_main",
        desc="Celebrity works with a professional celebrity stylist for appearances",
        parent=core,
        critical=True,
    )
    sty_exists = bool(data.celebrity_name and data.stylist_name and len(data.stylist_sources) > 0)
    evaluator.add_custom_node(
        result=sty_exists,
        id="stylist_professional_exists",
        desc="Stylist name and at least one stylist-client source are provided",
        parent=seq_sty,
        critical=True,
    )
    sty_leaf = evaluator.add_leaf(
        id="stylist_professional_verify",
        desc="The celebrity works with the named professional stylist for red carpets/public appearances",
        parent=seq_sty,
        critical=True,
    )
    sty_claim = (
        f"{data.celebrity_name} works with professional celebrity stylist {data.stylist_name} for red carpet or public appearances."
    )
    await evaluator.verify(
        claim=sty_claim,
        node=sty_leaf,
        sources=data.stylist_sources,
        additional_instruction="The source(s) should explicitly credit or confirm the stylist-client relationship.",
    )

    # ------------------------ Core 6: Major 2026 Awards Attendance ------------------------
    seq_awd = evaluator.add_sequential(
        id="awards_attendance_main",
        desc="Celebrity attended at least one major 2026 awards show",
        parent=core,
        critical=True,
    )
    awards_primary_sources = combine_sources(data.primary_award_source, data.awards_sources)
    awd_exists = bool(data.celebrity_name and data.primary_award_2026 and len(awards_primary_sources) > 0)
    evaluator.add_custom_node(
        result=awd_exists,
        id="awards_attendance_exists",
        desc="A selected major awards show for 2026 and a supporting source are provided",
        parent=seq_awd,
        critical=True,
    )
    awd_leaf = evaluator.add_leaf(
        id="awards_attendance_verify",
        desc="The celebrity attended at least one major 2026 awards show",
        parent=seq_awd,
        critical=True,
    )
    awd_claim = f"{data.celebrity_name} attended the {data.primary_award_2026} in 2026."
    await evaluator.verify(
        claim=awd_claim,
        node=awd_leaf,
        sources=awards_primary_sources,
        additional_instruction=(
            "Confirm attendance at a major 2026 event (allowed examples: Golden Globes, Oscars/Academy Awards, Grammys, "
            "Critics Choice Awards, SAG Awards, BAFTA, Met Gala, Emmys). Allow common name variants (e.g., 'Oscars' = 'Academy Awards')."
        ),
    )

    # ======================== Supplemental (Non-Critical) ========================

    # 7) Stylist has multiple high-profile clients
    seq_multi_clients = evaluator.add_sequential(
        id="stylist_multiple_clients_main",
        desc="The stylist has multiple high-profile celebrity clients beyond this celebrity",
        parent=supplemental,
        critical=False,
    )
    multi_clients_sources = combine_sources(data.stylist_more_sources, data.stylist_sources)
    multi_clients_exists = bool(len(data.stylist_other_clients) > 0 and len(multi_clients_sources) > 0)
    evaluator.add_custom_node(
        result=multi_clients_exists,
        id="stylist_multiple_clients_exists",
        desc="Other clients and at least one supporting stylist source are provided",
        parent=seq_multi_clients,
        critical=False,
    )
    multi_clients_leaf = evaluator.add_leaf(
        id="stylist_multiple_clients_verify",
        desc="The stylist has multiple high-profile clients besides the chosen celebrity",
        parent=seq_multi_clients,
        critical=False,
    )
    multi_clients_list = ", ".join(data.stylist_other_clients[:5]) if data.stylist_other_clients else "others"
    multi_clients_claim = (
        f"Stylist {data.stylist_name} has other high-profile celebrity clients besides {data.celebrity_name}, "
        f"such as {multi_clients_list}."
    )
    await evaluator.verify(
        claim=multi_clients_claim,
        node=multi_clients_leaf,
        sources=multi_clients_sources,
        additional_instruction="The sources should credibly list multiple celebrity clients for the stylist.",
    )

    # 8) Stylist industry recognition in major publications
    seq_sty_rec = evaluator.add_sequential(
        id="stylist_industry_recognition_main",
        desc="The stylist is credited/recognized in major fashion/entertainment publications",
        parent=supplemental,
        critical=False,
    )
    sty_rec_sources = combine_sources(data.stylist_more_sources, data.redcarpet_media_sources, data.stylist_sources)
    sty_rec_exists = bool(len(sty_rec_sources) > 0 and data.stylist_name)
    evaluator.add_custom_node(
        result=sty_rec_exists,
        id="stylist_industry_recognition_exists",
        desc="At least one stylist/media source is provided",
        parent=seq_sty_rec,
        critical=False,
    )
    sty_rec_leaf = evaluator.add_leaf(
        id="stylist_industry_recognition_verify",
        desc="The stylist is recognized in major publications",
        parent=seq_sty_rec,
        critical=False,
    )
    sty_rec_claim = (
        f"Stylist {data.stylist_name} is credited or recognized in major fashion/entertainment publications."
    )
    await evaluator.verify(
        claim=sty_rec_claim,
        node=sty_rec_leaf,
        sources=sty_rec_sources,
        additional_instruction=(
            "Look for recognition/credits in reputable outlets (e.g., Vogue, WWD, Harper's Bazaar, Elle, "
            "The Hollywood Reporter, Vanity Fair, GQ)."
        ),
    )

    # 9) Ongoing stylist partnership (multiple documented collaborations)
    seq_ongoing = evaluator.add_sequential(
        id="ongoing_stylist_partnership_main",
        desc="The celebrity–stylist partnership has multiple documented collaborations over time",
        parent=supplemental,
        critical=False,
    )
    ongoing_sources = combine_sources(data.stylist_sources, data.awards_sources, data.designer_sources)
    ongoing_exists = bool(len(set(ongoing_sources)) >= 2 and data.celebrity_name and data.stylist_name)
    evaluator.add_custom_node(
        result=ongoing_exists,
        id="ongoing_stylist_partnership_exists",
        desc="At least two distinct sources implying multiple collaborations are provided",
        parent=seq_ongoing,
        critical=False,
    )
    ongoing_leaf = evaluator.add_leaf(
        id="ongoing_stylist_partnership_verify",
        desc="Multiple styling collaborations between the celebrity and stylist are documented",
        parent=seq_ongoing,
        critical=False,
    )
    ongoing_claim = (
        f"The professional relationship between {data.celebrity_name} and stylist {data.stylist_name} includes "
        f"more than one documented styling collaboration over time."
    )
    await evaluator.verify(
        claim=ongoing_claim,
        node=ongoing_leaf,
        sources=ongoing_sources,
        additional_instruction="Accept if there are multiple separate events/dates with the same stylist crediting the partnership.",
    )

    # 10) Specific designer worn at a 2026 awards show
    seq_designer = evaluator.add_sequential(
        id="designer_at_awards_main",
        desc="The celebrity wore a specific designer brand at a 2026 awards show",
        parent=supplemental,
        critical=False,
    )
    designer_sources = combine_sources(data.designer_sources, data.redcarpet_media_sources)
    designer_exists = bool(data.designer_worn_2026 and len(designer_sources) > 0)
    evaluator.add_custom_node(
        result=designer_exists,
        id="designer_at_awards_exists",
        desc="Designer name and at least one outfit-credit source are provided",
        parent=seq_designer,
        critical=False,
    )
    designer_leaf = evaluator.add_leaf(
        id="designer_at_awards_verify",
        desc="Specific designer credited for a 2026 awards appearance",
        parent=seq_designer,
        critical=False,
    )
    if data.designer_event_2026:
        designer_claim = (
            f"At the {data.designer_event_2026} in 2026, {data.celebrity_name} wore {data.designer_worn_2026}."
        )
    else:
        designer_claim = (
            f"At a 2026 major awards show appearance, {data.celebrity_name} wore {data.designer_worn_2026}."
        )
    await evaluator.verify(
        claim=designer_claim,
        node=designer_leaf,
        sources=designer_sources,
        additional_instruction="Confirm outfit credits explicitly list the designer for the 2026 event.",
    )

    # 11) Red carpet media coverage with styling credits
    seq_rc = evaluator.add_sequential(
        id="red_carpet_media_coverage_main",
        desc="The 2026 awards appearance received coverage with styling credits",
        parent=supplemental,
        critical=False,
    )
    rc_exists = bool(len(data.redcarpet_media_sources) > 0)
    evaluator.add_custom_node(
        result=rc_exists,
        id="red_carpet_media_coverage_exists",
        desc="At least one fashion/red-carpet media coverage source is provided",
        parent=seq_rc,
        critical=False,
    )
    rc_leaf = evaluator.add_leaf(
        id="red_carpet_media_coverage_verify",
        desc="Media coverage includes styling/outfit credits",
        parent=seq_rc,
        critical=False,
    )
    rc_claim = (
        f"There is fashion/red-carpet media coverage of {data.celebrity_name}'s 2026 awards appearance that includes styling/outfit credits."
    )
    await evaluator.verify(
        claim=rc_claim,
        node=rc_leaf,
        sources=data.redcarpet_media_sources,
        additional_instruction="Look for articles listing designer/stylist credits for the 2026 appearance.",
    )

    # 12) Campaign documented in multiple media sources with dated announcements
    seq_multidoc = evaluator.add_sequential(
        id="campaign_media_documentation_main",
        desc="Campaign participation documented in multiple dated media sources",
        parent=supplemental,
        critical=False,
    )
    multidoc_exists = bool(len(data.campaign_sources) >= 2)
    evaluator.add_custom_node(
        result=multidoc_exists,
        id="campaign_media_documentation_exists",
        desc="At least two distinct campaign sources are provided",
        parent=seq_multidoc,
        critical=False,
    )
    multidoc_leaf = evaluator.add_leaf(
        id="campaign_media_documentation_verify",
        desc="Multiple dated sources document the campaign participation",
        parent=seq_multidoc,
        critical=False,
    )
    multidoc_claim = (
        f"At least two distinct media sources document {data.celebrity_name}'s participation in {data.brand_name}'s Spring 2026 campaign, "
        f"with dated announcements."
    )
    await evaluator.verify(
        claim=multidoc_claim,
        node=multidoc_leaf,
        sources=data.campaign_sources,
        additional_instruction="Confirm that multiple distinct outlets document the campaign and include dates.",
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

    # Extract structured info from the agent's answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_celebrity_case(),
        template_class=CelebrityCaseExtraction,
        extraction_name="celebrity_case_extraction",
    )

    # Add contextual info for transparency
    evaluator.add_custom_info(
        info={
            "allowed_major_events": ALLOWED_MAJOR_EVENTS_2026,
            "timing_window": JFM_2026_RANGE_DESC,
            "expected_campaign_season": "Spring 2026 (a.k.a. Spring/Summer 2026, SS26)",
        },
        info_type="task_context",
        info_name="evaluation_context",
    )

    # Build verification tree and run checks
    await verify_celebrity_case(evaluator, root, extraction)

    # Return structured summary
    return evaluator.get_summary()