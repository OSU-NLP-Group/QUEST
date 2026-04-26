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
TASK_ID = "thanksgiving_week_2025_entertainment_releases"
TASK_DESCRIPTION = (
    "Identify three distinct entertainment properties (theatrical films or television series/seasons) that had a premiere, "
    "major release event, or season/series finale during Thanksgiving week 2025 (November 24-30, 2025). For each property, provide "
    "comprehensive verification including: (1) Basic Information: Official title, specific confirmed release date within the "
    "November 24-30, 2025 window, and property type; (2) Distribution Details: Primary distribution platform or theatrical distributor, "
    "and for theatrical films the subsequent streaming platform and its release date within 4 months, or for TV series the streaming/broadcast "
    "platform; (3) Industry Recognition: Significant industry recognition received in 2025 or 2026 such as Oscar nominations, major film festival "
    "awards, or equivalent television awards, including specific categories and year; (4) Specifications: For films confirm runtime of at least "
    "120 minutes, for TV series confirm at least 8 episodes OR designation as final/concluding season; (5) Creative Source: Verify the property is "
    "either based on/adapted from a published literary work or created by identified writer-producers/directors who served as showrunners/directors; "
    "(6) Personnel Verification: Identify at least one principal cast member or key creative and document their verifiable prior work in major films "
    "or TV series released between 2019-2024; (7) Performance Metrics: For theatrical releases confirm minimum $50 million worldwide box office, "
    "for streaming/TV confirm documented top-10 viewership ranking on its platform during release week. All facts must be supported with reference URLs "
    "from reliable sources."
)

THANKSGIVING_2025_START = "2025-11-24"
THANKSGIVING_2025_END = "2025-11-30"
THANKSGIVING_2025_HUMAN = "November 24–30, 2025"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BasicInfo(BaseModel):
    title: Optional[str] = None
    type: Optional[str] = None  # e.g., "theatrical film", "film", "tv series", "tv season"
    release_event: Optional[str] = None  # e.g., "premiere", "major release", "finale"
    release_date: Optional[str] = None   # Any human-readable date
    release_urls: List[str] = Field(default_factory=list)


class DistributionInfo(BaseModel):
    primary_distributor_or_platform: Optional[str] = None  # theatrical distributor OR primary platform (network/streamer)
    primary_urls: List[str] = Field(default_factory=list)


class StreamingInfo(BaseModel):
    platform: Optional[str] = None
    release_date: Optional[str] = None  # For theatrical film: streaming date; for TV: can be premiere date on platform or None
    urls: List[str] = Field(default_factory=list)


class AwardInfo(BaseModel):
    description: Optional[str] = None  # Free text summary (e.g., "Nominated for 3 Oscars")
    year: Optional[str] = None         # e.g., "2025" or "2026"
    categories: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class SpecInfo(BaseModel):
    film_runtime_minutes: Optional[str] = None  # e.g., "122 minutes"
    tv_episodes_count: Optional[str] = None     # e.g., "10"
    tv_is_final_season: Optional[bool] = None
    urls: List[str] = Field(default_factory=list)


class CreativeSourceInfo(BaseModel):
    adapted_from: Optional[str] = None  # Title or description of published literary work
    creators: List[str] = Field(default_factory=list)  # showrunner(s)/director(s)/writer-producers
    urls: List[str] = Field(default_factory=list)


class PersonnelInfo(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None  # actor/director/showrunner/etc.
    prior_work_title: Optional[str] = None
    prior_work_year: Optional[str] = None  # expected between 2019-2024
    urls: List[str] = Field(default_factory=list)


class PerformanceInfo(BaseModel):
    box_office_worldwide: Optional[str] = None  # e.g., "$120 million"; for films
    top10_viewership_platform: Optional[str] = None  # e.g., "Netflix Top 10"; for TV/streaming
    time_window: Optional[str] = None  # e.g., "release week" or a date range
    urls: List[str] = Field(default_factory=list)


class PropertyItem(BaseModel):
    basic: Optional[BasicInfo] = None
    distribution: Optional[DistributionInfo] = None
    streaming: Optional[StreamingInfo] = None
    recognition: Optional[AwardInfo] = None
    specifications: Optional[SpecInfo] = None
    creative_source: Optional[CreativeSourceInfo] = None
    personnel: Optional[PersonnelInfo] = None
    performance: Optional[PerformanceInfo] = None


class PropertiesExtraction(BaseModel):
    items: List[PropertyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_properties() -> str:
    return f"""
Extract up to five entertainment properties mentioned in the answer that the author claims had a premiere, major release, or a season/series finale during Thanksgiving week 2025 ({THANKSGIVING_2025_HUMAN}). For each property, extract as much of the following structure as is explicitly present in the answer. Include only URLs that are explicitly provided in the answer text.

Return a JSON object with a single field "items" which is an array of property objects. Each property object should include these nested objects/fields (use null when unknown; arrays may be empty):

- basic:
  - title: official title of the property
  - type: either "film" or "tv series" or "tv season" (use the closest phrasing provided)
  - release_event: "premiere", "major release", or "finale" (choose the best match if specified)
  - release_date: the specific date claimed to be within {THANKSGIVING_2025_HUMAN}
  - release_urls: array of URL strings that confirm the release event/date

- distribution:
  - primary_distributor_or_platform: theatrical distributor or main network/streaming platform
  - primary_urls: array of URL strings that confirm the primary distributor/platform

- streaming:
  - platform: for films, the streaming platform where it later became available; for TV, the platform/network of the premiere
  - release_date: for films, the streaming release date (ideally within 4 months of theatrical); for TV, this can be null
  - urls: array of URL strings that confirm the streaming/broadcast availability

- recognition:
  - description: text describing significant industry recognition (e.g., Oscar nominations, major festival awards, Emmys/Golden Globes, etc.)
  - year: year of the recognition (e.g., "2025" or "2026")
  - categories: array of specific award categories (strings)
  - urls: array of URL strings that confirm the awards/nominations

- specifications:
  - film_runtime_minutes: for films, runtime text (e.g., "122 minutes")
  - tv_episodes_count: for TV seasons/series, number of episodes (text or number as text)
  - tv_is_final_season: boolean true/false if the season is final/concluding (if clearly stated), else null
  - urls: array of URL strings that confirm runtime or episode/final-season details

- creative_source:
  - adapted_from: the published literary work if adapted
  - creators: array of creator/showrunner/director names (who served as showrunners/directors)
  - urls: array of URL strings that confirm adaptation or creators/roles

- personnel:
  - name: at least one principal cast member or key creative
  - role: their role (actor/director/showrunner/etc.)
  - prior_work_title: one prior major film/TV work credited to them
  - prior_work_year: the year of that prior work (should be between 2019–2024 if provided)
  - urls: array of URL strings that confirm the person and their prior work

- performance:
  - box_office_worldwide: for theatrical films, worldwide gross (text like "$123 million")
  - top10_viewership_platform: for streaming/TV, the name of the platform top-10 list (e.g., "Netflix Top 10")
  - time_window: text describing the time window (ideally release week)
  - urls: array of URL strings that confirm the performance metrics

IMPORTANT:
- Only include URLs explicitly present in the answer text (plain links or markdown links).
- Do not invent or infer URLs.
- Keep all date fields as strings exactly as in the answer.
- You may return fewer than five items if the answer mentions fewer. If more are present, include them all; downstream will take the first three.
    """.strip()


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def nonempty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def has_any_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip().startswith(("http://", "https://")) for u in urls)


def norm_type(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    low = t.strip().lower()
    if "film" in low or "movie" in low or low == "film":
        return "film"
    if "tv" in low or "television" in low or "series" in low or "season" in low:
        return "tv"
    return t.strip().lower()


def first_n(items: List[Any], n: int) -> List[Any]:
    return items[:n] if items else []


# --------------------------------------------------------------------------- #
# Verification logic per property                                             #
# --------------------------------------------------------------------------- #
async def verify_property(evaluator: Evaluator, parent_node, prop: PropertyItem, index_one_based: int) -> None:
    label = f"P{index_one_based}"

    # Top-level node for this property (non-critical: partial credit across properties)
    prop_node = evaluator.add_parallel(
        id=f"Property_{index_one_based}",
        desc=f"{['First','Second','Third','Fourth','Fifth'][index_one_based-1] if index_one_based<=5 else f'Property #{index_one_based}'} entertainment property meeting all specified criteria with Thanksgiving week 2025 release",
        parent=parent_node,
        critical=False,
    )

    # ------------------------ Identification ------------------------ #
    ident_node = evaluator.add_parallel(
        id=f"{label}_Identification",
        desc="Property identification and basic release information",
        parent=prop_node,
        critical=True,
    )

    title_ok = evaluator.add_custom_node(
        result=bool(prop and prop.basic and nonempty(prop.basic.title)),
        id=f"{label}_Title",
        desc="Official title of the entertainment property is provided",
        parent=ident_node,
        critical=True,
    )

    # Release date must be within the Thanksgiving 2025 window (simple logical check)
    rel_date_leaf = evaluator.add_leaf(
        id=f"{label}_Release_Date",
        desc=f"Specific confirmed release date during {THANKSGIVING_2025_HUMAN} is provided",
        parent=ident_node,
        critical=True,
    )
    date_str = prop.basic.release_date if (prop and prop.basic) else None
    await evaluator.verify(
        claim=f"The date '{date_str}' falls within the inclusive window {THANKSGIVING_2025_HUMAN} (i.e., between {THANKSGIVING_2025_START} and {THANKSGIVING_2025_END}).",
        node=rel_date_leaf,
        additional_instruction="Only check date range membership logically; do not require external sources for this check.",
    )

    # URL reference confirming the release date/event
    rel_ref_leaf = evaluator.add_leaf(
        id=f"{label}_Release_Date_Reference",
        desc="URL reference confirming the Thanksgiving week 2025 release date",
        parent=ident_node,
        critical=True,
    )
    title = prop.basic.title if (prop and prop.basic) else None
    event_kind = prop.basic.release_event if (prop and prop.basic) else None
    await evaluator.verify(
        claim=f"The property '{title}' had a {event_kind or 'premiere/major release/finale'} on {date_str}, which is during {THANKSGIVING_2025_HUMAN}.",
        node=rel_ref_leaf,
        sources=(prop.basic.release_urls if (prop and prop.basic) else None),
        additional_instruction="Verify that the cited page(s) explicitly mention this title and the stated date/event falling within the Thanksgiving 2025 week. Accept synonyms for event type.",
    )

    # Property type identified (film or TV)
    ptype_ok = evaluator.add_custom_node(
        result=bool(prop and prop.basic and nonempty(prop.basic.type)),
        id=f"{label}_Type",
        desc="Property type (theatrical film or TV series/season) is clearly identified",
        parent=ident_node,
        critical=True,
    )

    # ------------------------ Distribution ------------------------- #
    dist_node = evaluator.add_parallel(
        id=f"{label}_Distribution",
        desc="Distribution and platform information",
        parent=prop_node,
        critical=True,
    )

    primary_platform_leaf = evaluator.add_leaf(
        id=f"{label}_Primary_Platform",
        desc="Primary distribution platform or theatrical distributor is identified",
        parent=dist_node,
        critical=True,
    )
    primary_name = prop.distribution.primary_distributor_or_platform if prop and prop.distribution else None
    await evaluator.verify(
        claim=f"The primary distribution platform or theatrical distributor for '{title}' is '{primary_name}'.",
        node=primary_platform_leaf,
        sources=(prop.distribution.primary_urls if (prop and prop.distribution) else None),
        additional_instruction="Confirm on the referenced page(s) the main theatrical distributor (for films) or the primary network/streaming platform (for TV).",
    )

    primary_platform_ref = evaluator.add_custom_node(
        result=bool(prop and prop.distribution and has_any_urls(prop.distribution.primary_urls)),
        id=f"{label}_Primary_Platform_Reference",
        desc="URL reference confirming the primary distribution platform",
        parent=dist_node,
        critical=True,
    )

    streaming_info_leaf = evaluator.add_leaf(
        id=f"{label}_Streaming_Info",
        desc="For films: subsequent streaming platform and release date within 4 months is provided. For TV series: streaming/broadcast platform where the season premiered is confirmed.",
        parent=dist_node,
        critical=True,
    )
    ptype = norm_type(prop.basic.type) if (prop and prop.basic) else None
    stream_platform = prop.streaming.platform if (prop and prop.streaming) else None
    stream_date = prop.streaming.release_date if (prop and prop.streaming) else None
    # Build claim adjusted by type
    if ptype == "film":
        streaming_claim = (
            f"For the film '{title}', it later became available on streaming platform '{stream_platform}' on {stream_date}, "
            f"and that streaming date is within 4 months after the theatrical release date {date_str}."
        )
        streaming_ai = (
            "Verify the stated streaming platform and date on the cited page(s). Also check that the streaming date occurs within approximately "
            "4 months (≈120 days) after the theatrical release date; if not clearly within 4 months, consider this incorrect."
        )
    else:
        streaming_claim = (
            f"For the TV property '{title}', the season/premiere aired on or was available on the platform/network '{stream_platform}'."
        )
        streaming_ai = (
            "Verify the platform/network where the season premiered or was released, as stated on the cited page(s). A specific streaming date is "
            "not required for TV; confirming the correct platform/network suffices."
        )
    await evaluator.verify(
        claim=streaming_claim,
        node=streaming_info_leaf,
        sources=(prop.streaming.urls if (prop and prop.streaming) else None),
        additional_instruction=streaming_ai,
    )

    streaming_ref = evaluator.add_custom_node(
        result=bool(prop and prop.streaming and has_any_urls(prop.streaming.urls)),
        id=f"{label}_Streaming_Reference",
        desc="URL reference confirming streaming availability details",
        parent=dist_node,
        critical=True,
    )

    # ------------------------ Recognition -------------------------- #
    recog_node = evaluator.add_parallel(
        id=f"{label}_Recognition",
        desc="Industry recognition and awards information",
        parent=prop_node,
        critical=True,
    )

    award_info_leaf = evaluator.add_leaf(
        id=f"{label}_Award_Info",
        desc="Significant industry recognition in 2025 or 2026 is documented",
        parent=recog_node,
        critical=True,
    )
    award_desc = prop.recognition.description if (prop and prop.recognition) else None
    award_year = prop.recognition.year if (prop and prop.recognition) else None
    award_cats = prop.recognition.categories if (prop and prop.recognition) else []
    await evaluator.verify(
        claim=f"The property '{title}' received significant industry recognition in {award_year}, such as {award_desc}, including categories {award_cats}.",
        node=award_info_leaf,
        sources=(prop.recognition.urls if (prop and prop.recognition) else None),
        additional_instruction="Recognition includes major film festivals, Academy Awards nominations/wins, or equivalent TV awards. Confirm that the year is stated on the source page(s).",
    )

    award_cat_leaf = evaluator.add_leaf(
        id=f"{label}_Award_Category",
        desc="Specific award categories and year are provided",
        parent=recog_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The listed award year '{award_year}' is either 2025 or 2026 and at least one specific award category is provided: {award_cats}.",
        node=award_cat_leaf,
        additional_instruction="Purely logical check on provided fields: year must be 2025 or 2026 and categories must be a non-empty list.",
    )

    award_ref = evaluator.add_custom_node(
        result=bool(prop and prop.recognition and has_any_urls(prop.recognition.urls)),
        id=f"{label}_Award_Reference",
        desc="URL reference confirming the award nominations or wins",
        parent=recog_node,
        critical=True,
    )

    # ------------------------ Specifications ----------------------- #
    spec_node = evaluator.add_parallel(
        id=f"{label}_Specifications",
        desc="Runtime or episode specifications",
        parent=prop_node,
        critical=True,
    )

    duration_leaf = evaluator.add_leaf(
        id=f"{label}_Duration_Episodes",
        desc="For films: runtime ≥ 120 minutes. For TV: at least 8 episodes OR final/concluding season.",
        parent=spec_node,
        critical=True,
    )
    film_runtime = prop.specifications.film_runtime_minutes if (prop and prop.specifications) else None
    tv_eps = prop.specifications.tv_episodes_count if (prop and prop.specifications) else None
    tv_final = prop.specifications.tv_is_final_season if (prop and prop.specifications) else None

    if ptype == "film":
        dur_claim = f"The film '{title}' has a runtime of at least 120 minutes (reported as '{film_runtime}')."
        dur_ai = "Use the cited page(s) to confirm the runtime is 120 minutes or longer."
    else:
        if nonempty(tv_eps):
            dur_claim = f"The TV property '{title}' has at least 8 episodes (reported count: '{tv_eps}')."
            dur_ai = "Use the cited page(s) to confirm the episode count is 8 or more."
        elif tv_final is True:
            dur_claim = f"The TV property '{title}' is designated as a final or concluding season."
            dur_ai = "Use the cited page(s) to confirm that the season is the final/concluding season."
        else:
            dur_claim = f"The TV property '{title}' meets the specification of at least 8 episodes or being a final/concluding season."
            dur_ai = "Confirm on the cited page(s) that either the season has ≥8 episodes or it is explicitly marked as final/concluding."
    await evaluator.verify(
        claim=dur_claim,
        node=duration_leaf,
        sources=(prop.specifications.urls if (prop and prop.specifications) else None),
        additional_instruction=dur_ai,
    )

    duration_ref = evaluator.add_custom_node(
        result=bool(prop and prop.specifications and has_any_urls(prop.specifications.urls)),
        id=f"{label}_Duration_Reference",
        desc="URL reference confirming runtime or episode count specifications",
        parent=spec_node,
        critical=True,
    )

    # ------------------------ Creative Source ---------------------- #
    src_node = evaluator.add_parallel(
        id=f"{label}_Source_Creative",
        desc="Literary source or creative team information",
        parent=prop_node,
        critical=True,
    )

    src_leaf = evaluator.add_leaf(
        id=f"{label}_Literary_Source_OR_Creators",
        desc="Property is adapted from a published literary work OR created by identified writer-producers/directors who served as showrunners/directors",
        parent=src_node,
        critical=True,
    )
    adapted_from = prop.creative_source.adapted_from if (prop and prop.creative_source) else None
    creators = prop.creative_source.creators if (prop and prop.creative_source) else []
    if nonempty(adapted_from):
        src_claim = f"The property '{title}' is adapted from the published literary work '{adapted_from}'."
    else:
        src_claim = f"The property '{title}' was created by {creators}, who served as showrunners and/or directors."
    await evaluator.verify(
        claim=src_claim,
        node=src_leaf,
        sources=(prop.creative_source.urls if (prop and prop.creative_source) else None),
        additional_instruction="Verify adaptation from a published literary work if provided; otherwise confirm that the named creator(s) served in showrunner/director capacities.",
    )

    src_ref = evaluator.add_custom_node(
        result=bool(prop and prop.creative_source and has_any_urls(prop.creative_source.urls)),
        id=f"{label}_Source_Creative_Reference",
        desc="URL reference confirming literary source or creative team details",
        parent=src_node,
        critical=True,
    )

    # ------------------------ Personnel ---------------------------- #
    ppl_node = evaluator.add_parallel(
        id=f"{label}_Personnel",
        desc="Principal cast or key creative personnel verification",
        parent=prop_node,
        critical=True,
    )

    key_person_ok = evaluator.add_custom_node(
        result=bool(prop and prop.personnel and nonempty(prop.personnel.name)),
        id=f"{label}_Key_Personnel",
        desc="At least one principal cast member or key creative (director/creator) is identified",
        parent=ppl_node,
        critical=True,
    )

    prior_leaf = evaluator.add_leaf(
        id=f"{label}_Prior_Work",
        desc="The identified person's verifiable prior work in 2019–2024 is documented",
        parent=ppl_node,
        critical=True,
    )
    person = prop.personnel.name if (prop and prop.personnel) else None
    prior_title = prop.personnel.prior_work_title if (prop and prop.personnel) else None
    prior_year = prop.personnel.prior_work_year if (prop and prop.personnel) else None
    await evaluator.verify(
        claim=f"{person} previously worked on '{prior_title}' released in {prior_year}, and this prior work falls within 2019–2024.",
        node=prior_leaf,
        sources=(prop.personnel.urls if (prop and prop.personnel) else None),
        additional_instruction="Confirm on the cited source(s) both the person's credit and the release year; accept if the year is clearly between 2019 and 2024 inclusive.",
    )

    ppl_ref = evaluator.add_custom_node(
        result=bool(prop and prop.personnel and has_any_urls(prop.personnel.urls)),
        id=f"{label}_Personnel_Reference",
        desc="URL reference confirming personnel and their prior work",
        parent=ppl_node,
        critical=True,
    )

    # ------------------------ Performance -------------------------- #
    perf_node = evaluator.add_parallel(
        id=f"{label}_Performance",
        desc="Box office or viewership performance metrics",
        parent=prop_node,
        critical=True,
    )

    perf_leaf = evaluator.add_leaf(
        id=f"{label}_Performance_Metric",
        desc="For films: minimum $50M worldwide box office. For streaming/TV: documented top-10 viewership during release week.",
        parent=perf_node,
        critical=True,
    )
    box_office = prop.performance.box_office_worldwide if (prop and prop.performance) else None
    top10_platform = prop.performance.top10_viewership_platform if (prop and prop.performance) else None
    time_window = prop.performance.time_window if (prop and prop.performance) else None
    if ptype == "film":
        perf_claim = f"The film '{title}' earned at least $50 million in worldwide box office (reported as '{box_office}')."
        perf_ai = "Use reliable sources (e.g., Box Office Mojo, The Numbers, trade press) to confirm that worldwide gross meets or exceeds $50,000,000."
    else:
        perf_claim = f"The title '{title}' ranked in the '{top10_platform}' top-10 during its release week around {THANKSGIVING_2025_HUMAN} (time window: '{time_window}')."
        perf_ai = "Confirm that the property appeared in the platform's top-10 list during the release week; accept reasonable date-window phrasing that clearly aligns with Thanksgiving week 2025."
    await evaluator.verify(
        claim=perf_claim,
        node=perf_leaf,
        sources=(prop.performance.urls if (prop and prop.performance) else None),
        additional_instruction=perf_ai,
    )

    perf_ref = evaluator.add_custom_node(
        result=bool(prop and prop.performance and has_any_urls(prop.performance.urls)),
        id=f"{label}_Performance_Reference",
        desc="URL reference confirming box office or viewership performance",
        parent=perf_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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

    # Record task/window info
    evaluator.add_custom_info(
        info={
            "thanksgiving_week_2025_window": {
                "human": THANKSGIVING_2025_HUMAN,
                "start_iso": THANKSGIVING_2025_START,
                "end_iso": THANKSGIVING_2025_END,
            }
        },
        info_type="context",
        info_name="task_window_info",
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=PropertiesExtraction,
        extraction_name="properties_extraction",
    )

    # Use only the first three properties; pad if fewer
    props = first_n(extracted.items, 3)
    while len(props) < 3:
        props.append(PropertyItem())

    # Build property verification subtrees
    for i, prop in enumerate(props, start=1):
        await verify_property(evaluator, root, prop, i)

    return evaluator.get_summary()