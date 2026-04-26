import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "music_video_2026_verification"
TASK_DESCRIPTION = (
    "A music video was released in January 2026 as the lead single from an album that came out on March 6, 2026. "
    "The video's director was born and raised in a small French town called Saint-Malo. This director won the UK Music "
    "Video Awards (UKMVA) Best Director award in both 2023 and 2024—making it two consecutive years. The same director "
    "also won UKMVA Best New Director back in 2021 and had previously collaborated with this same artist on a 2022 music video "
    'called "Music for a Sushi Restaurant." The production company for this 2026 video was DIVISION, a global creative '
    "production company. The Director of Photography was Chris Ripley, and the editor was Gwen Ghelid. The artist is currently "
    'on a 2026 tour called "Together, Together" which includes a major 30-date residency at Madison Square Garden running from '
    "August 26 through October 31, 2026, with Jamie xx as the special guest. The tour also includes 10 shows at Johan Cruijff ArenA "
    "in Amsterdam from May 16 through June 5, 2026, with Robyn as the special guest for those dates. Provide the title of the music video, "
    "the artist's name, the director's full name, and verify director birthplace and consecutive UKMVA Best Director wins (2023-2024). "
    "Confirm the production company (DIVISION), the Director of Photography (Chris Ripley), and the editor (Gwen Ghelid). "
    "Additionally, provide the tour name, complete details of the Madison Square Garden residency (dates, number of shows, special guest, "
    "and approximate venue capacity), and the Amsterdam shows (venue name, dates, number of shows, special guest, and venue capacity range). "
    "Include reference URLs for all key information."
)

EXPECTED_CONSTRAINTS = {
    "video_release_month_year": "January 2026",
    "album_release_date": "March 6, 2026",
    "tour_name": "Together, Together",
    "msg_residency_dates": "August 26 through October 31, 2026",
    "msg_show_count": "30",
    "msg_special_guest": "Jamie xx",
    "amsterdam_venue": "Johan Cruijff ArenA",
    "amsterdam_dates": "May 16 through June 5, 2026",
    "amsterdam_show_count": "10",
    "amsterdam_special_guest": "Robyn",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VideoExtraction(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None
    release_timing_text: Optional[str] = None  # e.g., "January 2026"
    lead_single_album_date_text: Optional[str] = None  # e.g., "lead single" + "March 6, 2026"


class DirectorExtraction(BaseModel):
    full_name: Optional[str] = None
    birthplace_text: Optional[str] = None  # e.g., "Saint-Malo, France"
    ukmva_best_director_years: List[str] = Field(default_factory=list)  # Expecting ["2023", "2024"]
    ukmva_best_new_director_year: Optional[str] = None  # Expecting "2021"
    previous_collab_title: Optional[str] = None  # e.g., "Music for a Sushi Restaurant"
    previous_collab_year: Optional[str] = None  # e.g., "2022"


class ProductionExtraction(BaseModel):
    production_company: Optional[str] = None  # Expect "DIVISION"
    dop: Optional[str] = None  # Expect "Chris Ripley"
    editor: Optional[str] = None  # Expect "Gwen Ghelid"


class TourMSGExtraction(BaseModel):
    dates_range_text: Optional[str] = None  # e.g., "August 26 through October 31, 2026"
    show_count_text: Optional[str] = None  # e.g., "30"
    special_guest: Optional[str] = None  # e.g., "Jamie xx"
    capacity_approx_text: Optional[str] = None  # e.g., "about 19,500"


class TourAMSExtraction(BaseModel):
    venue: Optional[str] = None  # e.g., "Johan Cruijff ArenA"
    dates_range_text: Optional[str] = None  # e.g., "May 16 through June 5, 2026"
    show_count_text: Optional[str] = None  # e.g., "10"
    special_guest: Optional[str] = None  # e.g., "Robyn"
    capacity_range_text: Optional[str] = None  # e.g., "55,000–68,000"


class TourExtraction(BaseModel):
    tour_name: Optional[str] = None  # e.g., "Together, Together"
    msg: Optional[TourMSGExtraction] = None
    ams: Optional[TourAMSExtraction] = None


class ReferencesExtraction(BaseModel):
    video_album_context: List[str] = Field(default_factory=list)
    director_identity_birthplace: List[str] = Field(default_factory=list)
    ukmva_awards: List[str] = Field(default_factory=list)
    previous_collab: List[str] = Field(default_factory=list)
    production_credits: List[str] = Field(default_factory=list)
    tour_and_stops: List[str] = Field(default_factory=list)
    capacities: List[str] = Field(default_factory=list)


class FullExtraction(BaseModel):
    video: Optional[VideoExtraction] = None
    director: Optional[DirectorExtraction] = None
    production: Optional[ProductionExtraction] = None
    tour: Optional[TourExtraction] = None
    refs: Optional[ReferencesExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the following information exactly as it appears in the provided answer text. Do not invent or infer any data.

1) video:
- title: music video title
- artist: artist name
- release_timing_text: how the answer describes the video release timing (e.g., "January 2026")
- lead_single_album_date_text: how the answer describes the lead-single + album release date context (e.g., "lead single" and "March 6, 2026")

2) director:
- full_name: director's full name
- birthplace_text: how the answer states the birthplace/upbringing (e.g., "born and raised in Saint-Malo, France")
- ukmva_best_director_years: list of years explicitly mentioned for UKMVA Best Director wins (e.g., ["2023","2024"])
- ukmva_best_new_director_year: year explicitly mentioned for UKMVA Best New Director (e.g., "2021")
- previous_collab_title: title of the prior 2022 collaboration video (e.g., "Music for a Sushi Restaurant")
- previous_collab_year: year explicitly mentioned for that collaboration (e.g., "2022")

3) production:
- production_company: name of the production company (e.g., "DIVISION")
- dop: name of the Director of Photography (e.g., "Chris Ripley")
- editor: name of the editor (e.g., "Gwen Ghelid")

4) tour:
- tour_name: name of the current 2026 tour (e.g., "Together, Together")
- msg:
  - dates_range_text: the complete date range for Madison Square Garden residency (e.g., "August 26 through October 31, 2026")
  - show_count_text: show count or dates count for MSG (e.g., "30")
  - special_guest: special guest name for MSG (e.g., "Jamie xx")
  - capacity_approx_text: approximate MSG concert capacity text provided in the answer (e.g., "about 19,500")
- ams:
  - venue: Amsterdam venue name (e.g., "Johan Cruijff ArenA")
  - dates_range_text: the complete date range for Amsterdam run (e.g., "May 16 through June 5, 2026")
  - show_count_text: number of shows in Amsterdam (e.g., "10")
  - special_guest: special guest for Amsterdam (e.g., "Robyn")
  - capacity_range_text: capacity range text (e.g., "55,000–68,000")

5) refs:
Provide only URLs that appear in the answer text. Do not invent URLs. Extract them into these arrays:
- video_album_context: URLs that support video release timing and the lead-single/album release date
- director_identity_birthplace: URLs that support the director's identity (full name) and birthplace/upbringing (Saint-Malo, France)
- ukmva_awards: URLs that support UKMVA awards (Best Director 2023 and 2024, Best New Director 2021)
- previous_collab: URLs that support the 2022 "Music for a Sushi Restaurant" collaboration between the same director and artist
- production_credits: URLs that support the production company (DIVISION), DOP (Chris Ripley), and editor (Gwen Ghelid)
- tour_and_stops: URLs that support the tour name and the MSG/Amsterdam details (venues/dates/counts/guests)
- capacities: URLs that support the MSG approximate concert capacity and the Johan Cruijff ArenA capacity range

If any field is missing, set it to null (or an empty list for URL arrays).
Return a single JSON object with keys: video, director, production, tour, refs, following the exact structure.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _list_non_empty(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_music_video_nodes(evaluator: Evaluator, parent, data: FullExtraction):
    mv_node = evaluator.add_parallel(
        id="music_video",
        desc="Music video and artist are correctly identified and aligned with the given release/album context.",
        parent=parent,
        critical=True,
    )

    # Existence: Music video title provided
    title_exists = _non_empty(getattr(data.video, "title", None) if data.video else None)
    evaluator.add_custom_node(
        result=title_exists,
        id="video_title",
        desc="Music video title is provided.",
        parent=mv_node,
        critical=True,
    )

    # Existence: Artist name provided
    artist_exists = _non_empty(getattr(data.video, "artist", None) if data.video else None)
    evaluator.add_custom_node(
        result=artist_exists,
        id="artist_name",
        desc="Artist name is provided.",
        parent=mv_node,
        critical=True,
    )

    # Verify: Video release timing = January 2026
    release_leaf = evaluator.add_leaf(
        id="release_timing",
        desc="Video release timing is stated as January 2026.",
        parent=mv_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The music video was released in January 2026.",
        node=release_leaf,
        sources=(data.refs.video_album_context if data and data.refs else None),
        additional_instruction=(
            "Confirm the page indicates the music video release occurred in January 2026. "
            "Allow equivalent expressions like 'released on Jan XX, 2026' or 'January, 2026'."
        ),
    )

    # Verify: Lead single + album release date March 6, 2026
    lead_album_leaf = evaluator.add_leaf(
        id="lead_single_and_album_date",
        desc="States that the video/song is the lead single from an album released on March 6, 2026.",
        parent=mv_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This song/video is the lead single from an album released on March 6, 2026.",
        node=lead_album_leaf,
        sources=(data.refs.video_album_context if data and data.refs else None),
        additional_instruction=(
            "The evidence should explicitly indicate the track is the lead (or first) single and that the album released on March 6, 2026. "
            "Accept close wording such as 'first single' or date formats like '6 March 2026'."
        ),
    )


async def build_director_nodes(evaluator: Evaluator, parent, data: FullExtraction):
    dir_node = evaluator.add_parallel(
        id="director",
        desc="Director identity and requested verification points are provided.",
        parent=parent,
        critical=True,
    )

    # Existence: Director full name is provided
    dir_name = _safe(getattr(data.director, "full_name", None) if data.director else None)
    evaluator.add_custom_node(
        result=_non_empty(dir_name),
        id="director_full_name",
        desc="Director's full name is provided.",
        parent=dir_node,
        critical=True,
    )

    # Verify birthplace/upbringing Saint-Malo, France
    birthplace_leaf = evaluator.add_leaf(
        id="director_birthplace",
        desc="Verifies the director was born and raised in Saint-Malo, France.",
        parent=dir_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The director {_safe(dir_name)} was born and raised in Saint-Malo, France.",
        node=birthplace_leaf,
        sources=(data.refs.director_identity_birthplace if data and data.refs else None),
        additional_instruction=(
            "Look for mentions of 'Saint-Malo' (allow 'Saint Malo' without hyphen) and that it is in France. "
            "If the source states birthplace and/or upbringing in Saint-Malo, that satisfies the claim."
        ),
    )

    # Verify UKMVA Best Director consecutive wins 2023 and 2024
    ukmva_consec_leaf = evaluator.add_leaf(
        id="director_ukmva_best_director_consecutive",
        desc="Verifies the director won UKMVA Best Director in both 2023 and 2024 (consecutive years).",
        parent=dir_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The director {_safe(dir_name)} won the UK Music Video Awards Best Director in 2023 and 2024.",
        node=ukmva_consec_leaf,
        sources=(data.refs.ukmva_awards if data and data.refs else None),
        additional_instruction=(
            "The page(s) should explicitly show UKMVA Best Director wins for both 2023 and 2024 for this director."
        ),
    )

    # Verify UKMVA Best New Director 2021
    ukmva_new_dir_leaf = evaluator.add_leaf(
        id="director_ukmva_best_new_director_2021",
        desc="Verifies the director won UKMVA Best New Director in 2021.",
        parent=dir_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The director {_safe(dir_name)} won the UKMVA Best New Director in 2021.",
        node=ukmva_new_dir_leaf,
        sources=(data.refs.ukmva_awards if data and data.refs else None),
        additional_instruction="Accept 'Best New Director' (or equivalent phrasing) and confirm the year is 2021.",
    )

    # Verify previous collaboration in 2022 for 'Music for a Sushi Restaurant' with same artist
    artist_name = _safe(getattr(data.video, "artist", None) if data.video else None)
    prev_collab_leaf = evaluator.add_leaf(
        id="director_previous_collaboration_2022",
        desc='Verifies the director previously collaborated with the same artist by directing the 2022 music video "Music for a Sushi Restaurant."',
        parent=dir_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"In 2022, the director {_safe(dir_name)} directed the music video 'Music for a Sushi Restaurant' for the same artist "
            f"{artist_name}."
        ),
        node=prev_collab_leaf,
        sources=(data.refs.previous_collab if data and data.refs else None),
        additional_instruction=(
            "Confirm that the director is credited for the 2022 'Music for a Sushi Restaurant' video and that the artist matches the artist of the 2026 video."
        ),
    )


async def build_production_nodes(evaluator: Evaluator, parent, data: FullExtraction):
    prod_node = evaluator.add_parallel(
        id="production",
        desc="Production company and key crew credits are confirmed as stated in the question.",
        parent=parent,
        critical=True,
    )

    # Production company DIVISION
    prod_company_leaf = evaluator.add_leaf(
        id="production_company",
        desc="Production company is confirmed as DIVISION.",
        parent=prod_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The production company for the 2026 music video is DIVISION.",
        node=prod_company_leaf,
        sources=(data.refs.production_credits if data and data.refs else None),
        additional_instruction=(
            "Look for production credits explicitly naming DIVISION. Accept variations like 'Division' or 'DIVISION Paris'."
        ),
    )

    # DOP Chris Ripley
    dop_leaf = evaluator.add_leaf(
        id="dop",
        desc="Director of Photography is confirmed as Chris Ripley.",
        parent=prod_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Director of Photography (cinematographer) for the 2026 music video is Chris Ripley.",
        node=dop_leaf,
        sources=(data.refs.production_credits if data and data.refs else None),
        additional_instruction="Accept 'DoP', 'DP', or 'Cinematographer' as equivalent credit labels.",
    )

    # Editor Gwen Ghelid
    editor_leaf = evaluator.add_leaf(
        id="editor",
        desc="Editor is confirmed as Gwen Ghelid.",
        parent=prod_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The editor for the 2026 music video is Gwen Ghelid.",
        node=editor_leaf,
        sources=(data.refs.production_credits if data and data.refs else None),
        additional_instruction="Accept phrasing like 'Editing by' or 'Edited by'.",
    )


async def build_tour_nodes(evaluator: Evaluator, parent, data: FullExtraction):
    tour_node = evaluator.add_parallel(
        id="tour",
        desc="Tour name and both specified tour-stop detail sets are provided (including capacities as requested).",
        parent=parent,
        critical=True,
    )

    # Tour name verification
    tour_name_leaf = evaluator.add_leaf(
        id="tour_name",
        desc='Tour name is provided as "Together, Together."',
        parent=tour_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The artist is on a 2026 tour called 'Together, Together'.",
        node=tour_name_leaf,
        sources=(data.refs.tour_and_stops if data and data.refs else None),
        additional_instruction="Accept minor punctuation/casing variations; ensure the name refers to the 2026 tour.",
    )

    # MSG Residency subtree
    msg_node = evaluator.add_parallel(
        id="msg_residency",
        desc="Madison Square Garden residency details are complete.",
        parent=tour_node,
        critical=True,
    )

    msg_dates_leaf = evaluator.add_leaf(
        id="msg_dates",
        desc="MSG residency dates are provided as Aug 26 through Oct 31, 2026.",
        parent=msg_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Madison Square Garden residency runs from August 26 through October 31, 2026.",
        node=msg_dates_leaf,
        sources=(data.refs.tour_and_stops if data and data.refs else None),
        additional_instruction="Accept equivalent date formats or en dash ranges (e.g., 'Aug 26–Oct 31, 2026').",
    )

    msg_count_leaf = evaluator.add_leaf(
        id="msg_show_count",
        desc="MSG residency show count is provided as 30 dates.",
        parent=msg_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Madison Square Garden residency consists of 30 dates.",
        node=msg_count_leaf,
        sources=(data.refs.tour_and_stops if data and data.refs else None),
        additional_instruction="Accept synonyms like 'nights' or 'shows' when the number equals 30.",
    )

    msg_guest_leaf = evaluator.add_leaf(
        id="msg_special_guest",
        desc="MSG residency special guest is provided as Jamie xx.",
        parent=msg_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The special guest for the Madison Square Garden residency is Jamie xx.",
        node=msg_guest_leaf,
        sources=(data.refs.tour_and_stops if data and data.refs else None),
        additional_instruction="Confirm Jamie xx is listed specifically for the MSG residency dates.",
    )

    msg_capacity_leaf = evaluator.add_leaf(
        id="msg_capacity",
        desc="Approximate Madison Square Garden concert capacity is provided (a numeric estimate, described as approximate).",
        parent=msg_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The typical concert capacity of Madison Square Garden is approximately {_safe(getattr(getattr(data, 'tour', None), 'msg', None).capacity_approx_text if data and data.tour and data.tour.msg else None)}.",
        node=msg_capacity_leaf,
        sources=(data.refs.capacities if data and data.refs else None),
        additional_instruction=(
            "Verify the typical concert seating capacity figure. Allow approximations and small variation (e.g., 18,000–20,000). "
            "Focus on whether the webpage supports an approximately similar number to the one stated."
        ),
    )

    # Amsterdam Shows subtree
    ams_node = evaluator.add_parallel(
        id="amsterdam_shows",
        desc="Amsterdam shows details are complete.",
        parent=tour_node,
        critical=True,
    )

    ams_venue_leaf = evaluator.add_leaf(
        id="amsterdam_venue",
        desc="Amsterdam venue is provided as Johan Cruijff ArenA.",
        parent=ams_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Amsterdam venue is Johan Cruijff ArenA.",
        node=ams_venue_leaf,
        sources=(data.refs.tour_and_stops if data and data.refs else None),
        additional_instruction="Ensure the venue is in Amsterdam and named Johan Cruijff ArenA.",
    )

    ams_dates_leaf = evaluator.add_leaf(
        id="amsterdam_dates",
        desc="Amsterdam dates are provided as May 16 through Jun 5, 2026.",
        parent=ams_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Amsterdam shows run from May 16 through June 5, 2026.",
        node=ams_dates_leaf,
        sources=(data.refs.tour_and_stops if data and data.refs else None),
        additional_instruction="Accept alternative month spelling or date separators (e.g., 'May 16–June 5, 2026').",
    )

    ams_count_leaf = evaluator.add_leaf(
        id="amsterdam_show_count",
        desc="Amsterdam show count is provided as 10 shows.",
        parent=ams_node,
        critical=True,
    )
    await evaluator.verify(
        claim="There are 10 shows scheduled in Amsterdam for this run.",
        node=ams_count_leaf,
        sources=(data.refs.tour_and_stops if data and data.refs else None),
        additional_instruction="Accept '10 dates' or '10 nights' as equivalent.",
    )

    ams_guest_leaf = evaluator.add_leaf(
        id="amsterdam_special_guest",
        desc="Amsterdam special guest is provided as Robyn.",
        parent=ams_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The special guest for the Amsterdam dates is Robyn.",
        node=ams_guest_leaf,
        sources=(data.refs.tour_and_stops if data and data.refs else None),
        additional_instruction="Ensure Robyn is associated with the Amsterdam dates at Johan Cruijff ArenA.",
    )

    ams_capacity_leaf = evaluator.add_leaf(
        id="amsterdam_capacity",
        desc="Johan Cruijff ArenA venue capacity range is provided (a numeric range, described as configuration-dependent).",
        parent=ams_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Johan Cruijff ArenA has a concert capacity range of {_safe(getattr(getattr(data, 'tour', None), 'ams', None).capacity_range_text if data and data.tour and data.tour.ams else None)}.",
        node=ams_capacity_leaf,
        sources=(data.refs.capacities if data and data.refs else None),
        additional_instruction=(
            "Verify that the venue capacity is presented as a range or multiple figures depending on configuration. "
            "Accept approximate ranges; ensure consistency with the numbers stated on the webpage."
        ),
    )


def build_references_nodes(evaluator: Evaluator, parent, data: FullExtraction):
    refs_node = evaluator.add_parallel(
        id="references",
        desc="Reference URLs are provided for all key information (video/album context, director, awards, production credits, tour details, and capacities).",
        parent=parent,
        critical=True,
    )

    # Each of these is a critical existence check for at least one URL in the category
    evaluator.add_custom_node(
        result=_list_non_empty(data.refs.video_album_context if data and data.refs else None),
        id="ref_video_and_album_context",
        desc="At least one reference URL supports the video release timing (January 2026) and the lead-single/album release date context (album released March 6, 2026).",
        parent=refs_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_list_non_empty(data.refs.director_identity_birthplace if data and data.refs else None),
        id="ref_director_identity_and_birthplace",
        desc="At least one reference URL supports the director's identity (full name) and being born/raised in Saint-Malo, France.",
        parent=refs_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_list_non_empty(data.refs.ukmva_awards if data and data.refs else None),
        id="ref_ukmva_awards",
        desc="At least one reference URL supports the director’s UKMVA wins: Best Director (2023 and 2024) and Best New Director (2021).",
        parent=refs_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_list_non_empty(data.refs.previous_collab if data and data.refs else None),
        id="ref_previous_collaboration",
        desc='At least one reference URL supports the director’s 2022 "Music for a Sushi Restaurant" collaboration with the same artist.',
        parent=refs_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_list_non_empty(data.refs.production_credits if data and data.refs else None),
        id="ref_production_credits",
        desc="At least one reference URL supports the production company (DIVISION), DOP (Chris Ripley), and editor (Gwen Ghelid) credits.",
        parent=refs_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_list_non_empty(data.refs.tour_and_stops if data and data.refs else None),
        id="ref_tour_and_stops",
        desc='At least one reference URL supports the tour name ("Together, Together"), MSG residency details, and Amsterdam show details (venues/dates/counts/guests).',
        parent=refs_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_list_non_empty(data.refs.capacities if data and data.refs else None),
        id="ref_capacities",
        desc="At least one reference URL supports the stated MSG concert capacity estimate and the Johan Cruijff ArenA capacity range.",
        parent=refs_node,
        critical=True,
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
    # Initialize evaluator with a critical root (all children must pass)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Provide the music video title and artist, plus the requested director, production, tour details, and reference URLs supporting key claims.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )
    # Make root critical by design requirement: wrap a single critical parallel child or
    # directly make first-level nodes critical. We will add first-level nodes as critical.

    # Record constraints as ground truth context (for transparency, not strict scoring)
    evaluator.add_ground_truth(
        {
            "expected_release_timing": EXPECTED_CONSTRAINTS["video_release_month_year"],
            "expected_album_release_date": EXPECTED_CONSTRAINTS["album_release_date"],
            "expected_tour_name": EXPECTED_CONSTRAINTS["tour_name"],
            "expected_msg": {
                "dates": EXPECTED_CONSTRAINTS["msg_residency_dates"],
                "show_count": EXPECTED_CONSTRAINTS["msg_show_count"],
                "special_guest": EXPECTED_CONSTRAINTS["msg_special_guest"],
            },
            "expected_amsterdam": {
                "venue": EXPECTED_CONSTRAINTS["amsterdam_venue"],
                "dates": EXPECTED_CONSTRAINTS["amsterdam_dates"],
                "show_count": EXPECTED_CONSTRAINTS["amsterdam_show_count"],
                "special_guest": EXPECTED_CONSTRAINTS["amsterdam_special_guest"],
            },
        },
        gt_type="expected_constraints",
    )

    # Extract all structured information from the answer
    extracted: FullExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=FullExtraction,
        extraction_name="extracted_answer_info",
    )

    # Build the tree: First-level groups under root must be critical parallel nodes
    # We'll attach all verification subtrees directly under the root.

    # music_video subtree
    await build_music_video_nodes(evaluator, root, extracted)

    # director subtree
    await build_director_nodes(evaluator, root, extracted)

    # production subtree
    await build_production_nodes(evaluator, root, extracted)

    # tour subtree
    await build_tour_nodes(evaluator, root, extracted)

    # references subtree (existence checks for URL groups)
    build_references_nodes(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()