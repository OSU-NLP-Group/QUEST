import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "production_2023_2024_compliance"
TASK_DESCRIPTION = """Identify a theatrical film or television series that premiered or was released between January 1, 2023 and December 31, 2024, meeting ALL of the following requirements:

1. Director/Creator Requirement: The director (if a film) or creator/showrunner (if a series) must have been born between 1960 and 1990 (inclusive).

2. Budget and Scale:
   - If a theatrical film: The production budget must have been at least $100 million USD, and the theatrical runtime must be at least 100 minutes.
   - If a television series: The first season must consist of exactly 10 episodes, with an average episode runtime of at least 45 minutes.

3. Production Location: Principal photography must have taken place, at least partially, in North America (United States or Canada).

4. Technical Crew: The production must have:
   - At least one credited cinematographer or director of photography
   - At least one credited composer for the original score

5. Distribution: The production must have been distributed by or premiered on a major studio or streaming platform (such as Universal Pictures, Warner Bros., Disney, Netflix, FX, HBO, Amazon Prime Video, Hulu, or equivalent).

6. Awards Recognition: The production must have received at least one major award nomination or win from a recognized industry awards organization (Academy Awards, Emmy Awards, Golden Globe Awards, BAFTA Awards, or Screen Actors Guild Awards) in 2023, 2024, or 2025.

7. Language: The production must be primarily in English or feature English as one of the main languages.

Provide the following information for your identified production:
- Official title
- Type (film or series)
- Release/premiere year and date
- Director or creator name and birth year
- Production companies (at least two)
- Primary distributor or platform
- Budget (if film) or episode count (if series)
- Runtime information
- Filming locations
- Cinematographer name(s)
- Composer name(s)
- Major award(s) received or nominated for, with specific category
- Supporting reference URLs from credible sources (IMDb, Wikipedia, official studio sites, awards organization sites, or reputable entertainment industry publications)
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AwardInfo(BaseModel):
    organization: Optional[str] = None   # e.g., Academy Awards, Emmys, Golden Globes, BAFTA, SAG Awards
    year: Optional[str] = None           # e.g., 2024
    category: Optional[str] = None       # e.g., Best Picture
    result: Optional[str] = None         # e.g., Won, Nominated
    url: Optional[str] = None            # optional dedicated award URL if provided


class ProductionExtraction(BaseModel):
    title: Optional[str] = None
    type: Optional[str] = None  # film / series (allow synonyms like movie, television series, limited series)
    release_date: Optional[str] = None   # as stated in the answer
    release_year: Optional[str] = None   # year if mentioned separately
    release_mode: Optional[str] = None   # e.g., theatrical, streaming/platform/network premiere
    premiere_platform_network: Optional[str] = None  # if series

    director_or_creator: Optional[str] = None
    birth_year: Optional[str] = None

    production_companies: List[str] = Field(default_factory=list)

    primary_distributor_platform: Optional[str] = None

    budget_usd: Optional[str] = None                     # for films; keep as text as in the answer
    theatrical_runtime_minutes: Optional[str] = None     # for films; keep as text as in the answer

    episode_count_season1: Optional[str] = None          # for series
    avg_episode_runtime_minutes: Optional[str] = None    # for series

    filming_locations: List[str] = Field(default_factory=list)

    cinematographers: List[str] = Field(default_factory=list)
    composers: List[str] = Field(default_factory=list)
    editors: List[str] = Field(default_factory=list)

    languages: List[str] = Field(default_factory=list)

    awards: List[AwardInfo] = Field(default_factory=list)

    # URLs
    reference_urls: List[str] = Field(default_factory=list)
    awards_urls: List[str] = Field(default_factory=list)
    distribution_urls: List[str] = Field(default_factory=list)
    crew_urls_cinematography: List[str] = Field(default_factory=list)
    crew_urls_editing: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_production() -> str:
    return """
Extract structured details for exactly one production (film or TV series) that the answer proposes as meeting the constraints. Extract fields exactly as present in the answer text; do not invent or normalize beyond what the answer states. If any field is missing in the answer, set it to null or an empty array as appropriate.

Required fields:
- title: Official title of the production.
- type: The production type as stated (acceptable variants include "film", "movie", "feature film", "theatrical film", "series", "tv series", "television series", "limited series", "miniseries").
- release_date: The release/premiere date string exactly as provided in the answer (e.g., "2023-05-25" or "March 15, 2024"). If only a year is given, leave this null and use release_year.
- release_year: The year of release/premiere if present (e.g., "2023" or "2024").
- release_mode: A brief phrase indicating how it was released/premiered, for example "theatrical", "cinematic release", "platform premiere", "network premiere".
- premiere_platform_network: If a series, the platform or network where it premiered (e.g., "HBO", "Netflix", "FX", "Prime Video"). If a film, leave null.

- director_or_creator: Director (for film) or creator/showrunner (for series).
- birth_year: The birth year of the above person, exactly as presented.

- production_companies: List all production companies mentioned for this production in the answer (extract at least two if present).
- primary_distributor_platform: The primary distributor (for a film) or platform/network (for a series), as stated.

- budget_usd: If a film, extract the production budget string (e.g., "$150 million" or "US$200 million"); otherwise null.
- theatrical_runtime_minutes: If a film, extract the theatrical runtime string (e.g., "120 minutes" or "2h 10m"); otherwise null.

- episode_count_season1: If a series, extract the first season episode count as a string (e.g., "10" or "10 episodes"); otherwise null.
- avg_episode_runtime_minutes: If a series, extract the average episode runtime as a string (e.g., "45 minutes", "1h"); otherwise null.

- filming_locations: List of filming locations mentioned (city, state/province, country as given in the answer). If the answer mentions principal photography location(s), include them.

- cinematographers: List of credited cinematographer(s) / DoP(s).
- composers: List of credited original score composer(s).
- editors: List of credited editor(s) or editing team names.

- languages: List the languages mentioned for the production.

- awards: List major awards nominations or wins relevant to Academy Awards (Oscars), Emmys, Golden Globes, BAFTAs, or SAG Awards. For each, extract:
  - organization (e.g., "Academy Awards", "Oscars", "BAFTA", "Emmy Awards", "SAG Awards")
  - year (if stated)
  - category (e.g., "Best Picture", "Outstanding Drama Series")
  - result (e.g., "Nominated", "Won")
  - url (the specific supporting URL if provided)

- reference_urls: All credible supporting URLs the answer cites for any of the above (IMDb, Wikipedia, official studio/platform sites, awards organization sites, reputable entertainment publications).
- awards_urls: Award-specific URLs if provided (or leave empty).
- distribution_urls: URLs supporting the distributor/platform claims if provided (or leave empty).
- crew_urls_cinematography: URLs that directly support the cinematographer credit(s) if provided (or leave empty).
- crew_urls_editing: URLs that directly support the editor/editing team credit(s) if provided (or leave empty).

Return a single JSON object conforming exactly to the schema.
"""


# --------------------------------------------------------------------------- #
# Helper parsing and utility functions                                        #
# --------------------------------------------------------------------------- #
def normalize_type(tp: Optional[str]) -> Optional[str]:
    if not tp:
        return None
    s = tp.strip().lower()
    film_tokens = ["film", "movie", "feature", "theatrical"]
    series_tokens = ["series", "tv series", "television series", "limited series", "miniseries", "mini-series"]
    if any(tok in s for tok in film_tokens):
        return "film"
    if any(tok in s for tok in series_tokens):
        return "series"
    return None


def extract_first_year(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"(19|20)\d{2}", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def parse_int_from_text(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"\d+", s.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def parse_runtime_to_minutes(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    txt = s.lower().strip()
    # Try formats like "2h 10m", "2 h 10 min", "1h", "130 minutes"
    h_m = re.search(r"(?:(\d+)\s*h(?:ours?)?)?\s*(?:(\d+)\s*m(?:in(?:utes)?)?)?", txt)
    if h_m:
        h = h_m.group(1)
        m = h_m.group(2)
        if h or m:
            total = 0
            if h:
                try:
                    total += int(h) * 60
                except Exception:
                    pass
            if m:
                try:
                    total += int(m)
                except Exception:
                    pass
            if total > 0:
                return total
    # Fallback: first integer, assume minutes
    minutes = parse_int_from_text(txt)
    return minutes


def parse_budget_to_usd(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    txt = s.lower().replace(",", "").strip()
    # Extract all numbers (allow decimals)
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", txt)]
    if not nums:
        return None
    # Determine scale
    scale = 1.0
    if "billion" in txt or "bn" in txt:
        scale = 1_000_000_000.0
    elif "million" in txt or "mn" in txt or re.search(r"\b\d+\s*m\b", txt):
        scale = 1_000_000.0
    else:
        # If contains $, we may treat as absolute USD (already in dollars)
        if "$" in txt or "usd" in txt or "us$" in txt:
            scale = 1.0
        # Otherwise, leave as is
    # Use the maximum interpreted value to be safe on ranges
    max_num = max(nums)
    value = max_num * scale
    return value


def is_date_in_window(release_date: Optional[str], release_year: Optional[str]) -> bool:
    # Window: 2023-01-01 to 2024-12-31 inclusive
    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)
    # Try to parse explicit release_date; otherwise fallback to year
    if release_date:
        # Try common formats
        fmts = ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d", "%m/%d/%Y", "%Y"]
        for f in fmts:
            try:
                dt = datetime.strptime(release_date.strip(), f)
                return start <= dt <= end
            except Exception:
                continue
    # Fallback to year
    yr = extract_first_year(release_year or release_date)
    if yr in (2023, 2024):
        return True
    return False


def domain_of(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def domain_matches(host: str, base: str) -> bool:
    host = (host or "").lower()
    base = (base or "").lower()
    if not host or not base:
        return False
    return host == base or host.endswith("." + base)


CREDIBLE_DOMAINS = {
    # Major databases / official encyclopedic
    "imdb.com", "www.imdb.com",
    "wikipedia.org", "en.wikipedia.org",

    # Awards orgs
    "oscars.org", "academy.com",  # oscars.org is canonical; academy.com rarely used
    "bafta.org",
    "emmys.com",
    "sagawards.org",
    "goldenglobes.com",

    # Major studios/platforms (official)
    "disney.com", "thewaltdisneycompany.com", "waltdisneystudios.com",
    "universalpictures.com", "universalstudios.com", "universalpicturesinternational.com",
    "warnerbros.com", "warnerbros.co.uk", "warnerbroslatino.com",
    "netflix.com",
    "hbo.com", "max.com",
    "primevideo.com", "amazon.com", "amazon.co.uk",
    "hulu.com",
    "fxnetworks.com", "fxnow.fxnetworks.com", "fx.com",
    "paramount.com", "paramountpictures.com",
    "sonypictures.com", "sony.com",
    "apple.com", "tv.apple.com",
    "a24films.com", "a24.com",

    # Reputable industry publications
    "variety.com",
    "hollywoodreporter.com",
    "deadline.com",
    "indiewire.com",
    "thewrap.com",
}


def has_credible_url(urls: List[str]) -> bool:
    for u in urls:
        host = domain_of(u)
        for base in CREDIBLE_DOMAINS:
            if domain_matches(host, base):
                return True
    return False


MAJOR_AWARD_KEYWORDS = [
    "academy awards", "oscars", "oscar",
    "emmy", "emmys", "primetime emmy",
    "golden globe", "golden globes",
    "bafta", "british academy film awards",
    "sag awards", "screen actors guild",
]


def is_major_award_org(org: Optional[str]) -> bool:
    if not org:
        return False
    s = org.strip().lower()
    return any(k in s for k in MAJOR_AWARD_KEYWORDS)


def pick_major_award(awards: List[AwardInfo]) -> Optional[AwardInfo]:
    for a in awards:
        if is_major_award_org(a.organization):
            yr = extract_first_year(a.year)
            if yr in (2023, 2024, 2025):
                return a
    return None


# --------------------------------------------------------------------------- #
# Tree building and verification functions                                    #
# --------------------------------------------------------------------------- #
async def build_identity_and_release_group(evaluator: Evaluator, parent_node, info: ProductionExtraction):
    grp = evaluator.add_parallel(
        id="production_identity_and_release",
        desc="Production is identified and release/premiere constraints are satisfied.",
        parent=parent_node,
        critical=True
    )

    # Official title provided
    evaluator.add_custom_node(
        result=bool(info.title and info.title.strip()),
        id="official_title_provided",
        desc="Official title is provided.",
        parent=grp,
        critical=True
    )

    # Type is allowed (film or series; allow common synonyms)
    norm_type = normalize_type(info.type)
    evaluator.add_custom_node(
        result=(norm_type in ("film", "series")),
        id="type_is_allowed",
        desc="Type is provided and is either a film or a limited series.",
        parent=grp,
        critical=True
    )

    # Release/premiere date within window 2023-01-01 to 2024-12-31
    evaluator.add_custom_node(
        result=is_date_in_window(info.release_date, info.release_year),
        id="release_date_within_window",
        desc="Release/premiere date is provided and falls between 2023-01-01 and 2024-12-31 (inclusive).",
        parent=grp,
        critical=True
    )

    # Release mode matches constraint
    release_mode_node = evaluator.add_leaf(
        id="release_mode_matches_constraint",
        desc="Release mode matches the constraint: if film, it was released theatrically; if limited series, it premiered on a platform/network.",
        parent=grp,
        critical=True
    )
    # Prepare claim and sources
    refs = info.reference_urls or []
    if norm_type == "film":
        claim = "This production is a film that was released theatrically (in cinemas), not solely as a direct-to-streaming release."
        add_ins = "Check the referenced sources for clear indications of a theatrical release (e.g., phrases like 'theatrical release', 'opened in theaters', or box office reporting). Limited theatrical releases count."
    elif norm_type == "series":
        claim = "This production is a television series that premiered on a platform or network."
        add_ins = "Verify that the series had a platform/network premiere (e.g., on HBO, Netflix, FX, Prime Video, Hulu, etc.)."
    else:
        claim = "The release mode is correctly stated according to the production's type."
        add_ins = "If the type is unclear, evaluate based on the provided information and sources; if insufficient, mark as not supported."
    await evaluator.verify(
        claim=claim,
        node=release_mode_node,
        sources=refs,
        additional_instruction=add_ins
    )


async def build_director_creator_group(evaluator: Evaluator, parent_node, info: ProductionExtraction):
    grp = evaluator.add_parallel(
        id="director_or_creator_requirement",
        desc="Director (film) or creator/showrunner (limited series) is identified and meets birth-year constraint.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.director_or_creator and info.director_or_creator.strip()),
        id="director_creator_name_provided",
        desc="Director/creator/showrunner name is provided.",
        parent=grp,
        critical=True
    )

    byear = extract_first_year(info.birth_year)
    evaluator.add_custom_node(
        result=(byear is not None and 1960 <= byear <= 1990),
        id="birth_year_in_range",
        desc="Director/creator/showrunner birth year is provided and is between 1960 and 1990 inclusive.",
        parent=grp,
        critical=True
    )


async def build_budget_runtime_group(evaluator: Evaluator, parent_node, info: ProductionExtraction):
    grp = evaluator.add_parallel(
        id="budget_episode_and_runtime_constraints",
        desc="Budget/episode-count and runtime requirements are satisfied (conditional on film vs limited series).",
        parent=parent_node,
        critical=True
    )

    norm_type = normalize_type(info.type)

    # Budget or episode count requirement
    if norm_type == "film":
        budget_val = parse_budget_to_usd(info.budget_usd)
        evaluator.add_custom_node(
            result=(budget_val is not None and budget_val >= 100_000_000),
            id="budget_or_episode_count_requirement",
            desc="If film: production budget is provided and is ≥ $100M USD. If limited series: first season episode count is provided and equals exactly 10.",
            parent=grp,
            critical=True
        )
    elif norm_type == "series":
        ep_count = parse_int_from_text(info.episode_count_season1)
        evaluator.add_custom_node(
            result=(ep_count == 10),
            id="budget_or_episode_count_requirement",
            desc="If film: production budget is provided and is ≥ $100M USD. If limited series: first season episode count is provided and equals exactly 10.",
            parent=grp,
            critical=True
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="budget_or_episode_count_requirement",
            desc="If film: production budget is provided and is ≥ $100M USD. If limited series: first season episode count is provided and equals exactly 10.",
            parent=grp,
            critical=True
        )

    # Runtime requirement
    if norm_type == "film":
        rt_min = parse_runtime_to_minutes(info.theatrical_runtime_minutes)
        evaluator.add_custom_node(
            result=(rt_min is not None and rt_min >= 100),
            id="runtime_requirement",
            desc="If film: theatrical runtime is provided and is ≥ 100 minutes. If limited series: average episode runtime is provided and is ≥ 45 minutes.",
            parent=grp,
            critical=True
        )
    elif norm_type == "series":
        avg_rt = parse_runtime_to_minutes(info.avg_episode_runtime_minutes)
        evaluator.add_custom_node(
            result=(avg_rt is not None and avg_rt >= 45),
            id="runtime_requirement",
            desc="If film: theatrical runtime is provided and is ≥ 100 minutes. If limited series: average episode runtime is provided and is ≥ 45 minutes.",
            parent=grp,
            critical=True
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="runtime_requirement",
            desc="If film: theatrical runtime is provided and is ≥ 100 minutes. If limited series: average episode runtime is provided and is ≥ 45 minutes.",
            parent=grp,
            critical=True
        )


async def build_location_group(evaluator: Evaluator, parent_node, info: ProductionExtraction):
    grp = evaluator.add_parallel(
        id="production_location_requirement",
        desc="Filming location constraint is satisfied and locations are provided.",
        parent=parent_node,
        critical=True
    )

    # Verify North America (US or Canada) via sources
    na_node = evaluator.add_leaf(
        id="north_america_filming_constraint",
        desc="Principal photography/filming occurred at least partially in the United States or Canada.",
        parent=grp,
        critical=True
    )
    claim = "Principal photography (or significant filming) occurred at least partially in the United States or Canada."
    await evaluator.verify(
        claim=claim,
        node=na_node,
        sources=info.reference_urls,
        additional_instruction="Look for filming or principal photography locations that include US states/cities or Canada (provinces/cities) on the referenced pages."
    )

    # Filming locations provided (at least one)
    evaluator.add_custom_node(
        result=bool(info.filming_locations and len(info.filming_locations) > 0),
        id="filming_locations_provided",
        desc="Filming locations are listed (at least one location is provided).",
        parent=grp,
        critical=True
    )


async def build_technical_crew_group(evaluator: Evaluator, parent_node, info: ProductionExtraction):
    grp = evaluator.add_parallel(
        id="technical_crew_requirements",
        desc="Required credited technical crew are identified per constraints.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.cinematographers and len(info.cinematographers) > 0),
        id="cinematographer_named",
        desc="At least one credited cinematographer/director of photography is named.",
        parent=grp,
        critical=True
    )

    # Cinematographer credit supported by a major database
    cine_support_node = evaluator.add_leaf(
        id="cinematographer_credit_in_major_database",
        desc="At least one supporting URL from a major film/TV database verifies the cinematographer/DoP credit (per the constraint that the credit appears in a major database).",
        parent=grp,
        critical=True
    )
    cine_sources = (info.crew_urls_cinematography or []) + (info.reference_urls or [])
    cine_names = ", ".join(info.cinematographers) if info.cinematographers else "the cinematographer(s)"
    cine_claim = f"{cine_names} is credited as cinematographer/director of photography for this production."
    await evaluator.verify(
        claim=cine_claim,
        node=cine_support_node,
        sources=cine_sources,
        additional_instruction="Prefer IMDb or Wikipedia or official studio/platform pages to verify crew credits. Accept if any credible page clearly lists the cinematographer/DoP."
    )

    evaluator.add_custom_node(
        result=bool(info.composers and len(info.composers) > 0),
        id="composer_named",
        desc="At least one credited original-score composer is named.",
        parent=grp,
        critical=True
    )


async def build_editor_group(evaluator: Evaluator, parent_node, info: ProductionExtraction):
    grp = evaluator.add_parallel(
        id="editor_requirement",
        desc="Editor/editing team credit is provided as required by constraints.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.editors and len(info.editors) > 0),
        id="editor_named",
        desc="At least one editor or an editing team is identified.",
        parent=grp,
        critical=True
    )

    editor_support_node = evaluator.add_leaf(
        id="editor_credit_supported",
        desc="A credible/official source URL supports the editor/editing-team credit.",
        parent=grp,
        critical=True
    )
    editor_sources = (info.crew_urls_editing or []) + (info.reference_urls or [])
    editor_names = ", ".join(info.editors) if info.editors else "the editor(s)"
    editor_claim = f"{editor_names} is credited as editor/editing team for this production."
    await evaluator.verify(
        claim=editor_claim,
        node=editor_support_node,
        sources=editor_sources,
        additional_instruction="Verify editor credits on IMDb, Wikipedia, or official studio/platform pages. Accept if at least one credible page clearly lists the editor(s)."
    )


async def build_distribution_group(evaluator: Evaluator, parent_node, info: ProductionExtraction):
    grp = evaluator.add_parallel(
        id="distribution_requirement",
        desc="Distributor/platform information is provided and meets the 'major studio/streaming platform' constraint.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.primary_distributor_platform and info.primary_distributor_platform.strip()),
        id="primary_distributor_or_platform_provided",
        desc="Primary distributor (film) or platform/network (limited series) is identified.",
        parent=grp,
        critical=True
    )

    distrib_node = evaluator.add_leaf(
        id="distributor_is_major_or_equivalent",
        desc="Distributor/platform is one of the listed major entities or the answer provides a credible citation supporting that it is an equivalent major studio/streaming/network platform.",
        parent=grp,
        critical=True
    )
    distrib_name = info.primary_distributor_platform or "the stated distributor/platform"
    distrib_sources = (info.distribution_urls or []) + (info.reference_urls or [])
    distrib_claim = f"{distrib_name} is a major studio or streaming/network platform that distributed or premiered this production."
    await evaluator.verify(
        claim=distrib_claim,
        node=distrib_node,
        sources=distrib_sources,
        additional_instruction="Accept if the distributor/platform is among Universal Pictures, Warner Bros., Disney, Netflix, FX, HBO/Max, Amazon Prime Video, Hulu, Paramount, Sony, Apple TV+, or an equivalent major entity as evidenced by the provided credible sources."
    )


async def build_awards_group(evaluator: Evaluator, parent_node, info: ProductionExtraction):
    grp = evaluator.add_parallel(
        id="awards_recognition_requirement",
        desc="Awards recognition constraint is met and details are provided.",
        parent=parent_node,
        critical=True
    )

    # Pick one major award that fits 2023/2024/2025
    award = pick_major_award(info.awards or [])
    award_claim_node = evaluator.add_leaf(
        id="major_award_nomination_or_win",
        desc="At least one nomination or win from Oscars/Emmys/Golden Globes/BAFTAs/SAG Awards in 2023, 2024, or 2025 is documented.",
        parent=grp,
        critical=True
    )
    if award:
        org = award.organization or "a major awards organization"
        cat = award.category or "a major category"
        res = (award.result or "nominated/won").lower()
        yr = award.year or ""
        claim = f"The production {('won' if 'win' in res or 'won' in res else 'was nominated for')} {cat} at {org} in {yr}."
    else:
        claim = "The production received at least one nomination or win at the Academy Awards (Oscars), Emmys, Golden Globes, BAFTAs, or SAG Awards in 2023, 2024, or 2025."
    award_sources = (info.awards_urls or []) + (info.reference_urls or [])
    await evaluator.verify(
        claim=claim,
        node=award_claim_node,
        sources=award_sources,
        additional_instruction="Verify that the award is from one of: Academy Awards (Oscars), Emmy Awards, Golden Globe Awards, BAFTA Awards, or SAG Awards, and that the year is 2023, 2024, or 2025."
    )

    evaluator.add_custom_node(
        result=bool(award and award.category and award.category.strip()),
        id="award_category_provided",
        desc="The specific award category for the cited nomination/win is provided.",
        parent=grp,
        critical=True
    )


async def build_language_group(evaluator: Evaluator, parent_node, info: ProductionExtraction):
    grp = evaluator.add_parallel(
        id="language_requirement",
        desc="Language constraint is satisfied.",
        parent=parent_node,
        critical=True
    )

    lang_node = evaluator.add_leaf(
        id="english_language_constraint",
        desc="Production is primarily in English or English is one of the main languages (and language info is provided).",
        parent=grp,
        critical=True
    )
    languages_str = ", ".join(info.languages) if info.languages else "the stated languages"
    claim = f"English is a primary or one of the main languages of this production as per the provided sources for {languages_str}."
    await evaluator.verify(
        claim=claim,
        node=lang_node,
        sources=info.reference_urls,
        additional_instruction="Check language information on credible sources (IMDb, Wikipedia, official sites). Pass if English is listed as the primary language or one of the main languages."
    )


async def build_production_companies_group(evaluator: Evaluator, parent_node, info: ProductionExtraction):
    grp = evaluator.add_parallel(
        id="production_companies_requirement",
        desc="Production companies are provided and meet the minimum count requirement.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.production_companies and len([x for x in info.production_companies if x and x.strip()]) >= 2),
        id="at_least_two_production_companies_listed",
        desc="At least two production companies are identified.",
        parent=grp,
        critical=True
    )


async def build_references_group(evaluator: Evaluator, parent_node, info: ProductionExtraction):
    grp = evaluator.add_parallel(
        id="supporting_references_requirement",
        desc="Supporting reference URLs are provided from credible sources as specified and support the required claims.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_credible_url(info.reference_urls or []),
        id="credible_reference_urls_provided",
        desc="Credible reference URLs (e.g., IMDb, Wikipedia, official studio/network sites, awards organization sites, or reputable industry publications) are provided and support the required claims.",
        parent=grp,
        critical=True
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
    Evaluate one answer for the production identification task with 2023-2024 constraints.
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

    # Extract structured info from the answer
    info: ProductionExtraction = await evaluator.extract(
        prompt=prompt_extract_production(),
        template_class=ProductionExtraction,
        extraction_name="production_extraction"
    )

    # Add a critical top-level node to hold all rubric groups
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Identify one eligible production and provide all required fields with credible supporting URLs, satisfying all listed constraints.",
        parent=root,
        critical=True
    )

    # Build all rubric groups
    await build_identity_and_release_group(evaluator, task_root, info)
    await build_director_creator_group(evaluator, task_root, info)
    await build_budget_runtime_group(evaluator, task_root, info)
    await build_location_group(evaluator, task_root, info)
    await build_technical_crew_group(evaluator, task_root, info)
    await build_editor_group(evaluator, task_root, info)
    await build_distribution_group(evaluator, task_root, info)
    await build_awards_group(evaluator, task_root, info)
    await build_language_group(evaluator, task_root, info)
    await build_production_companies_group(evaluator, task_root, info)
    await build_references_group(evaluator, task_root, info)

    # Add some computed helper info to summary for transparency
    norm_type = normalize_type(info.type)
    byear = extract_first_year(info.birth_year)
    rt_film = parse_runtime_to_minutes(info.theatrical_runtime_minutes) if norm_type == "film" else None
    avg_rt_series = parse_runtime_to_minutes(info.avg_episode_runtime_minutes) if norm_type == "series" else None
    budget_val = parse_budget_to_usd(info.budget_usd) if norm_type == "film" else None
    ep_count = parse_int_from_text(info.episode_count_season1) if norm_type == "series" else None

    evaluator.add_custom_info(
        info={
            "normalized_type": norm_type,
            "director_or_creator_birth_year": byear,
            "film_budget_usd": budget_val,
            "film_runtime_minutes": rt_film,
            "series_episode_count_s1": ep_count,
            "series_avg_episode_runtime_minutes": avg_rt_series,
            "release_date_in_window": is_date_in_window(info.release_date, info.release_year),
            "credible_reference_present": has_credible_url(info.reference_urls or [])
        },
        info_type="computed_fields_summary"
    )

    return evaluator.get_summary()