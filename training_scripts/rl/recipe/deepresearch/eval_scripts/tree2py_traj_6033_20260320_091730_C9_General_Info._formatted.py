import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "entertainment_professionals_2024_2026"
TASK_DESCRIPTION = (
    "Identify four entertainment professionals (A, B, C, D) who each satisfy all of their respective "
    "criteria based on verifiable information about their professional activities and achievements "
    "during the 2024-2026 period. For each professional, the answer must provide the full name and URL "
    "references that verify every specified criterion."
)


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _nonempty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]

def _merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for urls in url_lists:
        if not urls:
            continue
        for u in urls:
            if isinstance(u, str) and u.strip() and u not in seen:
                merged.append(u)
                seen.add(u)
    return merged

def _safe_str(x: Optional[str]) -> str:
    return x or ""


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class ProfAProfessor(BaseModel):
    title: Optional[str] = None
    department_or_field: Optional[str] = None
    university: Optional[str] = None
    state_or_location: Optional[str] = None
    urls: List[str] = Field(default_factory=list)

class ProfALab(BaseModel):
    lab_name: Optional[str] = None
    lab_role: Optional[str] = None  # e.g., "Director"
    lab_university: Optional[str] = None
    urls: List[str] = Field(default_factory=list)

class ProfAPrevShow(BaseModel):
    show_name: Optional[str] = None
    season_or_year: Optional[str] = None  # e.g., "Season 5 (2014)" or "2016"
    urls: List[str] = Field(default_factory=list)

class ProfAReturnShow(BaseModel):
    show_name: Optional[str] = None
    season_or_year: Optional[str] = None  # e.g., "Season 2026" / "Season 7 (2026)"
    scheduled_air_window: Optional[str] = None  # e.g., "Spring 2026"
    urls: List[str] = Field(default_factory=list)

class ProfAParenthood(BaseModel):
    first_child_indicator: Optional[str] = None  # "first child" / "first-time parent" / yes/no text
    child_birth_date: Optional[str] = None
    filming_start_date_2026_season: Optional[str] = None
    urls: List[str] = Field(default_factory=list)

class ProfessionalA(BaseModel):
    full_name: Optional[str] = None
    professorship: Optional[ProfAProfessor] = None
    lab_director: Optional[ProfALab] = None
    previous_reality: Optional[ProfAPrevShow] = None
    return_2026: Optional[ProfAReturnShow] = None
    parenthood: Optional[ProfAParenthood] = None


class ProfBSeriesRole(BaseModel):
    series_name: Optional[str] = None
    character_name: Optional[str] = None
    relationship_description: Optional[str] = None  # e.g., "younger sibling to the lead characters"
    urls: List[str] = Field(default_factory=list)

class ProfBSeriesFinale(BaseModel):
    series_name: Optional[str] = None
    finale_date: Optional[str] = None  # aim for "December 31, 2025"
    urls: List[str] = Field(default_factory=list)

class ProfBHorror2023(BaseModel):
    film_title: Optional[str] = None  # should include "Evil Dead"
    release_year: Optional[str] = None  # "2023"
    urls: List[str] = Field(default_factory=list)

class ProfBAdditionalFilm2024(BaseModel):
    film_titles: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)

class ProfBDiscovery(BaseModel):
    description: Optional[str] = None  # how the creators discovered the performer via the 2023 horror film
    urls: List[str] = Field(default_factory=list)

class ProfessionalB(BaseModel):
    full_name: Optional[str] = None
    birth_date: Optional[str] = None
    birth_urls: List[str] = Field(default_factory=list)
    series_role: Optional[ProfBSeriesRole] = None
    series_finale: Optional[ProfBSeriesFinale] = None
    horror_2023: Optional[ProfBHorror2023] = None
    additional_2024: Optional[ProfBAdditionalFilm2024] = None
    casting_discovery: Optional[ProfBDiscovery] = None


class ProfCBirth(BaseModel):
    birth_date: Optional[str] = None  # target: February 1992
    urls: List[str] = Field(default_factory=list)

class ProfCSibling(BaseModel):
    sibling_name: Optional[str] = None
    relationship: Optional[str] = None  # "younger sibling", etc.
    sibling_is_major_franchise_star: Optional[str] = None  # free text yes/why
    urls: List[str] = Field(default_factory=list)

class ProfCSocialDeduction(BaseModel):
    show_name: Optional[str] = None
    season_or_year: Optional[str] = None
    premiere_date: Optional[str] = None  # "early 2025"
    urls: List[str] = Field(default_factory=list)

class ProfCDancing(BaseModel):
    show_name: Optional[str] = None
    network_or_platform: Optional[str] = None  # ensure "broadcast television"
    year: Optional[str] = None  # "2025"
    partner_first_name: Optional[str] = None  # expect "Daniella"
    partner_full_name: Optional[str] = None
    reached_round: Optional[str] = None  # "semifinals" etc.
    urls: List[str] = Field(default_factory=list)

class ProfCSiblingAttendance(BaseModel):
    date: Optional[str] = None  # "November 19, 2025"
    urls: List[str] = Field(default_factory=list)

class ProfessionalC(BaseModel):
    full_name: Optional[str] = None
    birth: Optional[ProfCBirth] = None
    sibling: Optional[ProfCSibling] = None
    social_deduction: Optional[ProfCSocialDeduction] = None
    dancing: Optional[ProfCDancing] = None
    sibling_attendance: Optional[ProfCSiblingAttendance] = None


class FilmCredit(BaseModel):
    title: Optional[str] = None
    release_date: Optional[str] = None  # any format; should include year
    is_animated: Optional[str] = None  # free text "yes/no"
    is_sequel: Optional[str] = None  # free text "yes/no"
    franchise_original_year: Optional[str] = None  # for sequel lineage (expect "1988" for the target)
    primary_genre: Optional[str] = None  # "comedy", etc.
    urls: List[str] = Field(default_factory=list)

class ProfDPortfolio(BaseModel):
    films_2023_2024: List[FilmCredit] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)  # general filmography pages ok

class ProfDStreamingCollection(BaseModel):
    platform_name: Optional[str] = None  # "Netflix", "Max", "Disney+", "Prime Video", etc.
    launch_date: Optional[str] = None  # prefer exact date or "February 2026"
    urls: List[str] = Field(default_factory=list)

class ProfessionalD(BaseModel):
    full_name: Optional[str] = None
    birth_year: Optional[str] = None
    birth_urls: List[str] = Field(default_factory=list)
    portfolio: Optional[ProfDPortfolio] = None
    collection: Optional[ProfDStreamingCollection] = None


class AllProfessionalsExtraction(BaseModel):
    professional_a: Optional[ProfessionalA] = None
    professional_b: Optional[ProfessionalB] = None
    professional_c: Optional[ProfessionalC] = None
    professional_d: Optional[ProfessionalD] = None


# -----------------------------------------------------------------------------
# Extraction prompt builders
# -----------------------------------------------------------------------------
def prompt_extract_all() -> str:
    return """
    Extract structured information for exactly four professionals labeled A, B, C, and D as they appear in the answer.
    For each professional, return all fields listed below. If something is not present in the answer, return null or [] accordingly.

    For Professional A:
    - full_name
    - professorship:
        - title (e.g., "Assistant Professor", "Associate Professor", etc.)
        - department_or_field (e.g., "Computer Science", "Engineering", "Mathematics", etc.)
        - university (full name)
        - state_or_location (e.g., "Florida", "FL", or city/state string)
        - urls (all URLs cited that support the professorship and institution details)
    - lab_director:
        - lab_name
        - lab_role (e.g., "Director")
        - lab_university (university/institution name if present)
        - urls (URLs confirming lab directorship and lab institution)
    - previous_reality:
        - show_name
        - season_or_year (e.g., "Season 3 (2014)" or "2016")
        - urls (URLs confirming previous contestant appearance)
    - return_2026:
        - show_name (the returning show)
        - season_or_year (e.g., "Season 2026")
        - scheduled_air_window (e.g., "Spring 2026")
        - urls (URLs confirming the return in 2026 and the airing window)
    - parenthood:
        - first_child_indicator (text indicating first-time parent if present)
        - child_birth_date (date string as provided)
        - filming_start_date_2026_season (date string for start of filming)
        - urls (URLs confirming first-time parenthood and timing vs filming)

    For Professional B:
    - full_name
    - birth_date (as provided in the answer)
    - birth_urls (URLs supporting the birth date)
    - series_role:
        - series_name
        - character_name
        - relationship_description (e.g., "younger sibling of the main characters")
        - urls (URLs supporting the role and relationship)
    - series_finale:
        - series_name
        - finale_date (e.g., "December 31, 2025")
        - urls (URLs supporting the final season and finale date)
    - horror_2023:
        - film_title (should include "Evil Dead" in the title)
        - release_year (e.g., "2023")
        - urls (URLs supporting the 2023 horror film appearance)
    - additional_2024:
        - film_titles (list of feature films in 2024 other than the 2023 horror film)
        - urls (URLs supporting the 2024 feature film appearances)
    - casting_discovery:
        - description (text indicating creators noticed the performer after the 2023 horror film)
        - urls (URLs supporting how the creators discovered the performer)

    For Professional C:
    - full_name
    - birth:
        - birth_date (target: February 1992, in any reasonable format if available)
        - urls (URLs supporting birth date)
    - sibling:
        - sibling_name (the famous actor sibling)
        - relationship (e.g., "younger sibling")
        - sibling_is_major_franchise_star (text; do not infer beyond the answer)
        - urls (URLs supporting sibling relationship and fame)
    - social_deduction:
        - show_name (social deduction/strategy-based reality competition)
        - season_or_year
        - premiere_date (target: early 2025)
        - urls (URLs supporting participation and premiere timing)
    - dancing:
        - show_name (broadcast TV dancing competition)
        - network_or_platform (e.g., ABC, FOX, etc.)
        - year ("2025")
        - partner_first_name (target: "Daniella")
        - partner_full_name (if present)
        - reached_round (e.g., "semifinals" or beyond)
        - urls (URLs supporting participation, partner, broadcast, and advancement)
    - sibling_attendance:
        - date (target: "November 19, 2025")
        - urls (URLs supporting sibling's attendance on that date)

    For Professional D:
    - full_name
    - birth_year (e.g., "1957")
    - birth_urls (URLs supporting birth year)
    - portfolio:
        - films_2023_2024: list of up to four FilmCredit objects, each:
            - title
            - release_date (include year)
            - is_animated (e.g., "yes"/"no")
            - is_sequel (e.g., "yes"/"no")
            - franchise_original_year (e.g., "1988" if relevant)
            - primary_genre (e.g., "comedy", "action", etc.)
            - urls (URLs supporting appearance and release date)
        - urls (general filmography pages; include all relevant URLs cited)
    - collection:
        - platform_name (major streaming platform that launched a special collection)
        - launch_date (preferably "February 2026" or an exact date in Feb 2026)
        - urls (URLs supporting the collection/launch)

    IMPORTANT:
    - Extract exactly what the answer explicitly states. Do not invent or infer beyond the answer text.
    - For any field with multiple possible URLs, return all URLs mentioned for that item.
    - When a date is required, return it as text as written in the answer; do not reformat strictly.
    - If any field is missing from the answer, return null or [] as appropriate.
    """


# -----------------------------------------------------------------------------
# URL-backed leaf helper
# -----------------------------------------------------------------------------
async def _add_url_verification_leaf(
    evaluator: Evaluator,
    *,
    parent,
    leaf_id: str,
    desc: str,
    claim: str,
    urls: Optional[List[str]],
    add_ins: Optional[str] = None,
    critical: bool = True,
):
    sources = _nonempty_urls(urls)
    if not sources:
        # No sources provided — fail this URL-backed requirement explicitly
        evaluator.add_custom_node(
            result=False,
            id=leaf_id,
            desc=f"{desc} (failed: no source URL provided)",
            parent=parent,
            critical=critical,
        )
        return

    node = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=add_ins or "Verify strictly using the provided webpages. If the pages don't state this clearly, mark as not supported."
    )


# -----------------------------------------------------------------------------
# Professional A verification
# -----------------------------------------------------------------------------
async def verify_professional_a(evaluator: Evaluator, parent, data: Optional[ProfessionalA]):
    prof_node = evaluator.add_parallel(
        id="professional_A",
        desc="Professional A: Florida STEM professor, lab director, returning 2026 reality contestant, first-time parent near filming",
        parent=parent,
        critical=False
    )

    name = _safe_str(data.full_name if data else None)

    # 1) Florida professorship in STEM
    prof_prof_node = evaluator.add_parallel(
        id="professional_A_florida_professor",
        desc="A holds a professorship position in a STEM field at a university located in Florida",
        parent=prof_node,
        critical=True
    )
    prof_prof = data.professorship if data else None

    # 1.a URL confirms professorship at Florida university
    await _add_url_verification_leaf(
        evaluator,
        parent=prof_prof_node,
        leaf_id="professional_A_florida_professor_url",
        desc="Provide URL reference confirming the professorship position at a Florida university",
        claim=f"This page confirms that {_safe_str(name)} holds a professorship (faculty position) at a Florida-based university.",
        urls=prof_prof.urls if prof_prof else [],
        add_ins="It's sufficient if the page shows a faculty appointment and the institution is in Florida (address, university name indicating Florida, 'FL', etc.).",
        critical=True
    )

    # 1.b STEM field
    node_stem = evaluator.add_leaf(
        id="professional_A_stem_field",
        desc="The professorship is specifically in a STEM field",
        parent=prof_prof_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The professorship/department for {_safe_str(name)} is in a STEM discipline (e.g., science, technology, engineering, mathematics).",
        node=node_stem,
        sources=prof_prof.urls if prof_prof else [],
        additional_instruction="Accept typical STEM departments such as Computer Science, Engineering, Mathematics, Physics, Biology, Chemistry, etc. Verify via the provided pages."
    )

    # 1.c University located in Florida
    node_florida = evaluator.add_leaf(
        id="professional_A_florida_location",
        desc="The university is located in the state of Florida",
        parent=prof_prof_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The university for {_safe_str(name)} is located in the U.S. state of Florida.",
        node=node_florida,
        sources=prof_prof.urls if prof_prof else [],
        additional_instruction="Accept if the page explicitly indicates Florida (FL) in address or widely-known Florida university names."
    )

    # 2) Lab director at same institution
    lab_node = evaluator.add_parallel(
        id="professional_A_lab_director",
        desc="A serves as a director of a research laboratory at the same institution",
        parent=prof_node,
        critical=True
    )
    lab = data.lab_director if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=lab_node,
        leaf_id="professional_A_lab_director_url",
        desc="Provide URL reference confirming the laboratory director position",
        claim=f"This page confirms that {_safe_str(name)} serves as a director of a research laboratory.",
        urls=lab.urls if lab else [],
        add_ins="Look for 'Director' or equivalent leadership title for a research lab or center.",
        critical=True
    )

    node_same_inst = evaluator.add_leaf(
        id="professional_A_same_institution",
        desc="The laboratory is at the same university as the professorship",
        parent=lab_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The research laboratory directed by {_safe_str(name)} is part of the same university as their professorship.",
        node=node_same_inst,
        sources=_merge_urls(lab.urls if lab else [], prof_prof.urls if prof_prof else []),
        additional_instruction="Confirm that the lab is under the same university name or umbrella as the professorship page."
    )

    # 3) Previously appeared as contestant on reality competition series before 2020
    prev_node = evaluator.add_parallel(
        id="professional_A_previous_reality_show",
        desc="A previously appeared as a contestant on a reality competition TV series that aired before 2020",
        parent=prof_node,
        critical=True
    )
    prev = data.previous_reality if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=prev_node,
        leaf_id="professional_A_previous_reality_url",
        desc="Provide URL reference confirming previous appearance on reality competition series",
        claim=f"This page confirms that {_safe_str(name)} previously appeared as a contestant on a reality competition television series.",
        urls=prev.urls if prev else [],
        add_ins="Look for contestant casting/participation info, season pages, or official bios.",
        critical=True
    )

    node_before_2020 = evaluator.add_leaf(
        id="professional_A_aired_before_2020",
        desc="The previous reality competition season aired before 2020",
        parent=prev_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The season or iteration of the reality competition featuring {_safe_str(name)} aired before the year 2020.",
        node=node_before_2020,
        sources=prev.urls if prev else [],
        additional_instruction="It's sufficient if the cited page clearly indicates the prior season's year < 2020."
    )

    # 4) Returning for new season in spring 2026 (same franchise)
    ret_node = evaluator.add_parallel(
        id="professional_A_2026_reality_return",
        desc="A is returning as a contestant for a new season in spring 2026 (same franchise)",
        parent=prof_node,
        critical=True
    )
    ret = data.return_2026 if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=ret_node,
        leaf_id="professional_A_2026_reality_url",
        desc="Provide URL reference confirming return for 2026 season of the same franchise",
        claim=f"This page confirms that {_safe_str(name)} is returning as a contestant in a 2026 season of the reality competition franchise.",
        urls=ret.urls if ret else [],
        add_ins="Look for announcements, official cast lists, or trade coverage that the person is returning in 2026.",
        critical=True
    )

    node_spring = evaluator.add_leaf(
        id="professional_A_spring_2026",
        desc="The new season is scheduled to air in spring 2026",
        parent=ret_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2026 season is scheduled to air in spring 2026 (e.g., March–May 2026).",
        node=node_spring,
        sources=ret.urls if ret else [],
        additional_instruction="Treat 'Spring 2026' as roughly March through May 2026 as commonly used in TV season announcements."
    )

    node_same_franchise = evaluator.add_leaf(
        id="professional_A_same_franchise",
        desc="The 2026 season is of the same reality competition franchise as the previous appearance",
        parent=ret_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2026 season that {_safe_str(name)} is returning to is the same reality competition franchise as the prior appearance.",
        node=node_same_franchise,
        sources=_merge_urls(ret.urls if ret else [], prev.urls if prev else []),
        additional_instruction="Cross-check the franchise/series name across both prior and 2026 links."
    )

    # 5) Became a parent for the first time within 6 weeks before filming began
    par_node = evaluator.add_parallel(
        id="professional_A_became_parent",
        desc="A became a first-time parent within 6 weeks before filming began for the 2026 season",
        parent=prof_node,
        critical=True
    )
    par = data.parenthood if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=par_node,
        leaf_id="professional_A_parent_url",
        desc="Provide URL reference confirming becoming a parent and the timing relative to filming",
        claim=f"This page confirms that {_safe_str(name)} became a parent and indicates timing relative to 2026 season filming.",
        urls=par.urls if par else [],
        add_ins="Look for mentions of a birth announcement/date and a filming start date for the 2026 season.",
        critical=True
    )

    node_first_child = evaluator.add_leaf(
        id="professional_A_first_child",
        desc="This was the professional's first child",
        parent=par_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The birth mentioned indicates that it was the first child for {_safe_str(name)}.",
        node=node_first_child,
        sources=par.urls if par else [],
        additional_instruction="Look for explicit phrasing like 'first child', 'first-time parent', or equivalent confirmation."
    )

    node_within_6w = evaluator.add_leaf(
        id="professional_A_within_6_weeks",
        desc="The child was born within 6 weeks before filming the 2026 reality competition",
        parent=par_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The child's birth date and the 2026 filming start date indicate that the birth occurred within 6 weeks (≤42 days) before filming started.",
        node=node_within_6w,
        sources=par.urls if par else [],
        additional_instruction="Use the dates on the page(s). If both dates are present, check if the difference is 42 days or less."
    )


# -----------------------------------------------------------------------------
# Professional B verification
# -----------------------------------------------------------------------------
async def verify_professional_b(evaluator: Evaluator, parent, data: Optional[ProfessionalB]):
    prof_node = evaluator.add_parallel(
        id="professional_B",
        desc="Professional B: Young performer in a concluded streaming series with 2023 horror ('Evil Dead') + 2024 films, discovered via that horror work",
        parent=parent,
        critical=False
    )

    name = _safe_str(data.full_name if data else None)

    # 1) Age requirement (<= 14 by 2025-12-31) with URL-backed birth date + logical check
    age_node = evaluator.add_parallel(
        id="professional_B_age_requirement",
        desc="B was 14 years old or younger as of December 31, 2025",
        parent=prof_node,
        critical=True
    )
    dob = _safe_str(data.birth_date if data else None)
    birth_urls = data.birth_urls if data else []

    await _add_url_verification_leaf(
        evaluator,
        parent=age_node,
        leaf_id="professional_B_age_url",
        desc="Provide URL reference confirming the birth date",
        claim=f"This page confirms the birth date of {_safe_str(name)} as {_safe_str(dob)}.",
        urls=birth_urls,
        add_ins="Confirm the on-page stated date of birth.",
        critical=True
    )

    node_age_calc = evaluator.add_leaf(
        id="professional_B_age_calculation",
        desc="The birth date confirms the professional was 14 or younger as of December 31, 2025",
        parent=age_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Given the birth date {_safe_str(dob)}, {_safe_str(name)} was 14 years old or younger on 2025-12-31.",
        node=node_age_calc,
        additional_instruction="Compute age as of 2025-12-31 from the provided birth date. If the date is missing or ambiguous, mark incorrect."
    )

    # 2) Born on or after Nov 1, 2011 (URL-backed)
    born_after_node = evaluator.add_parallel(
        id="professional_B_birth_date",
        desc="B was born on or after November 1, 2011",
        parent=prof_node,
        critical=True
    )
    await _add_url_verification_leaf(
        evaluator,
        parent=born_after_node,
        leaf_id="professional_B_birth_url",
        desc="Provide URL reference confirming birth date is November 1, 2011 or later",
        claim=f"The birth date of {_safe_str(name)} is on or after November 1, 2011.",
        urls=birth_urls,
        add_ins="Use the on-page birth date; verify it's >= 2011-11-01.",
        critical=True
    )

    # 3) Portrayed younger sibling of main characters in a streaming series
    role_node = evaluator.add_parallel(
        id="professional_B_streaming_series_character",
        desc="B portrayed a younger-sibling character in a streaming TV series",
        parent=prof_node,
        critical=True
    )
    role = data.series_role if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=role_node,
        leaf_id="professional_B_character_url",
        desc="Provide URL reference confirming the character portrayal in the streaming series",
        claim=f"This page confirms that {_safe_str(name)} portrayed the character '{_safe_str(role.character_name if role else None)}' in the streaming series '{_safe_str(role.series_name if role else None)}'.",
        urls=role.urls if role else [],
        add_ins="Confirm cast/role listing in the series.",
        critical=True
    )

    node_sibling_rel = evaluator.add_leaf(
        id="professional_B_sibling_relationship",
        desc="The character is the younger sibling of main characters in the series",
        parent=role_node,
        critical=True
    )
    await evaluator.verify(
        claim="The portrayed character is specifically the younger sibling of the main characters in the series.",
        node=node_sibling_rel,
        sources=role.urls if role else [],
        additional_instruction="Look for wording such as 'younger sister/brother' relative to the leads."
    )

    # 4) Series concluded with final season in 2025; finale on Dec 31, 2025
    finale_node = evaluator.add_parallel(
        id="professional_B_series_finale",
        desc="The streaming series concluded in 2025 with the finale on December 31, 2025",
        parent=prof_node,
        critical=True
    )
    finale = data.series_finale if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=finale_node,
        leaf_id="professional_B_finale_url",
        desc="Provide URL reference confirming the series finale date",
        claim="The series had its finale in 2025 and concluded its run in that year.",
        urls=finale.urls if finale else [],
        add_ins="Ideally the page states final season/year and a finale date.",
        critical=True
    )

    node_finale_date = evaluator.add_leaf(
        id="professional_B_finale_date",
        desc="The series finale was released on December 31, 2025",
        parent=finale_node,
        critical=True
    )
    await evaluator.verify(
        claim="The series finale episode was released on December 31, 2025.",
        node=node_finale_date,
        sources=finale.urls if finale else [],
        additional_instruction="Confirm the exact release date as 2025-12-31 (tolerate reasonable time zone wording if explicit)."
    )

    # 5) Appeared in a 2023 horror film with "Evil Dead" in its title
    horror_node = evaluator.add_parallel(
        id="professional_B_horror_film_2023",
        desc="B appeared in a 2023 horror film with 'Evil Dead' in the title",
        parent=prof_node,
        critical=True
    )
    horror = data.horror_2023 if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=horror_node,
        leaf_id="professional_B_horror_url",
        desc="Provide URL reference confirming appearance in the 2023 horror film",
        claim=f"This page confirms that {_safe_str(name)} appeared in the horror film '{_safe_str(horror.film_title if horror else None)}'.",
        urls=horror.urls if horror else [],
        add_ins="Confirm cast credit on the film page.",
        critical=True
    )

    node_evil_dead = evaluator.add_leaf(
        id="professional_B_evil_dead_title",
        desc="The horror film title includes 'Evil Dead'",
        parent=horror_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The film title includes the exact phrase 'Evil Dead'.",
        node=node_evil_dead,
        sources=horror.urls if horror else [],
        additional_instruction="Accept titles such as 'Evil Dead Rise' or similar variants that clearly include 'Evil Dead'."
    )

    node_2023_release = evaluator.add_leaf(
        id="professional_B_2023_release",
        desc="The horror film was released in 2023",
        parent=horror_node,
        critical=True
    )
    await evaluator.verify(
        claim="This horror film was released in 2023.",
        node=node_2023_release,
        sources=horror.urls if horror else [],
        additional_instruction="Confirm the year-of-release is 2023."
    )

    # 6) Appeared in at least one additional 2024 feature film
    add24_node = evaluator.add_parallel(
        id="professional_B_additional_2024_film",
        desc="B also appeared in at least one additional feature film released in 2024",
        parent=prof_node,
        critical=True
    )
    add24 = data.additional_2024 if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=add24_node,
        leaf_id="professional_B_2024_film_url",
        desc="Provide URL reference confirming appearance in at least one film released in 2024",
        claim=f"This page confirms that {_safe_str(name)} appeared in at least one feature film released in 2024.",
        urls=add24.urls if add24 else [],
        add_ins="Confirm at least one credited role in a 2024 feature film.",
        critical=True
    )

    # 7) Series creators discovered performer via the 2023 horror film
    disc_node = evaluator.add_parallel(
        id="professional_B_casting_discovery",
        desc="The series creators first noticed the performer after seeing the 2023 horror film",
        parent=prof_node,
        critical=True
    )
    disc = data.casting_discovery if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=disc_node,
        leaf_id="professional_B_discovery_url",
        desc="Provide URL reference confirming how the series creators discovered the performer",
        claim="This page states that the series creators first noticed/considered casting the performer after seeing their work in the 2023 'Evil Dead' film.",
        urls=disc.urls if disc else [],
        add_ins="Look for creator/executive statements in interviews or press that cite the 2023 horror film as the discovery.",
        critical=True
    )


# -----------------------------------------------------------------------------
# Professional C verification
# -----------------------------------------------------------------------------
async def verify_professional_c(evaluator: Evaluator, parent, data: Optional[ProfessionalC]):
    prof_node = evaluator.add_parallel(
        id="professional_C",
        desc="Professional C: Born Feb 1992, younger sibling of franchise star; competed in 2025 social deduction and 2025 broadcast dancing (partner Daniella), reached semifinals; sibling attended Nov 19, 2025",
        parent=parent,
        critical=False
    )

    name = _safe_str(data.full_name if data else None)

    # 1) Born in February 1992
    birth_node = evaluator.add_parallel(
        id="professional_C_birth_february_1992",
        desc="C was born in February 1992",
        parent=prof_node,
        critical=True
    )
    birth = data.birth if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=birth_node,
        leaf_id="professional_C_birthdate_url",
        desc="Provide URL reference confirming the birth date",
        claim=f"This page confirms the birth date of {_safe_str(name)} and indicates February 1992.",
        urls=birth.urls if birth else [],
        add_ins="The page should show a birth date in February 1992.",
        critical=True
    )

    node_feb = evaluator.add_leaf(
        id="professional_C_february",
        desc="The birth month is February",
        parent=birth_node,
        critical=True
    )
    await evaluator.verify(
        claim="The birth month is February.",
        node=node_feb,
        sources=birth.urls if birth else [],
        additional_instruction="Confirm the month shown is February."
    )

    node_1992 = evaluator.add_leaf(
        id="professional_C_1992",
        desc="The birth year is 1992",
        parent=birth_node,
        critical=True
    )
    await evaluator.verify(
        claim="The birth year is 1992.",
        node=node_1992,
        sources=birth.urls if birth else [],
        additional_instruction="Confirm the year shown is 1992."
    )

    # 2) Younger sibling of actor who has starred in major film franchises
    sib_node = evaluator.add_parallel(
        id="professional_C_younger_sibling",
        desc="C is the younger sibling of a major-franchise film star",
        parent=prof_node,
        critical=True
    )
    sib = data.sibling if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=sib_node,
        leaf_id="professional_C_sibling_url",
        desc="Provide URL reference confirming the sibling relationship",
        claim=f"This page confirms that {_safe_str(name)} is a sibling of {_safe_str(sib.sibling_name if sib else None)}.",
        urls=sib.urls if sib else [],
        add_ins="Look for explicit family relationship statements.",
        critical=True
    )

    node_younger = evaluator.add_leaf(
        id="professional_C_younger",
        desc="Is the younger sibling (not the older one)",
        parent=sib_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_safe_str(name)} is the younger sibling relative to {_safe_str(sib.sibling_name if sib else None)}.",
        node=node_younger,
        sources=sib.urls if sib else [],
        additional_instruction="Accept explicit phrases like 'younger brother/sister'."
    )

    node_famous = evaluator.add_leaf(
        id="professional_C_famous_actor",
        desc="The sibling is an actor who has starred in major film franchises",
        parent=sib_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_safe_str(sib.sibling_name if sib else None)} is an actor who has starred in major film franchises.",
        node=node_famous,
        sources=sib.urls if sib else [],
        additional_instruction="Look for widely known franchises (e.g., MCU, Star Wars, Fast & Furious, etc.) credited as starring roles."
    )

    # 3) Competed in social deduction reality show (season premiering early 2025)
    ded_node = evaluator.add_parallel(
        id="professional_C_social_deduction_show",
        desc="C competed in a social deduction/strategy reality show with season premiering early 2025",
        parent=prof_node,
        critical=True
    )
    ded = data.social_deduction if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=ded_node,
        leaf_id="professional_C_deduction_show_url",
        desc="Provide URL reference confirming participation in the social deduction reality show",
        claim=f"This page confirms that {_safe_str(name)} competed in the social deduction/strategy reality show '{_safe_str(ded.show_name if ded else None)}'.",
        urls=ded.urls if ded else [],
        add_ins="Look for contestant lists or cast announcements.",
        critical=True
    )

    node_early25 = evaluator.add_leaf(
        id="professional_C_early_2025_premiere",
        desc="The season premiered in early 2025",
        parent=ded_node,
        critical=True
    )
    await evaluator.verify(
        claim="The relevant season premiered in early 2025 (roughly Jan–Apr 2025).",
        node=node_early25,
        sources=ded.urls if ded else [],
        additional_instruction="Confirm season premiere timing as early 2025."
    )

    # 4) Competed in a broadcast TV dancing competition in 2025
    dance_node = evaluator.add_parallel(
        id="professional_C_dancing_competition",
        desc="C competed in a broadcast-TV dancing competition in 2025",
        parent=prof_node,
        critical=True
    )
    dance = data.dancing if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=dance_node,
        leaf_id="professional_C_dancing_url",
        desc="Provide URL reference confirming participation in the dancing competition",
        claim=f"This page confirms that {_safe_str(name)} competed in the broadcast television dancing competition '{_safe_str(dance.show_name if dance else None)}'.",
        urls=dance.urls if dance else [],
        add_ins="Look for contestant lists, official cast reveals, or weekly recap pages.",
        critical=True
    )

    node_broadcast = evaluator.add_leaf(
        id="professional_C_broadcast_tv",
        desc="The dancing competition airs on broadcast television",
        parent=dance_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The dancing competition '{_safe_str(dance.show_name if dance else None)}' aired on broadcast television (major broadcast network).",
        node=node_broadcast,
        sources=dance.urls if dance else [],
        additional_instruction="Accept if the show is carried by a broadcast TV network (e.g., ABC, FOX, CBS, NBC, etc.)."
    )

    node_2025 = evaluator.add_leaf(
        id="professional_C_2025",
        desc="Competed during 2025",
        parent=dance_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_safe_str(name)} competed in the 2025 edition/season of the dancing competition.",
        node=node_2025,
        sources=dance.urls if dance else [],
        additional_instruction="Confirm the participation occurred during calendar year 2025."
    )

    # 5) Partnered with pro whose first name is Daniella
    partner_node = evaluator.add_parallel(
        id="professional_C_partner_daniella",
        desc="C was partnered with a professional dancer whose first name is Daniella",
        parent=prof_node,
        critical=True
    )
    await _add_url_verification_leaf(
        evaluator,
        parent=partner_node,
        leaf_id="professional_C_partner_url",
        desc="Provide URL reference confirming the partner's name",
        claim=f"This page confirms that {_safe_str(name)}'s dance partner's first name is Daniella.",
        urls=dance.urls if dance else [],
        add_ins="Look for the pro partner listing and name; first name should be Daniella.",
        critical=True
    )

    node_daniella = evaluator.add_leaf(
        id="professional_C_daniella",
        desc="The professional dancer's first name is Daniella",
        parent=partner_node,
        critical=True
    )
    await evaluator.verify(
        claim="The professional dance partner's first name is 'Daniella'.",
        node=node_daniella,
        sources=dance.urls if dance else [],
        additional_instruction="Minor spelling/case variants are okay if obviously referring to 'Daniella'."
    )

    # 6) Advanced to at least the semi-final
    semi_node = evaluator.add_parallel(
        id="professional_C_semifinal_advancement",
        desc="C advanced to at least the semi-final round of the dancing competition",
        parent=prof_node,
        critical=True
    )
    await _add_url_verification_leaf(
        evaluator,
        parent=semi_node,
        leaf_id="professional_C_semifinal_url",
        desc="Provide URL reference confirming advancement to at least the semi-finals",
        claim=f"This page confirms that {_safe_str(name)} advanced to at least the semi-final round of the dancing competition.",
        urls=dance.urls if dance else [],
        add_ins="Look for episode summaries or standings specifying semi-finalist status or beyond.",
        critical=True
    )

    # 7) Famous actor sibling attended in person on November 19, 2025
    attend_node = evaluator.add_parallel(
        id="professional_C_sibling_attendance",
        desc="The actor sibling attended the dancing competition on November 19, 2025",
        parent=prof_node,
        critical=True
    )
    att = data.sibling_attendance if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=attend_node,
        leaf_id="professional_C_attendance_url",
        desc="Provide URL reference confirming the sibling's attendance on November 19, 2025",
        claim="This page confirms that the famous actor sibling attended the dancing competition in person on November 19, 2025.",
        urls=att.urls if att else [],
        add_ins="Look for press, recaps, or social posts tied to Nov 19, 2025 attendance.",
        critical=True
    )

    node_nov19 = evaluator.add_leaf(
        id="professional_C_november_19",
        desc="The attendance was specifically on November 19, 2025",
        parent=attend_node,
        critical=True
    )
    await evaluator.verify(
        claim="The sibling attendance occurred specifically on November 19, 2025.",
        node=node_nov19,
        sources=att.urls if att else [],
        additional_instruction="Confirm the exact date as November 19, 2025."
    )


# -----------------------------------------------------------------------------
# Professional D verification
# -----------------------------------------------------------------------------
async def verify_professional_d(evaluator: Evaluator, parent, data: Optional[ProfessionalD]):
    prof_node = evaluator.add_parallel(
        id="professional_D",
        desc="Professional D: Born in 1950s; exactly 4 films in 2023–2024 incl. 2024 animated voice and a 2024 sequel to a 1988-origin franchise; at least one comedy; special streaming collection launched Feb 2026",
        parent=parent,
        critical=False
    )

    name = _safe_str(data.full_name if data else None)
    portfolio = data.portfolio if data else None
    films = portfolio.films_2023_2024 if portfolio else []
    film_urls_agg = _merge_urls(*(f.urls for f in films), (portfolio.urls if portfolio else []))

    # 1) Born during the 1950s (before 1960)
    birth50_node = evaluator.add_parallel(
        id="professional_D_birth_1950s",
        desc="D was born during the 1950s decade (before 1960)",
        parent=prof_node,
        critical=True
    )

    await _add_url_verification_leaf(
        evaluator,
        parent=birth50_node,
        leaf_id="professional_D_birthyear_url",
        desc="Provide URL reference confirming the birth year",
        claim=f"This page confirms the birth year of {_safe_str(name)}.",
        urls=data.birth_urls if data else [],
        add_ins="Verify the stated birth year on the page.",
        critical=True
    )

    node_1950s = evaluator.add_leaf(
        id="professional_D_1950s_decade",
        desc="The birth year is in the 1950s (1950-1959)",
        parent=birth50_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The birth year of {_safe_str(name)} falls within 1950–1959 inclusive.",
        node=node_1950s,
        sources=data.birth_urls if data else [],
        additional_instruction="If the on-page year is between 1950 and 1959, inclusive, mark correct."
    )

    # 2) Appeared in exactly four feature films released between 2023-01-01 and 2024-12-31
    four_node = evaluator.add_parallel(
        id="professional_D_four_films",
        desc="D appeared in exactly four feature films released between Jan 1, 2023 and Dec 31, 2024",
        parent=prof_node,
        critical=True
    )

    await _add_url_verification_leaf(
        evaluator,
        parent=four_node,
        leaf_id="professional_D_four_films_url",
        desc="Provide URL references confirming all four film appearances and their release dates",
        claim=f"The provided pages list four feature film appearances by {_safe_str(name)} during 2023–2024, with release dates.",
        urls=film_urls_agg,
        add_ins="Use the film pages/filmography to identify 4 films and their release years in 2023 or 2024.",
        critical=True
    )

    node_exactly4 = evaluator.add_leaf(
        id="professional_D_exactly_four",
        desc="The total is exactly 4 films, not more or fewer",
        parent=four_node,
        critical=True
    )
    # Build a readable film list for the claim
    film_titles_list = [f"- {(_safe_str(f.title))} ({_safe_str(f.release_date)})" for f in films]
    claim_exactly4 = (
        f"{_safe_str(name)} appeared in exactly four (4) feature films during 2023–2024:\n" +
        ("\n".join(film_titles_list) if film_titles_list else "No films listed in extraction.")
    )
    await evaluator.verify(
        claim=claim_exactly4,
        node=node_exactly4,
        sources=film_urls_agg,
        additional_instruction="Verify that there are precisely four distinct feature films credited to this performer in 2023–2024."
    )

    node_window = evaluator.add_leaf(
        id="professional_D_2023_2024_window",
        desc="All four films were released between January 1, 2023 and December 31, 2024",
        parent=four_node,
        critical=True
    )
    await evaluator.verify(
        claim="All of the four credited feature films were released within 2023-01-01 through 2024-12-31 inclusive.",
        node=node_window,
        sources=film_urls_agg,
        additional_instruction="Check each film page's release date; all must be in 2023 or 2024."
    )

    # 3) At least one animated 2024 film with voice work
    anim_node = evaluator.add_parallel(
        id="professional_D_animated_2024",
        desc="At least one of the four films is a 2024 animated feature with voice work by D",
        parent=prof_node,
        critical=True
    )

    # Try to pick an animated 2024 film from the extracted list
    animated_candidates = [f for f in films if (_safe_str(f.is_animated).lower() == "yes" or "animated" in _safe_str(f.primary_genre).lower()) and "2024" in _safe_str(f.release_date)]
    chosen_anim = animated_candidates[0] if animated_candidates else (films[0] if films else None)
    anim_urls = _nonempty_urls(chosen_anim.urls) if chosen_anim else film_urls_agg

    await _add_url_verification_leaf(
        evaluator,
        parent=anim_node,
        leaf_id="professional_D_animated_url",
        desc="Provide URL reference confirming appearance in an animated film",
        claim=f"This page confirms that {_safe_str(name)} appeared in an animated feature film.",
        urls=anim_urls,
        add_ins="Confirm the film is animated and the performer is in the cast.",
        critical=True
    )

    node_voice = evaluator.add_leaf(
        id="professional_D_voice_work",
        desc="The professional provided voice work for the animated film",
        parent=anim_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_safe_str(name)}'s role in this animated film is voice acting (voice role).",
        node=node_voice,
        sources=anim_urls,
        additional_instruction="Look for 'voice' credit on the cast listing."
    )

    node_anim_2024 = evaluator.add_leaf(
        id="professional_D_animated_2024",
        desc="The animated film was released in 2024",
        parent=anim_node,
        critical=True
    )
    await evaluator.verify(
        claim="The animated feature film was released in 2024.",
        node=node_anim_2024,
        sources=anim_urls,
        additional_instruction="Confirm year-of-release as 2024."
    )

    # 4) At least one sequel released in 2024 to a franchise whose original film was released in 1988
    seq_node = evaluator.add_parallel(
        id="professional_D_sequel_1988",
        desc="At least one of the films is a 2024 sequel to a franchise whose original film released in 1988",
        parent=prof_node,
        critical=True
    )

    sequel_candidates = [f for f in films if _safe_str(f.is_sequel).lower() == "yes" and "2024" in _safe_str(f.release_date)]
    chosen_sequel = sequel_candidates[0] if sequel_candidates else (films[0] if films else None)
    sequel_urls = _nonempty_urls(chosen_sequel.urls) if chosen_sequel else film_urls_agg

    await _add_url_verification_leaf(
        evaluator,
        parent=seq_node,
        leaf_id="professional_D_sequel_url",
        desc="Provide URL reference confirming appearance in the sequel film",
        claim=f"This page confirms that {_safe_str(name)} appeared in a 2024 sequel film.",
        urls=sequel_urls,
        add_ins="Confirm sequel status (an entry in an existing franchise) and the performer's role.",
        critical=True
    )

    node_seq_2024 = evaluator.add_leaf(
        id="professional_D_sequel_2024",
        desc="The sequel was released in 2024",
        parent=seq_node,
        critical=True
    )
    await evaluator.verify(
        claim="The sequel film in question was released in 2024.",
        node=node_seq_2024,
        sources=sequel_urls,
        additional_instruction="Confirm year-of-release for the sequel as 2024."
    )

    node_original_1988 = evaluator.add_leaf(
        id="professional_D_original_1988",
        desc="The original film in the franchise was released in 1988",
        parent=seq_node,
        critical=True
    )
    await evaluator.verify(
        claim="The franchise's original film was released in 1988.",
        node=node_original_1988,
        sources=sequel_urls,
        additional_instruction="Use franchise/series info on the page (or linked franchise summary) to confirm the original 1988 release."
    )

    # 5) Among the four films, at least one is primarily categorized as a comedy film
    com_node = evaluator.add_parallel(
        id="professional_D_comedy_film",
        desc="At least one of the four films is primarily a comedy",
        parent=prof_node,
        critical=True
    )

    comedy_candidates = [f for f in films if "comedy" in _safe_str(f.primary_genre).lower()]
    chosen_comedy = comedy_candidates[0] if comedy_candidates else (films[0] if films else None)
    comedy_urls = _nonempty_urls(chosen_comedy.urls) if chosen_comedy else film_urls_agg

    await _add_url_verification_leaf(
        evaluator,
        parent=com_node,
        leaf_id="professional_D_comedy_url",
        desc="Provide URL reference confirming appearance in a comedy film",
        claim=f"This page confirms that at least one of {_safe_str(name)}'s 2023–2024 films is primarily categorized as a comedy.",
        urls=comedy_urls,
        add_ins="Check the film's primary genre labeling for 'comedy'.",
        critical=True
    )

    # 6) Major streaming platform launched special collection in Feb 2026
    coll_node = evaluator.add_parallel(
        id="professional_D_streaming_collection",
        desc="A major streaming platform launched a special collection of D's works in February 2026",
        parent=prof_node,
        critical=True
    )
    coll = data.collection if data else None

    await _add_url_verification_leaf(
        evaluator,
        parent=coll_node,
        leaf_id="professional_D_collection_url",
        desc="Provide URL reference confirming the collection launch in February 2026",
        claim=f"A major streaming platform launched a special collection featuring works by {_safe_str(name)} in February 2026.",
        urls=coll.urls if coll else [],
        add_ins="Look for an official platform page or credible trade press announcing a curated collection in Feb 2026.",
        critical=True
    )

    node_feb26 = evaluator.add_leaf(
        id="professional_D_february_2026",
        desc="The collection was launched in February 2026",
        parent=coll_node,
        critical=True
    )
    await evaluator.verify(
        claim="The collection's launch date falls in February 2026.",
        node=node_feb26,
        sources=coll.urls if coll else [],
        additional_instruction="Confirm launch/announcement explicitly dated to February 2026."
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the entertainment professionals (2024–2026) task.
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

    # Extract all structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllProfessionalsExtraction,
        extraction_name="professionals_extraction"
    )

    # Build the verification tree according to rubric (root parallel over four professionals)
    # Note: We keep root as non-critical to satisfy the framework's strict critical-child constraint.
    # Each professional group internally marks critical leaves as required by the rubric.

    # Professional A
    await verify_professional_a(evaluator, root, extracted.professional_a if extracted else None)

    # Professional B
    await verify_professional_b(evaluator, root, extracted.professional_b if extracted else None)

    # Professional C
    await verify_professional_c(evaluator, root, extracted.professional_c if extracted else None)

    # Professional D
    await verify_professional_d(evaluator, root, extracted.professional_d if extracted else None)

    return evaluator.get_summary()