import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "taron_egerton_profile"
TASK_DESCRIPTION = (
    "Provide a comprehensive profile of Welsh actor Taron Egerton, including: "
    "(1) his complete biographical information (full name, birth date, birthplace, nationality, and height), "
    "(2) his educational background including the drama school attended, graduation year, and degree obtained, "
    "(3) details about his Golden Globe Award win including the category, film, and year won, along with general "
    "information about when Golden Globe ceremonies typically start and the traditional venue, "
    "(4) his role in the Kingsman film franchise including character name and the first film's title and year, "
    "(5) his portrayal in the biographical film Rocketman including the character portrayed and whether he performed his own singing, "
    "and (6) information about his 2024 Netflix thriller film including title, character name, and co-star. "
    "All information must be supported by credible reference URLs."
)

# Ground-truth values/expectations derived from the rubric (used for value compliance checks)
EXPECTED = {
    "birth_date": "November 10, 1989",
    "birth_place": "Birkenhead, Merseyside, England",
    "nationality": "Welsh",
    "height_options": [
        "5 ft 9 in", "5 feet 9 inches", "1.75 m", "1.75 meters", "175 cm"
    ],
    "drama_school_synonyms": [
        "Royal Academy of Dramatic Art", "RADA"
    ],
    "graduation_year": "2012",
    "degree_synonyms": [
        "BA (Hons) Acting", "Bachelor of Arts (Hons) in Acting", "BA (Hons) in Acting"
    ],
    "gg_category_synonyms": [
        "Best Actor in a Motion Picture - Musical or Comedy",
        "Best Actor in a Motion Picture – Musical or Comedy",
        "Best Performance by an Actor in a Motion Picture – Musical or Comedy"
    ],
    "gg_film": "Rocketman",
    "gg_edition": "77th",
    "gg_year": "2020",
    "gg_month": "January",
    "gg_start_time_variants": [
        "8:00 PM ET / 5:00 PM PT",
        "8 p.m. ET / 5 p.m. PT",
        "8 pm ET / 5 pm PT",
        "8 PM Eastern / 5 PM Pacific"
    ],
    "gg_venue_keywords": ["Beverly Hilton", "Beverly Hills, California"],
    "gg_venue_since_year": "1961",
    "kingsman_character_synonyms": [
        "Gary 'Eggsy' Unwin", "Gary “Eggsy” Unwin", "Gary Eggsy Unwin", "Eggsy"
    ],
    "kingsman_first_film_title": "Kingsman: The Secret Service",
    "kingsman_first_film_year": "2014",
    "rocketman_title": "Rocketman",
    "rocketman_year": "2019",
    "rocketman_character": "Elton John",
    "rocketman_singing": True,
    "netflix2024_title": "Carry-On",
    "netflix2024_character": "Ethan Kopek",
    "netflix2024_character_desc_keywords": ["TSA agent", "Transportation Security Administration"],
    "netflix2024_costar": "Jason Bateman",
}

CREDIBLE_DOMAIN_KEYWORDS = [
    "wikipedia.org",
    "imdb.com",
    "goldenglobes.com",
    "bbc.co.uk",
    "theguardian.com",
    "variety.com",
    "hollywoodreporter.com",
    "deadline.com",
    "netflix.com",
    "paramount.com",
    "universalpictures.com",
    "rottentomatoes.com",
    "empireonline.com",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BiographicalInfo(BaseModel):
    full_name: Optional[str] = None
    birth_date: Optional[str] = None
    birth_place: Optional[str] = None
    nationality: Optional[str] = None
    height: Optional[str] = None


class EducationInfo(BaseModel):
    drama_school: Optional[str] = None
    graduation_year: Optional[str] = None
    degree: Optional[str] = None


class GoldenGlobeWinInfo(BaseModel):
    category: Optional[str] = None
    film: Optional[str] = None
    ceremony_info: Optional[str] = None
    ceremony_number: Optional[str] = None  # e.g. "77th"
    win_month: Optional[str] = None  # e.g. "January"
    win_year: Optional[str] = None   # e.g. "2020"


class GoldenGlobesGeneralInfo(BaseModel):
    typical_start_time: Optional[str] = None
    traditional_venue: Optional[str] = None
    since_year: Optional[str] = None


class KingsmanInfo(BaseModel):
    character_name: Optional[str] = None
    first_film_title: Optional[str] = None
    first_film_year: Optional[str] = None


class RocketmanInfo(BaseModel):
    film_title: Optional[str] = None
    film_year: Optional[str] = None
    character_portrayed: Optional[str] = None
    performed_own_singing: Optional[str] = None  # yes/no/true/false


class NetflixThrillerInfo(BaseModel):
    film_title: Optional[str] = None
    character_name: Optional[str] = None
    character_description: Optional[str] = None
    co_star: Optional[str] = None


class ProfileExtraction(BaseModel):
    subject_name: Optional[str] = None
    subject_profession: Optional[str] = None  # e.g., "Welsh actor"

    bio: BiographicalInfo = BiographicalInfo()
    education: EducationInfo = EducationInfo()
    gg_win: GoldenGlobeWinInfo = GoldenGlobeWinInfo()
    gg_general: GoldenGlobesGeneralInfo = GoldenGlobesGeneralInfo()
    kingsman: KingsmanInfo = KingsmanInfo()
    rocketman: RocketmanInfo = RocketmanInfo()
    netflix2024: NetflixThrillerInfo = NetflixThrillerInfo()

    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_profile() -> str:
    return """
    Extract a structured profile of Taron Egerton from the provided answer. Return a single JSON object with these fields:
    - subject_name: The primary subject's name as presented.
    - subject_profession: The subject's profession as stated (e.g., "Welsh actor").
    - bio: {
        full_name: The subject's full name as stated in the answer,
        birth_date: The birth date as stated (string, keep the format used),
        birth_place: The birthplace as stated,
        nationality: The nationality as stated,
        height: The height as stated (free-form, e.g., "5 ft 9 in (1.75 m)")
      }
    - education: {
        drama_school: The drama school attended (as written, e.g., "Royal Academy of Dramatic Art (RADA)"),
        graduation_year: The graduation year (string),
        degree: The degree obtained (e.g., "BA (Hons) Acting")
      }
    - gg_win: {
        category: The Golden Globe category won (as written),
        film: The film for which the award was won,
        ceremony_info: Any description of the ceremony (free-form),
        ceremony_number: The ceremony edition in ordinal form if mentioned (e.g., "77th"),
        win_month: The month when the ceremony occurred if mentioned,
        win_year: The year when the award was won if mentioned
      }
    - gg_general: {
        typical_start_time: The typical start time of the Golden Globes as stated (e.g., "8:00 PM ET / 5:00 PM PT"),
        traditional_venue: The traditional venue as stated (e.g., "Beverly Hilton Hotel in Beverly Hills, California"),
        since_year: The year since it has been the traditional venue if provided (e.g., "1961")
      }
    - kingsman: {
        character_name: The character name he portrays in the Kingsman franchise,
        first_film_title: The title of the first film in which he appeared in the franchise,
        first_film_year: The year of that first film
      }
    - rocketman: {
        film_title: The biographical film title (e.g., "Rocketman"),
        film_year: The year of that film,
        character_portrayed: The character portrayed in the film,
        performed_own_singing: Whether he performed his own singing (yes/no/true/false as string)
      }
    - netflix2024: {
        film_title: The title of the 2024 Netflix thriller,
        character_name: The character name,
        character_description: The brief description of the character (e.g., "TSA agent"),
        co_star: A named co-star
      }
    - all_urls: An array containing ALL URLs mentioned anywhere in the answer (including any references section). Extract actual URLs even if they appear in markdown links.
    
    Rules:
    - Do not invent information. If an item is not present, set it to null.
    - Preserve the wording of the fields as presented in the answer.
    - Ensure 'all_urls' includes all URLs present in the answer (deduplicate if possible).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return s is not None and isinstance(s, str) and s.strip() != ""


def _lc(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _contains_any(s: Optional[str], keywords: List[str]) -> bool:
    sl = _lc(s)
    return any(k.lower() in sl for k in keywords)


def _height_matches_required(height: Optional[str]) -> bool:
    if not _nonempty(height):
        return False
    # Accept variants that indicate approximately 5'9" or 1.75 m (175 cm)
    patterns = EXPECTED["height_options"]
    return _contains_any(height, patterns)


def _is_affirmative(s: Optional[str]) -> bool:
    if not _nonempty(s):
        return False
    sl = _lc(s)
    return sl in {"yes", "true", "y", "t"} or "yes" in sl or "true" in sl


def _any_credible_url(urls: List[str]) -> bool:
    for u in urls:
        try:
            host = urlparse(u).netloc.lower()
        except Exception:
            host = ""
        if any(dom in host for dom in CREDIBLE_DOMAIN_KEYWORDS):
            return True
    return False


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if uu and uu not in seen:
            seen.add(uu)
            deduped.append(uu)
    return deduped


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_subject_identification(evaluator: Evaluator, parent, ex: ProfileExtraction, sources: List[str]) -> None:
    node = evaluator.add_parallel(
        id="SubjectIdentification",
        desc="Correctly identifies the subject as Taron Egerton and identifies him as an actor.",
        parent=parent,
        critical=True
    )

    # Existence check (subject name and profession mention)
    exists = _nonempty(ex.subject_name) and (_nonempty(ex.subject_profession) and ("actor" in _lc(ex.subject_profession)))
    evaluator.add_custom_node(
        result=exists,
        id="subject_exists",
        desc="Subject name present and identified as an actor in the answer",
        parent=node,
        critical=True
    )

    # Value check: subject name matches "Taron Egerton" (allow 'Taron David Egerton' as the full name)
    leaf_name_val = evaluator.add_leaf(
        id="subject_name_value",
        desc="Subject name corresponds to 'Taron Egerton'",
        parent=node,
        critical=True
    )
    claim_name = f"The extracted subject name '{ex.subject_name or ''}' refers to Taron Egerton (allowing the full name 'Taron David Egerton')."
    await evaluator.verify(
        claim=claim_name,
        node=leaf_name_val,
        additional_instruction="Consider it a match if the name clearly refers to the same person, allowing middle names and minor formatting differences."
    )

    # Source support: Taron Egerton is an actor
    leaf_actor_src = evaluator.add_leaf(
        id="subject_actor_supported",
        desc="Sources support that Taron Egerton is an actor",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton is an actor.",
        node=leaf_actor_src,
        sources=sources,
        additional_instruction="Verify that at least one cited webpage explicitly identifies Taron Egerton as an actor. Allow minor wording variants like 'Welsh actor'."
    )


async def build_biographical_information(evaluator: Evaluator, parent, ex: ProfileExtraction, sources: List[str]) -> None:
    node = evaluator.add_parallel(
        id="BiographicalInformation",
        desc="Basic biographical facts as constrained.",
        parent=parent,
        critical=True
    )

    # Gate: sources present for biographical info
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="bio_sources_present",
        desc="Sources are provided to support biographical information",
        parent=node,
        critical=True
    )

    # Full Name
    evaluator.add_custom_node(
        result=_nonempty(ex.bio.full_name),
        id="bio_fullname_exists",
        desc="Full name is provided",
        parent=node,
        critical=True
    )
    leaf_fullname_src = evaluator.add_leaf(
        id="bio_fullname_source",
        desc="Sources support the full name",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Taron Egerton's full name is '{ex.bio.full_name or ''}'.",
        node=leaf_fullname_src,
        sources=sources,
        additional_instruction="Verify that at least one cited webpage states or clearly implies this full name. Allow middle names (e.g., 'Taron David Egerton')."
    )

    # Birth Date
    evaluator.add_custom_node(
        result=_nonempty(ex.bio.birth_date),
        id="bio_birthdate_exists",
        desc="Birth date is provided",
        parent=node,
        critical=True
    )
    leaf_birthdate_val = evaluator.add_leaf(
        id="bio_birthdate_value",
        desc="Birth date equals November 10, 1989 (allowing day-month order)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"These two date expressions refer to the same date: '{ex.bio.birth_date or ''}' and '{EXPECTED['birth_date']}'.",
        node=leaf_birthdate_val,
        additional_instruction="Treat '10 November 1989' as equivalent to 'November 10, 1989'. Minor punctuation or ordinal suffixes are acceptable."
    )
    leaf_birthdate_src = evaluator.add_leaf(
        id="bio_birthdate_source",
        desc="Sources support the stated birth date",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton was born on November 10, 1989.",
        node=leaf_birthdate_src,
        sources=sources,
        additional_instruction="Verify at least one cited page states this birth date. Accept variants like '10 November 1989'."
    )

    # Birth Place
    evaluator.add_custom_node(
        result=_nonempty(ex.bio.birth_place),
        id="bio_birthplace_exists",
        desc="Birthplace is provided",
        parent=node,
        critical=True
    )
    leaf_birthplace_val = evaluator.add_leaf(
        id="bio_birthplace_value",
        desc="Birthplace equals Birkenhead, Merseyside, England (allowing minor formatting variants)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided birthplace '{ex.bio.birth_place or ''}' refers to 'Birkenhead, Merseyside, England'.",
        node=leaf_birthplace_val,
        additional_instruction="Allow variants like including 'UK' or commas/ordering differences as long as it clearly refers to Birkenhead, Merseyside, England."
    )
    leaf_birthplace_src = evaluator.add_leaf(
        id="bio_birthplace_source",
        desc="Sources support the stated birthplace",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton was born in Birkenhead, Merseyside, England.",
        node=leaf_birthplace_src,
        sources=sources,
        additional_instruction="Verify at least one cited page states this birthplace. Allow 'Birkenhead, Merseyside, England, UK' as equivalent."
    )

    # Nationality
    evaluator.add_custom_node(
        result=_nonempty(ex.bio.nationality),
        id="bio_nationality_exists",
        desc="Nationality is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_contains_any(ex.bio.nationality, ["Welsh"]),
        id="bio_nationality_value",
        desc="Nationality identified as Welsh",
        parent=node,
        critical=True
    )
    leaf_nationality_src = evaluator.add_leaf(
        id="bio_nationality_source",
        desc="Sources support the stated nationality",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton is Welsh.",
        node=leaf_nationality_src,
        sources=sources,
        additional_instruction="Verify at least one cited page describes him as Welsh (e.g., 'Welsh actor')."
    )

    # Height
    evaluator.add_custom_node(
        result=_nonempty(ex.bio.height),
        id="bio_height_exists",
        desc="Height is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_height_matches_required(ex.bio.height),
        id="bio_height_value",
        desc="Height stated as 5 ft 9 in or 1.75 m (approx. 175 cm)",
        parent=node,
        critical=True
    )
    leaf_height_src = evaluator.add_leaf(
        id="bio_height_source",
        desc="Sources support the stated height (~5 ft 9 in / 1.75 m)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton's height is about 5 ft 9 in (1.75 m).",
        node=leaf_height_src,
        sources=sources,
        additional_instruction="Verify at least one cited page indicates a height close to 5'9\" (1.75 m or 175 cm). Allow small rounding differences."
    )


async def build_education(evaluator: Evaluator, parent, ex: ProfileExtraction, sources: List[str]) -> None:
    node = evaluator.add_parallel(
        id="EducationalBackground",
        desc="Education details as constrained.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="edu_sources_present",
        desc="Sources are provided to support education details",
        parent=node,
        critical=True
    )

    # Drama School
    evaluator.add_custom_node(
        result=_nonempty(ex.education.drama_school),
        id="edu_school_exists",
        desc="Drama school is provided",
        parent=node,
        critical=True
    )
    leaf_school_val = evaluator.add_leaf(
        id="edu_school_value",
        desc="Drama school corresponds to Royal Academy of Dramatic Art (RADA)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided drama school '{ex.education.drama_school or ''}' refers to the Royal Academy of Dramatic Art (RADA).",
        node=leaf_school_val,
        additional_instruction="Consider it a match if it clearly refers to RADA or spelled-out Royal Academy of Dramatic Art."
    )
    leaf_school_src = evaluator.add_leaf(
        id="edu_school_source",
        desc="Sources support RADA attendance",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton attended the Royal Academy of Dramatic Art (RADA).",
        node=leaf_school_src,
        sources=sources,
        additional_instruction="Verify at least one cited page indicates he attended RADA."
    )

    # Graduation Year
    evaluator.add_custom_node(
        result=_nonempty(ex.education.graduation_year),
        id="edu_grad_exists",
        desc="Graduation year is provided",
        parent=node,
        critical=True
    )
    leaf_grad_val = evaluator.add_leaf(
        id="edu_grad_value",
        desc="Graduation year equals 2012",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided graduation year '{ex.education.graduation_year or ''}' equals '{EXPECTED['graduation_year']}'.",
        node=leaf_grad_val
    )
    leaf_grad_src = evaluator.add_leaf(
        id="edu_grad_source",
        desc="Sources support the graduation year",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton graduated in 2012.",
        node=leaf_grad_src,
        sources=sources,
        additional_instruction="Verify at least one cited page indicates 2012 as his graduation year."
    )

    # Degree
    evaluator.add_custom_node(
        result=_nonempty(ex.education.degree),
        id="edu_degree_exists",
        desc="Degree is provided",
        parent=node,
        critical=True
    )
    leaf_degree_val = evaluator.add_leaf(
        id="edu_degree_value",
        desc="Degree corresponds to BA (Hons) Acting",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided degree '{ex.education.degree or ''}' corresponds to 'BA (Hons) Acting' (or equivalent phrasing).",
        node=leaf_degree_val,
        additional_instruction="Accept equivalent phrasing such as 'Bachelor of Arts (Hons) in Acting'."
    )
    leaf_degree_src = evaluator.add_leaf(
        id="edu_degree_source",
        desc="Sources support the degree",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton obtained a BA (Hons) in Acting.",
        node=leaf_degree_src,
        sources=sources,
        additional_instruction="Verify at least one cited page indicates a BA (Hons) Acting (or equivalent phrasing)."
    )


async def build_golden_globe_win(evaluator: Evaluator, parent, ex: ProfileExtraction, sources: List[str]) -> None:
    node = evaluator.add_parallel(
        id="GoldenGlobeWin",
        desc="Golden Globe win details as constrained.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="ggwin_sources_present",
        desc="Sources are provided to support Golden Globe win",
        parent=node,
        critical=True
    )

    # Award Category
    evaluator.add_custom_node(
        result=_nonempty(ex.gg_win.category),
        id="gg_category_exists",
        desc="Golden Globe category is provided",
        parent=node,
        critical=True
    )
    leaf_category_val = evaluator.add_leaf(
        id="gg_category_value",
        desc="Category equals Best Actor in a Motion Picture - Musical or Comedy (allowing stylistic variants)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided category '{ex.gg_win.category or ''}' corresponds to 'Best Actor in a Motion Picture - Musical or Comedy'.",
        node=leaf_category_val,
        additional_instruction="Accept stylistic variants or en dash usage such as 'Best Performance by an Actor in a Motion Picture – Musical or Comedy'."
    )
    leaf_category_src = evaluator.add_leaf(
        id="gg_category_source",
        desc="Sources support the stated category for the win",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton won the Golden Globe for Best Actor in a Motion Picture – Musical or Comedy.",
        node=leaf_category_src,
        sources=sources,
        additional_instruction="Verify at least one cited page explicitly states this category was won by Taron Egerton."
    )

    # Award Film
    evaluator.add_custom_node(
        result=_nonempty(ex.gg_win.film),
        id="gg_film_exists",
        desc="Golden Globe winning film is provided",
        parent=node,
        critical=True
    )
    leaf_film_val = evaluator.add_leaf(
        id="gg_film_value",
        desc="Winning film identified as Rocketman",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided winning film '{ex.gg_win.film or ''}' equals 'Rocketman'.",
        node=leaf_film_val
    )
    leaf_film_src = evaluator.add_leaf(
        id="gg_film_source",
        desc="Sources support the stated winning film",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton won the Golden Globe for his role in Rocketman.",
        node=leaf_film_src,
        sources=sources,
        additional_instruction="Verify at least one cited page states Rocketman is the film for which he won."
    )

    # Ceremony and Date
    evaluator.add_custom_node(
        result=_nonempty(ex.gg_win.ceremony_info) or _nonempty(ex.gg_win.ceremony_number) or _nonempty(ex.gg_win.win_year),
        id="gg_ceremony_exists",
        desc="Golden Globe ceremony information is provided",
        parent=node,
        critical=True
    )
    leaf_ceremony_val = evaluator.add_leaf(
        id="gg_ceremony_value",
        desc="Win stated as at the 77th Golden Globe Awards in January 2020",
        parent=node,
        critical=True
    )
    edition = ex.gg_win.ceremony_number or ""
    year = ex.gg_win.win_year or ""
    month = ex.gg_win.win_month or ""
    claim_combo = f"The provided ceremony info indicates the 77th Golden Globe Awards in January 2020 (edition='{edition}', month='{month}', year='{year}')."
    await evaluator.verify(
        claim=claim_combo,
        node=leaf_ceremony_val,
        additional_instruction="Accept if the provided fields together clearly indicate 77th edition and January 2020."
    )
    leaf_ceremony_src = evaluator.add_leaf(
        id="gg_ceremony_source",
        desc="Sources support 77th Golden Globes in January 2020 for his win",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton's Golden Globe win was at the 77th Golden Globe Awards in January 2020.",
        node=leaf_ceremony_src,
        sources=sources,
        additional_instruction="Verify at least one cited page indicates the 77th edition and January 2020."
    )


async def build_golden_globes_general_info(evaluator: Evaluator, parent, ex: ProfileExtraction, sources: List[str]) -> None:
    node = evaluator.add_parallel(
        id="GoldenGlobesCeremonyGeneralInfo",
        desc="General Golden Globes ceremony info as constrained.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="gggen_sources_present",
        desc="Sources are provided to support general ceremony info",
        parent=node,
        critical=True
    )

    # Typical Start Time
    evaluator.add_custom_node(
        result=_nonempty(ex.gg_general.typical_start_time),
        id="gggen_start_exists",
        desc="Typical start time is provided",
        parent=node,
        critical=True
    )
    leaf_start_val = evaluator.add_leaf(
        id="gggen_start_value",
        desc="Typical start time equals 8:00 PM ET / 5:00 PM PT (allowing style variants)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided typical start time '{ex.gg_general.typical_start_time or ''}' corresponds to '8:00 PM ET / 5:00 PM PT' (or equivalent phrasing).",
        node=leaf_start_val,
        additional_instruction="Accept equivalences like '8 p.m. ET / 5 p.m. PT'."
    )
    leaf_start_src = evaluator.add_leaf(
        id="gggen_start_source",
        desc="Sources support the typical start time",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Golden Globes typically start at 8:00 PM ET / 5:00 PM PT.",
        node=leaf_start_src,
        sources=sources,
        additional_instruction="Verify that at least one cited page indicates this typical start time (accept minor formatting variants)."
    )

    # Traditional Venue Since 1961
    evaluator.add_custom_node(
        result=_nonempty(ex.gg_general.traditional_venue) or _nonempty(ex.gg_general.since_year),
        id="gggen_venue_exists",
        desc="Traditional venue/since-year info is provided",
        parent=node,
        critical=True
    )
    # Value check with custom boolean to ensure both elements are present/consistent
    venue_ok = _contains_any(ex.gg_general.traditional_venue, EXPECTED["gg_venue_keywords"]) and \
               (_lc(ex.gg_general.since_year) == _lc(EXPECTED["gg_venue_since_year"]))
    evaluator.add_custom_node(
        result=venue_ok,
        id="gggen_venue_value",
        desc="Venue identified as Beverly Hilton in Beverly Hills, since 1961",
        parent=node,
        critical=True
    )
    leaf_venue_src = evaluator.add_leaf(
        id="gggen_venue_source",
        desc="Sources support the traditional venue and since 1961",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Golden Globes are traditionally held at the Beverly Hilton Hotel in Beverly Hills, California, since 1961.",
        node=leaf_venue_src,
        sources=sources,
        additional_instruction="Verify at least one cited page indicates the Beverly Hilton in Beverly Hills and notes the tradition since 1961."
    )


async def build_kingsman_role(evaluator: Evaluator, parent, ex: ProfileExtraction, sources: List[str]) -> None:
    node = evaluator.add_parallel(
        id="KingsmanFranchiseRole",
        desc="Kingsman franchise role details as constrained.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="kingsman_sources_present",
        desc="Sources are provided to support Kingsman details",
        parent=node,
        critical=True
    )

    # Character Name
    evaluator.add_custom_node(
        result=_nonempty(ex.kingsman.character_name),
        id="kingsman_char_exists",
        desc="Kingsman character name is provided",
        parent=node,
        critical=True
    )
    leaf_char_val = evaluator.add_leaf(
        id="kingsman_char_value",
        desc="Character identified as Gary 'Eggsy' Unwin (allowing variants)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided Kingsman character '{ex.kingsman.character_name or ''}' refers to Gary 'Eggsy' Unwin.",
        node=leaf_char_val,
        additional_instruction="Accept variants including quotes or nickname-only 'Eggsy'."
    )
    leaf_char_src = evaluator.add_leaf(
        id="kingsman_char_source",
        desc="Sources support the Kingsman character name",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton plays Gary 'Eggsy' Unwin in the Kingsman franchise.",
        node=leaf_char_src,
        sources=sources,
        additional_instruction="Verify at least one cited page states he plays Gary 'Eggsy' Unwin."
    )

    # First Film Title and Year
    evaluator.add_custom_node(
        result=_nonempty(ex.kingsman.first_film_title) or _nonempty(ex.kingsman.first_film_year),
        id="kingsman_first_exists",
        desc="First Kingsman film title/year provided",
        parent=node,
        critical=True
    )
    leaf_first_val = evaluator.add_leaf(
        id="kingsman_first_value",
        desc="First film identified as Kingsman: The Secret Service (2014)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided first film '{ex.kingsman.first_film_title or ''}' ({ex.kingsman.first_film_year or ''}) corresponds to 'Kingsman: The Secret Service (2014)'.",
        node=leaf_first_val,
        additional_instruction="Allow minor punctuation differences, but the title and year should match."
    )
    leaf_first_src = evaluator.add_leaf(
        id="kingsman_first_source",
        desc="Sources support first Kingsman film title and year",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton first appeared as Eggsy in Kingsman: The Secret Service (2014).",
        node=leaf_first_src,
        sources=sources,
        additional_instruction="Verify at least one cited page states the first film and year."
    )


async def build_rocketman_portrayal(evaluator: Evaluator, parent, ex: ProfileExtraction, sources: List[str]) -> None:
    node = evaluator.add_parallel(
        id="RocketmanPortrayal",
        desc="Rocketman portrayal details as constrained.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="rocketman_sources_present",
        desc="Sources are provided to support Rocketman details",
        parent=node,
        critical=True
    )

    # Film Identification
    evaluator.add_custom_node(
        result=_nonempty(ex.rocketman.film_title) or _nonempty(ex.rocketman.film_year),
        id="rocketman_film_exists",
        desc="Rocketman film identification is provided",
        parent=node,
        critical=True
    )
    leaf_film_val = evaluator.add_leaf(
        id="rocketman_film_value",
        desc="Film identified as Rocketman (2019)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided Rocketman identification '{ex.rocketman.film_title or ''}' ({ex.rocketman.film_year or ''}) corresponds to 'Rocketman (2019)'.",
        node=leaf_film_val
    )
    leaf_film_src = evaluator.add_leaf(
        id="rocketman_film_source",
        desc="Sources support Rocketman (2019)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Rocketman (2019) is the biographical musical film starring Taron Egerton.",
        node=leaf_film_src,
        sources=sources,
        additional_instruction="Verify at least one cited page identifies Rocketman (2019)."
    )

    # Character Portrayed
    evaluator.add_custom_node(
        result=_nonempty(ex.rocketman.character_portrayed),
        id="rocketman_char_exists",
        desc="Rocketman character portrayed is provided",
        parent=node,
        critical=True
    )
    leaf_char_val = evaluator.add_leaf(
        id="rocketman_char_value",
        desc="Character portrayed identified as Elton John",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided character '{ex.rocketman.character_portrayed or ''}' corresponds to Elton John.",
        node=leaf_char_val
    )
    leaf_char_src = evaluator.add_leaf(
        id="rocketman_char_source",
        desc="Sources support that Egerton portrays Elton John",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In Rocketman, Taron Egerton portrays Elton John.",
        node=leaf_char_src,
        sources=sources,
        additional_instruction="Verify at least one cited page states he portrays Elton John."
    )

    # Performed Own Singing
    evaluator.add_custom_node(
        result=_nonempty(ex.rocketman.performed_own_singing),
        id="rocketman_sing_exists",
        desc="Whether he performed his own singing is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_affirmative(ex.rocketman.performed_own_singing),
        id="rocketman_sing_value",
        desc="States that Egerton performed his own singing in Rocketman",
        parent=node,
        critical=True
    )
    leaf_sing_src = evaluator.add_leaf(
        id="rocketman_sing_source",
        desc="Sources support that he performed his own singing",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton performed his own singing in Rocketman.",
        node=leaf_sing_src,
        sources=sources,
        additional_instruction="Verify at least one cited page explicitly states he performed his own singing."
    )


async def build_netflix_2024(evaluator: Evaluator, parent, ex: ProfileExtraction, sources: List[str]) -> None:
    node = evaluator.add_parallel(
        id="NetflixThriller2024",
        desc="Information about the 2024 Netflix thriller film as constrained.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="netflix_sources_present",
        desc="Sources are provided to support 2024 Netflix thriller details",
        parent=node,
        critical=True
    )

    # Film Title
    evaluator.add_custom_node(
        result=_nonempty(ex.netflix2024.film_title),
        id="netflix_title_exists",
        desc="2024 Netflix film title is provided",
        parent=node,
        critical=True
    )
    leaf_title_val = evaluator.add_leaf(
        id="netflix_title_value",
        desc="Film title identified as Carry-On",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided film title '{ex.netflix2024.film_title or ''}' equals 'Carry-On'.",
        node=leaf_title_val
    )
    leaf_title_src = evaluator.add_leaf(
        id="netflix_title_source",
        desc="Sources support the film title Carry-On",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Taron Egerton stars in the 2024 Netflix thriller 'Carry-On'.",
        node=leaf_title_src,
        sources=sources,
        additional_instruction="Verify at least one cited page mentions the title 'Carry-On' for the 2024 Netflix thriller."
    )

    # Character Name and Description
    evaluator.add_custom_node(
        result=_nonempty(ex.netflix2024.character_name) or _nonempty(ex.netflix2024.character_description),
        id="netflix_char_exists",
        desc="2024 Netflix character name/description is provided",
        parent=node,
        critical=True
    )
    name_ok = _contains_any(ex.netflix2024.character_name, [EXPECTED["netflix2024_character"]])
    desc_ok = _contains_any(ex.netflix2024.character_description, EXPECTED["netflix2024_character_desc_keywords"])
    evaluator.add_custom_node(
        result=(name_ok and desc_ok),
        id="netflix_char_value",
        desc="Character identified as Ethan Kopek, a TSA agent",
        parent=node,
        critical=True
    )
    leaf_char_src = evaluator.add_leaf(
        id="netflix_char_source",
        desc="Sources support character name and TSA agent description",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In Carry-On, Taron Egerton plays Ethan Kopek, a TSA agent.",
        node=leaf_char_src,
        sources=sources,
        additional_instruction="Verify at least one cited page mentions the character name Ethan Kopek and that he is a TSA agent."
    )

    # Co-star
    evaluator.add_custom_node(
        result=_nonempty(ex.netflix2024.co_star),
        id="netflix_costar_exists",
        desc="Co-star name is provided",
        parent=node,
        critical=True
    )
    leaf_costar_val = evaluator.add_leaf(
        id="netflix_costar_value",
        desc="Co-star identified as Jason Bateman",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided co-star '{ex.netflix2024.co_star or ''}' equals 'Jason Bateman'.",
        node=leaf_costar_val
    )
    leaf_costar_src = evaluator.add_leaf(
        id="netflix_costar_source",
        desc="Sources support the co-star Jason Bateman",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Jason Bateman is a co-star with Taron Egerton in Carry-On.",
        node=leaf_costar_src,
        sources=sources,
        additional_instruction="Verify at least one cited page lists Jason Bateman as a co-star in Carry-On."
    )


async def build_references(evaluator: Evaluator, parent, ex: ProfileExtraction) -> None:
    node = evaluator.add_parallel(
        id="References",
        desc="Provides credible reference URLs that collectively support the required facts.",
        parent=parent,
        critical=True
    )

    urls = _dedup_urls(ex.all_urls or [])

    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="refs_provided",
        desc="At least one reference URL is provided in the answer",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_any_credible_url(urls),
        id="refs_credible",
        desc="At least one reference URL is from a credible domain (e.g., Wikipedia, IMDb, Golden Globes, reputable news, Netflix)",
        parent=node,
        critical=True
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Taron Egerton comprehensive profile task.
    """
    # Initialize evaluator with a parallel root (children are critical and gate the overall score)
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

    # Extract structured profile from the answer
    extracted: ProfileExtraction = await evaluator.extract(
        prompt=prompt_extract_profile(),
        template_class=ProfileExtraction,
        extraction_name="taron_egerton_profile"
    )

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth({
        "expected": EXPECTED
    }, gt_type="rubric_expectations")

    # Prepare sources (all URLs from answer)
    sources_all: List[str] = _dedup_urls(extracted.all_urls or [])

    # Build verification subtrees (all critical under root)
    await build_subject_identification(evaluator, root, extracted, sources_all)
    await build_biographical_information(evaluator, root, extracted, sources_all)
    await build_education(evaluator, root, extracted, sources_all)
    await build_golden_globe_win(evaluator, root, extracted, sources_all)
    await build_golden_globes_general_info(evaluator, root, extracted, sources_all)
    await build_kingsman_role(evaluator, root, extracted, sources_all)
    await build_rocketman_portrayal(evaluator, root, extracted, sources_all)
    await build_netflix_2024(evaluator, root, extracted, sources_all)
    await build_references(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()