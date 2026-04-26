import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "2024_theatrical_milestones"
TASK_DESCRIPTION = """
A film industry database is compiling a comprehensive reference guide titled "2024's Most Significant Theatrical Milestones" featuring four distinct achievement categories. Your task is to identify one 2024 U.S. theatrical release for each of the four categories below, ensuring each film meets all specified criteria for its category.

Film #1 - Box Office Titan:
Identify a film that:
- Achieved worldwide box office gross exceeding $1 billion USD
- Was released in U.S. theaters between June 1, 2024 and August 31, 2024
- Has a theatrical runtime of at least 100 minutes
- Was distributed by a major Hollywood studio

Film #2 - Awards Heavyweight:
Identify a film that:
- Received at least 8 nominations at the 97th Academy Awards (2025 ceremony, honoring 2024 releases)
- Has a theatrical runtime of at least 135 minutes
- Was released in U.S. theaters between September 1, 2024 and December 31, 2024
- Is classified primarily as drama, thriller, or musical genre

Film #3 - Epic Scale Production:
Identify a film that:
- Has a theatrical runtime of at least 165 minutes
- Achieved worldwide box office gross exceeding $650 million USD
- Was released in U.S. theaters between February 1, 2024 and April 30, 2024
- Received at least 5 nominations at the 97th Academy Awards

Film #4 - Prestige Festival Winner:
Identify a film that:
- Won either the Palme d'Or at the 2024 Cannes Film Festival OR won Best Picture at the 97th Academy Awards (2025 ceremony)
- Has a theatrical runtime of at least 130 minutes
- Was released in U.S. theaters during calendar year 2024
- Has a director who received a Best Director nomination at the 97th Academy Awards

For each of the four films, provide the following information:
1. Complete official film title
2. Exact worldwide box office gross (in millions USD, to one decimal place)
3. Exact theatrical runtime (in minutes)
4. Number of Academy Award nominations received at the 97th Oscars
5. U.S. theatrical release date (in MM/DD/YYYY format)
6. Director's full name
7. Primary distributor/studio name
8. At least one reference URL from verified sources supporting the provided information
"""

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

MAJOR_STUDIOS = {
    "disney", "walt disney", "searchlight", "20th century", "20th century studios",
    "pixar",  # still Disney distributor for some releases
    "universal", "universal pictures", "focus features",
    "warner", "warner bros", "warner bros.", "warner bros pictures", "new line",
    "paramount", "paramount pictures",
    "sony", "sony pictures", "columbia", "tri-star", "tristar",
}

def is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    return url.startswith("http://") or url.startswith("https://")

def has_valid_sources(urls: List[str]) -> bool:
    return any(is_valid_url(u) for u in urls or [])

def parse_number(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s_clean = s.lower().replace(",", "").strip()
    # Extract first number
    m = re.search(r'(-?\d+(?:\.\d+)?)', s_clean)
    if not m:
        return None
    val = float(m.group(1))
    # Unit handling
    if "billion" in s_clean or "bn" in s_clean:
        return val * 1000.0  # convert to millions
    # If mentions "million" explicitly, keep val as millions
    return val

def parse_int_from_text(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    m = re.search(r'(\d+)', s.replace(",", ""))
    return int(m.group(1)) if m else None

def parse_runtime_minutes(s: Optional[str]) -> Optional[int]:
    return parse_int_from_text(s)

def parse_date_str(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    txt = s.strip()
    # Try MM/DD/YYYY
    try:
        return datetime.strptime(txt, "%m/%d/%Y")
    except Exception:
        pass
    # Try Month D, YYYY
    try:
        parts = txt.replace(",", "").split()
        if len(parts) >= 3:
            month_txt = parts[0].lower()
            day = int(parts[1])
            year = int(parts[2])
            month = MONTHS.get(month_txt)
            if month:
                return datetime(year, month, day)
    except Exception:
        pass
    return None

def in_date_range(date_val: Optional[datetime], start_mmddyyyy: str, end_mmddyyyy: str) -> bool:
    if date_val is None:
        return False
    start = datetime.strptime(start_mmddyyyy, "%m/%d/%Y")
    end = datetime.strptime(end_mmddyyyy, "%m/%d/%Y")
    return start <= date_val <= end

def is_major_studio(distributor: Optional[str]) -> bool:
    if not distributor:
        return False
    d = distributor.lower()
    return any(key in d for key in MAJOR_STUDIOS)

def primary_genre_is_in(gen: Optional[str], allowed: List[str]) -> bool:
    if not gen:
        return False
    g = gen.lower()
    return any(a in g for a in [v.lower() for v in allowed])

def truthy(s: Optional[str]) -> bool:
    if s is None:
        return False
    return s.strip().lower() in {"true", "yes", "y", "1"}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmEntry(BaseModel):
    title: Optional[str] = None
    worldwide_box_office_millions: Optional[str] = None  # in millions USD, to one decimal place
    runtime_minutes: Optional[str] = None                # minutes
    oscar_nominations_97th: Optional[str] = None         # count at 97th Oscars
    us_release_date: Optional[str] = None                # MM/DD/YYYY
    director: Optional[str] = None
    distributor: Optional[str] = None
    primary_genre: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    major_awards: List[str] = Field(default_factory=list)  # e.g., ["Palme d'Or", "Best Picture"]
    director_best_director_nomination_97th: Optional[str] = None  # "yes"/"no"/None
    sources: List[str] = Field(default_factory=list)

class MilestonesExtraction(BaseModel):
    film1: Optional[FilmEntry] = None
    film2: Optional[FilmEntry] = None
    film3: Optional[FilmEntry] = None
    film4: Optional[FilmEntry] = None

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_milestones() -> str:
    return """
    Extract up to four films presented in the answer, one per category (Film #1 Box Office Titan, Film #2 Awards Heavyweight, Film #3 Epic Scale Production, Film #4 Prestige Festival Winner).
    For each film, extract the following fields exactly as stated in the answer:
    - title: Complete official film title
    - worldwide_box_office_millions: Exact worldwide box office gross in millions USD, to one decimal place (e.g., "1100.5"); if expressed in billions (e.g., "1.10 billion"), convert to millions string to one decimal place if the answer already presents it that way; otherwise extract as given
    - runtime_minutes: Exact theatrical runtime in minutes (numeric or textual containing minutes)
    - oscar_nominations_97th: Number of Academy Award nominations at the 97th Oscars (2025 ceremony), extract the numeric count as presented
    - us_release_date: U.S. theatrical release date in MM/DD/YYYY format if provided; if presented in another format in the answer, extract that string as is
    - director: Director's full name
    - distributor: Primary distributor/studio name
    - primary_genre: The primary genre if the answer states one; otherwise null
    - genres: List of genres if multiple are mentioned
    - major_awards: List of major wins explicitly stated (e.g., "Palme d'Or", "Best Picture")
    - director_best_director_nomination_97th: "yes" if the director is stated as nominated for Best Director at the 97th Oscars, "no" if stated otherwise, null if not mentioned
    - sources: All reference URLs cited for this film; include every URL (plain or inside markdown links) that supports any of the above facts

    Map the extracted films to:
    - film1: The film claimed for Box Office Titan category
    - film2: The film claimed for Awards Heavyweight category
    - film3: The film claimed for Epic Scale Production category
    - film4: The film claimed for Prestige Festival Winner category

    If any field is missing in the answer for a film, set it to null or empty list accordingly.
    Ensure URLs are valid and complete. Do not invent information not in the answer.
    """

# --------------------------------------------------------------------------- #
# Common verification builder                                                 #
# --------------------------------------------------------------------------- #
async def add_common_leaves_and_verify(
    evaluator: Evaluator,
    film_node,
    entry: FilmEntry,
    prefix: str,
) -> None:
    # Title provided (critical)
    evaluator.add_custom_node(
        result=bool(entry.title and entry.title.strip()),
        id=f"{prefix}_Title_Provided",
        desc="The complete official film title is provided",
        parent=film_node,
        critical=True,
    )

    # Reference URL provided (critical)
    evaluator.add_custom_node(
        result=has_valid_sources(entry.sources),
        id=f"{prefix}_Reference_URL",
        desc="At least one valid reference URL from search results is provided that supports the film's information",
        parent=film_node,
        critical=True,
    )

    # Create leaves for accuracy verifications
    # Box office figure accurate
    boxoffice_leaf = evaluator.add_leaf(
        id=f"{prefix}_BoxOffice_Figure_Accurate",
        desc="The exact worldwide box office figure is provided in millions USD to one decimal place and matches verified sources",
        parent=film_node,
        critical=True,
    )
    boxoffice_claim = (
        f"The worldwide box office gross of '{entry.title}' is {entry.worldwide_box_office_millions} million USD."
        if entry.title and entry.worldwide_box_office_millions else
        f"The worldwide box office gross matches the figure stated in the answer for the film."
    )
    await evaluator.verify(
        claim=boxoffice_claim,
        node=boxoffice_leaf,
        sources=entry.sources,
        additional_instruction="Use cited sources (e.g., Box Office Mojo, The Numbers, press releases) to confirm the exact worldwide gross figure. Allow minor reporting variations within rounding tolerance, but the stated value should match the source."
    )

    # Runtime accurate
    runtime_leaf = evaluator.add_leaf(
        id=f"{prefix}_Runtime_Accurate",
        desc="The exact runtime in minutes is provided and matches verified sources",
        parent=film_node,
        critical=True,
    )
    runtime_claim = (
        f"The theatrical runtime of '{entry.title}' is {entry.runtime_minutes} minutes."
        if entry.title and entry.runtime_minutes else
        "The film's theatrical runtime matches what is stated in the answer."
    )
    await evaluator.verify(
        claim=runtime_claim,
        node=runtime_leaf,
        sources=entry.sources,
        additional_instruction="Confirm the official theatrical runtime in minutes from credible sources (e.g., studio sites, press kits, trade publications)."
    )

    # Oscar nominations accurate (97th)
    oscars_leaf = evaluator.add_leaf(
        id=f"{prefix}_Oscar_Nominations_Accurate",
        desc="The number of Academy Award nominations (97th Oscars) is provided accurately",
        parent=film_node,
        critical=True,
    )
    oscars_claim = (
        f"'{entry.title}' received {entry.oscar_nominations_97th} nominations at the 97th Academy Awards."
        if entry.title and entry.oscar_nominations_97th else
        "The film's nomination count at the 97th Academy Awards matches what is stated in the answer."
    )
    await evaluator.verify(
        claim=oscars_claim,
        node=oscars_leaf,
        sources=entry.sources,
        additional_instruction="Check the official Academy site or reputable publications to confirm the film's total nominations count at the 97th Oscars."
    )

    # U.S. release date accurate
    release_leaf = evaluator.add_leaf(
        id=f"{prefix}_Release_Date_Accurate",
        desc="The U.S. theatrical release date is provided in correct format and matches verified sources",
        parent=film_node,
        critical=True,
    )
    release_claim = (
        f"The U.S. theatrical release date of '{entry.title}' was {entry.us_release_date}."
        if entry.title and entry.us_release_date else
        "The U.S. theatrical release date stated in the answer matches credible sources."
    )
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        sources=entry.sources,
        additional_instruction="Confirm the U.S. theatrical release date. Accept equivalent date formats if they represent the same calendar date."
    )

    # Director name correct
    director_leaf = evaluator.add_leaf(
        id=f"{prefix}_Director_Name_Correct",
        desc="The director's full name is provided correctly",
        parent=film_node,
        critical=True,
    )
    director_claim = (
        f"The director of '{entry.title}' is {entry.director}."
        if entry.title and entry.director else
        "The director name matches what is stated in the answer."
    )
    await evaluator.verify(
        claim=director_claim,
        node=director_leaf,
        sources=entry.sources,
        additional_instruction="Verify the film's director via credible sources (e.g., studio site, Academy, trade press). Allow minor name formatting variations."
    )

    # Distributor/studio correct
    distributor_leaf = evaluator.add_leaf(
        id=f"{prefix}_Distributor_Name_Correct",
        desc="The primary distributor/studio name is provided correctly and matches verified sources",
        parent=film_node,
        critical=True,
    )
    distributor_claim = (
        f"The primary distributor/studio for '{entry.title}' is {entry.distributor}."
        if entry.title and entry.distributor else
        "The primary distributor/studio matches what is stated in the answer."
    )
    await evaluator.verify(
        claim=distributor_claim,
        node=distributor_leaf,
        sources=entry.sources,
        additional_instruction="Confirm the film's primary distributor/studio from credible sources. Subsidiaries (e.g., Searchlight, Focus Features) are acceptable as distributors."
    )

# --------------------------------------------------------------------------- #
# Category-specific verification functions                                    #
# --------------------------------------------------------------------------- #
async def verify_film_1_box_office_titan(evaluator: Evaluator, root_node, entry: FilmEntry) -> None:
    film_node = evaluator.add_parallel(
        id="Film_1_Box_Office_Titan",
        desc="Identify a 2024 film that achieved box office titan status (>$1B worldwide, Jun-Aug 2024 release)",
        parent=root_node,
        critical=False,
    )

    # Threshold checks (critical custom nodes)
    gross_mil = parse_number(entry.worldwide_box_office_millions)
    evaluator.add_custom_node(
        result=(gross_mil is not None and gross_mil > 1000.0),
        id="Film1_Worldwide_BoxOffice_Exceeds_1B",
        desc="The film's worldwide box office gross exceeds $1 billion USD",
        parent=film_node,
        critical=True,
    )

    rel_date = parse_date_str(entry.us_release_date)
    evaluator.add_custom_node(
        result=in_date_range(rel_date, "06/01/2024", "08/31/2024"),
        id="Film1_US_Release_June_Aug_2024",
        desc="The film was released in U.S. theaters between June 1, 2024 and August 31, 2024",
        parent=film_node,
        critical=True,
    )

    runtime_min = parse_runtime_minutes(entry.runtime_minutes)
    evaluator.add_custom_node(
        result=(runtime_min is not None and runtime_min >= 100),
        id="Film1_Runtime_At_Least_100_Minutes",
        desc="The film's theatrical runtime is at least 100 minutes",
        parent=film_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=is_major_studio(entry.distributor),
        id="Film1_Major_Studio_Distributor",
        desc="The film was distributed by a major Hollywood studio (e.g., Disney, Universal, Warner Bros., Paramount, Sony, 20th Century)",
        parent=film_node,
        critical=True,
    )

    # Common leaves & verifications
    await add_common_leaves_and_verify(evaluator, film_node, entry, "Film1")


async def verify_film_2_awards_heavyweight(evaluator: Evaluator, root_node, entry: FilmEntry) -> None:
    film_node = evaluator.add_parallel(
        id="Film_2_Awards_Heavyweight",
        desc="Identify a 2024 film that achieved awards heavyweight status (≥8 Oscar nominations, Sep-Dec 2024 release)",
        parent=root_node,
        critical=False,
    )

    nominations = parse_int_from_text(entry.oscar_nominations_97th)
    evaluator.add_custom_node(
        result=(nominations is not None and nominations >= 8),
        id="Film2_Oscar_Nominations_At_Least_8",
        desc="The film received at least 8 nominations at the 97th Academy Awards",
        parent=film_node,
        critical=True,
    )

    runtime_min = parse_runtime_minutes(entry.runtime_minutes)
    evaluator.add_custom_node(
        result=(runtime_min is not None and runtime_min >= 135),
        id="Film2_Runtime_At_Least_135_Minutes",
        desc="The film's theatrical runtime is at least 135 minutes",
        parent=film_node,
        critical=True,
    )

    rel_date = parse_date_str(entry.us_release_date)
    evaluator.add_custom_node(
        result=in_date_range(rel_date, "09/01/2024", "12/31/2024"),
        id="Film2_US_Release_Sept_Dec_2024",
        desc="The film was released in U.S. theaters between September 1, 2024 and December 31, 2024",
        parent=film_node,
        critical=True,
    )

    # Genre check: primary genre is drama/thriller/musical
    primary = entry.primary_genre or (entry.genres[0] if entry.genres else None)
    evaluator.add_custom_node(
        result=primary_genre_is_in(primary, ["Drama", "Thriller", "Musical"]),
        id="Film2_Genre_Drama_Thriller_Musical",
        desc="The film is classified primarily as drama, thriller, or musical genre",
        parent=film_node,
        critical=True,
    )

    await add_common_leaves_and_verify(evaluator, film_node, entry, "Film2")


async def verify_film_3_epic_scale(evaluator: Evaluator, root_node, entry: FilmEntry) -> None:
    film_node = evaluator.add_parallel(
        id="Film_3_Epic_Scale_Production",
        desc="Identify a 2024 film that represents epic scale production (≥165 min runtime, >$650M box office, Feb-Apr 2024 release)",
        parent=root_node,
        critical=False,
    )

    runtime_min = parse_runtime_minutes(entry.runtime_minutes)
    evaluator.add_custom_node(
        result=(runtime_min is not None and runtime_min >= 165),
        id="Film3_Runtime_At_Least_165_Minutes",
        desc="The film's theatrical runtime is at least 165 minutes",
        parent=film_node,
        critical=True,
    )

    gross_mil = parse_number(entry.worldwide_box_office_millions)
    evaluator.add_custom_node(
        result=(gross_mil is not None and gross_mil > 650.0),
        id="Film3_Worldwide_BoxOffice_Exceeds_650M",
        desc="The film's worldwide box office gross exceeds $650 million USD",
        parent=film_node,
        critical=True,
    )

    rel_date = parse_date_str(entry.us_release_date)
    evaluator.add_custom_node(
        result=in_date_range(rel_date, "02/01/2024", "04/30/2024"),
        id="Film3_US_Release_Feb_Apr_2024",
        desc="The film was released in U.S. theaters between February 1, 2024 and April 30, 2024",
        parent=film_node,
        critical=True,
    )

    nominations = parse_int_from_text(entry.oscar_nominations_97th)
    evaluator.add_custom_node(
        result=(nominations is not None and nominations >= 5),
        id="Film3_Oscar_Nominations_At_Least_5",
        desc="The film received at least 5 nominations at the 97th Academy Awards",
        parent=film_node,
        critical=True,
    )

    await add_common_leaves_and_verify(evaluator, film_node, entry, "Film3")


async def verify_film_4_prestige_winner(evaluator: Evaluator, root_node, entry: FilmEntry) -> None:
    film_node = evaluator.add_parallel(
        id="Film_4_Prestige_Festival_Winner",
        desc="Identify a 2024 film that achieved prestige status through major awards (Palme d'Or or Best Picture winner, director Oscar-nominated)",
        parent=root_node,
        critical=False,
    )

    # Award win verification (critical leaf, source-grounded)
    award_leaf = evaluator.add_leaf(
        id="Film4_Won_Palme_Or_Or_Best_Picture",
        desc="The film won either the Palme d'Or at the 2024 Cannes Film Festival OR won Best Picture at the 97th Academy Awards",
        parent=film_node,
        critical=True,
    )
    awards_lower = [a.lower() for a in entry.major_awards or []]
    won_best_picture = any("best picture" in a for a in awards_lower)
    won_palme = any("palme" in a and "or" not in a for a in awards_lower) or any("palme d'or" in a for a in awards_lower)
    award_claim = ""
    if won_best_picture and entry.title:
        award_claim = f"'{entry.title}' won Best Picture at the 97th Academy Awards."
    elif won_palme and entry.title:
        award_claim = f"'{entry.title}' won the Palme d'Or at the 2024 Cannes Film Festival."
    else:
        # Generalized claim if extraction didn't specify which of the two
        award_claim = "The film won either the Palme d'Or at the 2024 Cannes Film Festival or Best Picture at the 97th Academy Awards."
    await evaluator.verify(
        claim=award_claim,
        node=award_leaf,
        sources=entry.sources,
        additional_instruction="Confirm from credible sources (e.g., official Cannes, Academy site, reputable press) that the film won the specified top award."
    )

    # Runtime threshold
    runtime_min = parse_runtime_minutes(entry.runtime_minutes)
    evaluator.add_custom_node(
        result=(runtime_min is not None and runtime_min >= 130),
        id="Film4_Runtime_At_Least_130_Minutes",
        desc="The film's theatrical runtime is at least 130 minutes",
        parent=film_node,
        critical=True,
    )

    # US Release in 2024
    rel_date = parse_date_str(entry.us_release_date)
    evaluator.add_custom_node(
        result=(rel_date is not None and rel_date.year == 2024),
        id="Film4_US_Release_2024",
        desc="The film was released in U.S. theaters during calendar year 2024",
        parent=film_node,
        critical=True,
    )

    # Director Best Director nomination threshold
    evaluator.add_custom_node(
        result=truthy(entry.director_best_director_nomination_97th),
        id="Film4_Director_Oscar_Nominated",
        desc="The film's director received a Best Director nomination at the 97th Academy Awards",
        parent=film_node,
        critical=True,
    )

    # Common details accuracy
    await add_common_leaves_and_verify(evaluator, film_node, entry, "Film4")

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
    """
    Evaluate an answer for the 2024 Theatrical Milestones task.
    """
    # Initialize evaluator (root parallel for four independent categories)
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

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_milestones(),
        template_class=MilestonesExtraction,
        extraction_name="milestones_extraction",
    )

    # Normalize film entries (pad with empty if None)
    film1 = extraction.film1 or FilmEntry()
    film2 = extraction.film2 or FilmEntry()
    film3 = extraction.film3 or FilmEntry()
    film4 = extraction.film4 or FilmEntry()

    # Build verification subtrees for each category
    await verify_film_1_box_office_titan(evaluator, root, film1)
    await verify_film_2_awards_heavyweight(evaluator, root, film2)
    await verify_film_3_epic_scale(evaluator, root, film3)
    await verify_film_4_prestige_winner(evaluator, root, film4)

    # Return evaluation summary
    return evaluator.get_summary()