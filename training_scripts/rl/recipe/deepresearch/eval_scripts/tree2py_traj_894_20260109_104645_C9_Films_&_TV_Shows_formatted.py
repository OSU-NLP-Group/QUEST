import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "award_productions_2024"
TASK_DESCRIPTION = (
    "Identify the four specified 2024 award-winning productions and provide the required verified attributes for each: "
    "1) Best Picture at the 96th Academy Awards, 2) Palme d'Or at the 77th Cannes Film Festival, "
    "3) Outstanding Drama Series at the 76th Emmy Awards, 4) Outstanding Limited or Anthology Series at the 76th Emmy Awards. "
    "For each, include Award Info (category, ceremony/festival name, award date, verification URL), "
    "Director/Creator Info with supporting URL, Production Details with supporting database URL, "
    "Cast information with supporting URL, and optionally Critical Reception with rating source URL."
)


# ----------------------------- Data Models --------------------------------- #

class AwardInfo(BaseModel):
    category: Optional[str] = None
    ceremony_name: Optional[str] = None
    award_date: Optional[str] = None
    award_url: Optional[str] = None


class FilmDirectorInfo(BaseModel):
    name: Optional[str] = None
    best_director_status_note: Optional[str] = None  # "won", "not_won", or "unknown"
    info_url: Optional[str] = None


class TVCreatorInfo(BaseModel):
    name: Optional[str] = None
    info_url: Optional[str] = None


class ProductionDetailsFilm(BaseModel):
    production_companies: List[str] = Field(default_factory=list)
    distributor: Optional[str] = None
    release_date: Optional[str] = None
    runtime: Optional[str] = None
    details_url: Optional[str] = None


class ProductionDetailsTV(BaseModel):
    production_companies: List[str] = Field(default_factory=list)
    platform: Optional[str] = None
    premiere_date: Optional[str] = None
    total_episodes: Optional[str] = None
    details_url: Optional[str] = None


class CastBestPicture(BaseModel):
    best_actor_name: Optional[str] = None
    best_actor_character: Optional[str] = None
    supporting_actor_name: Optional[str] = None
    supporting_actor_character: Optional[str] = None
    cast_url: Optional[str] = None


class CastPalmeDor(BaseModel):
    lead_actress_name: Optional[str] = None
    lead_actress_character: Optional[str] = None
    other_cast_name: Optional[str] = None
    other_cast_role: Optional[str] = None
    cast_url: Optional[str] = None


class CastEmmyDrama(BaseModel):
    main_cast_1_name: Optional[str] = None
    main_cast_1_character: Optional[str] = None
    main_cast_2_name: Optional[str] = None
    main_cast_2_character: Optional[str] = None
    cast_url: Optional[str] = None


class CastEmmyLimited(BaseModel):
    lead_actor_name: Optional[str] = None
    lead_actor_character: Optional[str] = None
    other_cast_name: Optional[str] = None
    other_cast_role: Optional[str] = None
    cast_url: Optional[str] = None


class RatingInfo(BaseModel):
    rotten_tomatoes_score: Optional[str] = None
    imdb_rating: Optional[str] = None
    rating_url: Optional[str] = None


class BestPictureItem(BaseModel):
    title: Optional[str] = None
    award: Optional[AwardInfo] = None
    director: Optional[FilmDirectorInfo] = None
    production: Optional[ProductionDetailsFilm] = None
    cast: Optional[CastBestPicture] = None
    reception: Optional[RatingInfo] = None


class PalmeDorItem(BaseModel):
    title: Optional[str] = None
    award: Optional[AwardInfo] = None
    director: Optional[FilmDirectorInfo] = None
    production: Optional[ProductionDetailsFilm] = None
    cast: Optional[CastPalmeDor] = None
    reception: Optional[RatingInfo] = None


class EmmyDramaItem(BaseModel):
    title: Optional[str] = None
    award: Optional[AwardInfo] = None
    creator: Optional[TVCreatorInfo] = None
    production: Optional[ProductionDetailsTV] = None
    cast: Optional[CastEmmyDrama] = None
    reception: Optional[RatingInfo] = None


class EmmyLimitedItem(BaseModel):
    title: Optional[str] = None
    award: Optional[AwardInfo] = None
    creator: Optional[TVCreatorInfo] = None
    production: Optional[ProductionDetailsTV] = None
    cast: Optional[CastEmmyLimited] = None
    reception: Optional[RatingInfo] = None


# --------------------------- Extraction Prompts ----------------------------- #

def prompt_extract_best_picture() -> str:
    return (
        "Extract the film that won Best Picture at the 96th Academy Awards and all requested attributes. "
        "Return a JSON object with the structure:\n"
        "{\n"
        '  "title": string or null,\n'
        '  "award": {\n'
        '    "category": string or null,  // e.g., "Best Picture"\n'
        '    "ceremony_name": string or null,  // e.g., "96th Academy Awards"\n'
        '    "award_date": string or null,  // date string as mentioned in the answer\n'
        '    "award_url": string or null    // URL from oscars.org or reputable source explicitly confirming the win\n'
        "  },\n"
        '  "director": {\n'
        '    "name": string or null,\n'
        '    "best_director_status_note": string or null,  // "won", "not_won", or "unknown" depending on what the answer states\n'
        '    "info_url": string or null   // URL confirming director information (and Best Director status if claimed)\n'
        "  },\n"
        '  "production": {\n'
        '    "production_companies": [strings],\n'
        '    "distributor": string or null,\n'
        '    "release_date": string or null,  // theatrical release date\n'
        '    "runtime": string or null,       // e.g., "2h 31m"\n'
        '    "details_url": string or null    // IMDb or reputable database URL supporting production details\n'
        "  },\n"
        '  "cast": {\n'
        '    "best_actor_name": string or null,\n'
        '    "best_actor_character": string or null,\n'
        '    "supporting_actor_name": string or null,\n'
        '    "supporting_actor_character": string or null,\n'
        '    "cast_url": string or null      // URL confirming cast and roles\n'
        "  },\n"
        '  "reception": {\n'
        '    "rotten_tomatoes_score": string or null,\n'
        '    "imdb_rating": string or null,\n'
        '    "rating_url": string or null    // URL from rating source\n'
        "  }\n"
        "If any field is missing in the answer, set it to null. Extract only URLs explicitly present."
    )


def prompt_extract_palme_dor() -> str:
    return (
        "Extract the film that won the Palme d'Or at the 77th Cannes Film Festival and all requested attributes. "
        "Return a JSON object with the structure:\n"
        "{\n"
        '  "title": string or null,\n'
        '  "award": {\n'
        '    "category": string or null,  // e.g., "Palme d\'Or"\n'
        '    "ceremony_name": string or null,  // e.g., "77th Cannes Film Festival"\n'
        '    "award_date": string or null,\n'
        '    "award_url": string or null    // URL from festival-cannes.com or reputable source confirming the win\n'
        "  },\n"
        '  "director": {\n'
        '    "name": string or null,\n'
        '    "best_director_status_note": string or null,  // usually not applicable; set "unknown" if not stated\n'
        '    "info_url": string or null\n'
        "  },\n"
        '  "production": {\n'
        '    "production_companies": [strings],\n'
        '    "distributor": string or null,\n'
        '    "release_date": string or null,\n'
        '    "runtime": string or null,\n'
        '    "details_url": string or null\n'
        "  },\n"
        '  "cast": {\n'
        '    "lead_actress_name": string or null,\n'
        '    "lead_actress_character": string or null,\n'
        '    "other_cast_name": string or null,\n'
        '    "other_cast_role": string or null,\n'
        '    "cast_url": string or null\n'
        "  },\n"
        '  "reception": {\n'
        '    "rotten_tomatoes_score": string or null,\n'
        '    "imdb_rating": string or null,\n'
        '    "rating_url": string or null\n'
        "  }\n"
        "Set missing fields to null and include only URLs explicitly present."
    )


def prompt_extract_emmy_drama() -> str:
    return (
        "Extract the TV series that won Outstanding Drama Series at the 76th Emmy Awards and all requested attributes. "
        "Return a JSON object with the structure:\n"
        "{\n"
        '  "title": string or null,\n'
        '  "award": {\n'
        '    "category": string or null,  // e.g., "Outstanding Drama Series"\n'
        '    "ceremony_name": string or null,  // e.g., "76th Emmy Awards"\n'
        '    "award_date": string or null,\n'
        '    "award_url": string or null   // URL from televisionacademy.com or reputable source confirming the win\n'
        "  },\n"
        '  "creator": {\n'
        '    "name": string or null,       // creator/showrunner/primary writer\n'
        '    "info_url": string or null\n'
        "  },\n"
        '  "production": {\n'
        '    "production_companies": [strings],\n'
        '    "platform": string or null,   // network/streaming platform\n'
        '    "premiere_date": string or null,\n'
        '    "total_episodes": string or null,\n'
        '    "details_url": string or null\n'
        "  },\n"
        '  "cast": {\n'
        '    "main_cast_1_name": string or null,\n'
        '    "main_cast_1_character": string or null,\n'
        '    "main_cast_2_name": string or null,\n'
        '    "main_cast_2_character": string or null,\n'
        '    "cast_url": string or null\n'
        "  },\n"
        '  "reception": {\n'
        '    "rotten_tomatoes_score": string or null,\n'
        '    "imdb_rating": string or null,\n'
        '    "rating_url": string or null\n'
        "  }\n"
        "Set missing fields to null and include only URLs explicitly present."
    )


def prompt_extract_emmy_limited() -> str:
    return (
        "Extract the TV series that won Outstanding Limited or Anthology Series at the 76th Emmy Awards and all requested attributes. "
        "Return a JSON object with the structure:\n"
        "{\n"
        '  "title": string or null,\n'
        '  "award": {\n'
        '    "category": string or null,  // e.g., "Outstanding Limited or Anthology Series"\n'
        '    "ceremony_name": string or null,  // e.g., "76th Emmy Awards"\n'
        '    "award_date": string or null,\n'
        '    "award_url": string or null   // URL from televisionacademy.com or reputable source confirming the win\n'
        "  },\n"
        '  "creator": {\n'
        '    "name": string or null,\n'
        '    "info_url": string or null\n'
        "  },\n"
        '  "production": {\n'
        '    "production_companies": [strings],\n'
        '    "platform": string or null,   // network/streaming platform\n'
        '    "premiere_date": string or null,  // premiere or release date\n'
        '    "total_episodes": string or null,\n'
        '    "details_url": string or null\n'
        "  },\n"
        '  "cast": {\n'
        '    "lead_actor_name": string or null,         // lead actor who also won an acting Emmy for this series\n'
        '    "lead_actor_character": string or null,\n'
        '    "other_cast_name": string or null,\n'
        '    "other_cast_role": string or null,\n'
        '    "cast_url": string or null\n'
        "  },\n"
        '  "reception": {\n'
        '    "rotten_tomatoes_score": string or null,\n'
        '    "imdb_rating": string or null,\n'
        '    "rating_url": string or null\n'
        "  }\n"
        "Set missing fields to null and include only URLs explicitly present."
    )


# ----------------------------- Helper Functions ---------------------------- #

def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip() != "")


def _url_exists(evaluator: Evaluator, url: Optional[str], node_id: str, desc: str, parent, critical: bool = True):
    return evaluator.add_custom_node(
        result=_non_empty_str(url),
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )


def _list_to_readable(items: List[str]) -> str:
    cleaned = [x.strip() for x in items if _non_empty_str(x)]
    if not cleaned:
        return ""
    return ", ".join(cleaned)


# ----------------------------- Verification Logic -------------------------- #

async def verify_best_picture_item(evaluator: Evaluator, root, bp: BestPictureItem):
    item_node = evaluator.add_parallel(
        id="Item_1_Best_Picture_96th_Academy_Awards",
        desc="Film that won Best Picture at the 96th Academy Awards",
        parent=root,
        critical=False
    )

    title = bp.title or "Unknown Title"

    # Award Information (critical)
    award_info_node = evaluator.add_parallel(
        id="Item_1_Award_Information",
        desc="Provide award category, ceremony/festival name, award date, and a supporting URL",
        parent=item_node,
        critical=True
    )
    award_url_exists = _url_exists(
        evaluator, bp.award.award_url if bp.award else None,
        "Item_1_Award_Verification_URL",
        "Provides a real URL from oscars.org or another reputable source that directly supports the win claim",
        award_info_node, critical=True
    )

    # Award Category
    award_category_node = evaluator.add_leaf(
        id="Item_1_Award_Category",
        desc="States the exact award category won (Best Picture)",
        parent=award_info_node,
        critical=True
    )
    award_category = bp.award.category if bp.award else None
    await evaluator.verify(
        claim=f"The film '{title}' won {award_category}.",
        node=award_category_node,
        sources=(bp.award.award_url if bp.award else None),
        additional_instruction="Verify that the provided page clearly confirms the film won the specified category.",
        extra_prerequisites=[award_url_exists]
    )

    # Ceremony Name
    ceremony_node = evaluator.add_leaf(
        id="Item_1_Ceremony_Name",
        desc="States the full name of the ceremony (96th Academy Awards)",
        parent=award_info_node,
        critical=True
    )
    ceremony_name = bp.award.ceremony_name if bp.award else None
    await evaluator.verify(
        claim=f"The award was presented at the {ceremony_name}.",
        node=ceremony_node,
        sources=(bp.award.award_url if bp.award else None),
        additional_instruction="Check that the page includes the full ceremony name and edition number.",
        extra_prerequisites=[award_url_exists]
    )

    # Award Date
    award_date_node = evaluator.add_leaf(
        id="Item_1_Award_Presented_Or_Announced_Date",
        desc="Provides the date the award was presented or announced",
        parent=award_info_node,
        critical=True
    )
    award_date = bp.award.award_date if bp.award else None
    await evaluator.verify(
        claim=f"The award was presented or announced on {award_date}.",
        node=award_date_node,
        sources=(bp.award.award_url if bp.award else None),
        additional_instruction="Verify the specific award date from the page.",
        extra_prerequisites=[award_url_exists]
    )

    # Director Information (critical)
    director_info_node = evaluator.add_parallel(
        id="Item_1_Director_Information",
        desc="Provide director identity, Best Director note, and a supporting URL",
        parent=item_node,
        critical=True
    )
    director_url_exists = _url_exists(
        evaluator, bp.director.info_url if bp.director else None,
        "Item_1_Director_Info_URL",
        "Provides a real URL that directly supports the director information (and Best Director status if claimed)",
        director_info_node, critical=True
    )

    director_name_node = evaluator.add_leaf(
        id="Item_1_Director_Full_Name",
        desc="Provides the director's full name",
        parent=director_info_node,
        critical=True
    )
    director_name = bp.director.name if bp.director else None
    await evaluator.verify(
        claim=f"The director of '{title}' is {director_name}.",
        node=director_name_node,
        sources=(bp.director.info_url if bp.director else None),
        additional_instruction="Confirm the director's full name from the provided source.",
        extra_prerequisites=[director_url_exists]
    )

    best_dir_status_node = evaluator.add_leaf(
        id="Item_1_Best_Director_Status_Noted",
        desc="Explicitly notes whether the director also won Best Director at the same ceremony",
        parent=director_info_node,
        critical=True
    )
    status_note = (bp.director.best_director_status_note if bp.director else None) or "unknown"
    # We treat this leaf as a simple presence check in the answer text
    await evaluator.verify(
        claim=f"The answer explicitly states the Best Director status for the director (status: {status_note}).",
        node=best_dir_status_node,
        sources=None,
        additional_instruction="Verify in the answer text whether the Best Director status is explicitly noted, regardless of whether it is 'won' or 'not_won'."
    )

    # Production Details (critical)
    prod_details_node = evaluator.add_parallel(
        id="Item_1_Production_Details",
        desc="Provide production companies, distributor, release date, runtime, and a supporting database URL",
        parent=item_node,
        critical=True
    )
    prod_url_exists = _url_exists(
        evaluator, bp.production.details_url if bp.production else None,
        "Item_1_Production_Details_URL",
        "Provides a real URL from IMDb or another reputable database that directly supports the production details",
        prod_details_node, critical=True
    )

    # Primary Production Companies
    prod_companies_node = evaluator.add_leaf(
        id="Item_1_Primary_Production_Companies",
        desc="Identifies the primary production companies",
        parent=prod_details_node,
        critical=True
    )
    companies_text = _list_to_readable(bp.production.production_companies if bp.production else [])
    await evaluator.verify(
        claim=f"The primary production companies for '{title}' include: {companies_text}.",
        node=prod_companies_node,
        sources=(bp.production.details_url if bp.production else None),
        additional_instruction="Validate that the listed companies appear on the source page. Partial matches are acceptable if the named companies are present.",
        extra_prerequisites=[prod_url_exists]
    )

    # Distributor
    distributor_node = evaluator.add_leaf(
        id="Item_1_Distributor",
        desc="Identifies the theatrical distributor",
        parent=prod_details_node,
        critical=True
    )
    distributor = bp.production.distributor if bp.production else None
    await evaluator.verify(
        claim=f"The theatrical distributor for '{title}' is {distributor}.",
        node=distributor_node,
        sources=(bp.production.details_url if bp.production else None),
        additional_instruction="Confirm the distributor listed on the database page.",
        extra_prerequisites=[prod_url_exists]
    )

    # Theatrical Release Date
    release_date_node = evaluator.add_leaf(
        id="Item_1_Theatrical_Release_Date",
        desc="Provides the theatrical release date",
        parent=prod_details_node,
        critical=True
    )
    release_date = bp.production.release_date if bp.production else None
    await evaluator.verify(
        claim=f"The theatrical release date of '{title}' is {release_date}.",
        node=release_date_node,
        sources=(bp.production.details_url if bp.production else None),
        additional_instruction="Verify the release date as shown on the source page. Minor format variations are acceptable.",
        extra_prerequisites=[prod_url_exists]
    )

    # Runtime
    runtime_node = evaluator.add_leaf(
        id="Item_1_Runtime_Hours_Minutes",
        desc="Provides runtime in hours and minutes",
        parent=prod_details_node,
        critical=True
    )
    runtime = bp.production.runtime if bp.production else None
    await evaluator.verify(
        claim=f"The runtime of '{title}' is {runtime}.",
        node=runtime_node,
        sources=(bp.production.details_url if bp.production else None),
        additional_instruction="Confirm the runtime exactly or with minor formatting variations.",
        extra_prerequisites=[prod_url_exists]
    )

    # Cast and Acting Awards (critical)
    cast_node = evaluator.add_parallel(
        id="Item_1_Cast_And_Acting_Awards",
        desc="Provide required cast/acting-award details and a supporting URL",
        parent=item_node,
        critical=True
    )
    cast_url_exists = _url_exists(
        evaluator, bp.cast.cast_url if bp.cast else None,
        "Item_1_Cast_Info_URL",
        "Provides a real URL that directly supports the cast and role claims",
        cast_node, critical=True
    )

    # Best Actor Winner With Character
    best_actor_node = evaluator.add_leaf(
        id="Item_1_Best_Actor_Winner_With_Character",
        desc="Identifies the Best Actor winner for this film and provides the character name",
        parent=cast_node,
        critical=True
    )
    ba_name = bp.cast.best_actor_name if bp.cast else None
    ba_char = bp.cast.best_actor_character if bp.cast else None
    await evaluator.verify(
        claim=f"{ba_name} won Best Actor for this film and portrayed '{ba_char}'.",
        node=best_actor_node,
        sources=[u for u in [bp.award.award_url if bp.award else None, bp.cast.cast_url if bp.cast else None] if _non_empty_str(u)],
        additional_instruction="Verify both the acting award and the character role for this film. Minor variations in character naming are acceptable.",
        extra_prerequisites=[cast_url_exists]
    )

    # Best Supporting Actor Winner With Character
    supp_actor_node = evaluator.add_leaf(
        id="Item_1_Best_Supporting_Actor_Winner_With_Character",
        desc="Identifies the Best Supporting Actor winner for this film and provides the character name",
        parent=cast_node,
        critical=True
    )
    sa_name = bp.cast.supporting_actor_name if bp.cast else None
    sa_char = bp.cast.supporting_actor_character if bp.cast else None
    await evaluator.verify(
        claim=f"{sa_name} won Best Supporting Actor for this film and portrayed '{sa_char}'.",
        node=supp_actor_node,
        sources=[u for u in [bp.award.award_url if bp.award else None, bp.cast.cast_url if bp.cast else None] if _non_empty_str(u)],
        additional_instruction="Verify both the acting award and the character role for this film. Minor variations in character naming are acceptable.",
        extra_prerequisites=[cast_url_exists]
    )

    # Critical Reception (optional, non-critical)
    reception_node = evaluator.add_parallel(
        id="Item_1_Critical_Reception_Optional",
        desc="Optional but recommended critical reception data with supporting URL if provided",
        parent=item_node,
        critical=False
    )
    # Ratings leaf
    rating_leaf = evaluator.add_leaf(
        id="Item_1_RT_Tomatometer_AndOr_IMDb_Rating",
        desc="Provides Rotten Tomatoes Tomatometer score and/or IMDb rating (if included in the response)",
        parent=reception_node,
        critical=False
    )
    rt_score = bp.reception.rotten_tomatoes_score if bp.reception else None
    imdb_rating = bp.reception.imdb_rating if bp.reception else None
    ratings_text = f"Rotten Tomatoes: {rt_score}; IMDb: {imdb_rating}"
    await evaluator.verify(
        claim=f"Critical reception for '{title}' includes: {ratings_text}.",
        node=rating_leaf,
        sources=(bp.reception.rating_url if bp.reception else None),
        additional_instruction="If provided, verify that the rating values match what is shown on the rating source page. Minor rounding is acceptable."
    )
    # Reception URL existence (non-critical)
    _url_exists(
        evaluator, bp.reception.rating_url if bp.reception else None,
        "Item_1_Reception_URL",
        "If reception data is provided, includes a real URL from the rating source that supports it",
        reception_node, critical=False
    )


async def verify_palme_dor_item(evaluator: Evaluator, root, pd: PalmeDorItem):
    item_node = evaluator.add_parallel(
        id="Item_2_Palme_dOr_77th_Cannes_Film_Festival",
        desc="Film that won the Palme d'Or at the 77th Cannes Film Festival",
        parent=root,
        critical=False
    )

    title = pd.title or "Unknown Title"

    # Award Information (critical)
    award_info_node = evaluator.add_parallel(
        id="Item_2_Award_Information",
        desc="Provide award category/name, festival name, award date, and a supporting URL",
        parent=item_node,
        critical=True
    )
    award_url_exists = _url_exists(
        evaluator, pd.award.award_url if pd.award else None,
        "Item_2_Award_Verification_URL",
        "Provides a real URL from festival-cannes.com or another reputable source that directly supports the win claim",
        award_info_node, critical=True
    )

    # Award Category or Name
    award_category_node = evaluator.add_leaf(
        id="Item_2_Award_Category_Or_Name",
        desc="States the exact award category/name won (Palme d'Or)",
        parent=award_info_node,
        critical=True
    )
    category = pd.award.category if pd.award else None
    await evaluator.verify(
        claim=f"The film '{title}' won {category}.",
        node=award_category_node,
        sources=(pd.award.award_url if pd.award else None),
        additional_instruction="Confirm that the page states the film won the Palme d'Or.",
        extra_prerequisites=[award_url_exists]
    )

    # Festival Name
    festival_node = evaluator.add_leaf(
        id="Item_2_Festival_Name",
        desc="States the full name of the festival (77th Cannes Film Festival)",
        parent=award_info_node,
        critical=True
    )
    festival_name = pd.award.ceremony_name if pd.award else None
    await evaluator.verify(
        claim=f"The award was presented at the {festival_name}.",
        node=festival_node,
        sources=(pd.award.award_url if pd.award else None),
        additional_instruction="Verify the festival name and edition number on the page.",
        extra_prerequisites=[award_url_exists]
    )

    # Award Date
    award_date_node = evaluator.add_leaf(
        id="Item_2_Award_Presented_Or_Announced_Date",
        desc="Provides the date the award was presented or announced",
        parent=award_info_node,
        critical=True
    )
    award_date = pd.award.award_date if pd.award else None
    await evaluator.verify(
        claim=f"The award was presented or announced on {award_date}.",
        node=award_date_node,
        sources=(pd.award.award_url if pd.award else None),
        additional_instruction="Verify the award date as shown on the page.",
        extra_prerequisites=[award_url_exists]
    )

    # Director Information (critical)
    director_node = evaluator.add_parallel(
        id="Item_2_Director_Information",
        desc="Provide director identity and a supporting URL",
        parent=item_node,
        critical=True
    )
    director_url_exists = _url_exists(
        evaluator, pd.director.info_url if pd.director else None,
        "Item_2_Director_Info_URL",
        "Provides a real URL that directly supports the director information",
        director_node, critical=True
    )
    director_name_leaf = evaluator.add_leaf(
        id="Item_2_Director_Full_Name",
        desc="Provides the director's full name",
        parent=director_node,
        critical=True
    )
    director_name = pd.director.name if pd.director else None
    await evaluator.verify(
        claim=f"The director of '{title}' is {director_name}.",
        node=director_name_leaf,
        sources=(pd.director.info_url if pd.director else None),
        additional_instruction="Confirm the director's full name from the provided source.",
        extra_prerequisites=[director_url_exists]
    )

    # Production Details (critical)
    prod_node = evaluator.add_parallel(
        id="Item_2_Production_Details",
        desc="Provide production companies, distributor, release date, runtime, and a supporting database URL",
        parent=item_node,
        critical=True
    )
    prod_url_exists = _url_exists(
        evaluator, pd.production.details_url if pd.production else None,
        "Item_2_Production_Details_URL",
        "Provides a real URL from IMDb or another reputable database that directly supports the production details",
        prod_node, critical=True
    )

    # Primary Production Companies
    prod_companies_leaf = evaluator.add_leaf(
        id="Item_2_Primary_Production_Companies",
        desc="Identifies the primary production companies",
        parent=prod_node,
        critical=True
    )
    companies_text = _list_to_readable(pd.production.production_companies if pd.production else [])
    await evaluator.verify(
        claim=f"The primary production companies for '{title}' include: {companies_text}.",
        node=prod_companies_leaf,
        sources=(pd.production.details_url if pd.production else None),
        additional_instruction="Validate that the listed companies appear on the source page.",
        extra_prerequisites=[prod_url_exists]
    )

    # Distributor
    distributor_leaf = evaluator.add_leaf(
        id="Item_2_Distributor",
        desc="Identifies the theatrical distributor",
        parent=prod_node,
        critical=True
    )
    distributor = pd.production.distributor if pd.production else None
    await evaluator.verify(
        claim=f"The theatrical distributor for '{title}' is {distributor}.",
        node=distributor_leaf,
        sources=(pd.production.details_url if pd.production else None),
        additional_instruction="Confirm the distributor listed on the database page.",
        extra_prerequisites=[prod_url_exists]
    )

    # Theatrical Release Date
    release_leaf = evaluator.add_leaf(
        id="Item_2_Theatrical_Release_Date",
        desc="Provides the theatrical release date",
        parent=prod_node,
        critical=True
    )
    release_date = pd.production.release_date if pd.production else None
    await evaluator.verify(
        claim=f"The theatrical release date of '{title}' is {release_date}.",
        node=release_leaf,
        sources=(pd.production.details_url if pd.production else None),
        additional_instruction="Verify the release date from the source page.",
        extra_prerequisites=[prod_url_exists]
    )

    # Runtime
    runtime_leaf = evaluator.add_leaf(
        id="Item_2_Runtime_Hours_Minutes",
        desc="Provides runtime in hours and minutes",
        parent=prod_node,
        critical=True
    )
    runtime = pd.production.runtime if pd.production else None
    await evaluator.verify(
        claim=f"The runtime of '{title}' is {runtime}.",
        node=runtime_leaf,
        sources=(pd.production.details_url if pd.production else None),
        additional_instruction="Confirm the runtime exactly or with minor formatting variations.",
        extra_prerequisites=[prod_url_exists]
    )

    # Cast Information (critical)
    cast_node = evaluator.add_parallel(
        id="Item_2_Cast_Information",
        desc="Provide required cast details and a supporting URL",
        parent=item_node,
        critical=True
    )
    cast_url_exists = _url_exists(
        evaluator, pd.cast.cast_url if pd.cast else None,
        "Item_2_Cast_Info_URL",
        "Provides a real URL that directly supports the cast and role claims",
        cast_node, critical=True
    )

    # Lead Actress With Character
    lead_actress_leaf = evaluator.add_leaf(
        id="Item_2_Lead_Actress_With_Character",
        desc="Identifies the lead actress and provides the character name",
        parent=cast_node,
        critical=True
    )
    la_name = pd.cast.lead_actress_name if pd.cast else None
    la_char = pd.cast.lead_actress_character if pd.cast else None
    await evaluator.verify(
        claim=f"The lead actress in '{title}' is {la_name}, portraying '{la_char}'.",
        node=lead_actress_leaf,
        sources=(pd.cast.cast_url if pd.cast else None),
        additional_instruction="Confirm that the named actress and character appear on the source page.",
        extra_prerequisites=[cast_url_exists]
    )

    # Other Main Cast With Role
    other_cast_leaf = evaluator.add_leaf(
        id="Item_2_Other_Main_Cast_With_Role",
        desc="Identifies one other main cast member and provides their role/character name",
        parent=cast_node,
        critical=True
    )
    oc_name = pd.cast.other_cast_name if pd.cast else None
    oc_role = pd.cast.other_cast_role if pd.cast else None
    await evaluator.verify(
        claim=f"Another main cast member in '{title}' is {oc_name}, playing '{oc_role}'.",
        node=other_cast_leaf,
        sources=(pd.cast.cast_url if pd.cast else None),
        additional_instruction="Confirm the cast member and role on the source page.",
        extra_prerequisites=[cast_url_exists]
    )

    # Critical Reception (optional)
    reception_node = evaluator.add_parallel(
        id="Item_2_Critical_Reception_Optional",
        desc="Optional but recommended critical reception data with supporting URL if provided",
        parent=item_node,
        critical=False
    )
    rating_leaf = evaluator.add_leaf(
        id="Item_2_RT_Tomatometer_AndOr_IMDb_Rating",
        desc="Provides Rotten Tomatoes score and/or IMDb rating (if included in the response)",
        parent=reception_node,
        critical=False
    )
    rt_score = pd.reception.rotten_tomatoes_score if pd.reception else None
    imdb_rating = pd.reception.imdb_rating if pd.reception else None
    ratings_text = f"Rotten Tomatoes: {rt_score}; IMDb: {imdb_rating}"
    await evaluator.verify(
        claim=f"Critical reception for '{title}' includes: {ratings_text}.",
        node=rating_leaf,
        sources=(pd.reception.rating_url if pd.reception else None),
        additional_instruction="If provided, verify that the rating values match what is shown on the rating source page."
    )
    _url_exists(
        evaluator, pd.reception.rating_url if pd.reception else None,
        "Item_2_Reception_URL",
        "If reception data is provided, includes a real URL from the rating source that supports it",
        reception_node, critical=False
    )


async def verify_emmy_drama_item(evaluator: Evaluator, root, ed: EmmyDramaItem):
    item_node = evaluator.add_parallel(
        id="Item_3_Outstanding_Drama_Series_76th_Emmy_Awards",
        desc="TV series that won Outstanding Drama Series at the 76th Emmy Awards",
        parent=root,
        critical=False
    )

    title = ed.title or "Unknown Title"

    # Award Information (critical)
    award_info_node = evaluator.add_parallel(
        id="Item_3_Award_Information",
        desc="Provide award category, ceremony name, award date, and a supporting URL",
        parent=item_node,
        critical=True
    )
    award_url_exists = _url_exists(
        evaluator, ed.award.award_url if ed.award else None,
        "Item_3_Award_Verification_URL",
        "Provides a real URL from televisionacademy.com or another reputable source that directly supports the win claim",
        award_info_node, critical=True
    )

    # Award Category
    award_category_leaf = evaluator.add_leaf(
        id="Item_3_Award_Category",
        desc="States the exact award category won (Outstanding Drama Series)",
        parent=award_info_node,
        critical=True
    )
    category = ed.award.category if ed.award else None
    await evaluator.verify(
        claim=f"The TV series '{title}' won {category}.",
        node=award_category_leaf,
        sources=(ed.award.award_url if ed.award else None),
        additional_instruction="Confirm that the page states the series won Outstanding Drama Series.",
        extra_prerequisites=[award_url_exists]
    )

    # Ceremony Name
    ceremony_leaf = evaluator.add_leaf(
        id="Item_3_Ceremony_Name",
        desc="States the full name of the ceremony (76th Emmy Awards)",
        parent=award_info_node,
        critical=True
    )
    ceremony_name = ed.award.ceremony_name if ed.award else None
    await evaluator.verify(
        claim=f"The award was presented at the {ceremony_name}.",
        node=ceremony_leaf,
        sources=(ed.award.award_url if ed.award else None),
        additional_instruction="Verify the ceremony name and edition number on the page.",
        extra_prerequisites=[award_url_exists]
    )

    # Award Date
    award_date_leaf = evaluator.add_leaf(
        id="Item_3_Award_Presented_Or_Announced_Date",
        desc="Provides the date the award was presented or announced",
        parent=award_info_node,
        critical=True
    )
    award_date = ed.award.award_date if ed.award else None
    await evaluator.verify(
        claim=f"The award was presented or announced on {award_date}.",
        node=award_date_leaf,
        sources=(ed.award.award_url if ed.award else None),
        additional_instruction="Verify the specific award date from the page.",
        extra_prerequisites=[award_url_exists]
    )

    # Creator/Showrunner/Primary Writer (critical)
    creator_node = evaluator.add_parallel(
        id="Item_3_Creator_Showrunner_Primary_Writer",
        desc="Provide creator/showrunner/primary writer identity and a supporting URL",
        parent=item_node,
        critical=True
    )
    creator_url_exists = _url_exists(
        evaluator, ed.creator.info_url if ed.creator else None,
        "Item_3_Creator_Info_URL",
        "Provides a real URL that directly supports the creator/showrunner/primary writer information",
        creator_node, critical=True
    )
    creator_name_leaf = evaluator.add_leaf(
        id="Item_3_Creator_Or_Showrunner_Name",
        desc="Identifies the creator, showrunner, or primary writer",
        parent=creator_node,
        critical=True
    )
    creator_name = ed.creator.name if ed.creator else None
    await evaluator.verify(
        claim=f"The creator/showrunner/primary writer of '{title}' is {creator_name}.",
        node=creator_name_leaf,
        sources=(ed.creator.info_url if ed.creator else None),
        additional_instruction="Confirm the individual's role and name on the source page.",
        extra_prerequisites=[creator_url_exists]
    )

    # Production Details (critical)
    prod_node = evaluator.add_parallel(
        id="Item_3_Production_Details",
        desc="Provide production companies, platform/network, premiere date, total episodes, and a supporting database URL",
        parent=item_node,
        critical=True
    )
    prod_url_exists = _url_exists(
        evaluator, ed.production.details_url if ed.production else None,
        "Item_3_Production_Details_URL",
        "Provides a real URL from IMDb or another reputable database that directly supports the production details",
        prod_node, critical=True
    )

    # Production companies
    prod_companies_leaf = evaluator.add_leaf(
        id="Item_3_Primary_Production_Companies",
        desc="Identifies the primary production companies",
        parent=prod_node,
        critical=True
    )
    companies_text = _list_to_readable(ed.production.production_companies if ed.production else [])
    await evaluator.verify(
        claim=f"The primary production companies for '{title}' include: {companies_text}.",
        node=prod_companies_leaf,
        sources=(ed.production.details_url if ed.production else None),
        additional_instruction="Validate that the listed companies appear on the source page.",
        extra_prerequisites=[prod_url_exists]
    )

    # Network/Platform
    platform_leaf = evaluator.add_leaf(
        id="Item_3_Network_Or_Streaming_Platform",
        desc="Identifies the network/streaming platform (the distributor for TV)",
        parent=prod_node,
        critical=True
    )
    platform = ed.production.platform if ed.production else None
    await evaluator.verify(
        claim=f"The network/streaming platform for '{title}' is {platform}.",
        node=platform_leaf,
        sources=(ed.production.details_url if ed.production else None),
        additional_instruction="Confirm the platform listed on the source page.",
        extra_prerequisites=[prod_url_exists]
    )

    # Premiere Date
    premiere_leaf = evaluator.add_leaf(
        id="Item_3_Series_Premiere_Date",
        desc="Provides the premiere date",
        parent=prod_node,
        critical=True
    )
    premiere_date = ed.production.premiere_date if ed.production else None
    await evaluator.verify(
        claim=f"The series premiere date of '{title}' is {premiere_date}.",
        node=premiere_leaf,
        sources=(ed.production.details_url if ed.production else None),
        additional_instruction="Verify the premiere date from the source page.",
        extra_prerequisites=[prod_url_exists]
    )

    # Total Episodes
    episodes_leaf = evaluator.add_leaf(
        id="Item_3_Total_Number_Of_Episodes",
        desc="Provides the total number of episodes",
        parent=prod_node,
        critical=True
    )
    total_eps = ed.production.total_episodes if ed.production else None
    await evaluator.verify(
        claim=f"The total number of episodes of '{title}' is {total_eps}.",
        node=episodes_leaf,
        sources=(ed.production.details_url if ed.production else None),
        additional_instruction="Confirm the total episode count from the source page.",
        extra_prerequisites=[prod_url_exists]
    )

    # Main Cast Information (critical)
    cast_node = evaluator.add_parallel(
        id="Item_3_Main_Cast_Information",
        desc="Provide two main cast members with character names and a supporting URL",
        parent=item_node,
        critical=True
    )
    cast_url_exists = _url_exists(
        evaluator, ed.cast.cast_url if ed.cast else None,
        "Item_3_Cast_Info_URL",
        "Provides a real URL that directly supports the cast and character claims",
        cast_node, critical=True
    )

    cast1_leaf = evaluator.add_leaf(
        id="Item_3_Main_Cast_Member_1_With_Character",
        desc="Identifies one main cast member and character name",
        parent=cast_node,
        critical=True
    )
    mc1 = ed.cast.main_cast_1_name if ed.cast else None
    ch1 = ed.cast.main_cast_1_character if ed.cast else None
    await evaluator.verify(
        claim=f"'{title}' features {mc1} as '{ch1}'.",
        node=cast1_leaf,
        sources=(ed.cast.cast_url if ed.cast else None),
        additional_instruction="Confirm that the named actor and character appear on the source page.",
        extra_prerequisites=[cast_url_exists]
    )

    cast2_leaf = evaluator.add_leaf(
        id="Item_3_Main_Cast_Member_2_With_Character",
        desc="Identifies a second main cast member and character name",
        parent=cast_node,
        critical=True
    )
    mc2 = ed.cast.main_cast_2_name if ed.cast else None
    ch2 = ed.cast.main_cast_2_character if ed.cast else None
    await evaluator.verify(
        claim=f"'{title}' also features {mc2} as '{ch2}'.",
        node=cast2_leaf,
        sources=(ed.cast.cast_url if ed.cast else None),
        additional_instruction="Confirm that the named actor and character appear on the source page.",
        extra_prerequisites=[cast_url_exists]
    )

    # Critical Reception (optional)
    reception_node = evaluator.add_parallel(
        id="Item_3_Critical_Reception_Optional",
        desc="Optional but recommended critical reception data with supporting URL if provided",
        parent=item_node,
        critical=False
    )
    rating_leaf = evaluator.add_leaf(
        id="Item_3_RT_Tomatometer_AndOr_IMDb_Rating",
        desc="Provides Rotten Tomatoes score and/or IMDb rating (if included in the response)",
        parent=reception_node,
        critical=False
    )
    rt_score = ed.reception.rotten_tomatoes_score if ed.reception else None
    imdb_rating = ed.reception.imdb_rating if ed.reception else None
    ratings_text = f"Rotten Tomatoes: {rt_score}; IMDb: {imdb_rating}"
    await evaluator.verify(
        claim=f"Critical reception for '{title}' includes: {ratings_text}.",
        node=rating_leaf,
        sources=(ed.reception.rating_url if ed.reception else None),
        additional_instruction="If provided, verify that the rating values match what is shown on the rating source page."
    )
    _url_exists(
        evaluator, ed.reception.rating_url if ed.reception else None,
        "Item_3_Reception_URL",
        "If reception data is provided, includes a real URL from the rating source that supports it",
        reception_node, critical=False
    )


async def verify_emmy_limited_item(evaluator: Evaluator, root, el: EmmyLimitedItem):
    item_node = evaluator.add_parallel(
        id="Item_4_Outstanding_Limited_Or_Anthology_Series_76th_Emmy_Awards",
        desc="TV series that won Outstanding Limited or Anthology Series at the 76th Emmy Awards",
        parent=root,
        critical=False
    )

    title = el.title or "Unknown Title"

    # Award Information (critical)
    award_info_node = evaluator.add_parallel(
        id="Item_4_Award_Information",
        desc="Provide award category, ceremony name, award date, and a supporting URL",
        parent=item_node,
        critical=True
    )
    award_url_exists = _url_exists(
        evaluator, el.award.award_url if el.award else None,
        "Item_4_Award_Verification_URL",
        "Provides a real URL from televisionacademy.com or another reputable source that directly supports the win claim",
        award_info_node, critical=True
    )

    # Award Category
    award_category_leaf = evaluator.add_leaf(
        id="Item_4_Award_Category",
        desc="States the exact award category won (Outstanding Limited or Anthology Series)",
        parent=award_info_node,
        critical=True
    )
    category = el.award.category if el.award else None
    await evaluator.verify(
        claim=f"The TV series '{title}' won {category}.",
        node=award_category_leaf,
        sources=(el.award.award_url if el.award else None),
        additional_instruction="Confirm that the page states the series won Outstanding Limited or Anthology Series.",
        extra_prerequisites=[award_url_exists]
    )

    # Ceremony Name
    ceremony_leaf = evaluator.add_leaf(
        id="Item_4_Ceremony_Name",
        desc="States the full name of the ceremony (76th Emmy Awards)",
        parent=award_info_node,
        critical=True
    )
    ceremony_name = el.award.ceremony_name if el.award else None
    await evaluator.verify(
        claim=f"The award was presented at the {ceremony_name}.",
        node=ceremony_leaf,
        sources=(el.award.award_url if el.award else None),
        additional_instruction="Verify the ceremony name and edition number on the page.",
        extra_prerequisites=[award_url_exists]
    )

    # Award Date
    award_date_leaf = evaluator.add_leaf(
        id="Item_4_Award_Presented_Or_Announced_Date",
        desc="Provides the date the award was presented or announced",
        parent=award_info_node,
        critical=True
    )
    award_date = el.award.award_date if el.award else None
    await evaluator.verify(
        claim=f"The award was presented or announced on {award_date}.",
        node=award_date_leaf,
        sources=(el.award.award_url if el.award else None),
        additional_instruction="Verify the specific award date from the page.",
        extra_prerequisites=[award_url_exists]
    )

    # Creator/Showrunner/Primary Writer (critical)
    creator_node = evaluator.add_parallel(
        id="Item_4_Creator_Showrunner_Primary_Writer",
        desc="Provide creator/showrunner/primary writer identity and a supporting URL",
        parent=item_node,
        critical=True
    )
    creator_url_exists = _url_exists(
        evaluator, el.creator.info_url if el.creator else None,
        "Item_4_Creator_Info_URL",
        "Provides a real URL that directly supports the creator/showrunner/primary writer information",
        creator_node, critical=True
    )
    creator_name_leaf = evaluator.add_leaf(
        id="Item_4_Creator_Or_Writer_Name",
        desc="Identifies the creator, showrunner, or primary writer",
        parent=creator_node,
        critical=True
    )
    creator_name = el.creator.name if el.creator else None
    await evaluator.verify(
        claim=f"The creator/showrunner/primary writer of '{title}' is {creator_name}.",
        node=creator_name_leaf,
        sources=(el.creator.info_url if el.creator else None),
        additional_instruction="Confirm the individual's role and name on the source page.",
        extra_prerequisites=[creator_url_exists]
    )

    # Production Details (critical)
    prod_node = evaluator.add_parallel(
        id="Item_4_Production_Details",
        desc="Provide production companies, platform/network, premiere/release date, total episodes, and a supporting database URL",
        parent=item_node,
        critical=True
    )
    prod_url_exists = _url_exists(
        evaluator, el.production.details_url if el.production else None,
        "Item_4_Production_Details_URL",
        "Provides a real URL from IMDb or another reputable database that directly supports the production details",
        prod_node, critical=True
    )

    # Production companies
    prod_companies_leaf = evaluator.add_leaf(
        id="Item_4_Primary_Production_Companies",
        desc="Identifies the primary production companies",
        parent=prod_node,
        critical=True
    )
    companies_text = _list_to_readable(el.production.production_companies if el.production else [])
    await evaluator.verify(
        claim=f"The primary production companies for '{title}' include: {companies_text}.",
        node=prod_companies_leaf,
        sources=(el.production.details_url if el.production else None),
        additional_instruction="Validate that the listed companies appear on the source page.",
        extra_prerequisites=[prod_url_exists]
    )

    # Network/Platform
    platform_leaf = evaluator.add_leaf(
        id="Item_4_Network_Or_Streaming_Platform",
        desc="Identifies the network/streaming platform (the distributor for TV)",
        parent=prod_node,
        critical=True
    )
    platform = el.production.platform if el.production else None
    await evaluator.verify(
        claim=f"The network/streaming platform for '{title}' is {platform}.",
        node=platform_leaf,
        sources=(el.production.details_url if el.production else None),
        additional_instruction="Confirm the platform listed on the source page.",
        extra_prerequisites=[prod_url_exists]
    )

    # Premiere or Release Date
    premiere_leaf = evaluator.add_leaf(
        id="Item_4_Series_Premiere_Or_Release_Date",
        desc="Provides the premiere or release date",
        parent=prod_node,
        critical=True
    )
    premiere_date = el.production.premiere_date if el.production else None
    await evaluator.verify(
        claim=f"The premiere/release date of '{title}' is {premiere_date}.",
        node=premiere_leaf,
        sources=(el.production.details_url if el.production else None),
        additional_instruction="Verify the date from the source page.",
        extra_prerequisites=[prod_url_exists]
    )

    # Total Episodes
    episodes_leaf = evaluator.add_leaf(
        id="Item_4_Total_Number_Of_Episodes",
        desc="Provides the total number of episodes",
        parent=prod_node,
        critical=True
    )
    total_eps = el.production.total_episodes if el.production else None
    await evaluator.verify(
        claim=f"The total number of episodes of '{title}' is {total_eps}.",
        node=episodes_leaf,
        sources=(el.production.details_url if el.production else None),
        additional_instruction="Confirm the total episode count from the source page.",
        extra_prerequisites=[prod_url_exists]
    )

    # Cast and Acting Awards (critical)
    cast_node = evaluator.add_parallel(
        id="Item_4_Cast_And_Acting_Awards",
        desc="Provide required acting-award-linked cast details and a supporting URL",
        parent=item_node,
        critical=True
    )
    cast_url_exists = _url_exists(
        evaluator, el.cast.cast_url if el.cast else None,
        "Item_4_Cast_Info_URL",
        "Provides a real URL that directly supports the cast/role and acting-award linkage claims",
        cast_node, critical=True
    )

    # Lead Actor Also Acting Emmy Winner With Character
    lead_actor_leaf = evaluator.add_leaf(
        id="Item_4_Lead_Actor_Also_Acting_Emmy_Winner_With_Character",
        desc="Identifies the lead actor who also won an acting Emmy for this series and provides the character name",
        parent=cast_node,
        critical=True
    )
    la_name = el.cast.lead_actor_name if el.cast else None
    la_char = el.cast.lead_actor_character if el.cast else None
    await evaluator.verify(
        claim=f"{la_name} won an acting Emmy for this series and portrayed '{la_char}'.",
        node=lead_actor_leaf,
        sources=[u for u in [el.award.award_url if el.award else None, el.cast.cast_url if el.cast else None] if _non_empty_str(u)],
        additional_instruction="Verify the acting Emmy win linkage to this series and the character role.",
        extra_prerequisites=[cast_url_exists]
    )

    # Other Main Cast With Role
    other_cast_leaf = evaluator.add_leaf(
        id="Item_4_Other_Main_Cast_With_Role",
        desc="Identifies one other main cast member and provides their role/character name",
        parent=cast_node,
        critical=True
    )
    oc_name = el.cast.other_cast_name if el.cast else None
    oc_role = el.cast.other_cast_role if el.cast else None
    await evaluator.verify(
        claim=f"Another main cast member in '{title}' is {oc_name}, playing '{oc_role}'.",
        node=other_cast_leaf,
        sources=(el.cast.cast_url if el.cast else None),
        additional_instruction="Confirm the cast member and role on the source page.",
        extra_prerequisites=[cast_url_exists]
    )

    # Critical Reception (optional)
    reception_node = evaluator.add_parallel(
        id="Item_4_Critical_Reception_Optional",
        desc="Optional but recommended critical reception data with supporting URL if provided",
        parent=item_node,
        critical=False
    )
    rating_leaf = evaluator.add_leaf(
        id="Item_4_RT_Tomatometer_AndOr_IMDb_Rating",
        desc="Provides Rotten Tomatoes score and/or IMDb rating (if included in the response)",
        parent=reception_node,
        critical=False
    )
    rt_score = el.reception.rotten_tomatoes_score if el.reception else None
    imdb_rating = el.reception.imdb_rating if el.reception else None
    ratings_text = f"Rotten Tomatoes: {rt_score}; IMDb: {imdb_rating}"
    await evaluator.verify(
        claim=f"Critical reception for '{title}' includes: {ratings_text}.",
        node=rating_leaf,
        sources=(el.reception.rating_url if el.reception else None),
        additional_instruction="If provided, verify that the rating values match what is shown on the rating source page."
    )
    _url_exists(
        evaluator, el.reception.rating_url if el.reception else None,
        "Item_4_Reception_URL",
        "If reception data is provided, includes a real URL from the rating source that supports it",
        reception_node, critical=False
    )


# ----------------------------- Main Evaluation ----------------------------- #

async def evaluate_answer(
    client: Any,
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

    # Extract items concurrently
    best_picture_task = evaluator.extract(
        prompt=prompt_extract_best_picture(),
        template_class=BestPictureItem,
        extraction_name="item_1_best_picture"
    )
    palme_dor_task = evaluator.extract(
        prompt=prompt_extract_palme_dor(),
        template_class=PalmeDorItem,
        extraction_name="item_2_palme_dor"
    )
    emmy_drama_task = evaluator.extract(
        prompt=prompt_extract_emmy_drama(),
        template_class=EmmyDramaItem,
        extraction_name="item_3_emmy_drama"
    )
    emmy_limited_task = evaluator.extract(
        prompt=prompt_extract_emmy_limited(),
        template_class=EmmyLimitedItem,
        extraction_name="item_4_emmy_limited"
    )

    bp, pd, ed, el = await asyncio.gather(
        best_picture_task, palme_dor_task, emmy_drama_task, emmy_limited_task
    )

    # Build verification tree and run checks
    await verify_best_picture_item(evaluator, root, bp)
    await verify_palme_dor_item(evaluator, root, pd)
    await verify_emmy_drama_item(evaluator, root, ed)
    await verify_emmy_limited_item(evaluator, root, el)

    # Optional: record titles extracted for quick overview
    evaluator.add_custom_info(
        info={
            "best_picture_title": bp.title,
            "palme_dor_title": pd.title,
            "emmy_drama_title": ed.title,
            "emmy_limited_title": el.title,
        },
        info_type="extracted_titles",
        info_name="extracted_titles_summary"
    )

    return evaluator.get_summary()