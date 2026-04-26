import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "detroit_thanksgiving_halftime_2025"
TASK_DESCRIPTION = (
    "Identify the rock musician who performed at the Detroit Lions NFL Thanksgiving Day halftime show "
    "on November 27, 2025, and provide the following information:\n\n"
    "1. The musician's full name\n"
    "2. Their birth year (must be 1975)\n"
    "3. The name of the rock duo they co-founded in 1997\n"
    "4. The title of this duo's breakthrough album released in 2001\n"
    "5. The title of the duo's 2003 album that includes the song \"Seven Nation Army\"\n"
    "6. The name of the second band they formed, whose debut album was released in 2006\n"
    "7. The name of the third band they formed, whose debut album was released in 2009\n"
    "8. The name of the record label they co-founded in 2001\n"
    "9. The total number of Grammy Awards they have won (must be 12)\n\n"
    "The musician must be a Detroit native and must have been born in 1975. Provide reference URLs for each piece of information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MusicianExtraction(BaseModel):
    # Identity and event verification
    full_name: Optional[str] = None
    full_name_sources: List[str] = Field(default_factory=list)

    birth_year: Optional[str] = None
    birth_year_sources: List[str] = Field(default_factory=list)

    detroit_native_statement: Optional[str] = None  # e.g., "Born in Detroit, Michigan"
    detroit_native_sources: List[str] = Field(default_factory=list)

    rock_musician_sources: List[str] = Field(default_factory=list)

    halftime_show_sources: List[str] = Field(default_factory=list)

    # Career-related information with sources
    rock_duo_name: Optional[str] = None
    rock_duo_1997_sources: List[str] = Field(default_factory=list)

    breakthrough_album_2001_title: Optional[str] = None
    breakthrough_album_2001_sources: List[str] = Field(default_factory=list)

    duo_album_2003_title: Optional[str] = None
    duo_album_2003_sources: List[str] = Field(default_factory=list)

    second_band_name: Optional[str] = None
    second_band_debut_2006_sources: List[str] = Field(default_factory=list)

    third_band_name: Optional[str] = None
    third_band_debut_2009_sources: List[str] = Field(default_factory=list)

    record_label_name: Optional[str] = None
    record_label_2001_sources: List[str] = Field(default_factory=list)

    grammy_awards_total: Optional[str] = None
    grammy_awards_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_musician_profile() -> str:
    return """
    Extract the specific musician and all requested attributes from the answer text. Return the following fields
    exactly as they appear in the answer. For each field that requires a citation, extract all URLs provided in the answer
    that directly support that field. Only extract URLs explicitly present in the answer text (including markdown links).

    Required JSON fields:
    - full_name: The musician's full name.
    - full_name_sources: Array of URLs supporting the full name identity.

    - birth_year: The musician's birth year as a string (e.g., "1975"). If presented as a date, extract the year only.
    - birth_year_sources: Array of URLs supporting the stated birth year.

    - detroit_native_statement: A short statement string from the answer indicating Detroit origin (e.g., "Born in Detroit" or "Detroit native").
    - detroit_native_sources: Array of URLs supporting that the musician is a Detroit native (born or raised in Detroit, Michigan).

    - rock_musician_sources: Array of URLs supporting that the identified person is a rock musician/rock artist.

    - halftime_show_sources: Array of URLs supporting that the musician performed at the Detroit Lions NFL Thanksgiving Day halftime show on November 27, 2025.

    - rock_duo_name: Name of the rock duo the musician co-founded.
    - rock_duo_1997_sources: Array of URLs supporting that this duo was co-founded in 1997.

    - breakthrough_album_2001_title: The duo's breakthrough album title released in 2001.
    - breakthrough_album_2001_sources: Array of URLs supporting that this album was released in 2001 and is described in the answer as the breakthrough.

    - duo_album_2003_title: The duo's 2003 album title that includes the song "Seven Nation Army".
    - duo_album_2003_sources: Array of URLs supporting the album title, release year 2003, and that the album includes "Seven Nation Army".

    - second_band_name: The name of the second band formed by the musician.
    - second_band_debut_2006_sources: Array of URLs supporting that the second band's debut album was released in 2006.

    - third_band_name: The name of the third band formed by the musician.
    - third_band_debut_2009_sources: Array of URLs supporting that the third band's debut album was released in 2009.

    - record_label_name: The name of the record label co-founded by the musician.
    - record_label_2001_sources: Array of URLs supporting that the record label was co-founded in 2001.

    - grammy_awards_total: The total number of Grammy Awards the musician has won as stated in the answer (should be "12").
    - grammy_awards_sources: Array of URLs supporting that the total number of Grammy Awards is 12.

    Rules:
    - Only extract information explicitly present in the answer.
    - If a field is missing, set it to null (value) or [] (sources).
    - For all *sources fields, include only valid URLs explicitly shown in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(urls: Optional[List[str]]) -> List[str]:
    return urls or []


def _has_year(text: Optional[str], year: str) -> bool:
    if not text:
        return False
    # Extract 4-digit tokens and check match
    digits = re.findall(r"\b(\d{4})\b", text)
    return year in digits or text.strip() == year


def _add_url_existence_node(
    evaluator: Evaluator,
    parent,
    id_base: str,
    desc: str,
    urls: Optional[List[str]],
    critical: bool = True,
):
    return evaluator.add_custom_node(
        result=bool(urls and len(urls) > 0),
        id=f"{id_base}_urls_provided",
        desc=desc,
        parent=parent,
        critical=critical,
    )


def _add_value_existence_node(
    evaluator: Evaluator,
    parent,
    id_base: str,
    desc: str,
    value: Optional[str],
    critical: bool = True,
):
    return evaluator.add_custom_node(
        result=bool(value and value.strip()),
        id=f"{id_base}_value_provided",
        desc=desc,
        parent=parent,
        critical=critical,
    )


# --------------------------------------------------------------------------- #
# Verification construction functions                                         #
# --------------------------------------------------------------------------- #
async def build_identification_checks(
    evaluator: Evaluator,
    parent_node,
    data: MusicianExtraction,
) -> None:
    """
    Build and verify identification checks:
    - halftime_show_performance_with_url
    - rock_musician_with_url
    - detroit_native_with_url
    - birth_year_1975_with_url (+ ensure extracted birth_year states 1975)
    - full_name_with_url
    """
    id_node = evaluator.add_parallel(
        id="identify_musician",
        desc="Correctly identify the musician matching the event and identity constraints (with citations)",
        parent=parent_node,
        critical=True,
    )

    # Prepare existence gating nodes (critical siblings)
    _add_url_existence_node(
        evaluator,
        id_node,
        "halftime_show_performance_with_url",
        "At least one URL is provided for the halftime show performance evidence",
        data.halftime_show_sources,
        critical=True,
    )
    _add_url_existence_node(
        evaluator,
        id_node,
        "rock_musician_with_url",
        "At least one URL is provided that the person is a rock musician",
        data.rock_musician_sources,
        critical=True,
    )
    _add_url_existence_node(
        evaluator,
        id_node,
        "detroit_native_with_url",
        "At least one URL is provided for Detroit native evidence",
        data.detroit_native_sources,
        critical=True,
    )
    _add_url_existence_node(
        evaluator,
        id_node,
        "birth_year_1975_with_url",
        "At least one URL is provided for the birth year evidence",
        data.birth_year_sources,
        critical=True,
    )
    _add_value_existence_node(
        evaluator,
        id_node,
        "birth_year_1975_with_url",
        "The answer provides a birth year value",
        data.birth_year,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_year(data.birth_year, "1975"),
        id="birth_year_is_1975_value_check",
        desc="The provided birth year value equals 1975 (as required)",
        parent=id_node,
        critical=True,
    )
    _add_url_existence_node(
        evaluator,
        id_node,
        "full_name_with_url",
        "At least one URL is provided for the full name evidence",
        data.full_name_sources,
        critical=True,
    )
    _add_value_existence_node(
        evaluator,
        id_node,
        "full_name_with_url",
        "The answer provides a full name value",
        data.full_name,
        critical=True,
    )

    # Create leaf nodes
    halftime_leaf = evaluator.add_leaf(
        id="halftime_show_performance_with_url",
        desc="Provide evidence (URL) that the musician performed at the Detroit Lions NFL Thanksgiving Day halftime show on November 27, 2025",
        parent=id_node,
        critical=True,
    )
    rock_leaf = evaluator.add_leaf(
        id="rock_musician_with_url",
        desc="Provide evidence (URL) that the identified person is a rock musician",
        parent=id_node,
        critical=True,
    )
    detroit_leaf = evaluator.add_leaf(
        id="detroit_native_with_url",
        desc="Provide evidence (URL) that the musician is a Detroit native",
        parent=id_node,
        critical=True,
    )
    birth_leaf = evaluator.add_leaf(
        id="birth_year_1975_with_url",
        desc="Provide evidence (URL) that the musician was born in 1975",
        parent=id_node,
        critical=True,
    )
    fullname_leaf = evaluator.add_leaf(
        id="full_name_with_url",
        desc="Provide the musician's full name with a supporting URL",
        parent=id_node,
        critical=True,
    )

    # Build claims and perform batch verification
    claims = [
        (
            # Keep the name in the claim to strengthen matching when provided
            f"{(data.full_name or 'The musician')} performed at the Detroit Lions NFL Thanksgiving Day halftime show on November 27, 2025.",
            _safe_sources(data.halftime_show_sources),
            halftime_leaf,
            "Verify the event (Detroit Lions NFL Thanksgiving Day) and date (Nov 27, 2025) explicitly. "
            "The source must clearly state the performer and the halftime show context.",
        ),
        (
            f"{(data.full_name or 'The identified person')} is a rock musician (rock artist).",
            _safe_sources(data.rock_musician_sources),
            rock_leaf,
            "Accept reasonable wording variants (e.g., 'rock singer', 'rock guitarist', 'rock artist'). "
            "The source must clearly categorize the musician within rock.",
        ),
        (
            f"{(data.full_name or 'The musician')} is a Detroit native.",
            _safe_sources(data.detroit_native_sources),
            detroit_leaf,
            "Being 'born in Detroit, Michigan' or described as a 'Detroit native' counts. "
            "The source must explicitly indicate Detroit origin (born or raised).",
        ),
        (
            f"{(data.full_name or 'The musician')} was born in 1975.",
            _safe_sources(data.birth_year_sources),
            birth_leaf,
            "Check the birth year equals 1975. If a full birth date is shown, ensure the year is 1975.",
        ),
        (
            f"The musician's full name is {(data.full_name or '').strip()}.",
            _safe_sources(data.full_name_sources),
            fullname_leaf,
            "Verify the full legal/stage name as presented in reliable sources (allow common stage name precedence).",
        ),
    ]

    await evaluator.batch_verify(claims)


async def build_career_checks(
    evaluator: Evaluator,
    parent_node,
    data: MusicianExtraction,
) -> None:
    """
    Build and verify all required career-related checks (each with citations):
    - rock_duo_1997_with_url
    - breakthrough_album_2001_with_url
    - duo_album_2003_includes_song_with_url
    - second_band_debut_2006_with_url
    - third_band_debut_2009_with_url
    - record_label_2001_with_url
    - grammy_awards_12_with_url
    """
    career_node = evaluator.add_parallel(
        id="required_career_information",
        desc="Provide all required career-related information about the identified musician (each with citations)",
        parent=parent_node,
        critical=True,
    )

    # Existence gating nodes
    _add_value_existence_node(
        evaluator,
        career_node,
        "rock_duo_1997_with_url",
        "The answer provides a rock duo name",
        data.rock_duo_name,
        critical=True,
    )
    _add_url_existence_node(
        evaluator,
        career_node,
        "rock_duo_1997_with_url",
        "At least one URL supports the rock duo co-founded in 1997",
        data.rock_duo_1997_sources,
        critical=True,
    )

    _add_value_existence_node(
        evaluator,
        career_node,
        "breakthrough_album_2001_with_url",
        "The answer provides a breakthrough album title (2001)",
        data.breakthrough_album_2001_title,
        critical=True,
    )
    _add_url_existence_node(
        evaluator,
        career_node,
        "breakthrough_album_2001_with_url",
        "At least one URL supports the breakthrough album and 2001 release",
        data.breakthrough_album_2001_sources,
        critical=True,
    )

    _add_value_existence_node(
        evaluator,
        career_node,
        "duo_album_2003_includes_song_with_url",
        "The answer provides the duo's 2003 album title that includes 'Seven Nation Army'",
        data.duo_album_2003_title,
        critical=True,
    )
    _add_url_existence_node(
        evaluator,
        career_node,
        "duo_album_2003_includes_song_with_url",
        "At least one URL supports the 2003 album title and that it includes 'Seven Nation Army'",
        data.duo_album_2003_sources,
        critical=True,
    )

    _add_value_existence_node(
        evaluator,
        career_node,
        "second_band_debut_2006_with_url",
        "The answer provides the second band's name",
        data.second_band_name,
        critical=True,
    )
    _add_url_existence_node(
        evaluator,
        career_node,
        "second_band_debut_2006_with_url",
        "At least one URL supports the second band's debut album release year (2006)",
        data.second_band_debut_2006_sources,
        critical=True,
    )

    _add_value_existence_node(
        evaluator,
        career_node,
        "third_band_debut_2009_with_url",
        "The answer provides the third band's name",
        data.third_band_name,
        critical=True,
    )
    _add_url_existence_node(
        evaluator,
        career_node,
        "third_band_debut_2009_with_url",
        "At least one URL supports the third band's debut album release year (2009)",
        data.third_band_debut_2009_sources,
        critical=True,
    )

    _add_value_existence_node(
        evaluator,
        career_node,
        "record_label_2001_with_url",
        "The answer provides the record label name",
        data.record_label_name,
        critical=True,
    )
    _add_url_existence_node(
        evaluator,
        career_node,
        "record_label_2001_with_url",
        "At least one URL supports that the record label was co-founded in 2001",
        data.record_label_2001_sources,
        critical=True,
    )

    _add_value_existence_node(
        evaluator,
        career_node,
        "grammy_awards_12_with_url",
        "The answer provides the total Grammy count value",
        data.grammy_awards_total,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_year(data.grammy_awards_total, "12") or (data.grammy_awards_total or "").strip() == "12",
        id="grammy_awards_total_is_12_value_check",
        desc="The provided Grammy Awards total equals 12 (as required)",
        parent=career_node,
        critical=True,
    )
    _add_url_existence_node(
        evaluator,
        career_node,
        "grammy_awards_12_with_url",
        "At least one URL supports the total Grammy count equals 12",
        data.grammy_awards_sources,
        critical=True,
    )

    # Create leaf nodes
    duo1997_leaf = evaluator.add_leaf(
        id="rock_duo_1997_with_url",
        desc="Provide the name of the rock duo the musician co-founded and evidence (URL) it was co-founded in 1997",
        parent=career_node,
        critical=True,
    )
    breakthrough2001_leaf = evaluator.add_leaf(
        id="breakthrough_album_2001_with_url",
        desc="Provide the duo’s breakthrough album title and evidence (URL) it was released in 2001",
        parent=career_node,
        critical=True,
    )
    album2003_leaf = evaluator.add_leaf(
        id="duo_album_2003_includes_song_with_url",
        desc="Provide the duo’s 2003 album title and evidence (URL) that it was released in 2003 and includes the song \"Seven Nation Army\"",
        parent=career_node,
        critical=True,
    )
    second2006_leaf = evaluator.add_leaf(
        id="second_band_debut_2006_with_url",
        desc="Provide the name of the second band formed by the musician and evidence (URL) its debut album was released in 2006",
        parent=career_node,
        critical=True,
    )
    third2009_leaf = evaluator.add_leaf(
        id="third_band_debut_2009_with_url",
        desc="Provide the name of the third band formed by the musician and evidence (URL) its debut album was released in 2009",
        parent=career_node,
        critical=True,
    )
    label2001_leaf = evaluator.add_leaf(
        id="record_label_2001_with_url",
        desc="Provide the name of the record label co-founded by the musician and evidence (URL) it was co-founded in 2001",
        parent=career_node,
        critical=True,
    )
    grammy12_leaf = evaluator.add_leaf(
        id="grammy_awards_12_with_url",
        desc="Provide evidence (URL) that the musician has won exactly 12 Grammy Awards total",
        parent=career_node,
        critical=True,
    )

    claims = [
        (
            f"The musician co-founded the rock duo '{(data.rock_duo_name or '').strip()}' in 1997.",
            _safe_sources(data.rock_duo_1997_sources),
            duo1997_leaf,
            "Verify the duo name and that co-founding occurred in 1997.",
        ),
        (
            f"The duo's breakthrough album titled '{(data.breakthrough_album_2001_title or '').strip()}' was released in 2001.",
            _safe_sources(data.breakthrough_album_2001_sources),
            breakthrough2001_leaf,
            "Confirm the album title and that its release year is 2001. The page should indicate it as a breakthrough or equivalent widely recognized milestone.",
        ),
        (
            f"The duo's 2003 album titled '{(data.duo_album_2003_title or '').strip()}' was released in 2003 and includes the song 'Seven Nation Army'.",
            _safe_sources(data.duo_album_2003_sources),
            album2003_leaf,
            "The source must show both: release year 2003 and that 'Seven Nation Army' is on this album.",
        ),
        (
            f"The musician formed the band '{(data.second_band_name or '').strip()}', and that band's debut album was released in 2006.",
            _safe_sources(data.second_band_debut_2006_sources),
            second2006_leaf,
            "Verify the band name and that its debut album release year is 2006.",
        ),
        (
            f"The musician formed the band '{(data.third_band_name or '').strip()}', and that band's debut album was released in 2009.",
            _safe_sources(data.third_band_debut_2009_sources),
            third2009_leaf,
            "Verify the band name and that its debut album release year is 2009.",
        ),
        (
            f"The musician co-founded the record label '{(data.record_label_name or '').strip()}' in 2001.",
            _safe_sources(data.record_label_2001_sources),
            label2001_leaf,
            "Verify the label name and co-founding year 2001.",
        ),
        (
            "The musician has won exactly 12 Grammy Awards in total.",
            _safe_sources(data.grammy_awards_sources),
            grammy12_leaf,
            "Verify that the total Grammy count equals 12 (exactly). Use reliable sources; accept official counts.",
        ),
    ]

    await evaluator.batch_verify(claims)


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
    """
    Entry point for evaluating the musician identification and career details task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root logical order: identify first, then career info
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify the correct musician for the specified halftime show and provide all required attributes with citations",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_musician_profile(),
        template_class=MusicianExtraction,
        extraction_name="musician_profile",
    )

    # Build a critical sequential "task main" node under root to strictly gate the two stages
    task_main = evaluator.add_sequential(
        id="task_main",
        desc="Identify the correct musician for the specified halftime show and provide all required attributes with citations",
        parent=root,
        critical=True,
    )

    # Stage 1: Identification (critical parallel checks)
    await build_identification_checks(evaluator, task_main, extraction)

    # Stage 2: Career information (critical parallel checks)
    await build_career_checks(evaluator, task_main, extraction)

    # Return the structured evaluation summary
    return evaluator.get_summary()