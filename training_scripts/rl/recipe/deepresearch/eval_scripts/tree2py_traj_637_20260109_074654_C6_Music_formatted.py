import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "complete_music_industry_research"
TASK_DESCRIPTION = (
    "Identify a music producer who won the Grammy Award for Producer of the Year, Non-Classical, at least once between "
    "2023 and 2025 (inclusive). This producer must have produced at least one Grammy-nominated album for an artist who "
    "won the Grammy Award for Best New Artist in 2022 or later. Provide the producer's full name. Then, specify one album "
    "that this producer worked on for a Grammy Best New Artist winner (who won in 2022 or later). Include the album title, "
    "artist name, and release year. The album must have received at least one Grammy nomination in any category. Next, "
    "identify a concert performance by that album artist that took place in the United States during 2023 or 2024. Provide the "
    "concert date (month and year is sufficient) and the venue name. For the concert venue, provide complete specifications including: "
    "the venue's official name, the city and state where it is located, and the venue's total seating capacity. The venue must be a "
    "major indoor arena or stadium with a capacity between 15,000 and 20,000 seats (inclusive). Include URL references for: "
    "(1) the producer's Grammy win, (2) the producer's work on the album, (3) the album's Grammy nomination, "
    "(4) evidence of the concert performance, and (5) the venue's capacity specifications."
)


class ProducerInfo(BaseModel):
    name: Optional[str] = None
    grammy_win_year: Optional[str] = None
    grammy_win_urls: List[str] = Field(default_factory=list)


class AlbumInfo(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None
    release_year: Optional[str] = None
    producer_credit_urls: List[str] = Field(default_factory=list)
    album_nomination_urls: List[str] = Field(default_factory=list)
    artist_best_new_artist_year: Optional[str] = None
    artist_best_new_artist_urls: List[str] = Field(default_factory=list)


class ConcertInfo(BaseModel):
    artist: Optional[str] = None
    date: Optional[str] = None
    venue_name: Optional[str] = None
    evidence_urls: List[str] = Field(default_factory=list)


class VenueInfo(BaseModel):
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)


class MusicIndustryExtraction(BaseModel):
    producer: Optional[ProducerInfo] = None
    album: Optional[AlbumInfo] = None
    concert: Optional[ConcertInfo] = None
    venue: Optional[VenueInfo] = None


def prompt_extract_music_industry() -> str:
    return (
        "Extract the following structured information exactly as it appears in the answer. If any item is not present, "
        "return null or an empty list accordingly.\n\n"
        "Producer:\n"
        "- name: The full name of the producer.\n"
        "- grammy_win_year: The year (four digits) the producer won the Grammy Award 'Producer of the Year, Non-Classical' "
        "between 2023 and 2025 inclusive, if the answer specifies a particular year; otherwise null.\n"
        "- grammy_win_urls: A list of URLs provided in the answer that document the producer's Grammy win for "
        "'Producer of the Year, Non-Classical'. Include only URLs explicitly present in the answer.\n\n"
        "Album:\n"
        "- title: The album title.\n"
        "- artist: The album artist name.\n"
        "- release_year: The album release year (as stated; use the exact text, typically a four-digit year).\n"
        "- producer_credit_urls: A list of URLs provided in the answer that show the producer's credits/work on this album.\n"
        "- album_nomination_urls: A list of URLs provided in the answer that document at least one Grammy nomination for this album.\n"
        "- artist_best_new_artist_year: If the answer states a year for the artist's 'Best New Artist' Grammy win, extract that year; "
        "otherwise null.\n"
        "- artist_best_new_artist_urls: A list of URLs provided in the answer that document the artist's 'Best New Artist' win, if any.\n\n"
        "Concert:\n"
        "- artist: The performing artist for the concert (should match the album artist).\n"
        "- date: The concert date (month and year is sufficient, e.g., 'May 2023').\n"
        "- venue_name: The concert venue name.\n"
        "- evidence_urls: A list of URLs provided in the answer that serve as evidence of the concert performance.\n\n"
        "Venue:\n"
        "- official_name: The venue's official name.\n"
        "- city: The city in which the venue is located.\n"
        "- state: The state in which the venue is located (use the state abbreviation or full name as provided).\n"
        "- capacity: The venue's total seating capacity stated as a numeric value. Extract the numeric text only as it appears "
        "(e.g., '19,000' or '19000'). If multiple capacities are provided, extract the main stated capacity.\n"
        "- capacity_urls: A list of URLs provided in the answer that document the venue's seating capacity specifications.\n\n"
        "Rules:\n"
        "- Extract only explicit information from the answer. Do not invent or infer missing details.\n"
        "- For all URL fields, extract only valid URLs explicitly present in the answer (plain or markdown), and include the full protocol.\n"
        "- If any field is missing, set it to null (or empty list for URL lists).\n"
    )


def _parse_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(20\d{2})", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _parse_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d{4,6})", text)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return int(num)
    except Exception:
        return None


async def verify_producer_task(
    evaluator: Evaluator,
    parent_node,
    producer: Optional[ProducerInfo]
) -> None:
    node = evaluator.add_parallel(
        id="Producer_Identification_Task",
        desc="Provide a producer who won Producer of the Year, Non-Classical between 2023–2025 and provide required evidence URL.",
        parent=parent_node,
        critical=True,
    )

    name_provided = bool(producer and producer.name and producer.name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id="Producer_Name_Provided",
        desc="Producer's full name is stated.",
        parent=node,
        critical=True
    )

    url_provided = bool(producer and producer.grammy_win_urls and len(producer.grammy_win_urls) > 0)
    evaluator.add_custom_node(
        result=url_provided,
        id="Producer_Grammy_Win_URL",
        desc="URL is provided documenting the producer's Grammy win (Producer of the Year, Non-Classical).",
        parent=node,
        critical=True
    )

    grammy_leaf = evaluator.add_leaf(
        id="Producer_Grammy_Win_2023_2025",
        desc="Producer won Grammy Award for Producer of the Year, Non-Classical at least once between 2023 and 2025 (inclusive).",
        parent=node,
        critical=True
    )

    producer_name = producer.name if producer and producer.name else ""
    year_num = _parse_year(producer.grammy_win_year if producer else None)
    if year_num:
        claim = f"{producer_name} won the Grammy Award 'Producer of the Year, Non-Classical' in {year_num}."
        add_ins = (
            "Verify on the provided URL(s) that the producer won 'Producer of the Year, Non-Classical' in the stated year. "
            "Minor name variations are acceptable."
        )
    else:
        claim = (
            f"{producer_name} won the Grammy Award 'Producer of the Year, Non-Classical' at least once between 2023 and 2025 (inclusive)."
        )
        add_ins = (
            "Check the provided URL(s) to confirm at least one win for 'Producer of the Year, Non-Classical' occurred in 2023, 2024, or 2025."
        )

    await evaluator.verify(
        claim=claim,
        node=grammy_leaf,
        sources=(producer.grammy_win_urls if producer else []),
        additional_instruction=add_ins
    )


async def verify_album_task(
    evaluator: Evaluator,
    parent_node,
    album: Optional[AlbumInfo],
    producer: Optional[ProducerInfo]
) -> None:
    node = evaluator.add_parallel(
        id="Album_Identification_Task",
        desc="Provide one album the producer worked on for an artist who won Best New Artist in 2022 or later; album must have at least one Grammy nomination; include required fields and URLs.",
        parent=parent_node,
        critical=True
    )

    title_provided = bool(album and album.title and album.title.strip())
    evaluator.add_custom_node(
        result=title_provided,
        id="Album_Title_Provided",
        desc="Album title is stated.",
        parent=node,
        critical=True
    )

    artist_provided = bool(album and album.artist and album.artist.strip())
    evaluator.add_custom_node(
        result=artist_provided,
        id="Album_Artist_Provided",
        desc="Album artist name is stated.",
        parent=node,
        critical=True
    )

    release_year_provided = bool(album and album.release_year and album.release_year.strip())
    evaluator.add_custom_node(
        result=release_year_provided,
        id="Album_Release_Year_Provided",
        desc="Album release year is stated.",
        parent=node,
        critical=True
    )

    bna_leaf = evaluator.add_leaf(
        id="Album_Artist_Best_New_Artist_2022_Plus",
        desc="The album artist won the Grammy Award for Best New Artist in 2022 or later.",
        parent=node,
        critical=True
    )
    artist_name = album.artist if album and album.artist else ""
    bna_year_num = _parse_year(album.artist_best_new_artist_year if album else None)
    if bna_year_num:
        bna_claim = f"{artist_name} won the Grammy Award for Best New Artist in {bna_year_num}, which is 2022 or later."
        bna_add_ins = (
            "Confirm the Best New Artist win year and ensure it is 2022 or later. Minor name variations acceptable."
        )
    else:
        bna_claim = f"{artist_name} won the Grammy Award for Best New Artist in 2022 or later."
        bna_add_ins = (
            "Verify the claim using any provided source(s) if available; otherwise rely on the statement. Ensure the win year is 2022+."
        )
    await evaluator.verify(
        claim=bna_claim,
        node=bna_leaf,
        sources=(album.artist_best_new_artist_urls if album else []),
        additional_instruction=bna_add_ins
    )

    credit_url_provided = bool(album and album.producer_credit_urls and len(album.producer_credit_urls) > 0)
    evaluator.add_custom_node(
        result=credit_url_provided,
        id="Producer_Album_Credit_URL",
        desc="URL is provided showing the producer's work/credits on the album.",
        parent=node,
        critical=True
    )

    worked_leaf = evaluator.add_leaf(
        id="Producer_Worked_On_Album",
        desc="The identified producer has production credits on the identified album.",
        parent=node,
        critical=True
    )
    producer_name = producer.name if producer and producer.name else "the producer"
    album_title = album.title if album and album.title else "the album"
    album_artist = album.artist if album and album.artist else "the artist"
    worked_claim = (
        f"{producer_name} has production credits on the album '{album_title}' by {album_artist}."
    )
    worked_add_ins = (
        "Verify on the provided URL(s) that the producer is credited on the album. Credits such as 'producer', 'co-producer', "
        "'executive producer' are acceptable."
    )
    await evaluator.verify(
        claim=worked_claim,
        node=worked_leaf,
        sources=(album.producer_credit_urls if album else []),
        additional_instruction=worked_add_ins
    )

    nomination_url_provided = bool(album and album.album_nomination_urls and len(album.album_nomination_urls) > 0)
    evaluator.add_custom_node(
        result=nomination_url_provided,
        id="Album_Grammy_Nomination_URL",
        desc="URL is provided documenting at least one Grammy nomination for the album.",
        parent=node,
        critical=True
    )

    nomination_leaf = evaluator.add_leaf(
        id="Album_Has_Grammy_Nomination",
        desc="The identified album received at least one Grammy nomination in any category.",
        parent=node,
        critical=True
    )
    nomination_claim = f"The album '{album_title}' by {album_artist} received at least one Grammy nomination."
    nomination_add_ins = (
        "Check the provided URL(s) for any Grammy nomination associated with the album. Any category counts."
    )
    await evaluator.verify(
        claim=nomination_claim,
        node=nomination_leaf,
        sources=(album.album_nomination_urls if album else []),
        additional_instruction=nomination_add_ins
    )


async def verify_concert_task(
    evaluator: Evaluator,
    parent_node,
    concert: Optional[ConcertInfo],
    album: Optional[AlbumInfo]
) -> None:
    node = evaluator.add_parallel(
        id="Concert_Performance_Task",
        desc="Identify a US concert performance by the album artist in 2023 or 2024 and provide required fields and evidence URL.",
        parent=parent_node,
        critical=True
    )

    evidence_url_provided = bool(concert and concert.evidence_urls and len(concert.evidence_urls) > 0)
    evaluator.add_custom_node(
        result=evidence_url_provided,
        id="Concert_Evidence_URL",
        desc="URL is provided as evidence of the concert performance.",
        parent=node,
        critical=True
    )

    date_provided = bool(concert and concert.date and concert.date.strip())
    evaluator.add_custom_node(
        result=date_provided,
        id="Concert_Date_Provided",
        desc="Concert date is provided (month and year is sufficient).",
        parent=node,
        critical=True
    )

    venue_name_provided = bool(concert and concert.venue_name and concert.venue_name.strip())
    evaluator.add_custom_node(
        result=venue_name_provided,
        id="Concert_Venue_Name_Provided",
        desc="Concert venue name is stated.",
        parent=node,
        critical=True
    )

    performed_leaf = evaluator.add_leaf(
        id="Concert_Performed_By_Album_Artist",
        desc="The concert performance is by the same artist as the identified album artist.",
        parent=node,
        critical=True
    )
    album_artist = album.artist if album and album.artist else ""
    performed_claim = (
        f"The concert evidence indicates the performance is by {album_artist} (the same artist as the identified album)."
    )
    performed_add_ins = (
        "On the provided concert evidence URL(s), confirm that the performer is the same artist as the album artist. "
        "Allow minor spelling/casing variations."
    )
    await evaluator.verify(
        claim=performed_claim,
        node=performed_leaf,
        sources=(concert.evidence_urls if concert else []),
        additional_instruction=performed_add_ins
    )

    year_check_leaf = evaluator.add_leaf(
        id="Concert_In_2023_Or_2024",
        desc="Concert took place in 2023 or 2024.",
        parent=node,
        critical=True
    )
    extracted_year = _parse_year(concert.date if concert else None)
    if extracted_year:
        year_claim = f"The concert took place in {extracted_year}, which should be either 2023 or 2024."
        year_add_ins = (
            "Use the provided concert evidence URL(s) to verify the event year and confirm it is 2023 or 2024."
        )
    else:
        year_claim = "The concert took place in either 2023 or 2024."
        year_add_ins = (
            "Verify from the provided concert evidence URL(s) that the concert year is 2023 or 2024. "
            "Month-year formats are acceptable."
        )
    await evaluator.verify(
        claim=year_claim,
        node=year_check_leaf,
        sources=(concert.evidence_urls if concert else []),
        additional_instruction=year_add_ins
    )

    venue_us_leaf = evaluator.add_leaf(
        id="Concert_Venue_In_United_States",
        desc="Concert venue is located in the United States.",
        parent=node,
        critical=True
    )
    venue_name = concert.venue_name if concert and concert.venue_name else ""
    venue_us_claim = f"The concert venue '{venue_name}' is located in the United States."
    venue_us_add_ins = (
        "Confirm the venue location on the provided concert evidence URL(s). Look for the city and state or an explicit country reference."
    )
    await evaluator.verify(
        claim=venue_us_claim,
        node=venue_us_leaf,
        sources=(concert.evidence_urls if concert else []),
        additional_instruction=venue_us_add_ins
    )


async def verify_venue_task(
    evaluator: Evaluator,
    parent_node,
    venue: Optional[VenueInfo],
    concert: Optional[ConcertInfo]
) -> None:
    node = evaluator.add_parallel(
        id="Venue_Specifications_Task",
        desc="Provide venue specifications (official name, city/state, capacity) meeting the capacity range and venue-type requirement, with evidence URL.",
        parent=parent_node,
        critical=True
    )

    official_name = venue.official_name if venue and venue.official_name else ""
    concert_venue_name = concert.venue_name if concert and concert.venue_name else ""

    match_leaf = evaluator.add_leaf(
        id="Venue_Official_Name_Matches_Concert_Venue",
        desc="Venue official name is provided and corresponds to the concert venue identified in the prior step.",
        parent=node,
        critical=True
    )
    match_claim = (
        f"The venue official name '{official_name}' refers to the same venue as the concert venue '{concert_venue_name}'."
    )
    match_add_ins = (
        "Consider minor naming variations (e.g., 'Arena' vs. 'Center', sponsor prefixes/suffixes). "
        "Judge whether they refer to the same physical venue."
    )
    await evaluator.verify(
        claim=match_claim,
        node=match_leaf,
        sources=None,
        additional_instruction=match_add_ins
    )

    city_state_provided = bool(venue and venue.city and venue.city.strip() and venue.state and venue.state.strip())
    evaluator.add_custom_node(
        result=city_state_provided,
        id="Venue_City_And_State_Provided",
        desc="Venue city and state are stated.",
        parent=node,
        critical=True
    )

    type_leaf = evaluator.add_leaf(
        id="Venue_Is_Major_Indoor_Arena_Or_Stadium",
        desc="Venue is a major indoor arena or stadium (as described by the source(s)).",
        parent=node,
        critical=True
    )
    type_claim = (
        f"The venue '{official_name}' is described as an indoor arena or stadium on the provided source(s)."
    )
    type_add_ins = (
        "Check the capacity/venue source URL(s) description for terms like 'indoor arena', 'arena', or 'stadium'. "
        "The classification should indicate a major arena/stadium."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=(venue.capacity_urls if venue else []),
        additional_instruction=type_add_ins
    )

    capacity_num = _parse_int(venue.capacity if venue else None)
    capacity_provided = capacity_num is not None
    evaluator.add_custom_node(
        result=capacity_provided,
        id="Venue_Seating_Capacity_Provided",
        desc="Venue total seating capacity is stated as a numeric value.",
        parent=node,
        critical=True
    )

    in_range = bool(capacity_num is not None and 15000 <= capacity_num <= 20000)
    evaluator.add_custom_node(
        result=in_range,
        id="Venue_Capacity_15000_20000",
        desc="Venue seating capacity is between 15,000 and 20,000 inclusive.",
        parent=node,
        critical=True
    )

    capacity_url_provided = bool(venue and venue.capacity_urls and len(venue.capacity_urls) > 0)
    evaluator.add_custom_node(
        result=capacity_url_provided,
        id="Venue_Capacity_URL",
        desc="URL is provided documenting the venue's seating capacity specifications.",
        parent=node,
        critical=True
    )


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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_music_industry(),
        template_class=MusicIndustryExtraction,
        extraction_name="music_industry_info"
    )

    top = evaluator.add_sequential(
        id="Complete_Music_Industry_Research",
        desc=("Identify a qualifying Grammy-winning producer (2023–2025), one qualifying Grammy-nominated album for a Best New Artist winner (2022+), "
              "a qualifying US concert (2023–2024), and a qualifying venue (15,000–20,000 capacity) with required URLs."),
        parent=root,
        critical=True
    )

    await verify_producer_task(evaluator, top, extracted.producer)
    await verify_album_task(evaluator, top, extracted.album, extracted.producer)
    await verify_concert_task(evaluator, top, extracted.concert, extracted.album)
    await verify_venue_task(evaluator, top, extracted.venue, extracted.concert)

    return evaluator.get_summary()