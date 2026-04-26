import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "actors_books_2000_2020"
TASK_DESCRIPTION = (
    "Identify three professional actors who published both a memoir or non-fiction book AND a fiction work "
    "(novel or short story collection) between 2000 and 2020, where the memoir/non-fiction was published before "
    "the fiction work. For each actor, provide: (1) their name, (2) the title, publication year, and publisher of "
    "their memoir/non-fiction book, (3) the title, publication year, and publisher of their fiction work, and "
    "(4) a reference URL for each book from an official publisher website, major book retailer (such as Amazon, "
    "Barnes & Noble, or Goodreads), or established literary database."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookInfo(BaseModel):
    title: Optional[str] = None
    year: Optional[str] = None
    publisher: Optional[str] = None
    url: Optional[str] = None
    category: Optional[str] = None  # e.g., "memoir", "non-fiction", "novel", "short story collection"


class ActorEntry(BaseModel):
    name: Optional[str] = None
    identity_url: Optional[str] = None  # e.g., IMDb, Wikipedia page for actor
    memoir: Optional[BookInfo] = None
    fiction: Optional[BookInfo] = None


class ActorsExtraction(BaseModel):
    actors: List[ActorEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_actors() -> str:
    return """
    Extract up to three actors and the required book details exactly as they appear in the answer.

    For each actor, return the following structure:
    - name: The actor's full name (string)
    - identity_url: One reference URL proving professional film/TV actor status (IMDb, Wikipedia, or other credible industry page). Extract an actual URL if present; otherwise null.
    - memoir: An object for the memoir/non-fiction book with fields:
        * title: Book title as stated
        * year: Publication year
        * publisher: Publisher name
        * url: A reference URL for the book from an official publisher site, major book retailer (Amazon, Barnes & Noble, Goodreads), or established literary database; extract the actual URL if present; otherwise null.
        * category: A short label describing type, e.g., "memoir", "non-fiction", "autobiography", "essays"
    - fiction: An object for the fiction work with fields:
        * title: Book title as stated
        * year: Publication year
        * publisher: Publisher name
        * url: A reference URL for the book from an official publisher site, major book retailer (Amazon, Barnes & Noble, Goodreads), or established literary database; extract the actual URL if present; otherwise null.
        * category: A short label describing type, e.g., "novel", "novella", "short story collection"

    Rules:
    - Do not invent any information; only extract what is explicitly present in the answer.
    - If the answer lists more than three actors, include only the first three.
    - If any field is missing, set it to null.
    - For URLs, extract the actual link (plaintext or markdown); if not provided, return null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _to_int_year(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        return int(s.strip()[:4])
    except Exception:
        return None


def _year_in_range(y: Optional[int], start: int = 2000, end: int = 2020) -> bool:
    return y is not None and start <= y <= end


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def _verify_identity(
    evaluator: Evaluator,
    parent_node,
    actor: ActorEntry,
    idx: int
) -> None:
    identity_node = evaluator.add_parallel(
        id=f"actor_{idx}_identity",
        desc=f"Actor {idx}: Valid professional actor with verifiable acting career",
        parent=parent_node,
        critical=True
    )

    # Name provided (existence)
    evaluator.add_custom_node(
        result=bool(actor.name and actor.name.strip()),
        id=f"actor_{idx}_name",
        desc=f"Actor {idx}: Name provided",
        parent=identity_node,
        critical=True
    )

    # Identity URL: provided and credible source verifying actor status
    id_url_leaf = evaluator.add_leaf(
        id=f"actor_{idx}_identity_url",
        desc=f"Actor {idx}: Reference URL verifying actor status (e.g., IMDb, Wikipedia)",
        parent=identity_node,
        critical=True
    )
    id_url_claim = f"The provided identity URL is a credible page that verifies {actor.name} is a professional film/TV actor."
    await evaluator.verify(
        claim=id_url_claim,
        node=id_url_leaf,
        sources=actor.identity_url,
        additional_instruction=(
            "Judge false if the URL is missing or invalid. Credible identity sources include IMDb, Wikipedia, major studio/agency pages, or notable industry databases. "
            "Confirm the page actually corresponds to the named actor and indicates an acting career."
        )
    )

    # Profession confirmation via identity URL (actor status)
    profession_leaf = evaluator.add_leaf(
        id=f"actor_{idx}_profession",
        desc=f"Actor {idx}: Confirmed as professional film/TV actor",
        parent=identity_node,
        critical=True
    )
    profession_claim = f"The identity URL shows {actor.name} has a professional acting career in film or television."
    await evaluator.verify(
        claim=profession_claim,
        node=profession_leaf,
        sources=actor.identity_url,
        additional_instruction=(
            "Verify that the page evidences film/TV acting roles (filmography, credits). "
            "Minor or unrelated roles (e.g., only theater or voice if not film/TV) should be treated cautiously. "
            "If the page does not clearly show professional acting, judge false."
        )
    )


async def _verify_book_group(
    evaluator: Evaluator,
    parent_node,
    actor: ActorEntry,
    idx: int,
    kind: str,  # "memoir" or "fiction"
) -> None:
    book: BookInfo = getattr(actor, kind) or BookInfo()
    group_node = evaluator.add_parallel(
        id=f"actor_{idx}_{kind}",
        desc=f"Actor {idx}: {('Memoir/non-fiction' if kind=='memoir' else 'Fiction')} book details",
        parent=parent_node,
        critical=True
    )

    # Title provided (existence)
    evaluator.add_custom_node(
        result=bool(book.title and book.title.strip()),
        id=f"actor_{idx}_{kind}_title",
        desc=f"Actor {idx} {kind}: Title provided",
        parent=group_node,
        critical=True
    )

    # Publisher provided (existence)
    evaluator.add_custom_node(
        result=bool(book.publisher and book.publisher.strip()),
        id=f"actor_{idx}_{kind}_publisher",
        desc=f"Actor {idx} {kind}: Publisher name provided",
        parent=group_node,
        critical=True
    )

    # Reference URL provided and valid source for the specific book
    url_leaf = evaluator.add_leaf(
        id=f"actor_{idx}_{kind}_url",
        desc=f"Actor {idx} {kind}: Reference URL from official source provided",
        parent=group_node,
        critical=True
    )
    url_claim = (
        f"The provided URL is a valid official publisher page, major book retailer page (Amazon, Barnes & Noble, Goodreads, Bookshop), "
        f"or an established literary database page for the book '{book.title}' by {actor.name}."
    )
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=book.url,
        additional_instruction=(
            "Judge false if the URL is missing, invalid, unrelated, or from an untrusted source. "
            "Trusted examples: penguinrandomhouse.com, harpercollins.com, simonandschuster.com, hachettebookgroup.com, macmillan.com, amazon.com, barnesandnoble.com, goodreads.com, bookshop.org. "
            "Ensure the page corresponds to the stated book (title and author match or are clearly equivalent)."
        )
    )

    # Year between 2000-2020 and supported by the book URL
    year_leaf = evaluator.add_leaf(
        id=f"actor_{idx}_{kind}_year",
        desc=f"Actor {idx} {kind}: Publication year between 2000-2020",
        parent=group_node,
        critical=True
    )
    year_claim = (
        f"The book page indicates the publication year is {book.year}, and that year is between 2000 and 2020 inclusive."
    )
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=book.url,
        additional_instruction=(
            "Confirm the page lists the publication year matching the extracted value. "
            "If the page shows a different year or does not provide a year, judge false. "
            "Only years 2000-2020 inclusive are acceptable."
        )
    )

    # Type confirmation (category)
    type_leaf = evaluator.add_leaf(
        id=f"actor_{idx}_{kind}_type",
        desc=(
            f"Actor {idx} {kind}: Confirmed as "
            + ("memoir or non-fiction (not poetry, graphic novel, or children's picture book)"
               if kind == "memoir"
               else "novel or short story collection (not poetry, graphic novel, children's picture book, or primarily co-authored)")
        ),
        parent=group_node,
        critical=True
    )
    if kind == "memoir":
        type_claim = (
            f"The book is memoir or non-fiction (e.g., memoir, autobiography, essays) and not poetry, graphic novel, or a children's picture book."
        )
        type_instruction = (
            "Use the provided book URL to check the work's genre/category/description. "
            "Accept memoir, autobiography, personal essays, or other non-fiction narrative categories. "
            "Reject poetry collections, graphic novels/comics, children's picture books, or unclear categories."
        )
    else:
        type_claim = (
            f"The book is fiction as a novel, novella, or short story collection (not poetry, graphic novel, children's picture book, or primarily co-authored anthology)."
        )
        type_instruction = (
            "Use the provided book URL to confirm it is a novel or short story collection. "
            "Reject poetry collections, graphic novels/comics, children's picture books, or anthologies primarily co-authored unless the actor is clearly the sole or primary author."
        )

    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=book.url,
        additional_instruction=type_instruction
    )


async def _verify_chronology(
    evaluator: Evaluator,
    parent_node,
    actor: ActorEntry,
    idx: int
) -> None:
    # Simple logical check: memoir year < fiction year
    mem_year_int = _to_int_year(actor.memoir.year if actor.memoir else None)
    fic_year_int = _to_int_year(actor.fiction.year if actor.fiction else None)
    chronology_ok = (
        (mem_year_int is not None)
        and (fic_year_int is not None)
        and _year_in_range(mem_year_int)
        and _year_in_range(fic_year_int)
        and (mem_year_int < fic_year_int)
    )

    evaluator.add_custom_node(
        result=chronology_ok,
        id=f"actor_{idx}_chronology",
        desc=f"Actor {idx}: Memoir published before fiction work",
        parent=parent_node,
        critical=True
    )


async def verify_actor(
    evaluator: Evaluator,
    root_node,
    actor: ActorEntry,
    idx0_based: int
) -> None:
    idx = idx0_based + 1
    actor_node = evaluator.add_parallel(
        id=f"actor_{idx}",
        desc=f"{['First','Second','Third'][idx0_based]} actor who published memoir/non-fiction before fiction (2000-2020)",
        parent=root_node,
        critical=False
    )

    # Identity
    await _verify_identity(evaluator, actor_node, actor, idx)

    # Memoir group
    await _verify_book_group(evaluator, actor_node, actor, idx, kind="memoir")

    # Fiction group
    await _verify_book_group(evaluator, actor_node, actor, idx, kind="fiction")

    # Chronology
    await _verify_chronology(evaluator, actor_node, actor, idx)


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
    Evaluate an answer for the 'actors_books_2000_2020' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates three actors independently with partial credit
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

    # Extract structured actor/book info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_actors(),
        template_class=ActorsExtraction,
        extraction_name="actors_books_extraction"
    )

    # Prepare exactly 3 actors (pad if necessary)
    actors_list: List[ActorEntry] = list(extracted.actors[:3])
    while len(actors_list) < 3:
        actors_list.append(ActorEntry())

    # Optional: record allowed sources guidance
    evaluator.add_custom_info(
        info={
            "allowed_sources_examples": [
                "Official publisher websites",
                "amazon.com",
                "barnesandnoble.com",
                "goodreads.com",
                "bookshop.org",
                "penguinrandomhouse.com",
                "harpercollins.com",
                "simonandschuster.com",
                "hachettebookgroup.com",
                "macmillan.com"
            ],
            "year_range": "2000-2020 inclusive",
            "type_requirements": {
                "memoir": "memoir/non-fiction (accept autobiography, essays; reject poetry, graphic novel, children's picture book)",
                "fiction": "novel/novella/short story collection (reject poetry, graphic novel, children's picture book, co-authored anthologies)"
            }
        },
        info_type="guidelines",
        info_name="verification_guidelines"
    )

    # Verify each actor subtree
    for i, actor in enumerate(actors_list):
        await verify_actor(evaluator, root, actor, i)

    return evaluator.get_summary()