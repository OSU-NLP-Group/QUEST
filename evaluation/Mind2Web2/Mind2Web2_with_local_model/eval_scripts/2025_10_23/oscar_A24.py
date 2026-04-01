import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "oscar_A24"
TASK_DESCRIPTION = """
Over the past five Academy Awards, which nominated films are either produced or distributed by the company A24? For each film, I want to know its production budge, worldwide box office, the role of A24 (production or distribution), the year in which it was nominated, the specific award nominations, and whether it won each of those nominations.
"""

# Ground truth data for A24 films
GROUND_TRUTH_FILMS = {
    "Minari": {"year": "2021", "ceremony": "93rd Academy Awards"},
    "The Tragedy of Macbeth": {"year": "2022", "ceremony": "94th Academy Awards"},
    "Everything Everywhere All at Once": {"year": "2023", "ceremony": "95th Academy Awards"},
    "The Whale": {"year": "2023", "ceremony": "95th Academy Awards"},
    "Aftersun": {"year": "2023", "ceremony": "95th Academy Awards"},
    "Causeway": {"year": "2023", "ceremony": "95th Academy Awards"},
    "Close": {"year": "2023", "ceremony": "95th Academy Awards"},
    "Marcel the Shell with Shoes On": {"year": "2023", "ceremony": "95th Academy Awards"},
    "Past Lives": {"year": "2024", "ceremony": "96th Academy Awards"},
    "The Zone of Interest": {"year": "2024", "ceremony": "96th Academy Awards"},
    "The Brutalist": {"year": "2025", "ceremony": "97th Academy Awards"}
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmTitle(BaseModel):
    title: Optional[str] = None

class FilmTitles(BaseModel):
    films: List[FilmTitle] = Field(default_factory=list)

class AwardNomination(BaseModel):
    category: Optional[str] = None
    won: Optional[bool] = None

class FilmDetails(BaseModel):
    budget: Optional[str] = None
    box_office: Optional[str] = None
    a24_role: Optional[str] = None
    year_nominated: Optional[str] = None
    nominations: List[AwardNomination] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)

class FilmLinks(BaseModel):
    urls: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_film_titles() -> str:
    return """
    Extract all A24 films mentioned in the answer that were nominated for Academy Awards in the past five years.
    
    For each film, extract only the film title.
    Extract ALL films mentioned in the answer, even if there are more than expected.
    """

def prompt_extract_film_details(film_title: str) -> str:
    return f"""
    For the film "{film_title}", extract the following details:
    1. Production budget (if mentioned)
    2. Worldwide box office (if mentioned)  
    3. A24's role (production or distribution)
    4. Year the film was nominated
    5. List of specific award nominations and whether it won each nomination
    
    If any information is missing, return null for that field.
    """

def prompt_extract_film_links(film_title: str) -> str:
    return f"""
    Extract all URLs/links in the answer that are specifically related to information about the film "{film_title}".
    Only extract URLs that appear to provide information about this specific film's budget, box office, A24's role, 
    nominations, or wins.
    
    Be thorough and include any URL that might contain relevant information about this film.
    """

# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_film_title(title: str) -> str:
    """Normalize film title for more robust matching."""
    return title.lower().strip()

async def match_film_to_ground_truth(
    film_title: str,
    evaluator: Evaluator
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Match a film title to ground truth data using LLM-based verification.
    Returns (is_match, year, ceremony) tuple.
    """
    # Try to find a match in ground truth data
    for gt_title, gt_data in GROUND_TRUTH_FILMS.items():
        # Use verify to check if the film titles match
        claim = f"The film '{film_title}' and the film '{gt_title}' refer to the same movie."
        
        is_match = await evaluator.verify(
            claim=claim,
            node=None,  # Don't assign to any node
            additional_instruction="Consider that film titles might have slight variations, including abbreviations, punctuation differences, or missing/added articles. Focus on whether they refer to the same film."
        )
        
        if is_match:
            return True, gt_data["year"], gt_data["ceremony"]
    
    return False, None, None

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_film_identified(
    evaluator: Evaluator,
    parent_node,
    film_title: str,
    urls: List[str],
) -> Tuple[Any, bool, Optional[str], Optional[str]]:
    """
    Verify if the film is correctly identified as an A24 film nominated for Academy Awards.
    Returns (node, is_verified, ground_truth_year, ground_truth_ceremony) tuple.
    """
    # First check against ground truth using LLM-based verification
    is_match, gt_year, gt_ceremony = await match_film_to_ground_truth(film_title, evaluator)
    
    node = evaluator.add_leaf(
        id=f"verify_film_identified_{film_title}",
        desc=f"Verify that '{film_title}' is an A24 film nominated for Academy Awards in the past five years",
        parent=parent_node,
        critical=True,
    )
    
    if is_match:
        is_verified = True
        node.score = 1.0
        node.status = "passed"
    else:
        # If not in ground truth, verify through web
        claim = f"The film '{film_title}' was produced or distributed by A24 and was nominated for Academy Awards in the past five years."
        is_verified = await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
        )
    
    return node, is_verified, gt_year, gt_ceremony

async def verify_film_budget(
    evaluator: Evaluator,
    parent_node,
    film_title: str,
    budget: Optional[str],
    urls: List[str],
) -> None:
    """Verify the production budget of the film."""
    # Existence check
    budget_exists = evaluator.add_custom_node(
        result=budget is not None and budget.strip() != "",
        id=f"budget_exists_{film_title}",
        desc=f"Check if budget information was provided for '{film_title}'",
        parent=parent_node,
        critical=True
    )
    
    # Actual verification
    budget_node = evaluator.add_leaf(
        id=f"verify_budget_{film_title}",
        desc=f"Verify the production budget of '{film_title}'",
        parent=parent_node,
        critical=True,
    )
    
    claim = f"The production budget of '{film_title}' was {budget}."
    await evaluator.verify(
        claim=claim,
        node=budget_node,
        sources=urls,
    )

async def verify_film_box_office(
    evaluator: Evaluator,
    parent_node,
    film_title: str,
    box_office: Optional[str],
    urls: List[str],
) -> None:
    """Verify the worldwide box office of the film."""
    # Existence check
    box_office_exists = evaluator.add_custom_node(
        result=box_office is not None and box_office.strip() != "",
        id=f"box_office_exists_{film_title}",
        desc=f"Check if box office information was provided for '{film_title}'",
        parent=parent_node,
        critical=True
    )
    
    # Actual verification
    box_office_node = evaluator.add_leaf(
        id=f"verify_box_office_{film_title}",
        desc=f"Verify the worldwide box office of '{film_title}'",
        parent=parent_node,
        critical=True,
    )
    
    claim = f"The worldwide box office for '{film_title}' was {box_office}."
    await evaluator.verify(
        claim=claim,
        node=box_office_node,
        sources=urls,
    )

async def verify_a24_role(
    evaluator: Evaluator,
    parent_node,
    film_title: str,
    a24_role: Optional[str],
    urls: List[str],
) -> None:
    """Verify A24's role in the film (production or distribution)."""
    # Existence check
    role_exists = evaluator.add_custom_node(
        result=a24_role is not None and a24_role.strip() != "",
        id=f"a24_role_exists_{film_title}",
        desc=f"Check if A24's role information was provided for '{film_title}'",
        parent=parent_node,
        critical=True
    )
    
    # Actual verification
    role_node = evaluator.add_leaf(
        id=f"verify_a24_role_{film_title}",
        desc=f"Verify A24's role in '{film_title}' (production or distribution)",
        parent=parent_node,
        critical=True,
    )
    
    claim = f"A24's role in '{film_title}' was {a24_role}."
    await evaluator.verify(
        claim=claim,
        node=role_node,
        sources=urls,
    )

async def verify_year_nominated(
    evaluator: Evaluator,
    parent_node,
    film_title: str,
    year_nominated: Optional[str],
    gt_year: Optional[str],
    gt_ceremony: Optional[str],
    urls: List[str],
) -> None:
    """Verify the year the film was nominated for Academy Awards."""
    # Existence check
    year_exists = evaluator.add_custom_node(
        result=year_nominated is not None and year_nominated.strip() != "",
        id=f"year_exists_{film_title}",
        desc=f"Check if year nomination information was provided for '{film_title}'",
        parent=parent_node,
        critical=True
    )
    
    # Actual verification
    year_node = evaluator.add_leaf(
        id=f"verify_year_nominated_{film_title}",
        desc=f"Verify the year '{film_title}' was nominated for Academy Awards",
        parent=parent_node,
        critical=True,
    )
    
    # Always use ground truth verification since we only reach here if film was identified
    claim = f"The year '{year_nominated}' for film '{film_title}' corresponds to the correct Academy Awards year '{gt_year}'."
    await evaluator.verify(
        claim=claim,
        node=year_node,
        additional_instruction=f"The film was actually nominated at the {gt_year} Academy Awards (also known as the {gt_ceremony}). Consider that the answer might express the year in different formats, such as just the year, year with ceremony, or other variations."
    )

async def verify_nominations_and_wins(
    evaluator: Evaluator,
    parent_node,
    film_title: str,
    nominations: List[AwardNomination],
    urls: List[str],
) -> None:
    """Verify all nominations and wins for the film."""
    # Create verification parent
    nominations_parent = evaluator.add_parallel(
        id=f"nominations_verification_{film_title}",
        desc=f"Academy Award nominations and wins verification for '{film_title}'",
        parent=parent_node,
        critical=True,
    )
    
    # Filter valid nominations upfront
    valid_nominations = [
        n for n in (nominations or [])
        if n.category and n.category.strip()
    ]
    
    # Existence check - now includes URLs check
    nominations_exist = evaluator.add_custom_node(
        result=bool(valid_nominations) and bool(urls),
        id=f"nominations_exist_{film_title}",
        desc=f"Check if valid nomination information and source URLs were provided for '{film_title}'",
        parent=nominations_parent,
        critical=True
    )
    
    # Single node to verify ALL nominations
    all_nominations_node = evaluator.add_leaf(
        id=f"verify_all_nominations_{film_title}",
        desc=f"Verify all Academy Award nominations and wins for '{film_title}'",
        parent=nominations_parent,
        critical=True,
    )
    
    # Always proceed with verification - let the framework handle gating
    try:
        results = []
        for nom in valid_nominations:
            win_status = "won" if nom.won else "was nominated for but did not win"
            claim = f"The film '{film_title}' {win_status} the Academy Award for '{nom.category}'."
            
            result = await evaluator.verify(
                claim=claim,
                node=None,  # Don't assign to any node
                sources=urls,
                additional_instruction="Verify both the nomination category and win/loss outcome."
            )
            results.append(result)
        
        # Set final score (all-or-nothing)
        all_nominations_node.score = 1.0 if results and all(results) else 0.0
        all_nominations_node.status = "passed" if results and all(results) else "failed"
        
    except Exception as e:
        evaluator.verifier.logger.error(f"Error verifying nominations for '{film_title}': {e}")
        all_nominations_node.score = 0.0
        all_nominations_node.status = "failed"


async def verify_film_details(
    evaluator: Evaluator,
    parent_node,
    film_title: str,
    film_details: FilmDetails,
) -> None:
    """Verify all details for a single film."""
    # Create a SEQUENTIAL parent for the entire film
    film_node = evaluator.add_sequential(
        id=f"film_{film_title}",
        desc=f"All verifications for '{film_title}'",
        parent=parent_node,
        critical=False,
    )
    
    # First verify this is indeed an A24 film nominated for Academy Awards
    identification_node, is_verified, gt_year, gt_ceremony = await verify_film_identified(
        evaluator, film_node, film_title, film_details.source_urls
    )
    
    # Now add all detail verifications directly to film_node
    # They will be skipped automatically if identification fails!
    await verify_film_budget(evaluator, film_node, film_title, film_details.budget, film_details.source_urls)
    await verify_film_box_office(evaluator, film_node, film_title, film_details.box_office, film_details.source_urls)
    await verify_a24_role(evaluator, film_node, film_title, film_details.a24_role, film_details.source_urls)
    await verify_year_nominated(evaluator, film_node, film_title, film_details.year_nominated, gt_year, gt_ceremony, film_details.source_urls)
    await verify_nominations_and_wins(evaluator, film_node, film_title, film_details.nominations, film_details.source_urls)

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
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator
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

    # -------- 2. First extract just the film titles ---------------------- #
    film_titles = await evaluator.extract(
        prompt=prompt_extract_film_titles(),
        template_class=FilmTitles,
        extraction_name="film_titles"
    )
    
    # Dictionary to store complete film information
    films_data = {}

    
    # -------- 3. For each film, extract details and links ---------------- #
    for film in film_titles.films:

        if len(films_data) == len(GROUND_TRUTH_FILMS):
            break

        if not film.title:
            continue
            
        # Extract details for this specific film
        film_details = await evaluator.extract(
            prompt=prompt_extract_film_details(film.title),
            template_class=FilmDetails,
            extraction_name=f"film_details_{film.title}"
        )
        
        # Extract specific links for this film
        film_links = await evaluator.extract(
            prompt=prompt_extract_film_links(film.title),
            template_class=FilmLinks,
            extraction_name=f"film_links_{film.title}"
        )
        
        # Add extracted links to the film's source_urls
        for url in film_links.urls:
            if url not in film_details.source_urls:
                film_details.source_urls.append(url)
        
        # Store the complete film data
        films_data[film.title] = film_details

    # -------- 4. Build verification tree -------------------------------- #
    
    # Verify details for each film
    valid_count = 0
    for film_title, film_details in films_data.items():
        await verify_film_details(evaluator, root, film_title, film_details)
        valid_count += 1
    
    while valid_count < len(GROUND_TRUTH_FILMS):
        await verify_film_details(evaluator, root, None, FilmDetails())

    # Add ground truth info for reference
    evaluator.add_ground_truth({
        "ground_truth_films": GROUND_TRUTH_FILMS
    })

    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()