import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.eval_toolkit import create_evaluator, Extractor, Verifier
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "jojo_references"
TASK_DESCRIPTION = """
The JoJo's Bizarre Adventure is a Japanese manga/anime series with 9 parts. In parts 4‐6, which characters or Stands are named after songs or albums released in the 1970s?
Please find five examples and list the name of the character or Stand along with the specific reference (song or album). For each, include the artist or band who created the song or album, the year it was released, and a link to its Spotify page.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class JojoCharacter(BaseModel):
    """Basic information about a JoJo character/Stand."""
    name: Optional[str] = None
    part: Optional[str] = None  # Part 4, 5, or 6

class JojoCharacterList(BaseModel):
    """List of JoJo characters/Stands from the answer."""
    characters: List[JojoCharacter] = Field(default_factory=list)

class MusicReference(BaseModel):
    """Single JoJo character/Stand music reference with related information."""
    character_or_stand_name: Optional[str] = None
    reference_type: Optional[str] = None  # "song" or "album"
    reference_name: Optional[str] = None  # Name of the song or album
    artist_or_band: Optional[str] = None
    release_year: Optional[str] = None
    spotify_url: Optional[str] = None

class JojoReferenceDetails(BaseModel):
    """Details of a reference for a specific JoJo character/Stand."""
    reference: Optional[MusicReference] = None

class ExternalLink(BaseModel):
    """URLs for verification."""
    url: Optional[str] = None
    description: Optional[str] = None

class ExternalLinks(BaseModel):
    """Collection of URLs for verification."""
    links: List[ExternalLink] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                         #
# --------------------------------------------------------------------------- #
def prompt_extract_jojo_characters() -> str:
    """Extract JoJo character/Stand names from the answer."""
    return """
    Extract all JoJo character or Stand names mentioned in the answer. 
    Focus only on characters or Stands that are explicitly mentioned as being named after songs or albums from the 1970s.
    
    For each character/Stand, extract:
    1. The name of the character or Stand
    2. The JoJo part it appeared in (should be part 4, 5, or 6)
    
    If any of these details are missing, set the corresponding field to null.
    Extract all characters/Stands mentioned, even if there are more than 5.
    """

def prompt_extract_reference_details(character_name: str) -> str:
    """Extract reference details for a specific character/Stand."""
    return f"""
    Extract the music reference details for the JoJo character/Stand named '{character_name}' from the answer.
    
    Extract:
    1. Whether it references a song or album
    2. The name of the song or album
    3. The artist or band who created it
    4. The release year (should be in the 1970s: 1970-1979)
    5. The Spotify URL if provided
    
    If any of these details are missing in the answer, set the corresponding field to null.
    """

def prompt_extract_verification_urls(music_ref: MusicReference) -> str:
    """Extract URLs for verification of a specific music reference."""
    return f"""
    Extract all URLs mentioned in the answer that could be used to verify information about:
    1. The character/Stand '{music_ref.character_or_stand_name}'
    2. The {music_ref.reference_type} '{music_ref.reference_name}'
    3. The artist/band '{music_ref.artist_or_band}'
    
    For each URL, provide a brief description of what it verifies.
    Include the Spotify URL if present, as well as any other URLs that might help verify the JoJo character/Stand reference.
    
    If no URLs are provided in the answer, return an empty list.
    """

def prompt_find_year_verification_urls(music_ref: MusicReference) -> str:
    """Find URLs that can verify the release year."""
    return f"""
    Extract any URLs mentioned in the answer that could help verify that the {music_ref.reference_type} '{music_ref.reference_name}' by '{music_ref.artist_or_band}' was released in {music_ref.release_year}.
    
    Focus specifically on URLs that might contain release date information, such as:
    - Spotify pages
    - Music database websites
    - Artist official websites
    - Wikipedia or other encyclopedia pages
    
    For each URL, provide a brief description of what it verifies.
    If no appropriate URLs are found, return an empty list.
    """

# --------------------------------------------------------------------------- #
# Reference verification functions                                            #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Reference verification functions                                            #
# --------------------------------------------------------------------------- #
async def verify_single_reference(
    evaluator: Evaluator,
    parent_node,
    reference: MusicReference,
    index: int,
    verification_urls: List[str]
) -> None:
    """
    Verify all aspects of a single JoJo reference.
    """
    # Create sequential parent node for this reference
    reference_node = evaluator.add_parallel(
        id=f"reference_{index+1}",
        desc=f"Reference {index+1}: '{reference.character_or_stand_name}' is named after '{reference.reference_name}' by {reference.artist_or_band} ({reference.release_year})",
        parent=parent_node
    )
    
    # 1. Verify reference completeness (prerequisite)
    is_complete = (
        reference.character_or_stand_name is not None and 
        reference.reference_type is not None and 
        reference.reference_name is not None and 
        reference.artist_or_band is not None and 
        reference.release_year is not None and 
        reference.spotify_url is not None
    )
    
    completeness_node = evaluator.add_custom_node(
        result=is_complete,
        id=f"reference_{index+1}_completeness",
        desc=f"Reference {index+1} includes all required information: character/Stand name, reference type, reference name, artist/band, release year, and Spotify URL",
        parent=reference_node,
        critical=True
    )
    
    # 2. Verify character is from JoJo parts 4-6
    character_node = evaluator.add_leaf(
        id=f"reference_{index+1}_character_from_parts_4_to_6",
        desc=f"'{reference.character_or_stand_name}' is a character or Stand from JoJo parts 4-6",
        parent=reference_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"'{reference.character_or_stand_name}' is a character or Stand from JoJo's Bizarre Adventure parts 4, 5, or 6.",
        node=character_node,
        sources=verification_urls,
        additional_instruction="Verify if the character or Stand mentioned is indeed from JoJo's Bizarre Adventure parts 4, 5, or 6. Look for explicit mentions of which part the character appears in."
    )
    
    # 3. Verify release year is in 1970s
    year_node = evaluator.add_leaf(
        id=f"reference_{index+1}_year_in_1970s",
        desc=f"'{reference.reference_name}' was released in the 1970s (1970-1979)",
        parent=reference_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The {reference.reference_type} '{reference.reference_name}' by '{reference.artist_or_band}' was released in {reference.release_year}, which is between 1970 and 1979 inclusive.",
        node=year_node,
        sources=verification_urls,
        additional_instruction="Verify if the release year falls within the 1970s (1970-1979) and if it's accurate for the specific song or album. Look for explicit mention of the release date. If no release date is mentioned on the page, return false"
    )
    
    # 4. Verify Spotify URL content
    spotify_node = evaluator.add_leaf(
        id=f"reference_{index+1}_spotify_url_content",
        desc=f"Spotify URL leads to '{reference.reference_name}' by '{reference.artist_or_band}'",
        parent=reference_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The Spotify URL '{reference.spotify_url}' leads to '{reference.reference_name}' by '{reference.artist_or_band}'.",
        node=spotify_node,
        sources=reference.spotify_url,
        additional_instruction="Verify if the Spotify page matches the claimed song/album and artist. Check if the song/album title and artist name visible on the Spotify page match what was claimed."
    )
    
    # 5. Verify named after reference
    naming_node = evaluator.add_leaf(
        id=f"reference_{index+1}_named_after_reference",
        desc=f"'{reference.character_or_stand_name}' is named after the {reference.reference_type} '{reference.reference_name}'",
        parent=reference_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The character or Stand '{reference.character_or_stand_name}' from JoJo's Bizarre Adventure is named after the {reference.reference_type} '{reference.reference_name}' by {reference.artist_or_band}.",
        node=naming_node,
        sources=verification_urls,
        additional_instruction="Verify if the character or Stand is explicitly stated to be named after the specific song or album. Look for clear statements about the naming inspiration."
    )
    
    # 6. Verify artist accuracy
    artist_node = evaluator.add_leaf(
        id=f"reference_{index+1}_artist_accuracy",
        desc=f"'{reference.reference_name}' was created by '{reference.artist_or_band}'",
        parent=reference_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The {reference.reference_type} '{reference.reference_name}' was created by '{reference.artist_or_band}'.",
        node=artist_node,
        sources=verification_urls,
        additional_instruction="Verify if the artist or band mentioned is indeed the creator of the specific song or album referenced. Look for explicit mention of the artist name associated with the song or album."
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
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Initialize evaluator ------------------------------------ #
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
    
    # -------- 2. First extract JoJo characters/Stands ------------------- #
    jojo_characters = await evaluator.extract(
        prompt=prompt_extract_jojo_characters(),
        template_class=JojoCharacterList,
        extraction_name="jojo_characters"
    )
    
    # -------- 3. Then extract detailed references for each character ----- #
    complete_references = []
    verification_urls_by_reference = {}
    
    for character in jojo_characters.characters:
        if character.name:
            # Extract reference details for this character
            reference_details = await evaluator.extract(
                prompt=prompt_extract_reference_details(character.name),
                template_class=JojoReferenceDetails,
                extraction_name=f"reference_details_{character.name}"
            )
            
            if reference_details.reference:
                # Ensure character name is set
                reference_details.reference.character_or_stand_name = character.name
                
                # Extract verification URLs for this reference
                external_links = await evaluator.extract(
                    prompt=prompt_extract_verification_urls(reference_details.reference),
                    template_class=ExternalLinks,
                    extraction_name=f"verification_urls_{character.name}"
                )
                
                # Also extract specific URLs for year verification
                year_links = await evaluator.extract(
                    prompt=prompt_find_year_verification_urls(reference_details.reference),
                    template_class=ExternalLinks,
                    extraction_name=f"year_urls_{character.name}"
                )
                
                # Combine all URLs
                all_urls = [link.url for link in external_links.links]
                if reference_details.reference.spotify_url:
                    all_urls.append(reference_details.reference.spotify_url)
                all_urls.extend([link.url for link in year_links.links])
                
                # Remove duplicates while preserving order
                unique_urls = []
                for url in all_urls:
                    if url not in unique_urls:
                        unique_urls.append(url)
                
                # Store the reference and its verification URLs
                complete_references.append(reference_details.reference)
                verification_urls_by_reference[character.name] = unique_urls
    
    # -------- 4. Pad missing references with empty objects -------------- #
    required_references_count = 5
    while len(complete_references) < required_references_count:
        complete_references.append(MusicReference())
    
    # -------- 5. Verify each reference uniformly ------------------------ #
    for i, reference in enumerate(complete_references[:required_references_count]):
        urls_for_verification = verification_urls_by_reference.get(
            reference.character_or_stand_name, 
            []
        )
        await verify_single_reference(
            evaluator,
            root, 
            reference, 
            i, 
            urls_for_verification
        )
    
    # -------- 6. Add custom info and get results ------------------------ #
    evaluator.add_custom_info({
        "references_count": len([r for r in complete_references if r.character_or_stand_name]),
        "references": [ref.dict() for ref in complete_references[:5] if ref.character_or_stand_name],
        "verification_urls": verification_urls_by_reference
    }, "extracted_references")
    
    # -------- 7. Return structured result ------------------------------- #
    return evaluator.get_summary()