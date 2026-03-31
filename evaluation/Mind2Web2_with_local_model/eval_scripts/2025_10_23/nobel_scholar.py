import asyncio
import logging
from typing import Optional, List, Dict, Any

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nobel_scholar"
TASK_DESCRIPTION = """
Who won the 2024 Nobel Prize in Physics? Can you show me the Wikipedia link of each laureate individually? Then, for each of them, where did they receive their PhD? For each university, how many other Nobel laureates have graduated from there to date? Please also include the names and links to the official Nobel Prize biographies of those other laureates.
"""

JUDGE_MODEL = "o4-mini"

# Known ground truth
GROUND_TRUTH_LAUREATES = ["John J. Hopfield", "Geoffrey Hinton"]
GROUND_TRUTH_REASON = "their foundational discoveries and inventions that enable machine learning with artificial neural networks"

# Reference URLs for university laureate verification
CORNELL_REFERENCE_URL = "https://news.cornell.edu/content/nobel-laureates-affiliated-cornell-university"
EDINBURGH_REFERENCE_URL = "https://www.ed.ac.uk/about/people/prize-winners/nobel"


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class LaureateInfo(BaseModel):
    """Information about a single Nobel laureate."""
    name: Optional[str] = None
    wikipedia_url: Optional[str] = None
    phd_university: Optional[str] = None


class LaureatesExtraction(BaseModel):
    """Container for laureate information."""
    laureates: List[LaureateInfo] = Field(default_factory=list)


class OtherLaureateInfo(BaseModel):
    """Information about other Nobel laureates from the same university."""
    name: Optional[str] = None
    nobel_bio_url: Optional[str] = None


class UniversityInfo(BaseModel):
    """Information about a university and its Nobel laureates."""
    name: Optional[str] = None
    laureate_count: Optional[int] = None
    other_laureates: List[OtherLaureateInfo] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    """Container for university information."""
    universities: List[UniversityInfo] = Field(default_factory=list)


class UniversityLaureates(BaseModel):
    """Container for ground truth laureate information from a university."""
    university_name: str
    count: int
    laureate_names: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_laureates() -> str:
    return """
    Extract information about the 2024 Nobel Prize in Physics laureates from the answer.
    For each laureate, extract:
    1. Their full name
    2. The Wikipedia URL provided for them
    3. The university where they received their PhD

    Return this information in a structured format with a list of laureates, where each laureate has the fields:
    - name: The full name of the laureate
    - wikipedia_url: The Wikipedia URL for the laureate
    - phd_university: The name of the university where they received their PhD

    If any information is missing, use null for that field.
    """


def prompt_extract_universities() -> str:
    return """
    Extract information about the universities where the 2024 Nobel Prize in Physics laureates received their PhDs.
    For each university mentioned, extract:
    1. The name of the university
    2. The number of other Nobel laureates who have graduated from that university (as mentioned in the answer)
    3. The list of other laureates' names and their Nobel Prize biography URLs

    Return this information in a structured format with a list of universities, where each university has the fields:
    - name: The name of the university
    - laureate_count: The number of other Nobel laureates who graduated from this university
    - other_laureates: A list of objects, each with:
      - name: The name of the other laureate
      - nobel_bio_url: The URL to their official Nobel Prize biography

    If any information is missing, use null for that field.
    If the laureate count is mentioned but no specific laureates are listed, include the count but leave the other_laureates list empty.
    """


def prompt_extract_cornell_laureates() -> str:
    return """
    Extract a list of all Nobel laureates who have graduated from Cornell University, as mentioned on this webpage.

    For each laureate, extract their full name.

    Return this information in a structured format with:
    - university_name: "Cornell University"
    - count: The total number of Nobel laureates listed
    - laureate_names: A list of the full names of all Nobel laureates mentioned

    If the exact count is not explicitly stated, count the number of laureates you extract.
    If a laureate's affiliation with Cornell is unclear or they're not a graduate, do not include them even if they're listed on this page!
    
    A special case is John J. Hopfield who won 2024 Nobel Prize in Physics. Please ignore him and do not count or include him in the extraction result.
    """


def prompt_extract_edinburgh_laureates() -> str:
    return """
    Extract a list of all Nobel laureates who have graduated from the University of Edinburgh, as mentioned on this webpage.

    For each laureate, extract their full name.

    Return this information in a structured format with:
    - university_name: "University of Edinburgh"
    - count: The total number of Nobel laureates listed
    - laureate_names: A list of the full names of all Nobel laureates mentioned

    If the exact count is not explicitly stated, count the number of laureates you extract.
    If a laureate's affiliation with Edinburgh is unclear or they're not a graduate, do not include them even if they're listed on this page!
    
    A special case is Geoffrey Hinton who won 2024 Nobel Prize in Physics. Please ignore him and do not count or include him in the extraction result.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_url(url: str) -> str:
    """Normalize a URL by ensuring it has a protocol and removing trailing slashes."""
    if not url:
        return ""

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    return url.rstrip("/")


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_laureate_identification(
        evaluator: Evaluator,
        parent_node,
        laureates_info: LaureatesExtraction,
) -> None:
    """
    Verify that the 2024 Nobel Prize in Physics laureates are correctly identified.
    This is a critical node as it's fundamental to the task.
    """
    # Extract names of laureates from the answer
    extracted_names = [l.name for l in laureates_info.laureates if l.name]

    # Create verification node
    laureate_node = evaluator.add_leaf(
        id="laureate_identification",
        desc="The answer correctly identifies the 2024 Nobel Prize in Physics laureates: John J. Hopfield and Geoffrey Hinton",
        parent=parent_node,
        critical=True,
    )

    # Directly verify with LLM
    claim = f"The two names extracted from the answer {str(extracted_names)} match the ground truth list {str(GROUND_TRUTH_LAUREATES)}"
    await evaluator.verify(
        claim=claim,
        node=laureate_node,
        additional_instruction="The order doesn't matter. Accept reasonable name variations including different spellings (Geoffrey/Geoff), presence/absence of middle names or initials, as long as it clearly refers to the same person."
    )

async def verify_wikipedia_links(
        evaluator: Evaluator,
        parent_node,
        laureates_info: LaureatesExtraction,
) -> None:
    """
    Verify that the Wikipedia links for the laureates are correct and functional.
    """
    wiki_node = evaluator.add_parallel(
        id="wikipedia_links",
        desc="The answer provides correct Wikipedia links for each laureate",
        parent=parent_node,
        critical=True
    )

    # Prepare information for each laureate
    hopfield_info = next((l for l in laureates_info.laureates if l.name and "hopfield" in l.name.lower()), None)
    hinton_info = next((l for l in laureates_info.laureates if l.name and "hinton" in l.name.lower()), None)

    # Verify Hopfield's Wikipedia link
    hopfield_verification = evaluator.add_parallel(
        id="wiki_link_hopfield_verification",
        desc="Wikipedia link verification for John J. Hopfield",
        parent=wiki_node,
        critical=True
    )

    hopfield_exists = evaluator.add_custom_node(
        result=bool(hopfield_info and hopfield_info.wikipedia_url),
        id="hopfield_wiki_exists",
        desc="Check if Wikipedia URL for Hopfield was provided",
        parent=hopfield_verification,
        critical=True
    )

    hopfield_node = evaluator.add_leaf(
        id="wiki_link_hopfield",
        desc="The Wikipedia link for John J. Hopfield is correct and functional",
        parent=hopfield_verification,
        critical=True
    )

    url = normalize_url(hopfield_info.wikipedia_url) if hopfield_info and hopfield_info.wikipedia_url else None
    link_claim = f"This URL ({url}) is the Wikipedia page for John J. Hopfield"
    await evaluator.verify(
        claim=link_claim,
        node=hopfield_node,
        sources=url
    )

    # Verify Hinton's Wikipedia link
    hinton_verification = evaluator.add_parallel(
        id="wiki_link_hinton_verification",
        desc="Wikipedia link verification for Geoffrey Hinton",
        parent=wiki_node,
        critical=True
    )

    hinton_exists = evaluator.add_custom_node(
        result=bool(hinton_info and hinton_info.wikipedia_url),
        id="hinton_wiki_exists",
        desc="Check if Wikipedia URL for Hinton was provided",
        parent=hinton_verification,
        critical=True
    )

    hinton_node = evaluator.add_leaf(
        id="wiki_link_hinton",
        desc="The Wikipedia link for Geoffrey Hinton is correct and functional",
        parent=hinton_verification,
        critical=True
    )

    url = normalize_url(hinton_info.wikipedia_url) if hinton_info and hinton_info.wikipedia_url else None
    link_claim = f"This URL ({url}) is the Wikipedia page for Geoffrey Hinton"
    await evaluator.verify(
        claim=link_claim,
        node=hinton_node,
        sources=url
    )


async def verify_phd_institutions(
        evaluator: Evaluator,
        parent_node,
        laureates_info: LaureatesExtraction,
) -> None:
    """
    Verify that the PhD institutions for the laureates are correctly identified.
    """
    phd_node = evaluator.add_parallel(
        id="phd_institutions",
        desc="The answer correctly identifies where each laureate received their PhD",
        parent=parent_node,
        critical=True,
    )

    # Prepare information for each laureate
    hopfield_info = next((l for l in laureates_info.laureates if l.name and "hopfield" in l.name.lower()), None)
    hinton_info = next((l for l in laureates_info.laureates if l.name and "hinton" in l.name.lower()), None)

    # Verify Hopfield's PhD institution
    hopfield_phd_verification = evaluator.add_parallel(
        id="phd_hopfield_verification",
        desc="PhD institution verification for John J. Hopfield",
        parent=phd_node,
        critical=True
    )

    hopfield_phd_exists = evaluator.add_custom_node(
        result=bool(hopfield_info and hopfield_info.phd_university),
        id="hopfield_phd_exists",
        desc="Check if PhD university for Hopfield was provided",
        parent=hopfield_phd_verification,
        critical=True
    )

    hopfield_phd_node = evaluator.add_leaf(
        id="phd_institution_hopfield",
        desc="The PhD institution for John J. Hopfield is correctly identified",
        parent=hopfield_phd_verification,
        critical=True,
    )

    # Use Wikipedia to verify the PhD information
    hopfield_wiki = hopfield_info.wikipedia_url if hopfield_info and hopfield_info.wikipedia_url else "https://en.wikipedia.org/wiki/John_Hopfield"
    phd_claim = f"John J. Hopfield received his PhD from {hopfield_info.phd_university if hopfield_info else 'unknown'}"
    
    await evaluator.verify(
        claim=phd_claim,
        node=hopfield_phd_node,
        sources=normalize_url(hopfield_wiki)
    )

    # Verify Hinton's PhD institution
    hinton_phd_verification = evaluator.add_parallel(
        id="phd_hinton_verification",
        desc="PhD institution verification for Geoffrey Hinton",
        parent=phd_node,
        critical=True
    )

    hinton_phd_exists = evaluator.add_custom_node(
        result=bool(hinton_info and hinton_info.phd_university),
        id="hinton_phd_exists",
        desc="Check if PhD university for Hinton was provided",
        parent=hinton_phd_verification,
        critical=True
    )

    hinton_phd_node = evaluator.add_leaf(
        id="phd_institution_hinton",
        desc="The PhD institution for Geoffrey Hinton is correctly identified",
        parent=hinton_phd_verification,
        critical=True,
    )

    # Use Wikipedia to verify the PhD information
    hinton_wiki = hinton_info.wikipedia_url if hinton_info and hinton_info.wikipedia_url else "https://en.wikipedia.org/wiki/Geoffrey_Hinton"
    phd_claim = f"Geoffrey Hinton received his PhD from {hinton_info.phd_university if hinton_info else 'unknown'}"
    
    await evaluator.verify(
        claim=phd_claim,
        node=hinton_phd_node,
        sources=normalize_url(hinton_wiki)
    )


async def verify_single_laureate(
        evaluator: Evaluator,
        parent_node,
        laureate: OtherLaureateInfo,
        index: int,
        university_name: str,
        ground_truth_names: set,
        reference_url: str
) -> None:
    """
    Verify a single laureate from a university.
    """
    laureate_node = evaluator.add_parallel(
        id=f"{university_name.lower()}_laureate_{index}",
        desc=f"Laureate #{index + 1} {laureate.name if laureate.name else ''} from {university_name} is correctly identified with a Nobel Prize biography link",
        parent=parent_node,
        critical=False
    )

    # Combined existence check
    exists_node = evaluator.add_custom_node(
        result=bool(laureate.name and laureate.nobel_bio_url),
        id=f"{university_name.lower()}_laureate_{index}_exists",
        desc=f"Check if laureate #{index + 1} name and URL were provided",
        parent=laureate_node,
        critical=True
    )

    # Verify name
    name_node = evaluator.add_leaf(
        id=f"{university_name.lower()}_name_{index}",
        desc=f"The name of laureate #{index + 1} from {university_name} is accurate",
        parent=laureate_node,
        critical=True,
    )

    name_claim = f"{laureate.name if laureate.name else 'Unknown'} is a Nobel laureate who graduated from {university_name}"
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=reference_url
    )

    # Verify Nobel Prize biography link
    link_node = evaluator.add_leaf(
        id=f"{university_name.lower()}_link_{index}",
        desc=f"The Nobel Prize biography link for laureate #{index + 1} from {university_name} is valid",
        parent=laureate_node,
        critical=True,
    )

    url = normalize_url(laureate.nobel_bio_url) if laureate.nobel_bio_url else None
    link_claim = f"This URL ({url}) is an official Nobel Prize biography page for Nobel laureate {laureate.name if laureate.name else 'Unknown'}"
    await evaluator.verify(
        claim=link_claim,
        node=link_node,
        sources=url
    )


async def verify_other_laureates(
        evaluator: Evaluator,
        parent_node,
        universities_info: UniversitiesExtraction,
        laureates_info: LaureatesExtraction,
        university_ground_truths: Dict[str, UniversityLaureates],
) -> None:
    """
    Verify information about other Nobel laureates from each university.
    """
    other_laureates_node = evaluator.add_parallel(
        id="other_laureates",
        desc="The answer includes accurate names and Nobel Prize biography links for other laureates from each university",
        parent=parent_node,
        critical=False
    )

    # Create parallel nodes for each university
    cornell_node = evaluator.add_parallel(
        id="cornell_laureates",
        desc="The answer correctly identifies other Nobel laureates from Cornell University with their Nobel Prize biography links",
        parent=other_laureates_node,
        critical=False
    )

    edinburgh_node = evaluator.add_parallel(
        id="edinburgh_laureates",
        desc="The answer correctly identifies other Nobel laureates from University of Edinburgh with their Nobel Prize biography links",
        parent=other_laureates_node,
        critical=False
    )

    # Extract university information from the answer
    cornell_info = next((u for u in universities_info.universities if u.name and "cornell" in u.name.lower()), None)
    edinburgh_info = next((u for u in universities_info.universities if
                           u.name and ("edinburgh" in u.name.lower() or "edinburg" in u.name.lower())), None)

    # Get reference data
    cornell_ref = university_ground_truths.get("cornell")
    edinburgh_ref = university_ground_truths.get("edinburgh")

    # Process Cornell laureates
    if cornell_ref:
        ground_truth_names = set(name.lower() for name in cornell_ref.laureate_names)
        expected_count = len(ground_truth_names)

        # Get laureates from answer, pad if necessary
        cornell_laureates = cornell_info.other_laureates if cornell_info else []
        while len(cornell_laureates) < expected_count:
            cornell_laureates.append(OtherLaureateInfo())

        # Verify each laureate
        for i in range(expected_count):
            await verify_single_laureate(
                evaluator,
                cornell_node,
                cornell_laureates[i],
                i,
                "Cornell University",
                ground_truth_names,
                CORNELL_REFERENCE_URL
            )

    # Process Edinburgh laureates
    if edinburgh_ref:
        ground_truth_names = set(name.lower() for name in edinburgh_ref.laureate_names)
        expected_count = len(ground_truth_names)

        # Get laureates from answer, pad if necessary
        edinburgh_laureates = edinburgh_info.other_laureates if edinburgh_info else []
        while len(edinburgh_laureates) < expected_count:
            edinburgh_laureates.append(OtherLaureateInfo())

        # Verify each laureate
        for i in range(expected_count):
            await verify_single_laureate(
                evaluator,
                edinburgh_node,
                edinburgh_laureates[i],
                i,
                "University of Edinburgh",
                ground_truth_names,
                EDINBURGH_REFERENCE_URL
            )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
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
    
    # Initialize evaluator with parallel strategy for root
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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Establish ground truth data from reference URLs -------- #
    # Extract laureate information from Cornell University
    cornell_laureates = await evaluator.extract(
        prompt=prompt_extract_cornell_laureates(),
        template_class=UniversityLaureates,
        extraction_name="cornell_ground_truth",
        source=CORNELL_REFERENCE_URL
    )

    logger.info(f"Cornell Laureates ({cornell_laureates.count}): {cornell_laureates}")

    # Extract laureate information from University of Edinburgh
    edinburgh_laureates = await evaluator.extract(
        prompt=prompt_extract_edinburgh_laureates(),
        template_class=UniversityLaureates,
        extraction_name="edinburgh_ground_truth",
        source=EDINBURGH_REFERENCE_URL
    )
    
    logger.info(f"Edinburgh Laureates ({edinburgh_laureates.count}): {edinburgh_laureates}")

    # Store ground truth data for later use
    university_ground_truths = {
        "cornell": cornell_laureates,
        "edinburgh": edinburgh_laureates
    }

    # -------- 3. Extract structured info from the answer ---------------- #
    # Extract information about the laureates
    laureates_info = await evaluator.extract(
        prompt=prompt_extract_laureates(),
        template_class=LaureatesExtraction,
        extraction_name="laureates_info"
    )

    # Extract information about the universities
    universities_info = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_info"
    )

    # -------- 4. Build verification tree -------------------------------- #
    # First part: Laureate identification, Wikipedia links, and PhD institutions
    laureate_part = evaluator.add_sequential(
        id="laureate_part",
        desc="The answer correctly identifies the laureates, provides Wikipedia links, and identifies their PhD institutions",
        critical=False
    )

    # Verify the laureate identification
    await verify_laureate_identification(evaluator, laureate_part, laureates_info)

    # Verify Wikipedia links
    await verify_wikipedia_links(evaluator, laureate_part, laureates_info)

    # Verify PhD institutions
    await verify_phd_institutions(evaluator, laureate_part, laureates_info)

    # Verify other laureates and their Nobel Prize biography links using ground truth data
    await verify_other_laureates(
        evaluator,
        root,
        universities_info,
        laureates_info,
        university_ground_truths
    )

    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()