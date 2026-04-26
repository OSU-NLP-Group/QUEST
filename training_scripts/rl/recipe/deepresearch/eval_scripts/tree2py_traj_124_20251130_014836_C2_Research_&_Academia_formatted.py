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
TASK_ID = "mars_lightning_nature_2025"
TASK_DESCRIPTION = (
    "In November 2025, Nature journal published a groundbreaking study on the detection of electrical "
    "discharges (lightning) on Mars, based on data from NASA's Perseverance rover. Identify the lead author "
    "of this research paper, their primary institutional affiliation, and the geographic location (city and country) "
    "of that research institution."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StudyExtraction(BaseModel):
    """Structured extraction of key facts and cited sources from the answer."""
    # Core paper identification
    paper_title: Optional[str] = None
    journal: Optional[str] = None
    publication_date_text: Optional[str] = None  # e.g., "November 2025"
    rover_or_mission: Optional[str] = None       # e.g., "NASA's Perseverance rover"
    topic_keywords: List[str] = Field(default_factory=list)  # e.g., ["lightning", "electrical discharge", "Mars"]

    # Required outputs (claimed by the answer)
    lead_author: Optional[str] = None
    primary_affiliation: Optional[str] = None
    institution_city: Optional[str] = None
    institution_country: Optional[str] = None

    # Cited sources
    nature_urls: List[str] = Field(default_factory=list)
    institutional_urls: List[str] = Field(default_factory=list)
    nasa_urls: List[str] = Field(default_factory=list)
    all_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_study() -> str:
    return (
        "Extract the specific facts and URLs the answer claims for the described Nature paper. "
        "Return a JSON with the following fields:\n"
        "1) paper_title: the paper title as stated in the answer (if present)\n"
        "2) journal: the journal name as stated (e.g., 'Nature')\n"
        "3) publication_date_text: the publication month/year text (e.g., 'November 2025') as stated in the answer\n"
        "4) rover_or_mission: the rover/mission used (e.g., 'NASA's Perseverance rover') if stated\n"
        "5) topic_keywords: keywords describing the topic, focusing on 'lightning'/'electrical discharges' on Mars\n"
        "6) lead_author: the lead (first) author name as claimed in the answer\n"
        "7) primary_affiliation: the lead author's primary institutional affiliation as claimed\n"
        "8) institution_city: the city of that institution as claimed\n"
        "9) institution_country: the country of that institution as claimed\n"
        "10) nature_urls: all Nature journal URLs explicitly mentioned in the answer that refer to the paper\n"
        "11) institutional_urls: official institutional URLs explicitly mentioned (e.g., the institution's site or the author's profile)\n"
        "12) nasa_urls: NASA URLs explicitly mentioned in the answer (e.g., NASA news or mission pages)\n"
        "13) all_sources: list of all URLs explicitly mentioned in the answer, including any above\n\n"
        "Rules:\n"
        "- Extract ONLY what is explicitly present in the answer. Do not invent missing data.\n"
        "- For URLs, extract actual URLs from the answer text (including markdown links). If none are present for a field, return an empty list.\n"
        "- If a field is missing in the answer, return null (or empty list for URL fields).\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    cleaned = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


def _combine_sources(extracted: StudyExtraction, preferred_groups: List[List[str]]) -> Optional[List[str]]:
    """
    Combine URL lists in order of preference. The first non-empty group is returned after de-duplication.
    If all are empty, return None.
    """
    for group in preferred_groups:
        combined = []
        for attr_name in group:
            list_val = getattr(extracted, attr_name, []) or []
            combined.extend(list_val)
        combined = _dedupe_urls(combined)
        if combined:
            return combined
    return None


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_mars_lightning_task(
    evaluator: Evaluator,
    parent_node,
    extracted: StudyExtraction,
) -> None:
    """
    Build the sequential critical verification nodes and execute verifications.
    """
    # Create the task node as a critical sequential aggregator under root
    task_node = evaluator.add_sequential(
        id="Mars_Lightning_Research_Location_Task",
        desc="Identify the lead author, their primary institutional affiliation, and the institution's city and country for the Nature paper described in the question.",
        parent=parent_node,
        critical=True,
    )

    # 1) Paper identification and constraints check (Critical leaf)
    paper_check_node = evaluator.add_leaf(
        id="Paper_Identification_Meets_All_Constraints",
        desc=("Identifies the correct paper and confirms it meets ALL stated constraints: published in Nature journal, "
              "publication date in November 2025, about Mars lightning/electrical discharge detection, and based on data "
              "from NASA's Perseverance rover."),
        parent=task_node,
        critical=True,
    )
    # Prefer Nature URLs; fallback to all sources if Nature URLs are missing
    paper_sources = _combine_sources(extracted, [
        ["nature_urls"],
        ["all_sources"]
    ])

    paper_claim = (
        "This page is a Nature journal article published in November 2025 that reports detection of electrical "
        "discharges (lightning) on Mars based on data from NASA's Perseverance rover."
    )
    await evaluator.verify(
        claim=paper_claim,
        node=paper_check_node,
        sources=paper_sources,
        additional_instruction=(
            "Verify ALL constraints simultaneously on the page: "
            "1) It is in 'Nature' (the flagship journal), "
            "2) Publication month is November 2025 (exact month; day may vary), "
            "3) Topic explicitly involves electrical discharges/lightning on Mars, "
            "4) The study leverages data from NASA's Perseverance rover (Mars 2020 mission). "
            "If any constraint is not supported, judge as not supported."
        ),
    )

    # 2) Lead author identification (Critical leaf)
    lead_author_node = evaluator.add_leaf(
        id="Lead_Author_Identification",
        desc="Correctly identifies the lead (first) author from the paper's author list.",
        parent=task_node,
        critical=True,
    )
    lead_author_name = extracted.lead_author or ""
    lead_author_claim = f"The first (lead) author of this Nature paper is '{lead_author_name}'."
    await evaluator.verify(
        claim=lead_author_claim,
        node=lead_author_node,
        sources=extracted.nature_urls if extracted.nature_urls else None,
        additional_instruction=(
            "Check the author list on the Nature article page. The first-listed author is considered the lead author. "
            "Allow minor variations such as middle initials, accent marks, or formatting differences."
        ),
    )

    # 3) Primary institutional affiliation (Critical leaf)
    affiliation_node = evaluator.add_leaf(
        id="Primary_Institutional_Affiliation",
        desc="Correctly identifies the lead author's primary institutional affiliation (as indicated in the paper and/or official institutional sources).",
        parent=task_node,
        critical=True,
    )
    primary_affiliation_text = extracted.primary_affiliation or ""
    affiliation_claim = f"The lead author's primary institutional affiliation is '{primary_affiliation_text}'."
    affiliation_sources = _combine_sources(extracted, [
        ["nature_urls", "institutional_urls"],
        ["institutional_urls"],
        ["all_sources"]
    ])
    await evaluator.verify(
        claim=affiliation_claim,
        node=affiliation_node,
        sources=affiliation_sources,
        additional_instruction=(
            "Use affiliations listed beside the lead author's name on the Nature page; if multiple affiliations are listed, "
            "consider the primary or first-listed one. Official institutional pages may also confirm the affiliation."
        ),
    )

    # 4) Institution geographic location (Critical leaf)
    location_node = evaluator.add_leaf(
        id="Institution_Geographic_Location",
        desc="Correctly provides the institution's geographic location including BOTH city and country.",
        parent=task_node,
        critical=True,
    )
    city = extracted.institution_city or ""
    country = extracted.institution_country or ""
    institution_name = extracted.primary_affiliation or ""
    location_claim = f"'{institution_name}' is located in {city}, {country}."
    location_sources = _combine_sources(extracted, [
        ["institutional_urls"],
        ["nature_urls", "institutional_urls"],
        ["all_sources"]
    ])
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=location_sources,
        additional_instruction=(
            "Confirm the institution's city and country from official institutional pages or the Nature article's affiliation section. "
            "Allow common naming variants (e.g., 'United States' vs 'USA')."
        ),
    )

    # 5) Verifiability check (Critical leaf)
    verifiability_node = evaluator.add_leaf(
        id="Verifiability_Check",
        desc="All provided facts (lead author, primary affiliation, city, country) are verifiable against the published paper and/or official institutional sources.",
        parent=task_node,
        critical=True,
    )
    verifiability_sources = _combine_sources(extracted, [
        ["nature_urls", "institutional_urls"],
        ["institutional_urls"],
        ["nature_urls"],
        ["all_sources"]
    ])
    verifiability_claim = (
        f"The following facts are fully supported by the cited sources: "
        f"lead author = '{lead_author_name}', primary affiliation = '{primary_affiliation_text}', "
        f"location = '{city}, {country}'."
    )
    await evaluator.verify(
        claim=verifiability_claim,
        node=verifiability_node,
        sources=verifiability_sources,
        additional_instruction=(
            "Judge whether a single provided source page fully supports all listed facts together. "
            "If no single page confirms all facts (lead author, affiliation, city, country), this check should fail."
        ),
    )


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
) -> Dict[str, Any]:
    """
    Evaluate the answer for the Mars Lightning Nature 2025 task.
    """
    # Initialize evaluator (root is non-critical by framework; we add a critical child node)
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
        default_model=model,
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_study(),
        template_class=StudyExtraction,
        extraction_name="study_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_mars_lightning_task(
        evaluator=evaluator,
        parent_node=root,
        extracted=extracted,
    )

    # Return standardized summary
    return evaluator.get_summary()