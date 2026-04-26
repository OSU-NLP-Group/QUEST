import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dino_2024_us"
TASK_DESCRIPTION = """
Identify three distinct dinosaur species that were formally described in peer-reviewed scientific journals in 2024 by researchers affiliated with universities or research institutions in the United States. For each species, provide: (1) the scientific name (genus and species), (2) at least one author's full name and their academic role (such as PhD student, faculty member, or research scientist), (3) the author's affiliated institution name, (4) the complete institutional address (including street address, city, state, and ZIP code), (5) the name of the peer-reviewed journal in which the species was described, (6) the publication date (month and year in 2024), and (7) a reference URL to either the institution's official news page about the discovery or the journal article itself.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class InstitutionAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None


class SpeciesEntry(BaseModel):
    scientific_name: Optional[str] = None  # binomial
    author_full_name: Optional[str] = None
    author_role: Optional[str] = None
    institution_name: Optional[str] = None
    institution_address: Optional[InstitutionAddress] = None
    journal_name: Optional[str] = None
    publication_date: Optional[str] = None  # "Month 2024" preferred
    # URLs explicitly provided in the answer for this species:
    reference_urls: List[str] = Field(default_factory=list)  # union of journal/news or any relevant sources mentioned
    journal_article_url: Optional[str] = None  # if explicitly provided in the answer
    institution_news_url: Optional[str] = None  # if explicitly provided in the answer
    # Optional: any official institutional page(s) that show the address (if explicitly provided in the answer)
    address_source_urls: List[str] = Field(default_factory=list)


class SpeciesListExtraction(BaseModel):
    species: List[SpeciesEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_species_list() -> str:
    return """
    From the provided answer text, extract up to 5 dinosaur species entries with the following fields for each:
    - scientific_name: The scientific binomial (genus + species), exactly as written in the answer.
    - author_full_name: At least one author's full name who is credited with the 2024 species description.
    - author_role: The academic role of that author (e.g., PhD student, doctoral candidate, faculty, assistant professor, research scientist). Use the exact phrasing from the answer when possible.
    - institution_name: The U.S. university or research institution named as the author's affiliation in the answer (use the exact name as appears).
    - institution_address: A JSON object with
        - street
        - city
        - state
        - zip
      Fill any missing component with null.
    - journal_name: Name of the peer-reviewed journal where the species was described.
    - publication_date: The publication month and year for the species' description (e.g., "March 2024"). If the answer provides a full date, extract the month and year; if only the year 2024 is provided, return "2024".
    - reference_urls: An array of all URLs explicitly mentioned in the answer for this species that are either:
        • the official institutional news page about the discovery, or
        • the journal article page (or the publisher/DOI landing page).
      Include all such URLs that appear in the answer, in the order they appear.
    - journal_article_url: If the answer explicitly provides a direct journal article or publisher landing page URL, put it here; otherwise null.
    - institution_news_url: If the answer explicitly provides an official institutional news page URL, put it here; otherwise null.
    - address_source_urls: Any official institutional "contact", "about", or similar page URLs explicitly cited in the answer that show the address; otherwise an empty array.

    RULES:
    1) Do not invent or infer information that is not explicitly present in the answer text. If a field is not present, return null (or [] for arrays).
    2) Only include URLs that appear in the answer text. If a URL is presented without protocol, prepend http://.
    3) Keep text exactly as in the answer (respect capitalization, punctuation) where practical.
    4) If multiple species are mentioned, extract them as multiple objects in the 'species' array, preserving order of appearance.

    Return a JSON object with a single field:
    {
      "species": [ ... up to 5 items ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def collect_all_sources(sp: SpeciesEntry) -> List[str]:
    urls: List[str] = []
    if sp.journal_article_url:
        urls.append(sp.journal_article_url)
    if sp.institution_news_url:
        urls.append(sp.institution_news_url)
    urls.extend(sp.reference_urls or [])
    return _unique_preserve_order(urls)


def format_full_address(addr: Optional[InstitutionAddress]) -> str:
    if not addr:
        return ""
    parts = []
    if addr.street: parts.append(addr.street.strip())
    locality = ", ".join(p for p in [addr.city or "", addr.state or ""] if p)
    if locality:
        parts.append(locality)
    if addr.zip:
        parts.append(addr.zip.strip())
    return ", ".join(parts)


def has_complete_address(addr: Optional[InstitutionAddress]) -> bool:
    if not addr:
        return False
    return all([
        isinstance(addr.street, str) and addr.street.strip() != "",
        isinstance(addr.city, str) and addr.city.strip() != "",
        isinstance(addr.state, str) and addr.state.strip() != "",
        isinstance(addr.zip, str) and addr.zip.strip() != "",
    ])


# --------------------------------------------------------------------------- #
# Verification for a single species                                           #
# --------------------------------------------------------------------------- #
async def verify_single_species(
    evaluator: Evaluator,
    parent_node,
    sp: SpeciesEntry,
    index: int,
) -> None:
    i = index + 1  # Human-readable 1-based index
    species_node = evaluator.add_parallel(
        id=f"Species_{i}",
        desc=f"{['First','Second','Third','Fourth','Fifth'][index] if index < 5 else f'Item #{i}'} dinosaur species with complete information",
        parent=parent_node,
        critical=False  # Parent species node allows partial credit across species
    )

    # Gather sources
    all_sources = collect_all_sources(sp)
    addr_sources = sp.address_source_urls if sp.address_source_urls else all_sources

    # ---------- Identification (critical group) ----------
    ident_node = evaluator.add_parallel(
        id=f"Species_{i}_Identification",
        desc="Correctly identify a dinosaur species described in 2024",
        parent=species_node,
        critical=True
    )

    # Scientific name leaf
    sci_leaf = evaluator.add_leaf(
        id=f"Species_{i}_Scientific_Name",
        desc="Provide the correct scientific binomial name (genus and species)",
        parent=ident_node,
        critical=True
    )
    sci_claim = f"This page describes a dinosaur species with the scientific name '{sp.scientific_name or ''}'."
    await evaluator.verify(
        claim=sci_claim,
        node=sci_leaf,
        sources=all_sources,
        additional_instruction="Verify that the exact binomial appears and refers to a dinosaur species described on this page. Allow minor punctuation/capitalization variations."
    )

    # Publication year 2024 leaf
    year_leaf = evaluator.add_leaf(
        id=f"Species_{i}_Publication_Year",
        desc="Verify the species was formally described in 2024",
        parent=ident_node,
        critical=True
    )
    year_claim = f"The species '{sp.scientific_name or ''}' was formally described and published in the year 2024."
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=all_sources,
        additional_instruction="Judge based on the actual publication year of the species description (not discovery or press release). Use the journal page if available; otherwise use credible text on the provided page."
    )

    # Peer review leaf
    pr_leaf = evaluator.add_leaf(
        id=f"Species_{i}_Peer_Review",
        desc="Verify publication in a peer-reviewed scientific journal",
        parent=ident_node,
        critical=True
    )
    pr_claim = "The species description was published in a peer-reviewed scientific journal."
    await evaluator.verify(
        claim=pr_claim,
        node=pr_leaf,
        sources=all_sources,
        additional_instruction="Confirm the venue is a recognized peer‑reviewed scientific journal (e.g., check publisher/journal page text). Institutional news explicitly stating it was published in a peer‑reviewed journal also suffices."
    )

    # ---------- Author info (critical group) ----------
    author_node = evaluator.add_parallel(
        id=f"Species_{i}_Author_Info",
        desc="Provide at least one author with U.S. university affiliation",
        parent=species_node,
        critical=True
    )

    # Author name
    auth_name_leaf = evaluator.add_leaf(
        id=f"Species_{i}_Author_Name",
        desc="Provide the author's full name",
        parent=author_node,
        critical=True
    )
    auth_name_claim = f"One of the authors of the species description for '{sp.scientific_name or ''}' is '{sp.author_full_name or ''}'."
    await evaluator.verify(
        claim=auth_name_claim,
        node=auth_name_leaf,
        sources=all_sources,
        additional_instruction="Check the author list on the journal/publisher page first; if unavailable, check institutional news text listing the author."
    )

    # Author role
    auth_role_leaf = evaluator.add_leaf(
        id=f"Species_{i}_Author_Role",
        desc="Specify the author's academic role (graduate student, faculty, etc.)",
        parent=author_node,
        critical=True
    )
    auth_role_claim = f"The author {sp.author_full_name or ''} has the academic role '{sp.author_role or ''}' (e.g., PhD student, doctoral candidate, faculty, or research scientist) in connection with this research."
    await evaluator.verify(
        claim=auth_role_claim,
        node=auth_role_leaf,
        sources=all_sources,
        additional_instruction="Look for explicit mentions like 'PhD student', 'doctoral candidate', 'assistant professor', 'research scientist', etc., on the provided pages."
    )

    # Institution name (and that it's U.S.-based)
    inst_name_leaf = evaluator.add_leaf(
        id=f"Species_{i}_Institution_Name",
        desc="Provide the affiliated U.S. university or research institution name",
        parent=author_node,
        critical=True
    )
    inst_claim = f"The author {sp.author_full_name or ''} is affiliated with '{sp.institution_name or ''}', which is a U.S. university or U.S. research institution."
    await evaluator.verify(
        claim=inst_claim,
        node=inst_name_leaf,
        sources=all_sources,
        additional_instruction="Verify both the institution name and that it is U.S.-based (e.g., .edu domain, U.S. state, or explicit statement)."
    )

    # ---------- Institution address (critical group) ----------
    addr_node = evaluator.add_parallel(
        id=f"Species_{i}_Institution_Address",
        desc="Provide the complete institutional address",
        parent=species_node,
        critical=True
    )

    # Address completeness (custom local check)
    complete = has_complete_address(sp.institution_address)
    evaluator.add_custom_node(
        result=complete,
        id=f"Species_{i}_Address_Complete",
        desc="Include street address, city, state, and ZIP code",
        parent=addr_node,
        critical=True
    )

    # Address verification against official sources
    addr_verify_leaf = evaluator.add_leaf(
        id=f"Species_{i}_Address_Verification",
        desc="Address must be verifiable through official sources",
        parent=addr_node,
        critical=True
    )
    full_addr = format_full_address(sp.institution_address)
    addr_claim = f"The official address for {sp.institution_name or 'the institution'} is '{full_addr}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_verify_leaf,
        sources=addr_sources,
        additional_instruction="Confirm this full mailing address on an official institutional page (e.g., contact/about/department page). If multiple campus addresses exist, the provided one must match an official page for that institution."
    )

    # ---------- Publication details (critical group) ----------
    pub_node = evaluator.add_parallel(
        id=f"Species_{i}_Publication_Details",
        desc="Provide complete publication information",
        parent=species_node,
        critical=True
    )

    # Journal name
    journal_leaf = evaluator.add_leaf(
        id=f"Species_{i}_Journal_Name",
        desc="Provide the journal name",
        parent=pub_node,
        critical=True
    )
    journal_claim = f"The species '{sp.scientific_name or ''}' was published in the journal '{sp.journal_name or ''}'."
    await evaluator.verify(
        claim=journal_claim,
        node=journal_leaf,
        sources=all_sources,
        additional_instruction="Prefer verifying on the journal/publisher page. If using institutional news, ensure it explicitly states the journal name."
    )

    # Publication date (month and year in 2024)
    pubdate_leaf = evaluator.add_leaf(
        id=f"Species_{i}_Publication_Date",
        desc="Provide the publication month and year (2024)",
        parent=pub_node,
        critical=True
    )
    pubdate_claim = f"The publication date for the species description is '{sp.publication_date or ''}', and it is in 2024."
    await evaluator.verify(
        claim=pubdate_claim,
        node=pubdate_leaf,
        sources=all_sources,
        additional_instruction="Match the publication month/year on the journal page if available. Accept full dates if they clearly indicate a month in 2024."
    )

    # ---------- Reference URL (critical leaf) ----------
    # If no URLs were provided at all, fail this leaf directly
    if not all_sources:
        evaluator.add_custom_node(
            result=False,
            id=f"Species_{i}_Reference_URL",
            desc="Provide a valid URL to the institutional news page or journal article",
            parent=species_node,
            critical=True
        )
    else:
        ref_leaf = evaluator.add_leaf(
            id=f"Species_{i}_Reference_URL",
            desc="Provide a valid URL to the institutional news page or journal article",
            parent=species_node,
            critical=True
        )
        ref_claim = f"This page is either (a) an official institutional news page about the discovery of '{sp.scientific_name or ''}', or (b) the scholarly journal article (or publisher landing page) where it was described."
        await evaluator.verify(
            claim=ref_claim,
            node=ref_leaf,
            sources=all_sources,
            additional_instruction="Accept .edu or clearly official research institution domains for news pages, or recognized journal/publisher/DOI pages for the article itself. The page should clearly concern this species."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'dino_2024_us' task using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates species in parallel
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

    # IMPORTANT: Make root non-critical to allow partial scoring across species (adjusting JSON criticality)
    root.critical = False

    # 1) Extract species list from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_species_list(),
        template_class=SpeciesListExtraction,
        extraction_name="species_extraction"
    )

    # 2) Select first three items; pad with empty entries if fewer than 3
    species_items: List[SpeciesEntry] = list(extracted.species[:3])
    while len(species_items) < 3:
        species_items.append(SpeciesEntry())

    # 3) Build tree per species
    for idx, sp in enumerate(species_items[:3]):
        await verify_single_species(evaluator, root, sp, idx)

    # 4) Return unified summary
    return evaluator.get_summary()