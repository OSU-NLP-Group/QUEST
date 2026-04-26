import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "aanda_3i_atlas_2025_letter"
TASK_DESCRIPTION = (
    "In 2025, a peer-reviewed research paper was published in the journal Astronomy & Astrophysics that specifically "
    "focused on the temporal evolution of the interstellar comet 3I/ATLAS, reporting spectroscopic and photometric "
    "observations conducted during July 2025. Identify this paper and provide the following information: (1) The name of "
    "the lead author (first author listed); (2) The lead author's primary institutional affiliation (the first institution "
    "listed in the affiliations); (3) The country where this primary institution is located; (4) The name of the telescope "
    "facility that conducted spectroscopic observations on July 15, 2025; (5) The rotation period of 3I/ATLAS as measured "
    "and reported in the paper, including the uncertainty. For each piece of information, provide the URL of the source "
    "that confirms your answer."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AandaAtlasPaperExtraction(BaseModel):
    paper_urls: List[str] = Field(default_factory=list)

    lead_author_name: Optional[str] = None
    lead_author_source_urls: List[str] = Field(default_factory=list)

    primary_affiliation_name: Optional[str] = None
    primary_affiliation_source_urls: List[str] = Field(default_factory=list)

    institution_country: Optional[str] = None
    institution_country_source_urls: List[str] = Field(default_factory=list)

    spectroscopy_facility_name: Optional[str] = None
    spectroscopy_observation_date: Optional[str] = None
    spectroscopy_facility_source_urls: List[str] = Field(default_factory=list)

    rotation_period_with_uncertainty: Optional[str] = None
    rotation_period_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_info() -> str:
    return (
        "Extract structured information about the identified Astronomy & Astrophysics (A&A) paper concerning the "
        "interstellar comet 3I/ATLAS from the provided answer.\n"
        "Return the following fields:\n"
        "1) paper_urls: List of URLs provided that point to the paper or its official landing pages (e.g., A&A page, publisher page, arXiv if applicable). "
        "Extract only valid full URLs mentioned in the answer text. If none are provided, return an empty list.\n"
        "2) lead_author_name: The name of the lead author (first author listed in the paper). If not explicitly stated, return null.\n"
        "3) lead_author_source_urls: URLs that directly confirm the lead author (e.g., the paper page or PDF showing the author list). Return an empty list if none.\n"
        "4) primary_affiliation_name: The lead author's primary institutional affiliation (the first affiliation listed in the paper). If not stated, return null.\n"
        "5) primary_affiliation_source_urls: URLs that directly confirm the primary affiliation (e.g., affiliations block in the paper). Return an empty list if none.\n"
        "6) institution_country: The country where the lead author's primary institution is located. If not stated, return null.\n"
        "7) institution_country_source_urls: URLs that support the country of the institution (paper affiliation block or authoritative institutional page). Return an empty list if none.\n"
        "8) spectroscopy_facility_name: The telescope facility used for spectroscopic observations on July 15, 2025. If not explicitly stated, return null.\n"
        "9) spectroscopy_observation_date: The date for the spectroscopic observation (e.g., 'July 15, 2025'). If not stated, return null.\n"
        "10) spectroscopy_facility_source_urls: URLs that confirm the facility and date (paper page/PDF observation log/text). Return an empty list if none.\n"
        "11) rotation_period_with_uncertainty: The rotation period of 3I/ATLAS as reported in the paper, including the ± uncertainty (e.g., '8.0 ± 0.5 hours'). If not stated, return null.\n"
        "12) rotation_period_source_urls: URLs that confirm the rotation period with its uncertainty (paper page/PDF). Return an empty list if none.\n"
        "Important: Do not invent information. Only extract what is explicitly present in the answer. For URLs, only extract valid URLs that appear in the answer; do not construct or infer new URLs."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_urls(primary: List[str], fallback: List[str]) -> Optional[List[str]]:
    """Combine URLs, de-duplicate, and return None if empty."""
    s = list(dict.fromkeys((primary or []) + (fallback or [])))
    return s if s else None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_identify_correct_paper_nodes(
    evaluator: Evaluator,
    parent: VerificationNode,
    extracted: AandaAtlasPaperExtraction
) -> VerificationNode:
    """
    Build the 'identify_correct_paper' subtree and perform verifications.
    """
    node = evaluator.add_parallel(
        id="identify_correct_paper",
        desc="Correctly identify the target paper that satisfies the stated publication and topic constraints, and provide a source URL/DOI.",
        parent=parent,
        critical=True
    )

    # Existence check for URLs/DOI (treated as critical gate for subsequent verifications)
    urls_exist = bool(extracted.paper_urls and len(extracted.paper_urls) > 0)
    evaluator.add_custom_node(
        result=urls_exist,
        id="paper_url_or_doi_provided",
        desc="A valid URL or DOI is provided that resolves to the identified paper.",
        parent=node,
        critical=True
    )

    paper_sources = extracted.paper_urls if extracted.paper_urls else None

    # Create leaf nodes
    aanda_leaf = evaluator.add_leaf(
        id="paper_journal_is_aanda",
        desc="Identified paper is published in Astronomy & Astrophysics (A&A).",
        parent=node,
        critical=True
    )
    year_leaf = evaluator.add_leaf(
        id="paper_published_in_2025",
        desc="Identified paper is published in 2025.",
        parent=node,
        critical=True
    )
    letter_leaf = evaluator.add_leaf(
        id="paper_is_letter_to_editor",
        desc="Identified paper is a Letter to the Editor.",
        parent=node,
        critical=True
    )
    topic_leaf = evaluator.add_leaf(
        id="paper_topic_matches",
        desc="Identified paper specifically focuses on the temporal evolution/characterization of interstellar comet 3I/ATLAS.",
        parent=node,
        critical=True
    )
    july_obs_leaf = evaluator.add_leaf(
        id="paper_includes_july_2025_obs",
        desc="Identified paper reports spectroscopic and photometric observations conducted during July 2025.",
        parent=node,
        critical=True
    )

    # Prepare batch claims
    claims_and_sources = [
        (
            "This paper is published in Astronomy & Astrophysics (A&A).",
            paper_sources,
            aanda_leaf,
            "Verify the journal name on the article landing page or PDF; look for 'Astronomy & Astrophysics' or 'A&A'."
        ),
        (
            "This paper was published in 2025.",
            paper_sources,
            year_leaf,
            "Confirm the publication year on the article page or PDF; accept '2025' as the publication year."
        ),
        (
            "This paper is categorized as a 'Letter to the Editor' (A&A Letters).",
            paper_sources,
            letter_leaf,
            "Look for an explicit 'Letter to the Editor' label or 'A&A Letters' categorization on the page or PDF."
        ),
        (
            "The paper specifically focuses on the temporal evolution and characterization of the interstellar comet 3I/ATLAS.",
            paper_sources,
            topic_leaf,
            "Check the title/abstract/body for explicit focus on '3I/ATLAS' and its temporal evolution or characterization."
        ),
        (
            "The paper reports spectroscopic and photometric observations conducted during July 2025.",
            paper_sources,
            july_obs_leaf,
            "Look for an observation log or text indicating spectroscopic and photometric observations in July 2025."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)
    return node


async def build_extract_required_information_nodes(
    evaluator: Evaluator,
    parent: VerificationNode,
    extracted: AandaAtlasPaperExtraction
) -> VerificationNode:
    """
    Build the 'extract_required_information' subtree and perform verifications for all requested fields.
    """
    node = evaluator.add_parallel(
        id="extract_required_information",
        desc="Extract and report all requested fields from the identified paper, each with a confirming URL.",
        parent=parent,
        critical=True
    )

    # Lead author group
    lead_group = evaluator.add_parallel(
        id="lead_author",
        desc="Provide the lead author (first author listed) and a URL that confirms it.",
        parent=node,
        critical=True
    )
    lead_author_sources = _combine_urls(extracted.lead_author_source_urls, extracted.paper_urls)
    lead_name_leaf = evaluator.add_leaf(
        id="lead_author_name_correct",
        desc="Lead author name matches the first author listed in the paper.",
        parent=lead_group,
        critical=True
    )
    lead_name_claim = f"The lead (first) author listed on the paper is '{extracted.lead_author_name or ''}'."
    await evaluator.verify(
        claim=lead_name_claim,
        node=lead_name_leaf,
        sources=lead_author_sources,
        additional_instruction="Check the author list on the paper page or PDF; allow minor variants (middle initials, accents)."
    )
    evaluator.add_custom_node(
        result=bool(extracted.lead_author_source_urls),
        id="lead_author_source_url",
        desc="A URL is provided that directly supports the lead author claim (e.g., paper page/PDF author list).",
        parent=lead_group,
        critical=True
    )

    # Primary affiliation group
    affil_group = evaluator.add_parallel(
        id="primary_affiliation",
        desc="Provide the lead author's primary institutional affiliation (first affiliation listed) and a URL that confirms it.",
        parent=node,
        critical=True
    )
    affiliation_sources = _combine_urls(extracted.primary_affiliation_source_urls, extracted.paper_urls)
    affiliation_leaf = evaluator.add_leaf(
        id="primary_institution_correct",
        desc="Primary institutional affiliation corresponds to the first affiliation listed for the lead author in the paper.",
        parent=affil_group,
        critical=True
    )
    affiliation_claim = (
        f"The lead author's primary (first-listed) institutional affiliation is '{extracted.primary_affiliation_name or ''}'."
    )
    await evaluator.verify(
        claim=affiliation_claim,
        node=affiliation_leaf,
        sources=affiliation_sources,
        additional_instruction="Verify the first-listed affiliation for the lead author in the affiliations section of the paper."
    )
    evaluator.add_custom_node(
        result=bool(extracted.primary_affiliation_source_urls),
        id="primary_institution_source_url",
        desc="A URL is provided that directly supports the primary affiliation claim (e.g., paper page/PDF affiliations).",
        parent=affil_group,
        critical=True
    )

    # Institution country group
    country_group = evaluator.add_parallel(
        id="institution_country",
        desc="Provide the country where the lead author's primary institution is located and a URL that confirms it.",
        parent=node,
        critical=True
    )
    country_sources = _combine_urls(extracted.institution_country_source_urls, extracted.paper_urls)
    country_leaf = evaluator.add_leaf(
        id="country_correct",
        desc="Country stated matches the location of the lead author's primary institution.",
        parent=country_group,
        critical=True
    )
    country_claim = f"The country of the lead author's primary institution is '{extracted.institution_country or ''}'."
    await evaluator.verify(
        claim=country_claim,
        node=country_leaf,
        sources=country_sources,
        additional_instruction="If the country is not explicit in the paper, verify via an authoritative institutional page."
    )
    evaluator.add_custom_node(
        result=bool(extracted.institution_country_source_urls),
        id="country_source_url",
        desc="A URL is provided that supports the country claim (paper affiliation block or authoritative institutional source).",
        parent=country_group,
        critical=True
    )

    # Spectroscopy facility for July 15, 2025
    facility_group = evaluator.add_parallel(
        id="spectroscopy_facility_july_15_2025",
        desc="Provide the telescope facility used for spectroscopic observations on July 15, 2025 (must be SALT per constraints) and a confirming URL.",
        parent=node,
        critical=True
    )
    facility_sources = _combine_urls(extracted.spectroscopy_facility_source_urls, extracted.paper_urls)
    facility_leaf = evaluator.add_leaf(
        id="facility_and_date_correct",
        desc="Paper is cited to show that the spectroscopic observation on July 15, 2025 was conducted with the Southern African Large Telescope (SALT).",
        parent=facility_group,
        critical=True
    )
    facility_claim = (
        "The paper states that the spectroscopic observation on July 15, 2025 was conducted with the Southern African Large Telescope (SALT)."
    )
    await evaluator.verify(
        claim=facility_claim,
        node=facility_leaf,
        sources=facility_sources,
        additional_instruction="Check the observation log or methods section indicating SALT was used on 15 July 2025."
    )
    evaluator.add_custom_node(
        result=bool(extracted.spectroscopy_facility_source_urls),
        id="facility_date_source_url",
        desc="A URL is provided that directly supports the facility-and-date claim (paper page/PDF observation log/text).",
        parent=facility_group,
        critical=True
    )

    # Rotation period with uncertainty
    rotation_group = evaluator.add_parallel(
        id="rotation_period_with_uncertainty",
        desc="Provide the rotation period measurement of 3I/ATLAS including the reported uncertainty (±) and a confirming URL.",
        parent=node,
        critical=True
    )
    rotation_sources = _combine_urls(extracted.rotation_period_source_urls, extracted.paper_urls)
    rotation_leaf = evaluator.add_leaf(
        id="rotation_period_includes_uncertainty",
        desc="Rotation period is reported with its uncertainty exactly as in the paper (value plus ± uncertainty).",
        parent=rotation_group,
        critical=True
    )
    rotation_claim = (
        f"The rotation period of 3I/ATLAS is reported in the paper as '{extracted.rotation_period_with_uncertainty or ''}'."
    )
    await evaluator.verify(
        claim=rotation_claim,
        node=rotation_leaf,
        sources=rotation_sources,
        additional_instruction="Ensure the value includes both the period and ± uncertainty exactly as shown in the paper; allow minor formatting variants."
    )
    evaluator.add_custom_node(
        result=bool(extracted.rotation_period_source_urls),
        id="rotation_period_source_url",
        desc="A URL is provided that directly supports the rotation period-with-uncertainty claim (paper page/PDF).",
        parent=rotation_group,
        critical=True
    )

    return node


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate the answer for the A&A 3I/ATLAS 2025 paper identification and information extraction task.
    """
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_paper_info(),
        template_class=AandaAtlasPaperExtraction,
        extraction_name="aanda_atlas_paper_info"
    )

    # Build the top-level critical sequential node to mirror the rubric's root critical node
    research_task_node = evaluator.add_sequential(
        id="research_task",
        desc="Identify the specified 2025 Astronomy & Astrophysics Letter about the temporal evolution of interstellar comet 3I/ATLAS and extract required fields with supporting URLs.",
        parent=root,
        critical=True
    )

    # 1) Identify the correct paper
    await build_identify_correct_paper_nodes(evaluator, research_task_node, extracted)

    # 2) Extract required information
    await build_extract_required_information_nodes(evaluator, research_task_node, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()