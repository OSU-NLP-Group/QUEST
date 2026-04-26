import asyncio
import logging
from typing import Any, List, Optional, Dict, Set

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "giant_lamniform_shark_2025_cardabiodontid"
TASK_DESCRIPTION = """
Identify a peer-reviewed research paper published between July and November 2025 that reports on the discovery of a giant lamniform shark fossil from the mid-Cretaceous period (approximately 115 million years ago) in the Darwin Formation of northern Australia. The paper must meet ALL of the following criteria:

1. Published in a peer-reviewed academic journal (not a preprint or arXiv submission)
2. The shark fossil must be identified as belonging to the cardabiodontid group
3. The estimated length of the shark must be at least 6 meters
4. The lead (first) author must be affiliated with a university in the United States
5. The corresponding author must be from the same institution as the lead author
6. The research must involve an international collaboration with authors from at least 3 different continents
7. At least one co-author must be affiliated with an Australian research institution
8. At least one co-author must be affiliated with the Western Australian Museum
9. The research methodology must include statistical analysis using comparative data from modern sharks
10. The research methodology must include micro-CT scanning techniques

Provide the following information:
- Full paper title
- All authors listed in order
- Journal name
- Publication date (month and year)
- DOI or permanent URL to the published paper
- Lead author's institutional affiliation
- Corresponding author name
- Complete list of all countries represented in the author institutional affiliations
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PaperExtraction(BaseModel):
    # Required response fields
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    journal: Optional[str] = None
    publication_month: Optional[str] = None  # e.g., "August" or "Aug"
    publication_year: Optional[str] = None   # e.g., "2025"
    publication_date_text: Optional[str] = None  # e.g., "August 2025" or "2025-08-13"
    doi: Optional[str] = None  # e.g., "10.1038/s41586-..." (raw DOI string without protocol)
    permanent_url: Optional[str] = None  # canonical or stable URL to the article (doi.org or publisher)
    lead_author_affiliation: Optional[str] = None
    corresponding_author_name: Optional[str] = None
    countries: List[str] = Field(default_factory=list)

    # General sources mentioned in the answer (article page, DOI link, journal, supplementary, etc.)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_paper() -> str:
    return """
Extract the bibliographic and affiliation information for the single paper the answer proposes as meeting the constraints.

Return a JSON object with these fields:
- title: Full paper title (string)
- authors: Array of author full names in the exact order listed in the answer (array of strings). If the answer lists authors, include them in that exact order.
- journal: Journal name (string)
- publication_month: The publication month (string as it appears or normalized to month name, e.g., "July", "Aug", or "07")
- publication_year: The publication year (string, e.g., "2025")
- publication_date_text: The publication date text as written in the answer (e.g., "August 2025" or "2025-08-13")
- doi: The DOI string if present in the answer (e.g., "10.1038/s41586-...."). If the answer only provides a DOI URL (e.g., https://doi.org/10.1234/xyz), extract only the DOI portion "10.1234/xyz". If no DOI is present, set to null.
- permanent_url: A permanent or canonical URL to the published paper as explicitly given in the answer (e.g., a DOI URL like "https://doi.org/xx" or the publisher's article page). Do not invent a URL.
- lead_author_affiliation: The lead (first) author's institutional affiliation as given in the answer (string). If not present, set to null.
- corresponding_author_name: The corresponding author name as given in the answer (string). If not present, set to null.
- countries: An array listing all countries that the answer claims are represented in the author affiliations (array of strings). If not explicitly listed, return an empty array.
- source_urls: Array of all URLs explicitly mentioned in the answer that are relevant to this paper (e.g., DOI links, publisher links, journal links, supplementary material). Include every URL you find in the answer.

General rules:
- Extract only what appears in the answer text; do not infer or add missing information.
- Keep strings exactly as in the answer as much as possible.
- If a field is not present in the answer, set it to null (or empty array for list fields).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def build_url_set(extracted: PaperExtraction) -> List[str]:
    """
    Aggregate all available URLs from the extraction for verification.
    - Include permanent_url if present
    - Include https://doi.org/{doi} if doi present
    - Include any source_urls
    De-duplicate and return as list.
    """
    urls: Set[str] = set()
    if extracted.permanent_url and extracted.permanent_url.strip():
        urls.add(extracted.permanent_url.strip())
    if extracted.doi and extracted.doi.strip():
        doi_str = extracted.doi.strip()
        if doi_str.lower().startswith("http://") or doi_str.lower().startswith("https://"):
            urls.add(doi_str)
        else:
            urls.add(f"https://doi.org/{doi_str}")
    for u in extracted.source_urls:
        if u and isinstance(u, str) and u.strip():
            urls.add(u.strip())
    return list(urls)


def first_author_name(extracted: PaperExtraction) -> str:
    return extracted.authors[0] if extracted.authors else ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def add_eligibility_checks(evaluator: Evaluator, parent, extracted: PaperExtraction) -> None:
    """
    Build all critical eligibility checks and verify them (parallel).
    Each check is a distinct leaf node with a binary outcome.
    """
    eligibility_node = evaluator.add_parallel(
        id="eligibility_criteria",
        desc="Paper satisfies all mandatory constraints from the question/constraints",
        parent=parent,
        critical=True,
    )

    urls = build_url_set(extracted)
    fauth = first_author_name(extracted)
    lead_affil = extracted.lead_author_affiliation or ""
    corr = extracted.corresponding_author_name or ""

    claims_and_instructions: List[Dict[str, Any]] = []

    # 1. Peer-reviewed journal (not preprint)
    node_peer_reviewed = evaluator.add_leaf(
        id="publication_peer_reviewed",
        desc="Paper is published in a peer-reviewed academic journal (not a preprint/arXiv)",
        parent=eligibility_node,
        critical=True,
    )
    claims_and_instructions.append(dict(
        claim="This article is a peer-reviewed journal publication (not a preprint or arXiv submission).",
        node=node_peer_reviewed,
        sources=urls,
        add_ins="Look for cues such as 'Received/Accepted' dates, 'Article' type in a known scholarly journal, or clear indication of journal publication on the publisher page. Preprint servers like arXiv, bioRxiv, Research Square are not acceptable."
    ))

    # 2. Publication date window: July–November 2025 (inclusive)
    node_date_window = evaluator.add_leaf(
        id="publication_date_window",
        desc="Publication date is between July and November 2025 (inclusive)",
        parent=eligibility_node,
        critical=True,
    )
    claims_and_instructions.append(dict(
        claim="The publication date of this article falls between July 2025 and November 2025 inclusive (accept 'online first' dates within this window).",
        node=node_date_window,
        sources=urls,
        add_ins="Accept 'Published online' or 'First published' as the date if clearly shown. Months accepted: July, August, September, October, November 2025."
    ))

    # 3. Topic: giant lamniform shark fossil discovery
    node_topic = evaluator.add_leaf(
        id="topic_lamniform_giant_discovery",
        desc="Reports on discovery of a giant lamniform shark fossil",
        parent=eligibility_node,
        critical=True,
    )
    claims_and_instructions.append(dict(
        claim="This paper reports the discovery of a giant lamniform shark fossil.",
        node=node_topic,
        sources=urls,
        add_ins="Check title/abstract/main text for 'lamniform', 'giant shark', 'large-bodied', 'fossil discovery', etc."
    ))

    # 4. Site: Darwin Formation (northern Australia)
    node_site = evaluator.add_leaf(
        id="site_darwin_formation_northern_australia",
        desc="Fossil locality is the Darwin Formation in northern Australia",
        parent=eligibility_node,
        critical=True,
    )
    claims_and_instructions.append(dict(
        claim="The fossil locality is the Darwin Formation in northern Australia.",
        node=node_site,
        sources=urls,
        add_ins="Look for explicit mention of 'Darwin Formation' and its location in northern Australia (e.g., Northern Territory). Do not confuse with other 'Darwin' usages."
    ))

    # 5. Age: mid-Cretaceous ~115 Ma
    node_age = evaluator.add_leaf(
        id="age_mid_cretaceous_approx_115_ma",
        desc="Fossil age is mid-Cretaceous, approximately 115 million years ago",
        parent=eligibility_node,
        critical=True,
    )
    claims_and_instructions.append(dict(
        claim="The fossil is dated to the mid-Cretaceous, approximately 115 million years ago (allow approximate).",
        node=node_age,
        sources=urls,
        add_ins="Allow approximate phrasing like 'ca. 115 Ma', '∼115 Ma', or ranges that center around ~115 Ma in the mid-Cretaceous."
    ))

    # 6. Taxonomy: cardabiodontid
    node_tax = evaluator.add_leaf(
        id="taxonomy_cardabiodontid",
        desc="Fossil is identified as belonging to the cardabiodontid group",
        parent=eligibility_node,
        critical=True,
    )
    claims_and_instructions.append(dict(
        claim="The shark fossil is identified as belonging to the cardabiodontid group (Cardabiodontidae).",
        node=node_tax,
        sources=urls,
        add_ins="Look for 'Cardabiodontid', 'Cardabiodontidae', or closely related taxonomic terminology."
    ))

    # 7. Size: at least 6 meters
    node_size = evaluator.add_leaf(
        id="size_at_least_6m",
        desc="Estimated shark length is at least 6 meters",
        parent=eligibility_node,
        critical=True,
    )
    claims_and_instructions.append(dict(
        claim="The estimated total length of the shark is at least 6 meters.",
        node=node_size,
        sources=urls,
        add_ins="Allow rounding; '≥ 6 m', 'about 6 m', 'approx. 6–7 m' are acceptable."
    ))

    # 8. Lead author US university
    node_lead_us = evaluator.add_leaf(
        id="lead_author_us_university",
        desc="Lead (first) author is affiliated with a university in the United States",
        parent=eligibility_node,
        critical=True,
    )
    claims_and_instructions.append(dict(
        claim=f"The first (lead) author {fauth if fauth else '[first author]'} is affiliated with a university in the United States.",
        node=node_lead_us,
        sources=urls,
        add_ins="Check the first author's affiliation and confirm it is a university located in the USA (e.g., address includes 'USA' or 'United States')."
    ))

    # 9. Corresponding author same institution as lead author
    node_corr_same = evaluator.add_leaf(
        id="corresponding_same_institution_as_lead",
        desc="Corresponding author is from the same institution as the lead author",
        parent=eligibility_node,
        critical=True,
    )
    claims_and_instructions.append(dict(
        claim=f"The corresponding author {corr if corr else '[corresponding author]'} is affiliated with the same institution as the lead author (lead affiliation: '{lead_affil}').",
        node=node_corr_same,
        sources=urls,
        add_ins="Compare the named institutions for the corresponding and first authors. Allow minor textual variations in institution naming (e.g., department names)."
    ))

    # 10. International collaboration: 3+ continents
    node_3_continents = evaluator.add_leaf(
        id="international_collaboration_3_continents",
        desc="Authorship includes affiliations spanning at least 3 different continents",
        parent=eligibility_node,
        critical=True,
    )
    countries_hint = ", ".join(extracted.countries) if extracted.countries else "not provided"
    claims_and_instructions.append(dict(
        claim="The authors' affiliations collectively span at least three different continents.",
        node=node_3_continents,
        sources=urls,
        add_ins=f"Use the affiliations list on the article page to map countries to continents (Africa, Asia, Europe, North America, South America, Oceania). Countries mentioned in the answer: {countries_hint}."
    ))

    # 11. At least one Australian research institution co-author
    node_aus_inst = evaluator.add_leaf(
        id="has_australian_research_institution_coauthor",
        desc="At least one co-author is affiliated with an Australian research institution",
        parent=eligibility_node,
        critical=True,
    )
    claims_and_instructions.append(dict(
        claim="At least one co-author is affiliated with an Australian research institution.",
        node=node_aus_inst,
        sources=urls,
        add_ins="Look for affiliations in Australia (e.g., Australian universities, museums, CSIRO, etc.)."
    ))

    # 12. Western Australian Museum co-author
    node_wam = evaluator.add_leaf(
        id="has_western_australian_museum_coauthor",
        desc="At least one co-author is affiliated with the Western Australian Museum",
        parent=eligibility_node,
        critical=True,
    )
    claims_and_instructions.append(dict(
        claim="At least one co-author lists 'Western Australian Museum' in their affiliation.",
        node=node_wam,
        sources=urls,
        add_ins="Search the author affiliation list for 'Western Australian Museum'."
    ))

    # 13. Method: statistical analysis using comparative data from modern sharks
    node_stats_modern = evaluator.add_leaf(
        id="method_statistical_comparative_modern_sharks",
        desc="Methodology includes statistical analysis using comparative data from modern sharks",
        parent=eligibility_node,
        critical=True,
    )
    claims_and_instructions.append(dict(
        claim="The study includes statistical analysis using comparative data from modern (extant) sharks.",
        node=node_stats_modern,
        sources=urls,
        add_ins="Look in Methods/Results/Supplement for comparisons to living shark datasets and explicit statistical analyses (e.g., regressions, models)."
    ))

    # 14. Method: micro-CT scanning
    node_micro_ct = evaluator.add_leaf(
        id="method_micro_ct_scanning",
        desc="Methodology includes micro-CT scanning techniques",
        parent=eligibility_node,
        critical=True,
    )
    claims_and_instructions.append(dict(
        claim="The study uses micro-CT (μCT) scanning techniques.",
        node=node_micro_ct,
        sources=urls,
        add_ins="Search for 'micro-CT', 'μCT', or 'microcomputed tomography' in the methods/materials."
    ))

    # Execute all eligibility verifications in parallel
    await evaluator.batch_verify([
        (item["claim"], item["sources"], item["node"], item["add_ins"])
        for item in claims_and_instructions
    ])


async def add_required_response_field_checks(evaluator: Evaluator, parent, extracted: PaperExtraction) -> None:
    """
    Build all critical checks ensuring the response included all requested fields.
    These are existence checks only (binary custom nodes).
    """
    req_node = evaluator.add_parallel(
        id="required_response_fields",
        desc="Response includes all explicitly requested bibliographic and affiliation metadata fields",
        parent=parent,
        critical=True,
    )

    # Provides the full paper title
    evaluator.add_custom_node(
        result=bool(extracted.title and extracted.title.strip()),
        id="provide_full_paper_title",
        desc="Provides the full paper title",
        parent=req_node,
        critical=True
    )

    # Provides all authors listed in order (presence check)
    evaluator.add_custom_node(
        result=bool(extracted.authors and len(extracted.authors) > 0),
        id="provide_all_authors_in_order",
        desc="Provides all authors listed in order",
        parent=req_node,
        critical=True
    )

    # Provides the journal name
    evaluator.add_custom_node(
        result=bool(extracted.journal and extracted.journal.strip()),
        id="provide_journal_name",
        desc="Provides the journal name",
        parent=req_node,
        critical=True
    )

    # Provides the publication date (month and year)
    evaluator.add_custom_node(
        result=bool((extracted.publication_month and extracted.publication_month.strip())
                    and (extracted.publication_year and extracted.publication_year.strip())),
        id="provide_publication_date_month_year",
        desc="Provides the publication date as month and year",
        parent=req_node,
        critical=True
    )

    # Provides the DOI or a permanent URL
    has_doi_or_url = bool((extracted.doi and extracted.doi.strip()) or (extracted.permanent_url and extracted.permanent_url.strip()))
    evaluator.add_custom_node(
        result=has_doi_or_url,
        id="provide_doi_or_permanent_url",
        desc="Provides the DOI or a permanent URL to the published paper",
        parent=req_node,
        critical=True
    )

    # Provides lead author's institutional affiliation
    evaluator.add_custom_node(
        result=bool(extracted.lead_author_affiliation and extracted.lead_author_affiliation.strip()),
        id="provide_lead_author_institutional_affiliation",
        desc="Provides the lead author's institutional affiliation",
        parent=req_node,
        critical=True
    )

    # Provides the corresponding author name
    evaluator.add_custom_node(
        result=bool(extracted.corresponding_author_name and extracted.corresponding_author_name.strip()),
        id="provide_corresponding_author_name",
        desc="Provides the corresponding author name",
        parent=req_node,
        critical=True
    )

    # Provides all countries represented (presence: at least one)
    evaluator.add_custom_node(
        result=bool(extracted.countries and len(extracted.countries) > 0),
        id="provide_all_countries_represented",
        desc="Provides a complete list of all countries represented in the author institutional affiliations",
        parent=req_node,
        critical=True
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
    Evaluate a single answer for the giant lamniform shark paper selection task.
    """
    # Initialize evaluator
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_paper(),
        template_class=PaperExtraction,
        extraction_name="paper_extraction",
    )

    # Build verification tree
    await add_eligibility_checks(evaluator, root, extracted)
    await add_required_response_field_checks(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()