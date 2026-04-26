import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "paleo_shark_2025"
TASK_DESCRIPTION = (
    "I recently heard about a 2025 paleontology study reporting the discovery of fossilized shark remains "
    "approximately 115 million years old from the Darwin area in northern Australia. The research was coordinated "
    "by scientists at the Swedish Museum of Natural History and reportedly represents an important finding for "
    "understanding the evolution of gigantic lamniform sharks. I need to locate this paper for my literature review. "
    "Please identify this research paper and provide the following information: (1) The journal name and DOI where "
    "the paper was published, (2) The name of the lead (first) author and their institutional affiliation, "
    "(3) The coordinating institution of the senior/corresponding author, and (4) A valid reference URL (DOI link or "
    "direct article link) to access the paper."
)

EXPECTED_JOURNAL = "Communications Biology"
EXPECTED_DOI = "10.1038/s42003-025-08930-y"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PaperExtraction(BaseModel):
    """
    Extracted paper metadata from the agent's answer.
    All fields should come directly from the answer text without inventing information.
    """
    title: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    lead_author_name: Optional[str] = None
    lead_author_affiliation: Optional[str] = None
    senior_author_institution: Optional[str] = None
    access_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_fields() -> str:
    return """
    You must extract the single research paper the answer identifies that matches the task constraints.
    Extract only what is explicitly present in the answer text; do not infer or invent.

    Return a JSON object with the following fields:
    - title: The paper's title, if provided.
    - journal: The journal name for the identified paper (e.g., "Communications Biology"), if provided.
    - doi: The DOI string as written (e.g., "10.1038/s42003-025-08930-y") or a DOI URL (e.g., "https://doi.org/10.1038/s42003-025-08930-y"), if provided.
    - lead_author_name: The lead (first) author's name, if provided.
    - lead_author_affiliation: The lead (first) author's institutional affiliation, if provided.
    - senior_author_institution: The coordinating institution of the senior/corresponding author, if provided.
    - access_url: A single URL that directly accesses the paper (prefer a DOI link or the direct journal article page URL). If none is provided, set to null.
    - additional_urls: An array of any other URLs mentioned in the answer that relate to this paper or its discovery (press releases, institutional pages, etc.). If none, return an empty array.

    URL extraction rules:
    - Extract URLs exactly as they appear. Valid formats include bare URLs or markdown links; return the actual URL.
    - Do not fabricate URLs.
    - If a URL lacks a protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_and_merge_sources(extracted: PaperExtraction) -> List[str]:
    """
    Build a deduplicated list of sources to verify claims:
    - access_url (if provided)
    - additional_urls (all)
    - doi (converted to a DOI URL if it's a DOI string; use as-is if already a URL)
    """
    urls: List[str] = []

    def _add(u: Optional[str]):
        if not u:
            return
        u = u.strip()
        if not u:
            return
        if u not in urls:
            urls.append(u)

    # Access URL
    _add(extracted.access_url)

    # Additional URLs
    for u in extracted.additional_urls:
        _add(u)

    # DOI as URL
    if extracted.doi:
        doi_val = extracted.doi.strip()
        if doi_val:
            if doi_val.lower().startswith("http"):
                _add(doi_val)
            else:
                _add(f"https://doi.org/{doi_val}")

    return urls


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_identify_correct_paper(
    evaluator: Evaluator,
    parent_node,
    extracted: PaperExtraction,
    sources: List[str],
):
    """
    Build and execute the 'Identify_Correct_Paper' verification subtree.
    All leaf nodes are critical.
    """
    identify_node = evaluator.add_parallel(
        id="Identify_Correct_Paper",
        desc="The paper identified by the answer satisfies all stated constraints.",
        parent=parent_node,
        critical=True,
    )

    # Prepare all leaf nodes
    leaf_nodes = []

    # 1) Peer_Reviewed_2025
    n_peer_2025 = evaluator.add_leaf(
        id="Peer_Reviewed_2025",
        desc="Paper is published in a peer-reviewed scientific journal in 2025.",
        parent=identify_node,
        critical=True,
    )
    claim_peer_2025 = (
        "This page presents a peer-reviewed scientific journal article that was published in 2025 "
        "(not a preprint, news item, or blog)."
    )
    leaf_nodes.append((
        claim_peer_2025,
        sources,
        n_peer_2025,
        "Confirm the page type is a peer‑reviewed journal article and the publication year is 2025."
    ))

    # 2) Journal_Constraint
    n_journal = evaluator.add_leaf(
        id="Journal_Constraint",
        desc="Paper is published in Communications Biology.",
        parent=identify_node,
        critical=True,
    )
    claim_journal = f"The journal name of this article is '{EXPECTED_JOURNAL}'."
    leaf_nodes.append((
        claim_journal,
        sources,
        n_journal,
        "Look for the journal branding or citation metadata that clearly states 'Communications Biology'."
    ))

    # 3) Publication_Month
    n_pub_month = evaluator.add_leaf(
        id="Publication_Month",
        desc="Paper is published in October or November 2025.",
        parent=identify_node,
        critical=True,
    )
    claim_month = "The article was published in October or November 2025."
    leaf_nodes.append((
        claim_month,
        sources,
        n_pub_month,
        "Accept wording like 'Published: 2025-10-..' or '2025-11-..'. If multiple dates exist, use the publication/online date."
    ))

    # 4) Discovery_Location
    n_loc = evaluator.add_leaf(
        id="Discovery_Location",
        desc="Discovery location is in the Darwin area of northern Australia.",
        parent=identify_node,
        critical=True,
    )
    claim_loc = (
        "The study concerns fossil material discovered in the Darwin area of northern Australia (Northern Territory)."
    )
    leaf_nodes.append((
        claim_loc,
        sources,
        n_loc,
        "Accept mentions of Darwin, Darwin region/harbour area, or Northern Territory near Darwin."
    ))

    # 5) Fossil_Age
    n_age = evaluator.add_leaf(
        id="Fossil_Age",
        desc="Fossils are dated to approximately 115 million years old (upper Aptian period).",
        parent=identify_node,
        critical=True,
    )
    claim_age = (
        "The fossils are approximately 115 million years old (upper Aptian of the Early Cretaceous)."
    )
    leaf_nodes.append((
        claim_age,
        sources,
        n_age,
        "Allow reasonable phrasing like ~115 Ma or upper Aptian; small rounding differences are acceptable."
    ))

    # 6) Fossils_In_Australia
    n_in_aus = evaluator.add_leaf(
        id="Fossils_In_Australia",
        desc="Paper focuses on fossilized shark remains discovered in Australia.",
        parent=identify_node,
        critical=True,
    )
    claim_in_aus = (
        "This paper focuses on fossilized shark remains discovered in Australia."
    )
    leaf_nodes.append((
        claim_in_aus,
        sources,
        n_in_aus,
        "Look for mentions that the fossil remains were found in Australia."
    ))

    # 7) Topic_Evolution
    n_topic = evaluator.add_leaf(
        id="Topic_Evolution",
        desc="Paper reports findings about cardabiodontid and/or lamniform shark evolution (including implications for gigantic lamniform shark evolution).",
        parent=identify_node,
        critical=True,
    )
    claim_topic = (
        "The article discusses cardabiodontid and/or lamniform shark evolution, including implications for gigantic lamniform sharks."
    )
    leaf_nodes.append((
        claim_topic,
        sources,
        n_topic,
        "Accept 'Cardabiodontidae' as equivalent to cardabiodontid. Look for 'lamniform' and evolutionary implications for giant/large lamniforms."
    ))

    # 8) Coordinated_By_Swedish_Museum
    n_coord = evaluator.add_leaf(
        id="Coordinated_By_Swedish_Museum",
        desc="Study is coordinated by the Swedish Museum of Natural History.",
        parent=identify_node,
        critical=True,
    )
    claim_coord = (
        "The study was coordinated by scientists at the Swedish Museum of Natural History (Naturhistoriska riksmuseet)."
    )
    leaf_nodes.append((
        claim_coord,
        sources,
        n_coord,
        "Support may appear in the article, acknowledgements, author information, or official institutional news/press release."
    ))

    # 9) Senior_Author_Affiliation
    n_senior_aff = evaluator.add_leaf(
        id="Senior_Author_Affiliation",
        desc="Senior/corresponding author is affiliated with the Swedish Museum of Natural History.",
        parent=identify_node,
        critical=True,
    )
    claim_senior_aff = (
        "The senior/corresponding author of the paper is affiliated with the Swedish Museum of Natural History."
    )
    leaf_nodes.append((
        claim_senior_aff,
        sources,
        n_senior_aff,
        "Check corresponding author details or author affiliations; accept 'Naturhistoriska riksmuseet' as equivalent."
    ))

    # 10) Lead_Author_Affiliation
    n_lead_aff = evaluator.add_leaf(
        id="Lead_Author_Affiliation",
        desc="Lead (first) author is affiliated with Stanford University, Department of Earth and Planetary Sciences.",
        parent=identify_node,
        critical=True,
    )
    claim_lead_aff = (
        "The first author is affiliated with Stanford University, Department of Earth and Planetary Sciences."
    )
    leaf_nodes.append((
        claim_lead_aff,
        sources,
        n_lead_aff,
        "Allow 'Stanford Doerr School of Sustainability, Department of Earth & Planetary Sciences' or equivalent phrasing."
    ))

    # 11) DOI_Constraint
    n_doi_const = evaluator.add_leaf(
        id="DOI_Constraint",
        desc="Paper DOI is 10.1038/s42003-025-08930-y.",
        parent=identify_node,
        critical=True,
    )
    claim_doi_const = f"The DOI of the article is {EXPECTED_DOI}."
    # Prefer verifying via DOI / article page; ensure we include the DOI URL if present
    leaf_nodes.append((
        claim_doi_const,
        sources,
        n_doi_const,
        "Verify that the displayed DOI exactly matches 10.1038/s42003-025-08930-y."
    ))

    # Execute batch verification for this parallel subtree
    await evaluator.batch_verify(leaf_nodes)


async def build_provide_requested_information(
    evaluator: Evaluator,
    parent_node,
    extracted: PaperExtraction,
    sources: List[str],
):
    """
    Build and execute the 'Provide_Requested_Information' verification subtree.
    All leaf nodes are critical.
    """
    provide_node = evaluator.add_parallel(
        id="Provide_Requested_Information",
        desc="Answer provides the requested fields and the values are correct for the identified paper.",
        parent=parent_node,
        critical=True,
    )

    leaf_nodes = []

    # Journal_Name_Provided
    n_journal_prov = evaluator.add_leaf(
        id="Journal_Name_Provided",
        desc="Provides the correct journal name for the identified paper.",
        parent=provide_node,
        critical=True,
    )
    journal_val = (extracted.journal or "").strip()
    claim_journal_prov = f"The journal name of this paper is '{journal_val}'." if journal_val else "The answer provides the correct journal name for this paper."
    leaf_nodes.append((
        claim_journal_prov,
        sources,
        n_journal_prov,
        "Confirm that the stated journal name matches the journal shown on the article/DOI page. Treat comparisons case-insensitively."
    ))

    # DOI_Provided
    n_doi_prov = evaluator.add_leaf(
        id="DOI_Provided",
        desc="Provides the correct DOI for the identified paper.",
        parent=provide_node,
        critical=True,
    )
    doi_val = (extracted.doi or "").strip()
    claim_doi_prov = f"The DOI of the paper is '{doi_val}'." if doi_val else "The answer provides the correct DOI for this paper."
    leaf_nodes.append((
        claim_doi_prov,
        sources,
        n_doi_prov,
        "Compare the provided DOI to the DOI shown on the article page; accept equality ignoring trivial URL formatting differences."
    ))

    # Lead_Author_Name_Provided
    n_lead_name = evaluator.add_leaf(
        id="Lead_Author_Name_Provided",
        desc="Provides the correct lead (first) author name for the identified paper.",
        parent=provide_node,
        critical=True,
    )
    lead_name_val = (extracted.lead_author_name or "").strip()
    claim_lead_name = f"The first (lead) author of the paper is '{lead_name_val}'." if lead_name_val else "The provided first author name matches the first author listed on the paper."
    leaf_nodes.append((
        claim_lead_name,
        sources,
        n_lead_name,
        "Check author list; allow minor spelling/casing variations or inclusion/exclusion of middle initials."
    ))

    # Lead_Author_Affiliation_Provided
    n_lead_aff_prov = evaluator.add_leaf(
        id="Lead_Author_Affiliation_Provided",
        desc="Provides the correct lead (first) author institutional affiliation for the identified paper.",
        parent=provide_node,
        critical=True,
    )
    lead_aff_val = (extracted.lead_author_affiliation or "").strip()
    claim_lead_aff_prov = (
        f"The first author's institutional affiliation is '{lead_aff_val}'."
        if lead_aff_val else "The provided first author affiliation matches the affiliation listed on the paper."
    )
    leaf_nodes.append((
        claim_lead_aff_prov,
        sources,
        n_lead_aff_prov,
        "Confirm the affiliation from the author information; allow minor formatting differences."
    ))

    # Senior_Coordinating_Institution_Provided
    n_senior_inst = evaluator.add_leaf(
        id="Senior_Coordinating_Institution_Provided",
        desc="Provides the correct coordinating institution of the senior/corresponding author for the identified paper.",
        parent=provide_node,
        critical=True,
    )
    senior_inst_val = (extracted.senior_author_institution or "").strip()
    claim_senior_inst = (
        f"The senior/corresponding author's coordinating institution is '{senior_inst_val}'."
        if senior_inst_val else "The provided coordinating institution for the senior/corresponding author matches what's listed for the paper."
    )
    leaf_nodes.append((
        claim_senior_inst,
        sources,
        n_senior_inst,
        "Check the corresponding/senior author information or affiliation list."
    ))

    # Access_URL_Provided
    n_access_url = evaluator.add_leaf(
        id="Access_URL_Provided",
        desc="Provides a valid reference URL (DOI link or direct article link) that resolves to the identified paper.",
        parent=provide_node,
        critical=True,
    )
    # For this check, verify using the provided access_url only (if any)
    # so that we actually validate the supplied link.
    access_source = extracted.access_url if (extracted.access_url and extracted.access_url.strip()) else None
    claim_access_url = (
        "This URL is either a DOI landing page or the direct journal article page for the identified paper."
    )
    leaf_nodes.append((
        claim_access_url,
        access_source,
        n_access_url,
        "Verify that the URL opens the article or DOI landing page for the same paper; press releases or unrelated pages do not count."
    ))

    # Execute batch verification for this parallel subtree
    await evaluator.batch_verify(leaf_nodes)


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
) -> Dict:
    """
    Evaluate an answer for the 2025 paleontology shark paper identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root internal wrapper; actual flow controlled by children
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

    # Add top-level sequential node as per rubric
    research_task_node = evaluator.add_sequential(
        id="Research_Task",
        desc="Identify the specific 2025 peer-reviewed paper matching the given discovery/study constraints and provide the requested bibliographic/author/access details for that same paper.",
        parent=root,
        critical=True,
    )

    # Extraction step
    extracted = await evaluator.extract(
        prompt=prompt_extract_paper_fields(),
        template_class=PaperExtraction,
        extraction_name="paper_extraction",
    )

    # Build sources list for verification (using URLs explicitly present in the answer)
    sources = _normalize_and_merge_sources(extracted)

    # Record GT/reference info (for transparency; not used for scoring)
    evaluator.add_ground_truth({
        "expected_journal": EXPECTED_JOURNAL,
        "expected_doi": EXPECTED_DOI,
        "expected_publication_window": "October or November 2025",
        "expected_location": "Darwin area, Northern Territory, Australia",
        "expected_age": "~115 Ma (upper Aptian)",
        "expected_topics": "cardabiodontid / lamniform evolution; gigantic lamniform sharks",
        "expected_institutions": {
            "coordination": "Swedish Museum of Natural History",
            "senior_corresponding": "Swedish Museum of Natural History",
            "lead_affiliation": "Stanford University, Department of Earth and Planetary Sciences"
        }
    })

    evaluator.add_custom_info(
        info={
            "assembled_sources": sources,
            "access_url": extracted.access_url,
            "doi_raw": extracted.doi,
        },
        info_type="debug",
        info_name="verification_sources"
    )

    # Build Identify_Correct_Paper (critical, parallel)
    await build_identify_correct_paper(
        evaluator=evaluator,
        parent_node=research_task_node,
        extracted=extracted,
        sources=sources,
    )

    # Build Provide_Requested_Information (critical, parallel)
    await build_provide_requested_information(
        evaluator=evaluator,
        parent_node=research_task_node,
        extracted=extracted,
        sources=sources,
    )

    # Return evaluation summary
    return evaluator.get_summary()