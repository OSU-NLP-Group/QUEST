import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nature_comm_2024_hominin_species_publication"
TASK_DESCRIPTION = """
In 2024, a research article published in Nature Communications proposed a new hominin (human ancestor) species based on fossil evidence from eastern Asia. The research was conducted collaboratively by a senior professor from the Institute of Vertebrate Paleontology and Paleoanthropology at the Chinese Academy of Sciences in Beijing, China, who served as lead author on the taxonomic assignment, and a professor from the Department of Anthropology at the University of Hawaiʻi at Mānoa. The proposed species lived during the Late Middle to early Late Pleistocene epoch (approximately 300,000 to 50,000 years ago), and the research aimed to clarify a confusing hominin fossil record from the region. Identify this publication and provide the following information: (1) The taxonomic name of the proposed new hominin species, (2) The name of the lead author from the Chinese Academy of Sciences and their specific institutional affiliation, (3) The name of the co-author from the University of Hawaiʻi at Mānoa and their department affiliation, (4) A reference URL to the publication or an official institutional announcement about it.
"""


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class PublicationExtraction(BaseModel):
    # Publication identification
    journal_name: Optional[str] = None
    publication_year: Optional[str] = None
    title: Optional[str] = None
    doi: Optional[str] = None
    citation: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)

    # Requested details
    species_name: Optional[str] = None
    lead_author_name: Optional[str] = None
    lead_author_affiliation: Optional[str] = None
    coauthor_name: Optional[str] = None
    coauthor_affiliation: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_publication() -> str:
    return """
Extract exactly one publication from the answer that best matches ALL of the following constraints:
- It is a Nature Communications article in 2024.
- It proposes a new hominin (human ancestor) species based on fossil evidence from eastern Asia.
- Authorship collaboration includes: a senior professor from the Institute of Vertebrate Paleontology and Paleoanthropology (IVPP), Chinese Academy of Sciences (CAS), Beijing, China (lead author on the taxonomic assignment), and a professor from the Department of Anthropology, University of Hawaiʻi at Mānoa.
- The study places the species in the Late Middle to early Late Pleistocene (~300,000–50,000 years ago) and aims to clarify a confusing regional hominin fossil record.

From the answer text ONLY, extract the following fields (use null for any missing field):
- journal_name: the journal name as written (expect "Nature Communications" or close variant).
- publication_year: the year for the Nature Communications publication (expect "2024").
- title: the publication title if present.
- doi: the DOI if present (e.g., "10.1038/..." – exact string as in the answer).
- citation: any full citation text for the article if present.
- reference_urls: a list of URLs the answer cites that point to the publication or official institutional announcements (e.g., nature.com, ivpp.cas.cn, cas.cn, hawaii.edu/manoa/), in any reasonable format.

Requested details (as reported in the answer):
- species_name: the taxonomic name for the proposed new hominin species (exactly as written in the answer).
- lead_author_name: the CAS/IVPP senior professor (lead author on the taxonomic assignment) name.
- lead_author_affiliation: the stated affiliation for that lead author (should be IVPP, Chinese Academy of Sciences, Beijing, China; keep as written).
- coauthor_name: the University of Hawaiʻi at Mānoa co-author's name.
- coauthor_affiliation: that co-author’s department affiliation (should be Department of Anthropology, University of Hawaiʻi at Mānoa; keep as written).

Rules:
- Extract strictly from the provided answer. Do not invent or infer missing values.
- For URLs, extract real URLs cited in the answer (plain or markdown). If a URL lacks protocol, prepend http:// as needed.
- If multiple plausible URLs are provided, include all of them in reference_urls.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Deduplicate and strip
    seen = set()
    out = []
    for u in urls:
        if not _non_empty(u):
            continue
        u2 = u.strip()
        if u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_identify_publication(
    evaluator: Evaluator,
    parent,
    info: PublicationExtraction,
) -> None:
    """
    Build 'Identify_Publication' critical-parallel node and its leaves.
    """
    node = evaluator.add_parallel(
        id="Identify_Publication",
        desc="Publication is identified and matches venue/year constraints, with an accessible reference link.",
        parent=parent,
        critical=True,
    )

    sources = _clean_urls(info.reference_urls)

    # Existence of a reference URL (gates other checks in this branch)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Reference_URL_Provided",
        desc="At least one reference URL is provided to the publication or an official institutional announcement.",
        parent=node,
        critical=True,
    )

    # Journal Name check
    journal_leaf = evaluator.add_leaf(
        id="Journal_Name",
        desc="The identified publication is in the journal Nature Communications.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The publication is in the journal Nature Communications.",
        node=journal_leaf,
        sources=sources,
        additional_instruction=(
            "Check the page(s) to confirm the venue is 'Nature Communications'. "
            "Accept reasonable variants (e.g., 'Nature Communications' spelled with minor punctuation differences). "
            "If the URL is an official announcement, it must explicitly mention 'Nature Communications' regarding the publication."
        ),
    )

    # Publication Year check
    year_leaf = evaluator.add_leaf(
        id="Publication_Year",
        desc="The identified publication year is 2024.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The publication year is 2024.",
        node=year_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the publication's year is 2024. If the page shows online publication/press release dates, "
            "ensure it refers to the Nature Communications article in 2024."
        ),
    )

    # Reference URL relevance/working check
    url_leaf = evaluator.add_leaf(
        id="Reference_URL",
        desc="Provides a working reference URL to the publication or an official institutional announcement about it.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This page is either the Nature Communications article itself about the proposed new hominin species in 2024, "
            "or an official institutional announcement (e.g., from CAS/IVPP or UH Mānoa) about that publication."
        ),
        node=url_leaf,
        sources=sources,
        additional_instruction=(
            "Mark as supported if at least one URL is a valid working page that clearly corresponds to the described Nature Communications publication "
            "or an official announcement about it. Institutional domains may include nature.com, cas.cn, ivpp.cas.cn, hawaii.edu, or manoa.hawaii.edu."
        ),
    )

    # Publication Identifier (adjusted to critical due to framework constraint)
    id_leaf = evaluator.add_leaf(
        id="Publication_Identifier",
        desc="Provides an additional identifier sufficient to uniquely indicate the publication (e.g., title and/or DOI and/or full citation).",
        parent=node,
        critical=True,  # Adjusted to satisfy 'critical parent must have critical children'
    )
    # Build a claim based on availability: prefer DOI, else title, else citation
    if _non_empty(info.doi):
        claim = f"The DOI of the publication is {info.doi}."
        add_ins = (
            "Confirm the DOI string is explicitly shown on the page(s) or within the official announcement referencing the publication."
        )
    elif _non_empty(info.title):
        claim = f"The title of the publication is '{info.title}'."
        add_ins = (
            "Verify the page(s) show a publication title matching or equivalent to the extracted title. Allow minor punctuation/case differences."
        )
    elif _non_empty(info.citation):
        # Use a truncated citation snippet to reduce overly long claims
        snip = info.citation.strip()
        if len(snip) > 180:
            snip = snip[:180] + "..."
        claim = f"The publication can be uniquely identified by the citation text: '{snip}'."
        add_ins = (
            "Check whether the citation text (or a clear subset) appears on the page(s) identifying the publication. "
            "Allow standard citation formatting differences."
        )
    else:
        # No identifier provided; make a claim that will be judged against sources (likely to fail)
        claim = "The page(s) provide a unique identifier (title or DOI or full citation) for the publication."
        add_ins = (
            "Fail if there is no explicit title, DOI, or citation text sufficient to uniquely indicate the publication."
        )

    await evaluator.verify(
        claim=claim,
        node=id_leaf,
        sources=sources,
        additional_instruction=add_ins,
    )


async def build_provide_requested_information(
    evaluator: Evaluator,
    parent,
    info: PublicationExtraction,
) -> None:
    """
    Build 'Provide_Requested_Information' critical-parallel node and its leaves/subnodes.
    """
    node = evaluator.add_parallel(
        id="Provide_Requested_Information",
        desc="Report the required species and author/affiliation information consistent with the identified publication.",
        parent=parent,
        critical=True,
    )
    sources = _clean_urls(info.reference_urls)

    # Proposed species name
    species_leaf = evaluator.add_leaf(
        id="Proposed_Species",
        desc="Reports the taxonomic name of the proposed new hominin species as stated in the identified publication.",
        parent=node,
        critical=True,
    )
    species_claim = (
        f"The taxonomic name of the proposed new hominin species is '{info.species_name}'."
        if _non_empty(info.species_name)
        else "The publication explicitly states the taxonomic name of the proposed new hominin species."
    )
    await evaluator.verify(
        claim=species_claim,
        node=species_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the specific species binomial (or formal taxonomic name) as presented in the Nature Communications article or official announcement. "
            "Allow italicization/formatting differences; match the wording."
        ),
    )

    # Lead Author from CAS IVPP
    lead_node = evaluator.add_parallel(
        id="Lead_Author_From_CAS_IVPP",
        desc="Reports the lead author from the Chinese Academy of Sciences and their specific institutional affiliation.",
        parent=node,
        critical=True,
    )

    lead_name_leaf = evaluator.add_leaf(
        id="Lead_Author_Name",
        desc="Provides the lead author's name (the CAS-affiliated senior professor who served as lead author on the taxonomic assignment per the question/constraints).",
        parent=lead_node,
        critical=True,
    )
    lead_name_claim = (
        f"The publication/announcements indicate that the lead author on the taxonomic assignment is {info.lead_author_name}."
        if _non_empty(info.lead_author_name)
        else "The publication/announcements identify the CAS/IVPP senior professor who led the taxonomic assignment."
    )
    await evaluator.verify(
        claim=lead_name_claim,
        node=lead_name_leaf,
        sources=sources,
        additional_instruction=(
            "Accept phrasing like 'led by', 'lead author', or equivalent leadership wording tied to the taxonomic assignment, "
            "and confirm the person's role relates directly to the publication."
        ),
    )

    lead_aff_leaf = evaluator.add_leaf(
        id="Lead_Author_Affiliation",
        desc="Lead author's affiliation is Institute of Vertebrate Paleontology and Paleoanthropology, Chinese Academy of Sciences, Beijing, China.",
        parent=lead_node,
        critical=True,
    )
    lead_aff_text = (
        info.lead_author_affiliation
        if _non_empty(info.lead_author_affiliation)
        else "Institute of Vertebrate Paleontology and Paleoanthropology, Chinese Academy of Sciences, Beijing, China"
    )
    await evaluator.verify(
        claim=(
            f"The lead author's affiliation is '{lead_aff_text}', i.e., the Institute of Vertebrate Paleontology and Paleoanthropology (IVPP), "
            "Chinese Academy of Sciences, Beijing, China."
        ),
        node=lead_aff_leaf,
        sources=sources,
        additional_instruction=(
            "Allow standard variants/abbreviations like 'IVPP, CAS' and minor punctuation differences. "
            "The affiliation should clearly indicate IVPP and CAS, located in Beijing, China."
        ),
    )

    # Co-author from UH Mānoa Anthropology
    co_node = evaluator.add_parallel(
        id="Coauthor_From_UH_Anthropology",
        desc="Reports the University of Hawaiʻi at Mānoa co-author and their department affiliation.",
        parent=node,
        critical=True,
    )

    co_name_leaf = evaluator.add_leaf(
        id="Coauthor_Name",
        desc="Provides the co-author's name (the UH Mānoa-affiliated professor specified by the question/constraints).",
        parent=co_node,
        critical=True,
    )
    co_name_claim = (
        f"The publication/announcements list {info.coauthor_name} as a co-author affiliated with the University of Hawaiʻi at Mānoa."
        if _non_empty(info.coauthor_name)
        else "The publication/announcements list a UH Mānoa-affiliated professor as a co-author."
    )
    await evaluator.verify(
        claim=co_name_claim,
        node=co_name_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the named person is a co-author and is affiliated with the University of Hawaiʻi at Mānoa. "
            "Allow minor spelling/diacritic variations (e.g., 'Hawaiʻi' vs 'Hawaii') and common abbreviations ('UH Mānoa')."
        ),
    )

    co_aff_leaf = evaluator.add_leaf(
        id="Coauthor_Department_Affiliation",
        desc="Co-author affiliation is Department of Anthropology, University of Hawaiʻi at Mānoa.",
        parent=co_node,
        critical=True,
    )
    co_aff_text = (
        info.coauthor_affiliation
        if _non_empty(info.coauthor_affiliation)
        else "Department of Anthropology, University of Hawaiʻi at Mānoa"
    )
    await evaluator.verify(
        claim=f"The co-author's department affiliation is '{co_aff_text}', i.e., Department of Anthropology, University of Hawaiʻi at Mānoa.",
        node=co_aff_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the co-author is affiliated with the Department of Anthropology, UH Mānoa. "
            "Allow minor wording variants such as 'Anthropology Department' or 'UH Mānoa Anthropology'."
        ),
    )


async def build_validate_research_constraints(
    evaluator: Evaluator,
    parent,
    info: PublicationExtraction,
) -> None:
    """
    Build 'Validate_Research_Constraints' critical-parallel node and its leaves.
    """
    node = evaluator.add_parallel(
        id="Validate_Research_Constraints",
        desc="Checks that the identified publication matches the remaining research-content constraints.",
        parent=parent,
        critical=True,
    )
    sources = _clean_urls(info.reference_urls)

    # New species proposal
    nsp_leaf = evaluator.add_leaf(
        id="New_Species_Proposal",
        desc="The publication explicitly describes proposing a new hominin (human ancestor) species.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The publication explicitly proposes a new hominin (human ancestor) species.",
        node=nsp_leaf,
        sources=sources,
        additional_instruction="Look for phrases like 'we propose a new species' or equivalent in the article or official announcements.",
    )

    # Eastern Asia fossil evidence focus
    east_asia_leaf = evaluator.add_leaf(
        id="Eastern_Asia_Fossil_Evidence_Focus",
        desc="The research focuses on fossil evidence from eastern Asia.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The research focuses on fossil evidence from eastern Asia.",
        node=east_asia_leaf,
        sources=sources,
        additional_instruction="The page(s) should indicate the fossils and the geographic focus is in eastern Asia (e.g., China or neighboring regions).",
    )

    # Time period constraint
    time_leaf = evaluator.add_leaf(
        id="Time_Period_Constraint",
        desc="The proposed species is placed in the Late Middle to early Late Pleistocene (~300,000 to 50,000 years ago).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The proposed species is placed in the Late Middle to early Late Pleistocene (approximately 300,000 to 50,000 years ago).",
        node=time_leaf,
        sources=sources,
        additional_instruction="Allow slight paraphrases that clearly place the species in this timeframe.",
    )

    # Clarify confusing fossil record
    clarify_leaf = evaluator.add_leaf(
        id="Clarify_Confusing_Fossil_Record",
        desc="The research aims to organize/clarify a previously confusing hominin fossil record from the region.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The research aims to organize or clarify a previously confusing hominin fossil record from the region.",
        node=clarify_leaf,
        sources=sources,
        additional_instruction="Look for statements about resolving confusion, organizing, or clarifying the regional hominin fossil record.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 2024 Nature Communications hominin species publication task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root aggregator; we will add a critical sequential top-level task node
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

    # Extract structured info from the provided answer text
    extracted: PublicationExtraction = await evaluator.extract(
        prompt=prompt_extract_publication(),
        template_class=PublicationExtraction,
        extraction_name="publication_extraction",
    )

    # Add top-level critical sequential node to mirror rubric root
    task_node = evaluator.add_sequential(
        id="Publication_Task",
        desc="Identify the 2024 Nature Communications publication and provide the requested species, author/affiliation, and reference details under the stated constraints.",
        parent=root,
        critical=True,
    )

    # 1) Identify publication (critical, parallel)
    await build_identify_publication(evaluator, task_node, extracted)

    # 2) Provide requested information (critical, parallel)
    await build_provide_requested_information(evaluator, task_node, extracted)

    # 3) Validate research constraints (critical, parallel)
    await build_validate_research_constraints(evaluator, task_node, extracted)

    # Provide a small note about a minor rubric adjustment for framework consistency
    evaluator.add_custom_info(
        info={
            "note": "Publication_Identifier was set to critical to satisfy 'critical parent must have critical children' constraint.",
            "extracted_reference_urls_count": len(_clean_urls(extracted.reference_urls)),
        },
        info_type="implementation_notes",
    )

    return evaluator.get_summary()