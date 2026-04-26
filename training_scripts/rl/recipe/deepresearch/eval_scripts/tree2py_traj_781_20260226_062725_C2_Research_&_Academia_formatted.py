import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wam_researcher_commbio_2025_cardabiodontidae_1999"
TASK_DESCRIPTION = """
A researcher from the Western Australian Museum co-authored a paper published in Communications Biology in 2025 that describes ancient shark fossils discovered in the Darwin Formation in northern Australia. This same researcher also first described the extinct shark family Cardabiodontidae in 1999. What is the name of this researcher?
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResearcherExtraction(BaseModel):
    """
    Extracted researcher identification and cited sources from the agent's answer.
    """
    researcher_name: Optional[str] = None
    commbio_2025_urls: List[str] = Field(default_factory=list)
    affiliation_urls: List[str] = Field(default_factory=list)
    cardabiodontidae_1999_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_researcher() -> str:
    return """
    Extract the single researcher’s full name and all cited source URLs from the answer that are relevant to each of the following checks.

    Return a JSON object with the fields:
    - researcher_name: The full name of the researcher identified in the answer.
    - commbio_2025_urls: An array of URLs cited in the answer that directly correspond to, or credibly document, the 2025 Communications Biology paper describing ancient shark fossils from the Darwin Formation in northern Australia (e.g., the journal article page, a publisher page, or an official press release/news item explicitly about that paper).
    - affiliation_urls: An array of URLs cited in the answer that explicitly show this researcher’s affiliation with the Western Australian Museum (e.g., WA Museum staff page, official bio, or credible third-party profile).
    - cardabiodontidae_1999_urls: An array of URLs cited in the answer that explicitly document that this researcher first described (established/named) the extinct shark family Cardabiodontidae in 1999 (e.g., taxonomy papers, museum pages, or reliable encyclopedia entries).

    Important rules:
    - Only extract information explicitly present in the answer. Do not invent or infer details that are not stated.
    - For each URL field, include only valid URLs that are actually present in the answer (plain URLs or inside markdown links). If a URL is missing a protocol, prepend http://.
    - If the answer mentions the source but does not provide an actual URL, do not add it; instead, leave that list empty.
    - If multiple researchers are mentioned, choose the one the answer connects to BOTH the 2025 Communications Biology paper and the 1999 Cardabiodontidae description. If the answer is ambiguous, select the most likely single name that the answer attributes to both criteria.
    - If any field is not provided in the answer, set it to null (for the name) or to an empty array (for URL lists).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_publication_criteria(
    evaluator: Evaluator,
    parent_node,
    extracted: ResearcherExtraction,
) -> None:
    """
    Build and verify the Publication Criteria subtree:
    - Co-authorship on the 2025 Communications Biology paper
    - Journal is Communications Biology
    - Publication year is 2025
    - Paper context: ancient shark fossils in the Darwin Formation, northern Australia
    - Affiliation with Western Australian Museum
    """
    pub_node = evaluator.add_parallel(
        id="publication_criteria",
        desc="Verify the researcher's involvement in the 2025 Communications Biology publication and current institutional affiliation",
        parent=parent_node,
        critical=True
    )

    # Existence checks for sources (critical, to gate subsequent verification)
    urls_2025_present = evaluator.add_custom_node(
        result=bool(extracted.commbio_2025_urls),
        id="commbio_2025_sources_present",
        desc="Sources for the 2025 Communications Biology paper are provided in the answer",
        parent=pub_node,
        critical=True
    )

    aff_urls_present = evaluator.add_custom_node(
        result=bool(extracted.affiliation_urls),
        id="affiliation_sources_present",
        desc="Sources for Western Australian Museum affiliation are provided in the answer",
        parent=pub_node,
        critical=True
    )

    # 1) Co-authorship on the paper
    coauthor_leaf = evaluator.add_leaf(
        id="coauthor_2025_paper",
        desc="The researcher is a co-author of the paper shown in the provided sources",
        parent=pub_node,
        critical=True
    )
    coauthor_claim = f"The person named '{extracted.researcher_name or ''}' is listed as an author/co-author on the paper shown in the provided sources."
    await evaluator.verify(
        claim=coauthor_claim,
        node=coauthor_leaf,
        sources=extracted.commbio_2025_urls,
        additional_instruction=(
            "Check the author list on the page(s). Accept synonyms such as 'author', 'co-author', or 'contributor'. "
            "Confirm the specific individual appears among the authors."
        ),
    )

    # 2) Journal is Communications Biology
    journal_leaf = evaluator.add_leaf(
        id="journal_is_communications_biology",
        desc="The paper shown in the sources was published by the journal Communications Biology",
        parent=pub_node,
        critical=True
    )
    journal_claim = "The paper shown in the provided sources is published in Communications Biology."
    await evaluator.verify(
        claim=journal_claim,
        node=journal_leaf,
        sources=extracted.commbio_2025_urls,
        additional_instruction=(
            "Verify that the journal name is explicitly 'Communications Biology' on at least one source page."
        ),
    )

    # 3) Year is 2025
    year_leaf = evaluator.add_leaf(
        id="published_in_2025",
        desc="The paper was published in 2025",
        parent=pub_node,
        critical=True
    )
    year_claim = "The paper shown in the provided sources was published in 2025."
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=extracted.commbio_2025_urls,
        additional_instruction=(
            "Confirm that the publication date/year shown on the page(s) is 2025. "
            "Minor date formatting differences are acceptable as long as the year is clearly 2025."
        ),
    )

    # 4) Darwin Formation context
    darwin_leaf = evaluator.add_leaf(
        id="darwin_formation_context",
        desc="The paper describes ancient shark fossils discovered in the Darwin Formation in northern Australia",
        parent=pub_node,
        critical=True
    )
    darwin_claim = (
        "The paper described in the provided sources concerns ancient shark fossils discovered in the Darwin Formation in northern Australia."
    )
    await evaluator.verify(
        claim=darwin_claim,
        node=darwin_leaf,
        sources=extracted.commbio_2025_urls,
        additional_instruction=(
            "Look for explicit mentions of 'Darwin Formation' and 'northern Australia' together with the fossil discovery context. "
            "Allow reasonable variants (e.g., 'Darwin Fm.', region wording) if clearly equivalent."
        ),
    )

    # 5) Affiliation with Western Australian Museum
    affiliation_leaf = evaluator.add_leaf(
        id="museum_affiliation",
        desc="The researcher is affiliated with the Western Australian Museum",
        parent=pub_node,
        critical=True
    )
    affiliation_claim = f"The person named '{extracted.researcher_name or ''}' is affiliated with the Western Australian Museum."
    await evaluator.verify(
        claim=affiliation_claim,
        node=affiliation_leaf,
        sources=extracted.affiliation_urls,
        additional_instruction=(
            "Verify the page(s) explicitly state the individual's affiliation with the Western Australian Museum "
            "(WA Museum) via staff page, official profile, or equivalent credible documentation."
        ),
    )


async def verify_historical_contribution(
    evaluator: Evaluator,
    parent_node,
    extracted: ResearcherExtraction,
) -> None:
    """
    Build and verify the Historical Contribution subtree (sequential):
    - Sources present
    - The researcher first described Cardabiodontidae
    - The year of first description is 1999
    """
    hist_node = evaluator.add_sequential(
        id="historical_contribution",
        desc="Verify the researcher's historical taxonomic contribution to shark paleontology",
        parent=parent_node,
        critical=True
    )

    # Source existence (first, to gate subsequent checks)
    hist_urls_present = evaluator.add_custom_node(
        result=bool(extracted.cardabiodontidae_1999_urls),
        id="cardabiodontidae_1999_sources_present",
        desc="Sources for the Cardabiodontidae 1999 description are provided in the answer",
        parent=hist_node,
        critical=True
    )

    # 1) First described by the named researcher
    first_desc_leaf = evaluator.add_leaf(
        id="cardabiodontidae_first_described_by_researcher",
        desc="Cardabiodontidae was first described by the named researcher",
        parent=hist_node,
        critical=True
    )
    first_desc_claim = f"The extinct shark family Cardabiodontidae was first described (established/named) by '{extracted.researcher_name or ''}'."
    await evaluator.verify(
        claim=first_desc_claim,
        node=first_desc_leaf,
        sources=extracted.cardabiodontidae_1999_urls,
        additional_instruction=(
            "Look for explicit attribution such as 'first described by', 'family established by', or 'named by' the specified researcher."
        ),
    )

    # 2) Year is 1999
    year_1999_leaf = evaluator.add_leaf(
        id="cardabiodontidae_first_description_year",
        desc="The first description of Cardabiodontidae occurred in 1999",
        parent=hist_node,
        critical=True
    )
    year_1999_claim = "The first description of Cardabiodontidae occurred in 1999."
    await evaluator.verify(
        claim=year_1999_claim,
        node=year_1999_leaf,
        sources=extracted.cardabiodontidae_1999_urls,
        additional_instruction=(
            "Confirm the year of the first description is 1999. Accept reasonable date formats as long as the year is clearly 1999."
        ),
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
    Evaluate the agent's answer for identifying the Western Australian Museum researcher
    tied to the 2025 Communications Biology paper and the 1999 Cardabiodontidae description.
    """
    # Initialize evaluator with parallel aggregation at root level
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

    # Extract structured information from the answer
    extracted: ResearcherExtraction = await evaluator.extract(
        prompt=prompt_extract_researcher(),
        template_class=ResearcherExtraction,
        extraction_name="researcher_extraction",
    )

    # Build top-level critical node (to reflect rubric root semantics)
    researcher_root = evaluator.add_parallel(
        id="researcher_identification",
        desc="Identify the researcher from the Western Australian Museum who meets all specified criteria related to the 2025 ancient shark fossil publication and historical taxonomic contribution",
        parent=root,
        critical=True
    )

    # Critical existence check for the researcher's name
    name_present = evaluator.add_custom_node(
        result=bool(extracted.researcher_name and extracted.researcher_name.strip()),
        id="researcher_name_present",
        desc="The answer provides a specific researcher name",
        parent=researcher_root,
        critical=True
    )

    # Verify Publication Criteria subtree
    await verify_publication_criteria(evaluator, researcher_root, extracted)

    # Verify Historical Contribution subtree
    await verify_historical_contribution(evaluator, researcher_root, extracted)

    # Return the structured evaluation summary
    return evaluator.get_summary()