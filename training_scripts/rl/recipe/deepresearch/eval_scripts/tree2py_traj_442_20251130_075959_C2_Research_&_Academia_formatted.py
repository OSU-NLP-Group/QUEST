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
TASK_ID = "wa_shark_combio_2025"
TASK_DESCRIPTION = (
    "As of November 30, 2025, a major paleontological discovery about a giant ancient shark was published in the journal "
    "Communications Biology in October 2025. The paper, titled \"Early gigantic lamniform marks the onset of mega-body size "
    "in modern shark evolution,\" reports on 115-million-year-old shark vertebrae found near Darwin, northern Australia. "
    "Identify the co-author of this paper who is affiliated with at least one research institution located in Western Australia. "
    "Provide the researcher's full name and the name(s) of their Western Australian institution(s) as listed in the paper's "
    "author affiliations section. Include a reference URL to the Communications Biology paper."
)

EXPECTED_JOURNAL = "Communications Biology"
EXPECTED_TITLE = "Early gigantic lamniform marks the onset of mega-body size in modern shark evolution"
EXPECTED_PUBLICATION_DATE = "October 25, 2025"  # Allow minor formatting variations such as "25 October 2025"
EXPECTED_DISCOVERY_TOPIC = (
    "The paper reports on 115-million-year-old giant lamniform shark vertebrae found near Darwin, northern Australia."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WAResearcherExtraction(BaseModel):
    """Information extracted from the agent's answer."""
    reference_url: Optional[str] = None
    researcher_full_name: Optional[str] = None
    wa_institutions: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return (
        "Extract the following information exactly as presented in the answer:\n"
        "1) reference_url: A single working URL that points to the Communications Biology paper page in question. "
        "If multiple URLs are present, select the one that most directly points to the paper on the Communications Biology site.\n"
        "2) researcher_full_name: The full name of the identified co-author who the answer claims has Western Australian affiliation(s). "
        "This should be the exact full name used in the answer (including middle initials if provided).\n"
        "3) wa_institutions: An array of institution names located in Western Australia that the answer claims are listed in the paper's affiliations for this researcher. "
        "Use institution names as written in the answer. If none are provided, return an empty array.\n"
        "Return null for any field that is missing in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_nonempty_string(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())

def format_institution_list(institutions: List[str]) -> str:
    if not institutions:
        return "none"
    return "; ".join([inst.strip() for inst in institutions if inst and inst.strip()])

# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_paper_identification(
    evaluator: Evaluator,
    parent_node,
    extracted: WAResearcherExtraction
) -> None:
    """
    Build the 'Paper_Identification' subtree:
    - Check reference URL is provided and working
    - Verify journal, publication date, title, and topic on the paper page
    """
    paper_node = evaluator.add_sequential(
        id="Paper_Identification",
        desc="Identify and reference the specific Communications Biology paper matching the given constraints.",
        parent=parent_node,
        critical=True
    )

    # 1. Reference URL provided (existence gate)
    ref_url_exists = has_nonempty_string(extracted.reference_url)
    evaluator.add_custom_node(
        result=ref_url_exists,
        id="Reference_URL_Provided",
        desc="A reference URL is provided in the answer.",
        parent=paper_node,
        critical=True
    )

    # 2. Reference URL correctness and working
    ref_leaf = evaluator.add_leaf(
        id="Reference_URL",
        desc="A working reference URL to the Communications Biology paper is provided.",
        parent=paper_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL points to an accessible Communications Biology journal article page.",
        node=ref_leaf,
        sources=extracted.reference_url,
        additional_instruction=(
            "Confirm the page is a Communications Biology article page (Nature Portfolio). "
            "If the page is inaccessible, irrelevant, or not a Communications Biology article, mark as incorrect."
        )
    )

    # 3. Paper details (journal, publication date, title, discovery topic)
    details_node = evaluator.add_parallel(
        id="Paper_Details",
        desc="The paper metadata and topic match the constrained target paper.",
        parent=paper_node,
        critical=True
    )

    # 3.1 Journal
    journal_leaf = evaluator.add_leaf(
        id="Journal",
        desc="The journal is Communications Biology.",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim="The article is published in Communications Biology.",
        node=journal_leaf,
        sources=extracted.reference_url,
        additional_instruction=(
            "Check the journal branding or article metadata on the page. "
            "Allow minor wording variations (e.g., 'Nature Communications Biology')."
        )
    )

    # 3.2 Publication date
    pub_date_leaf = evaluator.add_leaf(
        id="Publication_Date",
        desc="The publication date is October 25, 2025.",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The article's publication date is {EXPECTED_PUBLICATION_DATE}.",
        node=pub_date_leaf,
        sources=extracted.reference_url,
        additional_instruction=(
            "Verify the publication date shown on the article page. "
            "Allow minor formatting variations (e.g., '25 October 2025')."
        )
    )

    # 3.3 Paper title
    title_leaf = evaluator.add_leaf(
        id="Paper_Title",
        desc=f'The paper title is "{EXPECTED_TITLE}".',
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The article title on the page is "{EXPECTED_TITLE}".',
        node=title_leaf,
        sources=extracted.reference_url,
        additional_instruction=(
            "Confirm the displayed title matches exactly or with only trivial formatting differences (quotes, capitalization)."
        )
    )

    # 3.4 Discovery topic
    topic_leaf = evaluator.add_leaf(
        id="Discovery_Topic",
        desc="The paper is about the giant lamniform shark vertebrae fossil discovery found near Darwin, northern Australia (as specified in the constraints).",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=EXPECTED_DISCOVERY_TOPIC,
        node=topic_leaf,
        sources=extracted.reference_url,
        additional_instruction=(
            "Look for mentions of 115-million-year-old shark vertebrae and the location near Darwin, Northern Territory, Australia."
        )
    )


async def build_wa_coauthor_identification(
    evaluator: Evaluator,
    parent_node,
    extracted: WAResearcherExtraction
) -> None:
    """
    Build the 'WA_Coauthor_Identification' subtree:
    - Researcher identity (name provided, coauthor status)
    - WA affiliations (names provided, match affiliations, located in WA)
    """
    wa_node = evaluator.add_parallel(
        id="WA_Coauthor_Identification",
        desc="Identify an author of the paper who has at least one affiliation located in Western Australia, and report the required author and affiliation details.",
        parent=parent_node,
        critical=True
    )

    # Subtree: Researcher_Identity
    identity_node = evaluator.add_parallel(
        id="Researcher_Identity",
        desc="The identified researcher is a co-author and is named fully.",
        parent=wa_node,
        critical=True
    )

    # 1. Full name provided
    full_name_ok = has_nonempty_string(extracted.researcher_full_name) and len(extracted.researcher_full_name.strip().split()) >= 2
    evaluator.add_custom_node(
        result=full_name_ok,
        id="Full_Name_Provided",
        desc="The researcher's full name is provided (sufficient to uniquely identify the co-author in the paper).",
        parent=identity_node,
        critical=True
    )

    # 2. Coauthor status on the identified paper
    coauthor_leaf = evaluator.add_leaf(
        id="Coauthor_Status",
        desc="The named researcher is listed as a co-author on the identified paper.",
        parent=identity_node,
        critical=True
    )
    await evaluator.verify(
        claim=f'{extracted.researcher_full_name or "Unknown"} is listed as an author of this article.',
        node=coauthor_leaf,
        sources=extracted.reference_url,
        additional_instruction=(
            "Check the author list on the article page. Allow minor name variations (middle initials, diacritics, casing). "
            "If the person is not listed, mark as incorrect."
        )
    )

    # Subtree: WA_Affiliations
    wa_aff_node = evaluator.add_parallel(
        id="WA_Affiliations",
        desc="The researcher has at least one institutional affiliation located in Western Australia, and the institution name(s) are taken from the paper’s affiliations section.",
        parent=wa_node,
        critical=True
    )

    # 3. WA institution names provided
    has_wa_names = bool(extracted.wa_institutions) and any(has_nonempty_string(x) for x in extracted.wa_institutions)
    evaluator.add_custom_node(
        result=has_wa_names,
        id="WA_Institution_Names_Provided",
        desc="At least one Western Australian institution name is provided for the researcher.",
        parent=wa_aff_node,
        critical=True
    )

    # 4. Institutions match those listed in affiliations
    inst_list_str = format_institution_list(extracted.wa_institutions)
    inst_match_leaf = evaluator.add_leaf(
        id="Institutions_As_Listed_In_Affiliations",
        desc="The provided institution name(s) match the institution name(s) as listed in the paper’s author affiliations section for that researcher.",
        parent=wa_aff_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f'The author affiliations shown on the article page for "{extracted.researcher_full_name or "Unknown"}" include: {inst_list_str}. '
            f'These names match what is listed on the page for that researcher.'
        ),
        node=inst_match_leaf,
        sources=extracted.reference_url,
        additional_instruction=(
            "Match the institution names under the affiliations section for the named researcher. "
            "Allow minor formatting variations (department/unit order, punctuation). "
            "If the listed affiliations do not include the provided names, mark as incorrect."
        )
    )

    # 5. At least one affiliation located in WA
    wa_loc_leaf = evaluator.add_leaf(
        id="Institutions_Located_In_WA",
        desc="At least one of the provided affiliations is located in Western Australia (WA), as indicated in the paper’s author affiliations section.",
        parent=wa_aff_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f'At least one affiliation listed on the article page for "{extracted.researcher_full_name or "Unknown"}" is located in Western Australia (WA).'
        ),
        node=wa_loc_leaf,
        sources=extracted.reference_url,
        additional_instruction=(
            "Check the affiliation location strings for markers like 'Western Australia', 'WA', 'Perth, WA', or city names in WA (e.g., Perth, Crawley, Nedlands). "
            "If none of the affiliations indicate Western Australia, mark as incorrect."
        )
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
    Evaluate an answer for the WA co-author identification on the Communications Biology paper.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root orchestration
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

    # Add top-level critical node to mirror rubric Root
    rubric_root = evaluator.add_sequential(
        id="Root",
        desc="Identify a co-author of the specified Communications Biology paper who has at least one Western Australian institutional affiliation, and provide their name, WA institution(s), and a reference URL.",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=WAResearcherExtraction,
        extraction_name="wa_researcher_extraction"
    )

    # Ground truth info about the target paper (for transparency)
    evaluator.add_ground_truth({
        "expected_journal": EXPECTED_JOURNAL,
        "expected_title": EXPECTED_TITLE,
        "expected_publication_date": EXPECTED_PUBLICATION_DATE,
        "expected_topic": EXPECTED_DISCOVERY_TOPIC
    }, gt_type="paper_expectations")

    # Build paper identification subtree
    await build_paper_identification(evaluator, rubric_root, extracted)

    # Build WA coauthor identification subtree
    await build_wa_coauthor_identification(evaluator, rubric_root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()