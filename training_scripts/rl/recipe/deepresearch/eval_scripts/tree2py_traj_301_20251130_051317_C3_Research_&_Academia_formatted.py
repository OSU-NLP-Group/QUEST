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
    "In November 2025, a research team published a paper in the journal Nature reporting the first-time detection of "
    "electrical discharges (lightning) on Mars, using audio recordings from NASA's Perseverance rover's SuperCam "
    "microphone. Identify this research paper and provide the following information: (1) The complete title of the research "
    "paper, (2) The exact publication date (month, day, and year), (3) The name of the lead (first) author, "
    "(4) The lead author's primary research institution (provide the full name), (5) The lead author's affiliation with the "
    "French national research organization CNRS (specify their role/status), (6) The lead author's university affiliation. "
    "Provide URL references to credible sources (e.g., Nature.com, institutional websites, or major science news outlets) "
    "to support your answer."
)

# Ground truth / expected values to support evaluation logic
EXPECTED_TITLE = "Detection of triboelectric discharges during dust events on Mars"
EXPECTED_JOURNAL = "Nature"
EXPECTED_PUBLICATION_DATE = "November 26, 2025"
EXPECTED_LEAD_AUTHOR = "Baptiste Chide"
EXPECTED_PRIMARY_INSTITUTION_FULL = "Institut de Recherche en Astrophysique et Planétologie (IRAP)"
EXPECTED_CNRS_ROLE = "CNRS researcher"
EXPECTED_UNIVERSITY = "Université de Toulouse"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    """Paper-level bibliographic and context info extracted from the answer."""
    title: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)
    # Optional textual cues that may appear in the answer; we keep them as strings for flexibility
    topic_first_time_electrical_discharges: Optional[str] = None
    uses_perseverance: Optional[str] = None
    instrument_supercam_microphone: Optional[str] = None


class AuthorInfo(BaseModel):
    """Lead author identity and affiliations extracted from the answer."""
    lead_author_name: Optional[str] = None
    primary_institution_full_name: Optional[str] = None
    cnrs_role_status: Optional[str] = None
    university_affiliation: Optional[str] = None
    affiliation_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_info() -> str:
    return """
    Extract the bibliographic and paper-level information for the Nature paper described in the answer.
    Return a JSON object with the following fields:
    - title: The complete title of the research paper.
    - journal: The journal name as provided in the answer (e.g., "Nature").
    - publication_date: The exact publication date (month day, year) as stated in the answer (e.g., "November 26, 2025").
    - supporting_urls: An array of all URLs provided in the answer that directly correspond to and/or credibly support the identified paper (e.g., Nature.com article page, DOI landing page, publisher page, NASA page, or major science news outlets). Extract actual URLs only.
    - topic_first_time_electrical_discharges: If the answer states the paper reports the first-time detection of electrical discharges (lightning) on Mars, extract the relevant phrase or statement from the answer; otherwise null.
    - uses_perseverance: If the answer states the paper uses data collected by NASA's Perseverance rover, extract the relevant phrase or statement; otherwise null.
    - instrument_supercam_microphone: If the answer states the paper specifically uses audio recordings from the SuperCam microphone, extract the relevant phrase or statement; otherwise null.

    SPECIAL RULES FOR URL SOURCES EXTRACTION:
    - Extract only URLs explicitly present in the answer text (including markdown links or plain URLs).
    - If a URL is missing a protocol (http:// or https://), prepend http://.
    - Include all credible URLs tied to the paper-level claims (Nature.com publisher page, DOI, institutional press releases, or reputable news coverage).

    If any field is not mentioned in the answer, set it to null (or empty list for supporting_urls).
    """


def prompt_extract_author_info() -> str:
    return """
    Extract the lead (first) author's identity and affiliations from the answer.
    Return a JSON object with the following fields:
    - lead_author_name: The full name of the lead (first) author as stated.
    - primary_institution_full_name: The lead author's primary research institution; provide the full official name (e.g., "Institut de Recherche en Astrophysique et Planétologie (IRAP)").
    - cnrs_role_status: The lead author's CNRS affiliation and role/status if given (e.g., "CNRS researcher", "Chargé de recherche CNRS").
    - university_affiliation: The lead author's university affiliation (e.g., "Université de Toulouse").
    - affiliation_urls: An array of all URLs provided in the answer that support the author affiliation/status claims (e.g., institutional pages, lab pages, Nature.com author affiliations, CNRS pages, or reputable news coverage). Extract actual URLs only.

    SPECIAL RULES FOR URL SOURCES EXTRACTION:
    - Extract only URLs explicitly present in the answer text (including markdown links or plain URLs).
    - If a URL is missing a protocol (http:// or https://), prepend http://.
    - Include credible URLs tied to the author-level claims and affiliations.

    If any field is not mentioned in the answer, set it to null (or empty list for affiliation_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_url_list(urls: List[str]) -> List[str]:
    """Normalize URLs: strip, ensure protocol, remove empties and duplicates."""
    seen = set()
    result: List[str] = []
    for u in urls or []:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        if not (s.startswith("http://") or s.startswith("https://")):
            s = "http://" + s
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def union_urls(*lists: List[str]) -> List[str]:
    """Union multiple URL lists with normalization."""
    combined: List[str] = []
    for lst in lists:
        combined.extend(lst or [])
    return normalize_url_list(combined)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_paper_details(
    evaluator: Evaluator,
    parent_node,
    paper: PaperInfo,
) -> None:
    """
    Build and verify the 'Paper_Identification_And_Bibliographic_Details' subtree.
    """
    # Create the paper details parent (critical, parallel aggregation)
    paper_node = evaluator.add_parallel(
        id="Paper_Identification_And_Bibliographic_Details",
        desc="Verify the identified paper matches paper-level constraints and provide requested bibliographic fields with supporting credible URL(s).",
        parent=parent_node,
        critical=True,
    )

    # Normalize URLs
    supp_urls = normalize_url_list(paper.supporting_urls)

    # Existence check for supporting URLs (custom leaf, critical)
    paper_urls_exist_node = evaluator.add_custom_node(
        result=(len(supp_urls) > 0),
        id="Paper_Supporting_URLs_Exist",
        desc="At least one paper-level supporting URL is provided in the answer.",
        parent=paper_node,
        critical=True,
    )

    # Leaf: Paper_Supporting_URLs_Provided (credibility & correspondence)
    urls_provided_leaf = evaluator.add_leaf(
        id="Paper_Supporting_URLs_Provided",
        desc="Provides at least one credible supporting URL that corresponds to the identified paper (e.g., publisher page, DOI landing page, or major science news coverage).",
        parent=paper_node,
        critical=True,
    )
    claim_urls = (
        f"At least one of the provided URLs is a credible source and corresponds to the Nature paper titled "
        f"'{EXPECTED_TITLE}' published in 2025."
    )
    await evaluator.verify(
        claim=claim_urls,
        node=urls_provided_leaf,
        sources=supp_urls,
        additional_instruction=(
            "Consider URLs credible if they are publisher pages (nature.com), DOI landing pages, institutional press releases "
            "(e.g., nasa.gov, cnrs.fr, irap.omp.eu), or reputable science news outlets. The page should clearly correspond to "
            "the specified paper (matching title and/or journal context)."
        ),
        extra_prerequisites=[paper_urls_exist_node],
    )

    # Leaf: Paper_Title
    paper_title_leaf = evaluator.add_leaf(
        id="Paper_Title",
        desc="Paper title matches: 'Detection of triboelectric discharges during dust events on Mars'.",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The paper's title is '{EXPECTED_TITLE}'.",
        node=paper_title_leaf,
        sources=supp_urls,
        additional_instruction="Allow minor punctuation or casing variations, but the title should clearly match the specified wording.",
        extra_prerequisites=[paper_urls_exist_node],
    )

    # Leaf: Journal_Is_Nature
    journal_leaf = evaluator.add_leaf(
        id="Journal_Is_Nature",
        desc="Paper is published in the peer-reviewed journal Nature.",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The paper is published in the journal Nature.",
        node=journal_leaf,
        sources=supp_urls,
        additional_instruction="Ensure the journal is 'Nature' specifically, not 'Nature Communications', 'Nature Astronomy', or other Nature-branded journals.",
        extra_prerequisites=[paper_urls_exist_node],
    )

    # Leaf: Publication_Date
    pub_date_leaf = evaluator.add_leaf(
        id="Publication_Date",
        desc="Publication date is November 26, 2025 (month/day/year).",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The publication date is {EXPECTED_PUBLICATION_DATE}.",
        node=pub_date_leaf,
        sources=supp_urls,
        additional_instruction="Accept formats like '26 November 2025' or 'November 26, 2025' as equivalent.",
        extra_prerequisites=[paper_urls_exist_node],
    )

    # Leaf: Topic_First_Time_Electrical_Discharges_On_Mars
    topic_leaf = evaluator.add_leaf(
        id="Topic_First_Time_Electrical_Discharges_On_Mars",
        desc="Paper reports first-time detection of electrical discharges (lightning) on Mars.",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The paper reports the first-time detection of electrical discharges (lightning) on Mars.",
        node=topic_leaf,
        sources=supp_urls,
        additional_instruction=(
            "Recognize 'triboelectric discharges during dust events on Mars' as electrical discharges. "
            "The page should clearly state that this is the first detection on Mars."
        ),
        extra_prerequisites=[paper_urls_exist_node],
    )

    # Leaf: Uses_Perseverance_Data
    perseverance_leaf = evaluator.add_leaf(
        id="Uses_Perseverance_Data",
        desc="Paper uses data collected by NASA's Perseverance rover.",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The paper uses data collected by NASA's Perseverance rover.",
        node=perseverance_leaf,
        sources=supp_urls,
        additional_instruction="Look for explicit mention of the Perseverance rover as the data source.",
        extra_prerequisites=[paper_urls_exist_node],
    )

    # Leaf: Instrument_SuperCam_Microphone
    supercam_leaf = evaluator.add_leaf(
        id="Instrument_SuperCam_Microphone",
        desc="Paper specifically uses audio recordings from the SuperCam microphone.",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The paper specifically uses audio recordings from the SuperCam microphone.",
        node=supercam_leaf,
        sources=supp_urls,
        additional_instruction="Accept phrasing such as 'SuperCam’s microphone', 'the SuperCam microphone', or equivalent wording.",
        extra_prerequisites=[paper_urls_exist_node],
    )


async def verify_author_affiliations(
    evaluator: Evaluator,
    parent_node,
    paper: PaperInfo,
    author: AuthorInfo,
) -> None:
    """
    Build and verify the 'Lead_Author_And_Affiliations' subtree.
    """
    # Create the author/affiliations parent (critical, parallel aggregation)
    author_node = evaluator.add_parallel(
        id="Lead_Author_And_Affiliations",
        desc="Provide and verify the lead (first) author identity and required affiliations/status with supporting credible URLs.",
        parent=parent_node,
        critical=True,
    )

    # Normalize URLs and build union for verification
    aff_urls = normalize_url_list(author.affiliation_urls)
    supp_urls = normalize_url_list(paper.supporting_urls)
    union_aff_support_urls = union_urls(aff_urls, supp_urls)

    # Existence check for affiliation URLs (custom leaf, critical)
    aff_urls_exist_node = evaluator.add_custom_node(
        result=(len(aff_urls) > 0),
        id="Affiliation_Supporting_URLs_Exist",
        desc="At least one author-affiliation supporting URL is provided in the answer.",
        parent=author_node,
        critical=True,
    )

    # Leaf: Affiliation_Supporting_URLs_Provided (credibility)
    aff_urls_provided_leaf = evaluator.add_leaf(
        id="Affiliation_Supporting_URLs_Provided",
        desc="Provides at least one credible supporting URL that supports the author affiliation/status claims (e.g., institutional page, lab page, or reputable reporting citing the affiliations).",
        parent=author_node,
        critical=True,
    )
    claim_aff_urls = (
        "At least one of the provided URLs is a credible page that supports the lead author's affiliations "
        "(IRAP, CNRS status, Université de Toulouse)."
    )
    await evaluator.verify(
        claim=claim_aff_urls,
        node=aff_urls_provided_leaf,
        sources=aff_urls,
        additional_instruction=(
            "Consider URLs credible if they are institutional pages (e.g., irap.omp.eu, cnrs.fr, univ-toulouse.fr), "
            "Nature.com author affiliation listings, or reputable coverage explicitly citing these affiliations."
        ),
        extra_prerequisites=[aff_urls_exist_node],
    )

    # Leaf: Lead_Author_Name
    lead_author_leaf = evaluator.add_leaf(
        id="Lead_Author_Name",
        desc="Lead (first) author is Baptiste Chide.",
        parent=author_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The lead (first) author of the paper is {EXPECTED_LEAD_AUTHOR}.",
        node=lead_author_leaf,
        sources=union_aff_support_urls,
        additional_instruction="Check the author list and ensure Baptiste Chide is listed first.",
    )

    # Leaf: Primary_Institution_Full_Name
    primary_inst_leaf = evaluator.add_leaf(
        id="Primary_Institution_Full_Name",
        desc="Lead author's primary research institution is IRAP (Institut de Recherche en Astrophysique et Planétologie) (full name provided).",
        parent=author_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The lead author's primary research institution is {EXPECTED_PRIMARY_INSTITUTION_FULL}.",
        node=primary_inst_leaf,
        sources=union_aff_support_urls,
        additional_instruction="Look for the full name 'Institut de Recherche en Astrophysique et Planétologie (IRAP)' in author affiliations or institutional profiles.",
    )

    # Leaf: CNRS_Role_Status
    cnrs_leaf = evaluator.add_leaf(
        id="CNRS_Role_Status",
        desc="Lead author has CNRS affiliation and their role/status is identified as a CNRS researcher.",
        parent=author_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The lead author has CNRS affiliation and their role/status is a {EXPECTED_CNRS_ROLE}.",
        node=cnrs_leaf,
        sources=union_aff_support_urls,
        additional_instruction=(
            "Accept equivalent French titles such as 'chargé de recherche CNRS' or 'chercheur CNRS' indicating CNRS researcher status."
        ),
    )

    # Leaf: University_Affiliation
    university_leaf = evaluator.add_leaf(
        id="University_Affiliation",
        desc="Lead author has a university affiliation with Université de Toulouse.",
        parent=author_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The lead author has a university affiliation with {EXPECTED_UNIVERSITY}.",
        node=university_leaf,
        sources=union_aff_support_urls,
        additional_instruction="Accept 'Université de Toulouse' or 'University of Toulouse' as equivalent phrasing.",
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Mars Lightning Nature 2025 paper identification task.
    """
    # Initialize evaluator (root node is non-critical by framework design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Overall evaluation follows the task's logical order
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

    # Create a critical top-level task node reflecting the rubric root
    task_main = evaluator.add_sequential(
        id="Mars_Lightning_Nature_Paper_Task",
        desc="Identify the specified Nature paper (Nov 2025) about first-time detection of electrical discharges on Mars using Perseverance SuperCam microphone audio, and provide required bibliographic and author-affiliation details with supporting credible URLs.",
        parent=root,
        critical=True,
    )

    # Extract paper info and author info concurrently
    paper_info_task = evaluator.extract(
        prompt=prompt_extract_paper_info(),
        template_class=PaperInfo,
        extraction_name="paper_info",
    )
    author_info_task = evaluator.extract(
        prompt=prompt_extract_author_info(),
        template_class=AuthorInfo,
        extraction_name="author_info",
    )
    paper_info, author_info = await asyncio.gather(paper_info_task, author_info_task)

    # Add ground truth information to the summary
    evaluator.add_ground_truth({
        "expected_title": EXPECTED_TITLE,
        "expected_journal": EXPECTED_JOURNAL,
        "expected_publication_date": EXPECTED_PUBLICATION_DATE,
        "expected_lead_author": EXPECTED_LEAD_AUTHOR,
        "expected_primary_institution_full_name": EXPECTED_PRIMARY_INSTITUTION_FULL,
        "expected_cnrs_role_status": EXPECTED_CNRS_ROLE,
        "expected_university_affiliation": EXPECTED_UNIVERSITY,
    }, gt_type="expected_values")

    # Build verification subtrees under the critical task node
    await verify_paper_details(evaluator, task_main, paper_info)
    await verify_author_affiliations(evaluator, task_main, paper_info, author_info)

    # Return structured evaluation summary
    return evaluator.get_summary()