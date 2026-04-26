import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nature_mars_tribo_lead_author"
TASK_DESCRIPTION = (
    "What is the name and institutional affiliation of the lead author of the Nature article "
    "published on November 26, 2025, that reports the first in situ detection of triboelectric "
    "discharges (electrical activity) on Mars using the SuperCam microphone aboard NASA's Perseverance rover?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LeadAuthorExtraction(BaseModel):
    """
    Information extracted from the agent's answer.
    """
    lead_author_name: Optional[str] = None
    lead_author_affiliation: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_lead_author_and_sources() -> str:
    return """
    From the answer text, extract the following information:

    1) lead_author_name: The lead author's (first author's) full name as explicitly written in the answer. 
       - If multiple authors are listed, choose the first author as the lead author.
       - Do not infer or guess; use exactly what the answer states.
       - If missing, return null.

    2) lead_author_affiliation: The institutional affiliation for that lead (first) author as explicitly written in the answer.
       - If multiple affiliations are listed, return the primary or the specific one that the answer associates with the lead author.
       - Do not infer or guess; use exactly what the answer states.
       - If missing, return null.

    3) source_urls: A list of all URLs cited or provided in the answer that are intended to support the information 
       (especially the Nature article page and/or DOI link). 
       - Extract actual URLs only (including those inside markdown links).
       - Include all relevant URLs mentioned in the answer.
       - If none, return an empty array.

    Return a JSON object containing these fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def pick_nature_article_url(urls: List[str]) -> Optional[str]:
    """
    Choose a likely Nature journal article URL if present.
    Preference: URLs on nature.com that contain '/articles/'.
    """
    if not urls:
        return None
    for u in urls:
        low = u.lower().strip()
        if "nature.com" in low and "/articles/" in low:
            return u
    # Fallback: any nature.com URL
    for u in urls:
        if "nature.com" in u.lower():
            return u
    return None


def choose_sources(primary: Optional[str], fallback: List[str]) -> Optional[List[str] | str]:
    """
    Choose the best sources to pass to the verifier:
    - If primary is provided, return it (single URL verification).
    - Else if fallback is non-empty, return it (single or multi URL).
    - Else return None to trigger simple verification (not recommended, but supported).
    """
    if primary:
        return primary
    if fallback:
        return fallback if len(fallback) > 1 else fallback[0]
    return None


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: LeadAuthorExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """

    # Prepare sources
    all_urls = extraction.source_urls or []
    primary_nature_url = pick_nature_article_url(all_urls)
    selected_sources = choose_sources(primary_nature_url, all_urls)

    # Add some helpful custom info for debugging
    evaluator.add_custom_info(
        {
            "all_urls": all_urls,
            "primary_nature_url": primary_nature_url,
        },
        info_type="debug",
        info_name="extracted_sources"
    )

    # ------------------------- Node 1: Identify Correct Article -------------------------
    identify_node = evaluator.add_parallel(
        id="Identify_Correct_Article",
        desc="The answer targets the Nature research article matching all stated publication and content constraints.",
        parent=evaluator.root,
        critical=True
    )

    # 1.1 Published in Nature (not another Nature-branded journal)
    pub_in_nature_leaf = evaluator.add_leaf(
        id="Published_in_Nature_Journal",
        desc="The referenced/used article is published in the journal Nature (not another Nature-branded journal).",
        parent=identify_node,
        critical=True
    )
    claim_pub_in_nature = (
        "This webpage is an article published in the flagship journal 'Nature' (the general journal), "
        "not another Nature-branded journal such as 'Nature Communications', 'Nature Astronomy', or 'Nature Geoscience'."
    )
    await evaluator.verify(
        claim=claim_pub_in_nature,
        node=pub_in_nature_leaf,
        sources=selected_sources,
        additional_instruction=(
            "Check the journal masthead and metadata on the article page. The journal must be exactly 'Nature' "
            "(the core journal). Reject specialty journals even if they are Nature-branded."
        ),
    )

    # 1.2 Publication Date = November 26, 2025
    date_leaf = evaluator.add_leaf(
        id="Publication_Date_Nov_26_2025",
        desc="The referenced/used article's publication date is November 26, 2025.",
        parent=identify_node,
        critical=True
    )
    claim_date = "This article was published on 26 November 2025 (equivalently: November 26, 2025)."
    await evaluator.verify(
        claim=claim_date,
        node=date_leaf,
        sources=selected_sources,
        additional_instruction=(
            "Confirm the publication date on the Nature article page (look for 'Published:' or metadata). "
            "Accept minor formatting variations like '26 November 2025' or 'November 26, 2025'."
        ),
    )

    # 1.3 Reports first in situ triboelectric discharges on Mars
    tribo_leaf = evaluator.add_leaf(
        id="Reports_First_In_Situ_Triboelectric_Discharges_On_Mars",
        desc="The referenced/used article reports the first in situ detection of triboelectric discharges (electrical activity) on Mars.",
        parent=identify_node,
        critical=True
    )
    claim_tribo = (
        "This article reports the first in situ detection of triboelectric discharges (electrical activity) on Mars."
    )
    await evaluator.verify(
        claim=claim_tribo,
        node=tribo_leaf,
        sources=selected_sources,
        additional_instruction=(
            "Look for explicit wording like 'first in situ detection' and 'triboelectric discharges' or equivalent phrasing "
            "in the title, abstract, or main text. Do not rely on other sources if they are not the Nature article."
        ),
    )

    # 1.4 Detection uses SuperCam microphone on Perseverance
    supercam_leaf = evaluator.add_leaf(
        id="Detection_Uses_SuperCam_Microphone_On_Perseverance",
        desc="The referenced/used article states the detection was made using the SuperCam microphone aboard NASA's Perseverance rover.",
        parent=identify_node,
        critical=True
    )
    claim_supercam = (
        "The detection described in this article was made using the SuperCam microphone aboard NASA's Perseverance rover."
    )
    await evaluator.verify(
        claim=claim_supercam,
        node=supercam_leaf,
        sources=selected_sources,
        additional_instruction=(
            "Verify that the text mentions 'SuperCam' and 'microphone' on the Perseverance rover. "
            "The Nature article itself must state this."
        ),
    )

    # ------------------------- Node 2: Lead author name and affiliation -------------------------
    # This node will be automatically gated by Identify_Correct_Article (critical and preceding in a sequential root).
    provide_node = evaluator.add_parallel(
        id="Provide_Lead_Author_Name_And_Affiliation",
        desc="Provide the lead author (defined as the first author on the publication) and that author's institutional affiliation.",
        parent=evaluator.root,
        critical=True
    )

    # 2.1 Lead author name matches the first author on the Nature publication
    lead_name_leaf = evaluator.add_leaf(
        id="Lead_Author_Name_Matches_First_Author",
        desc="The provided lead-author name corresponds to the first author listed on the identified Nature publication.",
        parent=provide_node,
        critical=True
    )
    provided_name = extraction.lead_author_name or ""
    claim_lead_name = (
        f"The first (lead) author listed on the Nature article is '{provided_name}'. "
        "Allow minor name variations (e.g., diacritics, middle initials, ordering) if clearly the same person."
    )
    await evaluator.verify(
        claim=claim_lead_name,
        node=lead_name_leaf,
        sources=selected_sources,
        additional_instruction=(
            "Check the author list on the Nature article page. The 'first author' should correspond to the provided name. "
            "Accept minor formatting differences (case, initials, hyphens, accents)."
        ),
    )

    # 2.2 Lead author affiliation provided and matches the publication
    lead_affil_leaf = evaluator.add_leaf(
        id="Lead_Author_Institutional_Affiliation_Provided_And_Matches",
        desc="The provided institutional affiliation corresponds to the identified lead (first) author as given in the publication.",
        parent=provide_node,
        critical=True
    )
    provided_affil = extraction.lead_author_affiliation or ""
    claim_affil = (
        f"On the Nature article, the first author's institutional affiliation includes or matches '{provided_affil}'. "
        "If multiple affiliations are listed for the first author, matching any one of them is acceptable."
    )
    await evaluator.verify(
        claim=claim_affil,
        node=lead_affil_leaf,
        sources=selected_sources,
        additional_instruction=(
            "Confirm the first author's affiliation(s) shown on the Nature article page. "
            "A match is acceptable if the provided affiliation is a clear synonym/abbreviation/full form (e.g., 'CNRS' vs "
            "'Centre National de la Recherche Scientifique (CNRS)'), or if it appears among multiple affiliations."
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
) -> Dict:
    """
    Evaluate an answer for the Nature article lead author and affiliation task.
    """
    # Initialize evaluator with a SEQUENTIAL root to enforce gating:
    # If the article identification fails, the lead author checks will be skipped automatically.
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Add ground-truth-like constraints (for transparency; not used for direct judging)
    evaluator.add_ground_truth(
        {
            "must_be_journal": "Nature (flagship core journal)",
            "required_publication_date": "26 November 2025",
            "key_content": [
                "first in situ detection of triboelectric discharges on Mars",
                "detected using the SuperCam microphone aboard NASA's Perseverance rover",
            ],
        },
        gt_type="target_constraints",
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_lead_author_and_sources(),
        template_class=LeadAuthorExtraction,
        extraction_name="lead_author_and_sources",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()