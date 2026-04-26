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
TASK_ID = "perseverance_supercam_nature_2025"
TASK_DESCRIPTION = (
    "Identify the peer-reviewed Nature journal paper published in the latter half of 2025 (after June) that reports "
    "atmospheric or surface phenomena detected on Mars by NASA's Perseverance rover using acoustic detection methods "
    "with the SuperCam instrument, where the lead author is affiliated with a French research institution. Provide the "
    "following information: (1) The exact publication date of the paper, (2) The lead author's name and primary "
    "institutional affiliation as stated in the paper, (3) The detection method and instrument used, (4) At least two "
    "international collaborating institutions (beyond the lead author's institution) mentioned in the author "
    "affiliations, and (5) A reference URL to the Nature paper."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SelectedPaperExtraction(BaseModel):
    """Information about the selected Nature paper as extracted from the agent's answer."""
    paper_title: Optional[str] = None
    nature_url: Optional[str] = None
    publication_date: Optional[str] = None
    lead_author_name: Optional[str] = None
    lead_author_affiliation: Optional[str] = None
    detection_method: Optional[str] = None
    instrument: Optional[str] = None
    collaborating_institutions: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_selected_paper() -> str:
    return """
    From the answer, extract information about the single Nature paper that the answer selected.

    Required fields to extract:
    1) paper_title: The exact title of the selected paper as written in the answer (string).
    2) nature_url: The URL to the paper on the Nature website (must be a Nature domain URL; include full protocol).
    3) publication_date: The exact publication date as stated in the answer (e.g., '17 September 2025' or '2025-09-17').
    4) lead_author_name: The lead (first) author's full name as given in the answer.
    5) lead_author_affiliation: The lead author's primary institutional affiliation as stated in the answer (string).
    6) detection_method: The detection method used as described in the answer (e.g., 'acoustic detection', 'microphone-based analysis', etc.).
    7) instrument: The instrument used (expected: 'SuperCam' on Perseverance).
    8) collaborating_institutions: A list of at least two collaborating institutions (beyond the lead author's primary institution) that the answer claims are listed in the article's affiliations. Use the names exactly as they appear in the answer. If fewer than two are provided, include those mentioned; if none are given, return an empty list.

    Rules:
    - Only extract information explicitly present in the answer; do not infer or invent.
    - If any field is missing, set it to null (or empty array for collaborating_institutions).
    - For nature_url, extract only if a valid URL is provided in the answer text.

    Return a JSON object matching the specified fields.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_nature_url(url: Optional[str]) -> bool:
    """Basic validation for a Nature article URL."""
    if not url or not isinstance(url, str):
        return False
    u = url.strip().lower()
    return u.startswith("http") and "nature.com" in u and "/articles" in u


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _at_least_two(names: List[str]) -> bool:
    clean = [x.strip() for x in names or [] if isinstance(x, str) and x.strip()]
    return len(clean) >= 2


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    selected: SelectedPaperExtraction,
) -> None:
    """
    Build the verification tree based on the rubric and run leaf verifications.
    """

    # Root already exists from initialize(); it's sequential by requirement
    root = evaluator.root

    # Create both major child nodes under root (order per rubric: constraints first, then required outputs)
    constraints_node = evaluator.add_parallel(
        id="paper_meets_constraints",
        desc="Selected paper satisfies all stated constraints from the question/constraints",
        parent=root,
        critical=True
    )

    outputs_node = evaluator.add_parallel(
        id="required_outputs_provided",
        desc="Answer provides all requested fields (1)–(5) for the identified paper",
        parent=root,
        critical=True
    )

    # ---------- Required outputs (existence) ----------
    # (1) Exact publication date provided
    exact_date_node = evaluator.add_custom_node(
        result=_nonempty(selected.publication_date),
        id="exact_publication_date_provided",
        desc="Provides the exact publication date of the paper",
        parent=outputs_node,
        critical=True
    )

    # (2) Lead author name and affiliation provided
    lead_name_aff_node = evaluator.add_custom_node(
        result=_nonempty(selected.lead_author_name) and _nonempty(selected.lead_author_affiliation),
        id="lead_author_name_and_affiliation_provided",
        desc="Provides the lead author’s name and primary institutional affiliation exactly as stated in the paper",
        parent=outputs_node,
        critical=True
    )

    # (3) Method and instrument described
    method_instrument_node = evaluator.add_custom_node(
        result=_nonempty(selected.detection_method) and _nonempty(selected.instrument),
        id="method_and_instrument_described",
        desc="Describes the detection method and the instrument used",
        parent=outputs_node,
        critical=True
    )

    # (4) Two international collaborating institutions listed
    collab_two_node = evaluator.add_custom_node(
        result=_at_least_two(selected.collaborating_institutions),
        id="two_international_collaborating_institutions",
        desc="Lists at least two international collaborating institutions beyond the lead author’s institution that appear in the author affiliations",
        parent=outputs_node,
        critical=True
    )

    # (5) Nature reference URL provided
    nature_url_ok = _is_valid_nature_url(selected.nature_url)
    nature_url_node = evaluator.add_custom_node(
        result=nature_url_ok,
        id="nature_reference_url_provided",
        desc="Provides a valid reference URL to the Nature paper",
        parent=outputs_node,
        critical=True
    )

    # Determine the URL to use for evidence-based verification
    source_url: Optional[str] = selected.nature_url if nature_url_ok else None

    # Helper: for all constraint checks, depend on the Nature URL existence node,
    # so that if URL is missing/invalid, those leaves are skipped (and the parent fails as critical).
    extra_pre = [nature_url_node]

    # ---------- Constraints verifications (all critical leaves) ----------
    # 1) Published in Nature (flagship journal, not portfolio titles)
    pub_in_nature_leaf = evaluator.add_leaf(
        id="published_in_nature",
        desc="Paper is published in the journal Nature",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="This paper is published in the flagship journal Nature (not Nature Communications, Nature Astronomy, Nature Geoscience, or other portfolio journals).",
        node=pub_in_nature_leaf,
        sources=source_url,
        additional_instruction="Verify the journal branding on the page. Accept only 'Nature' (the flagship journal). If the page is any other Nature-branded journal, mark as not supported.",
        extra_prerequisites=extra_pre
    )

    # 2) Published in H2 2025 (between July 1 and Dec 31 inclusive)
    pub_h2_leaf = evaluator.add_leaf(
        id="published_in_h2_2025",
        desc="Paper publication date is between July 1, 2025 and December 31, 2025 (inclusive)",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="This paper was published between July 1, 2025 and December 31, 2025 (inclusive).",
        node=pub_h2_leaf,
        sources=source_url,
        additional_instruction="Use the 'Published' or 'Published online' date on the Nature article. If the date falls outside the range, mark as not supported.",
        extra_prerequisites=extra_pre
    )

    # 3) Peer-reviewed (Nature research article, not editorial)
    peer_reviewed_leaf = evaluator.add_leaf(
        id="peer_reviewed",
        desc="Paper is peer-reviewed (i.e., a peer-reviewed Nature research article, not a non-peer-reviewed format)",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="This Nature page is a peer-reviewed research article (e.g., Article or Letter), not an editorial, news, comment, or other non-peer-reviewed format.",
        node=peer_reviewed_leaf,
        sources=source_url,
        additional_instruction="Check the article type on the page (e.g., 'Article', 'Letter'). If it is 'News & Views', 'Editorial', 'Comment', etc., mark as not supported.",
        extra_prerequisites=extra_pre
    )

    # 4) Based on NASA's Perseverance rover
    perseverance_leaf = evaluator.add_leaf(
        id="perseverance_based",
        desc="Findings are based on data from NASA’s Perseverance rover",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The research described in the paper is based on data from NASA's Perseverance Mars rover.",
        node=perseverance_leaf,
        sources=source_url,
        additional_instruction="Look for explicit mentions of 'Perseverance'. If the rover is not clearly involved, mark as not supported.",
        extra_prerequisites=extra_pre
    )

    # 5) SuperCam used
    supercam_leaf = evaluator.add_leaf(
        id="supercam_used",
        desc="Findings are based on data from the SuperCam instrument",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The study uses data from the SuperCam instrument on Perseverance.",
        node=supercam_leaf,
        sources=source_url,
        additional_instruction="Confirm that SuperCam is explicitly mentioned as an instrument used in the methods/findings.",
        extra_prerequisites=extra_pre
    )

    # 6) Acoustic detection involved
    acoustic_leaf = evaluator.add_leaf(
        id="acoustic_detection_involved",
        desc="Research involves acoustic detection methods as part of the findings",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The research employs acoustic detection or analysis (e.g., microphone/sound/acoustic measurements) as part of its methodology or findings.",
        node=acoustic_leaf,
        sources=source_url,
        additional_instruction="Look for terms like 'acoustic', 'microphone', 'sound', or 'audio'. If no acoustic methodology is involved, mark as not supported.",
        extra_prerequisites=extra_pre
    )

    # 7) Reports atmospheric or surface phenomena on Mars
    phenomena_leaf = evaluator.add_leaf(
        id="mars_atmospheric_or_surface_phenomena",
        desc="Paper reports detection of atmospheric or surface phenomena on Mars",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The paper reports the detection or characterization of atmospheric or surface phenomena on Mars.",
        node=phenomena_leaf,
        sources=source_url,
        additional_instruction="Examples include wind, turbulence, dust devils, atmospheric properties, or surface mechanical/acoustic properties. If not present, mark as not supported.",
        extra_prerequisites=extra_pre
    )

    # 8) Lead author’s primary affiliation is a French research institution
    lead_fr_leaf = evaluator.add_leaf(
        id="lead_author_french_institution",
        desc="Lead author’s primary institutional affiliation is a French research institution (as stated in the paper)",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The first (lead) author's primary affiliation listed in the article is a French research institution located in France.",
        node=lead_fr_leaf,
        sources=source_url,
        additional_instruction="Confirm from the author list and affiliations that the first author has a primary affiliation with an institution in France (e.g., CNRS, IRAP, ISAE-SUPAERO, Université de Toulouse). If not clearly in France, mark as not supported.",
        extra_prerequisites=extra_pre
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
    Evaluate an answer for the Perseverance/SuperCam Nature 2025 task.
    """
    # Initialize evaluator with sequential root aggregation (as per rubric)
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
        default_model=model
    )

    # Extract structured information from the answer
    selected = await evaluator.extract(
        prompt=prompt_extract_selected_paper(),
        template_class=SelectedPaperExtraction,
        extraction_name="selected_paper"
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, selected)

    # Return evaluation summary
    return evaluator.get_summary()