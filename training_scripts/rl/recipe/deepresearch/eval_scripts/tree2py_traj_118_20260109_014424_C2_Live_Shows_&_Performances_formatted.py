import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_accessibility_2026"
TASK_DESCRIPTION = """
I am planning to attend a Broadway show in New York City for a family member who uses a wheelchair and has hearing impairment. I need to find a show that is currently playing (as of January 2026) at a theater that provides all of the following accessibility features:

1. Wheelchair-accessible seating on the orchestra level (ground floor)
2. Captioning devices (closed captioning) available for the performance
3. Assistive listening devices (such as infrared headsets or induction loops) available for the performance

Please identify one Broadway show that meets all these requirements. For your answer, provide:
- The name of the show
- The name of the theater where it is playing
- Confirmation that wheelchair-accessible seating is available on the orchestra level
- Confirmation that both captioning devices and assistive listening devices are offered at this theater
- A reference URL that confirms the accessibility features of the theater
"""


# --------------------------------------------------------------------------- #
# Data models for information extraction                                      #
# --------------------------------------------------------------------------- #
class AccessibilityConfirmations(BaseModel):
    wheelchair_orchestra_confirmation: Optional[str] = None
    captioning_confirmation: Optional[str] = None
    assistive_listening_confirmation: Optional[str] = None


class ShowSelectionExtraction(BaseModel):
    show_name: Optional[str] = None
    theater_name: Optional[str] = None
    currently_playing_as_of_jan_2026: Optional[str] = None
    accessibility_confirmations: Optional[AccessibilityConfirmations] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_show_selection() -> str:
    return """
    Extract the key details for the Broadway show selection from the answer. If multiple shows are mentioned, extract only the first one that appears in the answer. Return null for any field that is not explicitly stated in the answer.

    Required fields to extract:
    - show_name: The exact name of the Broadway show.
    - theater_name: The exact name of the theater where the show is playing.
    - currently_playing_as_of_jan_2026: Copy the sentence or short phrase from the answer that asserts the show is "currently playing" (or equivalent wording such as "running," "now playing") on Broadway in New York City as of January 2026. If not stated, return null.
    - accessibility_confirmations:
        - wheelchair_orchestra_confirmation: Copy the sentence or short phrase that confirms wheelchair-accessible seating is available specifically on the orchestra level (ground floor). If not stated, return null.
        - captioning_confirmation: Copy the sentence or short phrase that confirms captioning devices (closed captioning) are available. If not stated, return null.
        - assistive_listening_confirmation: Copy the sentence or short phrase that confirms assistive listening devices (e.g., infrared headsets, induction loops/hearing loops) are available. If not stated, return null.
    - reference_urls: Extract all URLs present in the answer that are intended to support the accessibility claims or theater information. The URLs may appear as plain links or markdown links; return the actual URLs. Only extract URLs explicitly present in the answer. Include complete URLs; if a URL lacks a protocol, prepend http://

    Do not invent any information. Copy text exactly where applicable.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: ShowSelectionExtraction
) -> None:
    """
    Build the verification tree according to the rubric and perform verifications.
    """
    # Top-level critical node representing the entire task (sequential: eligibility first, then accessibility)
    top_node = evaluator.add_sequential(
        id="Broadway_Show_Selection",
        desc="Select one Broadway show currently playing (as of Jan 2026) in NYC that meets all specified accessibility requirements, and provide required supporting information.",
        parent=evaluator.root,
        critical=True
    )

    # -----------------------------------------------------------------------
    # 1) Show and Theater Details (parallel critical)
    # -----------------------------------------------------------------------
    details_node = evaluator.add_parallel(
        id="Show_And_Theater_Details",
        desc="The answer identifies the show and theater and establishes eligibility (currently playing on Broadway in NYC as of Jan 2026).",
        parent=top_node,
        critical=True
    )

    # 1.a Show name provided (critical existence check)
    show_name_exists = evaluator.add_custom_node(
        result=(extracted.show_name is not None and str(extracted.show_name).strip() != ""),
        id="Show_Name_Provided",
        desc="Provides the name of the show.",
        parent=details_node,
        critical=True
    )

    # 1.b Theater name provided (critical existence check)
    theater_name_exists = evaluator.add_custom_node(
        result=(extracted.theater_name is not None and str(extracted.theater_name).strip() != ""),
        id="Theater_Name_Provided",
        desc="Provides the name of the theater where the show is playing.",
        parent=details_node,
        critical=True
    )

    # 1.c Currently playing as of Jan 2026 (critical verification based on the answer content)
    currently_playing_leaf = evaluator.add_leaf(
        id="Currently_Playing_As_Of_Jan_2026",
        desc="Confirms the show is currently playing on Broadway in New York City as of January 2026.",
        parent=details_node,
        critical=True
    )
    show_label = extracted.show_name.strip() if extracted.show_name else "the show"
    theater_label = extracted.theater_name.strip() if extracted.theater_name else "the theater"
    claim_currently_playing = (
        f"As of January 2026, {show_label} is currently playing at {theater_label} on Broadway in New York City."
    )
    await evaluator.verify(
        claim=claim_currently_playing,
        node=currently_playing_leaf,
        additional_instruction=(
            "Judge only based on what the answer explicitly asserts. Accept equivalent wordings like "
            "'currently running', 'now playing', or 'currently on Broadway at <theater>'. If the answer does not "
            "explicitly claim current Broadway performance in NYC as of Jan 2026, mark as incorrect."
        )
    )

    # -----------------------------------------------------------------------
    # 2) Accessibility and Sourcing (parallel critical)
    # -----------------------------------------------------------------------
    acc_parent = evaluator.add_parallel(
        id="Accessibility_And_Sourcing",
        desc="All required accessibility features are confirmed for the theater, supported by verifiable source URL(s).",
        parent=top_node,
        critical=True
    )

    # 2.b Reference URL provided and relevant (create this first so feature checks can depend on it)
    # If there are URLs, verify by fetching and confirming the page(s) support the accessibility features.
    # If no URLs, make this node fail directly via custom node.
    theater_name = extracted.theater_name.strip() if extracted.theater_name else "the theater"
    urls = list(dict.fromkeys(extracted.reference_urls)) if extracted.reference_urls else []

    if urls:
        ref_leaf = evaluator.add_leaf(
            id="Reference_URL_Provided_And_Relevant",
            desc="Provides at least one reference URL that supports the claimed accessibility features for the specified theater.",
            parent=acc_parent,
            critical=True
        )
        claim_reference_supports = (
            f"At least one of these URLs is an official or reliable page that confirms that {theater_name} "
            f"provides wheelchair-accessible seating on the orchestra (ground floor), captioning devices, and "
            f"assistive listening devices."
        )
        await evaluator.verify(
            claim=claim_reference_supports,
            node=ref_leaf,
            sources=urls,
            additional_instruction=(
                "Verify that at least one URL is specifically about the named theater (the venue itself) and "
                "explicitly mentions accessibility features including: (1) wheelchair-accessible seating, "
                "(2) captioning or closed captioning (including smartphone-based solutions like GalaPro or "
                "open-caption offerings), and (3) assistive listening devices (infrared headsets, induction/hearing "
                "loops, or equivalent). Pages from the theater/operator (e.g., Shubert, Nederlander, Jujamcyn), "
                "Broadway League, or the theater's official site are acceptable, as long as the specific theater is "
                "clearly indicated and the features are explicitly stated."
            )
        )
    else:
        # No URLs were provided in the answer; fail this critical requirement
        ref_leaf = evaluator.add_custom_node(
            result=False,
            id="Reference_URL_Provided_And_Relevant",
            desc="Provides at least one reference URL that supports the claimed accessibility features for the specified theater.",
            parent=acc_parent,
            critical=True
        )

    # 2.a Required accessibility features (parallel critical)
    features_parent = evaluator.add_parallel(
        id="Required_Accessibility_Features",
        desc="The theater provides all three required accessibility features.",
        parent=acc_parent,
        critical=True
    )

    # Wheelchair-accessible seating on orchestra level
    wc_leaf = evaluator.add_leaf(
        id="Wheelchair_Accessible_Orchestra_Level",
        desc="Confirms wheelchair-accessible seating is available on the orchestra level (ground floor).",
        parent=features_parent,
        critical=True
    )
    claim_wc = (
        f"{theater_name} provides wheelchair-accessible seating on the orchestra level (ground floor)."
    )
    await evaluator.verify(
        claim=claim_wc,
        node=wc_leaf,
        sources=urls,  # depend on the provided reference URLs
        additional_instruction=(
            "Look for the exact theater's accessibility information. Accept equivalent statements such as "
            "'accessible seating is in the Orchestra' or 'wheelchair seating located on the Orchestra level.' "
            "If the provided URLs are irrelevant or do not mention this, mark as not supported."
        )
    )

    # Captioning devices (closed captioning) available
    cc_leaf = evaluator.add_leaf(
        id="Captioning_Available",
        desc="Confirms captioning devices (closed captioning) are available for the performance.",
        parent=features_parent,
        critical=True
    )
    claim_cc = (
        f"{theater_name} offers captioning devices or closed captioning for performances."
    )
    await evaluator.verify(
        claim=claim_cc,
        node=cc_leaf,
        sources=urls,
        additional_instruction=(
            "Evidence can include 'closed captioning', 'CC available via device', 'GalaPro closed captioning', "
            "'open caption performances', or similar. The page must clearly indicate captioning support is available "
            "for audience members at this theater."
        )
    )

    # Assistive listening devices available
    al_leaf = evaluator.add_leaf(
        id="Assistive_Listening_Available",
        desc="Confirms assistive listening devices are available for the performance.",
        parent=features_parent,
        critical=True
    )
    claim_al = (
        f"{theater_name} provides assistive listening devices for performances, such as infrared headsets or "
        f"induction (hearing) loops."
    )
    await evaluator.verify(
        claim=claim_al,
        node=al_leaf,
        sources=urls,
        additional_instruction=(
            "Accept mentions of 'assistive listening system', 'infrared headsets', 'FM system', 'hearing loop', "
            "'induction loop', 'telecoil/T-coil', or similar. The page must clearly connect these devices to "
            "this specific theater's performances."
        )
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Broadway accessibility task and return a structured result.
    """
    # Initialize evaluator with a meaningful root
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Overall flow: details first, then accessibility
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_show_selection(),
        template_class=ShowSelectionExtraction,
        extraction_name="show_selection_extraction"
    )

    # Build tree and verify
    await build_and_verify_tree(evaluator, extracted)

    # Optional: record custom info for debugging
    evaluator.add_custom_info(
        info={
            "note": "Evaluation completed for Broadway accessibility selection",
            "extracted_show_name": extracted.show_name,
            "extracted_theater_name": extracted.theater_name,
            "num_reference_urls": len(extracted.reference_urls) if extracted.reference_urls else 0
        },
        info_type="evaluation_metadata"
    )

    # Return structured summary
    return evaluator.get_summary()