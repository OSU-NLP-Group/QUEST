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
TASK_ID = "opec_venezuela_ops"
TASK_DESCRIPTION = (
    "Venezuela is one of the founding members of OPEC (Organization of the Petroleum Exporting Countries). "
    "What year was OPEC founded, and how many countries were founding members? Additionally, name one U.S.-based oil "
    "company that currently operates in Venezuela through joint ventures with PDVSA (Petróleos de Venezuela S.A.), "
    "Venezuela's state-owned oil company."
)

# Ground truth (for reference in summary only; not used as an oracle in verification)
GROUND_TRUTH = {
    "opec": {
        "founding_date": "September 14, 1960",
        "founding_location": "Baghdad, Iraq",
        "founding_year": "1960",
        "founding_member_count": "5",
        "founding_members": ["Iran", "Iraq", "Kuwait", "Saudi Arabia", "Venezuela"],
        "venezuela_current_member": True,
    },
    "us_oil_company_expected": "Chevron",
    "example_jv": "Petropiar"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OPECExtraction(BaseModel):
    founding_date_text: Optional[str] = None
    founding_year: Optional[str] = None
    founding_location: Optional[str] = None
    founding_member_count: Optional[str] = None
    founding_members: List[str] = Field(default_factory=list)
    venezuela_current_member: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class USOilExtraction(BaseModel):
    company_name: Optional[str] = None
    operates_through_pdvsajv: Optional[bool] = None
    mentions_petropiar: Optional[bool] = None
    jv_names: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class FullExtraction(BaseModel):
    opec: Optional[OPECExtraction] = None
    us_oil: Optional[USOilExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the answer's key claims and the URLs/citations it provides for those claims.

    Return a JSON object with two top-level fields: "opec" and "us_oil".

    For "opec", extract:
    - founding_date_text: The exact founding date text stated (e.g., "September 14, 1960"). If only the year is stated (e.g., "1960"), still capture that.
    - founding_year: The founding year if stated (e.g., "1960").
    - founding_location: The location if stated (e.g., "Baghdad, Iraq").
    - founding_member_count: The number of founding members if stated (e.g., "5" or "five" — return a normalized string like "5" if possible; otherwise keep the original text).
    - founding_members: The list of founding countries if stated (e.g., ["Iran", "Iraq", "Kuwait", "Saudi Arabia", "Venezuela"]). Normalize common variants (e.g., "Kingdom of Saudi Arabia" -> "Saudi Arabia", "Islamic Republic of Iran" -> "Iran") when reasonable.
    - venezuela_current_member: true/false depending on whether the answer explicitly claims Venezuela is still an OPEC member.
    - sources: an array of URLs cited in the answer specifically to support any OPEC founding/membership claims (official or reliable sources only if available). Extract only URLs explicitly present in the answer.

    For "us_oil", extract:
    - company_name: The U.S.-based oil company the answer names as currently operating in Venezuela (expected example: "Chevron").
    - operates_through_pdvsajv: true/false depending on whether the answer explicitly claims the company operates through joint ventures with PDVSA.
    - mentions_petropiar: true/false depending on whether the answer mentions "Petropiar" or a Chevron-PDVSA JV that includes Petropiar.
    - jv_names: list any named JVs mentioned (e.g., ["Petropiar", "Petroboscán"]).
    - sources: an array of URLs cited in the answer specifically to support the company's operations in Venezuela through PDVSA JVs (official pages like chevron.com, PDVSA, or reliable news outlets). Extract only URLs explicitly present in the answer.

    URL extraction rules:
    - Extract only valid URLs that are explicitly present in the answer text (including markdown links).
    - Do not invent or infer URLs.
    - If no URLs are provided for a category, return an empty list for "sources".
    """


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_opec_subtree(evaluator: Evaluator, parent_node, extraction: Optional[FullExtraction]) -> None:
    """
    Build and verify the OPEC founding/membership claims subtree.
    These checks focus on whether the answer text itself states the required information.
    """
    opec_node = evaluator.add_parallel(
        id="OPEC_Founding_And_Membership_Claims",
        desc="Provides OPEC founding details and founding members as specified by constraints, including Venezuela’s current membership status.",
        parent=parent_node,
        critical=True
    )

    # 1) Founding date and location (answer content check)
    n_date_loc = evaluator.add_leaf(
        id="OPEC_Founding_Date_And_Location",
        desc="States OPEC was founded on September 14, 1960, in Baghdad, Iraq.",
        parent=opec_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that OPEC was founded on September 14, 1960, in Baghdad, Iraq.",
        node=n_date_loc,
        additional_instruction=(
            "Judge based on the answer text only. Accept minor formatting variants such as '14 September 1960'. "
            "For the location, 'Baghdad' (optionally followed by ', Iraq') should be treated as acceptable."
        )
    )

    # 2) Founding member count (answer content check)
    n_count = evaluator.add_leaf(
        id="OPEC_Founding_Member_Count",
        desc="States OPEC was founded by exactly 5 countries.",
        parent=opec_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that OPEC was founded by exactly 5 countries.",
        node=n_count,
        additional_instruction="Accept both the numeral '5' and the word 'five'. Judge based on the answer text only."
    )

    # 3) Founding members list (answer content check)
    n_members = evaluator.add_leaf(
        id="OPEC_Founding_Members_List",
        desc="Names the five founding members: Iran, Iraq, Kuwait, Saudi Arabia, and Venezuela.",
        parent=opec_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer lists the five founding members of OPEC as Iran, Iraq, Kuwait, Saudi Arabia, and Venezuela.",
        node=n_members,
        additional_instruction=(
            "Judge based on the answer text only. Allow reasonable variants like 'Kingdom of Saudi Arabia' for Saudi Arabia "
            "and 'Islamic Republic of Iran' for Iran; capitalization and punctuation differences are acceptable."
        )
    )

    # 4) Venezuela current OPEC member (answer content check)
    n_venez_member = evaluator.add_leaf(
        id="Venezuela_Current_OPEC_Member",
        desc="States that Venezuela remains a current OPEC member.",
        parent=opec_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that Venezuela remains a current OPEC member of OPEC.",
        node=n_venez_member,
        additional_instruction="Judge based on the answer text only."
    )


async def verify_us_ops_subtree(evaluator: Evaluator, parent_node, extraction: Optional[FullExtraction]) -> None:
    """
    Build and verify the Chevron-in-Venezuela claims subtree.
    These checks focus on whether the answer text itself states the required information.
    """
    us_ops_node = evaluator.add_parallel(
        id="US_Oil_Company_Venezuela_Operations",
        desc="Identifies the required U.S.-based oil company and its Venezuela operations via PDVSA joint ventures, including the specified JV interest per constraints.",
        parent=parent_node,
        critical=True
    )

    # 1) Chevron named as operator (answer content check)
    n_chevron = evaluator.add_leaf(
        id="Chevron_Identified_As_US_Based_Operator",
        desc="Names Chevron as the U.S.-based oil company currently operating in Venezuela.",
        parent=us_ops_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer names Chevron as the U.S.-based oil company currently operating in Venezuela.",
        node=n_chevron,
        additional_instruction="Judge based on the answer text only. Minor capitalization or wording variations are acceptable."
    )

    # 2) Operates through PDVSA JVs (answer content check)
    n_jv = evaluator.add_leaf(
        id="Operates_Through_PDVSA_Joint_Ventures",
        desc="States Chevron operates in Venezuela through joint ventures with PDVSA.",
        parent=us_ops_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that Chevron operates in Venezuela through joint ventures with PDVSA.",
        node=n_jv,
        additional_instruction="Judge based on the answer text only. Variants like 'joint ventures with Venezuela's state oil company PDVSA' are acceptable."
    )

    # 3) Mentions Petropiar JV interest (answer content check)
    n_petropiar = evaluator.add_leaf(
        id="Mentions_Petropiar_JV_Interest",
        desc="Mentions that Chevron holds interests in a specific joint venture including Petropiar.",
        parent=us_ops_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer mentions that Chevron holds an interest in the Petropiar joint venture.",
        node=n_petropiar,
        additional_instruction="Judge based on the answer text only. Accept wording variants like 'Chevron-PDVSA Petropiar JV'."
    )


async def verify_sources_subtree(evaluator: Evaluator, parent_node, extraction: Optional[FullExtraction]) -> None:
    """
    Build and verify the sources subtree.
    These checks verify that the claims are supported by cited web sources included in the answer.
    """
    sources_node = evaluator.add_parallel(
        id="Sources_Verifiability",
        desc="Provides citations such that all required claims are verifiable via official or reliable sources.",
        parent=parent_node,
        critical=True
    )

    opec_sources: List[str] = []
    chevron_sources: List[str] = []

    if extraction and extraction.opec and extraction.opec.sources:
        # Filter out obviously invalid entries (basic heuristic)
        opec_sources = [u for u in extraction.opec.sources if isinstance(u, str) and u.strip().startswith(("http://", "https://"))]

    if extraction and extraction.us_oil and extraction.us_oil.sources:
        chevron_sources = [u for u in extraction.us_oil.sources if isinstance(u, str) and u.strip().startswith(("http://", "https://"))]

    # 1) OPEC sources collectively support all OPEC claims (date, location, count, list)
    n_opec_sources = evaluator.add_leaf(
        id="Source_For_All_OPEC_Claims",
        desc="Includes at least one citation/reference supporting the OPEC founding date/location and the founding-member count and/or list (collectively covering all OPEC claims made).",
        parent=sources_node,
        critical=True
    )
    await evaluator.verify(
        claim="OPEC was founded on September 14, 1960 in Baghdad, Iraq by five countries: Iran, Iraq, Kuwait, Saudi Arabia, and Venezuela.",
        node=n_opec_sources,
        sources=opec_sources if opec_sources else None,
        additional_instruction=(
            "Verify this claim strictly against the cited webpage(s). If no valid URL citations were provided in the answer for the OPEC claims, "
            "or if the provided pages do not support these details, judge as NOT SUPPORTED."
        )
    )

    # 2) Chevron-in-Venezuela sources support current JV operations with PDVSA including Petropiar
    n_chevron_sources = evaluator.add_leaf(
        id="Source_For_Chevron_Venezuela_JV_Claims",
        desc="Includes at least one citation/reference supporting Chevron’s current operations in Venezuela via PDVSA joint ventures, including the Petropiar-related JV interest.",
        parent=sources_node,
        critical=True
    )
    await evaluator.verify(
        claim="Chevron currently operates in Venezuela through joint ventures with PDVSA, including the Petropiar joint venture.",
        node=n_chevron_sources,
        sources=chevron_sources if chevron_sources else None,
        additional_instruction=(
            "Verify strictly against the cited webpage(s). Accept reliable sources such as chevron.com, PDVSA, government releases, "
            "or reputable news organizations. If the answer provided no valid URL citations for these Chevron-in-Venezuela JV claims, "
            "or if the pages do not support them, judge as NOT SUPPORTED."
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
    Evaluate an answer for the OPEC founding and U.S. oil operations in Venezuela task.
    """
    # 1) Initialize evaluator (root is a non-critical container)
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

    # 2) Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=FullExtraction,
        extraction_name="opec_us_oil_extraction",
    )

    # 3) Add GT info (for transparency only)
    evaluator.add_ground_truth(
        {"expected": GROUND_TRUTH},
        gt_type="ground_truth"
    )

    # 4) Build top-level critical node required by rubric
    top = evaluator.add_parallel(
        id="Complete_OPEC_and_Venezuela_Oil_Information",
        desc="Answer includes all required OPEC founding details and the required U.S.-based oil company operating in Venezuela via PDVSA JVs, with verifiable sources.",
        parent=root,
        critical=True
    )

    # 5) Construct and verify subtrees
    await verify_opec_subtree(evaluator, top, extraction)
    await verify_us_ops_subtree(evaluator, top, extraction)
    await verify_sources_subtree(evaluator, top, extraction)

    # 6) Return summary with verification tree and scores
    return evaluator.get_summary()