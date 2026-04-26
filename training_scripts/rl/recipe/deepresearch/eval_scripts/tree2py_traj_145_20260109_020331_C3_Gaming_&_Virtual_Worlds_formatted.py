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
TASK_ID = "vr_game_lineage_2024"
TASK_DESCRIPTION = (
    "Identify the video game that won the Best VR/AR category at The Game Awards 2024. Then, find the developer studio "
    "that created this game (not the publisher). Next, identify one of the founders of this developer studio. Finally, "
    "trace this founder's career history to find a game development company where they worked before founding their "
    "current studio, and provide the title of a specific game they worked on at that previous company, along with their "
    "role on that project. For your answer, provide: 1. The title of the VR game that won Best VR/AR at The Game Awards "
    "2024, 2. A reference URL from The Game Awards official website confirming this winner, 3. The name of the developer "
    "studio that created this winning game, 4. A reference URL confirming this studio developed the game, 5. The full name "
    "of one founder of this developer studio, 6. A reference URL confirming this person founded the studio, 7. The name of "
    "the game development company where this founder previously worked before founding their current studio, 8. A reference "
    "URL confirming the founder's employment at this previous company, 9. The title of a specific game the founder worked "
    "on at that previous company, 10. The founder's role or position on that game project, 11. A reference URL confirming "
    "the founder's work on this game."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VRGameLineageExtraction(BaseModel):
    # Winning game and official confirmation
    winning_game_title: Optional[str] = None
    tga_winner_url: Optional[str] = None

    # Developer studio and attribution source
    developer_studio_name: Optional[str] = None
    developer_attribution_url: Optional[str] = None

    # Founder and founder confirmation source
    founder_full_name: Optional[str] = None
    founder_confirmation_url: Optional[str] = None

    # Previous employer and employment confirmation source
    previous_company_name: Optional[str] = None
    employment_confirmation_url: Optional[str] = None

    # Specific prior game credit and role and proof source
    prior_game_title: Optional[str] = None
    prior_game_role: Optional[str] = None
    prior_game_credit_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_lineage() -> str:
    return """
    Extract the following fields as they are explicitly stated in the answer. Do not invent any information.

    Required fields to extract:
    1. winning_game_title: The title of the VR/AR game that won Best VR/AR at The Game Awards 2024.
    2. tga_winner_url: A URL from The Game Awards official website that confirms the 2024 Best VR/AR winner. Must be an explicit URL in the answer.
    3. developer_studio_name: The name of the developer studio that created the winning game (do not provide a publisher name).
    4. developer_attribution_url: A URL that confirms the named studio developed the winning game (i.e., is the developer, not just the publisher).
    5. founder_full_name: The full name of one founder of the developer studio.
    6. founder_confirmation_url: A URL that confirms this person founded/co-founded the studio.
    7. previous_company_name: The name of a game development company where the founder worked prior to founding the current studio.
    8. employment_confirmation_url: A URL that confirms the founder’s employment at that previous company.
    9. prior_game_title: The title of a specific game the founder worked on at the previous company.
    10. prior_game_role: The founder’s role/position on that specific game project (e.g., "Lead Programmer", "Designer", "Producer").
    11. prior_game_credit_url: A URL confirming the founder’s work/role on that specific game.

    Rules:
    - If any item is missing in the answer, set the corresponding field to null.
    - For URLs, extract the actual URL strings (including protocol); they must be explicitly present in the answer text.
    - Do not infer or add any data that is not stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _sanitize_urls(*urls: Optional[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if _non_empty(u):
            u_str = u.strip()
            if u_str not in seen:
                seen.add(u_str)
                result.append(u_str)
    return result


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_winning_game_checks(
    evaluator: Evaluator,
    parent_node,
    ex: VRGameLineageExtraction
) -> None:
    """Winning game identification and official citation checks."""
    # Parallel group: Winning_Game
    win_node = evaluator.add_parallel(
        id="Winning_Game",
        desc="Identify the Best VR/AR winner at The Game Awards 2024 and cite the official winner page.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Winning_Game_Title -> existence check
    evaluator.add_custom_node(
        result=_non_empty(ex.winning_game_title),
        id="Winning_Game_Title",
        desc="Provide the title of the VR/AR game that won Best VR/AR at The Game Awards 2024.",
        parent=win_node,
        critical=True
    )

    # Leaf: TGA_Winner_URL -> verify the winner claim at official site
    tga_leaf = evaluator.add_leaf(
        id="TGA_Winner_URL",
        desc="Provide a The Game Awards official website URL confirming the Best VR/AR 2024 winner.",
        parent=win_node,
        critical=True
    )
    claim = (
        f"The The Game Awards official website page confirms that the Best VR/AR winner for 2024 is "
        f"'{ex.winning_game_title}'."
    )
    await evaluator.verify(
        claim=claim,
        node=tga_leaf,
        sources=ex.tga_winner_url,
        additional_instruction=(
            "Verify this page belongs to The Game Awards official website (thegameawards.com domain or equivalent official "
            "subpaths) and explicitly confirms the 2024 Best VR/AR winner's title. The page may be a winners list or a "
            "category page showing 'Winner'."
        ),
    )


async def build_developer_studio_checks(
    evaluator: Evaluator,
    parent_node,
    ex: VRGameLineageExtraction
) -> None:
    """Developer studio identification and attribution checks."""
    dev_node = evaluator.add_parallel(
        id="Developer_Studio",
        desc="Identify the actual developer studio of the winning game (not merely the publisher) and cite a source that supports developer attribution.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Developer_Studio_Name -> existence check
    evaluator.add_custom_node(
        result=_non_empty(ex.developer_studio_name),
        id="Developer_Studio_Name",
        desc="Provide the name of the developer studio that created the winning game.",
        parent=dev_node,
        critical=True
    )

    # Leaf: Developer_Attribution_URL -> verify that the source confirms studio as developer
    dev_attr_leaf = evaluator.add_leaf(
        id="Developer_Attribution_URL",
        desc="Provide a URL that confirms the named studio developed the winning game (i.e., it is the developer, not just the publisher).",
        parent=dev_node,
        critical=True
    )
    claim = (
        f"This source confirms that '{ex.developer_studio_name}' is the developer (not just the publisher) of "
        f"the winning game '{ex.winning_game_title}'."
    )
    await evaluator.verify(
        claim=claim,
        node=dev_attr_leaf,
        sources=ex.developer_attribution_url,
        additional_instruction=(
            "Confirm that the page explicitly attributes development to the named studio (phrases like 'developed by', "
            "'developer', 'game studio') rather than only listing a publisher."
        ),
    )


async def build_founder_checks(
    evaluator: Evaluator,
    parent_node,
    ex: VRGameLineageExtraction
) -> None:
    """Founder identification and confirmation checks."""
    founder_node = evaluator.add_parallel(
        id="Studio_Founder",
        desc="Identify one founder of the developer studio and cite a source confirming founder status.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Founder_Full_Name -> existence check
    evaluator.add_custom_node(
        result=_non_empty(ex.founder_full_name),
        id="Founder_Full_Name",
        desc="Provide the full name of one founder of the developer studio.",
        parent=founder_node,
        critical=True
    )

    # Leaf: Founder_Confirmation_URL -> verify that the source confirms founder/co-founder status
    founder_confirm_leaf = evaluator.add_leaf(
        id="Founder_Confirmation_URL",
        desc="Provide a URL confirming this person founded/co-founded the studio.",
        parent=founder_node,
        critical=True
    )
    claim = (
        f"This source confirms that '{ex.founder_full_name}' founded or co-founded the studio '{ex.developer_studio_name}'."
    )
    await evaluator.verify(
        claim=claim,
        node=founder_confirm_leaf,
        sources=ex.founder_confirmation_url,
        additional_instruction=(
            "Accept terms like 'founder', 'co-founder', 'founding member'. The page should clearly link the person "
            "to the founding of the named studio."
        ),
    )


async def build_previous_employer_checks(
    evaluator: Evaluator,
    parent_node,
    ex: VRGameLineageExtraction
) -> None:
    """Founder previous employer identification and employment checks."""
    prev_node = evaluator.add_parallel(
        id="Previous_Employer",
        desc="Identify a game development company the founder worked at before founding the current studio and cite employment evidence.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Previous_Company_Name -> existence check
    evaluator.add_custom_node(
        result=_non_empty(ex.previous_company_name),
        id="Previous_Company_Name",
        desc="Provide the name of the founder’s previous employer (before founding the studio).",
        parent=prev_node,
        critical=True
    )

    # Leaf: Employment_Confirmation_URL -> verify employment at previous company
    employment_leaf = evaluator.add_leaf(
        id="Employment_Confirmation_URL",
        desc="Provide a URL confirming the founder’s employment at the previous company.",
        parent=prev_node,
        critical=True
    )
    claim_employment = (
        f"This source confirms that '{ex.founder_full_name}' worked at '{ex.previous_company_name}'."
    )
    await evaluator.verify(
        claim=claim_employment,
        node=employment_leaf,
        sources=ex.employment_confirmation_url,
        additional_instruction=(
            "The page can be a professional profile (e.g., LinkedIn), credits page (e.g., MobyGames, IGDB), or official "
            "company/press pages that explicitly state employment history."
        ),
    )

    # Leaf: Previous_Company_Is_Game_Dev -> verify company is a game development company
    is_dev_leaf = evaluator.add_leaf(
        id="Previous_Company_Is_Game_Dev",
        desc="Evidence must support that the previous employer is a game development company (not a publisher-only entity).",
        parent=prev_node,
        critical=True
    )
    claim_is_dev = (
        f"The company '{ex.previous_company_name}' is a game development company (a studio involved in developing games), "
        f"not solely a publisher."
    )
    await evaluator.verify(
        claim=claim_is_dev,
        node=is_dev_leaf,
        sources=ex.employment_confirmation_url,
        additional_instruction=(
            "Confirm from the provided source whether the company is described as a developer/studio or otherwise clearly "
            "involved in the development of games. If it is described only as a publisher with no development role, "
            "it should not pass."
        ),
    )

    # Leaf: Employment_Before_Founding_Check -> verify timeline (employment before founding current studio)
    before_found_leaf = evaluator.add_leaf(
        id="Employment_Before_Founding_Check",
        desc="Evidence must support that this employment occurred before the founder founded the current studio.",
        parent=prev_node,
        critical=True
    )
    claim_before_found = (
        f"The founder's employment at '{ex.previous_company_name}' occurred before the founding of the studio "
        f"'{ex.developer_studio_name}'."
    )
    # Use both employment and founder confirmation URLs for timeline reasoning
    await evaluator.verify(
        claim=claim_before_found,
        node=before_found_leaf,
        sources=_sanitize_urls(ex.employment_confirmation_url, ex.founder_confirmation_url),
        additional_instruction=(
            "Use explicit dates or timeline statements from the pages (employment dates, founding year) to confirm that the "
            "employment predates the founding of the current studio. If dates are not explicit, accept clear wording "
            "indicating 'prior to founding'."
        ),
    )


async def build_previous_game_credit_checks(
    evaluator: Evaluator,
    parent_node,
    ex: VRGameLineageExtraction
) -> None:
    """Specific prior game credit verification (title, role, and citation)."""
    prior_node = evaluator.add_parallel(
        id="Previous_Game_Credit",
        desc="Provide one specific game the founder worked on at the previous employer, the founder’s role, and a citation confirming the credit/role.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Prior_Game_Title -> existence check
    evaluator.add_custom_node(
        result=_non_empty(ex.prior_game_title),
        id="Prior_Game_Title",
        desc="Provide the title of a specific game the founder worked on at the previous employer.",
        parent=prior_node,
        critical=True
    )

    # Leaf: Prior_Game_Role -> existence check
    evaluator.add_custom_node(
        result=_non_empty(ex.prior_game_role),
        id="Prior_Game_Role",
        desc="Provide the founder’s role/position on that prior game project.",
        parent=prior_node,
        critical=True
    )

    # Leaf: Prior_Game_Credit_URL -> verify game credit and role
    prior_credit_leaf = evaluator.add_leaf(
        id="Prior_Game_Credit_URL",
        desc="Provide a URL confirming the founder’s work/role on that specific game.",
        parent=prior_node,
        critical=True
    )
    claim_prior_credit = (
        f"This source confirms that '{ex.founder_full_name}' worked on the game '{ex.prior_game_title}' at "
        f"'{ex.previous_company_name}' with the role '{ex.prior_game_role}'."
    )
    await evaluator.verify(
        claim=claim_prior_credit,
        node=prior_credit_leaf,
        sources=ex.prior_game_credit_url,
        additional_instruction=(
            "Accept credible credits pages (e.g., MobyGames, IGDB, official site credits, press releases) that explicitly "
            "list the person and role on the specified game."
        ),
    )


async def build_timeline_consistency_check(
    evaluator: Evaluator,
    parent_node,
    ex: VRGameLineageExtraction
) -> None:
    """Verify that the studio existed (was founded) before the award-winning game's development/release."""
    final_leaf = evaluator.add_leaf(
        id="Founding_Before_Winning_Game_Check",
        desc="The provided evidence must support that the developer studio existed (was founded) before the award-winning game’s development/release (i.e., the timeline is logically consistent).",
        parent=parent_node,
        critical=True
    )
    claim_timeline = (
        f"The studio '{ex.developer_studio_name}' existed (was founded) before the development/release of "
        f"the winning game '{ex.winning_game_title}'."
    )
    await evaluator.verify(
        claim=claim_timeline,
        node=final_leaf,
        sources=_sanitize_urls(ex.developer_attribution_url, ex.founder_confirmation_url, ex.tga_winner_url),
        additional_instruction=(
            "Use dates or explicit timeline statements from the provided sources to confirm founding precedes the game's "
            "development/release. Accept if the founding year is earlier than the game's release year, or if the sources "
            "explicitly state development started after the studio's founding."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the VR game development lineage task using the Mind2Web2 framework.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Overall must follow the lineage order
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

    # Extract all structured fields from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_lineage(),
        template_class=VRGameLineageExtraction,
        extraction_name="vr_game_lineage_extraction"
    )

    # Build the critical sequential lineage node (root-level child)
    lineage_root = evaluator.add_sequential(
        id="VR_Game_Development_Lineage",
        desc="Verify the complete lineage from The Game Awards 2024 Best VR/AR winning game to the developer studio founder's prior work, with required citations and chronological consistency.",
        parent=root,
        critical=True
    )

    # Build sub-checks in order (sequential aggregation enforces gating)
    await build_winning_game_checks(evaluator, lineage_root, extraction)
    await build_developer_studio_checks(evaluator, lineage_root, extraction)
    await build_founder_checks(evaluator, lineage_root, extraction)
    await build_previous_employer_checks(evaluator, lineage_root, extraction)
    await build_previous_game_credit_checks(evaluator, lineage_root, extraction)
    await build_timeline_consistency_check(evaluator, lineage_root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()