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
TASK_ID = "wicked_director_awards"
TASK_DESCRIPTION = """
Identify the director who directed both Wicked (2024) and Wicked: For Good (2025). For Wicked (2024), provide the total number of Academy Award nominations it received at the 97th Academy Awards, the total number of awards it won, and identify the winners of the Best Costume Design and Best Production Design categories. For Wicked: For Good (2025), provide the total number of Academy Award nominations it received at the 2026 Academy Awards. Include reference URLs for all information.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DirectorClaim(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ReleaseDateClaim(BaseModel):
    date_us: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NominationsClaim(BaseModel):
    total: Optional[str] = None
    included_categories: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class WinsClaim(BaseModel):
    total: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CostumeDesignClaim(BaseModel):
    winner: Optional[str] = None
    first_african_american_male: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class ProductionDesignClaim(BaseModel):
    winners: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class WFGNominationsClaim(BaseModel):
    total: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class WickedAwardsExtraction(BaseModel):
    director: Optional[DirectorClaim] = None
    wicked_2024_release_us: Optional[ReleaseDateClaim] = None
    wfg_2025_release_us: Optional[ReleaseDateClaim] = None
    wicked_2024_97th_noms: Optional[NominationsClaim] = None
    wicked_2024_97th_wins: Optional[WinsClaim] = None
    wicked_2024_best_costume: Optional[CostumeDesignClaim] = None
    wicked_2024_best_production: Optional[ProductionDesignClaim] = None
    wfg_2026_noms: Optional[WFGNominationsClaim] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_wicked_awards() -> str:
    return """
    Extract the following information exactly as stated in the answer, along with the explicit reference URLs the answer provides for each claim. Do not infer or invent anything. If an item is not present in the answer, set it to null (or an empty list for arrays).

    1) director:
       - name: The person the answer states directed BOTH "Wicked (2024)" and "Wicked: For Good (2025)".
       - sources: All URLs the answer cites to support that director claim. Include only actual URLs.

    2) wicked_2024_release_us:
       - date_us: The US release date the answer states for "Wicked (2024)" (e.g., "November 22, 2024").
       - sources: All URLs the answer cites to support this release date.

    3) wfg_2025_release_us:
       - date_us: The US release date stated for "Wicked: For Good (2025)". Note: In some sources it may be styled as "Wicked: Part Two – For Good" or "Wicked Part Two".
       - sources: All URLs the answer cites to support this release date.

    4) wicked_2024_97th_noms:
       - total: The total number of nominations the answer states "Wicked (2024)" received at the 97th Academy Awards.
       - included_categories: If the answer explicitly lists nominated categories (and any names in parentheses), capture each as a single string item, e.g., "Best Picture", "Best Actress (Cynthia Erivo)", "Best Supporting Actress (Ariana Grande)". If not listed, return an empty array.
       - sources: All URLs the answer cites to support this nominations statement.

    5) wicked_2024_97th_wins:
       - total: The total number of Academy Award wins the answer states "Wicked (2024)" received at the 97th Academy Awards.
       - sources: All URLs the answer cites to support this wins statement.

    6) wicked_2024_best_costume:
       - winner: The person the answer states won Best Costume Design for "Wicked (2024)".
       - first_african_american_male: Set to true only if the answer explicitly says this win made him "the first African American male costume designer to win this award"; otherwise set to null if not mentioned.
       - sources: All URLs the answer cites to support this statement.

    7) wicked_2024_best_production:
       - winners: The list of individual people the answer states won Best Production Design for "Wicked (2024)" (each person as a separate string).
       - sources: All URLs the answer cites to support this statement.

    8) wfg_2026_noms:
       - total: The total number of nominations the answer states "Wicked: For Good (2025)" received at the 2026 Academy Awards (also known as the 98th Academy Awards).
       - sources: All URLs the answer cites to support this statement.

    URL extraction rules:
    - Extract only actual URLs from the answer (plain or inside markdown).
    - Do not include non-URL citations like "according to Wikipedia" without a URL.
    - Return complete URLs with protocol.

    Return a single JSON object strictly conforming to the specified schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_nonempty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def _list_icontains_any(items: List[str], substr: str) -> bool:
    substr_l = substr.lower()
    for it in items:
        if isinstance(it, str) and substr_l in it.lower():
            return True
    return False


def _names_in_list(items: List[str], required_names: List[str]) -> bool:
    items_l = [i.lower() for i in items if isinstance(i, str)]
    return all(any(rn.lower() in it for it in items_l) for rn in required_names)


def _categories_contain_required(included: List[str]) -> bool:
    """
    Check that the answer's included categories list mentions the required items:
      - Best Picture
      - Best Actress (Cynthia Erivo)
      - Best Supporting Actress (Ariana Grande)
    Use case-insensitive substring checks to be tolerant of minor formatting differences.
    """
    included_l = [c.lower() for c in included if isinstance(c, str)]
    # Best Picture
    bp = any("best picture" in c for c in included_l)
    # Best Actress (Cynthia Erivo)
    ba = any(("best actress" in c) and ("cynthia" in c) and ("erivo" in c) for c in included_l)
    # Best Supporting Actress (Ariana Grande)
    bsa = any(("best supporting actress" in c) and ("ariana" in c) and ("grande" in c) for c in included_l)
    return bp and ba and bsa


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_director_checks(evaluator: Evaluator, parent, extracted: WickedAwardsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Director",
        desc="State that the director who directed both Wicked (2024) and Wicked: For Good (2025) is Jon M. Chu, with a supporting reference URL.",
        parent=parent,
        critical=True
    )
    info = extracted.director or DirectorClaim()
    urls = _valid_urls(info.sources)

    presence = evaluator.add_custom_node(
        result=_has_nonempty_str(info.name) and len(urls) > 0,
        id="Director_presence",
        desc="Director name for both films is provided with source URL(s)",
        parent=node,
        critical=True
    )

    leaf_wicked_2024 = evaluator.add_leaf(
        id="Director_wicked_2024_supported",
        desc="The stated director directed Wicked (2024)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The director of Wicked (2024) is {info.name}.",
        node=leaf_wicked_2024,
        sources=urls,
        additional_instruction="Verify that the page explicitly credits this person as the director of the 2024 film 'Wicked'. Minor name formatting variations are acceptable."
    )

    leaf_wicked_for_good = evaluator.add_leaf(
        id="Director_wicked_for_good_supported",
        desc="The stated director directed Wicked: For Good (2025)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The director of Wicked: For Good (2025) is {info.name}.",
        node=leaf_wicked_for_good,
        sources=urls,
        additional_instruction="Verify that the page credits this person as the director of 'Wicked: For Good (2025)'. Allow the film to be titled 'Wicked: Part Two – For Good' or 'Wicked Part Two'."
    )


async def build_release_date_checks_2024(evaluator: Evaluator, parent, extracted: WickedAwardsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Wicked_2024_Release_Date_US",
        desc="State that Wicked (2024) US release date is November 22, 2024, with a supporting reference URL.",
        parent=parent,
        critical=True
    )
    info = extracted.wicked_2024_release_us or ReleaseDateClaim()
    urls = _valid_urls(info.sources)

    presence = evaluator.add_custom_node(
        result=_has_nonempty_str(info.date_us) and len(urls) > 0,
        id="Wicked_2024_Release_Date_US_presence",
        desc="US release date for Wicked (2024) is provided with source URL(s)",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Wicked_2024_Release_Date_US_supported",
        desc="US release date for Wicked (2024) is correctly supported",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Wicked (2024) was released in the United States on {info.date_us}.",
        node=leaf,
        sources=urls,
        additional_instruction="Confirm the United States theatrical release date. Accept reasonable date formatting variants (e.g., 'Nov 22, 2024' vs 'November 22, 2024')."
    )


async def build_release_date_checks_2025(evaluator: Evaluator, parent, extracted: WickedAwardsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Wicked_For_Good_2025_Release_Date_US",
        desc="State that Wicked: For Good (2025) US release date is November 21, 2025, with a supporting reference URL.",
        parent=parent,
        critical=True
    )
    info = extracted.wfg_2025_release_us or ReleaseDateClaim()
    urls = _valid_urls(info.sources)

    presence = evaluator.add_custom_node(
        result=_has_nonempty_str(info.date_us) and len(urls) > 0,
        id="Wicked_For_Good_2025_Release_Date_US_presence",
        desc="US release date for Wicked: For Good (2025) is provided with source URL(s)",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Wicked_For_Good_2025_Release_Date_US_supported",
        desc="US release date for Wicked: For Good (2025) is correctly supported",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Wicked: For Good (2025) was released in the United States on {info.date_us}.",
        node=leaf,
        sources=urls,
        additional_instruction="Verify the US release date for the 2025 sequel. Also accept the title 'Wicked: Part Two – For Good' or 'Wicked Part Two' as the same film."
    )


async def build_nominations_checks_2024(evaluator: Evaluator, parent, extracted: WickedAwardsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Wicked_2024_97th_Oscars_Nominations",
        desc="State that Wicked (2024) received 10 nominations at the 97th Academy Awards and that these nominations include Best Picture, Best Actress (Cynthia Erivo), and Best Supporting Actress (Ariana Grande), with a supporting reference URL.",
        parent=parent,
        critical=True
    )
    info = extracted.wicked_2024_97th_noms or NominationsClaim()
    urls = _valid_urls(info.sources)

    presence = evaluator.add_custom_node(
        result=_has_nonempty_str(info.total) and len(urls) > 0,
        id="Wicked_2024_97th_Oscars_Nominations_presence",
        desc="Total nominations count for Wicked (2024) is provided with source URL(s)",
        parent=node,
        critical=True
    )

    required_categories_mentioned = evaluator.add_custom_node(
        result=_categories_contain_required(info.included_categories or []),
        id="Wicked_2024_97th_Oscars_Nominations_required_categories_in_answer",
        desc="Answer mentions the required nominated categories: Best Picture; Best Actress (Cynthia Erivo); Best Supporting Actress (Ariana Grande)",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Wicked_2024_97th_Oscars_Nominations_supported",
        desc="Nominations count and required categories for Wicked (2024) at the 97th Oscars are supported by sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Wicked (2024) received {info.total} nominations at the 97th Academy Awards, including Best Picture, Best Actress (Cynthia Erivo), and Best Supporting Actress (Ariana Grande).",
        node=leaf,
        sources=urls,
        additional_instruction="Verify both the total nomination count and that the specific categories listed are among its nominations. Allow minor naming variants (e.g., capitalization or formatting)."
    )


async def build_wins_total_checks_2024(evaluator: Evaluator, parent, extracted: WickedAwardsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Wicked_2024_97th_Oscars_Wins_Total",
        desc="State that Wicked (2024) won 2 Academy Awards at the 97th Academy Awards, with a supporting reference URL.",
        parent=parent,
        critical=True
    )
    info = extracted.wicked_2024_97th_wins or WinsClaim()
    urls = _valid_urls(info.sources)

    presence = evaluator.add_custom_node(
        result=_has_nonempty_str(info.total) and len(urls) > 0,
        id="Wicked_2024_97th_Oscars_Wins_Total_presence",
        desc="Total wins count for Wicked (2024) is provided with source URL(s)",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Wicked_2024_97th_Oscars_Wins_Total_supported",
        desc="Wins total for Wicked (2024) at the 97th Oscars is supported by sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Wicked (2024) won {info.total} Academy Award(s) at the 97th Academy Awards.",
        node=leaf,
        sources=urls,
        additional_instruction="Verify the total number of wins at the 97th Academy Awards; accept singular/plural variations."
    )


async def build_best_costume_checks_2024(evaluator: Evaluator, parent, extracted: WickedAwardsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Wicked_2024_97th_Oscars_Best_Costume_Design",
        desc="State that Best Costume Design was won by Paul Tazewell and include the constrained fact that this made him the first African American male costume designer to win this award, with supporting reference URL(s).",
        parent=parent,
        critical=True
    )
    info = extracted.wicked_2024_best_costume or CostumeDesignClaim()
    urls = _valid_urls(info.sources)

    presence = evaluator.add_custom_node(
        result=_has_nonempty_str(info.winner) and (info.first_african_american_male is True) and len(urls) > 0,
        id="Wicked_2024_97th_Oscars_Best_Costume_Design_presence",
        desc="Best Costume Design winner and 'first African American male' fact provided with source URL(s)",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Wicked_2024_97th_Oscars_Best_Costume_Design_supported",
        desc="Best Costume Design winner and 'first African American male' fact are supported by sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="For Wicked (2024), Paul Tazewell won the Academy Award for Best Costume Design at the 97th Oscars, and he became the first African American male costume designer to win this award.",
        node=leaf,
        sources=urls,
        additional_instruction="Verify both the winner and the 'first African American male costume designer' milestone are explicitly supported."
    )


async def build_best_production_checks_2024(evaluator: Evaluator, parent, extracted: WickedAwardsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Wicked_2024_97th_Oscars_Best_Production_Design",
        desc="State that Best Production Design was won by Nathan Crowley and Lee Sandales, with a supporting reference URL.",
        parent=parent,
        critical=True
    )
    info = extracted.wicked_2024_best_production or ProductionDesignClaim()
    urls = _valid_urls(info.sources)

    presence = evaluator.add_custom_node(
        result=(len(info.winners) > 0 and _names_in_list(info.winners, ["Nathan Crowley", "Lee Sandales"])) and len(urls) > 0,
        id="Wicked_2024_97th_Oscars_Best_Production_Design_presence",
        desc="Best Production Design winners (Nathan Crowley and Lee Sandales) are provided in the answer with source URL(s)",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Wicked_2024_97th_Oscars_Best_Production_Design_supported",
        desc="Best Production Design winners for Wicked (2024) are supported by sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="For Wicked (2024), the Academy Award for Best Production Design at the 97th Oscars was won by Nathan Crowley and Lee Sandales.",
        node=leaf,
        sources=urls,
        additional_instruction="Verify the named winners. Accept minor formatting or ordering differences."
    )


async def build_wfg_nominations_checks_2026(evaluator: Evaluator, parent, extracted: WickedAwardsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Wicked_For_Good_2025_2026_Oscars_Nominations_Total",
        desc="State that Wicked: For Good (2025) received 0 nominations at the 2026 Academy Awards, with a supporting reference URL.",
        parent=parent,
        critical=True
    )
    info = extracted.wfg_2026_noms or WFGNominationsClaim()
    urls = _valid_urls(info.sources)

    presence = evaluator.add_custom_node(
        result=_has_nonempty_str(info.total) and len(urls) > 0,
        id="Wicked_For_Good_2025_2026_Oscars_Nominations_Total_presence",
        desc="Total nominations for Wicked: For Good (2025) at the 2026 Oscars is provided with source URL(s)",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Wicked_For_Good_2025_2026_Oscars_Nominations_Total_supported",
        desc="Nominations total for Wicked: For Good (2025) at the 2026 Oscars is supported by sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Wicked: For Good (2025) received {info.total} nominations at the 2026 Academy Awards (the 98th Academy Awards).",
        node=leaf,
        sources=urls,
        additional_instruction="Verify the total number of nominations at the 2026/98th Academy Awards; accept '0' and 'zero' equivalently if clearly stated."
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
    Evaluate an answer for the Wicked films director and Academy Awards details task.
    """
    # Initialize evaluator
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
        default_model=model
    )

    # Create a critical main node under root (since root itself is non-critical by design)
    main_node = evaluator.add_parallel(
        id="Wicked_Films_Director_and_Awards",
        desc="Verify all required constrained facts about the director and Academy Awards details for Wicked (2024) and Wicked: For Good (2025), each supported by reference URLs.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_wicked_awards(),
        template_class=WickedAwardsExtraction,
        extraction_name="wicked_awards_extraction"
    )

    # Build verification subtrees
    await build_director_checks(evaluator, main_node, extracted)
    await build_release_date_checks_2024(evaluator, main_node, extracted)
    await build_release_date_checks_2025(evaluator, main_node, extracted)
    await build_nominations_checks_2024(evaluator, main_node, extracted)
    await build_wins_total_checks_2024(evaluator, main_node, extracted)
    await build_best_costume_checks_2024(evaluator, main_node, extracted)
    await build_best_production_checks_2024(evaluator, main_node, extracted)
    await build_wfg_nominations_checks_2026(evaluator, main_node, extracted)

    # Return evaluation summary
    return evaluator.get_summary()