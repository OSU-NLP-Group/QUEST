import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "jpo_palworld_prior_art"
TASK_DESCRIPTION = (
    "In the ongoing patent infringement lawsuit filed by Nintendo Co., Ltd. and The Pokémon Company against "
    "Pocketpair, Inc. (developer of Palworld) in Tokyo District Court on September 18, 2024, the Japan Patent Office "
    "issued a Notice of Reasons for Refusal on October 22, 2025, rejecting one of Nintendo's related patent applications "
    "(No. 2024-031879) for lack of inventive step under Article 29(2) of the Patent Act. The JPO examiner's rejection "
    "notice cited multiple prior art references, with Cited Document 1 being a 2020 YouTube gameplay video of a specific "
    "video game that demonstrated the claimed game mechanics already existed before Nintendo's priority date. Identify "
    "the video game shown in this primary prior art citation (Cited Document 1) and provide: (1) the name of the game's "
    "developer or publisher, and (2) the year this game was originally released."
)

GROUND_TRUTH = {
    "game_name": "ARK: Survival Evolved",
    "developer_publisher": "Studio Wildcard",
    "release_year": "2017",
}


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class PrimaryPriorArtExtraction(BaseModel):
    """
    Structured extraction from the agent's answer.

    Fields:
      - game_name: The game the answer claims is shown in Cited Document 1 (primary prior art video).
      - developer_publisher: The developer or publisher name provided for that game.
      - release_year: The claimed original release year for that game (string to allow flexible formats).
      - cited_doc1_urls: URL(s) to the YouTube video(s) the answer claims is Cited Document 1.
      - jpo_notice_urls: URL(s) that allegedly point to the JPO Notice of Reasons for Refusal or an authoritative mirror/record.
      - lawsuit_urls: URL(s) evidencing the 2024-09-18 Tokyo District Court lawsuit by Nintendo and The Pokémon Company vs Pocketpair.
      - supporting_urls: Any additional URL(s) the answer cites to support developer/publisher or release year claims (e.g., Wikipedia, official sites, press).
    """
    game_name: Optional[str] = None
    developer_publisher: Optional[str] = None
    release_year: Optional[str] = None
    cited_doc1_urls: List[str] = Field(default_factory=list)
    jpo_notice_urls: List[str] = Field(default_factory=list)
    lawsuit_urls: List[str] = Field(default_factory=list)
    supporting_urls: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_primary_prior_art() -> str:
    return """
    Extract the information specifically requested below from the provided answer text. Only extract data explicitly present in the answer.

    Return a single JSON object with these fields:
    - game_name: the name of the game that the answer claims is shown in "Cited Document 1" (the primary prior art YouTube video).
    - developer_publisher: the developer or publisher the answer provides for that game.
    - release_year: the year the answer provides as the game's original release year (string; do not coerce to integer).
    - cited_doc1_urls: an array of the exact URL(s) to the YouTube video(s) the answer claims is "Cited Document 1".
    - jpo_notice_urls: an array of the exact URL(s) for the JPO Notice of Reasons for Refusal (or official mirror/record) concerning application number 2024-031879 and dated October 22, 2025.
    - lawsuit_urls: an array of the exact URL(s) that evidence the patent infringement lawsuit filed by Nintendo and The Pokémon Company against Pocketpair in the Tokyo District Court on September 18, 2024.
    - supporting_urls: an array of any other URL(s) explicitly cited that support the developer/publisher and/or release year claims for the identified game (e.g., Wikipedia, official site, press, stores).

    Special rules for URLs:
    - Extract only URLs explicitly present in the answer (plain or Markdown link). Do not invent or infer URLs.
    - Include full URLs; if protocol missing, prepend http://.
    - If a category has no URLs in the answer, return an empty array.

    If any textual field is not present in the answer, return null for that field.
    """


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
def _non_empty(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _merge_sources(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if isinstance(u, str) and u.strip() and u not in merged:
                merged.append(u)
    return merged


# -----------------------------------------------------------------------------
# Verification logic
# -----------------------------------------------------------------------------
async def build_and_verify_tree(evaluator: Evaluator, info: PrimaryPriorArtExtraction) -> None:
    """
    Construct the verification tree per the rubric and run all checks.
    All nodes under the critical parents are also marked critical to satisfy framework constraints.
    """

    # Critical overall wrapper reflecting the rubric root
    task_node = evaluator.add_sequential(
        id="task_overall",
        desc="Complete task: Identify the video game cited as primary prior art in the JPO rejection and provide its developer/publisher and release year",
        parent=evaluator.root,
        critical=True,
    )

    # 1) Lawsuit Context (critical, sequential)
    lawsuit_node = evaluator.add_sequential(
        id="lawsuit_context",
        desc="Correctly identify the patent infringement lawsuit filed by Nintendo and The Pokémon Company against Pocketpair in Tokyo District Court on September 18, 2024",
        parent=task_node,
        critical=True,
    )

    # Existence gate for lawsuit sources
    evaluator.add_custom_node(
        result=len(_non_empty(info.lawsuit_urls)) > 0,
        id="lawsuit_sources_present",
        desc="At least one lawsuit source URL is provided in the answer",
        parent=lawsuit_node,
        critical=True,
    )

    # Verify lawsuit details against provided sources
    lawsuit_verify_leaf = evaluator.add_leaf(
        id="lawsuit_context_verified",
        desc="Lawsuit context is correctly described (Nintendo + The Pokémon Company vs Pocketpair at Tokyo District Court on 2024-09-18)",
        parent=lawsuit_node,
        critical=True,
    )
    await evaluator.verify(
        claim="On September 18, 2024, Nintendo Co., Ltd. and The Pokémon Company filed a patent infringement lawsuit against Pocketpair, Inc. in the Tokyo District Court.",
        node=lawsuit_verify_leaf,
        sources=_non_empty(info.lawsuit_urls),
        additional_instruction="Check the page for parties (Nintendo, The Pokémon Company, Pocketpair), the Tokyo District Court venue, and the filing date 2024-09-18.",
    )

    # 2) Patent Rejection Event (critical, sequential)
    rejection_node = evaluator.add_sequential(
        id="patent_rejection_event",
        desc="Correctly identify the JPO Notice of Reasons for Refusal rejecting Nintendo's application No. 2024-031879 on 2025-10-22 under Patent Act Art. 29(2) lack of inventive step",
        parent=lawsuit_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_non_empty(info.jpo_notice_urls)) > 0,
        id="jpo_sources_present",
        desc="At least one JPO notice/record source URL is provided in the answer",
        parent=rejection_node,
        critical=True,
    )

    rejection_verify_leaf = evaluator.add_leaf(
        id="jpo_rejection_verified",
        desc="JPO Notice of Reasons for Refusal details are correctly described (App. 2024-031879; 2025-10-22; lack of inventive step under Art. 29(2))",
        parent=rejection_node,
        critical=True,
    )
    await evaluator.verify(
        claim="On October 22, 2025, the Japan Patent Office issued a Notice of Reasons for Refusal rejecting Nintendo's patent application No. 2024-031879 for lack of inventive step under Article 29(2) of the Patent Act.",
        node=rejection_verify_leaf,
        sources=_non_empty(info.jpo_notice_urls),
        additional_instruction="Verify application number 2024-031879, the date 2025-10-22, and the basis 'lack of inventive step' under Article 29(2).",
    )

    # 3) Prior Art Citation (critical, sequential)
    prior_art_node = evaluator.add_sequential(
        id="prior_art_citation",
        desc="Correctly identify that the examiner cited a 2020 YouTube gameplay video as Cited Document 1 (primary prior art)",
        parent=rejection_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_non_empty(info.cited_doc1_urls)) > 0,
        id="cd1_sources_present",
        desc="At least one Cited Document 1 YouTube source URL is provided in the answer",
        parent=prior_art_node,
        critical=True,
    )

    # Verify the cited video is indeed a 2020 YouTube upload
    cd1_2020_leaf = evaluator.add_leaf(
        id="cd1_youtube_2020",
        desc="The Cited Document 1 YouTube video was published in 2020",
        parent=prior_art_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This YouTube video was published in 2020.",
        node=cd1_2020_leaf,
        sources=_non_empty(info.cited_doc1_urls),
        additional_instruction="Check the video's publish date on the YouTube page; allow 'Premiered' labels. The year must be 2020.",
    )

    # Verify the JPO notice describes CD1 as a 2020 YouTube gameplay video (primary prior art)
    cd1_cited_in_notice_leaf = evaluator.add_leaf(
        id="cd1_cited_in_notice",
        desc="The JPO Notice identifies Cited Document 1 as a 2020 YouTube gameplay video (primary prior art)",
        parent=prior_art_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the JPO Notice of Reasons for Refusal for application number 2024-031879, 'Cited Document 1' is a 2020 YouTube gameplay video that the examiner treats as a primary prior art reference.",
        node=cd1_cited_in_notice_leaf,
        sources=_non_empty(info.jpo_notice_urls),
        additional_instruction="Look for the section listing cited documents; confirm that 'Cited Document 1' is a YouTube video with a 2020 publication year and that it demonstrates relevant gameplay mechanics.",
    )

    # 4) Final Answer Components (critical, parallel)
    final_node = evaluator.add_parallel(
        id="final_answer_components",
        desc="Provide all three required pieces of information about the game cited as primary prior art",
        parent=prior_art_node,
        critical=True,
    )

    # 4.1 GameName: Must be ARK: Survival Evolved (grounded by the cited video)
    game_name_leaf = evaluator.add_leaf(
        id="game_name",
        desc="Correctly identify ARK: Survival Evolved as the game shown in the cited YouTube video",
        parent=final_node,
        critical=True,
    )
    # Prefer grounding with the cited video URLs to ensure the identification matches the video content
    await evaluator.verify(
        claim="This cited YouTube video shows gameplay of ARK: Survival Evolved.",
        node=game_name_leaf,
        sources=_non_empty(info.cited_doc1_urls),
        additional_instruction="Verify the video's title/description/screenshot text to confirm it is for ARK: Survival Evolved (allow common abbreviations like 'ARK').",
    )

    # 4.2 Developer/Publisher: Verify the value provided by the answer against sources (should be Studio Wildcard)
    developer_value = info.developer_publisher or ""
    dev_pub_leaf = evaluator.add_leaf(
        id="developer_publisher",
        desc="Correctly identify Studio Wildcard as the developer/publisher of ARK: Survival Evolved",
        parent=final_node,
        critical=True,
    )
    dev_pub_sources = _non_empty(info.supporting_urls)
    if not dev_pub_sources:
        # Fallback to cited video URLs if no dedicated supporting URLs were provided
        dev_pub_sources = _non_empty(info.cited_doc1_urls)
    await evaluator.verify(
        claim=f"The developer or publisher of ARK: Survival Evolved is {developer_value}.",
        node=dev_pub_leaf,
        sources=dev_pub_sources,
        additional_instruction="Use authoritative sources (e.g., official site, Wikipedia, major press). Accept Studio Wildcard as correct. Minor branding variants are acceptable.",
    )

    # 4.3 ReleaseYear: Verify the year provided by the answer (should be 2017)
    release_year_value = info.release_year or ""
    release_year_leaf = evaluator.add_leaf(
        id="release_year",
        desc="Correctly provide 2017 as the release year of ARK: Survival Evolved",
        parent=final_node,
        critical=True,
    )
    year_sources = _non_empty(info.supporting_urls)
    if not year_sources:
        year_sources = _non_empty(info.cited_doc1_urls)
    await evaluator.verify(
        claim=f"ARK: Survival Evolved was originally released in {release_year_value}.",
        node=release_year_leaf,
        sources=year_sources,
        additional_instruction="Interpret 'original release year' as the full official release (leaving Early Access) widely recognized as 2017. If a source distinguishes Early Access (2015) vs 1.0 (2017), use 2017.",
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an agent's answer for the JPO primary prior art identification task.
    Returns a structured summary with the verification tree and scores.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root orchestrates high-level sequential gating
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_primary_prior_art(),
        template_class=PrimaryPriorArtExtraction,
        extraction_name="primary_prior_art_extraction",
    )

    # Record expected ground truth for transparency
    evaluator.add_ground_truth(
        {
            "expected_game_name": GROUND_TRUTH["game_name"],
            "expected_developer_publisher": GROUND_TRUTH["developer_publisher"],
            "expected_release_year": GROUND_TRUTH["release_year"],
        },
        gt_type="expected_facts",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted)

    # Return unified summary
    return evaluator.get_summary()