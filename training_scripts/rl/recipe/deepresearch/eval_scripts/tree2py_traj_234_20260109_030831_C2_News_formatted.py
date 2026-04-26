import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "goldsmith_2025_winner"
TASK_DESCRIPTION = """
Identify the winner of the 2025 Goldsmith Prize for Investigative Reporting, which carries a $25,000 prize for the winning entry. Provide the following information: (1) The names of all reporters credited for the winning work, (2) The title of the winning investigation, and (3) All organizations involved in publishing the work (including any news organizations and academic institutions that collaborated on the project). Include reference URLs to support your answer.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WinnerExtraction(BaseModel):
    """
    Structured extraction of the answer's provided information for the 2025 Goldsmith Prize winner.
    """
    award_name: Optional[str] = None
    award_year: Optional[str] = None
    prize_amount: Optional[str] = None

    reporters: List[str] = Field(default_factory=list)
    investigation_title: Optional[str] = None
    organizations: List[str] = Field(default_factory=list)

    award_urls: List[str] = Field(default_factory=list)
    reporters_urls: List[str] = Field(default_factory=list)
    title_urls: List[str] = Field(default_factory=list)
    org_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_winner_info() -> str:
    return """
    Extract the information the answer provides about the 2025 Goldsmith Prize for Investigative Reporting.

    You must return a JSON object with the following fields:
    - award_name: The award name exactly as stated in the answer (e.g., "Goldsmith Prize for Investigative Reporting"). If shorthand like "Goldsmith Prize" is used, extract that string verbatim.
    - award_year: The year indicated for the award (e.g., "2025"). Extract as a string. If not stated, return null.
    - prize_amount: The top prize amount mentioned for the winning entry (e.g., "$25,000", "25k", "USD 25,000"). Extract as a string. If not stated, return null.

    - reporters: An array of all reporter names credited for the winning work, exactly as in the answer. If none are given, return an empty array.
    - investigation_title: The complete title of the winning investigation, exactly as in the answer. If absent, return null.
    - organizations: An array of all organizations involved in publishing the work, including any news organizations and collaborating academic institutions, exactly as in the answer. If none are given, return an empty array.

    - award_urls: An array of URLs that support the award identification and/or the prize amount. Only include actual URLs explicitly present in the answer.
    - reporters_urls: An array of URLs that support the credited reporter names. Only include actual URLs explicitly present in the answer.
    - title_urls: An array of URLs that support the investigation title. Only include actual URLs explicitly present in the answer.
    - org_urls: An array of URLs that support the publishing organizations listed. Only include actual URLs explicitly present in the answer.

    IMPORTANT:
    - Extract only what is explicitly present in the answer. Do not invent or infer any information.
    - For URLs, include full URLs. Accept plain URLs or markdown links, but extract the actual URL targets.
    - If a URL field has no URLs in the answer, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def join_list(items: List[str]) -> str:
    return ", ".join([s.strip() for s in items if s and s.strip()])


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_award_verification(
    evaluator: Evaluator,
    parent_node,
    extracted: WinnerExtraction
) -> None:
    """
    Build and verify the 'Award_Verification' branch (critical, parallel).
    """
    award_node = evaluator.add_parallel(
        id="Award_Verification",
        desc="Correctly verify the award and required award details",
        parent=parent_node,
        critical=True
    )

    # Award name check
    award_name_leaf = evaluator.add_leaf(
        id="Award_Name",
        desc="Identifies the award as the Goldsmith Prize for Investigative Reporting",
        parent=award_node,
        critical=True
    )
    claim_award_name = (
        f"The award identified in the answer is the Goldsmith Prize for Investigative Reporting."
    )
    await evaluator.verify(
        claim=claim_award_name,
        node=award_name_leaf,
        additional_instruction=(
            "Judge based on the answer text. Treat 'Goldsmith Prize' as shorthand for "
            "'Goldsmith Prize for Investigative Reporting' only if the context clearly indicates "
            "the investigative reporting prize administered by the Shorenstein Center at Harvard Kennedy School. "
            "Do not confuse it with other Goldsmith categories."
        )
    )

    # Award year check (must be 2025)
    award_year_leaf = evaluator.add_leaf(
        id="Award_Year_Announced",
        desc="States or otherwise clearly indicates the award announcement year is 2025",
        parent=award_node,
        critical=True
    )
    claim_award_year = "The award announcement year is 2025."
    await evaluator.verify(
        claim=claim_award_year,
        node=award_year_leaf,
        additional_instruction=(
            "Determine whether the answer states or implies that the winner pertains to the year 2025."
        )
    )

    # Prize amount check ($25,000)
    prize_amount_leaf = evaluator.add_leaf(
        id="Prize_Amount",
        desc="States that the winning entry receives a $25,000 prize",
        parent=award_node,
        critical=True
    )
    claim_prize_amount = "The winning entry receives a $25,000 prize."
    await evaluator.verify(
        claim=claim_prize_amount,
        node=prize_amount_leaf,
        additional_instruction=(
            "Verify that the answer states the top prize amount for the winning entry as $25,000. "
            "Consider 'USD 25,000' or '25k' as equivalent to $25,000."
        )
    )


async def build_winner_information(
    evaluator: Evaluator,
    parent_node,
    extracted: WinnerExtraction
) -> None:
    """
    Build and verify the 'Winner_Information' branch (critical, parallel).
    Includes existence gating via custom critical nodes and correctness statements.
    """
    win_node = evaluator.add_parallel(
        id="Winner_Information",
        desc="Provide complete information about the winning work",
        parent=parent_node,
        critical=True
    )

    # Reporters existence gating (critical)
    reporters_exist = evaluator.add_custom_node(
        result=(len(extracted.reporters) > 0),
        id="Reporter_Names_Provided",
        desc="Reporter names are provided in the answer",
        parent=win_node,
        critical=True
    )

    # Reporters correctness statement (critical)
    reporters_leaf = evaluator.add_leaf(
        id="Reporter_Names",
        desc="Provides the names of all reporters credited for the winning work",
        parent=win_node,
        critical=True
    )
    reporters_str = join_list(extracted.reporters)
    claim_reporters = (
        f"The reporters credited for the winning work are: {reporters_str}."
        if reporters_str else "No reporters are listed in the answer."
    )
    await evaluator.verify(
        claim=claim_reporters,
        node=reporters_leaf,
        additional_instruction=(
            "Judge based on the answer text. Consider reasonable name variants (e.g., middle initials, casing)."
        )
    )

    # Title existence gating (critical)
    title_exist = evaluator.add_custom_node(
        result=(extracted.investigation_title is not None and extracted.investigation_title.strip() != ""),
        id="Investigation_Title_Provided",
        desc="Investigation title is provided in the answer",
        parent=win_node,
        critical=True
    )

    # Title correctness statement (critical)
    title_leaf = evaluator.add_leaf(
        id="Investigation_Title",
        desc="Provides the complete title of the winning investigation",
        parent=win_node,
        critical=True
    )
    title_text = extracted.investigation_title or ""
    claim_title = (
        f"The title of the winning investigation is '{title_text}'."
        if title_text else "No investigation title is provided in the answer."
    )
    await evaluator.verify(
        claim=claim_title,
        node=title_leaf,
        additional_instruction=(
            "Judge based on the answer text. Allow minor punctuation or casing differences as equivalent."
        )
    )

    # Organizations existence gating (critical)
    orgs_exist = evaluator.add_custom_node(
        result=(len(extracted.organizations) > 0),
        id="Publishing_Organizations_Provided",
        desc="Publishing organizations are provided in the answer",
        parent=win_node,
        critical=True
    )

    # Organizations correctness statement (critical)
    orgs_leaf = evaluator.add_leaf(
        id="Publishing_Organizations",
        desc="Identifies all organizations involved in publishing the work, including any news organizations and collaborating academic institutions",
        parent=win_node,
        critical=True
    )
    orgs_str = join_list(extracted.organizations)
    claim_orgs = (
        f"The organizations involved in publishing the winning work include: {orgs_str}."
        if orgs_str else "No publishing organizations are listed in the answer."
    )
    await evaluator.verify(
        claim=claim_orgs,
        node=orgs_leaf,
        additional_instruction=(
            "Judge based on the answer text. Include both news organizations and any collaborating academic institutions."
        )
    )


async def build_source_citations(
    evaluator: Evaluator,
    parent_node,
    extracted: WinnerExtraction
) -> None:
    """
    Build and verify the 'Source_Citations' branch (critical, parallel).
    Each leaf requires URLs and verifies that the claim is supported by at least one reputable/official source.
    """
    src_node = evaluator.add_parallel(
        id="Source_Citations",
        desc="All required information is supported by reference URLs from official or reputable sources",
        parent=parent_node,
        critical=True
    )

    # Award and prize supported by URLs
    award_prize_leaf = evaluator.add_leaf(
        id="Cite_Award_And_Prize",
        desc="Provides at least one reference URL from an official or reputable source supporting the award identification and/or prize amount",
        parent=src_node,
        critical=True
    )
    claim_award_prize = (
        "These source URLs support that the award is the Goldsmith Prize for Investigative Reporting "
        "and that the winning entry receives a $25,000 prize."
    )
    await evaluator.verify(
        claim=claim_award_prize,
        node=award_prize_leaf,
        sources=extracted.award_urls,
        additional_instruction=(
            "Consider as supported only if the page clearly states the award identity and the prize amount. "
            "Prefer official sources such as Harvard Kennedy School, Shorenstein Center, or the Goldsmith Awards page. "
            "If multiple URLs are provided, it is acceptable if any one of them explicitly supports both facts."
        )
    )

    # Reporters supported by URLs
    cite_reporters_leaf = evaluator.add_leaf(
        id="Cite_Reporters",
        desc="Provides at least one reference URL from an official or reputable source supporting the credited reporter names",
        parent=src_node,
        critical=True
    )
    reporters_str = join_list(extracted.reporters)
    claim_reporters_supported = (
        f"These source URLs support that the credited reporters for the 2025 Goldsmith Prize winning work are: {reporters_str}."
        if reporters_str else "These source URLs support the credited reporters for the winning work."
    )
    reporters_support_urls = (extracted.reporters_urls or []) + (extracted.award_urls or [])
    await evaluator.verify(
        claim=claim_reporters_supported,
        node=cite_reporters_leaf,
        sources=reporters_support_urls,
        additional_instruction=(
            "Treat as supported only if the page explicitly lists the credited reporters for the winning work. "
            "Official or reputable sources include Harvard Kennedy School/Shorenstein Center pages and the publishing organizations' announcements."
        )
    )

    # Title supported by URLs
    cite_title_leaf = evaluator.add_leaf(
        id="Cite_Title",
        desc="Provides at least one reference URL from an official or reputable source supporting the investigation title",
        parent=src_node,
        critical=True
    )
    title_text = extracted.investigation_title or ""
    claim_title_supported = (
        f"These source URLs support that the title of the winning investigation is '{title_text}'."
        if title_text else "These source URLs support the investigation title for the winning work."
    )
    title_support_urls = (extracted.title_urls or []) + (extracted.award_urls or [])
    await evaluator.verify(
        claim=claim_title_supported,
        node=cite_title_leaf,
        sources=title_support_urls,
        additional_instruction=(
            "Treat as supported only if the page explicitly shows the investigation title associated with the winning work."
        )
    )

    # Organizations supported by URLs
    cite_org_leaf = evaluator.add_leaf(
        id="Cite_Organizations",
        desc="Provides at least one reference URL from an official or reputable source supporting the publishing organizations listed",
        parent=src_node,
        critical=True
    )
    orgs_str = join_list(extracted.organizations)
    claim_orgs_supported = (
        f"These source URLs support that the organizations involved in publishing the winning work include: {orgs_str}."
        if orgs_str else "These source URLs support the organizations involved in publishing the winning work."
    )
    org_support_urls = (extracted.org_urls or []) + (extracted.award_urls or [])
    await evaluator.verify(
        claim=claim_orgs_supported,
        node=cite_org_leaf,
        sources=org_support_urls,
        additional_instruction=(
            "Treat as supported only if the page explicitly identifies the publishing organizations (news orgs and any collaborating academic institutions)."
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
    Evaluate the answer for the 2025 Goldsmith Prize for Investigative Reporting task.
    Builds a verification tree based on the provided rubric and returns a structured summary.
    """
    # Initialize evaluator with a parallel root; add a critical Task_Completion node beneath it.
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_winner_info(),
        template_class=WinnerExtraction,
        extraction_name="winner_extraction"
    )

    # Add ground truth hints (non-binding; for context)
    evaluator.add_ground_truth({
        "expected_award": "Goldsmith Prize for Investigative Reporting",
        "expected_prize_amount": "$25,000",
        "target_year": "2025"
    }, gt_type="expected_facts")

    # Build Task_Completion node as critical parallel aggregator
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify the 2025 Goldsmith Prize for Investigative Reporting winner and provide all required details with supporting URLs",
        parent=root,
        critical=True
    )

    # Sub-branches under Task_Completion
    await build_award_verification(evaluator, task_node, extracted)
    await build_winner_information(evaluator, task_node, extracted)
    await build_source_citations(evaluator, task_node, extracted)

    # Return final summary
    return evaluator.get_summary()