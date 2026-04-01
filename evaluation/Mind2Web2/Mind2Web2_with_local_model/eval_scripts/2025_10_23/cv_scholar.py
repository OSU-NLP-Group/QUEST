import asyncio
import logging
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.verification_tree import VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cv_scholar"
TASK_DESCRIPTION = """
I've heard that there is an AI researcher ranking based on DBLP data, supported by a German university and the BMBF of Germany. Please help me locate the ranking page. Then, identify the top 10 researchers in the world in Computer Vision by selected h-index, from 1970 to the most recent year available, according to that ranking. For each researcher, please provide their rank, their Google Scholar profile, the title of their most cited paper, and the title of their most recent paper on Google Scholar.
"""

# Ground truth ranking URL
GT_RANKING_URL = "https://airankings.professor-x.de/?country=All&orderby=h_index&venues=computer_vision,CVPR,ICCV,ECCV,ACCV,TPAMI"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #

class ResearcherName(BaseModel):
    """Basic researcher info with just name and rank."""
    name: Optional[str] = None
    rank: Optional[int] = None

class ResearcherNames(BaseModel):
    """List of researcher names from the answer."""
    researchers: List[ResearcherName] = Field(default_factory=list)

class ResearcherProfile(BaseModel):
    """Google Scholar profile information."""
    profile_url: Optional[str] = None

class ResearcherPaper(BaseModel):
    """Paper information."""
    title: Optional[str] = None
    urls: List[str] = Field(default_factory=list)

class RankingPageInfo(BaseModel):
    """Information about the AI rankings page."""
    urls: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #

def prompt_extract_ranking_page() -> str:
    """Prompt to extract information about the AI rankings page."""
    return """
    Extract information about the AI researcher ranking page mentioned in the answer.
    Specifically:
    1. Any URLs mentioned that relate to this ranking system
    
    Only extract URLs that are explicitly mentioned in the answer.
    """

def prompt_extract_researcher_names() -> str:
    """Prompt to extract the names and ranks of researchers mentioned in the answer."""
    return """
    Extract the names and ranks of all Computer Vision researchers mentioned in the answer.
    For each researcher, extract:
    1. Their name
    2. Their rank in the ranking (from 1 to 10)
    
    Return the researchers in the order they appear in the answer.
    """

def prompt_extract_researcher_profile(researcher_name: str) -> str:
    """Prompt to extract Google Scholar profile information for a specific researcher."""
    return f"""
    Extract Google Scholar profile information for the researcher named "{researcher_name}" from the answer.
    Specifically:
    1. Their Google Scholar profile URL
    
    Return null if the URL is not explicitly mentioned in the answer.
    """

def prompt_extract_most_cited_paper(researcher_name: str) -> str:
    """Prompt to extract information about a researcher's most cited paper."""
    return f"""
    Extract information about the most cited paper for the researcher named "{researcher_name}" from the answer.
    Specifically:
    1. The title of their most cited paper
    
    Return null if the title is not explicitly mentioned in the answer.
    """

def prompt_extract_most_recent_paper(researcher_name: str) -> str:
    """Prompt to extract information about a researcher's most recent paper."""
    return f"""
    Extract information about the most recent paper for the researcher named "{researcher_name}" from the answer.
    Specifically:
    1. The title of their most recent paper
    
    Return null if the title is not explicitly mentioned in the answer.
    """

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #

async def verify_ranking_page(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    ranking_info: RankingPageInfo,
) -> None:
    """
    Verify if the answer correctly identifies the AI rankings page.
    """
    ranking_node = evaluator.add_parallel(
        id="ranking_page_verification",
        desc="Verify if the answer correctly identifies the correct AI rankings page",
        parent=parent_node,
        critical=True,
    )
    
    # Check if correct URL is mentioned
    correct_url_found = False
    if ranking_info.urls:
        for url in ranking_info.urls:
            if "airankings.professor-x.de" in url:
                correct_url_found = True
                break
    
    if correct_url_found:
        # If the correct URL is found, just verify it's correct
        evaluator.add_custom_node(
            result=True,
            id="correct_ranking_url",
            desc="The answer mentions the correct airankings.professor-x.de URL",
            parent=ranking_node,
            critical=True
        )
    else:
        # If not found, verify the provided URLs meet our ranking criteria
        if ranking_info.urls:
            ranking_verification_node = evaluator.add_leaf(
                id="correct_ranking_url",
                desc="The answer mentions the correct AI Ressercher ranking URL",
                parent=ranking_node,
                critical=True
            )
            
            await evaluator.verify(
                claim="This ranking system is based on DBLP data, supported by a German university and the BMBF of Germany, and provides AI researcher rankings.",
                node=ranking_verification_node,
                sources=ranking_info.urls,
                additional_instruction="""
                Check if the webpage:
                1. Is a researcher ranking system based on DBLP data
                2. Is supported by a German university
                3. Is supported by BMBF (Federal Ministry of Education and Research of Germany)
                4. Provides rankings of AI/Computer Vision researchers
                """
            )
        else:
            # No URLs provided at all
            evaluator.add_custom_node(
                result=False,
                id="correct_ranking_url",
                desc="No ranking page URL was provided in the answer",
                parent=ranking_node,
                critical=True
            )

async def verify_researcher_info(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    researcher_name: str,
    researcher_rank: Optional[int],
    researcher_profile: ResearcherProfile,
    most_cited_paper: ResearcherPaper,
    most_recent_paper: ResearcherPaper,
    position: int,  # 1-based position in the list
) -> None:
    """
    Verify all information for a specific researcher.
    """
    researcher_name = researcher_name if researcher_name else f"Missing Researcher #{position}"
    researcher_node = evaluator.add_parallel(
        id=f"researcher_{position}_{researcher_name.replace(' ', '_')}",
        desc=f"Verify information for researcher #{position}: {researcher_name}",
        parent=parent_node,
        critical=False,  # Not critical so we can give partial credit
    )
    
    # 1. Check if all required information is provided
    evaluator.add_custom_node(
        result=researcher_name is not None and researcher_name != f"Missing Researcher #{position}" and researcher_rank is not None,
        id=f"info_exists_{position}",
        desc="Check if researcher name is provided",
        parent=researcher_node,
        critical=True
    )

    # 2. Verify rank matches the ground truth
    rank_verification_node = evaluator.add_leaf(
        id=f"rank_verification_{position}",
        desc=f"Verify if {researcher_name} is correctly ranked #{researcher_rank}",
        parent=researcher_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"{researcher_name} is ranked #{researcher_rank}, as shown on the webpage, placing them among the top 10 researchers.",
        node=rank_verification_node,
        sources=GT_RANKING_URL,
        additional_instruction=f"""
        Check if {researcher_name} appears at position #{researcher_rank} in the ranking.
        Allow for minor name variations (e.g., different spellings, middle names).
        """
    )
    
    # 3. Verify Google Scholar profile
    scholar_parent_node = evaluator.add_parallel(
        id=f"scholar_{position}",
        desc=f"Verify if the provided Google Scholar profile URL for {researcher_name} is correct",
        parent=researcher_node,
        critical=False
    )

    profile_exist_node = evaluator.add_custom_node(
        result=researcher_profile.profile_url is not None,
        id=f"profile_exists_{position}",
        desc=f"Check if profile URL is provided for {researcher_name}",
        parent=scholar_parent_node,
        critical=True
    )

    scholar_verification_node = evaluator.add_leaf(
        id=f"scholar_verification_{position}",
        desc=f"Verify if the Google Scholar profile URL is correct for {researcher_name}",
        parent=scholar_parent_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"This is the correct Google Scholar profile for {researcher_name}.",
        node=scholar_verification_node,
        sources=researcher_profile.profile_url,
        additional_instruction=f"""
        Check if this URL leads to a valid Google Scholar profile page for {researcher_name}.
        The name on the profile should match or be very similar to {researcher_name} (allow some common variants in spelling etc).
        """
    )
    
    # 4. Verify most cited paper
    cited_paper_parent = evaluator.add_parallel(
        id=f"most_cited_{position}",
        desc=f"Verify if the most cited paper for {researcher_name} is correctly identified",
        parent=researcher_node,
        critical=False,  # Not critical for overall evaluation
    )

    evaluator.add_custom_node(
        result=most_cited_paper.title is not None,
        id=f"cited_paper_exists_{position}",
        desc=f"Check if most cited paper title is provided for {researcher_name}",
        parent=cited_paper_parent,
        critical=True
    )

    cited_paper_verification_node = evaluator.add_leaf(
        id=f"cited_paper_verification_{position}",
        desc=f"Verify if the most cited paper is correct for {researcher_name}",
        parent=cited_paper_parent,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"'{most_cited_paper.title}' is the most cited paper by {researcher_name}.",
        node=cited_paper_verification_node,
        sources=researcher_profile.profile_url,
        additional_instruction=f"""
        Check if '{most_cited_paper.title}' is indeed the most cited paper by {researcher_name}.
        The paper titles might not match exactly word for word, but they should refer to the same paper.
        Look at the citation counts for each paper and verify this has the highest number of citations.
        """,
        extra_prerequisites=[profile_exist_node, scholar_verification_node]
    )
    
    # 5. Verify most recent paper
    recent_paper_parent = evaluator.add_parallel(
        id=f"most_recent_{position}",
        desc=f"Verify if the most recent paper for {researcher_name} is correctly identified",
        parent=researcher_node,
        critical=False,  # Not critical for overall evaluation
    )

    evaluator.add_custom_node(
        result=most_recent_paper.title is not None,
        id=f"recent_paper_exists_{position}",
        desc=f"Check if most recent paper title is provided for {researcher_name}",
        parent=recent_paper_parent,
        critical=True
    )

    recent_paper_verification_node = evaluator.add_leaf(
        id=f"recent_paper_verification_{position}",
        desc=f"Verify if the most recent paper is correct for {researcher_name}",
        parent=recent_paper_parent,
        critical=True
    )
    
    # Create URL for papers sorted by date
    sorted_papers_url = researcher_profile.profile_url
    if sorted_papers_url and not sorted_papers_url.endswith("&view_op=list_works&sortby=pubdate"):
        if "?" in sorted_papers_url:
            sorted_papers_url += "&view_op=list_works&sortby=pubdate"
        else:
            sorted_papers_url += "?view_op=list_works&sortby=pubdate"
    
    await evaluator.verify(
        claim=f"'{most_recent_paper.title}' is the most recent paper by {researcher_name}.",
        node=recent_paper_verification_node,
        sources=sorted_papers_url,
        additional_instruction=f"""
        Check if '{most_recent_paper.title}' is indeed the most recent paper by {researcher_name}
        according to their Google Scholar profile sorted by publication date.
        The paper titles might not match exactly, but they should refer to the same paper.
        """,
        extra_prerequisites=[profile_exist_node, scholar_verification_node]
    )

# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #

async def evaluate_answer(
    client,  # Using any client compatible with the API
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Create evaluator ---------------------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
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

    # -------- 2. Extract ranking page information ----------------------- #
    ranking_info = await evaluator.extract(
        prompt=prompt_extract_ranking_page(),
        template_class=RankingPageInfo,
        extraction_name="ranking_page"
    )

    # -------- 3. Verify ranking page (critical) ------------------------- #
    await verify_ranking_page(evaluator, root, ranking_info)

    # -------- 4. Extract researcher information ------------------------- #
    researcher_names = await evaluator.extract(
        prompt=prompt_extract_researcher_names(),
        template_class=ResearcherNames,
        extraction_name="researcher_names"
    )

    # -------- 5. Create a non-critical parent for all researchers ------- #
    researchers_parent = evaluator.add_parallel(
        id="all_researchers",
        desc="Verify information for all 10 researchers",
        parent=root,
        critical=False  # Non-critical as requested
    )

    # -------- 6. Extract and verify each researcher (up to 10) ---------- #
    # Ensure we have exactly 10 researcher slots
    while len(researcher_names.researchers) < 10:
        researcher_names.researchers.append(ResearcherName(name=None, rank=None))
    
    # Limit to exactly 10
    researcher_names.researchers = researcher_names.researchers[:10]
    
    # Store extracted info for custom info
    extracted_info = []
    
    for i, researcher in enumerate(researcher_names.researchers):
        position = i + 1  # 1-based position
        
        if not researcher.name:
            # No researcher at this position
            profile_info = ResearcherProfile()
            most_cited_info = ResearcherPaper()
            most_recent_info = ResearcherPaper()
        else:
            # Extract profile information
            profile_info = await evaluator.extract(
                prompt=prompt_extract_researcher_profile(researcher.name),
                template_class=ResearcherProfile,
                extraction_name=f"profile_{researcher.name}"
            )
            
            # Extract most cited paper information
            most_cited_info = await evaluator.extract(
                prompt=prompt_extract_most_cited_paper(researcher.name),
                template_class=ResearcherPaper,
                extraction_name=f"most_cited_{researcher.name}"
            )
            
            # Extract most recent paper information
            most_recent_info = await evaluator.extract(
                prompt=prompt_extract_most_recent_paper(researcher.name),
                template_class=ResearcherPaper,
                extraction_name=f"most_recent_{researcher.name}"
            )
        
        # Store extracted information
        researcher_info = {
            "position": position,
            "name": researcher.name,
            "rank": researcher.rank,
            "google_scholar_url": profile_info.profile_url,
            "most_cited_paper": most_cited_info.title,
            "most_recent_paper": most_recent_info.title
        }
        extracted_info.append(researcher_info)
        
        # Verify this researcher
        await verify_researcher_info(
            evaluator,
            researchers_parent,
            researcher.name,
            researcher.rank,
            profile_info,
            most_cited_info,
            most_recent_info,
            position
        )

    # Add custom info about the extraction details
    evaluator.add_custom_info(
        {
            "ranking_page": ranking_info.dict(),
            "researchers": extracted_info
        },
        "extraction_details"
    )

    # -------- 7. Return structured result ------------------------------- #
    return evaluator.get_summary()