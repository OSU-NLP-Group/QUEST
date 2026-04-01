import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "french_election_2022"
TASK_DESCRIPTION = """
I'm interested in the 2022 French presidential election. Please list all the candidates who participated in the first round. For each candidate, include their name, indicate whether they advanced to the second round, and specify the year or years in which they had previously run in a French presidential election before 2022. If a candidate had not participated in any presidential election prior to 2022, please note this as well.

Additionally, for each candidate, provide a direct Getty Images link to an editorial photograph clearly showing the candidate actively participating in an official 2022 presidential campaign event, such as delivering a speech, addressing supporters, speaking at a podium, or standing in front of campaign signage.
"""

EVAL_NOTES = ""

GROUND_TRUTH = {
    "candidates": [
        {"name": "Nathalie Arthaud", "advanced": False, "previous_elections": ["2012", "2017"]},
        {"name": "Fabien Roussel", "advanced": False, "previous_elections": []},
        {"name": "Emmanuel Macron", "advanced": True, "previous_elections": ["2017"]},
        {"name": "Jean-Luc Mélenchon", "advanced": False, "previous_elections": ["2012", "2017"]},
        {"name": "Jean Lassalle", "advanced": False, "previous_elections": ["2017"]},
        {"name": "Marine Le Pen", "advanced": True, "previous_elections": ["2012", "2017"]},
        {"name": "Éric Zemmour", "advanced": False, "previous_elections": []},
        {"name": "Anne Hidalgo", "advanced": False, "previous_elections": []},
        {"name": "Yannick Jadot", "advanced": False, "previous_elections": ["2017"]},
        {"name": "Valérie Pécresse", "advanced": False, "previous_elections": []},
        {"name": "Philippe Poutou", "advanced": False, "previous_elections": ["2012", "2017"]},
        {"name": "Nicolas Dupont-Aignan", "advanced": False, "previous_elections": ["2012", "2017"]}
    ]
}


class CandidateNames(BaseModel):
    """All candidate names extracted from the answer"""
    names: List[str] = Field(default_factory=list, description="List of all candidate names")


class CandidateDetails(BaseModel):
    """Detailed information for a single candidate"""
    advanced_to_second_round: Optional[bool] = Field(default=None, description="Whether they advanced to second round")
    previous_elections: List[str] = Field(default_factory=list, description="Years of previous presidential elections")


class CandidateRelatedURLs(BaseModel):
    """URLs related to a candidate"""
    urls: List[str] = Field(default_factory=list, description="All URLs associated with this candidate (excluding Getty Images)")


class CandidateGettyLinks(BaseModel):
    """Getty Images links for a candidate"""
    getty_urls: List[str] = Field(default_factory=list, description="All Getty Images URLs for this candidate")


def prompt_extract_candidate_names() -> str:
    """Extract all candidate names from the answer"""
    return """
    Extract the names of ALL French presidential election 2022 first-round candidates mentioned in the answer.

    Return a list of candidate names exactly as they appear in the answer.
    Include every candidate mentioned, in the order they appear.
    """


def prompt_extract_candidate_details(candidate_name: str) -> str:
    """Extract details for a specific candidate"""
    return f"""
    Extract the following information about the candidate "{candidate_name}" from the answer:

    1. advanced_to_second_round: Whether they advanced to the second round (true/false)
    2. previous_elections: List of years when they previously ran for French president (before 2022)
       - Extract years as strings (e.g., "2012", "2017")
       - If they had no previous elections, return an empty list

    Only extract information explicitly stated about this specific candidate.

    NOTE: Allow for common name variations (e.g., "Le Pen" for "Marine Le Pen", "Macron" for "Emmanuel Macron", 
    "Mélenchon" for "Jean-Luc Mélenchon", "JLM" for "Jean-Luc Mélenchon", "MLP" for "Marine Le Pen", etc.)
    and different spellings or accents.
    """


def prompt_extract_candidate_urls(candidate_name: str) -> str:
    """Extract URLs related to a specific candidate"""
    return f"""
    Extract ALL URLs from the answer that are associated with the candidate "{candidate_name}".

    Include URLs that:
    - Provide information about this candidate
    - Support claims about their advancement to second round
    - Support claims about their previous election participations
    - Are explicitly linked to this candidate in the answer

    Exclude:
    - Getty Images URLs (those will be handled separately)
    
    Return all such URLs exactly as they appear in the answer.

    NOTE: Allow for common name variations (e.g., "Le Pen" for "Marine Le Pen", "Macron" for "Emmanuel Macron", 
    "Mélenchon" for "Jean-Luc Mélenchon", "JLM" for "Jean-Luc Mélenchon", "MLP" for "Marine Le Pen", etc.)
    and different spellings or accents.
    """


def prompt_extract_getty_links(candidate_name: str) -> str:
    """Extract Getty Images links for a specific candidate"""
    return f"""
    Extract ALL Getty Images URLs mentioned in the answer that are associated with the candidate "{candidate_name}".

    Include only URLs that:
    - Are explicitly associated with this candidate
    - Appear to be Getty Images links

    Return all such URLs exactly as they appear in the answer.

    NOTE: Allow for common name variations (e.g., "Le Pen" for "Marine Le Pen", "Macron" for "Emmanuel Macron", 
    "Mélenchon" for "Jean-Luc Mélenchon", "JLM" for "Jean-Luc Mélenchon", "MLP" for "Marine Le Pen", etc.)
    and different spellings or accents.
    """


def format_candidate_info_for_verification(name: str, advanced: bool, previous_elections: List[str]) -> str:
    """Format candidate information into a single string for verification"""
    elections_str = ", ".join(previous_elections) if previous_elections else "no previous elections"
    return f"{name} - Advanced to second round: {advanced} - Previous elections: {elections_str}"


def format_ground_truth_candidates() -> List[str]:
    """Format all ground truth candidates into strings for matching"""
    formatted = []
    for candidate in GROUND_TRUTH["candidates"]:
        elections_str = ", ".join(candidate["previous_elections"]) if candidate[
            "previous_elections"] else "no previous elections"
        formatted.append(
            f"{candidate['name']} - Advanced to second round: {candidate['advanced']} - Previous elections: {elections_str}")
    return formatted


async def verify_candidate(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        candidate_name: str,
        candidate_details: CandidateDetails,
        candidate_urls: CandidateRelatedURLs,
        getty_links: CandidateGettyLinks,
        candidate_index: int,
) -> None:
    """Verify a single candidate's information"""

    # Create sequential node for this candidate
    candidate_node = evaluator.add_sequential(
        id=f"candidate_{candidate_index}",
        desc=f"Candidate {candidate_index + 1}: {candidate_name}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial scoring
    )

    # 1. Info node with sub-verifications
    info_node = evaluator.add_parallel(
        id=f"candidate_{candidate_index}_info",
        desc=f"Candidate information verification",
        parent=candidate_node,
        critical=False,
    )

    # 1a. Info provided check
    info_provided_node = evaluator.add_custom_node(
        result=bool(candidate_name and candidate_urls.urls),
        id=f"candidate_{candidate_index}_info_provided",
        desc="Candidate information is provided",
        parent=info_node,
        critical=True,
    )

    # 1b. Match ground truth
    match_gt_node = evaluator.add_leaf(
        id=f"candidate_{candidate_index}_match_gt",
        desc=f"Information matches ground truth",
        parent=info_node,
        critical=True,
    )

    # Format candidate info and ground truth for comparison
    candidate_info_str = format_candidate_info_for_verification(
        candidate_name,
        candidate_details.advanced_to_second_round or False,
        candidate_details.previous_elections
    )

    ground_truth_strings = format_ground_truth_candidates()
    ground_truth_combined = "\n".join(ground_truth_strings)

    await evaluator.verify(
        claim=f"The candidate information '{candidate_info_str}' matches one of the ground truth candidates in this list:\n{ground_truth_combined}",
        node=match_gt_node,
        additional_instruction="Check if the candidate's name, advancement status, and previous election years match any entry in the ground truth list. Allow reasonable name variations."
    )

    # 1c. Advance supported by URLs
    advanced_to_second_round = candidate_details.advanced_to_second_round or False
    advance_supported_node = evaluator.add_leaf(
        id=f"candidate_{candidate_index}_advance_supported",
        desc=f"Advancement claim '{advanced_to_second_round}' is supported by URLs",
        parent=info_node,
        critical=True,
    )

    advancement_claim = f"{candidate_name} {'advanced' if advanced_to_second_round else 'did not advance'} to the second round of the 2022 French presidential election"
    
    await evaluator.verify(
        claim=advancement_claim,
        node=advance_supported_node,
        sources=candidate_urls.urls,
        additional_instruction="Verify if the webpage supports the claim about whether this candidate advanced to the second round."
    )

    # 1d. Previous runs supported by URLs
    previous_runs_verified = True
    for year in candidate_details.previous_elections:
        year_verified = await evaluator.verify(
            claim=f"{candidate_name} participated in the {year} French presidential election",
            node=None,
            sources=candidate_urls.urls,
            additional_instruction=f"Verify if the webpage mentions that {candidate_name} was a candidate in the {year} French presidential election."
        )
        if not year_verified:
            previous_runs_verified = False
            break
    
    previous_runs_node = evaluator.add_custom_node(
        result=previous_runs_verified,
        id=f"candidate_{candidate_index}_previous_runs_supported",
        desc=f"Verify whether previous election participations in '{candidate_details.previous_elections}' are supported",
        parent=info_node,
        critical=True,
    )


    # 2. Getty node verification
    getty_node = evaluator.add_parallel(
        id=f"candidate_{candidate_index}_getty",
        desc=f"Getty Images link verification",
        parent=candidate_node,
        critical=False,
    )

    # 2a. Getty exists
    getty_exists_node = evaluator.add_custom_node(
        result=getty_links.getty_urls and len(getty_links.getty_urls) > 0,
        id=f"candidate_{candidate_index}_getty_exists",
        desc="Getty Images URL is provided",
        parent=getty_node,
        critical=True,
    )

    # if getty_links.getty_urls:
    # 2b. Getty valid
    getty_valid_node = evaluator.add_leaf(
        id=f"candidate_{candidate_index}_getty_valid",
        desc="At least one URL is a valid Getty Images link",
        parent=getty_node,
        critical=True,
    )

    # Check if any URL contains gettyimages
    urls_str = ", ".join(getty_links.getty_urls)
    await evaluator.verify(
        claim=f"At least one of these URLs contains 'gettyimages' domain: {urls_str}",
        node=getty_valid_node,
        additional_instruction="Check if any URL contains 'gettyimages.com' or 'gettyimages' in the domain"
    )

    # 2c. Getty campaign content
    # Verify the content shows campaign event
    getty_content_node = evaluator.add_leaf(
        id=f"candidate_{candidate_index}_getty_campaign",
        desc="Getty image shows candidate at 2022 campaign event",
        parent=getty_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The webpage shows {candidate_name} at a 2022 French presidential campaign event (with campaign-related activities like speech, rally, podium, or campaign signage). Either the image itself or the webpage description should confirm: (1) this is {candidate_name}, and (2) this is from a 2022 campaign event.",
        node=getty_content_node,
        sources=getty_links.getty_urls,  # Use verify_by_urls
        additional_instruction="Verify both: (1) The person matches the candidate name (through image text, facial recognition context, or webpage description), and (2) The image shows campaign activities (speech, rally, podium, campaign signage, etc.) from the 2022 election. Either the visual content OR the text description can provide this evidence."
    )


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
    """Main evaluation function for French Election 2022 task"""

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
        default_model=model,
    )

    # Add ground truth information
    evaluator.add_ground_truth(GROUND_TRUTH, "ground_truth_candidates")

    # Step 1: Extract all candidate names
    candidate_names = await evaluator.extract(
        prompt=prompt_extract_candidate_names(),
        template_class=CandidateNames,
        extraction_name="candidate_names",
    )

    # Filter to first 12 candidates (ground truth length)
    num_candidates = len(GROUND_TRUTH["candidates"])
    selected_candidates = candidate_names.names[:num_candidates]

    # if less than 12, fill with placeholders
    while len(selected_candidates) < num_candidates:
        selected_candidates.append(f"Candidate {len(selected_candidates) + 1} - Missing from answer")

    # Add extraction statistics
    evaluator.add_custom_info({
        "total_candidates_extracted": len(candidate_names.names),
        "candidates_evaluated": len(selected_candidates),
        "expected_candidates": num_candidates,
    }, "extraction_statistics")

    # Step 2: Process each candidate
    for i in range(num_candidates):
        if i < len(selected_candidates):
            candidate_name = selected_candidates[i]

            # Extract candidate details (without URLs)
            candidate_details = await evaluator.extract(
                prompt=prompt_extract_candidate_details(candidate_name),
                template_class=CandidateDetails,
                extraction_name=f"candidate_{i}_details",
            )

            # Extract related URLs for this candidate
            candidate_urls = await evaluator.extract(
                prompt=prompt_extract_candidate_urls(candidate_name),
                template_class=CandidateRelatedURLs,
                extraction_name=f"candidate_{i}_urls",
            )

            # Extract Getty links for this candidate
            getty_links = await evaluator.extract(
                prompt=prompt_extract_getty_links(candidate_name),
                template_class=CandidateGettyLinks,
                extraction_name=f"candidate_{i}_getty_links",
            )

            # Verify this candidate
            await verify_candidate(
                evaluator,
                root,
                candidate_name,
                candidate_details,
                candidate_urls,
                getty_links,
                i
            )

    # Return evaluation summary
    return evaluator.get_summary()