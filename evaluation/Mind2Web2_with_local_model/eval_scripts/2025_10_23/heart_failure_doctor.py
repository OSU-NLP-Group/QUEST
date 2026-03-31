import asyncio
import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "heart_failure_doctor"
TASK_DESCRIPTION = """
I am researching the top hospitals in the US for cardiology care, specifically focused on heart failure treatment. Please help me find top 5 best hospitals for cardiology according to US News ranking. For each hospital, please provide the hospital's official website link, the names and official profile links of three doctors who explicitly specialize in heart failure treatment, as clearly stated in their profiles on the hospital's official website.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Doctor(BaseModel):
    name: Optional[str] = None
    profile_url: Optional[str] = None


class SingleHospital(BaseModel):
    name: Optional[str] = None
    website_url: Optional[str] = None
    doctors: List[Doctor] = Field(default_factory=list)


class RankingURLs(BaseModel):
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hospital_by_rank(rank: int) -> str:
    return f"""
    Extract information about the hospital ranked #{rank} for cardiology according to US News in the answer.
    
    Extract:
    1. The hospital name
    2. The hospital's official website URL
    3. A list of doctors who specialize in heart failure treatment at that hospital, with their names and profile URLs
    
    If the hospital ranked #{rank} is not mentioned in the answer, or if any information is missing, set the corresponding field to null.
    """


def prompt_extract_ranking_urls() -> str:
    return """
    Extract all URLs from the answer that appear to be from US News & World Report and likely contain hospital cardiology rankings.
    These URLs typically contain domains like "usnews.com" or "health.usnews.com" and may contain terms like "best-hospitals", "cardiology", "heart", etc.
    Extract only the URLs themselves, not any surrounding text.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_ranking_source(
    evaluator: Evaluator,
    parent_node,
    ranking_urls: List[str],
) -> None:
    """Verify that the hospitals are from US News rankings."""
    # Create parent node for ranking verification
    ranking_parent = evaluator.add_parallel(
        id="ranking_verification",
        desc="Verify ranking source from US News",
        parent=parent_node,
        critical=True,
    )
    
    # Critical existence check
    ranking_exists = evaluator.add_custom_node(
        result=bool(ranking_urls),
        id="ranking_urls_exist",
        desc="Check if ranking URLs were provided",
        parent=ranking_parent,
        critical=True
    )
    
    # Actual verification
    ranking_node = evaluator.add_leaf(
        id="ranking_source_verification",
        desc="Verify that the hospital rankings are sourced from US News & World Report rankings",
        parent=ranking_parent,
        critical=True,
    )

    claim = "The hospital rankings for cardiology/heart care are sourced from US News & World Report."
    await evaluator.verify(
        claim=claim,
        node=ranking_node,
        sources=ranking_urls,
    )


async def verify_hospital(
    evaluator: Evaluator,
    parent_node,
    hospital_idx: int,
    hospital: SingleHospital,
    ranking_urls: List[str],
) -> None:
    """
    Verify a single hospital and its doctors.
    All verifications are parallel since they're independent requirements.
    """
    hospital_node = evaluator.add_parallel(
        id=f"hospital_{hospital_idx}",
        desc=f"Verification for hospital #{hospital_idx+1}: {hospital.name if hospital.name else 'Missing hospital'}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit across hospitals
    )

    # Hospital information completeness check (combines name and website checks)
    hospital_complete = evaluator.add_custom_node(
        result=(hospital.name is not None and hospital.name.strip() != "" and 
                hospital.website_url is not None and hospital.website_url.strip() != ""),
        id=f"hospital_{hospital_idx}_complete",
        desc=f"Check if hospital #{hospital_idx+1} has both name and website URL",
        parent=hospital_node,
        critical=True
    )

    # Verify hospital is in US News rankings
    ranking_node = evaluator.add_leaf(
        id=f"hospital_{hospital_idx}_ranking",
        desc=f"Verify that {hospital.name if hospital.name else f'hospital #{hospital_idx+1}'} is ranked #{hospital_idx+1} in US News cardiology rankings",
        parent=hospital_node,
        critical=True,
    )

    rank_claim = f"{hospital.name} is ranked {hospital_idx+1} among cardiology/heart hospitals according to US News rankings."
    await evaluator.verify(
        claim=rank_claim,
        node=ranking_node,
        sources=ranking_urls,
    )

    # Verify hospital website is valid
    website_node = evaluator.add_leaf(
        id=f"hospital_{hospital_idx}_website",
        desc=f"Verify that the URL is the official website of {hospital.name if hospital.name else f'hospital #{hospital_idx+1}'}",
        parent=hospital_node,
        critical=False,
    )
    
    await evaluator.verify(
        claim=f"The webpage is the official website of hospital {hospital.name}.",
        node=website_node,
        sources=hospital.website_url,
    )

    # Verify doctors for this hospital
    doctors_node = evaluator.add_parallel(
        id=f"hospital_{hospital_idx}_doctors",
        desc=f"Verify doctors for {hospital.name if hospital.name else f'hospital #{hospital_idx+1}'}",
        parent=hospital_node,
        critical=False,
    )

    # Ensure we have exactly 3 doctors (pad with empty doctors if needed)
    doctors_to_verify = list(hospital.doctors)
    while len(doctors_to_verify) < 3:
        doctors_to_verify.append(Doctor(name=None, profile_url=None))

    # Verify each doctor
    for doctor_idx, doctor in enumerate(doctors_to_verify[:3]):
        await verify_doctor(
            evaluator=evaluator,
            parent_node=doctors_node,
            hospital_name=hospital.name if hospital.name else f"Hospital #{hospital_idx+1}",
            doctor_idx=doctor_idx,
            doctor=doctor,
        )


async def verify_doctor(
    evaluator: Evaluator,
    parent_node,
    hospital_name: str,
    doctor_idx: int,
    doctor: Doctor,
) -> None:
    """
    Verify a single doctor's information.
    Uses parallel verification since profile URL and specialization are independent checks.
    """
    doctor_node = evaluator.add_parallel(
        id=f"doctor_{doctor_idx}",
        desc=f"Doctor #{doctor_idx+1} at {hospital_name}",
        parent=parent_node,
        critical=False,
    )

    # Doctor information completeness check (combines name and profile URL checks)
    doctor_complete = evaluator.add_custom_node(
        result=(doctor.name is not None and doctor.name.strip() != "" and 
                doctor.profile_url is not None and doctor.profile_url.strip() != ""),
        id=f"doctor_{doctor_idx}_complete",
        desc=f"Check if doctor #{doctor_idx+1} has both name and profile URL",
        parent=doctor_node,
        critical=True
    )

    # Verify doctor profile URL is from hospital website
    profile_url_node = evaluator.add_leaf(
        id=f"doctor_{doctor_idx}_profile_url",
        desc=f"Verify profile URL for {doctor.name if doctor.name else f'doctor #{doctor_idx+1}'}",
        parent=doctor_node,
        critical=True,
    )
    
    url_claim = f"The page is the profile page of doctor {doctor.name} on {hospital_name}'s official website domain."
    await evaluator.verify(
        claim=url_claim,
        node=profile_url_node,
        sources=doctor.profile_url,
    )

    # Verify doctor specializes in heart failure
    specialization_node = evaluator.add_leaf(
        id=f"doctor_{doctor_idx}_specialization",
        desc=f"Verify heart failure specialization for {doctor.name if doctor.name else f'doctor #{doctor_idx+1}'}",
        parent=doctor_node,
        critical=True,
    )

    specialization_claim = f"Dr. {doctor.name}'s profile explicitly states specialization in heart failure treatment."
    await evaluator.verify(
        claim=specialization_claim,
        node=specialization_node,
        sources=doctor.profile_url,
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client,
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
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator with sequential strategy for root
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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Extract structured info from the answer ---------------- #
    # Extract each of the top 5 hospitals individually by rank
    hospitals = []
    for rank in range(1, 6):
        hospital_info = await evaluator.extract(
            prompt=prompt_extract_hospital_by_rank(rank),
            template_class=SingleHospital,
            extraction_name=f"hospital_{rank}"
        )
        hospitals.append(hospital_info)
    
    # Extract US News ranking URLs
    ranking_urls_data = await evaluator.extract(
        prompt=prompt_extract_ranking_urls(),
        template_class=RankingURLs,
        extraction_name="ranking_urls"
    )
    ranking_urls = ranking_urls_data.urls

    # -------- 3. Build verification tree -------------------------------- #
    # Verify hospitals are from US News rankings
    await verify_ranking_source(
        evaluator=evaluator,
        parent_node=root,
        ranking_urls=ranking_urls,
    )

    # Verify each hospital
    for hospital_idx, hospital in enumerate(hospitals):
        await verify_hospital(
            evaluator=evaluator,
            parent_node=root,
            hospital_idx=hospital_idx,
            hospital=hospital,
            ranking_urls=ranking_urls,
        )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()