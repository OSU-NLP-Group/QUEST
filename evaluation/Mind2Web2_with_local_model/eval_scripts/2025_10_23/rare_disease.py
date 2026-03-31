import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "rare_disease"
TASK_DESCRIPTION = """
I've heard of a rare genetic disorder involving abnormal skull fusion generally caused by a genetic mutation in the FGFR2 gene, leading to abnormal shape of head and face, with no abnormalities of the hands or feet. Please identify this condition, and provide a webpage link that outlines both the common issues if left untreated and the common treatment options. Then, locate a doctor whose profile explicitly states that they specialize in this specific disease - not just a general term like Craniosynostosis -  and provide a direct link to their profile.
"""

GROUND_TRUTH = "Crouzon Syndrome"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DiseaseInfo(BaseModel):
    """Information about the identified rare genetic disorder"""
    disease_name: Optional[str] = None
    disease_url: Optional[str] = None

class DoctorInfo(BaseModel):
    """Information about the doctor specializing in the disease"""
    doctor_name: Optional[str] = None
    doctor_url: Optional[str] = None

class ProvLinks(BaseModel):
    """URLs provided as evidence in the answer"""
    disease_info_urls: List[str] = Field(default_factory=list)
    doctor_profile_urls: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_disease_info() -> str:
    return """
    Extract the name of the rare genetic disorder identified in the answer and the URL provided for information about this disease.
    
    You should extract:
    1. disease_name: The name of the rare genetic disorder identified in the answer
    2. disease_url: The URL provided in the answer that contains information about the disease, its issues if left untreated, and treatment options
    
    If any information is missing, return null for that field.
    """

def prompt_extract_doctor_info() -> str:
    return """
    Extract information about the doctor who specializes in the specific disease (not just a general term like Craniosynostosis).
    
    You should extract:
    1. doctor_name: The name of the doctor who specializes in the disease
    2. doctor_url: The URL linking to the doctor's profile
    
    If any information is missing, return null for that field.
    """

def prompt_extract_provided_urls() -> str:
    return """
    Extract all URLs provided in the answer, categorizing them into two groups:
    
    1. disease_info_urls: URLs that provide information about the disease, its issues if left untreated, and treatment options
    2. doctor_profile_urls: URLs that link to profiles of doctors who specialize in the disease
    
    Extract each URL exactly as it appears in the answer. If a URL is missing a protocol (http:// or https://), prepend http://. 
    If no URLs are provided for a category, return an empty list for that category.
    """

# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: Any,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the rare_disease task, checking for correct disease identification,
    appropriate information source, and specialist doctor identification.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        agent_name=agent_name,
        answer_name=answer_name,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )
    
    # Extract structured information from the answer
    disease_info = await evaluator.extract(
        prompt=prompt_extract_disease_info(),
        template_class=DiseaseInfo,
        extraction_name="disease_info"
    )
    
    doctor_info = await evaluator.extract(
        prompt=prompt_extract_doctor_info(),
        template_class=DoctorInfo,
        extraction_name="doctor_info"
    )
    
    prov_links = await evaluator.extract(
        prompt=prompt_extract_provided_urls(),
        template_class=ProvLinks,
        extraction_name="prov_links"
    )
    
    # Step 1: Verify disease identification (critical)
    disease_id_node = evaluator.add_leaf(
        id="disease_identification",
        desc="Verify that the identified disease is Crouzon Syndrome",
        critical=True,
    )
    
    if disease_info.disease_name:
        claim = f"The identified disease '{disease_info.disease_name}' is the same as or equivalent to Crouzon Syndrome."
        await evaluator.verify(
            claim=claim,
            node=disease_id_node,
            additional_instruction="Compare the identified disease name to 'Crouzon Syndrome'. They should refer to the same condition. Allow for minor variations in spelling or terminology, but the core identification should be correct."
        )
    else:
        disease_id_node.score = 0.0
        disease_id_node.status = "failed"
    
    # Step 2: Verify disease URL information (non-critical)
    disease_url_parent = evaluator.add_parallel(
        id="disease_url_verification",
        desc="Verify that the provided URL contains information about issues if left untreated and treatment options for Crouzon Syndrome",
        critical=False,
    )
    
    # Check if URL is provided
    url_exists_node = evaluator.add_custom_node(
        result=(disease_info.disease_url is not None and disease_info.disease_url != ""),
        id="disease_url_exists",
        desc="Check that a URL is provided for the disease information",
        parent=disease_url_parent,
        critical=True,
    )
    
    # Create nodes for untreated issues and treatment options
    issues_node = evaluator.add_leaf(
        id="untreated_issues_verification",
        desc="Verify that the URL contains information about issues if Crouzon Syndrome is left untreated",
        parent=disease_url_parent,
        critical=True,
    )
    
    treatment_node = evaluator.add_leaf(
        id="treatment_options_verification",
        desc="Verify that the URL contains information about treatment options for Crouzon Syndrome",
        parent=disease_url_parent,
        critical=True,
    )
    
    # Get all URLs to check
    urls_to_check = []
    if disease_info.disease_url:
        urls_to_check.append(disease_info.disease_url)
    if prov_links.disease_info_urls:
        for url in prov_links.disease_info_urls:
            if url not in urls_to_check:
                urls_to_check.append(url)
    
    # Verify URL contains info about untreated issues
    await evaluator.verify(
        claim="The webpage contains information about the problems, complications, or issues that can occur if Crouzon Syndrome is left untreated.",
        node=issues_node,
        sources=urls_to_check,
        additional_instruction="Look for explicit mentions of complications, problems, or consequences that can happen if the condition is not treated or treated late. This could include information about increased intracranial pressure, vision problems, breathing difficulties, developmental delays, or other health issues."
    )
    
    # Verify URL contains info about treatment options
    await evaluator.verify(
        claim="The webpage contains information about treatment options available for Crouzon Syndrome.",
        node=treatment_node,
        sources=urls_to_check,
        additional_instruction="Look for explicit mentions of treatments, interventions, surgeries, therapies, or management approaches for this condition. This might include information about surgical interventions, timing of treatment, multidisciplinary care, or specific procedures like craniofacial surgeries."
    )
    
    # Step 3: Verify doctor specialization (non-critical)
    doctor_parent = evaluator.add_parallel(
        id="doctor_evaluation",
        desc="Evaluation of doctor specialization in the specific disease",
        critical=False,
    )
    
    # Check if doctor URL is provided
    doctor_url_exists = evaluator.add_custom_node(
        result=(doctor_info.doctor_url is not None and doctor_info.doctor_url != ""),
        id="doctor_url_exists",
        desc="Check that a URL to a doctor's profile is provided",
        parent=doctor_parent,
        critical=True,
    )
    
    # Verify doctor specialization
    specialization_node = evaluator.add_leaf(
        id="doctor_specialization_verification",
        desc="Verify that the doctor explicitly specializes in Crouzon Syndrome, not just Craniosynostosis or general craniofacial disorders",
        parent=doctor_parent,
        critical=True,
    )
    
    # Get all URLs to check
    urls_to_check = []
    if doctor_info.doctor_url:
        urls_to_check.append(doctor_info.doctor_url)
    if prov_links.doctor_profile_urls:
        for url in prov_links.doctor_profile_urls:
            if url not in urls_to_check:
                urls_to_check.append(url)
    
    # Get the disease name to check for
    disease_name = disease_info.disease_name if disease_info.disease_name else "Crouzon Syndrome"
    
    await evaluator.verify(
        claim=f"The doctor's profile explicitly states that they specialize in {disease_name}, not just general terms like Craniosynostosis or craniofacial disorders.",
        node=specialization_node,
        sources=urls_to_check,
        additional_instruction=f"Check if the doctor's profile specifically mentions '{disease_name}' as an area of expertise, interest, or specialization. The mention should be explicit, not just implied by general terms like 'craniosynostosis' or 'craniofacial disorders'. Look for sections like 'Areas of Expertise', 'Specializations', 'Clinical Interests', or similar sections in the doctor's profile."
    )
    
    # Return structured result
    return evaluator.get_summary()