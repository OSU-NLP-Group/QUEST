import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cvs_ny_minuteclinic_24h"
TASK_DESCRIPTION = """
Find a CVS location in New York, New York that meets all of the following requirements:
1. The location must have a MinuteClinic on-site
2. The MinuteClinic must offer cholesterol screening services
3. The MinuteClinic must accept walk-in patients without requiring appointments
4. The CVS Pharmacy at this location must operate 24 hours a day

Provide the complete street address of one location that satisfies all these requirements, along with reference URLs that verify each requirement.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CVSLocationExtraction(BaseModel):
    """Information extracted from the agent's answer for a single CVS location."""
    address_full: Optional[str] = None

    ny_verification_urls: List[str] = Field(default_factory=list)
    minuteclinic_onsite_urls: List[str] = Field(default_factory=list)
    cholesterol_screening_urls: List[str] = Field(default_factory=list)
    walkin_urls: List[str] = Field(default_factory=list)
    pharmacy_24_hours_urls: List[str] = Field(default_factory=list)
    age_eligibility_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cvs_location() -> str:
    return """
    You will extract the chosen CVS location and all reference URLs provided in the answer.

    Extract the following fields:
    1) address_full: The complete street address of the chosen CVS location as provided in the answer (include street, city, state, and ZIP if present).
    2) ny_verification_urls: All URLs that show the chosen CVS is located in New York, NY (an address listing or official store page that includes city/state).
    3) minuteclinic_onsite_urls: All URLs that verify there is an on-site MinuteClinic at the chosen CVS location (ideally the store or clinic page for that location).
    4) cholesterol_screening_urls: All URLs that verify the MinuteClinic at the chosen location offers cholesterol screening services (ideally the specific clinic/location services page; if the answer cites a general official MinuteClinic services page, include it as well).
    5) walkin_urls: All URLs that verify walk-in patients are accepted without requiring appointments for the chosen location’s MinuteClinic (a location-specific page or an explicit universal MinuteClinic policy statement page).
    6) pharmacy_24_hours_urls: All URLs that verify the CVS Pharmacy at the chosen location operates 24 hours a day (typically the official CVS store page with hours).
    7) age_eligibility_urls: All URLs that verify MinuteClinic services are available for patients 18 months and older (official policy page or location page that states this).

    IMPORTANT URL RULES:
    - Only include URLs explicitly present in the answer text (including markdown links).
    - If the answer does not provide a URL for a requested field, return an empty array for that field.
    - Do not invent or infer URLs.

    Return a single JSON object exactly with these fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_complete_ny_address(address: Optional[str]) -> bool:
    """Heuristic check that the provided address is a complete New York, NY street address."""
    if not address:
        return False
    s = " ".join(address.split())
    has_city_state = bool(
        re.search(r"\bNew\s+York\s*,\s*NY\b", s, flags=re.IGNORECASE) or
        re.search(r"\bNew\s+York\s*,\s*New\s+York\b", s, flags=re.IGNORECASE)
    )
    has_zip = bool(re.search(r"\b\d{5}(?:-\d{4})?\b", s))
    has_street_number = bool(re.search(r"\b\d{1,6}\b", s))  # loose check for a street number
    return has_city_state and has_zip and has_street_number


async def add_constraint_group(
    evaluator: Evaluator,
    parent_node,
    *,
    group_id: str,
    group_desc: str,
    urls: List[str],
    claim: str,
    additional_instruction: str,
    critical: bool = True,
):
    """
    Create a sequential constraint group with:
    - Critical existence check for URL(s)
    - Critical verification leaf grounded by the provided URL(s)
    """
    grp = evaluator.add_sequential(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=critical
    )

    # Existence of at least one verifying URL
    evaluator.add_custom_node(
        result=bool(urls),
        id=f"{group_id}_urls_provided",
        desc=f"{group_desc} — At least one verifying URL is provided in the answer",
        parent=grp,
        critical=True
    )

    # Evidence-grounded verification
    verify_leaf = evaluator.add_leaf(
        id=f"{group_id}_verified",
        desc=group_desc,
        parent=grp,
        critical=True
    )
    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=urls,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extracted: CVSLocationExtraction
) -> None:
    """
    Build the verification tree according to the rubric, including:
    - Provide_Location_Address (critical)
    - Verify_Constraints_With_References (critical, parallel), each with its own existence + verification
    """
    # Root already initialized as SEQUENTIAL in evaluate_answer()

    # 1) Provide_Location_Address (critical existence/format check)
    evaluator.add_custom_node(
        result=is_complete_ny_address(extracted.address_full),
        id="Provide_Location_Address",
        desc="Provide the complete street address for one specific CVS location (street, city, state, ZIP).",
        parent=evaluator.root,
        critical=True
    )

    # 2) Verify_Constraints_With_References (parallel critical)
    constraints_parent = evaluator.add_parallel(
        id="Verify_Constraints_With_References",
        desc="Provide reference URL(s) that verify each required constraint is satisfied for the chosen location.",
        parent=evaluator.root,
        critical=True
    )

    address_text = extracted.address_full or "(no address provided)"

    # 2.a) In_New_York_NY_With_URL
    await add_constraint_group(
        evaluator,
        constraints_parent,
        group_id="In_New_York_NY_With_URL",
        group_desc="Provide a reference URL showing the chosen CVS location is in New York, NY (address listing).",
        urls=extracted.ny_verification_urls,
        claim=f"The referenced page shows that the CVS location at '{address_text}' is located in New York, NY.",
        additional_instruction="Verify the page clearly indicates city='New York' and state='NY' for the listed address. "
                               "Allow minor formatting differences (e.g., punctuation, abbreviations, ZIP+4). Focus on confirming New York, NY."
    )

    # 2.b) MinuteClinic_Onsite_With_URL
    await add_constraint_group(
        evaluator,
        constraints_parent,
        group_id="MinuteClinic_Onsite_With_URL",
        group_desc="Provide a reference URL verifying the chosen CVS location has an on-site MinuteClinic.",
        urls=extracted.minuteclinic_onsite_urls,
        claim=f"The referenced page confirms that the CVS store at '{address_text}' has an on-site MinuteClinic.",
        additional_instruction="Look for explicit mentions such as 'MinuteClinic at CVS' on a store/clinic page tied to this location. "
                               "The evidence should reasonably connect the clinic to the same physical location."
    )

    # 2.c) Cholesterol_Screening_At_Chosen_Clinic_With_URL
    await add_constraint_group(
        evaluator,
        constraints_parent,
        group_id="Cholesterol_Screening_At_Chosen_Clinic_With_URL",
        group_desc="Provide a reference URL verifying the chosen location’s MinuteClinic offers cholesterol screening services (location-specific evidence, e.g., that clinic’s services list).",
        urls=extracted.cholesterol_screening_urls,
        claim="The referenced page shows that the MinuteClinic at the chosen CVS location offers cholesterol screening services.",
        additional_instruction="Prefer a location-specific services list. Accept synonymous service names (e.g., 'cholesterol test', 'lipid panel'). "
                               "If a general official MinuteClinic services page is cited, it must clearly indicate the service is offered."
    )

    # 2.d) WalkIn_No_Appointment_With_URL
    await add_constraint_group(
        evaluator,
        constraints_parent,
        group_id="WalkIn_No_Appointment_With_URL",
        group_desc="Provide a reference URL verifying the chosen location’s MinuteClinic accepts walk-in patients without requiring appointments (location-specific evidence or an explicitly universal MinuteClinic policy statement).",
        urls=extracted.walkin_urls,
        claim="The referenced page states that the MinuteClinic accepts walk-in patients without requiring appointments, "
              "applicable to the chosen location or as an explicit universal policy.",
        additional_instruction="Evidence can be a location page that states walk-ins welcome / no appointment required, "
                               "or an official MinuteClinic policy page that explicitly states this is true for all locations."
    )

    # 2.e) Pharmacy_24_Hours_With_URL
    await add_constraint_group(
        evaluator,
        constraints_parent,
        group_id="Pharmacy_24_Hours_With_URL",
        group_desc="Provide a reference URL verifying the CVS Pharmacy at the chosen location operates 24 hours a day.",
        urls=extracted.pharmacy_24_hours_urls,
        claim="The referenced page confirms that the CVS Pharmacy at the chosen location operates 24 hours a day.",
        additional_instruction="Check the store's official hours page; phrasing like 'Open 24 hours' or a 24/7 schedule for the pharmacy section should be present. "
                               "Ensure the page pertains to the same physical location."
    )

    # 2.f) Age_Eligibility_18_Months_With_URL
    await add_constraint_group(
        evaluator,
        constraints_parent,
        group_id="Age_Eligibility_18_Months_With_URL",
        group_desc="Provide a reference URL verifying MinuteClinic services are available for patients 18 months and older.",
        urls=extracted.age_eligibility_urls,
        claim="The referenced page states that MinuteClinic services are available for patients 18 months and older.",
        additional_instruction="Look for explicit language such as 'we treat patients 18 months and older' on an official MinuteClinic page "
                               "or a location page reflecting the same policy."
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
    Evaluate an agent's answer for the CVS New York MinuteClinic 24h task.
    """
    # Initialize evaluator with a SEQUENTIAL root to enforce order:
    # 1) Provide address, then 2) Verify constraints.
    evaluator = Evaluator()
    evaluator.initialize(
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
        default_model=model,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_cvs_location(),
        template_class=CVSLocationExtraction,
        extraction_name="chosen_cvs_location"
    )

    # Optionally record simple custom info for debugging
    evaluator.add_custom_info(
        info={
            "address_full": extracted.address_full,
            "counts": {
                "ny_verification_urls": len(extracted.ny_verification_urls),
                "minuteclinic_onsite_urls": len(extracted.minuteclinic_onsite_urls),
                "cholesterol_screening_urls": len(extracted.cholesterol_screening_urls),
                "walkin_urls": len(extracted.walkin_urls),
                "pharmacy_24_hours_urls": len(extracted.pharmacy_24_hours_urls),
                "age_eligibility_urls": len(extracted.age_eligibility_urls),
            }
        },
        info_type="extraction_debug",
        info_name="extraction_overview"
    )

    # Build the verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return final structured summary
    return evaluator.get_summary()