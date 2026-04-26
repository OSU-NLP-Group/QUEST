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
TASK_ID = "sdhs_adoption_trial_veteran_benefit"
TASK_DESCRIPTION = (
    "A veteran living in San Diego County is interested in adopting an adult dog from San Diego Humane Society and "
    "wants to use their Adoption Trial Program before making a final commitment. What are the eligibility requirements "
    "for participating in the Adoption Trial Program, and what adoption fee benefit is available to veterans?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TrialVeteranExtraction(BaseModel):
    # Participant requirements (copy exact phrases from the answer if present)
    age_requirement_text: Optional[str] = None
    transportation_requirement_text: Optional[str] = None
    phone_requirement_text: Optional[str] = None

    # Trial parameters and restrictions (copy exact phrases from the answer if present)
    geographic_restriction_text: Optional[str] = None
    trial_duration_text: Optional[str] = None
    pet_age_requirement_text: Optional[str] = None

    # Source URLs explicitly mentioned in the answer (intended to be sdhumane.org)
    trial_reference_urls: List[str] = Field(default_factory=list)

    # Veteran benefit (copy exact phrase from the answer if present) and supporting URLs
    veteran_benefit_text: Optional[str] = None
    veteran_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trial_veteran() -> str:
    return """
Extract the following information exactly as stated in the answer. Use null for any missing field. Do not infer or invent.

1) Adoption Trial Program – participant eligibility statements (copy the exact phrase if present):
   - age_requirement_text: A sentence/phrase stating that participants must be at least 18 years old (e.g., "You must be 18 or older.")
   - transportation_requirement_text: A sentence/phrase stating that participants must have reliable transportation.
   - phone_requirement_text: A sentence/phrase stating that participants must have a working phone (or active/valid phone number).

2) Adoption Trial Program – trial parameters/restrictions (copy the exact phrase if present):
   - geographic_restriction_text: A sentence/phrase stating pets must remain within San Diego County during the trial.
   - trial_duration_text: A sentence/phrase stating the trial period length (look for "14 days" or "two weeks").
   - pet_age_requirement_text: A sentence/phrase stating the program is for adult pets, defined as over 7 months old for dogs and cats (accept equivalent phrasing like "7+ months" or "7 months and older").

3) Reference URLs (explicit URLs written in the answer):
   - trial_reference_urls: All URLs in the answer that are intended to support the Adoption Trial Program requirements. Only include URLs that belong to the San Diego Humane Society domain (sdhumane.org). If none are provided, return an empty array.
   - veteran_reference_urls: All URLs in the answer that are intended to support the veteran benefit. Only include URLs that belong to the San Diego Humane Society domain (sdhumane.org). If none are provided, return an empty array.

4) Veteran benefit (copy the exact phrase if present):
   - veteran_benefit_text: A sentence/phrase stating the adoption fee benefit for veterans (e.g., "Veterans receive fee‑waived adoptions"). Use null if not present.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _filter_sdhumane_urls(urls: List[str]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if isinstance(u, str) and "sdhumane.org" in u:
            out.append(u.strip())
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for u in out:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: TrialVeteranExtraction):
    """
    Build the verification tree per rubric and run verifications.
    Note: Root node created by evaluator is non-critical; we enforce criticality on children to gate scoring.
    """

    root = evaluator.root

    # Top-level: Adoption Trial Requirements (critical, parallel)
    adoption_trial_node = evaluator.add_parallel(
        id="adoption_trial_requirements",
        desc="Identify all eligibility criteria for participating in the Adoption Trial Program",
        parent=root,
        critical=True
    )

    # Sub: Participant requirements (critical, parallel)
    participant_req_node = evaluator.add_parallel(
        id="participant_requirements",
        desc="State the requirements that the adopter must meet to participate",
        parent=adoption_trial_node,
        critical=True
    )

    # Leaf: Age requirement (critical)
    age_leaf = evaluator.add_leaf(
        id="age_requirement",
        desc="The answer must state that participants must be at least 18 years old",
        parent=participant_req_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that participants must be at least 18 years old (e.g., '18 or older', 'at least 18', 'must be 18+').",
        node=age_leaf,
        additional_instruction="Search the answer text for variants like '18 or older', 'at least 18', 'eighteen years old', or '18+'."
    )

    # Leaf: Transportation requirement (critical)
    transportation_leaf = evaluator.add_leaf(
        id="transportation_requirement",
        desc="The answer must state that participants must have reliable transportation",
        parent=participant_req_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that participants must have reliable transportation.",
        node=transportation_leaf,
        additional_instruction="Accept phrasing variants like 'reliable transportation', 'dependable transportation', or 'access to reliable transportation'."
    )

    # Leaf: Phone requirement (critical)
    phone_leaf = evaluator.add_leaf(
        id="phone_requirement",
        desc="The answer must state that participants must have a working phone",
        parent=participant_req_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that participants must have a working phone.",
        node=phone_leaf,
        additional_instruction="Accept variants like 'working phone', 'active phone number', 'valid phone number', or 'reachable by phone'."
    )

    # Sub: Trial parameters (critical, parallel)
    trial_params_node = evaluator.add_parallel(
        id="trial_parameters",
        desc="State the requirements and restrictions that apply to the trial period and pets",
        parent=adoption_trial_node,
        critical=True
    )

    # Leaf: Geographic restriction (critical)
    geo_leaf = evaluator.add_leaf(
        id="geographic_restriction",
        desc="The answer must state that pets must remain within San Diego County during the trial period",
        parent=trial_params_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that pets must remain within San Diego County during the trial period.",
        node=geo_leaf,
        additional_instruction="Accept variants like 'must stay within San Diego County', 'must remain in San Diego County', or 'cannot leave San Diego County' during the trial."
    )

    # Leaf: Trial duration (critical)
    duration_leaf = evaluator.add_leaf(
        id="trial_duration",
        desc="The answer must state that the trial period is 14 days",
        parent=trial_params_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the trial period is 14 days.",
        node=duration_leaf,
        additional_instruction="Also accept 'two weeks' as equivalent to 14 days."
    )

    # Leaf: Pet age requirement (critical)
    pet_age_leaf = evaluator.add_leaf(
        id="pet_age_requirement",
        desc="The answer must state that the program is available for adult pets (over 7 months old for dogs and cats)",
        parent=trial_params_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the Adoption Trial Program is available for adult pets, defined as over 7 months old for dogs and cats.",
        node=pet_age_leaf,
        additional_instruction="Accept phrasing like 'adult pets 7+ months', 'adult cats and dogs (7 months and older)', or clearly equivalent wording."
    )

    # Leaf: Reference URL from sdhumane.org for the trial program (critical)
    trial_urls = _filter_sdhumane_urls(extracted.trial_reference_urls or [])
    if len(trial_urls) == 0:
        evaluator.add_custom_node(
            result=False,
            id="reference_url_trial",
            desc="Provide a valid reference URL from sdhumane.org that confirms the Adoption Trial Program requirements",
            parent=adoption_trial_node,
            critical=True
        )
    else:
        ref_trial_leaf = evaluator.add_leaf(
            id="reference_url_trial",
            desc="Provide a valid reference URL from sdhumane.org that confirms the Adoption Trial Program requirements",
            parent=adoption_trial_node,
            critical=True
        )
        claim_trial = (
            "This page on sdhumane.org explains San Diego Humane Society's Adoption Trial Program and confirms its "
            "eligibility requirements and parameters, including minimum age (18+), having reliable transportation, "
            "having a working phone, a 14-day trial period (two weeks), pets must remain within San Diego County, "
            "and that the program is for adult pets (over 7 months old for dogs and cats)."
        )
        await evaluator.verify(
            claim=claim_trial,
            node=ref_trial_leaf,
            sources=trial_urls,
            additional_instruction=(
                "Verify that the URL is on the sdhumane.org domain and that the page content confirms the Adoption Trial "
                "Program and its requirements/parameters. Allow minor wording differences (e.g., 'two weeks' for 14 days; "
                "'valid/active phone number' for 'working phone'). If the URL is not on sdhumane.org or does not discuss the "
                "Adoption Trial Program requirements, mark as not supported."
            )
        )

    # Top-level: Veteran Benefit (critical, parallel)
    veteran_node = evaluator.add_parallel(
        id="veteran_benefit",
        desc="Identify the adoption fee benefit available to veterans",
        parent=root,
        critical=True
    )

    # Leaf: Fee waived statement (critical)
    fee_waived_leaf = evaluator.add_leaf(
        id="fee_waived",
        desc="The answer must state that veterans receive fee-waived adoptions",
        parent=veteran_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that veterans receive fee-waived adoptions.",
        node=fee_waived_leaf,
        additional_instruction="Accept variants like 'adoption fees are waived for veterans', 'no adoption fee for veterans', or 'free adoptions for veterans'."
    )

    # Leaf: Reference URL for veteran benefit from sdhumane.org (critical)
    veteran_urls = _filter_sdhumane_urls(extracted.veteran_reference_urls or [])
    if len(veteran_urls) == 0:
        evaluator.add_custom_node(
            result=False,
            id="reference_url_veteran",
            desc="Provide a valid reference URL from sdhumane.org that confirms the veteran benefit",
            parent=veteran_node,
            critical=True
        )
    else:
        ref_veteran_leaf = evaluator.add_leaf(
            id="reference_url_veteran",
            desc="Provide a valid reference URL from sdhumane.org that confirms the veteran benefit",
            parent=veteran_node,
            critical=True
        )
        claim_veteran = (
            "This page on sdhumane.org confirms that veterans receive fee-waived adoptions."
        )
        await evaluator.verify(
            claim=claim_veteran,
            node=ref_veteran_leaf,
            sources=veteran_urls,
            additional_instruction=(
                "Verify the URL is on sdhumane.org and that the content clearly indicates fee-waived adoptions for veterans "
                "(accept wording like 'adoption fees waived for veterans' or 'free adoptions for veterans', possibly with valid ID)."
            )
        )

    # Record some custom info for transparency
    evaluator.add_custom_info(
        {
            "extracted_trial_urls_raw": extracted.trial_reference_urls,
            "extracted_trial_urls_filtered_sdhumane": trial_urls,
            "extracted_veteran_urls_raw": extracted.veteran_reference_urls,
            "extracted_veteran_urls_filtered_sdhumane": veteran_urls,
        },
        info_type="extraction_debug",
        info_name="url_extraction_debug"
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
    Evaluate an answer for San Diego Humane Society's Adoption Trial Program eligibility and veteran fee benefit.
    """
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_trial_veteran(),
        template_class=TrialVeteranExtraction,
        extraction_name="trial_veteran_extraction"
    )

    # Build tree and verify
    await build_and_verify_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()