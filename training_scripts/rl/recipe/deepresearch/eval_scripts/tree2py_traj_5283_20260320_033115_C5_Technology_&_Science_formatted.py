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
TASK_ID = "chips_profiles_6b"
TASK_DESCRIPTION = """
Among U.S. semiconductor manufacturing companies that received CHIPS Act direct funding awards of at least $6 billion for commercial fabrication facilities, compile a comprehensive profile for each qualifying company. For each company, provide: (1) the company name, (2) the exact dollar amount of direct CHIPS Act funding awarded (not including loans or tax credits), (3) all U.S. states where the company's CHIPS-funded commercial fabrication facilities are located, (4) the most advanced (smallest) technology node that will be manufactured at these facilities, (5) the production capacity in wafers per month for at least one facility, and (6) a reference URL from an official government or company source.
"""

# Constraint expectations encoded from rubric (used in verification prompts)
EXPECTED = {
    "intel": {
        "name": "Intel Corporation",
        "funding": "up to $7.86 billion",  # Direct funding only (per rubric constraint)
        "states": ["Arizona", "New Mexico", "Ohio", "Oregon"],
        "node": "18A",
        "capacity_hint": "more than 40,000 wafers per month",
    },
    "tsmc": {
        "name": "TSMC Arizona",
        "funding": "up to $6.6 billion",  # Direct funding only (per rubric constraint)
        "states": ["Arizona"],
        "node_acceptable": ["A16", "2nm", "N2", "2-nanometer"],  # Acceptable set
        "capacity_hint": "approximately 20,000 wafers per month",
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CompanyProfile(BaseModel):
    name: Optional[str] = None
    direct_funding_amount: Optional[str] = None  # Keep as string to allow formats like "up to $7.86B"
    facility_states: List[str] = Field(default_factory=list)
    most_advanced_node: Optional[str] = None
    capacity_wafers_per_month: Optional[str] = None  # e.g., "more than 40,000"
    capacity_facility_context: Optional[str] = None  # e.g., "Intel Fab 52 (Arizona)"
    official_urls: List[str] = Field(default_factory=list)  # government/company sources only


class CHIPSProfilesExtraction(BaseModel):
    intel: Optional[CompanyProfile] = None
    tsmc_arizona: Optional[CompanyProfile] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_profiles() -> str:
    return """
    Extract structured CHIPS Act profile information for the following two qualifying companies if present in the answer:
    - Intel Corporation
    - TSMC Arizona

    For each company, extract the following fields exactly as present in the answer:
    1) name: the company name string as stated in the answer (allow variants like "Intel" or "TSMC Arizona Corporation")
    2) direct_funding_amount: the CHIPS Act direct funding award amount stated in the answer (DIRECT FUNDING ONLY; exclude loans/tax credits). Preserve wording such as "up to $X.YZ billion" if used.
    3) facility_states: a list of all U.S. states (strings) where the company's CHIPS-funded commercial fabrication facilities are located, as listed in the answer. Use full state names (e.g., "Arizona", "Ohio").
    4) most_advanced_node: the most advanced (smallest) technology node the answer claims will be manufactured at these facilities (e.g., "18A", "2nm", "A16"). Use the exact token from the answer.
    5) capacity_wafers_per_month: a string capturing the answer's stated production capacity for at least one facility (e.g., "more than 40,000", "~20,000", "approx. 20,000").
    6) capacity_facility_context: the facility context text near the capacity (e.g., "Intel Fab 52 (Arizona)", "first fab").
    7) official_urls: an array of URLs cited in the answer for this company. Include ONLY official government or the company’s own sources (e.g., domains ending in .gov, commerce.gov, whitehouse.gov, or the company domain such as intel.com, tsmc.com, newsroom subdomains, investor relations). Do NOT include third-party news or blogs.

    Return a JSON object with two top-level fields:
    - "intel": a CompanyProfile object for Intel (or null if not present in the answer)
    - "tsmc_arizona": a CompanyProfile object for TSMC Arizona (or null if not present in the answer)

    If any field is missing for a company, set it to null (for strings) or [] (for lists).
    Ensure all URLs are valid (prepend http:// if protocol missing).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(val: Optional[str]) -> str:
    return val if val is not None else ""


def _join_states(states: List[str]) -> str:
    return ", ".join(states) if states else ""


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_intel(evaluator: Evaluator, parent_node, profile: Optional[CompanyProfile]) -> None:
    prof = profile or CompanyProfile()

    # 1) Company name (simple matching with tolerance)
    name_node = evaluator.add_leaf(
        id="intel_name",
        desc="Provides the company name (Intel Corporation).",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided company name refers to Intel Corporation. Extracted name: '{_safe_str(prof.name)}'. "
              f"Treat short variants like 'Intel' as referring to Intel Corporation.",
        node=name_node,
        additional_instruction="Judge True if the extracted name clearly refers to Intel Corporation (allow 'Intel')."
    )

    # 2) Direct funding amount (must be direct-only; match rubric constraint; supported by official sources)
    funding_node = evaluator.add_leaf(
        id="intel_direct_funding_amount_direct_only",
        desc="Provides the direct CHIPS Act funding award amount for Intel (explicitly direct funding only, excluding loans/tax credits) and it matches the constraint value: up to $7.86 billion.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer's stated Intel direct funding amount is '{_safe_str(prof.direct_funding_amount)}'. "
            f"This must equal or closely match 'up to $7.86 billion' and refer specifically to direct funding only "
            f"(excluding loans and tax credits)."
        ),
        node=funding_node,
        sources=prof.official_urls,
        additional_instruction=(
            "Use the provided URLs to confirm the official direct funding amount and that it is direct funding only "
            "(not loans, not tax credits). Mark Correct ONLY IF: "
            "1) the sources support 'up to $7.86 billion' as the direct funding figure for Intel's commercial fabs; and "
            "2) the answer's amount string is consistent with that figure (allow minor formatting differences like 'bn' vs 'billion')."
        )
    )

    # 3) Facility states (must match exactly AZ, NM, OH, OR; supported by official sources)
    states_node = evaluator.add_leaf(
        id="intel_facility_states",
        desc="Lists all U.S. states where Intel's CHIPS-funded commercial fabrication facilities are located, matching the constraint list: Arizona, New Mexico, Ohio, Oregon.",
        parent=parent_node,
        critical=True
    )
    extracted_states = [s.strip() for s in (prof.facility_states or []) if s and isinstance(s, str)]
    await evaluator.verify(
        claim=(
            "Intel's CHIPS-funded commercial fabrication facilities are in the following U.S. states: "
            "Arizona, New Mexico, Ohio, and Oregon — and no other U.S. states are included. "
            f"The answer's extracted states are: [{_join_states(extracted_states)}]."
        ),
        node=states_node,
        sources=prof.official_urls,
        additional_instruction=(
            "Compare the extracted set of states to the exact expected set {Arizona, New Mexico, Ohio, Oregon} "
            "ignoring order and capitalization. Use the official URLs to confirm these locations are indeed covered by "
            "Intel's CHIPS-funded commercial fabs. Mark Correct ONLY IF the sets match exactly."
        )
    )

    # 4) Most advanced node (must be 18A; supported by official sources)
    node_node = evaluator.add_leaf(
        id="intel_most_advanced_node",
        desc="Identifies the most advanced (smallest) technology node to be manufactured at these facilities, matching the constraint value: 18A.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The most advanced (smallest) technology node that will be manufactured at Intel's CHIPS-funded facilities "
            f"is 18A. The answer extracted node is '{_safe_str(prof.most_advanced_node)}'."
        ),
        node=node_node,
        sources=prof.official_urls,
        additional_instruction=(
            "Use official sources to confirm 18A. Mark Correct ONLY IF the extracted node equals '18A' (allow case "
            "insensitivity and minor formatting like 'Intel 18A')."
        )
    )

    # 5) Capacity (at least one facility >40,000 wpm; supported by official sources)
    cap_node = evaluator.add_leaf(
        id="intel_capacity_wafers_per_month",
        desc="Provides production capacity in wafers per month for at least one Intel CHIPS-funded facility (with facility context) and matches the constraint example: Intel Fab 52 (Arizona) produces more than 40,000 wafers per month.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "At least one Intel CHIPS-funded facility has a production capacity greater than 40,000 wafers per month. "
            f"The answer's capacity phrase is '{_safe_str(prof.capacity_wafers_per_month)}' with context "
            f"'{_safe_str(prof.capacity_facility_context)}'."
        ),
        node=cap_node,
        sources=prof.official_urls,
        additional_instruction=(
            "Use the URLs to confirm a capacity threshold > 40,000 wafers/month for at least one relevant Intel fab "
            "(e.g., 'more than 40,000'). Allow approximate phrasings like 'over 40k'. "
            "Mark Correct ONLY IF the sources support > 40,000 wpm and the answer includes a compatible capacity string "
            "with some facility context."
        )
    )

    # 6) Official reference URL (at least one official government or Intel/company source)
    ref_node = evaluator.add_leaf(
        id="intel_official_reference_url",
        desc="Provides at least one supporting reference URL from an official government or Intel/company source.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official source from either a U.S. government (.gov) domain or Intel's official domain(s).",
        node=ref_node,
        sources=prof.official_urls,
        additional_instruction=(
            "Pass if ANY provided URL is clearly official: acceptable government domains include .gov, commerce.gov, "
            "nist.gov, whitehouse.gov, chips.gov; acceptable Intel domains include intel.com and its official "
            "subdomains (e.g., newsroom.intel.com). If no URLs are provided, mark Incorrect."
        )
    )


async def verify_tsmc(evaluator: Evaluator, parent_node, profile: Optional[CompanyProfile]) -> None:
    prof = profile or CompanyProfile()

    # 1) Company name (simple matching with tolerance)
    name_node = evaluator.add_leaf(
        id="tsmc_name",
        desc="Provides the company name (TSMC Arizona).",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided company name refers to TSMC Arizona. Extracted name: '{_safe_str(prof.name)}'. "
              f"Treat variants like 'TSMC Arizona Corporation' or 'TSMC AZ' as referring to TSMC Arizona.",
        node=name_node,
        additional_instruction="Judge True if the extracted name clearly refers to TSMC's Arizona subsidiary/operations."
    )

    # 2) Direct funding amount (must be direct-only; match rubric constraint; supported by official sources)
    funding_node = evaluator.add_leaf(
        id="tsmc_direct_funding_amount_direct_only",
        desc="Provides the direct CHIPS Act funding award amount for TSMC Arizona (explicitly direct funding only, excluding loans/tax credits) and it matches the constraint value: up to $6.6 billion.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer's stated TSMC Arizona direct funding amount is '{_safe_str(prof.direct_funding_amount)}'. "
            f"This must equal or closely match 'up to $6.6 billion' and refer specifically to direct funding only "
            f"(excluding loans and tax credits)."
        ),
        node=funding_node,
        sources=prof.official_urls,
        additional_instruction=(
            "Use the provided URLs to confirm the official direct funding amount and that it is direct funding only "
            "(not loans, not tax credits). Mark Correct ONLY IF: "
            "1) the sources support 'up to $6.6 billion' as the direct funding figure for TSMC Arizona; and "
            "2) the answer's amount string is consistent with that figure (allow minor formatting differences)."
        )
    )

    # 3) Facility states (must be Arizona; supported by official sources)
    states_node = evaluator.add_leaf(
        id="tsmc_facility_states",
        desc="Lists all U.S. states where TSMC Arizona's CHIPS-funded commercial fabrication facilities are located, matching the constraint location: Arizona (Phoenix).",
        parent=parent_node,
        critical=True
    )
    extracted_states = [s.strip() for s in (prof.facility_states or []) if s and isinstance(s, str)]
    await evaluator.verify(
        claim=(
            "TSMC Arizona's CHIPS-funded commercial fabrication facilities are located in Arizona (Phoenix area). "
            "No other U.S. states are included. "
            f"The answer's extracted states are: [{_join_states(extracted_states)}]."
        ),
        node=states_node,
        sources=prof.official_urls,
        additional_instruction=(
            "Compare the extracted set of states to the exact expected set {Arizona} ignoring capitalization. "
            "Use the official URLs to confirm the Arizona location. Mark Correct ONLY IF the set equals {Arizona}."
        )
    )

    # 4) Most advanced node (acceptable if answer names A16 or 2nm as most advanced; supported by official sources)
    node_node = evaluator.add_leaf(
        id="tsmc_most_advanced_node",
        desc="Identifies a single most advanced (smallest) technology node to be manufactured at these facilities; acceptable as consistent with the constraints if the answer names A16 or 2nm as the most advanced node for the CHIPS-funded facilities (does not require stating both).",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The most advanced node for TSMC Arizona's CHIPS-funded fabs, as stated by the answer, is "
            f"'{_safe_str(prof.most_advanced_node)}'. This should be considered correct ONLY if it is one of: "
            f"A16 or 2nm (including synonyms like 'N2' or '2-nanometer')."
        ),
        node=node_node,
        sources=prof.official_urls,
        additional_instruction=(
            "Use official sources to confirm that the Arizona fabs include 2nm (N2) and/or A16. "
            "Mark Correct ONLY IF the extracted node string clearly corresponds to 'A16' or a 2nm-class name "
            "(e.g., '2nm', 'N2', '2-nanometer')."
        )
    )

    # 5) Capacity (first fab ~20,000 wpm; supported by official sources)
    cap_node = evaluator.add_leaf(
        id="tsmc_capacity_wafers_per_month",
        desc="Provides production capacity in wafers per month for at least one TSMC Arizona CHIPS-funded facility (with facility context) and matches the constraint example: the first fab produces approximately 20,000 wafers per month.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "TSMC Arizona's first fab has a production capacity of approximately 20,000 wafers per month. "
            f"The answer's capacity phrase is '{_safe_str(prof.capacity_wafers_per_month)}' with context "
            f"'{_safe_str(prof.capacity_facility_context)}'."
        ),
        node=cap_node,
        sources=prof.official_urls,
        additional_instruction=(
            "Use the URLs to confirm an approximate capacity around 20,000 wafers/month for the first fab. "
            "Allow phrasing such as '~20,000', 'about 20,000', or 'approximately 20,000'. "
            "Mark Correct ONLY IF the sources support ~20,000 wpm and the answer includes a compatible capacity string "
            "with some facility context."
        )
    )

    # 6) Official reference URL (at least one official government or TSMC/company source)
    ref_node = evaluator.add_leaf(
        id="tsmc_official_reference_url",
        desc="Provides at least one supporting reference URL from an official government or TSMC/company source.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official source from either a U.S. government (.gov) domain or TSMC's official domain(s).",
        node=ref_node,
        sources=prof.official_urls,
        additional_instruction=(
            "Pass if ANY provided URL is clearly official: acceptable government domains include .gov, commerce.gov, "
            "nist.gov, whitehouse.gov, chips.gov; acceptable TSMC domains include tsmc.com and official subdomains. "
            "If no URLs are provided, mark Incorrect."
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
    Evaluate an answer for the CHIPS Act ≥$6B commercial fab profiles (Intel, TSMC Arizona).
    """
    # Initialize evaluator (root kept non-critical due to framework constraints on critical parents)
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

    # Record rubric-based expected constraints for transparency
    evaluator.add_ground_truth({
        "expected_intel": {
            "name": EXPECTED["intel"]["name"],
            "direct_funding": EXPECTED["intel"]["funding"],
            "facility_states": EXPECTED["intel"]["states"],
            "most_advanced_node": EXPECTED["intel"]["node"],
            "capacity_example": EXPECTED["intel"]["capacity_hint"],
        },
        "expected_tsmc_arizona": {
            "name": EXPECTED["tsmc"]["name"],
            "direct_funding": EXPECTED["tsmc"]["funding"],
            "facility_states": EXPECTED["tsmc"]["states"],
            "most_advanced_node_acceptable": EXPECTED["tsmc"]["node_acceptable"],
            "capacity_example": EXPECTED["tsmc"]["capacity_hint"],
        }
    }, gt_type="rubric_expectations")

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_profiles(),
        template_class=CHIPSProfilesExtraction,
        extraction_name="chips_profiles_extraction",
    )

    # Build per-item verification subtrees
    intel_node = evaluator.add_parallel(
        id="item_1_intel_profile",
        desc="Profile for qualifying company item: Intel Corporation",
        parent=root,
        critical=False  # Allow partial credit across items
    )
    await verify_intel(evaluator, intel_node, extracted.intel)

    tsmc_node = evaluator.add_parallel(
        id="item_2_tsmc_arizona_profile",
        desc="Profile for qualifying company item: TSMC Arizona",
        parent=root,
        critical=False  # Allow partial credit across items
    )
    await verify_tsmc(evaluator, tsmc_node, extracted.tsmc_arizona)

    # Return structured result
    return evaluator.get_summary()