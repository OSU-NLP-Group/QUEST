import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "america_the_beautiful_pass_profiles"
TASK_DESCRIPTION = (
    "For each of the following five visitor profiles planning to visit U.S. national parks, "
    "identify the most cost-effective America the Beautiful pass option they qualify for. "
    "For each profile, provide: (1) the specific pass type they are eligible for, (2) the cost of that pass, "
    "and (3) at least one key benefit or feature of that pass.\n\n"
    "Profile 1: A 65-year-old US citizen retired teacher planning to visit multiple national parks throughout the year.\n\n"
    "Profile 2: A US military veteran planning to make annual visits to national parks.\n\n"
    "Profile 3: A US permanent resident with a permanent disability who wants long-term access to national parks.\n\n"
    "Profile 4: A family with a 4th grade student planning to visit Yellowstone National Park this summer.\n\n"
    "Profile 5: A 35-year-old US citizen planning to visit three different national parks over a two-week vacation period."
)

# Profile descriptions used for verification instructions
PROFILE_DESCS: Dict[str, str] = {
    "profile_1": "A 65-year-old US citizen planning to visit multiple national parks throughout a single year.",
    "profile_2": "A US military veteran planning annual visits to national parks.",
    "profile_3": "A US permanent resident with a permanent disability seeking long-term access to national parks.",
    "profile_4": "A family with a 4th grade student planning to visit Yellowstone this summer.",
    "profile_5": "A 35-year-old US citizen visiting three different national parks over a two-week vacation.",
}

# Ground-truth expectations for most cost-effective qualifying pass, cost, and sample key benefits
EXPECTED_PROFILE: Dict[str, Dict[str, Any]] = {
    "profile_1": {
        "pass_name": "Senior Pass (Annual)",
        "cost": "$20",
        "benefits": [
            "Valid for 12 months from purchase",
            "Covers entrance fees at federal recreational lands",
            "Covers pass holder and accompanying passengers in a single private non-commercial vehicle or up to 3 adults at sites with per-person fees",
            "Senior Passes also provide 50% discount on some amenity fees (e.g., camping, boat launching) when applicable",
        ],
        "cost_effectiveness_hint": "For a one-year timeframe, the $20 annual Senior Pass is more cost-effective than the $80 lifetime Senior Pass unless multi-year use is intended.",
    },
    "profile_2": {
        "pass_name": "Military Pass (Lifetime for Veterans & Gold Star Families)",
        "cost": "$0",
        "benefits": [
            "Free lifetime access for U.S. veterans and Gold Star Families",
            "Covers entrance fees at federal recreational lands",
        ],
        "cost_effectiveness_hint": "Veterans qualify for the free lifetime Military Pass, which is more cost-effective than any paid alternative.",
    },
    "profile_3": {
        "pass_name": "Access Pass (Lifetime)",
        "cost": "$0",
        "benefits": [
            "Free lifetime access for U.S. citizens or permanent residents with permanent disabilities",
            "Covers entrance fees at federal recreational lands",
            "Provides 50% discount on some amenity fees (e.g., camping) when applicable",
        ],
        "cost_effectiveness_hint": "The Access Pass is free for qualifying individuals and therefore is the most cost-effective option.",
    },
    "profile_4": {
        "pass_name": "4th Grade Pass (Every Kid Outdoors)",
        "cost": "$0",
        "benefits": [
            "Free entrance for the 4th grade student and accompanying family",
            "Valid for the 4th grade school year",
            "Covers entrance fees at participating federal lands",
        ],
        "cost_effectiveness_hint": "The 4th Grade Pass is free and designed for the student and family, making it the most cost-effective for this profile.",
    },
    "profile_5": {
        "pass_name": "Annual Pass",
        "cost": "$80",
        "benefits": [
            "Valid for one year from the month of purchase",
            "Covers entrance fees at federal recreational lands",
            "Covers pass holder and accompanying passengers in a single private non-commercial vehicle or up to 3 adults at sites with per-person fees",
        ],
        "cost_effectiveness_hint": "Typical park entrance fees are around $30–$35 per park; visiting three parks would likely exceed $80, so the Annual Pass is more cost-effective.",
    },
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProfilePassInfo(BaseModel):
    pass_type: Optional[str] = None
    cost: Optional[str] = None
    benefits: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class PassSelectionExtraction(BaseModel):
    profile_1: Optional[ProfilePassInfo] = None
    profile_2: Optional[ProfilePassInfo] = None
    profile_3: Optional[ProfilePassInfo] = None
    profile_4: Optional[ProfilePassInfo] = None
    profile_5: Optional[ProfilePassInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pass_selection() -> str:
    return (
        "Extract, from the answer, the selected America the Beautiful pass information for each profile. "
        "For every profile (profile_1 through profile_5), return an object with:\n"
        "- pass_type: the specific pass name/type stated (e.g., 'Senior Pass (Annual)', 'Access Pass', '4th Grade Pass', 'Military Pass', 'Annual Pass').\n"
        "- cost: the stated cost for that pass (use the exact wording from the answer, e.g., '$80', '$20', '$0', 'free', 'no cost').\n"
        "- benefits: a list of at least one key benefit or feature mentioned for that pass (e.g., duration/validity, entrance coverage, amenity discounts, who is covered).\n"
        "- sources: any URLs cited in the answer that support the pass details; extract actual URLs if present, including plain links or markdown links. If none, return an empty list.\n\n"
        "If any field is missing for a profile, return null for that field or an empty list for lists.\n"
        "Do not invent information. Only extract what is explicitly stated in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper to build additional instruction for cost-effectiveness verification  #
# --------------------------------------------------------------------------- #
def build_cost_effectiveness_instruction(profile_key: str) -> str:
    exp = EXPECTED_PROFILE[profile_key]
    expected_name = exp["pass_name"]
    hint = exp.get("cost_effectiveness_hint", "")
    synonyms = (
        "Allow reasonable name variants and synonyms, such as 'Interagency Senior Pass', 'America the Beautiful Senior Pass', "
        "'Access Pass', 'Military (Veterans) Lifetime Pass', 'Every Kid Outdoors (4th Grade Pass)', or 'Interagency Annual Pass'. "
        "Minor naming differences should be considered equivalent if they clearly refer to the same pass type."
    )
    return (
        f"Profile: {PROFILE_DESCS[profile_key]}\n"
        f"Expected most cost-effective eligible pass: '{expected_name}'.\n"
        f"Reasoning hint: {hint}\n"
        f"{synonyms}\n"
        "Judge whether the chosen pass type is both eligible for the profile and is the most cost-effective option given the scenario. "
        "If the selected pass is a costlier alternative where a free or cheaper eligible pass exists for this profile, it should be judged incorrect."
    )


def build_cost_instruction(profile_key: str) -> str:
    exp = EXPECTED_PROFILE[profile_key]
    expected_name = exp["pass_name"]
    expected_cost = exp["cost"]
    return (
        f"Verify that the stated cost matches the official/expected cost for '{expected_name}'. "
        f"Expected cost: '{expected_cost}'. Consider minor formatting variations equivalent (e.g., '$0', 'free', 'no cost', '0 dollars'; '$20' vs '20 dollars'). "
        "Focus on whether the numeric value and free/paid status align, not exact punctuation."
    )


def build_benefit_instruction(profile_key: str) -> str:
    exp = EXPECTED_PROFILE[profile_key]
    expected_name = exp["pass_name"]
    expected_benefits = "; ".join(exp["benefits"])
    return (
        f"Verify that the stated benefit is a true key feature of '{expected_name}'. "
        f"Some acceptable key benefits include: {expected_benefits}. "
        "Allow synonymous wording (e.g., 'entrance fees covered', 'valid for one year', '50% amenity discount'). "
        "It is sufficient if the provided benefit matches at least one legitimate feature."
    )


# --------------------------------------------------------------------------- #
# Verification per profile                                                    #
# --------------------------------------------------------------------------- #
async def verify_profile(
    evaluator: Evaluator,
    parent_node,
    profile_key: str,
    profile_extracted: Optional[ProfilePassInfo],
) -> None:
    # Create profile node (parallel aggregation, non-critical to allow partial credit across profiles)
    description = {
        "profile_1": "Profile 1: 65-year-old US citizen visiting multiple national parks throughout the year.",
        "profile_2": "Profile 2: US military veteran planning annual visits.",
        "profile_3": "Profile 3: US permanent resident with permanent disability seeking long-term access.",
        "profile_4": "Profile 4: Family with a 4th grade student visiting Yellowstone this summer.",
        "profile_5": "Profile 5: 35-year-old US citizen visiting three national parks over two weeks.",
    }[profile_key]

    profile_node = evaluator.add_parallel(
        id=profile_key,
        desc=description,
        parent=parent_node,
        critical=False,
    )

    # Safely read extracted fields
    pass_type = profile_extracted.pass_type if profile_extracted else None
    pass_cost = profile_extracted.cost if profile_extracted else None
    benefits = profile_extracted.benefits if (profile_extracted and profile_extracted.benefits) else []
    sources = profile_extracted.sources if (profile_extracted and profile_extracted.sources) else []

    # 1) Most cost-effective qualifying pass
    mce_leaf = evaluator.add_leaf(
        id=f"{profile_key}_most_cost_effective_qualifying_pass",
        desc="Identifies a pass type the visitor qualifies for AND that is the most cost-effective option for this profile, using only the given eligibility and price constraints.",
        parent=profile_node,
        critical=True,
    )
    expected_name = EXPECTED_PROFILE[profile_key]["pass_name"]
    mce_claim = (
        f"For {PROFILE_DESCS[profile_key]}, the answer identifies the pass type '{pass_type}'. "
        f"This pass type should be eligible for the visitor and should be the most cost-effective option. "
        f"The expected most cost-effective eligible pass is '{expected_name}'. "
        "Judge whether the identified pass is equivalent to the expected one and cost-effective for the scenario."
    )
    await evaluator.verify(
        claim=mce_claim,
        node=mce_leaf,
        sources=None,  # Logical check; typically not tied to specific cited page
        additional_instruction=build_cost_effectiveness_instruction(profile_key),
    )

    # 2) Pass cost correctness
    cost_leaf = evaluator.add_leaf(
        id=f"{profile_key}_pass_cost",
        desc="States the correct cost for the identified pass type, consistent with the given constraints.",
        parent=profile_node,
        critical=True,
    )
    expected_cost = EXPECTED_PROFILE[profile_key]["cost"]
    cost_claim = (
        f"The stated cost for the identified pass ('{pass_cost}') matches the expected official cost for '{expected_name}', "
        f"which is '{expected_cost}'. Minor formatting differences like '$0' vs 'free' or '$20' vs '20 dollars' should be considered equivalent."
    )
    await evaluator.verify(
        claim=cost_claim,
        node=cost_leaf,
        sources=sources if sources else None,
        additional_instruction=build_cost_instruction(profile_key),
    )

    # 3) Key benefit or feature correctness (check at least one)
    benefit_leaf = evaluator.add_leaf(
        id=f"{profile_key}_key_benefit_or_feature",
        desc="Provides at least one accurate key benefit/feature of the identified pass, consistent with the given constraints.",
        parent=profile_node,
        critical=True,
    )
    benefit_text = benefits[0] if benefits else ""
    benefit_claim = (
        f"The provided benefit/feature '{benefit_text}' is a legitimate key benefit of '{expected_name}' "
        "for America the Beautiful passes. At least one accurate key benefit must be present."
    )
    await evaluator.verify(
        claim=benefit_claim,
        node=benefit_leaf,
        sources=sources if sources else None,
        additional_instruction=build_benefit_instruction(profile_key),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
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
    Evaluate an answer for the America the Beautiful pass selection task across five profiles.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Profiles are independent; allow partial credit
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

    # Extract pass selections per profile from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_pass_selection(),
        template_class=PassSelectionExtraction,
        extraction_name="pass_selection",
    )

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth(
        {
            "expected_per_profile": EXPECTED_PROFILE,
            "profile_descriptions": PROFILE_DESCS,
        },
        gt_type="expected_pass_info",
    )

    # Build verification subtrees for each profile
    await verify_profile(evaluator, root, "profile_1", extracted.profile_1)
    await verify_profile(evaluator, root, "profile_2", extracted.profile_2)
    await verify_profile(evaluator, root, "profile_3", extracted.profile_3)
    await verify_profile(evaluator, root, "profile_4", extracted.profile_4)
    await verify_profile(evaluator, root, "profile_5", extracted.profile_5)

    # Return structured evaluation summary
    return evaluator.get_summary()