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
TASK_ID = "coaching_lineage_mcvay"
TASK_DESCRIPTION = (
    "Trace the coaching mentorship lineage of Sean McVay, the current head coach of the Los Angeles Rams, by identifying "
    "the head coaches under whom he and his coaching mentors served as assistant coaches in the NFL. Starting with Sean McVay, "
    "go back four generations through this coaching lineage.\n\n"
    "For each of the four generations, provide:\n"
    "1. The name of the head coach who served as the mentor\n"
    "2. The NFL team where this coaching relationship occurred\n"
    "3. The year(s) or time period when they coached together\n"
    "4. The position or role of the assistant coach during that time (if clearly documented)\n"
    "5. A reference URL that verifies this coaching relationship\n\n"
    "The four generations to trace are:\n"
    "- Generation 1: The head coach under whom Sean McVay first coached in the NFL\n"
    "- Generation 2: The head coach under whom the Generation 1 coach served as an NFL assistant\n"
    "- Generation 3: The head coach under whom the Generation 2 coach served as an NFL assistant\n"
    "- Generation 4: The head coach under whom the Generation 3 coach served as an assistant\n\n"
    "Each generation must be correctly identified before proceeding to the next, as each step depends on the previous identification."
)

# Ground truth / expected lineage checkpoints
STARTING_NAME = "Sean McVay"
STARTING_TEAM = "Los Angeles Rams"

GEN_EXPECTATIONS = {
    1: {
        "assistant": "Sean McVay",
        "mentor": "Jon Gruden",
        "team": "Tampa Bay Buccaneers",
        "period": "2008",
        "role_required": True,
        "allowed_roles": ["assistant wide receivers coach", "offensive assistant"],
    },
    2: {
        "assistant": "Jon Gruden",
        "mentor": "Mike Holmgren",
        "team": "Green Bay Packers",
        "period": "1992–1994",
        "role_required": False,
        "allowed_roles": [],  # optional role
    },
    3: {
        "assistant": "Mike Holmgren",
        "mentor": "Bill Walsh",
        "team": "San Francisco 49ers",
        "period": "1986–1988",
        "role_required": True,
        "allowed_roles": ["quarterbacks coach", "QB coach"],
    },
    4: {
        "assistant": "Bill Walsh",
        "mentor": "Paul Brown",
        "team": "Cincinnati Bengals",
        "period": "1968–1975",
        "role_required": False,
        "allowed_roles": [],  # optional role
    },
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GenerationInfo(BaseModel):
    mentor_head_coach: Optional[str] = None
    team: Optional[str] = None
    time_period: Optional[str] = None
    assistant_role: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CoachingLineageExtraction(BaseModel):
    starting_person: Optional[str] = None
    starting_person_team: Optional[str] = None
    starting_person_role: Optional[str] = None  # e.g., "current head coach"
    gen1: Optional[GenerationInfo] = None
    gen2: Optional[GenerationInfo] = None
    gen3: Optional[GenerationInfo] = None
    gen4: Optional[GenerationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coaching_lineage() -> str:
    return """
    Extract the coaching lineage information from the answer text.

    You must return a JSON object with these fields:
    - starting_person: The name used in the answer for the lineage starting point (should be Sean McVay if present)
    - starting_person_team: The NFL team mentioned for the starting person (e.g., "Los Angeles Rams") as stated in the answer
    - starting_person_role: The role or status of the starting person as stated (e.g., "current head coach")
    - gen1: Object containing fields for Generation 1 (Sean McVay under an NFL head coach)
      - mentor_head_coach: The head coach name
      - team: The NFL team for this relationship
      - time_period: The year or range (e.g., "2008", or "1992–1994")
      - assistant_role: The assistant's role/title if explicitly stated; otherwise null
      - reference_urls: All URLs in the answer that verify this coaching relationship (array). Extract exactly as URLs appear; include full http(s) URLs.
    - gen2: Object for Generation 2 (the Gen1 coach under a head coach) with the same five fields
    - gen3: Object for Generation 3 (the Gen2 coach under a head coach) with the same five fields
    - gen4: Object for Generation 4 (the Gen3 coach under a head coach) with the same five fields

    Rules:
    - Do not invent any data not explicitly present in the answer.
    - If a field is missing in the answer, set it to null (or [] for the URLs list).
    - For reference_urls, extract only actual URLs (plain or markdown links). Include all relevant URLs the answer provided for that generation's verification.
    - Preserve the exact strings found in the answer (names, team names, roles, time ranges).
    """


# --------------------------------------------------------------------------- #
# Helper: build relationship verification claim                               #
# --------------------------------------------------------------------------- #
def build_relationship_claim(
    assistant_name: str,
    mentor_name: str,
    team: str,
    period: str,
    role_required: bool,
    allowed_roles: List[str],
    extracted_role: Optional[str],
    generation_index: int,
) -> str:
    """
    Build a concise claim describing the assistant-under-mentor relationship for URL verification.
    """
    base = (
        f"For Generation {generation_index}, {assistant_name} served on head coach {mentor_name}'s staff "
        f"with the {team} during {period}."
    )

    if role_required:
        if allowed_roles:
            allowed_hint = " or ".join(allowed_roles)
            if extracted_role:
                return (
                    base
                    + f" The assistant role in the answer ('{extracted_role}') is equivalent to {allowed_hint}."
                )
            else:
                return base + f" The role is documented as {allowed_hint} by reliable sources."
        else:
            return base + " The assistant role is clearly stated in the answer."
    else:
        # Role is optional; don't require it in the claim
        return base


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_starting_point(evaluator: Evaluator, parent_node, data: CoachingLineageExtraction) -> None:
    """
    Verify that the answer explicitly starts with Sean McVay and identifies him as
    the current head coach of the Los Angeles Rams.
    """
    start_leaf = evaluator.add_leaf(
        id="Starting_Point",
        desc="The answer explicitly starts the lineage from Sean McVay and identifies him as the current head coach of the Los Angeles Rams.",
        parent=parent_node,
        critical=True,
    )
    claim = (
        "The answer explicitly identifies Sean McVay as the starting point and also states he is the "
        "current head coach of the Los Angeles Rams (LA Rams)."
    )
    await evaluator.verify(
        claim=claim,
        node=start_leaf,
        additional_instruction=(
            "Accept synonyms like 'LA Rams' for 'Los Angeles Rams'. "
            "Focus on whether the answer clearly starts from Sean McVay and attributes the 'current head coach' role."
        ),
    )


async def verify_generation_required(
    evaluator: Evaluator,
    parent_node,
    generation_index: int,
    gen_info: Optional[GenerationInfo],
    assistant_name: str,
    mentor_expected: str,
    team_expected: str,
    period_expected: str,
    role_required: bool,
    allowed_roles: List[str],
) -> None:
    """
    Build the required verification node for a generation:
    - Mentor head coach matches expectation
    - Team matches expectation
    - Time period matches expectation
    - Assistant role (if required)
    - Reference URL(s) support the relationship (primary URL check)
    """
    gen_node = evaluator.add_parallel(
        id=f"Generation_{generation_index}",
        desc=f"Generation {generation_index}: Required checks (mentor, team, period, role-if-required, and source verification).",
        parent=parent_node,
        critical=True,  # Required checks are critical for progression
    )

    # Mentor head coach check
    mentor_leaf = evaluator.add_leaf(
        id=f"Gen{generation_index}_Mentor_Head_Coach",
        desc=f"Identify {mentor_expected} as the Generation {generation_index} mentor head coach (per constraints).",
        parent=gen_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"For Generation {generation_index}, the mentor head coach is {mentor_expected}."
        ),
        node=mentor_leaf,
        additional_instruction=(
            "Judge only based on the answer text: it should clearly identify the mentor for this generation by this name. "
            "Allow minor name variants (letter casing, middle initials)."
        ),
    )

    # Team check
    team_leaf = evaluator.add_leaf(
        id=f"Gen{generation_index}_Team",
        desc=f"Identify {team_expected} as the team where the Generation {generation_index} coaching relationship occurred.",
        parent=gen_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"For Generation {generation_index}, the team for the mentoring relationship is {team_expected}."
        ),
        node=team_leaf,
        additional_instruction=(
            "Accept reasonable team name variants or abbreviations (e.g., 'TB Buccaneers' for 'Tampa Bay Buccaneers', "
            "'49ers' for 'San Francisco 49ers')."
        ),
    )

    # Time period check
    period_leaf = evaluator.add_leaf(
        id=f"Gen{generation_index}_Time_Period",
        desc=f"Identify {period_expected} as the time period of the Generation {generation_index} coaching relationship.",
        parent=gen_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"For Generation {generation_index}, the time period is {period_expected}."
        ),
        node=period_leaf,
        additional_instruction=(
            "Allow minor formatting variants for ranges: '–'/'-' or 'to', and season notation (e.g., '2008 season')."
        ),
    )

    # Assistant role (if required)
    if role_required:
        role_leaf = evaluator.add_leaf(
            id=f"Gen{generation_index}_Assistant_Role",
            desc=(
                f"State {assistant_name}'s assistant role for Generation {generation_index} as required "
                f"(expected one of: {', '.join(allowed_roles) if allowed_roles else 'role stated'})."
            ),
            parent=gen_node,
            critical=True,
        )
        if allowed_roles:
            # If we have specific allowed roles, compare equivalence
            role_val = gen_info.assistant_role if gen_info else None
            if role_val and isinstance(role_val, str):
                claim_role = (
                    f"The assistant role provided in the answer for Generation {generation_index} "
                    f"('{role_val}') is equivalent to "
                    f"{' or '.join(allowed_roles)}."
                )
            else:
                claim_role = (
                    f"For Generation {generation_index}, the answer clearly indicates an assistant role equivalent to "
                    f"{' or '.join(allowed_roles)}."
                )
        else:
            # Generic role presence
            claim_role = (
                f"For Generation {generation_index}, the answer clearly states the assistant role for {assistant_name}."
            )
        await evaluator.verify(
            claim=claim_role,
            node=role_leaf,
            additional_instruction=(
                "Accept minor synonyms and abbreviations (e.g., 'WR' for 'wide receivers', 'QB' for 'quarterbacks'). "
                "Focus on whether the role communicated aligns with the expected role(s)."
            ),
        )

    # Reference URL verification (primary evidence check)
    url_leaf = evaluator.add_leaf(
        id=f"Gen{generation_index}_Reference_URL",
        desc=(
            f"Provide a valid reference URL verifying the Generation {generation_index} coaching relationship "
            f"({assistant_name} under {mentor_expected} at {team_expected} in {period_expected})."
        ),
        parent=gen_node,
        critical=True,
    )
    urls = gen_info.reference_urls if (gen_info and gen_info.reference_urls) else []
    extracted_role = gen_info.assistant_role if gen_info else None
    relationship_claim = build_relationship_claim(
        assistant_name=assistant_name,
        mentor_name=mentor_expected,
        team=team_expected,
        period=period_expected,
        role_required=role_required,
        allowed_roles=allowed_roles,
        extracted_role=extracted_role,
        generation_index=generation_index,
    )

    await evaluator.verify(
        claim=relationship_claim,
        node=url_leaf,
        sources=urls,
        additional_instruction=(
            "Judge strictly by the provided URL(s). Confirm the assistant served under the specified head coach, "
            "for the specified team, during the specified period. Accept minor role title variants if the relationship "
            "is otherwise clearly supported. If the URLs are missing or irrelevant, mark as not supported."
        ),
    )


async def verify_optional_role(
    evaluator: Evaluator,
    parent_node,
    generation_index: int,
    gen_info: Optional[GenerationInfo],
    assistant_name: str,
) -> None:
    """
    Optional role verification for generations where the role is not strictly required:
    Pass if either (a) a role is explicitly provided, or (b) the answer explicitly notes it is not clearly documented.
    """
    opt_leaf = evaluator.add_leaf(
        id=f"Gen{generation_index}_Assistant_Role_If_Documented",
        desc=(
            f"Provide the Generation {generation_index} assistant role for {assistant_name} if clearly documented, "
            f"or explicitly note it is not clearly documented."
        ),
        parent=parent_node,
        critical=False,
    )

    # Determine which claim to verify based on extraction presence
    role_val = (gen_info.assistant_role if gen_info else None) or ""
    role_val = role_val.strip()

    if role_val:
        claim = (
            f"For Generation {generation_index}, the answer documents the assistant role for {assistant_name} as '{role_val}'."
        )
        add_ins = (
            "Verify that the answer clearly states this role string (allowing minor wording variants)."
        )
    else:
        claim = (
            f"For Generation {generation_index}, the answer explicitly notes that the assistant role for {assistant_name} "
            f"is not clearly documented (or equivalent wording indicating uncertainty/unavailability)."
        )
        add_ins = (
            "Pass if the answer explicitly communicates that the role is not clearly documented or unavailable. "
            "Fail if the answer is silent and does not mention the role nor its lack of documentation."
        )

    await evaluator.verify(
        claim=claim,
        node=opt_leaf,
        additional_instruction=add_ins,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Sean McVay coaching mentorship lineage task.
    """
    # Initialize evaluator with a SEQUENTIAL root to enforce generation order.
    # Note: We keep root non-critical to allow optional checks without violating
    # the framework's critical-child constraint and to allow partial credit.
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
        default_model=model,
    )

    # Extract structured lineage info from the answer
    extracted: CoachingLineageExtraction = await evaluator.extract(
        prompt=prompt_extract_coaching_lineage(),
        template_class=CoachingLineageExtraction,
        extraction_name="coaching_lineage_extraction",
    )

    # Add ground truth information
    evaluator.add_ground_truth({
        "starting_point_expected": {"name": STARTING_NAME, "team": STARTING_TEAM, "role": "current head coach"},
        "generations_expected": {
            "gen1": {
                "assistant": GEN_EXPECTATIONS[1]["assistant"],
                "mentor": GEN_EXPECTATIONS[1]["mentor"],
                "team": GEN_EXPECTATIONS[1]["team"],
                "period": GEN_EXPECTATIONS[1]["period"],
                "role_required": GEN_EXPECTATIONS[1]["role_required"],
                "allowed_roles": GEN_EXPECTATIONS[1]["allowed_roles"],
            },
            "gen2": {
                "assistant": GEN_EXPECTATIONS[2]["assistant"],
                "mentor": GEN_EXPECTATIONS[2]["mentor"],
                "team": GEN_EXPECTATIONS[2]["team"],
                "period": GEN_EXPECTATIONS[2]["period"],
                "role_required": GEN_EXPECTATIONS[2]["role_required"],
                "allowed_roles": GEN_EXPECTATIONS[2]["allowed_roles"],
            },
            "gen3": {
                "assistant": GEN_EXPECTATIONS[3]["assistant"],
                "mentor": GEN_EXPECTATIONS[3]["mentor"],
                "team": GEN_EXPECTATIONS[3]["team"],
                "period": GEN_EXPECTATIONS[3]["period"],
                "role_required": GEN_EXPECTATIONS[3]["role_required"],
                "allowed_roles": GEN_EXPECTATIONS[3]["allowed_roles"],
            },
            "gen4": {
                "assistant": GEN_EXPECTATIONS[4]["assistant"],
                "mentor": GEN_EXPECTATIONS[4]["mentor"],
                "team": GEN_EXPECTATIONS[4]["team"],
                "period": GEN_EXPECTATIONS[4]["period"],
                "role_required": GEN_EXPECTATIONS[4]["role_required"],
                "allowed_roles": GEN_EXPECTATIONS[4]["allowed_roles"],
            },
        }
    }, gt_type="ground_truth")

    # 1) Starting point check
    await verify_starting_point(evaluator, root, extracted)

    # 2) Generation 1 required checks
    await verify_generation_required(
        evaluator=evaluator,
        parent_node=root,
        generation_index=1,
        gen_info=extracted.gen1,
        assistant_name=GEN_EXPECTATIONS[1]["assistant"],
        mentor_expected=GEN_EXPECTATIONS[1]["mentor"],
        team_expected=GEN_EXPECTATIONS[1]["team"],
        period_expected=GEN_EXPECTATIONS[1]["period"],
        role_required=GEN_EXPECTATIONS[1]["role_required"],
        allowed_roles=GEN_EXPECTATIONS[1]["allowed_roles"],
    )

    # 3) Generation 2 required checks
    await verify_generation_required(
        evaluator=evaluator,
        parent_node=root,
        generation_index=2,
        gen_info=extracted.gen2,
        assistant_name=GEN_EXPECTATIONS[2]["assistant"],
        mentor_expected=GEN_EXPECTATIONS[2]["mentor"],
        team_expected=GEN_EXPECTATIONS[2]["team"],
        period_expected=GEN_EXPECTATIONS[2]["period"],
        role_required=GEN_EXPECTATIONS[2]["role_required"],
        allowed_roles=GEN_EXPECTATIONS[2]["allowed_roles"],
    )

    # 4) Generation 3 required checks
    await verify_generation_required(
        evaluator=evaluator,
        parent_node=root,
        generation_index=3,
        gen_info=extracted.gen3,
        assistant_name=GEN_EXPECTATIONS[3]["assistant"],
        mentor_expected=GEN_EXPECTATIONS[3]["mentor"],
        team_expected=GEN_EXPECTATIONS[3]["team"],
        period_expected=GEN_EXPECTATIONS[3]["period"],
        role_required=GEN_EXPECTATIONS[3]["role_required"],
        allowed_roles=GEN_EXPECTATIONS[3]["allowed_roles"],
    )

    # 5) Generation 4 required checks
    await verify_generation_required(
        evaluator=evaluator,
        parent_node=root,
        generation_index=4,
        gen_info=extracted.gen4,
        assistant_name=GEN_EXPECTATIONS[4]["assistant"],
        mentor_expected=GEN_EXPECTATIONS[4]["mentor"],
        team_expected=GEN_EXPECTATIONS[4]["team"],
        period_expected=GEN_EXPECTATIONS[4]["period"],
        role_required=GEN_EXPECTATIONS[4]["role_required"],
        allowed_roles=GEN_EXPECTATIONS[4]["allowed_roles"],
    )

    # 6) Optional role checks placed AFTER required chain, so failures won't block earlier generations
    #    (They are non-critical and evaluated at the end of the sequential chain)
    await verify_optional_role(
        evaluator=evaluator,
        parent_node=root,
        generation_index=2,
        gen_info=extracted.gen2,
        assistant_name=GEN_EXPECTATIONS[2]["assistant"],
    )
    await verify_optional_role(
        evaluator=evaluator,
        parent_node=root,
        generation_index=4,
        gen_info=extracted.gen4,
        assistant_name=GEN_EXPECTATIONS[4]["assistant"],
    )

    # Return structured evaluation result
    return evaluator.get_summary()