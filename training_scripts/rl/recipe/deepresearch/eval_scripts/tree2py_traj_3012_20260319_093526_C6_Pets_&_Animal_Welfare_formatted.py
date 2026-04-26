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
TASK_ID = "thanksgiving_2025_bis_handler"
TASK_DESCRIPTION = (
    "Identify the professional dog handler who handled the Best in Show winner at the nationally televised dog show "
    "that aired on Thanksgiving Day 2025. Provide the handler's name and the U.S. state where they are located. "
    "Additionally, provide the following information: "
    "(1) The name of the winning dog and its breed; "
    "(2) The AKC judge qualification requirements under the 12-5-4 method, including the minimum years of experience "
    "exhibiting in conformation, the minimum number of litters that must be bred and raised on premises, and the "
    "minimum number of champions that must be bred from those litters; "
    "(3) The mandatory pre-application requirements for new AKC breed judges, including the required number of "
    "stewarding assignments, judging assignments, and the Basic Institute attendance requirement; "
    "(4) The name of the show host and confirm whether they have hosted the show for at least 20 consecutive years "
    "as of 2025; "
    "(5) Confirm whether the handler's state has regulations requiring animal shelters to sterilize animals before "
    "adoption; "
    "(6) Confirm whether major pet store chains (such as PetSmart) are closed on both Thanksgiving and Christmas in "
    "the handler's state."
)

THANKSGIVING_2025_DATE_TEXT = "Thursday, November 27, 2025"

# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
US_STATES: Dict[str, str] = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY"
}
US_STATE_ABBRS = set(US_STATES.values())
US_STATE_FULLS = set(US_STATES.keys())


def is_valid_us_state(state: Optional[str]) -> bool:
    if not state:
        return False
    s = state.strip().upper()
    return s in US_STATE_ABBRS or s in US_STATE_FULLS


def combine_urls(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


def parse_affirmative(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    t = text.strip().lower()
    if not t:
        return None
    # Simple heuristics:
    pos_tokens = ["yes", "require", "required", "mandatory", "must", "closed on both", "closed both", "closed thanksgiving and christmas"]
    neg_tokens = ["no", "not required", "does not require", "no statewide", "open", "open on thanksgiving", "open on christmas"]
    # If explicit "closed on both" appears, treat as True for holiday closures.
    if any(tok in t for tok in pos_tokens):
        return True
    if any(tok in t for tok in neg_tokens):
        return False
    return None


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ShowCoreExtraction(BaseModel):
    show_name: Optional[str] = None
    broadcast_network: Optional[str] = None
    aired_date_text: Optional[str] = None
    show_urls: List[str] = Field(default_factory=list)

    dog_name: Optional[str] = None
    dog_breed: Optional[str] = None
    dog_urls: List[str] = Field(default_factory=list)

    handler_name: Optional[str] = None
    handler_state: Optional[str] = None
    handler_urls: List[str] = Field(default_factory=list)


class AKC1254Extraction(BaseModel):
    years_exhibiting_min: Optional[str] = None
    litters_min: Optional[str] = None
    champions_min: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AKCPreappExtraction(BaseModel):
    stewarding_assignments_required: Optional[str] = None
    stewarding_time_window: Optional[str] = None
    judging_assignments_required: Optional[str] = None
    judging_assignment_types: Optional[str] = None
    basic_institute_requirement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HostTenureExtraction(BaseModel):
    host_name: Optional[str] = None
    tenure_statement_as_of_2025: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StatePoliciesExtraction(BaseModel):
    spay_neuter_shelter_requirement_in_state: Optional[str] = None
    spay_neuter_sources: List[str] = Field(default_factory=list)

    petsmart_holiday_closures_in_state: Optional[str] = None
    petsmart_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_show_core() -> str:
    return """
    Extract the show, winning dog, and handler details explicitly stated in the answer.

    Fields to extract:
    - show_name: The name/title of the nationally televised dog show referenced in the answer.
    - broadcast_network: The TV network named in the answer (e.g., NBC, ABC, CBS, FOX) that carries the show nationally.
    - aired_date_text: The date reference used in the answer for when the show aired (e.g., "Thanksgiving Day 2025", or a specific date).
    - show_urls: All URLs in the answer that support the show identity, schedule, or broadcast info.

    - dog_name: The name of the Best in Show winning dog.
    - dog_breed: The breed of the Best in Show winning dog.
    - dog_urls: All URLs in the answer that support the dog's name/breed and Best in Show result.

    - handler_name: The full name of the handler who handled the Best in Show winner.
    - handler_state: The U.S. state where the handler is located (either full state name or two-letter abbreviation), as stated in the answer.
    - handler_urls: All URLs in the answer that support the handler identification and role.

    Rules:
    - Extract only what is explicitly present in the answer. If a field is missing, return null (or an empty array for URL lists).
    - For URL fields, include only valid URLs that are explicitly mentioned in the answer (plain URLs or markdown links).
    """


def prompt_extract_akc_1254() -> str:
    return """
    Extract the AKC judge qualification requirements under the "12-5-4" method, as presented in the answer.

    Fields:
    - years_exhibiting_min: The minimum years of experience exhibiting in conformation (typically 12).
    - litters_min: The minimum number of litters that must be bred and raised on premises (typically 5).
    - champions_min: The minimum number of champions that must be bred from those litters (typically 4).
    - sources: All URLs cited in the answer that support the AKC 12-5-4 method requirements.

    Rules:
    - Return exact numbers/phrases as stated in the answer text.
    - If any field is not present in the answer, return null for that field (or an empty array for URLs).
    """


def prompt_extract_akc_preapp() -> str:
    return """
    Extract the mandatory AKC pre-application requirements for new AKC breed judges, as presented in the answer.

    Fields:
    - stewarding_assignments_required: The required number of stewarding assignments (e.g., "6").
    - stewarding_time_window: The time window for those stewarding assignments (e.g., "in the 3 years immediately preceding application", and that they must be at AKC member/licensed shows).
    - judging_assignments_required: The required number of judging assignments (e.g., "6").
    - judging_assignment_types: The sanctioned event types for those judging assignments (e.g., "AKC sanctioned matches, Open Shows, 4–6 Month Beginner Puppy, etc.", as stated in the answer).
    - basic_institute_requirement: The Basic Institute attendance requirement description, using the answer’s wording (e.g., "must attend prior to requesting regular status; not earlier than two years prior to initial application submission").
    - sources: All URLs cited in the answer that support these pre-application requirements.

    Rules:
    - Capture the answer’s own phrasing; do not invent missing details.
    - If any requested field is absent in the answer, set it to null (or empty array for URLs).
    """


def prompt_extract_host_tenure() -> str:
    return """
    Extract the show host information and tenure statement (as of 2025), as presented in the answer.

    Fields:
    - host_name: The name of the show host.
    - tenure_statement_as_of_2025: The explicit statement or claim in the answer regarding whether the host has hosted the show for at least 20 consecutive years as of 2025 (capture the answer's wording or "yes/no" with brief reasoning).
    - sources: All URLs cited in the answer that support the host identity and tenure history.

    Rules:
    - Use only information explicitly present in the answer. If missing, return null (or empty array for URLs).
    """


def prompt_extract_state_policies() -> str:
    return """
    Extract the handler-state policy confirmations from the answer.

    Fields:
    - spay_neuter_shelter_requirement_in_state: The answer’s statement on whether the handler’s state has regulations requiring animal shelters to sterilize animals before adoption (e.g., "yes", "no", or a short description).
    - spay_neuter_sources: All URLs provided that support the shelter spay/neuter regulatory status in that state.

    - petsmart_holiday_closures_in_state: The answer’s statement on whether major pet store chains (such as PetSmart) are closed on BOTH Thanksgiving and Christmas in the handler’s state (e.g., "yes", "no", or a short description).
    - petsmart_sources: All URLs provided that support the PetSmart (or similar chain) holiday closures for that state.

    Rules:
    - Extract only claims explicitly present in the answer. If a field is missing, set it to null (or empty array for URL lists).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_show_core(
    evaluator: Evaluator,
    parent,
    core: ShowCoreExtraction
) -> None:
    core_node = evaluator.add_parallel(
        id="Show_Handler_Dog_Core",
        desc="Correctly identify the qualifying show context plus the Best in Show winning dog and its handler (including handler’s U.S. state).",
        parent=parent,
        critical=True
    )

    # 1) The show is nationally televised on a major broadcast network.
    show_net_node = evaluator.add_leaf(
        id="Show_Nationally_Televised_Major_Broadcast_Network",
        desc="The show is nationally televised on a major broadcast network.",
        parent=core_node,
        critical=True
    )
    net = core.broadcast_network or "a major U.S. broadcast network"
    show_name = core.show_name or "the referenced nationally televised dog show"
    await evaluator.verify(
        claim=f"The show {show_name} is nationally televised on {net}.",
        node=show_net_node,
        sources=core.show_urls,
        additional_instruction="Treat NBC, ABC, CBS, or FOX as major U.S. broadcast networks. The page should explicitly indicate national broadcast coverage on one of these networks."
    )

    # 2) The show aired on Thanksgiving Day 2025.
    show_air_node = evaluator.add_leaf(
        id="Show_Aired_Thanksgiving_Day_2025",
        desc="The show aired on Thanksgiving Day 2025.",
        parent=core_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The show {show_name} aired on Thanksgiving Day 2025 ({THANKSGIVING_2025_DATE_TEXT}).",
        node=show_air_node,
        sources=core.show_urls,
        additional_instruction="Confirm the 2025 broadcast occurred on Thanksgiving Day. It is acceptable if a source states the show airs 'on Thanksgiving Day' annually and confirms the 2025 schedule/date."
    )

    # 3) Best in Show winning dog's name and breed provided correctly.
    dog_node = evaluator.add_leaf(
        id="Winning_Dog_Name_And_Breed_Provided_Correctly",
        desc="Provide the Best in Show winning dog’s name and breed correctly.",
        parent=core_node,
        critical=True
    )
    dog_name = core.dog_name or ""
    dog_breed = core.dog_breed or ""
    await evaluator.verify(
        claim=f"The Best in Show winner at {show_name} in 2025 was '{dog_name}', a '{dog_breed}'.",
        node=dog_node,
        sources=combine_urls(core.dog_urls, core.show_urls),
        additional_instruction="The source should explicitly list the Best in Show winner's registered name and breed for the 2025 show. Minor formatting variations (e.g., punctuation, suffixes) are acceptable."
    )

    # 4) Handler correctly identified as the professional handler who handled the BIS winner at the qualifying show.
    handler_bis_node = evaluator.add_leaf(
        id="Handler_Correctly_Identified_As_BIS_Handler_And_Professional",
        desc="Provide the handler’s name and correctly identify them as the professional handler who handled the Best in Show winner at the qualifying show.",
        parent=core_node,
        critical=True
    )
    handler_name = core.handler_name or ""
    await evaluator.verify(
        claim=f"{handler_name} is the professional handler who handled the Best in Show winner '{dog_name}' at {show_name} in 2025.",
        node=handler_bis_node,
        sources=combine_urls(core.handler_urls, core.dog_urls, core.show_urls),
        additional_instruction="The supporting page(s) should explicitly indicate that this handler handled the Best in Show winner at the 2025 edition of the show and that they are a professional handler (e.g., explicitly called a professional handler)."
    )

    # 5) Handler state provided and it is a U.S. state (custom existence/format check).
    state_valid = is_valid_us_state(core.handler_state)
    evaluator.add_custom_node(
        result=bool(core.handler_state) and state_valid,
        id="Handler_State_Provided_And_Is_US_State",
        desc="Provide the U.S. state where the handler is located, and it is a U.S. state.",
        parent=core_node,
        critical=True
    )


async def verify_akc_1254(
    evaluator: Evaluator,
    parent,
    a: AKC1254Extraction
) -> None:
    node = evaluator.add_parallel(
        id="AKC_12_5_4_Method_Requirements",
        desc="Provide AKC judge qualification requirements under the 12-5-4 method (the three requested minimums).",
        parent=parent,
        critical=True
    )

    # Years exhibiting minimum
    years_node = evaluator.add_leaf(
        id="AKC_12_Years_Exhibiting_Minimum",
        desc="State the minimum years of experience exhibiting in conformation required under the 12-5-4 method (12+ years).",
        parent=node,
        critical=True
    )
    years_text = a.years_exhibiting_min or ""
    await evaluator.verify(
        claim=f"Under the AKC '12-5-4' method, the minimum years of experience exhibiting in conformation is '{years_text}'.",
        node=years_node,
        sources=a.sources,
        additional_instruction="Verify that the AKC page states a minimum of 12 years exhibiting in conformation for the 12-5-4 path. Allow equivalent phrasings like 'at least 12 years'."
    )

    # Litters minimum
    litters_node = evaluator.add_leaf(
        id="AKC_5_Litters_Minimum",
        desc="State the minimum number of litters that must be bred and raised on premises under the 12-5-4 method (5+ litters).",
        parent=node,
        critical=True
    )
    litters_text = a.litters_min or ""
    await evaluator.verify(
        claim=f"Under the AKC '12-5-4' method, the minimum number of litters bred and raised on premises is '{litters_text}'.",
        node=litters_node,
        sources=a.sources,
        additional_instruction="Verify that the AKC page states a minimum of 5 litters that must be bred and raised on the applicant's premises. Allow 'at least 5 litters' as equivalent."
    )

    # Champions minimum
    champs_node = evaluator.add_leaf(
        id="AKC_4_Champions_Minimum",
        desc="State the minimum number of champions that must be bred from those litters under the 12-5-4 method (4+ champions).",
        parent=node,
        critical=True
    )
    champs_text = a.champions_min or ""
    await evaluator.verify(
        claim=f"Under the AKC '12-5-4' method, the minimum number of champions that must be bred from those litters is '{champs_text}'.",
        node=champs_node,
        sources=a.sources,
        additional_instruction="Verify that the AKC page states a minimum of 4 champions bred from those litters. Allow 'at least 4 champions' as equivalent."
    )


async def verify_akc_preapp(
    evaluator: Evaluator,
    parent,
    p: AKCPreappExtraction
) -> None:
    node = evaluator.add_parallel(
        id="AKC_Preapplication_Requirements",
        desc="Provide the mandatory AKC pre-application requirements for new AKC breed judges (three requested elements).",
        parent=parent,
        critical=True
    )

    # Stewarding assignments and timing
    st_node = evaluator.add_leaf(
        id="Preapp_6_Stewarding_Assignments",
        desc="State the required number of stewarding assignments and timing window (6 stewarding assignments at AKC member/licensed shows in the 3 years immediately preceding application).",
        parent=node,
        critical=True
    )
    steward_cnt = p.stewarding_assignments_required or ""
    steward_time = p.stewarding_time_window or ""
    await evaluator.verify(
        claim=f"New AKC breed judges must complete '{steward_cnt}' stewarding assignments '{steward_time}'.",
        node=st_node,
        sources=p.sources,
        additional_instruction="Check both the count (6) and the timing/window (at AKC member/licensed shows within the 3 years immediately preceding application). If either element is missing or wrong, mark incorrect."
    )

    # Judging assignments
    jud_node = evaluator.add_leaf(
        id="Preapp_6_Judging_Assignments",
        desc="State the required number of judging assignments (6 judging assignments at the specified sanctioned event types).",
        parent=node,
        critical=True
    )
    judging_cnt = p.judging_assignments_required or ""
    judging_types = p.judging_assignment_types or ""
    await evaluator.verify(
        claim=f"New AKC breed judges must complete '{judging_cnt}' judging assignments at the specified sanctioned event types: '{judging_types}'.",
        node=jud_node,
        sources=p.sources,
        additional_instruction="Verify the required count (6) and that the event types match AKC policy (e.g., AKC sanctioned matches, Open Shows, 4–6 Month Beginner Puppy, etc.)."
    )

    # Basic Institute attendance timing
    bi_node = evaluator.add_leaf(
        id="Preapp_Basic_Institute_Attendance_Timing",
        desc="State the Basic Institute attendance requirement (must attend prior to requesting regular status; not earlier than two years prior to initial application submission).",
        parent=node,
        critical=True
    )
    bi_text = p.basic_institute_requirement or ""
    await evaluator.verify(
        claim=f"The Basic Institute attendance requirement is described as: '{bi_text}'.",
        node=bi_node,
        sources=p.sources,
        additional_instruction="This must convey two points: (1) attendance required prior to requesting regular status; and (2) attendance must be no earlier than two years prior to initial application submission. If the quoted description omits or contradicts either, mark incorrect."
    )


async def verify_host_and_tenure(
    evaluator: Evaluator,
    parent,
    h: HostTenureExtraction,
    show_name: Optional[str]
) -> None:
    node = evaluator.add_parallel(
        id="Show_Host_And_Tenure",
        desc="Provide the show host's name and confirm the tenure condition as of 2025.",
        parent=parent,
        critical=True
    )
    sname = show_name or "the show"

    # Host name
    host_node = evaluator.add_leaf(
        id="Host_Name_Provided",
        desc="Provide the name of the show host.",
        parent=node,
        critical=True
    )
    host_name = h.host_name or ""
    await evaluator.verify(
        claim=f"The host of {sname} is '{host_name}'.",
        node=host_node,
        sources=h.sources,
        additional_instruction="The source should explicitly name the show host."
    )

    # Tenure: at least 20 consecutive years as of 2025
    tenure_node = evaluator.add_leaf(
        id="Host_Has_At_Least_20_Consecutive_Years_As_Of_2025",
        desc="Confirm whether the host has hosted the show for at least 20 consecutive years as of 2025 (must satisfy the stated constraint).",
        parent=node,
        critical=True
    )
    tenure_text = h.tenure_statement_as_of_2025 or "at least 20 consecutive years as of 2025"
    await evaluator.verify(
        claim=f"As of 2025, '{host_name}' has hosted {sname} for at least 20 consecutive years.",
        node=tenure_node,
        sources=h.sources,
        additional_instruction="The evidence should support a start year and continuous hosting through 2025 amounting to 20+ consecutive years (e.g., hosting since 2002 would satisfy 20+ consecutive years by 2025)."
    )


async def verify_state_policy_and_retail(
    evaluator: Evaluator,
    parent,
    sp: StatePoliciesExtraction,
    handler_state: Optional[str]
) -> None:
    # Handler state sterilization regulation confirmation
    steril_node = evaluator.add_leaf(
        id="Handler_State_Shelter_Sterilization_Regulation",
        desc="Confirm whether the handler's state has regulations requiring animal shelters to sterilize animals before adoption (must satisfy the stated constraint).",
        parent=parent,
        critical=True
    )
    state_text = handler_state or "the state in question"
    ster_val = parse_affirmative(sp.spay_neuter_shelter_requirement_in_state)
    if ster_val is True:
        steril_claim = f"{state_text} has a statewide requirement that animal shelters sterilize animals before adoption."
    elif ster_val is False:
        steril_claim = f"{state_text} does not have a statewide requirement that animal shelters sterilize animals before adoption."
    else:
        # If unclear in the answer, make a neutral restatement which will likely fail without clear evidence
        steril_claim = f"The statement about whether {state_text} requires shelters to sterilize animals before adoption is correct as described in the answer."
    await evaluator.verify(
        claim=steril_claim,
        node=steril_node,
        sources=sp.spay_neuter_sources,
        additional_instruction="Focus on statewide statutes/regulations (not just city/county ordinances). If only local ordinances exist without a statewide mandate, treat 'state requires' as false."
    )

    # PetSmart closures on both Thanksgiving and Christmas in the handler's state
    petsmart_node = evaluator.add_leaf(
        id="PetSmart_Closed_Thanksgiving_And_Christmas_In_Handler_State",
        desc="Confirm whether PetSmart (explicitly referenced as an example in the prompt) is closed on both Thanksgiving and Christmas in the handler's state (must satisfy the stated constraint).",
        parent=parent,
        critical=True
    )
    ps_val = parse_affirmative(sp.petsmart_holiday_closures_in_state)
    if ps_val is True:
        ps_claim = f"PetSmart stores in {state_text} are closed on both Thanksgiving Day and Christmas Day."
    elif ps_val is False:
        ps_claim = f"PetSmart stores in {state_text} are not closed on both Thanksgiving Day and Christmas Day."
    else:
        ps_claim = f"The statement about PetSmart stores in {state_text} being closed on both Thanksgiving and Christmas is correct as described in the answer."
    await evaluator.verify(
        claim=ps_claim,
        node=petsmart_node,
        sources=sp.petsmart_sources,
        additional_instruction="Accept official PetSmart corporate pages or local store pages that clearly list holiday closures. Both holidays must be closed to count as 'closed on both'. If a store is open with limited hours on Thanksgiving, that is not 'closed'."
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

    # Top-level critical task node
    task_node = evaluator.add_parallel(
        id="Handler_Identification_Task",
        desc="Identify the professional handler of the Best in Show winner at the nationally televised dog show that aired on Thanksgiving Day 2025, and provide all requested accompanying information per the prompt/constraints.",
        parent=root,
        critical=True
    )

    # Run all extractions in parallel
    core_task = evaluator.extract(
        prompt=prompt_extract_show_core(),
        template_class=ShowCoreExtraction,
        extraction_name="show_core"
    )
    akc1254_task = evaluator.extract(
        prompt=prompt_extract_akc_1254(),
        template_class=AKC1254Extraction,
        extraction_name="akc_12_5_4"
    )
    akc_preapp_task = evaluator.extract(
        prompt=prompt_extract_akc_preapp(),
        template_class=AKCPreappExtraction,
        extraction_name="akc_preapplication"
    )
    host_task = evaluator.extract(
        prompt=prompt_extract_host_tenure(),
        template_class=HostTenureExtraction,
        extraction_name="host_tenure"
    )
    state_policies_task = evaluator.extract(
        prompt=prompt_extract_state_policies(),
        template_class=StatePoliciesExtraction,
        extraction_name="state_policies"
    )

    core, akc1254, akc_preapp, host_info, state_pols = await asyncio.gather(
        core_task, akc1254_task, akc_preapp_task, host_task, state_policies_task
    )

    # Optionally record GT or custom info (none provided); record handler state used
    evaluator.add_custom_info(
        info={"handler_state_extracted": core.handler_state or None},
        info_type="meta",
        info_name="handler_state_context"
    )

    # Build verification subtrees (in parallel)
    await asyncio.gather(
        verify_show_core(evaluator, task_node, core),
        verify_akc_1254(evaluator, task_node, akc1254),
        verify_akc_preapp(evaluator, task_node, akc_preapp),
        verify_host_and_tenure(evaluator, task_node, host_info, core.show_name),
        verify_state_policy_and_retail(evaluator, task_node, state_pols, core.handler_state),
    )

    return evaluator.get_summary()