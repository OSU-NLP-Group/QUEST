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
TASK_ID = "mi_veteran_ptsd_byod_cgc_iaadp"
TASK_DESCRIPTION = """A military veteran with PTSD who recently relocated to Michigan already has a well-behaved dog at home and is seeking a service dog training program. Identify a Michigan-based service dog training organization that meets all of the following requirements:

1. The organization must provide service dog training specifically for military veterans diagnosed with PTSD and/or TBI
2. The organization must offer a "Bring Your Own Dog" (BYOD) program option where veterans can have their own dogs trained as service dogs
3. The training program must meet or exceed the IAADP (International Association of Assistance Dog Partners) minimum training standards of at least 120 hours of training over a period of six months or more
4. The training program must include all three levels of Canine Good Citizen testing: Canine Good Citizen (CGC), Canine Good Citizen Advanced (CGC Advanced), and Canine Good Citizen Urban (CGC Urban)

For the identified organization, provide:
- The organization's name
- The specific city in Michigan where their training center is located
- The total number of training hours included in their program
- Confirmation that all three required CGC testing levels (CGC, CGC Advanced, and CGC Urban) are included in the program
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MIServiceDogOrg(BaseModel):
    org_name: Optional[str] = None
    michigan_city: Optional[str] = None
    program_name: Optional[str] = None
    training_hours_total: Optional[str] = None
    duration_text: Optional[str] = None
    byod_available: Optional[bool] = None
    byod_eligibility: List[str] = Field(default_factory=list)
    includes_cgc: Optional[bool] = None
    includes_cgc_advanced: Optional[bool] = None
    includes_cgc_urban: Optional[bool] = None
    veteran_ptsd_tbi_specific: Optional[bool] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_org() -> str:
    return """
    Extract the details for ONE identified Michigan-based service dog training organization from the answer.
    Return the following fields:
    - org_name: The name of the organization.
    - michigan_city: The specific Michigan city where their training center is located (just the city name).
    - program_name: The name of the specific service dog training program, if mentioned; else null.
    - training_hours_total: The total number of training hours stated in the program (e.g., "120 hours", "200+ hours"); if only a minimum is stated, extract what is stated (e.g., "at least 120 hours"). If not provided, set null.
    - duration_text: The program duration text if mentioned (e.g., "6 months", "24+ weeks", "1 year"); else null.
    - byod_available: true/false if the answer explicitly indicates a Bring Your Own Dog option (synonyms: train-your-own-dog, handler-owned/owner-trained dog), else null.
    - byod_eligibility: List of eligibility requirements for BYOD mentioned in the answer (e.g., "basic obedience completed", "spayed/neutered", "up-to-date vaccinations", "rabies license", "current county license", "no history of aggression"). Use the exact phrases from the answer when possible. If none, return empty list.
    - includes_cgc: true/false if CGC testing is included as part of the program, else null.
    - includes_cgc_advanced: true/false if CGC Advanced (Community Canine / CGCA) is included as part of the program, else null.
    - includes_cgc_urban: true/false if CGC Urban (CGCU) is included as part of the program, else null.
    - veteran_ptsd_tbi_specific: true/false if the organization specifically provides service dog training for military veterans with PTSD and/or TBI, else null.
    - source_urls: A list of all URLs explicitly cited in the answer that support this organization's program details. Only include valid, fully qualified URLs explicitly present in the answer text or its sources section.

    If any field is not mentioned in the answer, return null for single-value fields or [] for lists.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _nonempty(text: Optional[str]) -> bool:
    return text is not None and isinstance(text, str) and text.strip() != ""


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, root, info: MIServiceDogOrg) -> None:
    # Top-level children are all critical (root is critical)
    # 1) Organization_Details (parallel, critical)
    org_details_node = evaluator.add_parallel(
        id="Organization_Details",
        desc="Provide required identifying details for the organization and its Michigan training center city",
        parent=root,
        critical=True
    )

    # 1.1 Organization_Name (existence - critical leaf)
    evaluator.add_custom_node(
        result=_nonempty(info.org_name),
        id="Organization_Name",
        desc="The organization's name is provided",
        parent=org_details_node,
        critical=True
    )

    # 1.2 Michigan_Based (verify with URLs - critical leaf)
    michigan_based_leaf = evaluator.add_leaf(
        id="Michigan_Based",
        desc="The organization is based in Michigan and operates within the state",
        parent=org_details_node,
        critical=True
    )
    if _nonempty(info.michigan_city):
        mi_claim = f"This organization operates in Michigan and has a training location in {info.michigan_city}, Michigan."
    else:
        mi_claim = "This organization operates in the state of Michigan."
    await evaluator.verify(
        claim=mi_claim,
        node=michigan_based_leaf,
        sources=info.source_urls,
        additional_instruction="Accept evidence that the organization is located in Michigan or runs its service dog training in Michigan. If multiple locations exist, Michigan should be explicitly included."
    )

    # 1.3 Training_Center_City (existence - critical leaf)
    evaluator.add_custom_node(
        result=_nonempty(info.michigan_city),
        id="Training_Center_City",
        desc="The specific city in Michigan where the training center is located is provided",
        parent=org_details_node,
        critical=True
    )

    # 2) Target_Population (leaf, critical)
    target_population_leaf = evaluator.add_leaf(
        id="Target_Population",
        desc="Organization provides service dog training specifically for military veterans diagnosed with PTSD and/or TBI",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="This organization provides service dog training specifically for military veterans with PTSD and/or TBI.",
        node=target_population_leaf,
        sources=info.source_urls,
        additional_instruction="Look for explicit mention of military veterans and PTSD and/or TBI (post-traumatic stress disorder, traumatic brain injury). Accept if the program is tailored for veterans with PTSD, TBI, or both."
    )

    # 3) BYOD_Program (sequential, critical)
    byod_node = evaluator.add_sequential(
        id="BYOD_Program",
        desc="Organization offers a Bring Your Own Dog (BYOD) option for veterans, and the BYOD option has the required dog eligibility criteria",
        parent=root,
        critical=True
    )

    # 3.1 BYOD_Availability (leaf, critical)
    byod_avail_leaf = evaluator.add_leaf(
        id="BYOD_Availability",
        desc="The organization offers a BYOD program option where veterans can have their own dogs trained as service dogs",
        parent=byod_node,
        critical=True
    )
    await evaluator.verify(
        claim="This organization offers a Bring Your Own Dog (BYOD) program option that allows veterans to have their own dog trained as a service dog.",
        node=byod_avail_leaf,
        sources=info.source_urls,
        additional_instruction="Accept synonyms such as train-your-own-dog, handler-owned dog, owner-trained service dog, or similar phrasing that clearly indicates veterans can use their own dog within the program."
    )

    # 3.2 BYOD_Dog_Eligibility_Requirements (leaf, critical)
    byod_elig_leaf = evaluator.add_leaf(
        id="BYOD_Dog_Eligibility_Requirements",
        desc="BYOD program requires dogs to meet the specified eligibility criteria (basic obedience completed, spayed/neutered, veterinary health certification with up-to-date vaccinations and rabies license, current county license, and no history of aggression)",
        parent=byod_node,
        critical=True
    )
    elig_claim = (
        "The organization's BYOD program requires dogs to meet ALL of the following eligibility criteria: "
        "1) basic obedience completed; 2) spayed/neutered; 3) veterinary health certification with up-to-date vaccinations and a rabies license; "
        "4) current county dog license (local/city/county dog license acceptable if equivalent); and 5) no history of aggression."
    )
    await evaluator.verify(
        claim=elig_claim,
        node=byod_elig_leaf,
        sources=info.source_urls,
        additional_instruction=(
            "Verify that the BYOD eligibility requirements explicitly include all listed elements. "
            "Synonyms are acceptable: 'altered/fixed' for spayed/neutered; 'current vaccinations' for up-to-date vaccines; "
            "'rabies certificate/tag/license' for rabies license; 'local/city/county dog license' for county license; "
            "'no aggression' can appear as 'no bite history' or 'no aggressive behavior'."
        )
    )

    # 4) IAADP_Minimum_Standards (parallel, critical)
    iaadp_node = evaluator.add_parallel(
        id="IAADP_Minimum_Standards",
        desc="Program meets or exceeds IAADP minimums: at least 120 training hours over at least six months",
        parent=root,
        critical=True
    )

    # 4.1 Training_Hours (sequential, critical)
    hours_node = evaluator.add_sequential(
        id="Training_Hours",
        desc="Total number of training hours is provided and meets the 120-hour minimum",
        parent=iaadp_node,
        critical=True
    )

    # 4.1.1 Hours_Provided (existence - critical)
    evaluator.add_custom_node(
        result=_nonempty(info.training_hours_total),
        id="Hours_Provided",
        desc="The total number of training hours included in the program is stated",
        parent=hours_node,
        critical=True
    )

    # 4.1.2 Hours_Meet_Minimum (verify - critical)
    hours_min_leaf = evaluator.add_leaf(
        id="Hours_Meet_Minimum",
        desc="The stated total training hours meet or exceed 120 hours",
        parent=hours_node,
        critical=True
    )
    hours_claim = "The program includes at least 120 total hours of training."
    add_ins_hours = (
        "Check the program page(s) for the total hours. "
        "If the answer stated a number, confirm that number meets or exceeds 120 hours. "
        "Accept phrasing like '120+ hours', 'at least 120 hours', '200 hours', etc."
    )
    if _nonempty(info.training_hours_total):
        add_ins_hours += f" The answer stated: {info.training_hours_total}."
    await evaluator.verify(
        claim=hours_claim,
        node=hours_min_leaf,
        sources=info.source_urls,
        additional_instruction=add_ins_hours
    )

    # 4.2 Training_Duration (verify - critical)
    duration_leaf = evaluator.add_leaf(
        id="Training_Duration",
        desc="Evidence is provided that program duration meets or exceeds six months",
        parent=iaadp_node,
        critical=True
    )
    duration_claim = "The program duration is at least six months in length."
    add_ins_duration = (
        "Look for explicit duration statements: '6 months', '24 weeks', '6+ months', 'one year', 'two semesters', etc. "
        "Any duration equal to or exceeding 6 months should pass."
    )
    if _nonempty(info.duration_text):
        add_ins_duration += f" The answer stated: {info.duration_text}."
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=info.source_urls,
        additional_instruction=add_ins_duration
    )

    # 5) CGC_Testing (parallel, critical)
    cgc_node = evaluator.add_parallel(
        id="CGC_Testing",
        desc="Program includes all three required CGC testing levels as part of the same training program",
        parent=root,
        critical=True
    )

    # 5.1 Includes_CGC (leaf, critical)
    includes_cgc_leaf = evaluator.add_leaf(
        id="Includes_CGC",
        desc="Program includes Canine Good Citizen (CGC) testing",
        parent=cgc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program includes Canine Good Citizen (CGC) testing.",
        node=includes_cgc_leaf,
        sources=info.source_urls,
        additional_instruction="Accept references to AKC CGC or 'Canine Good Citizen'. It must be included as part of the service dog training program requirements or curriculum."
    )

    # 5.2 Includes_CGC_Advanced (leaf, critical)
    includes_cgca_leaf = evaluator.add_leaf(
        id="Includes_CGC_Advanced",
        desc="Program includes Canine Good Citizen Advanced (CGC Advanced / Community Canine) testing",
        parent=cgc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program includes Canine Good Citizen Advanced, also known as AKC Community Canine (CGCA).",
        node=includes_cgca_leaf,
        sources=info.source_urls,
        additional_instruction="Accept synonyms: CGC Advanced, Community Canine, AKC CGCA. It must be part of the program."
    )

    # 5.3 Includes_CGC_Urban (leaf, critical)
    includes_cgcu_leaf = evaluator.add_leaf(
        id="Includes_CGC_Urban",
        desc="Program includes Canine Good Citizen Urban (CGC Urban) testing",
        parent=cgc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program includes Canine Good Citizen Urban (CGCU) testing.",
        node=includes_cgcu_leaf,
        sources=info.source_urls,
        additional_instruction="Accept synonyms: Urban CGC, AKC CGCU. It must be part of the program."
    )

    # 5.4 All_Three_In_Same_Program (leaf, critical)
    all_three_leaf = evaluator.add_leaf(
        id="All_Three_In_Same_Program",
        desc="All three CGC testing levels are part of the same training program",
        parent=cgc_node,
        critical=True
    )
    org_for_claim = info.org_name if _nonempty(info.org_name) else "the organization"
    all_three_claim = (
        f"The same service dog training program offered by {org_for_claim} includes all three CGC testing levels: "
        "CGC, CGC Advanced/Community Canine (CGCA), and CGC Urban (CGCU)."
    )
    await evaluator.verify(
        claim=all_three_claim,
        node=all_three_leaf,
        sources=info.source_urls,
        additional_instruction="Confirm that all three CGC levels are included within the same program pathway or curriculum, not merely offered as unrelated stand‑alone classes."
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
    Evaluate an answer for the Michigan veteran PTSD BYOD CGC IAADP organization task.
    """
    # Initialize evaluator with critical root (parallel aggregation at root)
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
    # Ensure root is critical per rubric
    root.critical = True

    # Extract organization/program details from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_org(),
        template_class=MIServiceDogOrg,
        extraction_name="selected_org"
    )

    # Build verification tree and run checks
    await build_and_verify(evaluator, root, extracted_info)

    # Return standardized summary
    return evaluator.get_summary()