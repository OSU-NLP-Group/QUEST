import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "beginner_scarf_project"
TASK_DESCRIPTION = (
    "I want to start knitting or crocheting my first scarf as a complete beginner. Between Michaels and Joann Fabrics, "
    "which nationwide craft store chain is currently operational where I can purchase supplies? In how many US states "
    "does this operational chain have store locations? What yarn weight is recommended for beginner scarf projects, and "
    "how much yarn (in yards) would I need for an average scarf? If I choose to crochet, what hook size should I use "
    "for the recommended yarn weight? If I choose to knit instead, what needle size should I use? Finally, what fiber "
    "type of yarn is most popular and recommended for beginners?"
)


class BeginnerScarfPlan(BaseModel):
    operational_chain: Optional[str] = None
    operational_chain_sources: List[str] = Field(default_factory=list)

    state_coverage_number: Optional[str] = None
    state_coverage_sources: List[str] = Field(default_factory=list)

    recommended_yarn_weight: Optional[str] = None
    yarn_weight_sources: List[str] = Field(default_factory=list)

    yarn_amount_yards: Optional[str] = None
    yarn_amount_sources: List[str] = Field(default_factory=list)

    crochet_hook_size: Optional[str] = None
    crochet_hook_sources: List[str] = Field(default_factory=list)

    knitting_needle_size: Optional[str] = None
    knitting_needle_sources: List[str] = Field(default_factory=list)

    beginner_yarn_type: Optional[str] = None
    beginner_yarn_type_sources: List[str] = Field(default_factory=list)


def prompt_extract_beginner_scarf_plan() -> str:
    return (
        "Extract a complete beginner scarf project info package from the answer. Return the following fields:\n"
        "1) operational_chain: The chain chosen between 'Michaels' and 'Joann Fabrics' that the answer states is currently operational.\n"
        "2) operational_chain_sources: All URLs cited that support the operational status or store availability of the chosen chain.\n"
        "3) state_coverage_number: The number of US states in which the chosen chain has store locations, as stated in the answer (extract as a string; ranges or approximate words allowed).\n"
        "4) state_coverage_sources: All URLs cited that support the state coverage number.\n"
        "5) recommended_yarn_weight: The recommended yarn weight for beginner scarf projects (e.g., 'Medium/Worsted (4)').\n"
        "6) yarn_weight_sources: All URLs cited that support the recommended yarn weight.\n"
        "7) yarn_amount_yards: The yarn yardage required for an average beginner scarf (e.g., '250–400 yards', or a single value). Extract exactly as stated.\n"
        "8) yarn_amount_sources: All URLs cited that support the yarn yardage.\n"
        "9) crochet_hook_size: The recommended crochet hook size for worsted (medium, #4) yarn (include both mm and US size if available, e.g., '5.5 mm (I-9)').\n"
        "10) crochet_hook_sources: All URLs cited that support the crochet hook size.\n"
        "11) knitting_needle_size: The recommended knitting needle size (US size, optionally mm) for worsted (medium, #4) yarn (e.g., 'US 7–9 (4.5–5.5 mm)').\n"
        "12) knitting_needle_sources: All URLs cited that support the knitting needle size.\n"
        "13) beginner_yarn_type: The most popular yarn fiber type recommended for beginners (e.g., 'acrylic').\n"
        "14) beginner_yarn_type_sources: All URLs cited that support the beginner yarn fiber recommendation.\n"
        "Rules:\n"
        "- Extract only from the answer; do not invent.\n"
        "- For each *_sources field, extract only valid URLs explicitly present in the answer (plain URLs or markdown links).\n"
        "- If a field is missing, set it to null; if sources are missing, return an empty list.\n"
    )


async def verify_operational_chain(evaluator: Evaluator, parent_node, plan: BeginnerScarfPlan) -> None:
    node = evaluator.add_sequential(
        id="Operational_Craft_Store_Chain",
        desc="Identify which major nationwide craft store chain (between Michaels and Joann Fabrics) is currently operational for purchasing supplies",
        parent=parent_node,
        critical=False,
    )

    provided = bool(plan.operational_chain and plan.operational_chain.strip())
    has_sources = bool(plan.operational_chain_sources)
    evaluator.add_custom_node(
        result=provided and has_sources,
        id="Operational_Chain_Provided_With_Sources",
        desc="Operational chain name is provided and includes at least one source URL",
        parent=node,
        critical=True,
    )

    choice_node = evaluator.add_leaf(
        id="Operational_Chain_Is_Michaels_Or_Joann",
        desc="Selected chain is either 'Michaels' or 'Joann Fabrics'",
        parent=node,
        critical=False,
    )
    claim_choice = f"The selected chain '{plan.operational_chain or ''}' is either 'Michaels' or 'Joann Fabrics'."
    await evaluator.verify(
        claim=claim_choice,
        node=choice_node,
        additional_instruction="Consider minor casing/punctuation variations. Only these two chains are valid options."
    )

    verify_node = evaluator.add_leaf(
        id="Operational_Chain_Supported_By_Sources",
        desc="The chosen chain is currently operational for purchasing supplies",
        parent=node,
        critical=True,
    )
    claim_operational = (
        f"The craft store chain '{plan.operational_chain or ''}' is currently operational with stores open for purchasing supplies."
    )
    await evaluator.verify(
        claim=claim_operational,
        node=verify_node,
        sources=plan.operational_chain_sources,
        additional_instruction=(
            "Use the provided URLs to confirm the chain is actively operating (e.g., store locator pages, official announcements, "
            "current store lists). If the URLs are irrelevant or inaccessible, mark as not supported."
        ),
    )


async def verify_state_coverage(evaluator: Evaluator, parent_node, plan: BeginnerScarfPlan) -> None:
    node = evaluator.add_sequential(
        id="Store_State_Coverage",
        desc="Provide the number of US states where the operational craft store chain has locations",
        parent=parent_node,
        critical=False,
    )

    provided = bool(plan.state_coverage_number and plan.state_coverage_number.strip())
    has_sources = bool(plan.state_coverage_sources)
    evaluator.add_custom_node(
        result=provided and has_sources,
        id="State_Coverage_Provided_With_Sources",
        desc="State coverage number is provided and includes at least one source URL",
        parent=node,
        critical=True,
    )

    verify_node = evaluator.add_leaf(
        id="State_Coverage_Supported_By_Sources",
        desc="State coverage number is supported by cited sources",
        parent=node,
        critical=True,
    )
    chain_name = plan.operational_chain or "the chosen chain"
    claim_states = (
        f"{chain_name} has store locations in {plan.state_coverage_number or ''} US states."
    )
    await evaluator.verify(
        claim=claim_states,
        node=verify_node,
        sources=plan.state_coverage_sources,
        additional_instruction=(
            "Confirm the stated count via store locator, official site, or reliable sources. Allow reasonable wording variations "
            "like 'in X states and DC' but ensure the number matches the claim or is clearly equivalent."
        ),
    )


async def verify_yarn_weight(evaluator: Evaluator, parent_node, plan: BeginnerScarfPlan) -> None:
    node = evaluator.add_sequential(
        id="Yarn_Weight_Recommendation",
        desc="Specify the recommended yarn weight category for beginner scarf projects",
        parent=parent_node,
        critical=False,
    )

    provided = bool(plan.recommended_yarn_weight and plan.recommended_yarn_weight.strip())
    has_sources = bool(plan.yarn_weight_sources)
    evaluator.add_custom_node(
        result=provided and has_sources,
        id="Yarn_Weight_Provided_With_Sources",
        desc="Recommended yarn weight is provided and includes at least one source URL",
        parent=node,
        critical=True,
    )

    verify_node = evaluator.add_leaf(
        id="Yarn_Weight_Supported_By_Sources",
        desc="Recommended yarn weight is supported by cited sources",
        parent=node,
        critical=True,
    )
    claim_weight = (
        f"The recommended yarn weight for beginner scarf projects is {plan.recommended_yarn_weight or ''}."
    )
    await evaluator.verify(
        claim=claim_weight,
        node=verify_node,
        sources=plan.yarn_weight_sources,
        additional_instruction=(
            "Common recommendations include Medium/Worsted (#4). Allow minor naming variations (e.g., 'Worsted', 'Medium', '#4'). "
            "Verify the answer's specific wording is supported by the sources."
        ),
    )


async def verify_yarn_amount(evaluator: Evaluator, parent_node, plan: BeginnerScarfPlan) -> None:
    node = evaluator.add_sequential(
        id="Yarn_Amount_Required",
        desc="Provide the yarn yardage range required for making an average beginner scarf",
        parent=parent_node,
        critical=False,
    )

    provided = bool(plan.yarn_amount_yards and plan.yarn_amount_yards.strip())
    has_sources = bool(plan.yarn_amount_sources)
    evaluator.add_custom_node(
        result=provided and has_sources,
        id="Yarn_Amount_Provided_With_Sources",
        desc="Yarn amount (yards) is provided and includes at least one source URL",
        parent=node,
        critical=True,
    )

    verify_node = evaluator.add_leaf(
        id="Yarn_Amount_Supported_By_Sources",
        desc="Yarn amount (yards) is supported by cited sources",
        parent=node,
        critical=True,
    )
    claim_amount = (
        f"An average beginner scarf typically requires {plan.yarn_amount_yards or ''} of yarn."
    )
    await evaluator.verify(
        claim=claim_amount,
        node=verify_node,
        sources=plan.yarn_amount_sources,
        additional_instruction=(
            "Check the source content for yardage guidance for scarf projects. Accept ranges or single values if they clearly match "
            "what the answer states."
        ),
    )


async def verify_crochet_hook_size(evaluator: Evaluator, parent_node, plan: BeginnerScarfPlan) -> None:
    node = evaluator.add_sequential(
        id="Crochet_Hook_Size",
        desc="Specify the recommended crochet hook size (in mm and US size) for worsted weight yarn",
        parent=parent_node,
        critical=False,
    )

    provided = bool(plan.crochet_hook_size and plan.crochet_hook_size.strip())
    has_sources = bool(plan.crochet_hook_sources)
    evaluator.add_custom_node(
        result=provided and has_sources,
        id="Crochet_Hook_Provided_With_Sources",
        desc="Crochet hook size is provided and includes at least one source URL",
        parent=node,
        critical=True,
    )

    format_node = evaluator.add_leaf(
        id="Crochet_Hook_Format_Check",
        desc="Crochet hook size includes both mm and US size (non-critical formatting check)",
        parent=node,
        critical=False,
    )
    claim_format = (
        f"The crochet hook size '{plan.crochet_hook_size or ''}' includes both a millimeter measurement and a US hook letter/size."
    )
    await evaluator.verify(
        claim=claim_format,
        node=format_node,
        additional_instruction="Look for patterns like 'X mm (Letter-Number)' or equivalent. Allow minor punctuation differences."
    )

    verify_node = evaluator.add_leaf(
        id="Crochet_Hook_Supported_By_Sources",
        desc="Crochet hook size for worsted (#4) yarn is supported by cited sources",
        parent=node,
        critical=True,
    )
    claim_hook = (
        f"For worsted (medium, #4) yarn, the recommended crochet hook size is {plan.crochet_hook_size or ''}."
    )
    await evaluator.verify(
        claim=claim_hook,
        node=verify_node,
        sources=plan.crochet_hook_sources,
        additional_instruction=(
            "Verify that the sources recommend the stated hook size for worsted/medium/#4 yarn. Accept reasonable ranges (e.g., H-8 to I-9)."
        ),
    )


async def verify_knitting_needle_size(evaluator: Evaluator, parent_node, plan: BeginnerScarfPlan) -> None:
    node = evaluator.add_sequential(
        id="Knitting_Needle_Size",
        desc="Specify the recommended knitting needle size (US size) for worsted weight yarn",
        parent=parent_node,
        critical=False,
    )

    provided = bool(plan.knitting_needle_size and plan.knitting_needle_size.strip())
    has_sources = bool(plan.knitting_needle_sources)
    evaluator.add_custom_node(
        result=provided and has_sources,
        id="Knitting_Needle_Provided_With_Sources",
        desc="Knitting needle size is provided and includes at least one source URL",
        parent=node,
        critical=True,
    )

    format_node = evaluator.add_leaf(
        id="Knitting_Needle_Format_Check",
        desc="Knitting needle size is expressed in US size (non-critical formatting check)",
        parent=node,
        critical=False,
    )
    claim_format = (
        f"The knitting needle size '{plan.knitting_needle_size or ''}' is expressed in US size (optionally with mm in parentheses)."
    )
    await evaluator.verify(
        claim=claim_format,
        node=format_node,
        additional_instruction="Look for patterns like 'US X' optionally followed by '(Y mm)'. Allow ranges like 'US 7–9'."
    )

    verify_node = evaluator.add_leaf(
        id="Knitting_Needle_Supported_By_Sources",
        desc="Knitting needle size for worsted (#4) yarn is supported by cited sources",
        parent=node,
        critical=True,
    )
    claim_needle = (
        f"For worsted (medium, #4) yarn, the recommended knitting needle size is {plan.knitting_needle_size or ''}."
    )
    await evaluator.verify(
        claim=claim_needle,
        node=verify_node,
        sources=plan.knitting_needle_sources,
        additional_instruction=(
            "Verify that the sources recommend the stated needle size for worsted/medium/#4 yarn. Accept reasonable ranges (e.g., US 7–9)."
        ),
    )


async def verify_beginner_yarn_type(evaluator: Evaluator, parent_node, plan: BeginnerScarfPlan) -> None:
    node = evaluator.add_sequential(
        id="Beginner_Yarn_Type",
        desc="Identify the most popular yarn fiber type recommended for beginner knitters",
        parent=parent_node,
        critical=False,
    )

    provided = bool(plan.beginner_yarn_type and plan.beginner_yarn_type.strip())
    has_sources = bool(plan.beginner_yarn_type_sources)
    evaluator.add_custom_node(
        result=provided and has_sources,
        id="Yarn_Type_Provided_With_Sources",
        desc="Beginner yarn fiber type is provided and includes at least one source URL",
        parent=node,
        critical=True,
    )

    verify_node = evaluator.add_leaf(
        id="Yarn_Type_Supported_By_Sources",
        desc="Beginner yarn fiber recommendation is supported by cited sources",
        parent=node,
        critical=True,
    )
    claim_type = (
        f"The most popular yarn fiber type recommended for beginners is {plan.beginner_yarn_type or ''}."
    )
    await evaluator.verify(
        claim=claim_type,
        node=verify_node,
        sources=plan.beginner_yarn_type_sources,
        additional_instruction=(
            "Common beginner-friendly fibers include acrylic. Verify that the sources explicitly recommend the stated fiber as most suitable for beginners."
        ),
    )


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

    plan = await evaluator.extract(
        prompt=prompt_extract_beginner_scarf_plan(),
        template_class=BeginnerScarfPlan,
        extraction_name="beginner_scarf_plan",
    )

    parent = evaluator.add_parallel(
        id="Beginner_Scarf_Project_Requirements",
        desc="Complete information package for starting a beginner scarf knitting or crochet project with store availability",
        parent=root,
        critical=False,
    )

    await verify_operational_chain(evaluator, parent, plan)
    await verify_state_coverage(evaluator, parent, plan)
    await verify_yarn_weight(evaluator, parent, plan)
    await verify_yarn_amount(evaluator, parent, plan)
    await verify_crochet_hook_size(evaluator, parent, plan)
    await verify_knitting_needle_size(evaluator, parent, plan)
    await verify_beginner_yarn_type(evaluator, parent, plan)

    return evaluator.get_summary()