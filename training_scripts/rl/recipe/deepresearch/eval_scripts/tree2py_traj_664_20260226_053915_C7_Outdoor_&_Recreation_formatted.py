import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_state_outdoor_digital_id_guide"
TASK_DESCRIPTION = (
    "Identify a US state that meets ALL of the following requirements for a comprehensive outdoor recreation "
    "and modern travel infrastructure guide:\n\n"
    "1. The state must have TSA-approved digital ID capability (as of 2024-2026)\n"
    "2. The state's digital ID must be compatible with Apple Wallet\n"
    "3. The state's digital ID must be compatible with Google Wallet\n"
    "4. The state must have its own dedicated state-issued digital ID application (DMV app or similar)\n"
    "5. The state must have at least one major international airport where TSA accepts digital IDs\n"
    "6. The state must contain at least one National Park managed by the National Park Service\n"
    "7. The state must have at least one Six Flags theme park location\n"
    "8. The state must offer beach tourism destinations (ocean or coastal beaches)\n"
    "9. The state must have ski resort facilities available to the public\n"
    "10. Your answer must include the correct current cost of the America the Beautiful annual pass for US residents "
    "(effective January 1, 2026)\n\n"
    "Additionally, if possible, mention whether the state has:\n"
    "- Water park facilities\n"
    "- Camping facilities in national forests or parks\n"
    "- Established hiking trails on public lands\n\n"
    "Name the state and provide specific examples or details for each requirement."
)

EXPECTED_PASS_COST_USD = "$80"  # Ground truth target for the America's Beautiful annual pass cost


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CriterionData(BaseModel):
    examples: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class StateCriteriaExtraction(BaseModel):
    state_name: Optional[str] = None

    tsa_digital_id: CriterionData = Field(default_factory=CriterionData)
    apple_wallet: CriterionData = Field(default_factory=CriterionData)
    google_wallet: CriterionData = Field(default_factory=CriterionData)
    state_digital_id_app: CriterionData = Field(default_factory=CriterionData)
    airport_digital_id: CriterionData = Field(default_factory=CriterionData)
    national_park: CriterionData = Field(default_factory=CriterionData)
    six_flags: CriterionData = Field(default_factory=CriterionData)
    beach: CriterionData = Field(default_factory=CriterionData)
    ski_resort: CriterionData = Field(default_factory=CriterionData)

    pass_cost: Optional[str] = None
    pass_cost_sources: List[str] = Field(default_factory=list)

    water_park: CriterionData = Field(default_factory=CriterionData)
    camping: CriterionData = Field(default_factory=CriterionData)
    hiking: CriterionData = Field(default_factory=CriterionData)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_state_criteria() -> str:
    return (
        "Extract structured information from the answer for a single identified US state. "
        "If multiple states are mentioned, choose the first one that is explicitly presented as satisfying the criteria. "
        "Return the following fields:\n"
        "- state_name: the primary US state identified by the answer.\n"
        "- For each criterion below, extract:\n"
        "  • examples: a list of specific examples or named entities (e.g., airport name, park name, beach, ski resort, app name) "
        "mentioned for that criterion. If none are provided, return an empty list.\n"
        "  • sources: a list of URLs explicitly cited in the answer that support this criterion. If none are provided, return an empty list.\n"
        "Criteria keys to fill:\n"
        "  tsa_digital_id, apple_wallet, google_wallet, state_digital_id_app, airport_digital_id, "
        "  national_park, six_flags, beach, ski_resort, water_park, camping, hiking.\n"
        "- pass_cost: the America the Beautiful annual pass cost as explicitly stated in the answer (string, such as \"$80\"). "
        "If not stated, return null.\n"
        "- pass_cost_sources: a list of URLs cited in the answer that support the pass cost. If none are provided, return an empty list.\n\n"
        "Rules:\n"
        "1) Extract only URLs that appear in the answer (including markdown links). Do not invent any.\n"
        "2) Always include full URLs. If a URL is missing protocol, prepend http://.\n"
        "3) Do not perform any verification; just extract faithfully.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_example(examples: List[str]) -> str:
    return examples[0] if examples else ""


async def _add_seq_criterion_with_sources(
    evaluator: Evaluator,
    parent_node,
    *,
    node_id: str,
    desc: str,
    sources: List[str],
    claim: str,
    additional_instruction: str,
    critical: bool = True,
) -> None:
    """
    Create a sequential verification node that first checks for source existence,
    then verifies the claim using those sources.
    """
    seq_node = evaluator.add_sequential(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )

    # Gate: must have at least one source URL
    sources_exist = bool(sources)
    evaluator.add_custom_node(
        result=sources_exist,
        id=f"{node_id}_sources_exist",
        desc=f"Sources provided for: {desc}",
        parent=seq_node,
        critical=True,
    )

    # Leaf: verify claim by URLs
    leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=desc,
        parent=seq_node,
        critical=True,
    )

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources if sources_exist else None,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification sub-tree builders                                              #
# --------------------------------------------------------------------------- #
async def build_mandatory_criteria(
    evaluator: Evaluator,
    root,
    data: StateCriteriaExtraction,
) -> None:
    """
    Build and verify all mandatory criteria (1-10).
    """
    # Add a critical group node for mandatory criteria
    mandatory_group = evaluator.add_parallel(
        id="mandatory_criteria",
        desc="All required criteria must be satisfied",
        parent=root,
        critical=True,
    )

    state = data.state_name or "the state"

    # 1. TSA-approved digital ID capability
    await _add_seq_criterion_with_sources(
        evaluator,
        mandatory_group,
        node_id="criterion_1",
        desc="The state has TSA-approved digital ID capability",
        sources=data.tsa_digital_id.sources,
        claim=f"TSA accepts this state's digital ID / mobile driver's license program for identity verification at airport checkpoints in {state}.",
        additional_instruction=(
            "Determine whether TSA has approved acceptance of this state's digital ID (mDL) at airport checkpoints. "
            "Use official TSA or credible sources. Allow reasonable wording variations like 'accepted', 'approved', or 'supported'."
        ),
        critical=True,
    )

    # 2. Apple Wallet compatibility
    await _add_seq_criterion_with_sources(
        evaluator,
        mandatory_group,
        node_id="criterion_2",
        desc="The state's digital ID is compatible with Apple Wallet",
        sources=data.apple_wallet.sources,
        claim=f"This state's digital driver's license or digital ID is available in Apple Wallet for residents of {state}.",
        additional_instruction=(
            "Verify that Apple Wallet supports the state's driver's license or digital ID. "
            "Look for Apple Wallet pages, state DMV announcements, or major tech press confirming Apple Wallet availability for this state."
        ),
        critical=True,
    )

    # 3. Google Wallet compatibility
    await _add_seq_criterion_with_sources(
        evaluator,
        mandatory_group,
        node_id="criterion_3",
        desc="The state's digital ID is compatible with Google Wallet",
        sources=data.google_wallet.sources,
        claim=f"This state's digital driver's license or digital ID is available in Google Wallet for residents of {state}.",
        additional_instruction=(
            "Verify that Google Wallet supports the state's driver's license or digital ID. "
            "Look for Google Wallet pages, state DMV announcements, or credible reporting confirming Google Wallet availability."
        ),
        critical=True,
    )

    # 4. Dedicated DMV/state-issued digital ID application
    app_example = _first_example(data.state_digital_id_app.examples)
    await _add_seq_criterion_with_sources(
        evaluator,
        mandatory_group,
        node_id="criterion_4",
        desc="The state has its own dedicated DMV or state-issued digital ID application",
        sources=data.state_digital_id_app.sources,
        claim=(
            f"{state} provides an official DMV or state-issued digital ID application "
            f"{('such as ' + app_example) if app_example else ''}."
        ),
        additional_instruction=(
            "Confirm the existence of an official DMV/state digital ID app (or equivalent), "
            "issued by the state government or DMV. The app should be state-backed."
        ),
        critical=True,
    )

    # 5. Major international airport accepts digital IDs
    airport_example = _first_example(data.airport_digital_id.examples)
    await _add_seq_criterion_with_sources(
        evaluator,
        mandatory_group,
        node_id="criterion_5",
        desc="The state has at least one major international airport that accepts digital IDs",
        sources=data.airport_digital_id.sources,
        claim=(
            f"There is at least one major international airport in {state} "
            f"{('such as ' + airport_example) if airport_example else ''} "
            "where TSA accepts digital IDs/mDLs."
        ),
        additional_instruction=(
            "Confirm TSA acceptance of digital IDs/mDLs at a major international airport in the state. "
            "Airport or TSA pages, or credible news/press releases are acceptable."
        ),
        critical=True,
    )

    # 6. At least one National Park managed by NPS
    park_example = _first_example(data.national_park.examples)
    await _add_seq_criterion_with_sources(
        evaluator,
        mandatory_group,
        node_id="criterion_6",
        desc="The state contains at least one National Park managed by the National Park Service",
        sources=data.national_park.sources,
        claim=(
            f"{state} contains at least one National Park managed by the National Park Service "
            f"{('for example, ' + park_example) if park_example else ''}."
        ),
        additional_instruction=(
            "Verify that at least one named location is an NPS-managed National Park in the specified state. "
            "Use NPS.gov or other authoritative references."
        ),
        critical=True,
    )

    # 7. At least one Six Flags theme park
    six_flags_example = _first_example(data.six_flags.examples)
    await _add_seq_criterion_with_sources(
        evaluator,
        mandatory_group,
        node_id="criterion_7",
        desc="The state has at least one Six Flags theme park location",
        sources=data.six_flags.sources,
        claim=(
            f"{state} has a Six Flags theme park "
            f"{('such as ' + six_flags_example) if six_flags_example else ''}."
        ),
        additional_instruction=(
            "Verify the existence of a Six Flags park located within the state. "
            "Use Six Flags official site pages or credible location sources."
        ),
        critical=True,
    )

    # 8. Beach tourism destinations (ocean/coastal)
    beach_example = _first_example(data.beach.examples)
    await _add_seq_criterion_with_sources(
        evaluator,
        mandatory_group,
        node_id="criterion_8",
        desc="The state has established beach tourism destinations",
        sources=data.beach.sources,
        claim=(
            f"{state} offers ocean or coastal beach tourism destinations "
            f"{('such as ' + beach_example) if beach_example else ''}."
        ),
        additional_instruction=(
            "Confirm that the state has ocean/coastal beaches suitable for tourism. "
            "Use tourism guides, official state tourism pages, or reputable travel sources."
        ),
        critical=True,
    )

    # 9. Ski resort facilities available to the public
    ski_example = _first_example(data.ski_resort.examples)
    await _add_seq_criterion_with_sources(
        evaluator,
        mandatory_group,
        node_id="criterion_9",
        desc="The state has ski resort facilities",
        sources=data.ski_resort.sources,
        claim=(
            f"{state} has public ski resort facilities "
            f"{('such as ' + ski_example) if ski_example else ''}."
        ),
        additional_instruction=(
            "Verify that the state has at least one public ski resort. "
            "Use resort websites or reputable travel/recreation sources."
        ),
        critical=True,
    )

    # 10. America the Beautiful pass cost is $80; verify both answer content and source support
    # Build a sequential node with two critical leaves:
    # (a) The answer states $80; (b) sources support $80 (as of Jan 1, 2026).
    criterion10 = evaluator.add_sequential(
        id="criterion_10",
        desc="The answer correctly states the America the Beautiful annual pass cost for US residents as $80",
        parent=mandatory_group,
        critical=True,
    )

    has_pass_info = (data.pass_cost is not None and str(data.pass_cost).strip() != "") and bool(data.pass_cost_sources)
    evaluator.add_custom_node(
        result=has_pass_info,
        id="criterion_10_data_provided",
        desc="The answer includes a stated pass cost and provides at least one supporting source URL",
        parent=criterion10,
        critical=True,
    )

    # (a) Check the answer text itself says $80 (simple verify, no sources)
    leaf_answer_cost = evaluator.add_leaf(
        id="criterion_10_answer_cost_correct",
        desc="The answer text states the annual pass costs $80 (effective Jan 1, 2026)",
        parent=criterion10,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer explicitly states that the America the Beautiful annual pass costs $80 for US residents "
            "(effective January 1, 2026)."
        ),
        node=leaf_answer_cost,
        sources=None,
        additional_instruction=(
            "Inspect the provided answer and decide if it clearly states $80 as the cost. "
            "Minor formatting variants like 'USD 80' or '$80.00' should count as $80."
        ),
    )

    # (b) Verify by URLs that $80 is correct
    leaf_cost_supported = evaluator.add_leaf(
        id="criterion_10_cost_supported",
        desc="The $80 annual pass cost is supported by cited sources (effective Jan 1, 2026)",
        parent=criterion10,
        critical=True,
    )
    await evaluator.verify(
        claim="The America the Beautiful annual pass costs $80 for US residents, effective January 1, 2026.",
        node=leaf_cost_supported,
        sources=data.pass_cost_sources if has_pass_info else None,
        additional_instruction=(
            "Confirm using official sources (e.g., NPS.gov or USGS Store) or other credible references "
            "that the annual pass price is $80 as of Jan 1, 2026."
        ),
    )


async def build_optional_criteria(
    evaluator: Evaluator,
    root,
    data: StateCriteriaExtraction,
) -> None:
    """
    Build and verify optional criteria (11-13). These allow partial credit.
    """
    optional_group = evaluator.add_parallel(
        id="optional_criteria",
        desc="Optional recreation criteria (partial credit)",
        parent=root,
        critical=False,
    )

    state = data.state_name or "the state"

    # 11. Water park facilities
    water_example = _first_example(data.water_park.examples)
    await _add_seq_criterion_with_sources(
        evaluator,
        optional_group,
        node_id="criterion_11",
        desc="The state has water park facilities",
        sources=data.water_park.sources,
        claim=(
            f"{state} has water park facilities "
            f"{('such as ' + water_example) if water_example else ''}."
        ),
        additional_instruction=(
            "Verify existence of at least one water park in the state using park/operator websites or reputable sources."
        ),
        critical=False,
    )

    # 12. Camping in national forests or parks
    camp_example = _first_example(data.camping.examples)
    await _add_seq_criterion_with_sources(
        evaluator,
        optional_group,
        node_id="criterion_12",
        desc="The state has designated camping facilities in national forests or parks",
        sources=data.camping.sources,
        claim=(
            f"{state} provides designated camping facilities in national forests or parks "
            f"{('for example, ' + camp_example) if camp_example else ''}."
        ),
        additional_instruction=(
            "Confirm that camping is available in national forests or parks in the state. "
            "Use USFS/NPS or official park pages, or other credible camping references."
        ),
        critical=False,
    )

    # 13. Established hiking trails on public lands
    hike_example = _first_example(data.hiking.examples)
    await _add_seq_criterion_with_sources(
        evaluator,
        optional_group,
        node_id="criterion_13",
        desc="The state has established hiking trails in public lands",
        sources=data.hiking.sources,
        claim=(
            f"{state} has established hiking trails on public lands "
            f"{('such as ' + hike_example) if hike_example else ''}."
        ),
        additional_instruction=(
            "Verify that the state has established hiking trails on public lands. "
            "Use state parks, NPS, USFS, or reputable trail databases."
        ),
        critical=False,
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
    Evaluate an agent's answer for the US state outdoor recreation and digital ID infrastructure task.
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
        default_model=model,
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_state_criteria(),
        template_class=StateCriteriaExtraction,
        extraction_name="state_criteria_extraction",
    )

    # Record ground truth info for pass cost expectation
    evaluator.add_ground_truth({
        "expected_america_the_beautiful_annual_pass_cost_2026": EXPECTED_PASS_COST_USD
    })

    # Add a critical precondition: state must be identified
    state_exists_node = evaluator.add_custom_node(
        result=bool(extracted.state_name and extracted.state_name.strip()),
        id="state_identified",
        desc="State name is identified in the answer",
        parent=root,
        critical=True,
    )

    # Build mandatory and optional criteria
    await build_mandatory_criteria(evaluator, root, extracted)
    await build_optional_criteria(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()