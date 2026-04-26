import asyncio
import logging
from typing import Optional, List, Dict, Any, Callable

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "scotus_ieepa_tariff_2026"
TASK_DESCRIPTION = (
    "On February 20, 2026, the U.S. Supreme Court issued a major ruling regarding presidential authority to impose tariffs under the International Emergency Economic Powers Act (IEEPA). "
    "Following this ruling, President Trump immediately invoked alternative statutory authority to implement new tariffs. Provide the following information: "
    "(1) The full official case name as it appears in Supreme Court documents; "
    "(2) The Supreme Court docket number for this case; "
    "(3) The date the Supreme Court decision was issued; "
    "(4) The vote breakdown among the justices (in X-Y format); "
    "(5) The name of the Justice who wrote the majority opinion; "
    "(6) The Court's holding on whether IEEPA authorizes the President to impose tariffs; "
    "(7) The specific statutory authority (statute name and section number) that President Trump invoked for implementing new tariffs immediately after the Supreme Court ruling; "
    "(8) The tariff rate (as a percentage) that ultimately took effect under this new statutory authority; "
    "(9) The effective date when the new tariffs under this alternative authority became operative."
)

MAIN_NODE_DESC = "Verification of information about the February 2026 Supreme Court tariff ruling and the subsequent Section 122 tariff implementation"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SimpleInfoWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AuthorityExtraction(BaseModel):
    statute_name: Optional[str] = None
    section_number: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SupremeCourtTariffExtraction(BaseModel):
    case_name: Optional[SimpleInfoWithSources] = None
    docket_number: Optional[SimpleInfoWithSources] = None
    decision_date: Optional[SimpleInfoWithSources] = None
    vote_breakdown: Optional[SimpleInfoWithSources] = None
    majority_author: Optional[SimpleInfoWithSources] = None
    legal_holding: Optional[SimpleInfoWithSources] = None  # Holding re: IEEPA authorizes tariffs or not
    section_122_authority: Optional[AuthorityExtraction] = None  # Alternative statutory authority invoked
    section_122_rate: Optional[SimpleInfoWithSources] = None  # Tariff rate (percentage string)
    section_122_effective_date: Optional[SimpleInfoWithSources] = None  # Effective date of new tariffs


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_scotus_tariff_info() -> str:
    return """
    Extract the requested nine items from the answer text exactly as stated by the answer. For each item, also extract any URLs cited in the answer that substantiate the item.

    Return a single JSON object with these keys and structures:

    - case_name: { "value": string|null, "sources": string[] } 
        • The full official case name as it appears in Supreme Court documents (caption/slip opinion).
        • sources: all URLs cited in the answer that support this case name.

    - docket_number: { "value": string|null, "sources": string[] } 
        • The Supreme Court docket number (e.g., "No. 23-123").
        • sources: URLs cited that show the docket number.

    - decision_date: { "value": string|null, "sources": string[] } 
        • The date the Supreme Court decision was issued (e.g., "February 20, 2026").
        • sources: URLs cited that show the decision date.

    - vote_breakdown: { "value": string|null, "sources": string[] } 
        • The numerical vote split among the justices in X-Y format (e.g., "6-3").
        • sources: URLs cited that show the vote breakdown.

    - majority_author: { "value": string|null, "sources": string[] } 
        • The name of the Justice (or Chief Justice) who authored the majority opinion.
        • sources: URLs cited that show the opinion author.

    - legal_holding: { "value": string|null, "sources": string[] }
        • A concise statement of the Court's holding regarding whether IEEPA authorizes the President to impose tariffs (e.g., "IEEPA does not authorize the President to impose tariffs").
        • sources: URLs cited that support this holding.

    - section_122_authority: { "statute_name": string|null, "section_number": string|null, "sources": string[] }
        • The specific statutory authority invoked by President Trump after the ruling: statute name (e.g., "Trade Act of 1974") and section number (e.g., "Section 122").
        • sources: URLs cited that show this authority (e.g., proclamation, Federal Register, official statements).

    - section_122_rate: { "value": string|null, "sources": string[] }
        • The tariff rate (percentage string, e.g., "10%") that ultimately took effect under this alternative authority.
        • sources: URLs cited that show the rate.

    - section_122_effective_date: { "value": string|null, "sources": string[] }
        • The effective date when the new tariffs under this alternative authority became operative.
        • sources: URLs cited that show the effective date.

    IMPORTANT:
    - Only extract URLs explicitly present in the answer text. Do not invent or infer URLs.
    - Accept plain URLs and markdown links. Extract the actual URLs behind markdown links.
    - If a field is missing, set "value" (or the specific subfield) to null; if no sources are given, return an empty array.
    - Do not normalize or reformat values beyond what the answer states (e.g., keep date strings as-is).
    """


# --------------------------------------------------------------------------- #
# Helper verification builders                                                #
# --------------------------------------------------------------------------- #
async def verify_simple_item(
    evaluator: Evaluator,
    parent_node,
    item_id: str,
    item_desc: str,
    info: Optional[SimpleInfoWithSources],
    claim_text_builder: Callable[[str], str],
    additional_instruction: str,
) -> None:
    """
    Build a sequential verification node for a single value-with-sources item:
    - Existence with sources (critical)
    - Value supported by cited sources (critical)
    """
    item_node = evaluator.add_sequential(
        id=item_id,
        desc=item_desc,
        parent=parent_node,
        critical=False,  # Non-critical at item level; allows partial credit across items
    )

    value_present = bool(info and info.value and str(info.value).strip())
    sources_present = bool(info and info.sources and len(info.sources) > 0)

    evaluator.add_custom_node(
        result=(value_present and sources_present),
        id=f"{item_id}_exists",
        desc=f"{item_desc} - value present and supported by cited sources",
        parent=item_node,
        critical=True,
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{item_id}_supported",
        desc=f"{item_desc} - supported by sources",
        parent=item_node,
        critical=True,
    )

    value_str = info.value if info and info.value else ""
    claim = claim_text_builder(value_str)

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=(info.sources if info else []),
        additional_instruction=additional_instruction,
    )


async def verify_section_122_authority(
    evaluator: Evaluator,
    parent_node,
    info: Optional[AuthorityExtraction],
) -> None:
    """
    Build the Section 122 authority verification subtree:
    - Existence with both statute_name and section_number and sources (critical)
    - Parallel critical checks: statute_name supported, section_number supported
    """
    item_id = "Section_122_Authority"
    item_desc = "The specific legal authority (statute and section number) that Trump invoked for new tariffs after the Supreme Court ruling"

    item_node = evaluator.add_sequential(
        id=item_id,
        desc=item_desc,
        parent=parent_node,
        critical=False,
    )

    value_present = bool(info and info.statute_name and str(info.statute_name).strip()) and \
                    bool(info and info.section_number and str(info.section_number).strip())
    sources_present = bool(info and info.sources and len(info.sources) > 0)

    evaluator.add_custom_node(
        result=(value_present and sources_present),
        id=f"{item_id}_exists",
        desc=f"{item_desc} - statute name and section number present, with sources",
        parent=item_node,
        critical=True,
    )

    detail_node = evaluator.add_parallel(
        id=f"{item_id}_detail_checks",
        desc=f"{item_desc} - detail verification",
        parent=item_node,
        critical=True,  # If either detail check fails, the authority item fails
    )

    # Statute name check
    statute_leaf = evaluator.add_leaf(
        id=f"{item_id}_statute_supported",
        desc=f"{item_desc} - statute name supported by sources",
        parent=detail_node,
        critical=True,
    )
    statute_val = info.statute_name if info and info.statute_name else ""
    statute_claim = f"Immediately after the Supreme Court ruling, President Trump invoked the statutory authority named '{statute_val}'."

    await evaluator.verify(
        claim=statute_claim,
        node=statute_leaf,
        sources=(info.sources if info else []),
        additional_instruction=(
            "Verify that the cited pages explicitly identify the statute name (e.g., 'Trade Act of 1974') as the authority invoked for the new tariffs."
        ),
    )

    # Section number check
    section_leaf = evaluator.add_leaf(
        id=f"{item_id}_section_supported",
        desc=f"{item_desc} - section number supported by sources",
        parent=detail_node,
        critical=True,
    )
    section_val = info.section_number if info and info.section_number else ""
    section_claim = f"The section number invoked for the new tariffs was '{section_val}'."

    await evaluator.verify(
        claim=section_claim,
        node=section_leaf,
        sources=(info.sources if info else []),
        additional_instruction=(
            "Verify that the cited pages explicitly identify the section number (e.g., 'Section 122') as the legal basis for the tariffs."
        ),
    )


# --------------------------------------------------------------------------- #
# Per-item claim builders and instructions                                    #
# --------------------------------------------------------------------------- #
def build_case_name_claim(v: str) -> str:
    return f"The full official Supreme Court case name is '{v}'."

CASE_NAME_INSTR = (
    "Check the official Supreme Court case caption or slip opinion for the full case name. "
    "Allow minor formatting variations (e.g., punctuation, capitalization, 'et al.')."
)

def build_docket_claim(v: str) -> str:
    return f"The Supreme Court docket number for the case is '{v}'."

DOCKET_INSTR = (
    "Confirm the docket number shown on the Supreme Court docket page or slip opinion (e.g., appears as 'No. 23-123')."
)

def build_decision_date_claim(v: str) -> str:
    return f"The Supreme Court issued its decision on '{v}'."

DECISION_DATE_INSTR = (
    "Verify the decision date on the official slip opinion or Supreme Court docket. "
    "Ensure this is the date of issuance/decision, not the argument date."
)

def build_vote_claim(v: str) -> str:
    return f"The vote breakdown among the justices was '{v}'."

VOTE_INSTR = (
    "Check the opinion and reliable reports for the numerical vote split (e.g., '6-3'). "
    "Minor formatting variants are acceptable (e.g., '6–3')."
)

def build_majority_author_claim(v: str) -> str:
    return f"The majority opinion was authored by Justice '{v}'."

MAJ_AUTH_INSTR = (
    "Verify that the opinion of the Court was delivered by the named Justice (or Chief Justice). "
    "Allow minor name formatting differences or inclusion/exclusion of titles."
)

def build_holding_claim(v: str) -> str:
    return f"The Court's holding regarding whether IEEPA authorizes the President to impose tariffs is: '{v}'."

HOLDING_INSTR = (
    "Confirm the opinion's core legal conclusion on whether IEEPA authorizes the President to impose tariffs. "
    "Paraphrased but equivalent statements are acceptable as long as the substance matches."
)

def build_rate_claim(v: str) -> str:
    return f"The tariff rate that ultimately took effect under the alternative authority was '{v}'."

RATE_INSTR = (
    "Verify the specific percentage rate (e.g., '10%') as stated in official materials (e.g., proclamation, Federal Register, or equivalent). "
    "Minor rounding differences are acceptable."
)

def build_effective_date_claim(v: str) -> str:
    return f"The tariffs under the alternative authority became effective on '{v}'."

EFFECTIVE_DATE_INSTR = (
    "Verify the effective date indicated in official materials (e.g., proclamation, Federal Register) for when the tariffs took effect."
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
    Evaluate an answer for the Supreme Court IEEPA/tariff ruling and subsequent Section 122 implementation.
    Returns a structured summary with the verification tree and overall score.
    """
    # Initialize evaluator with parallel root (partial credit across items)
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

    # Add a top-level domain node to mirror rubric naming
    main_node = evaluator.add_parallel(
        id="Supreme_Court_Tariff_Case_Information",
        desc=MAIN_NODE_DESC,
        parent=root,
        critical=False,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_scotus_tariff_info(),
        template_class=SupremeCourtTariffExtraction,
        extraction_name="scotus_tariff_info",
    )

    # Build verification subtrees for each requested item

    # (1) Case Name
    await verify_simple_item(
        evaluator=evaluator,
        parent_node=main_node,
        item_id="Case_Name",
        item_desc="The full official name of the Supreme Court case as it appears in court documents",
        info=extracted.case_name,
        claim_text_builder=build_case_name_claim,
        additional_instruction=CASE_NAME_INSTR,
    )

    # (2) Docket Number
    await verify_simple_item(
        evaluator=evaluator,
        parent_node=main_node,
        item_id="Case_Number",
        item_desc="The Supreme Court docket number assigned to the case",
        info=extracted.docket_number,
        claim_text_builder=build_docket_claim,
        additional_instruction=DOCKET_INSTR,
    )

    # (3) Decision Date
    await verify_simple_item(
        evaluator=evaluator,
        parent_node=main_node,
        item_id="Decision_Date",
        item_desc="The date the Supreme Court issued its decision",
        info=extracted.decision_date,
        claim_text_builder=build_decision_date_claim,
        additional_instruction=DECISION_DATE_INSTR,
    )

    # (4) Vote Breakdown
    await verify_simple_item(
        evaluator=evaluator,
        parent_node=main_node,
        item_id="Vote_Breakdown",
        item_desc="The numerical vote split among the justices (e.g., X-Y format)",
        info=extracted.vote_breakdown,
        claim_text_builder=build_vote_claim,
        additional_instruction=VOTE_INSTR,
    )

    # (5) Majority Author
    await verify_simple_item(
        evaluator=evaluator,
        parent_node=main_node,
        item_id="Majority_Author",
        item_desc="The name of the Chief Justice or Justice who wrote the majority opinion",
        info=extracted.majority_author,
        claim_text_builder=build_majority_author_claim,
        additional_instruction=MAJ_AUTH_INSTR,
    )

    # (6) Legal Holding on IEEPA Tariff Authority
    await verify_simple_item(
        evaluator=evaluator,
        parent_node=main_node,
        item_id="Legal_Holding",
        item_desc="The Court's core legal conclusion regarding whether IEEPA authorizes the President to impose tariffs",
        info=extracted.legal_holding,
        claim_text_builder=build_holding_claim,
        additional_instruction=HOLDING_INSTR,
    )

    # (7) Section 122 Authority (statute name + section number)
    await verify_section_122_authority(
        evaluator=evaluator,
        parent_node=main_node,
        info=extracted.section_122_authority,
    )

    # (8) Section 122 Rate
    await verify_simple_item(
        evaluator=evaluator,
        parent_node=main_node,
        item_id="Section_122_Rate",
        item_desc="The tariff rate (percentage) that Trump ultimately set for the Section 122 tariffs that took effect",
        info=extracted.section_122_rate,
        claim_text_builder=build_rate_claim,
        additional_instruction=RATE_INSTR,
    )

    # (9) Section 122 Effective Date
    await verify_simple_item(
        evaluator=evaluator,
        parent_node=main_node,
        item_id="Section_122_Effective_Date",
        item_desc="The date when the Section 122 tariffs became effective",
        info=extracted.section_122_effective_date,
        claim_text_builder=build_effective_date_claim,
        additional_instruction=EFFECTIVE_DATE_INSTR,
    )

    # Return the final structured summary
    return evaluator.get_summary()