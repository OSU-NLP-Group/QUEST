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
TASK_ID = "rn_licensure_4_states"
TASK_DESCRIPTION = """I am a nursing graduate preparing to apply for my Registered Nurse (RN) license and considering which state to practice in. I need comprehensive information about the initial RN licensure requirements for four states I am considering: California, Texas, Florida, and New York.

For each of these four states, please research and provide the following information:

1. Educational Requirements: What is the minimum degree or educational credential required (e.g., Associate Degree in Nursing, Bachelor of Science in Nursing)? Must the program be Board-approved or accredited?

2. Examination Requirements: What national examination must be passed to obtain licensure?

3. Application Fees: What is the exact application fee amount in USD for initial RN licensure by examination?

4. Background Check: What are the criminal background check or fingerprinting requirements?

5. Additional Coursework: Are there any state-specific additional coursework requirements (such as infection control, nursing jurisprudence, child abuse, implicit bias training, or other topics)?

6. Renewal Period: How frequently must the RN license be renewed (e.g., every 2 years, every 3 years)?

7. Continuing Education for Renewal: How many continuing education (CE) contact hours are required for license renewal?

8. Official Source: Provide a direct URL link to the official state Board of Nursing website or the official application portal for each state.

Please organize the information in a clear format that allows me to compare requirements across all four states.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StateRequirements(BaseModel):
    official_source_urls: List[str] = Field(default_factory=list)
    educational_degree_type: Optional[str] = None
    educational_program_approval: Optional[str] = None
    examination_requirement: Optional[str] = None
    application_fee_usd: Optional[str] = None
    background_check: Optional[str] = None
    additional_coursework: Optional[str] = None
    renewal_period: Optional[str] = None
    ce_hours_for_renewal: Optional[str] = None


class StatesRequirementsExtraction(BaseModel):
    california: Optional[StateRequirements] = None
    texas: Optional[StateRequirements] = None
    florida: Optional[StateRequirements] = None
    new_york: Optional[StateRequirements] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all_states() -> str:
    return """
Extract the RN initial licensure information for the following four states from the answer text: California, Texas, Florida, and New York.

For each state, extract these fields exactly as they appear in the answer:
- official_source_urls: an array of all official URLs (Board of Nursing site pages and/or official application portals) explicitly cited in the answer for that state's RN licensure info. Include only URLs that appear in the answer.
- educational_degree_type: the minimum degree or credential required (e.g., ADN, BSN, Diploma). If multiple options are listed, include them as a single descriptive string exactly as written.
- educational_program_approval: any statement about needing a Board-approved, state-approved, or accredited nursing program, exactly as written.
- examination_requirement: the national exam required (e.g., NCLEX-RN), exactly as written.
- application_fee_usd: the exact initial application fee amount for RN licensure by examination in USD, exactly as written (include $ sign if present).
- background_check: the criminal background check / fingerprint requirement description, exactly as written.
- additional_coursework: any state-specific additional coursework requirements (e.g., infection control, jurisprudence, implicit bias, child abuse) exactly as written. If the answer says none, extract that exact wording (e.g., "None", "No additional coursework").
- renewal_period: the renewal frequency, exactly as written (e.g., "every 2 years", "biennially").
- ce_hours_for_renewal: the CE contact hours required for renewal, exactly as written (e.g., "30 hours", "24 contact hours").

Return a JSON object with the following top-level keys:
- california: object with the fields above (or null if missing)
- texas: object with the fields above (or null if missing)
- florida: object with the fields above (or null if missing)
- new_york: object with the fields above (or null if missing)

If any field is missing in the answer for a state, set it to null (or an empty array for official_source_urls).
Do not invent information. Extract only what is explicitly written in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _none_like(text: Optional[str]) -> bool:
    if text is None:
        return True
    t = text.strip().lower()
    return t in {"", "none", "no", "n/a", "not required", "no additional coursework", "no coursework required"}


def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst or []


def _or_placeholder(text: Optional[str]) -> str:
    return text or ""


def _official_source_claim(state_code: str) -> str:
    # Returns a claim string asserting the provided pages are official sources for that state
    mapping = {
        "CA": "This webpage is an official resource of the California Department of Consumer Affairs Board of Registered Nursing (BRN) or another official California state government site related to RN licensure.",
        "TX": "This webpage is an official resource of the Texas Board of Nursing (BON) or another official Texas state government site related to RN licensure.",
        "FL": "This webpage is an official resource of the Florida Board of Nursing (Florida Department of Health) or another official Florida state government site related to RN licensure.",
        "NY": "This webpage is an official resource of the New York State Education Department (NYSED) Office of the Professions (Nursing) or another official New York State government site related to RN licensure.",
    }
    return mapping.get(state_code, "This webpage is an official state government or state board of nursing resource.")


def _official_source_instruction(state_code: str) -> str:
    mapping = {
        "CA": "Confirm the page branding and domain (e.g., ca.gov, rn.ca.gov, dca.ca.gov). Accept official portals like BreEZe. If it's not clearly an official California government or BRN site, mark as not supported.",
        "TX": "Confirm the page branding and domain (e.g., bon.texas.gov, texas.gov). Accept the official Texas 'Nurse Portal'. If not a Texas BON or state site, mark as not supported.",
        "FL": "Confirm the page branding and domain (e.g., floridasnursing.gov, doh.state.fl.us, flhealthsource.gov). Accept Florida Department of Health official portals. If not clearly official, mark as not supported.",
        "NY": "Confirm the page branding and domain (e.g., op.nysed.gov, nysed.gov). Accept NYSED Office of the Professions pages. If not clearly official, mark as not supported.",
    }
    return mapping.get(state_code, "Only accept clearly official state government or board of nursing websites.")


# --------------------------------------------------------------------------- #
# Verification builders per state                                             #
# --------------------------------------------------------------------------- #
async def verify_state_requirements(
    evaluator: Evaluator,
    parent_node,
    state_code: str,             # "CA" | "TX" | "FL" | "NY"
    state_label: str,            # "California" | ...
    state_node_id: str,          # "California_Requirements" | ...
    extraction: Optional[StateRequirements],
) -> None:
    """
    Build verification subtree for a specific state from the extracted content.
    """
    # Top-level node for the state (non-critical to allow partial credit across states)
    state_node = evaluator.add_parallel(
        id=state_node_id,
        desc=f"Complete RN licensure requirements for {state_label}",
        parent=parent_node,
        critical=False,
    )

    urls = _safe_list(extraction.official_source_urls if extraction else [])

    # 1) Source Verification (sequential): first ensure URL exists, then verify official
    source_seq = evaluator.add_sequential(
        id=f"{state_code}_Source_Verification",
        desc=f"Information sourced from {state_label} official resources",
        parent=state_node,
        critical=True,
    )

    # 1.a) URL existence
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"{state_code}_Source_URL",
        desc=f"Direct URL link to {state_label} Board of Nursing (or official resource) is provided",
        parent=source_seq,
        critical=True,
    )

    # 1.b) Official source verification
    n_official = evaluator.add_leaf(
        id=f"{state_code}_Official_Source",
        desc=f"Information is from official {state_label} Board/government resource",
        parent=source_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=_official_source_claim(state_code),
        node=n_official,
        sources=urls,
        additional_instruction=_official_source_instruction(state_code),
    )

    # 2) Educational Requirements (parallel)
    edu_par = evaluator.add_parallel(
        id=f"{state_code}_Educational_Requirements",
        desc=f"Minimum educational requirements for {state_label} RN licensure",
        parent=state_node,
        critical=True,
    )
    # 2.a) Degree type
    n_deg = evaluator.add_leaf(
        id=f"{state_code}_Degree_Type",
        desc="Specific degree type(s) required (e.g., ADN, BSN, diploma) is identified",
        parent=edu_par,
        critical=True,
    )
    deg_txt = _or_placeholder(extraction.educational_degree_type if extraction else None)
    await evaluator.verify(
        claim=(
            f"According to the official page(s), the minimum educational credential for initial RN licensure by examination"
            f" in {state_label} matches the answer's description: '{deg_txt}'."
        ),
        node=n_deg,
        sources=urls,
        additional_instruction="Look for phrases like 'graduate of' an approved nursing program or degree types (ADN/BSN/diploma). Allow paraphrasing and equivalent degree naming.",
    )

    # 2.b) Program approval / accreditation
    n_approval = evaluator.add_leaf(
        id=f"{state_code}_Program_Approval",
        desc="Requirement for Board-approved or accredited program is specified",
        parent=edu_par,
        critical=True,
    )
    approval_txt = _or_placeholder(extraction.educational_program_approval if extraction else None)
    await evaluator.verify(
        claim=(
            f"The official page(s) state that the nursing program must be Board-approved/state-approved and/or accredited,"
            f" consistent with the answer's description: '{approval_txt}'."
        ),
        node=n_approval,
        sources=urls,
        additional_instruction="Accept mentions of 'approved', 'state-approved', 'board-approved', or program accreditation where applicable.",
    )

    # 3) Examination Requirements (NCLEX)
    exam_grp = evaluator.add_parallel(
        id=f"{state_code}_Examination",
        desc=f"National examination requirement for {state_label}",
        parent=state_node,
        critical=True,
    )
    n_nclex = evaluator.add_leaf(
        id=f"{state_code}_NCLEX_Requirement",
        desc="Requirement to pass NCLEX-RN examination is confirmed",
        parent=exam_grp,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Applicants must pass the NCLEX-RN examination to become licensed as an RN in {state_label}.",
        node=n_nclex,
        sources=urls,
        additional_instruction="Verify any mention of NCLEX-RN for initial licensure by examination.",
    )

    # 4) Fees
    fees_grp = evaluator.add_parallel(
        id=f"{state_code}_Fees",
        desc=f"Application and licensing fees for {state_label}",
        parent=state_node,
        critical=True,
    )
    n_fee = evaluator.add_leaf(
        id=f"{state_code}_Fee_Amount",
        desc="Specific application fee amount in USD is provided",
        parent=fees_grp,
        critical=True,
    )
    fee_txt = _or_placeholder(extraction.application_fee_usd if extraction else None)
    await evaluator.verify(
        claim=(
            f"The application fee for initial RN licensure by examination in {state_label} is {fee_txt}."
        ),
        node=n_fee,
        sources=urls,
        additional_instruction="Focus on the application fee for 'by examination'. Ignore separate fingerprint/vendor fees. Allow '$' sign variations and minor formatting.",
    )

    # 5) Background Check
    bg_grp = evaluator.add_parallel(
        id=f"{state_code}_Background_Check",
        desc=f"Criminal background check requirements for {state_label}",
        parent=state_node,
        critical=True,
    )
    n_bg = evaluator.add_leaf(
        id=f"{state_code}_Background_Requirement",
        desc="Background check or fingerprinting requirement is specified",
        parent=bg_grp,
        critical=True,
    )
    bg_txt = _or_placeholder(extraction.background_check if extraction else None)
    await evaluator.verify(
        claim=(
            f"{state_label} requires a criminal background check via fingerprinting for RN initial licensure, consistent with the answer's description: '{bg_txt}'."
        ),
        node=n_bg,
        sources=urls,
        additional_instruction="Look for 'fingerprinting', 'criminal background check', 'Livescan', or similar official instructions.",
    )

    # 6) Additional Coursework
    addc_grp = evaluator.add_parallel(
        id=f"{state_code}_Additional_Coursework",
        desc=f"Additional state-specific coursework requirements for {state_label}",
        parent=state_node,
        critical=True,
    )
    n_addc = evaluator.add_leaf(
        id=f"{state_code}_Special_Courses",
        desc="Any required special coursework topics are identified",
        parent=addc_grp,
        critical=True,
    )
    addc_txt = extraction.additional_coursework if extraction else None
    if _none_like(addc_txt):
        claim_addc = f"There are no explicit state-specific additional coursework requirements for initial RN licensure in {state_label}."
        addc_instruction = "If the official page lists additional required courses for initial licensure, mark this as not supported. If none are listed, support the claim."
    else:
        claim_addc = (
            f"{state_label} requires specific additional coursework or training consistent with the answer's description: '{addc_txt}'."
        )
        addc_instruction = "Allow paraphrases and closely matching topic names (e.g., infection control, jurisprudence, implicit bias, child abuse)."
    await evaluator.verify(
        claim=claim_addc,
        node=n_addc,
        sources=urls,
        additional_instruction=addc_instruction,
    )

    # 7) Renewal requirements (period + CE hours)
    ren_par = evaluator.add_parallel(
        id=f"{state_code}_Renewal_Requirements",
        desc=f"License renewal requirements for {state_label}",
        parent=state_node,
        critical=True,
    )
    # 7.a) Renewal period
    n_ren = evaluator.add_leaf(
        id=f"{state_code}_Renewal_Period",
        desc="License renewal period/frequency is specified",
        parent=ren_par,
        critical=True,
    )
    ren_txt = _or_placeholder(extraction.renewal_period if extraction else None)
    await evaluator.verify(
        claim=(
            f"In {state_label}, the RN license renewal frequency matches the answer's description: '{ren_txt}'."
        ),
        node=n_ren,
        sources=urls,
        additional_instruction="Accept equivalent expressions (e.g., 'every 2 years' equals 'biennially').",
    )

    # 7.b) CE hours
    n_ce = evaluator.add_leaf(
        id=f"{state_code}_CE_Hours",
        desc="Continuing education (CE) hour requirements for renewal are specified",
        parent=ren_par,
        critical=True,
    )
    ce_txt = _or_placeholder(extraction.ce_hours_for_renewal if extraction else None)
    await evaluator.verify(
        claim=(
            f"For RN license renewal in {state_label}, the required continuing education matches the answer's description: '{ce_txt}'."
        ),
        node=n_ce,
        sources=urls,
        additional_instruction="Focus on total CE/contact hours for RN renewal. Allow variations like 'contact hours', 'CEUs' with equivalent conversion if explicitly stated.",
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
    Evaluate an answer for the RN initial licensure requirements across CA, TX, FL, and NY.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Parallel across states
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

    # Extract all four states' data at once
    extracted: StatesRequirementsExtraction = await evaluator.extract(
        prompt=prompt_extract_all_states(),
        template_class=StatesRequirementsExtraction,
        extraction_name="states_requirements",
    )

    # Build and verify per-state subtrees
    await verify_state_requirements(
        evaluator=evaluator,
        parent_node=root,
        state_code="CA",
        state_label="California",
        state_node_id="California_Requirements",
        extraction=extracted.california,
    )

    await verify_state_requirements(
        evaluator=evaluator,
        parent_node=root,
        state_code="TX",
        state_label="Texas",
        state_node_id="Texas_Requirements",
        extraction=extracted.texas,
    )

    await verify_state_requirements(
        evaluator=evaluator,
        parent_node=root,
        state_code="FL",
        state_label="Florida",
        state_node_id="Florida_Requirements",
        extraction=extracted.florida,
    )

    await verify_state_requirements(
        evaluator=evaluator,
        parent_node=root,
        state_code="NY",
        state_label="New York",
        state_node_id="NewYork_Requirements",
        extraction=extracted.new_york,
    )

    return evaluator.get_summary()