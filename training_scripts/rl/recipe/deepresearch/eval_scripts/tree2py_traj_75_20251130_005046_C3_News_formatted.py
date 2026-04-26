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
TASK_ID = "ofac_sebin_cousin_2024_11_27"
TASK_DESCRIPTION = (
    "On November 27, 2024, the U.S. Department of the Treasury's Office of Foreign Assets Control (OFAC) "
    "designated 21 Venezuelan officials under Executive Order 13692 in response to actions related to the July 28, 2024 Venezuelan presidential election. "
    "Among these sanctioned officials, one individual serves as the director of SEBIN (Servicio Bolivariano de Inteligencia Nacional - Bolivarian National Intelligence Service). "
    "This individual is also identified as being the cousin of another Venezuelan official who has been under OFAC sanctions since 2018 and currently holds the position of Minister of Interior, Justice, and Peace under the Maduro administration. "
    "Identify this Venezuelan official and provide the following information: "
    "(1) The individual's full name exactly as it appears on OFAC's Specially Designated Nationals (SDN) List; "
    "(2) Confirmation that the individual serves as director of SEBIN; "
    "(3) Confirmation of the family relationship (cousin) to the minister sanctioned in 2018, including the minister's name; "
    "(4) Confirmation that the designation occurred on November 27, 2024 under Executive Order 13692; "
    "(5) The individual's Venezuelan cédula (national identification) number as recorded in OFAC's designation records. "
    "Provide reference URLs from official OFAC sources or U.S. Treasury Department press releases to support each element of your answer."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OFACTaskExtraction(BaseModel):
    # Identity
    individual_full_name: Optional[str] = None
    sdn_urls: List[str] = Field(default_factory=list)  # Official OFAC/Treasury SDN entry or related official pages

    # SEBIN role
    sebin_sources: List[str] = Field(default_factory=list)

    # Cousin relationship and minister identity
    cousin_minister_name: Optional[str] = None
    relationship_sources: List[str] = Field(default_factory=list)

    # Minister sanctioned since 2018 and position
    minister_sources: List[str] = Field(default_factory=list)

    # Designation date and Executive Order
    designation_date: Optional[str] = None
    executive_order: Optional[str] = None
    designation_sources: List[str] = Field(default_factory=list)

    # Action context (July 28, 2024 election) and count 21 officials
    action_context_sources: List[str] = Field(default_factory=list)
    count_sources: List[str] = Field(default_factory=list)

    # Cédula number
    cedula_number: Optional[str] = None
    cedula_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ofac_task() -> str:
    return """
    Extract the details provided in the answer about the OFAC-designated Venezuelan official described in the task. 
    Return a JSON object with the following fields; if any field is not mentioned, set it to null or [] accordingly.

    Required fields:
    1. individual_full_name: The individual's full name exactly as provided in the answer (intended to match the OFAC SDN List primary name).
    2. sdn_urls: A list of URL(s) from official OFAC or U.S. Treasury websites that are cited for the SDN entry or for the individual's sanctioned status (e.g., SDN search entry, OFAC page, Treasury press release). Extract only URLs explicitly present in the answer.

    3. sebin_sources: A list of URL(s) from official OFAC or U.S. Treasury websites cited to confirm that the identified individual serves as the director of SEBIN.

    4. cousin_minister_name: The minister-relative’s full name as stated in the answer.
    5. relationship_sources: A list of URL(s) from official OFAC or U.S. Treasury websites that explicitly confirm the cousin relationship between the identified individual and the named minister.

    6. minister_sources: A list of URL(s) from official OFAC or U.S. Treasury websites that confirm BOTH:
       - the minister-relative has been under OFAC sanctions since 2018, and
       - the minister-relative holds the position of Minister of Interior, Justice, and Peace (under the Maduro administration).

    7. designation_date: The designation date provided in the answer (preferably 'November 27, 2024' or '2024-11-27').
    8. executive_order: The executive order number provided (e.g., 'Executive Order 13692' or 'E.O. 13692').
    9. designation_sources: A list of URL(s) from official OFAC or U.S. Treasury websites used to support the designation date and EO claim.

    10. action_context_sources: A list of URL(s) from official OFAC or U.S. Treasury websites that confirm the Nov 27, 2024 designations were in response to actions related to the July 28, 2024 Venezuelan presidential election.

    11. count_sources: A list of URL(s) from official OFAC or U.S. Treasury websites that confirm 21 Venezuelan officials were designated on Nov 27, 2024.

    12. cedula_number: The individual's Venezuelan cédula (national ID) number exactly as provided in the answer.
    13. cedula_sources: A list of URL(s) from official OFAC or U.S. Treasury websites that show the cédula number as recorded in OFAC designation or SDN records.

    IMPORTANT RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer; do not infer or fabricate URLs.
    - Include full URLs (with protocol), and prefer official sources on the domains 'treasury.gov' or 'treas.gov' (e.g., home.treasury.gov, ofac.treasury.gov, sanctionssearch.ofac.treas.gov).
    - If the answer references a source without a URL (e.g., 'according to OFAC press release'), return an empty list for that corresponding sources field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_official_treasury_url(url: str) -> bool:
    """Heuristic check: treat URLs containing treasury.gov or treas.gov as official."""
    if not isinstance(url, str):
        return False
    u = url.lower().strip()
    return ("treasury.gov" in u) or ("treas.gov" in u)


def all_official_urls(urls: List[str]) -> bool:
    """Return True if every URL in list is official Treasury/OFAC."""
    if not urls:
        return False
    return all(is_official_treasury_url(u) for u in urls)


def ensure_sources_node(
    evaluator: Evaluator, parent, node_id: str, desc: str, urls: List[str], require_official: bool = True
):
    """
    Add a critical existence check node for sources presence and official domain check.
    """
    # Presence check
    presence_result = bool(urls)
    evaluator.add_custom_node(
        result=presence_result,
        id=f"{node_id}_sources_present",
        desc=f"{desc} - official source URLs are present in the answer",
        parent=parent,
        critical=True,
    )
    # Official domain check (critical)
    official_result = all_official_urls(urls) if require_official else presence_result
    evaluator.add_custom_node(
        result=official_result,
        id=f"{node_id}_sources_official",
        desc=f"{desc} - provided sources are official OFAC/Treasury domains",
        parent=parent,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_identify_individual(
    evaluator: Evaluator,
    parent_node,
    ex: OFACTaskExtraction,
):
    """
    Build the 'Identify_Individual' subtree:
    - Require individual's full name and official SDN/OFAC/Treasury URLs.
    - Verify the name exactly as appears on OFAC SDN List supported by provided URLs.
    """
    node = evaluator.add_parallel(
        id="Identify_Individual",
        desc="Identify the specific sanctioned individual and give their full name exactly as shown on the OFAC SDN List (with official OFAC/Treasury URL supporting the SDN entry/name).",
        parent=parent_node,
        critical=True,
    )

    # Basic existence check for name
    name_present = bool(ex.individual_full_name and ex.individual_full_name.strip())
    evaluator.add_custom_node(
        result=name_present,
        id="Identify_Individual_name_present",
        desc="Individual's full name is provided in the answer",
        parent=node,
        critical=True,
    )

    # Sources presence + official check
    ensure_sources_node(
        evaluator,
        node,
        node_id="Identify_Individual",
        desc="SDN entry/name confirmation",
        urls=ex.sdn_urls,
        require_official=True,
    )

    # Leaf: exact name verification against OFAC/Treasury URLs
    name_verify_leaf = evaluator.add_leaf(
        id="Identify_Individual_name_exact_match",
        desc="The individual's full name appears EXACTLY as provided on OFAC's SDN List",
        parent=node,
        critical=True,
    )
    name_claim = (
        f"The individual's SDN List primary name is exactly '{ex.individual_full_name}'. "
        f"Treat any difference in punctuation, accents/diacritics, spacing, or letter case as a mismatch."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_verify_leaf,
        sources=ex.sdn_urls,
        additional_instruction=(
            "Use the official OFAC SDN entry page to confirm the primary name. "
            "Check the printed 'Name' field; aliases should not count. "
            "If the page shows a different spelling or formatting, mark incorrect."
        ),
    )


async def verify_provide_confirmations_and_id(
    evaluator: Evaluator,
    parent_node,
    ex: OFACTaskExtraction,
):
    """
    Build the 'Provide_Required_Confirmations_And_ID' subtree with parallel children:
    - Confirm designation date and EO 13692 (split into two leaves).
    - Confirm action context re: July 28, 2024 election.
    - Confirm count of 21 officials.
    - Confirm SEBIN director role.
    - Confirm cousin relationship (and minister's name).
    - Confirm minister sanctioned since 2018 and position.
    - Provide cédula number (verify with OFAC/SDN sources).
    """
    main = evaluator.add_parallel(
        id="Provide_Required_Confirmations_And_ID",
        desc="Provide each required confirmation/detail about the identified individual, each supported by official OFAC/Treasury sources.",
        parent=parent_node,
        critical=True,
    )

    # 1) Confirm Designation Date and EO (split into date + EO)
    cde = evaluator.add_parallel(
        id="Confirm_Designation_Date_And_EO",
        desc="Confirms the individual’s designation occurred on November 27, 2024 and was pursuant to Executive Order 13692, with official OFAC/Treasury URL(s).",
        parent=main,
        critical=True,
    )
    # Existence checks
    # Date presence
    date_present = bool(ex.designation_date and ex.designation_date.strip())
    evaluator.add_custom_node(
        result=date_present,
        id="Confirm_Designation_Date_And_EO_date_present",
        desc="Designation date is provided in the answer",
        parent=cde,
        critical=True,
    )
    # EO presence
    eo_present = bool(ex.executive_order and ex.executive_order.strip())
    evaluator.add_custom_node(
        result=eo_present,
        id="Confirm_Designation_Date_And_EO_eo_present",
        desc="Executive Order number is provided in the answer",
        parent=cde,
        critical=True,
    )
    # Sources presence + official
    ensure_sources_node(
        evaluator,
        cde,
        node_id="Confirm_Designation_Date_And_EO",
        desc="Designation date and EO confirmation",
        urls=ex.designation_sources,
        require_official=True,
    )
    # Leaf: date verification
    date_leaf = evaluator.add_leaf(
        id="Confirm_Designation_Date_And_EO_date_verified",
        desc="Designation occurred on November 27, 2024",
        parent=cde,
        critical=True,
    )
    date_claim = (
        f"The individual {ex.individual_full_name} was designated on November 27, 2024."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=ex.designation_sources,
        additional_instruction=(
            "Verify the date shown on the official OFAC/Treasury page states November 27, 2024 "
            "and explicitly references the individual by name."
        ),
    )
    # Leaf: EO verification
    eo_leaf = evaluator.add_leaf(
        id="Confirm_Designation_Date_And_EO_eo_verified",
        desc="Designation was pursuant to Executive Order 13692",
        parent=cde,
        critical=True,
    )
    eo_claim = (
        f"The individual's designation was pursuant to Executive Order 13692."
    )
    await evaluator.verify(
        claim=eo_claim,
        node=eo_leaf,
        sources=ex.designation_sources,
        additional_instruction=(
            "Confirm that the official page explicitly cites Executive Order 13692 for the individual's designation."
        ),
    )

    # 2) Confirm Action Context: response to July 28, 2024 election
    cac = evaluator.add_parallel(
        id="Confirm_Action_Context_July28_Election",
        desc="Confirms (via official OFAC/Treasury source) that the Nov 27, 2024 designations were in response to actions related to the July 28, 2024 Venezuelan presidential election.",
        parent=main,
        critical=True,
    )
    ensure_sources_node(
        evaluator,
        cac,
        node_id="Confirm_Action_Context_July28_Election",
        desc="Action context (July 28, 2024 election) confirmation",
        urls=ex.action_context_sources,
        require_official=True,
    )
    action_leaf = evaluator.add_leaf(
        id="Confirm_Action_Context_July28_Election_verified",
        desc="Nov 27, 2024 designations were in response to actions related to the July 28, 2024 Venezuelan presidential election",
        parent=cac,
        critical=True,
    )
    action_claim = (
        "The Nov 27, 2024 OFAC designations were in response to actions related to the July 28, 2024 Venezuelan presidential election."
    )
    await evaluator.verify(
        claim=action_claim,
        node=action_leaf,
        sources=ex.action_context_sources,
        additional_instruction="Verify the official press release or OFAC/Treasury page states this context clearly.",
    )

    # 3) Confirm Count: 21 officials
    cco = evaluator.add_parallel(
        id="Confirm_Count_21_Officials",
        desc="Confirms (via official OFAC/Treasury source) that 21 Venezuelan officials were designated in the Nov 27, 2024 action.",
        parent=main,
        critical=True,
    )
    ensure_sources_node(
        evaluator,
        cco,
        node_id="Confirm_Count_21_Officials",
        desc="Count of designated officials confirmation",
        urls=ex.count_sources,
        require_official=True,
    )
    count_leaf = evaluator.add_leaf(
        id="Confirm_Count_21_Officials_verified",
        desc="21 Venezuelan officials were designated on Nov 27, 2024",
        parent=cco,
        critical=True,
    )
    count_claim = "There were 21 Venezuelan officials designated on November 27, 2024."
    await evaluator.verify(
        claim=count_claim,
        node=count_leaf,
        sources=ex.count_sources,
        additional_instruction="Verify the official page explicitly states the count is 21.",
    )

    # 4) Confirm SEBIN Director Role
    sdr = evaluator.add_parallel(
        id="Confirm_SEBIN_Director_Role",
        desc="Confirms the identified individual serves as director of SEBIN, with official OFAC/Treasury URL(s).",
        parent=main,
        critical=True,
    )
    ensure_sources_node(
        evaluator,
        sdr,
        node_id="Confirm_SEBIN_Director_Role",
        desc="SEBIN director role confirmation",
        urls=ex.sebin_sources,
        require_official=True,
    )
    sebin_leaf = evaluator.add_leaf(
        id="Confirm_SEBIN_Director_Role_verified",
        desc="The identified individual serves as director of SEBIN",
        parent=sdr,
        critical=True,
    )
    sebin_claim = f"{ex.individual_full_name} serves as the director of SEBIN (Servicio Bolivariano de Inteligencia Nacional)."
    await evaluator.verify(
        claim=sebin_claim,
        node=sebin_leaf,
        sources=ex.sebin_sources,
        additional_instruction="Confirm that the official sources explicitly state the individual's SEBIN director role.",
    )

    # 5) Confirm Cousin Relationship and Minister Name
    crm = evaluator.add_parallel(
        id="Confirm_Cousin_Relationship_And_Minister_Name",
        desc="Confirms the identified individual is a cousin of the minister, and provides the minister’s name, with official OFAC/Treasury URL(s).",
        parent=main,
        critical=True,
    )
    # Minister name presence
    minister_name_present = bool(ex.cousin_minister_name and ex.cousin_minister_name.strip())
    evaluator.add_custom_node(
        result=minister_name_present,
        id="Confirm_Cousin_Relationship_And_Minister_Name_minister_name_present",
        desc="Minister-relative's name is provided in the answer",
        parent=crm,
        critical=True,
    )
    # Sources presence + official
    # Use relationship_sources (primary) and also allow minister_sources as supplemental
    combined_rel_sources = list(dict.fromkeys((ex.relationship_sources or []) + (ex.minister_sources or [])))
    ensure_sources_node(
        evaluator,
        crm,
        node_id="Confirm_Cousin_Relationship_And_Minister_Name",
        desc="Cousin relationship and minister identity confirmation",
        urls=combined_rel_sources,
        require_official=True,
    )
    # Leaf: cousin relationship
    cousin_leaf = evaluator.add_leaf(
        id="Confirm_Cousin_Relationship_And_Minister_Name_cousin_verified",
        desc="The identified individual is a cousin of the named minister",
        parent=crm,
        critical=True,
    )
    cousin_claim = f"{ex.individual_full_name} is a cousin of {ex.cousin_minister_name}."
    await evaluator.verify(
        claim=cousin_claim,
        node=cousin_leaf,
        sources=combined_rel_sources,
        additional_instruction="Confirm the familial relationship 'cousin' between the two named individuals on official OFAC/Treasury sources.",
    )
    # Leaf: minister identity (name)
    minister_name_leaf = evaluator.add_leaf(
        id="Confirm_Cousin_Relationship_And_Minister_Name_minister_name_verified",
        desc="The minister-relative's name is correctly identified",
        parent=crm,
        critical=True,
    )
    minister_name_claim = f"The minister-relative's name is {ex.cousin_minister_name}."
    await evaluator.verify(
        claim=minister_name_claim,
        node=minister_name_leaf,
        sources=combined_rel_sources,
        additional_instruction="Verify the minister-relative's identity/name is correctly cited on official pages.",
    )

    # 6) Confirm Minister Sanctioned Since 2018 and Position
    msp = evaluator.add_parallel(
        id="Confirm_Minister_Sanctioned_Since_2018_And_Position",
        desc="Confirms (with official OFAC/Treasury URL(s)) that the named minister-relative has been under OFAC sanctions since 2018 and holds the position of Minister of Interior, Justice, and Peace under the Maduro administration.",
        parent=main,
        critical=True,
    )
    ensure_sources_node(
        evaluator,
        msp,
        node_id="Confirm_Minister_Sanctioned_Since_2018_And_Position",
        desc="Minister sanctioned since 2018 and position confirmation",
        urls=ex.minister_sources,
        require_official=True,
    )
    # Leaf: sanctioned since 2018
    sanctioned_leaf = evaluator.add_leaf(
        id="Confirm_Minister_Sanctioned_Since_2018_And_Position_sanctioned_since_2018_verified",
        desc="The named minister-relative has been under OFAC sanctions since 2018",
        parent=msp,
        critical=True,
    )
    sanctioned_claim = f"{ex.cousin_minister_name} has been under OFAC sanctions since 2018."
    await evaluator.verify(
        claim=sanctioned_claim,
        node=sanctioned_leaf,
        sources=ex.minister_sources,
        additional_instruction=(
            "Check official OFAC/Treasury pages to confirm the minister-relative was designated or sanctioned in 2018; "
            "explicit 'since 2018' or initial designation date in 2018 suffices."
        ),
    )
    # Leaf: minister position
    position_leaf = evaluator.add_leaf(
        id="Confirm_Minister_Sanctioned_Since_2018_And_Position_position_verified",
        desc="The named minister-relative holds the position of Minister of Interior, Justice, and Peace under the Maduro administration",
        parent=msp,
        critical=True,
    )
    position_claim = (
        f"{ex.cousin_minister_name} holds the position of Minister of Interior, Justice, and Peace under the Maduro administration."
    )
    await evaluator.verify(
        claim=position_claim,
        node=position_leaf,
        sources=ex.minister_sources,
        additional_instruction="Verify the official sources explicitly state this ministerial role/title under the Maduro administration.",
    )

    # 7) Provide Cédula Number
    ced = evaluator.add_parallel(
        id="Provide_Cedula_Number",
        desc="Provides the identified individual’s Venezuelan cédula (national ID) number as recorded in OFAC designation/SDN records, with official OFAC/Treasury URL(s).",
        parent=main,
        critical=True,
    )
    # Cédula presence
    cedula_present = bool(ex.cedula_number and ex.cedula_number.strip())
    evaluator.add_custom_node(
        result=cedula_present,
        id="Provide_Cedula_Number_value_present",
        desc="Cédula (national ID) number is provided in the answer",
        parent=ced,
        critical=True,
    )
    # Sources presence + official
    ensure_sources_node(
        evaluator,
        ced,
        node_id="Provide_Cedula_Number",
        desc="Cédula number confirmation",
        urls=ex.cedula_sources,
        require_official=True,
    )
    # Leaf: cédula verification
    cedula_leaf = evaluator.add_leaf(
        id="Provide_Cedula_Number_verified",
        desc="The individual's Venezuelan cédula number matches OFAC designation/SDN records",
        parent=ced,
        critical=True,
    )
    ced_claim = f"{ex.individual_full_name} has Venezuelan cédula number {ex.cedula_number} in OFAC's designation/SDN records."
    await evaluator.verify(
        claim=ced_claim,
        node=cedula_leaf,
        sources=ex.cedula_sources,
        additional_instruction=(
            "Use the official OFAC SDN entry or designation documentation to confirm the national ID number exactly matches the provided value."
        ),
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
    Evaluate an answer for the OFAC SEBIN director cousin-of-minister task.
    """
    # Initialize evaluator with a SEQUENTIAL root to respect dependency ordering.
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

    # Extract structured information from the answer
    ex = await evaluator.extract(
        prompt=prompt_extract_ofac_task(),
        template_class=OFACTaskExtraction,
        extraction_name="ofac_task_extraction",
    )

    # Build verification tree according to rubric
    # 1) Identify Individual (critical, first in sequence)
    await verify_identify_individual(evaluator, root, ex)

    # 2) Provide Required Confirmations and ID (critical, parallel children)
    await verify_provide_confirmations_and_id(evaluator, root, ex)

    # Optionally record custom info stats
    def count_official(urls: List[str]) -> int:
        return sum(1 for u in urls if is_official_treasury_url(u))

    evaluator.add_custom_info(
        info={
            "sdn_urls_total": len(ex.sdn_urls),
            "sdn_urls_official": count_official(ex.sdn_urls),
            "sebin_sources_total": len(ex.sebin_sources),
            "sebin_sources_official": count_official(ex.sebin_sources),
            "relationship_sources_total": len(ex.relationship_sources),
            "relationship_sources_official": count_official(ex.relationship_sources),
            "minister_sources_total": len(ex.minister_sources),
            "minister_sources_official": count_official(ex.minister_sources),
            "designation_sources_total": len(ex.designation_sources),
            "designation_sources_official": count_official(ex.designation_sources),
            "action_context_sources_total": len(ex.action_context_sources),
            "action_context_sources_official": count_official(ex.action_context_sources),
            "count_sources_total": len(ex.count_sources),
            "count_sources_official": count_official(ex.count_sources),
            "cedula_sources_total": len(ex.cedula_sources),
            "cedula_sources_official": count_official(ex.cedula_sources),
        },
        info_type="source_domain_stats",
        info_name="official_source_domain_statistics",
    )

    # Return evaluation summary
    return evaluator.get_summary()