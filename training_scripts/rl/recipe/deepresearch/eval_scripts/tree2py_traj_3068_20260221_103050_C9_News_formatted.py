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
TASK_ID = "navy_amphib_ship_2025_2026"
TASK_DESCRIPTION = (
    "During the 2025-2026 timeframe, identify the U.S. Navy amphibious assault ship that met ALL of the following criteria: "
    "(1) Was based at Naval Station Norfolk, (2) Deployed to the Caribbean Sea/SOUTHCOM region in 2025, "
    "(3) Underwent at least one change of command ceremony during the 2024-2025 period, and "
    "(4) Experienced a U.S. Marine Corps personnel casualty (death) during or shortly after its 2025-2026 deployment. "
    "For the identified ship, provide: (1) Complete vessel identification including official ship name, hull classification number, and ship class, "
    "(2) Home port location, (3) Complete chain of command for 2024-2025 including names and ranks of all commanding officers during this period, "
    "dates of their assumption of command, and their command order numbers, (4) Deployment information including departure date for the 2025 deployment, "
    "deployment region, whether it was part of an Amphibious Ready Group, and any associated Marine Expeditionary Unit, and "
    "(5) Casualty incident details including type of incident, date of occurrence, full name and rank of the casualty, age, home state, date officially declared, "
    "final outcome status, and duration of any search and rescue operations."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Commander(BaseModel):
    name: Optional[str] = None
    rank: Optional[str] = None
    assumption_date: Optional[str] = None  # free-form string date
    command_order_number: Optional[str] = None  # e.g., "17th CO"


class ShipIdentity(BaseModel):
    ship_name: Optional[str] = None
    hull_classification_number: Optional[str] = None  # e.g., "LHD-1", "LHA-6"
    ship_class: Optional[str] = None  # e.g., "Wasp-class", "America-class"
    sources: List[str] = Field(default_factory=list)


class HomeportInfo(BaseModel):
    home_port: Optional[str] = None  # e.g., "Naval Station Norfolk" or "Norfolk, VA"
    sources: List[str] = Field(default_factory=list)


class CommandChain(BaseModel):
    commanders: List[Commander] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class DeploymentInfo(BaseModel):
    departure_date_2025: Optional[str] = None  # departure date for the 2025 deployment
    region: Optional[str] = None  # free-form, should indicate Caribbean/SOUTHCOM if applicable
    arg_participation: Optional[str] = None  # "yes"/"no"/group name or description
    associated_meu: Optional[str] = None  # e.g., "26th MEU"; null if not applicable
    sources: List[str] = Field(default_factory=list)


class CasualtyInfo(BaseModel):
    incident_type: Optional[str] = None  # e.g., "man overboard", "training accident"
    incident_date: Optional[str] = None
    casualty_rank: Optional[str] = None
    casualty_name: Optional[str] = None
    casualty_age: Optional[str] = None
    casualty_home_state: Optional[str] = None
    officially_declared_date: Optional[str] = None
    final_outcome_status: Optional[str] = None  # e.g., "declared deceased"
    search_and_rescue_duration: Optional[str] = None  # e.g., "72 hours"
    sources: List[str] = Field(default_factory=list)


class ShipExtraction(BaseModel):
    identity: Optional[ShipIdentity] = None
    homeport: Optional[HomeportInfo] = None
    command_chain: Optional[CommandChain] = None
    deployment_2025: Optional[DeploymentInfo] = None
    casualty: Optional[CasualtyInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ship() -> str:
    return """
    Extract a single identified U.S. Navy amphibious assault ship and all requested details from the answer. 
    If multiple ships are mentioned, choose the primary ship that the answer claims meets ALL the stated criteria 
    (base at Naval Station Norfolk, deployed to Caribbean/SOUTHCOM in 2025, change of command in 2024–2025, 
    and a U.S. Marine Corps fatality during/after the 2025–2026 deployment). If the answer lists multiple 
    candidates, choose the first one that is asserted to satisfy all criteria.

    Return a JSON object with these nested structures:

    identity:
      - ship_name: official ship name as written
      - hull_classification_number: e.g., "LHD-xx" or "LHA-xx"
      - ship_class: e.g., "Wasp-class", "America-class"
      - sources: array of URLs explicitly cited in the answer that identify the ship/class/type

    homeport:
      - home_port: ship homeport/home base as stated, e.g., "Naval Station Norfolk" or equivalent
      - sources: array of URLs cited specifically to support the homeport

    command_chain:
      - commanders: array of objects for all commanding officers in 2024–2025, each with:
        * name
        * rank
        * assumption_date (when they assumed command)
        * command_order_number (e.g., "17th CO")
      - sources: array of URLs cited that support the chain of command details or change-of-command events

    deployment_2025:
      - departure_date_2025: the departure/sail date for the 2025 deployment
      - region: deployment region (should indicate Caribbean Sea and/or SOUTHCOM if applicable)
      - arg_participation: "yes", "no", an ARG name, or descriptive text that makes clear whether it was part of an ARG
      - associated_meu: the associated Marine Expeditionary Unit if applicable (e.g., "26th MEU"); if clearly not applicable in the answer, set to null
      - sources: array of URLs cited that support the 2025 deployment details

    casualty:
      - incident_type: type of incident
      - incident_date: date of occurrence
      - casualty_rank: rank of the Marine casualty
      - casualty_name: full name of the Marine casualty
      - casualty_age: age of the casualty (as stated)
      - casualty_home_state: home state of the casualty
      - officially_declared_date: date officially declared (e.g., date declared deceased)
      - final_outcome_status: final outcome status (e.g., declared deceased, lost at sea)
      - search_and_rescue_duration: duration of any search and rescue operation (if applicable)
      - sources: array of URLs cited that support the casualty details

    IMPORTANT RULES:
    - Only extract information explicitly mentioned in the answer; do NOT invent or infer missing fields.
    - For any item not mentioned, set it to null (or an empty array for sources).
    - Extract URLs exactly as written in the answer (markdown links should be resolved to plain URLs).
    - Prefer strings for dates and numbers to maximize compatibility with varied answer formats.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _list_or_empty(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


def _safe_str(val: Optional[str]) -> str:
    return val if isinstance(val, str) else ""


def _compose_full_name_with_rank(rank: Optional[str], name: Optional[str]) -> str:
    rank_s = _safe_str(rank).strip()
    name_s = _safe_str(name).strip()
    if rank_s and name_s:
        return f"{rank_s} {name_s}"
    return name_s or rank_s


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_eligibility_checks(evaluator: Evaluator, parent, data: ShipExtraction) -> None:
    """
    Build and evaluate the 'eligibility_criteria' subtree (critical).
    """
    node = evaluator.add_parallel(
        id="eligibility_criteria",
        desc="Ship meets all eligibility constraints stated in the question.",
        parent=parent,
        critical=True
    )

    identity = data.identity or ShipIdentity()
    homeport = data.homeport or HomeportInfo()
    deploy = data.deployment_2025 or DeploymentInfo()
    coc = data.command_chain or CommandChain()
    casualty = data.casualty or CasualtyInfo()

    # 1) U.S. Navy amphibious assault ship
    leaf1 = evaluator.add_leaf(
        id="is_us_navy_amphibious_assault_ship",
        desc="Identified vessel is a U.S. Navy amphibious assault ship.",
        parent=node,
        critical=True
    )
    claim1 = (
        f"The vessel {_safe_str(identity.ship_name)} ({_safe_str(identity.hull_classification_number)}) "
        f"is a U.S. Navy amphibious assault ship (e.g., LHD or LHA), specifically a member of the {_safe_str(identity.ship_class)} class."
    ).strip()
    add_ins1 = (
        "Verify that the cited webpages explicitly identify this vessel as a U.S. Navy amphibious assault ship. "
        "Allow standard abbreviations (LHD/LHA) and class names (e.g., Wasp-class, America-class). "
        "If no URLs are provided or the pages do not support this classification, mark as not supported."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=_list_or_empty(identity.sources),
        additional_instruction=add_ins1
    )

    # 2) Based at Naval Station Norfolk
    leaf2 = evaluator.add_leaf(
        id="based_at_naval_station_norfolk",
        desc="Vessel is based at Naval Station Norfolk (homeported there) during the relevant timeframe.",
        parent=node,
        critical=True
    )
    claim2 = (
        f"The vessel {_safe_str(identity.ship_name)} was homeported at Naval Station Norfolk (Norfolk, VA) in the relevant timeframe (around 2025)."
    )
    add_ins2 = (
        "Confirm that the webpages indicate the ship's homeport as Naval Station Norfolk (or clearly Norfolk, Virginia). "
        "If homeport changed, ensure that during the relevant 2025 timeframe it was at Naval Station Norfolk."
    )
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=_list_or_empty(homeport.sources),
        additional_instruction=add_ins2
    )

    # 3) Deployed to Caribbean/SOUTHCOM in 2025
    leaf3 = evaluator.add_leaf(
        id="deployed_caribbean_or_southcom_in_2025",
        desc="Vessel deployed to the Caribbean Sea and/or SOUTHCOM region during 2025.",
        parent=node,
        critical=True
    )
    claim3 = (
        f"In 2025, {_safe_str(identity.ship_name)} deployed to the Caribbean Sea and/or the U.S. SOUTHCOM area of responsibility."
    )
    add_ins3 = (
        "Look for 2025 deployment announcements, press releases, or news articles indicating operations in the Caribbean or SOUTHCOM AOR. "
        "If sources do not clearly indicate Caribbean/SOUTHCOM deployment in 2025, mark as unsupported."
    )
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=_list_or_empty(deploy.sources),
        additional_instruction=add_ins3
    )

    # 4) Change of command in 2024–2025
    leaf4 = evaluator.add_leaf(
        id="change_of_command_2024_2025",
        desc="Vessel underwent at least one change-of-command ceremony during 2024–2025.",
        parent=node,
        critical=True
    )
    claim4 = (
        f"The vessel {_safe_str(identity.ship_name)} had at least one change-of-command ceremony during 2024–2025."
    )
    add_ins4 = (
        "Use official Navy releases or reliable reports indicating a change-of-command event within 2024 or 2025 (inclusive). "
        "If sources cite assumption of command ceremonies in that window, this criterion is met."
    )
    await evaluator.verify(
        claim=claim4,
        node=leaf4,
        sources=_list_or_empty(coc.sources),
        additional_instruction=add_ins4
    )

    # 5) USMC fatality during or shortly after 2025–2026 deployment
    leaf5 = evaluator.add_leaf(
        id="usmc_fatality_during_or_shortly_after_2025_2026_deployment",
        desc="A U.S. Marine Corps personnel casualty resulting in death occurred during or shortly after the ship’s 2025–2026 deployment timeframe.",
        parent=node,
        critical=True
    )
    claim5 = (
        f"A U.S. Marine associated with {_safe_str(identity.ship_name)} died during or shortly after its 2025–2026 deployment."
    )
    add_ins5 = (
        "Verify that the cited sources report a Marine Corps fatality (death) tied to this ship and that the timing was during or shortly after the 2025–2026 deployment."
    )
    await evaluator.verify(
        claim=claim5,
        node=leaf5,
        sources=_list_or_empty(casualty.sources),
        additional_instruction=add_ins5
    )


async def build_required_outputs(evaluator: Evaluator, parent, data: ShipExtraction) -> None:
    """
    Build and evaluate the 'required_outputs' subtree (critical).
    """
    node = evaluator.add_parallel(
        id="required_outputs",
        desc="Provides all requested details for the identified ship.",
        parent=parent,
        critical=True
    )

    identity = data.identity or ShipIdentity()
    homeport = data.homeport or HomeportInfo()
    coc = data.command_chain or CommandChain()
    deploy = data.deployment_2025 or DeploymentInfo()
    casualty = data.casualty or CasualtyInfo()

    # Vessel identification (presence checks; critical under required outputs)
    vi = evaluator.add_parallel(
        id="vessel_identification",
        desc="Provides complete vessel identification (official ship name, hull classification number, and ship class).",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_str(identity.ship_name)),
        id="official_ship_name",
        desc="Official U.S. Navy ship name is provided.",
        parent=vi,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_str(identity.hull_classification_number)),
        id="hull_classification_number",
        desc="Hull classification number is provided.",
        parent=vi,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_str(identity.ship_class)),
        id="ship_class_designation",
        desc="Ship class designation is provided.",
        parent=vi,
        critical=True
    )

    # Home port location (presence check; eligibility separately verifies Norfolk)
    evaluator.add_custom_node(
        result=bool(_safe_str(homeport.home_port)),
        id="home_port_location",
        desc="Home port location is provided.",
        parent=node,
        critical=True
    )

    # Chain of command 2024–2025 (verify against sources)
    coc_node = evaluator.add_parallel(
        id="chain_of_command_2024_2025",
        desc="Provides the complete chain of command for 2024–2025 as requested.",
        parent=node,
        critical=True
    )

    # Compose summarized claims from extracted commanders
    commanders = coc.commanders or []
    names_list = [c.name for c in commanders if _safe_str(c.name)]
    ranks_list = [c.rank for c in commanders if _safe_str(c.rank)]
    assumptions = [c.assumption_date for c in commanders if _safe_str(c.assumption_date)]
    order_numbers = [c.command_order_number for c in commanders if _safe_str(c.command_order_number)]

    # Includes all commanding officers in period
    leaf_coc_all = evaluator.add_leaf(
        id="includes_all_commanding_officers_in_period",
        desc="Includes every commanding officer who held command at any time during 2024–2025 (i.e., chain is complete for the period).",
        parent=coc_node,
        critical=True
    )
    claim_coc_all = (
        f"The listed 2024–2025 commanding officers for {_safe_str(identity.ship_name)} are: {', '.join([_safe_str(n) for n in names_list])}. "
        f"These names cover all COs who held command at any time during 2024–2025."
    )
    add_ins_coc_all = (
        "Check the cited webpages to determine who held command during 2024–2025 and whether all such officers are included in the provided list."
    )
    await evaluator.verify(
        claim=claim_coc_all,
        node=leaf_coc_all,
        sources=_list_or_empty(coc.sources),
        additional_instruction=add_ins_coc_all
    )

    # Names and ranks
    leaf_coc_names = evaluator.add_leaf(
        id="commanding_officer_names_and_ranks",
        desc="For all listed COs in 2024–2025, provides each CO's full name and rank.",
        parent=coc_node,
        critical=True
    )
    names_ranks_str = "; ".join(
        [f"{_compose_full_name_with_rank(c.rank, c.name)}" for c in commanders if _safe_str(c.name) or _safe_str(c.rank)]
    )
    claim_coc_names = (
        f"The 2024–2025 COs (names with ranks) for {_safe_str(identity.ship_name)} are: {names_ranks_str}."
    )
    add_ins_coc_names = "Verify that the sources list the same names and corresponding ranks for the 2024–2025 commanding officers."
    await evaluator.verify(
        claim=claim_coc_names,
        node=leaf_coc_names,
        sources=_list_or_empty(coc.sources),
        additional_instruction=add_ins_coc_names
    )

    # Assumption dates
    leaf_coc_dates = evaluator.add_leaf(
        id="commanding_officer_assumption_dates",
        desc="For all listed COs in 2024–2025, provides the date each assumed command.",
        parent=coc_node,
        critical=True
    )
    dates_str = "; ".join(
        [f"{_compose_full_name_with_rank(c.rank, c.name)} assumed command on {_safe_str(c.assumption_date)}"
         for c in commanders if _safe_str(c.assumption_date)]
    )
    claim_coc_dates = (
        f"For 2024–2025, the assumption-of-command dates for COs of {_safe_str(identity.ship_name)} are: {dates_str}."
    )
    add_ins_coc_dates = "Confirm that each listed assumption-of-command date matches the cited sources."
    await evaluator.verify(
        claim=claim_coc_dates,
        node=leaf_coc_dates,
        sources=_list_or_empty(coc.sources),
        additional_instruction=add_ins_coc_dates
    )

    # Command order numbers
    leaf_coc_orders = evaluator.add_leaf(
        id="commanding_officer_command_order_numbers",
        desc="For all listed COs in 2024–2025, provides each CO's command order number/designation (e.g., 17th CO).",
        parent=coc_node,
        critical=True
    )
    orders_str = "; ".join(
        [f"{_compose_full_name_with_rank(c.rank, c.name)} was {_safe_str(c.command_order_number)}"
         for c in commanders if _safe_str(c.command_order_number)]
    )
    claim_coc_orders = (
        f"The command order numbers during 2024–2025 for {_safe_str(identity.ship_name)} are as follows: {orders_str}."
    )
    add_ins_coc_orders = "Verify that each CO's command order number (e.g., 17th CO) matches what is stated in the cited sources."
    await evaluator.verify(
        claim=claim_coc_orders,
        node=leaf_coc_orders,
        sources=_list_or_empty(coc.sources),
        additional_instruction=add_ins_coc_orders
    )

    # Deployment information 2025
    dep_node = evaluator.add_parallel(
        id="deployment_information_2025",
        desc="Provides the required 2025 deployment details.",
        parent=node,
        critical=True
    )

    # Departure date
    leaf_dep_date = evaluator.add_leaf(
        id="deployment_departure_date",
        desc="Provides the specific departure date for the 2025 deployment.",
        parent=dep_node,
        critical=True
    )
    claim_dep_date = (
        f"The 2025 deployment departure date for {_safe_str(identity.ship_name)} was {_safe_str(deploy.departure_date_2025)}."
    )
    add_ins_dep_date = "Check the cited sources for the explicit departure/sail date in 2025."
    await evaluator.verify(
        claim=claim_dep_date,
        node=leaf_dep_date,
        sources=_list_or_empty(deploy.sources),
        additional_instruction=add_ins_dep_date
    )

    # Deployment region
    leaf_dep_region = evaluator.add_leaf(
        id="deployment_region",
        desc="Specifies the deployment region (Caribbean Sea and/or SOUTHCOM).",
        parent=dep_node,
        critical=True
    )
    claim_dep_region = (
        f"In 2025, the deployment region was stated as: {_safe_str(deploy.region)}."
    )
    add_ins_dep_region = (
        "Verify that the region matches the cited sources and clearly indicates Caribbean Sea and/or SOUTHCOM if applicable."
    )
    await evaluator.verify(
        claim=claim_dep_region,
        node=leaf_dep_region,
        sources=_list_or_empty(deploy.sources),
        additional_instruction=add_ins_dep_region
    )

    # ARG participation
    leaf_dep_arg = evaluator.add_leaf(
        id="arg_participation",
        desc="States whether the deployment was part of an Amphibious Ready Group (ARG).",
        parent=dep_node,
        critical=True
    )
    # Build a neutral claim that the sources can affirm either way
    arg_text = _safe_str(deploy.arg_participation)
    if arg_text.lower().strip() in {"no", "not part", "not an arg", "none"}:
        claim_dep_arg = (
            f"The 2025 deployment of {_safe_str(identity.ship_name)} was NOT part of an Amphibious Ready Group."
        )
    else:
        claim_dep_arg = (
            f"The 2025 deployment of {_safe_str(identity.ship_name)} WAS part of an Amphibious Ready Group (ARG): {arg_text}."
        )
    add_ins_dep_arg = (
        "Confirm from the sources whether the 2025 deployment was part of an ARG. "
        "If the sources show no ARG association, the 'NOT part of an ARG' claim is correct; "
        "otherwise, verify the ARG identification."
        )
    await evaluator.verify(
        claim=claim_dep_arg,
        node=leaf_dep_arg,
        sources=_list_or_empty(deploy.sources),
        additional_instruction=add_ins_dep_arg
    )

    # Associated MEU
    leaf_dep_meu = evaluator.add_leaf(
        id="associated_meu",
        desc="Identifies any associated Marine Expeditionary Unit (MEU) (if applicable).",
        parent=dep_node,
        critical=True
    )
    meu_text = _safe_str(deploy.associated_meu)
    if meu_text:
        claim_dep_meu = (
            f"The associated Marine Expeditionary Unit (MEU) for the 2025 deployment was {meu_text}."
        )
        add_ins_dep_meu = "Verify that the cited sources identify this specific MEU as associated with the deployment."
    else:
        claim_dep_meu = (
            f"There was no associated Marine Expeditionary Unit (MEU) for the 2025 deployment of {_safe_str(identity.ship_name)}."
        )
        add_ins_dep_meu = (
            "Verify that the cited sources indicate no MEU association; if they do name a MEU, then this claim is incorrect."
        )
    await evaluator.verify(
        claim=claim_dep_meu,
        node=leaf_dep_meu,
        sources=_list_or_empty(deploy.sources),
        additional_instruction=add_ins_dep_meu
    )

    # Casualty incident details
    cas_node = evaluator.add_parallel(
        id="casualty_incident_details",
        desc="Provides all required details about the Marine casualty incident.",
        parent=node,
        critical=True
    )

    # Incident type
    leaf_inc_type = evaluator.add_leaf(
        id="incident_type",
        desc="Type of incident is provided.",
        parent=cas_node,
        critical=True
    )
    claim_inc_type = f"The incident type was: {_safe_str(casualty.incident_type)}."
    await evaluator.verify(
        claim=claim_inc_type,
        node=leaf_inc_type,
        sources=_list_or_empty(casualty.sources),
        additional_instruction="Verify that the sources explicitly state this incident type."
    )

    # Incident date
    leaf_inc_date = evaluator.add_leaf(
        id="incident_date",
        desc="Date of occurrence is provided.",
        parent=cas_node,
        critical=True
    )
    claim_inc_date = f"The incident occurred on {_safe_str(casualty.incident_date)}."
    await evaluator.verify(
        claim=claim_inc_date,
        node=leaf_inc_date,
        sources=_list_or_empty(casualty.sources),
        additional_instruction="Verify that the sources provide this date of occurrence."
    )

    # Casualty name and rank
    leaf_cas_name_rank = evaluator.add_leaf(
        id="casualty_name_and_rank",
        desc="Full name and rank of the casualty is provided.",
        parent=cas_node,
        critical=True
    )
    claim_cas_name_rank = f"The Marine casualty was {_compose_full_name_with_rank(casualty.caszalty_rank if hasattr(casualty, 'caszalty_rank') else casualty.casualty_rank, casualty.casualty_name)}."
    # Note: The above handles any accidental mis-typing; prefer casualty.casualty_rank
    claim_cas_name_rank = f"The Marine casualty was {_compose_full_name_with_rank(casualty.casualty_rank, casualty.casualty_name)}."
    await evaluator.verify(
        claim=claim_cas_name_rank,
        node=leaf_cas_name_rank,
        sources=_list_or_empty(casualty.sources),
        additional_instruction="Verify that the sources provide both the correct rank and full name."
    )

    # Casualty age
    leaf_cas_age = evaluator.add_leaf(
        id="casualty_age",
        desc="Age of the casualty is provided.",
        parent=cas_node,
        critical=True
    )
    claim_cas_age = f"The casualty's age was {_safe_str(casualty.casualty_age)}."
    await evaluator.verify(
        claim=claim_cas_age,
        node=leaf_cas_age,
        sources=_list_or_empty(casualty.sources),
        additional_instruction="Verify that the cited sources explicitly mention this age."
    )

    # Casualty home state
    leaf_cas_state = evaluator.add_leaf(
        id="casualty_home_state",
        desc="Home state of the casualty is provided.",
        parent=cas_node,
        critical=True
    )
    claim_cas_state = f"The casualty's home state was {_safe_str(casualty.casualty_home_state)}."
    await evaluator.verify(
        claim=claim_cas_state,
        node=leaf_cas_state,
        sources=_list_or_empty(casualty.sources),
        additional_instruction="Verify that the cited sources explicitly mention this home state."
    )

    # Officially declared date
    leaf_decl_date = evaluator.add_leaf(
        id="officially_declared_date",
        desc="Date the casualty was officially declared is provided.",
        parent=cas_node,
        critical=True
    )
    claim_decl_date = f"The casualty was officially declared on {_safe_str(casualty.officially_declared_date)}."
    await evaluator.verify(
        claim=claim_decl_date,
        node=leaf_decl_date,
        sources=_list_or_empty(casualty.sources),
        additional_instruction="Verify that the cited sources explicitly provide this official declaration date."
    )

    # Final outcome status
    leaf_outcome = evaluator.add_leaf(
        id="final_outcome_status",
        desc="Final outcome status is provided (e.g., declared deceased, lost at sea).",
        parent=cas_node,
        critical=True
    )
    claim_outcome = f"The final outcome status was: {_safe_str(casualty.final_outcome_status)}."
    await evaluator.verify(
        claim=claim_outcome,
        node=leaf_outcome,
        sources=_list_or_empty(casualty.sources),
        additional_instruction="Verify that the cited sources explicitly state this final outcome status."
    )

    # Search and rescue duration
    leaf_sar = evaluator.add_leaf(
        id="search_and_rescue_duration",
        desc="Duration of any search and rescue operations is provided (if applicable).",
        parent=cas_node,
        critical=True
    )
    if _safe_str(casualty.search_and_rescue_duration):
        claim_sar = f"The search and rescue operation lasted {_safe_str(casualty.search_and_rescue_duration)}."
        add_ins_sar = "Verify that the sources explicitly state this SAR duration."
    else:
        claim_sar = "There was no stated search and rescue duration (not applicable)."
        add_ins_sar = (
            "Verify from sources whether a SAR duration was reported. If none was reported, consider this claim correct."
        )
    await evaluator.verify(
        claim=claim_sar,
        node=leaf_sar,
        sources=_list_or_empty(casualty.sources),
        additional_instruction=add_ins_sar
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
    model: str = "o4-mini"
) -> Dict:
    """
    Entry point for evaluating an answer to the Navy amphibious assault ship task.
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
        default_model=model
    )
    # Make root critical to match rubric (all children must be critical as well)
    if evaluator.root:
        evaluator.root.critical = True

    # Extract structured information
    extraction = await evaluator.extract(
        prompt=prompt_extract_ship(),
        template_class=ShipExtraction,
        extraction_name="ship_extraction"
    )

    # Build tree: eligibility and required outputs
    await build_eligibility_checks(evaluator, evaluator.root, extraction)
    await build_required_outputs(evaluator, evaluator.root, extraction)

    # Return summarized results
    return evaluator.get_summary()