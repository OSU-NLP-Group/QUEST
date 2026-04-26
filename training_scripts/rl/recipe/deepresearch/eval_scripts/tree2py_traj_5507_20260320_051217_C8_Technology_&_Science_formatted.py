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
TASK_ID = "enterprise_colocation_selection"
TASK_DESCRIPTION = """
An enterprise IT organization is expanding its infrastructure and needs to identify suitable enterprise-grade colocation data center facilities in four major US markets. For each of the following markets, identify one colocation facility that meets the specified requirements:

Northern Virginia market:
- Minimum power capacity of 10 MW
- Uptime Institute Tier III or Tier IV certification
- Carrier-neutral connectivity with access to multiple fiber providers

Dallas-Fort Worth market:
- Minimum power capacity of 5 MW
- Uptime Institute Tier III or Tier IV certification
- 24/7/365 on-site security monitoring

Phoenix market:
- Minimum power capacity of 5 MW
- Uptime Institute Tier II or higher certification
- Environmental controls maintaining temperature within 64.4-80.6°F (18-27°C) range

Chicago market:
- Minimum power capacity of 5 MW
- Uptime Institute Tier III or Tier IV certification
- Direct fiber optic connectivity infrastructure

For each facility, provide the facility name, location details, and a reference URL documenting that the facility meets the specified requirements.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityInfo(BaseModel):
    facility_name: Optional[str] = None
    location: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    northern_virginia: Optional[FacilityInfo] = None
    dallas_fort_worth: Optional[FacilityInfo] = None
    phoenix: Optional[FacilityInfo] = None
    chicago: Optional[FacilityInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
    Extract the selected colocation facility details for each market as stated in the answer.
    For each of the following markets, extract:
      - facility_name: the name of the selected facility (not the operator's company name unless that is the facility name used in the answer)
      - location: location details mentioned (city, campus, neighborhood, or address fragment)
      - reference_urls: all URLs provided in the answer that substantiate the facility's specifications or requirements for that market.
        Extract actual URLs even if embedded in markdown links.

    Markets to extract (use these exact field names):
      - northern_virginia  (aka: Northern Virginia, NoVA, Ashburn, IAD region)
      - dallas_fort_worth  (aka: Dallas, DFW)
      - phoenix            (aka: PHX)
      - chicago            (aka: CHI)

    Rules:
      - Only include URLs explicitly present in the answer. If none, return an empty list.
      - If a facility name or location is not given for a market, set it to null.
      - Deduplicate URLs but keep different pages if they are distinct.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _valid_urls(urls: List[str]) -> List[str]:
    out = []
    for u in urls or []:
        if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")):
            out.append(u.strip())
    return out


def _display_name(market_label: str, info: Optional[FacilityInfo]) -> str:
    if info and info.facility_name:
        return info.facility_name.strip()
    return f"the selected facility in the {market_label}"


def _where_str(info: Optional[FacilityInfo]) -> str:
    return f" in {info.location.strip()}" if info and info.location else ""


# --------------------------------------------------------------------------- #
# Verification logic per market                                               #
# --------------------------------------------------------------------------- #
async def verify_northern_virginia(evaluator: Evaluator, parent, info: Optional[FacilityInfo]) -> None:
    node = evaluator.add_parallel(
        id="Northern_Virginia_Facility",
        desc="Identify a suitable colocation facility in the Northern Virginia market",
        parent=parent,
        critical=False
    )

    urls = _valid_urls(info.reference_urls if info else [])
    # Reference URL presence (critical)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="NV_Reference_URL",
        desc="A valid reference URL is provided documenting the Northern Virginia facility's specifications",
        parent=node,
        critical=True
    )

    name = _display_name("Northern Virginia market", info)
    where = _where_str(info)

    # Power capacity ≥ 10 MW
    leaf_power = evaluator.add_leaf(
        id="NV_Power_Capacity",
        desc="The facility provides at least 10 MW of power capacity",
        parent=node,
        critical=True
    )
    claim_power = f"The facility '{name}'{where} provides at least 10 MW of power capacity (utility/critical IT load) available to customers."
    await evaluator.verify(
        claim=claim_power,
        node=leaf_power,
        sources=urls,
        additional_instruction="Look for explicit statements of total available power, IT load, critical load, or utility capacity being >= 10 MW at the site or campus. Accept campus-level capacity if the facility is part of that campus."
    )

    # Uptime Tier III or IV
    leaf_tier = evaluator.add_leaf(
        id="NV_Tier_Certification",
        desc="The facility holds Uptime Institute Tier III or Tier IV certification",
        parent=node,
        critical=True
    )
    claim_tier = f"The facility '{name}'{where} holds an Uptime Institute Tier III or Tier IV certification (Design or Constructed Facility)."
    await evaluator.verify(
        claim=claim_tier,
        node=leaf_tier,
        sources=urls,
        additional_instruction="Require explicit 'Uptime Institute Tier III' or 'Tier IV' certification. Accept 'Tier III Design' or 'Tier III Constructed Facility'. Do NOT accept vague phrases like 'meets Tier III' without certification."
    )

    # Carrier-neutral connectivity with multiple fiber providers
    leaf_connect = evaluator.add_leaf(
        id="NV_Carrier_Connectivity",
        desc="The facility offers carrier-neutral connectivity with access to multiple fiber providers",
        parent=node,
        critical=True
    )
    claim_connect = f"The facility '{name}'{where} is carrier-neutral and offers access to multiple (more than one) fiber or network providers."
    await evaluator.verify(
        claim=claim_connect,
        node=leaf_connect,
        sources=urls,
        additional_instruction="Confirm both 'carrier-neutral' and presence of multiple providers (e.g., 'many carriers on-site', 'access to X+ networks')."
    )


async def verify_dallas(evaluator: Evaluator, parent, info: Optional[FacilityInfo]) -> None:
    node = evaluator.add_parallel(
        id="Dallas_Facility",
        desc="Identify a suitable colocation facility in the Dallas-Fort Worth market",
        parent=parent,
        critical=False
    )

    urls = _valid_urls(info.reference_urls if info else [])
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Dallas_Reference_URL",
        desc="A valid reference URL is provided documenting the Dallas facility's specifications",
        parent=node,
        critical=True
    )

    name = _display_name("Dallas-Fort Worth market", info)
    where = _where_str(info)

    # Power capacity ≥ 5 MW
    leaf_power = evaluator.add_leaf(
        id="Dallas_Power_Capacity",
        desc="The facility provides at least 5 MW of power capacity",
        parent=node,
        critical=True
    )
    claim_power = f"The facility '{name}'{where} provides at least 5 MW of power capacity (utility/critical IT load) available to customers."
    await evaluator.verify(
        claim=claim_power,
        node=leaf_power,
        sources=urls,
        additional_instruction="Look for explicit statements of total available power, IT load, critical load, or utility capacity being >= 5 MW."
    )

    # Uptime Tier III or IV
    leaf_tier = evaluator.add_leaf(
        id="Dallas_Tier_Certification",
        desc="The facility holds Uptime Institute Tier III or Tier IV certification",
        parent=node,
        critical=True
    )
    claim_tier = f"The facility '{name}'{where} holds an Uptime Institute Tier III or Tier IV certification (Design or Constructed Facility)."
    await evaluator.verify(
        claim=claim_tier,
        node=leaf_tier,
        sources=urls,
        additional_instruction="Require explicit Uptime Institute Tier III or Tier IV certification. Accept 'Design' or 'Constructed Facility' certifications."
    )

    # 24/7/365 on-site security monitoring
    leaf_sec = evaluator.add_leaf(
        id="Dallas_Security",
        desc="The facility provides 24/7/365 on-site security monitoring",
        parent=node,
        critical=True
    )
    claim_sec = f"The facility '{name}'{where} provides 24/7/365 on-site security monitoring."
    await evaluator.verify(
        claim=claim_sec,
        node=leaf_sec,
        sources=urls,
        additional_instruction="Look for '24/7', '24x7', '365', 'on-site security', 'security operations center', 'manned guard'."
    )


async def verify_phoenix(evaluator: Evaluator, parent, info: Optional[FacilityInfo]) -> None:
    node = evaluator.add_parallel(
        id="Phoenix_Facility",
        desc="Identify a suitable colocation facility in the Phoenix market",
        parent=parent,
        critical=False
    )

    urls = _valid_urls(info.reference_urls if info else [])
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Phoenix_Reference_URL",
        desc="A valid reference URL is provided documenting the Phoenix facility's specifications",
        parent=node,
        critical=True
    )

    name = _display_name("Phoenix market", info)
    where = _where_str(info)

    # Power capacity ≥ 5 MW
    leaf_power = evaluator.add_leaf(
        id="Phoenix_Power_Capacity",
        desc="The facility provides at least 5 MW of power capacity",
        parent=node,
        critical=True
    )
    claim_power = f"The facility '{name}'{where} provides at least 5 MW of power capacity (utility/critical IT load) available to customers."
    await evaluator.verify(
        claim=claim_power,
        node=leaf_power,
        sources=urls,
        additional_instruction="Look for explicit statements of total available power, IT load, critical load, or utility capacity being >= 5 MW."
    )

    # Uptime Tier II or higher
    leaf_tier = evaluator.add_leaf(
        id="Phoenix_Tier_Certification",
        desc="The facility holds Uptime Institute Tier II or higher certification",
        parent=node,
        critical=True
    )
    claim_tier = f"The facility '{name}'{where} holds an Uptime Institute Tier II or higher certification (Tier II, III, or IV)."
    await evaluator.verify(
        claim=claim_tier,
        node=leaf_tier,
        sources=urls,
        additional_instruction="Require explicit Uptime Institute certification at Tier II or above. Accept 'Design' or 'Constructed Facility'."
    )

    # Environmental controls within 64.4–80.6°F (18–27°C)
    leaf_cooling = evaluator.add_leaf(
        id="Phoenix_Cooling",
        desc="The facility maintains environmental controls within 64.4-80.6°F (18-27°C) temperature range",
        parent=node,
        critical=True
    )
    claim_cooling = f"The facility '{name}'{where} maintains temperature within 64.4–80.6°F (18–27°C) in data halls."
    await evaluator.verify(
        claim=claim_cooling,
        node=leaf_cooling,
        sources=urls,
        additional_instruction="Look for explicit temperature targets or adherence to ASHRAE recommended 18–27°C range. General 'precision cooling' without the range is insufficient."
    )


async def verify_chicago(evaluator: Evaluator, parent, info: Optional[FacilityInfo]) -> None:
    node = evaluator.add_parallel(
        id="Chicago_Facility",
        desc="Identify a suitable colocation facility in the Chicago market",
        parent=parent,
        critical=False
    )

    urls = _valid_urls(info.reference_urls if info else [])
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Chicago_Reference_URL",
        desc="A valid reference URL is provided documenting the Chicago facility's specifications",
        parent=node,
        critical=True
    )

    name = _display_name("Chicago market", info)
    where = _where_str(info)

    # Power capacity ≥ 5 MW
    leaf_power = evaluator.add_leaf(
        id="Chicago_Power_Capacity",
        desc="The facility provides at least 5 MW of power capacity",
        parent=node,
        critical=True
    )
    claim_power = f"The facility '{name}'{where} provides at least 5 MW of power capacity (utility/critical IT load) available to customers."
    await evaluator.verify(
        claim=claim_power,
        node=leaf_power,
        sources=urls,
        additional_instruction="Look for explicit statements of total available power, IT load, critical load, or utility capacity being >= 5 MW."
    )

    # Uptime Tier III or IV
    leaf_tier = evaluator.add_leaf(
        id="Chicago_Tier_Certification",
        desc="The facility holds Uptime Institute Tier III or Tier IV certification",
        parent=node,
        critical=True
    )
    claim_tier = f"The facility '{name}'{where} holds an Uptime Institute Tier III or Tier IV certification (Design or Constructed Facility)."
    await evaluator.verify(
        claim=claim_tier,
        node=leaf_tier,
        sources=urls,
        additional_instruction="Require explicit Uptime Institute Tier III or IV certification. Accept 'Design' or 'Constructed Facility'."
    )

    # Direct fiber optic connectivity infrastructure
    leaf_fiber = evaluator.add_leaf(
        id="Chicago_Fiber_Connectivity",
        desc="The facility provides direct fiber optic connectivity infrastructure",
        parent=node,
        critical=True
    )
    claim_fiber = f"The facility '{name}'{where} provides direct fiber optic connectivity infrastructure (e.g., on-site fiber, diverse fiber paths, or on-net carriers)."
    await evaluator.verify(
        claim=claim_fiber,
        node=leaf_fiber,
        sources=urls,
        additional_instruction="Look for terms like 'direct fiber', 'on-net carriers', 'multiple fiber routes', 'diverse fiber entrances', or 'dark fiber'."
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
    Evaluate an answer for the Enterprise Colocation Selection task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across markets
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

    # Extract structured facility info from answer
    extracted: FacilitiesExtraction = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction"
    )

    # Build verification tree aligned with rubric
    # Enterprise_Colocation_Selection (as a grouping node under root for clarity)
    grouping = evaluator.add_parallel(
        id="Enterprise_Colocation_Selection",
        desc="Evaluate the selection of enterprise-grade colocation facilities in four major US data center markets, with each facility meeting specific technical and operational requirements",
        parent=root,
        critical=False
    )

    # Per-market verification
    await verify_northern_virginia(evaluator, grouping, extracted.northern_virginia)
    await verify_dallas(evaluator, grouping, extracted.dallas_fort_worth)
    await verify_phoenix(evaluator, grouping, extracted.phoenix)
    await verify_chicago(evaluator, grouping, extracted.chicago)

    return evaluator.get_summary()