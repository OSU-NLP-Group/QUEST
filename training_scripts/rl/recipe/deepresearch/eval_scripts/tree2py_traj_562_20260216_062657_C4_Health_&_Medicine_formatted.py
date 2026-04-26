import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cdc_medicare_68yo_bucks_pa"
TASK_DESCRIPTION = (
    "A 68-year-old adult living in Bucks County, Pennsylvania has Medicare coverage and wants to know which vaccinations "
    "are currently recommended for their age group. Identify the vaccines that the CDC recommends for this age group, "
    "specify the appropriate vaccine types or dosing schedules where applicable, state which part of Medicare (Part B or Part D) "
    "covers each vaccine at no out-of-pocket cost, and provide at least one specific location in Bucks County with a complete "
    "address where these vaccines can be obtained."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Location(BaseModel):
    """A single vaccination location extracted from the answer."""
    name: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class VaccinesExtraction(BaseModel):
    """
    Consolidated extraction of vaccine-related evidence and location(s) from the answer.
    Sources must be URLs explicitly present in the answer.
    """
    # Influenza
    influenza_preferred_types: Optional[str] = None  # Free text the answer used (e.g., "high-dose, recombinant, or adjuvanted")
    influenza_coverage_part: Optional[str] = None  # e.g., "Part B"
    influenza_sources: List[str] = Field(default_factory=list)

    # Pneumococcal
    pneumococcal_options: Optional[str] = None  # e.g., "PCV20, PCV21, or PCV15 then PPSV23"
    pneumococcal_coverage_part: Optional[str] = None  # e.g., "Part B"
    pneumococcal_sources: List[str] = Field(default_factory=list)

    # Shingles (Shingrix)
    shingles_schedule: Optional[str] = None  # e.g., "2 doses separated by 2–6 months"
    shingles_coverage_part: Optional[str] = None  # e.g., "Part D"
    shingles_sources: List[str] = Field(default_factory=list)

    # Tdap/Td booster
    tdap_frequency: Optional[str] = None  # e.g., "every 10 years"
    tdap_coverage_part: Optional[str] = None  # e.g., "Part D"
    tdap_sources: List[str] = Field(default_factory=list)

    # COVID-19 (optional coverage mention)
    covid_coverage_part: Optional[str] = None  # e.g., "Part B"
    covid_sources: List[str] = Field(default_factory=list)

    # Locations (at least one in Bucks County, PA, with complete address)
    locations: List[Location] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_vaccines_data() -> str:
    return """
    Extract the following structured information exactly as presented in the answer. Do NOT invent anything.

    1) For each vaccine/topic below, collect all supporting source URLs (as they appear in the answer text, including markdown links).
       If no URLs are provided for a topic in the answer, return an empty list for that topic's sources.
       Also extract the coverage "part" text (e.g., "Part B", "Part D") if mentioned, otherwise null.
       Additionally, extract the specific schedule or types text if the answer mentions them.

       a) Influenza (flu)
          - influenza_sources: array of URLs supporting the flu recommendation(s) and/or Medicare coverage
          - influenza_coverage_part: string or null
          - influenza_preferred_types: string or null (e.g., "high-dose, recombinant, or adjuvanted")

       b) Pneumococcal
          - pneumococcal_sources: array of URLs
          - pneumococcal_coverage_part: string or null
          - pneumococcal_options: string or null (e.g., "PCV20, PCV21, or PCV15 followed by PPSV23")

       c) Shingles (Shingrix)
          - shingles_sources: array of URLs
          - shingles_coverage_part: string or null
          - shingles_schedule: string or null (e.g., "2 doses separated by 2–6 months")

       d) Tdap/Td booster
          - tdap_sources: array of URLs
          - tdap_coverage_part: string or null
          - tdap_frequency: string or null (e.g., "every 10 years")

       e) COVID-19 (optional coverage mention)
          - covid_sources: array of URLs
          - covid_coverage_part: string or null

    2) Extract up to 3 vaccination locations listed in the answer (if any). For each, return:
       - name
       - street (line 1 street address)
       - city
       - state (should be "PA" if it’s in Pennsylvania)
       - zip (5-digit or ZIP+4)
       - urls: array of one or more URLs in the answer that specifically refer to that location (e.g., provider or store page showing the address/services)

       If no locations are mentioned in the answer, return an empty array.

    Return a single JSON object matching the VaccinesExtraction schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def first_location(extracted: VaccinesExtraction) -> Optional[Location]:
    return extracted.locations[0] if extracted.locations else None


def full_address(loc: Optional[Location]) -> str:
    if not loc:
        return ""
    parts = [loc.street or "", loc.city or "", loc.state or "", loc.zip or ""]
    # Format: "street, city, state zip" (skip empty safely)
    street = (loc.street or "").strip()
    city = (loc.city or "").strip()
    state = (loc.state or "").strip()
    zipcode = (loc.zip or "").strip()
    if street and city and state and zipcode:
        return f"{street}, {city}, {state} {zipcode}"
    return " ".join(p for p in parts if p).strip()


def has_complete_pa_address(loc: Optional[Location]) -> bool:
    if not loc:
        return False
    if not (loc.street and loc.city and loc.state and loc.zip):
        return False
    # State must be PA (case-insensitive)
    if (loc.state or "").strip().upper() != "PA":
        return False
    # ZIP must be 5-digit or ZIP+4
    if not re.match(r"^\d{5}(?:-\d{4})?$", (loc.zip or "").strip()):
        return False
    return True


def safe_sources(urls: Optional[List[str]]) -> List[str]:
    return urls or []


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_influenza(evaluator: Evaluator, parent_node, data: VaccinesExtraction) -> None:
    node = evaluator.add_parallel(
        id="Influenza_Vaccine",
        desc="Influenza vaccine information (recommendation, preferred type, Medicare coverage).",
        parent=parent_node,
        critical=True
    )
    srcs = safe_sources(data.influenza_sources)

    # Annually recommended (65+)
    leaf1 = evaluator.add_leaf(
        id="Flu_Recommended_Annually",
        desc="States that an influenza vaccine is recommended annually for this age group (65+).",
        parent=node,
        critical=True
    )
    claim1 = "An influenza vaccine is recommended every year (annually) for adults aged 65 years and older."
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=srcs,
        additional_instruction="Confirm using CDC or ACIP guidance. Minor wording variants are acceptable if the meaning is the same."
    )

    # Preferred types 65+
    leaf2 = evaluator.add_leaf(
        id="Flu_Preferred_Types_65plus",
        desc="Specifies that for adults 65+ high-dose, recombinant, or adjuvanted flu vaccines are preferred.",
        parent=node,
        critical=True
    )
    claim2 = (
        "For adults 65 years and older, the preferred influenza vaccines are high-dose inactivated, adjuvanted inactivated, "
        "or recombinant (e.g., Fluzone High-Dose, Fluad, or Flublok)."
    )
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=srcs,
        additional_instruction="Check CDC/ACIP statements about preferred influenza vaccine products for ≥65 years."
    )

    # Medicare coverage (Part B)
    leaf3 = evaluator.add_leaf(
        id="Flu_Medicare_Coverage",
        desc="States Medicare Part B covers the flu vaccine at no out-of-pocket cost.",
        parent=node,
        critical=True
    )
    claim3 = "Medicare Part B covers the seasonal influenza (flu) shot with no out-of-pocket cost."
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=srcs,
        additional_instruction="Prefer official Medicare or CMS sources. If the provided URL is Medicare/CMS or an official plan page citing Part B, that supports the claim."
    )


async def verify_pneumococcal(evaluator: Evaluator, parent_node, data: VaccinesExtraction) -> None:
    node = evaluator.add_parallel(
        id="Pneumococcal_Vaccine",
        desc="Pneumococcal vaccine information (recommendation, acceptable options, Medicare coverage).",
        parent=parent_node,
        critical=True
    )
    srcs = safe_sources(data.pneumococcal_sources)

    # Recommended for adults 50+
    leaf1 = evaluator.add_leaf(
        id="Pneumococcal_Recommended_50plus",
        desc="States pneumococcal vaccination is recommended for adults aged 50 and older.",
        parent=node,
        critical=True
    )
    claim1 = "Pneumococcal vaccination is recommended for adults aged 50 years and older."
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=srcs,
        additional_instruction="Use the latest CDC/ACIP guidance; minor wording differences are acceptable if the recommendation clearly includes age 50+."
    )

    # Acceptable options
    leaf2 = evaluator.add_leaf(
        id="Pneumococcal_Acceptable_Options",
        desc="Specifies acceptable options: PCV20, PCV21, or PCV15 followed by PPSV23.",
        parent=node,
        critical=True
    )
    claim2 = "Acceptable adult pneumococcal options include PCV20 or PCV21 alone, or PCV15 followed by PPSV23."
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=srcs,
        additional_instruction="Check CDC adult schedule notes for pneumococcal vaccination options."
    )

    # Medicare coverage (Part B)
    leaf3 = evaluator.add_leaf(
        id="Pneumococcal_Medicare_Coverage",
        desc="States Medicare Part B covers pneumococcal vaccines at no out-of-pocket cost.",
        parent=node,
        critical=True
    )
    claim3 = "Medicare Part B covers pneumococcal shots at no out-of-pocket cost."
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=srcs,
        additional_instruction="Prefer Medicare/CMS pages that explicitly state Part B coverage for pneumococcal shots."
    )


async def verify_shingles(evaluator: Evaluator, parent_node, data: VaccinesExtraction) -> None:
    node = evaluator.add_parallel(
        id="Shingles_Vaccine",
        desc="Shingles vaccine (Shingrix) information (recommendation, dosing schedule, Medicare coverage).",
        parent=parent_node,
        critical=True
    )
    srcs = safe_sources(data.shingles_sources)

    # Shingrix recommended 50+
    leaf1 = evaluator.add_leaf(
        id="Shingrix_Recommended_50plus",
        desc="States Shingrix (RZV) is recommended for adults aged 50 and older to prevent shingles.",
        parent=node,
        critical=True
    )
    claim1 = "Shingrix (recombinant zoster vaccine, RZV) is recommended for adults aged 50 years and older to prevent shingles."
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=srcs,
        additional_instruction="Use CDC shingles vaccine guidance; acceptance of equivalent wording is allowed."
    )

    # Shingrix 2-dose schedule
    leaf2 = evaluator.add_leaf(
        id="Shingrix_2_Dose_Schedule",
        desc="Specifies 2 doses of Shingrix separated by 2–6 months.",
        parent=node,
        critical=True
    )
    claim2 = "Shingrix is administered as 2 doses separated by 2 to 6 months."
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=srcs,
        additional_instruction="Verify dosing schedule on CDC or manufacturer pages."
    )

    # Medicare coverage (Part D)
    leaf3 = evaluator.add_leaf(
        id="Shingrix_Medicare_Coverage",
        desc="States Medicare Part D covers Shingrix at no out-of-pocket cost.",
        parent=node,
        critical=True
    )
    claim3 = "Shingrix is covered under Medicare Part D at no out-of-pocket cost."
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=srcs,
        additional_instruction="Post-2023 IRA changes eliminated cost-sharing for Part D vaccines. Prefer Medicare/CMS sources."
    )


async def verify_tdap(evaluator: Evaluator, parent_node, data: VaccinesExtraction) -> None:
    node = evaluator.add_parallel(
        id="Tdap_Td_Booster",
        desc="Tetanus/diphtheria/pertussis booster information (frequency and Medicare coverage).",
        parent=parent_node,
        critical=True
    )
    srcs = safe_sources(data.tdap_sources)

    # Every 10 years
    leaf1 = evaluator.add_leaf(
        id="Tdap_Td_Every_10_Years",
        desc="States adults should receive a Tdap or Td booster every 10 years.",
        parent=node,
        critical=True
    )
    claim1 = "Adults should receive a Td or Tdap booster every 10 years."
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=srcs,
        additional_instruction="Use CDC adult immunization schedule or Td/Tdap vaccine page."
    )

    # Medicare coverage (Part D)
    leaf2 = evaluator.add_leaf(
        id="Tdap_Td_Medicare_Coverage",
        desc="States Medicare Part D covers Tdap/Td (as a non-Part B vaccine) at no out-of-pocket cost.",
        parent=node,
        critical=True
    )
    claim2 = "Tdap/Td is covered under Medicare Part D at no out-of-pocket cost."
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=srcs,
        additional_instruction="Prefer Medicare/CMS sources indicating $0 cost for Part D vaccines due to IRA changes."
    )


async def verify_covid_optional(evaluator: Evaluator, parent_node, data: VaccinesExtraction) -> None:
    # Optional single leaf
    leaf = evaluator.add_leaf(
        id="COVID19_Medicare_Coverage_Optional",
        desc="Mentions that Medicare Part B covers COVID-19 vaccines at no out-of-pocket cost.",
        parent=parent_node,
        critical=False
    )
    claim = "Medicare Part B covers COVID-19 vaccines at no out-of-pocket cost."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=safe_sources(data.covid_sources),
        additional_instruction="Prefer Medicare/CMS or CDC pages that specify Part B coverage for COVID-19 vaccines."
    )


async def verify_local_location(evaluator: Evaluator, parent_node, data: VaccinesExtraction) -> None:
    node = evaluator.add_parallel(
        id="Local_Vaccination_Location",
        desc="Provides at least one Bucks County vaccination location with a complete address where vaccines can be obtained.",
        parent=parent_node,
        critical=True
    )

    loc = first_location(data)
    addr_ok = has_complete_pa_address(loc)
    addr_text = full_address(loc)
    loc_name = (loc.name if loc and loc.name else "the listed site").strip()
    loc_sources = safe_sources(loc.urls if loc else [])

    # 1) Location is in Bucks County, Pennsylvania (supported by page evidence)
    leaf1 = evaluator.add_leaf(
        id="Location_In_Bucks_County",
        desc="At least one named vaccination site is located in Bucks County, Pennsylvania.",
        parent=node,
        critical=True
    )
    claim1 = f"The vaccination site '{loc_name}' at address '{addr_text}' is in Bucks County, Pennsylvania."
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=loc_sources,
        additional_instruction=(
            "Use the webpage evidence. If the page shows a city that is within Bucks County, PA (e.g., Doylestown, Newtown, Levittown, Yardley, Langhorne, Bensalem, Warminster, Warrington, etc.), "
            "consider that sufficient evidence even if 'Bucks County' is not explicitly written. The key is that the address corresponds to a location in Bucks County."
        )
    )

    # 2) Complete physical address provided in the answer (street, city, state, ZIP)
    evaluator.add_custom_node(
        result=addr_ok,
        id="Complete_Physical_Address",
        desc="Provides a complete physical address for at least one listed site (street address, city, state, ZIP).",
        parent=node,
        critical=True
    )

    # 3) Site presented as a vaccination provider
    leaf3 = evaluator.add_leaf(
        id="Presented_As_Vaccination_Provider",
        desc="Indicates the listed site provides vaccination/immunization services.",
        parent=node,
        critical=True
    )
    claim3 = f"The site '{loc_name}' provides vaccination or immunization services."
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=loc_sources,
        additional_instruction="Accept synonyms like 'vaccines', 'immunizations', or 'vaccination clinic' on the page as evidence."
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
    Evaluate an answer for CDC vaccine recommendations and Medicare coverage for a 68-year-old in Bucks County, PA.
    """
    # Initialize evaluator and root
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
        prompt=prompt_extract_vaccines_data(),
        template_class=VaccinesExtraction,
        extraction_name="vaccines_extraction",
    )

    # Build top-level evaluation node (parallel aggregation).
    # Note: Set as non-critical to allow optional COVID coverage child while preserving strictness inside each group.
    top = evaluator.add_parallel(
        id="Vaccine_Recommendations_Complete",
        desc="Provides the CDC-recommended vaccines specified for a 68-year-old, including vaccine type/schedule where applicable, correct Medicare Part B vs Part D no-out-of-pocket coverage, and at least one Bucks County vaccination location with a complete address.",
        parent=root,
        critical=False
    )

    # Subtree verifications
    await verify_influenza(evaluator, top, extracted)
    await verify_pneumococcal(evaluator, top, extracted)
    await verify_shingles(evaluator, top, extracted)
    await verify_tdap(evaluator, top, extracted)
    await verify_covid_optional(evaluator, top, extracted)
    await verify_local_location(evaluator, top, extracted)

    # Optionally store a brief ground-truth summary (for transparency only; not used to score)
    evaluator.add_ground_truth({
        "expected_components": [
            "Influenza: annual; preferred for 65+ are high-dose/adjuvanted/recombinant; Medicare Part B",
            "Pneumococcal: recommended for older adults; options include PCV20/PCV21 or PCV15 then PPSV23; Medicare Part B",
            "Shingles (Shingrix): recommended 50+; 2 doses 2–6 months; Medicare Part D",
            "Tdap/Td: booster every 10 years; Medicare Part D",
            "COVID-19 (optional mention): Medicare Part B coverage",
            "At least one Bucks County location with complete address and presented as a vaccination provider"
        ]
    })

    # Return structured evaluation summary
    return evaluator.get_summary()