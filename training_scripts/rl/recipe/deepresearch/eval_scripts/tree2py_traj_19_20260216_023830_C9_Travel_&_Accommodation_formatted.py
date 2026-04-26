import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


TASK_ID = "lux_islands_itinerary_4props"
TASK_DESCRIPTION = """
A luxury travel agency specializing in island destinations is preparing a comprehensive multi-destination itinerary for discerning clients. The agency needs to identify 4 specific properties across different destinations, each meeting precise criteria related to accommodations, location, accessibility, and available activities.

Your task is to research and identify the following 4 properties:

Property 1 - Cook Islands (Aitutaki): Find the resort in Aitutaki that is uniquely positioned as the only property offering overwater bungalows with private island access. This resort must be: Located in Aitutaki, Cook Islands; Accessible via inter-island flights from Rarotonga International Airport (RAR); The sole Aitutaki property featuring overwater bungalows; Served by Air Rarotonga's inter-island service (approximately 50-minute flight from Rarotonga).

Property 2 - Cook Islands (Aitutaki): Find a different 5-star luxury resort in Aitutaki that maintains an adults-oriented atmosphere through age restrictions. This resort must be: Located in Aitutaki, Cook Islands (different from Property 1); Hold a 5-star rating; Have a minimum guest age requirement of 12 years; Be accessible via the same Rarotonga-Aitutaki flight route.

Property 3 - Portugal (Azores): Find a hotel in São Miguel island, Azores, that provides access to whale watching activities. This property must be: Located in or near Ponta Delgada city on São Miguel island; Be near Ponta Delgada Airport (PDL), the Azores' main international gateway; Provide access to whale watching tour operators (optimal season April-October); Serve as a base for exploring Azores marine wildlife.

Property 4 - United States (Charlotte, NC): Find a hotel in Charlotte, North Carolina, a city with expanding international connectivity. The city must be: Served by Charlotte Douglas International Airport (CLT); Have CLT's 5 concourses (A, B, C, D, E) with American Airlines as the hub carrier; Feature new international service: Etihad Airways' CLT-Abu Dhabi route starting March 20, 2026, operated by Boeing 787-9 Dreamliner; Have completed its Terminal Lobby Expansion in September 2025.

For each property, provide: (1) Official property name, (2) Specific location (island/city), (3) Key features matching the criteria, (4) Reference URL(s) confirming property details, airport codes, flight services, and other specified information.
"""


# ----------------------------- Data Models --------------------------------- #
class PropertyFields(BaseModel):
    official_name: Optional[str] = None
    specific_location: Optional[str] = None
    key_features_summary: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class FourPropertiesExtraction(BaseModel):
    property1: Optional[PropertyFields] = None
    property2: Optional[PropertyFields] = None
    property3: Optional[PropertyFields] = None
    property4: Optional[PropertyFields] = None


# --------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_properties() -> str:
    return """
    Extract the structured information for four properties mentioned in the answer. For each of Property 1, Property 2, Property 3, and Property 4, return a JSON object with the following fields:
    - official_name: The official property name exactly as provided in the answer (string). If missing, return null.
    - specific_location: The specific island/city location string as stated in the answer (string). If missing, return null.
    - key_features_summary: A concise summary in 1–3 sentences explaining how the property meets the requested criteria as described in the answer (string). If missing, return null.
    - reference_urls: All URLs explicitly cited in the answer for that property and/or its required facts (array of strings). These may include property websites, airline or airport pages, tourism sites, and operator pages. Extract actual URLs; include full protocol; deduplicate; if none provided, return an empty array.

    Return the object with fields: property1, property2, property3, property4, each following the schema above.

    Important:
    - Do not invent any URLs; only use those explicitly present in the answer (including inside markdown links).
    - Keep names and locations exactly as stated in the answer.
    """


# ----------------------------- Helper Utils -------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _get_sources(prop: PropertyFields) -> List[str]:
    if not prop or not isinstance(prop.reference_urls, list):
        return []
    # Deduplicate and keep non-empty strings
    seen = set()
    out: List[str] = []
    for u in prop.reference_urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out


async def _build_response_fields(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    prop: PropertyFields,
    pid_prefix: str,
    location_kind: str,
) -> Dict[str, VerificationNode]:
    """
    Create 'response_fields' group and verify location while checking other fields exist.
    Returns a dict of important leaf/custom nodes for potential prerequisites.
    """
    resp_node = evaluator.add_parallel(
        id=f"{pid_prefix}_response_fields",
        desc=f"Provide required response fields for Property {pid_prefix[-1].upper()}.",
        parent=parent_node,
        critical=True
    )

    # official_name existence
    name_exists_node = evaluator.add_custom_node(
        result=_non_empty_str(prop.official_name),
        id=f"{pid_prefix}_official_name",
        desc=f"Provide the official property name for Property {pid_prefix[-1].upper()}.",
        parent=resp_node,
        critical=True
    )

    # specific_location verification (with sources)
    loc_leaf = evaluator.add_leaf(
        id=f"{pid_prefix}_specific_location",
        desc=(
            "State the specific location and confirm it is in "
            f"{'Aitutaki, Cook Islands' if location_kind=='aitutaki' else ('in or near Ponta Delgada on São Miguel island (Azores)' if location_kind=='ponta_delgada' else 'Charlotte, North Carolina')}."
        ),
        parent=resp_node,
        critical=True
    )
    # Build claim depending on property type
    if location_kind == "aitutaki":
        location_claim = f"The property is located in Aitutaki, Cook Islands."
    elif location_kind == "ponta_delgada":
        location_claim = "The property is located in or near Ponta Delgada on São Miguel island (Azores)."
    else:
        location_claim = "The property is located in Charlotte, North Carolina."

    await evaluator.verify(
        claim=location_claim,
        node=loc_leaf,
        sources=_get_sources(prop),
        additional_instruction="Use the provided reference URLs (property site, tourism, maps, airport pages) to confirm the stated location. Allow reasonable descriptions such as 'near' when applicable."
    )

    # key_features_summary existence
    features_exists_node = evaluator.add_custom_node(
        result=_non_empty_str(prop.key_features_summary),
        id=f"{pid_prefix}_key_features_summary",
        desc=f"Provide a key-features summary describing how Property {pid_prefix[-1].upper()} meets the requested criteria.",
        parent=resp_node,
        critical=True
    )

    # reference_urls existence (at least one)
    refs_exists_node = evaluator.add_custom_node(
        result=len(_get_sources(prop)) > 0,
        id=f"{pid_prefix}_reference_urls",
        desc=f"Provide reference URL(s) that substantiate the required Property {pid_prefix[-1].upper()} claims.",
        parent=resp_node,
        critical=True
    )

    return {
        "resp_node": resp_node,
        "name_exists_node": name_exists_node,
        "features_exists_node": features_exists_node,
        "refs_exists_node": refs_exists_node,
        "location_leaf": loc_leaf
    }


# --------------------------- Property Verifiers ---------------------------- #
async def verify_property_1(
    evaluator: Evaluator,
    root: VerificationNode,
    p1: PropertyFields,
) -> None:
    prop_node = evaluator.add_parallel(
        id="property_1_cook_islands_aitutaki_unique_overwater",
        desc="Property 1 (Aitutaki): resort meeting uniqueness + access criteria, with required write-up and sources.",
        parent=root,
        critical=False
    )

    resp_nodes = await _build_response_fields(
        evaluator, prop_node, p1, "p1", location_kind="aitutaki"
    )

    sources = _get_sources(p1)

    # Only overwater in Aitutaki
    leaf_overwater = evaluator.add_leaf(
        id="p1_only_overwater_in_aitutaki",
        desc="Confirm Property 1 is the only Aitutaki property offering overwater bungalows (as required).",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="This resort is the only property in Aitutaki that offers overwater bungalows.",
        node=leaf_overwater,
        sources=sources,
        additional_instruction="Look for explicit statements such as 'only overwater bungalows in Aitutaki' or equivalent wording. Accept synonyms such as 'over-water' or 'bungalows on stilts'."
    )

    # Private island access
    leaf_private_island = evaluator.add_leaf(
        id="p1_private_island_access",
        desc="Confirm Property 1 offers private island access / private-island positioning (as stated in the question).",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="The resort offers private island access or is located on a private island.",
        node=leaf_private_island,
        sources=sources,
        additional_instruction="Verify that the resort is on a private motu/island or provides exclusive private-island access."
    )

    # Air Rarotonga inter-island flights from RAR
    leaf_air_rar = evaluator.add_leaf(
        id="p1_air_rarotonga_from_rar",
        desc="Confirm Property 1 is accessible via Air Rarotonga inter-island flights from Rarotonga International Airport (RAR) to Aitutaki.",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="Air Rarotonga operates inter-island flights from Rarotonga International Airport (RAR) to Aitutaki.",
        node=leaf_air_rar,
        sources=sources,
        additional_instruction="Prefer official airline or tourism sources. Confirm RAR→Aitutaki service."
    )

    # Flight duration approximately 50 minutes
    leaf_flight_dur = evaluator.add_leaf(
        id="p1_flight_duration_approx_50_min",
        desc="Confirm the Rarotonga → Aitutaki flight duration is approximately 50 minutes (as stated).",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="The flight time from Rarotonga to Aitutaki is approximately 50 minutes.",
        node=leaf_flight_dur,
        sources=sources,
        additional_instruction="Allow reasonable approximations (e.g., 45–55 minutes) as 'approximately 50 minutes'. Prefer airline/tourism sources."
    )


async def verify_property_2(
    evaluator: Evaluator,
    root: VerificationNode,
    p2: PropertyFields,
    p1: PropertyFields,
) -> None:
    prop_node = evaluator.add_parallel(
        id="property_2_cook_islands_aitutaki_5star_age12",
        desc="Property 2 (Aitutaki): different from Property 1; 5-star; minimum age 12; accessible via RAR↔Aitutaki route; with required write-up and sources.",
        parent=root,
        critical=False
    )

    resp_nodes = await _build_response_fields(
        evaluator, prop_node, p2, "p2", location_kind="aitutaki"
    )

    sources2 = _get_sources(p2)
    sources1 = _get_sources(p1)
    combined_sources = sources1 + [u for u in sources2 if u not in sources1]

    # Distinct from Property 1
    leaf_distinct = evaluator.add_leaf(
        id="p2_distinct_from_p1",
        desc="Confirm Property 2 is a different resort/property than Property 1.",
        parent=prop_node,
        critical=True
    )
    distinct_claim = f"Property 2 is a different resort/property than Property 1."
    await evaluator.verify(
        claim=distinct_claim,
        node=leaf_distinct,
        sources=combined_sources,
        additional_instruction="Use the referenced property websites and sources to confirm they are distinct entities (different names, branding, addresses)."
    )

    # Five-star rating
    leaf_5star = evaluator.add_leaf(
        id="p2_five_star_rating",
        desc="Confirm Property 2 has a 5-star rating (as required).",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="This property holds a 5-star rating.",
        node=leaf_5star,
        sources=sources2,
        additional_instruction="Look for official statements, accreditation, or recognized listings that explicitly indicate '5-star'."
    )

    # Minimum age 12
    leaf_age12 = evaluator.add_leaf(
        id="p2_minimum_age_12",
        desc="Confirm Property 2 has a minimum guest age requirement of 12 years.",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="This property has a minimum guest age requirement of at least 12 years.",
        node=leaf_age12,
        sources=sources2,
        additional_instruction="Accept wording like '12 years and older', 'minimum age 12', or 'adults-oriented with 12+ policy'."
    )

    # Accessible via RAR ↔ Aitutaki flights
    leaf_access_rar = evaluator.add_leaf(
        id="p2_accessible_via_rar_aitutaki_flights",
        desc="Confirm Property 2 is accessible via the Rarotonga (RAR) ↔ Aitutaki flight route.",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property is accessible via flights between Rarotonga (RAR) and Aitutaki.",
        node=leaf_access_rar,
        sources=sources2,
        additional_instruction="Confirm that typical access involves flying RAR↔Aitutaki (e.g., via Air Rarotonga)."
    )


async def verify_property_3(
    evaluator: Evaluator,
    root: VerificationNode,
    p3: PropertyFields,
) -> None:
    prop_node = evaluator.add_parallel(
        id="property_3_azores_sao_miguel_whale_watching",
        desc="Property 3 (São Miguel/Azores): in/near Ponta Delgada; near PDL; access to whale watching; Apr–Oct seasonal note; with required write-up and sources.",
        parent=root,
        critical=False
    )

    resp_nodes = await _build_response_fields(
        evaluator, prop_node, p3, "p3", location_kind="ponta_delgada"
    )

    sources = _get_sources(p3)

    # Near PDL airport
    leaf_near_pdl = evaluator.add_leaf(
        id="p3_near_pdl_airport",
        desc="Confirm Property 3 is near Ponta Delgada Airport (PDL).",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property is near Ponta Delgada Airport (PDL).",
        node=leaf_near_pdl,
        sources=sources,
        additional_instruction="Accept short drive proximity. PDL refers to João Paulo II Airport in Ponta Delgada."
    )

    # PDL main gateway claim
    leaf_pdl_gateway = evaluator.add_leaf(
        id="p3_pdl_main_gateway_claim",
        desc="Confirm PDL is the Azores’ main international gateway (as stated).",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="PDL is the Azores’ main international gateway.",
        node=leaf_pdl_gateway,
        sources=sources,
        additional_instruction="Prefer official airport/airline/tourism sources indicating PDL's gateway status."
    )

    # Access to whale watching operators
    leaf_whale_ops = evaluator.add_leaf(
        id="p3_access_to_whale_watching_operators",
        desc="Confirm the property/location provides access to whale watching tour operators / serves as a base for marine wildlife exploration.",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property/location provides access to whale watching tour operators and serves as a base to explore Azores marine wildlife.",
        node=leaf_whale_ops,
        sources=sources,
        additional_instruction="Look for whale-watching operators based in/near Ponta Delgada (e.g., marina-based). Confirm the hotel’s proximity or access arrangement."
    )

    # Whale season April–October
    leaf_whale_season = evaluator.add_leaf(
        id="p3_whale_season_apr_oct",
        desc="Confirm whale watching in the Azores is most favorable from April to October (as stated).",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="Whale watching in the Azores is most favorable from April to October.",
        node=leaf_whale_season,
        sources=sources,
        additional_instruction="Prefer operator or official tourism sources indicating seasonal peaks (April–October). Allow minor variations like late March or early November being shoulder season."
    )


async def verify_property_4(
    evaluator: Evaluator,
    root: VerificationNode,
    p4: PropertyFields,
) -> None:
    prop_node = evaluator.add_parallel(
        id="property_4_charlotte_nc_hotel_and_clt_facts",
        desc="Property 4 (Charlotte, NC): hotel in Charlotte; CLT facts; Terminal Lobby Expansion (Sep 2025); Etihad CLT–Abu Dhabi (Mar 20, 2026, 787-9); with required write-up and sources.",
        parent=root,
        critical=False
    )

    resp_nodes = await _build_response_fields(
        evaluator, prop_node, p4, "p4", location_kind="charlotte"
    )

    sources = _get_sources(p4)

    # Charlotte served by CLT
    leaf_served_by_clt = evaluator.add_leaf(
        id="p4_charlotte_served_by_clt",
        desc="Confirm Charlotte is served by Charlotte Douglas International Airport (CLT).",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="Charlotte is served by Charlotte Douglas International Airport (CLT).",
        node=leaf_served_by_clt,
        sources=sources,
        additional_instruction="Prefer official airport/city sources confirming CLT serves Charlotte."
    )

    # CLT has 5 concourses A–E
    leaf_5_concourses = evaluator.add_leaf(
        id="p4_clt_has_5_concourses_abcde",
        desc="Confirm CLT has 5 concourses labeled A, B, C, D, and E.",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="Charlotte Douglas International Airport (CLT) has five concourses labeled A, B, C, D, and E.",
        node=leaf_5_concourses,
        sources=sources,
        additional_instruction="Prefer official CLT materials (terminal map, facilities page) listing concourses A–E."
    )

    # American Airlines hub carrier at CLT
    leaf_aa_hub = evaluator.add_leaf(
        id="p4_american_airlines_hub_carrier",
        desc="Confirm American Airlines is the primary hub carrier at CLT (as stated).",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="American Airlines is the primary hub carrier at CLT.",
        node=leaf_aa_hub,
        sources=sources,
        additional_instruction="Prefer airport/corporate sources stating CLT is an AA hub."
    )

    # Terminal Lobby Expansion September 2025
    leaf_tle_sep2025 = evaluator.add_leaf(
        id="p4_terminal_lobby_expansion_sep_2025",
        desc="Confirm CLT's Terminal Lobby Expansion was completed in September 2025.",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="CLT's Terminal Lobby Expansion was completed in September 2025.",
        node=leaf_tle_sep2025,
        sources=sources,
        additional_instruction="Prefer CLT official news releases or project updates confirming completion date."
    )

    # Etihad CLT–Abu Dhabi start date Mar 20, 2026
    leaf_etihad_start = evaluator.add_leaf(
        id="p4_etihad_clt_auh_service_start_date",
        desc="Confirm Etihad launched CLT → Abu Dhabi service on March 20, 2026 (as stated).",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="Etihad launched CLT–Abu Dhabi service on March 20, 2026.",
        node=leaf_etihad_start,
        sources=sources,
        additional_instruction="Prefer Etihad press releases or credible aviation news confirming the start date."
    )

    # Etihad aircraft 787-9
    leaf_etihad_7879 = evaluator.add_leaf(
        id="p4_etihad_clt_auh_aircraft_7879",
        desc="Confirm the Etihad CLT → Abu Dhabi route is operated using Boeing 787-9 Dreamliner aircraft (as stated).",
        parent=prop_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Etihad CLT–Abu Dhabi route is operated by Boeing 787-9 Dreamliner.",
        node=leaf_etihad_7879,
        sources=sources,
        additional_instruction="Prefer airline or credible news sources confirming aircraft type."
    )


# ------------------------- Main Evaluation Function ------------------------ #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Build the verification tree and evaluate the agent's answer against the rubric.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Properties evaluated independently
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

    # Extract 4 properties from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=FourPropertiesExtraction,
        extraction_name="properties_extraction"
    )

    # Fallbacks if any property block is missing
    p1 = extracted.property1 or PropertyFields()
    p2 = extracted.property2 or PropertyFields()
    p3 = extracted.property3 or PropertyFields()
    p4 = extracted.property4 or PropertyFields()

    # Verify each property according to rubric
    await verify_property_1(evaluator, root, p1)
    await verify_property_2(evaluator, root, p2, p1)
    await verify_property_3(evaluator, root, p3)
    await verify_property_4(evaluator, root, p4)

    # Return structured evaluation summary
    return evaluator.get_summary()