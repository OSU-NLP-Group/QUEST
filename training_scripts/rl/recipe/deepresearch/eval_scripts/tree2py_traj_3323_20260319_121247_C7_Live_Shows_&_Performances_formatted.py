import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_broadway_theater"
TASK_DESCRIPTION = """
Identify the Broadway theater in New York City with the highest seating capacity and provide the following comprehensive information about it: 
1. The theater's name, 
2. Its exact seating capacity, 
3. Its location or address, 
4. The current production (if any) showing at the theater, 
5. The year the theater originally opened, 
6. The current owner or operating organization, 
7. Technical specifications about the stage, 
8. Notable architectural features or design elements, 
9. The theater's historical significance to Broadway, 
10. Examples of notable productions that have played there, 
11. Accessibility features for patrons with disabilities, 
12. Ticketing and box office information, 
13. Any recent renovations or updates, 
14. The seating layout configuration (orchestra, mezzanine, balcony, etc.). 
For each piece of information provided, include reference URLs that support your answer.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    """Generic text field supported by reference URLs."""
    text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class LargestBroadwayTheaterExtraction(BaseModel):
    """Structured information extracted from the agent's answer."""
    # Identification and justification
    theater_name: Optional[FieldWithSources] = None
    seating_capacity: Optional[FieldWithSources] = None  # expected to be an exact number in text form
    largest_by_capacity: Optional[FieldWithSources] = None  # statement + URLs that explicitly support "largest by capacity"

    # Requested attributes with citations
    address: Optional[FieldWithSources] = None
    theater_district: Optional[FieldWithSources] = None
    current_production: Optional[FieldWithSources] = None
    opening_year: Optional[FieldWithSources] = None
    owner_or_operator: Optional[FieldWithSources] = None
    stage_technical_specifications: Optional[FieldWithSources] = None
    architectural_or_design_features: Optional[FieldWithSources] = None
    historical_significance: Optional[FieldWithSources] = None
    notable_productions: Optional[FieldWithSources] = None
    accessibility_features: Optional[FieldWithSources] = None
    ticketing_and_box_office: Optional[FieldWithSources] = None
    recent_renovations_or_updates: Optional[FieldWithSources] = None
    seating_layout_configuration: Optional[FieldWithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_largest_broadway_theater() -> str:
    return """
    You must extract structured information about the New York City Broadway theater that the answer claims has the highest seating capacity.
    Return a single JSON object with the following keys. For each key, extract:
    - text: the exact text provided in the answer for that attribute (summarized if needed, but do not invent)
    - urls: an array of all reference URLs explicitly cited in the answer that support that specific attribute.
    
    Required keys and their meaning:
    1) theater_name: The selected theater’s name.
    2) seating_capacity: The exact seating capacity as a specific number (keep any commas or formatting as in the answer).
    3) largest_by_capacity: A short statement that the theater is the largest by seating capacity among Broadway theaters; include URLs that explicitly support this claim (e.g., ranking/list/authoritative statement).
    4) address: The theater’s location or address.
    5) theater_district: A statement indicating the theater is located in NYC’s Theater District (if provided).
    6) current_production: The currently running production title, or an explicit statement that no production is currently running.
    7) opening_year: The year the theater originally opened (just the year, if available).
    8) owner_or_operator: The current owner or the operating organization (e.g., Shubert Organization, Nederlander, etc.).
    9) stage_technical_specifications: At least one specific, verifiable stage or technical specification (e.g., stage width/depth, rigging capacity); you may summarize briefly in text.
    10) architectural_or_design_features: At least one specific architectural or design feature.
    11) historical_significance: At least one specific, verifiable historical significance fact about the theater’s importance to Broadway.
    12) notable_productions: One or more examples of notable productions that have played at the theater (list them in a comma-separated string).
    13) accessibility_features: Accessibility features or accommodations for patrons with disabilities.
    14) ticketing_and_box_office: Ticketing or box office details (e.g., box office hours, official ticketing link, policies).
    15) recent_renovations_or_updates: Recent renovations or updates; if the answer explicitly says none found, put that in text and include a URL supporting it.
    16) seating_layout_configuration: The seating layout configuration (e.g., orchestra, mezzanine, balcony levels).
    
    Special rules for URLs:
    - Only include URLs that are explicitly present in the answer. Do not invent URLs.
    - Include full URLs, valid format. If none are provided for a field, return an empty array for 'urls'.
    
    If any field is missing from the answer, set its 'text' to null and 'urls' to [].
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_text_and_urls(field: Optional[FieldWithSources]) -> bool:
    return bool(field and field.text and str(field.text).strip() and field.urls and len(field.urls) > 0)


def get_text(field: Optional[FieldWithSources]) -> str:
    return (field.text or "").strip() if field else ""


def get_urls(field: Optional[FieldWithSources]) -> List[str]:
    return field.urls if (field and field.urls) else []


def collect_all_urls(info: LargestBroadwayTheaterExtraction) -> List[str]:
    fields: List[Optional[FieldWithSources]] = [
        info.theater_name,
        info.seating_capacity,
        info.largest_by_capacity,
        info.address,
        info.theater_district,
        info.current_production,
        info.opening_year,
        info.owner_or_operator,
        info.stage_technical_specifications,
        info.architectural_or_design_features,
        info.historical_significance,
        info.notable_productions,
        info.accessibility_features,
        info.ticketing_and_box_office,
        info.recent_renovations_or_updates,
        info.seating_layout_configuration,
    ]
    urls: List[str] = []
    for f in fields:
        if f and f.urls:
            urls.extend([u for u in f.urls if isinstance(u, str) and u.strip() != ""])
    return urls


def unique_domains(urls: List[str]) -> List[str]:
    doms = []
    for u in urls:
        try:
            net = urlparse(u).netloc.lower().strip()
            if net.startswith("www."):
                net = net[4:]
            if net and net not in doms:
                doms.append(net)
        except Exception:
            continue
    return doms


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_identify_largest_theater(
    evaluator: Evaluator,
    parent_node,
    info: LargestBroadwayTheaterExtraction
) -> None:
    """
    Build and run checks under 'Identify_Largest_Theater'
    """
    identify_node = evaluator.add_parallel(
        id="Identify_Largest_Theater",
        desc="Correctly identifies which Broadway theater (in NYC) is the largest by seating capacity, with citations supporting the selection.",
        parent=parent_node,
        critical=True
    )

    # 1) Theater name with citation
    name_exist = evaluator.add_custom_node(
        result=has_text_and_urls(info.theater_name),
        id="Theater_Name_With_Citation_exists",
        desc="Theater name provided with at least one supporting URL (existence check).",
        parent=identify_node,
        critical=True
    )
    name_leaf = evaluator.add_leaf(
        id="Theater_Name_With_Citation",
        desc="Provides the selected theater’s name and at least one supporting reference URL.",
        parent=identify_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater's name is '{get_text(info.theater_name)}'.",
        node=name_leaf,
        sources=get_urls(info.theater_name),
        additional_instruction="Verify that the cited page(s) clearly state the theater's name. Minor variants (e.g., presence/absence of 'The') are acceptable.",
        extra_prerequisites=[name_exist]
    )

    # 2) Exact seating capacity (>=500) with citation
    cap_exist = evaluator.add_custom_node(
        result=has_text_and_urls(info.seating_capacity),
        id="Exact_Seating_Capacity_GTE_500_With_Citation_exists",
        desc="Exact seating capacity provided with at least one supporting URL (existence check).",
        parent=identify_node,
        critical=True
    )
    cap_leaf = evaluator.add_leaf(
        id="Exact_Seating_Capacity_GTE_500_With_Citation",
        desc="Provides the exact seating capacity as a specific number (and it is ≥500) with at least one supporting reference URL.",
        parent=identify_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The seating capacity of '{get_text(info.theater_name)}' is {get_text(info.seating_capacity)} seats, and this number is at least 500.",
        node=cap_leaf,
        sources=get_urls(info.seating_capacity),
        additional_instruction="Verify the exact capacity shown on the cited page. Treat 1,933 and 1933 as the same number. If multiple capacities are given, use the primary stated capacity. Confirm it is >= 500.",
        extra_prerequisites=[cap_exist]
    )

    # 3) Largest-by-capacity justification with citation
    largest_exist = evaluator.add_custom_node(
        result=has_text_and_urls(info.largest_by_capacity),
        id="Largest_By_Capacity_Justification_With_Citation_exists",
        desc="At least one URL is provided that explicitly supports 'largest by seating capacity' among Broadway theaters (existence check).",
        parent=identify_node,
        critical=True
    )
    largest_leaf = evaluator.add_leaf(
        id="Largest_By_Capacity_Justification_With_Citation",
        desc="Provides at least one citation that explicitly supports the claim that this theater is the highest-capacity Broadway theater (e.g., a ranking/listing or authoritative statement).",
        parent=identify_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{get_text(info.theater_name)}' has the highest seating capacity among Broadway theaters in New York City.",
        node=largest_leaf,
        sources=get_urls(info.largest_by_capacity),
        additional_instruction="Look for explicit language like 'largest Broadway theater by capacity' or a ranking/list that places this theater at the top. Synonyms like 'biggest Broadway house' are acceptable.",
        extra_prerequisites=[largest_exist]
    )


async def verify_requested_details(
    evaluator: Evaluator,
    parent_node,
    info: LargestBroadwayTheaterExtraction
) -> None:
    """
    Build and run checks under 'Provide_Requested_Details_With_Citations'
    """
    details_node = evaluator.add_parallel(
        id="Provide_Requested_Details_With_Citations",
        desc="Provides each remaining requested attribute about the identified theater, each with at least one supporting reference URL.",
        parent=parent_node,
        critical=True
    )

    theater_name_text = get_text(info.theater_name)

    # Prepare existence checks and leaves, then run a batch verification
    claims_nodes_sources: List[tuple[str, List[str], Any, Optional[str]]] = []

    # Helper to add existence node + leaf + push to batch list
    def prepare_attr(
        field: Optional[FieldWithSources],
        exist_id: str,
        leaf_id: str,
        leaf_desc: str,
        claim: str,
        add_ins: str
    ):
        exist_node = evaluator.add_custom_node(
            result=has_text_and_urls(field),
            id=exist_id,
            desc=f"{leaf_desc} (existence check for value + at least one URL).",
            parent=details_node,
            critical=True
        )
        leaf_node = evaluator.add_leaf(
            id=leaf_id,
            desc=leaf_desc,
            parent=details_node,
            critical=True
        )
        # Collect for batch verification
        claims_nodes_sources.append((
            claim,
            get_urls(field),
            leaf_node,
            add_ins
        ))
        # Note: Automatic preconditions will detect failed critical siblings (exist_node)
        # and skip verification for this leaf if the existence check fails.

    # 1) Location/Address
    prepare_attr(
        info.address,
        exist_id="Location_Or_Address_With_Citation_exists",
        leaf_id="Location_Or_Address_With_Citation",
        leaf_desc="Provides the theater’s location or address with at least one supporting reference URL.",
        claim=f"The address/location of '{theater_name_text}' is '{get_text(info.address)}'.",
        add_ins="Verify the cited page shows the stated address for the specific theater. Minor formatting differences (e.g., abbreviations) are acceptable."
    )

    # 2) Theater District location constraint
    prepare_attr(
        info.theater_district,
        exist_id="Theater_District_Location_Constraint_With_Citation_exists",
        leaf_id="Theater_District_Location_Constraint_With_Citation",
        leaf_desc="Provides at least one supporting reference URL indicating the theater is located in NYC’s Theater District.",
        claim=f"'{theater_name_text}' is located in New York City's Theater District.",
        add_ins="Look for explicit mention of 'Theater District' or authoritative confirmation that the theater is within NYC's Theater District/Midtown Broadway area."
    )

    # 3) Current production (or none)
    current_prod_text = get_text(info.current_production).lower()
    if any(kw in current_prod_text for kw in ["none", "no current", "dark", "not currently", "no production", "closed"]):
        cp_claim = f"There is currently no production playing at '{theater_name_text}'."
    else:
        cp_claim = f"The current production playing at '{theater_name_text}' is '{get_text(info.current_production)}'."
    prepare_attr(
        info.current_production,
        exist_id="Current_Production_With_Citation_exists",
        leaf_id="Current_Production_With_Citation",
        leaf_desc="States the current production playing at the theater OR explicitly states that there is no current production, with at least one supporting reference URL.",
        claim=cp_claim,
        add_ins="Verify the production status/title on the cited page. If the answer claims no production, the page should indicate the theater is dark or has no current show."
    )

    # 4) Opening year
    prepare_attr(
        info.opening_year,
        exist_id="Opening_Year_With_Citation_exists",
        leaf_id="Opening_Year_With_Citation",
        leaf_desc="Provides the year the theater originally opened with at least one supporting reference URL.",
        claim=f"'{theater_name_text}' originally opened in {get_text(info.opening_year)}.",
        add_ins="Minor differences in formatting (e.g., including month/day) are acceptable as long as the year matches."
    )

    # 5) Owner or operator
    prepare_attr(
        info.owner_or_operator,
        exist_id="Owner_Or_Operating_Organization_With_Citation_exists",
        leaf_id="Owner_Or_Operating_Organization_With_Citation",
        leaf_desc="Identifies the current owner or operating organization with at least one supporting reference URL.",
        claim=f"The current owner or operating organization for '{theater_name_text}' is '{get_text(info.owner_or_operator)}'.",
        add_ins="Verify that the cited page explicitly associates the organization with operating/owning the theater at present."
    )

    # 6) Stage technical specifications
    prepare_attr(
        info.stage_technical_specifications,
        exist_id="Stage_Technical_Specifications_With_Citation_exists",
        leaf_id="Stage_Technical_Specifications_With_Citation",
        leaf_desc="Provides at least one specific, verifiable stage/technical specification with at least one supporting reference URL.",
        claim=f"A stage/technical specification for '{theater_name_text}' is: '{get_text(info.stage_technical_specifications)}'.",
        add_ins="Confirm at least one concrete spec (e.g., dimensions, depth/width, rigging capacity) appears on the cited page."
    )

    # 7) Architectural/Design features
    prepare_attr(
        info.architectural_or_design_features,
        exist_id="Architectural_Or_Design_Features_With_Citation_exists",
        leaf_id="Architectural_Or_Design_Features_With_Citation",
        leaf_desc="Describes at least one specific architectural/design feature with at least one supporting reference URL.",
        claim=f"An architectural/design feature of '{theater_name_text}' is: '{get_text(info.architectural_or_design_features)}'.",
        add_ins="The cited page should explicitly mention the described architectural/design element."
    )

    # 8) Historical significance
    prepare_attr(
        info.historical_significance,
        exist_id="Historical_Significance_With_Citation_exists",
        leaf_id="Historical_Significance_With_Citation",
        leaf_desc="Provides at least one specific, verifiable fact about the theater’s historical significance to Broadway with at least one supporting reference URL.",
        claim=f"A historical significance fact about '{theater_name_text}' is: '{get_text(info.historical_significance)}'.",
        add_ins="Look for concrete, verifiable historical facts (e.g., landmark status, oldest/firsts, pivotal events)."
    )

    # 9) Notable productions
    prepare_attr(
        info.notable_productions,
        exist_id="Examples_Of_Notable_Productions_With_Citation_exists",
        leaf_id="Examples_Of_Notable_Productions_With_Citation",
        leaf_desc="Gives one or more examples of notable productions that have played there with at least one supporting reference URL.",
        claim=f"Examples of notable productions that have played at '{theater_name_text}' include: {get_text(info.notable_productions)}.",
        add_ins="The cited page should list or mention these productions having played at the theater."
    )

    # 10) Accessibility features
    prepare_attr(
        info.accessibility_features,
        exist_id="Accessibility_Features_With_Citation_exists",
        leaf_id="Accessibility_Features_With_Citation",
        leaf_desc="Describes accessibility features/accommodations for patrons with disabilities with at least one supporting reference URL.",
        claim=f"Accessibility features/accommodations at '{theater_name_text}' include: '{get_text(info.accessibility_features)}'.",
        add_ins="Confirm that the cited page mentions accessibility/accommodations (e.g., wheelchair access, assistive listening)."
    )

    # 11) Ticketing and box office info
    prepare_attr(
        info.ticketing_and_box_office,
        exist_id="Ticketing_And_Box_Office_Info_With_Citation_exists",
        leaf_id="Ticketing_And_Box_Office_Info_With_Citation",
        leaf_desc="Provides ticketing and/or box office information with at least one supporting reference URL.",
        claim=f"Ticketing/box office information for '{theater_name_text}': '{get_text(info.ticketing_and_box_office)}'.",
        add_ins="Verify that the cited page provides ticketing or box office details such as hours, official ticket link, or policies."
    )

    # 12) Recent renovations/updates
    prepare_attr(
        info.recent_renovations_or_updates,
        exist_id="Recent_Renovations_Or_Updates_With_Citation_exists",
        leaf_id="Recent_Renovations_Or_Updates_With_Citation",
        leaf_desc="Provides information about any recent renovations/updates, with at least one supporting reference URL (or explicitly states none found, with a URL supporting that claim).",
        claim=f"Recent renovations or updates for '{theater_name_text}': '{get_text(info.recent_renovations_or_updates)}'.",
        add_ins="If the statement is 'none found', the cited page should reasonably support that conclusion (e.g., recent news/history page without mention of updates). Otherwise, confirm the renovation/update details."
    )

    # 13) Seating layout configuration
    prepare_attr(
        info.seating_layout_configuration,
        exist_id="Seating_Layout_Configuration_With_Citation_exists",
        leaf_id="Seating_Layout_Configuration_With_Citation",
        leaf_desc="Describes the seating layout configuration (e.g., orchestra/mezzanine/balcony levels) with at least one supporting reference URL.",
        claim=f"The seating layout configuration for '{theater_name_text}' is: '{get_text(info.seating_layout_configuration)}'.",
        add_ins="Look for explicit mention of sections (e.g., orchestra, mezzanine, balcony) or a seating map that confirms the configuration."
    )

    # Execute all detail verifications in parallel
    await evaluator.batch_verify(claims_nodes_sources)


def build_multiple_sources_constraint(
    evaluator: Evaluator,
    parent_node,
    info: LargestBroadwayTheaterExtraction
) -> None:
    """
    Add the 'Multiple_Reliable_Sources_Constraint' node as a custom critical check:
    we require that the total set of cited URLs spans more than one distinct domain.
    """
    urls = collect_all_urls(info)
    domains = unique_domains(urls)
    result = len(domains) >= 2

    evaluator.add_custom_node(
        result=result,
        id="Multiple_Reliable_Sources_Constraint",
        desc="Across the provided reference URLs, citations collectively include multiple reliable sources overall (i.e., more than one reliable source is used).",
        parent=parent_node,
        critical=True
    )

    # Record helpful diagnostics
    evaluator.add_custom_info(
        info={
            "total_urls": len(urls),
            "unique_domains_count": len(domains),
            "unique_domains": domains
        },
        info_type="url_coverage",
        info_name="reference_url_statistics"
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
    Evaluate an answer for the 'Largest Broadway Theater Information' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # overall sequential: identify -> provide details -> meta constraints
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

    # Create a critical main node under the root to mirror the rubric's root criticality
    main_node = evaluator.add_sequential(
        id="Largest_Broadway_Theater_Information",
        desc="Identify the NYC Broadway theater with the highest seating capacity and provide the requested details, each supported by reference URLs, while satisfying stated constraints (e.g., Theater District location and ≥500 seats).",
        parent=root,
        critical=True
    )

    # 1) Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_largest_broadway_theater(),
        template_class=LargestBroadwayTheaterExtraction,
        extraction_name="largest_broadway_theater_extraction"
    )

    # 2) Identify largest theater with citations
    await verify_identify_largest_theater(evaluator, main_node, extracted_info)

    # 3) Provide all requested details with citations
    await verify_requested_details(evaluator, main_node, extracted_info)

    # 4) Multiple reliable sources constraint
    build_multiple_sources_constraint(evaluator, main_node, extracted_info)

    # Return standardized summary
    return evaluator.get_summary()