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
TASK_ID = "ut_amphitheater_1995"
TASK_DESCRIPTION = (
    "Identify the outdoor amphitheater in Utah that opened in 1995, is surrounded by 1,500-foot red rock cliffs, "
    "and produces Broadway-style musicals. Provide the following information about this amphitheater: "
    "(1) The seating capacity of the outdoor amphitheater, "
    "(2) The specific city in Utah where it is located, "
    "(3) The name(s) of the founder(s) who established it, "
    "(4) The name of the canyon at whose mouth the amphitheater is located, "
    "(5) Whether it is currently operational. "
    "For each piece of information, include reference URLs that support your answer."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AmphitheaterIdentification(BaseModel):
    amphitheater_name: Optional[str] = None
    identification_sources: List[str] = Field(default_factory=list)


class AmphitheaterDetails(BaseModel):
    seating_capacity: Optional[str] = None
    seating_capacity_urls: List[str] = Field(default_factory=list)

    city: Optional[str] = None
    city_urls: List[str] = Field(default_factory=list)

    founders: List[str] = Field(default_factory=list)
    founders_urls: List[str] = Field(default_factory=list)

    canyon_name: Optional[str] = None
    canyon_urls: List[str] = Field(default_factory=list)

    currently_operational: Optional[str] = None  # Accepts free-form (e.g., "yes", "no", "currently operating", etc.)
    operational_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_identification() -> str:
    return (
        "You must extract the identification information for the Utah outdoor amphitheater described in the answer. "
        "Specifically extract:\n"
        "1) amphitheater_name: The full proper name of the amphitheater identified in the answer.\n"
        "2) identification_sources: A list of URLs explicitly cited in the answer that support the amphitheater's identification and its key characteristics "
        "(being in Utah, opened in 1995, surrounded by ~1,500-foot red rock cliffs, producing Broadway-style musicals, and being an outdoor venue).\n\n"
        "Rules:\n"
        "- Only include URLs explicitly present in the answer (plain URLs or markdown links). Do not invent or infer URLs.\n"
        "- If no URLs are provided, return an empty list for identification_sources.\n"
        "- If the amphitheater name is not provided, return null for amphitheater_name.\n"
    )


def prompt_extract_details() -> str:
    return (
        "Extract the requested attributes for the identified amphitheater and, for each attribute, list the explicit supporting URLs provided in the answer. "
        "Return the following fields:\n"
        "1) seating_capacity: The seating capacity.\n"
        "   seating_capacity_urls: A list of URLs in the answer that support the stated capacity (prefer official/reliable sources if present).\n"
        "2) city: The specific city in Utah where the amphitheater is located.\n"
        "   city_urls: URLs in the answer that support the stated city.\n"
        "3) founders: The list of names of the founder(s) who established it.\n"
        "   founders_urls: URLs in the answer that support the founders.\n"
        "4) canyon_name: The name of the canyon at whose mouth the amphitheater is located.\n"
        "   canyon_urls: URLs in the answer that support the canyon name.\n"
        "5) currently_operational: Whether it is currently operational (free text, e.g., 'yes', 'no', 'operational', 'open').\n"
        "   operational_urls: URLs in the answer that support the current operational status.\n\n"
        "Rules:\n"
        "- Only include URLs explicitly present in the answer (plain URLs or markdown links). Do not invent or infer URLs.\n"
        "- If a value is missing in the answer, set it to null (or an empty list for founders). If URLs are missing for a value, return an empty list for its URLs field.\n"
        "- Prefer official or otherwise reliable sources when they are available in the answer, but still list any URLs that are provided.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def safe_name(ident: AmphitheaterIdentification) -> str:
    return ident.amphitheater_name.strip() if ident and ident.amphitheater_name else "the amphitheater identified in the answer"


def names_list_to_text(names: List[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def operational_claim_text(value: Optional[str]) -> str:
    """
    Convert a free-form 'currently_operational' value into a verification claim.
    """
    if not value:
        # default to a positive claim if unspecified (will likely fail unless sources support it)
        return "It is currently operational."
    v = value.strip().lower()
    positives = ["yes", "true", "operational", "currently operational", "open", "currently open", "operating", "in operation", "in-season", "season underway", "active"]
    negatives = ["no", "false", "not operational", "closed", "defunct", "ceased", "inactive"]
    if any(p in v for p in positives):
        return "It is currently operational."
    if any(n in v for n in negatives):
        return "It is not currently operational."
    # fallback: use the original text in a neutral yes/no claim reader
    return f"The amphitheater's current operational status can be described as: {value}."


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_identification(
    evaluator: Evaluator,
    parent_node,
    ident: AmphitheaterIdentification,
) -> None:
    """
    Build the 'Amphitheater_Identification' subtree and run verifications.
    """
    # Create the identification parent node (critical; parallel aggregation)
    ident_node = evaluator.add_parallel(
        id="Amphitheater_Identification",
        desc="Correctly identify an amphitheater that satisfies all stated identification constraints, with supporting reference URL(s).",
        parent=parent_node,
        critical=True,
    )

    # Existence of identification reference URLs (critical)
    ident_urls_node = evaluator.add_custom_node(
        result=bool(ident.identification_sources),
        id="Identification_Reference_URLs",
        desc="Provide reference URL(s) that support the amphitheater identification and the above constraints.",
        parent=ident_node,
        critical=True,
    )

    amph_name = safe_name(ident)

    # Utah location
    utah_loc_node = evaluator.add_leaf(
        id="Utah_Location",
        desc="Amphitheater is located in the state of Utah.",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{amph_name} is located in Utah.",
        node=utah_loc_node,
        sources=ident.identification_sources,
        additional_instruction="Verify using the provided URLs that the amphitheater is in the U.S. state of Utah.",
        extra_prerequisites=[ident_urls_node],
    )

    # Opened in 1995
    opened_1995_node = evaluator.add_leaf(
        id="Opened_1995",
        desc="Amphitheater opened in 1995.",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{amph_name} opened in 1995.",
        node=opened_1995_node,
        sources=ident.identification_sources,
        additional_instruction="Confirm that the amphitheater's opening year is 1995 (allow phrasing like 'since 1995', 'opened in 1995').",
        extra_prerequisites=[ident_urls_node],
    )

    # Surrounded by ~1,500-foot red rock cliffs
    cliffs_node = evaluator.add_leaf(
        id="Surrounded_By_1500ft_Red_Rock_Cliffs",
        desc="Amphitheater is surrounded by 1,500-foot red rock cliffs.",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{amph_name} is surrounded by red rock cliffs that rise around 1,500 feet.",
        node=cliffs_node,
        sources=ident.identification_sources,
        additional_instruction=(
            "Look for language indicating towering red rock cliffs about 1,500 feet; "
            "allow reasonable approximations (e.g., 'approximately 1,500 feet', '1,500-foot cliffs')."
        ),
        extra_prerequisites=[ident_urls_node],
    )

    # Produces Broadway-style musicals
    musicals_node = evaluator.add_leaf(
        id="Produces_Broadway_Style_Musicals",
        desc="Amphitheater produces Broadway-style musicals.",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{amph_name} produces Broadway-style musicals (Broadway-caliber/level musicals).",
        node=musicals_node,
        sources=ident.identification_sources,
        additional_instruction=(
            "Accept equivalent phrasing such as 'Broadway-style', 'Broadway-caliber', 'Broadway-level' musicals. "
            "The sources should explicitly describe the amphitheater's production of such musicals."
        ),
        extra_prerequisites=[ident_urls_node],
    )

    # Outdoor venue
    outdoor_node = evaluator.add_leaf(
        id="Outdoor_Venue",
        desc="Amphitheater is an outdoor venue.",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{amph_name} is an outdoor amphitheater.",
        node=outdoor_node,
        sources=ident.identification_sources,
        additional_instruction="Confirm that the venue is outdoors (open-air amphitheater).",
        extra_prerequisites=[ident_urls_node],
    )


async def build_and_verify_details(
    evaluator: Evaluator,
    parent_node,
    ident: AmphitheaterIdentification,
    details: AmphitheaterDetails,
) -> None:
    """
    Build the 'Required_Amphitheater_Details_With_Sources' subtree and run verifications.
    """
    amph_name = safe_name(ident)

    details_node = evaluator.add_parallel(
        id="Required_Amphitheater_Details_With_Sources",
        desc="Provide each requested attribute, and include at least one supporting reference URL for each attribute.",
        parent=parent_node,
        critical=True,
    )

    # Seating Capacity
    capacity_exists_node = evaluator.add_custom_node(
        result=(bool(details.seating_capacity) and bool(details.seating_capacity_urls)),
        id="Seating_Capacity_Provided",
        desc="Seating capacity value and at least one supporting URL are provided.",
        parent=details_node,
        critical=True,
    )
    capacity_leaf = evaluator.add_leaf(
        id="Seating_Capacity_With_Official_Or_Reliable_Source_URL",
        desc="Provide the seating capacity AND a supporting reference URL from an official or otherwise reliable source.",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The seating capacity of {amph_name} is {details.seating_capacity}.",
        node=capacity_leaf,
        sources=details.seating_capacity_urls,
        additional_instruction=(
            "Verify the stated seating capacity from the provided URLs. Prefer official or otherwise reliable sources if present. "
            "Allow minor variations, ranges, or approximations (e.g., 'about 2,000', '2,000+')."
        ),
        extra_prerequisites=[capacity_exists_node],
    )

    # City in Utah
    city_exists_node = evaluator.add_custom_node(
        result=(bool(details.city) and bool(details.city_urls)),
        id="City_Provided",
        desc="City name and at least one supporting URL are provided.",
        parent=details_node,
        critical=True,
    )
    city_leaf = evaluator.add_leaf(
        id="City_In_Utah_With_URL",
        desc="Provide the specific city in Utah where it is located AND a supporting reference URL.",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{amph_name} is located in {details.city}, Utah.",
        node=city_leaf,
        sources=details.city_urls,
        additional_instruction=(
            "Verify the amphitheater's city in Utah using the provided URLs. "
            "Allow reasonable naming variants (e.g., 'Ivins' vs. 'Ivins City')."
        ),
        extra_prerequisites=[city_exists_node],
    )

    # Founder names
    founders_text = names_list_to_text(details.founders)
    founders_exists_node = evaluator.add_custom_node(
        result=(bool(details.founders) and bool(details.founders_urls)),
        id="Founder_Names_Provided",
        desc="Founder name(s) and at least one supporting URL are provided.",
        parent=details_node,
        critical=True,
    )
    founders_leaf = evaluator.add_leaf(
        id="Founder_Names_With_URL",
        desc="Provide the name(s) of the founder(s) who established it AND a supporting reference URL.",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{amph_name} was established by {founders_text}.",
        node=founders_leaf,
        sources=details.founders_urls,
        additional_instruction=(
            "Verify the founders from the provided URLs. "
            "Accept cases where founders are listed alongside partners or organizations if the page clearly indicates founding."
        ),
        extra_prerequisites=[founders_exists_node],
    )

    # Canyon name
    canyon_exists_node = evaluator.add_custom_node(
        result=(bool(details.canyon_name) and bool(details.canyon_urls)),
        id="Canyon_Name_Provided",
        desc="Canyon name and at least one supporting URL are provided.",
        parent=details_node,
        critical=True,
    )
    canyon_leaf = evaluator.add_leaf(
        id="Canyon_Name_With_URL",
        desc="Provide the name of the canyon at whose mouth it is located AND a supporting reference URL.",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{amph_name} is located at the mouth of {details.canyon_name}.",
        node=canyon_leaf,
        sources=details.canyon_urls,
        additional_instruction=(
            "Verify the canyon name using the provided URLs. "
            "Allow contextual phrasing like 'at the mouth of Padre Canyon' or 'situated in/near [canyon name]'."
        ),
        extra_prerequisites=[canyon_exists_node],
    )

    # Current operational status
    operational_exists_node = evaluator.add_custom_node(
        result=(details.currently_operational is not None and bool(details.operational_urls)),
        id="Operational_Status_Provided",
        desc="Operational status text and at least one supporting URL are provided.",
        parent=details_node,
        critical=True,
    )
    operational_leaf = evaluator.add_leaf(
        id="Current_Operational_Status_With_URL",
        desc="Indicate whether it is currently operational AND a supporting reference URL.",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim=operational_claim_text(details.currently_operational),
        node=operational_leaf,
        sources=details.operational_urls,
        additional_instruction=(
            "Use the provided URLs (e.g., official site pages like 'Season', 'Tickets', schedules, or recent news) to verify whether the amphitheater is currently operational. "
            "Accept evidence of active programming, ticket sales, upcoming shows, or statements confirming ongoing operation."
        ),
        extra_prerequisites=[operational_exists_node],
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
    Evaluate an answer to the Utah amphitheater identification and information task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # sequential as per rubric root
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

    # Extract identification and details (can be done concurrently)
    ident_task = evaluator.extract(
        prompt=prompt_extract_identification(),
        template_class=AmphitheaterIdentification,
        extraction_name="amphitheater_identification",
    )
    details_task = evaluator.extract(
        prompt=prompt_extract_details(),
        template_class=AmphitheaterDetails,
        extraction_name="amphitheater_details",
    )
    ident_info, details_info = await asyncio.gather(ident_task, details_task)

    # Build and verify identification subtree
    await build_and_verify_identification(evaluator, root, ident_info)

    # Build and verify details subtree (sequential dependency will auto skip if identification fails)
    await build_and_verify_details(evaluator, root, ident_info, details_info)

    # Optionally add custom info summary
    evaluator.add_custom_info(
        info={
            "amphitheater_name": ident_info.amphitheater_name,
            "identification_sources_count": len(ident_info.identification_sources),
            "details": {
                "seating_capacity": details_info.seating_capacity,
                "seating_capacity_urls_count": len(details_info.seating_capacity_urls),
                "city": details_info.city,
                "city_urls_count": len(details_info.city_urls),
                "founders": details_info.founders,
                "founders_urls_count": len(details_info.founders_urls),
                "canyon_name": details_info.canyon_name,
                "canyon_urls_count": len(details_info.canyon_urls),
                "currently_operational": details_info.currently_operational,
                "operational_urls_count": len(details_info.operational_urls),
            },
        },
        info_type="extraction_overview",
    )

    return evaluator.get_summary()