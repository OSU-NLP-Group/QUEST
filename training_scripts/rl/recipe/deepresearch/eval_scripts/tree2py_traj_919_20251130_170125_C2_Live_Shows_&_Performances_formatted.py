import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "columbus_spanish_baroque_venue"
TASK_DESCRIPTION = "Identify the historic performing arts venue in Columbus, Ohio that features Spanish-Baroque architecture. Provide its exact seating capacity and describe one of its notable architectural features, with supporting references from official or authoritative sources."


class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    seating_capacity: Optional[str] = None
    notable_feature: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


def prompt_extract_venue_info() -> str:
    return (
        "From the answer, extract the single historic performing arts venue in Columbus, Ohio that is described as having Spanish-Baroque architecture. "
        "Return a JSON object with the following fields:\n"
        "1) venue_name: The exact name of the venue identified.\n"
        "2) seating_capacity: The exact seating capacity stated in the answer (as a specific number or numeric string). If not provided, return null.\n"
        "3) notable_feature: A description of one notable architectural feature of the venue (a short phrase or sentence). If not provided, return null.\n"
        "4) source_urls: An array of the URLs that the answer cites as references or sources for this venue. Extract only actual URLs explicitly present in the answer; include all relevant official or authoritative sources if they are present. If none are provided, return an empty array.\n"
        "If multiple venues are mentioned, select the one that matches Spanish-Baroque architecture and appears to be the main subject. If any field is missing, set it to null."
    )


async def verify_venue_identification(evaluator: Evaluator, parent_node, extracted: VenueExtraction) -> None:
    identification_node = evaluator.add_parallel(
        id="Venue_Identification",
        desc="Identify a venue that satisfies the required identification constraints.",
        parent=parent_node,
        critical=True,
    )

    name_provided = bool(extracted.venue_name and extracted.venue_name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id="Venue_Name_Provided",
        desc="The venue name is provided in the answer.",
        parent=identification_node,
        critical=True,
    )

    # Columbus, Ohio location
    loc_node = evaluator.add_leaf(
        id="Columbus_Ohio_Location",
        desc="The identified venue is located in Columbus, Ohio.",
        parent=identification_node,
        critical=True,
    )
    loc_claim = f"{extracted.venue_name} is located in Columbus, Ohio."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=extracted.source_urls,
        additional_instruction="Verify that the page explicitly shows the venue in Columbus, Ohio. Accept 'Columbus, OH' as equivalent to 'Columbus, Ohio'."
    )

    # Performing arts facility
    paf_node = evaluator.add_leaf(
        id="Performing_Arts_Facility",
        desc="The identified venue is a performing arts facility/venue.",
        parent=identification_node,
        critical=True,
    )
    paf_claim = f"{extracted.venue_name} is a performing arts venue or theater."
    await evaluator.verify(
        claim=paf_claim,
        node=paf_node,
        sources=extracted.source_urls,
        additional_instruction="Confirm the venue is used for performing arts: theater, concert hall, opera house, performing arts center, etc."
    )

    # Spanish-Baroque architecture
    sba_node = evaluator.add_leaf(
        id="Spanish_Baroque_Architecture",
        desc="The identified venue features Spanish-Baroque architectural design.",
        parent=identification_node,
        critical=True,
    )
    sba_claim = f"{extracted.venue_name} features Spanish-Baroque architectural design."
    await evaluator.verify(
        claim=sba_claim,
        node=sba_node,
        sources=extracted.source_urls,
        additional_instruction="Look for explicit mention of 'Spanish Baroque' or 'Spanish Baroque Revival'. Allow closely equivalent phrasing like 'Spanish-style Baroque' or 'Baroque Revival with Spanish details' if clearly referring to the same architectural classification."
    )

    # Historic status
    hist_node = evaluator.add_leaf(
        id="Historic_Status",
        desc="The identified venue is historic (listed on the National Register of Historic Places or has an official state historic designation).",
        parent=identification_node,
        critical=True,
    )
    hist_claim = f"{extracted.venue_name} is listed on the National Register of Historic Places or has an official state historic designation."
    await evaluator.verify(
        claim=hist_claim,
        node=hist_node,
        sources=extracted.source_urls,
        additional_instruction="Verify that the page shows NRHP listing (e.g., listing date, reference number) or a recognized official state/city historic designation. Prefer official/government or organizational sources when available."
    )


async def verify_venue_details(evaluator: Evaluator, parent_node, extracted: VenueExtraction) -> None:
    details_node = evaluator.add_parallel(
        id="Venue_Details",
        desc="Provide the required factual details about the identified venue.",
        parent=parent_node,
        critical=True,
    )

    # Seating capacity existence
    cap_provided = bool(extracted.seating_capacity and extracted.seating_capacity.strip())
    evaluator.add_custom_node(
        result=cap_provided,
        id="Seating_Capacity_Provided",
        desc="Seating capacity is provided in the answer.",
        parent=details_node,
        critical=True,
    )

    # Seating capacity verification
    cap_node = evaluator.add_leaf(
        id="Seating_Capacity_Number",
        desc="Provide the venue’s exact seating capacity as a specific number.",
        parent=details_node,
        critical=True,
    )
    cap_claim = f"The seating capacity of {extracted.venue_name} is {extracted.seating_capacity}."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_node,
        sources=extracted.source_urls,
        additional_instruction="Check for the exact capacity number on authoritative pages. Minor typographical variants are acceptable only if clearly the same number; otherwise require an exact match."
    )

    # Notable architectural feature existence
    feature_provided = bool(extracted.notable_feature and extracted.notable_feature.strip())
    evaluator.add_custom_node(
        result=feature_provided,
        id="Feature_Provided",
        desc="At least one notable architectural feature is provided in the answer.",
        parent=details_node,
        critical=True,
    )

    # Notable architectural feature verification
    feature_node = evaluator.add_leaf(
        id="Notable_Architectural_Feature_Description",
        desc="Identify and describe at least one notable architectural feature of the venue.",
        parent=details_node,
        critical=True,
    )
    feature_claim = f"{extracted.venue_name} has {extracted.notable_feature}."
    await evaluator.verify(
        claim=feature_claim,
        node=feature_node,
        sources=extracted.source_urls,
        additional_instruction="Confirm that the specified feature is clearly described on the source page as a notable or distinctive architectural element of the venue. Allow synonyms and close paraphrases."
    )


async def verify_authoritative_references(evaluator: Evaluator, parent_node, extracted: VenueExtraction) -> None:
    refs_node = evaluator.add_parallel(
        id="Authoritative_References",
        desc="Provide official or authoritative sources sufficient to verify the key required claims (location, Spanish-Baroque style, historic status, seating capacity, and the architectural feature).",
        parent=parent_node,
        critical=True,
    )

    sources_exist = bool(extracted.source_urls and len(extracted.source_urls) > 0)
    evaluator.add_custom_node(
        result=sources_exist,
        id="Sources_Provided",
        desc="At least one source URL is provided in the answer.",
        parent=refs_node,
        critical=True,
    )

    # Authoritative source supports location
    auth_loc_node = evaluator.add_leaf(
        id="Authoritative_Location_Support",
        desc="Authoritative source supports Columbus, Ohio location.",
        parent=refs_node,
        critical=True,
    )
    auth_loc_claim = f"This page is an official or authoritative source and explicitly states that {extracted.venue_name} is located in Columbus, Ohio."
    await evaluator.verify(
        claim=auth_loc_claim,
        node=auth_loc_node,
        sources=extracted.source_urls,
        additional_instruction="Consider authoritative sources as: official venue or operator websites (e.g., organization that runs the theater), government (.gov) or state sites, major cultural institutions, National Register documentation, or other well-established publications. The page must clearly state the venue is in Columbus, Ohio."
    )

    # Authoritative source supports Spanish-Baroque style
    auth_style_node = evaluator.add_leaf(
        id="Authoritative_Style_Support",
        desc="Authoritative source supports Spanish-Baroque architectural style.",
        parent=refs_node,
        critical=True,
    )
    auth_style_claim = f"This page is an official or authoritative source and explicitly states that {extracted.venue_name} features Spanish-Baroque (or Spanish Baroque Revival) architecture."
    await evaluator.verify(
        claim=auth_style_claim,
        node=auth_style_node,
        sources=extracted.source_urls,
        additional_instruction="Prefer official or authoritative pages. Accept 'Spanish Baroque' or 'Spanish Baroque Revival' and clearly equivalent phrasing."
    )

    # Authoritative source supports historic status
    auth_hist_node = evaluator.add_leaf(
        id="Authoritative_Historic_Status_Support",
        desc="Authoritative source supports historic status (NRHP or official state designation).",
        parent=refs_node,
        critical=True,
    )
    auth_hist_claim = f"This page is an official or authoritative source and confirms that {extracted.venue_name} is historic, such as being listed on the National Register of Historic Places or having an official state designation."
    await evaluator.verify(
        claim=auth_hist_claim,
        node=auth_hist_node,
        sources=extracted.source_urls,
        additional_instruction="NRHP listings, state historic registers, or official governmental or institutional pages count as authoritative. The page should clearly confirm the status."
    )

    # Authoritative source supports seating capacity
    auth_cap_node = evaluator.add_leaf(
        id="Authoritative_Capacity_Support",
        desc="Authoritative source supports the exact seating capacity.",
        parent=refs_node,
        critical=True,
    )
    auth_cap_claim = f"This page is an official or authoritative source and confirms that the seating capacity of {extracted.venue_name} is {extracted.seating_capacity}."
    await evaluator.verify(
        claim=auth_cap_claim,
        node=auth_cap_node,
        sources=extracted.source_urls,
        additional_instruction="Prefer official venue/operator pages, government, or institutional sites. Confirm the exact capacity number."
    )

    # Authoritative source supports notable feature
    auth_feat_node = evaluator.add_leaf(
        id="Authoritative_Feature_Support",
        desc="Authoritative source supports the described notable architectural feature.",
        parent=refs_node,
        critical=True,
    )
    auth_feat_claim = f"This page is an official or authoritative source and clearly describes {extracted.notable_feature} as a notable architectural feature of {extracted.venue_name}."
    await evaluator.verify(
        claim=auth_feat_claim,
        node=auth_feat_node,
        sources=extracted.source_urls,
        additional_instruction="Prefer official or authoritative sources. The page should explicitly mention or describe the specified feature as notable or distinctive."
    )


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    main_node = evaluator.add_sequential(
        id="Historic_Venue_Research",
        desc="Identify a historic Columbus, Ohio performing arts venue with Spanish-Baroque architecture and provide required details with authoritative sourcing.",
        parent=root,
        critical=True,
    )

    await verify_venue_identification(evaluator, main_node, extracted)
    await verify_venue_details(evaluator, main_node, extracted)
    await verify_authoritative_references(evaluator, main_node, extracted)

    return evaluator.get_summary()