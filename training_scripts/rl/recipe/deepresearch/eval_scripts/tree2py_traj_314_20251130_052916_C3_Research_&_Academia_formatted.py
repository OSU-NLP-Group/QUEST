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
TASK_ID = "stanford_shark_commbio_2025"
TASK_DESCRIPTION = (
    "In October 2025, a research paper was published in the journal Communications Biology documenting the discovery "
    "of five fossilized vertebrae from a cardabiodontid shark found in the Darwin Formation of northern Australia. "
    "The fossils are approximately 115 million years old (upper Aptian period), and the largest vertebrae exceed 12 cm "
    "in diameter, indicating the shark was 6-8 meters in length. The discovery pushes back the evolutionary timeline of "
    "mega-body size in lamniform sharks by approximately 15 million years. Find this research paper and identify its lead "
    "author. The lead author is affiliated with Stanford University's Department of Earth and Planetary Sciences. Then, "
    "locate the current chair of that department and provide: (1) the department chair's official Stanford email address, "
    "and (2) either the chair's primary research focus area or office location (building and room number)."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperDetails(BaseModel):
    paper_title: Optional[str] = None
    journal_name: Optional[str] = None
    publication_date: Optional[str] = None  # e.g., "October 2025"
    paper_url: Optional[str] = None
    other_urls: List[str] = Field(default_factory=list)

    vertebrae_count: Optional[str] = None                       # Expect strings like "five", "5"
    shark_taxon: Optional[str] = None                           # e.g., "cardabiodontid", "Cardabiodontidae"
    formation_location: Optional[str] = None                    # e.g., "Darwin Formation, northern Australia"
    fossil_age_mya: Optional[str] = None                        # e.g., "approximately 115 million years"
    period: Optional[str] = None                                # e.g., "upper Aptian"
    vertebrae_diameter_cm: Optional[str] = None                 # e.g., ">12 cm", "exceed 12 cm"
    estimated_length_m: Optional[str] = None                    # e.g., "6–8 meters", "6-8 m"
    evolutionary_significance: Optional[str] = None             # e.g., "pushes back by ~15 million years"


class LeadAuthorDetails(BaseModel):
    lead_author_name: Optional[str] = None
    affiliation_university: Optional[str] = None                # e.g., "Stanford University"
    affiliation_department: Optional[str] = None                # e.g., "Department of Earth and Planetary Sciences"
    lead_author_urls: List[str] = Field(default_factory=list)   # URLs that support lead author & affiliation


class ChairDetails(BaseModel):
    chair_name: Optional[str] = None
    chair_official_listing_url: Optional[str] = None            # Official department page that lists chair
    chair_profile_url: Optional[str] = None                     # Individual profile page (optional)
    chair_email: Optional[str] = None                           # Must be @stanford.edu
    chair_research_focus: Optional[str] = None                  # Primary research focus (optional)
    chair_office_location: Optional[str] = None                 # Building + room number (optional)
    additional_urls: List[str] = Field(default_factory=list)    # Other official sources


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_details() -> str:
    return (
        "From the answer, extract the details of the Communications Biology paper that matches the described fossil "
        "discovery. Return a JSON object with the following fields:\n"
        "1. paper_title: The title of the paper.\n"
        "2. journal_name: The journal name (e.g., 'Communications Biology').\n"
        "3. publication_date: Month and year as stated (e.g., 'October 2025').\n"
        "4. paper_url: The URL to the paper's webpage (preferably the journal or publisher page).\n"
        "5. other_urls: An array of any other URLs cited for this paper.\n"
        "6. vertebrae_count: The count of fossilized vertebrae described.\n"
        "7. shark_taxon: The taxonomic reference (e.g., 'cardabiodontid', 'Cardabiodontidae').\n"
        "8. formation_location: Geologic formation and region (e.g., 'Darwin Formation, northern Australia').\n"
        "9. fossil_age_mya: The stated age in million years (e.g., 'approximately 115 million years').\n"
        "10. period: The geologic period/age (e.g., 'upper Aptian').\n"
        "11. vertebrae_diameter_cm: Statement about largest vertebrae diameter (e.g., 'exceed 12 cm').\n"
        "12. estimated_length_m: Estimated shark length (e.g., '6–8 meters').\n"
        "13. evolutionary_significance: Statement about pushing back mega-body size timeline by ~15 million years.\n"
        "If any field is not explicitly present, return null, and for other_urls return an empty array if none."
    )


def prompt_extract_lead_author_details() -> str:
    return (
        "From the answer, extract the lead author's information for the identified paper. "
        "Treat 'lead author' as the first listed author unless otherwise explicitly defined (e.g., 'lead author', 'first author'). "
        "Return a JSON object with:\n"
        "1. lead_author_name: The name of the lead author.\n"
        "2. affiliation_university: The university affiliation (e.g., 'Stanford University').\n"
        "3. affiliation_department: The department affiliation (e.g., 'Department of Earth and Planetary Sciences').\n"
        "4. lead_author_urls: An array of URLs that support the lead author's identity and affiliation (e.g., paper page, Stanford profile).\n"
        "If any field is not present, return null, and for lead_author_urls return an empty array if none."
    )


def prompt_extract_chair_details() -> str:
    return (
        "From the answer, extract details for the current chair of Stanford University's Department of Earth and Planetary Sciences. "
        "Return a JSON object with:\n"
        "1. chair_name: The chair's name.\n"
        "2. chair_official_listing_url: The official department webpage URL that lists the current chair.\n"
        "3. chair_profile_url: The chair's individual Stanford profile page URL (if provided).\n"
        "4. chair_email: The chair's official Stanford email address.\n"
        "5. chair_research_focus: The chair's primary research focus area (if provided).\n"
        "6. chair_office_location: The chair's office location including building and room number (if provided).\n"
        "7. additional_urls: Any other official URLs supporting the chair's identity, email, or details.\n"
        "If any field is not present, return null, and for URL lists return empty arrays if none."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _collect_sources(*url_lists: List[Optional[str]]) -> List[str]:
    urls: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if u and isinstance(u, str) and u.strip():
                urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def _first_non_empty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v and v.strip():
            return v.strip()
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_paper_identification(
    evaluator: Evaluator,
    parent_node,
    paper: PaperDetails,
) -> None:
    """
    Build the Paper_Identification parallel node with critical leaves verifying the paper constraints.
    """
    paper_node = evaluator.add_parallel(
        id="Paper_Identification",
        desc="Identify the correct Communications Biology paper published in October 2025 matching the specified fossil discovery constraints",
        parent=parent_node,
        critical=True,
    )

    # Existence gate: Require a paper URL to validate against evidence
    evaluator.add_custom_node(
        result=bool(paper.paper_url and paper.paper_url.strip()),
        id="Paper_URL_Provided",
        desc="Paper URL is provided to support verification",
        parent=paper_node,
        critical=True,
    )

    sources_list = _collect_sources([paper.paper_url] if paper.paper_url else [], paper.other_urls)

    # Correct_Journal
    node_journal = evaluator.add_leaf(
        id="Correct_Journal",
        desc="Paper is published in Communications Biology",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This paper is published in Communications Biology.",
        node=node_journal,
        sources=sources_list,
        additional_instruction="Confirm the journal name on the paper page. It should explicitly say 'Communications Biology'.",
    )

    # Correct_Publication_Date
    node_pubdate = evaluator.add_leaf(
        id="Correct_Publication_Date",
        desc="Paper publication date is October 2025",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This paper's publication date is October 2025.",
        node=node_pubdate,
        sources=sources_list,
        additional_instruction="Check the paper page for the publication date and confirm the month is October and the year is 2025.",
    )

    # Vertebrae_Count
    node_vert_count = evaluator.add_leaf(
        id="Vertebrae_Count",
        desc="Paper documents discovery of five fossilized vertebrae",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The paper documents the discovery of five fossilized vertebrae.",
        node=node_vert_count,
        sources=sources_list,
        additional_instruction="Look for language like 'five vertebrae' or equivalent phrasing that clearly indicates the count is five.",
    )

    # Shark_Taxon
    node_taxon = evaluator.add_leaf(
        id="Shark_Taxon",
        desc="Fossils are from a cardabiodontid shark (Cardabiodontidae)",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The fossils are from a cardabiodontid shark (family Cardabiodontidae).",
        node=node_taxon,
        sources=sources_list,
        additional_instruction="Confirm that the taxon is identified as 'cardabiodontid' or 'Cardabiodontidae' on the page.",
    )

    # Geologic_Formation_Location
    node_geo = evaluator.add_leaf(
        id="Geologic_Formation_Location",
        desc="Fossils are from the Darwin Formation in northern Australia",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The fossils were found in the Darwin Formation in northern Australia.",
        node=node_geo,
        sources=sources_list,
        additional_instruction="Verify the locality information mentions the Darwin Formation and northern Australia.",
    )

    # Fossil_Age_Period
    node_age = evaluator.add_leaf(
        id="Fossil_Age_Period",
        desc="Fossils are approximately 115 million years old (upper Aptian period)",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The fossils are approximately 115 million years old and from the upper Aptian period.",
        node=node_age,
        sources=sources_list,
        additional_instruction="Confirm both the age (~115 million years) and the period (upper Aptian) are stated on the paper page.",
    )

    # Vertebrae_Diameter
    node_diam = evaluator.add_leaf(
        id="Vertebrae_Diameter",
        desc="Largest vertebrae exceed 12 cm in diameter",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The largest vertebrae exceed 12 cm in diameter.",
        node=node_diam,
        sources=sources_list,
        additional_instruction="Look for quantitative measurements indicating the largest vertebrae are >12 cm.",
    )

    # Estimated_Shark_Length
    node_len = evaluator.add_leaf(
        id="Estimated_Shark_Length",
        desc="Estimated shark length is 6–8 meters",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The estimated shark length is between 6 and 8 meters.",
        node=node_len,
        sources=sources_list,
        additional_instruction="Confirm that the paper estimates the shark's length in the range of 6–8 meters.",
    )

    # Evolutionary_Significance
    node_sig = evaluator.add_leaf(
        id="Evolutionary_Significance",
        desc="Discovery pushes back the evolutionary timeline of mega-body size in lamniform sharks by approximately 15 million years",
        parent=paper_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The discovery pushes back the evolutionary timeline of mega-body size in lamniform sharks by approximately 15 million years.",
        node=node_sig,
        sources=sources_list,
        additional_instruction="Confirm the statement about pushing back the timeline by ~15 million years appears in the paper or its abstract.",
    )


async def build_lead_author_and_affiliation(
    evaluator: Evaluator,
    parent_node,
    paper: PaperDetails,
    lead: LeadAuthorDetails,
) -> None:
    """
    Build the Lead_Author_and_Affiliation parallel node with critical leaves verifying identity and Stanford affiliation.
    """
    lead_node = evaluator.add_parallel(
        id="Lead_Author_and_Affiliation",
        desc="Identify the paper's lead author and verify required affiliation details",
        parent=parent_node,
        critical=True,
    )

    # Existence gate: Lead author name must be provided
    evaluator.add_custom_node(
        result=bool(lead.lead_author_name and lead.lead_author_name.strip()),
        id="Lead_Author_Name_Provided",
        desc="Lead author name string is provided",
        parent=lead_node,
        critical=True,
    )

    sources_list = _collect_sources(
        [paper.paper_url] if paper.paper_url else [],
        lead.lead_author_urls
    )

    # Lead_Author_Name
    node_lead_name = evaluator.add_leaf(
        id="Lead_Author_Name",
        desc="Lead author is identified (name provided)",
        parent=lead_node,
        critical=True,
    )
    paper_title_for_claim = _first_non_empty(paper.paper_title, "the paper")
    await evaluator.verify(
        claim=f"The lead author of the paper '{paper_title_for_claim}' is {lead.lead_author_name}.",
        node=node_lead_name,
        sources=sources_list,
        additional_instruction="Treat 'lead author' as the first listed author unless explicitly defined otherwise.",
    )

    # Lead_Author_Stanford_Affiliation
    node_lead_affil = evaluator.add_leaf(
        id="Lead_Author_Stanford_Affiliation",
        desc="Lead author is affiliated with Stanford University",
        parent=lead_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The lead author is affiliated with Stanford University.",
        node=node_lead_affil,
        sources=sources_list,
        additional_instruction="Confirm that the lead author's affiliation includes 'Stanford University' on the paper or official profile.",
    )

    # Lead_Author_Department
    node_lead_dept = evaluator.add_leaf(
        id="Lead_Author_Department",
        desc="Lead author's department is Stanford University's Department of Earth and Planetary Sciences",
        parent=lead_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The lead author's department is Stanford University's Department of Earth and Planetary Sciences.",
        node=node_lead_dept,
        sources=sources_list,
        additional_instruction="Look for explicit mention of 'Department of Earth and Planetary Sciences' at Stanford.",
    )


async def build_department_chair_details(
    evaluator: Evaluator,
    parent_node,
    chair: ChairDetails,
) -> None:
    """
    Build the Department_Chair_Details parallel node with critical leaves for chair identity, email, and additional info.
    """
    chair_node = evaluator.add_parallel(
        id="Department_Chair_Details",
        desc="Locate the current chair of Stanford's Department of Earth and Planetary Sciences and provide required contact/details",
        parent=parent_node,
        critical=True,
    )

    # Existence gates: listing URL and email should be present to proceed
    evaluator.add_custom_node(
        result=bool(chair.chair_official_listing_url and chair.chair_official_listing_url.strip()),
        id="Chair_Listing_URL_Provided",
        desc="Official department listing URL for chair is provided",
        parent=chair_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(chair.chair_name and chair.chair_name.strip()),
        id="Chair_Name_Provided",
        desc="Chair name is provided",
        parent=chair_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(chair.chair_email and chair.chair_email.strip()),
        id="Chair_Email_Provided",
        desc="Chair email is provided",
        parent=chair_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(chair.chair_email and "@stanford.edu" in (chair.chair_email or "").lower()),
        id="Chair_Email_Domain_Valid",
        desc="Chair email is within the @stanford.edu domain",
        parent=chair_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool((chair.chair_research_focus and chair.chair_research_focus.strip()) or (chair.chair_office_location and chair.chair_office_location.strip())),
        id="Chair_Additional_Info_Provided",
        desc="Either chair's primary research focus or office location is provided",
        parent=chair_node,
        critical=True,
    )

    sources_list = _collect_sources(
        [chair.chair_official_listing_url] if chair.chair_official_listing_url else [],
        [chair.chair_profile_url] if chair.chair_profile_url else [],
        chair.additional_urls
    )

    # Chair_Identity_Official_Department_Listing
    node_chair_identity = evaluator.add_leaf(
        id="Chair_Identity_Official_Department_Listing",
        desc="Chair identified is the current chair as listed on the official Department of Earth and Planetary Sciences website",
        parent=chair_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The current chair of Stanford's Department of Earth and Planetary Sciences is {chair.chair_name}.",
        node=node_chair_identity,
        sources=sources_list,
        additional_instruction="Confirm the individual is listed as 'Department Chair' or equivalent title on the official E&PS departmental website.",
    )

    # Chair_Stanford_Email
    node_chair_email = evaluator.add_leaf(
        id="Chair_Stanford_Email",
        desc="Provide the chair's official Stanford email address (must be in the @stanford.edu domain)",
        parent=chair_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The chair's official Stanford email address is {chair.chair_email}.",
        node=node_chair_email,
        sources=sources_list,
        additional_instruction="Verify that the listed email belongs to the identified chair and is presented on an official Stanford page.",
    )

    # Chair_Additional_Info
    chosen_info = None
    info_kind = None
    if chair.chair_research_focus and chair.chair_research_focus.strip():
        chosen_info = chair.chair_research_focus.strip()
        info_kind = "primary research focus area"
    elif chair.chair_office_location and chair.chair_office_location.strip():
        chosen_info = chair.chair_office_location.strip()
        info_kind = "office location (building and room number)"
    else:
        chosen_info = ""

    node_chair_extra = evaluator.add_leaf(
        id="Chair_Additional_Info",
        desc="Provide either (a) the chair's primary research focus area OR (b) the chair's office location (building and room number)",
        parent=chair_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The chair's {info_kind} is '{chosen_info}'.",
        node=node_chair_extra,
        sources=sources_list,
        additional_instruction="Confirm the provided detail is explicitly stated on the official department page or the chair's official Stanford profile.",
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
    Evaluate the answer for locating the Communications Biology paper, lead author, and Stanford E&PS chair details.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is non-critical; create a critical sequential child below
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

    # Extract structured information from the answer (can be parallelized)
    paper_task = evaluator.extract(
        prompt=prompt_extract_paper_details(),
        template_class=PaperDetails,
        extraction_name="paper_details",
    )
    lead_task = evaluator.extract(
        prompt=prompt_extract_lead_author_details(),
        template_class=LeadAuthorDetails,
        extraction_name="lead_author_details",
    )
    chair_task = evaluator.extract(
        prompt=prompt_extract_chair_details(),
        template_class=ChairDetails,
        extraction_name="chair_details",
    )
    paper_details, lead_details, chair_details = await asyncio.gather(paper_task, lead_task, chair_task)

    # Build the critical sequential investigation node as per rubric
    investigation_node = evaluator.add_sequential(
        id="Research_Investigation",
        desc="Complete sequential investigation from paper identification through department chair details",
        parent=root,
        critical=True,
    )

    # Build subtrees following rubric
    await build_paper_identification(evaluator, investigation_node, paper_details)
    await build_lead_author_and_affiliation(evaluator, investigation_node, paper_details, lead_details)
    await build_department_chair_details(evaluator, investigation_node, chair_details)

    # Return structured summary
    return evaluator.get_summary()