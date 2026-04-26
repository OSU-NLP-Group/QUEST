import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "uark_sic_fab_2025"
TASK_DESCRIPTION = (
    "Identify the facility, principal investigator, research center acronym, IEEE PELS award name, and the founding "
    "director's publicly listed phone/email, while satisfying all stated constraints: In 2025, a unique silicon "
    "carbide (SiC) fabrication facility was dedicated in the United States, located in Arkansas at the Arkansas "
    "Research and Technology Park, costing $95 million, spanning 22,000 square feet, funded in part by NSF's "
    "Mid-Scale Research Infrastructure Program, and dedicated in November 2025; The facility is described as the only "
    "openly accessible SiC fabrication facility in the U.S. Identify the principal investigator who is a Distinguished "
    "Professor at the University of Arkansas, holds the Twenty-First Century Research Leadership Chair in Engineering, "
    "named a 2025 NAI Fellow (announcement in Dec 2025), founded an NSF I/UCRC in 2009 focusing on grid-connected "
    "advanced power electronic systems and serves as its founding director; Provide the acronym of that center; "
    "Provide the full name of the 2025 IEEE Power Electronics Society award (R. David Middlebrook Achievement Award) "
    "received by the PI with citation mentioning contributions to power semiconductor device modeling and packaging; "
    "Lastly, provide the publicly listed phone number and email address for the founding director on the center’s website."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityExtraction(BaseModel):
    facility_name: Optional[str] = None
    facility_sources: List[str] = Field(default_factory=list)


class PIExtraction(BaseModel):
    pi_name: Optional[str] = None
    pi_sources: List[str] = Field(default_factory=list)


class CenterExtraction(BaseModel):
    center_acronym: Optional[str] = None
    center_name: Optional[str] = None
    center_sources: List[str] = Field(default_factory=list)
    founding_director_phone: Optional[str] = None
    founding_director_email: Optional[str] = None
    center_contact_sources: List[str] = Field(default_factory=list)


class AwardExtraction(BaseModel):
    award_full_name: Optional[str] = None
    award_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facility() -> str:
    return (
        "Extract the silicon carbide (SiC) fabrication facility described in the answer.\n"
        "Return a JSON object containing:\n"
        "• facility_name: the name/identifier of the facility\n"
        "• facility_sources: an array of URLs explicitly cited in the answer that support facts about the facility. "
        "These should include official university pages, press releases, NSF pages, ARTP pages, or credible news pages "
        "that corroborate details such as dedication date, location in Arkansas at ARTP, $95M cost, 22,000 sq ft size, "
        "SiC specialization, and funding by NSF Mid-Scale Research Infrastructure Program.\n"
        "If any field is missing, set it to null or an empty array."
    )


def prompt_extract_pi() -> str:
    return (
        "Extract the principal investigator (PI) associated with the facility.\n"
        "Return a JSON object containing:\n"
        "• pi_name: the PI's full name\n"
        "• pi_sources: an array of URLs explicitly cited in the answer that support facts about the PI, including being "
        "the facility's principal investigator, Distinguished Professor at the University of Arkansas, holding the "
        "Twenty-First Century Research Leadership Chair in Engineering, being named a 2025 NAI Fellow (announced in "
        "December 2025).\n"
        "If any field is missing, set it to null or an empty array."
    )


def prompt_extract_center() -> str:
    return (
        "Extract the research center founded by the PI in 2009 that focuses on grid-connected advanced power electronic "
        "systems and for which the PI serves as founding director.\n"
        "Return a JSON object containing:\n"
        "• center_acronym: the acronym of the center (e.g., GRAPES)\n"
        "• center_name: the full name of the center (if provided)\n"
        "• center_sources: an array of URLs explicitly cited in the answer that support the center being an NSF I/UCRC, "
        "its focus on grid-connected advanced power electronic systems, the founding year (2009), and the PI being "
        "listed as founding director.\n"
        "• founding_director_phone: the publicly listed phone number for the founding director on the center website\n"
        "• founding_director_email: the publicly listed email address for the founding director on the center website\n"
        "• center_contact_sources: an array of URLs (preferably center official website pages) where the contact "
        "information is publicly listed\n"
        "If any field is missing, set it to null or an empty array."
    )


def prompt_extract_award() -> str:
    return (
        "Extract the IEEE Power Electronics Society (PELS) 2025 award received by the PI.\n"
        "Return a JSON object containing:\n"
        "• award_full_name: the full name of the award (e.g., 'R. David Middlebrook Achievement Award')\n"
        "• award_sources: an array of URLs explicitly cited in the answer that support the award details and citation "
        "mentioning contributions to power semiconductor device modeling and packaging.\n"
        "If any field is missing, set it to null or an empty array."
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: List[str]) -> List[str]:
    """Combine and deduplicate URL lists, filter out empties."""
    seen = set()
    out: List[str] = []
    for lst in lists:
        for url in lst or []:
            if not url:
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification step builders                                                  #
# --------------------------------------------------------------------------- #
async def build_step_1_facility(
    evaluator: Evaluator,
    parent_node,
    facility: FacilityExtraction,
):
    """Step 1: Facility verification (parallel)."""
    step_node = evaluator.add_parallel(
        id="Step_1_Facility",
        desc="Identify the SiC fabrication facility and ensure it matches all facility constraints.",
        parent=parent_node,
        critical=False,
    )

    fac_name = (facility.facility_name or "").strip()
    fac_sources = facility.facility_sources or []

    # Provide facility name (existence check)
    evaluator.add_custom_node(
        result=bool(fac_name),
        id="Provide_Facility_Name",
        desc="Provide the name/identifier of the SiC fabrication facility.",
        parent=step_node,
        critical=True,
    )

    # Facility dedicated in Nov 2025 (US)
    node_dedicated = evaluator.add_leaf(
        id="Facility_Dedicated_Nov_2025",
        desc="Facility was dedicated in November 2025 (in the United States).",
        parent=step_node,
        critical=True,
    )
    claim_dedicated = (
        f"The facility '{fac_name}' was dedicated in November 2025 in the United States."
        if fac_name else "The facility was dedicated in November 2025 in the United States."
    )
    await evaluator.verify(
        claim=claim_dedicated,
        node=node_dedicated,
        sources=fac_sources,
        additional_instruction="Confirm the dedication month/year and location context on the cited pages. Minor wording variations are acceptable.",
    )

    # Facility specializes in SiC
    node_sic = evaluator.add_leaf(
        id="Facility_Specializes_in_SiC",
        desc="Facility specializes in silicon carbide (SiC) technology.",
        parent=step_node,
        critical=True,
    )
    claim_sic = (
        f"The facility '{fac_name}' specializes in silicon carbide (SiC) technology."
        if fac_name else "The facility specializes in silicon carbide (SiC) technology."
    )
    await evaluator.verify(
        claim=claim_sic,
        node=node_sic,
        sources=fac_sources,
        additional_instruction="Check that the facility is explicitly described as focusing on SiC technology.",
    )

    # Only openly accessible SiC fab in US
    node_open = evaluator.add_leaf(
        id="Facility_Only_Openly_Accessible_SiC_Fab_US",
        desc="Facility is described as the only openly accessible SiC fabrication facility in the United States.",
        parent=step_node,
        critical=True,
    )
    claim_open = (
        f"The facility '{fac_name}' is described as the only openly accessible SiC fabrication facility in the United States."
        if fac_name else "The facility is described as the only openly accessible SiC fabrication facility in the United States."
    )
    await evaluator.verify(
        claim=claim_open,
        node=node_open,
        sources=fac_sources,
        additional_instruction="Look for phrasing indicating uniqueness and open access of the SiC fabrication capability.",
    )

    # Location: Arkansas at ARTP
    node_loc = evaluator.add_leaf(
        id="Facility_Location_Arkansas_ARTP",
        desc="Facility is located in Arkansas at the Arkansas Research and Technology Park.",
        parent=step_node,
        critical=True,
    )
    claim_loc = (
        f"The facility '{fac_name}' is located in Arkansas at the Arkansas Research and Technology Park."
        if fac_name else "The facility is located in Arkansas at the Arkansas Research and Technology Park."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=node_loc,
        sources=fac_sources,
        additional_instruction="Confirm the location details precisely, including the ARTP mention.",
    )

    # Cost: $95M
    node_cost = evaluator.add_leaf(
        id="Facility_Cost_95M",
        desc="Facility total project cost is $95 million.",
        parent=step_node,
        critical=True,
    )
    claim_cost = (
        f"The facility '{fac_name}' has a total project cost of $95 million."
        if fac_name else "The facility has a total project cost of $95 million."
    )
    await evaluator.verify(
        claim=claim_cost,
        node=node_cost,
        sources=fac_sources,
        additional_instruction="Confirm the stated total project cost; allow '$95M' or '95 million'.",
    )

    # Size: 22,000 sq ft
    node_size = evaluator.add_leaf(
        id="Facility_Size_22000_sqft",
        desc="Facility size is 22,000 square feet.",
        parent=step_node,
        critical=True,
    )
    claim_size = (
        f"The facility '{fac_name}' spans approximately 22,000 square feet."
        if fac_name else "The facility spans approximately 22,000 square feet."
    )
    await evaluator.verify(
        claim=claim_size,
        node=node_size,
        sources=fac_sources,
        additional_instruction="Confirm the facility size; allow minor formatting differences (e.g., commas, 'sq ft').",
    )

    # Funding: NSF Mid-Scale Research Infrastructure Program
    node_mid = evaluator.add_leaf(
        id="Facility_Funded_By_NSF_MidScale",
        desc="Facility was funded in part by the NSF Mid-Scale Research Infrastructure Program.",
        parent=step_node,
        critical=True,
    )
    claim_mid = (
        f"The facility '{fac_name}' was funded in part by the National Science Foundation's Mid-Scale Research Infrastructure Program."
        if fac_name else "The facility was funded in part by the National Science Foundation's Mid-Scale Research Infrastructure Program."
    )
    await evaluator.verify(
        claim=claim_mid,
        node=node_mid,
        sources=fac_sources,
        additional_instruction="Verify that an NSF Mid-Scale (MSRI) Program contribution is explicitly stated.",
    )


async def build_step_2_pi(
    evaluator: Evaluator,
    parent_node,
    facility: FacilityExtraction,
    pi: PIExtraction,
):
    """Step 2: Principal Investigator verification (parallel)."""
    step_node = evaluator.add_parallel(
        id="Step_2_Principal_Investigator",
        desc="Identify the principal investigator and ensure all PI constraints are satisfied.",
        parent=parent_node,
        critical=False,
    )

    pi_name = (pi.pi_name or "").strip()
    src = _combine_sources(pi.pi_sources, facility.facility_sources)

    # Provide PI name (existence)
    evaluator.add_custom_node(
        result=bool(pi_name),
        id="Provide_PI_Name",
        desc="Provide the name of the facility's principal investigator.",
        parent=step_node,
        critical=True,
    )

    # PI is principal investigator of facility
    node_pi_of_fac = evaluator.add_leaf(
        id="PI_Is_Principal_Investigator_of_Facility",
        desc="The provided person is the principal investigator of the identified facility.",
        parent=step_node,
        critical=True,
    )
    claim_pi_of_fac = (
        f"{pi_name} is the principal investigator of the facility '{facility.facility_name or ''}'."
        if pi_name else "The provided person is the principal investigator of the identified facility."
    )
    await evaluator.verify(
        claim=claim_pi_of_fac,
        node=node_pi_of_fac,
        sources=src,
        additional_instruction="Confirm PI role linked to the named facility; allow minor title variations.",
    )

    # Distinguished Professor at UArk
    node_dist_prof = evaluator.add_leaf(
        id="PI_Distinguished_Professor_UArk",
        desc="PI is a Distinguished Professor at the University of Arkansas.",
        parent=step_node,
        critical=True,
    )
    claim_dist_prof = f"{pi_name} is a Distinguished Professor at the University of Arkansas." if pi_name else "The PI is a Distinguished Professor at the University of Arkansas."
    await evaluator.verify(
        claim=claim_dist_prof,
        node=node_dist_prof,
        sources=src,
        additional_instruction="Confirm the Distinguished Professor title at UArk; allow department variations.",
    )

    # Holds Twenty-First Century Research Leadership Chair in Engineering
    node_chair = evaluator.add_leaf(
        id="PI_Holds_21st_Century_Research_Leadership_Chair",
        desc="PI holds the Twenty-First Century Research Leadership Chair in Engineering.",
        parent=step_node,
        critical=True,
    )
    claim_chair = f"{pi_name} holds the Twenty-First Century Research Leadership Chair in Engineering." if pi_name else "The PI holds the Twenty-First Century Research Leadership Chair in Engineering."
    await evaluator.verify(
        claim=claim_chair,
        node=node_chair,
        sources=src,
        additional_instruction="Confirm the specific chair title; minor wording variations acceptable.",
    )

    # NAI Fellow 2025 announced Dec 2025
    node_nai = evaluator.add_leaf(
        id="PI_NAI_Fellow_2025_Announced_Dec_2025",
        desc="PI was named a 2025 Fellow of the National Academy of Inventors, with the announcement made in December 2025.",
        parent=step_node,
        critical=True,
    )
    claim_nai = (
        f"In December 2025, it was announced that {pi_name} was named a 2025 Fellow of the National Academy of Inventors."
        if pi_name else "In December 2025, it was announced that the PI was named a 2025 Fellow of the National Academy of Inventors."
    )
    await evaluator.verify(
        claim=claim_nai,
        node=node_nai,
        sources=src,
        additional_instruction="Confirm both the Fellow year (2025) and announcement timing (Dec 2025).",
    )


async def build_step_3_center(
    evaluator: Evaluator,
    parent_node,
    center: CenterExtraction,
    pi: PIExtraction,
):
    """Step 3: Research Center verification (parallel)."""
    step_node = evaluator.add_parallel(
        id="Step_3_Research_Center",
        desc="Identify the NSF I/UCRC founded in 2009 by the PI and provide its acronym; verify center constraints.",
        parent=parent_node,
        critical=False,
    )

    acr = (center.center_acronym or "").strip()
    pi_name = (pi.pi_name or "").strip()
    src_center = _combine_sources(center.center_sources)
    src_all = _combine_sources(center.center_sources, pi.pi_sources)

    # Provide center acronym (existence)
    evaluator.add_custom_node(
        result=bool(acr),
        id="Provide_Center_Acronym",
        desc="Provide the acronym of the research center founded by the PI in 2009.",
        parent=step_node,
        critical=True,
    )

    # Center is NSF I/UCRC
    node_iucrc = evaluator.add_leaf(
        id="Center_Is_NSF_IUCRC",
        desc="The center is an NSF Industry/University Cooperative Research Center (I/UCRC).",
        parent=step_node,
        critical=True,
    )
    claim_iucrc = f"The {acr} center is an NSF Industry/University Cooperative Research Center (I/UCRC)." if acr else "The center is an NSF Industry/University Cooperative Research Center (I/UCRC)."
    await evaluator.verify(
        claim=claim_iucrc,
        node=node_iucrc,
        sources=src_center or src_all,
        additional_instruction="Confirm 'NSF I/UCRC' designation from official center/NSF sources.",
    )

    # Center founded 2009 by PI
    node_founded = evaluator.add_leaf(
        id="Center_Founded_2009_By_PI",
        desc="The PI founded the center in 2009.",
        parent=step_node,
        critical=True,
    )
    claim_founded = (
        f"{pi_name} founded the {acr} center in 2009."
        if pi_name and acr else "The PI founded the center in 2009."
    )
    await evaluator.verify(
        claim=claim_founded,
        node=node_founded,
        sources=src_center or src_all,
        additional_instruction="Confirm the founding year (2009) and founder identity (PI).",
    )

    # Center focus: grid-connected advanced power electronic systems
    node_focus = evaluator.add_leaf(
        id="Center_Focus_GridConnected_Advanced_Power_Electronic_Systems",
        desc="The center focuses on grid-connected advanced power electronic systems.",
        parent=step_node,
        critical=True,
    )
    claim_focus = (
        f"The {acr} center focuses on grid-connected advanced power electronic systems."
        if acr else "The center focuses on grid-connected advanced power electronic systems."
    )
    await evaluator.verify(
        claim=claim_focus,
        node=node_focus,
        sources=src_center or src_all,
        additional_instruction="Confirm phrasing indicating the center's focus on grid-connected advanced power electronic systems.",
    )

    # PI is listed as founding director
    node_fd = evaluator.add_leaf(
        id="PI_Is_Founding_Director_Listed",
        desc="The PI is listed as the founding director of the center.",
        parent=step_node,
        critical=True,
    )
    claim_fd = (
        f"{pi_name} is listed as the founding director of the {acr} center."
        if pi_name and acr else "The PI is listed as the founding director of the center."
    )
    await evaluator.verify(
        claim=claim_fd,
        node=node_fd,
        sources=src_center or src_all,
        additional_instruction="Confirm that the center's official site or credible sources list the PI as founding director.",
    )


async def build_step_4_award(
    evaluator: Evaluator,
    parent_node,
    award: AwardExtraction,
    pi: PIExtraction,
):
    """Step 4: IEEE PELS award verification (parallel)."""
    step_node = evaluator.add_parallel(
        id="Step_4_IEEE_PELS_Award",
        desc="Provide the full name of the PI's 2025 IEEE PELS award and verify award constraints.",
        parent=parent_node,
        critical=False,
    )

    award_name = (award.award_full_name or "").strip()
    pi_name = (pi.pi_name or "").strip()
    src = _combine_sources(award.award_sources, pi.pi_sources)

    # Provide award full name (existence)
    evaluator.add_custom_node(
        result=bool(award_name),
        id="Provide_Award_Full_Name",
        desc="Provide the full name of the 2025 IEEE Power Electronics Society award received by the PI.",
        parent=step_node,
        critical=True,
    )

    # Award is IEEE PELS 2025
    node_pels_2025 = evaluator.add_leaf(
        id="Award_Is_IEEE_PELS_2025",
        desc="PI received a 2025 award from the IEEE Power Electronics Society.",
        parent=step_node,
        critical=True,
    )
    claim_pels_2025 = f"In 2025, {pi_name} received an award from the IEEE Power Electronics Society." if pi_name else "In 2025, the PI received an award from the IEEE Power Electronics Society."
    await evaluator.verify(
        claim=claim_pels_2025,
        node=node_pels_2025,
        sources=src,
        additional_instruction="Confirm year (2025) and awarding body (IEEE PELS).",
    )

    # Award is R. David Middlebrook Achievement Award
    node_middlebrook = evaluator.add_leaf(
        id="Award_Is_R_David_Middlebrook_Achievement_Award",
        desc="The award is the R. David Middlebrook Achievement Award.",
        parent=step_node,
        critical=True,
    )
    claim_middlebrook = f"The award received by {pi_name} is the R. David Middlebrook Achievement Award." if pi_name else "The award is the R. David Middlebrook Achievement Award."
    await evaluator.verify(
        claim=claim_middlebrook,
        node=node_middlebrook,
        sources=src,
        additional_instruction="Confirm the exact award name; allow minor punctuation variations.",
    )

    # Citation mentions contributions to device modeling and packaging
    node_citation = evaluator.add_leaf(
        id="Award_Citation_Device_Modeling_Packaging",
        desc="The award citation mentions contributions to power semiconductor device modeling and packaging.",
        parent=step_node,
        critical=True,
    )
    claim_citation = "The award citation mentions contributions to power semiconductor device modeling and packaging."
    await evaluator.verify(
        claim=claim_citation,
        node=node_citation,
        sources=src,
        additional_instruction="Confirm the citation language includes device modeling and packaging contributions.",
    )


async def build_step_5_contact(
    evaluator: Evaluator,
    parent_node,
    center: CenterExtraction,
    pi: PIExtraction,
):
    """Step 5: Founding director contact info from center website (parallel)."""
    step_node = evaluator.add_parallel(
        id="Step_5_Center_Contact_Info",
        desc="Provide founding director contact info from the center website and ensure it is publicly listed.",
        parent=parent_node,
        critical=False,
    )

    phone = (center.founding_director_phone or "").strip()
    email = (center.founding_director_email or "").strip()
    pi_name = (pi.pi_name or "").strip()
    src = _combine_sources(center.center_contact_sources, center.center_sources)

    # Public phone for founding director (verify by URL)
    node_phone = evaluator.add_leaf(
        id="Provide_Public_Phone_For_Founding_Director",
        desc="Provide the publicly listed phone number for the founding director on the research center's website.",
        parent=step_node,
        critical=True,
    )
    claim_phone = (
        f"The founding director ({pi_name}) has a publicly listed phone number '{phone}' on the center's official website."
        if phone and pi_name
        else f"The founding director has a publicly listed phone number '{phone}' on the center's official website."
        if phone
        else "The founding director's publicly listed phone number appears on the center's official website."
    )
    await evaluator.verify(
        claim=claim_phone,
        node=node_phone,
        sources=src,
        additional_instruction="Verify the phone number exists on the center's official site and is associated with the founding director; allow formatting variations.",
    )

    # Public email for founding director (verify by URL)
    node_email = evaluator.add_leaf(
        id="Provide_Public_Email_For_Founding_Director",
        desc="Provide the publicly listed email address for the founding director on the research center's website.",
        parent=step_node,
        critical=True,
    )
    claim_email = (
        f"The founding director ({pi_name}) has a publicly listed email address '{email}' on the center's official website."
        if email and pi_name
        else f"The founding director has a publicly listed email address '{email}' on the center's official website."
        if email
        else "The founding director's publicly listed email address appears on the center's official website."
    )
    await evaluator.verify(
        claim=claim_email,
        node=node_email,
        sources=src,
        additional_instruction="Verify the email address exists on the center's official site (including mailto links) and is associated with the founding director; allow case variations.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the UArk SiC fabrication facility and related PI/center/award/contact info.
    """
    # Initialize evaluator (root is the overall Multi-Step task; make it sequential)
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

    # IMPORTANT: Root critical must be False to allow non-critical children per framework constraints
    root.critical = False

    # Extract all required structured info (in parallel)
    facility_task = evaluator.extract(
        prompt=prompt_extract_facility(),
        template_class=FacilityExtraction,
        extraction_name="facility_extraction",
    )
    pi_task = evaluator.extract(
        prompt=prompt_extract_pi(),
        template_class=PIExtraction,
        extraction_name="pi_extraction",
    )
    center_task = evaluator.extract(
        prompt=prompt_extract_center(),
        template_class=CenterExtraction,
        extraction_name="center_extraction",
    )
    award_task = evaluator.extract(
        prompt=prompt_extract_award(),
        template_class=AwardExtraction,
        extraction_name="award_extraction",
    )

    facility, pi, center, award = await asyncio.gather(facility_task, pi_task, center_task, award_task)

    # Record a concise summary of extracted key fields
    evaluator.add_custom_info(
        info={
            "facility_name": facility.facility_name,
            "pi_name": pi.pi_name,
            "center_acronym": center.center_acronym,
            "center_name": center.center_name,
            "award_full_name": award.award_full_name,
            "founding_director_phone": center.founding_director_phone,
            "founding_director_email": center.founding_director_email,
        },
        info_type="extracted_summary",
        info_name="extracted_key_fields",
    )

    # Build and run verification steps following rubric tree
    await build_step_1_facility(evaluator, root, facility)
    await build_step_2_pi(evaluator, root, facility, pi)
    await build_step_3_center(evaluator, root, center, pi)
    await build_step_4_award(evaluator, root, award, pi)
    await build_step_5_contact(evaluator, root, center, pi)

    # Return the evaluator summary, including verification tree and final score
    return evaluator.get_summary()