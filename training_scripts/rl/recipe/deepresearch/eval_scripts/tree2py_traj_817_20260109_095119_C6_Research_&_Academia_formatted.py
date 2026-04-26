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
TASK_ID = "ca_bio_postdoc_center"
TASK_DESCRIPTION = (
    "A computational biologist who recently completed their PhD is seeking a postdoctoral research opportunity in California that will provide access to both wet lab and computational research infrastructure. "
    "They need to identify a research institute or center that meets all of the following requirements:\n\n"
    "1. Location and Affiliation: The research institute must be located in California and affiliated with a university\n"
    "2. Federal Funding: The institute must be supported by active federal funding from the National Science Foundation (NSF) or National Institutes of Health (NIH) as a research center or institute (not just individual investigator grants)\n"
    "3. Research Focus: The institute must conduct active research in computational biology, bioinformatics, or closely related computational life sciences fields\n"
    "4. Biosafety Laboratory Facilities: The institute or its affiliated university must have on-site Biosafety Level 2 (BSL-2) certified laboratory facilities that are accessible to postdoctoral researchers\n"
    "5. High-Performance Computing: The institute or affiliated university must provide access to high-performance computing (HPC) infrastructure suitable for computational biology research\n"
    "6. Postdoctoral Opportunities: The institute should offer or support postdoctoral research positions or fellowships\n\n"
    "Identify ONE specific research institute or center in California that satisfies all of these requirements. Provide the following information:\n\n"
    "- The name of the research institute/center\n"
    "- The affiliated university\n"
    "- Evidence of federal funding (NSF or NIH) supporting the center\n"
    "- Confirmation of computational biology research focus\n"
    "- Evidence of BSL-2 laboratory facilities\n"
    "- Evidence of HPC infrastructure availability\n"
    "- Information about postdoctoral opportunities (if available)\n\n"
    "For each requirement, provide the specific URL reference from the institute's or university's official website that documents compliance with that requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class InstituteSelection(BaseModel):
    """Single institute/center selection with official URLs for each requirement."""
    institute_name: Optional[str] = None
    affiliated_university: Optional[str] = None
    homepage_url: Optional[str] = None

    # In case the answer mentions more than one entity; we enforce exactly one via existence check
    additional_entities: List[str] = Field(default_factory=list)

    # Location & affiliation evidence URLs
    location_url: Optional[str] = None
    affiliation_url: Optional[str] = None
    active_operation_url: Optional[str] = None

    # Federal funding evidence URLs
    funding_funder: Optional[str] = None  # Expected values: "NSF", "NIH"; but extractor returns text as-is
    funding_url: Optional[str] = None
    funding_center_level_url: Optional[str] = None
    funding_active_url: Optional[str] = None

    # Computational biology focus & active program URLs
    research_focus_url: Optional[str] = None
    active_program_url: Optional[str] = None

    # BSL-2 requirements URLs
    bsl2_facility_url: Optional[str] = None
    bsl2_postdoc_access_url: Optional[str] = None
    bsl2_features_url: Optional[str] = None
    bsl2_training_url: Optional[str] = None
    bsl2_inspection_url: Optional[str] = None

    # HPC requirements URLs
    hpc_available_url: Optional[str] = None
    hpc_suitable_url: Optional[str] = None
    hpc_postdoc_access_url: Optional[str] = None

    # Optional postdoctoral opportunities URL
    postdoc_opportunities_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_institute_selection() -> str:
    return (
        "Extract exactly ONE recommended research institute or center in California as presented in the answer and the official URLs that support each requirement. "
        "If the answer lists multiple entities, pick the primary recommended one and list other entities in 'additional_entities'. "
        "Return a JSON object with the following fields:\n\n"
        "1) institute_name: The exact official name of the institute/center.\n"
        "2) affiliated_university: The exact name of the university affiliation.\n"
        "3) homepage_url: The official homepage (or 'About' page) URL of the institute/center.\n"
        "4) additional_entities: Array of other institutes/centers mentioned (if any).\n\n"
        "Location & Affiliation URLs:\n"
        "5) location_url: Official page URL confirming the institute is in California (can be address/contact page).\n"
        "6) affiliation_url: Official page URL confirming formal affiliation with the university.\n"
        "7) active_operation_url: Official page URL evidencing active operations in 2025–2026 (e.g., current events, updated news, recent publications).\n\n"
        "Federal Funding URLs:\n"
        "8) funding_funder: The text stating NSF or NIH (e.g., 'NSF' or 'NIH').\n"
        "9) funding_url: Official page URL indicating NSF/NIH support.\n"
        "10) funding_center_level_url: Official page URL indicating center-level or institute-level support (not merely individual PI grants).\n"
        "11) funding_active_url: Official page URL indicating the funding is active in 2025–2026.\n\n"
        "Computational Biology Focus URLs:\n"
        "12) research_focus_url: Official page URL explicitly mentioning computational biology/bioinformatics as a research area.\n"
        "13) active_program_url: Official page URL showing active research program (projects, publications, active investigators).\n\n"
        "BSL-2 URLs (institute or affiliated university):\n"
        "14) bsl2_facility_url: Official page URL confirming on-site BSL-2 certified labs exist.\n"
        "15) bsl2_postdoc_access_url: Official page URL confirming postdocs can access the BSL-2 labs.\n"
        "16) bsl2_features_url: Official page URL listing required BSL-2 biosafety features (biohazard signage, hand-washing sinks, eye-wash stations).\n"
        "17) bsl2_training_url: Official page URL confirming mandatory biosafety/BSL-2 training.\n"
        "18) bsl2_inspection_url: Official page URL confirming annual safety inspections for BSL-2 labs.\n\n"
        "HPC URLs (institute or affiliated university):\n"
        "19) hpc_available_url: Official page URL confirming HPC cluster/resources availability.\n"
        "20) hpc_suitable_url: Official page URL supporting suitability for computational biology/bioinformatics (e.g., life sciences software modules, GPU/large memory nodes, bioinformatics examples).\n"
        "21) hpc_postdoc_access_url: Official page URL confirming postdoctoral eligibility/access to HPC resources.\n\n"
        "Postdoctoral Opportunities (optional):\n"
        "22) postdoc_opportunities_url: Official page URL evidencing postdoc positions or fellowships (if available).\n\n"
        "Rules:\n"
        "- Extract only URLs explicitly provided in the answer; if a required URL is not present, return null.\n"
        "- Use full URLs; handle markdown links by extracting the actual URL.\n"
        "- Do not invent or infer any URLs or details.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_value(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _gate_url(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    url: Optional[str],
    critical: bool = True
):
    """Add a critical existence gate requiring a URL to be provided."""
    return evaluator.add_custom_node(
        result=_has_value(url),
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_single_institute_section(evaluator: Evaluator, root, info: InstituteSelection):
    """Single institute identification and identity checks."""
    parent = evaluator.add_parallel(
        id="Single_Institute_Identified",
        desc="Exactly one specific institute/center is identified; name and affiliated university are clearly stated.",
        parent=root,
        critical=True
    )

    # Leaf: Institute name provided and exactly one entity selected
    name_node = evaluator.add_custom_node(
        result=_has_value(info.institute_name) and len(info.additional_entities) == 0,
        id="Institute_or_Center_Name_Provided",
        desc="Provides the name of exactly one specific research institute/center.",
        parent=parent,
        critical=True
    )

    # Leaf: Affiliated university provided
    uni_node = evaluator.add_custom_node(
        result=_has_value(info.affiliated_university),
        id="Affiliated_University_Provided",
        desc="Provides the name of the affiliated university.",
        parent=parent,
        critical=True
    )

    # Gate for homepage URL
    _gate_url(
        evaluator,
        parent,
        "Entity_Homepage_URL_Provided",
        "Official homepage URL is provided for the entity.",
        info.homepage_url,
        critical=True
    )

    # Leaf: Entity is institute/center (not department)
    leaf = evaluator.add_leaf(
        id="Entity_Is_Institute_or_Center_Not_Department",
        desc="Official webpage evidence shows the entity is a research institute or research center (not merely an academic department).",
        parent=parent,
        critical=True
    )

    claim = (
        f"The official webpage indicates that '{info.institute_name}' is a research institute or research center, "
        f"not just an academic department."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.homepage_url,
        additional_instruction="Check the page title and descriptors for terms like 'Institute', 'Center', or 'Research Center'; "
                              "if it says 'Department', that does not satisfy this requirement."
    )


async def build_location_affiliation_section(evaluator: Evaluator, root, info: InstituteSelection):
    """California location and formal university affiliation requirements."""
    parent = evaluator.add_parallel(
        id="Location_and_Affiliation",
        desc="Meets the California location and university affiliation requirements with official URL evidence.",
        parent=root,
        critical=True
    )

    # Located in California
    _gate_url(
        evaluator,
        parent,
        "Located_in_California_URL_Provided",
        "URL evidence for California location is provided.",
        info.location_url,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id="Located_in_California_With_URL",
        desc="Official webpage evidence confirms the institute/center is located in California.",
        parent=parent,
        critical=True
    )
    claim = f"The institute/center '{info.institute_name}' is located in California, USA."
    await evaluator.verify(
        claim=claim,
        node=loc_leaf,
        sources=info.location_url,
        additional_instruction="Look for a California address, campus location in California, or explicit mention of 'California' or 'CA'."
    )

    # University affiliation
    _gate_url(
        evaluator,
        parent,
        "University_Affiliation_URL_Provided",
        "URL evidence for formal university affiliation is provided.",
        info.affiliation_url,
        critical=True
    )
    aff_leaf = evaluator.add_leaf(
        id="University_Affiliation_With_URL",
        desc="Official webpage evidence confirms the institute/center is formally affiliated with or part of a university.",
        parent=parent,
        critical=True
    )
    claim = (
        f"The institute/center '{info.institute_name}' is formally affiliated with or part of '{info.affiliated_university}'."
    )
    await evaluator.verify(
        claim=claim,
        node=aff_leaf,
        sources=info.affiliation_url,
        additional_instruction="The page should clearly indicate the institute is part of, affiliated with, or hosted by the specified university."
    )

    # Active operations as of 2026
    _gate_url(
        evaluator,
        parent,
        "Actively_Operating_As_of_2026_URL_Provided",
        "URL evidence that the institute/center is actively operating (2025–2026) is provided.",
        info.active_operation_url,
        critical=True
    )
    active_leaf = evaluator.add_leaf(
        id="Actively_Operating_As_of_2026_With_URL",
        desc="Official webpage evidence supports that the institute/center is actively operating as of 2026.",
        parent=parent,
        critical=True
    )
    claim = (
        f"The institute/center '{info.institute_name}' shows evidence of active operations in 2025–2026 "
        f"(e.g., current events, updated news, recent publications)."
    )
    await evaluator.verify(
        claim=claim,
        node=active_leaf,
        sources=info.active_operation_url,
        additional_instruction="Look for recent timestamps (2025 or 2026), upcoming events, updated news posts, or current projects/people pages."
    )


async def build_federal_funding_section(evaluator: Evaluator, root, info: InstituteSelection):
    """NSF/NIH federal funding at center level, currently active."""
    parent = evaluator.add_parallel(
        id="Federal_Funding",
        desc="Meets the NSF/NIH active federal funding requirement for the center/institute itself with official URL evidence.",
        parent=root,
        critical=True
    )

    # Funder evidence
    _gate_url(
        evaluator,
        parent,
        "Funding_Source_URL_Provided",
        "URL evidence indicating NSF/NIH support is provided.",
        info.funding_url,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="Funding_Source_NSF_or_NIH_With_URL",
        desc="Official webpage evidence indicates the center/institute is supported by NSF or NIH.",
        parent=parent,
        critical=True
    )
    funder_text = info.funding_funder or "NSF or NIH"
    claim = (
        f"The institute/center '{info.institute_name}' is supported by {funder_text} (federal support)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf1,
        sources=info.funding_url,
        additional_instruction="The page should clearly reference NSF or NIH as funders supporting the center/institute."
    )

    # Center-level (not just individual investigator grants)
    _gate_url(
        evaluator,
        parent,
        "Funding_Center_Level_URL_Provided",
        "URL evidence indicating center-level/institute-level support (not only PI grants) is provided.",
        info.funding_center_level_url,
        critical=True
    )
    leaf2 = evaluator.add_leaf(
        id="Funding_Is_Center_Level_Not_Just_Individual_Grants_With_URL",
        desc="Official webpage evidence indicates the funding supports the center/institute itself (not only individual investigator awards).",
        parent=parent,
        critical=True
    )
    claim = (
        f"The federal funding referenced supports '{info.institute_name}' at the center/institute level, "
        f"rather than only individual PI grants."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf2,
        sources=info.funding_center_level_url,
        additional_instruction="Look for terms like 'center grant', 'institute grant', P30/P50/P01 cores, or NSF center awards. "
                              "Text that only lists individual PI awards does not satisfy center-level funding."
    )

    # Active in 2025–2026
    _gate_url(
        evaluator,
        parent,
        "Funding_Active_URL_Provided",
        "URL evidence indicating the relevant federal funding is active (2025–2026) is provided.",
        info.funding_active_url,
        critical=True
    )
    leaf3 = evaluator.add_leaf(
        id="Funding_Currently_Active_2025_2026_With_URL",
        desc="Official webpage evidence indicates the relevant federal funding is currently active as of 2025–2026.",
        parent=parent,
        critical=True
    )
    claim = (
        f"The federal funding for '{info.institute_name}' is active during 2025–2026."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf3,
        sources=info.funding_active_url,
        additional_instruction="Look for active grant periods including 2025 or 2026, renewal notices, or statements that funding is current."
    )


async def build_research_focus_section(evaluator: Evaluator, root, info: InstituteSelection):
    """Computational biology/bioinformatics focus and active program evidence."""
    parent = evaluator.add_parallel(
        id="Computational_Biology_Research_Focus",
        desc="Meets the computational biology/bioinformatics research focus and active program requirements with official URL evidence.",
        parent=root,
        critical=True
    )

    # Explicit computational biology/bioinformatics mention
    _gate_url(
        evaluator,
        parent,
        "Computational_Focus_URL_Provided",
        "URL evidence explicitly mentioning computational biology/bioinformatics is provided.",
        info.research_focus_url,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="Explicit_Computational_Biology_or_Bioinformatics_Mention_With_URL",
        desc="Official webpage evidence explicitly mentions computational biology and/or bioinformatics as a research area for the institute/center.",
        parent=parent,
        critical=True
    )
    claim = (
        f"The institute/center '{info.institute_name}' explicitly lists computational biology or bioinformatics among its research areas."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf1,
        sources=info.research_focus_url,
        additional_instruction="Look for explicit text such as 'computational biology', 'bioinformatics', 'computational genomics', "
                              "'systems biology', or similar phrasing."
    )

    # Active research program evidence
    _gate_url(
        evaluator,
        parent,
        "Active_Program_URL_Provided",
        "URL evidence showing an ongoing research program is provided.",
        info.active_program_url,
        critical=True
    )
    leaf2 = evaluator.add_leaf(
        id="Active_Research_Program_Evidence_With_URL",
        desc="Official webpage evidence shows an ongoing research program (e.g., publications, projects, or active investigators).",
        parent=parent,
        critical=True
    )
    claim = (
        f"The institute/center '{info.institute_name}' shows evidence of active research programs (e.g., current publications, projects, or active investigators)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf2,
        sources=info.active_program_url,
        additional_instruction="Look for lists of recent publications, active labs/projects, or up-to-date investigator/staff pages."
    )


async def build_bsl2_section(evaluator: Evaluator, root, info: InstituteSelection):
    """BSL-2 facility existence, access, features, training, inspections."""
    parent = evaluator.add_parallel(
        id="BSL2_Laboratory_Requirements",
        desc="Meets BSL-2 facility existence and access requirements (including required training and inspections) with official URL evidence.",
        parent=root,
        critical=True
    )

    # On-site BSL-2 labs
    _gate_url(
        evaluator,
        parent,
        "BSL2_Facility_URL_Provided",
        "URL evidence confirming on-site BSL-2 certified laboratory facilities exists.",
        info.bsl2_facility_url,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="Onsite_BSL2_Certified_Labs_With_URL",
        desc="Official webpage evidence confirms on-site BSL-2 certified laboratory facilities exist at the institute or affiliated university.",
        parent=parent,
        critical=True
    )
    claim = (
        "On-site BSL-2 certified laboratory facilities exist at the institute or its affiliated university."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf1,
        sources=info.bsl2_facility_url,
        additional_instruction="Look for explicit mention of 'BSL-2' facilities or lab classification."
    )

    # Postdoc access to BSL-2
    _gate_url(
        evaluator,
        parent,
        "BSL2_Postdoc_Access_URL_Provided",
        "URL evidence confirming postdoctoral researcher access to BSL-2 labs is provided.",
        info.bsl2_postdoc_access_url,
        critical=True
    )
    leaf2 = evaluator.add_leaf(
        id="Postdoc_Access_To_BSL2_With_URL",
        desc="Official webpage evidence confirms postdoctoral researchers have access to the BSL-2 facilities.",
        parent=parent,
        critical=True
    )
    claim = (
        "Postdoctoral researchers have access to use the BSL-2 laboratory facilities."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf2,
        sources=info.bsl2_postdoc_access_url,
        additional_instruction="Eligibility pages or lab access policy should include postdocs or 'all lab personnel including postdoctoral scholars'."
    )

    # Required BSL-2 biosafety features
    _gate_url(
        evaluator,
        parent,
        "BSL2_Features_URL_Provided",
        "URL evidence listing required BSL-2 biosafety features is provided.",
        info.bsl2_features_url,
        critical=True
    )
    leaf3 = evaluator.add_leaf(
        id="BSL2_Required_Biosafety_Features_With_URL",
        desc="Official webpage evidence supports that BSL-2 facilities include required biosafety features (biohazard signage, hand-washing sinks, and eye-washing stations).",
        parent=parent,
        critical=True
    )
    claim = (
        "BSL-2 facilities include required biosafety features: biohazard signage, hand-washing sinks, and eye-wash stations."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf3,
        sources=info.bsl2_features_url,
        additional_instruction="Verify the listed features include biohazard signage, hand-washing sinks, and eye-wash stations; synonymous terms acceptable."
    )

    # Mandatory biosafety training
    _gate_url(
        evaluator,
        parent,
        "BSL2_Training_URL_Provided",
        "URL evidence confirming mandatory biosafety training is required is provided.",
        info.bsl2_training_url,
        critical=True
    )
    leaf4 = evaluator.add_leaf(
        id="Mandatory_BSL2_Biosafety_Training_With_URL",
        desc="Official webpage evidence confirms mandatory biosafety training is required for BSL-2 laboratory personnel.",
        parent=parent,
        critical=True
    )
    claim = (
        "Mandatory biosafety training (including BSL-2 training) is required for laboratory personnel."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf4,
        sources=info.bsl2_training_url,
        additional_instruction="Look for language indicating required safety training or mandatory biosafety training for BSL-2 lab work."
    )

    # Annual safety inspections
    _gate_url(
        evaluator,
        parent,
        "BSL2_Inspection_URL_Provided",
        "URL evidence confirming annual safety inspections for BSL-2 labs is provided.",
        info.bsl2_inspection_url,
        critical=True
    )
    leaf5 = evaluator.add_leaf(
        id="Annual_BSL2_Safety_Inspections_With_URL",
        desc="Official webpage evidence confirms BSL-2 laboratories undergo annual safety inspections.",
        parent=parent,
        critical=True
    )
    claim = (
        "BSL-2 laboratories undergo annual safety inspections."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf5,
        sources=info.bsl2_inspection_url,
        additional_instruction="EHS/Environmental Health & Safety pages often outline lab inspection cadence; confirm annual inspection for BSL-2."
    )


async def build_hpc_section(evaluator: Evaluator, root, info: InstituteSelection):
    """HPC existence, suitability for computational biology, and postdoc access."""
    parent = evaluator.add_parallel(
        id="HPC_Requirements",
        desc="Meets HPC existence, suitability, and postdoc access requirements with official URL evidence.",
        parent=root,
        critical=True
    )

    # HPC available
    _gate_url(
        evaluator,
        parent,
        "HPC_Available_URL_Provided",
        "URL evidence confirming HPC cluster/resources availability is provided.",
        info.hpc_available_url,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="HPC_Available_With_URL",
        desc="Official webpage evidence confirms the institute/university operates or provides access to HPC cluster/resources.",
        parent=parent,
        critical=True
    )
    claim = (
        "High-performance computing (HPC) cluster/resources are available at the institute or affiliated university."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf1,
        sources=info.hpc_available_url,
        additional_instruction="Look for research computing services, clusters, or supercomputing resources accessible to campus researchers."
    )

    # HPC suitable for computational biology/bioinformatics
    _gate_url(
        evaluator,
        parent,
        "HPC_Suitable_URL_Provided",
        "URL evidence supporting suitability for computational biology/bioinformatics is provided.",
        info.hpc_suitable_url,
        critical=True
    )
    leaf2 = evaluator.add_leaf(
        id="HPC_Suitable_For_Computational_Biology_With_URL",
        desc="Official webpage evidence supports that the computing resources are suitable for computational biology/bioinformatics research applications.",
        parent=parent,
        critical=True
    )
    claim = (
        "The HPC resources are suitable for computational biology/bioinformatics research (e.g., life sciences modules, bioinformatics software, GPU/large memory nodes)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf2,
        sources=info.hpc_suitable_url,
        additional_instruction="Look for mentions of life sciences software (e.g., BLAST, Bowtie, BWA, GATK), bioinformatics modules, or capabilities like high-memory/GPU nodes used in computational biology."
    )

    # Postdoc access to HPC
    _gate_url(
        evaluator,
        parent,
        "HPC_Postdoc_Access_URL_Provided",
        "URL evidence confirming postdoc eligibility/access to HPC is provided.",
        info.hpc_postdoc_access_url,
        critical=True
    )
    leaf3 = evaluator.add_leaf(
        id="Postdoc_Access_To_HPC_With_URL",
        desc="Official webpage evidence confirms postdoctoral researchers have access/eligibility to use the HPC resources.",
        parent=parent,
        critical=True
    )
    claim = (
        "Postdoctoral researchers are eligible and have access to use the HPC resources."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf3,
        sources=info.hpc_postdoc_access_url,
        additional_instruction="Eligibility or access policy should explicitly include postdocs, or state 'faculty/staff/students/postdoctoral scholars'."
    )


async def build_postdoc_opportunities_section(evaluator: Evaluator, root, info: InstituteSelection):
    """Optional postdoctoral opportunities evidence (non-critical)."""
    parent = evaluator.add_parallel(
        id="Postdoctoral_Opportunities",
        desc="The institute should offer or support postdoctoral positions/fellowships, with official URL evidence if available.",
        parent=root,
        critical=False
    )

    # Gate provided (non-critical)
    _gate_url(
        evaluator,
        parent,
        "Postdoc_Opportunities_URL_Provided",
        "URL evidence for postdoc positions/fellowships is provided (if available).",
        info.postdoc_opportunities_url,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="Postdoc_Opportunities_Evidence_With_URL",
        desc="Provides official webpage evidence that the institute/university offers or supports postdoctoral research positions or fellowships (if available).",
        parent=parent,
        critical=False
    )
    claim = (
        "The institute or affiliated university offers or supports postdoctoral research positions or fellowships."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.postdoc_opportunities_url,
        additional_instruction="Look for 'Postdoctoral Scholar', 'Postdoc Fellowship', 'Postdoctoral Programs', or job postings explicitly for postdocs."
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
    Evaluate an answer for the California institute/center selection task.
    Builds a verification tree aligned with the rubric and returns the summary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates independent requirement groups
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

    # Extract the selected institute/center and all requirement URLs from the answer
    info: InstituteSelection = await evaluator.extract(
        prompt=prompt_extract_institute_selection(),
        template_class=InstituteSelection,
        extraction_name="institute_selection"
    )

    # Build verification sections according to rubric
    await build_single_institute_section(evaluator, root, info)
    await build_location_affiliation_section(evaluator, root, info)
    await build_federal_funding_section(evaluator, root, info)
    await build_research_focus_section(evaluator, root, info)
    await build_bsl2_section(evaluator, root, info)
    await build_hpc_section(evaluator, root, info)
    await build_postdoc_opportunities_section(evaluator, root, info)

    # Return structured summary including verification tree and scores
    return evaluator.get_summary()