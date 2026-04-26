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
TASK_ID = "r1_interdisciplinary_comp_biomed_programs"
TASK_DESCRIPTION = """
Identify four R1 universities in the United States (classified as 'Doctoral Universities – Very High Research Activity' according to the Carnegie Classification of Institutions of Higher Education) that meet all of the following comprehensive requirements for establishing a new interdisciplinary computational biomedical research program:

Graduate Program Requirements:
- Must offer PhD programs in both data science (or closely related computational field) and biomedical engineering (or closely related field)
- Must provide graduate research assistantships with a minimum annual stipend of at least $18,000 per academic year

Research Infrastructure Requirements:
- Must have an active Institutional Review Board (IRB)
- Must have a research compliance office
- Must have a technology transfer office

Postdoctoral Program Requirements:
- Must offer structured postdoctoral research positions in relevant fields
- Must provide career development resources and support for postdoctoral scholars

Funding Support Requirements:
- Must provide support services for NSF grant applications and administration
- Must provide guidance and resources for preparing data management plans

Faculty Support Requirements:
- Must provide faculty startup packages that include equipment, supplies, and personnel support
- Must support conference presentations in both oral and poster formats

For each of the four universities you identify, provide:
1. The university name and confirmation of its R1 classification status
2. Evidence of PhD programs in both required fields (data science/computational science and biomedical engineering)
3. Documentation of graduate assistantship stipend amounts meeting the minimum requirement
4. Verification of all three required research infrastructure offices (IRB, research compliance, technology transfer)
5. Evidence of postdoctoral programs and career development support
6. Documentation of NSF grant support and data management plan resources
7. Evidence of faculty startup packages and conference support

Include specific reference URLs from each university's official website to verify each requirement.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityEvidence(BaseModel):
    """Evidence bundle for a single university."""
    name: Optional[str] = None

    # R1 status evidence
    r1_urls: List[str] = Field(default_factory=list)

    # Graduate programs
    ds_program_name: Optional[str] = None
    ds_program_urls: List[str] = Field(default_factory=list)
    bme_program_name: Optional[str] = None
    bme_program_urls: List[str] = Field(default_factory=list)

    # Stipend evidence
    stipend_min: Optional[str] = None  # Keep as string to allow ranges (e.g., '$20k-$25k') or per-month text
    stipend_urls: List[str] = Field(default_factory=list)

    # Research infrastructure
    irb_urls: List[str] = Field(default_factory=list)
    compliance_urls: List[str] = Field(default_factory=list)
    tto_urls: List[str] = Field(default_factory=list)

    # Postdoctoral programs
    postdoc_program_urls: List[str] = Field(default_factory=list)
    postdoc_career_urls: List[str] = Field(default_factory=list)

    # Funding support
    nsf_support_urls: List[str] = Field(default_factory=list)
    dmp_support_urls: List[str] = Field(default_factory=list)

    # Faculty support
    startup_urls: List[str] = Field(default_factory=list)
    conference_support_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    """Top-level list of universities extracted from the answer."""
    universities: List[UniversityEvidence] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four distinct U.S. universities listed in the answer that the answer claims meet ALL the specified requirements for establishing a new interdisciplinary computational biomedical research program.

    For each university, return an object containing EXACT fields below, using only information explicitly present in the answer. Extract URLs as they appear (plain URLs or markdown links). If a field is missing, set it to null for single fields or an empty list for lists.

    Fields per university:
    - name: University name.
    - r1_urls: Array of URLs confirming the university’s R1 classification (prefer the Carnegie Classification site or official university documentation).
    - ds_program_name: Name of the PhD program in data science or a closely related computational field (e.g., Computer Science, Computational Science, Statistics, Informatics).
    - ds_program_urls: Array of official program URLs that specifically describe the PhD program in the data/computational field.
    - bme_program_name: Name of the PhD program in biomedical engineering or a closely related field (e.g., Bioengineering, Biomedical Informatics if it is explicitly a PhD).
    - bme_program_urls: Array of official program URLs for the PhD in biomedical engineering or closely related field.
    - stipend_min: The stated minimum annual stipend amount for graduate research assistantships (string as presented).
    - stipend_urls: Array of official URLs documenting graduate assistantship stipend amounts or policies.
    - irb_urls: Array of official URLs to the Institutional Review Board (IRB) office/website or human subjects research office.
    - compliance_urls: Array of official URLs to the research compliance office or compliance services website.
    - tto_urls: Array of official URLs to the technology transfer office (may be called intellectual property, innovation, tech licensing).
    - postdoc_program_urls: Array of official URLs describing structured postdoctoral positions/programs.
    - postdoc_career_urls: Array of official URLs describing postdoctoral career development/professional development resources.
    - nsf_support_urls: Array of official URLs showing university services supporting NSF grant applications and administration (e.g., Office of Sponsored Programs, Research Development).
    - dmp_support_urls: Array of official URLs providing guidance/resources for data management plans (e.g., library research data services, DMPTool pages).
    - startup_urls: Array of official URLs describing faculty startup packages, policies, or typical components (equipment, supplies, personnel).
    - conference_support_urls: Array of official URLs describing support for conference presentations (including oral and poster formats), such as travel funds or presentation policies.

    Return a JSON object with a top-level 'universities' array containing up to 4 such objects. Do not invent URLs or names that are not in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal_name(index: int) -> str:
    mapping = {0: "First", 1: "Second", 2: "Third", 3: "Fourth"}
    return mapping.get(index, f"University #{index + 1}")


def has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls or [])


def safe_name(name: Optional[str]) -> str:
    return name.strip() if isinstance(name, str) else "the university"


# --------------------------------------------------------------------------- #
# Verification builder for one university                                     #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityEvidence,
    idx: int,
) -> None:
    """
    Build verification sub-tree and run checks for a single university.
    """

    # University-level container (non-critical to allow partial credit per university)
    u_node = evaluator.add_parallel(
        id=f"University_{idx + 1}",
        desc=f"{ordinal_name(idx)} identified R1 university meets all program requirements",
        parent=parent_node,
        critical=False
    )

    # ----------------------------- Carnegie R1 ----------------------------- #
    carnegie_node = evaluator.add_parallel(
        id=f"Carnegie_Classification_U{idx + 1}",
        desc=f"{ordinal_name(idx)} university is classified as R1 (Doctoral Universities – Very High Research Activity)",
        parent=u_node,
        critical=True
    )

    # URL presence check
    evaluator.add_custom_node(
        result=has_urls(uni.r1_urls),
        id=f"R1_Reference_URL_U{idx + 1}",
        desc="Provided valid reference URL confirming R1 status",
        parent=carnegie_node,
        critical=True
    )

    # R1 status supported by sources
    r1_leaf = evaluator.add_leaf(
        id=f"R1_Status_Verification_U{idx + 1}",
        desc="Verified R1 classification status through official sources",
        parent=carnegie_node,
        critical=True
    )
    r1_claim = f"{safe_name(uni.name)} is classified as R1 (Doctoral Universities – Very High Research Activity) under the Carnegie Classification."
    await evaluator.verify(
        claim=r1_claim,
        node=r1_leaf,
        sources=uni.r1_urls,
        additional_instruction="Prefer evidence from the Carnegie Classification website or official university pages that explicitly state R1 status."
    )

    # ------------------------- Graduate Programs -------------------------- #
    grad_node = evaluator.add_parallel(
        id=f"Graduate_Programs_U{idx + 1}",
        desc=f"{ordinal_name(idx)} university offers the required doctoral programs and stipend",
        parent=u_node,
        critical=True
    )

    # Data Science / Computational PhD
    ds_node = evaluator.add_parallel(
        id=f"Data_Science_PhD_U{idx + 1}",
        desc="Offers PhD program in data science or closely related computational field",
        parent=grad_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(uni.ds_program_urls),
        id=f"Data_Science_Program_URL_U{idx + 1}",
        desc="Provided valid URL to data/computational PhD program website",
        parent=ds_node,
        critical=True
    )
    ds_leaf = evaluator.add_leaf(
        id=f"Data_Science_Program_Exists_U{idx + 1}",
        desc="Verified existence of data/computational science PhD program",
        parent=ds_node,
        critical=True
    )
    ds_claim = f"{safe_name(uni.name)} offers an official Ph.D. program in a data science or closely related computational field (e.g., Computer Science, Computational Science, Statistics, Informatics)."
    await evaluator.verify(
        claim=ds_claim,
        node=ds_leaf,
        sources=uni.ds_program_urls,
        additional_instruction="Confirm the page explicitly describes a Ph.D./Doctoral program (not just MS) in a data/computational discipline. Tracks or concentrations are acceptable if within a Ph.D."
    )

    # Biomedical Engineering PhD
    bme_node = evaluator.add_parallel(
        id=f"Biomedical_Engineering_PhD_U{idx + 1}",
        desc="Offers PhD program in biomedical engineering or closely related field",
        parent=grad_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(uni.bme_program_urls),
        id=f"Biomed_Program_URL_U{idx + 1}",
        desc="Provided valid URL to biomedical engineering PhD program website",
        parent=bme_node,
        critical=True
    )
    bme_leaf = evaluator.add_leaf(
        id=f"Biomed_Program_Exists_U{idx + 1}",
        desc="Verified existence of biomedical engineering PhD program",
        parent=bme_node,
        critical=True
    )
    bme_claim = f"{safe_name(uni.name)} offers an official Ph.D. program in biomedical engineering or a closely related field (e.g., Bioengineering)."
    await evaluator.verify(
        claim=bme_claim,
        node=bme_leaf,
        sources=uni.bme_program_urls,
        additional_instruction="Confirm the page explicitly describes a Ph.D./Doctoral program (not just MS) in Biomedical Engineering or an equivalent field."
    )

    # Stipend minimum
    stipend_node = evaluator.add_parallel(
        id=f"Graduate_Assistantship_Stipend_U{idx + 1}",
        desc="Graduate research assistantships provide minimum stipend of at least $18,000 per academic year",
        parent=grad_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(uni.stipend_urls),
        id=f"Stipend_Reference_URL_U{idx + 1}",
        desc="Provided valid URL documenting stipend amounts",
        parent=stipend_node,
        critical=True
    )
    stipend_leaf = evaluator.add_leaf(
        id=f"Stipend_Amount_Verified_U{idx + 1}",
        desc="Verified that stipend meets or exceeds $18,000 minimum",
        parent=stipend_node,
        critical=True
    )
    stipend_claim = f"The minimum annual stipend for graduate research assistantships at {safe_name(uni.name)} is at least $18,000 per academic year."
    await evaluator.verify(
        claim=stipend_claim,
        node=stipend_leaf,
        sources=uni.stipend_urls,
        additional_instruction="Use the provided page(s) to determine the minimum stipend for graduate research assistantships (Ph.D.). If values are monthly/semester, convert to annual; accept ranges if the minimum is >= $18,000."
    )

    # ---------------------- Research Infrastructure ----------------------- #
    infra_node = evaluator.add_parallel(
        id=f"Research_Infrastructure_U{idx + 1}",
        desc="Has required research compliance and support infrastructure",
        parent=u_node,
        critical=True
    )

    # IRB
    irb_node = evaluator.add_parallel(
        id=f"IRB_Office_U{idx + 1}",
        desc="Has active Institutional Review Board (IRB)",
        parent=infra_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(uni.irb_urls),
        id=f"IRB_URL_U{idx + 1}",
        desc="Provided valid URL to IRB office website",
        parent=irb_node,
        critical=True
    )
    irb_leaf = evaluator.add_leaf(
        id=f"IRB_Exists_U{idx + 1}",
        desc="Verified existence of IRB office",
        parent=irb_node,
        critical=True
    )
    irb_claim = f"{safe_name(uni.name)} has an active Institutional Review Board (IRB) office."
    await evaluator.verify(
        claim=irb_claim,
        node=irb_leaf,
        sources=uni.irb_urls,
        additional_instruction="Look for official IRB or Human Subjects Research office pages, policies, and guidance indicating active IRB oversight."
    )

    # Research Compliance Office
    comp_node = evaluator.add_parallel(
        id=f"Research_Compliance_Office_U{idx + 1}",
        desc="Has research compliance office",
        parent=infra_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(uni.compliance_urls),
        id=f"Compliance_Office_URL_U{idx + 1}",
        desc="Provided valid URL to compliance office website",
        parent=comp_node,
        critical=True
    )
    comp_leaf = evaluator.add_leaf(
        id=f"Compliance_Office_Exists_U{idx + 1}",
        desc="Verified existence of research compliance office",
        parent=comp_node,
        critical=True
    )
    comp_claim = f"{safe_name(uni.name)} has an official research compliance office that oversees compliance services."
    await evaluator.verify(
        claim=comp_claim,
        node=comp_leaf,
        sources=uni.compliance_urls,
        additional_instruction="Confirm existence of a research compliance office or similar unit providing compliance services (human subjects, conflicts, export control, etc.)."
    )

    # Technology Transfer Office
    tto_node = evaluator.add_parallel(
        id=f"Technology_Transfer_Office_U{idx + 1}",
        desc="Has technology transfer office",
        parent=infra_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(uni.tto_urls),
        id=f"TTO_URL_U{idx + 1}",
        desc="Provided valid URL to technology transfer office website",
        parent=tto_node,
        critical=True
    )
    tto_leaf = evaluator.add_leaf(
        id=f"TTO_Exists_U{idx + 1}",
        desc="Verified existence of technology transfer office",
        parent=tto_node,
        critical=True
    )
    tto_claim = f"{safe_name(uni.name)} has an official technology transfer office (e.g., tech licensing, innovation, intellectual property)."
    await evaluator.verify(
        claim=tto_claim,
        node=tto_leaf,
        sources=uni.tto_urls,
        additional_instruction="Check official pages for technology transfer, innovation, IP/licensing offices that support commercialization."
    )

    # ---------------------- Postdoctoral Programs ------------------------- #
    postdoc_node = evaluator.add_parallel(
        id=f"Postdoctoral_Program_U{idx + 1}",
        desc="Offers structured postdoctoral programs with career development support",
        parent=u_node,
        critical=True
    )

    # Postdoc positions
    postdoc_pos_node = evaluator.add_parallel(
        id=f"Postdoc_Positions_Available_U{idx + 1}",
        desc="Offers postdoctoral research positions in relevant fields",
        parent=postdoc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(uni.postdoc_program_urls),
        id=f"Postdoc_URL_U{idx + 1}",
        desc="Provided valid URL to postdoctoral program information",
        parent=postdoc_pos_node,
        critical=True
    )
    postdoc_leaf = evaluator.add_leaf(
        id=f"Postdoc_Program_Exists_U{idx + 1}",
        desc="Verified availability of postdoctoral positions",
        parent=postdoc_pos_node,
        critical=True
    )
    postdoc_claim = f"{safe_name(uni.name)} offers structured postdoctoral research positions."
    await evaluator.verify(
        claim=postdoc_claim,
        node=postdoc_leaf,
        sources=uni.postdoc_program_urls,
        additional_instruction="Look for official 'Postdoctoral Affairs', 'Postdoctoral Program', or similar pages indicating structured postdoc positions."
    )

    # Postdoc career development
    postdoc_career_node = evaluator.add_parallel(
        id=f"Postdoc_Career_Development_U{idx + 1}",
        desc="Provides career development resources for postdoctoral scholars",
        parent=postdoc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(uni.postdoc_career_urls),
        id=f"Career_Resources_URL_U{idx + 1}",
        desc="Provided valid URL to career development resources",
        parent=postdoc_career_node,
        critical=True
    )
    postdoc_career_leaf = evaluator.add_leaf(
        id=f"Career_Resources_Exist_U{idx + 1}",
        desc="Verified existence of career development support",
        parent=postdoc_career_node,
        critical=True
    )
    postdoc_career_claim = f"{safe_name(uni.name)} provides career development or professional development resources for postdoctoral scholars."
    await evaluator.verify(
        claim=postdoc_career_claim,
        node=postdoc_career_leaf,
        sources=uni.postdoc_career_urls,
        additional_instruction="Check for resources such as mentoring, workshops, training, career services explicitly for postdocs."
    )

    # -------------------------- Funding Support --------------------------- #
    funding_node = evaluator.add_parallel(
        id=f"Funding_Support_U{idx + 1}",
        desc="Provides comprehensive research funding support including NSF and data management requirements",
        parent=u_node,
        critical=True
    )

    # NSF grants support
    nsf_node = evaluator.add_parallel(
        id=f"NSF_Grant_Support_U{idx + 1}",
        desc="Provides support for NSF grant applications and administration",
        parent=funding_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(uni.nsf_support_urls),
        id=f"NSF_Support_URL_U{idx + 1}",
        desc="Provided valid URL to NSF grant support information",
        parent=nsf_node,
        critical=True
    )
    nsf_leaf = evaluator.add_leaf(
        id=f"NSF_Support_Exists_U{idx + 1}",
        desc="Verified NSF grant application support services",
        parent=nsf_node,
        critical=True
    )
    nsf_claim = f"{safe_name(uni.name)} provides services that support NSF grant applications and administration."
    await evaluator.verify(
        claim=nsf_claim,
        node=nsf_leaf,
        sources=uni.nsf_support_urls,
        additional_instruction="Look for 'Office of Sponsored Programs', 'Research Development', or similar units that explicitly support NSF proposals and awards."
    )

    # Data Management Plan (DMP) support
    dmp_node = evaluator.add_parallel(
        id=f"Data_Management_Plan_Support_U{idx + 1}",
        desc="Provides guidance and resources for preparing data management plans",
        parent=funding_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(uni.dmp_support_urls),
        id=f"DMP_Support_URL_U{idx + 1}",
        desc="Provided valid URL to data management resources",
        parent=dmp_node,
        critical=True
    )
    dmp_leaf = evaluator.add_leaf(
        id=f"DMP_Support_Exists_U{idx + 1}",
        desc="Verified data management plan preparation support",
        parent=dmp_node,
        critical=True
    )
    dmp_claim = f"{safe_name(uni.name)} provides guidance/resources for preparing data management plans."
    await evaluator.verify(
        claim=dmp_claim,
        node=dmp_leaf,
        sources=uni.dmp_support_urls,
        additional_instruction="Check for library research data services, DMP guidance pages, DMPTool portals, or similar official resources."
    )

    # -------------------------- Faculty Support --------------------------- #
    faculty_node = evaluator.add_parallel(
        id=f"Faculty_Support_U{idx + 1}",
        desc="Provides faculty startup packages and conference support",
        parent=u_node,
        critical=True
    )

    # Startup packages
    startup_node = evaluator.add_parallel(
        id=f"Startup_Package_Available_U{idx + 1}",
        desc="Provides faculty startup packages including equipment, supplies, and personnel support",
        parent=faculty_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(uni.startup_urls),
        id=f"Startup_Package_URL_U{idx + 1}",
        desc="Provided valid URL documenting startup package information",
        parent=startup_node,
        critical=True
    )
    startup_leaf = evaluator.add_leaf(
        id=f"Startup_Package_Exists_U{idx + 1}",
        desc="Verified availability of faculty startup packages",
        parent=startup_node,
        critical=True
    )
    startup_claim = f"{safe_name(uni.name)} provides faculty startup packages that include equipment, supplies, and personnel support."
    await evaluator.verify(
        claim=startup_claim,
        node=startup_leaf,
        sources=uni.startup_urls,
        additional_instruction="Look for policies or pages describing new faculty startup packages, typical components (equipment, supplies, personnel), or college-level guidelines."
    )

    # Conference support
    conf_node = evaluator.add_parallel(
        id=f"Conference_Support_U{idx + 1}",
        desc="Supports conference presentations (oral and poster formats)",
        parent=faculty_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(uni.conference_support_urls),
        id=f"Conference_Support_URL_U{idx + 1}",
        desc="Provided valid URL to conference support information",
        parent=conf_node,
        critical=True
    )
    conf_leaf = evaluator.add_leaf(
        id=f"Conference_Support_Exists_U{idx + 1}",
        desc="Verified conference attendance and presentation support",
        parent=conf_node,
        critical=True
    )
    conf_claim = f"{safe_name(uni.name)} supports conference presentations, including oral and poster formats."
    await evaluator.verify(
        claim=conf_claim,
        node=conf_leaf,
        sources=uni.conference_support_urls,
        additional_instruction="Confirm official support for presenting at conferences. Prefer pages that explicitly mention support for presentations (oral/poster), travel funds, or presentation policies; if the page indicates support for presenting in general, treat it as inclusive of common formats."
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
    Evaluate the answer for the R1 interdisciplinary computational biomedical program requirements task.
    """
    # Initialize evaluator (use non-critical root to allow partial scoring across universities)
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

    # Add top-level node to mirror rubric root (non-critical to avoid strict-all fail)
    main_node = evaluator.add_parallel(
        id="Research_Program_Establishment",
        desc="Evaluate identification and verification of four R1 universities meeting all specified requirements",
        parent=root,
        critical=False
    )

    # Extract universities evidence from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_evidence"
    )

    # Prepare exactly 4 universities (pad if fewer)
    universities: List[UniversityEvidence] = (extracted.universities or [])[:4]
    while len(universities) < 4:
        universities.append(UniversityEvidence())

    # Add requirements summary as custom info
    evaluator.add_custom_info(
        info={
            "requirements": {
                "graduate_programs": [
                    "PhD in data/computational field",
                    "PhD in biomedical engineering",
                    "Minimum RA stipend >= $18,000/year"
                ],
                "research_infrastructure": ["IRB", "Research Compliance Office", "Technology Transfer Office"],
                "postdoc_programs": ["Structured postdoc positions", "Postdoc career development resources"],
                "funding_support": ["NSF grant support services", "Data Management Plan resources"],
                "faculty_support": ["Startup packages (equipment, supplies, personnel)", "Conference presentation support (oral and poster)"]
            },
            "expected_universities_count": 4
        },
        info_type="requirements_summary"
    )

    # Build verification trees for four universities
    for idx, uni in enumerate(universities):
        await verify_university(evaluator, main_node, uni, idx)

    # Return structured evaluation summary
    return evaluator.get_summary()