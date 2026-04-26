import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "caltech_10th_president_2026"
TASK_DESCRIPTION = (
    "In January 2026, the California Institute of Technology (Caltech) announced the appointment of their 10th president, "
    "with the appointment effective July 1, 2026. This individual has followed a progressive academic leadership career path, "
    "previously serving as provost at a major East Coast research university from 2023 to 2026, and before that, holding dean "
    "positions at two different universities: first as Dean of Science at a Canadian university (starting in 2014 and ending in 2018), "
    "and then as Dean of Arts and Sciences at an Ivy League institution (from 2018 to 2023). The individual holds a Ph.D. in astrophysics "
    "and conducted research in observational astrophysics. Identify this person by providing their full name, and verify their complete academic "
    "leadership career progression by documenting: (1) their educational background (undergraduate and doctoral institutions), (2) the specific details of "
    "each dean position (institution name, faculty/college led, and years of service), (3) the details of their provost position (institution name, title, and years of service), "
    "and (4) the details of their presidential appointment at Caltech (announcement date and effective date). All information must be supported with reference URLs."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PersonExtraction(BaseModel):
    full_name: Optional[str] = None
    caltech_ordinal: Optional[str] = None  # e.g., "10th"
    caltech_announcement_date: Optional[str] = None  # e.g., "January 12, 2026", or "January 2026"
    caltech_effective_date: Optional[str] = None  # e.g., "July 1, 2026"
    caltech_sources: List[str] = Field(default_factory=list)


class EducationResearchExtraction(BaseModel):
    undergraduate_institution: Optional[str] = None
    doctoral_institution: Optional[str] = None
    phd_field: Optional[str] = None  # should be "astrophysics" (allow variants)
    research_field: Optional[str] = None  # typically "observational astrophysics"
    education_sources: List[str] = Field(default_factory=list)


class RoleExtraction(BaseModel):
    institution_name: Optional[str] = None
    faculty_college_led: Optional[str] = None  # e.g., "Faculty of Science", "Faculty of Arts and Sciences"
    title: Optional[str] = None  # e.g., "Dean of Science", "Dean of Arts and Sciences", "Provost"
    start_year: Optional[str] = None  # e.g., "2014"
    end_year: Optional[str] = None  # e.g., "2018"
    sources: List[str] = Field(default_factory=list)


class ProvostExtraction(BaseModel):
    institution_name: Optional[str] = None
    title: Optional[str] = None  # e.g., "Provost", "Provost and Executive Vice President"
    start_year: Optional[str] = None  # e.g., "2023"
    end_year: Optional[str] = None  # e.g., "2026"
    sources: List[str] = Field(default_factory=list)


class CareerMetaExtraction(BaseModel):
    has_prior_faculty: Optional[bool] = None
    prior_faculty_details: Optional[str] = None
    meta_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_person() -> str:
    return (
        "Extract the individual's identification and Caltech presidency details as stated in the answer.\n"
        "Return the following fields:\n"
        "- full_name: the individual's full name.\n"
        "- caltech_ordinal: the ordinal number of the Caltech presidency (e.g., '10th').\n"
        "- caltech_announcement_date: the announcement date as stated (month and day if available, otherwise just month/year), e.g., 'January 2026' or 'January 12, 2026'.\n"
        "- caltech_effective_date: the effective date of the presidency (e.g., 'July 1, 2026').\n"
        "- caltech_sources: all URLs cited that support the presidency details.\n"
        "If any field is missing, set it to null. Ensure sources are actual URLs mentioned in the answer."
    )


def prompt_extract_education_research() -> str:
    return (
        "Extract the individual's education and research field details as stated in the answer.\n"
        "Return the following fields:\n"
        "- undergraduate_institution: the undergrad institution (name only).\n"
        "- doctoral_institution: the institution granting the Ph.D.\n"
        "- phd_field: the stated Ph.D. field (e.g., 'astrophysics').\n"
        "- research_field: the stated research field (e.g., 'observational astrophysics').\n"
        "- education_sources: all URLs cited that support the undergraduate institution, doctoral institution, Ph.D. field, or research field.\n"
        "If any field is missing, set it to null. Ensure sources are actual URLs mentioned in the answer."
    )


def prompt_extract_dean1() -> str:
    return (
        "Extract details of the first deanship (Dean of Science at a Canadian university, 2014–2018) as stated in the answer.\n"
        "Return the following fields:\n"
        "- institution_name: name of the university.\n"
        "- faculty_college_led: the faculty/college led (e.g., 'Faculty of Science').\n"
        "- title: the official dean title (e.g., 'Dean of Science').\n"
        "- start_year: starting year (e.g., '2014').\n"
        "- end_year: ending year (e.g., '2018').\n"
        "- sources: all URLs cited that support this deanship.\n"
        "Set missing fields to null. Extract only URLs actually present in the answer."
    )


def prompt_extract_dean2() -> str:
    return (
        "Extract details of the second deanship (Dean of Arts and Sciences at an Ivy League institution, 2018–2023) as stated in the answer.\n"
        "Return the following fields:\n"
        "- institution_name: name of the university.\n"
        "- faculty_college_led: the faculty/college led (e.g., 'Faculty of Arts and Sciences').\n"
        "- title: the official dean title (e.g., 'Dean of Arts and Sciences').\n"
        "- start_year: starting year (e.g., '2018').\n"
        "- end_year: ending year (e.g., '2023').\n"
        "- sources: all URLs cited that support this deanship.\n"
        "Set missing fields to null. Extract only URLs actually present in the answer."
    )


def prompt_extract_provost() -> str:
    return (
        "Extract details of the provost position (major East Coast research university, 2023–2026) as stated in the answer.\n"
        "Return the following fields:\n"
        "- institution_name: name of the university.\n"
        "- title: the official provost title used by the institution.\n"
        "- start_year: starting year (e.g., '2023').\n"
        "- end_year: ending year (e.g., '2026').\n"
        "- sources: all URLs cited that support the provost position.\n"
        "Set missing fields to null. Extract only URLs actually present in the answer."
    )


def prompt_extract_career_meta() -> str:
    return (
        "Extract meta-information about the individual's career progression trajectory as stated in the answer.\n"
        "Return the following fields:\n"
        "- has_prior_faculty: true/false depending on whether the answer explicitly states at least one faculty appointment before the first deanship.\n"
        "- prior_faculty_details: brief text describing the prior faculty appointment(s) as stated.\n"
        "- meta_sources: all URLs cited that support the overall career sequence or prior faculty appointment claim.\n"
        "If a field is missing, set it to null. Extract only URLs actually present in the answer."
    )


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _safe_int_year(year_str: Optional[str]) -> Optional[int]:
    if not year_str:
        return None
    try:
        return int(year_str.strip())
    except Exception:
        return None


def _merge_sources(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        for u in lst or []:
            if isinstance(u, str) and u.strip():
                merged.append(u.strip())
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for u in merged:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_individual_identification(
    evaluator: Evaluator,
    parent: Any,
    person: PersonExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Individual_Identification",
        desc="Identify the person and verify the Caltech presidential appointment details.",
        parent=parent,
        critical=True
    )

    # Name provided (existence check)
    evaluator.add_custom_node(
        result=bool(person.full_name and person.full_name.strip()),
        id="Name_Provided",
        desc="Provide the individual's full name.",
        parent=node,
        critical=True
    )

    # Presidency details group
    pres_node = evaluator.add_parallel(
        id="Presidential_Appointment_Details",
        desc="Confirm the Caltech appointment details match the constraints and are supported by citation(s).",
        parent=node,
        critical=True
    )

    # Citation existence (gate)
    evaluator.add_custom_node(
        result=bool(person.caltech_sources),
        id="Citation_Presidency",
        desc="Provide at least one reference URL that explicitly supports the presidential appointment details claimed (role/ordinal, announcement timing, and effective date).",
        parent=pres_node,
        critical=True
    )

    # Is 10th president
    leaf_10th = evaluator.add_leaf(
        id="Is_10th_President_of_Caltech",
        desc="Confirm the person is appointed as Caltech’s 10th president.",
        parent=pres_node,
        critical=True
    )
    claim_10th = f"{person.full_name or 'The individual'} was appointed as Caltech’s 10th president."
    await evaluator.verify(
        claim=claim_10th,
        node=leaf_10th,
        sources=person.caltech_sources,
        additional_instruction="Look for explicit mention of '10th' (tenth) presidency at Caltech. Minor wording variations are acceptable if clearly equivalent."
    )

    # Announcement in January 2026
    leaf_ann = evaluator.add_leaf(
        id="Announcement_In_January_2026",
        desc="Confirm the appointment was announced in January 2026.",
        parent=pres_node,
        critical=True
    )
    claim_ann = f"Caltech announced {person.full_name or 'the individual'}'s appointment in January 2026."
    await evaluator.verify(
        claim=claim_ann,
        node=leaf_ann,
        sources=person.caltech_sources,
        additional_instruction="Verify the announcement month is January and the year is 2026; the exact day may vary."
    )

    # Effective date July 1, 2026
    leaf_eff = evaluator.add_leaf(
        id="Effective_Date_July_1_2026",
        desc="Confirm the presidency is effective July 1, 2026.",
        parent=pres_node,
        critical=True
    )
    claim_eff = "The presidency is effective July 1, 2026."
    await evaluator.verify(
        claim=claim_eff,
        node=leaf_eff,
        sources=person.caltech_sources,
        additional_instruction="Confirm the effective/start date is July 1, 2026."
    )


async def build_education_research(
    evaluator: Evaluator,
    parent: Any,
    person: PersonExtraction,
    edu: EducationResearchExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Educational_and_Research_Background",
        desc="Provide and verify undergraduate/doctoral institutions and research field constraints, with citation(s).",
        parent=parent,
        critical=True
    )

    # Gate: at least one supporting URL
    evaluator.add_custom_node(
        result=bool(edu.education_sources),
        id="Citation_Education_Research",
        desc="Provide at least one reference URL that explicitly supports the education and research-field claims made.",
        parent=node,
        critical=True
    )

    # Undergraduate institution
    leaf_ug = evaluator.add_leaf(
        id="Undergraduate_Institution",
        desc="Identify the undergraduate institution.",
        parent=node,
        critical=True
    )
    claim_ug = f"{person.full_name or 'The individual'} completed undergraduate studies at {edu.undergraduate_institution or 'the stated undergraduate institution'}."
    await evaluator.verify(
        claim=claim_ug,
        node=leaf_ug,
        sources=edu.education_sources,
        additional_instruction="Accept synonyms such as 'bachelor's degree', 'undergraduate', or equivalent phrasing clearly identifying the institution."
    )

    # Doctoral institution
    leaf_phd_inst = evaluator.add_leaf(
        id="PhD_Institution",
        desc="Identify the institution granting the Ph.D.",
        parent=node,
        critical=True
    )
    claim_phd_inst = f"{person.full_name or 'The individual'} earned a Ph.D. from {edu.doctoral_institution or 'the stated doctoral institution'}."
    await evaluator.verify(
        claim=claim_phd_inst,
        node=leaf_phd_inst,
        sources=edu.education_sources,
        additional_instruction="Look for 'Ph.D.', 'doctoral', 'DPhil' or equivalent wording linking the institution to the degree."
    )

    # PhD in astrophysics
    leaf_phd_field = evaluator.add_leaf(
        id="PhD_in_Astrophysics",
        desc="Confirm the individual holds a Ph.D. in astrophysics.",
        parent=node,
        critical=True
    )
    claim_phd_field = f"{person.full_name or 'The individual'} holds a Ph.D. in astrophysics."
    await evaluator.verify(
        claim=claim_phd_field,
        node=leaf_phd_field,
        sources=edu.education_sources,
        additional_instruction="Allow reasonable variants like 'astronomy and astrophysics' if context clearly indicates a Ph.D. field centered on astrophysics."
    )

    # Observational astrophysics research
    leaf_obs = evaluator.add_leaf(
        id="Observational_Astrophysics_Research",
        desc="Confirm the individual conducted research in observational astrophysics.",
        parent=node,
        critical=True
    )
    claim_obs = f"{person.full_name or 'The individual'} conducted research in observational astrophysics."
    await evaluator.verify(
        claim=claim_obs,
        node=leaf_obs,
        sources=edu.education_sources,
        additional_instruction="Allow close wording variants such as 'observational astronomy' if clearly indicating the astrophysical observational domain."
    )


async def build_deanship1(
    evaluator: Evaluator,
    parent: Any,
    person: PersonExtraction,
    dean1: RoleExtraction
) -> None:
    node = evaluator.add_parallel(
        id="First_Deanship_Position",
        desc="Verify the first dean position: Dean of Science at a Canadian university, 2014–2018, with citation(s).",
        parent=parent,
        critical=True
    )

    # Citation gate
    evaluator.add_custom_node(
        result=bool(dean1.sources),
        id="Citation_Dean1",
        desc="Provide at least one reference URL that explicitly supports the first deanship details claimed.",
        parent=node,
        critical=True
    )

    # Institution name provided (existence check)
    evaluator.add_custom_node(
        result=bool(dean1.institution_name and dean1.institution_name.strip()),
        id="Dean1_Institution_Name",
        desc="Provide the institution name for the first deanship.",
        parent=node,
        critical=True
    )

    # Institution is Canadian
    leaf_canada = evaluator.add_leaf(
        id="Dean1_Is_Canadian_University",
        desc="Confirm the first deanship institution is in Canada.",
        parent=node,
        critical=True
    )
    claim_canada = f"{dean1.institution_name or 'The institution'} is in Canada."
    await evaluator.verify(
        claim=claim_canada,
        node=leaf_canada,
        sources=dean1.sources,
        additional_instruction="Accept clear geographic indicators (city/province in Canada, or explicit statement that the university is Canadian)."
    )

    # Title is Dean of Science (or equivalent)
    leaf_title = evaluator.add_leaf(
        id="Dean1_Title_Dean_of_Science",
        desc="Confirm the role is Dean of Science (or clearly equivalent).",
        parent=node,
        critical=True
    )
    claim_title = f"{person.full_name or 'The individual'} served as Dean of Science (or equivalent) at {dean1.institution_name or 'the institution'}."
    await evaluator.verify(
        claim=claim_title,
        node=leaf_title,
        sources=dean1.sources,
        additional_instruction="Allow equivalent titles (e.g., 'Dean, Faculty of Science') that clearly indicate leadership of the science faculty/college."
    )

    # Years 2014–2018
    leaf_years = evaluator.add_leaf(
        id="Dean1_Years_2014_2018",
        desc="Confirm the years of service for the first deanship are 2014–2018.",
        parent=node,
        critical=True
    )
    claim_years = f"{person.full_name or 'The individual'} served from 2014 to 2018 in the dean of science role at {dean1.institution_name or 'the institution'}."
    await evaluator.verify(
        claim=claim_years,
        node=leaf_years,
        sources=dean1.sources,
        additional_instruction="Exact months may vary; the year range must match 2014–2018."
    )


async def build_deanship2(
    evaluator: Evaluator,
    parent: Any,
    person: PersonExtraction,
    dean2: RoleExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Second_Deanship_Position",
        desc="Verify the second dean position: Dean of Arts and Sciences at an Ivy League institution, 2018–2023, with citation(s).",
        parent=parent,
        critical=True
    )

    # Citation gate
    evaluator.add_custom_node(
        result=bool(dean2.sources),
        id="Citation_Dean2",
        desc="Provide at least one reference URL that explicitly supports the second deanship details claimed.",
        parent=node,
        critical=True
    )

    # Institution name provided (existence check)
    evaluator.add_custom_node(
        result=bool(dean2.institution_name and dean2.institution_name.strip()),
        id="Dean2_Institution_Name",
        desc="Provide the institution name for the second deanship.",
        parent=node,
        critical=True
    )

    # Institution is Ivy League
    leaf_ivy = evaluator.add_leaf(
        id="Dean2_Is_Ivy_League",
        desc="Confirm the second deanship institution is Ivy League.",
        parent=node,
        critical=True
    )
    claim_ivy = f"{dean2.institution_name or 'The institution'} is an Ivy League university."
    await evaluator.verify(
        claim=claim_ivy,
        node=leaf_ivy,
        sources=dean2.sources,
        additional_instruction="Verify that the institution is a member of the Ivy League (e.g., Brown, Columbia, Cornell, Dartmouth, Harvard, Penn, Princeton, Yale)."
    )

    # Title is Dean of Arts and Sciences (or equivalent)
    leaf_title = evaluator.add_leaf(
        id="Dean2_Title_Dean_of_Arts_and_Sciences",
        desc="Confirm the role is Dean of Arts and Sciences (or clearly equivalent).",
        parent=node,
        critical=True
    )
    claim_title = f"{person.full_name or 'The individual'} served as Dean of Arts and Sciences (or equivalent) at {dean2.institution_name or 'the institution'}."
    await evaluator.verify(
        claim=claim_title,
        node=leaf_title,
        sources=dean2.sources,
        additional_instruction="Allow equivalent faculty naming (e.g., 'Faculty of Arts & Sciences') clearly indicating leadership of arts & sciences."
    )

    # Years 2018–2023
    leaf_years = evaluator.add_leaf(
        id="Dean2_Years_2018_2023",
        desc="Confirm the years of service for the second deanship are 2018–2023.",
        parent=node,
        critical=True
    )
    claim_years = f"{person.full_name or 'The individual'} served from 2018 to 2023 in the dean of arts and sciences role at {dean2.institution_name or 'the institution'}."
    await evaluator.verify(
        claim=claim_years,
        node=leaf_years,
        sources=dean2.sources,
        additional_instruction="Exact months may vary; the year range must match 2018–2023."
    )


async def build_provost(
    evaluator: Evaluator,
    parent: Any,
    person: PersonExtraction,
    prov: ProvostExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Provost_Position",
        desc="Verify the provost position: major East Coast research university, 2023–2026, with citation(s).",
        parent=parent,
        critical=True
    )

    # Citation gate
    evaluator.add_custom_node(
        result=bool(prov.sources),
        id="Citation_Provost",
        desc="Provide at least one reference URL that explicitly supports the provost position details claimed.",
        parent=node,
        critical=True
    )

    # Institution name provided (existence check)
    evaluator.add_custom_node(
        result=bool(prov.institution_name and prov.institution_name.strip()),
        id="Provost_Institution_Name",
        desc="Provide the institution name for the provost position.",
        parent=node,
        critical=True
    )

    # East Coast location evidence
    leaf_east = evaluator.add_leaf(
        id="Provost_Is_East_Coast",
        desc="Confirm the provost institution is located on the U.S. East Coast (as supported by cited source(s) or unambiguous geographic fact).",
        parent=node,
        critical=True
    )
    claim_east = f"{prov.institution_name or 'The institution'} is located on the U.S. East Coast."
    await evaluator.verify(
        claim=claim_east,
        node=leaf_east,
        sources=prov.sources,
        additional_instruction="Look for geographic location statements (state/city) that are clearly on the U.S. East Coast."
    )

    # Major research university evidence
    leaf_major = evaluator.add_leaf(
        id="Provost_Major_Research_Uni_Evidence",
        desc="Provide citation evidence that the provost institution is described as a major/leading research university (or equivalent wording) in the cited source(s).",
        parent=node,
        critical=True
    )
    claim_major = f"{prov.institution_name or 'The institution'} is described as a major or leading research university."
    await evaluator.verify(
        claim=claim_major,
        node=leaf_major,
        sources=prov.sources,
        additional_instruction="Accept equivalent wording such as 'leading research university', 'major research institution', or similar phrasing."
    )

    # Provost title
    leaf_title = evaluator.add_leaf(
        id="Provost_Title",
        desc="Provide the provost title used by the institution.",
        parent=node,
        critical=True
    )
    claim_title = f"{person.full_name or 'The individual'} held the provost title '{prov.title or 'Provost'}' at {prov.institution_name or 'the institution'}."
    await evaluator.verify(
        claim=claim_title,
        node=leaf_title,
        sources=prov.sources,
        additional_instruction="Verify the exact provost title as used by the institution (minor variations acceptable if clearly equivalent)."
    )

    # Provost years 2023–2026
    leaf_years = evaluator.add_leaf(
        id="Provost_Years_2023_2026",
        desc="Confirm the years of service for the provost role are 2023–2026.",
        parent=node,
        critical=True
    )
    claim_years = f"{person.full_name or 'The individual'} served as provost from 2023 to 2026 at {prov.institution_name or 'the institution'}."
    await evaluator.verify(
        claim=claim_years,
        node=leaf_years,
        sources=prov.sources,
        additional_instruction="Exact months may vary; the year range must match 2023–2026."
    )


async def build_career_constraints(
    evaluator: Evaluator,
    parent: Any,
    person: PersonExtraction,
    dean1: RoleExtraction,
    dean2: RoleExtraction,
    prov: ProvostExtraction,
    career: CareerMetaExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Career_Progression_Constraints",
        desc="Verify required meta-constraints about leadership trajectory, multi-institution experience, and approximate durations.",
        parent=parent,
        critical=True
    )

    # Faculty → Dean → Provost → President
    leaf_traj = evaluator.add_leaf(
        id="Faculty_to_Dean_to_Provost_to_President",
        desc="Verify the response demonstrates the stated trajectory faculty → dean → provost → president (includes at least one faculty appointment prior to the first deanship, and the administrative roles are in chronological order).",
        parent=node,
        critical=True
    )
    traj_claim = (
        f"The career progression shows at least one faculty appointment prior to 2014, followed by Dean of Science (2014–2018) at {dean1.institution_name or 'the first institution'}, "
        f"then Dean of Arts and Sciences (2018–2023) at {dean2.institution_name or 'the second institution'}, "
        f"then Provost (2023–2026) at {prov.institution_name or 'the third institution'}, and an appointment as Caltech’s 10th president effective July 1, 2026."
    )
    combined_sources = _merge_sources(career.meta_sources, dean1.sources, dean2.sources, prov.sources, person.caltech_sources)
    await evaluator.verify(
        claim=traj_claim,
        node=leaf_traj,
        sources=combined_sources,
        additional_instruction="It is acceptable if the sequence is corroborated across multiple cited URLs; the verification should allow reasonable aggregation of evidence."
    )

    # Multiple institution experience (custom logic)
    institutions = [
        dean1.institution_name or "",
        dean2.institution_name or "",
        prov.institution_name or "",
        "California Institute of Technology"
    ]
    uniq_insts = set([i for i in institutions if i.strip()])
    evaluator.add_custom_node(
        result=len(uniq_insts) >= 3,  # Must span multiple institutions
        id="Multiple_Institution_Experience",
        desc="Verify the roles span multiple institutions (not all at a single institution).",
        parent=node,
        critical=True
    )

    # Reasonable durations approx 3–5 years (custom logic)
    d1_s, d1_e = _safe_int_year(dean1.start_year), _safe_int_year(dean1.end_year)
    d2_s, d2_e = _safe_int_year(dean2.start_year), _safe_int_year(dean2.end_year)
    p_s, p_e = _safe_int_year(prov.start_year), _safe_int_year(prov.end_year)

    def _dur_ok(s: Optional[int], e: Optional[int]) -> bool:
        if s is None or e is None:
            return False
        diff = e - s
        return 3 <= diff <= 5

    evaluator.add_custom_node(
        result=_dur_ok(d1_s, d1_e) and _dur_ok(d2_s, d2_e) and _dur_ok(p_s, p_e),
        id="Reasonable_Durations_Approx_3_to_5_Years",
        desc="Verify each administrative position duration implied by the stated year ranges is approximately 3–5 years.",
        parent=node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
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
    Evaluate an answer for the Caltech 10th president task using the Mind2Web2 evaluation framework.
    """
    # Initialize evaluator (root is non-critical; we'll add a critical task node under root)
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
        default_model=model
    )

    # Perform extractions (in parallel for efficiency)
    person_task = evaluator.extract(
        prompt=prompt_extract_person(),
        template_class=PersonExtraction,
        extraction_name="person_identification"
    )
    edu_task = evaluator.extract(
        prompt=prompt_extract_education_research(),
        template_class=EducationResearchExtraction,
        extraction_name="education_research"
    )
    dean1_task = evaluator.extract(
        prompt=prompt_extract_dean1(),
        template_class=RoleExtraction,
        extraction_name="first_deanship"
    )
    dean2_task = evaluator.extract(
        prompt=prompt_extract_dean2(),
        template_class=RoleExtraction,
        extraction_name="second_deanship"
    )
    prov_task = evaluator.extract(
        prompt=prompt_extract_provost(),
        template_class=ProvostExtraction,
        extraction_name="provost_position"
    )
    career_task = evaluator.extract(
        prompt=prompt_extract_career_meta(),
        template_class=CareerMetaExtraction,
        extraction_name="career_meta"
    )

    person, edu, dean1, dean2, prov, career = await asyncio.gather(
        person_task, edu_task, dean1_task, dean2_task, prov_task, career_task
    )

    # Build the critical task-completion node
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify the individual appointed as Caltech’s 10th president (announced Jan 2026; effective July 1, 2026) and document/verify the specified education, research field, and leadership roles with supporting reference URLs.",
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_individual_identification(evaluator, task_node, person)
    await build_education_research(evaluator, task_node, person, edu)
    await build_deanship1(evaluator, task_node, person, dean1)
    await build_deanship2(evaluator, task_node, person, dean2)
    await build_provost(evaluator, task_node, person, prov)
    await build_career_constraints(evaluator, task_node, person, dean1, dean2, prov, career)

    # Return structured summary
    return evaluator.get_summary()