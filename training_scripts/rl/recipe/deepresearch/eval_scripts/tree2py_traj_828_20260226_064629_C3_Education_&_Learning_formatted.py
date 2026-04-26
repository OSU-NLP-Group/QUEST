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
TASK_ID = "academic_leader_identification"
TASK_DESCRIPTION = """Identify the academic leader who meets all of the following criteria:

Leadership Positions:
- Was appointed as the 21st president of Columbia University, with the appointment effective July 1, 2026
- Previously served as the 30th chancellor of the University of Wisconsin-Madison, starting August 4, 2022
- Previously served as dean of the UCLA School of Law from August 2015 to June 2022

Educational Background:
- Was born in 1967
- Received a PhD from the Massachusetts Institute of Technology (MIT) in 1999 in the field of history and sociology of science and technology

Family Connections:
- Has a father who is a professor at Harvard Law School specializing in negotiation and conflict resolution
- Is married to a professor of political science at the University of Wisconsin-Madison

Scholarly Work:
- Co-authored "The New Wigmore: A Treatise on Evidence: Expert Evidence"
- Was elected to the American Academy of Arts and Sciences on April 23, 2020

Provide the following information about this person:
1. The person's full name (first name, middle name if applicable, and last name)
2. The name of their father
3. The name of their spouse
4. Supporting URL references for: (a) the Columbia University presidential appointment, (b) the UW-Madison chancellor appointment, (c) the UCLA Law deanship, (d) their educational background, (e) their family connections, and (f) their scholarly contributions
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LeaderExtraction(BaseModel):
    person_full_name: Optional[str] = None
    father_name: Optional[str] = None
    spouse_name: Optional[str] = None

    columbia_appointment_urls: List[str] = Field(default_factory=list)
    uw_chancellor_urls: List[str] = Field(default_factory=list)
    ucla_deanship_urls: List[str] = Field(default_factory=list)
    education_urls: List[str] = Field(default_factory=list)
    family_urls: List[str] = Field(default_factory=list)
    scholarly_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_leader_info() -> str:
    return """
Extract the following items exactly as they appear in the answer text. Do not invent anything.

Required identity fields:
- person_full_name: The person's full name (include middle name/initial if present in the answer).
- father_name: The father's name.
- spouse_name: The spouse's name.

Supporting URLs (extract the actual URLs mentioned; include full http/https):
- columbia_appointment_urls: URLs cited to support the Columbia University presidential appointment (21st president; effective July 1, 2026).
- uw_chancellor_urls: URLs cited to support the UW–Madison chancellor appointment (30th chancellor; start date Aug 4, 2022).
- ucla_deanship_urls: URLs cited to support the UCLA School of Law deanship (Aug 2015–June 2022).
- education_urls: URLs cited to support educational/biographical details (e.g., degrees, birth year, etc.).
- family_urls: URLs cited to support family connections (father and spouse).
- scholarly_urls: URLs cited to support scholarly/professional contributions (e.g., treatise co-authorship, AAAS election, etc.).

If an item is not provided in the answer, return null for strings or an empty list for URL lists.
"""


# --------------------------------------------------------------------------- #
# Helper: presence checks                                                     #
# --------------------------------------------------------------------------- #
def has_two_words(name: Optional[str]) -> bool:
    if not name:
        return False
    parts = [p for p in name.strip().split() if p]
    return len(parts) >= 2


# --------------------------------------------------------------------------- #
# Build "Supporting_URL_References_Provided" subtree                          #
# --------------------------------------------------------------------------- #
def build_supporting_urls_group(
    evaluator: Evaluator,
    parent_node,
    extracted: LeaderExtraction
) -> Dict[str, Any]:
    urls_group = evaluator.add_parallel(
        id="Supporting_URL_References_Provided",
        desc="Provides supporting URL references for each required category (a)–(f).",
        parent=parent_node,
        critical=True
    )

    node_url_columbia = evaluator.add_custom_node(
        result=len(extracted.columbia_appointment_urls) > 0,
        id="URL_Columbia_Appointment",
        desc="Provides at least one supporting URL for the Columbia University presidential appointment (21st president; effective date).",
        parent=urls_group,
        critical=True
    )

    node_url_uw = evaluator.add_custom_node(
        result=len(extracted.uw_chancellor_urls) > 0,
        id="URL_UW_Chancellor_Appointment",
        desc="Provides at least one supporting URL for the UW–Madison chancellor appointment (30th chancellor; start date).",
        parent=urls_group,
        critical=True
    )

    node_url_ucla = evaluator.add_custom_node(
        result=len(extracted.ucla_deanship_urls) > 0,
        id="URL_UCLA_Deanship",
        desc="Provides at least one supporting URL for the UCLA Law deanship (term dates).",
        parent=urls_group,
        critical=True
    )

    node_url_edu = evaluator.add_custom_node(
        result=len(extracted.education_urls) > 0,
        id="URL_Educational_Background",
        desc="Provides at least one supporting URL for the educational background constraints (degrees and/or PhD details as required).",
        parent=urls_group,
        critical=True
    )

    node_url_family = evaluator.add_custom_node(
        result=len(extracted.family_urls) > 0,
        id="URL_Family_Connections",
        desc="Provides at least one supporting URL for the family-connection constraints (father and spouse).",
        parent=urls_group,
        critical=True
    )

    node_url_scholarly = evaluator.add_custom_node(
        result=len(extracted.scholarly_urls) > 0,
        id="URL_Scholarly_Contributions",
        desc="Provides at least one supporting URL for the scholarly/professional constraints (e.g., treatise coauthorship and AAAS election, and/or other listed scholarly constraints).",
        parent=urls_group,
        critical=True
    )

    return {
        "columbia": node_url_columbia,
        "uw": node_url_uw,
        "ucla": node_url_ucla,
        "edu": node_url_edu,
        "family": node_url_family,
        "scholarly": node_url_scholarly,
    }


# --------------------------------------------------------------------------- #
# Build "Required_Outputs" subtree                                            #
# --------------------------------------------------------------------------- #
def build_required_outputs_group(
    evaluator: Evaluator,
    parent_node,
    extracted: LeaderExtraction
) -> Dict[str, Any]:
    req_group = evaluator.add_parallel(
        id="Required_Outputs",
        desc="Provide the requested identity outputs.",
        parent=parent_node,
        critical=True
    )

    node_person = evaluator.add_custom_node(
        result=has_two_words(extracted.person_full_name),
        id="Person_Full_Name_Provided",
        desc="Provides the person’s full name (first and last at minimum; includes middle name/initial if applicable).",
        parent=req_group,
        critical=True
    )

    node_father = evaluator.add_custom_node(
        result=bool(extracted.father_name and extracted.father_name.strip()),
        id="Father_Name_Provided",
        desc="Provides the father’s name.",
        parent=req_group,
        critical=True
    )

    node_spouse = evaluator.add_custom_node(
        result=bool(extracted.spouse_name and extracted.spouse_name.strip()),
        id="Spouse_Name_Provided",
        desc="Provides the spouse’s name.",
        parent=req_group,
        critical=True
    )

    return {
        "person_name_node": node_person,
        "father_name_node": node_father,
        "spouse_name_node": node_spouse,
    }


# --------------------------------------------------------------------------- #
# Build "All_Constraints_Satisfied_By_Identified_Person" subtree              #
# --------------------------------------------------------------------------- #
async def build_all_constraints_group(
    evaluator: Evaluator,
    parent_node,
    extracted: LeaderExtraction,
    prereq_nodes: Dict[str, Any],
    req_output_nodes: Dict[str, Any]
):
    all_group = evaluator.add_parallel(
        id="All_Constraints_Satisfied_By_Identified_Person",
        desc="The identified person matches ALL constraints listed in the constraints section.",
        parent=parent_node,
        critical=True
    )

    # Convenience
    name_val = extracted.person_full_name or ""

    # ----------------------------- Leadership ----------------------------- #
    leadership = evaluator.add_parallel(
        id="Leadership_Positions",
        desc="Meets all leadership-position constraints.",
        parent=all_group,
        critical=True
    )

    # Columbia President
    leaf_columbia = evaluator.add_leaf(
        id="Columbia_President_Constraint",
        desc="Appointed as the 21st president of Columbia University, effective July 1, 2026.",
        parent=leadership,
        critical=True
    )
    claim_columbia = (
        f"{name_val} was appointed as the 21st president of Columbia University, "
        f"with the appointment effective July 1, 2026."
    )
    await evaluator.verify(
        claim=claim_columbia,
        node=leaf_columbia,
        sources=extracted.columbia_appointment_urls,
        extra_prerequisites=[prereq_nodes["columbia"], req_output_nodes["person_name_node"]],
        additional_instruction="Verify that the cited page(s) explicitly reference the person as the 21st President of Columbia University and the effective date July 1, 2026. Allow minor formatting differences; the person’s name can include or omit middle initial."
    )

    # UW Chancellor
    leaf_uw = evaluator.add_leaf(
        id="UW_Chancellor_Constraint",
        desc="Served as the 30th chancellor of the University of Wisconsin–Madison starting August 4, 2022.",
        parent=leadership,
        critical=True
    )
    claim_uw = (
        f"{name_val} served as the 30th chancellor of the University of Wisconsin–Madison, "
        f"starting on August 4, 2022."
    )
    await evaluator.verify(
        claim=claim_uw,
        node=leaf_uw,
        sources=extracted.uw_chancellor_urls,
        extra_prerequisites=[prereq_nodes["uw"], req_output_nodes["person_name_node"]],
        additional_instruction="Verify that the cited page(s) indicate the person as the 30th UW–Madison chancellor and the start date (Aug 4, 2022). Accept minor date formatting variations."
    )

    # UCLA Dean
    leaf_ucla = evaluator.add_leaf(
        id="UCLA_Dean_Constraint",
        desc="Served as dean of the UCLA School of Law from August 2015 to June 2022.",
        parent=leadership,
        critical=True
    )
    claim_ucla = (
        f"{name_val} served as dean of the UCLA School of Law from August 2015 to June 2022."
    )
    await evaluator.verify(
        claim=claim_ucla,
        node=leaf_ucla,
        sources=extracted.ucla_deanship_urls,
        extra_prerequisites=[prereq_nodes["ucla"], req_output_nodes["person_name_node"]],
        additional_instruction="Verify that the cited page(s) show the person as dean of UCLA School of Law with a tenure spanning from August 2015 through June 2022 (allowing reasonable month/day depiction)."
    )

    # --------------------- Education & Biographical ----------------------- #
    edu = evaluator.add_parallel(
        id="Education_And_Biographical_Constraints",
        desc="Meets all education/biographical constraints.",
        parent=all_group,
        critical=True
    )

    # Birth year 1967
    leaf_birth_year = evaluator.add_leaf(
        id="Birth_Year_1967",
        desc="Born in 1967.",
        parent=edu,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} was born in 1967.",
        node=leaf_birth_year,
        sources=extracted.education_urls,
        extra_prerequisites=[prereq_nodes["edu"], req_output_nodes["person_name_node"]],
        additional_instruction="Confirm that the page indicates a birth year of 1967 for this person."
    )

    # Birthplace Cambridge, MA (from rubric)
    leaf_birthplace = evaluator.add_leaf(
        id="Birthplace_Cambridge_MA",
        desc="Born in Cambridge, Massachusetts.",
        parent=edu,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} was born in Cambridge, Massachusetts.",
        node=leaf_birthplace,
        sources=extracted.education_urls,
        extra_prerequisites=[prereq_nodes["edu"], req_output_nodes["person_name_node"]],
        additional_instruction="Verify that the page(s) indicate Cambridge, Massachusetts as the person’s birthplace. Allow minor variations like 'Cambridge, MA'."
    )

    # BA Harvard 1988
    leaf_ba = evaluator.add_leaf(
        id="BA_Harvard_1988",
        desc="Received a Bachelor of Arts from Harvard University in 1988.",
        parent=edu,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} received a Bachelor of Arts from Harvard University in 1988.",
        node=leaf_ba,
        sources=extracted.education_urls,
        extra_prerequisites=[prereq_nodes["edu"], req_output_nodes["person_name_node"]],
        additional_instruction="Verify that the person holds a BA from Harvard with a 1988 graduation year. Allow minor phrasing variations."
    )

    # JD Yale 1995
    leaf_jd = evaluator.add_leaf(
        id="JD_Yale_1995",
        desc="Received a Juris Doctor from Yale Law School in 1995.",
        parent=edu,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} received a Juris Doctor from Yale Law School in 1995.",
        node=leaf_jd,
        sources=extracted.education_urls,
        extra_prerequisites=[prereq_nodes["edu"], req_output_nodes["person_name_node"]],
        additional_instruction="Verify that the person has a JD from Yale Law School with a 1995 graduation year."
    )

    # PhD MIT 1999 in HSST
    leaf_phd = evaluator.add_leaf(
        id="PhD_MIT_1999_HSST",
        desc="Received a PhD from MIT in 1999 in history and sociology of science and technology.",
        parent=edu,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} received a PhD from MIT in 1999 in history and sociology of science and technology.",
        node=leaf_phd,
        sources=extracted.education_urls,
        extra_prerequisites=[prereq_nodes["edu"], req_output_nodes["person_name_node"]],
        additional_instruction="Confirm PhD at MIT in 1999 in the field often described as 'history and sociology of science and technology' (allow abbreviations like HSST and close paraphrases)."
    )

    # UVA Law faculty 1998-2005
    leaf_uva = evaluator.add_leaf(
        id="UVA_Law_Faculty_1998_2005",
        desc="Was on the faculty of the University of Virginia School of Law from 1998 to 2005.",
        parent=edu,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} was on the faculty of the University of Virginia School of Law from 1998 to 2005.",
        node=leaf_uva,
        sources=extracted.education_urls,
        extra_prerequisites=[prereq_nodes["edu"], req_output_nodes["person_name_node"]],
        additional_instruction="Verify that the page(s) indicate a UVA Law faculty appointment spanning years 1998–2005. Allow minor phrasing variations."
    )

    # Harvard Visiting Professor 2002–2003
    leaf_hls_visiting = evaluator.add_leaf(
        id="Harvard_Visiting_Prof_2002_2003",
        desc="Spent one year (2002–2003) as a visiting professor at Harvard Law School.",
        parent=edu,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} spent the 2002–2003 academic year as a visiting professor at Harvard Law School.",
        node=leaf_hls_visiting,
        sources=extracted.education_urls,
        extra_prerequisites=[prereq_nodes["edu"], req_output_nodes["person_name_node"]],
        additional_instruction="Confirm a visiting professorship at Harvard Law School for the year 2002–2003."
    )

    # ----------------------------- Family --------------------------------- #
    family = evaluator.add_parallel(
        id="Family_Constraints",
        desc="Meets all family-connection constraints.",
        parent=all_group,
        critical=True
    )

    # Father: Robert Mnookin at HLS specializing in negotiation & conflict resolution
    leaf_father = evaluator.add_leaf(
        id="Father_Constraint",
        desc="Father is Robert Mnookin, a Harvard Law School professor specializing in negotiation and conflict resolution.",
        parent=family,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val}'s father is Robert Mnookin, a professor at Harvard Law School who specializes in negotiation and conflict resolution.",
        node=leaf_father,
        sources=extracted.family_urls,
        extra_prerequisites=[prereq_nodes["family"], req_output_nodes["person_name_node"]],
        additional_instruction="Verify that the cited page(s) link the person to father Robert Mnookin and indicate his HLS affiliation and specialization in negotiation/conflict resolution."
    )

    # Spouse: Joshua Foa Dienstag (UW–Madison political science) and married in 1996
    leaf_spouse = evaluator.add_leaf(
        id="Spouse_Constraint",
        desc="Married to Joshua Foa Dienstag, a professor of political science at the University of Wisconsin–Madison, and married in 1996.",
        parent=family,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} is married to Joshua Foa Dienstag, a professor of political science at the University of Wisconsin–Madison, and they were married in 1996.",
        node=leaf_spouse,
        sources=extracted.family_urls,
        extra_prerequisites=[prereq_nodes["family"], req_output_nodes["person_name_node"]],
        additional_instruction="Verify spouse name and UW–Madison political science affiliation; confirm marriage year 1996 if available on the cited page(s). Allow reasonable phrasing variations."
    )

    # -------------------- Scholarly & Professional ------------------------ #
    scholarly = evaluator.add_parallel(
        id="Scholarly_And_Professional_Constraints",
        desc="Meets all scholarly/professional constraints.",
        parent=all_group,
        critical=True
    )

    # AAAS election on 2020-04-23
    leaf_aaas = evaluator.add_leaf(
        id="AAAS_Elected_2020_04_23",
        desc="Elected to the American Academy of Arts and Sciences on April 23, 2020.",
        parent=scholarly,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} was elected to the American Academy of Arts and Sciences on April 23, 2020.",
        node=leaf_aaas,
        sources=extracted.scholarly_urls,
        extra_prerequisites=[prereq_nodes["scholarly"], req_output_nodes["person_name_node"]],
        additional_instruction="Verify that the page(s) indicate the person’s election to the American Academy of Arts and Sciences with the date April 23, 2020 (allow minor formatting differences)."
    )

    # New Wigmore coauthorship
    leaf_wigmore = evaluator.add_leaf(
        id="New_Wigmore_Coauthorship_Constraint",
        desc="Is a co-author of 'The New Wigmore: A Treatise on Evidence: Expert Evidence' with specified attribution details.",
        parent=scholarly,
        critical=True
    )
    claim_wigmore = (
        f"{name_val} is a co-author of 'The New Wigmore: A Treatise on Evidence: Expert Evidence' "
        f"(often associated with David H. Kaye and David E. Bernstein; general editor Richard D. Friedman)."
    )
    await evaluator.verify(
        claim=claim_wigmore,
        node=leaf_wigmore,
        sources=extracted.scholarly_urls,
        extra_prerequisites=[prereq_nodes["scholarly"], req_output_nodes["person_name_node"]],
        additional_instruction="Focus on confirming that the person is listed as an author/co-author (or co-editor) of the Expert Evidence volume in The New Wigmore series. Allow variations in how contributor roles are labeled."
    )

    # PCAST 2016 advisory group co-chair
    leaf_pcast = evaluator.add_leaf(
        id="PCAST_Advisory_Group_Cochair_2016",
        desc="Co-chaired an advisory group to the President’s Council of Advisors on Science and Technology in 2016 that issued a report on forensic science reliability.",
        parent=scholarly,
        critical=True
    )
    await evaluator.verify(
        claim=f"In 2016, {name_val} co-chaired an advisory group to the President’s Council of Advisors on Science and Technology that issued a report on forensic science reliability.",
        node=leaf_pcast,
        sources=extracted.scholarly_urls,
        extra_prerequisites=[prereq_nodes["scholarly"], req_output_nodes["person_name_node"]],
        additional_instruction="Verify that the page(s) credibly link the person to a 2016 PCAST-related advisory role and the forensic science reliability report."
    )

    # Specialization in evidence/forensics/expert testimony
    leaf_specializes = evaluator.add_leaf(
        id="Specializes_In_Evidence_Forensics_Expert_Testimony",
        desc="Specializes in evidence law, forensic science, and expert testimony.",
        parent=scholarly,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_val} specializes in evidence law, forensic science, and expert testimony.",
        node=leaf_specializes,
        sources=extracted.scholarly_urls,
        extra_prerequisites=[prereq_nodes["scholarly"], req_output_nodes["person_name_node"]],
        additional_instruction="Verify that the page(s) describe the person’s scholarly focus in these areas (evidence law, forensic science, expert testimony). Allow paraphrases."
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
    Evaluate an answer for the academic leader identification task.
    """
    # Initialize evaluator with a parallel root
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

    # Top-level rubric node
    top = evaluator.add_parallel(
        id="Academic_Leader_Identification",
        desc="Identify the academic leader matching all stated constraints and provide the requested outputs and supporting URLs.",
        parent=root,
        critical=True
    )

    # Extract structured information
    extracted: LeaderExtraction = await evaluator.extract(
        prompt=prompt_extract_leader_info(),
        template_class=LeaderExtraction,
        extraction_name="leader_extraction"
    )

    # Build URL supporting group first (so constraint leaves can depend on these)
    url_nodes = build_supporting_urls_group(evaluator, top, extracted)

    # Build Required Outputs group
    req_output_nodes = build_required_outputs_group(evaluator, top, extracted)

    # Build All Constraints group (with verification using URLs)
    await build_all_constraints_group(
        evaluator=evaluator,
        parent_node=top,
        extracted=extracted,
        prereq_nodes=url_nodes,
        req_output_nodes=req_output_nodes
    )

    # Return evaluation summary
    return evaluator.get_summary()