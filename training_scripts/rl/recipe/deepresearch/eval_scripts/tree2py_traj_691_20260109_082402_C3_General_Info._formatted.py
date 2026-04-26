import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "stanford_president_genealogy"
TASK_DESCRIPTION = (
    "Identify the individual who served as Stanford University's president from 2000 to 2016. "
    "For this president, provide the following information: "
    "(1) The institution where they earned their PhD, the year it was awarded, and the field of study; "
    "(2) The name of their doctoral advisor at that institution; "
    "(3) For that doctoral advisor: the institution where they earned their PhD, the year it was awarded, and the field of study; "
    "(4) The name of their doctoral advisor's doctoral advisor; "
    "(5) For this second-generation advisor: the institution where they earned their bachelor's degree and the year, the institution where they earned their PhD and the year, and any historically significant detail about their PhD (if applicable). "
    "Provide URL references for each piece of information to verify your findings."
)

# Ground-truth anchors explicitly specified by rubric constraints
GT_PRESIDENT_NAME = "John L. Hennessy"  # Accept minor variants like "John Hennessy"
GT_PRESIDENT_PHD_INSTITUTION_ALIASES = ["Stony Brook University", "SUNY Stony Brook", "Stony Brook (SUNY)"]
GT_PRESIDENT_PHD_YEAR = "1977"
GT_PRESIDENT_PHD_FIELD_ALIASES = ["Computer Science", "CS"]
GT_FIRST_ADVISOR_PHD_INSTITUTION = "University of Washington"
GT_FIRST_ADVISOR_PHD_YEAR = "1960"
GT_FIRST_ADVISOR_PHD_FIELD = "Electrical Engineering"
GT_SECOND_ADVISOR_BS_INSTITUTION = "University of Tokyo"
GT_SECOND_ADVISOR_BS_YEAR = "1951"
GT_SECOND_ADVISOR_PHD_INSTITUTION = "University of Washington"
GT_SECOND_ADVISOR_PHD_YEAR = "1958"
GT_SECOND_ADVISOR_HISTORICAL_DETAIL_KEY = (
    "first doctoral degree awarded by the University of Washington Electrical Engineering department"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PresidentInfo(BaseModel):
    name: Optional[str] = None
    identification_sources: List[str] = Field(default_factory=list)  # URLs confirming tenure 2000–2016

    phd_institution: Optional[str] = None
    phd_year: Optional[str] = None
    phd_field: Optional[str] = None
    phd_sources: List[str] = Field(default_factory=list)  # URLs supporting institution/year/field


class FirstAdvisorInfo(BaseModel):
    name: Optional[str] = None
    relationship_sources: List[str] = Field(default_factory=list)  # URLs confirming advisor relationship (president ↔ advisor)
    stony_brook_professor_sources: List[str] = Field(default_factory=list)  # URLs supporting professorship at SBU

    phd_institution: Optional[str] = None
    phd_year: Optional[str] = None
    phd_field: Optional[str] = None
    phd_sources: List[str] = Field(default_factory=list)  # URLs supporting institution/year/field


class SecondAdvisorInfo(BaseModel):
    name: Optional[str] = None
    relationship_sources: List[str] = Field(default_factory=list)  # URLs confirming advisor relationship (first advisor ↔ second advisor)
    uw_professor_sources: List[str] = Field(default_factory=list)  # URLs supporting professorship at UW

    bachelors_institution: Optional[str] = None
    bachelors_year: Optional[str] = None
    phd_institution: Optional[str] = None
    phd_year: Optional[str] = None
    historical_significance: Optional[str] = None
    education_sources: List[str] = Field(default_factory=list)  # URLs supporting BS/PhD details and historical note


class GenealogyExtraction(BaseModel):
    president: Optional[PresidentInfo] = None
    first_advisor: Optional[FirstAdvisorInfo] = None
    second_advisor: Optional[SecondAdvisorInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_genealogy() -> str:
    return """
    Extract a structured summary of the answer that focuses on:
    1) The Stanford University president who served from 2000 to 2016 (the identified person, not anybody else).
       - name: the individual's full name as stated in the answer (e.g., "John L. Hennessy" or "John Hennessy")
       - identification_sources: all URLs explicitly provided in the answer that confirm their Stanford presidential tenure covering 2000–2016.
    2) That president's PhD details:
       - phd_institution: the institution awarding the PhD as stated in the answer (e.g., "Stony Brook University", "SUNY Stony Brook")
       - phd_year: the year the PhD was awarded as stated in the answer (string, do not convert to int)
       - phd_field: the field/discipline of the PhD as stated in the answer (e.g., "Computer Science")
       - phd_sources: all URLs explicitly provided that support these PhD details.
    3) The president's doctoral advisor (first advisor):
       - name: the advisor's full name as stated in the answer.
       - relationship_sources: all URLs explicitly provided that confirm the advisor relationship between the president and this advisor.
       - stony_brook_professor_sources: all URLs explicitly provided that show this advisor was a professor at Stony Brook University.
       - phd_institution: the institution awarding this advisor's PhD as stated in the answer (e.g., "University of Washington")
       - phd_year: the year the advisor's PhD was awarded (string)
       - phd_field: the field/discipline of the advisor's PhD (e.g., "Electrical Engineering")
       - phd_sources: all URLs explicitly provided that support these advisor PhD details.
    4) The first advisor's doctoral advisor (second-generation advisor):
       - name: the person’s full name as stated in the answer.
       - relationship_sources: all URLs explicitly provided that confirm the advisor relationship between the first advisor and this second-generation advisor.
       - uw_professor_sources: all URLs explicitly provided that show this second-generation advisor was a professor at the University of Washington.
       - bachelors_institution: the institution awarding the bachelor’s degree
       - bachelors_year: the year of the bachelor’s degree (string)
       - phd_institution: the institution awarding the PhD
       - phd_year: the year of the PhD (string)
       - historical_significance: any historically significant detail about the PhD (e.g., 'the first doctoral degree awarded by the University of Washington Electrical Engineering department'), as stated in the answer
       - education_sources: all URLs that support the bachelor’s details, PhD details, and the historical PhD note.

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer (plain URLs or markdown links). Do not invent URLs.
    - If any field is missing in the answer, set it to null (or an empty array for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Filter obviously invalid entries
    return [u for u in urls if isinstance(u, str) and u.strip() and ("http://" in u or "https://" in u)]


def _name_equivalence_instruction() -> str:
    return (
        "When comparing names, allow minor variants (e.g., with/without middle initial), case differences, and "
        "reasonable formatting differences. Consider 'John L. Hennessy' and 'John Hennessy' equivalent."
    )


def _institution_alias_instruction() -> str:
    return (
        "Treat 'Stony Brook University' and 'SUNY Stony Brook' (or 'Stony Brook (SUNY)') as equivalent names for the same institution."
    )


def _field_alias_instruction_cs() -> str:
    return (
        "Treat 'Computer Science' and 'CS' (or minor paraphrases) as equivalent."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_president_identification(
    evaluator: Evaluator,
    parent_node,
    extracted: GenealogyExtraction,
):
    node = evaluator.add_parallel(
        id="stanford_president_identification",
        desc="Identify the individual who served as Stanford University president from 2000 to 2016",
        parent=parent_node,
        critical=True,
    )

    pres = extracted.president or PresidentInfo()

    # Leaf: president_name (check matches ground-truth person)
    leaf_name = evaluator.add_leaf(
        id="president_name",
        desc="Provide the name of the Stanford University president who served from 2000 to 2016",
        parent=node,
        critical=True,
    )
    claimed_name = pres.name or ""
    claim_name = (
        f"The individual who served as Stanford University's president from 2000 to 2016 is {GT_PRESIDENT_NAME}. "
        f"The answer identifies this person as '{claimed_name}'. These refer to the same person."
    )
    await evaluator.verify(
        claim=claim_name,
        node=leaf_name,
        additional_instruction=_name_equivalence_instruction(),
    )

    # Leaf: president_identification_reference (verify tenure on provided URLs)
    leaf_ref = evaluator.add_leaf(
        id="president_identification_reference",
        desc="Provide a valid URL reference confirming the presidential tenure dates",
        parent=node,
        critical=True,
    )
    name_for_claim = claimed_name if claimed_name else GT_PRESIDENT_NAME
    claim_tenure = (
        f"{name_for_claim} served as Stanford University's president from 2000 to 2016."
    )
    await evaluator.verify(
        claim=claim_tenure,
        node=leaf_ref,
        sources=_safe_urls(pres.identification_sources),
        additional_instruction="The page should explicitly confirm both the role (Stanford president) and the 2000–2016 tenure window. Allow minor phrasing variants like 'served from 2000 until 2016'.",
    )


async def build_president_phd_background(
    evaluator: Evaluator,
    parent_node,
    extracted: GenealogyExtraction,
):
    seq_node = evaluator.add_sequential(
        id="president_educational_background",
        desc="Provide and verify the PhD credentials of the identified Stanford president",
        parent=parent_node,
        critical=True,
    )
    pres = extracted.president or PresidentInfo()

    # Parallel details (institution/year/field)
    details = evaluator.add_parallel(
        id="president_phd_details",
        desc="Provide complete PhD information for the identified president",
        parent=seq_node,
        critical=True,
    )

    # Institution check (must match Stony Brook University or SUNY Stony Brook)
    leaf_phd_inst = evaluator.add_leaf(
        id="phd_institution",
        desc="Identify the institution that awarded the PhD (per constraints: Stony Brook University / SUNY Stony Brook)",
        parent=details,
        critical=True,
    )
    inst = pres.phd_institution or ""
    aliases_text = ", ".join(GT_PRESIDENT_PHD_INSTITUTION_ALIASES)
    claim_inst = (
        f"The PhD institution stated in the answer is '{inst}'. "
        f"This is equivalent to one of the following: {aliases_text}."
    )
    await evaluator.verify(
        claim=claim_inst,
        node=leaf_phd_inst,
        additional_instruction=_institution_alias_instruction(),
    )

    # Year check (must be 1977)
    leaf_phd_year = evaluator.add_leaf(
        id="phd_year",
        desc="Identify the year the PhD was awarded (per constraints: 1977)",
        parent=details,
        critical=True,
    )
    year = pres.phd_year or ""
    claim_year = (
        f"The PhD year stated in the answer is '{year}', and this equals {GT_PRESIDENT_PHD_YEAR}."
    )
    await evaluator.verify(
        claim=claim_year,
        node=leaf_phd_year,
        additional_instruction="Allow minor formatting differences; evaluate whether the numeric year equals 1977.",
    )

    # Field check (must be Computer Science)
    leaf_phd_field = evaluator.add_leaf(
        id="phd_field",
        desc="Identify the field of study for the PhD (per constraints: Computer Science)",
        parent=details,
        critical=True,
    )
    field = pres.phd_field or ""
    claim_field = (
        f"The PhD field stated in the answer is '{field}', and this is equivalent to 'Computer Science'."
    )
    await evaluator.verify(
        claim=claim_field,
        node=leaf_phd_field,
        additional_instruction=_field_alias_instruction_cs(),
    )

    # Reference check with URLs (supporting institution/year/field together)
    leaf_phd_ref = evaluator.add_leaf(
        id="president_phd_reference",
        desc="Provide a valid URL reference supporting the president's PhD institution, year, and field",
        parent=seq_node,
        critical=True,
    )
    name_for_claim = (extracted.president.name if extracted.president and extracted.president.name else GT_PRESIDENT_NAME)
    claim_ref = (
        f"According to the provided source, {name_for_claim} earned a PhD in Computer Science from Stony Brook University "
        f"(also known as SUNY Stony Brook) in 1977."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=leaf_phd_ref,
        sources=_safe_urls(pres.phd_sources),
        additional_instruction="The page should explicitly support the institution, the year 1977, and the field (Computer Science). Apply institution alias equivalence and field alias equivalence.",
    )


async def build_first_advisor_chain(
    evaluator: Evaluator,
    parent_node,
    extracted: GenealogyExtraction,
):
    seq_node = evaluator.add_sequential(
        id="first_advisor_chain",
        desc="Identify and verify the doctoral advisor of the Stanford president, including required affiliation constraint",
        parent=parent_node,
        critical=True,
    )
    pres = extracted.president or PresidentInfo()
    first = extracted.first_advisor or FirstAdvisorInfo()

    # Identification and affiliation (parallel)
    id_aff = evaluator.add_parallel(
        id="first_advisor_identification_and_affiliation",
        desc="Identify the president's doctoral advisor and verify the required affiliation condition",
        parent=seq_node,
        critical=True,
    )

    # Name provided (existence check)
    evaluator.add_custom_node(
        result=bool(first.name and first.name.strip()),
        id="first_advisor_name",
        desc="Provide the name of the president's doctoral advisor",
        parent=id_aff,
        critical=True,
    )

    # Relationship reference (advisor relationship supported by URLs)
    leaf_rel = evaluator.add_leaf(
        id="first_advisor_relationship_reference",
        desc="Provide a valid URL reference confirming the advisor relationship (president ↔ first advisor)",
        parent=id_aff,
        critical=True,
    )
    name_pres = pres.name or GT_PRESIDENT_NAME
    name_first = first.name or "the advisor named in the answer"
    claim_rel = f"{name_first} was the doctoral advisor (PhD advisor/supervisor) of {name_pres}."
    await evaluator.verify(
        claim=claim_rel,
        node=leaf_rel,
        sources=_safe_urls(first.relationship_sources),
        additional_instruction="The page should clearly state the doctoral advisor–advisee relationship between these two people (or equivalent phrasing).",
    )

    # Professorship at Stony Brook - reference presence (as a prerequisite)
    ref_presence_prof_sbu = evaluator.add_custom_node(
        result=len(_safe_urls(first.stony_brook_professor_sources)) > 0,
        id="first_advisor_stony_brook_professor_reference",
        desc="Provide a valid URL reference supporting the advisor's professorship at Stony Brook University",
        parent=id_aff,
        critical=True,
    )

    # Professorship at Stony Brook - actual verification using URLs
    leaf_prof_sbu = evaluator.add_leaf(
        id="first_advisor_stony_brook_professor_constraint",
        desc="Verify the constraint that the president's doctoral advisor was a professor at Stony Brook University",
        parent=id_aff,
        critical=True,
    )
    claim_prof_sbu = f"{name_first} was a professor at Stony Brook University."
    await evaluator.verify(
        claim=claim_prof_sbu,
        node=leaf_prof_sbu,
        sources=_safe_urls(first.stony_brook_professor_sources),
        additional_instruction="Accept synonyms like 'faculty member' or 'professor of X at Stony Brook'. The page should explicitly indicate an academic appointment at Stony Brook University.",
        extra_prerequisites=[ref_presence_prof_sbu],
    )

    # First advisor PhD background (sequential)
    edu_seq = evaluator.add_sequential(
        id="first_advisor_educational_background",
        desc="Provide and verify the PhD credentials of the first advisor",
        parent=seq_node,
        critical=True,
    )

    edu_details = evaluator.add_parallel(
        id="first_advisor_phd_details",
        desc="Provide complete PhD information for the first advisor",
        parent=edu_seq,
        critical=True,
    )

    # Institution equals UW
    leaf_fa_inst = evaluator.add_leaf(
        id="first_advisor_phd_institution",
        desc="Identify the institution that awarded the PhD to the first advisor (per constraints: University of Washington)",
        parent=edu_details,
        critical=True,
    )
    fa_inst = first.phd_institution or ""
    claim_fa_inst = (
        f"The first advisor's PhD institution stated in the answer is '{fa_inst}', and this equals '{GT_FIRST_ADVISOR_PHD_INSTITUTION}'."
    )
    await evaluator.verify(
        claim=claim_fa_inst,
        node=leaf_fa_inst,
        additional_instruction="Allow minor formatting differences; evaluate institutional equivalence.",
    )

    # Year equals 1960
    leaf_fa_year = evaluator.add_leaf(
        id="first_advisor_phd_year",
        desc="Identify the year the first advisor's PhD was awarded (per constraints: 1960)",
        parent=edu_details,
        critical=True,
    )
    fa_year = first.phd_year or ""
    claim_fa_year = (
        f"The first advisor's PhD year stated in the answer is '{fa_year}', and this equals {GT_FIRST_ADVISOR_PHD_YEAR}."
    )
    await evaluator.verify(
        claim=claim_fa_year,
        node=leaf_fa_year,
        additional_instruction="Allow minor formatting differences; evaluate whether the numeric year equals 1960.",
    )

    # Field equals Electrical Engineering
    leaf_fa_field = evaluator.add_leaf(
        id="first_advisor_phd_field",
        desc="Identify the field of study for the first advisor's PhD (per constraints: Electrical Engineering)",
        parent=edu_details,
        critical=True,
    )
    fa_field = first.phd_field or ""
    claim_fa_field = (
        f"The first advisor's PhD field stated in the answer is '{fa_field}', and this equals '{GT_FIRST_ADVISOR_PHD_FIELD}'."
    )
    await evaluator.verify(
        claim=claim_fa_field,
        node=leaf_fa_field,
        additional_instruction="Allow minor paraphrases like 'Electrical Eng.' to be considered equivalent.",
    )

    # Reference supporting first advisor PhD details
    leaf_fa_ref = evaluator.add_leaf(
        id="first_advisor_phd_reference",
        desc="Provide a valid URL reference supporting the first advisor's PhD institution, year, and field",
        parent=edu_seq,
        critical=True,
    )
    name_first_for_claim = first.name or "the first advisor named in the answer"
    claim_fa_ref = (
        f"According to the provided source, {name_first_for_claim} earned a PhD in Electrical Engineering "
        f"from the University of Washington in 1960."
    )
    await evaluator.verify(
        claim=claim_fa_ref,
        node=leaf_fa_ref,
        sources=_safe_urls(first.phd_sources),
        additional_instruction="The page should explicitly support the institution (University of Washington), the year 1960, and the field (Electrical Engineering).",
    )


async def build_second_advisor_chain(
    evaluator: Evaluator,
    parent_node,
    extracted: GenealogyExtraction,
):
    seq_node = evaluator.add_sequential(
        id="second_advisor_chain",
        desc="Identify and verify the doctoral advisor of the first advisor (second-generation), including required affiliation constraint",
        parent=parent_node,
        critical=True,
    )
    first = extracted.first_advisor or FirstAdvisorInfo()
    second = extracted.second_advisor or SecondAdvisorInfo()

    # Identification and affiliation (parallel)
    id_aff = evaluator.add_parallel(
        id="second_advisor_identification_and_affiliation",
        desc="Identify the doctoral advisor of the first advisor and verify the required affiliation condition",
        parent=seq_node,
        critical=True,
    )

    # Name provided (existence check)
    evaluator.add_custom_node(
        result=bool(second.name and second.name.strip()),
        id="second_advisor_name",
        desc="Provide the name of the second-generation advisor (first advisor's doctoral advisor)",
        parent=id_aff,
        critical=True,
    )

    # Relationship reference (advisor relationship supported by URLs)
    leaf_rel = evaluator.add_leaf(
        id="second_advisor_relationship_reference",
        desc="Provide a valid URL reference confirming the advisor relationship (first advisor ↔ second-generation advisor)",
        parent=id_aff,
        critical=True,
    )
    name_first = first.name or "the first advisor named in the answer"
    name_second = second.name or "the second-generation advisor named in the answer"
    claim_rel = f"{name_second} was the doctoral advisor (PhD advisor/supervisor) of {name_first}."
    await evaluator.verify(
        claim=claim_rel,
        node=leaf_rel,
        sources=_safe_urls(second.relationship_sources),
        additional_instruction="The page should clearly state the doctoral advisor–advisee relationship between these two people (or equivalent phrasing).",
    )

    # Professorship at University of Washington - reference presence (as prerequisite)
    ref_presence_prof_uw = evaluator.add_custom_node(
        result=len(_safe_urls(second.uw_professor_sources)) > 0,
        id="second_advisor_uw_professor_reference",
        desc="Provide a valid URL reference supporting the second-generation advisor's professorship at the University of Washington",
        parent=id_aff,
        critical=True,
    )

    # Professorship at University of Washington - actual verification using URLs
    leaf_prof_uw = evaluator.add_leaf(
        id="second_advisor_uw_professor_constraint",
        desc="Verify the constraint that the first advisor's doctoral advisor was a professor at the University of Washington",
        parent=id_aff,
        critical=True,
    )
    claim_prof_uw = f"{name_second} was a professor at the University of Washington."
    await evaluator.verify(
        claim=claim_prof_uw,
        node=leaf_prof_uw,
        sources=_safe_urls(second.uw_professor_sources),
        additional_instruction="Accept synonyms like 'faculty member' or 'professor of X at the University of Washington'. The page should explicitly indicate an academic appointment at UW.",
        extra_prerequisites=[ref_presence_prof_uw],
    )

    # Complete education (sequential)
    edu_seq = evaluator.add_sequential(
        id="second_advisor_complete_education",
        desc="Provide and verify the complete educational background of the second-generation advisor, including historical PhD detail",
        parent=seq_node,
        critical=True,
    )

    edu_details = evaluator.add_parallel(
        id="second_advisor_education_details",
        desc="Provide complete educational information for the second-generation advisor",
        parent=edu_seq,
        critical=True,
    )

    # Bachelor's institution
    leaf_bs_inst = evaluator.add_leaf(
        id="bachelors_institution",
        desc="Identify the institution that awarded the bachelor's degree (per constraints: University of Tokyo)",
        parent=edu_details,
        critical=True,
    )
    bs_inst = second.bachelors_institution or ""
    claim_bs_inst = (
        f"The second-generation advisor's bachelor's institution stated in the answer is '{bs_inst}', and this equals '{GT_SECOND_ADVISOR_BS_INSTITUTION}'."
    )
    await evaluator.verify(
        claim=claim_bs_inst,
        node=leaf_bs_inst,
        additional_instruction="Allow minor formatting differences; evaluate institutional equivalence.",
    )

    # Bachelor's year
    leaf_bs_year = evaluator.add_leaf(
        id="bachelors_year",
        desc="Identify the year the bachelor's degree was awarded (per constraints: 1951)",
        parent=edu_details,
        critical=True,
    )
    bs_year = second.bachelors_year or ""
    claim_bs_year = (
        f"The second-generation advisor's bachelor's year stated in the answer is '{bs_year}', and this equals {GT_SECOND_ADVISOR_BS_YEAR}."
    )
    await evaluator.verify(
        claim=claim_bs_year,
        node=leaf_bs_year,
        additional_instruction="Allow minor formatting differences; evaluate whether the numeric year equals 1951.",
    )

    # PhD institution (second)
    leaf_sa_phd_inst = evaluator.add_leaf(
        id="phd_institution_second",
        desc="Identify the institution that awarded the PhD (per constraints: University of Washington)",
        parent=edu_details,
        critical=True,
    )
    sa_phd_inst = second.phd_institution or ""
    claim_sa_inst = (
        f"The second-generation advisor's PhD institution stated in the answer is '{sa_phd_inst}', and this equals '{GT_SECOND_ADVISOR_PHD_INSTITUTION}'."
    )
    await evaluator.verify(
        claim=claim_sa_inst,
        node=leaf_sa_phd_inst,
        additional_instruction="Allow minor formatting differences; evaluate institutional equivalence.",
    )

    # PhD year (second)
    leaf_sa_phd_year = evaluator.add_leaf(
        id="phd_year_second",
        desc="Identify the year the PhD was awarded (per constraints: 1958)",
        parent=edu_details,
        critical=True,
    )
    sa_phd_year = second.phd_year or ""
    claim_sa_year = (
        f"The second-generation advisor's PhD year stated in the answer is '{sa_phd_year}', and this equals {GT_SECOND_ADVISOR_PHD_YEAR}."
    )
    await evaluator.verify(
        claim=claim_sa_year,
        node=leaf_sa_phd_year,
        additional_instruction="Allow minor formatting differences; evaluate whether the numeric year equals 1958.",
    )

    # Historical significance (first PhD in UW EE)
    leaf_hist = evaluator.add_leaf(
        id="historical_significance",
        desc="Identify the historically significant detail about the PhD (per constraints: first doctoral degree awarded by the University of Washington Electrical Engineering department)",
        parent=edu_details,
        critical=True,
    )
    hist_text = second.historical_significance or ""
    claim_hist = (
        f"The historical significance statement in the answer ('{hist_text}') conveys that it was the first doctoral degree "
        f"awarded by the University of Washington Electrical Engineering department."
    )
    await evaluator.verify(
        claim=claim_hist,
        node=leaf_hist,
        additional_instruction="Allow paraphrases that unambiguously indicate the 'first doctoral degree' awarded by UW's Electrical Engineering department.",
    )

    # Reference supporting second advisor education + historical note
    leaf_sa_ref = evaluator.add_leaf(
        id="second_advisor_education_reference",
        desc="Provide a valid URL reference supporting the second-generation advisor's bachelor's institution/year, PhD institution/year, and the historical significance detail",
        parent=edu_seq,
        critical=True,
    )
    name_second_for_claim = second.name or "the second-generation advisor named in the answer"
    claim_sa_ref = (
        f"According to the provided source, {name_second_for_claim} earned a bachelor's degree from the University of Tokyo in 1951 "
        f"and a PhD from the University of Washington in 1958; the PhD was the first doctoral degree awarded by the University of Washington Electrical Engineering department."
    )
    await evaluator.verify(
        claim=claim_sa_ref,
        node=leaf_sa_ref,
        sources=_safe_urls(second.education_sources),
        additional_instruction="The page should explicitly support all three: bachelor's (institution + 1951), PhD (institution + 1958), and the 'first doctoral degree' historical note (UW EE).",
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
    Evaluate an answer for the Stanford president (2000–2016) academic genealogy task.
    """
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

    # Add ground-truth anchors for transparency
    evaluator.add_ground_truth({
        "expected_president": GT_PRESIDENT_NAME,
        "president_phd": {
            "institution_aliases": GT_PRESIDENT_PHD_INSTITUTION_ALIASES,
            "year": GT_PRESIDENT_PHD_YEAR,
            "field_aliases": GT_PRESIDENT_PHD_FIELD_ALIASES,
        },
        "first_advisor_phd": {
            "institution": GT_FIRST_ADVISOR_PHD_INSTITUTION,
            "year": GT_FIRST_ADVISOR_PHD_YEAR,
            "field": GT_FIRST_ADVISOR_PHD_FIELD,
        },
        "second_advisor_education": {
            "bachelors_institution": GT_SECOND_ADVISOR_BS_INSTITUTION,
            "bachelors_year": GT_SECOND_ADVISOR_BS_YEAR,
            "phd_institution": GT_SECOND_ADVISOR_PHD_INSTITUTION,
            "phd_year": GT_SECOND_ADVISOR_PHD_YEAR,
            "historical_significance": GT_SECOND_ADVISOR_HISTORICAL_DETAIL_KEY,
        }
    }, gt_type="ground_truth_constraints")

    # Extract structured info
    extracted: GenealogyExtraction = await evaluator.extract(
        prompt=prompt_extract_genealogy(),
        template_class=GenealogyExtraction,
        extraction_name="genealogy_extraction",
    )

    # Build and verify according to rubric tree
    await build_president_identification(evaluator, root, extracted)
    await build_president_phd_background(evaluator, root, extracted)
    await build_first_advisor_chain(evaluator, root, extracted)
    await build_second_advisor_chain(evaluator, root, extracted)

    return evaluator.get_summary()