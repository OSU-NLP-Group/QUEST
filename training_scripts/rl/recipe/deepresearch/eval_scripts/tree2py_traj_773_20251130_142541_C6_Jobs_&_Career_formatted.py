import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "power5_d1_ad_candidate_profile"
TASK_DESCRIPTION = (
    "You are consulting for a search firm hired to fill the Athletic Director position at a Power 5 conference university. "
    "Based on documented hiring patterns and career progression data of current Division I athletic directors, create a detailed candidate profile document that specifies the following five components:\n\n"
    "1. Educational Credentials: State the minimum degree requirement and address the prevalence of graduate degrees among current Division I ADs. Identify at least three relevant degree fields that are appropriate for athletic administration careers.\n\n"
    "2. Experience Threshold: Specify the minimum total years of experience required in athletic administration for Division I AD positions, distinguishing between general administrative experience and managerial/leadership experience where appropriate.\n\n"
    "3. Career Progression Pathways: Identify the three most common immediately prior position titles held by successful Division I athletic director hires, and note what percentage of all Division I AD appointments these three pathways collectively represent. Also provide context about the relevance of prior college coaching experience.\n\n"
    "4. Professional Development: Identify relevant athletic administration certifications (such as those offered by NIAAA) and describe the basic requirements for obtaining such certifications, including degree, experience, and training components.\n\n"
    "5. Administrative Expertise: Specify key functional areas where demonstrated leadership and expertise are expected (such as budget management, compliance, fundraising, or operations), and describe the types of leadership competencies typically required for Division I athletic director positions.\n\n"
    "Each specification in your candidate profile must be grounded in documented patterns and statistics from research on actual Division I athletic director hires, with explicit URL references to your data sources."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ClaimWithSources(BaseModel):
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PercentClaimWithSources(BaseModel):
    statement: Optional[str] = None
    percent_text: Optional[str] = None  # e.g., "~80%", "about 75%"
    sources: List[str] = Field(default_factory=list)


class TitlesClaim(BaseModel):
    titles: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class AreasWithSources(BaseModel):
    items: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class EducationalCredentials(BaseModel):
    minimum_degree: Optional[ClaimWithSources] = None
    grad_degree_prevalence: Optional[PercentClaimWithSources] = None
    masters_preferred: Optional[ClaimWithSources] = None
    relevant_degree_fields: Optional[AreasWithSources] = None


class ExperienceThreshold(BaseModel):
    min_admin_experience: Optional[ClaimWithSources] = None
    high_end_threshold: Optional[ClaimWithSources] = None
    distinguishes_general_vs_managerial: Optional[ClaimWithSources] = None


class CareerProgressionPathways(BaseModel):
    top3_prior_titles: Optional[TitlesClaim] = None
    top3_combined_percent: Optional[PercentClaimWithSources] = None
    coaching_background_percent: Optional[PercentClaimWithSources] = None


class ProfessionalDevelopment(BaseModel):
    niaaa_caa: Optional[ClaimWithSources] = None
    caa_requirements: Optional[ClaimWithSources] = None


class AdministrativeExpertise(BaseModel):
    key_functional_areas: Optional[AreasWithSources] = None
    leadership_competencies: Optional[ClaimWithSources] = None


class CandidateProfileExtraction(BaseModel):
    educational_credentials: Optional[EducationalCredentials] = None
    experience_threshold: Optional[ExperienceThreshold] = None
    career_progression_pathways: Optional[CareerProgressionPathways] = None
    professional_development: Optional[ProfessionalDevelopment] = None
    administrative_expertise: Optional[AdministrativeExpertise] = None
    all_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_candidate_profile() -> str:
    return """
    Extract a structured candidate profile for a Division I Athletic Director (Power-5 context) from the answer. 
    For each component, extract the precise statement as written in the answer and the explicit URL sources cited in the answer text that support that statement.

    Return a JSON object with the following structure:

    {
      "educational_credentials": {
        "minimum_degree": { "statement": str|null, "sources": [urls...] },
        "grad_degree_prevalence": { "statement": str|null, "percent_text": str|null, "sources": [urls...] },
        "masters_preferred": { "statement": str|null, "sources": [urls...] },
        "relevant_degree_fields": { "items": [strings...], "sources": [urls...] }
      },
      "experience_threshold": {
        "min_admin_experience": { "statement": str|null, "sources": [urls...] },
        "high_end_threshold": { "statement": str|null, "sources": [urls...] },
        "distinguishes_general_vs_managerial": { "statement": str|null, "sources": [urls...] }
      },
      "career_progression_pathways": {
        "top3_prior_titles": { "titles": [strings...], "sources": [urls...] },
        "top3_combined_percent": { "statement": str|null, "percent_text": str|null, "sources": [urls...] },
        "coaching_background_percent": { "statement": str|null, "percent_text": str|null, "sources": [urls...] }
      },
      "professional_development": {
        "niaaa_caa": { "statement": str|null, "sources": [urls...] },
        "caa_requirements": { "statement": str|null, "sources": [urls...] }
      },
      "administrative_expertise": {
        "key_functional_areas": { "items": [strings...], "sources": [urls...] },
        "leadership_competencies": { "statement": str|null, "sources": [urls...] }
      },
      "all_sources": [urls...]
    }

    Rules:
    - "statement" must be copied exactly (verbatim or close paraphrase) from the answer’s text for the corresponding component.
    - For "percent_text", extract the percentage as written (e.g., "~80%", "approximately 75%").
    - For "titles" and "items" lists, include exactly the titles/areas enumerated in the answer.
    - "sources" must be actual URLs explicitly present in the answer for that specific statement. 
      If the answer cites sources in markdown form, extract the underlying URL. If no explicit URL is provided, return an empty list.
    - "all_sources" must include every URL present anywhere in the answer (deduplicated).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _list_has_items(items: Optional[List[str]], min_count: int) -> bool:
    return bool(items) and len([x for x in items if x and x.strip()]) >= min_count


def _has_sources(sources: Optional[List[str]]) -> bool:
    return bool(sources) and len(sources) > 0


def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _areas_cover_required(items: Optional[List[str]]) -> bool:
    """Check list contains budgeting/finance, compliance, fundraising/development, operations (allow synonyms)."""
    if not items:
        return False
    normalized = {_normalize_text(x) for x in items if x}
    # Synonym sets
    req_groups = [
        {"budget", "budgeting", "finance", "financial management", "budget management", "fiscal"},
        {"compliance", "ncaa compliance", "regulatory compliance"},
        {"fundraising", "development", "advancement", "donor relations", "revenue development"},
        {"operations", "athletics operations", "event operations", "game operations", "facility operations"}
    ]
    def group_present(group: set) -> bool:
        for g in group:
            for item in normalized:
                if g in item:
                    return True
        return False
    return all(group_present(g) for g in req_groups)


def _titles_include_top3(titles: Optional[List[str]]) -> bool:
    """Check presence of deputy AD, senior associate AD, and (sitting) AD."""
    if not titles:
        return False
    norm = {_normalize_text(t) for t in titles if t}
    def contains_any(subs: List[str]) -> bool:
        for s in subs:
            for t in norm:
                if s in t:
                    return True
        return False
    dep_variants = ["deputy athletic director", "deputy ad", "deputy director of athletics"]
    sr_assoc_variants = ["senior associate athletic director", "senior assoc ad", "senior associate ad"]
    ad_variants = ["athletic director", "director of athletics"]
    return contains_any(dep_variants) and contains_any(sr_assoc_variants) and contains_any(ad_variants)


# --------------------------------------------------------------------------- #
# Verification functions per section                                          #
# --------------------------------------------------------------------------- #
async def verify_educational_credentials(
    evaluator: Evaluator,
    parent_node,
    ec: Optional[EducationalCredentials]
) -> None:
    sec = evaluator.add_parallel(
        id="Educational_Credentials",
        desc="Educational credential requirements and degree-field fit for Division I ADs.",
        parent=parent_node,
        critical=True
    )

    # Minimum degree = bachelor's
    group1 = evaluator.add_sequential(
        id="Minimum_Degree_Is_Bachelors",
        desc="States that the minimum degree requirement is a bachelor's degree.",
        parent=sec,
        critical=True
    )
    exists1 = evaluator.add_custom_node(
        result=(ec is not None and ec.minimum_degree is not None and bool(ec.minimum_degree.statement)),
        id="Min_Degree_Statement_Present",
        desc="Minimum degree statement is present in the answer",
        parent=group1,
        critical=True
    )
    node1_text = evaluator.add_leaf(
        id="Min_Degree_Is_Bachelors_Text_Match",
        desc="Answer explicitly states minimum degree is bachelor's",
        parent=group1,
        critical=True
    )
    await evaluator.verify(
        claim="The candidate profile explicitly states that the minimum degree requirement is a bachelor's degree.",
        node=node1_text
    )
    node1_src = evaluator.add_leaf(
        id="Min_Degree_Bachelors_Source_Support",
        desc="Minimum degree (bachelor's) is supported by cited sources",
        parent=group1,
        critical=True
    )
    await evaluator.verify(
        claim="Division I athletic director job qualifications specify at least a bachelor's degree as the minimum requirement.",
        node=node1_src,
        sources=ec.minimum_degree.sources if (ec and ec.minimum_degree) else [],
        additional_instruction="Verify the job standards or hiring patterns indicating bachelor's as the minimum credential."
    )

    # Graduate degree prevalence ~80%
    group2 = evaluator.add_sequential(
        id="Graduate_Degree_Prevalence_80_Percent",
        desc="States that ~80% of current Division I athletic directors hold a graduate degree (master's or higher).",
        parent=sec,
        critical=True
    )
    exists2 = evaluator.add_custom_node(
        result=(ec is not None and ec.grad_degree_prevalence is not None and bool(ec.grad_degree_prevalence.statement) and _has_sources(ec.grad_degree_prevalence.sources)),
        id="Grad_Degree_Prevalence_Claim_And_Sources_Present",
        desc="Graduate degree prevalence statement and sources are present",
        parent=group2,
        critical=True
    )
    node2_val = evaluator.add_leaf(
        id="Grad_Degree_Prevalence_Approx80_Text_Check",
        desc="Reported prevalence is approximately 80%",
        parent=group2,
        critical=True
    )
    pct_txt = ec.grad_degree_prevalence.percent_text if (ec and ec.grad_degree_prevalence) else None
    await evaluator.verify(
        claim=f"The reported prevalence '{pct_txt or ''}' is approximately 80% (±5 percentage points acceptable).",
        node=node2_val,
        additional_instruction="Use the answer text to judge whether the stated value is close to 80%."
    )
    node2_src = evaluator.add_leaf(
        id="Grad_Degree_Prevalence_80_Source_Support",
        desc="~80% graduate degrees among D1 ADs supported by sources",
        parent=group2,
        critical=True
    )
    await evaluator.verify(
        claim="Approximately 80% of current Division I athletic directors hold a graduate degree (master’s or higher).",
        node=node2_src,
        sources=ec.grad_degree_prevalence.sources if (ec and ec.grad_degree_prevalence) else [],
        additional_instruction="Check whether the cited research/statistics support ~80% having graduate degrees."
    )

    # Master's preferred or required for competitive D1
    group3 = evaluator.add_sequential(
        id="Masters_Preferred_For_Competitive_D1",
        desc="States that for competitive Division I positions, a master's degree is typically preferred or required.",
        parent=sec,
        critical=True
    )
    exists3 = evaluator.add_custom_node(
        result=(ec is not None and ec.masters_preferred is not None and bool(ec.masters_preferred.statement) and _has_sources(ec.masters_preferred.sources)),
        id="Masters_Pref_Req_Statement_And_Sources_Present",
        desc="Master's preferred/required statement and sources are present",
        parent=group3,
        critical=True
    )
    node3_src = evaluator.add_leaf(
        id="Masters_Pref_Req_Source_Support",
        desc="Master's degree preference/requirement supported by sources",
        parent=group3,
        critical=True
    )
    await evaluator.verify(
        claim="For competitive Division I athletic director positions, a master's degree is typically preferred or required.",
        node=node3_src,
        sources=ec.masters_preferred.sources if (ec and ec.masters_preferred) else [],
        additional_instruction="Verify from job profiles or hiring patterns that a master’s is commonly preferred/required."
    )

    # Lists at least three relevant degree fields
    group4 = evaluator.add_sequential(
        id="Lists_At_Least_Three_Relevant_Degree_Fields",
        desc="Lists at least three relevant degree fields (sports management, physical education, business administration, education administration, or related).",
        parent=sec,
        critical=True
    )
    exists4 = evaluator.add_custom_node(
        result=(ec is not None and ec.relevant_degree_fields is not None and _list_has_items(ec.relevant_degree_fields.items, 3)),
        id="Relevant_Degree_Fields_Count_Check",
        desc="At least three relevant degree fields are listed",
        parent=group4,
        critical=True
    )
    node4_src = evaluator.add_leaf(
        id="Relevant_Degree_Fields_Source_Support",
        desc="Relevant degree fields are supported by sources",
        parent=group4,
        critical=True
    )
    fields_txt = ", ".join(ec.relevant_degree_fields.items) if (ec and ec.relevant_degree_fields and ec.relevant_degree_fields.items) else ""
    await evaluator.verify(
        claim=f"The following degree fields are commonly relevant for athletic administration careers: {fields_txt}.",
        node=node4_src,
        sources=ec.relevant_degree_fields.sources if (ec and ec.relevant_degree_fields) else [],
        additional_instruction="Check whether sources recognize these fields (e.g., sports management, PE, business, education admin) as appropriate for AD careers."
    )


async def verify_experience_threshold(
    evaluator: Evaluator,
    parent_node,
    et: Optional[ExperienceThreshold]
) -> None:
    sec = evaluator.add_parallel(
        id="Experience_Threshold",
        desc="Minimum experience thresholds for Division I AD candidacy and leadership/management expectations.",
        parent=parent_node,
        critical=True
    )

    # Minimum >=5 years admin experience
    g1 = evaluator.add_sequential(
        id="Minimum_D1_Admin_Experience_At_Least_5_Years",
        desc="Specifies that major university/Division I positions typically require a minimum of 5 years of athletics administration experience.",
        parent=sec,
        critical=True
    )
    exists1 = evaluator.add_custom_node(
        result=(et is not None and et.min_admin_experience is not None and bool(et.min_admin_experience.statement) and _has_sources(et.min_admin_experience.sources)),
        id="Min_5yrs_Stmt_And_Sources_Present",
        desc="Minimum 5+ years statement and sources present",
        parent=g1,
        critical=True
    )
    node1_text = evaluator.add_leaf(
        id="Min_5yrs_Text_Check",
        desc="Answer states minimum ≥5 years athletics admin experience",
        parent=g1,
        critical=True
    )
    await evaluator.verify(
        claim="The candidate profile states that Division I positions typically require at least 5 years of athletics administration experience.",
        node=node1_text
    )
    node1_src = evaluator.add_leaf(
        id="Min_5yrs_Source_Support",
        desc="Minimum 5+ years requirement supported by sources",
        parent=g1,
        critical=True
    )
    await evaluator.verify(
        claim="Division I athletic director positions typically require a minimum of 5 years of athletics administration experience.",
        node=node1_src,
        sources=et.min_admin_experience.sources if (et and et.min_admin_experience) else [],
        additional_instruction="Confirm job requirements or hiring data indicating ≥5 years admin experience."
    )

    # Higher end: 10+ total, ~6+ leadership
    g2 = evaluator.add_sequential(
        id="Higher_End_Threshold_10plus_Total_6plus_Leadership",
        desc="Includes the higher-end threshold that some roles require 10+ years total experience including ~6 years in managerial/leadership roles.",
        parent=sec,
        critical=True
    )
    exists2 = evaluator.add_custom_node(
        result=(et is not None and et.high_end_threshold is not None and bool(et.high_end_threshold.statement) and _has_sources(et.high_end_threshold.sources)),
        id="HighEnd_Threshold_Stmt_And_Sources_Present",
        desc="Higher-end threshold statement and sources present",
        parent=g2,
        critical=True
    )
    node2_text = evaluator.add_leaf(
        id="HighEnd_Threshold_Text_Check",
        desc="Answer specifies 10+ total and ~6+ leadership years",
        parent=g2,
        critical=True
    )
    await evaluator.verify(
        claim="The candidate profile specifies a higher-end threshold of 10+ total years including approximately 6 years in managerial/leadership roles.",
        node=node2_text
    )
    node2_src = evaluator.add_leaf(
        id="HighEnd_Threshold_Source_Support",
        desc="10+ total and ~6+ leadership years supported by sources",
        parent=g2,
        critical=True
    )
    await evaluator.verify(
        claim="Some executive-level AD roles require 10+ total years of experience including around 6 years in managerial/leadership roles.",
        node=node2_src,
        sources=et.high_end_threshold.sources if (et and et.high_end_threshold) else [],
        additional_instruction="Verify whether cited job standards or career analyses indicate these higher-end thresholds."
    )

    # Distinguishes general vs managerial experience
    g3 = evaluator.add_sequential(
        id="Distinguishes_General_vs_Managerial_Experience",
        desc="Clearly distinguishes general athletic administration experience vs managerial/leadership experience.",
        parent=sec,
        critical=True
    )
    exists3 = evaluator.add_custom_node(
        result=(et is not None and et.distinguishes_general_vs_managerial is not None and bool(et.distinguishes_general_vs_managerial.statement) and _has_sources(et.distinguishes_general_vs_managerial.sources)),
        id="Distinction_Stmt_And_Sources_Present",
        desc="Distinction statement and sources present",
        parent=g3,
        critical=True
    )
    node3_src = evaluator.add_leaf(
        id="Distinction_Source_Support",
        desc="Distinction between general and managerial experience supported by sources",
        parent=g3,
        critical=True
    )
    await evaluator.verify(
        claim="Division I AD hiring distinguishes general athletics administration experience from managerial/leadership/supervisory experience.",
        node=node3_src,
        sources=et.distinguishes_general_vs_managerial.sources if (et and et.distinguishes_general_vs_managerial) else [],
        additional_instruction="Check that sources explicitly separate general admin tenure from leadership/managerial experience."
    )


async def verify_career_progression_pathways(
    evaluator: Evaluator,
    parent_node,
    cp: Optional[CareerProgressionPathways]
) -> None:
    sec = evaluator.add_parallel(
        id="Career_Progression_Pathways",
        desc="Common immediately prior roles for Division I AD hires, combined share, and coaching-background context.",
        parent=parent_node,
        critical=True
    )

    # Top3 immediate prior titles specified (Deputy AD, Sitting AD from another institution, Senior Associate AD)
    g1 = evaluator.add_sequential(
        id="Top3_Immediate_Prior_Titles_Specified",
        desc="Identifies the three most common immediately prior titles: Deputy AD, Sitting AD (another institution), Senior Associate AD.",
        parent=sec,
        critical=True
    )
    exists1 = evaluator.add_custom_node(
        result=(cp is not None and cp.top3_prior_titles is not None and _titles_include_top3(cp.top3_prior_titles.titles) and _has_sources(cp.top3_prior_titles.sources)),
        id="Top3_Titles_List_And_Sources_Present",
        desc="Top three titles appear in the answer and sources are provided",
        parent=g1,
        critical=True
    )
    node1_src = evaluator.add_leaf(
        id="Top3_Titles_Source_Support",
        desc="Top three prior titles supported by sources",
        parent=g1,
        critical=True
    )
    await evaluator.verify(
        claim="The three most common immediately prior titles for Division I AD hires are deputy athletic director, sitting athletic director at another institution, and senior associate athletic director.",
        node=node1_src,
        sources=cp.top3_prior_titles.sources if (cp and cp.top3_prior_titles) else [],
        additional_instruction="Verify that the cited research/statistics identify these three roles as the most common feeder positions."
    )

    # Top3 pathways collectively ~75%
    g2 = evaluator.add_sequential(
        id="Top3_Pathways_Combined_75_Percent",
        desc="States that these three pathways collectively represent ~75% of all Division I AD appointments.",
        parent=sec,
        critical=True
    )
    exists2 = evaluator.add_custom_node(
        result=(cp is not None and cp.top3_combined_percent is not None and bool(cp.top3_combined_percent.statement) and _has_sources(cp.top3_combined_percent.sources)),
        id="Top3_CombinedPct_Stmt_And_Sources_Present",
        desc="Combined ~75% statement and sources present",
        parent=g2,
        critical=True
    )
    node2_val = evaluator.add_leaf(
        id="Top3_CombinedPct_Approx75_Text_Check",
        desc="Reported combined share is approximately 75%",
        parent=g2,
        critical=True
    )
    pct_txt2 = cp.top3_combined_percent.percent_text if (cp and cp.top3_combined_percent) else None
    await evaluator.verify(
        claim=f"The reported combined share '{pct_txt2 or ''}' is approximately 75% (±5 percentage points acceptable).",
        node=node2_val
    )
    node2_src = evaluator.add_leaf(
        id="Top3_CombinedPct_Source_Support",
        desc="~75% combined share supported by sources",
        parent=g2,
        critical=True
    )
    await evaluator.verify(
        claim="These three pathways collectively represent approximately 75% of all Division I AD appointments.",
        node=node2_src,
        sources=cp.top3_combined_percent.sources if (cp and cp.top3_combined_percent) else [],
        additional_instruction="Verify the proportion across appointments from the cited research/statistics."
    )

    # Coaching background ~29% with context
    g3 = evaluator.add_sequential(
        id="Coaching_Background_29_Percent_Context",
        desc="States that only ~29% of current Division I ADs were former college coaches and interprets the implication.",
        parent=sec,
        critical=True
    )
    exists3 = evaluator.add_custom_node(
        result=(cp is not None and cp.coaching_background_percent is not None and bool(cp.coaching_background_percent.statement) and _has_sources(cp.coaching_background_percent.sources)),
        id="CoachingPct_Stmt_And_Sources_Present",
        desc="~29% coaching background statement and sources present",
        parent=g3,
        critical=True
    )
    node3_val = evaluator.add_leaf(
        id="CoachingPct_Approx29_Text_Check",
        desc="Reported coaching-background share is approximately 29%",
        parent=g3,
        critical=True
    )
    pct_txt3 = cp.coaching_background_percent.percent_text if (cp and cp.coaching_background_percent) else None
    await evaluator.verify(
        claim=f"The reported share '{pct_txt3 or ''}' is approximately 29% (±5 percentage points acceptable).",
        node=node3_val
    )
    node3_src = evaluator.add_leaf(
        id="CoachingPct_Source_Support",
        desc="~29% former college coaches supported by sources",
        parent=g3,
        critical=True
    )
    await evaluator.verify(
        claim="Only about 29% of current Division I athletic directors were former college coaches.",
        node=node3_src,
        sources=cp.coaching_background_percent.sources if (cp and cp.coaching_background_percent) else [],
        additional_instruction="Verify the proportion and ensure the implication is that administrative pathways are more common."
    )


async def verify_professional_development(
    evaluator: Evaluator,
    parent_node,
    pd: Optional[ProfessionalDevelopment]
) -> None:
    sec = evaluator.add_parallel(
        id="Professional_Development",
        desc="Relevant certifications and their basic requirements.",
        parent=parent_node,
        critical=True
    )

    # Identifies NIAAA CAA
    g1 = evaluator.add_sequential(
        id="Identifies_NIAAA_CAA",
        desc="Identifies the NIAAA Certified Athletic Administrator (CAA) credential as relevant.",
        parent=sec,
        critical=True
    )
    exists1 = evaluator.add_custom_node(
        result=(pd is not None and pd.niaaa_caa is not None and bool(pd.niaaa_caa.statement) and _has_sources(pd.niaaa_caa.sources)),
        id="NIAAA_CAA_Stmt_And_Sources_Present",
        desc="NIAAA CAA statement and sources present",
        parent=g1,
        critical=True
    )
    node1_src = evaluator.add_leaf(
        id="NIAAA_CAA_Source_Support",
        desc="NIAAA CAA relevance supported by sources",
        parent=g1,
        critical=True
    )
    await evaluator.verify(
        claim="The NIAAA Certified Athletic Administrator (CAA) is a relevant athletic administration certification.",
        node=node1_src,
        sources=pd.niaaa_caa.sources if (pd and pd.niaaa_caa) else [],
        additional_instruction="Confirm from NIAAA or recognized authorities that CAA is a relevant credential for athletics administration."
    )

    # CAA requirements specified: bachelor's, 2+ years, LTC 501/502/503/504/506
    g2 = evaluator.add_sequential(
        id="CAA_Requirements_Specified",
        desc="Describes CAA requirements including bachelor's degree, 2+ years experience, LTC 501/502/503/504/506.",
        parent=sec,
        critical=True
    )
    exists2 = evaluator.add_custom_node(
        result=(pd is not None and pd.caa_requirements is not None and bool(pd.caa_requirements.statement) and _has_sources(pd.caa_requirements.sources)),
        id="CAA_Req_Stmt_And_Sources_Present",
        desc="CAA requirements statement and sources present",
        parent=g2,
        critical=True
    )
    node2_text = evaluator.add_leaf(
        id="CAA_Req_Text_Check",
        desc="Answer lists degree, experience, and specific LTC courses for CAA",
        parent=g2,
        critical=True
    )
    await evaluator.verify(
        claim="The profile lists that CAA requires a bachelor's degree, at least two years of experience, and completion of NIAAA Leadership Training Courses 501, 502, 503, 504, and 506.",
        node=node2_text
    )
    node2_src = evaluator.add_leaf(
        id="CAA_Req_Source_Support",
        desc="CAA requirements supported by sources",
        parent=g2,
        critical=True
    )
    await evaluator.verify(
        claim="NIAAA CAA requires a bachelor's degree, two or more years of experience, and completion of courses 501, 502, 503, 504, and 506.",
        node=node2_src,
        sources=pd.caa_requirements.sources if (pd and pd.caa_requirements) else [],
        additional_instruction="Verify from official NIAAA documentation the degree, experience, and course requirements."
    )


async def verify_administrative_expertise(
    evaluator: Evaluator,
    parent_node,
    ae: Optional[AdministrativeExpertise]
) -> None:
    sec = evaluator.add_parallel(
        id="Administrative_Expertise",
        desc="Functional leadership areas and leadership competencies expected of Division I ADs.",
        parent=parent_node,
        critical=True
    )

    # Key functional areas specified (budget/finance, compliance, fundraising/development, operations)
    g1 = evaluator.add_sequential(
        id="Key_Functional_Areas_Specified",
        desc="Specifies key functional leadership areas (budgeting/finance, compliance, fundraising/development, operations).",
        parent=sec,
        critical=True
    )
    exists1 = evaluator.add_custom_node(
        result=(ae is not None and ae.key_functional_areas is not None and _areas_cover_required(ae.key_functional_areas.items)),
        id="Functional_Areas_Contain_Required_Set",
        desc="Listed functional areas cover the required set",
        parent=g1,
        critical=True
    )
    node1_src = evaluator.add_leaf(
        id="Functional_Areas_Source_Support",
        desc="Functional areas are supported by sources",
        parent=g1,
        critical=True
    )
    areas_txt = ", ".join(ae.key_functional_areas.items) if (ae and ae.key_functional_areas and ae.key_functional_areas.items) else ""
    await evaluator.verify(
        claim=f"Key functional areas for Division I AD roles include budgeting/finance, NCAA compliance, fundraising/development, and athletics operations (as reflected in: {areas_txt}).",
        node=node1_src,
        sources=ae.key_functional_areas.sources if (ae and ae.key_functional_areas) else [],
        additional_instruction="Verify sources that enumerate these areas as core leadership responsibilities for ADs."
    )

    # Leadership competencies described
    g2 = evaluator.add_sequential(
        id="Leadership_Competencies_Described",
        desc="Describes typical leadership competencies required for Division I athletic director positions.",
        parent=sec,
        critical=True
    )
    exists2 = evaluator.add_custom_node(
        result=(ae is not None and ae.leadership_competencies is not None and bool(ae.leadership_competencies.statement) and _has_sources(ae.leadership_competencies.sources)),
        id="Leadership_Competencies_Stmt_And_Sources_Present",
        desc="Leadership competencies statement and sources present",
        parent=g2,
        critical=True
    )
    node2_src = evaluator.add_leaf(
        id="Leadership_Competencies_Source_Support",
        desc="Leadership competencies supported by sources",
        parent=g2,
        critical=True
    )
    await evaluator.verify(
        claim="Typical Division I AD leadership competencies include strategic planning, stakeholder engagement, communication, decision-making, staff supervision, and ethical governance.",
        node=node2_src,
        sources=ae.leadership_competencies.sources if (ae and ae.leadership_competencies) else [],
        additional_instruction="Verify whether sources articulate these competencies for AD hires."
    )


def _section_has_any_sources(ec: Optional[EducationalCredentials],
                             et: Optional[ExperienceThreshold],
                             cp: Optional[CareerProgressionPathways],
                             pd: Optional[ProfessionalDevelopment],
                             ae: Optional[AdministrativeExpertise]) -> Dict[str, bool]:
    def any_sources_in(lst: List[Optional[ClaimWithSources]]) -> bool:
        for x in lst:
            if x and _has_sources(x.sources):
                return True
        return False

    def any_sources_in_percent(lst: List[Optional[PercentClaimWithSources]]) -> bool:
        for x in lst:
            if x and _has_sources(x.sources):
                return True
        return False

    def any_sources_in_titles(x: Optional[TitlesClaim]) -> bool:
        return bool(x and _has_sources(x.sources))

    def any_sources_in_areas(x: Optional[AreasWithSources]) -> bool:
        return bool(x and _has_sources(x.sources))

    return {
        "Educational_Credentials": (
            any_sources_in([ec.minimum_degree if ec else None,
                            ec.masters_preferred if ec else None]) or
            any_sources_in_percent([ec.grad_degree_prevalence if ec else None]) or
            any_sources_in_areas(ec.relevant_degree_fields if ec else None)
        ),
        "Experience_Threshold": (
            any_sources_in([et.min_admin_experience if et else None,
                            et.high_end_threshold if et else None,
                            et.distinguishes_general_vs_managerial if et else None])
        ),
        "Career_Progression_Pathways": (
            any_sources_in_titles(cp.top3_prior_titles if cp else None) or
            any_sources_in_percent([cp.top3_combined_percent if cp else None,
                                    cp.coaching_background_percent if cp else None])
        ),
        "Professional_Development": (
            any_sources_in([pd.niaaa_caa if pd else None,
                            pd.caa_requirements if pd else None])
        ),
        "Administrative_Expertise": (
            any_sources_in_areas(ae.key_functional_areas if ae else None) or
            any_sources_in([ae.leadership_competencies if ae else None])
        ),
    }


async def verify_source_citations(
    evaluator: Evaluator,
    parent_node,
    extracted: CandidateProfileExtraction
) -> None:
    node = evaluator.add_sequential(
        id="Source_Citations",
        desc="Provides explicit URL references to sources supporting the key statistics and factual claims across all five components.",
        parent=parent_node,
        critical=True
    )
    section_sources = _section_has_any_sources(
        extracted.educational_credentials,
        extracted.experience_threshold,
        extracted.career_progression_pathways,
        extracted.professional_development,
        extracted.administrative_expertise
    )
    all_components_have_sources = all(section_sources.values())
    evaluator.add_custom_node(
        result=all_components_have_sources,
        id="All_Five_Components_Have_Citations",
        desc="Each of the five components includes at least one explicit URL source",
        parent=node,
        critical=True
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
    Evaluate a candidate profile document for a Power-5 / Division I Athletic Director search
    against the rubric and source-supported verification.
    """
    # Initialize evaluator (root node is non-critical by default)
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

    # Create a critical top-level node for the Candidate Profile Document
    doc_root = evaluator.add_parallel(
        id="Candidate_Profile_Document",
        desc="Produces a candidate profile for a Power-5 / Division I Athletic Director search that includes all required components and is grounded in documented patterns/statistics with explicit URL references.",
        parent=root,
        critical=True
    )

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_candidate_profile(),
        template_class=CandidateProfileExtraction,
        extraction_name="candidate_profile_extraction"
    )

    # Add custom info about expected numeric anchors
    evaluator.add_custom_info(
        info={"expected_prevalence_grad": "~80%", "expected_top3_share": "~75%", "expected_coaching_background": "~29%"},
        info_type="expected_values",
        info_name="numeric_expectations"
    )

    # Build verification tree per rubric
    await verify_educational_credentials(evaluator, doc_root, extracted.educational_credentials)
    await verify_experience_threshold(evaluator, doc_root, extracted.experience_threshold)
    await verify_career_progression_pathways(evaluator, doc_root, extracted.career_progression_pathways)
    await verify_professional_development(evaluator, doc_root, extracted.professional_development)
    await verify_administrative_expertise(evaluator, doc_root, extracted.administrative_expertise)
    await verify_source_citations(evaluator, doc_root, extracted)

    # Return structured summary
    return evaluator.get_summary()