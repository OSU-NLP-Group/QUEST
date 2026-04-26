import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "psu_bigten_headcoach_pathway"
TASK_DESCRIPTION = (
    "Based on Penn State's stated job requirements and current NCAA Division I standards, "
    "document the complete qualification pathway for an assistant basketball coach to become eligible "
    "for a Big Ten Conference head coaching position. Your answer must include: "
    "(1) Penn State's minimum educational requirement for their head basketball coach position, "
    "(2) Penn State's minimum Division I coaching experience requirement, "
    "(3) the NCAA certification test requirements including passing score, "
    "(4) the average total years of coaching experience and average years of Division I experience for first-time Division I head coaches, and "
    "(5) the percentage of college athletic directors who hold master's degrees and the most common degree field. "
    "Provide reference documentation for each requirement."
)

# Ground-truth expectations from rubric (used for simple checks that the answer claims the right values)
EXPECTED = {
    "psu_min_education": "bachelor's degree or higher",
    "psu_min_di_years": 8,
    "ncaa_test_questions": 30,
    "ncaa_passing_pct": 80,  # percent
    "ncaa_passing_frac": "24/30",
    "ncaa_period": "August 1 through July 31",
    "benchmark_avg_age": 42.6,
    "benchmark_total_exp": 15.6,
    "benchmark_di_exp": 9.8,
    "ad_masters_pct_overall": "80%",
    "ad_adv_deg_recent_hires": "nearly 90%",
    "ad_common_field": "sports administration",
    "ad_common_field_count": 92,
    "ad_common_field_denominator": 231,
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PathwayPresentation(BaseModel):
    has_stepwise_pathway: Optional[bool] = None
    steps_count: Optional[int] = None
    pathway_excerpt: Optional[str] = None


class PennStateRequirements(BaseModel):
    education_requirement_text: Optional[str] = None
    education_requirement_urls: List[str] = Field(default_factory=list)
    di_experience_requirement_text: Optional[str] = None
    di_experience_requirement_urls: List[str] = Field(default_factory=list)
    recruiting_ability_text: Optional[str] = None
    recruiting_ability_urls: List[str] = Field(default_factory=list)
    compliance_text: Optional[str] = None
    compliance_urls: List[str] = Field(default_factory=list)


class NCAACertification(BaseModel):
    annual_requirement_text: Optional[str] = None
    annual_requirement_urls: List[str] = Field(default_factory=list)
    test_specs_text: Optional[str] = None  # e.g., "30 questions"
    passing_score_text: Optional[str] = None  # e.g., "80%" or "24 of 30"
    specs_urls: List[str] = Field(default_factory=list)
    certification_period_text: Optional[str] = None  # e.g., "August 1 through July 31"
    certification_period_urls: List[str] = Field(default_factory=list)


class ExperienceBenchmarks(BaseModel):
    avg_age_text: Optional[str] = None  # e.g., "42.6"
    avg_total_experience_years_text: Optional[str] = None  # e.g., "15.6"
    age_exp_urls: List[str] = Field(default_factory=list)
    avg_di_experience_years_text: Optional[str] = None  # e.g., "9.8"
    di_exp_urls: List[str] = Field(default_factory=list)


class ADStandards(BaseModel):
    master_degree_pct_text: Optional[str] = None  # e.g., "80%"
    master_degree_pct_urls: List[str] = Field(default_factory=list)
    advanced_degree_pct_recent_hires_text: Optional[str] = None  # e.g., "nearly 90%"
    advanced_degree_recent_hires_urls: List[str] = Field(default_factory=list)
    most_common_master_field_text: Optional[str] = None  # e.g., "sports administration"
    most_common_master_field_count_text: Optional[str] = None  # e.g., "92 of 231"
    most_common_master_field_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_pathway_presentation() -> str:
    return """
    Determine whether the answer presents a stepwise qualification pathway (i.e., a clear sequence of ordered steps or stages)
    from assistant coach to Big Ten head-coach eligibility, rather than an unstructured or miscellaneous list.

    Extract the following:
    - has_stepwise_pathway: true if the answer clearly presents a step-by-step pathway (numbered steps, stages, or ordered progression).
    - steps_count: the number of distinct steps, if discernible; otherwise null.
    - pathway_excerpt: a short excerpt (1-3 lines) that best demonstrates the stepwise structure; otherwise null.
    """


def prompt_extract_penn_state_requirements() -> str:
    return """
    Extract Penn State's stated minimum requirements/expectations for the HEAD MEN'S BASKETBALL COACH role as presented in the answer, with source URLs explicitly cited in the answer.

    Extract:
    - education_requirement_text: the minimum educational requirement (e.g., "Bachelor’s degree or higher")
    - education_requirement_urls: list of URLs cited for the education requirement

    - di_experience_requirement_text: the minimum Division I coaching experience requirement (e.g., "at least 8 years of Division I coaching experience")
    - di_experience_requirement_urls: list of URLs cited for the DI experience requirement

    - recruiting_ability_text: the requirement that the head coach be able to recruit on a national level at major programs (or equivalent phrasing)
    - recruiting_ability_urls: list of URLs cited for the recruiting requirement

    - compliance_text: the requirement to ensure strict compliance with NCAA, conference, and university regulations (or equivalent phrasing)
    - compliance_urls: list of URLs cited for the compliance requirement

    IMPORTANT:
    - Only include URLs that are explicitly present in the answer.
    - If a field is not present in the answer, set the text to null and the corresponding URLs to an empty list.
    """


def prompt_extract_ncaa_certification() -> str:
    return """
    Extract NCAA Division I recruiting certification requirements as presented in the answer, with source URLs explicitly cited.

    Extract:
    - annual_requirement_text: statement that Division I coaches must pass the NCAA Coaches Certification (Recruiting) Test annually
    - annual_requirement_urls: list of URLs for the annual requirement

    - test_specs_text: statement about the test having 30 questions (or equivalent phrasing)
    - passing_score_text: statement about a minimum passing score of 80% (24 out of 30)
    - specs_urls: list of URLs for test specs and passing score

    - certification_period_text: statement of the certification period dates (e.g., "August 1 through July 31")
    - certification_period_urls: list of URLs for the certification period

    Only include URLs explicitly present in the answer text. If not mentioned, set missing text to null and URLs to an empty list.
    """


def prompt_extract_experience_benchmarks() -> str:
    return """
    Extract typical experience benchmarks for first-time NCAA Division I men's basketball head coaches as presented in the answer, with explicit URLs.

    Extract:
    - avg_age_text: the average age (e.g., "42.6")
    - avg_total_experience_years_text: the average total coaching experience in years (e.g., "15.6")
    - age_exp_urls: list of URLs supporting the average age and total experience

    - avg_di_experience_years_text: the average years of full-time Division I experience (e.g., "9.8")
    - di_exp_urls: list of URLs supporting the Division I experience average

    Only include URLs explicitly present in the answer text. Use strings for numbers if necessary.
    """


def prompt_extract_ad_standards() -> str:
    return """
    Extract educational attainment statistics for college athletic directors as presented in the answer, with explicit URLs.

    Extract:
    - master_degree_pct_text: percentage of all college athletic directors with master's degrees (e.g., "80%")
    - master_degree_pct_urls: list of URLs supporting this percentage

    - advanced_degree_pct_recent_hires_text: the percentage of athletic directors hired since 2009 who hold advanced degrees (e.g., "nearly 90%")
    - advanced_degree_recent_hires_urls: list of URLs supporting this claim

    - most_common_master_field_text: the most common master's degree field (e.g., "sports administration")
    - most_common_master_field_count_text: the associated count detail (e.g., "92 of 231")
    - most_common_master_field_urls: list of URLs supporting the field and count

    Only include URLs explicitly present in the answer text. If any are missing, return null or empty list as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _bool_present(text: Optional[str]) -> bool:
    return bool(text and str(text).strip())


def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def add_fact_with_reference_block(
    evaluator: Evaluator,
    parent_node,
    *,
    block_id: str,
    block_desc: str,
    claimed_text: Optional[str],
    sources: Optional[List[str]],
    value_check_id: Optional[str],
    value_check_desc: Optional[str],
    value_check_claim: Optional[str],
    value_check_additional: Optional[str],
    source_support_id: str,
    source_support_desc: str,
    source_support_claim: str,
    source_support_additional: Optional[str] = None,
) -> None:
    """
    Build a critical parallel block that requires:
      - the claim is stated in the answer,
      - at least one source URL is provided,
      - (optionally) the claim matches the specific value/constraint,
      - the claim is supported by the cited URLs.
    """
    block = evaluator.add_parallel(
        id=block_id,
        desc=block_desc,
        parent=parent_node,
        critical=True,
    )

    # Presence check: claim text provided in the answer
    evaluator.add_custom_node(
        result=_bool_present(claimed_text),
        id=f"{block_id}_stated_in_answer",
        desc=f"{block_desc} — stated in the answer",
        parent=block,
        critical=True,
    )

    # Sources provided
    evaluator.add_custom_node(
        result=_has_sources(sources),
        id=f"{block_id}_sources_provided",
        desc=f"{block_desc} — sources provided",
        parent=block,
        critical=True,
    )

    # Optional value/constraint check by simple verification
    if value_check_id and value_check_desc and value_check_claim:
        value_leaf = evaluator.add_leaf(
            id=value_check_id,
            desc=value_check_desc,
            parent=block,
            critical=True,
        )
        await evaluator.verify(
            claim=value_check_claim,
            node=value_leaf,
            additional_instruction=value_check_additional or "None",
        )

    # Source-supported verification
    support_leaf = evaluator.add_leaf(
        id=source_support_id,
        desc=source_support_desc,
        parent=block,
        critical=True,
    )
    await evaluator.verify(
        claim=source_support_claim,
        node=support_leaf,
        sources=sources or [],
        additional_instruction=source_support_additional or "None",
    )


async def verify_pathway_presentation(
    evaluator: Evaluator,
    parent_node,
) -> None:
    """
    Verify the answer presents a stepwise qualification pathway.
    """
    leaf = evaluator.add_leaf(
        id="Pathway_Presentation",
        desc="Presents a stepwise qualification pathway from assistant coach to Big Ten head-coach eligibility (not just an unstructured list).",
        parent=parent_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer presents a clear, stepwise qualification pathway (e.g., numbered steps or ordered stages) from assistant coach to Big Ten head-coach eligibility, not merely an unstructured list of items.",
        node=leaf,
        additional_instruction="If the answer uses numbered steps (1., 2., 3.), 'Step X', or an ordered sequence clearly indicating progression, consider it stepwise.",
    )


async def verify_penn_state_requirements(
    evaluator: Evaluator,
    parent_node,
    psu: PennStateRequirements,
) -> None:
    """
    Verify Penn State stated minimum requirements/expectations for the head coach.
    """
    group = evaluator.add_parallel(
        id="Penn_State_Head_Coach_Requirements",
        desc="Penn State stated minimum requirements/expectations for head coach are provided, each with a supporting reference.",
        parent=parent_node,
        critical=True,
    )

    # Educational credential requirement
    await add_fact_with_reference_block(
        evaluator,
        group,
        block_id="Educational_Credential_Requirement_With_Reference",
        block_desc="States Penn State minimum educational requirement (Bachelor's degree or higher) AND provides a supporting reference",
        claimed_text=psu.education_requirement_text,
        sources=psu.education_requirement_urls,
        value_check_id="Educational_Credential_Requirement_Value",
        value_check_desc="Answer states the minimum is a bachelor's degree or higher",
        value_check_claim="The answer states that Penn State's minimum educational requirement for its head basketball coach position is a bachelor's degree (or higher).",
        value_check_additional="Accept equivalent terms such as 'baccalaureate', 'BA/BS', or explicit mention that a master's or higher also satisfies the minimum.",
        source_support_id="Educational_Credential_Requirement_Source_Support",
        source_support_desc="Penn State minimum educational requirement is supported by cited sources",
        source_support_claim="According to the cited source(s), Penn State's minimum educational requirement for its head men's basketball coach is at least a bachelor's degree (or equivalent).",
        source_support_additional="Allow common phrasing variants that clearly indicate a bachelor's degree minimum or higher requirement.",
    )

    # Division I experience requirement (>= 8 years)
    await add_fact_with_reference_block(
        evaluator,
        group,
        block_id="Division_I_Experience_Requirement_With_Reference",
        block_desc="States Penn State minimum Division I coaching experience requirement (at least 8 years) AND provides a supporting reference",
        claimed_text=psu.di_experience_requirement_text,
        sources=psu.di_experience_requirement_urls,
        value_check_id="Division_I_Experience_Value",
        value_check_desc="Answer states the minimum Division I coaching experience is at least 8 years",
        value_check_claim="The answer states that Penn State's minimum Division I coaching experience requirement for the head basketball coach is at least 8 years.",
        value_check_additional="Allow equivalent wording such as 'eight (8) or more years' or 'minimum of eight years' of Division I coaching experience.",
        source_support_id="Division_I_Experience_Source_Support",
        source_support_desc="Penn State Division I experience minimum is supported by cited sources",
        source_support_claim="According to the cited source(s), Penn State requires at least 8 years of Division I coaching experience for its head men's basketball coach.",
        source_support_additional="Accept equivalent statements that clearly set the minimum at eight Division I coaching years.",
    )

    # Recruiting ability requirement (national level at major programs)
    await add_fact_with_reference_block(
        evaluator,
        group,
        block_id="Recruiting_Ability_Requirement_With_Reference",
        block_desc="Documents the requirement that head coaches demonstrate ability to recruit on a national level at major programs AND provides a supporting reference",
        claimed_text=psu.recruiting_ability_text,
        sources=psu.recruiting_ability_urls,
        value_check_id="Recruiting_Ability_Claim_Value",
        value_check_desc="Answer states requirement to recruit nationally at major programs",
        value_check_claim="The answer states that Penn State requires the head coach to demonstrate the ability to recruit on a national level at major programs (or equivalent phrasing).",
        value_check_additional="Minor wording variations acceptable as long as national-level recruiting at major programs is clearly conveyed.",
        source_support_id="Recruiting_Ability_Source_Support",
        source_support_desc="Recruiting ability requirement is supported by cited sources",
        source_support_claim="According to the cited source(s), Penn State expects the head coach to demonstrate the ability to recruit on a national level at major programs.",
        source_support_additional="Verify the cited page explicitly mentions national-level recruiting or equivalent scale/level of recruiting.",
    )

    # Compliance requirement (NCAA, conference, university regulations)
    await add_fact_with_reference_block(
        evaluator,
        group,
        block_id="Compliance_Requirement_With_Reference",
        block_desc="Documents the requirement that head coaches ensure strict compliance with NCAA, conference, and university regulations AND provides a supporting reference",
        claimed_text=psu.compliance_text,
        sources=psu.compliance_urls,
        value_check_id="Compliance_Requirement_Claim_Value",
        value_check_desc="Answer states requirement to ensure strict compliance with NCAA, conference, and university regulations",
        value_check_claim="The answer states that Penn State requires the head coach to ensure strict compliance with NCAA, conference, and university regulations (or equivalent phrasing).",
        value_check_additional="Accept minor phrasing variants that clearly encompass NCAA, conference, and university rules compliance.",
        source_support_id="Compliance_Requirement_Source_Support",
        source_support_desc="Compliance requirement is supported by cited sources",
        source_support_claim="According to the cited source(s), Penn State requires the head coach to ensure strict compliance with NCAA, conference, and university regulations.",
        source_support_additional="The page should clearly include all three: NCAA, conference, and university (or equivalent labels).",
    )


async def verify_ncaa_certification(
    evaluator: Evaluator,
    parent_node,
    ncaa: NCAACertification,
) -> None:
    """
    Verify NCAA Division I coaching certification test requirements, each with references.
    """
    group = evaluator.add_parallel(
        id="NCAA_Certification_Requirements",
        desc="NCAA Division I coaching certification test requirements are documented, each with a supporting reference.",
        parent=parent_node,
        critical=True,
    )

    # Annual passing requirement
    await add_fact_with_reference_block(
        evaluator,
        group,
        block_id="Annual_Passing_Requirement_With_Reference",
        block_desc="Documents that Division I coaches must pass the NCAA Coaches Certification (Recruiting) Test annually AND provides a supporting reference",
        claimed_text=ncaa.annual_requirement_text,
        sources=ncaa.annual_requirement_urls,
        value_check_id="Annual_Passing_Requirement_Value",
        value_check_desc="Answer states the NCAA Coaches Certification (Recruiting) Test is required annually",
        value_check_claim="The answer states that NCAA Division I coaches must pass the NCAA Coaches Certification (Recruiting) Test annually.",
        value_check_additional="Accept equivalent wording indicating an annual/once-per-year passing requirement.",
        source_support_id="Annual_Passing_Requirement_Source_Support",
        source_support_desc="Annual NCAA certification requirement is supported by cited sources",
        source_support_claim="According to the cited source(s), NCAA Division I coaches must pass the NCAA Coaches Certification (Recruiting) Test annually.",
        source_support_additional="The page should explicitly indicate an annual frequency requirement.",
    )

    # Test specs and passing score (30 questions, 80% = 24/30)
    both_present_text = None
    if _bool_present(ncaa.test_specs_text) and _bool_present(ncaa.passing_score_text):
        both_present_text = f"{ncaa.test_specs_text}; {ncaa.passing_score_text}"

    await add_fact_with_reference_block(
        evaluator,
        group,
        block_id="Test_Specs_And_Passing_Score_With_Reference",
        block_desc="Documents that the NCAA certification test consists of 30 questions AND the minimum passing score is 80% (24 out of 30 correct) AND provides a supporting reference",
        claimed_text=both_present_text,
        sources=ncaa.specs_urls,
        value_check_id="Test_Specs_And_Passing_Score_Value",
        value_check_desc="Answer states the test has 30 questions and the passing score is 80% (24/30)",
        value_check_claim="The answer states that the NCAA recruiting certification test consists of 30 questions and the minimum passing score is 80% (24 out of 30).",
        value_check_additional="Accept phrasing variants such as '30 multiple-choice questions' and '80 percent (24/30)'.",
        source_support_id="Test_Specs_And_Passing_Score_Source_Support",
        source_support_desc="NCAA test specs and passing score are supported by cited sources",
        source_support_claim="According to the cited source(s), the NCAA recruiting certification test consists of 30 questions and the minimum passing score is 80% (24/30).",
        source_support_additional="Verify that both the number of questions and the passing threshold are explicitly stated.",
    )

    # Certification period dates (August 1 through July 31)
    await add_fact_with_reference_block(
        evaluator,
        group,
        block_id="Certification_Period_Dates_With_Reference",
        block_desc="Documents the certification period effective dates (August 1 through July 31 annually) AND provides a supporting reference",
        claimed_text=ncaa.certification_period_text,
        sources=ncaa.certification_period_urls,
        value_check_id="Certification_Period_Dates_Value",
        value_check_desc="Answer states the certification period runs from August 1 through July 31",
        value_check_claim="The answer states that the certification period runs from August 1 through July 31 each year.",
        value_check_additional="Allow close phrasing variants such as 'Aug. 1 to July 31' or 'August 1 – July 31'.",
        source_support_id="Certification_Period_Dates_Source_Support",
        source_support_desc="NCAA certification period dates are supported by cited sources",
        source_support_claim="According to the cited source(s), the certification period runs from August 1 through July 31 each year.",
        source_support_additional="The page should explicitly cover the effective period spanning August 1 to July 31.",
    )


async def verify_experience_benchmarks(
    evaluator: Evaluator,
    parent_node,
    exp: ExperienceBenchmarks,
) -> None:
    """
    Verify benchmarks for first-time Division I head coaches, each with references.
    """
    group = evaluator.add_parallel(
        id="Typical_Experience_Benchmarks",
        desc="Benchmarks for first-time Division I head coaches are documented, each with a supporting reference.",
        parent=parent_node,
        critical=True,
    )

    # Average age and total experience
    both_present_text = None
    if _bool_present(exp.avg_age_text) and _bool_present(exp.avg_total_experience_years_text):
        both_present_text = f"{exp.avg_age_text}; {exp.avg_total_experience_years_text}"

    await add_fact_with_reference_block(
        evaluator,
        group,
        block_id="Average_Age_And_Total_Experience_With_Reference",
        block_desc="Documents that the average first-time Division I men's basketball head coach is 42.6 years old AND has 15.6 years of total coaching experience AND provides a supporting reference",
        claimed_text=both_present_text,
        sources=exp.age_exp_urls,
        value_check_id="Avg_Age_Total_Experience_Value",
        value_check_desc="Answer states average age 42.6 and total coaching experience 15.6 years",
        value_check_claim="The answer states that the average first-time Division I men's basketball head coach is 42.6 years old and has 15.6 years of total coaching experience.",
        value_check_additional="Minor rounding (e.g., 42.6→43) should be treated cautiously; prefer exact values where possible.",
        source_support_id="Avg_Age_Total_Experience_Source_Support",
        source_support_desc="Average age and total experience are supported by cited sources",
        source_support_claim="According to the cited source(s), the average first-time Division I men's basketball head coach is 42.6 years old and has 15.6 years of total coaching experience.",
        source_support_additional="Verify both figures (age and total experience) are present in the source material.",
    )

    # Division I experience average (9.8 years)
    await add_fact_with_reference_block(
        evaluator,
        group,
        block_id="Division_I_Experience_Average_With_Reference",
        block_desc="Documents the average years of full-time Division I experience for first-time Division I head coaches (9.8 years) AND provides a supporting reference",
        claimed_text=exp.avg_di_experience_years_text,
        sources=exp.di_exp_urls,
        value_check_id="DI_Experience_Average_Value",
        value_check_desc="Answer states average full-time Division I experience is 9.8 years",
        value_check_claim="The answer states that the average years of full-time Division I experience for first-time Division I head coaches is 9.8 years.",
        value_check_additional="Minor rounding variations should be handled carefully; prefer exact 9.8 if claimed in the rubric.",
        source_support_id="DI_Experience_Average_Source_Support",
        source_support_desc="Average Division I experience is supported by cited sources",
        source_support_claim="According to the cited source(s), the average full-time Division I experience for first-time Division I head coaches is 9.8 years.",
        source_support_additional="Ensure the term refers to full-time Division I experience.",
    )


async def verify_ad_standards(
    evaluator: Evaluator,
    parent_node,
    ad: ADStandards,
) -> None:
    """
    Verify athletic director education statistics, each with references.
    """
    group = evaluator.add_parallel(
        id="Athletic_Director_Educational_Standards",
        desc="Athletic director education statistics are documented, each with a supporting reference.",
        parent=parent_node,
        critical=True,
    )

    # Master's degree percentage overall (80%)
    await add_fact_with_reference_block(
        evaluator,
        group,
        block_id="Master_Degree_Percentage_Overall_With_Reference",
        block_desc="Documents the percentage of college-level athletic directors with master's degrees (80%) AND provides a supporting reference",
        claimed_text=ad.master_degree_pct_text,
        sources=ad.master_degree_pct_urls,
        value_check_id="Masters_Degree_Pct_Value",
        value_check_desc="Answer states that 80% of college athletic directors hold master's degrees",
        value_check_claim="The answer states that 80% of college athletic directors hold master's degrees.",
        value_check_additional="Accept '80 percent' or 'eight in ten' as equivalent phrasing.",
        source_support_id="Masters_Degree_Pct_Source_Support",
        source_support_desc="Master's degree percentage is supported by cited sources",
        source_support_claim="According to the cited source(s), 80% of college athletic directors hold master's degrees.",
        source_support_additional="Verify the 80% figure is explicitly stated for college-level athletic directors.",
    )

    # Advanced degrees among ADs hired since 2009 (nearly 90%)
    await add_fact_with_reference_block(
        evaluator,
        group,
        block_id="Advanced_Degree_Percentage_Recent_Hires_With_Reference",
        block_desc="Documents the percentage of athletic directors hired since 2009 with advanced degrees (nearly 90%) AND provides a supporting reference",
        claimed_text=ad.advanced_degree_pct_recent_hires_text,
        sources=ad.advanced_degree_recent_hires_urls,
        value_check_id="Advanced_Degree_Recent_Hires_Value",
        value_check_desc="Answer states that nearly 90% of ADs hired since 2009 hold advanced degrees",
        value_check_claim="The answer states that nearly 90% of athletic directors hired since 2009 hold advanced degrees.",
        value_check_additional="Accept phrasing like 'about 90%' or 'almost 90%' as 'nearly 90%'.",
        source_support_id="Advanced_Degree_Recent_Hires_Source_Support",
        source_support_desc="Advanced degree percentage for recent hires is supported by cited sources",
        source_support_claim="According to the cited source(s), nearly 90% of athletic directors hired since 2009 hold advanced degrees.",
        source_support_additional="Look for explicit coverage of the timeframe (since 2009) and advanced degree attainment.",
    )

    # Most common master's degree field and count (sports administration; 92 of 231)
    both_present_text = None
    if _bool_present(ad.most_common_master_field_text) and _bool_present(ad.most_common_master_field_count_text):
        both_present_text = f"{ad.most_common_master_field_text}; {ad.most_common_master_field_count_text}"

    await add_fact_with_reference_block(
        evaluator,
        group,
        block_id="Most_Common_Master_Degree_Field_And_Count_With_Reference",
        block_desc="Identifies the most common master's degree field for athletic directors (sports administration) AND documents the associated count detail (92 out of 231 documented cases) AND provides a supporting reference",
        claimed_text=both_present_text,
        sources=ad.most_common_master_field_urls,
        value_check_id="Most_Common_Field_Count_Value",
        value_check_desc="Answer states the most common master's field is sports administration with 92 of 231 cases",
        value_check_claim="The answer states that the most common master's degree field for athletic directors is sports administration and that there are 92 cases out of 231 documented.",
        value_check_additional="Accept close wording variants; both the field and the 92/231 count must be conveyed.",
        source_support_id="Most_Common_Field_Count_Source_Support",
        source_support_desc="Most common master's field and 92/231 count are supported by cited sources",
        source_support_claim="According to the cited source(s), the most common master's degree field for athletic directors is sports administration, with 92 out of 231 documented cases.",
        source_support_additional="Verify both the field and the 92/231 count are explicitly supported.",
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
    Evaluate an answer for the Penn State / Big Ten head coach eligibility pathway task.
    """
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

    # Add rubric "ground truth" expectations for transparency
    evaluator.add_ground_truth(
        {
            "expected": EXPECTED,
            "notes": "Numeric/textual benchmarks are required by rubric and should be source-supported in the answer.",
        },
        gt_type="rubric_expectations",
    )

    # Extract structured information (can be parallelized)
    pathway_task = evaluator.extract(
        prompt=prompt_extract_pathway_presentation(),
        template_class=PathwayPresentation,
        extraction_name="pathway_presentation",
    )
    psu_task = evaluator.extract(
        prompt=prompt_extract_penn_state_requirements(),
        template_class=PennStateRequirements,
        extraction_name="penn_state_requirements",
    )
    ncaa_task = evaluator.extract(
        prompt=prompt_extract_ncaa_certification(),
        template_class=NCAACertification,
        extraction_name="ncaa_certification",
    )
    exp_task = evaluator.extract(
        prompt=prompt_extract_experience_benchmarks(),
        template_class=ExperienceBenchmarks,
        extraction_name="experience_benchmarks",
    )
    ad_task = evaluator.extract(
        prompt=prompt_extract_ad_standards(),
        template_class=ADStandards,
        extraction_name="ad_standards",
    )

    pathway, psu, ncaa, exp, ad = await asyncio.gather(
        pathway_task, psu_task, ncaa_task, exp_task, ad_task
    )

    # Build top-level critical node (since Evaluator.root itself is always non-critical)
    top = evaluator.add_parallel(
        id="Complete_Qualification_Pathway_Documentation",
        desc="All required qualification components are documented (with references) and presented as a coherent eligibility pathway.",
        parent=root,
        critical=True,
    )

    # 1) Stepwise Pathway Presentation
    await verify_pathway_presentation(evaluator, top)

    # 2) Penn State Head Coach Requirements (with references)
    await verify_penn_state_requirements(evaluator, top, psu)

    # 3) NCAA Certification Requirements (with references)
    await verify_ncaa_certification(evaluator, top, ncaa)

    # 4) Typical Experience Benchmarks (with references)
    await verify_experience_benchmarks(evaluator, top, exp)

    # 5) Athletic Director Educational Standards (with references)
    await verify_ad_standards(evaluator, top, ad)

    # Return final structured summary
    return evaluator.get_summary()