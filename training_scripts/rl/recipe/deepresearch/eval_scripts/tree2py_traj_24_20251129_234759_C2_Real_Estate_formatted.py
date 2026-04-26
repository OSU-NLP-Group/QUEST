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
TASK_ID = "reit_ceo_cbs_bigfour_top10_2024_2025"
TASK_DESCRIPTION = (
    "Identify the full name of the Chief Executive Officer of a publicly traded real estate investment trust (REIT) "
    "company that meets all of the following criteria: (1) The CEO holds an MBA degree from Columbia Business School, "
    "with graduation occurring between 2005 and 2015 (inclusive). (2) The CEO holds an undergraduate degree in "
    "Engineering from a university located outside the United States. (3) The CEO began their professional career at "
    "one of the Big Four accounting firms (PricewaterhouseCoopers, Deloitte, Ernst & Young, or KPMG). "
    "(4) The company is ranked among the top 10 largest REITs in the United States by market capitalization as of "
    "2024-2025. Provide the CEO's full name and the name of the REIT company they lead."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CoreInfo(BaseModel):
    ceo_full_name: Optional[str] = None
    company_name: Optional[str] = None


class EducationInfo(BaseModel):
    mba_institution: Optional[str] = None
    mba_program: Optional[str] = None
    mba_graduation_year: Optional[str] = None  # keep as string for robustness
    undergrad_degree: Optional[str] = None     # e.g., "B.Eng. Mechanical Engineering"
    undergrad_institution: Optional[str] = None
    undergrad_institution_country: Optional[str] = None


class CareerInfo(BaseModel):
    first_employer: Optional[str] = None        # e.g., "PwC", "PricewaterhouseCoopers"
    first_role: Optional[str] = None            # Optional detail like "Audit Associate"
    career_start_phrase: Optional[str] = None   # e.g., "began his career at PwC"


class SourcesExtraction(BaseModel):
    general_urls: List[str] = Field(default_factory=list)
    role_urls: List[str] = Field(default_factory=list)
    education_urls: List[str] = Field(default_factory=list)
    career_urls: List[str] = Field(default_factory=list)
    company_urls: List[str] = Field(default_factory=list)
    ranking_urls: List[str] = Field(default_factory=list)
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_core_info() -> str:
    return """
    Extract the core outputs explicitly mentioned in the answer:
    - ceo_full_name: The full name of the identified CEO.
    - company_name: The name of the publicly traded REIT company they lead.
    Rules:
    - Return exactly the strings as written in the answer.
    - If any field is not explicitly stated, return null.
    """


def prompt_extract_education_info() -> str:
    return """
    Extract the CEO's education details as explicitly stated in the answer:
    - mba_institution: The institution/school that granted the MBA (e.g., "Columbia Business School").
    - mba_program: The MBA program name if mentioned (e.g., "MBA", "M.B.A.").
    - mba_graduation_year: The graduation year for the MBA if stated (as a string, e.g., "2010").
    - undergrad_degree: The undergraduate degree field/discipline (e.g., "Mechanical Engineering", "Electrical Engineering").
    - undergrad_institution: The undergraduate university name.
    - undergrad_institution_country: The country/location of the undergraduate university if stated.
    Rules:
    - Only extract what is explicitly present in the answer text.
    - If an item is not present, set it to null.
    """


def prompt_extract_career_info() -> str:
    return """
    Extract the CEO's early career details as explicitly stated in the answer:
    - first_employer: The name of the first employer where the CEO began their professional career.
    - first_role: The job title or role at that first employer if stated.
    - career_start_phrase: The exact phrase fragment if the answer explicitly states 'began/began his career at ...'.
    Rules:
    - Do not infer or invent. Use only what is in the answer.
    - If any field is missing, return null for that field.
    """


def prompt_extract_sources() -> str:
    return """
    Extract all URLs mentioned anywhere in the answer, categorizing them if possible:
    - general_urls: Any URLs not clearly tied to a specific claim.
    - role_urls: URLs that support the claim that the identified person is the CEO of the company.
    - education_urls: URLs that support MBA or undergraduate details.
    - career_urls: URLs that support the CEO's early career information.
    - company_urls: URLs about the company (official site, investor relations, profiles, etc.).
    - ranking_urls: URLs that discuss rankings or market capitalization position (e.g., lists of top REITs).
    - all_urls: Include every URL you find (complete list).
    Special rules for URL extraction:
    - Extract only URLs explicitly present in the answer.
    - URLs can be plain, markdown links, or otherwise embedded; extract the actual URL targets.
    - If a URL is missing a protocol, prepend 'http://'.
    - If a category is unclear, place the URL into general_urls and always include it in all_urls as well.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
BIG_FOUR_NAMES = [
    "PricewaterhouseCoopers", "PwC",
    "Deloitte",
    "Ernst & Young", "EY",
    "KPMG"
]


def _exists_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in lst or []:
            if not url:
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root_node,
    core: CoreInfo,
    edu: EducationInfo,
    career: CareerInfo,
    srcs: SourcesExtraction
) -> None:
    # CEO_Identification (critical, parallel)
    ceo_ident_node = evaluator.add_parallel(
        id="CEO_Identification",
        desc="Identify the full name of a CEO and their publicly traded REIT company meeting all specified education, career, and size/ranking criteria.",
        parent=root_node,
        critical=True
    )

    # Response_Contents (critical, parallel)
    response_contents = evaluator.add_parallel(
        id="Response_Contents",
        desc="The response includes the required outputs (CEO full name and REIT company name).",
        parent=ceo_ident_node,
        critical=True
    )

    # Provides_CEO_Full_Name (critical leaf via custom existence)
    evaluator.add_custom_node(
        result=_exists_str(core.ceo_full_name),
        id="Provides_CEO_Full_Name",
        desc="Provides the CEO's full name.",
        parent=response_contents,
        critical=True
    )

    # Provides_Company_Name (critical leaf via custom existence)
    evaluator.add_custom_node(
        result=_exists_str(core.company_name),
        id="Provides_Company_Name",
        desc="Provides the name of the REIT company the CEO leads.",
        parent=response_contents,
        critical=True
    )

    # Role_Verification (critical leaf)
    role_node = evaluator.add_leaf(
        id="Role_Verification",
        desc="The identified individual is the Chief Executive Officer of the identified company.",
        parent=ceo_ident_node,
        critical=True
    )
    role_sources = _combine_sources(srcs.role_urls, srcs.company_urls, srcs.general_urls, srcs.all_urls)
    ceo_name = core.ceo_full_name or "the identified person"
    company_name = core.company_name or "the identified company"
    role_claim = f"{ceo_name} is the Chief Executive Officer (CEO) of {company_name}."
    await evaluator.verify(
        claim=role_claim,
        node=role_node,
        sources=role_sources,
        additional_instruction=(
            "Verify that the person currently holds (or is described as holding) the CEO title at the specified company. "
            "Allow minor naming variations. Focus on evidence from the provided URLs."
        )
    )

    # Educational_Background (critical, parallel)
    edu_node = evaluator.add_parallel(
        id="Educational_Background",
        desc="Verify the CEO's graduate and undergraduate education requirements.",
        parent=ceo_ident_node,
        critical=True
    )
    edu_sources = _combine_sources(srcs.education_urls, srcs.general_urls, srcs.all_urls)

    # MBA_From_Columbia_Business_School (critical leaf)
    mba_cbs_node = evaluator.add_leaf(
        id="MBA_From_Columbia_Business_School",
        desc="The CEO holds an MBA degree from Columbia Business School.",
        parent=edu_node,
        critical=True
    )
    mba_cbs_claim = f"{ceo_name} holds an MBA degree from Columbia Business School."
    await evaluator.verify(
        claim=mba_cbs_claim,
        node=mba_cbs_node,
        sources=edu_sources,
        additional_instruction=(
            "Confirm that the MBA is specifically from Columbia Business School (part of Columbia University). "
            "Accept equivalent phrasing such as 'MBA from Columbia Business School' or 'Columbia University (Columbia Business School)'."
        )
    )

    # MBA_Graduation_Year_2005_to_2015 (critical leaf)
    mba_year_node = evaluator.add_leaf(
        id="MBA_Graduation_Year_2005_to_2015",
        desc="The CEO's MBA graduation year is between 2005 and 2015 inclusive.",
        parent=edu_node,
        critical=True
    )
    if _exists_str(edu.mba_graduation_year):
        year_text = edu.mba_graduation_year.strip()
        mba_year_claim = f"{ceo_name}'s MBA graduation year is {year_text}, which falls between 2005 and 2015 inclusive."
    else:
        mba_year_claim = f"{ceo_name}'s MBA graduation year occurred between 2005 and 2015 inclusive."
    await evaluator.verify(
        claim=mba_year_claim,
        node=mba_year_node,
        sources=edu_sources,
        additional_instruction=(
            "If a specific year is shown in the evidence, ensure it lies within 2005–2015 inclusive. "
            "Accept minor formatting differences (e.g., 'Class of 2010')."
        )
    )

    # Undergrad_Degree_In_Engineering (critical leaf)
    ug_eng_node = evaluator.add_leaf(
        id="Undergrad_Degree_In_Engineering",
        desc="The CEO holds an undergraduate degree in Engineering.",
        parent=edu_node,
        critical=True
    )
    ug_eng_claim = (
        f"{ceo_name} holds an undergraduate degree in engineering (any engineering discipline such as mechanical, "
        f"electrical, civil, chemical, industrial, etc.)."
    )
    await evaluator.verify(
        claim=ug_eng_claim,
        node=ug_eng_node,
        sources=edu_sources,
        additional_instruction=(
            "Look for explicit mention of an engineering undergraduate degree. "
            "Accept discipline-specific engineering degrees as valid (e.g., 'B.Eng. Mechanical Engineering')."
        )
    )

    # Undergrad_Institution_Outside_US (critical leaf)
    ug_outside_node = evaluator.add_leaf(
        id="Undergrad_Institution_Outside_US",
        desc="The CEO's undergraduate engineering degree is from a university located outside the United States.",
        parent=edu_node,
        critical=True
    )
    if _exists_str(edu.undergrad_institution):
        ug_outside_claim = (
            f"{ceo_name}'s undergraduate engineering degree was earned from {edu.undergrad_institution}, "
            f"which is located outside the United States."
        )
    else:
        ug_outside_claim = (
            f"{ceo_name}'s undergraduate engineering degree was earned from a university located outside the United States."
        )
    await evaluator.verify(
        claim=ug_outside_claim,
        node=ug_outside_node,
        sources=edu_sources,
        additional_instruction=(
            "Confirm the institution's country. It must be outside the United States. "
            "If the evidence shows the institution located in a non-U.S. country, consider this satisfied."
        )
    )

    # Career_Background (critical leaf)
    career_node = evaluator.add_leaf(
        id="Career_Background",
        desc="Verify the CEO began their professional career at a Big Four accounting firm (PwC, Deloitte, EY, or KPMG).",
        parent=ceo_ident_node,
        critical=True
    )
    career_sources = _combine_sources(srcs.career_urls, srcs.general_urls, srcs.all_urls)
    if _exists_str(career.first_employer):
        career_claim = (
            f"{ceo_name} began their professional career at {career.first_employer}, "
            f"which is one of the Big Four accounting firms (PricewaterhouseCoopers/PwC, Deloitte, Ernst & Young/EY, KPMG)."
        )
    else:
        career_claim = (
            f"{ceo_name} began their professional career at one of the Big Four accounting firms: "
            f"PricewaterhouseCoopers (PwC), Deloitte, Ernst & Young (EY), or KPMG."
        )
    await evaluator.verify(
        claim=career_claim,
        node=career_node,
        sources=career_sources,
        additional_instruction=(
            "Look for explicit wording such as 'began his/her career at ...' or 'started his/her career at ...'. "
            "Accept standard abbreviations (PwC, EY)."
        )
    )

    # Company_Verification (critical, parallel)
    company_node = evaluator.add_parallel(
        id="Company_Verification",
        desc="Verify the company meets the REIT/public-trading and top-10 market-cap ranking criteria for 2024–2025.",
        parent=ceo_ident_node,
        critical=True
    )
    company_sources = _combine_sources(srcs.company_urls, srcs.general_urls, srcs.all_urls)
    ranking_sources = _combine_sources(srcs.ranking_urls, srcs.company_urls, srcs.general_urls, srcs.all_urls)

    # Company_Is_REIT (critical leaf)
    is_reit_node = evaluator.add_leaf(
        id="Company_Is_REIT",
        desc="The company is a real estate investment trust (REIT).",
        parent=company_node,
        critical=True
    )
    reit_claim = f"{company_name} is a real estate investment trust (REIT)."
    await evaluator.verify(
        claim=reit_claim,
        node=is_reit_node,
        sources=company_sources,
        additional_instruction="Confirm that the company's legal/organizational form is a REIT."
    )

    # Company_Is_Publicly_Traded (critical leaf)
    is_public_node = evaluator.add_leaf(
        id="Company_Is_Publicly_Traded",
        desc="The company is publicly traded.",
        parent=company_node,
        critical=True
    )
    public_claim = f"{company_name} is publicly traded on a stock exchange (e.g., NYSE or Nasdaq)."
    await evaluator.verify(
        claim=public_claim,
        node=is_public_node,
        sources=company_sources,
        additional_instruction="Look for ticker/exchange information indicating public listing."
    )

    # Company_Top10_US_REIT_By_MarketCap_2024_2025 (critical leaf)
    top10_node = evaluator.add_leaf(
        id="Company_Top10_US_REIT_By_MarketCap_2024_2025",
        desc="The company is ranked among the top 10 largest REITs in the United States by market capitalization as of 2024–2025.",
        parent=company_node,
        critical=True
    )
    top10_claim = (
        f"{company_name} is ranked among the top 10 largest REITs in the United States by market capitalization "
        f"as of 2024 or 2025."
    )
    await evaluator.verify(
        claim=top10_claim,
        node=top10_node,
        sources=ranking_sources,
        additional_instruction=(
            "Focus on lists/rankings specifically for US REITs by market capitalization and dated for 2024 or 2025. "
            "Accept reliable sources (e.g., Nareit, S&P Global, reputable financial publications)."
        )
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
    Evaluate an answer for the REIT CEO identification task.
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
        default_model=model
    )

    # Extract structured info (run concurrently)
    core_task = evaluator.extract(
        prompt=prompt_extract_core_info(),
        template_class=CoreInfo,
        extraction_name="core_info"
    )
    edu_task = evaluator.extract(
        prompt=prompt_extract_education_info(),
        template_class=EducationInfo,
        extraction_name="education_info"
    )
    career_task = evaluator.extract(
        prompt=prompt_extract_career_info(),
        template_class=CareerInfo,
        extraction_name="career_info"
    )
    sources_task = evaluator.extract(
        prompt=prompt_extract_sources(),
        template_class=SourcesExtraction,
        extraction_name="sources_info"
    )

    core_info, education_info, career_info, sources_info = await asyncio.gather(
        core_task, edu_task, career_task, sources_task
    )

    # Build verification tree and run checks
    await build_and_verify_tree(
        evaluator=evaluator,
        root_node=root,
        core=core_info,
        edu=education_info,
        career=career_info,
        srcs=sources_info
    )

    return evaluator.get_summary()