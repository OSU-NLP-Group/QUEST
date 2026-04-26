import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "biz_dean_to_president_2025"
TASK_DESCRIPTION = (
    "A business school dean at a major public research university in the United States was appointed as president of "
    "the same institution in December 2025. This individual had served as dean of the business school for more than 10 years, "
    "beginning their tenure in 2015. Prior to joining academia, this person worked in the private sector for more than 20 years "
    "at a global management consulting firm. The individual holds a terminal degree from a top-tier U.S. institution. "
    "Identify this individual and provide the following information: (1) The person's full name, (2) The name of the business "
    "school where they served as dean, (3) The year they began serving as dean (must be 2015), (4) The name of their private "
    "sector employer before joining academia as dean, (5) At least one terminal degree they hold and the institution that granted it. "
    "All information must be supported by reference URLs from official university websites, news sources, or professional profiles."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PersonProfileExtraction(BaseModel):
    # Identity and institution
    full_name: Optional[str] = None
    institution_name: Optional[str] = None  # University where the business school resides (and presidency is held)
    business_school_name: Optional[str] = None

    # Dean role evidence
    dean_start_year: Optional[str] = None  # Expect "2015"
    dean_sources: List[str] = Field(default_factory=list)

    # Institution classification evidence
    institution_public_research_sources: List[str] = Field(default_factory=list)

    # Presidency evidence
    presidential_institution_name: Optional[str] = None
    presidential_sources: List[str] = Field(default_factory=list)

    # Private sector employer evidence
    employer_name: Optional[str] = None
    employer_sources: List[str] = Field(default_factory=list)
    private_sector_years_claim: Optional[str] = None  # e.g., "more than 20 years", "over two decades"
    private_sector_sources: List[str] = Field(default_factory=list)

    # Terminal degree evidence
    terminal_degree: Optional[str] = None  # e.g., "Ed.D. in Higher Education Management"
    terminal_degree_institution: Optional[str] = None  # e.g., "University of Pennsylvania"
    terminal_degree_sources: List[str] = Field(default_factory=list)

    # Top-tier support evidence (rankings, prestige references)
    top_tier_support_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_person_profile() -> str:
    return """
    Extract the individual's identity and all required attributes exactly as presented in the answer, along with the specific supporting URLs cited in the answer text.

    Required fields to extract:
    1) full_name: The individual's full name.
    2) institution_name: The name of the university where the business school resides (and where the presidency is held).
    3) business_school_name: The name of the business school where the individual served as dean.
    4) dean_start_year: The year the individual began serving as dean (should be "2015" if stated).
    5) dean_sources: All URLs in the answer that support the dean role and/or its start year and tenure.
    6) institution_public_research_sources: All URLs that support that the institution is a U.S. public research university.
    7) presidential_institution_name: The university at which the individual was appointed president (should match institution_name, possibly with minor naming variations).
    8) presidential_sources: All URLs that support the presidential appointment (especially that it was in December 2025).
    9) employer_name: The private-sector employer (global management consulting firm) where the individual worked prior to academia.
    10) employer_sources: All URLs that support the employment at that firm.
    11) private_sector_years_claim: The exact phrasing of the years of private-sector experience (e.g., "more than 20 years", "over two decades"), if present in the answer. If not present, return null.
    12) private_sector_sources: All URLs that support the years/tenure statement for the private-sector experience.
    13) terminal_degree: At least one terminal degree the individual holds (e.g., Ph.D., Ed.D., J.D., M.D.), with the degree name exactly as stated.
    14) terminal_degree_institution: The institution that granted the terminal degree.
    15) terminal_degree_sources: All URLs that support both the terminal degree and the granting institution.
    16) top_tier_support_sources: All URLs that support that the terminal-degree institution is top-tier/highly ranked/prestigious in the U.S. (e.g., explicit statements like "Ivy League", or reputable rankings/news pages).

    URL extraction rules:
    - Extract only URLs that are explicitly present in the answer (including markdown links).
    - Return full valid URLs. If a URL is missing the protocol, prepend http://
    - If multiple URLs are given for a field, include all of them in the corresponding list.

    If a particular attribute is missing in the answer, set the field to null. If its URLs are missing, return an empty array for that field's URL list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for urls in url_lists:
        if not urls:
            continue
        for u in urls:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification helpers (subtrees)                                             #
# --------------------------------------------------------------------------- #
async def verify_individual_full_name(evaluator: Evaluator, parent, data: PersonProfileExtraction) -> None:
    """
    Parallel critical group:
      - Existence of name
      - At least one URL supports/identifies the named individual (can be any strong page like official bio, news, or profile)
    """
    group = evaluator.add_parallel(
        id="individual_full_name_with_citation",
        desc="Provide the individual's full name and at least one reference URL identifying them.",
        parent=parent,
        critical=True
    )

    # Existence check (critical)
    evaluator.add_custom_node(
        result=bool(data.full_name and data.full_name.strip()),
        id="full_name_provided",
        desc="Full name is provided in the answer",
        parent=group,
        critical=True
    )

    # Verify name reference via any strong page(s)
    name_support_leaf = evaluator.add_leaf(
        id="full_name_supported_by_sources",
        desc="A cited URL clearly identifies the individual with the provided full name",
        parent=group,
        critical=True
    )
    name_support_sources = combine_sources(
        data.presidential_sources,
        data.dean_sources,
        data.employer_sources,
        data.terminal_degree_sources
    )
    await evaluator.verify(
        claim=f"This page clearly identifies the individual named '{data.full_name}'.",
        node=name_support_leaf,
        sources=name_support_sources,
        additional_instruction="The page should explicitly mention the person's full name in a headline, profile title, or equivalent authoritative context."
    )


async def verify_presidential_appointment(evaluator: Evaluator, parent, data: PersonProfileExtraction) -> None:
    """
    Parallel critical group:
      - Appointed president in December 2025 (with URL)
      - Presidency is at the same institution as the dean role (with URL)
    """
    group = evaluator.add_parallel(
        id="presidential_appointment_constraints",
        desc="Verify the presidential appointment constraints with supporting source(s).",
        parent=parent,
        critical=True
    )

    # Existence of presidential sources
    evaluator.add_custom_node(
        result=bool(data.presidential_sources),
        id="presidential_sources_present",
        desc="At least one presidential appointment source URL is provided",
        parent=group,
        critical=True
    )

    # Appointed president in Dec 2025
    appointed_leaf = evaluator.add_leaf(
        id="appointed_president_dec_2025_with_url",
        desc="Confirm (with a cited URL) that the individual was appointed as president in December 2025.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.full_name} was appointed as president of {data.presidential_institution_name or data.institution_name} in December 2025.",
        node=appointed_leaf,
        sources=data.presidential_sources,
        additional_instruction="The page should explicitly indicate appointment/selection/board approval in December 2025 (announcement date in Dec 2025 counts, even if start date is later)."
    )

    # Presidency same institution as dean
    same_inst_leaf = evaluator.add_leaf(
        id="president_same_institution_as_dean_with_url",
        desc="Confirm (with a cited URL) that the presidency is at the same institution where the individual served as business school dean.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The presidency is at the same university where {data.full_name} served as dean of {data.business_school_name} (i.e., {data.institution_name}).",
        node=same_inst_leaf,
        sources=combine_sources(data.presidential_sources, data.dean_sources),
        additional_instruction="Allow common institutional name variants and abbreviations; the page(s) should make clear it's the same university."
    )


async def verify_business_school_and_tenure(evaluator: Evaluator, parent, data: PersonProfileExtraction) -> None:
    """
    Parallel critical group:
      - Business school name with URL (dean role at that school/institution)
      - Institution is a U.S. public research university (with URL)
      - Dean start year is 2015 (with URL)
      - Served as dean for more than 10 years (with URL)
    """
    group = evaluator.add_parallel(
        id="business_school_and_dean_tenure_constraints",
        desc="Verify the dean role and tenure constraints with supporting source(s).",
        parent=parent,
        critical=True
    )

    # Existence of dean sources
    evaluator.add_custom_node(
        result=bool(data.dean_sources),
        id="dean_sources_present",
        desc="At least one dean-role source URL is provided",
        parent=group,
        critical=True
    )

    # Business school + dean role
    bs_leaf = evaluator.add_leaf(
        id="business_school_name_with_url",
        desc="Provide the name of the business school where the individual served as dean, with a cited URL supporting this.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.full_name} served as dean of the {data.business_school_name} at {data.institution_name}.",
        node=bs_leaf,
        sources=data.dean_sources,
        additional_instruction="The page should explicitly state the dean title and the business school name at the specified university."
    )

    # Institution classification: U.S. public research university
    pub_research_exist = evaluator.add_custom_node(
        result=bool(data.institution_public_research_sources),
        id="public_research_sources_present",
        desc="At least one source URL is provided to support that the institution is a U.S. public research university",
        parent=group,
        critical=True
    )
    pub_research_leaf = evaluator.add_leaf(
        id="institution_public_research_us_with_url",
        desc="Confirm (with a cited URL) that the institution is a U.S. public research university.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.institution_name} is a public research university in the United States.",
        node=pub_research_leaf,
        sources=data.institution_public_research_sources,
        additional_instruction="Accept official or well-recognized sources explicitly describing the institution as a 'public research university' or equivalent recognized classification."
    )

    # Dean start year = 2015
    start_year_leaf = evaluator.add_leaf(
        id="dean_start_year_2015_with_url",
        desc="Confirm (with a cited URL) that the individual began serving as dean in 2015.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.full_name} began serving as dean in 2015.",
        node=start_year_leaf,
        sources=data.dean_sources,
        additional_instruction="The page should explicitly mention the start year 2015 for the dean appointment."
    )

    # Dean tenure > 10 years
    tenure_leaf = evaluator.add_leaf(
        id="dean_tenure_more_than_10_years_with_url",
        desc="Confirm (with a cited URL) that the individual served as dean for more than 10 years.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.full_name} served as dean for more than 10 years.",
        node=tenure_leaf,
        sources=combine_sources(data.dean_sources, data.presidential_sources),
        additional_instruction="Accept phrasings like 'more than 10 years', 'over a decade', or explicit date ranges implying >10 years."
    )


async def verify_private_sector(evaluator: Evaluator, parent, data: PersonProfileExtraction) -> None:
    """
    Parallel critical group:
      - Employer name before academia (with URL)
      - Employer is a global management consulting firm (with URL)
      - Worked >20 years in private sector before becoming dean (with URL)
    """
    group = evaluator.add_parallel(
        id="private_sector_constraints",
        desc="Verify private-sector employment constraints with supporting source(s).",
        parent=parent,
        critical=True
    )

    # Existence of employer sources
    evaluator.add_custom_node(
        result=bool(data.employer_sources),
        id="employer_sources_present",
        desc="At least one employer source URL is provided",
        parent=group,
        critical=True
    )

    # Employer name with URL
    employer_leaf = evaluator.add_leaf(
        id="private_sector_employer_name_with_url",
        desc="Provide the name of the individual's private-sector employer before joining academia as dean, with a cited URL supporting this.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Before becoming dean in 2015, {data.full_name} worked at {data.employer_name}.",
        node=employer_leaf,
        sources=data.employer_sources,
        additional_instruction="The page should clearly associate the individual with the specified employer prior to their dean appointment."
    )

    # Employer classification: global management consulting firm
    mgmt_firm_leaf = evaluator.add_leaf(
        id="employer_global_management_consulting_firm_with_url",
        desc="Confirm (with a cited URL) that the employer is a global management consulting firm.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.employer_name} is a global management consulting firm.",
        node=mgmt_firm_leaf,
        sources=data.employer_sources,
        additional_instruction="Accept authoritative descriptions (official site, Wikipedia, major news/business profiles) explicitly describing it as a global management consulting firm."
    )

    # Private sector >20 years
    yrs_leaf = evaluator.add_leaf(
        id="private_sector_more_than_20_years_with_url",
        desc="Confirm (with a cited URL) that the individual worked in the private sector for more than 20 years before becoming dean.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"Before becoming dean in 2015, {data.full_name} worked in the private sector for more than 20 years (specifically at {data.employer_name}).",
        node=yrs_leaf,
        sources=combine_sources(data.private_sector_sources, data.employer_sources),
        additional_instruction="Accept phrasings like 'more than 20 years' or 'over two decades' clearly linked to the individual's tenure at the employer."
    )


async def verify_terminal_degree(evaluator: Evaluator, parent, data: PersonProfileExtraction) -> None:
    """
    Parallel critical group:
      - Terminal degree and granting institution (with URL)
      - Degree-granting institution is top-tier/highly ranked/prestigious in the U.S. (with URL)
    """
    group = evaluator.add_parallel(
        id="terminal_degree_constraints",
        desc="Provide at least one terminal degree and verify related constraints with supporting source(s).",
        parent=parent,
        critical=True
    )

    # Existence of terminal degree sources
    evaluator.add_custom_node(
        result=bool(data.terminal_degree_sources),
        id="terminal_degree_sources_present",
        desc="At least one terminal-degree source URL is provided",
        parent=group,
        critical=True
    )

    # Terminal degree and granting institution
    degree_leaf = evaluator.add_leaf(
        id="terminal_degree_and_granting_institution_with_url",
        desc="Provide at least one terminal degree the individual holds and the granting institution, with a cited URL supporting both.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.full_name} holds {data.terminal_degree} from {data.terminal_degree_institution}.",
        node=degree_leaf,
        sources=data.terminal_degree_sources,
        additional_instruction="The page should explicitly mention both the terminal degree (e.g., Ph.D., Ed.D., J.D., M.D.) and the granting institution."
    )

    # Existence of top-tier support sources
    evaluator.add_custom_node(
        result=bool(data.top_tier_support_sources),
        id="top_tier_sources_present",
        desc="At least one source URL is provided to support that the degree institution is top-tier/highly ranked/prestigious",
        parent=group,
        critical=True
    )

    # Degree institution top-tier
    top_tier_leaf = evaluator.add_leaf(
        id="degree_institution_top_tier_us_with_url",
        desc="Provide a cited source that explicitly supports the claim that the terminal-degree institution is top-tier/highly ranked/prestigious in the U.S.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.terminal_degree_institution} is a top-tier, highly ranked, or prestigious U.S. institution.",
        node=top_tier_leaf,
        sources=data.top_tier_support_sources,
        additional_instruction="Accept explicit prestige indicators such as 'Ivy League' or reputable rankings/recognitions clearly signaling top-tier status."
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
    Evaluate an answer for the 'business school dean -> president (Dec 2025)' identification and verification task.
    """
    # Initialize evaluator (root node is non-critical by design; we will mark top-level groups critical)
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_person_profile(),
        template_class=PersonProfileExtraction,
        extraction_name="person_profile_extraction"
    )

    # Build verification tree according to rubric
    # 1) Individual full name with citation
    await verify_individual_full_name(evaluator, root, extracted)

    # 2) Presidential appointment constraints
    await verify_presidential_appointment(evaluator, root, extracted)

    # 3) Business school and dean tenure constraints
    await verify_business_school_and_tenure(evaluator, root, extracted)

    # 4) Private-sector employment constraints
    await verify_private_sector(evaluator, root, extracted)

    # 5) Terminal degree constraints
    await verify_terminal_degree(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()