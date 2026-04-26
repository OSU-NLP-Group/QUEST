import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "project_genie_manuscript"
TASK_DESCRIPTION = (
    "Prepare a complete research paper manuscript for submission to a peer-reviewed computer science history journal "
    "about Project Genie, the pioneering computer research project at UC Berkeley. Your manuscript must meet all "
    "standard academic journal requirements and accurately document the historical facts about Project Genie.\n\n"
    "Your submission must include:\n\n"
    "1. Abstract (maximum 250 words) providing a clear overview of the paper including research problem, methodology approach, "
    "key findings, and implications\n\n"
    "2. Introduction following an inverted triangle structure (general to specific) that establishes the research problem, "
    "presents relevant background context, states research objectives, and includes proper URL references for background claims\n\n"
    "3. Methodology section describing all tools and materials used, explaining the data collection process, stating sampling "
    "procedures and criteria, specifying sample size or scope, and including proper URL references for methods used\n\n"
    "4. Results section presenting factual statements supported by evidence in logical sequence without interpretation or bias, "
    "accurately documenting Project Genie's historical facts including: start year (1964), institution (UC Berkeley), funding source (ARPA), "
    "project leader (Prof. Bob Evans), key contributors (Butler Lampson, Peter Deutsch, Chuck Thacker), and technical achievements "
    "(development of SDS-940 with paged virtual memory based on SDS-930), with proper URL references supporting historical claims\n\n"
    "5. Discussion section including interpretation of findings, analysis of results, and explanation of implications without duplicating "
    "the results section, with proper URL references for claims and interpretations\n\n"
    "6. References section with consistent citation style throughout, complete bibliographic information for each source, all sources used in "
    "the paper, and minimum 5 scholarly sources\n\n"
    "7. Author Information with primary affiliation correctly identifying the institution where research was conducted, and an author "
    "contributions statement specifying exact contributions of each author following recognized taxonomy and detailing specific role for each listed author\n\n"
    "The manuscript should demonstrate conciseness in the results section and proper formatting throughout (though formatting is not the primary evaluation criterion)."
)

# Ground truth facts for Project Genie (used in verification claims)
GT_START_YEAR = "1964"
GT_INSTITUTION = "UC Berkeley"
GT_FUNDING = "ARPA"
GT_LEADER = "Bob Evans"
GT_LEADER_ALT = "Robert Evans"
GT_KEY_CONTRIBUTORS = ["Butler Lampson", "Peter Deutsch", "Chuck Thacker"]
GT_TECH_ACHIEVEMENT = "development of the SDS-940 with paged virtual memory based on the SDS-930"


# -----------------------------------------------------------------------------
# Pydantic models for extraction
# -----------------------------------------------------------------------------
class SectionData(BaseModel):
    text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ReferenceItem(BaseModel):
    citation_text: Optional[str] = None
    url: Optional[str] = None
    is_scholarly: Optional[bool] = None


class AuthorItem(BaseModel):
    name: Optional[str] = None
    affiliation: Optional[str] = None
    contributions: List[str] = Field(default_factory=list)


class ProjectGenieFacts(BaseModel):
    start_year: Optional[str] = None
    institution: Optional[str] = None
    funding_source: Optional[str] = None
    project_leader: Optional[str] = None
    key_contributors: List[str] = Field(default_factory=list)
    technical_achievement: Optional[str] = None


class ManuscriptExtraction(BaseModel):
    abstract: SectionData = Field(default_factory=SectionData)
    introduction: SectionData = Field(default_factory=SectionData)
    methodology: SectionData = Field(default_factory=SectionData)
    results: SectionData = Field(default_factory=SectionData)
    discussion: SectionData = Field(default_factory=SectionData)

    references: List[ReferenceItem] = Field(default_factory=list)

    authors: List[AuthorItem] = Field(default_factory=list)
    primary_affiliation: Optional[str] = None
    taxonomy_name: Optional[str] = None

    facts: ProjectGenieFacts = Field(default_factory=ProjectGenieFacts)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_manuscript() -> str:
    return """
    Extract structured manuscript information from the answer. Return a JSON object with the following fields:

    1) abstract: { text, urls[] }
    2) introduction: { text, urls[] }
    3) methodology: { text, urls[] }
    4) results: { text, urls[] }
    5) discussion: { text, urls[] }
    6) references: [ { citation_text, url, is_scholarly } ... ]
       - is_scholarly: true if the source appears to be scholarly (e.g., peer-reviewed journal/conference paper, academic book), false otherwise.
    7) authors: [ { name, affiliation, contributions[] } ... ]
       - contributions[]: list each author's roles (e.g., conceptualization, methodology, investigation, writing—original draft, writing—review & editing, supervision, etc.)
    8) primary_affiliation: the primary institutional affiliation stated for the research
    9) taxonomy_name: the name of any recognized author contribution taxonomy mentioned (e.g., "CRediT")
    10) facts: {
         start_year,
         institution,
         funding_source,
         project_leader,
         key_contributors[],   // list of names found in the manuscript as key contributors
         technical_achievement // a concise phrase about the main technical achievement
       }

    RULES:
    - Extract section texts exactly as written in the manuscript. If a section is missing, set its text to null and urls to [].
    - Extract URLs explicitly present in each section. Include only valid URLs that appear in the text.
    - For references: include all items listed in the References section. Each item should have citation_text (the full citation as written), and url if present; set is_scholarly using your best judgment from the citation_text and venue/publisher.
    - For authors: list all authors named anywhere in the manuscript, capture affiliation if provided, and the contributions for each author if provided in a contributions statement.
    - For facts: read the Results section primarily; if facts appear elsewhere, still capture them. If a fact is not stated, set it to null or [].

    If any required field is missing in the answer, return null for that field or an empty array for lists. Do not add information not present in the answer.
    """


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _word_count(text: Optional[str]) -> int:
    if not text:
        return 0
    return len([w for w in text.strip().split() if w])


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _collect_all_section_urls(extr: ManuscriptExtraction) -> List[str]:
    urls: List[str] = []
    urls.extend(extr.introduction.urls or [])
    urls.extend(extr.methodology.urls or [])
    urls.extend(extr.results.urls or [])
    urls.extend(extr.discussion.urls or [])
    return list(dict.fromkeys(urls))  # deduplicate while preserving order


def _collect_reference_urls(extr: ManuscriptExtraction) -> List[str]:
    ref_urls = [r.url for r in extr.references if r.url]
    return list(dict.fromkeys(ref_urls))


def _min_scholarly_count(extr: ManuscriptExtraction) -> int:
    return sum(1 for r in extr.references if r.is_scholarly is True)


def _contributions_for_each_author(authors: List[AuthorItem]) -> bool:
    if not authors:
        return False
    return all((a.name and len(a.contributions) > 0) for a in authors)


def _contains_any(text: Optional[str], needles: List[str]) -> bool:
    if not text:
        return False
    lt = text.lower()
    return all(n.lower() in lt for n in needles)


def _name_in_list_or_text(name: str, names_list: List[str], text: Optional[str]) -> bool:
    # Check normalized presence
    if any(name.lower() in (n or "").lower() for n in names_list):
        return True
    if text and name.lower() in text.lower():
        return True
    return False


# -----------------------------------------------------------------------------
# Section verification builders
# -----------------------------------------------------------------------------
async def build_abstract_section(evaluator: Evaluator, parent, extr: ManuscriptExtraction):
    node = evaluator.add_parallel(
        id="AbstractSection",
        desc="Abstract meets stated requirements",
        parent=parent,
        critical=True
    )

    # Existence gate
    evaluator.add_custom_node(
        result=bool(extr.abstract.text and extr.abstract.text.strip()),
        id="AbstractExists",
        desc="Abstract text exists",
        parent=node,
        critical=True
    )

    # Word count
    wc = _word_count(extr.abstract.text)
    evaluator.add_custom_node(
        result=wc <= 250,
        id="AbstractWordCount",
        desc="Abstract does not exceed 250 words",
        parent=node,
        critical=True
    )

    # Break down required elements into separate leaf checks
    # Problem/focus
    leaf_problem = evaluator.add_leaf(
        id="AbstractProblemIncluded",
        desc="Abstract includes a statement of the research problem or focus",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The abstract includes a clear statement of the research problem or focus.",
        node=leaf_problem,
        additional_instruction="Focus only on the Abstract section. Accept reasonable synonyms indicating problem/focus."
    )

    # Methodology approach
    leaf_method = evaluator.add_leaf(
        id="AbstractMethodologyIncluded",
        desc="Abstract includes methodology approach",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The abstract describes the methodology approach used in the research.",
        node=leaf_method,
        additional_instruction="Focus only on the Abstract section. Accept concise method descriptions."
    )

    # Key findings
    leaf_findings = evaluator.add_leaf(
        id="AbstractFindingsIncluded",
        desc="Abstract includes key findings",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The abstract presents key findings from the research.",
        node=leaf_findings,
        additional_instruction="Focus only on the Abstract section. Accept summary-level findings."
    )

    # Implications
    leaf_implications = evaluator.add_leaf(
        id="AbstractImplicationsIncluded",
        desc="Abstract includes implications",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The abstract includes implications of the findings.",
        node=leaf_implications,
        additional_instruction="Focus only on the Abstract section. Accept implications phrased as significance or impact."
    )


async def build_introduction_section(evaluator: Evaluator, parent, extr: ManuscriptExtraction):
    node = evaluator.add_parallel(
        id="IntroductionSection",
        desc="Introduction meets stated structure/content and URL-citation requirements",
        parent=parent,
        critical=True
    )

    # Existence gate
    evaluator.add_custom_node(
        result=bool(extr.introduction.text and extr.introduction.text.strip()),
        id="IntroductionExists",
        desc="Introduction text exists",
        parent=node,
        critical=True
    )

    # Structure: inverted triangle
    leaf_structure = evaluator.add_leaf(
        id="IntroductionStructure",
        desc="Introduction follows inverted-triangle structure (general to specific)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Introduction follows an inverted-triangle structure (general to specific) before stating specific objectives.",
        node=leaf_structure,
        additional_instruction="Focus only on the Introduction section."
    )

    # Establish research problem
    leaf_problem = evaluator.add_leaf(
        id="IntroductionResearchProblem",
        desc="Introduction establishes the research problem",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Introduction establishes the research problem clearly.",
        node=leaf_problem,
        additional_instruction="Focus only on the Introduction section."
    )

    # Background context
    leaf_context = evaluator.add_leaf(
        id="IntroductionBackgroundContext",
        desc="Introduction presents relevant background context",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Introduction presents relevant background context for Project Genie.",
        node=leaf_context,
        additional_instruction="Focus only on the Introduction section."
    )

    # Objectives
    leaf_objectives = evaluator.add_leaf(
        id="IntroductionObjectives",
        desc="Introduction states research objectives",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Introduction states the research objectives.",
        node=leaf_objectives,
        additional_instruction="Focus only on the Introduction section."
    )

    # URL references presence
    evaluator.add_custom_node(
        result=_has_urls(extr.introduction.urls),
        id="IntroductionURLReferences",
        desc="Background/context factual claims in the Introduction are supported with proper URL references",
        parent=node,
        critical=True
    )


async def build_methodology_section(evaluator: Evaluator, parent, extr: ManuscriptExtraction):
    node = evaluator.add_parallel(
        id="MethodologySection",
        desc="Methodology includes all required components and URL-citation requirements",
        parent=parent,
        critical=True
    )

    # Existence gate
    evaluator.add_custom_node(
        result=bool(extr.methodology.text and extr.methodology.text.strip()),
        id="MethodologyExists",
        desc="Methodology text exists",
        parent=node,
        critical=True
    )

    # Tools and materials
    leaf_tools = evaluator.add_leaf(
        id="MethodologyToolsMaterials",
        desc="Methodology describes tools and materials used",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Methodology describes the tools and materials used.",
        node=leaf_tools,
        additional_instruction="Focus only on the Methodology section."
    )

    # Data collection
    leaf_data = evaluator.add_leaf(
        id="MethodologyDataCollection",
        desc="Methodology explains the data collection process",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Methodology explains the data collection process.",
        node=leaf_data,
        additional_instruction="Focus only on the Methodology section."
    )

    # Sampling procedures/criteria
    leaf_sampling = evaluator.add_leaf(
        id="MethodologySampling",
        desc="Methodology states sampling procedures and criteria",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Methodology states the sampling procedures and criteria.",
        node=leaf_sampling,
        additional_instruction="Focus only on the Methodology section."
    )

    # Sample size/scope
    leaf_sample_size = evaluator.add_leaf(
        id="MethodologySampleSize",
        desc="Methodology specifies sample size or scope",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Methodology specifies the sample size or scope.",
        node=leaf_sample_size,
        additional_instruction="Focus only on the Methodology section."
    )

    # URL references presence
    evaluator.add_custom_node(
        result=_has_urls(extr.methodology.urls),
        id="MethodologyURLReferences",
        desc="Methodology factual claims (e.g., methods/tools used) are supported with proper URL references",
        parent=node,
        critical=True
    )


async def build_results_section(evaluator: Evaluator, parent, extr: ManuscriptExtraction):
    node = evaluator.add_parallel(
        id="ResultsSection",
        desc="Results meet presentation standards, include required URL support, and accurately document specified Project Genie facts",
        parent=parent,
        critical=True
    )

    # Existence gate
    results_exists = evaluator.add_custom_node(
        result=bool(extr.results.text and extr.results.text.strip()),
        id="ResultsExists",
        desc="Results text exists",
        parent=node,
        critical=True
    )

    # Factual claims supported by URLs (gate for downstream fact verifications)
    urls_present_node = evaluator.add_custom_node(
        result=_has_urls(extr.results.urls),
        id="ResultsFactualClaimsSupportedByURLs",
        desc="Results present factual statements supported by evidence via proper URL references for historical/factual claims",
        parent=node,
        critical=True
    )

    # Logical sequence
    leaf_sequence = evaluator.add_leaf(
        id="ResultsLogicalSequence",
        desc="Results present information in a logical sequence",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Results section presents information in a logical, coherent sequence.",
        node=leaf_sequence,
        additional_instruction="Focus only on the Results section."
    )

    # No interpretation/bias
    leaf_no_interp = evaluator.add_leaf(
        id="ResultsNoInterpretation",
        desc="Results contain no interpretation or bias (i.e., no discussion/implications language)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Results section avoids interpretation or implications language and sticks to factual statements.",
        node=leaf_no_interp,
        additional_instruction="Focus only on the Results section."
    )

    # Conciseness
    leaf_concise = evaluator.add_leaf(
        id="ResultsConciseness",
        desc="Results are concise without excess words (i.e., avoids unnecessary verbosity beyond stating results)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Results section is concise and avoids unnecessary verbosity.",
        node=leaf_concise,
        additional_instruction="Focus only on the Results section."
    )

    # Historical facts subgroup
    facts_node = evaluator.add_parallel(
        id="ProjectGenieHistoricalFacts",
        desc="Results accurately document all specified Project Genie historical facts",
        parent=node,
        critical=True
    )

    # Helper: prerequisites for fact verification
    extra_prereqs = [results_exists, urls_present_node]

    # 1) Start year 1964
    evaluator.add_custom_node(
        result=("1964" in (extr.facts.start_year or "") or ("1964" in (extr.results.text or "").lower())),
        id="ProjectStartYear_Mentioned",
        desc="Results mention Project Genie started in 1964",
        parent=facts_node,
        critical=True
    )
    leaf_start_year_support = evaluator.add_leaf(
        id="ProjectStartYear_Supported",
        desc="States Project Genie started in 1964 (supported by URLs)",
        parent=facts_node,
        critical=True
    )
    await evaluator.verify(
        claim="Project Genie started in 1964.",
        node=leaf_start_year_support,
        sources=extr.results.urls,
        additional_instruction="Accept phrasing such as 'began in 1964' or 'started in 1964'.",
        extra_prerequisites=extra_prereqs
    )

    # 2) Institution UC Berkeley
    evaluator.add_custom_node(
        result=("berkeley" in (extr.facts.institution or "").lower() or ("berkeley" in (extr.results.text or "").lower())),
        id="ProjectInstitution_Mentioned",
        desc="Results mention Project Genie was at UC Berkeley",
        parent=facts_node,
        critical=True
    )
    leaf_institution_support = evaluator.add_leaf(
        id="ProjectInstitution_Supported",
        desc="States Project Genie was at UC Berkeley (supported by URLs)",
        parent=facts_node,
        critical=True
    )
    await evaluator.verify(
        claim="Project Genie was at the University of California, Berkeley (UC Berkeley).",
        node=leaf_institution_support,
        sources=extr.results.urls,
        additional_instruction="Allow minor variations in naming (e.g., 'University of California at Berkeley', 'UC Berkeley').",
        extra_prerequisites=extra_prereqs
    )

    # 3) Funding ARPA
    evaluator.add_custom_node(
        result=(
            ("arpa" in (extr.facts.funding_source or "").lower())
            or ("darpa" in (extr.facts.funding_source or "").lower())
            or ("arpa" in (extr.results.text or "").lower())
            or ("darpa" in (extr.results.text or "").lower())
        ),
        id="ProjectFunding_Mentioned",
        desc="Results mention Project Genie was ARPA-funded",
        parent=facts_node,
        critical=True
    )
    leaf_funding_support = evaluator.add_leaf(
        id="ProjectFunding_Supported",
        desc="States Project Genie was ARPA-funded (supported by URLs)",
        parent=facts_node,
        critical=True
    )
    await evaluator.verify(
        claim="Project Genie was funded by ARPA.",
        node=leaf_funding_support,
        sources=extr.results.urls,
        additional_instruction="Accept references to ARPA or DARPA (as ARPA's later name).",
        extra_prerequisites=extra_prereqs
    )

    # 4) Leader Prof. Bob Evans
    leader_mentioned = (
        _contains_any(extr.facts.project_leader or "", ["evans"])
        or _contains_any(extr.results.text or "", ["evans"])
    )
    evaluator.add_custom_node(
        result=leader_mentioned,
        id="ProjectLeader_Mentioned",
        desc="Results mention Project Genie was led by Prof. Bob Evans",
        parent=facts_node,
        critical=True
    )
    leaf_leader_support = evaluator.add_leaf(
        id="ProjectLeader_Supported",
        desc="States Project Genie was led by Prof. Bob Evans (supported by URLs)",
        parent=facts_node,
        critical=True
    )
    await evaluator.verify(
        claim="Project Genie was led by Prof. Bob Evans.",
        node=leaf_leader_support,
        sources=extr.results.urls,
        additional_instruction="Accept 'Robert Evans' or 'Robert P. Evans' as equivalent to 'Bob Evans'.",
        extra_prerequisites=extra_prereqs
    )

    # 5) Key contributors: Lampson, Deutsch, Thacker
    kc_present = all(
        _name_in_list_or_text(name, extr.facts.key_contributors or [], extr.results.text)
        for name in GT_KEY_CONTRIBUTORS
    )
    evaluator.add_custom_node(
        result=kc_present,
        id="KeyContributorsAllSpecified_Mentioned",
        desc="Results identify Butler Lampson, Peter Deutsch, and Chuck Thacker as key contributors",
        parent=facts_node,
        critical=True
    )
    leaf_kc_support = evaluator.add_leaf(
        id="KeyContributorsAllSpecified_Supported",
        desc="Identifies Butler Lampson, Peter Deutsch, and Chuck Thacker as key contributors (supported by URLs)",
        parent=facts_node,
        critical=True
    )
    await evaluator.verify(
        claim="Butler Lampson, Peter Deutsch, and Chuck Thacker were key contributors to Project Genie.",
        node=leaf_kc_support,
        sources=extr.results.urls,
        additional_instruction="Minor variations in name formatting or middle initials are acceptable.",
        extra_prerequisites=extra_prereqs
    )

    # 6) Technical achievement: SDS-940 with paged VM based on SDS-930
    tech_keywords_ok = _contains_any(extr.facts.technical_achievement or extr.results.text or "",
                                     ["sds-940", "paged", "virtual", "memory", "sds-930"])
    evaluator.add_custom_node(
        result=tech_keywords_ok,
        id="TechnicalAchievement_Mentioned",
        desc="Results state the technical achievement (SDS-940 with paged virtual memory based on SDS-930)",
        parent=facts_node,
        critical=True
    )
    leaf_tech_support = evaluator.add_leaf(
        id="TechnicalAchievement_Supported",
        desc="States the technical achievement: development of the SDS-940 with paged virtual memory based on the SDS-930 (supported by URLs)",
        parent=facts_node,
        critical=True
    )
    await evaluator.verify(
        claim="Project Genie developed the SDS-940 with paged virtual memory based on the SDS-930.",
        node=leaf_tech_support,
        sources=extr.results.urls,
        additional_instruction="Accept equivalent phrasing indicating SDS-940 used paged virtual memory derived from SDS-930.",
        extra_prerequisites=extra_prereqs
    )


async def build_discussion_section(evaluator: Evaluator, parent, extr: ManuscriptExtraction):
    node = evaluator.add_parallel(
        id="DiscussionSection",
        desc="Discussion provides analysis/interpretation/implications without duplicating Results and includes URL support where claims are made",
        parent=parent,
        critical=True
    )

    # Existence gate
    evaluator.add_custom_node(
        result=bool(extr.discussion.text and extr.discussion.text.strip()),
        id="DiscussionExists",
        desc="Discussion text exists",
        parent=node,
        critical=True
    )

    # Interpretation
    leaf_interp = evaluator.add_leaf(
        id="DiscussionInterpretation",
        desc="Discussion includes interpretation of findings",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Discussion includes interpretation of the findings.",
        node=leaf_interp,
        additional_instruction="Focus only on the Discussion section."
    )

    # Analysis
    leaf_analysis = evaluator.add_leaf(
        id="DiscussionAnalysis",
        desc="Discussion includes analysis of results",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Discussion includes analysis of the results.",
        node=leaf_analysis,
        additional_instruction="Focus only on the Discussion section."
    )

    # Implications
    leaf_implications = evaluator.add_leaf(
        id="DiscussionImplications",
        desc="Discussion explains implications",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Discussion explains the implications of the findings.",
        node=leaf_implications,
        additional_instruction="Focus only on the Discussion section."
    )

    # No duplication of Results
    leaf_no_dup = evaluator.add_leaf(
        id="DiscussionNoDuplicateResults",
        desc="Discussion does not duplicate the Results section (i.e., does not restate results as the primary content)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Discussion does not primarily duplicate or restate the Results; it focuses on interpretation and analysis.",
        node=leaf_no_dup,
        additional_instruction="Focus only on the Discussion section."
    )

    # URL references presence
    evaluator.add_custom_node(
        result=_has_urls(extr.discussion.urls),
        id="DiscussionURLReferences",
        desc="Non-trivial factual/interpretive claims in Discussion are supported with proper URL references",
        parent=node,
        critical=True
    )


async def build_references_section(evaluator: Evaluator, parent, extr: ManuscriptExtraction):
    node = evaluator.add_parallel(
        id="ReferencesSection",
        desc="References meet citation-style, completeness, coverage, and minimum-source requirements",
        parent=parent,
        critical=True
    )

    # Existence gate
    evaluator.add_custom_node(
        result=bool(extr.references and len(extr.references) > 0),
        id="ReferencesExist",
        desc="References section exists with at least one item",
        parent=node,
        critical=True
    )

    # Consistent citation style
    leaf_style = evaluator.add_leaf(
        id="ConsistentCitationStyle",
        desc="References use a consistent citation style throughout",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The References use a consistent citation style throughout.",
        node=leaf_style,
        additional_instruction="Judge based on the References section as written."
    )

    # Complete bibliographic information
    leaf_biblio = evaluator.add_leaf(
        id="CompleteBibliographicInfo",
        desc="Each reference includes complete bibliographic information",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Each reference includes complete bibliographic information (e.g., authors, year, title, venue/publisher).",
        node=leaf_biblio,
        additional_instruction="Judge based on the References section; allow minor variations typical of citation styles."
    )

    # All sources used are cited
    all_section_urls = _collect_all_section_urls(extr)
    ref_urls = _collect_reference_urls(extr)
    all_sources_in_refs = set(all_section_urls).issubset(set(ref_urls))
    evaluator.add_custom_node(
        result=all_sources_in_refs,
        id="AllSourcesUsedCited",
        desc="References include all sources used in the paper",
        parent=node,
        critical=True
    )

    # Minimum scholarly sources
    evaluator.add_custom_node(
        result=_min_scholarly_count(extr) >= 5,
        id="MinimumScholarlySources",
        desc="References include at least 5 scholarly sources",
        parent=node,
        critical=True
    )


async def build_author_information_section(evaluator: Evaluator, parent, extr: ManuscriptExtraction):
    node = evaluator.add_parallel(
        id="AuthorInformationSection",
        desc="Author information includes correct affiliation and a compliant contributions statement",
        parent=parent,
        critical=True
    )

    # Existence gate
    evaluator.add_custom_node(
        result=bool((extr.authors and len(extr.authors) > 0) or (extr.primary_affiliation and extr.primary_affiliation.strip())),
        id="AuthorInfoExists",
        desc="Author information exists",
        parent=node,
        critical=True
    )

    # Primary affiliation correctness
    leaf_affil = evaluator.add_leaf(
        id="PrimaryAffiliation",
        desc="Primary affiliation identifies the institution where the research was conducted",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The primary affiliation identifies the institution where the research was conducted.",
        node=leaf_affil,
        additional_instruction="Judge based on the Author Information section and any explicit institutional statements."
    )

    # Contributions statement subgroup
    contrib_node = evaluator.add_parallel(
        id="AuthorContributionsStatement",
        desc="Author contributions statement specifies each listed author's role/contributions",
        parent=node,
        critical=True
    )

    # Each author has contributions
    evaluator.add_custom_node(
        result=_contributions_for_each_author(extr.authors),
        id="ContributionsForEachAuthor",
        desc="Statement specifies contributions/roles for each listed author (no listed author is missing a role)",
        parent=contrib_node,
        critical=True
    )

    # Recognized taxonomy (override to critical to satisfy framework constraints)
    leaf_tax = evaluator.add_leaf(
        id="RecognizedTaxonomy",
        desc="Statement follows a recognized author-contribution taxonomy (e.g., CRediT)",
        parent=contrib_node,
        critical=True  # override to True due to critical parent constraint
    )
    await evaluator.verify(
        claim="The author contributions statement follows a recognized taxonomy such as CRediT.",
        node=leaf_tax,
        additional_instruction="Look for explicit mention of 'CRediT' or similar recognized taxonomy."
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate a manuscript answer for the Project Genie journal submission requirements.
    """
    # Initialize evaluator with a parallel root (overall compliance aggregator)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Manuscript about Project Genie satisfies all stated section, citation/URL, and historical-fact constraints",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured manuscript information
    extr: ManuscriptExtraction = await evaluator.extract(
        prompt=prompt_extract_manuscript(),
        template_class=ManuscriptExtraction,
        extraction_name="manuscript_extraction"
    )

    # Record ground truth for transparency
    evaluator.add_ground_truth({
        "expected_facts": {
            "start_year": GT_START_YEAR,
            "institution": GT_INSTITUTION,
            "funding": GT_FUNDING,
            "leader": GT_LEADER,
            "key_contributors": GT_KEY_CONTRIBUTORS,
            "technical_achievement": GT_TECH_ACHIEVEMENT
        }
    }, gt_type="project_genie_ground_truth")

    # Build sections
    await build_abstract_section(evaluator, root, extr)
    await build_introduction_section(evaluator, root, extr)
    await build_methodology_section(evaluator, root, extr)
    await build_results_section(evaluator, root, extr)
    await build_discussion_section(evaluator, root, extr)
    await build_references_section(evaluator, root, extr)
    await build_author_information_section(evaluator, root, extr)

    # Return summary
    return evaluator.get_summary()