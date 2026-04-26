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
TASK_ID = "scientific_reports_submission_requirements"
TASK_DESCRIPTION = (
    "A research team has completed a NASA-funded study on Mars atmospheric electrical activity using the Perseverance rover's SuperCam instrument. "
    "Their research analyzed 28 hours of microphone recordings collected over two Martian years (1,374 Earth days) and documented 55 instances of electrical discharges associated with dust devils and dust storm fronts. "
    "The team plans to submit their findings to Scientific Reports, an open-access journal in the Nature portfolio. "
    "Identify and document all mandatory and recommended manuscript submission requirements from Scientific Reports that apply to this research article, including: "
    "title length limit, abstract word count limit, main text word count recommendation, keywords limit, display items (figures/tables) maximum, figure legend word count limit, references count guideline, requirement for Data Availability Statement, "
    "requirement for Competing Interests Statement, requirement for Author Contributions Statement, cover letter components that must be included, ethics approval documentation requirements (if applicable), "
    "federal funding compliance requirements (Data Management and Sharing Plan for NASA funding), and verification that all key research details (study duration of 28 hours over 1,374 Earth days, 55 documented events, association with dust devils and dust storm fronts, and SuperCam instrument identification) "
    "must be accurately reported. For each requirement, provide the specific criterion or limit and reference the authoritative source from Scientific Reports' author guidelines or relevant federal funding policies."
)

# Expected constants according to the rubric description
EXPECTED_LIMITS = {
    "title_words_max": 20,
    "abstract_words_max": 200,
    "main_text_words_max": 4500,  # excluding Abstract/Methods/References/figure legends
    "keywords_max": 6,
    "display_items_max": 8,  # figures + tables combined
    "figure_legend_words_max": 350,
    "references_approx": 60
}

RESEARCH_EXPECTED = {
    "duration_hours": "28 hours",
    "earth_days": "1,374 Earth days",
    "martian_years": "two Martian years",
    "event_count": "55",
    "association_terms": ["dust devils", "dust storm fronts"],
    "instrument": "SuperCam",
    "platform": "Perseverance rover"
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NumericRequirement(BaseModel):
    stated_limit: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BinaryRequirement(BaseModel):
    stated_required: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class CoverLetterRequirement(BaseModel):
    components: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class EthicsRequirement(BaseModel):
    applies_to: Optional[str] = None  # e.g., "human participants or animals"
    required_statement: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class FundingRequirement(BaseModel):
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ResearchDetails(BaseModel):
    duration_hours: Optional[str] = None
    earth_days: Optional[str] = None
    martian_years: Optional[str] = None
    events_count: Optional[str] = None
    association_terms: List[str] = Field(default_factory=list)
    instrument: Optional[str] = None
    platform: Optional[str] = None


class ManuscriptRequirementsExtraction(BaseModel):
    title_limit: NumericRequirement = NumericRequirement()
    abstract_limit: NumericRequirement = NumericRequirement()
    main_text_limit: NumericRequirement = NumericRequirement()
    keywords_limit: NumericRequirement = NumericRequirement()
    display_items_limit: NumericRequirement = NumericRequirement()
    figure_legend_limit: NumericRequirement = NumericRequirement()
    references_guideline: NumericRequirement = NumericRequirement()

    data_availability_required: BinaryRequirement = BinaryRequirement()
    competing_interests_required: BinaryRequirement = BinaryRequirement()
    author_contributions_required: BinaryRequirement = BinaryRequirement()

    cover_letter_requirements: CoverLetterRequirement = CoverLetterRequirement()
    ethics_approval_requirements: EthicsRequirement = EthicsRequirement()

    funding_dms_plan: FundingRequirement = FundingRequirement()
    funding_public_access_2025: FundingRequirement = FundingRequirement()

    research_details: ResearchDetails = ResearchDetails()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract the Scientific Reports submission requirements and federal compliance notes as presented in the answer. 
    Return the following JSON fields. Use null for missing values. Extract only information explicitly present in the answer. 
    For each requirement, also extract ALL source URLs cited in the answer that support the requirement (Scientific Reports author guidelines or relevant federal policy pages).

    Fields to extract:
    - title_limit: 
        stated_limit: the stated title length limit (e.g., "≤20 words", "20 words", "no more than twenty words").
        sources: array of URLs cited for this item.
    - abstract_limit:
        stated_limit: the stated abstract word limit (e.g., "≤200 words", "200 words").
        sources: array of URLs.
    - main_text_limit:
        stated_limit: the stated main text word-count recommendation/limit (e.g., "≤4,500 words, excluding Abstract/Methods/References/figure legends").
        sources: array of URLs.
    - keywords_limit:
        stated_limit: the stated keywords limit (e.g., "up to 6 keywords").
        sources: array of URLs.
    - display_items_limit:
        stated_limit: the maximum number of display items (figures + tables combined, e.g., "≤8").
        sources: array of URLs.
    - figure_legend_limit:
        stated_limit: the figure legend word limit per legend (e.g., "≤350 words per legend").
        sources: array of URLs.
    - references_guideline:
        stated_limit: the references count guideline (e.g., "approximately 60").
        sources: array of URLs.

    - data_availability_required:
        stated_required: boolean (true if the answer says DAS is required for all manuscripts).
        sources: array of URLs.
    - competing_interests_required:
        stated_required: boolean (true if the answer says a Competing Interests statement is required for each author).
        sources: array of URLs.
    - author_contributions_required:
        stated_required: boolean (true if the answer says an Author Contributions statement is required).
        sources: array of URLs.

    - cover_letter_requirements:
        components: array of cover-letter components the answer lists (e.g., "corresponding author affiliation/contact", "suitability for the journal", "suggested reviewers").
        sources: array of URLs.

    - ethics_approval_requirements:
        applies_to: description of when ethics approval documentation is required (e.g., "human subjects or animals").
        required_statement: boolean (true if the answer states that an ethics approval statement must be included in Methods when applicable).
        sources: array of URLs.

    - funding_dms_plan:
        statement: the text summarizing Data Management and Sharing Plan requirements for NASA-funded research as stated in the answer.
        sources: array of URLs.

    - funding_public_access_2025:
        statement: the text summarizing the requirement that by end of 2025 federally funded research must be immediately publicly available, as stated in the answer.
        sources: array of URLs.

    - research_details:
        duration_hours: the stated total hours analyzed (e.g., "28 hours").
        earth_days: the stated Earth days (e.g., "1,374 Earth days").
        martian_years: the Martian-year span text (e.g., "two Martian years").
        events_count: the number of documented events (e.g., "55").
        association_terms: array of associations for events (e.g., ["dust devils", "dust storm fronts"]).
        instrument: instrument name (e.g., "SuperCam").
        platform: platform (e.g., "Perseverance rover").

    IMPORTANT:
    - For URLs, extract only actual URLs found in the answer (including markdown link targets). If none are provided for an item, return an empty array.
    - For limits/guidelines, keep them as short strings exactly as stated in the answer (do not normalize numbers beyond what the answer states).
    - If a field is not present in the answer, return null (or empty array for lists).
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_requirements(
    evaluator: Evaluator,
    parent_node,
    ex: ManuscriptRequirementsExtraction
) -> None:
    """
    Build the verification tree under the Manuscript_Requirements_Documentation node and run verifications.
    Each leaf represents one verification step following the rubric. For guideline-based checks, we verify claims against cited sources (URLs).
    For research detail accuracy checks, we verify against the provided answer and prompt context using simple verification.
    """

    # Title limit (critical)
    node_title = evaluator.add_leaf(
        id="Title_Limit_Documentation",
        desc="States the title length limit (≤20 words) and cites the Scientific Reports author/submission guidelines.",
        parent=parent_node,
        critical=True
    )
    claim_title = "Scientific Reports author/submission guidelines state that the manuscript title should be no more than 20 words (≤20 words)."
    await evaluator.verify(
        claim=claim_title,
        node=node_title,
        sources=ex.title_limit.sources,
        additional_instruction="Judge only based on the cited Scientific Reports/Nature guidelines page(s). If the page indicates a different limit or is ambiguous, mark Incorrect."
    )

    # Abstract limit (critical)
    node_abs = evaluator.add_leaf(
        id="Abstract_Limit_Documentation",
        desc="States the abstract word limit (≤200 words) and cites the Scientific Reports author/submission guidelines.",
        parent=parent_node,
        critical=True
    )
    claim_abs = "Scientific Reports author/submission guidelines state that the abstract should be at most 200 words (≤200 words)."
    await evaluator.verify(
        claim=claim_abs,
        node=node_abs,
        sources=ex.abstract_limit.sources,
        additional_instruction="Focus on explicit word limits for abstracts on Scientific Reports guideline pages."
    )

    # Main text limit/recommendation (non-critical)
    node_main = evaluator.add_leaf(
        id="Main_Text_Limit_Documentation",
        desc="States the main text word-count recommendation/limit (≤4,500 words, excluding Abstract/Methods/References/figure legends) and cites the Scientific Reports guidelines.",
        parent=parent_node,
        critical=False
    )
    claim_main = (
        "Scientific Reports author/submission guidelines recommend or limit the main text to at most 4,500 words, "
        "excluding Abstract, Methods, References, and figure legends."
    )
    await evaluator.verify(
        claim=claim_main,
        node=node_main,
        sources=ex.main_text_limit.sources,
        additional_instruction="Treat recommendations as supportive. The exclusion of Abstract/Methods/References/figure legends must be explicitly stated or clearly implied on the page."
    )

    # Keywords limit (non-critical)
    node_kw = evaluator.add_leaf(
        id="Keywords_Limit_Documentation",
        desc="States the keywords limit (up to 6) and cites the Scientific Reports author/submission guidelines.",
        parent=parent_node,
        critical=False
    )
    claim_kw = "Scientific Reports author/submission guidelines state that authors may provide up to 6 keywords."
    await evaluator.verify(
        claim=claim_kw,
        node=node_kw,
        sources=ex.keywords_limit.sources,
        additional_instruction="Check the relevant author instructions for keyword limits; allow minor wording variants like 'no more than six'."
    )

    # Display items limit (non-critical)
    node_disp = evaluator.add_leaf(
        id="Display_Items_Limit_Documentation",
        desc="States the maximum number of display items (≤8 figures/tables combined) and cites the Scientific Reports author/submission guidelines.",
        parent=parent_node,
        critical=False
    )
    claim_disp = "Scientific Reports author/submission guidelines limit the total number of display items (figures and tables combined) to no more than 8."
    await evaluator.verify(
        claim=claim_disp,
        node=node_disp,
        sources=ex.display_items_limit.sources,
        additional_instruction="Explicit wording for combined figures+tables should be present. If unclear or a different number is shown, mark Incorrect."
    )

    # Figure legend limit (non-critical)
    node_fig_leg = evaluator.add_leaf(
        id="Figure_Legend_Limit_Documentation",
        desc="States the figure-legend word limit (≤350 words per legend) and cites the Scientific Reports author/submission guidelines.",
        parent=parent_node,
        critical=False
    )
    claim_fig_leg = "Scientific Reports author/submission guidelines state that each figure legend should be at most 350 words."
    await evaluator.verify(
        claim=claim_fig_leg,
        node=node_fig_leg,
        sources=ex.figure_legend_limit.sources,
        additional_instruction="Verify per-legend word count limits; allow phrasing variants like 'no more than 350 words per figure legend'."
    )

    # References guideline (non-critical)
    node_refs = evaluator.add_leaf(
        id="References_Limit_Documentation",
        desc="States the references guideline (approximately 60) and cites the Scientific Reports author/submission guidelines.",
        parent=parent_node,
        critical=False
    )
    claim_refs = "Scientific Reports author/submission guidelines indicate an references count guideline of approximately 60 references."
    await evaluator.verify(
        claim=claim_refs,
        node=node_refs,
        sources=ex.references_guideline.sources,
        additional_instruction="Treat 'approximately', 'around', or similar wording as supportive of ~60 references."
    )

    # Data Availability Statement (critical)
    node_das = evaluator.add_leaf(
        id="Data_Availability_Statement_Documentation",
        desc="States that a Data Availability Statement is required for all manuscripts and cites the Scientific Reports author/submission guidelines.",
        parent=parent_node,
        critical=True
    )
    claim_das = "Scientific Reports requires a Data Availability Statement (DAS) for all manuscript submissions."
    await evaluator.verify(
        claim=claim_das,
        node=node_das,
        sources=ex.data_availability_required.sources,
        additional_instruction="Confirm the requirement applies to all submissions; if only optional or conditional wording is shown, mark Incorrect."
    )

    # Competing Interests (critical)
    node_ci = evaluator.add_leaf(
        id="Competing_Interests_Documentation",
        desc="States that a Competing Interests statement is required (explicit for each author) and cites the Scientific Reports author/submission guidelines.",
        parent=parent_node,
        critical=True
    )
    claim_ci = "Scientific Reports requires a Competing Interests statement for each author, including explicit declarations of any competing interests or a statement of none."
    await evaluator.verify(
        claim=claim_ci,
        node=node_ci,
        sources=ex.competing_interests_required.sources,
        additional_instruction="Look for explicit requirement language; statements must cover each author."
    )

    # Author Contributions (critical)
    node_ac = evaluator.add_leaf(
        id="Author_Contributions_Documentation",
        desc="States that an Author Contributions statement is required and cites the Scientific Reports author/submission guidelines.",
        parent=parent_node,
        critical=True
    )
    claim_ac = "Scientific Reports requires an Author Contributions statement in submissions."
    await evaluator.verify(
        claim=claim_ac,
        node=node_ac,
        sources=ex.author_contributions_required.sources,
        additional_instruction="Check for mandatory language regarding author contributions statements."
    )

    # Cover letter requirements (critical)
    node_cl = evaluator.add_leaf(
        id="Cover_Letter_Requirements_Documentation",
        desc="States required cover-letter components (corresponding author affiliation/contact; suitability for the journal; suggested reviewers) and cites the Scientific Reports author/submission guidelines.",
        parent=parent_node,
        critical=True
    )
    claim_cl = (
        "Scientific Reports' submission instructions indicate that the cover letter must include the corresponding author's affiliation and contact information, "
        "a brief statement explaining suitability for the journal, and suggested reviewers."
    )
    await evaluator.verify(
        claim=claim_cl,
        node=node_cl,
        sources=ex.cover_letter_requirements.sources,
        additional_instruction="Verify that the cover letter expectations include all three components; if any are missing or optional-only, mark Incorrect."
    )

    # Ethics approval documentation (critical)
    node_ethics = evaluator.add_leaf(
        id="Ethics_Approval_Documentation",
        desc="States the conditional ethics requirement: an ethics approval statement must be included in Methods when research involves human subjects or animals (and may explicitly note non-applicability if neither applies), citing the Scientific Reports author/submission guidelines.",
        parent=parent_node,
        critical=True
    )
    claim_ethics = (
        "Scientific Reports requires that when research involves human participants or animals, an ethics approval statement must be included in the Methods section; "
        "if neither applies, this may be explicitly noted as not applicable."
    )
    await evaluator.verify(
        claim=claim_ethics,
        node=node_ethics,
        sources=ex.ethics_approval_requirements.sources,
        additional_instruction="Look for conditional ethics requirements in Methods; explicit non-applicability statements are acceptable when neither human nor animal research is involved."
    )

    # Funding compliance: DMS plan (critical)
    node_dms = evaluator.add_leaf(
        id="Funding_Compliance_DMS_Plan_Documentation",
        desc="States the requirement to address a Data Management and Sharing Plan as specified in the constraints and cites an applicable authoritative federal funding policy source.",
        parent=parent_node,
        critical=True
    )
    claim_dms = (
        "For NASA-funded research, the submission and compliance should address a Data Management and Sharing Plan in accordance with applicable federal funding policy."
    )
    await evaluator.verify(
        claim=claim_dms,
        node=node_dms,
        sources=ex.funding_dms_plan.sources,
        additional_instruction="Accept authoritative federal sources such as NASA policy pages or OSTP/agency guidance that explicitly require data management/sharing planning for federally funded research."
    )

    # Funding compliance: immediate public access by end of 2025 (critical)
    node_pub = evaluator.add_leaf(
        id="Funding_Compliance_Immediate_Public_Access_2025_Documentation",
        desc="States the requirement that by end of 2025 federally funded research must be immediately publicly available (per constraints) and cites an applicable authoritative federal policy source.",
        parent=parent_node,
        critical=True
    )
    claim_pub = (
        "By the end of 2025, federally funded research articles are required to be made immediately publicly accessible under applicable federal policy."
    )
    await evaluator.verify(
        claim=claim_pub,
        node=node_pub,
        sources=ex.funding_public_access_2025.sources,
        additional_instruction="Accept authoritative federal policy sources (e.g., OSTP public access guidance) that specify immediate public access timelines by end of 2025."
    )

    # Research duration accuracy (critical) – verify against answer/prompt context
    node_duration = evaluator.add_leaf(
        id="Research_Duration_Accuracy",
        desc="Verifies the manuscript reports the study duration exactly as specified: 28 hours of recordings over two Martian years (1,374 Earth days).",
        parent=parent_node,
        critical=True
    )
    claim_duration = (
        "The answer explicitly states that the study analyzed 28 hours of microphone recordings collected over two Martian years (1,374 Earth days)."
    )
    await evaluator.verify(
        claim=claim_duration,
        node=node_duration,
        sources=None,
        additional_instruction="Check the answer text: does it include exactly '28 hours', 'two Martian years', and '1,374 Earth days'? Use the task description as ground truth context."
    )

    # Event count accuracy (critical)
    node_events = evaluator.add_leaf(
        id="Event_Count_Accuracy",
        desc="Verifies the manuscript reports the number of events exactly as specified: 55 instances of electrical discharges.",
        parent=parent_node,
        critical=True
    )
    claim_events = "The answer explicitly reports that 55 instances of electrical discharges were documented."
    await evaluator.verify(
        claim=claim_events,
        node=node_events,
        sources=None,
        additional_instruction="Check the answer text for '55' and the phrase 'electrical discharges'. Use the task description as ground truth."
    )

    # Event association accuracy (critical)
    node_assoc = evaluator.add_leaf(
        id="Event_Association_Accuracy",
        desc="Verifies the manuscript describes the events as associated with dust devils and dust storm fronts.",
        parent=parent_node,
        critical=True
    )
    claim_assoc = "The answer explicitly states that the documented electrical discharges were associated with dust devils and dust storm fronts."
    await evaluator.verify(
        claim=claim_assoc,
        node=node_assoc,
        sources=None,
        additional_instruction="Check the answer text for both terms: 'dust devils' and 'dust storm fronts'."
    )

    # Instrument identification accuracy (critical)
    node_instr = evaluator.add_leaf(
        id="Instrument_Identification_Accuracy",
        desc="Verifies the manuscript correctly identifies the instrument and platform: SuperCam on NASA's Perseverance rover.",
        parent=parent_node,
        critical=True
    )
    claim_instr = "The answer correctly identifies the instrument and platform as the SuperCam instrument on NASA's Perseverance rover."
    await evaluator.verify(
        claim=claim_instr,
        node=node_instr,
        sources=None,
        additional_instruction="Check that both 'SuperCam' and 'Perseverance rover' are explicitly present in the answer."
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
    Evaluate an answer for Scientific Reports submission requirements documentation and federal compliance checks.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # as specified in rubric
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

    # Add top-level node from rubric
    main_node = evaluator.add_parallel(
        id="Manuscript_Requirements_Documentation",
        desc=("Documentation identifies the specified Scientific Reports submission requirements and any listed federal-funding compliance requirements, "
              "stating each criterion/limit and citing an appropriate authoritative source (Scientific Reports author guidelines for journal requirements; "
              "applicable federal policy documents for funding/public-access requirements; prompt text may be cited for the provided study-detail verification)."),
        parent=root,
        critical=False
    )

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=ManuscriptRequirementsExtraction,
        extraction_name="requirements_extraction"
    )

    # Add ground truth info for rubric constants
    evaluator.add_ground_truth({
        "expected_limits": EXPECTED_LIMITS,
        "research_expected": RESEARCH_EXPECTED
    }, gt_type="rubric_expected_values")

    # Build and run verifications
    await build_and_verify_requirements(evaluator, main_node, extraction)

    # Return summary
    return evaluator.get_summary()