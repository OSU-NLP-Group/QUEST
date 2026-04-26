import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "polaris_dawn_trish_study"
TASK_DESCRIPTION = """
Identify one human health or physiology research study from the Polaris Dawn mission (September 10–15, 2024) that meets ALL of the following criteria:
(1) The study was supported by TRISH (Translational Research Institute for Space Health, “TRISH”),
(2) The study involved multi-phase data collection across pre-flight, in-flight, and post-flight phases,
(3) The study focused on human health or physiological adaptation to spaceflight.

For the identified study, provide:
A. The study name and research focus,
B. The principal investigator's name and their affiliated institution,
C. A description of the study methodology and objectives,
D. Documentation of ethics and IRB compliance requirements,
E. Verification that the study meets academic publication requirements, including: peer review capability, data availability requirements (e.g., NASA SPD-41a or relevant institutional policies), institutional documentation standards, and abstract and formatting requirements for space research journals.

Support the answer with reference URLs from official sources documenting the Polaris Dawn mission research portfolio, institutional affiliations, and publication guidelines.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PublicationReadinessInfo(BaseModel):
    peer_review_statement: Optional[str] = None
    peer_review_urls: List[str] = Field(default_factory=list)

    data_availability_statement: Optional[str] = None
    data_availability_urls: List[str] = Field(default_factory=list)

    institutional_standards_statement: Optional[str] = None
    institutional_standards_urls: List[str] = Field(default_factory=list)

    abstract_formatting_statement: Optional[str] = None
    abstract_formatting_urls: List[str] = Field(default_factory=list)


class OfficialRefs(BaseModel):
    study_trish_url: Optional[str] = None  # official URL that documents study inclusion/TRISH support
    pi_institution_url: Optional[str] = None  # official URL for PI identity and institution
    publication_guidelines_urls: List[str] = Field(default_factory=list)  # official guideline/policy/journal URLs


class SelectedStudy(BaseModel):
    # Core identity
    study_name: Optional[str] = None
    research_focus: Optional[str] = None

    # Mission association
    mission: Optional[str] = None
    mission_dates: Optional[str] = None
    mission_association_urls: List[str] = Field(default_factory=list)

    # TRISH support
    trish_support_statement: Optional[str] = None
    trish_support_urls: List[str] = Field(default_factory=list)

    # Human health/physiology focus
    human_focus_statement: Optional[str] = None
    human_focus_urls: List[str] = Field(default_factory=list)

    # Multi-phase collection
    phases_statement: Optional[str] = None
    phases: List[str] = Field(default_factory=list)  # e.g., ["pre-flight", "in-flight", "post-flight"]
    multi_phase_urls: List[str] = Field(default_factory=list)

    # PI and institution
    pi_name: Optional[str] = None
    pi_institution: Optional[str] = None
    pi_urls: List[str] = Field(default_factory=list)

    # Methods & objectives
    methodology_description: Optional[str] = None
    objectives_description: Optional[str] = None
    methods_urls: List[str] = Field(default_factory=list)

    # Ethics / IRB
    ethics_irb_statement: Optional[str] = None
    ethics_irb_urls: List[str] = Field(default_factory=list)

    # Publication readiness bundle
    publication: PublicationReadinessInfo = PublicationReadinessInfo()

    # Official references required by prompt
    official_refs: OfficialRefs = OfficialRefs()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_selected_study() -> str:
    return """
Extract exactly ONE study (the single study the answer selects) that the answer claims meets ALL selection constraints for the Polaris Dawn mission (Sep 10–15, 2024). Extract only what is explicitly present in the answer text and its listed URLs. Do not invent information.

Return a JSON object with the following fields:

- study_name: string | null
- research_focus: string | null

- mission: string | null       # e.g., "Polaris Dawn"
- mission_dates: string | null # e.g., "September 10–15, 2024"
- mission_association_urls: string[]          # URLs that show this study is in Polaris Dawn portfolio

- trish_support_statement: string | null
- trish_support_urls: string[]                # URLs that explicitly indicate TRISH support

- human_focus_statement: string | null
- human_focus_urls: string[]                  # URLs that show human health/physiology focus

- phases_statement: string | null
- phases: string[]                            # e.g., ["pre-flight","in-flight","post-flight"] if present in answer
- multi_phase_urls: string[]                  # URLs supporting multi-phase collection across pre-/in-/post-flight

- pi_name: string | null
- pi_institution: string | null
- pi_urls: string[]                           # URLs supporting PI identity and affiliation

- methodology_description: string | null
- objectives_description: string | null
- methods_urls: string[]                      # URLs describing methodology/objectives

- ethics_irb_statement: string | null
- ethics_irb_urls: string[]                   # URLs documenting ethics/IRB or equivalent compliance

- publication: {
    peer_review_statement: string | null
    peer_review_urls: string[]                # official journal "instructions for authors" or policy pages

    data_availability_statement: string | null
    data_availability_urls: string[]          # e.g., NASA SPD-41a or institutional data policy

    institutional_standards_statement: string | null
    institutional_standards_urls: string[]    # institutional documentation/affiliations standards/policies

    abstract_formatting_statement: string | null
    abstract_formatting_urls: string[]        # journal or policy pages with abstract length/format info
  }

- official_refs: {
    study_trish_url: string | null                   # an official URL that shows study in Polaris Dawn &/or TRISH support
    pi_institution_url: string | null                # an official URL for PI identity & institution
    publication_guidelines_urls: string[]            # official publication guidelines/requirements
}

SPECIAL RULES:
- Extract only URLs explicitly present in the answer. If absent, return empty arrays or nulls.
- Allow URLs to be in plain form or markdown links—extract the actual link targets.
- Do not normalize or infer phases—only list phases actually named in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    merged.append(uu)
                    seen.add(uu)
    return merged


async def _verify_leaf_with_urls(
    evaluator: Evaluator,
    *,
    id: str,
    desc: str,
    parent,
    claim: str,
    urls: List[str],
    critical: bool = True,
    additional_instruction: str = "None"
):
    """
    Create a rubric leaf (or a failed custom node if sources are missing) and run URL-grounded verification.
    Ensures source-grounding: when no URLs are available, directly fail this critical leaf.
    """
    if not urls:
        evaluator.add_custom_node(
            result=False,
            id=id,
            desc=f"{desc} (failed: no supporting URLs provided in the answer)",
            parent=parent,
            critical=critical,
        )
        return

    node = evaluator.add_leaf(
        id=id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls if len(urls) > 1 else urls[0],
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def _build_study_selection_criteria(
    evaluator: Evaluator,
    parent,
    study: SelectedStudy,
):
    """
    Study_Selection_Criteria (critical, parallel)
      - Polaris_Dawn_Mission_and_Dates (critical leaf)
      - TRISH_Support (critical leaf)
      - Human_Health_or_Physiology_Focus (critical leaf)
      - Multi_Phase_Data_Collection (critical leaf)
    """
    select_node = evaluator.add_parallel(
        id="Study_Selection_Criteria",
        desc="Selected study satisfies all selection constraints.",
        parent=parent,
        critical=True
    )

    # Polaris Dawn association (do not require the page to repeat dates verbatim)
    polaris_urls = _merge_urls(study.mission_association_urls, [study.official_refs.study_trish_url] if study.official_refs and study.official_refs.study_trish_url else [])
    await _verify_leaf_with_urls(
        evaluator,
        id="Polaris_Dawn_Mission_and_Dates",
        desc="Study is explicitly associated with the Polaris Dawn mission (Sep 10–15, 2024).",
        parent=select_node,
        claim="This source confirms that the study is associated with the Polaris Dawn mission.",
        urls=polaris_urls,
        additional_instruction="Mission association alone is sufficient; the page need not repeat the exact flight dates. Verify that the study is clearly linked to Polaris Dawn.",
    )

    # TRISH support
    trish_urls = _merge_urls(study.trish_support_urls, [study.official_refs.study_trish_url] if study.official_refs and study.official_refs.study_trish_url else [])
    await _verify_leaf_with_urls(
        evaluator,
        id="TRISH_Support",
        desc="Study is explicitly documented as supported by TRISH (Translational Research Institute for Space Health).",
        parent=select_node,
        claim="This source states or clearly indicates that the study is supported by TRISH (Translational Research Institute for Space Health).",
        urls=trish_urls,
        additional_instruction="Accept references to TRISH (the NASA-funded Translational Research Institute for Space Health).",
    )

    # Human health / physiology focus
    human_urls = _merge_urls(study.human_focus_urls, [study.official_refs.study_trish_url] if study.official_refs and study.official_refs.study_trish_url else [], study.methods_urls)
    await _verify_leaf_with_urls(
        evaluator,
        id="Human_Health_or_Physiology_Focus",
        desc="Study focus is human health or physiological adaptation to spaceflight.",
        parent=select_node,
        claim="This source shows the study focuses on human health or physiological adaptation to spaceflight.",
        urls=human_urls,
        additional_instruction="Allow synonyms such as 'biomedical', 'physiology', 'human performance', 'cardiovascular', 'neurovestibular', etc.",
    )

    # Multi-phase data collection
    multiphase_urls = _merge_urls(study.multi_phase_urls, study.methods_urls)
    await _verify_leaf_with_urls(
        evaluator,
        id="Multi_Phase_Data_Collection",
        desc="Study includes pre-flight, in-flight, and post-flight data collection phases.",
        parent=select_node,
        claim="This source confirms that the study includes data collection across all three phases: pre-flight, in-flight, and post-flight.",
        urls=multiphase_urls,
        additional_instruction="Accept synonymous phrasing: 'pre-mission', 'during mission/in-flight', 'post-mission/follow-up'. All three phases must be present.",
    )


async def _build_required_output_section(
    evaluator: Evaluator,
    parent,
    study: SelectedStudy,
):
    """
    Required_Output_For_Selected_Study (critical, parallel)
      - Study_Name_and_Research_Focus (critical leaf)
      - Principal_Investigator_and_Affiliated_Institution (critical leaf)
      - Methodology_and_Objectives (critical leaf)
      - Ethics_and_IRB_Compliance (critical leaf)
      - Academic_Publication_Readiness (critical, parallel)
           - Peer_Review_Capability_Addressed (critical leaf)
           - Data_Availability_Requirements_Addressed (critical leaf)
           - Institutional_Documentation_Standards_Addressed (critical leaf)
           - Abstract_and_Formatting_Requirements_Addressed (critical leaf)
      - Official_Source_References (critical, parallel)
           - Official_URL_for_Study_and_TRISH_Support (critical leaf)
           - Official_URL_for_PI_and_Institution (critical leaf)
           - Official_URL_for_Publication_Guidelines_or_Requirements (critical leaf)
    """
    req_node = evaluator.add_parallel(
        id="Required_Output_For_Selected_Study",
        desc="Provide all required fields for the selected study.",
        parent=parent,
        critical=True
    )

    # Study name and research focus
    name_focus_urls = _merge_urls(study.mission_association_urls, study.methods_urls, [study.official_refs.study_trish_url] if study.official_refs and study.official_refs.study_trish_url else [])
    study_name = study.study_name or ""
    research_focus = study.research_focus or ""
    await _verify_leaf_with_urls(
        evaluator,
        id="Study_Name_and_Research_Focus",
        desc="Provides the study name and research focus.",
        parent=req_node,
        claim=f"The official source identifies the study titled '{study_name}' and a research focus consistent with: '{research_focus}'.",
        urls=name_focus_urls,
        additional_instruction="Minor title variants or paraphrased focus are acceptable if clearly the same study.",
    )

    # PI and institution
    pi_urls_all = _merge_urls(study.pi_urls, [study.official_refs.pi_institution_url] if study.official_refs and study.official_refs.pi_institution_url else [], [study.official_refs.study_trish_url] if study.official_refs and study.official_refs.study_trish_url else [])
    pi_name = study.pi_name or ""
    pi_inst = study.pi_institution or ""
    await _verify_leaf_with_urls(
        evaluator,
        id="Principal_Investigator_and_Affiliated_Institution",
        desc="Identifies the principal investigator and their affiliated institution (must be identifiable).",
        parent=req_node,
        claim=f"The source confirms the principal investigator is {pi_name}, affiliated with {pi_inst}.",
        urls=pi_urls_all,
        additional_instruction="Allow common institution name variants and abbreviations (e.g., 'Univ.' vs 'University').",
    )

    # Methodology and objectives
    methodology = (study.methodology_description or "").strip()
    objectives = (study.objectives_description or "").strip()
    methods_urls = list(study.methods_urls)
    await _verify_leaf_with_urls(
        evaluator,
        id="Methodology_and_Objectives",
        desc="Describes the study methodology and objectives.",
        parent=req_node,
        claim=f"The official source describes methodology and objectives consistent with this summary: Methodology: {methodology[:400]}; Objectives: {objectives[:400]}",
        urls=methods_urls,
        additional_instruction="Do not require verbatim match; verify the page provides a methods/approach description and clear study aims/objectives aligning with the summary.",
    )

    # Ethics and IRB
    ethics_urls = list(study.ethics_irb_urls)
    await _verify_leaf_with_urls(
        evaluator,
        id="Ethics_and_IRB_Compliance",
        desc="Provides documentation/discussion of ethics and IRB (or equivalent) compliance requirements for human subjects research applicable to the study.",
        parent=req_node,
        claim="The provided source(s) document ethics and/or IRB (or equivalent) compliance requirements relevant to this human-subject study.",
        urls=ethics_urls,
        additional_instruction="Accept institutional IRB pages, official protocol notices, or equivalent compliance documentation; a clear connection to human-subjects oversight is required.",
    )

    # Academic Publication Readiness (critical, parallel)
    pub_node = evaluator.add_parallel(
        id="Academic_Publication_Readiness",
        desc="Addresses all requested publication-readiness elements.",
        parent=req_node,
        critical=True
    )

    # Peer review capability addressed
    peer_urls = _merge_urls(study.publication.peer_review_urls, study.official_refs.publication_guidelines_urls if study.official_refs else [])
    await _verify_leaf_with_urls(
        evaluator,
        id="Peer_Review_Capability_Addressed",
        desc="Addresses how the study can meet peer-review capability / be suitable for peer-reviewed publication.",
        parent=pub_node,
        claim="This source is an official journal/publisher/institution 'instructions for authors' or policy page relevant to preparing a manuscript for peer‑reviewed publication in space/biomedical research.",
        urls=peer_urls,
        additional_instruction="Look for indications like 'Instructions for Authors', 'Submission Guidelines', or equivalent official publication policy pages.",
    )

    # Data availability requirements addressed
    data_urls = _merge_urls(study.publication.data_availability_urls, study.official_refs.publication_guidelines_urls if study.official_refs else [])
    await _verify_leaf_with_urls(
        evaluator,
        id="Data_Availability_Requirements_Addressed",
        desc="Addresses data availability requirements per NASA SPD-41a or relevant institutional policies, as stated in the constraints.",
        parent=pub_node,
        claim="This source sets data availability/sharing requirements (e.g., NASA SPD‑41a or official institutional/journal data policy) applicable to publications from this study.",
        urls=data_urls,
        additional_instruction="Prefer NASA SPD‑41a or official institutional/journal data policies; reject blog posts or non-official sources.",
    )

    # Institutional documentation standards addressed
    inst_urls = _merge_urls(study.publication.institutional_standards_urls, [study.official_refs.pi_institution_url] if study.official_refs and study.official_refs.pi_institution_url else [])
    await _verify_leaf_with_urls(
        evaluator,
        id="Institutional_Documentation_Standards_Addressed",
        desc="Addresses institutional documentation standards, including that institutional affiliations are documented for all research team members (per constraints).",
        parent=pub_node,
        claim="This source is an official institutional policy/guidance indicating documentation/affiliation standards for authors/researchers.",
        urls=inst_urls,
        additional_instruction="Accept institutional authorship/affiliations policy pages or equivalent official documentation standards pages.",
    )

    # Abstract and formatting requirements addressed
    abstract_urls = _merge_urls(study.publication.abstract_formatting_urls, study.official_refs.publication_guidelines_urls if study.official_refs else [])
    await _verify_leaf_with_urls(
        evaluator,
        id="Abstract_and_Formatting_Requirements_Addressed",
        desc="Verifies capability to produce an abstract meeting typical journal requirements of 150–250 words and addresses formatting requirements for space research journals (per constraints).",
        parent=pub_node,
        claim="This source specifies abstract length and/or manuscript formatting requirements typical for space/biomedical journals (e.g., 150–250 word abstract).",
        urls=abstract_urls,
        additional_instruction="Look for explicit abstract word-count and formatting guidance in official 'Instructions for Authors' or equivalent pages.",
    )

    # Official Source References (critical, parallel)
    refs_node = evaluator.add_parallel(
        id="Official_Source_References",
        desc="Provides official-source reference URLs supporting required claims.",
        parent=req_node,
        critical=True
    )

    # Official URL for Study & TRISH support (single URL expected)
    study_trish_url = study.official_refs.study_trish_url if study.official_refs else None
    if study_trish_url:
        node = evaluator.add_leaf(
            id="Official_URL_for_Study_and_TRISH_Support",
            desc="Provides an official URL supporting the study’s inclusion in the Polaris Dawn research portfolio and TRISH support.",
            parent=refs_node,
            critical=True
        )
        await evaluator.verify(
            claim="This URL is an official page that documents the study within the Polaris Dawn research portfolio and/or acknowledges TRISH support.",
            node=node,
            sources=study_trish_url,
            additional_instruction="Accept official TRISH, Polaris Dawn, NASA, SpaceX, or partner institution pages that explicitly include the study and/or TRISH support.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Official_URL_for_Study_and_TRISH_Support",
            desc="Provides an official URL supporting the study’s inclusion in the Polaris Dawn research portfolio and TRISH support. (failed: no URL provided)",
            parent=refs_node,
            critical=True
        )

    # Official URL for PI and Institution (single URL expected)
    pi_inst_url = study.official_refs.pi_institution_url if study.official_refs else None
    if pi_inst_url:
        node = evaluator.add_leaf(
            id="Official_URL_for_PI_and_Institution",
            desc="Provides an official URL supporting the PI identity and affiliated institution.",
            parent=refs_node,
            critical=True
        )
        await evaluator.verify(
            claim="This URL is an official institution or lab profile page that confirms the PI identity and affiliated institution.",
            node=node,
            sources=pi_inst_url,
            additional_instruction="Prefer institution/lab/department profile pages. Reject social media or non-official directories.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Official_URL_for_PI_and_Institution",
            desc="Provides an official URL supporting the PI identity and affiliated institution. (failed: no URL provided)",
            parent=refs_node,
            critical=True
        )

    # Official URL(s) for Publication Guidelines or Requirements (multi-URL acceptable)
    pub_guidelines_urls = study.official_refs.publication_guidelines_urls if study.official_refs else []
    await _verify_leaf_with_urls(
        evaluator,
        id="Official_URL_for_Publication_Guidelines_or_Requirements",
        desc="Provides an official URL supporting the publication guideline/requirements claims (e.g., journal or institutional/NASA guidance relevant to the stated requirements).",
        parent=refs_node,
        claim="At least one of these URLs is an official publication guideline or policy relevant to peer review, data availability, abstract length, and/or formatting requirements.",
        urls=pub_guidelines_urls,
        additional_instruction="Accept official journal/publisher 'Instructions for Authors', NASA policies (e.g., SPD‑41a), or institutional publication policies.",
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Polaris Dawn TRISH-supported human health/physiology study task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # overall flow: selection criteria first, then outputs
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

    # Extract the selected study from the answer
    study: SelectedStudy = await evaluator.extract(
        prompt=prompt_extract_selected_study(),
        template_class=SelectedStudy,
        extraction_name="selected_study",
    )

    # Build main critical node mirroring rubric root
    main_node = evaluator.add_sequential(
        id="Study_Documentation_and_Publication_Readiness",
        desc="Identify one Polaris Dawn (Sep 10–15, 2024) human health/physiology study supported by TRISH with pre-, in-, and post-flight phases, and provide required study details, ethics/IRB documentation, publication-readiness elements, and official-source URLs.",
        parent=root,
        critical=True
    )

    # Part 1: Study selection criteria (critical, parallel)
    await _build_study_selection_criteria(evaluator, main_node, study)

    # Part 2: Required outputs for the selected study (critical, parallel)
    await _build_required_output_section(evaluator, main_node, study)

    # Return summary
    return evaluator.get_summary()