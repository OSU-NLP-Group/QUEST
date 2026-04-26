import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "four_uni_iss_experiments"
TASK_DESCRIPTION = """Identify four distinct university-led experiments that have been conducted on the International Space Station between 2020 and 2026. For each experiment, provide the following information:

1. Institution Information: Name of the U.S. university and the specific department or college
2. Principal Investigator: Full name and faculty title (e.g., professor, research professor) of the lead researcher
3. Timeline: Specific launch date, operational period, or completion date of the experiment
4. Research Domain: Classification as either physical sciences or life sciences, along with a brief description of the research objectives
5. Publication Status (if applicable): For completed experiments that have published results, provide the number of peer-reviewed publications or specific publication references

Each experiment must meet the following criteria:
- Led by a principal investigator who is a faculty member at a U.S.-based university
- Operational on or launched to the ISS between 2020 and 2026
- In the physical sciences or life sciences research domain
- Has publicly documented information available from official university news sources, NASA sources, or ISS National Lab sources

For each piece of information provided, include a reference URL that supports the claim.
"""


# ----------------------------- Data Models --------------------------------- #
class ExperimentInstitution(BaseModel):
    university: Optional[str] = None
    department: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ExperimentPI(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ExperimentTimeline(BaseModel):
    detail: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ExperimentDomain(BaseModel):
    classification: Optional[str] = None  # Expected values: "physical sciences" or "life sciences"
    objective: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ExperimentPublication(BaseModel):
    publication_count: Optional[str] = None  # Keep as string to allow ranges or textual notes
    references_text: Optional[str] = None    # Text listing references or notes
    urls: List[str] = Field(default_factory=list)


class ExperimentItem(BaseModel):
    experiment_name: Optional[str] = None
    institution: ExperimentInstitution = Field(default_factory=ExperimentInstitution)
    pi: ExperimentPI = Field(default_factory=ExperimentPI)
    timeline: ExperimentTimeline = Field(default_factory=ExperimentTimeline)
    domain: ExperimentDomain = Field(default_factory=ExperimentDomain)
    publication: ExperimentPublication = Field(default_factory=ExperimentPublication)


class ExperimentsExtraction(BaseModel):
    experiments: List[ExperimentItem] = Field(default_factory=list)


# -------------------------- Extraction Prompt ------------------------------ #
def prompt_extract_experiments() -> str:
    return """
    Extract up to four distinct university-led ISS experiments mentioned in the answer (only use the answer text).
    For each experiment, extract the following fields exactly as presented in the answer, along with the related URLs cited in the answer:

    For each experiment (in the order they appear), extract an object with:
    - experiment_name: Name of the experiment or project (if provided)
    - institution:
        - university: Name of the U.S.-based university that leads the experiment
        - department: Name of the specific department or college (e.g., 'College of Engineering', 'Department of Biology')
        - urls: All URLs in the answer that support the institution/department information
    - pi:
        - name: Full name of the principal investigator (PI)
        - title: Faculty title (e.g., 'Professor', 'Associate Professor', 'Research Professor', etc.)
        - urls: All URLs in the answer that support the PI information (e.g., university profile page, press release)
    - timeline:
        - detail: Specific launch date, operational period, or completion date as described in the answer (keep as free text)
        - urls: All URLs that support the timeline (e.g., NASA launches, mission updates, university news)
    - domain:
        - classification: Either 'physical sciences' or 'life sciences' (use exactly one of these two strings if specified in the answer; otherwise keep as null)
        - objective: Brief description of the research objectives (one or two sentences summarized from the answer)
        - urls: All URLs that support the domain/objective information
    - publication:
        - publication_count: Number of peer-reviewed publications (if provided; keep as text, e.g., '3', 'at least 2', or null if not specified)
        - references_text: Specific publication references or citation notes (if any; otherwise null)
        - urls: All URLs that support publication information

    Rules:
    - Extract only information explicitly present in the answer.
    - For any missing field, set it to null.
    - For each 'urls' array, include only valid URLs explicitly present in the answer (including markdown links).
    - Return exactly up to four experiments (truncate if more than four are present). If fewer than four are present, return only those found.
    """


# ------------------------------ Helpers ------------------------------------ #
def _first_n_experiments(extraction: ExperimentsExtraction, n: int = 4) -> List[ExperimentItem]:
    items = extraction.experiments[:n]
    # Pad with empty placeholders if fewer than n
    while len(items) < n:
        items.append(ExperimentItem())
    return items


def _nonempty_text(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _combine_sources(*lists: List[str]) -> List[str]:
    combined_set = set()
    for lst in lists:
        for url in lst:
            if _nonempty_text(url):
                combined_set.add(url.strip())
    return list(combined_set)


# -------------------------- Verification Logic ----------------------------- #
async def verify_single_experiment(
    evaluator: Evaluator,
    parent_node,
    exp: ExperimentItem,
    idx: int
) -> None:
    exp_label = f"Experiment_{idx + 1}"

    exp_node = evaluator.add_parallel(
        id=exp_label,
        desc=f"{['First','Second','Third','Fourth'][idx]} university-led ISS experiment meeting all criteria",
        parent=parent_node,
        critical=False
    )

    # Institution Info (critical group)
    inst_info_node = evaluator.add_parallel(
        id=f"Exp{idx + 1}_Institution_Info",
        desc=f"Institution and department information for Experiment {idx + 1}",
        parent=exp_node,
        critical=True
    )

    inst_content_node = evaluator.add_parallel(
        id=f"Exp{idx + 1}_Institution_Content",
        desc=f"Core institution and department details for Experiment {idx + 1}",
        parent=inst_info_node,
        critical=True
    )

    # Leaf: US-based university (verify with sources)
    inst_us_leaf = evaluator.add_leaf(
        id=f"Exp{idx + 1}_US_University",
        desc=f"Experiment {idx + 1} is led by a U.S.-based university",
        parent=inst_content_node,
        critical=True
    )
    claim_us = f"The experiment is led by a U.S.-based university named '{exp.institution.university}'."
    await evaluator.verify(
        claim=claim_us,
        node=inst_us_leaf,
        sources=exp.institution.urls,
        additional_instruction=(
            "Check the cited source(s) to confirm the institution is a U.S.-based university. "
            "Evidence may include a .edu domain, references to a U.S. city/state, or explicit statements. "
            "If university name is missing or not U.S.-based, mark as not supported."
        )
    )

    # Leaf: Department specified (existence check)
    dept_exists = _nonempty_text(exp.institution.department)
    evaluator.add_custom_node(
        result=dept_exists,
        id=f"Exp{idx + 1}_Dept_Specified",
        desc=f"Department or college name is explicitly provided for Experiment {idx + 1}",
        parent=inst_content_node,
        critical=True
    )

    # Leaf: Institution URL provided (existence of URLs)
    inst_urls_exist = len(exp.institution.urls) > 0
    evaluator.add_custom_node(
        result=inst_urls_exist,
        id=f"Exp{idx + 1}_Institution_URL",
        desc=f"URL reference supporting institution information for Experiment {idx + 1}",
        parent=inst_info_node,
        critical=True
    )

    # PI Info (critical group)
    pi_info_node = evaluator.add_parallel(
        id=f"Exp{idx + 1}_PI_Info",
        desc=f"Principal investigator information for Experiment {idx + 1}",
        parent=exp_node,
        critical=True
    )
    pi_content_node = evaluator.add_parallel(
        id=f"Exp{idx + 1}_PI_Content",
        desc=f"Core principal investigator details for Experiment {idx + 1}",
        parent=pi_info_node,
        critical=True
    )

    # Leaf: PI named
    pi_named_leaf = evaluator.add_leaf(
        id=f"Exp{idx + 1}_PI_Named",
        desc=f"Principal investigator is identified by name for Experiment {idx + 1}",
        parent=pi_content_node,
        critical=True
    )
    claim_pi_named = f"The principal investigator (PI) for this experiment is '{exp.pi.name}'."
    await evaluator.verify(
        claim=claim_pi_named,
        node=pi_named_leaf,
        sources=_combine_sources(exp.pi.urls, exp.institution.urls),
        additional_instruction=(
            "Verify that the cited source(s) explicitly name the PI for the experiment as given. "
            "Use university news, NASA/ISS announcements, or official profile pages."
        )
    )

    # Leaf: PI holds faculty position
    pi_faculty_leaf = evaluator.add_leaf(
        id=f"Exp{idx + 1}_PI_Faculty",
        desc=f"Principal investigator holds a faculty position (professor, research professor, or equivalent) for Experiment {idx + 1}",
        parent=pi_content_node,
        critical=True
    )
    claim_pi_faculty = (
        f"The principal investigator '{exp.pi.name}' holds a faculty position ({exp.pi.title}) at "
        f"'{exp.institution.university}'."
    )
    await evaluator.verify(
        claim=claim_pi_faculty,
        node=pi_faculty_leaf,
        sources=_combine_sources(exp.pi.urls, exp.institution.urls),
        additional_instruction=(
            "Verify that the PI holds a faculty appointment (e.g., professor, associate professor, assistant professor, "
            "research professor, or equivalent) at a U.S. university."
        )
    )

    # Leaf: PI URLs exist
    pi_urls_exist = len(exp.pi.urls) > 0
    evaluator.add_custom_node(
        result=pi_urls_exist,
        id=f"Exp{idx + 1}_PI_URL",
        desc=f"URL reference supporting PI information for Experiment {idx + 1}",
        parent=pi_info_node,
        critical=True
    )

    # Timeline Info (critical group)
    timeline_info_node = evaluator.add_parallel(
        id=f"Exp{idx + 1}_Timeline_Info",
        desc=f"Timeline and operational period for Experiment {idx + 1}",
        parent=exp_node,
        critical=True
    )
    timeline_content_node = evaluator.add_parallel(
        id=f"Exp{idx + 1}_Timeline_Content",
        desc=f"Core timeline details for Experiment {idx + 1}",
        parent=timeline_info_node,
        critical=True
    )

    # Leaf: Period 2020–2026
    period_leaf = evaluator.add_leaf(
        id=f"Exp{idx + 1}_Period_2020_2026",
        desc=f"Experiment {idx + 1} was operational on or launched to ISS between 2020-2026",
        parent=timeline_content_node,
        critical=True
    )
    claim_period = (
        "This experiment was operational on or launched to the International Space Station between 2020 and 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_period,
        node=period_leaf,
        sources=exp.timeline.urls,
        additional_instruction=(
            "Use the cited source(s) to confirm the timeline falls within Jan 1, 2020 to Dec 31, 2026. "
            "Accept explicit launch dates, mission dates, or completion dates tied to ISS operations."
        )
    )

    # Leaf: Specific timeline provided (verify text matches/support in sources)
    specific_timeline_leaf = evaluator.add_leaf(
        id=f"Exp{idx + 1}_Specific_Timeline",
        desc=f"Specific launch date, operational period, or completion date is provided for Experiment {idx + 1}",
        parent=timeline_content_node,
        critical=True
    )
    claim_timeline_detail = (
        f"The specific timeline details for the experiment include: {exp.timeline.detail}."
    )
    await evaluator.verify(
        claim=claim_timeline_detail,
        node=specific_timeline_leaf,
        sources=exp.timeline.urls,
        additional_instruction=(
            "Check that the cited source(s) explicitly support the timeline detail provided (launch date, operational period, "
            "or completion date) and that it refers to ISS operations."
        )
    )

    # Leaf: Timeline URLs exist
    timeline_urls_exist = len(exp.timeline.urls) > 0
    evaluator.add_custom_node(
        result=timeline_urls_exist,
        id=f"Exp{idx + 1}_Timeline_URL",
        desc=f"URL reference supporting timeline information for Experiment {idx + 1}",
        parent=timeline_info_node,
        critical=True
    )

    # Research Domain (critical group)
    domain_node = evaluator.add_parallel(
        id=f"Exp{idx + 1}_Research_Domain",
        desc=f"Research domain classification for Experiment {idx + 1}",
        parent=exp_node,
        critical=True
    )
    domain_content_node = evaluator.add_parallel(
        id=f"Exp{idx + 1}_Domain_Content",
        desc=f"Core research domain details for Experiment {idx + 1}",
        parent=domain_node,
        critical=True
    )

    # Leaf: Physical or Life Sciences classification
    domain_class_leaf = evaluator.add_leaf(
        id=f"Exp{idx + 1}_Physical_or_Life_Science",
        desc=f"Experiment {idx + 1} is in physical sciences or life sciences domain",
        parent=domain_content_node,
        critical=True
    )
    claim_domain_class = (
        f"This experiment is classified under the '{exp.domain.classification}' sciences."
    )
    await evaluator.verify(
        claim=claim_domain_class,
        node=domain_class_leaf,
        sources=exp.domain.urls,
        additional_instruction=(
            "Verify that the source(s) support classification as either 'physical sciences' or 'life sciences'. "
            "If sources indicate a different domain or classification is missing, mark as not supported."
        )
    )

    # Leaf: Research objective described
    domain_obj_leaf = evaluator.add_leaf(
        id=f"Exp{idx + 1}_Research_Objective",
        desc=f"Research objectives or goals are described for Experiment {idx + 1}",
        parent=domain_content_node,
        critical=True
    )
    claim_objective = f"The experiment’s research objectives include: {exp.domain.objective}."
    await evaluator.verify(
        claim=claim_objective,
        node=domain_obj_leaf,
        sources=exp.domain.urls,
        additional_instruction=(
            "Verify that the cited source(s) describe the experiment's objectives consistent with the provided text."
        )
    )

    # Leaf: Domain URLs exist
    domain_urls_exist = len(exp.domain.urls) > 0
    evaluator.add_custom_node(
        result=domain_urls_exist,
        id=f"Exp{idx + 1}_Domain_URL",
        desc=f"URL reference supporting research domain information for Experiment {idx + 1}",
        parent=domain_node,
        critical=True
    )

    # Publication Status (non-critical group)
    pub_node = evaluator.add_parallel(
        id=f"Exp{idx + 1}_Publication_Status",
        desc=f"Publication status and research outputs for Experiment {idx + 1}",
        parent=exp_node,
        critical=False
    )

    # Leaf: Publication info (verify if provided)
    pub_info_leaf = evaluator.add_leaf(
        id=f"Exp{idx + 1}_Publication_Info",
        desc=f"For completed experiments with published results: peer-reviewed publication count or specific publication references are provided for Experiment {idx + 1}",
        parent=pub_node,
        critical=False
    )
    if _nonempty_text(exp.publication.publication_count) or _nonempty_text(exp.publication.references_text):
        claim_pub = (
            f"The experiment has published results with count '{exp.publication.publication_count}' "
            f"or references: {exp.publication.references_text}."
        )
    else:
        claim_pub = "No publication information is provided."
    await evaluator.verify(
        claim=claim_pub,
        node=pub_info_leaf,
        sources=exp.publication.urls,
        additional_instruction=(
            "If publication info is provided, verify that the cited source(s) support the stated count or references. "
            "If no publication info is provided, this should not be considered a failure (non-critical)."
        )
    )

    # Leaf: Publication URL existence (non-critical)
    pub_urls_exist = len(exp.publication.urls) > 0
    evaluator.add_custom_node(
        result=pub_urls_exist,
        id=f"Exp{idx + 1}_Publication_URL",
        desc=f"URL reference supporting publication information for Experiment {idx + 1}",
        parent=pub_node,
        critical=False
    )


# -------------------------- Main Evaluation Entry -------------------------- #
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

    # Add a top-level task node (non-critical to allow partial scoring across experiments)
    task_node = evaluator.add_parallel(
        id="Four_University_ISS_Experiments",
        desc="Identify four distinct university-led ISS experiments that meet all specified criteria",
        parent=root,
        critical=False
    )

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_experiments(),
        template_class=ExperimentsExtraction,
        extraction_name="four_iss_experiments_extraction"
    )

    # Select first four experiments and pad if needed
    experiments = _first_n_experiments(extraction, 4)

    # Build verification subtree for each experiment
    for i, exp in enumerate(experiments):
        await verify_single_experiment(evaluator, task_node, exp, i)

    return evaluator.get_summary()