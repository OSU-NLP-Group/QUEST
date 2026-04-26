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
TASK_ID = "us_bci_groups_2024_2025"
TASK_DESCRIPTION = """
Identify three (3) distinct brain-computer interface (BCI) or neural interface research groups in the United States that were active in advancing BCI research in 2024-2025. For each research group, provide comprehensive information demonstrating they meet the following requirements:

1. Institutional Affiliation: The group must be affiliated with a U.S. university or research institution
2. Conference Participation: The group or its members participated in the Society for Neuroscience Annual Meeting 2025 (November 15-19, San Diego, CA)
3. High-Impact Publications: Published at least one peer-reviewed article in a neuroscience journal with impact factor ≥4.0 during 2024-2025
4. Research Focus: Primary research focus on brain-computer interfaces, neural interfaces, or neuroprosthetics
5. Clinical Engagement: Active involvement in or collaboration on human clinical trials/studies as of 2025
6. Federal Funding: Received NIH BRAIN Initiative, NSF, DARPA, or equivalent federal neuroscience funding (2023-2025)
7. Faculty Status: Principal investigator holds faculty position at Assistant Professor level or higher
8. Research Productivity: Published at least 3 peer-reviewed neuroscience articles in 2024-2025
9. Collaborative Research: Documented research collaborations with other institutions through co-authored publications
10. Technology Development: Develops or utilizes advanced neurotechnology for neural recording or stimulation
11. Research Team: Has multiple team members including postdoctoral fellows, graduate students, or research staff
12. Public Presence: Maintains an active lab website or institutional profile page

For each of the three research groups, provide:
- Principal Investigator name
- Affiliated institution
- Lab/group name
- Supporting reference URLs for each requirement
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class GroupURLs(BaseModel):
    us_affiliation: List[str] = Field(default_factory=list)
    sfn_2025: List[str] = Field(default_factory=list)
    research_focus: List[str] = Field(default_factory=list)
    faculty_status: List[str] = Field(default_factory=list)
    high_impact_publication: List[str] = Field(default_factory=list)
    research_productivity: List[str] = Field(default_factory=list)
    public_presence: List[str] = Field(default_factory=list)
    clinical_engagement: List[str] = Field(default_factory=list)
    federal_funding: List[str] = Field(default_factory=list)
    collaborative_research: List[str] = Field(default_factory=list)
    technology_development: List[str] = Field(default_factory=list)
    research_team: List[str] = Field(default_factory=list)


class ResearchGroup(BaseModel):
    pi_name: Optional[str] = None
    institution: Optional[str] = None
    lab_name: Optional[str] = None
    urls: GroupURLs = Field(default_factory=GroupURLs)


class GroupsExtraction(BaseModel):
    groups: List[ResearchGroup] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_groups() -> str:
    return """
You must extract up to THREE distinct U.S.-based brain-computer interface (BCI) or neural interface research groups exactly as presented in the answer. Do not infer any data not explicitly stated in the answer.

For each group, extract the following fields:
- pi_name: The principal investigator’s full name (string or null)
- institution: The affiliated institution (string or null)
- lab_name: The lab/group name (string or null)

- urls: An object of URL arrays for evidence supporting each requirement. Only include URLs explicitly provided in the answer. If a category is not supported by any URLs in the answer, return an empty array for that category.

The 'urls' object must have the following keys (each maps to an array of URLs):
- us_affiliation
- sfn_2025
- research_focus
- faculty_status
- high_impact_publication
- research_productivity
- public_presence
- clinical_engagement
- federal_funding
- collaborative_research
- technology_development
- research_team

Return a JSON object with:
{
  "groups": [
    {
      "pi_name": "...",
      "institution": "...",
      "lab_name": "...",
      "urls": {
        "us_affiliation": ["..."],
        "sfn_2025": ["..."],
        "research_focus": ["..."],
        "faculty_status": ["..."],
        "high_impact_publication": ["..."],
        "research_productivity": ["..."],
        "public_presence": ["..."],
        "clinical_engagement": ["..."],
        "federal_funding": ["..."],
        "collaborative_research": ["..."],
        "technology_development": ["..."],
        "research_team": ["..."]
      }
    }
  ]
}

Rules:
- Extract only the first three groups mentioned in the answer (if more than three are present).
- If fewer than three groups are present, return as many as exist.
- For each URL array, include only valid URLs explicitly in the answer. Do not infer or create URLs.
- If a field is missing in the answer, set it to null (for strings) or [] (for arrays).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["First", "Second", "Third"][n - 1] if 1 <= n <= 3 else f"{n}th"


def display_name(group: ResearchGroup) -> str:
    parts = []
    if group.lab_name:
        parts.append(group.lab_name)
    if group.institution:
        parts.append(group.institution)
    if group.pi_name:
        parts.append(f"(PI: {group.pi_name})")
    return " ".join(parts) if parts else "the research group"


def safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls or []


async def ensure_source_and_verify(
    evaluator: Evaluator,
    *,
    parent_node,
    source_list: List[str],
    existence_node_id: str,
    existence_desc: str,
    verify_leaf_id: str,
    verify_leaf_desc: str,
    claim: str,
    add_ins: str,
    critical: bool = True
):
    # Existence check (critical) – to enforce source-grounding
    evaluator.add_custom_node(
        result=bool(source_list),
        id=existence_node_id,
        desc=existence_desc,
        parent=parent_node,
        critical=critical
    )

    # Verification leaf (critical)
    leaf = evaluator.add_leaf(
        id=verify_leaf_id,
        desc=verify_leaf_desc,
        parent=parent_node,
        critical=critical
    )

    # This verify call will auto-skip if the existence node has failed (precondition)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=source_list,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Verification tree builders for each group                                   #
# --------------------------------------------------------------------------- #
async def build_institutional_and_focus(
    evaluator: Evaluator, group_node, group: ResearchGroup, idx: int
):
    sec = evaluator.add_parallel(
        id=f"group_{idx}_institutional_and_focus",
        desc=f"Group {idx} meets institutional affiliation and research focus requirements",
        parent=group_node,
        critical=True
    )

    name_str = display_name(group)

    # U.S. affiliation
    us_urls = safe_urls(group.urls.us_affiliation)
    await ensure_source_and_verify(
        evaluator,
        parent_node=evaluator.add_parallel(
            id=f"group_{idx}_us_affiliation",
            desc="Research group is affiliated with a U.S. university or research institution",
            parent=sec,
            critical=True
        ),
        source_list=us_urls,
        existence_node_id=f"group_{idx}_us_affiliation_source_provided",
        existence_desc="Source URL(s) provided for U.S. institutional affiliation",
        verify_leaf_id=f"group_{idx}_us_affiliation_url",
        verify_leaf_desc="URL reference confirms U.S. institutional affiliation",
        claim=f"The provided page(s) confirm that {name_str} is affiliated with a U.S. university or U.S.-based research institution.",
        add_ins="Accept evidence such as .edu domains, U.S. addresses, or explicit mention of a U.S. campus/location. The page should make U.S. affiliation clear."
    )

    # SfN 2025 participation
    sfn_urls = safe_urls(group.urls.sfn_2025)
    await ensure_source_and_verify(
        evaluator,
        parent_node=evaluator.add_parallel(
            id=f"group_{idx}_sfn_2025_participation",
            desc="Research group or members participated in SfN Annual Meeting 2025 (Nov 15-19, San Diego)",
            parent=sec,
            critical=True
        ),
        source_list=sfn_urls,
        existence_node_id=f"group_{idx}_sfn_source_provided",
        existence_desc="Source URL(s) provided for SfN 2025 participation",
        verify_leaf_id=f"group_{idx}_sfn_url",
        verify_leaf_desc="URL reference confirms SfN 2025 participation",
        claim=f"The provided page(s) show that members of {name_str} participated in the Society for Neuroscience (SfN) 2025 Annual Meeting in San Diego, November 15–19, 2025.",
        add_ins="Evidence can be a listing in SfN 2025 program/abstracts, posters, talks, or official announcements explicitly tied to SfN 2025 (San Diego, Nov 15–19). Disregard other years."
    )

    # BCI/neural interface research focus
    focus_urls = safe_urls(group.urls.research_focus)
    await ensure_source_and_verify(
        evaluator,
        parent_node=evaluator.add_parallel(
            id=f"group_{idx}_bci_research_focus",
            desc="Primary research focus is on brain-computer interfaces, neural interfaces, or neuroprosthetics",
            parent=sec,
            critical=True
        ),
        source_list=focus_urls,
        existence_node_id=f"group_{idx}_focus_source_provided",
        existence_desc="Source URL(s) provided for research focus",
        verify_leaf_id=f"group_{idx}_focus_url",
        verify_leaf_desc="URL reference confirms BCI/neural interface research focus",
        claim=f"The provided page(s) show that {name_str} primarily focuses on brain-computer interfaces, neural interfaces, or neuroprosthetics.",
        add_ins="Accept lab/PI websites or profiles that emphasize BCI/neural interfaces/neuroprosthetics as core research themes."
    )

    # Faculty status
    faculty_urls = safe_urls(group.urls.faculty_status)
    pi_name = group.pi_name or "the principal investigator"
    await ensure_source_and_verify(
        evaluator,
        parent_node=evaluator.add_parallel(
            id=f"group_{idx}_faculty_status",
            desc="Principal investigator holds faculty position at Assistant Professor level or higher",
            parent=sec,
            critical=True
        ),
        source_list=faculty_urls,
        existence_node_id=f"group_{idx}_faculty_source_provided",
        existence_desc="Source URL(s) provided for PI faculty status",
        verify_leaf_id=f"group_{idx}_faculty_url",
        verify_leaf_desc="URL reference confirms PI faculty status",
        claim=f"The provided page(s) confirm that {pi_name} holds a university faculty position at Assistant Professor level or higher.",
        add_ins="Titles such as Assistant/Associate/Full Professor (including 'Research Assistant Professor') qualify. Postdoc or Instructor alone does not qualify."
    )


async def build_research_output(
    evaluator: Evaluator, group_node, group: ResearchGroup, idx: int
):
    sec = evaluator.add_parallel(
        id=f"group_{idx}_research_output",
        desc=f"Group {idx} meets publication and dissemination requirements",
        parent=group_node,
        critical=True
    )

    name_str = display_name(group)

    # High-impact publication (IF ≥ 4) in 2024-2025
    hip_urls = safe_urls(group.urls.high_impact_publication)
    await ensure_source_and_verify(
        evaluator,
        parent_node=evaluator.add_parallel(
            id=f"group_{idx}_high_impact_publication",
            desc="Published at least one article in journal with impact factor ≥4.0 in 2024-2025",
            parent=sec,
            critical=True
        ),
        source_list=hip_urls,
        existence_node_id=f"group_{idx}_hip_source_provided",
        existence_desc="Source URL(s) provided for high-impact publication",
        verify_leaf_id=f"group_{idx}_publication_url",
        verify_leaf_desc="URL reference confirms high-impact publication in 2024-2025",
        claim=f"The provided page(s) show that {name_str} published at least one 2024 or 2025 peer‑reviewed article in a neuroscience journal with impact factor ≥ 4.0.",
        add_ins="Evidence should clearly indicate the journal and year (2024 or 2025), and the journal’s impact factor is ≥4. Accept publisher/journal pages that explicitly state impact factor; preprints alone do not count."
    )

    # Research productivity: ≥3 peer-reviewed neuroscience articles in 2024-2025
    prod_urls = safe_urls(group.urls.research_productivity)
    await ensure_source_and_verify(
        evaluator,
        parent_node=evaluator.add_parallel(
            id=f"group_{idx}_research_productivity",
            desc="Published at least 3 peer-reviewed neuroscience articles in 2024-2025",
            parent=sec,
            critical=True
        ),
        source_list=prod_urls,
        existence_node_id=f"group_{idx}_productivity_source_provided",
        existence_desc="Source URL(s) provided for publication output",
        verify_leaf_id=f"group_{idx}_productivity_url",
        verify_leaf_desc="URL reference confirms publication output in 2024-2025",
        claim=f"The provided page(s) demonstrate that {name_str} has at least three peer‑reviewed neuroscience articles published in 2024–2025.",
        add_ins="Count peer‑reviewed journal articles in 2024 or 2025. Conference abstracts alone do not count. 'In press' or 'early view' is acceptable if clearly peer‑reviewed."
    )

    # Public presence: active lab website or institutional profile
    web_urls = safe_urls(group.urls.public_presence)
    await ensure_source_and_verify(
        evaluator,
        parent_node=evaluator.add_parallel(
            id=f"group_{idx}_public_presence",
            desc="Maintains active lab website or institutional profile page",
            parent=sec,
            critical=True
        ),
        source_list=web_urls,
        existence_node_id=f"group_{idx}_website_source_provided",
        existence_desc="Source URL(s) provided for web presence",
        verify_leaf_id=f"group_{idx}_website_url",
        verify_leaf_desc="URL reference confirms active public web presence",
        claim=f"The provided page(s) show that {name_str} maintains an active lab website or institutional profile page.",
        add_ins="Accept a working lab website or an up‑to‑date institutional profile page. Any one valid page is sufficient."
    )


async def build_clinical_and_funding(
    evaluator: Evaluator, group_node, group: ResearchGroup, idx: int
):
    sec = evaluator.add_parallel(
        id=f"group_{idx}_clinical_and_funding",
        desc=f"Group {idx} meets clinical engagement and funding requirements",
        parent=group_node,
        critical=True
    )

    name_str = display_name(group)

    # Clinical engagement: human clinical trials/studies as of 2025
    clin_urls = safe_urls(group.urls.clinical_engagement)
    await ensure_source_and_verify(
        evaluator,
        parent_node=evaluator.add_parallel(
            id=f"group_{idx}_clinical_engagement",
            desc="Active involvement in or collaboration on human clinical trials/studies as of 2025",
            parent=sec,
            critical=True
        ),
        source_list=clin_urls,
        existence_node_id=f"group_{idx}_clinical_source_provided",
        existence_desc="Source URL(s) provided for clinical engagement",
        verify_leaf_id=f"group_{idx}_clinical_url",
        verify_leaf_desc="URL reference confirms clinical trial involvement or collaboration",
        claim=f"The provided page(s) show that {name_str} was actively involved in or collaborated on human clinical trials/studies as of 2025.",
        add_ins="Accept ClinicalTrials.gov listings, IRB-approved study pages, institutional announcements, or peer‑reviewed clinical reports involving human participants in 2025."
    )

    # Federal funding: NIH BRAIN, NSF, DARPA, or equivalent (2023-2025)
    fund_urls = safe_urls(group.urls.federal_funding)
    await ensure_source_and_verify(
        evaluator,
        parent_node=evaluator.add_parallel(
            id=f"group_{idx}_federal_funding",
            desc="Received NIH BRAIN Initiative, NSF, DARPA, or equivalent federal funding (2023-2025)",
            parent=sec,
            critical=True
        ),
        source_list=fund_urls,
        existence_node_id=f"group_{idx}_funding_source_provided",
        existence_desc="Source URL(s) provided for federal funding",
        verify_leaf_id=f"group_{idx}_funding_url",
        verify_leaf_desc="URL reference confirms federal neuroscience funding 2023-2025",
        claim=f"The provided page(s) confirm that {name_str} received federal neuroscience funding (NIH BRAIN, NSF, DARPA, or equivalent) during 2023–2025.",
        add_ins="Accept official grant pages, award announcements, or institutional listings that explicitly name the sponsor and timeframe within 2023–2025."
    )


async def build_collaboration_and_technology(
    evaluator: Evaluator, group_node, group: ResearchGroup, idx: int
):
    sec = evaluator.add_parallel(
        id=f"group_{idx}_collaboration_and_technology",
        desc=f"Group {idx} meets collaboration, technology, and team requirements",
        parent=group_node,
        critical=True
    )

    name_str = display_name(group)

    # Collaborative research: co-authored with other institutions
    collab_urls = safe_urls(group.urls.collaborative_research)
    await ensure_source_and_verify(
        evaluator,
        parent_node=evaluator.add_parallel(
            id=f"group_{idx}_collaborative_research",
            desc="Documented research collaborations with other institutions through co-authored publications",
            parent=sec,
            critical=True
        ),
        source_list=collab_urls,
        existence_node_id=f"group_{idx}_collab_source_provided",
        existence_desc="Source URL(s) provided for inter-institutional collaborations",
        verify_leaf_id=f"group_{idx}_collaboration_url",
        verify_leaf_desc="URL reference confirms inter-institutional collaborations",
        claim=f"The provided page(s) show that {name_str} has co‑authored publications with collaborators from other institutions.",
        add_ins="Accept publication pages that list multiple author affiliations or clearly indicate cross‑institution collaboration."
    )

    # Technology development/use: advanced neurotechnology
    tech_urls = safe_urls(group.urls.technology_development)
    await ensure_source_and_verify(
        evaluator,
        parent_node=evaluator.add_parallel(
            id=f"group_{idx}_technology_development",
            desc="Develops or utilizes advanced neurotechnology for neural recording or stimulation",
            parent=sec,
            critical=True
        ),
        source_list=tech_urls,
        existence_node_id=f"group_{idx}_tech_source_provided",
        existence_desc="Source URL(s) provided for technology development/use",
        verify_leaf_id=f"group_{idx}_technology_url",
        verify_leaf_desc="URL reference confirms advanced neurotechnology development/use",
        claim=f"The provided page(s) show that {name_str} develops or utilizes advanced neurotechnology for neural recording or stimulation.",
        add_ins="Examples include intracortical arrays, ECoG, neuromodulation devices, closed‑loop BCI systems, or similar advanced neural interfaces."
    )

    # Research team: multiple members including postdocs/grad/staff
    team_urls = safe_urls(group.urls.research_team)
    await ensure_source_and_verify(
        evaluator,
        parent_node=evaluator.add_parallel(
            id=f"group_{idx}_research_team",
            desc="Has multiple team members including postdocs, graduate students, or research staff",
            parent=sec,
            critical=True
        ),
        source_list=team_urls,
        existence_node_id=f"group_{idx}_team_source_provided",
        existence_desc="Source URL(s) provided for research team composition",
        verify_leaf_id=f"group_{idx}_team_url",
        verify_leaf_desc="URL reference confirms research team composition",
        claim=f"The provided page(s) show that {name_str} includes multiple team members such as postdocs, graduate students, and/or research staff.",
        add_ins="Accept lab/team pages listing members. At least two distinct members beyond the PI should be evident."
    )


async def verify_group(
    evaluator: Evaluator, parent_root, group: ResearchGroup, idx: int
):
    # Group container (non-critical as per rubric)
    group_node = evaluator.add_parallel(
        id=f"research_group_{idx}",
        desc=f"{ordinal(idx)} identified research group meets all requirements",
        parent=parent_root,
        critical=False
    )

    # Build critical requirement sections
    await build_institutional_and_focus(evaluator, group_node, group, idx)
    await build_research_output(evaluator, group_node, group, idx)
    await build_clinical_and_funding(evaluator, group_node, group, idx)
    await build_collaboration_and_technology(evaluator, group_node, group, idx)


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
    Evaluate an answer for identifying three U.S.-based BCI/neural interface research groups (2024-2025) that meet all specified requirements.
    """
    # Initialize evaluator
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

    # Extract groups and evidence URLs
    extracted = await evaluator.extract(
        prompt=prompt_extract_groups(),
        template_class=GroupsExtraction,
        extraction_name="extracted_groups"
    )

    # Normalize to exactly 3 groups (pad with empty structures if fewer)
    groups: List[ResearchGroup] = list(extracted.groups[:3])
    while len(groups) < 3:
        groups.append(ResearchGroup())  # empty placeholder with empty URLs

    # Optional: record the evaluation target context
    evaluator.add_custom_info(
        info={"num_groups_extracted": len(extracted.groups)},
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    # Build verification tree for each of the three groups
    for i in range(3):
        await verify_group(evaluator, root, groups[i], i + 1)

    # Return evaluation summary
    return evaluator.get_summary()