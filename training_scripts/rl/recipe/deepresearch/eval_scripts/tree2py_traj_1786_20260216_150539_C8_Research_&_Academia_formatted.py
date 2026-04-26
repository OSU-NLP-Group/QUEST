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
TASK_ID = "us_interdisciplinary_collabs_2"
TASK_DESCRIPTION = """Identify two active interdisciplinary research collaborations or centers in the United States that meet ALL of the following requirements:

1. The collaboration must involve researchers from at least three (3) distinct universities or research institutions located in the United States.

2. The collaboration must span at least three (3) distinct academic disciplines or fields of study.

3. The collaboration must have been formally established or initiated between January 1, 2020 and December 31, 2025.

4. The collaboration must have a publicly identified director, principal investigator, or equivalent leadership position with their name and institutional affiliation clearly listed.

5. The collaboration must have a formal governance structure, such as an advisory board, steering committee, executive committee, or equivalent body.

6. The collaboration must have a dedicated website or substantial web presence that provides information about the program, its participants, and activities.

7. The collaboration must have secured external funding from major federal agencies (such as NSF, NIH, DOE, or DOD) or major private foundations.

8. The collaboration must have clearly stated research goals, objectives, or focus areas that are publicly available.

9. The collaboration must have produced collaborative research outputs (such as peer-reviewed publications, conference proceedings, technical reports, or datasets) within the last 24 months (since February 2024).

10. The collaboration must include provisions for graduate student, postdoctoral fellow, or early-career researcher training and development.

11. The collaboration must have policies, infrastructure, or stated commitments regarding data sharing, resource sharing, or collaborative tools among partner institutions.

12. The collaboration must operate under formal partnership agreements or memoranda of understanding among participating institutions (this can be inferred from governance documents or official program descriptions).

13. At least two of the participating institutions must be located in different U.S. states.

14. The collaboration must be currently active (not concluded or terminated) as of February 2026.

For each collaboration identified, provide:
- The name of the collaboration
- A brief description
- The names of at least three participating institutions
- The names of at least three disciplines involved
- The name and affiliation of the director or principal investigator
- Evidence of establishment date
- Description of governance structure
- Website URL
- Funding source(s)
- Research objectives
- Examples of recent research outputs
- Description of training opportunities
- Information about data/resource sharing
- Evidence of formal partnerships
- State locations of participating institutions
- Evidence of current active status
- Reference URLs for all claims
"""

SINCE_DATE_STR = "February 2024"
ACTIVE_AS_OF_STR = "February 2026"

# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class CollaborationExtraction(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    website_url: Optional[str] = None

    institutions: List[str] = Field(default_factory=list)
    disciplines: List[str] = Field(default_factory=list)

    leader_name: Optional[str] = None
    leader_affiliation: Optional[str] = None

    establishment_date: Optional[str] = None  # free-form date text as provided in answer

    governance_structure: Optional[str] = None

    funding_sources: List[str] = Field(default_factory=list)

    research_objectives: Optional[str] = None

    recent_outputs: List[str] = Field(default_factory=list)  # titles or brief identifiers

    training_programs: Optional[str] = None

    data_sharing: Optional[str] = None

    partnership_agreements: Optional[str] = None

    institution_states: List[str] = Field(default_factory=list)

    active_status: Optional[str] = None  # free-form status or evidence text

    reference_urls: List[str] = Field(default_factory=list)  # must include all URLs cited in the answer for this collab


class TwoCollaborationsExtraction(BaseModel):
    collaborations: List[CollaborationExtraction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_two_collaborations() -> str:
    return """
Extract up to TWO (2) interdisciplinary research collaborations or centers in the United States that the answer presents as meeting the specified requirements.

For each collaboration (keep the original order; if more than two are present, only extract the first two; if fewer than two, return whatever is present), extract the following fields exactly as they appear in the answer without inventing or inferring anything:

- name: The name of the collaboration.
- description: A brief description or summary sentence(s) about the collaboration.
- website_url: The dedicated official website URL (if provided).
- institutions: A list of the names of participating universities or research institutions mentioned for this collaboration.
- disciplines: A list of at least three distinct academic disciplines/fields explicitly stated for this collaboration, if present.
- leader_name: The name of the director or principal investigator.
- leader_affiliation: The institutional affiliation for the director/PI.
- establishment_date: The establishment or initiation date as text (e.g., "2021", "April 2022", "Established in 2020") exactly as stated in the answer.
- governance_structure: The description of governance (e.g., "advisory board", "steering committee", "executive committee") as stated in the answer.
- funding_sources: A list of external funding sources (agencies or foundations) explicitly named in the answer for this collaboration.
- research_objectives: The stated goals/objectives/focus areas as text from the answer.
- recent_outputs: A list of examples of collaborative research outputs since February 2024 as stated in the answer (e.g., paper titles, reports, datasets). Include only those examples the answer explicitly lists.
- training_programs: Description of graduate/postdoc/early-career training opportunities as stated in the answer.
- data_sharing: Information about data or resource sharing policies, infrastructure, or tools as stated in the answer.
- partnership_agreements: Any statements about formal partnership agreements or MOUs as stated in the answer.
- institution_states: A list of U.S. state names or 2-letter abbreviations for the participating institutions, as explicitly provided in the answer. Do not infer or add states not present in the answer.
- active_status: Any explicit indication that the collaboration is currently active (e.g., events/news 2025-2026, "ongoing", "current" statements) as provided in the answer.
- reference_urls: A list of ALL URLs cited in the answer that support this collaboration (including the dedicated website and any other authoritative sources). Only include valid URLs explicitly present in the answer.

Return the result as:
{
  "collaborations": [
    { ... first collaboration fields ... },
    { ... second collaboration fields ... }
  ]
}
If any field is missing for a collaboration, set it to null (for single fields) or an empty list (for list fields).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(collab: CollaborationExtraction) -> List[str]:
    urls: List[str] = []
    if collab.website_url and collab.website_url.strip():
        urls.append(collab.website_url.strip())
    for u in collab.reference_urls:
        if u and isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _non_empty(text: Optional[str]) -> bool:
    return bool(text and isinstance(text, str) and text.strip() != "")


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_collaboration(
    evaluator: Evaluator,
    parent_node,
    collab: CollaborationExtraction,
    idx: int,
) -> None:
    """
    Build the verification subtree for a single collaboration and execute verifications.
    This function assumes that the parent_node is already created.
    """
    # Label for human readability
    which = "First" if idx == 0 else "Second"

    # Create the collaboration node (CRITICAL: this collaboration must meet all criteria to qualify)
    collab_node = evaluator.add_parallel(
        id=f"collab_{idx}",
        desc=f"{which} qualifying research collaboration",
        parent=parent_node,
        critical=True
    )

    sources = _combine_sources(collab)

    # 0) Reference URL existence (critical gating for subsequent evidence-based checks)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id=f"collab_{idx}_collaboration_reference",
        desc="Provide reference URL to the collaboration's official website or authoritative source",
        parent=collab_node,
        critical=True
    )

    # 1) Name and basic info group
    name_group = evaluator.add_parallel(
        id=f"collab_{idx}_name_group",
        desc="Provide the name and basic description of the collaboration",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(collab.name),
        id=f"collab_{idx}_name_exists",
        desc="Name is provided",
        parent=name_group,
        critical=True
    )
    name_leaf = evaluator.add_leaf(
        id=f"collab_{idx}_name_and_basic_info",
        desc="Collaboration name is supported by cited sources",
        parent=name_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The collaboration is named '{collab.name}'.",
        node=name_leaf,
        sources=sources,
        additional_instruction="Verify the official or authoritative sources explicitly use this name for the collaboration. Allow minor variations or abbreviations."
    )

    # 2) Multi-institutional group
    mi_group = evaluator.add_parallel(
        id=f"collab_{idx}_multi_institutional",
        desc="Verify at least three distinct U.S. universities or research institutions participate",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(collab.institutions) >= 3,
        id=f"collab_{idx}_institutions_count",
        desc="At least three participating institutions are provided",
        parent=mi_group,
        critical=True
    )
    for j in range(3):
        inst = collab.institutions[j] if j < len(collab.institutions) else None
        inst_leaf = evaluator.add_leaf(
            id=f"collab_{idx}_institution_{j+1}",
            desc=f"{['First','Second','Third'][j]} participating institution identified with evidence",
            parent=mi_group,
            critical=True
        )
        await evaluator.verify(
            claim=f"'{inst}' is a participating institution in the collaboration '{collab.name}'.",
            node=inst_leaf,
            sources=sources,
            additional_instruction="Check the cited sources for a participant list, consortium members, partners, or similar sections that explicitly list the named institution. Allow official abbreviations."
        )

    # 3) Interdisciplinary group
    id_group = evaluator.add_parallel(
        id=f"collab_{idx}_interdisciplinary",
        desc="Verify that at least three distinct academic disciplines are spanned",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(collab.disciplines) >= 3,
        id=f"collab_{idx}_disciplines_count",
        desc="At least three disciplines are provided",
        parent=id_group,
        critical=True
    )
    for j in range(3):
        disc = collab.disciplines[j] if j < len(collab.disciplines) else None
        disc_leaf = evaluator.add_leaf(
            id=f"collab_{idx}_discipline_{j+1}",
            desc=f"{['First','Second','Third'][j]} discipline identified with evidence",
            parent=id_group,
            critical=True
        )
        await evaluator.verify(
            claim=f"The collaboration '{collab.name}' involves the discipline '{disc}' (or an equivalent field).",
            node=disc_leaf,
            sources=sources,
            additional_instruction="Verify that the cited sources explicitly mention this discipline/field or a clear equivalent among the areas, thrusts, or focus fields."
        )

    # 4) Establishment timeline (2020-01-01 to 2025-12-31)
    est_group = evaluator.add_parallel(
        id=f"collab_{idx}_establishment_group",
        desc="Verify that the collaboration was formally established between 2020-01-01 and 2025-12-31",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(collab.establishment_date),
        id=f"collab_{idx}_establishment_exists",
        desc="Establishment date is provided",
        parent=est_group,
        critical=True
    )
    est_leaf = evaluator.add_leaf(
        id=f"collab_{idx}_establishment_timeline",
        desc="Establishment date falls within 2020-2025 and is supported by sources",
        parent=est_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The collaboration '{collab.name}' was established or initiated on or around {collab.establishment_date}, and this date falls between January 1, 2020 and December 31, 2025 (inclusive).",
        node=est_leaf,
        sources=sources,
        additional_instruction="Confirm both (1) that the date is supported by the sources and (2) that it lies within the inclusive range 2020-01-01 to 2025-12-31."
    )

    # 5) Leadership
    lead_group = evaluator.add_parallel(
        id=f"collab_{idx}_leadership_group",
        desc="Verify that a director/PI and their affiliation are publicly identified",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(collab.leader_name) and _non_empty(collab.leader_affiliation),
        id=f"collab_{idx}_leadership_exists",
        desc="Leader name and affiliation are provided",
        parent=lead_group,
        critical=True
    )
    lead_leaf = evaluator.add_leaf(
        id=f"collab_{idx}_leadership",
        desc="Leadership (name and affiliation) is supported by sources",
        parent=lead_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The collaboration's director or principal investigator is {collab.leader_name} from {collab.leader_affiliation}.",
        node=lead_leaf,
        sources=sources,
        additional_instruction="Verify that the sources explicitly list this person as the director/PI (or equivalent) and show the stated affiliation."
    )

    # 6) Governance
    gov_group = evaluator.add_parallel(
        id=f"collab_{idx}_governance_group",
        desc="Verify that a formal governance structure exists",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(collab.governance_structure),
        id=f"collab_{idx}_governance_exists",
        desc="Governance structure description is provided",
        parent=gov_group,
        critical=True
    )
    gov_leaf = evaluator.add_leaf(
        id=f"collab_{idx}_governance",
        desc="Formal governance structure is supported by sources",
        parent=gov_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The collaboration has a formal governance structure (e.g., advisory board, steering committee, or executive committee) as described: {collab.governance_structure}.",
        node=gov_leaf,
        sources=sources,
        additional_instruction="Look for explicit mentions of advisory boards, steering committees, executive committees, or similar governance bodies."
    )

    # 7) Web presence
    web_group = evaluator.add_parallel(
        id=f"collab_{idx}_web_group",
        desc="Verify dedicated website or substantial web presence",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(collab.website_url),
        id=f"collab_{idx}_web_exists",
        desc="Website URL is provided",
        parent=web_group,
        critical=True
    )
    web_leaf = evaluator.add_leaf(
        id=f"collab_{idx}_web_presence",
        desc="Dedicated website provides program information",
        parent=web_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage is the dedicated website for the collaboration '{collab.name}' and provides information about the program, participants, or activities.",
        node=web_leaf,
        sources=collab.website_url if collab.website_url else sources,
        additional_instruction="Verify that the page is clearly the official site or a substantial web presence for the collaboration and not merely a news article."
    )

    # 8) External funding
    fund_group = evaluator.add_parallel(
        id=f"collab_{idx}_funding_group",
        desc="Verify external funding from major federal agencies or major private foundations",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(collab.funding_sources) > 0,
        id=f"collab_{idx}_funding_exists",
        desc="Funding sources are provided",
        parent=fund_group,
        critical=True
    )
    fund_leaf = evaluator.add_leaf(
        id=f"collab_{idx}_external_funding",
        desc="External funding (major federal or major private foundation) is supported by sources",
        parent=fund_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The collaboration '{collab.name}' has secured external funding from the following organizations: {', '.join(collab.funding_sources)}. At least one of these is a major U.S. federal agency (NSF, NIH, DOE, DOD) or a major private foundation.",
        node=fund_leaf,
        sources=sources,
        additional_instruction="Look for explicit mention of funding by NSF, NIH, DOE, DOD (including agencies like DARPA), or well-known major private foundations. Consider the requirement met if at least one funder is in these categories."
    )

    # 9) Research objectives
    obj_group = evaluator.add_parallel(
        id=f"collab_{idx}_objectives_group",
        desc="Verify that research goals/objectives are publicly available",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(collab.research_objectives),
        id=f"collab_{idx}_objectives_exist",
        desc="Research objectives text is provided",
        parent=obj_group,
        critical=True
    )
    obj_leaf = evaluator.add_leaf(
        id=f"collab_{idx}_research_objectives",
        desc="Research objectives are supported by sources",
        parent=obj_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publicly available sources describe the research goals or objectives of '{collab.name}', for example: {collab.research_objectives}.",
        node=obj_leaf,
        sources=sources,
        additional_instruction="Verify that the sources explicitly state the goals/objectives/focus areas and that they correspond to the provided summary."
    )

    # 10) Recent publications/outputs since February 2024
    pub_group = evaluator.add_parallel(
        id=f"collab_{idx}_recent_outputs_group",
        desc=f"Verify the collaboration produced outputs since {SINCE_DATE_STR}",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(collab.recent_outputs) > 0,
        id=f"collab_{idx}_recent_outputs_exist",
        desc="At least one recent output example is provided",
        parent=pub_group,
        critical=True
    )
    pub_leaf = evaluator.add_leaf(
        id=f"collab_{idx}_recent_publications",
        desc=f"Outputs since {SINCE_DATE_STR} are supported by sources",
        parent=pub_group,
        critical=True
    )
    examples_str = "; ".join(collab.recent_outputs[:3]) if collab.recent_outputs else ""
    await evaluator.verify(
        claim=f"The collaboration '{collab.name}' has produced collaborative research outputs since {SINCE_DATE_STR}, for example: {examples_str}.",
        node=pub_leaf,
        sources=sources,
        additional_instruction=f"Confirm at least one output has a date on or after {SINCE_DATE_STR}. Accept peer-reviewed publications, conference papers, technical reports, or datasets."
    )

    # 11) Training component
    train_group = evaluator.add_parallel(
        id=f"collab_{idx}_training_group",
        desc="Verify training opportunities for graduate students, postdocs, or early-career researchers",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(collab.training_programs),
        id=f"collab_{idx}_training_exists",
        desc="Training description is provided",
        parent=train_group,
        critical=True
    )
    train_leaf = evaluator.add_leaf(
        id=f"collab_{idx}_training_component",
        desc="Training opportunities are supported by sources",
        parent=train_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The collaboration '{collab.name}' includes training opportunities for graduate students, postdoctoral fellows, or early-career researchers (e.g., {collab.training_programs}).",
        node=train_leaf,
        sources=sources,
        additional_instruction="Look for references to fellowships, training programs, mentoring, workshops, summer schools, or similar opportunities targeted at grads/postdocs/ECRs."
    )

    # 12) Data/resource sharing
    data_group = evaluator.add_parallel(
        id=f"collab_{idx}_data_group",
        desc="Verify data/resource sharing policies or infrastructure/tools",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(collab.data_sharing),
        id=f"collab_{idx}_data_exists",
        desc="Data/resource sharing info is provided",
        parent=data_group,
        critical=True
    )
    data_leaf = evaluator.add_leaf(
        id=f"collab_{idx}_data_sharing",
        desc="Data/resource sharing commitments are supported by sources",
        parent=data_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The collaboration '{collab.name}' has policies, infrastructure, or stated commitments for data/resource sharing or collaborative tools (e.g., {collab.data_sharing}).",
        node=data_leaf,
        sources=sources,
        additional_instruction="Look for statements about shared repositories, data portals, common toolchains, data management plans, or explicit sharing policies."
    )

    # 13) Partnership agreements/MOUs
    part_group = evaluator.add_parallel(
        id=f"collab_{idx}_partnership_group",
        desc="Verify formal partnership agreements or MOUs among institutions",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(collab.partnership_agreements),
        id=f"collab_{idx}_partnership_exists",
        desc="Partnership/MOU information is provided",
        parent=part_group,
        critical=True
    )
    part_leaf = evaluator.add_leaf(
        id=f"collab_{idx}_partnership_agreements",
        desc="Formal partnership agreements/MOUs are supported by sources",
        parent=part_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The collaboration '{collab.name}' operates under formal partnership agreements or memoranda of understanding among participating institutions (as described: {collab.partnership_agreements}).",
        node=part_leaf,
        sources=sources,
        additional_instruction="Confirm explicit partnership agreements, MOUs, or official consortium agreements. If governance documents imply such agreements, that is acceptable if clearly stated."
    )

    # 14) Geographic distribution (different states)
    geo_group = evaluator.add_parallel(
        id=f"collab_{idx}_geo_group",
        desc="Verify at least two participating institutions are located in different U.S. states",
        parent=collab_node,
        critical=True
    )
    # Prepare states info
    distinct_states = []
    if collab.institution_states:
        for s in collab.institution_states:
            if s and s.strip() and s.strip() not in distinct_states:
                distinct_states.append(s.strip())
    evaluator.add_custom_node(
        result=len(distinct_states) >= 2,
        id=f"collab_{idx}_geo_states_exist",
        desc="At least two distinct U.S. states are provided",
        parent=geo_group,
        critical=True
    )
    geo_leaf = evaluator.add_leaf(
        id=f"collab_{idx}_geographic_distribution",
        desc="Different-state participation is supported by sources",
        parent=geo_group,
        critical=True
    )
    if len(distinct_states) >= 2:
        geo_claim = f"At least two participating institutions are located in different U.S. states: {distinct_states[0]} and {distinct_states[1]}."
    else:
        geo_claim = "At least two participating institutions are located in different U.S. states."
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=sources,
        additional_instruction="Use the cited sources to confirm institutions' locations. Different state abbreviations or full names count as different states."
    )

    # 15) Active status as of February 2026
    act_group = evaluator.add_parallel(
        id=f"collab_{idx}_active_group",
        desc=f"Verify the collaboration is currently active as of {ACTIVE_AS_OF_STR}",
        parent=collab_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(collab.active_status),
        id=f"collab_{idx}_active_exists",
        desc="Active status evidence is provided",
        parent=act_group,
        critical=True
    )
    act_leaf = evaluator.add_leaf(
        id=f"collab_{idx}_active_status",
        desc=f"Active status as of {ACTIVE_AS_OF_STR} is supported by sources",
        parent=act_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The collaboration '{collab.name}' is currently active as of {ACTIVE_AS_OF_STR}.",
        node=act_leaf,
        sources=sources,
        additional_instruction=f"Look for recency indicators in 2025 or 2026 (e.g., events, news, updated pages) or explicit 'ongoing/current' statements in the sources."
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
    Evaluate an answer for the task of identifying two qualifying interdisciplinary collaborations.
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

    # Extract collaborations from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_two_collaborations(),
        template_class=TwoCollaborationsExtraction,
        extraction_name="extracted_collaborations"
    )

    # Normalize to exactly two collaboration entries (pad with empty if needed)
    collabs: List[CollaborationExtraction] = list(extracted.collaborations[:2])
    while len(collabs) < 2:
        collabs.append(CollaborationExtraction())

    # Critical gate at root: ensure the answer attempts to provide two collaborations (with at least basic source)
    provided_count = 0
    for c in collabs:
        if _non_empty(c.name) and ( _non_empty(c.website_url) or len(c.reference_urls) > 0 ):
            provided_count += 1
    evaluator.add_custom_node(
        result=(provided_count >= 2),
        id="two_collaborations_provided",
        desc="The answer provides two collaborations with at least one reference URL each",
        parent=root,
        critical=True
    )

    # Build verification subtrees for both collaborations
    for i in range(2):
        await verify_collaboration(evaluator, root, collabs[i], i)

    return evaluator.get_summary()