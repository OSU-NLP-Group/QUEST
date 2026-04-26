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
TASK_ID = "pentagon_senator_investigation_2025"
TASK_DESCRIPTION = (
    "In late November 2025, the Pentagon announced an investigation of a U.S. Senator who is a retired military "
    "officer, regarding a video in which the senator made statements directed at military personnel. Identify this "
    "senator and provide comprehensive information including: (1) Senator's Identity and Background: full name, state "
    "represented, political party affiliation, military branch in which they served, and the rank they achieved during "
    "their military service; (2) Investigation Details: the specific date the Pentagon investigation was announced, the "
    "governmental body conducting the investigation, the specific basis or conduct being investigated, and the potential "
    "legal or professional consequences the senator may face; (3) Video Content: the subject matter of the video, the key "
    "message or statement the senator made in the video, and information about other participants in the video if applicable; "
    "and (4) Legal Expert Analysis: legal experts' assessment of whether the Pentagon can successfully pursue action against "
    "the senator, and any legal or constitutional basis mentioned for their opinions. All information must be supported by "
    "valid reference URLs from your research."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SenatorIdentity(BaseModel):
    full_name: Optional[str] = None
    state: Optional[str] = None
    party: Optional[str] = None
    retired_officer_status: Optional[str] = None  # e.g., "retired", "retired officer", "not retired"
    branch: Optional[str] = None
    rank: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class InvestigationDetails(BaseModel):
    announcement_date: Optional[str] = None  # retain as string for flexibility
    investigating_body: Optional[str] = None
    specific_basis: Optional[str] = None
    potential_consequences: List[str] = Field(default_factory=list)
    recall_possibility: Optional[str] = None  # e.g., "possible", "yes", "no", "unclear"
    urls: List[str] = Field(default_factory=list)


class VideoContent(BaseModel):
    directed_at_military_personnel: Optional[str] = None  # e.g., "yes", "directed at service members"
    subject: Optional[str] = None
    key_message: Optional[str] = None
    other_participants: Optional[str] = None  # if none, can be "none"/"not applicable"
    urls: List[str] = Field(default_factory=list)


class LegalExpertAnalysis(BaseModel):
    viability_assessment: Optional[str] = None
    legal_or_constitutional_basis: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class SenatorTaskExtraction(BaseModel):
    senator: Optional[SenatorIdentity] = None
    investigation: Optional[InvestigationDetails] = None
    video: Optional[VideoContent] = None
    legal: Optional[LegalExpertAnalysis] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Identify the specific U.S. senator described in the question (a retired military officer under Pentagon investigation in late November 2025 regarding a video directed at military personnel), and extract all required information from the answer text exactly as presented. Return a single JSON with the following structure:

    {
      "senator": {
        "full_name": string|null,
        "state": string|null,                    // The state represented in the U.S. Senate
        "party": string|null,                    // Political party affiliation (e.g., "Democrat", "Republican")
        "retired_officer_status": string|null,   // e.g., "retired", "retired officer", "not retired"
        "branch": string|null,                   // e.g., "U.S. Army", "U.S. Navy"
        "rank": string|null,                     // e.g., "Lieutenant Colonel", "Captain"
        "urls": string[]                         // All URLs cited that support identity/background claims
      },
      "investigation": {
        "announcement_date": string|null,        // Specific date the Pentagon announced the investigation, as written (any reasonable format)
        "investigating_body": string|null,       // The governmental body conducting the investigation (e.g., "Department of Defense", specific office)
        "specific_basis": string|null,           // The specific conduct being investigated (e.g., statements in the video directed at service members)
        "potential_consequences": string[],      // Possible legal/professional consequences mentioned (e.g., UCMJ charges, recall to active duty)
        "recall_possibility": string|null,       // Whether recall to military service is mentioned (e.g., "possible", "yes", "no", "unclear")
        "urls": string[]                         // URLs cited that support investigation details
      },
      "video": {
        "directed_at_military_personnel": string|null, // e.g., "yes", "explicitly addressed service members"
        "subject": string|null,                // The video topic/subject matter
        "key_message": string|null,            // Key statement/message made by the senator
        "other_participants": string|null,     // Names/roles if others are present; otherwise "none"/"not applicable"
        "urls": string[]                       // URLs cited that describe/host/report on the video
      },
      "legal": {
        "viability_assessment": string|null,   // Legal experts' view on whether Pentagon action is viable/successful
        "legal_or_constitutional_basis": string[], // Legal/constitutional bases cited (e.g., UCMJ, First Amendment, Article I/II)
        "urls": string[]                       // URLs cited for legal-expert analysis
      }
    }

    IMPORTANT:
    - Extract only what appears explicitly in the answer. Do not invent or infer details.
    - For any missing field, return null (or empty array for list fields).
    - For URLs, include only valid URLs explicitly present in the answer (plain or markdown links). If none are present for a section, return an empty array.
    - Keep dates and ranks as strings exactly as written (do not normalize).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _fmt_list_to_string(items: List[str]) -> str:
    return "; ".join([s for s in items if s]) if items else ""

def _valid_url(u: str) -> bool:
    if not isinstance(u, str):
        return False
    u = u.strip()
    return (u.startswith("http://") or u.startswith("https://")) and "." in u and " " not in u

def _all_categories_have_valid_urls(data: SenatorTaskExtraction) -> bool:
    sen_urls = (data.senator.urls if data.senator else [])
    inv_urls = (data.investigation.urls if data.investigation else [])
    vid_urls = (data.video.urls if data.video else [])
    leg_urls = (data.legal.urls if data.legal else [])
    cats = [sen_urls, inv_urls, vid_urls, leg_urls]
    # Each category must have at least one syntactically valid URL
    return all(any(_valid_url(u) for u in cat) for cat in cats)

def _common_additional_instruction() -> str:
    # This instruction is used for all URL-based verifications, and ensures that missing URLs cause failure.
    return (
        "Use only the provided URLs to verify the claim. If no URLs are provided for this verification or if the URLs "
        "do not explicitly support the claim, you must judge the claim as 'Incorrect'. Allow minor paraphrases or "
        "reasonable wording differences, but do not rely on your own knowledge. Focus on explicit support in the webpages."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_senator_identity(evaluator: Evaluator, parent_node, data: SenatorTaskExtraction) -> None:
    node = evaluator.add_parallel(
        id="Senator_Eligibility_and_Background",
        desc="Correctly identify the senator and provide required background attributes per constraints.",
        parent=parent_node,
        critical=True  # All children here will be critical leaves
    )
    sen = data.senator or SenatorIdentity()

    # Full Name
    full_name_leaf = evaluator.add_leaf(
        id="Full_Name",
        desc="Provides the senator's full name.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The senator's full name is '{sen.full_name}'.",
        node=full_name_leaf,
        sources=sen.urls,
        additional_instruction=_common_additional_instruction()
    )

    # Current U.S. Senator as of November 2025
    current_sen_leaf = evaluator.add_leaf(
        id="Current_US_Senator_As_Of_Nov_2025",
        desc="Establishes the person is a current U.S. Senator as of November 2025.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{sen.full_name}' was a current U.S. Senator as of November 2025.",
        node=current_sen_leaf,
        sources=sen.urls,
        additional_instruction=_common_additional_instruction()
    )

    # State Represented
    state_leaf = evaluator.add_leaf(
        id="State_Represented",
        desc="Identifies the state the senator represents.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{sen.full_name}' represents the state of {sen.state} in the U.S. Senate.",
        node=state_leaf,
        sources=sen.urls,
        additional_instruction=_common_additional_instruction()
    )

    # Political Party
    party_leaf = evaluator.add_leaf(
        id="Political_Party",
        desc="Provides the senator's political party affiliation.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{sen.full_name}' is affiliated with the {sen.party} party.",
        node=party_leaf,
        sources=sen.urls,
        additional_instruction=_common_additional_instruction()
    )

    # Retired Military Officer Status
    retired_leaf = evaluator.add_leaf(
        id="Retired_Military_Officer_Status",
        desc="States and supports that the senator is a retired military officer.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{sen.full_name}' is a retired military officer.",
        node=retired_leaf,
        sources=sen.urls,
        additional_instruction=_common_additional_instruction()
    )

    # Military Branch
    branch_leaf = evaluator.add_leaf(
        id="Military_Branch",
        desc="Identifies the military branch in which the senator served.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{sen.full_name}' served in the {sen.branch}.",
        node=branch_leaf,
        sources=sen.urls,
        additional_instruction=_common_additional_instruction()
    )

    # Military Rank
    rank_leaf = evaluator.add_leaf(
        id="Military_Rank",
        desc="Provides the rank achieved during military service.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{sen.full_name}' achieved (and retired at) the rank of {sen.rank}.",
        node=rank_leaf,
        sources=sen.urls,
        additional_instruction=_common_additional_instruction()
    )


async def verify_investigation_details(evaluator: Evaluator, parent_node, data: SenatorTaskExtraction) -> None:
    node = evaluator.add_parallel(
        id="Investigation_Details",
        desc="Accurate and constraint-compliant details about the Pentagon investigation.",
        parent=parent_node,
        critical=True  # All children will be critical leaves
    )
    inv = data.investigation or InvestigationDetails()
    sen_name = (data.senator.full_name if data.senator and data.senator.full_name else "the senator")

    # Announcement Date within Nov 20–30, 2025
    date_leaf = evaluator.add_leaf(
        id="Announcement_Date_In_Range",
        desc="Provides the specific date the Pentagon investigation was announced, and it falls between Nov 20–30, 2025 (inclusive).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The Pentagon announced the investigation on {inv.announcement_date}, and this date falls between "
            f"November 20 and November 30, 2025 (inclusive)."
        ),
        node=date_leaf,
        sources=inv.urls,
        additional_instruction=(
            _common_additional_instruction()
            + " Confirm both the announcement date and that it lies within the specified range."
        )
    )

    # Investigating Body
    body_leaf = evaluator.add_leaf(
        id="Investigating_Body",
        desc="Identifies the governmental body conducting the investigation (as described in sources).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The investigation into {sen_name} is being conducted by {inv.investigating_body}.",
        node=body_leaf,
        sources=inv.urls,
        additional_instruction=_common_additional_instruction()
    )

    # Specific Basis
    basis_leaf = evaluator.add_leaf(
        id="Specific_Basis",
        desc="Describes the specific basis/conduct being investigated in a verifiable way.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The investigation concerns the following conduct: {inv.specific_basis}.",
        node=basis_leaf,
        sources=inv.urls,
        additional_instruction=_common_additional_instruction()
    )

    # Potential Legal/Professional Consequences
    cons_leaf = evaluator.add_leaf(
        id="Potential_Legal_Or_Professional_Consequences",
        desc="Describes the potential legal or professional consequences the senator may face (as supported by sources).",
        parent=node,
        critical=True
    )
    cons_text = _fmt_list_to_string(inv.potential_consequences)
    await evaluator.verify(
        claim=f"Potential consequences for {sen_name} include: {cons_text}.",
        node=cons_leaf,
        sources=inv.urls,
        additional_instruction=_common_additional_instruction()
    )

    # Recall to Military Service Possibility
    recall_leaf = evaluator.add_leaf(
        id="Recall_To_Military_Service_Possibility",
        desc="Documents that the investigation involves potential recall to military service for legal proceedings (per the explicit constraint).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The investigation involves (or mentions) possible recall to military service for legal proceedings.",
        node=recall_leaf,
        sources=inv.urls,
        additional_instruction=_common_additional_instruction()
    )


async def verify_video_content(evaluator: Evaluator, parent_node, data: SenatorTaskExtraction) -> None:
    # Set non-critical to allow partial credit for the optional participants leaf
    node = evaluator.add_parallel(
        id="Video_Content",
        desc="Information about the video that prompted the investigation, including that it addresses military personnel.",
        parent=parent_node,
        critical=False
    )
    vid = data.video or VideoContent()
    sen_name = (data.senator.full_name if data.senator and data.senator.full_name else "the senator")

    # Directed at Military Personnel (Critical)
    directed_leaf = evaluator.add_leaf(
        id="Directed_At_Military_Personnel",
        desc="Establishes that the video statements were directed at military personnel/service members.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The video statements were directed at military personnel/service members.",
        node=directed_leaf,
        sources=vid.urls,
        additional_instruction=_common_additional_instruction()
    )

    # Video Subject (Critical)
    subject_leaf = evaluator.add_leaf(
        id="Video_Subject",
        desc="Describes the subject matter/topic of the video.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The subject/topic of the video featuring {sen_name} is: {vid.subject}.",
        node=subject_leaf,
        sources=vid.urls,
        additional_instruction=_common_additional_instruction()
    )

    # Key Message (Critical)
    key_message_leaf = evaluator.add_leaf(
        id="Key_Message",
        desc="Provides the key message/statement made by the senator in the video.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the video, {sen_name} stated: {vid.key_message}.",
        node=key_message_leaf,
        sources=vid.urls,
        additional_instruction=_common_additional_instruction()
    )

    # Other Participants If Applicable (Non-critical)
    participants_leaf = evaluator.add_leaf(
        id="Other_Participants_If_Applicable",
        desc="If other participants are present per sources, identifies them or describes involvement; otherwise explicitly states none/not applicable.",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"Other participants involved: {vid.other_participants}. If none/not applicable, this statement should reflect that.",
        node=participants_leaf,
        sources=vid.urls,
        additional_instruction=_common_additional_instruction()
    )


async def verify_legal_expert_analysis(evaluator: Evaluator, parent_node, data: SenatorTaskExtraction) -> None:
    node = evaluator.add_parallel(
        id="Legal_Expert_Analysis",
        desc="Legal expert opinions on whether Pentagon action is viable and the legal/constitutional basis for those opinions.",
        parent=parent_node,
        critical=True  # All children will be critical leaves
    )
    leg = data.legal or LegalExpertAnalysis()

    # Expert Opinion on Viability
    viability_leaf = evaluator.add_leaf(
        id="Expert_Opinion_On_Viability",
        desc="Reports legal experts' assessment of whether the Pentagon can successfully pursue action against the senator.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Legal experts assess the viability of Pentagon action as follows: {leg.viability_assessment}.",
        node=viability_leaf,
        sources=leg.urls,
        additional_instruction=_common_additional_instruction()
    )

    # Legal or Constitutional Basis
    basis_leaf = evaluator.add_leaf(
        id="Legal_Or_Constitutional_Basis",
        desc="Provides the legal and/or constitutional basis cited for expert opinions, supported by sources.",
        parent=node,
        critical=True
    )
    basis_text = _fmt_list_to_string(leg.legal_or_constitutional_basis)
    await evaluator.verify(
        claim=f"Experts cite the following legal/constitutional bases: {basis_text}.",
        node=basis_leaf,
        sources=leg.urls,
        additional_instruction=_common_additional_instruction()
    )


async def add_reference_urls_gate(evaluator: Evaluator, parent_node, data: SenatorTaskExtraction) -> None:
    # A single critical check under root to enforce that all categories provide at least one valid URL.
    # Using add_custom_node to perform a syntactic validity check (starts with http and contains a domain).
    result = _all_categories_have_valid_urls(data)
    evaluator.add_custom_node(
        result=result,
        id="Reference_URLs_For_All_Claims",
        desc="Provides valid reference URL(s) that support all required claims across identity/background, investigation details, video content, and legal-expert analysis.",
        parent=parent_node,
        critical=True
    )

    # Also record basic URL statistics
    sen_urls = (data.senator.urls if data.senator else [])
    inv_urls = (data.investigation.urls if data.investigation else [])
    vid_urls = (data.video.urls if data.video else [])
    leg_urls = (data.legal.urls if data.legal else [])
    evaluator.add_custom_info(
        {
            "identity_url_count": len(sen_urls),
            "investigation_url_count": len(inv_urls),
            "video_url_count": len(vid_urls),
            "legal_url_count": len(leg_urls),
            "all_categories_have_valid_urls": result
        },
        info_type="url_coverage_stats"
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
    Evaluate an answer for the Pentagon senator investigation task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation as parallel per rubric
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

    # IMPORTANT: To comply with framework constraints (critical parent cannot have non-critical children),
    # we keep root as non-critical (default) and set criticality appropriately for subtrees.
    # The Video_Content subtree includes a non-critical leaf, so that subtree must be non-critical.

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=SenatorTaskExtraction,
        extraction_name="structured_extraction"
    )

    # Build verification tree according to rubric
    # 1) Senator identity and background (critical, parallel)
    await verify_senator_identity(evaluator, root, extraction)

    # 2) Investigation details (critical, parallel)
    await verify_investigation_details(evaluator, root, extraction)

    # 3) Video content (non-critical, parallel) – allows partial credit for optional participants
    await verify_video_content(evaluator, root, extraction)

    # 4) Legal expert analysis (critical, parallel)
    await verify_legal_expert_analysis(evaluator, root, extraction)

    # 5) Reference URLs for all claims (critical single check)
    await add_reference_urls_gate(evaluator, root, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()