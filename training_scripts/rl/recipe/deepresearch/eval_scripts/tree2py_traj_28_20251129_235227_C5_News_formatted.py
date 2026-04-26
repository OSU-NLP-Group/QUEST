import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tn07_special_dem_nominee_2025"
TASK_DESCRIPTION = (
    "Identify the full name of the Democratic nominee in the special congressional election scheduled for December 2, 2025, "
    "for Tennessee's 7th congressional district. The candidate you identify must meet ALL of the following criteria:\n\n"
    "1. Educational Background:\n"
    "   - Holds a Bachelor's degree in Psychology from the University of Texas at Austin\n"
    "   - Holds a Master of Science in Social Work (MSSW) with a focus in Administration and Policy\n\n"
    "2. Professional Credentials:\n"
    "   - Is a licensed social worker\n"
    "   - Currently serves as a Tennessee State Representative for District 51 (serving since 2023)\n\n"
    "3. Personal Information:\n"
    "   - Born on November 24, 1989\n\n"
    "4. Primary Election Performance:\n"
    "   - Won the Democratic primary held on October 7, 2025\n"
    "   - Received 27.89% of the vote\n"
    "   - Received exactly 8,653 votes\n\n"
    "For your answer, provide:\n"
    "- The candidate's full name\n"
    "- Reference URLs that verify each category of information (election details, educational background, professional credentials, biographical information, and primary election results)"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CoreExtraction(BaseModel):
    candidate_name: Optional[str] = None

    # Category-specific sources (as provided in the answer)
    election_detail_urls: List[str] = Field(default_factory=list)
    education_urls: List[str] = Field(default_factory=list)
    professional_urls: List[str] = Field(default_factory=list)
    biographical_urls: List[str] = Field(default_factory=list)
    primary_results_urls: List[str] = Field(default_factory=list)
    endorsements_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return (
        "Extract the following information strictly from the provided answer text:\n"
        "1) candidate_name: The full name of the person the answer identifies as the Democratic nominee for the special congressional election for Tennessee's 7th congressional district.\n"
        "2) election_detail_urls: A list of URLs cited that support election context details (e.g., election date, district, vacancy context, Cook PVI, early voting period).\n"
        "3) education_urls: A list of URLs cited that support the educational background claims.\n"
        "4) professional_urls: A list of URLs cited that support professional credentials (social work license and current office details).\n"
        "5) biographical_urls: A list of URLs cited that support personal biographical details (birth date).\n"
        "6) primary_results_urls: A list of URLs cited that support primary results (win/date/percentage/vote count).\n"
        "7) endorsements_urls: A list of URLs cited that support endorsements (AOC and Jasmine Crockett).\n\n"
        "Rules:\n"
        "- Extract only URLs that appear explicitly in the answer (plain or markdown link format). Do not invent URLs.\n"
        "- If a URL is missing protocol, prepend http://\n"
        "- If any category has no URLs provided, return an empty list for that category.\n"
        "- If the candidate name is not provided, return null for candidate_name."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if not (s.lower().startswith("http://") or s.lower().startswith("https://")):
            s = "http://" + s
        cleaned.append(s)
    return cleaned


def _has_any_valid_url(urls: Optional[List[str]]) -> bool:
    return len(_valid_urls(urls)) > 0


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_nominee_identity(evaluator: Evaluator, parent_node, data: CoreExtraction) -> None:
    node = evaluator.add_parallel(
        id="Nominee_Identity",
        desc="Provides the nominee’s name and explicitly asserts nominee status for the specified race (without duplicating election date/district requirements handled elsewhere).",
        parent=parent_node,
        critical=True,
    )

    # Candidate_Full_Name (custom existence check; requires a full name string)
    full_name_ok = bool(data.candidate_name and isinstance(data.candidate_name, str) and (" " in data.candidate_name.strip()))
    evaluator.add_custom_node(
        result=full_name_ok,
        id="Candidate_Full_Name",
        desc="Provides the candidate's full name.",
        parent=node,
        critical=True
    )

    # States_Democratic_Nominee_Status (verify that the answer itself states this)
    nominee_status_leaf = evaluator.add_leaf(
        id="States_Democratic_Nominee_Status",
        desc="States that this candidate is the Democratic nominee for the specified special congressional election (as defined in Election_Details).",
        parent=node,
        critical=True
    )
    name_for_claim = data.candidate_name or "the candidate"
    claim = (
        f"The answer states that {name_for_claim} is the Democratic nominee for the special congressional election "
        f"for Tennessee's 7th congressional district scheduled on December 2, 2025."
    )
    await evaluator.verify(
        claim=claim,
        node=nominee_status_leaf,
        additional_instruction="Check the answer text only: does it explicitly assert Democratic nominee status for the TN-7 special election on Dec 2, 2025?"
    )


async def build_election_details(evaluator: Evaluator, parent_node, data: CoreExtraction) -> None:
    node = evaluator.add_parallel(
        id="Election_Details",
        desc="Includes required election context details and supporting source(s).",
        parent=parent_node,
        critical=True
    )

    urls = _valid_urls(data.election_detail_urls)

    # Reference URL existence gate (critical)
    ref_urls_node = evaluator.add_custom_node(
        result=_has_any_valid_url(urls),
        id="Reference_URL_Election_Details",
        desc="Provides one or more valid reference URL(s) that collectively verify the election-details claims in this category.",
        parent=node,
        critical=True
    )

    # Leaves and verifications
    leaves_and_claims: List[Dict[str, Any]] = [
        dict(
            id="Special_Election_Date",
            desc="States the special election is scheduled for December 2, 2025.",
            claim="The special congressional election is scheduled for December 2, 2025.",
            add_ins="Verify the page explicitly mentions that the special election date is December 2, 2025. Accept reasonable date formatting variants."
        ),
        dict(
            id="District",
            desc="States the election is for Tennessee's 7th congressional district.",
            claim="The election is for Tennessee's 7th congressional district (TN-7).",
            add_ins="Verify the page indicates the special election pertains to Tennessee's 7th congressional district."
        ),
        dict(
            id="Vacancy_Context",
            desc="States the seat became vacant on July 20, 2025, following Mark Green's resignation.",
            claim="The seat became vacant on July 20, 2025, following Mark Green's resignation.",
            add_ins="Check that the page states the vacancy date (July 20, 2025) and connects it to Mark Green's resignation."
        ),
        dict(
            id="Cook_PVI",
            desc="States the district has a Cook PVI of R+10.",
            claim="The district has a Cook PVI of R+10.",
            add_ins="Verify the page states the Cook Partisan Voting Index (PVI) for TN-7 is R+10."
        ),
        dict(
            id="Early_Voting_Period",
            desc="States early voting ran from November 12 to November 26, 2025.",
            claim="Early voting ran from November 12 to November 26, 2025.",
            add_ins="Verify the page shows early voting window of Nov 12–Nov 26, 2025."
        ),
    ]

    batch: List[tuple] = []
    for item in leaves_and_claims:
        leaf = evaluator.add_leaf(
            id=item["id"],
            desc=item["desc"],
            parent=node,
            critical=True
        )
        batch.append((
            item["claim"],
            urls,
            leaf,
            item["add_ins"]
        ))

    # Ensure claims depend on reference URL existence
    await evaluator.batch_verify(
        [(c, s, n, a) for (c, s, n, a) in batch],
        extra_prerequisites=[ref_urls_node]
    )


async def build_education(evaluator: Evaluator, parent_node, data: CoreExtraction) -> None:
    node = evaluator.add_parallel(
        id="Educational_Background",
        desc="Verifies the candidate's educational credentials and provides supporting source(s).",
        parent=parent_node,
        critical=True
    )

    urls = _valid_urls(data.education_urls)

    ref_urls_node = evaluator.add_custom_node(
        result=_has_any_valid_url(urls),
        id="Reference_URL_Education",
        desc="Provides one or more valid reference URL(s) that collectively verify the educational background claims.",
        parent=node,
        critical=True
    )

    name = data.candidate_name or "The candidate"

    undergrad_leaf = evaluator.add_leaf(
        id="Undergraduate_Degree",
        desc="Holds a Bachelor's degree in Psychology from the University of Texas at Austin.",
        parent=node,
        critical=True
    )
    undergrad_claim = f"{name} holds a Bachelor's degree in Psychology from the University of Texas at Austin."
    await evaluator.verify(
        claim=undergrad_claim,
        node=undergrad_leaf,
        sources=urls,
        additional_instruction="Allow reasonable variants like 'BA in Psychology', 'B.A. in Psychology', and common UT Austin naming.",
        extra_prerequisites=[ref_urls_node]
    )

    grad_leaf = evaluator.add_leaf(
        id="Graduate_Degree",
        desc="Holds a Master of Science in Social Work (MSSW) with a focus in Administration and Policy.",
        parent=node,
        critical=True
    )
    grad_claim = f"{name} holds a Master of Science in Social Work (MSSW) with a focus in Administration and Policy."
    await evaluator.verify(
        claim=grad_claim,
        node=grad_leaf,
        sources=urls,
        additional_instruction="Accept variants like 'MSSW', 'M.S.S.W.', or 'MSW (Master of Science in Social Work)' so long as the program is explicitly the science-track with focus in Administration and Policy.",
        extra_prerequisites=[ref_urls_node]
    )


async def build_professional(evaluator: Evaluator, parent_node, data: CoreExtraction) -> None:
    node = evaluator.add_parallel(
        id="Professional_Credentials",
        desc="Verifies the candidate's professional qualifications and provides supporting source(s).",
        parent=parent_node,
        critical=True
    )

    urls = _valid_urls(data.professional_urls)

    ref_urls_node = evaluator.add_custom_node(
        result=_has_any_valid_url(urls),
        id="Reference_URL_Professional",
        desc="Provides one or more valid reference URL(s) that collectively verify the professional-credentials claims.",
        parent=node,
        critical=True
    )

    name = data.candidate_name or "The candidate"

    # Licensed social worker
    lsw_leaf = evaluator.add_leaf(
        id="Social_Work_License",
        desc="Is a licensed social worker.",
        parent=node,
        critical=True
    )
    lsw_claim = f"{name} is a licensed social worker."
    await evaluator.verify(
        claim=lsw_claim,
        node=lsw_leaf,
        sources=urls,
        additional_instruction="Accept license designations such as LSW, LMSW, LCSW, or equivalent active social work licensure.",
        extra_prerequisites=[ref_urls_node]
    )

    # Current political office
    office_leaf = evaluator.add_leaf(
        id="Current_Political_Office",
        desc="Currently serves as Tennessee State Representative for District 51 (serving since 2023).",
        parent=node,
        critical=True
    )
    office_claim = f"{name} currently serves as a Tennessee State Representative for District 51 and has been serving since 2023."
    await evaluator.verify(
        claim=office_claim,
        node=office_leaf,
        sources=urls,
        additional_instruction="The page should explicitly indicate service in the Tennessee House, District 51, and service since 2023 (e.g., elected or assumed office in 2023).",
        extra_prerequisites=[ref_urls_node]
    )


async def build_personal_info(evaluator: Evaluator, parent_node, data: CoreExtraction) -> None:
    node = evaluator.add_parallel(
        id="Personal_Information",
        desc="Verifies required biographical detail and provides supporting source(s).",
        parent=parent_node,
        critical=True
    )

    urls = _valid_urls(data.biographical_urls)

    ref_urls_node = evaluator.add_custom_node(
        result=_has_any_valid_url(urls),
        id="Reference_URL_Biographical",
        desc="Provides one or more valid reference URL(s) that collectively verify the birth date claim.",
        parent=node,
        critical=True
    )

    name = data.candidate_name or "The candidate"

    birth_leaf = evaluator.add_leaf(
        id="Birth_Date",
        desc="Born on November 24, 1989.",
        parent=node,
        critical=True
    )
    birth_claim = f"{name} was born on November 24, 1989."
    await evaluator.verify(
        claim=birth_claim,
        node=birth_leaf,
        sources=urls,
        additional_instruction="Allow minor date formatting variants (e.g., Nov 24, 1989, 11/24/1989). The page must clearly identify the same person.",
        extra_prerequisites=[ref_urls_node]
    )


async def build_primary_performance(evaluator: Evaluator, parent_node, data: CoreExtraction) -> None:
    node = evaluator.add_parallel(
        id="Primary_Election_Performance",
        desc="Verifies required primary outcome details and provides supporting source(s).",
        parent=parent_node,
        critical=True
    )

    urls = _valid_urls(data.primary_results_urls)

    ref_urls_node = evaluator.add_custom_node(
        result=_has_any_valid_url(urls),
        id="Reference_URL_Primary_Results",
        desc="Provides one or more valid reference URL(s) that collectively verify the primary-results claims (win/date/percentage/vote count).",
        parent=node,
        critical=True
    )

    name = data.candidate_name or "The candidate"

    # Primary date and win
    win_leaf = evaluator.add_leaf(
        id="Primary_Date_And_Win",
        desc="Won the Democratic primary held on October 7, 2025.",
        parent=node,
        critical=True
    )
    win_claim = f"{name} won the Democratic primary held on October 7, 2025."
    await evaluator.verify(
        claim=win_claim,
        node=win_leaf,
        sources=urls,
        additional_instruction="Verify the page states both the primary date (Oct 7, 2025) and that the candidate won the Democratic primary.",
        extra_prerequisites=[ref_urls_node]
    )

    # Vote percentage
    pct_leaf = evaluator.add_leaf(
        id="Vote_Percentage",
        desc="Received 27.89% of the vote in the primary.",
        parent=node,
        critical=True
    )
    pct_claim = f"In the Democratic primary, {name} received 27.89% of the vote."
    await evaluator.verify(
        claim=pct_claim,
        node=pct_leaf,
        sources=urls,
        additional_instruction="Allow minor rounding tolerance if the page shows 27.9% or 27.89%. It must clearly attribute the percentage to the candidate's primary result.",
        extra_prerequisites=[ref_urls_node]
    )

    # Vote count
    count_leaf = evaluator.add_leaf(
        id="Vote_Count",
        desc="Received exactly 8,653 votes in the primary.",
        parent=node,
        critical=True
    )
    count_claim = f"In the Democratic primary, {name} received exactly 8,653 votes."
    await evaluator.verify(
        claim=count_claim,
        node=count_leaf,
        sources=urls,
        additional_instruction="Accept number formatting with or without commas (8653 or 8,653). The page must attribute this vote count to the candidate.",
        extra_prerequisites=[ref_urls_node]
    )


async def build_endorsements(evaluator: Evaluator, parent_node, data: CoreExtraction) -> None:
    node = evaluator.add_parallel(
        id="Political_Endorsements",
        desc="Verifies required endorsements and provides supporting source(s).",
        parent=parent_node,
        critical=True
    )

    urls = _valid_urls(data.endorsements_urls)

    ref_urls_node = evaluator.add_custom_node(
        result=_has_any_valid_url(urls),
        id="Reference_URL_Endorsements",
        desc="Provides one or more valid reference URL(s) that collectively verify the endorsement claim.",
        parent=node,
        critical=True
    )

    name = data.candidate_name or "The candidate"

    endorse_leaf = evaluator.add_leaf(
        id="Endorsed_By_AOC_And_Crockett",
        desc="Received endorsements from Alexandria Ocasio-Cortez and Jasmine Crockett.",
        parent=node,
        critical=True
    )
    endorse_claim = f"{name} received endorsements from Alexandria Ocasio-Cortez and Jasmine Crockett."
    await evaluator.verify(
        claim=endorse_claim,
        node=endorse_leaf,
        sources=urls,
        additional_instruction="For Alexandria Ocasio-Cortez, allow 'AOC' as equivalent. The page(s) must clearly indicate both endorsements.",
        extra_prerequisites=[ref_urls_node]
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict:
    # Initialize evaluator with a container root
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
    extracted: CoreExtraction = await evaluator.extract(
        prompt=prompt_extract_core(),
        template_class=CoreExtraction,
        extraction_name="core_extraction"
    )

    # Add ground truth expectations for transparency
    evaluator.add_ground_truth({
        "election_details": {
            "special_election_date": "December 2, 2025",
            "district": "Tennessee's 7th congressional district",
            "vacancy_context": "Vacant on July 20, 2025 after Mark Green resigned",
            "cook_pvi": "R+10",
            "early_voting_period": "November 12–26, 2025"
        },
        "education": {
            "undergrad": "BA in Psychology, University of Texas at Austin",
            "graduate": "MSSW (Master of Science in Social Work), focus in Administration and Policy"
        },
        "professional": {
            "license": "Licensed social worker",
            "office": "Tennessee State Representative, District 51, serving since 2023"
        },
        "personal": {
            "birth_date": "November 24, 1989"
        },
        "primary_results": {
            "date_and_win": "Won the Democratic primary held on October 7, 2025",
            "vote_percentage": "27.89%",
            "vote_count": "8,653"
        },
        "endorsements": {
            "endorsers": ["Alexandria Ocasio-Cortez", "Jasmine Crockett"]
        }
    }, gt_type="expected_requirements")

    # Build the main rubric node (critical)
    main = evaluator.add_parallel(
        id="Democratic_Nominee_Identification",
        desc="Identifies the Democratic nominee and provides sources verifying all required constraints.",
        parent=root,
        critical=True
    )

    # Build each critical subtree
    await build_nominee_identity(evaluator, main, extracted)
    await build_election_details(evaluator, main, extracted)
    await build_education(evaluator, main, extracted)
    await build_professional(evaluator, main, extracted)
    await build_personal_info(evaluator, main, extracted)
    await build_primary_performance(evaluator, main, extracted)
    await build_endorsements(evaluator, main, extracted)

    # Return structured summary
    return evaluator.get_summary()