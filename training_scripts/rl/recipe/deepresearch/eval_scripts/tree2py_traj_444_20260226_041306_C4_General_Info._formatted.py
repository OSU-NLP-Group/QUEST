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
TASK_ID = "identify_sports_personality_psu_cbs_2018"
TASK_DESCRIPTION = (
    "Who is the sports personality that graduated from Pennsylvania State University in 2008 with a Bachelor of Arts "
    "degree in Broadcast Journalism from the John Curley Center for Sports Journalism, currently works for CBS Sports "
    "as an NFL sideline reporter, and joined CBS Sports in June 2018?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PersonExtraction(BaseModel):
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_person() -> str:
    return """
    Extract exactly one person (the primary subject) identified in the answer who meets the described criteria.
    Return the following fields:
    - name: The full name of the person explicitly stated in the answer. If multiple names appear, choose the one presented as the subject who fits the criteria. If no clear person is identified, return null.
    - urls: An array of ALL URLs cited anywhere in the answer that could be used as sources. Include:
      • Plain URLs
      • Markdown links (extract the underlying URL)
      • Any 'Sources' or 'References' section links
    Do not invent URLs. Deduplicate exact duplicates. Preserve full URLs with protocol (http/https).
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_person_criteria(evaluator: Evaluator, root_node, extracted: PersonExtraction) -> None:
    """
    Build the verification subtree according to the rubric and run evidence-grounded checks.
    All child nodes are critical; failure of any will fail this subtask.
    """
    # Parent node representing the rubric root (critical, parallel aggregation)
    identify_node = evaluator.add_parallel(
        id="identify_sports_personality",
        desc="Identify a sports personality who meets all specified educational and career criteria",
        parent=root_node,
        critical=True
    )

    # Precondition: Name and at least one source URL must be present
    name_ok = bool(extracted.name and extracted.name.strip())
    urls_ok = bool(extracted.urls)
    evaluator.add_custom_node(
        result=(name_ok and urls_ok),
        id="person_name_and_sources_provided",
        desc="The answer identifies the person's name and provides at least one source URL",
        parent=identify_node,
        critical=True
    )

    # Prepare node + claim definitions
    person_name = extracted.name or ""

    # Create leaf nodes (all critical) and prepare claims for batch verification
    claims_batch = []

    # 1) Penn State graduate
    node_psu = evaluator.add_leaf(
        id="penn_state_graduate",
        desc="The person graduated from Pennsylvania State University (Penn State)",
        parent=identify_node,
        critical=True
    )
    claim_psu = (
        f"The provided sources support that {person_name} graduated from Pennsylvania State University "
        f"(also known as 'The Pennsylvania State University' or 'Penn State')."
    )
    ins_psu = (
        "Accept common variants like 'Penn State', 'The Pennsylvania State University', or 'Penn State University'. "
        "The page should explicitly state graduation from Penn State."
    )
    claims_batch.append((claim_psu, extracted.urls, node_psu, ins_psu))

    # 2) Graduation year 2008
    node_year = evaluator.add_leaf(
        id="graduation_year_2008",
        desc="The person graduated in 2008",
        parent=identify_node,
        critical=True
    )
    claim_year = f"The provided sources support that {person_name} graduated in 2008 (e.g., 'Class of 2008')."
    ins_year = "Look for explicit graduation year references such as 'graduated in 2008' or 'Class of 2008'."
    claims_batch.append((claim_year, extracted.urls, node_year, ins_year))

    # 3) BA in Broadcast Journalism
    node_degree = evaluator.add_leaf(
        id="broadcast_journalism_degree",
        desc="The person holds a Bachelor of Arts degree in Broadcast Journalism",
        parent=identify_node,
        critical=True
    )
    claim_degree = (
        f"The provided sources support that {person_name} holds a Bachelor of Arts (B.A.) degree in Broadcast Journalism."
    )
    ins_degree = (
        "Accept reasonable variants like 'BA', 'B.A.', or 'Bachelor's in Broadcast Journalism'. "
        "It must clearly be a bachelor's degree in broadcast journalism."
    )
    claims_batch.append((claim_degree, extracted.urls, node_degree, ins_degree))

    # 4) John Curley Center affiliation for the degree
    node_curley = evaluator.add_leaf(
        id="curley_center_affiliation",
        desc="The person's journalism degree is from Penn State's John Curley Center for Sports Journalism",
        parent=identify_node,
        critical=True
    )
    claim_curley = (
        f"The provided sources support that {person_name}'s broadcast journalism degree is from or associated with "
        f"Penn State's John Curley Center for Sports Journalism (within the Donald P. Bellisario College of Communications)."
    )
    ins_curley = (
        "Accept mentions such as 'John Curley Center for Sports Journalism', 'Curley Center', or affiliation indicating the "
        "degree/program is through the Curley Center at Penn State."
    )
    claims_batch.append((claim_curley, extracted.urls, node_curley, ins_curley))

    # 5) Currently works for CBS Sports
    node_cbs = evaluator.add_leaf(
        id="cbs_sports_employment",
        desc="The person currently works for CBS Sports",
        parent=identify_node,
        critical=True
    )
    claim_cbs = (
        f"The provided sources support that {person_name} currently works for CBS Sports (e.g., described as an active CBS Sports "
        f"reporter/host/sideline reporter)."
    )
    ins_cbs = (
        "Prefer present-tense statements on official or reputable pages. Accept 'CBS Sports reporter', 'with CBS Sports', "
        "or similar language indicating ongoing employment."
    )
    claims_batch.append((claim_cbs, extracted.urls, node_cbs, ins_cbs))

    # 6) NFL sideline reporter role
    node_role = evaluator.add_leaf(
        id="nfl_sideline_reporter_role",
        desc="The person works as an NFL sideline reporter",
        parent=identify_node,
        critical=True
    )
    claim_role = (
        f"The provided sources support that {person_name} works as an NFL sideline reporter "
        f"(e.g., 'NFL on CBS sideline reporter')."
    )
    ins_role = (
        "Accept reasonable phrasing such as 'NFL on CBS sideline reporter', 'sideline reporter for NFL broadcasts', "
        "or equivalent descriptions."
    )
    claims_batch.append((claim_role, extracted.urls, node_role, ins_role))

    # 7) Joined CBS Sports in June 2018
    node_join = evaluator.add_leaf(
        id="cbs_join_date_june_2018",
        desc="The person joined CBS Sports in June 2018",
        parent=identify_node,
        critical=True
    )
    claim_join = (
        f"The provided sources support that {person_name} joined CBS Sports in June 2018."
    )
    ins_join = (
        "Look for explicit statements like 'joined CBS Sports in June 2018' or equivalent phrasing indicating the month and year."
    )
    claims_batch.append((claim_join, extracted.urls, node_join, ins_join))

    # Run verifications in parallel (precondition node already decided synchronously)
    await evaluator.batch_verify(claims_batch)


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
    Evaluate an answer for the sports personality identification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root aggregation strategy
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

    # Extract person name and all URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_person(),
        template_class=PersonExtraction,
        extraction_name="person_extraction"
    )

    # Optional: add custom info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "extracted_name": extracted.name,
            "url_count": len(extracted.urls),
            "urls_sample": extracted.urls[:5] if extracted.urls else []
        },
        info_type="extraction_overview",
        info_name="extraction_overview"
    )

    # Build and run verification sub-tree
    await verify_person_criteria(evaluator, root, extracted)

    # Return summary with verification tree and scoring
    return evaluator.get_summary()