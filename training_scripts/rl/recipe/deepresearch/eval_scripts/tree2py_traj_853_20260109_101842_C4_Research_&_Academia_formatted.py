import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nsf_career_2023_2025_ai_five_universities"
TASK_DESCRIPTION = (
    "I'm researching recent NSF CAREER award recipients in computer science who focus on artificial intelligence and machine learning. "
    "Identify one faculty member from each of the following five universities who received an NSF CAREER award between 2023-2025 and whose research focuses on AI, machine learning, robotics, or related areas:\n\n"
    "1. Carnegie Mellon University\n2. University of Michigan\n3. University of Illinois Urbana-Champaign\n4. University of Washington\n5. University of California, Los Angeles\n\n"
    "For each faculty member, provide:\n- Their full name\n- Their department/school affiliation within the university\n- A brief description of their NSF CAREER-funded research focus\n- A reference URL to an official announcement or university news page about their award"
)

UNIVERSITY_INFO = {
    "CMU": {
        "university": "Carnegie Mellon University",
        "domains": ["cmu.edu", "cs.cmu.edu", "ri.cmu.edu", "ece.cmu.edu", "s3d.cmu.edu"],
    },
    "UMich": {
        "university": "University of Michigan",
        "domains": ["umich.edu", "engin.umich.edu", "eecs.umich.edu", "cse.engin.umich.edu"],
    },
    "UIUC": {
        "university": "University of Illinois Urbana-Champaign",
        "domains": ["illinois.edu", "cs.illinois.edu", "ece.illinois.edu", "cs.illinois.edu"],
    },
    "UW": {
        "university": "University of Washington",
        "domains": ["washington.edu", "cs.washington.edu", "ece.uw.edu", "paulallen.ai"],
    },
    "UCLA": {
        "university": "University of California, Los Angeles",
        "domains": ["ucla.edu", "seas.ucla.edu", "cs.ucla.edu", "ee.ucla.edu"],
    },
}

ACCEPTABLE_AREAS_HINT = (
    "Artificial Intelligence (AI), Machine Learning (ML), Robotics, or closely related computational areas, "
    "including but not limited to Computer Vision, Natural Language Processing, Human-Computer Interaction with AI components, "
    "Autonomous Systems, Data Science/Mining with strong ML focus, and core CS/CE AI-adjacent fields."
)

YEAR_WINDOW_HINT = "The NSF CAREER award year must be 2023, 2024, or 2025 (inclusive). Use the page content or announcement date if needed."


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RecipientEntry(BaseModel):
    full_name: Optional[str] = None
    department: Optional[str] = None
    research_focus: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class NSFRecipientsExtraction(BaseModel):
    cmu: Optional[RecipientEntry] = None
    umich: Optional[RecipientEntry] = None
    uiuc: Optional[RecipientEntry] = None
    uw: Optional[RecipientEntry] = None
    ucla: Optional[RecipientEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_recipients() -> str:
    return """
Extract, for each of the five universities listed below, the single faculty member that the answer presents as an NSF CAREER award recipient (award year between 2023 and 2025 inclusive) in AI/ML/Robotics or closely related areas. If the answer lists multiple people for the same university, keep only the first one mentioned. If any field is not explicitly present in the answer text, set it to null (or empty list for URLs).

For each university, extract the following fields:
- full_name: The faculty member's full name exactly as written in the answer.
- department: The department/school/college affiliation within the university (e.g., 'School of Computer Science', 'Department of Computer Science & Engineering', 'Robotics Institute', 'Electrical & Computer Engineering', etc.).
- research_focus: A brief phrase/sentence describing the NSF CAREER-funded research focus, as stated in the answer.
- reference_urls: A list of all URLs that the answer cites to support the award claim for this person (university news, department pages, or NSF award pages). Include only valid URLs. If none are present in the answer, return an empty list.

Universities and their JSON keys:
- cmu: Carnegie Mellon University
- umich: University of Michigan
- uiuc: University of Illinois Urbana-Champaign
- uw: University of Washington
- ucla: University of California, Los Angeles

Return a JSON object with these five keys (cmu, umich, uiuc, uw, ucla), each mapping to an object with fields: full_name, department, research_focus, reference_urls.
If the answer does not provide an entry for a particular university, set that university's value to null.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s) and isinstance(s, str) and s.strip() != ""


def _safe_first_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Filter trivial invalid entries
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        # Prepend http if missing protocol (Extractor may handle this, but double-ensure)
        if not uu.startswith("http://") and not uu.startswith("https://"):
            uu = "http://" + uu
        cleaned.append(uu)
    return cleaned


def _recipient_or_default(entry: Optional[RecipientEntry]) -> RecipientEntry:
    return entry if entry is not None else RecipientEntry()


# --------------------------------------------------------------------------- #
# University-specific verification logic                                      #
# --------------------------------------------------------------------------- #
async def verify_university_recipient(
    evaluator: Evaluator,
    parent_node,
    uni_code: str,
    uni_name: str,
    domains: List[str],
    entry: RecipientEntry,
) -> None:
    """
    Build the verification subtree for a single university based on the rubric.
    All individual checks are implemented as leaf or custom nodes, strictly binary.
    """
    # Parallel aggregator for this university recipient
    uni_node = evaluator.add_parallel(
        id=f"{uni_code}_Recipient",
        desc=f"Recipient entry for {uni_name}.",
        parent=parent_node,
        critical=False,  # Non-critical at university node level; allows partial across schools
    )

    # Prepare data
    name = entry.full_name or ""
    dept = entry.department or ""
    focus = entry.research_focus or ""
    urls = _safe_first_urls(entry.reference_urls)

    # 1) Full name provided (critical, existence)
    evaluator.add_custom_node(
        result=_non_empty(name),
        id=f"{uni_code}_Full_Name_Provided",
        desc="Provides the faculty member’s full name.",
        parent=uni_node,
        critical=True,
    )

    # 2) Current affiliation at the specified university (critical, evidence-based)
    aff_leaf = evaluator.add_leaf(
        id=f"{uni_code}_Current_Affiliation_{uni_code}",
        desc=f"Faculty member is currently affiliated with {uni_name}.",
        parent=uni_node,
        critical=True,
    )
    aff_claim = (
        f"The webpage confirms that {name} is affiliated as faculty at {uni_name} "
        f"(e.g., assistant/associate/full professor) at the time of the NSF CAREER award or the announcement."
    )
    await evaluator.verify(
        claim=aff_claim,
        node=aff_leaf,
        sources=urls,
        additional_instruction=(
            "Verify affiliation on the page. Accept reasonable phrasing like 'Assistant Professor at "
            f"{uni_name}', 'faculty in the Department at {uni_name}', or similar. "
            "If multiple people are discussed on the page, focus on the named individual. "
            "If the page is an NSF award page, it should list the PI and their organization matching the university."
        ),
    )

    # 3) Department/school affiliation provided (critical, existence)
    evaluator.add_custom_node(
        result=_non_empty(dept),
        id=f"{uni_code}_Department_Affiliation_Provided",
        desc=f"Provides the department/school affiliation within {uni_name.split(',')[0] if ',' in uni_name else uni_name}.",
        parent=uni_node,
        critical=True,
    )

    # 4) Department is CS/CE/related engineering (critical, evidence-based)
    dept_leaf = evaluator.add_leaf(
        id=f"{uni_code}_Department_Is_CS_CE_or_Related_Engineering",
        desc="Department/school affiliation is within computer science, computer engineering, or a related engineering department.",
        parent=uni_node,
        critical=True,
    )
    dept_claim = (
        f"The page shows that {name}'s department or school is '{dept}' at {uni_name}, "
        "and this affiliation falls within computer science, computer engineering, robotics, "
        "electrical & computer engineering (ECE/EECS), or a closely related engineering/computing field."
    )
    await evaluator.verify(
        claim=dept_claim,
        node=dept_leaf,
        sources=urls,
        additional_instruction=(
            "Check the department/school named on the page. Consider as valid: School/Dept of Computer Science, "
            "Computer Science & Engineering (CSE/CS&E), Electrical & Computer Engineering (ECE/EECS), "
            "Robotics Institute/Program, or equivalent computing-centric units. "
            "Interdisciplinary units are acceptable if clearly AI/CS/CE-centric."
        ),
    )

    # 5) NSF CAREER award year in 2023–2025 inclusive (critical, evidence-based)
    year_leaf = evaluator.add_leaf(
        id=f"{uni_code}_NSF_CAREER_Award_Year_2023_2025",
        desc="Faculty member received an NSF CAREER award with award year between 2023 and 2025 (inclusive).",
        parent=uni_node,
        critical=True,
    )
    year_claim = (
        f"The webpage indicates that {name} received an NSF CAREER award in 2023, 2024, or 2025 "
        "(inclusive). If the exact award year is not explicitly stated, use the announcement/publication date "
        "or the NSF award 'Effective Date' when it corresponds to the CAREER award."
    )
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=urls,
        additional_instruction=YEAR_WINDOW_HINT,
    )

    # 6) Research focus description provided (critical, existence)
    evaluator.add_custom_node(
        result=_non_empty(focus),
        id=f"{uni_code}_Research_Focus_Description_Provided",
        desc="Provides a brief description of the NSF CAREER-funded research focus.",
        parent=uni_node,
        critical=True,
    )

    # 7) Research focus in AI/ML/Robotics/related (critical, evidence-based)
    focus_leaf = evaluator.add_leaf(
        id=f"{uni_code}_Research_Focus_AI_ML_Robotics_Related",
        desc="The described CAREER-funded research focus is in AI, machine learning, robotics, HCI with AI components, or closely related computational areas.",
        parent=uni_node,
        critical=True,
    )
    focus_claim = (
        f"The webpage describes the NSF CAREER project for {name} as being in AI, machine learning, robotics, "
        "HCI with AI components, or a closely related computational area. If multiple projects are mentioned, "
        "focus on the CAREER project specifically."
    )
    await evaluator.verify(
        claim=focus_claim,
        node=focus_leaf,
        sources=urls,
        additional_instruction=(
            f"Consider the following as acceptable scope: {ACCEPTABLE_AREAS_HINT} "
            "Reject if the page only describes unrelated research without an AI/ML/Robotics emphasis."
        ),
    )

    # 8) Reference URL provided (critical, existence)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"{uni_code}_Reference_URL_Provided",
        desc="Provides at least one reference URL for the award claim.",
        parent=uni_node,
        critical=True,
    )

    # 9) Reference URL is official or NSF verifying (critical, evidence-based)
    ref_leaf = evaluator.add_leaf(
        id=f"{uni_code}_Reference_URL_Is_Official_Or_NSF_Verifying",
        desc="Reference URL links to an official university news announcement, department page, or NSF award database entry that verifies the CAREER award.",
        parent=uni_node,
        critical=True,
    )
    domain_hints = ", ".join(sorted(set(domains + ["nsf.gov"])))
    ref_claim = (
        f"The webpage is either an official {uni_name} domain (e.g., university/college/school/department/newsroom) "
        f"or an nsf.gov award page, and it explicitly verifies that {name} received an NSF CAREER award."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=urls,
        additional_instruction=(
            "Check the URL domain and page content. Official examples include domains ending with: "
            f"{domain_hints}. NSF pages include nsf.gov (e.g., awardsearch or funding announcements). "
            "The page must clearly state an NSF CAREER award for the named individual."
        ),
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the NSF CAREER recipients (2023–2025) task.
    Returns a structured summary including the verification tree and final score.
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
        default_model=model,
    )

    # Extraction
    extracted: NSFRecipientsExtraction = await evaluator.extract(
        prompt=prompt_extract_recipients(),
        template_class=NSFRecipientsExtraction,
        extraction_name="nsf_career_recipients_extraction",
    )

    # Add a top-level aggregator node per rubric
    recipients_root = evaluator.add_parallel(
        id="NSF_CAREER_Recipients",
        desc="Identify NSF CAREER recipients (2023–2025) in AI/ML/robotics (or closely related) from each of the five specified universities, with required supporting details and verification URL(s).",
        parent=root,
        critical=False,
    )

    # Ground truth context note (not used as hard GT, just contextual info)
    evaluator.add_custom_info(
        info={
            "universities": [info["university"] for info in UNIVERSITY_INFO.values()],
            "year_window": [2023, 2024, 2025],
            "acceptable_areas_hint": ACCEPTABLE_AREAS_HINT,
        },
        info_type="context",
        info_name="evaluation_context",
    )

    # Verify each university recipient subtree
    uni_entries = [
        ("CMU", UNIVERSITY_INFO["CMU"]["university"], UNIVERSITY_INFO["CMU"]["domains"], _recipient_or_default(extracted.cmu)),
        ("UMich", UNIVERSITY_INFO["UMich"]["university"], UNIVERSITY_INFO["UMich"]["domains"], _recipient_or_default(extracted.umich)),
        ("UIUC", UNIVERSITY_INFO["UIUC"]["university"], UNIVERSITY_INFO["UIUC"]["domains"], _recipient_or_default(extracted.uiuc)),
        ("UW", UNIVERSITY_INFO["UW"]["university"], UNIVERSITY_INFO["UW"]["domains"], _recipient_or_default(extracted.uw)),
        ("UCLA", UNIVERSITY_INFO["UCLA"]["university"], UNIVERSITY_INFO["UCLA"]["domains"], _recipient_or_default(extracted.ucla)),
    ]

    # Build all five subtrees
    for uni_code, uni_name, domains, entry in uni_entries:
        await verify_university_recipient(
            evaluator=evaluator,
            parent_node=recipients_root,
            uni_code=uni_code,
            uni_name=uni_name,
            domains=domains,
            entry=entry,
        )

    # Return summary
    return evaluator.get_summary()