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
TASK_ID = "turing_2024_genealogy_umass1984"
TASK_DESCRIPTION = """
One of the two recipients of the 2024 ACM A.M. Turing Award completed their doctorate at the University of Massachusetts Amherst in 1984. Trace this person's academic genealogy by identifying:
(1) The name of this 2024 Turing Award recipient,
(2) The year (1984) and institution (University of Massachusetts Amherst) where they completed their PhD,
(3) The name and PhD credentials (year and institution) of their doctoral advisor,
(4) The name and PhD credentials (year and institution) of their advisor's advisor (the 'grand-advisor'),
(5) The current academic affiliation (title, institution, and department) of the grand-advisor, and
(6) The name and PhD credentials (year and institution) of the grand-advisor's advisor (the 'great-grand-advisor').
Provide verifiable URL references from academic databases, university websites, or reliable sources for each person in this academic lineage chain.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RecipientInfo(BaseModel):
    name: Optional[str] = None
    award_urls: List[str] = Field(default_factory=list)

    # PhD credentials
    phd_year: Optional[str] = None
    phd_institution: Optional[str] = None
    phd_urls: List[str] = Field(default_factory=list)


class AdvisorInfo(BaseModel):
    name: Optional[str] = None
    phd_year: Optional[str] = None
    phd_institution: Optional[str] = None
    phd_urls: List[str] = Field(default_factory=list)
    umass_affiliation_urls: List[str] = Field(default_factory=list)  # current or former faculty affiliation at UMass Amherst


class GrandAdvisorInfo(BaseModel):
    name: Optional[str] = None
    phd_year: Optional[str] = None
    phd_institution: Optional[str] = None
    phd_urls: List[str] = Field(default_factory=list)

    # Current affiliation
    current_title: Optional[str] = None  # e.g., "Professor Emeritus"
    current_institution: Optional[str] = None  # e.g., "University of Arizona"
    current_department: Optional[str] = None  # e.g., "Electrical and Computer Engineering"
    current_affiliation_urls: List[str] = Field(default_factory=list)


class GreatGrandAdvisorInfo(BaseModel):
    name: Optional[str] = None
    phd_year: Optional[str] = None
    phd_institution: Optional[str] = None
    phd_urls: List[str] = Field(default_factory=list)


class GenealogyLinks(BaseModel):
    # Dedicated genealogy database URLs for advisor relationships
    recipient_to_advisor_urls: List[str] = Field(default_factory=list)
    advisor_to_grand_urls: List[str] = Field(default_factory=list)
    grand_to_great_urls: List[str] = Field(default_factory=list)
    any_genealogy_db_urls: List[str] = Field(default_factory=list)  # Any URL from a genealogy DB covering the lineage


class GenealogyExtraction(BaseModel):
    recipient: Optional[RecipientInfo] = None
    advisor: Optional[AdvisorInfo] = None
    grand_advisor: Optional[GrandAdvisorInfo] = None
    great_grand_advisor: Optional[GreatGrandAdvisorInfo] = None
    genealogy_links: Optional[GenealogyLinks] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_genealogy() -> str:
    return """
    Extract structured information from the answer about the specified 2024 ACM A.M. Turing Award recipient and their academic genealogy.
    Follow these strict rules:
    - Extract only information explicitly present in the answer.
    - For any missing field, return null (for strings) or an empty list (for URLs).
    - Include full URLs as provided; do not invent or infer URLs.

    Fields to extract:

    recipient:
      - name: The full name of the 2024 ACM A.M. Turing Award recipient in question.
      - award_urls: URLs that explicitly verify this person is a recipient of the 2024 ACM A.M. Turing Award (e.g., ACM site, official announcements).
      - phd_year: The year the recipient received the PhD (should be 1984 for this task).
      - phd_institution: The institution where the recipient received the PhD (should be "University of Massachusetts Amherst" for this task).
      - phd_urls: URLs that verify the recipient's PhD year and institution.

    advisor:
      - name: The name of the recipient's doctoral advisor.
      - phd_year: The year the advisor received the PhD (target: 1975 in this task).
      - phd_institution: The institution where the advisor received the PhD (target: "University of Michigan" in this task).
      - phd_urls: URLs that verify the advisor's PhD year and institution.
      - umass_affiliation_urls: URLs that verify the advisor's current or former faculty affiliation at the University of Massachusetts Amherst.

    grand_advisor:
      - name: The name of the advisor's advisor (grand-advisor).
      - phd_year: The year the grand-advisor received the PhD (target: 1968 in this task).
      - phd_institution: The institution where the grand-advisor received the PhD (target: "University of Michigan" in this task).
      - phd_urls: URLs that verify the grand-advisor's PhD year and institution.
      - current_title: The current academic title (e.g., "Professor Emeritus") of the grand-advisor.
      - current_institution: The current institution (e.g., "University of Arizona").
      - current_department: The current department (e.g., "Electrical and Computer Engineering").
      - current_affiliation_urls: URLs that verify the grand-advisor's current title, institution, and department.

    great_grand_advisor:
      - name: The name of the grand-advisor's advisor (great-grand-advisor).
      - phd_year: The year the great-grand-advisor received the PhD (target: 1959 in this task).
      - phd_institution: The institution where the great-grand-advisor received the PhD (target: "University of Michigan" in this task).
      - phd_urls: URLs that verify the great-grand-advisor's PhD year and institution.

    genealogy_links:
      - recipient_to_advisor_urls: URLs (preferably from Mathematics Genealogy Project or equivalent) that verify recipient → advisor relationship.
      - advisor_to_grand_urls: URLs that verify advisor → grand-advisor relationship.
      - grand_to_great_urls: URLs that verify grand-advisor → great-grand-advisor relationship.
      - any_genealogy_db_urls: Any URLs from a recognized academic genealogy database that cover at least part of the lineage (e.g., mathgenealogy.org, academictree.org). If none, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return s is not None and isinstance(s, str) and s.strip() != ""


def _dedup_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result = []
    for lst in url_lists:
        for u in lst:
            if not isinstance(u, str):
                continue
            uu = u.strip()
            if uu and uu not in seen:
                seen.add(uu)
                result.append(uu)
    return result


def _is_genealogy_db_url(url: str) -> bool:
    u = (url or "").lower()
    return ("mathgenealogy" in u) or ("genealogy" in u and "math" in u) or ("academictree.org" in u) or ("neurotree.org" in u) or ("academic tree" in u)


# --------------------------------------------------------------------------- #
# Verification group functions                                                #
# --------------------------------------------------------------------------- #
async def verify_recipient_identity_and_award(
    evaluator: Evaluator,
    parent_node,
    data: GenealogyExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Recipient_Identity_and_Award",
        desc="Identify the intended 2024 ACM A.M. Turing Award recipient (one of the two recipients) and verify award-recipient status with a URL.",
        parent=parent_node,
        critical=True
    )

    # Recipient name provided (existence)
    name_provided = _non_empty_str(data.recipient.name if data.recipient else None)
    evaluator.add_custom_node(
        result=name_provided,
        id="Recipient_Name_Provided",
        desc="The recipient’s name is explicitly provided.",
        parent=node,
        critical=True
    )

    # Verify 2024 Turing recipient via URLs
    award_urls = data.recipient.award_urls if (data.recipient and data.recipient.award_urls) else []
    leaf = evaluator.add_leaf(
        id="Recipient_Is_2024_Turing_Recipient",
        desc="A provided URL verifies the person is a recipient of the 2024 ACM A.M. Turing Award.",
        parent=node,
        critical=True
    )
    recipient_name = data.recipient.name if data.recipient else ""
    claim = f"{recipient_name} is a recipient of the 2024 ACM A.M. Turing Award."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=award_urls,
        additional_instruction="Confirm the page explicitly states the person is a recipient (winner) of the 2024 ACM A.M. Turing Award."
    )


async def verify_recipient_phd_credentials(
    evaluator: Evaluator,
    parent_node,
    data: GenealogyExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Recipient_PhD_Credentials",
        desc="Verify the recipient’s PhD year and institution with URL evidence.",
        parent=parent_node,
        critical=True
    )

    phd_urls = data.recipient.phd_urls if (data.recipient and data.recipient.phd_urls) else []
    # Year: 1984
    year_leaf = evaluator.add_leaf(
        id="Recipient_PhD_Year_1984",
        desc="A provided URL verifies the recipient received their PhD in 1984.",
        parent=node,
        critical=True
    )
    claim_year = "The recipient received their PhD in 1984."
    await evaluator.verify(
        claim=claim_year,
        node=year_leaf,
        sources=phd_urls,
        additional_instruction="Verify the stated PhD year is 1984 on the provided page(s)."
    )

    # Institution: UMass Amherst
    inst_leaf = evaluator.add_leaf(
        id="Recipient_PhD_Institution_UMass_Amherst",
        desc="A provided URL verifies the recipient received their PhD from the University of Massachusetts Amherst.",
        parent=node,
        critical=True
    )
    claim_inst = "The recipient received their PhD from the University of Massachusetts Amherst."
    await evaluator.verify(
        claim=claim_inst,
        node=inst_leaf,
        sources=phd_urls,
        additional_instruction="Verify that the page explicitly shows University of Massachusetts Amherst as the PhD-granting institution."
    )


async def verify_advisor_details(
    evaluator: Evaluator,
    parent_node,
    data: GenealogyExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Advisor_Details",
        desc="Identify the recipient’s doctoral advisor and verify the advisor’s PhD credentials and required affiliation with URL evidence.",
        parent=parent_node,
        critical=True
    )

    advisor = data.advisor or AdvisorInfo()

    # Advisor name provided
    evaluator.add_custom_node(
        result=_non_empty_str(advisor.name),
        id="Advisor_Name_Provided",
        desc="The doctoral advisor’s name is explicitly provided.",
        parent=node,
        critical=True
    )

    # Advisor PhD year 1975
    adv_year_leaf = evaluator.add_leaf(
        id="Advisor_PhD_Year_1975",
        desc="A provided URL verifies the advisor received their PhD in 1975.",
        parent=node,
        critical=True
    )
    claim = "The advisor received their PhD in 1975."
    await evaluator.verify(
        claim=claim,
        node=adv_year_leaf,
        sources=advisor.phd_urls,
        additional_instruction="Confirm the advisor's PhD year is 1975."
    )

    # Advisor PhD institution UMich
    adv_inst_leaf = evaluator.add_leaf(
        id="Advisor_PhD_Institution_UMich",
        desc="A provided URL verifies the advisor received their PhD from the University of Michigan.",
        parent=node,
        critical=True
    )
    claim = "The advisor received their PhD from the University of Michigan."
    await evaluator.verify(
        claim=claim,
        node=adv_inst_leaf,
        sources=advisor.phd_urls,
        additional_instruction="Confirm the advisor's PhD institution is the University of Michigan."
    )

    # Advisor UMass faculty affiliation (current or former)
    umass_aff_leaf = evaluator.add_leaf(
        id="Advisor_UMass_Faculty_Affiliation",
        desc="A provided URL verifies the advisor has (current or former) faculty affiliation with the University of Massachusetts Amherst (satisfying the 'UMass faculty member' supervision condition).",
        parent=node,
        critical=True
    )
    claim = "The advisor has current or former faculty affiliation at the University of Massachusetts Amherst."
    await evaluator.verify(
        claim=claim,
        node=umass_aff_leaf,
        sources=advisor.umass_affiliation_urls,
        additional_instruction="Accept both current or former faculty appointments (e.g., professor, emeritus, etc.) at UMass Amherst."
    )


async def verify_grand_advisor_details(
    evaluator: Evaluator,
    parent_node,
    data: GenealogyExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Grand_Advisor_Details",
        desc="Identify the grand-advisor (advisor’s advisor) and verify required PhD credentials and current affiliation with URL evidence.",
        parent=parent_node,
        critical=True
    )

    grand = data.grand_advisor or GrandAdvisorInfo()

    # Name provided
    evaluator.add_custom_node(
        result=_non_empty_str(grand.name),
        id="Grand_Advisor_Name_Provided",
        desc="The grand-advisor’s name is explicitly provided.",
        parent=node,
        critical=True
    )

    # PhD year 1968
    year_leaf = evaluator.add_leaf(
        id="Grand_Advisor_PhD_Year_1968",
        desc="A provided URL verifies the grand-advisor received their PhD in 1968.",
        parent=node,
        critical=True
    )
    claim = "The grand-advisor received their PhD in 1968."
    await evaluator.verify(
        claim=claim,
        node=year_leaf,
        sources=grand.phd_urls,
        additional_instruction="Confirm the grand-advisor's PhD year is 1968."
    )

    # PhD institution UMich
    inst_leaf = evaluator.add_leaf(
        id="Grand_Advisor_PhD_Institution_UMich",
        desc="A provided URL verifies the grand-advisor received their PhD from the University of Michigan.",
        parent=node,
        critical=True
    )
    claim = "The grand-advisor received their PhD from the University of Michigan."
    await evaluator.verify(
        claim=claim,
        node=inst_leaf,
        sources=grand.phd_urls,
        additional_instruction="Confirm the grand-advisor's PhD institution is the University of Michigan."
    )

    # Current title: Professor Emeritus
    title_leaf = evaluator.add_leaf(
        id="Grand_Advisor_Current_Title_Professor_Emeritus",
        desc="A provided URL verifies the grand-advisor currently holds the title 'Professor Emeritus'.",
        parent=node,
        critical=True
    )
    claim = "The grand-advisor currently holds the title 'Professor Emeritus'."
    await evaluator.verify(
        claim=claim,
        node=title_leaf,
        sources=grand.current_affiliation_urls,
        additional_instruction="Confirm that the current title includes 'Professor Emeritus'. Minor variations in capitalization are acceptable."
    )

    # Current institution: University of Arizona
    inst_curr_leaf = evaluator.add_leaf(
        id="Grand_Advisor_Current_Institution_UArizona",
        desc="A provided URL verifies the grand-advisor’s current institution is the University of Arizona.",
        parent=node,
        critical=True
    )
    claim = "The grand-advisor’s current institution is the University of Arizona."
    await evaluator.verify(
        claim=claim,
        node=inst_curr_leaf,
        sources=grand.current_affiliation_urls,
        additional_instruction="Confirm that the current institution listed is the University of Arizona."
    )

    # Current department: Electrical and Computer Engineering (ECE)
    dept_leaf = evaluator.add_leaf(
        id="Grand_Advisor_Current_Department_ECE",
        desc="A provided URL verifies the grand-advisor’s current department is Electrical and Computer Engineering.",
        parent=node,
        critical=True
    )
    claim = "The grand-advisor’s current department is Electrical and Computer Engineering."
    await evaluator.verify(
        claim=claim,
        node=dept_leaf,
        sources=grand.current_affiliation_urls,
        additional_instruction="Confirm that the current department stated is Electrical and Computer Engineering (ECE)."
    )


async def verify_great_grand_advisor_details(
    evaluator: Evaluator,
    parent_node,
    data: GenealogyExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Great_Grand_Advisor_Details",
        desc="Identify the great-grand-advisor (grand-advisor’s advisor) and verify required PhD credentials with URL evidence.",
        parent=parent_node,
        critical=True
    )

    gg = data.great_grand_advisor or GreatGrandAdvisorInfo()

    # Name provided
    evaluator.add_custom_node(
        result=_non_empty_str(gg.name),
        id="Great_Grand_Advisor_Name_Provided",
        desc="The great-grand-advisor’s name is explicitly provided.",
        parent=node,
        critical=True
    )

    # PhD year 1959
    year_leaf = evaluator.add_leaf(
        id="Great_Grand_Advisor_PhD_Year_1959",
        desc="A provided URL verifies the great-grand-advisor received their PhD in 1959.",
        parent=node,
        critical=True
    )
    claim = "The great-grand-advisor received their PhD in 1959."
    await evaluator.verify(
        claim=claim,
        node=year_leaf,
        sources=gg.phd_urls,
        additional_instruction="Confirm the great-grand-advisor's PhD year is 1959."
    )

    # PhD institution UMich
    inst_leaf = evaluator.add_leaf(
        id="Great_Grand_Advisor_PhD_Institution_UMich",
        desc="A provided URL verifies the great-grand-advisor received their PhD from the University of Michigan.",
        parent=node,
        critical=True
    )
    claim = "The great-grand-advisor received their PhD from the University of Michigan."
    await evaluator.verify(
        claim=claim,
        node=inst_leaf,
        sources=gg.phd_urls,
        additional_instruction="Confirm the great-grand-advisor's PhD institution is the University of Michigan."
    )


async def verify_genealogy_traceability_and_links(
    evaluator: Evaluator,
    parent_node,
    data: GenealogyExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Genealogy_Traceability_and_Links",
        desc="Verify that the full lineage and advisor relationships are traceable via the Mathematics Genealogy Project (or equivalent academic genealogy database) with URL evidence.",
        parent=parent_node,
        critical=True
    )

    links = data.genealogy_links or GenealogyLinks()

    # Uses recognized genealogy database source (simple URL-domain check)
    all_gen_urls = _dedup_urls(
        links.any_genealogy_db_urls,
        links.recipient_to_advisor_urls,
        links.advisor_to_grand_urls,
        links.grand_to_great_urls
    )
    uses_genealogy_db = any(_is_genealogy_db_url(u) for u in all_gen_urls)
    evaluator.add_custom_node(
        result=uses_genealogy_db,
        id="Uses_Genealogy_Database_Source",
        desc="At least one provided URL is from the Mathematics Genealogy Project or an equivalent academic genealogy database covering the lineage.",
        parent=node,
        critical=True
    )

    # Prepare extra prerequisites across groups (names must be present)
    prereqs = [
        evaluator.find_node("Recipient_Name_Provided"),
        evaluator.find_node("Advisor_Name_Provided"),
        evaluator.find_node("Grand_Advisor_Name_Provided"),
        evaluator.find_node("Great_Grand_Advisor_Name_Provided"),
    ]
    prereqs = [p for p in prereqs if p is not None]

    # Recipient → Advisor
    r_name = data.recipient.name if (data.recipient and data.recipient.name) else ""
    a_name = data.advisor.name if (data.advisor and data.advisor.name) else ""
    leaf_r_a = evaluator.add_leaf(
        id="Recipient_to_Advisor_Link_Verified",
        desc="A provided genealogy-database URL verifies the recipient’s doctoral advisor relationship (recipient → advisor).",
        parent=node,
        critical=True
    )
    claim = f"{r_name} was advised by {a_name} for their doctoral studies."
    await evaluator.verify(
        claim=claim,
        node=leaf_r_a,
        sources=links.recipient_to_advisor_urls,
        additional_instruction="Verify that the source explicitly shows the doctoral advisor/supervisor relationship (recipient → advisor). Allow common synonyms like 'advisor' or 'supervisor'.",
        extra_prerequisites=prereqs
    )

    # Advisor → Grand-Advisor
    g_name = data.grand_advisor.name if (data.grand_advisor and data.grand_advisor.name) else ""
    leaf_a_g = evaluator.add_leaf(
        id="Advisor_to_Grand_Advisor_Link_Verified",
        desc="A provided genealogy-database URL verifies the advisor’s advisor relationship (advisor → grand-advisor).",
        parent=node,
        critical=True
    )
    claim = f"{a_name} was advised by {g_name} for their doctoral studies."
    await evaluator.verify(
        claim=claim,
        node=leaf_a_g,
        sources=links.advisor_to_grand_urls,
        additional_instruction="Verify the doctoral advisor/supervisor relationship (advisor → grand-advisor).",
        extra_prerequisites=prereqs
    )

    # Grand-Advisor → Great-Grand-Advisor
    gg_name = data.great_grand_advisor.name if (data.great_grand_advisor and data.great_grand_advisor.name) else ""
    leaf_g_gg = evaluator.add_leaf(
        id="Grand_to_Great_Grand_Advisor_Link_Verified",
        desc="A provided genealogy-database URL verifies the grand-advisor’s advisor relationship (grand-advisor → great-grand-advisor).",
        parent=node,
        critical=True
    )
    claim = f"{g_name} was advised by {gg_name} for their doctoral studies."
    await evaluator.verify(
        claim=claim,
        node=leaf_g_gg,
        sources=links.grand_to_great_urls,
        additional_instruction="Verify the doctoral advisor/supervisor relationship (grand-advisor → great-grand-advisor).",
        extra_prerequisites=prereqs
    )


async def verify_per_person_urls(
    evaluator: Evaluator,
    parent_node,
    data: GenealogyExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Per_Person_Verifiable_URLs",
        desc="Provide verifiable URL references for each person in the lineage from reliable sources (academic DBs, university pages, or equivalently reliable sources).",
        parent=parent_node,
        critical=True
    )

    recipient_urls = _dedup_urls(
        data.recipient.award_urls if (data.recipient and data.recipient.award_urls) else [],
        data.recipient.phd_urls if (data.recipient and data.recipient.phd_urls) else []
    )
    advisor_urls = _dedup_urls(
        data.advisor.phd_urls if (data.advisor and data.advisor.phd_urls) else [],
        data.advisor.umass_affiliation_urls if (data.advisor and data.advisor.umass_affiliation_urls) else []
    )
    grand_urls = _dedup_urls(
        data.grand_advisor.phd_urls if (data.grand_advisor and data.grand_advisor.phd_urls) else [],
        data.grand_advisor.current_affiliation_urls if (data.grand_advisor and data.grand_advisor.current_affiliation_urls) else []
    )
    great_grand_urls = _dedup_urls(
        data.great_grand_advisor.phd_urls if (data.great_grand_advisor and data.great_grand_advisor.phd_urls) else []
    )

    evaluator.add_custom_node(
        result=len(recipient_urls) > 0,
        id="Recipient_URLs_Provided",
        desc="At least one verifiable URL reference is provided for the recipient.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(advisor_urls) > 0,
        id="Advisor_URLs_Provided",
        desc="At least one verifiable URL reference is provided for the doctoral advisor.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(grand_urls) > 0,
        id="Grand_Advisor_URLs_Provided",
        desc="At least one verifiable URL reference is provided for the grand-advisor.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(great_grand_urls) > 0,
        id="Great_Grand_Advisor_URLs_Provided",
        desc="At least one verifiable URL reference is provided for the great-grand-advisor.",
        parent=node,
        critical=True
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
    Evaluate an answer for the academic genealogy task of the 2024 ACM A.M. Turing Award recipient with UMass Amherst PhD in 1984.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
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

    # Create a critical task node under root (because initialize() creates a non-critical root)
    task_node = evaluator.add_parallel(
        id="Academic_Genealogy_Task",
        desc="Identify the correct 2024 Turing Award recipient and trace their advisor lineage with required PhD credentials, current affiliation, and verifiable sources.",
        parent=root,
        critical=True
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_genealogy(),
        template_class=GenealogyExtraction,
        extraction_name="genealogy_extraction"
    )

    # Build verification tree according to rubric
    await verify_recipient_identity_and_award(evaluator, task_node, extracted)
    await verify_recipient_phd_credentials(evaluator, task_node, extracted)
    await verify_advisor_details(evaluator, task_node, extracted)
    await verify_grand_advisor_details(evaluator, task_node, extracted)
    await verify_great_grand_advisor_details(evaluator, task_node, extracted)
    await verify_genealogy_traceability_and_links(evaluator, task_node, extracted)
    await verify_per_person_urls(evaluator, task_node, extracted)

    # Return structured result
    return evaluator.get_summary()