import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "or_investment_advisors_cfp_series65"
TASK_DESCRIPTION = (
    "Identify 2 investment advisor representatives who are currently registered in Oregon and meet all of the "
    "following qualifications: (1) hold active CFP® certification from CFP Board, (2) have passed the Series 65 exam, "
    "(3) are currently registered as investment advisor representatives in Oregon, and (4) have no recent disciplinary "
    "actions on record. For each advisor, provide full name, current firm affiliation, and reference URLs to verify: "
    "CFP certification status (CFP Board's 'Verify a CFP Professional' tool), Series 65 licensing and Oregon "
    "registration status (FINRA BrokerCheck or SEC IAPD), and a clean disciplinary record."
)

# Domain/URL checks for required source categories
CFP_DOMAIN_KEYWORDS = ["cfp.net/verify", "cfp.net/verify-a-cfp", "cfp.net"]
SERIES65_DOMAIN_KEYWORDS = ["brokercheck.finra.org", "adviserinfo.sec.gov", "sec.gov/iapd"]
OREGON_REG_DOMAIN_KEYWORDS = ["brokercheck.finra.org", "adviserinfo.sec.gov", "oregon.gov"]
DISCIPLINE_DOMAIN_KEYWORDS = ["brokercheck.finra.org", "adviserinfo.sec.gov", "cfp.net"]


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class Advisor(BaseModel):
    full_name: Optional[str] = None
    firm: Optional[str] = None
    cfp_urls: List[str] = Field(default_factory=list)
    series65_urls: List[str] = Field(default_factory=list)
    oregon_urls: List[str] = Field(default_factory=list)
    discipline_urls: List[str] = Field(default_factory=list)


class AdvisorsExtraction(BaseModel):
    advisors: List[Advisor] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_advisors() -> str:
    return """
    Extract information for up to two investment advisor representatives presented in the answer who are currently registered in Oregon.
    For each advisor, extract the following fields:
    - full_name: Full name of the advisor as presented in the answer.
    - firm: Current firm affiliation of the advisor as presented in the answer.
    - cfp_urls: An array of URL(s) used to verify CFP® certification status; these should be from the CFP Board's verification tool (e.g., https://www.cfp.net/verify-a-cfp-professional or specific result pages on cfp.net).
    - series65_urls: An array of URL(s) from FINRA BrokerCheck or SEC IAPD that can verify Series 65 (Uniform Investment Adviser Law Examination) passage status.
    - oregon_urls: An array of URL(s) from FINRA BrokerCheck, SEC IAPD, or a relevant Oregon state site that can verify current registration as an Investment Adviser Representative in Oregon (OR).
    - discipline_urls: An array of URL(s) from FINRA BrokerCheck, SEC IAPD, or CFP Board that can verify the disciplinary record status for the advisor.

    Rules:
    1) Extract only URLs explicitly present in the answer. If the answer uses markdown links, extract the actual URLs.
    2) If the same URL is cited for multiple purposes, include it in each relevant array.
    3) If any field is missing, set it to null (for strings) or an empty array (for URL arrays).
    4) Return the first two advisors if more than two are mentioned.

    Return a JSON object with a single top-level key:
    {
      "advisors": [
        {
          "full_name": string | null,
          "firm": string | null,
          "cfp_urls": string[],
          "series65_urls": string[],
          "oregon_urls": string[],
          "discipline_urls": string[]
        }
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_url_from(urls: List[str], domain_keywords: List[str]) -> bool:
    """Return True if any url contains one of the domain keywords (case-insensitive)."""
    for u in urls:
        if not u:
            continue
        lu = u.lower()
        for kw in domain_keywords:
            if kw in lu:
                return True
    return False


def uniq_urls(*lists: List[str]) -> List[str]:
    """Return a de-duplicated list preserving order."""
    seen = set()
    out = []
    for lst in lists:
        for u in lst:
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_advisor(
    evaluator: Evaluator,
    parent_node,
    advisor: Advisor,
    index: int,
) -> None:
    """
    Build the verification subtree for a single advisor and run verifications.
    """
    idx_disp = index + 1
    advisor_node = evaluator.add_parallel(
        id=f"advisor_{idx_disp}",
        desc=f"{'First' if index == 0 else 'Second'} qualified investment advisor representative",
        parent=parent_node,
        critical=False
    )

    name = advisor.full_name or ""
    firm = advisor.firm or ""
    cfp_urls = advisor.cfp_urls or []
    series_urls = advisor.series65_urls or []
    oregon_urls = advisor.oregon_urls or []
    discipline_urls = advisor.discipline_urls or []
    all_verif_urls = uniq_urls(cfp_urls, series_urls, oregon_urls, discipline_urls)

    # --------------------- Basic Information (Critical) --------------------- #
    basic_node = evaluator.add_parallel(
        id=f"advisor_{idx_disp}_basic",
        desc="Basic identifying information for the advisor",
        parent=advisor_node,
        critical=True
    )
    # Full name provided
    evaluator.add_custom_node(
        result=bool(name.strip()),
        id=f"advisor_{idx_disp}_full_name",
        desc="Full name of the advisor is provided",
        parent=basic_node,
        critical=True
    )
    # Firm provided
    evaluator.add_custom_node(
        result=bool(firm.strip()),
        id=f"advisor_{idx_disp}_firm",
        desc="Current firm affiliation of the advisor is provided",
        parent=basic_node,
        critical=True
    )

    # --------------------- CFP Certification (Critical) --------------------- #
    cfp_node = evaluator.add_parallel(
        id=f"advisor_{idx_disp}_cfp",
        desc="CFP certification verification",
        parent=advisor_node,
        critical=True
    )
    # CFP URL provided (must be from CFP Board domain)
    evaluator.add_custom_node(
        result=has_url_from(cfp_urls, CFP_DOMAIN_KEYWORDS),
        id=f"advisor_{idx_disp}_cfp_url",
        desc="Reference URL to CFP Board verification tool is provided to verify CFP certification status",
        parent=cfp_node,
        critical=True
    )
    # CFP active status
    cfp_status_leaf = evaluator.add_leaf(
        id=f"advisor_{idx_disp}_cfp_status",
        desc="Advisor holds active CFP® certification from CFP Board",
        parent=cfp_node,
        critical=True
    )
    cfp_claim = f"{name} holds an active CFP certification according to the CFP Board verification page."
    await evaluator.verify(
        claim=cfp_claim,
        node=cfp_status_leaf,
        sources=cfp_urls,
        additional_instruction=(
            "Use the CFP Board 'Verify a CFP Professional' page(s). Consider the claim supported if the page "
            "explicitly shows the person is a CFP professional with a current/active certification status. "
            "Allow minor name formatting differences but ensure it refers to the same individual."
        )
    )

    # --------------- Series 65 Exam & Licensing (Critical) ------------------ #
    s65_node = evaluator.add_parallel(
        id=f"advisor_{idx_disp}_series65",
        desc="Series 65 exam and licensing verification",
        parent=advisor_node,
        critical=True
    )
    # Series 65 URL provided (FINRA BrokerCheck or SEC IAPD)
    evaluator.add_custom_node(
        result=has_url_from(series_urls, SERIES65_DOMAIN_KEYWORDS),
        id=f"advisor_{idx_disp}_series65_url",
        desc="Reference URL to FINRA BrokerCheck or SEC IAPD is provided to verify Series 65 licensing",
        parent=s65_node,
        critical=True
    )
    # Series 65 status: passed
    s65_status_leaf = evaluator.add_leaf(
        id=f"advisor_{idx_disp}_series65_status",
        desc="Advisor has passed the Series 65 exam (Uniform Investment Adviser Law Examination)",
        parent=s65_node,
        critical=True
    )
    s65_claim = f"{name} has passed the Series 65 exam (Uniform Investment Adviser Law Examination)."
    await evaluator.verify(
        claim=s65_claim,
        node=s65_status_leaf,
        sources=series_urls,
        additional_instruction=(
            "Check the FINRA BrokerCheck or SEC IAPD page. The claim is supported only if the page shows the "
            "Series 65 exam (Uniform Investment Adviser Law Examination) with a 'Passed' or equivalent status. "
            "Do NOT count exam waivers or exemptions (e.g., due to certain professional designations) as 'passed.'"
        )
    )

    # ---------------- Oregon IAR Registration (Critical) -------------------- #
    or_node = evaluator.add_parallel(
        id=f"advisor_{idx_disp}_oregon",
        desc="Oregon IAR registration verification",
        parent=advisor_node,
        critical=True
    )
    # Oregon registration URL provided
    evaluator.add_custom_node(
        result=has_url_from(oregon_urls, OREGON_REG_DOMAIN_KEYWORDS),
        id=f"advisor_{idx_disp}_oregon_url",
        desc="Reference URL to FINRA BrokerCheck or SEC IAPD is provided to verify Oregon registration status",
        parent=or_node,
        critical=True
    )
    # Oregon current registration status
    or_status_leaf = evaluator.add_leaf(
        id=f"advisor_{idx_disp}_oregon_status",
        desc="Advisor is currently registered in Oregon as an investment advisor representative",
        parent=or_node,
        critical=True
    )
    or_claim = f"{name} is currently registered in Oregon as an Investment Adviser Representative (IAR)."
    await evaluator.verify(
        claim=or_claim,
        node=or_status_leaf,
        sources=oregon_urls,
        additional_instruction=(
            "Use FINRA BrokerCheck or SEC IAPD (or an Oregon state regulator page) to verify that Oregon (OR) "
            "appears among the current state registrations for the person as an Investment Adviser Representative. "
            "Look for 'Oregon' or 'OR' specifically and ensure the registration is active/current."
        )
    )

    # ----------------- Disciplinary Record (Critical) ----------------------- #
    disc_node = evaluator.add_parallel(
        id=f"advisor_{idx_disp}_discipline",
        desc="Disciplinary record verification",
        parent=advisor_node,
        critical=True
    )
    # Discipline URL provided (BrokerCheck, IAPD, or CFP Board)
    evaluator.add_custom_node(
        result=has_url_from(discipline_urls, DISCIPLINE_DOMAIN_KEYWORDS),
        id=f"advisor_{idx_disp}_discipline_url",
        desc="Reference URL to FINRA BrokerCheck, SEC IAPD, or CFP Board is provided to verify clean disciplinary record",
        parent=disc_node,
        critical=True
    )
    # Clean record status
    clean_leaf = evaluator.add_leaf(
        id=f"advisor_{idx_disp}_clean_record",
        desc="Advisor has no recent disciplinary actions, customer complaints, regulatory sanctions, or bankruptcy disclosures on record",
        parent=disc_node,
        critical=True
    )
    clean_claim = (
        f"{name} has no recent disciplinary actions, customer complaints, regulatory sanctions, or bankruptcy disclosures reported."
    )
    await evaluator.verify(
        claim=clean_claim,
        node=clean_leaf,
        sources=uniq_urls(discipline_urls, series_urls, oregon_urls, cfp_urls),
        additional_instruction=(
            "On the BrokerCheck or SEC IAPD profile, look for sections about disclosures or disciplinary events. "
            "Consider the claim supported if the profile indicates 'No disclosures', 'No events', or otherwise "
            "clearly shows no disciplinary actions/complaints/sanctions/bankruptcy. If any relevant disclosures "
            "are present, the claim is not supported."
        )
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
    Evaluate an answer for the Oregon IAR + CFP + Series 65 + clean record task.
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

    # Extract advisors data
    extracted = await evaluator.extract(
        prompt=prompt_extract_advisors(),
        template_class=AdvisorsExtraction,
        extraction_name="advisors_extraction"
    )

    # Prepare exactly two advisor entries (pad if fewer)
    advisors: List[Advisor] = list(extracted.advisors[:2])
    while len(advisors) < 2:
        advisors.append(Advisor())

    # Add criteria summary as ground truth info for context
    evaluator.add_ground_truth({
        "required_count": 2,
        "criteria": [
            "Active CFP® certification from CFP Board (verify on cfp.net)",
            "Passed Series 65 exam (verify on FINRA BrokerCheck or SEC IAPD)",
            "Currently registered as IAR in Oregon (verify on BrokerCheck/IAPD or Oregon regulator site)",
            "No recent disciplinary actions/complaints/sanctions/bankruptcy (verify on BrokerCheck/IAPD/CFP Board)"
        ],
        "required_fields": ["full_name", "firm", "cfp_urls", "series65_urls", "oregon_urls", "discipline_urls"]
    }, gt_type="criteria")

    # Build child nodes for each advisor (parallel at root)
    for i in range(2):
        await verify_single_advisor(evaluator, root, advisors[i], i)

    return evaluator.get_summary()