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
TASK_ID = "bitcoin_etf_custody_2026"
TASK_DESCRIPTION = (
    "I'm researching the custody and regulatory framework of Bitcoin spot ETFs in the United States. "
    "Please conduct a comprehensive investigation starting with the largest Bitcoin spot ETF by assets under management as of January 2026. "
    "For this ETF, I need you to:\n\n"
    "1. Identify the ETF's official name and ticker symbol, along with a source URL confirming its position as the largest.\n\n"
    "2. Determine the primary Bitcoin custodian for this ETF and provide the custodian's full legal name, along with a URL from the ETF's prospectus or SEC filing that confirms this custodian relationship.\n\n"
    "3. Identify the regulatory authority that issued the custodian's charter, providing the regulatory body's full name and a URL that confirms this regulatory oversight.\n\n"
    "4. Classify the custodian's charter as either \"state-level\" or \"federal-level\" and provide a URL documenting this charter type.\n\n"
    "5. Identify at least one authorized participant (AP) that facilitates share creation and redemption for this ETF, providing the AP's full legal name and a URL from the ETF's prospectus or SEC filing confirming this role.\n\n"
    "For each piece of information, you must provide supporting URLs from official sources such as the ETF issuer's website, SEC filings, prospectuses, regulatory announcements, or verified financial data providers."
)

ALLOWED_CHARTER_OPTIONS_TEXT = (
    "Allowed charter types are limited to either (a) New York State Department of Financial Services (NYDFS) limited purpose trust "
    "company charter, or (b) Office of the Comptroller of the Currency (OCC) federal charter."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ETFId(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    largest_aum_urls: List[str] = Field(default_factory=list)


class CustodianInfo(BaseModel):
    custodian_name: Optional[str] = None
    custodian_urls: List[str] = Field(default_factory=list)  # Prefer ETF prospectus or SEC filings


class RegulatorInfo(BaseModel):
    regulator_name: Optional[str] = None
    regulator_urls: List[str] = Field(default_factory=list)  # Official regulator or equivalent oversight pages
    charter_type: Optional[str] = None  # e.g., "NYDFS limited purpose trust company", "OCC federal charter"
    charter_level: Optional[str] = None  # "state-level" or "federal-level"
    charter_level_urls: List[str] = Field(default_factory=list)  # URLs documenting charter type/level


class APInfo(BaseModel):
    ap_names: List[str] = Field(default_factory=list)  # At least one AP full legal name
    ap_urls: List[str] = Field(default_factory=list)   # ETF prospectus or SEC filings confirming AP role


class ETFInvestigationExtraction(BaseModel):
    etf: Optional[ETFId] = None
    custodian: Optional[CustodianInfo] = None
    regulator: Optional[RegulatorInfo] = None
    ap: Optional[APInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return (
        "Extract details for the largest U.S. Bitcoin spot ETF by AUM as of January 2026, based solely on the provided answer text.\n"
        "Return a JSON object with the following structure:\n"
        "{\n"
        '  "etf": {\n'
        '    "name": string or null,\n'
        '    "ticker": string or null,\n'
        '    "largest_aum_urls": [list of URLs explicitly present in the answer; may include issuer pages, SEC filings, or verified financial data providers]\n'
        "  },\n"
        '  "custodian": {\n'
        '    "custodian_name": string or null,\n'
        '    "custodian_urls": [list of URLs explicitly present in the answer; strictly prefer ETF prospectuses or SEC filings confirming the custodian]\n'
        "  },\n"
        '  "regulator": {\n'
        '    "regulator_name": string or null,\n'
        '    "regulator_urls": [list of URLs explicitly present in the answer; official regulator or equivalent oversight sources],\n'
        '    "charter_type": string or null,  // e.g., "NYDFS limited purpose trust company" or "OCC federal charter"\n'
        '    "charter_level": string or null, // exactly "state-level" or "federal-level"\n'
        '    "charter_level_urls": [list of URLs explicitly present in the answer; documenting the charter type/level]\n'
        "  },\n"
        '  "ap": {\n'
        '    "ap_names": [list of full legal names of authorized participants explicitly named in the answer],\n'
        '    "ap_urls": [list of URLs explicitly present in the answer; ETF prospectuses or SEC filings confirming AP role]\n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Extract only information explicitly present in the answer as written. Do not infer or invent.\n"
        "- For URLs, extract the actual links (plain or markdown) present in the answer. If missing, return an empty list.\n"
        "- If any field is missing, set it to null (for strings) or empty list (for arrays).\n"
        "- If multiple ETFs are mentioned, choose the one the answer claims is the largest by AUM and extract its details.\n"
        "- Prefer full legal names for organizations.\n"
        "- The charter_level must be exactly 'state-level' or 'federal-level' if provided in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst or []

def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())

def _combine_urls(*url_lists: List[str]) -> List[str]:
    combined: List[str] = []
    for ul in url_lists:
        if ul:
            combined.extend(ul)
    # Remove duplicates while preserving order
    seen = set()
    deduped = []
    for u in combined:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification step builders                                                  #
# --------------------------------------------------------------------------- #
async def build_step_1(
    evaluator: Evaluator,
    parent_node,
    data: ETFInvestigationExtraction,
) -> None:
    etf = data.etf or ETFId()
    etf_name = etf.name or ""
    etf_ticker = etf.ticker or ""
    aum_urls = _safe_list(etf.largest_aum_urls)

    step_node = evaluator.add_parallel(
        id="Step_1_Identify_Largest_Bitcoin_Spot_ETF",
        desc="Identify the U.S. Bitcoin spot ETF with the largest AUM as of January 2026, with supporting evidence.",
        parent=parent_node,
        critical=True,
    )

    # Existence gate
    exists_node = evaluator.add_custom_node(
        result=_non_empty_str(etf_name) and _non_empty_str(etf_ticker) and len(aum_urls) > 0,
        id="Step_1_required_data_present",
        desc="Step 1 required data present: ETF name, ticker, and at least one URL evidencing largest-by-AUM.",
        parent=step_node,
        critical=True,
    )

    # Leaf: Official name and ticker
    name_ticker_node = evaluator.add_leaf(
        id="ETF_Official_Name_and_Ticker",
        desc="Provide the ETF's official name and ticker symbol.",
        parent=step_node,
        critical=True,
    )
    claim_nt = f"The ETF's official name is '{etf_name}' and the ticker symbol is '{etf_ticker}'."
    await evaluator.verify(
        claim=claim_nt,
        node=name_ticker_node,
        sources=aum_urls,
        additional_instruction=(
            "Use the provided URL(s) to confirm the official ETF name and ticker. "
            "Minor naming variants are acceptable if they clearly refer to the same ETF."
        ),
        extra_prerequisites=[exists_node],
    )

    # Leaf: Largest by AUM evidence
    largest_node = evaluator.add_leaf(
        id="Largest_By_AUM_Evidence_URL",
        desc="Provide an official/verified source URL confirming the ETF is the largest by AUM as of January 2026.",
        parent=step_node,
        critical=True,
    )
    claim_largest = (
        f"As of January 2026, the ETF '{etf_name}' ({etf_ticker}) is the largest U.S. Bitcoin spot ETF by assets under management (AUM)."
    )
    await evaluator.verify(
        claim=claim_largest,
        node=largest_node,
        sources=aum_urls,
        additional_instruction=(
            "Confirm that the page(s) report this ETF as the largest by AUM in January 2026. "
            "Accept reputable official sources, issuer disclosures, SEC-linked pages, or verified financial data providers with explicit AUM rankings and dates."
        ),
        extra_prerequisites=[exists_node],
    )


async def build_step_2(
    evaluator: Evaluator,
    parent_node,
    data: ETFInvestigationExtraction,
) -> None:
    etf = data.etf or ETFId()
    cust = data.custodian or CustodianInfo()

    etf_name = etf.name or ""
    cust_name = cust.custodian_name or ""
    cust_urls = _safe_list(cust.custodian_urls)

    step_node = evaluator.add_parallel(
        id="Step_2_Identify_Primary_Bitcoin_Custodian",
        desc="Determine the ETF's primary Bitcoin custodian from official ETF documentation.",
        parent=parent_node,
        critical=True,
    )

    # Existence gate
    exists_node = evaluator.add_custom_node(
        result=_non_empty_str(cust_name) and len(cust_urls) > 0,
        id="Step_2_required_data_present",
        desc="Step 2 required data present: custodian full legal name and at least one prospectus/SEC URL.",
        parent=step_node,
        critical=True,
    )

    # Leaf: Custodian legal name
    cust_name_node = evaluator.add_leaf(
        id="Custodian_Full_Legal_Name",
        desc="Provide the primary Bitcoin custodian's full legal name.",
        parent=step_node,
        critical=True,
    )
    claim_cust = f"The primary Bitcoin custodian for the ETF '{etf_name}' is '{cust_name}'."
    await evaluator.verify(
        claim=claim_cust,
        node=cust_name_node,
        sources=cust_urls,
        additional_instruction=(
            "Confirm the custodian's full legal name using the linked ETF prospectus or SEC filing. "
            "Look for explicit mentions such as 'Bitcoin Custodian', 'Custodian', or similar role statements."
        ),
        extra_prerequisites=[exists_node],
    )

    # Leaf: Prospectus/SEC filing URL confirmation
    cust_url_node = evaluator.add_leaf(
        id="Custodian_Prospectus_or_SEC_Filing_URL",
        desc="Provide a URL from the ETF prospectus or SEC filing confirming the custodian relationship.",
        parent=step_node,
        critical=True,
    )
    claim_cust_url = (
        f"These URL(s) are official ETF prospectuses or SEC filings that explicitly confirm that '{cust_name}' is the custodian of '{etf_name}'."
    )
    await evaluator.verify(
        claim=claim_cust_url,
        node=cust_url_node,
        sources=cust_urls,
        additional_instruction=(
            "Verify that the page(s) are official ETF prospectus materials or SEC filings "
            "and that they explicitly state the custodian relationship for the ETF."
        ),
        extra_prerequisites=[exists_node],
    )


async def build_step_3(
    evaluator: Evaluator,
    parent_node,
    data: ETFInvestigationExtraction,
) -> None:
    etf = data.etf or ETFId()
    cust = data.custodian or CustodianInfo()
    reg = data.regulator or RegulatorInfo()

    etf_name = etf.name or ""
    cust_name = cust.custodian_name or ""
    regulator_name = reg.regulator_name or ""
    regulator_urls = _safe_list(reg.regulator_urls)
    charter_type = reg.charter_type or ""
    charter_level = (reg.charter_level or "").strip().lower()
    charter_level_urls = _safe_list(reg.charter_level_urls)

    step_node = evaluator.add_parallel(
        id="Step_3_Custodian_Charter_Regulator_and_Level",
        desc="Identify the custodian's chartering/oversight authority, enforce allowed charter types, and classify the charter as state-level or federal-level with documentation.",
        parent=parent_node,
        critical=True,
    )

    # Existence gate (requires all fields for this critical step)
    exists_node = evaluator.add_custom_node(
        result=_non_empty_str(regulator_name)
               and len(regulator_urls) > 0
               and _non_empty_str(charter_type)
               and (charter_level in {"state-level", "federal-level"})
               and len(charter_level_urls) > 0,
        id="Step_3_required_data_present",
        desc="Step 3 required data present: regulator name/URLs, charter type, charter level ('state-level' or 'federal-level'), and at least one charter-level documentation URL.",
        parent=step_node,
        critical=True,
    )

    # Leaf: Regulator full name
    regulator_name_node = evaluator.add_leaf(
        id="Charter_Issuing_Regulatory_Authority_Full_Name",
        desc="Provide the full name of the regulatory authority that issued the custodian's charter.",
        parent=step_node,
        critical=True,
    )
    claim_reg_name = f"The custodian '{cust_name}' is chartered/regulated by '{regulator_name}'."
    await evaluator.verify(
        claim=claim_reg_name,
        node=regulator_name_node,
        sources=regulator_urls,
        additional_instruction=(
            "Confirm that the pages are official sources (e.g., regulator websites or announcements) "
            "and that they explicitly identify the regulatory authority for the custodian."
        ),
        extra_prerequisites=[exists_node],
    )

    # Leaf: Regulatory oversight confirmation URL
    oversight_node = evaluator.add_leaf(
        id="Regulatory_Oversight_Confirmation_URL",
        desc="Provide a URL confirming the regulatory authority's oversight/chartering of the custodian (official source).",
        parent=step_node,
        critical=True,
    )
    claim_oversight = (
        f"These URL(s) are official sources that confirm that '{regulator_name}' oversees or chartered the custodian '{cust_name}'."
    )
    await evaluator.verify(
        claim=claim_oversight,
        node=oversight_node,
        sources=regulator_urls,
        additional_instruction=(
            "The page(s) should clearly indicate charter issuance, authorization, or oversight of the custodian by the named authority."
        ),
        extra_prerequisites=[exists_node],
    )

    # Leaf: Allowed charter type constraint
    charter_type_node = evaluator.add_leaf(
        id="Allowed_Charter_Type_Constraint",
        desc="Verify the custodian's charter matches allowed options: NYDFS New York state limited purpose trust company OR OCC federal charter.",
        parent=step_node,
        critical=True,
    )
    claim_charter_type = (
        f"The custodian's charter type '{charter_type}' is valid and matches one of the allowed options: "
        "NYDFS limited purpose trust company charter or OCC federal charter."
    )
    combined_urls = _combine_urls(regulator_urls, charter_level_urls)
    await evaluator.verify(
        claim=claim_charter_type,
        node=charter_type_node,
        sources=combined_urls,
        additional_instruction=(
            f"{ALLOWED_CHARTER_OPTIONS_TEXT} "
            "Use the linked official regulator or documentation pages to determine whether the custodian's charter type corresponds to one of these allowed options."
        ),
        extra_prerequisites=[exists_node],
    )

    # Leaf: Charter level classification
    charter_level_node = evaluator.add_leaf(
        id="Charter_Level_Classification",
        desc="Classify the charter as either 'state-level' or 'federal-level'.",
        parent=step_node,
        critical=True,
    )
    claim_charter_level = f"The custodian '{cust_name}' holds a '{charter_level}' charter level."
    await evaluator.verify(
        claim=claim_charter_level,
        node=charter_level_node,
        sources=combined_urls,
        additional_instruction=(
            "Classify as 'state-level' if charter is issued by a state regulator (e.g., NYDFS). "
            "Classify as 'federal-level' if charter is issued by a federal regulator (e.g., OCC). "
            "Validate classification using the provided documentation URLs."
        ),
        extra_prerequisites=[exists_node],
    )

    # Leaf: Charter level documentation URL
    charter_level_doc_node = evaluator.add_leaf(
        id="Charter_Level_Documentation_URL",
        desc="Provide a URL documenting the charter type/level classification.",
        parent=step_node,
        critical=True,
    )
    claim_charter_level_doc = (
        f"These URL(s) document that the custodian '{cust_name}' has a charter classified as '{charter_level}'."
    )
    await evaluator.verify(
        claim=claim_charter_level_doc,
        node=charter_level_doc_node,
        sources=charter_level_urls,
        additional_instruction=(
            "The documentation should make clear whether the charter is state-level (e.g., NYDFS limited purpose trust company) or federal-level (e.g., OCC)."
        ),
        extra_prerequisites=[exists_node],
    )


async def build_step_4(
    evaluator: Evaluator,
    parent_node,
    data: ETFInvestigationExtraction,
) -> None:
    etf = data.etf or ETFId()
    ap = data.ap or APInfo()

    etf_name = etf.name or ""
    ap_names = ap.ap_names or []
    ap_urls = _safe_list(ap.ap_urls)
    ap_name_first = ap_names[0] if ap_names else ""

    step_node = evaluator.add_parallel(
        id="Step_4_Identify_Authorized_Participant",
        desc="Identify at least one authorized participant (AP) for ETF creation/redemption from official ETF documentation.",
        parent=parent_node,
        critical=True,
    )

    # Existence gate
    exists_node = evaluator.add_custom_node(
        result=_non_empty_str(ap_name_first) and len(ap_urls) > 0,
        id="Step_4_required_data_present",
        desc="Step 4 required data present: at least one AP full legal name and at least one prospectus/SEC URL.",
        parent=step_node,
        critical=True,
    )

    # Leaf: AP full legal name (at least one)
    ap_name_node = evaluator.add_leaf(
        id="AP_Full_Legal_Name_At_Least_One",
        desc="Provide the full legal name of at least one authorized participant.",
        parent=step_node,
        critical=True,
    )
    claim_ap_name = f"The authorized participant for the ETF '{etf_name}' includes '{ap_name_first}'."
    await evaluator.verify(
        claim=claim_ap_name,
        node=ap_name_node,
        sources=ap_urls,
        additional_instruction=(
            "Confirm that the linked ETF prospectus or SEC filing explicitly lists the named firm as an Authorized Participant."
        ),
        extra_prerequisites=[exists_node],
    )

    # Leaf: AP prospectus/SEC filing URL
    ap_url_node = evaluator.add_leaf(
        id="AP_Prospectus_or_SEC_Filing_URL",
        desc="Provide a URL from the ETF prospectus or SEC filing confirming the AP role.",
        parent=step_node,
        critical=True,
    )
    claim_ap_url = (
        f"These URL(s) are official ETF prospectuses or SEC filings that confirm '{ap_name_first}' serves as an Authorized Participant for '{etf_name}'."
    )
    await evaluator.verify(
        claim=claim_ap_url,
        node=ap_url_node,
        sources=ap_urls,
        additional_instruction=(
            "Look for explicit 'Authorized Participant' sections or schedules in the ETF prospectus or SEC filings "
            "that list the firm and its AP role."
        ),
        extra_prerequisites=[exists_node],
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Bitcoin ETF custody and regulatory framework investigation task.
    """
    # Initialize evaluator with a sequential root to reflect step-by-step dependency
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Create a critical top-level node under root to enforce full gating
    main_node = evaluator.add_sequential(
        id="Bitcoin_ETF_Custody_Investigation",
        desc="Investigation of the largest U.S. Bitcoin spot ETF by AUM as of January 2026, including custodian, charter/regulator, charter level, and at least one authorized participant, each supported by official URLs.",
        parent=root,
        critical=True,
    )

    # Extract structured info from the provided answer text
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=ETFInvestigationExtraction,
        extraction_name="bitcoin_etf_custody_extraction",
    )

    # Optionally record allowed charter types as custom info in summary
    evaluator.add_custom_info(
        info={"allowed_charter_types": ["NYDFS limited purpose trust company", "OCC federal charter"]},
        info_type="allowed_charter_types"
    )

    # Build and verify each step according to rubric
    await build_step_1(evaluator, main_node, extracted)
    await build_step_2(evaluator, main_node, extracted)
    await build_step_3(evaluator, main_node, extracted)
    await build_step_4(evaluator, main_node, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()