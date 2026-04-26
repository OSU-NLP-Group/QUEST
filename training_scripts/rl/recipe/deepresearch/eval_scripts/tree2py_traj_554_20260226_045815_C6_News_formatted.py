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
TASK_ID = "events_2025_2026"
TASK_DESCRIPTION = """
Between July 2025 and February 2026, five major political and governmental events occurred in the United States and internationally that significantly shaped the political landscape. Research and provide detailed information about the following five events:

1. Corporation for Public Broadcasting Dissolution: Identify the exact date in January 2026 when the CPB board voted to dissolve the organization, specify whether the vote was unanimous, state how many years the organization had been in operation, and provide a reference URL from a news organization documenting this board vote.

2. Special Counsel Jack Smith's Final Actions: Identify the exact date in January 2025 when Special Counsel Jack Smith submitted his final report to the Department of Justice, specify how many volumes the report contained, identify the exact date (three days after the report submission) when he resigned from his position, and provide reference URLs documenting both the report submission and the resignation.

3. Germany Government Formation: Confirm that the German federal election occurred on February 23, 2025, identify the exact date in April 2025 when the CDU/CSU and SPD published their coalition agreement, identify which two party groups formed the coalition, identify the exact date in May 2025 when Friedrich Merz was elected Chancellor by the Bundestag, and provide reference URLs for the election, coalition agreement, and Chancellor election.

4. US-Venezuela Diplomatic Relations Restoration: State how many years diplomatic ties had been severed before restoration, identify the exact date in January 2026 when the US diplomatic representative arrived in Caracas to reopen the mission, provide the full name and official title of this diplomatic representative, and include a reference URL documenting the diplomatic mission reopening.

5. Major Federal Legislation: Identify the popular name of the major federal legislation signed into law on July 4, 2025, provide its official House bill number from the 119th Congress, specify its Public Law designation, and provide a reference URL from an official government source (such as Congress.gov) documenting this legislation.

For each event, ensure all dates, names, titles, and facts are accurate and verifiable through the provided reference URLs.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CPBInfo(BaseModel):
    vote_date: Optional[str] = None
    vote_unanimous: Optional[str] = None  # e.g., "unanimous", "yes", "true", "not unanimous"
    years_in_operation: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class JackSmithInfo(BaseModel):
    report_date: Optional[str] = None
    report_volumes: Optional[str] = None  # prefer strings for robustness (e.g., "2", "two")
    report_urls: List[str] = Field(default_factory=list)
    resignation_date: Optional[str] = None
    resignation_urls: List[str] = Field(default_factory=list)


class GermanyGovInfo(BaseModel):
    election_date: Optional[str] = None
    election_urls: List[str] = Field(default_factory=list)
    coalition_date: Optional[str] = None
    coalition_parties: List[str] = Field(default_factory=list)  # expected two groups
    coalition_urls: List[str] = Field(default_factory=list)
    chancellor_election_date: Optional[str] = None
    chancellor_name: Optional[str] = None
    chancellor_urls: List[str] = Field(default_factory=list)


class USVenezuelaInfo(BaseModel):
    gap_years: Optional[str] = None
    arrival_date: Optional[str] = None
    official_name: Optional[str] = None
    official_title: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class LegislationInfo(BaseModel):
    popular_name: Optional[str] = None
    house_bill_number: Optional[str] = None
    public_law_number: Optional[str] = None
    signing_date: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class EventsExtraction(BaseModel):
    cpb: Optional[CPBInfo] = None
    jack_smith: Optional[JackSmithInfo] = None
    germany_gov: Optional[GermanyGovInfo] = None
    us_venezuela: Optional[USVenezuelaInfo] = None
    legislation: Optional[LegislationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract structured information for five events as presented in the answer. Return a single JSON object matching this schema:

    {
      "cpb": {
        "vote_date": string | null,
        "vote_unanimous": string | null,
        "years_in_operation": string | null,
        "reference_urls": string[]  // all URLs that document the CPB board vote to dissolve
      },
      "jack_smith": {
        "report_date": string | null,            // exact date in January 2025
        "report_volumes": string | null,         // number of volumes (e.g., "2" or "two")
        "report_urls": string[],                 // URLs documenting the report submission
        "resignation_date": string | null,       // exact date in January 2025, expected three days after report submission
        "resignation_urls": string[]             // URLs documenting the resignation
      },
      "germany_gov": {
        "election_date": string | null,          // expected "February 23, 2025"
        "election_urls": string[],               // URLs documenting the federal election date
        "coalition_date": string | null,         // exact date in April 2025 (publication of coalition agreement)
        "coalition_parties": string[],           // list of the two party groups that formed the coalition
        "coalition_urls": string[],              // URLs documenting the coalition agreement
        "chancellor_election_date": string | null, // exact date in May 2025 when Chancellor was elected
        "chancellor_name": string | null,        // name of the person elected Chancellor (expected Friedrich Merz)
        "chancellor_urls": string[]              // URLs documenting the Chancellor election
      },
      "us_venezuela": {
        "gap_years": string | null,              // years diplomatic ties were severed (e.g., "7")
        "arrival_date": string | null,           // exact date in January 2026 when US representative arrived
        "official_name": string | null,          // full name of the diplomatic representative
        "official_title": string | null,         // official title (e.g., "Chargé d'Affaires")
        "reference_urls": string[]               // URLs documenting the mission reopening
      },
      "legislation": {
        "popular_name": string | null,           // popular name of legislation signed July 4, 2025
        "house_bill_number": string | null,      // e.g., "H.R. 1234" from the 119th Congress
        "public_law_number": string | null,      // e.g., "Pub. L. 119-12"
        "signing_date": string | null,           // expected "July 4, 2025"
        "reference_urls": string[]               // official government source URLs (e.g., Congress.gov) documenting this legislation
      }
    }

    Requirements:
    - Extract only from the provided answer. Do not invent any information.
    - Dates must be full dates as stated (include month, day, and year), if available.
    - For boolean-like fields (e.g., unanimous vote), capture the wording used (e.g., "unanimous", "yes", "true", "not unanimous").
    - For URLs, extract actual URL strings (including protocol). Include all relevant URLs mentioned in the answer for each item.
    - If some field is missing, set it explicitly to null. If no URLs are given, return an empty array for that URL field.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _str_or_empty(s: Optional[str]) -> str:
    return s or ""

def _normalize_unanimous_text(value: Optional[str]) -> str:
    v = (value or "").strip().lower()
    if any(tok in v for tok in ["unanim", "yes", "true", "all in favor", "without dissent"]):
        return "unanimous"
    if v == "":
        return ""  # unknown
    return "not unanimous"

def _join_list(items: List[str]) -> str:
    cleaned = [it for it in (items or []) if it]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + " and " + cleaned[-1]


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_cpb_dissolution(evaluator: Evaluator, parent_node, cpb: Optional[CPBInfo]) -> None:
    node = evaluator.add_parallel(
        id="CPB_Dissolution_Event",
        desc="Provide complete information about the Corporation for Public Broadcasting board vote to dissolve the organization",
        parent=parent_node,
        critical=False,
    )

    urls = cpb.reference_urls if (cpb and cpb.reference_urls) else []

    # Leaf: Vote Date
    leaf_date = evaluator.add_leaf(
        id="CPB_Vote_Date",
        desc="Identify the exact date in January 2026 when the CPB board voted to dissolve",
        parent=node,
        critical=True,
    )
    claim_date = f"On {_str_or_empty(cpb.vote_date)} the CPB board voted to dissolve the Corporation for Public Broadcasting."
    # Leaf: Vote Nature
    leaf_nature = evaluator.add_leaf(
        id="CPB_Vote_Nature",
        desc="Specify that the board vote was unanimous",
        parent=node,
        critical=True,
    )
    uni_text = _normalize_unanimous_text(cpb.vote_unanimous if cpb else None)
    if uni_text == "unanimous":
        claim_nature = "The CPB board vote to dissolve was unanimous."
    elif uni_text == "":
        claim_nature = "The CPB board vote to dissolve was unanimous."  # default expectation; if unknown, URL must refute/confirm
    else:
        claim_nature = "The CPB board vote to dissolve was not unanimous."

    # Leaf: Operational Duration
    leaf_years = evaluator.add_leaf(
        id="CPB_Operational_Duration",
        desc="State that CPB had been in operation for 58 years before dissolution",
        parent=node,
        critical=True,
    )
    claim_years = f"CPB had been in operation for {_str_or_empty(cpb.years_in_operation)} years before its dissolution vote."

    # Leaf: Reference URL validity and support
    leaf_ref = evaluator.add_leaf(
        id="CPB_Reference_URL",
        desc="Provide a valid reference URL from a news organization or official source documenting the board vote",
        parent=node,
        critical=True,
    )
    claim_ref = "This page documents the CPB board vote to dissolve the Corporation for Public Broadcasting in January 2026."

    await evaluator.batch_verify([
        (claim_date, urls, leaf_date, "Verify the exact vote date as stated on the page; allow minor formatting variations."),
        (claim_nature, urls, leaf_nature, "Confirm whether the board vote was unanimous; treat 'unanimous'/'without dissent' equivalently."),
        (claim_years, urls, leaf_years, "Confirm years-in-operation stated in the article (e.g., count from founding to dissolution)."),
        (claim_ref, urls, leaf_ref, "The URL must be a credible news organization or official source and explicitly document the board vote."),
    ])


async def verify_jack_smith(evaluator: Evaluator, parent_node, js: Optional[JackSmithInfo]) -> None:
    node = evaluator.add_sequential(
        id="Jack_Smith_Resignation",
        desc="Provide complete information about Special Counsel Jack Smith's resignation and final report",
        parent=parent_node,
        critical=False,
    )

    # Report Submission (critical group)
    rep_node = evaluator.add_parallel(
        id="Report_Submission",
        desc="Identify when Jack Smith submitted his final report in January 2025",
        parent=node,
        critical=True,
    )
    rep_urls = js.report_urls if (js and js.report_urls) else []

    leaf_rep_date = evaluator.add_leaf(
        id="Report_Date",
        desc="Identify the exact date in January 2025 when the report was submitted",
        parent=rep_node,
        critical=True,
    )
    claim_rep_date = f"On {_str_or_empty(js.report_date)} Special Counsel Jack Smith submitted his final report to the Department of Justice."

    leaf_rep_struct = evaluator.add_leaf(
        id="Report_Structure",
        desc="State that the report consisted of two volumes",
        parent=rep_node,
        critical=True,
    )
    claim_rep_struct = f"The final report consisted of {_str_or_empty(js.report_volumes)} volumes."

    leaf_rep_ref = evaluator.add_leaf(
        id="Report_Reference_URL",
        desc="Provide a valid reference URL documenting the report submission",
        parent=rep_node,
        critical=True,
    )
    claim_rep_ref = "This page documents Jack Smith's submission of his final report to the DOJ in January 2025."

    await evaluator.batch_verify([
        (claim_rep_date, rep_urls, leaf_rep_date, "Verify the exact submission date on the page; minor formatting differences allowed."),
        (claim_rep_struct, rep_urls, leaf_rep_struct, "Confirm the report's volume count (e.g., two volumes)."),
        (claim_rep_ref, rep_urls, leaf_rep_ref, "The page should clearly document the report submission event."),
    ])

    # Resignation Details (critical group)
    res_node = evaluator.add_parallel(
        id="Resignation_Details",
        desc="Identify when Jack Smith resigned, which occurred three days after the report submission",
        parent=node,
        critical=True,
    )
    res_urls = js.resignation_urls if (js and js.resignation_urls) else []

    leaf_res_date = evaluator.add_leaf(
        id="Resignation_Date",
        desc="Identify the exact resignation date in January 2025 (three days after the report submission)",
        parent=res_node,
        critical=True,
    )
    claim_res_date = f"Jack Smith resigned on {_str_or_empty(js.resignation_date)}."
    leaf_res_ref = evaluator.add_leaf(
        id="Resignation_Reference_URL",
        desc="Provide a valid reference URL documenting the resignation",
        parent=res_node,
        critical=True,
    )
    claim_res_ref = "This page documents Special Counsel Jack Smith's resignation."

    await evaluator.batch_verify([
        (claim_res_date, res_urls, leaf_res_date, "Verify the resignation date stated on the page."),
        (claim_res_ref, res_urls, leaf_res_ref, "Page must explicitly document the resignation event."),
    ])


async def verify_germany_formation(evaluator: Evaluator, parent_node, de: Optional[GermanyGovInfo]) -> None:
    node = evaluator.add_sequential(
        id="Germany_Government_Formation",
        desc="Provide complete information about Germany's 2025 government formation process following the federal election",
        parent=parent_node,
        critical=False,
    )

    # Federal Election (critical)
    ele_node = evaluator.add_parallel(
        id="Federal_Election",
        desc="Confirm the federal election date",
        parent=node,
        critical=True,
    )
    ele_urls = de.election_urls if (de and de.election_urls) else []

    leaf_ele_date = evaluator.add_leaf(
        id="Election_Date",
        desc="Confirm that the election occurred on February 23, 2025",
        parent=ele_node,
        critical=True,
    )
    # Use fixed date claim per rubric; allows robust checking even if answer's date varies
    claim_ele_date = "The German federal election occurred on February 23, 2025."
    leaf_ele_ref = evaluator.add_leaf(
        id="Election_Reference_URL",
        desc="Provide a valid reference URL documenting the election",
        parent=ele_node,
        critical=True,
    )
    claim_ele_ref = "This page documents the 2025 German federal election date."

    await evaluator.batch_verify([
        (claim_ele_date, ele_urls, leaf_ele_date, "Allow German-language sources; confirm the date explicitly."),
        (claim_ele_ref, ele_urls, leaf_ele_ref, "The page must be relevant and document the election date."),
    ])

    # Coalition Agreement (critical)
    coa_node = evaluator.add_parallel(
        id="Coalition_Agreement",
        desc="Identify when the CDU/CSU and SPD published their coalition agreement",
        parent=node,
        critical=True,
    )
    coa_urls = de.coalition_urls if (de and de.coalition_urls) else []

    leaf_coa_date = evaluator.add_leaf(
        id="Agreement_Date",
        desc="Identify the exact date in April 2025 when the coalition agreement was published",
        parent=coa_node,
        critical=True,
    )
    claim_coa_date = f"The coalition agreement was published on {_str_or_empty(de.coalition_date)} in April 2025."

    leaf_coa_parties = evaluator.add_leaf(
        id="Coalition_Parties",
        desc="Identify the two party groups that formed the coalition",
        parent=coa_node,
        critical=True,
    )
    parties_text = _join_list(de.coalition_parties if de else [])
    claim_coa_parties = f"The coalition was formed by {parties_text}."

    leaf_coa_ref = evaluator.add_leaf(
        id="Agreement_Reference_URL",
        desc="Provide a valid reference URL documenting the coalition agreement",
        parent=coa_node,
        critical=True,
    )
    claim_coa_ref = "This page documents the coalition agreement publication and the parties involved (CDU/CSU and SPD)."

    await evaluator.batch_verify([
        (claim_coa_date, coa_urls, leaf_coa_date, "Confirm the publication date in April 2025; allow minor formatting variants."),
        (claim_coa_parties, coa_urls, leaf_coa_parties, "Verify the two coalition party groups named on the page."),
        (claim_coa_ref, coa_urls, leaf_coa_ref, "Page must explicitly document the coalition agreement details."),
    ])

    # Chancellor Election (critical)
    ch_node = evaluator.add_parallel(
        id="Chancellor_Election",
        desc="Identify when the Chancellor was elected by the Bundestag",
        parent=node,
        critical=True,
    )
    ch_urls = de.chancellor_urls if (de and de.chancellor_urls) else []

    leaf_ch_date = evaluator.add_leaf(
        id="Chancellor_Election_Date",
        desc="Identify the exact date in May 2025 when the Chancellor was elected by the Bundestag",
        parent=ch_node,
        critical=True,
    )
    claim_ch_date = f"The Chancellor was elected by the Bundestag on {_str_or_empty(de.chancellor_election_date)}."

    leaf_ch_name = evaluator.add_leaf(
        id="Chancellor_Name",
        desc="Identify who was elected as Chancellor",
        parent=ch_node,
        critical=True,
    )
    # Use extracted name to validate the answer against sources
    claim_ch_name = f"{_str_or_empty(de.chancellor_name)} was elected Chancellor by the Bundestag."

    leaf_ch_ref = evaluator.add_leaf(
        id="Chancellor_Reference_URL",
        desc="Provide a valid reference URL documenting the Chancellor election",
        parent=ch_node,
        critical=True,
    )
    claim_ch_ref = "This page documents the Bundestag's election of the Chancellor in May 2025."

    await evaluator.batch_verify([
        (claim_ch_date, ch_urls, leaf_ch_date, "Confirm the exact Bundestag election date for the Chancellor."),
        (claim_ch_name, ch_urls, leaf_ch_name, "Verify the person elected Chancellor as stated."),
        (claim_ch_ref, ch_urls, leaf_ch_ref, "Page must explicitly document the Chancellor election."),
    ])


async def verify_us_venezuela(evaluator: Evaluator, parent_node, uv: Optional[USVenezuelaInfo]) -> None:
    node = evaluator.add_parallel(
        id="US_Venezuela_Diplomatic_Restoration",
        desc="Provide complete information about the restoration of US-Venezuela diplomatic relations",
        parent=parent_node,
        critical=False,
    )
    urls = uv.reference_urls if (uv and uv.reference_urls) else []

    # Diplomatic gap duration
    leaf_gap = evaluator.add_leaf(
        id="Diplomatic_Gap_Duration",
        desc="State that diplomatic ties had been severed for 7 years before restoration",
        parent=node,
        critical=True,
    )
    claim_gap = f"Diplomatic ties between the U.S. and Venezuela had been severed for {_str_or_empty(uv.gap_years)} years before restoration."

    # Arrival group (critical)
    arr_node = evaluator.add_parallel(
        id="Charge_Affairs_Arrival",
        desc="Identify when the US Chargé d'Affaires arrived in Caracas",
        parent=node,
        critical=True,
    )

    leaf_arrival_date = evaluator.add_leaf(
        id="Arrival_Date",
        desc="Identify the exact date in January 2026 when the diplomatic representative arrived in Caracas",
        parent=arr_node,
        critical=True,
    )
    claim_arrival_date = f"The U.S. diplomatic representative arrived in Caracas on {_str_or_empty(uv.arrival_date)} to reopen the mission."

    leaf_official_name = evaluator.add_leaf(
        id="Official_Name",
        desc="Provide the full name of the Chargé d'Affaires",
        parent=arr_node,
        critical=True,
    )
    claim_official_name = f"The U.S. diplomatic representative was {_str_or_empty(uv.official_name)}."

    leaf_official_title = evaluator.add_leaf(
        id="Official_Title",
        desc="Provide the official title of the diplomatic representative",
        parent=arr_node,
        critical=True,
    )
    claim_official_title = f"The official title of the U.S. representative was {_str_or_empty(uv.official_title)}."

    leaf_dip_ref = evaluator.add_leaf(
        id="Diplomatic_Reference_URL",
        desc="Provide a valid reference URL documenting the diplomatic mission reopening",
        parent=arr_node,
        critical=True,
    )
    claim_dip_ref = "This page documents the reopening of the U.S. diplomatic mission in Caracas in January 2026."

    await evaluator.verify(claim_gap, leaf_gap, sources=urls,
                           additional_instruction="Verify the stated duration (years) of severed diplomatic ties before restoration.")
    await evaluator.batch_verify([
        (claim_arrival_date, urls, leaf_arrival_date, "Confirm the arrival date and mission reopening."),
        (claim_official_name, urls, leaf_official_name, "Verify the full name of the U.S. diplomatic representative."),
        (claim_official_title, urls, leaf_official_title, "Verify the official title, e.g., 'Chargé d'Affaires'."),
        (claim_dip_ref, urls, leaf_dip_ref, "Page must explicitly document the mission reopening event."),
    ])


async def verify_legislation(evaluator: Evaluator, parent_node, leg: Optional[LegislationInfo]) -> None:
    node = evaluator.add_parallel(
        id="Major_Federal_Legislation",
        desc="Provide complete information about the major federal legislation signed on July 4, 2025",
        parent=parent_node,
        critical=False,
    )
    urls = leg.reference_urls if (leg and leg.reference_urls) else []

    # Popular name
    leaf_pop = evaluator.add_leaf(
        id="Bill_Popular_Name",
        desc="Identify the popular name of the legislation",
        parent=node,
        critical=True,
    )
    claim_pop = f"The popular name of the legislation signed on July 4, 2025 was '{_str_or_empty(leg.popular_name)}'."

    # Official designation (critical group)
    des_node = evaluator.add_parallel(
        id="Bill_Official_Designation",
        desc="Identify the official congressional designation of the legislation",
        parent=node,
        critical=True,
    )
    leaf_house = evaluator.add_leaf(
        id="House_Bill_Number",
        desc="Provide the House bill number from the 119th Congress",
        parent=des_node,
        critical=True,
    )
    claim_house = f"The House bill number from the 119th Congress was {_str_or_empty(leg.house_bill_number)}."
    leaf_pl = evaluator.add_leaf(
        id="Public_Law_Number",
        desc="Provide the Public Law designation",
        parent=des_node,
        critical=True,
    )
    claim_pl = f"The Public Law designation was {_str_or_empty(leg.public_law_number)}."

    # Signing date
    leaf_sign = evaluator.add_leaf(
        id="Bill_Signing_Date",
        desc="Confirm the signing date as July 4, 2025",
        parent=node,
        critical=True,
    )
    # Use fixed date per rubric
    claim_sign = "The legislation was signed into law on July 4, 2025."

    # Reference URL must be official government source
    leaf_ref = evaluator.add_leaf(
        id="Legislation_Reference_URL",
        desc="Provide a valid reference URL from an official government source documenting the legislation",
        parent=node,
        critical=True,
    )
    claim_ref = "This page is an official U.S. government source (e.g., Congress.gov) that documents this legislation and its details."

    await evaluator.batch_verify([
        (claim_pop, urls, leaf_pop, "Confirm the popular name on the official page; allow minor naming variants."),
        (claim_house, urls, leaf_house, "Verify the House bill number against the official legislative page."),
        (claim_pl, urls, leaf_pl, "Verify the Public Law number/designation on the official page."),
        (claim_sign, urls, leaf_sign, "Confirm that the signing date is July 4, 2025."),
        (claim_ref, urls, leaf_ref, "Ensure the URL is an official government source (Congress.gov, GovInfo, etc.) documenting the legislation."),
    ])


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
    Evaluate the answer for the five major events (July 2025 – February 2026).
    Builds a verification tree according to the rubric and returns a structured summary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel across events
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

    # Extract structured event data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction",
    )

    # Build and verify each event subtree
    await verify_cpb_dissolution(evaluator, root, extracted.cpb or CPBInfo())
    await verify_jack_smith(evaluator, root, extracted.jack_smith or JackSmithInfo())
    await verify_germany_formation(evaluator, root, extracted.germany_gov or GermanyGovInfo())
    await verify_us_venezuela(evaluator, root, extracted.us_venezuela or USVenezuelaInfo())
    await verify_legislation(evaluator, root, extracted.legislation or LegislationInfo())

    # Return structured summary with verification tree and score
    return evaluator.get_summary()