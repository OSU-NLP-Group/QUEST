import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chips_leading_edge_fabs_4cos"
TASK_DESCRIPTION = (
    "The U.S. government's CHIPS and Science Act has provided substantial funding to semiconductor companies to expand "
    "domestic manufacturing capabilities. As part of analyzing the impact of this initiative, identify 4 distinct "
    "semiconductor companies that meet ALL of the following criteria:\n\n"
    "1. Received a CHIPS and Science Act grant award (preliminary terms or final award) totaling at least $1 billion "
    "across all their U.S. projects (excluding any loans)\n"
    "2. Have at least one U.S. semiconductor manufacturing facility located in Arizona, Ohio, New York, or Texas\n"
    "3. That facility is designated to produce leading-edge logic semiconductors using process technology nodes of "
    "5 nanometers (5nm) or smaller\n"
    "4. That facility utilizes 300mm (12-inch) silicon wafers for production\n\n"
    "For each of the 4 companies you identify, provide:\n"
    "- The company name\n"
    "- The total CHIPS Act grant amount (in billions of dollars, excluding loans)\n"
    "- The complete physical address (street address, city, state, and ZIP code) of one qualifying U.S. facility\n"
    "- The specific process technology node(s) (e.g., 3nm, 2nm) that the facility produces or will produce\n\n"
    "All information must be verifiable through official U.S. Department of Commerce/NIST CHIPS Program Office sources "
    "or official company announcements."
)

ALLOWED_STATES = {"AZ", "Arizona", "OH", "Ohio", "NY", "New York", "TX", "Texas"}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None


class CompanyItem(BaseModel):
    company_name: Optional[str] = None
    # Textual/loose numeric representation is more robust
    total_grant_amount_billion: Optional[str] = None  # exclude loans
    award_type: Optional[str] = None  # e.g., "Preliminary Memorandum of Terms (PMT)" or "Final Award"
    award_date: Optional[str] = None  # any reasonable date format from answer
    facility_address: Optional[FacilityAddress] = None
    process_nodes: List[str] = Field(default_factory=list)  # e.g., ["3nm", "2nm"]
    wafer_size_mm: Optional[str] = None  # e.g., "300mm", "300 mm", "12-inch"
    production_status: Optional[str] = None  # e.g., "construction started 2023", "production 2025", etc.
    source_urls: List[str] = Field(default_factory=list)  # must be official CHIPS/NIST/Commerce/Company announcements


class CompaniesExtraction(BaseModel):
    companies: List[CompanyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_companies() -> str:
    return """
    Extract up to 4 semiconductor companies from the answer that the respondent claims satisfy the CHIPS and Science Act criteria.

    For each company, extract the following fields:
    - company_name: The company's name exactly as provided.
    - total_grant_amount_billion: The total CHIPS ACT GRANT amount in billions of USD (EXCLUDING any loans). If a range or approximate is given, extract the exact text (e.g., "≈3.5", "about 3.5", "3.5"). If multiple awards are mentioned, extract the summed total as presented in the answer. If absent, return null.
    - award_type: The award type string as shown in the answer if provided (e.g., "Preliminary Memorandum of Terms", "PMT", "final award"). If absent, return null.
    - award_date: The award announcement date if provided (any format, e.g., "April 15, 2024" or "2024-04-15"). If absent, return null.
    - facility_address: A structured object with:
        - street: The street address (e.g., "1234 Example Rd."). If absent, return null.
        - city: City name. If absent, return null.
        - state: State name or 2-letter abbreviation. If absent, return null.
        - zip: 5-digit ZIP code (or ZIP+4 if provided). If absent, return null.
    - process_nodes: An array of specific process nodes (e.g., ["3nm","2nm"]). If not explicitly provided, return an empty array.
    - wafer_size_mm: The wafer size text if provided (e.g., "300mm", "300 mm", "12-inch"). If absent, return null.
    - production_status: A brief text indicating whether this is a manufacturing fab and that construction has been initiated or that production is occurring (e.g., "construction started 2023", "volume production 2025"). If absent, return null.
    - source_urls: An array of all URLs cited for this company. Only include URLs explicitly present in the answer text. Prefer official U.S. Dept. of Commerce/NIST CHIPS Program Office pages (e.g., commerce.gov, chips.gov, nist.gov) and official company announcement pages (company domains). Do not invent or add URLs not present in the answer.

    Output a JSON object with a single key "companies" which is an array of at most 4 company objects following the schema. If the answer lists more than 4 companies, include only the first 4 in the output. If fewer than 4 are present, include what exists.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def format_address(addr: Optional[FacilityAddress]) -> str:
    if not addr:
        return ""
    parts = []
    if addr.street: parts.append(addr.street.strip())
    if addr.city: parts.append(addr.city.strip())
    if addr.state and addr.zip:
        parts.append(f"{addr.state.strip()} {addr.zip.strip()}")
    elif addr.state:
        parts.append(addr.state.strip())
    elif addr.zip:
        parts.append(addr.zip.strip())
    return ", ".join([p for p in parts if p])


def sources_for_company(company: CompanyItem) -> List[str]:
    return company.source_urls or []


def is_state_allowed(state_text: Optional[str]) -> bool:
    if not state_text:
        return False
    s = state_text.strip()
    return s in ALLOWED_STATES


def nodes_list_as_text(nodes: List[str]) -> str:
    return ", ".join(nodes) if nodes else ""


# --------------------------------------------------------------------------- #
# Verification for a single company                                           #
# --------------------------------------------------------------------------- #
async def verify_company(
    evaluator: Evaluator,
    parent_node,
    company: CompanyItem,
    index: int,
) -> None:
    """
    Build verification subtree for a single company according to the rubric.
    """
    company_tag = f"company_{index+1}"
    company_desc = f"{index+1}st listed company" if index == 0 else (
        f"{index+1}nd listed company" if index == 1 else (
            f"{index+1}rd listed company" if index == 2 else f"{index+1}th listed company"
        )
    )

    company_node = evaluator.add_parallel(
        id=company_tag,
        desc=f"{company_desc}: satisfies all eligibility criteria and required fields",
        parent=parent_node,
        critical=False
    )

    name = (company.company_name or "").strip()
    address_str = format_address(company.facility_address)
    nodes_text = nodes_list_as_text(company.process_nodes)
    urls = sources_for_company(company)

    # 1) Award type present (PMT/Final award)
    award_type_leaf = evaluator.add_leaf(
        id=f"{company_tag}_award_type_present",
        desc="Company has a CHIPS Act award announcement of an allowed type (preliminary terms/PMT or final award)",
        parent=company_node,
        critical=True
    )
    claim_award_type = (
        f"There is an official CHIPS Program (U.S. Department of Commerce/NIST) or official company announcement for "
        f"{name} indicating a CHIPS Act award of an allowed type: Preliminary Memorandum of Terms (PMT)/preliminary terms "
        f"or a final award."
    )
    await evaluator.verify(
        claim=claim_award_type,
        node=award_type_leaf,
        sources=urls,
        additional_instruction=(
            "Verify that at least one cited page explicitly states a CHIPS award type for the company: either "
            "Preliminary Memorandum of Terms (PMT)/preliminary terms or a final award. "
            "Accept minor naming variations (e.g., 'preliminary terms' vs 'preliminary memorandum of terms')."
        ),
    )

    # 2) Award date deadline (<= Dec 31, 2024)
    award_date_leaf = evaluator.add_leaf(
        id=f"{company_tag}_award_date_deadline",
        desc="The CHIPS award announcement occurred on or before December 31, 2024",
        parent=company_node,
        critical=True
    )
    claim_award_date = (
        f"The CHIPS award announcement for {name} occurred on or before December 31, 2024."
    )
    await evaluator.verify(
        claim=claim_award_date,
        node=award_date_leaf,
        sources=urls,
        additional_instruction=(
            "Use the publication date or explicit date on the cited official announcement (press release, PMT page, "
            "final award page). If multiple awards are mentioned across sources, consider the earliest relevant CHIPS "
            "award announcement date for this company and ensure it is on or before 2024-12-31."
        ),
    )

    # 3) Grant amount threshold (>= $1B, exclude loans)
    grant_leaf = evaluator.add_leaf(
        id=f"{company_tag}_grant_amount_threshold",
        desc="Total CHIPS Act grant amount across all U.S. projects is stated (excluding loans) and is at least $1B",
        parent=company_node,
        critical=True
    )
    claim_grant = (
        f"The total CHIPS Act grant funding (excluding any loans) awarded to {name} across its U.S. projects is "
        f"at least $1 billion."
    )
    await evaluator.verify(
        claim=claim_grant,
        node=grant_leaf,
        sources=urls,
        additional_instruction=(
            "Focus only on direct funding/grants; explicitly exclude loans or loan guarantees. If multiple announcements "
            "exist, consider their sum for direct funding. The statement should be clearly supported (e.g., 'up to $X billion "
            "in direct funding') with X ≥ 1.0."
        ),
    )

    # 4) Facility address and state (AZ/OH/NY/TX) with complete address
    addr_leaf = evaluator.add_leaf(
        id=f"{company_tag}_facility_address_and_state",
        desc="Provides a complete physical address (street, city, state, ZIP) for a qualifying CHIPS-funded manufacturing facility located in AZ, OH, NY, or TX",
        parent=company_node,
        critical=True
    )
    claim_address = (
        f"The qualifying CHIPS-funded semiconductor manufacturing facility for {name} has the following complete address "
        f"(street, city, state, ZIP) and is located in Arizona, Ohio, New York, or Texas: '{address_str}'."
    )
    await evaluator.verify(
        claim=claim_address,
        node=addr_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm that the page(s) explicitly state or clearly show the full street address, including street, city, "
            "state, and ZIP code, and that the state is one of AZ, OH, NY, or TX. The facility must be CHIPS-funded and "
            "a manufacturing fab (not purely R&D). If the provided address cannot be found in the cited sources, mark as not supported."
        ),
    )

    # 5) Facility production status (manufacturing, construction started or production occurring as of Jan 2025)
    prod_status_leaf = evaluator.add_leaf(
        id=f"{company_tag}_facility_production_status",
        desc="Facility is intended for semiconductor production (not purely R&D), with construction initiated or production occurring as of January 2025",
        parent=company_node,
        critical=True
    )
    claim_prod_status = (
        f"The facility for {name} is a manufacturing fab (not purely R&D) and, as of January 2025, construction has "
        f"been initiated or production is (or will imminently be) occurring."
    )
    await evaluator.verify(
        claim=claim_prod_status,
        node=prod_status_leaf,
        sources=urls,
        additional_instruction=(
            "Look for phrasing indicating manufacturing fab, groundbreaking/construction started, or production underway/starting. "
            "Statements like 'construction began in 2023', 'mass production in 2025', or 'manufacturing facility' are sufficient."
        ),
    )

    # 6) Node values provided (existence check)
    nodes_provided = bool(company.process_nodes and len(company.process_nodes) > 0)
    nodes_provided_leaf = evaluator.add_custom_node(
        result=nodes_provided,
        id=f"{company_tag}_node_values_provided",
        desc="Provides the specific process technology node(s) for the facility (e.g., 3nm, 2nm)",
        parent=company_node,
        critical=True
    )

    # 7) Leading-edge logic requirement (≤ 5nm)
    lead_edge_leaf = evaluator.add_leaf(
        id=f"{company_tag}_leading_edge_logic_requirement",
        desc="Facility is designated for leading-edge logic production at process nodes of 5nm or smaller (≤5nm)",
        parent=company_node,
        critical=True
    )
    claim_leading_edge = (
        f"The facility for {name} is designated for leading-edge logic production at process nodes of 5 nanometers or smaller "
        f"(≤5nm). Example nodes listed for this facility: {nodes_text if nodes_text else 'N/A'}."
    )
    await evaluator.verify(
        claim=claim_leading_edge,
        node=lead_edge_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm that the facility's planned/actual production technology is ≤ 5nm (e.g., 5nm, 4nm, 3nm, 2nm, 1.x nm). "
            "Allow common node naming conventions (e.g., 'N3', 'N3B', 'N2')."
        ),
    )

    # 8) Wafer size 300mm
    wafer_leaf = evaluator.add_leaf(
        id=f"{company_tag}_wafer_size",
        desc="Facility utilizes 300mm (12-inch) silicon wafers for production",
        parent=company_node,
        critical=True
    )
    claim_wafer = (
        f"The facility for {name} utilizes 300mm (12-inch) silicon wafers for production."
    )
    await evaluator.verify(
        claim=claim_wafer,
        node=wafer_leaf,
        sources=urls,
        additional_instruction=(
            "Look for explicit mention of '300mm' or '12-inch' wafers in the context of manufacturing at the facility. "
            "If only older 200mm is stated without 300mm manufacturing, mark as not supported."
        ),
    )

    # 9) Source verifiability: official Commerce/NIST CHIPS or official company announcements
    source_verif_leaf = evaluator.add_leaf(
        id=f"{company_tag}_source_verifiability",
        desc="Claims are supported by verifiable official U.S. Dept. of Commerce/NIST CHIPS sources and/or official company announcements (citations/links provided)",
        parent=company_node,
        critical=True
    )
    claim_sources = (
        f"The cited URLs for {name} are official U.S. Department of Commerce/NIST CHIPS Program Office pages "
        f"(e.g., commerce.gov, chips.gov, nist.gov) or are official company announcement/press release pages on the "
        f"company's own domain."
    )
    await evaluator.verify(
        claim=claim_sources,
        node=source_verif_leaf,
        sources=urls,
        additional_instruction=(
            "Judge whether the domain and page context correspond to an official government CHIPS page "
            "(e.g., *.commerce.gov, *.chips.gov, *.nist.gov) or an official company site/press release (company-owned domain). "
            "Ignore third-party news and blogs. If none of the URLs meet this requirement, mark as not supported."
        ),
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
    Evaluate an answer for the CHIPS leading-edge fabs (4 companies) task.
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

    # 1) Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_companies(),
        template_class=CompaniesExtraction,
        extraction_name="companies_extraction",
    )

    # Keep exactly first 4 entries (pad with empty objects if fewer)
    companies: List[CompanyItem] = list(extracted.companies[:4])
    while len(companies) < 4:
        companies.append(CompanyItem())

    # 2) Global distinctness check (critical)
    names = [(c.company_name or "").strip() for c in companies]
    non_empty_names = [n for n in names if n]
    distinct = len(non_empty_names) == 4 and len(set(n.lower() for n in non_empty_names)) == 4
    evaluator.add_custom_node(
        result=distinct,
        id="global_count_distinctness",
        desc="Response provides exactly 4 distinct (non-duplicated) semiconductor companies",
        parent=root,
        critical=True,
    )

    # 3) Per-company verification subtrees
    # Build each company's verification according to rubric
    for idx in range(4):
        await verify_company(evaluator, root, companies[idx], idx)

    # 4) Return summary
    return evaluator.get_summary()