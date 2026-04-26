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
TASK_ID = "venezuela_ops_2026_comprehensive_summary"
TASK_DESCRIPTION = """
Following the US military operation in Venezuela in early 2026, provide a comprehensive summary of the key developments including: 
(1) the exact date when the operation to capture Nicolás Maduro occurred, 
(2) who became interim president and when they were sworn in, 
(3) who serves as President of Venezuela's National Assembly, 
(4) which US oil company was the only one operating in Venezuela before the general licenses expansion in mid-February, 
(5) which five major oil companies were subsequently authorized to operate in Venezuela through OFAC general licenses, 
(6) the status of Venezuelan oil revenue by mid-February 2026 including the total amount and where it was initially deposited, and 
(7) when the amnesty bill for political prisoners passed its first reading in the National Assembly. 
Please provide reference URLs for each piece of information.
"""

# Expected facts used for simple checks
EXPECTED_OPERATION_DATE = "January 3, 2026"
EXPECTED_OPERATION_LOCATION = "Caracas, Venezuela"

EXPECTED_INTERIM_PRESIDENT = "Delcy Rodríguez"
EXPECTED_INTERIM_SWEARING_DATE = "January 5, 2026"

EXPECTED_ASSEMBLY_PRESIDENT = "Jorge Rodríguez"

EXPECTED_SOLE_US_COMPANY = "Chevron"

EXPECTED_GENERAL_LICENSES = ["GL 46", "GL 46A", "GL 47", "GL 48", "GL 49"]
EXPECTED_AUTHORIZED_COMPANIES = ["BP", "Chevron", "Eni", "Repsol", "Shell"]

EXPECTED_AMNESTY_FIRST_READING_DATE = "February 5, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OperationDetailsExtraction(BaseModel):
    date: Optional[str] = None
    location: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class InterimPresidentExtraction(BaseModel):
    name: Optional[str] = None
    sworn_in_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AssemblyPresidentExtraction(BaseModel):
    president_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PreExpansionCompanyExtraction(BaseModel):
    company_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OFACLicensesExtraction(BaseModel):
    license_identifiers: List[str] = Field(default_factory=list)
    time_window: Optional[str] = None
    authorized_companies: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class OilRevenueExtraction(BaseModel):
    total_amount: Optional[str] = None
    initial_deposit_location: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AmnestyLegislationExtraction(BaseModel):
    first_reading_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class VenezuelaUpdateExtraction(BaseModel):
    operation: Optional[OperationDetailsExtraction] = None
    interim_president: Optional[InterimPresidentExtraction] = None
    assembly: Optional[AssemblyPresidentExtraction] = None
    pre_expansion_company: Optional[PreExpansionCompanyExtraction] = None
    ofac: Optional[OFACLicensesExtraction] = None
    revenue: Optional[OilRevenueExtraction] = None
    amnesty: Optional[AmnestyLegislationExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venezuela_updates() -> str:
    return """
    Extract the requested structured information exactly as presented in the answer. 
    For each subsection, also extract the reference URL(s) cited in the answer that specifically support that subsection. 
    IMPORTANT: Only include URLs explicitly present in the answer (plain URLs or markdown links). Do not invent or infer URLs.

    Return a JSON object with the following structure (use null if a field is missing; for URL arrays, use empty array if none found):

    {
      "operation": {
        "date": string | null,  // the date of the operation to capture Nicolás Maduro, as stated in the answer
        "location": string | null, // the location of the operation, as stated in the answer
        "sources": string[]      // URLs cited for the operation details
      },
      "interim_president": {
        "name": string | null,       // interim president's name
        "sworn_in_date": string | null, // the swearing-in date
        "sources": string[]           // URLs cited for interim president + swearing-in
      },
      "assembly": {
        "president_name": string | null, // President of the National Assembly
        "sources": string[]               // URLs cited for this info
      },
      "pre_expansion_company": {
        "company_name": string | null,   // the only US oil company operating before mid-February expansion
        "sources": string[]               // URLs cited for this info
      },
      "ofac": {
        "license_identifiers": string[], // all GL identifiers mentioned (e.g., ["GL 46", "GL 46A", "GL 47", "GL 48", "GL 49"])
        "time_window": string | null,    // textual description of issuance window/timeframe if present
        "authorized_companies": string[],// list of authorized companies mentioned
        "sources": string[]              // URLs cited for GL identifiers/time window and authorized companies
      },
      "revenue": {
        "total_amount": string | null,           // exact figure or phrasing used in the answer (e.g., "$1.2 billion")
        "initial_deposit_location": string | null, // where revenue was initially deposited (e.g., "Qatar account")
        "sources": string[]                      // URLs cited for revenue amount and deposit location
      },
      "amnesty": {
        "first_reading_date": string | null, // date when amnesty bill passed first reading
        "sources": string[]                  // URLs cited for amnesty bill date
      }
    }

    Special URL rules:
    - Extract only valid URLs explicitly present in the answer.
    - If a URL is missing protocol, prepend "http://".
    - Do not include duplicate URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_operation_details(evaluator: Evaluator, parent_node, data: Optional[OperationDetailsExtraction]) -> None:
    node = evaluator.add_parallel(
        id="operation_details",
        desc="Correctly identifies key details of the US military operation to capture Nicolás Maduro (date and location), with supporting reference URL(s)",
        parent=parent_node,
        critical=True
    )

    # Leaf: operation_date (simple check—does the answer state the expected date)
    n_date = evaluator.add_leaf(
        id="operation_date",
        desc="The date provided for the operation is January 3, 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the date of the operation to capture Nicolás Maduro is stated as January 3, 2026.",
        node=n_date,
        additional_instruction="Judge only based on the answer text. Allow minor date-format variations such as '3 January 2026' or 'Jan 3, 2026'."
    )

    # Leaf: operation_location (simple check—does the answer state the expected location)
    n_loc = evaluator.add_leaf(
        id="operation_location",
        desc="States that the operation took place in Caracas, Venezuela",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the operation location is stated to be Caracas, Venezuela (the capital).",
        node=n_loc,
        additional_instruction="Judge only based on the answer text. Accept 'Caracas' or 'Caracas, Venezuela' or equivalent phrasing clearly indicating the capital."
    )

    # Leaf: reference_url_operation (URL-backed)
    n_ref = evaluator.add_leaf(
        id="reference_url_operation",
        desc="Provides valid reference URL(s) that support BOTH the operation date and the operation location claims",
        parent=node,
        critical=True
    )
    op_sources = _safe_sources(data.sources if data else [])
    await evaluator.verify(
        claim="The source(s) explicitly state that the operation to capture Nicolás Maduro occurred on January 3, 2026 in Caracas, Venezuela.",
        node=n_ref,
        sources=op_sources,
        additional_instruction="Check the page(s) to confirm both the specific date (January 3, 2026) and the location (Caracas, Venezuela) are explicitly mentioned."
    )


async def verify_interim_president(evaluator: Evaluator, parent_node, data: Optional[InterimPresidentExtraction]) -> None:
    node = evaluator.add_parallel(
        id="interim_president_info",
        desc="Correctly identifies who became interim president and when they were sworn in, with supporting reference URL(s)",
        parent=parent_node,
        critical=True
    )

    # Leaf: interim_president_name (simple)
    n_name = evaluator.add_leaf(
        id="interim_president_name",
        desc="Identifies Delcy Rodríguez as the person who became interim president",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the interim president is identified as Delcy Rodríguez.",
        node=n_name,
        additional_instruction="Judge only based on the answer text. Allow accent-insensitive matching and minor spelling/casing variations (e.g., 'Delcy Rodriguez')."
    )

    # Leaf: swearing_in_date (simple)
    n_sworn = evaluator.add_leaf(
        id="swearing_in_date",
        desc="The swearing-in date provided is January 5, 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the swearing-in date for the interim president is stated as January 5, 2026.",
        node=n_sworn,
        additional_instruction="Judge only based on the answer text. Allow minor date-format variations such as '5 January 2026' or 'Jan 5, 2026'."
    )

    # Leaf: reference_url_interim (URL-backed)
    n_ref = evaluator.add_leaf(
        id="reference_url_interim",
        desc="Provides valid reference URL(s) that support BOTH the interim president identity and the swearing-in date",
        parent=node,
        critical=True
    )
    sources = _safe_sources(data.sources if data else [])
    await evaluator.verify(
        claim="The source(s) explicitly state that Delcy Rodríguez became interim president and was sworn in on January 5, 2026.",
        node=n_ref,
        sources=sources,
        additional_instruction="Confirm both (a) the interim president identity (Delcy Rodríguez) and (b) the January 5, 2026 swearing-in date are explicitly supported by the provided URLs."
    )


async def verify_assembly_president(evaluator: Evaluator, parent_node, data: Optional[AssemblyPresidentExtraction]) -> None:
    node = evaluator.add_parallel(
        id="national_assembly_president",
        desc="Correctly identifies the President of Venezuela's National Assembly, with a supporting reference URL",
        parent=parent_node,
        critical=True
    )

    # Leaf: assembly_president_name (simple)
    n_name = evaluator.add_leaf(
        id="assembly_president_name",
        desc="Identifies Jorge Rodríguez as the President of the National Assembly",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the President of Venezuela's National Assembly is identified as Jorge Rodríguez.",
        node=n_name,
        additional_instruction="Judge only based on the answer text. Allow accent-insensitive matching and minor variations (e.g., 'Jorge Rodriguez')."
    )

    # Leaf: reference_url_assembly (URL-backed)
    n_ref = evaluator.add_leaf(
        id="reference_url_assembly",
        desc="Provides a valid reference URL supporting the National Assembly president information",
        parent=node,
        critical=True
    )
    sources = _safe_sources(data.sources if data else [])
    await evaluator.verify(
        claim="The source(s) explicitly state that Jorge Rodríguez serves as President of Venezuela's National Assembly.",
        node=n_ref,
        sources=sources,
        additional_instruction="Confirm the provided URL(s) clearly indicate that Jorge Rodríguez is the National Assembly President."
    )


async def verify_pre_expansion_company(evaluator: Evaluator, parent_node, data: Optional[PreExpansionCompanyExtraction]) -> None:
    node = evaluator.add_parallel(
        id="us_oil_company_pre_expansion",
        desc="Correctly identifies which US oil company was the only one operating in Venezuela before the mid-February general licenses expansion, with a supporting reference URL",
        parent=parent_node,
        critical=True
    )

    # Leaf: sole_company_name (simple)
    n_name = evaluator.add_leaf(
        id="sole_company_name",
        desc="Identifies Chevron as the only US oil company operating in Venezuela before mid-February 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, Chevron is identified as the only U.S. oil company operating in Venezuela before the mid-February 2026 general license expansion.",
        node=n_name,
        additional_instruction="Judge only based on the answer text. The answer should clearly indicate Chevron as the sole U.S. operator before the mid-February expansion."
    )

    # Leaf: reference_url_chevron (URL-backed)
    n_ref = evaluator.add_leaf(
        id="reference_url_chevron",
        desc="Provides a valid reference URL supporting the sole US oil company claim",
        parent=node,
        critical=True
    )
    sources = _safe_sources(data.sources if data else [])
    await evaluator.verify(
        claim="Before the mid-February 2026 general license expansion, Chevron was the only U.S. oil company operating in Venezuela.",
        node=n_ref,
        sources=sources,
        additional_instruction="Confirm the provided URL(s) explicitly support Chevron being the only U.S. oil company operating in Venezuela prior to the mid-February 2026 expansion."
    )


async def verify_ofac_licenses(evaluator: Evaluator, parent_node, data: Optional[OFACLicensesExtraction]) -> None:
    node = evaluator.add_parallel(
        id="ofac_general_licenses_and_authorizations",
        desc="Correctly identifies the OFAC general-license expansion context (license identifiers/time window) and the five authorized major oil companies, with supporting reference URL(s)",
        parent=parent_node,
        critical=True
    )

    # Leaf: general_license_identifiers_and_window (simple)
    n_gl = evaluator.add_leaf(
        id="general_license_identifiers_and_window",
        desc="States that multiple OFAC general licenses were issued in late January through mid-February 2026, specifically GL 46, 46A, 47, 48, and 49",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the OFAC general licenses are listed as GL 46, GL 46A, GL 47, GL 48, and GL 49, with issuance described as in late January through mid-February 2026.",
        node=n_gl,
        additional_instruction="Judge only based on the answer text. Allow minor phrasing differences for the time window (e.g., 'late Jan to mid-Feb 2026')."
    )

    # Leaf: authorized_companies_list (simple)
    n_companies = evaluator.add_leaf(
        id="authorized_companies_list",
        desc="Lists the five major companies authorized to operate in Venezuela through OFAC general licenses: BP, Chevron, Eni, Repsol, and Shell",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the five authorized companies are listed as BP, Chevron, Eni, Repsol, and Shell.",
        node=n_companies,
        additional_instruction="Judge only based on the answer text. The list should include exactly these five companies (order does not matter)."
    )

    # Leaf: reference_url_licenses (URL-backed)
    n_ref = evaluator.add_leaf(
        id="reference_url_licenses",
        desc="Provides valid reference URL(s) that support BOTH (a) the general-license identifiers/time window AND (b) the authorized companies list",
        parent=node,
        critical=True
    )
    sources = _safe_sources(data.sources if data else [])
    await evaluator.verify(
        claim="OFAC issued general licenses GL 46, GL 46A, GL 47, GL 48, and GL 49 in late January through mid-February 2026, and these licenses authorized BP, Chevron, Eni, Repsol, and Shell to operate in Venezuela.",
        node=n_ref,
        sources=sources,
        additional_instruction="Confirm that the provided URL(s) explicitly mention the listed GL identifiers/time window AND the set of authorized companies (BP, Chevron, Eni, Repsol, Shell)."
    )


async def verify_oil_revenue_status(evaluator: Evaluator, parent_node, data: Optional[OilRevenueExtraction]) -> None:
    node = evaluator.add_parallel(
        id="oil_revenue_status",
        desc="Correctly describes the status of Venezuelan oil revenue by mid-February 2026 (amount and initial deposit location), with supporting reference URL(s)",
        parent=parent_node,
        critical=True
    )

    # Leaf: revenue_total_amount_provided (custom existence/format check)
    amount_text = (data.total_amount if data else None)
    has_numeric_amount = isinstance(amount_text, str) and any(ch.isdigit() for ch in amount_text)
    evaluator.add_custom_node(
        result=bool(has_numeric_amount),
        id="revenue_total_amount_provided",
        desc="Provides a numeric total/amount for Venezuelan oil revenue by mid-February 2026",
        parent=node,
        critical=True
    )

    # Leaf: revenue_amount_threshold (simple)
    n_threshold = evaluator.add_leaf(
        id="revenue_amount_threshold",
        desc="Indicates that oil revenue exceeded $1 billion by mid-February 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, it is stated that by mid-February 2026 Venezuelan oil revenue exceeded $1 billion (e.g., 'over $1 billion').",
        node=n_threshold,
        additional_instruction="Judge only based on the answer text. Accept equivalent phrasings like 'more than $1 billion', '$1B+', '$1.1 billion', etc."
    )

    # Leaf: initial_account_location (simple)
    n_deposit = evaluator.add_leaf(
        id="initial_account_location",
        desc="Mentions that revenue was initially deposited in a Qatar account",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the initial deposit location for the oil revenue is stated to be a Qatar account (i.e., a Qatari account/bank).",
        node=n_deposit,
        additional_instruction="Judge only based on the answer text. Accept phrasing such as 'in Qatar', 'Qatari account', or a specific Qatari bank as equivalent."
    )

    # Leaf: reference_url_revenue (URL-backed)
    n_ref = evaluator.add_leaf(
        id="reference_url_revenue",
        desc="Provides valid reference URL(s) supporting the oil revenue amount and the initial deposit location",
        parent=node,
        critical=True
    )
    sources = _safe_sources(data.sources if data else [])
    amount_for_claim = amount_text if (isinstance(amount_text, str) and amount_text.strip()) else "over $1 billion"
    await evaluator.verify(
        claim=f"By mid-February 2026, Venezuelan oil revenue totaled {amount_for_claim}, and the funds were initially deposited in a Qatar account.",
        node=n_ref,
        sources=sources,
        additional_instruction="Confirm that the provided URL(s) explicitly support BOTH the total revenue amount and that the funds were initially deposited in a Qatar account."
    )


async def verify_amnesty_legislation(evaluator: Evaluator, parent_node, data: Optional[AmnestyLegislationExtraction]) -> None:
    node = evaluator.add_parallel(
        id="amnesty_legislation",
        desc="Correctly identifies when the amnesty bill for political prisoners passed its first reading in the National Assembly, with a supporting reference URL",
        parent=parent_node,
        critical=True
    )

    # Leaf: first_reading_date (simple)
    n_date = evaluator.add_leaf(
        id="first_reading_date",
        desc="The first reading passage date provided is February 5, 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the amnesty bill for political prisoners is stated to have passed its first reading on February 5, 2026.",
        node=n_date,
        additional_instruction="Judge only based on the answer text. Allow minor date-format variations such as '5 February 2026' or 'Feb 5, 2026'."
    )

    # Leaf: reference_url_amnesty (URL-backed)
    n_ref = evaluator.add_leaf(
        id="reference_url_amnesty",
        desc="Provides a valid reference URL supporting the amnesty bill first-reading date",
        parent=node,
        critical=True
    )
    sources = _safe_sources(data.sources if data else [])
    await evaluator.verify(
        claim="The source(s) explicitly state that the amnesty bill for political prisoners passed its first reading on February 5, 2026.",
        node=n_ref,
        sources=sources,
        additional_instruction="Confirm the provided URL(s) explicitly mention the first reading date as February 5, 2026 (format variations OK)."
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
    Evaluate an answer for the Venezuela early-2026 developments task.
    """
    # Initialize evaluator (framework root is non-critical by design)
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

    # Add a top-level critical aggregator to enforce overall criticality
    all_checks = evaluator.add_parallel(
        id="all_criteria",
        desc="Evaluate whether the answer provides all required accurate developments in Venezuela following the January 2026 US military operation, with reference URL(s) supporting each required piece of information",
        parent=root,
        critical=True
    )

    # Extraction
    extracted: VenezuelaUpdateExtraction = await evaluator.extract(
        prompt=prompt_extract_venezuela_updates(),
        template_class=VenezuelaUpdateExtraction,
        extraction_name="venezuela_update_extraction"
    )

    # Optional: Add expected facts to summary for context
    evaluator.add_ground_truth({
        "expected_operation_date": EXPECTED_OPERATION_DATE,
        "expected_operation_location": EXPECTED_OPERATION_LOCATION,
        "expected_interim_president": EXPECTED_INTERIM_PRESIDENT,
        "expected_interim_sworn_in_date": EXPECTED_INTERIM_SWEARING_DATE,
        "expected_assembly_president": EXPECTED_ASSEMBLY_PRESIDENT,
        "expected_sole_us_company_pre_expansion": EXPECTED_SOLE_US_COMPANY,
        "expected_general_licenses": EXPECTED_GENERAL_LICENSES,
        "expected_authorized_companies": EXPECTED_AUTHORIZED_COMPANIES,
        "expected_amnesty_first_reading_date": EXPECTED_AMNESTY_FIRST_READING_DATE
    }, gt_type="expected_facts")

    # Build verification tree according to rubric
    await verify_operation_details(evaluator, all_checks, extracted.operation if extracted else None)
    await verify_interim_president(evaluator, all_checks, extracted.interim_president if extracted else None)
    await verify_assembly_president(evaluator, all_checks, extracted.assembly if extracted else None)
    await verify_pre_expansion_company(evaluator, all_checks, extracted.pre_expansion_company if extracted else None)
    await verify_ofac_licenses(evaluator, all_checks, extracted.ofac if extracted else None)
    await verify_oil_revenue_status(evaluator, all_checks, extracted.revenue if extracted else None)
    await verify_amnesty_legislation(evaluator, all_checks, extracted.amnesty if extracted else None)

    # Return aggregated summary
    return evaluator.get_summary()