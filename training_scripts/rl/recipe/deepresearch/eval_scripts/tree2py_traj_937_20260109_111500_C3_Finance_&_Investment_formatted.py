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
TASK_ID = "franklin_xrp_etf_lmax_regulation_2025"
TASK_DESCRIPTION = (
    "Franklin Templeton launched an XRP ETF in 2025. Identify the regulatory body that oversees LMAX Digital, "
    "one of the constituent exchanges used in this ETF's benchmark index, and specify the type of regulatory license "
    "under which LMAX Digital operates."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFRegulationExtraction(BaseModel):
    # ETF identification
    etf_name: Optional[str] = None
    issuer: Optional[str] = None
    asset_exposure: Optional[str] = None
    launch_year: Optional[str] = None

    # Benchmark index details
    benchmark_index_name: Optional[str] = None
    constituent_exchanges: List[str] = Field(default_factory=list)

    # LMAX regulation details
    lmax_regulatory_body: Optional[str] = None
    lmax_license_type: Optional[str] = None

    # Sources present in the answer (as URLs only)
    etf_sources: List[str] = Field(default_factory=list)
    benchmark_index_sources: List[str] = Field(default_factory=list)
    lmax_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etf_and_lmax_details() -> str:
    return """
    From the answer text, extract the following fields related to the Franklin Templeton XRP ETF launched in 2025 and LMAX Digital:

    1) etf_name: the ETF's name as stated in the answer (e.g., "Franklin Templeton XRP ETF" or similar).
    2) issuer: the issuer/sponsor/registrant name (e.g., "Franklin Templeton"). Use exactly what appears in the answer.
    3) asset_exposure: which cryptoasset the ETF provides exposure to (e.g., "XRP").
    4) launch_year: the year the ETF launched (as stated in the answer). Provide just the 4-digit year string if available.
    5) benchmark_index_name: the name of the benchmark index used for NAV/valuation.
    6) constituent_exchanges: a list of the constituent exchanges explicitly mentioned for that index (e.g., "LMAX Digital", "Coinbase", etc.)
    7) lmax_regulatory_body: the name of the regulatory body that oversees/regulates LMAX Digital (as stated in the answer).
    8) lmax_license_type: the type of regulatory license/oversight under which LMAX Digital operates (as stated in the answer).

    Also extract URL sources that the answer cites for each area:
    9) etf_sources: URLs cited that support the ETF details (issuer/asset/launch year/index). Include only actual URLs shown in the answer.
    10) benchmark_index_sources: URLs that describe the benchmark index and its constituent exchanges/methodology.
    11) lmax_sources: URLs that support LMAX Digital's regulation and license type.

    IMPORTANT:
    - Only extract information explicitly present in the answer text.
    - For URLs, extract actual URLs (including those inside markdown links). Do not invent any.
    - If a field is not mentioned, set it to null. If a list field has no entries, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _union_sources(*source_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in source_lists:
        if not lst:
            continue
        for url in lst:
            if isinstance(url, str):
                u = url.strip()
                if u and u not in seen:
                    seen.add(u)
                    merged.append(u)
    return merged


def _nz(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, info: ETFRegulationExtraction) -> None:
    """
    Build the verification tree according to the rubric and perform verifications using the evaluator.
    The root is sequential and critical; all child groups and leaves are critical per rubric.
    """
    root = evaluator.root  # Root already initialized as non-critical in Evaluator.initialize; we will create a critical sequential node beneath to reflect rubric.

    # Create a critical, sequential top-level node to mirror the rubric's Research_Task
    research_task = evaluator.add_sequential(
        id="Research_Task",
        desc="Determine the regulator overseeing LMAX Digital (as a constituent exchange in the ETF's benchmark index) and the regulatory license/oversight type under which LMAX Digital operates, consistent with all stated constraints.",
        parent=root,
        critical=True
    )

    # 1) ETF_Identification_and_Eligibility (parallel, critical)
    etf_id_node = evaluator.add_parallel(
        id="ETF_Identification_and_Eligibility",
        desc="Identify the relevant Franklin Templeton XRP ETF and confirm it matches the stated constraints.",
        parent=research_task,
        critical=True
    )

    # 1.a) Issuer_Is_Franklin_Templeton
    issuer_leaf = evaluator.add_leaf(
        id="Issuer_Is_Franklin_Templeton",
        desc="Answer identifies an ETF issued by Franklin Templeton.",
        parent=etf_id_node,
        critical=True
    )
    issuer_claim_etf_named = f"The ETF named '{_nz(info.etf_name)}' is issued/sponsored by Franklin Templeton or a Franklin Templeton entity."
    issuer_claim_generic = "The ETF in question is issued/sponsored by Franklin Templeton or a Franklin Templeton entity."
    await evaluator.verify(
        claim=issuer_claim_etf_named if info.etf_name else issuer_claim_generic,
        node=issuer_leaf,
        sources=_union_sources(info.etf_sources),
        additional_instruction="Treat 'issuer', 'sponsor', 'trust sponsor', or 'registrant' as equivalent for ETF issuance. Accept reasonable variants like 'Franklin Templeton', 'Franklin Templeton Digital Assets', or closely related Franklin Templeton entities."
    )

    # 1.b) Provides_XRP_Exposure
    provides_xrp_leaf = evaluator.add_leaf(
        id="Provides_XRP_Exposure",
        desc="Answer confirms the ETF provides exposure to XRP (not another cryptoasset).",
        parent=etf_id_node,
        critical=True
    )
    exposure_claim = "The ETF provides exposure to XRP (Ripple's token), not a different cryptoasset."
    await evaluator.verify(
        claim=exposure_claim,
        node=provides_xrp_leaf,
        sources=_union_sources(info.etf_sources),
        additional_instruction="Check fund objective, holdings description, ticker references, or prospectus wording to confirm exposure to XRP specifically."
    )

    # 1.c) Launched_In_2025
    launched_leaf = evaluator.add_leaf(
        id="Launched_In_2025",
        desc="Answer confirms the ETF was launched in 2025.",
        parent=etf_id_node,
        critical=True
    )
    launched_claim = "The ETF launched/commenced trading/inception/effective date is in 2025."
    await evaluator.verify(
        claim=launched_claim,
        node=launched_leaf,
        sources=_union_sources(info.etf_sources),
        additional_instruction="Accept synonyms like 'launched', 'inception', 'first trading day', 'effective date', 'listing date' as evidence of 2025 launch."
    )

    # 2) Benchmark_Index_For_NAV (parallel, critical)
    index_node = evaluator.add_parallel(
        id="Benchmark_Index_For_NAV",
        desc="Identify the benchmark index used for NAV calculation and confirm it aggregates data from constituent exchanges.",
        parent=research_task,
        critical=True
    )

    # 2.a) Benchmark_Index_Identified
    idx_ident_leaf = evaluator.add_leaf(
        id="Benchmark_Index_Identified",
        desc="Answer identifies the benchmark index the ETF uses for NAV calculation.",
        parent=index_node,
        critical=True
    )
    idx_claim = f"The ETF uses the benchmark/index '{_nz(info.benchmark_index_name)}' for NAV or valuation." if info.benchmark_index_name else "The ETF uses a specific benchmark index for NAV or valuation."
    await evaluator.verify(
        claim=idx_claim,
        node=idx_ident_leaf,
        sources=_union_sources(info.etf_sources, info.benchmark_index_sources),
        additional_instruction="Look for explicit statements in prospectus, fact sheet, or methodology pages that the ETF's NAV/valuation references this index."
    )

    # 2.b) Index_Aggregates_Constituent_Exchanges
    idx_agg_leaf = evaluator.add_leaf(
        id="Index_Aggregates_Constituent_Exchanges",
        desc="Answer indicates the benchmark index aggregates data from constituent exchanges (not a single venue).",
        parent=index_node,
        critical=True
    )
    idx_agg_claim = f"The index '{_nz(info.benchmark_index_name)}' aggregates price data from multiple constituent exchanges (more than one venue)." if info.benchmark_index_name else "The index aggregates price data from multiple constituent exchanges (more than one venue)."
    await evaluator.verify(
        claim=idx_agg_claim,
        node=idx_agg_leaf,
        sources=_union_sources(info.benchmark_index_sources),
        additional_instruction="Check methodology or index description pages for wording like 'constituent exchanges', 'multi-venue', 'aggregated from several exchanges', or a listed set of exchanges."
    )

    # 3) LMAX_Is_Constituent_Exchange (parallel, critical)
    lmax_const_node = evaluator.add_parallel(
        id="LMAX_Is_Constituent_Exchange",
        desc="Confirm LMAX Digital is one of the benchmark index's constituent exchanges.",
        parent=research_task,
        critical=True
    )

    lmax_const_leaf = evaluator.add_leaf(
        id="LMAX_Digital_Listed_As_Constituent",
        desc="Answer states that LMAX Digital is a constituent exchange of the identified benchmark index.",
        parent=lmax_const_node,
        critical=True
    )
    lmax_const_claim = f"LMAX Digital is listed as one of the constituent exchanges used by the index '{_nz(info.benchmark_index_name)}'." if info.benchmark_index_name else "LMAX Digital is listed as one of the constituent exchanges used by the benchmark index."
    await evaluator.verify(
        claim=lmax_const_claim,
        node=lmax_const_leaf,
        sources=_union_sources(info.benchmark_index_sources),
        additional_instruction="Verify on official index pages or methodology documents. Accept reasonable naming variants like 'LMAX' or 'LMAX Digital'."
    )

    # 4) LMAX_Regulation_Details (parallel, critical)
    lmax_reg_node = evaluator.add_parallel(
        id="LMAX_Regulation_Details",
        desc="Provide LMAX Digital's regulator and the type of regulatory license/oversight under which it operates.",
        parent=research_task,
        critical=True
    )

    # 4.a) Regulatory_Body_Identified
    lmax_reg_body_leaf = evaluator.add_leaf(
        id="Regulatory_Body_Identified",
        desc="Answer provides the name of the regulatory body that oversees/regulates LMAX Digital.",
        parent=lmax_reg_node,
        critical=True
    )
    reg_body_claim = f"LMAX Digital is overseen/regul­ated by '{_nz(info.lmax_regulatory_body)}'." if info.lmax_regulatory_body else "LMAX Digital is overseen/regul­ated by a named regulatory authority."
    await evaluator.verify(
        claim=reg_body_claim,
        node=lmax_reg_body_leaf,
        sources=_union_sources(info.lmax_sources),
        additional_instruction="Prefer official regulatory pages or LMAX's official compliance/regulatory disclosures. Accept reasonable naming variants and acronyms (e.g., FCA for Financial Conduct Authority)."
    )

    # 4.b) License_or_Oversight_Type_Specified
    lmax_license_leaf = evaluator.add_leaf(
        id="License_or_Oversight_Type_Specified",
        desc="Answer specifies the type of regulatory license or oversight framework under which LMAX Digital operates.",
        parent=lmax_reg_node,
        critical=True
    )
    license_claim = f"LMAX Digital operates under this regulatory license/oversight type: '{_nz(info.lmax_license_type)}'." if info.lmax_license_type else "LMAX Digital operates under a specific regulatory license/oversight type."
    await evaluator.verify(
        claim=license_claim,
        node=lmax_license_leaf,
        sources=_union_sources(info.lmax_sources),
        additional_instruction="Verify the precise oversight type, such as 'investment firm authorization', 'MTF/MTF operator', 'DLT provider', 'VASP registration', 'money services business', etc., as stated on regulatory or official LMAX pages."
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
    Evaluate an answer for the Franklin Templeton XRP ETF (2025) and LMAX Digital regulation task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root combines as sequential, but we'll add a critical sequential node mirroring rubric
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

    # Extract structured info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_etf_and_lmax_details(),
        template_class=ETFRegulationExtraction,
        extraction_name="extracted_etf_lmax_info",
    )

    # Build tree and run verifications according to rubric
    await build_and_verify_tree(evaluator, extracted_info)

    # Return comprehensive summary including the verification tree and scores
    return evaluator.get_summary()