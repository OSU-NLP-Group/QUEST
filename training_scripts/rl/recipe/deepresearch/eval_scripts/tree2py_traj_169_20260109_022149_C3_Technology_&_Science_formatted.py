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
TASK_ID = "willow_nature_dec2024_qec_below_threshold"
TASK_DESCRIPTION = (
    "Find the Nature journal paper published in December 2024 about Google's Willow quantum chip that demonstrates "
    "quantum error correction below the surface code threshold. For this paper: "
    "(1) Provide the paper's title, DOI, and a URL reference to the Nature publication. "
    "(2) Identify the corresponding author of the paper, including their full name and the email address listed in the paper. "
    "(3) Determine the corresponding author's current title at Google and identify the year when they founded the Google Quantum AI lab. Include a URL reference supporting this information. "
    "(4) Extract the technical specifications for both Willow chip configurations from the official spec sheet or paper: For Chip 1 (QEC-optimized), provide the mean T1 coherence time (with units and uncertainty) and the two-qubit gate error rate (with gate type and uncertainty). "
    "For Chip 2 (RCS-optimized), provide the mean T1 coherence time (with units and uncertainty) and the two-qubit gate error rate (with gate type and uncertainty). Include a URL reference to the source of these specifications. "
    "(5) Extract the reported Lambda (Λ) value (with uncertainty) that measures the quantum error correction performance across code distances 3, 5, and 7, along with a URL reference to the source."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperMetadata(BaseModel):
    title: Optional[str] = None
    doi: Optional[str] = None
    nature_url: Optional[str] = None
    publication_date: Optional[str] = None  # Keep as string to allow flexibility (e.g., "December 2024")


class CorrespondingAuthorInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    info_url: Optional[str] = None  # Prefer Nature page or Nature PDF


class AuthorRoleFounding(BaseModel):
    current_title: Optional[str] = None
    founding_year: Optional[str] = None  # Keep as string to allow flexible formats
    url: Optional[str] = None  # Source URL supporting both title and founding year


class ChipSpec(BaseModel):
    mean_t1: Optional[str] = None  # Include units and uncertainty in a single string
    two_qubit_gate_error: Optional[str] = None  # Include gate type and uncertainty in a single string


class WillowSpecs(BaseModel):
    chip1: Optional[ChipSpec] = None  # QEC-optimized configuration
    chip2: Optional[ChipSpec] = None  # RCS-optimized configuration
    specs_source_url: Optional[str] = None  # Must be Nature paper page/PDF or official Willow spec sheet PDF


class LambdaMetric(BaseModel):
    lambda_value: Optional[str] = None  # Lambda (Λ) value with uncertainty across code distances 3, 5, 7
    source_url: Optional[str] = None  # Must be Nature paper page/PDF or official Willow spec sheet PDF


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_metadata() -> str:
    return (
        "Extract the core bibliographic metadata for the Nature (December 2024) paper about Google's Willow quantum chip "
        "that demonstrates quantum error correction below the surface code threshold. Return a JSON object with:\n"
        "• title: The exact title of the paper as stated on the Nature publication page or PDF.\n"
        "• doi: The DOI string as shown on Nature (e.g., '10.1038/xxxx').\n"
        "• nature_url: A single URL to the Nature publication page or Nature PDF for the paper.\n"
        "• publication_date: The publication date string as shown on Nature (e.g., 'December 2024' or '11 December 2024').\n"
        "If any field is missing in the answer, set it to null. Do not invent information. Extract exactly what the answer provided."
    )


def prompt_extract_corresponding_author_info() -> str:
    return (
        "Identify the corresponding author information for the target paper from the answer. Return a JSON object with:\n"
        "• name: Full name of the corresponding author as listed in the paper.\n"
        "• email: The email address provided for correspondence (e.g., 'name@google.com'), exactly as listed.\n"
        "• info_url: A single URL (prefer the Nature page or Nature PDF) that lists/indicates the corresponding author and their email.\n"
        "If not present, return null for missing fields. Do not infer or invent."
    )


def prompt_extract_author_role_and_lab_founding() -> str:
    return (
        "For the corresponding author identified in the answer, extract their current title at Google and the year they founded the Google Quantum AI lab. "
        "Return a JSON object with:\n"
        "• current_title: The current title/position at Google (exact phrasing, e.g., 'Director of Quantum AI').\n"
        "• founding_year: The year they founded the Google Quantum AI lab (e.g., '2013').\n"
        "• url: A single URL that supports both the current title and the founding year (e.g., an official Google profile, press release, or credible bio page).\n"
        "If any field is missing, return null for that field."
    )


def prompt_extract_willow_specs() -> str:
    return (
        "Extract the Willow chip technical specifications from the official Willow spec sheet or the Nature paper. "
        "Return a JSON object with:\n"
        "• chip1.mean_t1: The mean T1 coherence time (include units and uncertainty), for the QEC-optimized Willow chip.\n"
        "• chip1.two_qubit_gate_error: The two-qubit gate error rate (include gate type and uncertainty) for Chip 1.\n"
        "• chip2.mean_t1: The mean T1 coherence time (include units and uncertainty), for the RCS-optimized Willow chip.\n"
        "• chip2.two_qubit_gate_error: The two-qubit gate error rate (include gate type and uncertainty) for Chip 2.\n"
        "• specs_source_url: A single URL that is either the Nature paper page/PDF or the official Willow spec sheet PDF.\n"
        "If the answer provides multiple values, choose the most prominent single value for each required field. Use flexible strings to capture units and uncertainties. "
        "If any field is missing, set it to null."
    )


def prompt_extract_lambda_metric() -> str:
    return (
        "Extract the reported Lambda (Λ) value with uncertainty characterizing the quantum error correction performance across code distances 3, 5, and 7. "
        "Return a JSON object with:\n"
        "• lambda_value: A single string summarizing the reported Λ value with uncertainty (e.g., 'Λ = 0.73 ± 0.05 across d=3,5,7').\n"
        "• source_url: A single URL that is either the Nature paper page/PDF or the official Willow spec sheet PDF.\n"
        "If not present in the answer, set missing fields to null."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_paper_identification(
    evaluator: Evaluator,
    parent_node,
    paper_meta: PaperMetadata,
) -> Dict[str, Any]:
    """
    Build and verify the 'Paper_Identification' subtree:
    - Paper_Core_Metadata (title, DOI, Nature URL)
    - Paper_Eligibility (journal=Nature, publication date=Dec 2024, topic=Willow chip, QEC below threshold)
    Returns dict containing handles (e.g., the Nature URL leaf node) for downstream preconditions.
    """
    paper_id_node = evaluator.add_parallel(
        id="Paper_Identification",
        desc="Correctly identify the target Nature (Dec 2024) Willow-chip paper and provide its core metadata.",
        parent=parent_node,
        critical=True,
    )

    # Subnode: Paper_Core_Metadata
    core_meta_node = evaluator.add_parallel(
        id="Paper_Core_Metadata",
        desc="Provide core bibliographic identifiers for the paper.",
        parent=paper_id_node,
        critical=True,
    )

    # Leaf: Nature_Paper_URL
    nature_url_leaf = evaluator.add_leaf(
        id="Nature_Paper_URL",
        desc="Provide a URL reference to the Nature publication page for the paper.",
        parent=core_meta_node,
        critical=True,
    )
    claim_url_is_nature = (
        "This URL is the official Nature publication page or Nature-hosted PDF for the paper about Google's Willow quantum chip "
        "that demonstrates quantum error correction below the surface code threshold and was published in December 2024."
    )
    await evaluator.verify(
        claim=claim_url_is_nature,
        node=nature_url_leaf,
        sources=paper_meta.nature_url,
        additional_instruction=(
            "Verify that the page belongs to Nature (nature.com) or is a Nature-hosted PDF and corresponds to the Willow chip paper. "
            "If the URL is missing, invalid, or not Nature, mark as not supported."
        ),
    )

    # Leaf: Paper_Title
    title_leaf = evaluator.add_leaf(
        id="Paper_Title",
        desc="Provide the paper's title.",
        parent=core_meta_node,
        critical=True,
    )
    claim_title = f"The paper's title is '{paper_meta.title}'."
    await evaluator.verify(
        claim=claim_title,
        node=title_leaf,
        sources=paper_meta.nature_url,
        additional_instruction=(
            "Check the title shown on the Nature publication page or PDF. Allow minor formatting variations (e.g., punctuation or capitalization)."
        ),
        extra_prerequisites=[nature_url_leaf],
    )

    # Leaf: Paper_DOI
    doi_leaf = evaluator.add_leaf(
        id="Paper_DOI",
        desc="Provide the paper's DOI.",
        parent=core_meta_node,
        critical=True,
    )
    claim_doi = f"The DOI of the paper is '{paper_meta.doi}'."
    await evaluator.verify(
        claim=claim_doi,
        node=doi_leaf,
        sources=paper_meta.nature_url,
        additional_instruction=(
            "Verify the DOI exactly as shown on the Nature publication page or PDF. Minor formatting (prefix 'https://doi.org/') can be considered equivalent."
        ),
        extra_prerequisites=[nature_url_leaf],
    )

    # Subnode: Paper_Eligibility
    eligibility_node = evaluator.add_parallel(
        id="Paper_Eligibility",
        desc="Paper matches all required identification constraints.",
        parent=paper_id_node,
        critical=True,
    )

    # Leaf: Journal_Nature
    journal_leaf = evaluator.add_leaf(
        id="Journal_Nature",
        desc="Paper is published in the Nature journal.",
        parent=eligibility_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This publication is in Nature journal.",
        node=journal_leaf,
        sources=paper_meta.nature_url,
        additional_instruction="Confirm the journal branding and metadata indicate 'Nature'.",
        extra_prerequisites=[nature_url_leaf],
    )

    # Leaf: Publication_Date_Dec_2024
    pubdate_leaf = evaluator.add_leaf(
        id="Publication_Date_Dec_2024",
        desc="Paper publication date is in December 2024.",
        parent=eligibility_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The paper's publication date is in December 2024 (month 12 of 2024).",
        node=pubdate_leaf,
        sources=paper_meta.nature_url,
        additional_instruction="Check the publication date field on the Nature page or PDF. Accept formats like '11 December 2024' or 'December 2024'.",
        extra_prerequisites=[nature_url_leaf],
    )

    # Leaf: Topic_Willow_Chip
    topic_leaf = evaluator.add_leaf(
        id="Topic_Willow_Chip",
        desc="Paper is about Google's Willow quantum chip.",
        parent=eligibility_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This paper is about Google's Willow quantum chip.",
        node=topic_leaf,
        sources=paper_meta.nature_url,
        additional_instruction="Confirm the abstract/title/introduction explicitly mention 'Willow' as the quantum processor/chip.",
        extra_prerequisites=[nature_url_leaf],
    )

    # Leaf: QEC_Below_Surface_Code_Threshold
    qec_leaf = evaluator.add_leaf(
        id="QEC_Below_Surface_Code_Threshold",
        desc="Paper demonstrates quantum error correction below the surface code threshold.",
        parent=eligibility_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This paper demonstrates quantum error correction below the surface code threshold.",
        node=qec_leaf,
        sources=paper_meta.nature_url,
        additional_instruction="Look for explicit claims or results indicating QEC performance below the surface code threshold.",
        extra_prerequisites=[nature_url_leaf],
    )

    return {
        "nature_url_leaf": nature_url_leaf,
        "title_leaf": title_leaf,
        "doi_leaf": doi_leaf,
        "journal_leaf": journal_leaf,
        "pubdate_leaf": pubdate_leaf,
        "topic_leaf": topic_leaf,
        "qec_leaf": qec_leaf,
    }


async def build_required_extractions(
    evaluator: Evaluator,
    parent_node,
    author_info: CorrespondingAuthorInfo,
    role_info: AuthorRoleFounding,
    specs: WillowSpecs,
    lambda_metric: LambdaMetric,
) -> None:
    """
    Build and verify the 'Required_Extractions' subtree:
    - Corresponding_Author (name, email, info URL)
    - Author_Role_And_Lab_Founding (current title at Google, founding year, URL)
    - Willow_Technical_Specifications (Chip1/Chip2 values + allowed source URL)
    - Lambda_Performance_Metric (value + allowed source URL)
    """
    req_node = evaluator.add_parallel(
        id="Required_Extractions",
        desc="Extract corresponding author info, author role/founding year, chip specs, and Lambda metric with proper references.",
        parent=parent_node,
        critical=True,
    )

    # 1) Corresponding Author
    ca_node = evaluator.add_parallel(
        id="Corresponding_Author",
        desc="Identify the corresponding author and provide contact info as listed in the paper.",
        parent=req_node,
        critical=True,
    )

    # Leaf: Author_Info_URL
    ca_url_leaf = evaluator.add_leaf(
        id="Author_Info_URL",
        desc="Provide a URL reference supporting the corresponding author identification (e.g., the Nature paper page/PDF).",
        parent=ca_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This URL is the Nature paper page or a Nature-hosted PDF that lists the corresponding author and the correspondence email.",
        node=ca_url_leaf,
        sources=author_info.info_url,
        additional_instruction=(
            "Verify that the page shows 'Corresponding author' or 'correspondence to' and includes the email. "
            "If the URL is missing or not Nature, mark as not supported."
        ),
    )

    # Leaf: Author_Name
    ca_name_leaf = evaluator.add_leaf(
        id="Author_Name",
        desc="Provide the full name of the corresponding author.",
        parent=ca_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The corresponding author is '{author_info.name}'.",
        node=ca_name_leaf,
        sources=author_info.info_url,
        additional_instruction="Look for explicit 'Corresponding author' designation or notes. Allow minor name formatting variants.",
        extra_prerequisites=[ca_url_leaf],
    )

    # Leaf: Author_Email
    ca_email_leaf = evaluator.add_leaf(
        id="Author_Email",
        desc="Provide the corresponding author's email address as listed in the paper.",
        parent=ca_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The corresponding author's email address is '{author_info.email}'.",
        node=ca_email_leaf,
        sources=author_info.info_url,
        additional_instruction="Verify the email in the 'correspondence' section or footer of the Nature publication/PDF.",
        extra_prerequisites=[ca_url_leaf],
    )

    # 2) Author Role & Lab Founding
    role_node = evaluator.add_parallel(
        id="Author_Role_And_Lab_Founding",
        desc="Provide the corresponding author's current title at Google and the year they founded the Google Quantum AI lab, with a supporting URL.",
        parent=req_node,
        critical=True,
    )

    # Leaf: Role_And_Founding_URL
    role_url_leaf = evaluator.add_leaf(
        id="Role_And_Founding_URL",
        desc="Provide a URL reference supporting both the current title and the lab founding year.",
        parent=role_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"This URL includes both the author's current title at Google ('{role_info.current_title}') "
            f"and states they founded the Google Quantum AI lab in {role_info.founding_year}."
        ),
        node=role_url_leaf,
        sources=role_info.url,
        additional_instruction=(
            "Verify both pieces of information are present on the same page. Prefer official Google pages or credible biographies."
        ),
    )

    # Leaf: Current_Title
    current_title_leaf = evaluator.add_leaf(
        id="Current_Title",
        desc="Provide the corresponding author's current title/position at Google.",
        parent=role_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The corresponding author's current title at Google is '{role_info.current_title}'.",
        node=current_title_leaf,
        sources=role_info.url,
        additional_instruction="Allow reasonable wording variants for titles (e.g., 'Director of Quantum AI' vs 'Head of Quantum AI').",
        extra_prerequisites=[role_url_leaf],
    )

    # Leaf: Lab_Founding_Year
    founding_year_leaf = evaluator.add_leaf(
        id="Lab_Founding_Year",
        desc="Provide the year the corresponding author founded the Google Quantum AI lab.",
        parent=role_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"They founded the Google Quantum AI lab in {role_info.founding_year}.",
        node=founding_year_leaf,
        sources=role_info.url,
        additional_instruction="Confirm the founding year is explicitly stated; accept minor contextual phrasing indicating the year.",
        extra_prerequisites=[role_url_leaf],
    )

    # 3) Willow Technical Specifications
    specs_node = evaluator.add_parallel(
        id="Willow_Technical_Specifications",
        desc="Extract technical specs for both Willow chip configurations, including required fields and uncertainties, from allowed official sources.",
        parent=req_node,
        critical=True,
    )

    # Leaf: Specs_Source_URL_Allowed
    specs_url_leaf = evaluator.add_leaf(
        id="Specs_Source_URL_Allowed",
        desc="Provide a URL reference for the specs, and it must be either the official Willow spec sheet PDF or the Nature paper (as required by constraints).",
        parent=specs_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This URL is either the Nature publication page/PDF for the Willow chip paper or an official Willow spec sheet PDF.",
        node=specs_url_leaf,
        sources=specs.specs_source_url,
        additional_instruction=(
            "Check that the page is Nature (nature.com) or an official Willow spec sheet (PDF) published by Google Quantum AI or equivalent official source. "
            "If not, mark as not supported."
        ),
    )

    # Chip1 (QEC-optimized)
    chip1_node = evaluator.add_parallel(
        id="Chip1_QEC_Optimized",
        desc="Provide required Chip 1 (QEC-optimized) specifications.",
        parent=specs_node,
        critical=True,
    )

    chip1_t1_leaf = evaluator.add_leaf(
        id="Chip1_Mean_T1",
        desc="Provide Chip 1 mean T1 coherence time with units and uncertainty.",
        parent=chip1_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Chip 1 (QEC-optimized) mean T1 coherence time is '{specs.chip1.mean_t1 if specs.chip1 else None}'.",
        node=chip1_t1_leaf,
        sources=specs.specs_source_url,
        additional_instruction=(
            "From the allowed source, verify the Chip 1 mean T1 value including units (e.g., μs) and uncertainty (e.g., ± value). "
            "Minor formatting differences are acceptable."
        ),
        extra_prerequisites=[specs_url_leaf],
    )

    chip1_gate_leaf = evaluator.add_leaf(
        id="Chip1_TwoQubit_Gate_Error",
        desc="Provide Chip 1 two-qubit gate error rate including gate type and uncertainty.",
        parent=chip1_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Chip 1 (QEC-optimized) two-qubit gate error rate is '{specs.chip1.two_qubit_gate_error if specs.chip1 else None}'.",
        node=chip1_gate_leaf,
        sources=specs.specs_source_url,
        additional_instruction=(
            "Verify the two-qubit gate error rate and gate type (e.g., CZ, iSWAP) for Chip 1, including uncertainty. "
            "Minor formatting differences are acceptable."
        ),
        extra_prerequisites=[specs_url_leaf],
    )

    # Chip2 (RCS-optimized)
    chip2_node = evaluator.add_parallel(
        id="Chip2_RCS_Optimized",
        desc="Provide required Chip 2 (RCS-optimized) specifications.",
        parent=specs_node,
        critical=True,
    )

    chip2_t1_leaf = evaluator.add_leaf(
        id="Chip2_Mean_T1",
        desc="Provide Chip 2 mean T1 coherence time with units and uncertainty.",
        parent=chip2_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Chip 2 (RCS-optimized) mean T1 coherence time is '{specs.chip2.mean_t1 if specs.chip2 else None}'.",
        node=chip2_t1_leaf,
        sources=specs.specs_source_url,
        additional_instruction=(
            "From the allowed source, verify the Chip 2 mean T1 value including units (e.g., μs) and uncertainty (e.g., ± value). "
            "Minor formatting differences are acceptable."
        ),
        extra_prerequisites=[specs_url_leaf],
    )

    chip2_gate_leaf = evaluator.add_leaf(
        id="Chip2_TwoQubit_Gate_Error",
        desc="Provide Chip 2 two-qubit gate error rate including gate type and uncertainty.",
        parent=chip2_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Chip 2 (RCS-optimized) two-qubit gate error rate is '{specs.chip2.two_qubit_gate_error if specs.chip2 else None}'.",
        node=chip2_gate_leaf,
        sources=specs.specs_source_url,
        additional_instruction=(
            "Verify the two-qubit gate error rate and gate type (e.g., CZ, iSWAP) for Chip 2, including uncertainty. "
            "Minor formatting differences are acceptable."
        ),
        extra_prerequisites=[specs_url_leaf],
    )

    # 4) Lambda Performance Metric
    lambda_node = evaluator.add_parallel(
        id="Lambda_Performance_Metric",
        desc="Extract the reported Lambda (Λ) value (with uncertainty) for performance across code distances 3, 5, and 7, with an allowed-source URL.",
        parent=req_node,
        critical=True,
    )

    lambda_url_leaf = evaluator.add_leaf(
        id="Lambda_Source_URL_Allowed",
        desc="Provide a URL reference for the Lambda value, and it must be either the Nature paper or the official Willow spec sheet PDF (as required by constraints).",
        parent=lambda_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This URL is either the Nature publication page/PDF for the Willow chip paper or an official Willow spec sheet PDF.",
        node=lambda_url_leaf,
        sources=lambda_metric.source_url,
        additional_instruction=(
            "Check that the page is Nature (nature.com) or an official Willow spec sheet (PDF) published by Google Quantum AI or equivalent official source."
        ),
    )

    lambda_value_leaf = evaluator.add_leaf(
        id="Lambda_Value",
        desc="Provide the Lambda (Λ) value with uncertainty as reported for distances 3, 5, and 7.",
        parent=lambda_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The reported Lambda (Λ) performance metric across code distances 3, 5, and 7 is '{lambda_metric.lambda_value}'.",
        node=lambda_value_leaf,
        sources=lambda_metric.source_url,
        additional_instruction=(
            "Verify the Λ value and uncertainty from the text or figures. Accept equivalent formatting (e.g., parentheses vs ±)."
        ),
        extra_prerequisites=[lambda_url_leaf],
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
    Evaluate the answer for the Willow Nature December 2024 paper task.
    """
    # Initialize evaluator (framework root is non-critical; we create a critical top-level node under it)
    evaluator = Evaluator()
    framework_root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # We'll add our critical sequential 'Root' under this
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

    # Create task root as a critical sequential node (matching rubric Root)
    task_root = evaluator.add_sequential(
        id="Root",
        desc="Identify the specific December 2024 Nature paper about Google's Willow quantum chip and extract required metadata, author details, technical specifications, and Lambda metric with proper sourcing.",
        parent=framework_root,
        critical=True,
    )

    # Perform extractions (can be parallelized)
    paper_meta_task = evaluator.extract(
        prompt=prompt_extract_paper_metadata(),
        template_class=PaperMetadata,
        extraction_name="paper_metadata",
    )
    author_info_task = evaluator.extract(
        prompt=prompt_extract_corresponding_author_info(),
        template_class=CorrespondingAuthorInfo,
        extraction_name="corresponding_author_info",
    )
    role_info_task = evaluator.extract(
        prompt=prompt_extract_author_role_and_lab_founding(),
        template_class=AuthorRoleFounding,
        extraction_name="author_role_founding",
    )
    specs_task = evaluator.extract(
        prompt=prompt_extract_willow_specs(),
        template_class=WillowSpecs,
        extraction_name="willow_specs",
    )
    lambda_task = evaluator.extract(
        prompt=prompt_extract_lambda_metric(),
        template_class=LambdaMetric,
        extraction_name="lambda_metric",
    )

    paper_meta, author_info, role_info, specs, lambda_metric = await asyncio.gather(
        paper_meta_task, author_info_task, role_info_task, specs_task, lambda_task
    )

    # Build Paper Identification subtree first (so subsequent checks can depend on Nature URL leaf)
    nodes_ctx = await build_paper_identification(evaluator, task_root, paper_meta)

    # Build Required Extractions subtree
    await build_required_extractions(evaluator, task_root, author_info, role_info, specs, lambda_metric)

    # Return structured evaluation summary
    return evaluator.get_summary()