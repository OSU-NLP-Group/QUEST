import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "warsh_education_2026"
TASK_DESCRIPTION = (
    "Following President Trump's nomination of Kevin Warsh to serve as the next Federal Reserve Chair in January 2026, "
    "identify his educational background by providing: (1) His undergraduate degree (type of degree, institution, "
    "and year of graduation), and (2) His graduate degree (type of degree, institution, and year of graduation). "
    "Provide verifiable information with reference URLs supporting each credential."
)

# Expected ground-truth for comparison checks (string equivalence only; not used as web facts)
EXPECTED_UNDERGRAD = {
    "degree_type": "Bachelor of Arts (B.A.)",
    "field_of_study": "Public Policy",
    "institution": "Stanford University",
    "graduation_year": "1992",
}
EXPECTED_GRAD = {
    "degree_type": "Juris Doctor (J.D.)",
    "institution": "Harvard Law School",
    "graduation_year": "1995",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DegreeInfo(BaseModel):
    degree_type: Optional[str] = None
    field_of_study: Optional[str] = None
    institution: Optional[str] = None
    graduation_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EducationExtraction(BaseModel):
    undergraduate: Optional[DegreeInfo] = None
    graduate: Optional[DegreeInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_education() -> str:
    return """
    Extract Kevin Warsh's educational credentials as presented in the provided answer text. 
    Return a JSON object with two objects: 'undergraduate' and 'graduate'. For each, extract:
    - degree_type: The degree type exactly as stated (e.g., "Bachelor of Arts", "B.A.", "Juris Doctor", "J.D.").
    - field_of_study: The field/major as stated (e.g., "Public Policy"). If not mentioned, return null.
    - institution: The institution name as stated (e.g., "Stanford University", "Harvard Law School").
    - graduation_year: The graduation year as a string exactly as stated. If formatted as "'92" or "Class of 1992", extract the visible year text (e.g., "'92" or "1992"). If not mentioned, return null.
    - sources: A list of all URLs explicitly cited in the answer that support that specific credential. Only include URLs that are clearly associated with that credential.

    IMPORTANT:
    - Do NOT invent any information. Extract only what appears in the answer text.
    - If any field is missing, set it to null; if no URLs are provided for that credential, return an empty list for 'sources'.
    - Keep the extracted strings verbatim from the answer (do not normalize or reformat).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def build_full_credential_claim(person: str, deg: DegreeInfo, require_field: bool) -> str:
    """
    Build a human-readable claim sentence from extracted degree info.
    """
    deg_type = deg.degree_type or ""
    institution = deg.institution or ""
    year = deg.graduation_year or ""
    field = (deg.field_of_study or "").strip()

    if require_field and field:
        return f"{person} received a {deg_type} degree in {field} from {institution} in {year}."
    else:
        # Omit field if not required or not available
        return f"{person} received a {deg_type} degree from {institution} in {year}."


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_degree(
    evaluator: Evaluator,
    parent_node,
    label_id: str,
    description: str,
    extracted: Optional[DegreeInfo],
    expected: Dict[str, str],
    require_field: bool,
) -> None:
    """
    Build and run verification nodes for a single degree (undergraduate or graduate).
    - Sequential gating: if existence fails, subsequent checks are auto-skipped.
    - Components string-equivalence checks compare extracted strings to expected strings (no web evidence needed).
    - A dedicated source-grounded leaf verifies the full credential with the cited URLs.
    """
    degree_node = evaluator.add_sequential(
        id=label_id,
        desc=description,
        parent=parent_node,
        critical=False,  # allow partial across degrees
    )

    # Existence and minimal completeness (critical)
    has_core_fields = (
        extracted is not None
        and (extracted.degree_type or "").strip() != ""
        and (extracted.institution or "").strip() != ""
        and (extracted.graduation_year or "").strip() != ""
        and len(extracted.sources) > 0
    )
    evaluator.add_custom_node(
        result=has_core_fields,
        id=f"{label_id}_provided",
        desc=f"{label_id.capitalize()} information includes degree type, institution, graduation year, and at least one source URL",
        parent=degree_node,
        critical=True,
    )

    # Verification group (parallel, runs only if existence passed)
    verify_group = evaluator.add_parallel(
        id=f"{label_id}_verification",
        desc=f"{label_id.capitalize()} verification",
        parent=degree_node,
        critical=False,
    )

    # Components match expected (parallel)
    components = evaluator.add_parallel(
        id=f"{label_id}_components",
        desc=f"{label_id.capitalize()} components match expected",
        parent=verify_group,
        critical=False,
    )

    # Degree type match (critical within components)
    deg_type_leaf = evaluator.add_leaf(
        id=f"{label_id}_degree_type_match",
        desc=f"The extracted degree type matches the expected '{expected.get('degree_type', '')}'",
        parent=components,
        critical=True,
    )
    extracted_deg_type = extracted.degree_type if extracted else ""
    await evaluator.verify(
        claim=f"The extracted degree type '{extracted_deg_type}' is equivalent to '{expected.get('degree_type', '')}'.",
        node=deg_type_leaf,
        additional_instruction=(
            "Judge purely by string equivalence and common-sense abbreviation/synonym equivalence. "
            "For example: Bachelor of Arts ≈ B.A. ≈ AB; Juris Doctor ≈ J.D. ≈ Doctor of Jurisprudence. "
            "Case-insensitive; ignore punctuation and parentheses differences."
        ),
    )

    # Field of study match (only required for undergraduate based on task spec)
    field_leaf = evaluator.add_leaf(
        id=f"{label_id}_field_match",
        desc=f"The extracted field of study matches the expected '{expected.get('field_of_study', '')}'",
        parent=components,
        critical=True if require_field else False,
    )
    extracted_field = extracted.field_of_study if extracted else ""
    await evaluator.verify(
        claim=f"The extracted field of study '{extracted_field}' is equivalent to '{expected.get('field_of_study', '')}'.",
        node=field_leaf,
        additional_instruction=(
            "Judge purely by string/semantic equivalence. Accept minor variants (e.g., 'Public Policy' vs 'public policy'). "
            "If the extracted field is null/empty and a field is expected, this should be incorrect."
        ),
    )

    # Institution match (critical within components)
    inst_leaf = evaluator.add_leaf(
        id=f"{label_id}_institution_match",
        desc=f"The extracted institution matches the expected '{expected.get('institution', '')}'",
        parent=components,
        critical=True,
    )
    extracted_inst = extracted.institution if extracted else ""
    await evaluator.verify(
        claim=f"The extracted institution '{extracted_inst}' is the same as '{expected.get('institution', '')}'.",
        node=inst_leaf,
        additional_instruction=(
            "Judge purely by name equivalence and reasonable variants (e.g., abbreviations, added 'University' or 'School'). "
            "Case-insensitive; ignore punctuation."
        ),
    )

    # Graduation year match (critical within components)
    year_leaf = evaluator.add_leaf(
        id=f"{label_id}_year_match",
        desc=f"The extracted graduation year matches the expected '{expected.get('graduation_year', '')}'",
        parent=components,
        critical=True,
    )
    extracted_year = extracted.graduation_year if extracted else ""
    await evaluator.verify(
        claim=f"The extracted graduation year '{extracted_year}' represents the same year as '{expected.get('graduation_year', '')}'.",
        node=year_leaf,
        additional_instruction=(
            "Consider '1992' equivalent to \"'92\" or 'Class of 1992'. Focus on whether both denote the same year."
        ),
    )

    # Source-grounded verification for the full credential (critical)
    support_leaf = evaluator.add_leaf(
        id=f"{label_id}_supported_by_sources",
        desc=f"The full {label_id} credential is supported by cited sources",
        parent=verify_group,
        critical=True,
    )
    sources = extracted.sources if extracted else []
    claim_full = build_full_credential_claim("Kevin Warsh", extracted or DegreeInfo(), require_field=require_field)
    await evaluator.verify(
        claim=claim_full,
        node=support_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the provided webpages explicitly support the complete credential (person, degree type, "
            "institution, and graduation year; include field if present). Allow standard synonyms/abbreviations "
            "for degree types (e.g., B.A. for Bachelor of Arts; J.D. for Juris Doctor) and minor phrasing variants "
            "(e.g., 'Class of 1992'). The page must clearly be about Kevin Warsh."
        ),
    )


# Convenience wrappers for each degree
async def verify_undergraduate(
    evaluator: Evaluator,
    parent_node,
    extracted: Optional[DegreeInfo],
) -> None:
    await verify_degree(
        evaluator=evaluator,
        parent_node=parent_node,
        label_id="undergraduate_degree",
        description="Kevin Warsh received a Bachelor of Arts (B.A.) degree in Public Policy from Stanford University in 1992",
        extracted=extracted,
        expected=EXPECTED_UNDERGRAD,
        require_field=True,
    )


async def verify_graduate(
    evaluator: Evaluator,
    parent_node,
    extracted: Optional[DegreeInfo],
) -> None:
    await verify_degree(
        evaluator=evaluator,
        parent_node=parent_node,
        label_id="graduate_degree",
        description="Kevin Warsh received a Juris Doctor (J.D.) degree from Harvard Law School in 1995",
        extracted=extracted,
        expected=EXPECTED_GRAD,
        require_field=False,  # Field not required/expected for J.D.
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
    Evaluate an answer for Kevin Warsh's educational background.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level: undergraduate vs graduate can be evaluated independently
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
    edu_info = await evaluator.extract(
        prompt=prompt_extract_education(),
        template_class=EducationExtraction,
        extraction_name="education_extraction",
    )

    # Add ground truth (for transparency/debugging; verification still relies on sources)
    evaluator.add_ground_truth(
        {
            "expected_undergraduate": EXPECTED_UNDERGRAD,
            "expected_graduate": EXPECTED_GRAD,
        },
        gt_type="expected_credentials",
    )

    # Build rubric root node for educational credentials
    edu_root = evaluator.add_parallel(
        id="Educational_Credentials",
        desc="Verify Kevin Warsh's educational credentials including undergraduate and graduate degrees",
        parent=root,
        critical=False,
    )

    # Verify Undergraduate
    await verify_undergraduate(
        evaluator=evaluator,
        parent_node=edu_root,
        extracted=edu_info.undergraduate if edu_info else None,
    )

    # Verify Graduate
    await verify_graduate(
        evaluator=evaluator,
        parent_node=edu_root,
        extracted=edu_info.graduate if edu_info else None,
    )

    return evaluator.get_summary()