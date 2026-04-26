import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "learning_platform_selection"
TASK_DESCRIPTION = (
    "A major research university is establishing a comprehensive online education program and needs to select a primary learning management platform. "
    "The platform must meet strict compliance requirements, support modern interoperability standards, and provide essential features for effective online teaching and learning.\n\n"
    "Identify one learning management platform (with its official name and website URL) that satisfies the following requirements:\n\n"
    "Mandatory Compliance Requirements:\n"
    "- Must comply with FERPA (Family Educational Rights and Privacy Act) for protecting student educational records, with publicly available documentation\n"
    "- Must meet WCAG 2.1 Level AA accessibility standards, with publicly available conformance statement or VPAT documentation\n"
    "- Must support at least two of the following three interoperability standards: SCORM (Sharable Content Object Reference Model), xAPI/LRS (Experience API with Learning Record Store), or LTI (Learning Tools Interoperability by 1EdTech), with publicly available technical documentation\n\n"
    "Mandatory Functional Requirements:\n"
    "- Must include an automated assessment system with grading and feedback capabilities\n\n"
    "Additional Desired Features (at least three of the following):\n"
    "- Learning analytics dashboard with real-time data visualization and reporting\n"
    "- Documented API endpoints for external system integration\n"
    "- Mobile learning support (native app or responsive web design)\n"
    "- Collaborative learning tools (peer review, group workspaces, or discussion forums)\n"
    "- Alignment with Quality Matters rubric standards or equivalent quality framework\n"
    "- Support for Creative Commons licensed content or Open Educational Resources (OER)\n"
    "- Continuing Education Units (CEU) tracking capability following IACET standards (1 CEU = 10 contact hours)\n"
    "- Section 508 compliance for U.S. federal accessibility requirements\n\n"
    "Provide the platform name, official website URL, and documentation references that verify compliance with the mandatory requirements."
)

# Canonical choices for interoperability and features
INTEROP_CHOICES = ["SCORM", "xAPI/LRS", "LTI"]
FEATURE_CHOICES = [
    "Learning analytics dashboard",
    "API endpoints",
    "Mobile learning support",
    "Collaborative learning tools",
    "Quality Matters alignment",
    "OER/Creative Commons support",
    "CEU tracking (IACET)",
    "Section 508 compliance",
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PlatformInfo(BaseModel):
    name: Optional[str] = None
    official_url: Optional[str] = None


class InteropInfo(BaseModel):
    supported: List[str] = Field(default_factory=list)  # values from INTEROP_CHOICES
    docs_by_standard: Dict[str, List[str]] = Field(default_factory=dict)  # key in INTEROP_CHOICES -> list of URLs


class ComplianceInfo(BaseModel):
    ferpa_docs: List[str] = Field(default_factory=list)
    wcag_aa_docs: List[str] = Field(default_factory=list)
    interop: InteropInfo = Field(default_factory=InteropInfo)


class AssessmentInfo(BaseModel):
    has_automated_assessment: Optional[bool] = None
    description: Optional[str] = None
    docs: List[str] = Field(default_factory=list)


class FeaturesInfo(BaseModel):
    selected_features: List[str] = Field(default_factory=list)  # values from FEATURE_CHOICES
    docs_by_feature: Dict[str, List[str]] = Field(default_factory=dict)  # optional: feature -> list of URLs


class PlatformSelectionExtraction(BaseModel):
    platform: PlatformInfo = Field(default_factory=PlatformInfo)
    compliance: ComplianceInfo = Field(default_factory=ComplianceInfo)
    functional: AssessmentInfo = Field(default_factory=AssessmentInfo)
    features: FeaturesInfo = Field(default_factory=FeaturesInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_platform_selection() -> str:
    return (
        "Extract exactly one learning management platform from the answer. If multiple are mentioned, pick the first one.\n\n"
        "Return a structured JSON object with the following fields:\n"
        "platform:\n"
        "  - name: Official name of the LMS/platform (string; null if not provided)\n"
        "  - official_url: Official website URL or official product page URL (string; null if not provided)\n"
        "compliance:\n"
        "  - ferpa_docs: List of publicly accessible URLs that explicitly state FERPA compliance or privacy protections for student records\n"
        "  - wcag_aa_docs: List of publicly accessible URLs to a WCAG 2.1 Level AA conformance statement or VPAT document\n"
        "  - interop:\n"
        "      supported: List of supported interoperability standards chosen only from this canonical set: "
        f"{INTEROP_CHOICES}. Normalize synonyms (e.g., 'xAPI', 'Experience API', 'LRS' -> 'xAPI/LRS'; 'IMS LTI', 'LTI 1.3', '1EdTech LTI' -> 'LTI').\n"
        "      docs_by_standard: A mapping from each supported standard (key must be exactly one of the canonical choices) to a list of publicly accessible URLs that document technical support for that standard.\n\n"
        "functional:\n"
        "  - has_automated_assessment: Boolean indicating whether the platform includes automated assessment with grading and feedback capabilities (true/false; null if unclear)\n"
        "  - description: Short text snippet (from the answer) describing assessment/grading/feedback capabilities (string; null if not provided)\n"
        "  - docs: List of publicly accessible URLs that document assessment/grading/feedback capabilities\n\n"
        "features:\n"
        "  - selected_features: List of features supported by the platform, chosen only from this canonical list: "
        f"{FEATURE_CHOICES}. Normalize synonyms appropriately.\n"
        "  - docs_by_feature: Optional mapping from selected feature (canonical string) to a list of publicly accessible URLs documenting support for that feature.\n\n"
        "General rules:\n"
        "1) Do NOT invent information. Only extract details explicitly present in the answer.\n"
        "2) If a field is missing, set it to null or empty list appropriately.\n"
        "3) All URLs must be valid and complete. If protocol is missing, prepend 'http://'.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_nonempty_url_list(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


def _pick_two_supported_with_docs(interop: InteropInfo) -> List[Tuple[str, List[str]]]:
    """
    From interop.supported and interop.docs_by_standard, pick the first two standards
    that have at least one documentation URL. Returns list of (standard, urls).
    """
    pairs: List[Tuple[str, List[str]]] = []
    for std in interop.supported:
        if std in INTEROP_CHOICES:
            urls = interop.docs_by_standard.get(std, [])
            if _has_nonempty_url_list(urls):
                pairs.append((std, urls))
        # Stop after two
        if len(pairs) >= 2:
            break
    return pairs


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def _verify_platform_identification(
    evaluator: Evaluator,
    parent_node,
    data: PlatformSelectionExtraction
) -> None:
    """
    Build and verify the Platform Identification subtree:
    - Platform_Name (existence)
    - Official_Website_URL (existence)
    """
    node = evaluator.add_parallel(
        id="Platform_Identification",
        desc="Provide the platform’s official name and official website URL.",
        parent=parent_node,
        critical=True
    )

    # Platform name existence (critical)
    evaluator.add_custom_node(
        result=bool(data.platform and data.platform.name and data.platform.name.strip()),
        id="Platform_Name",
        desc="Provide the official name of the learning management platform.",
        parent=node,
        critical=True
    )

    # Official website URL existence (critical)
    official_url = data.platform.official_url or ""
    evaluator.add_custom_node(
        result=bool(official_url and official_url.strip()),
        id="Official_Website_URL",
        desc="Provide a URL to the platform’s official website (or official product page).",
        parent=node,
        critical=True
    )


async def _verify_compliance_block(
    evaluator: Evaluator,
    parent_node,
    data: PlatformSelectionExtraction
) -> None:
    """
    Build and verify Mandatory Compliance subtree:
    - FERPA_Compliance_With_Public_Docs: existence + URL verification
    - WCAG_2_1_AA_With_Public_Docs: existence + URL verification
    - Interoperability_At_Least_Two_With_Public_Docs: count check + per-standard verifications for at least two
    """
    comp_node = evaluator.add_parallel(
        id="Mandatory_Compliance_With_Public_Documentation",
        desc="Satisfy all mandatory compliance requirements with publicly available documentation references.",
        parent=parent_node,
        critical=True
    )

    # 1) FERPA
    ferpa_docs = data.compliance.ferpa_docs
    ferpa_docs_exist = _has_nonempty_url_list(ferpa_docs)
    evaluator.add_custom_node(
        result=ferpa_docs_exist,
        id="FERPA_Doc_Exists",
        desc="At least one publicly accessible URL provided for FERPA compliance.",
        parent=comp_node,
        critical=True
    )
    ferpa_leaf = evaluator.add_leaf(
        id="FERPA_Compliance_With_Public_Docs",
        desc="Shows the platform complies with FERPA and includes at least one publicly accessible documentation/reference URL supporting this claim.",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim="The platform complies with FERPA for protecting student educational records, as evidenced by the provided documentation.",
        node=ferpa_leaf,
        sources=ferpa_docs,
        additional_instruction="Verify that the provided URL(s) explicitly reference FERPA compliance or protections of student educational records."
    )

    # 2) WCAG 2.1 Level AA
    wcag_docs = data.compliance.wcag_aa_docs
    wcag_docs_exist = _has_nonempty_url_list(wcag_docs)
    evaluator.add_custom_node(
        result=wcag_docs_exist,
        id="WCAG_Doc_Exists",
        desc="At least one publicly accessible conformance statement or VPAT URL provided for WCAG 2.1 Level AA.",
        parent=comp_node,
        critical=True
    )
    wcag_leaf = evaluator.add_leaf(
        id="WCAG_2_1_AA_With_Public_Docs",
        desc="Shows the platform meets WCAG 2.1 Level AA and includes a publicly accessible conformance statement or VPAT URL supporting this claim.",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim="The platform meets WCAG 2.1 Level AA accessibility with a public conformance statement or VPAT.",
        node=wcag_leaf,
        sources=wcag_docs,
        additional_instruction="Verify that the provided URL(s) reference WCAG 2.1 Level AA specifically (e.g., VPAT, conformance statement)."
    )

    # 3) Interoperability: at least two of SCORM, xAPI/LRS, LTI with technical documentation
    interop_node = evaluator.add_parallel(
        id="Interoperability_At_Least_Two_With_Public_Docs",
        desc="Shows the platform supports at least two of {SCORM, xAPI/LRS, LTI} and includes publicly accessible technical documentation URL(s).",
        parent=comp_node,
        critical=True
    )
    pairs = _pick_two_supported_with_docs(data.compliance.interop)
    evaluator.add_custom_node(
        result=(len(pairs) >= 2),
        id="Interoperability_Count_At_Least_Two",
        desc="At least two interoperability standards are listed with documentation URLs.",
        parent=interop_node,
        critical=True
    )

    # Add one leaf per selected standard (exactly the two picked)
    for std, urls in pairs[:2]:
        leaf = evaluator.add_leaf(
            id=f"Interoperability_{std}_Supported",
            desc=f"Documentation shows the platform supports {std}.",
            parent=interop_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The platform supports {std}.",
            node=leaf,
            sources=urls,
            additional_instruction=(
                f"Confirm that the provided documentation explicitly states {std} support. "
                "For xAPI/LRS, references to Experience API or LRS are acceptable; for LTI, references to IMS/1EdTech LTI are acceptable."
            )
        )


async def _verify_functional_block(
    evaluator: Evaluator,
    parent_node,
    data: PlatformSelectionExtraction
) -> None:
    """
    Build and verify Mandatory Functional subtree:
    - Automated_Assessment_Grading_Feedback: verify via documentation URLs
    - Assessment_Documentation_Reference_URL: existence of at least one URL
    """
    func_node = evaluator.add_parallel(
        id="Mandatory_Functional_With_Documentation",
        desc="Satisfy the mandatory functional requirement and provide a supporting reference.",
        parent=parent_node,
        critical=True
    )

    # Existence of assessment docs (critical sibling so leaf verification will be auto-skipped if it fails)
    evaluator.add_custom_node(
        result=_has_nonempty_url_list(data.functional.docs),
        id="Assessment_Documentation_Reference_URL",
        desc="Provide at least one publicly accessible URL reference documenting the platform’s automated assessment/grading/feedback capabilities.",
        parent=func_node,
        critical=True
    )

    # Verification leaf for assessment capability
    assess_leaf = evaluator.add_leaf(
        id="Automated_Assessment_Grading_Feedback",
        desc="Platform includes an automated assessment system with grading and feedback capabilities.",
        parent=func_node,
        critical=True
    )
    # Prefer verifying with the provided documentation URLs
    await evaluator.verify(
        claim="The platform includes automated assessment with grading and feedback capabilities.",
        node=assess_leaf,
        sources=data.functional.docs,
        additional_instruction=(
            "Confirm that the documentation describes automated assessments (quizzes/tests), grading workflows, and feedback mechanisms."
        )
    )


async def _verify_additional_features(
    evaluator: Evaluator,
    parent_node,
    data: PlatformSelectionExtraction
) -> None:
    """
    Build and verify Additional Desired Features subtree:
    - Ensure at least three features are selected from the canonical list
    - Verify that the answer clearly indicates which ones
    """
    feat_node = evaluator.add_sequential(
        id="Additional_Desired_Features_Minimum_Three",
        desc="Platform supports at least three of the listed additional desired features, and the answer clearly indicates which ones (selected from the given list).",
        parent=parent_node,
        critical=True
    )

    # Step 1: Count check (critical)
    count_ok = len([f for f in data.features.selected_features if f in FEATURE_CHOICES]) >= 3
    evaluator.add_custom_node(
        result=count_ok,
        id="Additional_Features_Count_At_Least_Three",
        desc="At least three additional desired features are selected from the canonical set.",
        parent=feat_node,
        critical=True
    )

    # Step 2: Simple verification that the answer lists which features (no URL requirement in rubric)
    listed_leaf = evaluator.add_leaf(
        id="Additional_Features_Listed_Clearly",
        desc="The answer clearly indicates which additional features are supported.",
        parent=feat_node,
        critical=True
    )
    # Use simple verification leveraging the extracted list
    features_str = ", ".join(data.features.selected_features) if data.features.selected_features else "none"
    await evaluator.verify(
        claim=f"The platform supports at least three of the listed additional features: {features_str}.",
        node=listed_leaf,
        additional_instruction=(
            "Judge based on the provided answer context whether the platform supports at least three items from the canonical list; "
            "minor wording variations are acceptable as long as they correspond to the canonical features."
        )
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for selecting a compliant learning management platform.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extract structured selection info
    extraction = await evaluator.extract(
        prompt=prompt_extract_platform_selection(),
        template_class=PlatformSelectionExtraction,
        extraction_name="platform_selection_extraction"
    )

    # Add a critical top-level node reflecting the rubric's root
    selection_root = evaluator.add_parallel(
        id="Learning_Platform_Selection",
        desc="Identify one learning management platform and provide required identifiers and documentation showing it satisfies all stated requirements.",
        parent=root,
        critical=True
    )

    # Build subtrees
    await _verify_platform_identification(evaluator, selection_root, extraction)
    await _verify_compliance_block(evaluator, selection_root, extraction)
    await _verify_functional_block(evaluator, selection_root, extraction)
    await _verify_additional_features(evaluator, selection_root, extraction)

    # Record canonical sets and extracted summary as custom info
    evaluator.add_custom_info(
        {
            "interop_choices": INTEROP_CHOICES,
            "feature_choices": FEATURE_CHOICES
        },
        info_type="canonical_sets",
        info_name="canonical_sets"
    )
    evaluator.add_custom_info(
        {
            "platform": extraction.platform.dict() if extraction.platform else {},
            "compliance": {
                "ferpa_docs_count": len(extraction.compliance.ferpa_docs),
                "wcag_docs_count": len(extraction.compliance.wcag_aa_docs),
                "interop_supported": extraction.compliance.interop.supported,
                "interop_docs_by_standard": extraction.compliance.interop.docs_by_standard
            },
            "functional_docs_count": len(extraction.functional.docs),
            "features_selected": extraction.features.selected_features,
            "features_docs_by_feature": extraction.features.docs_by_feature
        },
        info_type="extracted_summary",
        info_name="extracted_summary"
    )

    # Return structured summary
    return evaluator.get_summary()