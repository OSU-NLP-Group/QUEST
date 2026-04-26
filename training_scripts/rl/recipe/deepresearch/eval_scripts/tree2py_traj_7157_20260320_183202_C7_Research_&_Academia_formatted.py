import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "federal_oa_compliance_2026"
TASK_DESCRIPTION = (
    "A biomedical researcher at Stanford University received an NIH R01 research grant (award R01GM135790) in August 2025 "
    "with a total award amount of $650,000 over 4 years. The researcher is now preparing to publish the first major findings "
    "from this funded research in the journal Science, with an expected publication date of April 2026.\n\n"
    "The researcher's current publication plan includes the following elements:\n\n"
    "1. Submit the manuscript to Science (impact factor 56.9) for standard peer review\n"
    "2. Upon acceptance, allow Science to publish the article under its traditional subscription model\n"
    "3. Make the article open access after a 6-month embargo period\n"
    "4. Deposit the accepted manuscript version in PubMed Central after the 6-month embargo\n"
    "5. Include acknowledgment of NIH funding and the grant number in the manuscript\n"
    "6. Make the underlying research data available \"upon reasonable request\" to interested researchers\n"
    "7. Include complete metadata with author affiliations and grant information\n\n"
    "Given that the federal open access mandate for all federally funded research became effective January 1, 2026, requiring immediate public access with no embargo periods:\n\n"
    "Verify whether this researcher's publication plan fully complies with all federal open access requirements that apply to publications accepted after January 1, 2026. "
    "For each requirement that is not met, explain what specific change is needed to achieve compliance."
)


# --------------------------------------------------------------------------- #
# Minimal extraction (optional: capture any URLs the answer cites)            #
# --------------------------------------------------------------------------- #
class AnswerURLs(BaseModel):
    """Extract all URLs cited in the answer (if any)."""
    urls: List[str] = Field(default_factory=list)


def prompt_extract_urls() -> str:
    return (
        "Extract every URL explicitly mentioned in the answer text. Return them as a JSON array under the key 'urls'. "
        "Include policy pages, agency guidance, journal policy pages, or repository links if present. If none, return an empty array."
    )


# --------------------------------------------------------------------------- #
# Helper for adding and verifying a single requirement leaf                   #
# --------------------------------------------------------------------------- #
async def add_and_verify_requirement(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    claim: str,
    critical: bool,
    add_ins: Optional[str] = None,
) -> None:
    """
    Add a leaf node and verify it against the agent's answer using simple verification.
    The claim should be phrased to check whether the answer clearly:
      1) understands the requirement as of Jan 1, 2026; and
      2) correctly evaluates the plan; and
      3) proposes a specific fix when the plan is non-compliant.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction=add_ins or "None",
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an agent's answer for compliance with the 2026 Federal Open Access requirements.
    We judge whether the answer:
      - correctly identifies each requirement that applies after Jan 1, 2026,
      - correctly evaluates the given plan against those requirements,
      - and, where non-compliant, provides a specific corrective action.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # parallel checks for independent requirements
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

    # Optional: extract any URLs in the answer (recorded in the summary; not required for verification)
    extracted_urls = await evaluator.extract(
        prompt=prompt_extract_urls(),
        template_class=AnswerURLs,
        extraction_name="answer_urls",
    )

    # Add a ground-truth sketch for transparency (what is expected given the described plan)
    # Note: Root in obj_task_eval cannot be critical if it has non-critical children, so we keep container non-critical.
    evaluator.add_ground_truth({
        "expected_application": {
            "federally_funded": True,
            "effective_date_applies": True,    # publication expected April 2026 ⇒ acceptance likely after 1/1/2026
        },
        "expected_compliance_by_requirement": {
            "No_Embargo_Immediate_Access": False,            # plan uses 6-month embargo ⇒ not compliant
            "Data_Sharing_Requirement": False,               # 'upon reasonable request' ⇒ not compliant
            "Repository_Deposit_Required": False,            # PMC after 6 months ⇒ not compliant
            "Federally_Funded_Research_Coverage": True,      # NIH R01 ⇒ covered
            "Effective_Date_Compliance": True,               # applies to this timeline
            "NIH_Funding_Acknowledgment": True,              # plan includes
            "Grant_Number_Tracking": True,                   # plan includes grant number
            "Open_Access_License": False,                    # plan does not specify an OA reuse license
            "Metadata_Quality": True,                        # plan includes complete metadata
            "Accepted_Manuscript_Accessibility": False,      # AM only after 6 months ⇒ not compliant
            "Data_Format_Accessibility": False,              # not specified; 'upon request' implies not
            "Data_Management_Plan_Adherence": None           # not specified in plan; desirable to confirm
        },
        "note": "We expect the answer to explicitly identify each non-compliance and prescribe a concrete fix."
    })

    # Build container node mirroring the rubric (set non-critical to allow soft children)
    # The provided rubric marked the container as critical, but it includes non-critical children,
    # which violates the framework constraint "critical parent cannot have non-critical children".
    # Therefore we set the container to non-critical here.
    container = evaluator.add_parallel(
        id="Federal_Open_Access_Compliance_2026",
        desc="Verify that the researcher's publication plan for NIH-funded research complies with all federal open access requirements effective January 1, 2026",
        parent=root,
        critical=False,
    )

    # Common instruction to enforce that the agent both diagnoses and prescribes
    must_diagnose_and_fix = (
        "Award credit only if the answer BOTH (1) correctly states the 2026 federal requirement and applies it to the given plan, "
        "(2) explicitly marks the plan as compliant or non-compliant for this item, and "
        "(3) when non-compliant, gives a specific actionable fix (what to change and how). "
        "Allow reasonable paraphrase and synonyms; prioritize substance over exact wording."
    )

    # 1) No embargo / immediate access (critical)
    await add_and_verify_requirement(
        evaluator,
        container,
        node_id="No_Embargo_Immediate_Access",
        node_desc="Publication must be made freely and immediately available to the public with no embargo period (immediate availability required as of January 2026)",
        critical=True,
        claim=(
            "The answer explicitly states that, effective January 1, 2026, federally funded publications must be publicly accessible immediately with no embargo, "
            "correctly judges the plan's 'open access after a 6-month embargo' as non-compliant, and recommends a concrete fix such as selecting an immediate open-access option "
            "or immediately making the accepted manuscript publicly available at publication."
        ),
        add_ins=must_diagnose_and_fix,
    )

    # 2) Data sharing requirement (critical)
    await add_and_verify_requirement(
        evaluator,
        container,
        node_id="Data_Sharing_Requirement",
        node_desc="Supporting research data must be shared publicly and immediately in accordance with federal requirements, not merely made available upon request",
        critical=True,
        claim=(
            "The answer explains that supporting research data must be shared publicly and immediately in an appropriate repository (FAIR, machine-readable) rather than 'upon reasonable request', "
            "judges the current plan as non-compliant, and prescribes a clear fix (e.g., deposit the datasets at publication in a suitable public repository with a data availability statement and persistent identifier)."
        ),
        add_ins=must_diagnose_and_fix,
    )

    # 3) Repository deposit required (critical)
    await add_and_verify_requirement(
        evaluator,
        container,
        node_id="Repository_Deposit_Required",
        node_desc="Publication must be deposited in an approved public repository such as PubMed Central immediately upon acceptance",
        critical=True,
        claim=(
            "The answer states that immediate deposit in an approved public repository (e.g., PubMed Central for NIH) is required at acceptance/publication with no embargo, "
            "identifies the plan's 'deposit to PMC after 6 months' as non-compliant, and recommends updating the plan to deposit immediately upon acceptance/publication."
        ),
        add_ins=must_diagnose_and_fix,
    )

    # 4) Federally funded coverage (critical)
    await add_and_verify_requirement(
        evaluator,
        container,
        node_id="Federally_Funded_Research_Coverage",
        node_desc="Research must be funded by a federal agency (NIH, NSF, or other US federal agency) to fall under the mandate",
        critical=True,
        claim=(
            "The answer acknowledges that the research is federally funded (NIH R01GM135790) and therefore falls under the 2026 federal open-access mandate."
        ),
        add_ins="Give credit if the answer clearly notes NIH funding and that the federal OA requirements therefore apply.",
    )

    # 5) Effective date compliance (critical)
    await add_and_verify_requirement(
        evaluator,
        container,
        node_id="Effective_Date_Compliance",
        node_desc="Requirements apply to publications submitted and accepted after January 1, 2026",
        critical=True,
        claim=(
            "The answer correctly applies the effective date, noting that the requirements apply to publications accepted after January 1, 2026, "
            "and that this manuscript (expected April 2026 publication) will be covered, so the 2026 requirements apply."
        ),
        add_ins="Allow reasonable inference from the April 2026 publication timeline; an explicit acceptance date is not required.",
    )

    # 6) NIH funding acknowledgment (critical)
    await add_and_verify_requirement(
        evaluator,
        container,
        node_id="NIH_Funding_Acknowledgment",
        node_desc="Publication includes proper acknowledgment of NIH grant funding as required for tracking compliance",
        critical=True,
        claim=(
            "The answer confirms that the publication should include a proper NIH funding acknowledgment and indicates that the plan already includes this element."
        ),
        add_ins="Give credit if the answer explicitly affirms inclusion of the NIH funding acknowledgment and identifies it as required.",
    )

    # 7) Grant number tracking (critical)
    await add_and_verify_requirement(
        evaluator,
        container,
        node_id="Grant_Number_Tracking",
        node_desc="NIH grant number is included in publication as required for compliance tracking",
        critical=True,
        claim=(
            "The answer confirms inclusion of the NIH grant number (R01GM135790) in the manuscript for compliance tracking and indicates that the plan includes it."
        ),
        add_ins="Give credit if the answer clearly identifies the grant number must be included and affirms that the plan does so.",
    )

    # 8) Open access license (critical in rubric)
    await add_and_verify_requirement(
        evaluator,
        container,
        node_id="Open_Access_License",
        node_desc="Publication uses an appropriate open access license allowing reuse and redistribution as required for federally funded publications",
        critical=True,
        claim=(
            "The answer notes that the publicly accessible version should carry an appropriate open-access license that permits reuse (e.g., CC BY or a similarly permissive license), "
            "and recommends selecting such a license rather than relying solely on a subscription-only model."
        ),
        add_ins=(
            "Award credit if the answer calls out the need for a reuse-permitting OA license and recommends a concrete license choice or path to ensure compliant reuse."
        ),
    )

    # 9) Metadata quality (non-critical)
    await add_and_verify_requirement(
        evaluator,
        container,
        node_id="Metadata_Quality",
        node_desc="Publication metadata is complete with grant numbers, author affiliations, and accurate information",
        critical=False,
        claim=(
            "The answer confirms that complete and accurate publication metadata (author affiliations and grant information, including grant numbers) will be provided."
        ),
        add_ins="Satisfy if the answer explicitly affirms complete metadata with affiliations and grant identifiers.",
    )

    # 10) Accepted manuscript accessibility (critical)
    await add_and_verify_requirement(
        evaluator,
        container,
        node_id="Accepted_Manuscript_Accessibility",
        node_desc="The peer-reviewed accepted manuscript version is made publicly accessible immediately as required",
        critical=True,
        claim=(
            "The answer specifies that the peer-reviewed accepted manuscript must be made publicly accessible immediately (no embargo), "
            "identifies the current 'after 6 months' plan as non-compliant, and prescribes an immediate public release (e.g., PMC at acceptance)."
        ),
        add_ins=must_diagnose_and_fix,
    )

    # 11) Data format accessibility (critical)
    await add_and_verify_requirement(
        evaluator,
        container,
        node_id="Data_Format_Accessibility",
        node_desc="Research data is provided in formats that are accessible and usable by the research community as required",
        critical=True,
        claim=(
            "The answer requires that research data be provided in accessible, usable, machine-readable formats with appropriate metadata, "
            "and recommends selecting community-standard formats and metadata rather than relying on 'upon request'."
        ),
        add_ins="Award credit if the answer explicitly addresses format/metadata usability (FAIR-aligned) and gives a concrete prescription.",
    )

    # 12) Data Management Plan adherence (non-critical)
    await add_and_verify_requirement(
        evaluator,
        container,
        node_id="Data_Management_Plan_Adherence",
        node_desc="The data management and sharing plan from the grant application is being followed",
        critical=False,
        claim=(
            "The answer states that the NIH data management and sharing plan (DMSP) filed with the grant will be followed and, if needed, updated to ensure immediate public data access at publication."
        ),
        add_ins="Give credit if the answer explicitly references adherence to (and updating if necessary) the NIH DMSP.",
    )

    # Optionally record the URLs the answer cited
    evaluator.add_custom_info(
        info={"urls_cited_in_answer": extracted_urls.urls},
        info_type="answer_citations",
        info_name="answer_policy_urls",
    )

    # Return the evaluation summary
    return evaluator.get_summary()