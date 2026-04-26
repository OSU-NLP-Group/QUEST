import asyncio
import logging
from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task constants and scenario facts (from the given problem statement)
# -----------------------------------------------------------------------------
TASK_ID = "trip_compliance_malaysia_china_202603"
TASK_DESCRIPTION = (
    "A United States family of four is planning a spring break trip from March 15-27, 2026. "
    "The itinerary includes: Kuala Lumpur, Malaysia (March 15-22, 2026, 7 days) and Shanghai, China (March 22-26, 2026, 4 days), "
    "with return to the United States on March 27, 2026. The family consists of: Parent A (US citizen, passport expires August 15, 2026), "
    "Parent B (US citizen, passport expires November 30, 2026), Child 1 (US citizen, age 12, passport expires July 20, 2026), "
    "and Child 2 (US citizen, age 9, passport expires October 10, 2026). Verify whether this family meets all entry requirements and "
    "documentation validity requirements for both Malaysia and China based on current 2026 regulations. Specifically, determine: "
    "(1) Whether US citizens need visas for Malaysia and China for the planned duration of stay, "
    "(2) Whether any additional entry documentation is required for Malaysia, "
    "(3) Whether all four family members' passports meet the passport validity requirements for both countries, and "
    "(4) Whether the trip is fully compliant with all requirements, and if not, identify which specific family members have documentation issues and for which destination(s)."
)

# Itinerary dates (given)
MALAYSIA_ARRIVAL = date(2026, 3, 15)
MALAYSIA_DEPARTURE = date(2026, 3, 22)
CHINA_ARRIVAL = date(2026, 3, 22)
CHINA_DEPARTURE = date(2026, 3, 26)

# Six-month rule target dates specified by the rubric (beyond exit date)
MALAYSIA_SIX_MONTH_TARGET = date(2026, 9, 22)  # 6 months beyond Malaysia exit date
CHINA_SIX_MONTH_TARGET = date(2026, 9, 26)     # 6 months beyond China exit date

# Family passports (given)
PARENT_A_EXP = date(2026, 8, 15)
PARENT_B_EXP = date(2026, 11, 30)
CHILD_1_EXP = date(2026, 7, 20)
CHILD_2_EXP = date(2026, 10, 10)

FAMILY_EXPIRATIONS = {
    "Parent A": PARENT_A_EXP,
    "Parent B": PARENT_B_EXP,
    "Child 1": CHILD_1_EXP,
    "Child 2": CHILD_2_EXP,
}

# Stays (given)
MALAYSIA_STAY_DAYS = (MALAYSIA_DEPARTURE - MALAYSIA_ARRIVAL).days  # 7
CHINA_STAY_DAYS = (CHINA_DEPARTURE - CHINA_ARRIVAL).days           # 4

# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class SourceExtraction(BaseModel):
    """
    Extract source URLs provided in the answer for each relevant policy area.
    Only extract URLs explicitly present in the answer.
    """
    malaysia_entry_urls: List[str] = Field(default_factory=list, description="URLs cited for Malaysia entry (visa/MDAC) requirements for U.S. citizens.")
    malaysia_passport_urls: List[str] = Field(default_factory=list, description="URLs cited for Malaysia passport validity requirements.")
    china_entry_urls: List[str] = Field(default_factory=list, description="URLs cited for China entry/visa policy for U.S. citizens.")
    china_passport_urls: List[str] = Field(default_factory=list, description="URLs cited for China passport validity requirements.")


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_sources() -> str:
    return """
    Extract the URLs (if any) that the answer cites for the following categories. Only extract URLs explicitly present in the answer text (including plain URLs or URLs inside markdown links). Do not invent or infer URLs.

    Fields to extract:
    - malaysia_entry_urls: All URLs that the answer uses to support Malaysia entry requirements for U.S. citizens (visa policy, visa-free duration, MDAC requirement).
    - malaysia_passport_urls: All URLs that the answer uses to support Malaysia passport validity requirements (e.g., 6-month validity).
    - china_entry_urls: All URLs that the answer uses to support China entry/visa requirements for U.S. citizens (including any stated visa-free policy and its end date).
    - china_passport_urls: All URLs that the answer uses to support China passport validity requirements (e.g., 6-month validity).

    Return a JSON object with exactly these four arrays of URLs. If no URLs are provided for a category, return an empty array for that category.
    """


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _is_valid_by(target: date, expiry: date) -> bool:
    """Return True if passport expiry is on or after the target date."""
    return expiry >= target


def _non_compliant_members(target: date, expirations: Dict[str, date]) -> List[str]:
    return [name for name, exp in expirations.items() if not _is_valid_by(target, exp)]


# -----------------------------------------------------------------------------
# Malaysia Segment Verification
# -----------------------------------------------------------------------------
async def build_malaysia_segment(evaluator: Evaluator, parent_node, sources: SourceExtraction) -> None:
    """
    Build and verify the Malaysia segment tree.

    Structure (faithful to rubric; all critical):
    - Malaysia_Segment (parallel, critical)
      - Malaysia_Entry_Authorization (parallel, critical)
        - Malaysia_Entry_Sources (leaf; sources presence)
        - Malaysia_Visa_Status (leaf; verify by URLs)
        - Malaysia_MDAC_Requirement (leaf; verify by URLs)
        - Malaysia_Stay_Duration (leaf; simple logic)
      - Malaysia_Passport_Validity (sequential, critical)
        - Malaysia_Passport_Sources (leaf; sources presence)
        - Malaysia_Six_Month_Rule (parallel, critical)
          - Parent_A_Malaysia_Validity (leaf; computed)
          - Parent_B_Malaysia_Validity (leaf; computed)
          - Child_1_Malaysia_Validity (leaf; computed)
          - Child_2_Malaysia_Validity (leaf; computed)
    """
    malaysia_node = evaluator.add_parallel(
        id="Malaysia_Segment",
        desc="Verify all entry and documentation requirements for the Malaysia portion (March 15-22, 2026) in Kuala Lumpur",
        parent=parent_node,
        critical=True
    )

    # 1) Entry Authorization
    entry_node = evaluator.add_parallel(
        id="Malaysia_Entry_Authorization",
        desc="Confirm the family meets Malaysia's entry authorization requirements for US citizens",
        parent=malaysia_node,
        critical=True
    )

    # 1.a) Sources presence (critical gating)
    evaluator.add_custom_node(
        result=bool(sources.malaysia_entry_urls),
        id="Malaysia_Entry_Sources",
        desc="Reference URLs confirming Malaysia entry requirements for US citizens (sources present)",
        parent=entry_node,
        critical=True
    )

    # 1.b) Visa status claim (verify by URLs)
    visa_leaf = evaluator.add_leaf(
        id="Malaysia_Visa_Status",
        desc="US citizens can enter Malaysia without a visa for tourism stays up to 90 days",
        parent=entry_node,
        critical=True
    )
    await evaluator.verify(
        claim="U.S. citizens do not need a visa for tourism/social visits to Malaysia for stays up to 90 days.",
        node=visa_leaf,
        sources=sources.malaysia_entry_urls,
        additional_instruction="Verify that the provided page(s) explicitly state visa-free entry (or visa exemption) for U.S. citizens up to 90 days for tourism or social visits. Treat '90 days' and '3 months' as equivalent."
    )

    # 1.c) MDAC requirement (verify by URLs)
    mdac_leaf = evaluator.add_leaf(
        id="Malaysia_MDAC_Requirement",
        desc="All foreign travelers must complete the Malaysia Digital Arrival Card (MDAC) before entry, effective December 1, 2023",
        parent=entry_node,
        critical=True
    )
    await evaluator.verify(
        claim="All foreign travelers must complete the Malaysia Digital Arrival Card (MDAC) prior to arrival in Malaysia, effective December 1, 2023.",
        node=mdac_leaf,
        sources=sources.malaysia_entry_urls,
        additional_instruction="Prefer official Malaysian Immigration sources; verify the effective date and that the MDAC applies to foreign travelers entering Malaysia."
    )

    # 1.d) Stay duration within limit (simple logic)
    stay_leaf = evaluator.add_leaf(
        id="Malaysia_Stay_Duration",
        desc="Planned stay of 7 days (March 15-22) is within the 90-day visa-free limit",
        parent=entry_node,
        critical=True
    )
    await evaluator.verify(
        claim="A 7-day stay is within a 90-day visa-free limit.",
        node=stay_leaf,
        sources=None,
        additional_instruction="This is a pure logic check; confirm that 7 ≤ 90."
    )

    # 2) Passport Validity
    passport_node = evaluator.add_sequential(
        id="Malaysia_Passport_Validity",
        desc="Verify all family members' passports meet Malaysia's validity requirements",
        parent=malaysia_node,
        critical=True
    )

    # 2.a) Passport validity sources presence (critical gating)
    evaluator.add_custom_node(
        result=bool(sources.malaysia_passport_urls),
        id="Malaysia_Passport_Sources",
        desc="Reference URLs confirming passport validity requirements for entry to Malaysia (sources present)",
        parent=passport_node,
        critical=True
    )

    # 2.b) Six-month rule member checks (computed)
    six_node = evaluator.add_parallel(
        id="Malaysia_Six_Month_Rule",
        desc="Verify that all family members' passports are valid for at least 6 months beyond the exit date from Malaysia (September 22, 2026)",
        parent=passport_node,
        critical=True
    )

    # Parent A
    evaluator.add_custom_node(
        result=_is_valid_by(MALAYSIA_SIX_MONTH_TARGET, PARENT_A_EXP),
        id="Parent_A_Malaysia_Validity",
        desc="Parent A's passport (expires August 15, 2026) must be valid until at least September 22, 2026",
        parent=six_node,
        critical=True
    )
    # Parent B
    evaluator.add_custom_node(
        result=_is_valid_by(MALAYSIA_SIX_MONTH_TARGET, PARENT_B_EXP),
        id="Parent_B_Malaysia_Validity",
        desc="Parent B's passport (expires November 30, 2026) must be valid until at least September 22, 2026",
        parent=six_node,
        critical=True
    )
    # Child 1
    evaluator.add_custom_node(
        result=_is_valid_by(MALAYSIA_SIX_MONTH_TARGET, CHILD_1_EXP),
        id="Child_1_Malaysia_Validity",
        desc="Child 1's passport (age 12, expires July 20, 2026) must be valid until at least September 22, 2026",
        parent=six_node,
        critical=True
    )
    # Child 2
    evaluator.add_custom_node(
        result=_is_valid_by(MALAYSIA_SIX_MONTH_TARGET, CHILD_2_EXP),
        id="Child_2_Malaysia_Validity",
        desc="Child 2's passport (age 9, expires October 10, 2026) must be valid until at least September 22, 2026",
        parent=six_node,
        critical=True
    )


# -----------------------------------------------------------------------------
# China Segment Verification
# -----------------------------------------------------------------------------
async def build_china_segment(evaluator: Evaluator, parent_node, sources: SourceExtraction) -> None:
    """
    Build and verify the China segment tree.

    Structure (faithful to rubric; all critical):
    - China_Segment (parallel, critical)
      - China_Entry_Authorization (parallel, critical)
        - China_Entry_Sources (leaf; sources presence)
        - China_Visa_Free_Eligibility (leaf; verify by URLs)
        - China_Stay_Duration (leaf; simple logic)
      - China_Passport_Validity (sequential, critical)
        - China_Passport_Sources (leaf; sources presence)
        - China_Six_Month_Rule (parallel, critical)
          - Parent_A_China_Validity (leaf; computed)
          - Parent_B_China_Validity (leaf; computed)
          - Child_1_China_Validity (leaf; computed)
          - Child_2_China_Validity (leaf; computed)
    """
    china_node = evaluator.add_parallel(
        id="China_Segment",
        desc="Verify all entry and documentation requirements for the China portion (March 22-26, 2026) in Shanghai",
        parent=parent_node,
        critical=True
    )

    # 1) Entry Authorization
    entry_node = evaluator.add_parallel(
        id="China_Entry_Authorization",
        desc="Confirm the family meets China's entry authorization requirements for US citizens",
        parent=china_node,
        critical=True
    )

    # 1.a) Sources presence (critical gating)
    evaluator.add_custom_node(
        result=bool(sources.china_entry_urls),
        id="China_Entry_Sources",
        desc="Reference URLs confirming China's visa-free policy extension and entry requirements for US citizens (sources present)",
        parent=entry_node,
        critical=True
    )

    # 1.b) Visa-free eligibility through Dec 31, 2026 (verify by URLs)
    visa_free_leaf = evaluator.add_leaf(
        id="China_Visa_Free_Eligibility",
        desc="US citizens are eligible for visa-free entry to China through December 31, 2026, under the extended unilateral visa-free policy",
        parent=entry_node,
        critical=True
    )
    await evaluator.verify(
        claim="U.S. citizens are eligible for visa-free entry to China through December 31, 2026, under an extended unilateral visa-free policy.",
        node=visa_free_leaf,
        sources=sources.china_entry_urls,
        additional_instruction="Carefully check whether the provided sources explicitly include U.S. citizens as beneficiaries of visa-free entry and that the policy remains in effect through December 31, 2026. If the sources exclude the U.S. or indicate a different end date or no visa-free for U.S. citizens, mark as not supported."
    )

    # 1.c) Stay duration within limit (simple logic)
    stay_leaf = evaluator.add_leaf(
        id="China_Stay_Duration",
        desc="Planned stay of 4 days (March 22-26) is within the 30-day visa-free limit",
        parent=entry_node,
        critical=True
    )
    await evaluator.verify(
        claim="A 4-day stay is within a 30-day limit.",
        node=stay_leaf,
        sources=None,
        additional_instruction="This is a pure logic check; confirm that 4 ≤ 30."
    )

    # 2) Passport Validity
    passport_node = evaluator.add_sequential(
        id="China_Passport_Validity",
        desc="Verify all family members' passports meet China's validity requirements",
        parent=china_node,
        critical=True
    )

    # 2.a) Passport validity sources presence (critical gating)
    evaluator.add_custom_node(
        result=bool(sources.china_passport_urls),
        id="China_Passport_Sources",
        desc="Reference URLs confirming passport validity requirements for entry to China (sources present)",
        parent=passport_node,
        critical=True
    )

    # 2.b) Six-month rule member checks (computed)
    six_node = evaluator.add_parallel(
        id="China_Six_Month_Rule",
        desc="Verify that all family members' passports are valid for at least 6 months beyond the exit date from China (September 26, 2026)",
        parent=passport_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_valid_by(CHINA_SIX_MONTH_TARGET, PARENT_A_EXP),
        id="Parent_A_China_Validity",
        desc="Parent A's passport (expires August 15, 2026) must be valid until at least September 26, 2026",
        parent=six_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_valid_by(CHINA_SIX_MONTH_TARGET, PARENT_B_EXP),
        id="Parent_B_China_Validity",
        desc="Parent B's passport (expires November 30, 2026) must be valid until at least September 26, 2026",
        parent=six_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_valid_by(CHINA_SIX_MONTH_TARGET, CHILD_1_EXP),
        id="Child_1_China_Validity",
        desc="Child 1's passport (age 12, expires July 20, 2026) must be valid until at least September 26, 2026",
        parent=six_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_valid_by(CHINA_SIX_MONTH_TARGET, CHILD_2_EXP),
        id="Child_2_China_Validity",
        desc="Child 2's passport (age 9, expires October 10, 2026) must be valid until at least September 26, 2026",
        parent=six_node,
        critical=True
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    Entry point to evaluate an agent's answer for the 2026 Malaysia/China trip compliance task.

    Returns:
        A structured evaluation summary (dict) with the verification tree and auxiliary information.
    """
    # Initialize evaluator (root is non-critical by design; we will add a critical child aggregation node)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root wrapper; actual critical sequential node added under it
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

    # Extract URLs cited in the answer
    extracted_sources = await evaluator.extract(
        prompt=prompt_extract_sources(),
        template_class=SourceExtraction,
        extraction_name="policy_source_urls",
    )

    # Add ground-truth scenario info for transparency (not used for verification)
    evaluator.add_ground_truth(
        {
            "itinerary": {
                "malaysia": {"arrival": str(MALAYSIA_ARRIVAL), "departure": str(MALAYSIA_DEPARTURE), "stay_days": MALAYSIA_STAY_DAYS},
                "china": {"arrival": str(CHINA_ARRIVAL), "departure": str(CHINA_DEPARTURE), "stay_days": CHINA_STAY_DAYS},
            },
            "six_month_targets": {
                "malaysia": str(MALAYSIA_SIX_MONTH_TARGET),
                "china": str(CHINA_SIX_MONTH_TARGET),
            },
            "family_passport_expirations": {k: str(v) for k, v in FAMILY_EXPIRATIONS.items()},
        },
        gt_type="scenario_facts",
    )

    # Build the critical sequential assessment node per rubric
    assess_node = evaluator.add_sequential(
        id="Trip_Compliance_Assessment",
        desc="Evaluate whether a US family of 4 can complete their planned two-destination spring break trip to Malaysia and China in March 2026 given their current passport expiration dates",
        parent=root,
        critical=True
    )

    # Malaysia segment (critical)
    await build_malaysia_segment(evaluator, assess_node, extracted_sources)

    # China segment (critical; will be auto-skipped if Malaysia segment fails due to sequential parent)
    await build_china_segment(evaluator, assess_node, extracted_sources)

    # Add computed compliance summaries as custom info for human readability
    non_compliant_malaysia = _non_compliant_members(MALAYSIA_SIX_MONTH_TARGET, FAMILY_EXPIRATIONS)
    non_compliant_china = _non_compliant_members(CHINA_SIX_MONTH_TARGET, FAMILY_EXPIRATIONS)
    evaluator.add_custom_info(
        {
            "malaysia": {
                "six_month_target": str(MALAYSIA_SIX_MONTH_TARGET),
                "non_compliant_members": non_compliant_malaysia,
            },
            "china": {
                "six_month_target": str(CHINA_SIX_MONTH_TARGET),
                "non_compliant_members": non_compliant_china,
            },
        },
        info_type="computed_passport_compliance",
    )

    return evaluator.get_summary()