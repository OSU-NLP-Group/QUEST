import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "il_rn_to_ca_childcare_director_2026_transition"
TASK_DESCRIPTION = (
    "A registered nurse currently holding an active Illinois RN license (with the current renewal cycle ending May 31, 2026) is planning to relocate to California in 2026 to open and direct a child care center. "
    "To ensure a smooth professional transition, they need comprehensive information about: "
    "(1) Illinois RN License Maintenance - What is the renewal deadline, and what are the complete continuing education requirements including total hours, timeframe, and mandatory topic breakdown? "
    "(2) California Child Care Center Director Qualifications - What are all the qualification pathways to serve as a director, including specific education unit requirements and experience verification standards? "
    "(3) California Licensing Pre-Application Requirements - What mandatory steps must be completed before submitting a license application, and what are the associated fees? "
    "(4) Cost Calculation - What is the total cost for completing the mandatory orientation (provide both online and in-person options) and obtaining one certified copy of a birth certificate from Kankakee County, Illinois? "
    "Provide a comprehensive response that addresses all four areas with specific details, numbers, and reference URLs for verification."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ILRNMaintenance(BaseModel):
    renewal_deadline: Optional[str] = None
    renewal_cycle_years: Optional[str] = None
    ce_total_hours: Optional[str] = None
    ce_timeframe: Optional[str] = None
    ce_mandatory_breakdown: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CADirectorQualifications(BaseModel):
    pathways: List[str] = Field(default_factory=list)
    pathway1_unit_breakdown: Optional[str] = None
    experience_standard: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CAPreApplication(BaseModel):
    orientation_requirement: Optional[str] = None
    orientation_fee_online: Optional[str] = None  # Preferably includes 54.85
    orientation_fee_inperson: Optional[str] = None  # Preferably 50
    orientation_nonrefundable_policy: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CostCalculation(BaseModel):
    birth_cert_first_fee: Optional[str] = None  # Expect "$10" or "10"
    birth_cert_additional_fee: Optional[str] = None  # Expect "$4" or "4"
    total_online_plus_birth: Optional[str] = None  # Stated total by the answer
    total_inperson_plus_birth: Optional[str] = None  # Stated total by the answer
    sources_birth_cert: List[str] = Field(default_factory=list)


class MasterExtraction(BaseModel):
    il: Optional[ILRNMaintenance] = None
    ca_director: Optional[CADirectorQualifications] = None
    ca_preapp: Optional[CAPreApplication] = None
    cost: Optional[CostCalculation] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_master() -> str:
    return """
Extract the following structured information exactly as stated in the answer. Return null for any field not explicitly stated. Include all relevant URLs provided in the answer for verification.

1) il (Illinois RN License Maintenance):
   - renewal_deadline: The stated renewal deadline date for the current cycle (e.g., "May 31, 2026").
   - renewal_cycle_years: The stated renewal cycle length (e.g., "2 years" or "biennial").
   - ce_total_hours: The stated total CE hours required (e.g., "20 contact hours").
   - ce_timeframe: The stated CE timeframe (e.g., "within the 24 months preceding license expiration").
   - ce_mandatory_breakdown: The stated mandatory topic breakdown (e.g., "1 hr sexual harassment prevention; 1 hr implicit bias; 1 hr Alzheimer's/dementia if providing care to adults 26+; all count toward the 20 hours").
   - sources: All URLs in the answer that support Illinois RN renewal/CE details.

2) ca_director (California Child Care Center Director Qualifications):
   - pathways: A list of all director qualification pathways enumerated in the answer (each pathway as a single descriptive string).
   - pathway1_unit_breakdown: The stated unit breakdown for the HS/GED + 15 units pathway (e.g., "3 units admin/staff relations + 12 units across child growth/development; child/family/community; program/curriculum").
   - experience_standard: The stated standard for verifying each year of teaching experience (e.g., "at least 3 hours/day for a minimum of 100 days per calendar year under qualified supervision").
   - sources: All URLs in the answer that support California director qualification requirements.

3) ca_preapp (California Licensing Pre-Application Requirements):
   - orientation_requirement: The stated rule that orientation must be completed before submitting an application (and that applications are not accepted before orientation).
   - orientation_fee_online: The stated online orientation fee (ideally including the breakdown, e.g., "$54.85 ($50 + $4.85 processing)").
   - orientation_fee_inperson: The stated in-person orientation fee (e.g., "$50").
   - orientation_nonrefundable_policy: The stated policy that orientation fees are non-refundable.
   - sources: All URLs in the answer that support the California orientation requirement and fees.

4) cost (Cost Calculation):
   - birth_cert_first_fee: The stated fee for the first certified copy of a Kankakee County, Illinois birth certificate (e.g., "$10").
   - birth_cert_additional_fee: The stated fee for each additional copy ordered at the same time (e.g., "$4").
   - total_online_plus_birth: The stated total cost for (online orientation + one certified birth certificate).
   - total_inperson_plus_birth: The stated total cost for (in-person orientation + one certified birth certificate).
   - sources_birth_cert: All URLs in the answer that support Kankakee County birth certificate fees.

Rules:
- Extract only what is explicitly stated in the answer.
- For URLs, include only actual URLs. If a URL is missing but a source is described without a link, do not invent a URL; leave it out.
- Preserve the text as stated where reasonable (e.g., keep "$54.85 ($50 + $4.85 processing)" if presented).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_text(x: Optional[str]) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Basic cleanup and de-duplication
    seen = set()
    clean: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u2 = u.strip()
        if not u2:
            continue
        # If protocol missing, Extractor SPECIAL RULES may add it; still normalize here
        if not (u2.startswith("http://") or u2.startswith("https://")):
            u2 = "http://" + u2
        if u2 not in seen:
            clean.append(u2)
            seen.add(u2)
    return clean


def _extract_amounts(val: str) -> List[float]:
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", val.replace(",", ""))
    try:
        return [float(n) for n in nums]
    except Exception:
        return []


def _parse_primary_amount(val: Optional[str]) -> Optional[float]:
    """Return the maximum numeric amount found (useful when a string includes a breakdown, e.g., '$50 + $4.85 = $54.85')."""
    if not _has_text(val):
        return None
    amounts = _extract_amounts(val or "")
    if not amounts:
        return None
    return max(amounts)


def _format_money(v: float) -> str:
    return f"${v:,.2f}"


async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    node,
    claim: str,
    urls: Optional[List[str]],
    additional_instruction: Optional[str] = None
) -> None:
    cleaned = _sanitize_urls(urls)
    if cleaned:
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=cleaned,
            additional_instruction=additional_instruction or "None",
        )
    else:
        # Missing sources – fail the leaf to enforce source-grounding
        node.score = 0.0
        node.status = "failed"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_il_rn_maintenance(
    evaluator: Evaluator,
    parent,
    il: Optional[ILRNMaintenance]
) -> None:
    il_node = evaluator.add_parallel(
        id="Illinois_RN_License_Maintenance",
        desc="Illinois RN renewal deadline and continuing education requirements for the renewal cycle ending May 31, 2026",
        parent=parent,
        critical=True
    )

    il_sources = _sanitize_urls(il.sources if il else [])

    # Renewal deadline as May 31, 2026
    leaf_deadline = evaluator.add_leaf(
        id="Renewal_Deadline_Date",
        desc="States the Illinois RN license renewal deadline as May 31, 2026",
        parent=il_node,
        critical=True
    )
    if not il or not _has_text(il.renewal_deadline):
        leaf_deadline.score = 0.0
        leaf_deadline.status = "failed"
    else:
        claim = f"The Illinois RN license renewal deadline in 2026 is {il.renewal_deadline}."
        add_ins = "Evidence such as 'RN licenses expire on May 31 of even-numbered years' supports May 31, 2026."
        await _verify_with_urls_or_fail(evaluator, leaf_deadline, claim, il_sources, add_ins)

    # Renewal cycle: every 2 years
    leaf_cycle = evaluator.add_leaf(
        id="Renewal_Cycle",
        desc="States that Illinois RN licenses renew every 2 years",
        parent=il_node,
        critical=True
    )
    if not il or not _has_text(il.renewal_cycle_years):
        leaf_cycle.score = 0.0
        leaf_cycle.status = "failed"
    else:
        claim = f"Illinois RN licenses renew every {il.renewal_cycle_years}."
        add_ins = "Allow equivalent phrasing such as 'biennially' or 'every two years'."
        await _verify_with_urls_or_fail(evaluator, leaf_cycle, claim, il_sources, add_ins)

    # CE total hours: exactly 20 contact hours
    leaf_ce_hours = evaluator.add_leaf(
        id="CE_Total_Hours",
        desc="States that exactly 20 contact hours of approved continuing education are required",
        parent=il_node,
        critical=True
    )
    if not il or not _has_text(il.ce_total_hours):
        leaf_ce_hours.score = 0.0
        leaf_ce_hours.status = "failed"
    else:
        claim = f"Exactly {il.ce_total_hours} of approved continuing education are required to renew an Illinois RN license."
        add_ins = "Verify that the CE requirement is 20 contact hours total per renewal cycle."
        await _verify_with_urls_or_fail(evaluator, leaf_ce_hours, claim, il_sources, add_ins)

    # CE timeframe: within the 24 months preceding license expiration
    leaf_ce_timeframe = evaluator.add_leaf(
        id="CE_Timeframe",
        desc="States that the CE must be completed within the 24 months preceding license expiration",
        parent=il_node,
        critical=True
    )
    if not il or not _has_text(il.ce_timeframe):
        leaf_ce_timeframe.score = 0.0
        leaf_ce_timeframe.status = "failed"
    else:
        claim = "The continuing education must be completed within the 24 months preceding license expiration."
        add_ins = "Allow equivalent phrasing such as 'within the two years before the license expires'."
        await _verify_with_urls_or_fail(evaluator, leaf_ce_timeframe, claim, il_sources, add_ins)

    # Mandatory CE topic breakdown
    leaf_ce_topics = evaluator.add_leaf(
        id="Mandatory_CE_Topic_Breakdown",
        desc="Includes the full mandatory CE topic breakdown within the 20 hours: 1 hour sexual harassment prevention, 1 hour implicit bias awareness, and 1 hour Alzheimer's/dementia care if providing care to adults aged 26 or older (all counting toward the 20 hours)",
        parent=il_node,
        critical=True
    )
    if not il or not _has_text(il.ce_mandatory_breakdown):
        leaf_ce_topics.score = 0.0
        leaf_ce_topics.status = "failed"
    else:
        claim = ("Within the 20 hours, Illinois requires: "
                 "1 hour sexual harassment prevention training; "
                 "1 hour implicit bias awareness training; and "
                 "1 hour Alzheimer's disease and other dementias training if providing care to adults aged 26 or older; "
                 "all count toward the 20 hours.")
        add_ins = "Verify each topic requirement and that these hours count toward the total 20 hours."
        await _verify_with_urls_or_fail(evaluator, leaf_ce_topics, claim, il_sources, add_ins)


async def build_ca_director_qualifications(
    evaluator: Evaluator,
    parent,
    ca_dir: Optional[CADirectorQualifications]
) -> None:
    ca_node = evaluator.add_parallel(
        id="California_Child_Care_Center_Director_Qualifications",
        desc="California child care center director qualification pathways, unit requirements, and experience verification standards",
        parent=parent,
        critical=True
    )

    ca_sources = _sanitize_urls(ca_dir.sources if ca_dir else [])

    # All four qualification pathways
    leaf_four_paths = evaluator.add_leaf(
        id="All_Four_Qualification_Pathways",
        desc="Identifies all four director qualification pathways as specified in the constraints",
        parent=ca_node,
        critical=True
    )
    if not ca_dir or len(ca_dir.pathways) < 4:
        leaf_four_paths.score = 0.0
        leaf_four_paths.status = "failed"
    else:
        listed = "; ".join(ca_dir.pathways[:4])
        claim = f"The California child care center director may qualify under the following four pathways: {listed}. This set matches the four regulatory pathways."
        add_ins = "Verify that exactly four qualification options are recognized and that the listed descriptions align with CA Title 22/CDSS guidance."
        await _verify_with_urls_or_fail(evaluator, leaf_four_paths, claim, ca_sources, add_ins)

    # Pathway 1 unit breakdown
    leaf_units = evaluator.add_leaf(
        id="Pathway_1_Unit_Breakdown",
        desc="For the High school/GED + 15 semester unit pathway: states that exactly 3 units are in administration or staff relations and the remaining 12 units cover child growth/development, child/family/community, and program/curriculum",
        parent=ca_node,
        critical=True
    )
    if not ca_dir or not _has_text(ca_dir.pathway1_unit_breakdown):
        leaf_units.score = 0.0
        leaf_units.status = "failed"
    else:
        claim = ("For the high school diploma or GED plus 15 semester units pathway: "
                 "exactly 3 units must be in administration or staff relations, and the remaining 12 units must cover "
                 "child growth and development; child, family, and community; and program/curriculum.")
        add_ins = "Match the specific distribution: 3 admin/staff-relations units + 12 ECE/CD units across the three specified content areas."
        await _verify_with_urls_or_fail(evaluator, leaf_units, claim, ca_sources, add_ins)

    # Experience verification standard
    leaf_experience = evaluator.add_leaf(
        id="Experience_Verification_Standard",
        desc="States the full required standard for verifying each year of teaching experience: at least 3 hours/day for a minimum of 100 days per calendar year, performed as a teacher under qualified supervision",
        parent=ca_node,
        critical=True
    )
    if not ca_dir or not _has_text(ca_dir.experience_standard):
        leaf_experience.score = 0.0
        leaf_experience.status = "failed"
    else:
        claim = ("One year of teaching experience is defined as at least 3 hours per day for a minimum of 100 days "
                 "in a calendar year, performed as a teacher under qualified supervision.")
        add_ins = "This is the California regulatory definition commonly used for experience verification."
        await _verify_with_urls_or_fail(evaluator, leaf_experience, claim, ca_sources, add_ins)


async def build_ca_pre_application(
    evaluator: Evaluator,
    parent,
    ca_pre: Optional[CAPreApplication]
) -> None:
    pre_node = evaluator.add_parallel(
        id="California_Licensing_Pre_Application_Requirements",
        desc="Mandatory steps before submitting a California child care center license application and associated fees",
        parent=parent,
        critical=True
    )

    pre_sources = _sanitize_urls(ca_pre.sources if ca_pre else [])

    # Orientation pre-application requirement
    leaf_pre_req = evaluator.add_leaf(
        id="Orientation_PreApplication_Requirement",
        desc="States that applicants must complete a mandatory orientation before submitting a license application AND that the licensing office will not accept applications until after orientation is completed",
        parent=pre_node,
        critical=True
    )
    if not ca_pre or not _has_text(ca_pre.orientation_requirement):
        leaf_pre_req.score = 0.0
        leaf_pre_req.status = "failed"
    else:
        claim = ("Applicants must complete the required orientation before submitting a license application, and the "
                 "licensing office will not accept applications until orientation is completed.")
        add_ins = "Confirm both requirements: completion before submission and non-acceptance of applications prior to orientation."
        await _verify_with_urls_or_fail(evaluator, leaf_pre_req, claim, pre_sources, add_ins)

    # Orientation fees and policies
    leaf_fees = evaluator.add_leaf(
        id="Orientation_Fees_And_Policies",
        desc="States the orientation fees for both options (online $54.85 including $50 + $4.85 processing; in-person $50) and that these fees are non-refundable",
        parent=pre_node,
        critical=True
    )
    if not ca_pre or not (_has_text(ca_pre.orientation_fee_online) and _has_text(ca_pre.orientation_fee_inperson) and _has_text(ca_pre.orientation_nonrefundable_policy)):
        leaf_fees.score = 0.0
        leaf_fees.status = "failed"
    else:
        claim = ("Orientation fees are: online $54.85 (comprising a $50 base fee plus a $4.85 processing fee) and in-person $50; "
                 "orientation fees are non-refundable.")
        add_ins = "Verify the exact dollar amounts for both options, the $50 + $4.85 breakdown for online, and the non-refundable policy."
        await _verify_with_urls_or_fail(evaluator, leaf_fees, claim, pre_sources, add_ins)


async def build_costs(
    evaluator: Evaluator,
    parent,
    ca_pre: Optional[CAPreApplication],
    cost: Optional[CostCalculation]
) -> None:
    cost_node = evaluator.add_parallel(
        id="Cost_Calculation",
        desc="Computes the total cost for mandatory orientation (both online and in-person options) and one certified birth certificate from Kankakee County, Illinois",
        parent=parent,
        critical=True
    )

    # Birth certificate fees verification (URL-grounded)
    leaf_bc_fees = evaluator.add_leaf(
        id="Birth_Certificate_Fees",
        desc="States Kankakee County certified birth certificate fees: $10 for the first copy and $4 for each additional copy ordered at the same time",
        parent=cost_node,
        critical=True
    )
    bc_sources = _sanitize_urls(cost.sources_birth_cert if cost else [])
    if not cost or not (_has_text(cost.birth_cert_first_fee) and _has_text(cost.birth_cert_additional_fee)):
        leaf_bc_fees.score = 0.0
        leaf_bc_fees.status = "failed"
    else:
        claim = ("Kankakee County certified birth certificate fees are $10 for the first copy and $4 for each additional copy "
                 "ordered at the same time.")
        add_ins = "Confirm both the first-copy fee and the same-order additional-copy fee."
        await _verify_with_urls_or_fail(evaluator, leaf_bc_fees, claim, bc_sources, add_ins)

    # Total costs arithmetic verification (simple logic check, but require stated totals)
    leaf_totals = evaluator.add_leaf(
        id="Total_Costs_For_Both_Orientation_Options",
        desc="Provides correct total cost calculations for BOTH (a) online orientation + one certified birth certificate and (b) in-person orientation + one certified birth certificate, using the stated fees (totals are arithmetically correct)",
        parent=cost_node,
        critical=True
    )

    # We require that the answer stated totals; if missing, fail this leaf
    if (
        not ca_pre or
        not cost or
        not (_has_text(cost.total_online_plus_birth) and _has_text(cost.total_inperson_plus_birth) and
             _has_text(ca_pre.orientation_fee_online) and _has_text(ca_pre.orientation_fee_inperson) and
             _has_text(cost.birth_cert_first_fee))
    ):
        leaf_totals.score = 0.0
        leaf_totals.status = "failed"
    else:
        online_fee = _parse_primary_amount(ca_pre.orientation_fee_online)
        inperson_fee = _parse_primary_amount(ca_pre.orientation_fee_inperson)
        birth_first = _parse_primary_amount(cost.birth_cert_first_fee)
        stated_online_total = _parse_primary_amount(cost.total_online_plus_birth)
        stated_inperson_total = _parse_primary_amount(cost.total_inperson_plus_birth)

        if online_fee is None or inperson_fee is None or birth_first is None or \
                stated_online_total is None or stated_inperson_total is None:
            leaf_totals.score = 0.0
            leaf_totals.status = "failed"
        else:
            expected_online = online_fee + birth_first
            expected_inperson = inperson_fee + birth_first

            claim = (
                f"In the answer, the total cost for (a) online orientation + one Kankakee certified birth certificate is stated as "
                f"{_format_money(stated_online_total)}, and for (b) in-person orientation + one Kankakee certified birth certificate is stated as "
                f"{_format_money(stated_inperson_total)}. Given the component fees in the answer (online orientation {_format_money(online_fee)}, "
                f"in-person orientation {_format_money(inperson_fee)}, and birth certificate first copy {_format_money(birth_first)}), "
                f"the correct totals are {_format_money(expected_online)} and {_format_money(expected_inperson)}, respectively. "
                f"Both stated totals are arithmetically correct."
            )
            add_ins = "Check that each stated total equals the sum of the respective orientation fee plus the birth certificate first-copy fee."
            # Simple logical verification (no URLs needed here because underlying fees are validated in other leaves)
            await evaluator.verify(claim=claim, node=leaf_totals, sources=None, additional_instruction=add_ins)


def build_reference_urls_check(
    evaluator: Evaluator,
    parent,
    il: Optional[ILRNMaintenance],
    ca_dir: Optional[CADirectorQualifications],
    ca_pre: Optional[CAPreApplication],
    cost: Optional[CostCalculation]
) -> None:
    # Single custom node per rubric to ensure at least one URL per area is provided in the answer
    has_il = bool(il and _sanitize_urls(il.sources))
    has_ca_dir = bool(ca_dir and _sanitize_urls(ca_dir.sources))
    has_ca_pre = bool(ca_pre and _sanitize_urls(ca_pre.sources))
    has_bc = bool(cost and _sanitize_urls(cost.sources_birth_cert))

    ok_all = has_il and has_ca_dir and has_ca_pre and has_bc

    evaluator.add_custom_node(
        result=ok_all,
        id="Reference_URLs_For_Verification",
        desc="Provides reference URL(s) that support the key factual/numeric requirements across all four areas (Illinois renewal/CE, CA director qualifications, CA orientation requirement/fees, and Kankakee birth certificate fees)",
        parent=parent,
        critical=True
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
    # Initialize evaluator (root is critical parallel per rubric)
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
    # Root is non-critical by default in Evaluator.initialize, but rubric requires critical.
    # We wrap children under a critical parallel node to align with rubric while respecting framework constraints.
    root_critical = evaluator.add_parallel(
        id="Root",
        desc="Comprehensive response covering Illinois RN renewal/CE, California child care center director qualifications, California pre-application requirements, required cost calculations, and reference URLs",
        parent=root,
        critical=True
    )

    # Extract structured information
    extraction: MasterExtraction = await evaluator.extract(
        prompt=prompt_extract_master(),
        template_class=MasterExtraction,
        extraction_name="structured_extraction"
    )

    il = extraction.il or ILRNMaintenance()
    ca_dir = extraction.ca_director or CADirectorQualifications()
    ca_pre = extraction.ca_preapp or CAPreApplication()
    cost = extraction.cost or CostCalculation()

    # Build subtrees
    await build_il_rn_maintenance(evaluator, root_critical, il)
    await build_ca_director_qualifications(evaluator, root_critical, ca_dir)
    await build_ca_pre_application(evaluator, root_critical, ca_pre)
    await build_costs(evaluator, root_critical, ca_pre, cost)

    # Reference URLs existence check across all four areas (single critical leaf as per rubric)
    build_reference_urls_check(evaluator, root_critical, il, ca_dir, ca_pre, cost)

    # Return summary
    return evaluator.get_summary()