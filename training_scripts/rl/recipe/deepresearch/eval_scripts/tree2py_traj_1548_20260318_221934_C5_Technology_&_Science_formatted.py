import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "verizon_outage_2026_jan14"
TASK_DESCRIPTION = (
    "On January 14, 2026, a major U.S. wireless carrier experienced a significant nationwide network outage that left "
    "millions of customers without service for several hours. Please provide comprehensive information about this outage "
    "incident, including: (1) The exact date of the outage and which wireless carrier was affected; (2) The specific start time, "
    "official resolution time, and total duration of the outage; (3) The technical cause of the outage, including which network "
    "component was affected; (4) How customers were impacted, including what their phones displayed, which services were "
    "disrupted, and the geographic scope of the outage; (5) Details about the compensation offered to affected customers, "
    "including the amount and how customers can claim it; (6) Information about the federal regulatory response, including which "
    "FCC bureau launched an investigation, the deadline for public comments, and the email address where customers can submit "
    "their experiences. Provide URL references from official or reputable news sources to support each category of information."
)

# Expected target facts for checks
EXPECTED_DATE = "January 14, 2026"
EXPECTED_CARRIER = "Verizon"
EXPECTED_START_TIME = "12:30 PM ET"
EXPECTED_RESOLUTION_TIME = "10:15 PM ET on January 14, 2026"
EXPECTED_DURATION_APPROX = "around 10 hours"
EXPECTED_CAUSE_KEYWORD = "software"
EXPECTED_COMPONENT = "5G Standalone (5G SA) core"
EXPECTED_SOS_DISPLAY = "SOS"
EXPECTED_SERVICES = ["voice calls", "text messages", "mobile data"]
EXPECTED_SCOPE = "nationwide across the United States"
EXPECTED_CREDIT_AMOUNT = "$20"
EXPECTED_CREDIT_METHOD_HINTS = ["myVerizon app", "text"]
EXPECTED_FCC_BUREAU = "Public Safety and Homeland Security Bureau"
EXPECTED_FCC_DOCKET = "26-21"
EXPECTED_COMMENT_DEADLINE = "March 16, 2026"
EXPECTED_COMMENT_EMAIL = "VerizonOutage2026@fcc.gov"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OutageBasicFacts(BaseModel):
    date: Optional[str] = None
    carrier: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TimelineInfo(BaseModel):
    start_time: Optional[str] = None
    resolution_time: Optional[str] = None
    duration: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TechnicalCause(BaseModel):
    cause: Optional[str] = None  # e.g., "software issue"
    component: Optional[str] = None  # e.g., "5G SA core"
    sources: List[str] = Field(default_factory=list)


class CustomerImpact(BaseModel):
    phone_display: Optional[str] = None  # e.g., "SOS only"
    services_disrupted: List[str] = Field(default_factory=list)  # e.g., ["voice calls", "text", "data"]
    geographic_scope: Optional[str] = None  # e.g., "nationwide"
    sources: List[str] = Field(default_factory=list)


class CompensationDetails(BaseModel):
    credit_amount: Optional[str] = None  # e.g., "$20"
    redemption_method: Optional[str] = None  # e.g., "redeem in myVerizon app after text notification"
    sources: List[str] = Field(default_factory=list)


class FCCResponse(BaseModel):
    bureau: Optional[str] = None  # e.g., "Public Safety and Homeland Security Bureau"
    docket_number: Optional[str] = None  # e.g., "26-21"
    comment_deadline: Optional[str] = None  # e.g., "March 16, 2026"
    comment_email: Optional[str] = None  # e.g., "VerizonOutage2026@fcc.gov"
    sources: List[str] = Field(default_factory=list)


class OutageExtraction(BaseModel):
    basic_facts: Optional[OutageBasicFacts] = None
    timeline: Optional[TimelineInfo] = None
    technical_cause: Optional[TechnicalCause] = None
    customer_impact: Optional[CustomerImpact] = None
    compensation: Optional[CompensationDetails] = None
    fcc_response: Optional[FCCResponse] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage() -> str:
    return """
    Extract structured information about the January 2026 U.S. wireless outage from the answer EXACTLY as stated.
    Return null for any field not explicitly present in the answer text.

    Fields to extract:

    basic_facts:
      - date: the stated date of the outage (e.g., "January 14, 2026")
      - carrier: the stated wireless carrier (e.g., "Verizon", "Verizon Wireless", "Verizon Communications")
      - sources: an array of URL(s) explicitly cited in the answer that support these basic facts

    timeline:
      - start_time: the stated outage start time, including timezone if provided (e.g., "12:30 PM ET")
      - resolution_time: the stated official resolution time with date/time/zone if provided
      - duration: the stated total duration (e.g., "about 10 hours")
      - sources: URL(s) cited in the answer that support timeline details

    technical_cause:
      - cause: the stated root cause (e.g., "software issue")
      - component: the affected network component (e.g., "5G Standalone (5G SA) core")
      - sources: URL(s) cited in the answer that support technical cause

    customer_impact:
      - phone_display: what phones showed (e.g., "SOS", "SOS only", "Emergency SOS")
      - services_disrupted: array of services disrupted (e.g., ["voice calls", "text messages", "mobile data"])
      - geographic_scope: the stated scope (e.g., "nationwide across the United States")
      - sources: URL(s) cited in the answer that support impact details

    compensation:
      - credit_amount: the stated compensation (e.g., "$20")
      - redemption_method: how customers redeem/receive it (e.g., "redeem via myVerizon app after text notification")
      - sources: URL(s) cited in the answer that support compensation info

    fcc_response:
      - bureau: which FCC bureau launched an investigation (e.g., "Public Safety and Homeland Security Bureau")
      - docket_number: the docket number (e.g., "26-21")
      - comment_deadline: the stated public comment deadline date
      - comment_email: the stated email address for comments
      - sources: URL(s) cited in the answer that support FCC response details

    SPECIAL URL RULES:
    - Extract only URLs that are explicitly present in the answer.
    - Accept plain URLs or markdown links; output the actual URL strings.
    - If a URL is missing a protocol, prepend http://.
    - If no URLs are provided for a section, return an empty array for that section's sources.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _nonempty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


# --------------------------------------------------------------------------- #
# Verification functions per rubric subtree                                   #
# --------------------------------------------------------------------------- #
async def verify_basic_facts(evaluator: Evaluator, parent) -> None:
    node = evaluator.add_parallel(
        id="basic_facts",
        desc="Core identifying information about the outage event",
        parent=parent,
        critical=True  # Basic facts are mandatory
    )

    data: OutageExtraction = evaluator.find_node("root")  # just to clarify type hints; will use captured extraction below
    # We'll read from the recorded extraction later (via closure or external variable).
    # Instead, pass the extracted object in outer scope and capture it via default arg.
    pass


# We'll implement with an explicit parameter carrying the extracted data in each function
async def verify_basic_facts_with_data(evaluator: Evaluator, parent, ex: OutageExtraction) -> None:
    node = evaluator.add_parallel(
        id="outage_basic_facts",
        desc="Core identifying information about the outage event",
        parent=parent,
        critical=True
    )

    # Extracted fields
    bf = ex.basic_facts or OutageBasicFacts()

    # Outage Date (simple equality to expected)
    date_node = evaluator.add_leaf(
        id="outage_date",
        desc="The outage occurred on January 14, 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The stated outage date "{(bf.date or "").strip()}" equals "{EXPECTED_DATE}" allowing minor format variants.',
        node=date_node,
        additional_instruction="Treat formats like 'Jan 14, 2026' or '2026-01-14' as equivalent to 'January 14, 2026'. If the value is missing or clearly different, mark incorrect."
    )

    # Carrier Identity (should be Verizon)
    carrier_node = evaluator.add_leaf(
        id="carrier_identity",
        desc="The outage affected Verizon Wireless/Verizon Communications",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The stated carrier "{(bf.carrier or "").strip()}" refers to Verizon (acceptable: "Verizon", "Verizon Wireless", or "Verizon Communications").',
        node=carrier_node,
        additional_instruction="Allow common variants like 'Verizon', 'Verizon Wireless', or 'Verizon Communications'. If it refers to a different carrier (e.g., AT&T, T-Mobile), mark incorrect."
    )

    # References: existence + supported-by-URLs
    ref_exist = evaluator.add_custom_node(
        result=_nonempty_urls(bf.sources),
        id="basic_facts_reference_exists",
        desc="Basic facts reference(s) are provided (at least one URL)",
        parent=node,
        critical=True
    )

    ref_supported = evaluator.add_leaf(
        id="basic_facts_reference_supported",
        desc="Provides valid URL reference supporting the basic facts",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'These sources report that a major outage occurred on {EXPECTED_DATE} and affected Verizon.',
        node=ref_supported,
        sources=bf.sources,
        additional_instruction="Only pass if at least one page explicitly mentions the outage date and that Verizon was the affected carrier. If URLs are irrelevant or inaccessible, mark incorrect."
    )


async def verify_timeline(evaluator: Evaluator, parent, ex: OutageExtraction) -> None:
    node = evaluator.add_parallel(
        id="timeline_information",
        desc="Specific timing details of the outage duration",
        parent=parent,
        critical=False
    )

    tl = ex.timeline or TimelineInfo()

    # Start Time (approx around 12:30 PM ET)
    start_node = evaluator.add_leaf(
        id="start_time",
        desc="Reports the outage start time around 12:30 PM Eastern Time",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f'The stated outage start time "{(tl.start_time or "").strip()}" is reasonably close to {EXPECTED_START_TIME} (±30 minutes).',
        node=start_node,
        additional_instruction="Treat within ~30 minutes as acceptable (e.g., 12:15–12:45 PM ET). If missing or far off, mark incorrect."
    )

    # Resolution Time (official resolution at 10:15 PM ET Jan 14, 2026)
    res_node = evaluator.add_leaf(
        id="resolution_time",
        desc="Reports the official resolution time at 10:15 PM ET on January 14, 2026",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f'The stated official resolution time "{(tl.resolution_time or "").strip()}" equals {EXPECTED_RESOLUTION_TIME} allowing minor format variants.',
        node=res_node,
        additional_instruction="Accept equivalent formats/time zone notations as long as it clearly corresponds to 10:15 PM ET on Jan 14, 2026."
    )

    # Duration (~10 hours)
    dur_node = evaluator.add_leaf(
        id="duration",
        desc="Indicates the outage lasted approximately 10 hours",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f'The stated total outage duration "{(tl.duration or "").strip()}" is approximately {EXPECTED_DURATION_APPROX} (tolerance ~±2 hours).',
        node=dur_node,
        additional_instruction="Interpret ranges like 'about 10 hours' or 'nearly 10 hours' as acceptable. Explicit durations 8–12 hours are acceptable approximations. If missing, mark incorrect."
    )

    # References: existence + supported-by-URL
    ref_exist = evaluator.add_custom_node(
        result=_nonempty_urls(tl.sources),
        id="timeline_reference_exists",
        desc="Timeline reference(s) are provided (at least one URL)",
        parent=node,
        critical=True
    )

    ref_supported = evaluator.add_leaf(
        id="timeline_reference_supported",
        desc="Provides valid URL reference supporting timeline details",
        parent=node,
        critical=True
    )
    # Build a composite claim using expected anchors (to avoid endorsing wrong values)
    await evaluator.verify(
        claim=f"These sources report that the outage started around {EXPECTED_START_TIME}, was officially resolved at {EXPECTED_RESOLUTION_TIME}, and lasted {EXPECTED_DURATION_APPROX}.",
        node=ref_supported,
        sources=tl.sources,
        additional_instruction="At least one source must clearly state each of start time (around 12:30 PM ET), official resolution time (10:15 PM ET 1/14/2026), and an overall duration of roughly 10 hours. If any of these cannot be supported, mark incorrect."
    )


async def verify_technical_cause(evaluator: Evaluator, parent, ex: OutageExtraction) -> None:
    node = evaluator.add_parallel(
        id="technical_cause",
        desc="Information about what caused the outage",
        parent=parent,
        critical=False
    )

    tc = ex.technical_cause or TechnicalCause()

    # Software Issue
    sw_node = evaluator.add_leaf(
        id="software_issue",
        desc="Identifies the cause as a software issue",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f'The stated cause "{(tc.cause or "").strip()}" indicates a software issue (allow synonyms like software bug, software error, software defect).',
        node=sw_node,
        additional_instruction="If the phrasing clearly points to software (not hardware, power, or fiber cuts), consider it correct."
    )

    # 5G SA Core
    core_node = evaluator.add_leaf(
        id="fiveg_sa_core",
        desc="Specifies the issue was in the 5G Standalone (5G SA) core network",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f'The stated affected component "{(tc.component or "").strip()}" refers to the 5G Standalone (5G SA) core network.',
        node=core_node,
        additional_instruction="Allow reasonable paraphrases like '5G SA core', '5G standalone core network', or '5G core (standalone)'."
    )

    # References
    ref_exist = evaluator.add_custom_node(
        result=_nonempty_urls(tc.sources),
        id="technical_cause_reference_exists",
        desc="Technical cause reference(s) are provided (at least one URL)",
        parent=node,
        critical=True
    )

    ref_supported = evaluator.add_leaf(
        id="technical_cause_reference_supported",
        desc="Provides valid URL reference supporting technical cause information",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"These sources report that the outage was caused by a software issue in the {EXPECTED_COMPONENT}.",
        node=ref_supported,
        sources=tc.sources,
        additional_instruction="Only pass if at least one source clearly states both 'software issue' and the 5G SA core context."
    )


async def verify_customer_impact(evaluator: Evaluator, parent, ex: OutageExtraction) -> None:
    node = evaluator.add_parallel(
        id="customer_impact",
        desc="Details about how customers were affected",
        parent=parent,
        critical=False
    )

    ci = ex.customer_impact or CustomerImpact()

    # SOS Mode
    sos_node = evaluator.add_leaf(
        id="sos_mode",
        desc="Reports that affected phones displayed 'SOS mode' or 'SOS only'",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f'The stated phone display "{(ci.phone_display or "").strip()}" indicates SOS/SOS only (or similar, e.g., "Emergency SOS").',
        node=sos_node,
        additional_instruction="Accept 'SOS', 'SOS only', 'Emergency SOS', or similar wording. If missing or unrelated, mark incorrect."
    )

    # Services Affected
    services_node = evaluator.add_leaf(
        id="services_affected",
        desc="Identifies that voice calls, text messages, and mobile data were disrupted",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The stated disrupted services {ci.services_disrupted} include all of: voice calls, text/SMS, and mobile data/internet.",
        node=services_node,
        additional_instruction="Consider synonyms (e.g., 'texts' for 'text messages', 'cellular data'/'mobile internet' for 'mobile data'). All three categories should be present."
    )

    # Geographic Scope
    geo_node = evaluator.add_leaf(
        id="geographic_scope",
        desc="Indicates the outage was nationwide across the United States",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f'The stated geographic scope "{(ci.geographic_scope or "").strip()}" indicates a nationwide U.S. outage.',
        node=geo_node,
        additional_instruction="Accept phrasing like 'nationwide', 'across the U.S.', 'across the United States'. If it suggests a limited region only, mark incorrect."
    )

    # References
    ref_exist = evaluator.add_custom_node(
        result=_nonempty_urls(ci.sources),
        id="customer_impact_reference_exists",
        desc="Customer impact reference(s) are provided (at least one URL)",
        parent=node,
        critical=True
    )

    ref_supported = evaluator.add_leaf(
        id="customer_impact_reference_supported",
        desc="Provides valid URL reference supporting customer impact details",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="These sources report that many affected phones showed 'SOS' or 'SOS only', that voice calls, text/SMS, and mobile data were disrupted, and that the outage was nationwide across the U.S.",
        node=ref_supported,
        sources=ci.sources,
        additional_instruction="At least one credible page should explicitly support each of the following: SOS display, disruption of calls/texts/data, and nationwide scope."
    )


async def verify_compensation(evaluator: Evaluator, parent, ex: OutageExtraction) -> None:
    node = evaluator.add_parallel(
        id="compensation_details",
        desc="Information about compensation offered to affected customers",
        parent=parent,
        critical=False
    )

    comp = ex.compensation or CompensationDetails()

    # Credit Amount
    amount_node = evaluator.add_leaf(
        id="credit_amount",
        desc="Reports the $20 account credit offered as compensation",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f'The stated compensation amount "{(comp.credit_amount or "").strip()}" equals {EXPECTED_CREDIT_AMOUNT} (allow variants like "20 dollars").',
        node=amount_node,
        additional_instruction="Minor format differences allowed (e.g., '20 USD', 'a $20 credit'). If different amount or missing, mark incorrect."
    )

    # Credit Distribution / How to redeem
    how_node = evaluator.add_leaf(
        id="credit_distribution",
        desc="Explains the credit is redeemable through the myVerizon app after text notification",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f'The stated redemption method "{(comp.redemption_method or "").strip()}" indicates customers receive a text and can redeem in the myVerizon app.',
        node=how_node,
        additional_instruction="Look for both 'text/SMS notification' and 'myVerizon app' cues in the phrasing. If one is missing or different method described, mark incorrect."
    )

    # References
    ref_exist = evaluator.add_custom_node(
        result=_nonempty_urls(comp.sources),
        id="compensation_reference_exists",
        desc="Compensation reference(s) are provided (at least one URL)",
        parent=node,
        critical=True
    )

    ref_supported = evaluator.add_leaf(
        id="compensation_reference_supported",
        desc="Provides valid URL reference supporting compensation information",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"These sources report that Verizon offered a {EXPECTED_CREDIT_AMOUNT} account credit and that customers can redeem it in the myVerizon app after receiving a text message or instructions.",
        node=ref_supported,
        sources=comp.sources,
        additional_instruction="At least one page should clearly state both the $20 amount and the redemption/notification method."
    )


async def verify_fcc(evaluator: Evaluator, parent, ex: OutageExtraction) -> None:
    node = evaluator.add_parallel(
        id="fcc_investigation",
        desc="Information about the Federal Communications Commission's response",
        parent=parent,
        critical=False
    )

    fcc = ex.fcc_response or FCCResponse()

    # FCC Bureau
    bureau_node = evaluator.add_leaf(
        id="fcc_bureau",
        desc="Identifies the FCC's Public Safety and Homeland Security Bureau launched an investigation",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f'The stated bureau "{(fcc.bureau or "").strip()}" is the FCC Public Safety and Homeland Security Bureau.',
        node=bureau_node,
        additional_instruction="Accept 'Public Safety and Homeland Security Bureau' or common abbreviation 'PSHSB'."
    )

    # Docket Number
    docket_node = evaluator.add_leaf(
        id="fcc_docket_number",
        desc="Reports the FCC docket number 26-21 for the investigation",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f'The stated docket number "{(fcc.docket_number or "").strip()}" equals "{EXPECTED_FCC_DOCKET}".',
        node=docket_node,
        additional_instruction="Treat '26–21' (en dash) as equivalent to '26-21'."
    )

    # Comment Deadline
    deadline_node = evaluator.add_leaf(
        id="comment_deadline",
        desc="Reports the public comment deadline of March 16, 2026",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f'The stated public comment deadline "{(fcc.comment_deadline or "").strip()}" equals "{EXPECTED_COMMENT_DEADLINE}" (format variants allowed).',
        node=deadline_node,
        additional_instruction="Accept equivalent formats like '2026-03-16' or 'Mar 16, 2026'."
    )

    # Comment Email
    email_node = evaluator.add_leaf(
        id="comment_email",
        desc="Provides the FCC email address VerizonOutage2026@fcc.gov for public comments",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f'The stated email "{(fcc.comment_email or "").strip()}" equals "{EXPECTED_COMMENT_EMAIL}" (case-insensitive).',
        node=email_node,
        additional_instruction="Compare case-insensitively. Minor punctuation surrounding the address should be ignored."
    )

    # References
    ref_exist = evaluator.add_custom_node(
        result=_nonempty_urls(fcc.sources),
        id="fcc_investigation_reference_exists",
        desc="FCC investigation reference(s) are provided (at least one URL)",
        parent=node,
        critical=True
    )

    ref_supported = evaluator.add_leaf(
        id="fcc_investigation_reference_supported",
        desc="Provides valid URL reference supporting FCC investigation details",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"These sources report that the FCC's {EXPECTED_FCC_BUREAU} launched an investigation under docket {EXPECTED_FCC_DOCKET}, that public comments are due by {EXPECTED_COMMENT_DEADLINE}, and that comments can be submitted via email to {EXPECTED_COMMENT_EMAIL}.",
        node=ref_supported,
        sources=fcc.sources,
        additional_instruction="At least one source should explicitly mention the bureau, docket number, deadline, and email. If any are missing or contradicted, mark incorrect."
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

    # Extract structured info
    extracted: OutageExtraction = await evaluator.extract(
        prompt=prompt_extract_outage(),
        template_class=OutageExtraction,
        extraction_name="outage_extraction",
    )

    # Top-level aggregator corresponding to the rubric root (set non-critical to allow partial credit on subtrees)
    top = evaluator.add_parallel(
        id="verizon_outage_january_2026",
        desc="Complete and accurate information about the major Verizon wireless network outage that occurred in January 2026",
        parent=root,
        critical=False
    )

    # Build verification subtrees
    await verify_basic_facts_with_data(evaluator, top, extracted)
    await verify_timeline(evaluator, top, extracted)
    await verify_technical_cause(evaluator, top, extracted)
    await verify_customer_impact(evaluator, top, extracted)
    await verify_compensation(evaluator, top, extracted)
    await verify_fcc(evaluator, top, extracted)

    return evaluator.get_summary()