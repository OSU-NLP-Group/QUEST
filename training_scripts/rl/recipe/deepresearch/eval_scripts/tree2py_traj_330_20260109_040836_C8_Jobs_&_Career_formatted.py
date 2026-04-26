import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "it_certs_progression"
TASK_DESCRIPTION = """
Identify three professional IT certifications that together form a logical career progression path, meeting ALL of the following criteria:

Certification Requirements:
- Each certification must be from a major industry-recognized provider (CompTIA, ISC2, AWS, Microsoft, Google Cloud, or similar established organizations)
- The three certifications must be from at least two different certification providers
- The three certifications must represent different professional levels (e.g., entry-level, associate, professional) to demonstrate career progression
- Each certification must cover a distinct or complementary IT domain (such as general IT support, networking, security, cloud computing, or systems administration)

Documentation Requirements:
For each certification, provide:
1. Official certification name
2. Certifying organization
3. Prerequisite requirements (or explicitly state "no prerequisites")
4. Recommended or required years of experience
5. Exam format details (number of questions, duration, or similar exam characteristics)
6. Approximate cost (exam fee)
7. Certification validity period and renewal/recertification requirements
8. Professional level (entry, associate, professional, or equivalent)
9. Primary IT domain(s) covered
10. Direct link to the official certification page

Career Progression:
- The three certifications should form a logical progression that an IT professional might pursue to advance their career from entry-level to more advanced positions
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CertificationItem(BaseModel):
    name: Optional[str] = None
    organization: Optional[str] = None
    prerequisites: Optional[str] = None  # "no prerequisites" if explicitly none
    experience: Optional[str] = None     # Keep as free text (e.g., "1 year", "2–3 years recommended")
    exam_details: Optional[str] = None   # e.g., "65 questions, 90 minutes, multiple choice"
    cost: Optional[str] = None           # e.g., "$150", "USD 300"
    renewal: Optional[str] = None        # validity period + renewal/recert details
    level: Optional[str] = None          # entry / associate / professional / expert / foundational / etc.
    domains: List[str] = Field(default_factory=list)  # primary IT domains
    official_link: Optional[str] = None  # direct official certification page URL


class CertificationsExtraction(BaseModel):
    certifications: List[CertificationItem] = Field(default_factory=list)
    progression_rationale: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_certifications() -> str:
    return """
    Extract exactly the information the answer presents about professional IT certifications.
    Return a JSON object with:
    - certifications: an array of certification objects in the order presented in the answer. For EACH certification, extract:
        • name: official certification name (string)
        • organization: certifying organization (string, e.g., CompTIA, (ISC)2, AWS, Microsoft, Google Cloud, Cisco, Red Hat, VMware, ISACA, GIAC, EC-Council, IBM, Oracle, Salesforce, etc.)
        • prerequisites: prerequisite requirements text as stated. If the answer explicitly states there are no prerequisites, set to "no prerequisites". If not specified, set null.
        • experience: recommended or required experience years (free text as stated); if not specified, set null.
        • exam_details: exam format details (e.g., number of questions, duration, type). If not specified, set null.
        • cost: approximate exam fee as presented (string); if not provided, set null.
        • renewal: certification validity and renewal/recertification requirements; if not provided, set null.
        • level: professional level label as stated (e.g., entry-level, foundational, associate, professional, expert, specialty, etc.). If not specified, set null.
        • domains: list of primary IT domain(s) covered (e.g., general IT support, networking, security, cloud computing, systems administration, devops, etc.).
                 If domains are presented as a sentence, split them into individual items by commas, slashes, or "and".
                 If not specified, return an empty list.
        • official_link: direct link (URL) to the official certification page (string). If the answer provides multiple links,
                 choose the one that appears to be the direct official page for this specific certification. If none is given, set null.
    - progression_rationale: the explanation given in the answer for why these three certifications form a logical career progression path.
                 If no explicit explanation is given, set null.

    IMPORTANT RULES:
    1) Extract only information explicitly present in the answer. Do not invent or infer missing details.
    2) Preserve the wording of numbers and durations as text (e.g., "90 minutes", "65 questions", "$150").
    3) If a field is not mentioned, return null (or empty array for 'domains').
    4) Only extract URLs that are explicitly present in the answer text. Use the one that appears most official for 'official_link'.
    """


# --------------------------------------------------------------------------- #
# Helper normalization and evaluation utilities                               #
# --------------------------------------------------------------------------- #
_MAJOR_PROVIDERS_CANON = {
    # Core examples
    "comptia", "(isc)2", "isc2", "aws", "amazonwebservices", "amazon web services",
    "microsoft", "google", "googlecloud", "google cloud",
    # Additional established cert orgs
    "cisco", "redhat", "vmware", "isaca", "giac", "ec-council", "eccouncil", "oracle", "ibm", "salesforce",
}

def _normalize_org_name(org: Optional[str]) -> str:
    if not org:
        return ""
    s = org.lower().strip()
    # Remove punctuation and spaces
    s = re.sub(r"[\s\-\.\(\)®™\u00AE\u2122]+", "", s)
    # Simple normalizations
    s = s.replace("amazonwebservices", "amazonwebservices")
    s = s.replace("googlecloudplatform", "googlecloud")
    return s


def is_major_provider(org: Optional[str]) -> bool:
    norm = _normalize_org_name(org)
    if not norm:
        return False
    # Check direct matches or substrings mapping
    for prov in _MAJOR_PROVIDERS_CANON:
        if prov in norm:
            return True
    return False


def normalize_domains_to_coarse(domains: List[str]) -> List[str]:
    coarse = []
    for d in domains:
        s = d.lower().strip()
        if not s:
            continue
        label = None
        # Coarse buckets
        if any(k in s for k in ["support", "help desk", "helpdesk", "service desk", "a+"]):
            label = "it_support"
        elif any(k in s for k in ["network", "ccna", "routing", "switching"]):
            label = "networking"
        elif any(k in s for k in ["security", "cyber", "cissp", "sec+"]):
            label = "security"
        elif any(k in s for k in ["cloud", "aws", "azure", "gcp", "google cloud", "solutions architect"]):
            label = "cloud"
        elif any(k in s for k in ["devops", "sre", "site reliability"]):
            label = "devops"
        elif any(k in s for k in ["system admin", "systems admin", "sysadmin", "linux", "windows server"]):
            label = "systems_admin"
        elif any(k in s for k in ["data", "database", "sql", "big data", "analytics"]):
            label = "data"
        elif any(k in s for k in ["virtualization", "vmware"]):
            label = "virtualization"
        elif any(k in s for k in ["governance", "risk", "compliance", "grc", "audit"]):
            label = "grc"
        else:
            # default to raw token simplified
            label = re.sub(r"[^a-z0-9]+", "_", s)
        coarse.append(label)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in coarse:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def level_to_rank(level_text: Optional[str]) -> Optional[int]:
    if not level_text:
        return None
    s = level_text.lower().strip()
    s = s.replace("–", "-").replace("—", "-")
    # Canonical mapping
    if any(k in s for k in ["entry", "foundational", "foundation", "beginner", "junior", "practitioner"]):
        return 1
    if any(k in s for k in ["associate", "administrator", "specialist", "analyst"]):
        return 2
    if any(k in s for k in ["professional", "advanced", "architect", "engineer", "solutions architect professional", "pro "]):
        return 3
    if any(k in s for k in ["expert", "master"]):
        return 4
    if "specialty" in s:
        return 3
    return None


def to_domains_list(domains_field: List[str] | str | None) -> List[str]:
    if domains_field is None:
        return []
    if isinstance(domains_field, list):
        return [d for d in domains_field if isinstance(d, str)]
    if isinstance(domains_field, str):
        # split by comma or slash or ' and '
        parts = re.split(r",|/| and ", domains_field)
        return [p.strip() for p in parts if p and isinstance(p, str)]
    return []


def non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification for a single certification                                     #
# --------------------------------------------------------------------------- #
async def verify_single_cert(
    evaluator: Evaluator,
    parent_node,
    cert: CertificationItem,
    idx: int,
) -> None:
    """
    Build and verify the subtree for one certification.
    Matches rubric subtree:
      Certification_i (parallel, critical)
        - Cert{i}_Name
        - Cert{i}_Organization
        - Cert{i}_Major_Provider
        - Cert{i}_Prerequisites
        - Cert{i}_Experience
        - Cert{i}_Exam_Details
        - Cert{i}_Cost
        - Cert{i}_Renewal
        - Cert{i}_Level
        - Cert{i}_Domain
        - Cert{i}_Official_Link
    """
    ci = idx + 1
    cert_node = evaluator.add_parallel(
        id=f"Certification_{ci}",
        desc=f"Certification {ci} details are provided and meet per-certification requirements",
        parent=parent_node,
        critical=True,
    )

    # Prepare commonly used values
    url = cert.official_link or ""
    name = cert.name or ""
    org = cert.organization or ""
    dom_list = to_domains_list(cert.domains)
    dom_join = ", ".join(dom_list) if dom_list else ""

    # Name provided -> verify against official page (checks both presence and correctness)
    name_node = evaluator.add_leaf(
        id=f"Cert{ci}_Name",
        desc="Official certification name is provided",
        parent=cert_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f'The official certification page shows the certification name as "{name}" or an equivalent naming of the same certification.',
        node=name_node,
        sources=url if non_empty(url) else None,
        additional_instruction="Use the page title, H1, or obvious certification name on the official page. Allow minor variations and trademarks (®/™)."
    )

    # Organization
    org_node = evaluator.add_leaf(
        id=f"Cert{ci}_Organization",
        desc="Certifying organization is identified",
        parent=cert_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f'The certifying organization for "{name}" is "{org}".',
        node=org_node,
        sources=url if non_empty(url) else None,
        additional_instruction="Confirm that this certification is offered by the stated organization; allow minor brand/format variants (e.g., (ISC)² vs ISC2)."
    )

    # Major provider (custom boolean)
    major_node = evaluator.add_custom_node(
        result=is_major_provider(org),
        id=f"Cert{ci}_Major_Provider",
        desc="Certifying organization qualifies as a major industry-recognized provider (e.g., one of the examples given or similarly established)",
        parent=cert_node,
        critical=True
    )

    # Prerequisites
    prereq_node = evaluator.add_leaf(
        id=f"Cert{ci}_Prerequisites",
        desc="Prerequisite requirements are documented (or explicitly stated as none)",
        parent=cert_node,
        critical=True,
    )
    prereq_text = cert.prerequisites or ""
    prereq_claim = f'This certification has no prerequisites.' if prereq_text.lower().strip() in {"no prerequisites", "none"} else f'The prerequisites for "{name}" include: "{prereq_text}".'
    await evaluator.verify(
        claim=prereq_claim,
        node=prereq_node,
        sources=url if non_empty(url) else None,
        additional_instruction="Verify whether the page explicitly indicates no prerequisites OR lists the stated prerequisites. Allow wording differences."
    )

    # Experience
    exp_node = evaluator.add_leaf(
        id=f"Cert{ci}_Experience",
        desc="Recommended or required years of experience are specified",
        parent=cert_node,
        critical=True,
    )
    exp_text = cert.experience or ""
    await evaluator.verify(
        claim=f'The recommended or required experience for "{name}" is described as: "{exp_text}".',
        node=exp_node,
        sources=url if non_empty(url) else None,
        additional_instruction="Check for mentions of recommended or required experience (years). Allow range expressions and approximate wording."
    )

    # Exam details
    exam_node = evaluator.add_leaf(
        id=f"Cert{ci}_Exam_Details",
        desc="Exam format details are provided (e.g., number of questions, duration, or similar exam characteristics)",
        parent=cert_node,
        critical=True,
    )
    exam_text = cert.exam_details or ""
    await evaluator.verify(
        claim=f'The exam format details for "{name}" include: "{exam_text}".',
        node=exam_node,
        sources=url if non_empty(url) else None,
        additional_instruction="Look for details like number of questions, duration, format type (e.g., multiple choice). Minor variations acceptable."
    )

    # Cost
    cost_node = evaluator.add_leaf(
        id=f"Cert{ci}_Cost",
        desc="Approximate exam cost (exam fee) is provided",
        parent=cert_node,
        critical=True,
    )
    cost_text = cert.cost or ""
    await evaluator.verify(
        claim=f'The exam fee (approximate) for "{name}" is "{cost_text}".',
        node=cost_node,
        sources=url if non_empty(url) else None,
        additional_instruction="Verify the stated fee or price tier on the official page. Currency symbols or minor regional variations are acceptable."
    )

    # Renewal
    renewal_node = evaluator.add_leaf(
        id=f"Cert{ci}_Renewal",
        desc="Certification validity period and renewal/recertification requirements are documented",
        parent=cert_node,
        critical=True,
    )
    renewal_text = cert.renewal or ""
    await evaluator.verify(
        claim=f'The certification validity/renewal for "{name}" is described as: "{renewal_text}".',
        node=renewal_node,
        sources=url if non_empty(url) else None,
        additional_instruction="Look for validity period (e.g., 3 years) and how to renew (e.g., CEUs/PDUs, retake exam). Accept synonymous phrasing."
    )

    # Level
    level_node = evaluator.add_leaf(
        id=f"Cert{ci}_Level",
        desc="Professional level is identified (entry/associate/professional or equivalent)",
        parent=cert_node,
        critical=True,
    )
    level_text = cert.level or ""
    await evaluator.verify(
        claim=f'The professional level designation for "{name}" is "{level_text}" or an equivalent label (e.g., foundational, associate, professional, expert).',
        node=level_node,
        sources=url if non_empty(url) else None,
        additional_instruction="Confirm level designation or an equivalent term on the page. Allow mapping of synonyms (foundational≈entry, expert≈advanced)."
    )

    # Domain(s)
    domain_node = evaluator.add_leaf(
        id=f"Cert{ci}_Domain",
        desc="Primary IT domain(s) covered by the certification are specified",
        parent=cert_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f'This certification primarily covers the following domain(s): "{dom_join}".',
        node=domain_node,
        sources=url if non_empty(url) else None,
        additional_instruction="Confirm that the stated domain(s) match the scope described on the page (e.g., security, cloud, networking, IT support)."
    )

    # Official link
    official_link_node = evaluator.add_leaf(
        id=f"Cert{ci}_Official_Link",
        desc="Direct link (URL) to the official certification page is provided",
        parent=cert_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f'This URL is the official certification page for "{name}" from "{org}".',
        node=official_link_node,
        sources=url if non_empty(url) else None,
        additional_instruction="The page should appear to be the official source (owned by the certifying organization) and clearly describe the certification."
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
    Evaluate an answer for the IT certifications progression task.
    """
    # Initialize evaluator (note: root here is a wrapper; real 'Root' will be a critical child)
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

    # Extract structured info from the answer
    extraction: CertificationsExtraction = await evaluator.extract(
        prompt=prompt_extract_certifications(),
        template_class=CertificationsExtraction,
        extraction_name="certifications_extraction"
    )

    # Create the real critical "Root" node from rubric
    root_main = evaluator.add_parallel(
        id="Root",
        desc="Overall evaluation of the three-certification career progression path",
        parent=root,
        critical=True
    )

    # Count exactly three certifications are identified in the answer (no fewer, no more)
    # Count all distinct certifications with at least a name present
    all_named = [c for c in extraction.certifications if non_empty(c.name)]
    exactly_three_bool = (len(all_named) == 3)
    evaluator.add_custom_node(
        result=exactly_three_bool,
        id="Count_Exactly_Three",
        desc="Exactly three certifications are identified (no fewer, no more)",
        parent=root_main,
        critical=True
    )

    # Select exactly the first 3 certifications for downstream checks (pad if needed)
    selected: List[CertificationItem] = []
    for c in extraction.certifications:
        if len(selected) >= 3:
            break
        selected.append(c)
    while len(selected) < 3:
        selected.append(CertificationItem())

    # Build subtrees for Certification 1..3
    for i in range(3):
        await verify_single_cert(evaluator, root_main, selected[i], i)

    # Provider diversity (at least two different providers among the three)
    orgs = [_normalize_org_name(c.organization) for c in selected if non_empty(c.organization)]
    unique_orgs = set([o for o in orgs if o])
    provider_diverse = len(unique_orgs) >= 2
    evaluator.add_custom_node(
        result=provider_diverse,
        id="Provider_Diversity",
        desc="The three certifications are from at least two different certification providers",
        parent=root_main,
        critical=True
    )

    # Domain set complementarity (not all the same; at least two distinct coarse domains)
    all_domains: List[str] = []
    per_cert_coarse: List[List[str]] = []
    for c in selected:
        coarse = normalize_domains_to_coarse(to_domains_list(c.domains))
        per_cert_coarse.append(coarse)
        all_domains.extend(coarse)
    unique_domains = set(all_domains)
    domain_complement = len(unique_domains) >= 2
    evaluator.add_custom_node(
        result=domain_complement,
        id="Domain_Set_Complementarity",
        desc="Across the set, the three certifications cover distinct or complementary IT domains (not all the same; collectively coherent coverage such as networking/security/cloud/etc.)",
        parent=root_main,
        critical=True
    )

    # Career Progression node (parallel, critical)
    prog_node = evaluator.add_parallel(
        id="Career_Progression",
        desc="The three certifications form a logical career progression from less advanced to more advanced",
        parent=root_main,
        critical=True
    )

    # Progression level differentiation: levels are different and ordered from lower to higher
    levels = [c.level for c in selected]
    ranks: List[Optional[int]] = [level_to_rank(lvl) for lvl in levels]
    # Strictly increasing (all three defined and rank1 < rank2 < rank3)
    if None in ranks:
        progression_ok = False
    else:
        progression_ok = bool(ranks[0] < ranks[1] < ranks[2])
    evaluator.add_custom_node(
        result=progression_ok,
        id="Progression_Level_Differentiation",
        desc="The three certifications represent different professional levels (e.g., entry, associate, professional or equivalent) and are ordered from lower to higher level",
        parent=prog_node,
        critical=True
    )

    # Progression rationale (LLM checks that a clear explanation is provided in the answer)
    rationale_leaf = evaluator.add_leaf(
        id="Progression_Rationale",
        desc="A clear explanation is provided for why the sequence is a realistic progression path an IT professional might pursue",
        parent=prog_node,
        critical=True
    )
    rationale_text = extraction.progression_rationale or ""
    await evaluator.verify(
        claim=f'The answer provides a clear, explicit rationale explaining why the selected three certifications form a realistic career progression path. Extracted rationale (if any): "{rationale_text}".',
        node=rationale_leaf,
        additional_instruction="Look for explicit explanation that connects levels and domains into a logical path (entry → more advanced). The explanation can be anywhere in the answer."
    )

    # Record some custom debug info to help interpretation
    evaluator.add_custom_info(
        info={
            "selected_cert_orgs": [c.organization for c in selected],
            "selected_cert_levels": levels,
            "level_ranks": ranks,
            "provider_unique_count": len(unique_orgs),
            "per_cert_domains_raw": [c.domains for c in selected],
            "per_cert_domains_coarse": per_cert_coarse,
            "unique_domains_count": len(unique_domains),
            "major_provider_checks": [
                {"organization": c.organization, "is_major": is_major_provider(c.organization)}
                for c in selected
            ],
            "extracted_total_named": len(all_named),
        },
        info_type="debug",
        info_name="evaluation_debug_info"
    )

    return evaluator.get_summary()