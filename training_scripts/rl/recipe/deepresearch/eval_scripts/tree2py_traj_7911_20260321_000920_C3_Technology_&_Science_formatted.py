import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# ------------------------------------------------------------------------------
# Task constants
# ------------------------------------------------------------------------------
TASK_ID = "oracle_cloud_breach_march_2025_eval"
TASK_DESCRIPTION = (
    "In March 2025, a major cybersecurity breach was reported affecting Oracle Cloud's authentication infrastructure. "
    "This incident involved the exploitation of a known CVE vulnerability, resulting in the exfiltration of millions of records from Oracle Cloud tenants. "
    "Your task is to conduct a thorough investigation of this breach and provide the following specific information:\n\n"
    "1. The exact date (including month, day, and year) when this breach was publicly reported\n"
    "2. The specific CVE identifier for the vulnerability that was exploited in this attack\n"
    "3. The name of the affected Oracle component, specifically identifying the agent or module within Oracle Fusion Middleware that was compromised\n"
    "4. The three specific version numbers of Oracle Fusion Middleware that were vulnerable to this CVE\n"
    "5. The handle or username of the threat actor who claimed responsibility for the breach and advertised the stolen data on dark web forums\n"
    "6. The specific Oracle Cloud subdomain that was compromised in this attack, which follows the pattern login.[region-name].oraclecloud.com\n\n"
    "Your answer must include credible source references (URLs) that verify each piece of information. The sources should be from cybersecurity research firms, official security advisories, or reputable technology news outlets."
)

# Expected values per rubric (used in claim phrasing)
EXPECTED_PUBLIC_REPORT_DATE = "March 21, 2025"
EXPECTED_CVE_ID = "CVE-2021-35587"
EXPECTED_COMPONENT = "Oracle Access Manager (OpenSSO Agent)"
EXPECTED_FMW_VERSIONS = ["11.1.2.3.0", "12.2.1.3.0", "12.2.1.4.0"]
EXPECTED_THREAT_ACTOR = "rose87168"
EXPECTED_EXFIL_RECORDS_APPROX_PHRASE = "approximately 6 million"
EXPECTED_TENANTS_EXCEED_PHRASE = "exceed 140,000"
EXPECTED_COMPROMISED_SUBDOMAIN = "login.us2.oraclecloud.com"


# ------------------------------------------------------------------------------
# Extraction models
# ------------------------------------------------------------------------------
class BreachFactsExtraction(BaseModel):
    # 1) Public report date
    date_reported: Optional[str] = None
    date_reported_sources: List[str] = Field(default_factory=list)

    # 2) CVE
    cve_id: Optional[str] = None
    cve_sources: List[str] = Field(default_factory=list)

    # 3) Affected Oracle Fusion Middleware component (agent/module)
    affected_component: Optional[str] = None
    component_sources: List[str] = Field(default_factory=list)

    # 4) Vulnerable Oracle Fusion Middleware versions (three specific versions)
    vulnerable_versions: List[str] = Field(default_factory=list)
    versions_sources: List[str] = Field(default_factory=list)

    # 5) Threat actor handle/username
    threat_actor_handle: Optional[str] = None
    actor_sources: List[str] = Field(default_factory=list)

    # 6) Compromised Oracle Cloud login subdomain
    compromised_subdomain: Optional[str] = None
    subdomain_sources: List[str] = Field(default_factory=list)

    # Additional facts in rubric
    oci_involvement_statement: Optional[str] = None
    oci_involvement_sources: List[str] = Field(default_factory=list)

    exfiltrated_records_statement: Optional[str] = None
    exfiltrated_sources: List[str] = Field(default_factory=list)

    affected_tenants_statement: Optional[str] = None
    tenants_sources: List[str] = Field(default_factory=list)

    unauthenticated_http_statement: Optional[str] = None
    unauthenticated_http_sources: List[str] = Field(default_factory=list)

    cisa_kev_date_statement: Optional[str] = None
    cisa_kev_sources: List[str] = Field(default_factory=list)

    # Optional catch-all of any URLs cited in the answer
    all_urls: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------
# Extraction prompt
# ------------------------------------------------------------------------------
def prompt_extract_breach_facts() -> str:
    return """
Extract the following fields exactly as stated in the answer, and for each fact also extract all URL citations that are explicitly associated with that fact in the answer. If a field is not mentioned, return null for value and an empty list for its sources. Do NOT invent data. Only use what is explicitly present in the answer. For URLs, extract the literal URLs (including those in markdown links).

Required JSON fields:
1) date_reported (string) — the exact public report date mentioned (e.g., "March 21, 2025")
   date_reported_sources (array of strings) — URLs that support this date

2) cve_id (string) — the CVE identifier (e.g., "CVE-2021-35587")
   cve_sources (array of strings)

3) affected_component (string) — the Oracle Fusion Middleware agent/module (e.g., "Oracle Access Manager (OpenSSO Agent)")
   component_sources (array of strings)

4) vulnerable_versions (array of strings) — list all Oracle Fusion Middleware versions stated as vulnerable (e.g., ["11.1.2.3.0","12.2.1.3.0","12.2.1.4.0"])
   versions_sources (array of strings)

5) threat_actor_handle (string) — the handle/username (e.g., "rose87168")
   actor_sources (array of strings)

6) compromised_subdomain (string) — the login subdomain (e.g., "login.us2.oraclecloud.com")
   subdomain_sources (array of strings)

Additional rubric facts:
- oci_involvement_statement (string) — statement indicating the breach involves Oracle Cloud infrastructure (OCI)
  oci_involvement_sources (array of strings)

- exfiltrated_records_statement (string) — statement regarding record count (e.g., "approximately 6 million records")
  exfiltrated_sources (array of strings)

- affected_tenants_statement (string) — statement of affected tenant count (e.g., "exceeds 140,000")
  tenants_sources (array of strings)

- unauthenticated_http_statement (string) — statement that exploitation is possible unauthenticated with network access via HTTP
  unauthenticated_http_sources (array of strings)

- cisa_kev_date_statement (string) — statement that the CVE was added to CISA's KEV catalog in a specified month/year (e.g., "December 2022")
  cisa_kev_sources (array of strings)

Finally:
- all_urls (array of strings) — every URL mentioned anywhere in the answer (deduplicated if possible).
"""


# ------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------
def _flatten_unique_url_lists(*url_lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not isinstance(u, str):
                continue
            s = u.strip()
            if not s:
                continue
            if s not in seen:
                seen.add(s)
                result.append(s)
    return result


# ------------------------------------------------------------------------------
# Verification builders
# ------------------------------------------------------------------------------
async def build_constraint_checks(evaluator: Evaluator, parent_node, facts: BreachFactsExtraction) -> Dict[str, Any]:
    """
    Build 'Constraint_Mandated_Facts' checks. These validate that the answer itself claims specific facts.
    Returns a mapping from logical fact key to the created leaf node for potential dependency use.
    """
    constraint_node = evaluator.add_parallel(
        id="Constraint_Mandated_Facts",
        desc="Answer states all constraint-mandated facts (including specific required values).",
        parent=parent_node,
        critical=True
    )

    nodes = {}

    # 1) Public report date = March 21, 2025
    n = evaluator.add_leaf(
        id="Public_Report_Date_Is_March_21_2025",
        desc="States the breach was publicly reported on March 21, 2025.",
        parent=constraint_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the breach was publicly reported on March 21, 2025.",
        node=n,
        additional_instruction="Judge only based on the provided answer text. Accept minor phrasing variations (e.g., 'reported on Mar 21, 2025')."
    )
    nodes["date"] = n

    # 2) Breach involves Oracle Cloud infrastructure (OCI)
    n = evaluator.add_leaf(
        id="Breach_Involves_Oracle_Cloud_Infrastructure",
        desc="States the breach involves Oracle Cloud infrastructure.",
        parent=constraint_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer clearly states that the breach involves Oracle Cloud (OCI) authentication infrastructure.",
        node=n,
        additional_instruction="Look for wording indicating Oracle Cloud infrastructure or OCI authentication being affected."
    )
    nodes["oci"] = n

    # 3) Exploited CVE = CVE-2021-35587
    n = evaluator.add_leaf(
        id="Exploited_Vulnerability_Is_CVE_2021_35587",
        desc="Identifies the exploited vulnerability as CVE-2021-35587.",
        parent=constraint_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer identifies the exploited vulnerability as {EXPECTED_CVE_ID}.",
        node=n,
        additional_instruction="Small formatting differences (hyphen vs. en dash) are okay as long as it is clearly CVE-2021-35587."
    )
    nodes["cve"] = n

    # 4) Affected component = Oracle Access Manager (OpenSSO Agent)
    n = evaluator.add_leaf(
        id="Affected_Component_Is_Oracle_Access_Manager_OpenSSO_Agent",
        desc="Identifies the affected Oracle Fusion Middleware component as Oracle Access Manager (OpenSSO Agent).",
        parent=constraint_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer identifies the compromised Oracle Fusion Middleware component as the Oracle Access Manager OpenSSO Agent (i.e., OAM OpenSSO Agent).",
        node=n,
        additional_instruction="Allow reasonable naming variants like 'OAM OpenSSO agent' or 'Oracle Access Manager (OAM) OpenSSO agent'."
    )
    nodes["component"] = n

    # 5) Vulnerable FMW versions are exactly the three listed
    n = evaluator.add_leaf(
        id="Vulnerable_FMW_Versions_Are_Exactly_11_1_2_3_0_12_2_1_3_0_12_2_1_4_0",
        desc="Lists the vulnerable Oracle Fusion Middleware versions as 11.1.2.3.0, 12.2.1.3.0, and 12.2.1.4.0 (as the three versions).",
        parent=constraint_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer lists the three vulnerable Oracle Fusion Middleware versions as exactly 11.1.2.3.0, 12.2.1.3.0, and 12.2.1.4.0.",
        node=n,
        additional_instruction="Accept minor punctuation/formatting differences; however, the three specific version numbers must be those and no different versions should be claimed."
    )
    nodes["versions"] = n

    # 6) Threat actor handle = 'rose87168'
    n = evaluator.add_leaf(
        id="Threat_Actor_Handle_Is_rose87168",
        desc="Identifies the threat actor handle as 'rose87168'.",
        parent=constraint_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer identifies the threat actor handle as '{EXPECTED_THREAT_ACTOR}'.",
        node=n,
        additional_instruction="Allow quotes or casing variations; it should be clearly the same handle."
    )
    nodes["actor"] = n

    # 7) Exfiltrated records ≈ 6 million
    n = evaluator.add_leaf(
        id="Exfiltrated_Records_Approximately_6_Million",
        desc="States the number of exfiltrated records is approximately 6 million.",
        parent=constraint_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that approximately 6 million records were exfiltrated.",
        node=n,
        additional_instruction="Allow approximate language like '~6 million', 'around 6M', or 'about six million'."
    )
    nodes["records"] = n

    # 8) Affected tenants exceed 140,000
    n = evaluator.add_leaf(
        id="Affected_Tenants_Exceed_140000",
        desc="States the number of affected cloud tenants exceeds 140,000.",
        parent=constraint_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the number of affected Oracle Cloud tenants exceeds 140,000.",
        node=n,
        additional_instruction="Allow wording variants like 'over 140,000', 'more than 140k'."
    )
    nodes["tenants"] = n

    # 9) Compromised subdomain = login.us2.oraclecloud.com
    n = evaluator.add_leaf(
        id="Compromised_Subdomain_Is_login_us2_oraclecloud_com",
        desc="Identifies the compromised Oracle Cloud subdomain as login.us2.oraclecloud.com.",
        parent=constraint_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer identifies the compromised Oracle Cloud login subdomain as {EXPECTED_COMPROMISED_SUBDOMAIN}.",
        node=n,
        additional_instruction="Small variations like code formatting are okay; the subdomain string should match."
    )
    nodes["subdomain"] = n

    # 10) Vulnerability allows unauthenticated attackers w/ network access via HTTP
    n = evaluator.add_leaf(
        id="Vulnerability_Allows_Unauthenticated_HTTP_Network_Access_Compromise",
        desc="States the vulnerability allows unauthenticated attackers with network access via HTTP to compromise the system.",
        parent=constraint_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that exploitation is possible by unauthenticated attackers with network access via HTTP to compromise the system.",
        node=n,
        additional_instruction="Minor paraphrasing is okay as long as unauthenticated + network access via HTTP + compromise are clearly conveyed."
    )
    nodes["unauth_http"] = n

    # 11) CVE added to CISA KEV in December 2022
    n = evaluator.add_leaf(
        id="CVE_Added_To_CISA_KEV_In_December_2022",
        desc="States the CVE was added to CISA's Known Exploited Vulnerabilities (KEV) catalog in December 2022.",
        parent=constraint_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer states that {EXPECTED_CVE_ID} was added to CISA's Known Exploited Vulnerabilities catalog in December 2022.",
        node=n,
        additional_instruction="Variants like 'Dec 2022' or 'December, 2022' are fine."
    )
    nodes["kev"] = n

    return nodes


async def build_source_checks(
    evaluator: Evaluator,
    parent_node,
    facts: BreachFactsExtraction,
    constraint_nodes: Dict[str, Any]
) -> None:
    """
    Build 'Source_Requirements' checks.
    Verifies that each constraint-mandated fact is supported by at least one cited, credible URL.
    """
    src_root = evaluator.add_parallel(
        id="Source_Requirements",
        desc="Answer provides credible URL citations that verify each constraint-mandated fact.",
        parent=parent_node,
        critical=True
    )

    # Per-fact citations present and supportive
    per_fact_node = evaluator.add_parallel(
        id="Per_Fact_Citations_Present",
        desc="Each constraint-mandated fact has at least one supporting URL citation.",
        parent=src_root,
        critical=True
    )

    # Helper to create a per‑fact verification leaf
    async def _verify_fact_with_sources(
        node_id: str,
        desc: str,
        claim: str,
        sources: List[str],
        prereq_key: Optional[str] = None,
        additional_instruction: Optional[str] = None
    ):
        ln = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=per_fact_node,
            critical=True
        )
        await evaluator.verify(
            claim=claim,
            node=ln,
            sources=sources,
            additional_instruction=additional_instruction or "None"
        )

    # 1) Public report date support
    await _verify_fact_with_sources(
        node_id="Citation_For_Public_Report_Date",
        desc="Includes at least one URL supporting the public report date (March 21, 2025).",
        claim=f"The breach was publicly reported on {EXPECTED_PUBLIC_REPORT_DATE}.",
        sources=facts.date_reported_sources,
        prereq_key="date",
        additional_instruction="Verify the publication/report date on the cited page(s) supports the stated date. If the pages are unrelated or do not mention such a date, return Incorrect."
    )

    # 2) Oracle Cloud infrastructure involvement support
    await _verify_fact_with_sources(
        node_id="Citation_For_Oracle_Cloud_Infrastructure_Involvement",
        desc="Includes at least one URL supporting Oracle Cloud infrastructure involvement.",
        claim="The breach involves Oracle Cloud infrastructure (OCI) and its authentication infrastructure component(s).",
        sources=facts.oci_involvement_sources,
        prereq_key="oci",
        additional_instruction="Verify that the page(s) explicitly indicate Oracle Cloud (OCI) infrastructure involvement. If ambiguous or unrelated, return Incorrect."
    )

    # 3) CVE support
    await _verify_fact_with_sources(
        node_id="Citation_For_CVE_2021_35587",
        desc="Includes at least one URL supporting CVE-2021-35587 as the exploited vulnerability.",
        claim=f"The exploited vulnerability in this breach is {EXPECTED_CVE_ID}.",
        sources=facts.cve_sources,
        prereq_key="cve",
        additional_instruction="The page(s) should explicitly mention CVE-2021-35587 in the context of the described incident or affected component."
    )

    # 4) Affected component support
    await _verify_fact_with_sources(
        node_id="Citation_For_Affected_Component",
        desc="Includes at least one URL supporting Oracle Access Manager (OpenSSO Agent) as the affected component.",
        claim="The compromised Oracle Fusion Middleware component was the Oracle Access Manager OpenSSO Agent (OAM OpenSSO Agent).",
        sources=facts.component_sources,
        prereq_key="component",
        additional_instruction="Allow reasonable naming variants (Oracle Access Manager agent, OAM OpenSSO agent). The page(s) should tie this component to the vulnerability/incident."
    )

    # 5) Vulnerable versions support
    await _verify_fact_with_sources(
        node_id="Citation_For_Vulnerable_Versions",
        desc="Includes at least one URL supporting the vulnerable versions 11.1.2.3.0, 12.2.1.3.0, and 12.2.1.4.0.",
        claim="The vulnerable Oracle Fusion Middleware versions are 11.1.2.3.0, 12.2.1.3.0, and 12.2.1.4.0.",
        sources=facts.versions_sources,
        prereq_key="versions",
        additional_instruction="The page(s) should explicitly enumerate these version numbers as affected/vulnerable."
    )

    # 6) Threat actor handle support
    await _verify_fact_with_sources(
        node_id="Citation_For_Threat_Actor_Handle",
        desc="Includes at least one URL supporting the threat actor handle 'rose87168'.",
        claim=f"The threat actor who claimed responsibility/advertised the data used the handle '{EXPECTED_THREAT_ACTOR}'.",
        sources=facts.actor_sources,
        prereq_key="actor",
        additional_instruction="The page(s) should explicitly reference the handle, ideally in the context of this Oracle incident."
    )

    # 7) Exfiltrated record count support
    await _verify_fact_with_sources(
        node_id="Citation_For_Exfiltrated_Record_Count",
        desc="Includes at least one URL supporting approximately 6 million exfiltrated records.",
        claim="Approximately 6 million records were exfiltrated in this incident.",
        sources=facts.exfiltrated_sources,
        prereq_key="records",
        additional_instruction="Allow approximations (e.g., ~6M, 'about 6 million') if clearly conveying the same magnitude."
    )

    # 8) Affected tenant count support
    await _verify_fact_with_sources(
        node_id="Citation_For_Affected_Tenant_Count",
        desc="Includes at least one URL supporting that affected tenants exceed 140,000.",
        claim="The number of affected Oracle Cloud tenants exceeds 140,000.",
        sources=facts.tenants_sources,
        prereq_key="tenants",
        additional_instruction="Support should indicate 'more than 140,000' or similar wording."
    )

    # 9) Compromised subdomain support
    await _verify_fact_with_sources(
        node_id="Citation_For_Compromised_Subdomain",
        desc="Includes at least one URL supporting the compromised subdomain login.us2.oraclecloud.com.",
        claim=f"The compromised Oracle Cloud login subdomain was {EXPECTED_COMPROMISED_SUBDOMAIN}.",
        sources=facts.subdomain_sources,
        prereq_key="subdomain",
        additional_instruction="The page(s) should mention that exact subdomain when describing the incident."
    )

    # 10) Unauthenticated HTTP condition support
    await _verify_fact_with_sources(
        node_id="Citation_For_Unauthenticated_HTTP_Condition",
        desc="Includes at least one URL supporting that exploitation is possible unauthenticated with network access via HTTP.",
        claim="The vulnerability allows unauthenticated attackers with network access via HTTP to compromise the system.",
        sources=facts.unauthenticated_http_sources,
        prereq_key="unauth_http",
        additional_instruction="This should be explicitly stated in the advisory/research page (e.g., 'unauthenticated network access via HTTP')."
    )

    # 11) CISA KEV inclusion (Dec 2022) support
    await _verify_fact_with_sources(
        node_id="Citation_For_CISA_KEV_Dec_2022",
        desc="Includes at least one URL supporting KEV inclusion in December 2022.",
        claim=f"{EXPECTED_CVE_ID} was added to CISA's Known Exploited Vulnerabilities catalog in December 2022.",
        sources=facts.cisa_kev_sources,
        prereq_key="kev",
        additional_instruction="The page(s) should show CISA KEV inclusion and indicate December 2022."
    )

    # Source credibility (all cited sources should be credible)
    all_src_urls = _flatten_unique_url_lists(
        facts.date_reported_sources,
        facts.cve_sources,
        facts.component_sources,
        facts.versions_sources,
        facts.actor_sources,
        facts.subdomain_sources,
        facts.oci_involvement_sources,
        facts.exfiltrated_sources,
        facts.tenants_sources,
        facts.unauthenticated_http_sources,
        facts.cisa_kev_sources,
        facts.all_urls,
    )
    credibility_node = evaluator.add_leaf(
        id="Source_Credibility",
        desc="Cited sources are from cybersecurity research firms, official security advisories, or reputable technology news outlets.",
        parent=src_root,
        critical=True
    )
    urls_for_prompt = "\n".join(f"- {u}" for u in all_src_urls) if all_src_urls else "(no URLs extracted)"
    credibility_claim = (
        "All of the following cited URLs are from credible cybersecurity research firms, official security advisories, "
        "or reputable technology news outlets:\n" + urls_for_prompt
    )
    await evaluator.verify(
        claim=credibility_claim,
        node=credibility_node,
        additional_instruction=(
            "Assess credibility primarily by domain/brand reputation. Credible examples include vendor advisories (e.g., oracle.com), "
            "government or standards orgs (e.g., cisa.gov, nist.gov), reputable security companies (e.g., Mandiant, CrowdStrike, "
            "Rapid7, Tenable, Qualys), and reputable tech/security media (e.g., The Register, Ars Technica, BleepingComputer). "
            "If ANY URL is from an untrusted paste site, anonymous forum, social media, or dubious aggregator, return Incorrect. "
            "If the list is empty, return Incorrect."
        )
    )


# ------------------------------------------------------------------------------
# Main evaluation entry point
# ------------------------------------------------------------------------------
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Oracle Cloud breach investigation task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: constraints vs. sources are independent groups
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

    # 1) Extract structured facts and per-fact citations from the answer
    facts: BreachFactsExtraction = await evaluator.extract(
        prompt=prompt_extract_breach_facts(),
        template_class=BreachFactsExtraction,
        extraction_name="breach_facts_extraction",
    )

    # 2) Ground-truth (rubric-expected) info for transparency/debugging
    evaluator.add_ground_truth({
        "expected_public_report_date": EXPECTED_PUBLIC_REPORT_DATE,
        "expected_cve_id": EXPECTED_CVE_ID,
        "expected_component": EXPECTED_COMPONENT,
        "expected_fmw_versions": EXPECTED_FMW_VERSIONS,
        "expected_threat_actor": EXPECTED_THREAT_ACTOR,
        "expected_compromised_subdomain": EXPECTED_COMPROMISED_SUBDOMAIN,
        "expected_records_approx": EXPECTED_EXFIL_RECORDS_APPROX_PHRASE,
        "expected_tenants_threshold": EXPECTED_TENANTS_EXCEED_PHRASE,
        "kev_inclusion_month_year": "December 2022"
    })

    # 3) Build verification tree
    rubric_root = evaluator.add_parallel(
        id="Oracle_Cloud_Breach_Investigation",
        desc="Evaluate whether the answer satisfies all stated constraints and provides credible URL citations for each required fact.",
        parent=root,
        critical=True
    )

    # 3.a) Constraint (answer-stated facts) checks
    constraint_nodes = await build_constraint_checks(evaluator, rubric_root, facts)

    # 3.b) Source requirements (per-fact support + credibility)
    await build_source_checks(evaluator, rubric_root, facts, constraint_nodes)

    # 4) Return summary
    return evaluator.get_summary()