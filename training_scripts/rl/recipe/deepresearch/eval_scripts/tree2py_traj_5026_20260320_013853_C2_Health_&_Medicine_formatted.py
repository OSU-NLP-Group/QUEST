import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sma_gene_therapy_2019"
TASK_DESCRIPTION = (
    "On May 24, 2019, the U.S. Food and Drug Administration approved a gene therapy for the treatment of spinal "
    "muscular atrophy (SMA). At the time of approval, this therapy was priced at $2.125 million, making it the most "
    "expensive drug ever approved. Identify the name of this gene therapy and provide the following verified "
    "information: (1) The specific viral vector technology (serotype) used in this gene therapy, (2) The method and "
    "frequency of administration (e.g., single-dose, multiple-dose, route), (3) URL references from credible sources "
    "(FDA, medical journals, or pharmaceutical company announcements) supporting each claim."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TherapyInfo(BaseModel):
    therapy_name: Optional[str] = None
    identification_urls: List[str] = Field(default_factory=list)


class RequiredClaims(BaseModel):
    fda_approval_date: Optional[str] = None
    fda_approval_urls: List[str] = Field(default_factory=list)

    indication: Optional[str] = None
    indication_urls: List[str] = Field(default_factory=list)

    price_at_approval: Optional[str] = None
    price_urls: List[str] = Field(default_factory=list)

    most_expensive_claim: Optional[str] = None
    most_expensive_urls: List[str] = Field(default_factory=list)

    viral_vector_serotype: Optional[str] = None
    vector_urls: List[str] = Field(default_factory=list)

    administration_method: Optional[str] = None  # e.g., "single-dose IV infusion" or "one-time intravenous infusion"
    administration_urls: List[str] = Field(default_factory=list)


class SMAEvaluationExtraction(BaseModel):
    therapy: Optional[TherapyInfo] = None
    claims: Optional[RequiredClaims] = None
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_sma_therapy_info() -> str:
    return """
    Extract the requested information from the provided answer text. Do not invent anything; only extract what is explicitly present.
    Return a JSON with the following structure:

    {
      "therapy": {
        "therapy_name": string or null,
        "identification_urls": [url, ...]  // URLs that directly support identifying the therapy by name
      },
      "claims": {
        "fda_approval_date": string or null,        // e.g., "May 24, 2019" if stated
        "fda_approval_urls": [url, ...],

        "indication": string or null,               // statement about SMA indication as written in the answer
        "indication_urls": [url, ...],

        "price_at_approval": string or null,        // e.g., "$2.125 million" if stated
        "price_urls": [url, ...],

        "most_expensive_claim": string or null,     // the exact phrasing (e.g., "most expensive drug at the time")
        "most_expensive_urls": [url, ...],

        "viral_vector_serotype": string or null,    // e.g., "AAV9" or "adeno-associated virus serotype 9"
        "vector_urls": [url, ...],

        "administration_method": string or null,    // include both route and frequency if present, e.g., "single-dose intravenous (IV) infusion"
        "administration_urls": [url, ...]
      },
      "all_urls": [url, ...] // every URL appearing anywhere in the answer, including a sources section or inline links
    }

    URL extraction rules:
    - Include only URLs that are explicitly present in the answer text (plain URLs or markdown links).
    - Do not fabricate or guess URLs.
    - Where the answer ties a specific URL to a specific claim, place it in that claim’s list.
    - If the answer provides a single combined sources list without mapping to specific claims, put all of them into "all_urls" and also duplicate into any relevant claim-specific lists if the answer’s text context clearly associates them.
    - If a field is not mentioned, set it to null (or empty list for URLs).

    Important: Keep strings exactly as in the answer (e.g., use "$2.125 million" if that's what is written). For administration_method, prefer a compact summary phrase directly quoted or closely paraphrased from the answer that includes both dosing frequency and route when available.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def _merge_urls(*url_lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        combined.extend(lst or [])
    return _dedup_urls(combined)


def _safe(s: Optional[str], default: str = "") -> str:
    return s if isinstance(s, str) else default


def _has_route_and_freq(admin_text: Optional[str]) -> bool:
    """
    Heuristic: consider 'administration_method' acceptable if it contains
    at least one frequency indicator and at least one route indicator.
    """
    if not admin_text:
        return False
    text = admin_text.lower()

    freq_tokens = [
        "single", "single-dose", "one-time", "one time", "once", "one‑time", "one–time", "one‐time"
    ]
    route_tokens = [
        "intravenous", "iv", "intrav", "intrathecal", "it", "infusion"
    ]
    has_freq = any(tok in text for tok in freq_tokens)
    has_route = any(tok in text for tok in route_tokens)
    return has_freq and has_route


def _therapy_label(therapy_name: Optional[str]) -> str:
    if therapy_name and therapy_name.strip():
        return therapy_name.strip()
    return "the SMA gene therapy"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_therapy_identification(
    evaluator: Evaluator,
    parent,
    data: SMAEvaluationExtraction,
) -> None:
    """
    Build the 'Therapy_Identification' subtree:
    - Ensure therapy name stated in answer.
    - Ensure at least one URL is provided for identification and that a cited page actually identifies the therapy by that name.
    """
    therapy = data.therapy or TherapyInfo()
    claims = data.claims or RequiredClaims()

    all_collected_urls = _merge_urls(
        therapy.identification_urls,
        claims.fda_approval_urls,
        claims.indication_urls,
        claims.price_urls,
        claims.most_expensive_urls,
        claims.vector_urls,
        claims.administration_urls,
        data.all_urls,
    )

    node = evaluator.add_parallel(
        id="Therapy_Identification",
        desc="Provide the gene therapy name and at least one supporting citation identifying it as the therapy described.",
        parent=parent,
        critical=True,
    )

    # 1) Therapy name stated (existence in the answer)
    evaluator.add_custom_node(
        result=bool(therapy.therapy_name and therapy.therapy_name.strip()),
        id="Therapy_Name",
        desc="Answer states the name of the gene therapy.",
        parent=node,
        critical=True,
    )

    # 2) Therapy identification supported by URL(s): break into a small sequential gate
    sub_seq = evaluator.add_sequential(
        id="Therapy_Name_Citation",
        desc="At least one URL supports identifying the therapy by name.",
        parent=node,
        critical=True,
    )

    # 2.a) URL exists
    has_any_ident_url = len(_merge_urls(therapy.identification_urls, all_collected_urls)) > 0
    evaluator.add_custom_node(
        result=has_any_ident_url,
        id="Therapy_Name_Citation_URL",
        desc="Provides at least one URL reference supporting the identification of the therapy name.",
        parent=sub_seq,
        critical=True,
    )

    # 2.b) Page(s) actually identify the therapy by name
    support_leaf = evaluator.add_leaf(
        id="Therapy_Name_Supported_By_URL",
        desc="A cited page clearly identifies the therapy by the stated name.",
        parent=sub_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The cited page identifies the gene therapy named '{_safe(therapy.therapy_name, 'Zolgensma or equivalent official name')}'. "
              f"Minor variations between brand and generic names (e.g., onasemnogene abeparvovec vs. brand) should be accepted.",
        node=support_leaf,
        sources=_merge_urls(therapy.identification_urls, all_collected_urls),
        additional_instruction="Accept brand/generic equivalence and minor formatting differences. The page should clearly tie the name to the SMA gene therapy."
    )


async def _build_claim_seq_stated_url_supported(
    evaluator: Evaluator,
    parent,
    base_id: str,
    stated_desc: str,
    stated_claim: str,
    support_desc: str,
    support_claim: str,
    urls: List[str],
    stated_additional_instruction: str = "None",
    support_additional_instruction: str = "None",
) -> None:
    """
    Generic pattern for a single required claim:
    - Leaf 1 (critical): The answer explicitly states the claim (simple verify).
    - Leaf 2 (critical): At least one URL is provided for this claim.
    - Leaf 3 (critical): The claim is supported by the cited URL(s).
    """
    seq = evaluator.add_sequential(
        id=base_id,
        desc=f"Verification for: {stated_desc}",
        parent=parent,
        critical=True,
    )

    # 1) Stated in answer
    stated_leaf = evaluator.add_leaf(
        id=f"{base_id}_Stated",
        desc=stated_desc,
        parent=seq,
        critical=True
    )
    await evaluator.verify(
        claim=stated_claim,
        node=stated_leaf,
        additional_instruction=stated_additional_instruction
    )

    # 2) URL present
    has_urls_leaf = evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"{base_id}_URL_Provided",
        desc="Provides at least one supporting URL reference in the answer for this claim.",
        parent=seq,
        critical=True
    )

    # 3) Supported by cited URL(s)
    supported_leaf = evaluator.add_leaf(
        id=f"{base_id}_Supported_By_URL",
        desc=support_desc,
        parent=seq,
        critical=True
    )
    await evaluator.verify(
        claim=support_claim,
        node=supported_leaf,
        sources=urls,
        additional_instruction=support_additional_instruction
    )


async def build_required_claims(
    evaluator: Evaluator,
    parent,
    data: SMAEvaluationExtraction,
) -> None:
    """
    Build the 'Required_Claims_With_Citations' parallel subtree, each claim as a sequential gate:
    - Stated in answer
    - URL present for the claim
    - Supported by the cited URL(s)
    Also includes a final credibility check over all provided URLs.
    """
    therapy = data.therapy or TherapyInfo()
    claims = data.claims or RequiredClaims()

    # Per-claim URL selection with fallback to any/all
    fda_urls = _merge_urls(claims.fda_approval_urls, data.all_urls)
    indication_urls = _merge_urls(claims.indication_urls, data.all_urls)
    price_urls = _merge_urls(claims.price_urls, data.all_urls)
    most_expensive_urls = _merge_urls(claims.most_expensive_urls, data.all_urls)
    vector_urls = _merge_urls(claims.vector_urls, data.all_urls)
    admin_urls = _merge_urls(claims.administration_urls, data.all_urls)

    required_node = evaluator.add_parallel(
        id="Required_Claims_With_Citations",
        desc="For each required claim, state it and provide at least one supporting credible URL reference.",
        parent=parent,
        critical=True,
    )

    # 1) FDA Approval Date: May 24, 2019
    await _build_claim_seq_stated_url_supported(
        evaluator=evaluator,
        parent=required_node,
        base_id="FDA_Approval_Date_May_24_2019",
        stated_desc="States that FDA approval occurred on May 24, 2019.",
        stated_claim="In the provided answer text, it explicitly states that the FDA approved the therapy on May 24, 2019.",
        support_desc="The FDA approval date (May 24, 2019) is supported by cited source(s).",
        support_claim=f"The FDA approved {_therapy_label(therapy.therapy_name)} on May 24, 2019.",
        urls=fda_urls,
        stated_additional_instruction="Allow typical date formatting variations (e.g., 'May 24, 2019', '2019-05-24'). The statement must be explicit in the answer text.",
        support_additional_instruction="Look for an explicit approval date on the cited page. Accept synonymous phrasing like 'approved on May 24, 2019' or 'approval dated May 24, 2019'."
    )

    # 2) Indication: SMA
    await _build_claim_seq_stated_url_supported(
        evaluator=evaluator,
        parent=required_node,
        base_id="SMA_Indication",
        stated_desc="States the therapy is indicated for spinal muscular atrophy (SMA).",
        stated_claim="In the provided answer text, it explicitly states that the therapy is indicated for spinal muscular atrophy (SMA).",
        support_desc="The SMA indication is supported by cited source(s).",
        support_claim=f"{_therapy_label(therapy.therapy_name)} is indicated for the treatment of spinal muscular atrophy (SMA).",
        urls=indication_urls,
        stated_additional_instruction="Accept the full term 'spinal muscular atrophy' or the abbreviation 'SMA'.",
        support_additional_instruction="Verify that the page indicates use for SMA (e.g., label, indication statement, or product description)."
    )

    # 3) Price at approval: $2.125 million
    await _build_claim_seq_stated_url_supported(
        evaluator=evaluator,
        parent=required_node,
        base_id="Price_2_125_Million_At_Approval",
        stated_desc="States the price at the time of approval was $2.125 million.",
        stated_claim="In the provided answer text, it explicitly states that the therapy's price at the time of approval was $2.125 million (USD).",
        support_desc="The $2.125 million price at approval is supported by cited source(s).",
        support_claim=f"At the time of its FDA approval in 2019, the list price of {_therapy_label(therapy.therapy_name)} was $2.125 million (USD).",
        urls=price_urls,
        stated_additional_instruction="Accept equivalent numeric forms such as '$2,125,000' or '$2.125 million'. The statement must clearly pertain to the time of approval.",
        support_additional_instruction="Check the cited page(s) mention the $2.125 million (USD) price at or around the time of approval."
    )

    # 4) Most expensive at the time
    await _build_claim_seq_stated_url_supported(
        evaluator=evaluator,
        parent=required_node,
        base_id="Most_Expensive_Drug_At_Time",
        stated_desc="States it was the most expensive drug ever approved at the time (2019).",
        stated_claim="In the provided answer text, it explicitly states that, at the time of approval in 2019, this therapy was the most expensive drug ever approved.",
        support_desc="The 'most expensive drug at the time' claim is supported by cited source(s).",
        support_claim=f"At the time of its approval in 2019, {_therapy_label(therapy.therapy_name)} was reported as the most expensive drug ever approved.",
        urls=most_expensive_urls,
        stated_additional_instruction="Allow synonymous phrasings like 'world's most expensive drug at launch' or 'most expensive medicine at the time of approval'.",
        support_additional_instruction="Look for supporting language on the cited page recognizing it as the most expensive drug at that time (2019)."
    )

    # 5) Viral vector serotype
    # Stated: ensure the answer names the serotype; Supported: verify that serotype with URLs
    vector_seq = evaluator.add_sequential(
        id="Viral_Vector_Serotype",
        desc="Specifies the viral vector technology serotype used and supports it with URL(s).",
        parent=required_node,
        critical=True
    )
    # 5a) Stated in answer (existence check)
    evaluator.add_custom_node(
        result=bool(claims.viral_vector_serotype and claims.viral_vector_serotype.strip()),
        id="Viral_Vector_Serotype_Stated",
        desc="The answer explicitly states the viral vector serotype used by the therapy (e.g., AAV9).",
        parent=vector_seq,
        critical=True
    )
    # 5b) URL present
    evaluator.add_custom_node(
        result=len(vector_urls) > 0,
        id="Viral_Vector_Serotype_URL_Provided",
        desc="Provides at least one supporting URL reference for the viral vector serotype.",
        parent=vector_seq,
        critical=True
    )
    # 5c) Supported by cited URLs
    vector_supported = evaluator.add_leaf(
        id="Viral_Vector_Serotype_Supported_By_URL",
        desc="The viral vector serotype is supported by the cited source(s).",
        parent=vector_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The viral vector serotype used by {_therapy_label(therapy.therapy_name)} is {_safe(claims.viral_vector_serotype, 'AAV9 or the correct official serotype')}."
              f" Accept equivalent phrasing like 'adeno-associated virus serotype 9 (AAV9)'.",
        node=vector_supported,
        sources=vector_urls,
        additional_instruction="Confirm the serotype on the cited page. Allow brand/generic naming differences and acronym expansions."
    )

    # 6) Administration method and frequency
    admin_seq = evaluator.add_sequential(
        id="Administration_Method_And_Frequency",
        desc="Specifies the administration route and dosing frequency, with supporting URL(s).",
        parent=required_node,
        critical=True
    )
    # 6a) Stated in answer: both route and frequency present
    evaluator.add_custom_node(
        result=_has_route_and_freq(claims.administration_method),
        id="Administration_Method_And_Frequency_Stated",
        desc="The answer explicitly describes both the dosing frequency (e.g., single-dose/one-time) and the route (e.g., IV/intravenous/intrathecal).",
        parent=admin_seq,
        critical=True
    )
    # 6b) URL present
    evaluator.add_custom_node(
        result=len(admin_urls) > 0,
        id="Administration_Method_And_Frequency_URL_Provided",
        desc="Provides at least one supporting URL reference for the administration method and frequency.",
        parent=admin_seq,
        critical=True
    )
    # 6c) Supported by cited URLs
    admin_supported = evaluator.add_leaf(
        id="Administration_Method_And_Frequency_Supported_By_URL",
        desc="The administration method and frequency are supported by the cited source(s).",
        parent=admin_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"This therapy is administered as {_safe(claims.administration_method, 'a single-dose intravenous (IV) infusion')}.",
        node=admin_supported,
        sources=admin_urls,
        additional_instruction="Accept synonymous phrasing for the same concept (e.g., 'one-time IV infusion', 'single-dose intravenous infusion')."
    )

    # 7) Reference source credibility checks (all URLs)
    # Build a small sequential gate: (a) at least one URL exists (b) each URL is from a credible source.
    cred_seq = evaluator.add_sequential(
        id="Reference_Source_Credibility",
        desc="All provided URL references are from credible sources (FDA, peer-reviewed journals, or official company announcements).",
        parent=required_node,
        critical=True
    )

    all_urls = _merge_urls(
        therapy.identification_urls,
        claims.fda_approval_urls,
        claims.indication_urls,
        claims.price_urls,
        claims.most_expensive_urls,
        claims.vector_urls,
        claims.administration_urls,
        data.all_urls,
    )

    evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id="Reference_Source_Credibility_URLs_Present",
        desc="At least one URL reference is present in the answer overall.",
        parent=cred_seq,
        critical=True
    )

    cred_parallel = evaluator.add_parallel(
        id="Reference_Source_Credibility_Checks",
        desc="Each referenced URL is a credible source (FDA, peer-reviewed journal, or official manufacturer announcement/press release).",
        parent=cred_seq,
        critical=True
    )

    # Limit to first 12 URLs to keep verification tractable
    for idx, url in enumerate(all_urls[:12]):
        leaf = evaluator.add_leaf(
            id=f"Credibility_URL_{idx+1}",
            desc=f"URL #{idx+1} is from a credible source: {url}",
            parent=cred_parallel,
            critical=True
        )
        await evaluator.verify(
            claim="This URL is a credible source (FDA/‘.fda.gov’, a peer‑reviewed medical journal article, "
                  "or an official pharmaceutical company page/press release/announcement).",
            node=leaf,
            sources=url,
            additional_instruction="Use the domain and page context to decide credibility. Manufacturer pages include official press releases or product pages from the company (e.g., Novartis/AveXis)."
        )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the SMA gene therapy (2019 FDA approval) task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # We will build a critical sequential main node under root
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_sma_therapy_info(),
        template_class=SMAEvaluationExtraction,
        extraction_name="sma_therapy_extraction"
    )

    # Build main critical sequential node (mirror of rubric root)
    main = evaluator.add_sequential(
        id="Gene_Therapy_Identification_and_Verification",
        desc="Identify the FDA-approved (May 24, 2019) SMA gene therapy and verify required attributes with credible citations.",
        parent=root,
        critical=True
    )

    # 1) Therapy identification block
    await build_therapy_identification(evaluator, main, extracted)

    # 2) Required claims with citations
    await build_required_claims(evaluator, main, extracted)

    return evaluator.get_summary()