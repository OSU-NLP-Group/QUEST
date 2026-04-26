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
TASK_ID = "tx_ag_primary_2026_candidate_identification"
TASK_DESCRIPTION = (
    "In the Texas Democratic primary for Attorney General scheduled for March 3, 2026, three candidates are competing "
    "for the nomination: Tony Box, Joe Jaworski, and Nathan Johnson. Identify which of these three candidates currently "
    "serves as a Texas State Senator AND has raised the highest total campaign funds among all three Democratic primary "
    "candidates. Provide the candidate's name and include reference URLs that verify both (1) their current service as a "
    "Texas State Senator and (2) their campaign finance data showing they have raised more total funds than the other two candidates."
)


ALLOWED_CANDIDATES = ["Tony Box", "Joe Jaworski", "Nathan Johnson"]


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CandidateEvidence(BaseModel):
    """
    Extract a single chosen candidate and grouped reference URLs from the answer.
    - candidate: exactly one candidate identified as meeting both criteria.
    - senator_urls: URLs used to support the claim that the candidate currently serves as a Texas State Senator.
    - finance_urls: URLs used to support the claim that the candidate has raised the highest total campaign funds among the three.
    """
    candidate: Optional[str] = None
    senator_urls: List[str] = Field(default_factory=list)
    finance_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_candidate_evidence() -> str:
    return """
    From the answer, extract the single candidate asserted to meet BOTH of the following:
    (1) currently serves as a Texas State Senator, and
    (2) has raised the highest total campaign funds among the three named candidates.

    Return the following fields:
    - candidate: the single candidate name exactly as presented in the answer text. It MUST be one of these names:
      "Tony Box", "Joe Jaworski", or "Nathan Johnson". If the answer mentions multiple candidates or is ambiguous,
      choose the one that the answer ultimately identifies as meeting both criteria; otherwise return null.
    - senator_urls: a list of all URLs PROVIDED IN THE ANSWER that the answer uses to support the senator-status claim
      for the selected candidate. Extract only actual URLs explicitly present in the answer (including markdown links).
    - finance_urls: a list of all URLs PROVIDED IN THE ANSWER that the answer uses to support the fundraising claim
      (i.e., that the selected candidate has raised the highest total among the three). Extract only actual URLs explicitly
      present in the answer.

    Important:
    - Do not infer or fabricate any URLs.
    - If a required set of URLs is not present, return an empty list for that field.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def canonicalize_candidate_name(name: Optional[str]) -> Optional[str]:
    """
    Map possibly formatted name variants to one of the canonical candidates.
    This is a tolerant match (ignores case, punctuation, titles, and middle initials).
    """
    if not name:
        return None
    s = name.lower()
    for target in ALLOWED_CANDIDATES:
        t = target.lower()
        t_first, t_last = t.split()
        has_first = t_first in s
        has_last = t_last in s
        if has_first and has_last:
            return target
    return None


def dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        u = (u or "").strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_candidate_identification_tree(
    evaluator: Evaluator,
    parent,
    extracted: CandidateEvidence,
) -> None:
    """
    Build the verification tree under the Candidate_Identification node according to rubric.
    Structure:
      Candidate_Identification (critical, sequential)
        ├─ Candidate_Name_Provided (critical, leaf/custom)
        └─ Verification_Evidence (critical, parallel)
            ├─ Evidence_Texas_State_Senator (critical, leaf)
            ├─ Evidence_Highest_Fundraising (critical, leaf)
            └─ Allowed_Source_URLs (critical, parallel)
                └─ Allowed_Source_URL_{i} (critical, leaf for each provided URL)
    """

    # Top-level sequential node (critical)
    cand_node = evaluator.add_sequential(
        id="Candidate_Identification",
        desc="Identify the correct candidate (among Tony Box, Joe Jaworski, Nathan Johnson) who both currently serves as a Texas State Senator and has raised the highest total campaign funds among the three, and provide reference URL(s) that verify both criteria using allowed sources.",
        parent=parent,
        critical=True,
    )

    # 1) Candidate_Name_Provided (critical)
    canonical = canonicalize_candidate_name(extracted.candidate)
    name_ok = canonical is not None
    evaluator.add_custom_node(
        result=name_ok,
        id="Candidate_Name_Provided",
        desc="Provides exactly one candidate name, and it is one of: Tony Box, Joe Jaworski, Nathan Johnson.",
        parent=cand_node,
        critical=True
    )

    # 2) Verification_Evidence (critical, parallel)
    evidence_node = evaluator.add_parallel(
        id="Verification_Evidence",
        desc="Provides reference URL evidence establishing both (1) current Texas State Senator status and (2) highest total campaign funds among the three, using only allowed sources.",
        parent=cand_node,
        critical=True
    )

    # Prepare URLs
    senator_urls = dedupe_urls(extracted.senator_urls or [])
    finance_urls = dedupe_urls(extracted.finance_urls or [])
    all_ref_urls = dedupe_urls(senator_urls + finance_urls)

    # 2.a) Evidence_Texas_State_Senator (critical leaf)
    senator_leaf = evaluator.add_leaf(
        id="Evidence_Texas_State_Senator",
        desc="Provides at least one reference URL that supports that the identified candidate currently serves as a Texas State Senator.",
        parent=evidence_node,
        critical=True
    )
    if not senator_urls or not canonical:
        # Fail early if missing URLs or candidate
        senator_leaf.score = 0.0
        senator_leaf.status = "failed"
    else:
        senator_claim = f"{canonical} currently serves as a Texas State Senator."
        await evaluator.verify(
            claim=senator_claim,
            node=senator_leaf,
            sources=senator_urls,
            additional_instruction=(
                "Verify that the webpage explicitly indicates that the named person is a current member of the Texas Senate "
                "(e.g., 'is a Texas state senator', 'member of the Texas Senate', 'State Senator for District X'). "
                "Allow minor naming variations and titles. The verification must be grounded in the provided URL(s)."
            ),
        )

    # 2.b) Evidence_Highest_Fundraising (critical leaf)
    funds_leaf = evaluator.add_leaf(
        id="Evidence_Highest_Fundraising",
        desc="Provides at least one reference URL to campaign finance data that supports the identified candidate has raised the highest total campaign funds among the three candidates (i.e., shows totals or an explicit comparison sufficient to establish 'highest among the three').",
        parent=evidence_node,
        critical=True
    )
    if not finance_urls or not canonical:
        funds_leaf.score = 0.0
        funds_leaf.status = "failed"
    else:
        finance_claim = (
            f"Among Tony Box, Joe Jaworski, and Nathan Johnson, {canonical} has raised the highest total campaign funds."
        )
        await evaluator.verify(
            claim=finance_claim,
            node=funds_leaf,
            sources=finance_urls,
            additional_instruction=(
                "Check the page for total fundraising/receipts/amount raised numbers or an explicit statement that the selected candidate "
                "leads in total funds among the three. Accept synonyms such as 'total raised', 'total receipts', 'contributions'. "
                "The claim is correct only if the source clearly indicates {canonical} > the other two."
            ),
        )

    # 2.c) Allowed_Source_URLs (critical, parallel)
    # All reference URLs must be from either:
    #  - Texas Tribune voter guide published on Jan 28, 2026 (updates OK if clearly the 2026 voter guide), or
    #  - Ballotpedia’s official page for the March 3, 2026 Democratic primary (for Texas Attorney General).
    allowed_parent = evaluator.add_parallel(
        id="Allowed_Source_URLs",
        desc="All reference URLs used for verification are from either the Texas Tribune voter guide published on Jan 28, 2026 or Ballotpedia’s official page for the March 3, 2026 Democratic primary.",
        parent=evidence_node,
        critical=True
    )

    # Build per-URL allowed-source leaves
    # If no URLs were provided at all, create one failing child to ensure correct gating.
    if not all_ref_urls:
        leaf = evaluator.add_leaf(
            id="Allowed_Source_URL_0",
            desc="Reference URL is from allowed sources (Texas Tribune 2026 voter guide or Ballotpedia page for the March 3, 2026 Democratic primary for Texas AG).",
            parent=allowed_parent,
            critical=True
        )
        leaf.score = 0.0
        leaf.status = "failed"
    else:
        claims_and_sources: List[Tuple[str, str, Any, Optional[str]]] = []
        for idx, url in enumerate(all_ref_urls):
            url_leaf = evaluator.add_leaf(
                id=f"Allowed_Source_URL_{idx+1}",
                desc="Reference URL is from allowed sources (Texas Tribune 2026 voter guide or Ballotpedia page for the March 3, 2026 Democratic primary for Texas AG).",
                parent=allowed_parent,
                critical=True
            )
            allowed_claim = (
                "This webpage is an allowed source: either (A) part of the Texas Tribune voter guide for the 2026 Texas primary "
                "that was originally published on January 28, 2026 (updates acceptable if it is clearly the 2026 voter guide), "
                "or (B) Ballotpedia’s official page for the March 3, 2026 Texas Democratic primary for Attorney General."
            )
            add_ins = (
                "Accept if the page is clearly within the Texas Tribune's 2026 Texas primary voter guide content (look for 'Voter guide', "
                "'2026', and publication text around January 28, 2026 or updated versions), OR if it is a Ballotpedia page specifically "
                "about the Texas Attorney General election Democratic primary on March 3, 2026. If the URL is from other sources or unrelated "
                "Ballotpedia/Texas Tribune pages, mark as not allowed."
            )
            claims_and_sources.append((allowed_claim, url, url_leaf, add_ins))

        # Run allowed-source checks in parallel
        await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the Texas AG Democratic primary candidate identification task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root can be parallel; main logic lives under the critical sequential child
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

    # 1) Extract structured info from answer
    extracted: CandidateEvidence = await evaluator.extract(
        prompt=prompt_extract_candidate_evidence(),
        template_class=CandidateEvidence,
        extraction_name="candidate_evidence",
    )

    # Optional: record normalized candidate for transparency
    evaluator.add_custom_info(
        info={"extracted_candidate_raw": extracted.candidate, "normalized_candidate": canonicalize_candidate_name(extracted.candidate)},
        info_type="extraction_notes",
        info_name="normalization"
    )

    # 2) Build verification tree according to rubric
    await build_candidate_identification_tree(evaluator, root, extracted)

    # 3) Return summary
    return evaluator.get_summary()