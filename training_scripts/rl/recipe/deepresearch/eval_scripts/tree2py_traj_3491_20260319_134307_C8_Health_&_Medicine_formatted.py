import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "healthcare_guidelines_analysis"
TASK_DESCRIPTION = """
Provide a comprehensive analysis of healthcare worker influenza management and dietary guidelines by answering the following:

Part 1 - Pennsylvania Requirements:
What is the Pennsylvania state statute that requires healthcare facilities to require documentation of annual influenza vaccination for employees? Provide the complete statutory citation and explain what this law requires.

Part 2 - Virginia Requirements:
Does Virginia have a similar state statute requiring hospitals to ensure employee influenza vaccination? Explain the regulatory status and what guidance Virginia healthcare workers follow regarding influenza vaccination.

Part 3 - CDC Influenza Guidelines:
According to current CDC guidelines:
- How long is influenza most contagious, and what is the total potential contagious period?
- When can the general population return to work or normal activities after flu?
- What are the specific return-to-work criteria for healthcare workers with influenza?
- What does "fever-free" mean in the context of these guidelines?

Part 4 - 2025-2030 Dietary Guidelines:
Regarding the 2025-2030 Dietary Guidelines for Americans:
- When were these guidelines released and by which federal agencies?
- What is the new protein recommendation for adults in grams per kilogram of body weight per day?
- How much of an increase does this represent compared to previous guidelines?
- What are three key recommendations or changes in these guidelines?

For each part, provide reference URLs from authoritative sources (government websites, CDC, USDA, HHS, state health departments) that support your answers.
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PAExtraction(BaseModel):
    citation: Optional[str] = None
    requirement_summary: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class VAExtraction(BaseModel):
    statute_presence: Optional[str] = None  # e.g., "present" or "absent" (free-text allowed)
    regulatory_status: Optional[str] = None
    guidance_followed: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CDCExtraction(BaseModel):
    most_contagious_window: Optional[str] = None
    total_contagious_period: Optional[str] = None
    return_to_activities_public: Optional[str] = None
    return_to_work_hcp: Optional[str] = None
    fever_free_definition: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class DGAExtraction(BaseModel):
    release_date: Optional[str] = None
    issuing_agencies: List[str] = Field(default_factory=list)
    protein_g_per_kg: Optional[str] = None
    protein_change_vs_prior: Optional[str] = None
    key_recommendation_1: Optional[str] = None
    key_recommendation_2: Optional[str] = None
    key_recommendation_3: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class FullExtraction(BaseModel):
    pa: Optional[PAExtraction] = None
    va: Optional[VAExtraction] = None
    cdc: Optional[CDCExtraction] = None
    dga: Optional[DGAExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_full() -> str:
    return """
    Extract structured information for four parts from the answer text. Extract ONLY what is explicitly present in the answer; if something is missing, return null (or an empty array for URLs).

    Part 1 - Pennsylvania (PA) Requirements:
    - pa.citation: The complete statutory or regulatory citation string that contains the requirement for healthcare facilities to require documentation of annual influenza vaccination for employees (e.g., include title/chapter/section numbers).
    - pa.requirement_summary: A concise summary of what the cited PA provision requires (who must comply, documentation/annual vaccination requirement).
    - pa.urls: All URLs provided to support the PA law/requirement.

    Part 2 - Virginia (VA) Requirements:
    - va.statute_presence: The stated presence/absence of a statewide VA statute/regulation mandating hospitals/healthcare facilities to ensure employee influenza vaccination (free-text allowed; e.g., "absent", "present", "no statewide mandate").
    - va.regulatory_status: The explanation of Virginia's regulatory status in the answer (e.g., statutory/regulatory vs facility policy vs guidance).
    - va.guidance_followed: The guidance followed in practice by VA healthcare workers/facilities per the answer (e.g., CDC/ACIP, VDH materials).
    - va.urls: All URLs provided for the Virginia section.

    Part 3 - CDC Influenza Guidelines:
    - cdc.most_contagious_window: The CDC-described timeframe when influenza is most contagious.
    - cdc.total_contagious_period: The CDC-described total potential contagious period (start/end relative to symptom onset).
    - cdc.return_to_activities_public: CDC return-to-work/normal-activities criteria for the general population.
    - cdc.return_to_work_hcp: CDC (or CDC-linked) return-to-work criteria specific to healthcare personnel.
    - cdc.fever_free_definition: The CDC definition of "fever-free" (include any time requirement and whether fever-reducing meds must not be used).
    - cdc.urls: All CDC URLs provided.

    Part 4 - 2025–2030 Dietary Guidelines for Americans (DGA):
    - dga.release_date: The release date (month/day/year or month-year as presented).
    - dga.issuing_agencies: The agencies that issued the guidelines (list names exactly as stated).
    - dga.protein_g_per_kg: The adult protein recommendation in grams per kilogram body weight per day.
    - dga.protein_change_vs_prior: How this differs from the prior guideline(s), including baseline used and the magnitude of increase.
    - dga.key_recommendation_1: A key recommendation/change stated.
    - dga.key_recommendation_2: A second key recommendation/change.
    - dga.key_recommendation_3: A third key recommendation/change.
    - dga.urls: All URLs provided for the DGA section (prefer dietaryguidelines.gov, usda.gov, hhs.gov).

    IMPORTANT:
    - Extract URLs exactly as written in the answer, including markdown links (extract the actual URL).
    - Do not fabricate values; use null for any missing field.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _mk_additional_instruction(base: str, urls: List[str], require_authoritative: bool = False,
                               authoritative_hint: Optional[str] = None) -> str:
    parts = [base.strip()] if base else []
    if require_authoritative and authoritative_hint:
        parts.append(f"Treat a page as authoritative only if its URL domain is one of: {authoritative_hint}.")
    parts.append("If the provided URL(s) do not explicitly support the claim, mark as Not Supported.")
    if not urls:
        parts.append("No source URLs were provided. You must judge this claim as Not Supported due to lack of source evidence.")
    return " ".join(parts)


def _presence_label(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()
    if any(x in t for x in ["absent", "no statewide", "no state", "does not have", "none"]):
        return "absent"
    if any(x in t for x in ["present", "yes", "has statewide", "does have"]):
        return "present"
    return None


def _agencies_to_str(agencies: List[str]) -> str:
    clean = [a.strip() for a in agencies if a and a.strip()]
    return ", ".join(clean) if clean else ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_pa(evaluator: Evaluator, parent_node, data: Optional[PAExtraction]) -> None:
    pa_node = evaluator.add_parallel(
        id="Pennsylvania_Requirements",
        desc="Identify the Pennsylvania legal authority requiring healthcare facilities to require documentation of annual influenza vaccination for employees; provide full citation, explain requirements, and cite authoritative source(s).",
        parent=parent_node,
        critical=True
    )

    citation = (data.citation if data else None) or ""
    req_summary = (data.requirement_summary if data else None) or ""
    urls = _normalize_urls(data.urls if data else [])

    # PA_Statute_Citation
    node_cite = evaluator.add_leaf(
        id="PA_Statute_Citation",
        desc="Provide the complete Pennsylvania statutory (or regulatory, if applicable) citation that contains the influenza vaccination documentation requirement for healthcare facility employees.",
        parent=pa_node,
        critical=True
    )
    claim_cite = (
        f"The provided authoritative page(s) contain or clearly identify the Pennsylvania provision cited as '{citation}', "
        f"and it is the section that addresses healthcare facility employee influenza vaccination documentation."
    )
    add_ins_cite = _mk_additional_instruction(
        base="Verify that the exact or clearly equivalent citation string appears on the page (title/chapter/section), and that it relates to influenza vaccination documentation for healthcare facility employees.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_cite, node=node_cite, sources=urls, additional_instruction=add_ins_cite)

    # PA_Law_What_It_Requires
    node_req = evaluator.add_leaf(
        id="PA_Law_What_It_Requires",
        desc="Accurately explain what the cited Pennsylvania provision requires (who must comply, what documentation/vaccination requirement applies, and the annual nature of the requirement).",
        parent=pa_node,
        critical=True
    )
    claim_req = (
        f"According to the provided source page(s), the cited Pennsylvania provision '{citation}' is accurately summarized as: {req_summary}. "
        f"This provision requires healthcare facilities to require documentation of annual influenza vaccination for employees (or a permitted alternative such as declination/exemption) each flu season."
    )
    add_ins_req = _mk_additional_instruction(
        base="Confirm the summary is accurate and that the page explicitly states an annual influenza vaccination documentation requirement for healthcare facility employees.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_req, node=node_req, sources=urls, additional_instruction=add_ins_req)

    # PA_Authoritative_URLs
    node_auth = evaluator.add_leaf(
        id="PA_Authoritative_URLs",
        desc="Provide at least one authoritative URL supporting the citation and the described requirement (e.g., official Pennsylvania code/statute site or PA government/health department source).",
        parent=pa_node,
        critical=True
    )
    claim_auth = (
        "This page is an authoritative Pennsylvania government or official code/statute source supporting the influenza vaccination documentation requirement "
        "(acceptable domains include: pa.gov, health.pa.gov, pacodeandbulletin.gov, legis.state.pa.us or other official Pennsylvania government sites)."
    )
    add_ins_auth = _mk_additional_instruction(
        base="Check the URL domain and page content for authoritativeness and relevance to the PA influenza vaccination documentation requirement.",
        urls=urls,
        require_authoritative=True,
        authoritative_hint=".pa.gov, health.pa.gov, pacodeandbulletin.gov, legis.state.pa.us"
    )
    await evaluator.verify(claim=claim_auth, node=node_auth, sources=urls, additional_instruction=add_ins_auth)


async def verify_va(evaluator: Evaluator, parent_node, data: Optional[VAExtraction]) -> None:
    va_node = evaluator.add_parallel(
        id="Virginia_Requirements",
        desc="Determine whether Virginia has a similar state statute requiring hospitals to ensure employee influenza vaccination; explain regulatory status and what guidance VA healthcare workers follow; include authoritative source(s).",
        parent=parent_node,
        critical=True
    )

    presence_text = (data.statute_presence if data else None) or ""
    presence = _presence_label(presence_text)
    reg_status = (data.regulatory_status if data else None) or ""
    guidance = (data.guidance_followed if data else None) or ""
    urls = _normalize_urls(data.urls if data else [])

    # VA_Statute_Presence_Or_Absence
    node_presence = evaluator.add_leaf(
        id="VA_Statute_Presence_Or_Absence",
        desc="State whether Virginia has a state statute (or binding statewide regulation) requiring hospitals/healthcare facilities to ensure employee influenza vaccination, and characterize it correctly (present vs absent).",
        parent=va_node,
        critical=True
    )
    if presence == "absent":
        claim_presence = (
            "Virginia does not have a statewide statute or binding statewide regulation that mandates hospitals or healthcare facilities to ensure employee influenza vaccination."
        )
    elif presence == "present":
        claim_presence = (
            "Virginia has a statewide statute or binding statewide regulation that mandates hospitals or healthcare facilities to ensure employee influenza vaccination."
        )
    else:
        claim_presence = (
            f"The answer states the following about the presence/absence of a statewide Virginia mandate: '{presence_text}'. "
            "The provided page(s) confirm this characterization."
        )
    add_ins_presence = _mk_additional_instruction(
        base="Determine from the provided page(s) whether Virginia has a binding statewide statute/regulation mandating employee influenza vaccination in healthcare facilities.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_presence, node=node_presence, sources=urls, additional_instruction=add_ins_presence)

    # VA_Regulatory_Status_Explanation
    node_status = evaluator.add_leaf(
        id="VA_Regulatory_Status_Explanation",
        desc="Explain the regulatory status in Virginia (e.g., whether requirements are statutory/regulatory vs facility policy vs nonbinding guidance) in a way consistent with cited sources.",
        parent=va_node,
        critical=True
    )
    claim_status = (
        f"According to the provided source page(s), the regulatory status in Virginia is accurately described as: {reg_status}."
    )
    add_ins_status = _mk_additional_instruction(
        base="Verify that the explanation of Virginia's regulatory status matches what is stated on the cited pages.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_status, node=node_status, sources=urls, additional_instruction=add_ins_status)

    # VA_Guidance_Healthcare_Workers_Follow
    node_guidance = evaluator.add_leaf(
        id="VA_Guidance_Healthcare_Workers_Follow",
        desc="Identify and summarize the main influenza vaccination guidance VA healthcare workers/facilities follow in practice (e.g., CDC/ACIP recommendations and/or Virginia Department of Health materials), consistent with cited sources.",
        parent=va_node,
        critical=True
    )
    claim_guidance = (
        f"The provided page(s) support that, in practice, Virginia healthcare workers/facilities follow this guidance: {guidance}."
    )
    add_ins_guidance = _mk_additional_instruction(
        base="Confirm that the pages explicitly describe the guidance VA healthcare workers/facilities follow (e.g., CDC/ACIP, VDH).",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_guidance, node=node_guidance, sources=urls, additional_instruction=add_ins_guidance)

    # VA_Authoritative_URLs
    node_auth = evaluator.add_leaf(
        id="VA_Authoritative_URLs",
        desc="Provide at least one authoritative URL supporting the Virginia mandate status and the guidance described (e.g., VDH, Virginia administrative code, CDC state law database).",
        parent=va_node,
        critical=True
    )
    claim_auth = (
        "This page is an authoritative government or CDC source supporting Virginia's mandate status and guidance (acceptable domains include: virginia.gov, vdh.virginia.gov, law.lis.virginia.gov, cdc.gov)."
    )
    add_ins_auth = _mk_additional_instruction(
        base="Check the URL domain and page content for authoritativeness and relevance to Virginia influenza vaccination policy/guidance.",
        urls=urls,
        require_authoritative=True,
        authoritative_hint="virginia.gov, vdh.virginia.gov, law.lis.virginia.gov, cdc.gov"
    )
    await evaluator.verify(claim=claim_auth, node=node_auth, sources=urls, additional_instruction=add_ins_auth)


async def verify_cdc(evaluator: Evaluator, parent_node, data: Optional[CDCExtraction]) -> None:
    cdc_node = evaluator.add_parallel(
        id="CDC_Flu_Guidelines",
        desc="Provide current CDC guidance on contagiousness and return-to-activity/work criteria (general public and healthcare personnel), including the definition of 'fever-free', with CDC URL(s).",
        parent=parent_node,
        critical=True
    )

    most_cont = (data.most_contagious_window if data else None) or ""
    total_period = (data.total_contagious_period if data else None) or ""
    rt_public = (data.return_to_activities_public if data else None) or ""
    rt_hcp = (data.return_to_work_hcp if data else None) or ""
    fever_def = (data.fever_free_definition if data else None) or ""
    urls = _normalize_urls(data.urls if data else [])

    # Flu_Most_Contagious_Window
    node_most = evaluator.add_leaf(
        id="Flu_Most_Contagious_Window",
        desc="State the CDC-described time window when influenza is most contagious, including an explicit duration/timeframe.",
        parent=cdc_node,
        critical=True
    )
    claim_most = f"According to the CDC page(s), influenza is most contagious as stated: {most_cont}."
    add_ins_most = _mk_additional_instruction(
        base="Check the CDC page(s) for an explicit description of when flu is most contagious.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_most, node=node_most, sources=urls, additional_instruction=add_ins_most)

    # Flu_Total_Potential_Contagious_Period
    node_total = evaluator.add_leaf(
        id="Flu_Total_Potential_Contagious_Period",
        desc="State the CDC-described total potential contagious period for influenza (start/end relative to symptom onset), including explicit timeframe(s).",
        parent=cdc_node,
        critical=True
    )
    claim_total = f"The CDC page(s) describe the overall potential contagious period as: {total_period}."
    add_ins_total = _mk_additional_instruction(
        base="Confirm the CDC timeframe for when people can spread flu, from before symptom onset to after.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_total, node=node_total, sources=urls, additional_instruction=add_ins_total)

    # CDC_Return_To_Activities_General_Public
    node_pub = evaluator.add_leaf(
        id="CDC_Return_To_Activities_General_Public",
        desc="Provide CDC return-to-work/normal-activities criteria for the general population after influenza, including any stated minimum time guidance and symptom/fever criteria.",
        parent=cdc_node,
        critical=True
    )
    claim_pub = f"The CDC return-to-activities guidance for the general population is: {rt_public}."
    add_ins_pub = _mk_additional_instruction(
        base="Verify the general public return-to-work/activities criteria on CDC pages, including any time thresholds and symptom/fever criteria.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_pub, node=node_pub, sources=urls, additional_instruction=add_ins_pub)

    # CDC_Return_To_Work_Healthcare_Personnel
    node_hcp = evaluator.add_leaf(
        id="CDC_Return_To_Work_Healthcare_Personnel",
        desc="Provide CDC (or CDC-linked) return-to-work criteria specific to healthcare personnel with influenza, including any stated time-from-onset and symptom/fever criteria.",
        parent=cdc_node,
        critical=True
    )
    claim_hcp = f"The CDC (or CDC-linked) return-to-work criteria for healthcare personnel with influenza are: {rt_hcp}."
    add_ins_hcp = _mk_additional_instruction(
        base="Verify healthcare personnel-specific return-to-work criteria on CDC or CDC-linked HCP guidance pages.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_hcp, node=node_hcp, sources=urls, additional_instruction=add_ins_hcp)

    # CDC_Fever_Free_Definition
    node_fever = evaluator.add_leaf(
        id="CDC_Fever_Free_Definition",
        desc="Define what 'fever-free' means per CDC guidance, including any minimum duration requirement and whether fever-reducing medications must not be used.",
        parent=cdc_node,
        critical=True
    )
    claim_fever = f"Per CDC guidance, 'fever-free' is defined as: {fever_def}."
    add_ins_fever = _mk_additional_instruction(
        base="Check the CDC pages for the explicit definition of 'fever-free' including time and medication criteria.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_fever, node=node_fever, sources=urls, additional_instruction=add_ins_fever)

    # CDC_Authoritative_URLs
    node_auth = evaluator.add_leaf(
        id="CDC_Authoritative_URLs",
        desc="Provide at least one CDC URL (cdc.gov) that supports the contagiousness and return-to-activity/work criteria stated.",
        parent=cdc_node,
        critical=True
    )
    claim_auth = "This page is an authoritative CDC source (domain cdc.gov) supporting the influenza contagiousness and return-to-activity/work criteria."
    add_ins_auth = _mk_additional_instruction(
        base="Confirm that the URL domain is cdc.gov and that the page content is relevant to the stated CDC guidance.",
        urls=urls,
        require_authoritative=True,
        authoritative_hint="cdc.gov"
    )
    await evaluator.verify(claim=claim_auth, node=node_auth, sources=urls, additional_instruction=add_ins_auth)


async def verify_dga(evaluator: Evaluator, parent_node, data: Optional[DGAExtraction]) -> None:
    dga_node = evaluator.add_parallel(
        id="Dietary_Guidelines_2025_2030",
        desc="Provide release details and key content from the 2025–2030 Dietary Guidelines for Americans, including protein recommendation, comparison to previous guidelines, and three key recommendations/changes, supported by authoritative URLs.",
        parent=parent_node,
        critical=True
    )

    release_date = (data.release_date if data else None) or ""
    agencies_str = _agencies_to_str((data.issuing_agencies if data else []) or [])
    protein_gkg = (data.protein_g_per_kg if data else None) or ""
    protein_change = (data.protein_change_vs_prior if data else None) or ""
    kr1 = (data.key_recommendation_1 if data else None) or ""
    kr2 = (data.key_recommendation_2 if data else None) or ""
    kr3 = (data.key_recommendation_3 if data else None) or ""
    urls = _normalize_urls(data.urls if data else [])

    # DGA_Release_Date
    node_rel = evaluator.add_leaf(
        id="DGA_Release_Date",
        desc="State when the 2025–2030 Dietary Guidelines for Americans were released (date/month-year as given by authoritative sources).",
        parent=dga_node,
        critical=True
    )
    claim_rel = f"The 2025–2030 Dietary Guidelines for Americans were released on: {release_date}."
    add_ins_rel = _mk_additional_instruction(
        base="Verify the release date on dietaryguidelines.gov or USDA/HHS pages.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_rel, node=node_rel, sources=urls, additional_instruction=add_ins_rel)

    # DGA_Issuing_Agencies
    node_ag = evaluator.add_leaf(
        id="DGA_Issuing_Agencies",
        desc="Identify which federal agencies issued the 2025–2030 Dietary Guidelines for Americans.",
        parent=dga_node,
        critical=True
    )
    claim_ag = f"The 2025–2030 DGA were issued by: {agencies_str}."
    add_ins_ag = _mk_additional_instruction(
        base="Confirm the issuing agencies (typically USDA and HHS) on authoritative pages.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_ag, node=node_ag, sources=urls, additional_instruction=add_ins_ag)

    # DGA_Protein_Recommendation_g_per_kg
    node_pro = evaluator.add_leaf(
        id="DGA_Protein_Recommendation_g_per_kg",
        desc="State the adult protein recommendation in grams per kilogram of body weight per day as described by the 2025–2030 guidelines (include the quantitative value/range).",
        parent=dga_node,
        critical=True
    )
    claim_pro = f"The 2025–2030 DGA adult protein recommendation is: {protein_gkg} grams per kilogram of body weight per day."
    add_ins_pro = _mk_additional_instruction(
        base="Verify the stated grams-per-kilogram-per-day value on the official guideline pages.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_pro, node=node_pro, sources=urls, additional_instruction=add_ins_pro)

    # DGA_Protein_Change_vs_Prior
    node_delta = evaluator.add_leaf(
        id="DGA_Protein_Change_vs_Prior",
        desc="Quantify how the new protein recommendation differs from the prior guideline(s) (include the prior baseline used and the computed/claimed increase).",
        parent=dga_node,
        critical=True
    )
    claim_delta = f"The change vs prior guidelines is accurately stated as: {protein_change}."
    add_ins_delta = _mk_additional_instruction(
        base="Verify both the prior baseline and the magnitude of increase using authoritative sources.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_delta, node=node_delta, sources=urls, additional_instruction=add_ins_delta)

    # DGA_Key_Recommendation_1
    node_k1 = evaluator.add_leaf(
        id="DGA_Key_Recommendation_1",
        desc="Provide one key recommendation/change from the 2025–2030 guidelines, stated clearly and distinctly.",
        parent=dga_node,
        critical=True
    )
    claim_k1 = f"Key recommendation/change: {kr1}."
    add_ins_k1 = _mk_additional_instruction(
        base="Confirm that this recommendation/change appears in the 2025–2030 guidelines.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_k1, node=node_k1, sources=urls, additional_instruction=add_ins_k1)

    # DGA_Key_Recommendation_2
    node_k2 = evaluator.add_leaf(
        id="DGA_Key_Recommendation_2",
        desc="Provide a second key recommendation/change from the 2025–2030 guidelines, stated clearly and distinctly.",
        parent=dga_node,
        critical=True
    )
    claim_k2 = f"Key recommendation/change: {kr2}."
    add_ins_k2 = _mk_additional_instruction(
        base="Confirm that this recommendation/change appears in the 2025–2030 guidelines.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_k2, node=node_k2, sources=urls, additional_instruction=add_ins_k2)

    # DGA_Key_Recommendation_3
    node_k3 = evaluator.add_leaf(
        id="DGA_Key_Recommendation_3",
        desc="Provide a third key recommendation/change from the 2025–2030 guidelines, stated clearly and distinctly.",
        parent=dga_node,
        critical=True
    )
    claim_k3 = f"Key recommendation/change: {kr3}."
    add_ins_k3 = _mk_additional_instruction(
        base="Confirm that this recommendation/change appears in the 2025–2030 guidelines.",
        urls=urls,
        require_authoritative=False
    )
    await evaluator.verify(claim=claim_k3, node=node_k3, sources=urls, additional_instruction=add_ins_k3)

    # DGA_Authoritative_URLs
    node_auth = evaluator.add_leaf(
        id="DGA_Authoritative_URLs",
        desc="Provide at least one authoritative URL supporting the Dietary Guidelines claims (prefer dietaryguidelines.gov and/or USDA/HHS pages).",
        parent=dga_node,
        critical=True
    )
    claim_auth = "This page is an authoritative source for the 2025–2030 DGA (acceptable domains include: dietaryguidelines.gov, usda.gov, hhs.gov)."
    add_ins_auth = _mk_additional_instruction(
        base="Validate that the URL is an official Dietary Guidelines/USDA/HHS page and relevant to the stated claims.",
        urls=urls,
        require_authoritative=True,
        authoritative_hint="dietaryguidelines.gov, usda.gov, hhs.gov"
    )
    await evaluator.verify(claim=claim_auth, node=node_auth, sources=urls, additional_instruction=add_ins_auth)


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
    Evaluate the answer for the healthcare guidelines analysis task using the obj_task_eval framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level aggregation across the four parts
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

    # Top-level critical node reflecting the rubric's main analysis item
    analysis_node = evaluator.add_parallel(
        id="Healthcare_Guidelines_Analysis",
        desc="Answer all four parts (PA law, VA status, CDC flu guidance, 2025–2030 Dietary Guidelines) and support each part with authoritative URLs.",
        parent=root,
        critical=True
    )

    # Extract all required information in one pass
    extracted: FullExtraction = await evaluator.extract(
        prompt=prompt_extract_full(),
        template_class=FullExtraction,
        extraction_name="full_extraction"
    )

    # Build subtrees for each part
    await verify_pa(evaluator, analysis_node, extracted.pa if extracted else None)
    await verify_va(evaluator, analysis_node, extracted.va if extracted else None)
    await verify_cdc(evaluator, analysis_node, extracted.cdc if extracted else None)
    await verify_dga(evaluator, analysis_node, extracted.dga if extracted else None)

    return evaluator.get_summary()