import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "stroke_medical_summary_2025"
TASK_DESCRIPTION = (
    "A 39-year-old patient experienced an ischemic subcortical stroke in the middle cerebral artery 9 months ago. "
    "They have been diagnosed with hypertension and are seeking comprehensive medical guidance. Prepare a detailed "
    "medical information summary that includes:\n\n"
    "1. Epidemiological and Risk Context:\n"
    "- The CDC 2024 report finding on stroke prevalence trends for young adults ages 18-44 (comparing 2011-2013 to 2020-2022)\n"
    "- Identification of the single most important modifiable risk factor for stroke\n"
    "- Explanation of patent foramen ovale (PFO) including its prevalence and relevance for young adult stroke patients\n\n"
    "2. Evidence-Based Prevention Plan:\n"
    "- The specific dietary pattern with strongest evidence for stroke prevention\n"
    "- The recommended daily exercise duration for stroke prevention\n\n"
    "3. Clinical Trial Eligibility:\n"
    "- Name and ClinicalTrials.gov identifier for the Phase 2B neural stem cell trial for chronic stroke (SuNR1se II)\n"
    "- Complete verification that the patient meets all three main eligibility criteria: age range, stroke type/location, and post-stroke time window\n\n"
    "4. 2025 Treatment Innovation:\n"
    "- Name and developing institution of a major stroke treatment breakthrough announced/published in 2025\n"
    "- Mechanism of how it works\n"
    "- Quantitative performance metrics comparing it to existing standard treatments\n\n"
    "Provide citations from reputable medical sources (2023-2025) for all information."
)

# Patient facts we will reference in verification (ground truth for the scenario)
PATIENT_FACTS = {
    "age_years": 39,
    "stroke_type_location": "ischemic subcortical MCA stroke",
    "months_since_stroke": 9,
    "has_hypertension": True,
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CDCTrend(BaseModel):
    percent_increase: Optional[str] = None  # e.g., "~15%", "14.6%", "about 15%"
    cdc_source_urls: List[str] = Field(default_factory=list)


class RiskFactor(BaseModel):
    risk_factor: Optional[str] = None  # expected "hypertension" or "high blood pressure"
    source_urls: List[str] = Field(default_factory=list)


class PFOSection(BaseModel):
    definition: Optional[str] = None
    prevalence: Optional[str] = None  # e.g., "~25%", "1 in 4"
    relevance: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class PreventionSection(BaseModel):
    diet_pattern: Optional[str] = None  # expected "Mediterranean diet"
    diet_sources: List[str] = Field(default_factory=list)
    exercise_daily_minutes: Optional[str] = None  # e.g., "30 minutes", "30"
    exercise_sources: List[str] = Field(default_factory=list)


class TrialSection(BaseModel):
    trial_name: Optional[str] = None  # expected "SuNR1se II"
    nct_id: Optional[str] = None      # e.g., "NCT0....."
    phase_description: Optional[str] = None  # expected "Phase 2B neural stem cell trial for chronic stroke"
    trial_sources: List[str] = Field(default_factory=list)  # should include ClinicalTrials.gov URL


class InnovationSection(BaseModel):
    name: Optional[str] = None
    institution: Optional[str] = None
    mechanism: Optional[str] = None
    quantitative_metrics: Optional[str] = None  # should include "~90% vs ~11%" if applicable
    sources: List[str] = Field(default_factory=list)


class CitationsSection(BaseModel):
    epidemiology_urls: List[str] = Field(default_factory=list)
    prevention_urls: List[str] = Field(default_factory=list)
    trial_urls: List[str] = Field(default_factory=list)
    innovation_urls: List[str] = Field(default_factory=list)
    all_urls: List[str] = Field(default_factory=list)


class MedicalSummaryExtraction(BaseModel):
    cdc_trend: Optional[CDCTrend] = None
    risk_factor: Optional[RiskFactor] = None
    pfo: Optional[PFOSection] = None
    prevention: Optional[PreventionSection] = None
    trial: Optional[TrialSection] = None
    innovation2025: Optional[InnovationSection] = None
    citations: Optional[CitationsSection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_medical_summary() -> str:
    return """
Extract the specific structured information from the answer to the stroke summary task.

Return a JSON object with this schema:
- cdc_trend:
  - percent_increase: string, the stated percent increase for CDC 2024 trend for ages 18–44 comparing 2011–2013 to 2020–2022 (e.g., "14.6%", "about 15%"); if not provided, null
  - cdc_source_urls: array of URL strings for the cited CDC source(s)
- risk_factor:
  - risk_factor: string, the single most important modifiable risk factor identified (expected "hypertension" / "high blood pressure")
  - source_urls: array of URLs cited for this claim
- pfo:
  - definition: string, the medical definition of patent foramen ovale
  - prevalence: string, the reported prevalence (expected "~25%" or "1 in 4")
  - relevance: string, the described clinical relevance for young adult stroke
  - source_urls: array of URLs cited for the PFO items
- prevention:
  - diet_pattern: string, the dietary pattern with the strongest evidence for stroke prevention (expected "Mediterranean diet")
  - diet_sources: array of URLs cited for the diet claim
  - exercise_daily_minutes: string, recommended daily exercise duration (e.g., "30 minutes")
  - exercise_sources: array of URLs cited for the exercise claim
- trial:
  - trial_name: string, expected "SuNR1se II"
  - nct_id: string, the ClinicalTrials.gov identifier (e.g., "NCT01234567")
  - phase_description: string, description like "Phase 2B neural stem cell trial for chronic stroke"
  - trial_sources: array of URLs cited for the trial (should include ClinicalTrials.gov)
- innovation2025:
  - name: string, the 2025 breakthrough name
  - institution: string, developing institution/organization
  - mechanism: string, how it works
  - quantitative_metrics: string, quantitative comparison vs standard (should include "~90% vs ~11%" if present)
  - sources: array of URLs cited for the innovation
- citations:
  - epidemiology_urls: array of URLs used for epidemiology/risk context
  - prevention_urls: array of URLs used for prevention plan
  - trial_urls: array of URLs used for trial/eligibility
  - innovation_urls: array of URLs used for innovation
  - all_urls: array with all citations across the answer (deduplicate if possible)

Rules:
- Extract only URLs explicitly present in the answer; keep full URLs with protocol.
- If a field is missing, set it to null (for strings) or [] (for lists).
- Do not invent values. If a numeric value is approximate, keep the text as shown (e.g., "about 15%").
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _uniq(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _flatten_urls(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        if lst:
            merged.extend(lst)
    return _uniq(merged)


def _safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_epidemiology_and_risk_context(
    evaluator: Evaluator,
    parent_node,
    extracted: MedicalSummaryExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Epidemiological_and_Risk_Context",
        desc="Includes CDC young-adult trend, most important modifiable risk factor, and PFO explanation (definition, prevalence, relevance).",
        parent=parent_node,
        critical=True
    )

    # CDC trend
    cdc_node = evaluator.add_leaf(
        id="CDC_Young_Adult_Trend_Finding",
        desc="States the CDC 2024 finding for ages 18–44 comparing 2011–2013 vs 2020–2022, including the approximate prevalence increase (~14.6–15%).",
        parent=node,
        critical=True
    )
    cdc_sources = extracted.cdc_trend.cdc_source_urls if extracted.cdc_trend and extracted.cdc_trend.cdc_source_urls else []
    cdc_pct = _safe(extracted.cdc_trend.percent_increase if extracted.cdc_trend else None)
    cdc_claim = (
        f"The CDC 2024 report on young adults (ages 18–44) states that stroke prevalence increased by approximately "
        f"{cdc_pct} when comparing 2011–2013 to 2020–2022."
    )
    await evaluator.verify(
        claim=cdc_claim,
        node=cdc_node,
        sources=cdc_sources,
        additional_instruction="Verify strictly against CDC 2024 materials. Accept values around 14.6–15% as 'approximately 15%'. If the answer does not give a concrete value or the value conflicts with the CDC page, mark incorrect."
    )

    # Risk factor identification (ensure the answer identifies hypertension)
    risk_node = evaluator.add_leaf(
        id="Most_Important_Modifiable_Risk_Factor",
        desc="Identifies hypertension/high blood pressure as the single most important modifiable risk factor for stroke.",
        parent=node,
        critical=True
    )
    rf = _safe(extracted.risk_factor.risk_factor if extracted.risk_factor else None)
    risk_claim = (
        f"The answer identifies '{rf}' as the single most important modifiable risk factor for stroke. "
        f"This should be 'hypertension' (high blood pressure). Mark correct only if the answer explicitly identifies hypertension."
    )
    await evaluator.verify(
        claim=risk_claim,
        node=risk_node,
        sources=None,
        additional_instruction="Use the provided answer text as context. This check is about whether the answer named hypertension as the top modifiable risk factor."
    )

    # PFO definition
    pfo_def_node = evaluator.add_leaf(
        id="PFO_Definition",
        desc="Defines patent foramen ovale (PFO) in medically correct terms.",
        parent=node,
        critical=True
    )
    pfo_def = _safe(extracted.pfo.definition if extracted.pfo else None)
    pfo_sources = extracted.pfo.source_urls if extracted.pfo and extracted.pfo.source_urls else []
    pfo_def_claim = (
        f"The provided definition of patent foramen ovale (PFO) — '{pfo_def}' — is medically correct and consistent with reputable sources "
        f"(i.e., a persistent flap-like opening between the right and left atria that fails to close after birth)."
    )
    await evaluator.verify(
        claim=pfo_def_claim,
        node=pfo_def_node,
        sources=pfo_sources,
        additional_instruction="Check that the phrasing aligns with standard medical definitions from reputable sources."
    )

    # PFO prevalence
    pfo_prev_node = evaluator.add_leaf(
        id="PFO_Prevalence",
        desc="States PFO prevalence as approximately 1 in 4 people (~25%).",
        parent=node,
        critical=True
    )
    pfo_prev = _safe(extracted.pfo.prevalence if extracted.pfo else None)
    pfo_prev_claim = (
        f"PFO prevalence in the general population is approximately {pfo_prev}, which corresponds to about 1 in 4 (~25%)."
    )
    await evaluator.verify(
        claim=pfo_prev_claim,
        node=pfo_prev_node,
        sources=pfo_sources,
        additional_instruction="Mark incorrect if the stated prevalence meaningfully deviates from ~25% or is unsupported by the sources."
    )

    # PFO relevance
    pfo_rel_node = evaluator.add_leaf(
        id="PFO_Relevance_To_Young_Adult_Stroke",
        desc="Explains clinical relevance of PFO for young adult stroke patients.",
        parent=node,
        critical=True
    )
    pfo_rel = _safe(extracted.pfo.relevance if extracted.pfo else None)
    pfo_rel_claim = (
        f"PFO is clinically relevant for young adult ischemic stroke patients. Specifically: {pfo_rel}"
    )
    await evaluator.verify(
        claim=pfo_rel_claim,
        node=pfo_rel_node,
        sources=pfo_sources,
        additional_instruction="The sources should support that PFO is a relevant consideration in young adult ischemic stroke, including paradoxical embolism risk context where appropriate."
    )


async def build_prevention_plan(
    evaluator: Evaluator,
    parent_node,
    extracted: MedicalSummaryExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Evidence_Based_Prevention_Plan",
        desc="Provides the strongest-evidence diet pattern and recommended daily exercise duration for stroke prevention.",
        parent=parent_node,
        critical=True
    )

    # Diet pattern
    diet_node = evaluator.add_leaf(
        id="Dietary_Pattern",
        desc="Identifies the Mediterranean diet as the dietary pattern with strongest evidence for stroke prevention.",
        parent=node,
        critical=True
    )
    diet = _safe(extracted.prevention.diet_pattern if extracted.prevention else None)
    diet_sources = extracted.prevention.diet_sources if extracted.prevention and extracted.prevention.diet_sources else []
    diet_claim = f"The dietary pattern with the strongest evidence for stroke prevention is the {diet}."
    await evaluator.verify(
        claim=diet_claim,
        node=diet_node,
        sources=diet_sources,
        additional_instruction="Expect 'Mediterranean diet' per recent guidelines/reviews (2023–2025). Mark incorrect if the claim or sources do not support this."
    )

    # Exercise duration
    ex_node = evaluator.add_leaf(
        id="Exercise_Duration",
        desc="States the recommended daily exercise duration as 30 minutes for stroke prevention.",
        parent=node,
        critical=True
    )
    exercise = _safe(extracted.prevention.exercise_daily_minutes if extracted.prevention else None)
    exercise_sources = extracted.prevention.exercise_sources if extracted.prevention and extracted.prevention.exercise_sources else []
    ex_claim = f"The recommended daily exercise duration for stroke prevention is {exercise} (approximately 30 minutes per day)."
    await evaluator.verify(
        claim=ex_claim,
        node=ex_node,
        sources=exercise_sources,
        additional_instruction="Verify the cited guidance recommends approximately 30 minutes daily (or ~150 minutes/week) of moderate activity; if not explicitly daily, ensure the answer's daily framing is reasonable."
    )


async def build_clinical_trial_eligibility(
    evaluator: Evaluator,
    parent_node,
    extracted: MedicalSummaryExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Clinical_Trial_Eligibility_SuNR1se_II",
        desc="Names the SuNR1se II Phase 2B chronic stroke neural stem cell trial, provides its ClinicalTrials.gov identifier, and verifies the patient meets the three main eligibility criteria (age, stroke type/location, and post-stroke time window).",
        parent=parent_node,
        critical=True
    )

    trial_sources = extracted.trial.trial_sources if extracted.trial and extracted.trial.trial_sources else []
    trial_name = _safe(extracted.trial.trial_name if extracted.trial else None)
    nct_id = _safe(extracted.trial.nct_id if extracted.trial else None)

    # Trial name and registry ID
    trial_id_node = evaluator.add_leaf(
        id="Trial_Name_And_Registry_ID",
        desc="Provides the trial name (SuNR1se II) and its ClinicalTrials.gov identifier.",
        parent=node,
        critical=True
    )
    trial_claim = f"The chronic stroke neural stem cell trial {trial_name} is a Phase 2B study with ClinicalTrials.gov identifier {nct_id}."
    await evaluator.verify(
        claim=trial_claim,
        node=trial_id_node,
        sources=trial_sources,
        additional_instruction="Verify on ClinicalTrials.gov (or official trial pages) that SuNR1se II is Phase 2B and the NCT ID matches."
    )

    # Eligibility checks depend on trial identification (as an extra prerequisite)
    age_node = evaluator.add_leaf(
        id="Eligibility_Age_Check",
        desc="Verifies patient age 39 meets the trial age range (18–75).",
        parent=node,
        critical=True
    )
    age_claim = "The trial's eligibility age range is 18–75 years, and a 39-year-old patient meets this criterion."
    await evaluator.verify(
        claim=age_claim,
        node=age_node,
        sources=trial_sources,
        additional_instruction="Confirm the trial lists the eligible ages as 18–75.",
        extra_prerequisites=[trial_id_node]
    )

    type_loc_node = evaluator.add_leaf(
        id="Eligibility_Stroke_Type_Location_Check",
        desc="Verifies the patient’s ischemic subcortical MCA stroke matches the trial-required stroke type/location (ischemic subcortical stroke in MCA and/or lenticulostriate artery).",
        parent=node,
        critical=True
    )
    type_loc_claim = (
        "The trial requires an ischemic subcortical stroke in the middle cerebral artery (MCA) territory and/or lenticulostriate artery; "
        "the patient’s ischemic subcortical MCA stroke matches this requirement."
    )
    await evaluator.verify(
        claim=type_loc_claim,
        node=type_loc_node,
        sources=trial_sources,
        additional_instruction="Confirm stroke type/location criteria as listed in the trial record.",
        extra_prerequisites=[trial_id_node]
    )

    time_node = evaluator.add_leaf(
        id="Eligibility_Time_Window_Check",
        desc="Verifies 9 months post-stroke is within the trial’s 6–60 month window.",
        parent=node,
        critical=True
    )
    time_claim = (
        "The trial requires that time since stroke is between 6 and 60 months; the patient's 9 months post-stroke is within this window."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_node,
        sources=trial_sources,
        additional_instruction="Confirm the time-since-stroke window is 6–60 months and that 9 months satisfies it.",
        extra_prerequisites=[trial_id_node]
    )


async def build_innovation_2025(
    evaluator: Evaluator,
    parent_node,
    extracted: MedicalSummaryExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Treatment_Innovation_2025",
        desc="Provides one major stroke treatment breakthrough announced/published in 2025 with institution, mechanism, and quantitative comparison metrics (as required by constraints).",
        parent=parent_node,
        critical=True
    )

    name = _safe(extracted.innovation2025.name if extracted.innovation2025 else None)
    institution = _safe(extracted.innovation2025.institution if extracted.innovation2025 else None)
    mechanism = _safe(extracted.innovation2025.mechanism if extracted.innovation2025 else None)
    metrics = _safe(extracted.innovation2025.quantitative_metrics if extracted.innovation2025 else None)
    sources = extracted.innovation2025.sources if extracted.innovation2025 and extracted.innovation2025.sources else []

    # Name and institution
    name_node = evaluator.add_leaf(
        id="Innovation_Name_And_Institution",
        desc="Names the 2025 breakthrough and identifies the developing institution/organization.",
        parent=node,
        critical=True
    )
    name_claim = f"The 2025 stroke treatment breakthrough '{name}' was developed by {institution}."
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=sources,
        additional_instruction="Confirm that the breakthrough and institution are correctly identified from the 2025 announcement/publication."
    )

    # Mechanism
    mech_node = evaluator.add_leaf(
        id="Innovation_Mechanism",
        desc="Explains how the innovation works (mechanism of action).",
        parent=node,
        critical=True
    )
    mech_claim = f"The mechanism of '{name}' is: {mechanism}"
    await evaluator.verify(
        claim=mech_claim,
        node=mech_node,
        sources=sources,
        additional_instruction="Verify that this mechanism description is accurate per the cited 2025 reports."
    )

    # Quantitative comparison
    metrics_node = evaluator.add_leaf(
        id="Innovation_Quantitative_Comparison",
        desc="Provides quantitative performance metrics comparing to existing standard treatments, including the ~90% vs ~11% first-try success comparison (per constraints).",
        parent=node,
        critical=True
    )
    metrics_claim = (
        f"The innovation reports quantitative performance compared with standard treatments as: {metrics}. "
        f"This should include approximately 90% vs approximately 11% first-try success."
    )
    await evaluator.verify(
        claim=metrics_claim,
        node=metrics_node,
        sources=sources,
        additional_instruction="Mark incorrect if the provided quantitative comparison does not include a ~90% vs ~11% first-try success figure or if the sources contradict it."
    )


async def build_citations(
    evaluator: Evaluator,
    parent_node,
    extracted: MedicalSummaryExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Citations",
        desc="Provides citations from reputable medical sources dated 2023–2025 for all requested information across the summary.",
        parent=parent_node,
        critical=True
    )

    # Collect URLs from extraction
    epi_urls = (extracted.citations.epidemiology_urls if extracted.citations else []) or []
    prev_urls = (extracted.citations.prevention_urls if extracted.citations else []) or []
    trial_urls = (extracted.citations.trial_urls if extracted.citations else []) or []
    innov_urls = (extracted.citations.innovation_urls if extracted.citations else []) or []
    all_urls = _uniq(
        (extracted.citations.all_urls if extracted.citations else []) or
        _flatten_urls(epi_urls, prev_urls, trial_urls, innov_urls)
    )

    # Reputable sources check - use simple verification with the answer context
    rep_node = evaluator.add_leaf(
        id="Reputable_Sources",
        desc="Citations are from reputable medical/health sources (e.g., CDC, ClinicalTrials.gov, peer-reviewed journals, academic medical centers, professional guidelines).",
        parent=node,
        critical=True
    )
    rep_claim = (
        "The citations provided in the answer are from reputable medical/health sources (e.g., CDC, ClinicalTrials.gov, "
        "peer-reviewed journals, academic medical centers, or professional guidelines). "
        f"Here are the cited URLs: {all_urls}"
    )
    await evaluator.verify(
        claim=rep_claim,
        node=rep_node,
        sources=None,
        additional_instruction="Judge reputability using the listed domains and the answer context. If many sources are not reputable, mark incorrect."
    )

    # Date range check - 2023–2025 inclusive (use simple verification over the answer's citation presentation)
    date_node = evaluator.add_leaf(
        id="Date_Range_2023_2025",
        desc="Cited sources are dated within 2023–2025 (inclusive).",
        parent=node,
        critical=True
    )
    date_claim = (
        "All citations used to support the summary are dated within 2023–2025 (inclusive). "
        f"Check the cited URLs and any provided publication/update years in the answer: {all_urls}"
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=None,
        additional_instruction="Use the answer's citation details and general knowledge about the sources. If any major citation clearly falls outside 2023–2025 or lacks a date while others do not meet the requirement, mark incorrect."
    )

    # Coverage for all sections - deterministic check from extracted structure
    coverage_ok = (len(epi_urls) > 0) and (len(prev_urls) > 0) and (len(trial_urls) > 0) and (len(innov_urls) > 0)
    evaluator.add_custom_node(
        result=coverage_ok,
        id="Coverage_For_All_Sections",
        desc="Each of the four requested sections includes at least one supporting citation.",
        parent=node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the requested medical information summary with hierarchical verification.
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

    # Extract structured information from the answer
    extracted: MedicalSummaryExtraction = await evaluator.extract(
        prompt=prompt_extract_medical_summary(),
        template_class=MedicalSummaryExtraction,
        extraction_name="medical_summary_extraction"
    )

    # Record scenario "ground truth" (patient facts) for context in outputs
    evaluator.add_ground_truth({
        "patient_age_years": PATIENT_FACTS["age_years"],
        "stroke_type_location": PATIENT_FACTS["stroke_type_location"],
        "months_since_stroke": PATIENT_FACTS["months_since_stroke"],
        "has_hypertension": PATIENT_FACTS["has_hypertension"],
    }, gt_type="scenario_facts")

    # Build top-level critical node that mirrors the rubric root
    mis_node = evaluator.add_parallel(
        id="Medical_Information_Summary",
        desc="Provides the requested medical information summary covering all four sections and includes citations as required.",
        parent=root,
        critical=True
    )

    # Build subtrees (can mostly run concurrently)
    await asyncio.gather(
        build_epidemiology_and_risk_context(evaluator, mis_node, extracted),
        build_prevention_plan(evaluator, mis_node, extracted),
        build_clinical_trial_eligibility(evaluator, mis_node, extracted),
        build_innovation_2025(evaluator, mis_node, extracted),
        build_citations(evaluator, mis_node, extracted),
    )

    # Return structured result
    return evaluator.get_summary()