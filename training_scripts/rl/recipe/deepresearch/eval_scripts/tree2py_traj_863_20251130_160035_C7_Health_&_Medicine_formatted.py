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
TASK_ID = "oral_glp1_2025_phase3"
TASK_DESCRIPTION = """
In 2025, a pharmaceutical company completed Phase 3 clinical trials for an oral small-molecule GLP-1 receptor agonist medication and plans to submit it for FDA regulatory review in the second half of 2025, with potential market availability in 2026. This medication represents a significant advancement as it can be taken once daily without specific food or water restrictions, unlike previous GLP-1 formulations. Identify this medication and provide the following comprehensive information: (1) The generic name of the medication, (2) The pharmaceutical company developing it, (3) The specific drug class and mechanism of action, (4) The name of at least one Phase 3 clinical trial, (5) Weight loss efficacy data or A1C reduction data from the Phase 3 trial(s), (6) The dosing regimen including frequency and available dose strengths tested, (7) Any food or water administration restrictions, (8) The patient population studied in the Phase 3 trials (obesity, type 2 diabetes, or both), (9) The duration of the Phase 3 trial(s) in weeks, (10) Safety profile including common adverse events and discontinuation rates, (11) The planned or actual FDA submission timeline, (12) The expected year of FDA approval, (13) The number of participants enrolled in the Phase 3 trial(s), (14) A reference URL from a peer-reviewed medical journal publication or official company press release that documents this information.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EfficacyEntry(BaseModel):
    metric_type: Optional[str] = None  # e.g., "weight_loss", "a1c"
    value: Optional[str] = None        # e.g., "14.5%" or "2.0%"
    trial_name: Optional[str] = None
    timepoint_weeks: Optional[str] = None  # e.g., "36"
    sources: List[str] = Field(default_factory=list)


class MedicationExtraction(BaseModel):
    # Identity and company
    generic_name: Optional[str] = None
    generic_name_sources: List[str] = Field(default_factory=list)

    manufacturer: Optional[str] = None
    manufacturer_sources: List[str] = Field(default_factory=list)

    # Class & mechanism
    drug_class_mechanism: Optional[str] = None
    drug_class_mechanism_sources: List[str] = Field(default_factory=list)

    # Phase 3 completion
    phase3_completed_2025_statement: Optional[str] = None
    phase3_completed_2025_sources: List[str] = Field(default_factory=list)

    # Trials info
    phase3_trial_names: List[str] = Field(default_factory=list)
    phase3_trial_names_sources: List[str] = Field(default_factory=list)

    # Efficacy entries (numeric)
    efficacy_items: List[EfficacyEntry] = Field(default_factory=list)

    # Dosing regimen
    dosing_frequency: Optional[str] = None  # e.g., "once daily"
    dosing_dose_strengths: List[str] = Field(default_factory=list)
    dosing_sources: List[str] = Field(default_factory=list)

    # Administration restrictions
    food_water_restrictions: Optional[str] = None  # e.g., "none", or specific text
    food_water_sources: List[str] = Field(default_factory=list)

    # Trial population & duration & sample size
    trial_population: Optional[str] = None  # e.g., "obesity", "type 2 diabetes", "both"
    trial_duration_weeks: List[str] = Field(default_factory=list)
    trial_sample_size: Optional[str] = None
    trial_info_sources: List[str] = Field(default_factory=list)

    # Safety profile
    safety_common_adverse_events: List[str] = Field(default_factory=list)
    safety_discontinuation_rates: List[str] = Field(default_factory=list)
    safety_sources: List[str] = Field(default_factory=list)

    # Regulatory timelines
    submission_timeline: Optional[str] = None  # e.g., "planned for H2 2025"
    submission_sources: List[str] = Field(default_factory=list)

    expected_approval_year: Optional[str] = None  # e.g., "2026"
    approval_year_sources: List[str] = Field(default_factory=list)

    market_availability: Optional[str] = None  # e.g., "expected in 2026"
    market_availability_sources: List[str] = Field(default_factory=list)

    # Reference URLs (peer-reviewed journal or official press release)
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_medication_info() -> str:
    return """
    Identify the oral small-molecule GLP-1 receptor agonist medication described in the answer and extract the following fields.
    For each field that requests sources, extract only the URLs explicitly provided in the answer text for that specific field.
    If multiple trials or figures are listed, extract all then we will pick the first if needed.
    If any field is missing, return null or an empty list accordingly.

    Fields to extract:
    - generic_name (string) and generic_name_sources (array of URLs)
    - manufacturer (string) and manufacturer_sources (array of URLs)
    - drug_class_mechanism (string) and drug_class_mechanism_sources (array of URLs)
    - phase3_completed_2025_statement (string) and phase3_completed_2025_sources (array of URLs)
    - phase3_trial_names (array of strings) and phase3_trial_names_sources (array of URLs)
    - efficacy_items (array of objects), each with:
        • metric_type (string; e.g., "weight_loss" or "a1c")
        • value (string; e.g., "14.5%" or "2.0%")
        • trial_name (string)
        • timepoint_weeks (string; e.g., "36")
        • sources (array of URLs)
    - dosing_frequency (string), dosing_dose_strengths (array of strings), dosing_sources (array of URLs)
    - food_water_restrictions (string; explicitly include "none" if stated), food_water_sources (array of URLs)
    - trial_population (string; e.g., "obesity", "type 2 diabetes", or "both")
      trial_duration_weeks (array of strings), trial_sample_size (string), trial_info_sources (array of URLs)
    - safety_common_adverse_events (array of strings), safety_discontinuation_rates (array of strings), safety_sources (array of URLs)
    - submission_timeline (string; e.g., "H2 2025"), submission_sources (array of URLs)
    - expected_approval_year (string), approval_year_sources (array of URLs)
    - market_availability (string; e.g., "2026"), market_availability_sources (array of URLs)
    - reference_urls (array of URLs; must be peer-reviewed journal publication or official company press release)

    Rules for URL extraction:
    - Extract only URLs explicitly present in the answer. Accept plain URLs or URLs inside markdown links.
    - Do not invent URLs. If a source is mentioned without a URL, ignore it.
    - Include full URLs with protocol (http/https). If missing protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(primary: Optional[List[str]], fallback: Optional[List[str]]) -> List[str]:
    ordered = []
    seen = set()
    for s in (primary or []):
        if isinstance(s, str) and s and s not in seen:
            ordered.append(s)
            seen.add(s)
    for s in (fallback or []):
        if isinstance(s, str) and s and s not in seen:
            ordered.append(s)
            seen.add(s)
    return ordered


def _first_nonempty(items: Optional[List[str]]) -> Optional[str]:
    if not items:
        return None
    for x in items:
        if isinstance(x, str) and x.strip():
            return x.strip()
    return None


def _safe_join(items: Optional[List[str]], sep: str = ", ") -> str:
    if not items:
        return ""
    return sep.join([i for i in items if isinstance(i, str) and i.strip()])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_medication_name_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="medication_name",
        desc="Provides the generic name of the medication.",
        parent=parent,
        critical=True
    )

    # Existence
    evaluator.add_custom_node(
        result=bool(info.generic_name and info.generic_name.strip()),
        id="medication_name_provided",
        desc="Generic name is provided.",
        parent=node,
        critical=True
    )

    # Supported by sources
    leaf = evaluator.add_leaf(
        id="medication_name_supported",
        desc="Generic name is supported by cited sources.",
        parent=node,
        critical=True
    )
    claim = f"The generic name of the medication is '{info.generic_name or ''}'."
    sources = _combine_sources(info.generic_name_sources, info.reference_urls)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify that the page(s) explicitly identify this medication by the stated generic name."
    )


async def build_manufacturer_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="manufacturer",
        desc="Identifies the pharmaceutical company developing the medication.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.manufacturer and info.manufacturer.strip()),
        id="manufacturer_provided",
        desc="Manufacturer is provided.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="manufacturer_supported",
        desc="Manufacturer is supported by cited sources.",
        parent=node,
        critical=True
    )
    claim = f"The pharmaceutical company developing {info.generic_name or 'the medication'} is '{info.manufacturer or ''}'."
    sources = _combine_sources(info.manufacturer_sources, info.reference_urls)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm the developer/manufacturer attribution from official or credible sources."
    )


async def build_drug_class_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="drug_class_mechanism",
        desc="Describes the drug class and mechanism of action, consistent with an oral small-molecule GLP-1 receptor agonist.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.drug_class_mechanism and info.drug_class_mechanism.strip()),
        id="drug_class_mechanism_provided",
        desc="Drug class/mechanism is provided.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="drug_class_mechanism_supported",
        desc="Drug class/mechanism is supported by cited sources.",
        parent=node,
        critical=True
    )
    claim = f"{info.generic_name or 'This medication'} is an oral small-molecule GLP-1 receptor agonist. Mechanism: {info.drug_class_mechanism or ''}."
    sources = _combine_sources(info.drug_class_mechanism_sources, info.reference_urls)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm that the medication is a small-molecule, oral GLP-1 receptor agonist and the mechanism description aligns."
    )


async def build_phase3_completion_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="phase3_completed_2025",
        desc="States (with supporting source) that Phase 3 clinical trials were completed in 2025.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.phase3_completed_2025_statement and info.phase3_completed_2025_statement.strip()),
        id="phase3_completed_2025_provided",
        desc="Statement that Phase 3 was completed in 2025 is provided.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="phase3_completed_2025_supported",
        desc="Phase 3 completion in 2025 is supported by sources.",
        parent=node,
        critical=True
    )
    claim = f"Phase 3 clinical trials for {info.generic_name or 'the medication'} were completed in 2025."
    sources = _combine_sources(info.phase3_completed_2025_sources, info.reference_urls)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify that the source explicitly states Phase 3 completion occurred in calendar year 2025."
    )


async def build_phase3_trial_name_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="phase3_trial_name",
        desc="Provides the name of at least one Phase 3 clinical trial.",
        parent=parent,
        critical=True
    )

    trial_name = _first_nonempty(info.phase3_trial_names)

    evaluator.add_custom_node(
        result=bool(trial_name),
        id="phase3_trial_name_provided",
        desc="At least one Phase 3 trial name is provided.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="phase3_trial_name_supported",
        desc="Phase 3 trial name is supported by sources.",
        parent=node,
        critical=True
    )
    claim = f"There is a Phase 3 clinical trial named '{trial_name or ''}' for {info.generic_name or 'the medication'}."
    sources = _combine_sources(info.phase3_trial_names_sources, info.reference_urls)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm that the named trial is a Phase 3 study associated with the medication."
    )


async def build_efficacy_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="efficacy_data",
        desc="Reports Phase 3 efficacy with numeric results (weight loss and/or A1C reduction) and indicates which trial(s) the figures come from.",
        parent=parent,
        critical=True
    )

    has_efficacy = any(
        (e.metric_type and e.value and e.trial_name)
        for e in (info.efficacy_items or [])
    )
    evaluator.add_custom_node(
        result=has_efficacy,
        id="efficacy_data_provided",
        desc="At least one Phase 3 numeric efficacy result is provided with trial attribution.",
        parent=node,
        critical=True
    )

    # Verify first efficacy item if present
    e_first = next((e for e in info.efficacy_items if e.metric_type and e.value and e.trial_name), None)
    leaf = evaluator.add_leaf(
        id="efficacy_data_supported",
        desc="Phase 3 numeric efficacy result is supported by sources.",
        parent=node,
        critical=True
    )

    if e_first:
        metric_txt = "weight loss" if (e_first.metric_type or "").lower().strip() in ["weight_loss", "weight loss", "weight"] else "A1C reduction"
        timepoint_part = f" at {e_first.timepoint_weeks} weeks" if e_first.timepoint_weeks else ""
        claim = f"In Phase 3 trial {e_first.trial_name}, {info.generic_name or 'the medication'} produced {e_first.value} {metric_txt}{timepoint_part}."
        sources = _combine_sources(e_first.sources, info.reference_urls)
    else:
        claim = "A Phase 3 numeric efficacy result is reported."
        sources = _combine_sources([], info.reference_urls)

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify numeric result and trial attribution. Allow reasonable rounding differences (e.g., 14.5% ≈ 14–15%)."
    )


async def build_dosing_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="dosing_regimen",
        desc="Specifies the dosing regimen including once-daily frequency and the dose strengths tested in the Phase 3 trial(s).",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.dosing_frequency and info.dosing_frequency.strip()),
        id="dosing_frequency_provided",
        desc="Dosing frequency is provided.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.dosing_dose_strengths),
        id="dosing_strengths_provided",
        desc="Dose strengths tested are provided.",
        parent=node,
        critical=True
    )

    # Frequency supported
    freq_leaf = evaluator.add_leaf(
        id="dosing_frequency_supported",
        desc="Once-daily dosing frequency is supported by sources.",
        parent=node,
        critical=True
    )
    freq_claim = f"{info.generic_name or 'the medication'} is dosed once daily."
    sources = _combine_sources(info.dosing_sources, info.reference_urls)
    await evaluator.verify(
        claim=freq_claim,
        node=freq_leaf,
        sources=sources,
        additional_instruction="Confirm dosing frequency is once daily (accept equivalent wording, e.g., 'once a day')."
    )

    # Dose strengths supported
    strengths_leaf = evaluator.add_leaf(
        id="dosing_strengths_supported",
        desc="Dose strengths tested are supported by sources.",
        parent=node,
        critical=True
    )
    strengths_txt = _safe_join(info.dosing_dose_strengths)
    strengths_claim = f"Phase 3 tested dose strengths for {info.generic_name or 'the medication'} include: {strengths_txt}."
    await evaluator.verify(
        claim=strengths_claim,
        node=strengths_leaf,
        sources=sources,
        additional_instruction="Verify the listed dose strengths from trial descriptions or official materials."
    )


async def build_restrictions_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="food_water_restrictions",
        desc="States any food or water administration restrictions (including explicitly stating if none).",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.food_water_restrictions is not None and str(info.food_water_restrictions).strip() != ""),
        id="restrictions_provided",
        desc="Administration restrictions are provided (or explicitly 'none').",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="restrictions_supported",
        desc="Administration restrictions (including 'none', if applicable) are supported by sources.",
        parent=node,
        critical=True
    )
    text = (info.food_water_restrictions or "").strip().lower()
    if text in ["none", "no restrictions", "no specific restrictions"]:
        claim = f"There are no specific food or water administration restrictions for {info.generic_name or 'the medication'}."
    else:
        claim = f"Administration restrictions for {info.generic_name or 'the medication'}: {info.food_water_restrictions or ''}."
    sources = _combine_sources(info.food_water_sources, info.reference_urls)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm whether there are specific food/water restrictions. If stated 'none', ensure sources explicitly indicate no special restrictions."
    )


async def build_population_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="trial_population",
        desc="Identifies the patient population studied in Phase 3 (obesity, type 2 diabetes, or both).",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.trial_population and info.trial_population.strip()),
        id="trial_population_provided",
        desc="Patient population is provided.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="trial_population_supported",
        desc="Patient population is supported by sources.",
        parent=node,
        critical=True
    )
    claim = f"Phase 3 studied population: {info.trial_population or ''}."
    sources = _combine_sources(info.trial_info_sources, info.reference_urls)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm whether Phase 3 targeted obesity, type 2 diabetes, or both (accept synonymous phrasing)."
    )


async def build_duration_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="trial_duration",
        desc="Provides the Phase 3 trial duration(s) in weeks.",
        parent=parent,
        critical=True
    )

    duration_weeks = _first_nonempty(info.trial_duration_weeks)
    evaluator.add_custom_node(
        result=bool(duration_weeks),
        id="trial_duration_provided",
        desc="At least one Phase 3 duration in weeks is provided.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="trial_duration_supported",
        desc="Phase 3 duration in weeks is supported by sources.",
        parent=node,
        critical=True
    )
    claim = f"The Phase 3 trial lasted {duration_weeks or ''} weeks."
    sources = _combine_sources(info.trial_info_sources, info.reference_urls)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify duration in weeks; allow small rounding if presented as a range or approximate."
    )


async def build_sample_size_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="trial_sample_size",
        desc="Provides the number of participants enrolled in the Phase 3 trial(s).",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.trial_sample_size and info.trial_sample_size.strip()),
        id="trial_sample_size_provided",
        desc="Phase 3 sample size is provided.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="trial_sample_size_supported",
        desc="Phase 3 sample size is supported by sources.",
        parent=node,
        critical=True
    )
    claim = f"The Phase 3 program enrolled {info.trial_sample_size or ''} participants."
    sources = _combine_sources(info.trial_info_sources, info.reference_urls)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm the number enrolled; accept reasonable rounding if the source presents a range or approximate figure."
    )


async def build_safety_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="safety_profile",
        desc="Summarizes safety profile including common adverse events and discontinuation rates from the Phase 3 trial(s).",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.safety_common_adverse_events),
        id="safety_ae_provided",
        desc="Common adverse events are provided.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.safety_discontinuation_rates),
        id="safety_disc_provided",
        desc="Discontinuation rates are provided.",
        parent=node,
        critical=True
    )

    # AEs supported
    ae_leaf = evaluator.add_leaf(
        id="safety_ae_supported",
        desc="Common adverse events are supported by sources.",
        parent=node,
        critical=True
    )
    ae_txt = _safe_join(info.safety_common_adverse_events)
    sources = _combine_sources(info.safety_sources, info.reference_urls)
    ae_claim = f"Common adverse events in Phase 3 included: {ae_txt}."
    await evaluator.verify(
        claim=ae_claim,
        node=ae_leaf,
        sources=sources,
        additional_instruction="Check trial safety results for common GI AEs (e.g., nausea, diarrhea, vomiting) or others listed."
    )

    # Discontinuation supported
    disc_leaf = evaluator.add_leaf(
        id="safety_disc_supported",
        desc="Discontinuation rates are supported by sources.",
        parent=node,
        critical=True
    )
    disc_txt = _safe_join(info.safety_discontinuation_rates)
    disc_claim = f"Treatment discontinuation rate(s) reported in Phase 3: {disc_txt}."
    await evaluator.verify(
        claim=disc_claim,
        node=disc_leaf,
        sources=sources,
        additional_instruction="Confirm discontinuation percentage(s) or qualitative rates as stated in Phase 3 reports."
    )


async def build_submission_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="fda_submission_timeline",
        desc="States the planned/actual FDA submission timeline and confirms it is planned for the second half of 2025 (with supporting source).",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.submission_timeline and info.submission_timeline.strip()),
        id="submission_timeline_provided",
        desc="FDA submission timeline is provided.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="submission_timeline_supported",
        desc="FDA submission timeline (H2 2025) is supported by sources.",
        parent=node,
        critical=True
    )
    claim = f"FDA submission for {info.generic_name or 'the medication'} is planned for the second half of 2025."
    sources = _combine_sources(info.submission_sources, info.reference_urls)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify that official or credible sources indicate a planned FDA submission in H2 2025."
    )


async def build_market_availability_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="market_availability_2026",
        desc="States the expected market availability timeframe and confirms it is expected in 2026 (with supporting source).",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.market_availability and info.market_availability.strip()),
        id="market_availability_provided",
        desc="Expected market availability timeframe is provided.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="market_availability_supported",
        desc="Expected market availability in 2026 is supported by sources.",
        parent=node,
        critical=True
    )
    claim = "Market availability is expected in 2026."
    sources = _combine_sources(info.market_availability_sources, info.reference_urls)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm that the source cites 2026 as the expected market availability timeframe."
    )


async def build_expected_approval_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="expected_approval_year",
        desc="Indicates the expected year of FDA approval (as stated by an official/credible source).",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.expected_approval_year and info.expected_approval_year.strip()),
        id="approval_year_provided",
        desc="Expected FDA approval year is provided.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="approval_year_supported",
        desc="Expected FDA approval year is supported by sources.",
        parent=node,
        critical=True
    )
    claim = f"The expected year of FDA approval is {info.expected_approval_year or ''}."
    sources = _combine_sources(info.approval_year_sources, info.reference_urls)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify that an official or credible source states the expected approval year."
    )


async def build_reference_nodes(evaluator: Evaluator, parent, info: MedicationExtraction):
    node = evaluator.add_parallel(
        id="reference_url",
        desc="Provides at least one verifiable reference URL from a peer-reviewed medical journal publication or an official company press release supporting the key claims.",
        parent=parent,
        critical=True
    )

    has_ref = bool(info.reference_urls)
    evaluator.add_custom_node(
        result=has_ref,
        id="reference_url_provided",
        desc="At least one reference URL is provided.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="reference_url_credible",
        desc="The provided reference URL is credible (peer-reviewed journal or official company press release).",
        parent=node,
        critical=True
    )
    first_ref = info.reference_urls[0] if info.reference_urls else None
    claim = "This URL is an official company press release or a peer-reviewed medical journal publication."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=first_ref,
        additional_instruction="Determine credibility by domain and page content (e.g., company newsroom press release page, or peer‑reviewed journal article)."
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
    Evaluate an answer for the oral GLP-1 2025 Phase 3 task.
    """
    # Initialize evaluator (root created internally as non-critical)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall items are independent checks
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

    # Add main critical node beneath root to reflect rubric root
    main = evaluator.add_parallel(
        id="oral_glp1_medication_information",
        desc="Identify the medication matching the prompt and provide the requested Phase 3, dosing, efficacy, safety, regulatory-timeline, and sourcing details.",
        parent=root,
        critical=True
    )

    # Extract structured information
    info: MedicationExtraction = await evaluator.extract(
        prompt=prompt_extract_medication_info(),
        template_class=MedicationExtraction,
        extraction_name="medication_extraction",
    )

    # Build verification subtrees
    await build_medication_name_nodes(evaluator, main, info)
    await build_manufacturer_nodes(evaluator, main, info)
    await build_drug_class_nodes(evaluator, main, info)
    await build_phase3_completion_nodes(evaluator, main, info)
    await build_phase3_trial_name_nodes(evaluator, main, info)
    await build_efficacy_nodes(evaluator, main, info)
    await build_dosing_nodes(evaluator, main, info)
    await build_restrictions_nodes(evaluator, main, info)
    await build_population_nodes(evaluator, main, info)
    await build_duration_nodes(evaluator, main, info)
    await build_sample_size_nodes(evaluator, main, info)
    await build_safety_nodes(evaluator, main, info)
    await build_submission_nodes(evaluator, main, info)
    await build_market_availability_nodes(evaluator, main, info)
    await build_expected_approval_nodes(evaluator, main, info)
    await build_reference_nodes(evaluator, main, info)

    # Return structured evaluation summary
    return evaluator.get_summary()