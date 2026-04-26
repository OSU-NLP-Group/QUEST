import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "cardio_stroke_guide_le8_befast"
TASK_DESCRIPTION = (
    "Given the CDC's recent finding that stroke prevalence among adults aged 18-44 increased by approximately 15% "
    "between 2011-2013 and 2020-2022, create a comprehensive cardiovascular health and stroke awareness reference guide "
    "for young adults (ages 18-44) in the United States. The guide must include three main sections: "
    "(1) Blood Pressure Screening Guidelines - provide the USPSTF-recommended screening frequencies for adults aged "
    "18-39 without increased risk and for adults aged 40 and older, along with the American Heart Association's optimal "
    "blood pressure target level; (2) Life's Essential 8 - list all eight components of the American Heart Association's "
    "Life's Essential 8 cardiovascular health measures with specific, measurable recommendations for each component "
    "(including dietary guidance, physical activity duration, sleep hours, BMI target, and monitoring metrics); and "
    "(3) BE FAST Stroke Warning Signs - describe all six components of the BE FAST acronym used to identify stroke "
    "symptoms, including what each letter stands for and the specific warning sign it represents. For each item, provide "
    "the specific guideline or recommendation along with a reference URL from an authoritative medical source."
)


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class AudienceCDCInfo(BaseModel):
    target_population_text: Optional[str] = None
    mentions_us: Optional[bool] = None
    mentions_age_18_44: Optional[bool] = None
    cdc_context_text: Optional[str] = None
    mentions_15_percent_increase: Optional[bool] = None
    mentions_year_range: Optional[bool] = None
    cdc_sources: List[str] = Field(default_factory=list)


class BPSection(BaseModel):
    section_present: Optional[bool] = None
    heading: Optional[str] = None

    freq_18_39_text: Optional[str] = None
    freq_18_39_sources: List[str] = Field(default_factory=list)

    freq_40_plus_text: Optional[str] = None
    freq_40_plus_sources: List[str] = Field(default_factory=list)

    aha_target_text: Optional[str] = None
    aha_sources: List[str] = Field(default_factory=list)


class EssentialItem(BaseModel):
    component_name: Optional[str] = None
    recommendation: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LE8Section(BaseModel):
    section_present: Optional[bool] = None
    items: List[EssentialItem] = Field(default_factory=list)


class BEFASTItem(BaseModel):
    letter: Optional[str] = None
    term: Optional[str] = None
    description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BEFASTSection(BaseModel):
    section_present: Optional[bool] = None
    items: List[BEFASTItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompts
# -----------------------------------------------------------------------------
def prompt_extract_audience_cdc() -> str:
    return """
    Extract information related to the intended audience and CDC context from the answer.

    Return a JSON object with:
    - target_population_text: The exact sentence/phrase describing the target population.
    - mentions_us: true/false if the answer explicitly mentions U.S./United States.
    - mentions_age_18_44: true/false if the answer explicitly mentions adults aged 18–44 (allow variants like 18-44).
    - cdc_context_text: The exact sentence/phrase describing the CDC context about stroke prevalence increase.
    - mentions_15_percent_increase: true/false if the answer clearly states approximately 15% increase.
    - mentions_year_range: true/false if the answer mentions the timeframes 2011–2013 and 2020–2022.
    - cdc_sources: Array of URLs cited for the CDC context (extract actual URLs mentioned).

    If a field is missing, return null (for text fields) or false (for booleans) and an empty array for missing URLs.
    """


def prompt_extract_bp_section() -> str:
    return """
    Extract the 'Blood Pressure Screening Guidelines' section content. Return:
    - section_present: true/false if there's a clearly labeled section about blood pressure screening guidelines (accept headings like 'Blood Pressure Screening Guidelines' or close variants).
    - heading: the heading text if present.
    - freq_18_39_text: the recommendation text for USPSTF screening frequency for ages 18–39 without increased risk and prior normal BP.
    - freq_18_39_sources: array of URLs cited for that 18–39 guideline.
    - freq_40_plus_text: the recommendation text for USPSTF screening frequency for adults aged 40 years and older.
    - freq_40_plus_sources: array of URLs cited for that 40+ guideline.
    - aha_target_text: the statement of the American Heart Association (AHA) optimal/normal BP target level (e.g., <120/80 mm Hg).
    - aha_sources: array of URLs cited for the AHA target.

    If any item is missing in the answer, set the text to null and the corresponding URLs array to empty.
    """


def prompt_extract_le8_section() -> str:
    return """
    Extract the 'Life's Essential 8' section. Return:
    - section_present: true/false if there's a clearly labeled Life's Essential 8 section (accept variants).
    - items: an array; for each of the eight Life's Essential 8 components, extract:
        - component_name: the component label as used in the answer (e.g., Diet, Physical Activity, Nicotine Exposure/Tobacco, Sleep, BMI/Body Mass Index, Cholesterol/Blood Lipids, Blood Sugar/Glucose, Blood Pressure).
        - recommendation: the specific, measurable recommendation text provided for that component.
        - sources: array of URLs cited specifically for that component's recommendation.

    Include all items the answer provides that correspond to Life's Essential 8. If fewer than 8 are present, return whatever exists. If more than 8, include them all but keep exact names used in the answer.
    """


def prompt_extract_befast_section() -> str:
    return """
    Extract the 'BE FAST' stroke warning signs section. Return:
    - section_present: true/false if there's a clearly labeled BE FAST section (accept variants).
    - items: an array with up to 6 entries for B, E, F, A, S, T. For each:
        - letter: one of B, E, F, A, S, T
        - term: what the letter stands for (e.g., Balance, Eyes, Face, Arms, Speech, Time)
        - description: the specific warning sign description stated in the answer for that letter.
        - sources: array of URLs cited for that letter/sign description.

    Return only the letters the answer provides; if some letters are missing, omit them.
    """


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def _uniq_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


# Synonyms and matching for Life's Essential 8 components
EXPECTED_LE8 = {
    "diet": {"names": ["diet", "healthy diet", "nutrition", "dietary pattern", "food", "eating pattern"]},
    "physical_activity": {"names": ["physical activity", "activity", "exercise", "aerobic"]},
    "nicotine": {"names": ["nicotine", "tobacco", "smoking", "vaping", "nicotine exposure"]},
    "sleep": {"names": ["sleep", "sleep health", "sleep duration"]},
    "bmi": {"names": ["bmi", "body mass index", "weight"]},
    "lipids": {"names": ["cholesterol", "blood lipids", "lipids", "non-hdl"]},
    "blood_sugar": {"names": ["blood sugar", "glucose", "hba1c", "diabetes"]},
    "blood_pressure": {"names": ["blood pressure", "bp", "hypertension"]},
}

DISPLAY_LE8 = {
    "diet": "Diet",
    "physical_activity": "Physical Activity",
    "nicotine": "Nicotine/Tobacco",
    "sleep": "Sleep",
    "bmi": "Body Mass Index (BMI)",
    "lipids": "Cholesterol/Blood Lipids (non-HDL)",
    "blood_sugar": "Blood Sugar (HbA1c)",
    "blood_pressure": "Blood Pressure",
}


def _match_le8_key(name: Optional[str]) -> Optional[str]:
    n = _normalize_text(name)
    for key, spec in EXPECTED_LE8.items():
        for syn in spec["names"]:
            if syn in n:
                return key
    return None


def _map_le8_items(items: List[EssentialItem]) -> Dict[str, EssentialItem]:
    result: Dict[str, EssentialItem] = {}
    for it in items or []:
        k = _match_le8_key(it.component_name)
        if k and k not in result:
            result[k] = it
    return result


# BE FAST expectations
BEFAST_EXPECTED = {
    "B": {"term": "Balance", "desc_keywords": ["balance", "coordination", "dizzy", "dizziness", "unsteady"]},
    "E": {"term": "Eyes", "desc_keywords": ["vision", "double vision", "loss of vision", "trouble seeing", "blurred"]},
    "F": {"term": "Face", "desc_keywords": ["face", "droop", "drooping", "uneven smile", "facial"]},
    "A": {"term": "Arms", "desc_keywords": ["arm", "arms", "weakness", "numbness", "one side", "drift"]},
    "S": {"term": "Speech", "desc_keywords": ["speech", "slurred", "difficulty speaking", "understanding", "confused"]},
    "T": {"term": "Time", "desc_keywords": ["time", "call 911", "emergency", "urgent", "immediately"]},
}


def _find_befast_item(items: List[BEFASTItem], letter: str) -> Optional[BEFASTItem]:
    for it in items or []:
        if (it.letter or "").strip().upper() == letter:
            return it
    return None


# -----------------------------------------------------------------------------
# Verification builders
# -----------------------------------------------------------------------------
async def verify_audience_cdc(evaluator: Evaluator, parent_node, audience: AudienceCDCInfo) -> None:
    node = evaluator.add_parallel(
        id="Audience_And_CDC_Context",
        desc="States/targets the intended population as U.S. adults ages 18–44 and includes the provided CDC context that stroke prevalence in ages 18–44 increased by ~15% from 2011–2013 to 2020–2022.",
        parent=parent_node,
        critical=True,
    )

    # 1) Targets U.S.
    leaf_us = evaluator.add_leaf(
        id="audience_us_mentioned",
        desc="The answer explicitly targets U.S. adults.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly targets U.S. adults (mentions U.S. or United States).",
        node=leaf_us,
        additional_instruction="Look for 'U.S.', 'US', or 'United States' in the audience statement or section header.",
    )

    # 2) Targets age 18–44
    leaf_age = evaluator.add_leaf(
        id="audience_18_44_mentioned",
        desc="The answer explicitly targets adults aged 18–44.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly targets adults aged 18–44 (accept 18-44 or 18 to 44).",
        node=leaf_age,
        additional_instruction="Check the audience statement for the age range.",
    )

    # 3) CDC context is stated in the answer
    leaf_cdc_claim = evaluator.add_leaf(
        id="cdc_context_stated",
        desc="The answer states that stroke prevalence among adults aged 18–44 increased by approximately 15% between 2011–2013 and 2020–2022.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that stroke prevalence among adults aged 18–44 increased by approximately 15% between 2011–2013 and 2020–2022.",
        node=leaf_cdc_claim,
        additional_instruction="Allow minor wording variations; 'about 15%' is acceptable.",
    )

    # 4) CDC sources exist
    cdc_sources = _uniq_urls((audience.cdc_sources if audience else []) or [])
    evaluator.add_custom_node(
        result=len(cdc_sources) > 0,
        id="cdc_sources_provided",
        desc="At least one authoritative reference URL is provided for the CDC context.",
        parent=node,
        critical=True,
    )

    # 5) CDC context supported by provided sources
    leaf_cdc_src = evaluator.add_leaf(
        id="cdc_context_supported_by_sources",
        desc="The provided reference URL(s) support the CDC context statement (~15% increase, 2011–2013 vs 2020–2022).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Stroke prevalence among adults aged 18–44 increased by approximately 15% between 2011–2013 and 2020–2022.",
        node=leaf_cdc_src,
        sources=cdc_sources,
        additional_instruction="Verify the page(s) mention this ~15% increase for the 18–44 age group across the specified timeframes.",
    )


async def verify_bp_section(evaluator: Evaluator, parent_node, bp: BPSection) -> None:
    node = evaluator.add_parallel(
        id="Section_1_Blood_Pressure_Screening_Guidelines",
        desc="Blood Pressure Screening Guidelines with USPSTF screening intervals and AHA optimal target, with references.",
        parent=parent_node,
        critical=True,
    )

    # 0) Section Label
    leaf_label = evaluator.add_leaf(
        id="bp_section_labeled",
        desc="Includes a clearly labeled Blood Pressure Screening Guidelines section.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes a clearly labeled 'Blood Pressure Screening Guidelines' section or an equivalent clear heading.",
        node=leaf_label,
        additional_instruction="Accept variations like 'BP Screening Guidelines'.",
    )

    # 1) USPSTF ages 18–39 every 3–5 years
    leaf_1839_claim = evaluator.add_leaf(
        id="uspstf_18_39_claimed",
        desc="States USPSTF screening frequency for ages 18–39 (every 3–5 years) when prior normal BP and no increased risk.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the USPSTF recommends blood pressure screening every 3–5 years for adults aged 18–39 with prior normal blood pressure and not at increased risk.",
        node=leaf_1839_claim,
        additional_instruction="Small wording variations allowed; the interval 'every 3 to 5 years' must be present.",
    )

    src_1839 = _uniq_urls((bp.freq_18_39_sources if bp else []) or [])
    evaluator.add_custom_node(
        result=len(src_1839) > 0,
        id="uspstf_18_39_sources_provided",
        desc="USPSTF 18–39 screening frequency has at least one reference URL.",
        parent=node,
        critical=True,
    )
    leaf_1839_src = evaluator.add_leaf(
        id="uspstf_18_39_supported",
        desc="USPSTF 18–39 every 3–5 years is supported by the provided source(s).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The USPSTF recommends blood pressure screening every 3–5 years for adults aged 18–39 who have prior normal blood pressure and are not at increased risk.",
        node=leaf_1839_src,
        sources=src_1839,
        additional_instruction="Verify the source states the 3–5 years interval for 18–39 with normal BP and no increased risk.",
    )

    # 2) USPSTF ages ≥40 annually
    leaf_40_claim = evaluator.add_leaf(
        id="uspstf_40_plus_claimed",
        desc="States USPSTF screening frequency for ages ≥40 (annually).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the USPSTF recommends annual blood pressure screening for adults aged 40 years or older.",
        node=leaf_40_claim,
        additional_instruction="Allow minor variations like 'once a year'.",
    )

    src_40 = _uniq_urls((bp.freq_40_plus_sources if bp else []) or [])
    evaluator.add_custom_node(
        result=len(src_40) > 0,
        id="uspstf_40_plus_sources_provided",
        desc="USPSTF ≥40 annual screening has at least one reference URL.",
        parent=node,
        critical=True,
    )
    leaf_40_src = evaluator.add_leaf(
        id="uspstf_40_plus_supported",
        desc="USPSTF ≥40 annual screening is supported by the provided source(s).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The USPSTF recommends annual blood pressure screening for adults aged 40 years or older.",
        node=leaf_40_src,
        sources=src_40,
        additional_instruction="Verify the source clearly indicates annual (yearly) screening for ≥40.",
    )

    # 3) AHA optimal BP target <120/80 mm Hg
    leaf_aha_claim = evaluator.add_leaf(
        id="aha_target_claimed",
        desc="States the AHA optimal/normal BP target is <120/80 mm Hg.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the American Heart Association's optimal or normal blood pressure target is less than 120/80 mm Hg.",
        node=leaf_aha_claim,
        additional_instruction="Accept phrasing like 'below 120/80' or 'under 120/80'.",
    )

    src_aha = _uniq_urls((bp.aha_sources if bp else []) or [])
    evaluator.add_custom_node(
        result=len(src_aha) > 0,
        id="aha_target_sources_provided",
        desc="AHA optimal BP target has at least one reference URL.",
        parent=node,
        critical=True,
    )
    leaf_aha_src = evaluator.add_leaf(
        id="aha_target_supported",
        desc="AHA <120/80 mm Hg target is supported by the provided source(s).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The American Heart Association defines normal blood pressure as less than 120/80 mm Hg.",
        node=leaf_aha_src,
        sources=src_aha,
        additional_instruction="Verify the source explicitly defines normal or optimal BP as <120/80 mm Hg.",
    )


async def _verify_le8_component(
    evaluator: Evaluator,
    parent_node,
    key: str,
    item: Optional[EssentialItem],
) -> None:
    """
    Add verification nodes for a single Life's Essential 8 component.
    All nodes added here are critical (due to parent critical constraint).
    """
    disp = DISPLAY_LE8[key]
    comp_node = evaluator.add_parallel(
        id=f"le8_{key}",
        desc=f"Life's Essential 8 component present with correct, measurable recommendation and reference: {disp}",
        parent=parent_node,
        critical=True,
    )

    # Existence
    evaluator.add_custom_node(
        result=item is not None and (item.component_name is not None),
        id=f"le8_{key}_exists",
        desc=f"{disp} component is provided.",
        parent=comp_node,
        critical=True,
    )

    rec_text = (item.recommendation or "") if item else ""
    srcs = _uniq_urls((item.sources if item else []) or [])

    # Source exists
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id=f"le8_{key}_sources_provided",
        desc=f"{disp} recommendation includes at least one reference URL.",
        parent=comp_node,
        critical=True,
    )

    # Specific measurable recommendation check in the answer (simple verify)
    measurable_leaf = evaluator.add_leaf(
        id=f"le8_{key}_measurable_in_answer",
        desc=f"The answer includes a specific, measurable recommendation for {disp}.",
        parent=comp_node,
        critical=True,
    )

    # Source-supported claim (verify_by_urls)
    supported_leaf = evaluator.add_leaf(
        id=f"le8_{key}_supported_by_sources",
        desc=f"The provided source(s) support the correct recommendation for {disp}.",
        parent=comp_node,
        critical=True,
    )

    # Build claims and additional instructions based on key
    if key == "diet":
        measurable_claim = (
            "In the Life's Essential 8 section, the diet recommendation includes specific and measurable guidance "
            "(for example, servings/day, sodium limits such as ≤2300 mg/day, or a standardized pattern like DASH or Mediterranean)."
        )
        support_claim = (
            "The source supports heart-healthy diet guidance consistent with the American Heart Association "
            "(e.g., a DASH/Mediterranean-style pattern and/or limiting sodium to 2300 mg/day or less)."
        )
    elif key == "physical_activity":
        measurable_claim = (
            "In the Life's Essential 8 section, the physical activity recommendation specifies duration (e.g., at least "
            "150 minutes/week of moderate-intensity aerobic activity, 75 minutes/week of vigorous activity, or an equivalent combination)."
        )
        support_claim = (
            "Adults should get at least 150 minutes per week of moderate-intensity aerobic activity, 75 minutes per week of vigorous "
            "activity, or an equivalent combination, per AHA guidance."
        )
    elif key == "sleep":
        measurable_claim = (
            "In the Life's Essential 8 section, the sleep recommendation specifies 7–9 hours of sleep per night for adults."
        )
        support_claim = "Adults should aim for 7 to 9 hours of sleep per night."
    elif key == "bmi":
        measurable_claim = (
            "In the Life's Essential 8 section, the BMI recommendation specifies a target of less than 25 (normal range)."
        )
        support_claim = "A BMI under 25 is considered the ideal/normal target for cardiovascular health in AHA Life's Essential 8."
    elif key == "lipids":
        measurable_claim = (
            "In the Life's Essential 8 section, the cholesterol/blood lipids recommendation specifies using non-HDL cholesterol as a monitoring metric."
        )
        support_claim = "Life's Essential 8 uses non-HDL cholesterol as the primary lipid monitoring metric."
    elif key == "blood_sugar":
        measurable_claim = (
            "In the Life's Essential 8 section, the blood sugar recommendation specifies using HbA1c as a monitoring metric."
        )
        support_claim = "HbA1c is a recommended metric for assessing blood sugar in AHA Life's Essential 8."
    elif key == "blood_pressure":
        measurable_claim = (
            "In the Life's Essential 8 section, the blood pressure recommendation provides a clear management target consistent with AHA guidance "
            "(e.g., keep BP <120/80 mm Hg if achievable)."
        )
        support_claim = "AHA guidance recommends maintaining a normal blood pressure of less than 120/80 mm Hg for optimal cardiovascular health."
    elif key == "nicotine":
        measurable_claim = (
            "In the Life's Essential 8 section, the nicotine/tobacco recommendation clearly states to avoid all nicotine/tobacco exposure (no smoking or vaping)."
        )
        support_claim = "Avoid all nicotine and tobacco exposure (no smoking or vaping) as part of AHA Life's Essential 8."
    else:
        measurable_claim = f"In the Life's Essential 8 section, the recommendation for {disp} is specific and measurable."
        support_claim = f"The source supports the recommendation for {disp}."

    await evaluator.verify(
        claim=measurable_claim,
        node=measurable_leaf,
        additional_instruction=f"Use the full answer to judge measurability. The extracted recommendation snippet is: '{rec_text}'."
    )

    await evaluator.verify(
        claim=support_claim,
        node=supported_leaf,
        sources=srcs,
        additional_instruction="Verify that the referenced page(s) explicitly endorse or state the recommendation claimed.",
    )


async def verify_le8_section(evaluator: Evaluator, parent_node, le8: LE8Section) -> None:
    node = evaluator.add_parallel(
        id="Section_2_Lifes_Essential_8",
        desc="Life’s Essential 8 section lists all 8 components with specific, measurable recommendations and reference URLs.",
        parent=parent_node,
        critical=True,
    )

    # 0) Section Label
    leaf_label = evaluator.add_leaf(
        id="le8_section_labeled",
        desc="Includes a clearly labeled Life’s Essential 8 section.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes a clearly labeled 'Life’s Essential 8' section (accept variants like 'Life's Essential 8').",
        node=leaf_label,
        additional_instruction="Look for a heading indicating 'Life’s Essential 8' or equivalent.",
    )

    items = (le8.items if le8 else []) or []
    mapping = _map_le8_items(items)

    # 1) All eight components are present
    all_present = all(k in mapping for k in EXPECTED_LE8.keys())
    evaluator.add_custom_node(
        result=all_present,
        id="le8_all_8_present",
        desc="All eight Life’s Essential 8 components are present.",
        parent=node,
        critical=True,
    )

    # 2) Per-component verification (existence, measurability in answer, source exists, supported by source)
    for key in EXPECTED_LE8.keys():
        await _verify_le8_component(evaluator, node, key, mapping.get(key))


async def _verify_befast_letter(
    evaluator: Evaluator,
    parent_node,
    letter: str,
    item: Optional[BEFASTItem],
) -> None:
    spec = BEFAST_EXPECTED[letter]
    term_expected = spec["term"]
    letter_node = evaluator.add_parallel(
        id=f"befast_{letter}",
        desc=f"BE FAST '{letter}' – {term_expected}: correct meaning and warning sign with reference.",
        parent=parent_node,
        critical=True,
    )

    # Existence
    evaluator.add_custom_node(
        result=item is not None and (item.term is not None),
        id=f"befast_{letter}_exists",
        desc=f"'{letter}' entry is present in BE FAST.",
        parent=letter_node,
        critical=True,
    )

    # Term/meaning is correct and warning sign described in the answer (simple verify)
    meaning_leaf = evaluator.add_leaf(
        id=f"befast_{letter}_meaning_in_answer",
        desc=f"In the answer, '{letter}' stands for {term_expected} and describes the correct warning sign.",
        parent=letter_node,
        critical=True,
    )
    desc_snippet = (item.description or "") if item else ""
    await evaluator.verify(
        claim=f"In the answer's BE FAST section, '{letter}' stands for {term_expected} and its description matches the expected warning sign.",
        node=meaning_leaf,
        additional_instruction=f"Look for keywords related to {term_expected}: {', '.join(spec['desc_keywords'])}. "
                               f"Use the snippet (if provided): '{desc_snippet}'. Allow equivalent phrasing."
    )

    # Source exists
    srcs = _uniq_urls((item.sources if item else []) or [])
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id=f"befast_{letter}_sources_provided",
        desc=f"'{letter}' entry includes at least one reference URL.",
        parent=letter_node,
        critical=True,
    )

    # Supported by sources
    supported_leaf = evaluator.add_leaf(
        id=f"befast_{letter}_supported_by_sources",
        desc=f"'{letter}' meaning and warning sign are supported by the provided source(s).",
        parent=letter_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"In BE FAST stroke recognition, '{letter}' stands for {term_expected}, with the described warning sign.",
        node=supported_leaf,
        sources=srcs,
        additional_instruction="Verify the reference explicitly explains the BE FAST letter and its meaning/warning sign.",
    )

    # Special for 'T' – Must say call 911 immediately
    if letter == "T":
        t911_leaf = evaluator.add_leaf(
            id="befast_T_call_911_in_answer",
            desc="The 'T' entry explicitly indicates to call 911 immediately.",
            parent=letter_node,
            critical=True,
        )
        await evaluator.verify(
            claim="In the answer's BE FAST section, 'T' indicates Time to call 911 immediately.",
            node=t911_leaf,
            additional_instruction="Confirm that calling 911 immediately is stated.",
        )


async def verify_befast_section(evaluator: Evaluator, parent_node, befast: BEFASTSection) -> None:
    node = evaluator.add_parallel(
        id="Section_3_BE_FAST_Stroke_Warning_Signs",
        desc="BE FAST section includes all 6 letters with correct meanings and references.",
        parent=parent_node,
        critical=True,
    )

    # 0) Section Label
    leaf_label = evaluator.add_leaf(
        id="befast_section_labeled",
        desc="Includes a clearly labeled BE FAST section.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes a clearly labeled 'BE FAST' stroke warning signs section (accept 'BEFAST' variants).",
        node=leaf_label,
        additional_instruction="Check for a clear heading for BE FAST.",
    )

    items = (befast.items if befast else []) or []

    # Ensure all letters exist
    for letter in ["B", "E", "F", "A", "S", "T"]:
        await _verify_befast_letter(evaluator, node, letter, _find_befast_item(items, letter))


# -----------------------------------------------------------------------------
# Main evaluation entry point
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
    # Initialize evaluator
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

    # Extract structured info concurrently
    audience_task = evaluator.extract(
        prompt=prompt_extract_audience_cdc(),
        template_class=AudienceCDCInfo,
        extraction_name="audience_cdc_info",
    )
    bp_task = evaluator.extract(
        prompt=prompt_extract_bp_section(),
        template_class=BPSection,
        extraction_name="bp_section",
    )
    le8_task = evaluator.extract(
        prompt=prompt_extract_le8_section(),
        template_class=LE8Section,
        extraction_name="le8_section",
    )
    befast_task = evaluator.extract(
        prompt=prompt_extract_befast_section(),
        template_class=BEFASTSection,
        extraction_name="befast_section",
    )

    audience_info, bp_info, le8_info, befast_info = await asyncio.gather(
        audience_task, bp_task, le8_task, befast_task
    )

    # Create top-level critical guide node (acts as the rubric's top-level)
    guide_node = evaluator.add_parallel(
        id="Cardiovascular_Health_And_Stroke_Awareness_Guide",
        desc="Reference guide for U.S. adults ages 18–44 including BP screening guidelines, Life’s Essential 8, and BE FAST with correct recommendations and authoritative reference URLs.",
        parent=root,
        critical=True,
    )

    # Build subtree verifications
    await verify_audience_cdc(evaluator, guide_node, audience_info)
    await verify_bp_section(evaluator, guide_node, bp_info)
    await verify_le8_section(evaluator, guide_node, le8_info)
    await verify_befast_section(evaluator, guide_node, befast_info)

    # Add ground truth hints for transparency (informational)
    evaluator.add_ground_truth(
        {
            "uspstf_18_39": "Every 3–5 years screening for adults 18–39 with prior normal BP and not at increased risk.",
            "uspstf_40_plus": "Annual screening for adults aged 40+.",
            "aha_normal_bp": "<120/80 mm Hg.",
            "le8_components": list(DISPLAY_LE8.values()),
            "befast_letters": ["B: Balance", "E: Eyes", "F: Face", "A: Arms", "S: Speech", "T: Time (call 911)"],
        },
        gt_type="expected_guidelines",
    )

    return evaluator.get_summary()