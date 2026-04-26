import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "ks_nh_graduation_requirements_comparison"
TASK_DESCRIPTION = """
A family is relocating from Kansas to New Hampshire and needs to understand how high school graduation requirements differ between the two states. Please provide a comprehensive comparison of the state-mandated minimum graduation requirements for both Kansas and New Hampshire high schools, covering the following categories: (1) Total minimum credits required for graduation, (2) English/Language Arts credit requirements, (3) Mathematics credit requirements (including any specific content requirements), (4) Science credit requirements (total, and specifically Physical Science and Biological Science), (5) Social Studies credit requirements, including U.S. and state history, Government/Civics, Economics, and Geography/World Studies, (6) Physical Education credit requirements, (7) Health Education credit requirements, (8) Arts Education credit requirements, (9) Digital Literacy requirements, and (10) Elective credit requirements. For each category, clearly specify the number of credits/units required in each state and note any important details about content requirements, waiver options, or structural differences between the two states' requirements.
"""


class CategoryEntry(BaseModel):
    kansas_value: Optional[str] = None
    kansas_sources: List[str] = Field(default_factory=list)
    kansas_notes: Optional[str] = None
    nh_value: Optional[str] = None
    nh_sources: List[str] = Field(default_factory=list)
    nh_notes: Optional[str] = None


class MathematicsCredits(BaseModel):
    kansas_value: Optional[str] = None
    kansas_sources: List[str] = Field(default_factory=list)
    kansas_content_requirements: Optional[str] = None
    nh_value: Optional[str] = None
    nh_sources: List[str] = Field(default_factory=list)
    nh_content_requirements: Optional[str] = None


class TotalScienceCredits(BaseModel):
    kansas_value: Optional[str] = None
    kansas_sources: List[str] = Field(default_factory=list)
    kansas_lab_requirement: Optional[str] = None
    nh_value: Optional[str] = None
    nh_sources: List[str] = Field(default_factory=list)
    nh_lab_requirement: Optional[str] = None


class HistoryCredits(BaseModel):
    kansas_us_history: Optional[str] = None
    kansas_state_history: Optional[str] = None
    kansas_sources: List[str] = Field(default_factory=list)
    nh_us_history: Optional[str] = None
    nh_state_history: Optional[str] = None
    nh_sources: List[str] = Field(default_factory=list)
    notes_kansas: Optional[str] = None
    notes_nh: Optional[str] = None


class DigitalLiteracyRequirement(BaseModel):
    kansas_requirement: Optional[str] = None
    kansas_sources: List[str] = Field(default_factory=list)
    nh_requirement: Optional[str] = None
    nh_sources: List[str] = Field(default_factory=list)
    notes_kansas: Optional[str] = None
    notes_nh: Optional[str] = None


class GraduationComparison(BaseModel):
    total_minimum_credits: Optional[CategoryEntry] = None
    english_credits: Optional[CategoryEntry] = None
    mathematics_credits: Optional[MathematicsCredits] = None
    physical_science_credits: Optional[CategoryEntry] = None
    biological_science_credits: Optional[CategoryEntry] = None
    total_science_credits: Optional[TotalScienceCredits] = None
    us_state_history_credits: Optional[HistoryCredits] = None
    government_civics_credits: Optional[CategoryEntry] = None
    economics_credits: Optional[CategoryEntry] = None
    geography_world_studies_credits: Optional[CategoryEntry] = None
    physical_education_credits: Optional[CategoryEntry] = None
    health_education_credits: Optional[CategoryEntry] = None
    arts_education_credits: Optional[CategoryEntry] = None
    digital_literacy_requirement: Optional[DigitalLiteracyRequirement] = None
    elective_credits: Optional[CategoryEntry] = None


def prompt_extract_graduation_comparison() -> str:
    return """
    Extract a structured, side-by-side comparison of Kansas and New Hampshire high school graduation requirements from the answer. For each category below, return both states' values exactly as stated in the answer (use strings, not numbers), any notes describing content/structure/waivers, and all explicit URL sources cited for that category. Do not invent information. If something is not mentioned, set it to null and sources to [].

    Categories to extract:

    1) total_minimum_credits:
       - kansas_value, kansas_notes, kansas_sources[]
       - nh_value, nh_notes, nh_sources[]

    2) english_credits:
       - kansas_value, kansas_notes, kansas_sources[]
       - nh_value, nh_notes, nh_sources[]

    3) mathematics_credits:
       - kansas_value, kansas_content_requirements, kansas_sources[]
       - nh_value, nh_content_requirements, nh_sources[]

    4) total_science_credits:
       - kansas_value, kansas_lab_requirement, kansas_sources[]
       - nh_value, nh_lab_requirement, nh_sources[]

    5) physical_science_credits:
       - kansas_value, kansas_notes, kansas_sources[]
       - nh_value, nh_notes, nh_sources[]

    6) biological_science_credits:
       - kansas_value, kansas_notes, kansas_sources[]
       - nh_value, nh_notes, nh_sources[]

    7) us_state_history_credits:
       - kansas_us_history, kansas_state_history, notes_kansas, kansas_sources[]
       - nh_us_history, nh_state_history, notes_nh, nh_sources[]

    8) government_civics_credits:
       - kansas_value, kansas_notes, kansas_sources[]
       - nh_value, nh_notes, nh_sources[]

    9) economics_credits:
       - kansas_value, kansas_notes, kansas_sources[]
       - nh_value, nh_notes, nh_sources[]

    10) geography_world_studies_credits:
       - kansas_value, kansas_notes, kansas_sources[]
       - nh_value, nh_notes, nh_sources[]

    11) physical_education_credits:
       - kansas_value, kansas_notes, kansas_sources[]
       - nh_value, nh_notes, nh_sources[]

    12) health_education_credits:
       - kansas_value, kansas_notes, kansas_sources[]
       - nh_value, nh_notes, nh_sources[]

    13) arts_education_credits:
       - kansas_value, kansas_notes, kansas_sources[]
       - nh_value, nh_notes, nh_sources[]

    14) digital_literacy_requirement:
       - kansas_requirement, notes_kansas, kansas_sources[]
       - nh_requirement, notes_nh, nh_sources[]

    15) elective_credits:
       - kansas_value, kansas_notes, kansas_sources[]
       - nh_value, nh_notes, nh_sources[]

    Sources must be explicit URLs appearing in the answer. Return all fields in a single JSON object using the schema provided.
    """


async def _verify_state_value(
    evaluator: Evaluator,
    parent_node,
    state_label: str,
    category_id: str,
    requirement_label: str,
    value: Optional[str],
    sources: List[str],
) -> None:
    exist_node = evaluator.add_custom_node(
        result=(bool(value and value.strip()) and len(sources) > 0),
        id=f"{category_id}_{state_label}_exists",
        desc=f"{state_label}: value and sources provided for {requirement_label}",
        parent=parent_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"{category_id}_{state_label}_value_supported",
        desc=f"{state_label} requirement supported by sources for {requirement_label}",
        parent=parent_node,
        critical=True
    )

    claim = f"{state_label} requires {value} credits/units for {requirement_label}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify the stated number of credits/units for this requirement on the provided official source(s). Treat 'credits' and 'units' as equivalent. Minor phrasing differences are acceptable as long as the numeric requirement and requirement scope match."
    )


async def _verify_state_note(
    evaluator: Evaluator,
    parent_node,
    state_label: str,
    category_id: str,
    note_desc: str,
    note_text: Optional[str],
    sources: List[str],
) -> None:
    note_present = evaluator.add_custom_node(
        result=bool(note_text and note_text.strip()),
        id=f"{category_id}_{state_label}_note_present",
        desc=f"{state_label}: note provided for {note_desc}",
        parent=parent_node,
        critical=False
    )

    note_leaf = evaluator.add_leaf(
        id=f"{category_id}_{state_label}_note_supported",
        desc=f"{state_label} note supported for {note_desc}",
        parent=parent_node,
        critical=False
    )

    claim = f"{state_label} detail for {note_desc}: {note_text}"
    await evaluator.verify(
        claim=claim,
        node=note_leaf,
        sources=sources,
        additional_instruction="Verify that this detail about content requirements, lab requirements, waivers, or structural notes is supported by the provided source(s). Paraphrase equivalence is acceptable.",
        extra_prerequisites=[note_present]
    )


async def build_simple_dual_state_category(
    evaluator: Evaluator,
    root_node,
    category_id: str,
    description: str,
    entry: Optional[CategoryEntry],
    requirement_label: str
) -> None:
    cat_node = evaluator.add_parallel(
        id=category_id,
        desc=description,
        parent=root_node,
        critical=False
    )

    ks_node = evaluator.add_parallel(
        id=f"{category_id}_Kansas",
        desc=f"Kansas - {requirement_label}",
        parent=cat_node,
        critical=False
    )

    nh_node = evaluator.add_parallel(
        id=f"{category_id}_New_Hampshire",
        desc=f"New Hampshire - {requirement_label}",
        parent=cat_node,
        critical=False
    )

    if entry is None:
        evaluator.add_custom_node(
            result=False,
            id=f"{category_id}_missing",
            desc=f"No information extracted for {requirement_label}",
            parent=cat_node,
            critical=False
        )
        return

    await _verify_state_value(
        evaluator,
        ks_node,
        "Kansas",
        category_id,
        requirement_label,
        entry.kansas_value,
        entry.kansas_sources
    )
    await _verify_state_note(
        evaluator,
        ks_node,
        "Kansas",
        category_id,
        f"{requirement_label} notes",
        entry.kansas_notes,
        entry.kansas_sources
    )

    await _verify_state_value(
        evaluator,
        nh_node,
        "New Hampshire",
        category_id,
        requirement_label,
        entry.nh_value,
        entry.nh_sources
    )
    await _verify_state_note(
        evaluator,
        nh_node,
        "New Hampshire",
        category_id,
        f"{requirement_label} notes",
        entry.nh_notes,
        entry.nh_sources
    )


async def build_mathematics_category(
    evaluator: Evaluator,
    root_node,
    entry: Optional[MathematicsCredits]
) -> None:
    category_id = "Mathematics_Credits"
    description = "Correctly identifies and compares the Mathematics credit requirements, including specific content requirements"
    cat_node = evaluator.add_parallel(
        id=category_id,
        desc=description,
        parent=root_node,
        critical=False
    )

    ks_node = evaluator.add_parallel(
        id=f"{category_id}_Kansas",
        desc="Kansas - Mathematics requirements",
        parent=cat_node,
        critical=False
    )
    nh_node = evaluator.add_parallel(
        id=f"{category_id}_New_Hampshire",
        desc="New Hampshire - Mathematics requirements",
        parent=cat_node,
        critical=False
    )

    if entry is None:
        evaluator.add_custom_node(
            result=False,
            id=f"{category_id}_missing",
            desc="No mathematics information extracted",
            parent=cat_node,
            critical=False
        )
        return

    await _verify_state_value(
        evaluator,
        ks_node,
        "Kansas",
        category_id,
        "mathematics",
        entry.kansas_value,
        entry.kansas_sources
    )
    await _verify_state_note(
        evaluator,
        ks_node,
        "Kansas",
        category_id,
        "mathematics content requirements",
        entry.kansas_content_requirements,
        entry.kansas_sources
    )

    await _verify_state_value(
        evaluator,
        nh_node,
        "New Hampshire",
        category_id,
        "mathematics",
        entry.nh_value,
        entry.nh_sources
    )
    await _verify_state_note(
        evaluator,
        nh_node,
        "New Hampshire",
        category_id,
        "mathematics content requirements",
        entry.nh_content_requirements,
        entry.nh_sources
    )


async def build_total_science_category(
    evaluator: Evaluator,
    root_node,
    entry: Optional[TotalScienceCredits]
) -> None:
    category_id = "Total_Science_Credits"
    description = "Correctly identifies and compares total science credits, including any laboratory requirements"
    cat_node = evaluator.add_parallel(
        id=category_id,
        desc=description,
        parent=root_node,
        critical=False
    )

    ks_node = evaluator.add_parallel(
        id=f"{category_id}_Kansas",
        desc="Kansas - Total science requirements",
        parent=cat_node,
        critical=False
    )
    nh_node = evaluator.add_parallel(
        id=f"{category_id}_New_Hampshire",
        desc="New Hampshire - Total science requirements",
        parent=cat_node,
        critical=False
    )

    if entry is None:
        evaluator.add_custom_node(
            result=False,
            id=f"{category_id}_missing",
            desc="No total science information extracted",
            parent=cat_node,
            critical=False
        )
        return

    await _verify_state_value(
        evaluator,
        ks_node,
        "Kansas",
        category_id,
        "total science",
        entry.kansas_value,
        entry.kansas_sources
    )
    await _verify_state_note(
        evaluator,
        ks_node,
        "Kansas",
        category_id,
        "science lab requirement",
        entry.kansas_lab_requirement,
        entry.kansas_sources
    )

    await _verify_state_value(
        evaluator,
        nh_node,
        "New Hampshire",
        category_id,
        "total science",
        entry.nh_value,
        entry.nh_sources
    )
    await _verify_state_note(
        evaluator,
        nh_node,
        "New Hampshire",
        category_id,
        "science lab requirement",
        entry.nh_lab_requirement,
        entry.nh_sources
    )


async def build_history_category(
    evaluator: Evaluator,
    root_node,
    entry: Optional[HistoryCredits]
) -> None:
    category_id = "US_State_History_Credits"
    description = "Correctly identifies and compares U.S. and state history credits"
    cat_node = evaluator.add_parallel(
        id=category_id,
        desc=description,
        parent=root_node,
        critical=False
    )

    ks_node = evaluator.add_parallel(
        id=f"{category_id}_Kansas",
        desc="Kansas - U.S. and State History",
        parent=cat_node,
        critical=False
    )
    nh_node = evaluator.add_parallel(
        id=f"{category_id}_New_Hampshire",
        desc="New Hampshire - U.S. and State History",
        parent=cat_node,
        critical=False
    )

    if entry is None:
        evaluator.add_custom_node(
            result=False,
            id=f"{category_id}_missing",
            desc="No U.S./State history information extracted",
            parent=cat_node,
            critical=False
        )
        return

    # Kansas US History
    us_provided_ks = evaluator.add_custom_node(
        result=bool(entry.kansas_us_history and entry.kansas_us_history.strip()),
        id=f"{category_id}_Kansas_us_provided",
        desc="Kansas: U.S. History value provided",
        parent=ks_node,
        critical=False
    )
    us_leaf_ks = evaluator.add_leaf(
        id=f"{category_id}_Kansas_us_supported",
        desc="Kansas: U.S. History credits supported",
        parent=ks_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Kansas requires {entry.kansas_us_history} credits in U.S. History.",
        node=us_leaf_ks,
        sources=entry.kansas_sources,
        additional_instruction="Verify the U.S. History credit requirement for Kansas. Accept phrasing variants like 'United States History'.",
        extra_prerequisites=[us_provided_ks]
    )

    # Kansas State History
    state_provided_ks = evaluator.add_custom_node(
        result=bool(entry.kansas_state_history and entry.kansas_state_history.strip()),
        id=f"{category_id}_Kansas_state_provided",
        desc="Kansas: State History value provided",
        parent=ks_node,
        critical=False
    )
    state_leaf_ks = evaluator.add_leaf(
        id=f"{category_id}_Kansas_state_supported",
        desc="Kansas: State History credits supported",
        parent=ks_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Kansas requires {entry.kansas_state_history} credits in Kansas State History (Kansas History).",
        node=state_leaf_ks,
        sources=entry.kansas_sources,
        additional_instruction="Verify the Kansas state history requirement. Accept synonyms like 'Kansas History'.",
        extra_prerequisites=[state_provided_ks]
    )

    await _verify_state_note(
        evaluator,
        ks_node,
        "Kansas",
        category_id,
        "history notes",
        entry.notes_kansas,
        entry.kansas_sources
    )

    # New Hampshire US History
    us_provided_nh = evaluator.add_custom_node(
        result=bool(entry.nh_us_history and entry.nh_us_history.strip()),
        id=f"{category_id}_New_Hampshire_us_provided",
        desc="New Hampshire: U.S. History value provided",
        parent=nh_node,
        critical=False
    )
    us_leaf_nh = evaluator.add_leaf(
        id=f"{category_id}_New_Hampshire_us_supported",
        desc="New Hampshire: U.S. History credits supported",
        parent=nh_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"New Hampshire requires {entry.nh_us_history} credits in U.S. History.",
        node=us_leaf_nh,
        sources=entry.nh_sources,
        additional_instruction="Verify the U.S. History credit requirement for New Hampshire.",
        extra_prerequisites=[us_provided_nh]
    )

    # New Hampshire State History
    state_provided_nh = evaluator.add_custom_node(
        result=bool(entry.nh_state_history and entry.nh_state_history.strip()),
        id=f"{category_id}_New_Hampshire_state_provided",
        desc="New Hampshire: State History value provided",
        parent=nh_node,
        critical=False
    )
    state_leaf_nh = evaluator.add_leaf(
        id=f"{category_id}_New_Hampshire_state_supported",
        desc="New Hampshire: State History credits supported",
        parent=nh_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"New Hampshire requires {entry.nh_state_history} credits in New Hampshire State History.",
        node=state_leaf_nh,
        sources=entry.nh_sources,
        additional_instruction="Verify any New Hampshire state history requirement if present. Accept variants like 'NH History' or 'New Hampshire History'.",
        extra_prerequisites=[state_provided_nh]
    )

    await _verify_state_note(
        evaluator,
        nh_node,
        "New Hampshire",
        category_id,
        "history notes",
        entry.notes_nh,
        entry.nh_sources
    )


async def build_digital_literacy_category(
    evaluator: Evaluator,
    root_node,
    entry: Optional[DigitalLiteracyRequirement]
) -> None:
    category_id = "Digital_Literacy_Requirement"
    description = "Correctly identifies and compares the digital literacy requirements"
    cat_node = evaluator.add_parallel(
        id=category_id,
        desc=description,
        parent=root_node,
        critical=False
    )

    ks_node = evaluator.add_parallel(
        id=f"{category_id}_Kansas",
        desc="Kansas - Digital literacy",
        parent=cat_node,
        critical=False
    )
    nh_node = evaluator.add_parallel(
        id=f"{category_id}_New_Hampshire",
        desc="New Hampshire - Digital literacy",
        parent=cat_node,
        critical=False
    )

    if entry is None:
        evaluator.add_custom_node(
            result=False,
            id=f"{category_id}_missing",
            desc="No digital literacy information extracted",
            parent=cat_node,
            critical=False
        )
        return

    ks_exist = evaluator.add_custom_node(
        result=(bool(entry.kansas_requirement and entry.kansas_requirement.strip()) and len(entry.kansas_sources) > 0),
        id=f"{category_id}_Kansas_exists",
        desc="Kansas: digital literacy requirement and sources provided",
        parent=ks_node,
        critical=True
    )
    ks_leaf = evaluator.add_leaf(
        id=f"{category_id}_Kansas_supported",
        desc="Kansas: digital literacy requirement supported",
        parent=ks_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Kansas digital literacy requirement: {entry.kansas_requirement}",
        node=ks_leaf,
        sources=entry.kansas_sources,
        additional_instruction="Confirm the presence/description of any state digital literacy or technology competency requirement for graduation. If none exists, the answer should explicitly state that, supported by sources."
    )
    await _verify_state_note(
        evaluator,
        ks_node,
        "Kansas",
        category_id,
        "digital literacy notes",
        entry.notes_kansas,
        entry.kansas_sources
    )

    nh_exist = evaluator.add_custom_node(
        result=(bool(entry.nh_requirement and entry.nh_requirement.strip()) and len(entry.nh_sources) > 0),
        id=f"{category_id}_New_Hampshire_exists",
        desc="New Hampshire: digital literacy requirement and sources provided",
        parent=nh_node,
        critical=True
    )
    nh_leaf = evaluator.add_leaf(
        id=f"{category_id}_New_Hampshire_supported",
        desc="New Hampshire: digital literacy requirement supported",
        parent=nh_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"New Hampshire digital literacy requirement: {entry.nh_requirement}",
        node=nh_leaf,
        sources=entry.nh_sources,
        additional_instruction="Confirm the presence/description of any state digital literacy or technology competency requirement for graduation. If none exists, the answer should explicitly state that, supported by sources."
    )
    await _verify_state_note(
        evaluator,
        nh_node,
        "New Hampshire",
        category_id,
        "digital literacy notes",
        entry.notes_nh,
        entry.nh_sources
    )


async def build_physical_bio_science_categories(
    evaluator: Evaluator,
    root_node,
    physical_entry: Optional[CategoryEntry],
    biological_entry: Optional[CategoryEntry]
) -> None:
    await build_simple_dual_state_category(
        evaluator,
        root_node,
        "Physical_Science_Credits",
        "Correctly identifies and compares Physical Science credits",
        physical_entry,
        "physical science"
    )

    await build_simple_dual_state_category(
        evaluator,
        root_node,
        "Biological_Science_Credits",
        "Correctly identifies and compares Biological Science credits",
        biological_entry,
        "biological science"
    )


async def build_social_studies_related_categories(
    evaluator: Evaluator,
    root_node,
    gov_entry: Optional[CategoryEntry],
    econ_entry: Optional[CategoryEntry],
    geo_entry: Optional[CategoryEntry]
) -> None:
    await build_simple_dual_state_category(
        evaluator,
        root_node,
        "Government_Civics_Credits",
        "Correctly identifies and compares Government/Civics credits",
        gov_entry,
        "government/civics"
    )
    await build_simple_dual_state_category(
        evaluator,
        root_node,
        "Economics_Credits",
        "Correctly identifies and compares Economics credits",
        econ_entry,
        "economics"
    )
    await build_simple_dual_state_category(
        evaluator,
        root_node,
        "Geography_World_Studies_Credits",
        "Correctly identifies and compares Geography/World Studies credits",
        geo_entry,
        "geography/world studies"
    )


async def build_other_categories(
    evaluator: Evaluator,
    root_node,
    english_entry: Optional[CategoryEntry],
    pe_entry: Optional[CategoryEntry],
    health_entry: Optional[CategoryEntry],
    arts_entry: Optional[CategoryEntry],
    elective_entry: Optional[CategoryEntry],
    total_entry: Optional[CategoryEntry]
) -> None:
    await build_simple_dual_state_category(
        evaluator,
        root_node,
        "English_Credits",
        "Correctly identifies and compares English/Language Arts credits",
        english_entry,
        "English/Language Arts"
    )
    await build_simple_dual_state_category(
        evaluator,
        root_node,
        "Physical_Education_Credits",
        "Correctly identifies and compares Physical Education credits",
        pe_entry,
        "physical education"
    )
    await build_simple_dual_state_category(
        evaluator,
        root_node,
        "Health_Education_Credits",
        "Correctly identifies and compares Health Education credits",
        health_entry,
        "health education"
    )
    await build_simple_dual_state_category(
        evaluator,
        root_node,
        "Arts_Education_Credits",
        "Correctly identifies and compares Arts Education credits",
        arts_entry,
        "arts education"
    )
    await build_simple_dual_state_category(
        evaluator,
        root_node,
        "Elective_Credits",
        "Correctly identifies and compares Elective credits",
        elective_entry,
        "electives"
    )
    await build_simple_dual_state_category(
        evaluator,
        root_node,
        "Total_Minimum_Credits",
        "Correctly identifies and compares total minimum credits required for graduation",
        total_entry,
        "total minimum credits for graduation"
    )


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_graduation_comparison(),
        template_class=GraduationComparison,
        extraction_name="graduation_requirements_comparison"
    )

    top_node = evaluator.add_parallel(
        id="Graduation_Requirements_Comparison",
        desc="Comprehensive comparison of high school graduation requirements between Kansas and New Hampshire",
        parent=root,
        critical=False
    )

    await build_other_categories(
        evaluator,
        top_node,
        english_entry=extracted.english_credits,
        pe_entry=extracted.physical_education_credits,
        health_entry=extracted.health_education_credits,
        arts_entry=extracted.arts_education_credits,
        elective_entry=extracted.elective_credits,
        total_entry=extracted.total_minimum_credits
    )

    await build_mathematics_category(evaluator, top_node, extracted.mathematics_credits)

    await build_total_science_category(evaluator, top_node, extracted.total_science_credits)

    await build_physical_bio_science_categories(
        evaluator,
        top_node,
        physical_entry=extracted.physical_science_credits,
        biological_entry=extracted.biological_science_credits
    )

    await build_history_category(evaluator, top_node, extracted.us_state_history_credits)

    await build_social_studies_related_categories(
        evaluator,
        top_node,
        gov_entry=extracted.government_civics_credits,
        econ_entry=extracted.economics_credits,
        geo_entry=extracted.geography_world_studies_credits
    )

    await build_digital_literacy_category(evaluator, top_node, extracted.digital_literacy_requirement)

    return evaluator.get_summary()