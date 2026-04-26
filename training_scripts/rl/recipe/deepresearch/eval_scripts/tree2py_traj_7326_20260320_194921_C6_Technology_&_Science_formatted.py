import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# ------------------------------------------------------------------------------
# Task metadata
# ------------------------------------------------------------------------------
TASK_ID = "telecom_disaster_requirements_20260120"
TASK_DESCRIPTION = (
    "A facilities-based wireless telecommunications provider operates in California's High Fire Threat Districts. "
    "The provider's network includes: (A) macro cell tower sites within High Fire Threat Districts, "
    "(B) a central office that directly serves Public Safety Answering Points (PSAPs), and (C) cell sites located "
    "outside High Fire Threat Districts.\n\n"
    "On January 20, 2026 at 8:00 AM Pacific Time (PT), a wildfire-related disaster occurs in the High Fire Threat "
    "District coverage area. At 3:00 PM Eastern Time (ET) the same day, the FCC activates its Disaster Information "
    "Reporting System (DIRS) for the affected area, with the first report due January 21, 2026 at 10:00 AM ET. "
    "At 9:15 AM PT on January 20, the provider discovers a network outage affecting 911 service at multiple PSAPs "
    "in the disaster area.\n\n"
    "Determine:\n"
    "1. The minimum backup power duration (in hours) required for macro cell tower sites located within High Fire "
    "Threat Districts, considering both federal FCC requirements and California state-specific requirements, and "
    "identify which requirement is more stringent\n"
    "2. The minimum backup power duration (in hours) required for the central office that directly serves PSAPs "
    "under federal FCC regulations\n"
    "3. The minimum backup power duration (in hours) required for cell sites located outside High Fire Threat "
    "Districts under federal FCC regulations\n"
    "4. The exact deadline (date and time in ET) by which the provider must submit its first DIRS report to the FCC\n"
    "5. The exact deadline (date and time in PT) by which the provider must provide initial notification to affected "
    "PSAPs about the 911-impacting outage discovered at 9:15 AM PT\n"
    "6. Whether facilities-based wireless providers are required under FCC's Mandatory Disaster Response Initiative (MDRI) "
    "to establish mutual aid arrangements with other wireless providers (Yes or No)"
)


# ------------------------------------------------------------------------------
# Expected normative values (for context logging only, not used as ground truth)
# ------------------------------------------------------------------------------
NORMATIVE_EXPECTED = {
    "federal_cell_site_hours": "8",
    "california_hftd_macro_hours": "72",
    "central_office_psap_hours": "24",
    "non_hftd_cell_site_hours": "8",
    "dirs_deadline_et": "January 21, 2026 at 10:00 AM ET",
    "psap_initial_deadline_pt": "January 20, 2026 at 9:45 AM PT",
    "mdri_mutual_aid_required": "Yes"
}


# ------------------------------------------------------------------------------
# Extraction Models
# ------------------------------------------------------------------------------
class MacroHFTDGroup(BaseModel):
    federal_cell_value_hours: Optional[str] = None
    federal_cell_source_urls: List[str] = Field(default_factory=list)
    california_hftd_value_hours: Optional[str] = None
    california_hftd_source_urls: List[str] = Field(default_factory=list)
    more_stringent_determination: Optional[str] = None  # e.g., "California", "State", "California 72 > Federal 8"


class CentralOfficeGroup(BaseModel):
    psap_central_office_value_hours: Optional[str] = None
    psap_central_office_source_urls: List[str] = Field(default_factory=list)


class NonHFTDCellGroup(BaseModel):
    non_hftd_cell_value_hours: Optional[str] = None
    non_hftd_cell_source_urls: List[str] = Field(default_factory=list)


class DIRSGroup(BaseModel):
    dirs_first_report_deadline_et: Optional[str] = None  # Expect "January 21, 2026 at 10:00 AM ET"
    dirs_deadline_explanation: Optional[str] = None
    dirs_requirement_source_urls: List[str] = Field(default_factory=list)


class PSAPGroup(BaseModel):
    psap_initial_notification_deadline_pt: Optional[str] = None  # Expect "January 20, 2026 at 9:45 AM PT"
    psap_notification_explanation: Optional[str] = None
    psap_notification_source_urls: List[str] = Field(default_factory=list)


class MDRIGroup(BaseModel):
    mdri_mutual_aid_required_yes_no: Optional[str] = None  # Expect "Yes" or "No"
    mdri_requirement_source_urls: List[str] = Field(default_factory=list)


class TelecomRequirementsExtraction(BaseModel):
    macro_hftd: Optional[MacroHFTDGroup] = None
    central_office: Optional[CentralOfficeGroup] = None
    non_hftd_cell: Optional[NonHFTDCellGroup] = None
    dirs: Optional[DIRSGroup] = None
    psap: Optional[PSAPGroup] = None
    mdri: Optional[MDRIGroup] = None


# ------------------------------------------------------------------------------
# Extraction Prompt
# ------------------------------------------------------------------------------
def prompt_extract_requirements() -> str:
    return """
Extract the following information exactly as stated in the answer. Do not infer or add anything not explicitly present.

1) Macro cell tower sites within California High Fire Threat Districts (HFTDs):
   - federal_cell_value_hours: The number of hours stated for the minimum backup power requirement under federal FCC rules for cell sites (e.g., "8", "8 hours"). Extract the numeric or textual value as written.
   - federal_cell_source_urls: All URLs cited to support the federal cell site backup power requirement.
   - california_hftd_value_hours: The number of hours stated for the minimum backup power requirement under California rules for macro cell sites in HFTDs (e.g., "72", "72 hours").
   - california_hftd_source_urls: All URLs cited to support the California HFTD backup power requirement.
   - more_stringent_determination: The answer's explicit statement identifying which requirement is more stringent (e.g., "California", "California's 72 hours is more stringent than federal 8 hours"). If not stated, return null.

2) Central office directly serving PSAPs:
   - psap_central_office_value_hours: The number of hours stated for the minimum backup power requirement under federal FCC rules for a central office that directly serves PSAPs (e.g., "24", "24 hours").
   - psap_central_office_source_urls: All URLs cited to support the central office backup power requirement.

3) Cell sites located outside HFTDs:
   - non_hftd_cell_value_hours: The number of hours stated for the minimum backup power requirement under federal FCC rules for cell sites outside HFTDs (e.g., "8", "8 hours").
   - non_hftd_cell_source_urls: All URLs cited to support this requirement.

4) DIRS first report deadline (in Eastern Time):
   - dirs_first_report_deadline_et: The exact date and time in ET for the first DIRS report deadline as stated in the answer (e.g., "January 21, 2026 at 10:00 AM ET").
   - dirs_deadline_explanation: If the answer explains that DIRS was activated Jan 20, 2026 at 3:00 PM ET and the first report is due the next day at 10:00 AM ET, extract that explanation text; otherwise, return null.
   - dirs_requirement_source_urls: All URLs cited to support DIRS reporting requirements.

5) PSAP initial notification deadline (in Pacific Time):
   - psap_initial_notification_deadline_pt: The exact date and time in PT for the initial PSAP notification deadline as stated in the answer (e.g., "January 20, 2026 at 9:45 AM PT").
   - psap_notification_explanation: If the answer explains discovery at 9:15 AM PT and that notification must occur within 30 minutes, extract that explanation; otherwise, return null.
   - psap_notification_source_urls: All URLs cited to support PSAP notification requirements.

6) MDRI mutual aid:
   - mdri_mutual_aid_required_yes_no: Whether the answer states that facilities-based wireless providers are required under MDRI to establish mutual aid arrangements with other wireless providers ("Yes" or "No").
   - mdri_requirement_source_urls: All URLs cited to support the MDRI mutual aid requirement.

Return a JSON object conforming to the provided schema. If any field is missing in the answer, set it to null or an empty list for URLs.
    """


# ------------------------------------------------------------------------------
# Helper: Safe list getter
# ------------------------------------------------------------------------------
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and len(u.strip()) > 0]


# ------------------------------------------------------------------------------
# Verification Tree Builders
# ------------------------------------------------------------------------------
async def build_backup_power_requirements_tree(
    evaluator: Evaluator,
    parent_node,
    data: TelecomRequirementsExtraction
) -> None:
    backup_node = evaluator.add_parallel(
        id="backup_power_requirements",
        desc="Correct identification of minimum backup power duration requirements for all three facility types",
        parent=parent_node,
        critical=False
    )

    # Macro cell towers within HFTDs
    macro_node = evaluator.add_parallel(
        id="macro_cell_towers_hftd",
        desc="Correct determination of backup power requirement for macro cell tower sites in High Fire Threat Districts",
        parent=backup_node,
        critical=False
    )
    macro = data.macro_hftd or MacroHFTDGroup()

    # Federal requirement for cell sites (value + source)
    fed_group = evaluator.add_parallel(
        id="federal_requirement_cell_towers",
        desc="Correct identification of federal FCC minimum backup power requirement for cell sites (8 hours)",
        parent=macro_node,
        critical=False
    )

    # Value leaf (critical)
    fed_value_leaf = evaluator.add_leaf(
        id="federal_cell_value",
        desc="States that federal FCC requires minimum 8 hours backup power for cell sites",
        parent=fed_group,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the federal FCC minimum backup power requirement for cell sites (e.g., macro cell sites or wireless base stations) is 8 hours.",
        node=fed_value_leaf,
        additional_instruction="Focus on whether the answer explicitly states 8 hours for federal cell site backup power. Allow minor phrasing variations (e.g., 'at least 8 hours')."
    )

    # Source leaf (non-critical)
    fed_source_leaf = evaluator.add_leaf(
        id="federal_cell_source",
        desc="Provides verifiable source documentation for federal cell site backup power requirement",
        parent=fed_group,
        critical=False
    )
    await evaluator.verify(
        claim="The federal FCC requires a minimum of 8 hours of backup power for cell sites during disasters.",
        node=fed_source_leaf,
        sources=_safe_urls(macro.federal_cell_source_urls),
        additional_instruction="Verify that at least one cited source explicitly supports an 8-hour minimum backup power requirement for wireless cell sites/base stations. Accept credible FCC orders/rules or official FCC publications."
    )

    # California HFTD requirement (value + source)
    ca_group = evaluator.add_parallel(
        id="california_requirement_hftd",
        desc="Correct identification of California state-specific backup power requirement for macro cell towers in High Fire Threat Districts (72 hours)",
        parent=macro_node,
        critical=False
    )

    ca_value_leaf = evaluator.add_leaf(
        id="california_hftd_value",
        desc="States that California requires 72 hours backup power for macro cell towers in High Fire Threat Districts",
        parent=ca_group,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that California requires 72 hours of backup power for macro cell sites in Tier 2 and Tier 3 High Fire-Threat Districts (HFTDs).",
        node=ca_value_leaf,
        additional_instruction="Focus on whether the answer explicitly states 72 hours for California macro cell sites in HFTDs. Allow minor formatting differences."
    )

    ca_source_leaf = evaluator.add_leaf(
        id="california_hftd_source",
        desc="Provides verifiable source documentation for California HFTD backup power requirement",
        parent=ca_group,
        critical=False
    )
    await evaluator.verify(
        claim="California requires at least 72 hours of backup power for macro cell sites located in High Fire-Threat Districts (HFTDs).",
        node=ca_source_leaf,
        sources=_safe_urls(macro.california_hftd_source_urls),
        additional_instruction="Verify that at least one cited source (e.g., CPUC decisions/rules) explicitly supports the 72-hour backup power requirement for macro cell sites in HFTDs."
    )

    # More stringent determination (critical)
    more_stringent_leaf = evaluator.add_leaf(
        id="more_stringent_determination",
        desc="Correct identification that California's 72-hour requirement is more stringent than federal 8-hour requirement for macro cell towers in HFTDs",
        parent=macro_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer correctly identifies that California's 72-hour requirement is more stringent than the federal 8-hour requirement for macro cell towers located in HFTDs.",
        node=more_stringent_leaf,
        additional_instruction="Acknowledge that 72 hours > 8 hours; the answer should explicitly convey that California is more stringent."
    )

    # Central office serving PSAPs
    co_node = evaluator.add_parallel(
        id="central_office_serving_psaps",
        desc="Correct determination of backup power requirement for central office serving PSAPs",
        parent=backup_node,
        critical=False
    )
    co = data.central_office or CentralOfficeGroup()

    co_value_leaf = evaluator.add_leaf(
        id="psap_central_office_value",
        desc="States that federal FCC requires minimum 24 hours backup power for central offices serving PSAPs",
        parent=co_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that, under federal FCC regulations, a central office that directly serves PSAPs must have a minimum of 24 hours of backup power.",
        node=co_value_leaf,
        additional_instruction="Focus on whether the answer explicitly states 24 hours for central offices serving PSAPs under FCC rules."
    )

    co_source_leaf = evaluator.add_leaf(
        id="psap_central_office_source",
        desc="Provides verifiable source documentation for central office backup power requirement",
        parent=co_node,
        critical=False
    )
    await evaluator.verify(
        claim="Under federal FCC rules, a central office that directly serves PSAPs must have at least 24 hours of backup power.",
        node=co_source_leaf,
        sources=_safe_urls(co.psap_central_office_source_urls),
        additional_instruction="Verify that at least one cited FCC rule/order/guidance explicitly supports a 24-hour minimum for central offices serving PSAPs."
    )

    # Cell sites outside HFTDs
    non_hftd_node = evaluator.add_parallel(
        id="cell_sites_outside_hftd",
        desc="Correct determination of backup power requirement for cell sites outside High Fire Threat Districts",
        parent=backup_node,
        critical=False
    )
    non = data.non_hftd_cell or NonHFTDCellGroup()

    non_value_leaf = evaluator.add_leaf(
        id="non_hftd_cell_value",
        desc="States that federal FCC requires minimum 8 hours backup power for cell sites outside HFTDs",
        parent=non_hftd_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that, under federal FCC regulations, cell sites located outside High Fire-Threat Districts must have a minimum of 8 hours of backup power.",
        node=non_value_leaf,
        additional_instruction="Focus on whether the answer explicitly states 8 hours for federal cell site backup power (outside HFTDs)."
    )

    non_source_leaf = evaluator.add_leaf(
        id="non_hftd_cell_source",
        desc="Provides verifiable source documentation for standard cell site backup power requirement",
        parent=non_hftd_node,
        critical=False
    )
    await evaluator.verify(
        claim="The federal FCC requires a minimum of 8 hours of backup power for cell sites.",
        node=non_source_leaf,
        sources=_safe_urls(non.non_hftd_cell_source_urls),
        additional_instruction="Verify that at least one cited source explicitly supports an 8-hour minimum backup power requirement for cell sites."
    )


async def build_disaster_reporting_deadlines_tree(
    evaluator: Evaluator,
    parent_node,
    data: TelecomRequirementsExtraction
) -> None:
    deadlines_node = evaluator.add_parallel(
        id="disaster_reporting_deadlines",
        desc="Correct calculation of all disaster reporting and notification deadlines",
        parent=parent_node,
        critical=False
    )

    # DIRS first report deadline
    dirs_node = evaluator.add_parallel(
        id="dirs_first_report_deadline",
        desc="Correct determination of DIRS first report deadline",
        parent=deadlines_node,
        critical=False
    )
    dirs = data.dirs or DIRSGroup()

    dirs_value_leaf = evaluator.add_leaf(
        id="dirs_deadline_value",
        desc="States the correct DIRS first report deadline: January 21, 2026 at 10:00 AM ET",
        parent=dirs_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the DIRS first report deadline is January 21, 2026 at 10:00 AM ET.",
        node=dirs_value_leaf,
        additional_instruction="Check whether the answer explicitly provides this exact ET date/time for the first DIRS report."
    )

    dirs_calc_leaf = evaluator.add_leaf(
        id="dirs_deadline_calculation",
        desc="Explains that DIRS was activated January 20, 2026 at 3:00 PM ET with first report due the next day at 10:00 AM ET as specified in the scenario",
        parent=dirs_node,
        critical=False
    )
    await evaluator.verify(
        claim="The answer explains that DIRS was activated on January 20, 2026 at 3:00 PM ET and that the first report is due the next day at 10:00 AM ET, consistent with the scenario.",
        node=dirs_calc_leaf,
        additional_instruction="Check for a clear explanation using those specific activation and due times; allow minor phrasing variations."
    )

    dirs_source_leaf = evaluator.add_leaf(
        id="dirs_requirement_source",
        desc="Provides verifiable source documentation for DIRS reporting requirements",
        parent=dirs_node,
        critical=False
    )
    await evaluator.verify(
        claim="FCC DIRS procedures/public notices establish reporting obligations when DIRS is activated, including filing timelines as specified in activation notices (e.g., first report due at 10:00 AM ET the next day).",
        node=dirs_source_leaf,
        sources=_safe_urls(dirs.dirs_requirement_source_urls),
        additional_instruction="Verify that at least one cited FCC source discusses DIRS reporting obligations/timelines (ideally including first-report timing as specified in an activation notice)."
    )

    # PSAP initial notification deadline
    psap_node = evaluator.add_parallel(
        id="psap_initial_notification_deadline",
        desc="Correct calculation of 911 PSAP initial notification deadline",
        parent=deadlines_node,
        critical=False
    )
    psap = data.psap or PSAPGroup()

    psap_value_leaf = evaluator.add_leaf(
        id="psap_notification_value",
        desc="States the correct PSAP initial notification deadline: January 20, 2026 at 9:45 AM PT (30 minutes after 9:15 AM PT discovery time)",
        parent=psap_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the initial PSAP notification deadline is January 20, 2026 at 9:45 AM PT.",
        node=psap_value_leaf,
        additional_instruction="Check that the answer explicitly provides this PT date/time as the initial PSAP notification deadline."
    )

    psap_calc_leaf = evaluator.add_leaf(
        id="psap_notification_calculation",
        desc="Explains that outage was discovered at 9:15 AM PT and notification must occur within 30 minutes",
        parent=psap_node,
        critical=False
    )
    await evaluator.verify(
        claim="The answer explains that the outage was discovered at 9:15 AM PT and that affected PSAPs must be notified within 30 minutes.",
        node=psap_calc_leaf,
        additional_instruction="Check for an explicit explanation of the 30-minute rule applied to the 9:15 AM PT discovery time."
    )

    psap_source_leaf = evaluator.add_leaf(
        id="psap_notification_source",
        desc="Provides verifiable source documentation for 911 PSAP notification requirements",
        parent=psap_node,
        critical=False
    )
    await evaluator.verify(
        claim="FCC rules require that affected PSAPs receive initial notification of a 911-impacting outage as soon as possible, and no later than 30 minutes after discovery of the outage.",
        node=psap_source_leaf,
        sources=_safe_urls(psap.psap_notification_source_urls),
        additional_instruction="Verify that at least one cited FCC rule/order/public notice explicitly supports the 30-minute initial notification requirement to PSAPs."
    )


async def build_mutual_aid_requirement_tree(
    evaluator: Evaluator,
    parent_node,
    data: TelecomRequirementsExtraction
) -> None:
    mdri_node = evaluator.add_parallel(
        id="mutual_aid_requirement",
        desc="Correct determination of MDRI mutual aid requirement",
        parent=parent_node,
        critical=False
    )
    mdri = data.mdri or MDRIGroup()

    mdri_value_leaf = evaluator.add_leaf(
        id="mdri_requirement_value",
        desc="Correctly states that facilities-based wireless providers ARE required to establish mutual aid arrangements under MDRI (answer: Yes)",
        parent=mdri_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer correctly states 'Yes' — facilities-based wireless providers are required under the FCC's Mandatory Disaster Response Initiative (MDRI) to establish mutual aid arrangements with other wireless providers.",
        node=mdri_value_leaf,
        additional_instruction="Check that the answer explicitly answers 'Yes' and ties the requirement to MDRI mutual aid."
    )

    mdri_source_leaf = evaluator.add_leaf(
        id="mdri_requirement_source",
        desc="Provides verifiable source documentation for MDRI mutual aid requirements",
        parent=mdri_node,
        critical=False
    )
    await evaluator.verify(
        claim="Under the FCC's Mandatory Disaster Response Initiative (MDRI), facilities-based mobile wireless providers must establish mutual aid arrangements with other wireless providers.",
        node=mdri_source_leaf,
        sources=_safe_urls(mdri.mdri_requirement_source_urls),
        additional_instruction="Verify that at least one cited FCC source (order, rule, or official FCC publication) explicitly requires mutual aid arrangements under MDRI."
    )


# ------------------------------------------------------------------------------
# Main evaluate function
# ------------------------------------------------------------------------------
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

    # Record expected normative values for context
    evaluator.add_ground_truth(
        {
            "expected_values": NORMATIVE_EXPECTED
        },
        gt_type="expected_norms"
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=TelecomRequirementsExtraction,
        extraction_name="telecom_requirements_extraction"
    )

    # Build and verify trees according to rubric
    await build_backup_power_requirements_tree(evaluator, root, extracted)
    await build_disaster_reporting_deadlines_tree(evaluator, root, extracted)
    await build_mutual_aid_requirement_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()