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
TASK_ID = "eligibility_guide_division_I_transfers"
TASK_DESCRIPTION = """A college sports academic advisor is preparing a comprehensive eligibility guide for Division I student-athletes considering transfer options. The guide needs to compare academic requirements at three universities across different athletic conferences: California State University, Northridge (CSUN), University of North Carolina at Chapel Hill (UNC), and Penn State University.

For each of the three universities, provide the following information:

1. For student-athletes entering their third year (beginning of year 3):
   - Minimum NCAA GPA requirement
   - Minimum NCAA degree-applicable credit hours required
   - Percentage of degree requirements that must be completed by the end of the second year (NCAA 40/60/80 rule)
   - For UNC only: institution-specific requirements including minimum cumulative UNC GPA, minimum semester credit hours by end of semester 4, and the attempted-to-completed credit hour ratio requirement

2. For student-athletes entering their fifth year (beginning of year 5):
   - Minimum NCAA GPA requirement
   - Minimum NCAA degree-applicable credit hours required
   - Major declaration requirement status
   - Percentage of degree requirements that must be completed by the end of the third year (NCAA 40/60/80 rule)
   - For UNC only: institution-specific requirements including minimum cumulative UNC GPA by end of third year, minimum semester credit hours by end of semester 6, and the attempted-to-completed credit hour ratio requirement

3. Conference affiliation: The name of the athletic conference each university competes in

4. General NCAA Division I requirements applicable to all three institutions:
   - Minimum degree-applicable credit hours required per semester
   - Minimum degree-applicable credit hours required per academic year (fall and spring semesters)
   - Minimum four-year Academic Progress Rate (APR) required for postseason competition eligibility

For each piece of information provided, include the reference URL from which the information was obtained. Present your answer in a structured format clearly organized by university and requirement category.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ValueWithSources(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Year3NCAA(BaseModel):
    ncaa_gpa: ValueWithSources = ValueWithSources()
    ncaa_credits: ValueWithSources = ValueWithSources()
    ncaa_percent: ValueWithSources = ValueWithSources()


class Year5NCAA(BaseModel):
    ncaa_gpa: ValueWithSources = ValueWithSources()
    ncaa_credits: ValueWithSources = ValueWithSources()
    major_declaration: ValueWithSources = ValueWithSources()
    ncaa_percent: ValueWithSources = ValueWithSources()


class UNCYear3Inst(BaseModel):
    inst_gpa: ValueWithSources = ValueWithSources()
    inst_sem4_credits: ValueWithSources = ValueWithSources()
    inst_ratio: ValueWithSources = ValueWithSources()


class UNCYear5Inst(BaseModel):
    inst_gpa: ValueWithSources = ValueWithSources()
    inst_sem6_credits: ValueWithSources = ValueWithSources()
    inst_ratio: ValueWithSources = ValueWithSources()


class ConferenceInfo(BaseModel):
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class UniversityRequirements(BaseModel):
    year3: Year3NCAA = Year3NCAA()
    year5: Year5NCAA = Year5NCAA()
    conference: ConferenceInfo = ConferenceInfo()
    # UNC-specific sections (optional, for UNC only)
    unc_year3_inst: Optional[UNCYear3Inst] = None
    unc_year5_inst: Optional[UNCYear5Inst] = None


class NCAAGeneral(BaseModel):
    semester_credits: ValueWithSources = ValueWithSources()
    annual_credits: ValueWithSources = ValueWithSources()
    apr: ValueWithSources = ValueWithSources()


class EligibilityExtraction(BaseModel):
    csun: Optional[UniversityRequirements] = None
    unc: Optional[UniversityRequirements] = None
    psu: Optional[UniversityRequirements] = None
    ncaa_general: Optional[NCAAGeneral] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_eligibility() -> str:
    return """
    Extract structured information from the answer for three universities (CSUN, UNC, and Penn State) and general NCAA requirements. 
    For each requested item, return both a 'value' (string, exactly as stated) and a list of 'urls' (all reference URLs given for that specific item).
    If an item is not mentioned, set its 'value' to null and its 'urls' to an empty list.

    Use the exact JSON schema below:

    {
      "csun": {
        "year3": {
          "ncaa_gpa": {"value": string|null, "urls": [string, ...]},
          "ncaa_credits": {"value": string|null, "urls": [string, ...]},
          "ncaa_percent": {"value": string|null, "urls": [string, ...]}
        },
        "year5": {
          "ncaa_gpa": {"value": string|null, "urls": [string, ...]},
          "ncaa_credits": {"value": string|null, "urls": [string, ...]},
          "major_declaration": {"value": string|null, "urls": [string, ...]},
          "ncaa_percent": {"value": string|null, "urls": [string, ...]}
        },
        "conference": {"name": string|null, "urls": [string, ...]}
      },
      "unc": {
        "year3": {
          "ncaa_gpa": {"value": string|null, "urls": [string, ...]},
          "ncaa_credits": {"value": string|null, "urls": [string, ...]},
          "ncaa_percent": {"value": string|null, "urls": [string, ...]}
        },
        "unc_year3_inst": {
          "inst_gpa": {"value": string|null, "urls": [string, ...]},
          "inst_sem4_credits": {"value": string|null, "urls": [string, ...]},
          "inst_ratio": {"value": string|null, "urls": [string, ...]}
        },
        "year5": {
          "ncaa_gpa": {"value": string|null, "urls": [string, ...]},
          "ncaa_credits": {"value": string|null, "urls": [string, ...]},
          "major_declaration": {"value": string|null, "urls": [string, ...]},
          "ncaa_percent": {"value": string|null, "urls": [string, ...]}
        },
        "unc_year5_inst": {
          "inst_gpa": {"value": string|null, "urls": [string, ...]},
          "inst_sem6_credits": {"value": string|null, "urls": [string, ...]},
          "inst_ratio": {"value": string|null, "urls": [string, ...]}
        },
        "conference": {"name": string|null, "urls": [string, ...]}
      },
      "psu": {
        "year3": {
          "ncaa_gpa": {"value": string|null, "urls": [string, ...]},
          "ncaa_credits": {"value": string|null, "urls": [string, ...]},
          "ncaa_percent": {"value": string|null, "urls": [string, ...]}
        },
        "year5": {
          "ncaa_gpa": {"value": string|null, "urls": [string, ...]},
          "ncaa_credits": {"value": string|null, "urls": [string, ...]},
          "major_declaration": {"value": string|null, "urls": [string, ...]},
          "ncaa_percent": {"value": string|null, "urls": [string, ...]}
        },
        "conference": {"name": string|null, "urls": [string, ...]}
      },
      "ncaa_general": {
        "semester_credits": {"value": string|null, "urls": [string, ...]},
        "annual_credits": {"value": string|null, "urls": [string, ...]},
        "apr": {"value": string|null, "urls": [string, ...]}
      }
    }

    Rules:
    - 'value' must be extracted verbatim from the answer (keep formatting like '2.0', '24 credits', '40%', etc.).
    - 'urls' must be actual URLs cited for that specific item; include all if multiple are given. Accept markdown link targets or plain URLs.
    - If citations are grouped, assign the most relevant URLs to the corresponding items; if uncertain, still include them.
    - Do not invent values or URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions for building verification nodes                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls or []


def _value_present(value: Optional[str]) -> bool:
    return bool(value) and bool(str(value).strip())


async def _add_value_verification(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    value: Optional[str],
    urls: Optional[List[str]],
    claim_template: str,
    add_ins: str,
    critical: bool = True,
) -> None:
    """
    Add a value-verification leaf. If value is missing, mark it as failed via custom node.
    Otherwise, verify the claim against provided URLs.
    """
    urls = _safe_urls(urls)
    if not _value_present(value):
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=node_desc,
            parent=parent_node,
            critical=critical
        )
        return

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=critical
    )
    claim = claim_template.format(value=value)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=add_ins
    )


def _add_url_existence(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    urls: Optional[List[str]],
    critical: bool = True
) -> None:
    """Add a URL existence check node."""
    urls = _safe_urls(urls)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=critical
    )


# Common additional instructions used across verifications
COMMON_ADD_INS_NCAA = (
    "Focus on NCAA Division I continuing-eligibility progress-toward-degree standards. "
    "'Entering semester 5' = beginning of year 3; 'Entering semester 9' = beginning of year 5. "
    "Accept school athletics/compliance pages that restate NCAA requirements. "
    "Verify the numeric threshold or percentage pertains to the specified checkpoint."
)

UNC_ADD_INS = (
    "For UNC-specific items, verify institution policies for cumulative UNC GPA minimums, "
    "semester-based cumulative credit milestones (end of semester 4 or 6), "
    "and attempted-to-completed credit hour ratio (credit completion rate). "
    "Use UNC official sources (athletics compliance, registrar, undergraduate retention) when available."
)

CONF_ADD_INS = (
    "Verify the athletic conference affiliation for the specified university using official athletics or conference websites. "
    "Allow minor naming variants (e.g., 'ACC' vs 'Atlantic Coast Conference')."
)

GENERAL_ADD_INS = (
    "Verify general NCAA Division I minimums for per-semester and per-year degree-applicable credits and the minimum four-year APR required "
    "for postseason eligibility. Use NCAA manuals, official releases, or credible compliance pages."
)


# --------------------------------------------------------------------------- #
# University verification builders                                            #
# --------------------------------------------------------------------------- #
async def build_year3_nodes(
    evaluator: Evaluator,
    parent,
    uni_key: str,
    uni_name: str,
    y3: Optional[Year3NCAA]
) -> None:
    y3_node = evaluator.add_parallel(
        id=f"{uni_key}_year3",
        desc=f"{uni_name} entering third year (beginning of Year 3 / entering semester 5): NCAA requirements with citations per field",
        parent=parent,
        critical=False
    )

    y3 = y3 or Year3NCAA()

    # GPA
    await _add_value_verification(
        evaluator,
        y3_node,
        node_id=f"{uni_key}_y3_ncaa_gpa_value",
        node_desc="Provides the correct NCAA minimum GPA requirement for entering semester 5 (per constraints)",
        value=y3.ncaa_gpa.value,
        urls=y3.ncaa_gpa.urls,
        claim_template="For NCAA Division I continuing eligibility at the beginning of Year 3 (entering semester 5), the minimum cumulative GPA requirement is '{value}'.",
        add_ins=COMMON_ADD_INS_NCAA
    )
    _add_url_existence(
        evaluator,
        y3_node,
        node_id=f"{uni_key}_y3_ncaa_gpa_url",
        node_desc="Provides a reference URL for the Year-3/entering-semester-5 NCAA GPA requirement",
        urls=y3.ncaa_gpa.urls
    )

    # Credits
    await _add_value_verification(
        evaluator,
        y3_node,
        node_id=f"{uni_key}_y3_ncaa_credits_value",
        node_desc="Provides the correct NCAA minimum degree-applicable credit hours required by entering semester 5 (per constraints)",
        value=y3.ncaa_credits.value,
        urls=y3.ncaa_credits.urls,
        claim_template="By entering semester 5 (start of the third year), student-athletes must have completed at least '{value}' degree-applicable credit hours.",
        add_ins=COMMON_ADD_INS_NCAA
    )
    _add_url_existence(
        evaluator,
        y3_node,
        node_id=f"{uni_key}_y3_ncaa_credits_url",
        node_desc="Provides a reference URL for the Year-3/entering-semester-5 NCAA credit-hours requirement",
        urls=y3.ncaa_credits.urls
    )

    # Percent (end of year 2)
    await _add_value_verification(
        evaluator,
        y3_node,
        node_id=f"{uni_key}_y3_ncaa_percent_value",
        node_desc="Provides the correct % of degree requirements that must be completed by end of the second year (per constraints)",
        value=y3.ncaa_percent.value,
        urls=y3.ncaa_percent.urls,
        claim_template="By the end of the second year, student-athletes must have completed at least '{value}' of their degree requirements (NCAA progress-toward-degree percentage).",
        add_ins=COMMON_ADD_INS_NCAA
    )
    _add_url_existence(
        evaluator,
        y3_node,
        node_id=f"{uni_key}_y3_ncaa_percent_url",
        node_desc="Provides a reference URL for the Year-3 degree-progress percentage requirement",
        urls=y3.ncaa_percent.urls
    )


async def build_year5_nodes(
    evaluator: Evaluator,
    parent,
    uni_key: str,
    uni_name: str,
    y5: Optional[Year5NCAA]
) -> None:
    y5_node = evaluator.add_parallel(
        id=f"{uni_key}_year5",
        desc=f"{uni_name} entering fifth year (beginning of Year 5 / entering semester 9): NCAA requirements with citations per field",
        parent=parent,
        critical=False
    )

    y5 = y5 or Year5NCAA()

    # GPA
    await _add_value_verification(
        evaluator,
        y5_node,
        node_id=f"{uni_key}_y5_ncaa_gpa_value",
        node_desc="Provides the correct NCAA minimum GPA requirement applicable for entering semester 9 (i.e., semester 7 and beyond) (per constraints)",
        value=y5.ncaa_gpa.value,
        urls=y5.ncaa_gpa.urls,
        claim_template="For NCAA Division I continuing eligibility at entering semester 9 (fifth year), the minimum cumulative GPA requirement is '{value}'.",
        add_ins=COMMON_ADD_INS_NCAA
    )
    _add_url_existence(
        evaluator,
        y5_node,
        node_id=f"{uni_key}_y5_ncaa_gpa_url",
        node_desc="Provides a reference URL for the Year-5/entering-semester-9 NCAA GPA requirement",
        urls=y5.ncaa_gpa.urls
    )

    # Credits
    await _add_value_verification(
        evaluator,
        y5_node,
        node_id=f"{uni_key}_y5_ncaa_credits_value",
        node_desc="Provides the correct NCAA minimum degree-applicable credit hours required by entering semester 9 (per constraints)",
        value=y5.ncaa_credits.value,
        urls=y5.ncaa_credits.urls,
        claim_template="By entering semester 9 (fifth year), student-athletes must have completed at least '{value}' degree-applicable credit hours.",
        add_ins=COMMON_ADD_INS_NCAA
    )
    _add_url_existence(
        evaluator,
        y5_node,
        node_id=f"{uni_key}_y5_ncaa_credits_url",
        node_desc="Provides a reference URL for the Year-5/entering-semester-9 NCAA credit-hours requirement",
        urls=y5.ncaa_credits.urls
    )

    # Major declaration (status by Year 5)
    await _add_value_verification(
        evaluator,
        y5_node,
        node_id=f"{uni_key}_y5_major_declaration_value",
        node_desc="States the NCAA major-declaration requirement status applicable by Year 5 (declared major required by entering semester 5, per constraints)",
        value=y5.major_declaration.value,
        urls=y5.major_declaration.urls,
        claim_template="NCAA DI requires a declared major by entering semester 5 (beginning of year 3). By Year 5, this requirement must already be met. The answer's stated status is '{value}'. Verify that the linked source supports the requirement.",
        add_ins=COMMON_ADD_INS_NCAA
    )
    _add_url_existence(
        evaluator,
        y5_node,
        node_id=f"{uni_key}_y5_major_declaration_url",
        node_desc="Provides a reference URL for the major-declaration requirement",
        urls=y5.major_declaration.urls
    )

    # Percent (end of year 3)
    await _add_value_verification(
        evaluator,
        y5_node,
        node_id=f"{uni_key}_y5_ncaa_percent_value",
        node_desc="Provides the correct % of degree requirements that must be completed by end of the third year (per constraints)",
        value=y5.ncaa_percent.value,
        urls=y5.ncaa_percent.urls,
        claim_template="By the end of the third year, student-athletes must have completed at least '{value}' of their degree requirements (NCAA progress-toward-degree percentage).",
        add_ins=COMMON_ADD_INS_NCAA
    )
    _add_url_existence(
        evaluator,
        y5_node,
        node_id=f"{uni_key}_y5_ncaa_percent_url",
        node_desc="Provides a reference URL for the Year-5 section’s degree-progress percentage requirement",
        urls=y5.ncaa_percent.urls
    )


async def build_unc_specific_year3(
    evaluator: Evaluator,
    parent,
    unc_inst: Optional[UNCYear3Inst]
) -> None:
    if not unc_inst:
        # If UNC-specific section not provided, still create nodes that fail (to reflect missing required items)
        unc_inst = UNCYear3Inst()

    # UNC-specific GPA
    await _add_value_verification(
        evaluator,
        parent,
        node_id="unc_y3_inst_gpa_value",
        node_desc="Provides UNC-specific minimum cumulative UNC GPA requirement (per constraints, if specified)",
        value=unc_inst.inst_gpa.value,
        urls=unc_inst.inst_gpa.urls,
        claim_template="UNC-specific minimum cumulative UNC GPA requirement by entering semester 5 is '{value}'.",
        add_ins=UNC_ADD_INS
    )
    _add_url_existence(
        evaluator,
        parent,
        node_id="unc_y3_inst_gpa_url",
        node_desc="Provides a reference URL for the UNC-specific cumulative GPA requirement",
        urls=unc_inst.inst_gpa.urls
    )

    # UNC-specific semester 4 credits
    await _add_value_verification(
        evaluator,
        parent,
        node_id="unc_y3_inst_sem4_credits_value",
        node_desc="Provides UNC-specific minimum semester credit hours by end of semester 4 (per constraints, if specified)",
        value=unc_inst.inst_sem4_credits.value,
        urls=unc_inst.inst_sem4_credits.urls,
        claim_template="UNC-specific minimum total degree-applicable credit hours required by the end of semester 4 is '{value}'.",
        add_ins=UNC_ADD_INS
    )
    _add_url_existence(
        evaluator,
        parent,
        node_id="unc_y3_inst_sem4_credits_url",
        node_desc="Provides a reference URL for the UNC-specific semester-4 credit-hours requirement",
        urls=unc_inst.inst_sem4_credits.urls
    )

    # UNC-specific attempted-to-completed ratio
    await _add_value_verification(
        evaluator,
        parent,
        node_id="unc_y3_inst_ratio_value",
        node_desc="Provides UNC-specific attempted-to-completed credit hour ratio requirement (per constraints, if specified)",
        value=unc_inst.inst_ratio.value,
        urls=unc_inst.inst_ratio.urls,
        claim_template="UNC-specific attempted-to-completed credit hour ratio requirement is '{value}'.",
        add_ins=UNC_ADD_INS
    )
    _add_url_existence(
        evaluator,
        parent,
        node_id="unc_y3_inst_ratio_url",
        node_desc="Provides a reference URL for the UNC-specific attempted-to-completed ratio requirement",
        urls=unc_inst.inst_ratio.urls
    )


async def build_unc_specific_year5(
    evaluator: Evaluator,
    parent,
    unc_inst: Optional[UNCYear5Inst]
) -> None:
    if not unc_inst:
        unc_inst = UNCYear5Inst()

    # UNC-specific GPA by end of third year
    await _add_value_verification(
        evaluator,
        parent,
        node_id="unc_y5_inst_gpa_value",
        node_desc="Provides UNC-specific minimum cumulative UNC GPA by end of the third year (as requested in the question; may be supported by constraints if given)",
        value=unc_inst.inst_gpa.value,
        urls=unc_inst.inst_gpa.urls,
        claim_template="UNC-specific minimum cumulative UNC GPA by the end of the third year is '{value}'.",
        add_ins=UNC_ADD_INS
    )
    _add_url_existence(
        evaluator,
        parent,
        node_id="unc_y5_inst_gpa_url",
        node_desc="Provides a reference URL for the UNC-specific cumulative GPA standard (Year 5 section)",
        urls=unc_inst.inst_gpa.urls
    )

    # UNC-specific semester 6 credits
    await _add_value_verification(
        evaluator,
        parent,
        node_id="unc_y5_inst_sem6_credits_value",
        node_desc="Provides UNC-specific minimum semester credit hours by end of semester 6 (per constraints, if specified)",
        value=unc_inst.inst_sem6_credits.value,
        urls=unc_inst.inst_sem6_credits.urls,
        claim_template="UNC-specific minimum total degree-applicable credit hours required by the end of semester 6 is '{value}'.",
        add_ins=UNC_ADD_INS
    )
    _add_url_existence(
        evaluator,
        parent,
        node_id="unc_y5_inst_sem6_credits_url",
        node_desc="Provides a reference URL for the UNC-specific semester-6 credit-hours requirement",
        urls=unc_inst.inst_sem6_credits.urls
    )

    # UNC-specific attempted-to-completed ratio
    await _add_value_verification(
        evaluator,
        parent,
        node_id="unc_y5_inst_ratio_value",
        node_desc="Provides UNC-specific attempted-to-completed credit hour ratio requirement (per constraints, if specified)",
        value=unc_inst.inst_ratio.value,
        urls=unc_inst.inst_ratio.urls,
        claim_template="UNC-specific attempted-to-completed credit hour ratio requirement is '{value}'.",
        add_ins=UNC_ADD_INS
    )
    _add_url_existence(
        evaluator,
        parent,
        node_id="unc_y5_inst_ratio_url",
        node_desc="Provides a reference URL for the UNC-specific attempted-to-completed ratio requirement (Year 5 section)",
        urls=unc_inst.inst_ratio.urls
    )


async def build_conference_nodes(
    evaluator: Evaluator,
    parent,
    uni_key: str,
    uni_name: str,
    conf: Optional[ConferenceInfo]
) -> None:
    conf_node = evaluator.add_parallel(
        id=f"{uni_key}_conference",
        desc=f"{uni_name} conference affiliation with citation",
        parent=parent,
        critical=False
    )
    conf = conf or ConferenceInfo()

    # Conference name verification
    await _add_value_verification(
        evaluator,
        conf_node,
        node_id=f"{uni_key}_conference_name",
        node_desc=f"Provides the correct athletic conference name for {uni_name.split('(')[0].strip()} (per constraints, if specified)" if uni_key != "unc" else "Provides the name of the athletic conference UNC competes in",
        value=conf.name,
        urls=conf.urls,
        claim_template=f"{uni_name} competes in the '{{value}}'.",
        add_ins=CONF_ADD_INS
    )
    _add_url_existence(
        evaluator,
        conf_node,
        node_id=f"{uni_key}_conference_url",
        node_desc=f"Provides a reference URL confirming {uni_name.split('(')[0].strip()} conference affiliation" if uni_key != "unc" else "Provides a reference URL confirming UNC conference affiliation",
        urls=conf.urls
    )


async def build_university_section(
    evaluator: Evaluator,
    root,
    uni_key: str,
    uni_name: str,
    uni_data: Optional[UniversityRequirements],
    include_unc_specific: bool = False
) -> None:
    # Parent node per university
    parent = evaluator.add_parallel(
        id=f"{uni_key}_requirements",
        desc=f"{uni_name}: required Year 3 + Year 5 NCAA requirements{' and conference affiliation' if not include_unc_specific else ', UNC-specific requirements, and conference affiliation'}, each with per-field citations",
        parent=root,
        critical=False
    )

    # Year 3
    await build_year3_nodes(evaluator, parent, uni_key, uni_name, (uni_data.year3 if uni_data else None))

    # UNC-specific Year 3 (only if UNC)
    if include_unc_specific:
        unc_y3_node = evaluator.add_parallel(
            id="unc_year3",
            desc="UNC entering third year (beginning of Year 3 / entering semester 5): NCAA + UNC-specific requirements with citations per field",
            parent=parent,
            critical=False
        )
        # NCAA items for UNC are already handled via build_year3_nodes under the same university section,
        # UNC-specific items:
        await build_unc_specific_year3(evaluator, unc_y3_node, (uni_data.unc_year3_inst if uni_data else None))

    # Year 5
    await build_year5_nodes(evaluator, parent, uni_key, uni_name, (uni_data.year5 if uni_data else None))

    # UNC-specific Year 5 (only if UNC)
    if include_unc_specific:
        unc_y5_node = evaluator.add_parallel(
            id="unc_year5",
            desc="UNC entering fifth year (beginning of Year 5 / entering semester 9): NCAA + UNC-specific requirements with citations per field",
            parent=parent,
            critical=False
        )
        await build_unc_specific_year5(evaluator, unc_y5_node, (uni_data.unc_year5_inst if uni_data else None))

    # Conference
    await build_conference_nodes(evaluator, parent, uni_key, uni_name, (uni_data.conference if uni_data else None))


# --------------------------------------------------------------------------- #
# General NCAA verification                                                   #
# --------------------------------------------------------------------------- #
async def build_ncaa_general_section(evaluator: Evaluator, root, general: Optional[NCAAGeneral]) -> None:
    parent = evaluator.add_parallel(
        id="ncaa_general",
        desc="General NCAA Division I requirements applicable to all three institutions, each with per-field citations",
        parent=root,
        critical=False
    )

    general = general or NCAAGeneral()

    # Per semester credits
    await _add_value_verification(
        evaluator,
        parent,
        node_id="ncaa_semester_credits_value",
        node_desc="Provides the NCAA minimum degree-applicable credit hours required per semester (per constraints)",
        value=general.semester_credits.value,
        urls=general.semester_credits.urls,
        claim_template="The NCAA Division I minimum degree-applicable credit hours required per semester is '{value}'.",
        add_ins=GENERAL_ADD_INS
    )
    _add_url_existence(
        evaluator,
        parent,
        node_id="ncaa_semester_credits_url",
        node_desc="Provides a reference URL for NCAA per-semester credit-hour requirement",
        urls=general.semester_credits.urls
    )

    # Per academic year credits
    await _add_value_verification(
        evaluator,
        parent,
        node_id="ncaa_annual_credits_value",
        node_desc="Provides the NCAA minimum degree-applicable credit hours required per academic year (fall + spring) (per constraints)",
        value=general.annual_credits.value,
        urls=general.annual_credits.urls,
        claim_template="The NCAA Division I minimum degree-applicable credit hours required per academic year (fall + spring) is '{value}'.",
        add_ins=GENERAL_ADD_INS
    )
    _add_url_existence(
        evaluator,
        parent,
        node_id="ncaa_annual_credits_url",
        node_desc="Provides a reference URL for NCAA per-year credit-hour requirement",
        urls=general.annual_credits.urls
    )

    # APR
    await _add_value_verification(
        evaluator,
        parent,
        node_id="ncaa_apr_value",
        node_desc="Provides the NCAA minimum four-year APR required for postseason eligibility (per constraints)",
        value=general.apr.value,
        urls=general.apr.urls,
        claim_template="The NCAA Division I minimum four-year APR required for postseason competition eligibility is '{value}'.",
        add_ins=GENERAL_ADD_INS
    )
    _add_url_existence(
        evaluator,
        parent,
        node_id="ncaa_apr_url",
        node_desc="Provides a reference URL for NCAA APR postseason eligibility requirement",
        urls=general.apr.urls
    )


# --------------------------------------------------------------------------- #
# Response structure verification                                             #
# --------------------------------------------------------------------------- #
async def verify_response_structure(evaluator: Evaluator, root) -> None:
    """
    Critical structural check: The answer must be clearly organized by university (CSUN, UNC, Penn State)
    and requirement categories: Year 3, Year 5, Conference, plus a General NCAA section.
    """
    node = evaluator.add_leaf(
        id="response_structure",
        desc="Answer is clearly structured by university and requirement category (Year 3, Year 5, Conference, and General NCAA section)",
        parent=root,
        critical=True
    )
    claim = (
        "The answer is clearly structured by university (CSUN, UNC, Penn State) and includes grouped sections for "
        "Year 3 (entering semester 5), Year 5 (entering semester 9), Conference affiliation for each, and a distinct "
        "General NCAA requirements section. Headings or grouped bullets make this organization evident."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction=(
            "Judge the formatting and grouping in the provided answer text itself. "
            "Look for clear headings or grouped bullets per university and required categories, "
            "and a separate General NCAA section."
        )
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
    Evaluate the eligibility guide answer for CSUN, UNC, and Penn State with NCAA general checks.
    """
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
        default_model=model
    )

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_eligibility(),
        template_class=EligibilityExtraction,
        extraction_name="eligibility_extraction"
    )

    # 1) Structure check (critical)
    await verify_response_structure(evaluator, root)

    # 2) University-specific sections
    await build_university_section(
        evaluator,
        root,
        uni_key="csun",
        uni_name="California State University, Northridge (CSUN)",
        uni_data=extracted.csun,
        include_unc_specific=False
    )

    await build_university_section(
        evaluator,
        root,
        uni_key="unc",
        uni_name="University of North Carolina at Chapel Hill (UNC)",
        uni_data=extracted.unc,
        include_unc_specific=True
    )

    await build_university_section(
        evaluator,
        root,
        uni_key="psu",
        uni_name="Penn State University",
        uni_data=extracted.psu,
        include_unc_specific=False
    )

    # 3) General NCAA requirements
    await build_ncaa_general_section(evaluator, root, extracted.ncaa_general)

    # Return summarized evaluation
    return evaluator.get_summary()