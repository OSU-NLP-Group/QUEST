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
TASK_ID = "maduro_case_research_2026"
TASK_DESCRIPTION = """Conduct comprehensive legal research on the United States federal prosecution of Nicolás Maduro following his capture in January 2026. Provide the following information, with each item supported by URLs to official government sources, court documents, or reputable news organizations:

1. Defendant Information:
   - The primary defendant's full legal name as it appears in court documents
   - The name and relationship of the co-defendant charged alongside the primary defendant

2. Court Jurisdiction and Officials:
   - The full official name of the U.S. federal district court with jurisdiction over this case
   - The city where the court proceedings are taking place
   - The full name and title of the presiding U.S. District Judge

3. Criminal Charges:
   - All criminal charges filed against the defendant in the federal indictment, including specific charges related to narco-terrorism, drug trafficking, and weapons

4. Timeline of Legal Proceedings:
   - The exact date (month, day, year) when the defendant was captured
   - The official code name of the U.S. military operation (if available)
   - The date of the defendant's arraignment
   - The plea entered by the defendant at arraignment
   - Whether bail was granted or denied
   - The date of the next scheduled court hearing

5. Detention Information:
   - The full official name of the federal detention facility where the defendant is being held
   - The city and state location of the detention facility

6. International Legal Framework:
   - Confirmation of whether a bilateral extradition treaty exists between the United States and Venezuela
   - The year when this extradition treaty was signed

7. Legal Precedents:
   - The full case name and year of the U.S. Supreme Court precedent establishing that forcible abduction from foreign soil does not preclude prosecution in U.S. courts
   - (Optional) Another Supreme Court case addressing sovereign or head-of-state immunity principles

8. Executive Branch Documentation:
   - The date when the Office of Legal Counsel (OLC) memo providing legal justification for the military operation was issued
   - (Optional) The name of the OLC official who signed the memo
   - The name of the U.S. Attorney General who announced the charges

9. Additional Information (if readily available):
   - The maximum reward amount offered by the U.S. government for the defendant's arrest
   - The month and year when criminal charges were first filed against the defendant in the Southern District of New York

For each piece of information, provide a direct link to an official source (such as Department of Justice press releases, State Department pages, federal court documents, Congressional Research Service reports, or credible news articles from established outlets) that confirms the stated fact.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DefendantInfo(BaseModel):
    full_legal_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CoDefendantInfo(BaseModel):
    name: Optional[str] = None
    relationship_to_primary_defendant: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CourtJurisdiction(BaseModel):
    district_court_name: Optional[str] = None
    court_location_city: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PresidingJudge(BaseModel):
    judge_name: Optional[str] = None
    judge_title: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ChargesInfo(BaseModel):
    narco_terrorism_charge: Optional[str] = None
    cocaine_importation_charge: Optional[str] = None
    weapons_charges: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CaptureInfo(BaseModel):
    exact_date: Optional[str] = None
    operation_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ArraignmentInfo(BaseModel):
    arraignment_date: Optional[str] = None
    plea_entered: Optional[str] = None
    bail_decision: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NextHearingInfo(BaseModel):
    hearing_date: Optional[str] = None
    hearing_type: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DetentionInfo(BaseModel):
    facility_name: Optional[str] = None
    facility_location: Optional[str] = None
    facility_type: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TreatyInfo(BaseModel):
    treaty_existence: Optional[str] = None  # expected values: "yes", "no", "unknown"
    signing_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PrecedentsInfo(BaseModel):
    alvarez_machain_case_name: Optional[str] = None
    alvarez_machain_year_decided: Optional[str] = None
    alvarez_sources: List[str] = Field(default_factory=list)

    head_of_state_case_name: Optional[str] = None
    head_of_state_year_decided: Optional[str] = None
    head_sources: List[str] = Field(default_factory=list)


class OLCMemoInfo(BaseModel):
    memo_date: Optional[str] = None
    memo_author: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AttorneyGeneralInfo(BaseModel):
    ag_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RewardInfo(BaseModel):
    maximum_reward: Optional[str] = None
    reward_increase_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class InitialChargesInfo(BaseModel):
    first_indictment_date: Optional[str] = None  # month and year, e.g., "March 2020"
    sources: List[str] = Field(default_factory=list)


class CaseExtraction(BaseModel):
    defendant: Optional[DefendantInfo] = None
    co_defendant: Optional[CoDefendantInfo] = None
    court: Optional[CourtJurisdiction] = None
    judge: Optional[PresidingJudge] = None
    charges: Optional[ChargesInfo] = None
    capture: Optional[CaptureInfo] = None
    arraignment: Optional[ArraignmentInfo] = None
    next_hearing: Optional[NextHearingInfo] = None
    detention: Optional[DetentionInfo] = None
    treaty: Optional[TreatyInfo] = None
    precedents: Optional[PrecedentsInfo] = None
    olc_memo: Optional[OLCMemoInfo] = None
    attorney_general: Optional[AttorneyGeneralInfo] = None
    reward: Optional[RewardInfo] = None
    initial_charges: Optional[InitialChargesInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_case_info() -> str:
    return """
    Extract structured information from the answer for the United States v. Nicolás Maduro federal case. For each field below, extract the value exactly as presented in the answer (do not infer), and also collect any URLs the answer cites to support that field. Only include URLs that appear in the answer text.

    You must fill these objects and fields:

    defendant:
      - full_legal_name (string or null)
      - sources (array of URLs; possibly empty)

    co_defendant:
      - name (string or null)
      - relationship_to_primary_defendant (string or null)
      - sources (array of URLs)

    court:
      - district_court_name (string or null)
      - court_location_city (string or null)
      - sources (array of URLs)

    judge:
      - judge_name (string or null)
      - judge_title (string or null)
      - sources (array of URLs)

    charges:
      - narco_terrorism_charge (string describing the charge or null)
      - cocaine_importation_charge (string describing the charge or null)
      - weapons_charges (string describing the charge or null)
      - sources (array of URLs)

    capture:
      - exact_date (string like 'January 12, 2026' or null)
      - operation_name (string or null)
      - sources (array of URLs)

    arraignment:
      - arraignment_date (string date or null)
      - plea_entered (string like 'not guilty' or null)
      - bail_decision (string like 'denied' or 'granted' or null)
      - sources (array of URLs)

    next_hearing:
      - hearing_date (string date or null)
      - hearing_type (string or null)
      - sources (array of URLs)

    detention:
      - facility_name (string or null)
      - facility_location (string like 'City, State' or null)
      - facility_type (string or null)
      - sources (array of URLs)

    treaty:
      - treaty_existence (string one of 'yes', 'no', or 'unknown'; do not invent)
      - signing_year (string like '1922' or null)
      - sources (array of URLs)

    precedents:
      - alvarez_machain_case_name (string case citation or null)
      - alvarez_machain_year_decided (string year or null)
      - alvarez_sources (array of URLs)
      - head_of_state_case_name (string or null)
      - head_of_state_year_decided (string or null)
      - head_sources (array of URLs)

    olc_memo:
      - memo_date (string date or null)
      - memo_author (string or null)
      - sources (array of URLs)

    attorney_general:
      - ag_name (string or null)
      - sources (array of URLs)

    reward:
      - maximum_reward (string like '$15,000,000' or '15 million' or null)
      - reward_increase_date (string like 'March 2020' or full date or null)
      - sources (array of URLs)

    initial_charges:
      - first_indictment_date (string 'Month Year' or null)
      - sources (array of URLs)

    Rules:
    - Return null for any missing field.
    - The URLs must be exactly as shown in the answer; extract full URLs. Accept plain URLs or those inside markdown links.
    - Do not deduplicate across categories; keep sources per category as cited for that category.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_sources(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


def non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_defendant_identification(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    node = evaluator.add_parallel(
        id="Defendant_Identification",
        desc="Correctly identify the primary defendant in the case with full legal name",
        parent=parent_node,
        critical=True
    )
    name_present = non_empty(data.defendant.full_legal_name) if data.defendant else False
    srcs = safe_sources(data.defendant.sources if data.defendant else [])
    sources_present = len(srcs) > 0

    evaluator.add_custom_node(
        result=name_present,
        id="Defendant_Name_Provided",
        desc="Defendant name provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=sources_present,
        id="Defendant_Reference_URL_Provided",
        desc="Provide a URL to an official source confirming the defendant's identity",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Full_Legal_Name",
        desc="Provide the defendant's complete legal name as it appears in court documents",
        parent=node,
        critical=True
    )
    claim = f"The primary defendant's full legal name is '{data.defendant.full_legal_name}' as stated in official court or DOJ documents."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction="Verify the name against DOJ press releases, court dockets, or indictments. Allow minor formatting variations (accents, capitalization)."
    )


async def build_co_defendant(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    node = evaluator.add_parallel(
        id="Co_Defendant",
        desc="Identify the co-defendant charged alongside the primary defendant",
        parent=parent_node,
        critical=True
    )
    info = data.co_defendant or CoDefendantInfo()
    srcs = safe_sources(info.sources)

    evaluator.add_custom_node(
        result=non_empty(info.name),
        id="Co_Defendant_Name_Provided",
        desc="Co-defendant name provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(info.relationship_to_primary_defendant),
        id="Co_Defendant_Relationship_Provided",
        desc="Relationship to primary defendant provided",
        parent=node,
        critical=True
    )

    name_leaf = evaluator.add_leaf(
        id="Co_Defendant_Name",
        desc="Provide the full name of the co-defendant",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The co-defendant's full name is '{info.name}'.",
        node=name_leaf,
        sources=srcs,
        additional_instruction="Confirm the co-defendant's name from DOJ, court documents, or credible reporting."
    )

    rel_leaf = evaluator.add_leaf(
        id="Relationship_to_Primary_Defendant",
        desc="Specify the relationship between the co-defendant and primary defendant",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The co-defendant's relationship to Nicolás Maduro is: {info.relationship_to_primary_defendant}.",
        node=rel_leaf,
        sources=srcs,
        additional_instruction="Verify that the described relationship is explicitly supported by the cited source(s)."
    )


async def build_court_jurisdiction(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    node = evaluator.add_parallel(
        id="Federal_Court_Jurisdiction",
        desc="Identify the specific United States federal district court with jurisdiction over the case",
        parent=parent_node,
        critical=True
    )
    info = data.court or CourtJurisdiction()
    srcs = safe_sources(info.sources)
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id="Court_Reference_URL_Provided",
        desc="Provide a URL to an official source confirming the court jurisdiction",
        parent=node,
        critical=True
    )

    court_leaf = evaluator.add_leaf(
        id="District_Court_Name",
        desc="Provide the full official name of the U.S. District Court",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The court with jurisdiction is '{info.district_court_name}'.",
        node=court_leaf,
        sources=srcs,
        additional_instruction="Confirm the district court name from DOJ or court docket sources."
    )

    city_leaf = evaluator.add_leaf(
        id="Court_Location_City",
        desc="Specify the city where the court proceedings are taking place",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The court proceedings are taking place in {info.court_location_city}.",
        node=city_leaf,
        sources=srcs,
        additional_instruction="Verify the city location of the proceedings (e.g., courtroom announcements, docket entries, DOJ press releases)."
    )


async def build_presiding_judge(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    # Critical group for judge name
    crit = evaluator.add_parallel(
        id="Presiding_Judge",
        desc="Identify the federal judge presiding over the case",
        parent=parent_node,
        critical=True
    )
    info = data.judge or PresidingJudge()
    srcs = safe_sources(info.sources)

    evaluator.add_custom_node(
        result=non_empty(info.judge_name),
        id="Judge_Name_Provided",
        desc="Judge name provided",
        parent=crit,
        critical=True
    )

    judge_leaf = evaluator.add_leaf(
        id="Judge_Name",
        desc="Provide the full name of the presiding U.S. District Judge",
        parent=crit,
        critical=True
    )
    await evaluator.verify(
        claim=f"The presiding judge is {info.judge_name}.",
        node=judge_leaf,
        sources=srcs,
        additional_instruction="Confirm presiding judge name via court docket or DOJ press release."
    )

    # Optional group for title
    opt = evaluator.add_parallel(
        id="Presiding_Judge_Optional",
        desc="Optional judge title verification",
        parent=parent_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=non_empty(info.judge_title),
        id="Judge_Title_Provided",
        desc="Judge title provided",
        parent=opt,
        critical=True
    )
    title_leaf = evaluator.add_leaf(
        id="Judge_Title",
        desc="Specify the judge's official title",
        parent=opt,
        critical=True
    )
    await evaluator.verify(
        claim=f"The judge's official title is '{info.judge_title}'.",
        node=title_leaf,
        sources=srcs,
        additional_instruction="Verify title such as 'U.S. District Judge' explicitly stated in the source."
    )


async def build_criminal_charges(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    node = evaluator.add_parallel(
        id="Criminal_Charges",
        desc="Identify all criminal charges filed against the defendant in the federal indictment",
        parent=parent_node,
        critical=True
    )
    info = data.charges or ChargesInfo()
    srcs = safe_sources(info.sources)
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id="Charges_Reference_URL_Provided",
        desc="Provide a URL to an official DOJ or court document listing the charges",
        parent=node,
        critical=True
    )

    narco_leaf = evaluator.add_leaf(
        id="Narco_Terrorism_Charge",
        desc="Verify that narco-terrorism conspiracy is among the charges",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The indictment includes a charge related to narco-terrorism conspiracy against Nicolás Maduro.",
        node=narco_leaf,
        sources=srcs,
        additional_instruction="Check DOJ press releases or indictments for 'narco-terrorism' or equivalent phrasing."
    )

    cocaine_leaf = evaluator.add_leaf(
        id="Cocaine_Importation_Charge",
        desc="Verify that conspiracy to import cocaine is among the charges",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The indictment includes a charge of conspiracy to import cocaine into the United States.",
        node=cocaine_leaf,
        sources=srcs,
        additional_instruction="Allow equivalent phrasing like 'conspiracy to distribute/import cocaine'."
    )

    weapons_leaf = evaluator.add_leaf(
        id="Weapons_Charges",
        desc="Verify that charges related to machine guns and destructive devices are included",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The indictment includes charges related to machine guns and destructive devices.",
        node=weapons_leaf,
        sources=srcs,
        additional_instruction="Look for references to 18 U.S.C. §§ 924(c), 924(o), etc., or machine guns/destructive devices being cited."
    )


async def build_capture_date(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    # Critical: exact capture date
    crit = evaluator.add_parallel(
        id="Capture_Date",
        desc="Identify when the defendant was captured by U.S. forces",
        parent=parent_node,
        critical=True
    )
    info = data.capture or CaptureInfo()
    srcs = safe_sources(info.sources)

    evaluator.add_custom_node(
        result=non_empty(info.exact_date),
        id="Capture_Exact_Date_Provided",
        desc="Exact capture date provided",
        parent=crit,
        critical=True
    )
    exact_leaf = evaluator.add_leaf(
        id="Exact_Date",
        desc="Provide the exact date of capture (month, day, and year)",
        parent=crit,
        critical=True
    )
    await evaluator.verify(
        claim=f"Nicolás Maduro was captured on {info.exact_date}.",
        node=exact_leaf,
        sources=srcs,
        additional_instruction="Verify the exact capture date from official statements or credible reporting."
    )

    # Optional: operation name
    opt = evaluator.add_parallel(
        id="Capture_Operation_Name_Optional",
        desc="Optional: Official code name of the U.S. military operation",
        parent=parent_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=non_empty(info.operation_name),
        id="Operation_Name_Provided",
        desc="Operation code name provided",
        parent=opt,
        critical=True
    )
    op_leaf = evaluator.add_leaf(
        id="Operation_Name",
        desc="Provide the official code name of the U.S. military operation",
        parent=opt,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official code name of the U.S. military operation was '{info.operation_name}'.",
        node=op_leaf,
        sources=srcs,
        additional_instruction="Confirm that the code name appears in official or reputable sources. If multiple names, accept reasonable variants."
    )


async def build_arraignment_info(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    node = evaluator.add_parallel(
        id="Arraignment_Information",
        desc="Provide details about the defendant's initial court appearance and arraignment",
        parent=parent_node,
        critical=True
    )
    info = data.arraignment or ArraignmentInfo()
    srcs = safe_sources(info.sources)

    # Date
    evaluator.add_custom_node(
        result=non_empty(info.arraignment_date),
        id="Arraignment_Date_Provided",
        desc="Arraignment date provided",
        parent=node,
        critical=True
    )
    date_leaf = evaluator.add_leaf(
        id="Arraignment_Date",
        desc="Specify the date when the arraignment occurred",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The arraignment occurred on {info.arraignment_date}.",
        node=date_leaf,
        sources=srcs,
        additional_instruction="Check court docket or DOJ press release for arraignment date."
    )

    # Plea
    evaluator.add_custom_node(
        result=non_empty(info.plea_entered),
        id="Plea_Entered_Provided",
        desc="Plea at arraignment provided",
        parent=node,
        critical=True
    )
    plea_leaf = evaluator.add_leaf(
        id="Plea_Entered",
        desc="State what plea the defendant entered at arraignment",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At arraignment, the defendant entered a plea of '{info.plea_entered}'.",
        node=plea_leaf,
        sources=srcs,
        additional_instruction="Verify wording such as 'not guilty' or 'guilty'."
    )

    # Bail
    evaluator.add_custom_node(
        result=non_empty(info.bail_decision),
        id="Bail_Decision_Provided",
        desc="Bail decision provided",
        parent=node,
        critical=True
    )
    bail_leaf = evaluator.add_leaf(
        id="Bail_Decision",
        desc="Indicate whether bail was granted or denied",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Bail was {info.bail_decision}.",
        node=bail_leaf,
        sources=srcs,
        additional_instruction="Confirm bail decision from docket or DOJ statements (e.g., 'detained' implies denied)."
    )


async def build_next_hearing(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    # Critical: hearing date
    crit = evaluator.add_parallel(
        id="Next_Court_Hearing",
        desc="Identify the date of the next scheduled court hearing or proceeding",
        parent=parent_node,
        critical=True
    )
    info = data.next_hearing or NextHearingInfo()
    srcs = safe_sources(info.sources)

    evaluator.add_custom_node(
        result=non_empty(info.hearing_date),
        id="Hearing_Date_Provided",
        desc="Next hearing date provided",
        parent=crit,
        critical=True
    )
    date_leaf = evaluator.add_leaf(
        id="Hearing_Date",
        desc="Provide the specific date of the next scheduled hearing",
        parent=crit,
        critical=True
    )
    await evaluator.verify(
        claim=f"The next court hearing is scheduled for {info.hearing_date}.",
        node=date_leaf,
        sources=srcs,
        additional_instruction="Verify from court docket or DOJ press release."
    )

    # Optional: hearing type
    opt = evaluator.add_parallel(
        id="Hearing_Type_Optional",
        desc="Optional: Type or purpose of the scheduled hearing",
        parent=parent_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=non_empty(info.hearing_type),
        id="Hearing_Type_Provided",
        desc="Hearing type provided",
        parent=opt,
        critical=True
    )
    type_leaf = evaluator.add_leaf(
        id="Hearing_Type",
        desc="Specify the type or purpose of the scheduled hearing",
        parent=opt,
        critical=True
    )
    await evaluator.verify(
        claim=f"The next hearing type/purpose is '{info.hearing_type}'.",
        node=type_leaf,
        sources=srcs,
        additional_instruction="Confirm hearing type (e.g., status conference, detention hearing) from docket."
    )


async def build_detention_info(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    # Critical: name and location
    crit = evaluator.add_parallel(
        id="Detention_Facility",
        desc="Identify where the defendant is being held in federal custody pending trial",
        parent=parent_node,
        critical=True
    )
    info = data.detention or DetentionInfo()
    srcs = safe_sources(info.sources)

    evaluator.add_custom_node(
        result=non_empty(info.facility_name),
        id="Detention_Facility_Name_Provided",
        desc="Detention facility name provided",
        parent=crit,
        critical=True
    )
    name_leaf = evaluator.add_leaf(
        id="Facility_Name",
        desc="Provide the full official name of the federal detention facility",
        parent=crit,
        critical=True
    )
    await evaluator.verify(
        claim=f"The defendant is held at '{info.facility_name}'.",
        node=name_leaf,
        sources=srcs,
        additional_instruction="Verify from BOP custody records, DOJ, or credible reporting."
    )

    evaluator.add_custom_node(
        result=non_empty(info.facility_location),
        id="Detention_Facility_Location_Provided",
        desc="Detention facility location provided",
        parent=crit,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id="Facility_Location",
        desc="Specify the city and state where the detention facility is located",
        parent=crit,
        critical=True
    )
    await evaluator.verify(
        claim=f"The detention facility location is {info.facility_location}.",
        node=loc_leaf,
        sources=srcs,
        additional_instruction="Confirm city and state of the detention facility in the source."
    )

    # Optional: type
    opt = evaluator.add_parallel(
        id="Detention_Facility_Optional",
        desc="Optional: Detention facility type verification",
        parent=parent_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=non_empty(info.facility_type),
        id="Detention_Facility_Type_Provided",
        desc="Detention facility type provided",
        parent=opt,
        critical=True
    )
    type_leaf = evaluator.add_leaf(
        id="Facility_Type",
        desc="Indicate the type of facility",
        parent=opt,
        critical=True
    )
    await evaluator.verify(
        claim=f"The detention facility type is '{info.facility_type}'.",
        node=type_leaf,
        sources=srcs,
        additional_instruction="Verify whether it's an MDC, FCI, or other facility type."
    )


async def build_extradition_treaty(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    node = evaluator.add_parallel(
        id="Extradition_Treaty",
        desc="Provide information about the bilateral extradition treaty between the United States and Venezuela",
        parent=parent_node,
        critical=True
    )
    info = data.treaty or TreatyInfo()
    srcs = safe_sources(info.sources)

    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id="Treaty_Reference_URL_Provided",
        desc="Provide a URL to an official government or treaty database source confirming the treaty details",
        parent=node,
        critical=True
    )

    exist_leaf = evaluator.add_leaf(
        id="Treaty_Existence",
        desc="Confirm whether a bilateral extradition treaty exists between the U.S. and Venezuela",
        parent=node,
        critical=True
    )
    existence_str = (info.treaty_existence or "").strip().lower()
    if existence_str == "yes":
        claim = "There is a bilateral extradition treaty between the United States and Venezuela."
    elif existence_str == "no":
        claim = "There is no bilateral extradition treaty between the United States and Venezuela."
    else:
        claim = "The sources indicate the status of a bilateral extradition treaty between the United States and Venezuela."
    await evaluator.verify(
        claim=claim,
        node=exist_leaf,
        sources=srcs,
        additional_instruction="Confirm treaty status from official sources (e.g., State Department, treaty databases)."
    )

    year_leaf = evaluator.add_leaf(
        id="Treaty_Signing_Year",
        desc="Provide the year when the extradition treaty was signed",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extradition treaty (if applicable) was signed in {info.signing_year}.",
        node=year_leaf,
        sources=srcs,
        additional_instruction="Verify the signing year from official treaty records or authoritative databases."
    )


async def build_legal_precedents(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    # Critical: Alvarez-Machain
    am_node = evaluator.add_parallel(
        id="Alvarez_Machain_Case",
        desc="Identify the Supreme Court case establishing that forcible abduction from foreign soil does not preclude prosecution in U.S. courts",
        parent=parent_node,
        critical=True
    )
    info = data.precedents or PrecedentsInfo()
    am_srcs = safe_sources(info.alvarez_sources)

    evaluator.add_custom_node(
        result=non_empty(info.alvarez_machain_case_name),
        id="Alvarez_Case_Name_Provided",
        desc="Alvarez-Machain case name provided",
        parent=am_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(info.alvarez_machain_year_decided),
        id="Alvarez_Year_Provided",
        desc="Alvarez-Machain year provided",
        parent=am_node,
        critical=True
    )

    case_leaf = evaluator.add_leaf(
        id="Alvarez_Case_Name",
        desc="Provide the full case citation",
        parent=am_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The precedent case establishing the principle is '{info.alvarez_machain_case_name}'.",
        node=case_leaf,
        sources=am_srcs,
        additional_instruction="Verify case name and that it establishes the abduction principle (e.g., United States v. Alvarez-Machain, 1992)."
    )

    year_leaf = evaluator.add_leaf(
        id="Alvarez_Year_Decided",
        desc="Provide the year the case was decided",
        parent=am_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The case was decided in {info.alvarez_machain_year_decided}.",
        node=year_leaf,
        sources=am_srcs,
        additional_instruction="Confirm the decision year from authoritative sources (Supreme Court websites, Oyez, Justia)."
    )

    # Optional: Head-of-state immunity case
    hs_node = evaluator.add_parallel(
        id="Head_of_State_Immunity_Case_Optional",
        desc="Optional Supreme Court case addressing sovereign or head-of-state immunity",
        parent=parent_node,
        critical=False
    )
    hs_srcs = safe_sources(info.head_sources)

    evaluator.add_custom_node(
        result=non_empty(info.head_of_state_case_name),
        id="Head_Case_Name_Provided",
        desc="Head-of-state immunity case name provided",
        parent=hs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(info.head_of_state_year_decided),
        id="Head_Year_Provided",
        desc="Head-of-state immunity case year provided",
        parent=hs_node,
        critical=True
    )

    hs_case_leaf = evaluator.add_leaf(
        id="Head_Case_Name",
        desc="Provide a case name that addresses sovereign or head-of-state immunity",
        parent=hs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The case addressing sovereign or head-of-state immunity is '{info.head_of_state_case_name}'.",
        node=hs_case_leaf,
        sources=hs_srcs,
        additional_instruction="Verify the case discusses sovereign or head-of-state immunity principles."
    )

    hs_year_leaf = evaluator.add_leaf(
        id="Head_Year_Decided",
        desc="Provide the year the case was decided",
        parent=hs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The head-of-state immunity case was decided in {info.head_of_state_year_decided}.",
        node=hs_year_leaf,
        sources=hs_srcs,
        additional_instruction="Confirm the decision year from credible or official case sources."
    )


async def build_olc_memo(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    # Critical: memo date + reference URL
    crit = evaluator.add_parallel(
        id="OLC_Legal_Memo",
        desc="Provide information about the Office of Legal Counsel memorandum",
        parent=parent_node,
        critical=True
    )
    info = data.olc_memo or OLCMemoInfo()
    srcs = safe_sources(info.sources)

    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id="OLC_Memo_Reference_URL_Provided",
        desc="Provide a URL where the memo or information about it can be found",
        parent=crit,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(info.memo_date),
        id="OLC_Memo_Date_Provided",
        desc="Memo date provided",
        parent=crit,
        critical=True
    )

    date_leaf = evaluator.add_leaf(
        id="Memo_Date",
        desc="Provide the date when the OLC memo was issued or signed",
        parent=crit,
        critical=True
    )
    await evaluator.verify(
        claim=f"The OLC memo was issued/signed on {info.memo_date}.",
        node=date_leaf,
        sources=srcs,
        additional_instruction="Verify the memo date from OLC or DOJ official postings."
    )

    # Optional: memo author
    opt = evaluator.add_parallel(
        id="OLC_Memo_Optional",
        desc="Optional: OLC memo author",
        parent=parent_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=non_empty(info.memo_author),
        id="OLC_Memo_Author_Provided",
        desc="Memo author provided",
        parent=opt,
        critical=True
    )
    author_leaf = evaluator.add_leaf(
        id="Memo_Author",
        desc="Identify the OLC official who signed or authored the memo",
        parent=opt,
        critical=True
    )
    await evaluator.verify(
        claim=f"The OLC memo was authored/signed by '{info.memo_author}'.",
        node=author_leaf,
        sources=srcs,
        additional_instruction="Confirm author/signatory from the memo page or official source."
    )


async def build_attorney_general(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    node = evaluator.add_parallel(
        id="Attorney_General",
        desc="Identify the U.S. Attorney General who announced the charges and indictment",
        parent=parent_node,
        critical=True
    )
    info = data.attorney_general or AttorneyGeneralInfo()
    srcs = safe_sources(info.sources)

    evaluator.add_custom_node(
        result=non_empty(info.ag_name),
        id="AG_Name_Provided",
        desc="Attorney General name provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="AG_Name",
        desc="Provide the full name of the Attorney General",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The U.S. Attorney General who announced the charges is {info.ag_name}.",
        node=leaf,
        sources=srcs,
        additional_instruction="Verify from DOJ press releases or official announcements."
    )


async def build_reward_amount(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    node = evaluator.add_parallel(
        id="Reward_Amount",
        desc="Identify the reward amount offered by the U.S. government for information leading to the defendant's arrest",
        parent=parent_node,
        critical=False
    )
    info = data.reward or RewardInfo()
    srcs = safe_sources(info.sources)

    # Maximum reward
    evaluator.add_custom_node(
        result=non_empty(info.maximum_reward),
        id="Maximum_Reward_Provided",
        desc="Maximum reward amount provided",
        parent=node,
        critical=True
    )
    max_leaf = evaluator.add_leaf(
        id="Maximum_Reward",
        desc="Provide the maximum dollar amount of the reward that was offered",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The maximum reward amount offered was {info.maximum_reward}.",
        node=max_leaf,
        sources=srcs,
        additional_instruction="Confirm reward amount from State Department rewards pages or DOJ announcements. Accept currency formatting variants."
    )

    # Reward increase date (optional under same non-critical group but gated)
    evaluator.add_custom_node(
        result=non_empty(info.reward_increase_date),
        id="Reward_Increase_Date_Provided",
        desc="Reward increase date provided",
        parent=node,
        critical=True
    )
    inc_leaf = evaluator.add_leaf(
        id="Reward_Increase_Date",
        desc="Specify when the reward was increased to its maximum amount",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The reward was increased to its maximum amount in {info.reward_increase_date}.",
        node=inc_leaf,
        sources=srcs,
        additional_instruction="Verify month/year or date when reward increase occurred from official sources."
    )


async def build_initial_charges_date(evaluator: Evaluator, parent_node, data: CaseExtraction) -> None:
    node = evaluator.add_parallel(
        id="Initial_Charges_Date",
        desc="Identify when criminal charges were first filed against the defendant",
        parent=parent_node,
        critical=False
    )
    info = data.initial_charges or InitialChargesInfo()
    srcs = safe_sources(info.sources)

    evaluator.add_custom_node(
        result=non_empty(info.first_indictment_date),
        id="First_Indictment_Date_Provided",
        desc="First indictment date provided",
        parent=node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="First_Indictment_Date",
        desc="Provide the month and year when the defendant was first indicted in the Southern District of New York",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Criminal charges were first filed in the Southern District of New York in {info.first_indictment_date}.",
        node=leaf,
        sources=srcs,
        additional_instruction="Verify the month and year from DOJ press releases or SDNY documents."
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
    Evaluate an answer for the Maduro legal case research task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: parallel aggregation
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_case_info(),
        template_class=CaseExtraction,
        extraction_name="case_extraction"
    )

    # Build verification tree according to rubric (with necessary criticality adjustments)
    # Root is non-critical to allow partial credit across the many sub-items.
    # Critical groups are enforced within their own nodes.

    await build_defendant_identification(evaluator, root, extracted)
    await build_co_defendant(evaluator, root, extracted)
    await build_court_jurisdiction(evaluator, root, extracted)
    await build_presiding_judge(evaluator, root, extracted)
    await build_criminal_charges(evaluator, root, extracted)
    await build_capture_date(evaluator, root, extracted)
    await build_arraignment_info(evaluator, root, extracted)
    await build_next_hearing(evaluator, root, extracted)
    await build_detention_info(evaluator, root, extracted)
    await build_extradition_treaty(evaluator, root, extracted)
    await build_legal_precedents(evaluator, root, extracted)
    await build_olc_memo(evaluator, root, extracted)
    await build_attorney_general(evaluator, root, extracted)
    await build_reward_amount(evaluator, root, extracted)
    await build_initial_charges_date(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()