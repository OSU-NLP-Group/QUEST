import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mesa_county_election_cases_2024_2025"
TASK_DESCRIPTION = """
Research and provide detailed information about two election-related criminal cases from Mesa County, Colorado, that resulted in sentencing in 2024-2025:

Case 1 - Tina Peters: The former Mesa County Clerk and Recorder who was sentenced in October 2024 for her role in a 2021 election systems security breach. Provide the following information:
- The defendant's full name and the official position she held
- The date of her sentencing and the name of the judge who imposed the sentence
- The length of her prison sentence
- The specific felony charges she was convicted of, including charges related to attempting to influence public servants and conspiracy to commit criminal impersonation
- The misdemeanor charges she was convicted of
- The financial cost to Mesa County resulting from her actions

Case 2 - Postal Worker Ballot Fraud: A former Mesa County postal worker who was sentenced in June 2025 for stealing ballots during the 2024 General Election. Provide the following information:
- The defendant's full name, age at sentencing, and occupation
- The date of the sentencing and the length of the sentence imposed
- The specific felony charges to which the defendant pleaded guilty (identity theft and forgery)
- The number of ballots that were stolen in the scheme
- The name of the accomplice who participated in the ballot fraud scheme

For each piece of information, include a reference URL from a reliable source (government website, established news organization, or official court document) that supports your answer.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PetersInfo(BaseModel):
    full_name: Optional[str] = None
    position: Optional[str] = None
    info_urls: List[str] = Field(default_factory=list)


class PetersSentencing(BaseModel):
    sentence_length: Optional[str] = None
    sentencing_date: Optional[str] = None
    judge_name: Optional[str] = None
    sentencing_urls: List[str] = Field(default_factory=list)


class PetersCharges(BaseModel):
    felony_charges: List[str] = Field(default_factory=list)
    misdemeanor_charges: List[str] = Field(default_factory=list)
    charges_urls: List[str] = Field(default_factory=list)


class PetersImpact(BaseModel):
    financial_cost: Optional[str] = None
    impact_urls: List[str] = Field(default_factory=list)


class StuartProfile(BaseModel):
    full_name: Optional[str] = None
    age: Optional[str] = None
    occupation: Optional[str] = None
    profile_urls: List[str] = Field(default_factory=list)


class StuartSentencing(BaseModel):
    sentence_length: Optional[str] = None
    sentencing_date: Optional[str] = None
    sentencing_urls: List[str] = Field(default_factory=list)


class StuartCharges(BaseModel):
    charges_list: List[str] = Field(default_factory=list)
    charges_urls: List[str] = Field(default_factory=list)


class StuartDetails(BaseModel):
    ballots_stolen_count: Optional[str] = None
    accomplice_name: Optional[str] = None
    details_urls: List[str] = Field(default_factory=list)


class CasesExtraction(BaseModel):
    peters_info: Optional[PetersInfo] = None
    peters_sentencing: Optional[PetersSentencing] = None
    peters_charges: Optional[PetersCharges] = None
    peters_impact: Optional[PetersImpact] = None

    stuart_profile: Optional[StuartProfile] = None
    stuart_sentencing: Optional[StuartSentencing] = None
    stuart_charges: Optional[StuartCharges] = None
    stuart_details: Optional[StuartDetails] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cases() -> str:
    return """
Extract the requested structured facts for the two Mesa County, Colorado election-related criminal cases from the answer text. Extract ONLY what is explicitly stated in the answer. If some field is not provided in the answer, set it to null (for strings) or [] (for lists). For each group of facts, also extract the list of reference URLs that the answer provided for that group. Do not invent URLs; include only explicit URLs mentioned in the answer text (markdown links allowed).

Return a single JSON object with the following structure:

{
  "peters_info": {
    "full_name": string or null,
    "position": string or null,
    "info_urls": [url, ...]
  },
  "peters_sentencing": {
    "sentence_length": string or null,
    "sentencing_date": string or null,
    "judge_name": string or null,
    "sentencing_urls": [url, ...]
  },
  "peters_charges": {
    "felony_charges": [string, ...],
    "misdemeanor_charges": [string, ...],
    "charges_urls": [url, ...]
  },
  "peters_impact": {
    "financial_cost": string or null,
    "impact_urls": [url, ...]
  },

  "stuart_profile": {
    "full_name": string or null,
    "age": string or null,
    "occupation": string or null,
    "profile_urls": [url, ...]
  },
  "stuart_sentencing": {
    "sentence_length": string or null,
    "sentencing_date": string or null,
    "sentencing_urls": [url, ...]
  },
  "stuart_charges": {
    "charges_list": [string, ...],
    "charges_urls": [url, ...]
  },
  "stuart_details": {
    "ballots_stolen_count": string or null,
    "accomplice_name": string or null,
    "details_urls": [url, ...]
  }
}

Guidance and constraints:
- Keep values as strings (including dates and amounts) to preserve original formatting (e.g., "$120,000", "October 23, 2024", "4 years", "18 ballots", etc.).
- For "felony_charges" and "charges_list", extract each charge as it appears (e.g., "attempting to influence a public servant", "conspiracy to commit criminal impersonation", "identity theft", "forgery").
- URLs must be exact and valid full URLs. If any URL is missing or not provided, return an empty list for that URL field.
- Do not infer values; only extract what the answer claims and cites.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst if lst else []


def _fmt_list(items: List[str]) -> str:
    return ", ".join(items) if items else ""


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_peters_sections(evaluator: Evaluator, parent_node, data: CasesExtraction) -> None:
    # ---------------- Peters: Defendant Information ----------------
    info = data.peters_info or PetersInfo()
    p_info_node = evaluator.add_parallel(
        id="Peters_Defendant_Information",
        desc="Provide basic identifying information about the defendant in the Tina Peters case.",
        parent=parent_node,
        critical=True
    )

    # URL presence node (critical)
    evaluator.add_custom_node(
        result=bool(_safe_list(info.info_urls)),
        id="Peters_Info_URL",
        desc="Provide a reference URL supporting the defendant information.",
        parent=p_info_node,
        critical=True
    )

    # Full name
    leaf = evaluator.add_leaf(
        id="Peters_Full_Name",
        desc="Provide the defendant's full name.",
        parent=p_info_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The defendant's full name is '{info.full_name}'.",
        node=leaf,
        sources=_safe_list(info.info_urls),
        additional_instruction="Verify the full legal name of the defendant on the provided page(s). Allow minor variations like middle initials."
    )

    # Position held
    leaf = evaluator.add_leaf(
        id="Peters_Position_Held",
        desc="Identify the official position the defendant held at the time of the criminal activity.",
        parent=p_info_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The defendant held the official position of '{info.position}' in Mesa County, Colorado.",
        node=leaf,
        sources=_safe_list(info.info_urls),
        additional_instruction="Confirm the defendant's official role or position title on the source page(s)."
    )

    # ---------------- Peters: Sentencing Details ----------------
    sent = data.peters_sentencing or PetersSentencing()
    p_sent_node = evaluator.add_parallel(
        id="Peters_Sentencing_Details",
        desc="Provide details about the sentencing in the Tina Peters case.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(sent.sentencing_urls)),
        id="Peters_Sentencing_URL",
        desc="Provide a reference URL supporting the sentencing details.",
        parent=p_sent_node,
        critical=True
    )

    # Sentence length
    leaf = evaluator.add_leaf(
        id="Peters_Sentence_Length",
        desc="Specify the length of the prison sentence imposed.",
        parent=p_sent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The length of the prison sentence was '{sent.sentence_length}'.",
        node=leaf,
        sources=_safe_list(sent.sentencing_urls),
        additional_instruction="Locate the sentence length (e.g., years or months) in the sentencing section of the source."
    )

    # Sentencing date
    leaf = evaluator.add_leaf(
        id="Peters_Sentencing_Date",
        desc="Provide the date when the sentencing occurred.",
        parent=p_sent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The sentencing occurred on '{sent.sentencing_date}'.",
        node=leaf,
        sources=_safe_list(sent.sentencing_urls),
        additional_instruction="Verify the exact sentencing date; minor formatting variations (e.g., 'Oct.' vs 'October') are acceptable."
    )

    # Judge name
    leaf = evaluator.add_leaf(
        id="Peters_Judge_Name",
        desc="Identify the judge who imposed the sentence.",
        parent=p_sent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The judge who imposed the sentence was '{sent.judge_name}'.",
        node=leaf,
        sources=_safe_list(sent.sentencing_urls),
        additional_instruction="Confirm the presiding judge's name associated with the sentencing in this case."
    )

    # ---------------- Peters: Criminal Charges ----------------
    ch = data.peters_charges or PetersCharges()
    p_ch_node = evaluator.add_parallel(
        id="Peters_Criminal_Charges",
        desc="Detail the criminal charges for which Tina Peters was convicted.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(ch.charges_urls)),
        id="Peters_Charges_URL",
        desc="Provide a reference URL supporting the conviction charges.",
        parent=p_ch_node,
        critical=True
    )

    # Felony charges (must include specified ones)
    leaf = evaluator.add_leaf(
        id="Peters_Felony_Charges",
        desc="List the specific felony charges for which the defendant was convicted, including attempting to influence a public servant and conspiracy to commit criminal impersonation.",
        parent=p_ch_node,
        critical=True
    )
    felony_list_text = _fmt_list(ch.felony_charges)
    claim_text = (
        "The defendant was convicted of the following felony charges: "
        f"{felony_list_text}. These include 'attempting to influence a public servant' "
        "and 'conspiracy to commit criminal impersonation'."
    )
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=_safe_list(ch.charges_urls),
        additional_instruction="Check the conviction section for the exact felony charge names; allow minor wording variants that clearly refer to the same offenses."
    )

    # Misdemeanor charges
    leaf = evaluator.add_leaf(
        id="Peters_Misdemeanor_Charges",
        desc="List the misdemeanor charges for which the defendant was convicted.",
        parent=p_ch_node,
        critical=True
    )
    misdemeanor_text = _fmt_list(ch.misdemeanor_charges)
    await evaluator.verify(
        claim=f"The defendant was convicted of the following misdemeanor charges: {misdemeanor_text}.",
        node=leaf,
        sources=_safe_list(ch.charges_urls),
        additional_instruction="Verify the misdemeanor convictions exactly as listed on the source page(s)."
    )

    # ---------------- Peters: Financial Impact ----------------
    imp = data.peters_impact or PetersImpact()
    p_imp_node = evaluator.add_parallel(
        id="Peters_Financial_Impact",
        desc="Report the financial cost or impact to Mesa County from the Tina Peters case.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(imp.impact_urls)),
        id="Peters_Impact_URL",
        desc="Provide a reference URL supporting the financial impact information.",
        parent=p_imp_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Peters_Cost_Amount",
        desc="Specify the dollar amount of financial cost to Mesa County.",
        parent=p_imp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The financial cost to Mesa County resulting from the defendant's actions was '{imp.financial_cost}'.",
        node=leaf,
        sources=_safe_list(imp.impact_urls),
        additional_instruction="Verify the reported total cost or financial impact figure attributed to Mesa County; allow formatting differences like commas or currency symbols."
    )


async def verify_stuart_sections(evaluator: Evaluator, parent_node, data: CasesExtraction) -> None:
    # ---------------- Stuart: Defendant Profile ----------------
    profile = data.stuart_profile or StuartProfile()
    s_prof_node = evaluator.add_parallel(
        id="Stuart_Defendant_Profile",
        desc="Provide identifying information about the defendant in the postal worker ballot fraud case.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(profile.profile_urls)),
        id="Stuart_Profile_URL",
        desc="Provide a reference URL supporting the defendant profile information.",
        parent=s_prof_node,
        critical=True
    )

    # Full name
    leaf = evaluator.add_leaf(
        id="Stuart_Full_Name",
        desc="Provide the defendant's full name.",
        parent=s_prof_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The defendant's full name is '{profile.full_name}'.",
        node=leaf,
        sources=_safe_list(profile.profile_urls),
        additional_instruction="Confirm the defendant's name on the source page; minor variations such as middle initials are acceptable."
    )

    # Age
    leaf = evaluator.add_leaf(
        id="Stuart_Age",
        desc="Specify the defendant's age at the time of sentencing.",
        parent=s_prof_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The defendant was '{profile.age}' years old at the time of sentencing.",
        node=leaf,
        sources=_safe_list(profile.profile_urls),
        additional_instruction="Locate the age at sentencing explicitly stated; accept minor phrasing variations."
    )

    # Occupation
    leaf = evaluator.add_leaf(
        id="Stuart_Occupation",
        desc="Identify the defendant's occupation at the time of the criminal activity.",
        parent=s_prof_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The defendant's occupation at the time was '{profile.occupation}'.",
        node=leaf,
        sources=_safe_list(profile.profile_urls),
        additional_instruction="Confirm that the defendant was a postal worker (or equivalent) through the source page(s)."
    )

    # ---------------- Stuart: Sentencing Details ----------------
    sent = data.stuart_sentencing or StuartSentencing()
    s_sent_node = evaluator.add_parallel(
        id="Stuart_Sentencing_Details",
        desc="Provide details about the sentencing in the Vicki Stuart case.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(sent.sentencing_urls)),
        id="Stuart_Sentencing_URL",
        desc="Provide a reference URL supporting the sentencing details.",
        parent=s_sent_node,
        critical=True
    )

    # Sentence length
    leaf = evaluator.add_leaf(
        id="Stuart_Sentence_Length",
        desc="Specify the length of the sentence imposed.",
        parent=s_sent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The length of the sentence imposed was '{sent.sentence_length}'.",
        node=leaf,
        sources=_safe_list(sent.sentencing_urls),
        additional_instruction="Verify the sentencing length in terms of months or years as stated on the page."
    )

    # Sentencing date
    leaf = evaluator.add_leaf(
        id="Stuart_Sentencing_Date",
        desc="Provide the date when the sentencing occurred.",
        parent=s_sent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The sentencing occurred on '{sent.sentencing_date}'.",
        node=leaf,
        sources=_safe_list(sent.sentencing_urls),
        additional_instruction="Confirm the specific sentencing date; allow standard date formatting variations."
    )

    # ---------------- Stuart: Criminal Charges ----------------
    ch = data.stuart_charges or StuartCharges()
    s_ch_node = evaluator.add_parallel(
        id="Stuart_Criminal_Charges",
        desc="Detail the criminal charges to which Vicki Stuart pleaded guilty.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(ch.charges_urls)),
        id="Stuart_Charges_URL",
        desc="Provide a reference URL supporting the guilty plea charges.",
        parent=s_ch_node,
        critical=True
    )

    # Identity theft charge
    leaf = evaluator.add_leaf(
        id="Stuart_Identity_Theft_Charge",
        desc="Confirm the guilty plea to identity theft charge.",
        parent=s_ch_node,
        critical=True
    )
    await evaluator.verify(
        claim="The defendant pleaded guilty to identity theft.",
        node=leaf,
        sources=_safe_list(ch.charges_urls),
        additional_instruction="Verify that the plea explicitly includes 'identity theft' (felony), allowing minor wording variants."
    )

    # Forgery charge
    leaf = evaluator.add_leaf(
        id="Stuart_Forgery_Charge",
        desc="Confirm the guilty plea to forgery charge.",
        parent=s_ch_node,
        critical=True
    )
    await evaluator.verify(
        claim="The defendant pleaded guilty to forgery.",
        node=leaf,
        sources=_safe_list(ch.charges_urls),
        additional_instruction="Verify that the plea explicitly includes 'forgery' (felony), allowing minor wording variants."
    )

    # ---------------- Stuart: Case Specifics ----------------
    det = data.stuart_details or StuartDetails()
    s_det_node = evaluator.add_parallel(
        id="Stuart_Case_Specifics",
        desc="Provide specific details about the Vicki Stuart ballot fraud case.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(det.details_urls)),
        id="Stuart_Details_URL",
        desc="Provide a reference URL supporting the case-specific details.",
        parent=s_det_node,
        critical=True
    )

    # Ballots stolen count
    leaf = evaluator.add_leaf(
        id="Stuart_Ballots_Stolen_Count",
        desc="Specify the number of ballots that were stolen.",
        parent=s_det_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The number of ballots stolen was '{det.ballots_stolen_count}'.",
        node=leaf,
        sources=_safe_list(det.details_urls),
        additional_instruction="Verify the exact count of ballots stolen; number must match the source (allow comma formatting differences only)."
    )

    # Accomplice name
    leaf = evaluator.add_leaf(
        id="Stuart_Accomplice_Name",
        desc="Identify the name of the accomplice involved in the scheme.",
        parent=s_det_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The accomplice involved in the ballot fraud scheme was '{det.accomplice_name}'.",
        node=leaf,
        sources=_safe_list(det.details_urls),
        additional_instruction="Verify the accomplice's full name as stated on the source page(s); allow minor name formatting variations."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Mesa County election-related criminal cases task.
    """
    # Initialize evaluator with a parallel root
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

    # Top-level critical node matching rubric
    top_node = evaluator.add_parallel(
        id="Mesa_County_Election_Cases_Research",
        desc="Research and provide detailed information about two specific election-related criminal cases from Mesa County, Colorado: the Tina Peters case and the Vicki Stuart postal worker ballot fraud case.",
        parent=root,
        critical=True
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_cases(),
        template_class=CasesExtraction,
        extraction_name="cases_extraction"
    )

    # Build verification subtrees
    await verify_peters_sections(evaluator, top_node, extracted)
    await verify_stuart_sections(evaluator, top_node, extracted)

    # Return structured result
    return evaluator.get_summary()