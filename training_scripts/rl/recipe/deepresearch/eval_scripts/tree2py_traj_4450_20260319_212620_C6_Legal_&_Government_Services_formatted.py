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
TASK_ID = "attorney_identification_assange_tankleff_krautz_maduro"
TASK_DESCRIPTION = """
Identify the full name (first, middle initial if applicable, and last name) of the attorney who satisfies all of the following criteria: The attorney must be a partner at a New York-based law firm and must have represented Julian Assange, the WikiLeaks founder, for more than a decade, ultimately securing a plea deal for Assange in 2024. This attorney must also have won a complete acquittal for Michael Krautz, an Enron accountant who faced criminal fraud charges, and must have helped overturn the wrongful convictions of Martin Tankleff, who had been imprisoned for killing his parents. As of January 2025, this attorney must currently represent Nicolás Maduro. Additionally, the attorney must be a Fellow of the American College of Trial Lawyers and must have previously served as president of the National Association of Criminal Defense Lawyers. The attorney's educational background must include a law degree from Georgetown University School of Law and an undergraduate degree from Indiana University. The attorney's legal career must span more than 30 years, and they must be affiliated with Harris St. Laurent & Wechsler LLP. Provide the attorney's full name along with reference URLs that verify each of the specified criteria, including law firm affiliation, client representations, professional credentials, and educational background.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AttorneyExtraction(BaseModel):
    # Core identification
    full_name: Optional[str] = None
    firm_name: Optional[str] = None  # If the answer states a firm name explicitly

    # Law firm + partner role evidence
    firm_affiliation_urls: List[str] = Field(default_factory=list)  # HSLW affiliation sources
    partner_ny_urls: List[str] = Field(default_factory=list)        # Evidence for "partner at a NY-based firm"

    # Julian Assange representation evidence
    assange_over_decade_urls: List[str] = Field(default_factory=list)
    assange_plea_2024_urls: List[str] = Field(default_factory=list)

    # Michael Krautz acquittal evidence
    krautz_acquittal_urls: List[str] = Field(default_factory=list)

    # Martin Tankleff wrongful convictions overturned evidence
    tankleff_overturn_urls: List[str] = Field(default_factory=list)

    # Nicolás Maduro current representation evidence
    maduro_current_urls: List[str] = Field(default_factory=list)

    # Professional credentials evidence
    actl_fellow_urls: List[str] = Field(default_factory=list)
    nacdl_president_urls: List[str] = Field(default_factory=list)

    # Education evidence
    georgetown_law_urls: List[str] = Field(default_factory=list)
    indiana_undergrad_urls: List[str] = Field(default_factory=list)

    # Career duration > 30 years evidence
    career_30_years_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_attorney_info() -> str:
    return """
    From the answer, extract the single attorney's full name and categorize all cited URLs that support each required criterion.
    
    Return fields:
    - full_name: The attorney's full name exactly as written in the answer (First [Middle initial if used] Last).
    - firm_name: The name of the law firm the answer associates the attorney with (if mentioned).
    - firm_affiliation_urls: All URLs that support affiliation with Harris St. Laurent & Wechsler LLP (HSLW/Harris St. Laurent & Wechsler).
    - partner_ny_urls: All URLs that support that the attorney is a partner at a New York-based law firm (e.g., the firm is based in NYC or the attorney is a partner in the New York office).
    - assange_over_decade_urls: URLs supporting that the attorney represented Julian Assange for more than a decade.
    - assange_plea_2024_urls: URLs supporting that the attorney secured (or helped secure) a plea deal for Assange in 2024.
    - krautz_acquittal_urls: URLs supporting that the attorney won a complete acquittal for Michael Krautz (an Enron accountant).
    - tankleff_overturn_urls: URLs supporting that the attorney helped overturn the wrongful convictions of Martin Tankleff.
    - maduro_current_urls: URLs supporting that, as of January 2025, the attorney currently represents Nicolás Maduro.
    - actl_fellow_urls: URLs supporting that the attorney is a Fellow of the American College of Trial Lawyers (ACTL).
    - nacdl_president_urls: URLs supporting that the attorney previously served as president of the National Association of Criminal Defense Lawyers (NACDL).
    - georgetown_law_urls: URLs supporting that the attorney earned a law degree (J.D.) from Georgetown University Law Center (also called Georgetown University School of Law).
    - indiana_undergrad_urls: URLs supporting that the attorney earned an undergraduate degree from Indiana University.
    - career_30_years_urls: URLs supporting that the attorney's legal career spans more than 30 years (e.g., "over 30 years," "more than three decades," or dates implying >30 years).

    Rules:
    - Only include URLs explicitly present in the answer (plain or markdown). Do not invent URLs.
    - If a category has no URLs in the answer, return an empty list for that field.
    - Keep duplicates if they appear in multiple categories; otherwise, deduplicate within the same category.
    - Preserve full URLs including http/https.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_valid_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")) for u in urls)


def safe_name(name: Optional[str]) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else "the attorney identified in the answer"


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_name_output_branch(
    evaluator: Evaluator,
    parent_node,
    extraction: AttorneyExtraction,
):
    name_branch = evaluator.add_parallel(
        id="Answer_Name_Output",
        desc="Provide the attorney's full name in the requested format.",
        parent=parent_node,
        critical=True,
    )

    full_name_leaf = evaluator.add_leaf(
        id="Full_Name_In_Requested_Format",
        desc="Response includes the attorney's full name as first name + (middle initial if applicable) + last name.",
        parent=name_branch,
        critical=True,
    )

    # Verify from the answer text directly (format/presence check)
    claimed_name = extraction.full_name or ""
    claim = (
        f"The answer provides the attorney's full name as '{claimed_name}', formatted as First name + "
        f"(optional middle initial) + Last name."
    )
    await evaluator.verify(
        claim=claim,
        node=full_name_leaf,
        additional_instruction=(
            "Judge strictly from the answer text whether a full personal name is presented. "
            "A valid full name must include at least a first and last name; a middle initial is optional. "
            "Minor formatting differences are acceptable."
        ),
    )
    return full_name_leaf


async def build_law_firm_affiliation_branch(
    evaluator: Evaluator,
    parent_node,
    extraction: AttorneyExtraction,
    name_leaf,
):
    law_branch = evaluator.add_parallel(
        id="Law_Firm_Affiliation",
        desc="Verify law firm affiliation and partner role requirements, with citations.",
        parent=parent_node,
        critical=True,
    )

    # Affiliation with Harris St. Laurent & Wechsler LLP (HSLW)
    aff_leaf = evaluator.add_leaf(
        id="Affiliated_With_Harris_St_Laurent_Wechsler",
        desc="Attorney is affiliated with Harris St. Laurent & Wechsler LLP.",
        parent=law_branch,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The page(s) show that {safe_name(extraction.full_name)} is affiliated with "
            f"Harris St. Laurent & Wechsler LLP (also referred to as 'Harris St. Laurent & Wechsler' or 'HSLW')."
        ),
        node=aff_leaf,
        sources=extraction.firm_affiliation_urls,
        additional_instruction=(
            "Look for explicit statements (e.g., biography, firm page) indicating the attorney is with "
            "Harris St. Laurent & Wechsler LLP. Accept common variants like 'Harris St. Laurent & Wechsler' or 'HSLW'."
        ),
        extra_prerequisites=[name_leaf],
    )

    url_aff_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.firm_affiliation_urls),
        id="URL_HSLW_Affiliation",
        desc="Provide at least one reference URL that supports the claim of affiliation with Harris St. Laurent & Wechsler LLP.",
        parent=law_branch,
        critical=True,
    )

    # Partner at a New York-based law firm
    partner_leaf = evaluator.add_leaf(
        id="Partner_At_NY_Based_Law_Firm",
        desc="Attorney is a partner at a New York-based law firm.",
        parent=law_branch,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The page(s) show that {safe_name(extraction.full_name)} is a partner at a New York-based law firm."
        ),
        node=partner_leaf,
        sources=extraction.partner_ny_urls,
        additional_instruction=(
            "Confirm both: (1) the attorney holds the title 'Partner' (or equivalent) and "
            "(2) the firm is New York-based (e.g., headquartered in NYC, primary office in New York, "
            "or the attorney is a Partner in the New York office)."
        ),
        extra_prerequisites=[name_leaf],
    )

    url_partner_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.partner_ny_urls),
        id="URL_Partner_NY_Based",
        desc="Provide at least one reference URL that supports the claim that the attorney is a partner at a New York-based law firm.",
        parent=law_branch,
        critical=True,
    )


async def build_assange_branch(
    evaluator: Evaluator,
    parent_node,
    extraction: AttorneyExtraction,
    name_leaf,
):
    assange_branch = evaluator.add_parallel(
        id="Julian_Assange_Representation",
        desc="Verify Julian Assange representation requirements, with citations.",
        parent=parent_node,
        critical=True,
    )

    decade_leaf = evaluator.add_leaf(
        id="Represented_Assange_Over_Decade",
        desc="Attorney represented Julian Assange for more than a decade.",
        parent=assange_branch,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page(s) show that {safe_name(extraction.full_name)} represented Julian Assange for more than a decade.",
        node=decade_leaf,
        sources=extraction.assange_over_decade_urls,
        additional_instruction=(
            "Evidence may include timelines or language like 'more than a decade', 'over ten years', or "
            "start year vs. 2024/2025 implying >= 10 years of representation."
        ),
        extra_prerequisites=[name_leaf],
    )

    url_decade_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.assange_over_decade_urls),
        id="URL_Assange_Over_Decade",
        desc="Provide at least one reference URL that supports the claim that the attorney represented Julian Assange for more than a decade.",
        parent=assange_branch,
        critical=True,
    )

    plea_leaf = evaluator.add_leaf(
        id="Secured_Assange_Plea_Deal_2024",
        desc="Attorney ultimately secured a plea deal for Assange in 2024.",
        parent=assange_branch,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The page(s) show that {safe_name(extraction.full_name)} secured or helped secure a plea deal "
            f"for Julian Assange in 2024."
        ),
        node=plea_leaf,
        sources=extraction.assange_plea_2024_urls,
        additional_instruction=(
            "Look for explicit mentions of a plea deal in 2024 and that the attorney was counsel who secured or "
            "was credited with securing the plea. Reliable news reports, court filings, or official statements suffice."
        ),
        extra_prerequisites=[name_leaf],
    )

    url_plea_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.assange_plea_2024_urls),
        id="URL_Assange_Plea_Deal_2024",
        desc="Provide at least one reference URL that supports the claim that the attorney secured a plea deal for Assange in 2024.",
        parent=assange_branch,
        critical=True,
    )


async def build_krautz_branch(
    evaluator: Evaluator,
    parent_node,
    extraction: AttorneyExtraction,
    name_leaf,
):
    krautz_branch = evaluator.add_parallel(
        id="Michael_Krautz_Acquittal",
        desc="Verify the Michael Krautz acquittal requirement, with citations.",
        parent=parent_node,
        critical=True,
    )

    acquit_leaf = evaluator.add_leaf(
        id="Complete_Acquittal_For_Michael_Krautz",
        desc="Attorney won a complete acquittal for Michael Krautz (Enron accountant facing criminal fraud charges).",
        parent=krautz_branch,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The page(s) show that {safe_name(extraction.full_name)} won a complete acquittal for "
            f"Michael Krautz, an Enron accountant who faced criminal fraud charges."
        ),
        node=acquit_leaf,
        sources=extraction.krautz_acquittal_urls,
        additional_instruction=(
            "Confirm the result was a complete acquittal for Michael Krautz in an Enron-related prosecution."
        ),
        extra_prerequisites=[name_leaf],
    )

    url_acquit_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.krautz_acquittal_urls),
        id="URL_Krautz_Acquittal",
        desc="Provide at least one reference URL that supports the claim of a complete acquittal for Michael Krautz.",
        parent=krautz_branch,
        critical=True,
    )


async def build_tankleff_branch(
    evaluator: Evaluator,
    parent_node,
    extraction: AttorneyExtraction,
    name_leaf,
):
    tankleff_branch = evaluator.add_parallel(
        id="Martin_Tankleff_Convictions_Overturned",
        desc="Verify the Martin Tankleff wrongful-conviction reversal requirement, with citations.",
        parent=parent_node,
        critical=True,
    )

    overturn_leaf = evaluator.add_leaf(
        id="Helped_Overturn_Tankleff_Wrongful_Convictions",
        desc="Attorney helped overturn the wrongful convictions of Martin Tankleff.",
        parent=tankleff_branch,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The page(s) show that {safe_name(extraction.full_name)} helped overturn Martin Tankleff's "
            f"wrongful convictions."
        ),
        node=overturn_leaf,
        sources=extraction.tankleff_overturn_urls,
        additional_instruction=(
            "Accept language indicating obtaining reversal/vacatur of convictions, securing a new trial that led to dismissal, "
            "or similar outcomes demonstrating the wrongful convictions were overturned with the attorney's involvement."
        ),
        extra_prerequisites=[name_leaf],
    )

    url_overturn_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.tankleff_overturn_urls),
        id="URL_Tankleff_Overturning",
        desc="Provide at least one reference URL that supports the claim that the attorney helped overturn Martin Tankleff's wrongful convictions.",
        parent=tankleff_branch,
        critical=True,
    )


async def build_maduro_branch(
    evaluator: Evaluator,
    parent_node,
    extraction: AttorneyExtraction,
    name_leaf,
):
    maduro_branch = evaluator.add_parallel(
        id="Nicolas_Maduro_Current_Representation",
        desc="Verify current representation of Nicolás Maduro as of January 2025, with citations.",
        parent=parent_node,
        critical=True,
    )

    current_leaf = evaluator.add_leaf(
        id="Represents_Maduro_As_Of_Jan_2025",
        desc="Attorney currently represents Nicolás Maduro as of January 2025.",
        parent=maduro_branch,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The page(s) show that {safe_name(extraction.full_name)} represents Nicolás Maduro as of January 2025."
        ),
        node=current_leaf,
        sources=extraction.maduro_current_urls,
        additional_instruction=(
            "Prefer sources dated near late 2024 or 2025 indicating ongoing or current representation. "
            "Wording like 'represents' or 'currently represents' suffices."
        ),
        extra_prerequisites=[name_leaf],
    )

    url_current_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.maduro_current_urls),
        id="URL_Maduro_Current_Representation",
        desc="Provide at least one reference URL that supports the claim that the attorney represents Nicolás Maduro as of January 2025.",
        parent=maduro_branch,
        critical=True,
    )


async def build_credentials_branch(
    evaluator: Evaluator,
    parent_node,
    extraction: AttorneyExtraction,
    name_leaf,
):
    cred_branch = evaluator.add_parallel(
        id="Professional_Credentials",
        desc="Verify required professional credentials and leadership role, with citations.",
        parent=parent_node,
        critical=True,
    )

    actl_leaf = evaluator.add_leaf(
        id="ACTL_Fellow",
        desc="Attorney is a Fellow of the American College of Trial Lawyers.",
        parent=cred_branch,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page(s) show that {safe_name(extraction.full_name)} is a Fellow of the American College of Trial Lawyers.",
        node=actl_leaf,
        sources=extraction.actl_fellow_urls,
        additional_instruction="Prefer ACTL's official directory or credible firm bios/press that clearly state ACTL Fellowship.",
        extra_prerequisites=[name_leaf],
    )

    url_actl_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.actl_fellow_urls),
        id="URL_ACTL_Fellow",
        desc="Provide at least one reference URL that supports the claim of ACTL fellowship.",
        parent=cred_branch,
        critical=True,
    )

    nacdl_leaf = evaluator.add_leaf(
        id="NACDL_President",
        desc="Attorney previously served as president of the National Association of Criminal Defense Lawyers.",
        parent=cred_branch,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page(s) show that {safe_name(extraction.full_name)} previously served as President of NACDL.",
        node=nacdl_leaf,
        sources=extraction.nacdl_president_urls,
        additional_instruction="Accept 'past president' or 'served as president' language; prefer NACDL official pages or credible bios.",
        extra_prerequisites=[name_leaf],
    )

    url_nacdl_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.nacdl_president_urls),
        id="URL_NACDL_President",
        desc="Provide at least one reference URL that supports the claim that the attorney served as NACDL president.",
        parent=cred_branch,
        critical=True,
    )


async def build_education_branch(
    evaluator: Evaluator,
    parent_node,
    extraction: AttorneyExtraction,
    name_leaf,
):
    edu_branch = evaluator.add_parallel(
        id="Educational_Background",
        desc="Verify required educational credentials, with citations.",
        parent=parent_node,
        critical=True,
    )

    gt_leaf = evaluator.add_leaf(
        id="Georgetown_Law_Degree",
        desc="Attorney graduated from Georgetown University School of Law.",
        parent=edu_branch,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The page(s) show that {safe_name(extraction.full_name)} earned a law degree (J.D.) from "
            f"Georgetown University Law Center (also known as Georgetown University School of Law)."
        ),
        node=gt_leaf,
        sources=extraction.georgetown_law_urls,
        additional_instruction="Accept 'J.D.' or 'law degree' from 'Georgetown University Law Center' or equivalent naming.",
        extra_prerequisites=[name_leaf],
    )

    url_gt_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.georgetown_law_urls),
        id="URL_Georgetown_Law_Degree",
        desc="Provide at least one reference URL that supports the claim of graduation from Georgetown University School of Law.",
        parent=edu_branch,
        critical=True,
    )

    iu_leaf = evaluator.add_leaf(
        id="Indiana_Undergrad_Degree",
        desc="Attorney graduated from Indiana University (undergraduate).",
        parent=edu_branch,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page(s) show that {safe_name(extraction.full_name)} earned an undergraduate degree from Indiana University.",
        node=iu_leaf,
        sources=extraction.indiana_undergrad_urls,
        additional_instruction="Accept any bachelor's degree (B.A., B.S., etc.) from Indiana University (any campus).",
        extra_prerequisites=[name_leaf],
    )

    url_iu_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.indiana_undergrad_urls),
        id="URL_Indiana_Undergrad_Degree",
        desc="Provide at least one reference URL that supports the claim of undergraduate graduation from Indiana University.",
        parent=edu_branch,
        critical=True,
    )


async def build_career_branch(
    evaluator: Evaluator,
    parent_node,
    extraction: AttorneyExtraction,
    name_leaf,
):
    career_branch = evaluator.add_parallel(
        id="Career_Duration",
        desc="Verify legal career duration requirement, with citations.",
        parent=parent_node,
        critical=True,
    )

    years_leaf = evaluator.add_leaf(
        id="Career_Over_30_Years",
        desc="Attorney's legal career spans more than 30 years.",
        parent=career_branch,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page(s) show that {safe_name(extraction.full_name)} has a legal career spanning more than 30 years.",
        node=years_leaf,
        sources=extraction.career_30_years_urls,
        additional_instruction=(
            "Accept explicit phrases such as 'over 30 years' or 'more than three decades', or dates implying >30 years of practice."
        ),
        extra_prerequisites=[name_leaf],
    )

    url_years_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.career_30_years_urls),
        id="URL_Career_Over_30_Years",
        desc="Provide at least one reference URL that supports the claim that the attorney's legal career spans more than 30 years.",
        parent=career_branch,
        critical=True,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the attorney identification task using the Mind2Web2 framework.
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
        default_model=model,
    )

    # Main critical node mirroring rubric root
    main_node = evaluator.add_parallel(
        id="Attorney_Identification",
        desc="Identify the attorney who satisfies all specified criteria and provide supporting reference URLs for each criterion.",
        parent=root,
        critical=True,
    )

    # Extract structured info
    extraction: AttorneyExtraction = await evaluator.extract(
        prompt=prompt_extract_attorney_info(),
        template_class=AttorneyExtraction,
        extraction_name="attorney_extraction",
    )

    # Build verification branches (all critical under main_node)
    # 1) Name/format check
    name_leaf = await build_name_output_branch(evaluator, main_node, extraction)

    # 2) Law firm affiliation & role
    await build_law_firm_affiliation_branch(evaluator, main_node, extraction, name_leaf)

    # 3) Julian Assange representation
    await build_assange_branch(evaluator, main_node, extraction, name_leaf)

    # 4) Michael Krautz acquittal
    await build_krautz_branch(evaluator, main_node, extraction, name_leaf)

    # 5) Martin Tankleff overturning
    await build_tankleff_branch(evaluator, main_node, extraction, name_leaf)

    # 6) Nicolás Maduro current representation (as of Jan 2025)
    await build_maduro_branch(evaluator, main_node, extraction, name_leaf)

    # 7) Professional credentials: ACTL Fellow; NACDL President
    await build_credentials_branch(evaluator, main_node, extraction, name_leaf)

    # 8) Educational background: Georgetown Law; Indiana University undergrad
    await build_education_branch(evaluator, main_node, extraction, name_leaf)

    # 9) Career duration: > 30 years
    await build_career_branch(evaluator, main_node, extraction, name_leaf)

    # Return the evaluation summary
    return evaluator.get_summary()