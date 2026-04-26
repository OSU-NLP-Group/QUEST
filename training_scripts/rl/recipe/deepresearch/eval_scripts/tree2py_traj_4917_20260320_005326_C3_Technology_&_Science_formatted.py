import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "cyber_acquisition_undergrad"
TASK_DESCRIPTION = (
    "A major enterprise software company announced a significant cybersecurity acquisition in December 2025 for "
    "$7.75 billion in cash. The acquired company, which was founded in November 2015, specializes in cyber exposure "
    "management and security. Identify the educational institution where the CEO and co-founder of the acquired "
    "company earned their undergraduate degree, and specify the field of study for that degree."
)


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class AcquisitionExtraction(BaseModel):
    # Core identity
    acquired_company_name: Optional[str] = None
    acquirer_company_name: Optional[str] = None

    # Constraint details as seen in the answer
    announcement_month_year: Optional[str] = None  # e.g., "December 2025"
    price_text: Optional[str] = None               # e.g., "$7.75 billion", "US$7.75B", etc.
    payment_nature: Optional[str] = None           # e.g., "cash", "all-cash"
    specialization_summary: Optional[str] = None   # e.g., "cyber exposure management and security"
    founded_month_year: Optional[str] = None       # e.g., "November 2015"

    # Source URLs provided in the answer; separate buckets for clarity
    acquisition_sources: List[str] = Field(default_factory=list)      # press release / coverage mentioning date/price
    specialization_sources: List[str] = Field(default_factory=list)   # company/about/analyst references specialization
    founding_sources: List[str] = Field(default_factory=list)         # company/about/LinkedIn pages showing Nov 2015


class CEOCofounderExtraction(BaseModel):
    person_name: Optional[str] = None
    ceo_role_sources: List[str] = Field(default_factory=list)
    cofounder_role_sources: List[str] = Field(default_factory=list)


class UndergraduateExtraction(BaseModel):
    institution: Optional[str] = None
    field_of_study: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_acquisition() -> str:
    return """
Extract details about the cybersecurity acquisition described in the answer. Return a single JSON object with:

- acquired_company_name: The name of the acquired company (the target).
- acquirer_company_name: The acquiring enterprise software company, if mentioned.
- announcement_month_year: The month and year of the announcement as written (e.g., "December 2025") if provided.
- price_text: The deal value text as written (e.g., "$7.75 billion", "US$7.75B", "7.75 billion USD").
- payment_nature: How it was paid if stated (e.g., "cash", "all-cash", "cash consideration").
- specialization_summary: A short phrase from the answer describing the acquired company's specialization (aim for wording close to the answer, e.g., "cyber exposure management and security").
- founded_month_year: The month and year the acquired company was founded as written (e.g., "November 2015") if provided.
- acquisition_sources: Array of URLs in the answer that substantiate the announcement timing and/or deal value (press releases, major news coverage).
- specialization_sources: Array of URLs in the answer that support the company's specialization/positioning.
- founding_sources: Array of URLs in the answer that support the founding month/year.

Rules:
- Extract only what is explicitly stated in the answer; do not infer.
- Return null for missing scalar fields and [] for missing arrays.
- For URLs, extract the actual URLs (including from markdown links). Do not invent URLs.
    """


def prompt_extract_ceo_cofounder() -> str:
    return """
Identify the person in the answer who is both the CEO and a co-founder of the acquired company. Return a single JSON object with:

- person_name: The full name of the person identified as both CEO and co-founder.
- ceo_role_sources: Array of URLs in the answer that verify this person is (or was at the relevant time) the CEO of the acquired company.
- cofounder_role_sources: Array of URLs in the answer that verify this person is a co-founder of the acquired company.

Rules:
- If the answer lists multiple people, pick the one who satisfies BOTH conditions (CEO and co-founder).
- Extract only from the answer; do not invent.
- Use [] for missing URL arrays and null for missing person_name.
    """


def prompt_extract_undergrad(person_placeholder: str = "the CEO/co-founder") -> str:
    return f"""
Extract the undergraduate education information for {person_placeholder} as presented in the answer. Return a single JSON object with:

- institution: The name of the educational institution where the person's undergraduate degree was earned.
- field_of_study: The field/major for the undergraduate degree as written (e.g., "Computer Science", "Electrical Engineering", "Economics").
- sources: Array of URLs in the answer that verify both the institution and the undergraduate field of study for this person.

Rules:
- If the answer provides multiple degrees, select the undergraduate one (bachelor level: BA, BS, BSc, etc.).
- If only the institution or only the field is provided, extract what is present and set the missing field to null.
- Return [] for missing sources.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_acquired_company(evaluator: Evaluator, parent_node, acq: AcquisitionExtraction) -> None:
    """
    Build and verify the Identify_Acquired_Company branch.
    """
    id_node = evaluator.add_parallel(
        id="Identify_Acquired_Company",
        desc="Determine the acquired company that matches the acquisition constraints.",
        parent=parent_node,
        critical=True
    )

    # Optional presence check for company name (gates other sub-branches conceptually)
    evaluator.add_custom_node(
        result=bool(acq.acquired_company_name and acq.acquired_company_name.strip()),
        id="Acquired_Company_Name_Present",
        desc="Acquired company name is present in the answer.",
        parent=id_node,
        critical=True
    )

    # 1) Constraints aggregator
    constraints_agg = evaluator.add_parallel(
        id="Acquired_Company_Matches_All_Constraints",
        desc="The named acquired company matches the stated acquisition constraints.",
        parent=id_node,
        critical=True
    )

    # 1.a) Announcement in December 2025
    branch_announce = evaluator.add_sequential(
        id="Constraint_Announcement_Dec2025_Branch",
        desc="Announcement timing constraint verification branch",
        parent=constraints_agg,
        critical=True
    )
    # Gating: need at least one acquisition source
    gate_announce = evaluator.add_custom_node(
        result=len(acq.acquisition_sources) > 0,
        id="Constraint_Announcement_Sources_Available",
        desc="At least one acquisition source URL is provided for announcement/price.",
        parent=branch_announce,
        critical=True
    )
    leaf_announce = evaluator.add_leaf(
        id="Acquisition_Announcement_Dec2025",
        desc="The acquisition announcement occurred in December 2025.",
        parent=branch_announce,
        critical=True
    )
    company = acq.acquired_company_name or "the company"
    acquirer = acq.acquirer_company_name
    if acquirer:
        claim_announce = f"A publicly available page states that {acquirer} announced its acquisition of {company} in December 2025."
    else:
        claim_announce = f"A publicly available page states that an acquisition of {company} was announced in December 2025."
    await evaluator.verify(
        claim=claim_announce,
        node=leaf_announce,
        sources=acq.acquisition_sources,
        additional_instruction="Allow phrasing variations such as 'announced in December 2025' or date formats on a press release or major news outlet page."
    )

    # 1.b) Price is $7.75B in cash
    branch_price = evaluator.add_sequential(
        id="Constraint_Price_Cash_7_75B_Branch",
        desc="Price/payment nature constraint verification branch",
        parent=constraints_agg,
        critical=True
    )
    gate_price = evaluator.add_custom_node(
        result=len(acq.acquisition_sources) > 0,
        id="Constraint_Price_Sources_Available",
        desc="At least one acquisition source URL is provided for deal value/payment.",
        parent=branch_price,
        critical=True
    )
    leaf_price = evaluator.add_leaf(
        id="Acquisition_Price_Cash_7_75B",
        desc="The acquisition was valued at $7.75 billion in cash.",
        parent=branch_price,
        critical=True
    )
    claim_price = "This page reports the deal value as approximately $7.75 billion and that the consideration was all-cash (or in cash)."
    await evaluator.verify(
        claim=claim_price,
        node=leaf_price,
        sources=acq.acquisition_sources,
        additional_instruction="Accept reasonable numeric formatting (e.g., $7.75B, US$7.75 billion) and synonyms for payment nature (e.g., 'all-cash', 'cash consideration')."
    )

    # 1.c) Specialization: cyber exposure management and security
    branch_spec = evaluator.add_sequential(
        id="Constraint_Specialization_Cyber_Exposure_Branch",
        desc="Specialization constraint verification branch",
        parent=constraints_agg,
        critical=True
    )
    gate_spec = evaluator.add_custom_node(
        result=len(acq.specialization_sources) > 0,
        id="Constraint_Specialization_Sources_Available",
        desc="At least one specialization source URL is provided.",
        parent=branch_spec,
        critical=True
    )
    leaf_spec = evaluator.add_leaf(
        id="Specialization_Cyber_Exposure",
        desc="The company specializes in cyber exposure management and security.",
        parent=branch_spec,
        critical=True
    )
    claim_spec = f"The page states that {company} specializes in cyber exposure management and security (or very close phrasing such as cyber exposure/exposure management in security)."
    await evaluator.verify(
        claim=claim_spec,
        node=leaf_spec,
        sources=acq.specialization_sources,
        additional_instruction="Allow close paraphrases such as 'cyber exposure', 'exposure management', 'exposure-based risk management', provided the meaning clearly matches."
    )

    # 1.d) Founded in November 2015
    branch_founding = evaluator.add_sequential(
        id="Constraint_Founded_Nov2015_Branch",
        desc="Founding date constraint verification branch",
        parent=constraints_agg,
        critical=True
    )
    gate_found = evaluator.add_custom_node(
        result=len(acq.founding_sources) > 0,
        id="Constraint_Founding_Sources_Available",
        desc="At least one founding date source URL is provided.",
        parent=branch_founding,
        critical=True
    )
    leaf_found = evaluator.add_leaf(
        id="Founded_November_2015",
        desc="The acquired company was founded in November 2015.",
        parent=branch_founding,
        critical=True
    )
    claim_found = f"The page explicitly indicates that {company} was founded in November 2015."
    await evaluator.verify(
        claim=claim_found,
        node=leaf_found,
        sources=acq.founding_sources,
        additional_instruction="Allow minor wording variations (e.g., 'founded Nov 2015', 'founded in November 2015')."
    )

    # 2) Public verifiability node
    # This checks that at least one provided URL is a publicly available article/press release about the acquisition itself.
    public_leaf = evaluator.add_leaf(
        id="Acquired_Company_Public_Verifiability",
        desc="Provide publicly available source(s) that substantiate the acquisition constraints for the named acquired company.",
        parent=id_node,
        critical=True
    )
    if acquirer:
        claim_pub = f"This page is a publicly available article or press release about {acquirer} acquiring {company}."
    else:
        claim_pub = f"This page is a publicly available article or press release about the acquisition of {company}."
    await evaluator.verify(
        claim=claim_pub,
        node=public_leaf,
        sources=acq.acquisition_sources,
        additional_instruction="The page should be accessible on the open web and clearly reference the acquisition event."
    )


async def verify_ceo_cofounder(evaluator: Evaluator, parent_node, ceo: CEOCofounderExtraction, acq: AcquisitionExtraction) -> None:
    """
    Build and verify the Identify_CEO_CoFounder branch.
    """
    ceo_node = evaluator.add_parallel(
        id="Identify_CEO_CoFounder",
        desc="Identify the person who is both CEO and a co-founder of the acquired company.",
        parent=parent_node,
        critical=True
    )

    # Optional presence check for person name
    evaluator.add_custom_node(
        result=bool(ceo.person_name and ceo.person_name.strip()),
        id="CEO_CoFounder_Name_Present",
        desc="The CEO/co-founder name is present in the answer.",
        parent=ceo_node,
        critical=True
    )

    # Core verification: both roles (using two sequential branches for gated checks)
    both_roles = evaluator.add_parallel(
        id="CEO_Is_CoFounder",
        desc="The identified individual is both (1) CEO and (2) a co-founder of the acquired company.",
        parent=ceo_node,
        critical=True
    )

    # CEO role branch
    ceo_branch = evaluator.add_sequential(
        id="CEO_Role_Branch",
        desc="CEO role verification branch",
        parent=both_roles,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ceo.ceo_role_sources) > 0,
        id="CEO_Role_Sources_Available",
        desc="At least one URL is provided verifying the CEO role.",
        parent=ceo_branch,
        critical=True
    )
    leaf_ceo = evaluator.add_leaf(
        id="CEO_Role_Verified",
        desc="The individual is the CEO of the acquired company.",
        parent=ceo_branch,
        critical=True
    )
    person = ceo.person_name or "the person"
    company = acq.acquired_company_name or "the company"
    claim_ceo = f"The page indicates that {person} is the CEO of {company}."
    await evaluator.verify(
        claim=claim_ceo,
        node=leaf_ceo,
        sources=ceo.ceo_role_sources,
        additional_instruction="Allow phrasing variations like 'Chief Executive Officer' or 'CEO'."
    )

    # Co-founder role branch
    cofound_branch = evaluator.add_sequential(
        id="CoFounder_Role_Branch",
        desc="Co-founder role verification branch",
        parent=both_roles,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ceo.cofounder_role_sources) > 0,
        id="CoFounder_Role_Sources_Available",
        desc="At least one URL is provided verifying the co-founder status.",
        parent=cofound_branch,
        critical=True
    )
    leaf_cofound = evaluator.add_leaf(
        id="CoFounder_Status_Verified",
        desc="The individual is a co-founder of the acquired company.",
        parent=cofound_branch,
        critical=True
    )
    claim_cofound = f"The page indicates that {person} is a co-founder of {company}."
    await evaluator.verify(
        claim=claim_cofound,
        node=leaf_cofound,
        sources=ceo.cofounder_role_sources,
        additional_instruction="Accept variations such as 'cofounder'/'co-founder' and pages like official bios or reputable press."
    )

    # Public verifiability for both roles (at least one page mentions both)
    public_leaf = evaluator.add_leaf(
        id="CEO_CoFounder_Public_Verifiability",
        desc="Provide publicly available source(s) verifying the individual's CEO role and co-founder status.",
        parent=ceo_node,
        critical=True
    )
    combined_sources = list(dict.fromkeys((ceo.ceo_role_sources or []) + (ceo.cofounder_role_sources or [])))
    claim_pub = f"This page mentions that {person} is both CEO and co-founder of {company}."
    await evaluator.verify(
        claim=claim_pub,
        node=public_leaf,
        sources=combined_sources,
        additional_instruction="A single page that mentions both roles suffices; otherwise, any one page that clearly contains both facts."
    )


async def verify_undergrad(evaluator: Evaluator, parent_node, edu: UndergraduateExtraction, ceo: CEOCofounderExtraction) -> None:
    """
    Build and verify the Undergraduate_Education branch.
    """
    ug_node = evaluator.add_parallel(
        id="Undergraduate_Education",
        desc="Provide the CEO/co-founder's undergraduate institution and field of study.",
        parent=parent_node,
        critical=True
    )

    person = ceo.person_name or "the person"
    institution = edu.institution or ""
    field = edu.field_of_study or ""
    sources = edu.sources or []

    # Institution branch
    inst_branch = evaluator.add_sequential(
        id="Undergrad_Institution_Branch",
        desc="Undergraduate institution verification branch",
        parent=ug_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0 and bool(institution.strip()),
        id="Undergrad_Institution_Inputs_Available",
        desc="Undergraduate institution and at least one source URL are provided.",
        parent=inst_branch,
        critical=True
    )
    leaf_inst = evaluator.add_leaf(
        id="Undergrad_Institution",
        desc="State the educational institution where the CEO/co-founder earned their undergraduate degree.",
        parent=inst_branch,
        critical=True
    )
    claim_inst = f"This page indicates that {person} earned a bachelor's degree from {institution}."
    await evaluator.verify(
        claim=claim_inst,
        node=leaf_inst,
        sources=sources,
        additional_instruction="Allow BA/BS/BSc naming variants and institution naming variants (e.g., abbreviations or official full names)."
    )

    # Field of study branch
    field_branch = evaluator.add_sequential(
        id="Undergrad_Field_Branch",
        desc="Undergraduate field of study verification branch",
        parent=ug_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0 and bool(field.strip()),
        id="Undergrad_Field_Inputs_Available",
        desc="Undergraduate field of study and at least one source URL are provided.",
        parent=field_branch,
        critical=True
    )
    leaf_field = evaluator.add_leaf(
        id="Undergrad_Field_Of_Study",
        desc="State the field of study for the CEO/co-founder's undergraduate degree.",
        parent=field_branch,
        critical=True
    )
    claim_field = f"This page indicates that {person}'s bachelor's degree field of study is {field}."
    await evaluator.verify(
        claim=claim_field,
        node=leaf_field,
        sources=sources,
        additional_instruction="Allow close variants and synonyms (e.g., 'CS' for 'Computer Science', 'EE' for 'Electrical Engineering')."
    )

    # Public verifiability leaf (both institution and field on at least one page)
    public_leaf = evaluator.add_leaf(
        id="Undergrad_Public_Verifiability",
        desc="Provide publicly available source(s) verifying the undergraduate institution and the undergraduate field of study.",
        parent=ug_node,
        critical=True
    )
    claim_pub = f"This page states that {person} earned a bachelor's degree in {field} from {institution}."
    await evaluator.verify(
        claim=claim_pub,
        node=public_leaf,
        sources=sources,
        additional_instruction="A single authoritative profile/bio or reputable coverage that clearly states both institution and field is acceptable."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the cybersecurity acquisition undergraduate education task.
    """
    # 1) Initialize evaluator (root is non-critical by framework design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # We'll add a critical sequential main node under this root
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

    # Create a critical sequential main node to enforce overall gating (as per rubric Root critical + sequential)
    main = evaluator.add_sequential(
        id="Root",
        desc="Identify the undergraduate institution and field of study of the CEO/co-founder of the acquired company defined by the acquisition constraints, with information verifiable via publicly available sources.",
        parent=root,
        critical=True,
    )

    # 2) Extract structured information
    acq_ext, ceo_ext, ug_ext = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_acquisition(),
            template_class=AcquisitionExtraction,
            extraction_name="acquisition_info",
        ),
        evaluator.extract(
            prompt=prompt_extract_ceo_cofounder(),
            template_class=CEOCofounderExtraction,
            extraction_name="ceo_cofounder_info",
        ),
        evaluator.extract(
            prompt=prompt_extract_undergrad(),
            template_class=UndergraduateExtraction,
            extraction_name="undergraduate_info",
        ),
    )

    # 3) Build verification tree following rubric structure
    await verify_acquired_company(evaluator, main, acq_ext)
    await verify_ceo_cofounder(evaluator, main, ceo_ext, acq_ext)
    await verify_undergrad(evaluator, main, ug_ext, ceo_ext)

    # 4) Return standardized summary
    return evaluator.get_summary()