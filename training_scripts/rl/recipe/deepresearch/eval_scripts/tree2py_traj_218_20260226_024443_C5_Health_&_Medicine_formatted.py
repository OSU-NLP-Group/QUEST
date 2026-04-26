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
TASK_ID = "vaccination_policy_comparison"
TASK_DESCRIPTION = (
    "Compare the childhood vaccination policies of Costa Rica, Denmark, and Morocco. "
    "For each country, determine: (1) whether childhood vaccinations are mandatory for school enrollment or voluntary, "
    "(2) the number of vaccines included in the national program or official scheme, "
    "(3) which specific diseases or vaccines are included versus excluded from routine childhood vaccination, and "
    "(4) how these vaccines are made accessible to families (cost structure and availability). "
    "Provide reference URLs supporting your findings for each country."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CostaRicaInfo(BaseModel):
    mandatory_status: Optional[str] = None
    school_enrollment_requirement: Optional[str] = None
    vaccine_count: Optional[str] = None
    vaccines_included: List[str] = Field(default_factory=list)
    vaccines_excluded: List[str] = Field(default_factory=list)
    cost_accessibility: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class DenmarkInfo(BaseModel):
    voluntary_status: Optional[str] = None
    vaccine_count: Optional[str] = None
    included_vaccines: List[str] = Field(default_factory=list)
    excluded_vaccines: List[str] = Field(default_factory=list)
    cost_accessibility: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class MoroccoInfo(BaseModel):
    mandatory_voluntary_status: Optional[str] = None
    nip_coverage_number: Optional[str] = None
    nip_vaccines_included: List[str] = Field(default_factory=list)
    complementary_vaccines: List[str] = Field(default_factory=list)
    cost_structure: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class VaccinationPoliciesExtraction(BaseModel):
    costa_rica: Optional[CostaRicaInfo] = None
    denmark: Optional[DenmarkInfo] = None
    morocco: Optional[MoroccoInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_policies() -> str:
    return """
    Extract structured information for childhood vaccination policies from the provided answer for the three countries:
    Costa Rica, Denmark, and Morocco. For each country, extract exactly what is stated in the answer (do not infer),
    and also extract all reference URLs that the answer associates with that country.

    For Costa Rica (costa_rica):
    - mandatory_status: Short phrase stating whether vaccinations are mandatory for school enrollment or voluntary (e.g., "mandatory for school enrollment", "voluntary").
    - school_enrollment_requirement: Short description if the answer mentions a requirement to present vaccination certificates for annual school enrollment.
    - vaccine_count: The number of vaccines in the official basic childhood scheme; keep as string if expressed non-numerically.
    - vaccines_included: List of vaccines/diseases stated as included in the official basic scheme.
    - vaccines_excluded: List of vaccines/diseases stated as excluded from the routine childhood schedule (if any are mentioned).
    - cost_accessibility: Short description of how vaccines are made accessible (e.g., "free via public health system", "available through campaigns").
    - source_urls: All URLs cited in the answer that support Costa Rica information.

    For Denmark (denmark):
    - voluntary_status: Short phrase stating that vaccinations are voluntary (not mandatory for school/daycare) if mentioned; otherwise what the answer claims.
    - vaccine_count: The number of vaccines in Denmark's childhood schedule as stated in the answer (string allowed).
    - included_vaccines: List of vaccines/diseases that the answer claims are included (e.g., diphtheria, tetanus, pertussis, polio, Hib, pneumococcal, measles, mumps, rubella).
    - excluded_vaccines: List of vaccines/diseases the answer claims Denmark does NOT routinely provide to healthy children (e.g., RSV, rotavirus, varicella, hepatitis B at birth, hepatitis A, influenza, meningococcal).
    - cost_accessibility: Short description of accessibility (e.g., "free under national program; voluntary; provided by GPs").
    - source_urls: All URLs cited in the answer that support Denmark information.

    For Morocco (morocco):
    - mandatory_voluntary_status: Short phrase stating whether participation is mandatory or voluntary if mentioned in the answer.
    - nip_coverage_number: Number of vaccine-preventable diseases covered by the National Immunization Program (string allowed).
    - nip_vaccines_included: List of vaccines/diseases included in Morocco's NIP.
    - complementary_vaccines: List of vaccines classified as complementary (not in NIP), e.g., varicella, hepatitis A, influenza, meningococcal.
    - cost_structure: Short description stating that NIP vaccines are free at public facilities while complementary vaccines are available in the private sector and must be paid for (as claimed).
    - source_urls: All URLs cited in the answer that support Morocco information.

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer. Do not invent URLs.
    - Include full URLs; if protocol is missing, prepend http:// as needed.
    - Assign each URL to the correct country bucket based on how the answer associates the reference.

    If any item is missing in the answer, return null for single fields and empty lists for list fields.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        t = u.strip()
        if not t:
            continue
        if not (t.startswith("http://") or t.startswith("https://")):
            t = "http://" + t
        cleaned.append(t)
    return cleaned


# --------------------------------------------------------------------------- #
# Verification logic per country                                              #
# --------------------------------------------------------------------------- #
async def verify_costa_rica(evaluator: Evaluator, parent_node, cr: Optional[CostaRicaInfo]) -> None:
    country_node = evaluator.add_parallel(
        id="costa_rica_policy",
        desc="Provide accurate information about Costa Rica's childhood vaccination policy",
        parent=parent_node,
        critical=False
    )

    sources = _safe_sources(cr.source_urls if cr else [])

    # Mandatory status for school enrollment
    mandatory_leaf = evaluator.add_leaf(
        id="cr_mandatory_status",
        desc="Correctly identify whether Costa Rica has mandatory vaccination requirements for school enrollment",
        parent=country_node,
        critical=True
    )
    status_text = (cr.mandatory_status or "").lower()
    if any(k in status_text for k in ["voluntary", "optional", "not mandatory"]):
        claim = "In Costa Rica, childhood vaccinations are voluntary (not required) for school enrollment."
    else:
        claim = "In Costa Rica, childhood vaccinations are mandatory or required for school enrollment."
    await evaluator.verify(
        claim=claim,
        node=mandatory_leaf,
        sources=sources,
        additional_instruction="Judge based on official policy references. Look for legal or Ministry of Health statements indicating either mandatory requirements for school enrollment or that vaccination is voluntary."
    )

    # School enrollment certificate requirement
    school_req_leaf = evaluator.add_leaf(
        id="cr_school_enrollment_requirement",
        desc="Accurately describe the requirement to present vaccination certificates for annual school enrollment",
        parent=country_node,
        critical=True
    )
    school_claim = (
        cr.school_enrollment_requirement
        if (cr and cr.school_enrollment_requirement)
        else "Costa Rica requires families to present a child's vaccination certificate (carné/certificado de vacunas) annually for school enrollment."
    )
    await evaluator.verify(
        claim=school_claim,
        node=school_req_leaf,
        sources=sources,
        additional_instruction="Look for mentions of 'carné de vacunas', 'certificado de vacunación', or annual school enrollment documentation requirements in Costa Rica."
    )

    # Vaccine count
    vcount_leaf = evaluator.add_leaf(
        id="cr_vaccine_count",
        desc="State the number of vaccines in Costa Rica's official basic scheme",
        parent=country_node,
        critical=True
    )
    if cr and cr.vaccine_count:
        vcount_claim = f"Costa Rica's official basic childhood vaccination scheme includes {cr.vaccine_count} vaccines."
    else:
        vcount_claim = "Costa Rica's official basic childhood vaccination scheme includes a specific number of vaccines as stated in authoritative references."
    await evaluator.verify(
        claim=vcount_claim,
        node=vcount_leaf,
        sources=sources,
        additional_instruction="Verify the count based on Ministry of Health/CCSS schedule pages or official documents. Minor numeric variations due to combinations are acceptable only if consistent with the stated number."
    )

    # Vaccines included/excluded - split into two leaf checks under a parallel aggregator
    include_exclude_node = evaluator.add_parallel(
        id="cr_vaccines_included_excluded",
        desc="Provide information about which specific vaccines or diseases are covered by Costa Rica's official basic scheme",
        parent=country_node,
        critical=True
    )

    cr_included_leaf = evaluator.add_leaf(
        id="cr_included_supported",
        desc="Included vaccines list for Costa Rica is supported by references",
        parent=include_exclude_node,
        critical=True
    )
    included_list = ", ".join(cr.vaccines_included) if cr and cr.vaccines_included else ""
    included_claim = (
        f"Costa Rica's official basic childhood scheme includes the following vaccines/diseases: {included_list}."
        if included_list
        else "Costa Rica's official basic childhood scheme includes specific vaccines/diseases as detailed in authoritative references."
    )
    await evaluator.verify(
        claim=included_claim,
        node=cr_included_leaf,
        sources=sources,
        additional_instruction="Confirm that the listed vaccines/diseases are included in Costa Rica's routine childhood immunization schedule. Allow for combination vaccines and minor naming variations."
    )

    cr_excluded_leaf = evaluator.add_leaf(
        id="cr_excluded_supported",
        desc="Excluded vaccines list for Costa Rica is supported by references",
        parent=include_exclude_node,
        critical=True
    )
    excluded_list = ", ".join(cr.vaccines_excluded) if cr and cr.vaccines_excluded else ""
    excluded_claim = (
        f"Costa Rica's routine childhood schedule does not include the following vaccines/diseases: {excluded_list}."
        if excluded_list
        else "Costa Rica's routine childhood schedule excludes certain vaccines/diseases, which are not part of the official basic scheme."
    )
    await evaluator.verify(
        claim=excluded_claim,
        node=cr_excluded_leaf,
        sources=sources,
        additional_instruction="Confirm that any listed vaccines are not part of the routine childhood schedule. If only recommended for risk groups or not in the official scheme, treat them as excluded from routine."
    )

    # Cost/accessibility
    cr_cost_leaf = evaluator.add_leaf(
        id="cr_cost_accessibility",
        desc="Describe how Costa Rica's vaccination program is made accessible to families (e.g., through public health system, vaccination campaigns, etc.)",
        parent=country_node,
        critical=True
    )
    cost_claim = (
        cr.cost_accessibility
        if (cr and cr.cost_accessibility)
        else "Childhood vaccines in Costa Rica are provided via the public health system (CCSS/MoH), widely accessible through clinics and vaccination campaigns; many are free of charge."
    )
    await evaluator.verify(
        claim=cost_claim,
        node=cr_cost_leaf,
        sources=sources,
        additional_instruction="Check whether references state provision through the public health system (e.g., CCSS), campaigns, and cost/free-of-charge details for families."
    )

    # Reference URL relevance
    cr_ref_leaf = evaluator.add_leaf(
        id="cr_reference",
        desc="Provide a valid reference URL supporting the Costa Rica vaccination policy information",
        parent=country_node,
        critical=True
    )
    await evaluator.verify(
        claim="These sources provide authoritative information about Costa Rica's childhood vaccination policy or schedule.",
        node=cr_ref_leaf,
        sources=sources,
        additional_instruction="Support only if at least one URL clearly relates to Costa Rica's childhood vaccination policy/schedule (e.g., Ministry of Health, CCSS, WHO/UNICEF country pages)."
    )


async def verify_denmark(evaluator: Evaluator, parent_node, dk: Optional[DenmarkInfo]) -> None:
    country_node = evaluator.add_parallel(
        id="denmark_policy",
        desc="Provide accurate information about Denmark's childhood vaccination policy",
        parent=parent_node,
        critical=False
    )

    sources = _safe_sources(dk.source_urls if dk else [])

    # Voluntary status (no mandatory requirement)
    voluntary_leaf = evaluator.add_leaf(
        id="dk_voluntary_status",
        desc="Correctly identify that Denmark does not have mandatory vaccination requirements for school or daycare attendance",
        parent=country_node,
        critical=True
    )
    status_text = (dk.voluntary_status or "").lower()
    if any(k in status_text for k in ["mandatory", "obligatory", "required"]):
        vol_claim = "Denmark does not require proof of vaccination for school or daycare attendance; participation in the childhood vaccination program is voluntary."
    else:
        vol_claim = "Denmark does not require proof of vaccination for school or daycare attendance; participation in the childhood vaccination program is voluntary."
    await evaluator.verify(
        claim=vol_claim,
        node=voluntary_leaf,
        sources=sources,
        additional_instruction="Confirm via official references (e.g., Danish Health Authority) that vaccinations are voluntary and not required for school/daycare attendance."
    )

    # Vaccine count
    dk_count_leaf = evaluator.add_leaf(
        id="dk_vaccine_count",
        desc="Provide information about the number of vaccines in Denmark's childhood vaccination schedule",
        parent=country_node,
        critical=True
    )
    if dk and dk.vaccine_count:
        count_claim = f"Denmark's childhood vaccination schedule includes {dk.vaccine_count} vaccines."
    else:
        count_claim = "Denmark's childhood vaccination schedule includes a specific number of vaccines as stated by authoritative references."
    await evaluator.verify(
        claim=count_claim,
        node=dk_count_leaf,
        sources=sources,
        additional_instruction="Verify the count using official schedule pages; allow for combination vaccines mapping and minor naming differences."
    )

    # Excluded vaccines (split into specific checks)
    excluded_main = evaluator.add_parallel(
        id="dk_excluded_vaccines",
        desc="Accurately list vaccines Denmark does NOT routinely provide to healthy children",
        parent=country_node,
        critical=True
    )

    excluded_items = [
        ("dk_excl_rsv", "RSV (respiratory syncytial virus)", "Denmark's routine childhood schedule does not include RSV vaccination for healthy children."),
        ("dk_excl_rotavirus", "rotavirus", "Denmark's routine childhood schedule does not include rotavirus vaccination for healthy children."),
        ("dk_excl_varicella", "varicella (chickenpox)", "Denmark's routine childhood schedule does not include varicella (chickenpox) vaccination for healthy children."),
        ("dk_excl_hepb_birth", "hepatitis B at birth", "Denmark does not routinely administer hepatitis B vaccination at birth to healthy infants as part of the standard childhood schedule."),
        ("dk_excl_hepa", "hepatitis A", "Denmark's routine childhood schedule does not include hepatitis A vaccination for healthy children."),
        ("dk_excl_influenza", "influenza", "Denmark's routine childhood schedule does not include seasonal influenza vaccination for healthy children."),
        ("dk_excl_meningococcal", "meningococcal", "Denmark's routine childhood schedule does not include meningococcal vaccination for healthy children."),
    ]

    batch = []
    for leaf_id, display_name, item_claim in excluded_items:
        leaf = evaluator.add_leaf(
            id=leaf_id,
            desc=f"Denmark excludes {display_name} from routine childhood vaccination",
            parent=excluded_main,
            critical=True
        )
        batch.append((
            item_claim,
            sources,
            leaf,
            "Consider 'not in routine schedule' as excluded. If offered only to special/risk groups or not part of the standard childhood program, treat as excluded from routine."
        ))

    # Summary check for excluded list declared in the answer
    excl_summary_leaf = evaluator.add_leaf(
        id="dk_excluded_summary",
        desc="Excluded vaccines summary is supported by references for Denmark",
        parent=excluded_main,
        critical=True
    )
    excluded_list = ", ".join(dk.excluded_vaccines) if dk and dk.excluded_vaccines else ""
    excl_summary_claim = (
        f"Denmark's routine childhood schedule excludes: {excluded_list}."
        if excluded_list else
        "Denmark's routine childhood schedule excludes certain vaccines for healthy children; confirm exclusion according to official references."
    )
    batch.append((
        excl_summary_claim,
        sources,
        excl_summary_leaf,
        "Confirm that any vaccines listed as excluded are indeed not part of the routine childhood program. Accept minor naming differences."
    ))

    await evaluator.batch_verify(batch)

    # Included vaccines (specific minimum set + summary)
    included_main = evaluator.add_parallel(
        id="dk_included_vaccines",
        desc="Accurately list vaccines Denmark DOES include in its childhood schedule",
        parent=country_node,
        critical=True
    )

    included_items = [
        ("dk_incl_dtap", "diphtheria, tetanus, pertussis (DTaP)", "Denmark's childhood schedule includes DTaP (diphtheria, tetanus, pertussis)."),
        ("dk_incl_polio", "polio", "Denmark's childhood schedule includes polio vaccination."),
        ("dk_incl_hib", "Haemophilus influenzae type b (Hib)", "Denmark's childhood schedule includes Hib vaccination."),
        ("dk_incl_pcv", "pneumococcal (PCV)", "Denmark's childhood schedule includes pneumococcal (PCV) vaccination."),
        ("dk_incl_mmr", "measles, mumps, rubella (MMR)", "Denmark's childhood schedule includes MMR (measles, mumps, rubella) vaccination."),
    ]

    batch2 = []
    for leaf_id, display_name, item_claim in included_items:
        leaf = evaluator.add_leaf(
            id=leaf_id,
            desc=f"Denmark includes {display_name} in routine childhood vaccination",
            parent=included_main,
            critical=True
        )
        batch2.append((
            item_claim,
            sources,
            leaf,
            "Confirm inclusion in Denmark's official childhood immunization schedule. Allow combination vaccines and minor naming variations."
        ))

    incl_summary_leaf = evaluator.add_leaf(
        id="dk_included_summary",
        desc="Included vaccines summary is supported by references for Denmark",
        parent=included_main,
        critical=True
    )
    included_list = ", ".join(dk.included_vaccines) if dk and dk.included_vaccines else ""
    incl_summary_claim = (
        f"Denmark's childhood schedule includes: {included_list}."
        if included_list else
        "Denmark's childhood schedule includes a defined set of vaccines for healthy children according to official references."
    )
    batch2.append((
        incl_summary_claim,
        sources,
        incl_summary_leaf,
        "Confirm that any vaccines listed as included are indeed part of the routine childhood schedule. Accept minor naming differences."
    ))

    await evaluator.batch_verify(batch2)

    # Cost/accessibility
    dk_cost_leaf = evaluator.add_leaf(
        id="dk_cost_accessibility",
        desc="Describe how Denmark's vaccination program is made accessible to families (e.g., public health system, voluntary participation, etc.)",
        parent=country_node,
        critical=True
    )
    cost_claim = (
        dk.cost_accessibility
        if (dk and dk.cost_accessibility)
        else "Denmark's childhood vaccinations are offered free under the national program, voluntary, and typically provided through general practitioners."
    )
    await evaluator.verify(
        claim=cost_claim,
        node=dk_cost_leaf,
        sources=sources,
        additional_instruction="Confirm free-of-charge provision under the Danish Childhood Vaccination Programme and delivery via GPs; participation is voluntary."
    )

    # Reference URL relevance
    dk_ref_leaf = evaluator.add_leaf(
        id="dk_reference",
        desc="Provide a valid reference URL supporting the Denmark vaccination policy information",
        parent=country_node,
        critical=True
    )
    await evaluator.verify(
        claim="These sources provide authoritative information about Denmark's childhood vaccination policy or schedule.",
        node=dk_ref_leaf,
        sources=sources,
        additional_instruction="Support only if at least one URL clearly relates to Denmark's childhood vaccination policy/schedule (e.g., Danish Health Authority pages)."
    )


async def verify_morocco(evaluator: Evaluator, parent_node, ma: Optional[MoroccoInfo]) -> None:
    country_node = evaluator.add_parallel(
        id="morocco_policy",
        desc="Provide accurate information about Morocco's childhood vaccination policy",
        parent=parent_node,
        critical=False
    )

    sources = _safe_sources(ma.source_urls if ma else [])

    # Mandatory/voluntary status
    ma_status_leaf = evaluator.add_leaf(
        id="ma_mandatory_voluntary_status",
        desc="Address whether Morocco's vaccination program involves mandatory or voluntary participation",
        parent=country_node,
        critical=True
    )
    status_text = (ma.mandatory_voluntary_status or "").lower()
    if any(k in status_text for k in ["voluntary", "optional", "not mandatory"]):
        claim = "Morocco's childhood vaccination program is voluntary (not legally mandatory)."
    else:
        claim = "Morocco's childhood vaccination program (NIP) has compulsory or mandated participation according to national policy."
    await evaluator.verify(
        claim=claim,
        node=ma_status_leaf,
        sources=sources,
        additional_instruction="Judge based on Moroccan Ministry of Health or authoritative references; identify whether NIP participation is compulsory or voluntary."
    )

    # NIP coverage number
    ma_coverage_leaf = evaluator.add_leaf(
        id="ma_nip_coverage",
        desc="State the number of vaccine-preventable diseases covered by Morocco's National Immunization Program",
        parent=country_node,
        critical=True
    )
    if ma and ma.nip_coverage_number:
        cov_claim = f"Morocco's National Immunization Program covers {ma.nip_coverage_number} vaccine-preventable diseases."
    else:
        cov_claim = "Morocco's National Immunization Program covers a specific number of vaccine-preventable diseases as stated by authoritative references."
    await evaluator.verify(
        claim=cov_claim,
        node=ma_coverage_leaf,
        sources=sources,
        additional_instruction="Confirm the number using official NIP documentation or Ministry of Health references; allow minor wording variations."
    )

    # NIP included vaccines summary
    ma_incl_leaf = evaluator.add_leaf(
        id="ma_nip_vaccines_included",
        desc="Provide information about which vaccines are included in Morocco's National Immunization Program",
        parent=country_node,
        critical=True
    )
    incl_list = ", ".join(ma.nip_vaccines_included) if ma and ma.nip_vaccines_included else ""
    incl_claim = (
        f"Morocco's NIP includes the following vaccines/diseases: {incl_list}."
        if incl_list else
        "Morocco's NIP includes a defined set of vaccines/diseases for children according to authoritative references."
    )
    await evaluator.verify(
        claim=incl_claim,
        node=ma_incl_leaf,
        sources=sources,
        additional_instruction="Confirm inclusion in Morocco's National Immunization Program. Allow combination vaccines and naming variations."
    )

    # Complementary vaccines - split into specific checks + summary
    comp_main = evaluator.add_parallel(
        id="ma_complementary_vaccines",
        desc="Accurately list vaccines classified as complementary (not in NIP) in Morocco",
        parent=country_node,
        critical=True
    )

    comp_items = [
        ("ma_comp_varicella", "varicella", "In Morocco, varicella (chickenpox) vaccination is classified as complementary (not part of NIP) and typically available in the private sector."),
        ("ma_comp_hepa", "hepatitis A", "In Morocco, hepatitis A vaccination is classified as complementary (not part of NIP) and typically available in the private sector."),
        ("ma_comp_influenza", "influenza", "In Morocco, seasonal influenza vaccination is classified as complementary (not part of NIP) and typically available in the private sector."),
        ("ma_comp_meningococcal", "meningococcal", "In Morocco, meningococcal vaccination is classified as complementary (not part of NIP) and typically available in the private sector."),
    ]
    batch = []
    for leaf_id, display_name, item_claim in comp_items:
        leaf = evaluator.add_leaf(
            id=leaf_id,
            desc=f"Morocco classifies {display_name} as complementary (not in NIP)",
            parent=comp_main,
            critical=True
        )
        batch.append((
            item_claim,
            sources,
            leaf,
            "Confirm classification as complementary (hors PNI/not part of NIP). If only provided in private sector or for special circumstances, treat as complementary."
        ))

    comp_summary_leaf = evaluator.add_leaf(
        id="ma_complementary_summary",
        desc="Complementary vaccines summary is supported by references for Morocco",
        parent=comp_main,
        critical=True
    )
    comp_list = ", ".join(ma.complementary_vaccines) if ma and ma.complementary_vaccines else ""
    comp_summary_claim = (
        f"In Morocco, the following vaccines are complementary (not in NIP): {comp_list}."
        if comp_list else
        "Morocco has a set of complementary vaccines outside the NIP, typically available in the private sector."
    )
    batch.append((
        comp_summary_claim,
        sources,
        comp_summary_leaf,
        "Confirm that any listed vaccines are outside the NIP and considered complementary."
    ))

    await evaluator.batch_verify(batch)

    # Cost structure
    ma_cost_leaf = evaluator.add_leaf(
        id="ma_cost_structure",
        desc="Correctly describe that NIP vaccines are free at public facilities while complementary vaccines are available in the private sector and must be paid for by families",
        parent=country_node,
        critical=True
    )
    cost_claim = (
        ma.cost_structure
        if (ma and ma.cost_structure)
        else "In Morocco, NIP vaccines are provided free of charge at public facilities; complementary vaccines are available in the private sector and must be paid by families."
    )
    await evaluator.verify(
        claim=cost_claim,
        node=ma_cost_leaf,
        sources=sources,
        additional_instruction="Confirm free access for NIP vaccines and paid/private sector access for complementary vaccines."
    )

    # Reference URL relevance
    ma_ref_leaf = evaluator.add_leaf(
        id="ma_reference",
        desc="Provide a valid reference URL supporting the Morocco vaccination policy information",
        parent=country_node,
        critical=True
    )
    await evaluator.verify(
        claim="These sources provide authoritative information about Morocco's childhood vaccination policy/NIP.",
        node=ma_ref_leaf,
        sources=sources,
        additional_instruction="Support only if at least one URL clearly relates to Morocco's NIP or childhood vaccination policy (e.g., MoH, WHO/UNICEF country pages)."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer comparing childhood vaccination policies of Costa Rica, Denmark, and Morocco.
    """
    # Initialize evaluator (root is non-critical per framework; allows partial credit across countries)
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

    # Extract structured information for all countries
    extraction = await evaluator.extract(
        prompt=prompt_extract_policies(),
        template_class=VaccinationPoliciesExtraction,
        extraction_name="vaccination_policies"
    )

    # Build and verify each country's subtree
    await verify_costa_rica(evaluator, root, extraction.costa_rica)
    await verify_denmark(evaluator, root, extraction.denmark)
    await verify_morocco(evaluator, root, extraction.morocco)

    # Return structured evaluation summary
    return evaluator.get_summary()