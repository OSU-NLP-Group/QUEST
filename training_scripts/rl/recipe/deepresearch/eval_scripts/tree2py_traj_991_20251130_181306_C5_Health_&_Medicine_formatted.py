import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "healthcare_service_research_walgreens_oh_glp1"
TASK_DESCRIPTION = (
    "For a resident of Columbus, Ohio who is considering healthcare services at Walgreens pharmacies, "
    "provide a comprehensive research report that addresses Walgreens/OH flu-shot administration rules "
    "and GLP-1 weight loss medication eligibility, with official-source URL citations supporting each required claim."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FluSources(BaseModel):
    walgreens_min_age_urls: List[str] = Field(default_factory=list)
    ohio_tech_min_age_urls: List[str] = Field(default_factory=list)
    provider_types_urls: List[str] = Field(default_factory=list)
    all_locations_urls: List[str] = Field(default_factory=list)
    conclusion_urls: List[str] = Field(default_factory=list)
    columbus_locations_urls: List[str] = Field(default_factory=list)


class GLP1Extraction(BaseModel):
    primary_bmi_urls: List[str] = Field(default_factory=list)
    alt_path_urls: List[str] = Field(default_factory=list)
    comorbidity_urls: List[str] = Field(default_factory=list)
    wegovy_approval_urls: List[str] = Field(default_factory=list)
    wegovy_manufacturer_urls: List[str] = Field(default_factory=list)
    ozempic_manufacturer_urls: List[str] = Field(default_factory=list)
    comorbidity_examples: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_flu_sources() -> str:
    return """
    Extract all official-source URLs mentioned in the answer that support the Walgreens/OH flu-shot administration section.
    Return the following fields, each as an array of URLs (or an empty array if not provided in the answer):
    - walgreens_min_age_urls: URLs from Walgreens official websites that support the minimum age policy for flu shots (e.g., walgreens.com or subdomains thereof).
    - ohio_tech_min_age_urls: URLs from official Ohio sources that support the pharmacy technician immunization minimum age restriction (e.g., pharmacy.ohio.gov, bop.ohio.gov, codes.ohio.gov, ohio.gov).
    - provider_types_urls: URLs from Walgreens official pages that describe provider types who administer flu shots (pharmacists, pharmacy interns, trained technicians).
    - all_locations_urls: URLs from Walgreens official pages used to support the claim that all Walgreens retail locations provide flu shot services.
    - conclusion_urls: URLs used to support the final conclusion for ages 3–4 at Ohio Walgreens locations (you may reuse Walgreens and Ohio sources).
    - columbus_locations_urls: URLs supporting that multiple Walgreens locations in Columbus, Ohio offer immunization services (prefer Walgreens store/immunization locator pages).
    
    IMPORTANT:
    - Extract only URLs that are explicitly present in the answer text. Do not infer or add URLs.
    - Keep URLs exactly as they appear (markdown links should be converted to the raw URL).
    - If no URL is provided for a field, output an empty array for that field.
    """


def prompt_extract_glp1_sources_and_comorbidities() -> str:
    return """
    Extract official-source URLs and the comorbidity examples mentioned in the answer that support GLP-1 weight-loss medication eligibility.
    Return the following fields:
    - primary_bmi_urls: Array of URLs (FDA labeling and/or manufacturer prescribing information) supporting primary BMI threshold (BMI ≥ 30).
    - alt_path_urls: Array of URLs (FDA labeling and/or manufacturer prescribing information) supporting the alternative pathway (BMI ≥ 27 + at least one weight-related comorbidity).
    - comorbidity_urls: Array of URLs (FDA/manufacturer) supporting the specific comorbidity examples mentioned in the answer.
    - wegovy_approval_urls: Array of URLs (FDA label/Drugs@FDA and/or manufacturer site) supporting Wegovy’s FDA approval for weight loss management (adult chronic weight management).
    - wegovy_manufacturer_urls: Array of URLs (manufacturer site or FDA source) supporting that Novo Nordisk manufactures Wegovy.
    - ozempic_manufacturer_urls: Array of URLs (manufacturer site or FDA source) supporting that Novo Nordisk manufactures Ozempic.
    - comorbidity_examples: Array of comorbidity names the answer explicitly lists as qualifying weight-related comorbidities (choose from terms like "high blood pressure"/"hypertension", "diabetes" (usually type 2), "high cholesterol"/"hyperlipidemia", "obstructive sleep apnea").
    
    IMPORTANT:
    - Extract only the URLs explicitly present in the answer. Do not infer or add URLs.
    - Extract comorbidity names exactly as they appear in the answer (normalize obvious synonyms if the answer already normalized them, otherwise keep as-is).
    - If the answer does not provide URLs for a field, output an empty array for that field.
    - If no comorbidity examples are mentioned, return an empty array for comorbidity_examples.
    """


# --------------------------------------------------------------------------- #
# Helper functions for leaf verification                                      #
# --------------------------------------------------------------------------- #
async def add_simple_leaf_and_verify(
    evaluator: Evaluator,
    leaf_id: str,
    desc: str,
    parent,
    claim: str,
    critical: bool = True,
    add_ins: Optional[str] = None,
) -> None:
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction=add_ins or "Focus on whether the answer explicitly states this. Accept reasonable paraphrases and synonyms."
    )


async def add_citation_leaf_and_verify(
    evaluator: Evaluator,
    leaf_id: str,
    desc: str,
    parent,
    claim: str,
    sources: Optional[List[str]],
    critical: bool = True,
    add_ins: Optional[str] = None,
) -> None:
    if not sources or len(sources) == 0:
        # No URLs provided in the answer -> fail this citation leaf directly
        evaluator.add_leaf(
            id=leaf_id,
            desc=desc,
            parent=parent,
            critical=critical,
            score=0.0,
            status="failed",
        )
        return

    node = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=add_ins or "Treat the page as supporting only if its content explicitly supports the claim."
    )


# --------------------------------------------------------------------------- #
# Flu-shot administration regulations verification                            #
# --------------------------------------------------------------------------- #
async def verify_flu_section(
    evaluator: Evaluator,
    parent_node,
    flu_sources: FluSources,
) -> None:
    flu_node = evaluator.add_parallel(
        id="Flu_Shot_Administration_Regulations",
        desc="Correctly states required flu-shot policy/regulatory facts and the resulting authorization for ages 3–4 in Ohio Walgreens locations, with official citations.",
        parent=parent_node,
        critical=True,
    )

    # Walgreens minimum age nationwide: 3+ years
    min_age_node = evaluator.add_parallel(
        id="Walgreens_Flu_Shot_Minimum_Age_Nationwide",
        desc="Walgreens minimum age for flu shots nationwide is correctly identified as 3 years old and older, with an official Walgreens citation.",
        parent=flu_node,
        critical=True,
    )

    await add_simple_leaf_and_verify(
        evaluator,
        leaf_id="States_Minimum_Age_Equals_3_Years_And_Older",
        desc="States that Walgreens administers flu shots to patients aged 3 years and older nationwide.",
        parent=min_age_node,
        claim="The answer explicitly states that Walgreens administers flu shots to patients aged 3 years and older nationwide.",
        critical=True,
        add_ins="Accept phrasings like 'age 3+', '3 and older', or 'three years and up'."
    )
    await add_citation_leaf_and_verify(
        evaluator,
        leaf_id="Cites_Official_Walgreens_Source",
        desc="Provides at least one official Walgreens URL supporting the ≥3 years nationwide minimum age policy.",
        parent=min_age_node,
        claim="Walgreens official page supports the policy that flu shots are available to patients 3 years and older.",
        sources=flu_sources.walgreens_min_age_urls,
        critical=True,
        add_ins="Consider a URL official only if the domain is walgreens.com (or a Walgreens-owned subdomain). If the URL does not appear to be official Walgreens, treat as not supported."
    )

    # Ohio pharmacy technician immunization minimum age: 5+ years
    ohio_tech_node = evaluator.add_parallel(
        id="Ohio_Technician_Immunization_Minimum_Age",
        desc="Ohio minimum patient age for pharmacy technicians administering immunizations is correctly identified as 5 years old and older (effective 2024), with an official Ohio citation.",
        parent=flu_node,
        critical=True,
    )

    await add_simple_leaf_and_verify(
        evaluator,
        leaf_id="States_Ohio_Technician_Minimum_Age_Equals_5_Years_And_Older",
        desc="States that in Ohio, pharmacy technicians may administer immunizations only to patients aged 5 years and older (per 2024-effective regulation).",
        parent=ohio_tech_node,
        claim="The answer explicitly states that in Ohio, pharmacy technicians may administer immunizations only to patients aged 5 years and older.",
        critical=True,
        add_ins="Accept explicit references to Ohio regulation changes effective in 2024."
    )
    await add_citation_leaf_and_verify(
        evaluator,
        leaf_id="Cites_Official_Ohio_Source",
        desc="Provides at least one official Ohio source URL (e.g., Ohio Board of Pharmacy or Ohio law/regulatory publication) supporting the ≥5 years technician restriction.",
        parent=ohio_tech_node,
        claim="The official Ohio source confirms that pharmacy technicians in Ohio are limited to administering immunizations to patients aged 5 years and older.",
        sources=flu_sources.ohio_tech_min_age_urls,
        critical=True,
        add_ins="Treat a source as official if the domain is ohio.gov, pharmacy.ohio.gov, bop.ohio.gov, codes.ohio.gov, or another recognized official Ohio government site."
    )

    # Walgreens provider types: pharmacists, pharmacy interns, trained technicians
    provider_types_node = evaluator.add_parallel(
        id="Walgreens_Flu_Shots_Administered_By_Provider_Types",
        desc="States the Walgreens provider types involved in flu-shot administration (pharmacists, pharmacy interns, trained technicians) with an official Walgreens citation.",
        parent=flu_node,
        critical=True,
    )

    await add_simple_leaf_and_verify(
        evaluator,
        leaf_id="States_Provider_Types_Pharmacist_Intern_Technician",
        desc="States that Walgreens flu shots are administered by licensed pharmacists, pharmacy interns, and trained technicians.",
        parent=provider_types_node,
        claim="The answer explicitly states that Walgreens flu shots are administered by licensed pharmacists, pharmacy interns, and trained technicians.",
        critical=True,
        add_ins="Accept reasonable variants such as 'pharmacy interns' or 'intern pharmacists'."
    )
    await add_citation_leaf_and_verify(
        evaluator,
        leaf_id="Cites_Official_Walgreens_Source_For_Provider_Types",
        desc="Provides at least one official Walgreens URL supporting the listed provider types.",
        parent=provider_types_node,
        claim="A Walgreens official page states that flu shots are administered by pharmacists, pharmacy interns, and trained technicians.",
        sources=flu_sources.provider_types_urls,
        critical=True,
        add_ins="Consider only walgreens.com (and Walgreens-owned subdomains) as official."
    )

    # All Walgreens retail locations provide flu shots
    all_locations_node = evaluator.add_parallel(
        id="All_Walgreens_Retail_Locations_Provide_Flu_Shots",
        desc="States that all Walgreens retail locations provide flu shot services, with an official Walgreens citation.",
        parent=flu_node,
        critical=True,
    )

    await add_simple_leaf_and_verify(
        evaluator,
        leaf_id="States_All_Retail_Locations_Provide_Flu_Shot_Service",
        desc="States that all Walgreens retail locations provide flu shot services.",
        parent=all_locations_node,
        claim="The answer explicitly states that all Walgreens retail locations provide flu shot services.",
        critical=True,
        add_ins="Be strict: the statement must claim 'all locations' or an equivalent universal quantifier."
    )
    await add_citation_leaf_and_verify(
        evaluator,
        leaf_id="Cites_Official_Walgreens_Source_For_All_Locations_Claim",
        desc="Provides at least one official Walgreens URL supporting the claim that all retail locations provide flu shot services.",
        parent=all_locations_node,
        claim="A Walgreens official page supports the claim that all Walgreens retail locations provide flu shot services.",
        sources=flu_sources.all_locations_urls,
        critical=True,
        add_ins="If the Walgreens page does not explicitly indicate 'all locations', treat as not supported."
    )

    # Authorized providers for ages 3–4 in Ohio Walgreens
    authorized_node = evaluator.add_parallel(
        id="Authorized_Providers_For_Ages_3_to_4_in_Ohio_Walgreens",
        desc="Correctly concludes who may administer flu shots to ages 3–4 at Walgreens in Ohio, based on Walgreens minimum age and Ohio technician restriction, with official citations.",
        parent=flu_node,
        critical=True,
    )

    await add_simple_leaf_and_verify(
        evaluator,
        leaf_id="States_Only_Pharmacists_Or_Interns_For_Ages_3_to_4",
        desc="States that for ages 3–4 at Ohio Walgreens locations, pharmacists and pharmacy interns are authorized, and pharmacy technicians are not authorized.",
        parent=authorized_node,
        claim="The answer explicitly states that at Ohio Walgreens locations, children aged 3–4 can receive flu shots from pharmacists or pharmacy interns, but not from pharmacy technicians.",
        critical=True,
        add_ins="Accept clear equivalents indicating technicians are not permitted for ages 3–4 while pharmacists/interns are."
    )
    await add_simple_leaf_and_verify(
        evaluator,
        leaf_id="Explains_Reasoning_Tying_To_Both_Rules",
        desc="Explains that the conclusion follows from (i) Walgreens flu shots allowed starting at age 3 and (ii) Ohio technicians limited to administering immunizations starting at age 5.",
        parent=authorized_node,
        claim="The answer explicitly explains the reasoning: Walgreens allows flu shots starting at age 3, and Ohio restricts technicians to age 5+, therefore ages 3–4 must be served by pharmacists or pharmacy interns.",
        critical=True,
        add_ins="Ensure both rules are referenced and logically connected."
    )
    combined_conclusion_sources: List[str] = list(set(flu_sources.conclusion_urls + flu_sources.walgreens_min_age_urls + flu_sources.ohio_tech_min_age_urls))
    await add_citation_leaf_and_verify(
        evaluator,
        leaf_id="Cites_Official_Sources_For_Conclusion",
        desc="Provides official-source URL citation(s) supporting the conclusion (can reuse the official Walgreens and official Ohio citations).",
        parent=authorized_node,
        claim="The official sources collectively confirm the underlying policies used in the conclusion: Walgreens flu shots available for ages ≥3 and Ohio technicians limited to ages ≥5 for immunizations.",
        sources=combined_conclusion_sources,
        critical=True,
        add_ins="Treat a citation as official only if it is an official Walgreens page or an official Ohio government source. The claim is supported if the provided sources confirm the two underlying policies."
    )

    # Columbus OH Walgreens locations offer immunizations (non-critical)
    columbus_node = evaluator.add_parallel(
        id="Columbus_OH_Walgreens_Locations_Offer_Immunizations",
        desc="States that multiple Walgreens locations in Columbus, Ohio offer immunization services, with supporting citation(s).",
        parent=flu_node,
        critical=False,
    )

    await add_simple_leaf_and_verify(
        evaluator,
        leaf_id="States_Multiple_Columbus_Locations_Offer_Immunizations",
        desc="States that multiple Walgreens locations in Columbus, Ohio offer immunization services.",
        parent=columbus_node,
        claim="The answer explicitly states that there are multiple Walgreens locations in Columbus, Ohio that offer immunization services.",
        critical=False,
        add_ins="Accept clear paraphrases indicating that many/multiple Columbus Walgreens offer immunizations."
    )
    await add_citation_leaf_and_verify(
        evaluator,
        leaf_id="Cites_Source_For_Columbus_Locations_Claim",
        desc="Provides a supporting URL (preferably Walgreens store/immunization locator pages) for the Columbus-locations claim.",
        parent=columbus_node,
        claim="The provided locator/store pages indicate Walgreens locations in Columbus, Ohio offer immunization services.",
        sources=flu_sources.columbus_locations_urls,
        critical=False,
        add_ins="Prefer Walgreens store or immunization locator pages; however, any cited source explicitly listing Walgreens Columbus immunization availability is acceptable."
    )


# --------------------------------------------------------------------------- #
# GLP-1 weight loss medication eligibility verification                       #
# --------------------------------------------------------------------------- #
async def verify_glp1_section(
    evaluator: Evaluator,
    parent_node,
    glp1: GLP1Extraction,
) -> None:
    glp1_node = evaluator.add_parallel(
        id="GLP1_Weight_Loss_Medication_Eligibility",
        desc="Correctly states the BMI-based eligibility pathways, provides qualifying comorbidity examples, confirms Wegovy’s FDA weight-loss approval, and identifies the manufacturer, each with official citations.",
        parent=parent_node,
        critical=True,
    )

    # Primary BMI threshold: BMI ≥ 30
    primary_node = evaluator.add_parallel(
        id="Primary_Eligibility_BMI_Threshold",
        desc="Primary BMI threshold is correctly stated as BMI ≥ 30, with an official citation.",
        parent=glp1_node,
        critical=True,
    )

    await add_simple_leaf_and_verify(
        evaluator,
        leaf_id="States_Primary_BMI_Threshold_Equals_30_or_Higher",
        desc="States that primary eligibility is BMI ≥ 30.",
        parent=primary_node,
        claim="The answer explicitly states that the primary eligibility threshold for GLP-1 weight-loss medications (e.g., Wegovy) is BMI ≥ 30.",
        critical=True,
        add_ins="Accept variants like 'BMI 30 or higher' or 'BMI of at least 30'."
    )
    await add_citation_leaf_and_verify(
        evaluator,
        leaf_id="Cites_Official_Source_For_Primary_Threshold",
        desc="Provides an official-source URL (FDA labeling and/or manufacturer prescribing information) supporting BMI ≥ 30.",
        parent=primary_node,
        claim="The official labeling/manufacturer prescribing information confirms that BMI ≥ 30 is an eligibility threshold for Wegovy's weight management indication.",
        sources=glp1.primary_bmi_urls,
        critical=True,
        add_ins="Treat a source as official if the domain is fda.gov or novonordisk.com (or an official manufacturer domain hosting prescribing information)."
    )

    # Alternative pathway: BMI ≥ 27 + at least one comorbidity
    alt_node = evaluator.add_parallel(
        id="Alternative_Eligibility_Pathway",
        desc="Alternative eligibility pathway is correctly stated as BMI ≥ 27 plus at least one weight-related comorbidity, with an official citation.",
        parent=glp1_node,
        critical=True,
    )

    await add_simple_leaf_and_verify(
        evaluator,
        leaf_id="States_Alternative_Pathway_BMI_27_plus_Comorbidity",
        desc="States that alternative eligibility is BMI ≥ 27 with at least one qualifying weight-related comorbidity.",
        parent=alt_node,
        claim="The answer explicitly states that the alternative eligibility pathway is BMI ≥ 27 with at least one qualifying weight-related comorbidity.",
        critical=True,
        add_ins="Accept phrasings like 'BMI 27 or higher with a weight-related comorbidity'."
    )
    await add_citation_leaf_and_verify(
        evaluator,
        leaf_id="Cites_Official_Source_For_Alternative_Pathway",
        desc="Provides an official-source URL (FDA labeling and/or manufacturer prescribing information) supporting BMI ≥ 27 + comorbidity.",
        parent=alt_node,
        claim="The official labeling/manufacturer prescribing information confirms that BMI ≥ 27 with at least one weight-related comorbidity is an eligibility pathway for Wegovy.",
        sources=glp1.alt_path_urls,
        critical=True,
        add_ins="Treat a source as official if it is fda.gov or novonordisk.com prescribing information."
    )

    # Comorbidity examples
    comorb_node = evaluator.add_parallel(
        id="Qualifying_Comorbidity_Examples",
        desc="Provides comorbidity examples consistent with the constraint list, with an official citation.",
        parent=glp1_node,
        critical=True,
    )

    # Verify the answer lists at least two from the specified list
    await add_simple_leaf_and_verify(
        evaluator,
        leaf_id="Includes_At_Least_Two_From_Specified_List",
        desc="Lists at least two qualifying comorbidities, drawn from: high blood pressure, diabetes, high cholesterol, obstructive sleep apnea.",
        parent=comorb_node,
        claim=(
            "The answer lists at least two qualifying weight-related comorbidities among the following set: "
            "hypertension (high blood pressure), diabetes, high cholesterol (hyperlipidemia/dyslipidemia), obstructive sleep apnea."
        ),
        critical=True,
        add_ins="Count explicit mentions in the answer. Accept common synonyms like 'hypertension' for 'high blood pressure' and 'hyperlipidemia/dyslipidemia' for 'high cholesterol'."
    )

    # Build a claim using up to two examples from the extracted list for source verification
    examples_for_claim = glp1.comorbidity_examples[:2]
    if examples_for_claim:
        claim_examples_text = ", ".join(examples_for_claim)
    else:
        claim_examples_text = "two qualifying comorbidities (e.g., hypertension and diabetes)"

    await add_citation_leaf_and_verify(
        evaluator,
        leaf_id="Cites_Official_Source_For_Comorbidity_Examples",
        desc="Provides an official-source URL (FDA labeling and/or manufacturer prescribing information) supporting that the listed conditions qualify.",
        parent=comorb_node,
        claim=(
            f"Official labeling/prescribing information supports that the following conditions qualify as weight-related comorbidities "
            f"under the BMI ≥ 27 pathway: {claim_examples_text}."
        ),
        sources=glp1.comorbidity_urls,
        critical=True,
        add_ins="Treat FDA (fda.gov) and official manufacturer prescribing information (novonordisk.com) as official sources."
    )

    # Wegovy FDA approved for weight loss
    wegovy_approval_node = evaluator.add_parallel(
        id="Wegovy_FDA_Approved_For_Weight_Loss",
        desc="States that Wegovy (semaglutide) is FDA-approved specifically for weight loss management, with an official citation.",
        parent=glp1_node,
        critical=True,
    )

    await add_simple_leaf_and_verify(
        evaluator,
        leaf_id="States_Wegovy_Is_FDA_Approved_For_Weight_Loss",
        desc="States that Wegovy is FDA-approved specifically for weight loss management.",
        parent=wegovy_approval_node,
        claim="The answer explicitly states that Wegovy (semaglutide) is FDA-approved for weight loss management (adult chronic weight management).",
        critical=True,
        add_ins="Accept 'adult chronic weight management' as equivalent phrasing."
    )
    await add_citation_leaf_and_verify(
        evaluator,
        leaf_id="Cites_Official_Source_For_FDA_Approval",
        desc="Provides an official-source URL (FDA label/Drugs@FDA and/or manufacturer prescribing information) supporting the approval/indication.",
        parent=wegovy_approval_node,
        claim="FDA or manufacturer materials explicitly indicate that Wegovy is approved for adult chronic weight management (weight loss).",
        sources=glp1.wegovy_approval_urls,
        critical=True,
        add_ins="Treat FDA (fda.gov) and official manufacturer labeling/prescribing info (novonordisk.com) as official."
    )

    # Wegovy manufacturer
    wegovy_manu_node = evaluator.add_parallel(
        id="Wegovy_Manufacturer",
        desc="Correctly identifies Wegovy’s manufacturer as Novo Nordisk, with an official citation.",
        parent=glp1_node,
        critical=True,
    )

    await add_simple_leaf_and_verify(
        evaluator,
        leaf_id="States_Manufacturer_Equals_Novo_Nordisk",
        desc="States that Novo Nordisk manufactures Wegovy.",
        parent=wegovy_manu_node,
        claim="The answer explicitly states that Wegovy is manufactured by Novo Nordisk.",
        critical=True,
        add_ins="Accept 'Novo Nordisk A/S' or 'Novo Nordisk' as manufacturer naming."
    )
    await add_citation_leaf_and_verify(
        evaluator,
        leaf_id="Cites_Official_Source_For_Manufacturer",
        desc="Provides at least one official-source URL (manufacturer site or FDA source) supporting Novo Nordisk as the manufacturer.",
        parent=wegovy_manu_node,
        claim="Official manufacturer site or FDA sources indicate that Novo Nordisk manufactures Wegovy.",
        sources=glp1.wegovy_manufacturer_urls,
        critical=True,
        add_ins="Treat novonordisk.com or fda.gov as official for confirming manufacturer."
    )

    # Ozempic manufacturer (non-critical)
    ozempic_node = evaluator.add_parallel(
        id="Ozempic_Also_Manufactured_By_Novo_Nordisk",
        desc="States that Novo Nordisk also manufactures Ozempic (as given in constraints), with an official citation.",
        parent=glp1_node,
        critical=False,
    )

    await add_simple_leaf_and_verify(
        evaluator,
        leaf_id="States_Novo_Nordisk_Manufactures_Ozempic",
        desc="States that Novo Nordisk manufactures Ozempic.",
        parent=ozempic_node,
        claim="The answer explicitly states that Novo Nordisk manufactures Ozempic.",
        critical=False,
        add_ins="Accept 'Novo Nordisk' as the manufacturer designation."
    )
    await add_citation_leaf_and_verify(
        evaluator,
        leaf_id="Cites_Official_Source_For_Ozempic_Manufacturer",
        desc="Provides an official-source URL supporting Novo Nordisk as Ozempic’s manufacturer.",
        parent=ozempic_node,
        claim="Official manufacturer site or FDA sources indicate that Novo Nordisk manufactures Ozempic.",
        sources=glp1.ozempic_manufacturer_urls,
        critical=False,
        add_ins="Treat novonordisk.com or fda.gov as official."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the answer for Walgreens/OH flu-shot administration rules and GLP-1 eligibility,
    using official-source URL verification where applicable.
    """
    # Initialize evaluator (root is non-critical by design; add a critical top-level node under it)
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

    top_node = evaluator.add_parallel(
        id="Healthcare_Service_Research",
        desc="Report addresses (1) Walgreens/OH flu-shot administration rules relevant to Ohio Walgreens locations and (2) eligibility criteria for FDA-approved GLP-1 weight-loss medications (including Wegovy), with official-source URL citations supporting each required claim.",
        parent=root,
        critical=True,
    )

    # Extract sources in parallel
    flu_sources_task = evaluator.extract(
        prompt=prompt_extract_flu_sources(),
        template_class=FluSources,
        extraction_name="flu_sources",
    )
    glp1_data_task = evaluator.extract(
        prompt=prompt_extract_glp1_sources_and_comorbidities(),
        template_class=GLP1Extraction,
        extraction_name="glp1_sources_and_comorbidities",
    )
    flu_sources_res, glp1_data_res = await asyncio.gather(flu_sources_task, glp1_data_task)

    # Build and verify the flu-shot section
    await verify_flu_section(evaluator, top_node, flu_sources_res)

    # Build and verify the GLP-1 section
    await verify_glp1_section(evaluator, top_node, glp1_data_res)

    # Return structured summary
    return evaluator.get_summary()