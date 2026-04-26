import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "therapeutic_candidate_pgdh_oa_2025"
TASK_DESCRIPTION = """Identify a therapeutic candidate that meets all of the following criteria:

1. Disease Target: The therapeutic must target a disease affecting at least 400 million people aged 55 years and older globally, primarily impacting middle-aged and older adults.

2. Molecular Mechanism: The therapeutic must work by inhibiting a specific enzyme called 15-hydroxy prostaglandin dehydrogenase (15-PGDH) or 15-PGDH, which is classified as a gerozyme (an age-related enzyme that increases in expression with aging in the target tissue). This enzyme must degrade prostaglandin E2 (PGE2), and the therapeutic effect must involve maintaining PGE2 at normal biological levels to support regeneration. The mechanism must shift existing chondrocyte gene expression patterns toward a more youthful state without requiring stem cell or progenitor cell proliferation.

3. Preclinical Evidence: The therapeutic must have demonstrated articular cartilage regeneration in preclinical mouse models using local injections administered twice weekly for four weeks. It must also have prevented the development of osteoarthritis after knee injuries mimicking ACL tears in these animal models. Additionally, human cartilage tissue obtained from joint replacement surgeries must have shown positive regenerative responses when treated ex vivo.

4. Clinical Development Status: An oral formulation of the same inhibitor class must have already completed Phase 1 clinical trials for a related indication (such as muscle weakness) and been reported as safe and active in healthy volunteers. For the osteoarthritis/cartilage regeneration indication specifically, the therapeutic must be in the preclinical-to-early-clinical development stage, with researchers expecting clinical trials for this indication to begin soon.

5. Publication and Institution: The key research demonstrating these findings must have been published in a peer-reviewed scientific journal in 2025 and must have been conducted at or led by Stanford Medicine or Stanford University.

Provide the name of this therapeutic approach, the specific disease it targets, and supporting URL references that verify each of the above criteria.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TherapeuticExtraction(BaseModel):
    """Structured extraction of the therapeutic candidate and per-criterion URLs."""
    therapeutic_name: Optional[str] = None
    target_disease: Optional[str] = None

    # Per-criterion supporting URLs as provided by the answer
    disease_burden_urls: List[str] = Field(default_factory=list)
    mechanism_urls: List[str] = Field(default_factory=list)
    preclinical_urls: List[str] = Field(default_factory=list)
    clinical_urls: List[str] = Field(default_factory=list)
    institutional_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_therapeutic_info() -> str:
    return """
    Extract from the answer the following structured information about the therapeutic candidate:

    1) therapeutic_name: The specific name or description of the therapeutic approach (e.g., the inhibitor name, class, or approach descriptor).
    2) target_disease: The specific disease the therapeutic targets (e.g., "osteoarthritis").
    3) disease_burden_urls: A list of URLs that support the global disease burden claims for this disease (e.g., prevalence ≥400 million aged 55+ and primarily affecting middle-aged/older adults).
    4) mechanism_urls: A list of URLs (ideally peer‑reviewed articles) that support the mechanism claims: inhibition of 15-hydroxy prostaglandin dehydrogenase (15-PGDH), classification as a "gerozyme" with age‑increased expression in articular cartilage, that 15-PGDH degrades PGE2, that maintaining PGE2 at normal biological levels supports tissue regeneration, that the approach shifts existing chondrocyte gene expression toward a youthful state, and that regeneration does not require stem/progenitor cell proliferation.
    5) preclinical_urls: A list of URLs (peer‑reviewed or authoritative preclinical reports) that document articular cartilage regeneration in mice with local injections twice weekly for four weeks, prevention/reduction of osteoarthritis development after ACL‑type knee injuries, and positive ex vivo responses in human osteoarthritic cartilage from joint replacement surgeries.
    6) clinical_urls: A list of URLs that document the clinical development status for related indications (e.g., muscle weakness) including Phase 1 completion and safety/activity in healthy volunteers, and that the osteoarthritis/cartilage regeneration indication is in preclinical‑to‑early‑clinical status with clinical trials expected soon; also include a URL confirming that the primary research study was published in 2025 in a peer‑reviewed journal.
    7) institutional_urls: A list of URLs that document that the research was conducted at or led by Stanford Medicine or Stanford University (e.g., journal affiliation lines, institutional press releases, or lab pages referencing the study).

    Rules:
    - Return only URLs explicitly present in the answer. Do not invent or infer URLs.
    - Include full, valid URLs. If a URL lacks protocol, prepend http://.
    - If a requested field is missing, set it to null (for single values) or return an empty list (for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root,
    ext: TherapeuticExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    All nodes under the main aggregator are critical; leaf checks use source-grounded verification where applicable.
    """

    # Main critical aggregator node (represents the overall task outcome)
    main_node = evaluator.add_parallel(
        id="Therapeutic_Candidate_Identification",
        desc="Identify a therapeutic candidate meeting all disease burden, mechanism, efficacy, and development criteria for osteoarthritis treatment, and provide its name, target disease, and supporting references",
        parent=root,
        critical=True
    )

    # 1) Name and Disease Provided (existence checks)
    name_node = evaluator.add_custom_node(
        result=_non_empty_str(ext.therapeutic_name),
        id="Therapeutic_Name_Provided",
        desc="Provide the specific name or description of the therapeutic approach",
        parent=main_node,
        critical=True
    )

    disease_node_basic = evaluator.add_custom_node(
        result=_non_empty_str(ext.target_disease),
        id="Target_Disease_Identified",
        desc="Provide the specific disease name that the therapeutic targets",
        parent=main_node,
        critical=True
    )

    # 2) Disease Target Verification
    disease_verif_node = evaluator.add_parallel(
        id="Disease_Target_Verification",
        desc="Verify the therapeutic targets a disease with substantial global burden affecting middle-aged and older adults",
        parent=main_node,
        critical=True
    )

    # 2.1 Global burden threshold
    global_burden_leaf = evaluator.add_leaf(
        id="Global_Burden_Threshold",
        desc="Disease affects at least 400 million people aged 55+ worldwide",
        parent=disease_verif_node,
        critical=True
    )
    burden_claim = (
        f"The disease '{ext.target_disease or 'the referenced disease'}' affects at least 400 million people "
        f"aged 55 years and older worldwide."
    )
    await evaluator.verify(
        claim=burden_claim,
        node=global_burden_leaf,
        sources=ext.disease_burden_urls,
        additional_instruction="Verify that the provided sources explicitly support a global burden ≥ 400 million among people aged 55+."
    )

    # 2.2 Age group primarily affected
    age_group_leaf = evaluator.add_leaf(
        id="Age_Group_Affected",
        desc="Disease primarily affects middle-aged and older adult populations",
        parent=disease_verif_node,
        critical=True
    )
    age_claim = (
        f"The disease '{ext.target_disease or 'the referenced disease'}' primarily affects middle-aged and older adults."
    )
    await evaluator.verify(
        claim=age_claim,
        node=age_group_leaf,
        sources=ext.disease_burden_urls,
        additional_instruction="Assess whether the disease burden described in the sources primarily concerns middle-aged and older adults."
    )

    # 2.3 Disease burden reference URL existence
    disease_ref_leaf = evaluator.add_custom_node(
        result=len(ext.disease_burden_urls) > 0,
        id="Disease_Burden_Reference_URL",
        desc="Provide authoritative source documenting global disease burden statistics",
        parent=disease_verif_node,
        critical=True
    )

    # 3) Mechanism Verification
    mechanism_node = evaluator.add_parallel(
        id="Mechanism_Verification",
        desc="Verify the therapeutic mechanism involves inhibition of a specific age-related enzyme affecting prostaglandin metabolism",
        parent=main_node,
        critical=True
    )

    # 3.1 Enzyme name 15-PGDH
    enzyme_name_leaf = evaluator.add_leaf(
        id="Enzyme_Name_15PGDH",
        desc="Enzyme is specifically named as 15-PGDH or 15-hydroxy prostaglandin dehydrogenase",
        parent=mechanism_node,
        critical=True
    )
    enzyme_name_claim = "The therapeutic approach works by inhibiting the enzyme 15-hydroxy prostaglandin dehydrogenase (15-PGDH)."
    await evaluator.verify(
        claim=enzyme_name_claim,
        node=enzyme_name_leaf,
        sources=ext.mechanism_urls,
        additional_instruction="Check that the sources explicitly mention inhibition of 15-PGDH (15-hydroxy prostaglandin dehydrogenase)."
    )

    # 3.2 Gerozyme classification (age-increased expression in articular cartilage)
    gerozyme_leaf = evaluator.add_leaf(
        id="Gerozyme_Classification",
        desc="Enzyme is classified as a gerozyme with expression that increases with aging in articular cartilage",
        parent=mechanism_node,
        critical=True
    )
    gero_claim = "15-PGDH is classified as a 'gerozyme' whose expression increases with aging in articular cartilage."
    await evaluator.verify(
        claim=gero_claim,
        node=gerozyme_leaf,
        sources=ext.mechanism_urls,
        additional_instruction="Verify the source explicitly classifies 15-PGDH as a 'gerozyme' and links age-increased expression in articular cartilage."
    )

    # 3.3 PGE2 degradation by 15-PGDH
    pge2_deg_leaf = evaluator.add_leaf(
        id="PGE2_Degradation",
        desc="Target enzyme degrades prostaglandin E2 (PGE2)",
        parent=mechanism_node,
        critical=True
    )
    pge2_deg_claim = "15-PGDH degrades prostaglandin E2 (PGE2)."
    await evaluator.verify(
        claim=pge2_deg_claim,
        node=pge2_deg_leaf,
        sources=ext.mechanism_urls,
        additional_instruction="Confirm that the sources state 15-PGDH degrades PGE2."
    )

    # 3.4 PGE2 at normal levels supports regeneration
    pge2_regen_leaf = evaluator.add_leaf(
        id="PGE2_Regenerative_Property",
        desc="PGE2 at normal biological levels supports tissue regeneration",
        parent=mechanism_node,
        critical=True
    )
    pge2_regen_claim = "Maintaining PGE2 at normal biological levels supports tissue regeneration."
    await evaluator.verify(
        claim=pge2_regen_claim,
        node=pge2_regen_leaf,
        sources=ext.mechanism_urls,
        additional_instruction="Check whether the sources state that normal physiological levels of PGE2 support tissue regeneration."
    )

    # 3.5 Chondrocyte gene expression shifts toward youthful
    chondro_shift_leaf = evaluator.add_leaf(
        id="Chondrocyte_Gene_Expression_Shift",
        desc="Mechanism shifts existing chondrocyte gene expression toward youthful state",
        parent=mechanism_node,
        critical=True
    )
    chondro_shift_claim = "The therapeutic shifts existing chondrocyte gene expression patterns toward a more youthful state."
    await evaluator.verify(
        claim=chondro_shift_claim,
        node=chondro_shift_leaf,
        sources=ext.mechanism_urls,
        additional_instruction="Verify that sources report youthful reprogramming of chondrocyte gene expression without requiring new cell proliferation."
    )

    # 3.6 No stem/progenitor cell proliferation required
    no_stem_leaf = evaluator.add_leaf(
        id="No_Stem_Cell_Proliferation",
        desc="Regeneration occurs without stem cell or progenitor cell proliferation",
        parent=mechanism_node,
        critical=True
    )
    no_stem_claim = "The regenerative effect occurs without requiring stem cell or progenitor cell proliferation."
    await evaluator.verify(
        claim=no_stem_claim,
        node=no_stem_leaf,
        sources=ext.mechanism_urls,
        additional_instruction="Verify that the sources explicitly indicate regeneration without stem/progenitor cell proliferation."
    )

    # 3.7 Mechanism references existence
    mech_ref_leaf = evaluator.add_custom_node(
        result=len(ext.mechanism_urls) > 0,
        id="Mechanism_Reference_URL",
        desc="Provide research publication documenting the enzyme, PGE2 mechanism, and chondrocyte reprogramming",
        parent=mechanism_node,
        critical=True
    )

    # 4) Preclinical Efficacy
    preclinical_node = evaluator.add_parallel(
        id="Preclinical_Efficacy",
        desc="Verify preclinical evidence from animal models and human tissue demonstrates therapeutic efficacy",
        parent=main_node,
        critical=True
    )

    # 4.1 Mouse cartilage regeneration
    mouse_regen_leaf = evaluator.add_leaf(
        id="Mouse_Cartilage_Regeneration",
        desc="Demonstrated articular cartilage regeneration in aged or injured mice",
        parent=preclinical_node,
        critical=True
    )
    mouse_regen_claim = "Preclinical studies demonstrated articular cartilage regeneration in aged or injured mice."
    await evaluator.verify(
        claim=mouse_regen_claim,
        node=mouse_regen_leaf,
        sources=ext.preclinical_urls,
        additional_instruction="Check the sources for explicit evidence of cartilage regeneration in mouse models."
    )

    # 4.2 Mouse treatment protocol (local injections, twice/week, 4 weeks)
    mouse_protocol_leaf = evaluator.add_leaf(
        id="Mouse_Treatment_Protocol",
        desc="Treatment delivered as local injections, twice per week, for 4-week duration",
        parent=preclinical_node,
        critical=True
    )
    mouse_protocol_claim = "In mouse models, treatment was delivered via local injections twice weekly for four weeks."
    await evaluator.verify(
        claim=mouse_protocol_claim,
        node=mouse_protocol_leaf,
        sources=ext.preclinical_urls,
        additional_instruction="Verify that the preclinical sources explicitly describe local injections administered twice weekly for four weeks."
    )

    # 4.3 Prevention after ACL-type injuries
    post_injury_leaf = evaluator.add_leaf(
        id="Post_Injury_OA_Prevention",
        desc="Treatment prevented or reduced OA development after ACL-type knee injuries in mice",
        parent=preclinical_node,
        critical=True
    )
    post_injury_claim = "The treatment prevented or reduced osteoarthritis development following ACL-type knee injuries in mice."
    await evaluator.verify(
        claim=post_injury_claim,
        node=post_injury_leaf,
        sources=ext.preclinical_urls,
        additional_instruction="Verify that the sources show reduced OA development after knee injuries mimicking ACL tears."
    )

    # 4.4 Human tissue ex vivo response
    human_ex_vivo_leaf = evaluator.add_leaf(
        id="Human_Tissue_Ex_Vivo_Response",
        desc="Human osteoarthritic cartilage from joint replacement surgeries responded positively ex vivo",
        parent=preclinical_node,
        critical=True
    )
    human_ex_vivo_claim = "Human osteoarthritic cartilage tissue from joint replacement surgeries exhibited positive regenerative responses ex vivo when treated."
    await evaluator.verify(
        claim=human_ex_vivo_claim,
        node=human_ex_vivo_leaf,
        sources=ext.preclinical_urls,
        additional_instruction="Check for ex vivo data on human osteoarthritic cartilage showing positive responses to treatment."
    )

    # 4.5 Preclinical references existence
    preclin_ref_leaf = evaluator.add_custom_node(
        result=len(ext.preclinical_urls) > 0,
        id="Preclinical_Evidence_Reference_URL",
        desc="Provide peer-reviewed publication documenting mouse model and human tissue ex vivo study results",
        parent=preclinical_node,
        critical=True
    )

    # 5) Clinical Development Status
    development_node = evaluator.add_parallel(
        id="Development_Stage_Verification",
        desc="Verify the clinical development status including related trials and publication timing",
        parent=main_node,
        critical=True
    )

    # 5.1 Phase 1 oral formulation completed (muscle-related indication)
    phase1_leaf = evaluator.add_leaf(
        id="Phase1_Oral_Formulation_Completed",
        desc="Oral formulation of same inhibitor class completed Phase 1 clinical trial for muscle-related indication",
        parent=development_node,
        critical=True
    )
    phase1_claim = "An oral formulation of the same 15-PGDH inhibitor class has completed a Phase 1 clinical trial for a muscle-related indication (e.g., muscle weakness)."
    await evaluator.verify(
        claim=phase1_claim,
        node=phase1_leaf,
        sources=ext.clinical_urls,
        additional_instruction="Verify Phase 1 completion for a related muscle indication using clinical sources."
    )

    # 5.2 Phase 1 safety and activity in healthy volunteers
    safety_activity_leaf = evaluator.add_leaf(
        id="Phase1_Safety_Activity",
        desc="Phase 1 trial reported treatment as safe and active in healthy volunteers",
        parent=development_node,
        critical=True
    )
    safety_claim = "The Phase 1 trial reported the oral formulation to be safe and active in healthy volunteers."
    await evaluator.verify(
        claim=safety_claim,
        node=safety_activity_leaf,
        sources=ext.clinical_urls,
        additional_instruction="Confirm explicit statements of safety and activity in healthy volunteers from Phase 1 sources."
    )

    # 5.3 OA indication development stage
    oa_stage_leaf = evaluator.add_leaf(
        id="OA_Preclinical_Stage",
        desc="Cartilage regeneration application is in preclinical-to-early-clinical stage with expectation of upcoming trials",
        parent=development_node,
        critical=True
    )
    oa_stage_claim = "For the osteoarthritis/cartilage regeneration indication, development is at the preclinical-to-early-clinical stage, with clinical trials expected to begin soon."
    await evaluator.verify(
        claim=oa_stage_claim,
        node=oa_stage_leaf,
        sources=ext.clinical_urls,
        additional_instruction="Check if sources indicate the OA/cartilage regeneration indication is nearing clinical trials (preclinical-to-early-clinical)."
    )

    # 5.4 Publication year 2025 in peer-reviewed journal
    pub_2025_leaf = evaluator.add_leaf(
        id="Publication_Year_2025",
        desc="Primary research study published in 2025 in peer-reviewed scientific journal",
        parent=development_node,
        critical=True
    )
    pub_2025_claim = "The primary research study was published in 2025 in a peer-reviewed scientific journal."
    await evaluator.verify(
        claim=pub_2025_claim,
        node=pub_2025_leaf,
        sources=ext.clinical_urls,
        additional_instruction="Use the provided sources to confirm the 2025 publication date and peer-reviewed journal status."
    )

    # 5.5 Clinical references existence
    clin_ref_leaf = evaluator.add_custom_node(
        result=len(ext.clinical_urls) > 0,
        id="Clinical_Development_Reference_URL",
        desc="Provide sources documenting Phase 1 trial completion, safety/activity results, and 2025 publication date",
        parent=development_node,
        critical=True
    )

    # 6) Institutional Origin: Stanford
    inst_node = evaluator.add_parallel(
        id="Institutional_Origin",
        desc="Verify the research was conducted at Stanford Medicine or Stanford University",
        parent=main_node,
        critical=True
    )

    stanford_leaf = evaluator.add_leaf(
        id="Stanford_Institution",
        desc="Research institution is Stanford Medicine or Stanford University",
        parent=inst_node,
        critical=True
    )
    stanford_claim = "The research was conducted at or led by Stanford Medicine or Stanford University."
    await evaluator.verify(
        claim=stanford_claim,
        node=stanford_leaf,
        sources=ext.institutional_urls,
        additional_instruction="Verify explicit institutional affiliation (Stanford Medicine or Stanford University) in the publication or official sources."
    )

    inst_ref_leaf = evaluator.add_custom_node(
        result=len(ext.institutional_urls) > 0,
        id="Institutional_Reference_URL",
        desc="Provide publication or news source documenting Stanford institutional affiliation",
        parent=inst_node,
        critical=True
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
    Evaluate an answer for the therapeutic candidate identification task.
    Returns the standard evaluation summary dictionary from the evaluator.
    """
    # Initialize evaluator (root is non-critical by design; we add a critical sub-root)
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_therapeutic_info(),
        template_class=TherapeuticExtraction,
        extraction_name="therapeutic_candidate_extraction"
    )

    # Optionally record ground-truth-like expectations (not true ground truth; for context)
    evaluator.add_ground_truth({
        "required_criteria": [
            "Global burden ≥ 400M aged 55+ and primarily affects older adults",
            "Mechanism: inhibition of 15-PGDH (gerozyme), reduces PGE2 degradation; maintain PGE2 at normal levels; youthful chondrocyte gene expression; no stem/progenitor proliferation",
            "Preclinical mouse: local injections twice/week for 4 weeks; OA prevention post ACL-like injuries; human OA cartilage ex vivo positive",
            "Clinical dev: oral same inhibitor class completed Phase 1 for muscle-related indication; safe and active; OA indication preclinical-to-early-clinical; publication year 2025",
            "Institution: Stanford Medicine or Stanford University"
        ]
    }, gt_type="criteria_requirements")

    # Build verification tree and run verifications
    await build_verification_tree(evaluator, root, extraction)

    # Return structured result
    return evaluator.get_summary()