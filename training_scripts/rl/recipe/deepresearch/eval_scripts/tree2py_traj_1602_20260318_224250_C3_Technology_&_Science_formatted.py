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
TASK_ID = "gh200_seismic_japan"
TASK_DESCRIPTION = """
Identify a research institute affiliated with a university in Japan that has deployed NVIDIA GH200 Grace Hopper Superchips specifically for seismic simulation or earthquake-related computational research. The deployment must meet all of the following criteria:

1. When scaled to a large multi-node supercomputer, the deployment achieved at least 50x speedup compared to a CPU-only baseline.
2. The deployment demonstrated at least 90% parallel efficiency when scaled to over 1,000 compute nodes.
3. The deployment reported at least 30x energy efficiency improvement compared to traditional CPU-only methods.
4. The research findings were presented at the WACCPD 2024 workshop.
5. The research has an associated ArXiv preprint publication.
6. The implementation utilizes the NVLink-C2C chip-to-chip interconnect feature of the GH200.
7. The deployment demonstrates a heterogeneous CPU-GPU computing approach leveraging both the Grace CPU and H100 GPU components.

For the identified institution, provide:
- The full name of the research institute
- The name of the affiliated university
- The specific speedup achieved (compared to CPU-only baseline) when scaled to the large supercomputer
- The exact parallel efficiency percentage achieved at scale
- The specific energy efficiency improvement factor achieved
- The ArXiv paper URL documenting this research
- A URL to an official NVIDIA blog post, developer page, or technical article that describes this specific GH200 deployment
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GH200SeismicExtraction(BaseModel):
    institute_name: Optional[str] = None
    university_name: Optional[str] = None
    research_focus: Optional[str] = None

    speedup_scaled: Optional[str] = None           # e.g., "52x", "≥50×"
    parallel_efficiency: Optional[str] = None      # e.g., "92%", "90.5%"
    energy_efficiency: Optional[str] = None        # e.g., "31x", ">=30x"

    nodes_scale: Optional[str] = None              # e.g., "over 1,000 nodes", "1024 nodes"

    arxiv_url: Optional[str] = None                # must be arxiv.org
    nvidia_url: Optional[str] = None               # must be nvidia.com
    metrics_urls: List[str] = Field(default_factory=list)     # pages that document metrics
    additional_urls: List[str] = Field(default_factory=list)  # other cited sources (institute/WACCPD pages, etc.)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_research_institution() -> str:
    return """
    You will extract structured information from the provided answer about a Japanese university-affiliated research institute that used NVIDIA GH200 for seismic/earthquake simulation research.

    Extract the following fields exactly as stated in the answer (use strings for numbers, keep units/suffixes like 'x' or '%'):
    - institute_name: Full name of the research institute or center
    - university_name: Name of the affiliated university
    - research_focus: Short phrase describing the GH200 deployment focus (should mention seismic or earthquake simulation)
    - speedup_scaled: Specific speedup figure compared to a CPU-only baseline when scaled on a large multi-node supercomputer (e.g., "52x")
    - parallel_efficiency: Exact parallel efficiency percentage achieved at scale (e.g., "92%")
    - energy_efficiency: Specific energy efficiency improvement factor over CPU-only (e.g., "31x")
    - nodes_scale: The reported node count context for scaling (e.g., "over 1,000 nodes", "1024 nodes")
    - arxiv_url: URL to the associated arXiv preprint documenting this GH200 seismic research (must be from arxiv.org)
    - nvidia_url: URL to an official NVIDIA blog/developer/technical article describing this specific GH200 deployment (must be from an nvidia.com domain)
    - metrics_urls: Array of URLs that document performance metrics (speedup, parallel efficiency, energy efficiency). Include any cited URLs that present these metrics (can include arXiv/NVIDIA pages if they document metrics; also include institute or supercomputer pages if applicable).
    - additional_urls: Array of any other cited URLs relevant to this research (e.g., institute homepage, lab page, WACCPD 2024 workshop page, system documentation, news releases).

    Rules:
    - If an item is missing in the answer, return null for single fields and [] for arrays.
    - Do not fabricate URLs. Only include URLs explicitly present in the answer.
    - Preserve units and symbols exactly as written (e.g., keep "x" for speedup, "%" for efficiency).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        u_s = u.strip()
        if not u_s:
            continue
        if u_s not in seen:
            seen.add(u_s)
            out.append(u_s)
    return out


def collect_all_sources(extracted: GH200SeismicExtraction) -> List[str]:
    urls: List[str] = []
    if extracted.arxiv_url:
        urls.append(extracted.arxiv_url)
    if extracted.nvidia_url:
        urls.append(extracted.nvidia_url)
    urls.extend(extracted.metrics_urls or [])
    urls.extend(extracted.additional_urls or [])
    return _dedupe_preserve_order(urls)


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: GH200SeismicExtraction) -> None:
    # Top-level critical sequential node (as per rubric root)
    root_node = evaluator.add_sequential(
        id="Research_Institution_Identification",
        desc="Identifies a Japanese university-affiliated research institute that deployed NVIDIA GH200 Grace Hopper Superchips for seismic/earthquake computational research",
        critical=True
    )

    # ----------------------- Institution Basic Criteria -----------------------
    basic_node = evaluator.add_parallel(
        id="Institution_Basic_Criteria",
        desc="Verifies the institution is a research institute affiliated with a Japanese university and focuses on earthquake/seismic research using GH200",
        parent=root_node,
        critical=True
    )

    # Existence/gating for basic info (critical)
    evaluator.add_custom_node(
        result=bool(extracted.institute_name and extracted.university_name),
        id="Basic_Info_Provided",
        desc="Institute and university names are provided",
        parent=basic_node,
        critical=True
    )

    # Japanese University Affiliation (leaf)
    jap_aff_node = evaluator.add_leaf(
        id="Japanese_University_Affiliation",
        desc="The identified institution is a research institute affiliated with a university located in Japan",
        parent=basic_node,
        critical=True
    )
    claim_aff = (
        f"The institution '{extracted.institute_name}' is a research institute affiliated with "
        f"'{extracted.university_name}', which is a university located in Japan."
    )
    await evaluator.verify(
        claim=claim_aff,
        node=jap_aff_node,
        sources=collect_all_sources(extracted),
        additional_instruction="Confirm both the affiliation (institute <-> university) and that the university is in Japan, based on the provided webpages."
    )

    # Seismic Research Focus (leaf)
    seismic_node = evaluator.add_leaf(
        id="Seismic_Research_Focus",
        desc="The institution's GH200 deployment specifically focuses on seismic simulation or earthquake-related computational research",
        parent=basic_node,
        critical=True
    )
    claim_focus = (
        f"The GH200 deployment at '{extracted.institute_name}' specifically focuses on seismic simulation "
        f"or earthquake-related computational research."
    )
    await evaluator.verify(
        claim=claim_focus,
        node=seismic_node,
        sources=collect_all_sources(extracted),
        additional_instruction="Look for terms like 'seismic', 'earthquake', 'seismic simulation', 'earthquake simulation', or equivalent phrasing on the provided pages."
    )

    # ----------------------- Performance Metrics Verification -----------------------
    perf_node = evaluator.add_parallel(
        id="Performance_Metrics_Verification",
        desc="Validates that the deployment achieved the required performance improvements on a large-scale supercomputer",
        parent=root_node,
        critical=True
    )

    # Scaled Speedup Achievement
    speed_parent = evaluator.add_parallel(
        id="Scaled_Speedup_Achievement",
        desc="Verifies the deployment achieved at least 50x speedup compared to CPU-only baseline when scaled to a multi-node supercomputer",
        parent=perf_node,
        critical=True
    )

    # Existence gating for speedup and parallel efficiency
    evaluator.add_custom_node(
        result=bool(extracted.speedup_scaled),
        id="Speedup_Value_Provided",
        desc="A scaled speedup value is provided",
        parent=speed_parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.parallel_efficiency),
        id="Parallel_Efficiency_Value_Provided",
        desc="A parallel efficiency percentage is provided",
        parent=speed_parent,
        critical=True
    )

    # Speedup magnitude >= 50x
    speed_leaf = evaluator.add_leaf(
        id="Speedup_Magnitude",
        desc="The reported speedup is at least 50x compared to CPU-only baseline",
        parent=speed_parent,
        critical=True
    )
    claim_speed = (
        "When scaled to a large multi-node supercomputer, the deployment achieved at least 50x speedup "
        "compared to a CPU-only baseline."
    )
    await evaluator.verify(
        claim=claim_speed,
        node=speed_leaf,
        sources=collect_all_sources(extracted),
        additional_instruction=(
            f"The answer reported a specific speedup: {extracted.speedup_scaled!r}. "
            "Verify from the provided webpages that the speedup at multi-node scale is >= 50x versus CPU-only."
        )
    )

    # Parallel efficiency >= 90% at > 1000 nodes
    pe_leaf = evaluator.add_leaf(
        id="Parallel_Efficiency",
        desc="The deployment achieved at least 90% parallel efficiency when scaled to over 1,000 compute nodes",
        parent=speed_parent,
        critical=True
    )
    claim_pe = (
        "The deployment achieved at least 90% parallel efficiency when scaled to over 1,000 compute nodes."
    )
    await evaluator.verify(
        claim=claim_pe,
        node=pe_leaf,
        sources=collect_all_sources(extracted),
        additional_instruction=(
            f"The answer reported a specific parallel efficiency: {extracted.parallel_efficiency!r} with scaling context {extracted.nodes_scale!r}. "
            "Confirm that at scale of >1,000 nodes (e.g., 1024 nodes), the reported parallel efficiency is >= 90%."
        )
    )

    # Energy Efficiency Improvement
    energy_parent = evaluator.add_parallel(
        id="Energy_Efficiency_Improvement",
        desc="Confirms the deployment demonstrated at least 30x energy efficiency improvement compared to traditional CPU-only methods",
        parent=perf_node,
        critical=True
    )

    # Existence gating for energy efficiency and metrics URLs
    evaluator.add_custom_node(
        result=bool(extracted.energy_efficiency),
        id="Energy_Efficiency_Value_Provided",
        desc="An energy efficiency improvement factor is provided",
        parent=energy_parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.metrics_urls and len(extracted.metrics_urls) > 0),
        id="Metrics_URLs_Provided",
        desc="At least one performance metrics URL is provided",
        parent=energy_parent,
        critical=True
    )

    # Energy efficiency >= 30x
    ee_leaf = evaluator.add_leaf(
        id="Energy_Efficiency_Ratio",
        desc="The reported energy efficiency improvement is at least 30x compared to CPU-only baseline",
        parent=energy_parent,
        critical=True
    )
    claim_ee = (
        "The deployment demonstrated at least 30x energy efficiency improvement compared to a CPU-only baseline."
    )
    await evaluator.verify(
        claim=claim_ee,
        node=ee_leaf,
        sources=collect_all_sources(extracted),
        additional_instruction=(
            f"The answer reported a specific energy efficiency factor: {extracted.energy_efficiency!r}. "
            "Verify that the reported energy efficiency improvement is >= 30x over CPU-only methods."
        )
    )

    # Performance metrics URL actually documents metrics
    perf_url_leaf = evaluator.add_leaf(
        id="Performance_Metrics_URL",
        desc="Provides a URL reference documenting the performance and energy efficiency metrics",
        parent=energy_parent,
        critical=True
    )
    claim_metrics_doc = (
        f"This page documents performance metrics (speedup, parallel efficiency, or energy efficiency) "
        f"for the GH200 deployment at '{extracted.institute_name}'."
    )
    await evaluator.verify(
        claim=claim_metrics_doc,
        node=perf_url_leaf,
        sources=extracted.metrics_urls,
        additional_instruction="The page should explicitly mention quantitative performance metrics (e.g., '50x speedup', '90% efficiency', '30x energy efficiency')."
    )

    # ----------------------- Publication Verification -----------------------
    pub_parent = evaluator.add_parallel(
        id="Publication_Verification",
        desc="Confirms the research was presented at WACCPD 2024 and has an associated ArXiv preprint",
        parent=root_node,
        critical=True
    )

    # WACCPD 2024 Presentation
    waccpd_leaf = evaluator.add_leaf(
        id="WACCPD_2024_Presentation",
        desc="The research findings were presented at the WACCPD 2024 workshop",
        parent=pub_parent,
        critical=True
    )
    claim_waccpd = "The research findings were presented at the WACCPD 2024 workshop."
    await evaluator.verify(
        claim=claim_waccpd,
        node=waccpd_leaf,
        sources=collect_all_sources(extracted),
        additional_instruction="Look for explicit mentions like 'presented at WACCPD 2024' or 'WACCPD 2024 workshop' tied to this research."
    )

    # ArXiv Publication (parallel, critical)
    arxiv_parent = evaluator.add_parallel(
        id="ArXiv_Publication",
        desc="The research has an associated ArXiv preprint publication documenting the methodology and results",
        parent=pub_parent,
        critical=True
    )

    # Existence gating for arXiv URL
    evaluator.add_custom_node(
        result=bool(extracted.arxiv_url),
        id="ArXiv_URL_Provided",
        desc="An ArXiv URL is provided",
        parent=arxiv_parent,
        critical=True
    )

    # ArXiv paper exists and is relevant
    arxiv_exists_leaf = evaluator.add_leaf(
        id="ArXiv_Paper_Exists",
        desc="An ArXiv preprint exists documenting this GH200 deployment for seismic research",
        parent=arxiv_parent,
        critical=True
    )
    claim_arxiv = (
        "This arXiv preprint documents an NVIDIA GH200 (Grace Hopper) accelerated seismic or earthquake simulation "
        f"research effort by or affiliated with '{extracted.institute_name}' / '{extracted.university_name}'."
    )
    await evaluator.verify(
        claim=claim_arxiv,
        node=arxiv_exists_leaf,
        sources=extracted.arxiv_url,
        additional_instruction="Confirm that the paper is a preprint on arXiv and clearly pertains to GH200-based seismic/earthquake simulation research for the identified institute/university."
    )

    # Publication URL is arXiv
    arxiv_url_leaf = evaluator.add_leaf(
        id="Publication_URL",
        desc="Provides the ArXiv URL for the paper",
        parent=arxiv_parent,
        critical=True
    )
    claim_arxiv_url = "This URL is an arXiv preprint page (hosted on arxiv.org)."
    await evaluator.verify(
        claim=claim_arxiv_url,
        node=arxiv_url_leaf,
        sources=extracted.arxiv_url,
        additional_instruction="Verify that the URL domain is arxiv.org and the page corresponds to a paper preprint."
    )

    # ----------------------- Technical Architecture Confirmation -----------------------
    tech_parent = evaluator.add_parallel(
        id="Technical_Architecture_Confirmation",
        desc="Validates the deployment utilized the key technical features of the GH200 architecture",
        parent=root_node,
        critical=True
    )

    # NVLink-C2C utilization
    c2c_leaf = evaluator.add_leaf(
        id="NVLink_C2C_Utilization",
        desc="The implementation utilizes the NVLink-C2C chip-to-chip interconnect between Grace CPU and Hopper GPU",
        parent=tech_parent,
        critical=True
    )
    claim_c2c = (
        "The implementation utilizes the NVLink-C2C (chip-to-chip) interconnect between the Grace CPU and the Hopper GPU "
        "in the GH200 Grace Hopper Superchip."
    )
    await evaluator.verify(
        claim=claim_c2c,
        node=c2c_leaf,
        sources=collect_all_sources(extracted),
        additional_instruction="Look for explicit mentions of 'NVLink-C2C', 'chip-to-chip', or equivalent descriptions tying Grace CPU to Hopper GPU."
    )

    # Heterogeneous CPU-GPU computing (Grace CPU + H100 GPU)
    hetero_leaf = evaluator.add_leaf(
        id="Heterogeneous_Computing_Approach",
        desc="The deployment demonstrates a heterogeneous CPU-GPU computing approach that leverages both the Grace CPU and H100 GPU components",
        parent=tech_parent,
        critical=True
    )
    claim_hetero = (
        "The deployment uses a heterogeneous CPU-GPU computing approach, leveraging both the Grace CPU and the H100 GPU components."
    )
    await evaluator.verify(
        claim=claim_hetero,
        node=hetero_leaf,
        sources=collect_all_sources(extracted),
        additional_instruction="Confirm that both the Grace CPU and H100 GPU are used collaboratively (not GPU-only), consistent with GH200's architecture."
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
    # Initialize evaluator and root
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # overall wrapper; rubric root is added under this
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_research_institution(),
        template_class=GH200SeismicExtraction,
        extraction_name="gh200_seismic_extraction"
    )

    # Build and execute verification tree
    await build_verification_tree(evaluator, extracted)

    # Return final summary
    return evaluator.get_summary()