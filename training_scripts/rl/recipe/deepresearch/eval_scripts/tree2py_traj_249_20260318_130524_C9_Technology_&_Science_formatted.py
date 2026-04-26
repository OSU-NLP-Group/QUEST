import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "auto_mem_2026_adas"
TASK_DESCRIPTION = """
In 2026, amid a global memory chip shortage driven by AI data center demand, an automotive Tier-1 supplier is sourcing memory components for a next-generation Level 2+ ADAS system scheduled for production in Q4 2026. The system requires high-performance, automotive-qualified memory to support real-time processing of sensor data at bandwidth levels exceeding 60 GB/s.

Identify 4 different automotive-grade memory products that meet ALL of the following requirements:

1. Manufacturer: Must be from one of the top 3 global memory chip producers (SK Hynix, Samsung, or Micron), as these account for more than 90% of global memory production and have confirmed 2026 supply capacity.

2. Memory Type: Must be automotive-grade LPDDR5X, DDR5, or LPDDR5 suitable for ADAS applications.

3. Capacity: Minimum 8GB per component.

4. Performance:
   - Data rate must meet or exceed 6400 Mbps for LPDDR5X (or equivalent performance for DDR5/LPDDR5)
   - System bandwidth capability must support at least 60 GB/s for L2+ ADAS processing requirements

5. Automotive Qualification:
   - Must be AEC-Q100 certified or have equivalent automotive qualification
   - Must be explicitly marketed as automotive-grade

6. Operating Conditions: Operating temperature range must be suitable for automotive environments (minimum -40°C to +105°C).

7. 2026 Availability: Must have confirmed availability in 2026, evidenced by current production status or announced 2026 launch, with supply capacity confirmed despite the ongoing shortage.

8. Manufacturer Diversity: The 4 products must come from at least 3 different manufacturers.

For each product, provide:
- Manufacturer name
- Product model/series name
- Memory type
- Capacity specification
- Data rate/bandwidth specification
- Operating temperature range
- Automotive qualification status
- 2026 availability confirmation
- Reference URLs supporting the above information
"""

TOP3_MANUFACTURERS = {"sk hynix", "skhynix", "sk‑hynix", "samsung", "samsung electronics", "micron", "micron technology"}
ALLOWED_TYPES = {"lpddr5x", "lpddr5", "ddr5"}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProductExtraction(BaseModel):
    manufacturer: Optional[str] = None
    product_name: Optional[str] = None

    memory_type: Optional[str] = None
    capacity_spec: Optional[str] = None
    data_rate_spec: Optional[str] = None
    bandwidth_spec: Optional[str] = None

    operating_temp: Optional[str] = None
    qualification: Optional[str] = None
    availability: Optional[str] = None

    # Optional/non-critical extras (existence checks only)
    density_spec: Optional[str] = None
    latency_spec: Optional[str] = None
    voltage_spec: Optional[str] = None
    power_spec: Optional[str] = None
    interface_spec: Optional[str] = None
    package_spec: Optional[str] = None
    ecc_spec: Optional[str] = None

    # References (URLs). Extract ONLY explicit URLs mentioned in the answer.
    manufacturer_refs: List[str] = Field(default_factory=list)
    product_refs: List[str] = Field(default_factory=list)
    spec_refs: List[str] = Field(default_factory=list)
    qualification_refs: List[str] = Field(default_factory=list)
    availability_refs: List[str] = Field(default_factory=list)
    other_refs: List[str] = Field(default_factory=list)


class ProductsExtraction(BaseModel):
    products: List[ProductExtraction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_products() -> str:
    return """
    Extract up to 6 automotive memory products mentioned in the answer. For each product, return:

    Required core fields:
    - manufacturer: The company name as written in the answer
    - product_name: The model or series name as written
    - memory_type: e.g., LPDDR5X, LPDDR5, DDR5; include "automotive" qualifier if present
    - capacity_spec: the stated capacity text (keep units as written, e.g., 8GB, 64Gb)
    - data_rate_spec: the stated speed/data rate text (e.g., 6400 Mbps, 8533 MT/s)
    - bandwidth_spec: any stated bandwidth figure(s) if provided (e.g., "up to 68 GB/s")
    - operating_temp: the stated operating temperature range text
    - qualification: the stated automotive/AEC qualification text (e.g., AEC-Q100, "automotive-grade")
    - availability: any stated note regarding production/mass production or 2026 availability

    Optional (non-critical, existence only):
    - density_spec, latency_spec, voltage_spec, power_spec, interface_spec, package_spec, ecc_spec

    URLs (extract only explicit URLs in the answer):
    - manufacturer_refs: URL(s) confirming manufacturer identity or company product page
    - product_refs: URL(s) that explicitly show the product name/model/series
    - spec_refs: URL(s) with technical specifications/datasheets/product briefs
    - qualification_refs: URL(s) stating AEC automotive qualification or "automotive-grade"
    - availability_refs: URL(s) confirming current production or 2026 availability/capacity
    - other_refs: any other relevant URL(s)

    Rules:
    - Do NOT invent URLs.
    - If a field is missing, set it to null (or empty list for URL arrays).
    - Return all fields exactly as they appear in the answer.
    - Prefer official manufacturer pages in refs if present; also include press releases, credible media, or distributor pages when cited.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _canon_manufacturer(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower().replace("‑", "-")
    if "sk" in s and "hynix" in s:
        return "SK hynix"
    if "samsung" in s:
        return "Samsung"
    if "micron" in s:
        return "Micron"
    return name.strip()


def _is_top3(name: Optional[str]) -> bool:
    if not name:
        return False
    s = name.strip().lower().replace("‑", "-")
    if ("sk" in s and "hynix" in s) or s in {"sk hynix", "skhynix", "sk-hynix"}:
        return True
    if "samsung" in s:
        return True
    if "micron" in s:
        return True
    return False


def _collect_urls(prod: ProductExtraction, fields: List[str]) -> List[str]:
    urls: List[str] = []
    for f in fields:
        v = getattr(prod, f, None)
        if isinstance(v, list):
            urls.extend([u for u in v if isinstance(u, str) and u.strip() != ""])
        elif isinstance(v, str) and v.strip():
            urls.append(v)
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def _exists_nonempty(val: Optional[str]) -> bool:
    return bool(val and val.strip())


# --------------------------------------------------------------------------- #
# URL-backed leaf creator (with hard fail if no URLs provided)                #
# --------------------------------------------------------------------------- #
async def _url_backed_leaf(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    critical: bool,
    claim: str,
    urls: List[str],
    add_ins: str
):
    if urls:
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent,
            critical=critical,
        )
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=add_ins
        )
    else:
        # No sources => fail this required URL-backed check
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=f"{desc} (failed: no source URLs provided in the answer)",
            parent=parent,
            critical=critical
        )


# --------------------------------------------------------------------------- #
# Per-product verification                                                    #
# --------------------------------------------------------------------------- #
async def verify_one_product(evaluator: Evaluator, parent_node, prod: ProductExtraction, idx: int) -> None:
    pfx = f"p{idx+1}"

    # product_i (Parallel, non-critical)
    product_node = evaluator.add_parallel(
        id=f"product_{idx+1}",
        desc=f"{['First','Second','Third','Fourth','Fifth','Sixth'][idx]} automotive memory product meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # ---------------------- Basic info (Parallel, critical) -------------------
    basic_info = evaluator.add_parallel(
        id=f"{pfx}_basic_info",
        desc="Basic product identification information",
        parent=product_node,
        critical=True
    )

    # Manufacturer (Sequential, critical)
    manu_seq = evaluator.add_sequential(
        id=f"{pfx}_manufacturer",
        desc="Manufacturer verification",
        parent=basic_info,
        critical=True
    )

    # p?_manufacturer_identity (leaf): top 3 membership (simple verify)
    manu_identity = evaluator.add_leaf(
        id=f"{pfx}_manufacturer_identity",
        desc="Manufacturer must be one of the top 3 global memory producers (SK Hynix, Samsung, or Micron)",
        parent=manu_seq,
        critical=True
    )
    manu_name = prod.manufacturer or ""
    await evaluator.verify(
        claim=f"The manufacturer '{manu_name}' is one of the top three global memory producers: SK Hynix, Samsung, or Micron.",
        node=manu_identity,
        additional_instruction="Treat 'SK hynix' and 'SK Hynix' as the same; 'Samsung Electronics' counts as 'Samsung'; 'Micron Technology' counts as 'Micron'. Minor casing differences are acceptable."
    )

    # p?_manufacturer_ref (leaf): URL(s) confirm manufacturer/product provenance
    manu_urls = _collect_urls(prod, ["manufacturer_refs", "product_refs", "spec_refs"])
    await _url_backed_leaf(
        evaluator,
        parent=manu_seq,
        node_id=f"{pfx}_manufacturer_ref",
        desc="URL reference confirming manufacturer identity",
        critical=True,
        claim=f"This webpage confirms that the product '{prod.product_name or ''}' is manufactured by {manu_name}.",
        urls=manu_urls,
        add_ins="Prefer official manufacturer domains; acceptable alternatives include credible press releases, reputable media, or authorized distributor pages that explicitly state the manufacturer for the named product."
    )

    # Product identification (Sequential, critical)
    prod_seq = evaluator.add_sequential(
        id=f"{pfx}_product_identification",
        desc="Product model identification",
        parent=basic_info,
        critical=True
    )

    # p?_product_name (existence, custom)
    evaluator.add_custom_node(
        result=_exists_nonempty(prod.product_name),
        id=f"{pfx}_product_name",
        desc="Product model or series name clearly identified",
        parent=prod_seq,
        critical=True
    )

    # p?_product_ref (URL-backed)
    prod_name_urls = _collect_urls(prod, ["product_refs", "spec_refs"])
    await _url_backed_leaf(
        evaluator,
        parent=prod_seq,
        node_id=f"{pfx}_product_ref",
        desc="URL reference for product identification",
        critical=True,
        claim=f"This webpage clearly identifies a memory product with model/series name '{prod.product_name or ''}'.",
        urls=prod_name_urls,
        add_ins="Allow minor formatting differences, hyphenation, or series suffixes/prefixes when matching the product name."
    )

    # --------------- Technical specifications (Parallel, critical) -----------
    tech_specs = evaluator.add_parallel(
        id=f"{pfx}_technical_specifications",
        desc="Technical specifications meeting ADAS requirements",
        parent=product_node,
        critical=True
    )

    # memory_characteristics (Parallel, critical)
    mem_chars = evaluator.add_parallel(
        id=f"{pfx}_memory_characteristics",
        desc="Core memory type and capacity specifications",
        parent=tech_specs,
        critical=True
    )

    # p?_memory_type (URL-backed)
    memtype_urls = _collect_urls(prod, ["spec_refs", "product_refs", "qualification_refs"])
    await _url_backed_leaf(
        evaluator,
        parent=mem_chars,
        node_id=f"{pfx}_memory_type",
        desc="Memory type is automotive-grade LPDDR5X, DDR5, or LPDDR5",
        critical=True,
        claim=f"This webpage states that the product is {prod.memory_type or 'the required memory type'} and that it targets automotive/ADAS use cases.",
        urls=memtype_urls,
        add_ins="Confirm that the memory type is one of: LPDDR5X, LPDDR5, or DDR5, and that the page positions it as automotive-grade (e.g., says 'Automotive', 'for ADAS', 'automotive-grade')."
    )

    # p?_capacity (URL-backed, >=8GB)
    cap_urls = _collect_urls(prod, ["spec_refs", "product_refs"])
    await _url_backed_leaf(
        evaluator,
        parent=mem_chars,
        node_id=f"{pfx}_capacity",
        desc="Memory capacity is at least 8GB",
        critical=True,
        claim="This webpage indicates that the device capacity per component is at least 8 GB (eight gigabytes). If density is listed in gigabits (Gb), 64 Gb equals 8 GB, 128 Gb equals 16 GB, etc.",
        urls=cap_urls,
        add_ins="Interpret density correctly: divide gigabits (Gb) by 8 to get gigabytes (GB). The requirement is >= 8 GB per component."
    )

    # p?_density (existence only, non-critical)
    evaluator.add_custom_node(
        result=_exists_nonempty(prod.density_spec),
        id=f"{pfx}_density",
        desc="Memory density information provided",
        parent=mem_chars,
        critical=False
    )

    # performance_metrics (Parallel, critical)
    perf = evaluator.add_parallel(
        id=f"{pfx}_performance_metrics",
        desc="Performance specifications for ADAS workloads",
        parent=tech_specs,
        critical=True
    )

    # p?_data_rate (URL-backed, >=6400 Mbps/MT/s)
    dr_urls = _collect_urls(prod, ["spec_refs", "product_refs"])
    await _url_backed_leaf(
        evaluator,
        parent=perf,
        node_id=f"{pfx}_data_rate",
        desc="Data rate meets or exceeds 6400 Mbps for LPDDR5X or equivalent for other types",
        critical=True,
        claim="This webpage indicates a nominal data rate/speed of at least 6400 Mbps (or MT/s) for the memory device.",
        urls=dr_urls,
        add_ins="Accept equivalent notations such as MT/s. Typical LPDDR5X speeds include 6400, 7500, 8533 MT/s. If the page lists a family range, confirm the selected device reaches >= 6400."
    )

    # p?_bandwidth (URL-backed, >=60 GB/s system capability)
    bw_urls = _collect_urls(prod, ["spec_refs", "product_refs"])
    await _url_backed_leaf(
        evaluator,
        parent=perf,
        node_id=f"{pfx}_bandwidth",
        desc="Total bandwidth sufficient for L2+ ADAS (at least 60 GB/s system bandwidth capability)",
        critical=True,
        claim="This evidence demonstrates that using this memory product in a typical automotive configuration enables at least 60 GB/s system memory bandwidth.",
        urls=bw_urls,
        add_ins=(
            "If the page directly states bandwidth >= 60 GB/s, that's sufficient. "
            "If not explicit, infer using bandwidth = data_rate (bytes/s) * bus_width / 8 per channel and combine channels "
            "if the page mentions multi-channel support. Use reasonable industry-standard widths (e.g., x32/x64) only if the page provides them."
        )
    )

    # p?_latency (existence only, non-critical)
    evaluator.add_custom_node(
        result=_exists_nonempty(prod.latency_spec),
        id=f"{pfx}_latency",
        desc="Latency specifications provided",
        parent=perf,
        critical=False
    )

    # operational_specs (Parallel, critical)
    ops = evaluator.add_parallel(
        id=f"{pfx}_operational_specs",
        desc="Operational specifications for automotive environment",
        parent=tech_specs,
        critical=True
    )

    # p?_operating_temp (URL-backed, -40C..+105C minimum)
    temp_urls = _collect_urls(prod, ["spec_refs", "qualification_refs", "product_refs"])
    await _url_backed_leaf(
        evaluator,
        parent=ops,
        node_id=f"{pfx}_operating_temp",
        desc="Operating temperature range suitable for automotive (-40°C to +105°C minimum)",
        critical=True,
        claim="This webpage states an operating temperature range that includes at least -40°C minimum and +105°C maximum (or higher, e.g., +125°C).",
        urls=temp_urls,
        add_ins="Verify the product's specified operating temperature range. Accept wider ranges (e.g., -40 to +125°C)."
    )

    # p?_voltage (existence only, non-critical)
    evaluator.add_custom_node(
        result=_exists_nonempty(prod.voltage_spec),
        id=f"{pfx}_voltage",
        desc="Operating voltage specification provided",
        parent=ops,
        critical=False
    )

    # p?_power_consumption (existence only, non-critical)
    evaluator.add_custom_node(
        result=_exists_nonempty(prod.power_spec),
        id=f"{pfx}_power_consumption",
        desc="Power consumption or efficiency metrics provided",
        parent=ops,
        critical=False
    )

    # interface_specs (Parallel, non-critical)
    iface = evaluator.add_parallel(
        id=f"{pfx}_interface_specs",
        desc="Interface and packaging specifications",
        parent=tech_specs,
        critical=False
    )
    evaluator.add_custom_node(
        result=_exists_nonempty(prod.package_spec),
        id=f"{pfx}_package_type",
        desc="Package type specification provided",
        parent=iface,
        critical=False
    )
    evaluator.add_custom_node(
        result=_exists_nonempty(prod.interface_spec),
        id=f"{pfx}_interface",
        desc="Interface type clearly specified",
        parent=iface,
        critical=False
    )

    # reliability_features (Parallel, non-critical)
    rel = evaluator.add_parallel(
        id=f"{pfx}_reliability_features",
        desc="Reliability and error correction features",
        parent=tech_specs,
        critical=False
    )
    evaluator.add_custom_node(
        result=_exists_nonempty(prod.ecc_spec),
        id=f"{pfx}_ecc_support",
        desc="ECC or error correction capability specified",
        parent=rel,
        critical=False
    )

    # p?_tech_specs_ref (URL-backed)
    specs_urls = _collect_urls(prod, ["spec_refs"])
    await _url_backed_leaf(
        evaluator,
        parent=tech_specs,
        node_id=f"{pfx}_tech_specs_ref",
        desc="URL reference for technical specifications",
        critical=True,
        claim=f"This webpage is a datasheet, product brief, or official technical specification page for '{prod.product_name or ''}'.",
        urls=specs_urls,
        add_ins="Prefer official manufacturer pages (datasheet, product brief). If distributor pages are cited, they must contain detailed specs for the exact model."
    )

    # --------------- Automotive qualification (Sequential, critical) ---------
    qual_seq = evaluator.add_sequential(
        id=f"{pfx}_automotive_qualification",
        desc="Product has automotive-grade certification",
        parent=product_node,
        critical=True
    )

    cert_parallel = evaluator.add_parallel(
        id=f"{pfx}_certification",
        desc="Automotive certification verification",
        parent=qual_seq,
        critical=True
    )

    qual_urls = _collect_urls(prod, ["qualification_refs", "spec_refs", "product_refs"])

    # p?_aec_q100 (URL-backed)
    await _url_backed_leaf(
        evaluator,
        parent=cert_parallel,
        node_id=f"{pfx}_aec_q100",
        desc="AEC-Q100 certified or equivalent automotive qualification",
        critical=True,
        claim="This webpage explicitly states that the device is AEC-Q100 qualified/compliant, or presents an equivalent recognized automotive IC qualification.",
        urls=qual_urls,
        add_ins="Look for 'AEC-Q100' or equivalent automotive IC qualification language. Phrases like 'AEC-Q100 qualified', 'compliant', or 'meets automotive qualification' count."
    )

    # p?_automotive_grade_explicit (URL-backed)
    await _url_backed_leaf(
        evaluator,
        parent=cert_parallel,
        node_id=f"{pfx}_automotive_grade_explicit",
        desc="Explicitly marketed as automotive-grade",
        critical=True,
        claim="This webpage explicitly markets the product as 'automotive-grade' or clearly positions it for automotive/ADAS applications.",
        urls=qual_urls,
        add_ins="Accept explicit labels like 'Automotive', 'AEC', 'Automotive-Grade', 'for ADAS', or an automotive product category page."
    )

    # p?_qual_ref (URL-backed, after certification)
    await _url_backed_leaf(
        evaluator,
        parent=qual_seq,
        node_id=f"{pfx}_qual_ref",
        desc="URL reference for qualification status",
        critical=True,
        claim=f"This webpage is an official or credible page that confirms the automotive qualification status of '{prod.product_name or ''}'.",
        urls=qual_urls,
        add_ins="Prefer official manufacturer or standards conformance pages; credible press releases accepted."
    )

    # --------------------- Availability 2026 (Sequential, critical) ----------
    avail_seq = evaluator.add_sequential(
        id=f"{pfx}_availability_2026",
        desc="Product availability confirmed for 2026",
        parent=product_node,
        critical=True
    )

    supply_parallel = evaluator.add_parallel(
        id=f"{pfx}_upply_status",
        desc="Supply and production status verification",
        parent=avail_seq,
        critical=True
    )

    avail_urls = _collect_urls(prod, ["availability_refs", "product_refs", "spec_refs", "manufacturer_refs"])

    # p?_production_status (URL-backed)
    await _url_backed_leaf(
        evaluator,
        parent=supply_parallel,
        node_id=f"{pfx}_production_status",
        desc="Product is in production or announced for 2026 availability",
        critical=True,
        claim="This webpage indicates that the product is in production now or explicitly announced for availability/mass production in 2026.",
        urls=avail_urls,
        add_ins="Evidence examples: 'mass production', 'now in production', 'sampling in 2025 with MP in 2026', 'available 2026', 'production ramp 2026'."
    )

    # p?_upply_confirmation (URL-backed) [typo in id aligned with parent id to keep consistency]
    await _url_backed_leaf(
        evaluator,
        parent=supply_parallel,
        node_id=f"{pfx}_supply_confirmation",
        desc="Evidence of supply or production capacity in 2026",
        critical=True,
        claim="This webpage provides evidence or statements regarding secured supply, capacity, or availability for 2026 despite shortages.",
        urls=avail_urls,
        add_ins="Accept official capacity statements, allocation commitments, or credible industry reports that explicitly tie to 2026 availability."
    )

    # p?_avail_ref (URL-backed, after supply status)
    await _url_backed_leaf(
        evaluator,
        parent=avail_seq,
        node_id=f"{pfx}_avail_ref",
        desc="URL reference for availability confirmation",
        critical=True,
        claim=f"This webpage serves as the evidence confirming 2026 availability/supply for '{prod.product_name or ''}'.",
        urls=avail_urls,
        add_ins="Prefer official manufacturer/press sources; distributors' 2026 delivery notes acceptable if they explicitly state availability."
    )


# --------------------------------------------------------------------------- #
# Manufacturer diversity check                                                #
# --------------------------------------------------------------------------- #
def _manufacturer_diversity_ok(products: List[ProductExtraction]) -> Tuple[bool, List[str]]:
    names: List[str] = []
    for p in products[:4]:
        if _is_top3(p.manufacturer):
            names.append(_canon_manufacturer(p.manufacturer) or (p.manufacturer or "").strip())
        elif p.manufacturer:
            names.append((p.manufacturer or "").strip())
    uniq = []
    seen = set()
    for n in names:
        if n not in seen and n:
            uniq.append(n)
            seen.add(n)
    return (len(uniq) >= 3, uniq)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    # Initialize evaluator (root must be non-critical to allow mixed critical children)
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

    # Extract structured products
    extracted: ProductsExtraction = await evaluator.extract(
        prompt=prompt_extract_products(),
        template_class=ProductsExtraction,
        extraction_name="memory_products_extraction"
    )

    # Keep first 4 products (pad with empty if fewer)
    prods: List[ProductExtraction] = list(extracted.products[:4])
    while len(prods) < 4:
        prods.append(ProductExtraction())

    # Verify each product subtree
    for idx in range(4):
        try:
            await verify_one_product(evaluator, root, prods[idx], idx)
        except Exception as e:
            # Hard fail entire product node if unexpected error
            product_node = evaluator.add_parallel(
                id=f"product_{idx+1}_error",
                desc=f"Product #{idx+1} verification aborted due to internal error: {e}",
                parent=root,
                critical=False
            )
            evaluator.add_custom_node(
                result=False,
                id=f"product_{idx+1}_unrecoverable",
                desc=f"Product #{idx+1} unrecoverable verification failure",
                parent=product_node,
                critical=True
            )

    # Manufacturer diversity (Critical)
    ok, uniq_names = _manufacturer_diversity_ok(prods)
    evaluator.add_custom_node(
        result=ok,
        id="manufacturer_diversity",
        desc="The 4 products must come from at least 3 different manufacturers among SK Hynix, Samsung, and Micron",
        parent=root,
        critical=True
    )

    # Add helpful info
    evaluator.add_custom_info(
        info={"unique_manufacturers_in_first_4": uniq_names},
        info_type="diversity_info",
        info_name="manufacturer_diversity_info"
    )

    # Add constraint reminder (as ground truth context)
    evaluator.add_ground_truth({
        "allowed_manufacturers": ["SK Hynix", "Samsung", "Micron"],
        "allowed_types": ["LPDDR5X", "LPDDR5", "DDR5"],
        "min_capacity_gb": 8,
        "min_data_rate_mbps": 6400,
        "min_system_bandwidth_gbs": 60,
        "min_operating_temp_range": "-40C to +105C",
        "availability_year": 2026,
        "diversity_requirement": ">= 3 different manufacturers across 4 products"
    })

    return evaluator.get_summary()