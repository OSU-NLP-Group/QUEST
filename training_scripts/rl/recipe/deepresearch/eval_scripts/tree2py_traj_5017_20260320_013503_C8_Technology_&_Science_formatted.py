import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "q2_2026_smartphones_us"
TASK_DESCRIPTION = """
A small technology startup based in California needs to purchase company smartphones for their development and testing team during Q2 2026. Due to the ongoing DRAM shortage affecting smartphone availability and specifications, they need to carefully select devices that still meet their minimum performance requirements.

Identify 4 different smartphone models that will be available for purchase in the United States during Q2 2026 (April 1 - June 30, 2026) and meet ALL of the following criteria:

1. At least 8GB of RAM (to handle development/testing workloads)
2. At least 128GB of internal storage
3. Price between $300 and $800 USD per unit
4. Manufactured by a major smartphone brand (among the top 10 global smartphone vendors by market share)
5. Confirmed to be available, announced, or released for the US market during Q2 2026 or earlier (must be purchasable during Q2)

For each phone model, provide:
- Exact model name
- RAM specification (in GB)
- Storage capacity (in GB)
- Price (in USD)
- Manufacturer name
- A reference URL from an official source (manufacturer website, authorized retailer, or reputable tech news site) that confirms the specifications and availability
"""

# A conservative superset of major global brands commonly seen in top-10 vendor rankings.
MAJOR_TOP10_BRANDS = {
    "Samsung", "Apple", "Xiaomi", "OPPO", "Vivo", "vivo", "HONOR", "Honor", "Huawei",
    "Transsion", "TECNO", "Infinix", "itel", "realme", "Motorola", "Lenovo", "OnePlus"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PhoneEntry(BaseModel):
    model_name: Optional[str] = None
    ram: Optional[str] = None
    storage: Optional[str] = None
    price: Optional[str] = None
    manufacturer: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class PhonesExtraction(BaseModel):
    phones: List[PhoneEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_phones() -> str:
    return """
    Extract up to 8 smartphone entries exactly as they appear in the answer. For each smartphone, return:
    - model_name: exact model name string as written in the answer
    - ram: RAM specification as written (e.g., "8GB", "12 GB", "8/12GB", "8 GB LPDDR5")
    - storage: internal storage as written (e.g., "128GB", "256 GB", "128/256GB UFS 3.1")
    - price: price in USD as written (e.g., "$399", "USD 499", "$349.99", "$300–$800")
    - manufacturer: manufacturer/brand name as written (e.g., "Samsung", "Apple", "Motorola")
    - reference_urls: a list of all URLs explicitly cited for this phone (manufacturer site, authorized US retailers, or reputable tech news sites). Extract only actual URLs present in the answer.
    
    Return an object with field "phones": an array of phone objects in the same order as the answer.
    For any missing field, set it to null. For reference_urls, if none are present, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def first_k(items: List[Any], k: int) -> List[Any]:
    return items[:k] if items else []


def pad_to_k(items: List[Any], k: int, pad_value_factory) -> List[Any]:
    out = list(items)
    while len(out) < k:
        out.append(pad_value_factory())
    return out


def build_sources_list(phone: PhoneEntry) -> Optional[List[str]]:
    if phone and phone.reference_urls:
        # Deduplicate while preserving order
        seen = set()
        urls: List[str] = []
        for u in phone.reference_urls:
            if isinstance(u, str) and u.strip() and u not in seen:
                urls.append(u.strip())
                seen.add(u.strip())
        return urls if urls else None
    return None


# --------------------------------------------------------------------------- #
# Verification for one phone                                                  #
# --------------------------------------------------------------------------- #
async def verify_one_phone(
    evaluator: Evaluator,
    parent_node,
    phone: PhoneEntry,
    idx: int
) -> None:
    """
    Build verification subtree for a single phone (Parallel aggregation).
    All leaf nodes are critical as per rubric.
    """
    phone_node = evaluator.add_parallel(
        id=f"phone_{idx+1}",
        desc=[
            "First smartphone model meets all requirements",
            "Second smartphone model meets all requirements",
            "Third smartphone model meets all requirements",
            "Fourth smartphone model meets all requirements",
        ][idx] if idx < 4 else f"Smartphone model #{idx+1} meets all requirements",
        parent=parent_node,
        critical=False
    )

    model_name = phone.model_name or "the phone"
    mfg_name = (phone.manufacturer or "").strip()
    sources = build_sources_list(phone)

    # 1) Reference URL validity (must be official manufacturer, authorized retailer, or reputable tech news)
    ref_leaf = evaluator.add_leaf(
        id=f"phone_{idx+1}_reference",
        desc="Valid reference URL provided from manufacturer or authorized retailer (or reputable tech news site)",
        parent=phone_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is a valid reference from an official manufacturer website, an authorized US retailer, or a reputable tech news outlet, containing information about this phone model.",
        node=ref_leaf,
        sources=sources,
        additional_instruction=(
            "Judge based on domain and page content (e.g., samsung.com, apple.com, motorola.com; "
            "bestbuy.com, amazon.com, verizon.com, att.com, t-mobile.com, walmart.com, target.com; "
            "or reputable outlets like gsmarena.com, theverge.com, cnet.com, androidauthority.com, "
            "tomsguide.com, engadget.com, etc.). The page should mention the phone model."
        )
    )

    # 2) RAM >= 8GB
    ram_leaf = evaluator.add_leaf(
        id=f"phone_{idx+1}_ram",
        desc="RAM specification meets the minimum requirement of 8GB or higher",
        parent=phone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The referenced page shows that {model_name} has at least 8 GB of RAM. The answer states RAM as '{phone.ram}'.",
        node=ram_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the device RAM is >= 8 GB for any purchasable US variant. "
            "Do not count 'virtual RAM' or 'RAM expansion' features. "
            "Minor formatting variants like '8GB', '8 GB', or '8 gigabytes' are acceptable."
        )
    )

    # 3) Storage >= 128GB
    storage_leaf = evaluator.add_leaf(
        id=f"phone_{idx+1}_storage",
        desc="Internal storage meets minimum 128GB requirement",
        parent=phone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The referenced page shows that {model_name} offers at least 128 GB of internal storage in a US-available configuration. The answer states storage as '{phone.storage}'.",
        node=storage_leaf,
        sources=sources,
        additional_instruction=(
            "Accept any US-sold configuration with >= 128 GB internal storage. "
            "Treat '128GB', '128 GB', or higher as satisfying the requirement."
        )
    )

    # 4) Price within $300-$800 USD
    price_leaf = evaluator.add_leaf(
        id=f"phone_{idx+1}_price",
        desc="Price is within $300-$800 range accounting for 2026 ASP increases",
        parent=phone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The US price for {model_name} is between $300 and $800 USD. The answer lists price as '{phone.price}'.",
        node=price_leaf,
        sources=sources,
        additional_instruction=(
            "Prefer MSRP or typical retailer price in USD for the US market. "
            "If multiple variants or storage options exist, it's acceptable if at least one US-sold configuration "
            "falls within $300-$800. Avoid relying on limited-time coupons or non-US currencies. "
            "If only non-US pricing is shown and no clear USD US price is provided, consider this unsupported."
        )
    )

    # 5) Availability in the US by Q2 2026 (or earlier, purchasable in Q2)
    avail_leaf = evaluator.add_leaf(
        id=f"phone_{idx+1}_availability",
        desc="Model is confirmed available or announced for Q2 2026 in US market",
        parent=phone_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The referenced page confirms that {model_name} is available for purchase in the United States "
            "during Q2 2026 (April 1 - June 30, 2026) or was released/announced earlier and purchasable by then."
        ),
        node=avail_leaf,
        sources=sources,
        additional_instruction=(
            "Evidence can include: 'available now', US retailer product pages, carrier listings, or official "
            "US announcements showing sales start before or within Q2 2026. If the page is clearly for another "
            "region or the device is not (yet) available to US buyers by Q2 2026, this should fail."
        )
    )

    # 6) Manufacturer is a major top-10 brand
    # We implement this as a deterministic custom check against a conservative set of major brands.
    is_major_brand = bool(mfg_name) and any(
        mfg_name.lower() == b.lower() for b in MAJOR_TOP10_BRANDS
    )
    evaluator.add_custom_node(
        result=is_major_brand,
        id=f"phone_{idx+1}_manufacturer",
        desc="Manufactured by a major brand among top 10 global smartphone vendors",
        parent=phone_node,
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
    Evaluate an answer for the Q2 2026 smartphones US purchase task.
    """
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

    # Extract structured phone entries
    extracted = await evaluator.extract(
        prompt=prompt_extract_phones(),
        template_class=PhonesExtraction,
        extraction_name="phones_extraction"
    )

    # Keep first 4, pad if necessary
    phones = first_k(extracted.phones, 4)
    phones = pad_to_k(phones, 4, pad_value_factory=PhoneEntry)

    # Record auxiliary info (e.g., major brand list and extracted model names)
    evaluator.add_custom_info(
        info={"allowed_major_brands": sorted(list(MAJOR_TOP10_BRANDS))},
        info_type="brand_policy",
        info_name="major_brand_whitelist"
    )
    evaluator.add_custom_info(
        info={"models": [p.model_name for p in phones]},
        info_type="extracted_models",
        info_name="models_listed_in_answer"
    )

    # Build verification subtrees for each of the 4 phones
    verify_tasks = []
    for i in range(4):
        verify_tasks.append(verify_one_phone(evaluator, root, phones[i], i))

    await asyncio.gather(*verify_tasks)

    return evaluator.get_summary()