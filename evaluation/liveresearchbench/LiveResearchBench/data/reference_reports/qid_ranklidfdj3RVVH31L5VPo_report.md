# Top 12 Running Shoes (Aug 2025) — Data Gaps, Method, and Delivery Plan

## Executive Summary
The brief requires building a ranked “Top 12” list from a candidate pool of currently available running shoes as of August 31, 2025, using an equal-weight composite of three factors: heel-to-toe drop (mm), per-shoe weight (oz), and Runner’s World (RW) numeric rating. The provided research transcript contains no collected shoe data, URLs, or Runner’s World rating pages. Because every factual statement and figure in the final output must be source-verified with direct URLs, the Top 12 rankings cannot be produced until live data gathering is completed.

What is ready now:
- A precise methodology (including normalization, tie-breakers, and handling edge cases).
- A fully specified data schema and output format that satisfy the brief (including three scenario rankings with min–max scaling).
- The exact conversion factor and citation to convert grams to ounces.

What is needed to complete the deliverable:
- Live collection of: (a) Runner’s World numeric ratings (US preferred; UK fallback when US lacks a numeric score), (b) brand-spec drop and weight with basis, and (c) current in-stock availability on the brand’s official site (or a major reputable retailer if the brand site lacks stock), each with direct URLs and access dates.

Approval requested:
- Proceed to collect sources and produce the full candidate pool and the three Top 12 rankings, adhering strictly to the brief and the documentation below.

## Requirements Restated (for accuracy control)
- Include only road and trail running shoes with current availability as of Aug 31, 2025 (brand’s US site preferred; major US retailer acceptable if brand site is out of stock).
- Include only models with:
  - A numeric Runner’s World score on the latest review/page available up to Aug 31, 2025 (RW US preferred; use RW UK if US lacks a numeric score and label it “RW UK”).
  - Brand-published heel-to-toe drop (mm) and per-shoe weight with basis; if the brand does not publish one of these, use a reputable retailer and label the exception.
- Availability region defaults to US; if only available in another region, note explicitly.
- Exclude walking, hiking, lifestyle, soccer/tennis shoes.
- Use NIST’s exact factor when converting grams to ounces (1 oz = 28.349523125 g) [1].

## Data Model (what will be captured per shoe)
- Brand
- Model and generation/version
- Category: Road or Trail (running only)
- Runner’s World rating (verbatim numeric score, scale as printed, e.g., 8.5/10 or 4/5; label edition “RW US” or “RW UK”)
- RW page URL and visible publication/update date
- Heel-to-toe drop (mm) with brand URL; note any discrepancies and include both URLs if needed
- Weight (with basis, e.g., Men’s US 9 or Women’s US 7); record grams and ounces (two decimals). If conversion is needed, compute oz = grams / 28.349523125 and cite NIST [1]
- Availability URL (brand preferred; major reputable retailer if brand is out of stock), region, and access date
- Notes/caveats (e.g., spec discrepancies, region-only availability)

## Ranking Methodology
- Candidate pool: All models meeting the inclusion criteria above.
- Normalization (min–max scaling to 0–100 across the pool) [2]:
  - Runner’s World rating: higher is better.
  - Weight: lower is better.
  - Heel-to-toe drop: three scenarios are computed separately (each metric weighted 1/3):
    1) Lower drop is better (minimize drop).
    2) Higher drop is better (maximize drop).
    3) Closer to the dataset median drop is better (score by absolute proximity to the median). The median will be computed on the final candidate pool and stated explicitly in the report with the list of underlying values.
- Composite score: mean of the three normalized sub-scores (each weight = 1/3).
- Tie-breakers (in order): higher Runner’s World rating, then lower weight, then alphabetical (Brand–Model).

## Output Format (three Top-12 tables)
For each of the three scenarios above, the report will present a Top 12 ranking table with the following columns:

- Rank
- Brand
- Model (version)
- Variant (men/women/unisex)
- Category (road/trail)
- Heel-to-toe drop (mm) with brand URL
- Weight (oz; basis and URL; grams→oz conversion when needed with NIST reference [1])
- Runner’s World rating (verbatim numeric score; RW US or RW UK label; URL; page date)
- Availability URL (brand preferred; or major retailer if brand is out of stock), region, and access date
- Normalized sub-scores (drop/weight/RW)
- Composite score
- Notes/caveats

## Normalization Details and Formulas
- Min–max scaling to 0–100 per metric (dataset-wide):
  - For metrics where “higher is better” (RW rating; also drop in scenario 2):
    - score = 100 × (value − min) / (max − min)
  - For metrics where “lower is better” (weight; also drop in scenario 1):
    - score = 100 × (max − value) / (max − min)
  - For “closer to the median is better” (scenario 3, drop):
    - Let m = median drop of the candidate pool; let d = |value − m|
    - Compute d_min and d_max across the pool
    - score = 100 × (d_max − d) / (d_max − d_min)
- Composite score = average of the three normalized sub-scores (equal weights).
- All rounding and numeric formatting will be consistent across tables and described in a short note under Methodology.

Reference for min–max scaling convention [2].

## Conversions
- If a brand provides weight only in grams, convert to ounces with two decimals using the NIST conversion factor:
  - 1 oz (avoirdupois) = 28.349523125 g (exact) [1]
  - oz = grams / 28.349523125
- The basis (e.g., “Men’s US 9” or “Women’s US 7”) will be captured verbatim from the brand page (or labeled as a retailer’s basis if brand data are unavailable).

## Handling Special Cases
- Runner’s World rating:
  - Use the most recent RW US review page that includes a numeric rating; if no numeric score appears on the US page, use RW UK if it shows a numeric or star score and label it “RW UK” in the table.
  - If neither US nor UK provides a numeric score, the model will be excluded (by brief).
- Specification discrepancies:
  - If brand and retailer specs (drop/weight) conflict, the brand’s numbers will be used and the discrepancy will be footnoted with direct URLs to both pages.
- Availability:
  - Default to brand’s US product page showing “Add to cart”/“In stock.” If unavailable on the brand site, use a major reputable retailer’s US page with an “In stock” indicator and label it as retailer-sourced.
  - Record access dates and region (US; or specify if only another region is available).

## What Will Be Delivered
- A candidate pool (minimum 20–30 models) satisfying all inclusion criteria with fully verified URLs and dates.
- Three Top 12 tables (scenarios 1–3) with:
  - All required columns and URLs (brand specs, RW rating page, availability page)
  - Normalized sub-scores and composite scores
  - Explicit median drop (for scenario 3) and the underlying drop values used to compute it.
- A short Methodology section in the report documenting:
  - Normalization formulas and directionality
  - Unit conversions with NIST citation [1]
  - How conflicting data were handled
  - Access dates for all availability checks

## Sample Table Schema (structure only)
This is the exact column structure that will be populated after source collection. It is shown here to confirm scope; it does not include any shoe data yet.

- Rank
- Brand
- Model (version)
- Variant (men/women/unisex)
- Category (road/trail)
- Heel-to-toe drop (mm) — URL
- Weight (oz) — basis and URL; grams and oz shown; conversion via NIST [1] when applicable
- Runner’s World rating (verbatim; RW US or RW UK) — URL — publication/update date
- Availability URL — region — access date
- Normalized sub-scores (drop/weight/RW)
- Composite score
- Notes/caveats

## Next Steps and Timing
- Upon approval to proceed, the full data collection and verification will be completed and delivered with all URLs and access dates. The three Top 12 rankings will then be computed and presented as specified.
- The data capture will adhere to the Aug 31, 2025 cutoff for “latest RW rating available” and “currently available” status, with access dates recorded for transparency.

### Action Requested
- Approve proceeding with live data collection using the above methodology, including the RW UK fallback when RW US lacks a numeric score.
- Optional: Provide any “must-include” models to prioritize in the initial batch.

### Sources
[1] NIST Guide to the SI, Appendix B.8 (conversion factors): https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b8  
[2] Scikit-learn: MinMaxScaler (min–max normalization reference): https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html