# Ranking the Top U.S. BEVs by Range, 0–60, and Starting MSRP (as of Sep 5, 2025)

## Scope and Criteria
- Vehicles: Battery electric vehicles (BEVs) sold in the U.S.; plug-in hybrids (PHEVs) and fuel-cell (FCEV) models excluded.
- Data vintage: Latest official U.S. EPA entries and U.S. manufacturer pages current as of Sep 5, 2025.
- Configuration basis: Each model’s U.S. base trim (lowest “starting” configuration) so range, 0–60, and starting MSRP refer to the same exact trim and model year.
- Metrics and weights (equal weighting):
  - EPA combined range (miles) — source priority: U.S. EPA trim card on fueleconomy.gov.
  - 0–60 mph acceleration (seconds) — source priority: Official U.S. manufacturer spec page; if not published, reputable instrumented tests (Car and Driver, MotorTrend, Edmunds, Consumer Reports, InsideEVs, or Road & Track) for that exact trim.
  - Starting MSRP (USD) — source: Official U.S. manufacturer page. Note whether it includes destination; if included by the manufacturer, this is stated explicitly.

Normalization (min–max, 0–100):
- Range (higher better): (value − min_range) / (max_range − min_range) × 100
- 0–60 (lower better): (max_0to60 − value) / (max_0to60 − min_0to60) × 100
- MSRP (lower better): (max_msrp − value) / (max_msrp − min_msrp) × 100  
Composite score = average of the three normalized scores.

If a metric for a model’s base trim cannot be credibly sourced, that model is excluded and listed separately with attempted sources.

## Results at a Glance
Due to source completeness requirements, two models fully qualify for ranking at this time based on verified, trim-matched EPA range, 0–60 mph, and starting MSRP. Additional models appear in the Excluded Models section with specific data gaps and direct source attempts.

## Ranked List (composite score; tie-breakers: higher EPA range, then quicker 0–60, then lower MSRP)

1) Tesla Model 3 (2025) — Long Range RWD
- EPA combined range (base trim): 363 miles. Trim card and EPA Model 3 list confirm 2025 Long Range RWD 363 miles; EPA also lists separate 19-inch wheel variant with lower range, demonstrating wheel impact on efficiency [1][2].
- 0–60 mph (base trim): 4.9 s; Tesla cites this for Long Range RWD. No 1-ft rollout note for this trim (Tesla marks rollout only on Performance) [4].
- Starting MSRP (USD): $44,130 “price before estimated savings,” which Tesla states includes Destination and Order Fees, and excludes taxes and other fees. Tesla’s pricing footnote clarifies fee handling on U.S. pages [4][5].
- Normalized scores:
  - Range: 100.00
  - 0–60: 100.00
  - MSRP: 0.00
- Composite score: 66.67

2) Chevrolet Equinox EV (2025) — LT 1 FWD
- EPA combined range (base trim): 319 miles (FWD). EPA trim card and 2025 Equinox EV list confirm 319 miles for FWD; AWD variant is listed separately at 307 miles [10][11].
- 0–60 mph (base trim): 7.4 s (instrumented test of 2025 Equinox EV LT FWD by MotorTrend; no 1-ft rollout note) [13].
- Starting MSRP (USD): $33,600 “Starting at” for LT on Chevrolet’s U.S. page (asterisked disclaimers shown on-page; consumer “Starting at” pricing typically excludes taxes and dealer fees; destination not indicated as included on this page and is treated as excluded unless itemized at checkout) [12].
- Normalized scores:
  - Range: 0.00
  - 0–60: 0.00
  - MSRP: 100.00
- Composite score: 33.33

## Models Excluded Due to Missing Trim-Matched Data (with attempted sources)
- Tesla Model Y (2025) — Long Range RWD
  - Available: EPA combined range 337 miles for Long Range RWD; trim card and 2025 Model Y list confirm RWD entry [6][7]. 0–60 mph 6.5 s for Long Range RWD published by Tesla [9].
  - Missing for inclusion: Official U.S. “starting” MSRP published for the Long Range RWD trim on a static Tesla page. Tesla publicly shows a “starts at” line for Long Range AWD but does not present a public “starts at” price line for the RWD trim as of Sep 5, 2025 on the U.S. Model Y page [8].  
  - Result: Excluded for lack of a verifiable U.S. manufacturer “starting MSRP” line for the exact base RWD trim.
- Chevrolet Blazer EV (2025) — LT (base)
  - Available: EPA shows 2025 entries for Blazer EV AWD (283 miles) and RWD (334 miles) with trim cards and Power Search listing [15][16][17]. Chevrolet’s U.S. page lists “LT — Starting at $44,600*” with asterisked disclaimers [18].
  - Missing for inclusion: Verifiable 0–60 mph for the LT base configuration (Chevrolet does not publish it for LT, and reputable instrumented tests on record as of Sep 5, 2025 cover non-base trims/powertrains like RS AWD and SS) [19][20][21].  
  - Result: Excluded for lack of trim-correct 0–60 mph data for LT.

## Normalization Bounds Used (across qualifying models only)
- Range (miles): min = 319 (Equinox EV LT FWD) [10][11]; max = 363 (Model 3 Long Range RWD) [1][2]
- 0–60 mph (seconds): min = 4.9 (Model 3 Long Range RWD) [4]; max = 7.4 (Equinox EV LT FWD) [13]
- Starting MSRP (USD): min = 33,600 (Equinox EV LT FWD) [12]; max = 44,130 (Model 3 Long Range RWD; includes Destination and Order Fees) [4][5]

## Reproducible Calculation Appendix

Formulas:
- norm_range = (range − 319) / (363 − 319) × 100
- norm_0to60 = (7.4 − accel) / (7.4 − 4.9) × 100
- norm_msrp = (44,130 − MSRP) / (44,130 − 33,600) × 100
- composite = (norm_range + norm_0to60 + norm_msrp) / 3

Worked values:
- Tesla Model 3 Long Range RWD (2025)
  - range = 363 → norm_range = (363−319)/(44)×100 = 100.00 [1][2]
  - accel = 4.9 → norm_0to60 = (7.4−4.9)/(2.5)×100 = 100.00 [4]
  - MSRP = 44,130 → norm_msrp = (44,130−44,130)/(10,530)×100 = 0.00 [4][5]
  - composite = (100 + 100 + 0)/3 = 66.67
- Chevrolet Equinox EV LT 1 FWD (2025)
  - range = 319 → norm_range = (319−319)/(44)×100 = 0.00 [10][11]
  - accel = 7.4 → norm_0to60 = (7.4−7.4)/(2.5)×100 = 0.00 [13]
  - MSRP = 33,600 → norm_msrp = (44,130−33,600)/(10,530)×100 = 100.00 [12]
  - composite = (0 + 0 + 100)/3 = 33.33

CSV (downloadable; one row per qualifying vehicle; values correspond to the exact trim used)
columns: make,model,model_year,trim,epa_combined_range_mi,epa_url,zero_to_sixty_s,zero_to_sixty_source_url,rollout_note,starting_msrp_usd,msrp_url,msrp_includes_destination,norm_range_0to100,norm_0to60_0to100,norm_msrp_0to100,composite_score
Tesla,Model 3,2025,Long Range RWD,363,https://www.fueleconomy.gov/feg/noframes/48765.shtml,4.9,https://www.tesla.com/model3-performance,"No rollout stated for this trim",44130,"https://www.tesla.com/model3-performance; https://www.tesla.com/compare?selectedModel=modely",Yes,100.00,100.00,0.00,66.67
Chevrolet,Equinox EV,2025,LT 1 FWD,319,https://www.fueleconomy.gov/feg/noframes/48697.shtml,7.4,https://www.motortrend.com/reviews/2025-chevrolet-equinox-ev-lt-first-test-review,"No rollout stated",33600,https://www.chevrolet.com/electric/equinox-ev,No,0.00,0.00,100.00,33.33

## Important Notes and Assumptions
- Matching trim configuration: All three metrics (range, 0–60, MSRP) are tied to the same base trim and model year per model. If the base trim has multiple wheel options affecting EPA range, the EPA card for the default/base wheels is used (e.g., Model 3 Long Range RWD shows separate EPA entries for 18-inch vs. 19-inch configurations, with the 18-inch base configuration at 363 miles) [1][2].
- Pricing handling:
  - Tesla: The publicly shown “price before estimated savings” for Model 3 Long Range RWD includes Destination and Order Fees and excludes taxes/other fees per Tesla’s footnotes [4][5].
  - Chevrolet: The Equinox EV “Starting at” price shown on the consumer model page carries asterisked disclaimers; as presented, it is treated as excluding destination, taxes, and dealer fees unless itemized at checkout on the builder page [12].
- Missing data policy: Where a required metric is not credibly published for the exact base trim (e.g., Model Y Long Range RWD “starting” price on a static U.S. page; Blazer EV LT 0–60 mph), the model is excluded and listed with attempted sources [6][7][8][9][15][16][17][18][19][20][21].
- Date accessed: All sources were checked on Sep 5, 2025.

## What This Means for Shoppers and Analysts
- With equal weighting of range, performance, and price, a model that is best-in-class in two categories can outrank a cheaper model that excels only on price (e.g., Model 3 Long Range RWD vs. Equinox EV LT FWD in this provisional ranking) [1][2][4][5][10][11][12][13].
- Wheel/tire and drivetrain choices materially affect both range and acceleration; EPA often lists separate entries for wheel sizes when they change efficiency (as with Model 3 Long Range RWD 18-inch vs. 19-inch), and independent testing confirms such impacts in road tests (Equinox EV LT FWD base wheels/tires noted in the test vehicle) [1][2][13].

### Sources
[1] 2025 Tesla Model 3 Long Range RWD — EPA trim page: https://www.fueleconomy.gov/feg/noframes/48765.shtml (Accessed 2025-09-05)  
[2] EPA Power Search — 2025 Tesla Model 3 listings: https://www.fueleconomy.gov/feg/PowerSearch.do?action=noform&baseModel=Model+3&make=Tesla&pageno=1&pubDate=20250225&srchtyp=ymm&year1=2025&year2=2025 (Accessed 2025-09-05)  
[3] Tesla — Model 3 U.S. page: https://www.tesla.com/model3 (Accessed 2025-09-05)  
[4] Tesla — Model 3 Performance page (specs include Long Range RWD 0–60 and pricing footnote context): https://www.tesla.com/model3-performance (Accessed 2025-09-05)  
[5] Tesla — Compare page (pricing footnotes, fee inclusion/exclusion): https://www.tesla.com/compare?selectedModel=modely (Accessed 2025-09-05)  
[6] 2025 Tesla Model Y Long Range RWD — EPA trim page: https://www.fueleconomy.gov/feg/noframes/48771.shtml (Accessed 2025-09-05)  
[7] EPA Power Search — 2025 Tesla Model Y listings: https://www.fueleconomy.gov/feg/PowerSearch.do?action=noform&baseModel=Model+Y&make=Tesla&year1=2025&year2=2025 (Accessed 2025-09-05)  
[8] Tesla — Model Y U.S. page (AWD “starts at” line and pricing footnote): https://www.tesla.com/modely (Accessed 2025-09-05)  
[9] Tesla — “original-modely” specs page (includes Long Range RWD 0–60): https://www.tesla.com/original-modely (Accessed 2025-09-05)  
[10] 2025 Chevrolet Equinox EV FWD — EPA trim page: https://www.fueleconomy.gov/feg/noframes/48697.shtml (Accessed 2025-09-05)  
[11] EPA Power Search — 2025 Chevrolet Equinox EV listings: https://www.fueleconomy.gov/feg/PowerSearch.do?action=noform&baseModel=Equinox&make=Chevrolet&pageno=1&srchtyp=ymm&year1=2025&year2=2025 (Accessed 2025-09-05)  
[12] Chevrolet — Equinox EV U.S. model page (“Starting at” price): https://www.chevrolet.com/electric/equinox-ev (Accessed 2025-09-05)  
[13] MotorTrend — 2025 Chevrolet Equinox EV LT FWD First Test (instrumented 0–60): https://www.motortrend.com/reviews/2025-chevrolet-equinox-ev-lt-first-test-review (Accessed 2025-09-05)  
[14] Car and Driver — Chevrolet Equinox EV model page (context on lineup/wheels): https://www.caranddriver.com/chevrolet/equinox-ev/ (Accessed 2025-09-05)  
[15] 2025 Chevrolet Blazer EV RWD — EPA trim page: https://www.fueleconomy.gov/feg/noframes/48694.shtml (Accessed 2025-09-05)  
[16] 2025 Chevrolet Blazer EV AWD — EPA trim page: https://www.fueleconomy.gov/feg/noframes/48342.shtml (Accessed 2025-09-05)  
[17] EPA Power Search — 2025 Chevrolet Blazer EV listings: https://www.fueleconomy.gov/feg/PowerSearch.do?action=noform&baseModel=Blazer&make=Chevrolet&pageno=1&pubDate=20250422&srchtyp=ymm&year1=2025&year2=2025 (Accessed 2025-09-05)  
[18] Chevrolet — Blazer EV U.S. model page (“LT — Starting at”): https://www.chevrolet.com/electric/blazer-ev (Accessed 2025-09-05)  
[19] Car and Driver — Chevrolet Blazer EV model page (0–60 times for non-base trims): https://www.caranddriver.com/chevrolet/blazer-ev/ (Accessed 2025-09-05)  
[20] MotorTrend — 2025 Chevrolet Blazer SS EV First Test: https://www.motortrend.com/reviews/2025-chevrolet-blazer-ss-ev-first-test-review (Accessed 2025-09-05)  
[21] Car and Driver — 2025 Chevrolet Blazer SS vs. Kia EV6 GT comparison (0–60 reference, non-base): https://www.caranddriver.com/reviews/comparison-test/a65413725/2025-chevrolet-blazer-ss-kia-ev6-gt-comparison-test/ (Accessed 2025-09-05)