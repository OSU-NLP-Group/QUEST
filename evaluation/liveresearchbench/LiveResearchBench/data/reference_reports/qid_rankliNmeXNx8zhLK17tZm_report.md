# Interim Ranking: Best Professional Laptops Available (as of September 2025) — Based on Collected, Fully Sourced Data

## Scope and Method (transparent and reproducible)
- Goal: Rank laptops currently sold in 2025 for professional use by an equal-weight composite of:
  1) Geekbench 6 performance (single-core and multi-core)
  2) Battery life (hours) from reputable, clearly described third-party tests; if none, manufacturer claim (labeled)
  3) Weight (pounds) from official U.S. specification pages
- This interim report covers the models for which complete, verifiable sources were collected in the research to date (4 models). Each data point is linked to a primary source.
- Performance scoring:
  - For each laptop, compute PerfScore = average of normalized Geekbench 6 single-core and multi-core scores, where normalization divides by the category maximum in this set.
- Battery scoring:
  - Use the most credible, recent independent test with explicit methodology (brightness, workload). If unavailable, use official claim (labeled). Normalize each value by the maximum hours in this set.
- Weight scoring:
  - Lower is better. Normalize as WeightScore = (minimum weight in set) / (device weight).
- Composite score:
  - Equal-weight baseline = average(PerfScore, BatteryScore, WeightScore).
- Important notes:
  - Where exact tested configuration entries are scarce for Chromebooks, a clearly equivalent Geekbench 6 result using the same CPU on ChromeOS (e.g., Crostini/Android path) is used, with the equivalence explained and linked.
  - OS/platform differences can affect Geekbench variably; CPU parity is maintained per the brief.

## Baseline Ranking (equal-weight composite of performance, battery, weight)
1) System76 Lemur Pro (lemp13) — Composite 0.959
2) Dell XPS 13 (9340 / Developer Edition platform) — Composite 0.909
3) Acer Chromebook Spin 714 (CP714‑1WN, 2023) — Composite 0.719
4) Acer Chromebook Plus 515 (CB515‑2H, 2023) — Composite 0.653

Calculation snapshot (normalized within this set):
- Performance (PerfScore): computed from Geekbench 6 (single/multi)
- BatteryScore: hours normalized; Lemur Pro uses the manufacturer claim (clearly labeled) [1]
- WeightScore: min weight divided by device weight (all official pages cited) [1][5][8][11]

## Full Entries (configuration, metrics, sources)

### 1) System76 Lemur Pro (lemp13)
- Configuration (as sold/current):
  - CPU/GPU: Intel Core Ultra 7 155U; integrated Intel Graphics [1]
  - Display: 14.0-inch 1920×1200 matte (16:10) [1]
  - RAM/Storage: user-configurable; Geekbench tested unit reports ~39 GB RAM (≈ 32–40 GB installed) [2]
  - OS: Pop!_OS / Ubuntu (Linux) [1]
  - Availability: current product page (U.S.) [1]
- Geekbench 6 performance (Linux; exact CPU match, same model):
  - Single-core: 2346; Multi-core: 8210 — Geekbench Browser (Lemur Pro, Core Ultra 7 155U) [2]
- Battery life:
  - Manufacturer claim: “14-hour battery” (no recent independent test located for lemp13) [1]
- Weight:
  - 2.2 lb (0.998 kg) — official System76 U.S. product page [1]
- Notes on configuration mapping:
  - The Geekbench entry identifies Lemur Pro with the exact CPU (Core Ultra 7 155U). RAM reported aligns with common high-memory configurations; storage not exposed by Geekbench [2].

### 2) Dell XPS 13 (9340; Developer Edition platform available)
- Configuration (tested/representative):
  - CPU/GPU: Intel Core Ultra 7 155H; integrated Intel Arc (Meteor Lake) [3]
  - Display (as tested by Notebookcheck): 13.4-inch 1920×1200 IPS, 120 Hz [3]
  - Tested battery config (Notebookcheck): Core Ultra 7 155H, 32 GB LPDDR5x, 512 GB SSD [3]
  - OS: Ubuntu (Developer Edition available and Ubuntu-certified); Windows was used for the cited Geekbench result [6][4]
- Geekbench 6 performance (same model/CPU on Windows):
  - Single-core: 2118; Multi-core: 10,925 — Geekbench Browser (XPS 13 9340, 155H) [4]
- Battery life (independent):
  - Notebookcheck standardized WLAN browsing at 150 cd/m²:
    - ≈13 hours at 60 Hz; ≈11 hours at 120 Hz; ≈17.5 hours video at 150 cd/m². Article details methodology; review published May 2024 [3]
  - Ranking baseline uses the 13-hour WLAN 150 cd/m², 60 Hz result [3]
- Weight:
  - “Starting at” 2.6 lb — official Dell U.S. XPS 13 9340 page [5]
- Linux readiness:
  - Ubuntu-certified (entry for XPS 13 9340) [6]
- Notes:
  - Performance result is an XPS 13 9340 unit with the identical CPU bin (155H) and similar RAM [4]. Battery numbers are from Notebookcheck’s standardized tests [3].

### 3) Acer Chromebook Spin 714 (CP714‑1WN, 2023)
- Configuration (typical U.S. SKU tested):
  - CPU/GPU: Intel Core i5‑1335U; integrated Intel Iris Xe [8]
  - Display: 14.0-inch 1920×1200 IPS touch (16:10) [8]
  - RAM/Storage: 8 GB LPDDR4x, 256 GB SSD (as tested in major reviews) [8][7]
- Geekbench 6 performance (ChromeOS/Android path; exact CPU on Spin 714):
  - Single-core: 1856; Multi-core: 6153 — Geekbench Browser (Spin 714, i5‑1335U) [9]
- Battery life (independent):
  - Laptop Mag continuous web surfing over Wi‑Fi at 150 nits: 10:45 (10.75 hours) [10]
  - PCWorld’s CrXPRT battery test: 11:46 (methodology described) [7]
  - Ranking baseline uses Laptop Mag’s 150‑nit web result (10.75 h) [10]
- Weight:
  - 3.09 lb — official Acer U.S. product page [8]
- Notes:
  - Multiple independent battery tests with clear methods exist; the ranking uses Laptop Mag’s standardized 150‑nit web test for consistency with other outlets [10][7].

### 4) Acer Chromebook Plus 515 (CB515‑2H, 2023)
- Configuration (U.S. store SKU):
  - CPU/GPU: Intel Core i3‑1215U; integrated Intel UHD/Iris Xe (platform-dependent) [11]
  - Display: 15.6-inch 1920×1080 IPS (16:9) [11]
  - RAM/Storage: 8 GB / 128 GB (CB515‑2H‑31NY SKU) [11]
- Geekbench 6 performance (equivalent configuration on ChromeOS; same CPU):
  - Single-core: 2288; Multi-core: 6125 — Geekbench Browser (ASUS Chromebook Plus CX34 Crostini run with i3‑1215U; clearly equivalent CPU for class performance) [12]
- Battery life (independent):
  - Laptop Mag comparison table (150‑nit continuous web test): Chromebook Plus 515 listed at 8:21 (8.35 hours) [13]
- Weight:
  - 3.7 lb — Acer Store U.S. official page [11]
- Notes:
  - Direct Geekbench entries for CB515‑2H are sparse; the equivalent CPU (i3‑1215U) result on ChromeOS provides a clear proxy for CPU-class performance, aligned with the brief’s allowance for clearly equivalent configurations [12]. The cited battery figure comes from a standard 150‑nit web test table [13].

## Metric Leaderboards (from collected set)

### Performance (Geekbench 6; single/multi and PerfScore)
- Dell XPS 13 9340 (Core Ultra 7 155H): 2118 / 10,925; PerfScore = 0.951 (normalized) [4]
- System76 Lemur Pro (Core Ultra 7 155U): 2346 / 8210; PerfScore = 0.876 [2]
- Acer Chromebook Plus 515 (i3‑1215U; equivalent ChromeOS run): 2288 / 6125; PerfScore = 0.768 [12]
- Acer Chromebook Spin 714 (i5‑1335U): 1856 / 6153; PerfScore = 0.677 [9]

Note: PerfScore = average(normalized single, normalized multi) within this set.

### Battery life (hours; test/method)
- System76 Lemur Pro: 14.0 h (manufacturer claim; independent test not found) [1]
- Dell XPS 13 9340: ≈13.0 h — WLAN browsing at 150 cd/m², 60 Hz (Notebookcheck, May 2024) [3]
- Acer Chromebook Spin 714: 10.75 h — continuous web over Wi‑Fi at 150 nits (Laptop Mag) [10]; PCWorld CrXPRT 11.77 h corroboration [7]
- Acer Chromebook Plus 515: 8.35 h — 150‑nit continuous web (Laptop Mag comparison table) [13]

### Weight (pounds; official U.S. spec pages)
- System76 Lemur Pro: 2.2 lb [1]
- Dell XPS 13 9340: 2.6 lb (starting) [5]
- Acer Chromebook Spin 714: 3.09 lb [8]
- Acer Chromebook Plus 515: 3.7 lb [11]

## Sensitivities, discrepancies, and notes
- Battery methodology and refresh rate matter:
  - Dell XPS 13 9340 shows ≈13 h at 60 Hz but ≈11 h at 120 Hz in the same standardized test; both at 150 cd/m² (Notebookcheck, May 2024) [3].
- Manufacturer claim vs. independent test:
  - Lemur Pro’s 14-hour figure is a vendor claim; an independent test may rank it differently. It remains labeled as such [1].
- Cross-platform Geekbench:
  - To preserve model/CPU equivalence on ChromeOS, clearly equivalent CPU runs are used when laptop-identified entries are limited (e.g., CB Plus 515 via an i3‑1215U Crostini result). This follows the brief’s equivalence allowance and prioritizes CPU bin parity over OS differences [12].
- Linux readiness:
  - XPS 13 9340 has an Ubuntu-certified SKU; Geekbench result cited is Windows for the same CPU bin, which is common when Linux laptop entries are fewer in the Browser [6][4].

## How to use this interim ranking
- Equal-weight composite is a sensible baseline when you care equally about speed, runtime, and portability.
- Alternative weighting:
  - If battery life is paramount, prioritize XPS 13 9340’s independently tested ~13 hours at 150 cd/m² or Chromebooks with standardized 150‑nit web results [3][10][13].
  - If raw CPU performance is paramount in this set, the XPS 13 9340 leads on multi-core while the Lemur Pro leads single-core [4][2].
  - If portability dominates, the Lemur Pro’s 2.2 lb design is the lightest here [1].

## Limitations and next steps
- This is an interim deliverable constrained to models with fully sourced metrics collected to date (4 devices). Building the requested full Top 25 ranking requires gathering:
  - Geekbench 6 Browser links for each exact or clearly equivalent configuration
  - Independent battery results with explicit methods/dates (or clearly labeled manufacturer claims if none)
  - Official U.S. spec-page weights
- Upon approval, this methodology will be applied to a broader cross-OS slate (Windows on ARM/x86, macOS, Linux, ChromeOS) to produce the full 25-model composite ranking and per-metric leaderboards.

### Sources
[1] System76 Lemur Pro (lemp13) – U.S. product/configure page (specs, weight, battery claim): https://system76.com/laptops/lemp13/configure  
[2] Geekbench Browser – Lemur Pro (Core Ultra 7 155U) result: https://browser.geekbench.com/v6/cpu/6573197  
[3] Notebookcheck – Dell XPS 13 9340 review (battery methodology and results at 150 cd/m²; May 2024): https://www.notebookcheck.net/Dell-XPS-13-9340-laptop-review-The-compact-subnotebook-with-long-runtimes-and-quite-a-few-flaws.837639.0.html  
[4] Geekbench Browser – XPS 13 9340 (Core Ultra 7 155H): https://browser.geekbench.com/v6/cpu/7428840  
[5] Dell U.S. – XPS 13 9340 product page (weight/specs): https://www.dell.com/en-us/shop/xps-laptops-series/13-new/spd/xps-13-9340-laptop  
[6] Ubuntu Certified – Dell XPS 13 9340 entry (Developer Edition context): https://ubuntu.com/certified/202405-34051  
[7] PCWorld – Acer Chromebook Spin 714 (2023) review (CrXPRT battery test and methodology): https://www.pcworld.com/article/2105472/acer-chromebook-spin-714-2023-review.html  
[8] Acer U.S. – Chromebook Enterprise Spin 714 (CP714‑1WN) product page (specs/weight): https://www.acer.com/us-en/chromebooks/acer-chromebook-enterprise-spin-714-cp714-1wn/pdp/NX.K44AA.003  
[9] Geekbench Browser – Acer Chromebook Spin 714 (Core i5‑1335U) result: https://browser.geekbench.com/v6/cpu/3092382  
[10] Laptop Mag – Acer Chromebook Spin 714 review (150‑nit continuous web battery result): https://www.laptopmag.com/reviews/acer-chromebook-spin-714-review  
[11] Acer Store U.S. – Chromebook Plus 515 (CB515‑2H‑31NY) page (specs/weight): https://store.acer.com/en-us/acer-chromebook-plus-515-cb515-2h-31ny  
[12] Geekbench Browser – ASUS Chromebook Plus CX34 (i3‑1215U, Crostini) result used as CPU-equivalent for CB515: https://browser.geekbench.com/v6/cpu/4764003  
[13] Laptop Mag – HP Chromebook Plus x360 2024 review (comparison table listing Chromebook Plus 515 battery at 8:21): https://www.laptopmag.com/laptops/chromebooks/hp-chromebook-plus-x360-2024-review