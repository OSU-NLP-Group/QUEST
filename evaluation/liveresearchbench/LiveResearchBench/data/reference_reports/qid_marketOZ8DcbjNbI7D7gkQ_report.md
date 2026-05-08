# 2024 Actuals and 2025 Outlook: AI Data Center Accelerators and Consumer Discrete GPUs

## Executive summary

- Data center AI accelerators (servers): 2024 global revenue ≈ $123B; 2025 outlook ≈ $207B (range $190–225B depending on supply, licensing to China, and hyperscaler capex). Units in 2024 are estimated at 8–10 million accelerators; 2025 units 10–13 million as HBM and CoWoS bottlenecks ease modestly [1][2][3][4][25][26][27][28][29][30][31].
- Consumer discrete GPUs (desktop AIBs + notebook dGPUs): 2024 desktop AIBs 34.7M units (sum of quarterly public JPR posts). Notebook dGPUs are estimated at 26–31M based on IDC gaming PC data; total 2024 consumer dGPU units ≈ 60–66M. FY2024 revenue ranges an estimated $22.6–30.4B (methodology shown; anchor pricing/ASP context provided) [54][55][56][57][62][63][64][65][66].
- 2024 vendor shares: in data center accelerators, NVIDIA ≈ $87B revenue (~71% share), AMD ≈ $4.5–5B, Intel Gaudi < $0.5B, with custom ASICs (TPU/Trainium/Ascend) ≈ 25–26% of market value; NVIDIA ≈ 64% of AI-chip units across GPU/ASIC (higher within GPU-only servers). In consumer desktop AIBs (2024 weighted), NVIDIA ~85–87%, AMD ~12–14%, Intel ~0–1%; early 2025 has NVIDIA ~94% AIB share per JPR [1][3][5][6][7][58][59][61].
- Geography (2024): AI accelerators concentrated in the U.S. (55–65% of units), with China ~10–15% under export controls and Europe ~10–15%; revenue skews even more to U.S./Europe due to higher-ASP deployments. For discrete GPUs, public trackers do not provide regional splits; estimates (method described) suggest U.S. ~30–33%, Europe ~23–26%, China ~26–29% of 2024 desktop AIB units [2][3][11][12][14][15][16][17][68][69].
- 2025 demand drivers: hyperscaler capex remains elevated (Alphabet, Microsoft, Amazon, Meta), while inference workloads expand; MLPerf Training v5.0 and Inference v5.0 confirm record participation and large-scale performance gains. AI PCs and Windows 10 end-of-support are near-term positives for notebook dGPU mix; discrete desktop demand also lifted by tariff-related pull-ins in 1H25 [14][15][16][17][18][19][20][21][70][71][72].
- Supply constraints/enablers: HBM remains tight but improving (sold out through 2024 and largely 2025), 12‑Hi ramps at SK hynix and Micron; TSMC CoWoS capacity expands into 2025 with OSAT support; rack power and liquid cooling requirements (e.g., NVL72 at ~120–140kW/rack) constrain hosting; grid constraints rise with AI data center power growth [25][26][27][28][29][30][31][32][33][34].
- Export controls: U.S. BIS October 2023 rules and April 2024 clarifications limited high-end shipments to China; in April 2025, new license requirements hit NVIDIA H20 and AMD MI308 (NVIDIA took a ~$5.5B charge; AMD flagged up to ~$800M impact), with indications in July 2025 of license resumption; Huawei Ascend output is supply/yield constrained in 2024–2025 [35][36][37][38][39][40][41][42][43][11][12][13].

---

## Scope and definitions

- In-scope markets:
  - Data center AI accelerators used in servers for training and inference (merchant GPUs plus custom accelerators like TPU/Trainium/Ascend) [1][3][44][45][46].
  - Consumer discrete GPUs: desktop add‑in boards (AIBs) and notebook dGPUs (gaming- and AI‑capable); integrated GPUs excluded (flagged where sources bundle totals) [54][55][56][57][62][63][67].
- Timeframe: 2024 full-year actuals and 2025 forecasts; 2025 YTD where available (e.g., JPR AIB shipments Q1–Q2’25) [60][61].
- Geography: Breakouts for U.S., China, Europe where possible; otherwise clearly labeled estimates [2][3][11][12][68][69].

---

## 1) Market size and growth

### 1.1 Data center AI accelerators (servers)

- 2024 revenue (global): $123B. Source: Omdia’s data-center AI processor market release summarizing 2024 actuals [1].
- 2025 revenue (global): $207B (central case; range $190–225B reflecting supply/licensing scenarios). Source: same Omdia release [1]; corroborating growth momentum from components trackers into 2025 [2][4].
- 2024 units (global): estimated 8.0–10.0 million accelerators.
  - Anchor 1: U.S. hyperscalers (Amazon, Google, Meta, Microsoft) “set to deploy over 5 million AI training‑capable accelerators in 2024” [2].
  - Anchor 2: TrendForce raised 2024 AI server shipments to 1.67M, with 71% GPU‑ and 26% ASIC‑based; converting systems to chips with 4.8–6.0 accelerators/server implies ~8–10M chips (method disclosed) [3].
- 2025 units (global): estimated 10–13 million (higher accelerators/server and easing of HBM/CoWoS bottlenecks; see Section 3) [4][25][26][27][28][29][30][31].

Vendor shares and market sizing (2024)

- NVIDIA data center GPU revenue ≈ $87B (Omdia projection cited), ~71% of the $123B accelerator market [5][1]. NVIDIA’s reported Data Center segment includes networking; SEC filing provides compute vs networking split to anchor mix [9][10].
- AMD Instinct GPU revenue “exceed $4.5B” (raised from “exceed $4B”) [6][7].
- Intel Gaudi 2024 revenue below $500M target [8].
- Custom ASICs (TPU/Trainium/Ascend, etc.) comprise the residual ~$31–32B (~25–26%) consistent with TrendForce’s 2024 architecture mix (ASIC servers ~26%) [1][3][44][45].
- Unit shares: TrendForce reported NVIDIA ≈ 64% when including all AI chips (GPU/ASIC/FPGA), and ≈ 90% within GPU-equipped AI servers; AMD ~8% within GPU servers (units) [3].

Geography (2024 units; estimates with anchors)

- U.S.: ~55–65% of global accelerator units. Anchor: >5M deployed by U.S.-headquartered hyperscalers in 2024; U.S. holds ~54% of hyperscale capacity (location bias) [2][9].
- China: ~10–15%. 2024 shipments of top-end merchant GPUs constrained by U.S. controls; domestic Huawei Ascend ramp constrained by low yields (20% improving toward ~40% by early 2025) and 2025 output limits signaled by U.S. officials [11][13][12].
- Europe: ~10–15%. Driven by U.S. hyperscalers’ EU regions and sovereign cloud buildouts (Microsoft, AWS, Google) [9][17][18][19][20][21][22].

Growth drivers (2025)

- Omdia’s 2025 $207B outlook assumes continued hyperscaler investment; Dell’Oro’s long-term forecast also signals rapid expansion (to $382B by 2029) [1][24].
- Potential moderation risks include licensing to China and supply constraints (see Sections 3–4) [38][39][40][41][42][43].

Table: Data center AI accelerators (global)

- 2024 revenue: $123B [1]
- 2024 units: 8–10M (estimate from [2][3])
- 2025 revenue: $207B (range $190–225B) [1]
- 2025 units: 10–13M (estimate) [4][25]

### 1.2 Consumer discrete GPUs (desktop AIB + notebook dGPU)

Units (global)

- Desktop AIB shipments (JPR public posts):
  - Q1’24: 8.7M; Q2’24: 9.5M; Q3’24: 8.1M; Q4’24: 8.4M → 2024 total 34.7M [54][55][56][57].
- Notebook dGPU units (estimate): 26–31M in 2024.
  - IDC: gaming PCs ~44M in 2023 and +1% in 2024, with notebooks the primary driver; IDC also cited 10.6M gaming PCs in Q2’24. Assuming 60–70% notebooks and near‑universal dGPU in gaming notebooks yields 26–31M notebook dGPUs in 2024. Handheld PCs (mostly iGPU/APU) are excluded (IDC ~1.5M in 2024) [62][63][64].

Revenue (global; estimates with transparent ASP bands)

- Desktop AIB revenue: $17.4–22.6B in 2024, assuming blended $500–650 ASPs across the 34.7M units (elevated high‑end pricing vs midrange softness; pricing analysis cited) [57][65].
- Notebook dGPU revenue: $5.2–7.8B, assuming OEM blended ASPs ~$200–250 for 26–31M units (no public OEM ASP series; labeled estimate).
- Total consumer discrete GPU revenue: ~$22.6–30.4B in 2024 (sum of ranges). NVIDIA’s FY2025 Gaming revenue (primarily GeForce dGPU) was $11.4B, a cross-check for the scale of the discrete market segments versus vendor reporting (note: vendor segment definitions vary) [66].

2025 YTD and outlook

- Desktop AIBs: Q1’25 9.2M; Q2’25 11.6M (+27% q/q) with NVIDIA ~94% share; strong 1H’25 linked to pre‑tariff inventory pull‑ins [60][59][61][72].
- 2025 AIB units: 40–44M (forecast), with normalization in H2 as tariffs shift timing and as new product cycles (RTX 50, RDNA 4 midrange) progress [59][61][72].
- 2025 mobile dGPU: 27–33M (forecast), aided by “AI PC” marketing and Windows 10 EOS; IDC expects 274M PCs in 2025 (tariff effects) [70][71].

Vendor shares (desktop AIBs)

- 2024 snapshot by quarter (JPR-based summaries):
  - Q4’24: NVIDIA 82%, AMD 17%, Intel ~1% [58].
- 2024 annual (weighted): NVIDIA ~85–87%; AMD ~12–14%; Intel ~0–1% (derived from quarterly shares/units) [54][55][56][57][58].
- 2025: NVIDIA share ~94% in Q2’25 (AMD ~6%, Intel ≈0%); similar in Q1’25 ~92–94% (various reports summarizing JPR) [59][61][41].

Regional mix (2024 desktop AIB units; estimates with method)

- No public AIB regional splits. Using 2024 PC sell‑in by region, gaming propensity multipliers, and Steam HWS as a qualitative cross‑check, estimated AIB split:
  - U.S. ~30–33%, Europe ~23–26%, China ~26–29%, Rest of World ~15–18% (sums to ~34–35M) [67][68][69][62].

---

## 2) Demand drivers

### 2.1 Training and inference workloads

- MLPerf Training v5.0 (June 2025) added very large LLMs (e.g., Llama 3.1 405B) and recorded 201 performance results across 20 submitters; NVIDIA and AMD submitted production systems, underscoring record scale for training [19].
- MLPerf Inference v5.0 (April 2025) introduced new gen‑AI tests (Llama 3.1 405B, LL70B‑Interactive) and showed large per‑GPU gains for Blackwell (e.g., GB200 NVL72 up to 3.4× per‑GPU throughput vs H200 on Llama 3.1‑405B), evidencing rapid inference scaling [18][92].
- MLPerf Training v4.0 (June 2024) demonstrated record scale on Hopper H100/H200; NVIDIA reported breaking prior training time records at massive scale [20].
- Intel Gaudi 2 participated in MLPerf Inference v4.0; Intel’s published figures detail relative positioning on GPT‑J and SDXL versus NVIDIA H100 baselines (v4.0) [21][94].

### 2.2 Hyperscaler capex and fleet disclosures

- 2024 capex and 2025 plans signal continued accelerator demand:
  - Alphabet 2024 capex $52.5B; commentary indicated higher 2025 investment [14].
  - Microsoft plans ~$80B capex in FY2025, concentrated in AI/data centers [15].
  - Amazon targets ~$100B capex in 2025 (from ~$83B in 2024) [16].
  - Meta guides $60–65B capex in 2025; CEO commentary in early 2024 cited targets of ~350k H100 by YE2024 and ~600k H100‑equivalents overall [17][23].
- Hyperscaler deployments: U.S. hyperscalers deployed “over 5 million training‑capable accelerators in 2024,” anchoring U.S.-led demand [2].

### 2.3 Client/edge and gaming (consumer dGPU)

- IDC: gaming PC market returned to growth in 2024 (Q2’24 10.6M units, first y/y growth since Q1’22), with notebooks the primary driver; this supports strong notebook dGPU volumes [62][63].
- “AI PC” share reached ~14% of quarterly PC shipments by Q2’24 (Canalys), with expectations for broader adoption in 2025 (definition includes NPUs and is broader than dGPU) [70][71].

---

## 3) Supply-side constraints and enablers

### 3.1 Advanced packaging (CoWoS/SoIC) and foundry dependencies

- TSMC’s advanced packaging capacity (CoWoS, SoIC) was fully booked by NVIDIA and AMD through 2025; multiple trackers cite expansion from ~35–40k 12” WPM CoWoS by YE2024 toward ≥50k in 2025, with scenarios up to ~65–75k WPM by Q4’25 as AP8 ramps (third‑party estimates) [25][26][27].
- OSAT support and U.S. packaging buildouts (e.g., Amkor) add capacity, but TSMC remains the primary bottleneck for high‑end accelerator substrates in 2024–2025 [25][27].
- Vendor/foundry reliance:
  - NVIDIA Hopper/Blackwell on TSMC custom 4N/N4P with CoWoS; AMD Instinct MI300/MI325X on TSMC N5/N6 with CoWoS [79][25][75].
  - Intel Gaudi 3 fabbed at TSMC (5nm) per Intel newsroom; OAM and PCIe variants [77].

### 3.2 HBM supply

- HBM remained sold‑out in 2024 and largely booked for 2025; SK hynix indicated HBM was “almost sold out for 2025” (May 2024). SK hynix and Micron ramped 12‑Hi HBM3E (36GB) in late‑2024/early‑2025, while Micron broke ground on a $7B HBM advanced-packaging facility in Singapore (ops 2026) [28][29][30][31].
- These expansions support 2025 unit growth, but tightness persists at the leading performance grades (HBM3E) [28][29][30].

### 3.3 Power and thermal constraints

- NVIDIA GB200 NVL72 racks require on the order of 120–140kW with liquid cooling designs; reference architectures from Schneider Electric and Vertiv were introduced to support these densities [32][33].
- Uptime’s 2024 survey shows growing high‑density adoption, but 30kW+ racks remain relatively rare, implying retrofits/new builds are needed to host NVL36/NVL72‑class systems at scale [34].

---

## 4) Geopolitics and export controls

- U.S. BIS rules constraining advanced computing items to China (Oct 2022; updated Oct 17, 2023) tightened thresholds and broadened restrictions; clarifications issued April 4, 2024. Dutch national controls impacted ASML DUV/EUV licensing (partial revocations effective Jan 1, 2024) [35][36][37].
- April 2025: U.S. imposed license requirements for NVIDIA H20 and AMD MI308 exports to China. NVIDIA disclosed a ~$5.5B charge; AMD flagged up to ~$800M impact. By mid‑July 2025, both indicated license reviews would resume, with expectations for approvals on H20/MI308 (policy risk remains) [38][39][40][41][42][43].
- China domestic alternatives: Huawei Ascend ramp faced low 2024 yields (~20%) improving toward ~40% in early‑2025; a U.S. official stated Huawei could not make more than 200k advanced AI chips in 2025, highlighting ongoing supply constraints [11][13][12].

Impact on 2024–2025 market

- 2024: China’s share of high‑end AI server shipments was depressed; revenue mix skewed toward U.S./Europe. NVIDIA’s 10‑K showed China at ~13% of revenue (bill‑to), corroborating limited contribution relative to U.S./EU in FY2025 (year ended Jan 2025) [7][66].
- 2025 scenarios (see Section 6) hinge on licensing cadence and Huawei yields.

---

## 5) Competitive comparison (NVIDIA vs AMD vs Intel Arc vs Qualcomm/others)

### 5.1 Processing performance and efficiency (data center)

MLPerf highlights (official)

- Training v5.0 (Jun 2025): record participation; new Llama 3.1 405B benchmark; available, closed submissions from NVIDIA GB200/B200 and AMD MI300X/MI325X; detailed per-system results in the MLCommons report and browser [19][73].
- Inference v5.0 (Apr 2025): added Llama 3.1 405B and interactive Llama workloads; NVIDIA reports GB200 NVL72 up to 3.4× per‑GPU throughput vs H200 on Llama 3.1‑405B; MLCommons provides the underlying tables/results [18][92][73].
- Inference v4.0/v4.1 (2024): H200 (available) and H100 (available) results established the Hopper baseline across LLM and vision; Intel Gaudi 2 published GPT‑J and SDXL figures; AMD MI300X/MI325X appeared in v5.0 with multi‑node entries by partners [21][91][94][93].

Flagship specifications (official pages)

- NVIDIA H200: 141GB HBM3e, 4.8TB/s, SXM up to ~700W; NVLink up to 900GB/s (board-level) [78].
- NVIDIA Blackwell (B200/GB200, NVL72): B200 with up to 192GB HBM3e; GB200 NVL72 rack aggregates 72 Blackwell GPUs + 36 Grace CPUs, up to 13.4TB HBM3e and 130TB/s NVLink in-rack; FP4 rack‑scale claims documented on official page/blog [79][74][80].
- AMD Instinct MI325X: 256GB HBM3e, 6TB/s per OAM; production late‑2024, broad availability Q1’25; MI350 series (2025) up to 288GB HBM3e and 8TB/s [75][76].
- Intel Gaudi 3: 128GB HBM2e, 3.7TB/s; OAM/PCIe (600W PCIe card); 24×200GbE fabric ports [77].

### 5.2 Consumer discrete GPUs (selected 2024–2025 launches)

- NVIDIA GeForce RTX 40 SUPER series (Jan 2024): RTX 4080 SUPER $999 MSRP; availability Jan 31, 2024 [81].
- NVIDIA GeForce RTX 5090/5080 (Jan 30, 2025): $1,999/$999 MSRPs; independent reviews show large 4K/RT gains with higher power budgets (5090 up to ~575W limits in FE) [82][83][84][85].
- AMD Radeon RX 9070 XT / RX 9070 (Mar 2025): $599/$549; 16GB GDDR6 (256‑bit), 304W/220W TDP (TechPowerUp DB) [86][87][88].
- Intel Arc B580 (Dec 2024): $249 MSRP; 12GB GDDR6 (192‑bit), 190W TDP (Intel SKU and TPU review) [89][90].

### 5.3 Market share snapshots

- Data center accelerators, 2024: NVIDIA ~71% revenue; AMD ~4%; Intel <0.5%; custom ASICs ~25–26% (Omdia/TrendForce triangulation) [1][5][6][7][8][3].
- Desktop AIBs: 2024 weighted ~NVIDIA 85–87% / AMD 12–14% / Intel ~0–1%; Q2’25 ~94/6/≈0 per JPR summaries [58][59][61].

### 5.4 Partnerships and design wins (2024–2025)

- NVIDIA Blackwell clouds: AWS EC2 P6 (B200) GA May 15, 2025; Google A4 (B200) preview Jan 31, 2025; Microsoft Azure and Oracle Cloud announced participation in Blackwell rollouts [47][48][79].
- AMD Instinct MI300X/MI325X: Azure ND MI300X v5 VMs GA May 21, 2024; Oracle Cloud MI300X Supercluster GA Sep 26, 2024 [49][50].
- Government/HPC: DOE NNSA El Capitan (fastest supercomputer, MI300A APUs) accepted Nov 18, 2024; EuroHPC JUPITER (first European exascale) powered by GH200 Grace Hopper [51][52][53].

---

## 6) 2025 scenarios, assumptions, and uncertainties

### 6.1 Data center accelerators

- Base case (aligned to Omdia $207B): continued U.S./EU hyperscaler ramp, partial resumption of licensed shipments to China (H20/MI308), Huawei Ascend output in the 300–500k range; HBM/CoWoS expand sufficiently to support 10–13M units (global) [1][42][43][25][26][27][28][29][30][31].
- Bear case: prolonged licensing frictions; Huawei capped near ~200k units; unit growth nearer the low end of the 10–13M range; revenue $190–200B [12][38][41].
- Bull case: broad licensing approvals, faster Huawei yields/output, robust HBM3E 12‑Hi supply; revenue $215–225B; modest share shift toward China (units) [42][43][29][30].

Key swing factors

- Policy/licensing to China; Huawei yields/output; HBM and CoWoS mix/throughput; rack power and grid availability; hyperscaler capex pacing and timing of Blackwell/Hopper transitions [35][36][11][12][13][28][29][25][26][33][14][15][16][17].

### 6.2 Consumer discrete GPUs

- 2025 desktop AIB units 40–44M: H1 pre‑tariff pull‑ins, H2 normalization; product cycles (RTX 50 series, RDNA 4 midrange) [59][61][72].
- 2025 notebook dGPU units 27–33M: “AI PC” and Windows 10 EOS support demand; some pull-forward related to tariffs and OEM launches [70][71][72].
- NVIDIA’s AIB share remains dominant (~90%+) in early 2025; AMD’s share stabilizes with RDNA 4 midrange breadth; Intel’s desktop AIB share remains de‑minimis in 2025 JPR snapshots [59][61][41].

---

## 7) Detailed tables

### 7.1 Market sizing and growth (global)

- Data center accelerators (servers)
  - 2024 revenue: $123B [1]
  - 2024 units: 8–10M (estimate from [2][3])
  - 2025 revenue: $207B (range $190–225B) [1]
  - 2025 units: 10–13M (estimate) [4][25]

- Consumer discrete GPUs
  - 2024 desktop AIB units: 34.7M (sum of Q1–Q4) [54][55][56][57]
  - 2024 notebook dGPU units: 26–31M (estimate; IDC gaming PC data) [62][63][64]
  - 2024 revenue: $22.6–30.4B (estimate; ASP methodology) [65][66]
  - 2025 AIB units: 40–44M (forecast) [60][59][61]
  - 2025 notebook dGPU units: 27–33M (forecast) [70][71]

### 7.2 2024 vendor shares

- Data center accelerators (revenue):
  - NVIDIA ≈ $87B (~71%); AMD ≈ $4.5–5B (~4%); Intel Gaudi < $0.5B; Custom ASICs ≈ 25–26% [5][1][6][7][8][3].
- Desktop AIBs (units, 2024 weighted):
  - NVIDIA ~85–87%; AMD ~12–14%; Intel ~0–1% [58][54][55][56][57].

### 7.3 2024 geography (units; estimates unless stated)

- Data center accelerators:
  - U.S. ~55–65% (anchor: >5M deployed by U.S. hyperscalers; U.S. ~54% of hyperscale capacity) [2][9].
  - China ~10–15% (export controls; Huawei yield constraints) [11][12][13].
  - Europe ~10–15% (EU region buildout) [9][17][18][19][20][21][22].
- Desktop AIBs:
  - U.S. ~30–33%; Europe ~23–26%; China ~26–29%; RoW ~15–18% (method detailed in §1.2) [67][68][69][62][63].

---

## 8) Launch timelines and availability (selected, 2024–2025)

- Data center
  - NVIDIA H200: shipping 2024; MLPerf available results Aug 2024 [91][78].
  - NVIDIA Blackwell (B200/GB200): announced Mar 18, 2024; AWS EC2 P6 (B200) GA May 15, 2025; Google A4 preview Jan 31, 2025 [79][47][48].
  - AMD MI325X: announced Jun 2024; production Q4’24; broad availability Q1’25. MI350 series announced 2025 (2H availability) [75][76].
  - Intel Gaudi 3: announced Apr 9, 2024; OAM/PCIe forms (600W PCIe) [77].
  - AWS Trainium2: GA Dec 3, 2024; AWS-designed chips roadmap scaling (Nov 28, 2023) [44][95].

- Consumer discrete GPUs
  - NVIDIA RTX 4080 SUPER: $999, available Jan 31, 2024 [81].
  - NVIDIA RTX 5090/5080: Jan 30, 2025 launch at $1,999/$999 [82][83].
  - AMD RX 9070 XT / RX 9070: Mar 6, 2025; $599/$549 [86][87][88].
  - Intel Arc B580: Dec 13, 2024; $249 [89][90].

---

## 9) Methodology notes and caveats

- Data center unit estimates derive from system shipments (TrendForce 1.67M AI servers in 2024 with 71% GPU and 26% ASIC) and U.S. hyperscaler deployments (>5M training‑class accelerators), assuming 4.8–6.0 accelerators/server across mixed configurations. Resulting 8–10M 2024 units are labeled as estimates [3][2].
- Discrete notebook GPU units are inferred from IDC gaming PC totals and the notebook share of gaming PCs; handhelds are excluded per IDC. Revenue ranges are unit×ASP models anchored in current and historic pricing context; they are labeled estimates as public OEM ASP series are not available [62][63][64][65].
- Regional splits for consumer AIBs are not published publicly by JPR/Mercury; estimates are derived from PC sell‑in by region and gaming propensity multipliers; Steam HWS is used only as a qualitative cross-check (survey bias acknowledged) [67][68][69][62].

---

### Sources

[1] Omdia: AI data center chip market (2024 actual $123B; 2025 $207B): https://omdia.tech.informa.com/pr/2025/aug/ai-data-center-chip-market-to-hit-286bn-growth-likely-peaking-as-custom-asics-gain-ground  
[2] Dell’Oro (Dec 19, 2024): US hyperscalers set to deploy >5M training-capable accelerators in 2024: https://www.prnewswire.com/news-releases/ai-accelerator-revenues-soar-130-percent-in-3q-2024-according-to-delloro-group-302335750.html  
[3] TrendForce (Jul 17, 2024): 1.67M AI servers (71% GPU, 26% ASIC): https://www.trendforce.com/presscenter/news/20240722-12227.html  
[4] TrendForce (Jan 6, 2025): AI server market value 2024/2025 outlook: https://www.trendforce.com/presscenter/news/20250106-12433.html  
[5] Tom’s Hardware (May 2025): Omdia projects NVIDIA ~$87B data center GPU revenue in 2024: https://www.tomshardware.com/pc-components/gpus/nvidia-is-the-kingmaker-says-analyst-firm-omdia-company-projected-to-make-dollar87-billion-from-data-center-gpu-revenue-in-2024  
[6] CNBC (Jul 30, 2024): AMD: 2024 DC GPU revenue to exceed $4.5B: https://www.cnbc.com/2024/07/30/amd-earnings-report-q2-2024.html  
[7] Motley Fool transcript (Apr 30, 2024): AMD to exceed $4B in 2024 DC GPU revenue: https://www.fool.com/earnings/call-transcripts/2024/04/30/advanced-micro-devices-amd-q1-2024-earnings-call-t/  
[8] The Verge (Oct 31, 2024): Intel Gaudi won’t meet $500M 2024 goal: https://www.theverge.com/2024/10/31/24284860/intel-gaudi-wont-meet-500-million-goal  
[9] NVIDIA 10‑Q (Nov 21, 2024): Compute vs networking in Data Center: https://www.sec.gov/Archives/edgar/data/1045810/000104581024000316/nvda-20241027.htm  
[10] Converge! Digest (Aug 2024): NVIDIA DC compute/networking split: https://convergedigest.com/nvidia-hits-30-billion-in-q2-as-data-center-surges-154-yoy/  
[11] TrendForce (Jun 28, 2024): Huawei Ascend yields ~20%: https://www.trendforce.com/news/2024/06/28/news-huawei-faces-production-challenges-with-20-yield-rate-for-ai-chip/  
[12] Reuters (Jun 12, 2025): US says Huawei can’t make >200k AI chips in 2025: https://www.reuters.com/world/china/us-says-chinas-huawei-cant-make-more-than-200-000-ai-chips-2025-2025-06-12/  
[13] TrendForce (Feb 25, 2025): Huawei yields ~40%: https://www.trendforce.com/news/2025/02/25/news-china-advances-in-ai-chips-huawei-reportedly-boosts-yield-to-40-making-ascend-line-profitable/  
[14] Alphabet 10‑K (filed Feb 5, 2025): 2024 capex $52.5B: https://www.sec.gov/Archives/edgar/data/1652044/000165204425000014/goog-20241231.htm  
[15] CNBC (Jan 3, 2025): Microsoft to spend ~$80B on AI data centers in FY2025: https://www.cnbc.com/2025/01/03/microsoft-expects-to-spend-80-billion-on-ai-data-centers-in-fy-2025.html  
[16] CNBC (Feb 6, 2025): Amazon expects ~$100B capex in 2025: https://www.cnbc.com/2025/02/06/amazon-expects-to-spend-100-billion-on-capital-expenditures-in-2025.html  
[17] Meta PR (Jan 29, 2025): FY2024 results; 2025 capex $60–65B: https://investor.atmeta.com/investor-news/press-release-details/2025/Meta-Reports-Fourth-Quarter-and-Full-Year-2024-Results/default.aspx  
[18] MLCommons (Apr 2, 2025): MLPerf Inference v5.0 results: https://mlcommons.org/2025/04/mlperf-inference-v5-0-results/  
[19] MLCommons (Jun 4, 2025): MLPerf Training v5.0 results: https://mlcommons.org/2025/06/mlperf-training-v5-0-results/  
[20] NVIDIA Dev Blog (Jun 12, 2024): MLPerf Training v4.0 records: https://developer.nvidia.com/blog/nvidia-sets-new-generative-ai-performance-and-scale-records-in-mlperf-training-v4-0/  
[21] MLCommons (Mar 27, 2024): MLPerf Inference v4.0: https://mlcommons.org/2024/03/mlperf-inference-v4/  
[22] Meta Engineering (Mar 12, 2024): Building Meta’s GenAI infrastructure: https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/  
[23] CNBC (Jan 18, 2024): Zuckerberg: 350,000 H100s by end of 2024: https://www.cnbc.com/2024/01/18/mark-zuckerberg-indicates-meta-is-spending-billions-on-nvidia-ai-chips.html  
[24] Dell’Oro (Feb 6, 2025): AI accelerators to $382B by 2029: https://www.prnewswire.com/news-releases/market-for-ai-accelerators-to-reach-382-billion-by-2029-according-to-delloro-group-302369698.html  
[25] TrendForce (May 6, 2024): TSMC advanced packaging capacity fully booked; CoWoS/SoIC context: https://www.trendforce.com/news/2024/05/06/news-tsmcs-advanced-packaging-capacity-fully-booked-by-nvidia-and-amd-through-next-year/  
[26] TrendForce (Mar 13, 2024): CoWoS capacity surge: https://www.trendforce.com/news/2024/03/13/news-tsmc-reportedly-sensing-increased-orders-again-cowos-production-capacity-surges/  
[27] DIGITIMES (Oct 25, 2024): CoWoS capacity could reach ~65–75k WPM by Q4’25: https://www.digitimes.com/news/a20241025VL210/cowos-demand-packaging-2025-capacity.html  
[28] Reuters (May 2, 2024): SK hynix HBM nearly sold out for 2025: https://www.reuters.com/technology/nvidia-supplier-sk-hynix-says-hbm-chips-almost-sold-out-2025-2024-05-02/  
[29] SK hynix PR (Sep 26, 2024): Volume production of 12‑Hi HBM3E: https://news.skhynix.com/sk-hynix-begins-volume-production-of-the-world-first-12-layer-hbm3e/  
[30] Micron PR (Feb 2024): Volume production of HBM3E: https://investors.micron.com/index.php/news-releases/news-release-details/micron-commences-volume-production-industry-leading-hbm3e  
[31] Micron PR (Jan 8, 2025): $7B HBM advanced packaging facility (Singapore): https://investors.micron.com/index.php/news-releases/news-release-details/micron-breaks-ground-new-hbm-advanced-packaging-facility  
[32] NVIDIA Docs: DGX GB200 User Guide (hardware; NVL72 cooling/power): https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html  
[33] TrendForce (Jul 30, 2024): Liquid cooling penetration; NVL36/72 power: https://www.trendforce.com/presscenter/news/20240730-12232.html  
[34] Uptime Institute (2024): Global Data Center Survey: https://intelligence.uptimeinstitute.com/resource/uptime-institute-global-data-center-survey-2024  
[35] BIS (Oct 17, 2023): Strengthened restrictions (advanced computing/SME): https://www.bis.gov/press-release/commerce-strengthens-restrictions-advanced-computing-semiconductors-semiconductor  
[36] BIS (Apr 4, 2024): Clarifications to export rules (advanced computing): https://www.bis.gov/press-release/commerce-releases-clarifications-export-control-rules-restrict-prcs-access-advanced-computing  
[37] ASML (Jan 1, 2024): Partial revocation of export licenses (NXT:2050i/2100i): https://www.asml.com/en/news/press-releases/2023/statement-regarding-partial-revocation-export-license  
[38] Reuters (Jun 12, 2025): NVIDIA to stop including China forecasts amid controls: https://www.reuters.com/world/china/nvidia-stop-including-china-forecasts-amid-us-chip-export-controls-cnn-reports-2025-06-12/  
[39] CNBC (May 29, 2025): NVIDIA CEO on China export control impact: https://www.cnbc.com/2025/05/29/nvidia-earnings-ceo-jensen-huang-china-export-control-failure.html  
[40] Reuters (Apr 16, 2025): AMD flags up to $800M hit from curbs: https://www.reuters.com/technology/amd-flags-800-million-hit-new-us-curbs-chip-exports-china-2025-04-16/  
[41] CNBC (Apr 16, 2025): AMD says export restrictions could cost up to $800M: https://www.cnbc.com/2025/04/16/amd-800-million-export-us-chip-restrictions-china.html  
[42] CNBC (Jul 15, 2025): NVIDIA to resume H20 sales to China (licenses): https://www.cnbc.com/2025/07/15/nvidia-says-us-government-will-allow-it-to-resume-h20-ai-chip-sales-to-china.html  
[43] DCD (Jul 16, 2025): AMD to resume MI308 shipments to China after approval: https://www.datacenterdynamics.com/en/news/amd-to-resume-shipments-of-instinct-mi308-chips-to-china-following-department-of-commerce-approval/  
[44] AWS PR (Dec 3, 2024): Trainium2 GA: https://press.aboutamazon.com/2024/12/aws-trainium2-instances-now-generally-available  
[45] Google Cloud docs: TPU v5p (tech/pricing pages): https://cloud.google.com/tpu/docs/v5p  
[46] Google Cloud: TPU pricing: https://cloud.google.com/tpu/pricing  
[47] AWS What’s New (May 15, 2025): EC2 P6 (B200) GA: https://aws.amazon.com/about-aws/whats-new/2025/05/amazon-ec2-p6-b200-instances-nvidia-b200-gpus/  
[48] Google Cloud Blog (Jan 31, 2025): A4 VMs (B200) preview: https://cloud.google.com/blog/products/compute/introducing-a4-vms-powered-by-nvidia-b200-gpu-aka-blackwell  
[49] AMD PR (May 21, 2024): Azure ND MI300X v5 VMs GA: https://www.amd.com/en/newsroom/press-releases/2024-5-21-amd-instinct-mi300x-accelerators-power-microsoft-a.html  
[50] AMD PR (Sep 26, 2024): Oracle MI300X Supercluster available: https://www.amd.com/en/newsroom/press-releases/2024-9-26-amd-instinct-mi300x-accelerators-available-on-orac.html  
[51] U.S. DOE/NNSA (Dec 10, 2024): El Capitan fastest supercomputer: https://www.energy.gov/nnsa/articles/nnsa-and-livermore-lab-achieve-milestone-el-capitan-worlds-fastest-supercomputer  
[52] EuroHPC JU (Jun 10, 2025): JUPITER/EuroHPC update: https://eurohpc-ju.europa.eu/eurohpc-supercomputers-put-europe-forefront-global-supercomputing-2025-06-10_en  
[53] NVIDIA Newsroom: JUPITER powers Europe’s fastest supercomputer: https://nvidianews.nvidia.com/news/nvidia-powers-europes-fastest-supercomputer  
[54] JPR (Jun 6, 2024): Q1’24 AIB 8.7M: https://www.jonpeddie.com/news/shipments-of-graphics-add-in-boards-decline-in-q1-of-24-as-the-market-experiences-a-return-to-seasonality/  
[55] JPR (Sep 24, 2024): Q2’24 AIB 9.5M: https://www.jonpeddie.com/news/shipments-of-graphics-aibs-see-significant-surge-in-q2-2024/  
[56] JPR (Dec 10, 2024): Q3’24 AIB 8.1M: https://www.jonpeddie.com/news/q324-pc-graphics-add-in-board-shipments-decreased-14-5-from-last-quarter/  
[57] JPR (Mar 5, 2025): Q4’24 AIB 8.4M: https://www.jonpeddie.com/news/pc-aib-shipments-follow-seasonality-show-nominal-increase-for-q424/  
[58] Tom’s (Mar 7, 2025): Q4’24 shares (NV 82/AMD 17/Intel 1): https://www.tomshardware.com/tech-industry/amd-grabs-a-share-of-the-gpu-market-from-nvidia-as-gpu-shipments-rise-slightly-in-q4  
[59] Tom’s (Sep 3, 2025): Q2’25 AIB share ~94% NVIDIA: https://www.tomshardware.com/pc-components/gpus/nvidia-dominates-gpu-shipments-with-94-percent-share-shipment-surge-likely-caused-by-customers-getting-ahead-of-tariffs  
[60] Gfxspeak (Jun 5, 2025): Q1’25 AIB 9.2M: https://gfxspeak.com/featured/q125-pc-graphics-add-in-board-shipments-increased-8-5-from-last-quarter/  
[61] TechSpot (Sep 2, 2025): NVIDIA ~94% share; shipments surge: https://www.techspot.com/news/109313-nvidia-hits-94-gpu-market-share-overall-shipments.html  
[62] DIGITIMES (Mar 20, 2024): IDC gaming PCs +1% in 2024; notebooks drive: https://www.digitimes.com/news/a20240320PR202/gaming-pc-recovery-growth-2024-idc.html  
[63] I‑Connect007 (Sep 30, 2024): IDC Q2’24 gaming PCs 10.6M: https://iconnect007.com/article/142456/idc-finds-global-gaming-pc-market-recovered-in-q2-2024-fueled-by-notebooks-and-monitors/142453/  
[64] The Verge (Feb 25, 2025): IDC handheld shipments context (exclude from dGPU): https://www.theverge.com/pc-gaming/618709/steam-deck-3-year-anniversary-handheld-gaming-shipments-idc  
[65] Tom’s Hardware: High-end GPU pricing (RTX 4090 elevated) in late‑2024: https://www.tomshardware.com/pc-components/gpus/nvidia-rtx-4090-pricing-is-too-damn-high-while-most-other-gpus-have-held-steady-or-declined-in-past-6-months-market-analysis  
[66] NVIDIA PR (Feb 26, 2025): FY2025 Gaming revenue $11.4B: https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2025  
[67] Tom’s (Jan 15, 2025): IDC says global PCs 262.7M in 2024: https://www.tomshardware.com/tech-industry/lenovo-led-global-pc-shipment-in-2024-with-61-8-million-units-apple-is-gaining-pc-market-share-with-a-17-3-percent-growth  
[68] Canalys newsroom (Feb 2025): U.S. PC market Q4’24 (regional anchor): https://www.canalys.com/newsroom/us-pc-market-q4-2024  
[69] Canalys newsroom (Feb 2025): China PC market 2024: https://www.canalys.com/newsroom/china-pc-market-q4-2024  
[70] Reuters (Aug 13, 2024): Canalys: AI PCs ~14% of Q2’24 shipments: https://www.reuters.com/technology/artificial-intelligence/ai-pcs-made-up-14-quarterly-personal-computer-shipments-canalys-says-2024-08-13/  
[71] Tom’s (Aug 2025): IDC expects 274M PCs in 2025 (tariffs): https://www.tomshardware.com/tech-industry/idc-says-pc-shipments-will-increase-because-of-tariffs-now-expects-274-million-pcs-to-ship-in-2025  
[72] Tom’s (Aug 2025): U.S. delays GPU tariffs until Nov 2025: https://www.tomshardware.com/pc-components/gpus/us-delays-gpu-tariffs-until-november-2025  
[73] MLPerf Inference v5.0 results browser: https://docs.mlcommons.org/inference_results_v5.0/  
[74] NVIDIA: GB200 NVL72 product page: https://www.nvidia.com/en-in/data-center/gb200-nvl72/  
[75] AMD PR (Oct 10, 2024): MI325X specs/availability: https://www.amd.com/en/newsroom/press-releases/2024-10-10-amd-delivers-leadership-ai-performance-with-amd-in.html  
[76] AMD product page: MI350 series: https://www.amd.com/en/products/accelerators/instinct/mi350.html  
[77] Intel Newsroom (Apr 9, 2024): Gaudi 3 details: https://www.intel.com/content/www/us/en/newsroom/news/vision-2024-gaudi-3-ai-accelerator.html  
[78] Lenovo Press: HGX H200 141GB/700W (spec): https://lenovopress.lenovo.com/lp1944-nvidia-hgx-h200-141gb-700w-8-gpu-board  
[79] NVIDIA IR (Mar 18, 2024): Blackwell platform announcement: https://investor.nvidia.com/news/press-release-details/2024/NVIDIA-Blackwell-Platform-Arrives-to-Power-a-New-Era-of-Computing/default.aspx  
[80] NVIDIA Dev Blog: GB200 NVL72 training/inference claims: https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/  
[81] TechPowerUp (Jan 31, 2024): RTX 4080 SUPER $999: https://www.techpowerup.com/318529/nvidia-geforce-rtx-4080-super-starts-selling-at-usd-999  
[82] TechPowerUp DB: RTX 5090 (MSRP/specs): https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216  
[83] TechPowerUp DB: RTX 5080 (MSRP/specs): https://www.techpowerup.com/gpu-specs/geforce-rtx-5080.c4217  
[84] GamersNexus: RTX 5090 FE review (perf/power): https://gamersnexus.net/gpus/nvidia-geforce-rtx-5090-founders-edition-review-benchmarks-gaming-thermals-power  
[85] The Verge: RTX 5080 review: https://www.theverge.com/2025/1/23/598045/nvidia-geforce-rtx-5080-review-a-new-king-of-4k-is-here  
[86] The Verge (Mar 2025): RX 9070 XT price/release: https://www.theverge.com/news/621339/amd-radeon-9070-xt-price-release-date-gpu  
[87] TechPowerUp DB: RX 9070 XT: https://www.techpowerup.com/gpu-specs/radeon-rx-9070-xt.c4229  
[88] TechPowerUp DB: RX 9070: https://www.techpowerup.com/gpu-specs/radeon-rx-9070.c4250  
[89] Intel: Arc B580 specifications: https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html  
[90] TechPowerUp: Arc B580 review: https://www.techpowerup.com/review/intel-arc-b580/  
[91] MLCommons (Aug 28, 2024): MLPerf Inference v4.1 (H200 available results): https://mlcommons.org/2024/08/mlperf-inference-v4-1/  
[92] NVIDIA Dev Blog (Apr 2, 2025): Inference v5.0 Blackwell gains: https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/  
[93] AMD ROCm Blog: MI325X MLPerf Inference v5.0 entry info: https://rocm.blogs.amd.com/artificial-intelligence/mi325x-accelerates-mlperf-inference/README.html  
[94] Intel Gaudi model performance (Inference v4.0): https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance-inference.html  
[95] AWS PR (Nov 28, 2023): Next‑gen AWS‑designed chips (Trainium2/Inferentia2): https://press.aboutamazon.com/2023/11/aws-unveils-next-generation-aws-designed-chips