# Top Smartphones as of August 2025 — Transparent Composite Scoring Framework and Provisional Dataset

## Executive Summary
- Goal: Identify and rank the top smartphones as of 31 Aug 2025 using a composite score that combines (a) tested battery life in hours, (b) main rear wide camera megapixels, and (c) retail price, with lower prices scoring higher. 
- Core battery metric: GSMArena’s Battery Life Test v2.0 “Active Use Score” in hours (comparable across devices tested under v2.0) [1]. The legacy Endurance rating (v1.0) is not comparable and is not mixed into the normalized results [2].
- Camera megapixels: Main rear wide camera megapixel count from official manufacturer spec pages (or official press releases).
- Region for price: United Kingdom (GBP) for base storage, unlocked/SIM‑free units as of 31 Aug 2025. Prices will be converted to USD using a single mid‑market GBP→USD rate on 2025‑08‑31 (XE/OANDA) and archived for verifiability.
- Status: Battery life and camera megapixel source links are compiled for 16 models with GSMArena v2.0 Active Use hours. UK price collection (live URLs, August 2025 Wayback snapshots, conversions) is pending; therefore, the final composite score ranking cannot yet be computed. A provisional battery‑only leaderboard is provided for transparency.

## Methods

### Scope and inclusion criteria
- Devices: Any smartphone model available new as of 31 Aug 2025, any OS/brand/form factor. Variants are treated as separate models only if battery life, main‑wide camera MP, or base price differ materially.
- Battery metric: GSMArena Battery Life Test v2.0 “Active Use Score” in hours, a combined figure derived from 4G VoLTE calls, Wi‑Fi web browsing, YouTube streaming, and gaming at 200 nits (Wi‑Fi on for screen‑on tests; calls on 4G) [1]. Devices without v2.0 results are excluded from the composite. Legacy v1.0 Endurance hours are documented separately if a still‑sold model lacks v2.0, but are not mixed with v2.0 due to non‑comparability [2].
- Camera megapixels: Main rear wide camera megapixel count from the official manufacturer page for each model (or official global page if UK page is not available).
- Price (UK policy): New, unlocked/SIM‑free, base storage variant in GBP as of 31 Aug 2025. Preferred source is the manufacturer’s UK online store; if not available, use a reputable UK retailer (e.g., Amazon UK, Currys, John Lewis, Argos) clearly labeled. Each price will be supported by:
  - Live product URL and an Internet Archive (Wayback Machine) snapshot captured in Aug 2025,
  - Base storage confirmation and “unlocked/SIM‑free” confirmation on the product page.
- Currency: Convert GBP to USD using a single mid‑market GBP→USD rate on 2025‑08‑31 from XE or OANDA (URL and, if possible, archived capture). USD price = GBP price × GBP→USD rate.

### Scoring and normalization
- Normalize each metric to 0–100 via min–max scaling across the included devices:
  - Battery (higher is better): B_norm = 100 × (B_i − B_min) / (B_max − B_min)
  - Camera MP (higher is better): C_norm = 100 × (C_i − C_min) / (C_max − C_min)
  - Price (lower is better): P_norm = 100 × (P_max − P_i) / (P_max − P_min)
- Composite score (equal weights): Composite = (B_norm + C_norm + P_norm) / 3
- Tie‑break rules: battery hours (descending), then price in GBP (ascending), then main‑wide MP (descending).

### Handling missing data
- Battery metric missing (GSMArena v2.0 hour): exclude from composite ranking; document separately.
- Price missing: cannot compute composite; device remains in dataset flagged “price pending.”
- No imputation is used that would distort comparability. Composite rankings are published only for rows with all three metrics present and sourced.

### Dates
- Data collection and pricing reference date: As of 31 Aug 2025.
- Each battery entry lists the GSMArena review/spec page URL containing the v2.0 “Active Use Score” hours and a method note referencing GSMArena v2.0 [1].

## Provisional Battery-Life Leaderboard (GSMArena v2.0 only; not the final composite)
This is a stopgap ranking by tested battery life hours while UK prices are being captured. Each hour figure is the GSMArena Battery Life Test v2.0 “Active Use Score” from the linked review/spec page.

1. Nothing Phone (2a) Plus — 16:48h [31] | Main camera: 50MP [32]  
2. Xiaomi 15 Ultra — 16:13h [29] | Main camera: 50MP [30]  
3. Apple iPhone 15 Pro Max — 16:01h [5] | Main camera: 48MP [6]  
4. Nothing Phone (2a) — 15:53h [15] | Main camera: 50MP [16]  
5. OnePlus 12R — 14:32h [19] | Main camera: 50MP [20]  
6. OnePlus 12 — 14:11h [17] | Main camera: 50MP [18]  
7. Samsung Galaxy A55 5G — 13:27h [25] | Main camera: 50MP [26]  
8. Apple iPhone 15 — 13:20h [3] | Main camera: 48MP [4]  
9. Motorola Edge 50 Ultra — 12:56h [13] | Main camera: 50MP [14]  
10. Samsung Galaxy S24+ — 12:30h [23] | Main camera: 50MP [24]  
11. Samsung Galaxy A35 5G — 12:26h [27] | Main camera: 50MP [28]  
12. Samsung Galaxy S24 — 12:07h [21] | Main camera: 50MP [22]  
13. Google Pixel 8a — 11:25h [11] | Main camera: 64MP [12]  
14. Google Pixel 8 — 11:17h [7] | Main camera: 50MP [8]  
15. Google Pixel 8 Pro — 11:14h [9] | Main camera: 50MP [10]  

Note: Additional devices (e.g., iPhone 15 Plus, iPhone 15 Pro, Samsung Galaxy S24 Ultra, Xiaomi 14) have GSMArena v2.0 coverage but require exact hour capture or price before inclusion in composite ranking [33][35][36].

## Dataset (Provisional; price pending)
- Region for price: UK (GBP) for base storage, unlocked/SIM‑free. GBP→USD rate will be fixed to a single mid‑market rate on 2025‑08‑31 from XE or OANDA; both GBP and USD will be listed with URLs and Wayback captures (pending).
- Battery method note for all entries: GSMArena Battery Life Test v2.0 “Active Use Score” (combined calls/web/video/gaming at 200 nits; Wi‑Fi on for screen‑on tests; calls over 4G VoLTE) [1].

| Provisional Rank (battery-only) | Brand | Model (variant) | Release year | Battery metric | Battery hours (HH:MM; decimal) | Battery source URL | Battery test date (page) | Method note | Main rear wide MP | Official spec URL | UK price GBP | Price URL | Price Wayback (Aug 2025) | Base storage | Unlocked? | GBP→USD rate (2025‑08‑31) | USD price | B_norm | C_norm | P_norm | Composite score | Flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Nothing | Phone (2a) Plus | 2025 | Active Use (v2.0) | 16:48; 16.80 | [31] | 2025‑06 | GSMArena v2.0 [1] | 50 | [32] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 100.00 | 12.50 | TBD | TBD | Price pending |
| 2 | Xiaomi | 15 Ultra (global) | 2025 | Active Use (v2.0) | 16:13; 16.22 | [29] | 2025‑03 | GSMArena v2.0 [1] | 50 | [30] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 89.59 | 12.50 | TBD | TBD | Price pending |
| 3 | Apple | iPhone 15 Pro Max | 2023 | Active Use (v2.0) | 16:01; 16.02 | [5] | 2023‑09 | GSMArena v2.0 [1] | 48 | [6] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 85.96 | 0.00 | TBD | TBD | Price pending |
| 4 | Nothing | Phone (2a) | 2024 | Active Use (v2.0) | 15:53; 15.88 | [15] | 2024‑03 | GSMArena v2.0 [1] | 50 | [16] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 83.48 | 12.50 | TBD | TBD | Price pending |
| 5 | OnePlus | 12R | 2024 | Active Use (v2.0) | 14:32; 14.53 | [19] | 2024‑02 | GSMArena v2.0 [1] | 50 | [20] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 59.25 | 12.50 | TBD | TBD | Price pending |
| 6 | OnePlus | 12 | 2023/2024 | Active Use (v2.0) | 14:11; 14.18 | [17] | 2024‑01 | GSMArena v2.0 [1] | 50 | [18] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 52.98 | 12.50 | TBD | TBD | Price pending |
| 7 | Samsung | Galaxy A55 5G | 2024 | Active Use (v2.0) | 13:27; 13.45 | [25] | 2024‑03 | GSMArena v2.0 [1] | 50 | [26] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 39.89 | 12.50 | TBD | TBD | Price pending |
| 8 | Apple | iPhone 15 | 2023 | Active Use (v2.0) | 13:20; 13.33 | [3] | 2023‑09 | GSMArena v2.0 [1] | 48 | [4] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 37.70 | 0.00 | TBD | TBD | Price pending |
| 9 | Motorola | Edge 50 Ultra | 2024 | Active Use (v2.0) | 12:56; 12.93 | [13] | 2024‑06 | GSMArena v2.0 [1] | 50 | [14] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 30.59 | 12.50 | TBD | TBD | Price pending |
| 10 | Samsung | Galaxy S24+ | 2024 | Active Use (v2.0) | 12:30; 12.50 | [23] | 2024‑01 | GSMArena v2.0 [1] | 50 | [24] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 22.81 | 12.50 | TBD | TBD | Price pending |
| 11 | Samsung | Galaxy A35 5G | 2024 | Active Use (v2.0) | 12:26; 12.43 | [27] | 2024‑03 | GSMArena v2.0 [1] | 50 | [28] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 21.54 | 12.50 | TBD | TBD | Price pending |
| 12 | Samsung | Galaxy S24 | 2024 | Active Use (v2.0) | 12:07; 12.12 | [21] | 2024‑01 | GSMArena v2.0 [1] | 50 | [22] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 15.98 | 12.50 | TBD | TBD | Price pending |
| 13 | Google | Pixel 8a | 2024 | Active Use (v2.0) | 11:25; 11.42 | [11] | 2024‑05 | GSMArena v2.0 [1] | 64 | [12] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 3.41 | 100.00 | TBD | TBD | Price pending |
| 14 | Google | Pixel 8 | 2023 | Active Use (v2.0) | 11:17; 11.28 | [7] | 2023‑10 | GSMArena v2.0 [1] | 50 | [8] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 0.90 | 12.50 | TBD | TBD | Price pending |
| 15 | Google | Pixel 8 Pro | 2023 | Active Use (v2.0) | 11:14; 11.23 | [9] | 2023‑10 | GSMArena v2.0 [1] | 50 | [10] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 0.00 | 12.50 | TBD | TBD | Price pending |

Additional models with GSMArena v2.0 coverage identified for inclusion once exact hours/prices are captured:
- Apple iPhone 15 Plus — battery hour pending capture from GSMArena review [33]; main camera 48MP (Apple spec) [4].
- Apple iPhone 15 Pro — battery hour pending capture [33]; main camera 48MP (Apple spec) [6].
- Samsung Galaxy S24 Ultra — GSMArena v2.0 page with battery section; exact “Active Use” hour pending capture [35]; main camera 200MP (Samsung spec) [36].
- Xiaomi 14 — GSMArena v2.0 states “almost 14h”; exact HH:MM pending capture [33]; main camera 50MP (Xiaomi UK spec) [34].

## Machine‑Readable CSV (Provisional; price fields pending)
Fields:
- brand, model_variant, release_year, battery_metric, battery_hours_hhmm, battery_hours_decimal, battery_source_url, battery_test_date, battery_method_note, main_wide_mp, official_spec_url, uk_price_gbp, price_url, price_wayback_url, base_storage_gb, unlocked_confirmed, gbp_usd_rate_2025_08_31, usd_price, b_norm, c_norm, p_norm, composite_score, flags

```
brand,model_variant,release_year,battery_metric,battery_hours_hhmm,battery_hours_decimal,battery_source_url,battery_test_date,battery_method_note,main_wide_mp,official_spec_url,uk_price_gbp,price_url,price_wayback_url,base_storage_gb,unlocked_confirmed,gbp_usd_rate_2025_08_31,usd_price,b_norm,c_norm,p_norm,composite_score,flags
Nothing,Phone (2a) Plus,2025,Active Use v2.0,"16:48",16.80,https://www.gsmarena.com/nothing_phone_2a_plus-review-2734p3.php,2025-06,"GSMArena Battery Life Test v2.0; calls/web/video/gaming @200 nits; Wi‑Fi on for screen-on; calls over 4G VoLTE (see method).",50,https://nothing.tech/pages/phone-2a-plus,,,,,,"",,100.00,12.50,,,"price pending"
Xiaomi,15 Ultra (global),2025,Active Use v2.0,"16:13",16.22,https://www.gsmarena.com/xiaomi_15_ultra-review-2802p3.php,2025-03,"GSMArena Battery Life Test v2.0; see method.",50,https://www.mi.com/global/product/xiaomi-15-ultra/,,,,,,"",,89.59,12.50,,,"price pending"
Apple,iPhone 15 Pro Max,2023,Active Use v2.0,"16:01",16.02,https://www.gsmarena.com/apple_iphone_15_pro_max-review-2618.php,2023-09,"GSMArena Battery Life Test v2.0; see method.",48,https://www.apple.com/iphone-15-pro/specs/,,,,,,"",,85.96,0.00,,,"price pending"
Nothing,Phone (2a),2024,Active Use v2.0,"15:53",15.88,https://www.gsmarena.com/nothing_phone_2a-review-2681p3.php,2024-03,"GSMArena Battery Life Test v2.0; see method.",50,https://nothing.tech/pages/phone-2a,,,,,,"",,83.48,12.50,,,"price pending"
OnePlus,12R,2024,Active Use v2.0,"14:32",14.53,https://www.gsmarena.com/oneplus_12r-12727.php,2024-02,"GSMArena Battery Life Test v2.0; see method.",50,https://www.oneplus.com/uk/oneplus-12r/specs,,,,,,"",,59.25,12.50,,,"price pending"
OnePlus,12,2023,Active Use v2.0,"14:11",14.18,https://www.gsmarena.com/oneplus_12-review-2659p3.php,2024-01,"GSMArena Battery Life Test v2.0; see method.",50,https://www.oneplus.com/uk/oneplus-12/specs,,,,,,"",,52.98,12.50,,,"price pending"
Samsung,Galaxy A55 5G,2024,Active Use v2.0,"13:27",13.45,https://www.gsmarena.com/samsung_galaxy_a55-review-2684p3.php,2024-03,"GSMArena Battery Life Test v2.0; see method.",50,https://www.samsung.com/uk/smartphones/galaxy-a55-5g/specs/,,,,,,"",,39.89,12.50,,,"price pending"
Apple,iPhone 15,2023,Active Use v2.0,"13:20",13.33,https://www.gsmarena.com/apple_iphone_15-review-2619p3.php,2023-09,"GSMArena Battery Life Test v2.0; see method.",48,https://www.apple.com/iphone-15/specs/,,,,,,"",,37.70,0.00,,,"price pending"
Motorola,Edge 50 Ultra,2024,Active Use v2.0,"12:56",12.93,https://www.gsmarena.com/motorola_edge_50_ultra-review-2708p3.php,2024-06,"GSMArena Battery Life Test v2.0; see method.",50,https://www.motorola.co.uk/smartphones-motorola-edge-50-ultra/p,,,,,,"",,30.59,12.50,,,"price pending"
Samsung,Galaxy S24+,2024,Active Use v2.0,"12:30",12.50,https://www.gsmarena.com/samsung_galaxy_s24_plus-review-2664p3.php,2024-01,"GSMArena Battery Life Test v2.0; see method.",50,https://www.samsung.com/uk/smartphones/galaxy-s24-plus/specs/,,,,,,"",,22.81,12.50,,,"price pending"
Samsung,Galaxy A35 5G,2024,Active Use v2.0,"12:26",12.43,https://www.gsmarena.com/samsung_galaxy_a35-review-2682p3.php,2024-03,"GSMArena Battery Life Test v2.0; see method.",50,https://www.samsung.com/uk/smartphones/galaxy-a35-5g/specs/,,,,,,"",,21.54,12.50,,,"price pending"
Samsung,Galaxy S24,2024,Active Use v2.0,"12:07",12.12,https://www.gsmarena.com/samsung_galaxy_s24-review-2663p3.php,2024-01,"GSMArena Battery Life Test v2.0; see method.",50,https://www.samsung.com/uk/smartphones/galaxy-s24/specs/,,,,,,"",,15.98,12.50,,,"price pending"
Google,Pixel 8a,2024,Active Use v2.0,"11:25",11.42,https://www.gsmarena.com/google_pixel_8a-review-2705p3.php,2024-05,"GSMArena Battery Life Test v2.0; see method.",64,https://store.google.com/gb/product/pixel_8a_specs,,,,,,"",,3.41,100.00,,,"price pending"
Google,Pixel 8,2023,Active Use v2.0,"11:17",11.28,https://www.gsmarena.com/google_pixel_8-12546.php,2023-10,"GSMArena Battery Life Test v2.0; see method.",50,https://store.google.com/gb/product/pixel_8_specs,,,,,,"",,0.90,12.50,,,"price pending"
Google,Pixel 8 Pro,2023,Active Use v2.0,"11:14",11.23,https://www.gsmarena.com/google_pixel_8_pro-12545.php,2023-10,"GSMArena Battery Life Test v2.0; see method.",50,https://store.google.com/gb/product/pixel_8_pro_specs,,,,,,"",,0.00,12.50,,,"price pending"
```

## What’s required to finalize the Top‑25 composite ranking
1) Capture UK prices (GBP) as of 31 Aug 2025 for base storage, unlocked/SIM‑free:
- Manufacturer UK store pages where possible; otherwise, reputable UK retailers.
- For each device: live URL, August 2025 Wayback snapshot URL (with capture date), and confirmation of base storage and unlocked status on the page.

2) Fix a single GBP→USD mid‑market rate on 2025‑08‑31 (XE or OANDA), record the exact rate with link (and archive if possible), and apply USD_price = GBP_price × rate to all rows.

3) Expand the device pool to at least 25 models with GSMArena v2.0 battery hours and UK pricing, including:
- Apple: iPhone 15 Plus [33], iPhone 15 Pro [33]
- Samsung: Galaxy S24 Ultra [35][36]
- Xiaomi: Xiaomi 14 (exact v2.0 hour) [33][34]
- Additional UK‑available models with GSMArena v2.0 battery results (e.g., OnePlus Nord series with v2.0, Nothing Phone (2) if still sold, Sony Xperia 1 VI/10 VI, Asus Zenfone/ROG 8 series, Honor Magic6 Pro, Motorola Edge 50 family variants), contingent on verifiable UK price pages.

4) Compute normalized metrics (B_norm, C_norm, P_norm) and the Composite score per formula; publish the final ranked Top 25 with full links and archived price evidence.

## Assumptions and limitations
- GSMArena v2.0 “Active Use Score” is a lab composite under controlled conditions (200 nits; specific mixes of calls/web/video/gaming) and is comparable within v2.0 only. Real‑world results vary with networks, software versions, and use patterns [1].
- Camera megapixel count captures only the main rear wide sensor as specified on the official product page; it is not a proxy for camera quality.
- Pricing uses UK market references and a single exchange rate date for consistent USD conversions. Market prices can vary by retailer, promotions, and time; archived pages provide verifiability on the 31 Aug 2025 snapshot.
- Models without complete, comparable battery data or price are not ranked in the composite to avoid mixing methodologies or introducing bias via imputation.

## Formulas (for reproducibility)
- Battery normalization: B_norm = 100 × (B_i − B_min) / (B_max − B_min)
- Camera megapixels normalization: C_norm = 100 × (C_i − C_min) / (C_max − C_min)
- Price normalization (invert): P_norm = 100 × (P_max − P_i) / (P_max − P_min)
- Composite score (equal weights): Composite = (B_norm + C_norm + P_norm) / 3
- Currency conversion: USD_price = GBP_price × GBP→USD_mid_market_rate_on_2025‑08‑31

## Data collection date
- As of 31 Aug 2025.

### Sources
[1] GSMArena: How we test battery life (Battery life test v2.0) — https://www.gsmarena.com/how_we_test_gsmarena_battery_life_test_v2-news-60429.php  
[2] GSMArena Lab tests – legacy Endurance rating explainer — https://fo.gsmarena.com/gsmarena_lab_tests-review-751p6.php  
[3] GSMArena: Apple iPhone 15 review – Battery (Active Use score 13:20h) — https://www.gsmarena.com/apple_iphone_15-review-2619p3.php  
[4] Apple: iPhone 15 – Tech Specs (main wide camera 48MP) — https://www.apple.com/iphone-15/specs/  
[5] GSMArena: Apple iPhone 15 Pro Max review – Battery (Active Use score 16:01h) — https://www.gsmarena.com/apple_iphone_15_pro_max-review-2618.php  
[6] Apple: iPhone 15 Pro – Tech Specs (main wide camera 48MP) — https://www.apple.com/iphone-15-pro/specs/  
[7] GSMArena: Google Pixel 8 specs (Active Use score 11:17h) — https://www.gsmarena.com/google_pixel_8-12546.php  
[8] Google Store UK: Pixel 8 – Tech Specs (main rear camera 50MP) — https://store.google.com/gb/product/pixel_8_specs  
[9] GSMArena: Google Pixel 8 Pro specs (Active Use score 11:14h) — https://www.gsmarena.com/google_pixel_8_pro-12545.php  
[10] Google Store UK: Pixel 8 Pro – Tech Specs (main rear camera 50MP) — https://store.google.com/gb/product/pixel_8_pro_specs  
[11] GSMArena: Google Pixel 8a review – Battery (Active Use score 11:25h) — https://www.gsmarena.com/google_pixel_8a-review-2705p3.php  
[12] Google Store UK: Pixel 8a – Tech Specs (main rear camera 64MP) — https://store.google.com/gb/product/pixel_8a_specs  
[13] GSMArena: Motorola Edge 50 Ultra review – Battery (Active Use score 12:56h) — https://www.gsmarena.com/motorola_edge_50_ultra-review-2708p3.php  
[14] Motorola UK: edge 50 ultra — https://www.motorola.co.uk/smartphones-motorola-edge-50-ultra/p  
[15] GSMArena: Nothing Phone (2a) review – Battery (Active Use score 15:53h) — https://www.gsmarena.com/nothing_phone_2a-review-2681p3.php  
[16] Nothing: Phone (2a) — https://nothing.tech/pages/phone-2a  
[17] GSMArena: OnePlus 12 review – Battery (Active Use score 14:11h) — https://www.gsmarena.com/oneplus_12-review-2659p3.php  
[18] OnePlus UK: OnePlus 12 – Specs (main rear camera 50MP) — https://www.oneplus.com/uk/oneplus-12/specs  
[19] GSMArena: OnePlus 12R specs (Active Use score 14:32h) — https://www.gsmarena.com/oneplus_12r-12727.php  
[20] OnePlus UK: OnePlus 12R – Specs (main rear camera 50MP) — https://www.oneplus.com/uk/oneplus-12r/specs  
[21] GSMArena: Samsung Galaxy S24 review – Battery (Active Use score 12:07h) — https://www.gsmarena.com/samsung_galaxy_s24-review-2663p3.php  
[22] Samsung UK: Galaxy S24 – Tech Specs (main rear camera 50MP) — https://www.samsung.com/uk/smartphones/galaxy-s24/specs/  
[23] GSMArena: Samsung Galaxy S24+ review – Battery (Active Use score 12:30h) — https://www.gsmarena.com/samsung_galaxy_s24_plus-review-2664p3.php  
[24] Samsung UK: Galaxy S24+ – Tech Specs (main rear camera 50MP) — https://www.samsung.com/uk/smartphones/galaxy-s24-plus/specs/  
[25] GSMArena: Samsung Galaxy A55 review – Battery (Active Use score 13:27h) — https://www.gsmarena.com/samsung_galaxy_a55-review-2684p3.php  
[26] Samsung UK: Galaxy A55 5G – Tech Specs (main rear camera 50MP) — https://www.samsung.com/uk/smartphones/galaxy-a55-5g/specs/  
[27] GSMArena: Samsung Galaxy A35 review – Battery (Active Use score 12:26h) — https://www.gsmarena.com/samsung_galaxy_a35-review-2682p3.php  
[28] Samsung UK: Galaxy A35 5G – Tech Specs (main rear camera 50MP) — https://www.samsung.com/uk/smartphones/galaxy-a35-5g/specs/  
[29] GSMArena: Xiaomi 15 Ultra review – Battery (Active Use score 16:13h) — https://www.gsmarena.com/xiaomi_15_ultra-review-2802p3.php  
[30] Xiaomi Global: Xiaomi 15 Ultra — https://www.mi.com/global/product/xiaomi-15-ultra/  
[31] GSMArena: Nothing Phone (2a) Plus review – Battery (Active Use score 16:48h) — https://www.gsmarena.com/nothing_phone_2a_plus-review-2734p3.php  
[32] Nothing: Phone (2a) Plus — https://nothing.tech/pages/phone-2a-plus  
[33] GSMArena: Xiaomi 14 review – Battery (Active Use “almost 14h”; exact HH:MM pending) — https://www.gsmarena.com/xiaomi_14-review-2672p3.php  
[34] Xiaomi UK: Xiaomi 14 — https://www.mi.com/uk/product/xiaomi-14/  
[35] GSMArena: Samsung Galaxy S24 Ultra review – Battery (Active Use; exact HH:MM pending) — https://www.gsmarena.com/samsung_galaxy_s24_ultra-review-2670p3.php  
[36] Samsung UK: Galaxy S24 Ultra – Tech Specs (main rear camera 200MP) — https://www.samsung.com/uk/smartphones/galaxy-s24-ultra/specs/