# 2025 US SMB Online Payments: Comparative Analysis (Online/E‑commerce)

## Scope, definitions, and how to read this comparison
- Scope: Online/e‑commerce use cases for US‑based small/medium businesses (SMBs) in 2025. In‑person/POS is excluded unless it directly affects online checkout.
- Providers: Stripe, PayPal, Square, Braintree (PayPal), Adyen, Shopify Payments, 2Checkout (Verifone), Amazon Pay, Apple Pay.
- “Domestic” vs “International” (PayPal definition): Domestic = both sender and receiver are registered in the same market; International = registered in different markets [1].
- Important dependency note for wallets: Apple Pay and Google Pay in online checkout are typically offered through a payment service provider (PSP)/gateway. For example, Square processes Apple Pay/Google Pay at the merchant’s standard online card rate [19][37][38]; PayPal offers Apple Pay/Google Pay as alternative payment methods within PayPal Checkout for eligible merchants and buyer contexts [12][13].
- Data status: All facts below are backed by a direct URL and access date. For providers/dimensions where 2025‑current, primary documentation could not be verified in this research pass, items are marked as “Unknown” rather than estimated.

## Executive summary (what an SMB should know first)
- Square offers a straightforward, blended online card rate (2.9% + $0.30) via APIs/SDKs; ACH is 1% ($1 min, $5 max); Payment Links are 3.3% + $0.30; Afterpay is 6% + $0.30 online. Multi‑currency presentment is not supported from a US account; transactions are in USD and buyer banks handle conversion. No PCI or dispute management fees; instant transfers cost 1.75% [19][20][21][23][24][26][28][29][34].
- PayPal exposes multiple flows: hosted PayPal Checkout (wallet, Venmo US, Pay Later, guest checkout) and embedded Advanced/Expanded Cards. Standard US fees: PayPal Checkout and Venmo 3.49% + fixed fee (USD fixed fee = $0.49); PayPal Pay Later 4.99% + fixed; standalone domestic card payments 2.99% + fixed; Advanced Cards 2.89% + fixed (or interchange++ option: interchange pass‑through + 0.49% + fixed). International commercial transactions commonly add 1.50%. ACH services (if enabled) 0.80% capped at $5. Card chargeback fee $20 (for card chargebacks not processed through a PayPal account). Refunds don’t return original fees [1].
- Apple Pay is a wallet, not a standalone processor. Fees and availability depend on the underlying PSP/gateway. Example: Square processes Apple Pay at standard online rates [19][37]. PayPal offers Apple Pay as an APM for eligible merchants/contexts [13].

Note: Detailed, source‑cited coverage is included for PayPal and Square. For Stripe, Braintree, Adyen, Shopify Payments, 2Checkout (Verifone), Amazon Pay, and Apple Pay (standalone), key items are listed as “Unknown” where 2025‑specific, primary documentation was not verified within this pass.

## Provider-by-provider analysis (US online/e‑commerce)

### Stripe
- Overall 2025 US SMB online facts (pricing, features, compliance, support, etc.): Unknown
- E‑commerce integration capabilities: Unknown
- US online pricing and fees: Unknown
- International and multi‑currency (presentment/settlement): Unknown
- Setup and PCI scope: Unknown
- Supported online payment methods: Unknown
- Fraud/security and dispute tooling: Unknown
- SMB support channels/hours: Unknown
- Scalability (rate limits, uptime/SLA, data export, subscriptions/marketplaces): Unknown

### PayPal
1) E‑commerce integration capabilities
- Hosted checkout: PayPal Checkout (Standard) integrates via the PayPal JS SDK buttons, presenting PayPal, Pay Later, Venmo (US), and debit/credit card options; described as a quick start for merchants [3].
- Embedded/Advanced cards: Advanced/Expanded Checkout offers embedded card fields (Card Fields/Hosted Fields) with merchant‑controlled UI and built‑in 3D Secure handling instructions; eligibility and supported currencies are documented [4][5][7][14][15].
- Webhooks/events: Webhooks API is provided; PayPal recommends webhooks and discourages polling to avoid rate limiting [9][10].
- Sandbox/testing & developer tools: Sandbox with negative testing and card generator; developer home and docs for Orders v2, APMs, and SDKs [2][11][12][13].
- APMs (wallets/local methods): Apple Pay and Google Pay supported as APMs (eligibility and buyer device constraints apply); Venmo via Checkout is US‑only [12][13][16][17].

2) US transaction fees and pricing (online)
- US Merchant Fees page (last updated Aug 18, 2025) includes:
  - PayPal Checkout & Guest Checkout: 3.49% + fixed fee (USD fixed fee $0.49) [1].
  - Pay with Venmo (US): 3.49% + fixed fee [1].
  - PayPal Pay Later (U.S.): 4.99% + fixed fee [1].
  - Standard Credit and Debit Card Payments (standalone): 2.99% + fixed fee [1].
  - Advanced Credit and Debit Card Payments (ACDC): 2.89% + fixed fee; or Interchange Pass‑through + 0.49% + fixed fee [1].
  - ACH Services (if enabled within Online Payment Services): 0.80% capped at $5 [1].
  - International commercial transactions: generally +1.50% [1].
  - Card verification transactions: $0.30 each (ACDC) [1].
  - Monthly/platform fees: ACDC and ACH Services no monthly fee; Payments Advanced $5; Payments Pro $30; Pro Payflow $30; Virtual Terminal $30; optional Recurring Billing $10/month [1].
  - Chargeback and dispute fees: card chargebacks (when not processed through a PayPal account) $20; PayPal dispute fees Standard $15 / High Volume $30 [1].
  - Refunds: refunding a commercial or invoicing transaction does not return the original fees [1].

3) International payment support and multi‑currency
- Presentment currencies: supported currency codes documented; includes zero‑decimal currency notes [29].
- Advanced Cards availability: country and currency eligibility lists (≈36–37 countries, 22 currencies) [14][15].
- FX services: FX as a Service allows holding funds in multiple currencies and converting to a preferred currency; reference documentation provides details and constraints by currency/country [30][31].
- Currency conversion spreads: 4.00% for some scenarios (e.g., paying for goods in a different currency), 3.00% for others (or as disclosed) [1].
- APMs by market: supported APMs matrix (e.g., iDEAL, Sofort, Bancontact, BLIK, Trustly); availability depends on buyer country and setup [17].

4) Ease of setup and technical requirements
- Onboarding/KYC/underwriting timelines for US SMBs: Unknown (not published as a standard timeline in the cited docs).
- PCI scope: PayPal states Card Fields is a PCI DSS service provider; using Card Fields reduces PCI scope when collecting card information. Specific SAQ type mapping is not publicly stated in cited docs [4]. PayPal highlights PCI compliance for its tokenization/vault/orchestration services [34].

5) Supported online payment methods
- Methods via hosted checkout: PayPal wallet, PayPal Pay Later, Venmo (US), guest card checkout [3][16][1].
- Advanced/embedded cards: major cards (availability by country/currency) [4][14][15].
- Apple Pay / Google Pay: offered via PayPal as APMs (eligibility and buyer device constraints apply) [13][12].
- ACH: ACH Services available within Online Payment Services (fees above) [1].
- Local/regional APMs: selection per supported APMs matrix [17].

6) Fraud protection and security
- PCI/security posture: PayPal states full PCI compliance; tokenization/vault/orchestration services reduce PCI burden; security/monitoring highlighted on consumer security page [34][35].
- AVS/CVV: Advanced Cards exposes AVS/CVV responses; testing guidance available [36][37].
- 3D Secure 2: Implementation guides for Card Fields/Hosted Fields; standard Checkout handles 3DS where required [7].
- Fraud tools: Fraud Protection and Fraud Protection Advanced are available; Chargeback Protection add‑ons for ACDC at 0.40% or 0.60% waive chargeback fees for eligible categories once info is submitted [38][39][40].
- Seller Protection (wallet transactions): legal program terms and requirements (including intangible goods specifics) [41].

7) Customer support for SMBs
- Channels/hours (US): Phone support noted as every day 8 AM–8 PM CST; Message Center agents Mon–Fri 7 AM–10 PM CST and Sat–Sun 8 AM–8 PM CST; merchant technical support portal available for technical issues [45][46][47].
- Developer support and docs: central developer support entry point [48].
- Published SLAs/dedicated AM for SMBs: Unknown (not publicly stated on the cited SMB pages).

8) Scalability
- API rate limiting: PayPal does not publish a fixed global policy; rate‑limit guidelines recommend webhooks and token caching; Payouts testing docs reiterate lack of a published global policy [10][49].
- Uptime/status: public status page is provided [50].
- Data portability/reporting: Activity Download (CSV/PDF/TAB/QuickBooks/Quicken) with file record limits; Transaction Search API (up to 3 years); SFTP reports (e.g., Transaction Detail Report, Payouts Reconciliation, Attempts & Declines) [51][52][53][54][55][56][57].
- Subscriptions/recurring: Business Subscriptions page outlines recurring via PayPal; fees defer to Merchant Fees page [58][1].
- Marketplaces/split payments: multiparty/platform documentation available via developer site [2].

### Square
1) E‑commerce integration capabilities
- Hosted checkout: Checkout API creates a Square‑hosted checkout page (square.link URL). No‑code Payment Links available via Dashboard [1][2][8][65][66][67].
- Embedded checkout: Web Payments SDK provides PCI‑compliant payment inputs for browser‑based checkout; supports cards, ACH (US), Apple Pay, Google Pay, Afterpay (regions), Cash App Pay (US), Square gift cards [3][39][40].
- Server‑side processing: Payments API handles authorization, delayed capture (complete) and cancel [4][5][36][84].
- Headless: Developers can build custom storefronts using Web Payments SDK + Payments API [3][4].
- Webhooks: Payments events (payment.created/updated) documented with retry behavior (up to 24 hours, exponential backoff) [12][57][70][71].
- Sandbox/testing and tools: Full sandbox, test cards, SCA test flows; API Explorer; SDKs; Web Payments SDK quickstart [14][15][16][17][18][58].
- Constraints/caveats: Cash App Pay is US‑only; ACH is US‑only; Afterpay online acceptance available in US/CA/AU/UK (Clearpay in UK) with online order size $1–$2,000 on Square Online; presentment in account currency only (no multi‑currency presentment) [77][20][21][22][23][24].

2) US transaction fees and pricing (online)
- Online cards via APIs/SDK: 2.9% + $0.30 per transaction [19].
- Square Online plan‑based rates: Free/Plus 2.9% + $0.30; Premium 2.6% + $0.30 [25].
- Payment Links / Online Checkout: 3.3% + $0.30 [26][65][66][67].
- Invoices online: card 3.3% + $0.30 (Invoices Free) or 2.9% + $0.30 (Invoices Plus); ACH via Invoices 1% ($1 min, Plus adds $10 cap) [26].
- ACH bank transfer via Web Payments SDK: 1% fee with $1 minimum and $5 maximum; no chargeback or additional fees (US‑only) [20][19].
- Wallets (Apple Pay, Google Pay) and Cash App Pay: processed at your online card rate; Cash App Pay US‑only and “same rate as cards” [19][37][38][39][77].
- Afterpay online: 6% + $0.30 per order (US), with order range $1–$2,000 on Square Online [21][26].
- Refunds: processing fees are not returned on refunds [28].
- Disputes: no fees for dispute management; Square covers the fee for every dispute contested together [29].
- PCI fees: none; PCI compliance included [30][31][41].
- Payouts: next‑business‑day transfers free; instant/same‑day transfers 1.75% [32][33][34].
- Pricing model: blended rates published (no public interchange++ for SMB online) [19].
- Volume/custom pricing: contact sales if annual sales exceed $250,000 [26][35][83].

3) International payment support and multi‑currency
- Cross‑border acceptance: accepts most internationally issued cards for online; transactions are processed in the account currency (USD for US accounts) [23][24].
- Multi‑currency presentment/DCC: not supported; buyer’s issuing bank handles conversion [23][24].
- Local APMs for cross‑border: methods vary by country; for US, methods include cards, Apple Pay/Google Pay, Cash App Pay (US), Afterpay (in specified regions); no broader local APM set is documented for US merchants beyond these [3][69].

4) Ease of setup and technical requirements
- Onboarding/KYB/KYC: identity and business details required; Square may request additional verification (e.g., selfie + government ID); pending activations can take 1–2 business days [47].
- PCI scope and SDK requirements: Square is PCI‑DSS compliant; Web Payments SDK provides PCI‑compliant inputs and tokens; secure contexts/CSP required in 2025 [30][31][3].

5) Supported online payment methods
- Cards: Visa, Mastercard, American Express, Discover; accepts most internationally issued cards [23][69].
- ACH: US‑only via Web Payments SDK [20][69].
- Wallets: Apple Pay (regions where Square operates), Google Pay (all except Japan), Cash App Pay (US‑only) [37][38][77][69].
- BNPL: Afterpay (US/CA/AU/UK; Clearpay in UK) with online order size $1–$2,000 on Square Online [21][22].
- Gift cards: Square gift cards supported; “no fee on redemption”; 2.5% load fee applies when loading gift cards [40][19].

6) Fraud protection and security
- PCI/security posture: Square states Level 1 PCI DSS compliance, end‑to‑end encryption, tokenization; no PCI fees for merchants using Square to store/process/transmit card data [31][41][73][30][74].
- Risk tools: Risk Manager with rules for AVS/CVV and optional 3D Secure invocation; 3DS can shift liability when applicable; card‑on‑file API payments not eligible; Square may cover certain disputes when 3DS should have applied but did not due to external factors [42][43][44][45].

7) Customer support for SMBs
- Channels/hours (US): phone support Mon–Fri 6 AM–6 PM PT; chat Mon–Fri 8 AM–8 PM CT; email responses typically within 24–48 hours; certain subscriptions/hardware include 24/7 phone support; US contact entries are published in Support Center [46][47][48].
- Developer resources: docs, API Explorer, forums, SDKs [49][16][18].

8) Scalability
- Rate limits: Square does not publish numeric REST rate limits; apps must handle HTTP 429 with backoff; GraphQL limits are documented (10 QPS, depth 10, complexity 250) [51][50][52].
- Status/uptime: public status page available [53].
- Data export/portability: Dashboard export to CSV for various reports [54].
- Subscriptions/recurring: Subscriptions API (plan creation, cycles, pause/resume); Checkout API can sell subscription plans with constraints (e.g., one paid phase); ACH not available for Subscriptions API [6][22][76].
- Marketplaces/platform fees: application fee (platform fee) supported; cannot split app fee across multiple accounts [55][85].
- Multi‑entity/cross‑region: must process in country of activation; cannot process in multiple countries/currencies with one account [23][79].

### Braintree (a PayPal service)
- Overall 2025 US SMB online facts (pricing, features, compliance, support, etc.): Unknown (no 2025‑verified primary sources included in this pass).
- Integration capabilities: Unknown
- US online pricing and fees: Unknown
- International/multi‑currency: Unknown
- Setup/PCI: Unknown
- Methods: Unknown
- Fraud/security: Unknown
- Support: Unknown
- Scalability: Unknown

### Adyen
- All dimensions for 2025 US SMB online: Unknown

### Shopify Payments
- All dimensions for 2025 US SMB online: Unknown

### 2Checkout (Verifone)
- All dimensions for 2025 US SMB online: Unknown

### Amazon Pay
- All dimensions for 2025 US SMB online: Unknown

### Apple Pay (as a payment method, not a processor)
- Nature of offering: Apple Pay is a digital wallet/payment method integrated via a PSP/gateway (e.g., Square, PayPal). Pricing and availability depend on the underlying processor.
  - Example: Square supports Apple Pay online and charges the standard online card rate [19][37].
  - Example: PayPal offers Apple Pay as an alternative payment method with eligibility and buyer device constraints [13].
- Direct Apple Pay merchant processor pricing (standalone) for US SMB online: Unknown (outside the payment method’s scope—depends on the PSP/gateway).
- Integration, PCI scope, and support for Apple Pay outside PSP docs: Unknown in this pass.

## Cross‑provider comparison by dimension (US online/e‑commerce)

1) E‑commerce integration capabilities
- Hosted checkout vs embedded:
  - PayPal: Hosted “Checkout” (buttons) and embedded Advanced/Expanded Cards [3][4][5].
  - Square: Hosted Checkout API/Payment Links and embedded Web Payments SDK [1][8][3].
  - Others: Unknown.

- Webhooks/testing/tooling:
  - PayPal: Webhooks API, sandbox/negative testing, rate‑limit guidance [9][11][10].
  - Square: Payments webhooks and documented retry behavior; sandbox, API Explorer, SDKs [12][57][14][16][18].
  - Others: Unknown.

- Constraints/caveats:
  - Venmo via PayPal is US‑only [16].
  - Cash App Pay via Square is US‑only; ACH via Square is US‑only [77][20].
  - Apple Pay/Google Pay availability depends on PSP and buyer device/browser contexts [37][38][12][13].
  - Square does not support multi‑currency presentment from a US account (present in USD only) [23][24].

2) US online transaction fees and pricing
- Cards, wallets, ACH, BNPL:
  - PayPal: Checkout/Venmo 3.49% + $0.49 (USD fixed fee $0.49); Pay Later 4.99% + fixed; standalone cards 2.99% + fixed; Advanced Cards 2.89% + fixed or interchange++ (interchange + 0.49% + fixed); ACH 0.80% capped $5 [1].
  - Square: Online cards 2.9% + $0.30; Payment Links 3.3% + $0.30; ACH 1% ($1 min, $5 max); Afterpay 6% + $0.30 [19][26][20][21].
  - Wallet charges (examples): Square Apple Pay/Google Pay at standard online card rate; Cash App Pay same as card (US‑only). PayPal lists Apple Pay/Google Pay as APMs; fees reflect the applicable product rate [19][37][38][39][12][13].

- International/cross‑border/currency conversion:
  - PayPal: +1.50% for international commercial transactions (most types); currency conversion spreads 4.00% or 3.00% depending on scenario [1].
  - Square: No separate published cross‑border surcharge; presentment in USD only; buyer’s bank handles FX [23][24].

- Other fees/policies:
  - PayPal: chargeback $20 (card chargebacks not processed through buyer’s PayPal account); dispute fees $15/$30; refunds don’t return original fees [1].
  - Square: no dispute management fee (Square covers dispute fee when contesting together); refunds don’t return processing fees [29][28].
  - Payouts: PayPal instant withdrawal 1.50% of amount (mins apply) [1]; Square instant/same‑day 1.75% [34].

3) International and multi‑currency
- PayPal: broad currency codes; FXaaS for multi‑currency balances and conversion; APMs matrix by buyer market [29][30][31][17].
- Square: accepts most international cards; presentment in USD only for US accounts; no DCC; limited APMs (Apple Pay/Google Pay, Afterpay in specified regions, Cash App Pay US) [23][24][3][69].

4) Setup and PCI scope
- PayPal: Card Fields is a PCI DSS service provider; reduces PCI scope; specific SAQ mapping not published in cited docs [4][34].
- Square: PCI DSS Level 1; Web Payments SDK provides PCI‑compliant inputs and tokenization; secure contexts/CSP required [31][3].
- Underwriting timing: PayPal unknown; Square typically 1–2 business days if additional review is needed [47].

5) Supported online payment methods
- PayPal: PayPal wallet, Venmo (US), Pay Later (US), cards (embedded or guest), Apple Pay/Google Pay (as APMs), ACH services (if enabled), plus local APMs by market [3][4][1][13][12][17].
- Square: Cards; ACH (US); Apple Pay/Google Pay; Cash App Pay (US); Afterpay (US/CA/AU/UK with limits); Square gift cards [3][20][37][38][39][21][40].
- Others: Unknown.

6) Fraud protection and security
- PayPal: PCI compliant; 3DS2 guidance; Fraud Protection/Fraud Protection Advanced; Chargeback Protection add‑on for Advanced Cards; Seller Protection program for wallet transactions [34][7][38][39][40][41].
- Square: PCI compliant; encryption/tokenization; Risk Manager with AVS/CVV rules and 3DS; coverage notes for disputes where 3DS should have applied but didn’t due to external factors [31][41][42][44][45].

7) SMB support
- PayPal: phone and message center hours published for US; merchant technical support portal; developer support entry point [45][46][47][48].
- Square: phone/chat/email with published US hours; help center and community; developer resources [46][47][48][49].
- SLAs/dedicated AMs for SMBs: Unknown for both in public docs.

8) Scalability
- PayPal: not publishing fixed rate‑limit policy; public status; robust reporting (online, API, SFTP); subscriptions/recurring; multiparty docs for marketplaces [10][50][51][52][54][58][2].
- Square: REST rate limits not publicly numeric; GraphQL limits published; public status; CSV exports; subscriptions API; platform fee (application fee) support; single‑country processing per account [51][50][53][54][6][55][23][79].

## Decision guide (based on verified 2025 sources)
- Prioritize the lowest blended online card rate with ACH option and free dispute handling: Square offers 2.9% + $0.30 for online API/SDK cards, ACH at 1% ($1 min/$5 max), no dispute management fee, instant/same‑day payouts at 1.75% [19][20][29][34]. Caveat: no multi‑currency presentment; international buyers pay in USD and their bank performs conversion [23][24].
- Prioritize PayPal wallet/Venmo reach with embedded cards and optional interchange++: PayPal Checkout (wallet, Venmo US) at 3.49% + $0.49; Pay Later 4.99% + fixed; standalone cards 2.99% + fixed; Advanced Cards 2.89% + fixed or interchange++ (+0.49% + fixed). ACH Services 0.80% capped at $5. Card chargeback fee $20 and dispute fees apply; refunds don’t return fees [1].
- Need Apple Pay/Google Pay with minimal extra setup: Both PayPal and Square support Apple Pay/Google Pay via their frameworks; pricing follows the product’s applicable online rate, not a special wallet fee in Square’s case [12][13][19][37][38].
- Require multi‑currency presentment with settlement control and broad APMs: PayPal documents broad currency codes, FXaaS, and a wide APM catalog by market [29][30][31][17]. Square does not support multi‑currency presentment from a US account [23][24].

## Gaps and unknowns to resolve before final selection
- Stripe, Braintree, Adyen, Shopify Payments, 2Checkout (Verifone), Amazon Pay, and standalone Apple Pay: 2025 US SMB online documentation (pricing, fees, methods, PCI scope, support, and international capabilities) must be verified directly from each provider’s official pricing, developer docs, legal, and help center pages. Items remain Unknown in this report.

### Sources
[1] PayPal Merchant Fees (U.S.): https://www.paypal.com/us/webapps/mpp/merchant-fees (accessed 2025-09-16)  
[2] PayPal Developer Home: https://developer.paypal.com/home/ (accessed 2025-09-16)  
[3] PayPal Checkout Standard (Hosted): https://developer.paypal.com/studio/checkout/standard (accessed 2025-09-16)  
[4] PayPal Checkout Advanced (Expanded): https://developer.paypal.com/studio/checkout/advanced (accessed 2025-09-16)  
[5] PayPal Advanced Checkout Docs Overview: https://developer.paypal.com/docs/checkout/advanced/ (accessed 2025-09-16)  
[6] PayPal Card Fields Events (Advanced): https://developer.paypal.com/docs/checkout/advanced/customize/card-fields-events/ (accessed 2025-09-16)  
[7] PayPal 3D Secure with Card Fields (SDK): https://developer.paypal.com/docs/checkout/advanced/customize/3d-secure/sdk/ (accessed 2025-09-16)  
[8] PayPal 3D Secure v1 (SDK): https://developer.paypal.com/docs/checkout/advanced/customize/3d-secure/v1/sdk/ (accessed 2025-09-16)  
[9] PayPal Webhooks API: https://developer.paypal.com/api/rest/webhooks/ (accessed 2025-09-16)  
[10] PayPal Rate Limiting Guidelines: https://developer.paypal.com/reference/guidelines/rate-limiting/ (accessed 2025-09-16)  
[11] PayPal Sandbox and Testing Tools: https://developer.paypal.com/tools/sandbox/ (accessed 2025-09-16)  
[12] PayPal Google Pay (APM): https://developer.paypal.com/docs/checkout/apm/google-pay/ (accessed 2025-09-16)  
[13] PayPal Apple Pay (APM): https://developer.paypal.com/docs/checkout/apm/apple-pay/ (accessed 2025-09-16)  
[14] PayPal Advanced Checkout Eligibility: https://developer.paypal.com/docs/checkout/advanced/eligibility/ (accessed 2025-09-16)  
[15] PayPal Currency Availability for Advanced Cards: https://developer.paypal.com/docs/checkout/advanced/currency-availability-advanced-cards/ (accessed 2025-09-16)  
[16] PayPal Pay with Venmo (Checkout Integration): https://developer.paypal.com/docs/checkout/pay-with-venmo/integrate/ (accessed 2025-09-16)  
[17] PayPal Supported APMs Matrix: https://developer.paypal.com/docs/multiparty/checkout/apm/supported-apms/ (accessed 2025-09-16)  
[18] Square WooCommerce and Square (Help): https://squareup.com/help/us/en/article/5922-woocommerce-and-square (accessed 2025-09-16)  
[19] Square Payments Pricing (Developers): https://developer.squareup.com/docs/payments-pricing (accessed 2025-09-16)  
[20] Square Add ACH (Web Payments SDK): https://developer.squareup.com/docs/web-payments/add-ach (accessed 2025-09-16)  
[21] Square Accept Afterpay/Clearpay on Square Online: https://squareup.com/help/us/en/article/7814-accept-payments-with-afterpay-clearpay-on-square-online (accessed 2025-09-16)  
[22] Square Checkout API Guidelines and Limitations: https://developer.squareup.com/docs/checkout-api/guidelines-and-limitations (accessed 2025-09-16)  
[23] Square Accepted Cards (international and currency notes): https://squareup.com/help/us/en/article/5085-accepted-cards (accessed 2025-09-16)  
[24] Square International Development – Payments: https://developer.squareup.com/docs/international-development/payments (accessed 2025-09-16)  
[25] Square Online Store Plans and Rates: https://squareup.com/us/en/online-store/plans (accessed 2025-09-16)  
[26] Square What are Square’s fees? (Help): https://squareup.com/help/us/en/article/5068-what-are-square-s-fees (accessed 2025-09-16)  
[27] Square Pricing Update (Bottom Line): https://squareup.com/us/en/the-bottom-line/managing-your-finances/square-pricing-update (accessed 2025-09-16)  
[28] Square Process Refunds (fees not returned): https://squareup.com/help/us/en/article/6116-process-refunds (accessed 2025-09-16)  
[29] Square Payment Disputes Walkthrough: https://my.squareup.com/help/us/en/article/3882-payment-disputes-walkthrough (accessed 2025-09-16)  
[30] Square Privacy and Security (Help): https://squareup.com/help/us/en/article/3796-privacy-and-security (accessed 2025-09-16)  
[31] Square PCI Compliance (Bottom Line): https://www.squareup.com/us/en/the-bottom-line/operating-your-business/pci-compliance (accessed 2025-09-16)  
[32] Square Next-business-day Deposit Schedule: https://squareup.com/help/us/en/article/5438-next-business-day-deposit-schedule (accessed 2025-09-16)  
[33] Square Set up and manage Instant Deposits: https://my.squareup.com/help/us/en/article/5405-set-up-and-manage-instant-deposits (accessed 2025-09-16)  
[34] Square Instant Transfers: https://squareup.com/us/en/payments/instant-transfers (accessed 2025-09-16)  
[35] Square Pricing Overview: https://squareup.com/pricing (accessed 2025-09-16)  
[36] Square Payments API CompletePayment: https://developer.squareup.com/reference/square/payments-api/CompletePayment (accessed 2025-09-16)  
[37] Square Apple Pay (Web Payments SDK): https://developer.squareup.com/docs/web-payments/apple-pay (accessed 2025-09-16)  
[38] Square Google Pay (Web Payments SDK): https://developer.squareup.com/docs/web-payments/google-pay (accessed 2025-09-16)  
[39] Square Add Cash App Pay (Web Payments SDK): https://developer.squareup.com/docs/web-payments/add-cash-app-pay (accessed 2025-09-16)  
[40] Square Gift Card (Web Payments SDK): https://developer.squareup.com/docs/web-payments/add-gift-card (accessed 2025-09-16)  
[41] Square Payments Security Overview: https://squareup.com/us/en/payments/secure (accessed 2025-09-16)  
[42] Square Risk Manager (Site Help): https://square.site/help/us/en/article/6816-navigating-square-risk-manager (accessed 2025-09-16)  
[43] Square Risk Manager FAQ: https://my.squareup.com/help/us/en/article/6815-risk-manager-faq (accessed 2025-09-16)  
[44] Square Risk Manager: 3D Secure (3DS) Help: https://squareup.com/help/us/en/article/7623-risk-manager-3d-secure-3ds (accessed 2025-09-16)  
[45] Square SCA Overview (Developers): https://developer.squareup.com/docs/sca-overview (accessed 2025-09-16)  
[46] Square Contact Support (API subdomain, US): https://api.squareup.com/help/us/en/article/4993-contact-square-support (accessed 2025-09-16)  
[47] Square Contact Support (US): https://my.squareup.com/help/us/en/article/4993-contact-square-support (accessed 2025-09-16)  
[48] Square Help Center: https://squareup.com/help (accessed 2025-09-16)  
[49] Square Developer Docs Home: https://developer.squareup.com/docs (accessed 2025-09-16)  
[50] Square GraphQL Limits: https://developer.squareup.com/docs/devtools/graphql (accessed 2025-09-16)  
[51] Square Handling Errors (rate limiting): https://developer.squareup.com/docs/build-basics/general-considerations/handling-errors (accessed 2025-09-16)  
[52] Square Developer Forums – Rate limit note: https://developer.squareup.com/forums/t/rate-limit-error-on-upsertcatalogobject-calls/6570/2 (accessed 2025-09-16)  
[53] Square Status – U.S.: https://www.issquareup.com/united-states (accessed 2025-09-16)  
[54] Square Print/Export/Email Reports: https://squareup.com/help/us/en/article/8362-print-export-or-email-your-reports (accessed 2025-09-16)  
[55] Square Collect Fees (Application Fee): https://developer.squareup.com/docs/payments-api/collect-fees (accessed 2025-09-16)  
[56] Square Inventory API – How it works: https://developer.squareup.com/docs/inventory-api/how-it-works (accessed 2025-09-16)  
[57] Square Webhooks Overview (retry/IPs): https://developer.squareup.com/docs/webhooks/overview (accessed 2025-09-16)  
[58] Square Web Payments SDK Quickstart: https://developer.squareup.com/docs/web-payments/quickstart/add-sdk-to-web-client (accessed 2025-09-16)  
[59] PayPal Developer Support: https://developer.paypal.com/support/ (accessed 2025-09-16)  
[60] PayPal Customer Disputes API v1: https://developer.paypal.com/docs/api/customer-disputes/v1/ (accessed 2025-09-16)  
[61] PayPal Seller Protection (Legal): https://www.paypal.com/us/legalhub/seller-protection (accessed 2025-09-16)  
[62] PayPal Disputes Overview: https://developer.paypal.com/docs/disputes/ (accessed 2025-09-16)  
[63] Square Payments API Webhooks: https://developer.squareup.com/docs/payments-api/webhooks (accessed 2025-09-16)  
[64] Square Checkout API Webhooks: https://developer.squareup.com/reference/square/checkout-api/webhooks (accessed 2025-09-16)  
[65] Square Online Checkout (product page): https://squareup.com/us/en/online-checkout (accessed 2025-09-16)  
[66] Square Payment Links (product page): https://squareup.com/us/en/payment-links (accessed 2025-09-16)  
[67] Square Buy Button (Payment Links): https://squareup.com/us/en/payment-links/buy-button (accessed 2025-09-16)  
[68] Square Test Connectivity Report (status link reference): https://square.site/help/us/en/article/8288-test-connectivity-report (accessed 2025-09-16)  
[69] Square Payment Method Support by Country: https://developer.squareup.com/docs/payment-card-support-by-country (accessed 2025-09-16)  
[70] Square payment.updated webhook: https://developer.squareup.com/reference/square/payments-api/webhooks/payment.updated (accessed 2025-09-16)  
[71] Square payment.created webhook: https://developer.squareup.com/reference/square/webhooks/payment.created (accessed 2025-09-16)  
[72] Square Take a Card Payment (Web Payments SDK): https://developer.squareup.com/docs/web-payments/take-card-payment (accessed 2025-09-16)  
[73] Square Secure Data Encryption (Help): https://squareup.com/help/us/en/article/3797-secure-data-encryption (accessed 2025-09-16)  
[74] Square Tokenization explainer (Bottom Line): https://www.squareup.com/us/en/the-bottom-line/managing-your-finances/what-does-tokenization-actually-mean (accessed 2025-09-16)  
[75] Square Contact Support (IE example for 24/7 references): https://my.squareup.com/help/ie/en/article/4993-contact-square-support (accessed 2025-09-16)  
[76] Square Subscriptions API – Subscription billing: https://developer.squareup.com/docs/subscriptions-api/subscription-billing (accessed 2025-09-16)  
[77] Square Cash App Payments (Payments API): https://developer.squareup.com/docs/payments-api/take-payments/cash-app-payments (accessed 2025-09-16)  
[78] Square Setting up rules with Risk Manager: https://square.site/help/us/en/article/6817-setting-up-rules-with-risk-manager (accessed 2025-09-16)  
[79] Square International Development (overview): https://developer.squareup.com/docs/international-development (accessed 2025-09-16)  
[80] Square Inventory API Webhooks: https://developer.squareup.com/docs/inventory-api/webhooks (accessed 2025-09-16)  
[81] Square Payment Disputes Walkthrough (API subdomain): https://api.squareup.com/help/us/en/article/3882-payment-disputes-walkthrough (accessed 2025-09-16)  
[82] Square Process Refunds (API subdomain variant): https://api.squareup.com/help/us/en/article/6116-process-refunds (accessed 2025-09-16)  
[83] Square Payment Processing Pricing: https://squareup.com/us/en/payment-processing/pricing (accessed 2025-09-16)  
[84] Square Payments API CancelPayment: https://developer.squareup.com/reference/square/payments-api/CancelPayment (accessed 2025-09-16)  
[85] Square Take payments and collect fees (application fees): https://developer.squareup.com/docs/payments-api/take-payments-and-collect-fees (accessed 2025-09-16)  
[86] PayPal Supported Currency Codes: https://developer.paypal.com/api/rest/reference/currency-codes/ (accessed 2025-09-16)  
[87] PayPal FX as a Service (Overview): https://developer.paypal.com/docs/checkout/fx-as-a-service/ (accessed 2025-09-16)  
[88] PayPal FX as a Service (Reference): https://developer.paypal.com/docs/checkout/fx-as-a-service/reference/ (accessed 2025-09-16)  
[89] PayPal Security and Protection: https://www.paypal.com/us/digital-wallet/security-and-protection (accessed 2025-09-16)  
[90] PayPal Payment Tokens API (Cards, AVS/CVV): https://developer.paypal.com/docs/checkout/save-payment-methods/purchase-later/payment-tokens-api/cards/ (accessed 2025-09-16)  
[91] PayPal Transaction Search API: https://developer.paypal.com/docs/api/transaction-search/v1/ (accessed 2025-09-16)  
[92] PayPal Transaction Search Docs: https://developer.paypal.com/docs/transaction-search/ (accessed 2025-09-16)  
[93] PayPal Activity Download (Online Reports): https://developer.paypal.com/docs/reports/online-reports/activity-download/ (accessed 2025-09-16)  
[94] PayPal SFTP Reports (Multiparty): https://developer.paypal.com/docs/multiparty/reports/sftp/ (accessed 2025-09-16)  
[95] PayPal Transaction Detail SFTP Report: https://developer.paypal.com/docs/reports/sftp-reports/transaction-detail/ (accessed 2025-09-16)  
[96] PayPal Payouts Reconciliation Report: https://developer.paypal.com/docs/multiparty/reports/payouts-reconciliation/ (accessed 2025-09-16)  
[97] PayPal Attempts and Declines Online Report: https://developer.paypal.com/docs/reports/online-reports/attempts-declines/ (accessed 2025-09-16)  
[98] PayPal Contact Customer Service (US Help): https://www.paypal.com/us/cshelp/article/how-do-i-contact-paypal-customer-service--help378 (accessed 2025-09-16)  
[99] PayPal Contact Customer Service (alt path): https://www.paypal.com/us/cshelp/article/contact-paypal-customer-service--help378 (accessed 2025-09-16)  
[100] PayPal Merchant Technical Support (Help): https://www.paypal.com/us/cshelp/article/how-do-i-contact-paypal-merchant-technical-support--ts2338 (accessed 2025-09-16)  
[101] PayPal Status: https://www.paypal-status.com/ (accessed 2025-09-16)  
[102] PayPal Recurring Payments (Business Subscriptions): https://securepayments.paypal.com/us/business/accept-payments/checkout/recurring (accessed 2025-09-16)