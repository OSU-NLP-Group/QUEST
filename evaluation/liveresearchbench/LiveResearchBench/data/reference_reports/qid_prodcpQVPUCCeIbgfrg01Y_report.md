# US Cloud Comparison (AWS vs Google Cloud vs Microsoft Azure vs Oracle Cloud) — Pricing, ML/AI, Support, Infrastructure, Security (as of Sep 5, 2025)

Note on scope and sources: This report summarizes US‑GA facts for Amazon Web Services (AWS) and Google Cloud Platform (GCP) with direct links to official pricing/docs and SLAs. Comparable, rigorously sourced detail for Microsoft Azure and Oracle Cloud Infrastructure (OCI) was not captured in the research set and is therefore omitted to avoid unsourced statements. Where pricing pages present dynamic region selectors or calculators, the official machine‑readable price lists or “all pricing” tables are referenced for authoritative numbers. Access date for all sources: 2025‑09‑05.

## Regions used for US price examples

- AWS: us‑east‑1 (N. Virginia) and us‑west‑2 (Oregon). These are long‑standing US commercial regions with broad service coverage; exact prices vary by Region and are published per Region via the AWS Price List (Bulk) API and pricing pages [1][2][3][4].
- Google Cloud: us‑east4 (Northern Virginia/Ashburn) and us‑central1 (Iowa). Google documents regions/zones and publishes per‑region price tables for core services [29][30][34].

## 1) US Pricing for Major Services (on‑demand/pay‑as‑you‑go), key cost drivers, and commitments

### 1.1 Compute (on‑demand, Spot/preemptible, commitments)

- AWS
  - On‑Demand model: EC2 is billed per second with a 60‑second minimum for Linux/Windows; prices exclude taxes and vary by Region [1].
  - Exact EC2 hourly rates by Region: AWS provides the authoritative per‑SKU prices via the Price List (Bulk) API (e.g., current us‑east‑1 and us‑west‑2 CSVs) for m7i/m7g, c7i/c7g, r7i/r7g, and GPU families; use these files for exact current USD/hour figures in each Region [2][3][4].
  - Spot Instances: “up to 90%” discount vs On‑Demand; spare capacity with 2‑minute interruption notification; prices vary based on long‑term supply and demand trends [5].
  - Savings Plans: Two types—Compute Savings Plans (most flexibility) and EC2 Instance Savings Plans (lowest prices); “save up to 72%” vs On‑Demand depending on term and coverage; commitments are in $/hour for 1‑ or 3‑year terms [6][7].
  - GPU price update: AWS announced up to 44% On‑Demand reductions for P5/P5en (NVIDIA H100/H200) and up to 33% for P4d/P4de (A100) effective June 2025; check the EC2 price list for your Region’s current hourly rates [8].
  - GPU instance families: P5 (H100/H200) and P4 (A100) with instance‑type details published on product pages; pricing is per Region via the EC2 price list [25][26].

- Google Cloud
  - Resource‑based VM pricing: total price = vCPU price × vCPU count + memory price × GiB (custom machine types add a 5% premium); machine‑family and per‑Region rates are published on the VM pricing page [30].
  - Exact per‑instance hourly examples: Google’s “All pricing” table shows concrete per‑instance USD/hour totals by Region (e.g., A3 UltraGPU 8×H100 in us‑central1 listed at $84.8069/h on the “All pricing” table at access time) [33].
  - Spot VMs: dynamic prices with large discounts; the VM pricing page notes Spot discounts commonly in the ~60–91% range versus on‑demand (discounts can be smaller for certain accelerators) [30].
  - Sustained Use Discounts (SUDs): up to 20% for N2/N2D/C2; up to 30% for N1 and memory‑optimized; GPUs eligible with exclusions (e.g., H100/A100/L4/B200/H200 not eligible); SUDs apply after usage thresholds each month [31].
  - Committed Use Discounts (CUDs): spend‑based flexible CUDs published by family with representative discount levels (e.g., 28% 1‑year, 46% 3‑year for several general/compute families; higher only on certain 3‑year memory‑optimized commitments); see the CUD overview for exact terms and family coverage [32].

### 1.2 Storage (object, block, file; snapshots/archives)

- AWS
  - Amazon S3: Storage classes (Standard, Standard‑IA, One Zone‑IA, Glacier Instant Retrieval, Glacier Flexible Retrieval, Glacier Deep Archive) are priced per GB‑month by Region with request/operation and retrieval charges; use the S3 pricing page with the Region selector for current us‑east‑1/us‑west‑2 USD rates per tier and request class [9].
  - Amazon EBS volumes:
    - gp3: $0.08/GB‑month; includes 3,000 IOPS and 125 MiB/s, with additional provisioned IOPS at $0.005/IOPS‑month and throughput at $0.04/MiB/s‑month [10].
    - io2: $0.125/GB‑month; IOPS billed in tiered $/IOPS‑month rates at high IOPS levels (see page for exact brackets) [10].
  - EBS snapshots:
    - Standard snapshot storage commonly listed at $0.05/GB‑month (per EBS docs and examples on snapshot pages) [11].
    - Snapshot Archive: $0.0125/GB‑month for archived snapshots; archive retrieval billed $0.03/GB; 90‑day minimum storage duration (see considerations) [11].
  - Amazon EFS: Per‑class storage pricing (Standard, Infrequent Access, Archive) varies by Region; EFS pricing page provides USD/GB‑month by Region (select us‑east‑1/us‑west‑2) and throughput modes [12].
  - Amazon FSx:
    - FSx for NetApp ONTAP: Effective $/GB‑month tables by Region for Single‑AZ and Multi‑AZ deployments, plus throughput capacity, IOPS, and backup charges [13].
    - FSx for Lustre: Per‑GB‑month storage rates (SSD and Intelligent‑Tiering classes), metadata IOPS charges, and backup rates published by Region [14].

- Google Cloud
  - Cloud Storage: Per‑Region $/GB‑month rates for Standard, Nearline, Coldline, Archive are published (e.g., us‑central1 and us‑east4); operation (Class A/B) and retrieval charges are listed. Note: at access time, two renderings of the pricing page showed slightly different per‑GB numbers for the same Regions; verify current SKUs in your billing catalog and calculator if you see discrepancies [34].
  - Persistent Disk and Hyperdisk: Per‑GiB‑hour rates for Standard, Balanced, SSD, and Regional PD (≈$‑per‑GB‑month shown by converting from GiB‑hour); snapshot standard and archive rates and retrieval pricing are documented [35].
  - Filestore (managed NFS): Pricing depends on tier (Zonal, Regional, Basic SSD/HDD, Enterprise legacy), allocated capacity, and Region; select us‑central1/us‑east4 in the pricing UI on the product page for current USD/GB‑month [36].
  - BigQuery storage: Logical and physical storage are priced per GiB‑hour, with long‑term discounts after 90 days of no changes; the pricing page shows per‑GiB‑hour rates and example monthly conversions [37].

### 1.3 Data transfer and networking

- AWS
  - Internet egress and data transfer: Data transfer charges vary by direction and destination; AWS publishes tiered internet egress rates and intra/Inter‑AZ and inter‑Region data transfer on the EC2/VPC pricing pages (use the current pages for the latest US tiers) [1][15].
  - NAT Gateway: Billed per hour and per GB processed (e.g., commonly $0.045/hour and $0.045/GB in a US Region like us‑east‑2; confirm Region‑specific rates on the VPC pricing and NAT docs) [15][16].
  - Elastic Load Balancing:
    - Application Load Balancer (ALB) includes a fixed hourly charge (e.g., $0.0225/hr in us‑east‑1 example) plus per LCU‑hour charges; Network Load Balancer (NLB) has per‑hour and NLCU components; see pricing page tables per Region [17].
  - Amazon CloudFront (US/Canada/Mexico): Tiered Data Transfer Out per GB (e.g., next 9 TB at $0.085/GB after the first free tier) and request pricing per 10,000 requests are published; see the US pricing tier table for full breakdowns [18].
  - AWS Direct Connect: Dedicated port‑hour rates (e.g., 1 Gbps $0.30/hr; 10 Gbps $2.25/hr; 100 Gbps $22.50/hr; 400 Gbps $85/hr) and per‑GB data transfer out via DX vary by source Region and DX location (US matrices listed on the pricing page) [19].

- Google Cloud
  - VPC network data transfer (Premium Network Tier): Inter‑zone (same Region) $0.01/GB; inter‑Region within North America $0.02/GB; internet egress tiers are published (pricing change announcement effective 2024‑02‑01 includes matrices and current per‑GB rates/tiering) [38][39].
  - Cloud NAT: Charges per VM/hour (per‑gateway scale band) plus $/GB processed, with static IP costs; see pricing page for exact per‑hour and per‑GB rates [40].
  - External Application Load Balancing: Per‑hour charge for forwarding rules plus per‑GB data processing both directions (e.g., US regions show $0.008/GB in each direction in the published table) [41].

### 1.4 Commitments, negotiated programs, and notes

- AWS
  - Savings Plans (above) provide published, programmatic discounts vs On‑Demand [6][7].
  - Private Offers/EDP context: While core AWS service discounts via Enterprise Discount Program (EDP) are not publicly documented in detail, AWS Marketplace supports negotiated Private Offers with custom pricing/terms for third‑party software and services, indicating available private/contractual pricing mechanisms via AWS sales/partners [28].
  - Discrepancy watch: GPU instance rates are often updated; the authoritative per‑Region hourly is the EC2 price list CSV/JSON and what the Pricing Calculator shows at estimate time [2][3][4][8].

- Google Cloud
  - SUDs and CUDs provide published discounts automatically (SUD) or with commitments (CUD) as documented above [31][32].
  - Discrepancy watch: The Cloud Storage pricing page exhibited inconsistent per‑Region $/GB values in two renderings at access time; verify in your SKU catalog and calculator [34].

## 2) ML/AI Services Capabilities and Pricing

- AWS
  - Foundation models via Amazon Bedrock: Model catalog includes providers such as Anthropic, Meta, Cohere, Amazon (e.g., Titan/Nova), Mistral AI, Stability AI, AI21 Labs, Writer, and others (availability varies by Region). Bedrock pricing publishes per‑model on‑demand usage rates (token/character based) and provisioned throughput pricing; US Regions include us‑east‑1/us‑west‑2 among others [27].
  - Costing guidance for generative AI: AWS blogs/solutions documents show how Bedrock model invocation, guardrails, embeddings, and data stores (e.g., OpenSearch Serverless) combine into end‑to‑end cost, with worked examples to estimate monthly spend [27].
  - Accelerators for training/inference: EC2 GPU instance families P5 (H100/H200) and P4 (A100) are documented; June 2025 price reductions were announced for these families; on‑demand hourly rates are Region‑specific in the EC2 price list [25][26][8].
  - Enterprise ML platform: (Note) SageMaker features/pricing were not captured in this research set; incorporate the official SageMaker pricing and docs for training, inference (real‑time/serverless), notebooks/Studio, Pipelines, Feature Store, and Model Registry in a subsequent update (to avoid unsourced claims).

- Google Cloud
  - Vertex AI platform: Unified managed ML platform with workbench/notebooks, training, endpoints (serverless and dedicated), pipelines, feature store, model registry; pricing is componentized and published on the Vertex AI pricing page [42].
  - Generative AI on Vertex AI (Gemini) pricing: Detailed per‑unit charges by model and mode (online vs batch) with context‑length tiers; examples at access time:
    - Gemini 1.5 Pro: ≤128k context—input $0.00125 per 1k characters; output $0.00375 per 1k characters (longer contexts priced higher). Batch mode discount and context caching pricing are published [43].
    - Gemini 1.5 Flash: ≤128k context—input $0.00001875 per 1k characters; output $0.000075 per 1k characters; long‑context pricing and batch discounts are documented [43].
  - Model availability and limits: Gemini 1.5 Pro documentation lists supported Regions (including US regions), modalities, and limits [44].
  - Accelerators: GPU pricing and policies (A100/H100/H200/B200, L4, etc.) are published, with notes on which families include accelerator cost in the machine price (e.g., A2/A3) vs discrete GPU adders; MIG partitioning and availability are documented in product docs (training via Vertex AI or Compute Engine) [30][33].

## 3) Enterprise Support (plan tiers, response times, TAM, SLAs)

- AWS
  - Support plan tiers: Developer, Business, Enterprise On‑Ramp, Enterprise. Target initial responses include: Business—<1h for “production system down”; Enterprise On‑Ramp—<30 minutes for “business‑critical system down”; Enterprise—<15 minutes for “business‑critical system down” (with broader 24×7 coverage and proactive features by tier) [20][22].
  - Pricing: Minimum monthly charges and percentage‑of‑usage ladders are published (e.g., Business minimum $100; Enterprise On‑Ramp minimum $5,500; Enterprise minimum $15,000, each with tiered percentage rates above minimums) [21].
  - TAM/Concierge: Enterprise includes a designated TAM; Enterprise On‑Ramp provides a pooled TAM model; Business includes access to features like the AWS Support App in Slack; Trusted Advisor coverage expands with tier [20][22].

- Google Cloud
  - Customer Care portfolio: Standard, Enhanced, and Premium plans with documented initial response SLOs and 24×7 coverage levels. Examples: Enhanced—P1 in 1 hour; Premium—P1 in 15 minutes; scope, language coverage, and escalation features documented per plan [45][46][47][48][49].
  - Pricing: Minimum monthly charges and percentage‑of‑usage tiers published per plan (e.g., Premium minimum $15,000/month) [45][48].
  - Compute SLA: GCE multi‑zone instances in Premium Network Tier target ≥99.99%; single instance families ≥99.9% or higher depending on family; definitions and credit terms published (Dec 2023 revision) [50].

- Third‑party perspective: No analyst or user‑review sources (e.g., Gartner Peer Insights, Forrester, IDC, G2) were captured in the research set; therefore, no overall support‑experience summary is included here to avoid unsourced claims.

## 4) Infrastructure and Availability (regions, zones/AZs, edge, sovereign/government)

- AWS
  - Global counts (discrepancy at access time): The AWS Global Infrastructure overview page lists “120 Availability Zones within 38 Geographic Regions,” along with counts for edge and specialized zones; other AWS pages list “117 AZs within 37 Regions.” Treat the overview page as the latest top‑line and confirm recent launches via What’s New and region pages [23][24].
  - Edge: The overview page also lists CloudFront/edge counts (e.g., “700+ CloudFront POPs and 13 Regional Edge Caches” at access time) and Local/Wavelength Zones counts; see the page for the current tallies [23].

- Google Cloud
  - Regions/zones: Product documentation explains regions/zones and best practices for distributing workloads; US regions include us‑central1 and us‑east4 among others [29].
  - Edge footprint: Cloud CDN publishes “more than 100 cache locations” and lists US/North America metros; Google’s network edge locations page lists “over 200” global locations for connecting into Google Cloud services [51][52].

- Sovereign/government
  - AWS: GovCloud and other sovereign offerings are enumerated across AWS infrastructure pages; confirm program details, eligibility, and service availability per Region on the AWS infra site (use the overview/regions pages for the latest region list) [23][24].
  - Google Cloud: FedRAMP guide describes Google’s approach to FedRAMP authorization, shared responsibilities, and control inheritance for US public sector workloads [53].

## 5) Security and Compliance (IAM, encryption/KMS/HSM, network security, shared responsibility, certifications)

- AWS
  - (Note) Detailed AWS IAM, KMS (BYOK, XKS), CloudHSM, default encryption, network security baseline, and certification/program listings (ISO, SOC, PCI, HIPAA, FedRAMP) were not captured in the research set; consult the AWS Security, KMS, Artifact/Compliance Center, and Shared Responsibility Model pages for authoritative details to maintain parity with the GCP coverage below.

- Google Cloud
  - IAM: Role‑based access control applied across the resource hierarchy with principals/roles/policies; least‑privilege practices documented [54].
  - Encryption and keys:
    - Cloud Storage: “always encrypts your data on the server side” before it is written to disk; supports CMEK (Cloud KMS/HSM/external) and customer‑supplied keys [55].
    - Cloud KMS External Key Manager (EKM): enables use of externally hosted keys (hold‑your‑own‑key) without Google storing key material; setup and connection requirements documented [56][57].
    - Firestore: automatically encrypts all data at rest, with CMEK option [58].
  - Compliance programs (selected):
    - FedRAMP: Guide describes Google’s FedRAMP posture and shared responsibilities/control inheritance for US public sector [53].
    - HIPAA: Overview and detailed pages describe the HIPAA program, Business Associate Addendum (BAA), shared responsibility, and covered services [59][60].
    - PCI DSS: Annual third‑party assessment and list of compliant services published; responsibilities outlined [61].
    - ISO/IEC 27001: Certification of Google Cloud Platform and common infrastructure is documented; compliance reports available [62].

## 6) Key discrepancies and validation notes

- AWS global counts: Conflicting figures (38 Regions/120 AZs vs 37/117) appear across AWS pages at access time; use the Global Infrastructure overview for the latest summary and confirm newest Region launches via What’s New and regional pages [23][24].
- AWS GPU pricing: June 2025 reductions are published as percentages; per‑Region hourly list rates are in the EC2 price list CSV/JSON and may differ from older blog examples or third‑party trackers; use the price list and Pricing Calculator when quoting numbers [2][3][4][8].
- Google Cloud Storage pricing: Two renderings of the pricing page displayed different per‑GB numbers for the same Regions at access time; validate against your billing SKUs and the “All pricing”/calculator before committing figures in contracts or cost models [34].

## 7) Coverage gaps and next steps

- Azure and OCI: This research set did not capture the official Azure or OCI pricing/docs/support/security pages required to present rigorous, side‑by‑side, per‑statement‑sourced facts. To complete the brief, gather:
  - Azure: VM pricing by Region (East US/West US 2), Managed Disks/Snapshots, Blob Storage tiers and operations, bandwidth/egress, Load Balancer/NAT Gateway, ExpressRoute; Azure ML/AI (Azure ML, model catalog, Phi‑3/OpenAI Azure OpenAI Service), GPU SKUs and pricing; support tiers (Developer/Standard/Professional Direct), SLAs; global infrastructure counts; Azure AD/Microsoft Entra ID, Key Vault/HSM (BYOK/HYOK), compliance and sovereign cloud (Azure Government).
  - OCI: Compute shapes/pricing (VM.Standard/E4/Flex; GPU shapes like BM.GPU.NV/MI), Block Volume/Object Storage tiers and requests, snapshots/archive; networking (egress, FastConnect); OCI Data Science/Generative AI service; support tiers and SLAs; global regions/A‑D (ADs), government/sovereign regions; IAM, Vault, KMS/HSM, compliance.
- Once collected, align region choices (e.g., Azure East US/West US 2; OCI us‑ashburn‑1/us‑phx‑1), and replicate the format used here for AWS and GCP to complete the four‑way comparison.

### Sources

[1] Amazon EC2 On‑Demand Pricing: https://aws.amazon.com/ec2/pricing/on-demand/  
[2] Using the AWS Price List Bulk API (docs): https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/using-the-aws-price-list-bulk-api-fetching-price-list-files-manually.html  
[3] AmazonEC2 current price list (us‑east‑1): https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.csv  
[4] AmazonEC2 current price list (us‑west‑2): https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-west-2/index.csv  
[5] Amazon EC2 Spot pricing overview: https://aws.amazon.com/ec2/spot/pricing/  
[6] AWS Savings Plans overview: https://aws.amazon.com/savingsplans/  
[7] Compute Savings Plans pricing: https://aws.amazon.com/savingsplans/compute-pricing/  
[8] AWS What’s New — EC2 NVIDIA GPU pricing reductions (June 2025): https://aws.amazon.com/about-aws/whats-new/2025/06/pricing-usage-model-ec2-instances-nvidia-gpus/  
[9] Amazon S3 Pricing: https://aws.amazon.com/s3/pricing/  
[10] Amazon EBS Volume Types (pricing details): https://aws.amazon.com/ebs/volume-types/  
[11] Amazon EBS Snapshot Archive Pricing (docs): https://docs.aws.amazon.com/ebs/latest/userguide/snapshot-archive-pricing.html  
[12] Amazon EFS Pricing: https://aws.amazon.com/efs/pricing/  
[13] Amazon FSx for NetApp ONTAP Pricing: https://aws.amazon.com/fsx/netapp-ontap/pricing/  
[14] Amazon FSx for Lustre Pricing: https://aws.amazon.com/fsx/lustre/pricing/  
[15] Amazon VPC Pricing: https://aws.amazon.com/vpc/pricing/  
[16] NAT Gateway pricing (docs): https://docs.aws.amazon.com/vpc/latest/userguide/nat-gateway-pricing.html  
[17] Elastic Load Balancing Pricing: https://aws.amazon.com/elasticloadbalancing/pricing/  
[18] Amazon CloudFront Pricing: https://aws.amazon.com/cloudfront/pricing/  
[19] AWS Direct Connect Pricing: https://aws.amazon.com/directconnect/pricing/  
[20] AWS Support Plans: https://aws.amazon.com/premiumsupport/plans/  
[21] AWS Support Pricing: https://aws.amazon.com/premiumsupport/pricing/  
[22] AWS Support — Enterprise On‑Ramp: https://aws.amazon.com/premiumsupport/plans/enterprise-onramp/  
[23] AWS Global Infrastructure (Overview): https://aws.amazon.com/about-aws/global-infrastructure/  
[24] AWS Global Infrastructure — Regions & AZs: https://aws.amazon.com/about-aws/global-infrastructure/regions_az/index.html  
[25] Amazon EC2 P5 instances (H100/H200): https://aws.amazon.com/ec2/instance-types/p5/  
[26] Amazon EC2 P4 instances (A100): https://aws.amazon.com/ec2/instance-types/p4/  
[27] Amazon Bedrock Pricing: https://aws.amazon.com/bedrock/pricing/  
[28] AWS Marketplace — Private Offers for partners: https://aws.amazon.com/marketplace/partners/private-offers  
[29] Google Cloud — Compute Engine regions and zones: https://cloud.google.com/compute/docs/regions-zones/  
[30] Google Cloud — Compute Engine VM instance pricing: https://cloud.google.com/compute/vm-instance-pricing/  
[31] Google Cloud — Sustained Use Discounts: https://cloud.google.com/compute/docs/sustained-use-discounts  
[32] Google Cloud — Committed Use Discounts overview: https://cloud.google.com/compute/docs/instances/committed-use-discounts-overview  
[33] Google Cloud — Compute Engine “All pricing” table: https://cloud.google.com/compute/all-pricing?hl=en_US  
[34] Google Cloud Storage pricing: https://cloud.google.com/storage/pricing/  
[35] Google Cloud — Disks and Images pricing (PD/Hyperdisk): https://cloud.google.com/compute/disks-image-pricing?hl=en  
[36] Google Cloud — Filestore (pricing via product page): https://cloud.google.com/filestore  
[37] Google Cloud — BigQuery pricing: https://cloud.google.com/bigquery/pricing/  
[38] Google Cloud — VPC network pricing: https://cloud.google.com/vpc/network-pricing  
[39] Google Cloud — VPC pricing change announcement: https://cloud.google.com/vpc/pricing-announce  
[40] Google Cloud — Cloud NAT pricing: https://cloud.google.com/nat/pricing  
[41] Google Cloud — Cloud Load Balancing pricing: https://cloud.google.com/load-balancing/pricing  
[42] Google Cloud — Vertex AI pricing: https://cloud.google.com/vertex-ai/pricing  
[43] Google Cloud — Generative AI on Vertex AI (Gemini) pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing/  
[44] Google Cloud — Gemini 1.5 Pro model page: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/1-5-pro  
[45] Google Cloud Support (portfolio landing): https://cloud.google.com/support  
[46] Google Cloud — Enhanced Support: https://cloud.google.com/support/enhanced  
[47] Google Cloud — Standard Support: https://cloud.google.com/support/standard  
[48] Google Cloud — Premium Support: https://cloud.google.com/support/premium  
[49] Google Cloud — Enhanced Support documentation: https://cloud.google.com/support/docs/enhanced  
[50] Google Cloud — Compute Engine SLA: https://cloud.google.com/compute/sla/  
[51] Google Cloud CDN — Cache locations: https://cloud.google.com/cdn/docs/locations  
[52] Google Cloud — Network edge locations (PoPs): https://cloud.google.com/vpc/docs/edge-locations  
[53] Google Cloud — FedRAMP guide: https://cloud.google.com/security/compliance/fedramp-guide  
[54] Google Cloud IAM overview: https://cloud.google.com/iam/docs/overview/  
[55] Cloud Storage — Encryption: https://cloud.google.com/storage/docs/encryption  
[56] Cloud KMS — External Key Manager (EKM): https://cloud.google.com/kms/docs/ekm.html  
[57] Cloud KMS — Create an EKM connection: https://cloud.google.com/kms/docs/create-ekm-connection  
[58] Firestore — Server‑side encryption: https://cloud.google.com/firestore/docs/server-side-encryption  
[59] Google Cloud — HIPAA compliance overview: https://cloud.google.com/security/compliance/hipaa-compliance  
[60] Google Cloud — HIPAA on Google Cloud (detailed): https://cloud.google.com/security/compliance/hipaa  
[61] Google Cloud — PCI DSS compliance: https://cloud.google.com/security/compliance/pci-dss  
[62] Google Cloud — ISO/IEC 27001 certification: https://cloud.google.com/security/compliance/iso-27001