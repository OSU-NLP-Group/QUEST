# gRPC API Layer Architecture Options for Low Latency and Backward Compatibility Across Heterogeneous Clients

## Assumptions and Goals
- Goals: low latency, backward-compatible API evolution, support for clients on iOS/Android, browsers, desktops, and constrained IoT/edge devices.
- Protobuf/gRPC contracts are mostly stable with incremental changes.
- Open parameters: target RPS, SLOs, runtime languages, deployment environment, budget.
- Transport facts:
  - gRPC requires HTTP/2 with TLS/ALPN negotiation (“h2”) in most internet-facing deployments [1][2][20].
  - Browsers can’t call native gRPC directly; use gRPC-Web or JSON/REST [8][9][10][11].
- Streaming patterns:
  - Native gRPC supports unary, server-streaming, client-streaming, and bidirectional streaming [2][7].
  - gRPC‑Web supports unary and server-streaming from browsers (no browser-native client-streaming/bidi) [8][9].
- IoT:
  - Constrained devices often have intermittent connectivity, limited bandwidth, and tight power/CPU budgets; MQTT features (QoS 0/1/2, retained, Will, sessions) directly address these constraints per the OASIS specs [41][42].

## Summary of Viable Architecture Options
- A. Pure gRPC everywhere, with edge translation for browsers via Envoy/NGINX gRPC‑Web.
- B. gRPC with REST/JSON transcoding at the edge (Envoy or Google Endpoints).
- C. Managed API gateway or service-mesh–terminated gRPC (ALB/Cloud LBs/Istio).
- D. Connect/Connect‑Web facade (single client story across gRPC, gRPC‑Web, and JSON).
- E. Hybrid: gRPC for rich clients + MQTT bridge for constrained IoT/edge.

Each option below includes: description/assumptions, exactly 3 pros and 3 cons, and a trade‑off analysis.

---

## Option A — Pure gRPC + Edge gRPC‑Web Translation (Envoy/NGINX)

Description and key assumptions
- All native/mobile/desktop clients use gRPC over HTTP/2 + TLS end-to-end [2][7]. Browsers use gRPC‑Web via an edge proxy (Envoy gRPC‑Web filter; NGINX proxies gRPC but does not translate gRPC‑Web) [9][17][8]. TLS terminates at edge; ALPN=h2 is required for gRPC [1][20].
- Edge handles JWT auth, CORS (for gRPC‑Web), retries, timeouts, and HTTP/2 tuning [12][13][14][15][50].

Pros (exactly 3)
- Lowest app‑layer latency for native clients (no JSON transcoding); full streaming (unary/server/client/bidi) for non-browser clients [2][7].
- Browser support via gRPC‑Web without changing backend proto/gRPC services [8][9].
- Rich L7 controls and observability at edge (timeouts, retries, JWT verification, per‑method metrics) [12][13][16][49].

Cons (exactly 3)
- Browser clients limited to unary and server‑streaming; no native client‑streaming/bidi in gRPC‑Web [8][9].
- Extra proxy hop adds small latency and needs careful timeout/stream‑idle configuration for long streams [12][9].
- Operational complexity for HTTP/2 flow-control, keepalives, and CORS at the edge [6][14][50].

Trade‑offs
- Scalability: Excellent. HTTP/2 multiplexing and long‑lived connections; tune flow control windows and concurrent streams in Envoy to match BDP [14][7].
- Developer productivity: High. Single proto contract; browser uses generated gRPC‑Web stubs [8].
- Infra cost: Moderate. One edge proxy tier (Envoy/NGINX) adds compute but can consolidate cross‑cutting concerns [12][16][17].
- Ease of client integration: High for native; Good for web via gRPC‑Web with CORS [8][50].  
- Performance/latency: One proxy hop; budget for edge filters and TLS. Mesh‑like overhead data point: Istio shows ~+0.18 ms p90 and ~+0.25 ms p99 in its test (HTTP/1.1 baseline) to set expectations for proxy overheads [25].  
- Streaming: Native clients—full; browsers—unary/server‑streaming via gRPC‑Web [8][2].  
- Payload/serialization: Protobuf end‑to‑end; compression optional (gzip, others) [35][36]. Defaults for max inbound message size ~4 MiB (adjust on server/client) [37][38][39][40].  
- Versioning/compatibility: Additive field changes, reserve removed field numbers/names, avoid moving fields into existing oneof; deprecate carefully [31][32][33][34].  
- Auth/security: TLS/mTLS; per‑RPC creds (JWT/OAuth2); JWT verified at edge [3][15].  
- Deployment: Edge Envoy/NGINX, optionally CDN edges (CloudFront/Cloudflare) terminate TLS and forward h2 [21][20].  
- Observability: OTel RPC/gRPC semconv; Envoy gRPC stats filter and access logs [28][29][49][16].

---

## Option B — gRPC with REST/JSON Transcoding at the Edge

Description and key assumptions
- Keep a gRPC backend and use an edge filter to map HTTP/JSON to gRPC methods (Envoy gRPC‑JSON transcoder or Google Cloud Endpoints/ESPv2) via google.api.http annotations [10][11].
- Browser and legacy clients use JSON/HTTP; modern clients can still use native gRPC.

Pros (exactly 3)
- Broadest client compatibility (any HTTP/JSON stack) without duplicating business logic [10][11].
- Can coexist with gRPC on the same service using annotations; single proto as source of truth [11].
- Easy to trial/phase‑in; JSON routes added incrementally via config [10][11].

Cons (exactly 3)
- Extra serialization overhead (JSON⇄Protobuf) increases CPU and can add latency versus pure gRPC [10][7].
- Streaming semantics are constrained (e.g., mapped as JSON arrays), not 1:1 with gRPC streaming [10][11].
- Must maintain annotations and keep transcoder config in sync; some advanced mappings have limitations [10].

Trade‑offs
- Scalability: Good, but CPU overhead for transcoding at higher RPS [10].  
- Developer productivity: High for heterogeneous clients; one schema annotated for both protocols [11].  
- Infra cost: Moderate; adds edge CPU/memory; may reduce need for custom web SDKs [10].  
- Ease of client integration: Excellent for browsers/legacy; native clients still use gRPC [11].  
- Performance/latency: Additional edge transform; tune Envoy timeouts and stream idle for streaming endpoints [12].  
- Streaming: Works for unary; server-streaming maps to arrays; client/bidi semantics are limited [10][11].  
- Payload/serialization: Protobuf on backend; JSON on the wire to clients; message size and compression tunables apply on gRPC side [35][37][38][39][40].  
- Versioning: Same Protobuf best practices; HTTP annotations must be maintained across versions [31][32][33][34].  
- Auth/security: JWT validation at edge; CORS for JSON endpoints [15][50].  
- Deployment: Edge proxies (Envoy, ESPv2/API Gateway) with global LBs if needed [10][11][23].  
- Observability: Standard OTel RPC on backend; HTTP metrics at edge; Envoy stats/logs [28][16].

---

## Option C — gRPC Terminated at API Gateway or Service Mesh

Description and key assumptions
- Use managed gateways/load balancers with gRPC support (AWS ALB, Google Cloud External HTTP(S) LB, Azure App Gateway for Containers) and/or a service mesh (Istio) for mTLS, retries, and observability [22][23][24][25][26].
- Termination can occur at edge or sidecar; mesh enforces policy and telemetry inside the cluster [26][25].

Pros (exactly 3)
- Managed, battle‑tested ingress with built‑in gRPC awareness, health checks, and routing [22][23][24].
- Mesh provides zero‑trust mTLS and uniform retries/timeouts/telemetry without app changes [26][25].
- Global edges (CloudFront/Cloudflare/Google LB) reduce last‑mile latency and support streaming gRPC [21][20][23].

Cons (exactly 3)
- Additional hops (edge + sidecar) introduce small latency; requires careful timeout alignment [25][12].
- Operational learning curve (mesh policy, gateway config, quotas) [26][25].
- Cost for managed gateways/mesh resources and potential over‑provisioning.

Trade‑offs
- Scalability: Excellent; LBs and meshes scale horizontally and offer advanced routing, connection pooling, HTTP/2 tuning [23][14].  
- Developer productivity: High—centralized policy, retries, and mTLS in infra; app code simpler [26].  
- Infra cost: Moderate to high—managed L7 and sidecars consume resources; often justified at scale.  
- Ease of client integration: High for native gRPC; browsers still need gRPC‑Web or JSON (pair with Option A or B) [8][11].  
- Performance/latency: Data point for mesh overhead ~+0.18 ms p90, ~+0.25 ms p99 (Istio tests) [25]. Keep-alives, retries, and deadlines must be coordinated to avoid retry storms and timeout amplification [12][4][5][6].  
- Streaming: Full gRPC streaming supported through these L7s; ensure stream idle timeouts are configured for long‑lived streams [23][12].  
- Payload/serialization: Protobuf end‑to‑end; compression optional; message size knobs per runtime [35][36][37][38][39][40].  
- Versioning: Standard Protobuf evolution practices [31][32][33][34].  
- Auth/security: mTLS in mesh (PeerAuthentication), JWT at edge/gateway [26][15].  
- Deployment: Managed LBs (ALB/Google/Azure), CDN edges (CloudFront/Cloudflare), sidecar or ambient meshes [22][23][24][21][20][26].  
- Observability: OTel RPC semantics; mesh/gateway metrics and traces; Envoy stats [28][29][16][25].

---

## Option D — Connect/Connect‑Web Facade (on top of the same gRPC services)

Description and key assumptions
- Use Connect libraries to expose the same Protobuf APIs over gRPC, gRPC‑Web, and the Connect protocol, giving uniform client ergonomics and content types (binary Protobuf or JSON) across environments. Connect‑Web works in browsers and interops with gRPC/gRPC‑Web servers [8] (use alongside Envoy’s gRPC‑Web where needed).  
- Note: Browsers still cannot speak native gRPC; Connect‑Web uses browser‑compatible transports similar to gRPC‑Web [8].

Pros (exactly 3)
- Unified client experience and consistent error model across native and web, while remaining compatible with gRPC/gRPC‑Web [8].
- Flexibility to serve JSON or Protobuf with the same schema; easier gradual web adoption without REST annotations.
- Reduces need to hand‑maintain separate web SDKs when paired with gRPC‑Web infra [8][9].

Cons (exactly 3)
- Still needs browser‑compatible transport (gRPC‑Web or Connect protocol) and CORS; does not remove the need for an edge proxy [8][9][50].
- Another framework/runtime to support and operate alongside gRPC.
- Limited benefit if you already standardized on gRPC‑Web or JSON transcoding with mature tooling.

Trade‑offs
- Scalability: Similar to Options A/B; performance depends on chosen transport (gRPC‑Web vs JSON) and proxying [8][9][10].  
- Developer productivity: High—single schema and cohesive client libraries across platforms.  
- Infra cost: Similar to A/B (still requires edge translation/proxy).  
- Ease of client integration: High—browser and native clients share consistent semantics; still subject to browser streaming limits [8].  
- Performance/latency: On par with gRPC‑Web or JSON depending on path; tune timeouts/keepalives at the edge [12][6].  
- Streaming: Browser support typically unary/server‑streaming; native clients get full streaming [8][2].  
- Payload/serialization: Protobuf and JSON choices; compression same as gRPC [35][36].  
- Versioning: Same Protobuf evolution rules [31][32][33][34].  
- Auth/security: Reuse gRPC creds/JWT via metadata; CORS for web [3][15][50].  
- Deployment: Pair with Envoy gRPC‑Web (or JSON transcoding) and L7s as in A/B/C [9][10][23].  
- Observability: OTel RPC; proxy stats; same as A/B/C [28][29][16].

---

## Option E — Hybrid gRPC + MQTT Bridge for Constrained IoT/Edge

Description and key assumptions
- Rich clients (mobile/desktop/browser) use gRPC or gRPC‑Web (Options A–D). Constrained devices use MQTT to an edge or cloud broker. A bridge translates MQTT topics/payloads to backend gRPC calls, or invokes HTTP/gRPC via broker integrations (e.g., EMQX gRPC integration; AWS IoT Core rules to HTTP) [43][46].  
- MQTT is optimized for intermittent connectivity, low bandwidth, and power efficiency; supports QoS 0/1/2, retained messages, Will messages, and persistent sessions per OASIS specs [41][42].

Pros (exactly 3)
- Tailored to IoT constraints (QoS, sessions, retained/Will) that gRPC/HTTP/2 doesn’t natively provide in browsers or microcontrollers [41][42].
- Decouples device connectivity from backend availability via broker buffering/routing patterns [41][42].
- Rich integration ecosystem (broker rules/bridges/extensions) to call gRPC/HTTP services or route to compute [43][46][45].

Cons (exactly 3)
- Added moving parts: broker cluster, bridge/integration code, topic schema governance.
- Two serialization formats in play (MQTT payload + Protobuf over gRPC) unless you standardize on Protobuf payloads in MQTT.
- End‑to‑end tracing is harder across MQTT and gRPC hops; requires explicit propagation.

Trade‑offs
- Scalability: High—brokers scale horizontally; MQTT minimizes wire overhead; multi‑region requires broker replication/bridges [45].  
- Developer productivity: Medium—separate device topic contracts and backend RPCs; however, mature broker SDKs help.  
- Infra cost: Moderate to high—operate broker(s) and a bridge/integration tier; potentially offset by device power/bandwidth savings.  
- Ease of client integration: Excellent for constrained devices; rich clients remain on gRPC/gRPC‑Web.  
- Performance/latency: MQTT is efficient on constrained links; bridge adds a hop. Use local edge brokers for ultra‑low latency, and rules/integrations to call gRPC backends [43][46].  
- Streaming: MQTT provides pub/sub; not gRPC streaming, but continuous telemetry is natural; backpressure via QoS and broker queueing [41][42].  
- Payload/serialization: MQTT payloads are opaque—choose Protobuf for payloads to align with backend; compress only if CPU permits.  
- Versioning: Govern topic schemas; when using Protobuf payloads, apply the same reserved‑field/additive evolution rules [31][32][33].  
- Auth/security: TLS/mTLS for MQTT; broker ACLs and JWT/SAS where supported [54][53].  
- Deployment: Device→Broker (edge or cloud), then rules/bridge→gRPC services; AWS IoT Core can invoke HTTPS via rules [46]; EMQX can call gRPC services directly [43].  
- Observability: Instrument bridge with OTel; correlate message IDs and RPC metadata end‑to‑end [28][29][30].

---

## Cross‑Cutting Technical Guidance

Performance and latency
- HTTP/2 and ALPN: Ensure TLS negotiates h2 at the edge for gRPC; browsers use gRPC‑Web over HTTP/1.1 or HTTP/2 via proxy [1][2][8][20].  
- Proxy overhead: Expect low‑millisecond latency per hop; Istio reports ~+0.18 ms p90 and ~+0.25 ms p99 data-plane latency under its test (1 kB @ 1k rps, 2 proxies) [25].  
- Keepalives/connection management: Use HTTP/2 PING conservatively; coordinate server policies to avoid connection churn [6].  
- Timeouts/deadlines/retries: Set per‑RPC deadlines, align proxy route timeouts and stream idle timeouts, and configure retry/hedging with backoff to avoid amplification [4][5][12][13].  
- Flow control: Tune initial connection/stream windows and max concurrent streams where large/long-lived streams require it [14][7].

Browser limitations and workarounds
- Browsers cannot speak native gRPC; use gRPC‑Web (unary + server‑streaming) and configure CORS [8][9][50].  
- If you need JSON clients, use REST/JSON transcoding (Envoy/ESPv2) with google.api.http annotations [10][11].

IoT constraints
- MQTT features address intermittent connectivity: QoS 0/1/2, retained messages, Will, persistent sessions (v3.1.1/v5.0) [41][42].  
- Broker integrations: EMQX can invoke gRPC services; AWS IoT Core rules can call HTTPS endpoints; Mosquitto bridges to other brokers [43][46][45].

Streaming patterns
- Native gRPC supports all four streaming modes [2].  
- gRPC‑Web supports unary and server‑streaming; long‑lived streams may need proxy idle timeouts lifted [8][12][19].  
- MQTT provides pub/sub streams, not gRPC streams; QoS/backoff provide flow control semantics [41][42].

Payloads, message limits, and compression
- Default inbound message limits are typically ~4 MiB; raise carefully on both client and server if needed [37][38][39][40].  
- Prefer chunking/streaming to very large unary messages; enable gzip or other codecs as needed [35][36][7].

Versioning and backward compatibility
- Add fields without reusing numbers; reserve removed field numbers/names; avoid moving existing fields into oneof; deprecate judiciously [31][32][33][34].  
- For transcoded JSON, keep annotations updated; for MQTT, standardize payload schema (e.g., Protobuf).

Auth and security
- Use TLS everywhere; consider mTLS for service‑to‑service (mesh) and device‑to‑broker (IoT) [3][26][54][53].  
- JWT/OAuth2 per‑RPC credentials; validate tokens at edge (Envoy JWT filter) [3][15].  
- Configure CORS for browser clients (gRPC‑Web/JSON) [50].

Deployment topologies
- Edge L7 with gRPC support: ALB, Google External HTTP(S) LB, Azure App Gateway; global edges with CloudFront/Cloudflare reduce last‑mile latency and support gRPC streaming [22][23][24][21][20].  
- HTTP/3: Consider cautiously—Envoy supports downstream HTTP/3; upstream support is evolving. Test and keep fallback to HTTP/2 [27].

Observability and operability
- Instrument clients and servers with OpenTelemetry RPC/gRPC semconv; export latency percentiles and status codes [28][29][30].  
- Use Envoy/Istio metrics, gRPC stats filter, and access logs; trace across edge/mesh and application to correlate retries/deadlines [16][49][25].

---

## Decision Guidance

- Prioritize Option A if:
  - Native clients dominate and you want lowest latency and full streaming; browsers can use gRPC‑Web with minimal changes [8][9][2].
- Prefer Option B if:
  - You must support broad browser/legacy clients with minimal client libraries; you can afford JSON transcoding overhead [10][11].
- Choose Option C if:
  - You want managed L7 and/or mesh policy, mTLS, and telemetry consistency at scale (accepting small latency overhead) [22][23][24][25][26].
- Consider Option D if:
  - You want a single coherent client experience across platforms while remaining compatible with gRPC/gRPC‑Web; pair with A or B [8][9].
- Add Option E in parallel if:
  - You have constrained IoT devices; use MQTT brokers and a bridge to your gRPC backend for reliability and power efficiency [41][42][43][46].

---

## Implementation Checklist (condensed)
- Protocols
  - Ensure ALPN=h2 for gRPC at the edge; keep connections warm; tune keepalives and stream idle timeouts [1][6][12].
- API stability
  - Apply Protobuf evolution rules: additive changes; reserve removed fields; avoid breaking oneof changes [31][32][33][34].
- Edge and mesh
  - Configure retries/backoff and global request timeouts coherently across clients/proxies; collect Envoy/Istio metrics [12][13][25][16].
- Web
  - Use gRPC‑Web or JSON transcoding; set CORS appropriately; document streaming limitations [8][10][50].
- IoT
  - Define MQTT topic schema and QoS; enable broker TLS/mTLS and bridge/integration to gRPC/HTTP [41][42][54][53][43][46].
- Observability
  - Adopt OTel RPC/gRPC conventions; enable Envoy gRPC stats and access logs; propagate correlation metadata across hops [28][29][49][16].

### Sources
[1] RFC 7301: Application-Layer Protocol Negotiation (ALPN): https://www.rfc-editor.org/rfc/rfc7301  
[2] RFC 7540: HTTP/2: https://httpwg.org/specs/rfc7540  
[3] gRPC Auth (TLS, OAuth2/JWT): https://grpc.io/docs/guides/auth/  
[4] gRPC Deadlines: https://grpc.io/docs/guides/deadlines/  
[5] gRPC Retry & Hedging: https://grpc.io/docs/guides/retry/  
[6] gRPC Keepalive: https://grpc.io/docs/guides/keepalive/  
[7] gRPC Performance Best Practices: https://grpc.io/docs/guides/performance/  
[8] gRPC‑Web Basics (browser support, streaming): https://grpc.io/docs/platforms/web/basics/  
[9] Envoy gRPC‑Web Filter: https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/grpc_web_filter.html  
[10] Envoy gRPC‑JSON Transcoder: https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/grpc_json_transcoder_filter  
[11] Google Cloud Endpoints gRPC Transcoding: https://cloud.google.com/endpoints/docs/grpc/transcoding  
[12] Envoy Timeouts FAQ: https://www.envoyproxy.io/docs/envoy/latest/faq/configuration/timeouts  
[13] Envoy Router (Retries/Backoff): https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/router_filter.html  
[14] Envoy HTTP/2 Protocol Options: https://www.envoyproxy.io/docs/envoy/latest/api-v3/config/core/v3/protocol.proto.html  
[15] Envoy JWT Authn Filter: https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/jwt_authn_filter.html  
[16] Envoy Observability — Statistics: https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/observability/statistics  
[17] NGINX gRPC Proxy Module: https://nginx.org/en/docs/http/ngx_http_grpc_module.html  
[18] NGINX grpc_socket_keepalive: https://nginx.org/r/grpc_socket_keepalive  
[19] ingress‑nginx gRPC example (streaming timeouts): https://github.com/kubernetes/ingress-nginx/blob/main/docs/examples/grpc/README.md  
[20] Cloudflare gRPC Support (TLS/ALPN h2): https://developers.cloudflare.com/network/grpc-connections/  
[21] AWS CloudFront gRPC: https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/distribution-using-grpc.html  
[22] AWS ALB gRPC Support: https://aws.amazon.com/about-aws/whats-new/2020/10/application-load-balancers-enable-grpc-workloads-end-to-end-http-2-support/  
[23] Google Cloud External HTTPS Load Balancing (gRPC/HTTP3): https://cloud.google.com/load-balancing/docs/https  
[24] Azure App Gateway for Containers — gRPC: https://learn.microsoft.com/en-us/azure/application-gateway/for-containers/grpc  
[25] Istio Performance & Scalability (latency percentiles): https://istio.io/v1.22/docs/ops/deployment/performance-and-scalability/  
[26] Istio PeerAuthentication (mTLS): https://istio.io/v1.24/docs/reference/config/security/peer_authentication/  
[27] Envoy HTTP/3 Overview: https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/http/http3.html  
[28] OpenTelemetry Semantic Conventions — RPC: https://opentelemetry.io/docs/specs/semconv/rpc/  
[29] OpenTelemetry Semantic Conventions — gRPC: https://opentelemetry.io/docs/specs/semconv/rpc/grpc/  
[30] gRPC OpenTelemetry Metrics: https://grpc.io/docs/guides/opentelemetry-metrics/  
[31] Protobuf API Best Practices: https://protobuf.dev/best-practices/api/  
[32] Protobuf Editions/Updating a Message: https://protobuf.dev/programming-guides/editions/  
[33] Protobuf Proto3 — Deleting/Reserved Fields: https://protobuf.dev/programming-guides/proto3/#deleting  
[34] Protobuf Proto3 — oneof: https://protobuf.dev/programming-guides/proto3/#oneof  
[35] gRPC Core Compression Overview: https://grpc.github.io/grpc/core/md_doc_compression.html  
[36] gRPC Compression Guide: https://grpc.io/docs/guides/compression/  
[37] grpc‑go MaxRecvMsgSize: https://pkg.go.dev/github.com/grpc/grpc-go#MaxRecvMsgSize  
[38] grpc‑java Server maxInboundMessageSize: https://grpc.github.io/grpc-java/javadoc/io/grpc/ForwardingServerBuilder.html#maxInboundMessageSize-int-  
[39] grpc‑java Channel maxInboundMessageSize: https://grpc.github.io/grpc-java/javadoc/io/grpc/ForwardingChannelBuilder.html#maxInboundMessageSize-int-  
[40] Python grpc.insecure_channel (channel args): https://grpc.github.io/grpc/python/_modules/grpc.html#insecure_channel  
[41] OASIS MQTT v3.1.1: https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.html  
[42] OASIS MQTT v5.0: https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html  
[43] EMQX Data Integration — gRPC: https://www.emqx.com/en/docs/emqx/latest/data-integration/grpc.html  
[44] HiveMQ Extension SDK Overview: https://www.hivemq.com/docs/extensions/latest/sdk/overview.html  
[45] Mosquitto Bridge: https://mosquitto.org/man/mosquitto-bridge-5.html  
[46] AWS IoT Core Rule Actions — HTTP: https://docs.aws.amazon.com/iot/latest/developerguide/iot-rule-actions.html#http-action  
[47] Envoy gRPC Stats Filter: https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/grpc_stats_filter  
[48] Envoy CORS Filter: https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/cors_filter  
[49] gRPC‑Web Basics (repeat for emphasis on browser limits): https://grpc.io/docs/platforms/web/basics/  
[50] Envoy CORS Filter (CORS for web): https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/cors_filter  
[51] Envoy HTTP/3 Overview (repeat for readiness): https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/http/http3.html  
[52] grpc‑java NettyChannelBuilder (HTTP/2 tuning): https://grpc.github.io/grpc-java/javadoc/io/grpc/netty/NettyChannelBuilder.html  
[53] EMQX TLS Listener Configuration: https://www.emqx.com/en/docs/emqx/latest/configuration/listeners/tls.html  
[54] HiveMQ Security (TLS/mTLS, Auth): https://www.hivemq.com/docs/hivemq/latest/user-guide/security.html