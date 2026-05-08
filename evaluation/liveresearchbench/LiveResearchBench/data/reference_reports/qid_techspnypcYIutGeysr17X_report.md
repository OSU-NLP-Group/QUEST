# Reference Architecture: Highly Available, Horizontally Scalable Real‑Time Chat for Millions of Concurrent Users

## Executive Summary
Design a real-time chat system around a durable, log-based backbone with strict per-conversation ordering, application-level idempotency, and transactional “outbox→bus→inbox” paths. Achieve practical end-to-end exactly-once semantics by combining:
- Broker features (Kafka transactions/idempotent producers; Pulsar transactions + broker dedup; NATS JetStream duplicate windows; Redis Streams explicit IDs) [1][2][4][5][7][8][9][10]
- Per-conversation monotonic sequence numbers and idempotent storage (UPSERT/conditional writes/LWT) [34][35][36][37]
- Write-time fan-out for durable per-user inboxes and read-time fan-out at the edge for ephemeral signals (typing/presence) [1][4][7][16]

Transport: WebSockets for client real-time paths, gRPC streams for service-to-service, optional WebRTC DataChannel for low-latency rooms; all over TLS 1.3 and zero-trust mTLS inside the mesh [12][15][14][21][52]. For large groups and strong post-compromise security, adopt Messaging Layer Security (MLS) for E2EE; for 1:1 and small groups, Signal’s X3DH + Double Ratchet (with sender keys) is proven and efficient [23][24].

Hot-path storage favors wide-column/kv stores (Cassandra/DynamoDB) for predictable per-partition low latency and scale; use distributed SQL (Spanner/CockroachDB) for globally strong metadata/ACLs; use Elastic/OpenSearch for message search; store media in object storage with CDN [37][38][39][40][41][42].

## Assumptions and Open Constraints (to finalize with stakeholders)
- Deployment: cloud vs on‑prem vs hybrid; region set and data locality by user residency.
- SLOs: availability (e.g., ≥99.99%), latency budgets (client p50/p99 for send/receive), reconnect targets.
- Scale: peak concurrent connections, messages/sec, average and max message sizes; group sizes (small vs “mega”).
- Media: object size limits, supported formats, AV scan/transcode needs, CDN and egress budgets.
- Security/compliance: GDPR/CCPA, data localization, retention/TTL vs legal hold, lawful intercept policy (E2EE implications).
- Costs: unit economics for storage, egress, compute, search; managed vs self‑hosted trade-offs.
- Platforms: iOS/Android/Web/Desktop specifics; offline delivery and push reliability requirements; accessibility and moderation tooling needs.

## Reference Architecture Overview
Core components and their responsibilities:
- Gateway/Edge: Global anycast or latency-based DNS to nearest region; L7 proxies terminate TLS, upgrade to WebSockets/HTTP/2, and manage keepalives/draining [17][18][19].
- Real-time Transport: WebSockets (RFC 6455) for bi‑directional client transport; optional SSE for uni‑directional updates; WebRTC DataChannel for P2P/rooms; gRPC streaming in the mesh [12][13][14][15].
- Messaging/Eventing Backbone: Kafka (idempotent/transactional), Pulsar (transactions + dedup), NATS JetStream (durable streams + duplicate windows), or Redis Streams (fast per-inbox queues) [1][2][4][5][7][8][9][10].
- Hot Message Store: Cassandra or DynamoDB for append-only per-conversation ordered logs and per-user inbox materialization [37][38][39].
- Metadata/ACLs/Profiles: Distributed SQL (Spanner or CockroachDB) or sharded Postgres/MySQL (Citus/Vitess) depending on global consistency needs [40][41][34][35].
- Search: Elasticsearch/OpenSearch with streaming sinks from the backbone [42][43][44].
- Media: Client direct upload to object storage (S3/GCS/Azure) via presigned/resumable uploads; serve via CDN with signed URLs; support Range requests [45][47][46].
- Presence/Typing: Ephemeral edge caches + pub/sub aggregation to reduce fan-out storms (Slack Flannel-style) [16].
- Observability/SRE: Distributed tracing, metrics, logs with PII redaction; SLO burn alerts; consumer lag dashboards; dead-letter streams.

## Core Components and Data Flows

### Data Model (logical)
- Conversation log: (conversation_id, seq_no, message_id, sender_id, body, attachments, created_at)
- Inbox materialization: (user_id, conversation_id, seq_no, delivery_state, expires_at)
- Read state: (user_id, conversation_id, last_read_seq, last_delivered_seq)
- Indices: search index with message_id as document id and fields for filters (conversation_id, sender_id, timestamps, has_media)

### Write Path (exactly-once friendly)
1) Client → Gateway (TLS 1.3). Gateway authenticates and authorizes.
2) Message service persists authoritative row and publishes event(s):
- Kafka: within a Kafka transaction, produce to “conversation-log” (key=conversation_id) and to recipient “inbox” topics, then commit; producer uses idempotent mode and transactional.id; consumers read read_committed [1][2][3][51].
- Pulsar: produce to multiple topics in a transaction; enable broker dedup; consumers read only committed messages [4][5].
- NATS JetStream: publish with Nats-Msg-Id; configure DuplicateWindow on the stream for dedup; consumers ack and control flow [7][8].
- Redis Streams: XADD with explicit unique idempotency IDs or application keys; consumer groups ensure at-least-once; consumers apply idempotently [9][10].

3) Hot store write: Upsert/conditional write into the per-conversation log and per-user inbox materializations with unique message_id constraints to ensure idempotency [34][35][36][37].
4) Search indexing: Stream to Elasticsearch via Kafka Connect or Pulsar IO with deterministic document IDs for idempotent indexing [43][44].
5) Push fan-out: Real-time notification via the backbone to online clients; store durable inbox entries for offline clients.

### Read Path
- Clients subscribe over WebSockets to their per-user inbox streams and presence channels.
- Apply only messages with conv_seq = last_applied + 1; buffer small gaps; ack broker offsets/IDs post-apply; persist last_applied_seq per conversation [12][7][8][10].
- For history or catch-up, fetch from hot store by (conversation_id, seq range) and reconcile.

### Presence/Typing
- Clients heartbeat to nearest PoP; edge aggregates to reduce updates (publish deltas every N seconds to subscribers); coalesce and rate-limit typing start/stop with TTL expiry [16].

## Scalability Mechanisms

### Partitioning and Shard Keys
- Per-conversation ordering: route conversation_id to the same partition/shard (Kafka key hash; Pulsar key routing; JetStream stream subject; Cassandra/DynamoDB partition key) [1][4][7][37][38].
- Avoid hotspots for mega-groups: use bucketization (conversation_id → N buckets via Jump Consistent Hash), reassign buckets-to-partitions without changing keys; or use sub-shards per conversation [11].
- Per-user inbox: shard by user_id for read locality and stable inbox cursors.

### Fan-out Strategies
- Write-time fan-out (durable): materialize per-user inbox entries at write time (Kafka/Pulsar transactionally) [1][4].
- Read-time fan-out (transient): compute ephemeral fan-out at the edge for presence/typing to reduce write amplification [16].
- Hybrid: durable for messages; transient for signals.

### Backpressure and Flow Control
- Kafka: tune max.poll, pause/resume partitions; use cooperative rebalancing (KIP-429) to reduce churn on scale events [3][20].
- NATS JetStream: pull consumers, MaxAckPending, AckWait, ordered consumers [8].
- Transport: batch small messages with tight max delay; coalesce “typing”; drop non-critical signals under pressure [12][15][16].

### Load Balancing and Autoscaling
- L7 proxies (HAProxy/Envoy/NGINX) with long-lived connection timeouts and drain on deploy; support WebSocket upgrades and HTTP/2 [17].
- Global routing: anycast front door and/or latency-based DNS to nearest region; fail over by steering clients to healthy regions [18][19].
- Autoscale edges by connection count and p95 message rate; backends by consumer lag, CPU, and queue depth.

## Exactly-Once Delivery End-to-End

### Core Principles
- Stable message_id (128-bit UUID or Snowflake-like) generated once; reused on retries.
- Per-conversation monotonic seq_no allocated by single-writer partition or guarded by atomic allocation in storage.
- Idempotent producers/consumers; idempotent storage (UPSERT, LWT, conditional writes).
- Transactional outbox/inbox: either (a) DB outbox with a relay to the bus or (b) broker-native transactions if the bus is authoritative [49][1][4].

### Backbone-Specific Recipes
- Kafka:
  - enable.idempotence=true; acks=all; retries>0; max.in.flight<=5; transactional.id per producer; consumer isolation read_committed; Streams processing.guarantee=exactly_once_v2 [1][2][3][51].
  - End-to-end EOS across topics within a cluster; cross-cluster replication is async; rely on idempotent message_id at replicas [3].
- Pulsar:
  - Enable broker transactions and server-side dedup; produce to multiple topics atomically; consumers see only committed [4][5].
  - Geo-replication is namespace-level async; replicated subscriptions sync consumer positions; small duplicate windows possible on failover [6].
- NATS JetStream:
  - Use Nats-Msg-Id with DuplicateWindow to suppress duplicates; per-consumer ack; mirrors/sources for cross-domain replication [7][8][50].
- Redis Streams:
  - XADD with explicit IDs or app-level keys; consumer groups (XREADGROUP) for at-least-once; idempotent apply with last_applied_seq and dedup set [9][10].

### Storage Idempotency
- PostgreSQL: INSERT … ON CONFLICT (id) DO UPDATE/DO NOTHING; optionally SELECT FOR UPDATE on per-conversation sequence allocator [34].
- MySQL: INSERT … ON DUPLICATE KEY UPDATE [35].
- DynamoDB: PutItem with ConditionExpression attribute_not_exists(PK) (insert-once); UpdateItem with conditional version checks [36].
- Cassandra: LWT (IF NOT EXISTS) for dedup of message_id when strictly required; otherwise idempotent upserts and per-partition sequencing [37].

### Client Ack Path and Retries
- Broker ack confirms durable publish (Kafka acks=all; Pulsar transaction commit; JetStream ack) [1][4][7].
- App-level delivery receipts reference conv_seq; server treats receipts idempotently and updates last_delivered_seq; client resends un-acked receipts after reconnect.

### Limitations and Failure Scenarios
- Cross-region EOS: Kafka/Pulsar transactions do not span clusters/regions; async replication can replay; suppress with message_id and last_applied_seq [3][6].
- Dedup windows: NATS DuplicateWindow is bounded; configure ≥ max end-to-end retry/replication lag [7][50].
- Reordering on retries: use per-conversation sequencing with gap detection/buffering; fallback to fetch-missing after timeout [1][7][9].

## High Availability and Fault Tolerance

### Multi-AZ/Region Topologies
- Backbone: Multiple brokers/partitions across AZs; async geo-replication (Kafka MM2 or managed equivalents; Pulsar namespaces; NATS mirrors/sources) [3][6][50].
- Hot stores:
  - Cassandra: NetworkTopologyStrategy across racks/AZs; RF=3; LOCAL_QUORUM for low-latency consistency; multi-DC with LOCAL_QUORUM per region [37].
  - DynamoDB: Global tables—MREC (async, LWW) or MRSC (synchronous across region set) for strong global reads; understand constraints (e.g., TTL/LSI unsupported in MRSC) [38][53].
  - Spanner: regional/dual/multi-region with synchronous replication and external consistency; availability up to 99.999% for multi-region; write latency tracks WAN RTT [40].
  - CockroachDB: region-survival goals; follower reads for local staleness; global table/locality tuning [41].
- DR:
  - Postgres WAL archiving + PITR [54].
  - DynamoDB PITR (1–35 days), restore to new table [55].
  - Elasticsearch ILM and snapshots; Delete-by-query for erasure [48][56].

### RPO/RTO Targets (illustrative)
- Active-active with async replication: RPO seconds to minutes (replication lag), RTO seconds to minutes with client reconnect/resume.
- Strong global (Spanner/Cockroach region-survival): RPO ≈ 0 for committed data; RTO seconds for leader failover; higher p99 write latency [40][41].

## Security and Encryption

### Transport and Mesh Security
- TLS 1.3 everywhere; prefer modern AEAD ciphers; avoid legacy; be cautious with 0-RTT replay risks [21][22].
- WebSockets (RFC 6455), HTTP/2 gRPC keepalives tuned to avoid GOAWAYs; long-lived timeouts at the edge [12][15][17].
- Zero-trust mTLS inside the mesh; automate cert issuance/rotation (e.g., Istio) [52].

### Encryption at Rest and Key Management
- Envelope encryption: per-object DEKs (AES-GCM) wrapped by KEKs in KMS/HSM; audit and rotate KEKs; cryptographic erasure by destroying keys [29][30].
- Immutable retention/legal hold for compliance via object-lock/retention policies where applicable [31].

### End-to-End Encryption (E2EE) Options
- Signal-style:
  - 1:1: X3DH for async setup + Double Ratchet for per-message FS/PCS; device pre-keys enable offline delivery; sender keys for small/medium groups [24].
- MLS (RFC 9420):
  - Tree-based group keying with FS and PCS at O(log n) update cost; superior for large groups and frequent membership changes; multi-device leafs, epoch-based message protection [23].
- Cryptographic choices:
  - Ed25519 for signing/identity [26]; AEAD for content (AES-GCM or ChaCha20-Poly1305) with strict nonce uniqueness [27][28][29].
  - Attachments: random content key per object; encrypt client-side; wrap keys via session/group secret or HPKE to recipients/devices [25].
- Push privacy:
  - Treat push (APNs/FCM) as a wake-up signal; never include plaintext; use background/data notifications and fetch ciphertext on reconnect [32][33].
- Metadata minimization:
  - Store minimal routing metadata; consider sealed-sender-like designs; short-lived delivery tokens (design-specific; not standardized).

## Database and Storage Options: Comparison
| Option | Strengths | Consistency | Multi-Region Behavior | Latency/Throughput Notes | Operational Complexity | Cost | Best Use | Source URL |
|---|---|---|---|---|---|---|---|---|
| Apache Cassandra | Write-optimized, scale-out, per-partition ordering with clustering columns; TTL/TWCS | Tunable; LOCAL_QUORUM for strong in-region; LWT for CAS | Multi-DC via NetworkTopology; LOCAL_QUORUM avoids WAN | Sub-ms local writes; linear scale; compaction/repair impact | Manage nodes, compaction, repair, backups | Node-based; efficient at sustained high write rates | Hot message log and per-user inbox at scale | https://docs.datastax.com/en/cassandra-oss/3.x/cassandra/dml/dmlConfigConsistency.html |
| Amazon DynamoDB | Fully managed, auto-scale, conditional writes; Global Tables | Strong reads in-region; GSIs eventual | Global Tables MREC (async LWW) or MRSC (sync strong) | Single-digit ms for singleton ops (service-internal) | Serverless; index design and hot-partition avoidance | Pay-per-request/storage; GSIs add cost | Hot message log/inbox with managed ops | https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/GlobalTables.html |
| PostgreSQL (+Citus) | Rich SQL/ACID; UPSERT; native partitioning; Citus sharding | Strong per-shard; distributed xacts via Citus | Typically single write region + async replicas | Great for transactional metadata; WAN RTT hurts cross-region writes | Operate shards, replicas, migrations, PITR | Infra-based; predictable | Metadata/ACLs; smaller-scale message storage | https://www.postgresql.org/docs/9.5/sql-insert.html |
| MySQL (+Vitess) | Mature SQL; Vitess sharding and resharding | Strong per-shard; Vitess vindexes | Region pinning for writers; async replicas | Similar to Postgres trade-offs | Operate Vitess control plane | Infra-based | Metadata/ACLs | https://vitess.io/docs/19.0/reference/features/vindexes/ |
| Google Cloud Spanner | Strong global consistency, automatic sharding | External consistency (serializable) | Synchronous multi-region; leader placement | p99 writes track inter-region RTT | Fully managed | Premium | Global metadata/ACLs; billing | https://cloud.google.com/spanner/docs/true-time-external-consistency |
| CockroachDB | Serializable SQL; multi-region localities; follower reads | Serializable | Region-survival goals configurable | Follower reads reduce read latency (stale) | Operate cluster; backups/PCR | Infra-based | Global metadata; moderate write globalism | https://www.cockroachlabs.com/docs/stable/multiregion-survival-goals |
| MongoDB (sharded) | Flexible schema; TTL indexes | Read concern up to linearizable | Sharded with chunk migration | Good for metadata if shard key fits | Balance chunks, backups | Infra-based or Atlas | Profiles/permissions; not primary hot path | https://www.mongodb.com/docs/manual/core/index-ttl/ |
| Elasticsearch/OpenSearch | Full-text, filters, NRT ~1s | N/A (search system) | Cross-cluster search/replication options | Refresh interval controls NRT vs throughput | Index lifecycle mgmt; backfills | Infra or managed | Search message history | https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-refresh.html |

Citations in text: [37][38][34][35][40][41][53][42]

## Messaging Backbones: Delivery Guarantees and Scale
| Backbone | Ordering Model | Exactly-Once Ingredients | Cross-Region | Notes | Source URL |
|---|---|---|---|---|---|
| Kafka | Per-partition total order | Idempotent producer + Transactions + read_committed consumers | Async geo-replication; EOS not cross-cluster | Transactions span multi-topic writes and offset commits | https://docs.confluent.io/kafka/design/delivery-semantics.html |
| Pulsar | Key-ordered; partitions; BookKeeper for storage | Transactions + Broker dedup | Namespace-level async; replicated subscriptions | Can approach end-to-end EOS with both features enabled | https://pulsar.apache.org/docs/2.11.x/txn-use/ |
| NATS JetStream | Subject/stream ordering per consumer | Nats-Msg-Id + DuplicateWindow; idempotent consumers | Mirrors/sources across domains/leafnodes | Very low overhead; application-level EOS | https://docs.nats.io/nats-concepts/jetstream/streams |
| Redis Streams | Stream order; consumer groups track PEL | Explicit XADD IDs + idempotent apply | Active-Active CRDT modes exist (special rules) | Great for per-user inboxes/notifications | https://redis.io/docs/latest/commands/xadd/ |

Citations in text: [1][4][7][9][50]

## Real-Time Transport, Presence, and Client Sync
- WebSockets (RFC 6455): Full-duplex, ordered; heartbeat via ping/pong; long-lived connection handling at L7 LB [12][17].
- SSE: Uni-directional, simple, keepalives; use for notifications/deltas [13].
- WebRTC DataChannel (RFC 8831): SCTP over DTLS; per-message reliability/ordering tunable; useful for P2P/rooms [14].
- gRPC streaming: HTTP/2; tune keepalives; drain with GOAWAY on deploy [15].
- Presence/Typing: Edge aggregation and differential publish reduce fan-out storms (Slack’s Flannel) [16].
- Global routing: Anycast and latency-based DNS to the nearest region; reconnect with exponential backoff and cursor resume [18][19].

## Search, Indexing, Retention
- Stream messages from the backbone to Elasticsearch/OpenSearch via Kafka Connect or Pulsar IO; set deterministic _id for idempotency [43][44].
- Target near-real-time by ~1s refresh interval; increase interval for higher ingest throughput [42].
- Retention with ILM (hot/warm/cold/delete); Delete-by-query for erasure; tombstones from the source propagate deletes [48][56].

## Media Storage and Delivery
- Client direct uploads: Presigned URLs (S3), resumable uploads (GCS), SAS (Azure) [45].
- Serve via CDN with signed URLs/cookies; support Range requests for partial fetch [47][46].
- Encrypt objects by default; manage keys via KMS envelope encryption; store SHA-256 as metadata for integrity/Dedup [30][45].

## E2EE Key Management Details
- Signal option: device identity keys (Ed25519), signed pre-keys, one-time pre-keys; Double Ratchet for FS/PCS; sender keys for groups [24][26].
- MLS option: HPKE-based authenticated tree; O(log n) updates; strong PCS at group scale; epoch-bound application keys [23][25].
- AEAD: AES-GCM or ChaCha20-Poly1305; enforce unique nonces per key; track per-session sequence/counter [27][28][29].
- Attachments: per-object random DEK; encrypt client-side; wrap DEK via session/group secret or HPKE per recipient/device [25].
- Push privacy: APNs/FCM deliver no plaintext; app wakes to fetch ciphertext [32][33].

## Operational Considerations
- Moderation/abuse: reporting flows; hash/block lists for media; safe-search filters on thumbnails; rate limits and velocity checks.
- Observability: consumer lag, partition skew, queue depth, p50/p99 latencies per path; error budgets/SLOs; DLQs for poisoning.
- Compliance and data lifecycle: per-tenant locality; encryption keys scoped by region/tenant; retention/TTL with legal hold and immutable object storage [31].
- Privacy and logs: PII redaction; no content in logs; sampled traces; role-based access to observability.

## Implementation Blueprint (Step-by-Step)

1) Partitioning scheme
- Conversation topics keyed by conversation_id; inbox topics keyed by user_id (bucketized if needed). Use Jump Consistent Hash to map logical buckets to partitions for smooth re-sharding [11].

2) Exactly-once write (Kafka example)
- Producer: enable.idempotence=true; acks=all; retries>0; max.in.flight<=5; transactional.id set [1][2].
- Begin DB tx: insert message row (message_id unique); optionally allocate seq_no.
- Begin Kafka tx: produce to conversation-log and inbox topics; commit; then commit DB tx (or use transactional outbox pattern) [1][49].
- Consumers: read_committed; idempotently upsert into hot store using message_id constraints; ack only after durable apply [1][34][36][37].

3) Read/apply
- Client maintains per-conversation last_applied_seq; applies only next seq; requests gap fill on timeout; server tracks last_delivered_seq for receipts.

4) Presence/typing
- Heartbeats at edge aggregated to deltas; publish via pub/sub; rate limit to ≥5s; drop under backpressure [16].

5) Search/media
- Stream to Elasticsearch with deterministic document IDs; ILM policies; Delete-by-query for erasure [43][48][56].
- Presigned uploads; CDN with signed URLs; Range support [45][47][46].

6) Multi-region and failover
- Clients connect to nearest region via anycast/DNS; on region failure, reconnect to secondary; resume from last_applied_seq; suppress duplicates by message_id and seq_no [18][19][3][6].

## Selected Tooling and Services (illustrative)
| Layer | Examples | Why | Source URL |
|---|---|---|---|
| Messaging | Kafka, Pulsar, NATS JetStream, Redis Streams | Durable logs, transactions/dedup, ordered consumption | https://docs.confluent.io/kafka/design/delivery-semantics.html |
| Hot Store | Cassandra, DynamoDB | Predictable per-partition low latency; massive scale | https://docs.datastax.com/en/cassandra-oss/3.x/cassandra/dml/dmlConfigConsistency.html |
| Metadata | Spanner, CockroachDB, Postgres+Citus, MySQL+Vitess | Strong/global consistency or familiar SQL with sharding | https://cloud.google.com/spanner/docs/true-time-external-consistency |
| Search | Elasticsearch/OpenSearch | Full-text, filters, NRT | https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-refresh.html |
| Transport | WebSockets, gRPC, WebRTC | Real-time client/server streams | https://datatracker.ietf.org/doc/html/rfc6455 |
| Edge/LB | HAProxy/Envoy/NGINX; Route53; Anycast CDNs | Long-lived connections, draining, global routing | https://www.haproxy.com/documentation/haproxy-configuration-tutorials/load-balancing/websocket/ |
| Security | TLS 1.3; Istio mTLS | Strong transport security; mesh zero trust | https://datatracker.ietf.org/doc/html/rfc8446 |

## Exactly-Once Design Checklists
- Producer: stable message_id; idempotent publishing; transactions where supported [1][2][4].
- Consumer: per-key dedup cache/window; last_applied_seq persisted; ack after durable apply [7][9][10][37].
- Storage: UPSERT/conditional write/LWT on message_id; monotonic seq_no per conversation [34][35][36][37].
- Fan-out: transactional write-time fan-out for durable inbox; edge read-time fan-out for ephemeral signals [1][4][16].
- Cross-region: accept async replication; rely on idempotent IDs to suppress duplicates; replicated subscriptions where supported [3][6][50].
- Backpressure: bounded in-flight; pause/resume; rate-limit non-critical events [20][8][16].

---

### Sources
[1] Confluent: Kafka Delivery Semantics (Idempotent Producer and Transactions): https://docs.confluent.io/kafka/design/delivery-semantics.html  
[2] Apache Kafka KIP-98: Exactly Once Delivery and Transactions: https://cwiki.apache.org/confluence/display/KAFKA/KIP-98%20-%20Exactly%20Once%20Delivery%20and%20Transactional%20Messaging  
[3] Apache Kafka Documentation (3.6): https://kafka.apache.org/36/documentation/  
[4] Apache Pulsar Docs: Transactions - Use: https://pulsar.apache.org/docs/2.11.x/txn-use/  
[5] Apache Pulsar Docs: Message Deduplication Cookbook: https://pulsar.apache.org/docs/3.0.x/cookbooks-deduplication/  
[6] Apache Pulsar Docs: Concepts - Replication: https://pulsar.apache.org/docs/2.10.x/concepts-replication/  
[7] NATS JetStream Streams (DuplicateWindow, headers): https://docs.nats.io/nats-concepts/jetstream/streams  
[8] NATS JetStream Consumers: https://docs.nats.io/nats-concepts/jetstream/consumers  
[9] Redis Docs: XADD: https://redis.io/docs/latest/commands/xadd/  
[10] Redis Docs: XREADGROUP: https://redis.io/docs/latest/commands//xreadgroup/  
[11] Jump Consistent Hash (Lamping, Veach): https://arxiv.org/abs/1406.2294  
[12] RFC 6455: The WebSocket Protocol: https://datatracker.ietf.org/doc/html/rfc6455  
[13] WHATWG: Server-Sent Events: https://html.spec.whatwg.org/dev/server-sent-events.html  
[14] RFC 8831: WebRTC Data Channels: https://www.rfc-editor.org/rfc/rfc8831  
[15] gRPC Keepalive: https://grpc.io/docs/guides/keepalive/  
[16] Slack Engineering: Flannel — Edge cache to make Slack scale: https://slack.engineering/flannel-an-application-level-edge-cache-to-make-slack-scale/  
[17] HAProxy: Load Balancing WebSocket Applications: https://www.haproxy.com/documentation/haproxy-configuration-tutorials/load-balancing/websocket/  
[18] Cloudflare Blog: Anycast architecture: https://blog.cloudflare.com/cloudflares-architecture-eliminating-single-p/  
[19] AWS Route53: Latency-based routing policy: https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/routing-policy-latency.html  
[20] Apache Kafka KIP-429: Incremental Rebalance Protocol: https://cwiki.apache.org/confluence/display/KAFKA/KIP-429%3A%2BKafka%2BConsumer%2BIncremental%2BRebalance%2BProtocol  
[21] RFC 8446: TLS 1.3: https://datatracker.ietf.org/doc/html/rfc8446  
[22] RFC 9325: Recommendations for Secure Use of TLS and DTLS: https://www.rfc-editor.org/rfc/rfc9325  
[23] RFC 9420: Messaging Layer Security (MLS): https://www.ietf.org/rfc/rfc9420.html  
[24] Signal Double Ratchet Specification: https://signal-ios.org/docs/specifications/doubleratchet/index.html  
[25] RFC 9180: Hybrid Public Key Encryption (HPKE): https://www.ietf.org/rfc/rfc9180.html  
[26] RFC 8032: Edwards-Curve Digital Signature Algorithm (EdDSA): https://www.rfc-editor.org/rfc/rfc8032  
[27] RFC 5116: An Interface and Algorithms for Authenticated Encryption (AEAD): https://datatracker.ietf.org/doc/rfc5116/  
[28] RFC 8439: ChaCha20-Poly1305: https://datatracker.ietf.org/doc/rfc8439/  
[29] NIST SP 800-38D: GCM and GMAC: https://csrc.nist.gov/pubs/sp/800/38/d/final  
[30] Google Cloud KMS – Envelope Encryption: https://cloud.google.com/kms/docs/envelope-encryption  
[31] Amazon S3 Object Lock: https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock.html  
[32] Apple Remote Notifications Payload Reference: https://developer.apple.com/library/archive/documentation/NetworkingInternet/Conceptual/RemoteNotificationsPG/CreatingtheNotificationPayload.html  
[33] Firebase Docs: Android message handling: https://firebase.google.com/docs/cloud-messaging/android/receive  
[34] PostgreSQL INSERT (UPSERT ON CONFLICT): https://www.postgresql.org/docs/9.5/sql-insert.html  
[35] MySQL INSERT ON DUPLICATE KEY UPDATE: https://dev.mysql.com/doc/mysql/en/insert-on-duplicate.html  
[36] Amazon DynamoDB PutItem (ConditionExpression): https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_PutItem.html  
[37] Apache Cassandra Consistency (QUORUM/LOCAL_QUORUM/LWT): https://docs.datastax.com/en/cassandra-oss/3.x/cassandra/dml/dmlConfigConsistency.html  
[38] Amazon DynamoDB Global Tables (MREC): https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/GlobalTables.html  
[39] Amazon DynamoDB Read Consistency: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.ReadConsistency.html  
[40] Google Cloud Spanner TrueTime and External Consistency: https://cloud.google.com/spanner/docs/true-time-external-consistency  
[41] CockroachDB Multi-region Survival Goals: https://www.cockroachlabs.com/docs/stable/multiregion-survival-goals  
[42] Elasticsearch Near Real Time and Refresh: https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-refresh.html  
[43] Confluent Kafka Connect Elasticsearch Sink: https://docs.confluent.io/kafka-connectors/elasticsearch/current/overview.html  
[44] Apache Pulsar IO Elasticsearch Sink: https://pulsar.apache.org/docs/3.3.x/io-elasticsearch-sink/  
[45] Amazon S3 Pre-signed URLs: https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html  
[46] RFC 7233: HTTP Range Requests: https://www.rfc-editor.org/rfc/rfc7233  
[47] Amazon CloudFront Signed URLs for Private Content: https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-signed-urls.html  
[48] Elasticsearch Index Lifecycle Management (ILM): https://www.elastic.co/guide/en/elasticsearch/reference/current/ilm-index-lifecycle.html  
[49] Transactional Outbox Pattern: https://microservices.io/patterns/data/transactional-outbox.html  
[50] NATS JetStream Source and Mirror (Cross-Domain/Leafnodes): https://docs.nats.io/nats-concepts/jetstream/source_and_mirror  
[51] Confluent: Kafka Streams Concepts (processing.guarantee): https://docs.confluent.io/platform/current/streams/concepts.html  
[52] Istio Security Concepts (mTLS): https://istio.io/latest/docs/concepts/security/