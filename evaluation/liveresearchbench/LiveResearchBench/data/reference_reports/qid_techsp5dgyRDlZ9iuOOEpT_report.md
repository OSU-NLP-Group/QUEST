# Resilient Offline-First React SPA: Reference Architecture and Implementation Plan

## Executive Summary

Design a React SPA that is fully usable offline and seamlessly syncs user‑generated content (UGC) when connectivity returns. Core tenets:
- App shell served and updated by a service worker (Workbox) with safe lifecycle practices, navigation fallback, and precise caching for static assets and runtime data [1][2][3][4][9][10].
- Durable client‑side data layer: IndexedDB for structured metadata and the Origin Private File System (OPFS) for large media, with encryption at rest via Web Crypto (AES‑GCM) and capability-aware fallbacks [16][18][19][21][22][23][24][25].
- Transport chosen per use case: REST/GraphQL over HTTP/2 or HTTP/3 for feed and CRUD with validators; WebSockets (or GraphQL subscriptions) for live updates; robust outbox with idempotency keys, conditional requests, backoff with jitter, and conflict resolution (LWW, field‑level merges, or CRDTs like Automerge/Yjs for collaborative content) [43][44][45][46][49][50][51][52][53][54][55][56][57][58][59][61][63][64][65][66][67][68][69][109].
- Performance: explicit caching strategies per resource class, prefetch/priorities, code splitting, virtualization, media optimization, compression (Brotli + Compression Streams), and field monitoring via Core Web Vitals [12][13][14][15][97][98][110][111][112][113][114][99][100][115][116].
- Security: OAuth 2.0 Authorization Code + PKCE with OIDC, session cookies (HttpOnly/SameSite), CSRF and XSS defenses (CSP, Trusted Types, SRI), WebAuthn for strong auth, TLS 1.3 and HSTS, encrypted local stores, and hardened sync endpoints with rate limiting and replay protection (idempotency keys) [75][76][77][78][79][81][82][83][5][84][85][21][86][87][88][89][91][96][109][73][74].
- Scalability: normalized client caches (Redux Toolkit Query/TanStack Query/Apollo), backpressure via Streams, circuit breakers/feature flags, CDN/edge for assets, resilient WebSocket reconnection, and resumable/chunked uploads (tus or multipart) [119][120][121][101][44][53][122][123].

Browser support: service workers, IndexedDB, Web Workers, Push, and Web Crypto are broadly available; one‑off Background Sync is Chromium‑only, Periodic Background Sync remains experimental, and iOS PWAs continue under WebKit (DMA updates) [42][16][33][31][32][29][30][71].

Deprecations/changes (Sep 2025): Periodic Background Sync remains limited; Background Fetch is Chromium‑only; Apple retained Home Screen web apps in the EU (WebKit-based) [30][72][71].


## 1) Offline‑First Components

### 1.1 Service worker strategy: installation, updates, navigation fallback, preload, versioning/rollback
- Lifecycle and update model: install → waiting → activate. Browsers check for updated worker scripts on navigations and events; a new worker activates only after existing clients are closed or when skipWaiting is called. If the new script fails, the previous worker remains active (safe rollback) [1]. Show “update available” UI instead of forcing skipWaiting to avoid cross‑version asset races [70][8].
- App shell and SPA routing: precache a content‑hashed app shell, route navigations to it (Workbox NavigationRoute), and serve contextual offline fallbacks (e.g., offline.html) [2][3][4][9][10].
- Navigation preload: reduce navigation latency by letting the browser start the network request while the SW initializes; enable via registration.navigationPreload and consume event.preloadResponse; add Vary: Service‑Worker‑Navigation‑Preload if responses vary [5][6][7].
- Versioning: use workbox‑precaching with revisioned manifests to update only changed assets and clean up old caches on activate [9][10].
- Rollback: redeploy a known-good worker at the same URL; the default lifecycle prevents broken activations [1].

Example (Workbox SPA shell, runtime strategies, fallback, preload):
```js
// sw.js
import {precacheAndRoute, createHandlerBoundToURL} from 'workbox-precaching';
import {registerRoute, NavigationRoute, setCatchHandler} from 'workbox-routing';
import {StaleWhileRevalidate, NetworkFirst} from 'workbox-strategies';
import {ExpirationPlugin} from 'workbox-expiration';

precacheAndRoute(self.__WB_MANIFEST); // hashed assets [9][10]

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    if (self.registration.navigationPreload) {
      await self.registration.navigationPreload.enable(); // [5][7]
    }
    // await self.clients.claim(); // optional [70]
  })());
});

// SPA navigations -> app shell
const appShellHandler = createHandlerBoundToURL('/index.html');
registerRoute(new NavigationRoute(appShellHandler)); // [3]

// Runtime caches
registerRoute(
  ({request}) => request.destination === 'image',
  new StaleWhileRevalidate({
    cacheName: 'images',
    plugins: [new ExpirationPlugin({maxEntries: 200, maxAgeSeconds: 60*60*24*30})],
  }),
); // [13][15]

// Optional network-first for HTML navigations (when not precaching)
registerRoute(({request}) => request.mode === 'navigate', new NetworkFirst({cacheName: 'html'})); // [13]

// Offline fallback
setCatchHandler(async ({event}) => {
  if (event.request.destination === 'document') return appShellHandler(event);
  return Response.error();
}); // [4]
```

### 1.2 Cache Storage vs HTTP caching; precache vs runtime cache
- The Cache API is script‑controlled and does not automatically enforce HTTP cache semantics; combine with explicit strategies in the SW (network‑first, cache‑first, stale‑while‑revalidate) [11][13][14].
- Use HTTP caching headers (Cache‑Control, ETag, Vary) for CDN/built‑in browser caches. SW runtime caching can revalidate independently of HTTP cache rules where needed [12][14].
- Precache deterministic, immutable build assets; use runtime caching with expiration and quota‑aware policies for API responses and media [9][10][13][15].

### 1.3 IndexedDB data modeling for UGC; large binaries and OPFS; encryption at rest
- IndexedDB schema/versioning: create object stores and indexes in onupgradeneeded; transactions and asynchronous API are required for consistency [16].
- Large media: store binaries in OPFS and keep references/metadata in IndexedDB for performance and stability across browsers (including Safari) [18][19].
- Thin wrapper library: idb provides a small Promise-based abstraction for IndexedDB [20].
- Encryption at rest: encrypt payloads with AES‑GCM before writing to IndexedDB/OPFS; use unique IVs per record and non‑extractable keys; derive keys via PBKDF2 or HKDF as appropriate [21][88][86][87].

Sketch (IDB metadata + OPFS media):
```js
import {openDB} from 'idb'; // [20]
const db = await openDB('ugc', 1, { upgrade(db){ db.createObjectStore('posts', {keyPath:'id'}); db.createObjectStore('media', {keyPath:'id'}); } });

const root = await navigator.storage.getDirectory(); // OPFS [18]
```

### 1.4 Storage quotas/eviction and persistence
- Storage is best‑effort by default and can be evicted under pressure; use StorageManager.estimate() to show usage/quota and request persistence via StorageManager.persist() to reduce eviction risk [22][24][23][25][26].
- Handle QuotaExceededError by LRU cleanup (e.g., delete older caches/media) and user prompts to free space or limit “save offline” scope [25][26].

### 1.5 Capability detection and graceful degradation
- Feature‑detect: service workers (navigator.serviceWorker), CacheStorage, IndexedDB, Background Sync (registration.sync), Periodic Sync (registration.periodicSync), Push/Notifications, Web Crypto, Web Workers/SharedWorkers, OPFS (navigator.storage.getDirectory). Treat navigator.onLine only as a hint [27][28][29][30][31][32][33][104][18][34][22].
- Provide fallbacks when features are absent (e.g., manual retry timers instead of Background Sync) [29][35].

### 1.6 Background Sync availability and limitations
- One‑off Background Sync (SyncManager) is supported in Chromium-based browsers; feature‑detect and fallback elsewhere [35][29].
- Periodic Background Sync remains experimental and only in Chromium variants; do not rely on it for critical flows [30].
- Workbox Background Sync provides a durable replay queue and graceful fallback when native APIs are absent [38].

### 1.7 Web Workers/Shared Workers for data processing
- Use Dedicated Workers for heavy client tasks (image transcoding, CRDT merges) to keep the UI responsive; use SharedWorker to coordinate across tabs of the same origin when supported [33][104].
- Transfer large data efficiently using transferable objects to avoid copies [108].

### 1.8 Offline error handling and UX
- Provide clear offline indicators, skeleton states, explicit “retry”/“save for later,” and a custom offline fallback page to avoid generic browser errors [39][40][41].


## 2) Data Sync and Conflict Resolution

### 2.1 Transport options and when to use them
- REST over HTTP/2 or HTTP/3 for feeds and CRUD, leveraging HTTP semantics, conditional requests, and caching (validators via ETag/If‑None‑Match) [43][44][45][46].
- GraphQL for flexible querying/mutations over HTTP; GraphQL over HTTP draft standardizes transport details [49][50][51][105][106].
- WebSockets (RFC 6455) for low‑latency bidirectional streams (presence, typing, notifications). Bootstrap via Extended CONNECT in HTTP/2 or HTTP/3 where applicable [53][54][55].
- GraphQL subscriptions over WebSockets for live updates; prefer graphql‑ws over archived subscriptions‑transport‑ws [52][107].

### 2.2 Offline queue/outbox with idempotency, batching, dedupe, retry
- Durable queue: persist mutations in IndexedDB with ordering and per‑resource dedupe; request persistent storage on first use [16][23].
- Idempotency: for non‑idempotent operations, attach an Idempotency‑Key so the server can deduplicate within a window; use HTTP preconditions (If‑Match with strong ETag) to avoid lost updates [109][59][61].
- Retry policy: exponential backoff with jitter; retry on network errors/408/429/5xx; surface 409/412 to trigger conflict flows [29][30][32].
- Batching: coalesce multiple updates to the same resource and send in order; preserve causal ordering per resource to minimize conflicts [43].

Example (Workbox Background Sync for POST/PUT/DELETE):
```js
import {registerRoute} from 'workbox-routing';
import {NetworkOnly} from 'workbox-strategies';
import {BackgroundSyncPlugin} from 'workbox-background-sync';

const outbox = new BackgroundSyncPlugin('ugc-outbox', { maxRetentionTime: 24*60 }); // [38]
const retryOn5xx = { fetchDidSucceed: ({response}) => { if (response.status >= 500) throw new Error('Server error'); return response; } };

registerRoute(
  ({url, request}) => url.pathname.startsWith('/api/posts') && ['POST','PUT','DELETE'].includes(request.method),
  new NetworkOnly({ plugins: [retryOn5xx, outbox] }),
);
```

### 2.3 Delta/patch‑based sync and versioning
- Partial updates: use HTTP PATCH for partial modifications, with JSON Patch (RFC 6902) or JSON Merge Patch (RFC 7396) depending on semantics and complexity [56][57][58].
- Validators: prefer ETag validators and preconditions (If‑Match/If‑None‑Match) for safe updates; timestamps are weaker [61][59][60][62][44].

### 2.4 Logical clocks and conflict detection
- Lamport clocks provide a total order compatible with causality; vector clocks detect concurrency across replicas and aid conflict detection/merging [63][64].

### 2.5 CRDT vs OT for concurrent edits; selection guidance
- CRDTs converge without central coordination (strong eventual consistency) and are well-suited for offline‑first multi‑writer systems [65][66].
- OT can work with centralized servers for editors but has a higher correctness burden, especially for rich text [67].
- Libraries:
  - Automerge: JSON‑like CRDT, compact binary format, works well for structured UGC and text [68].
  - Yjs: high‑performance CRDT with rich editor bindings and network‑agnostic sync [69].
- Guidance:
  - Structured documents/notes, multi‑device editors: CRDTs (Yjs/Automerge) [65][69][68].
  - Simple fields or append‑only logs: server‑authoritative merges with ETags are often simpler [61][43].

### 2.6 Server‑authoritative vs client‑merge models
- Server‑authoritative merges centralize business rules and audit but require offline outbox + conflict handling on reconnect [43][73][74].
- Client‑merge (CRDT) yields low‑latency local edits and offline resilience; ensure authorization/validation of incoming ops on the server [65][73][74].

### 2.7 Merge policies and conflict UI
- Last‑writer‑wins (attach logical clock + replica ID); suitable for non-critical fields like lastViewedAt but may lose intent [63].
- Field‑level merges with per‑field versions; prompt user choice when concurrent edits detected [59][61].
- Type-specific merges (sets with tombstones, append‑only logs) using CRDT patterns (e.g., OR‑Set) [65].
- UI: side‑by‑side diffs for text; per‑field “theirs vs mine” with author/time metadata; Yjs editor guidance offers patterns for versioning and conflict surfaces [69].

Example (simplified field‑level LWW merge for JSON):
```js
function mergeLWW(local, remote) {
  const merged = {...remote};
  for (const k of Object.keys(local)) {
    const l = local[k], r = remote[k];
    if (l && r && l.updatedAt && r.updatedAt) {
      merged[k] = (l.updatedAt >= r.updatedAt) ? l : r;
    } else if (l && !r) {
      merged[k] = l;
    }
  }
  return merged;
} // Clock fields derive from Lamport or server timestamps [63][61]
```


## 3) Performance Considerations

### 3.1 Caching policy matrix and invalidation
- Immutable build artifacts: content‑hash filenames; Cache‑Control: immutable/long max‑age; update via SW precache (revisioned) [9][10][12].
- API responses: choose network‑first or stale‑while‑revalidate depending on staleness tolerance; honor validators and leverage CDN [13][14][44].
- Media: cache‑first or stale‑while‑revalidate with explicit expiration to bound storage [13][15].

### 3.2 Prefetching and prioritization
- Use fetchpriority for critical images and appropriate rel=preload/modulepreload for critical resources; priority is a browser hint, not a guarantee [97][98].

### 3.3 Code splitting and lazy loading
- Use React.lazy and Suspense for route/component-level splitting; avoid unsafe-eval to keep CSP tight [110][111][82].

### 3.4 List virtualization and pagination
- For large feeds, use virtualization (e.g., react‑window) and server pagination/infinite scroll; combine with cache policies to keep memory bounded [112][44].

### 3.5 Image/video optimization
- Responsive images (srcset/sizes) and modern codecs (WebP/AVIF) based on client support; stream where needed via MSE pipelines when applicable [113][114].

### 3.6 Compression
- Enable Brotli on the server (RFC 7932) and consider client‑side compression with the Compression Streams API for large uploads [99][100].

### 3.7 Background sync optimization
- Batch queued mutations and schedule retries with exponential backoff; replay on SW activation/visibilitychange/focus, not just on line status [13][29][34].

### 3.8 Measuring and budgeting
- Track Core Web Vitals (LCP, CLS, INP) in the field using the web‑vitals library; report via sendBeacon/keepalive [115][116][117].


## 4) Security Measures

### 4.1 Threat model for offline‑first SPAs
- Service workers run only on secure contexts, intercept within scope, and cannot access the DOM; design CSRF/XSS defenses at the app and server layers [42][82].
- Use SRI to integrity‑check external scripts/styles [5].
- Treat client storage as user‑controlled; encrypt sensitive data before writing [21][18][25].

### 4.2 Authentication for SPAs
- Use OAuth 2.0 Authorization Code with PKCE (RFC 7636) and OpenID Connect; follow OAuth 2.0 Security BCP (BCP 240) and Browser‑Based Apps guidance [75][78][76][77].
- Prefer server‑side tokens with a BFF and session cookies (HttpOnly, Secure, SameSite) to reduce token exfiltration risk; avoid storing bearer tokens in Web Storage/IDB per RFC 6750 [79][80][81].

### 4.3 CSRF and XSS mitigations
- CSRF: SameSite cookies (Lax/Strict), synchronizer or double‑submit tokens, and Origin/Referer checks [81][79].
- XSS: strong CSP (script-src 'self'; no unsafe‑eval), Trusted Types for DOM sinks, contextual output encoding [82][83][81].

### 4.4 WebAuthn/FIDO2
- Offer passkeys/WebAuthn for strong MFA or passwordless flows; rely on W3C WebAuthn Level 2/3 for platform/roaming authenticators and PRF extensions [84][85].

### 4.5 Encryption at rest (IndexedDB/OPFS)
- Use AES‑GCM with unique IVs per encryption; derive/wrap keys with PBKDF2 or HKDF and keep CryptoKeys non‑extractable except for allowed wrap/unwrap operations [21][86][87][88].

### 4.6 Transport security and integrity
- Enforce TLS 1.2+ and prefer TLS 1.3; deploy HSTS to prevent downgrade attacks [89][91][90].
- Consider COOP/COEP/CORP to enable powerful APIs and isolate contexts when needed [92].

### 4.7 Dependency integrity and supply chain
- Lock dependency integrity (npm package‑lock) and adopt signed provenance (Sigstore/Cosign) aligned to SLSA; follow OWASP Software Supply Chain guidance [93][94][95][118].

### 4.8 Secure sync endpoints
- Apply authorization and validation server‑side; rate-limit and defend against replay with Idempotency‑Key; cap batch/payload sizes; follow OWASP API Security Top 10 (2023) [96][109].


## 5) Scalability Strategies

- Client state: normalized caches with RTK Query, TanStack Query, or Apollo; tune staleTime/cacheTime and invalidate precisely [119][120][121].
- Backpressure: use Streams API; avoid uncontrolled buffering for uploads/downloads [101].
- Circuit breakers and feature flags: gate background sync, WS reconnect, and heavy fetches to contain blast radius; keep kill switches server‑configurable [44].
- CDN/edge: serve hashed static assets with long-lived cache; honor ETags and SW precache for app shell [12][9][10][44].
- WebSocket reconnection: exponential backoff with jitter; heartbeats/pings; server caps for fan‑out [53][29].
- Large media uploads: resumable protocols (tus) or multipart uploads (e.g., S3 multipart); add idempotency per chunk and checksums [122][123].


## 6) Browser/Platform Support Matrix (critical capabilities)

- Service Workers: supported on modern Chrome/Edge/Firefox/Safari on desktop/mobile PWAs [42].
- Cache API and CacheStorage: broadly supported [11][28].
- IndexedDB: broadly supported [16].
- Background Sync (one‑off): Chromium/Edge; not Safari/Firefox—feature‑detect [35][29].
- Periodic Background Sync: experimental/Chromium‑only—feature‑detect and avoid reliance [30].
- Push + Notifications: supported in major browsers with SW; Web Push on iOS/iPadOS requires Home Screen web app and permission [31].
- Web Crypto (SubtleCrypto): widely supported, including in workers [32].
- Web Workers/Shared Workers: broadly supported with some platform caveats; verify for mobile [33][104].
- OPFS (File System API): supported in modern browsers; WebKit documents OPFS availability [18][19].
- iOS PWAs (EU): Apple restored Home Screen web apps (WebKit-based) in iOS/iPadOS 17.4 in the EU [71].


## 7) Key Flow Sequence Diagrams

### 7.1 First load and SW install/update
```
Browser -> Network: GET /
Page -> Browser: register service worker
SW (install): precache assets (hashed) [9][10]
SW (waiting) -> (activate when safe or skipWaiting carefully) [1][8]
SW (activate): cleanup old caches; enable navigation preload [1][5]
UI: prompt "Update available—Reload" when updatefound [70]
```

### 7.2 Offline create with outbox and replay on reconnect
```
User/UI -> LocalStore: edit/create UGC
LocalStore -> IndexedDB: write data + enqueue mutation with Idempotency-Key [16][109]
UI -> UI: render optimistic state
(Reconnect or SW start)
SW/Sync -> Server: POST/PUT/PATCH with If-Match/Idempotency-Key [59][61][109]
Server -> SW/Sync: 2xx + ETag or 409/412
alt conflict: 409/412 -> fetch latest -> merge (LWW/field/CRDT) -> retry [59][63][65]
SW/Sync -> IndexedDB: mark done; update ETag/version [16][61]
```

### 7.3 Collaborative edit bootstrapping (CRDT + WebSocket)
```
Client -> Server: GET snapshot (ETag) [61]
Client -> Server: WebSocket CONNECT (RFC 6455/8441/9220) [53][54][55]
Server <-> Client: exchange CRDT ops (Automerge/Yjs) [68][69]
Clients: apply ops locally (offline-capable), persist in IDB
```


## 8) Example Code Snippets

### 8.1 Background Sync feature detection
```js
const reg = await navigator.serviceWorker.ready; // [27]
if (reg.sync) await reg.sync.register('sync-outbox'); // [29]
```

### 8.2 Custom IndexedDB outbox (simplified)
```js
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('app-outbox', 1); // [16]
    req.onupgradeneeded = () => {
      const db = req.result;
      const s = db.createObjectStore('outbox', { keyPath: 'id' });
      s.createIndex('byCreatedAt', 'createdAt');
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
```

### 8.3 AES‑GCM encryption for IDB payloads
```js
async function encryptRecord(key, obj) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const pt = new TextEncoder().encode(JSON.stringify(obj));
  const ct = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, pt);
  return { iv, ct: new Uint8Array(ct) };
} // [21]
```

### 8.4 Conditional request with Idempotency‑Key
```js
async function sendWithIdempotency(url, init={}) {
  const headers = new Headers(init.headers || {});
  const method = (init.method || 'GET').toUpperCase();
  if (['POST','PUT','PATCH','DELETE'].includes(method)) {
    headers.set('Idempotency-Key', crypto.randomUUID()); // [109]
  }
  if (init.ifMatch) headers.set('If-Match', init.ifMatch); // [59]
  const res = await fetch(url, {...init, headers});
  return res;
}
```


## 9) Test Plan

- Unit/integration: mock IndexedDB (idb harness) and SW fetch handlers; assert outbox enqueue/drain semantics [16][13].
- E2E offline simulation: Chrome DevTools network emulation; Playwright “offline” toggle to verify the app shell and data survive reloads [42][124].
- Performance: Lighthouse lab checks and field Web Vitals via web‑vitals; export dashboards by URL/route [115][116].
- Security: CSP Report‑Only to tune policy, ZAP scans for API and XSS/CSRF issues, and OAuth flow tests; verify cookies’ SameSite/Secure/HttpOnly [82][125][79].


## 10) Observability and Telemetry (privacy‑aware)

- Queue telemetry in IDB while offline and flush on sync; send via sendBeacon or fetch with keepalive [117].
- Collect Core Web Vitals in the field (web‑vitals); sample to respect user privacy and consent [115][116].
- Use the Reporting API for CSP/reporting endpoints to detect violations in production [126].


## 11) Rollout and Migration

- Progressive enhancement: ship a network‑first app that upgrades to SW offline where supported; ensure a clean update path (updatefound UI; avoid auto‑skipWaiting) [1][70].
- Canary and feature flags: staged rollout of SW caching, background sync, and CRDT features, with rapid rollback plans [13].
- Data migrations: version object stores; write idempotent migration scripts in onupgradeneeded; test forward/backward compatibility [16].


## 12) Key Risks and Mitigations

- Storage eviction: request persistence, monitor quota, and implement LRU cleanup; ensure lossless re‑sync paths [23][24][25][26].
- Background Sync gaps: provide foreground retry fallbacks and Workbox Background Sync plugin [29][30][38].
- PWA platform changes: track iOS PWA policy changes (EU DMA) and WebKit quotas; design without assuming non‑WebKit engines on iOS [71].
- Security: prevent token exfiltration (cookies, PKCE), enforce CSP/Trusted Types, encrypt local data, and apply OWASP API Top 10 controls on sync endpoints [75][79][82][83][21][96].
- Update hazards: avoid skipWaiting by default; show update UI to prevent cross‑version asset mismatches [1][8][70].
- Large uploads on flaky networks: resumable/chunked protocols with idempotent parts and checksum verification [122][123].


## 13) Implementation Checklist (Phased)

- Phase 1 (foundation):
  - React app shell; SW with Workbox injectManifest; hashed assets; navigation fallback; navigation preload [2][9][10][3][5].
  - IndexedDB schema + OPFS integration; Web Crypto AES‑GCM encrypt/decrypt utilities [16][18][21].
  - HTTP cache headers on CDN and API: ETag/Vary/Cache‑Control [44][12][61].
- Phase 2 (UGC offline and sync):
  - Outbox queue (IDB) + Background Sync (Workbox plugin) with idempotency keys and If‑Match [38][59][109].
  - Conflict detection (ETag/412/409) and UI; implement merge policies appropriate to content type [59][61][63][65].
- Phase 3 (realtime and collaboration):
  - WebSocket gateway and reconnect logic; GraphQL subscriptions where applicable [53][52].
  - Integrate CRDTs (Yjs/Automerge) for collaborative content; persist updates in IDB [69][68].
- Phase 4 (security hardening):
  - PKCE/OIDC sign‑in, BFF session cookies; CSP/Trusted Types; SRI; WebAuthn bootstrap [75][78][82][83][5][84].
- Phase 5 (performance + observability):
  - Prefetch/priorities; code splitting; virtualization; Brotli + Compression Streams; Core Web Vitals collection [97][98][110][111][112][99][100][115][116].


### Sources
[1] Service worker lifecycle (web.dev): https://web.dev/service-worker-lifecycle/  
[2] App shell model (Workbox): https://developer.chrome.com/docs/workbox/app-shell-model/  
[3] Workbox routing: https://developer.chrome.com/docs/workbox/modules/workbox-routing/  
[4] Workbox fallback responses: https://developer.chrome.com/docs/workbox/managing-fallback-responses  
[5] NavigationPreloadManager (MDN): https://developer.mozilla.org/en-US/docs/Web/API/NavigationPreloadManager  
[6] Service-Worker-Navigation-Preload header (MDN): https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Service-Worker-Navigation-Preload  
[7] Workbox navigation-preload: https://developer.chrome.com/docs/workbox/modules/workbox-navigation-preload/  
[8] skipWaiting (MDN): https://developer.mozilla.org/en-US/docs/Web/API/ServiceWorkerGlobalScope/skipWaiting  
[9] Workbox precaching module: https://developer.chrome.com/docs/workbox/modules/workbox-precaching  
[10] Precaching with Workbox: https://developer.chrome.com/docs/workbox/precaching-with-workbox/  
[11] Cache API (MDN): https://developer.mozilla.org/docs/Web/API/Cache  
[12] Cache-Control (MDN): https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control  
[13] Workbox strategies: https://developer.chrome.com/docs/workbox/modules/workbox-strategies/  
[14] SW vs HTTP caching (web.dev): https://web.dev/articles/service-worker-caching-and-http-caching  
[15] Workbox expiration: https://developer.chrome.com/docs/workbox/modules/workbox-expiration/  
[16] Using IndexedDB (MDN): https://developer.mozilla.org/docs/Web/API/IndexedDB_API/Using_IndexedDB  
[17] —  
[18] OPFS (MDN): https://developer.mozilla.org/docs/Web/API/File_System_API/Origin_private_file_system  
[19] WebKit OPFS availability: https://webkit.org/blog/12257/the-file-system-access-api-with-origin-private-file-system/  
[20] idb library: https://github.com/jakearchibald/idb  
[21] SubtleCrypto.encrypt (AES-GCM) (MDN): https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/encrypt  
[22] StorageManager (MDN): https://developer.mozilla.org/en-US/docs/Web/API/StorageManager  
[23] StorageManager.persist (MDN): https://developer.mozilla.org/en-US/docs/Web/API/StorageManager/persist  
[24] StorageManager.estimate (MDN): https://developer.mozilla.org/docs/Web/API/StorageManager/estimate  
[25] Storage quotas and eviction (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria  
[26] Browser storage limits & eviction (MDN): https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API/Browser_storage_limits_and_eviction_criteria  
[27] Navigator.serviceWorker (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Navigator/serviceWorker  
[28] CacheStorage (MDN): https://developer.mozilla.org/docs/Web/API/CacheStorage  
[29] SyncManager.register (MDN): https://developer.mozilla.org/en-US/docs/Web/API/SyncManager/register  
[30] Periodic Background Sync (MDN): https://developer.mozilla.org/en-US/docs/Web/API/ServiceWorkerRegistration/periodicSync  
[31] Notifications API (MDN): https://developer.mozilla.org/docs/Web/API/Notifications_API  
[32] SubtleCrypto.decrypt (MDN): https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/decrypt  
[33] Web Workers API (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API  
[34] Navigator.onLine (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Navigator/onLine  
[35] Background Synchronization API (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Background_Synchronization_API  
[36] —  
[37] —  
[38] Workbox background sync: https://developer.chrome.com/docs/workbox/modules/workbox-background-sync  
[39] Offline UX guidelines (web.dev): https://web.dev/articles/offline-ux-design-guidelines  
[40] Offline fallback page (web.dev): https://web.dev/articles/offline-fallback-page  
[41] Offline Cookbook (web.dev): https://web.dev/articles/offline-cookbook  
[42] Service Worker API (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API  
[43] RFC 9110: HTTP Semantics: https://www.rfc-editor.org/rfc/rfc9110  
[44] RFC 9111: HTTP Caching: https://datatracker.ietf.org/doc/html/rfc9111  
[45] RFC 9113: HTTP/2: https://www.rfc-editor.org/rfc/rfc9113  
[46] RFC 9114: HTTP/3: https://www.rfc-editor.org/rfc/rfc9114  
[47] Can I use HTTP/3: https://caniuse.com/http3  
[48] —  
[49] GraphQL Specification: https://spec.graphql.org/  
[50] GraphQL over HTTP (draft): https://graphql.github.io/graphql-over-http/draft/  
[51] GraphQL: Serving over HTTP: https://graphql.org/learn/serving-over-http/  
[52] Apollo Client subscriptions: https://www.apollographql.com/docs/react/data/subscriptions/  
[53] RFC 6455: WebSocket: https://www.rfc-editor.org/rfc/rfc6455  
[54] RFC 8441: WebSockets over HTTP/2: https://www.rfc-editor.org/rfc/rfc8441  
[55] RFC 9220: WebSockets over HTTP/3: https://www.rfc-editor.org/rfc/rfc9220  
[56] RFC 5789: PATCH: https://www.rfc-editor.org/rfc/rfc5789  
[57] RFC 6902: JSON Patch: https://www.rfc-editor.org/rfc/rfc6902  
[58] RFC 7396: JSON Merge Patch: https://www.rfc-editor.org/rfc/rfc7396  
[59] If-Match (MDN): https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/If-Match  
[60] If-None-Match (MDN): https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/If-None-Match  
[61] RFC 9110: ETag: https://www.rfc-editor.org/rfc/rfc9110.html#field.etag  
[62] RFC 9110: Last-Modified: https://www.rfc-editor.org/rfc/rfc9110.html#field.last-modified  
[63] Lamport clocks: https://dl.acm.org/doi/10.1145/359545.359563  
[64] Fidge vector clocks: https://core.ac.uk/outputs/24281589/  
[65] CRDTs (INRIA report): https://inria.hal.science/inria-00609399v1  
[66] CRDT survey (ACM entry): https://dl.acm.org/doi/10.5555/2050613.2050642  
[67] OT (Sun & Ellis): https://core.ac.uk/outputs/24345065/  
[68] Automerge docs: https://automerge.org/docs/hello/  
[69] Yjs docs: https://docs.yjs.dev/  
[70] Learn PWA: Update (web.dev): https://web.dev/learn/pwa/update  
[71] Apple EU DMA and PWAs: https://developer.apple.com/jp/support/dma-and-apps-in-the-eu/  
[72] Background Fetch (Chrome only): https://developer.chrome.com/blog/background-fetch  
[73] OWASP ASVS: https://owasp.org/www-project-application-security-verification-standard/  
[74] OWASP Proactive Controls (C3): https://top10proactive.owasp.org/the-top-10/c3-validate-input-and-handle-exceptions/  
[75] RFC 7636: PKCE: https://www.rfc-editor.org/rfc/rfc7636.html  
[76] OAuth 2.0 Security BCP (BCP 240): https://www.rfc-editor.org/info/bcp240  
[77] OAuth for Browser-Based Apps (draft): https://datatracker.ietf.org/doc/html/draft-ietf-oauth-browser-based-apps  
[78] OpenID Connect Core: https://openid.net/specs/openid-connect-core-1_0-18.html  
[79] RFC 6265: Cookies: https://www.rfc-editor.org/info/rfc6265  
[80] RFC 6750: Bearer Tokens: https://www.rfc-editor.org/rfc/rfc6750  
[81] OWASP CSRF Prevention: https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html  
[82] CSP Level 3 (W3C WD): https://www.w3.org/TR/CSP3/  
[83] Trusted Types (MDN): https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Security-Policy/trusted-types  
[84] WebAuthn Level 2 (W3C news): https://www.w3.org/news/2021/web-authentication-an-api-for-accessing-public-key-credentials-level-2-is-a-w3c-recommendation/  
[85] WebAuthn Level 3 (WD): https://www.w3.org/TR/webauthn-3/  
[86] RFC 8018: PBKDF2: https://www.rfc-editor.org/rfc/rfc8018  
[87] RFC 5869: HKDF: https://www.rfc-editor.org/rfc/rfc5869  
[88] SubtleCrypto.generateKey (MDN): https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/generateKey  
[89] RFC 8446: TLS 1.3: https://www.rfc-editor.org/info/rfc8446  
[90] Chrome deprecates TLS 1.0/1.1: https://security.googleblog.com/2019/10/chrome-ui-for-deprecating-legacy-tls.html  
[91] RFC 6797: HSTS: https://www.rfc-editor.org/rfc/rfc6797  
[92] COOP (MDN): https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cross-Origin-Opener-Policy  
[93] npm package-lock.json: https://docs.npmjs.com/cli/v8/configuring-npm/package-lock-json  
[94] Sigstore docs: https://docs.sigstore.dev/  
[95] SLSA v1 final: https://slsa.dev/blog/2023/04/slsa-v1-final  
[96] OWASP API Security Top 10 (2023): https://owasp.org/API-Security/editions/2023/en/0x11-t10//  
[97] fetchPriority (MDN): https://developer.mozilla.org/en-US/docs/Web/API/HTMLImageElement/fetchPriority  
[98] WHATWG HTML (semantics): https://html.spec.whatwg.org/dev/semantics.html  
[99] RFC 7932: Brotli: https://www.rfc-editor.org/rfc/rfc7932  
[100] Compression Streams API (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Compression_Streams_API  
[101] WHATWG Streams Standard: https://streams.spec.whatwg.org/  
[102] WebSocket API (MDN): https://developer.mozilla.org/en-US/docs/Web/API/WebSocket  
[103] Broadcast Channel API (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Broadcast_Channel_API  
[104] SharedWorker (MDN): https://developer.mozilla.org/en-US/docs/Web/API/SharedWorker  
[105] GraphQL over HTTP (blog): https://graphql.org/blog/2022-11-07-graphql-http/  
[106] GraphQL HTTP reference impl: https://github.com/graphql/graphql-http  
[107] subscriptions-transport-ws (archived): https://github.com/apollographql/subscriptions-transport-ws  
[108] Transferable objects (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects  
[109] Idempotency-Key header (I-D): https://datatracker.ietf.org/doc/html/draft-ietf-httpapi-idempotency-key-header  
[110] React.lazy (React docs): https://react.dev/reference/react/lazy  
[111] React.Suspense (React docs): https://react.dev/reference/react/Suspense  
[112] react-window (GitHub): https://github.com/bvaughn/react-window  
[113] Responsive images (MDN): https://developer.mozilla.org/en-US/docs/Learn/HTML/Multimedia_and_embedding/Responsive_images  
[114] Media formats for the web (MDN): https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Containers  
[115] Web Vitals (web.dev): https://web.dev/vitals/  
[116] web-vitals library (GitHub): https://github.com/GoogleChrome/web-vitals  
[117] sendBeacon (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Navigator/sendBeacon  
[118] OWASP Software Supply Chain Security: https://cheatsheetseries.owasp.org/cheatsheets/Software_Supply_Chain_Security_Cheat_Sheet.html  
[119] Redux Toolkit Query: https://redux-toolkit.js.org/rtk-query/overview  
[120] TanStack Query: https://tanstack.com/query/latest  
[121] Apollo cache configuration: https://www.apollographql.com/docs/react/caching/cache-configuration/  
[122] tus resumable upload protocol: https://tus.io/protocols/resumable-upload.html  
[123] Amazon S3 multipart upload: https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html  
[124] Playwright offline mode: https://playwright.dev/docs/network#offline  
[125] OWASP ZAP: https://www.zaproxy.org/  
[126] Reporting API (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Reporting_API