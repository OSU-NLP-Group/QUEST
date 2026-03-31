/**
 * Background service worker for Cache Manager Capture extension.
 *
 * Handles:
 * - Keyboard shortcut (Alt+Shift+C) to capture current page
 * - Batch auto-capture mode with CAPTCHA detection
 * - Communication with the local backend at localhost:8000
 */

const BACKEND = 'http://127.0.0.1:8000';

// ---------------------------------------------------------------------------
// Batch mode state
// ---------------------------------------------------------------------------

let batchMode = false;
let batchTabId = null;
let captchaCheckTimer = null;
let batchProcessing = false;  // guard against re-entrant onUpdated calls
let pageTimeoutTimer = null;  // 15s page load timeout
let currentRetryCount = 0;    // retry count for current batch URL
let pauseOnCaptcha = false;   // false = capture CAPTCHA pages and move on; true = wait for user
const PAGE_TIMEOUT_MS = 15000;
const MAX_RETRIES = 2;
const MIN_BODY_LENGTH = 200;  // pages shorter than this get retried

// Rich batch status (for popup display)
let batchState = {
    total: 0,
    completed: 0,
    skipped: 0,
    currentUrl: '',
    status: '',       // 'loading', 'retrying', 'captcha', 'capturing', 'advancing', 'done'
    log: [],          // [{time, msg, type}] — last 20 entries
};

function batchLog(msg, type = 'info') {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    batchState.log.push({ time, msg, type });
    if (batchState.log.length > 30) batchState.log.shift();
}

function setBatchStatus(status, currentUrl = null) {
    batchState.status = status;
    if (currentUrl !== null) batchState.currentUrl = currentUrl;
}

// ---------------------------------------------------------------------------
// CAPTCHA detection (injected into target pages)
// ---------------------------------------------------------------------------

/**
 * Injected into the target page via chrome.scripting.executeScript.
 * Returns a string describing the CAPTCHA type, or null if none detected.
 */
function detectCaptcha() {
    const title = document.title.toLowerCase();
    const text = (document.body?.innerText || '').substring(0, 3000).toLowerCase();

    // Cloudflare challenge page
    if (title.includes('just a moment') || title.includes('attention required'))
        return 'cloudflare';
    if (document.querySelector('#challenge-form, .cf-challenge-running, #cf-challenge-running'))
        return 'cloudflare';
    if (text.includes('checking your browser') || text.includes('verify you are human'))
        return 'cloudflare';

    // Cloudflare Turnstile widget
    if (document.querySelector('iframe[src*="challenges.cloudflare.com"]'))
        return 'turnstile';

    // reCAPTCHA
    if (document.querySelector('iframe[src*="recaptcha"], .g-recaptcha'))
        return 'recaptcha';

    // hCaptcha
    if (document.querySelector('iframe[src*="hcaptcha"], .h-captcha'))
        return 'hcaptcha';

    // Generic access denied / blocked (only if page is very short)
    if (text.includes('access denied') && text.length < 2000) return 'blocked';
    if (text.includes('403 forbidden') && text.length < 2000) return 'blocked';

    return null;
}

/**
 * Run CAPTCHA detection in a tab.
 * @returns {Promise<string|null>}
 */
async function detectCaptchaInTab(tabId) {
    try {
        const results = await chrome.scripting.executeScript({
            target: { tabId },
            func: detectCaptcha,
        });
        return results?.[0]?.result || null;
    } catch (e) {
        console.warn('detectCaptchaInTab failed:', e);
        return null;
    }
}

// ---------------------------------------------------------------------------
// Keyboard shortcut handler
// ---------------------------------------------------------------------------

chrome.commands.onCommand.addListener(async (command) => {
    if (command === 'capture-page') {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab) {
            await capturePage(tab);
        }
    }
});

// ---------------------------------------------------------------------------
// Core capture function
// ---------------------------------------------------------------------------

/**
 * Capture the current page and send to backend.
 * @param {chrome.tabs.Tab} tab
 * @param {object} [opts] - Optional overrides for batch mode
 * @param {boolean} [opts.skipTargetFetch] - Skip fetching capture target (use opts.task_id/url)
 * @param {string} [opts.task_id]
 * @param {string} [opts.url]
 */
async function capturePage(tab, opts = {}) {
    try {
        let task_id, url;

        if (opts.skipTargetFetch) {
            task_id = opts.task_id;
            url = opts.url;
        } else {
            // Fetch capture target from backend
            const targetRes = await fetch(`${BACKEND}/api/capture/target`);
            const target = await targetRes.json();

            if (!target.active) {
                setBadge('!', '#dc2626', 3000);
                return { success: false, error: 'No capture target set. Select a URL in Cache Manager first.' };
            }
            task_id = target.task_id;
            url = target.url;
        }

        // Check if page is a PDF — handle differently
        const isPdf = await detectPdfInTab(tab.id) || (tab.url && tab.url.toLowerCase().endsWith('.pdf'));
        if (isPdf) {
            const actual_url = tab.url && tab.url !== url ? tab.url : undefined;
            const ok = await capturePdfAndUpload(task_id, url, actual_url);
            if (!ok) {
                setBadge('✗', '#dc2626', 3000);
                return { success: false, error: 'Failed to download PDF' };
            }
            if (batchMode) {
                setBadge('✓', '#22c55e', 1000);
                return { success: true, batch: true };
            }
            setBadge('✓', '#22c55e', 2000);
            await switchToCacheManager(tab.id);
            return { success: true };
        }

        // Extract text from the page
        const textResults = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: () => document.body?.innerText || '',
        });
        const text = textResults?.[0]?.result || '';

        // Ensure the target tab is active/visible before screenshot
        await chrome.tabs.update(tab.id, { active: true });
        await sleep(150);

        // Capture visible tab as screenshot
        const screenshotDataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
            format: 'jpeg',
            quality: 85,
        });
        const base64 = screenshotDataUrl.replace(/^data:image\/jpeg;base64,/, '');

        // Detect redirect: tab.url may differ from the original URL
        const actual_url = tab.url && tab.url !== url ? tab.url : undefined;

        // Send to backend
        const captureRes = await fetch(`${BACKEND}/api/capture`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_id,
                url,
                text,
                screenshot_base64: base64,
                ...(actual_url ? { actual_url } : {}),
            }),
        });

        if (!captureRes.ok) {
            throw new Error(`Backend returned ${captureRes.status}`);
        }

        // In batch mode, don't close tab or switch — batch logic handles advancement
        if (batchMode) {
            setBadge('✓', '#22c55e', 1000);
            return { success: true, batch: true };
        }

        // Normal mode — close tab and switch back
        setBadge('✓', '#22c55e', 2000);
        await switchToCacheManager(tab.id);
        return { success: true };
    } catch (err) {
        console.error('Capture failed:', err);
        setBadge('✗', '#dc2626', 3000);
        return { success: false, error: err.message };
    }
}

// ---------------------------------------------------------------------------
// Batch mode orchestration
// ---------------------------------------------------------------------------

async function startBatch(opts = {}) {
    try {
        const res = await fetch(`${BACKEND}/api/capture/batch/status`);
        const status = await res.json();

        if (!status.active || !status.current) {
            setBadge('!', '#dc2626', 3000);
            return { success: false, error: 'No active batch. Queue URLs from Cache Manager first.' };
        }

        batchMode = true;
        batchProcessing = false;
        currentRetryCount = 0;
        pauseOnCaptcha = !!opts.pauseOnCaptcha;
        batchState = { total: status.total, completed: 0, skipped: 0, currentUrl: status.current.url, status: 'loading', log: [] };
        batchLog(`Batch started: ${status.total} URLs (${pauseOnCaptcha ? 'pause on CAPTCHA' : 'auto'})`);
        batchLog(`Loading: ${truncUrl(status.current.url)}`);
        setBadge(`0/${status.total}`, '#2563eb');

        // Open the first URL in a new tab
        const tab = await chrome.tabs.create({ url: status.current.url });
        batchTabId = tab.id;
        startPageTimeout(tab.id);

        return { success: true, total: status.total };
    } catch (err) {
        console.error('startBatch failed:', err);
        batchMode = false;
        return { success: false, error: err.message };
    }
}

async function advanceBatch() {
    try {
        setBatchStatus('advancing');
        const res = await fetch(`${BACKEND}/api/capture/batch/status`);
        const status = await res.json();

        if (!status.active || !status.current) {
            // Batch complete
            endBatch();
            return;
        }

        batchState.completed = status.completed;
        batchState.total = status.total;
        setBadge(`${status.completed}/${status.total}`, '#2563eb');

        // Navigate existing tab to the next URL
        if (batchTabId) {
            batchProcessing = false;
            currentRetryCount = 0;  // reset retry count for new URL
            setBatchStatus('loading', status.current.url);
            batchLog(`Loading: ${truncUrl(status.current.url)}`);
            startPageTimeout(batchTabId);
            await chrome.tabs.update(batchTabId, { url: status.current.url });
        }
    } catch (err) {
        console.error('advanceBatch failed:', err);
        batchLog(`Error advancing: ${err.message}`, 'error');
        endBatch();
    }
}

async function endBatch() {
    batchMode = false;
    stopCaptchaPolling();
    clearPageTimeout();

    setBatchStatus('done');
    batchLog(`Batch finished: ${batchState.completed} captured, ${batchState.skipped} skipped`);
    setBadge('Done', '#22c55e', 3000);

    if (batchTabId) {
        await switchToCacheManager(batchTabId);
        batchTabId = null;
    }
}

async function stopBatch() {
    try {
        await fetch(`${BACKEND}/api/capture/batch/stop`, { method: 'POST' });
    } catch {}
    await endBatch();
}

// ---------------------------------------------------------------------------
// Tab monitoring — auto-capture on page load during batch mode
// ---------------------------------------------------------------------------

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
    // Only monitor our batch tab
    if (!batchMode || tabId !== batchTabId) return;
    if (changeInfo.status !== 'complete') return;

    // Guard against re-entrant calls
    if (batchProcessing) return;
    batchProcessing = true;

    // Page loaded — cancel timeout timer
    clearPageTimeout();

    // Wait for page to settle (JS rendering, redirects)
    await sleep(2500);

    // Verify we're still in batch mode and this is still our tab
    if (!batchMode || tabId !== batchTabId) {
        batchProcessing = false;
        return;
    }

    // Check for CAPTCHA
    const captchaType = await detectCaptchaInTab(tabId);

    if (captchaType && pauseOnCaptcha) {
        // CAPTCHA detected + pause mode — wait for user to solve
        console.log(`CAPTCHA detected: ${captchaType} (pausing)`);
        currentRetryCount = 0;
        setBatchStatus('captcha');
        batchLog(`CAPTCHA (${captchaType}) — solve in browser`, 'warn');
        setBadge('⏳', '#f59e0b');

        try {
            await fetch(`${BACKEND}/api/capture/batch/captcha`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: captchaType }),
            });
        } catch {}

        startCaptchaPolling(tabId);
    } else {
        // No CAPTCHA, or CAPTCHA but auto mode — capture and move on
        if (captchaType) {
            batchLog(`CAPTCHA (${captchaType}) — capturing anyway`, 'warn');
        }
        await autoCaptureAndAdvance(tabId, currentRetryCount);
    }
});

// Handle tab closure during batch
chrome.tabs.onRemoved.addListener((tabId) => {
    if (batchMode && tabId === batchTabId) {
        console.log('Batch tab was closed');
        batchTabId = null;
        endBatch();
    }
});

// ---------------------------------------------------------------------------
// CAPTCHA polling
// ---------------------------------------------------------------------------

function startCaptchaPolling(tabId) {
    stopCaptchaPolling();

    const poll = async () => {
        if (!batchMode || tabId !== batchTabId) return;

        try {
            const result = await detectCaptchaInTab(tabId);
            if (!result) {
                // CAPTCHA resolved!
                stopCaptchaPolling();
                batchLog('CAPTCHA resolved, capturing...', 'success');
                setBatchStatus('capturing');
                await sleep(1500); // Let page settle after CAPTCHA
                await autoCaptureAndAdvance(tabId);
                return;
            }
        } catch (e) {
            // Tab might be gone
            stopCaptchaPolling();
            return;
        }

        // Continue polling
        captchaCheckTimer = setTimeout(poll, 3000);
    };

    captchaCheckTimer = setTimeout(poll, 3000);
}

function stopCaptchaPolling() {
    if (captchaCheckTimer) {
        clearTimeout(captchaCheckTimer);
        captchaCheckTimer = null;
    }
}

// ---------------------------------------------------------------------------
// PDF detection and download
// ---------------------------------------------------------------------------

/**
 * Detect if the tab is showing Chrome's built-in PDF viewer.
 * Returns true if the page is a PDF.
 */
async function detectPdfInTab(tabId) {
    try {
        const results = await chrome.scripting.executeScript({
            target: { tabId },
            func: () => {
                // Chrome's PDF viewer uses an <embed type="application/pdf">
                const embed = document.querySelector('embed[type="application/pdf"]');
                if (embed) return true;
                // Also check if the content type meta tag says PDF
                const ct = document.contentType || '';
                if (ct === 'application/pdf') return true;
                return false;
            },
        });
        return results?.[0]?.result || false;
    } catch {
        return false;
    }
}

/**
 * Download PDF bytes from a URL and upload to the backend.
 * Returns true on success.
 */
async function capturePdfAndUpload(taskId, url, actualUrl) {
    const downloadUrl = actualUrl || url;
    try {
        const res = await fetch(downloadUrl);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();

        const form = new FormData();
        form.append('file', blob, 'page.pdf');

        const uploadRes = await fetch(
            `${BACKEND}/api/upload-pdf/${encodeURIComponent(taskId)}?url=${encodeURIComponent(url)}`,
            { method: 'POST', body: form }
        );
        if (!uploadRes.ok) throw new Error(`Upload failed: ${uploadRes.status}`);
        return true;
    } catch (err) {
        console.warn('capturePdfAndUpload failed:', err);
        return false;
    }
}

// ---------------------------------------------------------------------------
// Auto-capture helper
// ---------------------------------------------------------------------------

async function autoCaptureAndAdvance(tabId, retryCount = 0) {
    try {
        // Get current batch target
        const statusRes = await fetch(`${BACKEND}/api/capture/batch/status`);
        const status = await statusRes.json();

        if (!status.active || !status.current) {
            endBatch();
            return;
        }

        const tab = await chrome.tabs.get(tabId);

        // Check if page is actually a PDF (e.g., URL was misrecorded as "web")
        const isPdf = await detectPdfInTab(tabId);
        if (isPdf || (tab.url && tab.url.toLowerCase().endsWith('.pdf'))) {
            setBatchStatus('capturing');
            batchLog(`PDF detected: ${truncUrl(status.current.url)}`, 'info');
            const actual_url = tab.url && tab.url !== status.current.url ? tab.url : undefined;
            const ok = await capturePdfAndUpload(status.current.task_id, status.current.url, actual_url);
            if (ok) {
                batchLog(`PDF saved OK`, 'success');
                setBadge('✓', '#22c55e', 1000);
            } else {
                batchLog(`PDF download failed, skipping`, 'error');
                await skipAndAdvance();
                return;
            }
            await sleep(500);
            await advanceBatch();
            return;
        }

        // Check if page body is too short (may need retry)
        if (retryCount < MAX_RETRIES) {
            try {
                const textResults = await chrome.scripting.executeScript({
                    target: { tabId },
                    func: () => (document.body?.innerText || '').length,
                });
                const bodyLength = textResults?.[0]?.result || 0;
                if (bodyLength < MIN_BODY_LENGTH) {
                    console.log(`Page body too short (${bodyLength} chars), retry ${retryCount + 1}/${MAX_RETRIES}`);
                    setBatchStatus('retrying');
                    batchLog(`Short page (${bodyLength} chars), retry ${retryCount + 1}/${MAX_RETRIES}`, 'warn');
                    setBadge(`R${retryCount + 1}`, '#f59e0b');
                    // Reload the page and let onUpdated handle it again
                    batchProcessing = false;
                    currentRetryCount = retryCount + 1;
                    startPageTimeout(tabId);
                    await chrome.tabs.reload(tabId);
                    return;
                }
            } catch (e) {
                // If we can't check body length, proceed with capture
            }
        }

        setBatchStatus('capturing');
        batchLog(`Capturing: ${truncUrl(status.current.url)}`);
        const result = await capturePage(tab, {
            skipTargetFetch: true,
            task_id: status.current.task_id,
            url: status.current.url,
        });

        if (!result.success) {
            // Capture failed — skip this URL and advance
            console.warn(`Capture failed for ${status.current.url}: ${result.error}, skipping`);
            batchLog(`Failed: ${result.error || 'unknown'}, skipping`, 'error');
            await skipAndAdvance();
            return;
        }

        batchLog(`Captured OK`, 'success');
        // Wait briefly then advance
        await sleep(500);
        await advanceBatch();
    } catch (err) {
        // Unexpected error — skip current URL to avoid getting stuck
        console.error('autoCaptureAndAdvance failed:', err);
        batchLog(`Error: ${err.message}, skipping`, 'error');
        await skipAndAdvance();
    }
}

/**
 * Skip the current batch URL (on failure) and advance to next.
 */
async function skipAndAdvance() {
    try {
        batchState.skipped++;
        await fetch(`${BACKEND}/api/capture/batch/skip`, { method: 'POST' });
        await sleep(300);
        await advanceBatch();
    } catch (err) {
        console.error('skipAndAdvance failed:', err);
        batchLog(`Fatal error: ${err.message}`, 'error');
        endBatch();
    }
}

// ---------------------------------------------------------------------------
// Page load timeout (15s)
// ---------------------------------------------------------------------------

function startPageTimeout(tabId) {
    clearPageTimeout();
    pageTimeoutTimer = setTimeout(async () => {
        if (!batchMode || tabId !== batchTabId) return;
        if (batchProcessing) return;  // already being handled
        batchProcessing = true;

        console.log(`Page timeout (${PAGE_TIMEOUT_MS}ms) — force capturing`);
        setBatchStatus('timeout');
        batchLog(`Timeout (${PAGE_TIMEOUT_MS/1000}s) — force capturing`, 'warn');
        setBadge('T/O', '#f59e0b', 1500);
        // Force capture whatever is visible
        await autoCaptureAndAdvance(tabId, MAX_RETRIES);  // skip retries on timeout
    }, PAGE_TIMEOUT_MS);
}

function clearPageTimeout() {
    if (pageTimeoutTimer) {
        clearTimeout(pageTimeoutTimer);
        pageTimeoutTimer = null;
    }
}

// ---------------------------------------------------------------------------
// Tab management
// ---------------------------------------------------------------------------

async function switchToCacheManager(capturedTabId) {
    try {
        const capturedTab = await chrome.tabs.get(capturedTabId);
        const isCM = capturedTab.url?.startsWith(BACKEND);
        if (!isCM) {
            await chrome.tabs.remove(capturedTabId);
        }
        const cmTabs = await chrome.tabs.query({
            url: ['http://127.0.0.1:8000/*', 'http://localhost:8000/*'],
        });
        if (cmTabs.length > 0) {
            await chrome.tabs.update(cmTabs[0].id, { active: true });
            await chrome.windows.update(cmTabs[0].windowId, { focused: true });
        }
    } catch (e) {
        console.warn('switchToCacheManager:', e);
    }
}

// ---------------------------------------------------------------------------
// Badge helper
// ---------------------------------------------------------------------------

function setBadge(text, color, clearAfterMs = 0) {
    chrome.action.setBadgeText({ text });
    if (color) chrome.action.setBadgeBackgroundColor({ color });
    if (clearAfterMs > 0) {
        setTimeout(() => chrome.action.setBadgeText({ text: '' }), clearAfterMs);
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

function truncUrl(url, maxLen = 60) {
    if (!url) return '';
    try {
        const u = new URL(url);
        const short = u.hostname + u.pathname;
        return short.length > maxLen ? short.substring(0, maxLen) + '...' : short;
    } catch {
        return url.length > maxLen ? url.substring(0, maxLen) + '...' : url;
    }
}

// ---------------------------------------------------------------------------
// Message handler (from popup)
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.action === 'capture') {
        chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
            if (tab) capturePage(tab).then(sendResponse);
        });
        return true;
    }
    if (msg.action === 'start-batch') {
        startBatch({ pauseOnCaptcha: !!msg.pauseOnCaptcha }).then(sendResponse);
        return true;
    }
    if (msg.action === 'stop-batch') {
        stopBatch().then(sendResponse);
        return true;
    }
    if (msg.action === 'get-batch-status') {
        sendResponse({ batchMode, batchTabId, pauseOnCaptcha, ...batchState });
        return false;
    }
});
