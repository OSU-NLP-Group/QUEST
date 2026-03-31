/**
 * Popup script for Cache Manager Capture extension.
 *
 * Handles single capture, batch start/stop, and live batch status display.
 */

const BACKEND = 'http://127.0.0.1:8000';

const statusEl = document.getElementById('status');
const batchSection = document.getElementById('batch-section');
const batchRunningSection = document.getElementById('batch-running-section');
const singleSection = document.getElementById('single-section');
const targetSection = document.getElementById('target-section');
const captureBtn = document.getElementById('capture-btn');
const batchStartBtn = document.getElementById('batch-start-btn');
const batchStartPauseBtn = document.getElementById('batch-start-pause-btn');
const batchStopBtn = document.getElementById('batch-stop-btn');
const batchStopBtn2 = document.getElementById('batch-stop-btn-2');
const batchInfoEl = document.getElementById('batch-info');

let currentTarget = null;
let pollTimer = null;

const STATUS_LABELS = {
    loading: 'Loading',
    capturing: 'Capturing',
    retrying: 'Retrying',
    captcha: 'CAPTCHA',
    timeout: 'Timeout',
    advancing: 'Next...',
    done: 'Done',
};

function showSection(name) {
    batchSection.classList.remove('active');
    batchRunningSection.classList.remove('active');
    singleSection.classList.remove('active');
    document.getElementById(`${name}-section`).classList.add('active');
}

function updateBatchRunningUI(state) {
    const completed = state.completed || 0;
    const total = state.total || 0;
    const skipped = state.skipped || 0;
    const status = state.status || 'loading';
    const currentUrl = state.currentUrl || '';
    const log = state.log || [];

    // Progress count
    document.getElementById('br-completed').textContent = completed;
    document.getElementById('br-total').textContent = total;
    const skippedEl = document.getElementById('br-skipped');
    skippedEl.textContent = skipped > 0 ? `(${skipped} skipped)` : '';

    // Status badge
    const badgeEl = document.getElementById('br-status');
    badgeEl.textContent = STATUS_LABELS[status] || status;
    badgeEl.className = 'batch-status-badge ' + status;

    // Mode indicator
    const modeEl = document.getElementById('br-mode');
    if (modeEl) {
        modeEl.textContent = state.pauseOnCaptcha ? 'Mode: pause on CAPTCHA' : 'Mode: auto (skip CAPTCHA)';
    }

    // Progress bar
    const pct = total > 0 ? Math.round(((completed + skipped) / total) * 100) : 0;
    document.getElementById('br-progress-fill').style.width = pct + '%';

    // Current URL
    document.getElementById('br-url-text').textContent = currentUrl || '—';

    // Log entries
    const logEl = document.getElementById('br-log');
    const html = log.map(e =>
        `<div class="log-entry ${e.type || ''}">` +
        `<span class="log-time">${escHtml(e.time)}</span>` +
        `<span class="log-msg">${escHtml(e.msg)}</span>` +
        `</div>`
    ).join('');
    logEl.innerHTML = html;
    // Auto-scroll to bottom
    logEl.scrollTop = logEl.scrollHeight;
}

function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
        try {
            const state = await chrome.runtime.sendMessage({ action: 'get-batch-status' });
            if (!state?.batchMode) {
                // Batch ended
                stopPolling();
                updateBatchRunningUI(state);
                return;
            }
            updateBatchRunningUI(state);
        } catch {
            // Extension might have been reloaded
            stopPolling();
        }
    }, 1000);
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

async function init() {
    try {
        // Check backend connection
        const res = await fetch(`${BACKEND}/api/status`);
        const data = await res.json();

        if (data.loaded) {
            statusEl.className = 'status connected';
            statusEl.textContent = `Connected — ${data.agent_name}`;
        } else {
            statusEl.className = 'status connected';
            statusEl.textContent = 'Connected (no cache loaded)';
        }

        // Check if batch mode is running in the background service worker
        const bgStatus = await chrome.runtime.sendMessage({ action: 'get-batch-status' });
        if (bgStatus?.batchMode) {
            showSection('batch-running');
            updateBatchRunningUI(bgStatus);
            startPolling();
            return;
        }

        // Check if batch is queued on the backend
        const batchRes = await fetch(`${BACKEND}/api/capture/batch/status`);
        const batch = await batchRes.json();

        if (batch.active) {
            batchInfoEl.innerHTML = `
                <div>Batch capture queued:</div>
                <div><span class="batch-count">${batch.remaining}</span> URLs to recapture</div>
                ${batch.completed > 0 ? `<div>${batch.completed}/${batch.total} completed</div>` : ''}
            `;
            showSection('batch');
            return;
        }

        // Normal single-capture mode
        showSection('single');

        // Load capture target
        const targetRes = await fetch(`${BACKEND}/api/capture/target`);
        const target = await targetRes.json();

        if (target.active) {
            currentTarget = target;
            targetSection.innerHTML = `
                <div class="target-info">
                    <div class="label">Capture target:</div>
                    <div class="value task-id">${escHtml(target.task_id)}</div>
                    <div class="value">${escHtml(target.url)}</div>
                </div>
            `;
            captureBtn.disabled = false;
        } else {
            targetSection.innerHTML = `
                <div class="no-target">
                    No capture target set.<br>
                    Select a URL in Cache Manager and click "Open in Browser" first.
                </div>
            `;
            captureBtn.disabled = true;
        }
    } catch (err) {
        statusEl.className = 'status disconnected';
        statusEl.textContent = 'Cannot connect to backend (is it running?)';
        captureBtn.disabled = true;
    }
}

// Single capture
captureBtn.addEventListener('click', async () => {
    if (!currentTarget) return;
    captureBtn.disabled = true;
    captureBtn.textContent = 'Capturing...';

    try {
        const result = await chrome.runtime.sendMessage({ action: 'capture' });
        if (result?.success) {
            captureBtn.textContent = 'Captured!';
            captureBtn.className = 'capture-btn success';
            setTimeout(() => window.close(), 500);
        } else {
            throw new Error(result?.error || 'Capture failed');
        }
    } catch (err) {
        captureBtn.textContent = 'Failed: ' + err.message;
        captureBtn.className = 'capture-btn error';
        setTimeout(() => {
            captureBtn.textContent = 'Capture This Page';
            captureBtn.className = 'capture-btn';
            captureBtn.disabled = false;
        }, 3000);
    }
});

// Start batch (shared logic)
async function onStartBatch(pauseOnCaptcha, btn) {
    btn.disabled = true;
    btn.textContent = 'Starting...';

    try {
        const result = await chrome.runtime.sendMessage({
            action: 'start-batch',
            pauseOnCaptcha,
        });
        if (result?.success) {
            showSection('batch-running');
            const state = await chrome.runtime.sendMessage({ action: 'get-batch-status' });
            updateBatchRunningUI(state);
            startPolling();
        } else {
            throw new Error(result?.error || 'Failed to start batch');
        }
    } catch (err) {
        btn.textContent = 'Failed: ' + err.message;
        btn.className += ' error';
        setTimeout(() => {
            btn.textContent = pauseOnCaptcha ? 'Start Batch (pause on CAPTCHA)' : 'Start Batch (auto)';
            btn.className = pauseOnCaptcha ? 'capture-btn batch-pause' : 'capture-btn batch';
            btn.disabled = false;
        }, 3000);
    }
}

batchStartBtn.addEventListener('click', () => onStartBatch(false, batchStartBtn));
batchStartPauseBtn.addEventListener('click', () => onStartBatch(true, batchStartPauseBtn));

// Stop batch (both buttons)
async function onStopBatch() {
    stopPolling();
    await chrome.runtime.sendMessage({ action: 'stop-batch' });
    window.close();
}
batchStopBtn.addEventListener('click', onStopBatch);
batchStopBtn2.addEventListener('click', onStopBatch);

function escHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
}

init();
