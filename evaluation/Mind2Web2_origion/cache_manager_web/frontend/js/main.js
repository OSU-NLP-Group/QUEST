/**
 * Main entry point for Cache Manager Web UI.
 */
import { getState, setState, subscribe } from './store.js';
import * as api from './api.js';
import { selectTask, selectUrl, reloadCurrentTask, updateReviewProgress, incrementTaskIssueFixedCount, showStatus, toast, filterUrls, $ } from './actions.js';
import { initTaskPanel } from './components/task-panel.js';
import { initUrlList } from './components/url-list.js';
import { initPreview } from './components/preview.js';

// ============================================================
// Initialization
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    initTaskPanel();
    initUrlList();
    initPreview();
    initToolbar();
    initKeyboardShortcuts();
    initSSE();
    initDragDrop();

    // Check if cache was auto-loaded
    api.getStatus().then(async status => {
        if (status.loaded) {
            const result = await api.loadCache(status.agent_path);
            refreshAfterLoad(result);
        }
    }).catch(() => {});
});

// ============================================================
// Toolbar
// ============================================================

function initToolbar() {
    $('#btn-open').addEventListener('click', onOpenFolder);
    $('#btn-refresh').addEventListener('click', onRefresh);
    $('#btn-prev-issue').addEventListener('click', () => navigateIssue(-1));
    $('#btn-next-issue').addEventListener('click', () => navigateIssue(1));
    $('#btn-mark-reviewed').addEventListener('click', onMarkReviewed);
    $('#btn-open-browser').addEventListener('click', onOpenInBrowser);
    $('#btn-flag-issue').addEventListener('click', onFlagAsIssue);
    $('#btn-reset-url').addEventListener('click', onResetUrl);
    $('#btn-edit-url').addEventListener('click', onEditUrl);
    $('#btn-add-url').addEventListener('click', onAddUrl);
    $('#btn-delete-url').addEventListener('click', onDeleteUrl);
    $('#btn-upload-pdf').addEventListener('click', () => $('#pdf-picker').click());
    $('#btn-upload-mhtml').addEventListener('click', onUploadMhtml);
    $('#btn-recapture').addEventListener('click', onRecapture);
    $('#btn-batch').addEventListener('click', onBatchRecapture);

    // MHTML file picker callback
    $('#mhtml-picker').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const s = getState();
        if (!s.selectedTaskId || !s.selectedUrl) return;
        try {
            showStatus('Uploading MHTML...', 'warning');
            await api.uploadMhtml(s.selectedTaskId, s.selectedUrl, file);
            toast('MHTML uploaded successfully', 'success');
            await reloadCurrentTask();
        } catch (err) {
            toast('MHTML upload failed: ' + err.message, 'error');
        }
        e.target.value = '';
    });

    // PDF file picker callback
    $('#pdf-picker').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const s = getState();
        if (!s.selectedTaskId || !s.selectedUrl) return;
        try {
            showStatus('Uploading PDF...', 'warning');
            await api.uploadPdf(s.selectedTaskId, s.selectedUrl, file);
            toast('PDF uploaded — content type switched to PDF', 'success');
            setState({ contentVersion: s.contentVersion + 1 });
            await reloadCurrentTask();
            await updateReviewProgress();
        } catch (err) {
            toast('PDF upload failed: ' + err.message, 'error');
        }
        e.target.value = '';
    });

    subscribe((s) => {
        $('#btn-refresh').disabled = !s.loaded;
        const hasIssues = s.issueIndex.length > 0;
        $('#btn-prev-issue').disabled = !hasIssues;
        $('#btn-next-issue').disabled = !hasIssues;
        if (hasIssues && s.issueCursor >= 0) {
            $('#issue-counter').textContent = `Issue ${s.issueCursor + 1}/${s.issueIndex.length}`;
        } else {
            $('#issue-counter').textContent = hasIssues ? `${s.issueIndex.length} issues` : '';
        }
        $('#agent-info').textContent = s.loaded
            ? `${s.agentName} | ${s.stats.total_tasks || 0} tasks | ${s.stats.total_urls || 0} URLs`
            : 'No cache loaded';
        const hasUrl = !!(s.selectedTaskId && s.selectedUrl);
        const hasTask = !!s.selectedTaskId;
        $('#btn-mark-reviewed').disabled = !hasUrl;
        $('#btn-open-browser').disabled = !hasUrl;
        $('#btn-flag-issue').disabled = !hasUrl;
        $('#btn-reset-url').disabled = !hasUrl;
        $('#btn-edit-url').disabled = !hasUrl;
        $('#btn-add-url').disabled = !hasTask;
        $('#btn-delete-url').disabled = !hasUrl;
        $('#btn-upload-pdf').disabled = !hasUrl;
        $('#btn-upload-mhtml').disabled = !hasUrl;
        $('#btn-recapture').disabled = !hasUrl;
        // Batch button: enabled when there are definite-severity issues
        const hasDefiniteIssues = s.issueIndex.some(i => i.severity === 'definite');
        $('#btn-batch').disabled = !hasDefiniteIssues || s.batchActive;
        // Batch status display
        const batchEl = $('#batch-status');
        if (s.batchActive) {
            batchEl.textContent = `Batch: ${s.batchCompleted}/${s.batchTotal}`;
        } else {
            batchEl.textContent = '';
        }
    }, ['loaded', 'issueIndex', 'issueCursor', 'agentName', 'stats',
        'selectedTaskId', 'selectedUrl', 'batchActive', 'batchCompleted', 'batchTotal']);
}

// ============================================================
// Actions
// ============================================================

async function onOpenFolder() {
    const path = prompt('Enter the path to an agent cache folder:\n\nExample: /Users/you/data/JudyAgent');
    if (!path) return;
    try {
        showStatus('Loading cache...', 'warning');
        const result = await api.loadCache(path.trim());
        refreshAfterLoad(result);
        toast(`Loaded ${result.loaded_tasks}/${result.total_tasks} tasks`, 'success');
    } catch (err) {
        toast('Failed to load: ' + err.message, 'error');
        showStatus('Load failed', 'error');
    }
}

async function onRefresh() {
    const s = getState();
    if (!s.agentPath) return;
    try {
        showStatus('Refreshing & scanning...', 'warning');
        setState({ contentVersion: s.contentVersion + 1 });
        const result = await api.loadCache(s.agentPath);
        refreshAfterLoad(result);
        toast('Refreshed', 'success');
    } catch (err) {
        toast('Refresh failed: ' + err.message, 'error');
        showStatus('Refresh failed', 'error');
    }
}

async function refreshAfterLoad(result) {
    setState({
        loaded: true,
        agentName: result.agent_name || '',
        agentPath: result.agent_path || '',
        stats: result.stats || {},
        taskIssues: result.task_issues || {},
        issueIndex: result.issue_index || [],
        issueCursor: -1,
    });
    // Fetch task list
    try {
        const data = await api.getTasks();
        setState({ tasks: data.tasks || [] });
    } catch (err) {
        console.error('Failed to load tasks:', err);
    }
    await updateReviewProgress();
    showStatus('Ready');
}

async function onMarkReviewed() {
    const s = getState();
    if (!s.selectedTaskId || !s.selectedUrl) return;
    try {
        const urlData = s.urls.find(u => u.url === s.selectedUrl);
        const wasReviewed = urlData?.reviewed;
        await api.setReview(s.selectedTaskId, s.selectedUrl, 'ok');
        // Update local state
        const urls = s.urls.map(u => u.url === s.selectedUrl ? { ...u, reviewed: 'ok' } : u);
        setState({ urls });
        // Only update issue progress if this URL has issues
        if (!['ok', 'fixed', 'skip'].includes(wasReviewed) && urlData?.issues?.length > 0) {
            incrementTaskIssueFixedCount(s.selectedTaskId);
            await updateReviewProgress();
        }
        toast('Marked as reviewed');
    } catch (err) {
        toast('Failed: ' + err.message, 'error');
    }
}

async function onFlagAsIssue() {
    const s = getState();
    if (!s.selectedTaskId || !s.selectedUrl) return;
    try {
        await api.flagUrl(s.selectedTaskId, s.selectedUrl);
        const urlData = s.urls.find(u => u.url === s.selectedUrl);
        const isPdf = urlData?.content_type === 'pdf';
        // Update local state: mark URL as having issues, clear review
        const urls = s.urls.map(u => u.url === s.selectedUrl
            ? { ...u, issues: ['flagged'], severity: 'definite', reviewed: '' }
            : u);
        const updates = { urls };
        if (!isPdf) {
            updates.currentText = 'access denied';
            updates.currentIssues = { has_issues: true, severity: 'definite', keywords: ['flagged'], patterns: [] };
        }
        setState(updates);
        toast('Flagged as issue (red)');
    } catch (err) {
        toast('Flag failed: ' + err.message, 'error');
    }
}

async function onOpenInBrowser() {
    const s = getState();
    if (!s.selectedUrl) return;
    await api.setCaptureTarget(s.selectedTaskId, s.selectedUrl).catch(() => {});
    window.open(s.selectedUrl, '_blank');
    toast('Opened in browser. Use the extension to capture.');
}

async function onDeleteUrl() {
    const s = getState();
    if (!s.selectedTaskId || !s.selectedUrl) return;
    try {
        await api.deleteUrl(s.selectedTaskId, s.selectedUrl);
        setState({ selectedUrl: null, currentText: null, currentIssues: null });
        await reloadCurrentTask();
        toast('URL deleted');
    } catch (err) {
        toast('Delete failed: ' + err.message, 'error');
    }
}

async function onResetUrl() {
    const s = getState();
    if (!s.selectedTaskId || !s.selectedUrl) return;
    try {
        showStatus('Resetting URL...', 'warning');
        await api.resetUrl(s.selectedTaskId, s.selectedUrl);
        toast('URL cache reset and flagged for recapture', 'success');
        setState({ contentVersion: s.contentVersion + 1 });
        await reloadCurrentTask();
        await updateReviewProgress();
    } catch (err) {
        toast('Reset failed: ' + err.message, 'error');
    }
}

async function onEditUrl() {
    const s = getState();
    if (!s.selectedTaskId || !s.selectedUrl) return;
    const newUrl = prompt('Edit URL:', s.selectedUrl);
    if (!newUrl || newUrl === s.selectedUrl) return;
    try {
        showStatus('Renaming URL...', 'warning');
        await api.renameUrl(s.selectedTaskId, s.selectedUrl, newUrl.trim());
        toast('URL renamed successfully', 'success');
        setState({ selectedUrl: newUrl.trim(), contentVersion: s.contentVersion + 1 });
        await reloadCurrentTask();
    } catch (err) {
        toast('Rename failed: ' + err.message, 'error');
    }
}

async function onAddUrl() {
    const s = getState();
    if (!s.selectedTaskId) return;
    const url = prompt('Enter a new URL to add:');
    if (!url) return;
    try {
        showStatus('Adding URL...', 'warning');
        await api.addUrl(s.selectedTaskId, url.trim());
        toast('URL added and flagged for capture', 'success');
        setState({ contentVersion: s.contentVersion + 1 });
        await reloadCurrentTask();
        await updateReviewProgress();
        selectUrl(s.selectedTaskId, url.trim());
    } catch (err) {
        toast('Add URL failed: ' + err.message, 'error');
    }
}

function onUploadMhtml() {
    $('#mhtml-picker').click();
}

async function onRecapture() {
    const s = getState();
    if (!s.selectedUrl) return;
    await api.setCaptureTarget(s.selectedTaskId, s.selectedUrl).catch(() => {});
    window.open(s.selectedUrl, '_blank');
    toast('Page opened. Pass any verification, then use the extension to capture.', 'success');
}

async function onBatchRecapture() {
    const s = getState();
    // Build queue: definite-severity issues only
    const items = s.issueIndex
        .filter(i => i.severity === 'definite')
        .map(i => ({ task_id: i.task_id, url: i.url }));

    if (items.length === 0) {
        toast('No red (definite-issue) URLs to recapture.', 'error');
        return;
    }

    try {
        const result = await api.startBatch(items);
        if (result.total === 0) {
            toast('All red URLs are already reviewed.', 'success');
            return;
        }
        setState({ batchActive: true, batchTotal: result.total, batchCompleted: 0 });
        toast(`Batch queued: ${result.total} URLs. Click the extension icon to start.`, 'success');
    } catch (err) {
        toast('Batch start failed: ' + err.message, 'error');
    }
}

// ============================================================
// Issue Navigation
// ============================================================

function navigateIssue(direction) {
    const s = getState();
    if (!s.issueIndex.length) return;
    let cursor = s.issueCursor + direction;
    if (cursor < 0) cursor = s.issueIndex.length - 1;
    if (cursor >= s.issueIndex.length) cursor = 0;
    setState({ issueCursor: cursor });

    const entry = s.issueIndex[cursor];
    if (entry.task_id !== s.selectedTaskId) {
        selectTask(entry.task_id).then(() => {
            selectUrl(entry.task_id, entry.url);
        });
    } else {
        selectUrl(entry.task_id, entry.url);
    }
}

// ============================================================
// Keyboard Shortcuts
// ============================================================

function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        const tag = e.target.tagName;
        const inInput = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';

        // Ctrl/Cmd shortcuts work even in inputs
        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'Enter') { e.preventDefault(); onMarkReviewed(); return; }
            if (e.key === 'u' || e.key === 'U') { e.preventDefault(); onRecapture(); return; }
            if (e.key === 'o' || e.key === 'O') { e.preventDefault(); onOpenFolder(); return; }
        }

        // Don't handle single-key shortcuts when typing in inputs
        if (inInput) return;

        // Actions
        if (e.key === 'r') { e.preventDefault(); onMarkReviewed(); return; }
        if (e.key === 'f') { e.preventDefault(); onFlagAsIssue(); return; }
        if (e.key === 'd' || e.key === 'Backspace') { e.preventDefault(); onDeleteUrl(); return; }
        if (e.key === 'x') { e.preventDefault(); onResetUrl(); return; }
        if (e.key === 'e') { e.preventDefault(); onEditUrl(); return; }
        if (e.key === 'a') { e.preventDefault(); onAddUrl(); return; }
        if (e.key === 'o') { e.preventDefault(); onOpenInBrowser(); return; }
        if (e.key === 'u') { e.preventDefault(); onRecapture(); return; }

        // Issue navigation
        if (e.key === 'n') { e.preventDefault(); navigateIssue(1); return; }
        if (e.key === 'N') { e.preventDefault(); navigateIssue(-1); return; }

        // Preview modes
        if (e.key === '1') { setState({ previewMode: 'screenshot' }); return; }
        if (e.key === '2') { setState({ previewMode: 'text' }); return; }
        if (e.key === '3') { setState({ previewMode: 'answer' }); return; }
        if (e.key === ' ') {
            e.preventDefault();
            const s = getState();
            setState({ previewMode: s.previewMode === 'text' ? 'screenshot' : 'text' });
            return;
        }

        // URL list navigation
        if (e.key === 'ArrowDown' || e.key === 'j') {
            e.preventDefault();
            navigateUrlList(1);
            return;
        }
        if (e.key === 'ArrowUp' || e.key === 'k') {
            e.preventDefault();
            navigateUrlList(-1);
            return;
        }
    });
}

function navigateUrlList(direction) {
    const s = getState();
    if (!s.selectedTaskId) return;

    // Use filtered URLs to respect active filters (Issues, Todo, etc.)
    const filtered = filterUrls(s);

    // If no URLs in current task or navigating past boundaries, jump to adjacent task
    if (!filtered.length) {
        navigateToAdjacentTask(direction);
        return;
    }

    const currentIdx = filtered.findIndex(u => u.url === s.selectedUrl);

    // No URL selected yet — select first or last in current task
    if (currentIdx < 0) {
        const url = direction > 0 ? filtered[0].url : filtered[filtered.length - 1].url;
        selectUrl(s.selectedTaskId, url);
        return;
    }

    let nextIdx = currentIdx + direction;

    if (nextIdx < 0 || nextIdx >= filtered.length) {
        // Past the boundary — jump to next/previous task
        navigateToAdjacentTask(direction);
        return;
    }

    selectUrl(s.selectedTaskId, filtered[nextIdx].url);
}

function navigateToAdjacentTask(direction) {
    const s = getState();
    if (!s.tasks.length) return;
    const taskIdx = s.tasks.findIndex(t => t.task_id === s.selectedTaskId);
    if (taskIdx < 0) return;

    let nextTaskIdx = taskIdx + direction;
    if (nextTaskIdx < 0) nextTaskIdx = s.tasks.length - 1;
    if (nextTaskIdx >= s.tasks.length) nextTaskIdx = 0;

    const nextTask = s.tasks[nextTaskIdx];
    selectTask(nextTask.task_id).then(() => {
        const filtered = filterUrls(getState());
        if (filtered.length > 0) {
            // Down → select first URL; Up → select last URL
            const url = direction > 0 ? filtered[0].url : filtered[filtered.length - 1].url;
            selectUrl(nextTask.task_id, url);
        }
    });
}

// ============================================================
// SSE (real-time updates from extension captures)
// ============================================================

function initSSE() {
    api.subscribeEvents((data) => {
        if (data.type === 'capture_complete') {
            const s = getState();
            if (!s.batchActive) {
                toast(`Captured: ${data.url?.substring(0, 60)}...`, 'success');
            }
            setState({ contentVersion: s.contentVersion + 1 });
            reloadCurrentTask();
            updateReviewProgress();
        }
        if (data.type === 'batch_progress') {
            setState({ batchCompleted: data.completed });
        }
        if (data.type === 'batch_complete') {
            setState({ batchActive: false, batchCompleted: 0, batchTotal: 0 });
            toast(`Batch complete! ${data.completed}/${data.total} URLs captured.`, 'success');
            reloadCurrentTask();
            updateReviewProgress();
        }
        if (data.type === 'batch_stopped') {
            setState({ batchActive: false, batchCompleted: 0, batchTotal: 0 });
            toast('Batch stopped.');
        }
        if (data.type === 'batch_captcha') {
            toast(`CAPTCHA detected (${data.captcha_type}) — switch to the browser tab to solve it.`, 'warning');
        }
        if (data.type === 'batch_started') {
            setState({ batchActive: true, batchTotal: data.total, batchCompleted: 0 });
        }
    });
}

// ============================================================
// Drag & Drop MHTML
// ============================================================

function initDragDrop() {
    const preview = $('#preview-panel');
    preview.addEventListener('dragover', (e) => {
        e.preventDefault();
        preview.classList.add('drop-target');
    });
    preview.addEventListener('dragleave', (e) => {
        // Only remove if actually leaving the panel (not entering a child)
        if (!preview.contains(e.relatedTarget)) {
            preview.classList.remove('drop-target');
        }
    });
    preview.addEventListener('drop', async (e) => {
        e.preventDefault();
        preview.classList.remove('drop-target');
        const s = getState();
        if (!s.selectedTaskId || !s.selectedUrl) {
            toast('Select a URL first, then drop a file', 'error');
            return;
        }
        const files = [...(e.dataTransfer?.files || [])];
        const mhtml = files.find(f => /\.(mhtml|mht)$/i.test(f.name));
        const pdf = files.find(f => /\.pdf$/i.test(f.name));
        if (pdf) {
            try {
                showStatus('Uploading PDF...', 'warning');
                await api.uploadPdf(s.selectedTaskId, s.selectedUrl, pdf);
                toast('PDF uploaded — content type switched to PDF', 'success');
                setState({ contentVersion: s.contentVersion + 1 });
                await reloadCurrentTask();
                await updateReviewProgress();
            } catch (err) {
                toast('PDF upload failed: ' + err.message, 'error');
            }
        } else if (mhtml) {
            try {
                showStatus('Uploading MHTML...', 'warning');
                await api.uploadMhtml(s.selectedTaskId, s.selectedUrl, mhtml);
                toast('MHTML uploaded successfully', 'success');
                await reloadCurrentTask();
            } catch (err) {
                toast('MHTML upload failed: ' + err.message, 'error');
            }
        } else {
            toast('Please drop a .pdf, .mhtml, or .mht file', 'error');
        }
    });
}

// Re-export for backward compatibility if needed
export { selectTask, selectUrl, showStatus, toast, $ };
