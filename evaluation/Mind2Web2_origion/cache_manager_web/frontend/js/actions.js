/**
 * Shared actions module — breaks circular dependency between main.js and components.
 *
 * Components import actions from here instead of from main.js.
 */
import { getState, setState } from './store.js';
import * as api from './api.js';

// ---- Task & URL selection ----

export async function selectTask(taskId) {
    setState({ selectedTaskId: taskId, selectedUrl: null, urls: [], currentText: null, currentIssues: null, answers: [] });
    try {
        const data = await api.getUrls(taskId);
        setState({
            urls: data.urls || [],
            urlTotal: data.total || 0,
            urlReviewedCount: data.reviewed_count || 0,
        });
    } catch (err) {
        console.error('Failed to load URLs:', err);
    }
    // Load answers
    try {
        const data = await api.getAnswers(taskId);
        setState({ answers: data.files || [] });
    } catch {}
}

export async function selectUrl(taskId, url) {
    // Auto-switch from answer view to screenshot when selecting a URL
    const mode = getState().previewMode;
    const updates = { selectedUrl: url, currentText: null, currentIssues: null };
    if (mode === 'answer') updates.previewMode = 'screenshot';
    setState(updates);
    // Set capture target for the extension
    api.setCaptureTarget(taskId, url).catch(() => {});
    // Check if this is a PDF (no text content available)
    const s = getState();
    const urlData = s.urls.find(u => u.url === url);
    const isPdf = urlData?.content_type === 'pdf';

    if (isPdf) {
        setState({ currentText: '', currentIssues: { has_issues: false } });
        // Auto-mark unflagged PDF as reviewed when viewed
        if (urlData && !['ok', 'fixed', 'skip'].includes(urlData.reviewed)) {
            // Only auto-review if no definite issues (i.e., not flagged)
            if (urlData.severity !== 'definite') {
                api.setReview(taskId, url, 'ok').catch(() => {});
                const urls = s.urls.map(u => u.url === url ? { ...u, reviewed: 'ok' } : u);
                setState({ urls });
                if (urlData.issues?.length > 0) {
                    incrementTaskIssueFixedCount(taskId);
                    updateReviewProgress();
                }
            }
        }
        return;
    }

    // Load text content for web URLs
    try {
        const data = await api.getText(taskId, url);
        setState({ currentText: data.text, currentIssues: data.issues });

        // Auto-mark as reviewed when viewed:
        // - Clean URLs (no issues) — no progress impact
        // - Possible-issue URLs (yellow) — viewing confirms they're OK
        // - Definite-issue URLs (red) — require manual recapture/mark
        if (!data.issues?.has_issues || data.issues?.severity !== 'definite') {
            const fresh = getState();
            const ud = fresh.urls.find(u => u.url === url);
            if (ud && !['ok', 'fixed', 'skip'].includes(ud.reviewed)) {
                api.setReview(taskId, url, 'ok').catch(() => {});
                const urls = fresh.urls.map(u => u.url === url ? { ...u, reviewed: 'ok' } : u);
                setState({ urls });
                // Update issue progress for possible-issue URLs
                if (data.issues?.has_issues) {
                    incrementTaskIssueFixedCount(taskId);
                    updateReviewProgress();
                }
            }
        }
    } catch {
        setState({ currentText: null, currentIssues: null });
    }
}

// ---- Reload current task ----

export async function reloadCurrentTask() {
    const s = getState();
    if (!s.selectedTaskId) return;
    try {
        const data = await api.getUrls(s.selectedTaskId);
        setState({
            urls: data.urls || [],
            urlTotal: data.total || 0,
            urlReviewedCount: data.reviewed_count || 0,
        });
        // Re-select current URL if still exists
        if (s.selectedUrl && data.urls?.some(u => u.url === s.selectedUrl)) {
            selectUrl(s.selectedTaskId, s.selectedUrl);
        }
    } catch {}
}

// ---- Review progress ----

export async function updateReviewProgress() {
    try {
        const data = await api.getReviewProgress();
        const el = document.querySelector('#review-progress');
        if (el) {
            el.textContent = data.total > 0
                ? `Fixed: ${data.reviewed}/${data.total} issues`
                : '';
        }
    } catch {}
}

// ---- Task issue fixed count ----

export function incrementTaskIssueFixedCount(taskId) {
    const s = getState();
    const tasks = s.tasks.map(t =>
        t.task_id === taskId ? { ...t, issue_reviewed_count: (t.issue_reviewed_count || 0) + 1 } : t
    );
    setState({ tasks });
}

// ---- Toast & Status ----

export function showStatus(msg, cls = '') {
    const el = document.querySelector('#preview-status');
    if (el) {
        el.textContent = msg;
        el.className = 'status-text' + (cls ? ' ' + cls : '');
    }
}

let _toastTimer = null;
export function toast(msg, type = '') {
    const el = document.querySelector('#toast');
    if (!el) return;
    el.textContent = msg;
    el.className = 'toast visible' + (type ? ' ' + type : '');
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => { el.className = 'toast'; }, 3000);
}

// ---- URL filtering (shared between url-list.js and main.js) ----

export function filterUrls(s) {
    let urls = s.urls;
    if (s.urlSearch) {
        const q = s.urlSearch.toLowerCase();
        urls = urls.filter(u => u.url.toLowerCase().includes(q) || u.domain.toLowerCase().includes(q));
    }
    if (s.urlContentFilter !== 'all') {
        urls = urls.filter(u => u.content_type === s.urlContentFilter);
    }
    if (s.urlIssuesFilter) {
        urls = urls.filter(u => u.issues?.length > 0);
    }
    if (s.urlTodoFilter) {
        urls = urls.filter(u => !['ok', 'fixed', 'skip'].includes(u.reviewed));
    }
    return urls;
}

// ---- DOM helper ----

export function $(sel) { return document.querySelector(sel); }
