/**
 * API client for Cache Manager backend.
 */

const BASE = '';  // Same origin

async function request(method, path, body = null) {
    const opts = { method, headers: {} };
    if (body !== null) {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(body);
    }
    const res = await fetch(BASE + path, opts);
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res;
}

export async function getStatus() {
    return (await request('GET', '/api/status')).json();
}

export async function loadCache(path) {
    return (await request('POST', '/api/load', { path })).json();
}

export async function getTasks() {
    return (await request('GET', '/api/tasks')).json();
}

export async function getUrls(taskId) {
    return (await request('GET', `/api/tasks/${encodeURIComponent(taskId)}/urls`)).json();
}

export async function getText(taskId, url) {
    return (await request('GET', `/api/content/${encodeURIComponent(taskId)}/text?url=${encodeURIComponent(url)}`)).json();
}

export function screenshotUrl(taskId, url, version = 0) {
    return `/api/content/${encodeURIComponent(taskId)}/screenshot?url=${encodeURIComponent(url)}&v=${version}`;
}

export function pdfUrl(taskId, url) {
    return `/api/content/${encodeURIComponent(taskId)}/pdf?url=${encodeURIComponent(url)}`;
}

export async function getAnswers(taskId) {
    return (await request('GET', `/api/answers/${encodeURIComponent(taskId)}`)).json();
}

export async function setReview(taskId, url, status) {
    return (await request('POST', `/api/review/${encodeURIComponent(taskId)}`, { url, status })).json();
}

export async function getReviewProgress() {
    return (await request('GET', '/api/review-progress')).json();
}

export async function setCaptureTarget(taskId, url) {
    return (await request('POST', '/api/capture/target', { task_id: taskId, url })).json();
}

export async function deleteUrl(taskId, url) {
    return (await request('DELETE', `/api/urls/${encodeURIComponent(taskId)}?url=${encodeURIComponent(url)}`)).json();
}

export async function scanAll() {
    return (await request('POST', '/api/scan')).json();
}

export async function startBatch(items) {
    return (await request('POST', '/api/capture/batch/start', { items })).json();
}

export async function getBatchStatus() {
    return (await request('GET', '/api/capture/batch/status')).json();
}

export async function skipBatchUrl() {
    return (await request('POST', '/api/capture/batch/skip')).json();
}

export async function stopBatch() {
    return (await request('POST', '/api/capture/batch/stop')).json();
}

export async function flagUrl(taskId, url) {
    return (await request('POST', `/api/flag/${encodeURIComponent(taskId)}`, { url })).json();
}

export async function resetUrl(taskId, url) {
    return (await request('POST', `/api/reset/${encodeURIComponent(taskId)}`, { url })).json();
}

export async function renameUrl(taskId, oldUrl, newUrl) {
    return (await request('POST', `/api/urls/${encodeURIComponent(taskId)}/rename`, { old_url: oldUrl, new_url: newUrl })).json();
}

export async function addUrl(taskId, url) {
    return (await request('POST', `/api/urls/${encodeURIComponent(taskId)}`, { url, auto_flag: true })).json();
}

export async function uploadPdf(taskId, url, file) {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`/api/upload-pdf/${encodeURIComponent(taskId)}?url=${encodeURIComponent(url)}`, {
        method: 'POST',
        body: form,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
}

export async function uploadMhtml(taskId, url, file) {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`/api/upload-mhtml/${encodeURIComponent(taskId)}?url=${encodeURIComponent(url)}`, {
        method: 'POST',
        body: form,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
}

/**
 * Subscribe to Server-Sent Events.
 * @param {function} onEvent - callback(data)
 * @returns {EventSource}
 */
export function subscribeEvents(onEvent) {
    const es = new EventSource('/api/events');
    es.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            onEvent(data);
        } catch {}
    };
    es.onerror = () => {
        // Auto-reconnect is built into EventSource
    };
    return es;
}
