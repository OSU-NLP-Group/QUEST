/**
 * Minimal reactive state store for the Cache Manager UI.
 *
 * Supports selective subscriptions: subscribers can specify which state keys
 * they care about, and will only be notified when those keys change.
 */

const state = {
    // Cache status
    loaded: false,
    agentName: '',
    agentPath: '',
    stats: {},

    // Tasks
    tasks: [],            // [{task_id, total_urls, web_urls, pdf_urls, issue_urls, reviewed_count}]
    taskIssues: {},       // {task_id: {count, severity}}
    selectedTaskId: null,

    // URLs for current task
    urls: [],             // [{url, content_type, domain, path, issues, severity, reviewed}]
    selectedUrl: null,
    urlTotal: 0,
    urlReviewedCount: 0,

    // Preview
    previewMode: 'screenshot',  // 'screenshot' | 'text' | 'answer'
    currentText: null,
    currentIssues: null,
    answers: [],                // [{name, content}]

    // Issue navigation
    issueIndex: [],       // [{task_id, url, severity, ...}]
    issueCursor: -1,

    // Filters
    taskSearch: '',
    taskIssuesOnly: false,
    urlSearch: '',
    urlContentFilter: 'all',  // 'all' | 'web' | 'pdf'
    urlIssuesFilter: false,
    urlTodoFilter: false,

    // Batch capture
    batchActive: false,
    batchTotal: 0,
    batchCompleted: 0,

    // Content version — incremented on capture to bust browser cache
    contentVersion: 0,

    // Zoom
    fitToWidth: true,
    zoomLevel: 1.0,
};

// Listeners: {fn, keys: Set | null}
// If keys is null, the listener fires on any change (global listener).
const listeners = [];

export function getState() {
    return state;
}

export function setState(partial) {
    const changedKeys = new Set();
    for (const key of Object.keys(partial)) {
        if (state[key] !== partial[key]) {
            changedKeys.add(key);
        }
    }
    Object.assign(state, partial);
    if (changedKeys.size > 0) {
        notify(changedKeys);
    }
}

/**
 * Subscribe to state changes.
 * @param {function} fn - callback(state)
 * @param {string[]} [keys] - optional list of state keys to watch.
 *   If omitted, the listener fires on any state change.
 * @returns {function} unsubscribe function
 */
export function subscribe(fn, keys = null) {
    const entry = { fn, keys: keys ? new Set(keys) : null };
    listeners.push(entry);
    return () => {
        const idx = listeners.indexOf(entry);
        if (idx >= 0) listeners.splice(idx, 1);
    };
}

function notify(changedKeys) {
    for (const { fn, keys } of listeners) {
        try {
            if (!keys || setsOverlap(keys, changedKeys)) {
                fn(state);
            }
        } catch (e) {
            console.error('Store listener error:', e);
        }
    }
}

function setsOverlap(a, b) {
    for (const k of b) {
        if (a.has(k)) return true;
    }
    return false;
}
