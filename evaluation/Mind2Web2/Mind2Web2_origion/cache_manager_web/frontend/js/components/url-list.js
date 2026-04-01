/**
 * URL list component — renders URLs with filters, progress bar, and actions.
 */
import { getState, setState, subscribe } from '../store.js';
import { selectUrl, filterUrls } from '../actions.js';

export function initUrlList() {
    const searchInput = document.getElementById('url-search');
    const listEl = document.getElementById('url-list');
    const statsEl = document.getElementById('url-stats');

    // Search
    searchInput.addEventListener('input', () => {
        setState({ urlSearch: searchInput.value });
    });

    // Content type filter buttons
    document.querySelectorAll('#url-panel .filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const filter = btn.dataset.filter;
            if (['all', 'web', 'pdf'].includes(filter)) {
                // Content type filter — radio-like
                document.querySelectorAll('#url-panel .filter-btn[data-filter="all"], #url-panel .filter-btn[data-filter="web"], #url-panel .filter-btn[data-filter="pdf"]')
                    .forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                setState({ urlContentFilter: filter });
            } else if (filter === 'issues') {
                btn.classList.toggle('active');
                setState({ urlIssuesFilter: btn.classList.contains('active') });
            } else if (filter === 'todo') {
                btn.classList.toggle('active');
                setState({ urlTodoFilter: btn.classList.contains('active') });
            }
        });
    });

    // Only re-render when URL-related state changes
    subscribe((s) => {
        renderUrlList(listEl, s);
        renderUrlStats(statsEl, s);
        renderProgressBar(s);
    }, ['urls', 'selectedUrl', 'selectedTaskId', 'urlSearch', 'urlContentFilter',
        'urlIssuesFilter', 'urlTodoFilter', 'urlTotal', 'urlReviewedCount']);
}

function renderUrlList(container, s) {
    if (!s.selectedTaskId) {
        container.innerHTML = '<div class="empty-state">Select a task to view URLs</div>';
        return;
    }

    const filtered = filterUrls(s);

    if (filtered.length === 0 && s.urls.length > 0) {
        container.innerHTML = '<div class="empty-state">No URLs match the current filters</div>';
        return;
    }

    if (filtered.length === 0) {
        container.innerHTML = '<div class="empty-state">No URLs in this task</div>';
        return;
    }

    const html = filtered.map(u => {
        const isSelected = u.url === s.selectedUrl;
        let borderClass = 'clean';
        if (u.issues?.length > 0) {
            if (u.reviewed === 'recaptured') {
                // Batch-recaptured — blue, still needs human review
                borderClass = 'recaptured';
            } else if (['ok', 'fixed', 'skip'].includes(u.reviewed)) {
                borderClass = 'reviewed';
            } else {
                borderClass = u.severity === 'definite' ? 'definite' : 'possible';
            }
        } else if (u.reviewed === 'recaptured') {
            borderClass = 'recaptured';
        } else if (['ok', 'fixed', 'skip'].includes(u.reviewed)) {
            borderClass = 'reviewed';
        }

        const checkmark = ['ok', 'fixed'].includes(u.reviewed)
            ? '<span class="url-reviewed-check">&#10003;</span>' : '';

        return `<div class="url-item${isSelected ? ' selected' : ''}" data-url="${escAttr(u.url)}">
            <div class="url-border ${borderClass}"></div>
            <div class="url-info">
                <div class="url-top-row">
                    <span class="url-domain">${esc(u.domain)}</span>
                    <span class="url-badge">${esc(u.content_type)}</span>
                    ${checkmark}
                </div>
                <div class="url-path">${esc(u.path)}</div>
            </div>
        </div>`;
    }).join('');

    container.innerHTML = html;

    // Event delegation for clicks
    container.onclick = (e) => {
        const item = e.target.closest('.url-item');
        if (item) {
            const url = item.dataset.url;
            const s = getState();
            selectUrl(s.selectedTaskId, url);
        }
    };

    // Scroll selected into view
    const selectedEl = container.querySelector('.url-item.selected');
    if (selectedEl) {
        selectedEl.scrollIntoView({ block: 'nearest' });
    }
}

function renderUrlStats(el, s) {
    if (!s.selectedTaskId) {
        el.textContent = 'Select a task';
        return;
    }
    if (s.urls.length === 0) {
        el.textContent = 'No URLs';
        return;
    }
    const web = s.urls.filter(u => u.content_type === 'web').length;
    const pdf = s.urls.filter(u => u.content_type === 'pdf').length;
    const issues = s.urls.filter(u => u.issues?.length > 0).length;
    const parts = [`${s.urls.length} URLs`];
    if (web > 0 && pdf > 0) parts.push(`${web} web · ${pdf} PDF`);
    if (issues > 0) parts.push(`${issues} issues`);
    el.textContent = parts.join(' · ');
}

function renderProgressBar(s) {
    const bar = document.getElementById('url-progress-bar');
    // Progress tracks only issue URLs (yellow/red), not all URLs
    const issueUrls = s.urls.filter(u => u.issues?.length > 0);
    const fixedCount = issueUrls.filter(u =>
        ['ok', 'fixed', 'skip'].includes(u.reviewed) && u.reviewed !== 'recaptured'
    ).length;
    const issueTotal = issueUrls.length;
    if (issueTotal === 0) {
        bar.style.display = 'none';
        return;
    }
    bar.style.display = '';
    const pct = Math.round((fixedCount / issueTotal) * 100);
    bar.querySelector('.progress-fill').style.width = pct + '%';
    bar.querySelector('.progress-text').textContent = `Fixed: ${fixedCount}/${issueTotal} issues`;
}

function esc(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
}

function escAttr(str) {
    return (str || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
}
