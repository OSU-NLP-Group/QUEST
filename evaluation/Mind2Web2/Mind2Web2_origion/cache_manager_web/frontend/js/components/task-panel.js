/**
 * Task panel component — renders the task list with search and filters.
 */
import { getState, setState, subscribe } from '../store.js';
import { selectTask } from '../actions.js';

export function initTaskPanel() {
    const searchInput = document.getElementById('task-search');
    const issuesCheckbox = document.getElementById('task-issues-only');
    const listEl = document.getElementById('task-list');
    const statsEl = document.getElementById('task-stats');

    searchInput.addEventListener('input', () => {
        setState({ taskSearch: searchInput.value });
    });
    issuesCheckbox.addEventListener('change', () => {
        setState({ taskIssuesOnly: issuesCheckbox.checked });
    });

    // Only re-render when task-related state changes
    subscribe((s) => {
        renderTaskList(listEl, s);
        renderTaskStats(statsEl, s);
    }, ['tasks', 'taskIssues', 'selectedTaskId', 'taskSearch', 'taskIssuesOnly', 'issueIndex']);
}

function renderTaskList(container, s) {
    const filtered = filterTasks(s);
    const html = filtered.map(task => {
        const issueInfo = s.taskIssues[task.task_id] || {};
        const issueCount = issueInfo.count || 0;
        const severity = issueInfo.severity || '';
        const allFixed = issueCount > 0 && (task.issue_reviewed_count || 0) >= issueCount;
        const dotClass = allFixed ? 'clean' : (issueCount > 0 ? severity : 'clean');
        const isSelected = task.task_id === s.selectedTaskId;

        const detailParts = [`${task.total_urls} URLs`];
        if (issueCount > 0) {
            const fixedCount = task.issue_reviewed_count || 0;
            if (fixedCount > 0 && fixedCount >= issueCount) {
                detailParts.push(`<span class="task-issue-count" style="color: var(--c-success)">${issueCount} issues (all fixed)</span>`);
            } else if (fixedCount > 0) {
                detailParts.push(`<span class="task-issue-count">${fixedCount}/${issueCount} fixed</span>`);
            } else {
                detailParts.push(`<span class="task-issue-count">${issueCount} issues</span>`);
            }
        }

        return `<div class="task-item${isSelected ? ' selected' : ''}" data-task-id="${esc(task.task_id)}">
            <span class="task-dot ${dotClass}"></span>
            <div class="task-info">
                <div class="task-name">${esc(task.task_id)}</div>
                <div class="task-detail">${detailParts.join(' &middot; ')}</div>
            </div>
        </div>`;
    }).join('');

    container.innerHTML = html;

    // Attach click handlers via event delegation
    container.onclick = (e) => {
        const item = e.target.closest('.task-item');
        if (item) selectTask(item.dataset.taskId);
    };

    // Scroll selected into view
    const selectedEl = container.querySelector('.task-item.selected');
    if (selectedEl) {
        selectedEl.scrollIntoView({ block: 'nearest' });
    }
}

function renderTaskStats(el, s) {
    if (s.tasks.length === 0) {
        el.textContent = 'No tasks loaded';
    } else {
        const totalUrls = s.tasks.reduce((sum, t) => sum + t.total_urls, 0);
        const totalIssues = s.issueIndex.length;
        let text = `${s.tasks.length} tasks · ${totalUrls} URLs`;
        if (totalIssues > 0) text += ` · ${totalIssues} issues`;
        el.textContent = text;
    }
}

function filterTasks(s) {
    let tasks = s.tasks;
    if (s.taskSearch) {
        const q = s.taskSearch.toLowerCase();
        tasks = tasks.filter(t => t.task_id.toLowerCase().includes(q));
    }
    if (s.taskIssuesOnly) {
        tasks = tasks.filter(t => {
            const info = s.taskIssues[t.task_id];
            return info && info.count > 0;
        });
    }
    return tasks;
}

function esc(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
