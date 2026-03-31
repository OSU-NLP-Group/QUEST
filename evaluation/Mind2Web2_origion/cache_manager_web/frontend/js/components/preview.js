/**
 * Preview panel component — screenshot, text, and answer views.
 */
import { getState, setState, subscribe } from '../store.js';
import * as api from '../api.js';

let currentImgEl = null;  // current screenshot <img> element
let currentImgSrc = '';   // track current image source to avoid reloads

export function initPreview() {
    // Mode tabs
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            setState({ previewMode: btn.dataset.mode });
        });
    });

    // Zoom controls
    document.getElementById('btn-fit-width').addEventListener('click', () => {
        setState({ fitToWidth: !getState().fitToWidth, zoomLevel: 1.0 });
    });
    document.getElementById('btn-zoom-in').addEventListener('click', () => {
        setState({ fitToWidth: false, zoomLevel: Math.min(5, getState().zoomLevel * 1.15) });
    });
    document.getElementById('btn-zoom-out').addEventListener('click', () => {
        setState({ fitToWidth: false, zoomLevel: Math.max(0.1, getState().zoomLevel / 1.15) });
    });

    // Ctrl+Wheel zoom on screenshot
    document.getElementById('view-screenshot').addEventListener('wheel', (e) => {
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            const s = getState();
            const factor = e.deltaY < 0 ? 1.1 : 0.9;
            setState({ fitToWidth: false, zoomLevel: Math.max(0.1, Math.min(5, s.zoomLevel * factor)) });
        }
    }, { passive: false });

    // Answer file selector
    document.getElementById('answer-file-select').addEventListener('change', () => {
        renderAnswer(getState());
    });

    // Subscribe only to preview-related state
    subscribe(render, [
        'previewMode', 'selectedUrl', 'selectedTaskId', 'urls',
        'currentText', 'currentIssues', 'answers',
        'fitToWidth', 'zoomLevel', 'contentVersion',
    ]);
}

function render(s) {
    // Update mode tab active state
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === s.previewMode);
    });

    // Show/hide controls
    document.getElementById('screenshot-controls').style.display = s.previewMode === 'screenshot' ? '' : 'none';
    document.getElementById('answer-controls').style.display = s.previewMode === 'answer' ? '' : 'none';

    // Show active view
    document.querySelectorAll('#preview-content .view').forEach(v => v.classList.remove('active'));
    const activeView = document.getElementById(`view-${s.previewMode}`);
    if (activeView) activeView.classList.add('active');

    // Update URL display
    const urlEl = document.getElementById('preview-url');
    urlEl.textContent = s.selectedUrl || 'No URL selected';
    if (s.selectedUrl) {
        urlEl.title = s.selectedUrl;
    } else {
        urlEl.title = '';
    }

    // Render based on mode
    if (s.previewMode === 'screenshot') renderScreenshot(s);
    else if (s.previewMode === 'text') renderText(s);
    else if (s.previewMode === 'answer') renderAnswerPanel(s);

    // Update status
    updateStatus(s);

    // Update fit button
    document.getElementById('btn-fit-width').classList.toggle('active', s.fitToWidth);
    document.getElementById('zoom-label').textContent = s.fitToWidth ? 'Fit' : `${Math.round(s.zoomLevel * 100)}%`;
}

function renderScreenshot(s) {
    const container = document.getElementById('screenshot-container');

    if (!s.selectedTaskId || !s.selectedUrl) {
        container.innerHTML = '<div class="placeholder">Select a URL to preview</div>';
        currentImgEl = null;
        currentImgSrc = '';
        return;
    }

    // Check content type
    const urlData = s.urls.find(u => u.url === s.selectedUrl);
    if (urlData?.content_type === 'pdf') {
        const pdfSrc = api.pdfUrl(s.selectedTaskId, s.selectedUrl);
        container.innerHTML = `<iframe src="${pdfSrc}" class="pdf-embed"></iframe>`;
        currentImgEl = null;
        currentImgSrc = '';
        return;
    }

    const imgSrc = api.screenshotUrl(s.selectedTaskId, s.selectedUrl, s.contentVersion);

    // Recreate img if URL or version changed
    if (imgSrc !== currentImgSrc) {
        currentImgSrc = imgSrc;
        const img = document.createElement('img');
        img.style.opacity = '0';
        img.style.transition = 'opacity .2s ease';
        img.src = imgSrc;
        img.alt = 'Screenshot';
        img.addEventListener('load', () => {
            img.style.opacity = '1';
            const loader = container.querySelector('.screenshot-loading');
            if (loader) loader.remove();
            document.getElementById('screenshot-info').textContent =
                `${img.naturalWidth} x ${img.naturalHeight}`;
        });
        img.addEventListener('error', () => {
            container.innerHTML = '<div class="placeholder">Screenshot not available</div>';
            currentImgEl = null;
            currentImgSrc = '';
        });
        container.innerHTML = '<div class="screenshot-loading">Loading screenshot...</div>';
        container.appendChild(img);
        currentImgEl = img;
    }

    // Apply zoom / fit
    if (currentImgEl) {
        if (s.fitToWidth) {
            currentImgEl.style.maxWidth = '100%';
            currentImgEl.style.width = '100%';
            currentImgEl.style.height = 'auto';
            currentImgEl.classList.remove('original-size');
        } else {
            currentImgEl.style.maxWidth = 'none';
            const w = currentImgEl.naturalWidth || 800;
            currentImgEl.style.width = `${Math.round(w * s.zoomLevel)}px`;
            currentImgEl.style.height = 'auto';
            currentImgEl.classList.add('original-size');
        }
    }
}

function renderText(s) {
    const pre = document.getElementById('text-content');
    const urlData = s.urls.find(u => u.url === s.selectedUrl);
    if (urlData?.content_type === 'pdf') {
        const flagged = urlData.severity === 'definite';
        pre.textContent = flagged
            ? 'PDF content — flagged as having issues.\nUse the Screenshot tab to view, or Upload PDF to replace.'
            : 'PDF content — use the Screenshot tab to view.\nYou can also drag & drop a .pdf file onto the preview to replace it.';
    } else if (s.currentText != null) {
        pre.textContent = s.currentText;
    } else if (s.selectedUrl) {
        pre.textContent = 'Loading text...';
    } else {
        pre.textContent = 'Select a URL to view text content';
    }
}

function renderAnswerPanel(s) {
    const select = document.getElementById('answer-file-select');
    // Rebuild options if answer list changed
    const currentOptions = [...select.options].map(o => o.value).join(',');
    const newOptions = s.answers.map((_, i) => String(i)).join(',');
    if (currentOptions !== newOptions) {
        select.innerHTML = '';
        if (s.answers.length === 0) {
            const opt = document.createElement('option');
            opt.textContent = 'No answer files';
            opt.disabled = true;
            select.appendChild(opt);
        } else {
            s.answers.forEach((a, i) => {
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = a.name;
                select.appendChild(opt);
            });
        }
    }
    renderAnswer(s);
}

function renderAnswer(s) {
    const el = document.getElementById('answer-content');
    const select = document.getElementById('answer-file-select');
    const idx = parseInt(select.value, 10);
    if (s.answers.length > 0 && idx >= 0 && idx < s.answers.length) {
        let text = s.answers[idx].content;
        // Highlight current URL if present
        if (s.selectedUrl && text.includes(s.selectedUrl)) {
            text = text.replaceAll(s.selectedUrl, `**>>> ${s.selectedUrl} <<<**`);
        }
        el.innerHTML = typeof marked !== 'undefined' ? marked.parse(text) : `<pre>${text}</pre>`;
    } else {
        el.textContent = s.answers.length === 0
            ? (s.selectedTaskId ? 'No answer files found for this task.' : 'Select a task to view answers.')
            : '';
    }
}

function updateStatus(s) {
    const el = document.getElementById('preview-status');
    if (!s.selectedUrl) {
        el.textContent = 'Ready';
        el.className = 'status-text';
        return;
    }
    if (s.currentIssues?.has_issues) {
        const cnt = (s.currentIssues.keywords?.length || 0) + (s.currentIssues.patterns?.length || 0);
        const kw = [...(s.currentIssues.keywords || []), ...(s.currentIssues.patterns || [])].slice(0, 5).join(', ');
        el.textContent = `${cnt} issue(s) detected: ${kw}`;
        el.className = 'status-text warning';
    } else if (s.currentText != null) {
        el.textContent = 'No issues detected';
        el.className = 'status-text success';
    } else {
        el.textContent = 'Loading...';
        el.className = 'status-text';
    }
}
