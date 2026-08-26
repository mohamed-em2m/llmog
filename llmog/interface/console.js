<!-- ════════════════════════════════════════════════════════════════════════
     Gradio API Console — client-side JS helpers
     ════════════════════════════════════════════════════════════════════════ -->
<script>
// ── Copy panel text to clipboard ──────────────────────────────────────────
function copyOut(elementId, btn) {
    const wrapper = document.getElementById(elementId);
    if (!wrapper) return;
    if (!btn) {
        try { btn = (typeof event !== 'undefined' && event && event.currentTarget) || null; } catch (_) { btn = null; }
    }
    const textarea = wrapper.querySelector('textarea');
    const text = textarea ? textarea.value : (wrapper.innerText || wrapper.textContent || '');
    if (!text.trim()) return;
    navigator.clipboard.writeText(text).then(() => {
        if (!btn) return;
        const orig = btn.textContent;
        btn.textContent = '✓ Copied';
        btn.classList.add('copied');
        setTimeout(() => { btn.textContent = orig; btn.classList.remove('copied'); }, 2000);
    });
}

// ── Download panel text as a file ─────────────────────────────────────────
function downloadPanelText(rawTextareaId, filename) {
    const wrapper = document.getElementById(rawTextareaId);
    if (!wrapper) return;
    const textarea = wrapper.querySelector('textarea');
    const text = textarea ? textarea.value : (wrapper.innerText || wrapper.textContent || '');
    if (!text.trim()) { alert('No content to download yet.'); return; }
    const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'output.txt';
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); document.body.removeChild(a); }, 500);
}

// ── Auto-scroll a log textarea to the bottom ──────────────────────────────
function autoScrollLog(elementId) {
    const wrapper = document.getElementById(elementId);
    if (!wrapper) return;
    const textarea = wrapper.querySelector('textarea') || (wrapper.tagName === 'TEXTAREA' ? wrapper : null);
    if (textarea) {
        textarea.scrollTop = textarea.scrollHeight;
    }
}

// ── Observe log textareas for mutations and auto-scroll ───────────────────
function attachLogAutoScroll(elementId) {
    const tryAttach = () => {
        const wrapper = document.getElementById(elementId);
        if (!wrapper) { setTimeout(tryAttach, 600); return; }
        const textarea = wrapper.querySelector('textarea') || (wrapper.tagName === 'TEXTAREA' ? wrapper : null);
        if (!textarea) { setTimeout(tryAttach, 600); return; }

        const observer = new MutationObserver(() => {
            const distFromBottom = textarea.scrollHeight - textarea.scrollTop - textarea.clientHeight;
            if (distFromBottom < 160) {
                textarea.scrollTop = textarea.scrollHeight;
            }
        });
        observer.observe(textarea, { attributes: true, childList: true, subtree: true, characterData: true });
    };
    tryAttach();
}

// ── Add progress-bar striped class when active ────────────────────────────
function observeProgressBar() {
    const tryAttach = () => {
        const fills = document.querySelectorAll('.custom-progress-fill');
        if (!fills.length) { setTimeout(tryAttach, 800); return; }
        fills.forEach(fill => {
            const observer = new MutationObserver(() => {
                const w = parseInt(fill.style.width || '0');
                if (w > 0 && w < 100) {
                    fill.classList.add('striped');
                } else {
                    fill.classList.remove('striped');
                }
            });
            observer.observe(fill, { attributes: true, attributeFilter: ['style'] });
        });
    };
    tryAttach();
}

// ── Bauhaus ambient motion — decorative floating shapes ───────────────────
function initBauhausMotion() {
    try {
        const container = document.querySelector('.gradio-container');
        if (container && !container.querySelector('.bauhaus-bubble--yellow')) {
            const y = document.createElement('div');
            y.className = 'bauhaus-bubble bauhaus-bubble--yellow';
            y.setAttribute('aria-hidden', 'true');
            container.appendChild(y);
            const o = document.createElement('div');
            o.className = 'bauhaus-bubble bauhaus-bubble--orange';
            o.setAttribute('aria-hidden', 'true');
            container.appendChild(o);
        }
    } catch (_) {}
}

// ── Initialise on document ready & route transitions ──────────────────────
document.addEventListener('DOMContentLoaded', () => {
    try { attachLogAutoScroll('server-log-ta'); } catch (_) {}
    try { attachLogAutoScroll('pipeline-log-ta'); } catch (_) {}
    try { observeProgressBar(); } catch (_) {}
    try { initBauhausMotion(); } catch (_) {}
});

setTimeout(() => {
    try { attachLogAutoScroll('server-log-ta'); } catch (_) {}
    try { attachLogAutoScroll('pipeline-log-ta'); } catch (_) {}
    try { observeProgressBar(); } catch (_) {}
    try { initBauhausMotion(); } catch (_) {}
}, 1500);
</script>
