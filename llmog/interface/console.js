<!-- ════════════════════════════════════════════════════════════════════════
     Gradio API Console — client-side JS helpers
     Inject this with gr.HTML(value="", head=CONSOLE_JS) once, near the top of
     your gr.Blocks() layout. Gradio 6 renders gr.HTML values via innerHTML,
     which never executes <script> tags — but head= content is injected into
     the document <head>, where scripts do run.
     ════════════════════════════════════════════════════════════════════════ -->
<script>
// ── Copy panel text to clipboard ──────────────────────────────────────────
function copyOut(elementId) {
    const wrapper = document.getElementById(elementId);
    if (!wrapper) return;
    const textarea = wrapper.querySelector('textarea');
    const text = textarea ? textarea.value : (wrapper.innerText || wrapper.textContent);
    navigator.clipboard.writeText(text).then(() => {
        const btn = event.currentTarget;
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
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(url); document.body.removeChild(a); }, 500);
}

// ── Auto-scroll a log textarea to the bottom ──────────────────────────────
// Call this after each server update to keep the log tail visible.
function autoScrollLog(elementId) {
    const wrapper = document.getElementById(elementId);
    if (!wrapper) return;
    const textarea = wrapper.querySelector('textarea');
    if (textarea) {
        textarea.scrollTop = textarea.scrollHeight;
    }
}

// ── Observe log textareas for mutations and auto-scroll ───────────────────
// We attach a MutationObserver once the DOM is ready so logs always tail.
function attachLogAutoScroll(elementId) {
    const tryAttach = () => {
        const wrapper = document.getElementById(elementId);
        if (!wrapper) { setTimeout(tryAttach, 500); return; }
        const textarea = wrapper.querySelector('textarea');
        if (!textarea) { setTimeout(tryAttach, 500); return; }

        const observer = new MutationObserver(() => {
            // Only auto-scroll if user is near the bottom (within 120px)
            const distFromBottom = textarea.scrollHeight - textarea.scrollTop - textarea.clientHeight;
            if (distFromBottom < 120) {
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

// Initialise on load
document.addEventListener('DOMContentLoaded', () => {
    attachLogAutoScroll('server-log-ta');
    attachLogAutoScroll('pipeline-log-ta');
    observeProgressBar();
});
// Gradio re-renders after navigation, so also run after a short delay
setTimeout(() => {
    attachLogAutoScroll('server-log-ta');
    attachLogAutoScroll('pipeline-log-ta');
    observeProgressBar();
}, 2000);
</script>
