<!-- ════════════════════════════════════════════════════════════════════════
     Gradio API Console — client-side JS helpers
     Inject this with gr.HTML(value="", head=CONSOLE_JS) once, near the top of
     your gr.Blocks() layout. Gradio 6 renders gr.HTML values via innerHTML,
     which never executes <script> tags — but head= content is injected into
     the document <head>, where scripts do run.
     ════════════════════════════════════════════════════════════════════════ -->
<script>
// ── Copy panel text to clipboard ──────────────────────────────────────────
function copyOut(elementId, btn) {
    const wrapper = document.getElementById(elementId);
    if (!wrapper) return;
    // Capture the button NOW — `window.event` is unreliable inside async callbacks
    if (!btn) {
        try { btn = (typeof event !== 'undefined' && event && event.currentTarget) || null; } catch (_) { btn = null; }
    }
    const textarea = wrapper.querySelector('textarea');
    const text = textarea ? textarea.value : (wrapper.innerText || wrapper.textContent);
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

// ── Bauhaus cool motion — floating bubbles + ripple + staggered reveal ──
function initBauhausMotion() {
    // Inject extra floating bubbles if not present (yellow/orange) — CSS handles float/pulse
    const container = document.querySelector('.gradio-container');
    if (container && !container.querySelector('.bauhaus-bubble--yellow')) {
        const y = document.createElement('div');
        y.className = 'bauhaus-bubble bauhaus-bubble--yellow';
        y.setAttribute('aria-hidden','true');
        container.appendChild(y);
        const o = document.createElement('div');
        o.className = 'bauhaus-bubble bauhaus-bubble--orange';
        o.setAttribute('aria-hidden','true');
        container.appendChild(o);
    }
    // Ripple on dark pill buttons
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('button.primary, .gr-button.primary, .btn-canvas-primary');
        if (!btn) return;
        const rect = btn.getBoundingClientRect();
        const ripple = document.createElement('span');
        ripple.style.position = 'absolute';
        ripple.style.left = (e.clientX - rect.left) + 'px';
        ripple.style.top = (e.clientY - rect.top) + 'px';
        ripple.style.width = ripple.style.height = '10px';
        ripple.style.background = 'rgba(245,200,66,0.45)';
        ripple.style.borderRadius = '50%';
        ripple.style.transform = 'translate(-50%,-50%) scale(0)';
        ripple.style.pointerEvents = 'none';
        ripple.style.animation = 'btn-pop 0.55s ease-out forwards';
        btn.style.position = 'relative';
        btn.style.overflow = 'hidden';
        btn.appendChild(ripple);
        setTimeout(() => ripple.remove(), 600);
    });
    // Staggered reveal via IntersectionObserver for cards
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(ent => {
            if (ent.isIntersecting) {
                ent.target.style.animationPlayState = 'running';
                observer.unobserve(ent.target);
            }
        });
    }, { threshold: 0.12 });
    document.querySelectorAll('.gr-group, .config-card, .metric-card').forEach(el => {
        el.style.animationPlayState = 'paused';
        observer.observe(el);
    });
}

// ── Force dropdown option lists to escape overflow:hidden ancestors ─────
function fixDropdownClipping() {
    // Gradio renders option lists inside the component — any ancestor with
    // overflow:hidden (tabs/tabitem/accordion/container) clips the popup.
    // On every click inside a dropdown, force all ancestors to visible.
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.gr-dropdown, [data-testid="dropdown"]')) return;
        const chain = ['.gradio-container','.tabs','.tabitem','.gr-accordion','.gr-dropdown','.gr-group'];
        chain.forEach(sel => {
            document.querySelectorAll(sel).forEach(el => { el.style.overflow = 'visible'; });
        });
    }, true);
}

// Initialise on load
document.addEventListener('DOMContentLoaded', () => {
    attachLogAutoScroll('server-log-ta');
    attachLogAutoScroll('pipeline-log-ta');
    observeProgressBar();
    initBauhausMotion();
    fixDropdownClipping();
});
// Gradio re-renders after navigation, so also run after a short delay
setTimeout(() => {
    attachLogAutoScroll('server-log-ta');
    attachLogAutoScroll('pipeline-log-ta');
    observeProgressBar();
    initBauhausMotion();
    fixDropdownClipping();
}, 2000);
</script>
