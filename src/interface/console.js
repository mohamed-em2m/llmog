<!-- ════════════════════════════════════════════════════════════════════════
     Gradio API Console — client-side JS helpers
     Inject this with gr.HTML(CONSOLE_JS) once, near the top of your
     gr.Blocks() layout (right after the title header is fine). It must be
     wrapped in <script> tags as shown — Gradio's gr.HTML renders raw HTML,
     so the <script> tag is what gets it executed in the browser.

     Why JS instead of a Python event handler for copy/download?
     Gradio's Python callbacks round-trip through the server, which is
     wasteful (and sometimes flaky) for something as simple as "copy this
     text the browser already has." Reading straight from the hidden
     textarea's DOM node and using the Clipboard / Blob APIs keeps copy and
     download instant and fully client-side.
     ════════════════════════════════════════════════════════════════════════ -->
<script>
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
</script>
