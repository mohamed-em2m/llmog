"""
Draw & Recognize Reclassification Tab module.
Implements a high-performance Custom HTML5 Canvas Frontend paired with Gradio's Backend,
providing dedicated bounding box (rectangle), freehand brush stroke, circle, and eraser tools,
zoom/pan controls, drag-and-drop / clipboard image loading, undo/redo history,
and multi-mode VLM region recognition (Strict, Hybrid, Free) with YOLO dataset labeling.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import gradio as gr

from detection_viewer import DetectionViewer
from interface.batch.reclassification import (
    classify_regions_gui,
    _RECLS_EMPTY_TABLE,
)

_DEFAULT_CANVAS_IMAGE = Path(__file__).resolve().parents[2] / "assets" / "image.png"

# Preset category libraries for common inspection and detection domains
CATEGORY_PRESETS: Dict[str, Dict[str, str]] = {
    "Fabric & Surface Defects": {
        "classes": "hole, stain, tear, cut, knot, weaving_defect",
        "defs": (
            "- hole: missing fabric or puncture\n"
            "- stain: discoloration or surface contaminant\n"
            "- tear: frayed, uneven physical separation\n"
            "- cut: clean sharp slice or incision\n"
            "- knot: raised thread lump or snarl\n"
            "- weaving_defect: uneven thread density or missing yarn"
        ),
    },
    "General Objects (COCO)": {
        "classes": "person, car, bicycle, dog, cat, chair, bottle, laptop, cell_phone, book",
        "defs": (
            "- person: human body\n"
            "- car: passenger automobile\n"
            "- bicycle: two-wheeled pedal bike\n"
            "- dog: canine domestic animal\n"
            "- cat: feline domestic animal\n"
            "- chair: seating furniture\n"
            "- bottle: liquid beverage container\n"
            "- laptop: portable notebook computer\n"
            "- cell_phone: handheld smartphone\n"
            "- book: bound printed volume"
        ),
    },
    "Road & Traffic": {
        "classes": "car, truck, pedestrian, cyclist, traffic_light, traffic_sign, bus, motorcycle",
        "defs": (
            "- car: passenger sedan, coupe, or SUV\n"
            "- truck: heavy transport or cargo vehicle\n"
            "- pedestrian: person on foot\n"
            "- cyclist: person riding a bicycle\n"
            "- traffic_light: signal light lamp\n"
            "- traffic_sign: road regulatory or warning signboard\n"
            "- bus: public transit passenger bus\n"
            "- motorcycle: motorized two-wheeled vehicle"
        ),
    },
    "Retail & Packaging": {
        "classes": "box, barcode, product_label, bottle, can, pouch, blister_pack",
        "defs": (
            "- box: cardboard or corrugated carton\n"
            "- barcode: 1D or 2D scanner code\n"
            "- product_label: brand packaging label\n"
            "- bottle: glass or plastic container\n"
            "- can: aluminum or tin can\n"
            "- pouch: flexible plastic packaging\n"
            "- blister_pack: clear molded plastic bubble packaging"
        ),
    },
    "PCB & Electronics Defects": {
        "classes": "short_circuit, missing_component, solder_bridge, broken_trace, scratch, misalignment",
        "defs": (
            "- short_circuit: unintended electrical contact\n"
            "- missing_component: empty pad where SMD/component should be\n"
            "- solder_bridge: solder connecting adjacent pins\n"
            "- broken_trace: severed copper circuit trace\n"
            "- scratch: surface gouge across the solder mask\n"
            "- misalignment: component rotated or shifted off pad"
        ),
    },
    "Custom / Blank": {
        "classes": "",
        "defs": "",
    },
}


def _get_sample_image_b64() -> str:
    """Return the demo sample asset as a base64 JPEG/PNG data URI string."""
    if _DEFAULT_CANVAS_IMAGE.is_file():
        try:
            with open(_DEFAULT_CANVAS_IMAGE, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{data}"
        except Exception:
            pass
    return ""


def _clear_draw_results():
    """Reset recognition status, viewer, and YOLO textbox."""
    return (
        "**Status: Idle**",
        None,
        _RECLS_EMPTY_TABLE,
        "",
    )


def _check_draw_endpoint(
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    server_port: float | int | None,
) -> str:
    """Lightweight endpoint check for Draw tab – mirrors Batch/Realtime logic."""
    try:
        from interface.realtime.state import resolve_endpoint
        from openai import OpenAI

        base_url, api_key, model_name = resolve_endpoint(
            int(server_port) if server_port else 8080,
            bool(use_external_api),
            ext_api_url or "",
            ext_api_key or "",
            ext_model_name or "",
        )
        # External requires key, local requires healthy manager
        if use_external_api:
            if not ext_api_key or ext_api_key.strip() in ("", "your-key"):
                return "**Status: ⚠️ External API selected but no API key set – configure in 🧠 Model / Endpoint tab.**"
            # Light ping – list models or fail fast
            client = OpenAI(base_url=base_url, api_key=api_key)
            try:
                client.models.list()
                return f"**Status: ✅ Connected to External API `{model_name}` at `{base_url}`**"
            except Exception as e:
                return f"**Status: ⚠️ External API reachable but ping failed: {e} – check URL/key.**"
        else:
            from interface.state import state

            with state.server_lock:
                mgr = state.server_manager
                if mgr is None:
                    return "**Status: ❌ Local server not running – start it in 🧠 Model / Endpoint tab.**"
                if not mgr.is_healthy():
                    return "**Status: ⏳ Local server starting – check logs in 🧠 Model / Endpoint tab.**"
                return f"**Status: ✅ Local server healthy on port {mgr.port} (model `{mgr.model}`)**"
    except Exception as e:
        return f"**Status: ❌ Connection check failed: {e}**"


def _on_preset_change(preset_name: str) -> Tuple[gr.update, gr.update]:
    """Populate classes and definitions when a domain preset is chosen."""
    preset = CATEGORY_PRESETS.get(preset_name, CATEGORY_PRESETS["Custom / Blank"])
    return (
        gr.update(value=preset["classes"]),
        gr.update(value=preset["defs"]),
    )


def _on_class_mode_change(mode: str) -> Tuple[gr.update, gr.update, gr.update]:
    """Dynamically update labels and placeholders when switching classification mode."""
    if mode == "free":
        return (
            gr.update(
                label="Domain / Focus Hint (Optional)",
                placeholder="e.g. Focus on industrial defects, wildlife, electronics... (or leave blank)",
                info="Free Mode: Agent autonomously names any object/defect. Predefined classes are not required.",
            ),
            gr.update(
                label="Domain Guidance / Prompt Context (Optional)",
                placeholder="Optional domain context or special inspection criteria...",
                info="Optional domain guidance.",
            ),
            gr.update(visible=False),
        )
    elif mode == "hybrid":
        return (
            gr.update(
                label="Priority Target Classes (comma-separated)",
                placeholder="e.g. hole, stain, tear, cut",
                info="Hybrid Mode: Agent prioritizes these classes, but can discover and name new classes if detected.",
            ),
            gr.update(
                label="Category Definitions & Novel Discovery Criteria",
                placeholder="Definitions for priority classes...",
                info="Definitions for priority classes.",
            ),
            gr.update(visible=True),
        )
    else:  # strict
        return (
            gr.update(
                label="Target Classes (Strict - Comma Separated)",
                placeholder="hole, stain, tear, cut, knot, weaving_defect",
                info="Strict Mode: Agent is locked to these classes (or 'none').",
            ),
            gr.update(
                label="Class Definitions / Distinguishing Rules",
                placeholder="Detailed criteria for distinguishing each class...",
                info="Criteria for distinguishing each class.",
            ),
            gr.update(visible=True),
        )


# ── Custom Canvas Frontend HTML Template ───────────────────────────────────────
_CUSTOM_CANVAS_HTML = """
<div id="llmog-custom-canvas-app" class="custom-canvas-container">
    <!-- Top Interactive Toolbar -->
    <div class="canvas-toolbar">
        <div class="canvas-tool-group">
            <span class="tool-group-label">Tools</span>
            <button type="button" class="canvas-tool-btn active" id="tool-bbox" title="Bounding Box (Drag rectangle) [B]">
                <span class="tool-icon">🔲</span> Box
            </button>
            <button type="button" class="canvas-tool-btn" id="tool-brush" title="Freehand Brush [P]">
                <span class="tool-icon">🖌️</span> Brush
            </button>
            <button type="button" class="canvas-tool-btn" id="tool-circle" title="Circle / Ellipse [C]">
                <span class="tool-icon">⭕</span> Circle
            </button>
            <button type="button" class="canvas-tool-btn" id="tool-eraser" title="Eraser / Delete [E]">
                <span class="tool-icon">🧽</span> Eraser
            </button>
        </div>

        <div class="canvas-toolbar-divider"></div>

        <div class="canvas-tool-group">
            <span class="tool-group-label">Color</span>
            <div class="color-palette-bar" id="palette-swatches">
                <button type="button" class="color-swatch active" style="background:#ff3c3c" data-color="#ff3c3c"></button>
                <button type="button" class="color-swatch" style="background:#0096ff" data-color="#0096ff"></button>
                <button type="button" class="color-swatch" style="background:#00d250" data-color="#00d250"></button>
                <button type="button" class="color-swatch" style="background:#ffd214" data-color="#ffd214"></button>
                <button type="button" class="color-swatch" style="background:#ffa014" data-color="#ffa014"></button>
                <button type="button" class="color-swatch" style="background:#963cff" data-color="#963cff"></button>
                <button type="button" class="color-swatch" style="background:#00d7d7" data-color="#00d7d7"></button>
                <button type="button" class="color-swatch" style="background:#ffffff" data-color="#ffffff"></button>
            </div>
            <input type="color" id="custom-color-picker" value="#ff3c3c" title="Custom color" class="color-picker-input">
        </div>

        <div class="canvas-toolbar-divider"></div>

        <div class="canvas-tool-group">
            <span class="tool-group-label">Size: <b id="brush-size-val">3</b>px</span>
            <input type="range" id="brush-size-slider" min="1" max="40" value="3" class="canvas-range-slider" title="Stroke thickness">
        </div>

        <div class="canvas-toolbar-divider"></div>

        <div class="canvas-tool-group">
            <span class="tool-group-label">Actions</span>
            <button type="button" class="canvas-tool-btn" id="btn-undo" title="Undo [Ctrl+Z]">↩️ Undo</button>
            <button type="button" class="canvas-tool-btn" id="btn-redo" title="Redo [Ctrl+Y]">🔁 Redo</button>
            <button type="button" class="canvas-tool-btn" id="btn-clear-drawings" title="Clear drawn boxes & strokes only">🧽 Clear Drawings</button>
            <button type="button" class="canvas-tool-btn danger" id="btn-clear-all" title="Clear image & drawings completely">🧹 Reset All</button>
        </div>

        <div class="canvas-toolbar-divider"></div>

        <div class="canvas-tool-group">
            <span class="tool-group-label">Zoom</span>
            <button type="button" class="canvas-tool-btn icon-only" id="btn-zoom-in" title="Zoom in">➕</button>
            <button type="button" class="canvas-tool-btn icon-only" id="btn-zoom-out" title="Zoom out">➖</button>
            <button type="button" class="canvas-tool-btn" id="btn-zoom-fit" title="Fit to viewport">📐 Fit</button>
            <span id="zoom-level-text" class="zoom-indicator">100%</span>
        </div>

        <div class="canvas-toolbar-divider"></div>

        <div class="canvas-tool-group">
            <span class="tool-group-label">Image</span>
            <button type="button" class="canvas-tool-btn upload-btn" id="btn-toolbar-upload" title="Upload an image from your computer">
                📁 Upload Image
            </button>
        </div>
    </div>

    <!-- Canvas Stage Viewport -->
    <div class="canvas-stage-wrapper" id="canvas-stage-wrapper">
        <canvas id="custom-annotation-canvas"></canvas>
        
        <div id="canvas-empty-overlay" class="canvas-empty-state">
            <div class="empty-icon">🎨</div>
            <h3>Interactive Detection Canvas</h3>
            <p>Upload an image or load the sample, then draw bounding boxes or strokes over objects.</p>
            <div class="empty-actions">
                <button type="button" class="btn-canvas-primary" id="btn-empty-upload">📁 Choose Image</button>
                <button type="button" class="btn-canvas-secondary" id="btn-empty-sample">🖼️ Load Sample</button>
            </div>
            <span class="drag-hint">or drag &amp; drop an image here / paste from clipboard (Ctrl+V)</span>
        </div>

        <input type="file" id="canvas-file-input" accept="image/*" style="display:none">
    </div>

    <!-- Regions Live Summary Bar -->
    <div class="canvas-status-bar">
        <div class="status-left">
            <span class="regions-count-badge" id="regions-count-badge">0 Region(s)</span>
            <span class="canvas-hint-text">💡 Tip: Select <b>Box</b> to drag bounding boxes, or <b>Brush</b> for arbitrary strokes. Mouse wheel zooms.</span>
        </div>
        <div class="regions-list-chips" id="regions-chips-container">
            <!-- Dynamically populated region chips -->
        </div>
    </div>
</div>
"""

# ── Custom Canvas Frontend JavaScript Controller ──────────────────────────────
_CUSTOM_CANVAS_JS = """
(function() {
    window.CustomCanvasController = {
        image: null,
        imageSrc: null,
        imageWidth: 0,
        imageHeight: 0,
        
        mode: 'bbox',
        color: '#ff3c3c',
        size: 3,
        
        scale: 1.0,
        offsetX: 0,
        offsetY: 0,
        isPanning: false,
        panStartX: 0,
        panStartY: 0,
        
        isDrawing: false,
        startX: 0,
        startY: 0,
        currentX: 0,
        currentY: 0,
        
        currentStroke: [],
        regions: [],
        undoStack: [],
        redoStack: [],
        
        container: null,
        canvas: null,
        ctx: null,
        wrapper: null,
        emptyOverlay: null,
        fileInput: null,
        
        init: function() {
            this.container = document.getElementById('llmog-custom-canvas-app');
            if (!this.container) return;
            
            this.canvas = document.getElementById('custom-annotation-canvas');
            if (!this.canvas) return;
            this.ctx = this.canvas.getContext('2d');
            this.wrapper = document.getElementById('canvas-stage-wrapper');
            this.emptyOverlay = document.getElementById('canvas-empty-overlay');
            this.fileInput = document.getElementById('canvas-file-input');
            
            this.bindEvents();
            this.resizeCanvas();
            this.syncGradioPayload();
        },
        
        bindEvents: function() {
            const self = this;
            window.addEventListener('resize', () => self.resizeCanvas());
            
            const toolBbox = document.getElementById('tool-bbox');
            const toolBrush = document.getElementById('tool-brush');
            const toolCircle = document.getElementById('tool-circle');
            const toolEraser = document.getElementById('tool-eraser');
            
            const setTool = (mode, btn) => {
                self.mode = mode;
                [toolBbox, toolBrush, toolCircle, toolEraser].forEach(b => b && b.classList.remove('active'));
                if (btn) btn.classList.add('active');
                if (mode === 'brush' && self.size < 6) {
                    self.size = 10;
                    const sl = document.getElementById('brush-size-slider');
                    const sv = document.getElementById('brush-size-val');
                    if (sl) sl.value = 10;
                    if (sv) sv.textContent = 10;
                } else if (mode === 'bbox' && self.size > 8) {
                    self.size = 3;
                    const sl = document.getElementById('brush-size-slider');
                    const sv = document.getElementById('brush-size-val');
                    if (sl) sl.value = 3;
                    if (sv) sv.textContent = 3;
                }
                self.render();
            };
            
            if (toolBbox) toolBbox.onclick = () => setTool('bbox', toolBbox);
            if (toolBrush) toolBrush.onclick = () => setTool('brush', toolBrush);
            if (toolCircle) toolCircle.onclick = () => setTool('circle', toolCircle);
            if (toolEraser) toolEraser.onclick = () => setTool('eraser', toolEraser);
            
            const swatches = document.querySelectorAll('#palette-swatches .color-swatch');
            const customPicker = document.getElementById('custom-color-picker');
            
            swatches.forEach(sw => {
                sw.onclick = () => {
                    swatches.forEach(s => s.classList.remove('active'));
                    sw.classList.add('active');
                    self.color = sw.dataset.color;
                    if (customPicker) customPicker.value = self.color;
                };
            });
            
            if (customPicker) {
                customPicker.oninput = (e) => {
                    self.color = e.target.value;
                    swatches.forEach(s => s.classList.remove('active'));
                };
            }
            
            const sizeSlider = document.getElementById('brush-size-slider');
            const sizeVal = document.getElementById('brush-size-val');
            if (sizeSlider) {
                sizeSlider.oninput = (e) => {
                    self.size = parseInt(e.target.value, 10);
                    if (sizeVal) sizeVal.textContent = self.size;
                };
            }
            
            const btnUndo = document.getElementById('btn-undo');
            const btnRedo = document.getElementById('btn-redo');
            const btnClearDrawings = document.getElementById('btn-clear-drawings');
            const btnClearAll = document.getElementById('btn-clear-all');
            
            if (btnUndo) btnUndo.onclick = () => self.undo();
            if (btnRedo) btnRedo.onclick = () => self.redo();
            if (btnClearDrawings) btnClearDrawings.onclick = () => self.clearDrawings();
            if (btnClearAll) btnClearAll.onclick = () => self.clearAll();
            
            const btnZoomIn = document.getElementById('btn-zoom-in');
            const btnZoomOut = document.getElementById('btn-zoom-out');
            const btnZoomFit = document.getElementById('btn-zoom-fit');
            
            if (btnZoomIn) btnZoomIn.onclick = () => self.zoom(1.2);
            if (btnZoomOut) btnZoomOut.onclick = () => self.zoom(1 / 1.2);
            if (btnZoomFit) btnZoomFit.onclick = () => self.fitToScreen();
            
            // Toolbar upload button — always visible
            const btnToolbarUpload = document.getElementById('btn-toolbar-upload');
            if (btnToolbarUpload) {
                btnToolbarUpload.onclick = () => self.triggerFileUpload();
            }

            const btnEmptyUpload = document.getElementById('btn-empty-upload');
            const btnEmptySample = document.getElementById('btn-empty-sample');
            
            if (btnEmptyUpload) {
                btnEmptyUpload.onclick = () => self.triggerFileUpload();
            }
            if (btnEmptySample) {
                btnEmptySample.onclick = () => self.loadSampleImage();
            }

            // Wire the static hidden file input as a fallback
            if (self.fileInput) {
                self.fileInput.onchange = (e) => {
                    const file = e.target.files && e.target.files[0];
                    if (file) self.loadImageFromFile(file);
                    // Reset so the same file can be re-selected
                    self.fileInput.value = '';
                };
            }
            
            self.canvas.addEventListener('mousedown', (e) => self.onPointerDown(e));
            window.addEventListener('mousemove', (e) => self.onPointerMove(e));
            window.addEventListener('mouseup', (e) => self.onPointerUp(e));
            
            self.canvas.addEventListener('touchstart', (e) => {
                if (e.touches.length === 1) {
                    const touch = e.touches[0];
                    self.onPointerDown({ clientX: touch.clientX, clientY: touch.clientY, button: 0, preventDefault: () => e.preventDefault() });
                }
            }, { passive: false });
            
            window.addEventListener('touchmove', (e) => {
                if (self.isDrawing || self.isPanning) {
                    if (e.touches.length === 1) {
                        const touch = e.touches[0];
                        self.onPointerMove({ clientX: touch.clientX, clientY: touch.clientY });
                    }
                }
            }, { passive: false });
            
            window.addEventListener('touchend', (e) => {
                if (self.isDrawing || self.isPanning) {
                    self.onPointerUp(e);
                }
            });
            
            self.wrapper.addEventListener('wheel', (e) => {
                e.preventDefault();
                const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
                const rect = self.canvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;
                self.zoomAt(zoomFactor, mouseX, mouseY);
            }, { passive: false });
            
            ['dragenter', 'dragover'].forEach(name => {
                self.wrapper.addEventListener(name, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    self.wrapper.classList.add('drag-active');
                });
            });
            ['dragleave', 'drop'].forEach(name => {
                self.wrapper.addEventListener(name, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    self.wrapper.classList.remove('drag-active');
                });
            });
            self.wrapper.addEventListener('drop', (e) => {
                const files = e.dataTransfer && e.dataTransfer.files;
                if (files && files.length > 0 && files[0].type.startsWith('image/')) {
                    self.loadImageFromFile(files[0]);
                }
            });
            
            window.addEventListener('paste', (e) => {
                const items = (e.clipboardData || e.originalEvent.clipboardData).items;
                for (let i = 0; i < items.length; i++) {
                    if (items[i].type.indexOf('image') !== -1) {
                        const blob = items[i].getAsFile();
                        self.loadImageFromFile(blob);
                        break;
                    }
                }
            });
            
            window.addEventListener('keydown', (e) => {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
                
                if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
                    e.preventDefault();
                    if (e.shiftKey) self.redo();
                    else self.undo();
                } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Y')) {
                    e.preventDefault();
                    self.redo();
                } else if (e.key === 'b' || e.key === 'B') {
                    setTool('bbox', toolBbox);
                } else if (e.key === 'p' || e.key === 'P') {
                    setTool('brush', toolBrush);
                } else if (e.key === 'c' || e.key === 'C') {
                    setTool('circle', toolCircle);
                } else if (e.key === 'e' || e.key === 'E') {
                    setTool('eraser', toolEraser);
                }
            });
        },
        
        resizeCanvas: function() {
            if (!this.wrapper || !this.canvas) return;
            const w = this.wrapper.clientWidth;
            const h = this.wrapper.clientHeight || 580;
            this.canvas.width = w;
            this.canvas.height = h;
            this.render();
        },
        
        loadImageFromDataUrl: function(dataUrl) {
            const self = this;
            const img = new Image();
            img.onload = function() {
                self.image = img;
                self.imageSrc = dataUrl;
                self.imageWidth = img.naturalWidth || img.width;
                self.imageHeight = img.naturalHeight || img.height;
                self.regions = [];
                self.undoStack = [];
                self.redoStack = [];
                
                if (self.emptyOverlay) self.emptyOverlay.style.display = 'none';
                self.fitToScreen();
                self.syncGradioPayload();
            };
            img.src = dataUrl;
        },
        
        loadImageFromFile: function(file) {
            const self = this;
            const reader = new FileReader();
            reader.onload = function(e) {
                self.loadImageFromDataUrl(e.target.result);
            };
            reader.readAsDataURL(file);
        },
        
        triggerFileUpload: function() {
            // Create a fresh <input type="file"> every time.
            // This bypasses the Gradio innerHTML DOM issue where getElementById
            // may return null for elements injected via gr.HTML's innerHTML path.
            const self = this;
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.style.display = 'none';
            document.body.appendChild(input);
            input.onchange = function(e) {
                const file = e.target.files && e.target.files[0];
                if (file) self.loadImageFromFile(file);
                document.body.removeChild(input);
            };
            // Ensure the dialog is removed if the user cancels (focus returns)
            input.addEventListener('cancel', function() {
                try { document.body.removeChild(input); } catch(_) {}
            });
            window.addEventListener('focus', function cleanup() {
                setTimeout(function() {
                    try { if (input.parentNode) document.body.removeChild(input); } catch(_) {}
                    window.removeEventListener('focus', cleanup);
                }, 500);
            }, { once: true });
            input.click();
        },
        
        loadSampleImage: function() {
            const sampleBtn = document.getElementById('recls_sample_bridge_btn');
            if (sampleBtn) {
                sampleBtn.click();
            }
        },
        
        fitToScreen: function() {
            if (!this.image || !this.canvas) return;
            const cw = this.canvas.width;
            const ch = this.canvas.height;
            const iw = this.imageWidth;
            const ih = this.imageHeight;
            
            const scaleX = (cw - 40) / iw;
            const scaleY = (ch - 40) / ih;
            this.scale = Math.min(scaleX, scaleY, 1.0);
            if (this.scale <= 0) this.scale = 1.0;
            
            this.offsetX = (cw - iw * this.scale) / 2;
            this.offsetY = (ch - ih * this.scale) / 2;
            
            this.updateZoomIndicator();
            this.render();
        },
        
        zoom: function(factor) {
            const cw = this.canvas.width / 2;
            const ch = this.canvas.height / 2;
            this.zoomAt(factor, cw, ch);
        },
        
        zoomAt: function(factor, mouseX, mouseY) {
            const prevScale = this.scale;
            let newScale = this.scale * factor;
            newScale = Math.max(0.1, Math.min(newScale, 15.0));
            
            this.offsetX = mouseX - (mouseX - this.offsetX) * (newScale / prevScale);
            this.offsetY = mouseY - (mouseY - this.offsetY) * (newScale / prevScale);
            this.scale = newScale;
            
            this.updateZoomIndicator();
            this.render();
        },
        
        updateZoomIndicator: function() {
            const ind = document.getElementById('zoom-level-text');
            if (ind) ind.textContent = `${Math.round(this.scale * 100)}%`;
        },
        
        screenToImageCoords: function(screenX, screenY) {
            const imgX = (screenX - this.offsetX) / this.scale;
            const imgY = (screenY - this.offsetY) / this.scale;
            return {
                x: Math.max(0, Math.min(this.imageWidth, imgX)),
                y: Math.max(0, Math.min(this.imageHeight, imgY))
            };
        },
        
        onPointerDown: function(e) {
            if (!this.image) return;
            const rect = this.canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            if (e.button === 1 || e.spaceKey || (e.button === 0 && e.altKey)) {
                this.isPanning = true;
                this.panStartX = mouseX - this.offsetX;
                this.panStartY = mouseY - this.offsetY;
                return;
            }
            
            if (e.button !== 0) return;
            
            const pt = this.screenToImageCoords(mouseX, mouseY);
            
            if (this.mode === 'eraser') {
                this.eraseAt(pt.x, pt.y);
                return;
            }
            
            this.isDrawing = true;
            this.startX = pt.x;
            this.startY = pt.y;
            this.currentX = pt.x;
            this.currentY = pt.y;
            
            if (this.mode === 'brush') {
                this.currentStroke = [{ x: pt.x, y: pt.y }];
            }
        },
        
        onPointerMove: function(e) {
            if (!this.image) return;
            const rect = this.canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            if (this.isPanning) {
                this.offsetX = mouseX - this.panStartX;
                this.offsetY = mouseY - this.panStartY;
                this.render();
                return;
            }
            
            if (!this.isDrawing) return;
            
            const pt = this.screenToImageCoords(mouseX, mouseY);
            this.currentX = pt.x;
            this.currentY = pt.y;
            
            if (this.mode === 'brush') {
                this.currentStroke.push({ x: pt.x, y: pt.y });
            }
            this.render();
        },
        
        onPointerUp: function(e) {
            if (this.isPanning) {
                this.isPanning = false;
                return;
            }
            
            if (!this.isDrawing) return;
            this.isDrawing = false;
            
            this.saveStateForUndo();
            
            if (this.mode === 'bbox') {
                const x1 = Math.min(this.startX, this.currentX);
                const y1 = Math.min(this.startY, this.currentY);
                const x2 = Math.max(this.startX, this.currentX);
                const y2 = Math.max(this.startY, this.currentY);
                
                if (x2 - x1 >= 6 && y2 - y1 >= 6) {
                    this.regions.push({
                        type: 'bbox',
                        x1: Math.round(x1),
                        y1: Math.round(y1),
                        x2: Math.round(x2),
                        y2: Math.round(y2),
                        color: this.color,
                        size: this.size
                    });
                }
            } else if (this.mode === 'circle') {
                const x1 = Math.min(this.startX, this.currentX);
                const y1 = Math.min(this.startY, this.currentY);
                const x2 = Math.max(this.startX, this.currentX);
                const y2 = Math.max(this.startY, this.currentY);
                
                if (x2 - x1 >= 6 && y2 - y1 >= 6) {
                    this.regions.push({
                        type: 'circle',
                        x1: Math.round(x1),
                        y1: Math.round(y1),
                        x2: Math.round(x2),
                        y2: Math.round(y2),
                        color: this.color,
                        size: this.size
                    });
                }
            } else if (this.mode === 'brush' && this.currentStroke.length > 1) {
                let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
                this.currentStroke.forEach(p => {
                    minX = Math.min(minX, p.x);
                    minY = Math.min(minY, p.y);
                    maxX = Math.max(maxX, p.x);
                    maxY = Math.max(maxY, p.y);
                });
                const pad = this.size;
                this.regions.push({
                    type: 'stroke',
                    x1: Math.round(Math.max(0, minX - pad)),
                    y1: Math.round(Math.max(0, minY - pad)),
                    x2: Math.round(Math.min(this.imageWidth, maxX + pad)),
                    y2: Math.round(Math.min(this.imageHeight, maxY + pad)),
                    points: this.currentStroke,
                    color: this.color,
                    size: this.size
                });
                this.currentStroke = [];
            }
            
            this.updateRegionsList();
            this.syncGradioPayload();
            this.render();
        },
        
        eraseAt: function(imgX, imgY) {
            let hitIndex = -1;
            for (let i = this.regions.length - 1; i >= 0; i--) {
                const r = this.regions[i];
                if (imgX >= r.x1 && imgX <= r.x2 && imgY >= r.y1 && imgY <= r.y2) {
                    hitIndex = i;
                    break;
                }
            }
            if (hitIndex !== -1) {
                this.saveStateForUndo();
                this.regions.splice(hitIndex, 1);
                this.updateRegionsList();
                this.syncGradioPayload();
                this.render();
            }
        },
        
        removeRegion: function(index) {
            if (index >= 0 && index < this.regions.length) {
                this.saveStateForUndo();
                this.regions.splice(index, 1);
                this.updateRegionsList();
                this.syncGradioPayload();
                this.render();
            }
        },
        
        saveStateForUndo: function() {
            this.undoStack.push(JSON.parse(JSON.stringify(this.regions)));
            this.redoStack = [];
            if (this.undoStack.length > 30) this.undoStack.shift();
        },
        
        undo: function() {
            if (this.undoStack.length > 0) {
                this.redoStack.push(JSON.parse(JSON.stringify(this.regions)));
                this.regions = this.undoStack.pop();
                this.updateRegionsList();
                this.syncGradioPayload();
                this.render();
            }
        },
        
        redo: function() {
            if (this.redoStack.length > 0) {
                this.undoStack.push(JSON.parse(JSON.stringify(this.regions)));
                this.regions = this.redoStack.pop();
                this.updateRegionsList();
                this.syncGradioPayload();
                this.render();
            }
        },
        
        clearDrawings: function() {
            if (this.regions.length > 0) {
                this.saveStateForUndo();
                this.regions = [];
                this.updateRegionsList();
                this.syncGradioPayload();
                this.render();
            }
        },
        
        clearAll: function() {
            this.image = null;
            this.imageSrc = null;
            this.regions = [];
            this.undoStack = [];
            this.redoStack = [];
            if (this.emptyOverlay) this.emptyOverlay.style.display = 'flex';
            this.updateRegionsList();
            this.syncGradioPayload();
            this.render();
        },
        
        updateRegionsList: function() {
            const countBadge = document.getElementById('regions-count-badge');
            if (countBadge) countBadge.textContent = `${this.regions.length} Region(s)`;
            
            const container = document.getElementById('regions-chips-container');
            if (!container) return;
            container.innerHTML = '';
            
            this.regions.forEach((r, idx) => {
                const chip = document.createElement('div');
                chip.className = 'canvas-region-chip';
                const w = r.x2 - r.x1;
                const h = r.y2 - r.y1;
                chip.innerHTML = `
                    <span class="chip-color-dot" style="background:${r.color}"></span>
                    <span class="chip-title">#${idx + 1} ${r.type.toUpperCase()}</span>
                    <span class="chip-coords">${w}×${h}</span>
                    <button type="button" class="chip-del-btn" title="Delete region">✕</button>
                `;
                chip.querySelector('.chip-del-btn').onclick = (e) => {
                    e.stopPropagation();
                    window.CustomCanvasController.removeRegion(idx);
                };
                container.appendChild(chip);
            });
        },
        
        render: function() {
            if (!this.ctx || !this.canvas) return;
            const ctx = this.ctx;
            const w = this.canvas.width;
            const h = this.canvas.height;
            
            ctx.clearRect(0, 0, w, h);
            this.drawBackgroundPattern(ctx, w, h);
            
            if (!this.image) return;
            
            ctx.save();
            ctx.translate(this.offsetX, this.offsetY);
            ctx.scale(this.scale, this.scale);
            
            ctx.drawImage(this.image, 0, 0, this.imageWidth, this.imageHeight);
            
            this.regions.forEach((r, idx) => {
                this.drawSingleRegion(ctx, r, idx + 1);
            });
            
            if (this.isDrawing) {
                if (this.mode === 'bbox') {
                    const x1 = Math.min(this.startX, this.currentX);
                    const y1 = Math.min(this.startY, this.currentY);
                    const x2 = Math.max(this.startX, this.currentX);
                    const y2 = Math.max(this.startY, this.currentY);
                    this.drawSingleRegion(ctx, {
                        type: 'bbox',
                        x1, y1, x2, y2,
                        color: this.color,
                        size: this.size
                    }, this.regions.length + 1, true);
                } else if (this.mode === 'circle') {
                    const x1 = Math.min(this.startX, this.currentX);
                    const y1 = Math.min(this.startY, this.currentY);
                    const x2 = Math.max(this.startX, this.currentX);
                    const y2 = Math.max(this.startY, this.currentY);
                    this.drawSingleRegion(ctx, {
                        type: 'circle',
                        x1, y1, x2, y2,
                        color: this.color,
                        size: this.size
                    }, this.regions.length + 1, true);
                } else if (this.mode === 'brush' && this.currentStroke.length > 1) {
                    ctx.save();
                    ctx.strokeStyle = this.color;
                    ctx.lineWidth = this.size;
                    ctx.lineCap = 'round';
                    ctx.lineJoin = 'round';
                    ctx.beginPath();
                    ctx.moveTo(this.currentStroke[0].x, this.currentStroke[0].y);
                    for (let i = 1; i < this.currentStroke.length; i++) {
                        ctx.lineTo(this.currentStroke[i].x, this.currentStroke[i].y);
                    }
                    ctx.stroke();
                    ctx.restore();
                }
            }
            
            ctx.restore();
        },
        
        drawSingleRegion: function(ctx, r, labelNumber, isLive = false) {
            ctx.save();
            const strokeW = Math.max(2, (r.size || 3) / this.scale);
            
            if (r.type === 'bbox') {
                const rx = r.x1;
                const ry = r.y1;
                const rw = r.x2 - r.x1;
                const rh = r.y2 - r.y1;
                
                ctx.fillStyle = isLive ? 'rgba(255, 60, 60, 0.18)' : 'rgba(56, 189, 248, 0.12)';
                ctx.fillRect(rx, ry, rw, rh);
                
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = strokeW + (2 / this.scale);
                ctx.strokeRect(rx, ry, rw, rh);
                
                ctx.strokeStyle = r.color;
                ctx.lineWidth = strokeW;
                ctx.strokeRect(rx, ry, rw, rh);
                
                this.drawRegionBadge(ctx, rx, ry, `#${labelNumber}`, r.color);
            } else if (r.type === 'circle') {
                const cx = (r.x1 + r.x2) / 2;
                const cy = (r.y1 + r.y2) / 2;
                const rx = (r.x2 - r.x1) / 2;
                const ry = (r.y2 - r.y1) / 2;
                
                ctx.fillStyle = isLive ? 'rgba(255, 60, 60, 0.18)' : 'rgba(56, 189, 248, 0.12)';
                ctx.beginPath();
                ctx.ellipse(cx, cy, Math.abs(rx), Math.abs(ry), 0, 0, Math.PI * 2);
                ctx.fill();
                
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = strokeW + (2 / this.scale);
                ctx.beginPath();
                ctx.ellipse(cx, cy, Math.abs(rx), Math.abs(ry), 0, 0, Math.PI * 2);
                ctx.stroke();
                
                ctx.strokeStyle = r.color;
                ctx.lineWidth = strokeW;
                ctx.beginPath();
                ctx.ellipse(cx, cy, Math.abs(rx), Math.abs(ry), 0, 0, Math.PI * 2);
                ctx.stroke();
                
                this.drawRegionBadge(ctx, r.x1, r.y1, `#${labelNumber}`, r.color);
            } else if (r.type === 'stroke' && r.points && r.points.length > 0) {
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = (r.size || 10) + (3 / this.scale);
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.beginPath();
                ctx.moveTo(r.points[0].x, r.points[0].y);
                for (let i = 1; i < r.points.length; i++) {
                    ctx.lineTo(r.points[i].x, r.points[i].y);
                }
                ctx.stroke();
                
                ctx.strokeStyle = r.color;
                ctx.lineWidth = r.size || 10;
                ctx.beginPath();
                ctx.moveTo(r.points[0].x, r.points[0].y);
                for (let i = 1; i < r.points.length; i++) {
                    ctx.lineTo(r.points[i].x, r.points[i].y);
                }
                ctx.stroke();
                
                this.drawRegionBadge(ctx, r.x1, r.y1, `#${labelNumber}`, r.color);
            }
            ctx.restore();
        },
        
        drawRegionBadge: function(ctx, x, y, text, color) {
            const fontSize = Math.max(12, Math.round(14 / this.scale));
            ctx.font = `bold ${fontSize}px "JetBrains Mono", monospace`;
            const textWidth = ctx.measureText(text).width;
            const pad = 4 / this.scale;
            const badgeW = textWidth + pad * 2;
            const badgeH = fontSize + pad * 2;
            
            const badgeY = Math.max(0, y - badgeH);
            
            ctx.fillStyle = '#080d14';
            ctx.fillRect(x, badgeY, badgeW, badgeH);
            
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5 / this.scale;
            ctx.strokeRect(x, badgeY, badgeW, badgeH);
            
            ctx.fillStyle = '#ffffff';
            ctx.fillText(text, x + pad, badgeY + badgeH - pad * 1.5);
        },
        
        drawBackgroundPattern: function(ctx, w, h) {
            ctx.fillStyle = '#0a0e17';
            ctx.fillRect(0, 0, w, h);
            
            ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
            const step = 24;
            for (let x = 0; x < w; x += step) {
                for (let y = 0; y < h; y += step) {
                    ctx.fillRect(x, y, 2, 2);
                }
            }
        },
        
        syncGradioPayload: function() {
            const payload = {
                background: this.imageSrc || null,
                regions: this.regions.map(r => ({
                    x1: r.x1,
                    y1: r.y1,
                    x2: r.x2,
                    y2: r.y2,
                    type: r.type,
                    color: r.color
                })),
                layers: [],
                composite: null
            };
            
            const jsonStr = JSON.stringify(payload);
            window.__llmog_custom_canvas_payload__ = jsonStr;
            
            const hiddenTa = document.querySelector('#custom_draw_payload_box textarea');
            if (hiddenTa) {
                hiddenTa.value = jsonStr;
                hiddenTa.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }
    };
    
    window.getCustomDrawData = function() {
        if (window.CustomCanvasController) {
            window.CustomCanvasController.syncGradioPayload();
        }
        return window.__llmog_custom_canvas_payload__ || "{}";
    };
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => window.CustomCanvasController.init());
    } else {
        setTimeout(() => window.CustomCanvasController.init(), 100);
    }
})();
"""


def _load_sample_bridge():
    """Server-side handler to deliver the sample image to the Custom Frontend."""
    b64 = _get_sample_image_b64()
    if not b64:
        return ""
    payload = {
        "background": b64,
        "regions": [],
        "layers": [],
        "composite": None,
    }
    return json.dumps(payload)


def build_draw_tab() -> Dict[str, Any]:
    """Build the dedicated Draw & Recognize tab with Custom Frontend Canvas + DetectionViewer."""
    with gr.Row(equal_height=False, elem_classes=["draw-tab-row"]):
        # ── Left / Main: Custom Interactive HTML5 Canvas Frontend ────────────
        with gr.Column(scale=3, min_width=520):
            gr.HTML('<p class="section-label">🎨 Interactive Annotation Canvas</p>')

            # Embedded Custom Canvas Frontend — Gradio 4/5 compat: inject JS via <script> (js_on_load not supported in this version)
            custom_canvas_view = gr.HTML(
                value=_CUSTOM_CANVAS_HTML + f"<script>{_CUSTOM_CANVAS_JS}</script>",
                elem_id="draw-canvas-html",
            )

            # Hidden payload and sample bridge components
            custom_draw_payload = gr.Textbox(
                value="{}",
                visible=False,
                elem_id="custom_draw_payload_box",
            )
            recls_sample_bridge_btn = gr.Button(
                "Sample Bridge",
                visible=False,
                elem_id="recls_sample_bridge_btn",
            )

            with gr.Row(elem_classes=["btn-group"]):
                recls_connect_btn = gr.Button(
                    "🔌 Check Connection",
                    variant="secondary",
                    scale=1,
                    elem_id="draw-connect-btn",
                )
                recls_run_btn = gr.Button(
                    "🔎  Recognize Drawn Regions",
                    variant="primary",
                    scale=2,
                    interactive=True,
                )
                recls_clear_btn = gr.Button(
                    "🗑️ Clear Results",
                    variant="secondary",
                    scale=1,
                )

        # ── Right: Config & Recognition Results ───────────────────────────
        with gr.Column(scale=2, min_width=380, elem_classes=["draw-right-panel"]):
            gr.HTML('<p class="section-label">⚙️ Detection Strategy &amp; Classes</p>')

            # ── 1. Class Expectation Strategy ─────────────────────────────
            recls_class_mode = gr.Radio(
                label="🎯 Class Expectation Mode",
                choices=[
                    ("🔒 Strict (Closed-Set)", "strict"),
                    ("🔀 Hybrid (Extendable)", "hybrid"),
                    ("🌐 Free (Open-World)", "free"),
                ],
                value="strict",
                info="Control how the VLM assigns classes to your drawn regions.",
            )

            # ── 2. Preset library ──────────────────────────────────────────
            recls_preset_dropdown = gr.Dropdown(
                label="📋 Category Domain Presets",
                choices=list(CATEGORY_PRESETS.keys()),
                value="Fabric & Surface Defects",
                info="Quickly load target classes & expert distinguishing definitions.",
            )

            recls_classes_input = gr.Textbox(
                label="Target Classes (Strict - Comma Separated)",
                placeholder="hole, stain, tear, cut, knot, weaving_defect",
                value="hole, stain, tear, cut, knot, weaving_defect",
                lines=2,
                info="Strict Mode: Agent is locked to these classes (or 'none').",
            )

            recls_defs_input = gr.Textbox(
                label="Class Definitions / Distinguishing Rules",
                lines=4,
                value=CATEGORY_PRESETS["Fabric & Surface Defects"]["defs"],
                info="Detailed criteria for distinguishing each class.",
            )

            with gr.Accordion("⚙️ Advanced Filter & Context Settings", open=False):
                recls_conf_threshold = gr.Slider(
                    label="Minimum Confidence Threshold (%)",
                    minimum=0,
                    maximum=100,
                    step=5,
                    value=20,
                    info="Omit or flag recognitions with confidence below this threshold in YOLO outputs.",
                )

                recls_padding_slider = gr.Slider(
                    label="Region Context Padding (%)",
                    minimum=0,
                    maximum=50,
                    step=1,
                    value=10,
                    info="Extra visual context around each drawn region sent to the VLM.",
                )

                recls_request_mode = gr.Radio(
                    label="⚡ Request Mode (optional)",
                    choices=[
                        ("Sequential – 1 request per region", "sequential"),
                        ("Parallel – asyncio.gather concurrent", "parallel"),
                        ("Batched – single request with N images", "batched"),
                    ],
                    value="parallel",
                    info="Sequential: simple. Parallel: N concurrent via asyncio.gather (~N× faster). Batched: 1 request with N images (fewest round-trips).",
                )

            recls_status = gr.Markdown("**Status: Idle**")
            with gr.Group(elem_classes=["img-viewer-wrap"]):
                recls_annotated = DetectionViewer(
                    label="Annotated Recognition Result",
                    panel_title="Recognized Regions",
                    list_height=340,
                    elem_id="draw-detection-viewer",
                )

            recls_results = gr.HTML(value=_RECLS_EMPTY_TABLE)

            with gr.Accordion("YOLO Labels (<class_id> <xc> <yc> <w> <h>)", open=False):
                recls_yolo = gr.Textbox(
                    lines=8,
                    interactive=False,
                    label="Copy these lines into the image's .txt label file",
                )

    return dict(
        custom_canvas_view=custom_canvas_view,
        custom_draw_payload=custom_draw_payload,
        recls_sample_bridge_btn=recls_sample_bridge_btn,
        recls_class_mode=recls_class_mode,
        recls_preset_dropdown=recls_preset_dropdown,
        recls_classes_input=recls_classes_input,
        recls_defs_input=recls_defs_input,
        recls_conf_threshold=recls_conf_threshold,
        recls_padding_slider=recls_padding_slider,
        recls_request_mode=recls_request_mode,
        recls_connect_btn=recls_connect_btn,
        recls_run_btn=recls_run_btn,
        recls_clear_btn=recls_clear_btn,
        recls_status=recls_status,
        recls_annotated=recls_annotated,
        recls_results=recls_results,
        recls_yolo=recls_yolo,
    )


def wire_draw_events(
    c_draw: Dict[str, Any], c_srv: Dict[str, Any], c_bat: Dict[str, Any]
) -> None:
    """Wire interactive recognition, preset switching, and upload events for the Draw & Recognize tab."""
    # ── Category preset change ──────────────────────────────────────────────
    if "recls_preset_dropdown" in c_draw:
        c_draw["recls_preset_dropdown"].change(
            fn=_on_preset_change,
            inputs=[c_draw["recls_preset_dropdown"]],
            outputs=[c_draw["recls_classes_input"], c_draw["recls_defs_input"]],
        )

    # ── Class expectation mode change ───────────────────────────────────────
    if "recls_class_mode" in c_draw:
        c_draw["recls_class_mode"].change(
            fn=_on_class_mode_change,
            inputs=[c_draw["recls_class_mode"]],
            outputs=[
                c_draw["recls_classes_input"],
                c_draw["recls_defs_input"],
                c_draw["recls_preset_dropdown"],
            ],
        )

    # ── Sample image bridge handler ─────────────────────────────────────────
    if "recls_sample_bridge_btn" in c_draw:
        c_draw["recls_sample_bridge_btn"].click(
            fn=_load_sample_bridge,
            inputs=None,
            outputs=[c_draw["custom_draw_payload"]],
            js="() => { if (window.CustomCanvasController) { setTimeout(() => { const ta = document.querySelector('#custom_draw_payload_box textarea'); if (ta && ta.value) { try { const p = JSON.parse(ta.value); if (p.background) window.CustomCanvasController.loadImageFromDataUrl(p.background); } catch(e){} } }, 300); } }",
        )

    # ── Connect check – lets user verify global endpoint before running ─
    # Also auto-switches to 🧠 Model / Endpoint tab if check fails
    if "recls_connect_btn" in c_draw:
        c_draw["recls_connect_btn"].click(
            fn=_check_draw_endpoint,
            inputs=[
                c_srv["use_external_api_chk"],
                c_srv["ext_api_url"],
                c_srv["ext_api_key"],
                c_srv["ext_model_name"],
                c_srv["server_port_input"],
            ],
            outputs=[c_draw["recls_status"]],
        ).then(
            fn=None,
            inputs=[c_draw["recls_status"]],
            outputs=None,
            js="""(status) => {
                const s = document.querySelector('#draw-detection-viewer');
                if(s) s.scrollIntoView({behavior:'smooth', block:'center'});
                const msg = String(status||'');
                if (msg.includes('❌') || msg.includes('⚠️') || msg.includes('not running') || msg.includes('no API key')) {
                    const btns=[...document.querySelectorAll('.tab-nav button')];
                    const t=btns.find(b=>b.textContent.includes('Model / Endpoint'));
                    if(t){ 
                        t.click();
                        t.style.outline='2px solid #38bdf8'; 
                        setTimeout(()=>t.style.outline='', 2500); 
                    }
                }
            }""",
        )

    # ── Recognition execution with Gradio Backend ───────────────────────────
    # Endpoint is now global from Model / Endpoint tab (c_srv), not per-batch.
    # JS must preserve all inputs – previously `()=>[payload]` dropped sliders → None error.
    # Optional request_mode (sequential/parallel/batched) is now exposed.
    c_draw["recls_run_btn"].click(
        fn=classify_regions_gui,
        inputs=[
            c_draw["custom_draw_payload"],
            c_draw["recls_classes_input"],
            c_draw["recls_defs_input"],
            c_draw["recls_padding_slider"],
            c_draw["recls_class_mode"],
            c_draw["recls_conf_threshold"],
            c_srv["use_external_api_chk"],
            c_srv["ext_api_url"],
            c_srv["ext_api_key"],
            c_srv["ext_model_name"],
            c_srv["server_port_input"],
            c_draw["recls_request_mode"],
        ],
        outputs=[
            c_draw["recls_status"],
            c_draw["recls_annotated"],
            c_draw["recls_results"],
            c_draw["recls_yolo"],
        ],
        js="(p,c,d,pad,mode,conf,useExt,url,key,model,port,reqMode)=>{ const fresh=(window.getCustomDrawData?window.getCustomDrawData():p)||p; return [fresh,c,d,pad,mode,conf,useExt,url,key,model,port,reqMode]; }",
        api_name="classify_regions",
        concurrency_limit=1,
    )

    # ── Clear results ───────────────────────────────────────────────────────
    if "recls_clear_btn" in c_draw:
        c_draw["recls_clear_btn"].click(
            fn=_clear_draw_results,
            inputs=None,
            outputs=[
                c_draw["recls_status"],
                c_draw["recls_annotated"],
                c_draw["recls_results"],
                c_draw["recls_yolo"],
            ],
        )
