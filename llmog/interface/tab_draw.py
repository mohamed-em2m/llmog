"""
Draw & Recognize Reclassification Tab module.
Implements a high-performance Custom HTML5 Canvas Frontend paired with Gradio's Backend,
providing dedicated bounding box (rectangle), freehand brush stroke, circle, and eraser tools,
zoom/pan controls, drag-and-drop / clipboard image loading, undo/redo history,
instant Auto-Recognize mode, and multi-mode VLM region recognition (Strict, Hybrid, Free) with YOLO labeling.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, Any, Tuple
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


# ── Canvas Stage HTML — ribbon now only shows title/mode status ────────────────
# Upload / Recognize actions live in the toolbar's "Image" group (single source
# of truth) instead of being duplicated here.
_CANVAS_STAGE_HTML = """
<style>
/* Pin the canvas stage to the same height as the results viewer (520px,
   see DetectionViewer(list_height=520) in build_draw_tab) so input and
   output panels always match, and keep it fixed regardless of what's
   loaded into it — no element inside is allowed to grow the wrapper. */
#llmog-custom-canvas-app .canvas-stage-wrapper {
    position: relative;
    width: 100%;
    height: 520px;
    min-height: 520px;
    max-height: 520px;
    overflow: hidden;
    box-sizing: border-box;
}
#llmog-custom-canvas-app #custom-annotation-canvas {
    display: block;
    width: 100%;
    height: 100%;
}
#llmog-custom-canvas-app .canvas-empty-state {
    position: absolute;
    inset: 0;
    overflow: auto;
    box-sizing: border-box;
}
#llmog-custom-canvas-app .canvas-status-bar {
    flex-wrap: wrap;
    max-height: 110px;
    overflow-y: auto;
    box-sizing: border-box;
}
#llmog-custom-canvas-app .regions-list-chips {
    max-height: 70px;
    overflow-y: auto;
    flex-wrap: wrap;
}
</style>
<div id="llmog-custom-canvas-app" class="custom-canvas-container canvas-stage-card">
    <!-- Top Status Ribbon (read-only indicator, no duplicate action buttons) -->
    <div class="canvas-top-ribbon">
        <div class="ribbon-left">
            <span class="ribbon-title">🎨 Interactive Annotation Canvas</span>
            <span class="mode-badge-pill manual" id="draw-mode-indicator-pill">✋ Manual Mode</span>
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
            <span class="canvas-hint-text" id="draw-canvas-hint-text">💡 Draw a box or brush stroke over any object, then click <b>Recognize Now</b>.</span>
        </div>
        <div class="regions-list-chips" id="regions-chips-container"></div>
    </div>
</div>
"""

# ── Clean, Reorganized Toolbar below Canvas ────────────────────────────────────
# "Image" group now owns both Upload and Recognize — the single place to load
# an image and send it off, instead of splitting the actions across the ribbon
# and toolbar.
_TOOLBAR_HTML = """
<div id="llmog-custom-canvas-toolbar" class="custom-canvas-container draw-toolbar-below">
    <div class="canvas-toolbar">
        <!-- Workflow Execution Mode Toggle -->
        <div class="canvas-tool-group highlight-group">
            <span class="tool-group-label">Workflow Mode</span>
            <button type="button" class="canvas-tool-btn active" id="draw-mode-btn-manual" title="Manual: Draw regions freely, then click Recognize"><span class="tool-icon">✋</span> Manual</button>
            <button type="button" class="canvas-tool-btn auto-mode-btn" id="draw-mode-btn-auto" title="Auto: Instantly recognize each object as soon as drawing completes!"><span class="tool-icon">⚡</span> Auto-Recognize on Draw</button>
        </div>

        <div class="canvas-toolbar-divider"></div>

        <!-- Image: Upload + Recognize live together here -->
        <div class="canvas-tool-group highlight-group">
            <span class="tool-group-label">Image</span>
            <button type="button" class="canvas-tool-btn upload-btn" id="btn-toolbar-upload" title="Upload an image from your computer">📁 Upload Image</button>
            <button type="button" class="btn-ribbon-cta" id="btn-toolbar-recognize" title="Send drawn regions to VLM (Ctrl+Enter)">
                <span class="cta-icon">🔎</span> <b>Recognize Now</b>
            </button>
        </div>

        <div class="canvas-toolbar-divider"></div>

        <!-- Tools -->
        <div class="canvas-tool-group">
            <span class="tool-group-label">Tools</span>
            <button type="button" class="canvas-tool-btn active" id="tool-bbox" title="Bounding Box (Drag rectangle) [B]"><span class="tool-icon">🔲</span> Box</button>
            <button type="button" class="canvas-tool-btn" id="tool-brush" title="Freehand Brush [P]"><span class="tool-icon">🖌️</span> Brush</button>
            <button type="button" class="canvas-tool-btn" id="tool-circle" title="Circle / Ellipse [C]"><span class="tool-icon">⭕</span> Circle</button>
            <button type="button" class="canvas-tool-btn" id="tool-eraser" title="Eraser / Delete [E]"><span class="tool-icon">🧽</span> Eraser</button>
        </div>

        <div class="canvas-toolbar-divider"></div>

        <!-- Color & Size -->
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

        <!-- Actions -->
        <div class="canvas-tool-group">
            <span class="tool-group-label">Actions</span>
            <button type="button" class="canvas-tool-btn" id="btn-undo" title="Undo [Ctrl+Z]">↩️ Undo</button>
            <button type="button" class="canvas-tool-btn" id="btn-redo" title="Redo [Ctrl+Y]">🔁 Redo</button>
            <button type="button" class="canvas-tool-btn" id="btn-clear-drawings" title="Clear drawn boxes & strokes only">🧽 Clear Drawings</button>
            <button type="button" class="canvas-tool-btn danger" id="btn-clear-all" title="Clear image & drawings completely">🧹 Reset All</button>
        </div>

        <div class="canvas-toolbar-divider"></div>

        <!-- Zoom -->
        <div class="canvas-tool-group">
            <span class="tool-group-label">Zoom</span>
            <button type="button" class="canvas-tool-btn icon-only" id="btn-zoom-in" title="Zoom in">➕</button>
            <button type="button" class="canvas-tool-btn icon-only" id="btn-zoom-out" title="Zoom out">➖</button>
            <button type="button" class="canvas-tool-btn" id="btn-zoom-fit" title="Fit to viewport">📐 Fit</button>
            <span id="zoom-level-text" class="zoom-indicator">100%</span>
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

        executionMode: 'manual',
        isAutoSubmitting: false,

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

            // Gradio can re-render this component (e.g. tab switches) and re-run
            // this boot script against a fresh DOM tree. Window-level listeners
            // from a previous mount would otherwise pile up and fire repeatedly
            // against a stale controller. Tear those down before rebinding.
            this._teardownGlobalListeners();
            this.bindEvents();
            this.resizeCanvas();
            this.syncGradioPayload();
        },

        _teardownGlobalListeners: function() {
            const h = window.__llmog_canvas_global_handlers__;
            if (!h) return;
            window.removeEventListener('resize', h.resize);
            window.removeEventListener('mousemove', h.mousemove);
            window.removeEventListener('mouseup', h.mouseup);
            window.removeEventListener('touchmove', h.touchmove);
            window.removeEventListener('touchend', h.touchend);
            window.removeEventListener('paste', h.paste);
            window.removeEventListener('keydown', h.keydown);
            window.__llmog_canvas_global_handlers__ = null;
        },

        bindEvents: function() {
            const self = this;

            const isTabVisible = () => {
                const el = document.getElementById('llmog-custom-canvas-app');
                return !!(el && el.offsetParent !== null);
            };

            // ── Recognition Mode Toggle ────────────────────────────────────
            const btnModeManual = document.getElementById('draw-mode-btn-manual');
            const btnModeAuto = document.getElementById('draw-mode-btn-auto');
            const modePill = document.getElementById('draw-mode-indicator-pill');
            const hintText = document.getElementById('draw-canvas-hint-text');

            const setExecutionMode = (mode) => {
                self.executionMode = mode;
                if (btnModeManual) btnModeManual.classList.toggle('active', mode === 'manual');
                if (btnModeAuto) btnModeAuto.classList.toggle('active', mode === 'auto');

                if (modePill) {
                    modePill.className = 'mode-badge-pill ' + (mode === 'auto' ? 'auto' : 'manual');
                    modePill.innerHTML = mode === 'auto' ? '⚡ Auto-Recognize Active' : '✋ Manual Mode';
                }
                if (hintText) {
                    hintText.innerHTML = mode === 'auto'
                        ? '⚡ <b>Auto Mode Active:</b> Complete any box, circle, or stroke to recognize it immediately!'
                        : '💡 Draw shapes over objects, then click <b>Recognize Now</b>.';
                }
            };

            if (btnModeManual) btnModeManual.onclick = () => setExecutionMode('manual');
            if (btnModeAuto) btnModeAuto.onclick = () => setExecutionMode('auto');

            // ── Image group: Upload + Recognize (single source, lives in toolbar) ──
            const btnToolbarUpload = document.getElementById('btn-toolbar-upload');
            const btnToolbarRecognize = document.getElementById('btn-toolbar-recognize');
            const btnEmptyUpload = document.getElementById('btn-empty-upload');
            const btnEmptySample = document.getElementById('btn-empty-sample');

            if (btnToolbarUpload) btnToolbarUpload.onclick = () => self.triggerFileUpload();
            if (btnToolbarRecognize) btnToolbarRecognize.onclick = () => self.triggerRecognize();
            if (btnEmptyUpload) btnEmptyUpload.onclick = () => self.triggerFileUpload();
            if (btnEmptySample) btnEmptySample.onclick = () => self.loadSampleImage();

            // Reuse the single hidden <input type=file> already in the DOM instead
            // of creating (and hoping to clean up) a throwaway input per click.
            if (self.fileInput) {
                self.fileInput.onchange = function(e) {
                    const file = e.target.files && e.target.files[0];
                    if (file) self.loadImageFromFile(file);
                    self.fileInput.value = '';
                };
            }

            // ── Tool Selection ────────────────────────────────────────────
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

            // ── Colors ────────────────────────────────────────────────────
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

            // ── Size ──────────────────────────────────────────────────────
            const sizeSlider = document.getElementById('brush-size-slider');
            const sizeVal = document.getElementById('brush-size-val');
            if (sizeSlider) {
                sizeSlider.oninput = (e) => {
                    self.size = parseInt(e.target.value, 10);
                    if (sizeVal) sizeVal.textContent = self.size;
                };
            }

            // ── History & Clear Actions ───────────────────────────────────
            const btnUndo = document.getElementById('btn-undo');
            const btnRedo = document.getElementById('btn-redo');
            const btnClearDrawings = document.getElementById('btn-clear-drawings');
            const btnClearAll = document.getElementById('btn-clear-all');

            if (btnUndo) btnUndo.onclick = () => self.undo();
            if (btnRedo) btnRedo.onclick = () => self.redo();
            if (btnClearDrawings) btnClearDrawings.onclick = () => self.clearDrawings();
            if (btnClearAll) btnClearAll.onclick = () => self.clearAll();

            // ── Zoom Controls ─────────────────────────────────────────────
            const btnZoomIn = document.getElementById('btn-zoom-in');
            const btnZoomOut = document.getElementById('btn-zoom-out');
            const btnZoomFit = document.getElementById('btn-zoom-fit');

            if (btnZoomIn) btnZoomIn.onclick = () => self.zoom(1.2);
            if (btnZoomOut) btnZoomOut.onclick = () => self.zoom(1 / 1.2);
            if (btnZoomFit) btnZoomFit.onclick = () => self.fitToScreen();

            // ── Pointer Handlers (element-level; safe to rebind each mount) ──
            self.canvas.addEventListener('mousedown', (e) => self.onPointerDown(e));

            self.canvas.addEventListener('touchstart', (e) => {
                if (e.touches.length === 1) {
                    const touch = e.touches[0];
                    self.onPointerDown({ clientX: touch.clientX, clientY: touch.clientY, button: 0, preventDefault: () => e.preventDefault() });
                }
            }, { passive: false });

            self.wrapper.addEventListener('wheel', (e) => {
                e.preventDefault();
                const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
                const rect = self.canvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;
                self.zoomAt(zoomFactor, mouseX, mouseY);
            }, { passive: false });

            // ── Drag & Drop ───────────────────────────────────────────────
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

            // ── Window-level Handlers ────────────────────────────────────
            // Named + stored so a later mount can remove exactly these before
            // adding its own (see _teardownGlobalListeners).
            const onResize = () => self.resizeCanvas();
            const onMouseMove = (e) => self.onPointerMove(e);
            const onMouseUp = (e) => self.onPointerUp(e);
            const onTouchMove = (e) => {
                if (self.isDrawing || self.isPanning) {
                    if (e.touches.length === 1) {
                        const touch = e.touches[0];
                        self.onPointerMove({ clientX: touch.clientX, clientY: touch.clientY });
                    }
                }
            };
            const onTouchEnd = (e) => {
                if (self.isDrawing || self.isPanning) {
                    self.onPointerUp(e);
                }
            };
            const onPaste = (e) => {
                if (!isTabVisible()) return;
                const items = (e.clipboardData || e.originalEvent.clipboardData).items;
                for (let i = 0; i < items.length; i++) {
                    if (items[i].type.indexOf('image') !== -1) {
                        const blob = items[i].getAsFile();
                        self.loadImageFromFile(blob);
                        break;
                    }
                }
            };
            const onKeyDown = (e) => {
                if (!isTabVisible()) return;
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                    e.preventDefault();
                    self.triggerRecognize();
                } else if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
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
            };

            window.addEventListener('resize', onResize);
            window.addEventListener('mousemove', onMouseMove);
            window.addEventListener('mouseup', onMouseUp);
            window.addEventListener('touchmove', onTouchMove, { passive: false });
            window.addEventListener('touchend', onTouchEnd);
            window.addEventListener('paste', onPaste);
            window.addEventListener('keydown', onKeyDown);

            window.__llmog_canvas_global_handlers__ = {
                resize: onResize,
                mousemove: onMouseMove,
                mouseup: onMouseUp,
                touchmove: onTouchMove,
                touchend: onTouchEnd,
                paste: onPaste,
                keydown: onKeyDown,
            };
        },

        resizeCanvas: function() {
            if (!this.wrapper || !this.canvas) return;
            const w = this.wrapper.clientWidth;
            const h = this.wrapper.clientHeight || 520;
            this.canvas.width = w;
            this.canvas.height = h;
            if (this.image) this.clampOffsets();
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
                self.render();
            };
            img.onerror = function() {
                alert('That file could not be loaded as an image.');
            };
            img.src = dataUrl;
        },

        loadImageFromFile: function(file) {
            const self = this;
            const reader = new FileReader();
            reader.onload = function(e) {
                self.loadImageFromDataUrl(e.target.result);
            };
            reader.onerror = function() {
                alert('Could not read that file.');
            };
            reader.readAsDataURL(file);
        },

        triggerFileUpload: function() {
            if (this.fileInput) {
                this.fileInput.click();
            }
        },

        loadSampleImage: function() {
            const sampleBtn = document.getElementById('recls_sample_bridge_btn');
            if (sampleBtn) {
                sampleBtn.click();
            }
        },

        getFitScale: function() {
            const cw = this.canvas.width, ch = this.canvas.height;
            const iw = this.imageWidth || 1, ih = this.imageHeight || 1;
            return Math.min(cw / iw, ch / ih);
        },

        clampOffsets: function() {
            const cw = this.canvas.width, ch = this.canvas.height;
            const iw = this.imageWidth, ih = this.imageHeight;
            if (!iw || !ih) return;
            const sw = iw * this.scale, sh = ih * this.scale;
            if (sw <= cw) {
                this.offsetX = (cw - sw) / 2;
            } else {
                this.offsetX = Math.max(cw - sw, Math.min(0, this.offsetX));
            }
            if (sh <= ch) {
                this.offsetY = (ch - sh) / 2;
            } else {
                this.offsetY = Math.max(ch - sh, Math.min(0, this.offsetY));
            }
        },

        fitToScreen: function() {
            if (!this.image || !this.canvas) return;
            const cw = this.canvas.width;
            const ch = this.canvas.height;
            const iw = this.imageWidth;
            const ih = this.imageHeight;
            this.scale = this.getFitScale();
            if (!isFinite(this.scale) || this.scale <= 0) this.scale = 1.0;
            this.scale = Math.max(0.05, Math.min(this.scale, 15.0));
            this.offsetX = (cw - iw * this.scale) / 2;
            this.offsetY = (ch - ih * this.scale) / 2;
            this.clampOffsets();
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
            const fit = this.getFitScale();
            let newScale = this.scale * factor;
            newScale = Math.max(fit, Math.min(newScale, 15.0));
            this.offsetX = mouseX - (mouseX - this.offsetX) * (newScale / prevScale);
            this.offsetY = mouseY - (mouseY - this.offsetY) * (newScale / prevScale);
            this.scale = newScale;
            this.clampOffsets();
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

            // Middle-click, or left-click + Alt, pans the canvas.
            if (e.button === 1 || (e.button === 0 && e.altKey)) {
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
                this.clampOffsets();
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

            let newRegion = null;

            if (this.mode === 'bbox') {
                const x1 = Math.min(this.startX, this.currentX);
                const y1 = Math.min(this.startY, this.currentY);
                const x2 = Math.max(this.startX, this.currentX);
                const y2 = Math.max(this.startY, this.currentY);

                if (x2 - x1 >= 6 && y2 - y1 >= 6) {
                    newRegion = {
                        type: 'bbox',
                        x1: Math.round(x1),
                        y1: Math.round(y1),
                        x2: Math.round(x2),
                        y2: Math.round(y2),
                        color: this.color,
                        size: this.size
                    };
                }
            } else if (this.mode === 'circle') {
                const x1 = Math.min(this.startX, this.currentX);
                const y1 = Math.min(this.startY, this.currentY);
                const x2 = Math.max(this.startX, this.currentX);
                const y2 = Math.max(this.startY, this.currentY);

                if (x2 - x1 >= 6 && y2 - y1 >= 6) {
                    newRegion = {
                        type: 'circle',
                        x1: Math.round(x1),
                        y1: Math.round(y1),
                        x2: Math.round(x2),
                        y2: Math.round(y2),
                        color: this.color,
                        size: this.size
                    };
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
                newRegion = {
                    type: 'stroke',
                    x1: Math.round(Math.max(0, minX - pad)),
                    y1: Math.round(Math.max(0, minY - pad)),
                    x2: Math.round(Math.min(this.imageWidth, maxX + pad)),
                    y2: Math.round(Math.min(this.imageHeight, maxY + pad)),
                    points: this.currentStroke.slice(),
                    color: this.color,
                    size: this.size
                };
            }

            if (newRegion) {
                if (this.executionMode === 'auto') {
                    this.saveStateForUndo();
                    this.regions = [newRegion];
                } else {
                    this.saveStateForUndo();
                    this.regions.push(newRegion);
                }

                this.updateRegionsList();
                this.syncGradioPayload();

                if (this.executionMode === 'auto') {
                    this.triggerRecognize();
                }
            }

            this.currentStroke = [];
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

            ctx.fillStyle = '#1A1145';
            ctx.fillRect(x, badgeY, badgeW, badgeH);

            ctx.strokeStyle = '#F5C842';
            ctx.lineWidth = 1.5 / this.scale;
            ctx.strokeRect(x, badgeY, badgeW, badgeH);

            ctx.fillStyle = '#FFF8EC';
            ctx.fillText(text, x + pad, badgeY + badgeH - pad * 1.5);
        },

        drawBackgroundPattern: function(ctx, w, h) {
            ctx.fillStyle = '#FFF8EC';
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
            return jsonStr;
        },

        triggerRecognize: function() {
            if (this.isAutoSubmitting) return;
            if (!this.image) {
                alert("Please upload an image first.");
                return;
            }
            if (this.regions.length === 0) {
                alert("Please draw at least one bounding box, circle, or brush stroke over an object.");
                return;
            }

            this.isAutoSubmitting = true;
            this.syncGradioPayload();

            setTimeout(() => {
                const gradioBtn = document.getElementById('draw_run_recognize_btn');
                if (gradioBtn) {
                    gradioBtn.click();
                }
                setTimeout(() => { this.isAutoSubmitting = false; }, 300);
            }, 100);
        }
    };

    window.getCustomDrawData = function() {
        if (window.CustomCanvasController) {
            return window.CustomCanvasController.syncGradioPayload();
        }
        return window.__llmog_custom_canvas_payload__ || "{}";
    };

    const boot = () => {
        const root = document.getElementById('llmog-custom-canvas-app');
        if (!root) { setTimeout(boot, 200); return; }
        if (window.__draw_canvas_booted_root__ === root) return;
        window.__draw_canvas_booted_root__ = root;
        window.CustomCanvasController.init();
    };
    setTimeout(boot, 50);
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
    """Build the dedicated Draw & Recognize tab — twin 520px (canvas + results same level), tools below canvas."""
    # ── Top twin: canvas stage (520) | recognition viewer (520) — same level ──
    with gr.Row(equal_height=True, elem_classes=["draw-tab-row", "twin-screens-row"]):
        with gr.Column(scale=1, min_width=420, elem_classes=["batch-bottom-col"]):
            custom_canvas_view = gr.HTML(
                value=_CANVAS_STAGE_HTML,
                js_on_load=_CUSTOM_CANVAS_JS,
                elem_id="draw-canvas-html",
            )
        with gr.Column(scale=1, min_width=420, elem_classes=["batch-bottom-col"]):
            gr.HTML('<p class="section-label">👁️ Recognition Result (Interactive)</p>')
            recls_status = gr.Markdown(
                "**Status: Idle — upload an image and draw regions to recognize**"
            )
            with gr.Group(elem_classes=["img-viewer-wrap"]):
                recls_annotated = DetectionViewer(
                    label="Annotated Recognition Result",
                    panel_title="Recognized Regions",
                    list_height=520,
                    elem_id="draw-detection-viewer",
                )
            recls_results = gr.HTML(value=_RECLS_EMPTY_TABLE)
            with gr.Accordion("YOLO Labels (<class_id> <xc> <yc> <w> <h>)", open=False):
                recls_yolo = gr.Textbox(
                    lines=8,
                    interactive=False,
                    label="Copy these lines into the image's .txt label file",
                )

    # ── Below twin: toolbar + controls ──
    toolbar_view = gr.HTML(value=_TOOLBAR_HTML, elem_id="draw-toolbar-html")

    # ── Hidden State Bridges ──
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

    # ── Prominent Action Row (Linked to Toolbar's Image group CTA) ──
    with gr.Row(elem_classes=["btn-group"]):
        recls_run_btn = gr.Button(
            "🔎  Recognize Drawn Regions",
            variant="primary",
            scale=2,
            interactive=True,
            elem_id="draw_run_recognize_btn",
        )
        recls_clear_btn = gr.Button(
            "🗑️ Clear Results",
            variant="secondary",
            scale=1,
        )

    # ── Target classes & detection mode (input) ──
    with gr.Accordion("🎯 Target Classes & Detection Mode", open=False):
        recls_class_mode = gr.Radio(
            label="🎯 Class Expectation Mode",
            choices=[
                ("🔒 Strict (Closed-Set)", "strict"),
                ("🔀 Hybrid (Extendable)", "hybrid"),
                ("🌐 Free (Open-World)", "free"),
            ],
            value="free",
            info="Free: Agent autonomously names any object/defect. Strict: locked to listed classes.",
        )
        recls_preset_dropdown = gr.Dropdown(
            label="📋 Category Domain Presets",
            choices=list(CATEGORY_PRESETS.keys()),
            value="Fabric & Surface Defects",
            visible=False,
            info="Quickly load target classes & expert distinguishing definitions.",
        )
        recls_classes_input = gr.Textbox(
            label="Domain / Focus Hint (Optional)",
            placeholder="e.g. Focus on industrial defects, wildlife, electronics... (or leave blank)",
            value="",
            lines=2,
            info="Free Mode: Agent autonomously names any object/defect. Predefined classes are not required.",
        )
        recls_defs_input = gr.Textbox(
            label="Domain Guidance / Prompt Context (Optional)",
            placeholder="Optional domain context or special inspection criteria...",
            lines=4,
            value="",
            info="Optional domain guidance.",
        )

    # ── Advanced Context Settings ──
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
            info="Parallel: N concurrent requests (~N× faster). Batched: 1 request with N images (fewest round-trips).",
        )

    return dict(
        custom_canvas_view=custom_canvas_view,
        toolbar_view=toolbar_view,
        custom_draw_payload=custom_draw_payload,
        recls_sample_bridge_btn=recls_sample_bridge_btn,
        recls_class_mode=recls_class_mode,
        recls_preset_dropdown=recls_preset_dropdown,
        recls_classes_input=recls_classes_input,
        recls_defs_input=recls_defs_input,
        recls_conf_threshold=recls_conf_threshold,
        recls_padding_slider=recls_padding_slider,
        recls_request_mode=recls_request_mode,
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
            js="() => { if (window.CustomCanvasController) { setTimeout(() => { const ta = document.querySelector('#custom_draw_payload_box textarea'); if (ta && ta.value) { try { const p = JSON.parse(ta.value); if (p.background) { window.CustomCanvasController.loadImageFromDataUrl(p.background); } else { alert('Sample image asset not found on server.'); } } catch(e){} } }, 300); } }",
        )

    # ── Recognition execution with Gradio Backend ───────────────────────────
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
