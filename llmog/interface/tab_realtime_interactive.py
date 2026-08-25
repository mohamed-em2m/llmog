"""
Real-Time Interactive Camera & Draw Tab Module.
Allows users to open their webcam directly inside the interactive canvas,
draw bounding boxes, circles, or brush strokes directly over live or frozen camera feeds,
and classify the selected objects using Vision-Language Models (VLM) in Strict, Hybrid, or Free mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import gradio as gr

from detection_viewer import DetectionViewer
from interface.batch.reclassification import (
    classify_regions_gui,
    _RECLS_EMPTY_TABLE,
    CATEGORY_PRESETS,
)

_SAMPLE_IMAGE_PATH = Path(__file__).resolve().parents[2] / "assets" / "image.png"


def _get_sample_image_b64() -> str:
    if _SAMPLE_IMAGE_PATH.exists():
        try:
            import base64

            with open(_SAMPLE_IMAGE_PATH, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{data}"
        except Exception:
            pass
    return ""


def _on_realtime_interactive_preset_change(preset_name: str) -> Tuple[gr.update, gr.update]:
    preset = CATEGORY_PRESETS.get(preset_name, CATEGORY_PRESETS["Custom / Blank"])
    return gr.update(value=preset["classes"]), gr.update(value=preset["defs"])


def _on_realtime_interactive_mode_change(mode: str) -> Tuple[gr.update, gr.update, gr.update]:
    if mode == "free":
        return (
            gr.update(
                label="Domain / Focus Hint (Optional)",
                placeholder="e.g. Focus on defects, wildlife, tools, packaging... (or leave blank)",
                info="Free Mode: Agent autonomously identifies and names whatever objects are drawn.",
            ),
            gr.update(
                label="Domain Guidance (Optional)",
                placeholder="Optional domain context or special inspection criteria...",
                info="Optional domain guidance.",
            ),
            gr.update(visible=False),
        )
    elif mode == "hybrid":
        return (
            gr.update(
                label="Priority Target Classes (comma-separated)",
                placeholder="e.g. person, car, phone, cup",
                info="Hybrid Mode: Agent prioritizes these classes, but will name novel objects if detected.",
            ),
            gr.update(
                label="Category Definitions & Discovery Guidelines",
                placeholder="Definitions for priority classes...",
                info="Definitions for priority classes.",
            ),
            gr.update(visible=True),
        )
    else:  # strict
        return (
            gr.update(
                label="Target Classes (Strict - Comma Separated)",
                placeholder="person, car, bottle, laptop, chair",
                info="Strict Mode: Agent is strictly restricted to these classes (or 'none').",
            ),
            gr.update(
                label="Class Definitions / Distinguishing Rules",
                placeholder="Write criteria for distinguishing each class...",
                info="Detailed criteria for distinguishing each class.",
            ),
            gr.update(visible=True),
        )


# ── Interactive HTML5 Canvas with Native Live Camera Feed ──────────────────────
_RT_DRAW_CANVAS_HTML = """
<div id="llmog-custom-canvas-app-rt" class="custom-canvas-container canvas-stage-card">
    <!-- Canvas Stage Viewport with Live Camera — video kept visible to compositing (opacity 0, NOT display:none so drawImage works) -->
    <div class="canvas-stage-wrapper" id="rt-canvas-stage-wrapper">
        <video id="rt-camera-video" playsinline muted autoplay style="position:absolute; width:2px; height:2px; opacity:0.01; pointer-events:none; top:0; left:0;"></video>
        <canvas id="rt-custom-annotation-canvas"></canvas>

        <div id="rt-canvas-empty-overlay" class="canvas-empty-state">
            <div class="empty-icon">📹</div>
            <h3>Real-Time Interactive Camera &amp; Draw</h3>
            <p>Start your camera and draw bounding boxes or brush strokes directly over objects to recognize them.</p>
            <div class="empty-actions">
                <button type="button" class="btn-canvas-primary" id="rt-btn-empty-camera">📹 Start Camera</button>
                <button type="button" class="btn-canvas-secondary" id="rt-btn-empty-upload">📁 Choose Image</button>
                <button type="button" class="btn-canvas-secondary" id="rt-btn-empty-sample">🖼️ Load Sample</button>
            </div>
            <span class="drag-hint">or drag &amp; drop an image here / paste from clipboard (Ctrl+V)</span>
        </div>

        <input type="file" id="rt-canvas-file-input" accept="image/*" style="display:none">
    </div>

    <!-- Regions Live Summary Bar -->
    <div class="canvas-status-bar">
        <div class="status-left">
            <span class="regions-count-badge" id="rt-regions-count-badge">0 Region(s)</span>
            <span class="canvas-hint-text">💡 Tip: Start camera, select <b>Box</b> to mark objects directly on the camera feed.</span>
        </div>
        <div class="regions-list-chips" id="rt-regions-chips-container"></div>
    </div>
</div>
"""

# ── RT camera/annotation toolbar extracted below canvas (same IDs — JS binds globally) ──
_RT_TOOLBAR_HTML = """
<div id="llmog-custom-canvas-toolbar-rt" class="custom-canvas-container draw-toolbar-below">
    <div class="canvas-toolbar">
        <div class="canvas-tool-group">
            <span class="tool-group-label">Camera</span>
            <button type="button" class="canvas-tool-btn camera-btn" id="rt-btn-start-camera" title="Start Live Webcam Feed"><span class="tool-icon">📹</span> Start Camera</button>
            <button type="button" class="canvas-tool-btn" id="rt-btn-freeze-camera" style="display:none;" title="Freeze / Resume Live Video [Space]"><span class="tool-icon">⏸️</span> Freeze</button>
            <button type="button" class="canvas-tool-btn icon-only" id="rt-btn-flip-camera" style="display:none;" title="Flip Front / Back Camera">🔄</button>
            <button type="button" class="canvas-tool-btn danger" id="rt-btn-stop-camera" style="display:none;" title="Stop Camera">⏹️ Stop</button>
        </div>
        <div class="canvas-toolbar-divider"></div>
        <div class="canvas-tool-group">
            <span class="tool-group-label">Tools</span>
            <button type="button" class="canvas-tool-btn active" id="rt-tool-bbox" title="Bounding Box [B]"><span class="tool-icon">🔲</span> Box</button>
            <button type="button" class="canvas-tool-btn" id="rt-tool-brush" title="Freehand Brush [P]"><span class="tool-icon">🖌️</span> Brush</button>
            <button type="button" class="canvas-tool-btn" id="rt-tool-circle" title="Circle / Ellipse [C]"><span class="tool-icon">⭕</span> Circle</button>
            <button type="button" class="canvas-tool-btn" id="rt-tool-eraser" title="Eraser / Delete Region [E]"><span class="tool-icon">🧽</span> Eraser</button>
        </div>
        <div class="canvas-toolbar-divider"></div>
        <div class="canvas-tool-group">
            <span class="tool-group-label">Color</span>
            <div class="color-palette-bar" id="rt-palette-swatches">
                <button type="button" class="color-swatch active" style="background:#00ffcc" data-color="#00ffcc"></button>
                <button type="button" class="color-swatch" style="background:#ff3c3c" data-color="#ff3c3c"></button>
                <button type="button" class="color-swatch" style="background:#0096ff" data-color="#0096ff"></button>
                <button type="button" class="color-swatch" style="background:#ffd214" data-color="#ffd214"></button>
                <button type="button" class="color-swatch" style="background:#963cff" data-color="#963cff"></button>
                <button type="button" class="color-swatch" style="background:#ffffff" data-color="#ffffff"></button>
            </div>
            <input type="color" id="rt-custom-color-picker" value="#00ffcc" title="Custom color" class="color-picker-input">
        </div>
        <div class="canvas-toolbar-divider"></div>
        <div class="canvas-tool-group">
            <span class="tool-group-label">Size: <b id="rt-brush-size-val">3</b>px</span>
            <input type="range" id="rt-brush-size-slider" min="1" max="40" value="3" class="canvas-range-slider" title="Stroke thickness">
        </div>
        <div class="canvas-toolbar-divider"></div>
        <div class="canvas-tool-group">
            <span class="tool-group-label">Actions</span>
            <button type="button" class="canvas-tool-btn" id="rt-btn-undo" title="Undo [Ctrl+Z]">↩️ Undo</button>
            <button type="button" class="canvas-tool-btn" id="rt-btn-redo" title="Redo [Ctrl+Y]">🔁 Redo</button>
            <button type="button" class="canvas-tool-btn" id="rt-btn-clear-drawings" title="Clear drawn boxes & strokes only">🧽 Clear Drawings</button>
            <button type="button" class="canvas-tool-btn danger" id="rt-btn-clear-all" title="Clear camera & drawings completely">🧹 Reset All</button>
        </div>
        <div class="canvas-toolbar-divider"></div>
        <div class="canvas-tool-group">
            <span class="tool-group-label">Zoom</span>
            <button type="button" class="canvas-tool-btn icon-only" id="rt-btn-zoom-in" title="Zoom in">➕</button>
            <button type="button" class="canvas-tool-btn icon-only" id="rt-btn-zoom-out" title="Zoom out">➖</button>
            <button type="button" class="canvas-tool-btn" id="rt-btn-zoom-fit" title="Fit to viewport">📐 Fit</button>
            <span id="rt-zoom-level-text" class="zoom-indicator">100%</span>
        </div>
        <div class="canvas-toolbar-divider"></div>
        <div class="canvas-tool-group">
            <span class="tool-group-label">File</span>
            <button type="button" class="canvas-tool-btn upload-btn" id="rt-btn-toolbar-upload" title="Upload an image from your computer">📁 Upload</button>
        </div>
    </div>
</div>
"""

# ── JavaScript Controller for Real-Time Interactive Canvas ───────────────────
_RT_DRAW_CANVAS_JS = """
(function() {
    window.CustomCanvasControllerRT = {
        video: null,
        stream: null,
        isCameraRunning: false,
        isFrozen: false,
        facingMode: 'user',
        animFrameId: null,

        image: null,
        imageSrc: null,
        imageWidth: 0,
        imageHeight: 0,

        mode: 'bbox',
        color: '#00ffcc',
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
            this.container = document.getElementById('llmog-custom-canvas-app-rt');
            if (!this.container) return;

            this.canvas = document.getElementById('rt-custom-annotation-canvas');
            if (!this.canvas) return;
            this.ctx = this.canvas.getContext('2d');
            this.wrapper = document.getElementById('rt-canvas-stage-wrapper');
            this.emptyOverlay = document.getElementById('rt-canvas-empty-overlay');
            this.fileInput = document.getElementById('rt-canvas-file-input');
            this.video = document.getElementById('rt-camera-video');

            this.bindEvents();
            this.resizeCanvas();
            this.syncGradioPayload();
        },

        bindEvents: function() {
            const self = this;
            window.addEventListener('resize', () => self.resizeCanvas());

            // ── Camera Controls ───────────────────────────────────────────
            const btnStartCam = document.getElementById('rt-btn-start-camera');
            const btnEmptyCam = document.getElementById('rt-btn-empty-camera');
            const btnStopCam = document.getElementById('rt-btn-stop-camera');
            const btnFreezeCam = document.getElementById('rt-btn-freeze-camera');
            const btnFlipCam = document.getElementById('rt-btn-flip-camera');

            if (btnStartCam) btnStartCam.onclick = () => self.startCamera();
            if (btnEmptyCam) btnEmptyCam.onclick = () => self.startCamera();
            if (btnStopCam) btnStopCam.onclick = () => self.stopCamera();
            if (btnFreezeCam) btnFreezeCam.onclick = () => self.toggleFreezeCamera();
            if (btnFlipCam) btnFlipCam.onclick = () => self.flipCamera();

            // ── Drawing Tools ─────────────────────────────────────────────
            const toolBbox = document.getElementById('rt-tool-bbox');
            const toolBrush = document.getElementById('rt-tool-brush');
            const toolCircle = document.getElementById('rt-tool-circle');
            const toolEraser = document.getElementById('rt-tool-eraser');

            const setTool = (mode, btn) => {
                self.mode = mode;
                [toolBbox, toolBrush, toolCircle, toolEraser].forEach(b => b && b.classList.remove('active'));
                if (btn) btn.classList.add('active');
                if (mode === 'brush' && self.size < 6) {
                    self.size = 10;
                    const sl = document.getElementById('rt-brush-size-slider');
                    const sv = document.getElementById('rt-brush-size-val');
                    if (sl) sl.value = 10;
                    if (sv) sv.textContent = 10;
                } else if (mode === 'bbox' && self.size > 8) {
                    self.size = 3;
                    const sl = document.getElementById('rt-brush-size-slider');
                    const sv = document.getElementById('rt-brush-size-val');
                    if (sl) sl.value = 3;
                    if (sv) sv.textContent = 3;
                }
                self.render();
            };

            if (toolBbox) toolBbox.onclick = () => setTool('bbox', toolBbox);
            if (toolBrush) toolBrush.onclick = () => setTool('brush', toolBrush);
            if (toolCircle) toolCircle.onclick = () => setTool('circle', toolCircle);
            if (toolEraser) toolEraser.onclick = () => setTool('eraser', toolEraser);

            // ── Color Swatches ────────────────────────────────────────────
            const swatches = document.querySelectorAll('#rt-palette-swatches .color-swatch');
            const customPicker = document.getElementById('rt-custom-color-picker');

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

            // ── Size Slider ───────────────────────────────────────────────
            const sizeSlider = document.getElementById('rt-brush-size-slider');
            const sizeVal = document.getElementById('rt-brush-size-val');
            if (sizeSlider) {
                sizeSlider.oninput = (e) => {
                    self.size = parseInt(e.target.value, 10);
                    if (sizeVal) sizeVal.textContent = self.size;
                };
            }

            // ── Actions ───────────────────────────────────────────────────
            const btnUndo = document.getElementById('rt-btn-undo');
            const btnRedo = document.getElementById('rt-btn-redo');
            const btnClearDrawings = document.getElementById('rt-btn-clear-drawings');
            const btnClearAll = document.getElementById('rt-btn-clear-all');

            if (btnUndo) btnUndo.onclick = () => self.undo();
            if (btnRedo) btnRedo.onclick = () => self.redo();
            if (btnClearDrawings) btnClearDrawings.onclick = () => self.clearDrawings();
            if (btnClearAll) btnClearAll.onclick = () => self.clearAll();

            // ── Zoom Controls ─────────────────────────────────────────────
            const btnZoomIn = document.getElementById('rt-btn-zoom-in');
            const btnZoomOut = document.getElementById('rt-btn-zoom-out');
            const btnZoomFit = document.getElementById('rt-btn-zoom-fit');

            if (btnZoomIn) btnZoomIn.onclick = () => self.zoom(1.2);
            if (btnZoomOut) btnZoomOut.onclick = () => self.zoom(1 / 1.2);
            if (btnZoomFit) btnZoomFit.onclick = () => self.fitToScreen();

            // ── Upload & Sample ───────────────────────────────────────────
            const btnToolbarUpload = document.getElementById('rt-btn-toolbar-upload');
            const btnEmptyUpload = document.getElementById('rt-btn-empty-upload');
            const btnEmptySample = document.getElementById('rt-btn-empty-sample');

            if (btnToolbarUpload) btnToolbarUpload.onclick = () => self.triggerFileUpload();
            if (btnEmptyUpload) btnEmptyUpload.onclick = () => self.triggerFileUpload();
            if (btnEmptySample) btnEmptySample.onclick = () => self.loadSampleImage();

            // ── Mouse & Touch Pointer Handlers ────────────────────────────
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

            // ── Clipboard Paste ───────────────────────────────────────────
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

            // ── Keyboard Shortcuts ────────────────────────────────────────
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
                } else if (e.key === ' ' && self.isCameraRunning) {
                    e.preventDefault();
                    self.toggleFreezeCamera();
                }
            });
        },

        // ── Camera Streaming Logic ────────────────────────────────────────
        startCamera: function() {
            const self = this;
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                alert("Webcam access is not supported by your browser or connection is not HTTPS/localhost.");
                return;
            }

            const constraints = {
                video: {
                    facingMode: self.facingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };

            navigator.mediaDevices.getUserMedia(constraints)
                .then(function(stream) {
                    self.stream = stream;
                    if (!self.video) {
                        self.video = document.getElementById('rt-camera-video');
                    }
                    self.video.srcObject = stream;
                    self.video.play();

                    self.video.onloadedmetadata = function() {
                        self.isCameraRunning = true;
                        self.isFrozen = false;
                        self.imageWidth = self.video.videoWidth || 1280;
                        self.imageHeight = self.video.videoHeight || 720;
                        self.image = null; // streaming live video

                        if (self.emptyOverlay) self.emptyOverlay.style.display = 'none';
                        self.updateCameraUI(true);
                        self.fitToScreen();
                        self.startRenderLoop();
                    };
                })
                .catch(function(err) {
                    console.error("Camera access error:", err);
                    alert("Could not access camera: " + (err.message || err.name));
                });
        },

        stopCamera: function() {
            if (this.stream) {
                this.stream.getTracks().forEach(t => t.stop());
                this.stream = null;
            }
            if (this.video) {
                this.video.pause();
                this.video.srcObject = null;
            }
            this.isCameraRunning = false;
            this.isFrozen = false;
            if (this.animFrameId) {
                cancelAnimationFrame(this.animFrameId);
                this.animFrameId = null;
            }
            this.updateCameraUI(false);
            if (!this.image) {
                if (this.emptyOverlay) this.emptyOverlay.style.display = 'flex';
                this.render();
            }
        },

        toggleFreezeCamera: function() {
            if (!this.isCameraRunning && !this.isFrozen) return;

            if (!this.isFrozen) {
                // Freeze: snapshot current video frame into static Image
                const offscreen = document.createElement('canvas');
                offscreen.width = this.imageWidth;
                offscreen.height = this.imageHeight;
                const octx = offscreen.getContext('2d');
                octx.drawImage(this.video, 0, 0, this.imageWidth, this.imageHeight);
                const dataUrl = offscreen.toDataURL('image/jpeg', 0.95);

                const frozenImg = new Image();
                const self = this;
                frozenImg.onload = function() {
                    self.image = frozenImg;
                    self.imageSrc = dataUrl;
                    self.isFrozen = true;
                    if (self.animFrameId) {
                        cancelAnimationFrame(self.animFrameId);
                        self.animFrameId = null;
                    }
                    const btnFreeze = document.getElementById('rt-btn-freeze-camera');
                    if (btnFreeze) {
                        btnFreeze.innerHTML = '<span class="tool-icon">▶️</span> Resume';
                        btnFreeze.classList.add('camera-active');
                    }
                    self.render();
                    self.syncGradioPayload();
                };
                frozenImg.src = dataUrl;
            } else {
                // Resume live streaming
                this.isFrozen = false;
                this.image = null;
                const btnFreeze = document.getElementById('rt-btn-freeze-camera');
                if (btnFreeze) {
                    btnFreeze.innerHTML = '<span class="tool-icon">⏸️</span> Freeze';
                    btnFreeze.classList.remove('camera-active');
                }
                this.startRenderLoop();
            }
        },

        flipCamera: function() {
            this.facingMode = (this.facingMode === 'user') ? 'environment' : 'user';
            this.stopCamera();
            this.startCamera();
        },

        updateCameraUI: function(running) {
            const btnStart = document.getElementById('rt-btn-start-camera');
            const btnStop = document.getElementById('rt-btn-stop-camera');
            const btnFreeze = document.getElementById('rt-btn-freeze-camera');
            const btnFlip = document.getElementById('rt-btn-flip-camera');

            if (btnStart) btnStart.style.display = running ? 'none' : 'inline-flex';
            if (btnStop) btnStop.style.display = running ? 'inline-flex' : 'none';
            if (btnFreeze) btnFreeze.style.display = running ? 'inline-flex' : 'none';
            if (btnFlip) btnFlip.style.display = running ? 'inline-flex' : 'none';
        },

        startRenderLoop: function() {
            const self = this;
            if (self.animFrameId) cancelAnimationFrame(self.animFrameId);

            function loop() {
                if (self.isCameraRunning && !self.isFrozen) {
                    self.render();
                    self.animFrameId = requestAnimationFrame(loop);
                }
            }
            loop();
        },

        getFitScale: function() {
            const cw = this.canvas.width, ch = this.canvas.height;
            const iw = this.imageWidth || 1280, ih = this.imageHeight || 720;
            return Math.min(cw / iw, ch / ih);
        },
        clampOffsets: function() {
            const cw = this.canvas.width, ch = this.canvas.height;
            const iw = this.imageWidth || 1280, ih = this.imageHeight || 720;
            if (!iw || !ih) return;
            const sw = iw * this.scale, sh = ih * this.scale;
            if (sw <= cw) this.offsetX = (cw - sw) / 2;
            else this.offsetX = Math.max(cw - sw, Math.min(0, this.offsetX));
            if (sh <= ch) this.offsetY = (ch - sh) / 2;
            else this.offsetY = Math.max(ch - sh, Math.min(0, this.offsetY));
        },
        // ── Canvas Sizing & Navigation ────────────────────────────────────
        resizeCanvas: function() {
            if (!this.wrapper || !this.canvas) return;
            const w = this.wrapper.clientWidth;
            const h = this.wrapper.clientHeight || 580;
            this.canvas.width = w;
            this.canvas.height = h;
            this.clampOffsets();
            this.render();
        },

        loadImageFromDataUrl: function(dataUrl) {
            const self = this;
            const img = new Image();
            img.onload = function() {
                if (self.isCameraRunning) {
                    self.stopCamera();
                }
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
            const sampleBtn = document.getElementById('rt_sample_bridge_btn');
            if (sampleBtn) {
                sampleBtn.click();
            }
        },

        fitToScreen: function() {
            if ((!this.image && !this.isCameraRunning) || !this.canvas) return;
            const cw = this.canvas.width;
            const ch = this.canvas.height;
            const iw = this.imageWidth || 1280;
            const ih = this.imageHeight || 720;
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
            const ind = document.getElementById('rt-zoom-level-text');
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

        // ── Pointer Handlers ──────────────────────────────────────────────
        onPointerDown: function(e) {
            if (!this.image && !this.isCameraRunning) return;
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

            if (!this.isCameraRunning || this.isFrozen) {
                this.render();
            }
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

                if ((x2 - x1) > 5 && (y2 - y1) > 5) {
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

                if ((x2 - x1) > 5 && (y2 - y1) > 5) {
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

                const pad = (this.size || 10) / 2;
                newRegion = {
                    type: 'stroke',
                    points: this.currentStroke.slice(),
                    x1: Math.max(0, Math.round(minX - pad)),
                    y1: Math.max(0, Math.round(minY - pad)),
                    x2: Math.min(this.imageWidth, Math.round(maxX + pad)),
                    y2: Math.min(this.imageHeight, Math.round(maxY + pad)),
                    color: this.color,
                    size: this.size
                };
            }

            if (newRegion) {
                this.saveHistoryState();
                this.regions.push(newRegion);
                this.redoStack = [];
                this.updateRegionsList();
                this.syncGradioPayload();
            }

            this.currentStroke = [];
            this.render();
        },

        eraseAt: function(imgX, imgY) {
            for (let i = this.regions.length - 1; i >= 0; i--) {
                const r = this.regions[i];
                if (imgX >= r.x1 && imgX <= r.x2 && imgY >= r.y1 && imgY <= r.y2) {
                    this.saveHistoryState();
                    this.regions.splice(i, 1);
                    this.redoStack = [];
                    this.updateRegionsList();
                    this.syncGradioPayload();
                    this.render();
                    break;
                }
            }
        },

        removeRegion: function(idx) {
            if (idx >= 0 && idx < this.regions.length) {
                this.saveHistoryState();
                this.regions.splice(idx, 1);
                this.redoStack = [];
                this.updateRegionsList();
                this.syncGradioPayload();
                this.render();
            }
        },

        // ── History (Undo / Redo) ─────────────────────────────────────────
        saveHistoryState: function() {
            this.undoStack.push(JSON.parse(JSON.stringify(this.regions)));
            if (this.undoStack.length > 30) this.undoStack.shift();
        },

        undo: function() {
            if (this.undoStack.length === 0) return;
            this.redoStack.push(JSON.parse(JSON.stringify(this.regions)));
            this.regions = this.undoStack.pop();
            this.updateRegionsList();
            this.syncGradioPayload();
            this.render();
        },

        redo: function() {
            if (this.redoStack.length === 0) return;
            this.undoStack.push(JSON.parse(JSON.stringify(this.regions)));
            this.regions = this.redoStack.pop();
            this.updateRegionsList();
            this.syncGradioPayload();
            this.render();
        },

        clearDrawings: function() {
            if (this.regions.length === 0) return;
            this.saveHistoryState();
            this.regions = [];
            this.redoStack = [];
            this.updateRegionsList();
            this.syncGradioPayload();
            this.render();
        },

        clearAll: function() {
            if (this.isCameraRunning) {
                this.stopCamera();
            }
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
            const countBadge = document.getElementById('rt-regions-count-badge');
            if (countBadge) countBadge.textContent = `${this.regions.length} Region(s)`;

            const container = document.getElementById('rt-regions-chips-container');
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
                    window.CustomCanvasControllerRT.removeRegion(idx);
                };
                container.appendChild(chip);
            });
        },

        // ── Canvas Rendering Engine ───────────────────────────────────────
        render: function() {
            if (!this.ctx || !this.canvas) return;
            const ctx = this.ctx;
            const w = this.canvas.width;
            const h = this.canvas.height;

            ctx.clearRect(0, 0, w, h);
            this.drawBackgroundPattern(ctx, w, h);

            const hasSource = (this.isCameraRunning && this.video && this.video.readyState >= 2) || this.image;
            if (!hasSource) return;

            ctx.save();
            ctx.translate(this.offsetX, this.offsetY);
            ctx.scale(this.scale, this.scale);

            // Draw video or static image
            if (this.isCameraRunning && !this.isFrozen && this.video) {
                ctx.drawImage(this.video, 0, 0, this.imageWidth, this.imageHeight);
            } else if (this.image) {
                ctx.drawImage(this.image, 0, 0, this.imageWidth, this.imageHeight);
            }

            // Draw all completed regions
            this.regions.forEach((r, idx) => {
                this.drawSingleRegion(ctx, r, idx + 1);
            });

            // Draw live in-progress drawing preview
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

                ctx.fillStyle = isLive ? 'rgba(0, 255, 204, 0.2)' : 'rgba(0, 255, 204, 0.12)';
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

                ctx.fillStyle = isLive ? 'rgba(0, 255, 204, 0.2)' : 'rgba(0, 255, 204, 0.12)';
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

            ctx.fillStyle = 'rgba(232,75,138,0.06)';
            const step = 24;
            for (let x = 0; x < w; x += step) {
                for (let y = 0; y < h; y += step) {
                    ctx.fillRect(x, y, 2, 2);
                }
            }
        },

        syncGradioPayload: function() {
            let bgData = this.imageSrc || null;

            // If camera is streaming live and not frozen, capture current frame
            if (this.isCameraRunning && !this.isFrozen && this.video && this.video.readyState >= 2) {
                try {
                    const snapCanvas = document.createElement('canvas');
                    snapCanvas.width = this.imageWidth || this.video.videoWidth || 1280;
                    snapCanvas.height = this.imageHeight || this.video.videoHeight || 720;
                    const sctx = snapCanvas.getContext('2d');
                    sctx.drawImage(this.video, 0, 0, snapCanvas.width, snapCanvas.height);
                    bgData = snapCanvas.toDataURL('image/jpeg', 0.92);
                } catch(e) {
                    console.error("Frame capture error:", e);
                }
            }

            const payload = {
                background: bgData,
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
            window.__rt_interactive_payload__ = jsonStr;

            const hiddenTa = document.querySelector('#rt_interactive_payload_box textarea');
            if (hiddenTa) {
                hiddenTa.value = jsonStr;
                hiddenTa.dispatchEvent(new Event('input', { bubbles: true }));
            }
            return jsonStr;
        }
    };

    window.getRtInteractiveDrawData = function() {
        if (window.CustomCanvasControllerRT) {
            return window.CustomCanvasControllerRT.syncGradioPayload();
        }
        return window.__rt_interactive_payload__ || "{}";
    };

    const boot = () => {
        const root = document.getElementById('llmog-custom-canvas-app-rt');
        if (!root) { setTimeout(boot, 200); return; }
        if (window.__rt_canvas_booted_root__ === root) return;
        window.__rt_canvas_booted_root__ = root;
        window.CustomCanvasControllerRT.init();
    };
    setTimeout(boot, 50);
})();
"""


def _load_sample_bridge_rt():
    """Server-side handler to deliver the sample image to the Real-Time Canvas."""
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


def build_realtime_interactive_tab() -> Dict[str, Any]:
    """Real-Time Interactive — twin 520px (camera canvas | recognition viewer), camera toolbar below canvas."""
    # ── Top twin: RT canvas stage (520) | recognition viewer — same level ──
    with gr.Row(equal_height=True, elem_classes=["draw-tab-row", "twin-screens-row"]):
        with gr.Column(scale=1, min_width=420, elem_classes=["batch-bottom-col"]):
            gr.HTML('<p class="section-label">🎥 Live Camera &amp; Object Annotation Canvas</p>')
            custom_canvas = gr.HTML(
                value=_RT_DRAW_CANVAS_HTML,
                js_on_load=_RT_DRAW_CANVAS_JS,
                elem_id="rt-interactive-canvas-html",
            )
        with gr.Column(scale=1, min_width=420, elem_classes=["batch-bottom-col"]):
            gr.HTML('<p class="section-label">👁️ Recognition Result (Interactive)</p>')
            status = gr.Markdown("**Status: Idle – open camera, draw boxes over objects, then Recognize**")
            with gr.Group(elem_classes=["img-viewer-wrap"]):
                viewer = DetectionViewer(
                    label="Annotated Recognition Result",
                    panel_title="Recognized Camera Objects",
                    list_height=520,
                    elem_id="rt-interactive-viewer",
                )
            results = gr.HTML(value=_RECLS_EMPTY_TABLE)
            with gr.Accordion("YOLO Labels (<class_id> <xc> <yc> <w> <h>)", open=False):
                yolo = gr.Textbox(
                    lines=8,
                    interactive=False,
                    label="Copy these lines into the image's .txt label file",
                )
    # ── Below twin: camera toolbar card (same IDs — JS binds globally) ──
    toolbar_view = gr.HTML(value=_RT_TOOLBAR_HTML, elem_id="rt-interactive-toolbar-html")
    payload_box = gr.Textbox(value="{}", visible=False, elem_id="rt_interactive_payload_box")
    sample_bridge_btn = gr.Button("Sample Bridge", visible=False, elem_id="rt_sample_bridge_btn")
    with gr.Row(elem_classes=["btn-group"]):
        run_btn = gr.Button("🔎  Recognize Drawn Objects", variant="primary", scale=2)
        clear_btn = gr.Button("🗑️ Clear Results", variant="secondary", scale=1)
    with gr.Accordion("🎯 Target Classes & Detection Mode", open=False):
        class_mode = gr.Radio(
            label="🎯 Class Expectation Mode",
            choices=[
                ("🔒 Strict (Closed-Set)", "strict"),
                ("🔀 Hybrid (Extendable)", "hybrid"),
                ("🌐 Free (Open-World)", "free"),
            ],
            value="free",
            info="Free: Agent autonomously identifies and names whatever objects are drawn.",
        )
        preset_dropdown = gr.Dropdown(
            label="📋 Category Domain Presets",
            choices=list(CATEGORY_PRESETS.keys()),
            value="General Objects (COCO)",
            visible=False,
            info="Quickly load target classes & expert distinguishing definitions.",
        )
        classes_input = gr.Textbox(
            label="Domain / Focus Hint (Optional)",
            placeholder="e.g. Focus on defects, wildlife, tools, packaging... (or leave blank)",
            value="",
            lines=2,
            info="Free Mode: Agent autonomously identifies and names whatever objects are drawn.",
        )
        defs_input = gr.Textbox(
            label="Domain Guidance (Optional)",
            placeholder="Optional domain context or special inspection criteria...",
            lines=4,
            value="",
            info="Optional domain guidance.",
        )
    with gr.Accordion("⚙️ Advanced Filter & Context Settings", open=False):
        conf_threshold = gr.Slider(
            label="Minimum Confidence Threshold (%)",
            minimum=0, maximum=100, step=5, value=20,
            info="Omit or flag recognitions with confidence below this threshold in YOLO outputs.",
        )
        padding_slider = gr.Slider(
            label="Region Context Padding (%)",
            minimum=0, maximum=50, step=1, value=10,
            info="Extra visual context around each drawn region sent to the VLM.",
        )
        request_mode = gr.Radio(
            label="⚡ Request Mode (optional)",
            choices=[
                ("Sequential – 1 request per region", "sequential"),
                ("Parallel – asyncio.gather concurrent", "parallel"),
                ("Batched – single request with N images", "batched"),
            ],
            value="parallel",
            info="Parallel: N concurrent requests (~N× faster). Batched: 1 request with N crops.",
        )

    return dict(
        custom_canvas=custom_canvas,
        toolbar_view=toolbar_view,
        payload_box=payload_box,
        sample_bridge_btn=sample_bridge_btn,
        class_mode=class_mode,
        preset_dropdown=preset_dropdown,
        classes_input=classes_input,
        defs_input=defs_input,
        conf_threshold=conf_threshold,
        padding_slider=padding_slider,
        request_mode=request_mode,
        run_btn=run_btn,
        clear_btn=clear_btn,
        status=status,
        viewer=viewer,
        results=results,
        yolo=yolo,
    )


def wire_realtime_interactive_events(
    c_rt_interactive: Dict[str, Any], c_srv: Dict[str, Any]
) -> None:
    """Wire interactive camera drawing, preset selection, and recognition events."""
    # ── Category preset change ──────────────────────────────────────────────
    c_rt_interactive["preset_dropdown"].change(
        fn=_on_realtime_interactive_preset_change,
        inputs=[c_rt_interactive["preset_dropdown"]],
        outputs=[c_rt_interactive["classes_input"], c_rt_interactive["defs_input"]],
    )

    # ── Class expectation mode change ───────────────────────────────────────
    c_rt_interactive["class_mode"].change(
        fn=_on_realtime_interactive_mode_change,
        inputs=[c_rt_interactive["class_mode"]],
        outputs=[
            c_rt_interactive["classes_input"],
            c_rt_interactive["defs_input"],
            c_rt_interactive["preset_dropdown"],
        ],
    )

    # ── Sample bridge handler ───────────────────────────────────────────────
    c_rt_interactive["sample_bridge_btn"].click(
        fn=_load_sample_bridge_rt,
        inputs=None,
        outputs=[c_rt_interactive["payload_box"]],
        js="() => { if (window.CustomCanvasControllerRT) { setTimeout(() => { const ta = document.querySelector('#rt_interactive_payload_box textarea'); if (ta && ta.value) { try { const p = JSON.parse(ta.value); if (p.background) window.CustomCanvasControllerRT.loadImageFromDataUrl(p.background); } catch(e){} } }, 300); } }",
    )

    # ── Run recognition on drawn camera regions ─────────────────────────────
    c_rt_interactive["run_btn"].click(
        fn=classify_regions_gui,
        inputs=[
            c_rt_interactive["payload_box"],
            c_rt_interactive["classes_input"],
            c_rt_interactive["defs_input"],
            c_rt_interactive["padding_slider"],
            c_rt_interactive["class_mode"],
            c_rt_interactive["conf_threshold"],
            c_srv["use_external_api_chk"],
            c_srv["ext_api_url"],
            c_srv["ext_api_key"],
            c_srv["ext_model_name"],
            c_srv["server_port_input"],
            c_rt_interactive["request_mode"],
        ],
        outputs=[
            c_rt_interactive["status"],
            c_rt_interactive["viewer"],
            c_rt_interactive["results"],
            c_rt_interactive["yolo"],
        ],
        js="(p,c,d,pad,mode,conf,useExt,url,key,model,port,reqMode)=>{ const fresh=(window.getRtInteractiveDrawData?window.getRtInteractiveDrawData():p)||p; return [fresh,c,d,pad,mode,conf,useExt,url,key,model,port,reqMode]; }",
        api_name="classify_realtime_regions",
        concurrency_limit=1,
    )

    # ── Clear results ───────────────────────────────────────────────────────
    c_rt_interactive["clear_btn"].click(
        fn=lambda: (
            "**Status: Idle – open camera, draw boxes over objects, then Recognize**",
            None,
            _RECLS_EMPTY_TABLE,
            "",
        ),
        inputs=None,
        outputs=[
            c_rt_interactive["status"],
            c_rt_interactive["viewer"],
            c_rt_interactive["results"],
            c_rt_interactive["yolo"],
        ],
    )