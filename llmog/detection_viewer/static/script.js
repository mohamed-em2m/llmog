"use strict";

    var MAX_CANVAS_HEIGHT = 600;
    var KEYPOINT_RADIUS = 3;
    var HIT_RADIUS = 10;
    var MASK_ALPHA = 0.4;
    var CONNECTION_ALPHA = 0.7;
    var CONNECTION_WIDTH = 2;
    var BBOX_LINE_WIDTH = 2;
    var BBOX_LABEL_FONT = "bold 11px -apple-system, BlinkMacSystemFont, sans-serif";
    var DIM_ALPHA = 0.2;
    var MIN_ZOOM = 1;
    var MAX_ZOOM = 20;
    var ZOOM_SENSITIVITY = 0.001;
    var DRAG_THRESHOLD = 4;

    // ── DOM References ─────────────────────────────────────────────

    var container = element.querySelector(".pose-viewer-container");
    var canvasWrapper = element.querySelector(".canvas-wrapper");
    var canvas = element.querySelector("canvas");
    var ctx = canvas.getContext("2d");
    var tooltip = element.querySelector(".tooltip");
    var controlPanel = element.querySelector(".control-panel");
    var annotationList = element.querySelector(".annotation-list");
    var controlPanelBody = element.querySelector(".control-panel-body");
    var annotationDropdownBtn = element.querySelector(".annotation-dropdown-btn");

    // Read configurable defaults from props (available in js_on_load scope)
    var initialScoreThresholdMin = props.score_threshold_min || 0;
    var initialScoreThresholdMax = props.score_threshold_max != null ? props.score_threshold_max : 1;
    var initialKeypointThreshold = props.keypoint_threshold || 0;
    var initialKeypointRadius = props.keypoint_radius >= 1 ? props.keypoint_radius : KEYPOINT_RADIUS;
    var toggleImageBtn = element.querySelector(".toggle-image-btn");
    var resetBtn = element.querySelector(".reset-btn");
    var maximizeBtn = element.querySelector(".maximize-btn");
    var helpBtn = element.querySelector(".help-btn");
    var helpOverlay = element.querySelector(".help-overlay");
    var helpCloseBtn = element.querySelector(".help-close-btn");
    var loadingIndicator = element.querySelector(".loading-indicator");
    var placeholder = element.querySelector(".placeholder");
    var countEl = element.querySelector(".control-panel-count");

    // ── Ancestor Opacity Pinning ────────────────────────────────────
    // Gradio's "pending" class sets opacity < 1 on ancestors, creating a
    // stacking context that traps z-index.  Pin ancestors to opacity: 1
    // while the maximized overlay is active.

    var _opacityPinnedEls = [];

    function pinAncestorOpacity() {
        _opacityPinnedEls.forEach(function (el) {
            el.style.removeProperty("opacity");
        });
        _opacityPinnedEls = [];
        if (!state.maximized) return;
        var el = element.parentElement;
        while (el && el !== document.documentElement) {
            el.style.setProperty("opacity", "1", "important");
            _opacityPinnedEls.push(el);
            el = el.parentElement;
        }
    }

    // ── State ──────────────────────────────────────────────────────

    var state = {
        image: null,
        annotations: [],
        scale: 1,
        visibility: [],
        selectedIndex: -1,
        showImage: true,
        layers: { masks: true, boxes: true, skeleton: true, keypoints: true },
        thresholdMin: initialScoreThresholdMin,
        thresholdMax: initialScoreThresholdMax,
        keypointThreshold: initialKeypointThreshold,
        keypointRadius: initialKeypointRadius,
        connectionWidth: CONNECTION_WIDTH,
        maskAlpha: MASK_ALPHA,
        connectionAlpha: CONNECTION_ALPHA,
        bboxLineWidth: BBOX_LINE_WIDTH,
        labelVisibility: {},
        maskImages: [],
        zoom: 1,
        panX: 0,
        panY: 0,
        isPanning: false,
        panStartX: 0,
        panStartY: 0,
        panStartPanX: 0,
        panStartPanY: 0,
        didDrag: false,
        maximized: false,
        sortMode: "none",
        sortedIndices: [],
        expandedIndex: -1,
        drawOptionsOpen: false,
        annotationDropdownOpen: false
    };
    var latestValueRequestId = 0;

    // ── Watch for Python-initiated value changes ──────────────────
    // The watch() function (from Gradio's gr.HTML) fires only when
    // the backend updates props — never on JS-initiated changes.

    watch("value", function () {
        handleValueChange();
    });

    // Re-assert JS-managed CSS classes after Gradio's DOM morphing
    // wipes them back to template defaults.  The "maximized" class lives
    // on `element` (outside morph scope), so it survives automatically.
    function syncUIState() {
        toggleImageBtn.classList.toggle("active", state.showImage);
        if (state.image) {
            canvasWrapper.classList.add("visible");
        }
    }
    var containerObserver = new MutationObserver(function () {
        syncUIState();
    });
    containerObserver.observe(container, {
        attributes: true, attributeFilter: ["class"]
    });

    // ── Annotation List Dropdown (minimal space) ───────────────────
    function syncDropdownButton() {
        var isOpen = !!state.annotationDropdownOpen;
        if (annotationDropdownBtn) {
            annotationDropdownBtn.setAttribute("aria-expanded", isOpen ? "true" : "false");
            annotationDropdownBtn.classList.toggle("active", isOpen);
            // Keep label compact: List ▾ / List ▴  + count handled via countEl
            annotationDropdownBtn.textContent = isOpen ? "List \u25B2" : "List \u25BC";
        }
        element.classList.toggle("annotation-dropdown-open", isOpen);
        if (container) container.classList.toggle("annotation-dropdown-open", isOpen);
        if (controlPanel) controlPanel.classList.toggle("annotation-dropdown-open", isOpen);
        if (controlPanelBody) {
            // Ensure body visibility is forced even if CSS specificity fails (inline fallback)
            controlPanelBody.style.display = isOpen ? "block" : "none";
        }
    }

    function setAnnotationDropdown(open) {
        state.annotationDropdownOpen = !!open;
        syncDropdownButton();
    }

    if (annotationDropdownBtn) {
        annotationDropdownBtn.addEventListener("click", function (e) {
            e.stopPropagation();
            // Don't open empty list
            if (state.annotations.length === 0) return;
            setAnnotationDropdown(!state.annotationDropdownOpen);
        });
    }

    // Click outside closes dropdown
    document.addEventListener("click", function (e) {
        if (!state.annotationDropdownOpen) return;
        if (element.contains(e.target)) return;
        setAnnotationDropdown(false);
    });

    // Escape closes dropdown before other handlers
    element.addEventListener("keydown", function (e) {
        if (e.key === "Escape" && state.annotationDropdownOpen) {
            setAnnotationDropdown(false);
            e.stopPropagation();
        }
    });

    // ── Event Delegation on annotationList ──────────────────────────
    // One-time setup: delegate click, dblclick, input, and change
    // events so that renderControlPanel() never re-binds listeners.

    annotationList.addEventListener("click", function (e) {
        var target = e.target;

        // Ignore clicks on checkboxes (handled by change delegation)
        if (target.closest(".ann-checkbox") || target.closest(".select-all-checkbox")) {
            return;
        }

        var layerBtn = target.closest(".layer-btn");
        if (layerBtn) {
            var layer = layerBtn.getAttribute("data-layer");
            if (layer && state.layers.hasOwnProperty(layer)) {
                state.layers[layer] = !state.layers[layer];
                render();
                renderControlPanel();
            }
            return;
        }

        var labelBtn = target.closest(".label-filter-btn");
        if (labelBtn) {
            handleLabelFilterClick(labelBtn);
            return;
        }

        var sortBtn = target.closest(".sort-btn");
        if (sortBtn) {
            handleSortToggle(sortBtn);
            return;
        }

        var drawToggle = target.closest(".draw-options-toggle");
        if (drawToggle) {
            var section = drawToggle.closest(".draw-options");
            section.classList.toggle("open");
            state.drawOptionsOpen = section.classList.contains("open");
            return;
        }

        var expandBtn = target.closest(".ann-expand");
        if (expandBtn) {
            var idx = parseInt(expandBtn.getAttribute("data-index"), 10);
            state.expandedIndex = state.expandedIndex === idx ? -1 : idx;
            renderControlPanel();
            return;
        }

        var row = target.closest(".annotation-row");
        if (row) {
            var idx = parseInt(row.getAttribute("data-index"), 10);
            state.selectedIndex = state.selectedIndex === idx ? -1 : idx;
            render();
            renderControlPanel();
        }
    });

    annotationList.addEventListener("dblclick", function (e) {
        var labelBtn = e.target.closest(".label-filter-btn");
        if (labelBtn) {
            e.preventDefault();
            handleLabelFilterDblClick(labelBtn);
        }
    });

    annotationList.addEventListener("input", function (e) {
        var target = e.target;
        if (target.classList.contains("threshold-slider-min")) handleThresholdMinChange(e);
        else if (target.classList.contains("threshold-slider-max")) handleThresholdMaxChange(e);
        else if (target.classList.contains("keypoint-threshold-slider")) handleKeypointThresholdChange(e);
        else if (target.classList.contains("mask-alpha-slider")) handleMaskAlphaChange(e);
        else if (target.classList.contains("keypoint-radius-slider")) handleKeypointRadiusChange(e);
        else if (target.classList.contains("connection-width-slider")) handleConnectionWidthChange(e);
        else if (target.classList.contains("connection-alpha-slider")) handleConnectionAlphaChange(e);
        else if (target.classList.contains("bbox-line-width-slider")) handleBboxLineWidthChange(e);
    });

    annotationList.addEventListener("change", function (e) {
        var target = e.target;
        if (target.classList.contains("select-all-checkbox")) {
            var newVal = target.checked;
            for (var i = 0; i < state.visibility.length; i++) {
                state.visibility[i] = newVal;
            }
            render();
            renderControlPanel();
            return;
        }
        if (target.classList.contains("ann-checkbox")) {
            var idx = parseInt(target.getAttribute("data-index"), 10);
            state.visibility[idx] = target.checked;
            render();
            updateHeaderStats();
        }
    });

    // Handle initial value (watch doesn't fire at mount)
    if (props.value) {
        handleValueChange();
    }

    // ── Value Change Handler ───────────────────────────────────────

    function handleValueChange() {
        latestValueRequestId += 1;
        var requestId = latestValueRequestId;
        var raw = typeof props.value === "string" ? props.value.trim() : "";
        if (!raw || raw === "null") {
            showPlaceholder();
            return;
        }

        var data;
        try {
            data = JSON.parse(raw);
        } catch (e) {
            showPlaceholder();
            return;
        }

        if (!data || !data.image) {
            showPlaceholder();
            return;
        }

        // Update score threshold if provided in payload
        if (data.scoreThresholdMin != null) {
            state.thresholdMin = data.scoreThresholdMin;
        } else {
            state.thresholdMin = initialScoreThresholdMin;
        }
        if (data.scoreThresholdMax != null) {
            state.thresholdMax = data.scoreThresholdMax;
        } else {
            state.thresholdMax = initialScoreThresholdMax;
        }

        // Show loading spinner while the image is being fetched.
        showLoading();
        syncUIState();
        pinAncestorOpacity();

        var img = new Image();
        img.onload = function () {
            if (requestId !== latestValueRequestId) return;
            state.image = img;
            state.annotations = data.annotations || [];
            state.visibility = [];
            for (var i = 0; i < state.annotations.length; i++) {
                state.visibility.push(true);
            }
            state.selectedIndex = -1;

            // Build label visibility map
            state.labelVisibility = {};
            for (var i = 0; i < state.annotations.length; i++) {
                var lbl = state.annotations[i].label;
                if (lbl && !state.labelVisibility.hasOwnProperty(lbl)) {
                    state.labelVisibility[lbl] = true;
                }
            }

            // Decode RLE masks to offscreen canvases
            state.maskImages = [];
            for (var i = 0; i < state.annotations.length; i++) {
                if (state.annotations[i].mask) {
                    state.maskImages[i] = createMaskCanvas(state.annotations[i].mask, state.annotations[i].color);
                } else {
                    state.maskImages[i] = null;
                }
            }

            showContent();
            syncUIState();
            state.sortMode = getDefaultSortMode();
            requestAnimationFrame(function () {
                fitCanvas();
                state.zoom = 1;
                state.panX = 0;
                state.panY = 0;
                render();
                renderControlPanel();
            });
        };
        img.src = data.image;
    }

    // ── Display States: placeholder / loading / content ──────────

    function showPlaceholder() {
        if (state.maximized) {
            state.maximized = false;
            element.classList.remove("maximized");
            pinAncestorOpacity();
        }
        document.body.style.overflow = "";
        state.image = null;
        state.annotations = [];
        state.visibility = [];
        state.selectedIndex = -1;
        state.expandedIndex = -1;
        state.maskImages = [];
        state.labelVisibility = {};
        canvasWrapper.classList.remove("visible");
        loadingIndicator.classList.remove("visible");
        controlPanel.classList.remove("visible");
        tooltip.classList.remove("visible");
        helpOverlay.classList.remove("visible");
        placeholder.classList.remove("hidden");
        setAnnotationDropdown(false);
        if (annotationDropdownBtn) annotationDropdownBtn.style.display = "none";
        element.classList.remove("annotation-dropdown-open");
    }

    function showLoading() {
        placeholder.classList.add("hidden");
        canvasWrapper.classList.remove("visible");
        controlPanel.classList.remove("visible");
        loadingIndicator.classList.add("visible");
        setAnnotationDropdown(false);
    }

    function showContent() {
        placeholder.classList.add("hidden");
        loadingIndicator.classList.remove("visible");
        canvasWrapper.classList.add("visible");
        if (annotationDropdownBtn) annotationDropdownBtn.style.display = "";
    }

    // ── Canvas Sizing ─────────────────────────────────────────────

    function fitCanvas() {
        if (!state.image) return;

        var img = state.image;
        var newW, newH;

        if (state.maximized) {
            var wrapperW = canvasWrapper.clientWidth || window.innerWidth;
            var wrapperH = canvasWrapper.clientHeight || window.innerHeight;

            var w = wrapperW;
            var h = img.naturalHeight * (w / img.naturalWidth);

            if (h > wrapperH) {
                h = wrapperH;
                w = img.naturalWidth * (h / img.naturalHeight);
            }

            newW = Math.round(w);
            newH = Math.round(h);
        } else {
            var maxWidth = canvasWrapper.clientWidth || 800;

            var w = maxWidth;
            var h = img.naturalHeight * (w / img.naturalWidth);

            if (h > MAX_CANVAS_HEIGHT) {
                h = MAX_CANVAS_HEIGHT;
                w = img.naturalWidth * (h / img.naturalHeight);
            }

            newW = Math.round(w);
            newH = Math.round(h);
        }

        // Guard: assigning the same value clears canvas content per HTML5 spec
        if (canvas.width !== newW || canvas.height !== newH) {
            canvas.width = newW;
            canvas.height = newH;
        }
        state.scale = canvas.width / img.naturalWidth;
    }

    // ── Zoom/Pan Helpers ────────────────────────────────────────────

    function clientToCanvas(clientX, clientY) {
        var rect = canvas.getBoundingClientRect();
        var cssX = (clientX - rect.left) * (canvas.width / rect.width);
        var cssY = (clientY - rect.top) * (canvas.height / rect.height);
        return {
            x: (cssX - state.panX) / state.zoom,
            y: (cssY - state.panY) / state.zoom
        };
    }

    function clampPan() {
        if (state.zoom <= 1) {
            state.panX = 0;
            state.panY = 0;
            return;
        }
        var maxPanX = 0;
        var minPanX = canvas.width - canvas.width * state.zoom;
        var maxPanY = 0;
        var minPanY = canvas.height - canvas.height * state.zoom;
        if (state.panX > maxPanX) state.panX = maxPanX;
        if (state.panX < minPanX) state.panX = minPanX;
        if (state.panY > maxPanY) state.panY = maxPanY;
        if (state.panY < minPanY) state.panY = minPanY;
    }

    function zoomToCenter(newZoom) {
        newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));
        var cx = canvas.width / 2;
        var cy = canvas.height / 2;
        state.panX = cx - (cx - state.panX) * (newZoom / state.zoom);
        state.panY = cy - (cy - state.panY) * (newZoom / state.zoom);
        state.zoom = newZoom;
        clampPan();
        render();
    }

    function resetZoom() {
        state.zoom = 1;
        state.panX = 0;
        state.panY = 0;
        render();
    }

    function resetToDefaults() {
        if (!state.image) return;

        state.showImage = true;
        toggleImageBtn.classList.add("active");

        state.layers.masks = true;
        state.layers.boxes = true;
        state.layers.skeleton = true;
        state.layers.keypoints = true;

        state.thresholdMin = initialScoreThresholdMin;
        state.thresholdMax = initialScoreThresholdMax;
        state.keypointThreshold = initialKeypointThreshold;
        state.keypointRadius = initialKeypointRadius;
        state.connectionWidth = CONNECTION_WIDTH;
        state.maskAlpha = MASK_ALPHA;
        state.connectionAlpha = CONNECTION_ALPHA;
        state.bboxLineWidth = BBOX_LINE_WIDTH;

        for (var label in state.labelVisibility) {
            if (state.labelVisibility.hasOwnProperty(label)) {
                state.labelVisibility[label] = true;
            }
        }
        for (var i = 0; i < state.visibility.length; i++) {
            state.visibility[i] = true;
        }

        state.selectedIndex = -1;
        state.expandedIndex = -1;
        state.sortMode = getDefaultSortMode();
        state.zoom = 1;
        state.panX = 0;
        state.panY = 0;
        canvas.style.cursor = "default";

        render();
        renderControlPanel();
    }

    // ── Rendering ─────────────────────────────────────────────────

    function isAnnotationVisible(i) {
        if (!state.visibility[i]) return false;
        var ann = state.annotations[i];
        if (ann.score != null && (ann.score < state.thresholdMin || ann.score > state.thresholdMax)) return false;
        if (ann.label && state.labelVisibility.hasOwnProperty(ann.label) && !state.labelVisibility[ann.label]) return false;
        return true;
    }

    function isKeypointVisible(kp) {
        if (kp.x == null || kp.y == null) return false;
        if (kp.confidence != null && kp.confidence < state.keypointThreshold) return false;
        return true;
    }

    function render() {
        if (!state.image) return;

        // Clear in screen space
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Dark background visible when zoomed/panned
        if (state.zoom > 1) {
            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        // Apply zoom+pan transform
        ctx.setTransform(state.zoom, 0, 0, state.zoom, state.panX, state.panY);

        if (state.showImage) {
            ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
        } else {
            ctx.fillStyle = "#2a2a2a";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        var hasSelection = state.selectedIndex >= 0;

        // Draw order: masks (back) → bbox → connections → keypoints (front)
        for (var pass = 0; pass < 4; pass++) {
            for (var i = 0; i < state.annotations.length; i++) {
                if (!isAnnotationVisible(i)) continue;

                var ann = state.annotations[i];
                var dim = hasSelection && i !== state.selectedIndex;

                ctx.save();
                if (dim) ctx.globalAlpha = DIM_ALPHA;

                if (pass === 0 && state.layers.masks) drawMask(i);
                else if (pass === 1 && state.layers.boxes) drawBbox(ann);
                else if (pass === 2 && state.layers.skeleton) drawConnections(ann);
                else if (pass === 3 && state.layers.keypoints) drawKeypoints(ann);

                ctx.restore();
            }
        }

        // Draw zoom indicator in screen space
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        if (state.zoom > 1) {
            var zoomText = Math.round(state.zoom * 100) + "%";
            ctx.font = "bold 12px -apple-system, BlinkMacSystemFont, sans-serif";
            var tm = ctx.measureText(zoomText);
            var px = canvas.width - tm.width - 12;
            var py = 8;
            ctx.fillStyle = "rgba(0,0,0,0.5)";
            ctx.beginPath();
            ctx.roundRect(px - 6, py - 2, tm.width + 12, 20, 4);
            ctx.fill();
            ctx.fillStyle = "#fff";
            ctx.textBaseline = "top";
            ctx.fillText(zoomText, px, py + 2);
        }
    }

    function drawMask(i) {
        var maskImg = state.maskImages[i];
        if (!maskImg) return;
        var baseAlpha = ctx.globalAlpha;
        ctx.globalAlpha = baseAlpha * state.maskAlpha;
        ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = baseAlpha;
    }

    function drawBbox(ann) {
        var bbox = ann.bbox;
        if (!bbox) return;

        var x = bbox.x * state.scale;
        var y = bbox.y * state.scale;
        var w = bbox.width * state.scale;
        var h = bbox.height * state.scale;
        var baseAlpha = ctx.globalAlpha;
        var iz = 1 / state.zoom;

        // Bauhaus geometric bbox — crisp stroke with subtle outer glow for contrast
        ctx.save();
        ctx.strokeStyle = ann.color;
        ctx.lineWidth = state.bboxLineWidth * iz;
        ctx.shadowColor = "rgba(0,0,0,0.25)";
        ctx.shadowBlur = 4 * iz;
        // Rounded bbox for softer Bauhaus feel (fallback to rect if roundRect unavailable)
        if (ctx.roundRect) {
            ctx.beginPath();
            ctx.roundRect(x, y, w, h, 3 * iz);
            ctx.stroke();
        } else {
            ctx.strokeRect(x, y, w, h);
        }
        ctx.restore();
        // Inner thin white outline for visibility on dark images
        ctx.save();
        ctx.strokeStyle = "rgba(255,255,255,0.6)";
        ctx.lineWidth = 0.7 * iz;
        if (ctx.roundRect) {
            ctx.beginPath();
            ctx.roundRect(x, y, w, h, 3 * iz);
            ctx.stroke();
        }
        ctx.restore();

        // Label + score — formatted, human-friendly
        var rawLabel = ann.label || "";
        // Humanize: tennis_racket -> Tennis Racket,  person -> Person
        var displayLabel = rawLabel.replace(/[_-]+/g, " ").replace(/\b\w/g, function(c){return c.toUpperCase();});
        var labelText = displayLabel;
        if (ann.score != null) {
            var pct = (ann.score * 100).toFixed(0);
            labelText += " " + pct + "%";
        }
        if (labelText) {
            var fontSize = 12 * iz;
            ctx.font = "800 " + fontSize + "px 'Inter','Poppins',system-ui,sans-serif";
            var textMetrics = ctx.measureText(labelText);
            var padX = 7 * iz;
            var padY = 4 * iz;
            var textH = 15 * iz;
            // Allow label to be wider than bbox so text is never sliced by bbox width.
            // Only clamp to canvas width, not bbox width.
            var desiredW = textMetrics.width + padX * 2;
            var maxCanvasW = canvas.width - 4 * iz;
            var bgW = desiredW > maxCanvasW ? maxCanvasW : desiredW;
            var bgH = textH + padY;
            var labelAbove = y - bgH - 2*iz >= 0;
            var labelBelowFits = y + h + bgH + 2*iz <= canvas.height;
            var bgX = x;
            // Keep label fully inside canvas horizontally (allow overflow beyond bbox)
            if (bgX + bgW > canvas.width - 2*iz) bgX = Math.max(2*iz, canvas.width - bgW - 2*iz);
            if (bgX < 2*iz) bgX = 2*iz;
            var bgY;
            if (labelAbove) {
                bgY = y - bgH - 2*iz;
            } else if (labelBelowFits) {
                // If no space above, try below bbox (better than inside tiny box)
                bgY = y + h + 2*iz;
            } else {
                bgY = y + 2*iz;
                // If bbox is tiny and label would overflow bbox bottom, center or clamp
                if (bgY + bgH > y + h && h < bgH + 8*iz) {
                    // Center vertically relative to bbox but keep inside canvas
                    bgY = y + (h - bgH)/2;
                    if (bgY < 2*iz) bgY = 2*iz;
                    if (bgY + bgH > canvas.height - 2*iz) bgY = Math.max(2*iz, canvas.height - bgH - 2*iz);
                }
            }
            var r = 7 * iz;
            ctx.save();
            // Pill background — Bauhaus: solid color with navy text for yellow, white for others
            var isYellow = ann.color.toLowerCase() === "#f5c842" || ann.color.toLowerCase() === "#f5c842";
            ctx.fillStyle = ann.color;
            ctx.globalAlpha = baseAlpha * 0.92;
            if (ctx.roundRect) {
                ctx.beginPath();
                ctx.roundRect(bgX, bgY, bgW, bgH, r);
                ctx.fill();
            } else {
                ctx.fillRect(bgX, bgY, bgW, bgH);
            }
            // Subtle border
            ctx.globalAlpha = baseAlpha * 0.9;
            ctx.strokeStyle = "rgba(26,17,69,0.12)";
            ctx.lineWidth = 1 * iz;
            if (ctx.roundRect) {
                ctx.beginPath();
                ctx.roundRect(bgX, bgY, bgW, bgH, r);
                ctx.stroke();
            }
            ctx.globalAlpha = baseAlpha;
            ctx.fillStyle = (isYellow || ann.color === "#F5C842") ? "#1A1A1A" : "#FFFFFF";
            ctx.textBaseline = "middle";
            ctx.textAlign = "left";
            // Truncate if still too wide
            var maxTextW = bgW - padX*2;
            var finalText = labelText;
            if (textMetrics.width > maxTextW) {
                while (finalText.length > 3 && ctx.measureText(finalText + "…").width > maxTextW) {
                    finalText = finalText.slice(0, -2);
                }
                finalText += "…";
            }
            ctx.fillText(finalText, bgX + padX, bgY + bgH/2 + 0.5*iz);
            ctx.restore();
        }
    }

    function drawConnections(ann) {
        var kps = ann.keypoints;
        var conns = ann.connections;
        if (!conns || conns.length === 0 || !kps || kps.length === 0) return;

        var baseAlpha = ctx.globalAlpha;
        ctx.strokeStyle = ann.color;
        ctx.lineWidth = state.connectionWidth / state.zoom;
        ctx.globalAlpha = baseAlpha * state.connectionAlpha;

        for (var i = 0; i < conns.length; i++) {
            var idxA = conns[i][0];
            var idxB = conns[i][1];
            if (idxA < 0 || idxA >= kps.length || idxB < 0 || idxB >= kps.length) continue;

            var a = kps[idxA];
            var b = kps[idxB];

            if (!isKeypointVisible(a) || !isKeypointVisible(b)) continue;

            ctx.beginPath();
            ctx.moveTo(a.x * state.scale, a.y * state.scale);
            ctx.lineTo(b.x * state.scale, b.y * state.scale);
            ctx.stroke();
        }

        ctx.globalAlpha = baseAlpha;
    }

    function drawKeypoints(ann) {
        var kps = ann.keypoints;
        if (!kps || kps.length === 0) return;

        var iz = 1 / state.zoom;
        for (var i = 0; i < kps.length; i++) {
            var kp = kps[i];
            if (!isKeypointVisible(kp)) continue;

            var cx = kp.x * state.scale;
            var cy = kp.y * state.scale;

            // White border
            ctx.beginPath();
            ctx.arc(cx, cy, (state.keypointRadius + 1) * iz, 0, 2 * Math.PI);
            ctx.fillStyle = "#ffffff";
            ctx.fill();

            // Colored fill
            ctx.beginPath();
            ctx.arc(cx, cy, state.keypointRadius * iz, 0, 2 * Math.PI);
            ctx.fillStyle = ann.color;
            ctx.fill();
        }
    }

    // ── RLE Decode ──────────────────────────────────────────────────

    function createMaskCanvas(rle, colorHex) {
        var counts = rle.counts;
        var h = rle.size[0], w = rle.size[1];
        var r = parseInt(colorHex.slice(1, 3), 16);
        var g = parseInt(colorHex.slice(3, 5), 16);
        var b = parseInt(colorHex.slice(5, 7), 16);
        var offscreen = document.createElement("canvas");
        offscreen.width = w;
        offscreen.height = h;
        var offCtx = offscreen.getContext("2d");
        var imageData = offCtx.createImageData(w, h);
        var data = imageData.data;
        // RLE is column-major (COCO format), convert to row-major ImageData
        var pos = 0;
        for (var i = 0; i < counts.length; i++) {
            var c = counts[i];
            if (i % 2 === 1) {
                var end = pos + c;
                for (var j = pos; j < end; j++) {
                    var row = j % h;
                    var col = (j / h) | 0;
                    var idx = (row * w + col) * 4;
                    data[idx] = r;
                    data[idx + 1] = g;
                    data[idx + 2] = b;
                    data[idx + 3] = 255;
                }
            }
            pos += c;
        }
        offCtx.putImageData(imageData, 0, 0);
        return offscreen;
    }

    // ── Helpers ─────────────────────────────────────────────────────

    function buildAnnotationSummary(ann) {
        // Bauhaus: prioritize score, then geometry
        if (ann.score != null) {
            return (ann.score * 100).toFixed(0) + "%";
        }
        var parts = [];
        if (ann.mask) {
            parts.push("mask");
        }
        if (ann.bbox) {
            var area = ann.bbox.width * ann.bbox.height;
            if (area > 0) {
                var size = area > 50000 ? "Large" : area > 10000 ? "Medium" : "Small";
                parts.push(size);
            } else {
                parts.push("bbox");
            }
        }
        var kps = ann.keypoints || [];
        if (kps.length > 0) {
            var validCount = 0;
            for (var j = 0; j < kps.length; j++) {
                if (isKeypointVisible(kps[j])) validCount++;
            }
            parts.push(validCount + " pts");
        }
        if (ann.score != null) {
            parts.push((ann.score * 100).toFixed(1) + "%");
        }
        return parts.join(", ") || "empty";
    }

    function escapeHtml(text) {
        var div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function humanizeLabel(label) {
        if (!label) return "";
        return label.replace(/[_-]+/g, " ").replace(/\b\w/g, function(c){ return c.toUpperCase(); });
    }

    function getLabelColor(label) {
        // Return the color of the first annotation with this label
        for (var i = 0; i < state.annotations.length; i++) {
            if (state.annotations[i].label === label) {
                return state.annotations[i].color;
            }
        }
        return "#888";
    }

    function getDefaultSortMode() {
        var hasScores = false;
        var hasBboxes = false;
        for (var i = 0; i < state.annotations.length; i++) {
            if (state.annotations[i].score != null) hasScores = true;
            if (state.annotations[i].bbox) hasBboxes = true;
            if (hasScores && hasBboxes) break;
        }
        if (hasScores) return "score-desc";
        if (hasBboxes) return "size-desc";
        return "none";
    }

    function bboxArea(ann) {
        if (!ann.bbox) return null;
        return ann.bbox.width * ann.bbox.height;
    }

    function computeSortedIndices() {
        var n = state.annotations.length;
        var indices = [];
        for (var i = 0; i < n; i++) indices.push(i);

        var mode = state.sortMode;
        if (mode === "score-desc" || mode === "score-asc") {
            var dir = mode === "score-desc" ? -1 : 1;
            indices.sort(function (a, b) {
                var sa = state.annotations[a].score;
                var sb = state.annotations[b].score;
                var hasA = sa != null;
                var hasB = sb != null;
                if (hasA && hasB) return (sa - sb) * dir || a - b;
                if (hasA) return -1;
                if (hasB) return 1;
                return a - b;
            });
        } else if (mode === "size-desc" || mode === "size-asc") {
            var dir = mode === "size-desc" ? -1 : 1;
            indices.sort(function (a, b) {
                var aa = bboxArea(state.annotations[a]);
                var ab = bboxArea(state.annotations[b]);
                var hasA = aa != null;
                var hasB = ab != null;
                if (hasA && hasB) return (aa - ab) * dir || a - b;
                if (hasA) return -1;
                if (hasB) return 1;
                return a - b;
            });
        }

        // Stable partition: visible first, hidden last (only when some label filter is OFF)
        var anyLabelFiltered = false;
        var labels = Object.keys(state.labelVisibility);
        for (var li = 0; li < labels.length; li++) {
            if (!state.labelVisibility[labels[li]]) { anyLabelFiltered = true; break; }
        }
        if (anyLabelFiltered) {
            var visible = [];
            var hidden = [];
            for (var j = 0; j < indices.length; j++) {
                var idx = indices[j];
                var ann = state.annotations[idx];
                var labelHidden = ann.label && state.labelVisibility.hasOwnProperty(ann.label) && !state.labelVisibility[ann.label];
                if (labelHidden) hidden.push(idx);
                else visible.push(idx);
            }
            indices = visible.concat(hidden);
        }

        state.sortedIndices = indices;
    }

    function handleSortToggle(btn) {
        var key = btn.getAttribute("data-sort-key");
        if (!key) return;
        // Toggle direction if same key is already active
        if (state.sortMode === key + "-desc") {
            state.sortMode = key + "-asc";
        } else if (state.sortMode === key + "-asc") {
            state.sortMode = key + "-desc";
        } else {
            state.sortMode = key + "-desc";
        }
        computeSortedIndices();
        renderControlPanel();
    }

    function updateHeaderStats() {
        if (!countEl) return;

        var total = state.annotations.length;
        var visibleCount = 0;
        var minScore = Infinity;
        var maxScore = -Infinity;
        var hasScores = false;

        for (var i = 0; i < state.annotations.length; i++) {
            if (!isAnnotationVisible(i)) continue;
            visibleCount++;
            var ann = state.annotations[i];
            if (ann.score != null) {
                hasScores = true;
                if (ann.score < minScore) minScore = ann.score;
                if (ann.score > maxScore) maxScore = ann.score;
            }
        }

        var text = visibleCount + " / " + total;
        if (hasScores && visibleCount > 0) {
            text += " \u00B7 " + (minScore * 100).toFixed(0) + "\u2013" + (maxScore * 100).toFixed(0) + "%";
        }
        countEl.textContent = text;
    }

    // ── Control Panel ─────────────────────────────────────────────

    function renderControlPanel() {
        if (state.annotations.length === 0) {
            controlPanel.classList.remove("visible");
            annotationList.innerHTML = "";
            setAnnotationDropdown(false);
            if (annotationDropdownBtn) annotationDropdownBtn.style.display = "none";
            return;
        }
        if (annotationDropdownBtn) annotationDropdownBtn.style.display = "";

        computeSortedIndices();

        var html = "";

        // Detect which layer types are present in annotations
        var availableLayers = [];
        var hasMasks = false, hasBoxes = false, hasSkeleton = false, hasKeypoints = false;
        for (var i = 0; i < state.annotations.length; i++) {
            var ann = state.annotations[i];
            if (ann.mask) hasMasks = true;
            if (ann.bbox) hasBoxes = true;
            if (ann.connections && ann.connections.length > 0) hasSkeleton = true;
            if (ann.keypoints && ann.keypoints.length > 0) hasKeypoints = true;
        }
        if (hasMasks) availableLayers.push("masks");
        if (hasBoxes) availableLayers.push("boxes");
        if (hasSkeleton) availableLayers.push("skeleton");
        if (hasKeypoints) availableLayers.push("keypoints");

        var layerLabels = { masks: "Masks", boxes: "Boxes", skeleton: "Skeleton", keypoints: "Keypoints" };

        // Only show layer toggles when 2+ types are present
        if (availableLayers.length >= 2) {
            html += '<div class="layer-toggles">';
            for (var i = 0; i < availableLayers.length; i++) {
                var layer = availableLayers[i];
                html += '<button class="layer-btn' + (state.layers[layer] ? ' active' : '') + '" data-layer="' + layer + '">' + layerLabels[layer] + '</button>';
            }
            html += '</div>';
        }

        // Label filters removed — clean viewer without label choosing per request

        // Sort controls
        var hasScoresForSort = false;
        var hasBboxesForSort = false;
        for (var si = 0; si < state.annotations.length; si++) {
            if (state.annotations[si].score != null) hasScoresForSort = true;
            if (state.annotations[si].bbox) hasBboxesForSort = true;
        }
        if (hasScoresForSort || hasBboxesForSort) {
            var sortButtons = [];
            if (hasScoresForSort) sortButtons.push("score");
            if (hasBboxesForSort) sortButtons.push("size");
            // Only show controls when 2+ sort options exist
            if (sortButtons.length >= 2) {
                var sortLabels = { score: "Score", size: "Size" };
                html += '<div class="sort-controls">';
                html += '<span class="sort-controls-title">Sort</span>';
                for (var sbi = 0; sbi < sortButtons.length; sbi++) {
                    var sk = sortButtons[sbi];
                    var isDesc = state.sortMode === sk + "-desc";
                    var isAsc = state.sortMode === sk + "-asc";
                    var isActive = isDesc || isAsc;
                    var arrow = isActive ? (isDesc ? " \u25BC" : " \u25B2") : "";
                    html += '<button class="sort-btn' + (isActive ? ' active' : '') + '" data-sort-key="' + sk + '">' + sortLabels[sk] + arrow + '</button>';
                }
                html += '</div>';
            }
        }

        // Dual-thumb score threshold slider
        html += '<div class="threshold-row">';
        html += '<span>Score</span>';
        html += '<div class="dual-range-wrapper">';
        html += '<input type="range" class="threshold-slider-min" min="0" max="100" value="' + Math.round(state.thresholdMin * 100) + '">';
        html += '<input type="range" class="threshold-slider-max" min="0" max="100" value="' + Math.round(state.thresholdMax * 100) + '">';
        html += '</div>';
        html += '<span class="threshold-value">' + Math.round(state.thresholdMin * 100) + '%\u2013' + Math.round(state.thresholdMax * 100) + '%</span>';
        html += '</div>';

        // Keypoint threshold slider (only when annotations have keypoints)
        if (hasKeypoints) {
            html += '<div class="threshold-row">';
            html += '<span>Keypoint &ge;</span>';
            html += '<input type="range" class="keypoint-threshold-slider" min="0" max="100" value="' + Math.round(state.keypointThreshold * 100) + '">';
            html += '<span class="keypoint-threshold-value">' + Math.round(state.keypointThreshold * 100) + '%</span>';
            html += '</div>';
        }

        // Draw options removed — clean viewer per request

        // Select-all row + scrollable annotation rows container
        html += '<div class="annotation-rows">';

        var allChecked = true;
        var anyChecked = false;
        for (var i = 0; i < state.visibility.length; i++) {
            if (state.visibility[i]) anyChecked = true;
            else allChecked = false;
        }
        html += '<div class="select-all-row">';
        html += '<input type="checkbox" class="select-all-checkbox"' + (anyChecked ? ' checked' : '') + '>';
        html += '<span class="select-all-label">All</span>';
        html += '</div>';

        // Annotation rows (iterate in sorted/grouped order)
        var anyLabelFiltered = false;
        var labelKeys = Object.keys(state.labelVisibility);
        for (var li = 0; li < labelKeys.length; li++) {
            if (!state.labelVisibility[labelKeys[li]]) { anyLabelFiltered = true; break; }
        }
        var separatorInserted = false;

        for (var si = 0; si < state.sortedIndices.length; si++) {
            var i = state.sortedIndices[si];
            var ann = state.annotations[i];
            var visible = state.visibility[i];
            var selected = state.selectedIndex === i;
            var summary = buildAnnotationSummary(ann);
            var outsideRange = ann.score != null && (ann.score < state.thresholdMin || ann.score > state.thresholdMax);
            var labelHidden = ann.label && state.labelVisibility.hasOwnProperty(ann.label) && !state.labelVisibility[ann.label];

            // Insert separator before the first label-hidden item
            if (anyLabelFiltered && labelHidden && !separatorInserted) {
                html += '<div class="annotation-group-separator"><span>Hidden</span></div>';
                separatorInserted = true;
            }

            var expanded = state.expandedIndex === i;

            var displayLabel = humanizeLabel(ann.label);
           html += '<div class="annotation-row' + (selected ? ' selected' : '') + (outsideRange ? ' below-threshold' : '') + (labelHidden ? ' filtered-out' : '') + '" data-index="' + i + '" style="display:flex;align-items:center;gap:6px;padding:2px 4px;font-size:11px;line-height:1.2;min-height:18px;">';
            html += '<span class="ann-dot" style="width:6px;height:6px;background:' + ann.color + '; box-shadow:0 0 0 2px ' + ann.color + '22"></span>';
            html += '<input type="checkbox" class="ann-checkbox" data-index="' + i + '" style="width:12px;height:12px;margin:0;"' + (visible ? ' checked' : '') + '>';
            html += '<span class="ann-label" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + escapeHtml(displayLabel) + '</span>';
            html += '<span class="ann-summary" style="opacity:.7;">' + escapeHtml(summary) + '</span>';
            html += '<button class="ann-expand' + (expanded ? ' expanded' : '') + '" data-index="' + i + '" style="width:14px;height:14px;padding:0;font-size:9px;">&#9654;</button>';
            html += '</div>';

            // Detail panel (shown when expand button is clicked)
            html += '<div class="annotation-detail' + (expanded ? ' visible' : '') + '" data-index="' + i + '">';
            html += buildDetailHtml(ann);
            html += '</div>';
        }

        html += '</div>'; // close .annotation-rows

        annotationList.innerHTML = html;
        controlPanel.classList.add("visible");
        updateHeaderStats();
        syncDropdownButton();

        // Initialize dual-range track highlight
        var dualWrapper = annotationList.querySelector(".dual-range-wrapper");
        if (dualWrapper) updateDualRangeTrack(dualWrapper);

        // Initialize single slider track fills
        var singleSliders = annotationList.querySelectorAll('.threshold-row input[type="range"]:not(.threshold-slider-min):not(.threshold-slider-max)');
        for (var i = 0; i < singleSliders.length; i++) {
            updateSliderTrack(singleSliders[i]);
        }

        // Set indeterminate state on select-all checkbox
        var selectAllCb = annotationList.querySelector(".select-all-checkbox");
        if (selectAllCb && anyChecked && !allChecked) {
            selectAllCb.indeterminate = true;
        }
    }

    function buildDetailHtml(ann) {
        var html = "";
        if (ann.bbox) {
            html += '<div class="detail-section-title">Bounding Box</div>';
            html += '<table>';
            html += '<tr><td>Position:</td><td>' + ann.bbox.x.toFixed(1) + ', ' + ann.bbox.y.toFixed(1) + '</td></tr>';
            html += '<tr><td>Size:</td><td>' + ann.bbox.width.toFixed(1) + ' &times; ' + ann.bbox.height.toFixed(1) + '</td></tr>';
            if (ann.score != null) {
                html += '<tr><td>Score:</td><td>' + (ann.score * 100).toFixed(1) + '%</td></tr>';
            }
            html += '</table>';
        }
        var kps = ann.keypoints || [];
        if (kps.length > 0) {
            html += '<div class="detail-section-title">Keypoints</div>';
            html += '<table>';
            for (var j = 0; j < kps.length; j++) {
                var kp = kps[j];
                var name = kp.name || ("kp" + j);
                var coords = (kp.x != null && kp.y != null) ? kp.x.toFixed(1) + ', ' + kp.y.toFixed(1) : "missing";
                var conf = kp.confidence != null ? (kp.confidence * 100).toFixed(1) + '%' : "";
                html += '<tr><td>' + escapeHtml(name) + '</td><td>' + coords + '</td><td>' + conf + '</td></tr>';
            }
            html += '</table>';
        }
        return html || '<span>No details available</span>';
    }

    // ── Event Handlers ─────────────────────────────────────────────

    var labelClickTimer = null;

    function handleLabelFilterClick(btn) {
        var label = btn.getAttribute("data-label");
        if (!label || !state.labelVisibility.hasOwnProperty(label)) return;
        if (labelClickTimer) clearTimeout(labelClickTimer);
        labelClickTimer = setTimeout(function () {
            labelClickTimer = null;
            state.labelVisibility[label] = !state.labelVisibility[label];
            render();
            renderControlPanel();
        }, 200);
    }

    function handleLabelFilterDblClick(btn) {
        if (labelClickTimer) {
            clearTimeout(labelClickTimer);
            labelClickTimer = null;
        }
        var label = btn.getAttribute("data-label");
        if (!label || !state.labelVisibility.hasOwnProperty(label)) return;

        // Check if this label is already solo (only this one is ON)
        var labels = Object.keys(state.labelVisibility);
        var onlyThisOn = labels.every(function (l) {
            return l === label ? state.labelVisibility[l] : !state.labelVisibility[l];
        });

        if (onlyThisOn) {
            // Unsolo: turn all labels ON
            for (var li = 0; li < labels.length; li++) {
                state.labelVisibility[labels[li]] = true;
            }
        } else {
            // Solo: turn only this label ON
            for (var li = 0; li < labels.length; li++) {
                state.labelVisibility[labels[li]] = (labels[li] === label);
            }
        }
        render();
        renderControlPanel();
    }

    function updateThresholdUI() {
        render();
        var label = annotationList.querySelector(".threshold-value");
        if (label) label.textContent = Math.round(state.thresholdMin * 100) + '%\u2013' + Math.round(state.thresholdMax * 100) + '%';
        // Update row opacity for outside-range items
        var rows = annotationList.querySelectorAll(".annotation-row");
        for (var i = 0; i < rows.length; i++) {
            var idx = parseInt(rows[i].getAttribute("data-index"), 10);
            var ann = state.annotations[idx];
            var outsideRange = ann.score != null && (ann.score < state.thresholdMin || ann.score > state.thresholdMax);
            if (outsideRange) {
                rows[i].classList.add("below-threshold");
            } else {
                rows[i].classList.remove("below-threshold");
            }
        }
        // Update track highlight
        var wrapper = annotationList.querySelector(".dual-range-wrapper");
        if (wrapper) updateDualRangeTrack(wrapper);
        updateHeaderStats();
    }

    function handleThresholdMinChange(e) {
        var val = parseInt(e.target.value, 10) / 100;
        state.thresholdMin = Math.min(val, state.thresholdMax);
        e.target.value = Math.round(state.thresholdMin * 100);
        updateThresholdUI();
    }

    function handleThresholdMaxChange(e) {
        var val = parseInt(e.target.value, 10) / 100;
        state.thresholdMax = Math.max(val, state.thresholdMin);
        e.target.value = Math.round(state.thresholdMax * 100);
        updateThresholdUI();
    }

    function updateSliderTrack(slider) {
        var min = parseFloat(slider.min) || 0;
        var max = parseFloat(slider.max) || 100;
        var val = parseFloat(slider.value) || 0;
        var pct = ((val - min) / (max - min)) * 100;
        var t = 'calc(50% - 2px)';
        var b = 'calc(50% + 2px)';
        slider.style.background =
            'linear-gradient(to bottom, transparent ' + t + ', var(--color-accent, #2196F3) ' + t + ', var(--color-accent, #2196F3) ' + b + ', transparent ' + b + ') 0 0 / ' + pct + '% 100% no-repeat, ' +
            'linear-gradient(to bottom, transparent ' + t + ', var(--border-color-primary, #d0d0d0) ' + t + ', var(--border-color-primary, #d0d0d0) ' + b + ', transparent ' + b + ')';
    }

    function updateDualRangeTrack(wrapper) {
        var minVal = Math.round(state.thresholdMin * 100);
        var maxVal = Math.round(state.thresholdMax * 100);
        wrapper.style.background = 'linear-gradient(to right, var(--border-color-primary, #d0d0d0) ' + minVal + '%, var(--color-accent, #2196F3) ' + minVal + '%, var(--color-accent, #2196F3) ' + maxVal + '%, var(--border-color-primary, #d0d0d0) ' + maxVal + '%)';
    }

    function handleKeypointThresholdChange(e) {
        state.keypointThreshold = parseInt(e.target.value, 10) / 100;
        render();
        updateSliderTrack(e.target);
        // Update keypoint threshold label
        var label = annotationList.querySelector(".keypoint-threshold-value");
        if (label) label.textContent = Math.round(state.keypointThreshold * 100) + '%';
        updateHeaderStats();
        // Update annotation summaries
        var summaryEls = annotationList.querySelectorAll(".ann-summary");
        for (var i = 0; i < summaryEls.length; i++) {
            var row = summaryEls[i].closest(".annotation-row");
            if (row) {
                var idx = parseInt(row.getAttribute("data-index"), 10);
                summaryEls[i].textContent = buildAnnotationSummary(state.annotations[idx]);
            }
        }
    }

    function handleKeypointRadiusChange(e) {
        state.keypointRadius = parseInt(e.target.value, 10);
        render();
        updateSliderTrack(e.target);
        var label = annotationList.querySelector(".keypoint-radius-value");
        if (label) label.textContent = state.keypointRadius;
    }

    function handleConnectionWidthChange(e) {
        state.connectionWidth = parseInt(e.target.value, 10);
        render();
        updateSliderTrack(e.target);
        var label = annotationList.querySelector(".connection-width-value");
        if (label) label.textContent = state.connectionWidth;
    }

    function handleMaskAlphaChange(e) {
        state.maskAlpha = parseInt(e.target.value, 10) / 100;
        render();
        updateSliderTrack(e.target);
        var label = annotationList.querySelector(".mask-alpha-value");
        if (label) label.textContent = Math.round(state.maskAlpha * 100) + '%';
    }

    function handleConnectionAlphaChange(e) {
        state.connectionAlpha = parseInt(e.target.value, 10) / 100;
        render();
        updateSliderTrack(e.target);
        var label = annotationList.querySelector(".connection-alpha-value");
        if (label) label.textContent = Math.round(state.connectionAlpha * 100) + '%';
    }

    function handleBboxLineWidthChange(e) {
        state.bboxLineWidth = parseInt(e.target.value, 10);
        render();
        updateSliderTrack(e.target);
        var label = annotationList.querySelector(".bbox-line-width-value");
        if (label) label.textContent = state.bboxLineWidth;
    }

    // Toggle base image
    toggleImageBtn.addEventListener("click", function () {
        state.showImage = !state.showImage;
        toggleImageBtn.classList.toggle("active", state.showImage);
        render();
    });

    // Reset to defaults
    resetBtn.addEventListener("click", resetToDefaults);

    // Maximize / minimize
    function toggleMaximize() {
        state.maximized = !state.maximized;
        element.classList.toggle("maximized", state.maximized);
        pinAncestorOpacity();

        if (state.maximized) {
            document.body.style.overflow = "hidden";
        } else {
            document.body.style.overflow = "";
        }

        if (state.image) {
            requestAnimationFrame(function () {
                fitCanvas();
                resetZoom();
            });
        }
    }

    maximizeBtn.addEventListener("click", toggleMaximize);

    // Help dialog
    function toggleHelp() {
        helpOverlay.classList.toggle("visible");
    }

    helpBtn.addEventListener("click", toggleHelp);
    helpCloseBtn.addEventListener("click", toggleHelp);
    helpOverlay.addEventListener("click", function (e) {
        if (e.target === helpOverlay) toggleHelp();
    });

    // ── Canvas Mouse Interaction (Pan + Selection) ────────────────

    canvas.addEventListener("mousedown", function (e) {
        if (e.button !== 0) return;
        if (!state.image) return;

        state.isPanning = true;
        state.didDrag = false;
        state.panStartX = e.clientX;
        state.panStartY = e.clientY;
        state.panStartPanX = state.panX;
        state.panStartPanY = state.panY;

        if (state.zoom > 1) {
            canvas.style.cursor = "grabbing";
        }
    });

    window.addEventListener("mousemove", function (e) {
        if (!state.image) return;

        if (state.isPanning) {
            var dx = e.clientX - state.panStartX;
            var dy = e.clientY - state.panStartY;

            if (!state.didDrag && (Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD)) {
                state.didDrag = true;
            }

            if (state.didDrag && state.zoom > 1) {
                var rect = canvas.getBoundingClientRect();
                var cssToCanvasX = canvas.width / rect.width;
                var cssToCanvasY = canvas.height / rect.height;
                state.panX = state.panStartPanX + dx * cssToCanvasX;
                state.panY = state.panStartPanY + dy * cssToCanvasY;
                clampPan();
                render();
            }
            return;
        }

        // Tooltip logic (only when not dragging)
        var rect = canvas.getBoundingClientRect();
        if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) return;
        if (state.annotations.length === 0) return;

        var pt = clientToCanvas(e.clientX, e.clientY);
        var cx = pt.x;
        var cy = pt.y;

        // Update cursor
        var hitIndex = findHitAnnotationIndex(cx, cy);
        if (state.zoom > 1) {
            canvas.style.cursor = hitIndex >= 0 ? "pointer" : "grab";
        } else {
            canvas.style.cursor = hitIndex >= 0 ? "pointer" : "default";
        }

        var nearest = findNearestKeypoint(cx, cy);
        var tooltipText = "";
        if (nearest) {
            tooltipText = nearest.kp.name;
            if (nearest.kp.confidence != null) {
                tooltipText += " (" + (nearest.kp.confidence * 100).toFixed(1) + "%)";
            }
        } else {
            var bboxHit = findBboxAt(cx, cy);
            if (bboxHit) {
                tooltipText = bboxHit.label || "";
                if (bboxHit.score != null) {
                    tooltipText += (tooltipText ? " " : "") + (bboxHit.score * 100).toFixed(1) + "%";
                }
            }
        }

        if (tooltipText) {
            tooltip.textContent = tooltipText;
            tooltip.classList.add("visible");

            var containerRect = container.getBoundingClientRect();
            var tooltipX = e.clientX - containerRect.left + 12;
            var tooltipY = e.clientY - containerRect.top - 8;
            tooltip.style.left = tooltipX + "px";
            tooltip.style.top = tooltipY + "px";
        } else {
            tooltip.classList.remove("visible");
        }
    });

    window.addEventListener("mouseup", function (e) {
        if (!state.isPanning) return;
        state.isPanning = false;

        if (state.zoom > 1) {
            canvas.style.cursor = "grab";
        }

        if (!state.didDrag && state.image && state.annotations.length > 0) {
            var pt = clientToCanvas(e.clientX, e.clientY);
            var hits = findAllHitAnnotationIndices(pt.x, pt.y);
            if (e.shiftKey && hits.length > 0) {
                // Shift+click: hide the topmost hit
                var target = hits[0];
                state.visibility[target] = false;
                if (state.selectedIndex === target) state.selectedIndex = -1;
            } else if (hits.length === 0) {
                state.selectedIndex = -1;
            } else if (hits.length === 1) {
                // Single hit: toggle as before
                state.selectedIndex = state.selectedIndex === hits[0] ? -1 : hits[0];
            } else {
                // Multiple overlapping hits: cycle through them
                var curPos = hits.indexOf(state.selectedIndex);
                if (curPos < 0) {
                    state.selectedIndex = hits[0];
                } else {
                    state.selectedIndex = hits[(curPos + 1) % hits.length];
                }
            }
            render();
            renderControlPanel();
        }
    });

    function findHitAnnotationIndex(cx, cy) {
        var hits = findAllHitAnnotationIndices(cx, cy);
        return hits.length > 0 ? hits[0] : -1;
    }

    function findAllHitAnnotationIndices(cx, cy) {
        var hitR = HIT_RADIUS / state.zoom;
        var hitR2 = hitR * hitR;
        var result = [];
        var keypointHitSet = {};
        // Check keypoints first (more precise)
        for (var i = 0; i < state.annotations.length; i++) {
            if (!isAnnotationVisible(i)) continue;
            var kps = state.annotations[i].keypoints || [];
            for (var j = 0; j < kps.length; j++) {
                var kp = kps[j];
                if (!isKeypointVisible(kp)) continue;
                var dx = cx - kp.x * state.scale;
                var dy = cy - kp.y * state.scale;
                if (dx * dx + dy * dy < hitR2) {
                    result.push(i);
                    keypointHitSet[i] = true;
                    break;
                }
            }
        }
        // Then check bboxes (topmost first)
        for (var i = state.annotations.length - 1; i >= 0; i--) {
            if (keypointHitSet[i]) continue;
            if (!isAnnotationVisible(i)) continue;
            var bbox = state.annotations[i].bbox;
            if (!bbox) continue;
            var bx = bbox.x * state.scale;
            var by = bbox.y * state.scale;
            if (cx >= bx && cx <= bx + bbox.width * state.scale && cy >= by && cy <= by + bbox.height * state.scale) {
                result.push(i);
            }
        }
        return result;
    }

    // ── Tooltip (mouseleave) ────────────────────────────────────

    canvas.addEventListener("mouseleave", function () {
        tooltip.classList.remove("visible");
        if (!state.isPanning) {
            canvas.style.cursor = state.zoom > 1 ? "grab" : "default";
        }
    });

    function findNearestKeypoint(cx, cy) {
        var best = null;
        var hitR = HIT_RADIUS / state.zoom;
        var bestDist = hitR * hitR;

        for (var i = 0; i < state.annotations.length; i++) {
            if (!isAnnotationVisible(i)) continue;
            var ann = state.annotations[i];
            var kps = ann.keypoints || [];
            for (var j = 0; j < kps.length; j++) {
                var kp = kps[j];
                if (!isKeypointVisible(kp)) continue;

                var kx = kp.x * state.scale;
                var ky = kp.y * state.scale;
                var dx = cx - kx;
                var dy = cy - ky;
                var dist = dx * dx + dy * dy;

                if (dist < bestDist) {
                    bestDist = dist;
                    best = { kp: kp, ann: ann };
                }
            }
        }
        return best;
    }

    function findBboxAt(cx, cy) {
        for (var i = state.annotations.length - 1; i >= 0; i--) {
            if (!isAnnotationVisible(i)) continue;
            var ann = state.annotations[i];
            var bbox = ann.bbox;
            if (!bbox) continue;

            var bx = bbox.x * state.scale;
            var by = bbox.y * state.scale;
            var bw = bbox.width * state.scale;
            var bh = bbox.height * state.scale;

            if (cx >= bx && cx <= bx + bw && cy >= by && cy <= by + bh) {
                return ann;
            }
        }
        return null;
    }

    // ── Wheel Zoom ─────────────────────────────────────────────────

    canvas.addEventListener("wheel", function (e) {
        if (!state.image) return;
        e.preventDefault();

        var delta = e.deltaY;
        if (e.deltaMode === 1) delta *= 16;
        else if (e.deltaMode === 2) delta *= 100;

        var newZoom = state.zoom * (1 - delta * ZOOM_SENSITIVITY);
        newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));

        // Zoom toward cursor position
        var rect = canvas.getBoundingClientRect();
        var mx = (e.clientX - rect.left) * (canvas.width / rect.width);
        var my = (e.clientY - rect.top) * (canvas.height / rect.height);

        state.panX = mx - (mx - state.panX) * (newZoom / state.zoom);
        state.panY = my - (my - state.panY) * (newZoom / state.zoom);
        state.zoom = newZoom;
        clampPan();
        render();
    }, { passive: false });

    // ── Double-click to Reset Zoom ──────────────────────────────

    canvas.addEventListener("dblclick", function (e) {
        if (!state.image) return;
        e.preventDefault();
        resetZoom();
        canvas.style.cursor = "default";
    });

    // ── Touch Support (Pinch-Zoom + Pan) ────────────────────────

    var touchState = { lastDist: 0, lastCenterX: 0, lastCenterY: 0, touchCount: 0 };

    function getTouchDistance(t1, t2) {
        var dx = t1.clientX - t2.clientX;
        var dy = t1.clientY - t2.clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function getTouchCenter(t1, t2) {
        return { x: (t1.clientX + t2.clientX) / 2, y: (t1.clientY + t2.clientY) / 2 };
    }

    canvas.addEventListener("touchstart", function (e) {
        if (!state.image) return;
        e.preventDefault();

        touchState.touchCount = e.touches.length;

        if (e.touches.length === 1) {
            state.isPanning = true;
            state.didDrag = false;
            state.panStartX = e.touches[0].clientX;
            state.panStartY = e.touches[0].clientY;
            state.panStartPanX = state.panX;
            state.panStartPanY = state.panY;
        } else if (e.touches.length === 2) {
            state.isPanning = false;
            touchState.lastDist = getTouchDistance(e.touches[0], e.touches[1]);
            var center = getTouchCenter(e.touches[0], e.touches[1]);
            touchState.lastCenterX = center.x;
            touchState.lastCenterY = center.y;
        }
    }, { passive: false });

    canvas.addEventListener("touchmove", function (e) {
        if (!state.image) return;
        e.preventDefault();

        if (e.touches.length === 1 && state.isPanning) {
            var dx = e.touches[0].clientX - state.panStartX;
            var dy = e.touches[0].clientY - state.panStartY;

            if (!state.didDrag && (Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD)) {
                state.didDrag = true;
            }

            if (state.didDrag && state.zoom > 1) {
                var rect = canvas.getBoundingClientRect();
                state.panX = state.panStartPanX + dx * (canvas.width / rect.width);
                state.panY = state.panStartPanY + dy * (canvas.height / rect.height);
                clampPan();
                render();
            }
        } else if (e.touches.length === 2) {
            var dist = getTouchDistance(e.touches[0], e.touches[1]);
            var center = getTouchCenter(e.touches[0], e.touches[1]);

            if (touchState.lastDist > 0) {
                var scale = dist / touchState.lastDist;
                var newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, state.zoom * scale));

                var rect = canvas.getBoundingClientRect();
                var mx = (center.x - rect.left) * (canvas.width / rect.width);
                var my = (center.y - rect.top) * (canvas.height / rect.height);

                state.panX = mx - (mx - state.panX) * (newZoom / state.zoom);
                state.panY = my - (my - state.panY) * (newZoom / state.zoom);

                // Simultaneous pan
                var panDx = (center.x - touchState.lastCenterX) * (canvas.width / rect.width);
                var panDy = (center.y - touchState.lastCenterY) * (canvas.height / rect.height);
                state.panX += panDx;
                state.panY += panDy;

                state.zoom = newZoom;
                clampPan();
                render();
            }

            touchState.lastDist = dist;
            touchState.lastCenterX = center.x;
            touchState.lastCenterY = center.y;
        }
    }, { passive: false });

    canvas.addEventListener("touchend", function (e) {
        if (!state.image) return;
        e.preventDefault();

        if (e.touches.length === 0) {
            if (!state.didDrag && touchState.touchCount === 1 && state.annotations.length > 0) {
                var touch = e.changedTouches[0];
                var pt = clientToCanvas(touch.clientX, touch.clientY);
                var hits = findAllHitAnnotationIndices(pt.x, pt.y);
                if (hits.length === 0) {
                    state.selectedIndex = -1;
                } else if (hits.length === 1) {
                    state.selectedIndex = state.selectedIndex === hits[0] ? -1 : hits[0];
                } else {
                    var curPos = hits.indexOf(state.selectedIndex);
                    if (curPos < 0) {
                        state.selectedIndex = hits[0];
                    } else {
                        state.selectedIndex = hits[(curPos + 1) % hits.length];
                    }
                }
                render();
                renderControlPanel();
            }
            state.isPanning = false;
            touchState.lastDist = 0;
            touchState.touchCount = 0;
        } else if (e.touches.length === 1) {
            // Transitioned from two fingers to one — restart single-finger pan
            state.isPanning = true;
            state.didDrag = false;
            state.panStartX = e.touches[0].clientX;
            state.panStartY = e.touches[0].clientY;
            state.panStartPanX = state.panX;
            state.panStartPanY = state.panY;
            touchState.lastDist = 0;
        }
    }, { passive: false });

    // ── Keyboard Shortcuts ────────────────────────────────────────

    element.setAttribute("tabindex", "0");
    element.style.outline = "none";

    element.addEventListener("keydown", function (e) {
        // Help dialog shortcuts (work even without image data)
        if (e.key === "?" && e.target.tagName !== "INPUT") {
            toggleHelp();
            e.preventDefault();
            return;
        }
        if (e.key === "Escape" && helpOverlay.classList.contains("visible")) {
            toggleHelp();
            e.preventDefault();
            return;
        }

        if (!state.image) return;

        if (e.key === "Escape") {
            if (state.maximized) {
                toggleMaximize();
                e.preventDefault();
            } else if (state.selectedIndex >= 0) {
                state.selectedIndex = -1;
                render();
                renderControlPanel();
                e.preventDefault();
            }
        } else if (e.key === "f" || e.key === "F") {
            if (e.target.tagName === "INPUT") return;
            toggleMaximize();
            e.preventDefault();
        } else if (e.key === "i" || e.key === "I") {
            if (e.target.tagName === "INPUT") return;
            state.showImage = !state.showImage;
            toggleImageBtn.classList.toggle("active", state.showImage);
            render();
            e.preventDefault();
        } else if (e.key === "a" || e.key === "A") {
            if (e.target.tagName === "INPUT") return;
            var anyVisible = false;
            for (var i = 0; i < state.visibility.length; i++) {
                if (state.visibility[i]) { anyVisible = true; break; }
            }
            var newVal = !anyVisible;
            for (var i = 0; i < state.visibility.length; i++) {
                state.visibility[i] = newVal;
            }
            render();
            renderControlPanel();
            e.preventDefault();
        } else if (e.key === "h" || e.key === "H") {
            if (e.target.tagName === "INPUT") return;
            if (state.selectedIndex >= 0) {
                var idx = state.selectedIndex;
                state.visibility[idx] = false;
                state.selectedIndex = -1;
                render();
                renderControlPanel();
            }
            e.preventDefault();
        } else if (e.key === "r" || e.key === "R") {
            if (e.target.tagName === "INPUT") return;
            resetToDefaults();
            e.preventDefault();
        } else if (e.key === "+" || e.key === "=") {
            if (e.target.tagName === "INPUT") return;
            zoomToCenter(state.zoom * 1.25);
            e.preventDefault();
        } else if (e.key === "-" || e.key === "_") {
            if (e.target.tagName === "INPUT") return;
            zoomToCenter(state.zoom / 1.25);
            e.preventDefault();
        } else if (e.key === "0") {
            if (e.target.tagName === "INPUT") return;
            resetZoom();
            canvas.style.cursor = "default";
            e.preventDefault();
        }
    });

    // ── Window Resize ─────────────────────────────────────────────

    var resizeTimer = null;
    window.addEventListener("resize", function () {
        if (resizeTimer) clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function () {
            if (state.image) {
                fitCanvas();
                if (!state.maximized) resetZoom();
                else render();
            }
        }, 150);
    });
    // annotationList dropdown handles its own height/overflow via CSS