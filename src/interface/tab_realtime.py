"""
Real-time webcam streaming & video frame detection tab UI and processing functions.

Enhanced Real-time Streaming & Tracking architecture:
  1. Live video preview stays strictly in native browser window (HTML5 <video>) for zero latency.
  2. Single combined display view (no duplicate/secondary image windows or split components).
  3. Continuous background VLM inference pipeline without stopping video or freezing UI.
  4. Real-time ByteTrack (Kalman Filter + Hungarian IoU matching) running locally on every frame tick:
     - Maintains smooth bounding box prediction & object tracking across frame ticks even while VLM inference is running.
     - Smoothly incorporates newly arrived VLM detection predictions when model output completes.
  5. Per-session detector state preventing multi-user cross-contamination.
"""

import io
import time
import html
import itertools
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Any, Optional, List, Tuple

import gradio as gr
from PIL import Image
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from openai import OpenAI
from free_detection.detection_pipeline import (
    ObjectDetectionPipeline,
    pil_to_data_uri,
    parse_detections,
    validate_detections,
)
from free_detection.image_preprocessing import (
    preprocess_resolution,
    map_bbox_to_original,
)
from free_detection.bytetrack import ByteTracker

DEFAULT_HUD = '<div class="neo-retro-hud-stat">STATUS: INITIALIZED</div>'

# ---------------------------------------------------------------------------
# Client cache (safe to share across sessions -- stateless/read-only)
# ---------------------------------------------------------------------------
_client_cache: Dict[Tuple[str, str, str], "ObjectDetectionPipeline"] = {}
_client_cache_lock = Lock()


def _get_pipeline(
    base_url: str, api_key: str, model_name: str
) -> ObjectDetectionPipeline:
    key = (base_url, api_key, model_name)
    with _client_cache_lock:
        pipeline = _client_cache.get(key)
        if pipeline is None:
            client = OpenAI(base_url=base_url, api_key=api_key)
            pipeline = ObjectDetectionPipeline(client=client, detector_model=model_name)
            _client_cache[key] = pipeline
        return pipeline


# ---------------------------------------------------------------------------
# Per-session detection & ByteTrack state
# ---------------------------------------------------------------------------
@dataclass
class SessionDetector:
    """
    One instance of this lives inside a gr.State for each connected browser
    session, so concurrent users never share detection results or the
    in-flight flag. Includes ByteTracker for frame-to-frame box tracking.
    """

    lock: Lock = field(default_factory=Lock)
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="det"
        )
    )
    future: Optional[Future] = None
    tracker: ByteTracker = field(default_factory=lambda: ByteTracker(high_thresh=0.4, low_thresh=0.1))
    last_raw_boxes: List[Any] = field(default_factory=list)
    last_tracked_boxes: List[Any] = field(default_factory=list)
    last_hud: str = DEFAULT_HUD
    last_applied_frame_id: int = -1
    _frame_counter: "itertools.count" = field(
        default_factory=lambda: itertools.count(1)
    )

    reference_gray: Optional[np.ndarray] = None
    last_detect_time: float = 0.0

    def next_frame_id(self) -> int:
        return next(self._frame_counter)

    def is_busy(self) -> bool:
        with self.lock:
            return self.future is not None and not self.future.done()

    def submit(self, frame_id, fn, *args) -> None:
        with self.lock:
            self.future = self.executor.submit(self._run_and_store, frame_id, fn, *args)

    def _run_and_store(self, frame_id, fn, *args):
        try:
            boxes, hud = fn(*args)
        except Exception as e:  # noqa: BLE001
            boxes, hud = None, (
                f'<div class="neo-retro-hud-stat" style="color:#ff0055 !important;">'
                f"ERROR: {html.escape(str(e))}</div>"
            )
        with self.lock:
            if frame_id >= self.last_applied_frame_id:
                self.last_applied_frame_id = frame_id
                if boxes is not None:
                    self.last_raw_boxes = boxes
                    # Feed new VLM predictions into ByteTracker
                    formatted_dets = []
                    for b in boxes:
                        # b: [ymin, xmin, ymax, xmax, label]
                        if len(b) >= 4:
                            lbl = b[4] if len(b) >= 5 else ""
                            formatted_dets.append([b[0], b[1], b[2], b[3], lbl, 0.9])
                    self.last_tracked_boxes = self.tracker.update(formatted_dets)
                self.last_hud = hud

    def update_tracking_only(self) -> List[Any]:
        """Runs ByteTracker prediction tick using existing tracks when no new VLM detection has completed."""
        with self.lock:
            if self.last_raw_boxes:
                formatted_dets = []
                for b in self.last_raw_boxes:
                    lbl = b[4] if len(b) >= 5 else ""
                    formatted_dets.append([b[0], b[1], b[2], b[3], lbl, 0.85])
                self.last_tracked_boxes = self.tracker.update(formatted_dets)
            return list(self.last_tracked_boxes)

    def snapshot(self):
        with self.lock:
            return list(self.last_tracked_boxes), self.last_hud

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


def new_session_detector() -> SessionDetector:
    return SessionDetector()


# ---------------------------------------------------------------------------
# Drawing & Preprocessing
# ---------------------------------------------------------------------------
def _to_small_gray(frame_rgb: np.ndarray, size=(160, 120)) -> Optional[np.ndarray]:
    """Downscaled, blurred grayscale version of a frame, cheap to diff."""
    if cv2 is None:
        return None
    small = cv2.resize(frame_rgb, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def _scene_has_changed(
    current_gray: np.ndarray,
    reference_gray: Optional[np.ndarray],
    pixel_diff_thresh: int = 25,
    change_ratio_thresh: float = 0.015,
) -> bool:
    if reference_gray is None:
        return True
    if cv2 is None:
        return True
    diff = cv2.absdiff(current_gray, reference_gray)
    _, thresholded = cv2.threshold(diff, pixel_diff_thresh, 255, cv2.THRESH_BINARY)
    changed_pixels = cv2.countNonZero(thresholded)
    total_pixels = thresholded.shape[0] * thresholded.shape[1]
    return (changed_pixels / total_pixels) > change_ratio_thresh


def draw_boxes_opencv(image_np: np.ndarray, boxes: List[Any]) -> np.ndarray:
    """Fast OpenCV bounding box rendering for video frame exports."""
    if cv2 is None or not boxes:
        return image_np
    img = image_np.copy()
    for box in boxes:
        if len(box) >= 4:
            ymin, xmin, ymax, xmax = box[:4]
            label = str(box[4]) if len(box) >= 5 else ""
            track_id = f" #{box[5]}" if len(box) >= 6 else ""
            full_label = f"{label}{track_id}"
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))
            cv2.rectangle(img, pt1, pt2, (204, 255, 0), 2)
            if full_label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(
                    full_label, font, font_scale, thickness
                )
                lbl_pt1 = (int(xmin), max(0, int(ymin) - text_h - 6))
                lbl_pt2 = (int(xmin) + text_w + 6, max(text_h + 6, int(ymin)))
                cv2.rectangle(img, lbl_pt1, lbl_pt2, (204, 255, 0), -1)
                cv2.putText(
                    img,
                    full_label,
                    (int(xmin) + 3, max(text_h + 2, int(ymin) - 3)),
                    font,
                    font_scale,
                    (17, 8, 5),
                    thickness,
                    cv2.LINE_AA,
                )
    return img


# ---------------------------------------------------------------------------
# Core detection call
# ---------------------------------------------------------------------------
def _detect(
    frame: np.ndarray,
    categories: list,
    base_url: str,
    api_key: str,
    model_name: str,
    prep_info: dict,
) -> Tuple[List[Any], str]:
    """Runs VLM detection on a single frame and returns (boxes_in_original_pixel_space, hud_html)."""
    start_time = time.time()
    pil_img = Image.fromarray(frame).convert("RGB")
    pipeline = _get_pipeline(base_url, api_key, model_name)
    img_uri = pil_to_data_uri(pil_img)
    raw_output = pipeline.run_inference(
        image_uris=img_uri,
        categories=categories,
        category_definitions="",
    )
    parsed_dets = parse_detections(raw_output)
    valid_dets = validate_detections(parsed_dets, categories)

    orig_w = prep_info["orig_w"]
    orig_h = prep_info["orig_h"]
    boxes = []
    for d in valid_dets:
        bbox = d.get("bbox_2d", [])
        lbl = d.get("label", "")
        if len(bbox) == 4:
            x1, y1, x2, y2 = map_bbox_to_original(list(bbox), prep_info)
            ymin = y1 * orig_h / 1000.0
            xmin = x1 * orig_w / 1000.0
            ymax = y2 * orig_h / 1000.0
            xmax = x2 * orig_w / 1000.0
            boxes.append([ymin, xmin, ymax, xmax, lbl])

    elapsed = (time.time() - start_time) * 1000.0
    fps = 1000.0 / max(elapsed, 1.0)
    hud = (
        f'<div class="neo-retro-hud-stat">FPS: {fps:.1f} | '
        f"LATENCY: {elapsed:.0f}ms | DETECTED: {len(boxes)}</div>"
    )
    return boxes, hud


def _resolve_endpoint(
    server_port, use_external_api, ext_api_url, ext_api_key, ext_model_name
):
    base_url = ext_api_url if use_external_api else f"http://127.0.0.1:{server_port}/v1"
    api_key = ext_api_key if use_external_api else "no-key"
    model_name = ext_model_name if use_external_api else "local-model"
    return base_url, api_key, model_name


# ---------------------------------------------------------------------------
# Webcam streaming tick
# ---------------------------------------------------------------------------
def process_single_frame(
    frame: np.ndarray,
    categories_str: str,
    server_port: int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    confidence_thresh: float,
    max_resolution: int,
    motion_sensitivity_pct: float,
    stale_refresh_seconds: float,
    session: SessionDetector,
) -> tuple[dict, str, SessionDetector]:
    """
    Continuous Live Streaming Processor:
    - Real-time video preview is shown seamlessly without freezing or stopping.
    - Sends frames to background thread pool for VLM detection.
    - In between VLM responses, runs ByteTrack locally to update and smooth bounding boxes across ticks.
    - Client-side JS draws the overlay dynamically on top of the live video preview.
    """
    if session is None:
        session = new_session_detector()

    if frame is None:
        boxes, hud = session.snapshot()
        return {"boxes": boxes, "frame_w": 0, "frame_h": 0}, hud, session

    pil_img = Image.fromarray(frame).convert("RGB")
    frame_h, frame_w = frame.shape[0], frame.shape[1]

    # 1) Execute ByteTrack update for continuous tracking across stream ticks
    tracked_boxes = session.update_tracking_only()
    hud = session.last_hud

    # 2) Dispatch background detection when ready
    if not session.is_busy():
        gray_small = _to_small_gray(frame)
        change_ratio_thresh = max(0.0, float(motion_sensitivity_pct or 1.5)) / 100.0
        now = time.time()
        stale = (now - session.last_detect_time) >= max(
            0.5, float(stale_refresh_seconds or 6.0)
        )
        changed = (
            _scene_has_changed(
                gray_small,
                session.reference_gray,
                change_ratio_thresh=change_ratio_thresh,
            )
            if gray_small is not None
            else True
        )

        if changed or stale:
            max_res = int(max_resolution or 640)
            categories = [
                c.strip() for c in categories_str.split(",") if c.strip()
            ] or ["object"]
            base_url, api_key, model_name = _resolve_endpoint(
                server_port, use_external_api, ext_api_url, ext_api_key, ext_model_name
            )
            proc_img, prep_info = preprocess_resolution(
                pil_img, enabled=True, target_short_edge=max_res
            )
            frame_id = session.next_frame_id()
            session.submit(
                frame_id,
                _detect,
                np.array(proc_img),
                categories,
                base_url,
                api_key,
                model_name,
                prep_info,
            )
            session.reference_gray = gray_small
            session.last_detect_time = now

    return (
        {"boxes": tracked_boxes, "frame_w": frame_w, "frame_h": frame_h},
        hud,
        session,
    )


def reset_session(session: Optional[SessionDetector]) -> SessionDetector:
    if session is not None:
        session.shutdown()
    return new_session_detector()


# ---------------------------------------------------------------------------
# Video file processing
# ---------------------------------------------------------------------------
def process_video_frames(
    video_path: str,
    sample_interval: float,
    categories_str: str,
    server_port: int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    max_resolution: int = 640,
    progress=gr.Progress(),
) -> tuple[List[np.ndarray], str]:
    if not video_path:
        return [], "No video file uploaded."
    if cv2 is None:
        return [], "OpenCV (cv2) is required for video processing."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], "Failed to open video file."

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(max(1, fps * sample_interval))
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
    cap.release()

    if not frames:
        return [], "No frames could be sampled from this video."

    categories = [c.strip() for c in categories_str.split(",") if c.strip()] or [
        "object"
    ]
    base_url, api_key, model_name = _resolve_endpoint(
        server_port, use_external_api, ext_api_url, ext_api_key, ext_model_name
    )
    max_res = int(max_resolution or 640)

    tracker = ByteTracker(high_thresh=0.4, low_thresh=0.1)
    annotated_frames = []
    errors = 0
    for idx, f in enumerate(frames):
        progress(
            (idx + 1) / len(frames), desc=f"Detecting frame {idx + 1}/{len(frames)}"
        )
        pil_img = Image.fromarray(f).convert("RGB")
        proc_img, prep_info = preprocess_resolution(
            pil_img, enabled=True, target_short_edge=max_res
        )
        try:
            boxes, _hud = _detect(
                np.array(proc_img), categories, base_url, api_key, model_name, prep_info
            )
            formatted_dets = []
            for b in boxes:
                lbl = b[4] if len(b) >= 5 else ""
                formatted_dets.append([b[0], b[1], b[2], b[3], lbl, 0.9])
            tracked_boxes = tracker.update(formatted_dets)
        except Exception:
            tracked_boxes = []
            errors += 1
        annotated_frames.append(draw_boxes_opencv(np.array(pil_img), tracked_boxes))

    status = f"Successfully processed {len(annotated_frames)} frames from video!"
    if errors:
        status += f" ({errors} frame(s) failed detection and were shown unannotated.)"
    return annotated_frames, status


# ---------------------------------------------------------------------------
# UI Construction
# ---------------------------------------------------------------------------
def _build_realtime_tab() -> Dict[str, Any]:
    c = {}
    c["session_state"] = gr.State(new_session_detector)

    with gr.Column(elem_classes=["neo-retro-card"]):
        gr.HTML(
            """
        <div style="padding: 10px; border-bottom: 2px solid #00ffcc; background: #050811;">
            <span class="neo-retro-badge">LIVE CYBER-STREAM</span>
            <h2 style="color: #00ffcc; font-family: 'JetBrains Mono', monospace; margin: 5px 0 0;">
                ⚡ REAL-TIME WEBCAM & VIDEO FRAME DETECTOR (BYTETRACK INTEGRATED)
            </h2>
        </div>
        """
        )
        with gr.Row():
            with gr.Column(scale=1):
                c["stream_mode"] = gr.Radio(
                    choices=["Webcam Stream", "Video Upload (1s Sampling)"],
                    value="Webcam Stream",
                    label="STREAM INPUT SOURCE",
                )
                c["categories_input"] = gr.Textbox(
                    value="person, car, dog, bottle, phone",
                    label="TARGET CATEGORIES (comma-separated)",
                )
                c["max_resolution"] = gr.Slider(
                    minimum=384,
                    maximum=1084,
                    step=128,
                    value=640,
                    label="MAX FRAME RESOLUTION (PX)",
                    info="Lower resolution = faster real-time processing and lower latency.",
                )
                c["motion_sensitivity"] = gr.Slider(
                    minimum=0.5,
                    maximum=10.0,
                    step=0.5,
                    value=1.5,
                    label="MOTION SENSITIVITY (% PIXELS CHANGED)",
                    info="Webcam mode: a new detection only runs once at least this % of the "
                    "frame changes (absdiff+threshold). Lower = more sensitive/more model calls.",
                )
                c["stale_refresh"] = gr.Slider(
                    minimum=2.0,
                    maximum=20.0,
                    step=1.0,
                    value=6.0,
                    label="STALE REFRESH FALLBACK (SECONDS)",
                    info="Webcam mode: re-detect anyway after this long, even with no motion.",
                )
                c["sample_interval"] = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    step=0.5,
                    value=1.0,
                    label="VIDEO FRAME SAMPLING INTERVAL (SECONDS)",
                )
                c["process_video_btn"] = gr.Button(
                    "⚡ PROCESS VIDEO FRAMES",
                    variant="primary",
                    elem_classes=["neo-retro-badge"],
                )
                c["hud_status"] = gr.HTML(value=DEFAULT_HUD)
            with gr.Column(scale=2):
                # Inject the overlay canvas via JS directly onto document.body so it
                # floats above the video element without being constrained by Gradio's
                # nested wrapper divs (which break position:absolute stacking).
                gr.HTML(
                    """
                    <style>
                    #rt_float_canvas {
                        position: fixed;
                        pointer-events: none;
                        z-index: 9999;
                        box-sizing: border-box;
                    }
                    </style>
                    <script>
                    (function() {
                        if (document.getElementById('rt_float_canvas')) return;
                        var cv = document.createElement('canvas');
                        cv.id = 'rt_float_canvas';
                        document.body.appendChild(cv);

                        function syncPosition() {
                            // Find the live <video> inside the webcam component
                            var anchor = document.getElementById('rt_webcam_input');
                            if (!anchor) return;
                            var video = anchor.querySelector('video');
                            var target = video || anchor;
                            var r = target.getBoundingClientRect();
                            cv.style.left   = r.left + 'px';
                            cv.style.top    = r.top  + 'px';
                            cv.style.width  = r.width  + 'px';
                            cv.style.height = r.height + 'px';
                            if (cv.width  !== Math.round(r.width))  cv.width  = Math.round(r.width);
                            if (cv.height !== Math.round(r.height)) cv.height = Math.round(r.height);
                        }

                        // Reposition on every animation frame so scrolling/resize is instant
                        function loop() { syncPosition(); requestAnimationFrame(loop); }
                        requestAnimationFrame(loop);
                    })();
                    </script>
                    """
                )
                with gr.Group(elem_id="rt_webcam_wrap") as webcam_wrap:
                    c["webcam_input"] = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="LIVE WEBCAM STREAM",
                        type="numpy",
                        elem_id="rt_webcam_input",
                    )
                c["webcam_wrap_group"] = webcam_wrap
                c["boxes_json_state"] = gr.JSON(visible=False)
                c["video_input"] = gr.Video(
                    label="INPUT VIDEO FILE",
                    visible=False,
                )
                c["video_gallery_output"] = gr.Gallery(
                    label="SAMPLED FRAME DETECTIONS (EVERY 1 SEC)",
                    visible=False,
                    columns=3,
                )
    return c


def _wire_realtime_events(
    c_real: Dict[str, Any], c_srv: Dict[str, Any], c_bat: Dict[str, Any]
):
    def toggle_mode(mode, session):
        is_cam = mode == "Webcam Stream"
        fresh_session = reset_session(session)
        return (
            gr.update(visible=is_cam),
            gr.update(visible=not is_cam),
            gr.update(visible=not is_cam),
            fresh_session,
        )

    c_real["stream_mode"].change(
        toggle_mode,
        inputs=[c_real["stream_mode"], c_real["session_state"]],
        outputs=[
            c_real["webcam_wrap_group"],
            c_real["video_input"],
            c_real["video_gallery_output"],
            c_real["session_state"],
        ],
    )

    c_real["webcam_input"].stream(
        fn=process_single_frame,
        inputs=[
            c_real["webcam_input"],
            c_real["categories_input"],
            c_srv["server_port_input"],
            c_bat["use_external_api_chk"],
            c_bat["ext_api_url"],
            c_bat["ext_api_key"],
            c_bat["ext_model_name"],
            gr.State(0.3),
            c_real["max_resolution"],
            c_real["motion_sensitivity"],
            c_real["stale_refresh"],
            c_real["session_state"],
        ],
        outputs=[
            c_real["boxes_json_state"],
            c_real["hud_status"],
            c_real["session_state"],
        ],
        stream_every=0.1,
    )

    c_real["boxes_json_state"].change(
        fn=None,
        inputs=[c_real["boxes_json_state"]],
        outputs=[],
        js="""
        (payload) => {
            // Use the body-level floating canvas that is always positioned over
            // the live <video> element via the rAF loop set up at build time.
            const canvas = document.getElementById('rt_float_canvas');
            if (!canvas || !payload) return;

            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const boxes = payload.boxes || [];
            const frameW = payload.frame_w || canvas.width;
            const frameH = payload.frame_h || canvas.height;
            if (!frameW || !frameH || boxes.length === 0) return;

            const scaleX = canvas.width  / frameW;
            const scaleY = canvas.height / frameH;

            ctx.lineWidth = 2;
            ctx.font = '12px "JetBrains Mono", monospace';

            for (const box of boxes) {
                if (!box || box.length < 4) continue;
                const [ymin, xmin, ymax, xmax, label, trackId] = box;
                const x = xmin * scaleX, y = ymin * scaleY;
                const w = (xmax - xmin) * scaleX, h = (ymax - ymin) * scaleY;

                // Draw box outline with neon glow effect
                ctx.strokeStyle = '#00ffcc';
                ctx.shadowColor  = '#00ffcc';
                ctx.shadowBlur   = 6;
                ctx.strokeRect(x, y, w, h);
                ctx.shadowBlur   = 0;

                const tag = (trackId !== undefined && trackId !== null)
                    ? `${label || ''} #${trackId}`
                    : `${label || ''}`;
                if (tag.trim()) {
                    const textW = ctx.measureText(tag).width;
                    const bh = 18;
                    const by = y > bh ? y - bh : y + h;
                    ctx.fillStyle = 'rgba(0,255,204,0.85)';
                    ctx.fillRect(x, by, textW + 8, bh);
                    ctx.fillStyle = '#050811';
                    ctx.fillText(tag, x + 4, by + bh - 4);
                }
            }
        }
        """,
    )

    c_real["process_video_btn"].click(
        fn=process_video_frames,
        inputs=[
            c_real["video_input"],
            c_real["sample_interval"],
            c_real["categories_input"],
            c_srv["server_port_input"],
            c_bat["use_external_api_chk"],
            c_bat["ext_api_url"],
            c_bat["ext_api_key"],
            c_bat["ext_model_name"],
            c_real["max_resolution"],
        ],
        outputs=[c_real["video_gallery_output"], c_real["hud_status"]],
    )
