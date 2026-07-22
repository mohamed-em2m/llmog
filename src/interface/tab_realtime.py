"""
Real-time webcam streaming & video frame detection tab UI and processing functions.

Fixes vs. previous version:
  1. Per-session state (was: module-level globals shared across every connected user).
     Each browser session now gets its own SessionDetector via gr.State, so two people
     streaming at once no longer stomp on each other's boxes / detecting flag.
  2. Detections are frame-correlated with a monotonic frame_id, so a slow detection
     that finishes late can never overwrite a newer result ("flicker back" bug).
  3. Video-file processing now waits for each frame's detection to actually complete
     (bounded via Future.result()) instead of firing a background thread and grabbing
     whatever _last_boxes happens to contain. This was the "results after model ended"
     bug: the loop would move to the next frame before the previous detection thread
     had written its result, and the very last frame(s)' detections were dropped
     entirely because the thread finished after process_video_frames had returned.
  4. OpenAI client + ObjectDetectionPipeline are cached per (base_url, api_key, model)
     instead of being rebuilt on every single tick.
  5. ThreadPoolExecutor(max_workers=1) + Future replaces the raw Thread + bool flag,
     which was a race: `_is_detecting` could be read stale between the check and set.
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

DEFAULT_HUD = '<div class="neo-retro-hud-stat">STATUS: INITIALIZED</div>'

# ---------------------------------------------------------------------------
# Client cache (safe to share across sessions -- it's stateless/read-only,
# unlike the detection results which must stay per-session)
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
# Per-session detection state
# ---------------------------------------------------------------------------
@dataclass
class SessionDetector:
    """
    One instance of this lives inside a gr.State for each connected browser
    session, so concurrent users never share detection results or the
    in-flight flag.
    """

    lock: Lock = field(default_factory=Lock)
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="det"
        )
    )
    future: Optional[Future] = None
    last_boxes: List[Any] = field(default_factory=list)
    last_hud: str = DEFAULT_HUD
    last_applied_frame_id: int = -1
    _frame_counter: "itertools.count" = field(
        default_factory=lambda: itertools.count(1)
    )

    # Motion-gating state: the small grayscale frame that was last SENT to
    # detection, and when that happened. New frames are diffed against this
    # to decide whether the scene has actually changed enough to bother
    # calling the model again -- replaces the old "just poll every N sec".
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
            # Only apply if this frame is newer than whatever we last applied.
            # Prevents a slow, stale detection from clobbering a fresher result.
            if frame_id >= self.last_applied_frame_id:
                self.last_applied_frame_id = frame_id
                if boxes is not None:
                    self.last_boxes = boxes
                self.last_hud = hud

    def snapshot(self):
        with self.lock:
            return list(self.last_boxes), self.last_hud

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


def new_session_detector() -> SessionDetector:
    return SessionDetector()


# ---------------------------------------------------------------------------
# Drawing
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
    """
    absdiff + threshold motion detection: compares `current_gray` against
    the last frame that was actually sent to the model. Returns True if
    enough pixels changed to be worth re-running detection.

    - No reference yet (first frame) -> always True.
    - `pixel_diff_thresh`: how much a pixel's brightness must change (0-255)
      to count as "different" -- filters out sensor noise.
    - `change_ratio_thresh`: fraction of pixels that must be "different"
      for the frame as a whole to count as changed (e.g. 0.015 = 1.5%).
    """
    if reference_gray is None:
        return True
    if cv2 is None:
        return True  # can't diff without cv2 -- fail open, always detect
    diff = cv2.absdiff(current_gray, reference_gray)
    _, thresholded = cv2.threshold(diff, pixel_diff_thresh, 255, cv2.THRESH_BINARY)
    changed_pixels = cv2.countNonZero(thresholded)
    total_pixels = thresholded.shape[0] * thresholded.shape[1]
    return (changed_pixels / total_pixels) > change_ratio_thresh


def draw_boxes_opencv(image_np: np.ndarray, boxes: List[Any]) -> np.ndarray:
    """Fast OpenCV bounding box rendering (<1ms vs ~200ms for Matplotlib)."""
    if cv2 is None or not boxes:
        return image_np
    img = image_np.copy()
    for box in boxes:
        if len(box) >= 4:
            ymin, xmin, ymax, xmax = box[:4]
            label = str(box[4]) if len(box) >= 5 else ""
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))
            # Vibrant cyber cyan bounding box (#00ffcc -> BGR (204, 255, 0))
            cv2.rectangle(img, pt1, pt2, (204, 255, 0), 2)
            if label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                lbl_pt1 = (int(xmin), max(0, int(ymin) - text_h - 6))
                lbl_pt2 = (int(xmin) + text_w + 6, max(text_h + 6, int(ymin)))
                cv2.rectangle(img, lbl_pt1, lbl_pt2, (204, 255, 0), -1)
                cv2.putText(
                    img,
                    label,
                    (int(xmin) + 3, max(text_h + 2, int(ymin) - 3)),
                    font,
                    font_scale,
                    (17, 8, 5),
                    thickness,
                    cv2.LINE_AA,
                )
    return img


# ---------------------------------------------------------------------------
# Core detection call (pure -- no globals, no gradio-specific side effects)
# ---------------------------------------------------------------------------
def _detect(
    frame: np.ndarray,
    categories: list,
    base_url: str,
    api_key: str,
    model_name: str,
    prep_info: dict,
) -> Tuple[List[Any], str]:
    """Runs VLM detection on a single (already-downscaled) frame and returns
    (boxes_in_original_pixel_space, hud_html). Raises on failure -- caller
    is responsible for catching."""
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
# Webcam streaming tick (async / non-blocking -- must never stall the UI)
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
    Called on every Gradio webcam stream tick.

    - This does NOT render or return an image. The webcam feed the user
      sees is the browser's own native `<video>` preview -- it is already
      real-time and never touches this function. What this returns is a
      small JSON payload (`{"boxes": [...], "frame_w": W, "frame_h": H}`)
      that a client-side JS snippet (wired in _wire_realtime_events) uses
      to redraw a transparent <canvas> overlaid on top of that live video.
      No image round-trips through the server for display -- only box
      coordinates, which is what makes the overlay feel real-time instead
      of laggy.
    - In the background, at most one detection runs at a time. Instead of
      deciding "should I detect?" from a fixed timer, we diff the current
      frame against the last frame we actually sent to the model
      (absdiff + threshold on a small grayscale copy). If the scene hasn't
      meaningfully changed, we skip the model call entirely and just keep
      reusing the boxes we already have -- they stay on screen until a real
      change triggers a fresh detection that replaces them.
    - `stale_refresh_seconds` is a small safety net: even with zero motion,
      we still re-check every so often (default a few seconds) so a missed
      detection (e.g. object present before the stream even started) isn't
      stuck forever. Set it high (or math.inf) to disable and rely on
      motion alone.
    """
    if session is None:
        session = new_session_detector()

    if frame is None:
        boxes, hud = session.snapshot()
        return {"boxes": boxes, "frame_w": 0, "frame_h": 0}, hud, session

    pil_img = Image.fromarray(frame).convert("RGB")
    frame_h, frame_w = frame.shape[0], frame.shape[1]

    # 1) Grab whatever boxes are already cached -- NO image is rendered here.
    #    The browser already has a live, native webcam feed on screen; we
    #    just hand it the latest box coordinates as JSON and a small bit of
    #    client-side JS (wired in _wire_realtime_events) draws them onto a
    #    transparent <canvas> sitting on top of that live video. That keeps
    #    the video itself genuinely real-time (it never goes through this
    #    Python round trip at all) while the overlay updates as boxes arrive.
    boxes_to_draw, hud = session.snapshot()

    # 2) Decide, in the background, whether this frame is worth detecting.
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
        {"boxes": boxes_to_draw, "frame_w": frame_w, "frame_h": frame_h},
        hud,
        session,
    )


def reset_session(session: Optional[SessionDetector]) -> SessionDetector:
    """Call when the stream stops / mode toggles, so a stale in-flight
    detection from the old context can't leak into the new one."""
    if session is not None:
        session.shutdown()
    return new_session_detector()


# ---------------------------------------------------------------------------
# Video file processing (SYNCHRONOUS per sampled frame -- deliberately not
# fire-and-forget, so every sampled frame's result is guaranteed to be
# captured before the function returns, and progress reporting stays honest)
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
            # Blocking call -- we WAIT for this frame's result before moving on,
            # so nothing is ever dropped or shifted relative to the frame it came from.
            boxes, _hud = _detect(
                np.array(proc_img), categories, base_url, api_key, model_name, prep_info
            )
        except Exception:
            boxes = []
            errors += 1
        annotated_frames.append(draw_boxes_opencv(np.array(pil_img), boxes))

    status = f"Successfully processed {len(annotated_frames)} frames from video!"
    if errors:
        status += f" ({errors} frame(s) failed detection and were shown unannotated.)"
    return annotated_frames, status


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _build_realtime_tab() -> Dict[str, Any]:
    c = {}
    # Per-session detection state -- NOT shared between browser tabs/users.
    c["session_state"] = gr.State(new_session_detector)

    with gr.Column(elem_classes=["neo-retro-card"]):
        gr.HTML(
            """
        <div style="padding: 10px; border-bottom: 2px solid #00ffcc; background: #050811;">
            <span class="neo-retro-badge">LIVE CYBER-STREAM</span>
            <h2 style="color: #00ffcc; font-family: 'JetBrains Mono', monospace; margin: 5px 0 0;">
                ⚡ REAL-TIME WEBCAM & VIDEO FRAME DETECTOR
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
                # elem_id lets the JS below find this component's DOM
                # wrapper to size/position the overlay canvas on top of it.
                gr.HTML(
                    """
                    <style>
                    #rt_webcam_wrap { position: relative; }
                    #rt_overlay_canvas {
                        position: absolute;
                        top: 0; left: 0;
                        width: 100%; height: 100%;
                        pointer-events: none;
                        z-index: 5;
                    }
                    </style>
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
                    c["overlay_canvas_html"] = gr.HTML(
                        '<canvas id="rt_overlay_canvas"></canvas>'
                    )
                c["webcam_wrap_group"] = webcam_wrap
                # Hidden data channel: box coordinates + frame size only --
                # never an image. The JS bound in _wire_realtime_events
                # draws this onto #rt_overlay_canvas whenever it changes.
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
        # Tear down any in-flight detection from the mode we're leaving so it
        # can't write stale results into the mode we're entering.
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
        stream_every=0.3,
    )

    # Pure client-side redraw: fires whenever boxes_json_state's value
    # changes (i.e. every stream tick), with NO extra server round trip --
    # the data already arrived as part of the stream() response above. This
    # is what makes the overlay track the (always-live) video in real time
    # instead of waiting on a second image to render server-side.
    c_real["boxes_json_state"].change(
        fn=None,
        inputs=[c_real["boxes_json_state"]],
        outputs=[],
        js="""
        (payload) => {
            const wrap = document.getElementById('rt_webcam_wrap');
            const canvas = document.getElementById('rt_overlay_canvas');
            if (!wrap || !canvas || !payload) return;

            // Keep the canvas's pixel buffer matched to its on-screen size
            // (CSS already stretches it to cover the wrapper via 100%/100%).
            const rect = wrap.getBoundingClientRect();
            if (canvas.width !== rect.width || canvas.height !== rect.height) {
                canvas.width = rect.width;
                canvas.height = rect.height;
            }
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const boxes = payload.boxes || [];
            const frameW = payload.frame_w || canvas.width;
            const frameH = payload.frame_h || canvas.height;
            if (!frameW || !frameH || boxes.length === 0) return;

            const scaleX = canvas.width / frameW;
            const scaleY = canvas.height / frameH;

            ctx.strokeStyle = '#00ffcc';
            ctx.lineWidth = 2;
            ctx.font = '12px "JetBrains Mono", monospace';

            for (const box of boxes) {
                if (!box || box.length < 4) continue;
                const [ymin, xmin, ymax, xmax, label] = box;
                const x = xmin * scaleX, y = ymin * scaleY;
                const w = (xmax - xmin) * scaleX, h = (ymax - ymin) * scaleY;
                ctx.strokeRect(x, y, w, h);
                if (label) {
                    const text = String(label);
                    const textW = ctx.measureText(text).width;
                    ctx.fillStyle = '#00ffcc';
                    ctx.fillRect(x, Math.max(0, y - 16), textW + 6, 16);
                    ctx.fillStyle = '#110805';
                    ctx.fillText(text, x + 3, Math.max(11, y - 4));
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
